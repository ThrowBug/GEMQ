"""
Diagnostic for the DeepSeek decode path producing nan non-deterministically.

Observation that motivated this file: test_decode_path_matches_reference[deepseek]
passed on one run and returned nan on the next, with no change to the numerics in
between. Mixtral, whose forward_one_token does not touch the shared-expert kernels,
stayed clean at ~9e-4 on both runs.

That pattern is the signature of a kernel that does not write every element of its
output buffer. fused_dequant_up_proj_triton allocates x1/x3 with torch.empty, so any
element the kernel skips keeps whatever the caching allocator left there -- benign
numbers on one run, nan on the next.

These tests deliberately fill freed memory with nan first, so an unwritten element
shows up as nan every time instead of once in a while, then walk the decode path
stage by stage to find which kernel is responsible.

Run with:  pytest tests/test_decode_nan_hunt.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F

from test_moe_block_equiv import (
    build_deepseek_blocks, build_mixtral_blocks, build_olmoe_blocks,
    build_qwen3moe_blocks, HIDDEN_SIZE,
)


REPEATS = 5


def poison_allocator(device, nbytes=256 * 1024 * 1024):
    """
    Fill a chunk of the caching allocator with nan and release it. Subsequent
    torch.empty calls of similar size are likely to be handed this memory back, so
    anything a kernel fails to write reads as nan rather than plausible garbage.
    """
    junk = torch.full((nbytes // 2,), float("nan"), dtype=torch.float16, device=device)
    del junk


def nan_report(name, tensor):
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    total = tensor.numel()
    status = "CLEAN" if (n_nan == 0 and n_inf == 0) else "DIRTY"
    print(f"    {name:<28} {status}  nan={n_nan:>6}/{total:<7} inf={n_inf}")
    return n_nan + n_inf == 0


@pytest.mark.cuda
@pytest.mark.parametrize(
    "builder",
    [build_deepseek_blocks, build_mixtral_blocks, build_olmoe_blocks, build_qwen3moe_blocks],
    ids=["deepseek", "mixtral", "olmoe", "qwen3moe"],
)
def test_decode_output_is_deterministic(device, builder):
    """
    Same block, same input, repeated. A kernel that reads uninitialized memory gives
    different answers across repeats; a correct one is bit-identical.
    """
    quant_block, _, _ = builder(device)

    torch.manual_seed(3)
    x = (torch.randn(1, 1, HIDDEN_SIZE, device=device) * 0.5).half()

    from conftest import relative_error

    outs = []
    print()
    for i in range(REPEATS):
        poison_allocator(device)
        with torch.no_grad():
            out = quant_block(x)
            out = out[0] if isinstance(out, (tuple, list)) else out
        outs.append(out.float().clone())
        n_nan = torch.isnan(outs[-1]).sum().item()
        print(f"    repeat {i}: nan={n_nan:>5}/{outs[-1].numel()}  sum={outs[-1].nansum().item():.6f}")

    identical = all(torch.equal(outs[0], o) for o in outs[1:])
    spread = max(relative_error(o, outs[0]) for o in outs[1:])
    print(f"    bit-identical={identical}  max spread={spread:.3e}")

    assert not torch.isnan(outs[0]).any(), "forward_one_token produced nan"
    # DeepSeek is expected to be non-bit-identical: its shared experts go through the
    # split-K GEMV, whose atomic accumulation order varies. Mixtral has no shared
    # experts and must be exactly reproducible.
    if builder is not build_deepseek_blocks:
        assert identical, (
            f"{builder.__name__} decode should be bit-reproducible; only DeepSeek's shared "
            f"experts touch the split-K kernel"
        )
    assert spread <= 1e-2, (
        f"decode output varies by {spread:.3e} between identical calls -- larger than "
        f"accumulation-order rounding explains"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_deepseek_block_has_no_nan_across_inputs(device, seed):
    """
    test_decode_path_matches_reference[deepseek] once reported nan at seed 2 and a clean
    result on a later run. Sweep inputs and check the fused block and its dequantized
    reference separately, so a future nan is attributed to one side rather than guessed at.
    """
    quant_block, ref_block, _ = build_deepseek_blocks(device)

    torch.manual_seed(seed)
    x = (torch.randn(1, 1, HIDDEN_SIZE, device=device) * 0.5).half()

    with torch.no_grad():
        y_quant = quant_block(x)
        y_ref = ref_block(x)
    y_quant = y_quant[0] if isinstance(y_quant, (tuple, list)) else y_quant
    y_ref = y_ref[0] if isinstance(y_ref, (tuple, list)) else y_ref

    nq = torch.isnan(y_quant).sum().item()
    nr = torch.isnan(y_ref).sum().item()
    print(f"\n    seed={seed}  fused nan={nq}  reference nan={nr}  "
          f"|quant|max={y_quant.abs().max().item():.4g}  |ref|max={y_ref.abs().max().item():.4g}")

    assert nq == 0, f"fused block produced {nq} nan at seed {seed}"
    assert nr == 0, f"dequantized reference block produced {nr} nan at seed {seed}"


@pytest.mark.cuda
def test_deepseek_decode_stage_by_stage(device):
    """
    Walk DeepSeek's forward_one_token manually and report which stage first goes
    dirty. Mirrors moe_block.QuantFusedDeepseekV2MoEBlock.forward_one_token exactly.
    """
    from gemq.triton_kernels.dequant_gemv import dequant_splitk_gemv_triton
    from gemq.triton_kernels.fused_dequant_bmm import (
        fused_dequant_up_proj_triton, fused_dequant_down_proj_triton,
    )

    block, _, _ = build_deepseek_blocks(device)

    torch.manual_seed(3)
    hidden = (torch.randn(1, 1, HIDDEN_SIZE, device=device) * 0.5).half()
    x = hidden.view(-1, block.hidden_dim)

    router_logits = F.linear(x.float(), block.gate.weight.float(), None)
    expert_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    expert_weights, expert_indices = torch.topk(expert_weights, block.top_k, dim=-1)
    if block.norm_topk_prob:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

    clean = {}
    print()
    for i in range(REPEATS):
        print(f"  repeat {i}:")
        poison_allocator(device)
        with torch.no_grad():
            x1, x3 = fused_dequant_up_proj_triton(
                x, expert_indices[0], block.w1_wq, block.w3_wq,
                block.w1_scales, block.w1_zeros, block.w3_scales, block.w3_zeros,
                block.w1_nbits, block.w1_wq_strides, group_size=block.block_group_size,
            )
            ok_x1 = nan_report("routed up-proj x1", x1)
            ok_x3 = nan_report("routed up-proj x3", x3)

            x2 = fused_dequant_down_proj_triton(
                x1, x3, expert_indices[0], block.w2_wq,
                block.w2_scales, block.w2_zeros, block.w2_nbits, block.w2_wq_strides,
                group_size=block.block_group_size,
            )
            ok_x2 = nan_report("routed down-proj x2", x2)

            s1 = dequant_splitk_gemv_triton(
                x, block.shared_w1_wq, block.shared_w1_scales, block.shared_w1_zeros,
                block.shared_nbits, block.shared_group_size,
            )
            ok_s1 = nan_report("shared gate_proj", s1)
            s3 = dequant_splitk_gemv_triton(
                x, block.shared_w3_wq, block.shared_w3_scales, block.shared_w3_zeros,
                block.shared_nbits, block.shared_group_size,
            )
            ok_s3 = nan_report("shared up_proj", s3)
            s2 = dequant_splitk_gemv_triton(
                F.silu(s1) * s3, block.shared_w2_wq, block.shared_w2_scales,
                block.shared_w2_zeros, block.shared_nbits, block.shared_group_size,
            )
            ok_s2 = nan_report("shared down_proj", s2)

        for name, ok in [
            ("routed_x1", ok_x1), ("routed_x3", ok_x3), ("routed_x2", ok_x2),
            ("shared_s1", ok_s1), ("shared_s3", ok_s3), ("shared_s2", ok_s2),
        ]:
            clean.setdefault(name, []).append(ok)

    print("\n  summary (True = clean on that repeat):")
    for name, flags in clean.items():
        print(f"    {name:<12} {flags}")

    dirty = [name for name, flags in clean.items() if not all(flags)]
    assert not dirty, f"stages producing nan/inf: {dirty}"


@pytest.mark.cuda
def test_shared_expert_gemv_shapes_are_written_fully(device):
    """
    Isolate the shared-expert GEMV at its real shapes. Level 1 covered
    dequant_splitk_gemv_triton at 256x512; the shared experts use different ones
    (n_shared * moe_intermediate), which is the untested corner.
    """
    from gemq.triton_kernels.dequant_gemv import dequant_splitk_gemv_triton
    from conftest import make_gemlite_linear, relative_error, fp16_matmul_noise

    # shapes taken from build_deepseek_blocks: 2 shared experts x 512 intermediate
    shapes = [
        ("shared gate/up  [1024, 512]", 1024, 512),
        ("shared down     [512, 1024]", 512, 1024),
    ]
    print()
    for label, out_features, in_features in shapes:
        for nbits in (1, 2, 3, 4):
            torch.manual_seed(0)
            W = (torch.randn(out_features, in_features, device=device) * 0.02).half()
            x = (torch.randn(1, in_features, device=device) * 0.5).half()
            gl, _, W_deq = make_gemlite_linear(W, nbits, 128, device=device)

            poison_allocator(device)
            y = dequant_splitk_gemv_triton(x, gl.W_q, gl.scales, gl.zeros, gl.W_nbits, gl.group_size)
            n_nan = torch.isnan(y).sum().item()
            noise, ref32 = fp16_matmul_noise(x, W_deq)
            err = relative_error(y, ref32) if n_nan == 0 else float("nan")
            print(f"    {label}  nbits={nbits}  nan={n_nan:>4}  rel_err={err:.3e}")

            assert n_nan == 0, f"{label} nbits={nbits}: {n_nan} nan in output"
            # NOTE: split-K accumulates partial sums over K blocks, so its rounding is
            # worse than a single fp16 matmul -- and worse the larger K is. 10x the
            # plain-matmul noise still leaves two orders of magnitude of headroom
            # before anything a real bug would produce.
            assert err <= max(10.0 * noise, 2e-3), f"{label} nbits={nbits}: rel err {err:.3e}"


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 3, 4])
def test_splitk_gemv_is_nondeterministic(device, nbits):
    """
    Pin down *why* DeepSeek's decode path is non-deterministic while Mixtral's is not.

    dequant_splitk_gemv_triton launches a 2-D grid over (N, K) and accumulates partial
    products into the output. Float atomics commit in nondeterministic order, so the
    rounding differs run to run. Only DeepSeek reaches this kernel from
    forward_one_token (three calls, for the shared experts); Mixtral has no shared
    experts and is bit-stable.

    This test records the behaviour rather than demanding it: it reports the spread and
    only fails if the variation is large enough to be a correctness problem rather than
    accumulation-order rounding.
    """
    from gemq.triton_kernels.dequant_gemv import dequant_splitk_gemv_triton
    from conftest import make_gemlite_linear, relative_error

    torch.manual_seed(0)
    W = (torch.randn(512, 1024, device=device) * 0.02).half()
    x = (torch.randn(1, 1024, device=device) * 0.5).half()
    gl, _, _ = make_gemlite_linear(W, nbits, 128, device=device)

    runs = []
    for _ in range(REPEATS):
        y = dequant_splitk_gemv_triton(x, gl.W_q, gl.scales, gl.zeros, gl.W_nbits, gl.group_size)
        runs.append(y.float().clone())

    identical = all(torch.equal(runs[0], r) for r in runs[1:])
    spread = max(relative_error(r, runs[0]) for r in runs[1:])
    print(f"\n    nbits={nbits}  bit-identical={identical}  max spread={spread:.3e}")

    assert spread <= 1e-2, (
        f"split-K GEMV varies by {spread:.3e} between identical calls -- too large to be "
        f"atomic accumulation order alone"
    )
