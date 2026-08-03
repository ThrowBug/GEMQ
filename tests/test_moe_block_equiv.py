"""
Level 2: does a QuantFused*MoEBlock compute the same thing as the stock HF block
holding the dequantized fp16 weights?

Two things make this the most valuable file in the suite:

  1. It covers *both* dispatch branches of QuantFused*MoEBlock.forward. A
     perplexity run only ever feeds whole sequences, so it exclusively exercises
     forward_n_tokens (group-GEMM kernels). The decode path used by
     benchmark_generate only ever exercises forward_one_token (fused bmm / splitk
     gemv kernels). Neither end-to-end check covers the other half.

  2. It uses mixed bit-widths across experts, which is the entire point of GEMQ
     and the case where the per-expert stride bookkeeping in the fused blocks can
     go wrong.

The blocks are built from synthetic weights, so this runs in seconds and needs no
checkpoint.
"""

import copy

import pytest
import torch

from conftest import quantize_weight, relative_error


HIDDEN_SIZE = 512
MOE_INTERMEDIATE = 512
NUM_EXPERTS = 8
TOP_K = 2
GROUP_SIZE = 128
NUM_TOKENS = 16

# per-expert bit-widths, interleaved the way a real GEMQ allocation looks
EXPERT_BITS = [1, 2, 3, 2, 3, 1, 2, 3]
SHARED_BITS = 3

# fused MoE blocks route in fp32 and accumulate in fp16; allow a little more slack
# than the single-linear tests, still far below what a real bug would produce
BLOCK_REL_ERROR_TOL = 5e-2


# --------------------------------------------------------------------------- builders
def _deepseek_config():
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

    return DeepseekV2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=NUM_EXPERTS,
        n_shared_experts=2,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        first_k_dense_replace=0,
    )


def _mixtral_config():
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig

    return MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )


def _assert_finite_parameters(module, label):
    """
    Guard against a fixture handing the tests a module with uninitialized parameters.
    HF modules that build parameters with torch.empty only get initialized by
    from_pretrained/post_init, so directly constructed blocks can carry garbage.
    """
    for name, param in module.named_parameters():
        assert torch.isfinite(param).all(), (
            f"{label}: parameter '{name}' contains nan/inf before quantization -- the "
            f"module was constructed without initializing it"
        )


def _quantize_linear_in_place(hf_module, ref_module, attr, nbits, device):
    """
    Quantize `hf_module.<attr>`, write the dequantized fp16 weight into the
    reference module, and swap the HF module's linear for an HQQLinear.
    """
    from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights

    linear = getattr(hf_module, attr)
    W = linear.weight.data
    Q, scales, zeros, W_deq = quantize_weight(W, nbits, GROUP_SIZE)

    getattr(ref_module, attr).weight.data = W_deq.clone()

    hqq_linear = create_hqq_linear_from_quantized_weights(
        Q, scales, zeros, tuple(W.shape), nbits, GROUP_SIZE,
        bias=linear.bias, device=device,
    )
    setattr(hf_module, attr, hqq_linear)


def build_deepseek_blocks(device):
    """Returns (quant_fused_block, reference_block_with_dequantized_weights, config)."""
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE
    from gemq.inference.patch import replace_linear_recursive
    from gemq.inference.moe_block import QuantFusedDeepseekV2MoEBlock

    config = _deepseek_config()
    torch.manual_seed(0)
    hf_block = DeepseekV2MoE(config)

    # NOTE: DeepseekV2MoEGate allocates its router with torch.empty and relies on
    # _init_weights, which only runs under from_pretrained/post_init. Constructing the
    # module directly leaves the router reading whatever was in that memory -- usually
    # plausible numbers, occasionally nan. Initialize it explicitly.
    torch.nn.init.normal_(hf_block.gate.weight, std=0.02)

    hf_block = hf_block.to(device).half().eval()
    _assert_finite_parameters(hf_block, "deepseek reference block")
    ref_block = copy.deepcopy(hf_block)

    for e, expert in enumerate(hf_block.experts):
        for attr in ("gate_proj", "up_proj", "down_proj"):
            _quantize_linear_in_place(expert, ref_block.experts[e], attr, EXPERT_BITS[e], device)
    for attr in ("gate_proj", "up_proj", "down_proj"):
        _quantize_linear_in_place(hf_block.shared_experts, ref_block.shared_experts, attr, SHARED_BITS, device)

    replace_linear_recursive(hf_block)
    quant_block = QuantFusedDeepseekV2MoEBlock.from_hf(config, hf_block)
    return quant_block, ref_block, config


def build_mixtral_blocks(device):
    """Returns (quant_fused_block, reference_block_with_dequantized_weights, config)."""
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from gemq.inference.patch import replace_linear_recursive
    from gemq.inference.moe_block import QuantFusedMixtralMoEBlock

    config = _mixtral_config()
    torch.manual_seed(0)
    hf_block = MixtralSparseMoeBlock(config).to(device).half().eval()
    _assert_finite_parameters(hf_block, "mixtral reference block")
    ref_block = copy.deepcopy(hf_block)

    for e, expert in enumerate(hf_block.experts):
        for attr in ("w1", "w2", "w3"):
            _quantize_linear_in_place(expert, ref_block.experts[e], attr, EXPERT_BITS[e], device)

    replace_linear_recursive(hf_block)
    quant_block = QuantFusedMixtralMoEBlock.from_hf(config, hf_block)
    return quant_block, ref_block, config


def _as_tensor(out):
    """Mixtral blocks return (hidden_states, router_logits); DeepSeek returns a tensor."""
    return out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- guards
class _OfficialBlockStub:
    """
    Stands in for a block coming from the official modeling_deepseek.py loaded via
    trust_remote_code, which lands in a `transformers_modules.*` package. Only the
    class's module path matters to the guard.
    """


_OfficialBlockStub.__module__ = "transformers_modules.deepseek.modeling_deepseek"


@pytest.mark.cuda
def test_deepseek_feature_guard_accepts_default_config(device):
    """Expert-selection features the fused forward does not implement."""
    from gemq.inference.moe_block import check_deepseek_routing_supported

    check_deepseek_routing_supported(_deepseek_config())


@pytest.mark.cuda
@pytest.mark.parametrize(
    "field, value",
    [("topk_method", "group_limited_greedy"), ("scoring_func", "sigmoid")],
)
def test_deepseek_feature_guard_rejects_unsupported_config(device, field, value):
    from gemq.inference.moe_block import check_deepseek_routing_supported

    config = _deepseek_config()
    setattr(config, field, value)
    with pytest.raises(AssertionError):
        check_deepseek_routing_supported(config)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "official, norm_topk_prob, top_k, scaling, should_pass",
    [
        # official gate: normalization and scaling are mutually exclusive branches
        (True, False, 2, 1.0, True),    # neither side scales
        (True, False, 2, 16.0, False),  # official scales, fused does not
        (True, True, 2, 16.0, True),    # both normalize; scaling is not on that branch
        (True, True, 1, 1.0, False),    # official scales a single weight, fused makes it 1.0
        # HF built-in gate: always scales, never normalizes
        (False, False, 2, 1.0, True),
        (False, True, 2, 1.0, False),   # fused normalizes, built-in does not
        (False, False, 2, 16.0, False),
    ],
)
def test_deepseek_gate_guard_matches_reference_semantics(
    device, official, norm_topk_prob, top_k, scaling, should_pass
):
    """
    The two reference implementations combine routing weights differently, so the
    guard has to key off which one the model was loaded with. This pins the whole
    truth table rather than one config.
    """
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE
    from gemq.inference.moe_block import check_deepseek_gate_matches

    config = _deepseek_config()
    config.norm_topk_prob = norm_topk_prob
    config.num_experts_per_tok = top_k
    config.routed_scaling_factor = scaling

    block = _OfficialBlockStub() if official else DeepseekV2MoE.__new__(DeepseekV2MoE)

    if should_pass:
        check_deepseek_gate_matches(config, block)
    else:
        with pytest.raises(AssertionError):
            check_deepseek_gate_matches(config, block)


@pytest.mark.cuda
@pytest.mark.parametrize("builder", [build_deepseek_blocks, build_mixtral_blocks],
                         ids=["deepseek", "mixtral"])
def test_one_token_path_bitwidth_invariant(device, builder):
    """
    forward_one_token passes only w1's nbits/strides and reuses them for w3, which
    holds because bit allocation is per-expert. moe_block.check_w1_w3_aligned now
    enforces it at load time; this confirms a real block satisfies it.
    """
    quant_block, _, _ = builder(device)

    assert torch.equal(quant_block.w1_nbits, quant_block.w3_nbits)
    assert torch.equal(quant_block.w1_wq_strides, quant_block.w3_wq_strides)


# --------------------------------------------------------------------------- equivalence
@pytest.mark.cuda
@pytest.mark.parametrize("builder", [build_deepseek_blocks, build_mixtral_blocks],
                         ids=["deepseek", "mixtral"])
def test_prefill_path_matches_reference(device, builder):
    """forward_n_tokens (group-GEMM kernels) -- the path a perplexity run exercises."""
    quant_block, ref_block, _ = builder(device)

    torch.manual_seed(1)
    x = (torch.randn(1, NUM_TOKENS, HIDDEN_SIZE, device=device) * 0.5).half()

    with torch.no_grad():
        y_quant = _as_tensor(quant_block(x))
        y_ref = _as_tensor(ref_block(x))

    err = relative_error(y_quant, y_ref)
    print(f"\n[prefill] rel err = {err:.3e} (tol {BLOCK_REL_ERROR_TOL:.1e})")
    assert err <= BLOCK_REL_ERROR_TOL, (
        f"fused block diverges from the dequantized HF block on the prefill path: "
        f"rel err {err:.3e} > {BLOCK_REL_ERROR_TOL:.3e}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("builder", [build_deepseek_blocks, build_mixtral_blocks],
                         ids=["deepseek", "mixtral"])
def test_decode_path_matches_reference(device, builder):
    """
    forward_one_token (fused bmm / splitk gemv kernels) -- the path benchmark_generate
    actually runs and that no perplexity number can ever cover.
    """
    quant_block, ref_block, _ = builder(device)

    torch.manual_seed(2)
    x = (torch.randn(1, 1, HIDDEN_SIZE, device=device) * 0.5).half()

    with torch.no_grad():
        y_quant = _as_tensor(quant_block(x))
        y_ref = _as_tensor(ref_block(x))

    err = relative_error(y_quant, y_ref)
    print(f"\n[decode] rel err = {err:.3e} (tol {BLOCK_REL_ERROR_TOL:.1e})")
    assert err <= BLOCK_REL_ERROR_TOL, (
        f"fused block diverges from the dequantized HF block on the decode path: "
        f"rel err {err:.3e} > {BLOCK_REL_ERROR_TOL:.3e}"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("builder", [build_deepseek_blocks, build_mixtral_blocks],
                         ids=["deepseek", "mixtral"])
def test_decode_path_agrees_with_prefill_path(device, builder):
    """
    Feeding the same single token through both dispatch branches must give the same
    answer. This isolates a kernel disagreement from a shared upstream problem: if
    both branches are wrong in the same way, the two tests above fail while this
    one passes.
    """
    quant_block, _, _ = builder(device)

    torch.manual_seed(3)
    x = (torch.randn(1, 1, HIDDEN_SIZE, device=device) * 0.5).half()

    with torch.no_grad():
        y_one = _as_tensor(quant_block.forward_one_token(x))
        y_many = _as_tensor(quant_block.forward_n_tokens(x))

    err = relative_error(y_one, y_many)
    print(f"\n[one-vs-many] rel err = {err:.3e} (tol {BLOCK_REL_ERROR_TOL:.1e})")
    assert err <= BLOCK_REL_ERROR_TOL, (
        f"forward_one_token and forward_n_tokens disagree on the same input: "
        f"rel err {err:.3e} > {BLOCK_REL_ERROR_TOL:.3e}"
    )
