"""
Level 3: end-to-end perplexity comparison for a saved real-quant checkpoint.

The fake-quant twin is *derived from the real checkpoint* by dequantizing every
HQQLinear back to an fp16 nn.Linear. That matters: both sides then carry bit-identical
weights, so any perplexity gap is attributable to the inference path alone rather
than to two separate quantization runs drifting apart.

Three variants are measured, matching the three code paths that exist today:

    fake          nn.Linear (fp16 dequantized) + HF MoE blocks
    real-unpatched HQQLinear                   + HF MoE blocks   (no --compile)
    real-patched  GemLiteLinearTriton          + QuantFused*MoEBlock (--compile)

Note that gemq/quantize.py evaluates perplexity *before* replace_linears packs the
weights, so the number printed by a --real_quant run has always been the fake-quant
perplexity. This file measures the other two for the first time.

Run with:
    pytest tests/test_real_vs_fake_ppl.py -s \
        --model-path results/real_quant_models/deepseek-ai/DeepSeek-V2-Lite/GEMQ/... \
        --model-name deepseek-ai/DeepSeek-V2-Lite
"""

import gc

import pytest
import torch


def _free(*objs):
    for obj in objs:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def _load_real(model_path, trust_remote_code=True, device="cuda", compute_dtype=torch.float16,
               mscale_fix=True):
    """Load a real-quant checkpoint on the modeling code it was quantized with."""
    from gemq.utils.hf_loading import load_quantized_model, align_deepseek_softmax_scale

    model = load_quantized_model(
        model_path, compute_dtype=compute_dtype, device=device,
        trust_remote_code=trust_remote_code,
    )
    if mscale_fix:
        align_deepseek_softmax_scale(model, verbose=False)
    return model.eval()


def _hqq_to_dense(module):
    """
    Recursively swap every HQQLinear for an fp16 nn.Linear holding its dequantized weight.

    NOTE: the twin is the full unquantized size (~31 GB for DeepSeek-V2-Lite, ~93 GB for
    Mixtral-8x7B). Models that do not fit on one card need CPU staging plus
    dispatch_model_to_all_devices; not implemented.
    """
    from hqq.core.quantize import HQQLinear

    for name, child in module.named_children():
        if isinstance(child, HQQLinear):
            W = child.dequantize()
            out_features, in_features = child.meta["shape"]
            W = W.reshape(out_features, in_features)

            dense = torch.nn.Linear(
                in_features, out_features, bias=child.bias is not None,
                device=W.device, dtype=W.dtype,
            )
            dense.weight.data = W.contiguous()
            if child.bias is not None:
                dense.bias.data = child.bias.data.clone().to(W.dtype)
            setattr(module, name, dense)
        else:
            _hqq_to_dense(child)


def _perplexity(model, tokenizer, seqlen, nseq, tag):
    from gemq.utils.eval_utils import get_testenc, compute_perplexity

    model.seqlen = seqlen
    testenc = get_testenc(tokenizer, "wikitext2", seqlen)
    input_ids = testenc.input_ids[:, : nseq * seqlen]

    # NOTE: mirrors eval_utils.evaluate_perplexity. Not just speed: DeepSeek's official
    # modeling code calls Cache.get_usable_length (removed in 4.57) when use_cache is on.
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.inference_mode():
            return compute_perplexity(model, input_ids, tag)
    finally:
        model.config.use_cache = use_cache


@pytest.fixture(scope="session")
def ppl_table(model_path, model_name, seqlen, nseq, trust_remote_code, mscale_fix):
    """
    Measure all three variants once. Each model is loaded, evaluated and freed
    before the next one, so peak memory stays at roughly one fp16 copy.
    """
    pytest.importorskip("hqq")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    from transformers import AutoTokenizer
    from gemq.inference.patch import prepare_for_inference

    from gemq.utils.hf_loading import describe_model_impl, uses_remote_code

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    results = {}

    # 1) fake twin: dequantize everything back to fp16
    model = _load_real(model_path, trust_remote_code, mscale_fix=mscale_fix)
    print(f"\n[modeling] implementation: {describe_model_impl(model)}")
    ships_remote_code = bool(getattr(model.config, "auto_map", None))

    if not uses_remote_code(model):
        from gemq.utils.hf_loading import deepseek_mscale_correction
        factor = deepseek_mscale_correction(model.config)
        if factor != 1.0:
            state = "applied" if mscale_fix else "skipped"
            print(f"[modeling] YaRN mscale^2 = {factor:.4f} ({state})")

    # only meaningful for checkpoints that actually ship their own modeling code;
    # Mixtral is natively supported and has no auto_map
    if ships_remote_code and trust_remote_code and not uses_remote_code(model):
        pytest.fail(
            f"asked for the checkpoint's own modeling code but got "
            f"{describe_model_impl(model)}. These numbers would not correspond to the "
            f"implementation the model was quantized and evaluated with."
        )
    _hqq_to_dense(model)
    results["fake"] = _perplexity(model, tokenizer, seqlen, nseq, "fake")
    _free(model)

    # 2) real, unpatched: HQQLinear + stock HF MoE blocks
    model = _load_real(model_path, trust_remote_code, mscale_fix=mscale_fix)
    results["real_unpatched"] = _perplexity(model, tokenizer, seqlen, nseq, "real-unpatched")
    _free(model)

    # 3) real, patched: GemLite + fused MoE blocks (what --compile runs)
    model = _load_real(model_path, trust_remote_code, mscale_fix=mscale_fix)
    prepare_for_inference(model, model_name, is_fp=False)
    results["real_patched"] = _perplexity(model, tokenizer, seqlen, nseq, "real-patched")
    _free(model)

    print("\n" + "=" * 56)
    print(f" wikitext2 perplexity over {nseq} x {seqlen} tokens")
    print("-" * 56)
    for key, value in results.items():
        delta = (value - results["fake"]) / results["fake"] * 100.0
        print(f" {key:<16} {value:>10.4f}   ({delta:+.3f} % vs fake)")
    print("=" * 56)

    return results


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_real_unpatched_matches_fake(ppl_table):
    """
    HQQ's own dequant-matmul against an fp16 matmul on the same weights. This is the
    tightest of the three comparisons -- the MoE blocks are identical, only the
    linear implementation differs.
    """
    fake, real = ppl_table["fake"], ppl_table["real_unpatched"]
    rel = abs(real - fake) / fake
    assert rel <= 0.01, (
        f"unpatched real-quant perplexity differs from fake by {rel * 100:.3f} % "
        f"({real:.4f} vs {fake:.4f}); the linear replacement alone should not move it"
    )


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_real_patched_matches_fake(ppl_table):
    """
    The full real-quant inference stack: GemLite linears plus the fused MoE blocks
    and their Triton kernels. Only exercises forward_n_tokens -- see
    test_decode_equiv.py for the decode half.
    """
    fake, real = ppl_table["fake"], ppl_table["real_patched"]
    rel = abs(real - fake) / fake
    assert rel <= 0.01, (
        f"patched real-quant perplexity differs from fake by {rel * 100:.3f} % "
        f"({real:.4f} vs {fake:.4f}); suspect the fused MoE blocks or their kernels"
    )


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_perplexity_is_finite(ppl_table):
    """Guard against a silently broken run reporting nan/inf rather than a number."""
    for key, value in ppl_table.items():
        assert value == value and value != float("inf"), f"{key} perplexity is {value}"
