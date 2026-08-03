"""
Shared fixtures and helpers for the real-vs-fake quantization equivalence tests.

The tests are layered by cost:
  * test_quant_linear_equiv  -- one linear layer, synthetic weights, seconds
  * test_moe_block_equiv     -- one MoE block, synthetic weights, seconds
  * test_real_vs_fake_ppl    -- full model, needs a saved real-quant checkpoint
  * test_decode_equiv        -- full model, decode path, needs the same checkpoint

Everything here imports torch/hqq/gemlite lazily, so the suite can at least be
collected on a machine without CUDA or without those packages installed.
"""

import pytest


# --------------------------------------------------------------------------- options
def pytest_addoption(parser):
    parser.addoption(
        "--model-path", action="store", default=None,
        help="Path to a saved real-quant checkpoint (results/real_quant_models/...)",
    )
    parser.addoption(
        "--model-name", action="store", default="deepseek-ai/DeepSeek-V2-Lite",
        help="Key into gemq.utils.model_utils.NAME_TO_MODEL for the checkpoint",
    )
    parser.addoption(
        "--nseq", action="store", type=int, default=8,
        help="Number of sequences used for the perplexity comparison",
    )
    parser.addoption(
        "--seqlen", action="store", type=int, default=2048,
        help="Sequence length used for the perplexity comparison",
    )
    parser.addoption(
        "--ndecode", action="store", type=int, default=32,
        help="Number of tokens to greedily decode in the decode-path test",
    )
    parser.addoption(
        "--no-mscale-fix", action="store_true", default=False,
        help="Skip align_deepseek_softmax_scale when running on HF's built-in "
             "implementation, to measure the gap it is meant to close.",
    )
    parser.addoption(
        "--no-trust-remote-code", action="store_true", default=False,
        help="Load the checkpoint with HF's built-in modeling code instead of the "
             "modeling_*.py shipped in the checkpoint. The DeepSeek scripts default to "
             "the official implementation (use_official_impl=true), so the tests do too.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA device")
    config.addinivalue_line("markers", "checkpoint: requires --model-path")


@pytest.fixture(scope="session")
def model_path(request):
    path = request.config.getoption("--model-path")
    if not path:
        pytest.skip("needs --model-path pointing at a real-quant checkpoint")
    return path


@pytest.fixture(scope="session")
def model_name(request):
    return request.config.getoption("--model-name")


@pytest.fixture(scope="session")
def nseq(request):
    return request.config.getoption("--nseq")


@pytest.fixture(scope="session")
def seqlen(request):
    return request.config.getoption("--seqlen")


@pytest.fixture(scope="session")
def ndecode(request):
    return request.config.getoption("--ndecode")


@pytest.fixture(scope="session")
def trust_remote_code(request):
    return not request.config.getoption("--no-trust-remote-code")


@pytest.fixture(scope="session")
def mscale_fix(request):
    return not request.config.getoption("--no-mscale-fix")


@pytest.fixture(scope="session")
def device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return "cuda"


@pytest.fixture(autouse=True)
def _free_cuda():
    """Keep peak memory down between tests."""
    yield
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- helpers
def quantize_weight(W, nbits, group_size):
    """
    RTN-quantize a [out_features, in_features] weight exactly the way the pipeline
    treats expert weights: grouping along the input dim, i.e. HQQ axis=1.

    Returns (Q, scales, zeros, W_dequant) where Q/scales/zeros are shaped
    [num_groups, group_size] / [num_groups, 1] / [num_groups, 1] and W_dequant is
    the fp16 weight the *fake* quant model would carry.
    """
    from gemq.quantizers.rtn import RTNWeightQuantizer

    quantizer = RTNWeightQuantizer(W, nbits=nbits, groupsize=group_size)
    Q, scales, zeros = quantizer.quantize()
    W_deq = quantizer.dequantize(Q, scales, zeros).reshape(W.shape).to(W.dtype)
    return Q, scales, zeros, W_deq


def make_gemlite_linear(W, nbits, group_size, bias=None, device="cuda"):
    """
    Build the real-quant representation of a weight: RTN -> HQQLinear -> GemLite,
    mirroring quant_utils.replace_linears + patch.create_gemlite_from_hqq.

    Returns (gemlite_linear, hqq_linear, W_dequant).
    """
    from gemq.utils.quant_utils import create_hqq_linear_from_quantized_weights
    from gemq.inference.patch import create_gemlite_from_hqq

    Q, scales, zeros, W_deq = quantize_weight(W, nbits, group_size)
    hqq_linear = create_hqq_linear_from_quantized_weights(
        Q, scales, zeros, tuple(W.shape), nbits, group_size, bias=bias, device=device
    )
    gemlite_linear = create_gemlite_from_hqq(hqq_linear)
    return gemlite_linear, hqq_linear, W_deq


def relative_error(actual, reference):
    """Frobenius-norm relative error, computed in fp32."""
    actual = actual.float()
    reference = reference.float()
    denom = reference.norm().clamp(min=1e-12)
    return ((actual - reference).norm() / denom).item()


def fp16_matmul_noise(x, W):
    """
    Relative error that a *plain* torch fp16 matmul already incurs against an fp32
    reference. Used as the yardstick for kernel tolerances: a custom kernel is
    considered equivalent when its error is within a small multiple of this.
    """
    import torch.nn.functional as F

    ref32 = F.linear(x.float(), W.float())
    ref16 = F.linear(x, W)
    return relative_error(ref16, ref32), ref32
