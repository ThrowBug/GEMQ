"""
Level 4: the decode path, end to end.

A perplexity run feeds whole sequences, so QuantFused*MoEBlock.forward always takes
the forward_n_tokens branch. Everything benchmark_generate does after prefill takes
the forward_one_token branch instead, through a different set of Triton kernels and
through gemq/inference/kv_cache.StaticCache -- none of which a perplexity number can
reach.

Comparison is **teacher forced**: both models are fed exactly the same tokens at every
step. Free-running greedy decode is useless as an equivalence test, because a single
argmax flip -- which a 1e-3 logit difference can cause when the top two candidates are
close -- sends the two models into different contexts, after which the traces are not
comparable at all.

The tests compute a 2x2 to localize any divergence:

                    full-sequence forward     token-by-token with cache
    fake            A                         B
    real-patched    C                         D

    A vs C : prefill path. Already known good from the perplexity comparison.
    A vs B : does the fake model agree with itself through StaticCache?
    C vs D : does the real model's forward_one_token agree with its forward_n_tokens?
    B vs D : the actual question.

If B vs D is large while A vs B is ~0 and C vs D is large, the decode kernels are at
fault. If A vs B is also large, the cache plumbing is implicated instead.
"""

import gc
import inspect

import pytest
import torch

from test_real_vs_fake_ppl import _load_real, _hqq_to_dense, _free


# long enough to prefill on and still have tokens left to force
TEXT = (
    "The key property of a mixture-of-experts model is that only a small subset of "
    "the parameters is active for any given token, which keeps the computational cost "
    "far below what the total parameter count would suggest."
)


@torch.inference_mode()
def _forward_full(model, input_ids):
    """Logits for every position in one shot -- the forward_n_tokens branch."""
    return model(input_ids, use_cache=False).logits.float().cpu()


@torch.inference_mode()
def _forward_stepwise(model, input_ids, prompt_len, n_steps):
    """
    Prefill `prompt_len` tokens, then feed the next `n_steps` tokens one at a time
    through StaticCache -- takes the forward_one_token branch. Teacher forced: the fed
    tokens come from input_ids, never from the model's own predictions.

    Returns logits [n_steps, vocab]; row t is the prediction made after seeing
    input_ids[:prompt_len + t].
    """
    from gemq.inference.kv_cache import StaticCache

    device = input_ids.device
    total = prompt_len + n_steps
    kv_cache = StaticCache(model.config, max_cache_len=total)

    cache_position = torch.arange(0, prompt_len, device=device)
    out = model(input_ids[:, :prompt_len], past_key_values=kv_cache, cache_position=cache_position)
    logits = [out.logits[:, -1, :].float().cpu()]

    for t in range(n_steps - 1):
        pos = prompt_len + t
        cache_position = torch.tensor([pos], device=device)
        out = model(
            input_ids[:, pos : pos + 1], past_key_values=kv_cache, cache_position=cache_position
        )
        logits.append(out.logits[:, -1, :].float().cpu())

    return torch.cat(logits, dim=0)


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


def _argmax_agreement(a, b):
    return (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()


@pytest.fixture(scope="session")
def decode_traces(model_path, model_name, ndecode, trust_remote_code):
    pytest.importorskip("hqq")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    from transformers import AutoTokenizer
    from gemq.inference.patch import prepare_for_inference

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    input_ids = tokenizer(TEXT, return_tensors="pt").input_ids.to("cuda")

    n_steps = min(ndecode, input_ids.shape[1] // 2)
    prompt_len = input_ids.shape[1] - n_steps
    assert prompt_len > 0, "prompt text is too short for the requested number of steps"

    traces = {}

    # fake twin
    model = _load_real(model_path, trust_remote_code)
    # The decode loop needs `cache_position`, which the official modeling code predates,
    # so unlike the perplexity comparison this can only run on HF's built-in code.
    if "cache_position" not in inspect.signature(model.forward).parameters:
        _free(model)
        pytest.skip(
            f"{type(model).__module__} has no `cache_position` parameter, so it cannot "
            f"drive StaticCache. Re-run with --no-trust-remote-code to compare the decode "
            f"path on HF's built-in implementation."
        )
    _hqq_to_dense(model)
    full = _forward_full(model, input_ids)
    traces["fake_full"] = full[0, prompt_len - 1 : prompt_len - 1 + n_steps, :]
    traces["fake_step"] = _forward_stepwise(model, input_ids, prompt_len, n_steps)
    _free(model)

    # real, patched -- the configuration the bench script runs
    model = _load_real(model_path, trust_remote_code)
    prepare_for_inference(model, model_name, is_fp=False)
    full = _forward_full(model, input_ids)
    traces["real_full"] = full[0, prompt_len - 1 : prompt_len - 1 + n_steps, :]
    traces["real_step"] = _forward_stepwise(model, input_ids, prompt_len, n_steps)
    # NOTE: decode is not bit-reproducible (split-K atomics in the shared-expert GEMV).
    # Measure that noise floor; no real-vs-fake number means anything below it.
    traces["real_step_again"] = _forward_stepwise(model, input_ids, prompt_len, n_steps)
    _free(model)

    pairs = [
        ("noise floor real_step vs real_step_again", "real_step", "real_step_again"),
        ("A vs C  prefill      fake_full  vs real_full", "fake_full", "real_full"),
        ("A vs B  fake self    fake_full  vs fake_step", "fake_full", "fake_step"),
        ("C vs D  real self    real_full  vs real_step", "real_full", "real_step"),
        ("B vs D  decode       fake_step  vs real_step", "fake_step", "real_step"),
    ]
    print("\n" + "=" * 68)
    print(f" teacher-forced logits over {n_steps} steps (prompt_len={prompt_len})")
    print("-" * 68)
    print(f" {'comparison':<46} {'rel err':>10} {'argmax':>8}")
    for label, left, right in pairs:
        rel = _rel(traces[left], traces[right])
        agree = _argmax_agreement(traces[left], traces[right])
        print(f" {label:<46} {rel:>10.3e} {agree:>7.0%}")
    print("=" * 68)

    return traces


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_fake_model_agrees_with_itself_through_cache(decode_traces):
    """
    Control. The fake twin is plain HF modules, so feeding tokens one at a time through
    StaticCache must reproduce its own full-sequence logits. If this fails, the cache
    plumbing is broken and nothing below is interpretable.
    """
    rel = _rel(decode_traces["fake_step"], decode_traces["fake_full"])
    assert rel <= 2e-2, (
        f"the fake model disagrees with itself between full-sequence and cached "
        f"step-by-step forwards (rel err {rel:.3e}); suspect StaticCache, not the kernels"
    )


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_real_model_decode_agrees_with_its_own_prefill(decode_traces):
    """forward_one_token must agree with forward_n_tokens on the same real model."""
    rel = _rel(decode_traces["real_step"], decode_traces["real_full"])
    assert rel <= 2e-2, (
        f"the real model's decode path disagrees with its own prefill path "
        f"(rel err {rel:.3e}); the fused one-token kernels are the prime suspect"
    )


@pytest.mark.cuda
@pytest.mark.checkpoint
def test_decode_logits_match_fake(decode_traces):
    """
    The actual question: real decode vs fake decode, step for step.

    Judged against the model's own run-to-run noise floor, since the decode path is not
    bit-reproducible (split-K atomics in the shared-expert GEMV). A real-vs-fake gap of
    the same order as real-vs-itself says the paths are equivalent to within what the
    kernels can even reproduce.
    """
    rel = _rel(decode_traces["real_step"], decode_traces["fake_step"])
    floor = _rel(decode_traces["real_step"], decode_traces["real_step_again"])
    prefill_rel = _rel(decode_traces["real_full"], decode_traces["fake_full"])

    assert rel <= max(10 * floor, 2e-2), (
        f"real-quant decode logits differ from fake by {rel:.3e} relative, well above "
        f"the {floor:.3e} the same model shows against itself, and above the {prefill_rel:.3e} "
        f"the prefill path shows for the same positions. The decode-only kernels "
        f"(fused_dequant_up/down_proj, splitk gemv) are implicated."
    )
