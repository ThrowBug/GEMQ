"""Estimate BitDistiller-style max-prob confidence from existing teacher caches.

This follows the official code's max over vocabulary, not paper Eq. (5)'s
probability of the observed target token. Unlike the upstream estimator, we
exclude padding/final positions and divide by the actual valid-token count.
No full teacher forward or cache-format change is needed.
"""

import torch

from gemq.router_finetune.config import parse_cakld_gamma


@torch.no_grad()
def estimate_cakld_gamma(lm_head, teacher_targets, nsamples, *, token_chunk_size=256):
    """Stream cached teacher hidden states through the unchanged, frozen head.

    Chunking bounds temporary vocabulary tensors independently of calibration
    set size. token_chunk_size is an implementation/testing knob, not a training
    hyperparameter. This function does not change model parameters or RNG state.
    """
    hidden = teacher_targets.final_hidden_states
    if hidden is None:
        raise ValueError("CAKLD confidence requires cached teacher final hidden states.")
    if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(teacher_targets.input_ids.shape):
        raise ValueError("Teacher hidden states must match the cached input token shape.")
    if not 0 < nsamples <= hidden.shape[0] or hidden.shape[1] < 2:
        raise ValueError("CAKLD confidence requires available samples with at least two tokens.")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive.")
    attention_mask = teacher_targets.attention_mask
    if attention_mask is not None and tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("Teacher attention mask must match the cached input token shape.")

    head_parameter = next(lm_head.parameters())
    probability_sum = 0.0
    valid_tokens = 0
    for sample in range(nsamples):
        for start in range(0, hidden.shape[1] - 1, token_chunk_size):
            end = min(start + token_chunk_size, hidden.shape[1] - 1)
            chunk = hidden[sample, start:end]
            if attention_mask is not None:
                mask = attention_mask[sample].to(dtype=torch.bool)
                valid = mask[start:end] & mask[start + 1:end + 1]
                chunk = chunk[valid.to(chunk.device)]
            if chunk.shape[0] == 0:
                continue
            logits = lm_head(chunk.to(device=head_parameter.device, dtype=head_parameter.dtype)).float()
            # max softmax(z) = exp(max(z) - logsumexp(z)); no dense probability cache.
            max_probs = (logits.amax(-1) - torch.logsumexp(logits, dim=-1)).exp()
            if not torch.isfinite(max_probs).all():
                raise ValueError("Non-finite teacher probabilities while estimating CAKLD gamma.")
            probability_sum += max_probs.double().sum().item()
            valid_tokens += max_probs.numel()
            del logits, max_probs, chunk
    if valid_tokens == 0:
        raise ValueError("No valid next-token positions for CAKLD confidence estimation.")
    gamma = probability_sum / valid_tokens
    parse_cakld_gamma(gamma)
    return {
        "resolved_gamma": gamma,
        "confidence_definition": "max_prob",
        "gamma_source": "teacher_cache",
        "confidence_reduction": "valid_token_mean",
        "confidence_samples": nsamples,
        "confidence_valid_tokens": valid_tokens,
    }


def resolve_cakld_gamma(lm_head, teacher_targets, nsamples, gamma="auto"):
    """Manual gamma skips even the head-only confidence pass."""
    gamma = parse_cakld_gamma(gamma)
    if gamma == "auto":
        return estimate_cakld_gamma(lm_head, teacher_targets, nsamples)
    return {
        "resolved_gamma": gamma,
        "confidence_definition": "max_prob",
        "gamma_source": "manual",
        "confidence_reduction": "valid_token_mean",
        "confidence_samples": 0,
        "confidence_valid_tokens": 0,
    }
