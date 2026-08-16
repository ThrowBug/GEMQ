"""Static micro-batch generation for GEMQ's OpenAI-compatible server.

The implementation intentionally starts with the existing multi-token MoE path for
``batch_size > 1``.  It establishes correct padding, cache, sampling, and per-request
stopping semantics before a dedicated batched decode Triton kernel is introduced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Collection, Optional, Sequence

import torch

from gemq.benchmark_generate import logits_to_probs, multinomial_sample_one_no_sync
from gemq.inference.kv_cache import StaticCache


@dataclass
class BatchGenerationResult:
    token_ids: list[list[int]]
    generated_tokens: list[int]
    stopped_on_eos: list[bool]
    prefill_latency: float
    decode_latency: float
    decode_throughput: float


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_batch(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """Sample one token per row from ``[batch, vocab]`` logits."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1).to(dtype=torch.long)

    probabilities = logits_to_probs(logits, temperature=temperature, top_k=top_k)
    return multinomial_sample_one_no_sync(probabilities).reshape(-1).to(dtype=torch.long)


def _matches_eos(tokens: torch.Tensor, eos_tensor: Optional[torch.Tensor]) -> torch.Tensor:
    if eos_tensor is None:
        return torch.zeros_like(tokens, dtype=torch.bool)
    return torch.isin(tokens, eos_tensor)


@torch.inference_mode()
def generate_batch(
    model,
    prompts: Sequence[torch.Tensor],
    max_new_tokens: Sequence[int],
    pad_token_id: int,
    eos_token_ids: Optional[Collection[int]] = None,
    eos_check_interval: int = 8,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> BatchGenerationResult:
    """Generate a static batch with independent length and EOS stopping.

    Prompts are left padded so every sequence ends at the same physical cache
    position.  ``position_ids`` still follow each prompt's real, unpadded length.
    Finished rows remain in the physical batch until all rows finish; their subsequent
    tokens are ignored.  This keeps the first implementation simple and deterministic.
    """
    if not prompts:
        raise ValueError("generate_batch requires at least one prompt")
    if len(prompts) != len(max_new_tokens):
        raise ValueError("prompts and max_new_tokens must have the same length")
    if eos_check_interval < 1:
        raise ValueError("eos_check_interval must be at least 1")
    if any(limit < 1 for limit in max_new_tokens):
        raise ValueError("every max_new_tokens value must be at least 1")
    if any(prompt.ndim != 1 or prompt.numel() < 1 for prompt in prompts):
        raise ValueError("every prompt must be a non-empty one-dimensional tensor")

    device = prompts[0].device
    if any(prompt.device != device for prompt in prompts):
        raise ValueError("all prompts must be on the same device")

    batch_size = len(prompts)
    prompt_lengths = torch.tensor(
        [int(prompt.numel()) for prompt in prompts],
        device=device,
        dtype=torch.long,
    )
    padded_prompt_length = int(prompt_lengths.max().item())
    max_generation_length = max(int(limit) for limit in max_new_tokens)

    input_ids = torch.full(
        (batch_size, padded_prompt_length),
        int(pad_token_id),
        device=device,
        dtype=prompts[0].dtype,
    )
    full_attention_mask = torch.zeros(
        (batch_size, padded_prompt_length + max_generation_length),
        device=device,
        dtype=torch.long,
    )
    for row, prompt in enumerate(prompts):
        length = int(prompt.numel())
        input_ids[row, -length:] = prompt
        full_attention_mask[row, padded_prompt_length - length : padded_prompt_length] = 1

    attention_mask = full_attention_mask[:, :padded_prompt_length]

    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    all_cache_positions = torch.arange(
        padded_prompt_length + max_generation_length,
        device=device,
        dtype=torch.long,
    )
    cache_position = all_cache_positions[:padded_prompt_length]
    kv_cache = StaticCache(
        model.config,
        max_cache_len=padded_prompt_length + max_generation_length,
    )

    _device_sync(device)
    prefill_started = time.perf_counter()
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=kv_cache,
        cache_position=cache_position,
        use_cache=True,
    )
    first_tokens = _sample_batch(
        outputs.logits[:, -1, :],
        temperature=temperature,
        top_k=top_k,
    )
    _device_sync(device)
    prefill_latency = time.perf_counter() - prefill_started

    generated = torch.full(
        (batch_size, max_generation_length),
        int(pad_token_id),
        device=device,
        dtype=torch.long,
    )
    generated[:, 0] = first_tokens
    generated_lengths = torch.ones(batch_size, device=device, dtype=torch.long)
    generation_limits = torch.tensor(max_new_tokens, device=device, dtype=torch.long)

    eos_ids = sorted(set(int(token_id) for token_id in (eos_token_ids or [])))
    eos_tensor = (
        torch.tensor(eos_ids, device=device, dtype=first_tokens.dtype)
        if eos_ids
        else None
    )
    stopped_on_eos = _matches_eos(first_tokens, eos_tensor)
    finished = stopped_on_eos | (generated_lengths >= generation_limits)

    decode_latency = 0.0
    # Prefill was synchronized for latency measurement above, so checking once here
    # does not introduce a new per-token synchronization point.
    all_finished_after_prefill = bool(finished.all().item())
    if max_generation_length > 1 and not all_finished_after_prefill:
        _device_sync(device)
        decode_started = time.perf_counter()
        current_tokens = first_tokens.view(batch_size, 1)

        for step in range(1, max_generation_length):
            active_before_step = ~finished
            model_input = torch.where(
                active_before_step.view(batch_size, 1),
                current_tokens,
                torch.full_like(current_tokens, int(pad_token_id)),
            )
            full_attention_mask[:, padded_prompt_length + step - 1] = (
                active_before_step.to(dtype=torch.long)
            )
            decode_attention_mask = full_attention_mask[
                :, : padded_prompt_length + step
            ]
            decode_position_ids = torch.where(
                active_before_step,
                prompt_lengths + step - 1,
                torch.zeros_like(prompt_lengths),
            ).view(batch_size, 1)
            decode_cache_position = all_cache_positions[
                padded_prompt_length + step - 1 : padded_prompt_length + step
            ]

            outputs = model(
                model_input,
                attention_mask=decode_attention_mask,
                position_ids=decode_position_ids,
                past_key_values=kv_cache,
                cache_position=decode_cache_position,
                use_cache=True,
            )
            next_tokens = _sample_batch(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
            )

            generated[:, step] = torch.where(
                active_before_step,
                next_tokens,
                torch.full_like(next_tokens, int(pad_token_id)),
            )
            generated_lengths += active_before_step.to(dtype=torch.long)

            hit_eos = active_before_step & _matches_eos(next_tokens, eos_tensor)
            stopped_on_eos |= hit_eos
            finished |= hit_eos | (generated_lengths >= generation_limits)
            current_tokens = next_tokens.view(batch_size, 1)

            should_check_finished = (
                (step + 1) % eos_check_interval == 0
                or step == max_generation_length - 1
            )
            if should_check_finished and bool(finished.all().item()):
                break

        _device_sync(device)
        decode_latency = time.perf_counter() - decode_started

    generated_cpu = generated.cpu()
    lengths_cpu = generated_lengths.cpu().tolist()
    stopped_cpu = stopped_on_eos.cpu().tolist()
    token_ids = [
        generated_cpu[row, : int(length)].tolist()
        for row, length in enumerate(lengths_cpu)
    ]
    decoded_token_count = sum(max(int(length) - 1, 0) for length in lengths_cpu)

    return BatchGenerationResult(
        token_ids=token_ids,
        generated_tokens=[int(length) for length in lengths_cpu],
        stopped_on_eos=[bool(value) for value in stopped_cpu],
        prefill_latency=prefill_latency,
        decode_latency=decode_latency,
        decode_throughput=(
            decoded_token_count / decode_latency if decode_latency > 0 else 0.0
        ),
    )
