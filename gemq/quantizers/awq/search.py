"""AWQ scale and clipping search, adapted from MIT-HAN-Lab/llm-awq."""

from __future__ import annotations

import gc

import torch
import torch.nn as nn

from .quantizer import fake_quantize_grouped, fake_quantize_weight


def _tree_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    return value


def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor or tensor-first output, got {type(output)!r}")


def _chunks(value, chunk_size):
    return [
        value[start : start + chunk_size]
        for start in range(0, value.shape[0], chunk_size)
    ]


@torch.no_grad()
def search_scale(
    module,
    linears,
    input_feat,
    bits,
    groupsize=128,
    n_grid=20,
    search_batch_size=1,
    module_kwargs=None,
):
    """Find a single activation-aware input-channel scale for ``linears``."""
    if not linears or len(linears) != len(bits):
        raise ValueError("linears and bits must be non-empty lists of equal length")
    if n_grid <= 0 or search_batch_size <= 0:
        raise ValueError("n_grid and search_batch_size must be positive")
    if any(not isinstance(bit, int) or not 1 <= bit < 16 for bit in bits):
        raise ValueError("Scale-search bits must be integers in [1, 15]")

    device = next(module.parameters()).device
    module_kwargs = dict(module_kwargs or {})
    module_kwargs.pop("use_cache", None)
    module_kwargs = _tree_to_device(module_kwargs, device)
    chunk_size = search_batch_size if input_feat.ndim >= 3 else 4096
    input_chunks = _chunks(input_feat, chunk_size)

    reference_outputs = []
    for chunk in input_chunks:
        reference_outputs.append(
            _first_tensor(
                module(chunk.to(device=device, non_blocking=True), **module_kwargs)
            )
            .detach()
            .to("cpu")
        )

    flattened = input_feat.reshape(-1, input_feat.shape[-1])
    activation_scale = flattened.abs().mean(dim=0).float().clamp(min=1e-4)
    original_weights = [linear.weight.detach().to("cpu").clone() for linear in linears]

    best_error = float("inf")
    best_scale = None
    try:
        for grid_idx in range(n_grid):
            ratio = grid_idx / n_grid
            scale = activation_scale.pow(ratio).clamp(min=1e-4)
            scale = scale / torch.sqrt(scale.max() * scale.min())

            for linear, original, bit in zip(linears, original_weights, bits):
                original_device = original.to(device=device, dtype=linear.weight.dtype)
                device_scale = scale.to(device=device, dtype=linear.weight.dtype)
                trial = fake_quantize_weight(
                    original_device * device_scale.view(1, -1),
                    nbits=bit,
                    groupsize=groupsize,
                )
                linear.weight.data.copy_(trial / device_scale.view(1, -1))

            squared_error = 0.0
            numel = 0
            for chunk, reference in zip(input_chunks, reference_outputs):
                output = _first_tensor(
                    module(chunk.to(device=device, non_blocking=True), **module_kwargs)
                )
                difference = output.float() - reference.to(
                    device=device, dtype=torch.float32
                )
                squared_error += difference.square().sum().item()
                numel += difference.numel()
            error = squared_error / max(numel, 1)
            if error < best_error:
                best_error = error
                best_scale = scale.clone()

            for linear, original in zip(linears, original_weights):
                linear.weight.data.copy_(
                    original.to(device=device, dtype=linear.weight.dtype)
                )
    finally:
        for linear, original in zip(linears, original_weights):
            linear.weight.data.copy_(
                original.to(device=device, dtype=linear.weight.dtype)
            )

    if best_scale is None or not torch.isfinite(best_scale).all():
        raise RuntimeError("AWQ scale search did not produce a finite scale")
    return best_scale.to("cpu"), best_error


@torch.no_grad()
def apply_norm_input_scale(norm, linears, scale):
    """Absorb ``x / scale`` into a norm and all of its linear consumers."""
    if not hasattr(norm, "weight") or norm.weight is None:
        raise TypeError("The preceding normalization module must have a weight")
    device = norm.weight.device
    scale = scale.to(device=device, dtype=norm.weight.dtype)
    if norm.weight.numel() != scale.numel():
        raise ValueError("Normalization weight and AWQ scale sizes do not match")
    norm.weight.div_(scale)
    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scale)
    for linear in linears:
        if linear.weight.shape[1] != scale.numel():
            raise ValueError("Linear input dimension and AWQ scale sizes do not match")
        linear.weight.mul_(
            scale.view(1, -1).to(linear.weight.device, linear.weight.dtype)
        )


@torch.no_grad()
def apply_linear_pair_scale(previous, following, scale):
    """Absorb an exact internal scale between two linear transformations."""
    if not isinstance(previous, nn.Linear) or not isinstance(following, nn.Linear):
        raise TypeError("AWQ internal scaling expects two nn.Linear modules")
    scale = scale.to(device=previous.weight.device, dtype=previous.weight.dtype)
    if previous.weight.shape[0] != scale.numel():
        raise ValueError(
            "Previous linear output dimension and scale sizes do not match"
        )
    if following.weight.shape[1] != scale.numel():
        raise ValueError(
            "Following linear input dimension and scale sizes do not match"
        )
    previous.weight.div_(scale.view(-1, 1))
    if previous.bias is not None:
        previous.bias.div_(scale)
    following.weight.mul_(
        scale.view(1, -1).to(following.weight.device, following.weight.dtype)
    )


@torch.no_grad()
def search_clip(
    weight,
    input_feat,
    nbits,
    groupsize=128,
    n_grid=20,
    max_shrink=0.5,
    n_sample_token=512,
    output_chunk_size=256,
):
    """Search symmetric per-output/per-group clipping thresholds."""
    if weight.ndim != 2:
        raise ValueError("weight must be rank 2")
    if weight.shape[1] % groupsize != 0:
        raise ValueError(
            f"weight input dimension {weight.shape[1]} is not divisible by groupsize={groupsize}"
        )
    if n_grid <= 0 or not 0 < max_shrink <= 1:
        raise ValueError("n_grid must be positive and max_shrink must be in (0, 1]")
    if n_sample_token <= 0 or output_chunk_size <= 0:
        raise ValueError("n_sample_token and output_chunk_size must be positive")

    device = weight.device
    flattened = input_feat.reshape(-1, input_feat.shape[-1])
    sample_step = max(1, flattened.shape[0] // n_sample_token)
    sampled = flattened[::sample_step][:n_sample_token]
    sampled = sampled.to(device=device, dtype=weight.dtype)
    sampled = sampled.reshape(1, sampled.shape[0], -1, groupsize)
    grouped_weight = weight.reshape(weight.shape[0], 1, -1, groupsize)
    steps = max(1, int(max_shrink * n_grid))
    best_chunks = []

    for start in range(0, weight.shape[0], output_chunk_size):
        end = min(start + output_chunk_size, weight.shape[0])
        current = grouped_weight[start:end]
        original_max = current.abs().amax(dim=-1, keepdim=True)
        best_max = original_max.clone()
        minimum_error = torch.full_like(original_max, float("inf"))
        reference = (sampled * current).sum(dim=-1)

        for shrink_idx in range(steps):
            candidate_max = original_max * (1 - shrink_idx / n_grid)
            clipped = current.clamp(-candidate_max, candidate_max)
            quantized = fake_quantize_grouped(clipped, nbits)
            output = (sampled * quantized).sum(dim=-1)
            error = (output - reference).square().mean(dim=1).view_as(minimum_error)
            better = error < minimum_error
            minimum_error[better] = error[better]
            best_max[better] = candidate_max[better]
        best_chunks.append(best_max.squeeze(1).to("cpu"))

    del sampled, grouped_weight
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(best_chunks, dim=0)


@torch.no_grad()
def apply_clip(linear, clip_max):
    clip_max = clip_max.to(linear.weight.device, linear.weight.dtype)
    if clip_max.ndim == 2:
        clip_max = clip_max.unsqueeze(-1)
    if clip_max.ndim != 3 or clip_max.shape[0] != linear.weight.shape[0]:
        raise ValueError(
            "clip_max must have shape [out_features, num_groups, 1]"
        )
    if clip_max.shape[-1] != 1 or linear.weight.shape[1] % clip_max.shape[1] != 0:
        raise ValueError("clip_max groups do not partition the linear input dimension")
    grouped = linear.weight.data.reshape(clip_max.shape[0], clip_max.shape[1], -1)
    linear.weight.data = grouped.clamp(-clip_max, clip_max).reshape_as(linear.weight)
