"""Group-wise fake-quantization primitives used by GEMQ-AWQ."""

from __future__ import annotations

import torch


def _validate_quant_args(weight, nbits, groupsize):
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError("weight must be a rank-2 tensor")
    if not isinstance(nbits, int) or not 1 <= nbits <= 16:
        raise ValueError(f"nbits must be an integer in [1, 16], got {nbits!r}")
    if not isinstance(groupsize, int) or groupsize <= 0:
        raise ValueError(f"groupsize must be a positive integer, got {groupsize!r}")
    if weight.shape[1] % groupsize != 0:
        raise ValueError(
            f"weight input dimension {weight.shape[1]} is not divisible by "
            f"groupsize={groupsize}"
        )


def _normalize_clip_max(clip_max, grouped_weight):
    clip_max = torch.as_tensor(
        clip_max, device=grouped_weight.device, dtype=grouped_weight.dtype
    )
    expected = grouped_weight.shape[:2]
    if clip_max.ndim == 2:
        clip_max = clip_max.unsqueeze(-1)
    if clip_max.shape != (*expected, 1):
        raise ValueError(
            "clip_max must have shape [out_features, num_groups] or "
            f"[out_features, num_groups, 1], got {tuple(clip_max.shape)}; "
            f"expected {(*expected, 1)}"
        )
    if (clip_max < 0).any() or not torch.isfinite(clip_max).all():
        raise ValueError("clip_max must be finite and non-negative")
    return clip_max


@torch.no_grad()
def fake_quantize_grouped(grouped_weight, nbits):
    """Quantize a tensor whose final dimension is one quantization group.

    W1 deliberately follows GEMQ's historical binary convention: each group
    is represented by ``+/- mean(abs(weight))``.  W2 and above use the same
    zero-inclusive asymmetric min/max grid as GEMQ's MCMoe RTN path.
    """
    if grouped_weight.ndim < 2:
        raise ValueError("grouped_weight must have at least two dimensions")
    if not isinstance(nbits, int) or not 1 <= nbits <= 16:
        raise ValueError(f"nbits must be an integer in [1, 16], got {nbits!r}")

    if nbits == 1:
        magnitude = grouped_weight.abs().mean(dim=-1, keepdim=True)
        return torch.where(grouped_weight >= 0, magnitude, -magnitude)

    zero = torch.zeros(
        grouped_weight.shape[:-1] + (1,),
        dtype=grouped_weight.dtype,
        device=grouped_weight.device,
    )
    minimum = torch.minimum(grouped_weight.amin(dim=-1, keepdim=True), zero)
    maximum = torch.maximum(grouped_weight.amax(dim=-1, keepdim=True), zero)
    max_int = 2**nbits - 1
    scale = ((maximum - minimum) / max_int).clamp(min=1e-5)
    zero_point = torch.round(-minimum / scale).clamp_(0, max_int)
    quantized = torch.round(grouped_weight / scale) + zero_point
    quantized.clamp_(0, max_int)
    return (quantized - zero_point) * scale


@torch.no_grad()
def fake_quantize_weight(weight, nbits, groupsize=128, clip_max=None):
    """Return dequantized approximate weights without modifying ``weight``."""
    _validate_quant_args(weight, nbits, groupsize)
    original_shape = weight.shape
    grouped = weight.reshape(weight.shape[0], -1, groupsize)
    if clip_max is not None:
        clip_max = _normalize_clip_max(clip_max, grouped)
        grouped = grouped.clamp(-clip_max, clip_max)
    quantized = fake_quantize_grouped(grouped, nbits)
    if not torch.isfinite(quantized).all():
        raise RuntimeError("AWQ fake quantization produced non-finite weights")
    return quantized.reshape(original_shape)
