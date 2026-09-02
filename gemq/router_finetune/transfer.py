"""Probability-mass transfer utilities for pruned MoE routers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


ROUTER_STATS_VERSION = 1
TRANSFER_LOSS_TYPES = ("none", "prob_corr_kl")


@dataclass(frozen=True)
class FrozenRouterWeights:
    """CPU snapshot of one complete, pre-pruning router."""

    weight: torch.Tensor
    bias: torch.Tensor | None


def snapshot_reference_routers(model, model_name):
    """Clone complete pre-pruning routers without retaining the source model."""
    from gemq.utils.model_utils import get_router_modules

    snapshots = []
    for _name, router in get_router_modules(model, model_name):
        weight = router.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        bias = None
        if router.bias is not None:
            bias = router.bias.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        snapshots.append(
            FrozenRouterWeights(
                weight=weight.clone(),
                bias=None if bias is None else bias.clone(),
            )
        )
    if not snapshots:
        raise ValueError("The model has no router modules to snapshot.")
    return tuple(snapshots)


def _validate_probability_matrix(probabilities):
    if probabilities.ndim != 2:
        raise ValueError(
            f"Router probabilities must be two-dimensional, got {probabilities.shape}."
        )
    if probabilities.shape[0] == 0:
        raise ValueError("Router statistics require at least one valid token.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Router probabilities contain NaN or infinity.")


def router_statistics_from_probabilities(probabilities, eps=1e-12):
    """Compute stable Pearson statistics from dense router probabilities.

    Accumulation and covariance construction use float64 on CPU. Experts whose
    probabilities have zero variance receive zero off-diagonal correlations.
    """
    probabilities = probabilities.detach().to(device="cpu", dtype=torch.float64)
    _validate_probability_matrix(probabilities)
    count = probabilities.shape[0]
    mean = probabilities.mean(dim=0)
    centered = probabilities - mean
    covariance = centered.transpose(0, 1).matmul(centered) / float(count)
    variance = covariance.diagonal().clamp_min(0.0)
    std = variance.sqrt()
    denominator = std[:, None] * std[None, :]
    valid = std > eps
    valid_pairs = valid[:, None] & valid[None, :]
    correlation = torch.zeros_like(covariance)
    correlation[valid_pairs] = covariance[valid_pairs] / denominator[valid_pairs]
    correlation.clamp_(-1.0, 1.0)
    diagonal = torch.arange(correlation.shape[0])
    correlation[diagonal[valid], diagonal[valid]] = 1.0
    return {
        "count": torch.tensor(count, dtype=torch.long),
        "mean": mean.float(),
        "std": std.float(),
        "correlation": correlation.float(),
        "valid_variance": valid,
    }


def _router_statistics_from_moments(count, mean, centered_square_sum, eps):
    if count <= 0:
        raise ValueError("Router statistics require at least one valid token.")
    covariance = centered_square_sum / float(count)
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    variance = covariance.diagonal().clamp_min(0.0)
    std = variance.sqrt()
    denominator = std[:, None] * std[None, :]
    valid = std > eps
    valid_pairs = valid[:, None] & valid[None, :]
    correlation = torch.zeros_like(covariance)
    correlation[valid_pairs] = covariance[valid_pairs] / denominator[valid_pairs]
    correlation.clamp_(-1.0, 1.0)
    diagonal = torch.arange(correlation.shape[0])
    correlation[diagonal[valid], diagonal[valid]] = 1.0
    return {
        "count": torch.tensor(count, dtype=torch.long),
        "mean": mean.float(),
        "std": std.float(),
        "correlation": correlation.float(),
        "valid_variance": valid,
    }


def router_statistics_from_logits(
    router_logits, token_mask=None, eps=1e-12, chunk_size=16384
):
    """Compute dense-probability statistics from cached router logits."""
    if router_logits.ndim < 2:
        raise ValueError("Router logits must have an expert dimension.")
    if chunk_size <= 0:
        raise ValueError("Router-statistics chunk size must be positive.")
    num_experts = router_logits.shape[-1]
    flattened = router_logits.detach().reshape(-1, num_experts)
    mask = None
    if token_mask is not None:
        mask = token_mask.reshape(-1).to(device=flattened.device, dtype=torch.bool)
        if mask.numel() != flattened.shape[0]:
            raise ValueError("Token mask does not match cached router logits.")
    count = 0
    mean = torch.zeros(num_experts, dtype=torch.float64)
    centered_square_sum = torch.zeros(
        num_experts, num_experts, dtype=torch.float64
    )
    for start in range(0, flattened.shape[0], chunk_size):
        end = min(flattened.shape[0], start + chunk_size)
        probabilities = F.softmax(flattened[start:end].float(), dim=-1)
        if mask is not None:
            probabilities = probabilities[mask[start:end]]
        if probabilities.shape[0] == 0:
            continue
        probabilities = probabilities.to("cpu")
        chunk_count = probabilities.shape[0]
        chunk_mean = probabilities.sum(dim=0, dtype=torch.float64) / float(
            chunk_count
        )
        centered = probabilities - chunk_mean.float()
        chunk_square_sum = centered.transpose(0, 1).matmul(centered).double()

        if count == 0:
            mean = chunk_mean
            centered_square_sum = chunk_square_sum
            count = chunk_count
            continue
        combined_count = count + chunk_count
        delta = chunk_mean - mean
        centered_square_sum += chunk_square_sum + torch.outer(delta, delta) * (
            float(count * chunk_count) / float(combined_count)
        )
        mean += delta * (float(chunk_count) / float(combined_count))
        count = combined_count
    return _router_statistics_from_moments(count, mean, centered_square_sum, eps)


def stack_router_statistics(per_layer_statistics):
    if not per_layer_statistics:
        raise ValueError("No per-layer router statistics were provided.")
    return {
        "version": ROUTER_STATS_VERSION,
        "similarity": "pearson",
        "router_correlation": torch.stack(
            [stats["correlation"] for stats in per_layer_statistics]
        ),
        "router_mean": torch.stack([stats["mean"] for stats in per_layer_statistics]),
        "router_std": torch.stack([stats["std"] for stats in per_layer_statistics]),
        "valid_variance": torch.stack(
            [stats["valid_variance"] for stats in per_layer_statistics]
        ),
        "token_count": torch.stack([stats["count"] for stats in per_layer_statistics]),
    }


def router_statistics_from_cached_layers(router_logits, token_mask=None):
    return stack_router_statistics(
        [router_statistics_from_logits(logits, token_mask) for logits in router_logits]
    )


def validate_router_statistics(stats, expected_layers=None, expected_experts=None):
    if stats is None:
        raise ValueError("Router correlation statistics are missing.")
    if stats.get("version") != ROUTER_STATS_VERSION:
        raise ValueError(f"Unsupported router statistics version: {stats.get('version')!r}")
    if stats.get("similarity") != "pearson":
        raise ValueError(f"Unsupported router similarity: {stats.get('similarity')!r}")
    correlation = stats.get("router_correlation")
    if not torch.is_tensor(correlation) or correlation.ndim != 3:
        raise ValueError("router_correlation must have shape [layers, experts, experts].")
    if correlation.shape[-1] != correlation.shape[-2]:
        raise ValueError("Router correlation matrices must be square.")
    if expected_layers is not None and correlation.shape[0] != expected_layers:
        raise ValueError(
            f"Router statistics have {correlation.shape[0]} layers; expected {expected_layers}."
        )
    if expected_experts is not None and correlation.shape[-1] != expected_experts:
        raise ValueError(
            f"Router statistics have {correlation.shape[-1]} experts; expected {expected_experts}."
        )
    if not torch.isfinite(correlation).all():
        raise ValueError("Router correlation statistics contain NaN or infinity.")


def _layer_pruning_ids(pruning_metadata, layer_idx):
    if pruning_metadata is None:
        raise ValueError("Probability transfer requires physical-pruning metadata.")
    layer_metadata = pruning_metadata.get("layers", {}).get(str(layer_idx))
    if layer_metadata is None:
        raise ValueError(f"Pruning metadata is missing layer {layer_idx}.")
    kept = tuple(int(value) for value in layer_metadata.get("kept_old_ids", []))
    pruned = tuple(int(value) for value in layer_metadata.get("pruned_old_ids", []))
    if not kept:
        raise ValueError(f"Layer {layer_idx} keeps no experts.")
    if not pruned:
        raise ValueError(f"Layer {layer_idx} prunes no experts; transfer is undefined.")
    if len(set(kept)) != len(kept) or len(set(pruned)) != len(pruned):
        raise ValueError(f"Layer {layer_idx} contains duplicate expert IDs.")
    if set(kept) & set(pruned):
        raise ValueError(f"Layer {layer_idx} has overlapping kept and pruned expert IDs.")
    return kept, pruned


def build_transfer_matrices(router_stats, pruning_metadata, temperature):
    """Build allocation-specific soft transfer matrices in student expert order."""
    if temperature <= 0.0:
        raise ValueError("Transfer temperature must be positive.")
    validate_router_statistics(router_stats)
    correlation = router_stats["router_correlation"].float()
    matrices = []
    kept_by_layer = []
    pruned_by_layer = []
    effective_support = []
    for layer_idx in range(correlation.shape[0]):
        kept, pruned = _layer_pruning_ids(pruning_metadata, layer_idx)
        if set(kept) | set(pruned) != set(range(correlation.shape[-1])):
            raise ValueError(
                f"Layer {layer_idx} kept/pruned IDs do not partition all original experts."
            )
        if (
            max((*kept, *pruned)) >= correlation.shape[-1]
            or min((*kept, *pruned)) < 0
        ):
            raise ValueError(f"Layer {layer_idx} pruning IDs exceed the correlation matrix.")
        pruned_index = torch.tensor(pruned, dtype=torch.long)
        kept_index = torch.tensor(kept, dtype=torch.long)
        selected = correlation[layer_idx].index_select(0, pruned_index).index_select(
            1, kept_index
        )
        matrix = F.softmax(selected / float(temperature), dim=-1)
        if not torch.isfinite(matrix).all():
            raise ValueError(f"Layer {layer_idx} transfer matrix contains NaN or infinity.")
        if not torch.allclose(
            matrix.sum(dim=-1), torch.ones(matrix.shape[0]), atol=1e-6, rtol=1e-6
        ):
            raise ValueError(f"Layer {layer_idx} transfer rows do not sum to one.")
        entropy = -(matrix * matrix.clamp_min(1e-30).log()).sum(dim=-1)
        effective_support.append(entropy.exp())
        matrices.append(matrix.contiguous())
        kept_by_layer.append(kept)
        pruned_by_layer.append(pruned)
    return (
        tuple(matrices),
        tuple(kept_by_layer),
        tuple(pruned_by_layer),
        torch.cat(effective_support).mean().item(),
    )


def transfer_reference_probabilities(
    reference_logits, transfer_matrix, kept_ids, pruned_ids
):
    """Move each pruned expert's dense probability mass to surviving experts."""
    if reference_logits.shape[-1] != len(kept_ids) + len(pruned_ids):
        raise ValueError(
            "Reference router width does not match the kept/pruned expert partition."
        )
    probabilities = F.softmax(reference_logits.float(), dim=-1)
    kept_index = torch.as_tensor(kept_ids, device=probabilities.device, dtype=torch.long)
    pruned_index = torch.as_tensor(pruned_ids, device=probabilities.device, dtype=torch.long)
    kept_probabilities = probabilities.index_select(-1, kept_index)
    pruned_probabilities = probabilities.index_select(-1, pruned_index)
    matrix = transfer_matrix.to(device=probabilities.device, dtype=probabilities.dtype)
    target = kept_probabilities + pruned_probabilities.matmul(matrix)
    if not torch.isfinite(target).all():
        raise ValueError("Transferred router probabilities contain NaN or infinity.")
    return target


def transferred_router_kl(student_logits, target_probabilities, token_mask=None):
    """KL(P || Q), where P is fixed transfer target and Q is the student router."""
    if student_logits.shape != target_probabilities.shape:
        raise ValueError(
            f"Student/target router shapes differ: {student_logits.shape} vs "
            f"{target_probabilities.shape}."
        )
    num_experts = student_logits.shape[-1]
    student_log_probabilities = F.log_softmax(
        student_logits.reshape(-1, num_experts).float(), dim=-1
    )
    target = target_probabilities.detach().reshape(-1, num_experts).float()
    per_token = torch.sum(
        target * (target.clamp_min(1e-30).log() - student_log_probabilities), dim=-1
    )
    if token_mask is None:
        return per_token.mean()
    mask = token_mask.reshape(-1).to(device=per_token.device, dtype=per_token.dtype)
    if mask.numel() != per_token.numel():
        raise ValueError("Token mask does not match the number of router tokens.")
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def linear_transfer_weight(initial_weight, anneal_ratio, step, total_steps):
    if initial_weight < 0.0:
        raise ValueError("Transfer weight must be non-negative.")
    if not 0.0 < anneal_ratio <= 1.0:
        raise ValueError("Transfer anneal ratio must be in (0, 1].")
    if total_steps <= 0:
        raise ValueError("Total training steps must be positive.")
    cutoff = anneal_ratio * total_steps
    return float(initial_weight) * max(1.0 - float(step) / cutoff, 0.0)
