"""Validation and compatibility helpers for expert-cost artifacts."""

from __future__ import annotations

from pathlib import Path

import torch


def impute_zero_frequency_costs(costs, counts):
    """Fill count-zero entries with the same-layer, same-bit observed mean.

    Costs belonging to an observed expert must always be finite.  A layer with no
    observed experts has no defensible mean and is rejected instead of silently
    manufacturing optimization coefficients.
    """
    costs = torch.as_tensor(costs)
    counts = torch.as_tensor(counts)
    if costs.ndim != 3:
        raise ValueError(f"costs must have shape [layers, experts, bits], got {tuple(costs.shape)}")
    if counts.shape != costs.shape[:2]:
        raise ValueError(
            f"counts must have shape {tuple(costs.shape[:2])}, got {tuple(counts.shape)}"
        )
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    if not costs.is_floating_point():
        costs = costs.to(torch.float64)

    observed = counts > 0
    bad_observed = observed.unsqueeze(-1) & ~torch.isfinite(costs)
    if bad_observed.any():
        locations = bad_observed.nonzero()[:8].tolist()
        raise ValueError(
            "Observed experts contain non-finite costs; first locations "
            f"[layer, expert, bit]={locations}"
        )
    negative_observed = observed.unsqueeze(-1) & (costs < 0)
    if negative_observed.any():
        locations = negative_observed.nonzero()[:8].tolist()
        raise ValueError(
            "Observed experts contain negative costs; first locations "
            f"[layer, expert, bit]={locations}"
        )

    filled = costs.clone()
    for layer_idx in range(costs.shape[0]):
        layer_observed = observed[layer_idx]
        if not layer_observed.any():
            raise ValueError(
                f"Layer {layer_idx} has no selected experts in the calibration set; "
                "same-layer imputation is undefined."
            )
        layer_means = costs[layer_idx, layer_observed].mean(dim=0)
        if not torch.isfinite(layer_means).all():
            raise ValueError(f"Layer {layer_idx} produced non-finite imputation means.")
        filled[layer_idx, ~layer_observed] = layer_means

    if not torch.isfinite(filled).all():
        raise ValueError("Filled expert costs still contain non-finite values.")
    return filled, ~observed


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_expert_cost_artifact(path):
    """Load both format-v1 (raw NaNs) and format-v2 (raw + filled) artifacts."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = _torch_load_cpu(path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expert-cost artifact must be a dict, got {type(artifact)!r}")
    required = {"costs", "counts", "candidate_bits"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"Expert-cost artifact is missing keys: {sorted(missing)}")

    stored_costs = torch.as_tensor(artifact["costs"]).to(device="cpu")
    raw_costs = torch.as_tensor(artifact.get("raw_costs", stored_costs)).to(device="cpu")
    counts = torch.as_tensor(artifact["counts"]).to(device="cpu", dtype=torch.long)
    candidate_bits = [int(bit) for bit in torch.as_tensor(artifact["candidate_bits"]).tolist()]

    if len(candidate_bits) != raw_costs.shape[-1]:
        raise ValueError(
            f"candidate_bits has {len(candidate_bits)} entries but costs has "
            f"{raw_costs.shape[-1]} columns"
        )
    if len(set(candidate_bits)) != len(candidate_bits):
        raise ValueError(f"candidate_bits contains duplicates: {candidate_bits}")
    if any(bit < 0 for bit in candidate_bits):
        raise ValueError(f"candidate_bits must be non-negative: {candidate_bits}")

    filled_costs, imputed_mask = impute_zero_frequency_costs(raw_costs, counts)
    result = dict(artifact)
    result.update(
        {
            "costs": filled_costs,
            "raw_costs": raw_costs,
            "counts": counts,
            "imputed_mask": imputed_mask,
            "candidate_bits": torch.tensor(candidate_bits, dtype=torch.int64),
        }
    )
    return result


def select_candidate_costs(artifact, requested_bits):
    """Select cost columns by bit value, preserving the requested order."""
    requested_bits = [int(bit) for bit in requested_bits]
    if not requested_bits or len(set(requested_bits)) != len(requested_bits):
        raise ValueError("requested_bits must be a non-empty list without duplicates")
    available_bits = [int(bit) for bit in artifact["candidate_bits"].tolist()]
    missing = [bit for bit in requested_bits if bit not in available_bits]
    if missing:
        raise ValueError(
            f"Requested bits {missing} are absent from the expert-cost artifact; "
            f"available bits are {available_bits}."
        )
    columns = [available_bits.index(bit) for bit in requested_bits]
    return artifact["costs"][:, :, columns].contiguous()
