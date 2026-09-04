"""Online output-reconstruction targets for masked zero-bit experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_SOLVER_STEPS = 10
_SOLVER_CHUNK_SIZE = 256
_EPS = 1e-8


@dataclass
class OutputReconstructionRecord:
    """One MoE layer's differentiable loss and detached diagnostics."""

    loss_sum: torch.Tensor
    token_count: int
    active_count: int
    lost_mass_sum: torch.Tensor
    fit_error_sum: torch.Tensor
    layer_error_before_sum: torch.Tensor
    layer_error_after_sum: torch.Tensor
    relative_improvement_sum: torch.Tensor


@dataclass
class OutputReconstructionBatch:
    """Aggregated output-reconstruction result from a complete model forward."""

    loss: torch.Tensor
    token_count: int
    active_count: int
    lost_mass: float
    fit_error: float
    layer_error_before: float
    layer_error_after: float
    relative_improvement: float

    @property
    def pruned_hit_rate(self):
        return self.active_count / max(self.token_count, 1)


def project_probability_simplex(values):
    """Project every row onto the non-negative unit simplex."""
    if values.ndim != 2 or values.shape[-1] == 0:
        raise ValueError(f"values must have shape [tokens, experts], got {values.shape}")
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    cumulative = sorted_values.cumsum(dim=-1) - 1.0
    ranks = torch.arange(
        1, values.shape[-1] + 1, device=values.device, dtype=values.dtype
    ).unsqueeze(0)
    support = sorted_values - cumulative / ranks > 0
    rho = support.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    theta = (cumulative / ranks).gather(-1, rho)
    projected = (values - theta).clamp_min(0.0)
    return projected / projected.sum(dim=-1, keepdim=True).clamp_min(_EPS)


@torch.no_grad()
def solve_transfer_coefficients(
    candidate_outputs,
    missing_outputs,
    missing_mass,
    initial_probabilities,
):
    """Solve the batched convex reconstruction problem with projected GD.

    Args:
        candidate_outputs: ``[tokens, top_k, hidden]`` quantized expert outputs.
        missing_outputs: ``[tokens, hidden]`` aggregate zero-bit contribution.
        missing_mass: ``[tokens]`` probability mass to redistribute.
        initial_probabilities: ``[tokens, top_k]`` masked forward probabilities.
    """
    if candidate_outputs.ndim != 3:
        raise ValueError("candidate_outputs must have shape [tokens, top_k, hidden]")
    tokens, top_k, hidden = candidate_outputs.shape
    if missing_outputs.shape != (tokens, hidden):
        raise ValueError("missing_outputs shape does not match candidate_outputs")
    if missing_mass.shape != (tokens,):
        raise ValueError("missing_mass must have shape [tokens]")
    if initial_probabilities.shape != (tokens, top_k):
        raise ValueError("initial_probabilities must have shape [tokens, top_k]")

    all_alpha = []
    for start in range(0, tokens, _SOLVER_CHUNK_SIZE):
        end = min(start + _SOLVER_CHUNK_SIZE, tokens)
        expert_outputs = candidate_outputs[start:end].float()
        mass = missing_mass[start:end].float().unsqueeze(-1)
        target = missing_outputs[start:end].float() / mass.clamp_min(_EPS)
        beta = initial_probabilities[start:end].float()
        beta = project_probability_simplex(beta)
        gram = torch.einsum("tkd,tjd->tkj", expert_outputs, expert_outputs)
        correlation = torch.einsum("tkd,td->tk", expert_outputs, target)
        # ||F||_F^2 upper-bounds lambda_max(F^T F), so this is a stable
        # per-token step size without an eigendecomposition.
        step_size = expert_outputs.square().sum(dim=(-1, -2)).clamp_min(_EPS).reciprocal()

        for _ in range(_SOLVER_STEPS):
            gradient = torch.einsum("tkj,tj->tk", gram, beta) - correlation
            beta = project_probability_simplex(beta - step_size.unsqueeze(-1) * gradient)
        all_alpha.append(beta * mass)

    return torch.cat(all_alpha, dim=0)


@torch.no_grad()
def build_output_reconstruction_target(
    candidate_outputs,
    candidate_indices,
    masked_probabilities,
    unmasked_indices,
    unmasked_probabilities,
    missing_outputs,
    missing_mass,
    current_output,
):
    """Build sparse target P and detached reconstruction diagnostics."""
    alpha = solve_transfer_coefficients(
        candidate_outputs,
        missing_outputs,
        missing_mass,
        masked_probabilities,
    )
    matches = candidate_indices.unsqueeze(-1).eq(unmasked_indices.unsqueeze(1))
    base = (
        matches.to(unmasked_probabilities.dtype)
        * unmasked_probabilities.unsqueeze(1)
    ).sum(dim=-1)
    target = base.float() + alpha
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(_EPS)

    outputs = candidate_outputs.float()
    fitted_missing = torch.einsum("tkd,tk->td", outputs, alpha)
    target_output = torch.einsum("tkd,tk->td", outputs, target)
    base_output = torch.einsum("tkd,tk->td", outputs, base.float())
    reference_output = base_output + missing_outputs.float()

    fit_error = (fitted_missing - missing_outputs.float()).norm(dim=-1) / (
        missing_outputs.float().norm(dim=-1) + _EPS
    )
    reference_norm = reference_output.norm(dim=-1) + _EPS
    error_before = (current_output.float() - reference_output).norm(dim=-1) / reference_norm
    error_after = (target_output - reference_output).norm(dim=-1) / reference_norm
    relative_improvement = (error_before - error_after) / (error_before + _EPS)
    diagnostics = {
        "lost_mass_sum": missing_mass.float().sum(),
        "fit_error_sum": fit_error.sum(),
        "layer_error_before_sum": error_before.sum(),
        "layer_error_after_sum": error_after.sum(),
        "relative_improvement_sum": relative_improvement.sum(),
    }
    return target, diagnostics


def output_reconstruction_kl(target, masked_probabilities):
    """Return one KL(P || Q) value per active token."""
    target = target.detach().float()
    predicted = masked_probabilities.float()
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    terms = torch.where(
        target > 0,
        target * (target.clamp_min(_EPS).log() - predicted.clamp_min(_EPS).log()),
        torch.zeros_like(target),
    )
    return terms.sum(dim=-1)


def aggregate_output_reconstruction_records(records, loss_device=None):
    """Average KL over all layer-token pairs and diagnostics over active pairs."""
    if not records:
        raise ValueError("No output-reconstruction records were produced.")
    if loss_device is None:
        loss_device = records[0].loss_sum.device
    token_count = sum(record.token_count for record in records)
    active_count = sum(record.active_count for record in records)
    loss = torch.zeros((), device=loss_device, dtype=records[0].loss_sum.dtype)
    for record in records:
        loss = loss + record.loss_sum.to(loss_device)
    loss = loss / max(token_count, 1)

    def active_average(name):
        total = torch.zeros((), device=loss_device, dtype=torch.float)
        for record in records:
            total = total + getattr(record, name).to(
                device=loss_device, dtype=torch.float
            )
        return float((total / max(active_count, 1)).item())

    return OutputReconstructionBatch(
        loss=loss,
        token_count=token_count,
        active_count=active_count,
        lost_mass=active_average("lost_mass_sum"),
        fit_error=active_average("fit_error_sum"),
        layer_error_before=active_average("layer_error_before_sum"),
        layer_error_after=active_average("layer_error_after_sum"),
        relative_improvement=active_average("relative_improvement_sum"),
    )
