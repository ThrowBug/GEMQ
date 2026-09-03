import math

import pytest
import torch

from gemq.router_finetune.output_reconstruction import (
    build_output_reconstruction_target,
    cosine_transfer_weight,
    output_reconstruction_kl,
    project_probability_simplex,
    solve_transfer_coefficients,
)


def test_cosine_transfer_weight_uses_the_full_run():
    values = [cosine_transfer_weight(2.0, step, 5) for step in range(5)]
    assert values[0] == pytest.approx(2.0)
    assert values[-1] == pytest.approx(0.0)
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert cosine_transfer_weight(2.0, 0, 1) == pytest.approx(2.0)


def test_simplex_projection_is_nonnegative_and_normalized():
    projected = project_probability_simplex(
        torch.tensor([[2.0, -1.0, 0.5], [-3.0, -2.0, -1.0]])
    )
    assert torch.all(projected >= 0)
    torch.testing.assert_close(projected.sum(dim=-1), torch.ones(2))


def test_solver_recovers_an_exact_convex_reconstruction():
    candidate_outputs = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
    )
    missing_mass = torch.tensor([0.4, 0.6])
    expected_alpha = torch.tensor([[0.1, 0.3], [0.45, 0.15]])
    missing_outputs = torch.einsum(
        "tkd,tk->td", candidate_outputs, expected_alpha
    )
    alpha = solve_transfer_coefficients(
        candidate_outputs,
        missing_outputs,
        missing_mass,
        expected_alpha / missing_mass.unsqueeze(-1),
    )
    assert torch.all(alpha >= 0)
    torch.testing.assert_close(alpha.sum(dim=-1), missing_mass)
    torch.testing.assert_close(alpha, expected_alpha, atol=1e-5, rtol=1e-5)


def test_target_is_sparse_normalized_and_kl_updates_only_prediction():
    candidate_outputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    candidate_indices = torch.tensor([[0, 2]])
    masked_probabilities = torch.tensor([[0.7, 0.3]], requires_grad=True)
    unmasked_indices = torch.tensor([[0, 1]])
    unmasked_probabilities = torch.tensor([[0.6, 0.4]])
    missing_outputs = torch.tensor([[0.1, 0.3]])
    missing_mass = torch.tensor([0.4])
    current_output = torch.einsum(
        "tkd,tk->td", candidate_outputs, masked_probabilities.detach()
    )

    target, diagnostics = build_output_reconstruction_target(
        candidate_outputs,
        candidate_indices,
        masked_probabilities.detach(),
        unmasked_indices,
        unmasked_probabilities,
        missing_outputs,
        missing_mass,
        current_output,
    )
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1))
    assert not target.requires_grad
    assert math.isfinite(float(diagnostics["fit_error_sum"]))

    loss = output_reconstruction_kl(target, masked_probabilities).mean()
    loss.backward()
    assert masked_probabilities.grad is not None
    assert torch.isfinite(masked_probabilities.grad).all()
