import pytest

torch = pytest.importorskip("torch")

from gemq.router_finetune.transfer import (  # noqa: E402
    ROUTER_STATS_VERSION,
    build_transfer_matrices,
    linear_transfer_weight,
    router_statistics_from_probabilities,
    stack_router_statistics,
    transfer_reference_probabilities,
    transferred_router_kl,
)


def _pruning_metadata():
    return {
        "layers": {
            "0": {
                "kept_old_ids": [0, 2, 3],
                "pruned_old_ids": [1],
                "old_to_new": {"0": 0, "2": 1, "3": 2},
            }
        }
    }


def test_zero_variance_expert_has_finite_neutral_correlations():
    probabilities = torch.tensor(
        [
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
            [0.2, 0.4, 0.4],
        ]
    )
    stats = router_statistics_from_probabilities(probabilities)

    assert not stats["valid_variance"][0]
    assert torch.equal(stats["correlation"][0], torch.zeros(3))
    assert torch.isfinite(stats["correlation"]).all()


def test_transfer_matrix_uses_original_ids_and_is_row_stochastic():
    correlation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 1.0, 0.8, 0.3],
            [0.0, 0.8, 1.0, 0.0],
            [0.0, 0.3, 0.0, 1.0],
        ]
    )
    stats = {
        "version": ROUTER_STATS_VERSION,
        "similarity": "pearson",
        "router_correlation": correlation.unsqueeze(0),
    }
    matrices, kept, pruned, effective_support = build_transfer_matrices(
        stats, _pruning_metadata(), temperature=0.2
    )

    assert kept == ((0, 2, 3),)
    assert pruned == ((1,),)
    assert matrices[0].shape == (1, 3)
    assert matrices[0][0].argmax().item() == 1  # original expert 2 -> student expert 1
    assert matrices[0].sum().item() == pytest.approx(1.0)
    assert 1.0 <= effective_support <= 3.0


def test_temperature_controls_softness():
    correlation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.1, 1.0, 0.8, 0.3],
            [0.0, 0.8, 1.0, 0.0],
            [0.0, 0.3, 0.0, 1.0],
        ]
    )
    stats = {
        "version": ROUTER_STATS_VERSION,
        "similarity": "pearson",
        "router_correlation": correlation.unsqueeze(0),
    }
    _, _, _, low_temperature_support = build_transfer_matrices(
        stats, _pruning_metadata(), temperature=0.05
    )
    _, _, _, high_temperature_support = build_transfer_matrices(
        stats, _pruning_metadata(), temperature=1.0
    )
    assert low_temperature_support < high_temperature_support


def test_probability_transfer_conserves_all_mass():
    reference_logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]])
    transfer_matrix = torch.tensor([[0.2, 0.3, 0.5]])
    target = transfer_reference_probabilities(
        reference_logits,
        transfer_matrix,
        kept_ids=(0, 2, 3),
        pruned_ids=(1,),
    )
    assert target.shape == (1, 1, 3)
    assert target.sum(dim=-1).item() == pytest.approx(1.0, abs=1e-6)


def test_transfer_kl_only_backpropagates_to_student_logits():
    student = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    loss = transferred_router_kl(student, target)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert target.grad is None


def test_linear_transfer_weight_reaches_zero_at_anneal_cutoff():
    assert linear_transfer_weight(2.0, 0.2, 0, 100) == pytest.approx(2.0)
    assert linear_transfer_weight(2.0, 0.2, 10, 100) == pytest.approx(1.0)
    assert linear_transfer_weight(2.0, 0.2, 20, 100) == pytest.approx(0.0)
    assert linear_transfer_weight(2.0, 0.2, 99, 100) == pytest.approx(0.0)


def test_stacked_router_statistics_have_versioned_shape():
    first = router_statistics_from_probabilities(torch.softmax(torch.randn(8, 4), -1))
    second = router_statistics_from_probabilities(torch.softmax(torch.randn(8, 4), -1))
    stats = stack_router_statistics([first, second])
    assert stats["version"] == ROUTER_STATS_VERSION
    assert stats["router_correlation"].shape == (2, 4, 4)
