import numpy as np
import pytest

from gemq.allocation.pmq_solver import PMQSolver


def _toy_stats(num_layers=2, num_experts=4):
    counts = {
        layer: np.arange(1, num_experts + 1, dtype=np.float64)
        for layer in range(num_layers)
    }
    weights = {
        layer: np.arange(num_experts, 0, -1, dtype=np.float64)
        for layer in range(num_layers)
    }
    quant_loss = {
        layer: {
            expert: {
                1: float(30 + expert),
                2: float(10 + expert),
                3: float(1 + expert),
            }
            for expert in range(num_experts)
        }
        for layer in range(num_layers)
    }
    return counts, weights, quant_loss


def test_pmq_objective_matches_released_mcmoe_formula():
    counts, weights, quant_loss = _toy_stats(num_layers=1)
    solver = PMQSolver(counts, weights, quant_loss, alpha=1, beta=1.5, gama=99)

    normalized_count = counts[0][0] / counts[0].sum()
    normalized_weight = weights[0][0] / weights[0].sum()
    expected = (
        normalized_count
        * normalized_weight**1.5
        * quant_loss[0][0][1]
        * 1000
    )
    assert solver.coefficients[0][0, 0] == pytest.approx(expected)


def test_pmq_solves_each_layer_at_exact_two_bpe_without_pruning():
    counts, weights, quant_loss = _toy_stats()
    solver = PMQSolver(counts, weights, quant_loss, backend="highs")

    allocation = solver.solve(bits_per_expert=2.0)

    assert set(allocation) == {0, 1}
    for layer_allocation in allocation.values():
        assert set(layer_allocation) == {0, 1, 2, 3}
        assert set(layer_allocation.values()) <= {1, 2, 3}
        assert sum(layer_allocation.values()) == 8
        assert 2 in layer_allocation.values()
        assert 3 in layer_allocation.values()
    assert solver.used_bits_per_layer == {0: 8, 1: 8}


def test_pmq_rejects_non_source_candidate_bits():
    counts, weights, quant_loss = _toy_stats(num_layers=1)

    with pytest.raises(ValueError, match="candidate_bits=\\(1, 2, 3\\)"):
        PMQSolver(counts, weights, quant_loss, candidate_bits=(0, 2, 3))


def test_pmq_rejects_mismatched_statistics():
    counts, weights, quant_loss = _toy_stats()
    del weights[1]

    with pytest.raises(ValueError, match="same layer IDs"):
        PMQSolver(counts, weights, quant_loss)

