import json
import pickle
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from gemq import allocate_pmq_bits
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


def test_pmq_exact_budget_is_enforced_inside_the_ilp():
    counts, weights, quant_loss = _toy_stats(num_layers=1)
    for expert_loss in quant_loss[0].values():
        expert_loss.update({1: 1.0, 2: 10.0, 3: 100.0})

    exact_solver = PMQSolver(counts, weights, quant_loss, backend="highs")
    exact_allocation = exact_solver.solve(bits_per_expert=2.0)
    assert sum(exact_allocation[0].values()) == 8

    source_solver = PMQSolver(counts, weights, quant_loss, backend="highs")
    source_allocation = source_solver.solve(
        bits_per_expert=2.0, require_full_budget=False
    )
    assert sum(source_allocation[0].values()) == 7


def test_pmq_rejects_non_source_candidate_bits():
    counts, weights, quant_loss = _toy_stats(num_layers=1)

    with pytest.raises(ValueError, match="candidate_bits=\\(1, 2, 3\\)"):
        PMQSolver(counts, weights, quant_loss, candidate_bits=(0, 2, 3))


def test_pmq_rejects_mismatched_statistics():
    counts, weights, quant_loss = _toy_stats()
    del weights[1]

    with pytest.raises(ValueError, match="same layer IDs"):
        PMQSolver(counts, weights, quant_loss)


def test_pmq_solves_each_layer_at_2p5_bpe_using_only_two_and_three_bits():
    counts, weights, quant_loss = _toy_stats(num_experts=128)
    solver = PMQSolver(counts, weights, quant_loss, candidate_bits=(2, 3))

    allocation = solver.solve(bits_per_expert=2.5)

    assert solver.statistics_candidate_bits == [1, 2, 3]
    assert solver.used_bits_per_layer == {0: 320, 1: 320}
    for layer_bits in allocation.values():
        assert Counter(layer_bits.values()) == {2: 64, 3: 64}


def test_pmq_two_and_three_bit_feasibility_reserves_one_expert_at_each_bit():
    solver = PMQSolver(*_toy_stats(num_layers=1), candidate_bits=(2, 3))
    with pytest.raises(ValueError, match=r"expected \[9, 11\]"):
        solver.solve(bits_per_expert=2.0)

    allocation = solver.solve(bits_per_expert=2.25)
    assert Counter(allocation[0].values()) == {2: 3, 3: 1}


def test_pmq_rejects_missing_candidate_loss_for_a_single_expert():
    counts, weights, quant_loss = _toy_stats(num_layers=1)
    del quant_loss[0][0][3]
    with pytest.raises(ValueError, match=r"quant_loss\[0\]\[0\] must contain bits \(2, 3\)"):
        PMQSolver(counts, weights, quant_loss, candidate_bits=(2, 3))


def test_pmq_cli_reuses_statistics_with_extra_bit_width(tmp_path, monkeypatch):
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    monkeypatch.setattr(allocate_pmq_bits, "get_model_info", lambda _: SimpleNamespace(
        first_k_dense_layers=0, num_layers=1, num_routed_experts_per_layer=4,
        num_shared_experts_per_layer=0,
    ))
    stat_names = allocate_pmq_bits.STAT_FILENAMES
    for stat, value in zip(("act_counts", "act_weights", "quant_loss"), _toy_stats(num_layers=1)):
        (tmp_path / stat_names[stat]).write_bytes(pickle.dumps(value))
    metadata_path = tmp_path / "metadata.json"
    stats_metadata = {"model_name": model_name, "candidate_bits": [1, 2, 3]}
    metadata_path.write_text(json.dumps(stats_metadata), encoding="utf-8")
    output = tmp_path / "E2.5_B2,3.pkl"
    args = allocate_pmq_bits.parse_args([
        "--model_name", model_name,
        "--stats_dir", str(tmp_path),
        "--save_path", str(output),
        "--bits_per_expert", "2.5",
        "--bit_candidates", "2,3",
    ])

    allocation, sidecar = allocate_pmq_bits.run(args)

    assert Counter(allocation[0].values()) == {2: 2, 3: 2}
    assert output.is_file()
    assert sidecar["statistics_candidate_bits"] == [1, 2, 3]
    assert sidecar["candidate_bits"] == [2, 3]
    assert sidecar["actual_bits_per_expert"] == 2.5
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == stats_metadata
