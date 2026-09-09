import torch
from types import SimpleNamespace

from gemq.allocate_bits import _auto_save_path
from gemq.allocation.ilp_solvers import ExpertCostSolver


def test_expert_cost_solver_global_budget_and_equal_pruning(tmp_path):
    # The artifact contains {0,1,2,3}, while the IP selects only {0,2,3}.
    # Budget 8 over 2x3 experts forces exactly one zero-bit expert per layer.
    # Costs make a different expert preferable in each.
    costs = torch.tensor(
        [
            [[0.1, 0.0, 10.0, 11.0], [9.0, 0.0, 1.0, 2.0], [8.0, 0.0, 1.0, 2.0]],
            [[8.0, 0.0, 1.0, 2.0], [0.1, 0.0, 10.0, 11.0], [9.0, 0.0, 1.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    path = tmp_path / "costs.pt"
    torch.save(
        {
            "costs": costs,
            "counts": torch.ones(2, 3, dtype=torch.long),
            "candidate_bits": torch.tensor([0, 1, 2, 3]),
            "metadata": {"format_version": 2},
        },
        path,
    )
    solver = ExpertCostSolver(
        path,
        x_space=[0, 2, 3],
        max_prune_ratio=0.34,
        top_k=1,
        backend="highs",
    )
    allocation = solver.solve_all(total_bits=8)

    assert sum(bit == 0 for bit in allocation[0].values()) == 1
    assert sum(bit == 0 for bit in allocation[1].values()) == 1
    assert allocation[0][0] == 0
    assert allocation[1][1] == 0
    assert all(
        bit != 1 for layer in allocation.values() for bit in layer.values()
    )
    assert sum(bit for layer in allocation.values() for bit in layer.values()) <= 8


def test_expert_cost_solver_rejects_infeasible_prune_cap(tmp_path):
    path = tmp_path / "costs.pt"
    torch.save(
        {
            "costs": torch.ones(2, 3, 2, dtype=torch.float64),
            "counts": torch.ones(2, 3, dtype=torch.long),
            "candidate_bits": torch.tensor([0, 2]),
        },
        path,
    )
    solver = ExpertCostSolver(
        path, x_space=[0, 2], max_prune_ratio=0.0, top_k=1, backend="highs"
    )
    try:
        solver.solve_all(total_bits=10)
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("Expected an infeasible-budget error")


def test_awq_allocation_filename_is_separate_without_renaming_rtn_outputs():
    args = SimpleNamespace(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        allocation_metric="expert_cost",
        bit_budget=2.0,
        max_prune_ratio=0.1,
    )
    source_path = "cache/ExpertCosts_c4-N128-L2048-Seed0.pt"
    awq_solver = SimpleNamespace(
        artifact_metadata={
            "quantizer_id": "awq",
            "context_mode": "uniform_bit",
            "average_bits": 2,
        }
    )
    rtn_solver = SimpleNamespace(
        artifact_metadata={
            "quantizer_id": "rtn",
            "context_mode": "uniform_bit",
            "average_bits": 2,
        }
    )

    awq_path = _auto_save_path(args, source_path, [0, 2, 3], awq_solver)
    rtn_path = _auto_save_path(args, source_path, [0, 2, 3], rtn_solver)

    assert "Quant-awq" in awq_path
    assert "B0,2,3" in awq_path
    assert "Quant-rtn" not in rtn_path
    assert "Metric-expert_cost-Ctxuniform2" in rtn_path
