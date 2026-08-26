import torch

from gemq.allocation.ilp_solvers import ExpertCostSolver


def test_expert_cost_solver_global_budget_and_equal_pruning(tmp_path):
    # Budget 8 over 2x3 experts with candidates {0,2,3} forces exactly one
    # zero-bit expert per layer.  Costs make a different expert preferable in each.
    costs = torch.tensor(
        [
            [[0.1, 10.0, 11.0], [9.0, 1.0, 2.0], [8.0, 1.0, 2.0]],
            [[8.0, 1.0, 2.0], [0.1, 10.0, 11.0], [9.0, 1.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    path = tmp_path / "costs.pt"
    torch.save(
        {
            "costs": costs,
            "counts": torch.ones(2, 3, dtype=torch.long),
            "candidate_bits": torch.tensor([0, 2, 3]),
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
