"""Expert-level PoP candidates and lazy descent at an exact bit budget."""

import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment


BITS = (0, 2, 3)


def feasible_prune_counts(num_layers, num_experts, max_pruned, target):
    return [
        p for p in range(max_pruned + 1)
        if 2 * num_layers * (num_experts - p)
        <= target <= 3 * num_layers * (num_experts - p)
    ]


def layer_assignment(costs, pruned, budget):
    """Solve the fixed-count PoP ILP as an equivalent slot assignment.

    There are p zero-bit slots, r three-bit slots, and E-p-r two-bit
    slots. Assigning every expert to exactly one slot enforces both equalities.
    """
    experts = len(costs)
    three_bit = budget - 2 * (experts - pruned)
    if not 0 <= pruned <= experts or not 0 <= three_bit <= experts - pruned:
        raise ValueError(f"Infeasible layer budget {budget} with {pruned} pruned experts.")
    slots = np.repeat(np.arange(len(BITS)), (pruned, experts - pruned - three_bit, three_bit))
    rows, columns = linear_sum_assignment(costs[:, slots])
    result = [0] * experts
    for expert, column in zip(rows, columns):
        result[expert] = BITS[slots[column]]
    return tuple(result)


def lazy_descent(num_layers, minimum, maximum, target, step, candidate, scorer):
    """Use equal-size moves per heap and rebuild when a finer step is needed."""
    if not minimum * num_layers <= target <= maximum * num_layers or step < 1:
        raise ValueError("Invalid search budget, layer range, or step.")
    if target == minimum * num_layers:
        for layer in range(num_layers):
            scorer.install(layer, candidate(layer, minimum))
        return [minimum] * num_layers, scorer.score(), 1, 0

    budgets = [maximum] * num_layers
    for layer in range(num_layers):
        scorer.install(layer, candidate(layer, maximum))

    score = scorer.score()
    evaluations = 1
    version = 0
    moves = 0
    remaining = maximum * num_layers - target
    while remaining:
        stride = min(step, remaining, max(budget - minimum for budget in budgets))
        heap = []

        def refresh(layer):
            nonlocal evaluations
            if budgets[layer] - stride < minimum:
                return
            old = candidate(layer, budgets[layer])
            proposed = candidate(layer, budgets[layer] - stride)
            try:
                scorer.install(layer, proposed)
                proposed_score = scorer.score()
                evaluations += 1
            finally:
                scorer.install(layer, old)
            heapq.heappush(heap, (proposed_score - score, layer, version, proposed_score))

        for layer in range(num_layers):
            refresh(layer)

        while heap and remaining >= stride:
            _, layer, stamp, proposed_score = heapq.heappop(heap)
            if stamp != version:
                refresh(layer)
                continue
            budgets[layer] -= stride
            scorer.install(layer, candidate(layer, budgets[layer]))
            score = proposed_score
            version += 1
            moves += 1
            remaining -= stride
            if remaining >= stride:
                refresh(layer)

    return budgets, score, evaluations, moves
