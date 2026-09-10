import math
from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


PMQ_CANDIDATE_BITS = (1, 2, 3)


def _as_nonnegative_vector(value, *, label, expected_size=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_size is not None and vector.size != expected_size:
        raise ValueError(
            f"{label} has {vector.size} entries; expected {expected_size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(vector < 0):
        raise ValueError(f"{label} contains negative values.")
    return vector


class PMQSolver:
    """Source-faithful MC-MoE/PMQ per-layer bit allocator.

    MC-MoE solves one independent ILP for every MoE layer. Its released source
    normalizes activation counts and routing weights along the expert dimension,
    multiplies the objective by 1000, and (despite exposing ``gama``) raises the
    quantization loss to ``alpha``. This implementation deliberately preserves
    that executable behavior.
    """

    def __init__(
        self,
        act_counts,
        act_weights,
        quant_loss,
        *,
        candidate_bits=PMQ_CANDIDATE_BITS,
        alpha=1.0,
        beta=1.5,
        gama=2.0,
        backend="highs",
    ):
        if backend not in ("highs", "gurobi"):
            raise ValueError(f"Unknown ILP backend: {backend!r}.")
        if tuple(candidate_bits) != PMQ_CANDIDATE_BITS:
            raise ValueError(
                "Source-faithful PMQ requires candidate_bits=(1, 2, 3); "
                f"got {tuple(candidate_bits)!r}."
            )
        for name, value in (("alpha", alpha), ("beta", beta), ("gama", gama)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")

        self.act_counts = act_counts
        self.act_weights = act_weights
        self.quant_loss = quant_loss
        self.candidate_bits = tuple(candidate_bits)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gama = float(gama)
        self.backend = backend

        self.layer_ids, self.num_experts = self._validate_stats()
        self.coefficients = self._build_coefficients()
        self.last_objective = None
        self.used_bits_per_layer = None

    def _validate_stats(self):
        for name, value in (
            ("act_counts", self.act_counts),
            ("act_weights", self.act_weights),
            ("quant_loss", self.quant_loss),
        ):
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty layer mapping.")

        layer_ids = sorted(self.act_counts)
        expected_layers = set(layer_ids)
        if set(self.act_weights) != expected_layers or set(self.quant_loss) != expected_layers:
            raise ValueError(
                "PMQ statistics do not cover the same layer IDs: "
                f"counts={sorted(self.act_counts)}, "
                f"weights={sorted(self.act_weights)}, "
                f"loss={sorted(self.quant_loss)}."
            )
        if not all(isinstance(layer_id, int) for layer_id in layer_ids):
            raise ValueError(f"Layer IDs must be integers, got {layer_ids!r}.")

        first_counts = _as_nonnegative_vector(
            self.act_counts[layer_ids[0]], label=f"act_counts[{layer_ids[0]}]"
        )
        num_experts = first_counts.size
        if num_experts == 0:
            raise ValueError("PMQ statistics contain no experts.")
        if num_experts < 2:
            raise ValueError(
                "PMQ needs at least two experts to assign both a 2-bit and a "
                "3-bit expert in every layer."
            )

        expected_experts = set(range(num_experts))
        expected_bits = set(self.candidate_bits)
        for layer_id in layer_ids:
            counts = _as_nonnegative_vector(
                self.act_counts[layer_id],
                label=f"act_counts[{layer_id}]",
                expected_size=num_experts,
            )
            weights = _as_nonnegative_vector(
                self.act_weights[layer_id],
                label=f"act_weights[{layer_id}]",
                expected_size=num_experts,
            )
            if counts.sum() <= 0:
                raise ValueError(f"act_counts[{layer_id}] has a zero sum.")
            if weights.sum() <= 0:
                raise ValueError(f"act_weights[{layer_id}] has a zero sum.")

            layer_loss = self.quant_loss[layer_id]
            if not isinstance(layer_loss, Mapping) or set(layer_loss) != expected_experts:
                raise ValueError(
                    f"quant_loss[{layer_id}] must contain expert IDs "
                    f"0..{num_experts - 1}."
                )
            for expert_id in range(num_experts):
                expert_loss = layer_loss[expert_id]
                if not isinstance(expert_loss, Mapping) or set(expert_loss) != expected_bits:
                    raise ValueError(
                        f"quant_loss[{layer_id}][{expert_id}] must contain bits "
                        f"{self.candidate_bits}."
                    )
                losses = _as_nonnegative_vector(
                    [expert_loss[bit] for bit in self.candidate_bits],
                    label=f"quant_loss[{layer_id}][{expert_id}]",
                    expected_size=len(self.candidate_bits),
                )
                if not np.all(np.isfinite(losses)):
                    raise ValueError(
                        f"quant_loss[{layer_id}][{expert_id}] contains non-finite values."
                    )

        return layer_ids, num_experts

    def _build_coefficients(self):
        coefficients = {}
        for layer_id in self.layer_ids:
            counts = _as_nonnegative_vector(self.act_counts[layer_id], label="counts")
            weights = _as_nonnegative_vector(self.act_weights[layer_id], label="weights")
            counts = counts / counts.sum()
            weights = weights / weights.sum()
            significance = np.power(counts, self.alpha) * np.power(weights, self.beta)

            layer_coefficients = np.empty(
                (self.num_experts, len(self.candidate_bits)), dtype=np.float64
            )
            for expert_id in range(self.num_experts):
                for bit_index, bit in enumerate(self.candidate_bits):
                    # Preserve the released MC-MoE implementation: gama is not used,
                    # and quantization loss is raised to alpha.
                    layer_coefficients[expert_id, bit_index] = (
                        significance[expert_id]
                        * float(self.quant_loss[layer_id][expert_id][bit]) ** self.alpha
                        * 1000.0
                    )
            coefficients[layer_id] = layer_coefficients
        return coefficients

    def _build_constraints(self, budget_per_layer):
        experts = self.num_experts
        num_bits = len(self.candidate_bits)
        num_vars = experts * num_bits

        budget_row = np.tile(
            np.asarray(self.candidate_bits, dtype=np.float64), experts
        )
        budget = LinearConstraint(
            sp.csr_matrix(budget_row.reshape(1, num_vars)),
            -np.inf,
            float(budget_per_layer),
        )
        one_bit_per_expert = LinearConstraint(
            sp.csr_matrix(
                (
                    np.ones(num_vars),
                    np.arange(num_vars),
                    np.arange(0, num_vars + 1, num_bits),
                ),
                shape=(experts, num_vars),
            ),
            np.ones(experts),
            np.ones(experts),
        )

        rows = []
        cols = []
        for row, bit in enumerate((3, 2)):
            bit_index = self.candidate_bits.index(bit)
            for expert_id in range(experts):
                rows.append(row)
                cols.append(expert_id * num_bits + bit_index)
        at_least_one_2bit_and_3bit = LinearConstraint(
            sp.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(2, num_vars)
            ),
            np.ones(2),
            np.full(2, np.inf),
        )
        return [budget, one_bit_per_expert, at_least_one_2bit_and_3bit]

    def _solve_highs(self, objective, constraints):
        result = milp(
            c=objective,
            constraints=constraints,
            integrality=np.ones(objective.size),
            bounds=Bounds(0, 1),
        )
        if not result.success:
            raise RuntimeError(f"HiGHS failed to solve the PMQ ILP: {result.message}")
        return result.x

    def _solve_gurobi(self, objective, budget_per_layer):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as error:
            raise ImportError(
                "The Gurobi backend requires `pip install -e '.[gurobi]'`."
            ) from error

        experts = self.num_experts
        num_bits = len(self.candidate_bits)
        model = gp.Model("pmq_layer")
        model.Params.OutputFlag = 0
        variables = model.addMVar(
            shape=objective.size, vtype=GRB.BINARY, name="x"
        )
        model.setObjective(objective @ variables, GRB.MINIMIZE)
        bit_values = np.tile(np.asarray(self.candidate_bits), experts)
        model.addConstr(bit_values @ variables <= budget_per_layer, name="budget")
        for expert_id in range(experts):
            start = expert_id * num_bits
            model.addConstr(
                variables[start : start + num_bits].sum() == 1,
                name=f"expert_{expert_id}",
            )
        for bit in (3, 2):
            bit_index = self.candidate_bits.index(bit)
            model.addConstr(
                variables[bit_index::num_bits].sum() >= 1,
                name=f"has_{bit}bit",
            )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            status = model.Status
            model.dispose()
            raise RuntimeError(
                f"Gurobi finished PMQ allocation with status {status}, not optimal."
            )
        solution = np.asarray(variables.X, dtype=np.float64).copy()
        model.dispose()
        return solution

    def solve(self, *, bits_per_expert=2.0, require_full_budget=True):
        if not math.isfinite(bits_per_expert) or bits_per_expert <= 0:
            raise ValueError(
                f"bits_per_expert must be finite and positive, got {bits_per_expert}."
            )
        raw_budget = bits_per_expert * self.num_experts
        budget_per_layer = round(raw_budget)
        if not math.isclose(raw_budget, budget_per_layer, abs_tol=1e-9):
            raise ValueError(
                "PMQ uses an integer per-layer budget, but "
                f"{bits_per_expert} * {self.num_experts} = {raw_budget}."
            )

        min_feasible = self.num_experts * min(self.candidate_bits) + 3
        # At least one expert must remain at 2-bit, so an all-3-bit layer is not
        # feasible under the released PMQ constraints.
        max_feasible = self.num_experts * max(self.candidate_bits) - 1
        if not min_feasible <= budget_per_layer <= max_feasible:
            raise ValueError(
                f"Per-layer budget {budget_per_layer} is infeasible for "
                f"{self.num_experts} experts and PMQ's 2/3-bit presence constraints; "
                f"expected [{min_feasible}, {max_feasible}]."
            )

        constraints = self._build_constraints(budget_per_layer)
        allocation = {}
        objectives = {}
        used_bits = {}
        num_bits = len(self.candidate_bits)
        for layer_id in self.layer_ids:
            objective = self.coefficients[layer_id].reshape(-1)
            if self.backend == "highs":
                solution = self._solve_highs(objective, constraints)
            else:
                solution = self._solve_gurobi(objective, budget_per_layer)

            selected = np.asarray(solution).reshape(self.num_experts, num_bits)
            selected = np.rint(selected)
            if not np.allclose(selected.sum(axis=1), 1):
                raise RuntimeError(
                    f"Layer {layer_id} solution does not select one bit per expert."
                )
            bit_indices = selected.argmax(axis=1)
            layer_allocation = {
                expert_id: self.candidate_bits[bit_indices[expert_id]]
                for expert_id in range(self.num_experts)
            }
            layer_used_bits = sum(layer_allocation.values())
            if layer_used_bits > budget_per_layer:
                raise RuntimeError(
                    f"Layer {layer_id} exceeds its PMQ budget: "
                    f"{layer_used_bits} > {budget_per_layer}."
                )
            if require_full_budget and layer_used_bits != budget_per_layer:
                raise RuntimeError(
                    f"Layer {layer_id} used {layer_used_bits}/{budget_per_layer} bits. "
                    "The source ILP uses a <= constraint, but this run requested an "
                    "exact average-bit checkpoint."
                )
            if 2 not in layer_allocation.values() or 3 not in layer_allocation.values():
                raise RuntimeError(
                    f"Layer {layer_id} violates PMQ's 2/3-bit presence constraints."
                )

            allocation[layer_id] = layer_allocation
            used_bits[layer_id] = layer_used_bits
            objectives[layer_id] = float(
                sum(
                    self.coefficients[layer_id][expert_id, bit_indices[expert_id]]
                    for expert_id in range(self.num_experts)
                )
            )

        self.used_bits_per_layer = used_bits
        self.last_objective = float(sum(objectives.values()))
        return allocation
