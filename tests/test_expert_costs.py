import torch
import torch.nn as nn

from gemq.expert_costs import _collect_active_inputs, _compute_layer_costs
from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.utils.expert_cost_utils import (
    impute_zero_frequency_costs,
    load_expert_cost_artifact,
    select_candidate_costs,
)


class _IdentityExpert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return hidden_states


class _TinyMoe(nn.Module):
    def __init__(self, num_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [_IdentityExpert(hidden_size) for _ in range(num_experts)]
        )


def test_collect_active_inputs_uses_only_topk_selected_tokens():
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]
    )
    selected_experts = torch.tensor(
        [[0, 2], [2, 1], [0, 2], [2, 0]], dtype=torch.long
    )
    routing_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.9, 0.1]]
    )

    active_inputs, active_gates = _collect_active_inputs(
        1, [hidden_states], [selected_experts], [routing_weights]
    )

    assert active_inputs.shape[0] == 1
    torch.testing.assert_close(active_inputs, torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(active_gates, torch.tensor([0.4]))


def test_zero_bit_cost_and_zero_frequency_policy():
    moe_block = _TinyMoe(num_experts=2, hidden_size=2)
    hidden_states = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    selected_experts = torch.tensor([[0], [0]], dtype=torch.long)
    routing_weights = torch.tensor([[0.5], [0.25]])

    costs, counts = _compute_layer_costs(
        moe_block=moe_block,
        moe_input_batches=[hidden_states],
        selected_expert_batches=[selected_experts],
        routing_weight_batches=[routing_weights],
        candidate_bits=[0],
        context_mode="fp",
        average_bits=2,
        blocksize=2,
        expert_batch_size=1,
        device=torch.device("cpu"),
    )

    # (0.5 * ||[3,4]||_2 + 0.25 * ||[0,2]||_2) / 2 = 1.5
    torch.testing.assert_close(costs[0, 0], torch.tensor(1.5, dtype=torch.float64))
    assert counts.tolist() == [2, 0]
    assert torch.isnan(costs[1, 0])


def test_rtn_zero_bit_and_partial_binary_block():
    weights = torch.tensor([[1.0, -2.0, 3.0]])

    zero_bit = MCMoeRTNWeightQuantizer.normal_quantize(
        weights, blocksize=2, wbit=0
    )
    one_bit = MCMoeRTNWeightQuantizer.normal_quantize(
        weights, blocksize=2, wbit=1
    )

    torch.testing.assert_close(zero_bit, torch.zeros_like(weights))
    assert one_bit.shape == weights.shape
    assert torch.isfinite(one_bit).all()


def test_zero_frequency_imputation_and_candidate_subset(tmp_path):
    raw = torch.tensor(
        [
            [
                [1.0, 10.0, 100.0, 1000.0],
                [3.0, 30.0, 300.0, 3000.0],
                [float("nan"), float("nan"), float("nan"), float("nan")],
            ]
        ],
        dtype=torch.float64,
    )
    counts = torch.tensor([[4, 2, 0]])
    filled, mask = impute_zero_frequency_costs(raw, counts)
    torch.testing.assert_close(
        filled[0, 2], torch.tensor([2.0, 20.0, 200.0, 2000.0], dtype=torch.float64)
    )
    assert mask.tolist() == [[False, False, True]]

    path = tmp_path / "v1.pt"
    torch.save(
        {
            "costs": raw,
            "counts": counts,
            "candidate_bits": torch.tensor([0, 1, 2, 3]),
            "metadata": {"format_version": 1},
        },
        path,
    )
    artifact = load_expert_cost_artifact(path)
    selected = select_candidate_costs(artifact, [0, 2, 3])
    assert selected.shape == (1, 3, 3)
    torch.testing.assert_close(
        selected[0, 0], torch.tensor([1.0, 100.0, 1000.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        selected[0, 2], torch.tensor([2.0, 200.0, 2000.0], dtype=torch.float64)
    )
