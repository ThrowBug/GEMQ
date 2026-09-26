from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gemq.compute_model_stats import _compute_qwen35_mcmoe_losses
from gemq.pruning.qwen35 import prune_qwen35_experts
from gemq.utils.model_utils import (
    LinearModuleType,
    get_module_type,
    get_router_params,
)
from gemq.utils.qwen35 import (
    clone_packed_expert_weights,
    copy_packed_expert_weights_,
    forward_packed_expert,
    install_rtn_packed_expert_,
)


MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"


class _PackedExperts(nn.Module):
    def __init__(self, num_experts=4, hidden_size=2, intermediate_size=3):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.arange(num_experts * 2 * intermediate_size * hidden_size)
            .reshape(num_experts, 2 * intermediate_size, hidden_size)
            .float()
        )
        self.down_proj = nn.Parameter(
            torch.arange(num_experts * hidden_size * intermediate_size)
            .reshape(num_experts, hidden_size, intermediate_size)
            .float()
        )
        self.num_experts = num_experts
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            if token_idx.numel() == 0:
                continue
            output = forward_packed_expert(
                SimpleNamespace(experts=self),
                expert_idx,
                hidden_states[token_idx],
            )
            output = output * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, output.to(final.dtype))
        return final


class _Router(nn.Module):
    def __init__(self, num_experts=4, hidden_size=2):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(num_experts * hidden_size)
            .reshape(num_experts, hidden_size)
            .float()
        )
        self.num_experts = num_experts
        self.top_k = 2

    def forward(self, hidden_states):
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        probabilities = logits.softmax(dim=-1, dtype=torch.float)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(logits.dtype)
        return logits, weights, indices


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedExperts()
        self.gate = _Router()
        self.shared_expert = nn.Linear(2, 2, bias=False)
        self.shared_expert_gate = nn.Linear(2, 1, bias=False)
        self.num_experts = 4

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        _, routing_weights, selected_experts = self.gate(flat)
        routed = self.experts(flat, selected_experts, routing_weights)
        shared = self.shared_expert(flat)
        shared = self.shared_expert_gate(flat).sigmoid() * shared
        return (routed + shared).reshape(batch_size, sequence_length, hidden_size)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Moe()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.config = SimpleNamespace(num_experts=4, num_experts_per_tok=2)


def test_packed_expert_forward_matches_explicit_projection():
    moe = _Moe()
    hidden = torch.tensor([[0.5, -1.0], [2.0, 0.25]])

    output = forward_packed_expert(moe, 1, hidden)
    gate, up = torch.nn.functional.linear(
        hidden, moe.experts.gate_up_proj[1]
    ).chunk(2, dim=-1)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, moe.experts.down_proj[1]
    )

    torch.testing.assert_close(output, expected)


def test_qwen35_pruning_slices_native_packed_weights_and_router():
    model = _Model()
    old_gate_up = [
        layer.mlp.experts.gate_up_proj.detach().clone()
        for layer in model.model.layers
    ]
    old_router = [
        layer.mlp.gate.weight.detach().clone() for layer in model.model.layers
    ]
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }

    result = prune_qwen35_experts(model, MODEL_NAME, allocation)

    assert model.config.num_experts == 3
    assert result.remapped_bit_config == {
        0: {0: 2, 1: 3, 2: 2},
        1: {0: 2, 1: 2, 2: 3},
    }
    for layer_idx, kept in enumerate(((0, 2, 3), (1, 2, 3))):
        moe = model.model.layers[layer_idx].mlp
        torch.testing.assert_close(
            moe.experts.gate_up_proj, old_gate_up[layer_idx][list(kept)]
        )
        torch.testing.assert_close(
            moe.gate.weight, old_router[layer_idx][list(kept)]
        )
        assert moe.experts.down_proj.shape[0] == 3
        assert moe.experts.num_experts == moe.gate.num_experts == 3


def test_qwen35_linear_module_policy_is_exact():
    assert (
        get_module_type("linear_attn.in_proj_qkv", MODEL_NAME)
        == LinearModuleType.LINEAR_ATTN
    )
    assert (
        get_module_type("linear_attn.out_proj", MODEL_NAME)
        == LinearModuleType.LINEAR_ATTN
    )
    assert (
        get_module_type("linear_attn.in_proj_z", MODEL_NAME)
        == LinearModuleType.OTHERS
    )
    assert (
        get_module_type("self_attn.q_proj", MODEL_NAME)
        == LinearModuleType.SOFTMAX_ATTN
    )
    assert (
        get_module_type("mlp.shared_expert.gate_proj", MODEL_NAME)
        == LinearModuleType.DENSE
    )
    assert get_module_type("mlp.gate", MODEL_NAME) == LinearModuleType.GATE
    assert (
        get_module_type("mlp.shared_expert_gate", MODEL_NAME)
        == LinearModuleType.OTHERS
    )


def test_qwen35_router_params_exclude_shared_expert_gate():
    model = _Model()
    router_params = get_router_params(model, MODEL_NAME)
    expected = [layer.mlp.gate.weight for layer in model.model.layers]

    assert [id(param) for param in router_params] == [
        id(param) for param in expected
    ]
    shared_gate_ids = {
        id(layer.mlp.shared_expert_gate.weight) for layer in model.model.layers
    }
    assert all(id(param) not in shared_gate_ids for param in router_params)


def test_active_token_pmq_loss_matches_full_moe_recomputation():
    torch.manual_seed(7)
    moe = _Moe()
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.copy_(torch.randn_like(parameter))
    block_inputs = [torch.randn(1, 5, 2), torch.randn(1, 3, 2)]
    baseline = [moe(values).clone() for values in block_inputs]
    bits = (1, 2, 3)

    expected = {}
    for expert_idx in range(moe.num_experts):
        original = clone_packed_expert_weights(moe, expert_idx)
        expected[expert_idx] = {}
        try:
            for bit in bits:
                install_rtn_packed_expert_(
                    moe, expert_idx, original, bit, blocksize=2
                )
                expected[expert_idx][bit] = sum(
                    torch.norm(reference.double() - moe(values).double()).item()
                    for values, reference in zip(block_inputs, baseline)
                )
                copy_packed_expert_weights_(moe, expert_idx, original)
        finally:
            copy_packed_expert_weights_(moe, expert_idx, original)

    actual = _compute_qwen35_mcmoe_losses(
        moe,
        block_inputs,
        bits,
        blocksize=2,
        expert_batch_size=2,
    )

    for expert_idx in range(moe.num_experts):
        for bit in bits:
            assert actual[expert_idx][bit] == pytest.approx(
                expected[expert_idx][bit], rel=1e-5, abs=1e-6
            )
