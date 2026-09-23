from types import SimpleNamespace

import torch
import torch.nn as nn

from gemq.pruning.qwen35 import prune_qwen35_experts
from gemq.utils.model_utils import LinearModuleType, get_module_type
from gemq.utils.qwen35 import forward_packed_expert


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


class _Moe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedExperts()
        self.gate = _Router()
        self.num_experts = 4


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
