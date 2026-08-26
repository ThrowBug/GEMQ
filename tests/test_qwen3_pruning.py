from types import SimpleNamespace

import torch
import torch.nn as nn

from gemq.pruning.qwen3 import prune_qwen3_experts
from gemq.router_finetune.targets import TeacherTargets, project_teacher_router_logits


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class _TinyMoe(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_experts)])
        self.gate = nn.Linear(2, num_experts, bias=True)
        self.num_experts = num_experts
        self.top_k = 2


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TinyMoe(4)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer(), _TinyLayer()])
        self.config = SimpleNamespace(num_experts=4, num_experts_per_tok=2)


def test_qwen3_physical_pruning_remaps_experts_and_router_rows():
    model = _TinyModel()
    old_weights = []
    for layer in model.model.layers:
        with torch.no_grad():
            layer.mlp.gate.weight.copy_(torch.arange(8).reshape(4, 2))
            layer.mlp.gate.bias.copy_(torch.arange(4))
        old_weights.append(layer.mlp.gate.weight.detach().clone())
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }

    result = prune_qwen3_experts(model, MODEL_NAME, allocation)

    assert model.config.num_experts == 3
    assert result.remapped_bit_config == {
        0: {0: 2, 1: 3, 2: 2},
        1: {0: 2, 1: 2, 2: 3},
    }
    torch.testing.assert_close(
        model.model.layers[0].mlp.gate.weight, old_weights[0][[0, 2, 3]]
    )
    torch.testing.assert_close(
        model.model.layers[1].mlp.gate.weight, old_weights[1][[1, 2, 3]]
    )
    assert all(len(layer.mlp.experts) == 3 for layer in model.model.layers)


def test_teacher_router_logits_are_projected_without_changing_final_targets():
    logits = (
        torch.arange(8).reshape(1, 2, 4),
        torch.arange(8, 16).reshape(1, 2, 4),
    )
    final_hidden = torch.randn(1, 2, 3)
    targets = TeacherTargets(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        attention_mask=None,
        router_logits=logits,
        final_hidden_states=final_hidden,
        metadata={},
    )
    projected = project_teacher_router_logits(targets, ((0, 2, 3), (1, 2, 3)))

    torch.testing.assert_close(projected.router_logits[0], logits[0][..., [0, 2, 3]])
    torch.testing.assert_close(projected.router_logits[1], logits[1][..., [1, 2, 3]])
    assert projected.final_hidden_states is final_hidden
