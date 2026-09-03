from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemq.pruning.qwen3 import (
    consume_qwen3_output_reconstruction,
    mask_qwen3_experts,
    prune_qwen3_experts,
    remap_quantized_module_names,
    restore_pruned_router_rows,
    set_qwen3_output_reconstruction,
    snapshot_pruned_router_rows,
)
from gemq.router_finetune.targets import TeacherTargets, project_teacher_router_logits


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class _TinyMoe(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_experts)])
        self.gate = nn.Linear(2, num_experts, bias=True)
        self.num_experts = num_experts
        self.top_k = 2
        self.norm_topk_prob = True

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(flat)
        weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        weights, indices = torch.topk(weights, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for expert_index, expert in enumerate(self.experts):
            tokens, positions = torch.where(indices.eq(expert_index))
            if tokens.numel() == 0:
                continue
            expert_output = expert(flat[tokens])
            output.index_add_(0, tokens, expert_output * weights[tokens, positions, None])
        return output.reshape(batch_size, sequence_length, hidden_dim), router_logits


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


def test_masked_pruning_keeps_topology_and_matches_final_physical_pruning():
    model = _TinyModel()
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }
    hidden_states = torch.randn(1, 3, 2)
    mask_qwen3_experts(model, MODEL_NAME, allocation)

    assert model.config.num_experts == 4
    assert all(len(layer.mlp.experts) == 4 for layer in model.model.layers)
    masked_output = model.model.layers[0].mlp(hidden_states)[0]
    masked_logits = model.model.layers[0].mlp(hidden_states)[1]
    selected = masked_logits.topk(2, dim=-1).indices
    assert not selected.eq(1).any()

    prune_qwen3_experts(model, MODEL_NAME, allocation)
    physical_output = model.model.layers[0].mlp(hidden_states)[0]
    torch.testing.assert_close(masked_output, physical_output)


def test_masked_forward_collects_differentiable_output_reconstruction():
    model = _TinyModel()
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }
    for layer in model.model.layers:
        with torch.no_grad():
            layer.mlp.gate.weight.zero_()
            layer.mlp.gate.bias.copy_(torch.tensor([0.0, 4.0, 3.0, 2.0]))
    mask_qwen3_experts(model, MODEL_NAME, allocation)
    set_qwen3_output_reconstruction(model, MODEL_NAME, True)

    hidden_states = torch.randn(1, 3, 2)
    value = hidden_states
    for layer in model.model.layers:
        value = layer.mlp(value)[0]
    batch = consume_qwen3_output_reconstruction(model, MODEL_NAME)

    assert batch.active_count > 0
    assert batch.token_count == 2 * hidden_states.numel() // hidden_states.shape[-1]
    assert 0.0 < batch.pruned_hit_rate <= 1.0
    assert torch.isfinite(batch.loss)
    batch.loss.backward()
    for layer_index, layer in enumerate(model.model.layers):
        assert layer.mlp.gate.weight.grad is not None
        pruned_id = 1 if layer_index == 0 else 0
        torch.testing.assert_close(
            layer.mlp.gate.weight.grad[pruned_id],
            torch.zeros_like(layer.mlp.gate.weight.grad[pruned_id]),
        )


def test_snapshot_restore_keeps_masked_router_rows_fixed():
    model = _TinyModel()
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }
    mask_qwen3_experts(model, MODEL_NAME, allocation)
    original_weights = [layer.mlp.gate.weight.detach().clone() for layer in model.model.layers]
    original_biases = [layer.mlp.gate.bias.detach().clone() for layer in model.model.layers]
    snapshots = snapshot_pruned_router_rows(model, MODEL_NAME)

    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.add_(1.0)
            layer.mlp.gate.bias.add_(1.0)
    restore_pruned_router_rows(snapshots)

    for layer_index, layer in enumerate(model.model.layers):
        pruned_id = 1 if layer_index == 0 else 0
        kept_id = 0 if layer_index == 0 else 1
        torch.testing.assert_close(
            layer.mlp.gate.weight[pruned_id], original_weights[layer_index][pruned_id]
        )
        torch.testing.assert_close(
            layer.mlp.gate.bias[pruned_id], original_biases[layer_index][pruned_id]
        )
        torch.testing.assert_close(
            layer.mlp.gate.weight[kept_id], original_weights[layer_index][kept_id] + 1.0
        )


def test_quantized_module_names_are_remapped_after_physical_pruning():
    model = _TinyModel()
    allocation = {
        0: {0: 2, 1: 0, 2: 3, 3: 2},
        1: {0: 0, 1: 2, 2: 2, 3: 3},
    }
    result = prune_qwen3_experts(model, MODEL_NAME, allocation)
    modules = {
        "0.self_attn.q_proj": object(),
        "0.mlp.experts.0.gate_proj": object(),
        "0.mlp.experts.1.gate_proj": object(),
        "0.mlp.experts.2.gate_proj": object(),
        "1.mlp.experts.3.down_proj": object(),
    }

    remapped = remap_quantized_module_names(modules, result)

    assert set(remapped) == {
        "0.self_attn.q_proj",
        "0.mlp.experts.0.gate_proj",
        "0.mlp.experts.1.gate_proj",
        "1.mlp.experts.2.down_proj",
    }
    assert remapped["0.mlp.experts.1.gate_proj"] is modules[
        "0.mlp.experts.2.gate_proj"
    ]
