"""Masked training and final physical pruning for Qwen3-MoE zero-bit experts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemq.router_finetune.output_reconstruction import (
    OutputReconstructionRecord,
    aggregate_output_reconstruction_records,
    build_output_reconstruction_target,
    output_reconstruction_kl,
)
from gemq.utils.model_utils import ModelType, NAME_TO_MODEL, get_blocks, get_moe_block


_MASK_BUFFER = "_gemq_zero_bit_expert_mask"
_ORIGINAL_FORWARD = "_gemq_unmasked_forward"
_COLLECT_TRANSFER = "_gemq_collect_output_reconstruction"
_TRANSFER_RECORD = "_gemq_output_reconstruction_record"


@dataclass
class PruningResult:
    remapped_bit_config: dict[int, dict[int, int]]
    kept_expert_ids: tuple[tuple[int, ...], ...]
    metadata: dict


def load_expert_bit_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        config = pickle.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Bit allocation must be a dict, got {type(config)!r}")
    normalized = {}
    for layer_idx, experts in config.items():
        if not isinstance(experts, dict):
            raise TypeError(f"Allocation for layer {layer_idx!r} must be a dict")
        normalized[int(layer_idx)] = {
            int(expert_idx): int(bit) for expert_idx, bit in experts.items()
        }
    return normalized


def has_zero_bit_experts(bit_config):
    return any(bit == 0 for experts in bit_config.values() for bit in experts.values())


def _build_pruning_result(bit_config, num_layers, num_experts, top_k):
    expected_layers = set(range(num_layers))
    if set(bit_config) != expected_layers:
        raise ValueError(
            "Bit allocation layer IDs must exactly match model layers: "
            f"expected {sorted(expected_layers)}, got {sorted(bit_config)}"
        )

    expected_experts = set(range(num_experts))
    kept_by_layer = []
    pruned_by_layer = []
    remapped = {}
    for layer_idx in range(num_layers):
        layer_config = bit_config[layer_idx]
        if set(layer_config) != expected_experts:
            raise ValueError(
                f"Layer {layer_idx} expert IDs must be 0..{num_experts - 1}; "
                f"got {sorted(layer_config)}"
            )
        invalid_bits = [bit for bit in layer_config.values() if bit < 0]
        if invalid_bits:
            raise ValueError(f"Layer {layer_idx} contains negative bit-widths")
        kept = tuple(i for i in range(num_experts) if layer_config[i] > 0)
        pruned = tuple(i for i in range(num_experts) if layer_config[i] == 0)
        if len(kept) < top_k:
            raise ValueError(
                f"Layer {layer_idx} keeps {len(kept)} experts, fewer than Top-K={top_k}."
            )
        kept_by_layer.append(kept)
        pruned_by_layer.append(pruned)
        remapped[layer_idx] = {
            new_id: layer_config[old_id] for new_id, old_id in enumerate(kept)
        }

    prune_counts = {len(ids) for ids in pruned_by_layer}
    if len(prune_counts) != 1:
        raise ValueError(
            "Physical Qwen3 pruning requires every layer to prune the same number "
            f"of experts, got counts {[len(ids) for ids in pruned_by_layer]}."
        )
    pruned_count = prune_counts.pop()
    remaining = num_experts - pruned_count
    metadata = {
        "format_version": 1,
        "model_type": "Qwen3-MoE",
        "allocation_id_space": "original_expert_ids",
        "original_num_experts": num_experts,
        "num_experts": remaining,
        "pruned_experts_per_layer": pruned_count,
        "layers": {
            str(layer_idx): {
                "kept_old_ids": list(kept_by_layer[layer_idx]),
                "pruned_old_ids": list(pruned_by_layer[layer_idx]),
                "old_to_new": {
                    str(old_id): new_id
                    for new_id, old_id in enumerate(kept_by_layer[layer_idx])
                },
            }
            for layer_idx in range(num_layers)
        },
    }
    return PruningResult(remapped, tuple(kept_by_layer), metadata)


def build_qwen3_pruning_plan(model, model_name, bit_config):
    """Validate a zero-bit allocation without changing model topology."""
    if NAME_TO_MODEL.get(model_name) != ModelType.QWEN3MOE:
        raise NotImplementedError("Zero-bit expert masking currently supports Qwen3-MoE only.")
    layers = get_blocks(model, model_name)
    if not layers:
        raise ValueError("The model has no decoder layers.")
    first_moe = get_moe_block(layers[0], model_name)
    if not isinstance(first_moe.experts, nn.ModuleList):
        raise TypeError(
            "Masked pruning expects the Transformers 4.57 Qwen3-MoE ModuleList "
            "expert implementation."
        )
    num_experts = len(first_moe.experts)
    top_k = int(
        getattr(first_moe, "top_k", getattr(model.config, "num_experts_per_tok", 1))
    )
    for layer_index, layer in enumerate(layers):
        moe = get_moe_block(layer, model_name)
        if not isinstance(moe.experts, nn.ModuleList):
            raise TypeError(
                f"Layer {layer_index} does not use the Transformers 4.57 "
                "Qwen3-MoE ModuleList expert implementation."
            )
        if len(moe.experts) != num_experts:
            raise ValueError(
                f"Layer {layer_index} has {len(moe.experts)} experts; "
                f"expected {num_experts}."
            )
        layer_top_k = int(
            getattr(moe, "top_k", getattr(model.config, "num_experts_per_tok", 1))
        )
        if layer_top_k != top_k:
            raise ValueError(
                f"Layer {layer_index} uses Top-K={layer_top_k}; expected {top_k}."
            )
    return _build_pruning_result(bit_config, len(layers), num_experts, top_k)


def _topk_probabilities(router_logits, top_k, normalize):
    probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float)
    probabilities, indices = torch.topk(probabilities, top_k, dim=-1)
    if normalize:
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities, indices


def _empty_transfer_record(router_probabilities, token_count):
    zero = router_probabilities.sum() * 0.0
    detached_zero = zero.detach()
    return OutputReconstructionRecord(
        loss_sum=zero,
        token_count=token_count,
        active_count=0,
        lost_mass_sum=detached_zero,
        fit_error_sum=detached_zero,
        layer_error_before_sum=detached_zero,
        layer_error_after_sum=detached_zero,
        relative_improvement_sum=detached_zero,
    )


def _masked_qwen3_moe_forward(self, hidden_states):
    """Transformers-4.57-compatible Qwen3 MoE forward with zero-bit masking."""
    if not isinstance(self.experts, nn.ModuleList):
        raise TypeError("Masked Qwen3 forward requires experts stored in nn.ModuleList.")
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat_hidden = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(flat_hidden)
    pruned_mask = getattr(self, _MASK_BUFFER)
    if pruned_mask.device != router_logits.device:
        raise RuntimeError("Qwen3 zero-bit expert mask did not move with its MoE layer.")
    masked_router_logits = router_logits.masked_fill(pruned_mask.unsqueeze(0), -torch.inf)
    routing_probabilities, selected_experts = _topk_probabilities(
        masked_router_logits, self.top_k, self.norm_topk_prob
    )
    routing_weights = routing_probabilities.to(flat_hidden.dtype)

    collect_transfer = bool(getattr(self, _COLLECT_TRANSFER, False))
    active_tokens = None
    active_lookup = None
    candidate_outputs = None
    unmasked_probabilities = None
    unmasked_experts = None
    lost_positions = None
    if collect_transfer:
        if not self.norm_topk_prob:
            raise ValueError(
                "Output-reconstruction KL requires Qwen3 norm_topk_prob=True so "
                "the routed FFN output is a convex combination."
            )
        unmasked_probabilities, unmasked_experts = _topk_probabilities(
            router_logits, self.top_k, True
        )
        lost_positions = pruned_mask[unmasked_experts]
        active_tokens = torch.where(lost_positions.any(dim=-1))[0]
        if active_tokens.numel() > 0:
            active_lookup = torch.full(
                (flat_hidden.shape[0],), -1, dtype=torch.long, device=flat_hidden.device
            )
            active_lookup[active_tokens] = torch.arange(
                active_tokens.numel(), device=flat_hidden.device
            )
            candidate_outputs = torch.zeros(
                (active_tokens.numel(), self.top_k, hidden_dim),
                dtype=flat_hidden.dtype,
                device=flat_hidden.device,
            )

    final_hidden_states = torch.zeros_like(flat_hidden)
    expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
    for expert_index in expert_hit.tolist():
        topk_position, token_index = torch.where(expert_mask[expert_index])
        current_state = flat_hidden[token_index]
        expert_output = self.experts[expert_index](current_state)
        weighted_output = expert_output * routing_weights[token_index, topk_position, None]
        final_hidden_states.index_add_(0, token_index, weighted_output.to(flat_hidden.dtype))

        if candidate_outputs is not None:
            active_rows = active_lookup[token_index]
            active_route = active_rows.ge(0)
            candidate_outputs[
                active_rows[active_route], topk_position[active_route]
            ] = expert_output[active_route].detach()

    if collect_transfer:
        if active_tokens is None or active_tokens.numel() == 0:
            setattr(
                self,
                _TRANSFER_RECORD,
                _empty_transfer_record(routing_probabilities, flat_hidden.shape[0]),
            )
        else:
            with torch.no_grad():
                missing_outputs = torch.zeros(
                    (active_tokens.numel(), hidden_dim),
                    dtype=torch.float,
                    device=flat_hidden.device,
                )
                pruned_hits = torch.unique(unmasked_experts[lost_positions])
                for expert_index in pruned_hits.tolist():
                    token_index, topk_position = torch.where(
                        unmasked_experts.eq(expert_index)
                    )
                    active_rows = active_lookup[token_index]
                    expert_output = self.experts[expert_index](
                        flat_hidden[token_index].detach()
                    ).float()
                    weighted_output = expert_output * unmasked_probabilities[
                        token_index, topk_position, None
                    ]
                    missing_outputs.index_add_(0, active_rows, weighted_output)

                active_unmasked_probabilities = unmasked_probabilities[active_tokens]
                missing_mass = (
                    active_unmasked_probabilities * lost_positions[active_tokens]
                ).sum(dim=-1)
                target, diagnostics = build_output_reconstruction_target(
                    candidate_outputs=candidate_outputs,
                    candidate_indices=selected_experts[active_tokens],
                    masked_probabilities=routing_probabilities[active_tokens].detach(),
                    unmasked_indices=unmasked_experts[active_tokens],
                    unmasked_probabilities=active_unmasked_probabilities,
                    missing_outputs=missing_outputs,
                    missing_mass=missing_mass,
                    current_output=final_hidden_states[active_tokens].detach(),
                )

            kl = output_reconstruction_kl(
                target, routing_probabilities[active_tokens]
            )
            setattr(
                self,
                _TRANSFER_RECORD,
                OutputReconstructionRecord(
                    loss_sum=kl.sum(),
                    token_count=flat_hidden.shape[0],
                    active_count=active_tokens.numel(),
                    **diagnostics,
                ),
            )

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return final_hidden_states, masked_router_logits


def mask_qwen3_experts(model, model_name, bit_config):
    """Logically prune zero-bit experts while retaining their frozen weights."""
    result = build_qwen3_pruning_plan(model, model_name, bit_config)
    layers = get_blocks(model, model_name)
    for layer_index, layer in enumerate(layers):
        moe = get_moe_block(layer, model_name)
        if hasattr(moe, _ORIGINAL_FORWARD) or hasattr(moe, _MASK_BUFFER):
            raise RuntimeError(f"Layer {layer_index} already has a GEMQ expert mask.")
    for layer_index, layer in enumerate(layers):
        moe = get_moe_block(layer, model_name)
        mask = torch.zeros(len(moe.experts), dtype=torch.bool, device=moe.gate.weight.device)
        mask[list(result.metadata["layers"][str(layer_index)]["pruned_old_ids"])] = True
        moe.register_buffer(_MASK_BUFFER, mask, persistent=False)
        setattr(moe, _ORIGINAL_FORWARD, moe.forward)
        setattr(moe, _COLLECT_TRANSFER, False)
        setattr(moe, _TRANSFER_RECORD, None)
        moe.forward = MethodType(_masked_qwen3_moe_forward, moe)

    result.metadata["state"] = "masked"
    print("Applied Qwen3 zero-bit expert masks; model topology remains unchanged.")
    return result


def remove_qwen3_expert_masks(model, model_name):
    """Restore the upstream MoE forward before physical pruning."""
    for layer in get_blocks(model, model_name):
        moe = get_moe_block(layer, model_name)
        if not hasattr(moe, _ORIGINAL_FORWARD):
            continue
        moe.forward = getattr(moe, _ORIGINAL_FORWARD)
        delattr(moe, _ORIGINAL_FORWARD)
        delattr(moe, _COLLECT_TRANSFER)
        delattr(moe, _TRANSFER_RECORD)
        delattr(moe, _MASK_BUFFER)


def set_qwen3_output_reconstruction(model, model_name, enabled):
    """Enable or disable collection of online transfer losses."""
    layers = get_blocks(model, model_name)
    masked_moes = []
    for layer in layers:
        moe = get_moe_block(layer, model_name)
        if hasattr(moe, _ORIGINAL_FORWARD):
            masked_moes.append(moe)
    if enabled and len(masked_moes) != len(layers):
        raise RuntimeError(
            "Output reconstruction requires a zero-bit expert mask on every "
            f"Qwen3 MoE layer, found {len(masked_moes)} of {len(layers)}."
        )
    if enabled and any(not moe.norm_topk_prob for moe in masked_moes):
        raise ValueError("Output reconstruction requires norm_topk_prob=True.")
    for moe in masked_moes:
        setattr(moe, _COLLECT_TRANSFER, bool(enabled))
        setattr(moe, _TRANSFER_RECORD, None)


def consume_qwen3_output_reconstruction(model, model_name, loss_device=None):
    """Consume exactly one transfer record from every masked MoE layer."""
    records = []
    for layer_index, layer in enumerate(get_blocks(model, model_name)):
        moe = get_moe_block(layer, model_name)
        if not hasattr(moe, _ORIGINAL_FORWARD):
            continue
        record = getattr(moe, _TRANSFER_RECORD)
        if record is None:
            raise RuntimeError(
                f"Masked Qwen3 MoE layer {layer_index} produced no transfer record."
            )
        records.append(record)
        setattr(moe, _TRANSFER_RECORD, None)
    return aggregate_output_reconstruction_records(records, loss_device=loss_device)


def snapshot_pruned_router_rows(model, model_name):
    """Save masked router rows so AdamW cannot decay them."""
    snapshots = []
    for layer_index, layer in enumerate(get_blocks(model, model_name)):
        moe = get_moe_block(layer, model_name)
        if not hasattr(moe, _MASK_BUFFER):
            raise RuntimeError(
                f"Layer {layer_index} has no zero-bit expert mask to snapshot."
            )
        mask = getattr(moe, _MASK_BUFFER)
        indices = torch.where(mask)[0]
        weight = moe.gate.weight.detach().index_select(0, indices).clone()
        bias = None
        if moe.gate.bias is not None:
            bias = moe.gate.bias.detach().index_select(0, indices).clone()
        snapshots.append((moe.gate, indices, weight, bias))
    return snapshots


@torch.no_grad()
def restore_pruned_router_rows(snapshots):
    for gate, indices, weight, bias in snapshots:
        gate.weight.index_copy_(0, indices, weight)
        if bias is not None:
            gate.bias.index_copy_(0, indices, bias)


def remap_quantized_module_names(quant_modules, pruning_result):
    """Update original expert IDs in the GPTQ module mapping after pruning."""
    remapped = {}
    for name, module in quant_modules.items():
        parts = name.split(".")
        layer_index = int(parts[0])
        try:
            experts_position = parts.index("experts")
        except ValueError:
            remapped[name] = module
            continue
        old_id = int(parts[experts_position + 1])
        old_to_new = pruning_result.metadata["layers"][str(layer_index)]["old_to_new"]
        if str(old_id) not in old_to_new:
            continue
        parts[experts_position + 1] = str(old_to_new[str(old_id)])
        new_name = ".".join(parts)
        if new_name in remapped:
            raise RuntimeError(f"Quantized module name collision after pruning: {new_name}")
        remapped[new_name] = module
    return remapped


def _slice_linear_output_rows(linear, kept_ids):
    index = torch.tensor(kept_ids, dtype=torch.long, device=linear.weight.device)
    weight = linear.weight.detach().index_select(0, index).clone()
    linear.weight = nn.Parameter(weight, requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        bias = linear.bias.detach().index_select(0, index).clone()
        linear.bias = nn.Parameter(bias, requires_grad=linear.bias.requires_grad)
    if hasattr(linear, "out_features"):
        linear.out_features = len(kept_ids)


def prune_qwen3_experts(model, model_name, bit_config):
    """Delete zero-bit experts and remap each layer's survivors contiguously."""
    result = build_qwen3_pruning_plan(model, model_name, bit_config)
    remove_qwen3_expert_masks(model, model_name)
    layers = get_blocks(model, model_name)
    num_experts = result.metadata["original_num_experts"]
    remaining = result.metadata["num_experts"]
    result.metadata["state"] = "physical"
    if remaining == num_experts:
        return result

    for layer_idx, layer in enumerate(layers):
        moe = get_moe_block(layer, model_name)
        if len(moe.experts) != num_experts:
            raise ValueError(
                f"Layer {layer_idx} has {len(moe.experts)} experts; expected {num_experts}."
            )
        kept_ids = result.kept_expert_ids[layer_idx]
        moe.experts = nn.ModuleList([moe.experts[old_id] for old_id in kept_ids])
        _slice_linear_output_rows(moe.gate, kept_ids)
        moe.num_experts = remaining
        if hasattr(moe, "config") and hasattr(moe.config, "num_experts"):
            moe.config.num_experts = remaining

    model.config.num_experts = remaining
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_experts"):
        text_config.num_experts = remaining
    print(
        f"Physically pruned {num_experts - remaining} experts from every Qwen3-MoE "
        f"layer; {remaining} experts/layer remain."
    )
    return result
