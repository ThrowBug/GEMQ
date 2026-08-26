"""Physical expert pruning for Qwen3-MoE zero-bit allocations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from gemq.utils.model_utils import ModelType, NAME_TO_MODEL, get_blocks, get_moe_block


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
    if NAME_TO_MODEL.get(model_name) != ModelType.QWEN3MOE:
        raise NotImplementedError("Physical zero-bit pruning currently supports Qwen3-MoE only.")

    layers = get_blocks(model, model_name)
    if not layers:
        raise ValueError("The model has no decoder layers.")
    first_moe = get_moe_block(layers[0], model_name)
    num_experts = len(first_moe.experts)
    top_k = int(
        getattr(
            first_moe,
            "top_k",
            getattr(model.config, "num_experts_per_tok", 1),
        )
    )
    result = _build_pruning_result(bit_config, len(layers), num_experts, top_k)
    remaining = result.metadata["num_experts"]
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
