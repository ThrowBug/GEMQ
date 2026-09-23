"""Utilities for Qwen3.5-MoE's packed routed-expert representation.

Transformers stores all routed experts in two rank-3 parameters instead of a
``ModuleList`` of MLPs.  Keeping that native representation is important: a
fake-quantized checkpoint can then be saved and reloaded by the stock
``Qwen3_5MoeForCausalLM`` class without custom modeling code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer


@dataclass(frozen=True)
class PackedExpertWeights:
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor


def get_num_routed_experts(moe_block) -> int:
    experts = moe_block.experts
    if not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        raise TypeError("Expected Qwen3.5 packed expert tensors.")
    return int(experts.gate_up_proj.shape[0])


def get_packed_expert_weights(moe_block, expert_idx: int) -> PackedExpertWeights:
    num_experts = get_num_routed_experts(moe_block)
    if not 0 <= expert_idx < num_experts:
        raise IndexError(f"expert_idx={expert_idx} is outside [0, {num_experts}).")
    return PackedExpertWeights(
        gate_up_proj=moe_block.experts.gate_up_proj[expert_idx],
        down_proj=moe_block.experts.down_proj[expert_idx],
    )


def clone_packed_expert_weights(moe_block, expert_idx: int) -> PackedExpertWeights:
    weights = get_packed_expert_weights(moe_block, expert_idx)
    return PackedExpertWeights(
        gate_up_proj=weights.gate_up_proj.detach().clone(),
        down_proj=weights.down_proj.detach().clone(),
    )


@torch.no_grad()
def copy_packed_expert_weights_(
    moe_block, expert_idx: int, weights: PackedExpertWeights
) -> None:
    destination = get_packed_expert_weights(moe_block, expert_idx)
    destination.gate_up_proj.copy_(weights.gate_up_proj)
    destination.down_proj.copy_(weights.down_proj)


def packed_expert_intermediate(
    hidden_states: torch.Tensor, gate_up_proj: torch.Tensor, act_fn
) -> torch.Tensor:
    gate, up = F.linear(hidden_states, gate_up_proj).chunk(2, dim=-1)
    return act_fn(gate) * up


def forward_packed_expert(
    moe_block,
    expert_idx: int,
    hidden_states: torch.Tensor,
    weights: PackedExpertWeights | None = None,
) -> torch.Tensor:
    if weights is None:
        weights = get_packed_expert_weights(moe_block, expert_idx)
    intermediate = packed_expert_intermediate(
        hidden_states, weights.gate_up_proj, moe_block.experts.act_fn
    )
    return F.linear(intermediate, weights.down_proj)


def qwen35_topk_routes(moe_block, hidden_states: torch.Tensor):
    """Return ``(indices, normalized_topk_weights)`` from the native router."""
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    output = moe_block.gate(flat_hidden)
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise TypeError(
            "Qwen3.5 router must return (logits, topk_weights, topk_indices)."
        )
    routing_weights, selected_experts = output[1], output[2]
    return selected_experts, routing_weights


@torch.no_grad()
def install_rtn_packed_expert_(
    moe_block,
    expert_idx: int,
    original: PackedExpertWeights,
    bit: int,
    blocksize: int,
) -> None:
    quantized = PackedExpertWeights(
        gate_up_proj=MCMoeRTNWeightQuantizer.normal_quantize(
            original.gate_up_proj, blocksize=blocksize, wbit=bit
        ),
        down_proj=MCMoeRTNWeightQuantizer.normal_quantize(
            original.down_proj, blocksize=blocksize, wbit=bit
        ),
    )
    copy_packed_expert_weights_(moe_block, expert_idx, quantized)


@torch.no_grad()
def prune_packed_experts_(moe_block, kept_ids) -> None:
    """Slice packed expert tensors and router rows while preserving HF key names."""
    kept = torch.as_tensor(
        tuple(int(expert_id) for expert_id in kept_ids),
        dtype=torch.long,
        device=moe_block.experts.gate_up_proj.device,
    )
    experts = moe_block.experts
    gate_up_requires_grad = experts.gate_up_proj.requires_grad
    down_requires_grad = experts.down_proj.requires_grad
    gate_up = experts.gate_up_proj.index_select(0, kept).clone()
    down = experts.down_proj.index_select(0, kept).clone()
    experts.gate_up_proj = nn.Parameter(
        gate_up, requires_grad=gate_up_requires_grad
    )
    experts.down_proj = nn.Parameter(
        down, requires_grad=down_requires_grad
    )
    experts.num_experts = len(kept_ids)

    router = moe_block.gate
    router_kept = kept.to(router.weight.device)
    router_requires_grad = router.weight.requires_grad
    router_weight = router.weight.index_select(0, router_kept).clone()
    router.weight = nn.Parameter(
        router_weight, requires_grad=router_requires_grad
    )
    router.num_experts = len(kept_ids)

    if hasattr(moe_block, "num_experts"):
        moe_block.num_experts = len(kept_ids)
    for config in (
        getattr(moe_block, "config", None),
        getattr(experts, "config", None),
        getattr(router, "config", None),
    ):
        if config is not None and hasattr(config, "num_experts"):
            config.num_experts = len(kept_ids)

