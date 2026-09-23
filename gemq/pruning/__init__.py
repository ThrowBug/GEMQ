"""Utilities for turning zero-bit allocations into physically pruned models."""

from gemq.pruning.qwen3 import (
    PruningResult,
    has_zero_bit_experts,
    kept_expert_ids_from_pruning_metadata,
    load_expert_bit_config,
    prune_qwen3_experts,
)
from gemq.pruning.qwen35 import prune_qwen35_experts
from gemq.utils.model_utils import ModelType, NAME_TO_MODEL


def prune_zero_bit_experts(model, model_name, bit_config):
    model_type = NAME_TO_MODEL.get(model_name)
    if model_type == ModelType.QWEN3MOE:
        return prune_qwen3_experts(model, model_name, bit_config)
    if model_type == ModelType.QWEN35MOE:
        return prune_qwen35_experts(model, model_name, bit_config)
    raise NotImplementedError(
        f"Physical zero-bit pruning is not implemented for {model_name!r}."
    )

__all__ = [
    "PruningResult",
    "has_zero_bit_experts",
    "kept_expert_ids_from_pruning_metadata",
    "load_expert_bit_config",
    "prune_qwen3_experts",
    "prune_qwen35_experts",
    "prune_zero_bit_experts",
]
