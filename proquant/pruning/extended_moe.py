"""Physical pruning with native checkpoint layouts for GLM and Trinity."""

import torch
from torch import nn

from proquant.pruning.qwen3 import _build_pruning_result, _slice_linear_output_rows
from proquant.utils.model_utils import ModelType, NAME_TO_MODEL, get_blocks, get_moe_block
from proquant.utils.qwen35 import get_num_routed_experts, slice_packed_expert_weights_


@torch.no_grad()
def prune_extended_moe_experts(model, model_name, bit_config):
    layers = get_blocks(model, model_name)
    blocks = [(i, get_moe_block(layer, model_name)) for i, layer in enumerate(layers)]
    blocks = [(i, moe) for i, moe in blocks if moe is not None]
    if not blocks:
        raise ValueError("The model has no routed layers.")
    first_moe_layer = blocks[0][0]
    if [i for i, _ in blocks] != list(range(first_moe_layer, len(layers))):
        raise ValueError("Physical pruning requires contiguous MoE layers after the dense prefix.")

    is_glm = NAME_TO_MODEL[model_name] == ModelType.GLM4MOELITE
    num_experts = model.config.n_routed_experts if is_glm else model.config.num_experts
    result = _build_pruning_result(
        bit_config, len(layers), num_experts, model.config.num_experts_per_tok,
        first_moe_layer=first_moe_layer, model_type=model.config.model_type,
    )
    remaining = result.metadata["num_experts"]
    for i, moe in blocks:
        size = get_num_routed_experts(moe) if is_glm else len(moe.experts)
        router = moe.gate if is_glm else moe.router.gate
        bias = router.e_score_correction_bias if is_glm else moe.expert_bias
        if size != num_experts or router.weight.shape[0] != size or bias.shape != (size,):
            raise ValueError(f"Expert weights, router rows or correction bias disagree in layer {i}.")
        if is_glm and remaining != num_experts and (moe.n_group != 1 or moe.topk_group != 1):
            raise ValueError("GLM pruning currently supports the single-group Flash router.")

    if remaining == num_experts:
        return result

    for (_, moe), kept in zip(blocks, result.kept_expert_ids):
        if is_glm:
            slice_packed_expert_weights_(moe.experts, kept)
            _slice_linear_output_rows(moe.gate, kept)
            bias = moe.gate.e_score_correction_bias
            moe.gate.e_score_correction_bias = bias[list(kept)].float()
            moe.n_routed_experts = moe.gate.n_routed_experts = remaining
        else:
            kept_set = set(kept)
            for expert_id in reversed(range(num_experts)):
                if expert_id not in kept_set:
                    del moe.experts[expert_id]
            moe.experts.num_experts = remaining
            _slice_linear_output_rows(moe.router.gate, kept)
            moe.router.num_experts = remaining
            moe.expert_bias = nn.Parameter(moe.expert_bias[list(kept)].float(), requires_grad=False)

    if is_glm:
        model.config.n_routed_experts = remaining
    else:
        model.config.num_experts = remaining
    print(f"Physically pruned {num_experts - remaining} experts from each of {len(blocks)} MoE layers.")
    return result
