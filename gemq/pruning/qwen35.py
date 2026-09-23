"""Physical zero-bit pruning for packed Qwen3.5-MoE experts."""

from gemq.pruning.qwen3 import _build_pruning_result
from gemq.utils.model_utils import ModelType, NAME_TO_MODEL, get_blocks, get_moe_block
from gemq.utils.qwen35 import get_num_routed_experts, prune_packed_experts_


def prune_qwen35_experts(model, model_name, bit_config):
    if NAME_TO_MODEL.get(model_name) != ModelType.QWEN35MOE:
        raise NotImplementedError("Expected the registered Qwen3.5-MoE model.")

    layers = get_blocks(model, model_name)
    if not layers:
        raise ValueError("The model has no decoder layers.")
    first_moe = get_moe_block(layers[0], model_name)
    num_experts = get_num_routed_experts(first_moe)
    top_k = int(
        getattr(
            first_moe.gate,
            "top_k",
            getattr(model.config, "num_experts_per_tok", 1),
        )
    )
    result = _build_pruning_result(bit_config, len(layers), num_experts, top_k)
    remaining = int(result.metadata["num_experts"])
    result.metadata["model_type"] = "Qwen3.5-MoE"
    result.metadata["expert_storage"] = "packed_rank3_tensors"
    if remaining == num_experts:
        return result

    for layer_idx, layer in enumerate(layers):
        moe = get_moe_block(layer, model_name)
        actual = get_num_routed_experts(moe)
        if actual != num_experts:
            raise ValueError(
                f"Layer {layer_idx} has {actual} packed experts; expected {num_experts}."
            )
        prune_packed_experts_(moe, result.kept_expert_ids[layer_idx])

    configs = [model.config, getattr(model.config, "text_config", None)]
    for config in configs:
        if config is not None and hasattr(config, "num_experts"):
            config.num_experts = remaining
    if hasattr(model, "num_experts"):
        model.num_experts = remaining
    if hasattr(model, "model") and hasattr(model.model, "num_experts"):
        model.model.num_experts = remaining

    print(
        f"Physically pruned {num_experts - remaining} packed experts from every "
        f"Qwen3.5-MoE layer; {remaining} experts/layer remain."
    )
    return result

