"""Native Transformers routing and precision rules for model extensions."""

import torch


NATIVE_MODEL_TYPES = {
    "zai-org/GLM-4.7-Flash": "glm4_moe_lite",
    "arcee-ai/Trinity-Mini": "afmoe",
    "mistralai/Mixtral-8x7B-v0.1": "mixtral",
}


def glm_topk_routes(moe, hidden_states):
    return moe.route_tokens_to_experts(moe.gate(hidden_states))


def trinity_topk_routes(moe, hidden_states):
    weights, indices = moe.router(hidden_states, moe.expert_bias)
    return indices, weights


def mixtral_topk_routes(moe, hidden_states):
    _, weights, indices = moe.gate(hidden_states)
    return indices, weights


def routing_biases(model):
    """Yield the frozen correction biases, skipping dense decoder layers."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type not in {"afmoe", "glm4_moe_lite"}:
        return
    for layer in model.model.layers:
        if not hasattr(layer.mlp, "experts"):
            continue
        if model_type == "afmoe":
            yield layer.mlp, "expert_bias"
        else:
            yield layer.mlp.gate, "e_score_correction_bias"


def cast_model_preserving_fp32(model, dtype):
    """Keep routing corrections and native RoPE frequencies in FP32 across casts."""
    keep = list(routing_biases(model))
    if getattr(getattr(model, "config", None), "model_type", None) in NATIVE_MODEL_TYPES.values():
        rotary = model.model.rotary_emb
        keep.extend((rotary, name) for name in ("inv_freq", "original_inv_freq"))
    originals = [(module, name, getattr(module, name).detach().float()) for module, name in keep]
    model.to(dtype)
    for module, name, original in originals:
        value = getattr(module, name)
        value.data = original.to(value.device)
    return model
