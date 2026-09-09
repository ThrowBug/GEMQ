"""Qwen3-MoE-specific AWQ search and application helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect

import torch

from .policy import ATTENTION_PROJECTIONS, EXPERT_PROJECTIONS, Qwen3MoeAWQPolicy
from .quantizer import fake_quantize_weight
from .search import (
    apply_clip,
    apply_linear_pair_scale,
    apply_norm_input_scale,
    search_clip,
    search_scale,
)


@dataclass(frozen=True)
class AWQSearchOptions:
    groupsize: int = 128
    scale_n_grid: int = 20
    clip_n_grid: int = 20
    clip_max_shrink: float = 0.5
    clip_n_sample_token: int = 512
    search_batch_size: int = 1

    def __post_init__(self):
        if self.groupsize <= 0:
            raise ValueError("groupsize must be positive")
        if self.scale_n_grid <= 0 or self.clip_n_grid <= 0:
            raise ValueError("AWQ grid sizes must be positive")
        if not 0 < self.clip_max_shrink <= 1:
            raise ValueError("clip_max_shrink must be in (0, 1]")
        if self.clip_n_sample_token <= 0 or self.search_batch_size <= 0:
            raise ValueError("AWQ sample and batch sizes must be positive")

    def to_dict(self):
        return asdict(self)


def validate_qwen3_moe_layer(layer, policy):
    missing = [
        name for name in ATTENTION_PROJECTIONS if not _has_qualified_attr(layer, name)
    ]
    if missing:
        raise ValueError(f"Qwen3-MoE layer is missing attention projections: {missing}")
    moe = getattr(layer, "mlp", None)
    if moe is None or not hasattr(moe, "experts") or not hasattr(moe, "gate"):
        raise ValueError("Expected layer.mlp with experts and a router gate")
    if len(moe.experts) != len(policy.expert_bits):
        raise ValueError(
            f"Layer has {len(moe.experts)} experts but policy has "
            f"{len(policy.expert_bits)} entries"
        )
    for expert_idx, expert in enumerate(moe.experts):
        missing_expert = [
            name for name in EXPERT_PROJECTIONS if not hasattr(expert, name)
        ]
        if missing_expert:
            raise ValueError(
                f"Expert {expert_idx} is missing projections: {missing_expert}"
            )


def _has_qualified_attr(module, name):
    current = module
    for piece in name.split("."):
        if not hasattr(current, piece):
            return False
        current = getattr(current, piece)
    return True


def attention_linears(layer):
    return {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.o_proj": layer.self_attn.o_proj,
    }


def expert_linears(expert):
    return {
        "gate_proj": expert.gate_proj,
        "up_proj": expert.up_proj,
        "down_proj": expert.down_proj,
    }


def _scaled_feature(input_feat, scale):
    view_shape = [1] * input_feat.ndim
    view_shape[-1] = scale.numel()
    return input_feat / scale.to(input_feat.device, input_feat.dtype).view(*view_shape)


def _filter_kwargs(module, kwargs):
    signature = inspect.signature(module.forward)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    accepted = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


@torch.no_grad()
def search_and_apply_attention_scales(
    layer,
    input_features,
    module_kwargs,
    policy,
    options,
):
    """Search and absorb the two standard Qwen attention AWQ scales."""
    qkv_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    )
    qkv = [attention_linears(layer)[name] for name in qkv_names]
    qkv_scale, qkv_error = search_scale(
        layer.self_attn,
        qkv,
        input_features["self_attn.q_proj"],
        [policy.attention_bits] * len(qkv),
        groupsize=options.groupsize,
        n_grid=options.scale_n_grid,
        search_batch_size=options.search_batch_size,
        module_kwargs=_filter_kwargs(layer.self_attn, module_kwargs),
    )
    apply_norm_input_scale(layer.input_layernorm, qkv, qkv_scale)
    scaled_qkv_input = _scaled_feature(input_features["self_attn.q_proj"], qkv_scale)
    for name in qkv_names:
        input_features[name] = scaled_qkv_input

    result = {"qkv_scale_error": qkv_error, "v_to_o_scale_error": None}
    value = layer.self_attn.v_proj
    output = layer.self_attn.o_proj
    if (
        value.weight.shape == output.weight.shape
        and "self_attn.o_proj" in input_features
    ):
        vo_scale, vo_error = search_scale(
            output,
            [output],
            input_features["self_attn.o_proj"],
            [policy.attention_bits],
            groupsize=options.groupsize,
            n_grid=options.scale_n_grid,
            search_batch_size=options.search_batch_size,
        )
        apply_linear_pair_scale(value, output, vo_scale)
        input_features["self_attn.o_proj"] = _scaled_feature(
            input_features["self_attn.o_proj"], vo_scale
        )
        result["v_to_o_scale_error"] = vo_error
    return result


@torch.no_grad()
def clip_and_quantize_attention(layer, input_features, policy, options):
    """Apply AWQ clipping and fake quantization to all attention projections."""
    clip_errors = {}
    for name, linear in attention_linears(layer).items():
        clip_max = None
        # AWQ deliberately avoids q/k range search because their error is
        # coupled through the QK product. They are still fake-quantized.
        if name not in {"self_attn.q_proj", "self_attn.k_proj"}:
            if name not in input_features:
                raise RuntimeError(f"Missing AWQ activation for {name}")
            clip_max = search_clip(
                linear.weight,
                input_features[name],
                policy.attention_bits,
                groupsize=options.groupsize,
                n_grid=options.clip_n_grid,
                max_shrink=options.clip_max_shrink,
                n_sample_token=options.clip_n_sample_token,
            )
            apply_clip(linear, clip_max)
            clip_errors[name] = float(clip_max.float().mean().item())
        linear.weight.data = fake_quantize_weight(
            linear.weight.data,
            nbits=policy.attention_bits,
            groupsize=options.groupsize,
        )
    return clip_errors


@torch.no_grad()
def search_and_apply_moe_input_scale(layer, moe_input, policy, options):
    """Jointly scale every expert input while preserving full-precision routing."""
    moe = layer.mlp
    targets = []
    bits = []
    for expert_idx, expert in enumerate(moe.experts):
        targets.extend([expert.gate_proj, expert.up_proj])
        bits.extend([policy.expert_bits[expert_idx]] * 2)

    scale, error = search_scale(
        moe,
        targets,
        moe_input,
        bits,
        groupsize=options.groupsize,
        n_grid=options.scale_n_grid,
        search_batch_size=options.search_batch_size,
    )
    # The router consumes the same inversely-scaled activation. Multiplying its
    # columns by the scale keeps router logits and Top-K identities unchanged.
    apply_norm_input_scale(
        layer.post_attention_layernorm,
        [moe.gate] + targets,
        scale,
    )
    return scale, error


@torch.no_grad()
def compute_expert_down_inputs(expert, active_inputs, batch_size, device):
    outputs = []
    activation = getattr(expert, "act_fn", None)
    if activation is None:
        raise TypeError("Qwen3-MoE expert is missing act_fn")
    for start in range(0, active_inputs.shape[0], batch_size):
        batch = active_inputs[start : start + batch_size].to(
            device=device, non_blocking=True
        )
        intermediate = activation(expert.gate_proj(batch)) * expert.up_proj(batch)
        outputs.append(intermediate.detach().to("cpu"))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def search_and_apply_expert_internal_scale(
    expert,
    down_inputs,
    bit,
    options,
):
    scale, error = search_scale(
        expert.down_proj,
        [expert.down_proj],
        down_inputs,
        [bit],
        groupsize=options.groupsize,
        n_grid=options.scale_n_grid,
        search_batch_size=options.search_batch_size,
    )
    apply_linear_pair_scale(expert.up_proj, expert.down_proj, scale)
    return scale, error, _scaled_feature(down_inputs, scale)


@torch.no_grad()
def search_expert_clips(expert, active_inputs, down_inputs, bit, options):
    features = {
        "gate_proj": active_inputs,
        "up_proj": active_inputs,
        "down_proj": down_inputs,
    }
    clips = {}
    for name, linear in expert_linears(expert).items():
        clips[name] = search_clip(
            linear.weight,
            features[name],
            bit,
            groupsize=options.groupsize,
            n_grid=options.clip_n_grid,
            max_shrink=options.clip_max_shrink,
            n_sample_token=options.clip_n_sample_token,
        )
    return clips


@torch.no_grad()
def install_expert_fake_quant(expert, bit, groupsize, clips=None):
    clips = clips or {}
    for name, linear in expert_linears(expert).items():
        clip_max = clips.get(name)
        linear.weight.data = fake_quantize_weight(
            linear.weight.data,
            nbits=bit,
            groupsize=groupsize,
            clip_max=clip_max,
        )


def layer_policy_from_bit_config(
    layer_bit_config,
    attention_bits=4,
    dense_bits=4,
):
    expected = set(range(len(layer_bit_config)))
    if set(layer_bit_config) != expected:
        raise ValueError(
            "Remapped layer bit config must have contiguous expert ids; "
            f"expected {sorted(expected)}, got {sorted(layer_bit_config)}"
        )
    bits = tuple(int(layer_bit_config[idx]) for idx in range(len(layer_bit_config)))
    if any(bit <= 0 for bit in bits):
        raise ValueError(
            "Physical pruning must remove every zero-bit expert before AWQ"
        )
    return Qwen3MoeAWQPolicy(
        expert_bits=bits,
        attention_bits=int(attention_bits),
        dense_bits=int(dense_bits),
    )
