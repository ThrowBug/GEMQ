"""End-to-end mixed-bit AWQ fake quantization for pruned Qwen3-MoE."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from gemq.expert_costs import (
    _capture_decoder_inputs,
    _capture_qwen3_attention_inputs,
    _capture_qwen3_moe_inputs,
    _collect_active_inputs,
    _empty_cuda_cache,
    _forward_layer_batches,
    _get_qwen3_expert,
)
from gemq.quantizers.awq.qwen3 import (
    AWQSearchOptions,
    clip_and_quantize_attention,
    compute_expert_down_inputs,
    install_expert_fake_quant,
    layer_policy_from_bit_config,
    search_and_apply_attention_scales,
    search_and_apply_expert_internal_scale,
    search_and_apply_moe_input_scale,
    search_expert_clips,
    validate_qwen3_moe_layer,
)
from gemq.utils.model_utils import ModelType, NAME_TO_MODEL, get_blocks


def calibration_input_hash(dataloader):
    input_ids = torch.cat([batch[0].detach().to("cpu") for batch in dataloader], dim=0)
    return hashlib.sha256(input_ids.contiguous().numpy().tobytes()).hexdigest()


def validate_awq_allocation_sidecar(
    bit_config_path,
    expected_input_hash=None,
    expected_model_name=None,
    expected_model_dtype=None,
    expected_awq_config=None,
    expert_bit_config=None,
):
    sidecar_path = Path(bit_config_path).with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "GEMQ-AWQ requires the allocation metadata sidecar: " f"{sidecar_path}"
        )
    with sidecar_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if str(metadata.get("source_quantizer", "")).lower() != "awq":
        raise ValueError(
            "The selected allocation was not produced from AWQ expert costs."
        )
    allocation_bits = metadata.get("candidate_bits")
    if (
        not isinstance(allocation_bits, list)
        or not allocation_bits
        or len(set(allocation_bits)) != len(allocation_bits)
        or any(
            not isinstance(bit, int) or isinstance(bit, bool) or bit < 0
            for bit in allocation_bits
        )
        or not any(bit > 0 for bit in allocation_bits)
    ):
        raise ValueError(
            "GEMQ-AWQ allocation candidate_bits must be a non-empty, unique "
            "integer list containing a positive bit-width."
        )
    # Older AWQ sidecars predate source_candidate_bits and necessarily used all
    # four candidates, so falling back to candidate_bits preserves compatibility.
    source_bits = metadata.get("source_candidate_bits", allocation_bits)
    if source_bits != [0, 1, 2, 3]:
        raise ValueError(
            "GEMQ-AWQ source expert costs must contain candidates [0, 1, 2, 3]."
        )
    if any(bit not in source_bits for bit in allocation_bits):
        raise ValueError(
            "GEMQ-AWQ allocation candidates must be a subset of the source "
            f"candidates: source={source_bits}, allocation={allocation_bits}."
        )
    if expert_bit_config is not None:
        assigned_bits = {
            int(bit)
            for experts in expert_bit_config.values()
            for bit in experts.values()
        }
        unexpected_bits = sorted(assigned_bits.difference(allocation_bits))
        if unexpected_bits:
            raise ValueError(
                "AWQ bit config contains widths excluded from its IP candidate "
                f"set: unexpected={unexpected_bits}, candidates={allocation_bits}."
            )
    if abs(float(metadata.get("max_prune_ratio", -1)) - 0.1) > 1e-12:
        raise ValueError("GEMQ-AWQ allocation must use max_prune_ratio=0.1.")
    if abs(float(metadata.get("average_bit_budget", -1)) - 2.0) > 1e-12:
        raise ValueError(
            "GEMQ-AWQ allocation must use a 2.0-bit budget over original experts."
        )
    if metadata.get("budget_denominator") not in {None, "original_experts"}:
        raise ValueError("GEMQ-AWQ expects a budget over the original experts.")
    stored_hash = metadata.get("source_input_ids_sha256")
    if expected_input_hash is not None and stored_hash != expected_input_hash:
        raise ValueError(
            "The final AWQ calibration input IDs do not match the expert-cost inputs: "
            f"allocation={stored_hash}, current={expected_input_hash}."
        )
    source_config = metadata.get("source_awq_config")
    if not isinstance(source_config, dict):
        raise ValueError("AWQ allocation sidecar is missing source_awq_config.")
    if (
        expected_model_name is not None
        and source_config.get("model_name") != expected_model_name
    ):
        raise ValueError("AWQ allocation model_name does not match final quantization.")
    if (
        expected_model_dtype is not None
        and source_config.get("model_dtype") != expected_model_dtype
    ):
        raise ValueError(
            "AWQ allocation model_dtype does not match final quantization."
        )
    for key, expected in (expected_awq_config or {}).items():
        if source_config.get(key) != expected:
            raise ValueError(
                f"AWQ allocation {key}={source_config.get(key)!r} does not match "
                f"final quantization value {expected!r}."
            )
    return metadata


@torch.inference_mode()
def quantize_weights_awq(model, dataloader, args, expert_bit_config):
    """Search, absorb, clip, and fake-quantize a mixed or uniform model."""
    if NAME_TO_MODEL.get(args.model_name) != ModelType.QWEN3MOE:
        raise NotImplementedError("GEMQ-AWQ final quantization supports Qwen3-MoE only")
    mixed_precision = expert_bit_config is not None
    if not mixed_precision and not 1 <= int(args.expert_wbits) < 16:
        raise ValueError("Uniform GEMQ-AWQ requires expert_wbits in [1, 15]")

    options = AWQSearchOptions(
        groupsize=args.groupsize,
        scale_n_grid=args.awq_scale_n_grid,
        clip_n_grid=args.awq_clip_n_grid,
        clip_max_shrink=args.awq_clip_max_shrink,
        clip_n_sample_token=args.awq_clip_n_sample_token,
        search_batch_size=args.awq_search_batch_size,
    )
    device = torch.device(getattr(args, "awq_device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GEMQ-AWQ final quantization")
    layers = get_blocks(model, args.model_name)
    expected_layers = set(range(len(layers)))
    if mixed_precision and set(expert_bit_config) != expected_layers:
        raise ValueError(
            "AWQ allocation layer IDs do not match the model after pruning: "
            f"expected {sorted(expected_layers)}, got {sorted(expert_bit_config)}"
        )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    hidden_batches, positional_batches, keyword_batches = _capture_decoder_inputs(
        model, dataloader, args.model_name, device
    )
    layer_metadata = []

    try:
        for layer_idx in tqdm(range(len(layers)), desc="AWQ Quantizing"):
            layer = layers[layer_idx].to(device)
            if mixed_precision:
                layer_bit_config = expert_bit_config[layer_idx]
            else:
                layer_bit_config = {
                    expert_idx: int(args.expert_wbits)
                    for expert_idx in range(len(layer.mlp.experts))
                }
            policy = layer_policy_from_bit_config(
                layer_bit_config,
                attention_bits=args.attn_wbits,
                dense_bits=args.dense_wbits,
            )
            validate_qwen3_moe_layer(layer, policy)

            attention_inputs = _capture_qwen3_attention_inputs(
                layer,
                hidden_batches,
                positional_batches,
                keyword_batches,
                device,
            )
            attention_metadata = search_and_apply_attention_scales(
                layer,
                attention_inputs,
                keyword_batches[0],
                policy,
                options,
            )
            attention_metadata["clip_means"] = clip_and_quantize_attention(
                layer, attention_inputs, policy, options
            )

            (
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                _,
            ) = _capture_qwen3_moe_inputs(
                layer,
                layer.mlp,
                hidden_batches,
                positional_batches,
                keyword_batches,
                device,
                stop_at_moe=True,
            )
            concatenated_moe_input = torch.cat(moe_input_batches, dim=0)
            common_scale, common_error = search_and_apply_moe_input_scale(
                layer, concatenated_moe_input, policy, options
            )
            scaled_moe_inputs = [
                batch.div_(
                    common_scale.to(batch.device, batch.dtype).view(
                        *([1] * (batch.ndim - 1)), -1
                    )
                )
                for batch in moe_input_batches
            ]
            del concatenated_moe_input

            unhit_experts = []
            internal_errors = {}
            clip_means = {}
            for expert_idx, bit in enumerate(policy.expert_bits):
                expert = _get_qwen3_expert(layer.mlp, expert_idx)
                active_inputs, _ = _collect_active_inputs(
                    expert_idx,
                    scaled_moe_inputs,
                    selected_expert_batches,
                    routing_weight_batches,
                )
                if active_inputs is None:
                    install_expert_fake_quant(
                        expert, bit, options.groupsize, clips=None
                    )
                    unhit_experts.append(expert_idx)
                    continue

                down_inputs = compute_expert_down_inputs(
                    expert,
                    active_inputs,
                    args.expert_batch_size,
                    device,
                )
                _, internal_error, scaled_down_inputs = (
                    search_and_apply_expert_internal_scale(
                        expert, down_inputs, bit, options
                    )
                )
                clips = search_expert_clips(
                    expert, active_inputs, scaled_down_inputs, bit, options
                )
                install_expert_fake_quant(expert, bit, options.groupsize, clips=clips)
                internal_errors[str(expert_idx)] = internal_error
                clip_means[str(expert_idx)] = {
                    name: float(value.float().mean().item())
                    for name, value in clips.items()
                }
                del active_inputs, down_inputs, scaled_down_inputs, clips

            hidden_batches = _forward_layer_batches(
                layer,
                hidden_batches,
                positional_batches,
                keyword_batches,
                device,
            )
            layer_metadata.append(
                {
                    "layer": layer_idx,
                    "policy": policy.to_dict(),
                    "attention": attention_metadata,
                    "moe_input_scale_error": common_error,
                    "expert_internal_scale_errors": internal_errors,
                    "expert_clip_means": clip_means,
                    "unhit_experts": unhit_experts,
                }
            )

            layers[layer_idx] = layer.to("cpu")
            del (
                attention_inputs,
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                scaled_moe_inputs,
            )
            _empty_cuda_cache(device)
    finally:
        model.config.use_cache = use_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "format_version": 1,
        "quantizer": "gemq-awq",
        "fake_quantized": True,
        "packed_integer_weights": False,
        "model_name": args.model_name,
        "expert_precision": {
            "mode": "mixed" if mixed_precision else "uniform",
            "uniform_expert_bits": (
                None if mixed_precision else int(args.expert_wbits)
            ),
        },
        "calibration": {
            "dataset": args.calib_dataset,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
            "input_ids_sha256": calibration_input_hash(dataloader),
        },
        "search_options": options.to_dict(),
        "one_bit_definition": "per-group symmetric +/- mean(abs(clipped_weight))",
        "zero_bit_definition": "physical expert removal before AWQ search",
        "layers": layer_metadata,
    }
