"""Layer-wise expert quantization costs for Qwen3-MoE.

For routed expert ``i`` in decoder layer ``l`` and candidate bit-width ``b``
this module computes

    C[l, i, b] = mean_{x in X[l, i]} g[l, i](x)
                  * ||f[l, i](x) - f_b[l, i](x)||_2,

where ``X[l, i]`` contains only tokens whose Top-K routing decision selects the
expert.  A zero-bit expert is defined to produce the zero vector.  Experts that
are never selected keep a NaN cost and have a count of zero.
"""

import gc
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from gemq.quantizers.rtn import MCMoeRTNWeightQuantizer
from gemq.quantizers.awq.policy import Qwen3MoeAWQPolicy
from gemq.quantizers.awq.qwen3 import (
    AWQSearchOptions,
    clip_and_quantize_attention,
    compute_expert_down_inputs,
    install_expert_fake_quant,
    search_and_apply_attention_scales,
    search_and_apply_expert_internal_scale,
    search_and_apply_moe_input_scale,
    search_expert_clips,
    validate_qwen3_moe_layer,
)
from gemq.utils.model_utils import (
    ModelType,
    NAME_TO_MODEL,
    get_blocks,
    get_moe_block,
    move_embed,
)


_EXPERT_LINEAR_NAMES = ("gate_proj", "up_proj", "down_proj")


class _DecoderInputCaptured(RuntimeError):
    pass


class _MoeInputCaptured(RuntimeError):
    pass


def _tree_map_tensors(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map_tensors(item, fn) for key, item in value.items()}
    return value


def _to_cpu(value: Any):
    return _tree_map_tensors(value, lambda tensor: tensor.detach().to("cpu"))


def _to_device(value: Any, device: torch.device):
    return _tree_map_tensors(
        value, lambda tensor: tensor.to(device=device, non_blocking=True)
    )


def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected a tensor or a tensor-first output, got {type(output)!r}")


def _empty_cuda_cache(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.inference_mode()
def _capture_decoder_inputs(model, dataloader, model_name, device):
    """Capture first-layer inputs and kwargs, offloading every batch to CPU."""
    layers = get_blocks(model, model_name)
    hidden_batches = []
    positional_batches = []
    keyword_batches = []

    move_embed(model, model_name, device)
    first_layer = layers[0].to(device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            # Transformers' Qwen3 model reads this attribute while constructing
            # the per-layer attention mask.
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, hidden_states, *args, **kwargs):
            hidden_batches.append(_to_cpu(hidden_states))
            positional_batches.append(_to_cpu(args))
            keyword_batches.append(_to_cpu(kwargs))
            raise _DecoderInputCaptured

    layers[0] = Catcher(first_layer)
    try:
        for batch in dataloader:
            try:
                model(batch[0].to(device=device, non_blocking=True))
            except _DecoderInputCaptured:
                pass
    finally:
        layers[0] = first_layer
        move_embed(model, model_name, "cpu")
        layers[0] = first_layer.to("cpu")

    if not hidden_batches:
        raise ValueError("The calibration loader did not produce any batches.")
    if not (
        len(hidden_batches) == len(positional_batches) == len(keyword_batches)
    ):
        raise RuntimeError("Captured decoder inputs and layer arguments are misaligned.")

    _empty_cuda_cache(device)
    return hidden_batches, positional_batches, keyword_batches


def _qwen3_topk_routes(moe_block, hidden_states):
    """Reproduce Qwen3-MoE's token-level router decision."""
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_logits = moe_block.gate(flat_hidden)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, moe_block.top_k, dim=-1
    )
    if getattr(moe_block, "norm_topk_prob", False):
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
    return selected_experts, routing_weights


@torch.inference_mode()
def _capture_qwen3_moe_inputs(
    layer,
    moe_block,
    hidden_batches,
    positional_batches,
    keyword_batches,
    device,
    stop_at_moe,
):
    """Capture MoE inputs/routes, optionally stopping before any expert runs."""
    moe_input_batches = []
    selected_expert_batches = []
    routing_weight_batches = []
    layer_output_batches = []

    def capture_hook(module, inputs):
        hidden_states = inputs[0]
        selected_experts, routing_weights = _qwen3_topk_routes(
            module, hidden_states
        )
        moe_input_batches.append(_to_cpu(hidden_states))
        selected_expert_batches.append(_to_cpu(selected_experts))
        routing_weight_batches.append(_to_cpu(routing_weights))
        if stop_at_moe:
            raise _MoeInputCaptured

    handle = moe_block.register_forward_pre_hook(capture_hook)
    try:
        for hidden_states, args, kwargs in zip(
            hidden_batches, positional_batches, keyword_batches
        ):
            hidden_states = hidden_states.to(device=device, non_blocking=True)
            args = _to_device(args, device)
            kwargs = _to_device(kwargs, device)
            try:
                output = layer(hidden_states, *args, **kwargs)
            except _MoeInputCaptured:
                if not stop_at_moe:
                    raise
            else:
                if stop_at_moe:
                    raise RuntimeError("Qwen3 MoE hook did not stop the layer forward.")
                layer_output_batches.append(_to_cpu(_first_tensor(output)))
    finally:
        handle.remove()

    expected = len(hidden_batches)
    if not (
        len(moe_input_batches)
        == len(selected_expert_batches)
        == len(routing_weight_batches)
        == expected
    ):
        raise RuntimeError("Did not capture exactly one Qwen3 MoE input per batch.")
    if not stop_at_moe and len(layer_output_batches) != expected:
        raise RuntimeError("Full-precision layer outputs are missing.")

    return (
        moe_input_batches,
        selected_expert_batches,
        routing_weight_batches,
        layer_output_batches,
    )


@torch.inference_mode()
def _forward_layer_batches(
    layer,
    hidden_batches,
    positional_batches,
    keyword_batches,
    device,
):
    outputs = []
    for hidden_states, args, kwargs in zip(
        hidden_batches, positional_batches, keyword_batches
    ):
        hidden_states = hidden_states.to(device=device, non_blocking=True)
        args = _to_device(args, device)
        kwargs = _to_device(kwargs, device)
        output = layer(hidden_states, *args, **kwargs)
        outputs.append(_to_cpu(_first_tensor(output)))
    return outputs


@torch.inference_mode()
def _capture_qwen3_attention_inputs(
    layer,
    hidden_batches,
    positional_batches,
    keyword_batches,
    device,
):
    """Capture QKV and output-projection inputs from the current FP layer."""
    features = {"self_attn.q_proj": [], "self_attn.o_proj": []}

    def capture(module, inputs, output, name):
        features[name].append(_to_cpu(inputs[0]))

    handles = [
        layer.self_attn.q_proj.register_forward_hook(
            lambda module, inputs, output: capture(
                module, inputs, output, "self_attn.q_proj"
            )
        ),
        layer.self_attn.o_proj.register_forward_hook(
            lambda module, inputs, output: capture(
                module, inputs, output, "self_attn.o_proj"
            )
        ),
    ]
    try:
        for hidden_states, args, kwargs in zip(
            hidden_batches, positional_batches, keyword_batches
        ):
            layer(
                hidden_states.to(device=device, non_blocking=True),
                *_to_device(args, device),
                **_to_device(kwargs, device),
            )
    finally:
        for handle in handles:
            handle.remove()

    expected = len(hidden_batches)
    if any(len(values) != expected for values in features.values()):
        raise RuntimeError("Did not capture exactly one attention input per batch")
    qkv_input = torch.cat(features["self_attn.q_proj"], dim=0)
    return {
        "self_attn.q_proj": qkv_input,
        "self_attn.k_proj": qkv_input,
        "self_attn.v_proj": qkv_input,
        "self_attn.o_proj": torch.cat(features["self_attn.o_proj"], dim=0),
    }


def _get_qwen3_expert(moe_block, expert_idx):
    try:
        expert = moe_block.experts[expert_idx]
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError(
            "This collector expects Qwen3MoeSparseMoeBlock.experts to be an "
            "indexable collection of per-expert MLP modules."
        ) from error

    missing = [name for name in _EXPERT_LINEAR_NAMES if not hasattr(expert, name)]
    if missing:
        raise TypeError(
            f"Qwen3 expert {expert_idx} is missing linear modules: {missing}"
        )
    return expert


def _expert_linears(expert):
    return [getattr(expert, name) for name in _EXPERT_LINEAR_NAMES]


def _restore_weights(linears, original_weights):
    for linear, weight in zip(linears, original_weights):
        linear.weight.data = weight


def _install_quantized_weights(linears, original_weights, bit, blocksize):
    for linear, weight in zip(linears, original_weights):
        linear.weight.data = MCMoeRTNWeightQuantizer.normal_quantize(
            weight, blocksize=blocksize, wbit=bit
        )


def _collect_active_inputs(
    expert_idx,
    moe_input_batches,
    selected_expert_batches,
    routing_weight_batches,
):
    active_inputs = []
    active_gates = []
    for hidden_states, selected_experts, routing_weights in zip(
        moe_input_batches, selected_expert_batches, routing_weight_batches
    ):
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        matches = selected_experts.eq(expert_idx)
        token_mask = matches.any(dim=-1)
        if not token_mask.any().item():
            continue

        # Top-K indices are unique, but summing over the Top-K axis also makes
        # this correct if a custom router ever emits a duplicate expert id.
        token_gates = (routing_weights * matches).sum(dim=-1)
        active_inputs.append(flat_hidden[token_mask])
        active_gates.append(token_gates[token_mask])

    if not active_inputs:
        return None, None
    return torch.cat(active_inputs, dim=0), torch.cat(active_gates, dim=0)


@torch.inference_mode()
def _reference_outputs(expert, active_inputs, expert_batch_size, device):
    outputs = []
    for start in range(0, active_inputs.shape[0], expert_batch_size):
        end = min(start + expert_batch_size, active_inputs.shape[0])
        expert_inputs = active_inputs[start:end].to(
            device=device, non_blocking=True
        )
        outputs.append(_to_cpu(_first_tensor(expert(expert_inputs))))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def _weighted_deviation_sum(
    expert,
    active_inputs,
    active_gates,
    reference_outputs,
    expert_batch_size,
    device,
    zero_output=False,
):
    total = 0.0
    for start in range(0, active_inputs.shape[0], expert_batch_size):
        end = min(start + expert_batch_size, active_inputs.shape[0])
        reference = reference_outputs[start:end].to(
            device=device, non_blocking=True
        )
        if zero_output:
            difference = reference.float()
        else:
            expert_inputs = active_inputs[start:end].to(
                device=device, non_blocking=True
            )
            quantized_output = _first_tensor(expert(expert_inputs))
            difference = reference.float() - quantized_output.float()

        token_costs = torch.linalg.vector_norm(difference, ord=2, dim=-1)
        gates = active_gates[start:end].to(
            device=device, dtype=token_costs.dtype, non_blocking=True
        )
        total += (gates * token_costs).double().sum().item()
    return total


@torch.inference_mode()
def _compute_layer_costs(
    moe_block,
    moe_input_batches,
    selected_expert_batches,
    routing_weight_batches,
    candidate_bits,
    context_mode,
    average_bits,
    blocksize,
    expert_batch_size,
    device,
):
    num_experts = moe_block.num_experts
    num_bits = len(candidate_bits)
    costs = torch.full((num_experts, num_bits), torch.nan, dtype=torch.float64)
    counts = torch.zeros(num_experts, dtype=torch.long)
    bit_to_column = {bit: column for column, bit in enumerate(candidate_bits)}

    for expert_idx in tqdm(
        range(num_experts), desc="experts", leave=False, dynamic_ncols=True
    ):
        expert = _get_qwen3_expert(moe_block, expert_idx)
        linears = _expert_linears(expert)
        original_weights = tuple(linear.weight.data for linear in linears)

        active_inputs, active_gates = _collect_active_inputs(
            expert_idx,
            moe_input_batches,
            selected_expert_batches,
            routing_weight_batches,
        )

        if active_inputs is None:
            # NaN is intentional: there is no empirical expectation to estimate.
            # In uniform mode we still quantize the expert so the subsequent full
            # layer forward represents an entirely uniform-bit MoE layer.
            if context_mode == "uniform_bit":
                _install_quantized_weights(
                    linears, original_weights, average_bits, blocksize
                )
            continue

        count = active_inputs.shape[0]
        counts[expert_idx] = count
        reference_outputs = _reference_outputs(
            expert, active_inputs, expert_batch_size, device
        )

        if 0 in bit_to_column:
            numerator = _weighted_deviation_sum(
                expert,
                active_inputs,
                active_gates,
                reference_outputs,
                expert_batch_size,
                device,
                zero_output=True,
            )
            costs[expert_idx, bit_to_column[0]] = numerator / count

        nonzero_bits = [bit for bit in candidate_bits if bit != 0]
        if context_mode == "uniform_bit":
            # Evaluate the context bit last and keep those weights installed for
            # the one full background forward that feeds the next layer.
            nonzero_bits = [
                bit for bit in nonzero_bits if bit != average_bits
            ] + [average_bits]

        keep_quantized = False
        try:
            for bit in nonzero_bits:
                _install_quantized_weights(
                    linears, original_weights, bit, blocksize
                )
                numerator = _weighted_deviation_sum(
                    expert,
                    active_inputs,
                    active_gates,
                    reference_outputs,
                    expert_batch_size,
                    device,
                )
                costs[expert_idx, bit_to_column[bit]] = numerator / count

                keep_this_bit = (
                    context_mode == "uniform_bit" and bit == average_bits
                )
                if keep_this_bit:
                    keep_quantized = True
                else:
                    _restore_weights(linears, original_weights)
        finally:
            if not keep_quantized:
                _restore_weights(linears, original_weights)

        del active_inputs, active_gates, reference_outputs

    return costs, counts


@torch.inference_mode()
def compute_qwen3_expert_costs(
    model,
    dataloader,
    model_name,
    candidate_bits,
    context_mode="uniform_bit",
    average_bits=2,
    blocksize=128,
    expert_batch_size=4096,
    device="cuda",
):
    """Compute ``[layer, expert, candidate_bit]`` costs and active counts."""
    if NAME_TO_MODEL.get(model_name) != ModelType.QWEN3MOE:
        raise NotImplementedError(
            "The first expert-cost implementation supports Qwen3-MoE only; "
            f"got model_name={model_name!r}."
        )
    if context_mode not in {"fp", "uniform_bit"}:
        raise ValueError(
            f"context_mode must be 'fp' or 'uniform_bit', got {context_mode!r}"
        )
    if not candidate_bits or any(
        not isinstance(bit, int) or bit < 0 for bit in candidate_bits
    ):
        raise ValueError("candidate_bits must contain non-negative integers.")
    if any(bit > 16 for bit in candidate_bits):
        raise ValueError("candidate_bits must be at most 16.")
    if len(set(candidate_bits)) != len(candidate_bits):
        raise ValueError("candidate_bits must not contain duplicates.")
    if not isinstance(average_bits, int) or average_bits <= 0:
        raise ValueError("average_bits must be a positive integer.")
    if context_mode == "uniform_bit" and average_bits not in candidate_bits:
        raise ValueError(
            "average_bits must be included in candidate_bits for uniform_bit context."
        )
    if blocksize <= 0 or expert_batch_size <= 0:
        raise ValueError("blocksize and expert_batch_size must be positive.")

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available to PyTorch.")

    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_blocks(model, model_name)

    hidden_batches, positional_batches, keyword_batches = _capture_decoder_inputs(
        model, dataloader, model_name, device
    )

    num_layers = len(layers)
    first_moe = get_moe_block(layers[0], model_name)
    num_experts = first_moe.num_experts
    costs = torch.full(
        (num_layers, num_experts, len(candidate_bits)),
        torch.nan,
        dtype=torch.float64,
    )
    counts = torch.zeros((num_layers, num_experts), dtype=torch.long)

    try:
        for layer_idx in tqdm(
            range(num_layers), desc="expert cost layers", dynamic_ncols=True
        ):
            layer = layers[layer_idx].to(device)
            moe_block = get_moe_block(layer, model_name)
            if moe_block.num_experts != num_experts:
                raise ValueError(
                    f"Layer {layer_idx} has {moe_block.num_experts} experts; "
                    f"expected {num_experts}."
                )

            (
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                fp_layer_outputs,
            ) = _capture_qwen3_moe_inputs(
                layer,
                moe_block,
                hidden_batches,
                positional_batches,
                keyword_batches,
                device,
                stop_at_moe=(context_mode == "uniform_bit"),
            )

            layer_costs, layer_counts = _compute_layer_costs(
                moe_block,
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                candidate_bits,
                context_mode,
                average_bits,
                blocksize,
                expert_batch_size,
                device,
            )
            costs[layer_idx] = layer_costs
            counts[layer_idx] = layer_counts

            del (
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
            )
            if context_mode == "uniform_bit":
                hidden_batches = _forward_layer_batches(
                    layer,
                    hidden_batches,
                    positional_batches,
                    keyword_batches,
                    device,
                )
            else:
                hidden_batches = fp_layer_outputs

            layers[layer_idx] = layer.to("cpu")
            del fp_layer_outputs, layer_costs, layer_counts
            _empty_cuda_cache(device)
    finally:
        model.config.use_cache = original_use_cache

    return costs, counts


def _clone_expert_weights(expert):
    return tuple(linear.weight.detach().clone() for linear in _expert_linears(expert))


@torch.inference_mode()
def _compute_awq_expert_candidates(
    expert,
    active_inputs,
    active_gates,
    down_inputs,
    candidate_bits,
    average_bits,
    options,
    expert_batch_size,
    device,
):
    """Evaluate candidates independently and leave the expert at context W2."""
    costs = torch.full((len(candidate_bits),), torch.nan, dtype=torch.float64)
    bit_to_column = {bit: column for column, bit in enumerate(candidate_bits)}
    reference_outputs = _reference_outputs(
        expert, active_inputs, expert_batch_size, device
    )
    count = active_inputs.shape[0]

    if 0 in bit_to_column:
        numerator = _weighted_deviation_sum(
            expert,
            active_inputs,
            active_gates,
            reference_outputs,
            expert_batch_size,
            device,
            zero_output=True,
        )
        costs[bit_to_column[0]] = numerator / count

    scaled_weights = _clone_expert_weights(expert)
    context_weights = None
    try:
        for bit in (candidate for candidate in candidate_bits if candidate > 0):
            _restore_weights(_expert_linears(expert), scaled_weights)
            clips = search_expert_clips(
                expert, active_inputs, down_inputs, bit, options
            )
            install_expert_fake_quant(
                expert,
                bit=bit,
                groupsize=options.groupsize,
                clips=clips,
            )
            numerator = _weighted_deviation_sum(
                expert,
                active_inputs,
                active_gates,
                reference_outputs,
                expert_batch_size,
                device,
            )
            costs[bit_to_column[bit]] = numerator / count
            if bit == average_bits:
                context_weights = _clone_expert_weights(expert)
    finally:
        if context_weights is None:
            _restore_weights(_expert_linears(expert), scaled_weights)
        else:
            _restore_weights(_expert_linears(expert), context_weights)

    if context_weights is None:
        raise RuntimeError(
            f"Context bit W{average_bits} was not evaluated for an active expert"
        )
    return costs


@torch.inference_mode()
def _compute_awq_layer_costs(
    layer,
    moe_input_batches,
    selected_expert_batches,
    routing_weight_batches,
    candidate_bits,
    average_bits,
    policy,
    options,
    expert_batch_size,
    device,
):
    moe = layer.mlp
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

    num_experts = len(moe.experts)
    costs = torch.full(
        (num_experts, len(candidate_bits)), torch.nan, dtype=torch.float64
    )
    counts = torch.zeros(num_experts, dtype=torch.long)
    unhit_experts = []
    internal_errors = {}

    for expert_idx in tqdm(
        range(num_experts), desc="AWQ experts", leave=False, dynamic_ncols=True
    ):
        expert = _get_qwen3_expert(moe, expert_idx)
        active_inputs, active_gates = _collect_active_inputs(
            expert_idx,
            scaled_moe_inputs,
            selected_expert_batches,
            routing_weight_batches,
        )
        if active_inputs is None:
            # The common scale is valid for all experts. With no routed tokens,
            # neither the internal scale nor activation-conditioned clipping is
            # identifiable, so use unclipped W2 for the sequential context.
            install_expert_fake_quant(
                expert, average_bits, options.groupsize, clips=None
            )
            unhit_experts.append(expert_idx)
            continue

        counts[expert_idx] = active_inputs.shape[0]
        down_inputs = compute_expert_down_inputs(
            expert, active_inputs, expert_batch_size, device
        )
        _, internal_error, scaled_down_inputs = (
            search_and_apply_expert_internal_scale(
                expert,
                down_inputs,
                average_bits,
                options,
            )
        )
        internal_errors[str(expert_idx)] = internal_error
        costs[expert_idx] = _compute_awq_expert_candidates(
            expert,
            active_inputs,
            active_gates,
            scaled_down_inputs,
            candidate_bits,
            average_bits,
            options,
            expert_batch_size,
            device,
        )
        del active_inputs, active_gates, down_inputs, scaled_down_inputs

    metadata = {
        "moe_input_scale_error": common_error,
        "expert_internal_scale_errors": internal_errors,
        "unhit_experts": unhit_experts,
    }
    return costs, counts, metadata


@torch.inference_mode()
def compute_qwen3_awq_expert_costs(
    model,
    dataloader,
    model_name,
    candidate_bits,
    average_bits=2,
    expert_batch_size=4096,
    device="cuda",
    groupsize=128,
    scale_n_grid=20,
    clip_n_grid=20,
    clip_max_shrink=0.5,
    clip_n_sample_token=512,
    search_batch_size=1,
    attention_bits=4,
    dense_bits=4,
):
    """Collect AWQ-conditioned expert costs in a sequential W2 context."""
    if NAME_TO_MODEL.get(model_name) != ModelType.QWEN3MOE:
        raise NotImplementedError("GEMQ-AWQ expert costs support Qwen3-MoE only")
    if not candidate_bits or len(set(candidate_bits)) != len(candidate_bits):
        raise ValueError("candidate_bits must be a non-empty list without duplicates")
    if any(not isinstance(bit, int) or bit < 0 or bit > 16 for bit in candidate_bits):
        raise ValueError("candidate_bits must contain integers in [0, 16]")
    if average_bits not in candidate_bits or average_bits <= 0:
        raise ValueError("average_bits must be a positive candidate bit-width")
    if attention_bits >= 16 or dense_bits >= 16:
        raise ValueError("The AWQ cost context expects quantized attention/dense bits")

    options = AWQSearchOptions(
        groupsize=groupsize,
        scale_n_grid=scale_n_grid,
        clip_n_grid=clip_n_grid,
        clip_max_shrink=clip_max_shrink,
        clip_n_sample_token=clip_n_sample_token,
        search_batch_size=search_batch_size,
    )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available to PyTorch")

    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_blocks(model, model_name)
    hidden_batches, positional_batches, keyword_batches = _capture_decoder_inputs(
        model, dataloader, model_name, device
    )
    num_layers = len(layers)
    num_experts = len(get_moe_block(layers[0], model_name).experts)
    costs = torch.full(
        (num_layers, num_experts, len(candidate_bits)),
        torch.nan,
        dtype=torch.float64,
    )
    counts = torch.zeros((num_layers, num_experts), dtype=torch.long)
    search_metadata = []

    try:
        for layer_idx in tqdm(
            range(num_layers), desc="AWQ expert cost layers", dynamic_ncols=True
        ):
            layer = layers[layer_idx].to(device)
            policy = Qwen3MoeAWQPolicy.uniform(
                num_experts,
                expert_bits=average_bits,
                attention_bits=attention_bits,
                dense_bits=dense_bits,
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
            layer_costs, layer_counts, moe_metadata = _compute_awq_layer_costs(
                layer,
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                candidate_bits,
                average_bits,
                policy,
                options,
                expert_batch_size,
                device,
            )
            costs[layer_idx] = layer_costs
            counts[layer_idx] = layer_counts
            search_metadata.append(
                {
                    "layer": layer_idx,
                    "attention": attention_metadata,
                    "moe": moe_metadata,
                }
            )

            hidden_batches = _forward_layer_batches(
                layer,
                hidden_batches,
                positional_batches,
                keyword_batches,
                device,
            )
            layers[layer_idx] = layer.to("cpu")
            del (
                attention_inputs,
                moe_input_batches,
                selected_expert_batches,
                routing_weight_batches,
                layer_costs,
                layer_counts,
            )
            _empty_cuda_cache(device)
    finally:
        model.config.use_cache = original_use_cache

    return costs, counts, {
        "search_options": options.to_dict(),
        "attention_bits": attention_bits,
        "dense_bits": dense_bits,
        "layers": search_metadata,
    }
