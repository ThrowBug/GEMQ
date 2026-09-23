import gc
import importlib.metadata
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
except ImportError:
    MixtralSparseMoeBlock = ()

try:
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE
except ImportError:
    try:
        # Transformers 5 renamed this class without changing the model key.
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
            DeepseekV2Moe as DeepseekV2MoE,
        )
    except ImportError:
        DeepseekV2MoE = ()

try:
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
except ImportError:
    OlmoeSparseMoeBlock = ()

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )
except ImportError:
    Qwen3MoeSparseMoeBlock = ()

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )
except ImportError:  # Keeps legacy environments able to import the Qwen3 code.
    Qwen3_5MoeSparseMoeBlock = ()

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory


class ModelType(Enum):
    # Dense models
    LLAMA2 = auto()
    QWEN3 = auto()

    # MoE models
    MIXTRAL = auto()
    DEEPSEEKV2 = auto()
    OLMOE = auto()
    QWEN3MOE = auto()
    QWEN35MOE = auto()
    

class LinearModuleType(Enum):
    ATTN = auto()
    LINEAR_ATTN = auto()
    SOFTMAX_ATTN = auto()
    GATE = auto()
    EXPERT = auto()
    DENSE = auto()
    OTHERS = auto()


NAME_TO_MODEL = {
    "meta-llama/Llama-2-7b-hf": ModelType.LLAMA2,
    "Qwen/Qwen3-8B": ModelType.QWEN3,

    "mistralai/Mixtral-8x7B-v0.1": ModelType.MIXTRAL,
    "deepseek-ai/DeepSeek-V2-Lite": ModelType.DEEPSEEKV2,
    "allenai/OLMoE-1B-7B-0924": ModelType.OLMOE,
    "allenai/OLMoE-1B-7B-0125-Instruct": ModelType.OLMOE,
    "Qwen/Qwen3-30B-A3B": ModelType.QWEN3MOE,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": ModelType.QWEN3MOE,
    "Qwen/Qwen3.5-35B-A3B": ModelType.QWEN35MOE,
}


@dataclass
class ModelInfo:
    num_layers: int
    first_k_dense_layers: int
    num_routed_experts_per_layer: int
    num_shared_experts_per_layer: int
    num_experts_per_token: int
    shared_experts_participate_in_allocation: bool = True

    @property
    def num_allocatable_experts_per_layer(self):
        shared = (
            self.num_shared_experts_per_layer
            if self.shared_experts_participate_in_allocation
            else 0
        )
        return self.num_routed_experts_per_layer + shared


def _format_memory_size(num_bytes):
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _normalize_device_label(device):
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, torch.device):
        return str(device)
    if isinstance(device, str) and device.isdigit():
        return f"cuda:{device}"
    return str(device)


def _compact_integer_ranges(values):
    values = sorted(set(values))
    if not values:
        return "none"

    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _summarize_device_map(device_map):
    if not device_map:
        return

    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    layers_by_device = defaultdict(list)
    other_entries_by_device = defaultdict(list)
    for module_name, device in device_map.items():
        device_label = _normalize_device_label(device)
        match = layer_pattern.search(module_name)
        if match:
            layers_by_device[device_label].append(int(match.group(1)))
        else:
            other_entries_by_device[device_label].append(module_name or "<root>")

    for device_label in sorted(set(layers_by_device) | set(other_entries_by_device)):
        parts = [f"layers={_compact_integer_ranges(layers_by_device[device_label])}"]
        other_entries = other_entries_by_device[device_label]
        if other_entries:
            preview = ", ".join(other_entries[:4])
            if len(other_entries) > 4:
                preview += f", ... (+{len(other_entries) - 4})"
            parts.append(f"other={preview}")
        print(f"  device map {device_label}: {'; '.join(parts)}")


def _tensor_storage_bytes_by_device(named_tensors):
    totals = defaultdict(int)
    seen_storages = set()
    for _, tensor in named_tensors:
        device_label = str(tensor.device)
        if tensor.device.type == "meta":
            storage_key = (device_label, id(tensor))
            num_bytes = tensor.numel() * tensor.element_size()
        else:
            try:
                storage = tensor.untyped_storage()
                storage_key = (device_label, storage.data_ptr(), storage.nbytes())
                num_bytes = storage.nbytes()
            except (RuntimeError, NotImplementedError):
                storage_key = (device_label, id(tensor))
                num_bytes = tensor.numel() * tensor.element_size()
        if storage_key not in seen_storages:
            seen_storages.add(storage_key)
            totals[device_label] += num_bytes
    return totals


def report_cuda_diagnostics(stage, model=None, device_map=None, include_versions=False):
    """Print read-only CUDA memory and model-placement diagnostics."""
    print(f"[CUDA diagnostics] {stage}")
    if include_versions:
        print(
            "  versions: "
            f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
            f"transformers={_package_version('transformers')}, "
            f"accelerate={_package_version('accelerate')}"
        )

    if not torch.cuda.is_available():
        print("  CUDA is unavailable to PyTorch")
    else:
        device_count = torch.cuda.device_count()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"  visible CUDA devices: {device_count} "
            f"(CUDA_VISIBLE_DEVICES={visible_devices})"
        )
        for device_index in range(device_count):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                print(
                    f"  cuda:{device_index} ({torch.cuda.get_device_name(device_index)}): "
                    f"free={_format_memory_size(free_bytes)}, "
                    f"total={_format_memory_size(total_bytes)}, "
                    f"allocated={_format_memory_size(torch.cuda.memory_allocated(device_index))}, "
                    f"reserved={_format_memory_size(torch.cuda.memory_reserved(device_index))}, "
                    f"peak_allocated={_format_memory_size(torch.cuda.max_memory_allocated(device_index))}, "
                    f"peak_reserved={_format_memory_size(torch.cuda.max_memory_reserved(device_index))}"
                )
            except RuntimeError as error:
                print(f"  cuda:{device_index}: unable to query memory ({error})")

    _summarize_device_map(device_map)
    if model is not None:
        parameter_bytes = _tensor_storage_bytes_by_device(model.named_parameters())
        buffer_bytes = _tensor_storage_bytes_by_device(model.named_buffers())
        for device_label in sorted(set(parameter_bytes) | set(buffer_bytes)):
            print(
                f"  model tensors {device_label}: "
                f"parameters={_format_memory_size(parameter_bytes[device_label])}, "
                f"buffers={_format_memory_size(buffer_bytes[device_label])}"
            )


def dispatch_model_to_all_devices(model, cuda_diagnostics=False):
    """
    Dispatch model to all available devices.
    """
    if cuda_diagnostics:
        report_cuda_diagnostics("before model dispatch", model=model)
    print("Dispatching model weights to all devices ... ", end="")
    t0 = time.time()
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "LlamaDecoderLayer",
            "Qwen3DecoderLayer",
            "MixtralDecoderLayer",
            "DeepseekV2DecoderLayer",
            "OlmoeDecoderLayer",
            "Qwen3MoeDecoderLayer",
            "Qwen3_5MoeDecoderLayer",
        ],
        max_memory=get_balanced_memory(model),
    )
    if cuda_diagnostics:
        report_cuda_diagnostics("after inferring the device map", device_map=device_map)
    model = dispatch_model(model, device_map=device_map)
    torch.cuda.synchronize()
    print(f"Done in {(time.time() - t0)/60:.2f} minutes")
    if cuda_diagnostics:
        report_cuda_diagnostics(
            "after model dispatch", model=model, device_map=device_map
        )
    return model


def get_model_info(model_name):
    """
    Get basic model info (#layers, #experts, etc.).

    TODO: Try parsing this information directly from the config file instead of hardcoding it
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        model_info = ModelInfo(
            num_layers=32,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=8,
            num_shared_experts_per_layer=0,
            num_experts_per_token=2,
        )
    elif model_type == ModelType.DEEPSEEKV2:
        model_info = ModelInfo(
            num_layers=27,
            first_k_dense_layers=1,
            num_routed_experts_per_layer=64,
            num_shared_experts_per_layer=2,
            num_experts_per_token=6,
        )
    elif model_type == ModelType.OLMOE:
        model_info = ModelInfo(
            num_layers=16,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=64,
            num_shared_experts_per_layer=0,
            num_experts_per_token=8,
        )
    elif model_type == ModelType.QWEN3MOE:
        model_info = ModelInfo(
            num_layers=48,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=128,
            num_shared_experts_per_layer=0,
            num_experts_per_token=8,
        )
    elif model_type == ModelType.QWEN35MOE:
        model_info = ModelInfo(
            num_layers=40,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=256,
            num_shared_experts_per_layer=1,
            num_experts_per_token=8,
            shared_experts_participate_in_allocation=False,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for getting model info.")

    return model_info


def get_blocks(model, model_name):
    """
    Retrieve a list of decoder blocks (layers).
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3, 
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
        ModelType.QWEN3MOE, ModelType.QWEN35MOE,
    ):
        blocks = model.model.layers
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for getting blocks.")
    return blocks


def move_embed(model, model_name, device):
    """
    Move the embedding layer to the specified device.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3,
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
        ModelType.QWEN3MOE, ModelType.QWEN35MOE,
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if (
            model_type == ModelType.QWEN35MOE
            and hasattr(model.model, "rotary_emb")
        ):
            model.model.rotary_emb = model.model.rotary_emb.to(device)


def move_head(model, model_name, device):
    """
    Move the LM head to the specified device.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3,
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
        ModelType.QWEN3MOE, ModelType.QWEN35MOE,
    ):
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)


def get_named_linears(module):
    """
    Return name-module pairs for linear sub-modules.
    module: a decoder layer
    """
    is_gate = lambda name: name.endswith("gate")
    return {
        name: m for name, m in module.named_modules()
        if (isinstance(m, nn.Linear) or is_gate(name))
    }


def get_moe_block(layer, model_name):
    """
    Get the MoE block from a decoder layer.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        moe_block = layer.block_sparse_moe
    elif model_type == ModelType.DEEPSEEKV2:
        moe_block = layer.mlp
    elif model_type == ModelType.OLMOE:
        moe_block = layer.mlp
    elif model_type == ModelType.QWEN3MOE:
        moe_block = layer.mlp
    elif model_type == ModelType.QWEN35MOE:
        moe_block = layer.mlp
    return moe_block


def get_shared_expert_block(moe_block, model_name):
    """
    Get the shared expert FFN from a moe block.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.DEEPSEEKV2:
        shared_expert = moe_block.shared_experts
    elif model_type == ModelType.QWEN35MOE:
        shared_expert = moe_block.shared_expert
    else:
        raise NotImplementedError(f"Model type {model_type} does not have shared experts.")
    return shared_expert


def get_sublinear_names(model_name):
    """
    Get names of sub-linear modules in a FFN.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        sublinear_names = ["w1", "w2", "w3"]
    elif model_type in (
        ModelType.DEEPSEEKV2,
        ModelType.OLMOE,
        ModelType.QWEN3MOE,
        ModelType.QWEN35MOE,
    ):
        sublinear_names = ["gate_proj", "up_proj", "down_proj"]
    
    return sublinear_names


def get_module_type(module_name, model_name):
    """
    Parse the type of **linear module** based on its name.
    """
    model_type = NAME_TO_MODEL[model_name]

    if model_type == ModelType.LLAMA2:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        else:
            mtype = LinearModuleType.DENSE
    
    elif model_type == ModelType.QWEN3:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        else:
            mtype = LinearModuleType.DENSE

    elif model_type == ModelType.MIXTRAL:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif "gate" in module_name:
            mtype = LinearModuleType.GATE
        elif "experts" in module_name:
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS

    elif model_type == ModelType.DEEPSEEKV2:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif ("gate" in module_name) and ("proj" not in module_name):
            mtype = LinearModuleType.GATE
        elif "mlp" in module_name and ("experts" not in module_name):
            mtype = LinearModuleType.DENSE
        elif ("experts" in module_name) or ("shared_experts" in module_name):
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS
    
    elif model_type == ModelType.OLMOE:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif ("gate" in module_name) and ("proj" not in module_name):
            mtype = LinearModuleType.GATE
        elif ("experts" in module_name):
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS

    elif model_type == ModelType.QWEN3MOE:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif ("gate" in module_name) and ("proj" not in module_name):
            mtype = LinearModuleType.GATE
        elif ("experts" in module_name):
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS

    elif model_type == ModelType.QWEN35MOE:
        if module_name in {"linear_attn.in_proj_qkv", "linear_attn.out_proj"}:
            mtype = LinearModuleType.LINEAR_ATTN
        elif module_name.startswith("linear_attn."):
            # in_proj_z/in_proj_a/in_proj_b and any future auxiliary projections
            # remain full precision by design.
            mtype = LinearModuleType.OTHERS
        elif module_name in {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        }:
            mtype = LinearModuleType.SOFTMAX_ATTN
        elif module_name.startswith("mlp.shared_expert."):
            mtype = LinearModuleType.DENSE
        elif module_name == "mlp.gate":
            mtype = LinearModuleType.GATE
        else:
            # In particular, shared_expert_gate is deliberately kept at full
            # precision and packed routed experts are handled outside this map.
            mtype = LinearModuleType.OTHERS

    return mtype


def get_expert_id(name, model_name):
    """
    Get the expert id from the name of the Linear module.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (ModelType.MIXTRAL, ModelType.OLMOE, ModelType.QWEN3MOE):
        exp_id = int(name.split(".")[-2])
    elif model_type == ModelType.DEEPSEEKV2:
        exp_id = 64 if "shared_experts" in name else int(name.split(".")[-2])

    return exp_id


def get_all_expert_names(model_name):
    """
    Get all expert linear module names (including both routed and shared) in the model.
    """
    model_type = NAME_TO_MODEL[model_name]
    model_info = get_model_info(model_name)
    if model_type == ModelType.MIXTRAL:
        all_expert_names = [f"block_sparse_moe.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)]
    elif model_type == ModelType.DEEPSEEKV2:
        all_expert_names = [f"mlp.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)] + ["mlp.shared_experts"]
    elif model_type == ModelType.OLMOE:
        all_expert_names = [f"mlp.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)]
    elif model_type == ModelType.QWEN3MOE:
        all_expert_names = [f"mlp.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)]
    elif model_type == ModelType.QWEN35MOE:
        # These are logical names. Transformers stores their weights in packed
        # tensors rather than actual child modules.
        all_expert_names = [
            f"mlp.experts.{i}"
            for i in range(model_info.num_routed_experts_per_layer)
        ]

    return all_expert_names


def get_router_params(model, model_name):
    """
    Get all router parameters in the model.
    """
    layers = get_blocks(model, model_name)
    router_params = []
    for layer in layers:
        linears = get_named_linears(layer)
        for name, m in linears.items():
            mtype = get_module_type(name, model_name)
            if mtype == LinearModuleType.GATE:
                for param in m.parameters(): # in case bias exists
                    router_params.append(param)

    return router_params


def get_router_module(layer, model_name):
    """Return the single router module in a decoder layer."""
    routers = []
    for name, module in get_named_linears(layer).items():
        if get_module_type(name, model_name) == LinearModuleType.GATE:
            routers.append((name, module))
    if len(routers) != 1:
        raise ValueError(f"Expected exactly one router in a decoder layer, found {len(routers)}.")
    return routers[0]


def get_router_modules(model, model_name):
    """Return one ``(qualified_name, module)`` pair per decoder layer."""
    routers = []
    for layer_idx, layer in enumerate(get_blocks(model, model_name)):
        name, module = get_router_module(layer, model_name)
        routers.append((f"{layer_idx}.{name}", module))
    return routers


def extract_router_logits(router_output):
    """Normalize router outputs across HF implementations."""
    if torch.is_tensor(router_output):
        return router_output
    if isinstance(router_output, (tuple, list)) and router_output and torch.is_tensor(router_output[0]):
        return router_output[0]
    raise TypeError(f"Could not extract router logits from output type {type(router_output)!r}.")


class _Qwen35DecoderContextCaptured(RuntimeError):
    pass


def _tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    return value


def move_tree_to_device(value, device):
    """Move tensors nested in decoder arguments without changing structure."""
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(move_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: move_tree_to_device(item, device) for key, item in value.items()
        }
    return value


@torch.inference_mode()
def capture_qwen35_decoder_context(model, batches, model_name, device="cuda"):
    """Capture text inputs and the per-layer masks built by the HF model.

    Qwen3.5 alternates linear and full attention, so a single kwargs dictionary
    cannot be shared by every layer. All decoder layers are temporarily replaced
    by identity recorders; this lets the text model build the authoritative mask
    for each layer without loading the vision tower or executing decoder weights.
    """
    if NAME_TO_MODEL[model_name] != ModelType.QWEN35MOE:
        raise ValueError("Per-layer decoder-context capture is Qwen3.5-specific.")
    layers = get_blocks(model, model_name)
    originals = list(layers)
    hidden_batches = []
    positional_templates = [None for _ in originals]
    keyword_templates = [None for _ in originals]
    context_by_layer_type = {}

    class Recorder(nn.Module):
        def __init__(self, module, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            for attribute in ("layer_type", "attention_type"):
                if hasattr(module, attribute):
                    setattr(self, attribute, getattr(module, attribute))

        def forward(self, hidden_states, *args, **kwargs):
            if self.layer_idx == 0:
                hidden_batches.append(_tree_to_cpu(hidden_states))
            # For fixed-length text calibration, Transformers passes identical
            # positions/cache arguments to every layer and one mask per attention
            # type. Keep just the first CPU copy for each type; copying rotary
            # embeddings and a 2048x2048 causal mask for every sample/layer would
            # otherwise consume many gigabytes of host memory.
            context_key = getattr(self, "layer_type", self.layer_idx)
            if context_key not in context_by_layer_type:
                context_by_layer_type[context_key] = (
                    _tree_to_cpu(args),
                    _tree_to_cpu(kwargs),
                )
            positional_templates[self.layer_idx], keyword_templates[self.layer_idx] = (
                context_by_layer_type[context_key]
            )
            if self.layer_idx == len(originals) - 1:
                raise _Qwen35DecoderContextCaptured
            return hidden_states

    move_embed(model, model_name, device)
    try:
        for layer_idx, module in enumerate(originals):
            layers[layer_idx] = Recorder(module, layer_idx)
        for batch in batches:
            input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            try:
                model(input_ids.to(device=device, non_blocking=True))
            except _Qwen35DecoderContextCaptured:
                pass
    finally:
        for layer_idx, module in enumerate(originals):
            layers[layer_idx] = module
        move_embed(model, model_name, "cpu")

    expected = len(hidden_batches)
    if expected == 0:
        raise ValueError("The calibration loader did not produce any batches.")
    for layer_idx in range(len(originals)):
        if (
            positional_templates[layer_idx] is None
            or keyword_templates[layer_idx] is None
        ):
            raise RuntimeError(
                f"Incomplete Qwen3.5 decoder context for layer {layer_idx}."
            )
    positional_by_layer = [
        [template] * expected for template in positional_templates
    ]
    keyword_by_layer = [[template] * expected for template in keyword_templates]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden_batches, positional_by_layer, keyword_by_layer


def compute_decoder_inputs(model, dataloader, model_name, device="cuda"):
    """
    Prepare input data for the first decoder block, and shared kwargs for all blocks.
    """
    layers = get_blocks(model, model_name)
    if NAME_TO_MODEL[model_name] == ModelType.QWEN35MOE:
        hidden, positional, keywords = capture_qwen35_decoder_context(
            model, dataloader, model_name, device
        )
        if any(any(args for args in layer_args) for layer_args in positional):
            raise RuntimeError("Unexpected positional decoder arguments for Qwen3.5.")
        # Calibration batches have a fixed shape, so masks/positions are equal in
        # shape and semantics. Keep one kwargs dictionary per decoder layer.
        layer_kwargs = [layer_keywords[0] for layer_keywords in keywords]
        # All layer-wise callers keep their rolling input/output activation
        # buffers on ``device``.  The recorder stores CPU copies to avoid
        # accumulating them while the model builds contexts, then transfers the
        # single concatenated tensor once after capture.
        return torch.cat(hidden, dim=0).to(
            device=device, non_blocking=True
        ), layer_kwargs

    # get input and kwargs to the first layer decoding layer
    # NOTE: kwargs are shared across all layers
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, device)
    layers[0] = layers[0].to(device)
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

            # NOTE: ad-hoc for Qwen3
            if hasattr(self.module, "attention_type"):
                self.attention_type = self.module.attention_type
            if hasattr(self.module, "layer_type"):
                self.layer_type = self.module.layer_type

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.cat(inps, dim=0)  # (nsamples, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()

    gc.collect()
    torch.cuda.empty_cache()

    return inps, layer_kwargs


def compute_gate_stats_hook_mixtral(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for MixtralSparseMoeBlock block.
    """
    assert isinstance(m, MixtralSparseMoeBlock)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    final_hidden_states = y[0]  # (bsz, seqlen, hidden_size)
    router_logits = y[1]  # (bsz * sequence_length, n_experts)
    
    # compute gate outputs
    # routing_weights:  (batch * sequence_length, topk)
    # selected_experts: (batch * sequence_length, topk)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, m.top_k, dim=-1)

    # compute weights
    actw = torch.zeros(m.num_experts, device=router_logits.device)
    actw.scatter_add_(0, selected_experts.view(-1), routing_weights.view(-1))
    weights.append(actw.to("cpu"))

    # compute counts
    actc = torch.zeros(m.num_experts, dtype=torch.long, device=router_logits.device)
    ones = torch.ones_like(selected_experts.view(-1), device=router_logits.device)
    actc.scatter_add_(0, selected_experts.view(-1), ones)
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_deepseekmoe(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for DeepseekV2MoE block.
    """
    # NOTE: this function should be compatible with deepseekmoe, but only tested on DeepseekV2MoE.
    assert isinstance(m, DeepseekV2MoE)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    device = hidden_states.device

    # NOTE: get weights before renormalization
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # topk_idx:    (bsz*seqlen, topk)
    # topk_weight: (bsz*seqlen, topk)
    hidden_states = hidden_states.view(-1, hidden_dim)     # (bsz*seqlen, hidden_size)
    logits = F.linear(hidden_states, m.gate.weight, None)  # (bsz*seqlen, 64)
    scores = logits.softmax(dim=-1, dtype=torch.float)     # (bsz*seqlen, 64)
    topk_weight, topk_idx = torch.topk(scores, k=m.num_experts_per_tok, dim=-1, sorted=False)

    # =================================
    # for routed experts
    # =================================
    # compute weights
    actw = torch.zeros(m.gate.n_routed_experts, device=device)
    actw.scatter_add_(0, topk_idx.view(-1), topk_weight.view(-1))

    # compute counts
    actc = torch.zeros(m.gate.n_routed_experts, dtype=torch.long, device=device)
    ones = torch.ones_like(topk_idx.view(-1), device=device)
    actc.scatter_add_(0, topk_idx.view(-1), ones)

    # =================================
    # for shared expert
    # =================================
    shared_actw = scores.shape[0]
    shared_actc = scores.shape[0]

    # =================================
    # combine shared and routed experts
    # =================================
    # NOTE: the shared expert is always put at the end
    shared_actw = actw.new_ones(1) * shared_actw
    actw = torch.cat([actw, shared_actw], dim=0)  # (num_experts + 1,)
    shared_actc = actc.new_ones(1) * shared_actc
    actc = torch.cat([actc, shared_actc], dim=0)  # (num_experts + 1,)
    weights.append(actw.to("cpu"))
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_olmoe(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for OlmoeSparseMoeBlock block.
    """
    assert isinstance(m, OlmoeSparseMoeBlock)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    final_hidden_states = y[0]  # (bsz, seqlen, hidden_size)
    router_logits = y[1]  # (bsz * sequence_length, n_experts)
    
    # compute gate outputs
    # routing_weights:  (batch * sequence_length, topk)
    # selected_experts: (batch * sequence_length, topk)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, m.top_k, dim=-1)

    # compute weights
    actw = torch.zeros(m.num_experts, device=router_logits.device)
    actw.scatter_add_(0, selected_experts.view(-1), routing_weights.view(-1))
    weights.append(actw.to("cpu"))

    # compute counts
    actc = torch.zeros(m.num_experts, dtype=torch.long, device=router_logits.device)
    ones = torch.ones_like(selected_experts.view(-1), device=router_logits.device)
    actc.scatter_add_(0, selected_experts.view(-1), ones)
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_qwen3moe(m, x, y, inps, outs, weights, counts):
    assert isinstance(m, Qwen3MoeSparseMoeBlock)
    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    final_hidden_states = y[0]  # (bsz, seqlen, hidden_size)
    router_logits = y[1]  # (bsz * sequence_length, n_experts)
    
    # compute gate outputs
    # routing_weights:  (batch * sequence_length, topk)
    # selected_experts: (batch * sequence_length, topk)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, m.top_k, dim=-1)

    # compute weights
    actw = torch.zeros(m.num_experts, device=router_logits.device)
    actw.scatter_add_(0, selected_experts.view(-1), routing_weights.view(-1))
    weights.append(actw.to("cpu"))

    # compute counts
    actc = torch.zeros(m.num_experts, dtype=torch.long, device=router_logits.device)
    ones = torch.ones_like(selected_experts.view(-1), device=router_logits.device)
    actc.scatter_add_(0, selected_experts.view(-1), ones)
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_qwen35moe(m, x, y, inps, outs, weights, counts):
    if Qwen3_5MoeSparseMoeBlock and not isinstance(m, Qwen3_5MoeSparseMoeBlock):
        raise TypeError(f"Expected Qwen3_5MoeSparseMoeBlock, got {type(m)!r}.")
    from gemq.utils.qwen35 import qwen35_topk_routes

    selected_experts, routing_weights = qwen35_topk_routes(m, x[0])
    num_experts = int(m.experts.num_experts)
    actw = torch.zeros(num_experts, device=routing_weights.device)
    actw.scatter_add_(0, selected_experts.reshape(-1), routing_weights.reshape(-1))
    actc = torch.zeros(num_experts, dtype=torch.long, device=routing_weights.device)
    actc.scatter_add_(
        0,
        selected_experts.reshape(-1),
        torch.ones_like(selected_experts.reshape(-1)),
    )
    weights.append(actw.to("cpu"))
    counts.append(actc.to("cpu"))
    if inps is not None:
        inps.append(x[0])
    if outs is not None:
        outs.append(y[0] if isinstance(y, (tuple, list)) else y)


def get_gate_stats_hook_fn(model_name):
    """
    Get the appropriate hook function for computing router statistics based on model type.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        hook_fn = compute_gate_stats_hook_mixtral
    elif model_type == ModelType.DEEPSEEKV2:
        hook_fn = compute_gate_stats_hook_deepseekmoe
    elif model_type == ModelType.OLMOE:
        hook_fn = compute_gate_stats_hook_olmoe
    elif model_type == ModelType.QWEN3MOE:
        hook_fn = compute_gate_stats_hook_qwen3moe
    elif model_type == ModelType.QWEN35MOE:
        hook_fn = compute_gate_stats_hook_qwen35moe
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for gate stats computation.")

    return hook_fn
