import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from gemq.router_finetune.transfer import (
    router_statistics_from_cached_layers,
    router_statistics_from_logits,
    stack_router_statistics,
    validate_router_statistics,
)
from gemq.utils.model_utils import (
    compute_decoder_inputs,
    extract_router_logits,
    get_blocks,
    get_router_module,
)


TEACHER_CACHE_VERSION = 1


@dataclass
class TeacherTargets:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    router_logits: tuple[torch.Tensor, ...] | None
    final_hidden_states: torch.Tensor | None
    metadata: dict
    router_stats: dict | None = None

    def router_logits_for_layer(self, layer_idx):
        if self.router_logits is None:
            raise RuntimeError("The teacher cache does not contain router logits.")
        return self.router_logits[layer_idx]


def project_teacher_router_logits(targets, kept_expert_ids):
    """Project original-model router targets onto physically surviving experts.

    The cached teacher artifact remains in the original expert-ID space; projection
    creates an in-memory view for the pruned student and never overwrites the cache.
    """
    if targets is None or targets.router_logits is None:
        return targets
    if len(targets.router_logits) != len(kept_expert_ids):
        raise ValueError(
            f"Teacher has {len(targets.router_logits)} router layers but pruning "
            f"provides {len(kept_expert_ids)} layers."
        )
    projected = []
    for layer_idx, (logits, kept_ids) in enumerate(
        zip(targets.router_logits, kept_expert_ids)
    ):
        if not kept_ids:
            raise ValueError(f"Layer {layer_idx} keeps no experts.")
        if min(kept_ids) < 0 or max(kept_ids) >= logits.shape[-1]:
            raise ValueError(
                f"Layer {layer_idx} kept IDs are outside teacher logits with "
                f"{logits.shape[-1]} experts."
            )
        index = torch.tensor(kept_ids, dtype=torch.long, device=logits.device)
        projected.append(logits.index_select(-1, index).contiguous())
    metadata = dict(targets.metadata)
    metadata["router_logits_projection"] = "kept original expert IDs after pruning"
    return TeacherTargets(
        input_ids=targets.input_ids,
        attention_mask=targets.attention_mask,
        router_logits=tuple(projected),
        final_hidden_states=targets.final_hidden_states,
        metadata=metadata,
        router_stats=targets.router_stats,
    )


def materialize_calibration_inputs(dataloader):
    """Materialize the exact, ordered token tensors used by quantization."""
    input_parts = []
    mask_parts = []
    saw_attention_mask = False
    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        else:
            input_ids = batch[0]
            attention_mask = None
        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        input_parts.append(input_ids)
        if attention_mask is None:
            mask_parts.append(torch.ones_like(input_ids, dtype=torch.bool))
        else:
            saw_attention_mask = True
            mask_parts.append(attention_mask.detach().to(device="cpu", dtype=torch.bool).contiguous())

    if not input_parts:
        raise ValueError("The calibration dataloader is empty.")
    input_ids = torch.cat(input_parts, dim=0).contiguous()
    attention_mask = torch.cat(mask_parts, dim=0).contiguous() if saw_attention_mask else None
    return input_ids, attention_mask


def _tensor_sha256(tensor):
    tensor = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _json_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_model_files_sha256(model_path):
    path = Path(model_path)
    if not path.is_dir():
        return None
    manifests = []
    patterns = ("config.json", "*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin")
    seen = set()
    for pattern in patterns:
        for file_path in path.glob(pattern):
            if not file_path.is_file() or file_path in seen:
                continue
            seen.add(file_path)
            stat = file_path.stat()
            manifests.append(
                {
                    "path": file_path.relative_to(path).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return _json_sha256(sorted(manifests, key=lambda item: item["path"]))


def _build_cache_identity(model, tokenizer, input_ids, attention_mask, args):
    config_dict = model.config.to_dict()
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    return {
        "cache_version": TEACHER_CACHE_VERSION,
        "model": args.model,
        "model_name": args.model_name,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "local_model_files_sha256": _local_model_files_sha256(args.model),
        "model_config_sha256": _json_sha256(config_dict),
        "model_dtype": str(next(model.parameters()).dtype),
        "attention_implementation": args.attn_impl,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__qualname__,
        "tokenizer_config_sha256": _json_sha256(tokenizer_kwargs),
        "transformers_version": transformers.__version__,
        "input_ids_sha256": _tensor_sha256(input_ids),
        "attention_mask_sha256": _tensor_sha256(attention_mask) if attention_mask is not None else None,
        "num_samples": input_ids.shape[0],
        "sequence_length": input_ids.shape[1],
    }


def _cache_path(base_dir, identity):
    identity_hash = _json_sha256(identity)[:24]
    return Path(base_dir) / identity_hash


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_cached_targets(
    cache_path,
    identity,
    input_ids,
    attention_mask,
    config,
    require_all=True,
):
    metadata_path = cache_path / "metadata.json"
    inputs_path = cache_path / "calibration_inputs.pt"
    if not metadata_path.is_file() or not inputs_path.is_file():
        return None

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cached_identity = metadata.get("identity")
    if cached_identity != identity:
        raise ValueError(f"Teacher cache identity mismatch in {cache_path}.")

    cached_inputs = _torch_load(inputs_path)
    if not torch.equal(cached_inputs["input_ids"], input_ids):
        raise ValueError(f"Teacher cache input_ids differ from the current calibration data: {cache_path}")
    cached_mask = cached_inputs.get("attention_mask")
    if (cached_mask is None) != (attention_mask is None):
        raise ValueError(f"Teacher cache attention-mask presence differs: {cache_path}")
    if attention_mask is not None and not torch.equal(cached_mask, attention_mask):
        raise ValueError(f"Teacher cache attention_mask differs from the current calibration data: {cache_path}")

    available = set(metadata.get("available_targets", []))
    required = set()
    if config.needs_router_targets:
        required.add("router_logits")
    if config.needs_output_targets:
        required.add("final_hidden_states")
    if getattr(config, "needs_router_statistics", False):
        required.add("router_correlation")
    if require_all and not required.issubset(available):
        return None

    router_logits = None
    if config.needs_router_targets and "router_logits" in available:
        router_payload = _torch_load(cache_path / "router_logits.pt")
        router_logits = tuple(router_payload["router_logits"])
    final_hidden_states = None
    if config.needs_output_targets and "final_hidden_states" in available:
        final_payload = _torch_load(cache_path / "final_hidden_states.pt")
        final_hidden_states = final_payload["final_hidden_states"]
    router_stats = None
    if (
        getattr(config, "needs_router_statistics", False)
        and "router_correlation" in available
    ):
        router_stats = _torch_load(cache_path / "router_stats.pt")

    if require_all:
        print(f"Loaded validated teacher targets from: {cache_path}")
    return TeacherTargets(
        input_ids=input_ids,
        attention_mask=attention_mask,
        router_logits=router_logits,
        final_hidden_states=final_hidden_states,
        metadata=metadata,
        router_stats=router_stats,
    )


def _atomic_torch_save(payload, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _save_targets(cache_path, identity, targets):
    cache_path.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {"input_ids": targets.input_ids, "attention_mask": targets.attention_mask},
        cache_path / "calibration_inputs.pt",
    )
    # Preserve other valid signal kinds already stored for this exact model/data
    # identity, so an output-only run does not make an earlier router cache invisible.
    available = set()
    metadata_path = cache_path / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)
        if previous_metadata.get("identity") == identity:
            available.update(previous_metadata.get("available_targets", []))
    if targets.router_logits is not None:
        _atomic_torch_save(
            {"router_logits": list(targets.router_logits)}, cache_path / "router_logits.pt"
        )
        available.add("router_logits")
    if targets.final_hidden_states is not None:
        _atomic_torch_save(
            {"final_hidden_states": targets.final_hidden_states},
            cache_path / "final_hidden_states.pt",
        )
        available.add("final_hidden_states")
    if targets.router_stats is not None:
        _atomic_torch_save(targets.router_stats, cache_path / "router_stats.pt")
        available.add("router_correlation")

    metadata = {"identity": identity, "available_targets": sorted(available)}
    temp_path = metadata_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temp_path, metadata_path)
    targets.metadata = metadata
    print(f"Saved teacher targets to: {cache_path}")


def _extract_layer_hidden(layer_output):
    if torch.is_tensor(layer_output):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output and torch.is_tensor(layer_output[0]):
        return layer_output[0]
    raise TypeError(f"Could not extract decoder hidden states from {type(layer_output)!r}.")


@torch.no_grad()
def collect_teacher_targets(
    model,
    dataloader,
    input_ids,
    attention_mask,
    args,
    config,
    collect_router_targets=None,
    collect_output_targets=None,
    collect_router_statistics=None,
):
    """Collect targets from one complete, unquantized layer-wise teacher pass."""
    print("Collecting full-precision teacher targets ...")
    original_training = model.training
    original_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = False

    inps, layer_kwargs = compute_decoder_inputs(model, dataloader, args.model_name, "cuda")
    if inps.shape[:2] != input_ids.shape:
        raise ValueError(
            f"Teacher activations {inps.shape[:2]} do not match calibration inputs {input_ids.shape}."
        )
    layers = get_blocks(model, args.model_name)
    outs = torch.empty_like(inps)
    collect_router_targets = (
        config.needs_router_targets
        if collect_router_targets is None
        else collect_router_targets
    )
    collect_output_targets = (
        config.needs_output_targets
        if collect_output_targets is None
        else collect_output_targets
    )
    collect_router_statistics = (
        getattr(config, "needs_router_statistics", False)
        if collect_router_statistics is None
        else collect_router_statistics
    )
    needs_router_forward = collect_router_targets or collect_router_statistics
    all_router_logits = [] if collect_router_targets else None
    all_router_statistics = [] if collect_router_statistics else None

    for layer_idx, layer in enumerate(layers):
        layer = layer.to("cuda")
        layers[layer_idx] = layer
        captured = []
        handle = None
        if needs_router_forward:
            _, router = get_router_module(layer, args.model_name)

            def capture_router_logits(_module, _inputs, output):
                captured.append(extract_router_logits(output).detach().to("cpu"))

            handle = router.register_forward_hook(capture_router_logits)

        for sample_idx in range(inps.shape[0]):
            output = layer(inps[sample_idx: sample_idx + 1], **layer_kwargs)
            outs[sample_idx] = _extract_layer_hidden(output)

        if handle is not None:
            handle.remove()
            if len(captured) != inps.shape[0]:
                raise RuntimeError(
                    f"Layer {layer_idx} router ran {len(captured)} times for {inps.shape[0]} samples."
                )
            layer_logits = torch.cat(captured, dim=0)
            layer_logits = layer_logits.reshape(input_ids.shape[0], input_ids.shape[1], -1).contiguous()
            if collect_router_targets:
                all_router_logits.append(layer_logits)
            if collect_router_statistics:
                all_router_statistics.append(
                    router_statistics_from_logits(layer_logits, attention_mask)
                )

        inps, outs = outs, inps
        layers[layer_idx] = layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    final_hidden_states = None
    if collect_output_targets:
        norm = model.model.norm.to("cuda")
        final_chunks = []
        for start in range(0, inps.shape[0], config.batch_size):
            final_chunks.append(norm(inps[start: start + config.batch_size]).to("cpu"))
        final_hidden_states = torch.cat(final_chunks, dim=0).contiguous()
        model.model.norm = norm.to("cpu")

    model.config.use_cache = original_use_cache
    model.train(original_training)
    del inps, outs
    gc.collect()
    torch.cuda.empty_cache()
    return TeacherTargets(
        input_ids=input_ids,
        attention_mask=attention_mask,
        router_logits=tuple(all_router_logits) if all_router_logits is not None else None,
        final_hidden_states=final_hidden_states,
        metadata={},
        router_stats=(
            stack_router_statistics(all_router_statistics)
            if all_router_statistics is not None
            else None
        ),
    )


def _validate_target_shapes(targets, model, args, config):
    expected_tokens = tuple(targets.input_ids.shape)
    expected_layers = len(get_blocks(model, args.model_name))
    expected_experts = getattr(model.config, "num_experts", None)
    if config.needs_router_targets:
        if targets.router_logits is None:
            raise ValueError("Teacher targets are missing router logits.")
        if len(targets.router_logits) != expected_layers:
            raise ValueError(
                f"Teacher cache has {len(targets.router_logits)} router layers; "
                f"the model has {expected_layers}."
            )
        for layer_idx, logits in enumerate(targets.router_logits):
            if tuple(logits.shape[:2]) != expected_tokens:
                raise ValueError(
                    f"Layer {layer_idx} teacher router shape {tuple(logits.shape)} does not "
                    f"match calibration inputs {expected_tokens}."
                )
            if expected_experts is not None and logits.shape[-1] != expected_experts:
                raise ValueError(
                    f"Layer {layer_idx} teacher cache has {logits.shape[-1]} experts; "
                    f"the model has {expected_experts}."
                )
    if config.needs_output_targets:
        if targets.final_hidden_states is None:
            raise ValueError("Teacher targets are missing final hidden states.")
        expected_shape = (*expected_tokens, model.config.hidden_size)
        if tuple(targets.final_hidden_states.shape) != expected_shape:
            raise ValueError(
                f"Teacher final hidden shape {tuple(targets.final_hidden_states.shape)} does not "
                f"match {expected_shape}."
            )
    if getattr(config, "needs_router_statistics", False):
        validate_router_statistics(
            targets.router_stats,
            expected_layers=expected_layers,
            expected_experts=expected_experts,
        )


def get_or_collect_teacher_targets(
    model, tokenizer, dataloader, input_ids, attention_mask, args, config
):
    identity = _build_cache_identity(model, tokenizer, input_ids, attention_mask, args)
    cache_path = _cache_path(config.teacher_cache_dir, identity)
    if not config.rebuild_teacher_cache:
        cached = _load_cached_targets(cache_path, identity, input_ids, attention_mask, config)
        if cached is not None:
            _validate_target_shapes(cached, model, args, config)
            return cached

        partial = _load_cached_targets(
            cache_path,
            identity,
            input_ids,
            attention_mask,
            config,
            require_all=False,
        )
        if partial is not None:
            if (
                getattr(config, "needs_router_statistics", False)
                and partial.router_stats is None
                and "router_logits" in partial.metadata.get("available_targets", [])
            ):
                payload = _torch_load(cache_path / "router_logits.pt")
                partial.router_stats = router_statistics_from_cached_layers(
                    tuple(payload["router_logits"]), attention_mask
                )
                print(
                    "Derived router correlation statistics from existing cached "
                    f"router logits: {cache_path}"
                )

            missing_router = config.needs_router_targets and partial.router_logits is None
            missing_output = (
                config.needs_output_targets and partial.final_hidden_states is None
            )
            missing_statistics = (
                getattr(config, "needs_router_statistics", False)
                and partial.router_stats is None
            )
            if not (missing_router or missing_output or missing_statistics):
                _validate_target_shapes(partial, model, args, config)
                _save_targets(cache_path, identity, partial)
                return partial

            collected = collect_teacher_targets(
                model,
                dataloader,
                input_ids,
                attention_mask,
                args,
                config,
                collect_router_targets=missing_router,
                collect_output_targets=missing_output,
                collect_router_statistics=missing_statistics,
            )
            targets = TeacherTargets(
                input_ids=input_ids,
                attention_mask=attention_mask,
                router_logits=(
                    collected.router_logits
                    if missing_router
                    else partial.router_logits
                ),
                final_hidden_states=(
                    collected.final_hidden_states
                    if missing_output
                    else partial.final_hidden_states
                ),
                metadata=partial.metadata,
                router_stats=(
                    collected.router_stats
                    if missing_statistics
                    else partial.router_stats
                ),
            )
            _validate_target_shapes(targets, model, args, config)
            _save_targets(cache_path, identity, targets)
            return targets

    targets = collect_teacher_targets(model, dataloader, input_ids, attention_mask, args, config)
    _validate_target_shapes(targets, model, args, config)
    _save_targets(cache_path, identity, targets)
    return targets
