"""Safe, reusable checkpoints captured after GPTQ and before router fine-tuning."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import uuid
from pathlib import Path


CHECKPOINT_FORMAT_VERSION = 1
METADATA_FILENAME = "gptq_checkpoint_meta.json"
PRUNING_FILENAME = "expert_pruning_map.json"
SUCCESS_FILENAME = "_SUCCESS"
EXPERT_PRUNING_STATES = {"none", "masked", "physical"}
QUANTIZATION_BUFFER_NAMES = {
    "quant_scales",
    "quant_zeros",
    "quant_nbits",
    "quant_groupsize",
}


def _json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor):
    tensor = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_gptq_checkpoint_identity(args, input_ids, attention_mask):
    """Build the exact identity validated when a GPTQ checkpoint is reused."""
    bit_config_sha256 = None
    if getattr(args, "mixed", False):
        if not args.bit_cfg:
            raise ValueError("Mixed-precision checkpointing requires --bit_cfg.")
        bit_config_sha256 = _file_sha256(args.bit_cfg)

    identity = {
        "source_model": args.model,
        "model_name": args.model_name,
        "model_dtype": args.model_dtype,
        "attention_implementation": args.attn_impl,
        "calibration": {
            "dataset": args.calib_dataset,
            "input_ids_sha256": _tensor_sha256(input_ids),
            "attention_mask_sha256": (
                _tensor_sha256(attention_mask) if attention_mask is not None else None
            ),
            "num_samples": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "seed": args.seed,
        },
        "allocation": {
            "mixed": bool(args.mixed),
            "bit_config_sha256": bit_config_sha256,
        },
        "quantization": {
            "quantizer": args.quantizer,
            "reproduce_mcmoe": bool(args.reproduce_mcmoe),
            "attn_wbits": args.attn_wbits,
            "gate_wbits": args.gate_wbits,
            "dense_wbits": args.dense_wbits,
            "expert_wbits": args.expert_wbits,
            "groupsize": args.groupsize,
            "blocksize": args.blocksize,
            "percdamp": args.percdamp,
            "mse": bool(args.mse),
            "actorder": bool(args.actorder),
            "static_groups": bool(args.static_groups),
        },
    }
    return identity


def build_gptq_checkpoint_metadata(
    identity,
    model,
    pruning_metadata=None,
    expert_pruning_state=None,
):
    if expert_pruning_state is None:
        expert_pruning_state = "physical" if pruning_metadata is not None else "none"
    if expert_pruning_state not in EXPERT_PRUNING_STATES:
        raise ValueError(f"Unsupported expert pruning state: {expert_pruning_state!r}")
    if expert_pruning_state == "none" and pruning_metadata is not None:
        raise ValueError("Unpruned GPTQ checkpoint metadata cannot contain a pruning map.")
    if expert_pruning_state != "none" and pruning_metadata is None:
        raise ValueError(
            f"GPTQ checkpoint state {expert_pruning_state!r} requires a pruning map."
        )
    metadata_state = (
        pruning_metadata.get("state") if isinstance(pruning_metadata, dict) else None
    )
    if metadata_state is not None and metadata_state != expert_pruning_state:
        raise ValueError(
            "Pruning-map state differs from GPTQ checkpoint state: "
            f"{metadata_state!r} != {expert_pruning_state!r}."
        )
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "identity_sha256": _json_sha256(identity),
        "identity": identity,
        "artifact": {
            "weight_dtype": str(next(model.parameters()).dtype),
            "format": "huggingface_fake_quant_w_hat",
        },
        "expert_pruning_state": expert_pruning_state,
        "pruning": pruning_metadata,
    }


def _without_quantization_buffers(state_dict):
    return {
        name: tensor
        for name, tensor in state_dict.items()
        if name.rsplit(".", 1)[-1] not in QUANTIZATION_BUFFER_NAMES
    }


def save_gptq_checkpoint(model, tokenizer, checkpoint_path, metadata):
    """Save without mutating the live model and without overwriting an old result."""
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError(
            f"GPTQ checkpoint already exists and will not be overwritten: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=False, exist_ok=False)

    state_dict = _without_quantization_buffers(model.state_dict())
    try:
        tokenizer.save_pretrained(staging)
        model.save_pretrained(staging, state_dict=state_dict)
        with (staging / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False, sort_keys=True)

        pruning_metadata = metadata.get("pruning")
        if pruning_metadata is not None:
            with (staging / PRUNING_FILENAME).open("w", encoding="utf-8") as handle:
                json.dump(
                    pruning_metadata, handle, indent=2, ensure_ascii=False, sort_keys=True
                )

        (staging / SUCCESS_FILENAME).write_text("complete\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(
                f"GPTQ checkpoint appeared while saving and will not be overwritten: {target}"
            )
        staging.rename(target)
    finally:
        del state_dict
        gc.collect()

    print(f"Saved reusable GPTQ checkpoint to: {target}")


def _identity_mismatches(expected, actual, prefix="identity"):
    mismatches = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child_prefix = f"{prefix}.{key}"
            if key not in expected:
                mismatches.append(f"{child_prefix}: unexpected value {actual[key]!r}")
            elif key not in actual:
                mismatches.append(f"{child_prefix}: missing")
            else:
                mismatches.extend(
                    _identity_mismatches(expected[key], actual[key], child_prefix)
                )
        return mismatches
    if expected != actual:
        mismatches.append(f"{prefix}: expected {expected!r}, found {actual!r}")
    return mismatches


def load_gptq_checkpoint_metadata(
    checkpoint_path,
    expected_identity,
    expected_expert_pruning_state=None,
):
    """Validate completeness and identity before loading any checkpoint weights."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GPTQ checkpoint directory not found: {checkpoint}")
    success_path = checkpoint / SUCCESS_FILENAME
    if not success_path.is_file():
        raise RuntimeError(
            f"GPTQ checkpoint is incomplete because {SUCCESS_FILENAME} is missing: {checkpoint}"
        )
    metadata_path = checkpoint / METADATA_FILENAME
    if not metadata_path.is_file():
        raise RuntimeError(f"GPTQ checkpoint metadata is missing: {metadata_path}")
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"GPTQ checkpoint model config is missing: {checkpoint}")
    weight_files = list(checkpoint.glob("*.safetensors"))
    weight_files.extend(checkpoint.glob("pytorch_model*.bin"))
    if not weight_files:
        raise RuntimeError(f"GPTQ checkpoint model weights are missing: {checkpoint}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported GPTQ checkpoint format version: "
            f"{metadata.get('format_version')!r}"
        )
    actual_identity = metadata.get("identity")
    mismatches = _identity_mismatches(expected_identity, actual_identity)
    if mismatches:
        preview = "\n  - ".join(mismatches[:12])
        if len(mismatches) > 12:
            preview += f"\n  - ... and {len(mismatches) - 12} more"
        raise ValueError(
            f"GPTQ checkpoint does not match the current quantization inputs:\n  - {preview}"
        )
    if metadata.get("identity_sha256") != _json_sha256(actual_identity):
        raise ValueError(f"GPTQ checkpoint identity hash is invalid: {metadata_path}")

    pruning_metadata = metadata.get("pruning")
    pruning_state = metadata.get("expert_pruning_state")
    if pruning_state is None:
        # Version-1 checkpoints predate masked pruning and were compacted before
        # GPTQ whenever a pruning map was present.
        pruning_state = "physical" if pruning_metadata is not None else "none"
    if pruning_state not in EXPERT_PRUNING_STATES:
        raise ValueError(
            f"Unsupported GPTQ checkpoint expert pruning state: {pruning_state!r}"
        )
    if pruning_state == "none" and pruning_metadata is not None:
        raise ValueError("Unpruned GPTQ checkpoint unexpectedly contains a pruning map.")
    if pruning_state != "none" and pruning_metadata is None:
        raise ValueError(
            f"GPTQ checkpoint state {pruning_state!r} is missing its pruning map."
        )
    metadata_state = (
        pruning_metadata.get("state") if isinstance(pruning_metadata, dict) else None
    )
    if metadata_state is not None and metadata_state != pruning_state:
        raise ValueError(
            "GPTQ checkpoint pruning-map state differs from its top-level state: "
            f"{metadata_state!r} != {pruning_state!r}."
        )
    metadata["expert_pruning_state"] = pruning_state
    if (
        expected_expert_pruning_state is not None
        and pruning_state != expected_expert_pruning_state
    ):
        raise ValueError(
            "GPTQ checkpoint expert pruning state differs: expected "
            f"{expected_expert_pruning_state!r}, found {pruning_state!r}."
        )
    pruning_path = checkpoint / PRUNING_FILENAME
    if pruning_metadata is not None:
        if not pruning_path.is_file():
            raise RuntimeError(f"GPTQ checkpoint pruning map is missing: {pruning_path}")
        with pruning_path.open("r", encoding="utf-8") as handle:
            stored_pruning_metadata = json.load(handle)
        if stored_pruning_metadata != pruning_metadata:
            raise ValueError(f"GPTQ checkpoint pruning metadata differs: {pruning_path}")

    print(f"Validated reusable GPTQ checkpoint: {checkpoint}")
    return metadata
