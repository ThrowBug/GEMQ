import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi


MIXED_CALIB_CACHE_VERSION = 1
MIXED_CHAT_EN_WEIGHTS = {
    "wildchat": 0.50,
    "ultrachat": 0.40,
    "fineweb_edu": 0.10,
}


@dataclass(frozen=True)
class StreamingSourceSpec:
    repo_id: str
    split: str
    config_name: str | None
    seed_offset: int
    shuffle_buffer_size: int


SOURCE_SPECS = {
    "wildchat": StreamingSourceSpec(
        repo_id="allenai/WildChat",
        split="train",
        config_name=None,
        seed_offset=1009,
        shuffle_buffer_size=2048,
    ),
    "ultrachat": StreamingSourceSpec(
        repo_id="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        config_name=None,
        seed_offset=2003,
        shuffle_buffer_size=2048,
    ),
    "fineweb_edu": StreamingSourceSpec(
        repo_id="HuggingFaceFW/fineweb-edu",
        split="train",
        config_name="sample-10BT",
        seed_offset=3001,
        shuffle_buffer_size=512,
    ),
}


def _json_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(tensor):
    tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def allocate_source_blocks(nsamples, weights=None):
    """Allocate full, fixed-length blocks with the largest-remainder method."""
    weights = MIXED_CHAT_EN_WEIGHTS if weights is None else weights
    if nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if not weights:
        raise ValueError("At least one source weight is required.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Source weights must be non-negative.")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Source weights must sum to a positive value.")

    normalized = {name: weight / total_weight for name, weight in weights.items()}
    raw_counts = {name: nsamples * weight for name, weight in normalized.items()}
    counts = {name: math.floor(value) for name, value in raw_counts.items()}
    remaining = nsamples - sum(counts.values())
    source_order = {name: index for index, name in enumerate(weights)}
    ranked = sorted(
        weights,
        key=lambda name: (-(raw_counts[name] - counts[name]), source_order[name]),
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def is_english_wildchat(example):
    """Use WildChat's row- and message-level metadata for strict English filtering."""
    if example.get("language") != "English":
        return False
    conversation = example.get("conversation") or []
    relevant_messages = [
        message for message in conversation
        if message.get("role") in {"user", "assistant"}
    ]
    if not relevant_messages:
        return False
    return all(message.get("language") == "English" for message in relevant_messages)


def _normalize_messages(messages):
    normalized = []
    for message in messages or []:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content})

    if normalized and normalized[0]["role"] == "system":
        dialogue = normalized[1:]
    else:
        dialogue = normalized
    if len(dialogue) < 2 or dialogue[0]["role"] != "user" or dialogue[-1]["role"] != "assistant":
        return None
    expected = "user"
    for message in dialogue:
        if message["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    return normalized


def _as_token_list(token_ids):
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one tokenized example, but received a batch.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _append_eos(token_ids, eos_token_id):
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id for calibration packing.")
    if not token_ids or token_ids[-1] != eos_token_id:
        token_ids.append(eos_token_id)
    return token_ids


def _record_id(source_name, example):
    if source_name == "wildchat":
        return str(example.get("conversation_id", ""))
    if source_name == "ultrachat":
        return str(example.get("prompt_id", ""))
    if source_name == "fineweb_edu":
        return str(example.get("id") or example.get("url") or "")
    raise ValueError(f"Unsupported mixed calibration source: {source_name}")


def tokenize_source_record(source_name, example, tokenizer):
    """Return token IDs and a content fingerprint without retaining raw source text."""
    if source_name == "wildchat":
        messages = _normalize_messages(example.get("conversation"))
        if messages is None:
            return None
        canonical_content = messages
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    elif source_name == "ultrachat":
        messages = _normalize_messages(example.get("messages"))
        if messages is None:
            return None
        canonical_content = messages
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    elif source_name == "fineweb_edu":
        text = example.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        canonical_content = text
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        raise ValueError(f"Unsupported mixed calibration source: {source_name}")

    token_ids = _append_eos(_as_token_list(token_ids), tokenizer.eos_token_id)
    return {
        "record_id": _record_id(source_name, example),
        "content_sha256": _json_sha256(canonical_content),
        "token_ids": token_ids,
    }


def pack_source_blocks(source_name, examples, tokenizer, target_blocks, seqlen, seed):
    if target_blocks < 0:
        raise ValueError("target_blocks must be non-negative.")
    if seqlen <= 0:
        raise ValueError("seqlen must be positive.")
    if target_blocks == 0:
        return [], [], {"examined": 0, "skipped": 0, "selected": 0}

    rng = random.Random(seed)
    token_buffer = []
    blocks = []
    selected_records = []
    examined = 0
    skipped = 0

    for example in examples:
        examined += 1
        record = tokenize_source_record(source_name, example, tokenizer)
        if record is None:
            skipped += 1
            continue

        token_ids = record.pop("token_ids")
        original_token_count = len(token_ids)
        crop_start = 0
        if len(token_ids) > seqlen:
            # Preserve the chat prefix; use a seeded random window for long raw documents.
            if source_name == "fineweb_edu":
                crop_start = rng.randint(0, len(token_ids) - seqlen)
            token_ids = token_ids[crop_start:crop_start + seqlen]

        selected_records.append(
            {
                **record,
                "original_token_count": original_token_count,
                "used_token_count": len(token_ids),
                "crop_start": crop_start,
            }
        )
        token_buffer.extend(token_ids)
        while len(token_buffer) >= seqlen and len(blocks) < target_blocks:
            blocks.append(token_buffer[:seqlen])
            del token_buffer[:seqlen]
        if len(blocks) == target_blocks:
            break

    if len(blocks) != target_blocks:
        raise RuntimeError(
            f"Streaming source {source_name!r} ended after producing {len(blocks)} of "
            f"the requested {target_blocks} blocks."
        )
    stats = {
        "examined": examined,
        "skipped": skipped,
        "selected": len(selected_records),
        "discarded_tail_tokens": len(token_buffer),
    }
    return blocks, selected_records, stats


def resolve_dataset_revisions(requested_revisions=None):
    requested_revisions = requested_revisions or {}
    api = HfApi()
    resolved = {}
    for source_name, spec in SOURCE_SPECS.items():
        requested = requested_revisions.get(source_name, "main")
        info = api.dataset_info(spec.repo_id, revision=requested)
        resolved[source_name] = info.sha
    return resolved


def load_streaming_sources(seed, resolved_revisions):
    sources = {}
    for source_name, spec in SOURCE_SPECS.items():
        kwargs = {
            "path": spec.repo_id,
            "split": spec.split,
            "streaming": True,
            "revision": resolved_revisions[source_name],
        }
        if spec.config_name is not None:
            kwargs["name"] = spec.config_name
        dataset = load_dataset(**kwargs)
        if source_name == "wildchat":
            dataset = dataset.filter(is_english_wildchat)
        dataset = dataset.shuffle(
            seed=seed + spec.seed_offset,
            buffer_size=spec.shuffle_buffer_size,
        )
        sources[source_name] = dataset
    return sources


def build_mixed_chat_en_calibration(
    tokenizer,
    sources,
    nsamples,
    seqlen,
    seed,
    dataset_revisions=None,
):
    block_counts = allocate_source_blocks(nsamples)
    source_blocks = []
    selected_records = {}
    source_stats = {}

    for source_name, target_blocks in block_counts.items():
        blocks, records, stats = pack_source_blocks(
            source_name=source_name,
            examples=sources[source_name],
            tokenizer=tokenizer,
            target_blocks=target_blocks,
            seqlen=seqlen,
            seed=seed + SOURCE_SPECS[source_name].seed_offset,
        )
        source_blocks.extend((source_name, block) for block in blocks)
        selected_records[source_name] = records
        source_stats[source_name] = stats

    random.Random(seed).shuffle(source_blocks)
    input_ids = torch.tensor([block for _, block in source_blocks], dtype=torch.long)
    if tuple(input_ids.shape) != (nsamples, seqlen):
        raise RuntimeError(
            f"Prepared calibration tensor has shape {tuple(input_ids.shape)}, expected "
            f"{(nsamples, seqlen)}."
        )

    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    metadata = {
        "cache_version": MIXED_CALIB_CACHE_VERSION,
        "dataset_name": "mixed_chat_en",
        "seed": seed,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "source_weights": MIXED_CHAT_EN_WEIGHTS,
        "source_blocks": block_counts,
        "block_sources_after_shuffle": [source for source, _ in source_blocks],
        "source_sampling": {
            source_name: {
                "repo_id": spec.repo_id,
                "config_name": spec.config_name,
                "split": spec.split,
                "shuffle_seed": seed + spec.seed_offset,
                "shuffle_buffer_size": spec.shuffle_buffer_size,
            }
            for source_name, spec in SOURCE_SPECS.items()
        },
        "dataset_revisions": dataset_revisions or {},
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__qualname__,
        "tokenizer_revision": tokenizer_kwargs.get("_commit_hash"),
        "tokenizer_config_sha256": _json_sha256(tokenizer_kwargs),
        "chat_template_sha256": _json_sha256(getattr(tokenizer, "chat_template", None)),
        "selected_records": selected_records,
        "source_stats": source_stats,
        "input_ids_sha256": tensor_sha256(input_ids),
    }
    return input_ids, metadata


def _atomic_torch_save(payload, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def save_prepared_calibration(input_ids, metadata, output_path):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_hash = tensor_sha256(input_ids)
    if metadata.get("input_ids_sha256") != expected_hash:
        raise ValueError("Calibration metadata does not match the input_ids tensor.")
    _atomic_torch_save(
        {"input_ids": input_ids.to(device="cpu", dtype=torch.long), "input_ids_sha256": expected_hash},
        output_path,
    )
    temp_metadata_path = metadata_path.with_suffix(".json.tmp")
    with temp_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temp_metadata_path, metadata_path)
    return output_path, metadata_path


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_prepared_calibration(
    output_path,
    expected_nsamples=None,
    expected_seqlen=None,
    expected_seed=None,
    tokenizer=None,
):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    if not output_path.is_file():
        raise FileNotFoundError(f"Prepared calibration tensor not found: {output_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared calibration metadata not found: {metadata_path}")

    payload = _torch_load(output_path)
    input_ids = payload["input_ids"].to(device="cpu", dtype=torch.long).contiguous()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    actual_hash = tensor_sha256(input_ids)
    stored_hash = payload.get("input_ids_sha256")
    if stored_hash != actual_hash or metadata.get("input_ids_sha256") != actual_hash:
        raise ValueError(f"Prepared calibration cache hash mismatch: {output_path}")
    if metadata.get("dataset_name") != "mixed_chat_en":
        raise ValueError(f"Unexpected calibration dataset in {metadata_path}")
    if metadata.get("cache_version") != MIXED_CALIB_CACHE_VERSION:
        raise ValueError(f"Unsupported calibration cache version in {metadata_path}")
    if expected_nsamples is not None and input_ids.shape[0] != expected_nsamples:
        raise ValueError(
            f"Calibration cache has {input_ids.shape[0]} samples, expected {expected_nsamples}."
        )
    if expected_seqlen is not None and input_ids.shape[1] != expected_seqlen:
        raise ValueError(
            f"Calibration cache sequence length is {input_ids.shape[1]}, expected {expected_seqlen}."
        )
    if expected_seed is not None and metadata.get("seed") != expected_seed:
        raise ValueError(
            f"Calibration cache seed is {metadata.get('seed')}, expected {expected_seed}."
        )
    if tokenizer is not None:
        tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
        expected_identity = {
            "tokenizer_class": tokenizer.__class__.__qualname__,
            "tokenizer_config_sha256": _json_sha256(tokenizer_kwargs),
            "chat_template_sha256": _json_sha256(getattr(tokenizer, "chat_template", None)),
        }
        for key, expected_value in expected_identity.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Calibration cache {key} differs from the current tokenizer: {output_path}"
                )
    return input_ids, metadata
