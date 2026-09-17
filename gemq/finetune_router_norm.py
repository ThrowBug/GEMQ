"""Post-GPTQ pruning-aware router/norm reconstruction for Qwen3-MoE.

Run this as a separate process from quantization so both the full-precision
teacher and the saved, physically pruned GPTQ student can stay on CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from gemq.pruning import kept_expert_ids_from_pruning_metadata
from gemq.router_finetune.router_norm_reconstruction import (
    ReconstructionConfig,
    finetune_router_norm_reconstruction,
)
from gemq.router_finetune.targets import materialize_calibration_inputs
from gemq.utils.data_utils import build_calib_loader, get_calib_loader
from gemq.utils.eval_utils import compute_perplexity_offload
from gemq.utils.gptq_checkpoint import (
    build_gptq_checkpoint_identity,
    load_gptq_checkpoint_metadata,
    METADATA_FILENAME,
    PRUNING_FILENAME,
)
from gemq.utils.hf_loading import load_causal_lm_checkpoint
from gemq.utils.model_utils import (
    compute_decoder_inputs,
    get_blocks,
    get_moe_block,
    get_router_module,
    move_head,
)


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--gptq_checkpoint_path", required=True)
    parser.add_argument("--bit_cfg", default="")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--calib_dataset", default="c4", choices=["c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--val_nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_prune_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_fast", action="store_true")
    parser.add_argument("--model_dtype", default="bfloat16", choices=["bfloat16"])
    parser.add_argument("--attn_impl", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--rft_trainer", default="router_norm_reconstruction",
                        choices=["router_norm_reconstruction"])
    parser.add_argument("--rft_stage", default="norm_only",
                        choices=["norm_only", "norm_then_router", "decoupled_joint"])
    parser.add_argument("--rft_epochs", type=int, default=1)
    parser.add_argument("--rft_batch_size", type=int, default=1)
    parser.add_argument("--rft_norm_lr", type=float, default=1e-4)
    parser.add_argument("--rft_router_lr", type=float, default=1e-5)
    return parser.parse_args()


def _read_checkpoint_identity(checkpoint_path):
    metadata_path = Path(checkpoint_path) / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing GPTQ checkpoint metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    identity = metadata.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("GPTQ checkpoint metadata contains no identity")
    return identity


def validate_checkpoint(args, input_ids, attention_mask):
    """Rebuild quantization identity from current tokens and the recorded settings."""
    recorded = _read_checkpoint_identity(args.gptq_checkpoint_path)
    quant = recorded["quantization"]
    if str(quant["quantizer"]).lower().split("-")[0] != "gptq":
        raise ValueError("This trainer requires a post-GPTQ checkpoint")
    allocation = recorded["allocation"]
    if allocation["mixed"] and not args.bit_cfg:
        raise ValueError("Mixed-precision GPTQ checkpoint requires --bit_cfg for hash validation")
    identity_args = SimpleNamespace(
        model=args.model,
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        attn_impl=args.attn_impl,
        calib_dataset=args.calib_dataset,
        seed=args.seed,
        mixed=allocation["mixed"],
        bit_cfg=args.bit_cfg,
        **quant,
    )
    expected = build_gptq_checkpoint_identity(identity_args, input_ids, attention_mask)
    metadata = load_gptq_checkpoint_metadata(args.gptq_checkpoint_path, expected)
    if metadata.get("artifact", {}).get("format") != "huggingface_fake_quant_w_hat":
        raise ValueError("This trainer requires an unpacked fake-quant GPTQ checkpoint")
    return metadata


def validate_pruning_shapes(teacher, student, model_name, pruning, max_prune_ratio):
    if not 0.0 <= max_prune_ratio <= 1.0:
        raise ValueError("--max_prune_ratio must be in [0, 1]")
    teacher_layers = get_blocks(teacher, model_name)
    student_layers = get_blocks(student, model_name)
    if len(teacher_layers) != len(student_layers):
        raise ValueError("Teacher and student have different layer counts")
    if pruning is None:
        for index, (teacher_layer, student_layer) in enumerate(zip(teacher_layers, student_layers)):
            if len(get_moe_block(teacher_layer, model_name).experts) != len(
                get_moe_block(student_layer, model_name).experts
            ):
                raise ValueError(f"Layer {index}: expert counts differ without pruning metadata")
            _, gate = get_router_module(student_layer, model_name)
            if gate.weight.shape[0] != len(get_moe_block(student_layer, model_name).experts):
                raise ValueError(f"Layer {index}: router output width differs from expert count")
        return
    kept = kept_expert_ids_from_pruning_metadata(pruning)
    if len(kept) != len(student_layers):
        raise ValueError("Pruning map layer count differs from the student")
    original_count = pruning["original_num_experts"]
    pruned_count = pruning["pruned_experts_per_layer"]
    if pruned_count / original_count > max_prune_ratio + 1e-12:
        raise ValueError(
            f"Checkpoint pruning ratio {pruned_count / original_count:.4f} exceeds "
            f"--max_prune_ratio={max_prune_ratio}"
        )
    for index, (teacher_layer, student_layer, old_ids) in enumerate(
        zip(teacher_layers, student_layers, kept)
    ):
        teacher_count = len(get_moe_block(teacher_layer, model_name).experts)
        student_count = len(get_moe_block(student_layer, model_name).experts)
        if teacher_count != pruning["original_num_experts"] or student_count != len(old_ids):
            raise ValueError(f"Layer {index}: checkpoint/pruning expert shape mismatch")
        _, gate = get_router_module(student_layer, model_name)
        if gate.weight.shape[0] != student_count:
            raise ValueError(f"Layer {index}: router output width differs from expert count")


@torch.no_grad()
def _first_layer_inputs(student, loader, model_name):
    inputs, kwargs = compute_decoder_inputs(student, loader, model_name, "cuda")
    cpu_inputs = inputs.detach().to("cpu")
    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return cpu_inputs, kwargs


@torch.no_grad()
def holdout_perplexity(student, model_name, validation_ids, label):
    value = compute_perplexity_offload(
        student, model_name, validation_ids.reshape(1, -1), label
    )
    move_head(student, model_name, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return value


def save_without_overwriting(student, tokenizer, save_path, metadata, config, result):
    target = Path(save_path)
    if os.path.lexists(target):
        raise FileExistsError(f"Output already exists and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir()
    tokenizer.save_pretrained(staging)
    student.save_pretrained(staging)
    if metadata.get("pruning") is not None:
        with (staging / PRUNING_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(metadata["pruning"], handle, indent=2, ensure_ascii=False)
    with (staging / "router_norm_reconstruction.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"trainer": "router_norm_reconstruction", "source_gptq_identity_sha256":
             metadata["identity_sha256"], "config": vars(config), "result": result},
            handle, indent=2, ensure_ascii=False,
        )
    if os.path.lexists(target):
        raise FileExistsError(f"Output appeared during save: {target}")
    staging.rename(target)
    print(f"Saved router/norm-reconstructed fake-quant model to: {target}")


def main():
    args = parse_args()
    if args.model_name != MODEL_NAME:
        raise ValueError(f"This trainer currently supports only {MODEL_NAME}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 30B layerwise trainer")
    if args.nsamples < 1 or args.val_nsamples < 1 or args.seqlen < 2:
        raise ValueError("nsamples, val_nsamples and seqlen must be positive")
    output = Path(args.save_path)
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    checkpoint_path = Path(args.gptq_checkpoint_path).resolve()
    resolved_output = output.resolve()
    if resolved_output == checkpoint_path or checkpoint_path in resolved_output.parents:
        raise ValueError("The output must not be the GPTQ checkpoint or inside it")
    source_path = Path(args.model)
    if source_path.is_dir() and source_path.resolve() in resolved_output.parents:
        raise ValueError("The output must not be inside the full-precision source model")
    config = ReconstructionConfig(
        args.rft_stage, args.rft_epochs, args.rft_batch_size,
        args.rft_norm_lr, args.rft_router_lr,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast, trust_remote_code=args.trust_remote_code
    )
    train_loader = get_calib_loader(tokenizer, args)
    train_ids, train_mask = materialize_calibration_inputs(train_loader)
    metadata = validate_checkpoint(args, train_ids, train_mask)
    val_loader = build_calib_loader(
        "c4", tokenizer, args.seqlen, args.val_nsamples, args.batch_size,
        num_workers=4, seed=args.seed, document_offset=args.nsamples * 16,
    )
    val_ids = torch.cat([batch["input_ids"] for batch in val_loader], dim=0)
    if val_ids.shape != (args.val_nsamples, args.seqlen):
        raise ValueError(f"Unexpected holdout shape: {tuple(val_ids.shape)}")

    print("Loading full-precision teacher on CPU ...")
    teacher = load_causal_lm_checkpoint(
        args.model, model_dtype=args.model_dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code, device_map="cpu",
    )
    print("Loading post-GPTQ student on CPU ...")
    student = load_causal_lm_checkpoint(
        args.gptq_checkpoint_path, model_dtype=args.model_dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code, device_map="cpu",
    )
    teacher.config.use_cache = False
    original_use_cache = student.config.use_cache
    student.config.use_cache = False
    student.seqlen = args.seqlen
    validate_pruning_shapes(
        teacher, student, args.model_name, metadata.get("pruning"), args.max_prune_ratio
    )

    print("Measuring baseline on disjoint C4 holdout ...")
    baseline_ppl = holdout_perplexity(student, args.model_name, val_ids, "holdout C4 baseline")
    train_inputs, layer_kwargs = _first_layer_inputs(student, train_loader, args.model_name)
    validation_loader = [(val_ids[index:index + 1], None) for index in range(val_ids.shape[0])]
    validation_inputs, _ = _first_layer_inputs(student, validation_loader, args.model_name)
    finetune_router_norm_reconstruction(
        teacher, student, train_inputs, validation_inputs, layer_kwargs,
        args.model_name, config,
    )
    del teacher, train_inputs, validation_inputs
    gc.collect()
    print("Measuring reconstructed student on the same holdout ...")
    candidate_ppl = holdout_perplexity(student, args.model_name, val_ids, "holdout C4 candidate")
    print(f"Holdout C4 PPL: {baseline_ppl:.4f} -> {candidate_ppl:.4f} (diagnostic only)")
    student.config.use_cache = original_use_cache
    save_without_overwriting(
        student, tokenizer, args.save_path, metadata, config,
        {"baseline_ppl": baseline_ppl, "candidate_ppl": candidate_ppl},
    )


if __name__ == "__main__":
    main()
