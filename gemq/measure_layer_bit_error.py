"""Measure one Qwen3-MoE layer's FP-vs-GPTQ output error at 1, 2, 3, 4 bits.

Only routed expert weights change. The input to the selected MoE, its router,
and every other model weight remain full precision for every candidate bit.
This command saves measurements; plotting is a separate command.
"""

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from gemq.expert_costs import (
    _capture_decoder_inputs,
    _capture_qwen3_moe_inputs,
    _first_tensor,
    _forward_layer_batches,
    _get_qwen3_expert,
)
from gemq.quantizers.gptq import GPTQWeightQuantizer
from gemq.utils.data_utils import get_calib_loader
from gemq.utils.hf_loading import load_causal_lm_checkpoint
from gemq.utils.model_utils import get_blocks, get_moe_block


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EXPERT_LINEARS = ("gate_proj", "up_proj", "down_proj")
BITS = (1, 2, 3, 4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=24, help="zero-based decoder layer")
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--attn_impl", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    for key in ("calib_samples", "eval_samples", "seqlen", "groupsize", "blocksize"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.layer < 0 or args.seed < 0 or args.percdamp < 0:
        parser.error("--layer, --seed and --percdamp must be non-negative")
    if args.output_dir is None:
        args.output_dir = Path("cache/layer_bit_error/Qwen3-30B-A3B-Instruct-2507") / (
            f"L{args.layer}-C4-Cal{args.calib_samples}-Eval{args.eval_samples}"
            f"-Len{args.seqlen}-Seed{args.seed}"
        )
    return args


def hash_input_ids(batches):
    digest = hashlib.sha256()
    for input_ids, _ in batches:
        digest.update(input_ids.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def capture_moe_inputs(model, batches, layer_idx, device):
    """Run an FP decoder prefix and stop before the selected MoE executes."""
    layers = get_blocks(model, MODEL_NAME)
    hidden, positional, keyword = _capture_decoder_inputs(
        model, batches, MODEL_NAME, device
    )
    for index in range(layer_idx):
        layer = layers[index].to(device)
        next_hidden = _forward_layer_batches(
            layer, hidden, positional, keyword, device
        )
        layers[index] = layer.to("cpu")
        del hidden
        hidden = next_hidden
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[FP prefix] layer {index + 1}/{layer_idx}", flush=True)

    layer = layers[layer_idx].to(device)
    moe = get_moe_block(layer, MODEL_NAME)
    inputs, _, _, _ = _capture_qwen3_moe_inputs(
        layer, moe, hidden, positional, keyword, device, stop_at_moe=True
    )
    del hidden, positional, keyword
    gc.collect()
    torch.cuda.empty_cache()
    return moe, inputs


@torch.inference_mode()
def collect_fp_hessians(moe, calibration_inputs, args, device):
    """Collect each expert linear's GPTQ Hessian on unchanged FP activations."""
    entries = []
    handles = []
    for expert_id in range(len(moe.experts)):
        expert = _get_qwen3_expert(moe, expert_id)
        for name in EXPERT_LINEARS:
            module = getattr(expert, name)
            columns = module.weight.shape[1]
            # Match quantize.py's 128 -> 64 fallback for incompatible widths.
            groupsize = args.groupsize if columns % args.groupsize == 0 else 64
            if columns % groupsize:
                raise ValueError(f"Expert {expert_id} {name}: unsupported input width {columns}")
            master = GPTQWeightQuantizer(
                module.weight.data, f"experts.{expert_id}.{name}", 2,
                args.blocksize, args.percdamp, groupsize,
                False, False, True,
            )
            entries.append((expert_id, name, module, master))

            def capture(_module, inputs, _output, quantizer=master):
                quantizer.add_batch(inputs[0].detach())

            handles.append(module.register_forward_hook(capture))

    try:
        for index, hidden in enumerate(calibration_inputs):
            moe(hidden.to(device=device, non_blocking=True))
            if (index + 1) % 16 == 0 or index + 1 == len(calibration_inputs):
                print(f"[GPTQ Hessian] {index + 1}/{len(calibration_inputs)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    missing = [(expert_id, name) for expert_id, name, _, master in entries
               if master.nsamples == 0]
    if missing:
        raise RuntimeError(f"Calibration never activated these expert linears: {missing[:12]}")
    return entries


@torch.inference_mode()
def install_bit_weights(entries, bit, args):
    """Quantize each expert linear from its original W and a fresh H copy."""
    started = time.monotonic()
    total_experts = len(entries) // len(EXPERT_LINEARS)
    for index, (expert_id, name, module, master) in enumerate(entries):
        quantizer = GPTQWeightQuantizer(
            master.W, master.name, bit, args.blocksize, args.percdamp,
            master.groupsize, False, False, True,
        )
        # GPTQ modifies H in place, so each bit must get the same FP H.
        quantizer.H = master.H.clone()
        quantizer.nsamples = master.nsamples
        codes, scales, zeros = quantizer.quantize()
        module.weight.data = quantizer.dequantize(codes, scales, zeros).reshape_as(
            module.weight.data
        )
        del quantizer, codes, scales, zeros
        if (index + 1) % (8 * len(EXPERT_LINEARS)) == 0:
            completed = expert_id + 1
            elapsed = time.monotonic() - started
            projected = elapsed * total_experts / completed
            print(
                f"[GPTQ {bit}-bit] {completed}/{total_experts} experts; "
                f"elapsed={elapsed / 60:.1f} min, projected={projected / 60:.1f} min",
                flush=True,
            )


def restore_fp_weights(entries):
    for _, _, module, master in entries:
        module.weight.data = master.W


def squared_error_sums(reference, candidate):
    """Return SSE and FP output energy using FP32 differences, FP64 reduction."""
    reference = reference.float()
    candidate = candidate.float()
    squared_error = (candidate - reference).square().sum(dtype=torch.float64).item()
    reference_squared = reference.square().sum(dtype=torch.float64).item()
    if reference_squared <= 0:
        raise ValueError("The FP MoE output has zero energy")
    return squared_error, reference_squared


@torch.inference_mode()
def measure_bits(moe, eval_inputs, entries, args, device):
    references = []
    for hidden in eval_inputs:
        reference = _first_tensor(moe(hidden.to(device=device, non_blocking=True)))
        references.append(reference.to("cpu"))

    summary_rows = []
    sequence_rows = []
    for bit in BITS:
        started = time.monotonic()
        try:
            install_bit_weights(entries, bit, args)
            torch.cuda.synchronize(device)
            quantize_seconds = time.monotonic() - started
            started = time.monotonic()
            bit_rows = []
            for sequence, (hidden, fp) in enumerate(zip(eval_inputs, references)):
                candidate = _first_tensor(moe(hidden.to(device=device, non_blocking=True)))
                error, baseline = squared_error_sums(fp.to(device), candidate)
                row = {
                    "bit_width": bit, "sequence": sequence,
                    "tokens": int(hidden.shape[0] * hidden.shape[1]),
                    "squared_error_sum": error,
                    "reference_squared_sum": baseline,
                    "relative_mse": error / baseline,
                }
                bit_rows.append(row)
                print(f"[measure {bit}-bit] {sequence + 1}/{len(eval_inputs)}", flush=True)
            torch.cuda.synchronize(device)
            eval_seconds = time.monotonic() - started
        finally:
            restore_fp_weights(entries)

        sequence_rows.extend(bit_rows)
        total_error = sum(row["squared_error_sum"] for row in bit_rows)
        total_baseline = sum(row["reference_squared_sum"] for row in bit_rows)
        summary_rows.append({
            "bit_width": bit,
            "tokens": sum(row["tokens"] for row in bit_rows),
            "squared_error_sum": total_error,
            "reference_squared_sum": total_baseline,
            "relative_mse": total_error / total_baseline,
            "quantize_seconds": quantize_seconds,
            "evaluate_seconds": eval_seconds,
        })
        print(
            f"[result {bit}-bit] relative MSE={total_error / total_baseline:.8g}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return summary_rows, sequence_rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This measurement requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    device = torch.device("cuda:0")
    run_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = load_causal_lm_checkpoint(
        args.model, model_dtype="bfloat16", attn_implementation=args.attn_impl,
        trust_remote_code=True, device_map="cpu",
    )
    layers = get_blocks(model, MODEL_NAME)
    if args.layer >= len(layers):
        raise ValueError(f"--layer must be less than {len(layers)}")
    selected_moe = get_moe_block(layers[args.layer], MODEL_NAME)
    if selected_moe.top_k != 8 or not selected_moe.norm_topk_prob:
        raise ValueError("Expected Qwen3-MoE top-8 with norm_topk_prob=True")
    if len(selected_moe.experts) != 128:
        raise ValueError("Expected 128 routed experts in the selected Qwen3 layer")
    model.config.use_cache = False

    loader_args = SimpleNamespace(
        calib_dataset="c4", model=args.model,
        nsamples=args.calib_samples + args.eval_samples,
        seqlen=args.seqlen, seed=args.seed, batch_size=1, use_fast=True,
    )
    batches = get_calib_loader(tokenizer, loader_args)
    if len(batches) != loader_args.nsamples:
        raise ValueError(f"Expected {loader_args.nsamples} C4 blocks, found {len(batches)}")
    calib_hash = hash_input_ids(batches[:args.calib_samples])
    eval_hash = hash_input_ids(batches[args.calib_samples:])
    if calib_hash == eval_hash:
        raise RuntimeError("Calibration and evaluation C4 blocks have identical hashes")
    calib_blocks = {hash_input_ids([batch]) for batch in batches[:args.calib_samples]}
    eval_blocks = {hash_input_ids([batch]) for batch in batches[args.calib_samples:]}
    if calib_blocks & eval_blocks:
        raise RuntimeError("Calibration and evaluation C4 blocks overlap")

    moe, all_inputs = capture_moe_inputs(model, batches, args.layer, device)
    if len(all_inputs) != len(batches):
        raise RuntimeError("Did not capture one MoE input for every C4 block")
    prefix_seconds = time.monotonic() - run_started
    print(f"[FP capture] {prefix_seconds / 60:.1f} min", flush=True)

    hessian_started = time.monotonic()
    entries = collect_fp_hessians(
        moe, all_inputs[:args.calib_samples], args, device
    )
    hessian_seconds = time.monotonic() - hessian_started
    del all_inputs[:args.calib_samples]
    gc.collect()
    torch.cuda.empty_cache()
    summary, per_sequence = measure_bits(moe, all_inputs, entries, args, device)

    metadata = {
        "format_version": 1,
        "experiment": "single_layer_uniform_expert_gptq_output_error",
        "model": args.model, "model_name": MODEL_NAME,
        "dataset": "c4", "layer": args.layer,
        "calib_samples": args.calib_samples, "eval_samples": args.eval_samples,
        "seqlen": args.seqlen, "seed": args.seed,
        "calibration_input_ids_sha256": calib_hash,
        "evaluation_input_ids_sha256": eval_hash,
        "bits": list(BITS), "attn_impl": args.attn_impl,
        "gptq": {"groupsize": args.groupsize, "blocksize": args.blocksize,
                 "percdamp": args.percdamp, "mse": True},
        "metric": "sum(||MoE_b(h)-MoE_FP(h)||_2^2) / sum(||MoE_FP(h)||_2^2)",
        "context": "FP prefix, fixed FP router and MoE inputs; only this layer's expert weights change",
        "prefix_and_load_seconds": prefix_seconds,
        "hessian_seconds": hessian_seconds,
        "total_seconds": time.monotonic() - run_started,
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "per_sequence.csv", per_sequence)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved measurements to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
