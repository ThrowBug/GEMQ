"""Measure cumulative decoder-output error for uniform 1/2/3/4-bit routed experts.

GPTQ Hessians are collected on the same full-precision C4 calibration path for
all four bit widths. Evaluation is different: each bit-width path receives its
own preceding layer output, so its error and routing changes can propagate.
Attention, dense MLPs, routers, and other non-routed-expert weights stay FP.
"""

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from gemq.expert_costs import (
    _capture_decoder_inputs,
    _capture_qwen3_moe_inputs,
    _forward_layer_batches,
)
from gemq.measure_layer_bit_error import (
    MODEL_NAME,
    collect_fp_hessians,
    hash_input_ids,
    install_bit_weights,
    restore_fp_weights,
    squared_error_sums,
)
from gemq.utils.data_utils import get_calib_loader
from gemq.utils.hf_loading import load_causal_lm_checkpoint
from gemq.utils.model_utils import get_blocks, get_moe_block


BITS = (1, 2, 3, 4)
CSV_COLUMNS = (
    "layer", "bit_width", "tokens", "squared_error_sum",
    "reference_squared_sum", "relative_mse",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
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
    if args.seed < 0 or not math.isfinite(args.percdamp) or args.percdamp < 0:
        parser.error("--seed and --percdamp must be non-negative and finite")
    if args.output_dir is None:
        args.output_dir = Path("cache/cumulative_bit_error/Qwen3-30B-A3B-Instruct-2507") / (
            f"C4-Cal{args.calib_samples}-Eval{args.eval_samples}"
            f"-Len{args.seqlen}-Seed{args.seed}"
        )
    return args


def validate_routed_moe(moe, layer_index):
    """Dense-only decoder layers are allowed, but routed layers must match Qwen3."""
    if not hasattr(moe, "experts"):
        return False
    if len(moe.experts) != 128 or moe.top_k != 8 or not moe.norm_topk_prob:
        raise ValueError(f"Layer {layer_index}: expected 128 routed experts and normalized top-8")
    return True


def error_row(layer_index, bit, reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("FP and quantized evaluation paths have different batch counts")
    error_total = 0.0
    reference_total = 0.0
    tokens = 0
    for fp_hidden, quant_hidden in zip(reference, candidate):
        if fp_hidden.shape != quant_hidden.shape:
            raise ValueError("FP and quantized layer outputs have different shapes")
        error, reference_energy = squared_error_sums(fp_hidden, quant_hidden)
        error_total += error
        reference_total += reference_energy
        tokens += int(fp_hidden.shape[0] * fp_hidden.shape[1])
    if not math.isfinite(error_total) or not math.isfinite(reference_total) or reference_total <= 0:
        raise ValueError(f"Layer {layer_index}, {bit}-bit: non-finite or zero error denominator")
    return {
        "layer": layer_index,
        "bit_width": bit,
        "tokens": tokens,
        "squared_error_sum": error_total,
        "reference_squared_sum": reference_total,
        "relative_mse": error_total / reference_total,
    }


@torch.inference_mode()
def measure_layers(model, batches, args, device):
    layers = get_blocks(model, MODEL_NAME)
    hidden, positional, keyword = _capture_decoder_inputs(
        model, batches, MODEL_NAME, device
    )
    if len(hidden) != len(batches):
        raise RuntimeError("Did not capture one decoder input for every C4 block")
    fp_calib = hidden[:args.calib_samples]
    fp_eval = hidden[args.calib_samples:]
    calib_positional = positional[:args.calib_samples]
    eval_positional = positional[args.calib_samples:]
    calib_keyword = keyword[:args.calib_samples]
    eval_keyword = keyword[args.calib_samples:]
    del hidden, positional, keyword
    bit_eval = {bit: list(fp_eval) for bit in BITS}
    rows = []
    routed_layers = []
    unactivated_experts = {}

    for index, layer in enumerate(layers):
        started = time.monotonic()
        layer.to(device)
        entries = None
        try:
            moe = get_moe_block(layer, MODEL_NAME)
            routed = validate_routed_moe(moe, index)
            if routed:
                captured = _capture_qwen3_moe_inputs(
                    layer, moe, fp_calib, calib_positional, calib_keyword,
                    device, stop_at_moe=True,
                )
                moe_inputs = captured[0]
                del captured
                entries = collect_fp_hessians(
                    moe, moe_inputs, args, device, allow_unactivated=True
                )
                del moe_inputs
                missing = sorted({expert_id for expert_id, _, _, master in entries
                                  if master.nsamples == 0})
                if missing:
                    unactivated_experts[index] = missing
                    print(f"[layer {index}] unactivated expert IDs: {missing}", flush=True)
                routed_layers.append(index)

            next_fp_calib = _forward_layer_batches(
                layer, fp_calib, calib_positional, calib_keyword, device
            )
            next_fp_eval = _forward_layer_batches(
                layer, fp_eval, eval_positional, eval_keyword, device
            )
            for bit in BITS:
                if routed:
                    try:
                        install_bit_weights(entries, bit, args)
                        next_bit_eval = _forward_layer_batches(
                            layer, bit_eval[bit], eval_positional, eval_keyword, device
                        )
                    finally:
                        restore_fp_weights(entries)
                else:
                    next_bit_eval = _forward_layer_batches(
                        layer, bit_eval[bit], eval_positional, eval_keyword, device
                    )
                row = error_row(index, bit, next_fp_eval, next_bit_eval)
                rows.append(row)
                bit_eval[bit] = next_bit_eval
                print(
                    f"[layer {index} | {bit}-bit] cumulative relative MSE="
                    f"{row['relative_mse']:.8g}", flush=True,
                )
            fp_calib, fp_eval = next_fp_calib, next_fp_eval
            print(f"[layer {index}] elapsed={time.monotonic() - started:.1f}s", flush=True)
        finally:
            if entries is not None:
                restore_fp_weights(entries)
            layer.to("cpu")
            del entries
            gc.collect()
            torch.cuda.empty_cache()

    if not routed_layers:
        raise ValueError("The model has no routed-expert decoder layers")
    return rows, routed_layers, unactivated_experts


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This measurement requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    device = torch.device("cuda:0")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = load_causal_lm_checkpoint(
        args.model, model_dtype="bfloat16", attn_implementation=args.attn_impl,
        trust_remote_code=True, device_map="cpu",
    )
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

    rows, routed_layers, unactivated_experts = measure_layers(model, batches, args, device)
    metadata = {
        "format_version": 1,
        "experiment": "cumulative_uniform_routed_expert_gptq_decoder_output_error",
        "model": args.model, "model_name": MODEL_NAME,
        "dataset": "c4", "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples, "seqlen": args.seqlen, "seed": args.seed,
        "calibration_input_ids_sha256": calib_hash,
        "evaluation_input_ids_sha256": eval_hash,
        "bits": list(BITS), "routed_layers": routed_layers,
        "unactivated_experts_by_layer": unactivated_experts,
        "unactivated_expert_fallback": "identity Hessian (weight-only per-group rounding with MSE clipping)",
        "attn_impl": args.attn_impl,
        "gptq": {"groupsize": args.groupsize, "blocksize": args.blocksize,
                 "percdamp": args.percdamp, "mse": True},
        "metric": "sum(||decoder_l_b - decoder_l_FP||_2^2) / sum(||decoder_l_FP||_2^2)",
        "context": "FP calibration Hessians; each bit-width evaluation path propagates its own hidden states; only routed expert weights are quantized",
        "total_seconds": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "cumulative_relative_mse.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved measurements to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
