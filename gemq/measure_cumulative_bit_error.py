"""Measure how the layer-0 expert bit width propagates through 2-bit layers.

Layer 0 branches into 1/2/3/4-bit routed-expert paths. Later routed layers all
use the same 2-bit weights, but each path receives its own preceding hidden
states. GPTQ Hessians come from the same full-precision C4 calibration path.
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


INITIAL_BITS = (1, 2, 3, 4)
CONTINUATION_BIT = 2
CSV_COLUMNS = (
    "layer", "initial_bit_width", "current_bit_width", "tokens", "squared_error_sum",
    "reference_squared_sum", "relative_mse",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--attn_impl", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--max_layers", type=int,
                        help="Measure only the first N decoder layers (indices 0 through N-1)")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    for key in ("calib_samples", "eval_samples", "seqlen", "groupsize", "blocksize"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.seed < 0 or not math.isfinite(args.percdamp) or args.percdamp < 0:
        parser.error("--seed and --percdamp must be non-negative and finite")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max_layers must be positive")
    if args.output_dir is None:
        run_name = (
            f"C4-Cal{args.calib_samples}-Eval{args.eval_samples}"
            f"-Len{args.seqlen}-Seed{args.seed}"
        )
        if args.max_layers is not None:
            run_name += f"-First{args.max_layers}"
        run_name += "-L0B1-2-3-4-RestB2"
        args.output_dir = Path("cache/cumulative_bit_error/Qwen3-30B-A3B-Instruct-2507") / run_name
    return args


def validate_routed_moe(moe, layer_index):
    """Dense-only decoder layers are allowed, but routed layers must match Qwen3."""
    if not hasattr(moe, "experts"):
        return False
    if len(moe.experts) != 128 or moe.top_k != 8 or not moe.norm_topk_prob:
        raise ValueError(f"Layer {layer_index}: expected 128 routed experts and normalized top-8")
    return True


def error_row(layer_index, initial_bit, current_bit, reference, candidate):
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
        raise ValueError(
            f"Layer {layer_index}, initial {initial_bit}-bit: non-finite or zero error denominator"
        )
    return {
        "layer": layer_index,
        "initial_bit_width": initial_bit,
        "current_bit_width": current_bit,
        "tokens": tokens,
        "squared_error_sum": error_total,
        "reference_squared_sum": reference_total,
        "relative_mse": error_total / reference_total,
    }


@torch.inference_mode()
def measure_layers(model, batches, args, device):
    layers = get_blocks(model, MODEL_NAME)
    if args.max_layers is not None and args.max_layers > len(layers):
        raise ValueError(f"--max_layers={args.max_layers} exceeds {len(layers)} decoder layers")
    layer_count = args.max_layers if args.max_layers is not None else len(layers)
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
    bit_eval = {bit: list(fp_eval) for bit in INITIAL_BITS}
    rows = []
    routed_layers = []
    unactivated_experts = {}

    for index in range(layer_count):
        layer = layers[index]
        started = time.monotonic()
        layer.to(device)
        entries = None
        try:
            moe = get_moe_block(layer, MODEL_NAME)
            routed = validate_routed_moe(moe, index)
            if index == 0 and not routed:
                raise ValueError("Layer 0 must contain routed experts for this experiment")
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
            if index == 0:
                for initial_bit in INITIAL_BITS:
                    try:
                        install_bit_weights(entries, initial_bit, args)
                        next_bit_eval = _forward_layer_batches(
                            layer, bit_eval[initial_bit], eval_positional, eval_keyword, device
                        )
                    finally:
                        restore_fp_weights(entries)
                    row = error_row(
                        index, initial_bit, initial_bit, next_fp_eval, next_bit_eval
                    )
                    rows.append(row)
                    bit_eval[initial_bit] = next_bit_eval
                    print(
                        f"[layer {index} | initial {initial_bit}-bit] relative MSE="
                        f"{row['relative_mse']:.8g}", flush=True,
                    )
            else:
                try:
                    if routed:
                        install_bit_weights(entries, CONTINUATION_BIT, args)
                    for initial_bit in INITIAL_BITS:
                        next_bit_eval = _forward_layer_batches(
                            layer, bit_eval[initial_bit], eval_positional, eval_keyword, device
                        )
                        row = error_row(
                            index, initial_bit, CONTINUATION_BIT if routed else 16,
                            next_fp_eval, next_bit_eval,
                        )
                        rows.append(row)
                        bit_eval[initial_bit] = next_bit_eval
                        print(
                            f"[layer {index} | initial {initial_bit}-bit, "
                            f"current {CONTINUATION_BIT if routed else 16}-bit] "
                            f"cumulative relative MSE={row['relative_mse']:.8g}", flush=True,
                        )
                finally:
                    if routed:
                        restore_fp_weights(entries)
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
        "experiment": "layer0_bit_propagation_through_2bit_routed_experts",
        "model": args.model, "model_name": MODEL_NAME,
        "dataset": "c4", "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples, "seqlen": args.seqlen, "seed": args.seed,
        "calibration_input_ids_sha256": calib_hash,
        "evaluation_input_ids_sha256": eval_hash,
        "initial_bits": list(INITIAL_BITS), "continuation_bit": CONTINUATION_BIT,
        "routed_layers": routed_layers,
        "max_layers": args.max_layers, "measured_layers": len(rows) // len(INITIAL_BITS),
        "unactivated_experts_by_layer": unactivated_experts,
        "unactivated_expert_fallback": "identity Hessian (weight-only per-group rounding with MSE clipping)",
        "attn_impl": args.attn_impl,
        "gptq": {"groupsize": args.groupsize, "blocksize": args.blocksize,
                 "percdamp": args.percdamp, "mse": True},
        "metric": "sum(||decoder_l_initial_b - decoder_l_FP||_2^2) / sum(||decoder_l_FP||_2^2)",
        "context": "FP calibration Hessians; layer 0 uses separate initial bits, later routed layers share 2-bit weights while each path propagates its own hidden states; only routed expert weights are quantized",
        "total_seconds": time.monotonic() - started,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "layer0_bit_propagation.csv").open(
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
