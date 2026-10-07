"""Token-wise PoP counterfactual at one full-precision Qwen3-MoE layer.

This is a local output-error experiment, not a deployable mixed-bit checkpoint:
the identity of the two 1-bit/pruned experts changes from token to token.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
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
LINEARS = ("gate_proj", "up_proj", "down_proj")


def route_scenarios(logits, top_k=8, removed=2):
    """Return top-k and rerouted top-k with separately normalized weights."""
    if removed <= 0 or top_k <= removed or logits.shape[-1] < top_k + removed:
        raise ValueError("Need 0 < removed < top_k and at least top_k + removed experts")
    scores = F.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(scores, top_k + removed, dim=-1)
    original = values[:, :top_k]
    rerouted = values[:, removed:removed + top_k]
    original = original / original.sum(dim=-1, keepdim=True)
    rerouted = rerouted / rerouted.sum(dim=-1, keepdim=True)
    return indices, original, rerouted


@torch.inference_mode()
def evaluate_moe_batch(moe, one_bit, two_bit, hidden, expert_batch_size):
    """Evaluate FP, 1+2-bit and token-wise 0+2-bit MoE outputs together."""
    if moe.top_k != 8 or not moe.norm_topk_prob:
        raise ValueError("This experiment requires Qwen3 top-8 with norm_topk_prob=True")
    flat = hidden.reshape(-1, hidden.shape[-1])
    # Use the model's actual FP MoE forward as teacher, including its native
    # accumulation dtype/order; do not substitute a reconstructed FP output.
    reference = _first_tensor(moe(hidden)).float()
    logits = moe.gate(flat)
    ids, original_weights, rerouted_weights = route_scenarios(logits)
    n_tokens, width = flat.shape
    outputs = [torch.zeros((n_tokens, width), device=flat.device, dtype=flat.dtype)
               for _ in range(2)]

    for expert_id in range(len(moe.experts)):
        token_index, rank = torch.where(ids == expert_id)
        if token_index.numel() == 0:
            continue
        for start in range(0, token_index.numel(), expert_batch_size):
            positions = slice(start, start + expert_batch_size)
            tokens = token_index[positions]
            ranks = rank[positions]
            inputs = flat[tokens]

            one_mask = ranks < 2
            if one_mask.any():
                q1 = _first_tensor(one_bit[expert_id](inputs[one_mask]))
                weights = original_weights[tokens[one_mask], ranks[one_mask], None].to(q1.dtype)
                weighted = q1 * weights
                outputs[0].index_add_(0, tokens[one_mask], weighted)

            two_mask = ranks >= 2
            if two_mask.any():
                q2 = _first_tensor(two_bit[expert_id](inputs[two_mask]))
                q2_tokens = tokens[two_mask]
                q2_ranks = ranks[two_mask]
                in_original = q2_ranks < 8
                if in_original.any():
                    weights = original_weights[
                        q2_tokens[in_original], q2_ranks[in_original], None
                    ].to(q2.dtype)
                    weighted = q2[in_original] * weights
                    outputs[0].index_add_(0, q2_tokens[in_original], weighted)
                weights = rerouted_weights[q2_tokens, q2_ranks - 2, None].to(q2.dtype)
                weighted = q2 * weights
                outputs[1].index_add_(0, q2_tokens, weighted)

    return reference, outputs[0].reshape_as(hidden).float(), outputs[1].reshape_as(hidden).float()


def batch_error_sums(reference, low_bit, rerouted):
    """Sum squared errors, and count paired wins, without unstable token ratios."""
    fp = reference.reshape(-1, reference.shape[-1]).float()
    low = low_bit.reshape_as(fp).float()
    pop = rerouted.reshape_as(fp).float()
    low_errors = (low - fp).square().sum(dim=-1)
    pop_errors = (pop - fp).square().sum(dim=-1)
    return {
        "reference_sq": fp.square().sum().double().item(),
        "low_bit_sq": low_errors.sum().double().item(),
        "pop_sq": pop_errors.sum().double().item(),
        "pop_wins": int((pop_errors < low_errors).sum().item()),
        "tokens": fp.shape[0],
    }


def aggregate(rows):
    totals = {name: sum(row[name] for row in rows)
              for name in ("reference_sq", "low_bit_sq", "pop_sq", "pop_wins", "tokens")}
    if totals["reference_sq"] <= 0 or totals["low_bit_sq"] <= 0:
        raise ValueError("Reference or low-bit error is zero; relative comparison is undefined")
    totals.update(
        low_bit_relative_mse=totals["low_bit_sq"] / totals["reference_sq"],
        pop_relative_mse=totals["pop_sq"] / totals["reference_sq"],
        pop_to_low_bit_error_ratio=totals["pop_sq"] / totals["low_bit_sq"],
        pop_token_win_rate=totals["pop_wins"] / totals["tokens"],
    )
    return totals


def _tensor_sha256(batches):
    digest = hashlib.sha256()
    for input_ids, _ in batches:
        digest.update(input_ids.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def capture_target_inputs(model, batches, layer_idx, device):
    """Run an FP prefix, then stop immediately before the selected MoE."""
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
        print(f"[FP prefix] completed layer {index}", flush=True)

    layer = layers[layer_idx].to(device)
    moe = get_moe_block(layer, MODEL_NAME)
    inputs, _, _, _ = _capture_qwen3_moe_inputs(
        layer, moe, hidden, positional, keyword, device, stop_at_moe=True
    )
    del hidden, positional, keyword
    gc.collect()
    torch.cuda.empty_cache()
    return layer, moe, inputs


@torch.inference_mode()
def build_quantized_experts(moe, calibration_inputs, device, blocksize, groupsize, percdamp):
    """Collect each FP linear's Hessian once; quantize 1 and 2 bits independently."""
    masters = {}
    handles = []
    for expert_id in range(len(moe.experts)):
        expert = _get_qwen3_expert(moe, expert_id)
        for name in LINEARS:
            module = getattr(expert, name)
            columns = module.weight.shape[1]
            effective_groupsize = groupsize if columns % groupsize == 0 else 64
            if columns % effective_groupsize:
                raise ValueError(f"{name}: input width {columns} has no supported groupsize")
            master = GPTQWeightQuantizer(
                module.weight.data, f"expert_{expert_id}.{name}", 2,
                blocksize, percdamp, effective_groupsize, False, False, True,
            )
            masters[expert_id, name] = master

            def capture(_module, arguments, _output, quantizer=master):
                quantizer.add_batch(arguments[0].detach())

            handles.append(module.register_forward_hook(capture))

    try:
        for index, hidden in enumerate(calibration_inputs):
            moe(hidden.to(device=device, non_blocking=True))
            if (index + 1) % 16 == 0 or index + 1 == len(calibration_inputs):
                print(f"[GPTQ calibration] {index + 1}/{len(calibration_inputs)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    variants = {1: [], 2: []}
    started = time.monotonic()
    for expert_id in range(len(moe.experts)):
        original = _get_qwen3_expert(moe, expert_id)
        copies = {bit: copy.deepcopy(original).cpu() for bit in (1, 2)}
        for name in LINEARS:
            master = masters.pop((expert_id, name))
            for bit in (1, 2):
                quantizer = GPTQWeightQuantizer(
                    master.W, master.name, bit, blocksize, percdamp,
                    master.groupsize, False, False, True,
                )
                # quantize() modifies H in place; both candidates must start
                # from the *same* full-precision calibration Hessian.
                quantizer.H = master.H.clone()
                quantizer.nsamples = master.nsamples
                codes, scales, zeros = quantizer.quantize()
                weight = quantizer.dequantize(codes, scales, zeros)
                module = getattr(copies[bit], name)
                module.weight.data = weight.reshape_as(module.weight).to("cpu")
                del quantizer, codes, scales, zeros, weight
            del master
        for bit in (1, 2):
            variants[bit].append(copies[bit])
        if (expert_id + 1) % 8 == 0:
            elapsed = time.monotonic() - started
            projected = elapsed * len(moe.experts) / (expert_id + 1)
            print(
                f"[GPTQ] {expert_id + 1}/{len(moe.experts)} experts, "
                f"elapsed={elapsed / 60:.1f} min, projected={projected / 60:.1f} min",
                flush=True,
            )
    return nn.ModuleList(variants[1]).to(device), nn.ModuleList(variants[2]).to(device)


def save_plot(metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    values = [metrics["low_bit_relative_mse"], metrics["pop_relative_mse"]]
    ax.bar([0, 1], values, color=["#D55E00", "#0072B2"], width=0.58)
    ax.set_xticks([0, 1], ["1+2 bit", "Prune+2 bit"])
    ax.set_ylabel("Relative MoE output MSE")
    ax.set_title(f"Layer {metrics['layer']} · C4 · {metrics['tokens']:,} tokens")
    ax.text(0.5, max(values) * 0.98,
            f"error ratio = {metrics['pop_to_low_bit_error_ratio']:.3f}\n"
            f"token wins = {metrics['pop_token_win_rate']:.1%}",
            ha="center", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=24, help="zero-based decoder layer")
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert_batch_size", type=int, default=1024)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--attn_impl", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("calib_samples", "eval_samples", "seqlen", "expert_batch_size", "groupsize", "blocksize"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.layer < 0 or args.seed < 0 or args.percdamp < 0:
        parser.error("--layer and --seed must be non-negative; --percdamp must be non-negative")
    if args.output_dir is None:
        args.output_dir = Path("cache/pop_output_error/Qwen3-30B-A3B-Instruct-2507") / (
            f"L{args.layer}-C4-Cal{args.calib_samples}-Eval{args.eval_samples}"
            f"-Len{args.seqlen}-Seed{args.seed}"
        )
    return args


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    try:
        import matplotlib  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for the figure; install GEMQ's plot extra first"
        ) from error
    device = torch.device("cuda:0")
    timings = {}
    started = time.monotonic()
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
    if len(selected_moe.experts) < 10:
        raise ValueError("At least 10 experts are required for top-8 replacement")
    model.config.use_cache = False
    loader_args = SimpleNamespace(
        calib_dataset="c4", model=args.model,
        nsamples=args.calib_samples + args.eval_samples,
        seqlen=args.seqlen, seed=args.seed, batch_size=1, use_fast=True,
    )
    batches = get_calib_loader(tokenizer, loader_args)
    if len(batches) != loader_args.nsamples:
        raise ValueError(f"Expected {loader_args.nsamples} C4 blocks, found {len(batches)}")
    calib_batches = batches[:args.calib_samples]
    eval_batches = batches[args.calib_samples:]
    calib_hash = _tensor_sha256(calib_batches)
    eval_hash = _tensor_sha256(eval_batches)
    if calib_hash == eval_hash:
        raise ValueError("Calibration and evaluation data hashes match")
    layer, moe, all_inputs = capture_target_inputs(model, batches, args.layer, device)
    timings["load_data_and_fp_prefix_seconds"] = time.monotonic() - started
    if len(all_inputs) != len(batches):
        raise RuntimeError("The target MoE did not receive every C4 block")
    print(
        f"[load, C4 and FP prefix] "
        f"{timings['load_data_and_fp_prefix_seconds'] / 60:.1f} min",
        flush=True,
    )

    started = time.monotonic()
    q1, q2 = build_quantized_experts(
        moe, all_inputs[:args.calib_samples], device,
        args.blocksize, args.groupsize, args.percdamp,
    )
    timings["gptq_seconds"] = time.monotonic() - started
    del all_inputs[:args.calib_samples]
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    started = time.monotonic()
    for index, hidden in enumerate(all_inputs):
        hidden = hidden.to(device=device, non_blocking=True)
        fp, low_bit, rerouted = evaluate_moe_batch(
            moe, q1, q2, hidden, args.expert_batch_size
        )
        row = {"sequence": index, **batch_error_sums(fp, low_bit, rerouted)}
        rows.append(row)
        print(f"[evaluation] {index + 1}/{len(all_inputs)}", flush=True)
    timings["evaluation_seconds"] = time.monotonic() - started
    metrics = {
        **aggregate(rows), "layer": args.layer,
        "model": args.model, "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples, "seqlen": args.seqlen,
        "seed": args.seed, "calib_ids_sha256": calib_hash,
        "eval_ids_sha256": eval_hash,
        "gptq": {"groupsize": args.groupsize, "blocksize": args.blocksize,
                 "percdamp": args.percdamp, "mse": True},
        "timings": timings,
        "note": "Token-wise counterfactual; not a deployable fixed expert-bit allocation.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "per_sequence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plot(metrics, args.output_dir / "pop_layer_error.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
