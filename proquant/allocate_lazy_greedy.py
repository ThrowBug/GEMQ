"""Refine Qwen3 PoP allocation using a full-model lazy greedy search."""

import argparse
import json
import math
import pickle
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate import dispatch_model
from transformers import AutoTokenizer

from proquant.allocate_bits import _auto_save_path
from proquant.allocation.ilp_solvers import AVAILABLE_BACKENDS, PoPSolver
from proquant.allocation.lazy_greedy import BITS, feasible_prune_counts, layer_assignment, lazy_descent
from proquant.allocation.qwen3_search import Qwen3Scorer, prepare_variants, scoring_device_map
from proquant.utils.data_utils import get_calib_loader
from proquant.utils.hf_loading import load_causal_lm_checkpoint
from proquant.utils.model_utils import get_blocks, get_model_info, get_moe_block


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, help="Original BF16 model checkpoint")
    parser.add_argument("--pop_cost_path", required=True, type=Path)
    parser.add_argument("--bit_budget", type=float, default=2.0)
    parser.add_argument("--max_prune_ratio", type=float, default=0.3)
    parser.add_argument("--ilp_backend", choices=AVAILABLE_BACKENDS, default="highs")
    parser.add_argument("--prune_counts", nargs="+", type=int,
                        help="Common prune counts to search. Defaults to up to three feasible counts nearest to PoP.")
    parser.add_argument("--budget_step", type=int, default=16,
                        help="Expert bit units freed in a move before refining to the exact target")
    parser.add_argument("--search_nsamples", type=int, default=4)
    parser.add_argument("--reference_nsamples", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=512, help="Search calibration sequence length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab_top_k", type=int, default=1000,
                        help="JSD groups all tokens outside the reference top-k into one tail bucket")
    parser.add_argument("--attn_wbits", type=int, choices=[2, 3, 4, 16], default=4)
    parser.add_argument("--scratch_dir", type=Path, default=Path("cache/lazy_greedy"))
    parser.add_argument("--save_path", type=Path, help="Output .pkl path")
    args = parser.parse_args(argv)
    if not math.isfinite(args.bit_budget) or args.bit_budget <= 0:
        parser.error("--bit_budget must be finite and positive.")
    if not (1 <= args.search_nsamples <= args.reference_nsamples):
        parser.error("Expected 1 <= search_nsamples <= reference_nsamples.")
    if args.seqlen < 2 or args.vocab_top_k < 1 or args.budget_step < 1:
        parser.error("Expected seqlen >= 2, vocab_top_k >= 1, and budget_step >= 1.")
    if args.save_path and args.save_path.suffix != ".pkl":
        parser.error("--save_path must end in .pkl.")
    args.model_name = MODEL_NAME
    args.allocation_metric = "pop"
    args.calib_dataset = "c4"
    args.nsamples = args.reference_nsamples
    args.batch_size = 1
    return args


def write_result(path, allocation, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with path.open("xb") as handle:
            created.append(path)
            pickle.dump(allocation, handle)
        with path.with_suffix(".json").open("x", encoding="utf-8") as handle:
            created.append(path.with_suffix(".json"))
            json.dump(metadata, handle, indent=2, allow_nan=False)
    except BaseException:
        for file in created:
            file.unlink()
        raise


def search_allocations(costs, target, prune_counts, step, baseline, scorer):
    num_layers, num_experts, _ = costs.shape
    baseline_bits = sum(sum(layer.values()) for layer in baseline.values())
    baseline_score = None
    if baseline_bits == target:
        for layer, expert_bits in sorted(baseline.items()):
            scorer.install(layer, tuple(expert_bits[i] for i in range(num_experts)))
        baseline_score = scorer.score()
        print(f"Original PoP JSD at the same budget: {baseline_score:.7f}", flush=True)

    results = []
    best = None
    for pruned in prune_counts:
        candidates = {}

        def candidate(layer, budget):
            key = layer, budget
            if key not in candidates:
                candidates[key] = layer_assignment(costs[layer], pruned, budget)
            return candidates[key]

        retained = num_experts - pruned
        print(f"Searching with {pruned} pruned experts per layer ...", flush=True)
        budgets, score, evaluations, moves = lazy_descent(
            num_layers, 2 * retained, 3 * retained, target, step, candidate, scorer,
        )
        results.append(dict(pruned_per_layer=pruned, jsd=score, evaluations=evaluations, moves=moves))
        print(f"p={pruned}: JSD={score:.7f}, evaluations={evaluations}, moves={moves}", flush=True)
        if best is None or score < best[0]:
            allocation = {
                layer: dict(enumerate(candidate(layer, budget)))
                for layer, budget in enumerate(budgets)
            }
            best = score, pruned, allocation
    return best, results, baseline_score


def main(argv=None):
    args = parse_args(argv)
    info = get_model_info(MODEL_NAME)
    solver = PoPSolver(
        expert_cost_path=args.pop_cost_path, x_space=BITS, max_prune_ratio=args.max_prune_ratio,
        top_k=info.num_experts_per_token, start_layer_idx=0, backend=args.ilp_backend,
    )
    metadata = solver.artifact_metadata
    if (
        solver.num_moe_layers != info.num_layers
        or solver.num_experts != info.num_routed_experts_per_layer
        or metadata.get("model_name", MODEL_NAME) != MODEL_NAME
        or metadata.get("layer_ids", list(range(info.num_layers))) != list(range(info.num_layers))
    ):
        raise ValueError("The expert-cost file must match Qwen3-30B-A3B-Instruct-2507.")
    if not torch.cuda.is_available():
        raise RuntimeError("Lazy greedy allocation requires CUDA.")
    target = math.floor(args.bit_budget * solver.num_experts * solver.num_moe_layers + 1e-12)
    feasible = feasible_prune_counts(
        solver.num_moe_layers, solver.num_experts, solver.max_pruned_per_layer, target,
    )
    if not feasible:
        raise ValueError("The target budget has no exact allocation under this pruning cap.")
    if args.prune_counts and any(p not in feasible for p in args.prune_counts):
        raise ValueError(f"Requested prune counts must be in {feasible}.")
    if args.save_path:
        output = args.save_path
    else:
        source_tag = str(args.pop_cost_path)
        if "calib_dataset" in metadata and "seed" in metadata:
            source_tag = f"{metadata['calib_dataset']}-Seed{metadata['seed']}"
        original = Path(_auto_save_path(args, source_tag, BITS, solver))
        output = original.with_name(f"{original.stem}_Search-lazy_greedy.pkl")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Allocation output already exists: {output}")

    baseline = solver.solve_all(total_bits=target)
    baseline_pruned = sum(bit == 0 for bit in baseline[0].values())
    neighbors = sorted(feasible, key=lambda p: (abs(p - baseline_pruned), p))[:3]
    prune_counts = sorted(set(args.prune_counts or neighbors))
    print(f"Prune counts to search: {prune_counts}; exact total budget: {target}", flush=True)

    model = load_causal_lm_checkpoint(args.model, trust_remote_code=True)
    model.config.use_cache = False
    blocks = get_blocks(model, MODEL_NAME)
    if model.config.model_type != "qwen3_moe" or len(blocks) != solver.num_moe_layers:
        raise ValueError("The search model must be the original Qwen3 model.")
    if any(len(get_moe_block(layer, MODEL_NAME).experts) != solver.num_experts for layer in blocks):
        raise ValueError("The search checkpoint must contain all original routed experts.")
    device_map = scoring_device_map(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    batches = get_calib_loader(tokenizer, args)
    if len(batches) != args.reference_nsamples or any(batch[0].shape[0] != 1 for batch in batches):
        raise ValueError("Expected one C4 sequence per calibration batch.")

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    expert_parameters = sum(
        parameter.numel() for layer in blocks
        for parameter in get_moe_block(layer, MODEL_NAME).experts.parameters()
    )
    # Each expert has two nibble-packed search variants, plus group metadata.
    estimated_bytes = math.ceil(1.15 * expert_parameters)
    if shutil.disk_usage(args.scratch_dir).free < estimated_bytes:
        raise OSError(f"Search cache needs about {estimated_bytes / 1e9:.1f} GB free in {args.scratch_dir}.")
    with tempfile.TemporaryDirectory(prefix="qwen3-", dir=args.scratch_dir) as cache:
        targets, unseen = prepare_variants(
            model, MODEL_NAME, batches, args.search_nsamples, args.vocab_top_k,
            args.attn_wbits, Path(cache),
        )
        model = dispatch_model(model, device_map=device_map)
        with Qwen3Scorer(model, MODEL_NAME, cache, batches[:args.search_nsamples], targets) as scorer:
            best, results, baseline_score = search_allocations(
                solver.coef, target, prune_counts, args.budget_step, baseline, scorer,
            )
    score, pruned, allocation = best
    used_bits = sum(sum(experts.values()) for experts in allocation.values())
    if used_bits != target or any(
        sum(bit == 0 for bit in experts.values()) != pruned for experts in allocation.values()
    ):
        raise RuntimeError("The chosen allocation violates its bit or pruning budget.")
    save_metadata = {
        "format_version": 1,
        "allocation_metric": "pop",
        "allocation_search": "lazy_greedy",
        "model_name": MODEL_NAME,
        "source_path": str(args.pop_cost_path),
        "average_bit_budget": args.bit_budget,
        "total_bit_budget": target,
        "used_bits": used_bits,
        "candidate_bits": list(BITS),
        "ilp_backend": args.ilp_backend,
        "max_prune_ratio": args.max_prune_ratio,
        "equal_prune_count_across_layers": True,
        "pruned_experts_per_layer": {str(layer): pruned for layer in allocation},
        "search_seed": args.seed,
        "search_nsamples": args.search_nsamples,
        "reference_nsamples": args.reference_nsamples,
        "search_seqlen": args.seqlen,
        "search_budget_step": args.budget_step,
        "search_attn_wbits": args.attn_wbits,
        "jsd_vocab_top_k_with_tail": args.vocab_top_k,
        "unseen_experts_with_identity_hessian": unseen,
        "baseline_pop_pruned_per_layer": baseline_pruned,
        "baseline_pop_jsd_at_target": baseline_score,
        "selected_jsd": score,
        "searched_prune_counts": results,
    }
    write_result(output, allocation, save_metadata)
    print(f"Bit config file saved to: {output}")
    print(f"Allocation metadata saved to: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
