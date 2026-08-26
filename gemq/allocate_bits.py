import argparse
import json
import math
import os
import os.path as osp
import pickle
import re

from gemq.allocation.ilp_solvers import (
    AVAILABLE_BACKENDS,
    ExpertCostSolver,
    GEMQSolver,
)
from gemq.utils.model_utils import get_model_info


ALLOCATION_METRICS = ("layer_re", "expert_cost")


def auto_parse_filename(stat_path):
    calib_str = ""
    normalized_path = stat_path.lower()
    if "mixed_chat_en" in normalized_path:
        calib_str = "MixedChatEn"
    elif "math+c4" in normalized_path:
        calib_str = "MATH+C4"
    elif "c4" in normalized_path:
        calib_str = "C4"
    elif "math" in normalized_path:
        calib_str = "MATH"
    else:
        raise ValueError(f"Cannot parse calibration dataset from statistics path: {stat_path}")

    match = re.search(r"Seed(\d+)", stat_path, flags=re.IGNORECASE)
    seed_num = match.group(1) if match else "00"
    calib_str += f"-Seed{seed_num}"

    model_str = ""
    if "uni" in normalized_path:
        model_str = "_QT"
    elif "qtft" in normalized_path:
        model_str = "_QTFT"
    return calib_str, model_str


def parse_bit_candidates(value):
    pieces = value.split(",")
    if not pieces or any(not piece.strip().isdigit() for piece in pieces):
        raise ValueError(
            f"--bit_candidates must be comma-separated non-negative integers, got {value!r}"
        )
    bits = [int(piece) for piece in pieces]
    if len(set(bits)) != len(bits):
        raise ValueError(f"--bit_candidates contains duplicates: {bits}")
    if any(bit > 16 for bit in bits):
        raise ValueError(f"--bit_candidates must not exceed 16: {bits}")
    if not any(bit > 0 for bit in bits):
        raise ValueError("--bit_candidates must contain at least one positive bit-width")
    return bits


def compute_total_bits(model_name, bpe, bit_cands):
    """Compute the legacy Layer-RE global budget, including shared experts."""
    m = get_model_info(model_name)
    bpl = (
        bpe * (m.num_routed_experts_per_layer + m.num_shared_experts_per_layer)
        - (max(0, m.num_shared_experts_per_layer - 1)) * max(bit_cands)
    )
    return math.floor(bpl * (m.num_layers - m.first_k_dense_layers) + 1e-12)


def _context_tag(metadata):
    mode = str(metadata.get("context_mode", "unknown")).replace("_", "")
    if mode == "uniformbit":
        average_bits = metadata.get("average_bits", "unknown")
        return f"Ctxuniform{average_bits}"
    return f"Ctx{mode}"


def _auto_save_path(args, source_path, bit_cands, solver):
    calib_str, model_str = auto_parse_filename(source_path)
    bc_str = ",".join(map(str, bit_cands))
    common = f"{calib_str}_Metric-{args.allocation_metric}"
    if args.allocation_metric == "layer_re":
        constraint_tag = "" if args.extra_constr == "none" else f"_{args.extra_constr}"
        filename = (
            f"{common}_E{args.bit_budget:.1f}_B{bc_str}"
            f"{constraint_tag}{model_str}.pkl"
        )
    else:
        filename = (
            f"{common}-{_context_tag(solver.artifact_metadata)}"
            f"_E{args.bit_budget:.1f}_B{bc_str}"
        )
        if 0 in bit_cands:
            filename += f"_Pmax{args.max_prune_ratio:g}_EqPrune"
        filename += ".pkl"
    return f"configs/{args.model_name}/GEMQ/{filename}"


def run_solver(args):
    m = get_model_info(args.model_name)
    bit_cands = parse_bit_candidates(args.bit_candidates)

    if args.allocation_metric == "layer_re":
        if not args.layer_re_path:
            raise ValueError("--layer_re_path is required when --allocation_metric=layer_re")
        if args.expert_cost_path:
            print("Ignoring --expert_cost_path because allocation_metric=layer_re.")
        source_path = args.layer_re_path
        total_bits = compute_total_bits(args.model_name, args.bit_budget, bit_cands)
        global_solver = GEMQSolver(
            layer_re_path=source_path,
            x_space=bit_cands,
            extra_constr=args.extra_constr,
            start_layer_idx=m.first_k_dense_layers,
            backend=args.ilp_backend,
        )
    else:
        if not args.expert_cost_path:
            raise ValueError(
                "--expert_cost_path is required when --allocation_metric=expert_cost"
            )
        if args.extra_constr != "none":
            raise ValueError(
                "--extra_constr is a legacy Layer-RE option; expert_cost allocation "
                "uses only the global budget and zero-bit pruning constraints."
            )
        source_path = args.expert_cost_path
        global_solver = ExpertCostSolver(
            expert_cost_path=source_path,
            x_space=bit_cands,
            max_prune_ratio=args.max_prune_ratio,
            top_k=m.num_experts_per_token,
            start_layer_idx=m.first_k_dense_layers,
            backend=args.ilp_backend,
        )
        expected_layers = m.num_layers - m.first_k_dense_layers
        if global_solver.num_moe_layers != expected_layers:
            raise ValueError(
                f"Expert-cost artifact has {global_solver.num_moe_layers} layers, "
                f"but {args.model_name} expects {expected_layers}."
            )
        if global_solver.num_experts != m.num_routed_experts_per_layer:
            raise ValueError(
                f"Expert-cost artifact has {global_solver.num_experts} experts/layer, "
                f"but {args.model_name} expects {m.num_routed_experts_per_layer}."
            )
        total_bits = math.floor(
            args.bit_budget
            * global_solver.num_moe_layers
            * global_solver.num_experts
            + 1e-12
        )

    opt_set = global_solver.solve_all(total_bits=total_bits)
    save_path = args.save_path or _auto_save_path(
        args, source_path, bit_cands, global_solver
    )
    save_dir = osp.dirname(osp.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    # Preserve GEMQ's downstream contract exactly:
    # {layer_idx: {original_expert_idx: bit}}.
    with open(save_path, "wb") as handle:
        pickle.dump(opt_set, handle)

    used_bits = sum(bit for experts in opt_set.values() for bit in experts.values())
    pruned_per_layer = {
        str(layer_idx): sum(bit == 0 for bit in experts.values())
        for layer_idx, experts in opt_set.items()
    }
    sidecar = {
        "format_version": 1,
        "allocation_metric": args.allocation_metric,
        "source_path": source_path,
        "model_name": args.model_name,
        "average_bit_budget": args.bit_budget,
        "total_bit_budget": total_bits,
        "used_bits": used_bits,
        "candidate_bits": bit_cands,
        "ilp_backend": args.ilp_backend,
        "objective": (
            float(global_solver.last_objective)
            if global_solver.last_objective is not None
            else None
        ),
        "max_prune_ratio": args.max_prune_ratio if 0 in bit_cands else None,
        "equal_prune_count_across_layers": 0 in bit_cands,
        "pruned_experts_per_layer": pruned_per_layer,
    }
    sidecar_path = osp.splitext(save_path)[0] + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False)

    print("Bit config file saved to:", save_path)
    print("Allocation metadata saved to:", sidecar_path)
    return opt_set, save_path


# Backward-compatible entry point for callers that imported the old helper.
run_gemq_solver = run_solver


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Global bit allocation for MoE models.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--allocation_metric",
        choices=ALLOCATION_METRICS,
        default="layer_re",
        help="Use legacy Layer-RE coefficients or the new expert-cost tensor.",
    )
    parser.add_argument("--layer_re_path", type=str, default="")
    parser.add_argument("--expert_cost_path", type=str, default="")
    parser.add_argument(
        "--bit_budget", type=float, required=True, help="Maximum average bits per expert"
    )
    parser.add_argument("--bit_candidates", type=str, default="1,2,3")
    parser.add_argument(
        "--ilp_solver", type=str, default="gemq", choices=["gemq"],
        help="Compatibility flag; both metrics use the global MILP implementation.",
    )
    parser.add_argument(
        "--ilp_backend", type=str, default="highs", choices=list(AVAILABLE_BACKENDS)
    )
    parser.add_argument(
        "--extra_constr", type=str, default="none",
        help="Legacy Layer-RE constraint (for example c2c3); invalid for expert_cost.",
    )
    parser.add_argument(
        "--max_prune_ratio", type=float, default=0.25,
        help="Per-layer upper bound on the zero-bit fraction; active only if bit 0 is a candidate.",
    )
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args(argv)
    if not math.isfinite(args.bit_budget) or args.bit_budget < 0:
        parser.error("--bit_budget must be a finite non-negative number")
    if not math.isfinite(args.max_prune_ratio) or not 0 <= args.max_prune_ratio <= 1:
        parser.error("--max_prune_ratio must be in [0, 1]")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    print(json.dumps(vars(cli_args), indent=4))
    run_solver(cli_args)
