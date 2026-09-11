import argparse
import json
import math
import os
import os.path as osp
import pickle

from gemq.allocation.pmq_solver import PMQ_CANDIDATE_BITS, PMQSolver
from gemq.utils.model_utils import get_model_info


STAT_FILENAMES = {
    "act_counts": "experts_act_counts.pkl",
    "act_weights": "experts_act_weights.pkl",
    "quant_loss": "experts_quant_loss.pkl",
}


def _load_pickle(path):
    if not osp.isfile(path):
        raise FileNotFoundError(f"Required PMQ statistics file not found: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _parse_candidate_bits(value):
    try:
        bits = tuple(int(piece) for piece in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Candidate bits must be comma-separated integers, got {value!r}."
        ) from error
    if bits != PMQ_CANDIDATE_BITS:
        raise argparse.ArgumentTypeError(
            "Source-faithful PMQ requires --bit_candidates 1,2,3."
        )
    return bits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Allocate expert bits with source-faithful MC-MoE PMQ."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--stats_dir", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--bits_per_expert", type=float, default=2.0)
    parser.add_argument(
        "--bit_candidates", type=_parse_candidate_bits, default=PMQ_CANDIDATE_BITS
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument(
        "--gama",
        type=float,
        default=2.0,
        help="Recorded for source compatibility; released MC-MoE code does not use it.",
    )
    parser.add_argument(
        "--ilp_backend", choices=("highs", "gurobi"), default="highs"
    )
    parser.add_argument(
        "--allow_under_budget",
        action="store_true",
        help="Allow the source <= constraint to produce less than the requested BPE.",
    )
    return parser.parse_args(argv)


def run(args):
    stat_paths = {
        key: osp.join(args.stats_dir, filename)
        for key, filename in STAT_FILENAMES.items()
    }
    stats_metadata_path = osp.join(args.stats_dir, "metadata.json")
    stats_metadata = None
    if osp.isfile(stats_metadata_path):
        with open(stats_metadata_path, "r", encoding="utf-8") as handle:
            stats_metadata = json.load(handle)
        if stats_metadata.get("model_name") != args.model_name:
            raise ValueError(
                "PMQ statistics model mismatch: "
                f"metadata has {stats_metadata.get('model_name')!r}, "
                f"CLI requested {args.model_name!r}."
            )
        if tuple(stats_metadata.get("candidate_bits", ())) != tuple(args.bit_candidates):
            raise ValueError(
                "PMQ statistics candidate-bit mismatch: "
                f"metadata has {stats_metadata.get('candidate_bits')!r}, "
                f"CLI requested {args.bit_candidates!r}."
            )
    stats = {key: _load_pickle(path) for key, path in stat_paths.items()}
    solver = PMQSolver(
        stats["act_counts"],
        stats["act_weights"],
        stats["quant_loss"],
        candidate_bits=args.bit_candidates,
        alpha=args.alpha,
        beta=args.beta,
        gama=args.gama,
        backend=args.ilp_backend,
    )

    model_info = get_model_info(args.model_name)
    expected_layer_ids = list(
        range(model_info.first_k_dense_layers, model_info.num_layers)
    )
    if solver.layer_ids != expected_layer_ids:
        raise ValueError(
            f"PMQ statistics contain layers {solver.layer_ids}, but {args.model_name} "
            f"expects {expected_layer_ids}."
        )
    expected_experts = (
        model_info.num_routed_experts_per_layer
        + model_info.num_shared_experts_per_layer
    )
    if solver.num_experts != expected_experts:
        raise ValueError(
            f"PMQ statistics contain {solver.num_experts} experts/layer, but "
            f"{args.model_name} expects {expected_experts}."
        )

    allocation = solver.solve(
        bits_per_expert=args.bits_per_expert,
        require_full_budget=not args.allow_under_budget,
    )
    save_dir = osp.dirname(osp.abspath(args.save_path))
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as handle:
        pickle.dump(allocation, handle)

    total_experts = len(allocation) * solver.num_experts
    total_used_bits = sum(solver.used_bits_per_layer.values())
    actual_bpe = total_used_bits / total_experts
    sidecar = {
        "format_version": 1,
        "method": "PMQ",
        "model_name": args.model_name,
        "stats_dir": args.stats_dir,
        "statistics": stat_paths,
        "statistics_metadata_path": (
            stats_metadata_path if stats_metadata is not None else None
        ),
        "statistics_metadata": stats_metadata,
        "candidate_bits": list(args.bit_candidates),
        "target_bits_per_expert": args.bits_per_expert,
        "actual_bits_per_expert": actual_bpe,
        "num_moe_layers": len(allocation),
        "num_experts_per_layer": solver.num_experts,
        "total_used_bits": total_used_bits,
        "used_bits_per_layer": solver.used_bits_per_layer,
        "objective": solver.last_objective,
        "objective_definition": (
            "1000 * normalized_activation_count**alpha * "
            "normalized_routing_weight**beta * quantization_loss**alpha"
        ),
        "alpha": args.alpha,
        "beta": args.beta,
        "gama": args.gama,
        "gama_is_unused_for_source_compatibility": True,
        "per_layer_budget_constraint": (
            "<=" if args.allow_under_budget else "=="
        ),
        "requires_at_least_one_2bit_and_3bit_expert_per_layer": True,
        "require_full_budget": not args.allow_under_budget,
        "ilp_backend": args.ilp_backend,
    }
    sidecar_path = osp.splitext(args.save_path)[0] + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=False)

    if not math.isclose(actual_bpe, args.bits_per_expert, abs_tol=1e-12):
        print(
            f"Warning: PMQ used {actual_bpe:.6f} bits/expert, below the "
            f"requested {args.bits_per_expert:.6f}."
        )
    print("PMQ bit config saved to:", args.save_path)
    print("PMQ allocation metadata saved to:", sidecar_path)
    print(
        f"PMQ allocation: layers={len(allocation)}, "
        f"experts/layer={solver.num_experts}, actual BPE={actual_bpe:.6f}"
    )
    return allocation, sidecar


if __name__ == "__main__":
    run(parse_args())
