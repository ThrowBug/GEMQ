"""Compare full-precision and fake-quant top-expert routing on identical inputs."""

import argparse
import csv
import json
from pathlib import Path

from gemq.plot_routing_jaccard import jaccard, read_routing_csv, top_experts


DATASETS = ("c4", "math_500", "gpqa_diamond")
METRIC = "mean_probability_when_active"
MATCHED_METADATA = (
    "dataset", "model", "records", "nsamples", "seqlen", "seed", "dtype",
    "attn_impl", "layers", "n_experts", "top_k", "scopes", "token_totals",
    "mass_definition", "c4_source",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized_model_path", required=True, type=Path,
                        help="Saved fake-quant model directory; its final component names the stats subdirectory")
    parser.add_argument("--stats_root", type=Path, default=Path("cache/routing_analysis/stats"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qa_scope", choices=("combined", "prompt", "answer"), default="combined")
    parser.add_argument("--top_n", type=int, default=32)
    parser.add_argument("--output_csv", type=Path)
    args = parser.parse_args(argv)
    if not args.quantized_model_path.is_dir():
        parser.error("--quantized_model_path must be an existing model directory")
    if args.quantized_model_path.name in ("", ".", ".."):
        parser.error("--quantized_model_path must name a model directory, not its parent")
    if args.nsamples <= 0 or args.seqlen <= 0 or args.top_n <= 0:
        parser.error("--nsamples, --seqlen and --top_n must be positive")
    args.model_name = args.quantized_model_path.name
    args.run_name = f"N{args.nsamples}-L{args.seqlen}-Seed{args.seed}"
    if args.output_csv is None:
        args.output_csv = (
            Path("cache/routing_analysis/comparisons") / args.model_name / args.run_name
            / f"top{args.top_n}_mean_routing_score_{args.qa_scope}_jaccard.csv"
        )
    return args


def read_metadata(stats_dir):
    path = stats_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Routing metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_matching_metadata(reference, quantized, dataset):
    if "quantized_model_path" in reference:
        raise ValueError(f"{dataset}: reference statistics are not from the full-precision model")
    if not quantized.get("quantized_model_path"):
        raise ValueError(f"{dataset}: quantized statistics lack quantized_model_path metadata")
    for key in MATCHED_METADATA:
        if key not in reference or key not in quantized or reference[key] != quantized[key]:
            raise ValueError(f"{dataset}: full-precision and quantized statistics disagree on {key}")


def compare_dataset(reference, quantized, dataset, scope, top_n):
    if reference.keys() != quantized.keys():
        raise ValueError(f"{dataset}: full-precision and quantized statistics have different layers")
    rows = []
    for layer in sorted(reference):
        if reference[layer].keys() != quantized[layer].keys():
            raise ValueError(f"{dataset} layer {layer}: expert IDs differ")
        reference_top = top_experts(reference[layer], METRIC, top_n)
        quantized_top = top_experts(quantized[layer], METRIC, top_n)
        rows.append({
            "dataset": dataset,
            "scope": scope,
            "layer": layer,
            "metric": METRIC,
            "top_n": top_n,
            "intersection": len(reference_top & quantized_top),
            "union": len(reference_top | quantized_top),
            "jaccard": jaccard(reference_top, quantized_top),
        })
    if not rows:
        raise ValueError(f"{dataset}: no layers to compare")
    return rows


def main(argv=None):
    args = parse_args(argv)
    if args.output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}")
    rows = []
    for dataset in DATASETS:
        scope = "combined" if dataset == "c4" else args.qa_scope
        reference_dir = args.stats_root / dataset / args.run_name
        quantized_dir = args.stats_root / args.model_name / dataset / args.run_name
        reference_metadata = read_metadata(reference_dir)
        quantized_metadata = read_metadata(quantized_dir)
        check_matching_metadata(reference_metadata, quantized_metadata, dataset)
        if quantized_metadata.get("quantized_model_name", Path(quantized_metadata["quantized_model_path"]).name) != args.model_name:
            raise ValueError(f"{dataset}: quantized statistics belong to a different model directory")
        reference_csv = reference_dir / "routing_stats.csv"
        quantized_csv = quantized_dir / "routing_stats.csv"
        reference = read_routing_csv(reference_csv, dataset, scope)
        quantized = read_routing_csv(quantized_csv, dataset, scope)
        rows.extend(compare_dataset(reference, quantized, dataset, scope, args.top_n))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "dataset", "scope", "layer", "metric", "top_n", "intersection", "union", "jaccard",
        ))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
