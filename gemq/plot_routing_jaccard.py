"""Plot layer-wise C4/QA overlap of top experts from routing_stats.csv."""

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = (
    ("activation_count", "Activation count", "o", "-"),
    ("mean_probability_when_active", "Mean routing score", "s", "--"),
)
DOMAINS = (
    ("math_500", "MATH-500", "#0072B2"),
    ("gpqa_diamond", "GPQA-Diamond", "#D55E00"),
)


def read_routing_csv(path, expected_dataset, scope):
    """Read one scope as layer -> expert -> metric values, validating the CSV."""
    data = {}
    required = {
        "dataset", "scope", "layer", "expert", "tokens", "activation_count",
        "mean_probability_when_active",
    }
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing routing_stats.csv columns: {sorted(required - set(reader.fieldnames or []))}")
        seen_scope = False
        for row in reader:
            if row["dataset"] != expected_dataset:
                raise ValueError(f"{path}: dataset is {row['dataset']!r}, expected {expected_dataset!r}")
            if row["scope"] != scope:
                continue
            seen_scope = True
            layer, expert = int(row["layer"]), int(row["expert"])
            count = int(row["activation_count"])
            if layer < 0 or expert < 0 or count < 0:
                raise ValueError(f"{path}: layer, expert and count must be non-negative")
            mean_text = row["mean_probability_when_active"].strip()
            if count == 0 and mean_text:
                raise ValueError(f"{path}: inactive expert {expert} has a mean routing score")
            if count > 0 and not mean_text:
                raise ValueError(f"{path}: active expert {expert} has no mean routing score")
            mean = float(mean_text) if mean_text else None
            if mean is not None and (not math.isfinite(mean) or not 0 <= mean <= 1):
                raise ValueError(f"{path}: invalid mean routing score {mean}")
            experts = data.setdefault(layer, {})
            if expert in experts:
                raise ValueError(f"{path}: duplicate layer={layer}, expert={expert}, scope={scope}")
            experts[expert] = {
                "activation_count": count,
                "mean_probability_when_active": mean,
            }
    if not seen_scope:
        raise ValueError(f"{path}: scope {scope!r} is absent")
    return data


def top_experts(experts, metric, top_n):
    """Rank by descending score, breaking ties with ascending expert ID."""
    ranked = [
        (expert, values[metric]) for expert, values in experts.items()
        if values[metric] is not None and values["activation_count"] > 0
    ]
    if len(ranked) < top_n:
        raise ValueError(f"Only {len(ranked)} experts have {metric}; --top_n={top_n} is too large")
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return {expert for expert, _ in ranked[:top_n]}


def jaccard(left, right):
    return len(left & right) / len(left | right)


def compare_domains(c4, target, target_name, top_n):
    """Build plot-ready rows for all common layers and both ranking metrics."""
    rows = []
    for layer in sorted(c4.keys() & target.keys()):
        if c4[layer].keys() != target[layer].keys():
            raise ValueError(f"Layer {layer}: C4 and {target_name} have different expert IDs")
        for metric, _, _, _ in METRICS:
            c4_top = top_experts(c4[layer], metric, top_n)
            target_top = top_experts(target[layer], metric, top_n)
            rows.append({
                "comparison": f"{target_name}_vs_c4",
                "layer": layer,
                "metric": metric,
                "top_n": top_n,
                "intersection": len(c4_top & target_top),
                "union": len(c4_top | target_top),
                "jaccard": jaccard(c4_top, target_top),
            })
    if not rows:
        raise ValueError(f"C4 and {target_name} have no common layers")
    return rows


def draw_figure(rows, output_png, output_pdf, top_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install with: pip install -e '.[plot]'") from exc

    fig, ax = plt.subplots(figsize=(8.2, 4.1), constrained_layout=True)
    for dataset, label, color in DOMAINS:
        for metric, metric_label, marker, linestyle in METRICS:
            points = sorted(
                (row for row in rows if row["comparison"] == f"{dataset}_vs_c4" and row["metric"] == metric),
                key=lambda row: row["layer"],
            )
            if points:
                ax.plot(
                    [point["layer"] for point in points],
                    [point["jaccard"] for point in points],
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=3.7, linewidth=1.45,
                    label=f"{label} vs C4 · {metric_label}",
                )
    ax.set_xlabel("MoE layer (0-based)")
    ax.set_ylabel(f"Top-{top_n} expert Jaccard")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stats_root = Path("cache/routing_analysis/stats")
    run_name = "N128-L2048-Seed0"
    parser.add_argument("--c4_csv", type=Path, default=stats_root / "c4" / run_name / "routing_stats.csv")
    parser.add_argument("--math_csv", type=Path, default=stats_root / "math_500" / run_name / "routing_stats.csv")
    parser.add_argument("--gpqa_csv", type=Path, default=stats_root / "gpqa_diamond" / run_name / "routing_stats.csv")
    parser.add_argument("--qa_scope", choices=("combined", "prompt", "answer"), default="combined")
    parser.add_argument("--top_n", type=int, default=32, help="Number of top-ranked experts per layer (default: 32)")
    parser.add_argument("--output_prefix", type=Path,
                        help="Output path without extension; creates .png, .pdf, .csv and .json in cache/")
    args = parser.parse_args(argv)
    if args.top_n <= 0:
        parser.error("--top_n must be positive")
    if args.output_prefix is None:
        args.output_prefix = Path("cache/routing_analysis/figures") / f"jaccard_layers_top{args.top_n}_{args.qa_scope}"
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.c4_csv.is_file():
        raise FileNotFoundError(f"C4 routing statistics not found: {args.c4_csv}")
    available = [(dataset, path) for dataset, path in (
        ("math_500", args.math_csv), ("gpqa_diamond", args.gpqa_csv)
    ) if path.is_file()]
    if not available:
        raise FileNotFoundError("Neither MATH-500 nor GPQA-Diamond routing statistics were found")
    c4 = read_routing_csv(args.c4_csv, "c4", "combined")
    rows = []
    for dataset, path in available:
        target = read_routing_csv(path, dataset, args.qa_scope)
        rows.extend(compare_domains(c4, target, dataset, args.top_n))
    for dataset, path in (("math_500", args.math_csv), ("gpqa_diamond", args.gpqa_csv)):
        if not path.is_file():
            print(f"Skipping {dataset}: {path} not found", flush=True)

    prefix = args.output_prefix
    outputs = {suffix: prefix.parent / f"{prefix.name}{suffix}" for suffix in (".png", ".pdf", ".csv", ".json")}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose another --output_prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    draw_figure(rows, outputs[".png"], outputs[".pdf"], args.top_n)
    with outputs[".csv"].open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=(
            "comparison", "layer", "metric", "top_n", "intersection", "union", "jaccard",
        ))
        writer.writeheader()
        writer.writerows(rows)
    outputs[".json"].write_text(json.dumps({
        "top_n": args.top_n,
        "qa_scope": args.qa_scope,
        "c4_csv": str(args.c4_csv.resolve()),
        "math_csv": str(args.math_csv.resolve()) if args.math_csv.is_file() else None,
        "gpqa_csv": str(args.gpqa_csv.resolve()) if args.gpqa_csv.is_file() else None,
        "metrics": [metric for metric, _, _, _ in METRICS],
    }, indent=2), encoding="utf-8")
    print(f"Saved {outputs['.png']} and {outputs['.pdf']}")


if __name__ == "__main__":
    main()
