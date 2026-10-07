"""Plot three fake-quant models' top-expert routing overlap with full precision."""

import argparse
import csv
from pathlib import Path

from gemq.compare_quant_routing_jaccard import (
    DATASETS,
    check_matching_metadata,
    compare_dataset,
    read_metadata,
)
from gemq.plot_routing_jaccard import read_routing_csv
from gemq.plot_style import configure_arial


DATASET_LABELS = {
    "c4": "C4",
    "math_500": "MATH-500",
    "gpqa_diamond": "GPQA-Diamond",
}
MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73")
MODEL_MARKERS = ("o", "s", "^")
CSV_COLUMNS = (
    "model_label", "quantized_model_name", "quantized_model_path", "dataset",
    "scope", "layer", "metric", "top_n", "intersection", "union", "jaccard",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized_model_paths", nargs=3, required=True, type=Path,
                        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
                        help="Three saved fake-quant model directories, in legend order")
    parser.add_argument("--labels", nargs=3, metavar=("LABEL_1", "LABEL_2", "LABEL_3"),
                        help="Short legend labels in the same order; defaults to directory names")
    parser.add_argument("--stats_root", type=Path, default=Path("cache/routing_analysis/stats"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qa_scope", choices=("combined", "prompt", "answer"), default="combined")
    parser.add_argument("--top_n", type=int, default=32)
    parser.add_argument("--output_prefix", required=True, type=Path,
                        help="Output path without extension; creates .csv, .png and .pdf")
    args = parser.parse_args(argv)
    if args.nsamples <= 0 or args.seqlen <= 0 or args.top_n <= 0:
        parser.error("--nsamples, --seqlen and --top_n must be positive")
    for path in args.quantized_model_paths:
        if not path.is_dir() or path.name in ("", ".", ".."):
            parser.error(f"--quantized_model_paths must contain named model directories: {path}")
    model_names = [path.name for path in args.quantized_model_paths]
    if len(set(model_names)) != 3:
        parser.error("The three model directory names must be distinct because they identify stats subdirectories")
    if args.labels is None:
        args.labels = model_names
    if any(not label.strip() for label in args.labels) or len(set(args.labels)) != 3:
        parser.error("--labels must contain three distinct, nonempty names")
    args.run_name = f"N{args.nsamples}-L{args.seqlen}-Seed{args.seed}"
    return args


def collect_rows(args):
    rows = []
    baseline_model = None
    baseline_layers = None
    for dataset in DATASETS:
        scope = "combined" if dataset == "c4" else args.qa_scope
        reference_dir = args.stats_root / dataset / args.run_name
        reference_metadata = read_metadata(reference_dir)
        if baseline_model is None:
            baseline_model = reference_metadata["model"]
            baseline_layers = reference_metadata["layers"]
        elif (reference_metadata["model"] != baseline_model
              or reference_metadata["layers"] != baseline_layers):
            raise ValueError(f"{dataset}: full-precision baseline model or profiled layers differ")
        reference = read_routing_csv(reference_dir / "routing_stats.csv", dataset, scope)
        for model_path, label in zip(args.quantized_model_paths, args.labels):
            quantized_dir = args.stats_root / model_path.name / dataset / args.run_name
            quantized_metadata = read_metadata(quantized_dir)
            check_matching_metadata(reference_metadata, quantized_metadata, dataset)
            recorded_name = quantized_metadata.get(
                "quantized_model_name", Path(quantized_metadata["quantized_model_path"]).name,
            )
            if recorded_name != model_path.name:
                raise ValueError(f"{dataset}: statistics under {model_path.name} belong to {recorded_name}")
            quantized = read_routing_csv(quantized_dir / "routing_stats.csv", dataset, scope)
            for row in compare_dataset(reference, quantized, dataset, scope, args.top_n):
                rows.append({
                    "model_label": label,
                    "quantized_model_name": model_path.name,
                    "quantized_model_path": str(model_path.resolve()),
                    **row,
                })
    return rows


def draw_figure(rows, labels, top_n, output_png, output_pdf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install with: pip install -e '.[plot]'") from exc

    configure_arial()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.65), sharey=True)
    legend_handles = []
    for ax, dataset in zip(axes, DATASETS):
        for index, label in enumerate(labels):
            points = sorted(
                (row for row in rows if row["dataset"] == dataset and row["model_label"] == label),
                key=lambda row: row["layer"],
            )
            if not points:
                raise ValueError(f"No Jaccard points for {dataset}, {label}")
            line, = ax.plot(
                [point["layer"] for point in points],
                [point["jaccard"] for point in points],
                color=MODEL_COLORS[index], marker=MODEL_MARKERS[index],
                linewidth=1.5, markersize=3.4, label=label,
            )
            if dataset == DATASETS[0]:
                legend_handles.append(line)
        ax.set_title(DATASET_LABELS[dataset], fontsize=11)
        ax.set_xlabel("MoE Layer Index")
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"Top-{top_n} Mean-Score Jaccard Similarity")
    fig.legend(legend_handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    outputs = {suffix: args.output_prefix.parent / f"{args.output_prefix.name}{suffix}"
               for suffix in (".csv", ".png", ".pdf")}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose another --output_prefix")
    rows = collect_rows(args)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    draw_figure(rows, args.labels, args.top_n, outputs[".png"], outputs[".pdf"])
    with outputs[".csv"].open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {outputs['.csv']}, {outputs['.png']} and {outputs['.pdf']}")


if __name__ == "__main__":
    main()
