"""Plot bit-width vs relative MoE output MSE from measure_layer_bit_error CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import warnings


BITS = (1, 2, 3, 4)
COLORS = ("#B4FFFD", "#ADFE88", "#F1FA74", "#FCBEE0", "#FFB10A")
MARKERS = ("o", "s", "^", "D", "v")
COMPARISON_FIELDS = (
    "model", "dataset", "calib_samples", "eval_samples", "seqlen",
    "seed", "bits", "attn_impl", "gptq", "metric", "context",
)


def read_summary(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"bit_width", "relative_mse"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or ())
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            bit = int(row["bit_width"])
            value = float(row["relative_mse"])
            if bit not in BITS or bit in rows or not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid or duplicate bit/error row: {row}")
            rows[bit] = value
    if set(rows) != set(BITS):
        raise ValueError(f"{path}: expected exactly 1, 2, 3, 4 bit rows; got {sorted(rows)}")
    return [rows[bit] for bit in BITS]


def read_series(path, index):
    """Read one existing measurement; metadata is optional for older CSVs."""
    path = Path(path)
    metadata_path = path.with_name("metadata.json")
    metadata = None
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_path}: expected a JSON object")

    layer = metadata.get("layer") if metadata is not None else None
    if isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0:
        label = f"Layer {layer}"
    else:
        match = re.match(r"^L(\d+)(?:-|$)", path.parent.name)
        label = f"Layer {match.group(1)}" if match else f"Series {index + 1}"
    return {"path": path, "label": label, "values": read_summary(path),
            "metadata": metadata}


def warn_if_incomparable(series):
    """Keep plotting possible, but flag mismatched or unverifiable settings."""
    if len(series) < 2:
        return
    if any(item["metadata"] is None for item in series):
        warnings.warn(
            "At least one input lacks metadata.json; comparison settings cannot be fully verified.",
            stacklevel=2,
        )
    reference = series[0]["metadata"]
    if reference is None:
        return
    for item in series[1:]:
        current = item["metadata"]
        if current is None:
            continue
        mismatches = [
            field for field in COMPARISON_FIELDS
            if field in reference and field in current
            and reference[field] != current[field]
        ]
        if mismatches:
            warnings.warn(
                f"{item['path']}: settings differ from {series[0]['path']} "
                f"in {', '.join(mismatches)}",
                stacklevel=2,
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, action="append",
        help="summary.csv from measurement; repeat for multiple layers",
    )
    parser.add_argument("--output", type=Path, help="PNG output; PDF is saved alongside it")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    parser.add_argument("--color", default="#B4FFFD")
    parser.add_argument("--marker", default="o")
    parser.add_argument("--fig_width", type=float, default=4.6)
    parser.add_argument("--fig_height", type=float, default=3.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)
    if args.fig_width <= 0 or args.fig_height <= 0 or args.dpi <= 0:
        parser.error("figure dimensions and DPI must be positive")
    if len({path.resolve() for path in args.input}) != len(args.input):
        parser.error("--input paths must be distinct")
    if args.output is None:
        if len(args.input) > 1:
            parser.error("--output is required when plotting multiple inputs")
        args.output = args.input[0].with_name("bit_width_relative_mse.png")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    return args


def main(argv=None):
    args = parse_args(argv)
    series = [read_series(path, index) for index, path in enumerate(args.input)]
    labels = [item["label"] for item in series]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate layer labels in inputs: {labels}")
    warn_if_incomparable(series)
    if args.yscale == "log" and any(
        value <= 0 for item in series for value in item["values"]
    ):
        raise ValueError("Log scale requires all relative MSE values to be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    for index, item in enumerate(series):
        color = args.color if index == 0 else COLORS[index % len(COLORS)]
        marker = args.marker if index == 0 else MARKERS[index % len(MARKERS)]
        ax.plot(
            BITS, item["values"], color=color, marker=marker,
            linewidth=1.8, markersize=6,
            label=item["label"] if len(series) > 1 else None,
        )
    ax.set_xticks(BITS)
    ax.set_xlim(0.8, 4.2)
    ax.set_xlabel("Bit-width")
    ax.set_ylabel("Relative MoE output MSE")
    ax.set_yscale(args.yscale)
    if args.yscale == "linear":
        ax.set_ylim(bottom=0)
    if args.title:
        ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
