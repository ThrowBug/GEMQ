"""Plot bit-width vs relative MoE output MSE from measure_layer_bit_error CSV."""

import argparse
import csv
import math
from pathlib import Path


BITS = (1, 2, 3, 4)


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="summary.csv from measurement")
    parser.add_argument("--output", type=Path, help="PNG output; PDF is saved alongside it")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    parser.add_argument("--color", default="#0072B2")
    parser.add_argument("--marker", default="o")
    parser.add_argument("--fig_width", type=float, default=3.6)
    parser.add_argument("--fig_height", type=float, default=2.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="")
    args = parser.parse_args(argv)
    if args.fig_width <= 0 or args.fig_height <= 0 or args.dpi <= 0:
        parser.error("figure dimensions and DPI must be positive")
    if args.output is None:
        args.output = args.input.with_name("bit_width_relative_mse.png")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    return args


def main(argv=None):
    args = parse_args(argv)
    values = read_summary(args.input)
    if args.yscale == "log" and any(value <= 0 for value in values):
        raise ValueError("Log scale requires all relative MSE values to be positive")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    ax.plot(BITS, values, color=args.color, marker=args.marker, linewidth=1.8,
            markersize=6)
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
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
