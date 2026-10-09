"""Plot propagation of four layer-0 expert bit widths through 2-bit layers."""

import argparse
import csv
import math
from pathlib import Path

from gemq.plot_style import configure_plot_font


INITIAL_BITS = (1, 2, 3, 4)
COLORS = {1: "#D55E00", 2: "#0072B2", 3: "#009E73", 4: "#CC79A7"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


def read_measurements(path, max_layers=None):
    values = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"layer", "initial_bit_width", "current_bit_width", "relative_mse"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            layer = int(row["layer"])
            bit = int(row["initial_bit_width"])
            current_bit = int(row["current_bit_width"])
            value = float(row["relative_mse"])
            valid_current = (bit,) if layer == 0 else (2, 16)
            if (layer < 0 or bit not in INITIAL_BITS or current_bit not in valid_current
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{path}: invalid layer, bit width or relative MSE: {row}")
            if (layer, bit) in values:
                raise ValueError(f"{path}: duplicate layer {layer}, {bit}-bit result")
            values[layer, bit] = value
    layers = sorted({layer for layer, _ in values})
    if not layers or layers != list(range(layers[-1] + 1)):
        raise ValueError(f"{path}: expected consecutive layer indices starting at zero")
    missing = [(layer, bit) for layer in layers for bit in INITIAL_BITS if (layer, bit) not in values]
    if missing:
        raise ValueError(f"{path}: missing layer/bit results: {missing[:12]}")
    if max_layers is not None:
        if max_layers <= 0 or max_layers > len(layers):
            raise ValueError(f"--max_layers must be between 1 and {len(layers)} for {path}")
        layers = layers[:max_layers]
    return layers, {bit: [values[layer, bit] for layer in layers] for bit in INITIAL_BITS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="layer0_bit_propagation.csv from the measurement script")
    parser.add_argument("--output", type=Path,
                        help="PNG output; a PDF is saved alongside it")
    parser.add_argument("--fig_width", type=float, default=6.0)
    parser.add_argument("--fig_height", type=float, default=3.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font_size", type=float, default=10.0,
                        help="Base font size in points (default: 10)")
    parser.add_argument("--max_layers", type=int,
                        help="Plot only the first N decoder layers (indices 0 through N-1)")
    args = parser.parse_args(argv)
    if args.fig_width <= 0 or args.fig_height <= 0 or args.dpi <= 0 or args.font_size <= 0:
        parser.error("figure dimensions, DPI and --font_size must be positive")
    if args.max_layers is not None and args.max_layers <= 0:
        parser.error("--max_layers must be positive")
    if args.output is None:
        suffix = f"_first{args.max_layers}" if args.max_layers is not None else ""
        args.output = args.input.with_name(f"{args.input.stem}{suffix}.png")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    return args


def draw_figure(layers, curves, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install with: pip install -e '.[plot]'") from exc

    configure_plot_font(args.font_size)
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    for bit in INITIAL_BITS:
        ax.plot(
            layers, curves[bit], color=COLORS[bit], marker=MARKERS[bit],
            linewidth=1.8, markersize=4.5, label=f"Layer 0: {bit} Bit" if bit == 1 else f"Layer 0: {bit} Bits",
        )
    ax.set_xlabel("Decoder Layer Index", fontsize=args.font_size)
    ax.set_ylabel("Cumulative Relative MSE", fontsize=args.font_size)
    if len(layers) == 1:
        ax.set_xlim(layers[0] - 0.5, layers[0] + 0.5)
    else:
        ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(axis="both", labelsize=args.font_size)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=args.font_size * 0.9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    layers, curves = read_measurements(args.input, args.max_layers)
    draw_figure(layers, curves, args)
    print(f"Saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
