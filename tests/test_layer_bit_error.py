import csv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

from gemq.plot_layer_bit_error import (
    main as plot_main,
    parse_args as parse_plot_args,
    read_series,
    read_summary,
    warn_if_incomparable,
)


class PlotInputTest(unittest.TestCase):
    @staticmethod
    def write_summary(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "bit_width,relative_mse\n1,0.5\n2,0.1\n3,0.03\n4,0.01\n",
            encoding="utf-8",
        )

    def test_read_four_bits_in_bit_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("bit_width", "relative_mse"))
                writer.writeheader()
                for bit, value in ((4, 0.01), (2, 0.1), (1, 0.5), (3, 0.03)):
                    writer.writerow({"bit_width": bit, "relative_mse": value})
            self.assertEqual(read_summary(path), [0.5, 0.1, 0.03, 0.01])

    def test_reject_missing_bit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.write_text("bit_width,relative_mse\n1,0.5\n2,0.1\n3,0.03\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_summary(path)

    def test_single_input_keeps_original_default_output(self):
        args = parse_plot_args(["--input", "measurements/summary.csv"])
        self.assertEqual(args.input, [Path("measurements/summary.csv")])
        self.assertEqual(args.output, Path("measurements/bit_width_relative_mse.png"))
        self.assertEqual((args.fig_width, args.fig_height), (4.6, 3.2))

    def test_multiple_inputs_require_separate_output(self):
        inputs = ["--input", "L5/summary.csv", "--input", "L24/summary.csv"]
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_plot_args(inputs)
        args = parse_plot_args(inputs + ["--output", "comparison.png"])
        self.assertEqual(len(args.input), 2)
        self.assertEqual(args.output, Path("comparison.png"))

    def test_existing_metadata_supplies_layer_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for layer in (5, 24, 42):
                path = Path(directory) / f"L{layer}-C4" / "summary.csv"
                self.write_summary(path)
                path.with_name("metadata.json").write_text(
                    json.dumps({"layer": layer, "dataset": "c4", "seed": 0}),
                    encoding="utf-8",
                )
                paths.append(path)
            series = [read_series(path, index) for index, path in enumerate(paths)]
            self.assertEqual(
                [item["label"] for item in series],
                ["Layer 5", "Layer 24", "Layer 42"],
            )
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                warn_if_incomparable(series)
            self.assertEqual(recorded, [])

    def test_missing_metadata_falls_back_and_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "L5-C4" / "summary.csv"
            second = Path(directory) / "older-result" / "summary.csv"
            self.write_summary(first)
            self.write_summary(second)
            first.with_name("metadata.json").write_text(
                json.dumps({"layer": 5, "dataset": "c4"}), encoding="utf-8"
            )
            series = [read_series(first, 0), read_series(second, 1)]
            self.assertEqual([item["label"] for item in series], ["Layer 5", "Series 2"])
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                warn_if_incomparable(series)
            self.assertTrue(any("lacks metadata" in str(w.message) for w in recorded))

    def test_mismatched_settings_warn_without_rejecting_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for layer, seed in ((5, 0), (24, 1)):
                path = Path(directory) / f"L{layer}" / "summary.csv"
                self.write_summary(path)
                path.with_name("metadata.json").write_text(
                    json.dumps({"layer": layer, "dataset": "c4", "seed": seed}),
                    encoding="utf-8",
                )
                paths.append(path)
            series = [read_series(path, index) for index, path in enumerate(paths)]
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                warn_if_incomparable(series)
            self.assertTrue(any("seed" in str(w.message) for w in recorded))

    def test_three_inputs_draw_three_curves_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for layer in (5, 24, 42):
                path = Path(directory) / f"L{layer}" / "summary.csv"
                self.write_summary(path)
                path.with_name("metadata.json").write_text(
                    json.dumps({"layer": layer, "seed": 0}), encoding="utf-8"
                )
                paths.append(path)
            figure, axes = Mock(), Mock()
            axes.spines = {"top": Mock(), "right": Mock()}
            matplotlib = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            matplotlib.use = Mock()
            pyplot.subplots = Mock(return_value=(figure, axes))
            pyplot.close = Mock()
            with patch.dict("sys.modules", {"matplotlib": matplotlib,
                                           "matplotlib.pyplot": pyplot}):
                argv = [item for path in paths for item in ("--input", str(path))]
                with patch("gemq.plot_layer_bit_error.configure_plot_font") as configure, redirect_stdout(StringIO()):
                    plot_main(argv + ["--font_size", "14", "--output", str(Path(directory) / "comparison.png")])
            configure.assert_called_once_with(14.0)
            self.assertEqual(axes.plot.call_count, 3)
            self.assertEqual(axes.legend.call_count, 1)
            axes.set_xlabel.assert_called_once_with("Bit Width", fontsize=14.0)
            axes.tick_params.assert_called_once_with(axis="both", labelsize=14.0)
            self.assertEqual(figure.savefig.call_count, 2)
            pyplot.subplots.assert_called_once_with(figsize=(4.6, 3.2))
            axes.spines["top"].set_visible.assert_called_once_with(False)
            axes.spines["right"].set_visible.assert_called_once_with(False)

    def test_single_input_still_draws_one_curve_without_legend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            self.write_summary(path)
            figure, axes = Mock(), Mock()
            axes.spines = {"top": Mock(), "right": Mock()}
            matplotlib = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            matplotlib.use = Mock()
            pyplot.subplots = Mock(return_value=(figure, axes))
            pyplot.close = Mock()
            with patch.dict("sys.modules", {"matplotlib": matplotlib,
                                           "matplotlib.pyplot": pyplot}):
                with patch("gemq.plot_layer_bit_error.configure_plot_font") as configure, redirect_stdout(StringIO()):
                    plot_main(["--input", str(path)])
            configure.assert_called_once_with(10.0)
            self.assertEqual(axes.plot.call_count, 1)
            axes.legend.assert_not_called()
            self.assertEqual(figure.savefig.call_count, 2)
            axes.spines["top"].set_visible.assert_called_once_with(False)
            axes.spines["right"].set_visible.assert_called_once_with(False)


class MeasurementMathTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")

    def test_squared_error_sums(self):
        import torch
        from gemq.measure_layer_bit_error import squared_error_sums

        reference = torch.tensor([[[1., 0.], [0., 2.]]])
        candidate = torch.tensor([[[0., 0.], [0., 1.]]])
        error, baseline = squared_error_sums(reference, candidate)
        self.assertEqual(error, 2.)
        self.assertEqual(baseline, 5.)

    def test_candidate_bits_reuse_fp_weight_and_hessian(self):
        import torch
        from torch import nn
        from types import SimpleNamespace
        from gemq.measure_layer_bit_error import install_bit_weights, restore_fp_weights
        from gemq.quantizers.gptq import GPTQWeightQuantizer

        linear = nn.Linear(4, 2, bias=False)
        original = linear.weight.detach().clone()
        master = GPTQWeightQuantizer(original, "toy", 2, 2, 0.01, 2,
                                     False, False, True)
        master.H = torch.eye(4)
        fp_hessian = master.H.clone()
        args = SimpleNamespace(blocksize=2, percdamp=0.01)
        entries = [(0, "gate_proj", linear, master)]
        for bit in (1, 2, 3, 4):
            install_bit_weights(entries, bit, args)
            self.assertTrue(torch.equal(master.H, fp_hessian))
            restore_fp_weights(entries)
            self.assertTrue(torch.equal(linear.weight, original))


if __name__ == "__main__":
    unittest.main()
