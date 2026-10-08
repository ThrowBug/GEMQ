import csv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gemq.plot_cumulative_bit_error import main as plot_main, read_measurements


def write_measurements(path, *, missing=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("layer", "bit_width", "relative_mse"))
        for layer in range(2):
            for bit in (1, 2, 3, 4):
                if (layer, bit) != missing:
                    writer.writerow((layer, bit, (layer + 1) / bit))


class CumulativePlotTest(unittest.TestCase):
    def test_reads_four_complete_bit_curves(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cumulative_relative_mse.csv"
            write_measurements(path)
            layers, curves = read_measurements(path)
            self.assertEqual(layers, [0, 1])
            self.assertEqual(curves[1], [1.0, 2.0])
            self.assertEqual(curves[3], [1 / 3, 2 / 3])
            self.assertEqual(curves[4], [0.25, 0.5])

    def test_rejects_missing_layer_bit_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cumulative_relative_mse.csv"
            write_measurements(path, missing=(1, 2))
            with self.assertRaisesRegex(ValueError, "missing layer/bit"):
                read_measurements(path)

    def test_draws_four_curves_and_both_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cumulative_relative_mse.csv"
            write_measurements(path)
            output = Path(temp) / "figure.png"
            figure, axes = Mock(), Mock()
            axes.spines = {"top": Mock(), "right": Mock()}
            matplotlib = types.ModuleType("matplotlib")
            pyplot = types.ModuleType("matplotlib.pyplot")
            matplotlib.use = Mock()
            pyplot.subplots = Mock(return_value=(figure, axes))
            pyplot.close = Mock()
            with patch.dict("sys.modules", {"matplotlib": matplotlib,
                                           "matplotlib.pyplot": pyplot}):
                with patch("gemq.plot_cumulative_bit_error.configure_plot_font") as configure:
                    with redirect_stdout(StringIO()):
                        plot_main(["--input", str(path), "--output", str(output),
                                   "--font_size", "12"])
            configure.assert_called_once_with(12.0)
            self.assertEqual(axes.plot.call_count, 4)
            self.assertEqual(figure.savefig.call_count, 2)
            axes.set_xlabel.assert_called_once_with("Decoder Layer Index", fontsize=12.0)
            axes.spines["top"].set_visible.assert_called_once_with(False)
            axes.spines["right"].set_visible.assert_called_once_with(False)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_actual_plot_renders(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cumulative_relative_mse.csv"
            write_measurements(path)
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                plot_main(["--input", str(path)])
            self.assertGreater(path.with_suffix(".png").stat().st_size, 0)
            self.assertGreater(path.with_suffix(".pdf").stat().st_size, 0)


class CumulativeMeasurementTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed")

    def test_quantized_hidden_states_propagate_between_layers(self):
        import torch
        from gemq import measure_cumulative_bit_error as measurement

        class ToyLayer:
            def __init__(self):
                self.quant_error = 0.0

            def to(self, _device):
                return self

        layers = [ToyLayer(), ToyLayer()]
        initial = [torch.ones(1, 1, 1), torch.ones(1, 1, 1)]

        def forward(layer, hidden, _positional, _keyword, _device):
            return [item * 2 + layer.quant_error for item in hidden]

        def install(entries, bit, _args):
            entries[0].quant_error = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125}[bit]

        def restore(entries):
            entries[0].quant_error = 0.0

        with patch.object(measurement, "get_blocks", return_value=layers), \
             patch.object(measurement, "_capture_decoder_inputs",
                          return_value=(initial, [(), ()], [{}, {}])), \
             patch.object(measurement, "get_moe_block", side_effect=lambda layer, _: layer), \
             patch.object(measurement, "validate_routed_moe", return_value=True), \
             patch.object(measurement, "_capture_qwen3_moe_inputs",
                          return_value=([initial[0]], [], [], [])), \
             patch.object(measurement, "collect_fp_hessians",
                          side_effect=lambda layer, *_: [layer]), \
             patch.object(measurement, "_forward_layer_batches", side_effect=forward), \
             patch.object(measurement, "install_bit_weights", side_effect=install), \
             patch.object(measurement, "restore_fp_weights", side_effect=restore):
            with redirect_stdout(StringIO()):
                rows, routed = measurement.measure_layers(
                    None, [None, None], types.SimpleNamespace(calib_samples=1),
                    torch.device("cpu"),
                )
        self.assertEqual(routed, [0, 1])
        one_bit = [row for row in rows if row["bit_width"] == 1]
        self.assertEqual([row["relative_mse"] for row in one_bit], [0.25, 9 / 16])
        self.assertEqual({row["bit_width"] for row in rows}, {1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
