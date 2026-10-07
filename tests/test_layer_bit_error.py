import csv
import tempfile
import unittest
from pathlib import Path

from gemq.plot_layer_bit_error import read_summary


class PlotInputTest(unittest.TestCase):
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
