import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gemq.plot_routing_jaccard import compare_domains, draw_figure, main, read_routing_csv, top_experts


def write_stats(path, dataset, counts, means):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "dataset", "scope", "layer", "expert", "tokens", "activation_count",
            "normalized_probability_sum", "mean_probability_when_active",
        ])
        for expert, (count, mean) in enumerate(zip(counts, means)):
            writer.writerow([dataset, "combined", 0, expert, 10, count,
                             count * mean if mean is not None else 0,
                             mean if mean is not None else ""])


class JaccardPlotDataTest(unittest.TestCase):
    def test_figure_uses_larger_text_sizes(self):
        figure, axes = Mock(), Mock()
        axes.spines = {"top": Mock(), "right": Mock()}
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib.use = Mock()
        pyplot.subplots = Mock(return_value=(figure, axes))
        pyplot.close = Mock()
        rows = [{
            "comparison": "math_500_vs_c4", "metric": "activation_count",
            "layer": 5, "jaccard": 0.5,
        }]
        with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}):
            with patch("gemq.plot_routing_jaccard.configure_plot_font") as configure:
                draw_figure(rows, "plot.png", "plot.pdf", 32)
        configure.assert_called_once_with(16.0)
        axes.set_xlabel.assert_called_once_with("MoE Layer Index", fontsize=16)
        axes.set_ylabel.assert_called_once_with("Jaccard Similarity", fontsize=16)
        axes.tick_params.assert_called_once_with(axis="both", labelsize=16)
        axes.legend.assert_called_once_with(frameon=False, ncol=1, fontsize=13, loc="best")
        self.assertEqual(figure.savefig.call_count, 2)

    def test_count_and_mean_use_different_top_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            c4_path, math_path = Path(temp) / "c4.csv", Path(temp) / "math.csv"
            write_stats(c4_path, "c4", [10, 9, 8, 7], [0.2, 0.1, 0.9, 0.8])
            write_stats(math_path, "math_500", [12, 5, 11, 4], [0.2, 0.1, 0.95, 0.85])
            c4 = read_routing_csv(c4_path, "c4", "combined")
            math = read_routing_csv(math_path, "math_500", "combined")
            rows = compare_domains(c4, math, "math_500", 2)
            by_metric = {row["metric"]: row for row in rows}
            self.assertAlmostEqual(by_metric["activation_count"]["jaccard"], 1 / 3)
            self.assertEqual(by_metric["mean_probability_when_active"]["jaccard"], 1.0)
            self.assertEqual(top_experts(c4[0], "activation_count", 2), {0, 1})

    def test_inactive_expert_has_no_mean_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "c4.csv"
            write_stats(path, "c4", [10, 0, 8], [0.2, None, 0.9])
            experts = read_routing_csv(path, "c4", "combined")[0]
            self.assertEqual(top_experts(experts, "mean_probability_when_active", 2), {0, 2})
            with self.assertRaises(ValueError):
                top_experts(experts, "activation_count", 3)

    def test_missing_gpqa_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            c4_path, math_path = directory / "c4.csv", directory / "math.csv"
            write_stats(c4_path, "c4", [10, 9, 8, 7], [0.2, 0.1, 0.9, 0.8])
            write_stats(math_path, "math_500", [12, 5, 11, 4], [0.2, 0.1, 0.95, 0.85])
            prefix = directory / "plot"

            def fake_draw(_rows, png, pdf, _top_n, font_size):
                self.assertEqual(font_size, 14.0)
                png.touch()
                pdf.touch()

            argv = [
                "--c4_csv", str(c4_path), "--math_csv", str(math_path),
                "--gpqa_csv", str(directory / "missing.csv"),
                "--top_n", "2", "--font_size", "14", "--output_prefix", str(prefix),
            ]
            with patch("gemq.plot_routing_jaccard.draw_figure", side_effect=fake_draw):
                main(argv)
            self.assertTrue((directory / "plot.png").is_file())
            self.assertTrue((directory / "plot.pdf").is_file())
            with (directory / "plot.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["comparison"] for row in rows}, {"math_500_vs_c4"})
            (directory / "plot.csv").write_text("obsolete", encoding="utf-8")
            (directory / "plot.json").write_text("obsolete", encoding="utf-8")
            with patch("gemq.plot_routing_jaccard.draw_figure", side_effect=fake_draw):
                main(argv)
            with (directory / "plot.csv").open(newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)
            self.assertIn('"top_n": 2', (directory / "plot.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
