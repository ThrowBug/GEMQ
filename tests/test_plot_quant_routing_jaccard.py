import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemq.compare_quant_routing_jaccard import DATASETS
from gemq.plot_quant_routing_jaccard import draw_figure, main, parse_args
from tests.test_compare_quant_routing_jaccard import write_stats


class ThreeModelRoutingPlotTest(unittest.TestCase):
    def test_three_models_and_datasets_export_csv_png_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stats = root / "stats"
            models = [root / name for name in ("quant-a", "quant-b", "quant-c")]
            for model in models:
                model.mkdir()
            for dataset in DATASETS:
                write_stats(stats / dataset / "N128-L2048-Seed0", dataset)
                for model in models:
                    write_stats(stats / model.name / dataset / "N128-L2048-Seed0", dataset, model)
            prefix = root / "figures" / "three_models"
            argv = [
                "--quantized_model_paths", *(str(model) for model in models),
                "--labels", "Model A", "Model B", "Model C",
                "--stats_root", str(stats), "--top_n", "2",
                "--qa_scope", "answer", "--font_size", "12", "--output_prefix", str(prefix),
            ]

            def fake_draw(rows, labels, top_n, png, pdf, font_size):
                self.assertEqual(len(rows), 9)
                self.assertEqual(labels, ["Model A", "Model B", "Model C"])
                self.assertEqual(top_n, 2)
                self.assertEqual(font_size, 12.0)
                png.touch()
                pdf.touch()

            with patch("gemq.plot_quant_routing_jaccard.draw_figure", side_effect=fake_draw):
                main(argv)
            self.assertTrue(prefix.with_suffix(".png").is_file())
            self.assertTrue(prefix.with_suffix(".pdf").is_file())
            with prefix.with_suffix(".csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 9)
            self.assertEqual({row["dataset"] for row in rows}, set(DATASETS))
            self.assertEqual({row["model_label"] for row in rows}, {"Model A", "Model B", "Model C"})
            self.assertEqual({row["metric"] for row in rows}, {"mean_probability_when_active"})
            self.assertTrue(all(abs(float(row["jaccard"]) - 1 / 3) < 1e-12 for row in rows))
            self.assertEqual({row["scope"] for row in rows if row["dataset"] == "c4"}, {"combined"})
            self.assertEqual({row["scope"] for row in rows if row["dataset"] != "c4"}, {"answer"})
            with self.assertRaises(FileExistsError):
                main(argv)

    def test_directory_names_are_default_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            models = [Path(temp) / name for name in ("first", "second", "third")]
            for model in models:
                model.mkdir()
            args = parse_args([
                "--quantized_model_paths", *(str(model) for model in models),
                "--output_prefix", str(Path(temp) / "comparison"),
            ])
            self.assertEqual(args.labels, ["first", "second", "third"])

    def test_missing_stats_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = [root / name for name in ("first", "second", "third")]
            for model in models:
                model.mkdir()
            prefix = root / "plot"
            with self.assertRaises(FileNotFoundError):
                main([
                    "--quantized_model_paths", *(str(model) for model in models),
                    "--output_prefix", str(prefix),
                ])
            self.assertFalse(prefix.with_suffix(".csv").exists())
            self.assertFalse(prefix.with_suffix(".png").exists())
            self.assertFalse(prefix.with_suffix(".pdf").exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_actual_figure_renders_when_matplotlib_is_available(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = ["Model A", "Model B", "Model C"]
            rows = [
                {"dataset": dataset, "model_label": label, "layer": layer, "jaccard": 0.5}
                for dataset in DATASETS for label in labels for layer in (5, 24, 42)
            ]
            png, pdf = root / "plot.png", root / "plot.pdf"
            draw_figure(rows, labels, 32, png, pdf)
            self.assertGreater(png.stat().st_size, 0)
            self.assertGreater(pdf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
