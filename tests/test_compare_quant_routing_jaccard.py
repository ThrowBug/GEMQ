import csv
import json
import tempfile
import unittest
from pathlib import Path

from gemq.compare_quant_routing_jaccard import DATASETS, check_matching_metadata, main


def write_stats(directory, dataset, quantized_model_path=None):
    directory.mkdir(parents=True)
    records = None if dataset == "c4" else "/same/records.jsonl"
    scopes = ["combined"] if dataset == "c4" else ["combined", "prompt", "answer"]
    metadata = {
        "dataset": dataset,
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "records": records,
        "nsamples": 128,
        "seqlen": 2048,
        "seed": 0,
        "dtype": "bfloat16",
        "attn_impl": "sdpa",
        "layers": [5],
        "n_experts": 4,
        "top_k": 2,
        "scopes": scopes,
        "token_totals": {scope: 10 for scope in scopes},
        "mass_definition": "topk(softmax(router_logits)) / sum(topk(softmax(router_logits)))",
        "c4_source": "gemq.utils.data_utils.get_calib_loader" if dataset == "c4" else None,
    }
    if quantized_model_path is not None:
        metadata["quantized_model_path"] = str(quantized_model_path)
        metadata["quantized_model_name"] = quantized_model_path.name
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    means = [0.9, 0.2, 0.8, 0.1] if quantized_model_path is not None else [0.9, 0.8, 0.2, 0.1]
    with (directory / "routing_stats.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "dataset", "scope", "layer", "expert", "tokens", "activation_count",
            "normalized_probability_sum", "mean_probability_when_active",
        ])
        for scope in scopes:
            for expert, mean in enumerate(means):
                writer.writerow([dataset, scope, 5, expert, 10, 5, 5 * mean, mean])


class QuantizedRoutingJaccardTest(unittest.TestCase):
    def test_all_three_datasets_compare_mean_score_with_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "fake-quant-model"
            model.mkdir()
            stats = root / "stats"
            for dataset in DATASETS:
                write_stats(stats / dataset / "N128-L2048-Seed0", dataset)
                write_stats(stats / model.name / dataset / "N128-L2048-Seed0", dataset, model)
            output = root / "comparison.csv"
            main([
                "--quantized_model_path", str(model), "--stats_root", str(stats),
                "--top_n", "2", "--qa_scope", "answer", "--output_csv", str(output),
            ])
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["dataset"] for row in rows}, set(DATASETS))
            self.assertEqual({row["metric"] for row in rows}, {"mean_probability_when_active"})
            self.assertEqual({row["layer"] for row in rows}, {"5"})
            self.assertEqual({row["intersection"] for row in rows}, {"1"})
            self.assertEqual({row["union"] for row in rows}, {"3"})
            self.assertTrue(all(abs(float(row["jaccard"]) - 1 / 3) < 1e-12 for row in rows))
            self.assertEqual({row["scope"] for row in rows if row["dataset"] == "c4"}, {"combined"})
            self.assertEqual({row["scope"] for row in rows if row["dataset"] != "c4"}, {"answer"})
            with self.assertRaises(FileExistsError):
                main([
                    "--quantized_model_path", str(model), "--stats_root", str(stats),
                    "--top_n", "2", "--qa_scope", "answer", "--output_csv", str(output),
                ])

    def test_record_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference_dir, quantized_dir = root / "reference", root / "quantized"
            write_stats(reference_dir, "math_500")
            write_stats(quantized_dir, "math_500", root / "model")
            reference = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
            quantized = json.loads((quantized_dir / "metadata.json").read_text(encoding="utf-8"))
            quantized["records"] = "/other/records.jsonl"
            with self.assertRaisesRegex(ValueError, "records"):
                check_matching_metadata(reference, quantized, "math_500")


if __name__ == "__main__":
    unittest.main()
