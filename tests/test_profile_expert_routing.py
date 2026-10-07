import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gemq.profile_expert_routing import load_sequences, parse_args, parse_layers


class LayerSelectionTest(unittest.TestCase):
    def test_all_and_subset(self):
        self.assertEqual(parse_layers("all", 4), [0, 1, 2, 3])
        self.assertEqual(parse_layers("0,3", 4), [0, 3])

    def test_invalid(self):
        for spec in ("", "0,0", "4", "-1"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_layers(spec, 4)

    def test_quantized_model_uses_its_directory_name_without_changing_original_defaults(self):
        original = parse_args(["--dataset", "c4"])
        self.assertEqual(
            original.output_dir,
            Path("cache/routing_analysis/stats/c4/N128-L2048-Seed0"),
        )
        with tempfile.TemporaryDirectory() as temp:
            quantized = Path(temp) / "fake-quant-model"
            quantized.mkdir()
            args = parse_args([
                "--dataset", "c4", "--quantized_model_path", str(quantized),
            ])
            self.assertEqual(
                args.output_dir,
                Path("cache/routing_analysis/stats/fake-quant-model/c4/N128-L2048-Seed0"),
            )


class RoutingStatisticsTest(unittest.TestCase):
    def test_renormalized_mass_and_slices(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        from gemq.profile_expert_routing import routes_from_logits, scope_statistics

        logits = torch.tensor([[4.0, 2.0, 0.0], [0.0, 3.0, 1.0]])
        indices, mass = routes_from_logits(logits, 2)
        self.assertEqual(indices.tolist(), [[0, 1], [1, 2]])
        self.assertTrue(torch.allclose(mass.sum(-1), torch.ones(2)))
        counts, sums = scope_statistics(indices, mass, slice(None), 3)
        self.assertEqual(counts.tolist(), [1, 2, 1])
        self.assertAlmostEqual(sums.sum(), 2.0, places=6)
        prompt_counts, prompt_sums = scope_statistics(indices, mass, slice(0, 1), 3)
        self.assertEqual(prompt_counts.tolist(), [1, 1, 0])
        self.assertAlmostEqual(prompt_sums.sum(), 1.0, places=6)


class RecordValidationTest(unittest.TestCase):
    def test_saved_qa_ids_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "metadata.json").write_text(json.dumps({
                "dataset": "math_500", "model": "checkpoint", "nsamples": 1, "max_seq_len": 8,
            }), encoding="utf-8")
            records = directory / "records.jsonl"
            records.write_text(json.dumps({
                "id": 0, "prompt_ids": [1, 2], "answer_ids": [3, 4],
            }) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                dataset="math_500", model="checkpoint", nsamples=1, seqlen=8,
                records=records, quantized_model_path=directory / "fake-quant-model",
            )
            self.assertEqual(list(load_sequences(args, None)), [(0, [1, 2, 3, 4], 2)])
            args.dataset = "gpqa_diamond"
            with self.assertRaises(ValueError):
                list(load_sequences(args, None))


if __name__ == "__main__":
    unittest.main()
