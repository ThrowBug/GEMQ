import csv
import json
import tempfile
import unittest
from pathlib import Path

from gemq.collect_dialog_examples import (
    collect, dataset_file_parts, get_score_value, main, response_text,
)


def write_run(root, label, dataset, rows, subset="default", flat=False):
    run = root / label
    for kind in ("predictions", "reviews"):
        folder = run / kind / label
        if not flat:
            folder /= dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (f"{dataset}.jsonl" if flat else f"{subset}.jsonl")
        with path.open("w", encoding="utf-8") as output:
            for index, prompt, correct, answer in rows:
                if kind == "predictions":
                    row = {
                        "index": index, "input": prompt,
                        "model_output": {"choices": [{"message": {
                            "role": "assistant", "content": answer}}]},
                    }
                else:
                    row = {
                        "index": index, "input": prompt, "target": "reference",
                        "sample_score": {"score": {"value": {"acc": float(correct)},
                                                  "prediction": answer}},
                    }
                output.write(json.dumps(row) + "\n")
    return run


class DialogExamplesTest(unittest.TestCase):
    def test_one_dataset_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ours = write_run(root, "ours", "math_500", [(0, "Q0", True, "ours answer"),
                                                           (1, "Q1", False, "bad")], "Level 1")
            base = write_run(root, "base", "math_500", [(0, "Q0", False, "baseline answer"),
                                                           (1, "Q1", True, "good")], "Level 1")
            fp = write_run(root, "fp", "math_500", [(0, "Q0", True, "fp answer"),
                                                       (1, "Q1", True, "fp")], "Level 1")
            output = root / "out"
            main(["--dataset", "math_500", "--run", f"ours={ours}", "--run", f"base={base}",
                  "--run", f"fp={fp}", "--ours", "ours", "--reference", "fp",
                  "--output_dir", str(output)])
            with (output / "candidates.csv").open(encoding="utf-8-sig", newline="") as source:
                candidates = list(csv.DictReader(source))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["sample_id"], "0")
            self.assertEqual(candidates[0]["failed_baselines"], "base")
            samples = [json.loads(line) for line in (output / "all_samples.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["models"]["ours"]["response"], "ours answer")
            report = json.loads((output / "alignment_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["candidate"], 1)
            with self.assertRaises(FileExistsError):
                main(["--dataset", "math_500", "--run", f"ours={ours}",
                      "--run", f"base={base}", "--run", f"fp={fp}",
                      "--ours", "ours", "--reference", "fp", "--output_dir", str(output)])

    def test_prompt_mismatch_and_missing_score_are_not_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ours = write_run(root, "ours", "gsm8k", [(0, "same", True, "yes"),
                                                        (1, "Q1", True, "yes")])
            base = write_run(root, "base", "gsm8k", [(0, "different", False, "no"),
                                                        (1, "Q1", False, "no")])
            fp = write_run(root, "fp", "gsm8k", [(0, "same", True, "yes"),
                                                    (1, "Q1", True, "yes")])
            review_file = base / "reviews" / "base" / "gsm8k" / "default.jsonl"
            rows = [json.loads(line) for line in review_file.read_text(encoding="utf-8").splitlines()]
            rows[1]["sample_score"]["score"]["value"] = {"unrecognized": 0.0}
            review_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            samples, report = collect("gsm8k", {"ours": ours, "base": base, "fp": fp},
                                      "ours", "fp")
            self.assertFalse(any(sample["candidate"] for sample in samples))
            self.assertEqual(report["counts"]["prompt_mismatch"], 1)
            self.assertEqual(report["counts"]["unknown_score"], 1)

    def test_ifeval_uses_prompt_level_strict_and_nested_score(self):
        review = {"sample_score": {"score": {"value": {
            "prompt_level_strict": 0.0, "inst_level_strict": 1.0,
        }}}}
        self.assertIs(get_score_value(review, "prompt_level_strict"), False)
        self.assertIsNone(get_score_value(review, "acc"))
        self.assertIs(get_score_value({"score": {"score": {"value": {"acc": 1.0}}}}, "acc"), True)

    def test_flat_and_nested_files_and_tool_calls(self):
        self.assertEqual(dataset_file_parts(Path("model/mmlu_redux.jsonl"), "mmlu_redux"),
                         ("model", "default"))
        self.assertEqual(dataset_file_parts(Path("model/math_500/Level 1.jsonl"), "math_500"),
                         ("model", "Level 1"))
        self.assertEqual(dataset_file_parts(Path("model/bfcl_v3@multi_turn.jsonl"), "bfcl_v3"),
                         ("model", "multi_turn"))
        text = response_text({"model_output": {"choices": [{"message": {
            "content": "", "tool_calls": [{"function": {"name": "weather", "arguments": "{}"}}],
        }}]}}, {})
        self.assertIn("weather", text)

    def test_duplicate_sample_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ours = write_run(root, "ours", "gpqa_diamond", [(0, "Q", True, "A"),
                                                               (0, "Q", True, "A")])
            base = write_run(root, "base", "gpqa_diamond", [(0, "Q", False, "B")])
            fp = write_run(root, "fp", "gpqa_diamond", [(0, "Q", True, "A")])
            with self.assertRaisesRegex(ValueError, "Duplicate prediction sample"):
                collect("gpqa_diamond", {"ours": ours, "base": base, "fp": fp},
                        "ours", "fp")

    def test_no_shared_ids_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ours = write_run(root, "ours", "mmlu_redux", [(0, "Q0", True, "A")], flat=True)
            base = write_run(root, "base", "mmlu_redux", [(1, "Q1", False, "B")], flat=True)
            fp = write_run(root, "fp", "mmlu_redux", [(0, "Q0", True, "A")], flat=True)
            with self.assertRaisesRegex(ValueError, "No shared mmlu_redux sample IDs"):
                collect("mmlu_redux", {"ours": ours, "base": base, "fp": fp},
                        "ours", "fp")

    def test_only_all_baselines_wrong_with_fp_and_ours_correct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [(0, "Q0", True, "answer"), (1, "Q1", True, "answer"),
                    (2, "Q2", True, "answer")]
            ours = write_run(root, "ours", "math_500", rows)
            fp = write_run(root, "fp", "math_500", [(*row[:2], index != 2, "answer")
                                                     for index, row in enumerate(rows)])
            base_a = write_run(root, "base_a", "math_500", [
                (0, "Q0", False, "wrong"), (1, "Q1", False, "wrong"),
                (2, "Q2", False, "wrong")])
            base_b = write_run(root, "base_b", "math_500", [
                (0, "Q0", True, "right"), (1, "Q1", False, "wrong"),
                (2, "Q2", False, "wrong")])
            samples, report = collect("math_500", {
                "ours": ours, "fp": fp, "base_a": base_a, "base_b": base_b,
            }, "ours", "fp")
            self.assertEqual([sample["sample_id"] for sample in samples if sample["candidate"]],
                             ["1"])
            self.assertEqual(report["counts"]["candidate"], 1)

    def test_math_and_gpqa_prompts_come_from_user_messages_not_answer_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dataset, question, metadata in (
                ("math_500", "What is 2 + 2?", {"solution": "4", "question_id": "m1"}),
                ("gpqa_diamond", "What is the energy? A. 1 B. 2 C. 3 D. 4",
                 {"correct_answer": "1", "incorrect_answers": ["2", "3", "4"]}),
            ):
                with self.subTest(dataset=dataset):
                    runs = {}
                    for label, correct in (("ours", True), ("fp", True), ("base", False)):
                        run = write_run(root, dataset + label, dataset,
                                        [(0, question, correct, "answer")])
                        for kind in ("predictions", "reviews"):
                            path = run / kind / (dataset + label) / dataset / "default.jsonl"
                            row = json.loads(path.read_text(encoding="utf-8"))
                            row.pop("input")
                            row["messages"] = [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": "answer"},
                            ]
                            row["metadata"] = metadata
                            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                        runs[label] = run
                    samples, _ = collect(dataset, runs, "ours", "fp")
                    self.assertEqual(samples[0]["prompt"], question)
                    self.assertEqual(samples[0]["prompt_source"], "messages.user")
                    self.assertTrue(samples[0]["candidate"])

                    if dataset == "gpqa_diamond":
                        # The same answer metadata cannot validate differently ordered choices.
                        path = runs["base"] / "reviews" / (dataset + "base") / dataset / "default.jsonl"
                        row = json.loads(path.read_text(encoding="utf-8"))
                        row["messages"][0]["content"] = "What is the energy? B. 1 A. 2 C. 3 D. 4"
                        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                        samples, report = collect(dataset, runs, "ours", "fp")
                        self.assertFalse(samples[0]["candidate"])
                        self.assertEqual(report["counts"]["prompt_mismatch"], 1)

                    # Answer-only metadata is not a question and cannot verify alignment.
                    for run in runs.values():
                        for kind in ("predictions", "reviews"):
                            path = run / kind / run.name / dataset / "default.jsonl"
                            row = json.loads(path.read_text(encoding="utf-8"))
                            row.pop("messages")
                            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    samples, report = collect(dataset, runs, "ours", "fp")
                    self.assertEqual(samples[0]["prompt"], "")
                    self.assertFalse(samples[0]["candidate"])
                    self.assertEqual(report["counts"]["prompt_unverified"], 1)


if __name__ == "__main__":
    unittest.main()
