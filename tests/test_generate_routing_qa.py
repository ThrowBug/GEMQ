import json
import tempfile
import unittest
from pathlib import Path

from gemq.generate_routing_qa import existing_record_count, select_fitting_prompts


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return messages[0]["ids"]


class LongPromptTest(unittest.TestCase):
    def test_skip_and_refill_without_truncating(self):
        candidates = [
            {"source_position": position, "messages": [{"ids": ids}]}
            for position, ids in enumerate(([1, 2], [3, 4], [5] * 9, [6, 7], [8, 9]))
        ]
        selected, skipped = select_fitting_prompts(candidates, FakeTokenizer(), 3, 8)
        self.assertEqual([item["source_position"] for item in selected], [0, 1, 3])
        self.assertEqual([item["prompt_ids"] for item in selected], [[1, 2], [3, 4], [6, 7]])
        self.assertEqual(skipped, [{"source_position": 2, "prompt_tokens": 9}])

    def test_resume_old_partial_records(self):
        prompts = [
            {"source_position": 10, "prompt_ids": [1, 2]},
            {"source_position": 12, "prompt_ids": [3, 4]},
        ]
        metadata = {"format_version": 1, "dataset": "math_500", "model": "m",
                    "nsamples": 2, "max_seq_len": 8, "max_new_tokens": 8,
                    "seed": 0, "dtype": "bfloat16", "attn_impl": "sdpa",
                    "local_dataset": None, "evalscope": {"evalscope_seed": 42}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (directory / "records.jsonl").write_text(json.dumps({
                "id": 0, "source_position": 10, "prompt_ids": [1, 2], "answer_ids": [5],
            }) + "\n", encoding="utf-8")
            self.assertEqual(existing_record_count(directory, metadata, prompts), 1)
            prompts[0]["source_position"] = 99
            with self.assertRaises(ValueError):
                existing_record_count(directory, metadata, prompts)


if __name__ == "__main__":
    unittest.main()
