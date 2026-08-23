from types import SimpleNamespace

import pytest
import torch

from gemq.utils.data_utils import get_calib_loader
from gemq.utils.mixed_calib import (
    allocate_source_blocks,
    build_mixed_chat_en_calibration,
    is_english_wildchat,
    load_prepared_calibration,
    save_prepared_calibration,
)


class FakeTokenizer:
    eos_token_id = 0
    name_or_path = "fake/tokenizer"
    chat_template = "fake-chat-template"
    init_kwargs = {"_commit_hash": "fake-revision"}

    def __init__(self):
        self.chat_calls = 0
        self.text_calls = 0

    @staticmethod
    def _encode(text, offset):
        return [offset + (ord(char) % 31) for char in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        self.chat_calls += 1
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return self._encode(rendered, 10)

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.text_calls += 1
        return {"input_ids": self._encode(text, 50)}


def make_sources():
    wildchat = []
    ultrachat = []
    fineweb = []
    for index in range(40):
        wildchat.append(
            {
                "conversation_id": f"wild-{index}",
                "language": "English",
                "conversation": [
                    {"role": "user", "content": f"question {index}", "language": "English"},
                    {"role": "assistant", "content": f"answer {index}", "language": "English"},
                ],
            }
        )
        ultrachat.append(
            {
                "prompt_id": f"ultra-{index}",
                "messages": [
                    {"role": "user", "content": f"instruction {index}"},
                    {"role": "assistant", "content": f"response {index}"},
                ],
            }
        )
        fineweb.append({"id": f"fine-{index}", "text": f"educational document {index} " * 4})
    return {"wildchat": wildchat, "ultrachat": ultrachat, "fineweb_edu": fineweb}


def test_allocate_source_blocks_uses_expected_50_40_10_split():
    assert allocate_source_blocks(128) == {
        "wildchat": 64,
        "ultrachat": 51,
        "fineweb_edu": 13,
    }


def test_wildchat_english_filter_checks_row_and_messages():
    english = make_sources()["wildchat"][0]
    assert is_english_wildchat(english)

    non_english_row = dict(english, language="Chinese")
    assert not is_english_wildchat(non_english_row)

    mixed_messages = dict(english)
    mixed_messages["conversation"] = [dict(message) for message in english["conversation"]]
    mixed_messages["conversation"][1]["language"] = "Spanish"
    assert not is_english_wildchat(mixed_messages)


def test_build_is_seeded_and_applies_templates_only_to_chat_sources():
    tokenizer_a = FakeTokenizer()
    ids_a, metadata_a = build_mixed_chat_en_calibration(
        tokenizer=tokenizer_a,
        sources=make_sources(),
        nsamples=10,
        seqlen=32,
        seed=7,
        dataset_revisions={"wildchat": "a", "ultrachat": "b", "fineweb_edu": "c"},
    )
    tokenizer_b = FakeTokenizer()
    ids_b, metadata_b = build_mixed_chat_en_calibration(
        tokenizer=tokenizer_b,
        sources=make_sources(),
        nsamples=10,
        seqlen=32,
        seed=7,
        dataset_revisions={"wildchat": "a", "ultrachat": "b", "fineweb_edu": "c"},
    )
    tokenizer_c = FakeTokenizer()
    ids_c, _ = build_mixed_chat_en_calibration(
        tokenizer=tokenizer_c,
        sources=make_sources(),
        nsamples=10,
        seqlen=32,
        seed=8,
    )

    assert ids_a.shape == (10, 32)
    assert torch.equal(ids_a, ids_b)
    assert metadata_a["input_ids_sha256"] == metadata_b["input_ids_sha256"]
    assert not torch.equal(ids_a, ids_c)
    assert metadata_a["source_blocks"] == {"wildchat": 5, "ultrachat": 4, "fineweb_edu": 1}
    assert tokenizer_a.chat_calls > 0
    assert tokenizer_a.text_calls > 0
    assert tokenizer_a.chat_calls == (
        metadata_a["source_stats"]["wildchat"]["selected"]
        + metadata_a["source_stats"]["ultrachat"]["selected"]
    )
    assert tokenizer_a.text_calls == metadata_a["source_stats"]["fineweb_edu"]["selected"]


def test_saved_cache_is_validated_and_loaded_by_unified_loader(tmp_path):
    input_ids, metadata = build_mixed_chat_en_calibration(
        tokenizer=FakeTokenizer(),
        sources=make_sources(),
        nsamples=10,
        seqlen=32,
        seed=3,
    )
    output_path = tmp_path / "mixed_chat_en-N10-L32-Seed3.pt"
    save_prepared_calibration(input_ids, metadata, output_path)

    loaded, loaded_metadata = load_prepared_calibration(
        output_path,
        expected_nsamples=10,
        expected_seqlen=32,
        expected_seed=3,
        tokenizer=FakeTokenizer(),
    )
    assert torch.equal(loaded, input_ids)
    assert loaded_metadata["input_ids_sha256"] == metadata["input_ids_sha256"]

    args = SimpleNamespace(
        calib_dataset="mixed_chat_en",
        calib_data_path=str(output_path),
        nsamples=10,
        seqlen=32,
        batch_size=2,
        seed=3,
    )
    loader = get_calib_loader(FakeTokenizer(), args)
    assert len(loader) == 5
    assert all(batch[0].shape == (2, 32) and batch[1] is None for batch in loader)
    assert torch.equal(torch.cat([batch[0] for batch in loader]), input_ids)

    with pytest.raises(ValueError, match="expected 11"):
        load_prepared_calibration(output_path, expected_nsamples=11)
