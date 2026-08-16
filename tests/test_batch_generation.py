from types import SimpleNamespace

import torch

from gemq.batched_generate import generate_batch


class FakeConfig:
    num_hidden_layers = 1
    sliding_window = None

    def get_text_config(self, decoder=True):
        return self


class ScriptedBatchModel:
    def __init__(self, scripted_tokens, vocab_size=128):
        self.config = FakeConfig()
        self.scripted_tokens = scripted_tokens
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": kwargs["attention_mask"].detach().clone(),
                "position_ids": kwargs["position_ids"].detach().clone(),
                "cache_position": kwargs["cache_position"].detach().clone(),
            }
        )
        scripted = self.scripted_tokens[len(self.calls) - 1]
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
            -1000.0,
            device=input_ids.device,
        )
        for row, token_id in enumerate(scripted):
            logits[row, -1, token_id] = 1000.0
        return SimpleNamespace(logits=logits)


def test_generate_batch_left_pads_and_stops_rows_independently():
    model = ScriptedBatchModel(
        [
            [5, 6],   # first tokens from batched prefill
            [2, 7],   # row 0 reaches EOS
            [99, 2],  # row 0 is inactive; row 1 reaches EOS
        ]
    )

    result = generate_batch(
        model,
        prompts=[torch.tensor([10, 11]), torch.tensor([20])],
        max_new_tokens=[4, 3],
        pad_token_id=0,
        eos_token_ids={2},
        eos_check_interval=1,
        temperature=0,
    )

    assert result.token_ids == [[5, 2], [6, 7, 2]]
    assert result.generated_tokens == [2, 3]
    assert result.stopped_on_eos == [True, True]

    prefill = model.calls[0]
    assert prefill["input_ids"].tolist() == [[10, 11], [0, 20]]
    assert prefill["attention_mask"].tolist() == [[1, 1], [0, 1]]
    assert prefill["position_ids"].tolist() == [[0, 1], [0, 0]]
    assert prefill["cache_position"].tolist() == [0, 1]

    first_decode = model.calls[1]
    assert first_decode["input_ids"].tolist() == [[5], [6]]
    assert first_decode["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert first_decode["position_ids"].tolist() == [[2], [1]]
    assert first_decode["cache_position"].tolist() == [2]

    second_decode = model.calls[2]
    assert second_decode["input_ids"].tolist() == [[0], [7]]
    assert second_decode["attention_mask"].tolist() == [
        [1, 1, 1, 0],
        [0, 1, 1, 1],
    ]
    assert second_decode["position_ids"].tolist() == [[0], [2]]
    assert second_decode["cache_position"].tolist() == [3]


def test_generate_batch_respects_per_row_max_tokens_without_eos():
    model = ScriptedBatchModel([[5, 6], [7, 8], [9, 10]])

    result = generate_batch(
        model,
        prompts=[torch.tensor([1]), torch.tensor([2])],
        max_new_tokens=[1, 3],
        pad_token_id=0,
        eos_token_ids={99},
        eos_check_interval=1,
        temperature=0,
    )

    assert result.token_ids == [[5], [6, 8, 10]]
    assert result.generated_tokens == [1, 3]
    assert result.stopped_on_eos == [False, False]
