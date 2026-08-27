from types import SimpleNamespace

import pytest
import torch

from gemq.utils.gptq_checkpoint import (
    METADATA_FILENAME,
    PRUNING_FILENAME,
    SUCCESS_FILENAME,
    build_gptq_checkpoint_identity,
    build_gptq_checkpoint_metadata,
    load_gptq_checkpoint_metadata,
    save_gptq_checkpoint,
)


def _args(**overrides):
    values = {
        "model": "org/example-model",
        "model_name": "org/example-model",
        "model_dtype": "bfloat16",
        "attn_impl": "eager",
        "calib_dataset": "mixed_chat_en",
        "seed": 0,
        "mixed": False,
        "bit_cfg": "",
        "quantizer": "gptq",
        "reproduce_mcmoe": False,
        "attn_wbits": 4,
        "gate_wbits": 16,
        "dense_wbits": 4,
        "expert_wbits": 2,
        "groupsize": 128,
        "blocksize": 128,
        "percdamp": 0.01,
        "mse": True,
        "actorder": False,
        "static_groups": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeTokenizer:
    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.linear.register_buffer("quant_scales", torch.ones(1))
        self.linear.register_buffer("quant_zeros", torch.zeros(1))
        self.linear.register_buffer("quant_nbits", torch.tensor(2))
        self.linear.register_buffer("quant_groupsize", torch.tensor(128))

    def save_pretrained(self, path, state_dict):
        torch.save(state_dict, path / "pytorch_model.bin")
        (path / "config.json").write_text("{}", encoding="utf-8")


def test_checkpoint_save_is_complete_non_mutating_and_non_overwriting(tmp_path):
    input_ids = torch.arange(12).reshape(2, 6)
    identity = build_gptq_checkpoint_identity(_args(), input_ids, None)
    model = _FakeModel()
    pruning_metadata = {"original_num_experts": 4, "num_experts": 3}
    metadata = build_gptq_checkpoint_metadata(identity, model, pruning_metadata)
    checkpoint = tmp_path / "checkpoint"

    save_gptq_checkpoint(model, _FakeTokenizer(), checkpoint, metadata)

    assert (checkpoint / SUCCESS_FILENAME).is_file()
    assert (checkpoint / METADATA_FILENAME).is_file()
    assert (checkpoint / PRUNING_FILENAME).is_file()
    saved = torch.load(checkpoint / "pytorch_model.bin", weights_only=True)
    assert set(saved) == {"linear.weight"}
    assert model.linear.quant_scales is not None
    assert model.linear.quant_zeros is not None

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        save_gptq_checkpoint(model, _FakeTokenizer(), checkpoint, metadata)


def test_checkpoint_load_validates_identity_and_completion(tmp_path):
    input_ids = torch.arange(12).reshape(2, 6)
    identity = build_gptq_checkpoint_identity(_args(), input_ids, None)
    model = _FakeModel()
    metadata = build_gptq_checkpoint_metadata(identity, model)
    checkpoint = tmp_path / "checkpoint"
    save_gptq_checkpoint(model, _FakeTokenizer(), checkpoint, metadata)

    loaded = load_gptq_checkpoint_metadata(checkpoint, identity)
    assert loaded["identity"] == identity

    different = build_gptq_checkpoint_identity(_args(seed=1), input_ids, None)
    with pytest.raises(ValueError, match="does not match"):
        load_gptq_checkpoint_metadata(checkpoint, different)

    (checkpoint / SUCCESS_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        load_gptq_checkpoint_metadata(checkpoint, identity)
