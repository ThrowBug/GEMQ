from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemq import evaluate_ppl  # noqa: E402


def _patch_runtime(monkeypatch):
    model = SimpleNamespace(seqlen=None)
    tokenizer = object()
    calls = {}

    def load_tokenizer(*args, **kwargs):
        calls["tokenizer"] = (args, kwargs)
        return tokenizer

    def load_model(*args, **kwargs):
        calls["model"] = (args, kwargs)
        return model

    def dispatch(candidate, cuda_diagnostics):
        calls["dispatch"] = (candidate, cuda_diagnostics)
        return candidate

    def evaluate(candidate, candidate_tokenizer, datasets, model_name, offload):
        calls["evaluate"] = (
            candidate,
            candidate_tokenizer,
            datasets,
            model_name,
            offload,
        )

    monkeypatch.setattr(evaluate_ppl.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(evaluate_ppl, "load_causal_lm_checkpoint", load_model)
    monkeypatch.setattr(evaluate_ppl, "dispatch_model_to_all_devices", dispatch)
    monkeypatch.setattr(evaluate_ppl, "evaluate_perplexity", evaluate)
    return model, tokenizer, calls


def test_default_path_dispatches_and_reuses_both_ppl_datasets(monkeypatch):
    model, tokenizer, calls = _patch_runtime(monkeypatch)
    args = evaluate_ppl.parse_args(
        [
            "--model",
            "/tmp/model",
            "--model_name",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "--use_fast",
        ]
    )

    evaluate_ppl.run(args)

    assert model.seqlen == 2048
    assert calls["tokenizer"] == (
        ("/tmp/model",),
        {"use_fast": True, "trust_remote_code": False},
    )
    assert calls["model"] == (
        ("/tmp/model",),
        {
            "model_dtype": "bfloat16",
            "attn_implementation": "eager",
            "trust_remote_code": False,
            "device_map": "cpu",
        },
    )
    assert calls["dispatch"] == (model, False)
    assert calls["evaluate"] == (
        model,
        tokenizer,
        ["wikitext2", "c4"],
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        False,
    )


def test_offload_skips_dispatch_and_forwards_dataset_selection(monkeypatch):
    model, tokenizer, calls = _patch_runtime(monkeypatch)
    args = evaluate_ppl.parse_args(
        [
            "--model",
            "/tmp/model",
            "--model_name",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "--datasets",
            "c4",
            "--seqlen",
            "1024",
            "--offload",
        ]
    )

    evaluate_ppl.run(args)

    assert model.seqlen == 1024
    assert "dispatch" not in calls
    assert calls["evaluate"] == (
        model,
        tokenizer,
        ["c4"],
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        True,
    )


def test_empty_dataset_list_is_rejected():
    with pytest.raises(SystemExit):
        evaluate_ppl.parse_args(
            [
                "--model",
                "/tmp/model",
                "--model_name",
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "--datasets",
                ",",
            ]
        )
