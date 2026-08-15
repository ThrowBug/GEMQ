import torch

from gemq import benchmark_generate


def _scripted_decoder(monkeypatch, token_ids):
    calls = {"count": 0}

    def fake_decode(*args, **kwargs):
        token_id = token_ids[calls["count"]]
        calls["count"] += 1
        return torch.tensor([token_id], dtype=torch.long)

    monkeypatch.setattr(benchmark_generate, "decode_one_token", fake_decode)
    return calls


def test_zero_temperature_uses_direct_greedy_decoding(monkeypatch):
    def fail_probability_path(*args, **kwargs):
        raise AssertionError("temperature=0 must not scale logits or run softmax")

    monkeypatch.setattr(benchmark_generate, "logits_to_probs", fail_probability_path)
    logits = torch.tensor([[[1.0, 7.0, 3.0]]], dtype=torch.float16)

    token, probabilities = benchmark_generate.sample(logits, temperature=0, top_k=1)

    assert token.tolist() == [1]
    assert probabilities is None


def test_decode_stops_at_first_eos_with_chunked_checks(monkeypatch):
    calls = _scripted_decoder(monkeypatch, [11, 12, 2, 99, 100])

    tokens = benchmark_generate.decode_n_tokens(
        model=None,
        cur_token=torch.tensor([[1]], dtype=torch.long),
        kv_cache=None,
        input_pos=torch.tensor([0]),
        num_new_tokens=5,
        eos_token_ids={2},
        eos_check_interval=4,
    )

    assert [int(token.item()) for token in tokens] == [11, 12, 2]
    # EOS was the third token, but chunked checking deliberately computed one extra.
    assert calls["count"] == 4


def test_decode_without_eos_preserves_requested_length(monkeypatch):
    calls = _scripted_decoder(monkeypatch, [11, 12, 13])

    tokens = benchmark_generate.decode_n_tokens(
        model=None,
        cur_token=torch.tensor([[1]], dtype=torch.long),
        kv_cache=None,
        input_pos=torch.tensor([0]),
        num_new_tokens=3,
        eos_token_ids={2},
        eos_check_interval=2,
    )

    assert [int(token.item()) for token in tokens] == [11, 12, 13]
    assert calls["count"] == 3


def test_generate_stops_when_prefill_returns_eos(monkeypatch):
    monkeypatch.setattr(
        benchmark_generate,
        "prefill",
        lambda *args, **kwargs: torch.tensor([2], dtype=torch.long),
    )
    monkeypatch.setattr(benchmark_generate, "device_sync", lambda *args, **kwargs: None)

    def fail_decode(*args, **kwargs):
        raise AssertionError("decode should not run after prefill returns EOS")

    monkeypatch.setattr(benchmark_generate, "decode_n_tokens", fail_decode)

    output, stats = benchmark_generate.generate(
        model=None,
        prompt=torch.tensor([7, 8], dtype=torch.long),
        max_new_tokens=32,
        kv_cache=None,
        eos_token_ids={2},
    )

    assert output.tolist() == [7, 8, 2]
    assert stats["generated_tokens"] == 1
    assert stats["stopped_on_eos"] is True
    assert stats["decode_latency"] == 0.0


def test_generate_supports_one_token_without_eos(monkeypatch):
    monkeypatch.setattr(
        benchmark_generate,
        "prefill",
        lambda *args, **kwargs: torch.tensor([9], dtype=torch.long),
    )
    monkeypatch.setattr(benchmark_generate, "device_sync", lambda *args, **kwargs: None)

    output, stats = benchmark_generate.generate(
        model=None,
        prompt=torch.tensor([7, 8], dtype=torch.long),
        max_new_tokens=1,
        kv_cache=None,
        eos_token_ids={2},
    )

    assert output.tolist() == [7, 8, 9]
    assert stats["generated_tokens"] == 1
    assert stats["stopped_on_eos"] is False
