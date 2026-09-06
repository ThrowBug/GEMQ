from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from gemq.router_finetune.confidence import estimate_cakld_gamma, resolve_cakld_gamma


class RecordingHead(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 3, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3))
        self.chunk_sizes = []

    def forward(self, hidden):
        assert not torch.is_grad_enabled()
        self.chunk_sizes.append(hidden.shape[0])
        return super().forward(hidden)


def targets(mask=None):
    return SimpleNamespace(
        # IDs deliberately need not match argmax: this is not target-token confidence.
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        attention_mask=mask,
        final_hidden_states=torch.tensor([
            [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [100., 0., 0.]],
            [[0., 1., 0.], [0., 0., 1.], [2., 0., 0.], [100., 0., 0.]],
        ]),
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 256])
@pytest.mark.parametrize("masked", [False, True])
def test_streaming_gamma_matches_valid_token_mean_and_preserves_state(chunk_size, masked):
    mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]]) if masked else None
    cached = targets(mask)
    head = RecordingHead()
    original = head.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    result = estimate_cakld_gamma(head, cached, 2, token_chunk_size=chunk_size)
    probs = cached.final_hidden_states[:, :-1].softmax(-1).amax(-1)
    valid = torch.ones_like(probs, dtype=torch.bool) if mask is None else mask[:, :-1].bool() & mask[:, 1:].bool()
    assert result["resolved_gamma"] == pytest.approx(probs[valid].mean().item(), abs=1e-7)
    assert result["confidence_valid_tokens"] == (4 if masked else 6)
    assert result["confidence_samples"] == 2
    assert result["gamma_source"] == "teacher_cache"
    assert max(head.chunk_sizes) <= chunk_size
    assert torch.equal(head.weight, original)
    assert head.weight.grad is None
    assert torch.equal(torch.get_rng_state(), rng)


def test_gamma_respects_sample_limit():
    cached = targets()
    result = estimate_cakld_gamma(RecordingHead(), cached, 1)
    expected = cached.final_hidden_states[0, :-1].softmax(-1).amax(-1).mean().item()
    assert result["resolved_gamma"] == pytest.approx(expected, abs=1e-7)
    assert result["confidence_valid_tokens"] == 3


def test_manual_gamma_skips_cache_and_head_access():
    result = resolve_cakld_gamma(None, None, 0, "0.25")
    assert result["resolved_gamma"] == 0.25
    assert result["gamma_source"] == "manual"
    assert result["confidence_valid_tokens"] == 0


def test_gamma_rejects_empty_or_nonfinite_confidence():
    with pytest.raises(ValueError, match="No valid"):
        estimate_cakld_gamma(RecordingHead(), targets(torch.zeros(2, 4)), 2)
    cached = targets()
    cached.final_hidden_states[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="Non-finite"):
        estimate_cakld_gamma(RecordingHead(), cached, 2)
