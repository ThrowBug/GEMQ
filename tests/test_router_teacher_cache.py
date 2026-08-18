from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemq.router_finetune.targets import (  # noqa: E402
    TeacherTargets,
    _load_cached_targets,
    _save_targets,
    materialize_calibration_inputs,
)


def _config(needs_router=True, needs_output=False):
    return SimpleNamespace(
        needs_router_targets=needs_router,
        needs_output_targets=needs_output,
    )


def test_materialize_calibration_inputs_preserves_exact_order():
    dataloader = [
        (torch.tensor([[1, 2], [3, 4]]), None),
        (torch.tensor([[5, 6]]), None),
    ]
    input_ids, attention_mask = materialize_calibration_inputs(dataloader)
    assert torch.equal(input_ids, torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert attention_mask is None


def test_cache_rejects_even_one_different_token(tmp_path):
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    identity = {"cache_version": 1, "input_ids_sha256": "test"}
    targets = TeacherTargets(
        input_ids=input_ids,
        attention_mask=None,
        router_logits=(torch.randn(2, 3, 4),),
        final_hidden_states=None,
        metadata={},
    )
    _save_targets(tmp_path, identity, targets)

    changed = input_ids.clone()
    changed[1, 2] = 99
    with pytest.raises(ValueError, match="input_ids differ"):
        _load_cached_targets(tmp_path, identity, changed, None, _config())


def test_cache_requires_requested_target_kinds(tmp_path):
    input_ids = torch.tensor([[1, 2, 3]])
    identity = {"cache_version": 1, "input_ids_sha256": "test"}
    targets = TeacherTargets(
        input_ids=input_ids,
        attention_mask=None,
        router_logits=(torch.randn(1, 3, 4),),
        final_hidden_states=None,
        metadata={},
    )
    _save_targets(tmp_path, identity, targets)
    assert _load_cached_targets(
        tmp_path, identity, input_ids, None, _config(needs_router=True, needs_output=True)
    ) is None
