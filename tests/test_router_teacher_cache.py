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


def _config(needs_router=True, needs_output=False, needs_statistics=False):
    return SimpleNamespace(
        needs_router_targets=needs_router,
        needs_output_targets=needs_output,
        needs_router_statistics=needs_statistics,
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


def test_old_cache_remains_readable_but_cannot_satisfy_router_statistics(tmp_path):
    input_ids = torch.tensor([[1, 2, 3]])
    identity = {"cache_version": 1, "input_ids_sha256": "test"}
    targets = TeacherTargets(
        input_ids=input_ids,
        attention_mask=None,
        router_logits=None,
        final_hidden_states=torch.randn(1, 3, 5),
        metadata={},
    )
    _save_targets(tmp_path, identity, targets)
    config = _config(needs_router=False, needs_output=True, needs_statistics=True)

    assert _load_cached_targets(tmp_path, identity, input_ids, None, config) is None
    partial = _load_cached_targets(
        tmp_path, identity, input_ids, None, config, require_all=False
    )
    assert partial.final_hidden_states is not None
    assert partial.router_stats is None


def test_router_statistics_sidecar_round_trips(tmp_path):
    input_ids = torch.tensor([[1, 2]])
    identity = {"cache_version": 1, "input_ids_sha256": "stats"}
    router_stats = {
        "version": 1,
        "similarity": "pearson",
        "router_correlation": torch.eye(3).unsqueeze(0),
        "router_mean": torch.full((1, 3), 1.0 / 3.0),
        "router_std": torch.ones(1, 3),
        "valid_variance": torch.ones(1, 3, dtype=torch.bool),
        "token_count": torch.tensor([2]),
    }
    targets = TeacherTargets(
        input_ids=input_ids,
        attention_mask=None,
        router_logits=None,
        final_hidden_states=torch.randn(1, 2, 4),
        metadata={},
        router_stats=router_stats,
    )
    _save_targets(tmp_path, identity, targets)

    loaded = _load_cached_targets(
        tmp_path,
        identity,
        input_ids,
        None,
        _config(needs_router=False, needs_output=True, needs_statistics=True),
    )
    assert torch.equal(
        loaded.router_stats["router_correlation"],
        router_stats["router_correlation"],
    )
