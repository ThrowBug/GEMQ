from types import SimpleNamespace

import pytest

from proquant.router_finetune.config import (
    DistillCEConfig,
    NORM_DISTILL_MODES,
    RFT_TRAINERS,
)


def _args(**overrides):
    values = {
        "rft_epochs": 1,
        "rft_batch_size": 1,
        "rft_lr": 1e-4,
        "rft_wd": 0.0,
        "rft_teacher_cache_dir": "cache/router_finetune",
        "rft_rebuild_teacher_cache": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_release_trainers_and_norm_modes():
    assert RFT_TRAINERS == ("router_ft", "norm_ft", "roco")
    assert NORM_DISTILL_MODES == {
        "norm_ft": {"optimize_input_norm": True, "router_compensated": False},
        "roco": {"optimize_input_norm": True, "router_compensated": True},
    }


def test_norm_teacher_targets_are_output_only():
    config = DistillCEConfig.from_args(_args())
    assert not config.needs_router_targets
    assert config.needs_output_targets


@pytest.mark.parametrize("override", [
    {"rft_epochs": 0},
    {"rft_batch_size": 0},
    {"rft_lr": 0.0},
    {"rft_wd": -1.0},
])
def test_invalid_norm_training_config_is_rejected(override):
    with pytest.raises(ValueError):
        DistillCEConfig.from_args(_args(**override))
