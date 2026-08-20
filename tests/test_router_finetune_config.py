from types import SimpleNamespace

import pytest

from gemq.router_finetune.config import DistillCEConfig, RFT_TRAINERS, RouterFinetuneConfig


def _args(**overrides):
    values = {
        "rft_timing": "after_all_quantization",
        "rft_router_loss": "kd",
        "rft_router_alpha": 0.0,
        "rft_router_loss_weight": 1.0,
        "rft_output_kl_weight": 0.0,
        "rft_epochs": 1,
        "rft_batch_size": 1,
        "rft_lr": 1e-4,
        "rft_wd": 1e-4,
        "rft_teacher_cache_dir": "cache/router_finetune",
        "rft_rebuild_teacher_cache": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.0, 8), (0.25, 22), (0.5, 36), (0.75, 50), (1.0, 64)],
)
def test_effective_top_m(alpha, expected):
    config = RouterFinetuneConfig.from_args(_args(rft_router_alpha=alpha))
    assert config.effective_top_m(num_experts=64, top_k=8) == expected


def test_both_zero_weights_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        RouterFinetuneConfig.from_args(
            _args(rft_router_loss_weight=0.0, rft_output_kl_weight=0.0)
        )


def test_interleaved_output_kl_is_explicitly_rejected():
    with pytest.raises(ValueError, match="not supported"):
        RouterFinetuneConfig.from_args(
            _args(
                rft_timing="after_each_layer_quantization",
                rft_output_kl_weight=0.1,
            )
        )


def test_target_requirements_follow_nonzero_weights():
    output_only = RouterFinetuneConfig.from_args(
        _args(rft_router_loss_weight=0.0, rft_output_kl_weight=1.0)
    )
    assert not output_only.needs_router_targets
    assert output_only.needs_output_targets

    router_only = RouterFinetuneConfig.from_args(_args())
    assert router_only.needs_router_targets
    assert not router_only.needs_output_targets
    assert router_only.is_router_only


def test_distill_ce_has_independent_target_requirements():
    config = DistillCEConfig.from_args(_args())
    assert "distill_ce" in RFT_TRAINERS
    assert not config.needs_router_targets
    assert config.needs_output_targets


def test_distill_ce_does_not_validate_layerwise_loss_arguments():
    config = DistillCEConfig.from_args(
        _args(
            rft_timing="after_each_layer_quantization",
            rft_router_alpha=99.0,
            rft_router_loss_weight=0.0,
            rft_output_kl_weight=0.0,
        )
    )
    assert config.needs_output_targets
