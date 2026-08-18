import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemq.router_finetune.config import RouterFinetuneConfig  # noqa: E402
from gemq.router_finetune.trainer import _train_router_from_inputs  # noqa: E402


def test_local_training_changes_only_the_supplied_router():
    torch.manual_seed(0)
    router = torch.nn.Linear(4, 64, bias=False)
    unrelated_router = torch.nn.Linear(4, 64, bias=False)
    unrelated_before = unrelated_router.weight.detach().clone()
    router_before = router.weight.detach().clone()

    inputs = torch.randn(4, 3, 4)
    teacher_weight = torch.randn_like(router.weight)
    teacher_logits = torch.nn.functional.linear(inputs, teacher_weight)
    config = RouterFinetuneConfig(
        timing="after_all_quantization",
        router_loss="l2",
        router_alpha=1.0,
        router_loss_weight=1.0,
        output_kl_weight=0.0,
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        teacher_cache_dir="cache/router_finetune",
        rebuild_teacher_cache=False,
    )

    _train_router_from_inputs(
        router,
        inputs,
        teacher_logits,
        None,
        "allenai/OLMoE-1B-7B-0125-Instruct",
        config,
        layer_idx=0,
    )

    assert not torch.equal(router.weight, router_before)
    assert torch.equal(unrelated_router.weight, unrelated_before)
    assert not router.weight.requires_grad
