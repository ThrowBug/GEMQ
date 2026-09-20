import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from torch import nn  # noqa: E402

import gemq.router_finetune.norm_distill as norm_distill  # noqa: E402


class _RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values):
        normalized = values.float() * values.float().square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).rsqrt()
        return self.weight * normalized.to(values.dtype)


class _Layer(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.input_layernorm = _RMSNorm(width)
        self.post_attention_layernorm = _RMSNorm(width)
        self.attention = nn.Linear(width, width, bias=False)
        self.router = nn.Linear(width, 3, bias=False)
        self.expert = nn.Linear(width, width, bias=False)

    def run(self, values):
        attention_input = self.input_layernorm(values)
        attention_output = self.attention(attention_input)
        normalized = self.post_attention_layernorm(values + attention_output)
        flattened = normalized.reshape(-1, normalized.shape[-1])
        return normalized, self.router(flattened), self.expert(flattened)


@pytest.fixture(autouse=True)
def _router_accessor(monkeypatch):
    monkeypatch.setattr(
        norm_distill,
        "get_router_module",
        lambda layer, _model_name: ("router", layer.router),
    )


def test_only_norm_scale_gets_gradient_and_same_input_router_is_unchanged():
    torch.manual_seed(0)
    layer = _Layer()
    values = torch.randn(2, 3, 4)
    baseline_norm, baseline_router, _ = layer.run(values)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    controller = norm_distill.RouterCompensatedNorm(layer, "unused")
    with torch.no_grad():
        controller.delta.copy_(torch.tensor([0.2, -0.1, 0.3, -0.25]))
    controller.install()
    scaled_norm, compensated_router, expert_output = layer.run(values)
    expert_output.square().mean().backward()
    controller.remove()

    assert not torch.equal(scaled_norm, baseline_norm)
    torch.testing.assert_close(compensated_router, baseline_router, rtol=0, atol=0)
    assert controller.delta.grad is not None
    assert controller.delta.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in layer.parameters())


def test_fold_matches_training_parameterization():
    torch.manual_seed(1)
    layer = _Layer()
    values = torch.randn(2, 3, 4)
    controller = norm_distill.RouterCompensatedNorm(layer, "unused")
    with torch.no_grad():
        controller.delta.copy_(torch.tensor([0.02, -0.03, 0.04, -0.01]))
    controller.install()
    hooked = layer.run(values)
    controller.remove()
    controller.fold()
    folded = layer.run(values)

    for actual, expected in zip(folded, hooked):
        torch.testing.assert_close(actual, expected)


def test_bfloat16_noop_norm_scale_does_not_change_router():
    layer = _Layer().to(torch.bfloat16)
    old_norm = layer.post_attention_layernorm.weight.detach().clone()
    old_router = layer.router.weight.detach().clone()
    controller = norm_distill.RouterCompensatedNorm(layer, "unused")
    with torch.no_grad():
        controller.delta.fill_(1e-4)
    controller.fold()

    assert torch.equal(layer.post_attention_layernorm.weight, old_norm)
    assert torch.equal(layer.router.weight, old_router)


def test_uncompensated_post_norm_scale_changes_router():
    torch.manual_seed(2)
    layer = _Layer()
    values = torch.randn(2, 3, 4)
    _, baseline_router, _ = layer.run(values)
    controller = norm_distill.NormScale(layer.post_attention_layernorm)
    with torch.no_grad():
        controller.delta.copy_(torch.tensor([0.2, -0.1, 0.3, -0.25]))
    controller.install()
    _, scaled_router, _ = layer.run(values)
    controller.remove()

    assert not torch.equal(scaled_router, baseline_router)


@pytest.mark.parametrize("router_compensated", [False, True])
def test_dual_norm_scales_train_together_and_fold(router_compensated):
    torch.manual_seed(3)
    layer = _Layer()
    values = torch.randn(2, 3, 4)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    input_controller = norm_distill.NormScale(layer.input_layernorm)
    if router_compensated:
        post_controller = norm_distill.RouterCompensatedNorm(layer, "unused")
    else:
        post_controller = norm_distill.NormScale(layer.post_attention_layernorm)
    with torch.no_grad():
        input_controller.delta.copy_(torch.tensor([0.03, -0.02, 0.01, -0.04]))
        post_controller.delta.copy_(torch.tensor([0.02, -0.03, 0.04, -0.01]))

    input_controller.install()
    post_controller.install()
    hooked = layer.run(values)
    hooked[-1].square().mean().backward()
    input_controller.remove()
    post_controller.remove()

    assert input_controller.delta.grad is not None
    assert input_controller.delta.grad.abs().sum().item() > 0
    assert post_controller.delta.grad is not None
    assert post_controller.delta.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in layer.parameters())

    input_controller.fold()
    post_controller.fold()
    folded = layer.run(values)
    for actual, expected in zip(folded, hooked):
        torch.testing.assert_close(actual, expected)
