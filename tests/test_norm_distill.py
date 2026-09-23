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


class _ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values):
        normalized = values.float() * values.float().square().mean(
            dim=-1, keepdim=True
        ).add(1e-6).rsqrt()
        return (1.0 + self.weight.float()) * normalized


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


class _Qwen35Layer(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.input_layernorm = _ZeroCenteredRMSNorm(width)
        self.post_attention_layernorm = _ZeroCenteredRMSNorm(width)
        self.attention = nn.Linear(width, width, bias=False)
        self.router = nn.Linear(width, 3, bias=False)
        self.mlp = nn.Module()
        self.mlp.shared_expert_gate = nn.Linear(width, 1, bias=False)
        self.expert = nn.Linear(width, width, bias=False)

    def run(self, values):
        attention_input = self.input_layernorm(values)
        attention_output = self.attention(attention_input)
        normalized = self.post_attention_layernorm(values + attention_output)
        flattened = normalized.reshape(-1, normalized.shape[-1])
        return (
            normalized,
            self.router(flattened),
            self.mlp.shared_expert_gate(flattened).sigmoid(),
            self.expert(flattened),
        )


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


def test_scale_summary_prints_key_distribution_and_changed_ratio(capsys):
    norm_distill._print_scale_summary(
        "input",
        [torch.tensor([0.9, 1.0, 1.1])],
        [torch.tensor([1.0, 1.0, 1.125])],
    )
    output = capsys.readouterr().out

    assert "[norm-scale input]" in output
    assert "learned p05=" in output
    assert "p50=1.000000" in output
    assert "realized p05=" in output
    assert "p95=1.112500" in output
    assert "changed=33.33%" in output


def test_zero_centered_norm_fold_uses_effective_gain():
    norm = _ZeroCenteredRMSNorm(4)
    controller = norm_distill.ZeroCenteredNormScale(norm)
    with torch.no_grad():
        controller.delta.copy_(torch.tensor([0.2, -0.1, 0.3, -0.25]))

    realized = controller.fold()
    expected = controller.delta.detach().exp()

    torch.testing.assert_close(norm.weight.float() + 1.0, expected)
    torch.testing.assert_close(realized, expected)
    assert not torch.equal(norm.weight, torch.zeros_like(norm.weight))


def test_qwen35_compensation_preserves_both_gates_and_folds_exactly():
    torch.manual_seed(4)
    layer = _Qwen35Layer()
    values = torch.randn(2, 3, 4)
    baseline = layer.run(values)
    controller = norm_distill.Qwen35RouterCompensatedNorm(layer, "unused")
    with torch.no_grad():
        controller.delta.copy_(torch.tensor([0.2, -0.1, 0.3, -0.25]))

    controller.install()
    hooked = layer.run(values)
    controller.remove()

    assert not torch.equal(hooked[0], baseline[0])
    torch.testing.assert_close(hooked[1], baseline[1], rtol=0, atol=0)
    torch.testing.assert_close(hooked[2], baseline[2], rtol=0, atol=0)
    assert not torch.equal(hooked[3], baseline[3])

    controller.fold()
    folded = layer.run(values)
    for actual, expected in zip(folded, hooked):
        torch.testing.assert_close(actual, expected)


def test_qwen35_dual_zero_centered_norm_scales_fold_together():
    torch.manual_seed(5)
    layer = _Qwen35Layer()
    values = torch.randn(2, 3, 4)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    input_controller = norm_distill.ZeroCenteredNormScale(layer.input_layernorm)
    post_controller = norm_distill.Qwen35RouterCompensatedNorm(layer, "unused")
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
