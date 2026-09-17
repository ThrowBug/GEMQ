"""Small CPU checks for the post-GPTQ pruning-aware trainer."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from torch import nn
import torch.nn.functional as F

import gemq.router_finetune.router_norm_reconstruction as reconstruction


class ToyMoe(nn.Module):
    def __init__(self, width=4, experts=3):
        super().__init__()
        self.gate = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(experts))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        scores = self.gate(x).softmax(-1)
        indices = scores.argmax(-1)
        selected = F.one_hot(indices, scores.shape[-1]).to(scores.dtype)
        values = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return ((scores * selected).unsqueeze(-1) * values).sum(-2), scores


class ToyLayer(nn.Module):
    def __init__(self, experts=3):
        super().__init__()
        self.attention = nn.Linear(4, 4, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = ToyMoe(experts=experts)

    def forward(self, x, **_kwargs):
        u = x + self.attention(x)
        return (u + self.mlp(self.post_attention_layernorm(u))[0],)


class ToyModel(nn.Module):
    def __init__(self, experts=3):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(experts=experts)])


@pytest.fixture(autouse=True)
def toy_accessors(monkeypatch):
    monkeypatch.setattr(reconstruction, "get_blocks", lambda model, _name: model.layers)
    monkeypatch.setattr(reconstruction, "get_moe_block", lambda layer, _name: layer.mlp)
    monkeypatch.setattr(
        reconstruction, "get_router_module", lambda layer, _name: ("mlp.gate", layer.mlp.gate)
    )


def test_step_zero_and_delta_only_routing_invariance():
    torch.manual_seed(3)
    layer = ToyLayer()
    u = torch.randn(2, 3, 4)
    baseline_norm = layer.post_attention_layernorm(u)
    baseline_logits = layer.mlp.gate(baseline_norm)
    baseline_output = layer.mlp(baseline_norm)[0]
    parametrization = reconstruction.DecoupledRouterNorm(layer, "toy")
    with parametrization:
        norm = layer.post_attention_layernorm(u)
        assert torch.allclose(layer.mlp.gate(norm), baseline_logits, atol=1e-6)
        assert torch.allclose(layer.mlp(norm)[0], baseline_output, atol=1e-6)
        with torch.no_grad():
            parametrization.delta.copy_(torch.tensor([0.1, -0.2, 0.05, 0.15]))
        norm = layer.post_attention_layernorm(u)
        assert torch.allclose(layer.mlp.gate(norm), baseline_logits, atol=1e-6)
        hooked = layer.mlp(norm)[0]
    parametrization.fold()
    folded = layer.mlp(layer.post_attention_layernorm(u))[0]
    assert torch.allclose(folded, hooked, atol=1e-6)


def test_delta_has_gradient_and_experts_stay_frozen():
    torch.manual_seed(9)
    layer = ToyLayer()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    u = torch.randn(2, 3, 4)
    parametrization = reconstruction.DecoupledRouterNorm(layer, "toy")
    with parametrization:
        result = layer.mlp(layer.post_attention_layernorm(u))[0]
        result.square().sum().backward()
    assert parametrization.delta.grad is not None
    assert parametrization.delta.grad.abs().sum() > 0
    assert all(expert.weight.grad is None for expert in layer.mlp.experts)


def test_capture_stops_before_expert_execution():
    layer = ToyLayer()
    inputs = torch.randn(3, 2, 4)
    u = reconstruction.collect_pre_moe_inputs(layer, inputs, {}, "cpu")
    assert u.shape == inputs.shape
    assert layer.mlp.calls == 0
    expected = inputs + layer.attention(inputs)
    assert torch.allclose(u, expected)


def test_teacher_targets_and_layer_propagation_preserve_batch_axis():
    layer = ToyLayer()
    inputs = torch.randn(3, 2, 4)
    targets = reconstruction.collect_teacher_moe_targets(layer, inputs, "toy", "cpu")
    expected_targets = layer.mlp(layer.post_attention_layernorm(inputs))[0]
    assert targets.shape == inputs.shape
    assert torch.allclose(targets, expected_targets)
    outputs = reconstruction.propagate_layer(layer, inputs, {}, "cpu")
    assert outputs.shape == inputs.shape
    assert torch.allclose(outputs, layer(inputs)[0])


@pytest.mark.parametrize("stage", ["norm_only", "norm_then_router", "decoupled_joint"])
def test_layerwise_trainer_keeps_final_stage_even_when_holdout_worsens(stage, monkeypatch):
    torch.manual_seed(17)
    teacher = ToyModel()
    student = ToyModel(experts=2)
    train = torch.randn(4, 2, 4)
    holdout = torch.randn(2, 2, 4)
    original_norm = student.layers[0].post_attention_layernorm.weight.detach().clone()
    original_gate = student.layers[0].mlp.gate.weight.detach().clone()
    original_expert = student.layers[0].mlp.experts[0].weight.detach().clone()
    config = reconstruction.ReconstructionConfig(stage=stage, epochs=1, batch_size=1)
    ran_stages = []

    def fake_train_stage(_layer, parametrization, _inputs, _targets, _name, _config,
                         stage_name, _layer_idx, _device):
        ran_stages.append(stage_name)
        with torch.no_grad():
            if stage_name == "norm":
                parametrization.delta.fill_(0.1)
            elif stage_name == "router":
                parametrization.v.fill_(0.2)
            else:
                parametrization.delta.fill_(0.3)
                parametrization.v.fill_(0.4)

    monkeypatch.setattr(reconstruction, "_train_stage", fake_train_stage)
    error_calls = [0]

    def worsening_holdout(*_args):
        error_calls[0] += 1
        return float(error_calls[0])

    monkeypatch.setattr(reconstruction, "local_error", worsening_holdout)
    result = reconstruction.finetune_router_norm_reconstruction(
        teacher, student, train, holdout, {}, "toy", config, device="cpu"
    )
    assert result is None
    expected_stages = {
        "norm_only": ["norm"],
        "norm_then_router": ["norm", "router"],
        "decoupled_joint": ["norm", "router", "joint"],
    }[stage]
    assert ran_stages == expected_stages
    delta = 0.3 if stage == "decoupled_joint" else 0.1
    expected_v = original_gate if stage == "norm_only" else torch.full_like(
        original_gate, 0.4 if stage == "decoupled_joint" else 0.2
    )
    assert torch.allclose(
        student.layers[0].post_attention_layernorm.weight,
        original_norm * torch.exp(torch.tensor(delta)),
    )
    assert torch.allclose(
        student.layers[0].mlp.gate.weight,
        expected_v * torch.exp(torch.tensor(-delta)),
    )
    assert torch.equal(student.layers[0].mlp.experts[0].weight, original_expert)


def test_output_is_new_and_keeps_pruning_metadata(tmp_path):
    from gemq.finetune_router_norm import save_without_overwriting

    class FakeArtifact:
        def save_pretrained(self, directory):
            (directory / "artifact.txt").write_text("saved", encoding="utf-8")

    output = tmp_path / "new-model"
    pruning = {"original_num_experts": 3, "num_experts": 2}
    metadata = {"identity_sha256": "abc", "pruning": pruning}
    config = reconstruction.ReconstructionConfig()
    save_without_overwriting(FakeArtifact(), FakeArtifact(), output, metadata, config, {})
    assert output.joinpath("expert_pruning_map.json").is_file()
    with pytest.raises(FileExistsError):
        save_without_overwriting(FakeArtifact(), FakeArtifact(), output, metadata, config, {})


def test_max_prune_ratio_one_disables_extra_ratio_cap(monkeypatch):
    import gemq.finetune_router_norm as runner

    monkeypatch.setattr(runner, "get_blocks", lambda model, _name: model.layers)
    monkeypatch.setattr(runner, "get_moe_block", lambda layer, _name: layer.mlp)
    monkeypatch.setattr(
        runner, "get_router_module", lambda layer, _name: ("mlp.gate", layer.mlp.gate)
    )
    pruning = {
        "original_num_experts": 3,
        "num_experts": 2,
        "pruned_experts_per_layer": 1,
        "layers": {"0": {"kept_old_ids": [0, 1]}},
    }
    teacher = ToyModel(experts=3)
    student = ToyModel(experts=2)
    runner.validate_pruning_shapes(teacher, student, "toy", pruning, 1.0)
    with pytest.raises(ValueError, match="exceeds"):
        runner.validate_pruning_shapes(teacher, student, "toy", pruning, 0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        runner.validate_pruning_shapes(teacher, student, "toy", pruning, 1.1)
