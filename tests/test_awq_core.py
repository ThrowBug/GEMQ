import torch
import torch.nn as nn

from gemq.quantizers.awq.policy import Qwen3MoeAWQPolicy
from gemq.quantizers.awq.quantizer import fake_quantize_weight
from gemq.quantizers.awq.search import (
    apply_clip,
    apply_linear_pair_scale,
    apply_norm_input_scale,
    search_clip,
    search_scale,
)


class _ScaleOnlyNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, value):
        return value * self.weight


def test_awq_w1_keeps_historical_gemq_binary_definition():
    weight = torch.tensor([[1.0, -3.0, 2.0, -2.0]])
    quantized = fake_quantize_weight(weight, nbits=1, groupsize=2)
    torch.testing.assert_close(
        quantized,
        torch.tensor([[2.0, -2.0, 2.0, -2.0]]),
    )


def test_awq_quantization_supports_independent_group_clipping():
    weight = torch.tensor([[1.0, -4.0, 8.0, -2.0]])
    clip = torch.tensor([[[2.0], [3.0]]])
    quantized = fake_quantize_weight(weight, nbits=2, groupsize=2, clip_max=clip)
    unbounded = fake_quantize_weight(weight, nbits=2, groupsize=2)
    assert quantized.shape == weight.shape
    assert torch.isfinite(quantized).all()
    assert not torch.equal(quantized, unbounded)


def test_awq_norm_and_linear_pair_scale_are_function_preserving():
    torch.manual_seed(0)
    norm = _ScaleOnlyNorm(4)
    first = nn.Linear(4, 6, bias=False)
    following = nn.Linear(6, 3, bias=False)
    value = torch.randn(5, 4)

    reference_first = first(norm(value))
    input_scale = torch.tensor([0.5, 1.0, 2.0, 4.0])
    apply_norm_input_scale(norm, [first], input_scale)
    torch.testing.assert_close(first(norm(value)), reference_first)

    reference_pair = following(reference_first)
    internal_scale = torch.linspace(0.5, 2.0, 6)
    apply_linear_pair_scale(first, following, internal_scale)
    torch.testing.assert_close(following(first(norm(value))), reference_pair)


def test_awq_clip_search_and_application_have_expected_shape():
    torch.manual_seed(1)
    linear = nn.Linear(8, 5, bias=False)
    inputs = torch.randn(32, 8)
    clip = search_clip(
        linear.weight,
        inputs,
        nbits=2,
        groupsize=4,
        n_grid=4,
        max_shrink=0.5,
        n_sample_token=16,
        output_chunk_size=3,
    )
    assert clip.shape == (5, 2, 1)
    original_shape = linear.weight.shape
    apply_clip(linear, clip)
    assert linear.weight.shape == original_shape
    assert torch.isfinite(linear.weight).all()


def test_qwen3_awq_policy_resolves_per_expert_bits():
    policy = Qwen3MoeAWQPolicy(expert_bits=(1, 3), attention_bits=4)
    assert policy.bit_for("self_attn.q_proj") == 4
    assert policy.bit_for("mlp.experts.0.gate_proj") == 1
    assert policy.bit_for("mlp.experts.1.down_proj") == 3
    assert policy.bit_for("mlp.gate") is None


def test_awq_scale_search_restores_original_weights():
    torch.manual_seed(2)
    linear = nn.Linear(4, 3, bias=False)
    inputs = torch.randn(12, 4)
    original = linear.weight.detach().clone()
    scale, error = search_scale(
        linear,
        [linear],
        inputs,
        [2],
        groupsize=2,
        n_grid=4,
        search_batch_size=4,
    )
    assert scale.shape == (4,)
    assert torch.isfinite(scale).all()
    assert error >= 0
    torch.testing.assert_close(linear.weight, original)
