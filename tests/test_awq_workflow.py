from types import SimpleNamespace
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from gemq.awq_quantize import (
    quantize_weights_awq,
    validate_awq_allocation_sidecar,
)
from gemq.expert_costs import compute_qwen3_awq_expert_costs

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class _TinyExpert(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, value):
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))


class _TinyMoe(nn.Module):
    def __init__(self, hidden_size=4, num_experts=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_TinyExpert(hidden_size) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = 1
        self.norm_topk_prob = True

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self.gate(flat)
        routing = F.softmax(logits.float(), dim=-1)
        weights, selected = torch.topk(routing, self.top_k, dim=-1)
        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            token_ids, topk_ids = torch.where(selected == expert_idx)
            if token_ids.numel():
                contribution = (
                    expert(flat[token_ids]) * weights[token_ids, topk_ids, None]
                )
                output.index_add_(0, token_ids, contribution.to(output.dtype))
        return output.reshape_as(hidden_states), logits


class _TinyAttention(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        del kwargs
        mixed = self.q_proj(hidden_states) + self.k_proj(hidden_states)
        mixed = mixed + self.v_proj(hidden_states)
        return self.o_proj(mixed), None


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _TinyAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _TinyMoe()
        self.attention_type = "full_attention"

    def forward(self, hidden_states, **kwargs):
        attention, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = hidden_states + attention
        moe_output, _ = self.mlp(self.post_attention_layernorm(hidden_states))
        return (hidden_states + moe_output,)


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 4)
        self.model.layers = nn.ModuleList([_TinyLayer(), _TinyLayer()])
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids):
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def test_mixed_bit_awq_workflow_runs_sequentially_on_tiny_qwen_model():
    torch.manual_seed(3)
    model = _TinyLM()
    original = model.model.layers[0].mlp.experts[0].gate_proj.weight.detach().clone()
    dataloader = [
        (torch.tensor([[1, 2, 3]]), None),
        (torch.tensor([[4, 5, 6]]), None),
    ]
    bit_config = {
        0: {0: 1, 1: 3},
        1: {0: 2, 1: 3},
    }
    args = SimpleNamespace(
        model_name=MODEL_NAME,
        groupsize=2,
        awq_scale_n_grid=2,
        awq_clip_n_grid=2,
        awq_clip_max_shrink=0.5,
        awq_clip_n_sample_token=8,
        awq_search_batch_size=1,
        expert_batch_size=8,
        attn_wbits=2,
        dense_wbits=2,
        calib_dataset="c4",
        nsamples=2,
        seqlen=3,
        seed=0,
        awq_device="cpu",
    )

    metadata = quantize_weights_awq(model, dataloader, args, bit_config)

    assert metadata["quantizer"] == "gemq-awq"
    assert len(metadata["layers"]) == 2
    assert metadata["layers"][0]["policy"]["expert_bits"] == [1, 3]
    assert not torch.equal(
        model.model.layers[0].mlp.experts[0].gate_proj.weight,
        original,
    )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_uniform_awq_workflow_does_not_require_bit_config():
    torch.manual_seed(5)
    model = _TinyLM()
    dataloader = [
        (torch.tensor([[1, 2, 3]]), None),
        (torch.tensor([[4, 5, 6]]), None),
    ]
    args = SimpleNamespace(
        model_name=MODEL_NAME,
        groupsize=2,
        awq_scale_n_grid=2,
        awq_clip_n_grid=2,
        awq_clip_max_shrink=0.5,
        awq_clip_n_sample_token=8,
        awq_search_batch_size=1,
        expert_batch_size=8,
        expert_wbits=3,
        attn_wbits=2,
        dense_wbits=2,
        calib_dataset="c4",
        nsamples=2,
        seqlen=3,
        seed=0,
        awq_device="cpu",
    )

    metadata = quantize_weights_awq(model, dataloader, args, None)

    assert metadata["expert_precision"] == {
        "mode": "uniform",
        "uniform_expert_bits": 3,
    }
    assert all(
        layer["policy"]["expert_bits"] == [3, 3]
        for layer in metadata["layers"]
    )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_awq_expert_costs_cover_all_candidates_and_keep_w2_context():
    torch.manual_seed(4)
    model = _TinyLM()
    dataloader = [
        (torch.tensor([[1, 2, 3]]), None),
        (torch.tensor([[4, 5, 6]]), None),
    ]

    costs, counts, metadata = compute_qwen3_awq_expert_costs(
        model,
        dataloader,
        MODEL_NAME,
        candidate_bits=[0, 1, 2, 3],
        average_bits=2,
        expert_batch_size=8,
        device="cpu",
        groupsize=2,
        scale_n_grid=2,
        clip_n_grid=2,
        clip_max_shrink=0.5,
        clip_n_sample_token=8,
        search_batch_size=1,
        attention_bits=2,
        dense_bits=2,
    )

    assert costs.shape == (2, 2, 4)
    assert counts.shape == (2, 2)
    observed = counts > 0
    assert torch.isfinite(costs[observed]).all()
    assert (costs[observed] >= 0).all()
    assert metadata["search_options"]["groupsize"] == 2
    assert len(metadata["layers"]) == 2
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_awq_allocation_sidecar_locks_calibration_and_search_config(tmp_path):
    bit_config_path = tmp_path / "allocation.pkl"
    bit_config_path.touch()
    search_options = {
        "groupsize": 128,
        "scale_n_grid": 20,
        "clip_n_grid": 20,
        "clip_max_shrink": 0.5,
        "clip_n_sample_token": 512,
        "search_batch_size": 1,
    }
    metadata = {
        "source_quantizer": "awq",
        "candidate_bits": [0, 2, 3],
        "source_candidate_bits": [0, 1, 2, 3],
        "max_prune_ratio": 0.1,
        "average_bit_budget": 2.0,
        "budget_denominator": "original_experts",
        "source_input_ids_sha256": "abc",
        "source_awq_config": {
            "model_name": MODEL_NAME,
            "model_dtype": "bfloat16",
            "groupsize": 128,
            "attn_wbits": 4,
            "dense_wbits": 4,
            "awq_search_options": search_options,
        },
    }
    bit_config_path.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    loaded = validate_awq_allocation_sidecar(
        bit_config_path,
        expected_input_hash="abc",
        expected_model_name=MODEL_NAME,
        expected_model_dtype="bfloat16",
        expected_awq_config={
            "groupsize": 128,
            "attn_wbits": 4,
            "dense_wbits": 4,
            "awq_search_options": search_options,
        },
        expert_bit_config={0: {0: 0, 1: 2, 2: 3}},
    )
    assert loaded == metadata

    try:
        validate_awq_allocation_sidecar(
            bit_config_path,
            expected_input_hash="different",
        )
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("Expected calibration hash mismatch to be rejected")

    try:
        validate_awq_allocation_sidecar(
            bit_config_path,
            expert_bit_config={0: {0: 1}},
        )
    except ValueError as error:
        assert "excluded from its IP candidate" in str(error)
    else:
        raise AssertionError("Expected an excluded 1-bit assignment to be rejected")

    # Sidecars produced before source_candidate_bits was introduced used the full
    # candidate set; they remain valid via the compatibility fallback.
    legacy_metadata = dict(metadata)
    legacy_metadata["candidate_bits"] = [0, 1, 2, 3]
    legacy_metadata.pop("source_candidate_bits")
    bit_config_path.with_suffix(".json").write_text(
        json.dumps(legacy_metadata), encoding="utf-8"
    )
    assert validate_awq_allocation_sidecar(
        bit_config_path,
        expert_bit_config={0: {0: 1, 1: 2, 2: 3}},
    ) == legacy_metadata
