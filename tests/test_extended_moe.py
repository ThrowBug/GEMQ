from types import SimpleNamespace

import pytest
import torch
from transformers import AfmoeConfig, AfmoeForCausalLM, Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from proquant import allocate_bits
from proquant.allocation.ilp_solvers import PoPSolver
from proquant.compute_model_stats import compute_faster_layer_re, compute_layer_grads
from proquant.expert_costs import compute_qwen3_expert_costs
from proquant.pruning import kept_expert_ids_from_pruning_metadata, prune_zero_bit_experts
from proquant.quantize import finetune_routers, quantize_weights_gptq, save_quantized_model
from proquant.router_finetune.config import NORM_DISTILL_MODES
from proquant.router_finetune.norm_distill import RouterCompensatedNorm, finetune_norms_distill_ce
from proquant.router_finetune.targets import TeacherTargets
from proquant.utils.expert_cost_utils import impute_zero_frequency_costs
from proquant.utils.extended_moe import cast_model_preserving_fp32, routing_biases
from proquant.utils.gptq_checkpoint import save_gptq_checkpoint
from proquant.utils.hf_loading import load_causal_lm_checkpoint
from proquant.utils.model_utils import ModelInfo, capture_qwen35_decoder_context, get_router_module


@pytest.fixture(params=["zai-org/GLM-4.7-Flash", "arcee-ai/Trinity-Mini"])
def tiny_model(request):
    torch.manual_seed(4)
    common = dict(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_experts_per_tok=2, use_cache=False,
    )
    if request.param.startswith("zai-org"):
        config = Glm4MoeLiteConfig(
            **common, num_key_value_heads=2, n_routed_experts=6, n_shared_experts=1,
            q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4,
            v_head_dim=4, routed_scaling_factor=1.8,
        )
        model = Glm4MoeLiteForCausalLM(config)
        first_moe = 1
    else:
        config = AfmoeConfig(
            **common, head_dim=8, num_key_value_heads=1, num_experts=6,
            num_shared_experts=1, num_dense_layers=2, sliding_window=3,
            global_attn_every_n_layers=2, route_scale=2.826, mup_enabled=True,
        )
        model = AfmoeForCausalLM(config)
        first_moe = 2
    model.config._attn_implementation = "eager"
    return request.param, model.eval(), first_moe


def test_native_decoder_replay_matches_full_forward(tiny_model):
    name, model, _ = tiny_model
    input_ids = torch.tensor([[2, 3, 5, 7, 11, 13], [3, 2, 8, 6, 5, 7]])
    with torch.no_grad():
        expected = model(input_ids).logits
        hidden, positional, keywords = capture_qwen35_decoder_context(model, [input_ids], name, "cpu")
        x = hidden[0]
        for layer, args, kwargs in zip(model.model.layers, positional, keywords):
            x = layer(x, *args[0], **kwargs[0])
        actual = model.lm_head(model.model.norm(x))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tiny_model", ["zai-org/GLM-4.7-Flash"], indirect=True)
def test_glm_layer_re_then_gptq_and_legacy_ce(tiny_model, tmp_path, monkeypatch):
    name, model, first_moe = tiny_model
    tokens = torch.tensor([[2, 3, 5, 7], [3, 2, 8, 6]])
    batches = [(sample.unsqueeze(0), None) for sample in tokens]
    cast_model_preserving_fp32(model, torch.bfloat16)
    stats_args = SimpleNamespace(
        model_name=name, layer_grads_path=str(tmp_path / "gradients.pt"),
        layer_re_path=str(tmp_path / "LayerRE_c4-Seed0.pkl"), forward_batch_size=2,
        wbits="1,2,3", blocksize=4,
    )
    compute_layer_grads(model, batches, stats_args)
    grads = torch.load(stats_args.layer_grads_path)
    assert set(grads) == {0, 1, 2}
    assert all(grad.isfinite().all() and grad.count_nonzero() for grad in grads.values())
    assert all(param.grad is None for param in model.parameters())
    compute_faster_layer_re(model, batches, stats_args, device="cpu")

    assert allocate_bits.compute_total_bits(name, 2.0, [1, 2, 3]) == 46 * 64 * 2
    monkeypatch.setattr(allocate_bits, "get_model_info", lambda _: ModelInfo(3, 1, 6, 1, 2, False))
    allocation, _ = allocate_bits.run_solver(SimpleNamespace(
        model_name=name, allocation_metric="layer_re", bit_candidates="1,2,3",
        layer_re_path=stats_args.layer_re_path, pop_cost_path=None, bit_budget=2.0,
        extra_constr="c2c3", ilp_backend="highs", max_prune_ratio=0.25,
        save_path=str(tmp_path / "allocation.pkl"),
    ))
    assert set(allocation) == {1, 2}
    assert all(set(bits) == set(range(6)) and set(bits.values()) <= {1, 2, 3} for bits in allocation.values())
    args = SimpleNamespace(
        model_name=name, groupsize=4, blocksize=4, percdamp=0.01, mse=False,
        actorder=False, static_groups=False, reproduce_mcmoe=False, verbose=False,
        gate_wbits=16, softmax_attn_wbits=4, dense_wbits=4,
        nsamples=2, rft_batch_size=2, rft_epochs=1, rft_lr=1e-2, rft_wd=0.0,
    )
    quantize_weights_gptq(model, batches, args, allocation, device="cpu")
    with torch.no_grad():
        for module, attr in routing_biases(model):
            getattr(module, attr).add_(0.000123)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    finetune_routers(model, batches, args)
    after = model.state_dict()
    changed = {key for key in before if not torch.equal(before[key], after[key])}
    expected = {f"model.layers.{i}.mlp.gate.weight" for i in range(first_moe, 3)}
    assert changed and changed <= expected
    for module, attr in routing_biases(model):
        assert getattr(module, attr).dtype == torch.float32


def test_pop_dense_prefix_and_pruned_checkpoint_round_trip(tiny_model, tmp_path):
    name, model, first_moe = tiny_model
    x = torch.tensor([[2, 3, 5, 7]])
    costs, counts = compute_qwen3_expert_costs(
        model, [x], name, candidate_bits=[0, 2, 3], blocksize=4, device="cpu",
    )
    assert costs.shape == (3 - first_moe, 6, 3)
    assert counts.sum(dim=1).tolist() == [8] * (3 - first_moe)
    assert costs[counts > 0].isfinite().all()
    filled, imputed = impute_zero_frequency_costs(costs, counts)
    cost_file = tmp_path / "pop.pt"
    torch.save({
        "costs": filled, "counts": counts, "imputed_mask": imputed,
        "candidate_bits": torch.tensor([0, 2, 3]),
    }, cost_file)
    optimized = PoPSolver(
        cost_file, [0, 2, 3], max_prune_ratio=0.34, top_k=2, start_layer_idx=first_moe,
    ).solve_all(total_bits=(3 - first_moe) * 10)
    assert list(optimized) == list(range(first_moe, 3))
    assert {len(bits) for bits in optimized.values()} == {6}

    original_bias = torch.tensor([-0.25, 0.125, -0.0625, 0.5, -0.5, 0.375])
    for module, attr in routing_biases(model):
        getattr(module, attr).copy_(original_bias)
    allocation = {i: {0: 2, 1: 0, 2: 2, 3: 3, 4: 0, 5: 3} for i in range(first_moe, 3)}
    result = prune_zero_bit_experts(model, name, allocation)
    assert result.remapped_bit_config[first_moe] == {0: 2, 1: 2, 2: 3, 3: 3}
    assert list(result.metadata["layers"]) == [str(i) for i in range(first_moe, 3)]
    assert kept_expert_ids_from_pruning_metadata(result.metadata) == result.kept_expert_ids

    expected_bias = original_bias[[0, 2, 3, 5]]
    cast_model_preserving_fp32(model, torch.bfloat16)
    model.save_pretrained(tmp_path)
    reloaded = load_causal_lm_checkpoint(tmp_path, model_name=name)
    for module, attr in routing_biases(reloaded):
        torch.testing.assert_close(getattr(module, attr), expected_bias)
    torch.testing.assert_close(model(x).logits, reloaded(x).logits, rtol=0, atol=0)


def test_gptq_then_roco_on_native_decoder(tiny_model, tmp_path):
    name, model, first_moe = tiny_model
    x = torch.tensor([[2, 3, 5, 7, 11, 13], [3, 2, 8, 6, 5, 7]])
    with torch.no_grad():
        # Deterministically leave some experts without calibration samples.
        for module, attr in routing_biases(model):
            getattr(module, attr).copy_(torch.arange(6) * 2.0)
        teacher = model.model(x).last_hidden_state
    targets = TeacherTargets(x, None, None, teacher, {})
    allocation = {i: {0: 2, 1: 0, 2: 2, 3: 3, 4: 0, 5: 3} for i in range(first_moe, 3)}
    pruned = prune_zero_bit_experts(model, name, allocation)
    moe = model.model.layers[first_moe].mlp
    packed = hasattr(moe.experts, "gate_up_proj")
    weight = moe.experts.gate_up_proj[0] if packed else moe.experts[0].gate_proj.weight
    before = weight.detach().clone()
    args = SimpleNamespace(
        model_name=name, groupsize=4, blocksize=4, percdamp=0.01, mse=False,
        actorder=False, static_groups=False, reproduce_mcmoe=False, verbose=False,
        gate_wbits=16, softmax_attn_wbits=4, dense_wbits=4,
        nsamples=2, rft_batch_size=1, rft_epochs=1, rft_lr=1e-3, rft_wd=0.0,
    )
    quantized = quantize_weights_gptq(model, [x], args, pruned.remapped_bit_config, device="cpu")
    weight = moe.experts.gate_up_proj[0] if packed else moe.experts[0].gate_proj.weight
    assert not torch.equal(weight, before)
    assert weight.count_nonzero()
    assert "0.mlp.up_proj" in quantized
    assert all(not name.endswith("self_attn.gate_proj") for name in quantized)

    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    checkpoint = tmp_path / "gptq"
    save_gptq_checkpoint(model, tokenizer, checkpoint, {"pruning": pruned.metadata})
    loaded = load_causal_lm_checkpoint(checkpoint, model_dtype="float32", model_name=name)
    torch.testing.assert_close(loaded(x).logits, model(x).logits, rtol=0, atol=0)
    model = loaded
    finetune_norms_distill_ce(model, targets, args, optimize_input_norm=True)
    model.eval()
    assert model(x).logits.isfinite().all()

    output = tmp_path / "roco"
    save_quantized_model(model, tokenizer, output, "bfloat16", real_quant=False)
    loaded = load_causal_lm_checkpoint(output, model_name=name)
    torch.testing.assert_close(loaded.model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq)
    torch.testing.assert_close(loaded(x).logits, model(x).logits, rtol=0, atol=0)


def test_roco_compensates_ffn_input_norm(tiny_model):
    name, model, first_moe = tiny_model
    layer = model.model.layers[first_moe]
    controller = RouterCompensatedNorm(layer, name)
    known_norm = layer.pre_mlp_layernorm if name.startswith("arcee-ai") else layer.post_attention_layernorm
    assert controller.norm is known_norm
    _, router = get_router_module(layer, name)
    x = torch.randn(1, 2, 16)
    with torch.no_grad():
        before = router(known_norm(x).reshape(-1, 16))
        controller.delta.fill_(0.2)
        controller.fold()
        after = router(known_norm(x).reshape(-1, 16))
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_direct_rmsnorm_only_changes_decoder_gamma(tiny_model, dtype, tmp_path):
    name, model, _ = tiny_model
    args = SimpleNamespace(
        model_name=name, nsamples=2, rft_batch_size=1, rft_epochs=1, rft_lr=0.05, rft_wd=0.0,
    )
    mode = NORM_DISTILL_MODES["direct_rmsnorm"]
    input_ids = torch.tensor([[2, 3, 5, 7], [3, 2, 8, 6]])
    teacher = TeacherTargets(input_ids, None, None, torch.randn(2, 4, 16), {})
    if name == "arcee-ai/Trinity-Mini":
        with pytest.raises(ValueError, match="requires GLM"):
            finetune_norms_distill_ce(model, teacher, args, **mode)
        return

    cast_model_preserving_fp32(model, dtype)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    finetune_norms_distill_ce(model, teacher, args, **mode)
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(before[key].dtype == after[key].dtype for key in before)
    changed = {key for key in before if not torch.equal(before[key], after[key])}
    expected = {
        f"model.layers.{i}.{norm}.weight"
        for i in range(model.config.num_hidden_layers)
        for norm in ("input_layernorm", "post_attention_layernorm")
    }
    assert changed == expected

    model.eval()
    if dtype == torch.bfloat16:
        tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
        save_quantized_model(model, tokenizer, tmp_path, "bfloat16", real_quant=False)
        loaded = load_causal_lm_checkpoint(tmp_path, model_name=name)
        torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits, rtol=0, atol=0)
