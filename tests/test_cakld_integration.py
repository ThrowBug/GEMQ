"""Exercise the real entry functions without importing optional CUDA quantizers.

Only top-level function definitions are extracted from quantize.py. Dependencies
are supplied explicitly so these tests do not need hqq/gemlite or model weights.
"""

import argparse
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

torch = pytest.importorskip("torch")

from gemq.router_finetune.config import (
    RFT_TIMINGS, RFT_TRAINERS, ROUTER_LOSS_TYPES, parse_cakld_gamma,
)
from gemq.router_finetune.confidence import resolve_cakld_gamma
from gemq.router_finetune.losses import compute_causal_output_cakld, compute_causal_output_distill_ce


def entry_functions():
    source = Path(__file__).resolve().parents[1] / "gemq" / "quantize.py"
    names = {
        "parse_args", "finetune_routers_distill_ce", "finetune_routers_cakld",
        "_finetune_routers_output_distillation",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {
        "argparse": argparse, "torch": torch, "math": math, "time": time,
        "RFT_TIMINGS": RFT_TIMINGS, "RFT_TRAINERS": RFT_TRAINERS,
        "ROUTER_LOSS_TYPES": ROUTER_LOSS_TYPES, "parse_cakld_gamma": parse_cakld_gamma,
        "resolve_cakld_gamma": resolve_cakld_gamma,
        "compute_causal_output_cakld": compute_causal_output_cakld,
        "compute_causal_output_distill_ce": compute_causal_output_distill_ce,
        "get_router_params": lambda model, name: list(model.gate.parameters()),
        "get_module_type": lambda name, model_name: "gate" if name.startswith("gate.") else "other",
        "LinearModuleType": SimpleNamespace(GATE="gate"),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def test_parser_retains_defaults_and_accepts_cakld(monkeypatch):
    parse = entry_functions()["parse_args"]
    base = ["quantize", "--model", "toy", "--model_name", "toy"]
    monkeypatch.setattr("sys.argv", base)
    args = parse()
    assert args.rft_trainer == "legacy_ce"
    assert args.rft_timing == "after_all_quantization"
    assert args.rft_cakld_gamma == "auto"
    monkeypatch.setattr("sys.argv", base + ["--rft_trainer", "cakld", "--rft_cakld_gamma", ".5"])
    assert parse().rft_cakld_gamma == 0.5
    monkeypatch.setattr("sys.argv", base + ["--rft_cakld_gamma", "nan"])
    with pytest.raises(SystemExit):
        parse()


class ToyRouterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.gate = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        self.config = SimpleNamespace(use_cache=True)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None):
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        hidden = hidden * (1.0 + self.gate(hidden).sigmoid())
        return SimpleNamespace(logits=self.lm_head(hidden))


@pytest.mark.cuda
def test_joint_trainer_gamma_is_resolved_once_and_only_routers_change(device, capsys):
    torch.manual_seed(42)
    model = ToyRouterModel().to(device=device, dtype=torch.bfloat16)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    cached = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]), attention_mask=None,
        final_hidden_states=torch.randn(2, 4, 4, dtype=torch.bfloat16),
    )
    args = SimpleNamespace(
        nsamples=2, model_name="toy", rft_epochs=2, rft_batch_size=1,
        rft_lr=0.02, rft_wd=0.01, verbose=False, rft_cakld_gamma="auto",
    )
    namespace = entry_functions()
    calls = []

    def resolve(*values):
        assert model.forward_calls == 0
        calls.append(True)
        return resolve_cakld_gamma(*values)

    namespace["resolve_cakld_gamma"] = resolve
    result = namespace["finetune_routers_cakld"](model, cached, args)
    assert calls == [True]
    assert 0 <= result["resolved_gamma"] <= 1
    assert result["confidence_valid_tokens"] == 6
    assert model.forward_calls == 4
    assert model.config.use_cache
    for name, param in model.named_parameters():
        if name.startswith("gate."):
            assert not torch.equal(original[name], param)
        else:
            assert torch.equal(original[name], param)
            assert param.grad is None
    output = capsys.readouterr().out
    assert "forward_kl=" in output and "reverse_kl=" in output
    assert "epoch running means" in output


@pytest.mark.cuda
def test_joint_cakld_zero_gamma_matches_existing_ce_updates(device):
    torch.manual_seed(43)
    ce_model = ToyRouterModel().to(device=device, dtype=torch.bfloat16)
    cakld_model = copy.deepcopy(ce_model)
    cached = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        final_hidden_states=torch.randn(2, 4, 4, dtype=torch.bfloat16),
    )
    args = SimpleNamespace(
        nsamples=2, model_name="toy", rft_epochs=1, rft_batch_size=1,
        rft_lr=0.02, rft_wd=0.01, verbose=False, rft_cakld_gamma=0.0,
    )
    namespace = entry_functions()
    namespace["finetune_routers_distill_ce"](ce_model, cached, args)
    result = namespace["finetune_routers_cakld"](cakld_model, cached, args)
    assert result["gamma_source"] == "manual"
    for ce, cakld in zip(ce_model.parameters(), cakld_model.parameters()):
        assert torch.equal(ce, cakld)
