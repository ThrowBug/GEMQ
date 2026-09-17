"""Pruning-aware local MoE reconstruction after a saved GPTQ checkpoint.

The teacher and student receive the *same student pre-MoE activation*.  Only
the current teacher or student layer is resident on CUDA at a time.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gemq.utils.model_utils import get_blocks, get_moe_block, get_router_module


def _hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected a tensor or a tuple beginning with a tensor, got {type(output)!r}")


def relative_moe_mse(prediction, target):
    prediction = prediction.float()
    target = target.float()
    return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)


@dataclass(frozen=True)
class ReconstructionConfig:
    stage: str = "norm_only"
    epochs: int = 1
    batch_size: int = 1
    norm_lr: float = 1e-4
    router_lr: float = 1e-5

    def __post_init__(self):
        if self.stage not in {"norm_only", "norm_then_router", "decoupled_joint"}:
            raise ValueError(f"Unsupported reconstruction stage: {self.stage}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.norm_lr <= 0 or self.router_lr <= 0:
            raise ValueError("norm_lr and router_lr must be positive")


class DecoupledRouterNorm:
    """Differentiable gamma/V parametrization, folded into normal weights at exit.

    If delta=0 and V=W0, the original student is recovered.  Delta alone
    approximately preserves router logits: (V / exp(delta)) @ (x * exp(delta)).
    The discrepancy in BF16 can still alter ties near a Top-K boundary.
    """

    def __init__(self, layer, model_name):
        self.norm = layer.post_attention_layernorm
        _, self.gate = get_router_module(layer, model_name)
        if not hasattr(self.norm, "weight") or not hasattr(self.gate, "weight"):
            raise TypeError("Expected a weighted post-attention norm and linear router")
        if self.gate.weight.shape[1] != self.norm.weight.numel():
            raise ValueError("Router input width differs from post-attention norm width")
        self.delta = torch.nn.Parameter(torch.zeros_like(self.norm.weight, dtype=torch.float32))
        self.v = torch.nn.Parameter(self.gate.weight.detach().float().clone())
        self._norm_handle = None
        self._gate_handle = None

    def __enter__(self):
        def scale_norm(_module, _inputs, output):
            return output * self.delta.exp().to(output.dtype)

        def replace_gate(module, inputs, _output):
            inputs0 = inputs[0]
            weight = (self.v * self.delta.neg().exp().unsqueeze(0)).to(inputs0.dtype)
            return F.linear(inputs0, weight, module.bias)

        self._norm_handle = self.norm.register_forward_hook(scale_norm)
        self._gate_handle = self.gate.register_forward_hook(replace_gate)
        return self

    def __exit__(self, *_exc):
        self._gate_handle.remove()
        self._norm_handle.remove()
        self._gate_handle = self._norm_handle = None

    @torch.no_grad()
    def fold(self):
        scale = self.delta.exp()
        self.norm.weight.copy_((self.norm.weight.float() * scale).to(self.norm.weight.dtype))
        self.gate.weight.copy_((self.v * scale.reciprocal().unsqueeze(0)).to(self.gate.weight.dtype))


@torch.no_grad()
def collect_pre_moe_inputs(layer, inputs, layer_kwargs, device):
    """Capture the post-attention residual u and skip the expensive MoE call."""
    result = torch.empty_like(inputs, device="cpu")
    current = [None]

    class _CapturedPreMoe(RuntimeError):
        pass

    def capture(_module, arguments):
        result[current[0]].copy_(arguments[0].detach().to("cpu"))
        raise _CapturedPreMoe

    handle = layer.post_attention_layernorm.register_forward_pre_hook(capture)
    try:
        for index in range(inputs.shape[0]):
            current[0] = index
            try:
                layer(inputs[index:index + 1].to(device), **layer_kwargs)
            except _CapturedPreMoe:
                pass
    finally:
        handle.remove()
    return result


@torch.no_grad()
def collect_teacher_moe_targets(teacher_layer, pre_moe_inputs, model_name, device):
    targets = torch.empty_like(pre_moe_inputs, device="cpu")
    teacher_moe = get_moe_block(teacher_layer, model_name)
    for index in range(pre_moe_inputs.shape[0]):
        u = pre_moe_inputs[index:index + 1].to(device)
        output = _hidden(teacher_moe(teacher_layer.post_attention_layernorm(u)))
        targets[index].copy_(output.to("cpu"))
    return targets


@torch.no_grad()
def local_error(layer, inputs, targets, model_name, batch_size, device):
    moe = get_moe_block(layer, model_name)
    numerator = denominator = 0.0
    for start in range(0, inputs.shape[0], batch_size):
        end = min(start + batch_size, inputs.shape[0])
        u = inputs[start:end].to(device)
        target = targets[start:end].to(device).float()
        output = _hidden(moe(layer.post_attention_layernorm(u))).float()
        numerator += (output - target).square().sum().item()
        denominator += target.square().sum().item()
    return numerator / max(denominator, 1e-8)


def _train_stage(layer, parametrization, inputs, targets, model_name, config, name, layer_idx, device):
    parametrization.delta.requires_grad_(name != "router")
    parametrization.v.requires_grad_(name != "norm")
    groups = []
    if parametrization.delta.requires_grad:
        groups.append({"params": [parametrization.delta], "lr": config.norm_lr})
    if parametrization.v.requires_grad:
        groups.append({"params": [parametrization.v], "lr": config.router_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    moe = get_moe_block(layer, model_name)
    with torch.enable_grad():
        for epoch in range(config.epochs):
            started = time.time()
            loss_sum = 0.0
            steps = 0
            for start in range(0, inputs.shape[0], config.batch_size):
                end = min(start + config.batch_size, inputs.shape[0])
                u = inputs[start:end].to(device)
                target = targets[start:end].to(device)
                output = _hidden(moe(layer.post_attention_layernorm(u)))
                loss = relative_moe_mse(output, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item()
                steps += 1
            print(
                f"[router-norm layer {layer_idx:>2} | {name} epoch {epoch:>2}] "
                f"relative_mse={loss_sum / steps:.6f}, elapsed={time.time() - started:.2f}s"
            )
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def propagate_layer(layer, inputs, layer_kwargs, device):
    outputs = torch.empty_like(inputs, device="cpu")
    for index in range(inputs.shape[0]):
        output = _hidden(layer(inputs[index:index + 1].to(device), **layer_kwargs))
        outputs[index].copy_(output.to("cpu"))
    return outputs


def _cpu_offload(layer):
    layer.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def finetune_router_norm_reconstruction(
    teacher, student, train_inputs, validation_inputs, layer_kwargs, model_name, config,
    device="cuda",
):
    """Tune layer by layer and return small baseline snapshots for global rollback."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-30B layerwise trainer")
    teacher_layers = get_blocks(teacher, model_name)
    student_layers = get_blocks(student, model_name)
    if len(teacher_layers) != len(student_layers):
        raise ValueError("Teacher and student layer counts differ")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    student.eval()
    snapshots = []
    for layer_idx, (teacher_layer, student_layer) in enumerate(zip(teacher_layers, student_layers)):
        norm = student_layer.post_attention_layernorm
        _, gate = get_router_module(student_layer, model_name)
        snapshots.append((norm.weight.detach().cpu().clone(), gate.weight.detach().cpu().clone()))

        student_layer.to(device)
        train_u = collect_pre_moe_inputs(student_layer, train_inputs, layer_kwargs, device)
        val_u = collect_pre_moe_inputs(student_layer, validation_inputs, layer_kwargs, device)
        _cpu_offload(student_layer)

        teacher_layer.to(device)
        train_targets = collect_teacher_moe_targets(teacher_layer, train_u, model_name, device)
        val_targets = collect_teacher_moe_targets(teacher_layer, val_u, model_name, device)
        _cpu_offload(teacher_layer)

        student_layer.to(device)
        baseline = local_error(student_layer, val_u, val_targets, model_name, config.batch_size, device)
        parametrization = DecoupledRouterNorm(student_layer, model_name)
        try:
            with parametrization:
                best_loss = baseline
                best_delta = best_v = None
                stages = ["norm"]
                if config.stage in {"norm_then_router", "decoupled_joint"}:
                    stages.append("router")
                if config.stage == "decoupled_joint":
                    stages.append("joint")
                for stage in stages:
                    _train_stage(
                        student_layer, parametrization, train_u, train_targets,
                        model_name, config, stage, layer_idx, device,
                    )
                    stage_loss = local_error(
                        student_layer, val_u, val_targets, model_name, config.batch_size, device
                    )
                    print(f"[router-norm layer {layer_idx:>2} | {stage}] holdout={stage_loss:.6f}")
                    if stage_loss < best_loss:
                        best_loss = stage_loss
                        best_delta = parametrization.delta.detach().clone()
                        best_v = parametrization.v.detach().clone()
                candidate = best_loss
                if best_delta is not None:
                    with torch.no_grad():
                        parametrization.delta.copy_(best_delta)
                        parametrization.v.copy_(best_v)
            if candidate < baseline:
                parametrization.fold()
                folded = local_error(
                    student_layer, val_u, val_targets, model_name, config.batch_size, device
                )
                if folded < baseline:
                    choice = f"accepted (folded={folded:.6f})"
                else:
                    norm.weight.copy_(snapshots[-1][0].to(norm.weight.device))
                    gate.weight.copy_(snapshots[-1][1].to(gate.weight.device))
                    choice = f"baseline retained after BF16 fold (folded={folded:.6f})"
            else:
                choice = "baseline retained"
            print(
                f"[router-norm layer {layer_idx:>2}] holdout={baseline:.6f} -> "
                f"{candidate:.6f} ({choice})"
            )
        finally:
            del parametrization
        del train_u, val_u, train_targets, val_targets

        next_train = propagate_layer(student_layer, train_inputs, layer_kwargs, device)
        next_validation = propagate_layer(student_layer, validation_inputs, layer_kwargs, device)
        _cpu_offload(student_layer)
        train_inputs, validation_inputs = next_train, next_validation
        gc.collect()
    return snapshots


@torch.no_grad()
def restore_router_norm_snapshot(student, model_name, snapshots):
    layers = get_blocks(student, model_name)
    if len(layers) != len(snapshots):
        raise ValueError("Snapshot layer count differs from student")
    for layer, (norm_weight, gate_weight) in zip(layers, snapshots):
        layer.post_attention_layernorm.weight.copy_(norm_weight)
        _, gate = get_router_module(layer, model_name)
        gate.weight.copy_(gate_weight)
