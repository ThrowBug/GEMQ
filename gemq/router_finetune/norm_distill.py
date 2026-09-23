"""Distilled-CE tuning of attention and MoE input norms."""

import time

import torch

from gemq.router_finetune.losses import compute_causal_output_distill_ce
from gemq.utils.model_utils import (
    NAME_TO_MODEL,
    ModelType,
    get_blocks,
    get_router_module,
)


class NormScale:
    """Learn a channel scale for one norm and fold it into the norm weight."""

    def __init__(self, norm):
        self.norm = norm
        self.delta = torch.nn.Parameter(
            torch.zeros_like(self.norm.weight, dtype=torch.float32)
        )
        self._norm_hook = None

    def install(self):
        def scale_norm(_module, _inputs, output):
            return (output.float() * self.delta.exp()).to(output.dtype)

        self._norm_hook = self.norm.register_forward_hook(scale_norm)

    def remove(self):
        if self._norm_hook is not None:
            self._norm_hook.remove()
        self._norm_hook = None

    @torch.no_grad()
    def fold(self):
        old_norm = self.norm.weight.detach().clone()
        new_norm = (old_norm.float() * self.delta.exp()).to(old_norm.dtype)

        # Return the scale actually representable by the stored dtype. Router
        # compensation must use this value rather than the FP32 training scale.
        realized_scale = torch.ones_like(self.delta)
        nonzero = old_norm.ne(0)
        realized_scale[nonzero] = (
            new_norm[nonzero].float() / old_norm[nonzero].float()
        )
        if not torch.isfinite(realized_scale).all() or not (realized_scale > 0).all():
            raise FloatingPointError("The learned norm scale cannot be folded safely.")
        self.norm.weight.copy_(new_norm)
        return realized_scale


class ZeroCenteredNormScale(NormScale):
    """Fold a channel scale into a zero-centered RMSNorm parameter.

    Qwen3.5 stores ``weight = gamma - 1`` and applies ``gamma = 1 + weight``
    in the forward pass.  Multiplying the stored parameter directly would lose
    the learned scale, especially for the zero initialization used by the model.
    """

    @torch.no_grad()
    def fold(self):
        old_weight = self.norm.weight.detach().clone()
        old_gain = old_weight.float().add(1.0)
        new_weight = (old_gain * self.delta.exp() - 1.0).to(old_weight.dtype)
        new_gain = new_weight.float().add(1.0)

        realized_scale = torch.ones_like(self.delta)
        nonzero = old_gain.ne(0)
        realized_scale[nonzero] = new_gain[nonzero] / old_gain[nonzero]
        if not torch.isfinite(realized_scale).all() or not (realized_scale > 0).all():
            raise FloatingPointError(
                "The learned zero-centered norm scale cannot be folded safely."
            )
        self.norm.weight.copy_(new_weight)
        return realized_scale


class RouterCompensatedNorm(NormScale):
    """Learn a post-attention norm scale while preserving router inputs."""

    def __init__(self, layer, model_name):
        super().__init__(layer.post_attention_layernorm)
        _, self.router = get_router_module(layer, model_name)
        if self.router.weight.shape[1] != self.norm.weight.numel():
            raise ValueError("Router input width differs from post-attention norm width.")
        if self.router.weight.device != self.norm.weight.device:
            raise ValueError("Router and post-attention norm must be on the same device.")
        self._router_hook = None
        self._unscaled = None

    def install(self):
        def scale_norm(_module, _inputs, output):
            if self._unscaled is not None:
                raise RuntimeError("Norm ran twice before the router consumed its output.")
            self._unscaled = output
            return (output.float() * self.delta.exp()).to(output.dtype)

        def restore_router_input(_module, inputs):
            if self._unscaled is None:
                raise RuntimeError("Router ran without a matching norm output.")
            unscaled = self._unscaled
            self._unscaled = None
            if inputs[0].numel() != unscaled.numel():
                raise RuntimeError("Router input shape is incompatible with the norm output.")
            # Qwen3-MoE flattens batch and sequence before invoking the router.
            return (unscaled.reshape_as(inputs[0]), *inputs[1:])

        self._norm_hook = self.norm.register_forward_hook(scale_norm)
        self._router_hook = self.router.register_forward_pre_hook(restore_router_input)

    def remove(self):
        if self._router_hook is not None:
            self._router_hook.remove()
        super().remove()
        self._router_hook = None
        self._unscaled = None

    @torch.no_grad()
    def fold(self):
        """Fold the learned scale into normal HF norm/router weights."""
        realized_scale = super().fold()
        self.router.weight.copy_(
            (
                self.router.weight.float()
                * realized_scale.reciprocal().unsqueeze(0)
            ).to(self.router.weight.dtype)
        )
        return realized_scale


class Qwen35RouterCompensatedNorm(ZeroCenteredNormScale):
    """Scale a Qwen3.5 MoE input while preserving both gating functions."""

    def __init__(self, layer, model_name):
        super().__init__(layer.post_attention_layernorm)
        _, self.router = get_router_module(layer, model_name)
        try:
            self.shared_expert_gate = layer.mlp.shared_expert_gate
        except AttributeError as exc:
            raise ValueError(
                "Qwen3.5 router compensation requires layer.mlp.shared_expert_gate."
            ) from exc

        width = self.norm.weight.numel()
        for label, gate in (
            ("router", self.router),
            ("shared-expert gate", self.shared_expert_gate),
        ):
            if gate.weight.shape[1] != width:
                raise ValueError(
                    f"Qwen3.5 {label} input width differs from post-attention "
                    "norm width."
                )
            if gate.weight.device != self.norm.weight.device:
                raise ValueError(
                    f"Qwen3.5 {label} and post-attention norm must be on the "
                    "same device."
                )

        self._gate_hooks = []
        self._unscaled = None
        self._pending_consumers = set()

    def install(self):
        expected_consumers = {"router", "shared_expert_gate"}

        def scale_norm(_module, _inputs, output):
            if self._unscaled is not None or self._pending_consumers:
                raise RuntimeError(
                    "Qwen3.5 norm ran again before both gates consumed its output."
                )
            self._unscaled = output
            self._pending_consumers = set(expected_consumers)
            return (output.float() * self.delta.exp()).to(output.dtype)

        def restore_gate_input(label):
            def restore(_module, inputs):
                if self._unscaled is None or label not in self._pending_consumers:
                    raise RuntimeError(
                        f"Qwen3.5 {label} ran without a matching norm output."
                    )
                unscaled = self._unscaled
                if inputs[0].numel() != unscaled.numel():
                    raise RuntimeError(
                        f"Qwen3.5 {label} input shape is incompatible with the "
                        "norm output."
                    )
                self._pending_consumers.remove(label)
                if not self._pending_consumers:
                    self._unscaled = None
                return (unscaled.reshape_as(inputs[0]), *inputs[1:])

            return restore

        self._norm_hook = self.norm.register_forward_hook(scale_norm)
        self._gate_hooks = [
            self.router.register_forward_pre_hook(restore_gate_input("router")),
            self.shared_expert_gate.register_forward_pre_hook(
                restore_gate_input("shared_expert_gate")
            ),
        ]

    def remove(self):
        for hook in self._gate_hooks:
            hook.remove()
        self._gate_hooks = []
        super().remove()
        self._unscaled = None
        self._pending_consumers.clear()

    @torch.no_grad()
    def fold(self):
        realized_scale = super().fold()
        inverse_scale = realized_scale.reciprocal().unsqueeze(0)
        for gate in (self.router, self.shared_expert_gate):
            gate.weight.copy_(
                (gate.weight.float() * inverse_scale).to(gate.weight.dtype)
            )
        return realized_scale


def _print_scale_summary(label, learned_scales, realized_scales):
    learned = torch.cat(
        [scale.detach().float().reshape(-1).cpu() for scale in learned_scales]
    )
    realized = torch.cat(
        [scale.detach().float().reshape(-1).cpu() for scale in realized_scales]
    )
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    learned_q = torch.quantile(learned, quantiles).tolist()
    realized_q = torch.quantile(realized, quantiles).tolist()
    changed = realized.ne(1.0).float().mean().item() * 100.0
    print(
        f"[norm-scale {label}] "
        f"learned p05={learned_q[0]:.6f} p50={learned_q[1]:.6f} "
        f"p95={learned_q[2]:.6f} | "
        f"realized p05={realized_q[0]:.6f} p50={realized_q[1]:.6f} "
        f"p95={realized_q[2]:.6f} | changed={changed:.2f}%"
    )


def finetune_norms_distill_ce(
    model,
    teacher_targets,
    args,
    *,
    optimize_input_norm=False,
    router_compensated=True,
):
    """Tune selected norm scales jointly with teacher soft-label CE."""
    if teacher_targets.final_hidden_states is None:
        raise RuntimeError("Norm distillation requires teacher final hidden states.")

    original_use_cache = model.config.use_cache
    original_dtype = next(model.parameters()).dtype
    model.config.use_cache = False
    model.train()
    model.to(torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_ids = teacher_targets.input_ids
    attention_mask = teacher_targets.attention_mask
    if input_ids.shape[0] < args.nsamples:
        raise ValueError(
            f"Teacher cache has {input_ids.shape[0]} samples, but --nsamples={args.nsamples}."
        )

    model_type = NAME_TO_MODEL[args.model_name]
    zero_centered = model_type == ModelType.QWEN35MOE
    controllers = []
    controller_groups = {"input": [], "post": []}
    for layer in get_blocks(model, args.model_name):
        if optimize_input_norm:
            input_controller = (
                ZeroCenteredNormScale(layer.input_layernorm)
                if zero_centered
                else NormScale(layer.input_layernorm)
            )
            controllers.append(input_controller)
            controller_groups["input"].append(input_controller)
        if router_compensated:
            post_controller = (
                Qwen35RouterCompensatedNorm(layer, args.model_name)
                if zero_centered
                else RouterCompensatedNorm(layer, args.model_name)
            )
        else:
            post_controller = (
                ZeroCenteredNormScale(layer.post_attention_layernorm)
                if zero_centered
                else NormScale(layer.post_attention_layernorm)
            )
        controllers.append(post_controller)
        controller_groups["post"].append(post_controller)
    for controller in controllers:
        controller.install()

    optimizer = torch.optim.AdamW(
        [controller.delta for controller in controllers],
        lr=args.rft_lr,
        weight_decay=args.rft_wd,
    )
    input_device = model.get_input_embeddings().weight.device
    head_parameter = next(model.lm_head.parameters())

    completed = False
    try:
        for epoch in range(args.rft_epochs):
            started = time.time()
            loss_sum = 0.0
            steps = 0
            for start in range(0, args.nsamples, args.rft_batch_size):
                end = min(start + args.rft_batch_size, args.nsamples)
                batch_ids = input_ids[start:end].to(input_device)
                batch_mask = None
                if attention_mask is not None:
                    batch_mask = attention_mask[start:end].to(input_device)
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)

                with torch.no_grad():
                    teacher_hidden = teacher_targets.final_hidden_states[start:end].to(
                        device=head_parameter.device, dtype=head_parameter.dtype
                    )
                    teacher_logits = model.lm_head(teacher_hidden).to(
                        outputs.logits.device
                    )
                loss_mask = (
                    batch_mask.to(outputs.logits.device)
                    if batch_mask is not None
                    else None
                )
                loss = compute_causal_output_distill_ce(
                    outputs.logits, teacher_logits, loss_mask
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item()
                steps += 1
                if steps == 1 or steps % 32 == 0:
                    print(
                        f"[norm distill | epoch {epoch} | step {steps - 1}] "
                        f"loss={loss_sum / steps:.6f}"
                    )
            print(
                f"[norm distill | epoch {epoch}] loss={loss_sum / steps:.6f}, "
                f"elapsed={time.time() - started:.2f}s"
            )
        completed = True
    finally:
        optimizer.zero_grad(set_to_none=True)
        for controller in controllers:
            controller.remove()
        model.config.use_cache = original_use_cache

    if completed:
        learned_scales = {
            id(controller): controller.delta.detach().exp()
            for controller in controllers
        }
        realized_scales = {}
        for controller in controllers:
            realized_scales[id(controller)] = controller.fold()
        for label in ("input", "post"):
            group = controller_groups[label]
            if group:
                _print_scale_summary(
                    label,
                    [learned_scales[id(controller)] for controller in group],
                    [realized_scales[id(controller)] for controller in group],
                )
    model.to(original_dtype)
