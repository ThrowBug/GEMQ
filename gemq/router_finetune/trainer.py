import gc
import time

import torch

from gemq.router_finetune.losses import compute_causal_output_kl, compute_router_loss
from gemq.utils.model_utils import (
    compute_decoder_inputs,
    extract_router_logits,
    get_blocks,
    get_model_info,
    get_router_module,
    get_router_modules,
)


def _extract_layer_hidden(layer_output):
    if torch.is_tensor(layer_output):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output and torch.is_tensor(layer_output[0]):
        return layer_output[0]
    raise TypeError(f"Could not extract decoder hidden states from {type(layer_output)!r}.")


def _set_requires_grad(parameters, enabled):
    for parameter in parameters:
        parameter.requires_grad = enabled


def _router_top_m(config, model_name, teacher_logits):
    top_k = get_model_info(model_name).num_experts_per_token
    return config.effective_top_m(teacher_logits.shape[-1], top_k)


def _train_router_from_inputs(
    router, router_inputs, teacher_logits, token_mask, model_name, config, layer_idx
):
    """Train one router from detached, actual student inputs."""
    parameters = list(router.parameters())
    if not parameters:
        raise ValueError(f"Layer {layer_idx} router has no parameters.")
    _set_requires_grad(parameters, True)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    top_m = _router_top_m(config, model_name, teacher_logits)
    num_samples = router_inputs.shape[0]

    with torch.enable_grad():
        for epoch in range(config.epochs):
            loss_sum = 0.0
            step_count = 0
            start_time = time.time()
            for start in range(0, num_samples, config.batch_size):
                end = min(num_samples, start + config.batch_size)
                inputs = router_inputs[start:end].detach()
                target = teacher_logits[start:end].to(device=inputs.device, non_blocking=True)
                output = router(inputs)
                student_logits = extract_router_logits(output).reshape_as(target)
                loss = compute_router_loss(
                    config.router_loss,
                    student_logits,
                    target,
                    top_m=top_m,
                    token_mask=token_mask[start:end] if token_mask is not None else None,
                )
                loss = config.router_loss_weight * loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_sum += loss.detach().item()
                step_count += 1
            print(
                f"[router layer {layer_idx:>2} | epoch {epoch:>2}] "
                f"loss={loss_sum / max(step_count, 1):.6f}, top_m={top_m}, "
                f"elapsed={time.time() - start_time:.2f}s"
            )
    optimizer.zero_grad(set_to_none=True)
    _set_requires_grad(parameters, False)


def finetune_router_after_layer_quantization(
    layer,
    layer_idx,
    decoder_inputs,
    scratch,
    layer_kwargs,
    teacher_targets,
    model_name,
    config,
):
    """Locally fine-tune the current router inside the layer-wise GPTQ loop."""
    if not config.is_router_only:
        raise ValueError("Interleaved router fine-tuning currently supports router-only losses.")
    _, router = get_router_module(layer, model_name)
    current_sample = {"index": None, "calls": 0}

    def capture_router_input(_module, inputs):
        sample_idx = current_sample["index"]
        if sample_idx is None:
            raise RuntimeError("Router input hook fired outside a calibration sample forward.")
        hidden = inputs[0].detach().reshape_as(scratch[sample_idx])
        scratch[sample_idx].copy_(hidden)
        current_sample["calls"] += 1

    handle = router.register_forward_pre_hook(capture_router_input)
    with torch.no_grad():
        for sample_idx in range(decoder_inputs.shape[0]):
            current_sample["index"] = sample_idx
            layer(decoder_inputs[sample_idx: sample_idx + 1], **layer_kwargs)
    current_sample["index"] = None
    handle.remove()
    if current_sample["calls"] != decoder_inputs.shape[0]:
        raise RuntimeError(
            f"Layer {layer_idx} router ran {current_sample['calls']} times for "
            f"{decoder_inputs.shape[0]} samples."
        )

    _train_router_from_inputs(
        router,
        scratch,
        teacher_targets.router_logits_for_layer(layer_idx),
        teacher_targets.attention_mask,
        model_name,
        config,
        layer_idx,
    )


def _finetune_after_all_router_only(model, dataloader, teacher_targets, args, config):
    original_training = model.training
    original_use_cache = model.config.use_cache
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    model.config.use_cache = False
    _set_requires_grad(model.parameters(), False)

    with torch.no_grad():
        inps, layer_kwargs = compute_decoder_inputs(model, dataloader, args.model_name, "cuda")
    layers = get_blocks(model, args.model_name)
    outs = torch.empty_like(inps)
    for layer_idx, layer in enumerate(layers):
        layer = layer.to("cuda")
        layers[layer_idx] = layer
        finetune_router_after_layer_quantization(
            layer,
            layer_idx,
            inps,
            outs,
            layer_kwargs,
            teacher_targets,
            args.model_name,
            config,
        )
        with torch.no_grad():
            for sample_idx in range(inps.shape[0]):
                output = layer(inps[sample_idx: sample_idx + 1], **layer_kwargs)
                outs[sample_idx] = _extract_layer_hidden(output)
        inps, outs = outs, inps
        layers[layer_idx] = layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    del inps, outs
    gc.collect()
    torch.cuda.empty_cache()
    for parameter, original in zip(model.parameters(), original_requires_grad):
        parameter.requires_grad = original
    model.config.use_cache = original_use_cache
    model.train(original_training)


def _finetune_after_all_with_output(model, teacher_targets, args, config):
    if teacher_targets.final_hidden_states is None:
        raise RuntimeError("Output KL requires cached teacher final hidden states.")

    original_training = model.training
    original_use_cache = model.config.use_cache
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    model.config.use_cache = False
    routers = [module for _, module in get_router_modules(model, args.model_name)]
    input_device = model.get_input_embeddings().weight.device
    head_device = next(model.lm_head.parameters()).device
    input_ids = teacher_targets.input_ids
    attention_mask = teacher_targets.attention_mask

    for layer_idx, router in enumerate(routers):
        _set_requires_grad(model.parameters(), False)
        router_parameters = list(router.parameters())
        _set_requires_grad(router_parameters, True)
        optimizer = torch.optim.AdamW(
            router_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        top_m = None
        if config.needs_router_targets:
            top_m = _router_top_m(
                config, args.model_name, teacher_targets.router_logits_for_layer(layer_idx)
            )

        for epoch in range(config.epochs):
            loss_sums = {"total": 0.0, "router": 0.0, "output": 0.0}
            step_count = 0
            start_time = time.time()
            for start in range(0, input_ids.shape[0], config.batch_size):
                end = min(input_ids.shape[0], start + config.batch_size)
                captured_router_logits = []
                handle = None
                if config.needs_router_targets:
                    handle = router.register_forward_hook(
                        lambda _module, _inputs, output: captured_router_logits.append(
                            extract_router_logits(output)
                        )
                    )

                data = input_ids[start:end].to(input_device)
                model_attention_mask = None
                if attention_mask is not None:
                    model_attention_mask = attention_mask[start:end].to(input_device)
                outputs = model(
                    input_ids=data,
                    attention_mask=model_attention_mask,
                    use_cache=False,
                )
                if handle is not None:
                    handle.remove()

                with torch.no_grad():
                    teacher_hidden = teacher_targets.final_hidden_states[start:end].to(head_device)
                    teacher_output_logits = model.lm_head(teacher_hidden)
                batch_token_mask = attention_mask[start:end] if attention_mask is not None else None
                output_loss = compute_causal_output_kl(
                    outputs.logits, teacher_output_logits, attention_mask=batch_token_mask
                )
                total_loss = config.output_kl_weight * output_loss

                router_loss = None
                if config.needs_router_targets:
                    if len(captured_router_logits) != 1:
                        raise RuntimeError(
                            f"Layer {layer_idx} router ran {len(captured_router_logits)} times in one batch."
                        )
                    teacher_router_logits = teacher_targets.router_logits_for_layer(layer_idx)[
                        start:end
                    ].to(captured_router_logits[0].device)
                    student_router_logits = captured_router_logits[0].reshape_as(teacher_router_logits)
                    router_loss = compute_router_loss(
                        config.router_loss,
                        student_router_logits,
                        teacher_router_logits,
                        top_m=top_m,
                        token_mask=batch_token_mask,
                    )
                    total_loss = total_loss + config.router_loss_weight * router_loss.to(
                        total_loss.device
                    )

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                loss_sums["total"] += total_loss.detach().item()
                loss_sums["output"] += output_loss.detach().item()
                if router_loss is not None:
                    loss_sums["router"] += router_loss.detach().item()
                step_count += 1

                del outputs, teacher_output_logits, total_loss
            print(
                f"[router layer {layer_idx:>2} | epoch {epoch:>2}] "
                f"total={loss_sums['total'] / max(step_count, 1):.6f}, "
                f"router={loss_sums['router'] / max(step_count, 1):.6f}, "
                f"output={loss_sums['output'] / max(step_count, 1):.6f}, "
                f"elapsed={time.time() - start_time:.2f}s"
            )
        optimizer.zero_grad(set_to_none=True)
        _set_requires_grad(router_parameters, False)
        gc.collect()
        torch.cuda.empty_cache()

    for parameter, original in zip(model.parameters(), original_requires_grad):
        parameter.requires_grad = original
    model.config.use_cache = original_use_cache
    model.train(original_training)


def finetune_routers_after_all_quantization(model, dataloader, teacher_targets, args, config):
    if config.is_router_only:
        _finetune_after_all_router_only(model, dataloader, teacher_targets, args, config)
    else:
        _finetune_after_all_with_output(model, teacher_targets, args, config)
