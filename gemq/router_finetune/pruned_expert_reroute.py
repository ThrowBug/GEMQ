"""Teacher collection and quantization-aware rerouting for pruned Qwen3 experts."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from gemq.router_finetune.targets import (
    TeacherTargets,
    _build_cache_identity,
    _extract_layer_hidden,
    _torch_load,
)
from gemq.utils.model_utils import compute_decoder_inputs, get_blocks, get_moe_block


CACHE_VERSION = 1
EXPERT_FORWARD_BATCH_SIZE = 512


def _hash_json(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class RerouteTeacherStore:
    path: Path
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    num_layers: int

    def load_layer(self, layer_idx):
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(layer_idx)
        return _torch_load(self.path / f"layer_{layer_idx:03d}.pt")

    def as_distill_targets(self):
        path = self.path / "final_hidden_states.pt"
        if not path.is_file():
            raise RuntimeError("This rerouting cache has no output-distillation targets.")
        return TeacherTargets(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            router_logits=None,
            final_hidden_states=_torch_load(path)["final_hidden_states"],
            metadata={"reroute_teacher_cache": str(self.path)},
        )


def _pruned_by_layer(bit_config, num_layers):
    if set(bit_config) != set(range(num_layers)):
        raise ValueError("Rerouting bit allocation must cover every model layer.")
    result = []
    prune_counts = set()
    for layer_idx in range(num_layers):
        pruned = tuple(sorted(e for e, bit in bit_config[layer_idx].items() if bit == 0))
        if not pruned:
            raise ValueError(f"Layer {layer_idx} has no zero-bit expert for rerouting.")
        result.append(pruned)
        prune_counts.add(len(pruned))
    if len(prune_counts) != 1:
        raise ValueError("Physical pruning requires the same number of zero-bit experts per layer.")
    return result


def _routes(moe, hidden):
    logits = moe.gate(hidden.reshape(-1, hidden.shape[-1])).float()
    probabilities = F.softmax(logits, dim=-1)
    weights, indices = probabilities.topk(int(moe.top_k), dim=-1)
    if getattr(moe, "norm_topk_prob", False):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights


def _update_reservoir(reservoir, hidden, indices, weights, valid, expert_id, capacity, generator):
    matches = indices.eq(expert_id)
    selected = matches.any(dim=-1) & valid
    count = int(selected.sum().item())
    if count == 0:
        return
    new_inputs = hidden[selected.to(hidden.device)].detach().to("cpu")
    new_weights = (weights * matches).sum(dim=-1)[selected].detach().to("cpu")
    new_keys = torch.rand(count, generator=generator)
    if reservoir["keys"].numel():
        new_keys = torch.cat((reservoir["keys"], new_keys))
        new_inputs = torch.cat((reservoir["inputs"], new_inputs))
        new_weights = torch.cat((reservoir["weights"], new_weights))
    keep = new_keys.topk(min(capacity, new_keys.numel()), largest=False).indices
    reservoir["keys"] = new_keys[keep]
    reservoir["inputs"] = new_inputs[keep]
    reservoir["weights"] = new_weights[keep]
    reservoir["seen"] += count


@torch.no_grad()
def _expert_forward(expert, inputs, device):
    outputs = []
    for start in range(0, inputs.shape[0], EXPERT_FORWARD_BATCH_SIZE):
        batch = inputs[start:start + EXPERT_FORWARD_BATCH_SIZE].to(device, non_blocking=True)
        output = expert(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        outputs.append(output.detach().to("cpu"))
    return torch.cat(outputs) if outputs else torch.empty_like(inputs)


@torch.no_grad()
def _collect(model, dataloader, input_ids, attention_mask, bit_config, config, args, cache_path):
    original_training = model.training
    original_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = False
    try:
        inps, layer_kwargs = compute_decoder_inputs(model, dataloader, args.model_name, "cuda")
        if tuple(inps.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("Teacher decoder inputs and calibration tokens are misaligned.")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("Teacher attention mask and calibration tokens are misaligned.")
        outs = torch.empty_like(inps)
        layers = get_blocks(model, args.model_name)
        pruned_by_layer = _pruned_by_layer(bit_config, len(layers))
        valid_masks = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        capacity = config.screen_tokens_per_expert + config.cost_tokens_per_expert

        for layer_idx, layer in enumerate(layers):
            layer = layer.to("cuda")
            layers[layer_idx] = layer
            moe = get_moe_block(layer, args.model_name)
            if not hasattr(moe, "experts") or not hasattr(moe, "gate"):
                raise TypeError("Pruned-expert rerouting requires Qwen3 per-expert MLPs and a gate.")
            if set(bit_config[layer_idx]) != set(range(len(moe.experts))):
                raise ValueError(f"Layer {layer_idx} bit allocation does not match teacher experts.")
            reservoirs = {
                expert_id: {"keys": torch.empty(0), "inputs": None, "weights": None, "seen": 0}
                for expert_id in pruned_by_layer[layer_idx]
            }
            generators = {
                expert_id: torch.Generator().manual_seed(args.seed + layer_idx * 100003 + expert_id)
                for expert_id in reservoirs
            }
            route_indices = []
            route_weights = []
            current_sample = {"index": None, "calls": 0}

            def capture_moe_inputs(module, inputs):
                sample_idx = current_sample["index"]
                if sample_idx is None:
                    raise RuntimeError("MoE input hook fired outside teacher sample forward.")
                hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                indices, weights = _routes(module, hidden)
                indices = indices.detach().to("cpu")
                weights = weights.detach().to("cpu")
                route_indices.append(indices.to(torch.int16))
                route_weights.append(weights.to(torch.float16))
                valid = valid_masks[sample_idx].reshape(-1)
                for expert_id, reservoir in reservoirs.items():
                    _update_reservoir(
                        reservoir, hidden, indices, weights, valid, expert_id,
                        capacity, generators[expert_id],
                    )
                current_sample["calls"] += 1

            handle = moe.register_forward_pre_hook(capture_moe_inputs)
            try:
                for sample_idx in range(inps.shape[0]):
                    current_sample["index"] = sample_idx
                    output = layer(inps[sample_idx:sample_idx + 1], **layer_kwargs)
                    outs[sample_idx] = _extract_layer_hidden(output)
            finally:
                current_sample["index"] = None
                handle.remove()
            if current_sample["calls"] != input_ids.shape[0]:
                raise RuntimeError(f"Layer {layer_idx} MoE was not called once per sample.")

            samples = {}
            for expert_id, reservoir in reservoirs.items():
                available = reservoir["keys"].numel()
                if available:
                    order = reservoir["keys"].argsort()
                    selected_inputs = reservoir["inputs"][order]
                    selected_weights = reservoir["weights"][order]
                    references = _expert_forward(moe.experts[expert_id], selected_inputs, "cuda")
                    screen_count = min(
                        config.screen_tokens_per_expert,
                        max(1, round(available * config.screen_tokens_per_expert / capacity)),
                    )
                    if available > config.screen_tokens_per_expert:
                        screen_count = config.screen_tokens_per_expert
                else:
                    selected_inputs = torch.empty(0, inps.shape[-1], dtype=inps.dtype)
                    selected_weights = torch.empty(0, dtype=torch.float32)
                    references = torch.empty_like(selected_inputs)
                    screen_count = 0
                samples[expert_id] = {
                    "screen_inputs": selected_inputs[:screen_count],
                    "screen_reference": references[:screen_count],
                    "screen_weights": selected_weights[:screen_count],
                    "cost_inputs": selected_inputs[screen_count:],
                    "cost_reference": references[screen_count:],
                    "cost_weights": selected_weights[screen_count:],
                    "seen": reservoir["seen"],
                }

            _atomic_save(
                {
                    "route_indices": torch.stack(route_indices),
                    "route_weights": torch.stack(route_weights),
                    "samples": samples,
                },
                cache_path / f"layer_{layer_idx:03d}.pt",
            )
            inps, outs = outs, inps
            layers[layer_idx] = layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[reroute teacher] collected layer {layer_idx + 1}/{len(layers)}")

        if config.then_distill:
            norm = model.model.norm.to("cuda")
            chunks = []
            for start in range(0, inps.shape[0], config.batch_size):
                chunks.append(norm(inps[start:start + config.batch_size]).to("cpu"))
            _atomic_save(
                {"final_hidden_states": torch.cat(chunks)},
                cache_path / "final_hidden_states.pt",
            )
            model.model.norm = norm.to("cpu")
    finally:
        model.config.use_cache = original_use_cache
        model.train(original_training)


def get_or_collect_reroute_teacher(model, tokenizer, dataloader, input_ids, attention_mask,
                                   bit_config, args, config):
    layers = get_blocks(model, args.model_name)
    _pruned_by_layer(bit_config, len(layers))
    for layer_idx, layer in enumerate(layers):
        moe = get_moe_block(layer, args.model_name)
        if set(bit_config[layer_idx]) != set(range(len(moe.experts))):
            raise ValueError(f"Layer {layer_idx} allocation does not cover all original experts.")
        if sum(bit > 0 for bit in bit_config[layer_idx].values()) < int(moe.top_k):
            raise ValueError(f"Layer {layer_idx} retains fewer experts than its Top-K.")
    identity = {
        "format_version": CACHE_VERSION,
        "teacher": _build_cache_identity(model, tokenizer, input_ids, attention_mask, args),
        "bit_config": _hash_json(bit_config),
        "screen_tokens_per_expert": config.screen_tokens_per_expert,
        "cost_tokens_per_expert": config.cost_tokens_per_expert,
        "then_distill": config.then_distill,
        "seed": args.seed,
    }
    base = Path(config.teacher_cache_dir) / "pruned_expert_reroute_v1"
    cache_path = base / _hash_json(identity)[:24]
    if config.rebuild_teacher_cache and cache_path.exists():
        cache_path = base / f"{cache_path.name}-rebuild-{uuid.uuid4().hex[:8]}"
    success = cache_path / "_SUCCESS"
    if success.is_file():
        with (cache_path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("identity") != identity:
            raise ValueError(f"Rerouting teacher cache identity mismatch: {cache_path}")
        for layer_idx in range(len(layers)):
            if not (cache_path / f"layer_{layer_idx:03d}.pt").is_file():
                raise FileNotFoundError(f"Rerouting teacher cache is incomplete: {cache_path}")
        if config.then_distill and not (cache_path / "final_hidden_states.pt").is_file():
            raise FileNotFoundError(f"Rerouting output targets are missing: {cache_path}")
        print(f"Loaded rerouting teacher cache from: {cache_path}")
    else:
        if cache_path.exists():
            raise FileExistsError(
                f"Incomplete rerouting cache will not be overwritten: {cache_path}; "
                "use --rft_rebuild_teacher_cache to create a separate copy."
            )
        cache_path.mkdir(parents=True, exist_ok=False)
        started = time.time()
        _collect(model, dataloader, input_ids, attention_mask, bit_config, config, args, cache_path)
        with (cache_path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump({"identity": identity, "num_layers": len(layers)}, handle,
                      indent=2, ensure_ascii=False, sort_keys=True)
        success.write_text("complete\n", encoding="utf-8")
        print(f"Saved rerouting teacher cache to {cache_path} in {time.time() - started:.1f}s")
    return RerouteTeacherStore(cache_path, input_ids, attention_mask, len(layers))


def weighted_output_error(reference, predicted, weights):
    """Weighted global NMSE; avoids instability from nearly-zero individual outputs."""
    if reference.shape != predicted.shape or reference.shape[0] != weights.shape[0]:
        raise ValueError("Expert replacement outputs and routing weights have incompatible shapes.")
    if reference.shape[0] == 0:
        return float("inf")
    weight = weights.float().reshape(-1, 1)
    reference = reference.float()
    predicted = predicted.float()
    numerator = (weight * (reference - predicted).square()).sum()
    denominator = (weight * reference.square()).sum().clamp_min(1e-12)
    return float((numerator / denominator).item())


def transfer_weights(costs):
    values = torch.as_tensor(costs, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Replacement costs must be a finite nonempty vector.")
    centered = values - values.min()
    scaled = centered / values.std(unbiased=False).clamp_min(1e-12)
    return torch.softmax(-scaled, dim=0).to(torch.float32)


def build_sparse_targets(route_indices, route_weights, transfer, top_k):
    """Move deleted-expert route mass into surviving expert IDs, then apply Top-K."""
    if route_indices.shape != route_weights.shape:
        raise ValueError("Teacher route indices and weights must have matching shapes.")
    old_experts, new_experts = transfer.shape
    if not 0 < top_k <= new_experts:
        raise ValueError("Invalid Top-K for the surviving experts.")
    if int(route_indices.min()) < 0 or int(route_indices.max()) >= old_experts:
        raise ValueError("Teacher route contains an invalid original expert ID.")
    if not torch.allclose(transfer.sum(dim=-1), torch.ones(old_experts), atol=1e-4):
        raise ValueError("Every transfer row must preserve its routing mass.")
    target = torch.zeros(*route_indices.shape[:-1], new_experts, dtype=torch.float32)
    for slot in range(route_indices.shape[-1]):
        target += (
            transfer[route_indices[..., slot].long()]
            * route_weights[..., slot].float().unsqueeze(-1)
        )
    values, indices = target.topk(top_k, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices, values


@torch.no_grad()
def compute_layer_transfer(moe, layer_data, kept_old_ids, layer_idx):
    """Screen all survivors on one split, estimate final costs on the other split."""
    started = time.time()
    old_experts = max(max(kept_old_ids), *(layer_data["samples"].keys())) + 1
    new_experts = len(kept_old_ids)
    if len(moe.experts) != new_experts:
        raise ValueError(f"Layer {layer_idx} pruning map and student experts differ.")
    if len(set(kept_old_ids)) != new_experts or any(
        old_id in layer_data["samples"] for old_id in kept_old_ids
    ) or len(layer_data["samples"]) + new_experts != old_experts:
        raise ValueError(f"Layer {layer_idx} pruning map does not partition original experts.")
    device = next(moe.parameters()).device
    top_k = int(moe.top_k)
    transfer = torch.zeros(old_experts, new_experts, dtype=torch.float32)
    for new_id, old_id in enumerate(kept_old_ids):
        transfer[old_id, new_id] = 1.0

    samples = layer_data["samples"]
    screen_costs = {old_id: torch.full((new_experts,), float("inf")) for old_id in samples}
    # Group inputs by surviving expert: one expert forward can serve all pruned rows.
    screen_ids = [old_id for old_id, row in samples.items() if row["screen_inputs"].shape[0]]
    if screen_ids:
        inputs = torch.cat([samples[e]["screen_inputs"] for e in screen_ids])
        sizes = [samples[e]["screen_inputs"].shape[0] for e in screen_ids]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        for new_id, expert in enumerate(moe.experts):
            predicted = _expert_forward(expert, inputs, device)
            for position, old_id in enumerate(screen_ids):
                row = samples[old_id]
                start, end = offsets[position:position + 2]
                screen_costs[old_id][new_id] = weighted_output_error(
                    row["screen_reference"], predicted[start:end], row["screen_weights"]
                )

    candidate_map = {}
    for old_id, row in samples.items():
        if not row["screen_inputs"].shape[0]:
            candidate_map[old_id] = list(range(min(top_k, new_experts)))
        else:
            candidate_map[old_id] = screen_costs[old_id].topk(
                min(top_k, new_experts), largest=False
            ).indices.tolist()

    final_costs = {old_id: {} for old_id in samples}
    for new_id, expert in enumerate(moe.experts):
        selected_ids = [
            old_id for old_id, candidates in candidate_map.items()
            if new_id in candidates and samples[old_id]["cost_inputs"].shape[0]
        ]
        if not selected_ids:
            continue
        inputs = torch.cat([samples[e]["cost_inputs"] for e in selected_ids])
        predicted = _expert_forward(expert, inputs, device)
        offset = 0
        for old_id in selected_ids:
            row = samples[old_id]
            end = offset + row["cost_inputs"].shape[0]
            final_costs[old_id][new_id] = weighted_output_error(
                row["cost_reference"], predicted[offset:end], row["cost_weights"]
            )
            offset = end

    report = {"layer": layer_idx, "experts": {}, "elapsed_seconds": time.time() - started}
    for old_id, row in samples.items():
        candidates = candidate_map[old_id]
        if not row["cost_inputs"].shape[0]:
            # No calibration evidence: this row is never used by its teacher routes.
            transfer[old_id, candidates] = 1.0 / len(candidates)
            costs = []
        else:
            costs = [final_costs[old_id][new_id] for new_id in candidates]
            transfer[old_id, candidates] = transfer_weights(costs)
        zero_cost = (
            weighted_output_error(
                row["cost_reference"], torch.zeros_like(row["cost_reference"]),
                row["cost_weights"],
            ) if row["cost_inputs"].shape[0] else None
        )
        report["experts"][str(old_id)] = {
            "seen_tokens": row["seen"],
            "screen_tokens": int(row["screen_inputs"].shape[0]),
            "cost_tokens": int(row["cost_inputs"].shape[0]),
            "candidate_old_ids": [kept_old_ids[j] for j in candidates],
            "candidate_new_ids": candidates,
            "costs": costs,
            "transfer_weights": transfer[old_id, candidates].tolist(),
            "zero_output_cost": zero_cost,
            "weighted_cost": (
                sum(weight * cost for weight, cost in zip(transfer[old_id, candidates].tolist(), costs))
                if costs else None
            ),
        }
    if not torch.allclose(transfer.sum(dim=-1), torch.ones(old_experts), atol=1e-5):
        raise RuntimeError(f"Layer {layer_idx} transfer matrix does not preserve mass.")
    return transfer, report
