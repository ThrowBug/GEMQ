"""Qwen3 search-only GPTQ variants and full-model JSD scoring."""

from contextlib import ExitStack
from pathlib import Path

import torch
from accelerate import infer_auto_device_map
from tqdm import tqdm

from proquant.expert_costs import (
    _capture_decoder_inputs,
    _capture_qwen3_moe_inputs,
    _collect_active_inputs,
    _expert_linears,
)
from proquant.quantizers.gptq import GPTQWeightQuantizer
from proquant.utils.model_utils import get_blocks, get_moe_block


def _new_quantizer(linear, bit):
    # Match the release Qwen3 quantize.sh quantizer settings.
    groupsize = 128 if linear.in_features % 128 == 0 else 64
    return GPTQWeightQuantizer(linear.weight, nbits=bit, groupsize=groupsize, mse=True)


def _pack_variant(quantizer):
    codes, scales, zeros = quantizer.quantize()
    codes = codes.to(torch.uint8)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return packed.cpu(), scales.cpu(), zeros.cpu()


@torch.no_grad()
def _install_variant(linear, variant):
    packed, scales, zeros = (tensor.to(linear.weight.device) for tensor in variant)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    weight = (codes.to(scales.dtype) - zeros) * scales
    linear.weight.copy_(weight.reshape_as(linear.weight))


def _attention_linears(layer):
    attention = layer.self_attn
    return attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj


def _capture_hessian(stack, linear, quantizer):
    hook = linear.register_forward_pre_hook(
        lambda _module, inputs: quantizer.add_batch(inputs[0])
    )
    stack.callback(hook.remove)


def _expert_variants(expert, active_inputs, device):
    linears = _expert_linears(expert)
    quantizers = [_new_quantizer(linear, 2) for linear in linears]
    if active_inputs is None:
        # An identity Hessian gives a weight-only fallback for unseen experts.
        for quantizer in quantizers:
            quantizer.H.diagonal().fill_(1)
    else:
        with ExitStack() as hooks:
            for linear, quantizer in zip(linears, quantizers):
                _capture_hessian(hooks, linear, quantizer)
            for batch in active_inputs.split(1024):
                expert(batch.to(device))

    variants = {2: [], 3: []}
    for linear, quantizer in zip(linears, quantizers):
        three_bit = _new_quantizer(linear, 3)
        three_bit.H.copy_(quantizer.H)
        variants[2].append(_pack_variant(quantizer))
        variants[3].append(_pack_variant(three_bit))
    return variants


@torch.no_grad()
def prepare_variants(
    model, model_name, batches, search_nsamples, vocab_top_k, attention_bits, cache, device="cuda:0",
):
    """Use one fixed full-precision calibration pass for both bit variants."""
    device = torch.device(device)
    layers = get_blocks(model, model_name)
    hidden, positional, keyword = _capture_decoder_inputs(model, batches, model_name, device)
    unseen = 0
    for index, layer in enumerate(tqdm(layers, desc="Caching GPTQ search variants")):
        layer.to(device)
        moe = get_moe_block(layer, model_name)
        attention = [_new_quantizer(linear, attention_bits)
                     for linear in _attention_linears(layer)] if attention_bits < 16 else []
        with ExitStack() as hooks:
            for linear, quantizer in zip(_attention_linears(layer), attention):
                _capture_hessian(hooks, linear, quantizer)
            inputs, selected, gates, hidden = _capture_qwen3_moe_inputs(
                layer, moe, hidden, positional, keyword, device, stop_at_moe=False,
            )
        experts = []
        for expert_id, expert in enumerate(moe.experts):
            active, _ = _collect_active_inputs(expert_id, inputs, selected, gates)
            unseen += active is None
            experts.append(_expert_variants(expert, active, device))
        torch.save(
            {"experts": experts, "attention": [_pack_variant(q) for q in attention]},
            cache / f"layer_{index}.pt",
        )
        layer.to("cpu")
        del attention

    # These logits come from the same unmodified full-precision reference pass.
    norm, head = model.model.norm.to(device), model.lm_head.to(device)
    targets = []
    for states in hidden[:search_nsamples]:
        indices, probabilities = [], []
        for tokens in states[0, :-1].split(128):
            logits = head(norm(tokens.to(device))).float()
            values, ids = logits.topk(min(vocab_top_k, logits.shape[-1]), dim=-1)
            indices.append(ids.cpu())
            probabilities.append(_with_tail(logits, values).cpu())
        targets.append((torch.cat(indices), torch.cat(probabilities)))
    model.model.norm.to("cpu")
    head.to("cpu")
    torch.cuda.empty_cache()
    return targets, unseen


def _with_tail(logits, selected_logits):
    probabilities = (selected_logits - logits.logsumexp(dim=-1, keepdim=True)).exp()
    tail = (1 - probabilities.sum(dim=-1, keepdim=True)).clamp_min(0)
    probabilities = torch.cat((probabilities, tail), dim=-1)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _jsd(reference, candidate):
    middle = (reference + candidate) * 0.5
    return 0.5 * (
        torch.special.xlogy(reference, reference) - torch.special.xlogy(reference, middle)
        + torch.special.xlogy(candidate, candidate) - torch.special.xlogy(candidate, middle)
    )


def scoring_device_map(model):
    memory = {
        device: max(0, torch.cuda.mem_get_info(device)[0] - (1 << 30))
        for device in range(torch.cuda.device_count())
    }
    device_map = infer_auto_device_map(
        model, max_memory=memory, no_split_module_classes=["Qwen3MoeDecoderLayer"],
    )
    if any(device in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(
            "Qwen3 search requires enough aggregate CUDA memory for the BF16 scoring model."
        )
    return device_map


class Qwen3Scorer:
    """Keep one assembled model resident and switch expert weights by layer."""

    def __init__(self, model, model_name, cache, batches, targets):
        self.model = model
        self.input_device = self.model.get_input_embeddings().weight.device
        self.batches = batches
        self.targets = targets
        self.layers = [get_moe_block(layer, model_name) for layer in get_blocks(model, model_name)]
        self.handles = []
        self.configs = [None] * len(self.layers)
        self.masks = [None] * len(self.layers)
        self.experts = []
        for index, layer in enumerate(get_blocks(model, model_name)):
            variants = torch.load(
                Path(cache) / f"layer_{index}.pt", map_location="cpu", mmap=True, weights_only=True,
            )
            self.experts.append(variants["experts"])
            for linear, variant in zip(_attention_linears(layer), variants["attention"]):
                _install_variant(linear, variant)
            self.handles.append(self.layers[index].gate.register_forward_hook(
                self._mask_hook(index)
            ))

    def _mask_hook(self, index):
        def apply_mask(_gate, _inputs, logits):
            return logits.masked_fill(self.masks[index].to(logits.device), float("-inf"))
        return apply_mask

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        for handle in self.handles:
            handle.remove()
        self.experts.clear()

    @torch.no_grad()
    def install(self, layer, bits):
        moe = self.layers[layer]
        old_bits = self.configs[layer]
        if old_bits == bits:
            return
        for index, (expert, bit) in enumerate(zip(moe.experts, bits)):
            if bit and (old_bits is None or bit != old_bits[index]):
                for linear, variant in zip(_expert_linears(expert), self.experts[layer][index][bit]):
                    _install_variant(linear, variant)
        self.masks[layer] = torch.tensor(
            [bit == 0 for bit in bits], device=moe.gate.weight.device,
        )
        self.configs[layer] = bits

    @torch.no_grad()
    def score(self):
        total = 0.0
        count = 0
        for batch, (indices, reference) in zip(self.batches, self.targets):
            scores = self.model(
                input_ids=batch[0].to(self.input_device), use_cache=False,
            ).logits[0, :-1]
            contributions = []
            for logits, ids, ref in zip(scores.split(128), indices.split(128), reference.split(128)):
                logits = logits.float()
                values = logits.gather(-1, ids.to(logits.device))
                candidate = _with_tail(logits, values)
                contributions.append(_jsd(ref.to(logits.device), candidate).sum())
            total += torch.stack(contributions).sum().item()
            count += len(indices)
        value = total / count
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError("Non-finite model JSD during lazy greedy allocation.")
        return value
