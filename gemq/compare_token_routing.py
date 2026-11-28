"""Compare per-token expert selection against one Qwen3-MoE checkpoint."""

import argparse
import csv
import gc
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from gemq.profile_expert_routing import MODEL, load_sequences, parse_layers, routes_from_logits


@dataclass(frozen=True)
class RoutingLayout:
    layers: tuple[int, ...]
    num_experts: int
    top_k: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before_model_path", type=Path, required=True)
    parser.add_argument("--after_model_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="One label per after-model, in order")
    parser.add_argument("--model", default=MODEL, help="Original tokenizer and QA records model ID")
    parser.add_argument("--dataset", choices=("c4", "math_500", "gpqa_diamond"), required=True)
    parser.add_argument("--records", type=Path, help="Existing records.jsonl for QA datasets")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="all", help="all or comma-separated zero-based layer indices")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn_impl", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.nsamples <= 0 or args.seqlen <= 0:
        parser.error("--nsamples and --seqlen must be positive")
    if args.dataset != "c4" and args.records is None:
        parser.error("--records is required for QA datasets")
    for path in (args.before_model_path, *args.after_model_paths):
        if not path.is_dir():
            parser.error(f"Checkpoint directory not found: {path}")
    if args.labels is None:
        args.labels = [path.resolve().name for path in args.after_model_paths]
    if len(args.labels) != len(args.after_model_paths):
        parser.error("Provide one --labels entry per --after_model_paths entry")
    if any(not label.strip() for label in args.labels) or len(set(args.labels)) != len(args.labels):
        parser.error("Model labels must be nonempty and unique")
    return args


def layer_overlap_counts(before, after):
    """Return the number of matching expert selections per layer for one sample."""
    if before.ndim != 3 or before.shape != after.shape:
        raise ValueError("Routes must have matching [layer, token, top_k] shapes")
    same_expert = before[..., :, None] == after[..., None, :]
    return same_expert.any(axis=-1).sum(axis=(1, 2))


def profile_model(path, sequences, args, consume, expected_layout=None):
    import numpy as np
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(path),
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    if getattr(model.config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"{path}: expected a Qwen3-MoE checkpoint")

    layer_ids = tuple(parse_layers(args.layers, len(model.layers)))
    moes = [model.layers[layer_id].mlp for layer_id in layer_ids]
    gate_specs = [
        (moe.gate.weight.shape[0], getattr(moe.gate, "top_k", getattr(moe, "top_k", None)))
        for moe in moes
    ]
    num_experts, top_k = gate_specs[0]
    if top_k is None or any(spec != gate_specs[0] for spec in gate_specs):
        raise ValueError(f"{path}: selected layers have different expert counts or Top-K values")
    layout = RoutingLayout(layer_ids, num_experts, top_k)
    if expected_layout is not None and layout != expected_layout:
        raise ValueError(f"{path}: layer IDs, expert count or Top-K differ from the reference")

    current = {}
    # Qwen3 has 128 experts, so one byte is enough for each stored expert ID.
    id_dtype = torch.uint8 if layout.num_experts <= 256 else torch.int32

    def make_hook(layer_id):
        def hook(_module, _inputs, output):
            if layer_id in current:
                raise RuntimeError(f"Layer {layer_id}: router called more than once")
            if isinstance(output, tuple) and len(output) == 3:
                indices = output[2].reshape(-1, layout.top_k)
            else:
                indices, _ = routes_from_logits(output, layout.top_k)
            current[layer_id] = indices.to(dtype=id_dtype)
        return hook

    handles = []
    input_device = model.get_input_embeddings().weight.device
    try:
        for layer_id, moe in zip(layer_ids, moes):
            handles.append(moe.gate.register_forward_hook(make_hook(layer_id)))
        with torch.inference_mode():
            for index, ids in enumerate(sequences):
                current.clear()
                input_ids = torch.tensor([ids], dtype=torch.long, device=input_device)
                model(input_ids=input_ids, use_cache=False)
                for layer_id in layer_ids:
                    if layer_id not in current or current[layer_id].shape != (len(ids), layout.top_k):
                        raise RuntimeError(f"{path}, sample {index}, layer {layer_id}: invalid routing output")
                routes = np.stack([current[layer_id].cpu().numpy() for layer_id in layer_ids])
                consume(index, routes)
                if (index + 1) % 16 == 0 or index + 1 == len(sequences):
                    print(f"  {index + 1}/{len(sequences)} samples", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return layout


def compare_models(args, sequences):
    import numpy as np
    import torch

    def release_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reference = []

    def remember(_index, routes):
        reference.append(routes)

    print(f"Reference: {args.before_model_path}", flush=True)
    layout = profile_model(args.before_model_path, sequences, args, remember)
    release_memory()
    denominator = sum(map(len, sequences)) * layout.top_k
    rows = []
    for label, path in zip(args.labels, args.after_model_paths):
        print(f"Comparing {label}: {path}", flush=True)
        counts = np.zeros(len(layout.layers), dtype=np.int64)

        def accumulate(index, routes):
            counts[:] += layer_overlap_counts(reference[index], routes)

        profile_model(path, sequences, args, accumulate, expected_layout=layout)
        release_memory()
        rows.append([label, *(f"{value:.6f}" for value in 100.0 * counts / denominator)])
    return layout.layers, rows


def main(argv=None):
    args = parse_args(argv)
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Expert routing collection requires CUDA.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    sequences = [ids for _, ids, _ in islice(load_sequences(args, tokenizer), args.nsamples)]
    if len(sequences) != args.nsamples:
        raise ValueError(f"Found {len(sequences)} samples, expected {args.nsamples}")

    layers, rows = compare_models(args, sequences)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["model_label", *layers])
        writer.writerows(rows)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
