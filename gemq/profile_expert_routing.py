"""Measure full-checkpoint Qwen3-MoE expert routing on C4 or saved QA traces."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def parse_layers(spec, total_layers):
    if spec == "all":
        return list(range(total_layers))
    layers = [int(value) for value in spec.split(",")]
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= total_layers for i in layers):
        raise ValueError(f"--layers must contain distinct indices in [0, {total_layers - 1}]")
    return layers


def routes_from_logits(logits, top_k):
    """Return Qwen3's top-k IDs and explicitly renormalized top-k mass."""
    import torch

    scores = torch.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1, dtype=torch.float32)
    mass, indices = torch.topk(scores, k=top_k, dim=-1)
    mass = mass / mass.sum(dim=-1, keepdim=True)
    return indices, mass


def scope_statistics(indices, mass, token_slice, n_experts):
    """Accumulate selected counts and selected, renormalized probabilities."""
    import torch

    indices = indices[token_slice].reshape(-1)
    mass = mass[token_slice].reshape(-1)
    counts = torch.bincount(indices, minlength=n_experts).to("cpu", dtype=torch.int64)
    sums = torch.zeros(n_experts, device=mass.device, dtype=torch.float64)
    sums.scatter_add_(0, indices, mass.to(torch.float64))
    return counts.numpy(), sums.cpu().numpy()


def load_sequences(args, tokenizer):
    if args.dataset == "c4":
        from gemq.utils.data_utils import get_calib_loader

        calib_args = SimpleNamespace(
            calib_dataset="c4", model=args.model, nsamples=args.nsamples,
            seqlen=args.seqlen, seed=args.seed, batch_size=1, use_fast=True,
        )
        loader = get_calib_loader(tokenizer, calib_args)
        for index, (input_ids, _) in enumerate(loader):
            ids = input_ids[0].tolist()
            if len(ids) != args.seqlen:
                raise ValueError(f"C4 block {index} has {len(ids)} tokens, expected {args.seqlen}")
            yield index, ids, len(ids)
    else:
        if args.records is None:
            raise ValueError("--records is required for QA datasets")
        generation_metadata = args.records.with_name("metadata.json")
        if not generation_metadata.is_file():
            raise FileNotFoundError(f"Missing generation metadata: {generation_metadata}")
        metadata = json.loads(generation_metadata.read_text(encoding="utf-8"))
        for key, expected in (
            ("dataset", args.dataset), ("model", args.model),
            ("nsamples", args.nsamples), ("max_seq_len", args.seqlen),
        ):
            if metadata.get(key) != expected:
                raise ValueError(f"Generation metadata {key}={metadata.get(key)!r}, expected {expected!r}")
        with args.records.open(encoding="utf-8") as source:
            for index, line in enumerate(source):
                record = json.loads(line)
                if record.get("id") != index:
                    raise ValueError(f"Record ID at line {index + 1} does not match its position")
                prompt_ids, answer_ids = record["prompt_ids"], record["answer_ids"]
                ids = prompt_ids + answer_ids
                if not ids or len(ids) > args.seqlen:
                    raise ValueError(f"Record {index} has invalid length {len(ids)}")
                yield index, ids, len(prompt_ids)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("c4", "math_500", "gpqa_diamond"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--records", type=Path, help="records.jsonl from generate_routing_qa")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="all", help="all or comma-separated zero-based layer indices")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn_impl", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    if args.nsamples <= 0 or args.seqlen <= 0:
        parser.error("nsamples and seqlen must be positive")
    if args.dataset != "c4" and args.records is None:
        parser.error("--records is required for math_500 and gpqa_diamond")
    if args.output_dir is None:
        args.output_dir = Path("cache/routing_analysis/stats") / args.dataset / f"N{args.nsamples}-L{args.seqlen}-Seed{args.seed}"
    return args


def write_aggregate(output_dir, dataset, layer_ids, scopes, all_counts, all_sums, token_totals):
    import numpy as np

    counts = np.stack(all_counts)  # sample, scope, layer, expert
    sums = np.stack(all_sums)
    np.savez_compressed(output_dir / "per_sample.npz", counts=counts, probability_sums=sums)
    with (output_dir / "routing_stats.csv").open("x", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "dataset", "scope", "layer", "expert", "tokens", "activation_count",
            "normalized_probability_sum", "mean_probability_when_active",
        ])
        for scope_index, scope in enumerate(scopes):
            for layer_index, layer in enumerate(layer_ids):
                for expert in range(counts.shape[-1]):
                    count = int(counts[:, scope_index, layer_index, expert].sum())
                    probability_sum = float(sums[:, scope_index, layer_index, expert].sum())
                    writer.writerow([
                        dataset, scope, layer, expert, token_totals[scope], count,
                        f"{probability_sum:.10f}", f"{probability_sum / count:.10f}" if count else "",
                    ])


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Full-checkpoint routing collection requires CUDA.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    if not hasattr(model, "layers"):
        raise TypeError("Expected a Qwen3MoeModel with model.layers")
    layer_ids = parse_layers(args.layers, len(model.layers))
    gates = {}
    for layer_id in layer_ids:
        moe = model.layers[layer_id].mlp
        if not hasattr(moe, "gate") or not hasattr(moe, "top_k"):
            raise TypeError(f"Layer {layer_id} does not contain a Qwen3-MoE router")
        gates[layer_id] = moe
    n_experts = gates[layer_ids[0]].gate.out_features
    top_k = gates[layer_ids[0]].top_k
    if any(moe.gate.out_features != n_experts or moe.top_k != top_k for moe in gates.values()):
        raise ValueError("Selected layers do not share expert count and top-k")
    scopes = ["combined"] if args.dataset == "c4" else ["combined", "prompt", "answer"]
    token_totals = {scope: 0 for scope in scopes}
    all_counts, all_sums = [], []
    current = {}
    handles = []

    def make_hook(layer_id):
        def hook(_module, _inputs, logits):
            if layer_id in current:
                raise RuntimeError(f"Router layer {layer_id} was called more than once")
            current[layer_id] = routes_from_logits(logits, top_k)
        return hook

    for layer_id, moe in gates.items():
        handles.append(moe.gate.register_forward_hook(make_hook(layer_id)))
    input_device = model.get_input_embeddings().weight.device
    try:
        for index, ids, prompt_length in load_sequences(args, tokenizer):
            if index >= args.nsamples:
                break
            current.clear()
            input_ids = torch.tensor([ids], dtype=torch.long, device=input_device)
            with torch.inference_mode():
                model(input_ids=input_ids, use_cache=False)
            if set(current) != set(layer_ids):
                raise RuntimeError(f"Sample {index}: missing router calls {set(layer_ids) - set(current)}")
            spans = {"combined": slice(None)}
            if args.dataset != "c4":
                spans.update(prompt=slice(0, prompt_length), answer=slice(prompt_length, None))
            scope_counts, scope_sums = [], []
            for scope in scopes:
                scope_tokens = len(ids) if scope == "combined" else (
                    prompt_length if scope == "prompt" else len(ids) - prompt_length
                )
                token_totals[scope] += scope_tokens
                layer_counts, layer_sums = [], []
                for layer_id in layer_ids:
                    indices, mass = current[layer_id]
                    if indices.shape[0] != len(ids):
                        raise RuntimeError(f"Layer {layer_id} returned {indices.shape[0]} routes for {len(ids)} tokens")
                    counts, sums = scope_statistics(indices, mass, spans[scope], n_experts)
                    if counts.sum() != top_k * scope_tokens or not np.isclose(sums.sum(), scope_tokens, atol=1e-3):
                        raise RuntimeError(f"Routing invariant failed at sample {index}, layer {layer_id}, {scope}")
                    layer_counts.append(counts)
                    layer_sums.append(sums)
                scope_counts.append(layer_counts)
                scope_sums.append(layer_sums)
            all_counts.append(np.asarray(scope_counts))
            all_sums.append(np.asarray(scope_sums))
            current.clear()
            print(f"[{args.dataset} {index + 1}/{args.nsamples}] tokens={len(ids)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(all_counts) != args.nsamples:
        raise ValueError(f"Found {len(all_counts)} samples, expected {args.nsamples}")
    args.output_dir.mkdir(parents=True)
    write_aggregate(args.output_dir, args.dataset, layer_ids, scopes, all_counts, all_sums, token_totals)
    metadata = {
        "format_version": 1,
        "dataset": args.dataset,
        "model": args.model,
        "records": str(args.records.resolve()) if args.records else None,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "layers": layer_ids,
        "n_experts": n_experts,
        "top_k": top_k,
        "scopes": scopes,
        "token_totals": token_totals,
        "mass_definition": "topk(softmax(router_logits)) / sum(topk(softmax(router_logits)))",
        "c4_source": "gemq.utils.data_utils.get_calib_loader" if args.dataset == "c4" else None,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {args.output_dir / 'routing_stats.csv'}")


if __name__ == "__main__":
    main()
