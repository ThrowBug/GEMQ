"""Generate reproducible EvalScope-style MATH-500 / GPQA-Diamond traces."""

import argparse
import csv
import json
import random
from pathlib import Path


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def load_evalscope_prompts(dataset, nsamples, seed, evalscope_seed, model, local_dataset, dataset_hub):
    """Let the installed EvalScope adapter select and format its own samples."""
    try:
        import evalscope
        from evalscope.api.registry import get_benchmark
        from evalscope.config import TaskConfig
        if dataset == "math_500":
            import evalscope.benchmarks.math_500.math_500_adapter  # noqa: F401
        else:
            import evalscope.benchmarks.gpqa.gpqa_adapter  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "This stage requires the same EvalScope installation/version used by eval_vllm.sh."
        ) from exc

    dataset_args = {dataset: {"few_shot_num": 0}}
    config = TaskConfig(
        model=model,
        datasets=[dataset],
        dataset_args=dataset_args,
        dataset_hub=dataset_hub,
        seed=evalscope_seed,
    )
    random.seed(evalscope_seed)  # GPQAAdapter._process_input uses the global RNG.
    adapter = get_benchmark(dataset, config)
    if adapter.few_shot_num != 0:
        raise ValueError("Only EvalScope zero-shot prompts are supported.")
    if local_dataset:
        local_path = Path(local_dataset)
        if dataset != "gpqa_diamond" or local_path.suffix.lower() != ".csv":
            raise ValueError("--local_dataset currently supports only a GPQA-Diamond CSV")
        with local_path.open(newline="", encoding="utf-8-sig") as source:
            samples = []
            for record in csv.DictReader(source):
                sample = adapter.record_to_sample(record)
                subset = sample.subset_key or "default"
                sample.input = adapter.process_sample_str_input(sample, subset)
                samples.append((subset, sample))
    else:
        data = adapter.load_dataset()
        samples = [(subset, sample) for subset in data.keys() for sample in data[subset]]
    if len(samples) < nsamples:
        raise ValueError(f"{dataset} has only {len(samples)} samples, need {nsamples}.")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(samples)), nsamples))
    chosen_set = set(chosen)
    remaining = [position for position in range(len(samples)) if position not in chosen_set]
    rng.shuffle(remaining)
    prompts = []
    for position in chosen + remaining:
        subset, sample = samples[position]
        messages = [
            {"role": str(message.role), "content": message.content}
            for message in sample.input
        ]
        if not all(isinstance(message["content"], str) for message in messages):
            raise ValueError("Only text-only EvalScope messages are supported.")
        prompts.append({
            "source_position": position,
            "subset": subset,
            "messages": messages,
            "question": messages[-1]["content"],
        })
    return prompts, {
        "evalscope_version": getattr(evalscope, "__version__", "unknown"),
        "dataset_id": str(Path(local_dataset).resolve()) if local_dataset else adapter.dataset_id,
        "dataset_hub": "local" if local_dataset else dataset_hub,
        "prompt_template": adapter.prompt_template,
        "system_prompt": adapter.system_prompt,
        "few_shot_num": adapter.few_shot_num,
        "evalscope_seed": evalscope_seed,
    }


def select_fitting_prompts(prompts, tokenizer, nsamples, max_seq_len):
    """Keep intact EvalScope prompts, replacing overlong ones deterministically."""
    selected, skipped = [], []
    for item in prompts:
        prompt_ids = tokenizer.apply_chat_template(
            item["messages"], tokenize=True, add_generation_prompt=True
        )
        if len(prompt_ids) >= max_seq_len:
            skipped.append({"source_position": item["source_position"], "prompt_tokens": len(prompt_ids)})
            continue
        selected.append({**item, "prompt_ids": prompt_ids})
        if len(selected) == nsamples:
            break
    if len(selected) < nsamples:
        raise ValueError(
            f"Only {len(selected)} complete prompts fit within {max_seq_len - 1} tokens; "
            f"need {nsamples}. Increase --max_seq_len or reduce --nsamples."
        )
    return selected, skipped


def existing_record_count(output_dir, metadata, prompts):
    """Validate a prior partial run before appending; never replace its records."""
    metadata_path = output_dir / "metadata.json"
    records_path = output_dir / "records.jsonl"
    if not metadata_path.is_file() or not records_path.is_file():
        raise FileExistsError(f"{output_dir} exists but is not a resumable generation directory")
    prior = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "format_version", "dataset", "model", "nsamples", "max_seq_len",
        "max_new_tokens", "seed", "dtype", "attn_impl", "local_dataset", "evalscope",
    ):
        if prior.get(key) != metadata.get(key):
            raise ValueError(f"Cannot resume: metadata field {key} differs in {output_dir}")
    count = 0
    with records_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            record = json.loads(line)
            if index >= len(prompts):
                raise ValueError(f"Too many records in {records_path}")
            if (
                record.get("id") != index
                or record.get("source_position") != prompts[index]["source_position"]
                or record.get("prompt_ids") != prompts[index]["prompt_ids"]
            ):
                raise ValueError(f"Record {index} differs from the deterministic prompt selection")
            count += 1
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math_500", "gpqa_diamond"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evalscope_seed", type=int, default=42,
                        help="EvalScope task seed for choice shuffling (its default is 42).")
    parser.add_argument("--dataset_hub", default="modelscope")
    parser.add_argument("--local_dataset", help="Optional local dataset CSV (e.g. authorized GPQA-Diamond).")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn_impl", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    if args.nsamples <= 0 or args.max_seq_len <= 1 or args.max_new_tokens <= 0:
        parser.error("nsamples, max_new_tokens must be positive and max_seq_len must exceed 1")
    if args.output_dir is None:
        args.output_dir = Path("cache/routing_analysis") / args.dataset / f"N{args.nsamples}-L{args.max_seq_len}-Seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a CUDA-capable PyTorch installation.")
    torch.manual_seed(args.seed)
    prompts, evalscope_info = load_evalscope_prompts(
        args.dataset, args.nsamples, args.seed, args.evalscope_seed,
        args.model, args.local_dataset, args.dataset_hub
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    prompts, skipped = select_fitting_prompts(prompts, tokenizer, args.nsamples, args.max_seq_len)
    metadata = {
        "format_version": 1,
        "dataset": args.dataset,
        "model": args.model,
        "nsamples": args.nsamples,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "generation": {"do_sample": False},
        "tokenizer_class": type(tokenizer).__name__,
        "evalscope": evalscope_info,
        "local_dataset": str(Path(args.local_dataset).resolve()) if args.local_dataset else None,
    }
    records_path = args.output_dir / "records.jsonl"
    if args.output_dir.exists():
        start = existing_record_count(args.output_dir, metadata, prompts)
        print(f"Resuming {records_path} at record {start}/{args.nsamples}", flush=True)
    else:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        records_path.touch(exist_ok=False)
        start = 0
    selection_path = args.output_dir / "selection.json"
    selection = {
        "policy": "skip intact prompts that leave no answer token; fill from seeded remaining samples",
        "source_positions": [item["source_position"] for item in prompts],
        "skipped": skipped,
    }
    if not selection_path.exists():
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    elif json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError(f"Cannot resume: prompt selection differs in {selection_path}")
    if start == args.nsamples:
        print(f"Already complete: {records_path}")
        return
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).eval()
    input_device = model.get_input_embeddings().weight.device
    with records_path.open("a", encoding="utf-8") as output:
        for index in range(start, len(prompts)):
            item = prompts[index]
            prompt_ids = item["prompt_ids"]
            allowance = min(args.max_new_tokens, args.max_seq_len - len(prompt_ids))
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=input_device)
            with torch.inference_mode():
                generated = model.generate(
                    prompt,
                    max_new_tokens=allowance,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0].tolist()
            answer_ids = generated[len(prompt_ids):]
            # The EOS token is predicted, but no generation forward is run on it.
            if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
                answer_ids.pop()
            record = {
                "id": index,
                "source_position": item["source_position"],
                "subset": item["subset"],
                "messages": item["messages"],
                "question": item["question"],
                "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "hit_length_limit": len(generated) >= args.max_seq_len,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{args.dataset} {index + 1}/{len(prompts)}] prompt={len(prompt_ids)} answer={len(answer_ids)}", flush=True)
    print(f"Saved {records_path}")


if __name__ == "__main__":
    main()
