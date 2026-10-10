"""Collect comparable per-sample EvalScope results for one benchmark at a time.

This is an offline reader: it never calls a model or re-scores an answer. Unknown
review formats and unverified sample identities are excluded from candidates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


METRICS = {
    "math_500": "acc",
    "gsm8k": "acc",
    "gpqa_diamond": "acc",
    "ifeval": "prompt_level_strict",
    "bfcl_v3": "acc",
    "mmlu_redux": "acc",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_file_parts(path, dataset):
    """Return (model directory, subset) for flat or per-subset EvalScope caches."""
    parts = path.parts
    for index, part in enumerate(parts):
        if part == dataset and index < len(parts) - 1:
            return "/".join(parts[:index]), Path(parts[-1]).stem
    stem = path.stem
    for separator in ("@", "_"):
        prefix = dataset + separator
        if stem.startswith(prefix):
            return "/".join(parts[:-1]), stem[len(prefix):]
    if stem == dataset:
        return "/".join(parts[:-1]), "default"
    return None


def find_cache_files(run_dir, kind, dataset):
    root = run_dir / kind
    if not root.is_dir():
        raise FileNotFoundError(f"EvalScope {kind} directory not found: {root}")
    found = []
    for path in sorted(root.rglob("*.jsonl")):
        parsed = dataset_file_parts(path.relative_to(root), dataset)
        if parsed is not None:
            found.append((path, *parsed))
    if not found:
        raise FileNotFoundError(f"No {dataset} JSONL files found in {root}")
    model_dirs = {model for _, model, _ in found}
    if len(model_dirs) != 1:
        raise ValueError(f"{root}: expected one model for {dataset}, found {sorted(model_dirs)}")
    return found


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def sample_id(row, kind):
    if row.get("index") is not None:
        return str(row["index"])
    if row.get("sample_id") is not None:
        return str(row["sample_id"])
    score = row.get("sample_score") or row.get("score") or {}
    if isinstance(score, dict) and score.get("sample_id") is not None:
        return str(score["sample_id"])
    metadata = row.get("metadata") or row.get("sample_metadata") or {}
    if isinstance(metadata, dict) and metadata.get("id") is not None:
        return str(metadata["id"])
    raise ValueError(f"{kind} record has no index/sample_id")


def read_cache(files, kind):
    records = {}
    for path, _, file_subset in files:
        for row in read_jsonl(path):
            subset = str(row.get("subset") or file_subset)
            key = (subset, sample_id(row, kind))
            if key in records:
                raise ValueError(f"Duplicate {kind} sample {key} in {path}; repeated generations are unsupported")
            records[key] = {"record": row, "source_file": str(path.resolve())}
    return records


def get_score_value(review, metric):
    values = score_evidence(review).get("value")
    if not isinstance(values, dict):
        return None
    value = values.get(metric)
    if value is None and metric == "acc":
        value = values.get("accuracy")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def score_evidence(review):
    container = review.get("sample_score", review)
    for _ in range(3):
        if not isinstance(container, dict):
            return {}
        if "value" in container:
            return container
        container = container.get("score")
    return {}


def prompt_and_identity(prediction, review):
    """Prefer the rendered prompt; fall back to substantive sample metadata."""
    context = {}
    for row in (review, prediction):
        metadata = row.get("sample_metadata") or row.get("metadata") or {}
        if isinstance(metadata, dict):
            for field in ("functions", "tools", "choices", "options"):
                if field in metadata and field not in context:
                    context[field] = metadata[field]
    for row in (review, prediction):
        value = row.get("input")
        if value:
            text = value if isinstance(value, str) else canonical(value)
            identity = canonical({"prompt": text, "context": context})
            return text, "input", hashlib.sha256(identity.encode("utf-8")).hexdigest()
    for row in (review, prediction):
        meta = row.get("sample_metadata") or row.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        for field in ("prompt", "question", "input"):
            if meta.get(field):
                value = meta[field]
                text = value if isinstance(value, str) else canonical(value)
                identity = canonical({"prompt": text, "context": context})
                return text, "metadata." + field, hashlib.sha256(identity.encode("utf-8")).hexdigest()
        substantive = {k: v for k, v in meta.items() if k not in ("id", "key", "index")}
        if substantive:
            text = canonical(substantive)
            return text, "metadata", hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "", "missing", None


def response_text(prediction, review):
    messages = prediction.get("messages") or []
    rendered = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if content:
                rendered.append(str(content))
            if message.get("tool_calls"):
                rendered.append(canonical(message["tool_calls"]))
    if rendered:
        return "\n".join(rendered)
    output = prediction.get("model_output") or {}
    if isinstance(output, dict):
        choices = output.get("choices") or []
        rendered = []
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                if message.get("content") is not None:
                    rendered.append(str(message["content"]))
                if message.get("tool_calls"):
                    rendered.append(canonical(message["tool_calls"]))
        if rendered:
            return "\n".join(rendered)
    for value in (prediction.get("output"), score_evidence(review).get("prediction")):
        if isinstance(value, str):
            return value
    return ""


def load_run(label, run_dir, dataset):
    predictions = read_cache(find_cache_files(run_dir, "predictions", dataset), "prediction")
    reviews = read_cache(find_cache_files(run_dir, "reviews", dataset), "review")
    joined = {}
    for key in predictions.keys() & reviews.keys():
        prediction = predictions[key]["record"]
        review = reviews[key]["record"]
        prompt, prompt_source, prompt_hash = prompt_and_identity(prediction, review)
        target = review.get("target", prediction.get("target"))
        joined[key] = {
            "prompt": prompt,
            "prompt_source": prompt_source,
            "prompt_hash": prompt_hash,
            "target": target,
            "response": response_text(prediction, review),
            "correct": get_score_value(review, METRICS[dataset]),
            "score": score_evidence(review),
            "prediction": prediction,
            "review": review,
            "prediction_file": predictions[key]["source_file"],
            "review_file": reviews[key]["source_file"],
        }
    return joined, {
        "run_dir": str(run_dir.resolve()),
        "config_files": [str(path.resolve()) for path in sorted((run_dir / "configs").glob("*.yaml"))]
        if (run_dir / "configs").is_dir() else [],
        "prediction_count": len(predictions),
        "review_count": len(reviews),
        "prediction_without_review": len(predictions.keys() - reviews.keys()),
        "review_without_prediction": len(reviews.keys() - predictions.keys()),
        "joined_count": len(joined),
    }


def collect(dataset, runs, ours, reference=None):
    loaded = {}
    report = {"dataset": dataset, "metric": METRICS[dataset], "runs": {}}
    for label, path in runs.items():
        loaded[label], report["runs"][label] = load_run(label, path, dataset)
    keys = set.intersection(*(set(rows) for rows in loaded.values()))
    if not keys:
        raise ValueError(f"No shared {dataset} sample IDs across all runs; inspect subsets and run directories")
    report["shared_sample_count"] = len(keys)
    report["configuration_note"] = (
        "Compare the listed EvalScope configs for dataset version, subset, seed, "
        "prompt settings and generation parameters before publishing examples."
    )
    report["missing_across_models"] = {
        label: len(set.union(*(set(rows) for rows in loaded.values())) - set(rows))
        for label, rows in loaded.items()
    }
    counts = {"prompt_mismatch": 0, "prompt_unverified": 0, "target_mismatch": 0,
              "unknown_score": 0, "candidate": 0}
    samples = []
    baselines = [label for label in runs if label not in (ours, reference)]
    def sort_key(key):
        return key[0], 0 if key[1].isdigit() else 1, int(key[1]) if key[1].isdigit() else key[1]

    for subset, index in sorted(keys, key=sort_key):
        key = (subset, index)
        per_model = {label: loaded[label][key] for label in runs}
        hashes = {row["prompt_hash"] for row in per_model.values()}
        targets = {canonical(row["target"]) for row in per_model.values() if row["target"] is not None}
        problems = []
        if None in hashes:
            problems.append("prompt_unverified")
        elif len(hashes) != 1:
            problems.append("prompt_mismatch")
        if len(targets) > 1:
            problems.append("target_mismatch")
        if any(row["correct"] is None for row in per_model.values()):
            problems.append("unknown_score")
        for problem in problems:
            counts[problem] += 1
        failed_baselines = [label for label in baselines if per_model[label]["correct"] is False]
        candidate = (not problems and per_model[ours]["correct"] is True
                     and bool(failed_baselines)
                     and (reference is None or per_model[reference]["correct"] is True))
        if candidate:
            counts["candidate"] += 1
        anchor = per_model[ours]
        samples.append({
            "dataset": dataset, "subset": subset, "sample_id": index,
            "metric": METRICS[dataset], "prompt": anchor["prompt"],
            "prompt_source": anchor["prompt_source"], "target": anchor["target"],
            "problems": problems, "candidate": candidate,
            "failed_baselines": failed_baselines, "models": per_model,
        })
    report["counts"] = counts
    return samples, report


def preview_text(value, limit):
    text = value if isinstance(value, str) else canonical(value)
    return text if len(text) <= limit else text[:limit] + "\n[preview truncated; see all_samples.jsonl]"


def write_outputs(output_dir, dataset, samples, report, labels, preview_limit, preview_chars,
                  overwrite=False):
    output_names = ("all_samples.jsonl", "candidates.csv", "candidates.md", "alignment_report.json")
    existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {existing}; choose another --output_dir or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_samples.jsonl").open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    candidates = [sample for sample in samples if sample["candidate"]]
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8-sig") as output:
        fields = ["dataset", "subset", "sample_id", "failed_baselines", "prompt_preview"]
        fields += [f"{label}_correct" for label in labels]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample in candidates:
            writer.writerow({
                "dataset": dataset, "subset": sample["subset"], "sample_id": sample["sample_id"],
                "failed_baselines": ",".join(sample["failed_baselines"]),
                "prompt_preview": preview_text(sample["prompt"], 200),
                **{f"{label}_correct": sample["models"][label]["correct"] for label in labels},
            })
    lines = [f"# {dataset} Dialog Example Candidates", "",
             "Automatically filtered from EvalScope reviews; inspect original JSONL before publication.", ""]
    for sample in candidates[:preview_limit]:
        lines += [f"## {sample['subset']} / {sample['sample_id']}", "",
                  f"Failed baselines: {', '.join(sample['failed_baselines'])}", "",
                  "Prompt:", "", "~~~~text", preview_text(sample["prompt"], preview_chars), "~~~~", "",
                  "Reference target:", "", "~~~~text",
                  preview_text(sample["target"], preview_chars), "~~~~", ""]
        for label in labels:
            model = sample["models"][label]
            lines += [f"### {label} (correct={model['correct']})", "", "~~~~text",
                      preview_text(model["response"], preview_chars), "~~~~", ""]
    (output_dir / "candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(METRICS))
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR",
                        help="Repeat for each model; RUN_DIR is the actual EvalScope timestamp directory")
    parser.add_argument("--ours", required=True, help="Label of the method being showcased")
    parser.add_argument("--reference", help="Optional FP label; candidates must also pass for this model")
    parser.add_argument("--output_dir", type=Path, help="Default: cache/dialog_examples/DATASET")
    parser.add_argument("--preview_limit", type=int, default=30)
    parser.add_argument("--preview_chars", type=int, default=2500)
    parser.add_argument("--overwrite", action="store_true", help="Replace this tool's existing output files")
    args = parser.parse_args(argv)
    runs = {}
    for item in args.run:
        if "=" not in item:
            parser.error(f"--run must be LABEL=RUN_DIR: {item}")
        label, path = item.split("=", 1)
        if not label or not path or not all(ch.isalnum() or ch in "_-" for ch in label):
            parser.error(f"Invalid --run value: {item}")
        if label in runs:
            parser.error(f"Duplicate run label: {label}")
        runs[label] = Path(path)
    if args.ours not in runs or (args.reference and args.reference not in runs):
        parser.error("--ours and --reference must name labels supplied by --run")
    if not any(label not in (args.ours, args.reference) for label in runs):
        parser.error("At least one baseline --run is required")
    if args.preview_limit < 0 or args.preview_chars <= 0:
        parser.error("--preview_limit must be nonnegative and --preview_chars positive")
    for label, path in runs.items():
        if not path.is_dir():
            parser.error(f"Run directory for {label} does not exist: {path}")
    args.runs = runs
    if args.output_dir is None:
        args.output_dir = Path("cache/dialog_examples") / args.dataset
    return args


def main(argv=None):
    args = parse_args(argv)
    samples, report = collect(args.dataset, args.runs, args.ours, args.reference)
    write_outputs(args.output_dir, args.dataset, samples, report, args.runs,
                  args.preview_limit, args.preview_chars, args.overwrite)
    print(f"Saved {len(samples)} aligned samples and {report['counts']['candidate']} candidates "
          f"to {args.output_dir}")
    print(f"Check {args.output_dir / 'alignment_report.json'} before using the candidates.")


if __name__ == "__main__":
    main()
