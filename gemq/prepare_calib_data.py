import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from gemq.utils.mixed_calib import (
    build_mixed_chat_en_calibration,
    load_prepared_calibration,
    load_streaming_sources,
    resolve_dataset_revisions,
    save_prepared_calibration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream, tokenize, and cache the English mixed Chat calibration dataset."
    )
    parser.add_argument("--model", required=True, help="Tokenizer model name or local path.")
    parser.add_argument("--output", required=True, help="Output .pt calibration cache.")
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--wildchat_revision", default="main")
    parser.add_argument("--ultrachat_revision", default="main")
    parser.add_argument("--fineweb_revision", default="main")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fast", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.suffix != ".pt":
        raise ValueError("--output must end in .pt")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.tokenizer_revision,
        use_fast=args.use_fast,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(f"Tokenizer {args.model!r} does not define a chat template.")
    if output_path.is_file() and not args.rebuild:
        _, metadata = load_prepared_calibration(
            output_path,
            expected_nsamples=args.nsamples,
            expected_seqlen=args.seqlen,
            expected_seed=args.seed,
            tokenizer=tokenizer,
        )
        print(f"Reusing prepared calibration data: {output_path}")
        print(json.dumps({
            "input_ids_sha256": metadata["input_ids_sha256"],
            "source_blocks": metadata["source_blocks"],
        }, indent=2))
        return

    requested_revisions = {
        "wildchat": args.wildchat_revision,
        "ultrachat": args.ultrachat_revision,
        "fineweb_edu": args.fineweb_revision,
    }
    resolved_revisions = resolve_dataset_revisions(requested_revisions)
    print("Resolved dataset revisions:")
    print(json.dumps(resolved_revisions, indent=2, sort_keys=True))
    sources = load_streaming_sources(args.seed, resolved_revisions)
    input_ids, metadata = build_mixed_chat_en_calibration(
        tokenizer=tokenizer,
        sources=sources,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        dataset_revisions=resolved_revisions,
    )
    tensor_path, metadata_path = save_prepared_calibration(input_ids, metadata, output_path)
    print(f"Saved calibration tensor:   {tensor_path}")
    print(f"Saved calibration metadata: {metadata_path}")
    print(json.dumps({
        "shape": list(input_ids.shape),
        "input_ids_sha256": metadata["input_ids_sha256"],
        "source_blocks": metadata["source_blocks"],
        "source_stats": metadata["source_stats"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
