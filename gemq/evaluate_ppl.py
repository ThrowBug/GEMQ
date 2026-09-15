"""Evaluate a saved Hugging Face checkpoint with GEMQ's perplexity path."""

import argparse

from transformers import AutoTokenizer, logging

from gemq.utils.eval_utils import evaluate_perplexity
from gemq.utils.hf_loading import load_causal_lm_checkpoint
from gemq.utils.model_utils import (
    dispatch_model_to_all_devices,
    report_cuda_diagnostics,
)


logging.set_verbosity_error()


def _parse_datasets(value):
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--datasets must contain at least one dataset.")
    return datasets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GEMQ/Hugging Face model with the same PPL code as quantize.py."
    )
    parser.add_argument("--model", required=True, help="Saved Hugging Face checkpoint path")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model key used by GEMQ's model-specific offloading utilities",
    )
    parser.add_argument(
        "--model_dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
    )
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=_parse_datasets("wikitext2,c4"),
        help="Comma-separated datasets understood by gemq.utils.eval_utils",
    )
    parser.add_argument("--attn_impl", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--use_fast", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Evaluate one decoder layer at a time instead of dispatching the model across GPUs",
    )
    parser.add_argument("--cuda_diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if args.seqlen <= 1:
        parser.error("--seqlen must be greater than 1.")
    return args


def run(args):
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast,
        trust_remote_code=args.trust_remote_code,
    )

    print("Loading saved model ...")
    model = load_causal_lm_checkpoint(
        args.model,
        model_dtype=args.model_dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )
    model.seqlen = args.seqlen
    if args.cuda_diagnostics:
        report_cuda_diagnostics("saved model loaded", model=model)

    if not args.offload:
        model = dispatch_model_to_all_devices(model, args.cuda_diagnostics)

    evaluate_perplexity(
        model,
        tokenizer,
        args.datasets,
        args.model_name,
        offload=args.offload,
    )


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
