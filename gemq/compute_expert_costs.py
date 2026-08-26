"""CLI for collecting REAP-style expert quantization costs."""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from gemq.expert_costs import compute_qwen3_expert_costs
from gemq.utils.data_utils import get_calib_loader
from gemq.utils.expert_cost_utils import impute_zero_frequency_costs
from gemq.utils.model_utils import ModelType, NAME_TO_MODEL


logging.set_verbosity_error()


def _parse_candidate_bits(value):
    pieces = value.split(",")
    if not pieces or any(not piece or not piece.isdigit() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "candidate bits must be comma-separated non-negative integers, "
            "for example 0,2,3"
        )
    bits = [int(piece) for piece in pieces]
    if len(set(bits)) != len(bits):
        raise argparse.ArgumentTypeError("candidate bits must not contain duplicates")
    if any(bit > 16 for bit in bits):
        raise argparse.ArgumentTypeError("candidate bits must be at most 16")
    return bits


def _positive_int(value):
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return integer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute C[l,i,b] = E_{x in X[l,i]}[g[l,i](x) "
            "||f[l,i](x)-f_b[l,i](x)||_2] for Qwen3-MoE."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--model_dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--attn_impl", choices=["eager", "sdpa"], default="eager"
    )
    parser.add_argument("--use_fast", action="store_true")

    parser.add_argument("--calib_dataset", default="mixed_chat_en")
    parser.add_argument("--calib_data_path", default="")
    parser.add_argument("--nsamples", type=_positive_int, default=128)
    parser.add_argument("--seqlen", type=_positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forward_batch_size",
        type=_positive_int,
        default=1,
        help="Number of calibration sequences in each decoder-layer forward.",
    )
    parser.add_argument(
        "--expert_batch_size",
        type=_positive_int,
        default=4096,
        help="Maximum number of active tokens in one expert forward.",
    )

    parser.add_argument(
        "--candidate_bits",
        type=_parse_candidate_bits,
        default=_parse_candidate_bits("0,1,2,3"),
    )
    parser.add_argument(
        "--context_mode",
        choices=["fp", "uniform_bit"],
        default="uniform_bit",
    )
    parser.add_argument("--average_bits", type=_positive_int, default=2)
    parser.add_argument("--blocksize", type=_positive_int, default=128)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args(argv)


def _validate_args(args):
    if args.model_name not in NAME_TO_MODEL:
        raise ValueError(f"Unknown --model_name: {args.model_name}")
    if NAME_TO_MODEL[args.model_name] != ModelType.QWEN3MOE:
        raise ValueError("This first implementation supports Qwen3-MoE only.")
    if args.average_bits > 16:
        raise ValueError("--average_bits must be at most 16.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if (
        args.context_mode == "uniform_bit"
        and args.average_bits not in args.candidate_bits
    ):
        raise ValueError(
            "--average_bits must occur in --candidate_bits when "
            "--context_mode=uniform_bit."
        )
    if args.nsamples % args.forward_batch_size != 0:
        raise ValueError(
            "--nsamples must be divisible by --forward_batch_size."
        )
    if args.calib_dataset == "mixed_chat_en":
        if not args.calib_data_path:
            raise ValueError(
                "--calib_data_path is required for mixed_chat_en calibration."
            )
        if not os.path.isfile(args.calib_data_path):
            raise FileNotFoundError(args.calib_data_path)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    if not torch.cuda.is_available():
        raise RuntimeError("Expert-cost collection requires a CUDA-capable PyTorch.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=args.model_dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    model.seqlen = args.seqlen

    # get_calib_loader uses batch_size; expose the more descriptive name in this
    # CLI while retaining compatibility with the shared loader.
    args.batch_size = args.forward_batch_size
    dataloader = get_calib_loader(tokenizer, args)

    costs, counts = compute_qwen3_expert_costs(
        model=model,
        dataloader=dataloader,
        model_name=args.model_name,
        candidate_bits=args.candidate_bits,
        context_mode=args.context_mode,
        average_bits=args.average_bits,
        blocksize=args.blocksize,
        expert_batch_size=args.expert_batch_size,
        device="cuda",
    )

    filled_costs, imputed_mask = impute_zero_frequency_costs(costs, counts)
    artifact = {
        "costs": filled_costs,
        "raw_costs": costs,
        "counts": counts,
        "imputed_mask": imputed_mask,
        "candidate_bits": torch.tensor(args.candidate_bits, dtype=torch.int64),
        "metadata": {
            "format_version": 2,
            "model": args.model,
            "model_name": args.model_name,
            "model_dtype": args.model_dtype,
            "calib_dataset": args.calib_dataset,
            "calib_data_path": args.calib_data_path,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
            "forward_batch_size": args.forward_batch_size,
            "expert_batch_size": args.expert_batch_size,
            "context_mode": args.context_mode,
            "average_bits": args.average_bits,
            "blocksize": args.blocksize,
            "cost_definition": (
                "mean over Top-K-selected tokens of "
                "g_i(x) * ||f_i(x) - f_i^b(x)||_2"
            ),
            "counts_definition": "number of tokens whose Top-K contains expert i",
            "routing_weight_definition": (
                "softmax router probability after optional Top-K renormalization"
            ),
            "quantizer": "GEMQ MCMoeRTN block-wise weight-only fake quantization",
            "zero_bit_definition": "f_i^0(x) = 0",
            "zero_frequency_policy": (
                "replace count-zero costs by the same-layer, same-bit mean over "
                "positive-count experts; preserve pre-imputation values in raw_costs"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(artifact, args.output_path)

    print(f"Saved expert cost artifact to: {args.output_path}")
    print(f"costs.shape={tuple(filled_costs.shape)}, counts.shape={tuple(counts.shape)}")
    print(f"zero-frequency experts: {int(imputed_mask.sum().item())}")
    print("costs (zero-frequency entries imputed):")
    print(filled_costs)
    print("counts:")
    print(counts)


if __name__ == "__main__":
    main()
