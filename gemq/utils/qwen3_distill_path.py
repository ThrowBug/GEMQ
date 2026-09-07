"""Resolve serving paths for Qwen3 distilled-CE runs from actual allocations.

No model/tensor dependencies; FQ_MODEL_PATH remains the explicit override in the
shell scripts, including for models moved away from their allocation files.
"""

import argparse
import os
from pathlib import Path

from gemq.router_finetune.config import validate_transfer_weight
from gemq.utils.expert_bit_config import has_zero_bit_experts, load_expert_bit_config


def resolve_distill_path(model_name, allocation_tag, bpe, attn_bits, dense_bits, weight, env):
    numeric_weight = validate_transfer_weight(weight)
    mixed = env.get("MIXED_PREC", "true") == "true"
    diagnostics = False
    if mixed:
        metric = env.get("ALLOCATION_METRIC", "expert_cost")
        if metric not in {"expert_cost", "layer_re"}:
            raise ValueError("ALLOCATION_METRIC must be expert_cost or layer_re.")
        bits = env.get("WBITS") or ("0,2,3" if metric == "expert_cost" else "1,2,3")
        prefix = f"{allocation_tag}_Metric-{metric}"
        if metric == "expert_cost":
            if env.get("EXTRA_CONSTR", "none") != "none":
                raise ValueError("EXTRA_CONSTR is only valid with ALLOCATION_METRIC=layer_re.")
            prefix += f"-Ctxuniform{env.get('EXPERT_COST_CONTEXT_BIT', '2')}"
        prefix += f"_E{float(bpe):.1f}_B{bits}"
        if metric == "expert_cost" and "0" in bits.split(","):
            prefix += f"_Pmax{float(env.get('MAX_PRUNE_RATIO', '0.25')):g}_EqPrune"
        if metric == "layer_re":
            constraints = env.get("EXTRA_CONSTR", "c2c3")
            if constraints != "none":
                prefix += f"_{constraints}"
        config = Path(env.get("BIT_CFG_PATH") or f"configs/{model_name}/GEMQ/{prefix}.pkl")
        if not config.is_file():
            raise FileNotFoundError(
                f"Cannot infer the distilled-CE result without its allocation: {config}. "
                "Set BIT_CFG_PATH (and matching quantization settings) or FQ_MODEL_PATH explicitly."
            )
        diagnostics = has_zero_bit_experts(load_expert_bit_config(config))
        prefix = config.name.removesuffix(".pkl")
        qtype = config.parent.name
    else:
        prefix, qtype = allocation_tag, "Uniform"
    if numeric_weight > 0 and not diagnostics:
        raise ValueError("Output-reconstruction transfer requires mixed allocation with actual 0-bit experts.")
    tag = "_RFT-distill_ce"
    if diagnostics:
        weight_tag = str(weight).replace(".", "p") if numeric_weight > 0 else "0p0"
        tag += f"-ReconKL-w{weight_tag}-const"
    return (
        f"results/fake_quant_models/{model_name}/{qtype}/"
        f"{prefix}_A{attn_bits}-G16-D{dense_bits}-E{bpe}{tag}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "allocation_tag", "bpe", "attn_bits", "dense_bits", "weight"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        print(resolve_distill_path(
            args.model, args.allocation_tag, args.bpe, args.attn_bits,
            args.dense_bits, args.weight, os.environ,
        ))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
