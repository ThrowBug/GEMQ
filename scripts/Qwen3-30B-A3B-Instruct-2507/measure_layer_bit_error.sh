#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

args=(
    --model "${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
    --layer "${LAYER:-24}"
    --calib_samples "${CALIB_SAMPLES:-128}"
    --eval_samples "${EVAL_SAMPLES:-8}"
    --seqlen "${SEQLEN:-2048}"
    --seed "${SEED:-0}"
    --groupsize "${GROUPSIZE:-128}"
    --blocksize "${BLOCKSIZE:-128}"
    --percdamp "${PERCDAMP:-0.01}"
    --attn_impl "${ATTN_IMPL:-eager}"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(--output_dir "${OUTPUT_DIR}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m gemq.measure_layer_bit_error "${args[@]}"

# Plot separately so styling changes never repeat GPTQ:
# python -m gemq.plot_layer_bit_error --input cache/layer_bit_error/.../summary.csv
