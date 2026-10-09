#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

args=(
    --model "${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
    --calib_samples "${CALIB_SAMPLES:-128}"
    --eval_samples "${EVAL_SAMPLES:-16}"
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
if [[ -n "${MAX_LAYERS:-}" ]]; then
    args+=(--max_layers "${MAX_LAYERS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python -m gemq.measure_cumulative_bit_error "${args[@]}"

# Plot separately without repeating GPTQ:
# python -m gemq.plot_cumulative_bit_error --input cache/cumulative_bit_error/.../layer0_bit_propagation.csv
