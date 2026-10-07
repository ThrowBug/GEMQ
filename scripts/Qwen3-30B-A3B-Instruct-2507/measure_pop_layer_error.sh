#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
layer="${LAYER:-24}"
calib_samples="${CALIB_SAMPLES:-128}"
eval_samples="${EVAL_SAMPLES:-8}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
expert_batch_size="${EXPERT_BATCH_SIZE:-1024}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"

args=(
    --model "${model}"
    --layer "${layer}"
    --calib_samples "${calib_samples}"
    --eval_samples "${eval_samples}"
    --seqlen "${seqlen}"
    --seed "${seed}"
    --expert_batch_size "${expert_batch_size}"
    --groupsize "${GROUPSIZE:-128}"
    --blocksize "${BLOCKSIZE:-128}"
    --percdamp "${PERCDAMP:-0.01}"
    --attn_impl "${ATTN_IMPL:-eager}"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    args+=(--output_dir "${OUTPUT_DIR}")
fi

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.measure_pop_layer_error "${args[@]}"
