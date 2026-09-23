#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
model="${MODEL:-${model_name}}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-1}"
candidate_bits="1,2,3"
stats_dir="${PMQ_STATS_DIR:-cache/${model_name}/PMQ/C4-N${nsamples}-L${seqlen}-Seed${seed}_B${candidate_bits}}"

[[ -f data/c4-train.00000-of-01024.json ]] || {
    echo "Missing C4 shard: data/c4-train.00000-of-01024.json" >&2
    exit 1
}
(( nsamples % batch_size == 0 )) || {
    echo "NSAMPLES must be divisible by BATCH_SIZE." >&2
    exit 1
}

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode mcmoe_stats \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${MODEL_DTYPE:-bfloat16}" \
    --attn_impl "${ATTN_IMPL:-eager}" \
    --use_fast \
    --calib_dataset c4 \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --batch_size "${batch_size}" \
    --seed "${seed}" \
    --wbits "${candidate_bits}" \
    --blocksize "${BLOCKSIZE:-128}" \
    --mcmoe_stats_dir "${stats_dir}"
