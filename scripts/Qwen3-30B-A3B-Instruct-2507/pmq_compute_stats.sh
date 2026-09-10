#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${MODEL:-${model_name}}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"

calib_dataset="c4"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-1}"
candidate_bits="1,2,3"
blocksize="${BLOCKSIZE:-128}"

if [[ ! -f "data/c4-train.00000-of-01024.json" ]]; then
    echo "C4 calibration shard not found: data/c4-train.00000-of-01024.json" >&2
    echo "Place the same C4 JSON shard used by GEMQ at that path." >&2
    exit 1
fi

stats_dir="${PMQ_STATS_DIR:-cache/${model_name}/PMQ/C4-N${nsamples}-L${seqlen}-Seed${seed}_B${candidate_bits}}"

echo "=============================================="
echo ">>> PMQ statistics: Qwen3-30B-A3B-Instruct-2507"
echo " Model:          ${model}"
echo " Dataset:        ${calib_dataset}"
echo " Samples/length: ${nsamples}/${seqlen}"
echo " Seed:           ${seed}"
echo " Candidate bits: ${candidate_bits}"
echo " Block size:     ${blocksize}"
echo " Output:         ${stats_dir}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode mcmoe_stats \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset "${calib_dataset}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --batch_size "${batch_size}" \
    --seed "${seed}" \
    --wbits "${candidate_bits}" \
    --blocksize "${blocksize}" \
    --mcmoe_stats_dir "${stats_dir}"
