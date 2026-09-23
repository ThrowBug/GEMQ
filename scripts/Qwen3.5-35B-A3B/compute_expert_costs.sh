#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
model="${MODEL:-${model_name}}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
forward_batch_size="${FORWARD_BATCH_SIZE:-1}"
expert_batch_size="${EXPERT_BATCH_SIZE:-4096}"
candidate_bits="${CANDIDATE_BITS:-0,1,2,3}"
average_bits="${AVERAGE_BITS:-2}"
context_mode="${CONTEXT_MODE:-uniform_bit}"
blocksize="${BLOCKSIZE:-128}"
output_path="${OUTPUT_PATH:-cache/${model_name}/ExpertCosts_c4-N${nsamples}-L${seqlen}-Seed${seed}_uniform${average_bits}bit_B${candidate_bits}.pt}"

[[ -f data/c4-train.00000-of-01024.json ]] || {
    echo "Missing C4 shard: data/c4-train.00000-of-01024.json" >&2
    exit 1
}
(( nsamples % forward_batch_size == 0 )) || {
    echo "NSAMPLES must be divisible by FORWARD_BATCH_SIZE." >&2
    exit 1
}

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_expert_costs \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset c4 \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --seed "${seed}" \
    --forward_batch_size "${forward_batch_size}" \
    --expert_batch_size "${expert_batch_size}" \
    --candidate_bits "${candidate_bits}" \
    --context_mode "${context_mode}" \
    --average_bits "${average_bits}" \
    --blocksize "${blocksize}" \
    --output_path "${output_path}"
