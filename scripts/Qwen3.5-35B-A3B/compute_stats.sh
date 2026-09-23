#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
model="${MODEL:-${model_name}}"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
wbits="${WBITS:-1,2,3}"
forward_batch_size="${FORWARD_BATCH_SIZE:-32}"

[[ -f data/c4-train.00000-of-01024.json ]] || {
    echo "Missing C4 shard: data/c4-train.00000-of-01024.json" >&2
    exit 1
}
(( nsamples % forward_batch_size == 0 )) || {
    echo "NSAMPLES must be divisible by FORWARD_BATCH_SIZE." >&2
    exit 1
}

layer_grads_path="cache/${model_name}/LayerGrads_c4-N${nsamples}-L${seqlen}-Seed${seed}.pt"
layer_re_path="cache/${model_name}/LayerRE_c4-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}_fast.pkl"

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode layer_grads \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset c4 \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --layer_grads_path "${layer_grads_path}"

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode layer_re \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset c4 \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --wbits "${wbits}" \
    --layer_grads_path "${layer_grads_path}" \
    --layer_re_path "${layer_re_path}" \
    --forward_batch_size "${forward_batch_size}"
