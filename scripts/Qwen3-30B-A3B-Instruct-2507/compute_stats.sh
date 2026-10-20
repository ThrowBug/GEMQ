#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# Qwen3-30B-A3B gradient statistics historically require about 3 x 80 GB GPUs.
# Override CUDA_VISIBLE_DEVICES for a different machine.
model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${MODEL:-${model_name}}"
model_str=""
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"
wbits="${WBITS:-1,2,3}"
forward_batch_size="${FORWARD_BATCH_SIZE:-32}"

dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"

layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
CUDA_VISIBLE_DEVICES="${gpus}" python -m proquant.compute_model_stats \
    --mode "layer_grads" \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --calib_dataset "${dataset}" \
    --use_fast \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --layer_grads_path "${layer_grads_path}"

layer_re_path="cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_fast.pkl"
CUDA_VISIBLE_DEVICES="${gpus}" python -m proquant.compute_model_stats \
    --mode "layer_re" \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --calib_dataset "${dataset}" \
    --use_fast \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --wbits "${wbits}" \
    --layer_grads_path "${layer_grads_path}" \
    --layer_re_path "${layer_re_path}" \
    --forward_batch_size "${forward_batch_size}"
