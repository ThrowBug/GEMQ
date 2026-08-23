#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# Qwen3-30B-A3B gradient statistics historically require about 3 x 80 GB GPUs.
# Override CUDA_VISIBLE_DEVICES for a different machine.
model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${model_name}"
model_str=""
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"
wbits="${WBITS:-1,2,3}"
forward_batch_size="${FORWARD_BATCH_SIZE:-32}"

dataset="${CALIB_DATASET:-mixed_chat_en}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
calib_data_path="${CALIB_DATA_PATH:-}"
calib_path_args=()

if [[ "${dataset}" == "mixed_chat_en" ]]; then
    if [[ -z "${calib_data_path}" ]]; then
        calib_data_path="cache/calibration/${model_name}/${dataset}-N${nsamples}-L${seqlen}-Seed${seed}.pt"
    fi
    if [[ ! -f "${calib_data_path}" ]]; then
        echo "Prepared calibration data not found: ${calib_data_path}" >&2
        echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/prepare_calib.sh first." >&2
        exit 1
    fi
    calib_path_args=(--calib_data_path "${calib_data_path}")
fi

layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode "layer_grads" \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --calib_dataset "${dataset}" \
    "${calib_path_args[@]}" \
    --use_fast \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --layer_grads_path "${layer_grads_path}"

layer_re_path="cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_fast.pkl"
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
    --mode "layer_re" \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --calib_dataset "${dataset}" \
    "${calib_path_args[@]}" \
    --use_fast \
    --seed "${seed}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --wbits "${wbits}" \
    --layer_grads_path "${layer_grads_path}" \
    --layer_re_path "${layer_re_path}" \
    --forward_batch_size "${forward_batch_size}"
