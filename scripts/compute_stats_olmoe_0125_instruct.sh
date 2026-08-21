#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

# ===============================
#  Model settings
# ===============================
model_name="allenai/OLMoE-1B-7B-0125-Instruct"
model="${model_name}"
model_str=""  # Empty means statistics are computed from the original model.
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
wbits="${WBITS:-1,2,3}"

# ===============================
#  Dataset settings
# ===============================
dataset="c4"
nsamples=128
seqlen=2048
seed=0


# =============================================================================
#  Step 1: Compute layer output gradients
# =============================================================================
layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
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


# =============================================================================
#  Step 2: Compute weighted layer reconstruction errors
# =============================================================================
layer_re_path="cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_fast.pkl"
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_model_stats \
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
    --forward_batch_size 32
