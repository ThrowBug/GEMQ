#!/bin/bash
set -euo pipefail

# First produce an untouched GPTQ checkpoint with, for example:
# CALIB_DATASET=c4 MAX_PRUNE_RATIO=0.1 FINETUNE_ROUTERS=false \
# SAVE_GPTQ_CHECKPOINT=true SAVE_MODEL=false bash scripts/Qwen3-30B-A3B-Instruct-2507/quantize.sh
# Then set GPTQ_CHECKPOINT_PATH and the exact BIT_CFG_PATH used above.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
checkpoint="${GPTQ_CHECKPOINT_PATH:-}"
bit_cfg="${BIT_CFG_PATH:-}"
stage="${RFT_STAGE:-norm_only}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${checkpoint}" || ! -f "${checkpoint}/_SUCCESS" ]]; then
    echo "Set GPTQ_CHECKPOINT_PATH to a complete post-GPTQ checkpoint." >&2
    exit 1
fi
if [[ -z "${bit_cfg}" ]]; then
    echo "Set BIT_CFG_PATH to the exact allocation file used for GPTQ (required for mixed precision)." >&2
    exit 1
fi
if [[ ! -f "${bit_cfg}" ]]; then
    echo "Allocation file not found: ${bit_cfg}" >&2
    exit 1
fi

checkpoint_name="$(basename "${checkpoint}")"
output="${OUTPUT_PATH:-results/router_norm_models/${model}/${checkpoint_name}_RFT-router_norm_reconstruction-${stage}-last_stage}"
if [[ -e "${output}" || -L "${output}" ]]; then
    echo "Output already exists and will not be overwritten: ${output}" >&2
    exit 1
fi

echo "Teacher: ${model}"
echo "GPTQ checkpoint: ${checkpoint}"
echo "Allocation: ${bit_cfg}"
echo "Stage: ${stage}"
echo "Output: ${output}"

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.finetune_router_norm \
    --model "${model}" \
    --model_name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --gptq_checkpoint_path "${checkpoint}" \
    --bit_cfg "${bit_cfg}" \
    --save_path "${output}" \
    --calib_dataset c4 \
    --nsamples "${NSAMPLES:-128}" \
    --val_nsamples "${VAL_NSAMPLES:-32}" \
    --seqlen "${SEQLEN:-2048}" \
    --seed "${SEED:-0}" \
    --max_prune_ratio "${MAX_PRUNE_RATIO:-0.1}" \
    --batch_size "${BATCH_SIZE:-1}" \
    --model_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL:-eager}" \
    --use_fast \
    --rft_trainer router_norm_reconstruction \
    --rft_stage "${stage}" \
    --rft_epochs "${RFT_EPOCHS:-1}" \
    --rft_batch_size "${RFT_BATCH_SIZE:-1}" \
    --rft_norm_lr "${RFT_NORM_LR:-1e-4}" \
    --rft_router_lr "${RFT_ROUTER_LR:-1e-5}" \
    "$@"
