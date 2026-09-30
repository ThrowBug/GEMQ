#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if (( $# > 1 )); then
    echo "Usage: bash ${BASH_SOURCE[0]} MODEL_PATH" >&2
    exit 1
fi
model_path="${1:-${MODEL_PATH:-}}"
if [[ -z "${model_path}" || ! -d "${model_path}" ]]; then
    echo "Pass an existing saved Qwen3 model directory as MODEL_PATH or the first argument." >&2
    exit 1
fi
# Resolve before changing directories so a relative path is relative to the caller.
model_path="$(cd "${model_path}" && pwd -P)"
if [[ ! -f "${model_path}/config.json" ]]; then
    echo "Missing config.json under: ${model_path}" >&2
    exit 1
fi
if [[ ! -f "${model_path}/tokenizer_config.json" && ! -f "${model_path}/tokenizer.json" ]]; then
    echo "Missing tokenizer files under: ${model_path}" >&2
    exit 1
fi
shopt -s nullglob
weight_files=("${model_path}"/*.safetensors "${model_path}"/*.bin)
shopt -u nullglob
if (( ${#weight_files[@]} == 0 )); then
    echo "No .safetensors or .bin model weights found under: ${model_path}" >&2
    exit 1
fi
if [[ "${model_path}" == *,* ]]; then
    echo "lm_eval model_args cannot parse a model path containing a comma: ${model_path}" >&2
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "python not found. Activate the GEMQ evaluation environment first." >&2
    exit 1
fi
python -c 'from lm_eval.models.huggingface import HFLM' || {
    echo "The lm_eval Hugging Face backend is unavailable in this Python environment." >&2
    exit 1
}

cd "${repo_root}"
gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
eval_batch_size="${EVAL_BATCH_SIZE:-1}"
num_fewshot="${NUM_FEWSHOT:-0}"
parallelize="${PARALLELIZE:-true}"
output_path="${OUTPUT_PATH:-}"
tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande"

if [[ "${parallelize}" != "true" && "${parallelize}" != "false" ]]; then
    echo "PARALLELIZE must be true or false, got: ${parallelize}" >&2
    exit 1
fi
if [[ ! "${eval_batch_size}" =~ ^[1-9][0-9]*$ || ! "${num_fewshot}" =~ ^[0-9]+$ ]]; then
    echo "EVAL_BATCH_SIZE must be positive and NUM_FEWSHOT must be nonnegative." >&2
    exit 1
fi

model_args="pretrained=${model_path},dtype=${model_dtype},parallelize=${parallelize},attn_implementation=eager"
eval_args=(
    --model hf
    --model_args "${model_args}"
    --tasks "${tasks}"
    --num_fewshot "${num_fewshot}"
    --batch_size "${eval_batch_size}"
    --device cuda:0
)
if [[ -n "${LIMIT:-}" ]]; then eval_args+=(--limit "${LIMIT}"); fi
if [[ -n "${output_path}" ]]; then eval_args+=(--output_path "${output_path}"); fi

echo "=============================================="
echo ">>> Qwen3 PIQA / ARC-Easy / ARC-Challenge / HellaSwag / Winogrande"
echo " Model path:    ${model_path}"
echo " CUDA devices:  ${gpus}"
echo " Dtype:         ${model_dtype}"
echo " Batch size:    ${eval_batch_size}"
echo " Few-shot:      ${num_fewshot}"
echo " Parallelize:   ${parallelize}"
echo " Output:        ${output_path:-stdout only}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m lm_eval "${eval_args[@]}"
