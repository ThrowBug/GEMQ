#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# Run this client in the GEMQ/EvalScope environment while the vLLM service is
# running in the separate vllm environment.
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

case "${model_variant}" in
    FP)
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fp}"
        ;;
    FQ)
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fq}"
        ;;
    *)
        echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2
        exit 1
        ;;
esac

if ! command -v evalscope >/dev/null 2>&1; then
    echo "evalscope command not found. Activate the GEMQ/EvalScope environment first." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the API health check." >&2
    exit 1
fi

host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_url="${VLLM_API_URL:-http://${host}:${port}/v1}"
api_url="${api_url%/}"
api_key="${VLLM_API_KEY:-EMPTY}"

datasets="${DATASETS:-gsm8k}"
eval_batch_size="${EVAL_BATCH_SIZE:-8}"
default_generation_config='{"temperature":0.0,"max_tokens":2048,"seed":0,"timeout":600,"retries":3}'
generation_config="${GENERATION_CONFIG:-${default_generation_config}}"
read -r -a dataset_args <<< "${datasets}"

if ! curl --fail --silent --show-error \
    --header "Authorization: Bearer ${api_key}" \
    "${api_url}/models" >/dev/null; then
    echo "vLLM API is unavailable at ${api_url}." >&2
    echo "Start scripts/OLMoE-1B-7B-0125-Instruct/serve_vllm.sh in the vllm environment first." >&2
    exit 1
fi

eval_args=(
    eval
    --model "${served_model_name}"
    --model-id "${served_model_name}"
    --api-url "${api_url}"
    --api-key "${api_key}"
    --eval-type openai_api
    --datasets "${dataset_args[@]}"
    --eval-batch-size "${eval_batch_size}"
    --generation-config "${generation_config}"
)

if [[ -n "${LIMIT:-}" ]]; then
    eval_args+=(--limit "${LIMIT}")
fi
if [[ -n "${DATASET_HUB:-}" ]]; then
    eval_args+=(--dataset-hub "${DATASET_HUB}")
fi
eval_args+=("$@")

echo "=============================================="
echo ">>> OLMoE-0125 EvalScope API evaluation"
echo " Model variant:       ${model_variant}"
echo " Served model name:   ${served_model_name}"
echo " API URL:             ${api_url}"
echo " Datasets:            ${datasets}"
echo " Eval batch size:     ${eval_batch_size}"
echo " Limit:               ${LIMIT:-full dataset}"
echo "=============================================="

evalscope "${eval_args[@]}"
