#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

# FP loads the original Hugging Face checkpoint. FQ loads GEMQ's standard
# BF16 fake-quant checkpoint containing dequantized approximate weights W_hat.
# Neither mode enables a vLLM quantizer or invokes GEMQ inference patches.
model_name="allenai/OLMoE-1B-7B-0125-Instruct"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

bpe="${BPE:-2.0}"
finetune_routers="${FINETUNE_ROUTERS:-true}"
rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    rft_tag="_RFT"
fi
default_fq_path="results/fake_quant_models/${model_name}/GEMQ/C4-Seed0-WT2_A4-G16-D4-E${bpe}${rft_tag}"

case "${model_variant}" in
    FP)
        model_path="${FP_MODEL_PATH:-${model_name}}"
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fp}"
        ;;
    FQ)
        model_path="${FQ_MODEL_PATH:-${default_fq_path}}"
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fq}"
        if [[ ! -d "${model_path}" ]]; then
            echo "Fake-quant checkpoint not found: ${model_path}" >&2
            echo "Run scripts/quantize_olmoe_0125_instruct.sh first or set FQ_MODEL_PATH." >&2
            exit 1
        fi
        if [[ ! -f "${model_path}/config.json" || ! -f "${model_path}/tokenizer_config.json" ]]; then
            echo "Not a complete Hugging Face checkpoint: ${model_path}" >&2
            echo "config.json and tokenizer_config.json are required." >&2
            exit 1
        fi
        shopt -s nullglob
        weight_files=("${model_path}"/*.safetensors "${model_path}"/*.bin)
        shopt -u nullglob
        if (( ${#weight_files[@]} == 0 )); then
            echo "No model weight files found under: ${model_path}" >&2
            exit 1
        fi
        ;;
    *)
        echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2
        exit 1
        ;;
esac

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm command not found. Activate the vllm environment first." >&2
    exit 1
fi

gpus="${CUDA_VISIBLE_DEVICES:-0}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
data_parallel_size="${DATA_PARALLEL_SIZE:-1}"
host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_key="${VLLM_API_KEY:-EMPTY}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"

if ! [[ "${tensor_parallel_size}" =~ ^[1-9][0-9]*$ && "${data_parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE and DATA_PARALLEL_SIZE must be positive integers." >&2
    exit 1
fi

visible_gpu_count=$(( $(tr -cd ',' <<< "${gpus}" | wc -c) + 1 ))
required_gpu_count=$(( tensor_parallel_size * data_parallel_size ))
if (( visible_gpu_count < required_gpu_count )); then
    echo "CUDA_VISIBLE_DEVICES exposes ${visible_gpu_count} GPU(s), but TP x DP requires ${required_gpu_count}." >&2
    exit 1
fi

serve_args=(
    serve "${model_path}"
    --served-model-name "${served_model_name}"
    --dtype bfloat16
    --host "${host}"
    --port "${port}"
    --api-key "${api_key}"
    --tensor-parallel-size "${tensor_parallel_size}"
    --data-parallel-size "${data_parallel_size}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --generation-config vllm
    --disable-log-requests
)

if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    serve_args+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ "${ENABLE_EXPERT_PARALLEL:-false}" == "true" ]]; then
    serve_args+=(--enable-expert-parallel)
fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
    serve_args+=(--trust-remote-code)
fi
serve_args+=("$@")

echo "=============================================="
echo ">>> Starting OLMoE-0125 vLLM service"
echo " Model variant:       ${model_variant}"
echo " Model path:          ${model_path}"
echo " Served model name:   ${served_model_name}"
echo " Compute dtype:       bfloat16"
echo " Tensor parallel:     ${tensor_parallel_size}"
echo " Data parallel:       ${data_parallel_size}"
echo " CUDA devices:        ${gpus}"
echo " Endpoint:            http://${host}:${port}/v1"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" vllm "${serve_args[@]}"
