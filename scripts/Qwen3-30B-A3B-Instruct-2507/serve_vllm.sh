#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# FP loads the original checkpoint. FQ loads GEMQ's standard BF16 fake-quant
# checkpoint containing dequantized approximate weights W_hat.
model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

bpe="${BPE:-2.5}"
attn_wbits="${ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"
calib_dataset="${CALIB_DATASET:-mixed_chat_en}"
calib_seed="${CALIB_SEED:-0}"
allocation_tag="${ALLOCATION_TAG:-}"
if [[ -z "${allocation_tag}" ]]; then
    case "${calib_dataset}" in
        mixed_chat_en) allocation_tag="MixedChatEn-Seed${calib_seed}" ;;
        c4) allocation_tag="C4-Seed${calib_seed}" ;;
        math) allocation_tag="MATH-Seed${calib_seed}" ;;
        math+c4) allocation_tag="MATH+C4-Seed${calib_seed}" ;;
        *) echo "Set ALLOCATION_TAG for CALIB_DATASET=${calib_dataset}." >&2; exit 1 ;;
    esac
fi

finetune_routers="${FINETUNE_ROUTERS:-true}"
rft_trainer="${RFT_TRAINER:-layerwise_teacher}"
rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    case "${rft_trainer}" in
        legacy_ce) rft_tag="_RFT-legacy_ce" ;;
        distill_ce) rft_tag="_RFT-distill_ce" ;;
        layerwise_teacher)
            rft_timing="${RFT_TIMING:-after_each_layer_quantization}"
            timing_tag="all"; [[ "${rft_timing}" == "after_each_layer_quantization" ]] && timing_tag="each"
            rft_router_loss="${RFT_ROUTER_LOSS:-kd_tail}"
            alpha_tag="${RFT_ROUTER_ALPHA:-1.0}"; alpha_tag="${alpha_tag//./p}"
            router_weight_tag="${RFT_ROUTER_LOSS_WEIGHT:-1.0}"; router_weight_tag="${router_weight_tag//./p}"
            output_weight_tag="${RFT_OUTPUT_KL_WEIGHT:-0.0}"; output_weight_tag="${output_weight_tag//./p}"
            rft_tag="_RFT-${timing_tag}-${rft_router_loss}-a${alpha_tag}-rw${router_weight_tag}-ow${output_weight_tag}"
            ;;
        *) echo "Unsupported RFT_TRAINER: ${rft_trainer}" >&2; exit 1 ;;
    esac
fi
default_fq_path="results/fake_quant_models/${model_name}/GEMQ/${allocation_tag}_A${attn_wbits}-G16-D${dense_wbits}-E${bpe}${rft_tag}"

case "${model_variant}" in
    FP)
        model_path="${FP_MODEL_PATH:-${model_name}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen3-30b-a3b-2507-fp}"
        ;;
    FQ)
        model_path="${FQ_MODEL_PATH:-${default_fq_path}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen3-30b-a3b-2507-fq}"
        if [[ ! -d "${model_path}" ]]; then
            echo "Fake-quant checkpoint not found: ${model_path}" >&2
            echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/quantize.sh first or set FQ_MODEL_PATH." >&2
            exit 1
        fi
        if [[ ! -f "${model_path}/config.json" || ! -f "${model_path}/tokenizer_config.json" ]]; then
            echo "Not a complete Hugging Face checkpoint: ${model_path}" >&2
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
    *) echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2; exit 1 ;;
esac

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm command not found. Activate the vllm environment first." >&2
    exit 1
fi

gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-2}"
pipeline_parallel_size="${PIPELINE_PARALLEL_SIZE:-1}"
data_parallel_size="${DATA_PARALLEL_SIZE:-1}"
host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_key="${VLLM_API_KEY:-EMPTY}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"

if ! [[ "${tensor_parallel_size}" =~ ^[1-9][0-9]*$ && "${pipeline_parallel_size}" =~ ^[1-9][0-9]*$ && "${data_parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TENSOR_PARALLEL_SIZE, PIPELINE_PARALLEL_SIZE, and DATA_PARALLEL_SIZE must be positive integers." >&2
    exit 1
fi
visible_gpu_count=$(( $(tr -cd ',' <<< "${gpus}" | wc -c) + 1 ))
required_gpu_count=$(( tensor_parallel_size * pipeline_parallel_size * data_parallel_size ))
if (( visible_gpu_count < required_gpu_count )); then
    echo "CUDA_VISIBLE_DEVICES exposes ${visible_gpu_count} GPU(s), but TP x PP x DP requires ${required_gpu_count}." >&2
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
    --pipeline-parallel-size "${pipeline_parallel_size}"
    --data-parallel-size "${data_parallel_size}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --generation-config vllm
    --disable-log-requests
)
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then serve_args+=(--max-model-len "${MAX_MODEL_LEN}"); fi
if [[ "${ENABLE_EXPERT_PARALLEL:-false}" == "true" ]]; then serve_args+=(--enable-expert-parallel); fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then serve_args+=(--trust-remote-code); fi
serve_args+=("$@")

echo "=============================================="
echo ">>> Starting Qwen3-30B-A3B-Instruct-2507 vLLM service"
echo " Model variant:       ${model_variant}"
echo " Model path:          ${model_path}"
echo " Served model name:   ${served_model_name}"
echo " Tensor parallel:     ${tensor_parallel_size}"
echo " Pipeline parallel:   ${pipeline_parallel_size}"
echo " Data parallel:       ${data_parallel_size}"
echo " CUDA devices:        ${gpus}"
echo " Endpoint:            http://${host}:${port}/v1"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" vllm "${serve_args[@]}"
