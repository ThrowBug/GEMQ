#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# FP loads the original checkpoint. FQ loads a standard BF16 fake-quant
# checkpoint containing dequantized approximate weights W_hat.
model_name="Qwen/Qwen3.5-35B-A3B"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

bpe="${BPE:-2.0}"
linear_attn_wbits="${LINEAR_ATTN_WBITS:-4}"
softmax_attn_wbits="${SOFTMAX_ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"
calib_dataset="${CALIB_DATASET:-c4}"
calib_seed="${CALIB_SEED:-${SEED:-0}}"
allocation_tag="${ALLOCATION_TAG:-}"
if [[ -z "${allocation_tag}" ]]; then
    case "${calib_dataset}" in
        c4) allocation_tag="C4-Seed${calib_seed}" ;;
        math) allocation_tag="MATH-Seed${calib_seed}" ;;
        math+c4) allocation_tag="MATH+C4-Seed${calib_seed}" ;;
        *) echo "Set ALLOCATION_TAG for CALIB_DATASET=${calib_dataset}." >&2; exit 1 ;;
    esac
fi

finetune="${FINETUNE:-true}"
rft_trainer="${RFT_TRAINER:-roco}"
rft_lr="${RFT_LR:-1e-4}"
rft_lr_tag="$(python -c 'import sys; from decimal import Decimal; print(format(Decimal(sys.argv[1]).normalize(), "E").lower().replace("e+", "e"))' "${rft_lr}")"
rft_tag=""
if [[ "${finetune}" == "true" ]]; then
    case "${rft_trainer}" in
        router_ft) rft_tag="_RFT-router_ft" ;;
        norm_ft|roco) rft_tag="_RFT-${rft_trainer}-lr${rft_lr_tag}" ;;
        *) echo "Unsupported RFT_TRAINER: ${rft_trainer}" >&2; exit 1 ;;
    esac
fi
if [[ -n "${MIXED_PREC+x}" ]]; then
    echo "MIXED_PREC is no longer supported; use an allocated model path." >&2
    exit 1
fi
allocation_metric="${ALLOCATION_METRIC:-pop}"
if [[ -n "${WBITS:-}" ]]; then
    wbits="${WBITS}"
elif [[ "${allocation_metric}" == "pop" ]]; then
    wbits="0,2,3"
else
    wbits="1,2,3"
fi
bpe_tag="$(printf '%.1f' "${bpe}")"
if [[ -n "${BIT_CFG_PATH:-}" ]]; then
    bit_cfg="${BIT_CFG_PATH}"
elif [[ "${allocation_metric}" == "pop" ]]; then
    context_bit="${POP_CONTEXT_BIT:-${AVERAGE_BITS:-2}}"
    count_tag=""
    [[ "${NORMALIZE_BY_ACTIVATION_COUNT:-true}" == "false" ]] && count_tag="-NoCountNorm"
    bit_cfg="configs/${model_name}/ProQuant/${allocation_tag}_Metric-pop-Ctxuniform${context_bit}${count_tag}_E${bpe_tag}_B${wbits}"
    if [[ ",${wbits}," == *,0,* ]]; then
        bit_cfg+="_Pmax$(printf '%g' "${MAX_PRUNE_RATIO:-0.25}")_EqPrune"
    fi
    bit_cfg+=".pkl"
elif [[ "${allocation_metric}" == "layer_re" ]]; then
    constraint="${EXTRA_CONSTR:-c2c3}"
    constraint_tag=""; [[ "${constraint}" != "none" ]] && constraint_tag="_${constraint}"
    bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_Metric-layer_re_E${bpe_tag}_B${wbits}${constraint_tag}.pkl"
else
    echo "ALLOCATION_METRIC must be pop or layer_re." >&2
    exit 1
fi
qtype="$(basename "$(dirname "${bit_cfg}")")"
prefix="$(basename "${bit_cfg}" .pkl)"
default_fq_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}-E${bpe}${rft_tag}"

case "${model_variant}" in
    FP)
        model_path="${FP_MODEL_PATH:-${model_name}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen35-35b-a3b-fp}"
        ;;
    FQ)
        model_path="${FQ_MODEL_PATH:-${default_fq_path}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen35-35b-a3b-fq}"
        if [[ ! -d "${model_path}" ]]; then
            echo "Fake-quant checkpoint not found: ${model_path}" >&2
            echo "Run scripts/Qwen3.5-35B-A3B/quantize.sh first or set FQ_MODEL_PATH." >&2
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
)
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then serve_args+=(--max-model-len "${MAX_MODEL_LEN}"); fi
if [[ "${ENABLE_EXPERT_PARALLEL:-false}" == "true" ]]; then serve_args+=(--enable-expert-parallel); fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then serve_args+=(--trust-remote-code); fi
serve_args+=("$@")

echo "=============================================="
echo ">>> Starting Qwen3.5-35B-A3B vLLM service"
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
