#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# FP loads the original checkpoint. FQ loads GEMQ's standard BF16 fake-quant
# checkpoint containing dequantized approximate weights W_hat.
model_name="Qwen/Qwen3.5-35B-A3B"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

seed="${SEED:-0}"
metric="${ALLOCATION_METRIC:-expert_cost}"
mixed_prec="${MIXED_PREC:-true}"
bpe="${BITS_PER_EXPERT:-${BPE:-2.0}}"
bpe_tag="$(printf '%.1f' "${bpe}")"
expert_wbits="${EXPERT_WBITS:-2}"
linear_attn_wbits="${LINEAR_ATTN_WBITS:-4}"
softmax_attn_wbits="${SOFTMAX_ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"

case "${metric}" in
    expert_cost)
        alloc_bits="${WBITS:-0,2,3}"
        max_prune_ratio="${MAX_PRUNE_RATIO:-0.1}"
        context_bit="${EXPERT_COST_CONTEXT_BIT:-2}"
        allocation_tag="C4-Seed${seed}_Metric-expert_cost-Ctxuniform${context_bit}_E${bpe_tag}_B${alloc_bits}_Pmax${max_prune_ratio}_EqPrune"
        ;;
    layer_re)
        alloc_bits="${WBITS:-1,2,3}"
        extra_constr="${EXTRA_CONSTR:-c2c3}"
        allocation_tag="C4-Seed${seed}_Metric-layer_re_E${bpe_tag}_B${alloc_bits}_${extra_constr}"
        ;;
    *)
        echo "ALLOCATION_METRIC must be expert_cost or layer_re." >&2
        exit 1
        ;;
esac

if [[ "${mixed_prec}" == "true" ]]; then
    quant_tag="${allocation_tag}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}"
else
    quant_tag="C4-Seed${seed}_Uniform-E${expert_wbits}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}"
fi

finetune_routers="${FINETUNE_ROUTERS:-false}"
rft_trainer="${RFT_TRAINER:-router_compensated_dual_norm_distill}"
rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    case "${rft_trainer}" in
        legacy_ce|dual_norm_distill|router_compensated_dual_norm_distill)
            rft_lr="${RFT_LR:-1e-4}"
            rft_lr_tag="$(python -c 'import sys; from decimal import Decimal; print(format(Decimal(sys.argv[1]).normalize(), "E").lower().replace("e+", "e"))' "${rft_lr}")"
            rft_tag="_RFT-${rft_trainer}-lr${rft_lr_tag}"
            ;;
        *)
            echo "Qwen3.5 supports only legacy_ce, dual_norm_distill, or router_compensated_dual_norm_distill; got ${rft_trainer}." >&2
            exit 1
            ;;
    esac
fi

default_fq_path="results/fake_quant_models/${model_name}/GEMQ/${quant_tag}${rft_tag}"
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
    *)
        echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2
        exit 1
        ;;
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
    --no-enable-log-requests
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
