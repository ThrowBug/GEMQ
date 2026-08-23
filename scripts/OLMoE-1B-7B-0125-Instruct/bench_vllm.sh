#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# This script benchmarks online serving performance. It does not calculate
# GSM8K or other task accuracy; use eval_vllm.sh for that.
model_name="allenai/OLMoE-1B-7B-0125-Instruct"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

bpe="${BPE:-2.0}"
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
        tokenizer_path="${FP_MODEL_PATH:-${model_name}}"
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fp}"
        ;;
    FQ)
        tokenizer_path="${FQ_MODEL_PATH:-${default_fq_path}}"
        served_model_name="${SERVED_MODEL_NAME:-olmoe-0125-fq}"
        if [[ ! -d "${tokenizer_path}" ]]; then
            echo "Fake-quant checkpoint not found: ${tokenizer_path}" >&2
            exit 1
        fi
        ;;
    *)
        echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2
        exit 1
        ;;
esac

if ! command -v vllm >/dev/null 2>&1; then
    echo "vllm command not found. Run this script in the vllm environment." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the API health check." >&2
    exit 1
fi

host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
base_url="${VLLM_BASE_URL:-http://${host}:${port}}"
base_url="${base_url%/}"
api_key="${VLLM_API_KEY:-EMPTY}"

num_prompts="${NUM_PROMPTS:-100}"
input_len="${INPUT_LEN:-1024}"
output_len="${OUTPUT_LEN:-128}"
range_ratio="${RANGE_RATIO:-0.0}"
request_rate="${REQUEST_RATE:-inf}"
max_concurrency="${MAX_CONCURRENCY:-16}"
seed="${SEED:-0}"
result_dir="${RESULT_DIR:-results/vllm_benchmarks/OLMoE-1B-7B-0125-Instruct/${model_variant}}"

if ! curl --fail --silent --show-error \
    --header "Authorization: Bearer ${api_key}" \
    "${base_url}/v1/models" >/dev/null; then
    echo "vLLM API is unavailable at ${base_url}/v1." >&2
    echo "Start scripts/OLMoE-1B-7B-0125-Instruct/serve_vllm.sh in another terminal first." >&2
    exit 1
fi

mkdir -p "${result_dir}"
bench_args=(
    bench serve
    --backend openai-chat
    --base-url "${base_url}"
    --endpoint /v1/chat/completions
    --model "${tokenizer_path}"
    --tokenizer "${tokenizer_path}"
    --served-model-name "${served_model_name}"
    --dataset-name random
    --num-prompts "${num_prompts}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    --random-range-ratio "${range_ratio}"
    --request-rate "${request_rate}"
    --max-concurrency "${max_concurrency}"
    --seed "${seed}"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,99
    --save-result
    --result-dir "${result_dir}"
    --metadata "model_variant=${model_variant}" "dtype=bfloat16"
)

if [[ "${IGNORE_EOS:-true}" == "true" ]]; then
    bench_args+=(--ignore-eos)
fi
if [[ "${SAVE_DETAILED:-false}" == "true" ]]; then
    bench_args+=(--save-detailed)
fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then
    bench_args+=(--trust-remote-code)
fi
bench_args+=("$@")

echo "=============================================="
echo ">>> OLMoE-0125 vLLM API benchmark"
echo " Model variant:       ${model_variant}"
echo " Served model name:   ${served_model_name}"
echo " API:                 ${base_url}/v1/chat/completions"
echo " Requests:            ${num_prompts}"
echo " Input/output tokens: ${input_len}/${output_len}"
echo " Request rate:        ${request_rate}"
echo " Max concurrency:     ${max_concurrency}"
echo " Results:             ${result_dir}"
echo "=============================================="

OPENAI_API_KEY="${api_key}" vllm "${bench_args[@]}"
