#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model_variant="${MODEL_VARIANT:-FQ}"
model_variant="${model_variant^^}"

bpe="${BPE:-2.0}"
attn_wbits="${ATTN_WBITS:-4}"
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
default_fq_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A${attn_wbits}-G16-D${dense_wbits}-E${bpe}${rft_tag}"

case "${model_variant}" in
    FP)
        tokenizer_path="${FP_MODEL_PATH:-${model_name}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen3-30b-a3b-2507-fp}"
        ;;
    FQ)
        tokenizer_path="${FQ_MODEL_PATH:-${default_fq_path}}"
        served_model_name="${SERVED_MODEL_NAME:-qwen3-30b-a3b-2507-fq}"
        if [[ ! -d "${tokenizer_path}" ]]; then
            echo "Fake-quant checkpoint not found: ${tokenizer_path}" >&2
            exit 1
        fi
        ;;
    *) echo "MODEL_VARIANT must be FP or FQ, got: ${model_variant}" >&2; exit 1 ;;
esac

if ! command -v vllm >/dev/null 2>&1; then echo "vllm command not found." >&2; exit 1; fi
if ! command -v curl >/dev/null 2>&1; then echo "curl is required." >&2; exit 1; fi

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
result_dir="${RESULT_DIR:-results/vllm_benchmarks/Qwen3-30B-A3B-Instruct-2507/${model_variant}}"

if ! curl --fail --silent --show-error --header "Authorization: Bearer ${api_key}" "${base_url}/v1/models" >/dev/null; then
    echo "vLLM API is unavailable at ${base_url}/v1." >&2
    echo "Start scripts/Qwen3-30B-A3B-Instruct-2507/serve_vllm.sh first." >&2
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
if [[ "${IGNORE_EOS:-true}" == "true" ]]; then bench_args+=(--ignore-eos); fi
if [[ "${SAVE_DETAILED:-false}" == "true" ]]; then bench_args+=(--save-detailed); fi
if [[ "${TRUST_REMOTE_CODE:-false}" == "true" ]]; then bench_args+=(--trust-remote-code); fi
bench_args+=("$@")

echo "=============================================="
echo ">>> Qwen3-30B-A3B-Instruct-2507 vLLM API benchmark"
echo " Model variant:       ${model_variant}"
echo " Served model name:   ${served_model_name}"
echo " API:                 ${base_url}/v1/chat/completions"
echo " Requests:            ${num_prompts}"
echo " Input/output tokens: ${input_len}/${output_len}"
echo " Results:             ${result_dir}"
echo "=============================================="

OPENAI_API_KEY="${api_key}" vllm "${bench_args[@]}"
