#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${MODEL:-${model_name}}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"

context_mode="${CONTEXT_MODE:-uniform_bit}"
average_bits="${AVERAGE_BITS:-2}"
candidate_bits="${CANDIDATE_BITS:-0,1,2,3}"
blocksize="${BLOCKSIZE:-128}"

dataset="${CALIB_DATASET:-mixed_chat_en}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
forward_batch_size="${FORWARD_BATCH_SIZE:-1}"
expert_batch_size="${EXPERT_BATCH_SIZE:-4096}"
calib_data_path="${CALIB_DATA_PATH:-}"

die() {
    echo "Error: $*" >&2
    exit 2
}

require_positive_int() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        die "${name} must be a positive integer; got '${value}'."
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || \
        die "${name} must be a non-negative integer; got '${value}'."
}

[[ "${context_mode}" == "fp" || "${context_mode}" == "uniform_bit" ]] || \
    die "CONTEXT_MODE must be 'fp' or 'uniform_bit'; got '${context_mode}'."
[[ "${model_dtype}" == "float16" || "${model_dtype}" == "bfloat16" ]] || \
    die "MODEL_DTYPE must be 'float16' or 'bfloat16'; got '${model_dtype}'."
[[ "${attn_impl}" == "eager" || "${attn_impl}" == "sdpa" ]] || \
    die "ATTN_IMPL must be 'eager' or 'sdpa'; got '${attn_impl}'."

require_positive_int "AVERAGE_BITS" "${average_bits}"
require_positive_int "BLOCKSIZE" "${blocksize}"
require_positive_int "NSAMPLES" "${nsamples}"
require_positive_int "SEQLEN" "${seqlen}"
require_positive_int "FORWARD_BATCH_SIZE" "${forward_batch_size}"
require_positive_int "EXPERT_BATCH_SIZE" "${expert_batch_size}"
require_nonnegative_int "SEED" "${seed}"
(( average_bits <= 16 )) || die "AVERAGE_BITS must be at most 16."
(( nsamples % forward_batch_size == 0 )) || \
    die "NSAMPLES must be divisible by FORWARD_BATCH_SIZE."

[[ "${candidate_bits}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || \
    die "CANDIDATE_BITS must be comma-separated non-negative integers without spaces."

IFS=',' read -r -a bit_values <<< "${candidate_bits}"
declare -A seen_bits=()
average_is_candidate=false
for bit in "${bit_values[@]}"; do
    (( bit <= 16 )) || die "Candidate bit '${bit}' exceeds 16."
    [[ -z "${seen_bits[${bit}]+x}" ]] || \
        die "CANDIDATE_BITS contains duplicate bit '${bit}'."
    seen_bits[${bit}]=1
    if (( bit == average_bits )); then
        average_is_candidate=true
    fi
done
if [[ "${context_mode}" == "uniform_bit" && "${average_is_candidate}" != "true" ]]; then
    die "AVERAGE_BITS must appear in CANDIDATE_BITS for uniform_bit context."
fi

calib_path_args=()
if [[ "${dataset}" == "mixed_chat_en" ]]; then
    if [[ -z "${calib_data_path}" ]]; then
        calib_data_path="cache/calibration/${model_name}/${dataset}-N${nsamples}-L${seqlen}-Seed${seed}.pt"
    fi
    [[ -f "${calib_data_path}" ]] || {
        echo "Prepared calibration data not found: ${calib_data_path}" >&2
        echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/prepare_calib.sh first." >&2
        exit 2
    }
    calib_path_args=(--calib_data_path "${calib_data_path}")
fi

context_tag="${context_mode}"
if [[ "${context_mode}" == "uniform_bit" ]]; then
    context_tag="uniform${average_bits}bit"
fi
default_output="cache/${model_name}/ExpertCosts_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_${context_tag}_B${candidate_bits}.pt"
output_path="${OUTPUT_PATH:-${default_output}}"

echo "Computing Qwen3-MoE expert costs"
echo "  context=${context_mode}, average_bits=${average_bits}, candidates=${candidate_bits}"
echo "  calibration=${dataset}, nsamples=${nsamples}, seqlen=${seqlen}"
echo "  output=${output_path}"

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_expert_costs \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset "${dataset}" \
    "${calib_path_args[@]}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --seed "${seed}" \
    --forward_batch_size "${forward_batch_size}" \
    --expert_batch_size "${expert_batch_size}" \
    --candidate_bits "${candidate_bits}" \
    --context_mode "${context_mode}" \
    --average_bits "${average_bits}" \
    --blocksize "${blocksize}" \
    --output_path "${output_path}"
