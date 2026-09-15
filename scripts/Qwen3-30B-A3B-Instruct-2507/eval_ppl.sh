#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model_path="${1:-${MODEL_PATH:-}}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ -z "${model_path}" ]]; then
    echo "Set MODEL_PATH or pass the saved fake-quant checkpoint as the first argument." >&2
    echo "Example: MODEL_PATH=results/fake_quant_models/... bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi
if [[ ! -d "${model_path}" ]]; then
    echo "Model checkpoint directory not found: ${model_path}" >&2
    exit 1
fi
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
if ! command -v python >/dev/null 2>&1; then
    echo "python command not found. Activate the GEMQ environment first." >&2
    exit 1
fi

gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
seqlen="${SEQLEN:-2048}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
ppl_datasets="${PPL_DATASETS:-wikitext2,c4}"
attn_impl="${ATTN_IMPL:-eager}"
offload="${OFFLOAD:-false}"
trust_remote_code="${TRUST_REMOTE_CODE:-false}"
cuda_diagnostics="${CUDA_DIAGNOSTICS:-false}"

for boolean_name in offload trust_remote_code cuda_diagnostics; do
    boolean_value="${!boolean_name}"
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "${boolean_name^^} must be true or false, got: ${boolean_value}" >&2
        exit 1
    fi
done

eval_args=(
    --model "${model_path}"
    --model_name "${model_name}"
    --model_dtype "${model_dtype}"
    --seqlen "${seqlen}"
    --datasets "${ppl_datasets}"
    --attn_impl "${attn_impl}"
    --use_fast
)
if [[ "${offload}" == "true" ]]; then eval_args+=(--offload); fi
if [[ "${trust_remote_code}" == "true" ]]; then eval_args+=(--trust_remote_code); fi
if [[ "${cuda_diagnostics}" == "true" ]]; then eval_args+=(--cuda_diagnostics); fi
eval_args+=("$@")

echo "=============================================="
echo ">>> Qwen3-30B-A3B-Instruct-2507 perplexity"
echo " Model path:       ${model_path}"
echo " CUDA devices:     ${gpus}"
echo " Model dtype:      ${model_dtype}"
echo " Sequence length:  ${seqlen}"
echo " Datasets:         ${ppl_datasets}"
echo " Attention impl:   ${attn_impl}"
echo " Layer offloading: ${offload}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.evaluate_ppl "${eval_args[@]}"
