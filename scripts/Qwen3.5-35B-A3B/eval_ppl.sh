#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_path="${1:-${MODEL_PATH:-}}"
if [[ $# -gt 0 ]]; then shift; fi
[[ -n "${model_path}" && -f "${model_path}/config.json" ]] || {
    echo "Pass a saved Qwen3.5 checkpoint or set MODEL_PATH." >&2
    exit 1
}

args=(
    --model "${model_path}"
    --model_name "Qwen/Qwen3.5-35B-A3B"
    --model_dtype "${MODEL_DTYPE:-bfloat16}"
    --seqlen "${SEQLEN:-2048}"
    --datasets "${PPL_DATASETS:-wikitext2,c4}"
    --attn_impl "${ATTN_IMPL:-eager}"
    --use_fast
)
[[ "${OFFLOAD:-false}" != true ]] || args+=(--offload)
args+=("$@")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    python -m gemq.evaluate_ppl "${args[@]}"
