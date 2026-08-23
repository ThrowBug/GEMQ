#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="allenai/OLMoE-1B-7B-0125-Instruct"
calib_dataset="${CALIB_DATASET:-mixed_chat_en}"
seed="${SEED:-0}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
default_output="cache/calibration/${model_name}/${calib_dataset}-N${nsamples}-L${seqlen}-Seed${seed}.pt"
output="${CALIB_DATA_PATH:-${default_output}}"

if [[ "${calib_dataset}" != "mixed_chat_en" ]]; then
    echo "CALIB_DATASET=${calib_dataset} uses the legacy data loader; no .pt cache is prepared."
    exit 0
fi

args=(
    --model "${model_name}"
    --output "${output}"
    --seed "${seed}"
    --nsamples "${nsamples}"
    --seqlen "${seqlen}"
    --use_fast
)
if [[ "${REBUILD_CALIB_CACHE:-false}" == "true" ]]; then
    args+=(--rebuild)
fi

python -m gemq.prepare_calib_data "${args[@]}"
