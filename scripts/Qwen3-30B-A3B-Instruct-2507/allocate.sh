#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
bits_per_expert="${BITS_PER_EXPERT:-2.5}"
wbits="${WBITS:-1,2,3}"
ilp_solver="gemq"
ilp_backend="${ILP_BACKEND:-highs}"
extra_constr="${EXTRA_CONSTR:-c2c3}"
calib_dataset="${CALIB_DATASET:-mixed_chat_en}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
default_layer_re_path="cache/${model_name}/LayerRE_${calib_dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}_fast.pkl"
layer_re_path="${LAYER_RE_PATH:-${default_layer_re_path}}"

if [[ ! -f "${layer_re_path}" ]]; then
    echo "Layer reconstruction statistics not found: ${layer_re_path}" >&2
    echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/compute_stats.sh first." >&2
    exit 1
fi

python -m gemq.allocate_bits \
    --model_name "${model_name}" \
    --layer_re_path "${layer_re_path}" \
    --bit_budget "${bits_per_expert}" \
    --bit_candidates "${wbits}" \
    --ilp_solver "${ilp_solver}" \
    --ilp_backend "${ilp_backend}" \
    --extra_constr "${extra_constr}"
