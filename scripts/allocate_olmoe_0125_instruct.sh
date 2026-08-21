#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

# ===============================
#  Allocation settings
# ===============================
model_name="allenai/OLMoE-1B-7B-0125-Instruct"
bits_per_expert="${BITS_PER_EXPERT:-2.0}"
wbits="${WBITS:-1,2,3}"
ilp_solver="gemq"
ilp_backend="highs"
extra_constr=${EXTRA_CONSTR:-"c2c3"}
default_layer_re_path="cache/${model_name}/LayerRE_c4-N128-L2048-Seed0_B${wbits}_fast.pkl"
layer_re_path="${LAYER_RE_PATH:-${default_layer_re_path}}"

if [[ ! -f "${layer_re_path}" ]]; then
    echo "Layer reconstruction statistics not found: ${layer_re_path}" >&2
    echo "Run scripts/compute_stats_olmoe_0125_instruct.sh first." >&2
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
