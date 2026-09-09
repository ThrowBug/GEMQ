#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
groupsize="${GROUPSIZE:-128}"
cost_candidate_bits="0,1,2,3"
allocation_bits="${BIT_CANDIDATES:-0,2,3}"
bit_budget="2.0"
max_prune_ratio="0.1"

default_cost_path="cache/${model_name}/ExpertCosts_c4-N${nsamples}-L${seqlen}-Seed${seed}_uniform2bit_B${cost_candidate_bits}_Quant-awq-G${groupsize}.pt"
expert_cost_path="${EXPERT_COST_PATH:-${default_cost_path}}"
if [[ ! -f "${expert_cost_path}" ]]; then
    echo "AWQ expert-cost artifact not found: ${expert_cost_path}" >&2
    echo "Run compute_expert_costs_awq.sh first." >&2
    exit 1
fi

save_args=()
if [[ -n "${ALLOCATION_OUTPUT_PATH:-}" ]]; then
    if [[ -e "${ALLOCATION_OUTPUT_PATH}" || -e "${ALLOCATION_OUTPUT_PATH%.pkl}.json" ]]; then
        echo "AWQ allocation output already exists and will not be overwritten." >&2
        exit 1
    fi
    save_args=(--save_path "${ALLOCATION_OUTPUT_PATH}")
fi

python -m gemq.allocate_bits \
    --model_name "${model_name}" \
    --allocation_metric expert_cost \
    --expert_cost_path "${expert_cost_path}" \
    --bit_budget "${bit_budget}" \
    --bit_candidates "${allocation_bits}" \
    --ilp_backend "${ILP_BACKEND:-highs}" \
    --extra_constr none \
    --max_prune_ratio "${max_prune_ratio}" \
    "${save_args[@]}"
