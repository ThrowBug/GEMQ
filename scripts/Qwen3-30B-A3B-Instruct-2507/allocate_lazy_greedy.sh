#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../.."

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
seed="${SEED:-0}"
cost_bits="${POP_COST_BITS:-${CANDIDATE_BITS:-0,1,2,3}}"
context_bit="${POP_CONTEXT_BIT:-${AVERAGE_BITS:-2}}"
default_cost_path="cache/${model_name}/PopCosts_c4-N${NSAMPLES:-128}-L${SEQLEN:-2048}-Seed${seed}_uniform${context_bit}bit_B${cost_bits}.pt"
case "${NORMALIZE_BY_ACTIVATION_COUNT:-true}" in
    true) ;;
    false) default_cost_path="${default_cost_path%.pt}_NoCountNorm.pt" ;;
    *) echo "NORMALIZE_BY_ACTIVATION_COUNT must be true or false." >&2; exit 1 ;;
esac
cost_path="${POP_COST_PATH:-${default_cost_path}}"
if [[ ! -f "${cost_path}" ]]; then
    echo "Expert-cost artifact not found: ${cost_path}. Set POP_COST_PATH or run compute_pop_costs.sh." >&2
    exit 1
fi

output_args=()
if [[ -n "${ALLOCATION_OUTPUT_PATH:-}" ]]; then
    output_args=(--save_path "${ALLOCATION_OUTPUT_PATH}")
fi

exec python -m proquant.allocate_lazy_greedy \
    --model "${MODEL:-${model_name}}" \
    --pop_cost_path "${cost_path}" \
    --bit_budget "${BITS_PER_EXPERT:-2.0}" \
    --max_prune_ratio "${MAX_PRUNE_RATIO:-0.3}" \
    --ilp_backend "${ILP_BACKEND:-highs}" \
    --search_nsamples "${SEARCH_NSAMPLES:-4}" \
    --reference_nsamples "${SEARCH_REFERENCE_NSAMPLES:-32}" \
    --seqlen "${SEARCH_SEQLEN:-512}" \
    --seed "${SEARCH_SEED:-${seed}}" \
    --budget_step "${SEARCH_BUDGET_STEP:-16}" \
    --attn_wbits "${ATTN_WBITS:-4}" \
    --scratch_dir "${SEARCH_SCRATCH_DIR:-cache/lazy_greedy}" \
    "${output_args[@]}" "$@"
