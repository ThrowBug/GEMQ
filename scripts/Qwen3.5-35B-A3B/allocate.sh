#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
allocation_metric="${ALLOCATION_METRIC:-pop}" # pop | layer_re
normalize_by_activation_count="${NORMALIZE_BY_ACTIVATION_COUNT:-true}"
bits_per_expert="${BITS_PER_EXPERT:-2.0}"
if [[ -n "${WBITS:-}" ]]; then
    wbits="${WBITS}"
elif [[ "${allocation_metric}" == "pop" ]]; then
    wbits="0,2,3"
else
    wbits="1,2,3"
fi
ilp_backend="${ILP_BACKEND:-highs}"
max_prune_ratio="${MAX_PRUNE_RATIO:-0.25}"
calib_dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
output_path="${ALLOCATION_OUTPUT_PATH:-}"

stat_args=()
case "${allocation_metric}" in
    pop)
        if [[ "${normalize_by_activation_count}" != "true" && "${normalize_by_activation_count}" != "false" ]]; then
            echo "NORMALIZE_BY_ACTIVATION_COUNT must be 'true' or 'false'." >&2
            exit 1
        fi
        extra_constr="${EXTRA_CONSTR:-none}"
        if [[ "${extra_constr}" != "none" ]]; then
            echo "EXTRA_CONSTR is only valid with ALLOCATION_METRIC=layer_re." >&2
            exit 1
        fi
        expert_cost_bits="${POP_COST_BITS:-${CANDIDATE_BITS:-0,1,2,3}}"
        expert_cost_context_bit="${POP_CONTEXT_BIT:-${AVERAGE_BITS:-2}}"
        default_expert_cost_path="cache/${model_name}/PopCosts_${calib_dataset}-N${nsamples}-L${seqlen}-Seed${seed}_uniform${expert_cost_context_bit}bit_B${expert_cost_bits}.pt"
        if [[ "${normalize_by_activation_count}" == "false" ]]; then
            default_expert_cost_path="${default_expert_cost_path%.pt}_NoCountNorm.pt"
        fi
        expert_cost_path="${POP_COST_PATH:-${default_expert_cost_path}}"
        if [[ ! -f "${expert_cost_path}" ]]; then
            echo "Expert-cost artifact not found: ${expert_cost_path}" >&2
            echo "Run scripts/Qwen3.5-35B-A3B/compute_pop_costs.sh first." >&2
            exit 1
        fi
        stat_args=(--pop_cost_path "${expert_cost_path}")
        ;;
    layer_re)
        extra_constr="${EXTRA_CONSTR:-c2c3}"
        default_layer_re_path="cache/${model_name}/LayerRE_${calib_dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}_fast.pkl"
        layer_re_path="${LAYER_RE_PATH:-${default_layer_re_path}}"
        if [[ ! -f "${layer_re_path}" ]]; then
            echo "Layer reconstruction statistics not found: ${layer_re_path}" >&2
            echo "Run scripts/Qwen3.5-35B-A3B/compute_stats.sh first." >&2
            exit 1
        fi
        stat_args=(--layer_re_path "${layer_re_path}")
        ;;
    *)
        echo "ALLOCATION_METRIC must be pop or layer_re, got: ${allocation_metric}" >&2
        exit 1
        ;;
esac

save_args=()
if [[ -n "${output_path}" ]]; then
    save_args=(--save_path "${output_path}")
fi

python -m proquant.allocate_bits \
    --model_name "${model_name}" \
    --allocation_metric "${allocation_metric}" \
    "${stat_args[@]}" \
    --bit_budget "${bits_per_expert}" \
    --bit_candidates "${wbits}" \
    --ilp_backend "${ilp_backend}" \
    --extra_constr "${extra_constr}" \
    --max_prune_ratio "${max_prune_ratio}" \
    "${save_args[@]}"
