#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
metric="${ALLOCATION_METRIC:-expert_cost}"
bpe="${BITS_PER_EXPERT:-2.0}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
max_prune_ratio="${MAX_PRUNE_RATIO:-0.1}"
backend="${ILP_BACKEND:-highs}"
save_path="${ALLOCATION_OUTPUT_PATH:-}"

case "${metric}" in
    expert_cost)
        bits="${WBITS:-0,2,3}"
        cost_bits="${EXPERT_COST_BITS:-0,1,2,3}"
        context_bit="${EXPERT_COST_CONTEXT_BIT:-2}"
        stat_path="${EXPERT_COST_PATH:-cache/${model_name}/ExpertCosts_c4-N${nsamples}-L${seqlen}-Seed${seed}_uniform${context_bit}bit_B${cost_bits}.pt}"
        stat_args=(--expert_cost_path "${stat_path}")
        extra_constr=none
        ;;
    layer_re)
        bits="${WBITS:-1,2,3}"
        stat_path="${LAYER_RE_PATH:-cache/${model_name}/LayerRE_c4-N${nsamples}-L${seqlen}-Seed${seed}_B${bits}_fast.pkl}"
        stat_args=(--layer_re_path "${stat_path}")
        extra_constr="${EXTRA_CONSTR:-c2c3}"
        ;;
    *)
        echo "ALLOCATION_METRIC must be expert_cost or layer_re." >&2
        exit 1
        ;;
esac

[[ -f "${stat_path}" ]] || {
    echo "Missing statistics artifact: ${stat_path}" >&2
    exit 1
}
save_args=()
[[ -z "${save_path}" ]] || save_args=(--save_path "${save_path}")

python -m gemq.allocate_bits \
    --model_name "${model_name}" \
    --allocation_metric "${metric}" \
    "${stat_args[@]}" \
    --bit_budget "${bpe}" \
    --bit_candidates "${bits}" \
    --ilp_solver gemq \
    --ilp_backend "${backend}" \
    --extra_constr "${extra_constr}" \
    --max_prune_ratio "${max_prune_ratio}" \
    "${save_args[@]}"
