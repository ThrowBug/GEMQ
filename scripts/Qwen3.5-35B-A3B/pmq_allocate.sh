#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
bpe="${BITS_PER_EXPERT:-2.0}"
bpe_tag="$(printf '%.1f' "${bpe}")"
candidate_bits="1,2,3"
stats_dir="${PMQ_STATS_DIR:-cache/${model_name}/PMQ/C4-N${nsamples}-L${seqlen}-Seed${seed}_B${candidate_bits}}"
bit_config="${PMQ_BIT_CONFIG:-configs/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B${candidate_bits}.pkl}"

for file in experts_act_counts.pkl experts_act_weights.pkl experts_quant_loss.pkl; do
    [[ -f "${stats_dir}/${file}" ]] || {
        echo "Missing PMQ statistics: ${stats_dir}/${file}" >&2
        exit 1
    }
done

python -m gemq.allocate_pmq_bits \
    --model_name "${model_name}" \
    --stats_dir "${stats_dir}" \
    --save_path "${bit_config}" \
    --bits_per_expert "${bpe}" \
    --bit_candidates "${candidate_bits}" \
    --alpha 1 \
    --beta 1.5 \
    --gama 2 \
    --ilp_backend "${ILP_BACKEND:-highs}"
