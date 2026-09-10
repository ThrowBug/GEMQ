#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
candidate_bits="1,2,3"
bits_per_expert="${BITS_PER_EXPERT:-2.0}"
backend="${ILP_BACKEND:-highs}"

bpe_tag="$(printf '%.1f' "${bits_per_expert}")"
stats_dir="${PMQ_STATS_DIR:-cache/${model_name}/PMQ/C4-N${nsamples}-L${seqlen}-Seed${seed}_B${candidate_bits}}"
bit_config="${PMQ_BIT_CONFIG:-configs/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B${candidate_bits}.pkl}"

for stats_file in experts_act_counts.pkl experts_act_weights.pkl experts_quant_loss.pkl; do
    if [[ ! -f "${stats_dir}/${stats_file}" ]]; then
        echo "Required PMQ statistics file not found: ${stats_dir}/${stats_file}" >&2
        echo "Run pmq_compute_stats.sh first or set PMQ_STATS_DIR." >&2
        exit 1
    fi
done

echo "=============================================="
echo ">>> PMQ allocation: Qwen3-30B-A3B-Instruct-2507"
echo " Statistics:     ${stats_dir}"
echo " Candidate bits: ${candidate_bits}"
echo " Target BPE:     ${bits_per_expert}"
echo " ILP backend:    ${backend}"
echo " Output:         ${bit_config}"
echo "=============================================="

python -m gemq.allocate_pmq_bits \
    --model_name "${model_name}" \
    --stats_dir "${stats_dir}" \
    --save_path "${bit_config}" \
    --bits_per_expert "${bits_per_expert}" \
    --bit_candidates "${candidate_bits}" \
    --alpha 1 \
    --beta 1.5 \
    --gama 2 \
    --ilp_backend "${backend}"
