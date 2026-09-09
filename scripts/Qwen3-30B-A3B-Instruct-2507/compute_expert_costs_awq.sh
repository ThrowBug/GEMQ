#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${MODEL:-${model_name}}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
forward_batch_size="${FORWARD_BATCH_SIZE:-1}"
expert_batch_size="${EXPERT_BATCH_SIZE:-4096}"
groupsize="${GROUPSIZE:-128}"
candidate_bits="0,1,2,3"
average_bits="2"

c4_path="data/c4-train.00000-of-01024.json"
if [[ ! -f "${c4_path}" ]]; then
    echo "C4 calibration shard not found: ${c4_path}" >&2
    exit 1
fi

default_output="cache/${model_name}/ExpertCosts_c4-N${nsamples}-L${seqlen}-Seed${seed}_uniform2bit_B${candidate_bits}_Quant-awq-G${groupsize}.pt"
output_path="${OUTPUT_PATH:-${default_output}}"
if [[ -e "${output_path}" ]]; then
    echo "AWQ expert-cost output already exists and will not be overwritten: ${output_path}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.compute_expert_costs \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype bfloat16 \
    --attn_impl eager \
    --use_fast \
    --quantizer awq \
    --calib_dataset c4 \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --seed "${seed}" \
    --forward_batch_size "${forward_batch_size}" \
    --expert_batch_size "${expert_batch_size}" \
    --candidate_bits "${candidate_bits}" \
    --context_mode uniform_bit \
    --average_bits "${average_bits}" \
    --groupsize "${groupsize}" \
    --attn_wbits 4 \
    --dense_wbits 4 \
    --awq_scale_n_grid "${AWQ_SCALE_N_GRID:-20}" \
    --awq_clip_n_grid "${AWQ_CLIP_N_GRID:-20}" \
    --awq_clip_max_shrink "${AWQ_CLIP_MAX_SHRINK:-0.5}" \
    --awq_clip_n_sample_token "${AWQ_CLIP_N_SAMPLE_TOKEN:-512}" \
    --awq_search_batch_size "${AWQ_SEARCH_BATCH_SIZE:-1}" \
    --output_path "${output_path}"
