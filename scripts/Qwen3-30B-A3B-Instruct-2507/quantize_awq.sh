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
groupsize="${GROUPSIZE:-128}"
allocation_bits="${BIT_CANDIDATES:-0,2,3}"
mixed_prec="${MIXED_PREC:-true}"
expert_wbits="${EXPERT_WBITS:-2}"

case "${mixed_prec}" in
    true|false) ;;
    *)
        echo "MIXED_PREC must be true or false, got: ${mixed_prec}" >&2
        exit 1
        ;;
esac

c4_path="data/c4-train.00000-of-01024.json"
if [[ ! -f "${c4_path}" ]]; then
    echo "C4 calibration shard not found: ${c4_path}" >&2
    exit 1
fi

precision_args=(--expert_wbits "${expert_wbits}")
if [[ "${mixed_prec}" == "true" ]]; then
    quantization_name="C4-Seed${seed}_Metric-expert_cost-Quant-awq-Ctxuniform2_E2.0_B${allocation_bits}_Pmax0.1_EqPrune"
    output_name="${quantization_name}_A4-G16-D4-E2.0"
    default_bit_cfg="configs/${model_name}/GEMQ/${quantization_name}.pkl"
    bit_cfg="${BIT_CFG_PATH:-${default_bit_cfg}}"
    if [[ ! -f "${bit_cfg}" || ! -f "${bit_cfg%.pkl}.json" ]]; then
        echo "Complete AWQ allocation not found: ${bit_cfg}" >&2
        echo "Run allocate_awq.sh first." >&2
        exit 1
    fi
    precision_args+=(--mixed --bit_cfg "${bit_cfg}")
else
    quantization_name="C4-Seed${seed}_UniformE${expert_wbits}"
    output_name="${quantization_name}_A4-G16-D4"
fi

default_save_path="results/fake_quant_models/${model_name}/GEMQ-AWQ/${output_name}"
save_path="${SAVE_PATH:-${default_save_path}}"
if [[ -e "${save_path}" ]]; then
    echo "AWQ model output already exists and will not be overwritten: ${save_path}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype bfloat16 \
    --attn_impl eager \
    --use_fast \
    --calib_dataset c4 \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --batch_size "${FORWARD_BATCH_SIZE:-1}" \
    --seed "${seed}" \
    --quantizer awq \
    "${precision_args[@]}" \
    --groupsize "${groupsize}" \
    --attn_wbits 4 \
    --dense_wbits 4 \
    --max_prune_ratio 0.1 \
    --expert_batch_size "${EXPERT_BATCH_SIZE:-4096}" \
    --awq_scale_n_grid "${AWQ_SCALE_N_GRID:-20}" \
    --awq_clip_n_grid "${AWQ_CLIP_N_GRID:-20}" \
    --awq_clip_max_shrink "${AWQ_CLIP_MAX_SHRINK:-0.5}" \
    --awq_clip_n_sample_token "${AWQ_CLIP_N_SAMPLE_TOKEN:-512}" \
    --awq_search_batch_size "${AWQ_SEARCH_BATCH_SIZE:-1}" \
    --save_path "${save_path}" \
    --save_dtype bfloat16
