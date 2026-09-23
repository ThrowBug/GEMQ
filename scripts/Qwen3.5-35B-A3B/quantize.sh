#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
model="${MODEL:-${model_name}}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-1}"
metric="${ALLOCATION_METRIC:-expert_cost}"
mixed_prec="${MIXED_PREC:-true}"
bpe="${BITS_PER_EXPERT:-2.0}"
bpe_tag="$(printf '%.1f' "${bpe}")"
expert_wbits="${EXPERT_WBITS:-2}"
linear_attn_wbits="${LINEAR_ATTN_WBITS:-4}"
softmax_attn_wbits="${SOFTMAX_ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"
groupsize="${GROUPSIZE:-128}"
blocksize="${BLOCKSIZE:-128}"
percdamp="${PERCDAMP:-0.01}"

case "${metric}" in
    expert_cost)
        alloc_bits="${WBITS:-0,2,3}"
        max_prune_ratio="${MAX_PRUNE_RATIO:-0.1}"
        context_bit="${EXPERT_COST_CONTEXT_BIT:-2}"
        allocation_tag="C4-Seed${seed}_Metric-expert_cost-Ctxuniform${context_bit}_E${bpe_tag}_B${alloc_bits}_Pmax${max_prune_ratio}_EqPrune"
        ;;
    layer_re)
        alloc_bits="${WBITS:-1,2,3}"
        extra_constr="${EXTRA_CONSTR:-c2c3}"
        allocation_tag="C4-Seed${seed}_Metric-layer_re_E${bpe_tag}_B${alloc_bits}_${extra_constr}"
        ;;
    *)
        echo "ALLOCATION_METRIC must be expert_cost or layer_re." >&2
        exit 1
        ;;
esac

bit_config="${BIT_CONFIG:-configs/${model_name}/GEMQ/${allocation_tag}.pkl}"
quant_tag="${allocation_tag}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}"
if [[ "${mixed_prec}" == "true" ]]; then
    [[ -f "${bit_config}" ]] || {
        echo "Missing bit allocation: ${bit_config}" >&2
        exit 1
    }
    mixed_args=(--mixed --bit_cfg "${bit_config}")
else
    quant_tag="C4-Seed${seed}_Uniform-E${expert_wbits}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}"
    mixed_args=()
fi

save_path="${SAVE_PATH:-results/fake_quant_models/${model_name}/GEMQ/${quant_tag}}"
[[ ! -e "${save_path}" ]] || {
    echo "Output already exists and will not be overwritten: ${save_path}" >&2
    exit 1
}
[[ -f data/c4-train.00000-of-01024.json ]] || {
    echo "Missing C4 shard: data/c4-train.00000-of-01024.json" >&2
    exit 1
}
(( nsamples % batch_size == 0 )) || {
    echo "NSAMPLES must be divisible by BATCH_SIZE." >&2
    exit 1
}

# Qwen3.5 support is intentionally fake-quant only. Routers, router gates,
# linear-attention auxiliary projections and non-target weights stay BF16.
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset c4 \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --batch_size "${batch_size}" \
    --seed "${seed}" \
    --quantizer gptq \
    "${mixed_args[@]}" \
    --expert_wbits "${expert_wbits}" \
    --attn_wbits 16 \
    --linear_attn_wbits "${linear_attn_wbits}" \
    --softmax_attn_wbits "${softmax_attn_wbits}" \
    --gate_wbits 16 \
    --dense_wbits "${dense_wbits}" \
    --groupsize "${groupsize}" \
    --blocksize "${blocksize}" \
    --percdamp "${percdamp}" \
    --mse \
    --save_path "${save_path}" \
    --save_dtype bfloat16

if [[ "${mixed_prec}" == "true" ]]; then
    cp "${bit_config}" "${save_path}/expert_bit_allocation.pkl"
    allocation_metadata="${bit_config%.pkl}.json"
    if [[ -f "${allocation_metadata}" ]]; then
        cp "${allocation_metadata}" "${save_path}/expert_allocation_metadata.json"
    fi
fi
