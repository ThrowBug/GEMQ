#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${MODEL:-${model_name}}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
gpus="${CUDA_VISIBLE_DEVICES:-0}"

calib_dataset="c4"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-1}"

candidate_bits="1,2,3"
bits_per_expert="${BITS_PER_EXPERT:-2.0}"
bpe_tag="$(printf '%.1f' "${bits_per_expert}")"
attn_wbits=4
gate_wbits=16
dense_wbits=4
groupsize="${GROUPSIZE:-128}"
blocksize="${BLOCKSIZE:-128}"
percdamp="${PERCDAMP:-0.01}"

bit_config="${PMQ_BIT_CONFIG:-configs/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B${candidate_bits}.pkl}"
save_path="${PMQ_SAVE_PATH:-results/fake_quant_models/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B${candidate_bits}_A${attn_wbits}-G${gate_wbits}-D${dense_wbits}}"

if [[ ! -f "${bit_config}" ]]; then
    echo "PMQ bit allocation not found: ${bit_config}" >&2
    echo "Run pmq_allocate.sh first or set PMQ_BIT_CONFIG." >&2
    exit 1
fi
if [[ ! -f "data/c4-train.00000-of-01024.json" ]]; then
    echo "C4 calibration shard not found: data/c4-train.00000-of-01024.json" >&2
    exit 1
fi

echo "=============================================="
echo ">>> PMQ fake quantization: Qwen3-30B-A3B-Instruct-2507"
echo " Model:          ${model}"
echo " Dataset:        ${calib_dataset}"
echo " Samples/length: ${nsamples}/${seqlen}"
echo " Seed:           ${seed}"
echo " Bit config:     ${bit_config}"
echo " Expert BPE:     ${bits_per_expert} (${candidate_bits})"
echo " Attention/gate: ${attn_wbits}/${gate_wbits} bit"
echo " GPTQ:           shared GEMQ implementation"
echo " Router FT:      disabled"
echo " Pruning:        disabled"
echo " Save dtype:     bfloat16"
echo " Output:         ${save_path}"
echo "=============================================="

# Intentionally omitted: --reproduce_mcmoe, --finetune_routers, --real_quant.
CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    --model "${model}" \
    --model_name "${model_name}" \
    --model_dtype "${model_dtype}" \
    --attn_impl "${attn_impl}" \
    --use_fast \
    --calib_dataset "${calib_dataset}" \
    --nsamples "${nsamples}" \
    --seqlen "${seqlen}" \
    --batch_size "${batch_size}" \
    --seed "${seed}" \
    --quantizer gptq \
    --mixed \
    --bit_cfg "${bit_config}" \
    --expert_wbits 2 \
    --attn_wbits "${attn_wbits}" \
    --gate_wbits "${gate_wbits}" \
    --dense_wbits "${dense_wbits}" \
    --groupsize "${groupsize}" \
    --blocksize "${blocksize}" \
    --percdamp "${percdamp}" \
    --mse \
    --save_path "${save_path}" \
    --save_dtype bfloat16

# Keep the source allocation with the otherwise standard Hugging Face checkpoint.
cp "${bit_config}" "${save_path}/pmq_bit_allocation.pkl"
allocation_metadata="${bit_config%.pkl}.json"
if [[ -f "${allocation_metadata}" ]]; then
    cp "${allocation_metadata}" "${save_path}/pmq_allocation_metadata.json"
fi
