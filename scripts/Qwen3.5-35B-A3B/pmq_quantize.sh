#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_name="Qwen/Qwen3.5-35B-A3B"
bpe="${BITS_PER_EXPERT:-2.0}"
bpe_tag="$(printf '%.1f' "${bpe}")"
seed="${SEED:-0}"
bit_config="${PMQ_BIT_CONFIG:-configs/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B1,2,3.pkl}"
save_path="${PMQ_SAVE_PATH:-results/fake_quant_models/${model_name}/PMQ/C4-Seed${seed}_E${bpe_tag}_B1,2,3_LA${LINEAR_ATTN_WBITS:-4}-SA${SOFTMAX_ATTN_WBITS:-4}-G16-D${DENSE_WBITS:-4}}"

[[ -f "${bit_config}" ]] || {
    echo "Missing PMQ bit allocation: ${bit_config}" >&2
    exit 1
}
[[ ! -e "${save_path}" ]] || {
    echo "Output already exists and will not be overwritten: ${save_path}" >&2
    exit 1
}

MIXED_PREC=true \
ALLOCATION_METRIC=layer_re \
BIT_CONFIG="${bit_config}" \
WBITS=1,2,3 \
SAVE_PATH="${save_path}" \
bash "${script_dir}/quantize.sh"

cp "${bit_config}" "${save_path}/pmq_bit_allocation.pkl"
allocation_metadata="${bit_config%.pkl}.json"
if [[ -f "${allocation_metadata}" ]]; then
    cp "${allocation_metadata}" "${save_path}/pmq_allocation_metadata.json"
fi
