#!/bin/bash
set -euo pipefail

# NOTE: these must match the settings scripts/quantize_olmoe.sh ran with, otherwise the
# derived model_path below will not exist.
model_name="allenai/OLMoE-1B-7B-0924"
bpe=2.0
finetune_routers=true

prompt="Although the experiment failed repeatedly, the researchers eventually"
max_new_tokens=200


rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    rft_tag="_RFT"
fi
model_path="results/real_quant_models/${model_name}/GEMQ/C4-Seed0-WT2_A4-G16-D4-E${bpe}${rft_tag}"

if [[ ! -d "$model_path" ]]; then
    echo "Checkpoint not found: $model_path"
    echo "Check that bpe=${bpe} and finetune_routers=${finetune_routers} match scripts/quantize_olmoe.sh."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 TORCH_LOGS="graph_breaks,recompiles" python -m gemq.benchmark_generate \
    --model_path $model_path \
    --model_name $model_name \
    --attn_impl eager \
    --prompt "$prompt" \
    --num_samples 10 \
    --max_new_tokens $max_new_tokens \
    --top_k 200 \
    --compile
