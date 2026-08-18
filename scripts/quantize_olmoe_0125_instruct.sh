#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

# ===============================
#  Model settings
# ===============================
model_name="allenai/OLMoE-1B-7B-0125-Instruct"
model="${model_name}"
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0}"

# ===============================
#  Dataset settings
# ===============================
calib_dataset="wikitext2"
nsamples=128
seqlen=2048

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe=2.0
mixed_prec=true
allocation_tag="C4-Seed0"
bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_E${bpe}_B1,2,3_c2c3.pkl"
reproduce_mcmoe=true

# ===============================
#  Router fine-tuning
# ===============================
finetune_routers=true
rft_trainer="layerwise_teacher" # legacy_ce | layerwise_teacher
rft_timing="after_each_layer_quantization" # after_all_quantization | after_each_layer_quantization
rft_router_loss="l2" # kd | kd_tail | l2 | l2_center
rft_router_alpha=0.0
rft_router_loss_weight=1.0
rft_output_kl_weight=0.0
rft_epochs=1
rft_batch_size=1
rft_lr=1e-4
rft_wd=0.0
rft_teacher_cache_dir="cache/router_finetune"
rft_rebuild_teacher_cache=false

# ===============================
#  Evaluation settings
# ===============================
eval_downstream=false
downstream_tasks="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mathqa,mmlu"

# ===============================
#  I/O settings
# ===============================
# This workflow intentionally saves dequantized approximate weights (W_hat).
# Packed integer checkpoints and GEMQ inference patches are not used by vLLM.
real_quant=false
save_model=true
save_dtype="bfloat16"

if [[ "${real_quant}" != "false" ]]; then
    echo "OLMoE-0125 vLLM workflow only supports real_quant=false." >&2
    exit 1
fi
if [[ "${mixed_prec}" == "true" && ! -f "${bit_cfg}" ]]; then
    echo "Bit allocation config not found: ${bit_cfg}" >&2
    echo "Run scripts/allocate_olmoe_0125_instruct.sh first." >&2
    exit 1
fi


# ===============================
#  AUTO argument construction
# ===============================
# OLMoE requires BF16 here because FP16 can produce NaNs.
model_args=(
    --model "${model}"
    --model_name "${model_name}"
    --use_fast
    --model_dtype "${model_dtype}"
)
data_args=(--calib_dataset "${calib_dataset}" --nsamples "${nsamples}" --seqlen "${seqlen}")

bpe_int=$(printf "%.0f" "${bpe}")
quant_args=(--quantizer "${quantizer}" --expert_wbits "${bpe_int}" --groupsize 128 --mse)
if [[ "${reproduce_mcmoe}" == "true" ]]; then
    quant_args+=(--reproduce_mcmoe)
fi
if [[ "${mixed_prec}" == "true" ]]; then
    qtype="$(basename "$(dirname "${bit_cfg}")")"
    quant_args+=(--mixed --bit_cfg "${bit_cfg}")
else
    qtype="Uniform"
fi

rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    if [[ "${rft_trainer}" == "legacy_ce" ]]; then
        # Keep the historical checkpoint path for the unchanged CE baseline.
        rft_tag="_RFT"
    else
        timing_tag="all"; [[ "${rft_timing}" == "after_each_layer_quantization" ]] && timing_tag="each"
        alpha_tag="${rft_router_alpha//./p}"
        router_weight_tag="${rft_router_loss_weight//./p}"
        output_weight_tag="${rft_output_kl_weight//./p}"
        rft_tag="_RFT-${timing_tag}-${rft_router_loss}-a${alpha_tag}-rw${router_weight_tag}-ow${output_weight_tag}"
    fi
    quant_args+=(
        --finetune_routers
        --rft_trainer "${rft_trainer}"
        --rft_timing "${rft_timing}"
        --rft_router_loss "${rft_router_loss}"
        --rft_router_alpha "${rft_router_alpha}"
        --rft_router_loss_weight "${rft_router_loss_weight}"
        --rft_output_kl_weight "${rft_output_kl_weight}"
        --rft_epochs "${rft_epochs}"
        --rft_batch_size "${rft_batch_size}"
        --rft_lr "${rft_lr}"
        --rft_wd "${rft_wd}"
        --rft_teacher_cache_dir "${rft_teacher_cache_dir}"
    )
    if [[ "${rft_rebuild_teacher_cache}" == "true" ]]; then
        quant_args+=(--rft_rebuild_teacher_cache)
    fi
fi

eval_args=()
if [[ "${eval_downstream}" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "${downstream_tasks}")
fi

prefix="${allocation_tag}-WT2"
if [[ "${save_model}" == "true" ]]; then
    save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
    io_args=(--save_path "${save_path}" --save_dtype "${save_dtype}")
else
    save_path="None"
    io_args=()
fi


# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> OLMoE-0125 fake-quant job"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Model dtype:      ${model_dtype}"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
echo " Quantizer:        ${quantizer}"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Finetune routers: ${finetune_routers} (trainer=${rft_trainer}, timing=${rft_timing})"
echo " Router objective: ${rft_router_loss} (alpha=${rft_router_alpha}, weight=${rft_router_loss_weight})"
echo " Output KL weight: ${rft_output_kl_weight}"
echo " RFT optimizer:    epochs=${rft_epochs}, batch=${rft_batch_size}, lr=${rft_lr}, wd=${rft_wd}"
echo " Real quant:       ${real_quant}"
echo " Save dtype:       ${save_dtype}"
echo " Save path:        ${save_path}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${io_args[@]}"
