#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# ===============================
#  Model settings
# ===============================
model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
model="${model_name}"
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# ===============================
#  Dataset settings
# ===============================
calib_dataset="${CALIB_DATASET:-mixed_chat_en}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
calib_data_path="${CALIB_DATA_PATH:-}"
calib_path_args=()

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe="${BPE:-2.5}"
wbits="${WBITS:-1,2,3}"
extra_constr="${EXTRA_CONSTR:-c2c3}"
mixed_prec=true
allocation_tag="${ALLOCATION_TAG:-}"
if [[ -z "${allocation_tag}" ]]; then
    case "${calib_dataset}" in
        mixed_chat_en) allocation_tag="MixedChatEn-Seed${seed}" ;;
        c4) allocation_tag="C4-Seed${seed}" ;;
        math) allocation_tag="MATH-Seed${seed}" ;;
        math+c4) allocation_tag="MATH+C4-Seed${seed}" ;;
        *)
            echo "Cannot infer the allocation config tag for CALIB_DATASET=${calib_dataset}." >&2
            echo "Set ALLOCATION_TAG explicitly, for example ALLOCATION_TAG=C4-Seed0." >&2
            exit 1
            ;;
    esac
fi
bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_E${bpe}_B${wbits}_${extra_constr}.pkl"
# The legacy Qwen script warns that the MC-MoE GPTQ implementation can produce NaNs.
reproduce_mcmoe="${REPRODUCE_MCMOE:-false}"
attn_wbits="${ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"

# ===============================
#  Router fine-tuning
# ===============================
finetune_routers="${FINETUNE_ROUTERS:-true}"
rft_trainer="${RFT_TRAINER:-layerwise_teacher}" # legacy_ce | distill_ce | layerwise_teacher
rft_timing="${RFT_TIMING:-after_each_layer_quantization}" # after_all_quantization | after_each_layer_quantization
rft_router_loss="${RFT_ROUTER_LOSS:-kd_tail}" # kd | kd_tail | l2 | l2_center
rft_router_alpha="${RFT_ROUTER_ALPHA:-1.0}"
rft_router_loss_weight="${RFT_ROUTER_LOSS_WEIGHT:-1.0}"
rft_output_kl_weight="${RFT_OUTPUT_KL_WEIGHT:-0.0}"
rft_epochs="${RFT_EPOCHS:-1}"
rft_batch_size="${RFT_BATCH_SIZE:-1}"
rft_lr="${RFT_LR:-1e-4}"
rft_wd="${RFT_WD:-0.0}"
rft_teacher_cache_dir="${RFT_TEACHER_CACHE_DIR:-cache/router_finetune}"
rft_rebuild_teacher_cache="${RFT_REBUILD_TEACHER_CACHE:-false}"

# ===============================
#  Evaluation settings
# ===============================
eval_downstream="${EVAL_DOWNSTREAM:-false}"
downstream_tasks="${DOWNSTREAM_TASKS:-piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mathqa,mmlu}"

# ===============================
#  Diagnostics
# ===============================
cuda_diagnostics="${CUDA_DIAGNOSTICS:-false}"

# ===============================
#  I/O settings
# ===============================
# Save dequantized approximate weights (W_hat) as a standard BF16 checkpoint for vLLM.
real_quant=false
save_model=true
save_dtype="bfloat16"

if [[ "${calib_dataset}" == "mixed_chat_en" ]]; then
    if [[ -z "${calib_data_path}" ]]; then
        calib_data_path="cache/calibration/${model_name}/${calib_dataset}-N${nsamples}-L${seqlen}-Seed${seed}.pt"
    fi
    if [[ ! -f "${calib_data_path}" ]]; then
        echo "Prepared calibration data not found: ${calib_data_path}" >&2
        echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/prepare_calib.sh first." >&2
        exit 1
    fi
    calib_path_args=(--calib_data_path "${calib_data_path}")
fi
if [[ "${mixed_prec}" == "true" && ! -f "${bit_cfg}" ]]; then
    echo "Bit allocation config not found: ${bit_cfg}" >&2
    echo "Run scripts/Qwen3-30B-A3B-Instruct-2507/allocate.sh first." >&2
    exit 1
fi

# ===============================
#  AUTO argument construction
# ===============================
model_args=(
    --model "${model}"
    --model_name "${model_name}"
    --use_fast
    --model_dtype "${model_dtype}"
)
data_args=(
    --calib_dataset "${calib_dataset}"
    "${calib_path_args[@]}"
    --nsamples "${nsamples}"
    --seqlen "${seqlen}"
    --seed "${seed}"
)

bpe_int=$(printf "%.0f" "${bpe}")
quant_args=(--quantizer "${quantizer}" --expert_wbits "${bpe_int}" --groupsize 128 --mse --attn_wbits "${attn_wbits}" --dense_wbits "${dense_wbits}")
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
    quant_args+=(
        --finetune_routers
        --rft_trainer "${rft_trainer}"
        --rft_epochs "${rft_epochs}"
        --rft_batch_size "${rft_batch_size}"
        --rft_lr "${rft_lr}"
        --rft_wd "${rft_wd}"
    )
    case "${rft_trainer}" in
        legacy_ce)
            rft_tag="_RFT-legacy_ce"
            ;;
        distill_ce)
            rft_tag="_RFT-distill_ce"
            quant_args+=(--rft_teacher_cache_dir "${rft_teacher_cache_dir}")
            ;;
        layerwise_teacher)
            timing_tag="all"; [[ "${rft_timing}" == "after_each_layer_quantization" ]] && timing_tag="each"
            alpha_tag="${rft_router_alpha//./p}"
            router_weight_tag="${rft_router_loss_weight//./p}"
            output_weight_tag="${rft_output_kl_weight//./p}"
            rft_tag="_RFT-${timing_tag}-${rft_router_loss}-a${alpha_tag}-rw${router_weight_tag}-ow${output_weight_tag}"
            quant_args+=(
                --rft_timing "${rft_timing}"
                --rft_router_loss "${rft_router_loss}"
                --rft_router_alpha "${rft_router_alpha}"
                --rft_router_loss_weight "${rft_router_loss_weight}"
                --rft_output_kl_weight "${rft_output_kl_weight}"
                --rft_teacher_cache_dir "${rft_teacher_cache_dir}"
            )
            ;;
        *)
            echo "Unsupported rft_trainer: ${rft_trainer}" >&2
            exit 1
            ;;
    esac
    if [[ "${rft_trainer}" != "legacy_ce" && "${rft_rebuild_teacher_cache}" == "true" ]]; then
        quant_args+=(--rft_rebuild_teacher_cache)
    fi
fi

eval_args=()
if [[ "${eval_downstream}" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "${downstream_tasks}")
fi

diagnostic_args=()
if [[ "${cuda_diagnostics}" == "true" ]]; then
    diagnostic_args=(--cuda_diagnostics)
fi

prefix="${allocation_tag}"
if [[ "${save_model}" == "true" ]]; then
    save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A${attn_wbits}-G16-D${dense_wbits}-E${bpe}${rft_tag}"
    io_args=(--save_path "${save_path}" --save_dtype "${save_dtype}")
else
    save_path="None"
    io_args=()
fi

# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> Qwen3-30B-A3B-Instruct-2507 fake-quant job"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Model dtype:      ${model_dtype}"
echo " CUDA devices:     ${gpus}"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
if [[ "${calib_dataset}" == "mixed_chat_en" ]]; then
    echo " Calibration file: ${calib_data_path}"
else
    echo " Calibration file: legacy ${calib_dataset} loader"
fi
echo " Quantizer:        ${quantizer} (reproduce_mcmoe=${reproduce_mcmoe})"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Attn wbits:       ${attn_wbits}"
echo " Dense wbits:      ${dense_wbits}"
echo " Finetune routers: ${finetune_routers} (trainer=${rft_trainer})"
if [[ "${rft_trainer}" == "legacy_ce" ]]; then
    echo " Router objective: hard-label autoregressive CE"
elif [[ "${rft_trainer}" == "distill_ce" ]]; then
    echo " Router objective: teacher soft-label autoregressive CE"
else
    echo " RFT timing:       ${rft_timing}"
    echo " Router objective: ${rft_router_loss} (alpha=${rft_router_alpha}, weight=${rft_router_loss_weight})"
    echo " Output KL weight: ${rft_output_kl_weight}"
fi
echo " RFT optimizer:    epochs=${rft_epochs}, batch=${rft_batch_size}, lr=${rft_lr}, wd=${rft_wd}"
echo " Real quant:       ${real_quant}"
echo " Save dtype:       ${save_dtype}"
echo " Save path:        ${save_path}"
echo " CUDA diagnostics: ${cuda_diagnostics}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${diagnostic_args[@]}" \
    "${io_args[@]}"
