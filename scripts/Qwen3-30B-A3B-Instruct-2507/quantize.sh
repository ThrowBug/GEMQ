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
allocation_metric="${ALLOCATION_METRIC:-expert_cost}" # expert_cost | layer_re
bpe="${BPE:-2.0}"
if [[ -n "${WBITS:-}" ]]; then
    wbits="${WBITS}"
elif [[ "${allocation_metric}" == "expert_cost" ]]; then
    wbits="0,2,3"
else
    wbits="1,2,3"
fi
max_prune_ratio="${MAX_PRUNE_RATIO:-0.25}"
expert_cost_context_bit="${EXPERT_COST_CONTEXT_BIT:-2}"
mixed_prec="${MIXED_PREC:-true}"
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
bpe_tag="$(printf '%.1f' "${bpe}")"
if [[ "${allocation_metric}" == "expert_cost" ]]; then
    extra_constr="${EXTRA_CONSTR:-none}"
    if [[ "${extra_constr}" != "none" ]]; then
        echo "EXTRA_CONSTR is only valid with ALLOCATION_METRIC=layer_re." >&2
        exit 1
    fi
    default_bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_Metric-expert_cost-Ctxuniform${expert_cost_context_bit}_E${bpe_tag}_B${wbits}"
    if [[ ",${wbits}," == *,0,* ]]; then
        max_prune_ratio_tag="$(printf '%g' "${max_prune_ratio}")"
        default_bit_cfg+="_Pmax${max_prune_ratio_tag}_EqPrune"
    fi
    default_bit_cfg+=".pkl"
elif [[ "${allocation_metric}" == "layer_re" ]]; then
    extra_constr="${EXTRA_CONSTR:-c2c3}"
    constraint_tag=""; [[ "${extra_constr}" != "none" ]] && constraint_tag="_${extra_constr}"
    default_bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_Metric-layer_re_E${bpe_tag}_B${wbits}${constraint_tag}.pkl"
else
    echo "ALLOCATION_METRIC must be expert_cost or layer_re, got: ${allocation_metric}" >&2
    exit 1
fi
bit_cfg="${BIT_CFG_PATH:-${default_bit_cfg}}"
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
rft_transfer_loss="${RFT_TRANSFER_LOSS:-none}" # none | prob_corr_kl
rft_transfer_weight="${RFT_TRANSFER_WEIGHT:-1.0}"
rft_transfer_anneal_ratio="${RFT_TRANSFER_ANNEAL_RATIO:-0.2}"
rft_transfer_temperature="${RFT_TRANSFER_TEMPERATURE:-0.2}"

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
save_gptq_checkpoint="${SAVE_GPTQ_CHECKPOINT:-false}"
load_gptq_checkpoint="${LOAD_GPTQ_CHECKPOINT:-false}"
gptq_checkpoint_root="${GPTQ_CHECKPOINT_ROOT:-results/gptq_checkpoints}"

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
if [[ "${rft_transfer_loss}" != "none" && ( "${finetune_routers}" != "true" || "${rft_trainer}" != "distill_ce" ) ]]; then
    echo "RFT_TRANSFER_LOSS is supported only with FINETUNE_ROUTERS=true and RFT_TRAINER=distill_ce." >&2
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
    bit_cfg_filename="$(basename "${bit_cfg}")"
    quant_allocation_tag="${bit_cfg_filename%.pkl}"
    quant_args+=(--mixed --bit_cfg "${bit_cfg}")
else
    qtype="Uniform"
    quant_allocation_tag="${allocation_tag}"
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
            quant_args+=(
                --rft_teacher_cache_dir "${rft_teacher_cache_dir}"
                --rft_transfer_loss "${rft_transfer_loss}"
                --rft_transfer_weight "${rft_transfer_weight}"
                --rft_transfer_anneal_ratio "${rft_transfer_anneal_ratio}"
                --rft_transfer_temperature "${rft_transfer_temperature}"
            )
            if [[ "${rft_transfer_loss}" != "none" ]]; then
                transfer_weight_tag="${rft_transfer_weight//./p}"
                transfer_anneal_tag="${rft_transfer_anneal_ratio//./p}"
                transfer_temperature_tag="${rft_transfer_temperature//./p}"
                rft_tag+="-transfer_${rft_transfer_loss}-w${transfer_weight_tag}-a${transfer_anneal_tag}-t${transfer_temperature_tag}"
            fi
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

prefix="${quant_allocation_tag}"
gptq_checkpoint_path="${gptq_checkpoint_root}/${model_name}/${qtype}/${prefix}_A${attn_wbits}-G16-D${dense_wbits}-E${bpe}"
checkpoint_args=()
if [[ "${save_gptq_checkpoint}" == "true" && "${load_gptq_checkpoint}" == "true" ]]; then
    echo "SAVE_GPTQ_CHECKPOINT and LOAD_GPTQ_CHECKPOINT cannot both be true." >&2
    exit 1
elif [[ "${save_gptq_checkpoint}" == "true" ]]; then
    if [[ -e "${gptq_checkpoint_path}" ]]; then
        echo "GPTQ checkpoint already exists and will not be overwritten: ${gptq_checkpoint_path}" >&2
        echo "Set LOAD_GPTQ_CHECKPOINT=true to reuse it." >&2
        exit 1
    fi
    checkpoint_args=(--save_gptq_checkpoint --gptq_checkpoint_path "${gptq_checkpoint_path}")
elif [[ "${load_gptq_checkpoint}" == "true" ]]; then
    if [[ ! -f "${gptq_checkpoint_path}/_SUCCESS" ]]; then
        echo "Complete GPTQ checkpoint not found: ${gptq_checkpoint_path}" >&2
        exit 1
    fi
    checkpoint_args=(--load_gptq_checkpoint --gptq_checkpoint_path "${gptq_checkpoint_path}")
fi

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
echo " Allocation metric:${allocation_metric}"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Allocation tag:   ${quant_allocation_tag}"
echo " Attn wbits:       ${attn_wbits}"
echo " Dense wbits:      ${dense_wbits}"
echo " Finetune routers: ${finetune_routers} (trainer=${rft_trainer})"
if [[ "${rft_trainer}" == "legacy_ce" ]]; then
    echo " Router objective: hard-label autoregressive CE"
elif [[ "${rft_trainer}" == "distill_ce" ]]; then
    echo " Router objective: teacher soft-label autoregressive CE"
    echo " Transfer loss:    ${rft_transfer_loss} (weight=${rft_transfer_weight}, anneal=${rft_transfer_anneal_ratio}, temperature=${rft_transfer_temperature})"
else
    echo " RFT timing:       ${rft_timing}"
    echo " Router objective: ${rft_router_loss} (alpha=${rft_router_alpha}, weight=${rft_router_loss_weight})"
    echo " Output KL weight: ${rft_output_kl_weight}"
fi
echo " RFT optimizer:    epochs=${rft_epochs}, batch=${rft_batch_size}, lr=${rft_lr}, wd=${rft_wd}"
echo " Real quant:       ${real_quant}"
echo " Save dtype:       ${save_dtype}"
echo " Save path:        ${save_path}"
echo " Save GPTQ ckpt:   ${save_gptq_checkpoint}"
echo " Load GPTQ ckpt:   ${load_gptq_checkpoint}"
if [[ "${save_gptq_checkpoint}" == "true" || "${load_gptq_checkpoint}" == "true" ]]; then
    echo " GPTQ ckpt path:   ${gptq_checkpoint_path}"
fi
echo " CUDA diagnostics: ${cuda_diagnostics}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${gpus}" python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${diagnostic_args[@]}" \
    "${checkpoint_args[@]}" \
    "${io_args[@]}"
