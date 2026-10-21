#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

# ===============================
#  Model settings
# ===============================
model_name="Qwen/Qwen3.5-35B-A3B"
model="${MODEL:-${model_name}}"
model_dtype="bfloat16"
gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# ===============================
#  Dataset settings
# ===============================
calib_dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
allocation_metric="${ALLOCATION_METRIC:-pop}" # pop | layer_re
normalize_by_activation_count="${NORMALIZE_BY_ACTIVATION_COUNT:-true}"
bpe="${BPE:-2.0}"
if [[ -n "${WBITS:-}" ]]; then
    wbits="${WBITS}"
elif [[ "${allocation_metric}" == "pop" ]]; then
    wbits="0,2,3"
else
    wbits="1,2,3"
fi
max_prune_ratio="${MAX_PRUNE_RATIO:-0.25}"
expert_cost_context_bit="${POP_CONTEXT_BIT:-${AVERAGE_BITS:-2}}"
if [[ -n "${MIXED_PREC+x}" ]]; then
    echo "MIXED_PREC is no longer supported; quantization always requires a bit-allocation file." >&2
    exit 1
fi
allocation_tag="${ALLOCATION_TAG:-}"
if [[ -z "${allocation_tag}" ]]; then
    case "${calib_dataset}" in
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
if [[ "${allocation_metric}" == "pop" ]]; then
    if [[ "${normalize_by_activation_count}" != "true" && "${normalize_by_activation_count}" != "false" ]]; then
        echo "NORMALIZE_BY_ACTIVATION_COUNT must be 'true' or 'false'." >&2
        exit 1
    fi
    extra_constr="${EXTRA_CONSTR:-none}"
    if [[ "${extra_constr}" != "none" ]]; then
        echo "EXTRA_CONSTR is only valid with ALLOCATION_METRIC=layer_re." >&2
        exit 1
    fi
    cost_norm_tag=""
    if [[ "${normalize_by_activation_count}" == "false" ]]; then
        cost_norm_tag="-NoCountNorm"
    fi
    default_bit_cfg="configs/${model_name}/ProQuant/${allocation_tag}_Metric-pop-Ctxuniform${expert_cost_context_bit}${cost_norm_tag}_E${bpe_tag}_B${wbits}"
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
    echo "ALLOCATION_METRIC must be pop or layer_re, got: ${allocation_metric}" >&2
    exit 1
fi
bit_cfg="${BIT_CFG_PATH:-${default_bit_cfg}}"
if [[ ! -f "${bit_cfg}" ]]; then
    echo "Bit allocation config not found: ${bit_cfg}" >&2
    echo "Run scripts/Qwen3.5-35B-A3B/allocate.sh first or set BIT_CFG_PATH." >&2
    exit 1
fi
# Optional compatibility switch for the inherited MC-MoE GPTQ implementation.
reproduce_mcmoe="${REPRODUCE_MCMOE:-false}"
linear_attn_wbits="${LINEAR_ATTN_WBITS:-4}"
softmax_attn_wbits="${SOFTMAX_ATTN_WBITS:-4}"
dense_wbits="${DENSE_WBITS:-4}"

# ===============================
#  Post-quantization fine-tuning
# ===============================
finetune="${FINETUNE:-true}"
rft_trainer="${RFT_TRAINER:-roco}" # roco | norm_ft | router_ft
rft_epochs="${RFT_EPOCHS:-1}"
rft_batch_size="${RFT_BATCH_SIZE:-1}"
rft_lr="${RFT_LR:-1e-4}"
rft_lr_tag="$(python -c 'import sys; from decimal import Decimal; print(format(Decimal(sys.argv[1]).normalize(), "E").lower().replace("e+", "e"))' "${rft_lr}")"
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
save_gptq_checkpoint="${SAVE_GPTQ_CHECKPOINT:-false}"
load_gptq_checkpoint="${LOAD_GPTQ_CHECKPOINT:-false}"
gptq_checkpoint_root="${GPTQ_CHECKPOINT_ROOT:-results/gptq_checkpoints}"

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
    --nsamples "${nsamples}"
    --seqlen "${seqlen}"
    --seed "${seed}"
)

bpe_int=$(printf "%.0f" "${bpe}")
quant_args=(--quantizer "${quantizer}" --expert_wbits "${bpe_int}" --groupsize 128 --mse --linear_attn_wbits "${linear_attn_wbits}" --softmax_attn_wbits "${softmax_attn_wbits}" --dense_wbits "${dense_wbits}")
if [[ "${reproduce_mcmoe}" == "true" ]]; then
    quant_args+=(--reproduce_mcmoe)
fi
qtype="$(basename "$(dirname "${bit_cfg}")")"
bit_cfg_filename="$(basename "${bit_cfg}")"
quant_allocation_tag="${bit_cfg_filename%.pkl}"
quant_args+=(--bit_cfg "${bit_cfg}")

rft_tag=""
if [[ "${finetune}" == "true" ]]; then
    quant_args+=(
        --finetune
        --rft_trainer "${rft_trainer}"
        --rft_epochs "${rft_epochs}"
        --rft_batch_size "${rft_batch_size}"
        --rft_lr "${rft_lr}"
        --rft_wd "${rft_wd}"
    )
    case "${rft_trainer}" in
        router_ft)
            rft_tag="_RFT-router_ft"
            ;;
        norm_ft|roco)
            rft_tag="_RFT-${rft_trainer}-lr${rft_lr_tag}"
            quant_args+=(--rft_teacher_cache_dir "${rft_teacher_cache_dir}")
            ;;
        *)
            echo "Unsupported rft_trainer: ${rft_trainer}" >&2
            exit 1
            ;;
    esac
    if [[ "${rft_trainer}" != "router_ft" && "${rft_rebuild_teacher_cache}" == "true" ]]; then
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
gptq_checkpoint_path="${GPTQ_CHECKPOINT_PATH:-${gptq_checkpoint_root}/${model_name}/${qtype}/${prefix}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}-E${bpe}}"
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
    save_path="${SAVE_PATH:-results/fake_quant_models/${model_name}/${qtype}/${prefix}_LA${linear_attn_wbits}-SA${softmax_attn_wbits}-G16-D${dense_wbits}-E${bpe}${rft_tag}}"
    case "${rft_trainer}" in
        norm_ft|roco)
            if [[ "${finetune}" == "true" && -e "${save_path}" ]]; then
                echo "Norm-distilled model output already exists and will not be overwritten: ${save_path}" >&2
                exit 1
            fi
            ;;
    esac
    io_args=(--save_path "${save_path}" --save_dtype "${save_dtype}")
else
    save_path="None"
    io_args=()
fi

# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> Qwen3.5-35B-A3B fake-quant job"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Model dtype:      ${model_dtype}"
echo " CUDA devices:     ${gpus}"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
echo " Calibration loader: ${calib_dataset}"
echo " Quantizer:        ${quantizer} (reproduce_mcmoe=${reproduce_mcmoe})"
echo " Allocation metric:${allocation_metric}"
echo " Expert budget:    ${bpe} bits"
echo " Bit config:       ${bit_cfg}"
echo " Allocation tag:   ${quant_allocation_tag}"
echo " Attention bits:   linear=${linear_attn_wbits}, softmax=${softmax_attn_wbits}"
echo " Dense wbits:      ${dense_wbits}"
echo " Fine-tuning:      ${finetune} (trainer=${rft_trainer})"
if [[ "${finetune}" == "true" && "${rft_trainer}" == "router_ft" ]]; then
    echo " Router objective: hard-label autoregressive CE"
elif [[ "${finetune}" == "true" && "${rft_trainer}" == "norm_ft" ]]; then
    echo " Norm parameters: input + post-attention"
    echo " Norm objective:   teacher soft-label autoregressive CE"
    echo " Router update:    none"
elif [[ "${finetune}" == "true" && "${rft_trainer}" == "roco" ]]; then
    echo " Norm parameters: input + post-attention"
    echo " Norm objective:   teacher soft-label autoregressive CE"
    echo " Router update:    inverse scale folding only (no optimizer)"
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

CUDA_VISIBLE_DEVICES="${gpus}" python -m proquant.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${diagnostic_args[@]}" \
    "${checkpoint_args[@]}" \
    "${io_args[@]}"
