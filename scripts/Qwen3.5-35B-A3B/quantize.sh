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

# Qwen3.5 supports the original hard-label CE router baseline and the two
# dual-norm distilled modes. Legacy CE updates only the routed ``mlp.gate``;
# the compensated norm mode inverse-folds its scale into both MoE gates.
finetune_routers="${FINETUNE_ROUTERS:-false}"
rft_trainer="${RFT_TRAINER:-router_compensated_dual_norm_distill}"
rft_epochs="${RFT_EPOCHS:-1}"
rft_batch_size="${RFT_BATCH_SIZE:-1}"
rft_lr="${RFT_LR:-1e-4}"
rft_wd="${RFT_WD:-0.0}"
rft_teacher_cache_dir="${RFT_TEACHER_CACHE_DIR:-cache/router_finetune}"
rft_rebuild_teacher_cache="${RFT_REBUILD_TEACHER_CACHE:-false}"

save_gptq_checkpoint="${SAVE_GPTQ_CHECKPOINT:-false}"
load_gptq_checkpoint="${LOAD_GPTQ_CHECKPOINT:-false}"
gptq_checkpoint_root="${GPTQ_CHECKPOINT_ROOT:-results/gptq_checkpoints}"

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

rft_args=()
rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    case "${rft_trainer}" in
        legacy_ce|dual_norm_distill|router_compensated_dual_norm_distill)
            ;;
        *)
            echo "Qwen3.5 supports only legacy_ce, dual_norm_distill, or router_compensated_dual_norm_distill; got ${rft_trainer}." >&2
            exit 1
            ;;
    esac
    rft_lr_tag="$(python -c 'import sys; from decimal import Decimal; print(format(Decimal(sys.argv[1]).normalize(), "E").lower().replace("e+", "e"))' "${rft_lr}")"
    rft_tag="_RFT-${rft_trainer}-lr${rft_lr_tag}"
    rft_args=(
        --finetune_routers
        --rft_trainer "${rft_trainer}"
        --rft_epochs "${rft_epochs}"
        --rft_batch_size "${rft_batch_size}"
        --rft_lr "${rft_lr}"
        --rft_wd "${rft_wd}"
    )
    if [[ "${rft_trainer}" != "legacy_ce" ]]; then
        rft_args+=(--rft_teacher_cache_dir "${rft_teacher_cache_dir}")
        if [[ "${rft_rebuild_teacher_cache}" == "true" ]]; then
            rft_args+=(--rft_rebuild_teacher_cache)
        fi
    fi
fi

gptq_checkpoint_path="${GPTQ_CHECKPOINT_PATH:-${gptq_checkpoint_root}/${model_name}/GEMQ/${quant_tag}}"
checkpoint_args=()
if [[ "${save_gptq_checkpoint}" == "true" && "${load_gptq_checkpoint}" == "true" ]]; then
    echo "SAVE_GPTQ_CHECKPOINT and LOAD_GPTQ_CHECKPOINT cannot both be true." >&2
    exit 1
elif [[ "${save_gptq_checkpoint}" == "true" ]]; then
    [[ ! -e "${gptq_checkpoint_path}" ]] || {
        echo "GPTQ checkpoint already exists and will not be overwritten: ${gptq_checkpoint_path}" >&2
        echo "Set LOAD_GPTQ_CHECKPOINT=true to reuse it." >&2
        exit 1
    }
    checkpoint_args=(--save_gptq_checkpoint --gptq_checkpoint_path "${gptq_checkpoint_path}")
elif [[ "${load_gptq_checkpoint}" == "true" ]]; then
    [[ -f "${gptq_checkpoint_path}/_SUCCESS" ]] || {
        echo "Complete GPTQ checkpoint not found: ${gptq_checkpoint_path}" >&2
        exit 1
    }
    checkpoint_args=(--load_gptq_checkpoint --gptq_checkpoint_path "${gptq_checkpoint_path}")
fi

save_path="${SAVE_PATH:-results/fake_quant_models/${model_name}/GEMQ/${quant_tag}${rft_tag}}"
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

# Qwen3.5 support is intentionally fake-quant only. Linear-attention auxiliary
# projections and non-target weights stay BF16.
echo "=============================================="
echo ">>> Qwen3.5-35B-A3B fake-quant job"
echo "----------------------------------------------"
echo " Allocation:       ${metric} (mixed=${mixed_prec})"
echo " Attention bits:   linear=${linear_attn_wbits}, softmax=${softmax_attn_wbits}"
echo " Dense bits:       ${dense_wbits}"
echo " Fine-tuning:      ${finetune_routers} (trainer=${rft_trainer})"
if [[ "${finetune_routers}" == "true" ]]; then
    echo " RFT optimizer:    epochs=${rft_epochs}, batch=${rft_batch_size}, lr=${rft_lr}, wd=${rft_wd}"
fi
echo " Save GPTQ ckpt:   ${save_gptq_checkpoint}"
echo " Load GPTQ ckpt:   ${load_gptq_checkpoint}"
if [[ "${save_gptq_checkpoint}" == "true" || "${load_gptq_checkpoint}" == "true" ]]; then
    echo " GPTQ ckpt path:   ${gptq_checkpoint_path}"
fi
echo " Save path:        ${save_path}"
echo "=============================================="

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
    "${rft_args[@]}" \
    "${checkpoint_args[@]}" \
    --save_path "${save_path}" \
    --save_dtype bfloat16

if [[ "${mixed_prec}" == "true" ]]; then
    cp "${bit_config}" "${save_path}/expert_bit_allocation.pkl"
    allocation_metadata="${bit_config%.pkl}.json"
    if [[ -f "${allocation_metadata}" ]]; then
        cp "${allocation_metadata}" "${save_path}/expert_allocation_metadata.json"
    fi
fi
