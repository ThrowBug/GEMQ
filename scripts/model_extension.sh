#!/usr/bin/env bash
set -euo pipefail

model_name="$1"
stage="$2"
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "$model_name" in
    zai-org/GLM-4.7-Flash) service_name=glm-4.7-flash ;;
    arcee-ai/Trinity-Mini) service_name=trinity-mini ;;
    mistralai/Mixtral-8x7B-v0.1) service_name=mixtral-8x7b-v0.1 ;;
    *) echo "Unsupported model: $model_name" >&2; exit 2 ;;
esac

die() { echo "$*" >&2; exit 2; }
dataset="${CALIB_DATASET:-c4}"
nsamples="${NSAMPLES:-128}"
seqlen="${SEQLEN:-2048}"
seed="${SEED:-0}"
sample_tag="${dataset}-N${nsamples}-L${seqlen}-Seed${seed}"
model="${MODEL:-$model_name}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
attn_impl="${ATTN_IMPL:-eager}"
allocation_metric="${ALLOCATION_METRIC:-pop}"

variant="${MODEL_VARIANT:-FQ}"
variant="${variant^^}"
served_model_name="${SERVED_MODEL_NAME:-${service_name}-${variant,,}}"

if [[ "$stage" == compute_stats || "$stage" == compute_pop_costs || "$stage" == allocate || "$stage" == quantize ||
      ( "$stage" == serve_vllm && "$variant" == FQ && -z "${FQ_MODEL_PATH:-}" ) ]]; then
    context_mode="${CONTEXT_MODE:-uniform_bit}"
    context_bit="${POP_CONTEXT_BIT:-${AVERAGE_BITS:-2}}"
    [[ "$stage" != compute_pop_costs ]] || context_bit="${AVERAGE_BITS:-2}"
    candidate_bits="${CANDIDATE_BITS:-0,1,2,3}"
    normalize="${NORMALIZE_BY_ACTIVATION_COUNT:-true}"
    case "$context_mode" in
        fp) context_tag=fp; allocation_context_tag=Ctxfp ;;
        uniform_bit) context_tag="uniform${context_bit}bit"; allocation_context_tag="Ctxuniform${context_bit}" ;;
        *) die "CONTEXT_MODE must be fp or uniform_bit." ;;
    esac
    case "$normalize" in
        true) count_tag=; allocation_count_tag= ;;
        false) count_tag=_NoCountNorm; allocation_count_tag=-NoCountNorm ;;
        *) die "NORMALIZE_BY_ACTIVATION_COUNT must be true or false." ;;
    esac
    pop_prefix="cache/${model_name}/PopCosts_${sample_tag}_${context_tag}_B"
    data_args=(--calib_dataset "$dataset" --nsamples "$nsamples" --seqlen "$seqlen" --seed "$seed")
    model_args=(--model "$model" --model_name "$model_name" --model_dtype "$model_dtype" --attn_impl "$attn_impl" --use_fast)
fi

if [[ "$stage" == allocate || "$stage" == quantize ||
      ( "$stage" == serve_vllm && "$variant" == FQ && -z "${FQ_MODEL_PATH:-}" ) ]]; then
    case "$allocation_metric" in
        pop)
            [[ "${EXTRA_CONSTR:-none}" == none ]] || die "EXTRA_CONSTR is only supported by Layer-RE."
            wbits="${WBITS:-0,2,3}"
            max_prune_ratio="${MAX_PRUNE_RATIO:-0.25}"
            ;;
        layer_re)
            [[ "$model_name" != arcee-ai/Trinity-Mini ]] || die "Trinity-Mini uses PoP allocation."
            wbits="${WBITS:-1,2,3}"
            extra_constr="${EXTRA_CONSTR:-c2c3}"
            [[ "$extra_constr" == none || "$extra_constr" == c2c3 ]] || die "EXTRA_CONSTR must be none or c2c3."
            [[ ",$wbits," != *,0,* ]] || die "Layer-RE requires positive candidate bit-widths."
            ;;
        *) die "ALLOCATION_METRIC must be pop or layer_re." ;;
    esac
    [[ -z "${MIXED_PREC+x}" ]] || die "MIXED_PREC is no longer supported. A bit-allocation file is required."
    if [[ "$stage" == allocate ]]; then
        bpe="${BITS_PER_EXPERT:-${BPE:-2.0}}"
    else
        bpe="${BPE:-${BITS_PER_EXPERT:-2.0}}"
    fi
    allocation_seed="$seed"
    [[ "$stage" != serve_vllm ]] || allocation_seed="${CALIB_SEED:-$seed}"
    allocation_tag="${ALLOCATION_TAG:-}"
    if [[ -z "$allocation_tag" ]]; then
        case "$dataset" in
            c4) allocation_tag="C4-Seed${allocation_seed}" ;;
            math) allocation_tag="MATH-Seed${allocation_seed}" ;;
            math+c4) allocation_tag="MATH+C4-Seed${allocation_seed}" ;;
            *) die "Set ALLOCATION_TAG for CALIB_DATASET=${dataset}." ;;
        esac
    fi
    bpe_tag="$(LC_ALL=C printf '%.1f' "$bpe")"
    if [[ "$allocation_metric" == layer_re ]]; then
        constraint_tag=""
        [[ "$extra_constr" == none ]] || constraint_tag="_${extra_constr}"
        default_bit_cfg="configs/${model_name}/GEMQ/${allocation_tag}_Metric-layer_re_E${bpe_tag}_B${wbits}${constraint_tag}.pkl"
        legacy_bit_cfg="$default_bit_cfg"
    else
        policy_tag="${allocation_tag}_Metric-pop-${allocation_context_tag}${allocation_count_tag}_E${bpe_tag}_B${wbits}"
        legacy_sample_tag="${dataset}-N${nsamples}-L${seqlen}-Seed${allocation_seed}"
        legacy_policy_tag="${legacy_sample_tag}_D${model_dtype}-${attn_impl}_Metric-pop-${context_tag}${count_tag}_E${bpe}_B${wbits}"
        if [[ ",$wbits," == *,0,* ]]; then
            policy_tag+="_Pmax$(LC_ALL=C printf '%g' "$max_prune_ratio")_EqPrune"
            legacy_policy_tag+="_Pmax${max_prune_ratio}_EqPrune"
        fi
        default_bit_cfg="configs/${model_name}/ProQuant/${policy_tag}.pkl"
        legacy_bit_cfg="configs/${model_name}/ProQuant/${legacy_policy_tag}.pkl"
    fi
    bit_cfg="${BIT_CFG_PATH:-$default_bit_cfg}"
    if [[ "$stage" == quantize && -z "${BIT_CFG_PATH:-}" && ! -f "$bit_cfg" && -f "$legacy_bit_cfg" ]]; then
        bit_cfg="$legacy_bit_cfg"
    fi
fi

if [[ "$stage" == quantize ||
      ( "$stage" == serve_vllm && "$variant" == FQ && -z "${FQ_MODEL_PATH:-}" ) ]]; then
    attn_wbits="${SOFTMAX_ATTN_WBITS:-4}"
    dense_wbits="${DENSE_WBITS:-4}"
    groupsize="${GROUPSIZE:-128}"
    blocksize="${BLOCKSIZE:-128}"
    percdamp="${PERCDAMP:-0.01}"
    allocation_name="$(basename "$bit_cfg" .pkl)"
    qtype="$(basename "$(dirname "$bit_cfg")")"
    model_suffix="_SA${attn_wbits}-G16-D${dense_wbits}-E${bpe_tag}"
    if [[ "$groupsize:$blocksize:$percdamp" != "128:128:0.01" ]]; then
        model_suffix+="-GS${groupsize}-BS${blocksize}-PD${percdamp}"
    fi
    if [[ "$model_dtype:$attn_impl" != "bfloat16:eager" ]]; then
        model_suffix+="-${model_dtype}-${attn_impl}"
    fi
    quant_tag="${allocation_name}${model_suffix}"
    legacy_suffix="_A${attn_wbits}-D${dense_wbits}-G${groupsize}-B${blocksize}-PD${percdamp}"
    legacy_allocation_name="$(basename "${BIT_CFG_PATH:-$legacy_bit_cfg}" .pkl)"

    finetune="${FINETUNE:-true}"
    rft_trainer="${RFT_TRAINER:-roco}"
    lr="${RFT_LR:-1e-4}"
    rft_tag=""
    case "$finetune" in
        true)
            case "$rft_trainer" in
                roco) ;;
                router_ft)
                    [[ "$model_name" != arcee-ai/Trinity-Mini ]] || die "Trinity-Mini supports only RoCo fine-tuning."
                    ;;
                direct_rmsnorm)
                    [[ "$model_name" == zai-org/GLM-4.7-Flash ]] || die "$rft_trainer requires GLM-4.7-Flash."
                    ;;
                *) die "Unknown RFT_TRAINER=$rft_trainer." ;;
            esac
            lr_tag="$(python -c 'from decimal import Decimal; import sys; print(format(Decimal(sys.argv[1]).normalize(), "E").lower().replace("e+", "e"))' "$lr")"
            rft_tag="_RFT-${rft_trainer}-lr${lr_tag}"
            ;;
        false) ;;
        *) die "FINETUNE must be true or false." ;;
    esac
    default_fq_path="results/fake_quant_models/${model_name}/${qtype}/${quant_tag}${rft_tag}"
    legacy_fq_path="results/fake_quant_models/${model_name}/ProQuant/${legacy_allocation_name}${legacy_suffix}${rft_tag}"
fi

case "$stage" in
    compute_stats)
        [[ "$model_name" != arcee-ai/Trinity-Mini ]] || die "Trinity-Mini uses PoP statistics."
        wbits="${WBITS:-1,2,3}"
        [[ ",$wbits," != *,0,* ]] || die "Layer-RE requires positive candidate bit-widths."
        gpus="${CUDA_VISIBLE_DEVICES:-0,1,2}"
        layer_grads_path="${LAYER_GRADS_PATH:-cache/${model_name}/LayerGrads_${sample_tag}.pt}"
        layer_re_path="${LAYER_RE_PATH:-cache/${model_name}/LayerRE_${sample_tag}_B${wbits}_fast.pkl}"
        CUDA_VISIBLE_DEVICES="$gpus" python -m proquant.compute_model_stats \
            "${model_args[@]}" "${data_args[@]}" --mode layer_grads --layer_grads_path "$layer_grads_path"
        CUDA_VISIBLE_DEVICES="$gpus" exec python -m proquant.compute_model_stats \
            "${model_args[@]}" "${data_args[@]}" --mode layer_re --layer_grads_path "$layer_grads_path" \
            --layer_re_path "$layer_re_path" --wbits "$wbits" --blocksize "${BLOCKSIZE:-128}" \
            --forward_batch_size "${FORWARD_BATCH_SIZE:-1}" "$@"
        ;;
    compute_pop_costs)
        output="${OUTPUT_PATH:-${pop_prefix}${candidate_bits}${count_tag}.pt}"
        printf 'Model: %s\nPoP output: %s\n' "$model_name" "$output"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" exec python -m proquant.compute_pop_costs "${model_args[@]}" "${data_args[@]}" \
            --context_mode "$context_mode" --average_bits "$context_bit" --candidate_bits "$candidate_bits" \
            --normalize_by_activation_count "$normalize" --blocksize "${BLOCKSIZE:-128}" \
            --forward_batch_size "${FORWARD_BATCH_SIZE:-1}" --expert_batch_size "${EXPERT_BATCH_SIZE:-4096}" \
            --output_path "$output" "$@"
        ;;
    allocate)
        if [[ "$allocation_metric" == layer_re ]]; then
            layer_re_path="${LAYER_RE_PATH:-cache/${model_name}/LayerRE_${sample_tag}_B${wbits}_fast.pkl}"
            [[ -f "$layer_re_path" ]] || die "Layer-RE file not found: $layer_re_path. Run compute_stats.sh or set LAYER_RE_PATH."
            stat_args=(--layer_re_path "$layer_re_path" --extra_constr "$extra_constr")
        else
            pop_cost="${POP_COST_PATH:-${pop_prefix}${POP_COST_BITS:-$candidate_bits}${count_tag}.pt}"
            [[ -f "$pop_cost" ]] || die "PoP cost file not found: $pop_cost. Run compute_pop_costs.sh or set POP_COST_PATH."
            stat_args=(--pop_cost_path "$pop_cost" --max_prune_ratio "$max_prune_ratio")
        fi
        output="${ALLOCATION_OUTPUT_PATH:-$bit_cfg}"
        exec python -m proquant.allocate_bits --model_name "$model_name" --allocation_metric "$allocation_metric" \
            "${stat_args[@]}" --bit_budget "$bpe" --bit_candidates "$wbits" --ilp_backend "${ILP_BACKEND:-highs}" \
            --save_path "$output" "$@"
        ;;
    quantize)
        [[ -f "$bit_cfg" ]] || die "Allocation file not found: $bit_cfg. Run allocate.sh or set BIT_CFG_PATH."
        checkpoint_root="${GPTQ_CHECKPOINT_ROOT:-results/gptq_checkpoints}/${model_name}"
        checkpoint="${GPTQ_CHECKPOINT_PATH:-${checkpoint_root}/${qtype}/${quant_tag}}"
        args=(--bit_cfg "$bit_cfg" --linear_attn_wbits 16 --softmax_attn_wbits "$attn_wbits"
            --dense_wbits "$dense_wbits" --gate_wbits 16 --groupsize "$groupsize"
            --blocksize "$blocksize" --percdamp "$percdamp" --mse --save_dtype bfloat16)

        case "${SAVE_GPTQ_CHECKPOINT:-false}:${LOAD_GPTQ_CHECKPOINT:-false}" in
            true:false)
                [[ ! -e "$checkpoint" ]] || die "GPTQ checkpoint already exists: $checkpoint. Set LOAD_GPTQ_CHECKPOINT=true to reuse it."
                args+=(--save_gptq_checkpoint --gptq_checkpoint_path "$checkpoint")
                ;;
            false:true)
                if [[ "$allocation_metric" == pop && -z "${GPTQ_CHECKPOINT_PATH:-}" && ! -f "$checkpoint/_SUCCESS" ]]; then
                    legacy_checkpoint="${checkpoint_root}/ProQuant/${legacy_allocation_name}${legacy_suffix}"
                    [[ ! -f "$legacy_checkpoint/_SUCCESS" ]] || checkpoint="$legacy_checkpoint"
                fi
                [[ -f "$checkpoint/_SUCCESS" ]] || die "Complete GPTQ checkpoint not found: $checkpoint"
                args+=(--load_gptq_checkpoint --gptq_checkpoint_path "$checkpoint")
                ;;
            false:false) ;;
            *) die "Set at most one of SAVE_GPTQ_CHECKPOINT and LOAD_GPTQ_CHECKPOINT to true." ;;
        esac
        if [[ "$finetune" == true ]]; then
            args+=(--finetune --rft_trainer "$rft_trainer" --rft_lr "$lr" --rft_epochs "${RFT_EPOCHS:-1}"
                --rft_batch_size "${RFT_BATCH_SIZE:-1}" --rft_wd "${RFT_WD:-0.0}")
            if [[ "$rft_trainer" != router_ft ]]; then
                args+=(--rft_teacher_cache_dir "${RFT_TEACHER_CACHE_DIR:-cache/router_finetune}")
                if [[ "${RFT_REBUILD_TEACHER_CACHE:-false}" == true ]]; then
                    args+=(--rft_rebuild_teacher_cache)
                fi
            fi
        fi
        [[ "${CUDA_DIAGNOSTICS:-false}" != true ]] || args+=(--cuda_diagnostics)
        [[ "${REPRODUCE_MCMOE:-false}" != true ]] || args+=(--reproduce_mcmoe)
        if [[ "${EVAL_DOWNSTREAM:-false}" == true ]]; then
            args+=(--eval_downstream --downstream_tasks "${DOWNSTREAM_TASKS:-piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande,mathqa,mmlu}")
        fi
        output="${SAVE_PATH:-$default_fq_path}"
        [[ ! -e "$output" ]] || die "Model output already exists: $output. Set SAVE_PATH to a new directory."
        printf 'Model: %s\nBit config: %s\nSave path: %s\n' "$model_name" "$bit_cfg" "$output"
        if [[ "${SAVE_GPTQ_CHECKPOINT:-false}" == true || "${LOAD_GPTQ_CHECKPOINT:-false}" == true ]]; then
            printf 'GPTQ checkpoint: %s\n' "$checkpoint"
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" exec python -m proquant.quantize "${model_args[@]}" "${data_args[@]}" "${args[@]}" \
            --save_path "$output" "$@"
        ;;
    eval_ppl)
        model_path="${MODEL_PATH:-}"
        if [[ $# -gt 0 && "$1" != --* ]]; then model_path="$1"; shift; fi
        [[ -n "$model_path" ]] || die "Pass a model directory or set MODEL_PATH."
        args=()
        [[ "${OFFLOAD:-false}" != true ]] || args+=(--offload)
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" exec python -m proquant.evaluate_ppl --model "$model_path" --model_name "$model_name" \
            --model_dtype "$model_dtype" --attn_impl "$attn_impl" --use_fast \
            --seqlen "$seqlen" --datasets "${PPL_DATASETS:-wikitext2,c4}" "${args[@]}" "$@"
        ;;
    serve_vllm)
        case "$variant" in
            FP) model_path="${FP_MODEL_PATH:-$model}" ;;
            FQ)
                model_path="${FQ_MODEL_PATH:-${default_fq_path:-}}"
                if [[ "$allocation_metric" == pop && -z "${FQ_MODEL_PATH:-}" && ! -d "$model_path" && -d "$legacy_fq_path" ]]; then
                    model_path="$legacy_fq_path"
                fi
                [[ -d "$model_path" ]] || die "Fake-quant checkpoint not found: $model_path. Run quantize.sh or set FQ_MODEL_PATH."
                [[ -f "$model_path/config.json" && -f "$model_path/tokenizer_config.json" ]] ||
                    die "Not a complete Hugging Face checkpoint: $model_path"
                shopt -s nullglob
                weight_files=("$model_path"/*.safetensors "$model_path"/*.bin)
                shopt -u nullglob
                (( ${#weight_files[@]} > 0 )) || die "No model weight files found under: $model_path"
                ;;
            *) die "MODEL_VARIANT must be FP or FQ." ;;
        esac
        command -v vllm >/dev/null || die "Activate the vLLM environment."
        args=()
        [[ -z "${MAX_MODEL_LEN:-}" ]] || args+=(--max-model-len "$MAX_MODEL_LEN")
        [[ "${ENABLE_EXPERT_PARALLEL:-false}" != true ]] || args+=(--enable-expert-parallel)
        gpus="${CUDA_VISIBLE_DEVICES:-0,1}"
        tp="${TENSOR_PARALLEL_SIZE:-2}"
        pp="${PIPELINE_PARALLEL_SIZE:-1}"
        dp="${DATA_PARALLEL_SIZE:-1}"
        for parallel_size in "$tp" "$pp" "$dp"; do
            [[ "$parallel_size" =~ ^[1-9][0-9]*$ ]] || die "Parallel sizes must be positive integers."
        done
        commas="${gpus//[^,]/}"
        (( ${#commas} + 1 >= tp * pp * dp )) || die "CUDA_VISIBLE_DEVICES=$gpus has fewer GPUs than TP x PP x DP requires."
        printf 'Model path: %s\nServed model name: %s\n' "$model_path" "$served_model_name"
        CUDA_VISIBLE_DEVICES="$gpus" exec vllm serve "$model_path" --served-model-name "$served_model_name" \
            --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8000}" \
            --api-key "${VLLM_API_KEY:-EMPTY}" --dtype bfloat16 --generation-config vllm \
            --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" --data-parallel-size "$dp" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" "${args[@]}" "$@"
        ;;
    eval_vllm)
        [[ "$variant" == FP || "$variant" == FQ ]] || die "MODEL_VARIANT must be FP or FQ."
        command -v evalscope >/dev/null || die "Activate the EvalScope environment."
        url="${VLLM_API_URL:-http://${VLLM_HOST:-127.0.0.1}:${VLLM_PORT:-8000}/v1}"
        url="${url%/}"
        api_key="${VLLM_API_KEY:-EMPTY}"
        curl --fail --silent --show-error --header "Authorization: Bearer $api_key" "$url/models" >/dev/null
        read -r -a datasets <<< "${DATASETS:-gsm8k}"
        generation_defaults='{"temperature":0.0,"max_tokens":2048,"seed":0,"timeout":600,"retries":3}'
        args=()
        [[ -z "${LIMIT:-}" ]] || args+=(--limit "$LIMIT")
        [[ -z "${DATASET_HUB:-}" ]] || args+=(--dataset-hub "$DATASET_HUB")
        exec evalscope eval --model "$served_model_name" --model-id "$served_model_name" \
            --api-url "$url" --api-key "$api_key" --eval-type openai_api --datasets "${datasets[@]}" \
            --eval-batch-size "${EVAL_BATCH_SIZE:-8}" --generation-config "${GENERATION_CONFIG:-$generation_defaults}" \
            "${args[@]}" "$@"
        ;;
    *) die "Unknown stage: $stage" ;;
esac
