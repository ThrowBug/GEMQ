#!/bin/bash
set -euo pipefail

# ===============================
#  Model settings
# ===============================
model_name="deepseek-ai/DeepSeek-V2-Lite"
# Path to a saved *real* quant checkpoint. Leave empty to run only the cheap
# synthetic tests (Level 1/2), which need no checkpoint at all.
model_path="results/real_quant_models/deepseek-ai/DeepSeek-V2-Lite/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0_RFT"

# ===============================
#  Evaluation settings
# ===============================
nseq=8                     # number of wikitext2 sequences for the ppl comparison
seqlen=2048
ndecode=32                 # tokens greedily decoded in the decode-path test

# ===============================
#  What to run
# ===============================
run_synthetic=false         # Level 1/2: single linear + MoE block, seconds, no checkpoint
run_endtoend=true          # Level 3/4: full-model ppl + decode, needs model_path


# ===============================
#  AUTO argument construction
# ===============================
synthetic_tests=(tests/test_quant_linear_equiv.py tests/test_moe_block_equiv.py)
endtoend_tests=(tests/test_real_vs_fake_ppl.py tests/test_decode_equiv.py)

ckpt_args=()
if [[ -n "$model_path" ]]; then
    ckpt_args=(--model-path "$model_path" --model-name "$model_name"
               --nseq "$nseq" --seqlen "$seqlen" --ndecode "$ndecode")
fi


# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> Real-vs-fake quantization equivalence tests"
echo "----------------------------------------------"
echo " Model:       ${model_name}"
echo " Checkpoint:  ${model_path:-<none, synthetic tests only>}"
echo " PPL window:  ${nseq} x ${seqlen} tokens"
echo "=============================================="

if [[ "${run_synthetic}" == "true" ]]; then
    echo ">>> Level 1/2: kernel and MoE block equivalence ..."
    python -m pytest "${synthetic_tests[@]}" -v
fi

if [[ "${run_endtoend}" == "true" ]]; then
    if [[ -z "$model_path" ]]; then
        echo ">>> Level 3/4 skipped: set model_path at the top of this script."
        exit 0
    fi
    echo ">>> Level 3/4: end-to-end perplexity and decode ..."
    # -s so the perplexity table and decoded text reach the terminal
    python -m pytest "${endtoend_tests[@]}" -v -s "${ckpt_args[@]}"
fi
