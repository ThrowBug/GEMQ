# ProQuant

This branch supports ProQuant on **Qwen3.5-35B-A3B**, **GLM-4.7-Flash**, and **Trinity-Mini** with Transformers 5.2.0. To reproduce Qwen3-30B-A3B-Instruct-2507, use the Qwen3 release branch and a separate Conda environment.

## Setup

Use a dedicated CUDA-enabled Conda environment for this branch:

```bash
conda create -n proquant-qwen35 python=3.10 -y
conda activate proquant-qwen35
pip install -e .
```

Place `c4-train.00000-of-01024.json` in `data/`. The statistics and quantization scripts use `Qwen/Qwen3.5-35B-A3B` by default; set `MODEL` to a local model directory if needed. Run the following commands from the repository root.

## 1. Compute expert quantization costs

```bash
AVERAGE_BITS=2 \
  CANDIDATE_BITS=0,2,3 \
  FORWARD_BATCH_SIZE=16 \
  bash scripts/Qwen3.5-35B-A3B/compute_pop_costs.sh
```

## 2. Allocate expert bit-widths

```bash
MAX_PRUNE_RATIO=0.15 \
  BITS_PER_EXPERT=2.0 \
  WBITS=0,2,3 \
  POP_COST_BITS=0,2,3 \
  bash scripts/Qwen3.5-35B-A3B/allocate.sh
```

## 3. Quantize and save a reusable GPTQ checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  MAX_PRUNE_RATIO=0.15 \
  BPE=2.0 WBITS=0,2,3 \
  FINETUNE=false \
  SAVE_GPTQ_CHECKPOINT=true \
  bash scripts/Qwen3.5-35B-A3B/quantize.sh
```

## 4. Fine-tune with RoCo

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  MAX_PRUNE_RATIO=0.15 \
  BPE=2.0 WBITS=0,2,3 \
  FINETUNE=true RFT_TRAINER=roco \
  RFT_LR=3e-3 LOAD_GPTQ_CHECKPOINT=true \
  bash scripts/Qwen3.5-35B-A3B/quantize.sh
```

The allocation file is required for both runs of `quantize.sh`. To select one explicitly, set `BIT_CFG_PATH` in both commands. Set `MODEL` consistently whenever using a local model directory.

## 5. Serve the fine-tuned model

We recommend installing vLLM in a separate Conda environment:

```bash
conda create -n proquant-vllm-qwen35 python=3.12 -y
conda activate proquant-vllm-qwen35
pip install vllm
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  DATA_PARALLEL_SIZE=1 \
  TENSOR_PARALLEL_SIZE=2 \
  PIPELINE_PARALLEL_SIZE=1 \
  MODEL_VARIANT=FQ \
  FQ_MODEL_PATH=/path/to/quantized/model \
  bash scripts/Qwen3.5-35B-A3B/serve_vllm.sh \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
    
```

Our experiments used the following additional environment variables when starting the vLLM service:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_XET=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="${CONDA_PREFIX}/cuda-compat"
```

Set these before running the serving command above. The CUDA compatibility settings require compatible `cuda-compat` libraries to be installed at the specified path; they are not needed on every machine.

## 6. Evaluate with EvalScope

We recommend installing EvalScope in another Conda environment:

```bash
conda create -n proquant-eval python=3.10 -y
conda activate proquant-eval
pip install evalscope
```

Run this in another terminal while the service is active:

```bash
VLLM_HOST=127.0.0.1 \
  VLLM_PORT=8000 \
  MODEL_VARIANT=FQ \
  DATASET_HUB=modelscope \
  EVAL_BATCH_SIZE=8 \
  DATASETS="gpqa_diamond" \
  GENERATION_CONFIG='{"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":8192,"seed":0,"timeout":3200,"retries":3,"extra_body":{"min_p":0.0, "chat_template_kwargs":{"enable_thinking":false}}}' \
  bash scripts/Qwen3.5-35B-A3B/eval_vllm.sh
```

## GLM-4.7-Flash and Trinity-Mini

Use the same quantization environment and C4 calibration file. Select `GLM-4.7-Flash` or `Trinity-Mini` below. The launchers use the Qwen3.5 PoP and RoCo parameters and infer the paths for each stage. Set `MODEL` consistently if you use a local checkpoint.

```bash
SCRIPTS=scripts/GLM-4.7-Flash  # or scripts/Trinity-Mini

AVERAGE_BITS=2 CANDIDATE_BITS=0,2,3 \
  bash "$SCRIPTS/compute_pop_costs.sh"

MAX_PRUNE_RATIO=0.15 BITS_PER_EXPERT=2.0 WBITS=0,2,3 POP_COST_BITS=0,2,3 \
  bash "$SCRIPTS/allocate.sh"

CUDA_VISIBLE_DEVICES=0,1 MAX_PRUNE_RATIO=0.15 BPE=2.0 WBITS=0,2,3 \
  FINETUNE=false SAVE_GPTQ_CHECKPOINT=true bash "$SCRIPTS/quantize.sh"

CUDA_VISIBLE_DEVICES=0,1 MAX_PRUNE_RATIO=0.15 BPE=2.0 WBITS=0,2,3 \
  FINETUNE=true RFT_TRAINER=roco RFT_LR=3e-3 LOAD_GPTQ_CHECKPOINT=true bash "$SCRIPTS/quantize.sh"

bash "$SCRIPTS/eval_ppl.sh" /path/to/quantized/model
```

Use `POP_COST_PATH`, `BIT_CFG_PATH`, and `GPTQ_CHECKPOINT_PATH` to select specific files. `SOFTMAX_ATTN_WBITS` and `DENSE_WBITS` default to 4. Trinity's attention gate and both models' router weights retain their original precision. Routing correction biases are used in FP32. GLM is exported for standard generation without MTP speculative decoding.

For GLM, `RFT_TRAINER=direct_rmsnorm` updates only the decoder `input_layernorm` and `post_attention_layernorm` weights with the same teacher loss. You can reuse the GPTQ checkpoint from above:

```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_PRUNE_RATIO=0.15 BPE=2.0 WBITS=0,2,3 \
  FINETUNE=true RFT_TRAINER=direct_rmsnorm RFT_LR=3e-3 LOAD_GPTQ_CHECKPOINT=true \
  bash scripts/GLM-4.7-Flash/quantize.sh
```

In the separate vLLM and EvalScope environments, use the matching script directory in both terminals:

```bash
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 MODEL_VARIANT=FQ \
  MAX_PRUNE_RATIO=0.15 BPE=2.0 WBITS=0,2,3 FINETUNE=true RFT_TRAINER=roco RFT_LR=3e-3 \
  bash "$SCRIPTS/serve_vllm.sh"
```

```bash
DATASETS="math_500" bash "$SCRIPTS/eval_vllm.sh"
```

Use the same `RFT_TRAINER` and `RFT_LR` as in quantization for automatic model lookup. Set `FQ_MODEL_PATH` to serve a specific saved model, including one written to a custom `SAVE_PATH`.

### GLM Layer-RE and Legacy CE

Layer-RE defaults to `WBITS=1,2,3` and applies the budget to the 64 routed experts in each of GLM's 46 MoE layers. `DENSE_WBITS` controls the shared experts and the first dense FFN separately. The stats script writes both the activation gradients and the Layer-RE coefficients.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/GLM-4.7-Flash/compute_stats.sh
ALLOCATION_METRIC=layer_re BITS_PER_EXPERT=2.0 bash scripts/GLM-4.7-Flash/allocate.sh

CUDA_VISIBLE_DEVICES=0,1 ALLOCATION_METRIC=layer_re BPE=2.0 \
  FINETUNE=false SAVE_GPTQ_CHECKPOINT=true bash scripts/GLM-4.7-Flash/quantize.sh

CUDA_VISIBLE_DEVICES=0,1 ALLOCATION_METRIC=layer_re BPE=2.0 \
  FINETUNE=true RFT_TRAINER=router_ft RFT_LR=1e-4 \
  LOAD_GPTQ_CHECKPOINT=true bash scripts/GLM-4.7-Flash/quantize.sh

CUDA_VISIBLE_DEVICES=0,1 MODEL_VARIANT=FQ ALLOCATION_METRIC=layer_re BPE=2.0 \
  FINETUNE=true RFT_TRAINER=router_ft RFT_LR=1e-4 bash scripts/GLM-4.7-Flash/serve_vllm.sh
```

`router_ft` uses autoregressive CE without a teacher. You can also run it on a PoP GPTQ checkpoint by using `ALLOCATION_METRIC=pop` and the settings that produced that checkpoint.

## License

This repository is derived from [GEMQ](https://github.com/jndeng/GEMQ). The original copyright notice and MIT license are retained in [LICENSE](LICENSE), alongside the notice for Jiaqi Chen's additions. See [third-party notices](THIRD_PARTY_NOTICES.md) for inherited source attribution.
