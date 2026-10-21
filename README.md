# ProQuant

This branch reproduces ProQuant on **Qwen3.5-35B-A3B**. Qwen3 and Qwen3.5 require different dependency versions. To reproduce Qwen3-30B-A3B-Instruct-2507, switch to the main branch and use a separate Conda environment for its dependencies.

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

## License

This repository is derived from [GEMQ](https://github.com/jndeng/GEMQ). The original copyright notice and MIT license are retained in [LICENSE](LICENSE), alongside the notice for Jiaqi Chen's additions. See [third-party notices](THIRD_PARTY_NOTICES.md) for inherited source attribution.
