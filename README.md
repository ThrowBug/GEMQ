<div align="center">

<h1>GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2605.23078-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2605.23078)

</div>

GEMQ is a post-training quantization framework for Mixture-of-Experts (MoE) LLMs that enables extreme low-bit quantization (down to 1.5 bits per expert) with minimal accuracy degradation. It works by:
1. automatically assigning different bit-widths to experts based on their importance;
2. fine-tuning the routers so they can better work with quantized experts;
3. optionally using progressive quantization to refine the bit allocation.


### What's in this repo
* An ILP solver for global expert-level bit allocation
* GPTQ-based quantization and router fine-tuning pipelines
* Efficient low-bit MoE triton kernels for **real** quantized inference


## Updates

- [2026/08] Bit allocation now runs on **HiGHS**, the ILP solver bundled with SciPy, so regenerating the bit configs no longer needs a Gurobi license. Gurobi stays available as an optional backend.
- [2026/08] Active workflows are organized by model under `scripts/<model>/`; the original real-quant scripts are retained under `scripts/deprecated/`.
- [2026/08] Added mixed-data calibration, teacher-guided router fine-tuning, and BF16 fake-quant/vLLM workflows for **OLMoE-1B-7B-0125-Instruct** and **Qwen3-30B-A3B-Instruct-2507**.
- [2026/08] Fixed a ~15% perplexity regression on DeepSeek-V2 caused by a missing YaRN `mscale` in HF's built-in implementation ([transformers#47435](https://github.com/huggingface/transformers/pull/47435)).


## Installation

```bash
conda create -n gemq python=3.10 -y
conda activate gemq
git clone https://github.com/jndeng/GEMQ
cd GEMQ
pip install -e .

# (Optional) only needed if you want to solve the bit allocation with Gurobi
# instead of the default HiGHS solver (and thus requires a Gurobi license):
# pip install -e ".[gurobi]"
```

> [!NOTE]
>
> By default, bit allocation is solved with **HiGHS**, which does not require a commercial license.
> In our experiments, however, we used **Gurobi** to produce the configs under `configs/`.
> Gurobi remains available as an optional backend -- install it as shown above, then run a model's `allocate.sh` with `ILP_BACKEND=gurobi`.


## Usage

Active workflows are grouped by model:

| Model | Script directory | Default BPE |
| --- | --- | ---: |
| OLMoE-1B-7B-0125-Instruct | `scripts/OLMoE-1B-7B-0125-Instruct/` | 2.0 |
| Qwen3-30B-A3B-Instruct-2507 | `scripts/Qwen3-30B-A3B-Instruct-2507/` | 2.5 |

The old paper-reproduction and packed real-quant workflows are available under
`scripts/deprecated/`.

### Calibration, allocation, and quantization

Both active workflows default to one prepared calibration tensor containing 50% English
WildChat, 40% UltraChat 200k, and 10% FineWeb-Edu blocks. The preparation command streams
only the records needed for the cache. The exact same input IDs are then used for
bit-allocation statistics, GPTQ, teacher-target collection, and router fine-tuning.

For OLMoE:

```bash
bash scripts/OLMoE-1B-7B-0125-Instruct/prepare_calib.sh
bash scripts/OLMoE-1B-7B-0125-Instruct/compute_stats.sh
bash scripts/OLMoE-1B-7B-0125-Instruct/allocate.sh
bash scripts/OLMoE-1B-7B-0125-Instruct/quantize.sh
```

For Qwen3-MoE:

```bash
bash scripts/Qwen3-30B-A3B-Instruct-2507/prepare_calib.sh
bash scripts/Qwen3-30B-A3B-Instruct-2507/compute_stats.sh
bash scripts/Qwen3-30B-A3B-Instruct-2507/allocate.sh
bash scripts/Qwen3-30B-A3B-Instruct-2507/quantize.sh
```

`SEED`, `NSAMPLES`, and `SEQLEN` are shared by the four stages. Set
`REBUILD_CALIB_CACHE=true` to rebuild an existing mixed cache. To use GEMQ's legacy C4
loader and C4 naming, skip `prepare_calib.sh` and pass `CALIB_DATASET=c4` to the remaining
three scripts. Qwen statistics default to `CUDA_VISIBLE_DEVICES=0,1,2`; override it for a
different machine.

> [!IMPORTANT]
>
> The configs reported in the original paper were produced with Gurobi. HiGHS solves the
> same ILP without a commercial license, but a non-unique optimum may result in a different
> allocation. Use `ILP_BACKEND=gurobi` when exact solver reproduction is required.

The quantization scripts expose router trainer, router loss, timing, loss weights, optimizer,
dataset, bit-width, and cache controls as environment variables near the top of each file.
They save dequantized approximate weights as standard BF16 Hugging Face checkpoints under
`results/fake_quant_models/`, which vLLM can load without a GEMQ runtime patch.

For pruning-aware router adaptation on a Qwen3 zero-bit allocation, run the same
`quantize.sh` with `RFT_TRAINER=pruned_expert_reroute` (or
`pruned_expert_reroute_then_distill` for a subsequent teacher-logit CE stage).
The new trainer requires mixed precision and runs after all GPTQ quantization.
Up to `RFT_SCREEN_TOKENS_PER_EXPERT` tokens (default 64) screen all surviving experts;
up to `RFT_COST_TOKENS_PER_EXPERT` disjoint tokens (default 256) estimate costs for the
shortlist. It caches full-precision teacher routes and pruned-expert samples separately
from legacy router targets. To reuse a validated post-GPTQ checkpoint, set
`LOAD_GPTQ_CHECKPOINT=true` with the same allocation and calibration settings. The
new trainer writes to a distinct `_RFT-pruned_expert_reroute*` output directory and
refuses to overwrite an existing result.

### PMQ baseline for Qwen3-MoE

The Qwen3-30B-A3B-Instruct-2507 workflow also includes an isolated MC-MoE PMQ
baseline. It uses C4 with seed 0, the released PMQ objective and per-layer ILP,
and expert bit candidates `{1,2,3}` without pruning. Final quantization uses the
same GEMQ GPTQ implementation as the GEMQ method: attention projections are 4-bit,
routers remain BF16, experts average 2 bits, and the dequantized approximate weights
are saved as a standard BF16 Hugging Face checkpoint.

Place `c4-train.00000-of-01024.json` under `data/`, then run the complete workflow:

```bash
bash scripts/Qwen3-30B-A3B-Instruct-2507/pmq_pipeline.sh
```

The three stages can also be run separately:

```bash
bash scripts/Qwen3-30B-A3B-Instruct-2507/pmq_compute_stats.sh
bash scripts/Qwen3-30B-A3B-Instruct-2507/pmq_allocate.sh
bash scripts/Qwen3-30B-A3B-Instruct-2507/pmq_quantize.sh
```

By default, PMQ statistics are written below `cache/.../PMQ/`, the independent
allocation and its JSON metadata below `configs/.../PMQ/`, and the loadable model
below `results/fake_quant_models/.../PMQ/`. `MODEL`, `CUDA_VISIBLE_DEVICES`,
`PMQ_STATS_DIR`, `PMQ_BIT_CONFIG`, and `PMQ_SAVE_PATH` can override their respective
locations. This baseline intentionally does not pass `--reproduce_mcmoe`,
`--finetune_routers`, or `--real_quant`.

### vLLM serving, benchmarking, and evaluation

Start OLMoE on one GPU:

```bash
MODEL_VARIANT=FQ CUDA_VISIBLE_DEVICES=0 \
    bash scripts/OLMoE-1B-7B-0125-Instruct/serve_vllm.sh
```

Start Qwen with its default tensor parallel size of two:

```bash
MODEL_VARIANT=FQ CUDA_VISIBLE_DEVICES=0,1 \
    bash scripts/Qwen3-30B-A3B-Instruct-2507/serve_vllm.sh
```

Set `MODEL_VARIANT=FP` to serve the original checkpoint. With the service running, use the
matching model directory for API performance and EvalScope accuracy evaluation:

```bash
MODEL_VARIANT=FQ bash scripts/Qwen3-30B-A3B-Instruct-2507/bench_vllm.sh
MODEL_VARIANT=FQ LIMIT=10 bash scripts/Qwen3-30B-A3B-Instruct-2507/eval_vllm.sh
```

The service, benchmark, and evaluation scripts share `VLLM_HOST`, `VLLM_PORT`,
`VLLM_API_KEY`, `MODEL_VARIANT`, and `SERVED_MODEL_NAME`. `FQ_MODEL_PATH` overrides the
automatically derived quantized checkpoint path. When quantization used a nonzero `SEED`,
set the serving and benchmark scripts' `CALIB_SEED` to the same value; benchmark `SEED`
remains reserved for request generation.


## License

Released under the [MIT License](LICENSE).

## Acknowledgements
This repository builds upon several excellent open-source projects, including [MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE), [GPTQ](https://github.com/IST-DASLab/gptq), [HQQ](https://github.com/dropbox/hqq), [GemLite](https://github.com/dropbox/gemlite), and [gpt-fast](https://github.com/meta-pytorch/gpt-fast). We sincerely thank the authors and contributors for making their code publicly available.

## Citation
If you find GEMQ useful for your research or project, please consider citing:
```bibtex
@article{deng2026gemq,
  title={GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs},
  author={Deng, Jianing and Wang, Song and Wang, Dongwei and Liu, Zijie and Chen, Tianlong and Yang, Huanrui and Hu, Jingtong},
  journal={arXiv preprint arXiv:2605.23078},
  year={2026}
}
```
