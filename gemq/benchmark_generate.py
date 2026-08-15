import argparse
import time
import json
import contextlib
import inspect
from typing import Any, Collection, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import (
    load_quantized_model, describe_model_impl, align_deepseek_softmax_scale,
)


def device_sync(device="cuda"):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    else:
        print(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = F.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    last_token_logits = logits[0, -1]
    if temperature <= 0:
        # Greedy decoding must bypass temperature scaling entirely. Replacing zero
        # with a tiny value overflows FP16 logits before top-k filtering and turns the
        # softmax into NaN (which commonly collapses generation to token id 0).
        idx_next = torch.argmax(last_token_logits, dim=-1, keepdim=True).to(dtype=torch.int)
        return idx_next, None

    probs = logits_to_probs(last_token_logits, temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model, x, kv_cache, input_pos, **sampling_kwargs):
    outputs = model(x, past_key_values=kv_cache, cache_position=input_pos)
    return sample(outputs.logits, **sampling_kwargs)[0]


def decode_one_token(model, x, kv_cache, input_pos, **sampling_kwargs):
    outputs = model(x, past_key_values=kv_cache, cache_position=input_pos)
    return sample(outputs.logits, **sampling_kwargs)[0]


def decode_n_tokens(
    model,
    cur_token,
    kv_cache,
    input_pos,
    num_new_tokens,
    eos_token_ids: Optional[Collection[int]] = None,
    eos_check_interval: int = 8,
    **sampling_kwargs,
):
    """Decode up to ``num_new_tokens`` and stop after the first EOS token.

    EOS is checked in small chunks rather than after every token. Reading a CUDA token
    on the host forces synchronization; checking every ``eos_check_interval`` tokens
    amortizes that cost while wasting at most ``eos_check_interval - 1`` decode steps.
    Returned tokens are always trimmed at the first EOS (inclusive).
    """
    if eos_check_interval < 1:
        raise ValueError("eos_check_interval must be at least 1")

    new_tokens = []
    eos_ids = sorted(set(int(token_id) for token_id in (eos_token_ids or [])))
    eos_tensor = None
    if eos_ids:
        eos_tensor = torch.tensor(eos_ids, device=cur_token.device, dtype=cur_token.dtype)

    unchecked_start = 0
    for i in range(num_new_tokens):
        next_token = decode_one_token(model, cur_token, kv_cache, input_pos, **sampling_kwargs)
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone().view(1, -1)

        should_check_eos = eos_tensor is not None and (
            len(new_tokens) - unchecked_start >= eos_check_interval
            or i == num_new_tokens - 1
        )
        if not should_check_eos:
            continue

        recent_tokens = torch.cat(new_tokens[unchecked_start:]).reshape(-1)
        matches = torch.isin(recent_tokens, eos_tensor)
        match_positions = torch.nonzero(matches, as_tuple=False).reshape(-1)
        if match_positions.numel() > 0:
            first_match = int(match_positions[0].item())
            new_tokens = new_tokens[:unchecked_start + first_match + 1]
            break
        unchecked_start = len(new_tokens)

    return new_tokens


def load_model(args, compute_dtype=torch.float16, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    if args.is_fp:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=compute_dtype, device_map=device,
            trust_remote_code=args.trust_remote_code
        )
    else:
        # NOTE: routed through load_quantized_model rather than calling
        # AutoHQQHFModel.from_quantized directly. hqq loads the config and builds the
        # model itself and drops trust_remote_code on the way, which would silently run
        # HF's built-in implementation instead of the modeling code the checkpoint was
        # quantized with. See gemq/utils/hf_loading.py.
        model = load_quantized_model(
            args.model_path, compute_dtype=compute_dtype, device=device,
            trust_remote_code=args.trust_remote_code,
        )
    print(f"Modeling implementation: {describe_model_impl(model)}")

    # The decode loop drives a StaticCache via `cache_position`. Modeling code predating
    # that convention cannot generate; fail here rather than inside the first forward.
    if "cache_position" not in inspect.signature(model.forward).parameters:
        raise RuntimeError(
            f"{describe_model_impl(model)} does not accept `cache_position` and cannot "
            f"drive StaticCache. Drop --trust_remote_code."
        )

    # Generation must run on HF's built-in implementation (see above), which omits the
    # YaRN mscale that the official code applies to the attention scale. Restoring it
    # brings wikitext2 ppl from 10.80 back to 9.38, i.e. within 0.1% of the official
    # implementation gemq.quantize evaluates on.
    align_deepseek_softmax_scale(model)

    # patch model for inference
    if args.compile:
        print("Patching model for inference ...")
        prepare_for_inference(model, args.model_name, is_fp=args.is_fp)

    model = model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    kv_cache: StaticCache,
    eos_token_ids: Optional[Collection[int]] = None,
    eos_check_interval: int = 8,
    **sampling_kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Generate at most ``max_new_tokens``, stopping early when EOS is produced.

    When ``eos_token_ids`` is omitted, behavior remains compatible with the original
    fixed-length benchmark path.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    device = prompt.device
    stats = {}

    T = prompt.size(0)
    input_pos = torch.arange(0, T, device=device)

    t0 = time.perf_counter()
    device_sync()
    next_token = prefill(
        model, prompt.view(1, -1), kv_cache, input_pos, **sampling_kwargs
    )
    device_sync()
    elapsed_time = time.perf_counter() - t0
    stats["prefill_latency"] = elapsed_time # in seconds
    stats["prefill_throughput"] = T / stats["prefill_latency"] # tokens per second

    eos_ids = set(int(token_id) for token_id in (eos_token_ids or []))
    stopped_on_eos = bool(eos_ids) and int(next_token.item()) in eos_ids
    generated_tokens = [next_token.clone()]
    input_pos = torch.tensor([T], device=device, dtype=torch.long)

    decoded_tokens = []
    decode_latency = 0.0
    if max_new_tokens > 1 and not stopped_on_eos:
        t0 = time.perf_counter()
        device_sync()
        decoded_tokens = decode_n_tokens(
            model,
            next_token.view(1, -1),
            kv_cache,
            input_pos,
            max_new_tokens - 1,
            eos_token_ids=eos_ids,
            eos_check_interval=eos_check_interval,
            **sampling_kwargs,
        )
        device_sync()
        decode_latency = time.perf_counter() - t0
        generated_tokens.extend(decoded_tokens)
        if eos_ids and decoded_tokens:
            stopped_on_eos = int(decoded_tokens[-1].item()) in eos_ids

    stats["decode_latency"] = decode_latency
    stats["decode_throughput"] = (
        len(decoded_tokens) / decode_latency if decode_latency > 0 else 0.0
    )
    stats["generated_tokens"] = len(generated_tokens)
    stats["stopped_on_eos"] = stopped_on_eos

    generated = torch.cat([token.reshape(-1) for token in generated_tokens])
    seq = torch.cat((prompt.reshape(-1), generated))

    return seq, stats


def main(args):
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NOTE: only supports single GPU inference for now
    device = "cuda"
    compute_dtype = torch.float16
    print(f"Using device={device}, dtype={compute_dtype}")
    
    # load and patch model
    print("Loading model ...")
    t0 = time.time()
    model, tokenizer = load_model(args, compute_dtype, device)
    device_sync()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # compile model
    if args.compile:
        global decode_one_token
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

    # encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.size(1) # inputs.input_ids [1, S]

    # setup kv cache; static cache is used for compatibility with torch.compile
    max_seq_length = min(prompt_length + args.max_new_tokens, model.config.max_position_embeddings)
    kv_cache = StaticCache(model.config, max_cache_len=max_seq_length)

    # run for `num_samples` rounds
    start = -1 if args.compile else 0
    for i in range(start, args.num_samples):
        device_sync()
        torch.cuda.reset_peak_memory_stats()
        kv_cache.reset()

        # setup profiler
        if i != args.num_samples - 1 or not args.profile:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()

        # run generation
        t0 = time.perf_counter()
        with prof:
            outputs, stats = generate(
                model,
                inputs.input_ids[0], # [T,]
                args.max_new_tokens,
                kv_cache,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        device_sync()
        t = time.perf_counter() - t0

        # print generated text
        if i == -1:
            print(f"Compilation time: {t:.2f} seconds")
            continue
        print("\n\n" + "="*40 + f"\n Round {i}\n" + "="*40)
        print(tokenizer.decode(outputs.tolist(), skip_special_tokens=True))

        # print stats
        tokens_generated = outputs.size(0) - prompt_length
        print("-"*40 + "\n" + "-"*40)
        print(f"Context length:   {prompt_length}")
        print(f"Generated length: {tokens_generated}")
        print(f"Memory used:      {torch.cuda.max_memory_reserved() / 1024**3:.02f} GB")
        print("-"*53)
        print(f"| {'Stage':<10} | {'Latency (sec)':>13} | {'Throughput (tok/sec)':>20} |")
        print(f"| {'-'*10} | {'-'*13} | {'-'*20} |")
        print(f"| {'Prefill':<10} | {stats['prefill_latency']:>13.2f} | {stats['prefill_throughput']:>20.2f} |")
        print(f"| {'Decode':<10} | {stats['decode_latency']:>13.2f} | {stats['decode_throughput']:>20.2f} |")
        print(f"| {'-'*10} | {'-'*13} | {'-'*20} |")
        print(f"| {'Overall':<10} | {t:>13.2f} | {tokens_generated / t:>20.2f} |")
        print("-"*53)

    # save profiling trace
    if hasattr(prof, "export_chrome_trace"):
        prof.export_chrome_trace(f"{args.profile}.json")


def parse_args():
    parser = argparse.ArgumentParser(description="GEMQ Inference")

    # model args
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model; used to load model-specific modules",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Enable `trust_remote_code` when loading the model from HuggingFace Hub",
    )
    parser.add_argument(
        "--is_fp", action="store_true",
        help="Whether the current model is in full-precision"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="eager", choices=["eager", "sdpa"],
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Whether to compile the model with torch.compile",
    )

    # inference args
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is",
        help="Input prompt."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples (rounds)."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200,
        help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--top_k", type=int, default=200,
        help="Top-k for sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--profile", type=str, default="",
        help="Profile path."
    )

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    main(args)
