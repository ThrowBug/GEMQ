"""Minimal OpenAI-compatible API server for GEMQ real-quant checkpoints.

Run from the repository root after installing the ``api`` extra::

    pip install -e ".[api]"
    python -m gemq.openai_server \
        --model-path results/real_quant_models/deepseek-ai/DeepSeek-V2-Lite/GEMQ/... \
        --model-name deepseek-ai/DeepSeek-V2-Lite

The server uses one process and one GPU. Compatible concurrent requests can be
micro-batched by the background generation worker.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from gemq import batched_generate, benchmark_generate
from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import align_deepseek_softmax_scale, load_quantized_model


@dataclass(frozen=True)
class ServerSettings:
    model_path: str
    model_name: str
    served_model_name: str
    device: str = "cuda"
    trust_remote_code: bool = False
    eos_check_interval: int = 8
    prompt_format: str = "auto"
    max_batch_size: int = 1
    batch_wait_ms: float = 20.0
    max_batch_padding_tokens: int = 256
    max_queue_size: int = 128


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions request used by EvalScope."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    top_k: Optional[int] = None
    stop: Optional[str | list[str]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1)


@dataclass
class _GenerationJob:
    request: ChatCompletionRequest
    input_ids: torch.Tensor
    prompt_tokens: int
    max_new_tokens: int
    future: concurrent.futures.Future


_STOP_WORKER = object()


class GEMQOpenAIService:

    def __init__(self, settings: ServerSettings):
        if settings.eos_check_interval < 1:
            raise ValueError("eos_check_interval must be at least 1")
        if settings.prompt_format not in {"auto", "raw", "chat"}:
            raise ValueError("prompt_format must be one of: auto, raw, chat")
        if settings.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if settings.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if settings.max_batch_padding_tokens < 0:
            raise ValueError("max_batch_padding_tokens must be non-negative")
        if settings.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if not settings.device.startswith("cuda"):
            raise ValueError("GEMQ real-quant API serving currently requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        requested_device = torch.device(settings.device)
        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)

        self.settings = settings
        self.prompt_format = self._resolve_prompt_format(settings)
        self._tokenizer_lock = threading.Lock()

        print(f"Loading tokenizer from {settings.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            trust_remote_code=settings.trust_remote_code,
        )
        if self.prompt_format == "chat" and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                "prompt_format=chat requires a tokenizer with a chat_template; "
                "use --prompt-format raw for a base completion model"
            )

        print(f"Loading GEMQ real-quant model from {settings.model_path}")
        self.model = load_quantized_model(
            settings.model_path,
            compute_dtype=torch.float16,
            device=settings.device,
            trust_remote_code=settings.trust_remote_code,
        )

        # DeepSeek-V2's built-in Transformers implementation may omit the YaRN mscale
        # used by the original modeling code. This helper is a no-op when no correction
        # is needed.
        align_deepseek_softmax_scale(self.model)

        print("Enabling GEMQ/GemLite inference kernels")
        prepare_for_inference(
            self.model,
            settings.model_name,
            is_fp=False,
        )
        self.model.eval()
        print(
            "GEMQ API model is ready "
            f"(served model: {settings.served_model_name}, dtype: torch.float16, "
            f"prompt format: {self.prompt_format}, max batch: {settings.max_batch_size})"
        )

        self._request_queue: queue.Queue = queue.Queue(maxsize=settings.max_queue_size)
        self._shutdown_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._worker_thread = threading.Thread(
            target=self._batch_worker,
            name="gemq-batch-worker",
            daemon=True,
        )
        self._worker_thread.start()

    @staticmethod
    def _validate_request(request: ChatCompletionRequest) -> None:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported; set stream=false")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")
        if request.top_p not in (None, 1.0):
            raise HTTPException(status_code=400, detail="top_p sampling is not currently supported; use top_p=1")

    @staticmethod
    def _batch_key(request: ChatCompletionRequest) -> tuple[float, Optional[int]]:
        top_k = request.top_k if request.top_k is None or request.top_k > 0 else None
        return request.temperature, top_k

    def _can_join_batch(
        self,
        candidate: _GenerationJob,
        batch_key: tuple[float, Optional[int]],
        min_prompt_tokens: int,
        max_prompt_tokens: int,
    ) -> bool:
        if self._batch_key(candidate.request) != batch_key:
            return False
        new_min = min(min_prompt_tokens, candidate.prompt_tokens)
        new_max = max(max_prompt_tokens, candidate.prompt_tokens)
        return new_max - new_min <= self.settings.max_batch_padding_tokens

    def submit(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Queue one OpenAI request and wait for the batch worker's response."""
        self._validate_request(request)
        if self._shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="GEMQ generation worker is shutting down")
        input_ids, prompt_tokens, max_new_tokens = self._encode_request(request)
        future: concurrent.futures.Future = concurrent.futures.Future()
        job = _GenerationJob(
            request=request,
            input_ids=input_ids,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            future=future,
        )
        with self._lifecycle_lock:
            if self._shutdown_event.is_set():
                raise HTTPException(
                    status_code=503,
                    detail="GEMQ generation worker is shutting down",
                )
            try:
                self._request_queue.put_nowait(job)
            except queue.Full as exc:
                raise HTTPException(
                    status_code=503,
                    detail="GEMQ generation queue is full",
                ) from exc
        return future.result()

    def shutdown(self) -> None:
        if not self._worker_thread.is_alive():
            return
        with self._lifecycle_lock:
            if self._shutdown_event.is_set():
                return
            self._shutdown_event.set()
            while True:
                try:
                    item = self._request_queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP_WORKER and not item.future.done():
                    item.future.set_exception(
                        HTTPException(
                            status_code=503,
                            detail="GEMQ generation worker is shutting down",
                        )
                    )
            self._request_queue.put_nowait(_STOP_WORKER)
        self._worker_thread.join(timeout=5.0)

    def _batch_worker(self) -> None:
        requested_device = torch.device(self.settings.device)
        if requested_device.type == "cuda" and requested_device.index is not None:
            torch.cuda.set_device(requested_device)

        pending: collections.deque[_GenerationJob] = collections.deque()
        stop_requested = False
        while True:
            if self._shutdown_event.is_set():
                for job in pending:
                    if not job.future.done():
                        job.future.set_exception(
                            HTTPException(
                                status_code=503,
                                detail="GEMQ generation worker is shutting down",
                            )
                        )
                break
            if pending:
                first_job = pending.popleft()
            elif stop_requested:
                break
            else:
                item = self._request_queue.get()
                if item is _STOP_WORKER:
                    break
                first_job = item

            jobs = [first_job]
            batch_key = self._batch_key(first_job.request)
            min_prompt_tokens = first_job.prompt_tokens
            max_prompt_tokens = first_job.prompt_tokens
            deadline = time.monotonic() + self.settings.batch_wait_ms / 1000.0

            # Reconsider deferred incompatible jobs once for every new batch.
            for _ in range(len(pending)):
                candidate = pending.popleft()
                if (
                    len(jobs) < self.settings.max_batch_size
                    and self._can_join_batch(
                        candidate,
                        batch_key,
                        min_prompt_tokens,
                        max_prompt_tokens,
                    )
                ):
                    jobs.append(candidate)
                    min_prompt_tokens = min(min_prompt_tokens, candidate.prompt_tokens)
                    max_prompt_tokens = max(max_prompt_tokens, candidate.prompt_tokens)
                else:
                    pending.append(candidate)

            while len(jobs) < self.settings.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _STOP_WORKER:
                    stop_requested = True
                    break
                if self._can_join_batch(
                    item,
                    batch_key,
                    min_prompt_tokens,
                    max_prompt_tokens,
                ):
                    jobs.append(item)
                    min_prompt_tokens = min(min_prompt_tokens, item.prompt_tokens)
                    max_prompt_tokens = max(max_prompt_tokens, item.prompt_tokens)
                else:
                    pending.append(item)

            try:
                if len(jobs) == 1:
                    responses = [self._complete_job(jobs[0])]
                else:
                    responses = self._complete_batch(jobs)
                if len(responses) != len(jobs):
                    raise RuntimeError(
                        "Batch generation returned a different number of responses "
                        "than requests"
                    )
                for job, response in zip(jobs, responses):
                    job.future.set_result(response)
            except Exception as exc:
                for job in jobs:
                    job.future.set_exception(exc)

    @staticmethod
    def _resolve_prompt_format(settings: ServerSettings) -> str:
        if settings.prompt_format != "auto":
            return settings.prompt_format

        # GEMQ's committed checkpoints are base completion models. Only opt into a
        # chat template when the requested model identity explicitly says that it is
        # instruction/chat tuned; the presence of tokenizer.chat_template alone is
        # not sufficient because base and chat checkpoints can share tokenizer files.
        model_identity = f"{settings.model_name} {settings.model_path}".lower()
        if "chat" in model_identity or "instruct" in model_identity:
            return "chat"
        return "raw"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text", None):
                    parts.append(str(part.get("text", "")))
            return "".join(parts)
        return str(content)

    def _build_prompt(self, messages: list[dict[str, Any]]) -> tuple[str, bool]:
        normalized = [
            {
                "role": message.get("role", "user"),
                "content": self._content_to_text(message.get("content", "")),
            }
            for message in messages
        ]

        if self.prompt_format == "chat":
            return (
                self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                False,
            )

        # EvalScope sends a benchmark prompt as a single user message. Base models
        # should continue that text directly rather than seeing User:/Assistant:
        # wrappers from a tokenizer template.
        return "\n".join(message["content"] for message in normalized), True

    def _get_eos_token_ids(self) -> set[int]:
        eos_ids: set[int] = set()
        tokenizer_eos = self.tokenizer.eos_token_id
        if isinstance(tokenizer_eos, int):
            eos_ids.add(tokenizer_eos)

        generation_config = getattr(self.model, "generation_config", None)
        config_eos = getattr(generation_config, "eos_token_id", None)
        if isinstance(config_eos, int):
            eos_ids.add(config_eos)
        elif isinstance(config_eos, (list, tuple, set)):
            eos_ids.update(int(item) for item in config_eos)

        return eos_ids

    @staticmethod
    def _truncate_at_eos(token_ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                return token_ids[:index], True
        return token_ids, False

    @staticmethod
    def _apply_stop_sequences(text: str, stop: Optional[str | list[str]]) -> tuple[str, bool]:
        if stop is None:
            return text, False
        stops = [stop] if isinstance(stop, str) else stop
        positions = [text.find(item) for item in stops if item and text.find(item) >= 0]
        if not positions:
            return text, False
        return text[:min(positions)], True

    def _encode_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[torch.Tensor, int, int]:
        # Tokenize before enqueueing so one malformed/overlong request cannot fail
        # every other member of a GPU batch. Keep queued prompts on CPU; the single
        # generation worker owns all CUDA transfers and model calls.
        with self._tokenizer_lock:
            prompt, add_special_tokens = self._build_prompt(request.messages)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
        input_ids = encoded.input_ids[0]
        prompt_tokens = int(input_ids.numel())

        max_position = int(
            getattr(
                self.model.config,
                "max_position_embeddings",
                prompt_tokens + request.max_tokens,
            )
        )
        max_new_tokens = min(request.max_tokens, max_position - prompt_tokens)
        if max_new_tokens < 1:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt has {prompt_tokens} tokens and exceeds the model context window",
            )
        return input_ids, prompt_tokens, max_new_tokens

    def _pad_token_id(self, eos_token_ids: set[int]) -> int:
        tokenizer_pad = self.tokenizer.pad_token_id
        if isinstance(tokenizer_pad, int):
            return tokenizer_pad
        if eos_token_ids:
            return min(eos_token_ids)
        tokenizer_bos = self.tokenizer.bos_token_id
        if isinstance(tokenizer_bos, int):
            return tokenizer_bos
        return 0

    def _build_response(
        self,
        request: ChatCompletionRequest,
        token_ids: list[int],
        prompt_tokens: int,
        generation_stats: dict[str, Any],
        batch_size: int,
    ) -> dict[str, Any]:
        eos_token_ids = self._get_eos_token_ids()
        generated_ids, hit_eos = self._truncate_at_eos(token_ids, eos_token_ids)
        with self._tokenizer_lock:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, hit_stop_sequence = self._apply_stop_sequences(text, request.stop)
        stopped = hit_eos or hit_stop_sequence
        with self._tokenizer_lock:
            completion_tokens = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
        print(
            "Completed request: "
            f"batch_size={batch_size}, "
            f"prompt_tokens={prompt_tokens}, "
            f"generated_tokens={generation_stats['generated_tokens']}, "
            f"stopped_on_eos={generation_stats['stopped_on_eos']}, "
            f"prefill={generation_stats['prefill_latency']:.3f}s, "
            f"decode={generation_stats['decode_latency']:.3f}s, "
            f"aggregate_decode_tps={generation_stats['decode_throughput']:.2f}"
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop" if stopped else "length",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Submit a request through the same single-GPU worker used by HTTP calls."""
        return self.submit(request)

    @torch.inference_mode()
    def _complete_job(self, job: _GenerationJob) -> dict[str, Any]:
        request = job.request
        input_ids = job.input_ids.to(self.settings.device)
        prompt_tokens = job.prompt_tokens
        max_new_tokens = job.max_new_tokens

        top_k = request.top_k
        if top_k is not None and top_k <= 0:
            top_k = None
        cache = StaticCache(
            self.model.config,
            max_cache_len=prompt_tokens + max_new_tokens,
        )
        sampling_args = {
            "temperature": request.temperature,
            "top_k": top_k,
        }
        eos_token_ids = self._get_eos_token_ids()

        output, generation_stats = benchmark_generate.generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            kv_cache=cache,
            eos_token_ids=eos_token_ids,
            eos_check_interval=self.settings.eos_check_interval,
            **sampling_args,
        )
        return self._build_response(
            request=request,
            token_ids=output[prompt_tokens:].tolist(),
            prompt_tokens=prompt_tokens,
            generation_stats=generation_stats,
            batch_size=1,
        )

    @torch.inference_mode()
    def _complete_batch(
        self,
        jobs: list[_GenerationJob],
    ) -> list[dict[str, Any]]:
        requests = [job.request for job in jobs]
        prompts = [job.input_ids.to(self.settings.device) for job in jobs]
        prompt_lengths = [job.prompt_tokens for job in jobs]
        generation_limits = [job.max_new_tokens for job in jobs]
        eos_token_ids = self._get_eos_token_ids()

        top_k = requests[0].top_k
        if top_k is not None and top_k <= 0:
            top_k = None
        result = batched_generate.generate_batch(
            self.model,
            prompts=prompts,
            max_new_tokens=generation_limits,
            pad_token_id=self._pad_token_id(eos_token_ids),
            eos_token_ids=eos_token_ids,
            eos_check_interval=self.settings.eos_check_interval,
            temperature=requests[0].temperature,
            top_k=top_k,
        )

        responses = []
        for index, request in enumerate(requests):
            generation_stats = {
                "generated_tokens": result.generated_tokens[index],
                "stopped_on_eos": result.stopped_on_eos[index],
                "prefill_latency": result.prefill_latency,
                "decode_latency": result.decode_latency,
                "decode_throughput": result.decode_throughput,
            }
            responses.append(
                self._build_response(
                    request=request,
                    token_ids=result.token_ids[index],
                    prompt_tokens=prompt_lengths[index],
                    generation_stats=generation_stats,
                    batch_size=len(requests),
                )
            )
        return responses


def create_app(settings: ServerSettings) -> FastAPI:
    service = GEMQOpenAIService(settings)
    app = FastAPI(title="GEMQ OpenAI-compatible API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.served_model_name,
            "max_batch_size": settings.max_batch_size,
            "queued_requests": service._request_queue.qsize(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.model != settings.served_model_name:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Unknown model {request.model!r}; "
                    f"use {settings.served_model_name!r}"
                ),
            )
        return service.submit(request)

    @app.on_event("shutdown")
    def shutdown_service() -> None:
        service.shutdown()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a GEMQ real-quant model through an OpenAI-compatible API")
    parser.add_argument("--model-path", required=True, help="Path to a GEMQ real-quant checkpoint")
    parser.add_argument(
        "--model-name",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Original model name used by GEMQ to select the fused MoE implementation",
    )
    parser.add_argument(
        "--served-model-name",
        default="gemq-deepseek-v2-lite",
        help="Model ID accepted by the OpenAI-compatible endpoint",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device used for inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prompt-format",
        choices=("auto", "raw", "chat"),
        default="auto",
        help=(
            "Prompt formatting mode. auto uses raw completion prompts for base models "
            "and chat templates for model names containing 'chat' or 'instruct'."
        ),
    )
    parser.add_argument(
        "--eos-check-interval",
        type=int,
        default=8,
        help=(
            "Check generated tokens for EOS every N decode steps. Smaller values stop "
            "sooner but force more CUDA synchronizations."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum number of compatible HTTP requests combined into one model batch.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=20.0,
        help="Maximum time to wait for compatible requests before launching a batch.",
    )
    parser.add_argument(
        "--max-batch-padding-tokens",
        type=int,
        default=256,
        help=(
            "Maximum prompt-length difference within one batch. Smaller values "
            "reduce left-padding and KV-cache waste."
        ),
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=128,
        help="Maximum number of queued generation requests before returning HTTP 503.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Load modeling code stored with the checkpoint. Do not enable this for "
            "DeepSeek-V2-Lite generation with the current GEMQ StaticCache path."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ServerSettings(
        model_path=args.model_path,
        model_name=args.model_name,
        served_model_name=args.served_model_name,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        eos_check_interval=args.eos_check_interval,
        prompt_format=args.prompt_format,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_padding_tokens=args.max_batch_padding_tokens,
        max_queue_size=args.max_queue_size,
    )

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
