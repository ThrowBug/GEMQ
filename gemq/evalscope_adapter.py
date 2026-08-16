"""EvalScope local-model adapter for GEMQ real-quant checkpoints.

EvalScope invokes ``ModelAPI.generate`` concurrently from its evaluation thread pool.
This adapter turns those independent calls into static GPU micro-batches, so no HTTP
server or OpenAI-compatible serialization layer is required.
"""

from __future__ import annotations

import atexit
import collections
import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from evalscope.api.messages import ChatMessage, ChatMessageAssistant
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import (
    ChatCompletionChoice,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
)
from evalscope.api.tool import ToolChoice, ToolInfo

from gemq import batched_generate, benchmark_generate
from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference
from gemq.utils.hf_loading import align_deepseek_softmax_scale, load_quantized_model


@dataclass
class _GenerationJob:
    input_ids: torch.Tensor
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_k: Optional[int]
    stop_sequences: Optional[list[str]]
    max_batch_size: int
    submitted_at: float
    enqueued_at: float
    future: concurrent.futures.Future[ModelOutput]


_STOP_WORKER = object()


class GEMQEvalScopeAPI(ModelAPI):
    """Run a GEMQ checkpoint directly inside EvalScope with static batching."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        model_path: Optional[str] = None,
        base_model_name: str = "deepseek-ai/DeepSeek-V2-Lite",
        device: str = "cuda",
        precision: str = "torch.float16",
        trust_remote_code: bool = False,
        prompt_format: str = "auto",
        eos_check_interval: int = 8,
        batch_wait_ms: float = 20.0,
        max_batch_padding_tokens: int = 256,
        max_batch_size: Optional[int] = None,
        max_queue_size: int = 128,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        if model_args:
            unknown = ", ".join(sorted(model_args))
            raise ValueError(f"Unsupported GEMQ model_args: {unknown}")
        if precision not in {"float16", "torch.float16", "half", "torch.half", "fp16"}:
            raise ValueError("GEMQ inference currently requires FP16 compute precision")
        if prompt_format not in {"auto", "raw", "chat"}:
            raise ValueError("prompt_format must be one of: auto, raw, chat")
        if eos_check_interval < 1:
            raise ValueError("eos_check_interval must be at least 1")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if max_batch_padding_tokens < 0:
            raise ValueError("max_batch_padding_tokens must be non-negative")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1 when provided")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if not device.startswith("cuda"):
            raise ValueError("GEMQ real-quant inference currently requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        requested_device = torch.device(device)
        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)

        self.model_path = model_path or model_name
        self.base_model_name = base_model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.prompt_format = self._resolve_prompt_format(prompt_format)
        self.eos_check_interval = eos_check_interval
        self.batch_wait_ms = batch_wait_ms
        self.max_batch_padding_tokens = max_batch_padding_tokens
        self.max_batch_size_cap = max_batch_size
        self._tokenizer_lock = threading.Lock()

        print(f"Loading EvalScope tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.prompt_format == "chat" and not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                "prompt_format=chat requires a tokenizer with a chat_template; "
                "use prompt_format=raw for a base completion model"
            )

        print(f"Loading GEMQ checkpoint for EvalScope from {self.model_path}")
        self.model = load_quantized_model(
            self.model_path,
            compute_dtype=torch.float16,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        align_deepseek_softmax_scale(self.model)
        prepare_for_inference(
            self.model,
            self.base_model_name,
            is_fp=False,
        )
        self.model.eval()

        self._request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._worker_thread = threading.Thread(
            target=self._batch_worker,
            name="gemq-evalscope-batch-worker",
            daemon=True,
        )
        self._worker_thread.start()
        atexit.register(self.close)
        print(
            "GEMQ EvalScope adapter is ready "
            f"(model={self.model_name}, dtype=torch.float16, prompt_format={self.prompt_format})"
        )

    def _resolve_prompt_format(self, prompt_format: str) -> str:
        if prompt_format != "auto":
            return prompt_format
        model_identity = f"{self.base_model_name} {self.model_path}".lower()
        if "chat" in model_identity or "instruct" in model_identity:
            return "chat"
        return "raw"

    def _build_prompt(self, messages: list[ChatMessage]) -> tuple[str, bool]:
        for message in messages:
            if isinstance(message.content, list) and any(
                getattr(part, "type", None) != "text" for part in message.content
            ):
                raise ValueError("GEMQ local evaluation supports text messages only")
        normalized = [
            {
                "role": message.role,
                "content": message.text,
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
        return "\n".join(message["content"] for message in normalized), True

    @staticmethod
    def _normalize_stop_sequences(stop_sequences: Optional[list[str]]) -> Optional[list[str]]:
        if not stop_sequences:
            return None
        normalized = [item for item in stop_sequences if item]
        return normalized or None

    def _prepare_job(
        self,
        messages: list[ChatMessage],
        config: GenerateConfig,
        submitted_at: float,
    ) -> _GenerationJob:
        if config.stream:
            raise ValueError("GEMQ local evaluation does not support streaming")
        if config.n not in (None, 1):
            raise ValueError("GEMQ local evaluation currently supports only n=1")
        if config.top_p not in (None, 1.0):
            raise ValueError("GEMQ local evaluation currently supports only top_p=1")
        if config.logprobs:
            raise ValueError("GEMQ local evaluation does not currently return logprobs")
        if config.top_logprobs not in (None, 0):
            raise ValueError("GEMQ local evaluation does not currently return top_logprobs")
        if config.frequency_penalty not in (None, 0.0):
            raise ValueError("GEMQ local evaluation does not support frequency_penalty")
        if config.presence_penalty not in (None, 0.0):
            raise ValueError("GEMQ local evaluation does not support presence_penalty")
        if config.repetition_penalty not in (None, 1.0):
            raise ValueError("GEMQ local evaluation does not support repetition_penalty")

        if config.do_sample is False:
            temperature = 0.0
        elif config.temperature is not None:
            temperature = float(config.temperature)
        else:
            temperature = 1.0 if config.do_sample else 0.0
        top_k = config.top_k if config.top_k is None or config.top_k > 0 else None
        max_tokens = config.max_tokens if config.max_tokens is not None else self.max_tokens()
        if max_tokens is None or max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        with self._tokenizer_lock:
            prompt, add_special_tokens = self._build_prompt(messages)
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
                prompt_tokens + max_tokens,
            )
        )
        max_new_tokens = min(int(max_tokens), max_position - prompt_tokens)
        if max_new_tokens < 1:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens and exceeds the model context window"
            )

        requested_batch_size = config.batch_size or 1
        if requested_batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_batch_size_cap is not None:
            requested_batch_size = min(requested_batch_size, self.max_batch_size_cap)
        return _GenerationJob(
            input_ids=input_ids,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_sequences=self._normalize_stop_sequences(config.stop_seqs),
            max_batch_size=max(1, int(requested_batch_size)),
            submitted_at=submitted_at,
            enqueued_at=time.perf_counter(),
            future=concurrent.futures.Future(),
        )

    def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Enqueue one EvalScope sample and return its local GEMQ output."""
        if tools:
            raise ValueError("GEMQ local evaluation does not support tool calls")
        if tool_choice not in (None, "none", "auto"):
            raise ValueError("GEMQ local evaluation does not support tool_choice")
        submitted_at = time.perf_counter()
        job = self._prepare_job(input, config, submitted_at)
        with self._lifecycle_lock:
            if self._shutdown_event.is_set():
                raise RuntimeError("GEMQ EvalScope worker is shutting down")
            if not self._worker_thread.is_alive():
                raise RuntimeError("GEMQ EvalScope worker stopped unexpectedly")
            try:
                self._request_queue.put_nowait(job)
            except queue.Full as exc:
                raise RuntimeError("GEMQ EvalScope generation queue is full") from exc
        timeout = None
        if config.timeout is not None:
            timeout = max(0.0, config.timeout - (time.perf_counter() - submitted_at))
        try:
            return job.future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            if job.future.done():
                raise
            job.future.cancel()
            raise TimeoutError(
                f"GEMQ generation exceeded the configured timeout of {config.timeout} seconds"
            ) from exc

    @staticmethod
    def _batch_key(job: _GenerationJob) -> tuple[float, Optional[int], int]:
        return job.temperature, job.top_k, job.max_batch_size

    def _can_join_batch(
        self,
        candidate: _GenerationJob,
        batch_key: tuple[float, Optional[int], int],
        min_prompt_tokens: int,
        max_prompt_tokens: int,
    ) -> bool:
        if self._batch_key(candidate) != batch_key:
            return False
        new_min = min(min_prompt_tokens, candidate.prompt_tokens)
        new_max = max(max_prompt_tokens, candidate.prompt_tokens)
        return new_max - new_min <= self.max_batch_padding_tokens

    def _batch_worker(self) -> None:
        requested_device = torch.device(self.device)
        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)

        pending: collections.deque[_GenerationJob] = collections.deque()
        while True:
            if self._shutdown_event.is_set():
                self._fail_pending(pending)
                break
            if pending:
                first_job = pending.popleft()
            else:
                item = self._request_queue.get()
                if item is _STOP_WORKER:
                    break
                first_job = item
            if first_job.future.cancelled():
                continue

            jobs = [first_job]
            batch_key = self._batch_key(first_job)
            max_batch_size = first_job.max_batch_size
            min_prompt_tokens = first_job.prompt_tokens
            max_prompt_tokens = first_job.prompt_tokens
            deadline = time.monotonic() + self.batch_wait_ms / 1000.0

            for _ in range(len(pending)):
                candidate = pending.popleft()
                if candidate.future.cancelled():
                    continue
                if len(jobs) < max_batch_size and self._can_join_batch(
                    candidate,
                    batch_key,
                    min_prompt_tokens,
                    max_prompt_tokens,
                ):
                    jobs.append(candidate)
                    min_prompt_tokens = min(min_prompt_tokens, candidate.prompt_tokens)
                    max_prompt_tokens = max(max_prompt_tokens, candidate.prompt_tokens)
                else:
                    pending.append(candidate)

            while len(jobs) < max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _STOP_WORKER:
                    self._shutdown_event.set()
                    break
                if item.future.cancelled():
                    continue
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
                outputs = self._run_jobs(jobs)
                if len(outputs) != len(jobs):
                    raise RuntimeError("GEMQ batch returned a different number of outputs than requests")
                for job, output in zip(jobs, outputs):
                    if not job.future.done():
                        job.future.set_result(output)
            except Exception as exc:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)

    def _fail_pending(self, pending: collections.deque[_GenerationJob]) -> None:
        error = RuntimeError("GEMQ EvalScope worker is shutting down")
        for job in pending:
            if not job.future.done():
                job.future.set_exception(error)
        while True:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP_WORKER and not item.future.done():
                item.future.set_exception(error)

    def _run_jobs(self, jobs: list[_GenerationJob]) -> list[ModelOutput]:
        batch_started_at = time.perf_counter()
        if len(jobs) == 1:
            token_ids, stats = self._generate_one(jobs[0])
            return [
                self._build_output(
                    jobs[0],
                    token_ids,
                    stats,
                    batch_size=1,
                    batch_started_at=batch_started_at,
                )
            ]

        eos_token_ids = self._get_eos_token_ids()
        result = batched_generate.generate_batch(
            self.model,
            prompts=[job.input_ids.to(self.device) for job in jobs],
            max_new_tokens=[job.max_new_tokens for job in jobs],
            pad_token_id=self._pad_token_id(eos_token_ids),
            eos_token_ids=eos_token_ids,
            eos_check_interval=self.eos_check_interval,
            temperature=jobs[0].temperature,
            top_k=jobs[0].top_k,
        )
        outputs = []
        for index, job in enumerate(jobs):
            stats = {
                "generated_tokens": result.generated_tokens[index],
                "stopped_on_eos": result.stopped_on_eos[index],
                "prefill_latency": result.prefill_latency,
                "decode_latency": result.decode_latency,
                "decode_throughput": result.decode_throughput,
            }
            outputs.append(
                self._build_output(
                    job,
                    result.token_ids[index],
                    stats,
                    batch_size=len(jobs),
                    batch_started_at=batch_started_at,
                )
            )
        return outputs

    @torch.inference_mode()
    def _generate_one(self, job: _GenerationJob) -> tuple[list[int], dict[str, Any]]:
        input_ids = job.input_ids.to(self.device)
        cache = StaticCache(
            self.model.config,
            max_cache_len=job.prompt_tokens + job.max_new_tokens,
        )
        output, stats = benchmark_generate.generate(
            self.model,
            input_ids,
            max_new_tokens=job.max_new_tokens,
            kv_cache=cache,
            eos_token_ids=self._get_eos_token_ids(),
            eos_check_interval=self.eos_check_interval,
            temperature=job.temperature,
            top_k=job.top_k,
        )
        return output[job.prompt_tokens :].tolist(), stats

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

    def _pad_token_id(self, eos_token_ids: set[int]) -> int:
        if isinstance(self.tokenizer.pad_token_id, int):
            return self.tokenizer.pad_token_id
        if eos_token_ids:
            return min(eos_token_ids)
        if isinstance(self.tokenizer.bos_token_id, int):
            return self.tokenizer.bos_token_id
        return 0

    @staticmethod
    def _truncate_at_eos(token_ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                return token_ids[:index], True
        return token_ids, False

    @staticmethod
    def _apply_stop_sequences(text: str, stop_sequences: Optional[list[str]]) -> tuple[str, bool]:
        if not stop_sequences:
            return text, False
        positions = [text.find(item) for item in stop_sequences if text.find(item) >= 0]
        if not positions:
            return text, False
        return text[: min(positions)], True

    def _build_output(
        self,
        job: _GenerationJob,
        token_ids: list[int],
        stats: dict[str, Any],
        batch_size: int,
        batch_started_at: float,
    ) -> ModelOutput:
        generated_ids, hit_eos = self._truncate_at_eos(token_ids, self._get_eos_token_ids())
        with self._tokenizer_lock:
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, hit_stop_sequence = self._apply_stop_sequences(text, job.stop_sequences)
        with self._tokenizer_lock:
            completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))

        latency = time.perf_counter() - job.submitted_at
        message = ChatMessageAssistant(
            content=text,
            model=self.model_name,
            source="generate",
            perf_metrics=PerformanceMetrics(
                latency=latency,
                ttft=None,
                input_tokens=job.prompt_tokens,
                output_tokens=completion_tokens,
            ),
        )
        stopped = hit_eos or hit_stop_sequence
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    stop_reason="stop" if stopped else "max_tokens",
                )
            ],
            usage=ModelUsage(
                input_tokens=job.prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=job.prompt_tokens + completion_tokens,
            ),
            time=latency,
            metadata={
                "gemq": {
                    "batch_size": batch_size,
                    "queue_wait": max(0.0, batch_started_at - job.enqueued_at),
                    "generated_tokens": int(stats["generated_tokens"]),
                    "stopped_on_eos": bool(stats["stopped_on_eos"]),
                    "prefill_latency": float(stats["prefill_latency"]),
                    "decode_latency": float(stats["decode_latency"]),
                    "aggregate_decode_tps": float(stats["decode_throughput"]),
                }
            },
        )

    def max_tokens(self) -> Optional[int]:
        """Return the default output-token budget for EvalScope."""
        return 2048

    def max_connections(self) -> int:
        """Return a conservative default batch size when EvalScope provides none."""
        return self.max_batch_size_cap or 1

    async def aclose(self) -> None:
        """Stop the local batch worker."""
        self.close()

    def close(self) -> None:
        """Stop accepting work and wake the batch worker."""
        worker = getattr(self, "_worker_thread", None)
        if worker is None or not worker.is_alive():
            return
        with self._lifecycle_lock:
            if self._shutdown_event.is_set():
                return
            self._shutdown_event.set()
            self._fail_pending(collections.deque())
            self._request_queue.put_nowait(_STOP_WORKER)
        worker.join(timeout=5.0)
