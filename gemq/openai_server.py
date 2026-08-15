"""Minimal OpenAI-compatible API server for GEMQ real-quant checkpoints.

Run from the repository root after installing the ``api`` extra::

    pip install -e ".[api]"
    python -m gemq.openai_server \
        --model-path results/real_quant_models/deepseek-ai/DeepSeek-V2-Lite/GEMQ/... \
        --model-name deepseek-ai/DeepSeek-V2-Lite

The server intentionally uses one process and serializes generation requests. GEMQ's
current real-quant generation path is single-GPU and each request owns a StaticCache.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from gemq import benchmark_generate
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


class GEMQOpenAIService:

    def __init__(self, settings: ServerSettings):
        if settings.eos_check_interval < 1:
            raise ValueError("eos_check_interval must be at least 1")
        if not settings.device.startswith("cuda"):
            raise ValueError("GEMQ real-quant API serving currently requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        requested_device = torch.device(settings.device)
        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)

        self.settings = settings
        self._generate_lock = threading.Lock()

        print(f"Loading tokenizer from {settings.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            trust_remote_code=settings.trust_remote_code,
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
            f"(served model: {settings.served_model_name}, dtype: torch.float16)"
        )

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

        if getattr(self.tokenizer, "chat_template", None):
            return (
                self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                False,
            )

        # DeepSeek-V2-Lite is a base model and may not define a chat template. Most
        # EvalScope benchmark calls contain one user message, for which this preserves
        # the raw benchmark prompt.
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

    @torch.inference_mode()
    def complete(self, request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported; set stream=false")
        if request.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is currently supported")
        if request.top_p not in (None, 1.0):
            raise HTTPException(status_code=400, detail="top_p sampling is not currently supported; use top_p=1")

        prompt, add_special_tokens = self._build_prompt(request.messages)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        input_ids = encoded.input_ids[0].to(self.settings.device)
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

        top_k = request.top_k
        if top_k is not None and top_k <= 0:
            top_k = None
        # The GEMQ demo sampler always samples. Restricting the candidate set to one
        # token makes temperature=0 deterministic greedy decoding.
        if request.temperature == 0:
            top_k = 1

        cache = StaticCache(
            self.model.config,
            max_cache_len=prompt_tokens + max_new_tokens,
        )
        sampling_args = {
            "temperature": max(request.temperature, 1e-5),
            "top_k": top_k,
        }
        eos_token_ids = self._get_eos_token_ids()

        with self._generate_lock:
            output, generation_stats = benchmark_generate.generate(
                self.model,
                input_ids,
                max_new_tokens=max_new_tokens,
                kv_cache=cache,
                eos_token_ids=eos_token_ids,
                eos_check_interval=self.settings.eos_check_interval,
                **sampling_args,
            )

        generated_ids, hit_eos = self._truncate_at_eos(
            output[prompt_tokens:].tolist(),
            eos_token_ids,
        )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, hit_stop_sequence = self._apply_stop_sequences(text, request.stop)
        stopped = hit_eos or hit_stop_sequence
        completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        print(
            "Completed request: "
            f"prompt_tokens={prompt_tokens}, "
            f"generated_tokens={generation_stats['generated_tokens']}, "
            f"stopped_on_eos={generation_stats['stopped_on_eos']}, "
            f"prefill={generation_stats['prefill_latency']:.3f}s, "
            f"decode={generation_stats['decode_latency']:.3f}s, "
            f"decode_tps={generation_stats['decode_throughput']:.2f}"
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


def create_app(settings: ServerSettings) -> FastAPI:
    service = GEMQOpenAIService(settings)
    app = FastAPI(title="GEMQ OpenAI-compatible API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.served_model_name}

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
        return service.complete(request)

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
        "--eos-check-interval",
        type=int,
        default=8,
        help=(
            "Check generated tokens for EOS every N decode steps. Smaller values stop "
            "sooner but force more CUDA synchronizations."
        ),
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
    )

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
