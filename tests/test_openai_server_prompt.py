import concurrent.futures
import queue
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from gemq.openai_server import ChatCompletionRequest, GEMQOpenAIService, ServerSettings


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        return f"User: {messages[0]['content']}\n\nAssistant:"


def _settings(model_name="deepseek-ai/DeepSeek-V2-Lite", prompt_format="auto"):
    return ServerSettings(
        model_path="/models/checkpoint",
        model_name=model_name,
        served_model_name="test-model",
        prompt_format=prompt_format,
    )


def _prompt_service(prompt_format):
    service = object.__new__(GEMQOpenAIService)
    service.prompt_format = prompt_format
    service.tokenizer = FakeTokenizer()
    return service


def test_auto_uses_raw_for_base_model_even_when_tokenizer_has_chat_template():
    assert GEMQOpenAIService._resolve_prompt_format(_settings()) == "raw"


def test_auto_uses_chat_for_instruction_model():
    settings = _settings(model_name="deepseek-ai/DeepSeek-V2-Lite-Chat")
    assert GEMQOpenAIService._resolve_prompt_format(settings) == "chat"


def test_raw_prompt_does_not_add_chat_wrappers():
    prompt, add_special_tokens = _prompt_service("raw")._build_prompt(
        [{"role": "user", "content": "Question and few-shot examples"}]
    )

    assert prompt == "Question and few-shot examples"
    assert add_special_tokens is True


def test_chat_prompt_uses_tokenizer_template():
    prompt, add_special_tokens = _prompt_service("chat")._build_prompt(
        [{"role": "user", "content": "Question"}]
    )

    assert prompt == "User: Question\n\nAssistant:"
    assert add_special_tokens is False


def test_batch_worker_combines_concurrent_compatible_requests():
    service = object.__new__(GEMQOpenAIService)
    service.settings = SimpleNamespace(
        device="cpu",
        max_batch_size=2,
        batch_wait_ms=100.0,
        max_batch_padding_tokens=256,
    )
    service._request_queue = queue.Queue(maxsize=8)
    service._shutdown_event = threading.Event()
    service._lifecycle_lock = threading.Lock()
    service._encode_request = lambda request: (None, 10, request.max_tokens)
    service._complete_job = lambda job: {"batch_size": 1}
    service._complete_batch = lambda jobs: [
        {"batch_size": len(jobs)} for _ in jobs
    ]
    service._worker_thread = threading.Thread(
        target=service._batch_worker,
        daemon=True,
    )
    service._worker_thread.start()

    requests = [
        ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": f"Question {index}"}],
            temperature=0,
        )
        for index in range(2)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.submit, requests))

    service.shutdown()
    assert results == [{"batch_size": 2}, {"batch_size": 2}]


def test_batch_compatibility_limits_padding_and_sampling_changes():
    service = object.__new__(GEMQOpenAIService)
    service.settings = SimpleNamespace(max_batch_padding_tokens=8)
    base_request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Question"}],
        temperature=0,
    )

    nearby = SimpleNamespace(request=base_request, prompt_tokens=18)
    too_far = SimpleNamespace(request=base_request, prompt_tokens=19)
    sampled = SimpleNamespace(
        request=base_request.model_copy(update={"temperature": 0.7}),
        prompt_tokens=12,
    )
    batch_key = service._batch_key(base_request)

    assert service._can_join_batch(nearby, batch_key, 10, 10)
    assert not service._can_join_batch(too_far, batch_key, 10, 10)
    assert not service._can_join_batch(sampled, batch_key, 10, 10)
