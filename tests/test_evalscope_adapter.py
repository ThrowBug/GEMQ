import collections
import concurrent.futures
import queue
import threading

import pytest

pytest.importorskip("evalscope")

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput

from gemq.evalscope_adapter import GEMQEvalScopeAPI, _GenerationJob


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        return f"User: {messages[0]['content']}\n\nAssistant:"


def _job(prompt_tokens=10, temperature=0.0, max_batch_size=2):
    return _GenerationJob(
        input_ids=None,
        prompt_tokens=prompt_tokens,
        max_new_tokens=8,
        temperature=temperature,
        top_k=None,
        stop_sequences=None,
        max_batch_size=max_batch_size,
        submitted_at=0.0,
        enqueued_at=0.0,
        future=concurrent.futures.Future(),
    )


def test_raw_and_chat_prompt_formatting():
    service = object.__new__(GEMQEvalScopeAPI)
    service.tokenizer = FakeTokenizer()
    messages = [ChatMessageUser(content="Question")]

    service.prompt_format = "raw"
    assert service._build_prompt(messages) == ("Question", True)

    service.prompt_format = "chat"
    assert service._build_prompt(messages) == ("User: Question\n\nAssistant:", False)


def test_auto_prompt_format_uses_raw_for_the_base_checkpoint():
    service = object.__new__(GEMQEvalScopeAPI)
    service.base_model_name = "deepseek-ai/DeepSeek-V2-Lite"
    service.model_path = "/models/gemq-checkpoint"

    assert service._resolve_prompt_format("auto") == "raw"

    service.base_model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert service._resolve_prompt_format("auto") == "chat"


def test_batch_compatibility_checks_sampling_batch_size_and_padding():
    service = object.__new__(GEMQEvalScopeAPI)
    service.max_batch_padding_tokens = 8
    first = _job(prompt_tokens=10)
    batch_key = service._batch_key(first)

    assert service._can_join_batch(_job(prompt_tokens=18), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=19), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=12, temperature=0.7), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=12, max_batch_size=4), batch_key, 10, 10)


def test_worker_combines_concurrent_evalscope_generate_calls():
    service = object.__new__(GEMQEvalScopeAPI)
    service.model_name = "test-model"
    service.device = "cpu"
    service.batch_wait_ms = 100.0
    service.max_batch_padding_tokens = 256
    service._request_queue = queue.Queue(maxsize=8)
    service._shutdown_event = threading.Event()
    service._lifecycle_lock = threading.Lock()
    service._prepare_job = lambda messages, config, submitted_at: _job(
        max_batch_size=config.batch_size or 1
    )
    service._run_jobs = lambda jobs: [
        ModelOutput.from_content(
            model="test-model",
            content=f"batch-size={len(jobs)}",
        )
        for _ in jobs
    ]
    service._worker_thread = threading.Thread(
        target=service._batch_worker,
        daemon=True,
    )
    service._worker_thread.start()

    config = GenerateConfig(batch_size=2, temperature=0, max_tokens=8)
    messages = [ChatMessageUser(content="Question")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.generate, messages, [], "none", config)
            for _ in range(2)
        ]
        outputs = [future.result() for future in futures]

    service.close()
    assert [output.completion for output in outputs] == ["batch-size=2", "batch-size=2"]


def test_fail_pending_completes_all_futures_with_an_error():
    service = object.__new__(GEMQEvalScopeAPI)
    queued = _job()
    pending = _job()
    service._request_queue = queue.Queue()
    service._request_queue.put(queued)

    service._fail_pending(collections.deque([pending]))

    assert isinstance(queued.future.exception(), RuntimeError)
    assert isinstance(pending.future.exception(), RuntimeError)
