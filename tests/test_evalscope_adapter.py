import collections
import concurrent.futures
import queue
import threading
import time

import pytest

pytest.importorskip("evalscope")

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput

import gemq.evalscope_adapter as evalscope_adapter
from gemq.evalscope_adapter import GEMQEvalScopeAPI, _DeviceWorker, _GenerationJob


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


def _fake_service(worker_count=1, batch_wait_ms=100.0):
    service = object.__new__(GEMQEvalScopeAPI)
    service.model_name = "test-model"
    service.devices = [f"cpu:{index}" for index in range(worker_count)]
    service.device = service.devices[0]
    service.batch_wait_ms = batch_wait_ms
    service.max_batch_padding_tokens = 256
    service.per_device_batch_size_cap = None
    service._shutdown_event = threading.Event()
    service._lifecycle_lock = threading.Lock()
    service._dispatch_lock = threading.RLock()
    service._next_worker_index = 0
    service._workers = [
        _DeviceWorker(
            worker_id=index,
            device=device,
            model=None,
            request_queue=queue.Queue(maxsize=8),
        )
        for index, device in enumerate(service.devices)
    ]
    service._prepare_job = lambda messages, config, submitted_at: _job(
        max_batch_size=service._worker_batch_size(config.batch_size or 1)
    )
    service._run_jobs = lambda worker, jobs: [
        ModelOutput.from_content(
            model="test-model",
            content=f"device={worker.device};batch-size={len(jobs)}",
        )
        for _ in jobs
    ]
    for worker in service._workers:
        worker.thread = threading.Thread(
            target=service._batch_worker,
            args=(worker,),
            daemon=True,
        )
        worker.thread.start()
    return service


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


def test_device_normalization_rejects_conflicts_and_duplicate_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)

    assert GEMQEvalScopeAPI._normalize_devices(None, ["cuda", "cuda:1"]) == [
        "cuda:0",
        "cuda:1",
    ]
    with pytest.raises(ValueError, match="either device or devices"):
        GEMQEvalScopeAPI._normalize_devices("cuda:0", ["cuda:1"])
    with pytest.raises(ValueError, match="Duplicate CUDA device"):
        GEMQEvalScopeAPI._normalize_devices(None, ["cuda", "cuda:0"])


def test_triton_autotuner_calls_are_serialized_across_worker_threads(monkeypatch):
    state_lock = threading.Lock()
    active_calls = 0
    peak_active_calls = 0

    class FakeAutotuner:
        def run(self):
            nonlocal active_calls, peak_active_calls
            with state_lock:
                active_calls += 1
                peak_active_calls = max(peak_active_calls, active_calls)
            time.sleep(0.02)
            with state_lock:
                active_calls -= 1

    monkeypatch.setattr("triton.runtime.autotuner.Autotuner", FakeAutotuner)
    monkeypatch.setattr(evalscope_adapter, "_TRITON_AUTOTUNER_PATCHED", False)

    evalscope_adapter._install_thread_safe_triton_autotuner()
    installed_run = FakeAutotuner.run
    evalscope_adapter._install_thread_safe_triton_autotuner()
    assert FakeAutotuner.run is installed_run

    autotuner = FakeAutotuner()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(autotuner.run) for _ in range(2)]
        for future in futures:
            future.result()

    assert peak_active_calls == 1


def test_batch_compatibility_checks_sampling_batch_size_and_padding():
    service = object.__new__(GEMQEvalScopeAPI)
    service.max_batch_padding_tokens = 8
    first = _job(prompt_tokens=10)
    batch_key = service._batch_key(first)

    assert service._can_join_batch(_job(prompt_tokens=18), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=19), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=12, temperature=0.7), batch_key, 10, 10)
    assert not service._can_join_batch(_job(prompt_tokens=12, max_batch_size=4), batch_key, 10, 10)


def test_worker_batch_size_preserves_single_gpu_and_splits_multi_gpu_capacity():
    service = object.__new__(GEMQEvalScopeAPI)
    service.per_device_batch_size_cap = None
    service.devices = ["cuda:0"]
    assert service._worker_batch_size(8) == 8

    service.devices = ["cuda:0", "cuda:1"]
    assert service._worker_batch_size(8) == 4

    service.per_device_batch_size_cap = 2
    assert service._worker_batch_size(8) == 2


def test_single_worker_combines_concurrent_evalscope_generate_calls():
    service = _fake_service(worker_count=1)
    config = GenerateConfig(batch_size=2, temperature=0, max_tokens=8)
    messages = [ChatMessageUser(content="Question")]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(service.generate, messages, [], "none", config)
                for _ in range(2)
            ]
            outputs = [future.result() for future in futures]
    finally:
        service.close()

    assert [output.completion for output in outputs] == [
        "device=cpu:0;batch-size=2",
        "device=cpu:0;batch-size=2",
    ]


def test_multi_worker_dispatches_and_batches_on_each_device():
    service = _fake_service(worker_count=2, batch_wait_ms=200.0)
    service.per_device_batch_size_cap = 2
    unsynchronized_run_jobs = service._run_jobs
    workers_ready = threading.Barrier(2)
    unsynchronized_prepare_job = service._prepare_job
    callers_ready = threading.Barrier(4)

    def synchronized_run_jobs(worker, jobs):
        workers_ready.wait(timeout=2.0)
        return unsynchronized_run_jobs(worker, jobs)

    def synchronized_prepare_job(messages, generation_config, submitted_at):
        job = unsynchronized_prepare_job(messages, generation_config, submitted_at)
        callers_ready.wait(timeout=2.0)
        return job

    service._run_jobs = synchronized_run_jobs
    service._prepare_job = synchronized_prepare_job
    config = GenerateConfig(batch_size=4, temperature=0, max_tokens=8)
    messages = [ChatMessageUser(content="Question")]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(service.generate, messages, [], "none", config)
                for _ in range(4)
            ]
            outputs = [future.result() for future in futures]
    finally:
        service.close()

    completions = [output.completion for output in outputs]
    assert completions.count("device=cpu:0;batch-size=2") == 2
    assert completions.count("device=cpu:1;batch-size=2") == 2
    assert all(worker.outstanding_jobs == 0 for worker in service._workers)


def test_fail_pending_completes_futures_and_releases_worker_load():
    service = object.__new__(GEMQEvalScopeAPI)
    service._dispatch_lock = threading.RLock()
    worker = _DeviceWorker(
        worker_id=0,
        device="cpu:0",
        model=None,
        request_queue=queue.Queue(),
        outstanding_jobs=2,
    )
    queued = _job()
    pending = _job()
    for job in (queued, pending):
        job.assigned_worker_id = worker.worker_id
    worker.request_queue.put(queued)

    service._fail_pending(worker, collections.deque([pending]))

    assert isinstance(queued.future.exception(), RuntimeError)
    assert isinstance(pending.future.exception(), RuntimeError)
    assert worker.outstanding_jobs == 0
