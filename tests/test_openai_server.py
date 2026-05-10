from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch

from torchinferno.openai_http import OpenAIHandler, enable_tcp_nodelay
from torchinferno.openai_server import (
    OpenAICompletionEngine,
    OpenAIServerConfig,
    _ByteFallbackTokenizer,
    _TransformersChatTokenizer,
    _distributed_server_command,
    _effective_openai_max_batch_size,
    _should_reexec_distributed_server,
    _sync_tensor_parallel_command,
    _tensor_parallel_worker_loop,
    _warmup_prefill_cache_token_counts,
    _warmup_prompt_token_counts,
    _warmup_prefix_suffix_cache_token_counts,
    _warmup_prefix_suffix_token_counts,
    _warmup_temperature_batch_sizes,
    _warmup_temperature_prompt_token_counts,
)


def test_openai_server_matches_openai_chat_contract() -> None:
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = [
        sys.executable,
        "-m",
        "torchinferno.openai_server",
        "--model",
        "tiny",
        "--model-kind",
        "tiny-deepseek",
        "--tokenizer",
        "byte",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--max-model-len",
        "32",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_models(port, proc)
        models = _json_get(f"http://127.0.0.1:{port}/v1/models")
        assert models["object"] == "list"
        assert models["data"][0]["id"] == "tiny"

        body = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": False,
        }
        completion = _json_post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["message"]["role"] == "assistant"
        assert completion["usage"]["prompt_tokens"] > 0
        assert completion["usage"]["completion_tokens"] == 2
        assert completion["usage"]["total_tokens"] == completion["usage"]["prompt_tokens"] + 2

        stream_body = {**body, "stream": True}
        lines = _stream_post(f"http://127.0.0.1:{port}/v1/chat/completions", stream_body)
        assert any(line.startswith("data: {") and "chat.completion.chunk" in line for line in lines)
        assert "data: [DONE]" in lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_openai_handler_writes_sse_frame_with_single_socket_write() -> None:
    handler = object.__new__(OpenAIHandler)
    writer = _CountingWriter()
    handler.wfile = writer

    handler._write_sse({"choices": [{"delta": {"content": "7"}}]})

    assert writer.write_calls == 1
    assert writer.payload == b'data: {"choices":[{"delta":{"content":"7"}}]}\n\n'
    assert writer.flush_calls == 1


def test_openai_handler_enables_tcp_nodelay() -> None:
    fake_socket = _FakeSocket()

    enable_tcp_nodelay(fake_socket)

    assert fake_socket.options == [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def test_openai_server_auto_launches_tensor_parallel_for_vanilla_provider(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_AUTO_TORCHRUN", raising=False)
    config = OpenAIServerConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
    )

    assert _should_reexec_distributed_server(config)
    command = _distributed_server_command(
        config,
        (
            "--model",
            config.model,
            "--tensor-parallel-size",
            "8",
            "--port",
            "8000",
        ),
    )

    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("torchinferno.openai_server") - 1] == "-m"


def test_openai_server_warmup_uses_generic_shape_buckets(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_PROMPT_TOKEN_BUCKETS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_PREFILL_CACHE_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_PREFIX_SUFFIX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_PREFIX_SUFFIX_CACHE_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS", raising=False)

    prompt_counts = set(_warmup_prompt_token_counts(32))
    prefix_suffix_counts = set(_warmup_prefix_suffix_token_counts())

    assert prompt_counts == {16, 32, 64, 128, 256}
    assert all(count & (count - 1) == 0 for count in prompt_counts)
    assert set(_warmup_prefill_cache_token_counts()) >= {128, 256, 512, 1024}
    assert prefix_suffix_counts == {(32, 16), (64, 16), (128, 32), (256, 32)}
    assert set(_warmup_prefix_suffix_cache_token_counts()) >= {128, 256, 512, 1024}
    assert set(_warmup_temperature_prompt_token_counts()) == {32, 55, 64}
    assert 8 in set(_warmup_temperature_batch_sizes())


def test_openai_temperature_warmup_uses_configured_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_BATCH_SIZES", "3")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_TEMPERATURE_PROMPT_TOKEN_BUCKETS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_PREFILL_CACHE_TOKENS", "4")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", "1")

    model = _WarmupShapeModel()
    engine = object.__new__(OpenAICompletionEngine)
    engine.model = model
    engine.device = torch.device("cpu")
    engine.cache_backend = "dense"
    engine.page_size = 16
    engine._cache_pool = {}
    engine._microbatch_cache_pool = {}

    engine._warmup_tensor_parallel_temperature_graphs(vocab_size=16)

    assert model.prefill_shapes == [(3, 2), (1, 2)]
    assert model.decode_shapes == [(3, 1), (3, 1)]


def test_openai_server_pipeline_parallelism_skips_auto_launch(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = OpenAIServerConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
        llama_parallelism="pipeline",
    )

    assert not _should_reexec_distributed_server(config)


def test_chat_template_batch_encoding_input_ids_are_extracted() -> None:
    tokenizer = _TransformersChatTokenizer(_BatchEncodingTokenizer())

    encoded = tokenizer.encode_messages([{"role": "user", "content": "hello"}])

    assert encoded == [7, 8, 9]


def test_transformers_chat_tokenizer_stops_on_llama_eot() -> None:
    tokenizer = _TransformersChatTokenizer(_LlamaStyleTokenizer())

    assert 128001 in tokenizer.stop_token_ids
    assert 128009 in tokenizer.stop_token_ids


def test_openai_engine_stops_generation_on_chat_terminator() -> None:
    model = _ScriptedTokenModel([2, 9, 3], vocab_size=16)
    engine = OpenAICompletionEngine(
        model,
        _StopTokenTokenizer(),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "stop"}],
                max_tokens=5,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2, 9]
    assert len(model.calls) == 2


def test_openai_engine_drains_tensor_parallel_direct_generator_on_close(monkeypatch) -> None:
    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_primary_model", lambda model: True)
    model = _ScriptedTokenModel([2, 3, 4], vocab_size=16)
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=16),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    engine.single_request_fast_path = True
    try:
        generator = engine.generate_chat_tokens(
            [{"role": "user", "content": "close"}],
            max_tokens=3,
            temperature=0.0,
        )
        assert next(generator) == 2
        generator.close()
    finally:
        engine.close()

    assert len(model.calls) == 3


def test_openai_engine_tensor_parallel_primary_queues_single_stream_request(monkeypatch) -> None:
    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_primary_model", lambda model: True)
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=0.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "single"}],
                max_tokens=2,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert engine.single_request_fast_path is False
    assert tokens == [2, 2]
    assert model.calls[0][0] == 1
    assert model.calls[0][2] == "torchinferno-openai-batcher"


def test_tensor_parallel_worker_loop_uses_batched_stream_path(monkeypatch) -> None:
    import torch.distributed as dist

    commands: list[dict[str, object]] = [
        {
            "op": "generate",
            "input_ids": [[1, 2, 3]],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        },
        {"op": "stop"},
    ]

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        payload[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    engine = _WorkerLoopRecordingEngine()

    _tensor_parallel_worker_loop(engine)

    assert engine.single_calls == []
    assert engine.batch_calls == [([[1, 2, 3]], 1, 0.0, False)]


def test_openai_stream_disconnect_drains_generation() -> None:
    engine = _DrainRecordingEngine()
    handler = object.__new__(OpenAIHandler)
    handler.server = type("Server", (), {"engine": engine})()
    handler.wfile = _FailingWriter(fail_on_call=2)
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda key, value: None
    handler.end_headers = lambda: None

    handler._stream_chat([{"role": "user", "content": "hi"}], max_tokens=3, temperature=0.0)

    assert engine.drained_tokens == [1, 2, 3]
    assert handler.close_connection is True
    assert handler.wfile.write_calls == 2


def test_openai_engine_microbatches_same_shape_requests() -> None:
    model = _BatchRecordingModel()
    tokenizer = _BarrierByteFallbackTokenizer(vocab_size=8, parties=2)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
    )
    barrier = threading.Barrier(3)
    results: list[list[int] | None] = [None, None]

    def run(index: int) -> None:
        barrier.wait()
        completion = engine.complete_chat(
            [{"role": "user", "content": "same"}],
            max_tokens=2,
            temperature=0.0,
        )
        results[index] = completion.tokens

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
    engine.close()

    assert results == [[2, 2], [2, 2]]
    assert model.calls[0][0] == 2


def test_openai_engine_single_request_skips_batch_wait() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    start = time.perf_counter()
    try:
        completion = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=1,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert completion.tokens == [2]
    assert time.perf_counter() - start < 0.5
    assert model.calls[0][0] == 1
    assert model.calls[0][2] != "torchinferno-openai-batcher"


def test_openai_engine_single_stream_request_uses_direct_path() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "single"}],
                max_tokens=2,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2, 2]
    assert model.calls[0][0] == 1
    assert model.calls[0][2] != "torchinferno-openai-batcher"


def test_openai_engine_batches_request_arriving_during_single_admission_wait(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SINGLE_ADMISSION_WAIT_MS", "50")
    model = _BatchRecordingModel()
    tokenizer = _FirstEncodeEventTokenizer(vocab_size=8)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
    )
    results: list[list[int] | None] = [None, None]
    second_waiting = threading.Event()

    def run_first() -> None:
        results[0] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "same"}],
                max_tokens=2,
                temperature=0.0,
            )
        )

    def run_second() -> None:
        second_waiting.set()
        assert tokenizer.first_encoded.wait(timeout=5)
        results[1] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "same"}],
                max_tokens=2,
                temperature=0.0,
            )
        )

    second = threading.Thread(target=run_second)
    first = threading.Thread(target=run_first)
    second.start()
    assert second_waiting.wait(timeout=5)
    first.start()
    for thread in (first, second):
        thread.join(timeout=10)
    engine.close()

    assert results == [[2, 2], [2, 2]]
    assert model.calls[0][0] == 1
    assert model.calls[0][2] == "torchinferno-openai-batcher"
    assert model.calls[1][0] == 2
    assert model.calls[1][2] == "torchinferno-openai-batcher"


def test_openai_engine_temperature_request_keeps_short_batch_window(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SINGLE_ADMISSION_WAIT_MS", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TEMPERATURE_ADMISSION_WAIT_MS", "50")
    model = _BatchRecordingModel()
    tokenizer = _FirstEncodeEventTokenizer(vocab_size=8)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
        single_request_admission_wait_ms=0.0,
    )
    results: list[list[int] | None] = [None, None]
    second_waiting = threading.Event()

    def run_first() -> None:
        results[0] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "same"}],
                max_tokens=2,
                temperature=1e-6,
            )
        )

    def run_second() -> None:
        second_waiting.set()
        assert tokenizer.first_encoded.wait(timeout=5)
        results[1] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "same"}],
                max_tokens=2,
                temperature=1e-6,
            )
        )

    second = threading.Thread(target=run_second)
    first = threading.Thread(target=run_first)
    second.start()
    assert second_waiting.wait(timeout=5)
    first.start()
    for thread in (first, second):
        thread.join(timeout=10)
    engine.close()

    assert results == [[2, 2], [2, 2]]
    assert model.calls[0][0] == 1
    assert model.calls[0][2] == "torchinferno-openai-batcher"
    assert model.calls[1][0] == 2
    assert model.calls[1][2] == "torchinferno-openai-batcher"


def test_openai_engine_single_request_does_not_use_batched_step_loop() -> None:
    model = _BatchRecordingModel()
    engine = _NoBatchStepEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        completion = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert completion.tokens == [2, 2]
    assert engine.batch_step_calls == 0


def test_openai_engine_reuses_resettable_generation_cache() -> None:
    model = _BatchRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=8),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=1000.0,
    )
    try:
        first = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
        second = engine.complete_chat(
            [{"role": "user", "content": "single"}],
            max_tokens=2,
            temperature=0.0,
        )
    finally:
        engine.close()

    assert first.tokens == [2, 2]
    assert second.tokens == [2, 2]
    assert model.cache_allocations == 1


def test_openai_engine_reuses_single_request_prefix_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    model = _PrefixRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _PrefixTokenizer(),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        first = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "first"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
        second = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "second"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert first == [2]
    assert second == [2]
    assert model.forward_inputs == [[10, 11], [2, 12]]


def test_openai_engine_uses_prefill_graph_for_prefix_suffix(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    model = _PrefixGraphRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _PrefixTokenizer(),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        first = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "first"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
        second = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "second"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert first == [2]
    assert second == [2]
    assert model.graph_inputs == [[10, 11], [2, 12]]
    assert model.forward_inputs == []


def test_openai_engine_skips_runtime_prefill_graph_capture_on_miss() -> None:
    model = _RuntimePrefillGraphCaptureModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=16),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "graph miss"}],
                max_tokens=1,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2]
    assert model.capture_flags == [False]
    assert model.graph_inputs == []
    assert model.forward_inputs


def test_openai_engine_skips_runtime_shared_prefix_capture_for_tensor_parallel(monkeypatch) -> None:
    model = _RuntimePrefillLogitsGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._generate_shared_prefix_prompt_list_steps(
            [[10, 11, 12], [10, 11, 13, 14]],
            prefix_tokens=2,
            max_tokens=1,
            temperature=0.0,
        )
    )

    assert steps == [[2, 2]]
    assert model.capture_flags == [False, False, False]
    assert model.graph_inputs == []
    assert model.forward_inputs == [[10, 11], [13, 14], [12]]


def test_openai_engine_uses_decode_graph_by_default() -> None:
    model = _DecodeGraphRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=16),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "decode graph"}],
                max_tokens=2,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2, 3]
    assert model.decode_graph_calls == 1


def test_openai_engine_decode_graph_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", "0")
    model = _DecodeGraphRecordingModel()
    engine = OpenAICompletionEngine(
        model,
        _ByteFallbackTokenizer(vocab_size=16),
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=1,
        batch_wait_ms=0.0,
    )
    try:
        tokens = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": "decode graph"}],
                max_tokens=2,
                temperature=0.0,
            )
        )
    finally:
        engine.close()

    assert tokens == [2, 2]
    assert model.decode_graph_calls == 0


def test_openai_engine_batches_shared_prefix_suffixes(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _SharedPrefixRecordingModel()
    tokenizer = _SharedPrefixTokenizer(parties=2)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
    )
    barrier = threading.Barrier(3)
    results: list[list[int] | None] = [None, None]

    def run(index: int, content: str) -> None:
        barrier.wait()
        results[index] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": content}],
                max_tokens=1,
                temperature=0.0,
            )
        )

    threads = [
        threading.Thread(target=run, args=(0, "left")),
        threading.Thread(target=run, args=(1, "right")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
    engine.close()

    assert results == [[2], [2]]
    assert model.forward_inputs[0] == [[10, 11]]
    assert sorted(model.forward_inputs[1]) == [[12], [13]]


def test_openai_engine_batches_variable_length_shared_prefix_suffixes(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _SharedPrefixRecordingModel()
    tokenizer = _SharedPrefixTokenizer(parties=2)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
    )

    def run_pair() -> list[list[int] | None]:
        barrier = threading.Barrier(3)
        results: list[list[int] | None] = [None, None]

        def run(index: int, content: str) -> None:
            barrier.wait()
            results[index] = list(
                engine.generate_chat_tokens(
                    [{"role": "user", "content": content}],
                    max_tokens=1,
                    temperature=0.0,
                )
            )

        threads = [
            threading.Thread(target=run, args=(0, "left")),
            threading.Thread(target=run, args=(1, "long")),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10)
        return results

    results = run_pair()
    assert results == [[2], [2]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[13, 14]],
        [[12]],
    ]

    model.forward_inputs.clear()
    results = run_pair()
    engine.close()

    assert results == [[2], [2]]
    assert model.forward_inputs == [
        [[13, 14]],
        [[12]],
    ]


def test_openai_engine_disables_prefix_cache_for_tensor_parallel(monkeypatch) -> None:
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine._prefix_cache_entry = object()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.tensor([[10, 11]], dtype=torch.long)
    cache = model.allocate_cache(1, 8)
    model.forward(input_ids, cache=cache, use_cache=True)

    assert engine._restore_prefix_cache(input_ids, cache) == 0
    assert engine._prefix_cache_entry is None

    engine._prefix_cache_entry = object()
    engine._save_prefix_cache(input_ids, [2], cache)
    assert engine._prefix_cache_entry is None

    engine._prefix_cache_entry = object()
    assert engine._restore_exact_prefix_cache(input_ids, cache) == 0
    assert engine._prefix_cache_entry is None

    engine._prefix_cache_entry = object()
    engine._save_prompt_prefix_cache(input_ids, cache)
    assert engine._prefix_cache_entry is None


def test_openai_engine_disables_shared_prefix_batching_for_tensor_parallel(monkeypatch) -> None:
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    input_ids = torch.tensor([[10, 11, 12], [10, 11, 13]], dtype=torch.long)
    prompts = [[10, 11, 12], [10, 11, 13, 14]]

    assert engine._shared_prefix_batch_tokens(input_ids) == 2
    assert engine._shared_prefix_prompt_list_tokens(prompts) == 2

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert engine._shared_prefix_batch_tokens(input_ids) == 0
    assert engine._shared_prefix_prompt_list_tokens(prompts) == 0


def test_openai_engine_cache_pool_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", "1")
    engine = _cache_only_engine()
    model = _BatchRecordingModel()

    first = engine._generation_cache(1, 4, model=model)
    second = engine._generation_cache(2, 4, model=model)

    assert model.cache_allocations == 2
    assert len(engine._cache_pool) == 1
    assert list(engine._cache_pool.values()) == [second]
    assert first not in engine._cache_pool.values()


def test_openai_engine_microbatch_cache_pool_replaces_slots_and_caps_entries(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_MICROBATCH_CACHE_POOL_MAX_ENTRIES", "3")
    engine = _cache_only_engine()
    model = _BatchRecordingModel()

    slot_zero_first = engine._generation_microbatch_cache(0, 1, 4, model=model)
    slot_zero_second = engine._generation_microbatch_cache(0, 2, 4, model=model)
    assert slot_zero_first not in engine._microbatch_cache_pool.values()
    assert list(key[0] for key in engine._microbatch_cache_pool) == [0]

    engine._generation_microbatch_cache(1, 1, 4, model=model)
    engine._generation_microbatch_cache(2, 1, 4, model=model)
    engine._generation_microbatch_cache(3, 1, 4, model=model)

    assert len(engine._microbatch_cache_pool) == 3
    assert list(key[0] for key in engine._microbatch_cache_pool) == [1, 2, 3]
    assert slot_zero_second not in engine._microbatch_cache_pool.values()


def test_openai_stream_microbatch_defaults_to_smaller_cuda_tp_chunks(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_DECODE_STEP_MAX_BATCH", raising=False)
    model = object()
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert engine._stream_microbatch_size(64) == 16
    assert engine._stream_microbatch_size(8) == 8


def test_openai_stream_microbatch_env_override_is_capped_to_batch(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE", "128")
    engine = _cache_only_engine()
    engine.model = object()

    assert engine._stream_microbatch_size(64) == 64


def test_openai_effective_max_batch_size_caps_cuda_tp_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_MAX_BATCH_SIZE", raising=False)
    model = object()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 64) == 16
    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 8) == 8
    assert _effective_openai_max_batch_size(model, torch.device("cpu"), 64) == 64


def test_openai_effective_max_batch_size_uses_tp_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_MAX_BATCH_SIZE", "32")
    model = object()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 64) == 32
    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 16) == 16


def test_openai_tensor_parallel_command_sync_uses_control_group(monkeypatch) -> None:
    monkeypatch.setattr("torchinferno.openai_server._TENSOR_PARALLEL_CONTROL_GROUP", None)
    model = type("FakeTPModel", (), {"world_size": 2})()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fake_new_group(*, backend: str) -> str:
        calls.append(("new_group", backend))
        return "control"

    def fake_barrier(**kwargs: object) -> None:
        calls.append(("barrier", kwargs))

    monkeypatch.setattr(dist, "new_group", fake_new_group)
    monkeypatch.setattr(dist, "barrier", fake_barrier)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device.type)))

    _sync_tensor_parallel_command(model, torch.device("cuda"))

    assert calls == [
        ("sync", "cuda"),
        ("new_group", "gloo"),
        ("barrier", {"group": "control"}),
        ("sync", "cuda"),
    ]


def test_openai_microbench_cli_runs_synthetic_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = [
        sys.executable,
        "-m",
        "torchinferno.cli",
        "openai-microbench",
        "--backend",
        "synthetic",
        "--device",
        "cpu",
        "--warmup",
        "0",
        "--iters",
        "1",
        "--prompt-tokens",
        "3",
        "--max-tokens",
        "2",
        "--compare-batcher",
        "--concurrency",
        "2",
    ]

    result = subprocess.run(cmd, cwd=root, env=env, text=True, capture_output=True, check=True, timeout=30)

    assert "TorchInferno OpenAI microbench" in result.stdout
    assert "case=single-direct" in result.stdout
    assert "case=single-batcher" in result.stdout
    assert "case=concurrent-2" in result.stdout


def test_openai_microbench_cases_preserve_tensor_parallel_default() -> None:
    from torchinferno.cli import _openai_microbench_cases

    class _Model:
        world_size = 8

    class _Engine:
        model = _Model()
        single_request_fast_path = False

    cases, skipped_batcher_compare = _openai_microbench_cases(
        _Engine(),
        compare_batcher=False,
        concurrency=64,
    )
    assert cases == [("single", False, 1), ("concurrent-64", False, 64)]
    assert skipped_batcher_compare is False

    compare_cases, skipped_batcher_compare = _openai_microbench_cases(
        _Engine(),
        compare_batcher=True,
        concurrency=64,
    )
    assert compare_cases == [("single-direct", True, 1), ("concurrent-64", False, 64)]
    assert skipped_batcher_compare is True


class _BatchEncodingTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize
        assert add_generation_prompt
        assert messages[0]["role"] == "user"
        return {"input_ids": [[7, 8, 9]], "attention_mask": [[1, 1, 1]]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("apply_chat_template should be used")

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return "".join(str(token_id) for token_id in token_ids)


class _LlamaStyleTokenizer(_BatchEncodingTokenizer):
    eos_token_id = 128001
    vocab_size = 128000
    all_special_tokens = ("<|end_of_text|>", "<|eot_id|>")

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|end_of_text|>":
            return 128001
        if token == "<|eot_id|>":
            return 128009
        return 0


class _StopTokenTokenizer:
    eos_token_id = 0
    stop_token_ids = (0, 9)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        return [1]

    def decode_token(self, token_id: int) -> str:
        return "" if token_id in self.stop_token_ids else str(token_id)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.decode_token(token_id) for token_id in token_ids)


class _BarrierByteFallbackTokenizer(_ByteFallbackTokenizer):
    def __init__(self, *, vocab_size: int, parties: int) -> None:
        super().__init__(vocab_size)
        self.barrier = threading.Barrier(parties)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        self.barrier.wait(timeout=5)
        return super().encode_messages(messages)


class _FirstEncodeEventTokenizer(_ByteFallbackTokenizer):
    def __init__(self, *, vocab_size: int) -> None:
        super().__init__(vocab_size)
        self.first_encoded = threading.Event()
        self._lock = threading.Lock()
        self._calls = 0

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        with self._lock:
            call_index = self._calls
            self._calls += 1
        tokens = super().encode_messages(messages)
        if call_index == 0:
            self.first_encoded.set()
        return tokens


class _BatchRecordingCache:
    def __init__(self) -> None:
        self.seq_len = 0


class _BatchRecordingModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 8})()
        self.calls: list[tuple[int, int, str]] = []
        self.cache_allocations = 0

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _BatchRecordingCache:
        self.cache_allocations += 1
        return _BatchRecordingCache()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _BatchRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        self.calls.append((input_ids.size(0), input_ids.size(1), threading.current_thread().name))
        cache.seq_len += input_ids.size(1)
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        logits = torch.zeros(input_ids.size(0), tokens, 8)
        logits[..., 2] = 1.0
        return logits, cache


class _WorkerLoopRecordingEngine:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.single_calls: list[tuple[list[list[int]], int, float, bool, bool]] = []
        self.batch_calls: list[tuple[list[list[int]], int, float, bool]] = []

    def _generate_single_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool,
        update_prefix_cache: bool = True,
    ):
        self.single_calls.append(
            (
                [[int(token_id) for token_id in row.tolist()] for row in input_ids],
                max_tokens,
                temperature,
                broadcast_tensor_parallel,
                update_prefix_cache,
            )
        )
        yield 2

    def _generate_batch_steps(
        self,
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool,
    ):
        self.batch_calls.append(
            (
                [[int(token_id) for token_id in row.tolist()] for row in input_ids],
                max_tokens,
                temperature,
                broadcast_tensor_parallel,
            )
        )
        yield [2 for _ in range(input_ids.size(0))]


class _ScriptedTokenModel:
    def __init__(self, tokens: list[int], *, vocab_size: int) -> None:
        self.config = type("Config", (), {"vocab_size": vocab_size})()
        self.tokens = tokens
        self.calls: list[tuple[int, int]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _BatchRecordingCache:
        return _BatchRecordingCache()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _BatchRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        del use_cache
        self.calls.append((input_ids.size(0), input_ids.size(1)))
        cache.seq_len += input_ids.size(1)
        token = self.tokens[min(len(self.calls) - 1, len(self.tokens) - 1)]
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        logits = torch.zeros(input_ids.size(0), tokens, self.config.vocab_size)
        logits[..., token] = 1.0
        return logits, cache


class _WarmupShapeCache:
    def __init__(self) -> None:
        self.seq_len = 0


class _WarmupShapeModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 16})()
        self.prefill_shapes: list[tuple[int, int]] = []
        self.decode_shapes: list[tuple[int, int]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _WarmupShapeCache:
        return _WarmupShapeCache()

    def try_prefill_logits_graph(self, input_ids: torch.Tensor, cache: _WarmupShapeCache) -> torch.Tensor:
        self.prefill_shapes.append((input_ids.size(0), input_ids.size(1)))
        return torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)

    def try_decode_one_token_logits_graph(self, input_ids: torch.Tensor, cache: _WarmupShapeCache) -> torch.Tensor:
        self.decode_shapes.append((input_ids.size(0), input_ids.size(1)))
        return torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)


class _PrefixTokenizer:
    eos_token_id = 0
    stop_token_ids: frozenset[int] = frozenset()

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        content = str(messages[-1]["content"])
        if content == "first":
            return [10, 11]
        if content == "second":
            return [10, 11, 2, 12]
        raise AssertionError(f"unexpected message content: {content}")

    def decode_token(self, token_id: int) -> str:
        return "A" if token_id == 2 else ""

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.decode_token(token_id) for token_id in token_ids)


class _SharedPrefixTokenizer:
    eos_token_id = 0
    stop_token_ids: frozenset[int] = frozenset()

    def __init__(self, *, parties: int) -> None:
        self.barrier = threading.Barrier(parties)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        self.barrier.wait(timeout=5)
        content = str(messages[-1]["content"])
        if content == "left":
            return [10, 11, 12]
        if content == "right":
            return [10, 11, 13]
        if content == "long":
            return [10, 11, 13, 14]
        raise AssertionError(f"unexpected message content: {content}")

    def decode_token(self, token_id: int) -> str:
        return "A" if token_id == 2 else ""

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.decode_token(token_id) for token_id in token_ids)


class _PrefixRecordingLayer:
    def __init__(self, batch_size: int, max_seq_len: int) -> None:
        self.keys = torch.zeros(batch_size, 1, max_seq_len, 1)
        self.values = torch.zeros(batch_size, 1, max_seq_len, 1)
        self.seq_len = 0


class _PrefixRecordingCache:
    def __init__(self, batch_size: int, max_seq_len: int) -> None:
        self.layers = [_PrefixRecordingLayer(batch_size, max_seq_len)]

    @property
    def seq_len(self) -> int:
        return self.layers[0].seq_len


class _PrefixRecordingModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 16})()
        self.forward_inputs: list[list[int]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _PrefixRecordingCache:
        return _PrefixRecordingCache(batch_size, max_seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _PrefixRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        del use_cache
        self.forward_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
        layer = cache.layers[0]
        start = layer.seq_len
        end = start + input_ids.size(1)
        layer.keys[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.values[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.seq_len = end
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        logits = torch.zeros(input_ids.size(0), tokens, 16)
        logits[..., 2] = 1.0
        return logits, cache


class _PrefixGraphRecordingModel(_PrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.graph_inputs: list[list[int]] = []

    def try_prefill_graph(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        del temperature
        self.graph_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
        layer = cache.layers[0]
        start = layer.seq_len
        end = start + input_ids.size(1)
        layer.keys[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.values[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.seq_len = end
        return torch.tensor([2], dtype=torch.long)


class _RuntimePrefillGraphCaptureModel(_PrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.capture_flags: list[bool] = []
        self.graph_inputs: list[list[int]] = []

    def try_prefill_graph(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        temperature: float = 0.0,
        capture_on_miss: bool = True,
    ) -> torch.Tensor | None:
        del temperature, cache
        self.capture_flags.append(capture_on_miss)
        if not capture_on_miss:
            return None
        self.graph_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
        return torch.tensor([2], dtype=torch.long)


class _RuntimePrefillLogitsGraphCaptureModel(_PrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.capture_flags: list[bool] = []
        self.graph_inputs: list[list[int]] = []

    def try_prefill_logits_graph(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        capture_on_miss: bool = True,
    ) -> torch.Tensor | None:
        del cache
        self.capture_flags.append(capture_on_miss)
        if not capture_on_miss:
            return None
        self.graph_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
        logits = torch.zeros(input_ids.size(0), 1, self.config.vocab_size)
        logits[..., 2] = 1.0
        return logits


class _DecodeGraphRecordingModel(_PrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.decode_graph_calls = 0

    def try_decode_one_token_graph(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        del input_ids, cache, temperature
        self.decode_graph_calls += 1
        return torch.tensor([3], dtype=torch.long)


class _SharedPrefixRecordingModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"vocab_size": 16})()
        self.forward_inputs: list[list[list[int]]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _PrefixRecordingCache:
        return _PrefixRecordingCache(batch_size, max_seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _PrefixRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        del use_cache
        self.forward_inputs.append([[int(token_id) for token_id in row.tolist()] for row in input_ids])
        layer = cache.layers[0]
        start = layer.seq_len
        end = start + input_ids.size(1)
        layer.keys[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.values[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.seq_len = end
        tokens = 1 if return_last_logits_only else input_ids.size(1)
        logits = torch.zeros(input_ids.size(0), tokens, self.config.vocab_size)
        logits[..., 2] = 1.0
        return logits, cache


class _NoBatchStepEngine(OpenAICompletionEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_step_calls = 0

    def _generate_batch_steps(self, *args, **kwargs):
        self.batch_step_calls += 1
        raise AssertionError("single-request fast path must not use batched step generation")


class _CountingWriter:
    def __init__(self) -> None:
        self.payload = b""
        self.write_calls = 0
        self.flush_calls = 0

    def write(self, payload: bytes) -> int:
        self.write_calls += 1
        self.payload += payload
        return len(payload)

    def flush(self) -> None:
        self.flush_calls += 1


class _FailingWriter:
    def __init__(self, *, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.write_calls = 0
        self.payload = b""

    def write(self, payload: bytes) -> int:
        self.write_calls += 1
        if self.write_calls >= self.fail_on_call:
            raise BrokenPipeError("client disconnected")
        self.payload += payload
        return len(payload)

    def flush(self) -> None:
        return


class _DrainRecordingEngine:
    model_id = "tiny"
    tokenizer = _ByteFallbackTokenizer(vocab_size=16)

    def __init__(self) -> None:
        self.drained_tokens: list[int] = []

    def generate_chat_tokens(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ):
        del messages, temperature
        for token_id in range(1, max_tokens + 1):
            self.drained_tokens.append(token_id)
            yield token_id


class _FakeSocket:
    def __init__(self) -> None:
        self.options: list[tuple[int, int, int]] = []

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))


def _cache_only_engine() -> OpenAICompletionEngine:
    engine = object.__new__(OpenAICompletionEngine)
    engine.cache_backend = "dense"
    engine.page_size = 16
    engine.device = torch.device("cpu")
    engine._cache_pool = {}
    engine._microbatch_cache_pool = {}
    return engine


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_models(port: int, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError(f"server exited early with code {proc.returncode}:\n{output}")
        try:
            _json_get(url)
            return
        except Exception:
            time.sleep(0.25)
    output = proc.stdout.read() if proc.stdout is not None else ""
    raise TimeoutError(f"server did not become ready:\n{output}")


def _json_get(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_post(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_post(url: str, payload: dict[str, object]) -> list[str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        lines = []
        while True:
            line = response.readline()
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            if not decoded:
                continue
            lines.append(decoded)
            if decoded == "data: [DONE]":
                break
        return lines
