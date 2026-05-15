from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest
import torch

from torchinferno.openai_http import (
    OpenAIHandler,
    OpenAIHTTPServer,
    _chat_completion_chunk_bytes,
    _chat_completion_chunk_prefix,
    _chat_delta_content,
    _chunked_stream_enabled,
    _stream_defer_role_enabled,
    _stream_inline_enabled,
    enable_tcp_nodelay,
)
from torchinferno.openai_server import (
    _GenerationDone,
    OpenAICompletionEngine,
    OpenAIServerConfig,
    _ByteFallbackTokenizer,
    _QueuedGeneration,
    _TransformersChatTokenizer,
    _cache_row_slice,
    _copy_generation_cache_first_row,
    _copy_generation_cache_row,
    _decode_next_token_ragged,
    _distributed_server_command,
    _effective_openai_max_batch_size,
    _identical_prompt_cache_pool_enabled,
    _openai_cuda_graph_enabled_for_model,
    _openai_decode_graph_enabled,
    _openai_ragged_decode_graph_enabled,
    _prefill_repeated_prefix_next_token,
    _prefers_exact_generation_cache,
    _runtime_prefill_graph_capture_enabled,
    _repeat_generation_cache_first_batch,
    _sampled_batch_shape_bucket_size,
    _should_reexec_distributed_server,
    _sync_tensor_parallel_command,
    _sync_tensor_parallel_continue,
    _tensor_parallel_worker_loop,
    _tp_command_cuda_sync_for_steps,
    _try_decode_ragged_token_graph,
    _try_decode_ragged_logits_graph,
    _try_decode_one_token_graph,
    _try_decode_one_token_logits_graph,
    _try_prefill_graph,
    _try_prefill_logits_graph,
    _warmup_prefill_cache_token_counts,
    _warmup_prompt_token_counts,
    _warmup_prefix_suffix_cache_token_counts,
    _warmup_prefix_suffix_token_counts,
    _warmup_ragged_decode_batch_sizes,
    _warmup_ragged_decode_cache_token_counts,
    _warmup_ragged_decode_prompt_tokens,
    _warmup_ragged_decode_row_counts,
    _warmup_temperature_batch_sizes,
    _warmup_temperature_prompt_token_counts,
    load_chat_tokenizer,
)
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelCache,
    Llama3TensorParallelLayerKVCache,
)
from torchinferno.server.openai_protocol import chat_completion_chunk


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


def test_openai_handler_fast_chat_chunk_bytes_match_protocol() -> None:
    completion_id = "chatcmpl-test"
    model_id = 'test-model-"quoted"'
    created = 123
    prefix = _chat_completion_chunk_prefix(completion_id, model_id, created)

    payload = _chat_completion_chunk_bytes(prefix, _chat_delta_content('hello "there"\n'), None)
    decoded = json.loads(payload.removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert decoded == chat_completion_chunk(
        completion_id=completion_id,
        model_id=model_id,
        created=created,
        delta={"content": 'hello "there"\n'},
    )

    payload = _chat_completion_chunk_bytes(prefix, b"{}", "stop")
    decoded = json.loads(payload.removeprefix(b"data: ").removesuffix(b"\n\n"))
    assert decoded == chat_completion_chunk(
        completion_id=completion_id,
        model_id=model_id,
        created=created,
        delta={},
        finish_reason="stop",
    )


def test_openai_handler_writes_sse_comment() -> None:
    handler = object.__new__(OpenAIHandler)
    writer = _CountingWriter()
    handler.wfile = writer

    handler._write_sse_comment("heartbeat")

    assert writer.write_calls == 1
    assert writer.payload == b": heartbeat\n\n"
    assert writer.flush_calls == 1


def test_openai_handler_writes_chunked_sse_frame() -> None:
    handler = object.__new__(OpenAIHandler)
    writer = _CountingWriter()
    handler.wfile = writer
    handler._chunked_sse = True
    handler.close_connection = False

    handler._write_sse_comment("heartbeat")
    handler._try_write_done()
    handler._finish_sse_response()

    assert writer.payload == b"d\r\n: heartbeat\n\n\r\ne\r\ndata: [DONE]\n\n\r\n0\r\n\r\n"
    assert handler.close_connection is False


def test_openai_chunked_stream_requires_http11(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_HTTP_CHUNKED_STREAM", raising=False)

    assert _chunked_stream_enabled("HTTP/1.1")
    assert not _chunked_stream_enabled("HTTP/1.0")

    monkeypatch.setenv("TORCHINFERNO_OPENAI_HTTP_CHUNKED_STREAM", "0")
    assert not _chunked_stream_enabled("HTTP/1.1")


def test_openai_http_server_suppresses_disconnect_tracebacks(capsys) -> None:
    server = object.__new__(OpenAIHTTPServer)

    try:
        raise ConnectionResetError("client disconnected")
    except ConnectionResetError:
        server.handle_error(object(), ("127.0.0.1", 12345))

    captured = capsys.readouterr()
    assert "client disconnected" not in captured.err
    assert "Exception occurred during processing" not in captured.err


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
    assert "--standalone" not in command
    assert command[command.index("--rdzv-backend") + 1] == "c10d"
    rdzv_endpoint = command[command.index("--rdzv-endpoint") + 1]
    assert rdzv_endpoint.startswith("127.0.0.1:")
    assert not rdzv_endpoint.endswith(":8000")
    assert command[command.index("--rdzv-id") + 1].startswith("torchinferno-openai-")
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
    assert set(_warmup_temperature_batch_sizes()) >= {1, 8, 15, 16, 64}
    assert set(_warmup_ragged_decode_batch_sizes()) == {56, 64}
    assert set(_warmup_ragged_decode_row_counts()) >= {16, 32, 64}
    assert set(_warmup_ragged_decode_cache_token_counts()) >= {256, 512}
    assert _warmup_ragged_decode_prompt_tokens(64) == 64


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


def test_openai_ragged_decode_warmup_uses_configured_shapes(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_BATCH_SIZES", "4")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_ROW_COUNTS", "4,2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_CACHE_TOKENS", "8")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_PROMPT_TOKENS", "3")

    model = _WarmupShapeModel()
    engine = object.__new__(OpenAICompletionEngine)
    engine.model = model
    engine.device = torch.device("cpu")
    engine.cache_backend = "dense"
    engine.page_size = 16
    engine._cache_pool = {}
    engine._microbatch_cache_pool = {}

    engine._warmup_tensor_parallel_ragged_decode_graphs(vocab_size=16)

    assert model.prefill_shapes == [(4, 3)]
    assert model.ragged_shapes == [
        (4, 1, (3, 3, 3, 3), None),
        (2, 1, (3, 3, 3, 3), (0, 1)),
    ]


def test_openai_server_pipeline_parallelism_skips_auto_launch(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = OpenAIServerConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
        llama_parallelism="pipeline",
    )

    assert not _should_reexec_distributed_server(config)


def test_load_chat_tokenizer_honors_explicit_tokenizer_for_tiny_model(monkeypatch) -> None:
    class FakeAutoTokenizer:
        calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: object) -> object:
            cls.calls.append((name, kwargs))
            return _BatchEncodingTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    config = OpenAIServerConfig(
        model="tiny",
        model_kind="tiny-deepseek",
        tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
        trust_remote_code=True,
        token="hf-token",
        revision="main",
        cache_dir="/tmp/cache",
    )

    tokenizer = load_chat_tokenizer(config, vocab_size=16)

    assert isinstance(tokenizer, _TransformersChatTokenizer)
    assert FakeAutoTokenizer.calls == [
        (
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            {
                "trust_remote_code": True,
                "token": "hf-token",
                "revision": "main",
                "cache_dir": "/tmp/cache",
            },
        )
    ]


def test_load_chat_tokenizer_defaults_tiny_model_to_byte_tokenizer() -> None:
    config = OpenAIServerConfig(model="tiny", model_kind="tiny-deepseek")

    tokenizer = load_chat_tokenizer(config, vocab_size=16)

    assert isinstance(tokenizer, _ByteFallbackTokenizer)


def test_chat_template_batch_encoding_input_ids_are_extracted() -> None:
    tokenizer = _TransformersChatTokenizer(_BatchEncodingTokenizer())

    encoded = tokenizer.encode_messages([{"role": "user", "content": "hello"}])

    assert encoded == [7, 8, 9]


def test_transformers_chat_tokenizer_stops_on_llama_eot() -> None:
    tokenizer = _TransformersChatTokenizer(_LlamaStyleTokenizer())

    assert 128001 in tokenizer.stop_token_ids
    assert 128008 in tokenizer.stop_token_ids
    assert 128009 in tokenizer.stop_token_ids
    assert 128010 in tokenizer.stop_token_ids


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


def test_tensor_parallel_worker_loop_receives_tensor_stream_command(monkeypatch) -> None:
    import torch.distributed as dist

    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([1, 1, 1, 3, 2, 1, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([3], dtype=torch.long),
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        tensor.copy_(payloads.pop(0).to(device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    engine = _WorkerLoopRecordingEngine()
    engine.model = model

    _tensor_parallel_worker_loop(engine)

    assert payloads == []
    assert engine.batch_calls == [([[1, 2, 3]], 2, 0.25, False)]


def test_tensor_parallel_worker_loop_receives_tensor_prompt_list_command(monkeypatch) -> None:
    import torch.distributed as dist

    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([2, 1, 2, 3, 4, 1, 0], dtype=torch.long),
        torch.tensor([0.75], dtype=torch.float64),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        torch.tensor([1, 4], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        tensor.copy_(payloads.pop(0).to(device=tensor.device, dtype=tensor.dtype))

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "barrier", lambda: None)

    engine = _WorkerLoopRecordingEngine()
    engine.model = model

    _tensor_parallel_worker_loop(engine)

    assert payloads == []
    assert engine.prompt_list_calls == [([[1, 2], [3, 4, 5]], 4, 0.75, False, [1, 4])]


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


def test_openai_stream_writes_heartbeat_while_waiting(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_INLINE", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_HEARTBEAT_SECONDS", "0.01")
    engine = _SlowStreamEngine(delay_s=0.03)
    handler = object.__new__(OpenAIHandler)
    handler.server = type("Server", (), {"engine": engine})()
    handler.wfile = _CountingWriter()
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda key, value: None
    handler.end_headers = lambda: None

    handler._stream_chat([{"role": "user", "content": "hi"}], max_tokens=1, temperature=0.0)

    assert b": torchinferno heartbeat\n\n" in handler.wfile.payload
    assert b"data: [DONE]\n\n" in handler.wfile.payload


def test_openai_stream_inline_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STREAM_INLINE", raising=False)

    assert _stream_inline_enabled(max_tokens=256, temperature=0.0)
    assert _stream_inline_enabled(max_tokens=256, temperature=0.7)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_INLINE", "0")
    assert not _stream_inline_enabled(max_tokens=256, temperature=0.7)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_INLINE", "1")
    assert _stream_inline_enabled(max_tokens=1024, temperature=0.0)


def test_openai_stream_defer_role_defaults_to_bounded_streams(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE_MAX_TOKENS", raising=False)

    assert _stream_defer_role_enabled(max_tokens=256, temperature=0.0)
    assert _stream_defer_role_enabled(max_tokens=400, temperature=0.7)
    assert not _stream_defer_role_enabled(max_tokens=512, temperature=0.0)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE_MAX_TOKENS", "512")
    assert _stream_defer_role_enabled(max_tokens=512, temperature=0.0)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE", "0")
    assert not _stream_defer_role_enabled(max_tokens=256, temperature=0.0)


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
    assert model.calls[1][0] == 1
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
    assert model.calls[1][0] == 1
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
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


def test_openai_engine_caches_encoded_chat_prompts(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PROMPT_TOKEN_CACHE_MAX_ENTRIES", "2")
    tokenizer = _CountingPromptTokenizer()
    engine = _cache_only_engine()
    engine.tokenizer = tokenizer
    engine.max_model_len = None

    messages = [{"role": "user", "content": "same"}]
    first = engine._encode_chat_prompt(messages, max_tokens=1)
    first.append(99)
    second = engine._encode_chat_prompt(messages, max_tokens=1)

    assert second == [4, 1]
    assert tokenizer.calls == 1


def test_openai_engine_prompt_token_cache_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PROMPT_TOKEN_CACHE_MAX_ENTRIES", "0")
    tokenizer = _CountingPromptTokenizer()
    engine = _cache_only_engine()
    engine.tokenizer = tokenizer
    engine.max_model_len = None

    messages = [{"role": "user", "content": "same"}]
    assert engine._encode_chat_prompt(messages, max_tokens=1) == [4, 1]
    assert engine._encode_chat_prompt(messages, max_tokens=1) == [4, 2]
    assert tokenizer.calls == 2


def test_openai_engine_keeps_multiple_prefix_cache_entries(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", "4")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    for tokens in ([10, 11], [20, 21]):
        input_ids = torch.tensor([tokens], dtype=torch.long)
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)

    restore_cache = model.allocate_cache(1, 8)
    restored = engine._restore_prefix_cache(
        torch.tensor([[10, 11, 2, 12]], dtype=torch.long),
        restore_cache,
    )

    assert restored == 2
    assert restore_cache.seq_len == 2


def test_openai_engine_prefix_cache_max_entries_evicts_oldest(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", "1")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    for tokens in ([10, 11], [20, 21]):
        input_ids = torch.tensor([tokens], dtype=torch.long)
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)

    assert (
        engine._restore_prefix_cache(
            torch.tensor([[10, 11, 2, 12]], dtype=torch.long),
            model.allocate_cache(1, 8),
        )
        == 0
    )
    assert (
        engine._restore_prefix_cache(
            torch.tensor([[20, 21, 2, 22]], dtype=torch.long),
            model.allocate_cache(1, 8),
        )
        == 2
    )


def test_openai_engine_default_prefix_cache_keeps_multi_turn_working_set(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", raising=False)
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    for index in range(128):
        input_ids = torch.tensor([[10_000 + index, 20_000 + index]], dtype=torch.long)
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)

    assert (
        engine._restore_prefix_cache(
            torch.tensor([[10_000, 20_000, 2, 12]], dtype=torch.long),
            model.allocate_cache(1, 8),
        )
        == 2
    )

    input_ids = torch.tensor([[30_000, 40_000]], dtype=torch.long)
    cache = model.allocate_cache(1, 8)
    model.forward(input_ids, cache=cache, use_cache=True)
    engine._save_prompt_prefix_cache(input_ids, cache)

    assert (
        engine._restore_prefix_cache(
            torch.tensor([[10_001, 20_001, 2, 12]], dtype=torch.long),
            model.allocate_cache(1, 8),
        )
        == 0
    )


def test_openai_engine_restores_older_exact_prefix_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", "4")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    first_ids = torch.tensor([[10, 11]], dtype=torch.long)
    second_ids = torch.tensor([[20, 21]], dtype=torch.long)
    for input_ids in (first_ids, second_ids):
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)

    restore_cache = model.allocate_cache(1, 8)

    assert engine._restore_exact_prefix_cache(first_ids, restore_cache) == 2
    assert restore_cache.seq_len == 2


def test_openai_prompt_prefix_cache_rows_respect_row_token_cap(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", "8")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_ROW_MAX_TOKENS", "2")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    cache = model.allocate_cache(1, 8)
    model.forward(input_ids, cache=cache, use_cache=True)

    engine._save_prompt_prefix_cache_rows([[10, 11, 12]], cache)
    assert engine._exact_prefix_cache_entry((10, 11, 12)) is None

    engine._save_prompt_prefix_cache(input_ids, cache)
    assert engine._exact_prefix_cache_entry((10, 11, 12)) is not None


def test_openai_prompt_prefix_cache_rows_default_to_short_row_cap(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_ROW_MAX_TOKENS", raising=False)
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    cache = model.allocate_cache(1, 32)
    short_prompt = list(range(16))
    long_prompt = list(range(17))
    model.forward(torch.tensor([long_prompt], dtype=torch.long), cache=cache, use_cache=True)

    engine._save_prompt_prefix_cache_rows([short_prompt], cache)
    assert engine._exact_prefix_cache_entry(tuple(short_prompt)) is not None

    engine._save_prompt_prefix_cache_rows([long_prompt], cache)
    assert engine._exact_prefix_cache_entry(tuple(long_prompt)) is None


def test_openai_prompt_list_batch_restores_cached_prefix_rows(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE", "1")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    for prompt in ([10, 11, 12], [10, 11, 13]):
        input_ids = torch.tensor([prompt], dtype=torch.long)
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)
    model.forward_inputs.clear()

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 12, 4], [10, 11, 13, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[4, 5]]
    assert model.forward_inputs == [[[4], [5]]]


def test_openai_prompt_list_single_request_reuses_prefix_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()

    first = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )
    second = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 2, 12]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert first == [[2]]
    assert second == [[2]]
    assert model.forward_inputs == [[10, 11], [2, 12]]


def test_openai_prompt_list_single_request_skips_tiny_suffix_prefix_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()

    list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )
    second = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 12]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert second == [[2]]
    assert model.forward_inputs == [[10, 11], [10, 11, 12]]


def test_openai_prompt_list_identical_prompts_skip_prefix_cache_restore() -> None:
    engine = _cache_only_engine()
    calls: list[tuple[list[int], int, int, float, list[int] | None]] = []

    def prefix_cached_prompt_groups(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("identical prompt batches should use the repeated-prefix path")

    def generate_identical_prompt_batch_steps(
        input_ids: torch.Tensor,
        *,
        batch_size: int,
        max_tokens: int,
        temperature: float,
        row_max_tokens: list[int] | None = None,
    ):
        calls.append((input_ids[0].tolist(), batch_size, max_tokens, temperature, row_max_tokens))
        yield [11, 11, 11]

    engine._prefix_cached_prompt_groups = prefix_cached_prompt_groups  # type: ignore[method-assign]
    engine._generate_identical_prompt_batch_steps = generate_identical_prompt_batch_steps  # type: ignore[method-assign]

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[1, 2], [1, 2], [1, 2]],
            max_tokens=4,
            temperature=0.7,
            broadcast_tensor_parallel=False,
            row_max_tokens=[1, 2, 3],
        )
    )

    assert steps == [[11, 11, 11]]
    assert calls == [([1, 2], 3, 4, 0.7, [1, 2, 3])]


def test_openai_prefill_repeated_prefix_uses_repeated_sampler() -> None:
    class _RepeatedSampleModel:
        def __init__(self) -> None:
            self.sample_calls: list[tuple[tuple[int, ...], int, float]] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            cache: object,
            use_cache: bool,
            return_last_logits_only: bool,
        ) -> tuple[torch.Tensor, object]:
            assert input_ids.tolist() == [[1, 2, 3]]
            assert use_cache
            assert return_last_logits_only
            logits = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
            return logits, cache

        def sample_repeated_next_token(
            self,
            logits: torch.Tensor,
            batch_size: int,
            temperature: float,
        ) -> torch.Tensor:
            self.sample_calls.append((tuple(logits.shape), batch_size, temperature))
            return torch.arange(batch_size, dtype=torch.long)

    model = _RepeatedSampleModel()
    cache = object()

    token, returned_cache = _prefill_repeated_prefix_next_token(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        cache,
        4,
        0.7,
    )

    assert token.tolist() == [0, 1, 2, 3]
    assert returned_cache is cache
    assert model.sample_calls == [((1, 12), 4, 0.7)]


def test_openai_identical_prompt_batch_reuses_exact_prompt_logits_cache() -> None:
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    first = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=1,
            temperature=0.0,
        )
    )
    second = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=1,
            temperature=0.0,
        )
    )

    assert first == [[2, 2, 2]]
    assert second == [[2, 2, 2]]
    assert model.forward_inputs == [[10, 11]]


def test_openai_identical_prompt_logits_cache_defers_kv_restore_until_decode() -> None:
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=1,
            temperature=0.0,
        )
    )
    model.forward_inputs.clear()

    steps = engine._generate_identical_prompt_batch_steps(
        input_ids,
        batch_size=3,
        max_tokens=2,
        temperature=0.0,
    )

    assert next(steps) == [2, 2, 2]
    assert model.forward_inputs == []
    assert next(steps) == [2, 2, 2]
    assert model.forward_inputs == [[2]]


def test_openai_identical_prompt_logits_cache_resamples_temperature_rows(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_SHARED_SAMPLE", "0")

    class _RepeatedSamplerModel(_PrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.sample_calls: list[tuple[tuple[int, ...], int, float]] = []

        def sample_repeated_next_token(
            self,
            logits: torch.Tensor,
            batch_size: int,
            temperature: float,
        ) -> torch.Tensor:
            self.sample_calls.append((tuple(logits.shape), batch_size, temperature))
            start = 3 + len(self.sample_calls)
            return torch.arange(start, start + batch_size, dtype=torch.long)

    model = _RepeatedSamplerModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    first = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=1,
            temperature=0.7,
        )
    )
    second = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=1,
            temperature=0.7,
        )
    )

    assert first == [[4, 5, 6]]
    assert second == [[5, 6, 7]]
    assert model.forward_inputs == [[10, 11]]
    assert model.sample_calls == [((1, 16), 3, 0.7), ((1, 16), 3, 0.7)]


def test_openai_identical_prompt_batch_decodes_uniform_rows_once() -> None:
    class _ShapeRecordingPrefixModel(_PrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.forward_shapes: list[tuple[int, int]] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            cache: _PrefixRecordingCache,
            use_cache: bool,
            return_last_logits_only: bool = False,
        ):
            self.forward_shapes.append((input_ids.size(0), input_ids.size(1)))
            return super().forward(
                input_ids,
                cache=cache,
                use_cache=use_cache,
                return_last_logits_only=return_last_logits_only,
            )

    model = _ShapeRecordingPrefixModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    steps = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=2,
            temperature=0.0,
        )
    )

    assert steps == [[2, 2, 2], [2, 2, 2]]
    assert model.forward_shapes == [(1, 2), (1, 1)]


def test_openai_identical_prompt_batch_reuses_uniform_decode_logits_cache() -> None:
    model = _PrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    first = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=2,
            temperature=0.0,
        )
    )
    second = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=2,
            temperature=0.0,
        )
    )

    assert first == [[2, 2, 2], [2, 2, 2]]
    assert second == [[2, 2, 2], [2, 2, 2]]
    assert model.forward_inputs == [[10, 11], [2]]


def test_openai_prompt_list_batch_restores_cached_prefix_rows_with_padded_suffixes(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE", "1")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    for prompt in ([10, 12], [11, 13]):
        input_ids = torch.tensor([prompt], dtype=torch.long)
        cache = model.allocate_cache(1, 8)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)
    model.forward_inputs.clear()

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 12, 4, 6], [11, 13, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert model.forward_inputs == [[[4, 6], [5, 0]]]


def test_openai_prompt_list_batch_restores_cached_prefix_rows_with_suffix_buckets(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE", "1")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    for prompt in ([10, 11], [10, 12], [10, 13], [10, 14]):
        input_ids = torch.tensor([prompt], dtype=torch.long)
        cache = model.allocate_cache(1, 16)
        model.forward(input_ids, cache=cache, use_cache=True)
        engine._save_prompt_prefix_cache(input_ids, cache)
    model.forward_inputs.clear()

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [
                [10, 11, 4, 4, 4, 4, 4, 6],
                [10, 12, 5, 5, 5, 5, 7],
                [10, 13, 8],
                [10, 14, 9],
            ],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 7, 8, 9], [3, 3, 3, 3]]
    assert model.forward_inputs == [
        [[4, 4, 4, 4, 4, 6], [5, 5, 5, 5, 7, 0]],
        [[8], [9]],
    ]
    assert model.ragged_calls == [
        (
            [[6], [7], [8], [9]],
            [8, 7, 3, 3],
            None,
        )
    ]


def test_openai_engine_uses_prefill_graph_for_prefix_suffix(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_SUFFIX_TOKENS", "1")
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


def test_openai_tp_single_prefill_capture_defaults_off(monkeypatch) -> None:
    model = _RuntimePrefillGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _ByteFallbackTokenizer(vocab_size=16)
    engine.stop_token_ids = frozenset()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SINGLE_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    first = list(engine._generate_single_tokens(input_ids, max_tokens=1, temperature=0.0))
    second = list(engine._generate_single_tokens(input_ids, max_tokens=1, temperature=0.0))

    assert first == [2]
    assert second == [2]
    assert model.capture_flags == [False, False]
    assert model.graph_inputs == []
    assert model.forward_inputs == [[1, 2], [1, 2]]


def test_openai_tp_single_prefill_capture_waits_for_repeated_shape(monkeypatch) -> None:
    model = _RuntimePrefillGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _ByteFallbackTokenizer(vocab_size=16)
    engine.stop_token_ids = frozenset()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SINGLE_RUNTIME_PREFILL_CAPTURE", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    first = list(engine._generate_single_tokens(input_ids, max_tokens=1, temperature=0.0))
    second = list(engine._generate_single_tokens(input_ids, max_tokens=1, temperature=0.0))

    assert first == [2]
    assert second == [2]
    assert model.capture_flags == [False, True]
    assert model.graph_inputs == [[1, 2]]
    assert model.forward_inputs == [[1, 2]]


def test_openai_tp_single_prefill_capture_explicit_env_captures_immediately(monkeypatch) -> None:
    model = _RuntimePrefillGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _ByteFallbackTokenizer(vocab_size=16)
    engine.stop_token_ids = frozenset()

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    tokens = list(
        engine._generate_single_tokens(
            torch.tensor([[1, 2]], dtype=torch.long),
            max_tokens=1,
            temperature=0.0,
        )
    )

    assert tokens == [2]
    assert model.capture_flags == [True]
    assert model.graph_inputs == [[1, 2]]
    assert model.forward_inputs == []


def test_openai_tp_runtime_prefill_capture_defaults_off_but_env_can_enable(monkeypatch) -> None:
    model = object()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "1")
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)


def test_openai_tp_runtime_prefill_capture_overrides_still_apply(monkeypatch) -> None:
    model = object()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE", "0")
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", "128")
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=128)


def test_openai_engine_uses_runtime_shared_prefix_capture_for_tensor_parallel(monkeypatch) -> None:
    model = _RuntimePrefillLogitsGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "1")
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
    assert model.capture_flags == [True, True, True]
    assert model.graph_inputs == [[10, 11], [13, 14], [12]]
    assert model.forward_inputs == []


def test_openai_engine_can_disable_runtime_shared_prefix_capture_for_tensor_parallel(monkeypatch) -> None:
    model = _RuntimePrefillLogitsGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "0")
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
        single_request_admission_wait_ms=50.0,
    )
    engine.single_request_fast_path = False
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
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MAX_ENTRIES", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _SharedPrefixRecordingModel()
    tokenizer = _SharedPrefixTokenizer(parties=1)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
        single_request_admission_wait_ms=50.0,
    )
    engine.single_request_fast_path = False

    prompts = [[10, 11, 12], [10, 11, 13, 14]]

    steps = list(
        engine._generate_prompt_list_batch_steps(
            prompts,
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )
    assert steps == [[2, 2]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[13, 14]],
        [[12]],
    ]

    model.forward_inputs.clear()
    steps = list(
        engine._generate_prompt_list_batch_steps(
            prompts,
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )
    engine.close()

    assert steps == [[2, 2]]
    assert model.forward_inputs == [
        [[13, 14]],
        [[12]],
    ]


def test_openai_tensor_parallel_refills_prefix_when_any_rank_misses_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _SharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    prefix_ids = torch.tensor([[10, 11]], dtype=torch.long)
    cache = model.allocate_cache(1, 4)
    model.forward(prefix_ids, cache=cache, use_cache=True)
    engine._save_prompt_prefix_cache(prefix_ids, cache)
    model.forward_inputs.clear()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    import torch.distributed as dist

    calls: list[int] = []
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def all_reduce(flag: torch.Tensor, op: object) -> None:
        assert op == dist.ReduceOp.MIN
        calls.append(int(flag.item()))
        flag.zero_()

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    steps = list(
        engine._generate_shared_prefix_prompt_list_steps(
            [[10, 11, 12], [10, 11, 13, 14]],
            prefix_tokens=2,
            max_tokens=1,
            temperature=0.0,
        )
    )

    assert steps == [[2, 2]]
    assert calls == [1, 0]
    assert model.forward_inputs[0] == [[10, 11]]


def test_openai_engine_ragged_decodes_variable_length_shared_prefix_suffixes(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    model = _RaggedSharedPrefixRecordingModel()
    tokenizer = _SharedPrefixTokenizer(parties=2)
    engine = OpenAICompletionEngine(
        model,
        tokenizer,
        model_id="tiny",
        device=torch.device("cpu"),
        max_batch_size=4,
        batch_wait_ms=50.0,
        single_request_admission_wait_ms=50.0,
    )
    barrier = threading.Barrier(3)
    results: list[list[int] | None] = [None, None]

    def run(index: int, content: str) -> None:
        barrier.wait()
        results[index] = list(
            engine.generate_chat_tokens(
                [{"role": "user", "content": content}],
                max_tokens=3,
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
    engine.close()

    assert results == [[2, 3, 4], [2, 3, 4]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[13, 14]],
        [[12]],
    ]
    assert len(model.ragged_calls) == 2
    first_input, first_lengths, first_rows = model.ragged_calls[0]
    second_input, second_lengths, second_rows = model.ragged_calls[1]
    assert first_input == [[2], [2]]
    assert sorted(first_lengths) == [3, 4]
    assert first_rows is None
    assert second_input == [[3], [3]]
    assert sorted(second_lengths) == [4, 5]
    assert second_rows is None


def test_openai_shared_prefix_uses_dense_decode_for_low_variance_length_groups(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_DENSE_GROUP_DECODE", "1")
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None
    prompts = [[10, 11, 20, 21] for _ in range(8)] + [[10, 11, 30] for _ in range(8)]

    steps = list(
        engine._generate_prompt_list_batch_steps(
            prompts,
            max_tokens=3,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[2 for _ in prompts] for _ in range(3)]
    assert model.ragged_calls == []
    assert len(model.forward_inputs) == 7


def test_openai_shared_prefix_padded_suffix_prefill_uses_true_lengths(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None
    prompts = [[10, 11, 4, 6], [10, 11, 5]]

    steps = list(
        engine._generate_prompt_list_batch_steps(
            prompts,
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 6], [5, 0]],
    ]
    assert model.ragged_calls == []


def test_openai_shared_prefix_padded_suffix_branch_requires_all_tp_ranks(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    calls: list[bool] = []

    def all_ranks_true(candidate: object, value: bool, device: torch.device) -> bool:
        assert candidate is model
        calls.append(value)
        if len(calls) == 2:
            return False
        return value

    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_all_ranks_true", all_ranks_true)

    def fail_padded_prefill(*args, **kwargs):
        raise AssertionError("padded suffix prefill must not run unless every TP rank selects it")

    monkeypatch.setattr(engine, "_prefill_shared_prefix_prompt_list_padded_suffixes", fail_padded_prefill)

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert calls == [False, True]
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 6]],
        [[5]],
    ]


def test_openai_shared_prefix_padded_suffix_prefill_skips_high_padding_waste(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    def fail_padded_prefill(*args, **kwargs):
        raise AssertionError("padded suffix prefill must not run when padding waste is high")

    monkeypatch.setattr(engine, "_prefill_shared_prefix_prompt_list_padded_suffixes", fail_padded_prefill)

    long_suffix = [4 for _ in range(50)]
    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, *long_suffix], [10, 11, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[4, 5]]
    assert model.forward_inputs == [
        [[10, 11]],
        [long_suffix],
        [[5]],
    ]


def test_openai_shared_prefix_padded_suffix_prefill_buckets_length_groups(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None
    prompts = [
        [10, 11, 4, 4, 4, 4, 4, 6],
        [10, 11, 5, 5, 5, 5, 7],
        [10, 11, 8],
        [10, 11, 9],
    ]

    steps = list(
        engine._generate_prompt_list_batch_steps(
            prompts,
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 7, 8, 9], [3, 3, 3, 3]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 4, 4, 4, 4, 6], [5, 5, 5, 5, 7, 0]],
        [[8], [9]],
    ]
    assert model.ragged_calls == [
        (
            [[6], [7], [8], [9]],
            [8, 7, 3, 3],
            None,
        )
    ]


def test_openai_shared_prefix_ragged_cache_can_be_disabled_for_tp(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE", "0")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    def fail_ragged_cache(*args, **kwargs):
        raise AssertionError("TP shared-prefix ragged cache must honor the disable env")

    monkeypatch.setattr(engine, "_shared_prefix_prompt_list_ragged_cache", fail_ragged_cache)

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5], [6, 5]]
    assert model.ragged_calls == []
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 6]],
        [[5]],
        [[6]],
        [[5]],
    ]


def test_openai_generation_cache_can_skip_pooling() -> None:
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()

    cache = engine._generation_cache(2, 8, model=model, pool=False)

    assert cache.seq_len == 0
    assert getattr(cache, "_torchinferno_ephemeral_cache") is True
    assert engine._cache_pool == {}


def test_openai_cache_pool_eviction_releases_graphs(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", "1")

    class _ReleaseRecordingModel:
        def __init__(self) -> None:
            self.allocated: list[_WarmupShapeCache] = []
            self.released: list[object] = []

        def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _WarmupShapeCache:
            del batch_size, max_seq_len, kwargs
            cache = _WarmupShapeCache()
            self.allocated.append(cache)
            return cache

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _ReleaseRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    first = engine._generation_cache(1, 8, model=model)
    second = engine._generation_cache(2, 8, model=model)

    assert model.allocated == [first, second]
    assert model.released == [first]
    assert list(engine._cache_pool.values()) == [second]


def test_llama_tp_cache_release_clears_prefill_and_decode_graphs() -> None:
    from torchinferno.models.llama3.tensor_parallel import Llama3TensorParallelForCausalLM

    model = object.__new__(Llama3TensorParallelForCausalLM)
    cache = object()
    other_cache = object()
    model._prefill_graphs = {
        (id(cache), 0, 1, (1, 1)): types.SimpleNamespace(cache=cache),
        (id(other_cache), 0, 1, (1, 1)): types.SimpleNamespace(cache=other_cache),
    }
    model._prefill_logits_graphs = {
        (id(cache), 0, 1, (1, 1)): types.SimpleNamespace(cache=cache),
    }
    model._decode_graphs = {
        (id(cache), 1, 8): types.SimpleNamespace(cache=cache),
    }
    model._decode_logits_graphs = {
        (id(cache), 1, 8): types.SimpleNamespace(cache=cache),
    }
    model._ragged_decode_logits_graphs = {
        (id(cache), 1, 8, False): types.SimpleNamespace(cache=cache),
    }

    model.release_decode_graphs_for_cache(cache)

    assert list(model._prefill_graphs.values()) == [types.SimpleNamespace(cache=other_cache)]
    assert model._prefill_logits_graphs == {}
    assert model._decode_graphs == {}
    assert model._decode_logits_graphs == {}
    assert model._ragged_decode_logits_graphs == {}


def test_llama_tp_decode_graph_capture_uses_rank_sync(monkeypatch) -> None:
    from torchinferno.models.llama3 import tensor_parallel as tp

    class _Cache:
        def __init__(self) -> None:
            self.seq_len = 0
            self.layers = [types.SimpleNamespace(max_seq_len=8)]

        def set_seq_len(self, seq_len: int) -> None:
            self.seq_len = seq_len

    model = object.__new__(tp.Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model._decode_graphs = {}
    model._decode_logits_graphs = {}
    model._ragged_decode_logits_graphs = {}
    cache = _Cache()
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    logits = torch.zeros((1, 1, 4))
    sync_calls: list[bool] = []
    capture_calls: list[str] = []

    def capture_needed_on_any_rank(needs_capture: bool, device: torch.device) -> bool:
        assert device == torch.device("cpu")
        sync_calls.append(needs_capture)
        return True

    monkeypatch.setattr(tp, "_capture_needed_on_any_rank", capture_needed_on_any_rank)

    def capture_decode(self, captured_input_ids, captured_cache, attention_block_size):  # noqa: ANN001
        del captured_input_ids, captured_cache, attention_block_size
        capture_calls.append("decode")
        return types.SimpleNamespace(output_token=torch.tensor([3]))

    def capture_decode_logits(self, captured_input_ids, captured_cache, attention_block_size):  # noqa: ANN001
        del self, captured_input_ids, captured_cache, attention_block_size
        capture_calls.append("decode_logits")
        return types.SimpleNamespace(output_logits=logits)

    def capture_ragged(self, captured_input_ids, captured_cache, seq_lens, row_indices):  # noqa: ANN001
        del self, captured_input_ids, captured_cache, seq_lens, row_indices
        capture_calls.append("ragged")
        return types.SimpleNamespace(output_logits=logits)

    model._capture_decode_step_graph = types.MethodType(capture_decode, model)
    model._capture_decode_step_logits_graph = types.MethodType(capture_decode_logits, model)
    model._capture_ragged_decode_logits_graph = types.MethodType(capture_ragged, model)
    existing_graph = types.SimpleNamespace(replay=lambda: capture_calls.append("replay"))
    attention_block_size = tp._decode_attention_block_size(cache.seq_len + 1, cache.layers[0].max_seq_len)
    model._decode_graphs[(id(cache), 1, attention_block_size, 0)] = types.SimpleNamespace(
        cache=cache,
        max_seq_len=cache.layers[0].max_seq_len,
        attention_block_size=attention_block_size,
        static_input_ids=input_ids.clone(),
        graph=existing_graph,
        output_token=torch.tensor([1]),
    )
    model._decode_logits_graphs[(id(cache), 1, attention_block_size, 0)] = types.SimpleNamespace(
        cache=cache,
        max_seq_len=cache.layers[0].max_seq_len,
        attention_block_size=attention_block_size,
        static_input_ids=input_ids.clone(),
        graph=existing_graph,
        output_logits=logits,
    )
    model._ragged_decode_logits_graphs[(id(cache), 1, cache.layers[0].max_seq_len, False, 0)] = types.SimpleNamespace(
        cache=cache,
        max_seq_len=cache.layers[0].max_seq_len,
        static_input_ids=input_ids.clone(),
        static_row_indices=None,
        graph=existing_graph,
        output_logits=logits,
    )

    assert model._run_decode_step_graph(input_ids, cache).tolist() == [3]
    cache.set_seq_len(0)
    assert model._run_decode_step_logits_graph(input_ids, cache) is logits
    assert model._run_ragged_decode_logits_graph(
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
    ) is logits

    assert sync_calls == [False, False, False]
    assert capture_calls == ["decode", "decode_logits", "ragged"]


def test_openai_ephemeral_cache_skips_ragged_decode_graph() -> None:
    class _GraphModel:
        calls = 0

        def try_decode_ragged_token_graph(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None,
            temperature: float,
        ) -> torch.Tensor:
            del input_ids, cache, seq_lens, row_indices, temperature
            self.calls += 1
            return torch.zeros(1, dtype=torch.long)

        def try_decode_ragged_logits_graph(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None,
        ) -> torch.Tensor:
            del input_ids, cache, seq_lens, row_indices
            self.calls += 1
            return torch.zeros(1, 1, 4)

    model = _GraphModel()
    cache = types.SimpleNamespace(_torchinferno_ephemeral_cache=True)

    token = _try_decode_ragged_token_graph(
        model,
        torch.tensor([[1]], dtype=torch.long),
        cache,
        seq_lens=torch.tensor([1], dtype=torch.long),
        row_indices=None,
        temperature=0.0,
    )
    logits = _try_decode_ragged_logits_graph(
        model,
        torch.tensor([[1]], dtype=torch.long),
        cache,
        seq_lens=torch.tensor([1], dtype=torch.long),
        row_indices=None,
    )

    assert token is None
    assert logits is None
    assert model.calls == 0


def test_openai_decode_next_token_ragged_prefers_token_graph(monkeypatch) -> None:
    class _GraphModel:
        world_size = 2
        token_graph_calls = 0
        logits_graph_calls = 0

        def try_decode_ragged_token_graph(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None,
            temperature: float,
        ) -> torch.Tensor:
            assert input_ids.tolist() == [[1]]
            assert seq_lens.tolist() == [1]
            assert row_indices is None
            assert temperature == 0.0
            self.token_graph_calls += 1
            return torch.tensor([7], dtype=torch.long)

        def try_decode_ragged_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.logits_graph_calls += 1
            raise AssertionError("ragged logits graph should not run after token graph succeeds")

        def decode_ragged_logits(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("eager ragged decode should not run after token graph succeeds")

        def _sample_next_token(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("sampling should be covered by the token graph")

    model = _GraphModel()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    cache = object()

    token, returned_cache = _decode_next_token_ragged(
        model,
        torch.tensor([[1]], dtype=torch.long),
        cache,
        torch.tensor([1], dtype=torch.long),
        None,
        0.0,
    )

    assert token.tolist() == [7]
    assert returned_cache is cache
    assert model.token_graph_calls == 1
    assert model.logits_graph_calls == 0


def test_openai_tp_shared_prefix_ragged_cache_can_be_ephemeral(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE_POOL", "0")

    class _EphemeralRecordingModel(_TokenEchoSharedPrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.ragged_cache_ephemeral: list[bool] = []
            self.ragged_graph_disabled: list[bool] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: _PrefixRecordingCache,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.ragged_cache_ephemeral.append(bool(getattr(cache, "_torchinferno_ephemeral_cache", False)))
            self.ragged_graph_disabled.append(bool(getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)))
            return super().decode_ragged_logits(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
            )

    model = _EphemeralRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5], [3, 3]]
    assert model.ragged_cache_ephemeral == [True]
    assert model.ragged_graph_disabled == [False]
    assert model.ragged_calls[0][2] == [0, 1]
    assert engine._cache_pool == {}


def test_openai_tp_shared_prefix_ragged_cache_uses_pool_by_default(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE", "1")
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE_POOL", raising=False)

    class _PoolingRecordingModel(_TokenEchoSharedPrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.ragged_cache_ephemeral: list[bool] = []
            self.ragged_graph_disabled: list[bool] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: _PrefixRecordingCache,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.ragged_cache_ephemeral.append(bool(getattr(cache, "_torchinferno_ephemeral_cache", False)))
            self.ragged_graph_disabled.append(bool(getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)))
            return super().decode_ragged_logits(
                input_ids,
                cache,
                seq_lens=seq_lens,
                row_indices=row_indices,
            )

    model = _PoolingRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5], [3, 3]]
    assert model.ragged_cache_ephemeral == [False]
    assert model.ragged_graph_disabled == [False]
    assert model.ragged_calls[0][2] == [0, 1]
    assert len(engine._cache_pool) == 1


def test_openai_tp_shared_prefix_ragged_graph_disabled_for_large_max_tokens(monkeypatch) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.ragged_graph_disabled: list[bool] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del seq_lens, row_indices
            self.ragged_graph_disabled.append(bool(getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)))
            logits = torch.zeros(input_ids.size(0), 1, 8)
            logits[..., 3] = 1.0
            return logits

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=types.SimpleNamespace(),
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=129,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [True]


def test_openai_tp_shared_prefix_ragged_graph_allowed_for_large_max_tokens(monkeypatch) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.ragged_graph_disabled: list[bool] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del seq_lens, row_indices
            self.ragged_graph_disabled.append(bool(getattr(cache, "_torchinferno_disable_ragged_decode_graph", False)))
            logits = torch.zeros(input_ids.size(0), 1, 8)
            logits[..., 3] = 1.0
            return logits

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_LARGE_MIN_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=types.SimpleNamespace(),
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=512,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [False]


def test_openai_ephemeral_cache_scoped_ragged_decode_releases_graphs(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_EPHEMERAL_RAGGED_CUDAGRAPH_MIN_STEP", raising=False)

    class _GraphModel:
        def __init__(self) -> None:
            self.calls = 0
            self.released: list[object] = []

        def try_decode_ragged_logits_graph(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None,
        ) -> torch.Tensor:
            del seq_lens, row_indices
            assert getattr(cache, "_torchinferno_ephemeral_ragged_graph_scope") is True
            self.calls += 1
            logits = torch.zeros(input_ids.size(0), 1, 8)
            logits[..., 3] = 1.0
            return logits

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _GraphModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = types.SimpleNamespace(_torchinferno_ephemeral_cache=True)

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=2,
            next_tokens=[1, 1],
            temperature=0.0,
        )
    )

    assert steps == [[3, 3]]
    assert model.calls == 1
    assert model.released == [cache]
    assert getattr(cache, "_torchinferno_ephemeral_ragged_graph_scope") is False


def test_openai_shared_prefix_ragged_cache_requires_all_tp_ranks(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CACHE", "1")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    calls: list[bool] = []

    def all_ranks_true(candidate: object, value: bool, device: torch.device) -> bool:
        assert candidate is model
        calls.append(value)
        if len(calls) == 3:
            return False
        return value

    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_all_ranks_true", all_ranks_true)

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5], [6, 5]]
    assert calls == [False, False, True]
    assert model.ragged_calls == []
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 6]],
        [[5]],
        [[6]],
        [[5]],
    ]


def test_openai_ragged_decode_skips_rows_after_per_request_max_tokens(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_PREFILL", "0")
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 12], [10, 11, 13, 14]],
            max_tokens=3,
            temperature=0.0,
            broadcast_tensor_parallel=False,
            row_max_tokens=[1, 3],
        )
    )

    assert steps == [[2, 2], [None, 3], [None, 4]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[13, 14]],
        [[12]],
    ]
    assert len(model.ragged_calls) == 2
    assert model.ragged_calls[0] == ([[2]], [3, 4], [1])
    assert model.ragged_calls[1] == ([[3]], [3, 5], [1])


def test_openai_ragged_decode_skips_inactive_rows_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION", raising=False)
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(8, 5)

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[False, True, True, True, True, True, True, True],
            prompt_lengths=[2 for _ in range(8)],
            max_tokens=3,
            next_tokens=[2 for _ in range(8)],
            temperature=0.0,
            row_max_tokens=[1, 3, 3, 3, 3, 3, 3, 3],
        )
    )

    assert steps == [[None, 3, 3, 3, 3, 3, 3, 3], [None, 4, 4, 4, 4, 4, 4, 4]]
    assert model.ragged_calls[0] == ([[2] for _ in range(7)], [2 for _ in range(8)], [1, 2, 3, 4, 5, 6, 7])
    assert model.ragged_calls[1] == (
        [[3] for _ in range(7)],
        [2, 3, 3, 3, 3, 3, 3, 3],
        [1, 2, 3, 4, 5, 6, 7],
    )


def test_openai_ragged_decode_can_keep_full_batch_while_most_rows_active(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION", "0.5")
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(8, 5)

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[False, True, True, True, True, True, True, True],
            prompt_lengths=[2 for _ in range(8)],
            max_tokens=3,
            next_tokens=[2 for _ in range(8)],
            temperature=0.0,
            row_max_tokens=[1, 3, 3, 3, 3, 3, 3, 3],
        )
    )

    assert steps == [[None, 3, 3, 3, 3, 3, 3, 3], [None, 4, 4, 4, 4, 4, 4, 4]]
    assert model.ragged_calls[0] == ([[2] for _ in range(8)], [2 for _ in range(8)], None)
    assert model.ragged_calls[1] == ([[3] for _ in range(8)], [3 for _ in range(8)], None)


def test_openai_ragged_decode_pads_to_power_of_two_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", "1")
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(8, 5)

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True, True, False, False, False, False, False],
            prompt_lengths=[2 for _ in range(8)],
            max_tokens=3,
            next_tokens=[2 for _ in range(8)],
            temperature=0.0,
            row_max_tokens=[3, 3, 3, 1, 1, 1, 1, 1],
        )
    )

    assert steps == [[3, 3, 3, None, None, None, None, None], [4, 4, 4, None, None, None, None, None]]
    assert model.ragged_calls[0] == ([[2], [2], [2], [2]], [2 for _ in range(8)], [0, 1, 2, 3])
    assert model.ragged_calls[1] == ([[3], [3], [3], [3]], [3, 3, 3, 2, 2, 2, 2, 2], [0, 1, 2, 3])


def test_openai_batch_steps_ragged_decode_skips_finished_rows() -> None:
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine.tokenizer = _ByteFallbackTokenizer(vocab_size=16)
    input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    steps = list(
        engine._generate_batch_steps(
            input_ids,
            max_tokens=3,
            temperature=0.0,
            broadcast_tensor_parallel=False,
            row_max_tokens=[1, 3],
        )
    )

    assert steps == [[2, 2], [None, 3], [None, 4]]
    assert model.forward_inputs == [[[1, 2], [3, 4]]]
    assert len(model.ragged_calls) == 2
    assert model.ragged_calls[0] == ([[2]], [2, 2], [1])
    assert model.ragged_calls[1] == ([[3]], [2, 3], [1])


def test_openai_engine_can_disable_prefix_cache_for_tensor_parallel(monkeypatch) -> None:
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

    engine._save_prompt_prefix_cache(input_ids, cache)
    assert engine._prefix_cache_entry is not None
    assert engine._restore_exact_prefix_cache(input_ids, cache) == 2

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_PREFIX_CACHE", "0")

    engine._prefix_cache_entry = object()
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


def test_openai_engine_can_disable_shared_prefix_batching_for_tensor_parallel(monkeypatch) -> None:
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

    assert engine._shared_prefix_batch_tokens(input_ids) == 2
    assert engine._shared_prefix_prompt_list_tokens(prompts) == 2

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_BATCH", "0")

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


def test_openai_stream_microbatch_defaults_to_decode_graph_batch_limit(monkeypatch) -> None:
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

    assert engine._stream_microbatch_size(128) == 64
    assert engine._stream_microbatch_size(64) == 64
    assert engine._stream_microbatch_size(8) == 8


def test_openai_stream_microbatch_env_override_is_capped_to_batch(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_STREAM_MICROBATCH_SIZE", "128")
    engine = _cache_only_engine()
    engine.model = object()

    assert engine._stream_microbatch_size(64) == 64


def test_openai_stream_microbatches_emit_full_batch_steps() -> None:
    model = _BatchRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    input_ids = torch.tensor(
        [
            [10, 11],
            [10, 12],
            [10, 13],
            [10, 14],
        ],
        dtype=torch.long,
    )

    steps = list(
        engine._generate_batch_steps_microbatched(
            input_ids,
            max_tokens=2,
            temperature=0.0,
            microbatch_size=2,
        )
    )

    assert steps == [[2, 2, 2, 2], [2, 2, 2, 2]]


def test_openai_shared_prefix_microbatches_emit_full_batch_steps() -> None:
    model = _BatchRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    active = [True, True, True, True]

    steps = list(
        engine._decode_shared_prefix_microbatches(
            cache_views=[_BatchRecordingCache(), _BatchRecordingCache()],
            active=active,
            batch_size=4,
            microbatch_size=2,
            max_tokens=3,
            next_token=torch.tensor([2, 2, 2, 2], dtype=torch.long),
            temperature=0.0,
        )
    )

    assert steps == [[2, 2, 2, 2], [2, 2, 2, 2]]
    assert active == [False, False, False, False]


def test_openai_cache_row_slice_uses_physical_row_views_for_llama_tp_cache() -> None:
    cache = _llama_tp_cache(batch_size=4, max_seq_len=8)
    for row in range(4):
        keys = torch.full((1, 1, 2, 2), float(row))
        values = keys + 100
        cache.for_rows((row,)).layers[0].append(keys, values)

    view = _cache_row_slice(cache, 2, 4)
    assert view is not None
    view.layers[0].append(
        torch.tensor([[[[20.0, 21.0]], [[30.0, 31.0]]]]).reshape(2, 1, 1, 2),
        torch.tensor([[[[120.0, 121.0]], [[130.0, 131.0]]]]).reshape(2, 1, 1, 2),
    )

    assert cache.for_rows((0, 1)).seq_len == 2
    assert cache.for_rows((2, 3)).seq_len == 3
    torch.testing.assert_close(cache.layers[0].keys[2, :, 2:3, :], torch.tensor([[[20.0, 21.0]]]))
    torch.testing.assert_close(cache.layers[0].keys[3, :, 2:3, :], torch.tensor([[[30.0, 31.0]]]))


def test_openai_cache_copy_helpers_preserve_llama_tp_row_lengths() -> None:
    source = _llama_tp_cache(batch_size=1, max_seq_len=8)
    target = _llama_tp_cache(batch_size=3, max_seq_len=8)
    source_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    source_values = source_keys + 100
    source.for_rows((0,)).layers[0].append(source_keys, source_values)
    target.for_rows((0,)).layers[0].append(torch.ones((1, 1, 1, 2)), torch.ones((1, 1, 1, 2)))

    _copy_generation_cache_row(source, target, source_row=0, target_row=2, seq_len=3)

    assert target.for_rows((0,)).seq_len == 1
    assert target.for_rows((1,)).seq_len == 0
    assert target.for_rows((2,)).seq_len == 3
    torch.testing.assert_close(target.layers[0].keys[2:3, :, :3, :], source_keys)

    clone = _llama_tp_cache(batch_size=2, max_seq_len=8)
    _copy_generation_cache_first_row(source, clone, batch_size=2)

    assert clone.for_rows((0, 1)).seq_len == 3
    torch.testing.assert_close(clone.layers[0].values[:2, :, :3, :], source_values.expand(2, -1, -1, -1))


def test_openai_repeat_generation_cache_first_batch_preserves_llama_tp_row_lengths() -> None:
    cache = _llama_tp_cache(batch_size=3, max_seq_len=8)
    keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    values = keys + 100
    cache.for_rows((0,)).layers[0].append(keys, values)

    _repeat_generation_cache_first_batch(cache, 3)

    assert cache.for_rows((0, 1, 2)).seq_len == 2
    torch.testing.assert_close(cache.layers[0].keys[:3, :, :2, :], keys.expand(3, -1, -1, -1))


def test_openai_prompt_prefix_cache_handles_ragged_llama_tp_cache() -> None:
    engine = _cache_only_engine()
    engine.model = object()
    cache = _llama_tp_cache(batch_size=2, max_seq_len=8)
    prompt_keys = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2)
    prompt_values = prompt_keys + 100
    cache.layers[0].append(prompt_keys, prompt_values)
    cache.for_rows((0,)).layers[0].append(
        torch.ones((1, 1, 1, 2)),
        torch.ones((1, 1, 1, 2)),
    )

    with pytest.raises(ValueError):
        _ = cache.seq_len

    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    engine._save_prompt_prefix_cache(input_ids, cache)

    entry = engine._exact_prefix_cache_entry((10, 11, 12))
    assert entry is not None
    assert entry.layers[0][0].shape == (1, 1, 3, 2)


def test_openai_effective_max_batch_size_caps_cuda_tp_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_MAX_BATCH_SIZE", raising=False)
    model = object()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 64) == 64
    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 128) == 128
    assert _effective_openai_max_batch_size(model, torch.device("cuda"), 256) == 128
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


def test_openai_short_tp_stream_uses_smaller_queue_batch_limit(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SHORT_STREAM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_SHORT_STREAM_HIGH_TOKEN_MIN", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_LARGE_STREAM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_LARGE_STREAM_MIN_TOKENS", raising=False)
    model = object()
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.max_batch_size = 128

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    short_stream = _QueuedGeneration([], 64, 0.0, True, queue.Queue())
    boundary_stream = _QueuedGeneration([], 256, 0.0, True, queue.Queue())
    sampled_short_stream = _QueuedGeneration([], 64, 0.7, True, queue.Queue())
    medium_stream = _QueuedGeneration([], 300, 0.0, True, queue.Queue())
    sampled_medium_stream = _QueuedGeneration([], 300, 0.7, True, queue.Queue())
    large_stream = _QueuedGeneration([], 512, 0.0, True, queue.Queue())
    short_completion = _QueuedGeneration([], 64, 0.0, False, queue.Queue())

    assert engine._queued_batch_limit(short_stream) == 56
    assert engine._queued_batch_limit(boundary_stream) == 48
    assert engine._queued_batch_limit(sampled_short_stream) == 128
    assert engine._queued_batch_limit(medium_stream) == 128
    assert engine._queued_batch_limit(sampled_medium_stream) == 128
    assert engine._queued_batch_limit(large_stream) == 32
    assert engine._queued_batch_limit(short_completion) == 128

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", "12")
    assert engine._queued_batch_limit(short_stream) == 12
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_SHORT_STREAM_HIGH_TOKEN_MIN", "300")
    assert engine._queued_batch_limit(boundary_stream) == 56
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_LARGE_STREAM_MAX_BATCH_SIZE", "20")
    assert engine._queued_batch_limit(large_stream) == 20


def test_openai_temperature_queue_batch_wait_uses_default_window(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", raising=False)
    engine = _cache_only_engine()
    engine.batch_wait_s = 0.010

    greedy = _QueuedGeneration([1, 2], 4, 0.0, True, queue.Queue())
    sampled = _QueuedGeneration([1, 2], 4, 0.7, True, queue.Queue())
    medium_sampled = _QueuedGeneration([1, 2], 300, 0.7, True, queue.Queue())
    long_sampled = _QueuedGeneration([1, 2], 600, 0.7, True, queue.Queue())

    assert engine._queued_batch_wait_s(greedy) == 0.010
    assert engine._queued_batch_wait_s(sampled) == 0.010
    assert engine._queued_batch_wait_s(medium_sampled) == 0.050
    assert engine._queued_batch_wait_s(long_sampled) == 0.010


def test_openai_temperature_queue_batch_wait_respects_env_floor(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MS", "5")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", "512")
    engine = _cache_only_engine()
    engine.batch_wait_s = 0.010

    sampled = _QueuedGeneration([1, 2], 300, 0.7, True, queue.Queue())

    assert engine._queued_batch_wait_s(sampled) == 0.010


def test_openai_tp_sampled_stream_uses_short_initial_queue_wait(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SAMPLED_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHORT_SAMPLED_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TEMPERATURE_BATCH_WAIT_MAX_TOKENS", raising=False)
    model = object()
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.max_batch_size = 16
    engine.batch_wait_s = 0.010

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    short_sampled = _QueuedGeneration([1, 2], 256, 0.7, True, queue.Queue())
    sampled = _QueuedGeneration([1, 2], 300, 0.7, True, queue.Queue())
    greedy = _QueuedGeneration([1, 2], 300, 0.0, True, queue.Queue())
    completion = _QueuedGeneration([1, 2], 300, 0.7, False, queue.Queue())
    long_sampled = _QueuedGeneration([1, 2], 600, 0.7, True, queue.Queue())

    assert engine._queued_initial_batch_wait_s(short_sampled) == 0.010
    assert engine._queued_initial_batch_wait_s(sampled) == 0.010
    assert engine._queued_initial_batch_wait_s(greedy) == 0.0
    assert engine._queued_initial_batch_wait_s(completion) == 0.0
    assert engine._queued_initial_batch_wait_s(long_sampled) == 0.0

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHORT_SAMPLED_INITIAL_BATCH_WAIT_MS", "3")
    assert engine._queued_initial_batch_wait_s(short_sampled) == 0.003
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SAMPLED_INITIAL_BATCH_WAIT_MS", "2")
    assert engine._queued_initial_batch_wait_s(sampled) == 0.002


def test_openai_sampled_batch_shape_bucket_uses_warmed_temperature_shapes(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SAMPLED_BATCH_SHAPE_BUCKETING", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SAMPLED_BATCH_SHAPE_BUCKET_MAX_RATIO", raising=False)
    model = object()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert _sampled_batch_shape_bucket_size(model, torch.device("cuda"), 7, 0.7) == 8
    assert _sampled_batch_shape_bucket_size(model, torch.device("cuda"), 9, 0.7) == 15
    assert _sampled_batch_shape_bucket_size(model, torch.device("cuda"), 17, 0.7) == 17
    assert _sampled_batch_shape_bucket_size(model, torch.device("cuda"), 7, 0.0) == 7
    assert _sampled_batch_shape_bucket_size(model, torch.device("cpu"), 7, 0.7) == 7

    monkeypatch.setenv("TORCHINFERNO_OPENAI_SAMPLED_BATCH_SHAPE_BUCKET_MAX_RATIO", "1.1")
    assert _sampled_batch_shape_bucket_size(model, torch.device("cuda"), 9, 0.7) == 9


def test_openai_collect_batch_stops_when_current_live_requests_are_ready() -> None:
    engine = _cache_only_engine()
    engine.max_batch_size = 16
    engine.batch_wait_s = 1.0
    engine._generation_queue = queue.Queue()
    engine._live_request_condition = threading.Condition()
    engine._live_requests = 2

    first = _QueuedGeneration([1], 300, 0.7, True, queue.Queue())
    second = _QueuedGeneration([2], 300, 0.7, True, queue.Queue())
    batch = [first, second]

    start = time.perf_counter()
    engine._collect_batch_until_deadline(batch, limit=16)
    elapsed = time.perf_counter() - start

    assert batch == [first, second]
    assert elapsed < 0.1


def test_openai_resident_temperature_warmup_keeps_short_batch_sizes(monkeypatch) -> None:
    engine = _cache_only_engine()
    engine.model = object()
    engine.device = torch.device("cpu")
    caches: list[tuple[int, int]] = []
    warmups: list[tuple[int, tuple[int, ...]]] = []

    class _Cache:
        seq_len = 0

    def generation_cache(batch_size: int, cache_tokens: int, *, model: object) -> _Cache:
        del model
        caches.append((batch_size, cache_tokens))
        return _Cache()

    def warmup_temperature(input_ids: torch.Tensor, cache: object, batch_size: int) -> None:
        del cache
        warmups.append((batch_size, tuple(int(dim) for dim in input_ids.shape)))

    monkeypatch.setattr(
        "torchinferno.openai_server._warmup_temperature_batch_sizes",
        lambda: (1, 8, 16, 64, 15),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._warmup_temperature_prompt_token_counts",
        lambda: (32, 55),
    )
    engine._generation_cache = generation_cache  # type: ignore[method-assign]
    engine._warmup_temperature_prefill_decode_graphs = warmup_temperature  # type: ignore[method-assign]

    engine._warmup_tensor_parallel_resident_temperature_graphs(vocab_size=128)

    assert caches == [(1, 512), (8, 512), (16, 512), (15, 512)]
    assert (64, (64, 32)) not in warmups
    assert (8, (8, 55)) in warmups
    assert (8, (1, 55)) in warmups
    assert (15, (15, 55)) in warmups
    assert (15, (1, 55)) in warmups


def test_openai_queued_batch_groups_streams_with_different_max_tokens() -> None:
    engine = _cache_only_engine()
    captured: list[list[int]] = []

    def run_stream_group(group: list[_QueuedGeneration]) -> None:
        captured.append([request.max_tokens for request in group])

    engine._run_queued_stream_group = run_stream_group  # type: ignore[method-assign]

    first = _QueuedGeneration([1, 2], 1, 0.0, True, queue.Queue())
    second = _QueuedGeneration([1, 3], 3, 0.0, True, queue.Queue())

    engine._run_queued_batch([first, second])

    assert captured == [[1, 3]]


def test_openai_stream_group_respects_per_request_max_tokens() -> None:
    engine = _cache_only_engine()
    engine.model = object()
    engine._shared_prefix_prompt_list_tokens = lambda prompts: 1  # type: ignore[method-assign]
    calls: list[tuple[list[list[int]], int, float]] = []

    def generate_prompt_list_batch_steps(
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: list[int] | None = None,
    ):
        del broadcast_tensor_parallel
        assert row_max_tokens == [1, 3]
        calls.append((prompts, max_tokens, temperature))
        yield [101, 201]
        assert any(isinstance(item, _GenerationDone) for item in first_queue.queue)
        yield [102, 202]
        yield [103, 203]

    engine._generate_prompt_list_batch_steps = generate_prompt_list_batch_steps  # type: ignore[method-assign]

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2], 1, 0.0, True, first_queue),
        _QueuedGeneration([1, 3], 3, 0.0, True, second_queue),
    ]

    engine._run_queued_stream_group(group)

    assert calls == [([[1, 2], [1, 3]], 3, 0.0)]
    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert first_items[:1] == [101]
    assert isinstance(first_items[1], _GenerationDone)
    assert second_items[:3] == [201, 202, 203]
    assert isinstance(second_items[3], _GenerationDone)


def test_openai_tp_single_stream_group_defaults_to_batch_path(monkeypatch) -> None:
    engine = _cache_only_engine()
    model = object()
    engine.model = model
    calls: list[tuple[list[list[int]], int, float, list[int] | None]] = []

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SINGLE_PROMPT_LIST_STREAM", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    def generate_prompt_list_batch_steps(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("single stream should keep the stable batch path by default")

    def generate_batch_steps(
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        temperature: float,
        row_max_tokens: list[int] | None = None,
    ):
        calls.append((input_ids.cpu().tolist(), max_tokens, temperature, row_max_tokens))
        yield [101]

    engine._generate_prompt_list_batch_steps = generate_prompt_list_batch_steps  # type: ignore[method-assign]
    engine._generate_batch_steps = generate_batch_steps  # type: ignore[method-assign]

    response_queue: queue.Queue[object] = queue.Queue()
    request = _QueuedGeneration([1, 2], 1, 0.0, True, response_queue)

    engine._run_queued_stream_group([request])

    assert calls == [([[1, 2]], 1, 0.0, [1])]
    items = _queue_items(response_queue)
    assert items[:1] == [101]
    assert isinstance(items[1], _GenerationDone)


def test_openai_tp_single_stream_group_uses_prompt_list_path(monkeypatch) -> None:
    engine = _cache_only_engine()
    model = object()
    engine.model = model
    calls: list[tuple[list[list[int]], int, float, list[int] | None]] = []

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SINGLE_PROMPT_LIST_STREAM", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    def generate_prompt_list_batch_steps(
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: list[int] | None = None,
    ):
        del broadcast_tensor_parallel
        calls.append((prompts, max_tokens, temperature, row_max_tokens))
        yield [101]

    def generate_batch_steps(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("single tensor-parallel stream should use prompt-list path")

    engine._generate_prompt_list_batch_steps = generate_prompt_list_batch_steps  # type: ignore[method-assign]
    engine._generate_batch_steps = generate_batch_steps  # type: ignore[method-assign]

    response_queue: queue.Queue[object] = queue.Queue()
    request = _QueuedGeneration([1, 2], 1, 0.0, True, response_queue)

    engine._run_queued_stream_group([request])

    assert calls == [([[1, 2]], 1, 0.0, [1])]
    items = _queue_items(response_queue)
    assert items[:1] == [101]
    assert isinstance(items[1], _GenerationDone)


def test_openai_stream_group_finishes_rows_on_stop_token() -> None:
    engine = _cache_only_engine()
    engine.model = object()
    engine.stop_token_ids = frozenset({99})
    captured: list[list[bool]] = []

    def generate_batch_steps(
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
            temperature: float,
            broadcast_tensor_parallel: bool = True,
            row_max_tokens: list[int] | None = None,
        ):
            del input_ids, max_tokens, temperature, broadcast_tensor_parallel, row_max_tokens
            yield [99, 201]
            captured.append([first.done, second.done])
            yield [None, 202]

    engine._generate_batch_steps = generate_batch_steps  # type: ignore[method-assign]

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 3, 0.0, True, first_queue)
    second = _QueuedGeneration([1, 3], 3, 0.0, True, second_queue)

    engine._run_queued_stream_group([first, second])

    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert isinstance(first_items[0], _GenerationDone)
    assert second_items[:2] == [201, 202]
    assert captured == [[True, False]]


def test_openai_completion_group_respects_per_request_max_tokens() -> None:
    engine = _cache_only_engine()
    engine.model = object()
    calls: list[tuple[list[list[int]], int, float]] = []

    def generate_batch_tokens(
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
    ) -> list[list[int]]:
        del broadcast_tensor_parallel
        calls.append((input_ids.tolist(), max_tokens, temperature))
        return [[11, 12, 13], [21, 22, 23]]

    engine._generate_batch_tokens = generate_batch_tokens  # type: ignore[method-assign]

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2], 1, 0.0, False, first_queue),
        _QueuedGeneration([3, 4], 3, 0.0, False, second_queue),
    ]

    engine._run_queued_completion_group(group)

    assert calls == [([[1, 2], [3, 4]], 3, 0.0)]
    assert first_queue.get_nowait().tokens == [11]
    assert second_queue.get_nowait().tokens == [21, 22, 23]


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
        ("new_group", "gloo"),
        ("barrier", {"group": "control"}),
    ]

    calls.clear()
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC", "1")
    _sync_tensor_parallel_command(model, torch.device("cuda"))

    assert calls == [
        ("sync", "cuda"),
        ("barrier", {"group": "control"}),
        ("sync", "cuda"),
    ]


def test_openai_tensor_parallel_short_streams_sync_after_default_window(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", raising=False)

    assert not _tp_command_cuda_sync_for_steps(0)
    assert not _tp_command_cuda_sync_for_steps(7)
    assert _tp_command_cuda_sync_for_steps(8)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", "3")
    assert not _tp_command_cuda_sync_for_steps(2)
    assert _tp_command_cuda_sync_for_steps(3)


def test_openai_tensor_parallel_sampled_identical_prompt_cache_defaults_pooled(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_IDENTICAL_PROMPT_CACHE_POOL", raising=False)
    model = type("FakeTPModel", (), {"world_size": 8})()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    assert _identical_prompt_cache_pool_enabled(model, 0.7)
    assert _identical_prompt_cache_pool_enabled(model, 0.0)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_IDENTICAL_PROMPT_CACHE_POOL", "0")
    assert not _identical_prompt_cache_pool_enabled(model, 0.7)


def test_openai_tensor_parallel_continue_sync_skips_broadcast_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SYNC_TP_CONTINUE", raising=False)
    model = type("FakeTPModel", (), {"world_size": 2})()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("broadcast")))

    assert _sync_tensor_parallel_continue(model, True, torch.device("cpu")) is True
    assert _sync_tensor_parallel_continue(model, False, torch.device("cpu")) is False


def test_openai_tensor_parallel_continue_sync_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SYNC_TP_CONTINUE", "1")
    model = type("FakeTPModel", (), {"world_size": 2})()
    calls: list[int] = []

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def broadcast(flag: torch.Tensor, *, src: int) -> None:
        assert src == 0
        calls.append(int(flag.item()))

    monkeypatch.setattr(dist, "broadcast", broadcast)

    assert _sync_tensor_parallel_continue(model, True, torch.device("cpu")) is True
    assert _sync_tensor_parallel_continue(model, False, torch.device("cpu")) is False
    assert calls == [1, 0]


def test_openai_tensor_parallel_decode_graphs_default_on(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_DECODE_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP", raising=False)

    class GraphProbeModel:
        world_size = 2
        prefill_graph_calls = 0
        prefill_logits_graph_calls = 0
        decode_graph_calls = 0
        decode_logits_graph_calls = 0
        ragged_decode_graph_calls = 0
        ragged_decode_logits_graph_calls = 0

        def try_prefill_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.prefill_graph_calls += 1
            return None

        def try_prefill_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.prefill_logits_graph_calls += 1
            return None

        def try_decode_one_token_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.decode_graph_calls += 1
            return None

        def try_decode_one_token_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.decode_logits_graph_calls += 1
            return None

        def try_decode_ragged_token_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.ragged_decode_graph_calls += 1
            return None

        def try_decode_ragged_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.ragged_decode_logits_graph_calls += 1
            return None

    model = GraphProbeModel()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    cache = object()

    assert _openai_cuda_graph_enabled_for_model(model)
    assert _openai_decode_graph_enabled(model)
    assert _openai_ragged_decode_graph_enabled(model)
    assert _prefers_exact_generation_cache(model)
    assert _try_prefill_graph(model, input_ids, cache, 0.0) is None
    assert _try_prefill_logits_graph(model, input_ids, cache) is None
    assert _try_decode_one_token_graph(model, input_ids, cache, 0.0) is None
    assert _try_decode_one_token_logits_graph(model, input_ids, cache) is None
    assert _try_decode_ragged_token_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
        temperature=0.0,
    ) is None
    assert _try_decode_ragged_logits_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
    ) is None
    assert model.prefill_graph_calls == 1
    assert model.prefill_logits_graph_calls == 1
    assert model.decode_graph_calls == 1
    assert model.decode_logits_graph_calls == 1
    assert model.ragged_decode_graph_calls == 1
    assert model.ragged_decode_logits_graph_calls == 1


def test_openai_tensor_parallel_ragged_decode_cuda_graphs_respect_low_level_env(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_CUDAGRAPH_DECODE_STEP", raising=False)
    monkeypatch.setenv("TORCHINFERNO_CUDAGRAPH_RAGGED_DECODE_STEP", "0")

    class GraphProbeModel:
        world_size = 2
        ragged_decode_graph_calls = 0
        ragged_decode_logits_graph_calls = 0

        def try_decode_ragged_token_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.ragged_decode_graph_calls += 1
            return None

        def try_decode_ragged_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.ragged_decode_logits_graph_calls += 1
            return None

    model = GraphProbeModel()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    cache = object()

    assert _openai_decode_graph_enabled(model)
    assert not _openai_ragged_decode_graph_enabled(model)
    assert _try_decode_ragged_token_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
        temperature=0.0,
    ) is None
    assert _try_decode_ragged_logits_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
    ) is None
    assert model.ragged_decode_graph_calls == 0
    assert model.ragged_decode_logits_graph_calls == 0


def test_openai_tensor_parallel_cuda_graphs_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_CUDAGRAPH", "0")

    class GraphProbeModel:
        world_size = 2

        def try_prefill_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("prefill graph should be disabled for TP serving")

        def try_prefill_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("prefill logits graph should be disabled for TP serving")

        def try_decode_one_token_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("decode graph should be disabled for TP serving")

        def try_decode_one_token_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("decode logits graph should be disabled for TP serving")

        def try_decode_ragged_token_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("ragged decode graph should be disabled for TP serving")

        def try_decode_ragged_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("ragged decode logits graph should be disabled for TP serving")

    model = GraphProbeModel()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    input_ids = torch.zeros((1, 1), dtype=torch.long)
    cache = object()

    assert not _openai_cuda_graph_enabled_for_model(model)
    assert not _prefers_exact_generation_cache(model)
    assert _try_prefill_graph(model, input_ids, cache, 0.0) is None
    assert _try_prefill_logits_graph(model, input_ids, cache) is None
    assert _try_decode_one_token_graph(model, input_ids, cache, 0.0) is None
    assert _try_decode_one_token_logits_graph(model, input_ids, cache) is None
    assert _try_decode_ragged_token_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
        temperature=0.0,
    ) is None
    assert _try_decode_ragged_logits_graph(
        model,
        input_ids,
        cache,
        seq_lens=torch.tensor([1]),
        row_indices=None,
    ) is None


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
    all_special_tokens = ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>")

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|end_of_text|>":
            return 128001
        if token == "<|eom_id|>":
            return 128008
        if token == "<|eot_id|>":
            return 128009
        return 0

    def get_added_vocab(self) -> dict[str, int]:
        return {
            "<|begin_of_text|>": 128000,
            "<|python_tag|>": 128010,
            "ordinary_added_token": 128011,
        }


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


class _CountingPromptTokenizer:
    eos_token_id = 0
    stop_token_ids: frozenset[int] = frozenset()

    def __init__(self) -> None:
        self.calls = 0

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        self.calls += 1
        content = str(messages[-1]["content"])
        return [len(content), self.calls]

    def decode_token(self, token_id: int) -> str:
        return str(token_id)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token_id) for token_id in token_ids)


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
        self.prompt_list_calls: list[tuple[list[list[int]], int, float, bool, list[int] | None]] = []

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
        row_max_tokens: list[int] | None = None,
    ):
        del row_max_tokens
        self.batch_calls.append(
            (
                [[int(token_id) for token_id in row.tolist()] for row in input_ids],
                max_tokens,
                temperature,
                broadcast_tensor_parallel,
            )
        )
        yield [2 for _ in range(input_ids.size(0))]

    def _generate_prompt_list_batch_steps(
        self,
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool,
        row_max_tokens: list[int] | None = None,
    ):
        self.prompt_list_calls.append(
            (
                [[int(token_id) for token_id in prompt] for prompt in prompts],
                max_tokens,
                temperature,
                broadcast_tensor_parallel,
                row_max_tokens,
            )
        )
        yield [2 for _ in prompts]


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
        self.ragged_shapes: list[tuple[int, int, tuple[int, ...], tuple[int, ...] | None]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _WarmupShapeCache:
        return _WarmupShapeCache()

    def try_prefill_logits_graph(self, input_ids: torch.Tensor, cache: _WarmupShapeCache) -> torch.Tensor:
        self.prefill_shapes.append((input_ids.size(0), input_ids.size(1)))
        return torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)

    def try_decode_one_token_logits_graph(self, input_ids: torch.Tensor, cache: _WarmupShapeCache) -> torch.Tensor:
        self.decode_shapes.append((input_ids.size(0), input_ids.size(1)))
        return torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)

    def try_decode_ragged_logits_graph(
        self,
        input_ids: torch.Tensor,
        cache: _WarmupShapeCache,
        *,
        seq_lens: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_tuple = None if row_indices is None else tuple(int(index) for index in row_indices.tolist())
        self.ragged_shapes.append(
            (
                input_ids.size(0),
                input_ids.size(1),
                tuple(int(seq_len) for seq_len in seq_lens.tolist()),
                row_tuple,
            )
        )
        return torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)

    def decode_ragged_logits(
        self,
        input_ids: torch.Tensor,
        cache: _WarmupShapeCache,
        *,
        seq_lens: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.try_decode_ragged_logits_graph(
            input_ids,
            cache,
            seq_lens=seq_lens,
            row_indices=row_indices,
        )


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


class _RaggedSharedPrefixRecordingModel(_SharedPrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.ragged_calls: list[tuple[list[list[int]], list[int], list[int] | None]] = []

    def decode_ragged_logits(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        seq_lens: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cache
        row_list = None if row_indices is None else [int(index) for index in row_indices.tolist()]
        self.ragged_calls.append(
            (
                [[int(token_id) for token_id in row.tolist()] for row in input_ids],
                [int(seq_len) for seq_len in seq_lens.tolist()],
                row_list,
            )
        )
        token = 2 + len(self.ragged_calls)
        logits = torch.zeros(input_ids.size(0), 1, self.config.vocab_size)
        logits[..., token] = 1.0
        return logits


class _TokenEchoSharedPrefixRecordingModel(_RaggedSharedPrefixRecordingModel):
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
        if return_last_logits_only:
            for row, token_id in enumerate(input_ids[:, -1].tolist()):
                logits[row, 0, int(token_id)] = 1.0
        else:
            for row in range(input_ids.size(0)):
                for offset, token_id in enumerate(input_ids[row].tolist()):
                    logits[row, offset, int(token_id)] = 1.0
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


class _SlowStreamEngine:
    model_id = "tiny"
    tokenizer = _ByteFallbackTokenizer(vocab_size=16)

    def __init__(self, *, delay_s: float) -> None:
        self.delay_s = delay_s

    def generate_chat_tokens(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ):
        del messages, max_tokens, temperature
        time.sleep(self.delay_s)
        yield 1


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
    engine._single_prefill_capture_seen = {}
    return engine


def _llama_tp_cache(batch_size: int, max_seq_len: int) -> Llama3TensorParallelCache:
    return Llama3TensorParallelCache(
        [
            Llama3TensorParallelLayerKVCache(
                batch_size,
                max_seq_len,
                1,
                2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]
    )


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


def _queue_items(items: queue.Queue[object]) -> list[object]:
    out: list[object] = []
    while not items.empty():
        out.append(items.get_nowait())
    return out


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
