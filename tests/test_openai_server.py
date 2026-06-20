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
from contextlib import nullcontext
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
    _fast_stream_end_bytes,
    _new_fast_http_stream_profile,
    _record_fast_http_profile,
    _stream_defer_role_enabled,
    _stream_inline_enabled,
    enable_tcp_nodelay,
)
from torchinferno.openai_server import (
    _GenerationDone,
    OpenAICompletionEngine,
    OpenAIServerConfig,
    _ByteFallbackTokenizer,
    _PersistentPromptListStepResult,
    _PersistentPromptListStepState,
    _QueuedGeneration,
    _TP_COMMAND_GENERATE_PROMPT_LISTS,
    _TP_COMMAND_GENERATE_TENSOR,
    _TP_COMMAND_META_FIELDS,
    _TP_COMMAND_ONLINE_CLOSE,
    _TP_COMMAND_ONLINE_START,
    _TP_COMMAND_ONLINE_STEP,
    _TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS,
    _TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE,
    _TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN,
    _TP_COMMAND_PROMPT_LIST_PERSISTENT_START,
    _TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP,
    _TP_COMMAND_STOP,
    _TP_COMMAND_TOKEN_BUDGET_CLOSE,
    _TP_COMMAND_TOKEN_BUDGET_DECODE_RUN,
    _TP_COMMAND_TOKEN_BUDGET_START,
    _TP_COMMAND_TOKEN_BUDGET_STEP,
    _TransformersChatTokenizer,
    _StreamRowState,
    _broadcast_tensor_parallel_generate,
    _broadcast_tensor_parallel_generate_prompt_lists,
    _broadcast_tensor_parallel_online_close,
    _broadcast_tensor_parallel_persistent_prompt_list_close,
    _broadcast_tensor_parallel_persistent_prompt_list_decode_run,
    _broadcast_tensor_parallel_persistent_prompt_list_start,
    _broadcast_tensor_parallel_persistent_prompt_list_step,
    _broadcast_tensor_parallel_token_budget_prompt_list_run,
    _broadcast_tensor_parallel_token_budget_close,
    _broadcast_tensor_parallel_token_budget_decode_run,
    _broadcast_tensor_parallel_token_budget_start,
    _broadcast_tensor_parallel_token_budget_step,
    _broadcast_tensor_parallel_online_start,
    _broadcast_tensor_parallel_online_step,
    _broadcast_tensor_parallel_online_submit_prompt_lists,
    _cache_row_slice,
    _copy_generation_cache_first_row,
    _copy_generation_cache_row,
    _copy_generation_cache_state_rows_padded,
    _decode_next_token_ragged,
    _disable_tp_shared_prefix_ragged_static_buckets,
    _distributed_server_command,
    _effective_openai_max_batch_size,
    _emit_stream_token,
    _flashinfer_prefill_runtime_enabled,
    _flashinfer_prefill_warmup_batch_sizes,
    _generation_cache_batch_capacity,
    _identical_prompt_cache_pool_enabled,
    _identical_prompt_prefill_graph_capture_enabled,
    _mark_generation_cache_prefix,
    _online_common_prefix_prefill_warmup_rows,
    _online_common_prefix_prefill_warmup_tokens,
    _online_common_prefix_suffix_prefill_warmup_batches,
    _online_common_prefix_suffix_prefill_warmup_tokens,
    _online_admit_per_step_cap,
    _online_decode_quantum,
    _online_initial_batch_wait_ms,
    _online_kv_bounded_concurrency_enabled,
    _online_kv_bounded_max_active_cap,
    _online_persistent_idle_ms,
    _online_refill_min_ready_requests,
    _online_step_sync_enabled,
    _openai_cuda_graph_enabled_for_model,
    _openai_decode_graph_enabled,
    _openai_ragged_decode_graph_enabled,
    _prepare_tensor_parallel_symm_mem_allreduce_auto,
    _prefill_cache_only,
    _persistent_prompt_list_scheduler_for_group,
    _persistent_prompt_list_decode_run_payload_from_tensor_payload,
    _persistent_prompt_list_decode_run_tensor_payload,
    _persistent_prompt_list_step_payload,
    _persistent_prompt_list_step_payload_from_tensor_payload,
    _persistent_prompt_list_step_tensor_payload,
    _prefill_repeated_prefix_next_token,
    _prefer_shared_prefix_padded_suffix_prefill,
    _prefers_exact_generation_cache,
    _prompt_list_tensor_payload,
    _runtime_ragged_decode_graph_capture_allowed_for_request,
    _runtime_prefill_graph_capture_enabled,
    _repeat_generation_cache_first_batch,
    _sampled_batch_shape_bucket_size,
    _set_generation_cache_rows_seq_lens,
    _should_reexec_distributed_server,
    _startup_warmup_enabled_for_cache_backend,
    _sync_tensor_parallel_command,
    _sync_tensor_parallel_continue,
    _set_shared_prefix_ragged_static_graph_bucket_mode,
    _shared_prefix_padded_suffix_bucketed_length,
    _shared_prefix_ragged_decode_row_plan,
    _tensor_parallel_symm_mem_allreduce_scope,
    _tensor_parallel_token_budget_lifecycle,
    _token_budget_scheduler_for_group,
    _token_budget_decode_run_payload,
    _token_budget_decode_run_payload_from_tensor_payload,
    _token_budget_decode_run_tensor_payload,
    _token_budget_prompt_list_step_payload,
    _token_budget_step_payload_from_tensor_payload,
    _token_budget_step_payload,
    _token_budget_step_tensor_payload,
    _tensor_parallel_worker_loop,
    _tp_command_cuda_sync_for_steps,
    _try_decode_ragged_token_graph,
    _try_decode_ragged_logits_graph,
    _try_decode_one_token_graph,
    _try_decode_one_token_logits_graph,
    _try_prefill_graph,
    _try_prefill_logits_graph,
    _try_prefill_selected_logits_graph,
    _warmup_prefill_cache_token_counts,
    _warmup_prompt_token_counts,
    _warmup_prefix_suffix_cache_token_counts,
    _warmup_prefix_suffix_token_counts,
    _warmup_ragged_decode_batch_sizes,
    _warmup_ragged_decode_cache_token_counts,
    _warmup_ragged_decode_extra_cache_specs,
    _warmup_ragged_decode_prompt_tokens,
    _warmup_ragged_decode_row_counts,
    _warmup_temperature_batch_sizes,
    _warmup_temperature_prompt_token_counts,
    build_parser,
    config_from_args,
    load_chat_tokenizer,
)
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelCache,
    Llama3TensorParallelLayerKVCache,
    PagedLlama3TensorParallelLayerKVCache,
)
from torchinferno.runtime.scheduler import TokenBudgetPlan, TokenBudgetScheduledChunk
from torchinferno.server.openai_protocol import chat_completion_chunk


def test_openai_server_cache_backend_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_BACKEND", "paged")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PAGE_SIZE", "32")

    args = build_parser().parse_args(["--model", "tiny"])
    config = config_from_args(args)

    assert config.cache_backend == "paged"
    assert config.page_size == 32


def test_openai_startup_warmup_skips_non_dense_cache_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STARTUP_WARMUP_NON_DENSE_CACHE", raising=False)

    assert _startup_warmup_enabled_for_cache_backend("dense")
    assert _startup_warmup_enabled_for_cache_backend("paged")

    monkeypatch.setenv("TORCHINFERNO_OPENAI_STARTUP_WARMUP_NON_DENSE_CACHE", "0")

    assert not _startup_warmup_enabled_for_cache_backend("paged")
    assert _startup_warmup_enabled_for_cache_backend("dense")


def test_flashinfer_prefill_warmup_batches_are_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_FLASHINFER_PREFILL_MAX_BATCH", raising=False)

    assert _flashinfer_prefill_warmup_batch_sizes([1, 2, 4, 64, 144]) == (1, 8)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_FLASHINFER_PREFILL_MAX_BATCH", "16")

    assert _flashinfer_prefill_warmup_batch_sizes([1, 2, 4, 64, 144]) == (1, 16)
    assert _flashinfer_prefill_warmup_batch_sizes([1, 2, 4]) == (1, 4)


def test_flashinfer_prefill_runtime_is_opt_in_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE", raising=False)

    assert not _flashinfer_prefill_runtime_enabled()

    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_FLASHINFER_PREFILL_DISABLE", "0")

    assert _flashinfer_prefill_runtime_enabled()


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


def test_openai_fast_stream_end_bytes_coalesce_final_frames() -> None:
    prefix = _chat_completion_chunk_prefix("chatcmpl-test", "test-model", 123)

    plain = _fast_stream_end_bytes(prefix, include_role=False, chunked=False)
    assert plain.endswith(b'data: [DONE]\n\n')
    assert b'"finish_reason":"stop"' in plain
    assert b'0\r\n\r\n' not in plain

    chunked = _fast_stream_end_bytes(prefix, include_role=True, chunked=True)
    assert chunked.endswith(b'0\r\n\r\n')
    assert b'{"role":"assistant"}' in chunked
    assert b'data: [DONE]\n\n' in chunked


def test_openai_fast_http_profile_writes_jsonl(monkeypatch, tmp_path) -> None:
    profile_path = tmp_path / "http-profile.jsonl"
    monkeypatch.setenv("TORCHINFERNO_OPENAI_FAST_HTTP_PROFILE_JSONL", str(profile_path))

    profile = _new_fast_http_stream_profile(
        max_tokens=3,
        temperature=0.7,
        keep_alive=True,
        request_ready_s=time.perf_counter(),
        parse_ms=1.5,
    )
    assert profile is not None
    profile["engine_tokens"] = 2
    profile["content_chunks"] = 2

    _record_fast_http_profile(profile)

    record = json.loads(profile_path.read_text())
    assert record["event"] == "fast_http_stream"
    assert record["max_tokens"] == 3
    assert record["temperature"] == 0.7
    assert record["keep_alive"] is True
    assert record["parse_ms"] == 1.5
    assert record["engine_tokens"] == 2
    assert record["content_chunks"] == 2
    assert record["total_ms"] >= 0.0
    assert "_start_s" not in record


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
    assert "--standalone" in command
    assert "--rdzv-endpoint" not in command
    assert command[command.index("--nproc-per-node") + 1] == "8"
    assert command[command.index("torchinferno.openai_server") - 1] == "-m"


def test_openai_server_auto_launch_honors_configured_rendezvous(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_TORCHRUN_RDZV_ENDPOINT", "127.0.0.1:29599")
    config = OpenAIServerConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tensor_parallel_size=8,
    )

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

    assert "--standalone" not in command
    assert command[command.index("--rdzv-backend") + 1] == "c10d"
    assert command[command.index("--rdzv-endpoint") + 1] == "127.0.0.1:29599"
    assert command[command.index("--rdzv-id") + 1].startswith("torchinferno-openai-")
    assert command[command.index("--rdzv-id") + 1].endswith("-29599")
    assert command[command.index("--rdzv-conf") + 1] == "is_host=true"


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
    assert set(_warmup_ragged_decode_batch_sizes()) == {8, 64}
    assert set(_warmup_ragged_decode_row_counts()) >= {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 56, 64}
    assert set(_warmup_ragged_decode_cache_token_counts()) >= {256, 512}
    assert (64, 1024) in set(_warmup_ragged_decode_extra_cache_specs())
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_EXTRA_CACHE_SPECS", "")
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


def test_openai_ragged_decode_warmup_captures_token_and_logits_graphs(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_BATCH_SIZES", "4")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_ROW_COUNTS", "4,2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_CACHE_TOKENS", "8")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_EXTRA_CACHE_SPECS", "")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_RAGGED_DECODE_PROMPT_TOKENS", "3")

    class _TokenWarmupShapeModel(_WarmupShapeModel):
        def __init__(self) -> None:
            super().__init__()
            self.token_ragged_shapes: list[tuple[int, int, tuple[int, ...] | None]] = []

        def try_decode_ragged_token_graph(
            self,
            input_ids: torch.Tensor,
            cache: _WarmupShapeCache,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
            temperature: float = 0.0,
        ) -> torch.Tensor:
            del cache, seq_lens, temperature
            row_tuple = None if row_indices is None else tuple(int(index) for index in row_indices.tolist())
            self.token_ragged_shapes.append((input_ids.size(0), input_ids.size(1), row_tuple))
            return torch.zeros(input_ids.size(0), dtype=torch.long)

    model = _TokenWarmupShapeModel()
    engine = object.__new__(OpenAICompletionEngine)
    engine.model = model
    engine.device = torch.device("cpu")
    engine.cache_backend = "dense"
    engine.page_size = 16
    engine._cache_pool = {}
    engine._microbatch_cache_pool = {}

    engine._warmup_tensor_parallel_ragged_decode_graphs(vocab_size=16)

    assert model.token_ragged_shapes == [(4, 1, None), (2, 1, (0, 1))]
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_UNIFIED_SCHEDULER", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "0")
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

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([1, 1, 1, 3, 2, 1, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([3], dtype=torch.long),
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
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

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([2, 1, 2, 3, 4, 1, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.75], dtype=torch.float64),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long),
        torch.tensor([1, 4], dtype=torch.long),
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
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


def test_tensor_parallel_worker_loop_handles_online_runtime_commands(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    commands: list[dict[str, object]] = [
        {
            "op": "online_start",
            "max_seq_len": 16,
            "max_active_requests": 4,
            "prefix_cache_capacity": 2,
            "prefill_token_budget": 8,
            "temperature": 0.25,
            "max_tokens": 6,
        },
        {
            "op": "online_submit",
            "input_id_lists": [[1, 2], [3]],
            "row_max_tokens": [5, 6],
            "eos_token_id": 0,
            "arrival_step": 7,
            "request_id_start": 10,
        },
        {"op": "online_step"},
        {"op": "online_step", "steps": 3},
        {"op": "online_close"},
        {"op": "stop"},
    ]
    instances: list[object] = []

    class RuntimeEngine:
        def __init__(
            self,
            model: object,
            *,
            device: torch.device,
            cache_backend: str,
            page_size: int,
            temperature: float,
            max_active_requests: int,
            prefix_cache_capacity: int,
            prefill_token_budget: int | None,
            enable_ragged_decode: bool = True,
            store_reusable_prefixes: bool = True,
            store_full_prompt_prefixes: bool = True,
            pin_shared_prefix: bool = False,
            graph_prefill: bool = False,
            prefill_chunk_size: int | None = None,
            admit_min_ready_requests: int | None = None,
            admit_per_step_cap: int | None = None,
        ) -> None:
            self.init_args = (
                model,
                device,
                cache_backend,
                page_size,
                temperature,
                max_active_requests,
                prefix_cache_capacity,
                prefill_token_budget,
                enable_ragged_decode,
                store_reusable_prefixes,
                store_full_prompt_prefixes,
                admit_min_ready_requests,
                admit_per_step_cap,
            )
            self.started: int | None = None
            self.submitted: list[object] = []
            self.steps = 0
            instances.append(self)

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            self.started = max_seq_len

        def submit_online(self, request: object) -> None:
            self.submitted.append(request)

        def step_online(self) -> list[object]:
            self.steps += 1
            return []

        def has_online_work(self) -> bool:
            return True

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        payload[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)

    engine = _WorkerLoopRecordingEngine()
    engine.model = object()
    engine.cache_backend = "paged"
    engine.page_size = 2

    _tensor_parallel_worker_loop(engine)

    assert commands == []
    assert len(instances) == 1
    runtime = instances[0]
    assert runtime.started == 16
    assert runtime.steps == 4
    assert runtime.init_args[2:] == ("paged", 2, 0.25, 4, 2, 8, True, True, True, None, 48)
    assert [(request.prompt, request.max_new_tokens, request.arrival_step, request.eos_token_id) for request in runtime.submitted] == [
        ((1, 2), 5, 7, 0),
        ((3,), 6, 7, 0),
    ]
    assert [request.request_id for request in runtime.submitted] == ["10", "11"]


def test_tensor_parallel_worker_loop_skips_online_step_barrier_when_disabled(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC", "0")
    commands: list[dict[str, object]] = [
        {
            "op": "online_start",
            "max_seq_len": 16,
            "max_active_requests": 4,
            "prefix_cache_capacity": 2,
            "prefill_token_budget": 8,
            "temperature": 0.0,
            "max_tokens": 6,
        },
        {"op": "online_step"},
        {"op": "stop"},
    ]
    syncs: list[str] = []

    class RuntimeEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            del max_seq_len, external_cache

        def has_online_work(self) -> bool:
            return False

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        payload[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )

    engine = _WorkerLoopRecordingEngine()
    engine.model = object()

    _tensor_parallel_worker_loop(engine)

    assert commands == []
    assert syncs == ["sync"]


def test_tensor_parallel_worker_loop_receives_online_tensor_commands(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_ONLINE_START, 0, 0, 0, 6, 1, 16, 4, 2, 8, 1], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([_TP_COMMAND_ONLINE_SUBMIT_PROMPT_LISTS, 1, 2, 2, 6, 1, 7, 0, 10, 0, 0], dtype=torch.long),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2, 1], dtype=torch.long),
        torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
        torch.tensor([5, 6], dtype=torch.long),
        torch.tensor([_TP_COMMAND_ONLINE_STEP, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_ONLINE_CLOSE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]
    instances: list[object] = []

    class RuntimeEngine:
        def __init__(
            self,
            model: object,
            *,
            device: torch.device,
            cache_backend: str,
            page_size: int,
            temperature: float,
            max_active_requests: int,
            prefix_cache_capacity: int,
            prefill_token_budget: int | None,
            enable_ragged_decode: bool = True,
            store_reusable_prefixes: bool = True,
            store_full_prompt_prefixes: bool = True,
            pin_shared_prefix: bool = False,
            graph_prefill: bool = False,
            prefill_chunk_size: int | None = None,
            admit_min_ready_requests: int | None = None,
            admit_per_step_cap: int | None = None,
        ) -> None:
            self.init_args = (
                model,
                device,
                cache_backend,
                page_size,
                temperature,
                max_active_requests,
                prefix_cache_capacity,
                prefill_token_budget,
                enable_ragged_decode,
                store_reusable_prefixes,
                store_full_prompt_prefixes,
                admit_min_ready_requests,
                admit_per_step_cap,
            )
            self.started: int | None = None
            self.submitted: list[object] = []
            self.steps = 0
            instances.append(self)

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            self.started = max_seq_len

        def submit_online(self, request: object) -> None:
            self.submitted.append(request)

        def step_online(self) -> list[object]:
            self.steps += 1
            return []

        def has_online_work(self) -> bool:
            return True

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
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)

    engine = _WorkerLoopRecordingEngine()
    engine.model = model
    engine.cache_backend = "paged"
    engine.page_size = 2

    _tensor_parallel_worker_loop(engine)

    assert payloads == []
    assert len(instances) == 1
    runtime = instances[0]
    assert runtime.started == 16
    assert runtime.steps == 3
    assert runtime.init_args[2:] == ("paged", 2, 0.25, 4, 2, 8, True, True, False, None, 48)
    assert [(request.request_id, request.prompt, request.max_new_tokens) for request in runtime.submitted] == [
        ("10", (1, 2), 5),
        ("11", (3,), 6),
    ]


def test_tensor_parallel_worker_loop_dispatches_persistent_prompt_list_step(monkeypatch) -> None:
    import torch.distributed as dist

    payload = {
        "op": "persistent_prompt_list_step",
        "step": 2,
        "decode_rows": [1],
        "prefill": [{"request_id": "3", "row": 0, "prompt": [1, 2, 3], "max_tokens": 4}],
    }
    commands: list[dict[str, object]] = [payload, {"op": "stop"}]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = object()
            self.handled: list[dict[str, object]] = []

        def _handle_persistent_prompt_list_step_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

    def broadcast_object_list(command: list[object], *, src: int) -> None:
        del src
        command[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert commands == []
    assert engine.handled == [payload]


def test_tensor_parallel_worker_loop_dispatches_token_budget_step(monkeypatch) -> None:
    import torch.distributed as dist

    payload = {
        "op": "token_budget_step",
        "step": 2,
        "decode_rows": [1],
        "prefill_rows": [0],
        "chunks": [
            {
                "request_id": "3",
                "row": 0,
                "kind": "prefill",
                "start_token": 2,
                "token_count": 1,
                "prompt_complete": True,
                "emits_token": True,
                "prompt_chunk": [9],
                "prompt_tokens": 3,
                "max_tokens": 4,
            }
        ],
        "finished_request_ids": [],
        "scheduled_tokens": 1,
    }
    commands: list[dict[str, object]] = [payload, {"op": "stop"}]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = object()
            self.handled: list[dict[str, object]] = []

        def _handle_token_budget_step_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

    def broadcast_object_list(command: list[object], *, src: int) -> None:
        del src
        command[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert commands == []
    assert engine.handled == [payload]


def test_tensor_parallel_worker_loop_dispatches_token_budget_decode_run(monkeypatch) -> None:
    import torch.distributed as dist

    payload = {
        "op": "token_budget_decode_run",
        "step_count": 2,
        "steps": [
            {
                "op": "token_budget_step",
                "step": 1,
                "decode_rows": [0],
                "prefill_rows": [],
                "chunks": [
                    {
                        "request_id": "0",
                        "row": 0,
                        "kind": "decode",
                        "start_token": 3,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    }
                ],
                "finished_request_ids": [],
                "scheduled_tokens": 1,
            }
        ],
    }
    commands: list[dict[str, object]] = [payload, {"op": "stop"}]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = object()
            self.handled: list[dict[str, object]] = []

        def _handle_token_budget_decode_run_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

    def broadcast_object_list(command: list[object], *, src: int) -> None:
        del src
        command[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert commands == []
    assert engine.handled == [payload]


def test_tensor_parallel_worker_loop_dispatches_token_budget_prompt_list_run(monkeypatch) -> None:
    import torch.distributed as dist

    payload = {
        "op": "token_budget_prompt_list_run",
        "input_id_lists": [[1, 2, 3], [1, 2, 4]],
        "max_tokens": 2,
        "row_max_tokens": [2, 1],
        "temperature": 0.0,
        "prefix_tokens": 2,
        "max_active_rows": 2,
        "max_scheduled_tokens": 8,
        "decode_run_steps": 2,
        "arrival_steps": [0, 1],
    }
    commands: list[dict[str, object]] = [payload, {"op": "stop"}]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = object()
            self.handled: list[dict[str, object]] = []

        def _handle_token_budget_prompt_list_run_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

    def broadcast_object_list(command: list[object], *, src: int) -> None:
        del src
        command[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert commands == []
    assert engine.handled == [payload]


def test_tensor_parallel_online_broadcast_helpers(monkeypatch) -> None:
    import torch.distributed as dist

    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0})()
    captured: list[dict[str, object]] = []

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        captured.append(dict(payload[0]))

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "0")

    _broadcast_tensor_parallel_online_start(
        model,
        max_seq_len=16,
        max_active_requests=4,
        prefix_cache_capacity=2,
        prefill_token_budget=None,
        temperature=0.25,
    )
    _broadcast_tensor_parallel_online_submit_prompt_lists(
        model,
        [[1, 2], [3]],
        max_tokens=6,
        row_max_tokens=[5, 6],
        arrival_step=7,
        eos_token_id=0,
    )
    _broadcast_tensor_parallel_online_step(model)
    _broadcast_tensor_parallel_online_step(model, 4)
    _broadcast_tensor_parallel_online_close(model)

    assert captured == [
        {
            "op": "online_start",
            "max_seq_len": 16,
            "max_active_requests": 4,
            "prefix_cache_capacity": 2,
            "prefill_token_budget": 0,
            "temperature": 0.25,
            "enable_ragged_decode": True,
            "store_reusable_prefixes": True,
            "store_full_prompt_prefixes": True,
            "max_tokens": 0,
        },
        {
            "op": "online_submit",
            "input_id_lists": [[1, 2], [3]],
            "max_tokens": 6,
            "row_max_tokens": [5, 6],
            "arrival_step": 7,
            "eos_token_id": 0,
            "request_id_start": 0,
        },
        {"op": "online_step"},
        {"op": "online_step", "steps": 4},
        {"op": "online_close"},
    ]


def test_tensor_parallel_token_budget_prompt_list_run_broadcast_helper(monkeypatch) -> None:
    import torch.distributed as dist

    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0})()
    captured: list[dict[str, object]] = []

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        captured.append(dict(payload[0]))

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)

    _broadcast_tensor_parallel_token_budget_prompt_list_run(
        model,
        [[1, 2, 3], [1, 2, 4]],
        max_tokens=3,
        temperature=0.25,
        row_max_tokens=[3, 1],
        prefix_tokens=2,
        max_active_rows=2,
        max_scheduled_tokens=8,
        prefill_chunk_size=4,
        decode_run_steps=2,
        arrival_steps=[0, 1],
        static_graph_buckets=True,
    )

    assert captured == [
        {
            "op": "token_budget_prompt_list_run",
            "input_id_lists": [[1, 2, 3], [1, 2, 4]],
            "max_tokens": 3,
            "row_max_tokens": [3, 1],
            "temperature": 0.25,
            "prefix_tokens": 2,
            "max_active_rows": 2,
            "max_scheduled_tokens": 8,
            "prefill_chunk_size": 4,
            "decode_run_steps": 2,
            "arrival_steps": [0, 1],
            "static_graph_buckets": True,
        }
    ]


def test_tensor_parallel_persistent_prompt_list_step_broadcast_helper(monkeypatch) -> None:
    import torch.distributed as dist

    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0})()
    captured: list[dict[str, object]] = []

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        captured.append(dict(payload[0]))

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "0")

    _broadcast_tensor_parallel_persistent_prompt_list_start(
        model,
        prefix=[1, 2, 3],
        cache_batch_size=4,
        max_seq_len=8,
        temperature=0.25,
    )
    _broadcast_tensor_parallel_persistent_prompt_list_step(
        model,
        {"step": 1, "decode_rows": [0], "prefill": []},
    )
    _broadcast_tensor_parallel_persistent_prompt_list_close(model)

    assert captured == [
        {
            "op": "persistent_prompt_list_start",
            "prefix": [1, 2, 3],
            "cache_batch_size": 4,
            "max_seq_len": 8,
            "temperature": 0.25,
            "max_tokens": 0,
        },
        {
            "op": "persistent_prompt_list_step",
            "step": 1,
            "decode_rows": [0],
            "prefill": [],
        },
        {"op": "persistent_prompt_list_close"},
    ]


def test_openai_persistent_prompt_list_step_tensor_payload_round_trips() -> None:
    payload = {
        "op": "persistent_prompt_list_step",
        "step": 4,
        "decode_request_ids": ["0"],
        "decode_rows": [0],
        "prefill": [
            {
                "request_id": "1",
                "row": 1,
                "prompt": [1, 2, 3, 5],
                "max_tokens": 2,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [5],
                "prefill_tokens": 1,
            },
            {
                "request_id": "2",
                "row": 2,
                "prompt": [1, 2, 3, 6, 7],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [6, 7],
                "prefill_tokens": 2,
            },
            {
                "request_id": "3",
                "row": 3,
                "prompt": [1, 2, 3, 8, 9],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [8],
                "prefill_tokens": 1,
                "start_token": 3,
                "prompt_chunk": [8],
                "prompt_complete": False,
                "emits_token": False,
            },
        ],
        "prefill_groups": [],
        "finished_after_prefill": ["1"],
        "temperature": 0.25,
        "static_graph_buckets": True,
    }

    meta, temp, decode_tensor, prefill_tensor, prompt_lengths, prompt_rows, finished_ids = (
        _persistent_prompt_list_step_tensor_payload(payload, torch.device("cpu"))
    )
    restored = _persistent_prompt_list_step_payload_from_tensor_payload(
        meta,
        temp,
        decode_tensor,
        prefill_tensor,
        prompt_lengths,
        prompt_rows,
        finished_ids,
    )

    assert restored == {
        "op": "persistent_prompt_list_step",
        "step": 4,
        "decode_request_ids": ["0"],
        "decode_rows": [0],
        "prefill": payload["prefill"],
        "prefill_groups": [
            {
                "request_ids": ["1", "2"],
                "rows": [1, 2],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1, 2],
            }
        ],
        "finished_after_prefill": ["1"],
        "temperature": 0.25,
        "static_graph_buckets": True,
    }


def test_openai_persistent_prompt_list_decode_run_tensor_payload_round_trips() -> None:
    meta, temp = _persistent_prompt_list_decode_run_tensor_payload(
        {
            "op": "persistent_prompt_list_decode_run",
            "start_step": 5,
            "step_count": 3,
            "temperature": 0.25,
            "static_graph_buckets": True,
        },
        torch.device("cpu"),
    )

    restored = _persistent_prompt_list_decode_run_payload_from_tensor_payload(meta, temp)

    assert restored == {
        "op": "persistent_prompt_list_decode_run",
        "start_step": 5,
        "step_count": 3,
        "temperature": 0.25,
        "static_graph_buckets": True,
    }


def test_tensor_parallel_persistent_prompt_list_tensor_broadcast(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_persistent_prompt_list_start(
        model,
        prefix=[1, 2, 3],
        cache_batch_size=4,
        max_seq_len=8,
        temperature=0.25,
        max_tokens=5,
    )
    _broadcast_tensor_parallel_persistent_prompt_list_step(
        model,
        {
            "op": "persistent_prompt_list_step",
            "step": 4,
            "decode_request_ids": ["0"],
            "decode_rows": [0],
            "prefill": [
                {
                    "request_id": "1",
                    "row": 1,
                    "prompt": [1, 2, 3, 5],
                    "max_tokens": 2,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [5],
                    "prefill_tokens": 1,
                }
            ],
            "prefill_groups": [],
            "finished_after_prefill": [],
            "temperature": 0.25,
        },
    )
    _broadcast_tensor_parallel_persistent_prompt_list_decode_run(
        model,
        {
            "op": "persistent_prompt_list_decode_run",
            "start_step": 5,
            "step_count": 2,
            "temperature": 0.25,
            "static_graph_buckets": True,
        },
    )
    _broadcast_tensor_parallel_persistent_prompt_list_close(model)

    assert captured[0].tolist() == [_TP_COMMAND_PROMPT_LIST_PERSISTENT_START, 0, 0, 0, 5, 3, 8, 4, 0, 0, 0]
    assert captured[1].tolist() == [0.25]
    assert captured[2].tolist() == [1, 2, 3]
    assert captured[3].tolist() == [_TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP, 4, 1, 1, 4, 0, 6, 2, 0, 0, 0]
    assert captured[4].tolist() == [0.25]
    assert captured[5].tolist() == [[0, 0]]
    assert captured[6].tolist() == [[1, 1, 2, 3, 1, 3]]
    assert captured[7].tolist() == [4]
    assert captured[8].tolist() == [[1, 2, 3, 5]]
    assert captured[9].tolist() == []
    assert captured[10].tolist() == [
        _TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN,
        5,
        2,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
    ]
    assert captured[11].tolist() == [0.25]
    assert captured[12].tolist() == [_TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_prompt_list_tensor_payload_pads_rows_once() -> None:
    token_rows, lengths = _prompt_list_tensor_payload(
        [[1, 2], [3, 4, 5], []],
        torch.device("cpu"),
    )

    assert lengths.tolist() == [2, 3, 0]
    assert token_rows.tolist() == [[1, 2, 0], [3, 4, 5], [0, 0, 0]]


def test_tensor_parallel_tensor_generate_broadcast_uses_full_meta(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_generate(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        max_tokens=4,
        temperature=0.5,
        stream=True,
        row_max_tokens=[2],
    )

    assert captured[0].numel() == _TP_COMMAND_META_FIELDS
    assert captured[0].tolist() == [_TP_COMMAND_GENERATE_TENSOR, 1, 1, 3, 4, 1, 0, 0, 0, 0, 0]
    assert captured[1].tolist() == [0.5]
    assert captured[2].tolist() == [3]
    assert captured[3].tolist() == [[1, 2, 3]]
    assert captured[4].tolist() == [2]


def test_tensor_parallel_tensor_prompt_list_broadcast_uses_full_meta(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_generate_prompt_lists(
        model,
        [[1, 2], [3, 4, 5]],
        max_tokens=6,
        temperature=0.0,
        stream=True,
        row_max_tokens=[4, 6],
    )

    assert captured[0].numel() == _TP_COMMAND_META_FIELDS
    assert captured[0].tolist() == [_TP_COMMAND_GENERATE_PROMPT_LISTS, 1, 2, 3, 6, 1, 0, 0, 0, 0, 0]
    assert captured[1].tolist() == [0.0]
    assert captured[2].tolist() == [2, 3]
    assert captured[3].tolist() == [[1, 2, 0], [3, 4, 5]]
    assert captured[4].tolist() == [4, 6]


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
    engine.tokenizer = types.SimpleNamespace(eos_token_id=None)

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


def test_openai_prompt_list_tp_prefix_cache_restore_syncs_empty_groups(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_BATCH_RESTORE", "1")
    model = _TokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    sync_values: list[int] = []

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_world_size", lambda candidate: 2)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_all_ranks_same_int",
        lambda model, value, device: sync_values.append(value) or True,
    )

    groups = engine._prefix_cached_prompt_groups([[10, 11, 12, 4], [10, 11, 13, 5]])

    assert groups == []
    assert sync_values == [0]


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
        **_kw: object,
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


def test_openai_tp_identical_temperature_prefill_capture_defaults_off(monkeypatch) -> None:
    model = _RuntimePrefillLogitsGraphCaptureModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.tokenizer = _PrefixTokenizer()
    engine.stop_token_ids = frozenset()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_IDENTICAL_TEMPERATURE_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    steps = list(
        engine._generate_identical_prompt_batch_steps(
            torch.tensor([[10, 11]], dtype=torch.long),
            batch_size=3,
            max_tokens=1,
            temperature=0.7,
        )
    )

    assert len(steps) == 1
    assert model.capture_flags == [False]
    assert model.graph_inputs == []
    assert model.forward_inputs == [[10, 11]]


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


def test_openai_identical_prompt_uniform_logits_cache_defers_kv_restore() -> None:
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
            max_tokens=2,
            temperature=0.0,
        )
    )
    model.forward_inputs.clear()

    restore_calls: list[list[list[int]]] = []
    original_restore = engine._restore_exact_prefix_cache

    def restore_exact_prefix_cache(input_ids: torch.Tensor, cache: object) -> int:
        restore_calls.append(input_ids.detach().cpu().tolist())
        return original_restore(input_ids, cache)

    engine._restore_exact_prefix_cache = restore_exact_prefix_cache  # type: ignore[method-assign]

    steps = list(
        engine._generate_identical_prompt_batch_steps(
            input_ids,
            batch_size=3,
            max_tokens=2,
            temperature=0.0,
        )
    )

    assert steps == [[2, 2, 2], [2, 2, 2]]
    assert model.forward_inputs == []
    assert restore_calls == []


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
    assert len(model.forward_inputs) >= 1
    first_call = model.forward_inputs[0]
    assert len(first_call) >= 2
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


def test_openai_prefill_cache_only_prefers_model_hook() -> None:
    class _CacheOnlyModel(_PrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.cache_only_inputs: list[list[int]] = []

        def prefill_cache_only(self, input_ids: torch.Tensor, cache: _PrefixRecordingCache) -> _PrefixRecordingCache:
            self.cache_only_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
            layer = cache.layers[0]
            start = layer.seq_len
            end = start + input_ids.size(1)
            layer.keys[: input_ids.size(0), :, start:end, :].fill_(2)
            layer.values[: input_ids.size(0), :, start:end, :].fill_(2)
            layer.seq_len = end
            return cache

    model = _CacheOnlyModel()
    cache = model.allocate_cache(1, 4)

    returned = _prefill_cache_only(model, torch.tensor([[1, 2, 3]]), cache)

    assert returned is cache
    assert model.cache_only_inputs == [[1, 2, 3]]
    assert model.forward_inputs == []
    assert cache.seq_len == 3
    assert torch.equal(cache.layers[0].keys[0, 0, :3, 0], torch.tensor([2.0, 2.0, 2.0]))


def test_openai_prefill_cache_only_keeps_short_tensor_parallel_prefixes_on_forward(monkeypatch) -> None:
    class _CacheOnlyModel(_PrefixRecordingModel):
        def __init__(self) -> None:
            super().__init__()
            self.cache_only_inputs: list[list[int]] = []

        def prefill_cache_only(self, input_ids: torch.Tensor, cache: _PrefixRecordingCache) -> _PrefixRecordingCache:
            self.cache_only_inputs.append([int(token_id) for token_id in input_ids[0].tolist()])
            return cache

    model = _CacheOnlyModel()
    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_model", lambda candidate: candidate is model)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    cache = model.allocate_cache(1, 4)

    returned = _prefill_cache_only(model, torch.tensor([[1, 2, 3]]), cache)

    assert returned is cache
    assert model.cache_only_inputs == []
    assert model.forward_inputs == [[1, 2, 3]]
    assert cache.seq_len == 3
    assert torch.equal(cache.layers[0].keys[0, 0, :3, 0], torch.tensor([1.0, 1.0, 1.0]))


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
    assert not any(model.capture_flags)


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


def test_openai_tp_batched_prefill_capture_waits_for_repeated_shape(monkeypatch) -> None:
    model = object()
    engine = _cache_only_engine()
    engine.device = torch.device("cuda")
    cache = type(
        "Cache",
        (),
        {
            "seq_len": 0,
            "layers": [type("Layer", (), {"max_seq_len": 256})()],
        },
    )()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.zeros((4, 16), dtype=torch.long)

    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
    )
    assert engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
    )


def test_openai_tp_batched_prefill_capture_uses_uniform_active_rows(monkeypatch) -> None:
    model = object()
    engine = _cache_only_engine()
    engine.device = torch.device("cuda")
    layer = type("Layer", (), {"max_seq_len": 256})()

    class NonUniformCache:
        layers = [layer]

        def __init__(self) -> None:
            self.sliced_rows: tuple[int, ...] | None = None

        @property
        def seq_len(self) -> int:
            raise ValueError("selected cache rows must have the same sequence length")

        def for_rows(self, rows: tuple[int, ...]) -> object:
            self.sliced_rows = rows
            return type(
                "UniformCache",
                (),
                {
                    "seq_len": 0,
                    "layers": [layer],
                },
            )()

    cache = NonUniformCache()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.zeros((4, 16), dtype=torch.long)

    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
        selected_logits=True,
    )
    assert engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
        selected_logits=True,
    )
    assert cache.sliced_rows == (0, 1, 2, 3)


def test_openai_tp_selected_logits_prefill_capture_is_opt_in(monkeypatch) -> None:
    model = object()
    engine = _cache_only_engine()
    engine.device = torch.device("cuda")
    cache = type(
        "Cache",
        (),
        {
            "seq_len": 0,
            "layers": [type("Layer", (), {"max_seq_len": 256})()],
        },
    )()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.zeros((4, 16), dtype=torch.long)

    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
        selected_logits=True,
    )


def test_openai_tp_selected_logits_prefill_capture_skips_short_default(monkeypatch) -> None:
    model = object()
    engine = _cache_only_engine()
    engine.device = torch.device("cuda")
    cache = type(
        "Cache",
        (),
        {
            "seq_len": 0,
            "layers": [type("Layer", (), {"max_seq_len": 256})()],
        },
    )()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SELECTED_LOGITS_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.zeros((64, 16), dtype=torch.long)

    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
        selected_logits=True,
    )
    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=256,
        selected_logits=True,
    )
    assert engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=256,
        selected_logits=True,
    )


def test_openai_tp_batched_prefill_capture_respects_explicit_disable(monkeypatch) -> None:
    model = object()
    engine = _cache_only_engine()
    engine.device = torch.device("cuda")
    cache = type(
        "Cache",
        (),
        {
            "seq_len": 0,
            "layers": [type("Layer", (), {"max_seq_len": 256})()],
        },
    )()

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    input_ids = torch.zeros((4, 16), dtype=torch.long)

    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
    )
    assert not engine._batched_prefill_graph_capture_enabled(
        model,
        input_ids,
        cache,
        temperature=0.0,
        max_tokens=128,
    )


def test_openai_tp_runtime_prefill_capture_defaults_on_but_env_can_disable(monkeypatch) -> None:
    model = object()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=256)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=96)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=128)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=384)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHORT_TEMPERATURE_PREFILL_CAPTURE", "1")
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=128)
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHORT_TEMPERATURE_PREFILL_CAPTURE", raising=False)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_IN_SKIP_WINDOW", "1")
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=256)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_IN_SKIP_WINDOW", raising=False)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", "0")
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)


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

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE", "0")
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", "128")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_IN_SKIP_WINDOW", "1")
    assert not _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=128)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_PREFILL_CAPTURE_IN_SKIP_WINDOW", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE_MAX_TOKENS", "512")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TEMPERATURE_PREFILL_CAPTURE_SKIP_MAX_TOKENS", "128")
    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=300)


def test_openai_tp_runtime_ragged_decode_capture_uses_short_and_long_windows(monkeypatch) -> None:
    model = object()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MIN_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=128)
    assert not _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=300)
    assert not _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=400)
    assert _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=512)
    assert _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=1024)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MAX_TOKENS", "0")
    assert not _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=128)
    assert _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=512)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_RUNTIME_RAGGED_DECODE_CAPTURE_MIN_TOKENS", "0")
    assert not _runtime_ragged_decode_graph_capture_allowed_for_request(model, max_tokens=512)


def test_openai_tp_identical_temperature_prefill_capture_has_specific_override(monkeypatch) -> None:
    model = object()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_RUNTIME_TEMPERATURE_PREFILL_CAPTURE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_IDENTICAL_TEMPERATURE_PREFILL_CAPTURE", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert _runtime_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=384)
    assert not _identical_prompt_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=384)
    assert _identical_prompt_prefill_graph_capture_enabled(model, temperature=0.0, max_tokens=512)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_IDENTICAL_TEMPERATURE_PREFILL_CAPTURE", "1")
    assert _identical_prompt_prefill_graph_capture_enabled(model, temperature=0.7, max_tokens=384)


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


def test_openai_prompt_list_segments_large_batches_without_nested_broadcast(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PROMPT_LIST_SEGMENT_ROWS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    model = _SharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    broadcasts: list[tuple[list[list[int]], dict[str, object]]] = []

    def broadcast_prompt_lists(model_arg: object, prompts: object, **kwargs: object) -> None:
        assert model_arg is model
        broadcasts.append(([[int(token_id) for token_id in prompt] for prompt in prompts], kwargs))

    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_generate_prompt_lists",
        broadcast_prompt_lists,
    )

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 12], [10, 11, 13], [10, 11, 14], [10, 11, 15]],
            max_tokens=2,
            temperature=0.0,
            broadcast_tensor_parallel=True,
        )
    )

    assert steps == [[2, 2, 2, 2], [2, 2, 2, 2]]
    assert len(broadcasts) == 1
    assert len(broadcasts[0][0]) == 4
    assert all(len(call) <= 2 for call in model.forward_inputs)


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


def test_openai_shared_prefix_padded_suffix_prefill_can_select_last_real_logits(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_SKIPPED_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_TOTAL_TOKENS", "1")
    model = _SelectedTokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
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
    assert model.selected_logit_positions == [[1, 0]]


def test_openai_shared_prefix_padded_suffix_prefill_can_use_static_physical_batch(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_STATIC_BATCH", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_SKIPPED_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_TOTAL_TOKENS", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._generation_cache_batch_capacity",
        lambda model, requested_batch: 4,
    )
    model = _SelectedTokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert model.forward_inputs == [
        [[10, 11]],
        [[4, 6], [5, 0], [4, 6], [4, 6]],
    ]
    assert model.selected_logit_positions == [[1, 0, 1, 1]]


def test_openai_shared_prefix_padded_suffix_prefill_can_use_selected_graph(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_SKIPPED_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_TOTAL_TOKENS", "1")
    model = _SelectedGraphTokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert model.selected_graph_capture_flags == [True]
    assert model.selected_logit_positions == [[1, 0]]


def test_openai_shared_prefix_padded_suffix_selected_logits_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_PREFIX_CACHE_MIN_TOKENS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_SKIPPED_TOKENS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SELECTED_PADDED_SUFFIX_LOGITS_MIN_TOTAL_TOKENS", "1")
    model = _SelectedTokenEchoSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine._prefix_cache_entry = None

    steps = list(
        engine._generate_prompt_list_batch_steps(
            [[10, 11, 4, 6], [10, 11, 5]],
            max_tokens=1,
            temperature=0.0,
            broadcast_tensor_parallel=False,
        )
    )

    assert steps == [[6, 5]]
    assert model.selected_logit_positions == []


def test_openai_generation_cache_rows_seq_lens_updates_dense_lists_directly() -> None:
    class DenseLengthLayer:
        def __init__(self) -> None:
            self._seq_lens = [0, 0, 0, 0]
            self._uniform_seq_len = [0]
            self.set_seq_len_calls = 0

        def set_seq_len(self, seq_len: int) -> None:
            self.set_seq_len_calls += 1
            self._seq_lens = [int(seq_len) for _ in self._seq_lens]

        def _physical_row(self, row: int) -> int:
            return int(row)

    class DenseLengthCache:
        def __init__(self) -> None:
            self.layers = [DenseLengthLayer(), DenseLengthLayer()]

    cache = DenseLengthCache()

    _set_generation_cache_rows_seq_lens(cache, [0, 1, 2], [5, 6, 5])

    for layer in cache.layers:
        assert layer._seq_lens == [5, 6, 5, 0]
        assert layer._uniform_seq_len == [None]
        assert layer.set_seq_len_calls == 0


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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_TOKENS", "100")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_RATIO", "1.5")
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


def test_openai_short_output_padded_suffix_prefill_relaxes_bounded_padding() -> None:
    prefix_tokens = 111
    short_prompts = [[1 for _ in range(prefix_tokens + 17)] for _ in range(24)]
    long_prompts = [[1 for _ in range(prefix_tokens + 75)] for _ in range(40)]
    length_groups = [
        [(index, prompt) for index, prompt in enumerate(short_prompts)],
        [(index + len(short_prompts), prompt) for index, prompt in enumerate(long_prompts)],
    ]
    prompt_lengths = [len(prompt) for prompt in [*short_prompts, *long_prompts]]

    assert _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=prompt_lengths,
        max_tokens=82,
    )
    assert _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=prompt_lengths,
        max_tokens=256,
    )


def test_openai_short_output_padded_suffix_prefill_allows_bounded_high_waste(
    monkeypatch,
) -> None:
    prefix_tokens = 111
    short_prompts = [[1 for _ in range(prefix_tokens + 17)] for _ in range(32)]
    long_prompts = [[1 for _ in range(prefix_tokens + 96)] for _ in range(32)]
    length_groups = [
        [(index, prompt) for index, prompt in enumerate(short_prompts)],
        [(index + len(short_prompts), prompt) for index, prompt in enumerate(long_prompts)],
    ]
    prompt_lengths = [len(prompt) for prompt in [*short_prompts, *long_prompts]]

    assert _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=prompt_lengths,
        max_tokens=82,
    )
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_PADDING_TOKENS", "2048")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_MAX_PADDING_TOKENS", "100")
    assert not _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=prompt_lengths,
        max_tokens=82,
    )
    monkeypatch.delenv("TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_PADDING_TOKENS")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHORT_OUTPUT_PADDED_SUFFIX_MAX_PADDING_RATIO", "1.5")
    assert not _prefer_shared_prefix_padded_suffix_prefill(
        length_groups,
        prefix_tokens=prefix_tokens,
        prompt_lengths=prompt_lengths,
        max_tokens=82,
    )


def test_openai_shared_prefix_padded_suffix_length_bucket_is_bounded(monkeypatch) -> None:
    model = object()
    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKETS", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    assert _shared_prefix_padded_suffix_bucketed_length(
        model,
        device=torch.device("cuda"),
        prompt_count=64,
        max_suffix_len=67,
    ) == 80
    assert _shared_prefix_padded_suffix_bucketed_length(
        model,
        device=torch.device("cuda"),
        prompt_count=64,
        max_suffix_len=19,
    ) == 19

    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKET_MAX_EXTRA_TOKENS", "256")
    assert _shared_prefix_padded_suffix_bucketed_length(
        model,
        device=torch.device("cuda"),
        prompt_count=64,
        max_suffix_len=67,
    ) == 67

    monkeypatch.setenv("TORCHINFERNO_OPENAI_SHARED_PREFIX_PADDED_SUFFIX_LENGTH_BUCKETS", "0")
    assert _shared_prefix_padded_suffix_bucketed_length(
        model,
        device=torch.device("cuda"),
        prompt_count=64,
        max_suffix_len=67,
    ) == 67
    assert _shared_prefix_padded_suffix_bucketed_length(
        model,
        device=torch.device("cpu"),
        prompt_count=64,
        max_suffix_len=67,
    ) == 67


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
    assert len(model.forward_inputs) >= 2
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


def test_openai_cache_pool_keeps_graph_resident_cache_on_eviction(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", "2")

    class _ReleaseRecordingModel:
        def __init__(self) -> None:
            self.allocated: list[_WarmupShapeCache] = []
            self.released: list[object] = []
            self._decode_graphs: dict[tuple[int, int, int, int], object] = {}

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
    model._decode_graphs[(id(first), 1, 1, 0)] = types.SimpleNamespace(cache=first)
    third = engine._generation_cache(3, 8, model=model)

    assert model.allocated == [first, second, third]
    assert model.released == [second]
    assert list(engine._cache_pool.values()) == [first, third]


def test_openai_cache_pool_evicts_fewest_graph_refs_when_all_resident(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", "2")

    class _ReleaseRecordingModel:
        def __init__(self) -> None:
            self.released: list[object] = []
            self._decode_graphs: dict[tuple[int, int, int, int], object] = {}
            self._ragged_decode_graphs: dict[tuple[int, int, int, bool, int], object] = {}

        def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _WarmupShapeCache:
            del batch_size, max_seq_len, kwargs
            return _WarmupShapeCache()

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _ReleaseRecordingModel()
    engine = _cache_only_engine()
    engine.model = model

    first = engine._generation_cache(1, 8, model=model)
    second = engine._generation_cache(2, 8, model=model)
    model._decode_graphs[(id(first), 1, 1, 0)] = types.SimpleNamespace(cache=first)
    model._ragged_decode_graphs[(id(first), 1, 8, True, 0)] = types.SimpleNamespace(cache=first)
    model._decode_graphs[(id(second), 1, 1, 0)] = types.SimpleNamespace(cache=second)
    third = engine._generation_cache(3, 8, model=model)

    assert model.released == [second]
    assert list(engine._cache_pool.values()) == [first, third]


def test_openai_cache_pool_syncs_before_releasing_graph_cache(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_CACHE_POOL_MAX_ENTRIES", "1")
    sync_devices: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: sync_devices.append(torch.device(device)))

    class _ReleaseRecordingModel:
        def __init__(self) -> None:
            self.released: list[object] = []
            self._decode_graphs: dict[tuple[int, int, int, int], object] = {}

        def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _WarmupShapeCache:
            del batch_size, max_seq_len, kwargs
            return _WarmupShapeCache()

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _ReleaseRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")

    first = engine._generation_cache(1, 8, model=model)
    model._decode_graphs[(id(first), 1, 1, 0)] = types.SimpleNamespace(cache=first)
    engine._generation_cache(2, 8, model=model)

    assert model.released == [first]
    assert sync_devices == [torch.device("cuda")]


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


def test_llama_tp_ragged_decode_graph_replay_updates_indexed_rows() -> None:
    from torchinferno.models.llama3 import tensor_parallel as tp

    class _Cache:
        def __init__(self) -> None:
            self.layers = [types.SimpleNamespace(max_seq_len=16)]

    model = object.__new__(tp.Llama3TensorParallelForCausalLM)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.world_size = 1
    model.rotary_cos_cache = torch.arange(64, dtype=torch.float32).view(16, 4)
    model.rotary_sin_cache = model.rotary_cos_cache + 1000
    model._ragged_decode_logits_graphs = {}

    cache = _Cache()
    logits = torch.zeros((2, 1, 4), dtype=torch.float32)
    replay_calls: list[int] = []
    captured = types.SimpleNamespace(
        cache=cache,
        max_seq_len=cache.layers[0].max_seq_len,
        static_input_ids=torch.empty((2, 1), dtype=torch.long),
        static_cache_positions=torch.empty((2,), dtype=torch.long),
        static_row_indices=torch.empty((2,), dtype=torch.long),
        static_rotary_cos=torch.empty((2, 4), dtype=torch.float32),
        static_rotary_sin=torch.empty((2, 4), dtype=torch.float32),
        graph=types.SimpleNamespace(replay=lambda: replay_calls.append(1)),
        output_logits=logits,
    )
    model._ragged_decode_logits_graphs[(id(cache), 2, cache.layers[0].max_seq_len, True, 0)] = captured

    seq_lens = torch.tensor([3, 7, 11, 13], dtype=torch.long)
    first_rows = torch.tensor([0, 2], dtype=torch.long)
    assert model._run_ragged_decode_logits_graph(
        torch.tensor([[10], [20]], dtype=torch.long),
        cache,
        seq_lens=seq_lens,
        row_indices=first_rows,
        capture_on_miss=False,
    ) is logits
    assert captured.static_row_indices.tolist() == [0, 2]
    assert captured.static_cache_positions.tolist() == [3, 11]
    assert captured.static_input_ids.tolist() == [[10], [20]]
    assert torch.equal(captured.static_rotary_cos, model.rotary_cos_cache.index_select(0, torch.tensor([3, 11])))
    assert torch.equal(captured.static_rotary_sin, model.rotary_sin_cache.index_select(0, torch.tensor([3, 11])))

    second_rows = torch.tensor([3, 1], dtype=torch.long)
    assert model._run_ragged_decode_logits_graph(
        torch.tensor([[30], [40]], dtype=torch.long),
        cache,
        seq_lens=seq_lens,
        row_indices=second_rows,
        capture_on_miss=False,
    ) is logits
    assert captured.static_row_indices.tolist() == [3, 1]
    assert captured.static_cache_positions.tolist() == [13, 7]
    assert captured.static_input_ids.tolist() == [[30], [40]]
    assert torch.equal(captured.static_rotary_cos, model.rotary_cos_cache.index_select(0, torch.tensor([13, 7])))
    assert torch.equal(captured.static_rotary_sin, model.rotary_sin_cache.index_select(0, torch.tensor([13, 7])))
    assert replay_calls == [1, 1]


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


def test_openai_decode_profile_records_ragged_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_path = tmp_path / "decode-profile.jsonl"
    monkeypatch.setenv("TORCHINFERNO_OPENAI_DECODE_PROFILE_JSONL", str(profile_path))

    class _GraphModel:
        rank = 0

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
            return torch.tensor([9], dtype=torch.long)

    row_indices = torch.tensor([1], dtype=torch.long)
    token, _ = _decode_next_token_ragged(
        _GraphModel(),
        torch.tensor([[1]], dtype=torch.long),
        object(),
        torch.tensor([4, 5], dtype=torch.long),
        row_indices,
        0.0,
    )

    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert token.tolist() == [9]
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "ragged_decode_step"
    assert record["mode"] == "token_graph"
    assert record["rank"] == 0
    assert record["batch_size"] == 1
    assert record["row_indices"] is True
    assert record["elapsed_ms"] >= 0.0


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


def test_openai_tp_shared_prefix_ragged_graph_respects_explicit_max_tokens(monkeypatch) -> None:
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", "128")
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


def test_openai_tp_shared_prefix_ragged_graph_allowed_for_overprovisioned_max_tokens(monkeypatch) -> None:
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", "1")
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", raising=False)
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
            max_tokens=400,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [False]


def test_openai_tp_shared_prefix_ragged_static_buckets_default_off_for_medium_long_generations(monkeypatch) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.ragged_graph_disabled: list[bool] = []
            self.released: list[object] = []

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

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKETS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MAX_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    cache = types.SimpleNamespace()

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=80,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [False]
    assert model.released == []
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets") is False
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released") is True
    assert _disable_tp_shared_prefix_ragged_static_buckets(model, max_tokens=80)


def test_openai_tp_shared_prefix_ragged_static_buckets_default_on_for_large_overprovisioned_generations(
    monkeypatch,
) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.ragged_graph_disabled: list[bool] = []
            self.released: list[object] = []

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

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MAX_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    cache = types.SimpleNamespace()

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=256,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [False]
    assert model.released == []
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets") is True
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released") is False
    assert not _disable_tp_shared_prefix_ragged_static_buckets(model, max_tokens=256)


def test_openai_tp_shared_prefix_ragged_static_buckets_require_initialized_physical_rows(
    monkeypatch,
) -> None:
    class _BatchRecordingModel:
        def __init__(self) -> None:
            self.seen: list[tuple[int, list[int] | None]] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del cache, seq_lens
            rows = None if row_indices is None else [int(row) for row in row_indices.cpu().tolist()]
            self.seen.append((int(input_ids.size(0)), rows))
            logits = torch.zeros(input_ids.size(0), 1, 8)
            logits[..., 3] = 1.0
            return logits

    model = _BatchRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MAX_TOKENS", raising=False)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    def make_cache(*, initialized: bool) -> object:
        cache = types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(
                    keys=torch.empty(64, 1, 16, 1),
                    values=torch.empty(64, 1, 16, 1),
                    batch_size=64,
                )
            ]
        )
        if initialized:
            setattr(cache, "_torchinferno_physical_rows_initialized", True)
        return cache

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=make_cache(initialized=False),
            active=[True] * 57,
            prompt_lengths=[2] * 57,
            max_tokens=256,
            next_tokens=[1] * 57,
            temperature=0.0,
            row_max_tokens=[2] * 57,
        )
    )

    assert steps == [[3] * 57]
    assert model.seen[-1][0] == 57
    assert model.seen[-1][1] == list(range(57))

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=make_cache(initialized=True),
            active=[True] * 57,
            prompt_lengths=[2] * 57,
            max_tokens=256,
            next_tokens=[1] * 57,
            temperature=0.0,
            row_max_tokens=[2] * 57,
        )
    )

    assert steps == [[3] * 57]
    assert model.seen[-1][0] == 64
    assert model.seen[-1][1] == list(range(64))


def test_openai_tp_shared_prefix_ragged_static_buckets_can_disable_for_long_generations(monkeypatch) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.ragged_graph_disabled: list[bool] = []
            self.released: list[object] = []

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

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS", "80")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    cache = types.SimpleNamespace()

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=80,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [False]
    assert model.released == []
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets") is False
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released") is True
    assert _disable_tp_shared_prefix_ragged_static_buckets(model, max_tokens=80)


def test_openai_tp_shared_prefix_ragged_static_buckets_drop_after_step_limit(monkeypatch) -> None:
    class _GraphFlagRecordingModel:
        def __init__(self) -> None:
            self.static_modes: list[bool] = []
            self.released: list[object] = []

        def decode_ragged_logits(
            self,
            input_ids: torch.Tensor,
            cache: object,
            *,
            seq_lens: torch.Tensor,
            row_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del seq_lens, row_indices
            self.static_modes.append(
                bool(getattr(cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets", False))
            )
            logits = torch.zeros(input_ids.size(0), 1, 8)
            logits[..., 3] = 1.0
            return logits

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _GraphFlagRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKETS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_DISABLE_MIN_TOKENS", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_STATIC_BUCKET_MAX_STEPS", "2")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )
    cache = types.SimpleNamespace()

    steps = list(
        engine._decode_shared_prefix_prompt_list_ragged(
            cache=cache,
            active=[True, True],
            prompt_lengths=[2, 3],
            max_tokens=3,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[3, 3],
        )
    )

    assert steps == [[3, 3], [3, 3]]
    assert model.static_modes == [True, False]
    assert model.released == [cache]
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets") is False
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released") is True


def test_openai_static_bucket_release_syncs_graph_cache(monkeypatch) -> None:
    sync_devices: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: sync_devices.append(torch.device(device)))

    class _ReleaseRecordingModel:
        def __init__(self) -> None:
            self.device = torch.device("cuda")
            self.released: list[object] = []
            self._ragged_decode_graphs: dict[tuple[int, int, int, bool, int], object] = {}

        def release_decode_graphs_for_cache(self, cache: object) -> None:
            self.released.append(cache)

    model = _ReleaseRecordingModel()
    cache = types.SimpleNamespace(_torchinferno_shared_prefix_ragged_static_graph_buckets=True)
    model._ragged_decode_graphs[(id(cache), 1, 8, True, 0)] = types.SimpleNamespace(cache=cache)
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    _set_shared_prefix_ragged_static_graph_bucket_mode(model, cache, static_graph_buckets=False)

    assert model.released == [cache]
    assert sync_devices == [torch.device("cuda")]
    assert getattr(cache, "_torchinferno_shared_prefix_ragged_nonstatic_graphs_released") is True


def test_openai_tp_shared_prefix_ragged_graph_can_disable_for_configured_long_generations(monkeypatch) -> None:
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHARED_PREFIX_RAGGED_CUDAGRAPH_DISABLE_MIN_TOKENS", "512")
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
            max_tokens=1024,
            next_tokens=[1, 1],
            temperature=0.0,
            row_max_tokens=[2, 2],
        )
    )

    assert steps == [[3, 3]]
    assert model.ragged_graph_disabled == [True]


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


def test_openai_shared_prefix_ragged_decode_row_plan_skips_inactive_rows(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION", raising=False)

    plan = _shared_prefix_ragged_decode_row_plan(
        [False, True, True, True],
        active_indices=[1, 2, 3],
        step=1,
        cache_batch_size=4,
        force_row_indices=False,
        static_graph_buckets=False,
    )

    assert plan.active_indices == (1, 2, 3)
    assert plan.decode_indices == (1, 2, 3)
    assert plan.row_indices == (1, 2, 3)
    assert plan.advance_row_indices is None


def test_openai_shared_prefix_ragged_decode_row_plan_uses_full_batch(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION", "0.5")

    plan = _shared_prefix_ragged_decode_row_plan(
        [False, True, True, True],
        active_indices=[1, 2, 3],
        step=1,
        cache_batch_size=4,
        force_row_indices=False,
        static_graph_buckets=False,
    )

    assert plan.decode_indices == (0, 1, 2, 3)
    assert plan.row_indices is None
    assert plan.advance_row_indices is None


def test_openai_shared_prefix_ragged_decode_row_plan_pads_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", "1")

    plan = _shared_prefix_ragged_decode_row_plan(
        [True, True, True, False, False, False, False, False],
        active_indices=[0, 1, 2],
        step=1,
        cache_batch_size=8,
        force_row_indices=False,
        static_graph_buckets=False,
    )

    assert plan.decode_indices == (0, 1, 2, 3)
    assert plan.row_indices == (0, 1, 2, 3)
    assert plan.advance_row_indices == (0, 1, 2)


def test_openai_shared_prefix_ragged_decode_row_plan_can_prefer_full_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", "1")

    plan = _shared_prefix_ragged_decode_row_plan(
        [True, True, True, False, False, False, False, False],
        active_indices=[0, 1, 2],
        step=1,
        cache_batch_size=8,
        force_row_indices=False,
        static_graph_buckets=True,
        prefer_full_bucket=True,
    )

    assert plan.decode_indices == (0, 1, 2, 3, 4, 5, 6, 7)
    assert plan.row_indices == (0, 1, 2, 3, 4, 5, 6, 7)
    assert plan.advance_row_indices == (0, 1, 2)


def test_openai_shared_prefix_ragged_decode_step_updates_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", str(tmp_path / "queue.jsonl"))
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(4, 5)
    active = [True, True, False, False]
    generated_tokens = [1, 1, 0, 0]
    seq_lens = torch.tensor([2, 2, 2, 2], dtype=torch.long)
    next_token_tensor = torch.tensor([2, 2, 0, 0], dtype=torch.long)

    result = engine._decode_shared_prefix_prompt_list_ragged_step(
        cache=cache,
        active=active,
        per_row_limits=[3, 2, 1, 1],
        generated_tokens=generated_tokens,
        seq_lens=seq_lens,
        next_token_tensor=next_token_tensor,
        step=1,
        cache_batch_size=4,
        temperature=0.0,
        static_graph_buckets=False,
    )

    assert result is not None
    returned_cache, step_tokens = result
    assert returned_cache is cache
    assert step_tokens == [3, 3, None, None]
    assert active == [True, False, False, False]
    assert generated_tokens == [2, 2, 0, 0]
    assert seq_lens.tolist() == [3, 3, 2, 2]
    assert next_token_tensor.tolist() == [3, 3, 0, 0]
    assert model.ragged_calls == [([[2], [2]], [2, 2, 2, 2], [0, 1])]
    profile_extra = engine._stream_group_profile_extra
    assert profile_extra["shared_prefix_ragged_prepare_ms"] >= 0.0
    assert profile_extra["shared_prefix_ragged_model_ms"] >= 0.0
    assert profile_extra["shared_prefix_ragged_cpu_tokens_ms"] >= 0.0
    assert profile_extra["shared_prefix_ragged_state_update_ms"] >= 0.0


def test_openai_ragged_decode_skips_inactive_rows_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_FULL_BATCH_MIN_ACTIVE_FRACTION", raising=False)
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(16, 5)

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
    cache = model.allocate_cache(16, 5)

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


def test_openai_tp_ragged_decode_pads_to_static_cache_bucket(monkeypatch) -> None:
    model = _RaggedSharedPrefixRecordingModel()
    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    cache = model.allocate_cache(16, 5)
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
            cache=cache,
            active=[True for _ in range(9)],
            prompt_lengths=[2 for _ in range(9)],
            max_tokens=3,
            next_tokens=[2 for _ in range(9)],
            temperature=0.0,
            row_max_tokens=[3 for _ in range(9)],
        )
    )

    assert steps == [[3 for _ in range(9)], [4 for _ in range(9)]]
    assert model.ragged_calls[0] == (
        [[2] for _ in range(9)] + [[0] for _ in range(7)],
        [2 for _ in range(16)],
        list(range(16)),
    )
    assert model.ragged_calls[1] == (
        [[3] for _ in range(9)] + [[0] for _ in range(7)],
        [3 for _ in range(9)] + [2 for _ in range(7)],
        list(range(16)),
    )


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


def test_openai_tensor_parallel_generation_cache_uses_batch_buckets(monkeypatch) -> None:
    engine = _cache_only_engine()
    model = _BatchRecordingModel()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_world_size",
        lambda candidate: 8 if candidate is model else 1,
    )

    dense = engine._generation_cache(1, 17, model=model)
    small = engine._generation_cache(
        7,
        17,
        model=model,
        batch_capacity=_generation_cache_batch_capacity(model, 7),
    )
    large = engine._generation_cache(
        56,
        129,
        model=model,
        batch_capacity=_generation_cache_batch_capacity(model, 56),
    )

    assert model.allocated_shapes == [(1, 32), (64, 32), (64, 256)]
    assert list(engine._cache_pool) == [
        (1, 32, "dense", 16, "cpu"),
        (64, 32, "dense", 16, "cpu"),
        (64, 256, "dense", 16, "cpu"),
    ]
    assert dense in engine._cache_pool.values()
    assert small in engine._cache_pool.values()
    assert large in engine._cache_pool.values()


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


def test_openai_cache_copy_state_rows_padded_bulk_copies_llama_tp_rows() -> None:
    source_a = _llama_tp_cache(batch_size=2, max_seq_len=8)
    source_a_keys = torch.arange(20, dtype=torch.float32).reshape(2, 1, 5, 2)
    source_a.layers[0].append(source_a_keys, source_a_keys + 100)
    source_b = _llama_tp_cache(batch_size=1, max_seq_len=8)
    source_b_keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + 1000
    source_b.layers[0].append(source_b_keys, source_b_keys + 100)
    source_c = _llama_tp_cache(batch_size=1, max_seq_len=8)
    source_c_keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2) + 2000
    source_c.layers[0].append(source_c_keys, source_c_keys + 100)
    target = _llama_tp_cache(batch_size=4, max_seq_len=8)

    copied = _copy_generation_cache_state_rows_padded(
        [
            {"cache": source_a, "indices": [2, 0]},
            {"cache": source_b, "indices": [3]},
            {"cache": source_c, "indices": [1]},
        ],
        target,
        prompt_lengths=[3, 2, 5, 4],
        prompt_count=4,
    )

    assert copied
    assert target.for_rows((0,)).seq_len == 3
    assert target.for_rows((1,)).seq_len == 2
    assert target.for_rows((2,)).seq_len == 5
    assert target.for_rows((3,)).seq_len == 4
    torch.testing.assert_close(target.layers[0].keys[2:3, :, :5, :], source_a_keys[0:1])
    torch.testing.assert_close(target.layers[0].keys[0:1, :, :3, :], source_a_keys[1:2, :, :3, :])
    torch.testing.assert_close(target.layers[0].keys[3:4, :, :4, :], source_b_keys)
    torch.testing.assert_close(target.layers[0].keys[1:2, :, :2, :], source_c_keys)


def test_openai_cache_copy_state_rows_padded_requires_complete_target_rows() -> None:
    source = _llama_tp_cache(batch_size=1, max_seq_len=8)
    source.layers[0].append(torch.ones((1, 1, 3, 2)), torch.ones((1, 1, 3, 2)))
    target = _llama_tp_cache(batch_size=2, max_seq_len=8)

    copied = _copy_generation_cache_state_rows_padded(
        [{"cache": source, "indices": [0]}],
        target,
        prompt_lengths=[3, 3],
        prompt_count=2,
    )

    assert not copied
    assert target.for_rows((0,)).seq_len == 0
    assert target.for_rows((1,)).seq_len == 0


class _RowTargetPrefillModel:
    config = type("Config", (), {"vocab_size": 32})()

    def __init__(self) -> None:
        self.forward_calls: list[tuple[tuple[int, ...], list[list[int]]]] = []
        self.ragged_calls: list[tuple[list[list[int]], list[int], list[int] | None]] = []

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> Llama3TensorParallelCache:
        del kwargs
        return _llama_tp_cache(batch_size=batch_size, max_seq_len=max_seq_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Llama3TensorParallelCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
    ):
        del use_cache, return_last_logits_only
        rows = cache.layers[0]._selected_rows(input_ids.size(0))
        self.forward_calls.append((rows, [[int(token_id) for token_id in row] for row in input_ids.tolist()]))
        keys = input_ids.to(torch.float32).view(input_ids.size(0), 1, input_ids.size(1), 1)
        keys = keys.expand(-1, -1, -1, 2).contiguous()
        cache.layers[0].append(keys, keys + 100)
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
        for row_index, row in enumerate(input_ids.tolist()):
            for token_index, token_id in enumerate(row):
                logits[row_index, token_index, min(int(token_id) + 1, self.config.vocab_size - 1)] = 1.0
        return logits, cache

    def decode_ragged_logits(
        self,
        input_ids: torch.Tensor,
        cache: Llama3TensorParallelCache,
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
        token = 20 + len(self.ragged_calls)
        logits = torch.zeros(input_ids.size(0), 1, self.config.vocab_size)
        logits[..., token] = 1.0
        return logits


class _PagedRowTargetPrefillModel(_RowTargetPrefillModel):
    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> Llama3TensorParallelCache:
        page_size = int(kwargs.get("page_size", 2))
        return _llama_tp_cache(batch_size=batch_size, max_seq_len=max_seq_len, cache_backend="paged", page_size=page_size)


def test_openai_padded_suffix_prefill_can_target_cache_rows() -> None:

    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    target_cache = _llama_tp_cache(batch_size=4, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 2, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)

    result = engine._prefill_shared_prefix_prompt_list_padded_suffix_rows(
        [[10, 11, 12], [10, 11, 13, 14]],
        prefix_cache=prefix_cache,
        target_cache=target_cache,
        target_rows=[2, 0],
        prefix_tokens=2,
        max_tokens=3,
        temperature=0.0,
        row_max_tokens=[3, 1],
        model=model,
    )

    assert result == ([13, 15], [True, False])
    assert model.forward_calls == [((2, 0), [[12, 0], [13, 14]])]
    assert target_cache.for_rows((0,)).seq_len == 4
    assert target_cache.for_rows((1,)).seq_len == 0
    assert target_cache.for_rows((2,)).seq_len == 3
    assert target_cache.for_rows((3,)).seq_len == 0
    torch.testing.assert_close(target_cache.layers[0].keys[2:3, :, :2, :], prefix_keys)
    torch.testing.assert_close(target_cache.layers[0].keys[0:1, :, :2, :], prefix_keys)
    torch.testing.assert_close(
        target_cache.layers[0].keys[2, :, 2:4, :],
        torch.tensor([[[12.0, 12.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        target_cache.layers[0].keys[0, :, 2:4, :],
        torch.tensor([[[13.0, 13.0], [14.0, 14.0]]]),
    )


def test_openai_padded_suffix_prefill_truncates_paged_cache_rows() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _PagedRowTargetPrefillModel()
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8, cache_backend="paged", page_size=2)
    target_cache = _llama_tp_cache(batch_size=4, max_seq_len=8, cache_backend="paged", page_size=2)
    prefix_keys = torch.full((1, 1, 2, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)

    result = engine._prefill_shared_prefix_prompt_list_padded_suffix_rows(
        [[10, 11, 12], [10, 11, 13, 14]],
        prefix_cache=prefix_cache,
        target_cache=target_cache,
        target_rows=[2, 0],
        prefix_tokens=2,
        max_tokens=3,
        temperature=0.0,
        row_max_tokens=[3, 3],
        model=model,
    )

    assert result == ([13, 15], [True, True])
    assert target_cache.for_rows((0,)).seq_len == 4
    assert target_cache.for_rows((2,)).seq_len == 3
    row2_keys, _row2_values = target_cache.layers[0].materialize_row(2)
    row0_keys, _row0_values = target_cache.layers[0].materialize_row(0)
    torch.testing.assert_close(row2_keys, torch.tensor([[[5.0, 5.0], [5.0, 5.0], [12.0, 12.0]]]))
    torch.testing.assert_close(
        row0_keys,
        torch.tensor([[[5.0, 5.0], [5.0, 5.0], [13.0, 13.0], [14.0, 14.0]]]),
    )


def test_openai_persistent_prompt_list_payload_prefills_groups_into_target_rows() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    target_cache = _llama_tp_cache(batch_size=4, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    payload = {
        "op": "persistent_prompt_list_step",
        "prefill": [
            {
                "request_id": "a",
                "row": 2,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [4],
                "prefill_tokens": 1,
            },
            {
                "request_id": "b",
                "row": 0,
                "prompt": [1, 2, 3, 5, 6],
                "max_tokens": 1,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [5, 6],
                "prefill_tokens": 2,
            },
        ],
        "prefill_groups": [
            {
                "request_ids": ["a", "b"],
                "rows": [2, 0],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1, 2],
            },
        ],
    }

    result = engine._prefill_persistent_prompt_list_payload_groups(
        payload,
        target_cache=target_cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        model=model,
        temperature=0.0,
    )

    assert result == {"a": (5, True), "b": (7, False)}
    assert model.forward_calls == [((2, 0), [[4, 0], [5, 6]])]
    assert target_cache.for_rows((0,)).seq_len == 5
    assert target_cache.for_rows((1,)).seq_len == 0
    assert target_cache.for_rows((2,)).seq_len == 4
    assert target_cache.for_rows((3,)).seq_len == 0
    torch.testing.assert_close(target_cache.layers[0].keys[2:3, :, :3, :], prefix_keys)
    torch.testing.assert_close(target_cache.layers[0].keys[0:1, :, :3, :], prefix_keys)
    torch.testing.assert_close(
        target_cache.layers[0].keys[2, :, 3:5, :],
        torch.tensor([[[4.0, 4.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        target_cache.layers[0].keys[0, :, 3:5, :],
        torch.tensor([[[5.0, 5.0], [6.0, 6.0]]]),
    )


def test_openai_persistent_prompt_list_step_executor_decodes_and_prefills_rows() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=4, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False, True, False, False],
        per_row_limits=[0, 2, 0, 0],
        generated_tokens=[0, 1, 0, 0],
        seq_lens=torch.tensor([0, 3, 0, 0], dtype=torch.long),
        next_token_tensor=torch.tensor([0, 9, 0, 0], dtype=torch.long),
        row_request_ids=[None, "old", None, None],
        cache_batch_size=4,
    )
    payload = {
        "op": "persistent_prompt_list_step",
        "step": 1,
        "decode_request_ids": ["old"],
        "decode_rows": [1],
        "prefill": [
            {
                "request_id": "new",
                "row": 2,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 2,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [4],
                "prefill_tokens": 1,
            },
        ],
        "prefill_groups": [
            {
                "request_ids": ["new"],
                "rows": [2],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1],
            },
        ],
        "finished_after_prefill": [],
    }

    result = engine._execute_persistent_prompt_list_step_payload(
        payload,
        state,
        temperature=0.0,
    )

    assert result.decode_tokens == {"old": 21}
    assert result.prefill_tokens == {"new": 5}
    assert result.finished_request_ids == ("old",)
    assert state.active == [False, False, True, False]
    assert state.row_request_ids == [None, None, "new", None]
    assert state.per_row_limits == [0, 2, 2, 0]
    assert state.seq_lens.tolist() == [0, 4, 4, 0]
    assert state.next_token_tensor.tolist() == [0, 21, 5, 0]
    assert model.ragged_calls == [([[9]], [0, 3, 0, 0], [1])]
    assert model.forward_calls == [((2,), [[4]])]
    torch.testing.assert_close(cache.layers[0].keys[2:3, :, :3, :], prefix_keys)


def test_openai_persistent_prompt_list_step_state_start_prefills_prefix() -> None:
    engine = _cache_only_engine()
    model = _RowTargetPrefillModel()
    engine.model = model

    state = engine._start_persistent_prompt_list_step_state(
        prefix=[1, 2, 3],
        cache_batch_size=2,
        max_seq_len=8,
        temperature=0.0,
    )

    assert engine._persistent_prompt_list_step_state is state
    assert state.cache_batch_size == 2
    assert state.active == [False, False]
    assert state.per_row_limits == [0, 0]
    assert state.seq_lens.tolist() == [0, 0]
    assert state.row_request_ids == [None, None]
    assert tuple(state.prefix_caches) == ((1, 2, 3),)
    assert state.prefix_caches[(1, 2, 3)].for_rows((0,)).seq_len == 3
    assert state.cache.layers[0].batch_size == 2
    assert state.cache.layers[0].max_seq_len == 8
    assert model.forward_calls == [((0,), [[1, 2, 3]])]

    engine._close_persistent_prompt_list_step_state()

    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None


def test_openai_persistent_prompt_list_step_handler_uses_installed_state() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=2, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False, False],
        per_row_limits=[0, 0],
        generated_tokens=[0, 0],
        seq_lens=torch.zeros(2, dtype=torch.long),
        next_token_tensor=torch.zeros(2, dtype=torch.long),
        row_request_ids=[None, None],
        cache_batch_size=2,
    )
    engine._persistent_prompt_list_step_state = state
    payload = {
        "op": "persistent_prompt_list_step",
        "step": 0,
        "temperature": 0.0,
        "decode_request_ids": [],
        "decode_rows": [],
        "prefill": [
            {
                "request_id": "new",
                "row": 1,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 2,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [4],
                "prefill_tokens": 1,
            },
        ],
        "prefill_groups": [
            {
                "request_ids": ["new"],
                "rows": [1],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1],
            },
        ],
        "finished_after_prefill": [],
    }

    result = engine._handle_persistent_prompt_list_step_payload(payload)

    assert result.prefill_tokens == {"new": 5}
    assert result.decode_tokens == {}
    assert result.finished_request_ids == ()
    assert engine._persistent_prompt_list_step_last_result is result
    assert state.active == [False, True]
    assert state.row_request_ids == [None, "new"]
    assert state.seq_lens.tolist() == [0, 4]
    assert model.forward_calls == [((1,), [[4]])]


def test_tensor_parallel_worker_loop_runs_persistent_prompt_list_lifecycle(monkeypatch) -> None:
    import torch.distributed as dist

    commands: list[dict[str, object]] = [
        {
            "op": "persistent_prompt_list_start",
            "prefix": [1, 2, 3],
            "cache_batch_size": 2,
            "max_seq_len": 8,
            "temperature": 0.0,
        },
        {
            "op": "persistent_prompt_list_step",
            "step": 0,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": "new",
                    "row": 1,
                    "prompt": [1, 2, 3, 4],
                    "max_tokens": 2,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [4],
                    "prefill_tokens": 1,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": ["new"],
                    "rows": [1],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [1],
                },
            ],
            "finished_after_prefill": [],
        },
        {"op": "persistent_prompt_list_close"},
        {"op": "stop"},
    ]

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        payload[0] = commands.pop(0)

    scope_events: list[tuple[str, object]] = []

    class Scope:
        def __enter__(self) -> None:
            scope_events.append(("enter", None))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            scope_events.append(("exit", None))

    def symm_scope(model_arg: object, device_arg: torch.device, **kwargs: object) -> Scope:
        scope_events.append(("create", (model_arg, device_arg, kwargs)))
        return Scope()

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope", symm_scope)

    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model

    _tensor_parallel_worker_loop(engine)

    assert commands == []
    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None
    assert model.forward_calls == [((0,), [[1, 2, 3]]), ((1,), [[4]])]
    assert scope_events == [
        ("create", (model, torch.device("cpu"), {"max_tokens": 5, "temperature": 0.0})),
        ("enter", None),
        ("exit", None),
    ]


def test_tensor_parallel_worker_loop_exits_persistent_prompt_list_scope_on_stop(monkeypatch) -> None:
    import torch.distributed as dist

    commands: list[dict[str, object]] = [
        {
            "op": "persistent_prompt_list_start",
            "prefix": [1, 2],
            "cache_batch_size": 1,
            "max_seq_len": 6,
            "temperature": 0.0,
        },
        {"op": "stop"},
    ]
    scope_events: list[str] = []

    class Scope:
        def __enter__(self) -> None:
            scope_events.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            scope_events.append("exit")

    def broadcast_object_list(payload: list[object], *, src: int) -> None:
        del src
        payload[0] = commands.pop(0)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: Scope(),
    )

    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()

    _tensor_parallel_worker_loop(engine)

    assert commands == []
    assert scope_events == ["enter", "exit"]


def test_tensor_parallel_worker_loop_receives_persistent_prompt_list_tensor_lifecycle(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_PROMPT_LIST_PERSISTENT_START, 0, 0, 0, 5, 3, 8, 4, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([_TP_COMMAND_PROMPT_LIST_PERSISTENT_STEP, 4, 1, 1, 4, 0, 6, 2, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([[0, 0]], dtype=torch.long),
        torch.tensor([[1, 1, 2, 3, 1, 3]], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([[1, 2, 3, 5]], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.tensor([_TP_COMMAND_PROMPT_LIST_PERSISTENT_DECODE_RUN, 5, 2, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([_TP_COMMAND_PROMPT_LIST_PERSISTENT_CLOSE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = model
            self.events: list[tuple[str, dict[str, object]]] = []

        def _start_persistent_prompt_list_step_state(self, **kwargs: object) -> None:
            self.events.append(("start", dict(kwargs)))

        def _handle_persistent_prompt_list_step_payload(self, command: dict[str, object]) -> None:
            self.events.append(("step", dict(command)))

        def _handle_persistent_prompt_list_decode_run_payload(self, command: dict[str, object]) -> None:
            self.events.append(("decode_run", dict(command)))

        def _close_persistent_prompt_list_step_state(self) -> None:
            self.events.append(("close", {}))

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        tensor.copy_(payloads.pop(0).to(device=tensor.device, dtype=tensor.dtype))

    scope_events: list[tuple[str, object]] = []

    class Scope:
        def __enter__(self) -> None:
            scope_events.append(("enter", None))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            scope_events.append(("exit", None))

    def symm_scope(model_arg: object, device_arg: torch.device, **kwargs: object) -> Scope:
        scope_events.append(("create", (model_arg, device_arg, kwargs)))
        return Scope()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope", symm_scope)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert payloads == []
    assert scope_events == [
        ("create", (model, torch.device("cpu"), {"max_tokens": 5, "temperature": 0.25})),
        ("enter", None),
        ("exit", None),
    ]
    assert engine.events == [
        (
            "start",
            {
                "prefix": [1, 2, 3],
                "cache_batch_size": 4,
                "max_seq_len": 8,
                "temperature": 0.25,
                "max_tokens": 5,
            },
        ),
        (
            "step",
            {
                "op": "persistent_prompt_list_step",
                "step": 4,
                "decode_request_ids": ["0"],
                "decode_rows": [0],
                "prefill": [
                    {
                        "request_id": "1",
                        "row": 1,
                        "prompt": [1, 2, 3, 5],
                        "max_tokens": 2,
                        "prefix_hit_tokens": 3,
                        "prefix": [1, 2, 3],
                        "suffix": [5],
                        "prefill_tokens": 1,
                    }
                ],
                "prefill_groups": [
                    {
                        "request_ids": ["1"],
                        "rows": [1],
                        "prefix_hit_tokens": 3,
                        "prefix": [1, 2, 3],
                        "suffix_tokens": [1],
                    }
                ],
                "finished_after_prefill": [],
                "temperature": 0.25,
            },
        ),
        (
            "decode_run",
            {
                "op": "persistent_prompt_list_decode_run",
                "start_step": 5,
                "step_count": 2,
                "temperature": 0.25,
                "static_graph_buckets": True,
            },
        ),
        ("close", {}),
    ]


def test_openai_persistent_prompt_list_step_executor_reuses_finished_row() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False],
        per_row_limits=[0],
        generated_tokens=[0],
        seq_lens=torch.zeros(1, dtype=torch.long),
        next_token_tensor=torch.zeros(1, dtype=torch.long),
        row_request_ids=[None],
        cache_batch_size=1,
    )

    first = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 0,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": "a",
                    "row": 0,
                    "prompt": [1, 2, 3, 4],
                    "max_tokens": 2,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [4],
                    "prefill_tokens": 1,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": ["a"],
                    "rows": [0],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [1],
                },
            ],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
    )
    second = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 1,
            "decode_request_ids": ["a"],
            "decode_rows": [0],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
    )
    third = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 2,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": "b",
                    "row": 0,
                    "prompt": [1, 2, 3, 5, 6],
                    "max_tokens": 3,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [5, 6],
                    "prefill_tokens": 2,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": ["b"],
                    "rows": [0],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [2],
                },
            ],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
    )

    assert first.prefill_tokens == {"a": 5}
    assert second.decode_tokens == {"a": 21}
    assert second.finished_request_ids == ("a",)
    assert third.prefill_tokens == {"b": 7}
    assert state.active == [True]
    assert state.row_request_ids == ["b"]
    assert state.per_row_limits == [3]
    assert state.seq_lens.tolist() == [5]
    assert state.next_token_tensor.tolist() == [7]
    assert cache.for_rows((0,)).seq_len == 5
    assert model.forward_calls == [((0,), [[4]]), ((0,), [[5, 6]])]
    assert model.ragged_calls == [([[5]], [4], None)]
    torch.testing.assert_close(cache.layers[0].keys[0:1, :, :3, :], prefix_keys)
    torch.testing.assert_close(
        cache.layers[0].keys[0, :, 3:5, :],
        torch.tensor([[[5.0, 5.0], [6.0, 6.0]]]),
    )

def test_openai_persistent_executor_uses_per_row_generated_counts_for_late_refill() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=10)
    cache = _llama_tp_cache(batch_size=1, max_seq_len=10)
    prefix_cache.for_rows((0,)).layers[0].append(
        torch.full((1, 1, 3, 2), 5.0),
        torch.full((1, 1, 3, 2), 105.0),
    )
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False],
        per_row_limits=[0],
        generated_tokens=[0],
        seq_lens=torch.zeros(1, dtype=torch.long),
        next_token_tensor=torch.zeros(1, dtype=torch.long),
        row_request_ids=[None],
        cache_batch_size=1,
    )

    def prefill_payload(request_id: str, *, step: int, prompt: list[int], max_tokens: int) -> dict[str, object]:
        return {
            "op": "persistent_prompt_list_step",
            "step": step,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": request_id,
                    "row": 0,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": prompt[3:],
                    "prefill_tokens": len(prompt) - 3,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": [request_id],
                    "rows": [0],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [len(prompt) - 3],
                },
            ],
            "finished_after_prefill": [],
        }

    def decode_payload(request_id: str, *, step: int) -> dict[str, object]:
        return {
            "op": "persistent_prompt_list_step",
            "step": step,
            "decode_request_ids": [request_id],
            "decode_rows": [0],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        }

    engine._execute_persistent_prompt_list_step_payload(
        prefill_payload("a", step=0, prompt=[1, 2, 3, 4], max_tokens=2),
        state,
        temperature=0.0,
    )
    first_finish = engine._execute_persistent_prompt_list_step_payload(
        decode_payload("a", step=1),
        state,
        temperature=0.0,
    )
    late_prefill = engine._execute_persistent_prompt_list_step_payload(
        prefill_payload("b", step=20, prompt=[1, 2, 3, 5], max_tokens=4),
        state,
        temperature=0.0,
    )
    late_first_decode = engine._execute_persistent_prompt_list_step_payload(
        decode_payload("b", step=21),
        state,
        temperature=0.0,
    )
    late_second_decode = engine._execute_persistent_prompt_list_step_payload(
        decode_payload("b", step=22),
        state,
        temperature=0.0,
    )
    late_finish = engine._execute_persistent_prompt_list_step_payload(
        decode_payload("b", step=23),
        state,
        temperature=0.0,
    )

    assert first_finish.finished_request_ids == ("a",)
    assert late_prefill.prefill_tokens == {"b": 6}
    assert late_first_decode.decode_tokens == {"b": 22}
    assert late_first_decode.finished_request_ids == ()
    assert late_second_decode.decode_tokens == {"b": 23}
    assert late_second_decode.finished_request_ids == ()
    assert late_finish.decode_tokens == {"b": 24}
    assert late_finish.finished_request_ids == ("b",)
    assert state.generated_tokens == [4]
    assert state.active == [False]
    assert state.row_request_ids == [None]


def test_openai_persistent_scheduler_executor_handles_budgeted_completion_order() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=2, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False, False],
        per_row_limits=[0, 0],
        generated_tokens=[0, 0],
        seq_lens=torch.zeros(2, dtype=torch.long),
        next_token_tensor=torch.zeros(2, dtype=torch.long),
        row_request_ids=[None, None],
        cache_batch_size=2,
    )
    group = [
        _QueuedGeneration([1, 2, 3, 4], 2, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 5], 1, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 6, 7], 2, 0.0, True, queue.Queue()),
    ]
    scheduler = _persistent_prompt_list_scheduler_for_group(
        group,
        max_active_rows=2,
        prefill_token_budget=1,
        prefix_tokens=3,
    )
    finished_request_ids: tuple[str, ...] = ()
    emitted_steps: list[list[int | None]] = []
    emitted_by_request: dict[str, list[int]] = {str(index): [] for index in range(len(group))}
    finished_seen: list[tuple[str, ...]] = []

    while scheduler.has_work() or finished_request_ids:
        plan = scheduler.step(finished_request_ids=finished_request_ids)
        finished_request_ids = ()
        if not plan.decode_request_ids and not plan.prefill_admissions:
            continue
        result = engine._execute_persistent_prompt_list_step_payload(
            _persistent_prompt_list_step_payload(plan, group),
            state,
            temperature=0.0,
        )
        step_tokens: list[int | None] = [None for _request in group]
        for request_id, token_id in result.decode_tokens.items():
            if token_id is not None:
                step_tokens[int(request_id)] = int(token_id)
                emitted_by_request[request_id].append(int(token_id))
        for request_id, token_id in result.prefill_tokens.items():
            if token_id is not None:
                step_tokens[int(request_id)] = int(token_id)
                emitted_by_request[request_id].append(int(token_id))
        emitted_steps.append(step_tokens)
        finished_request_ids = result.finished_request_ids
        if finished_request_ids:
            finished_seen.append(finished_request_ids)

    assert emitted_steps == [
        [5, None, None],
        [21, 6, None],
        [None, None, 8],
        [None, None, 22],
    ]
    assert emitted_by_request == {"0": [5, 21], "1": [6], "2": [8, 22]}
    assert finished_seen == [("0", "1"), ("2",)]
    assert state.active == [False, False]
    assert state.row_request_ids == [None, None]
    assert state.per_row_limits == [2, 1]
    assert state.seq_lens.tolist() == [6, 4]
    assert state.next_token_tensor.tolist() == [22, 6]
    assert model.forward_calls == [((0,), [[4]]), ((1,), [[5]]), ((0,), [[6, 7]])]
    assert model.ragged_calls == [
        ([[5]], [4], None),
        ([[8]], [5, 4], [0]),
    ]
    torch.testing.assert_close(cache.layers[0].keys[0:1, :, :3, :], prefix_keys)
    torch.testing.assert_close(
        cache.layers[0].keys[0, :, 3:5, :],
        torch.tensor([[[6.0, 6.0], [7.0, 7.0]]]),
    )


def test_openai_persistent_scheduler_executor_streams_refill_results_to_request_queues() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=2, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False, False],
        per_row_limits=[0, 0],
        generated_tokens=[0, 0],
        seq_lens=torch.zeros(2, dtype=torch.long),
        next_token_tensor=torch.zeros(2, dtype=torch.long),
        row_request_ids=[None, None],
        cache_batch_size=2,
    )
    response_queues = [queue.Queue() for _ in range(3)]
    group = [
        _QueuedGeneration([1, 2, 3, 4], 2, 0.0, True, response_queues[0]),
        _QueuedGeneration([1, 2, 3, 5], 1, 0.0, True, response_queues[1]),
        _QueuedGeneration([1, 2, 3, 6, 7], 2, 0.0, True, response_queues[2]),
    ]
    scheduler = _persistent_prompt_list_scheduler_for_group(
        group,
        max_active_rows=2,
        prefill_token_budget=1,
        prefix_tokens=3,
    )
    finished_request_ids: tuple[str, ...] = ()

    while scheduler.has_work() or finished_request_ids:
        plan = scheduler.step(finished_request_ids=finished_request_ids)
        finished_request_ids = ()
        if not plan.decode_request_ids and not plan.prefill_admissions:
            continue
        result = engine._execute_persistent_prompt_list_step_payload(
            _persistent_prompt_list_step_payload(plan, group),
            state,
            temperature=0.0,
        )
        for token_map in (result.decode_tokens, result.prefill_tokens):
            for request_id, token_id in token_map.items():
                if token_id is None:
                    continue
                request = group[int(request_id)]
                if not request.done:
                    request.responses.put(int(token_id))
        for request_id in result.finished_request_ids:
            request = group[int(request_id)]
            if not request.done:
                request.responses.put(_GenerationDone())
                request.done = True
        finished_request_ids = result.finished_request_ids

    assert _queue_items(response_queues[0]) == [5, 21, _GenerationDone()]
    assert _queue_items(response_queues[1]) == [6, _GenerationDone()]
    assert _queue_items(response_queues[2]) == [8, 22, _GenerationDone()]
    assert [request.done for request in group] == [True, True, True]
    assert state.active == [False, False]
    assert state.row_request_ids == [None, None]
    assert model.forward_calls == [((0,), [[4]]), ((1,), [[5]]), ((0,), [[6, 7]])]
    assert model.ragged_calls == [
        ([[5]], [4], None),
        ([[8]], [5, 4], [0]),
    ]


def test_openai_persistent_prompt_list_local_group_runner_streams_results() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    response_queues = [queue.Queue() for _ in range(3)]
    group = [
        _QueuedGeneration([1, 2, 3, 4], 2, 0.0, True, response_queues[0]),
        _QueuedGeneration([1, 2, 3, 5], 1, 0.0, True, response_queues[1]),
        _QueuedGeneration([1, 2, 3, 6, 7], 2, 0.0, True, response_queues[2]),
    ]

    stats = engine._run_persistent_prompt_list_group_local(
        group,
        max_active_rows=2,
        prefill_token_budget=1,
        prefix_tokens=3,
    )

    assert _queue_items(response_queues[0]) == [5, 21, _GenerationDone()]
    assert _queue_items(response_queues[1]) == [6, _GenerationDone()]
    assert _queue_items(response_queues[2]) == [8, 22, _GenerationDone()]
    assert [request.done for request in group] == [True, True, True]
    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None
    assert model.forward_calls == [
        ((0,), [[1, 2, 3]]),
        ((0,), [[4]]),
        ((1,), [[5]]),
        ((0,), [[6, 7]]),
    ]
    assert model.ragged_calls == [
        ([[5]], [4], None),
        ([[8]], [5, 4], [0]),
    ]
    assert stats.scheduler_steps == 5
    assert stats.step_commands == 4
    assert stats.decode_run_commands == 0
    assert stats.empty_plans == 1
    assert stats.decode_steps == 2
    assert stats.max_decode_run_steps == 0
    assert stats.prefill_admissions == 3
    assert stats.emitted_tokens == 5
    assert stats.finished_events == 3
    assert stats.closed


def test_openai_persistent_prompt_list_local_group_runner_guard_closes_state() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, queue.Queue()),
    ]

    with pytest.raises(RuntimeError, match="persistent prompt-list local runner exceeded max_scheduler_steps"):
        engine._run_persistent_prompt_list_group_local(
            group,
            max_active_rows=1,
            prefix_tokens=3,
            max_scheduler_steps=1,
        )

    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None


def test_openai_persistent_prompt_list_local_group_runner_coalesces_decode_run() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 3, 0.0, True, second_queue),
    ]

    stats = engine._run_persistent_prompt_list_group_local(
        group,
        max_active_rows=2,
        prefix_tokens=3,
        decode_run_steps=2,
    )

    assert _queue_items(first_queue) == [5, 21, 22, _GenerationDone()]
    assert _queue_items(second_queue) == [6, 21, 22, _GenerationDone()]
    assert [request.done for request in group] == [True, True]
    assert engine._persistent_prompt_list_step_state is None
    assert model.forward_calls == [
        ((0,), [[1, 2, 3]]),
        ((0, 1), [[4], [5]]),
    ]
    assert model.ragged_calls == [
        ([[5], [6]], [4, 4], None),
        ([[21], [21]], [5, 5], None),
    ]
    assert stats.scheduler_steps == 3
    assert stats.step_commands == 1
    assert stats.decode_run_commands == 1
    assert stats.empty_plans == 1
    assert stats.decode_steps == 2
    assert stats.max_decode_run_steps == 2
    assert stats.prefill_admissions == 2
    assert stats.emitted_tokens == 6
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_persistent_prompt_list_local_group_runner_broadcasts_decode_run(monkeypatch) -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()
    calls: list[tuple[str, dict[str, object]]] = []

    def start(model: object, **kwargs: object) -> None:
        del model
        calls.append(("start", dict(kwargs)))

    def step(model: object, payload: dict[str, object]) -> None:
        del model
        calls.append(("step", dict(payload)))

    def decode_run(model: object, payload: dict[str, object]) -> None:
        del model
        calls.append(("decode_run", dict(payload)))

    def close(model: object) -> None:
        del model
        calls.append(("close", {}))

    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_start", start)
    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_step", step)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_decode_run",
        decode_run,
    )
    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_close", close)

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 3, 0.0, True, second_queue),
    ]

    stats = engine._run_persistent_prompt_list_group_local(
        group,
        max_active_rows=2,
        prefix_tokens=3,
        decode_run_steps=2,
        broadcast_tensor_parallel=True,
    )

    assert _queue_items(first_queue) == [5, 21, 22, _GenerationDone()]
    assert _queue_items(second_queue) == [6, 21, 22, _GenerationDone()]
    assert [name for name, _payload in calls] == ["start", "step", "decode_run", "close"]
    assert calls[0][1] == {
        "prefix": (1, 2, 3),
        "cache_batch_size": 2,
        "max_seq_len": 7,
        "temperature": 0.0,
        "max_tokens": 3,
    }
    assert calls[1][1]["step"] == 0
    assert calls[2][1] == {
        "op": "persistent_prompt_list_decode_run",
        "start_step": 1,
        "step_count": 2,
        "temperature": 0.0,
    }
    assert stats.decode_run_commands == 1
    assert stats.closed


def test_openai_persistent_executor_static_bucket_reuse_updates_only_active_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_SIZES", "4")
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    cache = _llama_tp_cache(batch_size=4, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 3, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state = _PersistentPromptListStepState(
        cache=cache,
        prefix_caches={(1, 2, 3): prefix_cache},
        active=[False, False],
        per_row_limits=[0, 0],
        generated_tokens=[0, 0, 0, 0],
        seq_lens=torch.zeros(4, dtype=torch.long),
        next_token_tensor=torch.zeros(4, dtype=torch.long),
        row_request_ids=[None, None],
        cache_batch_size=4,
    )

    first = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 0,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": "a",
                    "row": 0,
                    "prompt": [1, 2, 3, 4],
                    "max_tokens": 2,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [4],
                    "prefill_tokens": 1,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": ["a"],
                    "rows": [0],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [1],
                },
            ],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
        static_graph_buckets=True,
    )
    second = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 1,
            "decode_request_ids": ["a"],
            "decode_rows": [0],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
        static_graph_buckets=True,
    )
    third = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 2,
            "decode_request_ids": [],
            "decode_rows": [],
            "prefill": [
                {
                    "request_id": "b",
                    "row": 1,
                    "prompt": [1, 2, 3, 5],
                    "max_tokens": 2,
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix": [5],
                    "prefill_tokens": 1,
                },
            ],
            "prefill_groups": [
                {
                    "request_ids": ["b"],
                    "rows": [1],
                    "prefix_hit_tokens": 3,
                    "prefix": [1, 2, 3],
                    "suffix_tokens": [1],
                },
            ],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
        static_graph_buckets=True,
    )
    fourth = engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 3,
            "decode_request_ids": ["b"],
            "decode_rows": [1],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
        static_graph_buckets=True,
    )

    assert first.prefill_tokens == {"a": 5}
    assert second.decode_tokens == {"a": 21}
    assert second.finished_request_ids == ("a",)
    assert third.prefill_tokens == {"b": 6}
    assert fourth.decode_tokens == {"b": 22}
    assert fourth.finished_request_ids == ("b",)
    assert state.active == [False, False]
    assert state.row_request_ids == [None, None]
    assert state.seq_lens.tolist() == [5, 5, 0, 0]
    assert state.next_token_tensor.tolist() == [22, 22, 0, 0]
    assert model.forward_calls == [((0,), [[4]]), ((1,), [[5]])]
    assert model.ragged_calls == [
        ([[5], [0], [0], [0]], [4, 0, 0, 0], [0, 1, 2, 3]),
        ([[6], [21], [0], [0]], [5, 4, 0, 0], [1, 0, 2, 3]),
    ]
    torch.testing.assert_close(cache.layers[0].keys[0:1, :, :3, :], prefix_keys)
    torch.testing.assert_close(cache.layers[0].keys[1:2, :, :3, :], prefix_keys)


def test_openai_cache_copy_skips_repeated_shared_prefix_copy() -> None:
    source = _llama_tp_cache(batch_size=1, max_seq_len=8)
    target = _llama_tp_cache(batch_size=3, max_seq_len=8)
    source_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    source_values = source_keys + 100
    source.for_rows((0,)).layers[0].append(source_keys, source_values)
    _mark_generation_cache_prefix(source, (10, 11, 12))

    _copy_generation_cache_first_row(source, target, batch_size=2)
    target.set_seq_len(0)
    source.layers[0].keys[:, :, :3, :].fill_(99)
    source.layers[0].values[:, :, :3, :].fill_(199)

    _copy_generation_cache_first_row(source, target, batch_size=2)

    assert target.for_rows((0, 1)).seq_len == 3
    torch.testing.assert_close(target.layers[0].keys[:2, :, :3, :], source_keys.expand(2, -1, -1, -1))
    torch.testing.assert_close(target.layers[0].values[:2, :, :3, :], source_values.expand(2, -1, -1, -1))

    target.set_seq_len(0)
    _mark_generation_cache_prefix(source, (20, 21, 22))
    _copy_generation_cache_first_row(source, target, batch_size=2)

    torch.testing.assert_close(target.layers[0].keys[:2, :, :3, :], torch.full((2, 1, 3, 2), 99.0))


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
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SAMPLED_SHORT_STREAM_MAX_BATCH_SIZE", raising=False)
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

    assert engine._queued_batch_limit(short_stream) == 64
    assert engine._queued_batch_limit(boundary_stream) == 64
    assert engine._queued_batch_limit(sampled_short_stream) == 64
    assert engine._queued_batch_limit(medium_stream) == 128
    assert engine._queued_batch_limit(sampled_medium_stream) == 128
    assert engine._queued_batch_limit(large_stream) == 32
    assert engine._queued_batch_limit(short_completion) == 128

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", "12")
    assert engine._queued_batch_limit(short_stream) == 12
    assert engine._queued_batch_limit(sampled_short_stream) == 12
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SHORT_STREAM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SAMPLED_SHORT_STREAM_MAX_BATCH_SIZE", "96")
    assert engine._queued_batch_limit(sampled_short_stream) == 96
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SAMPLED_SHORT_STREAM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_SHORT_STREAM_HIGH_TOKEN_MIN", "300")
    assert engine._queued_batch_limit(boundary_stream) == 64
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_LARGE_STREAM_MAX_BATCH_SIZE", "20")
    assert engine._queued_batch_limit(large_stream) == 20


def test_openai_symm_mem_scope_disables_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_BATCH", raising=False)
    model = object()
    captured: list[tuple[int | None, bool | None]] = []

    class FakeContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
            return False

    def fake_symm_scope(max_batch: int | None, *, enabled: bool | None = None) -> FakeContext:
        captured.append((max_batch, enabled))
        return FakeContext()

    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_model", lambda candidate: candidate is model)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_world_size", lambda candidate: 8)
    monkeypatch.setattr("torchinferno.openai_server.symm_mem_allreduce_max_batch", fake_symm_scope)

    with _tensor_parallel_symm_mem_allreduce_scope(
        model,
        torch.device("cuda"),
        max_tokens=64,
        temperature=0.0,
    ):
        pass

    assert captured == [(None, False)]


def test_openai_symm_mem_auto_probe_sets_worker_env_on_success(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_AUTO_PROBE", raising=False)
    calls: list[int] = []

    def fake_probe(config: OpenAIServerConfig) -> bool:
        calls.append(config.tensor_parallel_size)
        return True

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torchinferno.openai_server._run_tensor_parallel_symm_mem_allreduce_probe", fake_probe)

    config = OpenAIServerConfig(model="meta-llama/Llama-3.1", tensor_parallel_size=8)
    _prepare_tensor_parallel_symm_mem_allreduce_auto(config)

    assert calls == [8]
    assert os.environ["TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE"] == "1"


def test_openai_symm_mem_auto_probe_sets_worker_env_on_failure(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    calls: list[int] = []

    def fake_probe(config: OpenAIServerConfig) -> bool:
        calls.append(config.tensor_parallel_size)
        return False

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torchinferno.openai_server._run_tensor_parallel_symm_mem_allreduce_probe", fake_probe)

    config = OpenAIServerConfig(model="meta-llama/Llama-3.1", tensor_parallel_size=8)
    _prepare_tensor_parallel_symm_mem_allreduce_auto(config)

    assert calls == [8]
    assert os.environ["TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE"] == "0"


def test_openai_symm_mem_auto_probe_honors_explicit_env(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", "0")
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)

    def fail_probe(config: OpenAIServerConfig) -> bool:
        raise AssertionError("explicit symm-mem setting should not run auto probe")

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torchinferno.openai_server._run_tensor_parallel_symm_mem_allreduce_probe", fail_probe)

    config = OpenAIServerConfig(model="meta-llama/Llama-3.1", tensor_parallel_size=8)
    _prepare_tensor_parallel_symm_mem_allreduce_auto(config)

    assert os.environ["TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE"] == "0"


def test_openai_symm_mem_scope_enables_explicit_short_deterministic_decode(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", "1")
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE_MAX_BATCH", raising=False)
    model = object()
    captured: list[tuple[int | None, bool | None]] = []

    class FakeContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
            return False

    def fake_symm_scope(max_batch: int | None, *, enabled: bool | None = None) -> FakeContext:
        captured.append((max_batch, enabled))
        return FakeContext()

    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_model", lambda candidate: candidate is model)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_world_size", lambda candidate: 8)
    monkeypatch.setattr("torchinferno.openai_server.symm_mem_allreduce_max_batch", fake_symm_scope)

    with _tensor_parallel_symm_mem_allreduce_scope(
        model,
        torch.device("cuda"),
        max_tokens=64,
        temperature=0.0,
    ):
        pass

    assert captured == [(64, True)]


def test_openai_symm_mem_scope_disables_long_generations(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", "1")
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    model = object()
    captured: list[tuple[int | None, bool | None]] = []

    class FakeContext:
        def __enter__(self): return None
        def __exit__(self, *args): return False

    def fake_symm_scope(max_batch: int | None, *, enabled: bool | None = None):
        captured.append((max_batch, enabled))
        return FakeContext()

    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_model", lambda candidate: candidate is model)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_world_size", lambda candidate: 8)
    monkeypatch.setattr("torchinferno.openai_server.symm_mem_allreduce_max_batch", fake_symm_scope)

    with _tensor_parallel_symm_mem_allreduce_scope(
        model,
        torch.device("cuda"),
        max_tokens=64,
        temperature=0.7,
    ):
        pass
    with _tensor_parallel_symm_mem_allreduce_scope(
        model,
        torch.device("cuda"),
        max_tokens=2048,
        temperature=0.0,
    ):
        pass

    assert captured == [(64, True), (None, False)]


def test_openai_symm_mem_validation_is_openai_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_SYMM_MEM_ALLREDUCE", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_STARTUP_WARMUP", raising=False)
    model = object()
    calls: list[torch.device] = []

    def fake_validate(candidate: object, device: torch.device) -> None:
        assert candidate is model
        calls.append(device)

    monkeypatch.setattr("torchinferno.openai_server._is_tensor_parallel_model", lambda candidate: candidate is model)
    monkeypatch.setattr(
        "torchinferno.models.llama3.tensor_parallel.validate_symm_mem_allreduce_collective",
        fake_validate,
    )

    engine = object.__new__(OpenAICompletionEngine)
    engine.model = model
    engine.device = torch.device("cuda")
    engine.cache_backend = "dense"

    OpenAICompletionEngine._warmup_tensor_parallel_model(engine)
    assert calls == []

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SYMM_MEM_ALLREDUCE", "1")
    OpenAICompletionEngine._warmup_tensor_parallel_model(engine)
    assert calls == [torch.device("cuda")]


def test_openai_refill_min_ready_requests_defaults_for_short_greedy_caps(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_BATCH_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_MIN_READY_REQUESTS", raising=False)

    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=45) == 8
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=128) == 8
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=256) == 8
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=512) == 8
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=1024) is None

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_MIN_READY_REQUESTS", "16")
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=45) == 16
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=256) == 16
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=512) == 16
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=1024) is None
    assert _online_refill_min_ready_requests(temperature=0.7, max_tokens=45) is None
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=0) is None


def test_openai_refill_min_ready_requests_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_MIN_READY_REQUESTS", "8")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_BATCH_MAX_TOKENS", "32")

    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=45) is None

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_REFILL_BATCH_MAX_TOKENS", "64")
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=45) == 8

    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_ADMIT_MIN_READY_REQUESTS", "4")
    assert _online_refill_min_ready_requests(temperature=0.0, max_tokens=45) is None


def test_openai_online_admit_per_step_cap_uses_greedy_mid_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_CONTINUOUS_ADMIT_PER_STEP_CAP", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_ADMIT_PER_STEP_CAP", raising=False)

    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=64) == 48
    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=256) == 64
    assert _online_admit_per_step_cap(temperature=0.7, max_tokens=256) == 48
    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=512) == 48


def test_openai_online_admit_per_step_cap_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_ADMIT_PER_STEP_CAP", "12")
    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=256) == 12

    monkeypatch.delenv("TORCHINFERNO_CONTINUOUS_ADMIT_PER_STEP_CAP", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP", "20")
    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=256) == 20

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_ADMIT_PER_STEP_CAP", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_ADMIT_PER_STEP_CAP", "32")
    assert _online_admit_per_step_cap(temperature=0.0, max_tokens=256) == 32


def test_openai_online_decode_quantum_uses_greedy_mid_cap_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_DECODE_QUANTUM", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_GEN_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_GEN_DECODE_QUANTUM", raising=False)

    assert _online_decode_quantum(temperature=0.0, max_tokens=64) == 8
    assert _online_decode_quantum(temperature=0.0, max_tokens=256) == 5
    assert _online_decode_quantum(temperature=0.7, max_tokens=256) == 4
    assert _online_decode_quantum(temperature=0.0, max_tokens=512) == 16


def test_openai_online_decode_quantum_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_DECODE_QUANTUM", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "3")
    assert _online_decode_quantum(temperature=0.0, max_tokens=256) == 3

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "16")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_GEN_DECODE_QUANTUM", "7")
    assert _online_decode_quantum(temperature=0.0, max_tokens=64) == 7

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_MID_GEN_DECODE_QUANTUM", "6")
    assert _online_decode_quantum(temperature=0.0, max_tokens=256) == 6

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_SHORT_GEN_DECODE_QUANTUM", "2")
    assert _online_decode_quantum(temperature=0.0, max_tokens=256) == 2


def test_online_step_sync_enabled_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC", raising=False)
    assert _online_step_sync_enabled()

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_STEP_SYNC", "0")
    assert not _online_step_sync_enabled()


def test_online_kv_bounded_concurrency_defaults_to_short_outputs(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_ALLOW_SAMPLED", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_GREEDY_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_SAMPLED_MAX_TOKENS", raising=False)

    assert _online_kv_bounded_concurrency_enabled(temperature=0.0, max_tokens=64)
    assert _online_kv_bounded_concurrency_enabled(temperature=0.0, max_tokens=128)
    assert not _online_kv_bounded_concurrency_enabled(temperature=0.0, max_tokens=256)
    assert _online_kv_bounded_concurrency_enabled(temperature=0.7, max_tokens=64)
    assert _online_kv_bounded_concurrency_enabled(temperature=0.7, max_tokens=256)
    assert not _online_kv_bounded_concurrency_enabled(temperature=0.7, max_tokens=300)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_ALLOW_SAMPLED", "0")
    assert not _online_kv_bounded_concurrency_enabled(temperature=0.7, max_tokens=64)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", "0")
    assert not _online_kv_bounded_concurrency_enabled(temperature=0.0, max_tokens=64)


def test_online_kv_bounded_max_active_cap_targets_greedy(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_MAX_ACTIVE_CAP", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_KV_MAX_ACTIVE_CAP", raising=False)

    assert _online_kv_bounded_max_active_cap(temperature=0.0, base_cap=128) == 128
    assert _online_kv_bounded_max_active_cap(temperature=0.7, base_cap=128) == 128
    assert _online_kv_bounded_max_active_cap(temperature=0.0, base_cap=80) == 80

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_KV_MAX_ACTIVE_CAP", "112")
    assert _online_kv_bounded_max_active_cap(temperature=0.0, base_cap=128) == 112

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_KV_MAX_ACTIVE_CAP", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_MAX_ACTIVE_CAP", "128")
    assert _online_kv_bounded_max_active_cap(temperature=0.0, base_cap=128) == 128


def test_openai_online_persistent_idle_uses_sampled_short_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_IDLE_MAX_TOKENS", raising=False)

    assert _online_persistent_idle_ms(temperature=0.7, max_tokens=256) == 100.0
    assert _online_persistent_idle_ms(temperature=0.0, max_tokens=256) == 10.0
    assert _online_persistent_idle_ms(temperature=0.7, max_tokens=300) == 10.0


def test_openai_online_persistent_idle_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS", "25")
    assert _online_persistent_idle_ms(temperature=0.7, max_tokens=256) == 25.0

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT_IDLE_MS", raising=False)
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_IDLE_MAX_TOKENS", "300")
    assert _online_persistent_idle_ms(temperature=0.7, max_tokens=300) == 100.0


def test_openai_online_initial_batch_wait_uses_sampled_short_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MS",
        raising=False,
    )
    monkeypatch.delenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MAX_TOKENS",
        raising=False,
    )
    monkeypatch.delenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS",
        raising=False,
    )
    monkeypatch.delenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MAX_TOKENS",
        raising=False,
    )

    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=256) == 25.0
    assert _online_initial_batch_wait_ms(temperature=0.0, max_tokens=64) == 1.0
    assert _online_initial_batch_wait_ms(temperature=0.0, max_tokens=128) == 1.0
    assert _online_initial_batch_wait_ms(temperature=0.0, max_tokens=256) == 1.0
    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=300) == 25.0
    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=401) == 1.0


def test_openai_online_initial_batch_wait_respects_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "5")
    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=256) == 5.0

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.setenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MS",
        "7",
    )
    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=256) == 7.0
    monkeypatch.setenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_SAMPLED_SHORT_INITIAL_BATCH_WAIT_MAX_TOKENS",
        "300",
    )
    assert _online_initial_batch_wait_ms(temperature=0.7, max_tokens=300) == 7.0
    monkeypatch.setenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MS",
        "9",
    )
    monkeypatch.setenv(
        "TORCHINFERNO_OPENAI_TP_ONLINE_GREEDY_SHORT_INITIAL_BATCH_WAIT_MAX_TOKENS",
        "160",
    )
    assert _online_initial_batch_wait_ms(temperature=0.0, max_tokens=160) == 9.0
    assert _online_initial_batch_wait_ms(temperature=0.0, max_tokens=161) == 1.0


def test_online_common_prefix_prefill_warmup_filters_shapes(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_ROWS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_TOKENS", raising=False)

    assert _online_common_prefix_prefill_warmup_rows(49) == (48,)
    assert _online_common_prefix_prefill_warmup_rows(69) == (48, 53, 68)
    assert _online_common_prefix_prefill_warmup_rows(70) == (48, 53, 68, 69)
    assert _online_common_prefix_prefill_warmup_rows(144, 128) == (48, 53, 68, 69, 128)
    assert _online_common_prefix_prefill_warmup_tokens(64) == (45,)
    assert _online_common_prefix_suffix_prefill_warmup_tokens(64) == (16,)
    assert _online_common_prefix_suffix_prefill_warmup_batches(49, 48) == (
        1,
        2,
        4,
        8,
        16,
        32,
        48,
    )

    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_ROWS", "53,96,128")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_TOKENS", "32,128,256")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_SUFFIX_TOKENS", "8,16,256")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_WARMUP_ONLINE_COMMON_PREFIX_SUFFIX_BATCHES", "8,48,96")
    assert _online_common_prefix_prefill_warmup_rows(128, 64) == (53, 96)
    assert _online_common_prefix_prefill_warmup_tokens(128) == (32, 128)
    assert _online_common_prefix_suffix_prefill_warmup_tokens(128) == (8, 16)
    assert _online_common_prefix_suffix_prefill_warmup_batches(64, 128) == (8, 48)


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
    assert engine._queued_batch_wait_s(medium_sampled) == 0.010
    assert engine._queued_batch_wait_s(long_sampled) == 0.010


def test_openai_tp_greedy_stream_skips_queue_batch_wait(monkeypatch) -> None:
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_BATCH_WAIT_MS", raising=False)
    model = object()
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.batch_wait_s = 0.010

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    short_stream = _QueuedGeneration([1, 2], 64, 0.0, True, queue.Queue())
    medium_stream = _QueuedGeneration([1, 2], 256, 0.0, True, queue.Queue())
    long_stream = _QueuedGeneration([1, 2], 512, 0.0, True, queue.Queue())
    completion = _QueuedGeneration([1, 2], 256, 0.0, False, queue.Queue())

    assert engine._queued_batch_wait_s(short_stream) == 0.0
    assert engine._queued_batch_wait_s(medium_stream) == 0.0
    assert engine._queued_batch_wait_s(long_stream) == 0.010
    assert engine._queued_batch_wait_s(completion) == 0.010

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_GREEDY_LOW_LATENCY_BATCH_WAIT_MS", "3")
    assert engine._queued_batch_wait_s(short_stream) == 0.003
    assert engine._queued_batch_wait_s(medium_stream) == 0.003


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
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_SHORT_OUTPUT_INITIAL_BATCH_WAIT_MS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_GREEDY_SHORT_OUTPUT_INITIAL_BATCH_MAX_TOKENS", raising=False)
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
    short_greedy = _QueuedGeneration([1, 2], 64, 0.0, True, queue.Queue())
    greedy = _QueuedGeneration([1, 2], 300, 0.0, True, queue.Queue())
    completion = _QueuedGeneration([1, 2], 300, 0.7, False, queue.Queue())
    long_sampled = _QueuedGeneration([1, 2], 600, 0.7, True, queue.Queue())

    assert engine._queued_initial_batch_wait_s(short_sampled) == 0.001
    assert engine._queued_initial_batch_wait_s(sampled) == 0.001
    assert engine._queued_initial_batch_wait_s(short_greedy) == 0.005
    assert engine._queued_initial_batch_wait_s(greedy) == 0.005
    assert engine._queued_initial_batch_wait_s(completion) == 0.0
    assert engine._queued_initial_batch_wait_s(long_sampled) == 0.0

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SHORT_SAMPLED_INITIAL_BATCH_WAIT_MS", "3")
    assert engine._queued_initial_batch_wait_s(short_sampled) == 0.003
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SAMPLED_INITIAL_BATCH_WAIT_MS", "2")
    assert engine._queued_initial_batch_wait_s(sampled) == 0.002
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_GREEDY_INITIAL_BATCH_WAIT_MS", "4")
    assert engine._queued_initial_batch_wait_s(short_greedy) == 0.004
    assert engine._queued_initial_batch_wait_s(greedy) == 0.004


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


def test_openai_stream_group_respects_per_request_max_tokens(monkeypatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", "0")
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
        **_kw: object,
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


def test_openai_persistent_prompt_list_scheduler_maps_group_prefix_budget() -> None:
    group = [
        _QueuedGeneration([1, 2, 3, 4], 4, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 5, 6], 4, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 7], 4, 0.0, True, queue.Queue()),
    ]

    scheduler = _persistent_prompt_list_scheduler_for_group(
        group,
        max_active_rows=2,
        prefill_token_budget=4,
        prefix_tokens=3,
    )
    first = scheduler.step()
    refill = scheduler.step(finished_request_ids=("0",))

    assert [(item.request_id, item.row, item.prefill_tokens, item.prefix_key) for item in first.prefill_admissions] == [
        ("0", 0, 1, (1, 2, 3)),
        ("1", 1, 2, (1, 2, 3)),
    ]
    assert first.prefill_groups[0].request_ids == ("0", "1")
    assert first.prefill_groups[0].suffix_tokens == (1, 2)
    assert refill.decode_request_ids == ("1",)
    assert refill.decode_rows == (1,)
    assert [(item.request_id, item.row) for item in refill.prefill_admissions] == [("2", 0)]


def test_openai_persistent_prompt_list_scheduler_drops_invalid_shared_prefix_key() -> None:
    group = [
        _QueuedGeneration([1, 2, 3], 2, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 9, 3], 2, 0.0, True, queue.Queue()),
    ]

    scheduler = _persistent_prompt_list_scheduler_for_group(
        group,
        max_active_rows=2,
        prefix_tokens=2,
    )
    plan = scheduler.step()

    assert [item.prefix_hit_tokens for item in plan.prefill_admissions] == [0, 0]
    assert [item.prefix_key for item in plan.prefill_admissions] == [None, None]


def test_openai_token_budget_scheduler_maps_group_prefix_budget() -> None:
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 5, 6], 2, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 7], 1, 0.0, True, queue.Queue()),
    ]

    scheduler = _token_budget_scheduler_for_group(
        group,
        max_active_rows=2,
        max_scheduled_tokens=4,
        prefill_chunk_size=4,
        prefix_tokens=3,
    )
    first = scheduler.step()
    refill = scheduler.step(finished_request_ids=("0", "1"))

    assert [
        (
            chunk.request_id,
            chunk.row,
            chunk.kind,
            chunk.start_token,
            chunk.token_count,
            chunk.prompt_complete,
            chunk.emits_token,
            chunk.prefix_key,
        )
        for chunk in first.chunks
    ] == [
        ("0", 0, "prefill", 3, 1, True, True, (1, 2, 3)),
        ("1", 1, "prefill", 3, 2, True, True, (1, 2, 3)),
    ]
    assert first.finished_request_ids == ()
    assert [(chunk.request_id, chunk.row, chunk.start_token, chunk.token_count) for chunk in refill.chunks] == [
        ("2", 0, 3, 1),
    ]


def test_openai_token_budget_scheduler_drops_invalid_shared_prefix_key() -> None:
    group = [
        _QueuedGeneration([1, 2, 3], 2, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 9, 3], 2, 0.0, True, queue.Queue()),
    ]

    scheduler = _token_budget_scheduler_for_group(
        group,
        max_active_rows=2,
        max_scheduled_tokens=6,
        prefix_tokens=2,
    )
    plan = scheduler.step()

    assert [(chunk.request_id, chunk.start_token, chunk.prefix_key) for chunk in plan.chunks] == [
        ("0", 0, None),
        ("1", 0, None),
    ]


def test_openai_persistent_prompt_list_step_payload_binds_plan_to_prompts() -> None:
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 5], 3, 0.0, True, queue.Queue()),
        _QueuedGeneration([1, 2, 3, 6], 2, 0.0, True, queue.Queue()),
    ]
    scheduler = _persistent_prompt_list_scheduler_for_group(
        group,
        max_active_rows=2,
        prefill_token_budget=4,
        prefix_tokens=3,
    )

    first_payload = _persistent_prompt_list_step_payload(scheduler.step(), group)
    refill_payload = _persistent_prompt_list_step_payload(
        scheduler.step(finished_request_ids=("0",)),
        group,
    )

    assert first_payload == {
        "op": "persistent_prompt_list_step",
        "step": 0,
        "decode_request_ids": [],
        "decode_rows": [],
        "prefill": [
            {
                "request_id": "0",
                "row": 0,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [4],
                "prefill_tokens": 1,
            },
            {
                "request_id": "1",
                "row": 1,
                "prompt": [1, 2, 3, 5],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [5],
                "prefill_tokens": 1,
            },
        ],
        "prefill_groups": [
            {
                "request_ids": ["0", "1"],
                "rows": [0, 1],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1, 1],
            }
        ],
        "finished_after_prefill": [],
    }
    assert refill_payload["decode_request_ids"] == ["1"]
    assert refill_payload["decode_rows"] == [1]
    assert refill_payload["prefill"] == [
        {
            "request_id": "2",
            "row": 0,
            "prompt": [1, 2, 3, 6],
            "max_tokens": 2,
            "prefix_hit_tokens": 3,
            "prefix": [1, 2, 3],
            "suffix": [6],
            "prefill_tokens": 1,
        }
    ]
    assert refill_payload["prefill_groups"] == [
        {
            "request_ids": ["2"],
            "rows": [0],
            "prefix_hit_tokens": 3,
            "prefix": [1, 2, 3],
            "suffix_tokens": [1],
        }
    ]


def test_openai_token_budget_step_payload_binds_prompt_chunks() -> None:
    request_by_id = {
        "running": _QueuedGeneration([11, 12, 13], 3, 0.0, True, queue.Queue()),
        "chunked": _QueuedGeneration([21, 22, 23, 24, 25], 2, 0.0, True, queue.Queue()),
        "done": _QueuedGeneration([31, 32, 33, 34], 1, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=4,
        chunks=(
            TokenBudgetScheduledChunk(
                "running",
                row=0,
                kind="decode",
                start_token=3,
                token_count=1,
                prompt_complete=True,
                emits_token=True,
            ),
            TokenBudgetScheduledChunk(
                "chunked",
                row=1,
                kind="prefill",
                start_token=2,
                token_count=2,
            ),
            TokenBudgetScheduledChunk(
                "done",
                row=2,
                kind="prefill",
                start_token=1,
                token_count=3,
                prompt_complete=True,
                emits_token=True,
            ),
        ),
        finished_request_ids=("done",),
    )

    payload = _token_budget_step_payload(plan, request_by_id)

    assert payload == {
        "op": "token_budget_step",
        "step": 4,
        "chunks": [
            {
                "request_id": "running",
                "row": 0,
                "kind": "decode",
                "start_token": 3,
                "token_count": 1,
                "prompt_complete": True,
                "emits_token": True,
            },
            {
                "request_id": "chunked",
                "row": 1,
                "kind": "prefill",
                "start_token": 2,
                "token_count": 2,
                "prompt_complete": False,
                "emits_token": False,
                "prompt_chunk": [23, 24],
                "prompt_tokens": 5,
                "max_tokens": 2,
            },
            {
                "request_id": "done",
                "row": 2,
                "kind": "prefill",
                "start_token": 1,
                "token_count": 3,
                "prompt_complete": True,
                "emits_token": True,
                "prompt_chunk": [32, 33, 34],
                "prompt_tokens": 4,
                "max_tokens": 1,
            },
        ],
        "decode_rows": [0],
        "prefill_rows": [1, 2],
        "emit_request_ids": ["running", "done"],
        "emit_rows": [0, 2],
        "finished_request_ids": ["done"],
        "scheduled_tokens": 6,
    }


def test_openai_token_budget_step_payload_includes_prefix_for_suffix_prefill() -> None:
    request_by_id = {
        "0": _QueuedGeneration([11, 12, 13, 14], 1, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=0,
        chunks=(
            TokenBudgetScheduledChunk(
                "0",
                row=0,
                kind="prefill",
                start_token=2,
                token_count=2,
                prompt_complete=True,
                emits_token=True,
                prefix_key=(11, 12),
            ),
        ),
        finished_request_ids=("0",),
    )

    payload = _token_budget_step_payload(plan, request_by_id)

    assert payload["chunks"] == [
        {
            "request_id": "0",
            "row": 0,
            "kind": "prefill",
            "start_token": 2,
            "token_count": 2,
            "prompt_complete": True,
            "emits_token": True,
            "prompt_chunk": [13, 14],
            "prompt_tokens": 4,
            "max_tokens": 1,
            "prefix": [11, 12],
        }
    ]


def test_openai_token_budget_decode_run_payload_rejects_prefill_plan() -> None:
    request_by_id = {
        "0": _QueuedGeneration([1, 2], 1, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=0,
        chunks=(
            TokenBudgetScheduledChunk(
                "0",
                row=0,
                kind="prefill",
                start_token=0,
                token_count=2,
            ),
        ),
        finished_request_ids=(),
    )

    with pytest.raises(ValueError, match="decode-only"):
        _token_budget_decode_run_payload([plan], request_by_id)


def test_openai_token_budget_step_payload_rejects_bad_prefill_slice() -> None:
    request_by_id = {
        "bad": _QueuedGeneration([1, 2], 1, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=0,
        chunks=(TokenBudgetScheduledChunk("bad", row=0, kind="prefill", start_token=1, token_count=2),),
        finished_request_ids=(),
    )

    with pytest.raises(ValueError, match="outside the prompt"):
        _token_budget_step_payload(plan, request_by_id)


def test_openai_token_budget_step_tensor_payload_round_trips_numeric_ids() -> None:
    request_by_id = {
        "0": _QueuedGeneration([11, 12, 13], 3, 0.0, True, queue.Queue()),
        "1": _QueuedGeneration([21, 22, 23, 24, 25], 2, 0.0, True, queue.Queue()),
        "2": _QueuedGeneration([31, 32, 33, 34], 1, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=4,
        chunks=(
            TokenBudgetScheduledChunk(
                "0",
                row=0,
                kind="decode",
                start_token=3,
                token_count=1,
                prompt_complete=True,
                emits_token=True,
            ),
            TokenBudgetScheduledChunk(
                "1",
                row=1,
                kind="prefill",
                start_token=2,
                token_count=2,
            ),
            TokenBudgetScheduledChunk(
                "2",
                row=2,
                kind="prefill",
                start_token=1,
                token_count=3,
                prompt_complete=True,
                emits_token=True,
            ),
        ),
        finished_request_ids=("2",),
    )
    payload = _token_budget_step_payload(plan, request_by_id)

    meta, chunks, prefill_lengths, prefill_token_rows, finished_ids = _token_budget_step_tensor_payload(
        payload,
        torch.device("cpu"),
    )
    decoded = _token_budget_step_payload_from_tensor_payload(
        meta,
        chunks,
        prefill_lengths,
        prefill_token_rows,
        finished_ids,
    )

    assert meta.tolist() == [_TP_COMMAND_TOKEN_BUDGET_STEP, 4, 3, 10, 6, 2, 3, 2, 1, 0, 0]
    assert chunks.tolist() == [
        [0, 0, 1, 3, 1, 1, 1, -1, -1, -1],
        [1, 1, 0, 2, 2, 0, 0, 0, 5, 2],
        [2, 2, 0, 1, 3, 1, 1, 1, 4, 1],
    ]
    assert prefill_lengths.tolist() == [2, 3]
    assert prefill_token_rows.tolist() == [[23, 24, 0], [32, 33, 34]]
    assert finished_ids.tolist() == [2]
    assert decoded == payload


def test_openai_token_budget_decode_run_tensor_payload_round_trips_numeric_ids() -> None:
    payload = {
        "op": "token_budget_decode_run",
        "step_count": 2,
        "steps": [
            {
                "op": "token_budget_step",
                "step": 7,
                "chunks": [
                    {
                        "request_id": "0",
                        "row": 0,
                        "kind": "decode",
                        "start_token": 3,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    },
                    {
                        "request_id": "1",
                        "row": 1,
                        "kind": "decode",
                        "start_token": 4,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    },
                ],
                "decode_rows": [0, 1],
                "prefill_rows": [],
                "emit_request_ids": ["0", "1"],
                "emit_rows": [0, 1],
                "finished_request_ids": [],
                "scheduled_tokens": 2,
            },
            {
                "op": "token_budget_step",
                "step": 8,
                "chunks": [
                    {
                        "request_id": "0",
                        "row": 0,
                        "kind": "decode",
                        "start_token": 4,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    }
                ],
                "decode_rows": [0],
                "prefill_rows": [],
                "emit_request_ids": ["0"],
                "emit_rows": [0],
                "finished_request_ids": ["0"],
                "scheduled_tokens": 1,
            },
        ],
    }

    meta, step_tensor, chunk_tensor, finished_ids = _token_budget_decode_run_tensor_payload(
        payload,
        torch.device("cpu"),
    )
    decoded = _token_budget_decode_run_payload_from_tensor_payload(
        meta,
        step_tensor,
        chunk_tensor,
        finished_ids,
    )

    assert meta.tolist() == [_TP_COMMAND_TOKEN_BUDGET_DECODE_RUN, 2, 3, 10, 1, 5, 0, 0, 0, 0, 0]
    assert step_tensor.tolist() == [[7, 0, 2, 0, 0], [8, 2, 1, 0, 1]]
    assert chunk_tensor.tolist() == [
        [0, 0, 1, 3, 1, 1, 1, -1, -1, -1],
        [1, 1, 1, 4, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 4, 1, 1, 1, -1, -1, -1],
    ]
    assert finished_ids.tolist() == [0]
    assert decoded == payload


def test_openai_token_budget_step_tensor_payload_rejects_object_request_ids() -> None:
    payload = {
        "op": "token_budget_step",
        "step": 0,
        "chunks": [
            {
                "request_id": "not-numeric",
                "row": 0,
                "kind": "decode",
                "start_token": 2,
                "token_count": 1,
                "prompt_complete": True,
                "emits_token": True,
            }
        ],
        "finished_request_ids": [],
    }

    with pytest.raises(ValueError, match="numeric request ids"):
        _token_budget_step_tensor_payload(payload, torch.device("cpu"))


def test_tensor_parallel_token_budget_step_broadcast_uses_tensor_payload(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []
    payload = {
        "op": "token_budget_step",
        "step": 4,
        "chunks": [
            {
                "request_id": "0",
                "row": 0,
                "kind": "decode",
                "start_token": 3,
                "token_count": 1,
                "prompt_complete": True,
                "emits_token": True,
            },
            {
                "request_id": "1",
                "row": 1,
                "kind": "prefill",
                "start_token": 2,
                "token_count": 2,
                "prompt_complete": False,
                "emits_token": False,
                "prompt_chunk": [23, 24],
                "prompt_tokens": 5,
                "max_tokens": 2,
            },
        ],
        "decode_rows": [0],
        "prefill_rows": [1],
        "emit_request_ids": ["0"],
        "emit_rows": [0],
        "finished_request_ids": ["0"],
        "scheduled_tokens": 3,
    }

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_token_budget_step(model, payload)

    assert captured[0].tolist() == [_TP_COMMAND_TOKEN_BUDGET_STEP, 4, 2, 10, 3, 1, 2, 1, 1, 0, 0]
    assert captured[1].tolist() == [
        [0, 0, 1, 3, 1, 1, 1, -1, -1, -1],
        [1, 1, 0, 2, 2, 0, 0, 0, 5, 2],
    ]
    assert captured[2].tolist() == [2]
    assert captured[3].tolist() == [[23, 24]]
    assert captured[4].tolist() == [0]


def test_tensor_parallel_token_budget_decode_run_broadcast_uses_tensor_payload(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []
    payload = {
        "op": "token_budget_decode_run",
        "steps": [
            {
                "op": "token_budget_step",
                "step": 7,
                "chunks": [
                    {
                        "request_id": "0",
                        "row": 0,
                        "kind": "decode",
                        "start_token": 3,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    }
                ],
                "finished_request_ids": ["0"],
            }
        ],
    }

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_token_budget_decode_run(model, payload)

    assert captured[0].tolist() == [_TP_COMMAND_TOKEN_BUDGET_DECODE_RUN, 1, 1, 10, 1, 5, 0, 0, 0, 0, 0]
    assert captured[1].tolist() == [[7, 0, 1, 0, 1]]
    assert captured[2].tolist() == [[0, 0, 1, 3, 1, 1, 1, -1, -1, -1]]
    assert captured[3].tolist() == [0]


def test_tensor_parallel_token_budget_lifecycle_broadcast_uses_tensor_payload(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_token_budget_start(
        model,
        max_seq_len=128,
        max_active_rows=16,
        temperature=0.25,
        max_tokens=32,
    )
    _broadcast_tensor_parallel_token_budget_close(model)

    assert captured[0].tolist() == [_TP_COMMAND_TOKEN_BUDGET_START, 0, 0, 0, 32, 0, 128, 16, 0, 0, 0]
    assert captured[1].tolist() == [0.25]
    assert captured[2].tolist() == [_TP_COMMAND_TOKEN_BUDGET_CLOSE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_tensor_parallel_token_budget_start_broadcast_carries_prefix_tensor(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS_GLOO", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "device": torch.device("cpu")})()
    captured: list[torch.Tensor] = []

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        captured.append(tensor.detach().cpu().clone())

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)

    _broadcast_tensor_parallel_token_budget_start(
        model,
        max_seq_len=128,
        max_active_rows=16,
        temperature=0.25,
        max_tokens=32,
        prefix=(11, 12, 13),
    )

    assert captured[0].tolist() == [_TP_COMMAND_TOKEN_BUDGET_START, 0, 0, 0, 32, 3, 128, 16, 0, 0, 0]
    assert captured[1].tolist() == [0.25]
    assert captured[2].tolist() == [11, 12, 13]


def test_tensor_parallel_token_budget_lifecycle_closes_on_exception(monkeypatch) -> None:
    model = object()
    calls: list[object] = []

    def start(model_arg: object, **kwargs: object) -> None:
        calls.append(("start", model_arg, kwargs))

    def close(model_arg: object) -> None:
        calls.append(("close", model_arg))

    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_token_budget_start", start)
    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_token_budget_close", close)

    with pytest.raises(RuntimeError, match="boom"):
        with _tensor_parallel_token_budget_lifecycle(
            model,
            max_seq_len=128,
            max_active_rows=16,
            temperature=0.25,
            max_tokens=32,
        ):
            calls.append("body")
            raise RuntimeError("boom")

    assert calls == [
        (
            "start",
            model,
            {
                "max_seq_len": 128,
                "max_active_rows": 16,
                "temperature": 0.25,
                "max_tokens": 32,
                "prefix": (),
            },
        ),
        "body",
        ("close", model),
    ]


def test_tensor_parallel_worker_loop_receives_token_budget_tensor_step(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_STEP, 4, 2, 10, 3, 1, 2, 1, 1, 0, 0], dtype=torch.long),
        torch.tensor(
            [
                [0, 0, 1, 3, 1, 1, 1, -1, -1, -1],
                [1, 1, 0, 2, 2, 0, 0, 0, 5, 2],
            ],
            dtype=torch.long,
        ),
        torch.tensor([2], dtype=torch.long),
        torch.tensor([[23, 24]], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = model
            self.handled: list[dict[str, object]] = []

        def _handle_token_budget_step_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

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

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert payloads == []
    assert engine.handled == [
        {
            "op": "token_budget_step",
            "step": 4,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "decode",
                    "start_token": 3,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                },
                {
                    "request_id": "1",
                    "row": 1,
                    "kind": "prefill",
                    "start_token": 2,
                    "token_count": 2,
                    "prompt_complete": False,
                    "emits_token": False,
                    "prompt_chunk": [23, 24],
                    "prompt_tokens": 5,
                    "max_tokens": 2,
                },
            ],
            "decode_rows": [0],
            "prefill_rows": [1],
            "emit_request_ids": ["0"],
            "emit_rows": [0],
            "finished_request_ids": ["0"],
            "scheduled_tokens": 3,
        }
    ]


def test_tensor_parallel_worker_loop_receives_token_budget_tensor_decode_run(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_DECODE_RUN, 2, 2, 10, 1, 5, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([[7, 0, 1, 0, 0], [8, 1, 1, 0, 1]], dtype=torch.long),
        torch.tensor(
            [
                [0, 0, 1, 3, 1, 1, 1, -1, -1, -1],
                [0, 0, 1, 4, 1, 1, 1, -1, -1, -1],
            ],
            dtype=torch.long,
        ),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = model
            self.handled: list[dict[str, object]] = []

        def _handle_token_budget_decode_run_payload(self, command: dict[str, object]) -> None:
            self.handled.append(dict(command))

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

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert payloads == []
    assert engine.handled == [
        {
            "op": "token_budget_decode_run",
            "steps": [
                {
                    "op": "token_budget_step",
                    "step": 7,
                    "chunks": [
                        {
                            "request_id": "0",
                            "row": 0,
                            "kind": "decode",
                            "start_token": 3,
                            "token_count": 1,
                            "prompt_complete": True,
                            "emits_token": True,
                        }
                    ],
                    "decode_rows": [0],
                    "prefill_rows": [],
                    "emit_request_ids": ["0"],
                    "emit_rows": [0],
                    "finished_request_ids": [],
                    "scheduled_tokens": 1,
                },
                {
                    "op": "token_budget_step",
                    "step": 8,
                    "chunks": [
                        {
                            "request_id": "0",
                            "row": 0,
                            "kind": "decode",
                            "start_token": 4,
                            "token_count": 1,
                            "prompt_complete": True,
                            "emits_token": True,
                        }
                    ],
                    "decode_rows": [0],
                    "prefill_rows": [],
                    "emit_request_ids": ["0"],
                    "emit_rows": [0],
                    "finished_request_ids": ["0"],
                    "scheduled_tokens": 1,
                },
            ],
            "step_count": 2,
        }
    ]


def test_tensor_parallel_worker_loop_receives_token_budget_tensor_lifecycle(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_START, 0, 0, 0, 32, 0, 128, 16, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_STEP, 4, 1, 10, 1, 0, 0, 1, 1, 0, 0], dtype=torch.long),
        torch.tensor([[0, 0, 1, 3, 1, 1, 1, -1, -1, -1]], dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 0), dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_CLOSE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = model
            self.events: list[tuple[str, dict[str, object]]] = []

        def _handle_token_budget_start_payload(self, command: dict[str, object]) -> None:
            self.events.append(("start", dict(command)))

        def _handle_token_budget_step_payload(self, command: dict[str, object]) -> None:
            self.events.append(("step", dict(command)))

        def _handle_token_budget_close_payload(self, command: dict[str, object]) -> None:
            self.events.append(("close", dict(command)))

    def broadcast(tensor: torch.Tensor, *, src: int) -> None:
        del src
        tensor.copy_(payloads.pop(0).to(device=tensor.device, dtype=tensor.dtype))

    scope_events: list[tuple[str, object]] = []

    class Scope:
        def __enter__(self) -> None:
            scope_events.append(("enter", None))

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            scope_events.append(("exit", None))

    def symm_scope(model_arg: object, device_arg: torch.device, **kwargs: object) -> Scope:
        scope_events.append(("create", (model_arg, device_arg, kwargs)))
        return Scope()

    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    monkeypatch.setattr("torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope", symm_scope)

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert payloads == []
    assert scope_events == [
        ("create", (model, torch.device("cpu"), {"max_tokens": 32, "temperature": 0.25})),
        ("enter", None),
        ("exit", None),
    ]
    assert engine.events == [
        (
            "start",
            {
                "op": "token_budget_start",
                "max_seq_len": 128,
                "max_active_rows": 16,
                "temperature": 0.25,
                "max_tokens": 32,
                "prefix": [],
            },
        ),
        (
            "step",
            {
                "op": "token_budget_step",
                "step": 4,
                "chunks": [
                    {
                        "request_id": "0",
                        "row": 0,
                        "kind": "decode",
                        "start_token": 3,
                        "token_count": 1,
                        "prompt_complete": True,
                        "emits_token": True,
                    }
                ],
                "decode_rows": [0],
                "prefill_rows": [],
                "emit_request_ids": ["0"],
                "emit_rows": [0],
                "finished_request_ids": ["0"],
                "scheduled_tokens": 1,
            },
        ),
        ("close", {"op": "token_budget_close"}),
    ]


def test_tensor_parallel_worker_loop_exits_token_budget_scope_on_stop(monkeypatch) -> None:
    import torch.distributed as dist

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_TENSOR_COMMANDS", "1")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 1})()
    payloads = [
        torch.tensor([_TP_COMMAND_TOKEN_BUDGET_START, 0, 0, 0, 12, 0, 128, 8, 0, 0, 0], dtype=torch.long),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([_TP_COMMAND_STOP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
    ]
    scope_events: list[str] = []

    class Scope:
        def __enter__(self) -> None:
            scope_events.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            scope_events.append("exit")

    class Engine:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.model = model
            self.events: list[str] = []

        def _handle_token_budget_start_payload(self, command: dict[str, object]) -> None:
            del command
            self.events.append("start")

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
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: Scope(),
    )

    engine = Engine()
    _tensor_parallel_worker_loop(engine)  # type: ignore[arg-type]

    assert payloads == []
    assert engine.events == ["start"]
    assert scope_events == ["enter", "exit"]


def test_openai_token_budget_step_executor_chunks_prefill_then_decodes() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    state = engine._start_token_budget_step_state(cache_batch_size=2, max_seq_len=8)

    first = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 0,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "prefill",
                    "start_token": 0,
                    "token_count": 2,
                    "prompt_complete": False,
                    "emits_token": False,
                    "prompt_chunk": [1, 2],
                }
            ],
            "finished_request_ids": [],
        },
        state,
        temperature=0.0,
    )
    second = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 1,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "prefill",
                    "start_token": 2,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prompt_chunk": [3],
                }
            ],
            "finished_request_ids": [],
        },
        state,
        temperature=0.0,
    )
    third = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 2,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "decode",
                    "start_token": 3,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                }
            ],
            "finished_request_ids": ["0"],
        },
        state,
        temperature=0.0,
    )

    assert first.prefill_tokens == {"0": None}
    assert first.decode_tokens == {}
    assert second.prefill_tokens == {"0": 4}
    assert state.next_token_tensor.tolist() == [5, 0]
    assert third.decode_tokens == {"0": 5}
    assert third.finished_request_ids == ("0",)
    assert state.row_request_ids == [None, None]
    assert state.active == [False, False]
    assert state.seq_lens.tolist() == [4, 0]
    assert model.forward_calls == [((0,), [[1, 2]]), ((0,), [[3]]), ((0,), [[4]])]


def test_openai_token_budget_step_executor_batches_ragged_decode_rows() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    state = engine._start_token_budget_step_state(cache_batch_size=2, max_seq_len=8)
    engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 0,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "prefill",
                    "start_token": 0,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prompt_chunk": [2],
                },
                {
                    "request_id": "1",
                    "row": 1,
                    "kind": "prefill",
                    "start_token": 0,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prompt_chunk": [5],
                },
            ],
            "finished_request_ids": [],
        },
        state,
        temperature=0.0,
    )

    result = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 1,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "decode",
                    "start_token": 1,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                },
                {
                    "request_id": "1",
                    "row": 1,
                    "kind": "decode",
                    "start_token": 1,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                },
            ],
            "finished_request_ids": ["0", "1"],
        },
        state,
        temperature=0.0,
    )

    assert result.decode_tokens == {"0": 21, "1": 21}
    assert result.finished_request_ids == ("0", "1")
    assert model.ragged_calls == [([[3], [6]], [1, 1], [0, 1])]
    assert state.row_request_ids == [None, None]
    assert state.active == [False, False]
    assert state.seq_lens.tolist() == [2, 2]


def test_openai_token_budget_step_executor_copies_prefix_cache_before_suffix_prefill() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    state = engine._start_token_budget_step_state(cache_batch_size=2, max_seq_len=8)
    prefix_cache = _llama_tp_cache(batch_size=1, max_seq_len=8)
    prefix_keys = torch.full((1, 1, 2, 2), 5.0)
    prefix_cache.for_rows((0,)).layers[0].append(prefix_keys, prefix_keys + 100)
    state.prefix_caches = {(1, 2): prefix_cache}

    result = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 0,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 1,
                    "kind": "prefill",
                    "start_token": 2,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prefix": [1, 2],
                    "prompt_chunk": [3],
                }
            ],
            "finished_request_ids": [],
        },
        state,
        temperature=0.0,
    )

    assert result.prefill_tokens == {"0": 4}
    assert state.row_request_ids == [None, "0"]
    assert state.seq_lens.tolist() == [0, 3]
    assert model.forward_calls == [((1,), [[3]])]
    torch.testing.assert_close(state.cache.layers[0].keys[1:2, :, :2, :], prefix_keys)
    torch.testing.assert_close(
        state.cache.layers[0].keys[1, :, 2:3, :],
        torch.tensor([[[3.0, 3.0]]]),
    )


def test_openai_token_budget_step_executor_batches_shared_prefix_suffix_prefill() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    state = engine._start_token_budget_step_state(
        cache_batch_size=2,
        max_seq_len=8,
        prefix=[1, 2],
        temperature=0.0,
    )

    result = engine._execute_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 0,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "prefill",
                    "start_token": 2,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prefix": [1, 2],
                    "prompt_chunk": [3],
                    "prompt_tokens": 3,
                    "max_tokens": 1,
                },
                {
                    "request_id": "1",
                    "row": 1,
                    "kind": "prefill",
                    "start_token": 2,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prefix": [1, 2],
                    "prompt_chunk": [5],
                    "prompt_tokens": 3,
                    "max_tokens": 1,
                },
            ],
            "finished_request_ids": ["0", "1"],
        },
        state,
        temperature=0.0,
    )

    assert result.prefill_tokens == {"0": 4, "1": 6}
    assert result.finished_request_ids == ("0", "1")
    assert model.forward_calls == [((0,), [[1, 2]]), ((0, 1), [[3], [5]])]
    assert state.row_request_ids == [None, None]
    assert state.active == [False, False]
    torch.testing.assert_close(
        state.cache.layers[0].keys[:2, :, :2, :],
        torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]], [[[1.0, 1.0], [2.0, 2.0]]]]),
    )


def test_openai_persistent_prompt_list_state_sets_ragged_graph_capture(monkeypatch) -> None:
    engine = _cache_only_engine()
    model = _RowTargetPrefillModel()
    model.world_size = 2
    model.rank = 0
    engine.model = model
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    state = engine._start_persistent_prompt_list_step_state(
        prefix=[1, 2],
        cache_batch_size=2,
        max_seq_len=8,
        temperature=0.0,
        max_tokens=4,
    )

    assert getattr(state.cache, "_torchinferno_runtime_ragged_decode_capture") is True


def test_openai_persistent_prompt_list_decode_step_sets_static_bucket_mode(monkeypatch) -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    model.world_size = 2
    model.rank = 0
    engine.model = model
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    state = engine._start_persistent_prompt_list_step_state(
        prefix=[1, 2],
        cache_batch_size=2,
        max_seq_len=8,
        temperature=0.0,
        max_tokens=4,
    )
    state.active[0] = True
    state.per_row_limits[0] = 2
    state.generated_tokens[0] = 1
    state.seq_lens[0] = 3
    state.next_token_tensor[0] = 4
    state.row_request_ids[0] = "0"

    engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 1,
            "decode_request_ids": ["0"],
            "decode_rows": [0],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
        static_graph_buckets=True,
    )

    assert getattr(state.cache, "_torchinferno_shared_prefix_ragged_static_graph_buckets") is True


def test_openai_persistent_prompt_list_decode_step_sets_ephemeral_graph_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    released: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "torchinferno.openai_server._release_decode_graphs_for_cache",
        lambda model_arg, cache_arg: released.append((model_arg, cache_arg)),
    )
    state = engine._start_persistent_prompt_list_step_state(
        prefix=[1, 2],
        cache_batch_size=2,
        max_seq_len=8,
        temperature=0.0,
        max_tokens=4,
    )
    state.active[0] = True
    state.per_row_limits[0] = 2
    state.generated_tokens[0] = 1
    state.seq_lens[0] = 3
    state.next_token_tensor[0] = 4
    state.row_request_ids[0] = "0"

    engine._execute_persistent_prompt_list_step_payload(
        {
            "op": "persistent_prompt_list_step",
            "step": 1,
            "decode_request_ids": ["0"],
            "decode_rows": [0],
            "prefill": [],
            "prefill_groups": [],
            "finished_after_prefill": [],
        },
        state,
        temperature=0.0,
    )
    cache = state.cache

    assert state.ephemeral_graph_scope is True
    assert getattr(cache, "_torchinferno_ephemeral_ragged_graph_scope") is True

    engine._close_persistent_prompt_list_step_state()

    assert getattr(cache, "_torchinferno_ephemeral_ragged_graph_scope") is False
    assert released == [(model, cache)]


def test_openai_token_budget_step_handlers_manage_state_lifecycle() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model

    state = engine._handle_token_budget_start_payload(
        {
            "op": "token_budget_start",
            "max_active_rows": 1,
            "max_seq_len": 8,
            "temperature": 0.0,
        }
    )
    result = engine._handle_token_budget_step_payload(
        {
            "op": "token_budget_step",
            "step": 0,
            "temperature": 0.0,
            "chunks": [
                {
                    "request_id": "0",
                    "row": 0,
                    "kind": "prefill",
                    "start_token": 0,
                    "token_count": 1,
                    "prompt_complete": True,
                    "emits_token": True,
                    "prompt_chunk": [7],
                }
            ],
            "finished_request_ids": ["0"],
        }
    )

    assert engine._token_budget_step_state is state
    assert engine._token_budget_step_last_result is result
    assert result.prefill_tokens == {"0": 8}
    assert result.finished_request_ids == ("0",)

    engine._handle_token_budget_close_payload({"op": "token_budget_close"})

    assert engine._token_budget_step_state is None
    assert engine._token_budget_step_last_result is None


def test_openai_token_budget_local_group_runner_streams_results() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2], 2, 0.0, True, first_queue),
        _QueuedGeneration([5], 1, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=4,
        prefill_chunk_size=4,
    )

    assert _queue_items(first_queue) == [3, 4, _GenerationDone()]
    assert _queue_items(second_queue) == [6, _GenerationDone()]
    assert [request.done for request in group] == [True, True]
    assert engine._token_budget_step_state is None
    assert model.forward_calls == [((0,), [[1, 2]]), ((1,), [[5]]), ((0,), [[3]])]
    assert stats.scheduler_steps == 3
    assert stats.step_commands == 2
    assert stats.decode_run_commands == 0
    assert stats.empty_plans == 1
    assert stats.emitted_tokens == 3
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_token_budget_local_group_runner_uses_shared_prefix_batched_prefill() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3], 1, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 5], 1, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=4,
        prefix_tokens=2,
    )

    assert _queue_items(first_queue) == [4, _GenerationDone()]
    assert _queue_items(second_queue) == [6, _GenerationDone()]
    assert [request.done for request in group] == [True, True]
    assert engine._token_budget_step_state is None
    assert model.forward_calls == [((0,), [[1, 2]]), ((0, 1), [[3], [5]])]
    assert stats.scheduler_steps == 2
    assert stats.step_commands == 1
    assert stats.decode_run_commands == 0
    assert stats.empty_plans == 1
    assert stats.emitted_tokens == 2
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_token_budget_prompt_list_payload_binds_fast_shared_prefix_prefill() -> None:
    queues = [queue.Queue() for _ in range(2)]
    group = [
        _QueuedGeneration([1, 2, 3, 4], 2, 0.0, True, queues[0]),
        _QueuedGeneration([1, 2, 3, 5, 6], 3, 0.0, True, queues[1]),
    ]
    request_by_id = {str(index): request for index, request in enumerate(group)}
    scheduler = _token_budget_scheduler_for_group(
        group,
        max_active_rows=2,
        max_scheduled_tokens=8,
        prefix_tokens=3,
    )

    payload = _token_budget_prompt_list_step_payload(scheduler.step(), request_by_id)

    assert payload == {
        "op": "persistent_prompt_list_step",
        "step": 0,
        "decode_request_ids": [],
        "decode_rows": [],
        "prefill": [
            {
                "request_id": "0",
                "row": 0,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 2,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [4],
                "prefill_tokens": 1,
            },
            {
                "request_id": "1",
                "row": 1,
                "prompt": [1, 2, 3, 5, 6],
                "max_tokens": 3,
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix": [5, 6],
                "prefill_tokens": 2,
            },
        ],
        "prefill_groups": [
            {
                "request_ids": ["0", "1"],
                "rows": [0, 1],
                "prefix_hit_tokens": 3,
                "prefix": [1, 2, 3],
                "suffix_tokens": [1, 2],
            },
        ],
        "finished_after_prefill": [],
    }


def test_openai_token_budget_prompt_list_payload_allows_partial_prefill() -> None:
    request_by_id = {
        "0": _QueuedGeneration([1, 2, 3, 4], 2, 0.0, True, queue.Queue()),
    }
    plan = TokenBudgetPlan(
        step=0,
        chunks=(
            TokenBudgetScheduledChunk(
                "0",
                row=0,
                kind="prefill",
                start_token=0,
                token_count=2,
                prompt_complete=False,
                emits_token=False,
            ),
        ),
        finished_request_ids=(),
    )

    payload = _token_budget_prompt_list_step_payload(plan, request_by_id)

    assert payload == {
        "op": "persistent_prompt_list_step",
        "step": 0,
        "decode_request_ids": [],
        "decode_rows": [],
        "prefill": [
            {
                "request_id": "0",
                "row": 0,
                "prompt": [1, 2, 3, 4],
                "max_tokens": 2,
                "prefix_hit_tokens": 0,
                "start_token": 0,
                "prefix": [],
                "suffix": [1, 2],
                "prompt_chunk": [1, 2],
                "prefill_tokens": 2,
                "prompt_complete": False,
                "emits_token": False,
            }
        ],
        "prefill_groups": [],
        "finished_after_prefill": [],
    }


def test_openai_token_budget_prompt_list_group_runner_uses_fast_ragged_decode() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 3, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=16,
        prefix_tokens=3,
        decode_run_steps=2,
    )

    assert _queue_items(first_queue) == [5, 21, 22, _GenerationDone()]
    assert _queue_items(second_queue) == [6, 21, 22, _GenerationDone()]
    assert [request.done for request in group] == [True, True]
    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None
    assert model.forward_calls == [
        ((0,), [[1, 2, 3]]),
        ((0, 1), [[4], [5]]),
    ]
    assert model.ragged_calls == [
        ([[5], [6]], [4, 4], None),
        ([[21], [21]], [5, 5], None),
    ]
    assert stats.scheduler_steps == 4
    assert stats.step_commands == 1
    assert stats.decode_run_commands == 1
    assert stats.empty_plans == 1
    assert stats.decode_steps == 2
    assert stats.max_decode_run_steps == 2
    assert stats.prefill_admissions == 2
    assert stats.emitted_tokens == 6
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_token_budget_prompt_list_group_runner_mixes_decode_with_late_admission() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 1, 0.0, True, second_queue),
    ]
    payloads: list[dict[str, object]] = []
    original_handle = engine._handle_persistent_prompt_list_step_payload

    def handle(payload: dict[str, object]):
        payloads.append(dict(payload))
        return original_handle(payload)

    engine._handle_persistent_prompt_list_step_payload = handle  # type: ignore[method-assign]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=16,
        prefix_tokens=3,
        arrival_steps=[0, 1],
    )

    assert _queue_items(first_queue) == [5, 21, 22, _GenerationDone()]
    assert _queue_items(second_queue) == [6, _GenerationDone()]
    assert any(
        payload.get("decode_request_ids") == ["0"]
        and [item["request_id"] for item in payload.get("prefill", [])] == ["1"]
        for payload in payloads
    )
    assert stats.prefill_admissions == 2
    assert stats.decode_steps == 2
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_token_budget_prompt_list_run_payload_executes_local_schedule() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model

    stats = engine._handle_token_budget_prompt_list_run_payload(
        {
            "op": "token_budget_prompt_list_run",
            "input_id_lists": [[1, 2, 3, 4], [1, 2, 3, 5]],
            "max_tokens": 3,
            "row_max_tokens": [3, 1],
            "temperature": 0.0,
            "prefix_tokens": 3,
            "max_active_rows": 2,
            "max_scheduled_tokens": 16,
            "decode_run_steps": 2,
            "arrival_steps": [0, 1],
        }
    )

    assert engine._persistent_prompt_list_step_state is None
    assert model.forward_calls[:2] == [
        ((0,), [[1, 2, 3]]),
        ((0,), [[4]]),
    ]
    assert stats.prefill_admissions == 2
    assert stats.decode_steps == 2
    assert stats.emitted_tokens == 4
    assert stats.finished_events == 2
    assert stats.closed


def test_openai_token_budget_prompt_list_group_runner_chunks_prefill_before_emit() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    response_queue: queue.Queue[object] = queue.Queue()
    group = [_QueuedGeneration([1, 2, 3, 4, 5], 2, 0.0, True, response_queue)]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=1,
        max_scheduled_tokens=1,
        prefill_chunk_size=1,
        prefix_tokens=2,
    )

    assert _queue_items(response_queue) == [6, 21, _GenerationDone()]
    assert group[0].done
    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None
    assert model.forward_calls == [
        ((0,), [[1, 2]]),
        ((0,), [[3]]),
        ((0,), [[4]]),
        ((0,), [[5]]),
    ]
    assert model.ragged_calls == [([[6]], [5], None)]
    assert stats.scheduler_steps == 5
    assert stats.step_commands == 4
    assert stats.decode_run_commands == 0
    assert stats.empty_plans == 1
    assert stats.decode_steps == 1
    assert stats.prefill_admissions == 3
    assert stats.emitted_tokens == 2
    assert stats.finished_events == 1
    assert stats.closed


def test_openai_token_budget_prompt_list_group_runner_long32_progress_shape() -> None:
    class LongOutputModel(_RowTargetPrefillModel):
        config = type("Config", (), {"vocab_size": 128})()

    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = LongOutputModel()
    engine.model = model
    prefix = list(range(1, 9))
    response_queues = [queue.Queue() for _ in range(32)]
    group = [
        _QueuedGeneration([*prefix, 16 + index], 36, 0.0, True, response_queues[index])
        for index in range(32)
    ]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=64,
        max_scheduled_tokens=8192,
        prefix_tokens=len(prefix),
        decode_run_steps=32,
    )

    assert all(request.done for request in group)
    assert all(len(_queue_items(response_queue)) == 37 for response_queue in response_queues)
    assert engine._persistent_prompt_list_step_state is None
    assert model.forward_calls[:2] == [
        ((0,), [prefix]),
        (tuple(range(32)), [[16 + index] for index in range(32)]),
    ]
    assert len(model.ragged_calls) == 35
    assert stats.scheduler_steps == 37
    assert stats.step_commands == 1
    assert stats.decode_run_commands == 2
    assert stats.empty_plans == 1
    assert stats.decode_steps == 35
    assert stats.max_decode_run_steps == 32
    assert stats.prefill_admissions == 32
    assert stats.emitted_tokens == 32 * 36
    assert stats.finished_events == 32
    assert stats.closed


def test_openai_token_budget_prompt_list_group_runner_decodes_logical_rows_when_overallocated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_RAGGED_DECODE_BUCKET_MIN_STEP", "1")
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 4, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 4, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=4,
        max_scheduled_tokens=16,
        prefix_tokens=3,
        decode_run_steps=4,
        static_graph_buckets=False,
    )

    assert stats.decode_steps == 3
    assert model.ragged_calls
    assert all(len(input_rows) == 2 for input_rows, _seq_lens, _row_indices in model.ragged_calls)
    assert all(len(seq_lens) == 2 for _input_rows, seq_lens, _row_indices in model.ragged_calls)
    assert all(row_indices is None for _input_rows, _seq_lens, row_indices in model.ragged_calls)


def test_openai_token_budget_prompt_list_group_runner_broadcasts_decode_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()
    calls: list[tuple[str, dict[str, object]]] = []
    syncs: list[str] = []

    def start(model: object, **kwargs: object) -> None:
        del model
        calls.append(("start", dict(kwargs)))

    def step(model: object, payload: dict[str, object]) -> None:
        del model
        calls.append(("step", dict(payload)))

    def decode_run(model: object, payload: dict[str, object]) -> None:
        del model
        calls.append(("decode_run", dict(payload)))

    def close(model: object) -> None:
        del model
        calls.append(("close", {}))

    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_start", start)
    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_step", step)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_decode_run",
        decode_run,
    )
    monkeypatch.setattr("torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_close", close)
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, first_queue),
        _QueuedGeneration([1, 2, 3, 5], 3, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_prompt_list_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=16,
        prefix_tokens=3,
        decode_run_steps=2,
        broadcast_tensor_parallel=True,
        sync_tensor_parallel=True,
    )

    assert [name for name, _payload in calls] == ["start", "step", "decode_run", "close"]
    assert calls[0][1] == {
        "prefix": (1, 2, 3),
        "cache_batch_size": 2,
        "max_seq_len": 7,
        "temperature": 0.0,
        "max_tokens": 3,
    }
    assert calls[1][1]["op"] == "persistent_prompt_list_step"
    assert calls[1][1]["step"] == 0
    assert calls[2][1] == {
        "op": "persistent_prompt_list_decode_run",
        "start_step": 1,
        "step_count": 2,
        "temperature": 0.0,
    }
    assert syncs == ["sync", "sync", "sync", "sync"]
    assert stats.decode_run_commands == 1
    assert stats.closed


def test_openai_token_budget_prompt_list_group_runner_broadcast_closes_on_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()
    calls: list[str] = []
    syncs: list[str] = []

    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_start",
        lambda model, **kwargs: calls.append("start"),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_step",
        lambda model, payload: calls.append("step"),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_decode_run",
        lambda model, payload: calls.append("decode_run"),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_persistent_prompt_list_close",
        lambda model: calls.append("close"),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )
    group = [_QueuedGeneration([1, 2, 3, 4], 3, 0.0, True, queue.Queue())]

    with pytest.raises(RuntimeError, match="token-budget prompt-list local runner exceeded max_scheduler_steps"):
        engine._run_token_budget_prompt_list_group_local(
            group,
            max_active_rows=1,
            max_scheduled_tokens=16,
            prefix_tokens=3,
            broadcast_tensor_parallel=True,
            sync_tensor_parallel=True,
            max_scheduler_steps=0,
        )

    assert calls == ["start", "close"]
    assert syncs == ["sync", "sync"]
    assert engine._persistent_prompt_list_step_state is None
    assert engine._persistent_prompt_list_step_last_result is None


def test_openai_token_budget_local_group_runner_coalesces_decode_run() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    model = _RowTargetPrefillModel()
    engine.model = model
    decode_run_payloads: list[dict[str, object]] = []
    original_handle = engine._handle_token_budget_decode_run_payload

    def handle_decode_run(payload: dict[str, object]):
        decode_run_payloads.append(dict(payload))
        return original_handle(payload)

    engine._handle_token_budget_decode_run_payload = handle_decode_run  # type: ignore[method-assign]
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1], 3, 0.0, True, first_queue),
        _QueuedGeneration([5], 3, 0.0, True, second_queue),
    ]

    stats = engine._run_token_budget_group_local(
        group,
        max_active_rows=2,
        max_scheduled_tokens=4,
        decode_run_steps=2,
    )

    assert len(decode_run_payloads) == 1
    assert decode_run_payloads[0]["step_count"] == 2
    assert _queue_items(first_queue) == [2, 21, 22, _GenerationDone()]
    assert _queue_items(second_queue) == [6, 21, 22, _GenerationDone()]
    assert model.ragged_calls == [([[2], [6]], [1, 1], [0, 1]), ([[21], [21]], [2, 2], [0, 1])]
    assert engine._token_budget_step_state is None
    assert stats.scheduler_steps == 4
    assert stats.step_commands == 1
    assert stats.decode_run_commands == 1
    assert stats.empty_plans == 1
    assert stats.emitted_tokens == 6
    assert stats.finished_events == 2
    assert stats.max_decode_run_steps == 2
    assert stats.closed


def test_openai_emit_stream_token_uses_request_generated_count() -> None:
    response_queue: queue.Queue[object] = queue.Queue()
    request = _QueuedGeneration([1, 2], 3, 0.0, True, response_queue)

    _emit_stream_token(request, 10, generated_tokens=1)
    _emit_stream_token(request, 11, generated_tokens=3)

    assert _queue_items(response_queue) == [10, 11, _GenerationDone()]
    assert request.done


def test_openai_emit_stream_token_finishes_on_none_or_over_limit() -> None:
    response_queue: queue.Queue[object] = queue.Queue()
    request = _QueuedGeneration([1, 2], 2, 0.0, True, response_queue)

    _emit_stream_token(request, 10, generated_tokens=3)
    _emit_stream_token(request, 11, generated_tokens=1)

    assert _queue_items(response_queue) == [_GenerationDone()]
    assert request.done


def test_openai_stream_row_state_tracks_mid_run_admission() -> None:
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 3, 0.0, True, first_queue)
    second = _QueuedGeneration([3], 2, 0.0, True, second_queue)
    rows = _StreamRowState()

    rows.admit("first", 0, first)
    assert not rows.emit("first", 10)
    rows.admit("second", 1, second)
    assert rows.generated_tokens("first") == 1
    assert rows.generated_tokens("second") == 0

    assert not rows.emit("second", 20)
    assert not rows.emit("first", 11)
    assert rows.emit("first", 12)
    assert rows.active_request_ids == ("second",)
    assert rows.emit("second", 21)

    assert _queue_items(first_queue) == [10, 11, 12, _GenerationDone()]
    assert _queue_items(second_queue) == [20, 21, _GenerationDone()]
    assert rows.active_request_ids == ()


def test_openai_stream_row_state_rejects_occupied_rows() -> None:
    first = _QueuedGeneration([1], 2, 0.0, True, queue.Queue())
    second = _QueuedGeneration([2], 2, 0.0, True, queue.Queue())
    rows = _StreamRowState()

    rows.admit("first", 0, first)

    with pytest.raises(ValueError, match="already occupied"):
        rows.admit("second", 0, second)


def test_openai_persistent_prompt_stream_emit_uses_row_state_for_mid_run_admission() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 3, 0.0, True, first_queue)
    second = _QueuedGeneration([3, 4], 1, 0.0, True, second_queue)
    rows = _StreamRowState()
    group = [first, second]

    assert (
        engine._stream_persistent_prompt_batch_step_result(
            group,
            _PersistentPromptListStepResult(
                decode_tokens={},
                prefill_tokens={"0": 10},
                finished_request_ids=(),
            ),
            payload={"decode_request_ids": [], "decode_rows": [], "prefill": [{"request_id": "0", "row": 0}]},
            stream_rows=rows,
        )
        == ()
    )
    assert rows.generated_tokens("0") == 1

    assert engine._stream_persistent_prompt_batch_step_result(
        group,
        _PersistentPromptListStepResult(
            decode_tokens={"0": 11},
            prefill_tokens={"1": 20},
            finished_request_ids=(),
        ),
        payload={
            "decode_request_ids": ["0"],
            "decode_rows": [0],
            "prefill": [{"request_id": "1", "row": 1}],
        },
        stream_rows=rows,
    ) == ("1",)
    assert rows.active_request_ids == ("0",)

    assert engine._stream_persistent_prompt_batch_step_result(
        group,
        _PersistentPromptListStepResult(
            decode_tokens={"0": 12},
            prefill_tokens={},
            finished_request_ids=("0",),
        ),
        payload={"decode_request_ids": ["0"], "decode_rows": [0], "prefill": []},
        stream_rows=rows,
    ) == ("0",)

    assert _queue_items(first_queue) == [10, 11, 12, _GenerationDone()]
    assert _queue_items(second_queue) == [20, _GenerationDone()]
    assert rows.active_request_ids == ()


def test_openai_token_budget_local_group_runner_step_guard_closes_state() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset()
    engine.model = _RowTargetPrefillModel()
    first_queue: queue.Queue[object] = queue.Queue()
    group = [_QueuedGeneration([1], 3, 0.0, True, first_queue)]

    with pytest.raises(RuntimeError, match="max_scheduler_steps"):
        engine._run_token_budget_group_local(
            group,
            max_active_rows=1,
            max_scheduled_tokens=1,
            max_scheduler_steps=1,
        )

    assert engine._token_budget_step_state is None
    assert _queue_items(first_queue) == [2]


def test_openai_token_budget_local_group_runner_releases_stop_finished_rows() -> None:
    engine = _cache_only_engine()
    engine.stop_token_ids = frozenset({3})
    model = _RowTargetPrefillModel()
    engine.model = model
    first_queue: queue.Queue[object] = queue.Queue()
    group = [_QueuedGeneration([1], 5, 0.0, True, first_queue)]

    stats = engine._run_token_budget_group_local(
        group,
        max_active_rows=1,
        max_scheduled_tokens=4,
        decode_run_steps=1,
    )

    assert _queue_items(first_queue) == [2, _GenerationDone()]
    assert group[0].done
    assert engine._token_budget_step_state is None
    assert model.forward_calls == [((0,), [[1]]), ((0,), [[2]])]
    assert stats.scheduler_steps == 3
    assert stats.step_commands == 2
    assert stats.empty_plans == 1
    assert stats.emitted_tokens == 2
    assert stats.finished_events == 1
    assert stats.closed


def test_openai_queue_profile_records_stream_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", "0")
    profile_path = tmp_path / "queue-profile.jsonl"
    monkeypatch.setenv("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", str(profile_path))
    engine = _cache_only_engine()
    engine.model = object()
    engine._shared_prefix_prompt_list_tokens = lambda prompts: 1  # type: ignore[method-assign]

    def generate_prompt_list_batch_steps(
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: list[int] | None = None,
        **_kw: object,
    ):
        del prompts, max_tokens, temperature, broadcast_tensor_parallel, row_max_tokens
        yield [101, 201]
        yield [None, 202]

    engine._generate_prompt_list_batch_steps = generate_prompt_list_batch_steps  # type: ignore[method-assign]

    queued_at_s = time.perf_counter() - 0.01
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration(
            [1, 2],
            1,
            0.0,
            True,
            first_queue,
            queued_at_s=queued_at_s,
            queue_sequence=7,
        ),
        _QueuedGeneration(
            [1, 3, 4],
            2,
            0.0,
            True,
            second_queue,
            queued_at_s=queued_at_s,
            queue_sequence=8,
        ),
    ]

    engine._run_queued_stream_group(group)

    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "stream_group"
    assert record["group_kind"] == "prompt_list"
    assert record["batch_size"] == 2
    assert record["shared_prefix_tokens"] == 1
    assert record["completed_steps"] == 2
    assert record["emitted_tokens"] == 3
    assert record["queue_sequence_min"] == 7
    assert record["queue_sequence_max"] == 8
    assert record["queue_sequence_count"] == 2
    assert record["queue_wait_ms"] >= 0.0
    assert record["run_to_first_emit_ms"] is not None
    assert record["stream_emit_ms"] >= 0.0


def test_openai_queue_profile_records_runtime_engine_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "queue-profile.jsonl"
    monkeypatch.setenv("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", str(profile_path))
    engine = _cache_only_engine()

    class Stats:
        prefill_model_calls = 2
        prefill_batches = 2
        prefill_tokens = 17
        decode_model_calls = 5
        decode_batches = 5
        decode_tokens = 31
        ragged_decode_batches = 4
        ragged_decode_tokens = 29
        decode_graph_hits = 3
        decode_graph_misses = 1
        prefix_reuse_requests = 7
        prefix_reuse_tokens = 53
        queued_requests = 11
        scheduler_steps = 6
        max_model_batch_size = 8
        persistent_cache_rows = 12

    class RuntimeEngine:
        stats = Stats()

    engine._record_runtime_engine_queue_profile(
        "online_batcher",
        RuntimeEngine(),
        submitted_requests=11,
        decode_quantum=4,
    )

    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert records == [
        {
            "decode_quantum": 4,
            "event": "online_batcher",
            "runtime_decode_batches": 5,
            "runtime_decode_graph_hits": 3,
            "runtime_decode_graph_misses": 1,
            "runtime_decode_model_calls": 5,
            "runtime_decode_tokens": 31,
            "runtime_max_model_batch_size": 8,
            "runtime_persistent_cache_rows": 12,
            "runtime_prefill_batches": 2,
            "runtime_prefill_model_calls": 2,
            "runtime_prefill_tokens": 17,
            "runtime_prefix_reuse_requests": 7,
            "runtime_prefix_reuse_tokens": 53,
            "runtime_queued_requests": 11,
            "runtime_ragged_decode_batches": 4,
            "runtime_ragged_decode_tokens": 29,
            "runtime_scheduler_steps": 6,
            "submitted_requests": 11,
        }
    ]


def test_openai_stream_group_sync_policy_uses_emitted_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", "0")
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MAX_STEPS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_SKIP_MIN_EMITTED_TOKENS", raising=False)
    engine = _cache_only_engine()
    engine.model = object()
    engine._shared_prefix_prompt_list_tokens = lambda prompts: 1  # type: ignore[method-assign]

    sync_calls: list[bool | None] = []

    def sync_tensor_parallel_command(model: object, device: torch.device, *, cuda_sync: bool | None = None) -> None:
        del model, device
        sync_calls.append(cuda_sync)

    def generate_prompt_list_batch_steps(
        prompts: list[list[int]],
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
        row_max_tokens: list[int] | None = None,
        **_kw: object,
    ):
        del max_tokens, temperature, broadcast_tensor_parallel, row_max_tokens
        for _ in range(8):
            yield [101 for _ in prompts]

    monkeypatch.setattr("torchinferno.openai_server._sync_tensor_parallel_command", sync_tensor_parallel_command)
    engine._generate_prompt_list_batch_steps = generate_prompt_list_batch_steps  # type: ignore[method-assign]
    group = [
        _QueuedGeneration([1, 2], 8, 0.0, True, queue.Queue())
        for _ in range(64)
    ]

    engine._run_queued_stream_group(group)

    assert sync_calls == [False]


def test_openai_tp_single_stream_group_defaults_to_batch_path(monkeypatch) -> None:
    engine = _cache_only_engine()
    model = object()
    engine.model = model
    calls: list[tuple[list[list[int]], int, float, list[int] | None]] = []

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_SINGLE_PROMPT_LIST_STREAM", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )

    def generate_prompt_list_batch_steps(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("single stream should keep the stable batch path when opt-out")

    def generate_batch_steps(
        input_ids: torch.Tensor,
        *,
        max_tokens: int,
        temperature: float,
        row_max_tokens: list[int] | None = None,
        **_kw: object,
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
    monkeypatch.setenv("TORCHINFERNO_OPENAI_BATCH_EARLY_RESTART", "0")
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
        **_kw: object,
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


def test_openai_stream_group_can_use_runtime_continuous_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    class RuntimeCache:
        def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            *,
            rows: tuple[int, ...] | None = None,
            seq_lens: list[int] | None = None,
        ) -> None:
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            self._rows = tuple(range(batch_size)) if rows is None else rows
            self._seq_lens = [0 for _ in range(batch_size)] if seq_lens is None else seq_lens

        @property
        def seq_len(self) -> int:
            if not self._rows:
                return 0
            return self._seq_lens[self._rows[0]]

        def for_rows(self, rows: tuple[int, ...]) -> "RuntimeCache":
            physical = tuple(self._rows[int(row)] for row in rows)
            return RuntimeCache(len(physical), self.max_seq_len, rows=physical, seq_lens=self._seq_lens)

        def clear_row(self, row: int) -> None:
            self._seq_lens[self._rows[row]] = 0

        def copy_prefix_from(
            self,
            source: "RuntimeCache",
            tokens: int,
            *,
            source_row: int = 0,
            dest_row: int = 0,
        ) -> None:
            del source
            self._seq_lens[self._rows[dest_row]] = tokens

        def advance(self, tokens: int) -> None:
            for row in self._rows:
                self._seq_lens[row] += tokens

    class RuntimeModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"vocab_size": 32})()

        def to(self, device: torch.device) -> "RuntimeModel":
            return self

        def eval(self) -> "RuntimeModel":
            return self

        def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> RuntimeCache:
            del kwargs
            return RuntimeCache(batch_size, max_seq_len)

        def __call__(self, input_ids: torch.Tensor, *, cache: RuntimeCache, use_cache: bool):
            return self.forward(input_ids, cache=cache, use_cache=use_cache)

        def forward(self, input_ids: torch.Tensor, *, cache: RuntimeCache, use_cache: bool):
            del use_cache
            cache.advance(input_ids.size(1))
            next_ids = (input_ids[:, -1] + 1).remainder(self.config.vocab_size)
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[torch.arange(input_ids.size(0)), -1, next_ids] = 1.0
            return logits, cache

    monkeypatch.setenv("TORCHINFERNO_OPENAI_RUNTIME_CONTINUOUS_STREAM", "1")
    monkeypatch.setenv("TORCHINFERNO_CONTINUOUS_PADDED_BATCH_PREFILL", "0")
    engine = _cache_only_engine()
    engine.model = RuntimeModel()
    engine.stop_token_ids = frozenset()

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2], 3, 0.0, True, first_queue),
        _QueuedGeneration([5, 6, 7], 3, 0.0, True, second_queue),
    ]

    engine._run_queued_stream_group(group)

    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert first_items[:3] == [3, 4, 5]
    assert second_items[:3] == [8, 9, 10]
    assert isinstance(first_items[3], _GenerationDone)
    assert isinstance(second_items[3], _GenerationDone)


def test_openai_stream_group_can_drive_tensor_parallel_online_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []
    syncs: list[str] = []
    instances: list[object] = []

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.init = (args, kwargs)
            self.started: int | None = None
            self.submitted: list[object] = []
            self.event_batches = [
                [
                    types.SimpleNamespace(request_id="0", token=101, finished=False),
                    types.SimpleNamespace(request_id="1", token=201, finished=True),
                ],
                [types.SimpleNamespace(request_id="0", token=102, finished=True)],
            ]
            instances.append(self)

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            self.started = max_seq_len

        def submit_online(self, request: object) -> None:
            self.submitted.append(request)

        def has_online_work(self) -> bool:
            return bool(self.event_batches)

        def step_online(self) -> list[object]:
            return self.event_batches.pop(0)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", (prompts, kwargs))),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()

    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    group = [
        _QueuedGeneration([1, 2], 2, 0.0, True, first_queue),
        _QueuedGeneration([3, 4], 1, 0.0, True, second_queue),
    ]

    engine._run_queued_stream_group(group)

    assert len(instances) == 1
    runtime = instances[0]
    assert runtime.started == 4
    assert [(request.prompt, request.max_new_tokens) for request in runtime.submitted] == [
        ((1, 2), 2),
        ((3, 4), 1),
    ]
    assert commands == [
        (
            "start",
            {
                "max_seq_len": 4,
                "max_active_requests": 2,
                "prefix_cache_capacity": 2,
                "prefill_token_budget": None,
                "temperature": 0.0,
                "enable_ragged_decode": True,
                "store_reusable_prefixes": True,
                "store_full_prompt_prefixes": True,
                "max_tokens": 2,
            },
        ),
        ("submit", ([[1, 2], [3, 4]], {"max_tokens": 2, "row_max_tokens": [2, 1], "arrival_step": 0, "eos_token_id": None, "request_id_start": 0})),
        ("step", 1),
        ("step", 1),
        ("close", None),
    ]
    assert syncs == ["sync", "sync", "sync", "sync", "sync"]
    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert first_items[:2] == [101, 102]
    assert second_items[:1] == [201]
    assert isinstance(first_items[2], _GenerationDone)
    assert isinstance(second_items[1], _GenerationDone)


def test_openai_tensor_parallel_online_batcher_drains_ready_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []
    syncs: list[str] = []
    instances: list[object] = []

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.started: int | None = None
            self.pending: list[object] = []
            instances.append(self)

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            self.started = max_seq_len

        def submit_online(self, request: object) -> None:
            self.pending.append(request)

        def has_online_work(self) -> bool:
            return bool(self.pending)

        def step_online(self) -> list[object]:
            events = [
                types.SimpleNamespace(request_id=getattr(request, "request_id"), token=300 + index, finished=True)
                for index, request in enumerate(self.pending)
            ]
            self.pending.clear()
            return events

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", (prompts, kwargs))),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 4
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 1, 0.0, True, first_queue)
    second = _QueuedGeneration([3, 4], 1, 0.0, True, second_queue)
    engine._generation_queue.put(second)

    assert engine._should_use_tensor_parallel_online_batcher(first)
    engine._run_tensor_parallel_online_batcher(first)

    assert len(instances) == 1
    assert instances[0].started == 3
    assert commands == [
        ("start", {"max_seq_len": 3, "max_active_requests": 4, "prefix_cache_capacity": 1, "prefill_token_budget": None, "temperature": 0.0, "enable_ragged_decode": True, "store_reusable_prefixes": True, "store_full_prompt_prefixes": True, "max_tokens": 1}),
        ("submit", ([[1, 2], [3, 4]], {"max_tokens": 1, "row_max_tokens": [1, 1], "arrival_step": 0, "eos_token_id": None, "request_id_start": 0})),
        ("step", 1),
        ("close", None),
    ]
    assert syncs == ["sync", "sync", "sync", "sync"]
    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert first_items[0] == 300
    assert second_items[0] == 301
    assert isinstance(first_items[1], _GenerationDone)
    assert isinstance(second_items[1], _GenerationDone)


def test_openai_tensor_parallel_online_batcher_uses_queued_limit_for_default_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.pending: list[object] = []

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            del max_seq_len

        def submit_online(self, request: object) -> None:
            self.pending.append(request)

        def has_online_work(self) -> bool:
            return bool(self.pending)

        def step_online(self) -> list[object]:
            events = [
                types.SimpleNamespace(request_id=getattr(request, "request_id"), token=500 + index, finished=True)
                for index, request in enumerate(self.pending)
            ]
            self.pending.clear()
            return events

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", (prompts, kwargs))),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: None,
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 128
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1] * 120, 44, 0.0, True, first_queue)

    # Pin the BASE admission sizing; the KV-bounded concurrency boost (which would
    # raise max_active for this short 164-token seq) has its own test below.
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", "0")
    engine._run_tensor_parallel_online_batcher(first)

    assert commands[0] == (
        "start",
        {
            "max_seq_len": 164,
            "max_active_requests": 48,
            "prefix_cache_capacity": 1,
            "prefill_token_budget": None,
            "temperature": 0.0,
            "enable_ragged_decode": True,
            "store_reusable_prefixes": True,
            "store_full_prompt_prefixes": True,
            "max_tokens": 44,
        },
    )
    assert first_queue.get_nowait() == 500
    assert isinstance(first_queue.get_nowait(), _GenerationDone)


def test_openai_tensor_parallel_online_batcher_boost_uses_admitted_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 8, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.pending: list[object] = []

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            del max_seq_len, external_cache

        def submit_online(self, request: object) -> None:
            self.pending.append(request)

        def has_online_work(self) -> bool:
            return bool(self.pending)

        def step_online(self) -> list[object]:
            events = [
                types.SimpleNamespace(request_id=getattr(request, "request_id"), token=700 + index, finished=True)
                for index, request in enumerate(self.pending)
            ]
            self.pending.clear()
            return events

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN", "311")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_MAX_ACTIVE_CAP", "128")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._effective_openai_max_batch_size",
        lambda *args, **kwargs: 256,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", (prompts, kwargs))),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: None,
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 256
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1] * 36, 64, 0.0, True, first_queue)
    too_long = _QueuedGeneration([2] * 100, 256, 0.0, True, queue.Queue())
    engine._generation_queue.put(too_long)

    engine._run_tensor_parallel_online_batcher(first)

    assert commands[0] == (
        "start",
        {
            "max_seq_len": 311,
            "max_active_requests": 105,
            "prefix_cache_capacity": 1,
            "prefill_token_budget": None,
            "temperature": 0.0,
            "enable_ragged_decode": True,
            "store_reusable_prefixes": True,
            "store_full_prompt_prefixes": True,
            "max_tokens": 64,
        },
    )
    assert commands[1] == (
        "submit",
        (
            [[1] * 36],
            {
                "max_tokens": 64,
                "row_max_tokens": [64],
                "arrival_step": 0,
                "eos_token_id": None,
                "request_id_start": 0,
            },
        ),
    )
    assert first_queue.get_nowait() == 700
    assert isinstance(first_queue.get_nowait(), _GenerationDone)
    assert engine._generation_queue.get_nowait() is too_long


def test_openai_tensor_parallel_online_batcher_records_profile_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = tmp_path / "queue-profile.jsonl"
    monkeypatch.setenv("TORCHINFERNO_OPENAI_QUEUE_PROFILE_JSONL", str(profile_path))
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PROFILE_SNAPSHOT_COMMANDS", "2")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.request_id: str | None = None
            self.steps = 0
            self.stats = types.SimpleNamespace(
                decode_tokens=0,
                scheduler_steps=0,
                queued_requests=0,
            )

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            del max_seq_len, external_cache

        def submit_online(self, request: object) -> None:
            self.request_id = str(getattr(request, "request_id"))
            self.stats.queued_requests += 1

        def has_online_work(self) -> bool:
            return self.request_id is not None

        def step_online(self) -> list[object]:
            assert self.request_id is not None
            self.steps += 1
            self.stats.scheduler_steps += 1
            self.stats.decode_tokens += 1
            finished = self.steps >= 3
            event = types.SimpleNamespace(
                request_id=self.request_id,
                token=800 + self.steps,
                finished=finished,
            )
            if finished:
                self.request_id = None
            return [event]

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: None,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: None,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: None,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: None,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: None,
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 4
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 3, 0.0, True, first_queue)

    engine._run_tensor_parallel_online_batcher(first)

    assert _queue_items(first_queue) == [801, 802, 803, _GenerationDone()]
    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "online_batcher_progress",
        "online_batcher_progress",
        "online_batcher",
    ]
    assert records[0]["profile_snapshot_index"] == 1
    assert records[0]["online_step_commands"] == 2
    assert records[0]["runtime_decode_tokens"] == 2
    assert records[1]["profile_snapshot_index"] == 2
    assert records[1]["online_step_commands"] == 3
    assert records[1]["runtime_decode_tokens"] == 3
    assert records[2]["profile_snapshots"] == 2
    assert records[2]["online_step_commands"] == 3
    assert records[2]["runtime_decode_tokens"] == 3


def test_openai_tensor_parallel_online_default_prefix_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _cache_only_engine()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", raising=False)
    assert engine._online_serving_prefix_rows() == 64
    assert engine._online_serving_effective_prefix_rows(48) == 64
    assert engine._online_serving_effective_prefix_rows(128) == 16

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "7")
    assert engine._online_serving_prefix_rows() == 7
    assert engine._online_serving_effective_prefix_rows(140) == 4

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_TOTAL_ROWS_BUDGET", "0")
    assert engine._online_serving_effective_prefix_rows(140) == 7


def test_kv_bounded_concurrency_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    model = type("FakeTPModel", (), {"world_size": 8, "rank": 0})()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._effective_openai_max_batch_size",
        lambda *args, **kwargs: 256,
    )
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.max_batch_size = 256
    # Disabled: cap == base (no boosting).
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", "0")
    base = engine._online_serving_max_active()
    assert engine._kv_bounded_concurrency_cap() == base
    # Enabled: cap rises to the configured ceiling (clamped by effective batch).
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_BOUNDED_CONCURRENCY", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_KV_MAX_ACTIVE_CAP", "128")
    assert engine._kv_bounded_concurrency_cap() == 128
    assert engine._kv_bounded_concurrency_cap() >= base


def test_openai_tensor_parallel_long_prompt_short_stream_cap_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = type("FakeTPModel", (), {})()
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.max_batch_size = 128
    request = _QueuedGeneration([1] * 120, 44, 0.0, True, queue.Queue())

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_DETERMINISTIC_SHORT_STREAM_HIGH_TOKEN_MIN", "1")
    assert engine._queued_batch_limit(request) == 64
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_LONG_PROMPT_SHORT_STREAM_BATCH_CAP", "1")
    assert engine._queued_batch_limit(request) == 56


def test_openai_tensor_parallel_online_default_max_seq_len_adds_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _cache_only_engine()
    first = _QueuedGeneration([1, 2], 3, 0.0, True, queue.Queue())
    second = _QueuedGeneration([1, 2, 3, 4], 5, 0.0, True, queue.Queue())

    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", raising=False)
    assert engine._tp_online_default_max_seq_len([first, second]) == 9

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "128")
    assert engine._tp_online_default_max_seq_len([first, second]) == 137

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "4")
    assert engine._tp_online_default_max_seq_len([first, second]) == 13

    engine.max_model_len = 10
    assert engine._tp_online_default_max_seq_len([first, second]) == 10


def test_openai_tensor_parallel_online_batcher_sizes_cache_from_initial_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []
    started: list[int] = []

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.pending: list[object] = []

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            started.append(max_seq_len)

        def submit_online(self, request: object) -> None:
            self.pending.append(request)

        def has_online_work(self) -> bool:
            return bool(self.pending)

        def step_online(self) -> list[object]:
            events = [
                types.SimpleNamespace(request_id=getattr(request, "request_id"), token=600 + index, finished=True)
                for index, request in enumerate(self.pending)
            ]
            self.pending.clear()
            return events

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._tensor_parallel_symm_mem_allreduce_scope",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", (prompts, kwargs))),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: None,
    )

    engine = _cache_only_engine()
    engine.model = model
    engine.device = torch.device("cuda")
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 4
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 1, 0.0, True, first_queue)
    second = _QueuedGeneration([3, 4, 5, 6], 2, 0.0, True, second_queue)
    engine._generation_queue.put(second)

    engine._run_tensor_parallel_online_batcher(first)

    assert started == [6]
    assert commands[:2] == [
        ("start", {"max_seq_len": 6, "max_active_requests": 4, "prefix_cache_capacity": 1, "prefill_token_budget": None, "temperature": 0.0, "enable_ragged_decode": True, "store_reusable_prefixes": True, "store_full_prompt_prefixes": True, "max_tokens": 2}),
        ("submit", ([[1, 2], [3, 4, 5, 6]], {"max_tokens": 2, "row_max_tokens": [1, 2], "arrival_step": 0, "eos_token_id": None, "request_id_start": 0})),
    ]
    assert first_queue.get_nowait() == 600
    assert isinstance(first_queue.get_nowait(), _GenerationDone)
    assert second_queue.get_nowait() == 601
    assert isinstance(second_queue.get_nowait(), _GenerationDone)


def test_openai_tensor_parallel_online_batcher_drains_after_short_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PERSISTENT", "0")
    model = type("FakeTPModel", (), {"world_size": 2, "rank": 0, "allocate_cache": lambda self: None})()
    commands: list[tuple[str, object]] = []
    syncs: list[str] = []

    engine = _cache_only_engine()
    engine.model = model
    engine.stop_token_ids = frozenset()
    engine.max_batch_size = 4
    engine._generation_queue = queue.Queue()
    first_queue: queue.Queue[object] = queue.Queue()
    second_queue: queue.Queue[object] = queue.Queue()
    first = _QueuedGeneration([1, 2], 1, 0.0, True, first_queue)
    second = _QueuedGeneration([3, 4], 1, 0.0, True, second_queue)
    enqueued_after_first_step = False

    class RuntimeEngine:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.pending: list[object] = []

        def start_online(self, *, max_seq_len: int, external_cache: object | None = None) -> None:
            del max_seq_len

        def submit_online(self, request: object) -> None:
            self.pending.append(request)

        def has_online_work(self) -> bool:
            return bool(self.pending)

        def step_online(self) -> list[object]:
            nonlocal enqueued_after_first_step
            events = [
                types.SimpleNamespace(request_id=getattr(request, "request_id"), token=400 + index, finished=True)
                for index, request in enumerate(self.pending)
            ]
            self.pending.clear()
            if not enqueued_after_first_step:
                enqueued_after_first_step = True
                engine._generation_queue.put(second)
            return events

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_CONTINUOUS_BATCHER", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_DECODE_QUANTUM", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_INITIAL_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_IDLE_BATCH_WAIT_MS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFIX_ROWS", "1")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_MAX_SEQ_LEN_HEADROOM_TOKENS", "0")
    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_ONLINE_PREFILL_TOKEN_BUDGET", "0")
    monkeypatch.setattr(
        "torchinferno.openai_server._is_tensor_parallel_model",
        lambda candidate: candidate is model,
    )
    monkeypatch.setattr("torchinferno.openai_server._RuntimeContinuousBatchEngine", RuntimeEngine)
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_start",
        lambda model, **kwargs: commands.append(("start", kwargs)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_submit_prompt_lists",
        lambda model, prompts, **kwargs: commands.append(("submit", prompts)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_step",
        lambda model, steps=1: commands.append(("step", steps)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._broadcast_tensor_parallel_online_close",
        lambda model: commands.append(("close", None)),
    )
    monkeypatch.setattr(
        "torchinferno.openai_server._sync_tensor_parallel_command",
        lambda model, device, **kwargs: syncs.append("sync"),
    )

    engine._run_tensor_parallel_online_batcher(first)

    assert commands == [
        ("start", {"max_seq_len": 3, "max_active_requests": 4, "prefix_cache_capacity": 1, "prefill_token_budget": None, "temperature": 0.0, "enable_ragged_decode": True, "store_reusable_prefixes": True, "store_full_prompt_prefixes": True, "max_tokens": 1}),
        ("submit", [[1, 2]]),
        ("step", 1),
        ("submit", [[3, 4]]),
        ("step", 1),
        ("close", None),
    ]
    assert len(syncs) >= 5
    first_items = _queue_items(first_queue)
    second_items = _queue_items(second_queue)
    assert first_items[0] == 400
    assert second_items[0] == 400
    assert isinstance(first_items[1], _GenerationDone)
    assert isinstance(second_items[1], _GenerationDone)


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
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MAX_STEPS", raising=False)
    monkeypatch.delenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_SKIP_MIN_EMITTED_TOKENS", raising=False)

    assert not _tp_command_cuda_sync_for_steps(0)
    assert not _tp_command_cuda_sync_for_steps(7)
    assert _tp_command_cuda_sync_for_steps(8)
    assert _tp_command_cuda_sync_for_steps(33)
    assert not _tp_command_cuda_sync_for_steps(33, emitted_tokens=512)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MIN_STEPS", "3")
    assert not _tp_command_cuda_sync_for_steps(2)
    assert _tp_command_cuda_sync_for_steps(3)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_SKIP_MIN_EMITTED_TOKENS", "10")
    assert _tp_command_cuda_sync_for_steps(4, emitted_tokens=9)
    assert not _tp_command_cuda_sync_for_steps(4, emitted_tokens=10)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC_MAX_STEPS", "4")
    assert _tp_command_cuda_sync_for_steps(4)
    assert not _tp_command_cuda_sync_for_steps(5)

    monkeypatch.setenv("TORCHINFERNO_OPENAI_TP_COMMAND_CUDA_SYNC", "1")
    assert _tp_command_cuda_sync_for_steps(64, emitted_tokens=512)


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
        prefill_selected_logits_graph_calls = 0
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

        def try_prefill_selected_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.prefill_selected_logits_graph_calls += 1
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
    assert _try_prefill_selected_logits_graph(
        model,
        input_ids,
        cache,
        logit_positions=torch.tensor([0]),
    ) is None
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
    assert model.prefill_selected_logits_graph_calls == 1
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

        def try_prefill_selected_logits_graph(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("prefill selected logits graph should be disabled for TP serving")

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
    assert _try_prefill_selected_logits_graph(
        model,
        input_ids,
        cache,
        logit_positions=torch.tensor([0]),
    ) is None
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
        self.allocated_shapes: list[tuple[int, int]] = []
        self.cache_allocations = 0

    def allocate_cache(self, batch_size: int, max_seq_len: int, **kwargs) -> _BatchRecordingCache:
        self.allocated_shapes.append((batch_size, max_seq_len))
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
        **_kw: object,
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
        **_kw: object,
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


class _SelectedTokenEchoSharedPrefixRecordingModel(_TokenEchoSharedPrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.selected_logit_positions: list[list[int]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: _PrefixRecordingCache,
        use_cache: bool,
        return_last_logits_only: bool = False,
        logit_positions: torch.Tensor | None = None,
    ):
        if logit_positions is None:
            return super().forward(
                input_ids,
                cache=cache,
                use_cache=use_cache,
                return_last_logits_only=return_last_logits_only,
            )

        del use_cache, return_last_logits_only
        self.forward_inputs.append([[int(token_id) for token_id in row.tolist()] for row in input_ids])
        positions = [int(position) for position in logit_positions.tolist()]
        self.selected_logit_positions.append(positions)
        layer = cache.layers[0]
        start = layer.seq_len
        end = start + input_ids.size(1)
        layer.keys[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.values[: input_ids.size(0), :, start:end, :].fill_(1)
        layer.seq_len = end
        logits = torch.zeros(input_ids.size(0), 1, self.config.vocab_size)
        for row, offset in enumerate(positions):
            logits[row, 0, int(input_ids[row, offset])] = 1.0
        return logits, cache


class _SelectedGraphTokenEchoSharedPrefixRecordingModel(_SelectedTokenEchoSharedPrefixRecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.selected_graph_capture_flags: list[bool] = []

    def try_prefill_selected_logits_graph(
        self,
        input_ids: torch.Tensor,
        cache: _PrefixRecordingCache,
        *,
        logit_positions: torch.Tensor,
        capture_on_miss: bool = True,
    ) -> torch.Tensor | None:
        self.selected_graph_capture_flags.append(capture_on_miss)
        if not capture_on_miss:
            return None
        logits, _cache = super().forward(
            input_ids,
            cache=cache,
            use_cache=True,
            logit_positions=logit_positions,
        )
        return logits


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
    import threading
    engine = object.__new__(OpenAICompletionEngine)
    engine.cache_backend = "dense"
    engine.page_size = 16
    engine.device = torch.device("cpu")
    engine._cache_pool = {}
    engine._microbatch_cache_pool = {}
    engine._single_prefill_capture_seen = {}
    engine._batched_prefill_capture_seen = {}
    engine._model_lock = threading.Lock()
    return engine


def _llama_tp_cache(
    batch_size: int,
    max_seq_len: int,
    *,
    cache_backend: str = "dense",
    page_size: int = 16,
) -> Llama3TensorParallelCache:
    if cache_backend == "paged":
        layer = PagedLlama3TensorParallelLayerKVCache(
            batch_size,
            max_seq_len,
            1,
            2,
            page_size=page_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    else:
        layer = Llama3TensorParallelLayerKVCache(
            batch_size,
            max_seq_len,
            1,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    return Llama3TensorParallelCache(
        [layer],
        cache_backend=cache_backend,
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
