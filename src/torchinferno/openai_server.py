from __future__ import annotations

import argparse
import inspect
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor

from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, sample_next_token, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.models.llama3_family.pipeline import Llama3PipelineForCausalLM
from torchinferno.models.llama3_family.tensor_parallel import Llama3TensorParallelForCausalLM
from torchinferno.models.llama3_family.v0 import Llama3V0ForCausalLM, tiny_llama3_v0_config
from torchinferno.models.auto import load_model_auto


@dataclass(frozen=True)
class OpenAIServerConfig:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    model_kind: str = "auto"
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    devices: tuple[str, ...] = ()
    device: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    token: str | None = None
    revision: str | None = None
    cache_dir: str | None = None
    cache_backend: str = "dense"
    page_size: int = 16
    max_batch_size: int = 32
    batch_wait_ms: float = 2.0
    llama_parallelism: str = "auto"


@dataclass(frozen=True)
class CompletionResult:
    tokens: list[int]
    prompt_tokens: int


@dataclass
class _QueuedGeneration:
    prompt: list[int]
    max_tokens: int
    temperature: float
    stream: bool
    responses: "queue.Queue[object]"


@dataclass(frozen=True)
class _GenerationDone:
    pass


@dataclass(frozen=True)
class _GenerationResult:
    tokens: list[int]


class _ByteFallbackTokenizer:
    eos_token_id: int | None = None

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = max(2, vocab_size)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        return self.encode(_format_messages(messages))

    def encode(self, text: str) -> list[int]:
        limit = min(self.vocab_size, 256)
        return [min(ord(ch), limit - 1) for ch in text] or [1]

    def decode_token(self, token_id: int) -> str:
        if 32 <= token_id <= 126:
            return chr(token_id)
        return chr(32 + (token_id % 95))

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.decode_token(int(token_id)) for token_id in token_ids)


class _TransformersChatTokenizer:
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def encode_messages(self, messages: list[dict[str, object]]) -> list[int]:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            encoded = apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return _coerce_token_ids(encoded)
        return self.encode(_format_messages(messages))

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        return [int(token_id) for token_id in encoded] or [self.eos_token_id or 0]

    def decode_token(self, token_id: int) -> str:
        return str(self.tokenizer.decode([int(token_id)], skip_special_tokens=True))  # type: ignore[attr-defined]

    def decode(self, token_ids: Iterable[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))  # type: ignore[attr-defined]


def load_chat_tokenizer(
    config: OpenAIServerConfig,
    vocab_size: int,
) -> _ByteFallbackTokenizer | _TransformersChatTokenizer:
    tokenizer_name = config.tokenizer or config.model
    if tokenizer_name in {"byte", "bytes", "fallback", "tiny"} or config.model_kind.startswith("tiny"):
        return _ByteFallbackTokenizer(vocab_size)
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "transformers is required for OpenAI-compatible text serving. "
            "Install TorchInferno with the 'serve' extra or pass --tokenizer byte for smoke tests."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
        token=config.token,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )
    return _TransformersChatTokenizer(tokenizer)


def _coerce_token_ids(encoded: object) -> list[int]:
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, Mapping):
        input_ids = encoded.get("input_ids")
    if input_ids is not None:
        encoded = input_ids
    if isinstance(encoded, Tensor):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist") and not isinstance(encoded, (list, tuple, str, bytes)):
        encoded = encoded.tolist()  # type: ignore[assignment]
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and _is_token_sequence(encoded[0]):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]  # type: ignore[union-attr]


def _is_token_sequence(value: object) -> bool:
    if isinstance(value, Tensor):
        return value.ndim == 1
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, str, bytes)):
        value = value.tolist()
    return isinstance(value, (list, tuple))


class OpenAICompletionEngine:
    def __init__(
        self,
        model: object,
        tokenizer: _ByteFallbackTokenizer | _TransformersChatTokenizer,
        *,
        model_id: str,
        device: torch.device,
        cache_backend: str = "dense",
        page_size: int = 16,
        max_model_len: int | None = None,
        max_batch_size: int = 32,
        batch_wait_ms: float = 2.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.max_model_len = max_model_len
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_s = max(0.0, batch_wait_ms / 1000.0)
        self._generation_queue: "queue.Queue[_QueuedGeneration | None]" = queue.Queue()
        self._closed = False
        self._worker: threading.Thread | None = None
        if not _is_tensor_parallel_worker_model(model):
            self._worker = threading.Thread(target=self._batch_worker, name="torchinferno-openai-batcher", daemon=True)
            self._worker.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._generation_queue.put(None)
            self._worker.join(timeout=10)
        _broadcast_tensor_parallel_stop(self.model)

    def generate_chat_tokens(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[int]:
        prompt = self._encode_chat_prompt(messages, max_tokens=max_tokens)
        yield from self._submit_generation(prompt, max_tokens=max_tokens, temperature=temperature)

    def complete_chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        prompt = self._encode_chat_prompt(messages, max_tokens=max_tokens)
        tokens = self._submit_completion(prompt, max_tokens=max_tokens, temperature=temperature)
        return CompletionResult(tokens=tokens, prompt_tokens=len(prompt))

    def _encode_chat_prompt(self, messages: list[dict[str, object]], *, max_tokens: int) -> list[int]:
        prompt = self.tokenizer.encode_messages(messages)
        if self.max_model_len is not None and len(prompt) + max_tokens > self.max_model_len:
            prompt_budget = max(1, self.max_model_len - max_tokens)
            prompt = prompt[-prompt_budget:]
        return prompt

    def _submit_generation(self, prompt: list[int], *, max_tokens: int, temperature: float) -> Iterator[int]:
        if self._closed:
            raise RuntimeError("OpenAI completion engine is closed")
        responses: "queue.Queue[object]" = queue.Queue()
        self._generation_queue.put(_QueuedGeneration(prompt, max_tokens, temperature, True, responses))
        while True:
            item = responses.get()
            if isinstance(item, _GenerationDone):
                break
            if isinstance(item, BaseException):
                raise item
            yield int(item)

    def _submit_completion(self, prompt: list[int], *, max_tokens: int, temperature: float) -> list[int]:
        if self._closed:
            raise RuntimeError("OpenAI completion engine is closed")
        responses: "queue.Queue[object]" = queue.Queue()
        self._generation_queue.put(_QueuedGeneration(prompt, max_tokens, temperature, False, responses))
        item = responses.get()
        if isinstance(item, BaseException):
            raise item
        if not isinstance(item, _GenerationResult):
            raise RuntimeError(f"unexpected completion response: {item!r}")
        return item.tokens

    def _batch_worker(self) -> None:
        while True:
            first = self._generation_queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                timeout = max(0.0, deadline - time.perf_counter())
                if timeout == 0.0:
                    break
                try:
                    item = self._generation_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    self._generation_queue.put(None)
                    break
                batch.append(item)
            self._run_queued_batch(batch)

    def _run_queued_batch(self, batch: list[_QueuedGeneration]) -> None:
        groups: dict[tuple[int, int, float, bool], list[_QueuedGeneration]] = {}
        for request in batch:
            key = (len(request.prompt), request.max_tokens, request.temperature, request.stream)
            groups.setdefault(key, []).append(request)
        for group in groups.values():
            is_stream = group[0].stream
            try:
                input_ids = torch.tensor([request.prompt for request in group], dtype=torch.long, device=self.device)
                if is_stream:
                    for step_tokens in self._generate_batch_steps(
                        input_ids,
                        max_tokens=group[0].max_tokens,
                        temperature=group[0].temperature,
                    ):
                        for request, token_id in zip(group, step_tokens):
                            if token_id is not None:
                                request.responses.put(int(token_id))
                else:
                    rows = self._generate_batch_tokens(
                        input_ids,
                        max_tokens=group[0].max_tokens,
                        temperature=group[0].temperature,
                    )
                    for request, tokens in zip(group, rows):
                        request.responses.put(_GenerationResult(tokens))
            except BaseException as exc:
                for request in group:
                    request.responses.put(exc)
            finally:
                if is_stream:
                    for request in group:
                        request.responses.put(_GenerationDone())

    @torch.inference_mode()
    def _generate_batch_tokens(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
    ) -> list[list[int]]:
        if max_tokens <= 0:
            return [[] for _ in range(input_ids.size(0))]
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_generate(
                self.model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        eos_token_id = self.tokenizer.eos_token_id
        model = self.model
        if not hasattr(model, "allocate_cache") or not callable(getattr(model, "forward", None)):
            generated = model.generate(  # type: ignore[attr-defined]
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
            rows = generated[:, input_ids.size(1) :].detach().cpu().tolist()
            return _trim_rows_at_eos(rows, eos_token_id)

        cache = _allocate_cache(
            model,
            input_ids.size(0),
            input_ids.size(1) + max_tokens,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
        )
        logits, cache = _forward(model, input_ids, cache)
        next_token = _sample(model, logits[:, -1, :], temperature).to(self.device)
        generated_tokens: list[Tensor] = []
        active = (
            torch.ones(input_ids.size(0), dtype=torch.bool, device=self.device)
            if eos_token_id is not None
            else None
        )
        for step in range(max_tokens):
            generated_tokens.append(next_token[:, None])
            if active is not None:
                active &= next_token != eos_token_id
                # Keep non-stream decode mostly async; exact output is trimmed after the final transfer.
                should_check_eos = (step + 1) % 8 == 0 or step + 1 == max_tokens
                if should_check_eos and not bool(active.any()):
                    break
            if step + 1 == max_tokens:
                break
            logits, cache = _forward(model, next_token[:, None], cache)
            next_token = _sample(model, logits[:, -1, :], temperature).to(self.device)
        rows = torch.cat(generated_tokens, dim=1).detach().cpu().tolist()
        return _trim_rows_at_eos(rows, eos_token_id)

    @torch.inference_mode()
    def _generate_tokens(self, input_ids: Tensor, *, max_tokens: int, temperature: float) -> Iterator[int]:
        for step_tokens in self._generate_batch_steps(input_ids, max_tokens=max_tokens, temperature=temperature):
            token_id = step_tokens[0]
            if token_id is not None:
                yield int(token_id)

    @torch.inference_mode()
    def _generate_batch_steps(
        self,
        input_ids: Tensor,
        *,
        max_tokens: int,
        temperature: float,
        broadcast_tensor_parallel: bool = True,
    ) -> Iterator[list[int | None]]:
        if max_tokens <= 0:
            return
        if broadcast_tensor_parallel:
            _broadcast_tensor_parallel_generate(
                self.model,
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        eos_token_id = self.tokenizer.eos_token_id
        model = self.model
        if not hasattr(model, "allocate_cache") or not callable(getattr(model, "forward", None)):
            generated = model.generate(  # type: ignore[attr-defined]
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
            rows = generated[:, input_ids.size(1) :].detach().cpu().tolist()
            finished = [False for _ in rows]
            for step in range(max_tokens):
                step_tokens: list[int | None] = []
                for row_index, row in enumerate(rows):
                    if finished[row_index] or step >= len(row):
                        step_tokens.append(None)
                        continue
                    token_id = int(row[step])
                    step_tokens.append(token_id)
                    if eos_token_id is not None and token_id == eos_token_id:
                        finished[row_index] = True
                if all(token is None for token in step_tokens):
                    break
                yield step_tokens
            return

        cache = _allocate_cache(
            model,
            input_ids.size(0),
            input_ids.size(1) + max_tokens,
            device=self.device,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
        )
        active = [True for _ in range(input_ids.size(0))]
        logits, cache = _forward(model, input_ids, cache)
        next_token = _sample(model, logits[:, -1, :], temperature).to(self.device)
        for _ in range(max_tokens):
            token_ids = next_token.detach().cpu().tolist()
            step_tokens: list[int | None] = []
            for row, token_id in enumerate(token_ids):
                if not active[row]:
                    step_tokens.append(None)
                    continue
                token_id = int(token_id)
                step_tokens.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    active[row] = False
            yield step_tokens
            if not any(active):
                break
            logits, cache = _forward(model, next_token[:, None], cache)
            next_token = _sample(model, logits[:, -1, :], temperature).to(self.device)


class _OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "TorchInfernoOpenAI/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.engine.model_id,  # type: ignore[attr-defined]
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "torchinferno",
                        }
                    ],
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            payload = self._read_json()
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            max_tokens = int(payload.get("max_tokens", 256))
            temperature = float(payload.get("temperature", 0.0))
            stream = bool(payload.get("stream", False))
            if stream:
                self._stream_chat(messages, max_tokens=max_tokens, temperature=temperature)
            else:
                self._complete_chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            self._send_json({"error": {"message": str(exc), "type": exc.__class__.__name__}}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _complete_chat(self, messages: list[dict[str, object]], *, max_tokens: int, temperature: float) -> None:
        engine: OpenAICompletionEngine = self.server.engine  # type: ignore[attr-defined]
        completion = engine.complete_chat(messages, max_tokens=max_tokens, temperature=temperature)
        content = engine.tokenizer.decode(completion.tokens)
        self._send_json(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": len(completion.tokens),
                    "total_tokens": completion.prompt_tokens + len(completion.tokens),
                },
            }
        )

    def _stream_chat(self, messages: list[dict[str, object]], *, max_tokens: int, temperature: float) -> None:
        engine: OpenAICompletionEngine = self.server.engine  # type: ignore[attr-defined]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._write_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        for token_id in engine.generate_chat_tokens(messages, max_tokens=max_tokens, temperature=temperature):
            content = engine.tokenizer.decode_token(token_id)
            if not content:
                continue
            self._write_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_id,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )
        self._write_sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _send_json(self, payload: dict[str, object], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(self, payload: dict[str, object]) -> None:
        self.wfile.write(b"data: ")
        self.wfile.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()


class _OpenAIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], engine: OpenAICompletionEngine) -> None:
        super().__init__(server_address, _OpenAIHandler)
        self.engine = engine


def build_engine(config: OpenAIServerConfig) -> OpenAICompletionEngine:
    model, device = _load_model(config)
    vocab_size = int(getattr(getattr(model, "config", object()), "vocab_size", 256))
    tokenizer = load_chat_tokenizer(config, vocab_size)
    return OpenAICompletionEngine(
        model,
        tokenizer,
        model_id=config.model,
        device=device,
        cache_backend=config.cache_backend,
        page_size=config.page_size,
        max_model_len=config.max_model_len,
        max_batch_size=config.max_batch_size,
        batch_wait_ms=config.batch_wait_ms,
    )


def serve(config: OpenAIServerConfig) -> None:
    engine = build_engine(config)
    if _is_tensor_parallel_worker_model(engine.model):
        _tensor_parallel_worker_loop(engine)
        return
    server = _OpenAIServer((config.host, config.port), engine)
    print(
        f"TorchInferno OpenAI server listening on http://{config.host}:{server.server_port}/v1 "
        f"model={config.model}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        engine.close()


def _load_model(config: OpenAIServerConfig) -> tuple[object, torch.device]:
    kind = _infer_model_kind(config)
    dtype = _resolve_dtype(config.dtype)
    if kind == "tiny-deepseek":
        device = _primary_device(config)
        model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(max_position_embeddings=config.max_model_len or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "tiny-dsv4":
        device = _primary_device(config)
        model = DSv4ForCausalLM(tiny_dsv4_config(max_seq_len=config.max_model_len or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "tiny-llama3":
        device = _primary_device(config)
        model = Llama3V0ForCausalLM(tiny_llama3_v0_config(max_position_embeddings=config.max_model_len or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "llama3":
        if _llama_parallelism(config) == "tensor":
            model = Llama3TensorParallelForCausalLM.from_pretrained(
                config.model,
                dtype=config.dtype,
                token=config.token,
                revision=config.revision,
                cache_dir=config.cache_dir,
            ).eval()
            return model, model.device
        devices = _server_devices(config)
        model = Llama3PipelineForCausalLM.from_pretrained(
            config.model,
            devices=devices,
            dtype=config.dtype,
            token=config.token,
            revision=config.revision,
            cache_dir=config.cache_dir,
        ).eval()
        return model, torch.device(devices[0])
    device = _primary_device(config)
    model = load_model_auto(
        config.model,
        token=config.token,
        revision=config.revision,
        cache_dir=config.cache_dir,
        map_location=device,
        strict=True,
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval(), device


def _infer_model_kind(config: OpenAIServerConfig) -> str:
    kind = config.model_kind.lower()
    if kind != "auto":
        return kind
    model = config.model.lower()
    if "llama" in model:
        return "llama3"
    path = Path(config.model).expanduser()
    config_path = path / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())
        model_type = str(data.get("model_type", "")).lower()
        if "llama" in model_type:
            return "llama3"
        if "deepseek" in model_type:
            return "deepseek"
        if model_type == "dsv4":
            return "dsv4"
    return "auto"


def _primary_device(config: OpenAIServerConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    devices = _server_devices(config)
    return torch.device(devices[0])


def _server_devices(config: OpenAIServerConfig) -> tuple[str, ...]:
    if config.devices:
        return config.devices
    if config.device:
        return (config.device,)
    if torch.cuda.is_available():
        count = max(1, min(config.tensor_parallel_size, torch.cuda.device_count()))
        return tuple(f"cuda:{idx}" for idx in range(count))
    return ("cpu",)


def _llama_parallelism(config: OpenAIServerConfig) -> str:
    mode = config.llama_parallelism.lower()
    if mode == "pipeline":
        return "pipeline"
    if mode == "tensor":
        return "tensor"
    if mode != "auto":
        raise ValueError(f"unsupported llama parallelism: {config.llama_parallelism}")
    return "tensor" if _distributed_env_requested() else "pipeline"


def _distributed_env_requested() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    normalized = dtype.lower().replace("torch.", "")
    if normalized == "auto":
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def _is_tensor_parallel_model(model: object) -> bool:
    return isinstance(model, Llama3TensorParallelForCausalLM)


def _tensor_parallel_world_size(model: object) -> int:
    return int(getattr(model, "world_size", 1)) if _is_tensor_parallel_model(model) else 1


def _is_tensor_parallel_primary_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) == 0


def _is_tensor_parallel_worker_model(model: object) -> bool:
    return _is_tensor_parallel_model(model) and _tensor_parallel_world_size(model) > 1 and int(getattr(model, "rank", 0)) != 0


def _broadcast_tensor_parallel_generate(
    model: object,
    input_ids: Tensor,
    *,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    command = [
        {
            "op": "generate",
            "input_ids": input_ids.detach().cpu().tolist(),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
        }
    ]
    dist.broadcast_object_list(command, src=0)


def _broadcast_tensor_parallel_stop(model: object) -> None:
    if not _is_tensor_parallel_primary_model(model):
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list([{"op": "stop"}], src=0)


def _tensor_parallel_worker_loop(engine: OpenAICompletionEngine) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("tensor-parallel worker loop requires an initialized process group")
    while True:
        command: list[object] = [None]
        dist.broadcast_object_list(command, src=0)
        payload = command[0]
        if not isinstance(payload, dict):
            continue
        op = payload.get("op")
        if op == "stop":
            return
        if op != "generate":
            raise ValueError(f"unsupported tensor-parallel worker op: {op}")
        input_ids = torch.tensor(payload["input_ids"], dtype=torch.long, device=engine.device)
        if bool(payload.get("stream", True)):
            for _ in engine._generate_batch_steps(
                input_ids,
                max_tokens=int(payload["max_tokens"]),
                temperature=float(payload["temperature"]),
                broadcast_tensor_parallel=False,
            ):
                pass
        else:
            engine._generate_batch_tokens(
                input_ids,
                max_tokens=int(payload["max_tokens"]),
                temperature=float(payload["temperature"]),
                broadcast_tensor_parallel=False,
            )


def _allocate_cache(
    model: object,
    batch_size: int,
    max_seq_len: int,
    *,
    device: torch.device,
    cache_backend: str,
    page_size: int,
) -> object:
    allocate_cache = getattr(model, "allocate_cache")
    try:
        return allocate_cache(
            batch_size,
            max_seq_len,
            device=device,
            cache_backend=cache_backend,
            page_size=page_size,
        )
    except TypeError:
        try:
            return allocate_cache(batch_size, max_seq_len, device=device)
        except TypeError:
            return allocate_cache(batch_size, max_seq_len)


def _forward(model: object, input_ids: Tensor, cache: object) -> tuple[Tensor, object]:
    forward = model.forward  # type: ignore[attr-defined]
    parameters = _forward_parameter_names(type(model))
    kwargs: dict[str, object] = {"cache": cache, "use_cache": True}
    if "return_last_logits_only" in parameters:
        kwargs["return_last_logits_only"] = True
    if _is_tensor_parallel_model(model) and "return_sharded_logits" in parameters:
        kwargs["return_sharded_logits"] = True
    return forward(input_ids, **kwargs)


@lru_cache(maxsize=None)
def _forward_parameter_names(model_type: type) -> frozenset[str]:
    return frozenset(inspect.signature(model_type.forward).parameters)


def _sample(model: object, logits: Tensor, temperature: float) -> Tensor:
    sample = getattr(model, "_sample_next_token", None)
    if sample is not None:
        return sample(logits, temperature)
    return sample_next_token(logits, temperature)


def _trim_rows_at_eos(rows: Sequence[Sequence[int]], eos_token_id: int | None) -> list[list[int]]:
    trimmed_rows: list[list[int]] = []
    for row in rows:
        trimmed: list[int] = []
        for token_id in row:
            token = int(token_id)
            trimmed.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
        trimmed_rows.append(trimmed)
    return trimmed_rows


def _format_messages(messages: list[dict[str, object]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TorchInferno behind an OpenAI-compatible HTTP API.")
    parser.add_argument("--model", required=True, help="Model id or local checkpoint path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-kind", default="auto", help="auto, llama3, deepseek, dsv4, or tiny-* for smoke tests.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer id/path. Use 'byte' for smoke tests.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--devices", default=None, help="Comma-separated device list. Defaults to cuda:0..tp-1.")
    parser.add_argument("--device", default=None, help="Single-device fallback for non-Llama models.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"],
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-backend", choices=["dense", "paged"], default="dense")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument(
        "--llama-parallelism",
        choices=["auto", "pipeline", "tensor"],
        default="auto",
        help="Use pipeline placement by default, or tensor parallel when launched with torchrun.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OpenAIServerConfig:
    devices: Sequence[str] = ()
    if args.devices:
        devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    return OpenAIServerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        model_kind=args.model_kind,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        devices=tuple(devices),
        device=args.device,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        revision=args.revision,
        cache_dir=args.cache_dir,
        cache_backend=args.cache_backend,
        page_size=args.page_size,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        llama_parallelism=args.llama_parallelism,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    serve(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
