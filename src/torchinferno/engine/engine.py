from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchinferno.engine.config import EngineConfig
from torchinferno.engine.loader import load_model_for_engine
from torchinferno.engine.types import GenerateOutput, GenerateRequest, TokenOutput, Usage


@dataclass(frozen=True)
class EngineStats:
    active_requests: int
    cancelled_requests: int
    closed: bool


class InferenceEngine:
    """Stable synchronous generation API over TorchInferno runtime backends."""

    def __init__(self, backend: object, *, config: EngineConfig | None = None) -> None:
        self._backend = backend
        self.config = config
        self.model_id = str(getattr(backend, "model_id", config.model if config is not None else ""))
        self._lock = threading.Lock()
        self._active_requests: set[str] = set()
        self._cancelled_requests: set[str] = set()
        self._closed = False

    @classmethod
    def from_config(cls, config: EngineConfig) -> "InferenceEngine":
        from torchinferno.openai_server import OpenAICompletionEngine, load_chat_tokenizer

        model, device = load_model_for_engine(config)
        vocab_size = int(getattr(getattr(model, "config", object()), "vocab_size", 256))
        tokenizer = load_chat_tokenizer(config, vocab_size)
        backend = OpenAICompletionEngine(
            model,
            tokenizer,
            model_id=config.model,
            device=device,
            cache_backend=config.cache_backend,
            page_size=config.page_size,
            max_model_len=config.max_model_len,
            max_batch_size=config.max_batch_size,
            batch_wait_ms=config.batch_wait_ms,
            single_request_admission_wait_ms=config.single_request_admission_wait_ms,
        )
        return cls(backend, config=config)

    @classmethod
    def from_legacy_backend(
        cls,
        backend: object,
        *,
        config: EngineConfig | None = None,
    ) -> "InferenceEngine":
        return cls(backend, config=config)

    @property
    def backend(self) -> object:
        return self._backend

    @property
    def stats(self) -> EngineStats:
        with self._lock:
            return EngineStats(
                active_requests=len(self._active_requests),
                cancelled_requests=len(self._cancelled_requests),
                closed=self._closed,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def cancel(self, request_id: str) -> None:
        with self._lock:
            self._cancelled_requests.add(request_id)

    def generate(self, request: GenerateRequest) -> GenerateOutput:
        token_ids: list[int] = []
        texts: list[str] = []
        finish_reason = "length"
        usage: Usage | None = None
        for event in self.generate_stream(request):
            if event.token_id is not None:
                token_ids.append(event.token_id)
                texts.append(event.text)
            if event.finished:
                finish_reason = event.finish_reason or finish_reason
                usage = event.usage
        if usage is None:
            usage = Usage(prompt_tokens=self._prompt_token_count(request), completion_tokens=len(token_ids))
        return GenerateOutput(
            request_id=request.request_id,
            token_ids=tuple(token_ids),
            text="".join(texts),
            finish_reason=finish_reason,
            usage=usage,
        )

    def generate_stream(self, request: GenerateRequest) -> Iterator[TokenOutput]:
        self._ensure_open()
        self._enter_request(request.request_id)
        emitted = 0
        finish_reason = "length"
        prompt_tokens = 0
        token_iterator: Iterator[int] | None = None
        try:
            prompt_tokens, token_iterator = self._token_iterator(request)
            for token_id in token_iterator:
                if self._is_cancelled(request.request_id):
                    finish_reason = "cancelled"
                    _close_iterator(token_iterator)
                    break
                token_id = int(token_id)
                if token_id in request.sampling.stop_token_ids:
                    finish_reason = "stop"
                    break
                yield TokenOutput(
                    request_id=request.request_id,
                    token_id=token_id,
                    text=self._decode_token(token_id),
                    index=emitted,
                )
                emitted += 1
            else:
                if emitted < request.sampling.max_tokens:
                    finish_reason = "stop"
        finally:
            self._exit_request(request.request_id)
        yield TokenOutput(
            request_id=request.request_id,
            token_id=None,
            text="",
            index=emitted,
            finished=True,
            finish_reason=finish_reason,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=emitted),
        )

    def _token_iterator(self, request: GenerateRequest) -> tuple[int, Iterator[int]]:
        max_tokens = request.sampling.max_tokens
        temperature = request.sampling.temperature
        if request.messages:
            messages = [dict(message) for message in request.messages]
            prompt_tokens = self._chat_prompt_token_count(messages, max_tokens=max_tokens)
            return prompt_tokens, self._backend.generate_chat_tokens(  # type: ignore[attr-defined]
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        prompt = self._prompt_token_ids(request)
        input_ids = torch.tensor([prompt], dtype=torch.long, device=getattr(self._backend, "device"))
        return len(prompt), self._backend._generate_single_tokens(  # type: ignore[attr-defined]
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _prompt_token_ids(self, request: GenerateRequest) -> list[int]:
        if request.prompt_token_ids is not None:
            return [int(token_id) for token_id in request.prompt_token_ids]
        if request.prompt is not None:
            return list(self._backend.tokenizer.encode(request.prompt))  # type: ignore[attr-defined]
        raise ValueError("messages request does not have raw prompt tokens")

    def _prompt_token_count(self, request: GenerateRequest) -> int:
        if request.messages:
            return self._chat_prompt_token_count(
                [dict(message) for message in request.messages],
                max_tokens=request.sampling.max_tokens,
            )
        return len(self._prompt_token_ids(request))

    def _chat_prompt_token_count(self, messages: list[dict[str, object]], *, max_tokens: int) -> int:
        encode = getattr(self._backend, "_encode_chat_prompt", None)
        if callable(encode):
            return len(encode(messages, max_tokens=max_tokens))
        return len(self._backend.tokenizer.encode_messages(messages))  # type: ignore[attr-defined]

    def _decode_token(self, token_id: int) -> str:
        return str(self._backend.tokenizer.decode_token(token_id))  # type: ignore[attr-defined]

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("inference engine is closed")

    def _enter_request(self, request_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("inference engine is closed")
            self._active_requests.add(request_id)
            self._cancelled_requests.discard(request_id)

    def _exit_request(self, request_id: str) -> None:
        with self._lock:
            self._active_requests.discard(request_id)
            self._cancelled_requests.discard(request_id)

    def _is_cancelled(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancelled_requests

    def __enter__(self) -> "InferenceEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AsyncInferenceEngine:
    """Async facade for applications that need awaitable generation and streams."""

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    @classmethod
    def from_config(cls, config: EngineConfig) -> "AsyncInferenceEngine":
        return cls(InferenceEngine.from_config(config))

    @classmethod
    def from_sync_engine(cls, engine: InferenceEngine) -> "AsyncInferenceEngine":
        return cls(engine)

    @property
    def sync_engine(self) -> InferenceEngine:
        return self._engine

    @property
    def model_id(self) -> str:
        return self._engine.model_id

    async def generate(self, request: GenerateRequest) -> GenerateOutput:
        return await asyncio.to_thread(self._engine.generate, request)

    async def generate_stream(self, request: GenerateRequest):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[TokenOutput | BaseException | None] = asyncio.Queue()

        def run() -> None:
            try:
                for event in self._engine.generate_stream(request):
                    asyncio.run_coroutine_threadsafe(queue.put(event), loop).result()
            except BaseException as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        thread = threading.Thread(target=run, name=f"torchinferno-async-{request.request_id}", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if thread.is_alive():
                self.cancel(request.request_id)

    def cancel(self, request_id: str) -> None:
        self._engine.cancel(request_id)

    def close(self) -> None:
        self._engine.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    async def __aenter__(self) -> "AsyncInferenceEngine":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()


def _close_iterator(iterator: Iterator[Any]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
