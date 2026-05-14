from __future__ import annotations

import json
import queue
import socket
import threading
import time
import uuid
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from torchinferno.server.openai_protocol import (
    chat_completion_response,
    error_response,
    model_list_response,
    parse_chat_completion_request,
)
from torchinferno.runtime.options import env_flag, env_float, env_int


class OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TorchInfernoOpenAI/0.1"

    def setup(self) -> None:
        super().setup()
        enable_tcp_nodelay(self.connection)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/v1/models":
            engine = self._engine()
            self._send_json(model_list_response(engine.model_id))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            payload = self._read_json()
            request = parse_chat_completion_request(payload)
            if request.stream:
                self._stream_chat(request.messages, max_tokens=request.max_tokens, temperature=request.temperature)
            else:
                self._complete_chat(request.messages, max_tokens=request.max_tokens, temperature=request.temperature)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(error_response(str(exc), exc.__class__.__name__), status=400)
        except Exception as exc:
            self._send_json(error_response(str(exc), exc.__class__.__name__), status=500)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _engine(self) -> Any:
        return self.server.engine  # type: ignore[attr-defined]

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _complete_chat(self, messages: list[dict[str, object]], *, max_tokens: int, temperature: float) -> None:
        engine = self._engine()
        completion = engine.complete_chat(messages, max_tokens=max_tokens, temperature=temperature)
        content = engine.tokenizer.decode(completion.tokens)
        self._send_json(
            chat_completion_response(
                model_id=engine.model_id,
                content=content,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=len(completion.tokens),
            )
        )

    def _stream_chat(self, messages: list[dict[str, object]], *, max_tokens: int, temperature: float) -> None:
        engine = self._engine()
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        chunk_prefix = _chat_completion_chunk_prefix(completion_id, engine.model_id, created)
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            chunked_stream = _chunked_stream_enabled(getattr(self, "request_version", "HTTP/1.0"))
            if chunked_stream:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "keep-alive")
            else:
                self.send_header("Connection", "close")
            self._chunked_sse = chunked_stream
            self.end_headers()
        except OSError:
            self.close_connection = True
            return
        defer_role = _stream_defer_role_enabled(max_tokens=max_tokens, temperature=temperature)
        client_open = True
        role_sent = False
        if not defer_role:
            client_open = self._try_write_chat_completion_chunk(
                chunk_prefix,
                _CHAT_DELTA_ROLE,
            )
            role_sent = client_open
        if _stream_inline_enabled(max_tokens=max_tokens, temperature=temperature):
            for token_id in engine.generate_chat_tokens(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                if not client_open:
                    continue
                content = engine.tokenizer.decode_token(int(token_id))
                if not content:
                    continue
                delta = (
                    _chat_delta_role_content(content)
                    if not role_sent
                    else _chat_delta_content(content)
                )
                role_sent = True
                client_open = self._try_write_chat_completion_chunk(
                    chunk_prefix,
                    delta,
                )
            if client_open:
                if not role_sent:
                    client_open = self._try_write_chat_completion_chunk(
                        chunk_prefix,
                        _CHAT_DELTA_ROLE,
                    )
            if client_open:
                client_open = self._try_write_chat_completion_chunk(
                    chunk_prefix,
                    _CHAT_DELTA_EMPTY,
                    finish_reason="stop",
                )
            if client_open:
                self._try_write_done()
            self._finish_sse_response()
            return

        token_queue: "queue.Queue[object]" = queue.Queue()
        done = object()

        def produce_tokens() -> None:
            try:
                for token_id in engine.generate_chat_tokens(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    token_queue.put(int(token_id))
            except BaseException as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(done)

        producer = threading.Thread(target=produce_tokens, name="torchinferno-openai-stream-producer", daemon=True)
        producer.start()
        heartbeat_s = env_float("TORCHINFERNO_OPENAI_STREAM_HEARTBEAT_SECONDS", 15.0, minimum=0.0)
        while True:
            try:
                item = token_queue.get(timeout=heartbeat_s if heartbeat_s > 0.0 else None)
            except queue.Empty:
                if client_open:
                    client_open = self._try_write_sse_comment("torchinferno heartbeat")
                continue
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            if not client_open:
                continue
            content = engine.tokenizer.decode_token(int(item))
            if not content:
                continue
            delta = (
                _chat_delta_role_content(content)
                if not role_sent
                else _chat_delta_content(content)
            )
            role_sent = True
            client_open = self._try_write_chat_completion_chunk(
                chunk_prefix,
                delta,
            )
        if client_open:
            if not role_sent:
                client_open = self._try_write_chat_completion_chunk(
                    chunk_prefix,
                    _CHAT_DELTA_ROLE,
                )
        if client_open:
            client_open = self._try_write_chat_completion_chunk(
                chunk_prefix,
                _CHAT_DELTA_EMPTY,
                finish_reason="stop",
            )
        if client_open:
            self._try_write_done()
        self._finish_sse_response()

    def _write_sse_comment(self, comment: str) -> None:
        self._write_sse_bytes(b": " + comment.encode("utf-8") + b"\n\n")

    def _try_write_sse_comment(self, comment: str) -> bool:
        try:
            self._write_sse_comment(comment)
            return True
        except OSError:
            self.close_connection = True
            return False

    def _send_json(self, payload: dict[str, object], *, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(self, payload: dict[str, object]) -> None:
        self.wfile.write(
            b"data: "
            + json.dumps(payload, separators=(",", ":")).encode("utf-8")
            + b"\n\n"
        )
        self.wfile.flush()

    def _write_sse_bytes(self, payload: bytes) -> None:
        if getattr(self, "_chunked_sse", False):
            self.wfile.write(f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n")
        else:
            self.wfile.write(payload)
        self.wfile.flush()

    def _try_write_sse(self, payload: dict[str, object]) -> bool:
        try:
            self._write_sse(payload)
            return True
        except OSError:
            self.close_connection = True
            return False

    def _try_write_chat_completion_chunk(
        self,
        chunk_prefix: bytes,
        delta_json: bytes,
        *,
        finish_reason: str | None = None,
    ) -> bool:
        try:
            self._write_sse_bytes(_chat_completion_chunk_bytes(chunk_prefix, delta_json, finish_reason))
            return True
        except OSError:
            self.close_connection = True
            return False

    def _try_write_done(self) -> bool:
        try:
            self._write_sse_bytes(b"data: [DONE]\n\n")
            return True
        except OSError:
            self.close_connection = True
            return False

    def _finish_sse_response(self) -> None:
        if getattr(self, "_chunked_sse", False):
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True
            finally:
                self._chunked_sse = False
            return
        self.close_connection = True


def enable_tcp_nodelay(connection: object) -> None:
    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # type: ignore[attr-defined]
    except OSError:
        pass


class OpenAIHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, server_address: tuple[str, int], engine: object) -> None:
        super().__init__(server_address, OpenAIHandler)
        self.engine = engine


_CHAT_DELTA_ROLE = b'{"role":"assistant"}'
_CHAT_DELTA_EMPTY = b"{}"


def _chat_completion_chunk_prefix(completion_id: str, model_id: str, created: int) -> bytes:
    return (
        b'data: {"id":'
        + json.dumps(completion_id, separators=(",", ":")).encode("utf-8")
        + b',"object":"chat.completion.chunk","created":'
        + str(int(created)).encode("ascii")
        + b',"model":'
        + json.dumps(model_id, separators=(",", ":")).encode("utf-8")
        + b',"choices":[{"index":0,"delta":'
    )


@lru_cache(maxsize=8192)
def _chat_delta_content(content: str) -> bytes:
    return b'{"content":' + json.dumps(content, separators=(",", ":")).encode("utf-8") + b"}"


@lru_cache(maxsize=8192)
def _chat_delta_role_content(content: str) -> bytes:
    return b'{"role":"assistant","content":' + json.dumps(content, separators=(",", ":")).encode("utf-8") + b"}"


def _chat_completion_chunk_bytes(
    chunk_prefix: bytes,
    delta_json: bytes,
    finish_reason: str | None,
) -> bytes:
    finish_json = (
        b"null"
        if finish_reason is None
        else json.dumps(finish_reason, separators=(",", ":")).encode("utf-8")
    )
    return chunk_prefix + delta_json + b',"finish_reason":' + finish_json + b"}]}\n\n"


def _stream_inline_enabled(*, max_tokens: int, temperature: float) -> bool:
    del max_tokens, temperature
    return env_flag("TORCHINFERNO_OPENAI_STREAM_INLINE", True)


def _stream_defer_role_enabled(*, max_tokens: int, temperature: float) -> bool:
    del temperature
    if not env_flag("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE", True):
        return False
    max_defer_tokens = env_int("TORCHINFERNO_OPENAI_STREAM_DEFER_ROLE_MAX_TOKENS", 400, minimum=1)
    return max_tokens <= max_defer_tokens


def _chunked_stream_enabled(request_version: str) -> bool:
    return request_version == "HTTP/1.1" and env_flag("TORCHINFERNO_OPENAI_HTTP_CHUNKED_STREAM", True)
