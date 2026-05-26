from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class FastOpenAIHTTPServer:
    request_queue_size = 512

    def __init__(self, server_address: tuple[str, int], engine: object) -> None:
        self.engine = engine
        self._closed = threading.Event()
        self._executor = ThreadPoolExecutor(
            max_workers=env_int("TORCHINFERNO_OPENAI_HTTP_WORKERS", 256, minimum=1),
            thread_name_prefix="torchinferno-openai-http",
        )
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(server_address)
        self._socket.listen(self.request_queue_size)
        self._socket.settimeout(0.5)
        self.server_address = self._socket.getsockname()
        self.server_port = int(self.server_address[1])

    def serve_forever(self) -> None:
        while not self._closed.is_set():
            try:
                connection, client_address = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._closed.is_set():
                    return
                raise
            self._executor.submit(self._handle_connection, connection, client_address)

    def shutdown(self) -> None:
        self._closed.set()
        try:
            self._socket.close()
        except OSError:
            pass

    def server_close(self) -> None:
        self.shutdown()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _handle_connection(self, connection: socket.socket, client_address: object) -> None:
        del client_address
        with connection:
            enable_tcp_nodelay(connection)
            connection.settimeout(
                env_float("TORCHINFERNO_OPENAI_FAST_HTTP_IDLE_TIMEOUT_SECONDS", 5.0, minimum=0.05)
            )
            buffer = bytearray()
            keepalive_enabled = env_flag("TORCHINFERNO_OPENAI_FAST_HTTP_KEEPALIVE", True)
            while not self._closed.is_set():
                try:
                    request = _read_fast_http_request(connection, buffer)
                    if request is None:
                        return
                    method, path, headers, body = request
                    request_close = headers.get("connection", "").lower() == "close"
                    keep_alive = keepalive_enabled and not request_close
                    self._handle_request(
                        connection,
                        method,
                        path,
                        body,
                        keep_alive=keep_alive,
                        request_ready_s=time.perf_counter(),
                    )
                    if not keep_alive:
                        return
                except socket.timeout:
                    return
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError):
                    return
                except (ValueError, json.JSONDecodeError) as exc:
                    _send_fast_json(connection, error_response(str(exc), exc.__class__.__name__), status=400)
                    return
                except Exception as exc:
                    _send_fast_json(connection, error_response(str(exc), exc.__class__.__name__), status=500)
                    return

    def _handle_request(
        self,
        connection: socket.socket,
        method: str,
        path: str,
        body: bytes,
        *,
        keep_alive: bool,
        request_ready_s: float,
    ) -> None:
        route = path.partition("?")[0]
        if method == "GET" and route == "/health":
            _send_fast_json(connection, {"status": "ok"}, connection_close=not keep_alive)
            return
        if method == "GET" and route == "/v1/models":
            _send_fast_json(connection, model_list_response(self.engine.model_id), connection_close=not keep_alive)
            return
        if method != "POST" or route != "/v1/chat/completions":
            _send_fast_json(
                connection,
                error_response("not found", "NotFoundError"),
                status=404,
                connection_close=not keep_alive,
            )
            return
        parse_start_s = time.perf_counter()
        payload = json.loads(body.decode("utf-8"))
        request = parse_chat_completion_request(payload)
        parsed_s = time.perf_counter()
        if request.stream:
            _stream_fast_chat(
                connection,
                self.engine,
                request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                keep_alive=keep_alive,
                request_ready_s=request_ready_s,
                parse_ms=(parsed_s - parse_start_s) * 1000.0,
            )
            return
        completion = self.engine.complete_chat(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        content = self.engine.tokenizer.decode(completion.tokens)
        _send_fast_json(
            connection,
            chat_completion_response(
                model_id=self.engine.model_id,
                content=content,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=len(completion.tokens),
            ),
            connection_close=not keep_alive,
        )


_CHAT_DELTA_ROLE = b'{"role":"assistant"}'
_CHAT_DELTA_EMPTY = b"{}"

_FAST_HEADER_LIMIT = 65536


def _read_fast_http_request(
    connection: socket.socket,
    buffer: bytearray,
) -> tuple[str, str, dict[str, str], bytes] | None:
    while b"\r\n\r\n" not in buffer:
        chunk = connection.recv(65536)
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > _FAST_HEADER_LIMIT and b"\r\n\r\n" not in buffer:
            raise ValueError("HTTP request headers are too large")
    header_end = bytes(buffer).index(b"\r\n\r\n")
    header_bytes = bytes(buffer[:header_end])
    del buffer[:header_end + 4]
    lines = header_bytes.decode("iso-8859-1").split("\r\n")
    request_line = lines[0].split()
    if len(request_line) != 3:
        raise ValueError("malformed HTTP request line")
    method, path, _version = request_line
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    while len(buffer) < content_length:
        chunk = connection.recv(min(65536, content_length - len(buffer)))
        if not chunk:
            raise ConnectionResetError("client disconnected before request body completed")
        buffer.extend(chunk)
    body = bytes(buffer[:content_length])
    del buffer[:content_length]
    return method, path, headers, body


def _send_fast_json(
    connection: socket.socket,
    payload: dict[str, object],
    *,
    status: int = 200,
    connection_close: bool = True,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = _fast_response_headers(
        status,
        "application/json",
        content_length=len(body),
        connection_close=connection_close,
    )
    connection.sendall(headers + body)


def _stream_fast_chat(
    connection: socket.socket,
    engine: Any,
    messages: list[dict[str, object]],
    *,
    max_tokens: int,
    temperature: float,
    keep_alive: bool = False,
    request_ready_s: float | None = None,
    parse_ms: float = 0.0,
) -> None:
    profile = _new_fast_http_stream_profile(
        max_tokens=max_tokens,
        temperature=temperature,
        keep_alive=keep_alive,
        request_ready_s=request_ready_s,
        parse_ms=parse_ms,
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    chunk_prefix = _chat_completion_chunk_prefix(completion_id, engine.model_id, created)
    header_start_s = time.perf_counter()
    connection.sendall(
        _fast_response_headers(
            200,
            "text/event-stream",
            chunked=keep_alive,
            extra_headers=((b"Cache-Control", b"no-cache"),),
            connection_close=not keep_alive,
        )
    )
    _mark_fast_http_elapsed(profile, "headers_ms", header_start_s)
    _mark_fast_http_since_start(profile, "headers_sent_ms")
    defer_role = _stream_defer_role_enabled(max_tokens=max_tokens, temperature=temperature)
    client_open = True
    role_sent = False
    content_chunks = 0
    engine_tokens = 0
    empty_tokens = 0
    try:
        if not defer_role:
            role_start_s = time.perf_counter()
            client_open = _try_send_fast_chat_chunk(connection, chunk_prefix, _CHAT_DELTA_ROLE, chunked=keep_alive)
            role_sent = client_open
            _mark_fast_http_elapsed(profile, "role_send_ms", role_start_s)
        generate_start_s = time.perf_counter()
        for token_id in engine.generate_chat_tokens(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            engine_tokens += 1
            if engine_tokens == 1:
                _mark_fast_http_elapsed(profile, "engine_first_token_ms", generate_start_s)
                _mark_fast_http_since_start(profile, "first_engine_token_ms")
            if not client_open:
                continue
            decode_start_s = time.perf_counter()
            content = engine.tokenizer.decode_token(int(token_id))
            _add_fast_http_elapsed(profile, "decode_token_ms", decode_start_s)
            if not content:
                empty_tokens += 1
                continue
            delta = _chat_delta_role_content(content) if not role_sent else _chat_delta_content(content)
            role_sent = True
            send_start_s = time.perf_counter()
            client_open = _try_send_fast_chat_chunk(connection, chunk_prefix, delta, chunked=keep_alive)
            _add_fast_http_elapsed(profile, "content_send_ms", send_start_s)
            content_chunks += 1
            if content_chunks == 1:
                _mark_fast_http_since_start(profile, "first_content_sent_ms")
        if client_open:
            try:
                end_start_s = time.perf_counter()
                connection.sendall(
                    _fast_stream_end_bytes(
                        chunk_prefix,
                        include_role=not role_sent,
                        chunked=keep_alive,
                    )
                )
                _mark_fast_http_elapsed(profile, "finish_send_ms", end_start_s)
            except OSError:
                client_open = False
                return
    finally:
        if profile is not None:
            profile["engine_tokens"] = engine_tokens
            profile["content_chunks"] = content_chunks
            profile["empty_tokens"] = empty_tokens
            profile["client_open"] = bool(client_open)
            _record_fast_http_profile(profile)


_FAST_HTTP_PROFILE_LOCK = threading.Lock()


def _new_fast_http_stream_profile(
    *,
    max_tokens: int,
    temperature: float,
    keep_alive: bool,
    request_ready_s: float | None,
    parse_ms: float,
) -> dict[str, object] | None:
    if not _fast_http_profile_path():
        return None
    start_s = request_ready_s if request_ready_s is not None else time.perf_counter()
    return {
        "event": "fast_http_stream",
        "_start_s": start_s,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "keep_alive": bool(keep_alive),
        "parse_ms": float(parse_ms),
    }


def _mark_fast_http_elapsed(profile: dict[str, object] | None, field: str, start_s: float) -> None:
    if profile is not None:
        profile[field] = (time.perf_counter() - start_s) * 1000.0


def _add_fast_http_elapsed(profile: dict[str, object] | None, field: str, start_s: float) -> None:
    if profile is not None:
        profile[field] = float(profile.get(field, 0.0)) + (time.perf_counter() - start_s) * 1000.0


def _mark_fast_http_since_start(profile: dict[str, object] | None, field: str) -> None:
    if profile is None:
        return
    start_s = profile.get("_start_s")
    if isinstance(start_s, float):
        profile[field] = (time.perf_counter() - start_s) * 1000.0


def _fast_http_profile_path() -> str:
    return os.environ.get("TORCHINFERNO_OPENAI_FAST_HTTP_PROFILE_JSONL", "")


def _record_fast_http_profile(profile: dict[str, object]) -> None:
    path = _fast_http_profile_path()
    if not path:
        return
    start_s = profile.pop("_start_s", None)
    if isinstance(start_s, float):
        profile["total_ms"] = (time.perf_counter() - start_s) * 1000.0
    line = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with _FAST_HTTP_PROFILE_LOCK:
            with open(path, "a", encoding="utf-8") as profile_file:
                profile_file.write(line)
    except OSError:
        return


def _try_send_fast_chat_chunk(
    connection: socket.socket,
    chunk_prefix: bytes,
    delta_json: bytes,
    *,
    finish_reason: str | None = None,
    chunked: bool = False,
) -> bool:
    try:
        _send_fast_sse_bytes(
            connection,
            _chat_completion_chunk_bytes(chunk_prefix, delta_json, finish_reason),
            chunked=chunked,
        )
        return True
    except OSError:
        return False


def _send_fast_sse_bytes(connection: socket.socket, payload: bytes, *, chunked: bool) -> None:
    connection.sendall(_fast_sse_bytes(payload, chunked=chunked))


def _fast_sse_bytes(payload: bytes, *, chunked: bool) -> bytes:
    if chunked:
        return f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n"
    return payload


def _fast_stream_end_bytes(
    chunk_prefix: bytes,
    *,
    include_role: bool,
    chunked: bool,
) -> bytes:
    chunks: list[bytes] = []
    if include_role:
        chunks.append(
            _fast_sse_bytes(
                _chat_completion_chunk_bytes(chunk_prefix, _CHAT_DELTA_ROLE, None),
                chunked=chunked,
            )
        )
    chunks.append(
        _fast_sse_bytes(
            _chat_completion_chunk_bytes(chunk_prefix, _CHAT_DELTA_EMPTY, "stop"),
            chunked=chunked,
        )
    )
    chunks.append(_fast_sse_bytes(b"data: [DONE]\n\n", chunked=chunked))
    if chunked:
        chunks.append(b"0\r\n\r\n")
    return b"".join(chunks)


def _fast_response_headers(
    status: int,
    content_type: str,
    *,
    content_length: int | None = None,
    chunked: bool = False,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
    connection_close: bool,
) -> bytes:
    reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else "OK"
    headers = [
        f"HTTP/1.1 {status} {reason}\r\n".encode("ascii"),
        b"Server: TorchInfernoOpenAI/0.1\r\n",
        b"Content-Type: " + content_type.encode("ascii") + b"\r\n",
    ]
    if content_length is not None:
        headers.append(b"Content-Length: " + str(content_length).encode("ascii") + b"\r\n")
    if chunked:
        headers.append(b"Transfer-Encoding: chunked\r\n")
    for name, value in extra_headers:
        headers.append(name + b": " + value + b"\r\n")
    headers.append(b"Connection: close\r\n" if connection_close else b"Connection: keep-alive\r\n")
    headers.append(b"\r\n")
    return b"".join(headers)


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
