from __future__ import annotations

import json
import socket
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from torchinferno.server.openai_protocol import (
    chat_completion_chunk,
    chat_completion_response,
    error_response,
    model_list_response,
    parse_chat_completion_request,
)


class OpenAIHandler(BaseHTTPRequestHandler):
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
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
        except OSError:
            self.close_connection = True
            return
        client_open = self._try_write_sse(
            chat_completion_chunk(
                completion_id=completion_id,
                model_id=engine.model_id,
                created=created,
                delta={"role": "assistant"},
            )
        )
        for token_id in engine.generate_chat_tokens(messages, max_tokens=max_tokens, temperature=temperature):
            if not client_open:
                continue
            content = engine.tokenizer.decode_token(token_id)
            if not content:
                continue
            client_open = self._try_write_sse(
                chat_completion_chunk(
                    completion_id=completion_id,
                    model_id=engine.model_id,
                    created=created,
                    delta={"content": content},
                )
            )
        if client_open:
            client_open = self._try_write_sse(
                chat_completion_chunk(
                    completion_id=completion_id,
                    model_id=engine.model_id,
                    created=created,
                    delta={},
                    finish_reason="stop",
                )
            )
        if client_open:
            self._try_write_done()
        self.close_connection = True

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

    def _try_write_sse(self, payload: dict[str, object]) -> bool:
        try:
            self._write_sse(payload)
            return True
        except OSError:
            self.close_connection = True
            return False

    def _try_write_done(self) -> bool:
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return True
        except OSError:
            self.close_connection = True
            return False


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
