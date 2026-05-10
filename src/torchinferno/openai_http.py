from __future__ import annotations

import json
import socket
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


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
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": engine.model_id,
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
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": {"message": str(exc), "type": exc.__class__.__name__}}, status=400)
        except Exception as exc:
            self._send_json({"error": {"message": str(exc), "type": exc.__class__.__name__}}, status=500)

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
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        for token_id in engine.generate_chat_tokens(messages, max_tokens=max_tokens, temperature=temperature):
            if not client_open:
                continue
            content = engine.tokenizer.decode_token(token_id)
            if not content:
                continue
            client_open = self._try_write_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_id,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )
        if client_open:
            client_open = self._try_write_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
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
    request_queue_size = 128

    def __init__(self, server_address: tuple[str, int], engine: object) -> None:
        super().__init__(server_address, OpenAIHandler)
        self.engine = engine
