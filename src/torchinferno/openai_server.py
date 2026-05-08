from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor

from torchinferno.models.deepseek import DeepSeekV32ForCausalLM, sample_next_token, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.models.llama3_family.pipeline import Llama3PipelineForCausalLM
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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.cache_backend = cache_backend
        self.page_size = page_size
        self.max_model_len = max_model_len
        self._lock = threading.Lock()

    def generate_chat_tokens(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[int]:
        prompt = self.tokenizer.encode_messages(messages)
        if self.max_model_len is not None and len(prompt) + max_tokens > self.max_model_len:
            prompt_budget = max(1, self.max_model_len - max_tokens)
            prompt = prompt[-prompt_budget:]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with self._lock:
            yield from self._generate_tokens(input_ids, max_tokens=max_tokens, temperature=temperature)

    @torch.inference_mode()
    def _generate_tokens(self, input_ids: Tensor, *, max_tokens: int, temperature: float) -> Iterator[int]:
        if max_tokens <= 0:
            return
        eos_token_id = self.tokenizer.eos_token_id
        model = self.model
        if not hasattr(model, "allocate_cache") or not callable(getattr(model, "forward", None)):
            generated = model.generate(  # type: ignore[attr-defined]
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=eos_token_id,
            )
            for token in generated[0, input_ids.size(1) :].detach().cpu().tolist():
                yield int(token)
            return

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
        for _ in range(max_tokens):
            token_id = int(next_token.item())
            yield token_id
            if eos_token_id is not None and token_id == eos_token_id:
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
        tokens = list(engine.generate_chat_tokens(messages, max_tokens=max_tokens, temperature=temperature))
        content = engine.tokenizer.decode(tokens)
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
                    "prompt_tokens": 0,
                    "completion_tokens": len(tokens),
                    "total_tokens": len(tokens),
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
    )


def serve(config: OpenAIServerConfig) -> None:
    engine = build_engine(config)
    server = _OpenAIServer((config.host, config.port), engine)
    print(
        f"TorchInferno OpenAI server listening on http://{config.host}:{server.server_port}/v1 "
        f"model={config.model}",
        flush=True,
    )
    server.serve_forever()


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
    try:
        return model.forward(  # type: ignore[attr-defined]
            input_ids,
            cache=cache,
            use_cache=True,
            return_last_logits_only=True,
        )
    except TypeError:
        return model.forward(input_ids, cache=cache, use_cache=True)  # type: ignore[attr-defined]


def _sample(model: object, logits: Tensor, temperature: float) -> Tensor:
    sample = getattr(model, "_sample_next_token", None)
    if sample is not None:
        return sample(logits, temperature)
    return sample_next_token(logits, temperature)


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
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    serve(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
