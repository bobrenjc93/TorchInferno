from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OpenAIChatCompletionRequest:
    messages: list[dict[str, object]]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    model: str | None = None


def parse_chat_completion_request(payload: Mapping[str, object]) -> OpenAIChatCompletionRequest:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized_messages: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("messages entries must be objects")
        normalized_messages.append(dict(message))
    max_tokens = int(payload.get("max_tokens", 256))
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    temperature = float(payload.get("temperature", 0.0))
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    top_p = float(payload.get("top_p", 1.0))
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")
    if top_p != 1.0:
        raise ValueError("TorchInferno currently supports only top_p=1.0")
    model = payload.get("model")
    return OpenAIChatCompletionRequest(
        messages=normalized_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=bool(payload.get("stream", False)),
        model=str(model) if model is not None else None,
    )


def model_list_response(model_id: str, *, created: int | None = None) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()) if created is None else created,
                "owned_by": "torchinferno",
            }
        ],
    }


def chat_completion_response(
    *,
    model_id: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    completion_id: str | None = None,
    created: int | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def chat_completion_chunk(
    *,
    completion_id: str,
    model_id: str,
    delta: dict[str, object],
    finish_reason: str | None = None,
    created: int | None = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()) if created is None else created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def error_response(message: str, error_type: str) -> dict[str, object]:
    return {"error": {"message": message, "type": error_type}}
