from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


ChatMessage = Mapping[str, object]


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        object.__setattr__(self, "stop_token_ids", tuple(int(token_id) for token_id in self.stop_token_ids))


@dataclass(frozen=True)
class GenerateRequest:
    request_id: str
    prompt: str | None = None
    messages: Sequence[ChatMessage] = ()
    prompt_token_ids: Sequence[int] | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stream: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        sources = sum(
            [
                self.prompt is not None,
                bool(self.messages),
                self.prompt_token_ids is not None,
            ]
        )
        if sources != 1:
            raise ValueError("exactly one of prompt, messages, or prompt_token_ids must be provided")
        object.__setattr__(self, "messages", tuple(dict(message) for message in self.messages))
        if self.prompt_token_ids is not None:
            object.__setattr__(self, "prompt_token_ids", tuple(int(token_id) for token_id in self.prompt_token_ids))


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    token_id: int | None
    text: str
    index: int
    finished: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class GenerateOutput:
    request_id: str
    token_ids: tuple[int, ...]
    text: str
    finish_reason: str
    usage: Usage
