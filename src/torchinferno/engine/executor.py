from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from torchinferno.engine.types import GenerateRequest, TokenOutput


class GenerationExecutor(Protocol):
    def generate_stream(self, request: GenerateRequest) -> Iterator[TokenOutput]: ...

    def cancel(self, request_id: str) -> None: ...

    def close(self) -> None: ...
