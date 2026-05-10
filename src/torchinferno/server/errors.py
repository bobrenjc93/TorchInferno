from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIErrorPayload:
    message: str
    type: str = "invalid_request_error"
    code: str | None = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {"message": self.message, "type": self.type}
        if self.code is not None:
            payload["code"] = self.code
        return payload


class TorchInfernoAPIError(Exception):
    def __init__(self, message: str, *, status: int = 400, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status = status
        self.payload = APIErrorPayload(message, error_type)
