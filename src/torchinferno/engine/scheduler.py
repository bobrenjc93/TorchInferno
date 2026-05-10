from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestAdmissionPolicy:
    max_batch_size: int = 64
    batch_wait_ms: float = 10.0
    single_request_admission_wait_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if self.single_request_admission_wait_ms is not None and self.single_request_admission_wait_ms < 0:
            raise ValueError("single_request_admission_wait_ms must be non-negative")
