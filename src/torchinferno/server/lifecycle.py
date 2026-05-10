from __future__ import annotations

from dataclasses import dataclass

from torchinferno.engine import InferenceEngine


@dataclass
class ServerLifecycle:
    engine: InferenceEngine
    ready: bool = False

    def start(self) -> None:
        self.ready = True

    def health(self) -> dict[str, object]:
        return {"status": "ok" if self.ready else "starting", "model": self.engine.model_id}

    def close(self) -> None:
        self.ready = False
        self.engine.close()
