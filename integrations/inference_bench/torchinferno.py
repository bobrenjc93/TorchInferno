from __future__ import annotations

from . import register
from .base import Provider


@register("torchinferno")
class TorchInfernoProvider(Provider):
    name = "torchinferno"
    repo_url = "https://github.com/bobrenjc93/TorchInferno.git"

    def build(self) -> None:
        self._create_venv()
        self._pip_install("--upgrade", "pip")
        self._pip_install("-e", ".[serve]", cwd=self.repo_dir)

    def _server_cmd(self, model: str, tp: int, port: int) -> list[str]:
        return [
            self.venv_python,
            "-m",
            "torchinferno.openai_server",
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp),
            "--port",
            str(port),
            "--trust-remote-code",
        ]
