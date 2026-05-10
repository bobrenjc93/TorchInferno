from __future__ import annotations

from torchinferno.engine import EngineConfig, InferenceEngine


def build_engine(config: EngineConfig) -> InferenceEngine:
    return InferenceEngine.from_config(config)


def serve_legacy_openai(config: object) -> None:
    from torchinferno.openai_server import serve

    serve(config)


def main(argv: list[str] | None = None) -> int:
    from torchinferno.openai_server import main as legacy_main

    return legacy_main(argv)
