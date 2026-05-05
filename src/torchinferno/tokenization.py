from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class TokenizersAdapter:
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)  # type: ignore[attr-defined]
        return list(encoded.ids)

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids))  # type: ignore[attr-defined]


class TransformersAdapter:
    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))  # type: ignore[attr-defined]


def load_text_tokenizer(tokenizer_name_or_path: str | Path, *, trust_remote_code: bool = False) -> TextTokenizer:
    """Load a local or Hugging Face tokenizer behind a tiny encode/decode API."""

    path = Path(tokenizer_name_or_path).expanduser()
    tokenizer_json = path / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("tokenizers is required to load tokenizer.json") from exc
        return TokenizersAdapter(Tokenizer.from_file(str(tokenizer_json)))

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("transformers is required to load a Hugging Face tokenizer") from exc
    return TransformersAdapter(AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=trust_remote_code))
