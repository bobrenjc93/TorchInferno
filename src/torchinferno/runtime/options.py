from __future__ import annotations

import os
import sys
from typing import Final

_FALSE_VALUES: Final[set[str]] = {"", "0", "false", "no", "off"}
_WARNED_KEYS: set[str] = set()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    return max(minimum, value) if minimum is not None else value


def env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    return max(minimum, value) if minimum is not None else value


def warn_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    print(message, file=sys.stderr)


def warn_optional_failure(
    key: str,
    exc: BaseException,
    *,
    debug_env: str = "TORCHINFERNO_OPTIONAL_WARNINGS",
) -> None:
    if env_flag(debug_env, False):
        warn_once(key, f"TorchInferno optional path disabled for {key}: {exc}")
