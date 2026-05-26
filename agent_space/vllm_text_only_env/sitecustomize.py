"""Keep optional audio dependencies out of text-only vLLM profiling runs."""

from __future__ import annotations

import importlib.util as _importlib_util

_original_find_spec = _importlib_util.find_spec


def _find_spec_without_librosa(name: str, *args: object, **kwargs: object) -> object:
    if name == "librosa" or name.startswith("librosa."):
        return None
    return _original_find_spec(name, *args, **kwargs)


_importlib_util.find_spec = _find_spec_without_librosa
