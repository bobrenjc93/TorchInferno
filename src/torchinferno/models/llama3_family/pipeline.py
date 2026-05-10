from __future__ import annotations

import sys as _sys

from torchinferno.models.llama3 import pipeline as _module
from torchinferno.models.llama3.pipeline import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "pipeline", _module)
_sys.modules[__name__] = _module
