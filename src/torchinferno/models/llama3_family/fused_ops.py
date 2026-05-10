from __future__ import annotations

import sys as _sys

from torchinferno.models.llama3 import fused_ops as _module
from torchinferno.models.llama3.fused_ops import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "fused_ops", _module)
_sys.modules[__name__] = _module
