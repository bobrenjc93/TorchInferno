from __future__ import annotations

import sys as _sys

from torchinferno.models.llama3 import registry as _module
from torchinferno.models.llama3.registry import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "registry", _module)
_sys.modules[__name__] = _module
