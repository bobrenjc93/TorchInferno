from __future__ import annotations

import sys as _sys

from torchinferno.models.deepseek_v32 import v1 as _module
from torchinferno.models.deepseek_v32.v1 import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "v1", _module)
_sys.modules[__name__] = _module
