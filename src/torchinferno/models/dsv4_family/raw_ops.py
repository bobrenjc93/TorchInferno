from __future__ import annotations

import sys as _sys

from torchinferno.models.dsv4 import raw_ops as _module
from torchinferno.models.dsv4.raw_ops import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "raw_ops", _module)
_sys.modules[__name__] = _module
