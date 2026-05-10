from __future__ import annotations

import sys as _sys

from torchinferno.models.llama3 import tensor_parallel as _module
from torchinferno.models.llama3.tensor_parallel import *  # noqa: F401,F403

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "tensor_parallel", _module)
_sys.modules[__name__] = _module
