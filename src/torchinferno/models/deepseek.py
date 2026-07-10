"""Compatibility alias for the canonical DeepSeek-V3.2 model package.

New code should import from `torchinferno.models.deepseek_v32`. The module
object is aliased so existing monkeypatches against `torchinferno.models.deepseek`
still affect the implementation globals used by model methods.
"""

from __future__ import annotations

import sys as _sys

from torchinferno.models.deepseek_v32 import model as _model

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    setattr(_parent, "deepseek", _model)
_sys.modules[__name__] = _model
