"""Deprecated alias for :mod:`MomentEmu.core` (removed in 3.0.0).

Importing this module emits a DeprecationWarning; new code should import from
:mod:`MomentEmu.core`.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "MomentEmu.MomentEmu is deprecated; import from MomentEmu.core instead "
    "(the alias is removed in 3.0.0).",
    DeprecationWarning,
    stacklevel=2,
)

from MomentEmu import core as _core  # noqa: E402

for _name in dir(_core):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_core, _name))

__all__ = [name for name in dir(_core) if not name.startswith("_")]
