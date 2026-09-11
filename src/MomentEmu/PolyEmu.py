"""Deprecated alias for :mod:`MomentEmu.emulator` (removed in 3.0.0).

Importing this module emits a DeprecationWarning. It keeps
``MomentEmu.PolyEmu.PolyEmu`` importable so pre-2.0 pickles load; new code
should import from :mod:`MomentEmu.emulator`.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "MomentEmu.PolyEmu is deprecated; import from MomentEmu.emulator instead "
    "(the alias is removed in 3.0.0).",
    DeprecationWarning,
    stacklevel=2,
)

from MomentEmu import emulator as _emulator  # noqa: E402

for _name in dir(_emulator):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_emulator, _name))

__all__ = [name for name in dir(_emulator) if not name.startswith("_")]
