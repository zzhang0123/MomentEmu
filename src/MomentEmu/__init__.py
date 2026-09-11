"""MomentEmu: a lightweight, interpretable polynomial emulator for smooth mappings.

This package implements the moment-projection polynomial emulator introduced in
Zhang (2025): interpretable, closed-form polynomial emulators built from moment
matrices, with millisecond inference and symbolic transparency.

The autodiff backends (jax_momentemu, torch_momentemu, symbolic_momentemu) are
loaded lazily through PEP 562 __getattr__ so that import MomentEmu does not pay
for jax, torch or sklearn.  They are not in __all__ but are listed by dir().
"""
from __future__ import annotations

import importlib
import sys as _sys
import types as _types

from MomentEmu._stale_check import warn_if_stale
from MomentEmu.core import signal_aware_frac_err
from MomentEmu.emulator import PolyEmu, evaluate_emulator, symbolic_polynomial_expressions

warn_if_stale()

# B2: ``import MomentEmu.PolyEmu`` (the deprecated shim) makes the import system
# bind the *module* to ``MomentEmu.PolyEmu``, shadowing the class. Pre-2.0
# pickles store the class as ``MomentEmu PolyEmu``, so that shadowing breaks
# unpickling and ``from MomentEmu import PolyEmu``. Route attribute access on
# the package through a ModuleType subclass that always returns the class.
_POLYEMU_CLASS = PolyEmu


class _PackageModule(_types.ModuleType):
    def __getattribute__(self, name):  # noqa: D105
        if name == "PolyEmu":
            return _POLYEMU_CLASS
        return super().__getattribute__(name)


_sys.modules[__name__].__class__ = _PackageModule

_BACKENDS = {
    "jax_momentemu": "jax",
    "torch_momentemu": "torch",
    "symbolic_momentemu": "sympy",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        try:
            mod = importlib.import_module("." + name, __name__)
        except ImportError as exc:
            raise ImportError(
                f"MomentEmu.{name} needs the optional {_BACKENDS[name]!r} dependency: "
                f"{exc}. Install it, e.g. pip install MomentEmu[{_BACKENDS[name]}]."
            ) from exc
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKENDS))


__version__ = "2.0.0"
__author__ = "Zheng Zhang"
__all__ = [
    "PolyEmu",
    "evaluate_emulator",
    "symbolic_polynomial_expressions",
    "signal_aware_frac_err",
]
