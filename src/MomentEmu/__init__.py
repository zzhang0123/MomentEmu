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

from MomentEmu.PolyEmu import PolyEmu, evaluate_emulator, symbolic_polynomial_expressions
from MomentEmu.MomentEmu import signal_aware_frac_err
from MomentEmu._stale_check import warn_if_stale

warn_if_stale()

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


__version__ = "1.0.0"
__author__ = "Zheng Zhang"
__all__ = [
    "PolyEmu",
    "evaluate_emulator",
    "symbolic_polynomial_expressions",
    "signal_aware_frac_err",
]
