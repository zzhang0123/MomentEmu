"""cobaya Theory helper (optional, P2.5b, D17).

Import this module only when cobaya is available (``import MomentEmu`` does
not).  The wrapper returns out-of-box points as ``False``, which cobaya maps to
a -inf likelihood, rather than extrapolating the polynomial.
"""
from __future__ import annotations

import os

import numpy as np

try:
    from cobaya.theory import Theory
except ImportError as exc:  # pragma: no cover - exercised only without cobaya
    raise ImportError(
        "MomentEmu.cobaya_theory needs cobaya; install it with pip install cobaya."
    ) from exc


class MomentEmuTheory(Theory):
    """cobaya Theory around a fitted :class:`~MomentEmu.PolyEmu.PolyEmu`.

    Set ``emulator`` (a PolyEmu or a path to a saved one), ``param_names``
    (ordered like the emulator input columns) and ``result_name`` through the
    cobaya config or by subclassing.
    """

    emulator = None
    param_names: list = []
    result_name = "emu_spectrum"

    def initialize(self):
        if isinstance(self.emulator, (str, os.PathLike)):
            import pickle

            with open(self.emulator, "rb") as f:
                self.emulator = pickle.load(f)
        self._emu = self.emulator
        self.lo = np.asarray(self._emu.X_box_.lo, dtype=float)
        self.hi = np.asarray(self._emu.X_box_.hi, dtype=float)

    def get_requirements(self):
        return list(self.param_names)

    def get_can_provide(self):
        return [self.result_name]

    def calculate(self, state, want_derived=True, **params_values_dict):
        theta = np.array(
            [params_values_dict[p] for p in self.param_names], dtype=float
        )
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            # cobaya turns False into LogLikeOutOfBounds (-inf).
            return False
        state[self.result_name] = self._emu.forward_emulator(
            theta, extrapolation="ignore"
        )

    def get_emu_spectrum(self):
        return self.current_state[self.result_name]

    def get_version(self):
        from MomentEmu import __version__

        fp = getattr(self._emu, "hash_", None)
        return f"MomentEmu {__version__} ({fp})" if fp else f"MomentEmu {__version__}"
