"""P4.3: the renamed modules keep deprecated aliases."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

EMU_DIR = Path("/Users/zzhang/Workspace/MomentEmu-PolyCAMB-examples/emulators")


def test_polyemu_alias_warns_and_exposes_polyemu():
    sys.modules.pop("MomentEmu.PolyEmu", None)
    with pytest.warns(DeprecationWarning, match="MomentEmu.PolyEmu is deprecated"):
        from MomentEmu.PolyEmu import PolyEmu as AliasPolyEmu
    from MomentEmu.emulator import PolyEmu as NewPolyEmu

    assert AliasPolyEmu is NewPolyEmu


def test_core_alias_warns_and_exposes_functions():
    sys.modules.pop("MomentEmu.MomentEmu", None)
    with pytest.warns(DeprecationWarning, match="MomentEmu.MomentEmu is deprecated"):
        from MomentEmu.MomentEmu import signal_aware_frac_err as alias
    from MomentEmu.core import signal_aware_frac_err as new

    assert alias is new


def test_new_modules_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from MomentEmu.core import signal_aware_frac_err  # noqa: F401
        from MomentEmu.emulator import PolyEmu  # noqa: F401


@pytest.mark.skipif(not (EMU_DIR / "PolyCAMB_Dl_TT.pkl").exists(), reason="companion emulators absent")
def test_companion_notebook_pickle_loads():
    import pickle

    import numpy as np

    sys.modules.pop("MomentEmu.PolyEmu", None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        with open(EMU_DIR / "PolyCAMB_Dl_TT.pkl", "rb") as f:
            emu = pickle.load(f)
    theta = np.array([0.02242, 0.11933, 0.01041, 3.047, 0.9665, 0.0561])
    pred = emu.forward_emulator(theta, extrapolation="ignore")
    assert np.all(np.isfinite(pred))
