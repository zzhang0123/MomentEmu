"""P2.5b: cobaya Theory helper (optional module, -inf outside the box)."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu


def test_importing_momentemu_does_not_import_cobaya():
    env = dict(os.environ)
    env["MOMENTEMU_IGNORE_STALE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, MomentEmu; print('cobaya' in sys.modules)"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def _toy(seed=0, m=20):
    rng = np.random.default_rng(seed)
    lo = np.array([0.1, -1.0, 0.5])
    hi = np.array([0.9, 1.0, 1.5])
    X = rng.uniform(lo, hi, (3000, 3))
    k = np.linspace(0.1, 2.0, m)

    def sim(th):
        return (
            th[..., 0:1] * np.exp(-k * th[..., 1:2] ** 2) * np.sin(k * th[..., 2:3])
            + th[..., 2:3]
        )

    Y = sim(X)
    emu = PolyEmu(X, Y, max_degree_forward=5, verbose=0)
    return emu, sim, lo, hi


@pytest.mark.slow
def test_cobaya_theory_logpost_and_out_of_box():
    pytest.importorskip("cobaya")
    from cobaya.likelihood import Likelihood
    from cobaya.model import get_model

    from MomentEmu.cobaya_theory import MomentEmuTheory

    emu, sim, lo, hi = _toy()
    theta_fid = 0.5 * (lo + hi)
    data = sim(theta_fid)
    sig = 0.02 * np.abs(data).max()

    class GaussLike(Likelihood):
        def get_requirements(self):
            return {"emu_spectrum": None}

        def logp(self, **params):
            y = self.provider.get_emu_spectrum()
            return float(-0.5 * np.sum(((y - data) / sig) ** 2))

    info = {
        "params": {
            "a": {"prior": {"min": lo[0] - 1, "max": hi[0] + 1}, "ref": theta_fid[0], "proposal": 0.01},
            "b": {"prior": {"min": lo[1] - 1, "max": hi[1] + 1}, "ref": theta_fid[1], "proposal": 0.01},
            "c": {"prior": {"min": lo[2] - 1, "max": hi[2] + 1}, "ref": theta_fid[2], "proposal": 0.01},
        },
        "theory": {
            "momentemu": {
                "external": MomentEmuTheory,
                "emulator": emu,
                "param_names": ["a", "b", "c"],
            }
        },
        "likelihood": {"gauss": GaussLike},
    }
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(info)
    lp = model.logpost(np.asarray(theta_fid))
    ref = -0.5 * np.sum(((emu.forward_emulator(theta_fid) - data) / sig) ** 2) + model.logprior(
        np.asarray(theta_fid)
    )
    assert lp == ref
    # One ulp outside each face, still inside the (wider) prior: -inf.
    for i in range(3):
        theta_out = np.array(theta_fid, dtype=float)
        theta_out[i] = np.nextafter(hi[i], np.inf)
        assert model.logpost(theta_out) == -np.inf
