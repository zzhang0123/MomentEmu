"""P3.3: posterior-shift protocol (closed form and the author chains)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "validate_posterior.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_posterior", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shift_report_closed_form():
    mod = _load_module()
    names = ["a", "b", "c"]
    cov = np.diag([0.01, 0.04, 0.09])
    sd = np.sqrt(np.diag(cov))
    mean = np.zeros(3)
    shift = np.array([0.004, -0.008, 0.004])
    rep = mod.shift_report(mean, sd, cov, mean + shift, sd, cov, names)
    np.testing.assert_allclose(
        [r["shift_sigma"] for r in rep["rows"]], shift / sd, rtol=1e-12
    )
    assert rep["mahalanobis"] == pytest.approx(
        float(np.sqrt(shift @ np.linalg.solve(cov, shift))), rel=1e-12
    )
    assert rep["pass"] is True


def test_shift_report_flags_a_large_shift():
    mod = _load_module()
    cov = np.eye(2) * 0.01
    sd = np.sqrt(np.diag(cov))
    rep = mod.shift_report(np.zeros(2), sd, cov, np.array([0.5, 0.0]), sd, cov, ["a", "b"])
    assert rep["pass"] is False
    assert abs(rep["worst_shift"]["shift_sigma"]) > 0.1


CHAINS = "/Users/zzhang/Workspace/MomentEmu-PolyCAMB-examples/chains"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CHAINS, "planck_camb_pol.1.txt")),
    reason="companion chains not present",
)
@pytest.mark.slow
def test_author_chains_pass():
    mod = _load_module()
    report = mod._load_chains(CHAINS)
    assert report["pass"] is True
    assert abs(report["worst_shift"]["shift_sigma"]) == pytest.approx(0.025, abs=0.005)
    assert report["mahalanobis"] == pytest.approx(0.032, abs=0.005)
