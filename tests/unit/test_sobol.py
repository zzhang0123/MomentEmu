"""P5.5: Legendre-based Sobol report."""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


def _ishigami(X, a=7.0, b=0.1):
    return (
        np.sin(X[:, 0:1]) + a * np.sin(X[:, 1:2]) ** 2 + b * X[:, 2:3] ** 4 * np.sin(X[:, 0:1])
    )


def _ishigami_exact(a=7.0, b=0.1):
    pi = np.pi
    var_t = pi ** 8 / 9 - pi ** 8 / 25
    var = 0.5 * (1 + b * pi ** 4 / 5) ** 2 + a ** 2 / 8 + 0.5 * b ** 2 * var_t
    s1 = np.array([0.5 * (1 + b * pi ** 4 / 5) ** 2 / var, (a ** 2 / 8) / var, 0.0])
    s13 = 0.5 * b ** 2 * var_t / var
    st = np.array([s1[0] + s13, s1[1], s13])
    return s1, st, var


def test_polynomial_sobol_is_exact():
    # f = x0^2 + x1 on the exact [-1, 1]^2 box; Var(x0^2) = 4/45, Var(x1) = 1/3.
    rng = np.random.default_rng(7)
    n = 20000
    X = np.vstack([np.array(list(product([-1.0, 1.0], repeat=2))), rng.uniform(-1, 1, (n, 2))])
    Y = X[:, 0:1] ** 2 + X[:, 1:2]
    emu = PolyEmu(X, Y, init_deg_forward=2, max_degree_forward=2, RMSE_tol=0.0, verbose=0)
    r = emu.sobol_report(warn_uniform=False)
    total = 4.0 / 45.0 + 1.0 / 3.0
    np.testing.assert_allclose(r["S1"][:, 0], [4.0 / 45.0 / total, 1.0 / 3.0 / total], atol=1e-10)
    np.testing.assert_allclose(r["ST"][:, 0], r["S1"][:, 0], atol=1e-10)
    # shares_sum is the fraction of Var(Y) the Legendre decomposition explains;
    # compare it with the model R^2 on the same rows (it is no longer a tautology).
    pred = emu.forward_emulator(X, extrapolation="ignore")
    r2 = 1.0 - float(np.mean((pred - Y) ** 2)) / float(np.var(Y))
    assert abs(float(r["shares_sum"][0]) - r2) < 0.01 * abs(r2)


@pytest.mark.slow
def test_ishigami_s1_st_match_analytic():
    pi = np.pi
    rng = np.random.default_rng(0)
    corners = np.array(list(product([-pi, pi], repeat=3)))
    X = np.vstack([corners, rng.uniform(-pi, pi, (20000, 3))])
    Y = _ishigami(X)
    emu = PolyEmu(X, Y, init_deg_forward=14, max_degree_forward=14, RMSE_tol=0.0, verbose=0)
    r = emu.sobol_report()
    s1, st, _ = _ishigami_exact()
    assert np.max(np.abs(r["S1"][:, 0] - s1)) < 1e-7
    assert np.max(np.abs(r["ST"][:, 0] - st)) < 1e-7
    # The shares close and agree with the retained variance.
    pred = emu.forward_emulator(X)
    r2 = 1.0 - float(np.mean((pred - Y) ** 2)) / float(np.var(Y))
    assert abs(float(r["shares_sum"][0]) - r2) < 0.01 * abs(r2)


def test_nonuniform_design_warns():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 0.3, (3000, 2))
    Y = (X[:, 0:1] ** 2 + X[:, 1:2])
    emu = PolyEmu(X, Y, init_deg_forward=2, max_degree_forward=2, RMSE_tol=0.0, verbose=0)
    with pytest.warns(UserWarning, match="not uniform"):
        emu.sobol_report()


def test_report_attaches_sobol():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, (2000, 2))
    Y = X[:, 0:1] ** 2 + X[:, 1:2]
    emu = PolyEmu(X, Y, init_deg_forward=2, max_degree_forward=2, RMSE_tol=0.0, verbose=0)
    info = emu.report(include_sobol=True)
    assert "sobol" in info and info["sobol"]["S1"].shape == (2, 1)


def test_sobol_term_guard():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (50, 3))
    Y = X[:, 0:1]
    emu = PolyEmu(X, Y, init_deg_forward=1, max_degree_forward=1, RMSE_tol=0.0, verbose=0)
    with pytest.raises(ValueError, match="terms"):
        emu.sobol_report(degree=30, max_terms=100)
