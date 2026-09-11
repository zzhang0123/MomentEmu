"""P5.1: incremental bordered-moment forward sweep."""
from __future__ import annotations

import numpy as np

from MomentEmu.core import generate_moment_products, solve_emulator_coefficients
from MomentEmu.emulator import PolyEmu
from MomentEmu.monomials import MonomialPlan


def _target(X):
    return np.column_stack([
        np.tanh(3.0 * X[:, 0]) + X[:, 1] ** 2,
        np.sin(2.0 * X[:, 2]) + X[:, 3] * X[:, 4],
        X[:, 5] ** 3 - X[:, 0] * X[:, 1],
    ])


def test_incremental_coefficients_match_from_scratch():
    rng = np.random.default_rng(21)
    N = 3000
    X = rng.uniform(-1.0, 1.0, (N, 6))
    Y = _target(X)
    Xt = rng.uniform(-1.0, 1.0, (500, 6))
    Yt = _target(Xt)
    emu = PolyEmu(
        X, Y, X_test=Xt, Y_test=Yt,
        init_deg_forward=2, max_degree_forward=6,
        RMSE_tol=0.0, fRMSE_tol=0.0, verbose=0,
    )
    assert emu.forward_sweep_incremental_ is True
    Phi = MonomialPlan.build(emu.forward_multi_indices).evaluate(emu._X_train_scaled_)
    M, nu = generate_moment_products(Phi, emu._Y_train_scaled_)
    c_ref, _ = solve_emulator_coefficients(
        M, nu, on_singular="warn", degree=emu.forward_degree, return_cond=True
    )
    denom = float(np.max(np.abs(emu.forward_coeffs)))
    rel = float(np.max(np.abs(emu.forward_coeffs - c_ref))) / denom
    assert rel <= 1e-10, rel


def test_forward_sweep_attribute_present():
    assert hasattr(PolyEmu, "forward_sweep_incremental_")
