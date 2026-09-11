"""Regression tests for the 2026-09-12 independent review (B1-B9)."""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

from MomentEmu.basis import Basis
from MomentEmu.core import generate_moment_products, press_loo, select_best_model
from MomentEmu.emulator import (
    MonomialPlan,
    PolyEmu,
    _safe_cholesky,
    generate_multi_indices,
)


def _smooth(seed=0, N=400, n=2):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    return X, Y


def test_b1_legacy_pickle_without_log_y_transforms():
    X, Y = _smooth()
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    del emu.transform
    del emu.log_Y
    assert emu._transforms() == ("linear",)


def test_b2_shim_import_does_not_shadow_polyemu_class():
    # The shim may already be imported by another test; suppress/ignore the
    # deprecation warning and check the class is not shadowed either way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shim = importlib.import_module("MomentEmu.PolyEmu")
    assert shim is not None
    from MomentEmu import PolyEmu as package_attr

    assert isinstance(package_attr, type)
    import MomentEmu

    assert isinstance(MomentEmu.PolyEmu, type)


def test_b3_jacobian_per_output_transforms():
    rng = np.random.default_rng(0)
    N = 500
    X = rng.uniform(-1.0, 1.0, (N, 2))
    Y = np.column_stack([
        np.exp(0.7 * X[:, 0] + 0.3 * X[:, 1]) + 2.0,
        np.sinh(0.5 * X[:, 0] - 0.4 * X[:, 1]) + 1.0,
        X[:, 0] ** 2 + X[:, 1],
    ])
    emu = PolyEmu(
        X, Y, transform=["log", "asinh", "linear"],
        max_degree_forward=3, RMSE_tol=1e-300, verbose=0,
    )
    x0 = np.array([0.2, -0.3])
    J = emu.jacobian(x0)
    FD = np.stack(
        [
            (emu.forward_emulator(x0 + 1e-6 * e, extrapolation="ignore")
             - emu.forward_emulator(x0 - 1e-6 * e, extrapolation="ignore")) / 2e-6
            for e in np.eye(2)
        ],
        axis=1,
    )
    assert np.max(np.abs(J - FD)) / np.max(np.abs(FD)) < 1e-7


def test_b4_weighted_press_matches_brute_force():
    rng = np.random.default_rng(1)
    N = 150
    X = rng.uniform(-1.0, 1.0, (N, 2))
    Y = (np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)
    w = rng.uniform(0.2, 3.0, N)
    w = w / w.mean()
    mi = generate_multi_indices(2, 4)
    Phi = MonomialPlan.build(mi).evaluate(X)
    M, nu = generate_moment_products(Phi, Y, weights=w)
    _c, _cond, _loo, loo_per, _lev = press_loo(M, nu, Phi, Y, on_singular="warn", weights=w)
    press_int = N * loo_per ** 2
    press_bf = np.zeros(1)
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        wi, Pi, Yi = w[mask], Phi[mask], Y[mask]
        c = np.linalg.solve(Pi.T @ (Pi * wi[:, None]), Pi.T @ (Yi * wi[:, None]))
        e = Y[i] - Phi[i] @ c
        press_bf += w[i] * e ** 2
    assert abs(press_int[0] - press_bf[0]) / press_bf[0] < 1e-9


def test_b5_rmse_tol_selects_simplest_within_tolerance():
    assert select_best_model([1.0, 0.95, 0.94], rmse_tol=0.1) == 0
    assert select_best_model([1.0, 0.95, 0.94], rmse_tol=0.01) == 2


def test_b6_safe_cholesky_survives_singular_matrix():
    assert _safe_cholesky(np.zeros((3, 3))) is not None
    X, Y = _smooth(seed=2)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    h0 = emu.leverage(X[:10])
    emu.forward_chol_ = None  # force the pseudoinverse fallback
    h1 = emu.leverage(X[:10])
    np.testing.assert_allclose(h0, h1, rtol=1e-7, atol=1e-10)


def test_b7_compress_without_validation_still_gates():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (600, 2))
    Y = np.column_stack([X[:, 0] ** 2, np.sin(2.0 * X[:, 1]), X[:, 0] * X[:, 1]])
    emu = PolyEmu(X, Y, max_degree_forward=4, verbose=0)
    with pytest.raises(ValueError, match="more than"):
        emu.compress(1)


def test_b7_float32_difference_is_stored_and_exposed(tmp_path):
    X, Y = _smooth(seed=4, N=600)
    emu = PolyEmu(X, Y, max_degree_forward=4, verbose=0)
    path = tmp_path / "emu.npz"
    emu.save(path, float32=True)
    loaded = PolyEmu.load(path)
    assert loaded.float32_difference_ is not None


def test_b8_fit_resets_stale_backward_state():
    X, Y = _smooth(seed=5)
    emu = PolyEmu(
        X, Y, forward=True, backward=True,
        init_deg_backward=1, max_degree_backward=2, verbose=0,
    )
    assert hasattr(emu, "backward_coeffs")
    emu.fit(X, Y, forward=True, backward=False, max_degree_forward=3)
    assert not hasattr(emu, "backward_coeffs")


def test_b9_mismatched_validation_rows_raise():
    X, Y = _smooth(seed=6)
    with pytest.raises(ValueError, match="row-aligned"):
        PolyEmu(X, Y, X_test=X[:10], Y_test=Y[:5], verbose=0)


def test_b9_basis_with_backward_raises():
    X, Y = _smooth(seed=7)
    with pytest.raises(ValueError, match="basis="):
        PolyEmu(X, Y, basis=Basis.total_degree(), backward=True, verbose=0)

