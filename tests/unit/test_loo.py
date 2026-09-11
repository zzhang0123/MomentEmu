"""P1.5: exact LOO/PRESS selection on all N."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import cho_factor, solve_triangular

from MomentEmu.MomentEmu import press_loo
from MomentEmu.PolyEmu import PolyEmu, evaluate_monomials_lazy, generate_multi_indices


def _design(seed=0, N=120, n=2, d=5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.3 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    mi = generate_multi_indices(n, d)
    Phi = evaluate_monomials_lazy(X, mi)
    N = X.shape[0]
    return X, Y, mi, Phi


def test_press_equals_brute_force_refits():
    X, Y, mi, Phi = _design()
    N = X.shape[0]
    M = Phi.T @ Phi / N
    nu = Phi.T @ Y / N
    _c, _cond, _loo, loo_per_out, _lev = press_loo(M, nu, Phi, Y, on_singular="raise")
    # Brute force: refit without each row and predict it.
    press_bf = np.zeros(Y.shape[1])
    for i in range(N):
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        Pi, Yi = Phi[mask], Y[mask]
        c = np.linalg.lstsq(Pi, Yi, rcond=None)[0]
        r = (Y[i] - Phi[i] @ c) ** 2
        press_bf += r
    loo_bf = np.sqrt(press_bf / N)
    np.testing.assert_allclose(loo_per_out, loo_bf, rtol=1e-9, atol=1e-12)


def test_hat_diagonal_sums_to_D():
    X, Y, mi, Phi = _design(seed=1)
    N, D = Phi.shape[0], Phi.shape[1]
    M = Phi.T @ Phi / N
    nu = Phi.T @ Y / N
    _c, _cond, _loo, _per, lev_max = press_loo(M, nu, Phi, Y, on_singular="raise")
    cf, lower = cho_factor(M, lower=False)
    A = solve_triangular(cf, Phi.T, lower=lower, trans="T")
    h = np.einsum("ij,ij->j", A, A) / N
    assert h.sum() == pytest.approx(D, rel=1e-9)
    assert lev_max == pytest.approx(h.max(), rel=1e-12)


def test_no_split_when_x_test_is_none(monkeypatch):
    import sklearn.model_selection as ms

    calls = []
    monkeypatch.setattr(ms, "train_test_split", lambda *a, **k: calls.append(1))
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3)
    assert calls == []
    assert emu.loo_rmse_ is not None and np.isfinite(emu.loo_rmse_)
    assert 0.0 < emu.leverage_max_train_ < 1.0


def test_cross_validation_is_deprecated():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    with pytest.warns(DeprecationWarning, match="cross_validation"):
        PolyEmu(X, Y, cross_validation=True, max_degree_forward=2)


def _target(X):
    return (np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.3 * X[:, 0] ** 3).reshape(-1, 1)


@pytest.mark.slow
@pytest.mark.parametrize("N", [120, 400])
def test_loo_beats_split_seed_averaged(N):
    from sklearn.model_selection import train_test_split

    # Emulate the retired 85/15 split baseline explicitly and compare the
    # seed-averaged fresh RMSE of the LOO route (all N) against it.
    for target_seed in range(3):
        loo_rmses, split_rmses = [], []
        for seed in range(8):
            rng = np.random.default_rng(1000 * target_seed + seed)
            X = rng.uniform(-1.0, 1.0, (N, 2))
            Y = _target(X)
            loo = PolyEmu(X, Y, max_degree_forward=6, verbose=0)
            Xtr, Xval, Ytr, Yval = train_test_split(
                X, Y, test_size=0.15, random_state=seed
            )
            split = PolyEmu(
                Xtr, Ytr, X_test=Xval, Y_test=Yval, max_degree_forward=6, verbose=0
            )
            rng_fresh = np.random.default_rng(9000 + seed)
            Xf = rng_fresh.uniform(-1.0, 1.0, (400, 2))
            Yf = _target(Xf)
            scale = float(np.sqrt(np.mean(Yf ** 2)))
            loo_rmses.append(
                float(np.sqrt(np.mean((loo.forward_emulator(Xf, extrapolation="ignore") - Yf) ** 2))) / scale
            )
            split_rmses.append(
                float(np.sqrt(np.mean((split.forward_emulator(Xf, extrapolation="ignore") - Yf) ** 2))) / scale
            )
        assert np.mean(loo_rmses) <= np.mean(split_rmses)