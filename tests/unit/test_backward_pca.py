"""P5.8: backward PCA (output-side reduction) and per-parameter R^2."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu
from MomentEmu.guards import basis_size


def test_backward_pca_recovers_full_rank_linear_map():
    rng = np.random.default_rng(11)
    W = rng.normal(size=(4, 200))
    X = rng.uniform(-1.0, 1.0, (600, 4))
    Y = X @ W
    emu = PolyEmu(
        X, Y, init_deg_forward=1, max_degree_forward=1,
        init_deg_backward=1, max_degree_backward=1, verbose=0,
    )
    emu.backward_pca(rank=4, degree=1)
    X_pred = emu.backward_pca_predict(Y)
    assert emu.backward_pca_n_terms_ == 5
    assert np.max(np.abs(X_pred - X)) < 1e-8
    np.testing.assert_allclose(emu.backward_R2_per_parameter_, 1.0, atol=1e-10)


def test_non_injective_parameter_reports_nonpositive_r2():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, (800, 2))
    Y = X[:, 0:1]
    X_val, Y_val = X[600:], Y[600:]
    emu = PolyEmu(
        X[:600], Y[:600], init_deg_forward=1, max_degree_forward=1,
        init_deg_backward=1, max_degree_backward=1, verbose=0,
    )
    emu.backward_pca(rank=1, degree=1, X_val=X_val, Y_val=Y_val)
    r2 = emu.backward_R2_per_parameter_
    assert r2[0] > 0.99
    assert r2[1] <= 0.0
    assert 1 in emu.backward_unconstrained_parameters_
    # E[X | Y] for the unconstrained parameter is nearly constant.
    X_pred = emu.backward_pca_predict(Y_val)
    assert np.std(X_pred[:, 1]) < 0.1 * np.std(X_val[:, 1])


def test_backward_sweep_respects_fill_factor():
    rng = np.random.default_rng(13)
    X = rng.uniform(0.0, 1.0, (1000, 3))
    Y = np.stack([np.sin(3.0 * X[:, 0]), X[:, 1] ** 2, X[:, 2] * X[:, 0]], axis=1)
    Y = np.hstack([Y, rng.normal(0.0, 1e-3, (1000, 197))])
    emu = PolyEmu(
        X, Y, init_deg_forward=2, max_degree_forward=2, backward=True,
        init_deg_backward=1, max_degree_backward=1, verbose=0,
    )
    D = basis_size(emu.n_outputs, emu.backward_degree)
    assert emu.backward_N_train_ >= D * 2


def test_backward_pca_rank_guard():
    rng = np.random.default_rng(14)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = np.stack([X[:, 0], X[:, 1]], axis=1)
    emu = PolyEmu(X, Y, init_deg_forward=1, max_degree_forward=1, verbose=0)
    with pytest.raises(ValueError, match="rank"):
        emu.backward_pca(rank=9, degree=1)


def test_backward_pca_predict_requires_fit():
    rng = np.random.default_rng(15)
    X = rng.uniform(-1.0, 1.0, (200, 2))
    Y = X.copy()
    emu = PolyEmu(X, Y, init_deg_forward=1, max_degree_forward=1, verbose=0)
    assert hasattr(PolyEmu, "backward_pca")
    with pytest.raises(RuntimeError, match="backward_pca"):
        emu.backward_pca_predict(Y)


_ELL = np.arange(2, 202, dtype=float)
_CMB_LO = np.array([2.00e3, -0.10, 290.0, 1200.0, 0.50, -0.30])
_CMB_HI = np.array([3.00e3, 0.10, 310.0, 1600.0, 0.90, 0.30])


def _cmb_like(t):
    A, tilt, lA, lD, r1, ph = (t[:, i:i + 1] for i in range(6))
    ell = _ELL[None, :]
    env = A * (ell / 220.0) ** tilt / (1.0 + (ell / 900.0) ** 2) ** 0.9
    damp = np.exp(-((ell / lD) ** 1.6))
    x = 2.0 * np.pi * ell / lA
    osc = (r1 * np.cos(x + ph) + 0.40 * r1 * np.cos(2.0 * x + 2.0 * ph + 0.30)
           + 0.15 * r1 * np.cos(3.0 * x + 3.0 * ph - 0.50))
    return env * (1.0 + 0.6 * np.exp(-((ell / 60.0) ** 1.2))) * (1.0 + damp * osc)


def _lhs(n, rng):
    u = np.empty((n, 6))
    for j in range(6):
        strata = (np.arange(n) + rng.random(n)) / n
        u[:, j] = rng.permutation(strata)
    return _CMB_LO + u * (_CMB_HI - _CMB_LO)


@pytest.mark.slow
def test_backward_pca_cmb_like_median_error():
    rng = np.random.default_rng(20260910)
    X = _lhs(5000, rng)
    Xt = _lhs(2000, rng)
    Y = _cmb_like(X)
    Yt = _cmb_like(Xt)
    emu = PolyEmu(
        X, Y, log_Y=True, init_deg_forward=1, max_degree_forward=1,
        init_deg_backward=1, max_degree_backward=1, verbose=0,
    )
    emu.backward_pca(rank=8, degree=3)
    assert emu.backward_pca_n_terms_ == 165
    X_pred = emu.backward_pca_predict(Yt)
    med = float(np.median(np.abs(X_pred - Xt) / (_CMB_HI - _CMB_LO)))
    assert med < 2e-4, med
    assert np.all(emu.backward_R2_per_parameter_ > 0.5)
