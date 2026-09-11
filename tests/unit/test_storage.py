"""P5.7: reduced-rank and float32 storage."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu
from MomentEmu.io import load_emulator, save_emulator
from MomentEmu.storage import (
    rank_mse_increase,
    reduced_rank_coefficients,
)


def _design(seed=0, N=800, n=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Z = np.column_stack([np.exp(0.4 * X[:, 0]) + X[:, 1], np.sin(2 * X[:, 2]) + X[:, 3]])
    B = np.random.default_rng(seed + 100).normal(size=(2, 5))
    Y = Z @ B
    return X, Y


def test_reduced_rank_matches_hand_svd_and_mse_curve():
    X, Y = _design()
    Xs = (X - X.mean(0)) / X.std(0)
    Ys = (Y - Y.mean(0)) / Y.std(0)
    N = X.shape[0]
    from MomentEmu.emulator import MonomialPlan, generate_multi_indices

    mi = generate_multi_indices(4, 4)
    Phi = MonomialPlan.build(mi).evaluate(Xs)
    M = Phi.T @ Phi / N
    nu = Phi.T @ Ys / N
    rank = 3
    C_r, s = reduced_rank_coefficients(M, nu, rank)
    from scipy.linalg import solve_triangular

    L = np.linalg.cholesky(M)
    B = solve_triangular(L, nu, lower=True)
    U, sv, Vt = np.linalg.svd(B, full_matrices=False)
    hand = solve_triangular(L.T, (U[:, :rank] * sv[:rank]) @ Vt[:rank], lower=False)
    np.testing.assert_allclose(C_r, hand, rtol=1e-12, atol=1e-14)
    # MSE increase equals sum_{i>rank} s_i^2 / m.
    mse_full = float(np.mean((Ys - Phi @ np.linalg.solve(M, nu)) ** 2))
    mse_r = float(np.mean((Ys - Phi @ C_r) ** 2))
    assert mse_r - mse_full == pytest.approx(rank_mse_increase(sv, rank, Y.shape[1]), rel=1e-10)


def test_break_even_rank_is_refused():
    rng = np.random.default_rng(1)
    M = np.eye(10)
    nu = rng.normal(size=(10, 5))
    with pytest.raises(ValueError, match="break-even"):
        reduced_rank_coefficients(M, nu, 4)
    reduced_rank_coefficients(M, nu, 3)


def test_compress_stores_and_gates(tmp_path):
    X, Y = _design(seed=2)
    rng = np.random.default_rng(3)
    Xv = rng.uniform(-1.0, 1.0, (200, 4))
    Zv = np.column_stack([np.exp(0.4 * Xv[:, 0]) + Xv[:, 1], np.sin(2 * Xv[:, 2]) + Xv[:, 3]])
    Bv = np.random.default_rng(102).normal(size=(2, 5))
    Yv = Zv @ Bv
    emu = PolyEmu(X, Y, X_test=Xv, Y_test=Yv, max_degree_forward=4, verbose=0)
    full = emu.forward_emulator(Xv, extrapolation="ignore")
    emu.compress(3, X_val=Xv, Y_val=Yv)
    got = emu.forward_emulator(Xv, extrapolation="ignore")
    assert np.max(np.abs(got - full)) / np.max(np.abs(full)) < 1e-6
    assert emu.forward_rank_ == 3


def test_float32_round_trip_within_1e_6(tmp_path):
    X, Y = _design(seed=4)
    emu = PolyEmu(X, Y, max_degree_forward=4, verbose=0)
    path = tmp_path / "emu.npz"
    save_emulator(emu, path, float32=True)
    loaded = load_emulator(path)
    Xv = np.random.default_rng(5).uniform(-1.0, 1.0, (300, 4))
    ref = emu.forward_emulator(Xv, extrapolation="ignore")
    got = loaded.forward_emulator(Xv, extrapolation="ignore")
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-6
