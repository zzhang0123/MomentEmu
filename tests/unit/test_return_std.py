"""P3.5: noise-only predictive band and hat diagonal."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


def _truth(X):
    return (X[:, 0] ** 4 - 2 * X[:, 0] ** 2 + X[:, 1] ** 2 + 0.5 * X[:, 2] * X[:, 3]).reshape(-1, 1)


def _coverage(seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 4))
    Y_true = _truth(X)
    noise_std = 0.05 * np.abs(Y_true).max()
    Y = Y_true + rng.normal(0.0, noise_std, Y_true.shape)
    emu = PolyEmu(
        X, Y, init_deg_forward=4, max_degree_forward=4, RMSE_tol=1e-300, verbose=0
    )
    assert emu.forward_multi_indices.shape[0] == 70  # C(8,4)
    Xf = rng.uniform(-1.0, 1.0, (2000, 4))
    # A fresh noisy draw, so the band must cover the noise, not just the model.
    Yf = _truth(Xf) + rng.normal(0.0, noise_std, (2000, 1))
    pred, std = emu.forward_emulator(Xf, return_std=True, extrapolation="ignore")
    inside = np.abs(Yf - pred) <= std
    return float(np.mean(inside))


def test_one_sigma_coverage_for_iid_noise():
    cov = np.mean([_coverage(seed) for seed in range(10)])
    assert 0.66 <= cov <= 0.71, cov


def test_hat_diagonal_sums_to_D():
    rng = np.random.default_rng(100)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    h = emu.hat_diagonal(X)
    assert np.sum(h) == pytest.approx(emu.forward_multi_indices.shape[0], rel=1e-9)


def test_return_std_shapes():
    rng = np.random.default_rng(101)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    pred, std = emu.forward_emulator(X[:10], return_std=True)
    assert pred.shape == std.shape == (10, 1)
    assert np.all(std > 0)
    p1, s1 = emu.forward_emulator(X[0], return_std=True)
    assert p1.shape == s1.shape == (1,)
