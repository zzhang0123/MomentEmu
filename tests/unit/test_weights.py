"""P5.6: optional per-sample weights."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


def _truth(X):
    return (X[:, 0] ** 4 - 2 * X[:, 0] ** 2 + X[:, 1] ** 2 + 0.5 * X[:, 2]).reshape(-1, 1)


def test_weights_none_is_bit_identical_to_ones():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = _truth(X) + rng.normal(0.0, 0.05, _truth(X).shape)
    a = PolyEmu(X, Y, max_degree_forward=4, verbose=0)
    b = PolyEmu(X, Y, weights=np.ones(300), max_degree_forward=4, verbose=0)
    assert np.array_equal(a.forward_coeffs, b.forward_coeffs)


def test_wls_beats_ols_under_heteroscedastic_noise():
    ratios = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1.0, 1.0, (400, 3))
        Yt = _truth(X)
        sigma = np.exp(rng.uniform(np.log(0.01), np.log(0.1), 400))
        Y = Yt + rng.normal(0.0, 1.0, Yt.shape) * sigma[:, None]
        Xt = rng.uniform(-1.0, 1.0, (1500, 3))
        Yf = _truth(Xt)
        common = dict(
            X_test=Xt, Y_test=Yf, init_deg_forward=4, max_degree_forward=4,
            RMSE_tol=1e-300, verbose=0,
        )
        ols = PolyEmu(X, Y, **common)
        wls = PolyEmu(X, Y, weights=1.0 / sigma ** 2, **common)
        e_ols = float(np.sqrt(np.mean((ols.forward_emulator(Xt, extrapolation="ignore") - Yf) ** 2)))
        e_wls = float(np.sqrt(np.mean((wls.forward_emulator(Xt, extrapolation="ignore") - Yf) ** 2)))
        ratios.append(e_ols / e_wls)
    assert float(np.mean(ratios)) > 1.9, ratios


def test_weights_reject_bad_shapes():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    with pytest.raises(ValueError, match="weights has"):
        PolyEmu(X, Y, weights=np.ones(99), max_degree_forward=3, verbose=0)
