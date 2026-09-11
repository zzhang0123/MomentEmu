"""P3.4: two-stage zoom refit helper."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, refit_zoom


def _sim(X):
    return (np.exp(0.5 * X[:, 0]) + np.sin(2 * X[:, 1]) + X[:, 0] * X[:, 1]).reshape(-1, 1)


def test_refit_restricts_to_box():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, (2000, 2))
    Y = _sim(X)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    sub = emu.refit(X, Y, box=([-0.5, -0.5], [0.5, 0.5]), max_degree_forward=4, verbose=0)
    assert np.all(sub.X_box_.lo >= -0.5) and np.all(sub.X_box_.hi <= 0.5)
    with pytest.raises(ValueError, match="inside the refit box"):
        emu.refit(X, Y, box=([10.0, 10.0], [11.0, 11.0]))


def test_zoom_refit_improves_near_the_truth():
    # Coarse fit on a wide box.
    X = np.random.default_rng(1).uniform(-3.0, 3.0, (3000, 2))
    coarse = PolyEmu(X, _sim(X), max_degree_forward=3, verbose=0)
    truth = np.array([0.9, -0.7])
    sd = np.array([0.12, 0.12])
    zoom = refit_zoom(
        coarse, _sim, truth, sd, k=6.0, init_deg_forward=4, max_degree_forward=4,
        RMSE_tol=1e-300, verbose=0, seed=2,
    )
    rng = np.random.default_rng(3)
    Xf = truth + rng.normal(0.0, sd * 0.5, (1000, 2))
    Yf = _sim(Xf)
    err_coarse = float(np.sqrt(np.mean((coarse.forward_emulator(Xf, extrapolation="ignore") - Yf) ** 2)))
    err_zoom = float(np.sqrt(np.mean((zoom.forward_emulator(Xf, extrapolation="ignore") - Yf) ** 2)))
    assert err_zoom < 0.2 * err_coarse, (err_zoom, err_coarse)


def test_zoom_recovers_in_box():
    X = np.random.default_rng(4).uniform(-3.0, 3.0, (3000, 2))
    coarse = PolyEmu(X, _sim(X), max_degree_forward=3, verbose=0)
    truth = np.array([0.5, 0.5])
    sd = np.array([0.1, 0.1])
    zoom = refit_zoom(coarse, _sim, truth, sd, k=6.0, max_degree_forward=4, verbose=0, seed=5)
    # The zoom training box is within +/- 6 sd of the truth.
    assert np.all(zoom.X_box_.lo >= truth - 6 * sd - 1e-9)
    assert np.all(zoom.X_box_.hi <= truth + 6 * sd + 1e-9)
