"""P2.2: analytic NumPy Jacobian."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


def _fit(log_Y=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 5.0).reshape(-1, 1)
    return PolyEmu(X, Y, log_Y=log_Y, max_degree_forward=4, verbose=0)


def _points(emu, rng):
    lo = emu.X_box_.lo
    hi = emu.X_box_.hi
    corners = np.array(np.meshgrid(*zip(lo, hi))).T.reshape(-1, emu.n_params)
    return np.vstack([emu.scaler_X.mean_[None, :], corners, rng.uniform(lo, hi, (50, emu.n_params))])


@pytest.mark.parametrize("log_Y", [False, True])
def test_jacobian_matches_central_fd(log_Y):
    emu = _fit(log_Y=log_Y)
    rng = np.random.default_rng(1)
    X = _points(emu, rng)
    J = emu.jacobian(X)
    assert J.shape == (X.shape[0], emu.n_outputs, emu.n_params)
    eps = 1e-5
    fd = np.empty_like(J)
    for i in range(emu.n_params):
        Xp = X.copy()
        Xm = X.copy()
        Xp[:, i] += eps
        Xm[:, i] -= eps
        fd[:, :, i] = (
            emu.forward_emulator(Xp, extrapolation="ignore")
            - emu.forward_emulator(Xm, extrapolation="ignore")
        ) / (2 * eps)
    err = np.max(np.abs(J - fd)) / np.max(np.abs(fd))
    assert err < 1e-7, err


@pytest.mark.parametrize("log_Y", [False, True])
def test_jacobian_matches_jax_jacfwd(log_Y):
    import jax
    import jax.numpy as jnp

    from MomentEmu.jax_momentemu import create_jax_emulator

    jax.config.update("jax_enable_x64", True)
    emu = _fit(log_Y=log_Y, seed=2)
    f = create_jax_emulator(emu)
    rng = np.random.default_rng(3)
    X = _points(emu, rng)
    J = emu.jacobian(X)
    Jy = np.array([np.asarray(jax.jacfwd(f)(jnp.asarray(x))) for x in X])
    assert Jy.shape == J.shape
    assert np.max(np.abs(J - Jy)) / np.max(np.abs(Jy)) < 1e-12


def test_jacobian_single_point_shape():
    emu = _fit()
    J = emu.jacobian(emu.scaler_X.mean_)
    assert J.shape == (emu.n_outputs, emu.n_params)
