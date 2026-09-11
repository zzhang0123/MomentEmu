"""P2.3: Torch backend to the JAX contract (float64, log_Y, vmap)."""
from __future__ import annotations

import numpy as np
import pytest

import torch

from MomentEmu.emulator import PolyEmu
from MomentEmu.torch_momentemu import TorchMomentEmu, create_torch_emulator


def _col_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _fit(seed=0, log_Y=False, with_std=True):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 5.0).reshape(-1, 1)
    return PolyEmu(
        X, Y, log_Y=log_Y, standardize_Y_with_std=with_std, max_degree_forward=4, verbose=0
    )


@pytest.mark.backend
@pytest.mark.parametrize("log_Y", [False, True])
@pytest.mark.parametrize("with_std", [True, False])
def test_torch_matches_forward_emulator(log_Y, with_std):
    emu = _fit(seed=1, log_Y=log_Y, with_std=with_std)
    tm = TorchMomentEmu(emu)
    assert tm.coeffs.dtype == torch.float64
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (500, 3))
    with torch.no_grad():
        got = tm(torch.as_tensor(X, dtype=torch.float64)).numpy()
    ref = emu.forward_emulator(X, extrapolation="ignore")
    assert got.dtype == np.float64
    assert _col_rel(got, ref) < 1e-13


@pytest.mark.backend
def test_torch_vmap_grad_finite():
    emu = _fit(seed=3)
    tm = TorchMomentEmu(emu)
    rng = np.random.default_rng(4)
    X = torch.as_tensor(rng.uniform(-1.0, 1.0, (1000, 3)), dtype=torch.float64)
    grads = torch.func.vmap(torch.func.grad(lambda x: tm(x).sum()))(X)
    assert grads.shape == (1000, 3)
    assert bool(torch.isfinite(grads).all())


@pytest.mark.backend
def test_torch_hessian_finite_at_mean():
    emu = _fit(seed=5)
    tm = TorchMomentEmu(emu)
    x = torch.as_tensor(emu.scaler_X.mean_, dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda v: tm(v).sum(), x)
    assert H.shape == (3, 3)
    assert bool(torch.isfinite(H).all())


@pytest.mark.backend
def test_torch_dtype_argument_and_creation():
    emu = _fit(seed=6)
    tm32 = create_torch_emulator(emu, dtype=torch.float32)
    assert tm32.coeffs.dtype == torch.float32
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, (100, 3))
    with torch.no_grad():
        got = tm32(torch.as_tensor(X, dtype=torch.float32)).numpy()
    ref = emu.forward_emulator(X, extrapolation="ignore")
    assert _col_rel(got, ref) < 1e-6
