"""P0.8: backend stopgaps refuse the inputs they answered wrongly."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from MomentEmu.jax_momentemu import create_jax_emulator  # noqa: E402
from MomentEmu.symbolic_momentemu import create_symbolic_emulator  # noqa: E402
from MomentEmu.torch_momentemu import TorchMomentEmu  # noqa: E402


def _col_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(scope="module")
def plain_emu():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (X[:, 0] ** 2 + np.sin(X[:, 1]) + X[:, 2]).reshape(-1, 1)
    return PolyEmu(X, Y, cross_validation=False, max_degree_forward=3, dim_reduction=False)


@pytest.fixture(scope="module")
def no_std_emu():
    rng = np.random.default_rng(8)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(X[:, 0]) + X[:, 1] ** 2 + 5.0).reshape(-1, 1)
    return PolyEmu(
        X,
        Y,
        cross_validation=False,
        standardize_Y_with_std=False,
        max_degree_forward=3,
        dim_reduction=False,
    )


@pytest.fixture(scope="module")
def logy_emu():
    rng = np.random.default_rng(9)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = np.exp(X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)
    return PolyEmu(
        X, Y, log_Y=True, cross_validation=False, max_degree_forward=3, dim_reduction=False
    )


@pytest.mark.backend
def test_backend_rejects_logy(logy_emu):
    with pytest.raises(NotImplementedError):
        create_jax_emulator(logy_emu)
    with pytest.raises(NotImplementedError):
        TorchMomentEmu(logy_emu)
    with pytest.raises(NotImplementedError):
        create_symbolic_emulator(logy_emu)
    with pytest.raises(NotImplementedError):
        logy_emu.generate_forward_symb_emu()


@pytest.mark.backend
def test_jax_matches_forward_emulator_no_std(no_std_emu):
    rng = np.random.default_rng(10)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    f = create_jax_emulator(no_std_emu)
    got = np.asarray(f(jnp.asarray(X)))
    ref = no_std_emu.forward_emulator(X)
    assert _col_rel(got, ref) < 1e-12


@pytest.mark.backend
def test_torch_matches_forward_emulator_no_std(no_std_emu):
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    tm = TorchMomentEmu(no_std_emu)
    with torch.no_grad():
        got = tm(torch.as_tensor(X, dtype=torch.float64)).numpy()
    ref = no_std_emu.forward_emulator(X)
    assert _col_rel(got, ref) < 1e-6


@pytest.mark.backend
def test_n1_1d_batch(plain_emu):
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, (100, 1))
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    emu = PolyEmu(X, Y, cross_validation=False, max_degree_forward=3, dim_reduction=False)
    f = create_jax_emulator(emu)
    # n_params == 1: a length-1 1-D array is one sample.
    assert np.asarray(f(jnp.array([0.5]))).shape == (1,)
    # A length-3 1-D array is ambiguous for n_params = 1 and must raise.
    with pytest.raises(ValueError):
        f(jnp.array([0.1, 0.2, 0.3]))
    # The 2-D (N, 1) form works and returns (N, 1).
    assert np.asarray(f(jnp.asarray(X))).shape == (100, 1)


@pytest.mark.backend
def test_jax_hessian_finite_at_mean(plain_emu):
    f = create_jax_emulator(plain_emu)
    x = jnp.asarray(plain_emu.scaler_X.mean_)
    hess = jax.hessian(lambda v: f(v).sum())(x)
    assert np.all(np.isfinite(np.asarray(hess)))


@pytest.mark.backend
def test_torch_vmap_grad_finite(plain_emu):
    tm = TorchMomentEmu(plain_emu)
    rng = np.random.default_rng(13)
    X = torch.as_tensor(rng.uniform(-1.0, 1.0, (1000, 3)), dtype=torch.float64)

    def scalar(x):
        return tm(x).sum()

    grads = torch.func.vmap(torch.func.grad(scalar))(X)
    assert grads.shape == (1000, 3)
    assert bool(torch.isfinite(grads).all())
