"""P2.1: JAX backend on the shared plan (pytree, jit, derivatives)."""
from __future__ import annotations

import pickle

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from MomentEmu.emulator import (
    PolyEmu,
    evaluate_monomials_lazy,
)
from MomentEmu.jax_momentemu import JaxEmulator, create_jax_emulator  # noqa: E402


def _col_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _fit(seed=0, n=3, log_Y=False, with_std=True, backward=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, n))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, -1]) + 5.0).reshape(-1, 1)
    kw = dict(log_Y=log_Y, standardize_Y_with_std=with_std, max_degree_forward=4, verbose=0)
    if backward:
        kw.update(forward=False, backward=True, init_deg_backward=2, max_degree_backward=2)
    return PolyEmu(X, Y, **kw)


def _points(emu, rng):
    lo, hi = emu.X_box_.lo, emu.X_box_.hi
    corners = np.array(np.meshgrid(*zip(lo, hi))).T.reshape(-1, emu.n_params)
    mean = emu.scaler_X.mean_
    single = np.array([mean if i == j else mean for i, j in enumerate(range(emu.n_params))])
    coord = np.tile(mean, (emu.n_params, 1))
    for i in range(emu.n_params):
        coord[i, i] = lo[i] + 0.3 * (hi[i] - lo[i])
    n_random = max(1, 1000 - 1 - len(corners) - len(coord))
    pts = [mean[None, :], corners, coord, single, rng.uniform(lo, hi, (n_random, emu.n_params))]
    return np.vstack(pts)


@pytest.mark.backend
@pytest.mark.parametrize("log_Y", [False, True])
@pytest.mark.parametrize("with_std", [True, False])
def test_jax_forward_matches_forward_emulator(log_Y, with_std):
    emu = _fit(seed=1, log_Y=log_Y, with_std=with_std)
    f = create_jax_emulator(emu)
    rng = np.random.default_rng(2)
    X = _points(emu, rng)
    got = np.asarray(f(jnp.asarray(X)))
    ref = emu.forward_emulator(X, extrapolation="ignore")
    assert got.shape == ref.shape
    assert _col_rel(got, ref) < 1e-12


@pytest.mark.backend
def test_jax_backward_matches_backward_emulator():
    emu = _fit(seed=3, n=2, backward=True)
    je = JaxEmulator.from_polyemu(emu, direction="backward")
    rng = np.random.default_rng(4)
    Y = rng.uniform(emu.Y_box_.lo, emu.Y_box_.hi, (200, emu.n_outputs))
    got = np.asarray(je(jnp.asarray(Y)))
    ref = emu.backward_emulator(Y, extrapolation="ignore")
    assert _col_rel(got, ref) < 1e-12


@pytest.mark.backend
def test_jax_pruned_non_downward_closed_set():
    emu = _fit(seed=5)
    rng = np.random.default_rng(6)
    mi = emu.forward_multi_indices
    keep = rng.random(mi.shape[0]) > 0.4
    keep[0] = True
    pruned = mi[keep]
    # Rebuild the pruned model on the training rows.
    X = np.random.default_rng(5).uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 5.0).reshape(-1, 1)
    Xs = emu.scaler_X.transform(X)
    Ys = emu.scaler_Y.transform(np.log(Y) if emu.log_Y else Y)
    Phi = evaluate_monomials_lazy(Xs, pruned)
    c = np.linalg.lstsq(Phi, Ys, rcond=None)[0]
    emu.forward_multi_indices = pruned
    emu.forward_coeffs = c
    emu._build_forward_plan()
    f = create_jax_emulator(emu)
    pts = rng.uniform(emu.X_box_.lo, emu.X_box_.hi, (200, 3))
    got = np.asarray(f(jnp.asarray(pts)))
    ref = emu.forward_emulator(pts, extrapolation="ignore")
    assert _col_rel(got, ref) < 1e-12


@pytest.mark.backend
def test_jax_hessian_finite_and_picklable():
    emu = _fit(seed=7)
    f = create_jax_emulator(emu)
    x = jnp.asarray(emu.scaler_X.mean_)
    assert np.all(np.isfinite(np.asarray(f.hessian(x))))
    g = pickle.loads(pickle.dumps(f))
    assert _col_rel(np.asarray(g(x)), np.asarray(f(x))) < 1e-12


@pytest.mark.backend
def test_jax_un_jitted_evaluate_inside_user_jit():
    emu = _fit(seed=8)
    f = create_jax_emulator(emu)

    @jax.jit
    def logdensity(x):
        return -0.5 * jnp.sum(f.evaluate(x) ** 2)

    x = jnp.asarray(emu.scaler_X.mean_)
    assert np.isfinite(float(logdensity(x)))
    assert np.all(np.isfinite(np.asarray(jax.grad(logdensity)(x))))


@pytest.mark.backend
def test_jax_x64_guard():
    emu = _fit(seed=9)
    old = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="enable_x64"):
            JaxEmulator.from_polyemu(emu)
        JaxEmulator.from_polyemu(emu, dtype=jnp.float32)  # explicit float32 is allowed
    finally:
        jax.config.update("jax_enable_x64", old)


@pytest.mark.backend
def test_jax_value_and_grad_matches_jacobian():
    emu = _fit(seed=10)
    f = create_jax_emulator(emu)
    x = jnp.asarray(emu.scaler_X.mean_)
    J = np.asarray(f.jacobian(x))
    for j in range(emu.n_outputs):
        g = jax.grad(lambda v: f.evaluate(v)[j])(x)
        np.testing.assert_allclose(np.asarray(g), J[j], rtol=1e-12, atol=1e-12)


@pytest.mark.backend
def test_jax_derivative_helpers_use_the_jitted_path(monkeypatch):
    """P2.1: value_and_grad/jacobian/hessian must not differentiate the
    un-jitted Python evaluator (referee 2026-09-12: 21-25 ms per call).
    After the first compile, monkeypatching evaluate must not be hit."""
    emu = _fit(seed=11)
    f = create_jax_emulator(emu)
    x = jnp.asarray(emu.scaler_X.mean_)
    f.value_and_grad(x)
    f.jacobian(x)
    f.hessian(x)
    calls = []
    original = JaxEmulator.evaluate

    def spy(self, X):
        calls.append(1)
        return original(self, X)

    monkeypatch.setattr(JaxEmulator, "evaluate", spy)
    f.value_and_grad(x)
    f.jacobian(x)
    f.hessian(x)
    assert calls == []

