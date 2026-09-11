"""P2.5a: guarded JAX potential (finite, zero-gradient outside the box)."""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from MomentEmu.emulator import PolyEmu  # noqa: E402
from MomentEmu.jax_momentemu import (  # noqa: E402
    create_jax_emulator,
    log_prior_box,
    make_guarded_logdensity,
)


def _fit(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + 5.0).reshape(-1, 1)
    return PolyEmu(X, Y, max_degree_forward=3, verbose=0)


@pytest.mark.backend
def test_guarded_value_finite_and_grad_zero_outside():
    emu = _fit()
    f = create_jax_emulator(emu)
    logpost = make_guarded_logdensity(f, lambda x: jnp.array(0.0))
    lo, hi = np.asarray(f.box_lo), np.asarray(f.box_hi)
    center = 0.5 * (lo + hi)
    # 10 box widths out along every axis.
    x_far = jnp.asarray(center + 10.0 * (hi - lo))
    val = logpost(x_far)
    assert np.isfinite(float(val))
    g = jax.grad(logpost)(x_far)
    np.testing.assert_allclose(np.asarray(g), 0.0, atol=1e-12)
    # Inside, the gradient is generally non-zero.
    g_in = jax.grad(logpost)(jnp.asarray(center))
    assert np.any(np.abs(np.asarray(g_in)) > 1e-9)


@pytest.mark.backend
def test_guarded_clips_to_the_box():
    emu = _fit(seed=1)
    f = create_jax_emulator(emu)
    penalty = 1e10
    logpost = make_guarded_logdensity(f, lambda x: jnp.array(0.0), penalty=penalty)
    lo, hi = np.asarray(f.box_lo), np.asarray(f.box_hi)
    center = 0.5 * (lo + hi)
    x_far = jnp.asarray(center + 8.0 * (hi - lo))
    clipped = jnp.clip(x_far, f.box_lo, f.box_hi)
    # The value equals the in-box value at the clipped point minus the penalty,
    # i.e. the polynomial was evaluated at the clipped point.
    assert float(logpost(x_far)) == pytest.approx(float(logpost(clipped)) - penalty, rel=1e-12)


@pytest.mark.backend
def test_log_prior_box():
    lo = jnp.array([0.0, 0.0])
    hi = jnp.array([1.0, 1.0])
    assert float(log_prior_box(jnp.array([0.5, 0.5]), lo, hi)) == 0.0
    assert np.isneginf(float(log_prior_box(jnp.array([1.5, 0.5]), lo, hi)))
