"""JAX backend for MomentEmu (P0.8 stopgap; rebuilt on the P0.7 plan in P2.1).

This module wraps a fitted :class:`~MomentEmu.PolyEmu.PolyEmu` in a
jax.jit-able function.  The stopgap refuses inputs the old implementation
answered wrongly (log_Y, a missing output scale) and evaluates monomials with
static Python-int exponents so zero exponents never enter an autodiff graph
(the NaN-at-the-mean bug).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from MomentEmu.guards import check_backend_supports, output_scale


def evaluate_monomials_jax_static(X_scaled, multi_indices, *, x64=None):
    """Static-exponent monomial evaluation (skips degree 0).

    ``multi_indices`` is a NumPy int array, so every exponent is a Python int
    at trace time.  The old ``X ** multi_indices`` made JAX differentiate
    ``x ** 0``, whose derivative is NaN at x = 0 (the training mean after
    standardisation).
    """
    X_scaled = jnp.asarray(X_scaled)
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)
    N = X_scaled.shape[0]
    mi = np.asarray(multi_indices)
    columns = []
    for alpha in mi:
        monomial = jnp.ones(N, dtype=X_scaled.dtype)
        for i, deg in enumerate(alpha):
            if deg > 0:
                monomial = monomial * (X_scaled[:, i] ** int(deg))
        columns.append(monomial)
    return jnp.stack(columns, axis=1)


def create_jax_emulator(emulator):
    """Return a jitted JAX function equivalent to forward_emulator.

    Raises NotImplementedError for a log_Y fit (the backend would otherwise
    return log Y) and ValueError when the emulator has no forward coefficients.
    """
    check_backend_supports(emulator, "jax")

    coeffs = jnp.asarray(emulator.forward_coeffs)
    multi_indices = np.asarray(emulator.forward_multi_indices)
    input_mean = jnp.asarray(emulator.scaler_X.mean_)
    input_scale = jnp.asarray(emulator.scaler_X.scale_)
    output_mean = jnp.asarray(emulator.scaler_Y.mean_)
    output_scale_ = jnp.asarray(output_scale(emulator.scaler_Y, emulator.n_outputs))
    n_params = int(emulator.n_params)
    n_outputs = int(emulator.n_outputs)

    @jax.jit
    def jax_emulator(X):
        X = jnp.asarray(X)
        if X.ndim == 0:
            if n_params != 1:
                raise ValueError(
                    f"scalar input given but the emulator has n_params = {n_params}"
                )
            single = True
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            if X.shape[0] != n_params:
                raise ValueError(
                    f"1-D input has {X.shape[0]} elements; expected n_params = "
                    f"{n_params}. A 1-D array is a single sample only when its "
                    f"length equals n_params; pass a 2-D (N, n_params) array "
                    f"otherwise."
                )
            single = True
            X = X.reshape(1, n_params)
        else:
            if X.shape[-1] != n_params:
                raise ValueError(
                    f"input has {X.shape[-1]} elements along its last axis; "
                    f"expected n_params = {n_params}"
                )
            single = False
        X_scaled = (X - input_mean) / input_scale
        Phi = evaluate_monomials_jax_static(X_scaled, multi_indices)
        Y = Phi @ coeffs * output_scale_ + output_mean
        if single:
            Y = Y[0]
        return Y

    return jax_emulator


def demo_jax_autodiff():
    """Demonstrate JAX gradients on a small quadratic emulator."""
    from MomentEmu.PolyEmu import PolyEmu

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (200, 2))
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
    emu = PolyEmu(X, Y, cross_validation=False, max_degree_forward=2, dim_reduction=False)
    f = create_jax_emulator(emu)
    x = jnp.array([0.5, 0.3])
    print("prediction:", f(x))
    print("gradient:", jax.grad(lambda v: f(v).sum())(x))
    print("hessian:", jax.hessian(lambda v: f(v).sum())(x))


if __name__ == "__main__":
    demo_jax_autodiff()
