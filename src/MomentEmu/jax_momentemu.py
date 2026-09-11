"""JAX backend on the shared monomial plan (P2.1, D2).

A fitted PolyEmu converts to a frozen dataclass registered with
``jax.tree_util.register_dataclass``.  The array leaves are the folded
coefficients, the input scaler, the training box and the P0.7 plan tables
(int32); the static metadata is the parameter/output counts, the maximum
degree, the log flags and the direction.  ``evaluate`` is un-jitted so it can
be composed inside a user log-density; ``__call__`` is the jax.jit entry
point, and value_and_grad / jacfwd / hessian helpers wrap it.  There is no
equinox dependency and no eqx.filter_jit.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from MomentEmu.guards import output_scale


def _evaluate_impl(emulator, X):
    return emulator.evaluate(X)


_JITTED_EVALUATE = jax.jit(_evaluate_impl)


@dataclass(frozen=True)
class JaxEmulator:
    """JAX pytree emulator for a fitted forward or backward PolyEmu."""

    coeffs: object       # (D, m) folded affine coefficients
    input_mean: object
    input_scale: object
    box_lo: object
    box_hi: object
    closure_parent: object
    closure_var: object
    level_rows: object
    select: object
    n_params: int
    n_outputs: int
    dmax: int
    level_sizes: tuple
    input_codes: tuple
    output_codes: tuple
    direction: str
    dtype: object

    def evaluate(self, X):
        """Un-jitted evaluation: safe inside a user jax.jit log-density."""
        X = jnp.asarray(X, dtype=self.dtype)
        if X.ndim == 0:
            if self.n_params != 1:
                raise ValueError(
                    f"scalar input given but the emulator has n_params = {self.n_params}"
                )
            single = True
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            if X.shape[0] != self.n_params:
                raise ValueError(
                    f"1-D input has {X.shape[0]} elements; expected n_params = "
                    f"{self.n_params}. A 1-D array is a single sample only when its "
                    f"length equals n_params."
                )
            single = True
            X = X.reshape(1, self.n_params)
        else:
            if X.shape[-1] != self.n_params:
                raise ValueError(
                    f"input has {X.shape[-1]} elements along its last axis; expected "
                    f"n_params = {self.n_params}"
                )
            single = False
        if any(self.input_codes):
            X = self._apply_forward(X, self.input_codes)
        Xs = (X - self.input_mean) / self.input_scale
        N = Xs.shape[0]
        Dc = self.closure_parent.shape[0]
        buf = jnp.ones((Dc, N), dtype=self.dtype)
        xT = Xs.T
        offset = 0
        for size in self.level_sizes:
            rows = self.level_rows[offset:offset + size]
            parent = buf[self.closure_parent[rows]]
            var = xT[self.closure_var[rows]]
            buf = buf.at[rows].set(parent * var)
            offset += size
        Phi = buf[self.select].T
        Y = Phi @ self.coeffs
        if any(self.output_codes):
            Y = self._apply_inverse(Y, self.output_codes)
        return Y[0] if single else Y

    @staticmethod
    def _apply_forward(X, codes):
        """Apply the forward per-column transform (1 log, 2 asinh)."""
        if len(set(codes)) == 1:
            c = codes[0]
            return jnp.log(X) if c == 1 else jnp.arcsinh(X) if c == 2 else X
        cols = [X[:, j] for j in range(X.shape[1])]
        for j, c in enumerate(codes):
            if c == 1:
                cols[j] = jnp.log(cols[j])
            elif c == 2:
                cols[j] = jnp.arcsinh(cols[j])
        return jnp.stack(cols, axis=1)

    @staticmethod
    def _apply_inverse(Y, codes):
        """Apply the inverse per-column transform (1 exp, 2 sinh)."""
        if len(set(codes)) == 1:
            c = codes[0]
            return jnp.exp(Y) if c == 1 else jnp.sinh(Y) if c == 2 else Y
        cols = [Y[:, j] for j in range(Y.shape[1])]
        for j, c in enumerate(codes):
            if c == 1:
                cols[j] = jnp.exp(cols[j])
            elif c == 2:
                cols[j] = jnp.sinh(cols[j])
        return jnp.stack(cols, axis=1)

    def __call__(self, X):
        """jax.jit-evaluated prediction (the arrays of self are arguments)."""
        return _JITTED_EVALUATE(self, X)

    def value_and_grad(self, X, *, sum_outputs=True):
        f = (lambda x: self.evaluate(x).sum()) if sum_outputs else self.evaluate
        return jax.value_and_grad(f)(X)

    def jacobian(self, X):
        return jax.jacfwd(self.evaluate)(X)

    def hessian(self, X):
        return jax.hessian(lambda x: self.evaluate(x).sum())(X)

    @classmethod
    def from_polyemu(cls, emulator, *, direction="forward", dtype=None):
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be forward or backward, got {direction!r}")
        if dtype is None:
            dtype = jnp.float64
        if dtype == jnp.float64 and not jax.config.jax_enable_x64:
            raise ValueError(
                "the JAX backend needs jax.config.update('jax_enable_x64', True) for "
                "float64; enable x64 or pass dtype=jnp.float32 explicitly."
            )
        from MomentEmu.emulator import _normalize_transform, _transform_codes

        transform = getattr(emulator, "transform", None)
        if transform is None:
            transform = tuple(
                "log" if emulator.log_Y else "linear" for _ in range(emulator.n_outputs)
            )
        codes = _transform_codes(transform)
        linear = tuple(0 for _ in range(emulator.n_outputs))
        if direction == "forward":
            if not hasattr(emulator, "forward_coeffs"):
                raise ValueError("the JAX backend needs a forward emulator")
            coeffs = emulator.forward_coeffs_folded
            plan = emulator.forward_plan
            input_mean = emulator.scaler_X.mean_
            input_scale = emulator.scaler_X.scale_
            box = emulator.X_box_
            input_codes = tuple(0 for _ in range(emulator.n_params))
            output_codes = codes
            in_dim, out_dim = int(emulator.n_params), int(emulator.n_outputs)
        else:
            if not hasattr(emulator, "backward_coeffs"):
                raise ValueError("the JAX backend needs a backward emulator")
            coeffs = emulator.backward_coeffs_folded
            plan = emulator.backward_plan
            input_mean = emulator.scaler_Y.mean_
            input_scale = output_scale(emulator.scaler_Y, emulator.n_outputs)
            box = emulator.Y_box_
            input_codes = codes
            output_codes = linear
            in_dim, out_dim = int(emulator.n_outputs), int(emulator.n_params)
        levels = plan.levels
        level_sizes = tuple(int(a.shape[0]) for a in levels)
        level_rows = np.concatenate(levels) if levels else np.zeros(0, dtype=np.int64)
        return cls(
            coeffs=jnp.asarray(coeffs, dtype=dtype),
            input_mean=jnp.asarray(input_mean, dtype=dtype),
            input_scale=jnp.asarray(input_scale, dtype=dtype),
            box_lo=jnp.asarray(box.lo, dtype=dtype),
            box_hi=jnp.asarray(box.hi, dtype=dtype),
            closure_parent=jnp.asarray(plan.parent, dtype=jnp.int32),
            closure_var=jnp.asarray(plan.var, dtype=jnp.int32),
            level_rows=jnp.asarray(level_rows, dtype=jnp.int32),
            select=jnp.asarray(plan.select, dtype=jnp.int32),
            n_params=in_dim,
            n_outputs=out_dim,
            dmax=int(plan.max_degree),
            level_sizes=level_sizes,
            input_codes=input_codes,
            output_codes=output_codes,
            direction=direction,
            dtype=dtype,
        )


jax.tree_util.register_dataclass(
    JaxEmulator,
    data_fields=[
        "coeffs",
        "input_mean",
        "input_scale",
        "box_lo",
        "box_hi",
        "closure_parent",
        "closure_var",
        "level_rows",
        "select",
    ],
    meta_fields=[
        "n_params",
        "n_outputs",
        "dmax",
        "level_sizes",
        "input_codes",
        "output_codes",
        "direction",
        "dtype",
    ],
)


def create_jax_emulator(emulator, *, dtype=None):
    """Convert a fitted PolyEmu to a jax.jit callable (legacy shim, P2.1).

    ``legacy_positional`` keeps the old signature; the returned JaxEmulator is
    callable exactly like the old closure and also exposes evaluate(),
    value_and_grad(), jacobian() and hessian().
    """
    return JaxEmulator.from_polyemu(emulator, direction="forward", dtype=dtype)


def evaluate_monomials_jax_static(X_scaled, multi_indices):
    """Legacy static-exponent monomial build (kept for the P0.8 tests)."""
    X_scaled = jnp.asarray(X_scaled)
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)
    N = X_scaled.shape[0]
    columns = []
    for alpha in np.asarray(multi_indices):
        monomial = jnp.ones(N, dtype=X_scaled.dtype)
        for i, deg in enumerate(alpha):
            if deg > 0:
                monomial = monomial * (X_scaled[:, i] ** int(deg))
        columns.append(monomial)
    return jnp.stack(columns, axis=1)


def log_prior_box(x, box_lo, box_hi):
    """0 inside the closed box, -inf outside (a numpyro-compatible indicator).

    Use it as a numpyro factor or add it to a log-density so a sampler never
    wanders into the region where the polynomial is unconstrained.
    """
    x = jnp.asarray(x)
    inside = jnp.all((x >= box_lo) & (x <= box_hi))
    return jnp.where(inside, 0.0, -jnp.inf)


def make_guarded_logdensity(emulator, logprior, *, penalty=1e10):
    """Guarded log-density: clip into the box, finite penalty outside (P2.5a).

    The polynomial is always evaluated at ``clip(x, box_lo, box_hi)``, so it
    never sees an out-of-box point (the review measured evaluations up to 1e24
    half-widths out and values 1e191 times the truth under a wide prior).
    ``jnp.clip`` has zero derivative in the clipped region and the penalty is a
    constant branch, so the log-density has a finite value and exactly zero
    gradient outside the box. The numpyro prior should be the training box or
    narrower.

    Parameters
    ----------
    emulator : JaxEmulator
        Typically an un-jitted forward emulator; call it through the returned
        function inside a user jit.
    logprior : callable
        Prior log-density of x (may be constant).
    penalty : float
        Finite penalty subtracted outside the box.
    """
    lo = emulator.box_lo
    hi = emulator.box_hi

    def logpost(x):
        x = jnp.asarray(x, dtype=emulator.dtype)
        x_clipped = jnp.clip(x, lo, hi)
        y = emulator.evaluate(x_clipped)
        loglike = -0.5 * jnp.sum(y ** 2)
        outside = jnp.any((x < lo) | (x > hi))
        return logprior(x) + loglike + jnp.where(outside, -penalty, 0.0)

    return logpost


def demo_jax_autodiff():
    """Demonstrate the JAX backend on a small quadratic emulator."""
    from MomentEmu.emulator import PolyEmu

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (200, 2))
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=2, verbose=0)
    f = create_jax_emulator(emu)
    x = jnp.array([0.5, 0.3])
    print("prediction:", f(x))
    print("gradient:", f.value_and_grad(x)[1])
    print("hessian:", f.hessian(x))


if __name__ == "__main__":
    demo_jax_autodiff()
