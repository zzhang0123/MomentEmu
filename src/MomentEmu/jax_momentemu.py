"""JAX backend on the shared basis plans (P2.1, D2; T-007 for the tensor path).

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

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from MomentEmu.guards import output_scale


def _evaluate_impl(emulator, X):
    return emulator.evaluate(X)


def _value_impl(emulator, X):
    return emulator.evaluate(X).sum()


_JITTED_EVALUATE = jax.jit(_evaluate_impl)
# help(P2.1): the value/grad/jacobian/hessian helpers must differentiate the
# traced evaluator, not the un-jitted Python one. Differentiating the un-jitted
# `evaluate` costs ~20 ms per call instead of ~0.1 ms (referee 2026-09-12).
_JITTED_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(_value_impl, argnums=1))
_JITTED_JACFWD = jax.jit(jax.jacfwd(_evaluate_impl, argnums=1))
_JITTED_HESSIAN = jax.jit(jax.hessian(_value_impl, argnums=1))


def _one_dimensional_table(z, family, table_degree):
    """``(N, table_degree + 1)`` values of the 1-D family at ``z``.

    Mirrors ``monomials.LegendrePlan._one_dimensional`` and
    ``ChebyshevPlan._one_dimensional``; the tests pin the two against each
    other through the numpy fit, because a silent divergence here returns
    plausible numbers from the wrong basis.
    """
    if family not in ("monomial", "legendre", "chebyshev"):
        raise NotImplementedError(f"no 1-D recurrence for basis {family!r}")
    columns = [jnp.ones_like(z)]
    if table_degree >= 1:
        columns.append(z)
    for k in range(1, table_degree):
        if family == "legendre":
            # (k+1) P_{k+1} = (2k+1) z P_k - k P_{k-1}
            nxt = ((2 * k + 1) * z * columns[k] - k * columns[k - 1]) / (k + 1)
        elif family == "monomial":
            # z^{k+1} = z * z^k. Named rather than left to the else, which
            # would have evaluated monomials with the Chebyshev recurrence.
            nxt = z * columns[k]
        else:
            # T_{k+1} = 2 z T_k - T_{k-1}
            nxt = 2.0 * z * columns[k] - columns[k - 1]
        columns.append(nxt)
    table = jnp.stack(columns, axis=1)
    if family == "legendre":
        degrees = jnp.arange(table_degree + 1, dtype=table.dtype)
        table = table * jnp.sqrt(2.0 * degrees + 1.0)
    return table


def _tensor_design(Xs, multi_indices, family, table_degree, n_params):
    """``(N, D)`` design for a tensor-product basis, ``prod_i p_{alpha_i}(z_i)``.

    The argument is clipped to [-1, 1] for the families ``_TensorPlan``
    clips, so the basis saturates outside the training box rather than
    diverging, and the caller's extrapolation guard still reports the
    excursion. Monomials are NOT clipped: ``MonomialPlan`` does not, and a
    monomial fit describes a diverging model out there. Clipping them would
    agree inside the box and describe a different model outside it.
    """
    Z = Xs if family == "monomial" else jnp.clip(Xs, -1.0, 1.0)
    out = jnp.ones((Z.shape[0], multi_indices.shape[0]), dtype=Xs.dtype)
    for i in range(n_params):
        table = _one_dimensional_table(Z[:, i], family, table_degree)
        out = out * jnp.take(table, multi_indices[:, i], axis=1)
    return out


def _resolve_dtype(dtype):
    """Default to float64 and refuse it when x64 is off, naming the fix."""
    if dtype is None:
        dtype = jnp.float64
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        raise ValueError(
            "the JAX backend needs jax.config.update('jax_enable_x64', True) for "
            "float64; enable x64 or pass dtype=jnp.float32 explicitly."
        )
    return dtype


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
    multi_indices: object
    n_params: int
    n_outputs: int
    dmax: int
    level_sizes: tuple
    basis_family: str
    table_degree: int
    input_codes: tuple
    output_codes: tuple
    direction: str
    dtype: object
    # Any rather than object: these are indexed and matmul'd, which the
    # house "object" annotation on the other arrays does not allow.
    output_modes: Any = None      # (k, m) or None when uncompressed
    output_offset: Any = None     # (m,) or None

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
        # basis_family is static metadata, so this branch is resolved at trace
        # time and the unused path never reaches the graph.
        # The closure walk needs a DOWNWARD CLOSED index set. A dense fit has
        # one and carries its level tables; a sparse selection does not, which
        # is what sparsity means, and arrives with none. Branch on the tables
        # rather than on the family, so both reach the right path.
        if self.basis_family == "monomial" and self.level_sizes:
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
        else:
            Phi = _tensor_design(Xs, self.multi_indices, self.basis_family,
                                 self.table_degree, self.n_params)
        Y = Phi @ self.coeffs
        if self.output_modes is not None:
            # T-011: the fit carries k output modes; expand to the m the
            # caller asked for. Derived in CompressedEmu.export_payload.
            Y = Y @ self.output_modes + self.output_offset
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
        """Jitted (value, gradient); sum_outputs=False returns (value, jacobian)."""
        if sum_outputs:
            return _JITTED_VALUE_AND_GRAD(self, X)
        return _JITTED_EVALUATE(self, X), _JITTED_JACFWD(self, X)

    def jacobian(self, X):
        """Jitted Jacobian of the jitted evaluator."""
        return _JITTED_JACFWD(self, X)

    def hessian(self, X):
        """Jitted Hessian of the summed output."""
        return _JITTED_HESSIAN(self, X)

    @classmethod
    @classmethod
    def from_compressed(cls, emulator, *, dtype=None):
        """Export a :class:`MomentEmu.compress.CompressedEmu`.

        The inner model supplies everything but the last stage; the payload
        supplies the (D, k) coefficients and the (k, m) map, so the exported
        arrays are the compressed ones rather than the expansion of them.
        """
        dtype = _resolve_dtype(dtype)
        pay = emulator.export_payload()
        inner = cls.from_polyemu(emulator.model, dtype=dtype)
        return dataclasses.replace(
            inner,
            coeffs=jnp.asarray(pay.coeffs, dtype=dtype),
            n_outputs=int(pay.modes.shape[1]),
            output_codes=tuple(0 for _ in range(int(pay.modes.shape[1]))),
            output_modes=jnp.asarray(pay.modes, dtype=dtype),
            output_offset=jnp.asarray(pay.offset, dtype=dtype),
        )

    @classmethod
    def from_sparse(cls, emulator, *, dtype=None):
        """Export a :class:`MomentEmu.sparse.SparseEmu` fitted in a tensor basis.

        Every family goes through the multi-index path, not the closure walk.
        A SELECTED index set is not downward closed -- that is what sparsity
        means -- and the closure walk needs one; rebuilding it would put back
        exactly the terms the selection removed, so the fit that arrived would
        not be the fit evaluated.
        """
        from MomentEmu.emulator import BASIS_PLANS
        from MomentEmu.monomials import fold_output_affine

        dtype = _resolve_dtype(dtype)
        family = str(getattr(emulator, "basis_kind", "legendre"))
        mi = np.asarray(emulator.multi_indices, dtype=np.int64)
        plan_cls: Any = BASIS_PLANS[family]
        plan = plan_cls.build(mi)
        # SparseEmu maps its training box onto [-1, 1] as
        # 2 (X - lo) / span - 1, which is (X - mean) / scale with these two.
        span = np.asarray(emulator.span_, dtype=np.float64)
        input_scale = 0.5 * span
        input_mean = np.asarray(emulator.lo_, dtype=np.float64) + input_scale
        coeffs = fold_output_affine(
            np.asarray(emulator.coefficients, dtype=np.float64), mi,
            np.asarray(emulator.mean_Y_, dtype=np.float64),
            np.asarray(emulator.scale_Y_, dtype=np.float64),
        )
        empty = np.zeros(0, dtype=np.int64)
        n_out = int(coeffs.shape[1])
        return cls(
            coeffs=jnp.asarray(coeffs, dtype=dtype),
            input_mean=jnp.asarray(input_mean, dtype=dtype),
            input_scale=jnp.asarray(input_scale, dtype=dtype),
            box_lo=jnp.asarray(emulator.lo_, dtype=dtype),
            box_hi=jnp.asarray(emulator.hi_, dtype=dtype),
            closure_parent=jnp.asarray(empty, dtype=jnp.int32),
            closure_var=jnp.asarray(empty, dtype=jnp.int32),
            level_rows=jnp.asarray(empty, dtype=jnp.int32),
            select=jnp.asarray(empty, dtype=jnp.int32),
            multi_indices=jnp.asarray(mi, dtype=jnp.int32),
            n_params=int(mi.shape[1]),
            n_outputs=n_out,
            dmax=int(plan.max_degree),
            level_sizes=(),
            basis_family=family,
            table_degree=int(mi.max()) if mi.size else 0,
            input_codes=tuple(0 for _ in range(int(mi.shape[1]))),
            output_codes=tuple(0 for _ in range(n_out)),
            direction="forward",
            dtype=dtype,
        )

    @classmethod
    def from_polyemu(cls, emulator, *, direction="forward", dtype=None):
        if hasattr(emulator, "export_payload"):
            if direction != "forward":
                raise ValueError(
                    "a compressed model has no backward map; pass "
                    "direction='forward'"
                )
            return cls.from_compressed(emulator, dtype=dtype)
        if hasattr(emulator, "candidate_indices") and not hasattr(
            emulator, "forward_multi_indices"
        ):
            # A SparseEmu. Callers reach for the name they know, so dispatch
            # rather than fail on the first PolyEmu attribute that is missing.
            if direction != "forward":
                raise ValueError(
                    "a sparse fit has no backward map; pass direction='forward'"
                )
            return cls.from_sparse(emulator, dtype=dtype)
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be forward or backward, got {direction!r}")
        dtype = _resolve_dtype(dtype)
        from MomentEmu.emulator import _transform_codes

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
        family = getattr(plan, "family", "monomial")
        empty = np.zeros(0, dtype=np.int64)
        if family == "monomial":
            levels = plan.levels
            level_sizes = tuple(int(a.shape[0]) for a in levels)
            level_rows = np.concatenate(levels) if levels else empty
            closure_parent, closure_var, select = plan.parent, plan.var, plan.select
            multi_indices = np.zeros((0, 0), dtype=np.int64)
            table_degree = 0
        else:
            # A tensor-product basis is evaluated from its multi-indices and
            # the family's own recurrence; the monomial closure tables have no
            # meaning for it, so they are left empty rather than rebuilt from
            # the same indices, which would silently evaluate a DIFFERENT
            # basis (T-007: the Torch backend did exactly that).
            level_sizes, level_rows = (), empty
            closure_parent = closure_var = select = empty
            multi_indices = np.asarray(plan.multi_indices, dtype=np.int64)
            table_degree = int(multi_indices.max()) if multi_indices.size else 0
        return cls(
            coeffs=jnp.asarray(coeffs, dtype=dtype),
            input_mean=jnp.asarray(input_mean, dtype=dtype),
            input_scale=jnp.asarray(input_scale, dtype=dtype),
            box_lo=jnp.asarray(box.lo, dtype=dtype),
            box_hi=jnp.asarray(box.hi, dtype=dtype),
            closure_parent=jnp.asarray(closure_parent, dtype=jnp.int32),
            closure_var=jnp.asarray(closure_var, dtype=jnp.int32),
            level_rows=jnp.asarray(level_rows, dtype=jnp.int32),
            select=jnp.asarray(select, dtype=jnp.int32),
            multi_indices=jnp.asarray(multi_indices, dtype=jnp.int32),
            n_params=in_dim,
            n_outputs=out_dim,
            dmax=int(plan.max_degree),
            level_sizes=level_sizes,
            basis_family=str(family),
            table_degree=table_degree,
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
        "multi_indices",
        # T-011: arrays, so they are data fields and ride through a jit/vmap
        # trace. None when the model is uncompressed, which a pytree tolerates.
        "output_modes",
        "output_offset",
    ],
    meta_fields=[
        "n_params",
        "n_outputs",
        "dmax",
        "level_sizes",
        "basis_family",
        "table_degree",
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
