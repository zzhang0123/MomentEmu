"""Sparse basis selection over a large candidate set (T-002 / P3).

``Basis`` truncations are *prior*: they fix the index set before seeing the
response. Sparse polynomial chaos selects it *from* the response instead
(Blatman & Sudret 2011, J. Comput. Phys. 230, 2345), which expresses
asymmetries no fixed rule can state -- one pair of parameters needing degree 15
while another needs 2.

Three things shape this implementation.

Selection runs in the orthonormal Legendre product basis, not in monomials.
Greedy selection needs a dictionary of low mutual coherence, and monomials are
the opposite: on [-1, 1] the normalised correlation of the columns z^10 to z^18
with a pure z^12 signal spans 0.1981 to 0.2000, a 0.5 percent spread that
sampling noise overturns, so the greedy step picks a neighbouring power at
random. Legendre columns are orthogonal under a uniform design, which is what
makes the greedy step well posed and is why sparse polynomial chaos is
formulated that way (Blatman & Sudret 2011). Sparsity is a property of a basis,
not of a function: a target that is four monomials is not four Legendre terms,
and the reverse. What transfers between bases is accuracy per retained term,
which is what this optimises.

A Gram matrix over the whole candidate set is not an option. The point of the
method is a large candidate set, and ``D x D`` at D = 86,976 is 60 GB. The
selection therefore works from ``Phi`` with an incrementally extended Cholesky
factor of the ACTIVE block only, which is ``k x k``.

The index set is shared across outputs. Selecting per output would give each
its own set and cost the single-GEMM inference path that makes this package
fast, so the criterion is simultaneous (block) OMP: score a candidate by the
norm of its correlation across all outputs at once.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from MomentEmu.emulator import BASIS_PLANS, generate_multi_indices
from MomentEmu.guards import COND_WARN, IllConditionedWarning, as_float64, check_finite

#: Refuse to materialise a design matrix larger than this, in bytes.
_DESIGN_BUDGET = 4 * 1024 ** 3


def simultaneous_omp(
    Phi: np.ndarray,
    Y: np.ndarray,
    n_terms: int,
    force: Any = (),
    tol: float = 0.0,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedy simultaneous orthogonal matching pursuit.

    At each step the candidate whose column is most correlated with the current
    residual, measured across every output at once, joins the active set and
    the active block is re-solved exactly. The Cholesky factor of the active
    Gram is extended by one row and column per step, so a step costs O(N D) for
    the new Gram column and O(k^2) for the solve, not O(k^3).

    Args:
        Phi: (N, D) candidate design matrix.
        Y: (N, m) targets.
        n_terms: maximum size of the active set.
        force: candidate columns to seed the active set with, in order (the
            constant term belongs here).
        tol: stop when the best normalised correlation falls to this.
        ridge: Tikhonov term on the active solve.

    Returns:
        (active, coefficients, info) with active the selected columns in the
        order chosen, coefficients (k, m), and info carrying the residual path.

    The residual path is computed from ``||Y||^2 - c.b_A``, which cancels
    catastrophically once the fit is nearly exact: it bottoms out near the
    square root of machine epsilon (around 1e-8 relative) while the
    coefficients themselves stay accurate to 1e-15. Use it to watch progress,
    not as a machine-precision stopping rule.
    """
    from scipy.linalg import solve_triangular

    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    N, D = Phi.shape
    n_terms = int(min(max(int(n_terms), 1), D, N))

    b = Phi.T @ Y                                    # (D, m)
    diag = np.einsum("ij,ij->j", Phi, Phi)           # squared column norms
    scale = np.sqrt(np.maximum(diag, np.finfo(float).tiny))
    total = float(np.sum(Y * Y)) or 1.0

    active: list[int] = []
    # The pivot guard below is per column, but conditioning accumulates, so it
    # is tracked across the active set. lambda_min(G_AA) <= min_k L[k,k]**2 and
    # lambda_max(G_AA) >= max_k G[k,k], so this ratio is a lower bound on
    # cond(G_AA), costing O(1) per step. It runs about 4x low near the guard,
    # so it is used only to decide when the exact O(k**3) check is worth doing,
    # and that check runs only when a new smallest pivot appears.
    max_gjj, min_pivot, warned = 0.0, np.inf, False
    cond_estimate = 1.0
    G_active = np.empty((D, n_terms))                # Phi^T Phi[:, active]
    L = np.zeros((n_terms, n_terms))
    corr = b.copy()
    coef = np.zeros((0, Y.shape[1]))
    residual_path: list[float] = []
    queue = [int(j) for j in force]

    # The active set can fall behind the iteration count, because a candidate
    # that turns out to lie in the span of the active set is skipped rather
    # than added. Index the factor by len(active), never by the step counter.
    attempts = 0
    max_attempts = n_terms + min(D, 10 * n_terms)
    while len(active) < n_terms and attempts < max_attempts:
        attempts += 1
        k = len(active)
        if queue:
            j = queue.pop(0)
            if j in active:
                continue
        else:
            score = np.linalg.norm(corr, axis=1) / scale
            if active:
                score[active] = -np.inf
            j = int(np.argmax(score))
            if not np.isfinite(score[j]) or score[j] <= tol:
                break

        g = Phi.T @ Phi[:, j]                        # (D,) new Gram column
        w = np.zeros(k)
        if k:
            w = solve_triangular(
                L[:k, :k], g[active], lower=True, check_finite=False
            )
        pivot = g[j] + ridge - float(w @ w)
        if pivot <= 1e-12 * max(g[j], 1.0):
            # Numerically in the span of the active set: adding it would only
            # wreck the factor, so drop it from contention and move on.
            corr[j] = 0.0
            continue
        G_active[:, k] = g
        L[k, :k] = w
        L[k, k] = np.sqrt(pivot)
        active.append(j)
        k += 1
        max_gjj = max(max_gjj, float(g[j]))
        dropped = float(pivot) < min_pivot
        min_pivot = min(min_pivot, float(pivot))
        if not warned and dropped and min_pivot > 0.0:
            bound = max_gjj / min_pivot
            if bound >= COND_WARN * 0.01:
                cond_estimate = float(np.linalg.cond(L[:k, :k])) ** 2
                if cond_estimate >= COND_WARN:
                    warned = True
                    warnings.warn(
                        f"the active Gram reached cond {cond_estimate:.2e} "
                        f"after {k} term(s), past the {COND_WARN:.0e} level at "
                        "which the dense solver already calls its coefficients "
                        "untrustworthy. The greedy step admitted a column "
                        "nearly in the span of the ones before it: the per "
                        "column pivot guard bounds one step, not the "
                        "accumulation. Drop the last terms or separate the "
                        "candidate set.",
                        IllConditionedWarning,
                        stacklevel=2,
                    )

        rhs = b[active]                              # (k, m)
        z = solve_triangular(L[:k, :k], rhs, lower=True, check_finite=False)
        coef = solve_triangular(L[:k, :k].T, z, lower=False, check_finite=False)
        corr = b - G_active[:, :k] @ coef
        # ||Y - Phi_A c||^2 = ||Y||^2 - 2 c.b_A + c.(G_AA c); the last two
        # cancel at the exact solve, leaving ||Y||^2 - c.b_A.
        rss = max(total - float(np.sum(coef * rhs)), 0.0)
        residual_path.append(float(np.sqrt(rss / total)))

    return (
        np.array(active, dtype=np.int64),
        coef,
        {
            "residual_path": tuple(residual_path),
            "n_candidates": int(D),
            # Exact cond(G_AA) once the cheap bound made it worth computing,
            # otherwise that bound, which runs low.
            "cond": max(
                cond_estimate,
                float(max_gjj / min_pivot)
                if min_pivot not in (0.0, np.inf) else 1.0,
            ),
        },
    )


class SparseEmu:
    """Forward emulator on a response-selected subset of a large candidate set.

    Args:
        X, Y: training design and outputs.
        candidate: a :class:`~MomentEmu.basis.Basis`, an explicit (D, n)
            multi-index array, or None for the isotropic set of ``degree``.
        degree: degree passed to a Basis candidate, or used to build the
            isotropic default.
        n_terms: size of the active set, or "auto" to take the size that
            minimises the held-out error along the greedy path.
        max_terms: how far the path is walked when ``n_terms="auto"``. A path
            whose minimum lands on its last point was truncated rather than
            converged, and that case warns.
        validation_split: fraction held out when ``n_terms="auto"`` and no
            explicit test set is given.

    Attributes:
        multi_indices: (k, n) selected index set, in the order chosen.
        coefficients: (k, m) coefficients on the standardised scales.
        n_candidates: size of the candidate set searched.
        error_path: held-out error against active-set size, when available.
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        candidate: Any = None,
        degree: int | None = None,
        n_terms: Any = 200,
        max_terms: int = 300,
        X_test: Any = None,
        Y_test: Any = None,
        validation_split: float = 0.15,
        random_state: Any = None,
        tol: float = 0.0,
        ridge: float = 0.0,
        basis_kind: str = "legendre",
        parameter_names: Any = None,
    ) -> None:
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        check_finite(X, "X")
        check_finite(Y, "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if basis_kind not in BASIS_PLANS:
            raise ValueError(
                f"basis_kind must be one of {sorted(BASIS_PLANS)}, got "
                f"{basis_kind!r}"
            )
        self.basis_kind = basis_kind
        n = X.shape[1]
        self.parameter_names = (
            [f"x{i}" for i in range(n)] if parameter_names is None
            else list(parameter_names)
        )

        if candidate is None:
            if degree is None:
                raise ValueError("pass candidate= or degree= to build one")
            mi = generate_multi_indices(n, int(degree))
        elif hasattr(candidate, "build"):
            mi = candidate.build(self.parameter_names, degree)
        else:
            mi = np.asarray(candidate, dtype=np.int64)
            if mi.ndim != 2 or mi.shape[1] != n:
                raise ValueError(
                    f"candidate must be (D, {n}); got shape {mi.shape}"
                )
        self.candidate_indices = mi
        self.n_candidates = int(mi.shape[0])

        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        self.span_ = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        self.mean_Y_, yscale = Y.mean(axis=0), Y.std(axis=0)
        self.scale_Y_ = np.where(yscale > 0.0, yscale, 1.0)

        need = X.shape[0] * self.n_candidates * 8
        if need > _DESIGN_BUDGET:
            raise MemoryError(
                f"the candidate design is {need / 1024 ** 3:.1f} GiB "
                f"({X.shape[0]} x {self.n_candidates}); shrink the candidate "
                "set (a lower degree or max_interaction) or subsample the "
                "training set"
            )
        Xs = self._map(X)
        Ys = (Y - self.mean_Y_) / self.scale_Y_

        constant = np.flatnonzero(~mi.any(axis=1))
        force = [int(constant[0])] if constant.size else []

        auto = isinstance(n_terms, str) and n_terms == "auto"
        if auto:
            if X_test is None:
                rng = np.random.default_rng(random_state)
                order = rng.permutation(X.shape[0])
                cut = max(1, int(round(validation_split * X.shape[0])))
                hold, keep = order[:cut], order[cut:]
                Xf, Yf, Xv, Yv = Xs[keep], Ys[keep], Xs[hold], Ys[hold]
            else:
                if Y_test is None:
                    raise ValueError("X_test needs a matching Y_test")
                Xv = self._map(as_float64(np.asarray(X_test), "X_test"))
                Yv = (as_float64(np.asarray(Y_test), "Y_test").reshape(Xv.shape[0], -1)
                      - self.mean_Y_) / self.scale_Y_
                Xf, Yf = Xs, Ys
            budget = int(min(int(max_terms), Xf.shape[0] - 1, self.n_candidates))
            act, _, _ = simultaneous_omp(
                self._design(Xf), Yf, budget, force=force, tol=tol, ridge=ridge
            )
            Pv, Pf = self._design(Xv), self._design(Xf)
            path = []
            for k in range(1, act.size + 1):
                cols = act[:k]
                c = np.linalg.lstsq(Pf[:, cols], Yf, rcond=None)[0]
                path.append(float(np.sqrt(np.mean((Pv[:, cols] @ c - Yv) ** 2))))
            self.error_path = tuple(path)
            n_terms = int(np.argmin(path)) + 1
            if path and n_terms == len(path):
                import warnings as _w

                _w.warn(
                    f"the held-out error was still falling at the end of the "
                    f"path ({n_terms} of max_terms={max_terms}); the size was "
                    "truncated, not selected. Raise max_terms to see where it "
                    "turns.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.error_path = ()
            n_terms = int(n_terms)

        Phi = self._design(Xs)
        active, coef, info = simultaneous_omp(
            Phi, Ys, n_terms, force=force, tol=tol, ridge=ridge
        )
        self.active_ = active
        self.multi_indices = mi[active]
        self.coefficients = coef
        self.residual_path = info["residual_path"]
        self.cond = float(info["cond"])

    @property
    def n_params(self) -> int:
        """Input dimensions, the name the backends read."""
        return int(self.multi_indices.shape[1])

    @property
    def n_outputs(self) -> int:
        """Output dimensions, the name the backends read."""
        return int(np.asarray(self.coefficients).shape[1])

    def generate_forward_symb_emu(self, variable_names: Any = None) -> list:
        """Sympy expressions for the fitted model, one per output.

        Written in the family the fit used, through the same builder PolyEmu
        exports with, so the two agree term for term. The expression carries
        only the SELECTED terms, which is the point of exporting a sparse fit:
        the closure they were chosen from does not appear in it.
        """
        from MomentEmu.emulator import symbolic_polynomial_expressions

        # SparseEmu maps its box onto [-1, 1] as 2 (x - lo) / span - 1, which
        # is (x - mean) / std with these two; the builder takes the variance.
        scale = 0.5 * np.asarray(self.span_, dtype=np.float64)
        return symbolic_polynomial_expressions(
            np.asarray(self.coefficients, dtype=np.float64),
            np.asarray(self.multi_indices),
            variable_names=variable_names,
            input_means=np.asarray(self.lo_, dtype=np.float64) + scale,
            input_vars=scale**2,
            output_means=np.asarray(self.mean_Y_, dtype=np.float64),
            output_vars=np.asarray(self.scale_Y_, dtype=np.float64) ** 2,
            family=self.basis_kind,
        )

    @property
    def _plan_cls(self):
        """Basis plan class; a pickle from before basis_kind was added has none.

        The fallback is "legendre" and not "monomial", because that is the
        basis this selector always used.
        """
        return BASIS_PLANS[getattr(self, "basis_kind", "legendre")]

    def _map(self, X: np.ndarray) -> np.ndarray:
        """Map the training box onto [-1, 1], where the basis is orthonormal."""
        return 2.0 * (np.asarray(X, dtype=np.float64) - self.lo_) / self.span_ - 1.0

    def _design(self, V: np.ndarray, indices: Any = None) -> np.ndarray:
        """Product design over ``indices`` at points V, in the chosen basis.

        This used to carry its own orthonormal Legendre construction. The
        shared plan reproduces it to 1.5e-15, so the copy was removable, and
        every design in this class -- the greedy selection, the held-out path
        and the prediction -- now comes from one plan. Selecting in one basis
        and predicting in another is a silent failure, so they must not be
        able to drift apart.
        """
        mi = self.candidate_indices if indices is None else indices
        return self._plan_cls.build(mi).evaluate(V)

    def forward_emulator(self, X: Any) -> np.ndarray:
        """Predict at ``X`` in the original parameter coordinates."""
        arr = np.atleast_2d(as_float64(np.asarray(X), "X"))
        Ys = self._design(self._map(arr), self.multi_indices) @ self.coefficients
        return Ys * self.scale_Y_ + self.mean_Y_

    def report(self) -> dict:
        """Selected size, candidate size and the per-parameter reach kept."""
        mi = self.multi_indices
        return {
            "n_terms": int(mi.shape[0]),
            "basis_kind": self.basis_kind,
            "n_candidates": self.n_candidates,
            "compression": self.n_candidates / max(int(mi.shape[0]), 1),
            "max_total_degree": int(mi.sum(axis=1).max()) if mi.size else 0,
            "max_degree_per_parameter": tuple(int(v) for v in mi.max(axis=0))
            if mi.size else (),
            "max_interaction": int(np.count_nonzero(mi, axis=1).max())
            if mi.size else 0,
            "parameter_names": list(self.parameter_names),
            "error_path": self.error_path,
        }
