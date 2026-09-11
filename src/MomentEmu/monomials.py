"""Level-wise recursive monomial evaluation (P0.7).

The inference path must not rebuild the monomial basis term by term in a
Python loop, and it must handle an index set that is *not* downward closed
(dim_reduction produced such sets).  A :class:`MonomialPlan` takes the
downward closure of the requested indices, orders the closure by total
degree, records the parent row and variable of every non-constant row, and
evaluates all rows of one degree level with a single fancy-index product.
The requested rows are then selected.  NumPy only; no sklearn, sympy or JAX
so the plan can be reused by the JAX backend (P2.1) and the training build
(P5.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def downward_closure(multi_indices: np.ndarray) -> np.ndarray:
    """Return every multi-index <= a requested one (componentwise), sorted.

    The result includes the zero vector and is downward closed.  Rows are
    sorted by (total degree, index) so the first row is always the constant.
    """
    mi = np.asarray(multi_indices)
    if mi.ndim != 2:
        raise ValueError(f"multi_indices must be 2-D, got shape {mi.shape}")
    if mi.size == 0:
        return np.zeros((0, mi.shape[1]), dtype=np.int64)
    seen: set[tuple[int, ...]] = set()
    stack = [tuple(int(v) for v in row) for row in mi]
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        for i, e in enumerate(a):
            if e:
                b = list(a)
                b[i] -= 1
                stack.append(tuple(b))
    return np.array(sorted(seen, key=lambda a: (sum(a), a)), dtype=np.int64)


@dataclass
class MonomialPlan:
    """Precomputed parent/variable/level tables for one index set.

    Attributes
    ----------
    closure : (D_closure, n) int64
        Downward closure, ordered by total degree.
    parent : (D_closure,) int64
        Row index of the parent (alpha - e_i) in ``closure``; -1 for the
        constant row.
    var : (D_closure,) int64
        The variable i removed to form the parent; -1 for the constant row.
    levels : list of ndarray
        Row indices of each total degree 1, 2, ... in ``closure``.
    select : (D, ) int64
        Row of ``closure`` that each requested multi-index maps to.
    n_closure : int
    n_terms : int
    """

    closure: np.ndarray
    parent: np.ndarray
    var: np.ndarray
    levels: list
    select: np.ndarray

    @property
    def n_closure(self) -> int:
        return int(self.closure.shape[0])

    @property
    def n_terms(self) -> int:
        return int(self.select.shape[0])

    @property
    def max_degree(self) -> int:
        return int(self.closure.sum(axis=1).max()) if self.n_closure else 0

    @classmethod
    def build(cls, multi_indices: np.ndarray) -> "MonomialPlan":
        mi = np.asarray(multi_indices, dtype=np.int64)
        if mi.ndim != 2:
            raise ValueError(f"multi_indices must be 2-D, got shape {mi.shape}")
        closure = downward_closure(mi)
        index = {tuple(int(v) for v in row): j for j, row in enumerate(closure)}
        D = closure.shape[0]
        parent = np.full(D, -1, dtype=np.int64)
        var = np.full(D, -1, dtype=np.int64)
        for j, row in enumerate(closure):
            if not row.any():
                continue
            i = int(np.flatnonzero(row)[0])
            p = row.copy()
            p[i] -= 1
            parent[j] = index[tuple(int(v) for v in p)]
            var[j] = i
        degree = closure.sum(axis=1)
        dmax = int(degree.max()) if D else 0
        levels = [np.flatnonzero(degree == ell) for ell in range(1, dmax + 1)]
        select = np.array([index[tuple(int(v) for v in row)] for row in mi], dtype=np.int64)
        return cls(closure=closure, parent=parent, var=var, levels=levels, select=select)

    def evaluate(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return (N, D) monomials of the requested indices, in input order.

        ``X_scaled`` must already be standardised; the build is level-wise,
        with one fancy-index product per total degree.  Inputs are promoted to
        float64 (D1): a float32 or integer array never truncates the design.
        """
        X = np.asarray(X_scaled)
        if X.ndim != 2:
            X = X.reshape(-1, self.closure.shape[1])
        if X.shape[1] != self.closure.shape[1]:
            raise ValueError(
                f"input has {X.shape[1]} columns; the plan was built for "
                f"{self.closure.shape[1]}"
            )
        X = np.ascontiguousarray(X, dtype=np.float64)
        N = X.shape[0]
        buf = np.empty((self.n_closure, N), dtype=np.float64)
        if self.n_closure:
            buf[0] = 1.0
        xT = np.ascontiguousarray(X.T)
        for level in self.levels:
            buf[level] = buf[self.parent[level]] * xT[self.var[level]]
        # buf is (D_closure, N): return the (N, D) transpose the caller uses.
        return np.ascontiguousarray(buf[self.select].T)


def evaluate_monomials_fast(
    X_scaled: np.ndarray,
    multi_indices: np.ndarray,
    plan: MonomialPlan | None = None,
) -> np.ndarray:
    """Convenience wrapper: (N, D) monomials for ``multi_indices`` at ``X``."""
    if plan is None:
        plan = MonomialPlan.build(multi_indices)
    return plan.evaluate(X_scaled)


def fold_output_affine(
    coeffs: np.ndarray,
    multi_indices: np.ndarray,
    mean_Y: np.ndarray,
    scale_Y: np.ndarray,
) -> np.ndarray:
    """Fold Y = (Phi @ C) * scale_Y + mean_Y into C_fold (D, m) (P0.7).

    The constant monomial row absorbs ``mean_Y`` so inference is one GEMM and
    no separate inverse_transform.  Used only for prediction; the stored
    coefficients are untouched.
    """
    C = np.asarray(coeffs, dtype=np.float64) * np.asarray(scale_Y, dtype=np.float64)[None, :]
    mi = np.asarray(multi_indices)
    const = np.flatnonzero(~mi.any(axis=1))
    if const.size:
        C[const[0]] = C[const[0]] + np.asarray(mean_Y, dtype=np.float64)
    return C
