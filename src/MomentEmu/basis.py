"""Anisotropic / structured index sets (P5.3).

A Basis composes constraints by intersection: a total-degree bound, a
maximum interaction order, a weighted q-norm, per-parameter degree caps and
per-group degree limits.  Basis.total_degree().build(names, d) is row-
identical to generate_multi_indices.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from MomentEmu.emulator import generate_multi_indices


class Basis:
    """Composable index-set specification (P5.3).

    All constraints are applied together.  weights scales each parameter
    inside the q-norm; groups is a tuple of (indices, degree limit);
    per_parameter caps each parameter power (in addition to the D16 cap).
    The class method total_degree() builds the default isotropic basis.
    """

    def __init__(
        self,
        degree: int | None = None,
        max_interaction: int | None = None,
        q: float | None = None,
        weights: Any | None = None,
        groups: Any = (),
        per_parameter: Any | None = None,
    ) -> None:
        if degree is not None and int(degree) < 0:
            raise ValueError("degree must be >= 0")
        if max_interaction is not None and int(max_interaction) < 1:
            raise ValueError("max_interaction must be >= 1")
        if q is not None and float(q) <= 0.0:
            raise ValueError("q must be > 0")
        self.degree = None if degree is None else int(degree)
        self.max_interaction = None if max_interaction is None else int(max_interaction)
        self.q = None if q is None else float(q)
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        self.groups = tuple((tuple(int(i) for i in idx), int(lim)) for idx, lim in groups)
        self.per_parameter = (
            None if per_parameter is None else tuple(int(c) for c in per_parameter)
        )

    @classmethod
    def total_degree(cls, degree: int | None = None) -> Basis:
        """The default isotropic total-degree basis."""
        return cls(degree=degree)

    @classmethod
    def q_norm(
        cls,
        q: float,
        weights: Any | None = None,
        degree: int | None = None,
        **kwargs: Any,
    ) -> Basis:
        """A q-norm-weighted basis (weights default to ones)."""
        return cls(degree=degree, q=q, weights=weights, **kwargs)

    def build(self, names: Any, degree: int | None = None) -> np.ndarray:
        """Return the multi-indices satisfying every constraint (rows sorted)."""
        n = len(names)
        d = self.degree if degree is None else int(degree)
        if d is None:
            raise ValueError("a degree is required (pass it to build or set a degree)")
        if (
            self.max_interaction is None
            and self.q is None
            and self.per_parameter is None
            and not self.groups
        ):
            return generate_multi_indices(n, d)
        from itertools import product

        bounds = (
            [d] * n
            if self.per_parameter is None
            else [min(d, int(c)) for c in self.per_parameter]
        )
        w = None
        if self.q is not None:
            w = np.ones(n) if self.weights is None else np.asarray(self.weights, float)
            if w.shape != (n,):
                raise ValueError(f"weights has shape {w.shape}, expected ({n},)")
            for i in range(n):
                bounds[i] = min(bounds[i], int(np.floor(d / w[i])))
        rows = []
        for alpha in product(*[range(b + 1) for b in bounds]):
            arr = np.asarray(alpha, dtype=np.int64)
            total = int(arr.sum())
            if total > d:
                continue
            if (
                self.max_interaction is not None
                and int(np.count_nonzero(arr)) > self.max_interaction
            ):
                continue
            if self.q is not None and w is not None:
                if float(np.sum((arr * w) ** self.q) ** (1.0 / self.q)) > d + 1e-12:
                    continue
            if self.per_parameter is not None and np.any(arr > np.asarray(self.per_parameter)):
                continue
            if any(int(arr[list(idx)].sum()) > limit for idx, limit in self.groups):
                continue
            rows.append(tuple(int(v) for v in arr))
        if not rows:
            return np.zeros((0, n), dtype=np.int64)
        rows.sort(key=lambda a: (sum(a), a))
        return np.array(rows, dtype=np.int64)

    def spec(self) -> str:
        """A copy-pasteable constructor spec."""
        parts = [f"degree={self.degree}"]
        if self.max_interaction is not None:
            parts.append(f"max_interaction={self.max_interaction}")
        if self.q is not None:
            parts.append(f"q={self.q}")
        if self.weights is not None:
            parts.append(f"weights={list(self.weights)}")
        if self.per_parameter is not None:
            parts.append(f"per_parameter={list(self.per_parameter)}")
        if self.groups:
            parts.append(f"groups={list(self.groups)}")
        return "Basis(" + ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Basis):
            return NotImplemented
        return (
            self.degree == other.degree
            and self.max_interaction == other.max_interaction
            and self.q == other.q
            and self.weights == other.weights
            and self.groups == other.groups
            and self.per_parameter == other.per_parameter
        )

    def __repr__(self) -> str:
        return self.spec()
