"""Composing the two input preconditioners (T-002, P2 with P4).

Rotation (:mod:`MomentEmu.rotation`) and warping (:mod:`MomentEmu.warp`) both
change the coordinates the polynomial is built in, and they do not commute.
Which order wins is decided by where the structure lives, and neither order
dominates:

* A sharp feature along a ROTATED direction is invisible to a per-axis warp,
  because no single raw axis carries it. Rotation has to come first. Measured
  on such a target at seven parameters, warping alone changed nothing at all
  (every axis came back identity), rotation alone reached 17.17 percent, and
  rotating then warping reached 3.71 percent with the same 91 coefficients.

* A ridge in WARPED coordinates is not a ridge in the raw ones. The active
  subspace is a linear projection, so rank reduction applied first discards
  real signal: on a target of the form tanh(sum a_i log x_i), rotation alone
  scored 36.89 percent against 33.16 percent for no preconditioning at all --
  worse than doing nothing -- while warping first and then rotating reached
  7.06 percent at 19 times fewer coefficients. Warping concentrated the
  gradient-covariance spectrum from 0.832 to 0.961 in its leading direction,
  which is what made the projection safe.

So the order is a property of the problem, not a default to be argued about.
:class:`PreconditionedEmu` fits both and keeps the one with the lower held-out
error.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.guards import as_float64, check_finite
from MomentEmu.monomials import MonomialPlan
from MomentEmu.rotation import active_subspace, select_rank
from MomentEmu.warp import fit_warps

ORDERS = ("none", "warp", "rotate", "rotate_warp", "warp_rotate")
ESTIMATORS = ("polynomial", "sparse", "factored")


class _Rotate:
    """Standardise and project onto the leading active-subspace directions."""

    def __init__(self, X, Y, rank, variance_target, pilot_degree):
        # Only the rotation is needed here, so the gradient covariance is read
        # directly rather than by fitting a throwaway inner emulator.
        self.eigenvalues, V_full = active_subspace(X, Y, pilot_degree=pilot_degree)
        self.rank = select_rank(self.eigenvalues, rank, variance_target)
        self.V = V_full[:, : self.rank]
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)

    def __call__(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.V

    def spec(self):
        share = self.eigenvalues / self.eigenvalues.sum()
        return f"rotate(rank={self.rank}, leading_share={share[0]:.3f})"


class _Warp:
    """Per-axis monotone maps."""

    def __init__(self, X, Y, **kwargs):
        self.warps = fit_warps(X, Y, **kwargs)

    def __call__(self, X):
        A = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.column_stack([w(A[:, i]) for i, w in enumerate(self.warps)])

    def spec(self):
        return "warp(" + ", ".join(w.spec() for w in self.warps) + ")"


class PreconditionedEmu:
    """Forward emulator on rotated and/or warped inputs, order chosen by data.

    Args:
        order: one of :data:`ORDERS`, or "auto" to score every candidate and
            keep one.
        select: how "auto" chooses. "accuracy" takes the lowest held-out error.
            "parsimony" takes the fewest dimensions among the orders within
            ``parsimony_tol`` of the best error, which is usually what matters:
            on a log-ridge target, warping alone scored 0.0352 against 0.0534
            for warping then rotating, but the latter used 105 coefficients
            against 1,716 -- sixteen times fewer for one and a half times the
            error.
        parsimony_tol: the error multiple "parsimony" will pay for a smaller
            model.
        rank, variance_target, pilot_degree: passed to the rotation.
        warp_kwargs: passed to :func:`MomentEmu.warp.fit_warps`.
        estimator: what fits the preconditioned inputs. "polynomial" is a
            PolyEmu; "sparse" is a SparseEmu, which composes with any
            preconditioning and is the natural partner for a rotation, since
            the reduced dimension makes a large candidate set affordable;
            "factored" is a FactoredEmu, which needs a block partition of the
            inputs and therefore cannot follow a rotation, because a rotation
            replaces the parameters by linear combinations and the blocks stop
            meaning anything. That case raises rather than fitting something
            whose blocks refer to coordinates that no longer exist.

            Remaining keywords go to the estimator, so they are the
            estimator's, not a uniform set: ``extrapolation=`` on a prediction
            reaches PolyEmu and is rejected by the other two.
        scan_degree: ceiling on the degree used when comparing orders under
            the polynomial estimator. Each candidate is scored at the highest
            degree IT can afford, not at one degree shared by all: an order
            that cuts seven dimensions to two can afford a far higher degree,
            and scoring every order at a degree the full-dimensional ones can
            reach would hide exactly the benefit rotation exists for.
        scan_terms: active-set size used when scoring orders under the sparse
            estimator, capped at the caller's own ``n_terms``.

            Orders are scored with the estimator that will actually be used.
            Scoring a sparse or factored model with a dense polynomial proxy
            ranks the coordinates rather than the model, and the two need not
            agree: a rotation that helps a dense fit by cutting the dimension
            helps a sparse one less, because sparse selection was already
            paying only for the terms it kept.

    Attributes:
        order: the order actually used.
        steps: the transforms, applied left to right.
        scores: held-out error per candidate order, when ``order="auto"``.
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        order: str = "auto",
        rank: Any = "auto",
        variance_target: float = 0.999,
        pilot_degree: int = 3,
        warp_kwargs: dict | None = None,
        scan_degree: int = 12,
        estimator: str = "polynomial",
        scan_terms: int = 60,
        select: str = "accuracy",
        parsimony_tol: float = 2.0,
        validation_split: float = 0.2,
        random_state: Any = None,
        candidates: Any = None,
        **kwargs: Any,
    ) -> None:
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        check_finite(X, "X")
        check_finite(Y, "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if estimator not in ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {ESTIMATORS}, got {estimator!r}"
            )
        self.estimator = str(estimator)
        if candidates is None:
            # A rotation replaces the parameters by linear combinations, so a
            # factored model's blocks would no longer refer to anything. Those
            # orders are not candidates for it rather than errors to hit.
            candidates = (
                ("none", "warp") if self.estimator == "factored" else ORDERS
            )
        wk = dict(warp_kwargs or {})
        wk.setdefault("random_state", random_state)

        def build(name: str) -> list:
            if name == "none":
                return []
            if name == "warp":
                return [_Warp(X, Y, **wk)]
            if name == "rotate":
                return [_Rotate(X, Y, rank, variance_target, pilot_degree)]
            if name == "rotate_warp":
                rot = _Rotate(X, Y, rank, variance_target, pilot_degree)
                return [rot, _Warp(rot(X), Y, **wk)]
            if name == "warp_rotate":
                wrp = _Warp(X, Y, **wk)
                return [wrp, _Rotate(wrp(X), Y, rank, variance_target, pilot_degree)]
            raise ValueError(f"order must be one of {ORDERS} or 'auto', got {name!r}")

        self.scores: dict[str, float] = {}
        self.scan_degrees: dict[str, int] = {}
        self.dimensions: dict[str, int] = {}
        if order == "auto":
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(X.shape[0])
            cut = max(1, int(round(validation_split * X.shape[0])))
            hold, keep = perm[:cut], perm[cut:]
            built: dict[str, list] = {}
            dims: dict[str, int] = {}
            for name in candidates:
                steps = build(name)
                A = _chain(steps, X)
                err, deg = _score(A, Y, keep, hold, scan_degree,
                                  self.estimator, kwargs, scan_terms)
                built[name], dims[name] = steps, int(A.shape[1])
                self.scores[name] = err
                self.scan_degrees[name] = deg
            if select == "accuracy":
                best = min(self.scores, key=lambda k: self.scores[k])
            elif select == "parsimony":
                floor = min(self.scores.values())
                near = [k for k, v in self.scores.items()
                        if v <= floor * float(parsimony_tol)]
                best = min(near, key=lambda k: (dims[k], self.scores[k]))
            else:
                raise ValueError(
                    f"select must be 'accuracy' or 'parsimony', got {select!r}"
                )
            self.order, self.steps = str(best), built[best]
            self.dimensions = dict(dims)
        else:
            self.order, self.steps = str(order), build(str(order))

        # The chosen order fixes the dimensionality, so a degree the caller
        # pinned for a reduced fit can be unreachable for a full one. Clamp
        # rather than raise, and say so.
        self._n_in = int(X.shape[1])
        A = self.transform(X)
        asked = kwargs.get("max_degree_forward") if estimator == "polynomial" else None
        if asked is not None:
            affordable = _affordable_degree(A.shape[1], X.shape[0], int(asked))
            if affordable < int(asked):
                warnings.warn(
                    f"order {self.order!r} leaves {A.shape[1]} dimension(s), "
                    f"where degree {asked} needs more samples than the "
                    f"{X.shape[0]} available; using degree {affordable}",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs = dict(kwargs, max_degree_forward=affordable)
                if kwargs.get("init_deg_forward") is not None:
                    kwargs["init_deg_forward"] = min(
                        int(kwargs["init_deg_forward"]), affordable
                    )
        # One of three estimator types, so the attribute is deliberately
        # untyped rather than pinned to PolyEmu.
        self.emulator: Any
        if self.estimator == "polynomial":
            self.emulator = PolyEmu(A, Y, **kwargs)
        elif self.estimator == "sparse":
            from MomentEmu.sparse import SparseEmu

            kwargs.pop("max_degree_forward", None)
            kwargs.pop("init_deg_forward", None)
            kwargs.pop("RMSE_tol", None)
            kwargs.pop("verbose", None)
            self.emulator = SparseEmu(A, Y, **kwargs)
        else:
            from MomentEmu.factored import FactoredEmu

            if any(isinstance(step, _Rotate) for step in self.steps):
                raise ValueError(
                    f"estimator='factored' cannot follow a rotation, and the "
                    f"order chosen was {self.order!r}. A rotation replaces the "
                    "parameters by linear combinations, so a block partition "
                    "of the originals no longer refers to anything. Pass "
                    "order='warp' or order='none', or use estimator='sparse'."
                )
            for dead in ("max_degree_forward", "init_deg_forward", "RMSE_tol",
                         "verbose"):
                kwargs.pop(dead, None)
            self.emulator = FactoredEmu(A, Y, **kwargs)

    def transform(self, X: Any) -> np.ndarray:
        """Apply every step in order."""
        return _chain(self.steps, np.atleast_2d(as_float64(np.asarray(X), "X")))

    def forward_emulator(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict at ``X`` in the original parameter coordinates."""
        return self.emulator.forward_emulator(self.transform(X), **kwargs)

    def report(self) -> dict:
        """The chosen order, its steps, and the per-candidate scan results."""
        return {
            "order": self.order,
            "steps": tuple(s.spec() for s in self.steps),
            "scores": dict(self.scores),
            "scan_degrees": dict(self.scan_degrees),
            "dimensions": dict(getattr(self, "dimensions", {})),
            "estimator": self.estimator,
            "n_dims": int(self.transform(np.zeros((1, self._n_in))).shape[1]),
            "n_terms": _term_count(self.emulator),
        }


def _chain(steps, X):
    out = np.atleast_2d(np.asarray(X, dtype=np.float64))
    for step in steps:
        out = step(out)
    return out


def _affordable_degree(n_dims, n_rows, ceiling, oversample=3, cap_terms=6000):
    """Highest degree whose basis stays within the sample budget."""
    degree = 1
    while degree < int(ceiling):
        size = generate_multi_indices(n_dims, degree + 1).shape[0]
        if size * oversample > n_rows or size > cap_terms:
            break
        degree += 1
    return degree


def _score(A, Y, keep, hold, ceiling, estimator="polynomial",
           estimator_kwargs=None, scan_terms=60):
    """Held-out error of the ESTIMATOR THAT WILL BE USED on these coordinates.

    For the polynomial estimator each candidate order gets its own degree:
    scoring them all at one degree would compare a two-dimensional fit and a
    seven-dimensional one at a degree the latter can reach, which is precisely
    the comparison rotation is meant to escape. The sparse and factored
    estimators are scored as themselves, because a dense proxy ranks the
    coordinates rather than the model.
    """
    kw = dict(estimator_kwargs or {})
    if estimator == "sparse":
        from MomentEmu.sparse import SparseEmu

        kw.pop("n_terms", None)
        n_terms = min(int(scan_terms), max(1, keep.size // 4))
        try:
            pred = SparseEmu(
                A[keep], Y[keep], n_terms=n_terms, **kw
            ).forward_emulator(A[hold])
        except Exception:                                  # noqa: BLE001
            return float("inf"), 0
        return float(np.sqrt(np.mean((pred - Y[hold]) ** 2))), n_terms
    if estimator == "factored":
        from MomentEmu.factored import FactoredEmu

        kw = {k: v for k, v in kw.items() if k not in ("n_sweeps", "n_restarts")}
        try:
            pred = FactoredEmu(
                A[keep], Y[keep], n_sweeps=40, n_restarts=2, **kw
            ).forward_emulator(A[hold])
        except Exception:                                  # noqa: BLE001
            return float("inf"), 0
        return float(np.sqrt(np.mean((pred - Y[hold]) ** 2))), int(kw.get("rank", 1))
    n = A.shape[1]
    degree = _affordable_degree(n, keep.size, ceiling)
    plan = MonomialPlan.build(generate_multi_indices(n, degree))
    lo, hi = A[keep].min(axis=0), A[keep].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    P = plan.evaluate(2.0 * (A[keep] - lo) / span - 1.0)
    c = np.linalg.lstsq(P, Y[keep], rcond=None)[0]
    V = plan.evaluate(2.0 * (A[hold] - lo) / span - 1.0)
    return float(np.sqrt(np.mean((V @ c - Y[hold]) ** 2))), degree


def _term_count(model) -> int:
    """Retained basis terms, whichever estimator produced them."""
    mi = getattr(model, "forward_multi_indices", None)
    if mi is not None:
        return int(mi.shape[0])
    mi = getattr(model, "multi_indices", None)
    if mi is not None:
        return int(mi.shape[0])
    return int(getattr(model, "n_parameters", 0))
