"""Per-parameter monotone input warping (T-002 / P4).

P1 to P3 decide which basis functions to keep. This decides the coordinate the
basis is built in, which moves the per-degree convergence rate itself rather
than the term count at a fixed rate.

Polynomial approximation converges at ``rho ** -d``, where ``rho`` is set by the
distance from the interval to the nearest singularity of the target in the
COMPLEX plane, not by how sharply it bends on the real line. For
``tanh(20 (x - 0.3))`` the poles sit at ``0.3 + i pi / 40``, giving a predicted
rate of 0.921 against a measured 0.919. A monotone warp is a conformal map: it
moves those singularities, and with them the rate.

Measured on that target, a deliberately MISMATCHED smooth warp (a sharpness of
8 against the true 20) took the rate from 0.919 to 0.614, and the degree needed
for 1e-3 from 83 to 15. At seven parameters that is 7.5e9 isotropic terms
against 1.7e5. Because the term count grows as ``d ** p``, halving the degree
divides the basis by ``2 ** p``, so this multiplies with every reduction P1 to
P3 make rather than overlapping with them.

Two findings shaped the design.

Free-form arc-length equalisation, the obvious default, does not work. Equalising
purely by ``|f'|`` is degenerate: it yields ``u`` proportional to ``f`` itself,
so ``f`` composed with the inverse warp is linear by construction and the
one-dimensional problem answers itself. Blending in the x direction to avoid
that gave a rate of 0.905 against the raw 0.919 -- almost nothing -- and making
the same map smooth changed it not at all, so smoothness was not the obstacle.

A single-parameter family selected by a scan is both stronger and steadier. The
44,000-fold gain above came from a family that was wrong about the sharpness by
a factor of 2.5, which says the choice of family matters far more than the
precision of its parameter.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.guards import as_float64, check_finite
from MomentEmu.monomials import MonomialPlan

#: Shape families. Each is monotone on [-1, 1], smooth, onto [-1, 1], and
#: reduces to the identity in the limit of its neutral parameter.
FAMILIES = ("identity", "sinh", "kte")


@dataclass(frozen=True)
class Warp:
    """One axis's monotone map from the raw value onto [-1, 1].

    The pipeline is: an optional ``log`` pre-transform, an affine map of the
    training range onto [-1, 1], then a shape family.

    Attributes:
        pre: "identity" or "log"; "log" needs strictly positive values and is
            the case a user would otherwise have to know to apply by hand.
        shape: one of :data:`FAMILIES`.
        param: shape parameter; sharpness for "sinh", the map parameter in
            (0, 1) for "kte", unused for "identity".
        centre: shape centre in box coordinates, for an off-centre feature.
        lo, hi: training range AFTER the pre-transform.
    """

    pre: str
    shape: str
    param: float
    centre: float
    lo: float
    hi: float

    def __call__(self, x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        if self.pre == "log":
            v = np.log(np.maximum(v, np.finfo(float).tiny))
        span = self.hi - self.lo if self.hi > self.lo else 1.0
        v = 2.0 * (v - self.lo) / span - 1.0
        return self._shape(v)

    def _shape(self, v: np.ndarray) -> np.ndarray:
        if self.shape == "identity":
            return v
        if self.shape == "sinh":
            b, c = self.param, self.centre
            lo, hi = np.arcsinh(b * (-1.0 - c)), np.arcsinh(b * (1.0 - c))
            return 2.0 * (np.arcsinh(b * (v - c)) - lo) / (hi - lo) - 1.0
        if self.shape == "kte":
            a = self.param
            # arcsin needs |a v| < 1; outside the training box v can exceed 1,
            # so the argument is clipped and the map saturates there rather
            # than returning NaN. The caller's extrapolation guard still fires.
            return np.arcsin(np.clip(a * v, -1.0, 1.0)) / np.arcsin(a)
        raise ValueError(f"unknown shape {self.shape!r}")

    def spec(self) -> str:
        """A one-line description, for reporting and symbolic export."""
        if self.shape == "identity" and self.pre == "identity":
            return "identity"
        parts = [] if self.pre == "identity" else ["log"]
        if self.shape != "identity":
            parts.append(f"{self.shape}(param={self.param:g}, centre={self.centre:g})")
        return " -> ".join(parts) if parts else "identity"


def _candidates(column: np.ndarray, allow_log: bool) -> list[tuple[str, str, float, float]]:
    """(pre, shape, param, centre) tuples to try for one axis."""
    out: list[tuple[str, str, float, float]] = [("identity", "identity", 0.0, 0.0)]
    if allow_log and np.all(column > 0.0) and column.max() > column.min():
        # Offered whenever it is defined, with no dynamic-range threshold. An
        # earlier version required a range ratio above 10, which turned out to
        # be strictly harmful: the gain from a log warp rises smoothly from 23x
        # at a ratio of 2 to 691x at 10 with no break anywhere, so the
        # threshold blocked it exactly where it paid. Nothing is needed on the
        # other side, because `margin` already rejects a log that does not
        # help: on a response polynomial in x rather than in log x, the scan
        # returned identity at every ratio up to 1000, where the log warp would
        # have made the held-out error 0.0220 against 0.0000.
        out.append(("log", "identity", 0.0, 0.0))
    for b in (1.0, 3.0, 10.0, 30.0):
        for c in (-0.5, 0.0, 0.5):
            out.append(("identity", "sinh", b, c))
    for a in (0.5, 0.9, 0.99):
        out.append(("identity", "kte", a, 0.0))
    return out


def _make(pre: str, shape: str, param: float, centre: float, column: np.ndarray) -> Warp:
    v = np.log(column) if pre == "log" else column
    return Warp(pre, shape, param, centre, float(v.min()), float(v.max()))


def fit_warps(
    X: Any,
    Y: Any,
    scan_degree: int = 8,
    allow_log: bool = True,
    validation_split: float = 0.2,
    random_state: Any = None,
    criterion: str = "auto",
    joint_degree: int = 4,
    margin: float = 0.95,
    n_passes: int = 3,
    max_joint_terms: int = 4000,
) -> tuple[Warp, ...]:
    """Choose one warp per axis by a held-out scan over a candidate family.

    Args:
        criterion: "joint" scores a candidate by a low-degree fit over ALL
            axes with only this one warped; "marginal" regresses on the axis
            alone, which is cheaper but reads only that axis's marginal effect.
            "auto" takes joint while its basis stays under ``max_joint_terms``.
        joint_degree: degree of the joint scoring fit.
        margin: a candidate must cut the held-out error to this fraction of
            what the identity achieves before it is accepted. Without it an
            axis carrying no marginal signal -- one that acts only through an
            interaction -- is handed a warp fitted to noise.
        n_passes: sweeps of coordinate descent. One pass is not enough under
            the joint criterion, because an axis is judged against whatever
            the other axes currently are: on a target of the form
            f(log a, log b) the first axis fails the margin while the second
            is still unwarped, and is only picked up once the second has been.

    The marginal criterion has a known blind spot, which is why it is not the
    default: on a target where a parameter enters only through an interaction,
    its marginal effect was 0.04 of its true one and the scan missed the log
    transform it needed.
    """
    X = as_float64(np.asarray(X), "X")
    Y = as_float64(np.asarray(Y), "Y")
    check_finite(X, "X")
    check_finite(Y, "Y")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    cut = max(1, int(round(validation_split * X.shape[0])))
    hold, keep = order[:cut], order[cut:]

    n = X.shape[1]
    if criterion == "auto":
        size = generate_multi_indices(n, int(joint_degree)).shape[0]
        criterion = "joint" if (size <= max_joint_terms
                                and size * 3 <= keep.size) else "marginal"
    if criterion not in ("joint", "marginal"):
        raise ValueError(
            f"criterion must be 'auto', 'joint' or 'marginal', got {criterion!r}"
        )

    identity = [_make("identity", "identity", 0.0, 0.0, X[:, i]) for i in range(n)]
    chosen = list(identity)

    if criterion == "joint":
        plan = MonomialPlan.build(generate_multi_indices(n, int(joint_degree)))

        def score(trial: list[Warp]) -> float:
            A = np.column_stack([w(X[:, j]) for j, w in enumerate(trial)])
            P = plan.evaluate(A[keep])
            c = np.linalg.lstsq(P, Y[keep], rcond=None)[0]
            return float(np.sqrt(np.mean((plan.evaluate(A[hold]) @ c - Y[hold]) ** 2)))
    else:
        def score_axis(warp: Warp, i: int) -> float:
            u = warp(X[:, i])
            V = np.polynomial.legendre.legvander(u[keep], scan_degree)
            c = np.linalg.lstsq(V, Y[keep], rcond=None)[0]
            pred = np.polynomial.legendre.legvander(u[hold], scan_degree) @ c
            return float(np.sqrt(np.mean((pred - Y[hold]) ** 2)))

    for _pass in range(max(1, int(n_passes))):
        changed = False
        for i in range(n):
            col = X[:, i]
            if criterion == "joint":
                # The baseline is this axis UNWARPED with the others as they
                # stand. Scoring the current selection instead would credit a
                # good warp to the identity and then revert it, so a warp
                # accepted in one pass would be undone in the next.
                trial = list(chosen)
                trial[i] = identity[i]
                base = score(trial)
                best, best_err = identity[i], base
                for pre, shape, param, centre in _candidates(col, allow_log)[1:]:
                    trial[i] = _make(pre, shape, param, centre, col)
                    err = score(trial)
                    if err < best_err:
                        best, best_err = trial[i], err
            else:
                base = score_axis(identity[i], i)
                best, best_err = identity[i], base
                for pre, shape, param, centre in _candidates(col, allow_log)[1:]:
                    warp = _make(pre, shape, param, centre, col)
                    err = score_axis(warp, i)
                    if err < best_err:
                        best, best_err = warp, err
            # Only warp on evidence: without this an axis with no signal of
            # its own is handed whichever candidate best fitted the noise.
            pick = best if best_err < margin * base else identity[i]
            if pick != chosen[i]:
                chosen[i] = pick
                changed = True
        if not changed:
            break
    return tuple(chosen)


class WarpedEmu:
    """Forward emulator fitted in warped input coordinates.

    Every keyword this class does not consume goes to the inner
    :class:`~MomentEmu.emulator.PolyEmu`, so warping composes with ``basis=``
    and the rest.

    A warp does not commute with the tools that assume a uniform design. A
    design uniform in the raw parameters is NOT uniform in the warped ones, so
    ``sobol_report``, ``interaction_graph`` and ``degree_profile`` on the inner
    emulator describe the warped coordinates and their Sobol indices are
    box-uniform quantities there, not in the user's parameters.

    Attributes:
        warps: one :class:`Warp` per axis.
        emulator: the inner PolyEmu, fitted on the warped inputs.
        gain: held-out error without warping divided by the error with it,
            when a check was run; above one means the warp helped.
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        warps: Any = None,
        scan_degree: int = 8,
        allow_log: bool = True,
        criterion: str = "auto",
        check: bool = True,
        validation_split: float = 0.2,
        random_state: Any = None,
        **kwargs: Any,
    ) -> None:
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.warps = tuple(warps) if warps is not None else fit_warps(
            X, Y, scan_degree=scan_degree, allow_log=allow_log,
            validation_split=validation_split, random_state=random_state,
            criterion=criterion,
        )
        if len(self.warps) != X.shape[1]:
            raise ValueError(
                f"got {len(self.warps)} warps for {X.shape[1]} parameters"
            )
        self.gain: float | None = None
        if check:
            self.gain = self._check(X, Y, validation_split, random_state)
            if self.gain < 1.0:
                warnings.warn(
                    f"warping made the held-out error {1.0 / self.gain:.2f}x "
                    "WORSE on a degree-matched check. The per-axis scan reads "
                    "marginal effects only, so a target whose structure is in "
                    "its interactions can be misread; pass warps=None-free "
                    "coordinates or check=False to keep it anyway",
                    UserWarning,
                    stacklevel=2,
                )
        self.emulator = PolyEmu(self.transform(X), Y, **kwargs)

    def _check(self, X, Y, split, seed) -> float:
        """Held-out error of a degree-matched fit, warped against raw."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(X.shape[0])
        cut = max(1, int(round(split * X.shape[0])))
        hold, keep = order[:cut], order[cut:]
        n = X.shape[1]
        degree = 2
        while generate_multi_indices(n, degree + 1).shape[0] * 3 < keep.size and degree < 6:
            degree += 1
        plan = MonomialPlan.build(generate_multi_indices(n, degree))

        def err(A):
            lo, hi = A[keep].min(axis=0), A[keep].max(axis=0)
            span = np.where(hi > lo, hi - lo, 1.0)
            f = 2.0 * (A[keep] - lo) / span - 1.0
            v = 2.0 * (A[hold] - lo) / span - 1.0
            P = plan.evaluate(f)
            c = np.linalg.lstsq(P, Y[keep], rcond=None)[0]
            return float(np.sqrt(np.mean((plan.evaluate(v) @ c - Y[hold]) ** 2)))

        raw, warped = err(X), err(self.transform(X))
        return float(raw / warped) if warped > 0 else float("inf")

    def transform(self, X: Any) -> np.ndarray:
        """Apply every axis's warp to ``X``."""
        arr = np.atleast_2d(as_float64(np.asarray(X), "X"))
        return np.column_stack([w(arr[:, i]) for i, w in enumerate(self.warps)])

    def forward_emulator(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict at ``X`` in the original parameter coordinates."""
        return self.emulator.forward_emulator(self.transform(X), **kwargs)

    def report(self) -> dict:
        """One spec per axis, the held-out gain, and the fitted basis size."""
        return {
            "warps": tuple(w.spec() for w in self.warps),
            "gain": self.gain,
            "n_terms": int(self.emulator.forward_multi_indices.shape[0]),
            "degree": int(self.emulator.forward_degree),
        }
