"""Active-subspace preconditioning for the forward emulator (T-002 / P2).

A ridge target ``f(theta) = h(W theta)`` is low dimensional in the right
coordinates while carrying interactions at every ANOVA order in the original
ones. Interaction-order truncation therefore saturates on such a target, and
the isotropic basis pays ``C(d+n, n)`` for directions that carry no signal.

The gradient covariance ``C = E[J^T J]`` names the directions that matter. Its
leading eigenvectors span the active subspace; fitting a polynomial of the
same degree in ``r < n`` rotated coordinates costs ``C(d+r, r)`` instead.

The rotation is computed in standardised coordinates, so it does not depend on
the units of the parameters, and the gradient covariance uses standardised
outputs, so a large-scale output column does not dominate the directions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from MomentEmu.emulator import PolyEmu
from MomentEmu.guards import as_float64, check_finite


def _standardise(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (standardised, mean, scale) with a scale of 1 on constant columns."""
    mean = A.mean(axis=0)
    scale = A.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (A - mean) / scale, mean, scale


def _jacobian_block(J: Any, n_rows: int, n: int, m: int) -> np.ndarray:
    """Validate one (k, m, n) Jacobian block, promoting a 2-D single-output one."""
    J = as_float64(np.asarray(J), "jacobian")
    if J.ndim == 2:
        J = J[:, None, :]
    elif J.ndim != 3:
        raise ValueError(
            f"jacobian must be 2-D (N, n) or 3-D (N, m, n), got shape {J.shape}"
        )
    if J.shape[0] != n_rows:
        raise ValueError(
            f"jacobian has {J.shape[0]} rows, expected {n_rows} (one per sample)"
        )
    if J.shape[2] != n:
        raise ValueError(
            f"jacobian has {J.shape[2]} columns, expected {n} (one per parameter)"
        )
    if J.shape[1] != m:
        raise ValueError(
            f"jacobian has {J.shape[1]} outputs, expected {m} to match Y"
        )
    check_finite(J, "jacobian")
    return J


def _output_scaling(Y: np.ndarray, output_scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Ys, scale_Y)``: the outputs on a common footing, and by what.

    ``scale_Y`` is one divisor per output even under "global", where every
    entry is the same scalar, so the chain rule in
    :func:`_covariance_from_jacobian` reads the same in both cases.
    """
    if output_scaling == "per_output":
        Ys, _mean, scale_Y = _standardise(Y)
        return Ys, scale_Y
    Ys = Y - Y.mean(axis=0)
    span = float(np.sqrt(np.mean(Ys ** 2)))
    span = span if span > 0.0 else 1.0
    # One scalar for every output, so relative importance is preserved.
    return Ys / span, np.full(Y.shape[1], span)


def _symmetrised(C: Any, n: int, name: str) -> np.ndarray:
    """Validate an ``(n, n)`` covariance and return its symmetric part."""
    C = as_float64(np.asarray(C), name)
    check_finite(C, name)
    if C.shape != (n, n):
        raise ValueError(f"{name} must be ({n}, {n}), got shape {C.shape}")
    return 0.5 * (C + C.T)                   # eigh reads the lower triangle only


def _eigendecomposition(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues descending and clipped at zero, with matching eigenvectors."""
    evals, V = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return np.maximum(evals[order], 0.0), V[:, order]


def _covariance_from_jacobian(
    jacobian: Any,
    X: np.ndarray,
    scale_X: np.ndarray,
    scale_Y: np.ndarray,
    batch_size: int,
    jacobian_inputs: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate ``C = E[J^T J]`` from a supplied Jacobian, in the coordinates
    this module works in.

    A caller's model differentiates in physical units, while the rotation is
    computed on standardised X and Y. The chain rule between them is

        d(Ys)_m / d(Xs)_i = J_raw[m, i] * scale_X[i] / scale_Y[m],

    and it is applied here rather than asked of the caller. Dropping the
    ``scale_X`` factor returns a different leading direction whenever the
    parameters have different scales, and does so silently: the result is
    still a unit vector with a plausible spectrum.

    A callable is handed RAW X, one batch at a time, so a large design never
    needs its whole Jacobian in memory at once.

    ``jacobian_inputs`` is what the callable is evaluated at when that is not
    ``X`` itself. ``X`` then supplies only the row count, while ``scale_X``
    still comes from the coordinates the rotation is computed in. The case is
    a rotation that follows a warp: the caller differentiates at the raw
    points, and the result belongs to the warped axes.
    """
    N, n = X.shape
    m = scale_Y.shape[0]
    at = X if jacobian_inputs is None else jacobian_inputs
    if at.shape[0] != N:
        raise ValueError(
            f"jacobian_inputs has {at.shape[0]} rows, expected {N} (one per sample)"
        )
    is_callable = callable(jacobian)
    if not is_callable:
        jacobian = _jacobian_block(jacobian, N, n, m)
    C = np.zeros((n, n))
    step = max(int(batch_size), 1)
    for start in range(0, N, step):
        stop = min(start + step, N)
        chunk = at[start:stop]
        if is_callable:
            J = _jacobian_block(jacobian(chunk), chunk.shape[0], n, m)
        else:
            J = jacobian[start:stop]
        Js = J * scale_X[None, None, :] / scale_Y[None, :, None]
        C += np.einsum("kmi,kmj->ij", Js, Js)
    return C / float(N)


def gradient_covariance_matrix(
    X: Any,
    Y: Any,
    pilot_degree: int = 3,
    batch_size: int = 2048,
    output_scaling: str = "global",
    jacobian: Any = None,
    jacobian_inputs: Any = None,
    **pilot_kwargs: Any,
) -> np.ndarray:
    """``C = E[J^T J]`` in standardised coordinates, symmetrised.

    This is the matrix :func:`active_subspace` diagonalises. It is exposed on
    its own because one C serves many rotations of the same data, and
    rebuilding it is not cheap: :func:`scan_rank` fits an emulator per rank,
    and recomputing C for each one re-fits the pilot -- or, with ``jacobian``,
    calls back into the caller's model -- for a matrix that cannot change
    between ranks.

    ``jacobian_inputs`` is where a callable ``jacobian`` is evaluated, when
    that is not ``X``. It exists for a rotation that FOLLOWS a warp: the
    rotation is computed in the warped coordinates, so ``X`` and the scales
    taken from it are warped, while the caller differentiates at the raw
    points. The callable is then responsible for returning ``dY/dX`` in the
    coordinates of ``X``; :meth:`MomentEmu.warp.Warp.derivative` supplies the
    factor and :class:`MomentEmu.precondition._Rotate` composes the two.

    Args:
        jacobian: as in :func:`active_subspace`.
        jacobian_inputs: ``(N, n)`` points at which to call ``jacobian``,
            defaulting to ``X``.

    Returns:
        (n, n) symmetric C in standardised coordinates.
    """
    X = as_float64(np.asarray(X), "X")
    Y = as_float64(np.asarray(Y), "Y")
    check_finite(X, "X")
    check_finite(Y, "Y")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if output_scaling not in ("global", "per_output"):
        raise ValueError(
            f"output_scaling must be 'global' or 'per_output', got "
            f"{output_scaling!r}"
        )
    if jacobian_inputs is not None and jacobian is None:
        raise ValueError(
            "jacobian_inputs says where to evaluate a jacobian, so it has no "
            "meaning without jacobian="
        )
    n = X.shape[1]
    Xs, _mean, scale_X = _standardise(X)
    Ys, scale_Y = _output_scaling(Y, output_scaling)

    if jacobian is not None:
        at = (
            None if jacobian_inputs is None
            else as_float64(np.asarray(jacobian_inputs), "jacobian_inputs")
        )
        C = _covariance_from_jacobian(
            jacobian, X, scale_X, scale_Y, batch_size, at
        )
        return _symmetrised(C, n, "gradient covariance")

    pilot_kwargs.setdefault("forward", True)
    pilot_kwargs.setdefault("backward", False)
    pilot_kwargs.setdefault("RMSE_tol", 0.0)
    pilot_kwargs.setdefault("verbose", 0)
    pilot = PolyEmu(
        Xs, Ys,
        init_deg_forward=int(pilot_degree),
        max_degree_forward=int(pilot_degree),
        **pilot_kwargs,
    )

    C = np.zeros((n, n))
    # Batched: the Jacobian is (N, m, n) and m is the number of outputs, which
    # is a full spectrum in the case this was built for.
    for start in range(0, Xs.shape[0], batch_size):
        chunk = Xs[start:start + batch_size]
        J = np.atleast_3d(pilot.jacobian(chunk))
        C += np.einsum("kmi,kmj->ij", J, J)
    C /= Xs.shape[0]
    return _symmetrised(C, n, "gradient covariance")


def active_subspace(
    X: Any,
    Y: Any,
    pilot_degree: int = 3,
    batch_size: int = 2048,
    output_scaling: str = "global",
    jacobian: Any = None,
    gradient_covariance: Any = None,
    **pilot_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of the gradient covariance, largest first.

    Fits a cheap pilot emulator of degree ``pilot_degree``, reads its analytic
    Jacobian and accumulates ``C = E[J^T J]`` over the training design. Both X
    and Y are standardised first, so C measures relative sensitivity and the
    result is invariant to the units of either.

    ``output_scaling`` decides how the outputs are put on a common footing,
    and the default is not the obvious one. "per_output" divides each output
    column by its own standard deviation, which is right when the outputs are
    different physical quantities. It is wrong for the case this package is
    built for -- one quantity sampled at many points, a spectrum -- because it
    makes a near-silent channel as important as the loudest. On the 21cmGEM
    benchmark the per-bin standard deviations run from exactly zero to 88.8,
    and per-bin scaling moved the leading cumulative share from 0.53 to 0.74,
    blurring the rank gap it is read from. "global" centres each output and
    divides everything by one scalar, preserving relative importance.

    The pilot degree matters far less than the scaling, and only once the
    scaling is wrong does it look important. Under "per_output" on the 21cmGEM
    benchmark a degree-3 pilot scored 2.78 percent against degree 5's 2.41;
    under "global" the same pair is 2.25 against 2.23, which is noise. A
    higher-degree pilot is not free either: on a synthetic ridge whose true
    subspace is 2-dimensional, leakage into the five dead directions grew from
    0.0023 at degree 3 to 0.0077 at degree 7, blurring the gap rank selection
    reads. The default stays cheap; raise it only if the rotation looks
    unstable after the scaling is right.

    ``jacobian`` replaces the pilot with the caller's own derivatives, which is
    worth doing whenever the model is differentiable (JAX, PyTorch, an autodiff
    simulator): the pilot is an error source, not only a cost. It takes a
    callable ``f(X_chunk) -> (k, m, n)`` evaluated batch by batch on RAW X, or
    a precomputed ``(N, m, n)`` array, or ``(N, n)`` for a single output. The
    standardisation chain rule is applied here, so the caller supplies
    derivatives in its own physical units. ``gradient_covariance`` takes a
    ready ``(n, n)`` C instead and skips both the pilot and the accumulation.

    Returns:
        (eigenvalues, V): eigenvalues descending, V columns the matching
        eigenvectors in standardised-X coordinates.
    """
    X = as_float64(np.asarray(X), "X")
    Y = as_float64(np.asarray(Y), "Y")
    check_finite(X, "X")
    check_finite(Y, "Y")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if output_scaling not in ("global", "per_output"):
        raise ValueError(
            f"output_scaling must be 'global' or 'per_output', got "
            f"{output_scaling!r}"
        )
    if jacobian is not None and gradient_covariance is not None:
        raise ValueError(
            "pass jacobian or gradient_covariance, not both; they are two ways "
            "to supply the same C = E[J^T J]"
        )
    n = X.shape[1]
    if gradient_covariance is not None:
        C = _symmetrised(gradient_covariance, n, "gradient_covariance")
    else:
        C = gradient_covariance_matrix(
            X, Y, pilot_degree=pilot_degree, batch_size=batch_size,
            output_scaling=output_scaling, jacobian=jacobian, **pilot_kwargs,
        )
    return _eigendecomposition(C)


def select_rank(
    eigenvalues: np.ndarray,
    rank: Any = "auto",
    variance_target: float = 0.999,
) -> int:
    """Rank to retain, from the gradient-covariance spectrum.

    ``rank="auto"`` takes the first rank reaching ``variance_target`` and
    nothing more, which overshoots on a shallow tail: a direction costs a whole
    dimension of the C(d+r, r) growth however little variance it carries. Use
    :func:`scan_rank` when the budget matters.
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    n = evals.size
    if not 0.0 < float(variance_target) <= 1.0:
        raise ValueError(f"variance_target must be in (0, 1], got {variance_target}")
    if rank != "auto":
        rank = int(rank)
        if not 1 <= rank <= n:
            raise ValueError(
                f"rank must be in [1, {n}] for {n} parameters, got {rank}"
            )
        return rank
    total = float(evals.sum())
    shares = np.cumsum(evals) / total if total > 0.0 else np.ones(n)
    reached = shares >= float(variance_target) - 1e-12
    return int(np.argmax(reached)) + 1 if reached.any() else n


class ActiveSubspaceEmu:
    """Forward emulator fitted in the leading active-subspace coordinates.

    Wraps a :class:`PolyEmu` fitted on ``Z = standardise(X) @ V[:, :rank]``.
    Every keyword this class does not consume is forwarded to that inner fit,
    so a rotated model composes with ``basis=``, ``transform=`` and the rest.

    The rotation is a projection, not a reparameterisation: a direction left
    out is gone. Check :attr:`variance_share` before trusting a small rank.

    ``rank="auto"`` reaches ``variance_target`` and nothing more, which
    overshoots whenever the spectrum has a long shallow tail: on a
    travelling-trough target whose shares were 0.952, 0.030, 0.0099, 0.0033,
    0.0026, 0.0012, 0.0008, the 0.999 target returned rank 6 while rank 2 at a
    higher degree was both 38x smaller and more accurate. A variance share is
    the wrong currency here, because a direction costs a whole dimension of
    the ``C(d+r, r)`` growth however little variance it carries. Use
    :func:`scan_rank` to see the trade-off rather than trusting the target.

    Attributes:
        V: (n, rank) rotation in standardised-X coordinates.
        eigenvalues: (n,) gradient-covariance spectrum, descending.
        variance_share: cumulative share of the spectrum kept by ``rank``.
        rank: number of retained directions.
        emulator: the inner PolyEmu, fitted on the rotated inputs.
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        rank: Any = "auto",
        variance_target: float = 0.999,
        pilot_degree: int = 3,
        output_scaling: str = "global",
        pilot_kwargs: dict | None = None,
        jacobian: Any = None,
        gradient_covariance: Any = None,
        **kwargs: Any,
    ) -> None:
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n = X.shape[1]
        self.eigenvalues, V_full = active_subspace(
            X, Y, pilot_degree=pilot_degree, output_scaling=output_scaling,
            jacobian=jacobian, gradient_covariance=gradient_covariance,
            **(pilot_kwargs or {})
        )
        total = float(self.eigenvalues.sum())
        shares = (
            np.cumsum(self.eigenvalues) / total if total > 0.0
            else np.ones(n)
        )
        self.rank = select_rank(self.eigenvalues, rank, variance_target)
        self.variance_target = float(variance_target)
        self.variance_share = float(shares[self.rank - 1])
        self.V = V_full[:, : self.rank]

        _, self.mean_X_, self.scale_X_ = _standardise(X)
        self.n_params = n
        self.emulator = PolyEmu(self._rotate(X), Y, **kwargs)

    def _rotate(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_X_) / self.scale_X_) @ self.V

    def transform(self, X: Any) -> np.ndarray:
        """Standardise and project ``X`` onto the retained directions.

        Mirrors :meth:`MomentEmu.warp.WarpedEmu.transform`, so the two
        preconditioners compose in either order.
        """
        return self._rotate(np.atleast_2d(as_float64(np.asarray(X), "X")))

    def forward_emulator(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict at ``X`` in the original parameter coordinates."""
        arr = np.atleast_2d(as_float64(np.asarray(X), "X"))
        return self.emulator.forward_emulator(self._rotate(arr), **kwargs)

    def jacobian(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Analytic dY/dX in the original coordinates, shape (N, m, n)."""
        arr = np.atleast_2d(as_float64(np.asarray(X), "X"))
        J_z = np.atleast_3d(self.emulator.jacobian(self._rotate(arr), **kwargs))
        # Z = (X - mean) / scale @ V, so dZ_j/dX_i = V[i, j] / scale_i.
        return np.einsum("kmj,ij->kmi", J_z, self.V) / self.scale_X_[None, None, :]

    def report(self) -> dict:
        """Spectrum, retained rank and the loading of each parameter."""
        total = float(self.eigenvalues.sum())
        share = self.eigenvalues / total if total > 0.0 else self.eigenvalues
        return {
            "eigenvalues": self.eigenvalues,
            "eigenvalue_share": share,
            "cumulative_share": np.cumsum(share),
            "rank": self.rank,
            "variance_share": self.variance_share,
            "V": self.V,
            "n_terms": int(self.emulator.forward_multi_indices.shape[0]),
        }


def scan_rank(
    X: Any,
    Y: Any,
    ranks: Any,
    degree: Any,
    X_test: Any = None,
    Y_test: Any = None,
    pilot_degree: int = 3,
    output_scaling: str = "global",
    pilot_kwargs: dict | None = None,
    jacobian: Any = None,
    gradient_covariance: Any = None,
    **kwargs: Any,
) -> list[dict]:
    """Fit several (rank, degree) pairs and report accuracy against budget.

    Rank selection from the spectrum alone is unreliable, because a direction
    costs a whole dimension of the ``C(d+r, r)`` growth however little variance
    it carries. Scanning is what actually locates the knee: a rank one too low
    saturates at its own error floor no matter the degree, and a rank one too
    high is beaten by a smaller rank at a higher degree for the same budget.

    The rotation itself does not depend on the rank, so the gradient
    covariance is built once and every fit reads the same one. Rebuilding it
    per rank re-fits the pilot as many times as there are ranks, and with
    ``jacobian`` it is worse than redundant: each rebuild is another pass
    through the caller's model.

    Args:
        ranks: ranks to try.
        degree: one degree for every rank, or one per rank.
        X_test, Y_test: held-out set; without it the training set is scored and
            the numbers are optimistic.
        jacobian, gradient_covariance: as in :func:`active_subspace`. Either
            replaces the pilot for every rank in the scan.
        output_scaling, pilot_kwargs: how the shared covariance is built.

    Returns:
        One dict per fit with rank, degree, n_terms, variance_share and fom
        (rms error over the rms of the reference), sorted by n_terms.
    """
    if jacobian is not None and gradient_covariance is not None:
        raise ValueError(
            "pass jacobian or gradient_covariance, not both; they are two ways "
            "to supply the same C = E[J^T J]"
        )
    ranks = [int(r) for r in ranks]
    degrees = [int(degree)] * len(ranks) if np.ndim(degree) == 0 else [int(v) for v in degree]
    if len(degrees) != len(ranks):
        raise ValueError(
            f"degree must be a scalar or one value per rank; got {len(degrees)} "
            f"for {len(ranks)} ranks"
        )
    Xs = np.asarray(X) if X_test is None else np.asarray(X_test)
    Ys = np.asarray(Y) if Y_test is None else np.asarray(Y_test)
    if Ys.ndim == 1:
        Ys = Ys.reshape(-1, 1)
    den = float(np.sqrt(np.mean(Ys ** 2))) or 1.0

    C = (
        gradient_covariance_matrix(
            X, Y, pilot_degree=pilot_degree, output_scaling=output_scaling,
            jacobian=jacobian, **(pilot_kwargs or {}),
        )
        if gradient_covariance is None else gradient_covariance
    )

    rows = []
    for r, d in zip(ranks, degrees):
        try:
            emu = ActiveSubspaceEmu(
                X, Y, rank=r, gradient_covariance=C,
                init_deg_forward=d, max_degree_forward=d, **kwargs
            )
            pred = emu.forward_emulator(Xs, extrapolation="ignore")
            rows.append({
                "rank": r,
                "degree": d,
                "n_terms": int(emu.emulator.forward_multi_indices.shape[0]),
                "variance_share": emu.variance_share,
                "fom": float(np.sqrt(np.mean((pred - Ys) ** 2)) / den),
                "error": None,
            })
        except Exception as exc:                       # noqa: BLE001
            rows.append({"rank": r, "degree": d, "n_terms": None,
                         "variance_share": None, "fom": float("inf"),
                         "error": f"{type(exc).__name__}: {exc}"})
    return sorted(rows, key=lambda row: (row["n_terms"] is None, row["n_terms"] or 0))
