"""Input and numerics guards for MomentEmu.

Every function is pure: it takes arrays or scalars, returns a value or a new
array, and either warns (``warnings.warn``) or raises. Nothing here mutates
its arguments or touches a ``PolyEmu`` instance, so each guard can be dropped
into ``PolyEmu.__init__`` / ``generate_*_emulator`` / ``forward_emulator`` or
into the module-level solvers in ``MomentEmu.py`` without further plumbing.

Threshold constants are module-level so a test can import the same number the
guard uses. The cond(M) thresholds are calibrated in
``robust_cond_calibration.py`` (same directory); see the docstring of
``check_conditioning`` for the measured justification.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RmseList = Sequence[float] | np.ndarray

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
EPS64 = float(np.finfo(np.float64).eps)          # 2.22e-16
FLOAT64_DIGITS = -math.log10(EPS64)               # 15.65
FILL_FACTOR = 2.0        # require N_train >= FILL_FACTOR * D before warning
COND_WARN = 1e12         # coefficients keep ~4 digits; export unreliable
COND_RAISE = 1e16        # beyond 1/eps: M numerically singular
SWEEP_BLOWUP_FACTOR = 10.0
COLLINEAR_RTOL = 1e-12


# --------------------------------------------------------------------------
# Warning / error types
# --------------------------------------------------------------------------
class EmulatorWarning(UserWarning):
    """Base class for every warning emitted by the guards."""


class IllConditionedWarning(EmulatorWarning):
    """cond(M) exceeded COND_WARN (or COND_RAISE under a warn policy)."""


class ExtrapolationWarning(EmulatorWarning):
    """A prediction input lies outside the training box."""


class PrecisionWarning(EmulatorWarning):
    """An input was narrower than float64 and was promoted."""


class InsufficientSamplesError(ValueError):
    """N_train (or the number of distinct rows) does not exceed the basis size."""


class IllConditionedError(np.linalg.LinAlgError):
    """cond(M) exceeded COND_RAISE under a raise policy.

    Subclasses ``LinAlgError`` so existing ``except LinAlgError`` clauses
    keep working.
    """


# --------------------------------------------------------------------------
# Basis-size arithmetic (replacement for the off-by-one max_order)
# --------------------------------------------------------------------------
def basis_size(n_params: int, degree: int) -> int:
    """D = C(n + d, d): number of monomials of total degree <= d."""
    return math.comb(n_params + degree, degree)


def max_supported_degree(n_params: int, n_samples: int, *, fill: float = FILL_FACTOR) -> int:
    """Largest degree d with basis_size(n, d) * fill <= n_samples and D < N.

    The strict ``D < N`` matches check_sample_count, which raises at
    ``D >= N``: a degree whose basis size equals N is not identifiable,
    so the auto-cap must stop one rung earlier. ``fill=1`` therefore gives
    the largest identifiable degree and ``fill=2`` (the default) leaves a
    2x oversampling margin. Contrast with the old ``max_order``, which
    returned the SMALLEST k with D >= N (round-1 max-order-off-by-one).
    """
    if n_params < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")
    if fill <= 0:
        raise ValueError(f"fill must be > 0, got {fill}")
    k = 0
    while True:
        D_next = basis_size(n_params, k + 1)
        if D_next * fill <= n_samples and D_next < n_samples:
            k += 1
        else:
            break
    return k


def check_sample_count(
    n_samples: int,
    basis_dim: int,
    *,
    fill: float = FILL_FACTOR,
    degree: int | None = None,
    n_params: int | None = None,
) -> None:
    """Raise when N <= D; warn when N < fill * D.

    Where: ``PolyEmu.generate_*_emulator`` before building M at each degree,
    and inside ``solve_emulator_coefficients`` when N is known.
    """
    tag = f"degree {degree}: " if degree is not None else ""
    if n_samples <= basis_dim:
        hint = ""
        if n_params is not None:
            hint = (
                f" The largest degree supported by N_train = {n_samples} with fill "
                f"factor {fill:g} is {max_supported_degree(n_params, n_samples, fill=fill)}."
            )
        raise InsufficientSamplesError(
            f"{tag}basis size D = {basis_dim} >= N_train = {n_samples}; the moment "
            f"matrix is rank-deficient and the fit is not identifiable.{hint}"
        )
    if n_samples < fill * basis_dim:
        warnings.warn(
            f"{tag}N_train = {n_samples} < {fill:g} x D = {fill * basis_dim:.0f}; the fit "
            f"is barely determined (measured cond(M) 5.8e8 at N = 2D vs 5.3e12 at "
            f"N = D + 1 for n = 2, d = 10).",
            EmulatorWarning,
            stacklevel=2,
        )


def count_distinct_rows(X: np.ndarray) -> int:
    """Number of distinct rows of a 2-D array (exact bitwise equality).

    Uses a void-dtype row view: measured 3 ms at N = 20000 against 7-187 ms
    for ``np.unique(X, axis=0)``. Rows are compared bitwise, so -0.0 and 0.0
    count as distinct (irrelevant for the rank question this guard serves).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] == 0:
        return 0
    Xc = np.ascontiguousarray(X)
    rows = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    return int(np.unique(rows).size)


def check_distinct_rows(X: np.ndarray, basis_dim: int) -> int:
    """Raise when the number of DISTINCT training rows does not exceed D.

    This is the guard that separates the catastrophic round-1 cases (5 or 20
    distinct rows repeated to N = 2000, D = 66: 11986x / 1607x overshoot on
    fresh points) from a merely ill-conditioned high-degree fit, which
    cond(M) alone cannot do. Returns the distinct count.
    """
    n_distinct = count_distinct_rows(X)
    if n_distinct <= basis_dim:
        raise InsufficientSamplesError(
            f"only {n_distinct} distinct input rows for a basis of D = {basis_dim} "
            f"terms (N = {np.asarray(X).shape[0]} rows in total); the polynomial is "
            f"unconstrained between the distinct points. Add distinct samples or "
            f"lower the degree."
        )
    return n_distinct


# --------------------------------------------------------------------------
# Array-level validation
# --------------------------------------------------------------------------
def _first_bad_index(bad: np.ndarray) -> tuple[int, ...]:
    flat = int(np.argmax(bad))
    return tuple(int(i) for i in np.unravel_index(flat, bad.shape))


def check_finite(arr: np.ndarray, name: str) -> None:
    """Raise ValueError naming the count and first index of non-finite entries.

    Where: ``PolyEmu.__init__`` on X, Y, X_test, Y_test; ``forward_emulator``
    / ``backward_emulator`` on their input.
    """
    a = np.asarray(arr)
    if a.dtype.kind not in "fc":
        return
    bad = ~np.isfinite(a)
    if bad.any():
        raise ValueError(
            f"{name} contains {int(bad.sum())} non-finite value(s); first at index "
            f"{_first_bad_index(bad)}"
        )


def as_float64(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a float64 ndarray, warning when precision was narrower.

    Where: entry of ``PolyEmu.__init__``, ``evaluate_monomials*``,
    ``compute_moments_vector_output*``, ``evaluate_emulator*`` (round-1
    ``float32-silent-precision-loss``, ``integer-dtype-overflow``,
    ``batched-evaluator-dtype-truncation``).
    """
    a = np.asarray(arr)
    if a.dtype == np.float64:
        return a
    if a.dtype.kind == "f" and np.finfo(a.dtype).bits < 64:
        warnings.warn(
            f"{name} is {a.dtype}; promoting to float64 (the moment matrix and the "
            f"solve must not run below float64).",
            PrecisionWarning,
            stacklevel=2,
        )
    elif a.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a.astype(np.float64)


def check_xy_shapes(X: np.ndarray, Y: np.ndarray) -> tuple[int, int, int]:
    """Validate (N, n) / (N, m) shapes; return (N, n, m).

    Where: first lines of ``PolyEmu.__init__`` (round-1
    ``constructor-no-input-validation``).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    for name, a in (("X", X), ("Y", Y)):
        if a.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D with shape (N, {'n_params' if name == 'X' else 'n_outputs'}), "
                f"got shape {a.shape}; reshape to (N, 1) for a single "
                f"{'parameter' if name == 'X' else 'output'}."
            )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows; got X.shape = {X.shape}, "
            f"Y.shape = {Y.shape}"
        )
    if X.shape[0] < 2:
        raise ValueError(f"at least 2 samples are required, got N = {X.shape[0]}")
    if X.shape[1] < 1 or Y.shape[1] < 1:
        raise ValueError(f"X and Y need at least one column; got X.shape = {X.shape}, Y.shape = {Y.shape}")
    return X.shape[0], X.shape[1], Y.shape[1]


def check_design_columns(X: np.ndarray, *, collinear_rtol: float = COLLINEAR_RTOL) -> None:
    """Raise on constant columns; warn on (numerically) collinear columns.

    Where: ``PolyEmu.__init__`` before ``scaler_X.fit_transform`` (round-1
    ``constant-or-collinear-x-column``).
    """
    X = np.asarray(X, dtype=np.float64)
    ptp = X.max(axis=0) - X.min(axis=0)
    const = np.where(ptp == 0)[0]
    if const.size:
        raise ValueError(
            f"input column(s) {const.tolist()} are constant (value "
            f"{X[0, const].tolist()}); remove or vary them. A constant column makes "
            f"every monomial containing it zero after standardization and the "
            f"moment matrix singular."
        )
    if X.shape[1] < 2:
        return
    Xc = (X - X.mean(axis=0)) / X.std(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    if s[-1] <= collinear_rtol * s[0]:
        rank = int((s > collinear_rtol * s[0]).sum())
        warnings.warn(
            f"standardized X has numerical rank {rank} of {X.shape[1]} (smallest / "
            f"largest singular value = {s[-1] / s[0]:.1e}); some input columns are "
            f"collinear and the moment matrix will be singular at every degree.",
            EmulatorWarning,
            stacklevel=2,
        )


def check_log_domain(Y: np.ndarray, name: str = "Y") -> None:
    """Raise when log_Y=True would take log of a non-positive or non-finite entry.

    Where: ``PolyEmu.__init__`` before ``np.log(Y_train)``; ``backward_emulator``
    before ``np.log(Y)``.
    """
    a = np.asarray(Y, dtype=np.float64)
    bad = ~(a > 0)  # catches <= 0 and NaN; inf handled separately
    bad |= ~np.isfinite(a)
    if bad.any():
        idx = _first_bad_index(bad)
        raise ValueError(
            f"log_Y=True requires {name} > 0 and finite but {int(bad.sum())} "
            f"entr{'y is' if bad.sum() == 1 else 'ies are'} not; first at "
            f"(row {idx[0]}, col {idx[1] if len(idx) > 1 else 0}) = {float(a[idx])!r}"
        )


def check_test_pair(X_test: object, Y_test: object) -> None:
    """Raise when exactly one of X_test / Y_test is given.

    Where: ``PolyEmu.__init__`` (round-1 ``x-test-without-y-test-ignored``).
    """
    if (X_test is None) != (Y_test is None):
        raise ValueError(
            "X_test and Y_test must both be given or both omitted (got X_test="
            f"{'None' if X_test is None else 'given'}, Y_test="
            f"{'None' if Y_test is None else 'given'})"
        )


def check_validation_split(
    n_samples: int,
    test_size: float,
    *,
    cross_validation: bool = True,
    n_test: int | None = None,
) -> tuple[int, int]:
    """Return (n_train, n_val) and raise when either side has < 2 rows.

    ``n_test`` is the explicit test-set size when X_test/Y_test were given.
    Where: ``PolyEmu.__init__`` before ``train_test_split``.
    """
    if n_test is not None:
        n_train, n_val = n_samples, n_test
    elif not cross_validation:
        n_train, n_val = n_samples, n_samples
    else:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size!r}")
        n_val = int(math.ceil(test_size * n_samples))  # sklearn's rule
        n_train = n_samples - n_val
    if n_val < 2:
        raise ValueError(
            f"validation split has {n_val} row(s) (N = {n_samples}, test_size = "
            f"{test_size!r}); at least 2 are required for an RMSE. Provide X_test/Y_test "
            f"or increase N."
        )
    if n_train < 2:
        raise ValueError(
            f"training split has {n_train} row(s) (N = {n_samples}, test_size = "
            f"{test_size!r}); at least 2 are required."
        )
    return n_train, n_val


def check_degree_range(
    init_deg: int,
    max_degree: int,
    *,
    direction: str,
    cap: int | None = None,
    n_train: int | None = None,
    n_params: int | None = None,
) -> None:
    """Raise ValueError (not assert) when init_deg > max_degree, stating both.

    Where: ``PolyEmu.__init__`` after the max_order cap, replacing the two
    ``assert`` statements (round-1 ``assert-for-user-input-validation``).
    """
    if init_deg < 0:
        raise ValueError(f"init_deg_{direction} must be >= 0, got {init_deg}")
    if init_deg > max_degree:
        capped = ""
        if cap is not None and max_degree == cap:
            capped = f" (max_degree_{direction} was capped at {cap} from N_train = {n_train}, n = {n_params})"
        raise ValueError(
            f"init_deg_{direction} = {init_deg} exceeds max_degree_{direction} = "
            f"{max_degree}{capped}; lower init_deg_{direction} or supply more samples."
        )


# --------------------------------------------------------------------------
# Conditioning of the moment matrix
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionReport:
    cond: float
    rank: int
    size: int
    lambda_min: float  # nan on the cholesky path
    lambda_max: float  # nan on the cholesky path
    level: str  # "ok" | "warn" | "singular"
    method: str = "eigh"  # "eigh" | "cholesky" | "eigh-fallback"

    @property
    def digits_left(self) -> float:
        return max(FLOAT64_DIGITS - math.log10(max(self.cond, 1.0)), 0.0)


COND_EIGH_MAX_D = 200  # above this, eigvalsh costs more than the Gram build (375 vs 63 ms at D=1001)


def _cond_eigh(M: np.ndarray) -> tuple[float, int, float, float]:
    """Exact 2-norm cond and numerical rank of symmetric M from one eigvalsh.

    For a symmetric matrix the singular values are |eigenvalues|, so
    cond_2 = max|l| / min|l| equals ``np.linalg.cond(M)`` (agreement 1e-9
    relative measured). Rank tolerance ``D * eps * lambda_max`` (LAPACK).
    """
    w = np.linalg.eigvalsh(M)
    aw = np.abs(w)
    lmax = float(aw.max())
    lmin_abs = float(aw.min())
    cond = math.inf if lmin_abs == 0.0 else lmax / lmin_abs
    rank = int((w > M.shape[0] * EPS64 * lmax).sum())
    return cond, rank, float(w.min()), float(w.max())


def _cond_cholesky(M: np.ndarray) -> float | None:
    """1-norm condition estimate from a Cholesky factor (LAPACK dpocon).

    Returns None when M is not numerically positive definite (Cholesky
    fails), which is itself a singularity signal. Measured 5.6 ms at D=1001
    (0.09x the Gram build); overestimates the 2-norm cond by 2.7-6.3x on
    real moment matrices and is exact on diagonal ones.
    """
    from scipy.linalg import cho_factor, lapack

    try:
        c, _ = cho_factor(M, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        return None
    rcond, info = lapack.dpocon(c, np.linalg.norm(M, 1))
    if info != 0:
        return None
    return math.inf if rcond == 0.0 else 1.0 / float(rcond)


def assess_conditioning(
    M: np.ndarray,
    *,
    warn_at: float = COND_WARN,
    raise_at: float = COND_RAISE,
    method: str = "auto",
) -> ConditionReport:
    """Pure: condition number, numerical rank and level of symmetric M.

    ``method="eigh"`` is exact (2-norm) and gives the rank; ``"cholesky"``
    is the O(D^2)-after-factorisation LAPACK estimate and falls back to
    eigh when Cholesky fails; ``"auto"`` dispatches on D at
    ``COND_EIGH_MAX_D``.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2-D, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError("M contains non-finite entries; check X for NaN/inf or overflow")
    if method not in ("auto", "eigh", "cholesky"):
        raise ValueError(f"method must be 'auto', 'eigh' or 'cholesky', got {method!r}")
    D = M.shape[0]
    if method == "auto":
        method = "eigh" if D <= COND_EIGH_MAX_D else "cholesky"
    used = method
    if method == "cholesky":
        est = _cond_cholesky(M)
        if est is not None:
            cond, rank, lmin, lmax = est, D, math.nan, math.nan
        else:
            used = "eigh-fallback"
    if used != "cholesky":
        cond, rank, lmin, lmax = _cond_eigh(M)
    level = "singular" if cond >= raise_at else "warn" if cond >= warn_at else "ok"
    return ConditionReport(cond, rank, D, lmin, lmax, level, used)


def check_conditioning(
    M: np.ndarray,
    *,
    warn_at: float = COND_WARN,
    raise_at: float = COND_RAISE,
    on_singular: str = "raise",
    degree: int | None = None,
    n_samples: int | None = None,
    method: str = "auto",
) -> ConditionReport:
    """Warn at cond >= warn_at; raise (or warn) at cond >= raise_at.

    Calibration (robust_cond_calibration.py, N = 20000, in-span polynomial
    targets, five (n, distribution) families):

    * cond 1e12: coefficient relative error first exceeds 1e-6 between
      cond 8.1e12 (n=1 uniform) and 4.4e16 (n=1 Gaussian); at 1e12 the solve
      keeps ~3.7 of 15.7 float64 digits in the coefficients. Fresh-point
      prediction error is still <= 1e-8 everywhere below 1e16. So 1e12 is a
      warning about EXPORTED coefficients / symbolic expressions, not about
      predictions.
    * cond 1e16 (> 1/eps = 4.5e15): eigenvalues of M turn negative and the
      condition number is no longer resolvable in float64. Prediction error
      in well-sampled sweeps at the first degree past 1e16 was 1.1e-8,
      6.9e-11, 1.0e-9, 6.3e-9, 1.1e-8 (worse than the previous degree in
      5 of 5 families: 3.4e-10, 4.5e-12, 7.8e-10, 2.1e-9, 2.1e-11), while
      design-deficient cases at cond
      3.7e17 / 1.0e19 overshoot fresh points by 11986x / 1607x. cond(M)
      cannot separate the two, so the raise is paired with
      ``check_distinct_rows`` / ``check_sample_count``, and inside the
      degree sweep the recommended policy is ``on_singular="warn"`` and stop
      the sweep at the previous degree.

    Where: ``solve_emulator_coefficients`` (public: on_singular="raise");
    ``generate_*_emulator`` loop (on_singular="warn", then break).
    """
    if on_singular not in ("raise", "warn"):
        raise ValueError(f"on_singular must be 'raise' or 'warn', got {on_singular!r}")
    rep = assess_conditioning(M, warn_at=warn_at, raise_at=raise_at, method=method)
    tag = f"degree {degree}: " if degree is not None else ""
    if rep.level == "singular":
        msg = (
            f"{tag}cond(M) = {rep.cond:.2e} exceeds {raise_at:.0e} (1/eps = {1 / EPS64:.1e}): "
            f"M is numerically singular (numerical rank {rep.rank} of {rep.size}, "
            f"lambda_min = {rep.lambda_min:.1e}, {rep.method}); np.linalg.solve would return one "
            f"arbitrary member of a solution family. Reduce the degree, add distinct "
            f"samples{f' (N_train = {n_samples})' if n_samples is not None else ''}, "
            f"or check for duplicated / collinear inputs."
        )
        if on_singular == "raise":
            raise IllConditionedError(msg)
        warnings.warn(msg, IllConditionedWarning, stacklevel=2)
    elif rep.level == "warn":
        warnings.warn(
            f"{tag}cond(M) = {rep.cond:.2e} exceeds {warn_at:.0e}: the solve keeps about "
            f"{rep.digits_left:.1f} of {FLOAT64_DIGITS:.1f} float64 digits in the "
            f"coefficients. Predictions are usually unaffected, but exported "
            f"coefficients and symbolic expressions are not trustworthy beyond that "
            f"many digits. Consider a lower degree, more samples, or an orthonormal basis.",
            IllConditionedWarning,
            stacklevel=2,
        )
    return rep


def check_coefficients_finite(coeffs: np.ndarray, *, degree: int | None = None) -> None:
    """Raise when the solve produced NaN/inf coefficients.

    Where: immediately after ``solve_emulator_coefficients`` in the sweep.
    """
    c = np.asarray(coeffs)
    bad = ~np.isfinite(c)
    if bad.any():
        tag = f"degree {degree}: " if degree is not None else ""
        raise IllConditionedError(
            f"{tag}{int(bad.sum())} of {c.size} fitted coefficients are non-finite "
            f"(first at (basis {int(np.argmax(bad.any(axis=1)))}, output "
            f"{int(np.argmax(bad.any(axis=0)))})); the moment matrix is singular or "
            f"contains non-finite entries."
        )


def check_predictions_finite(Y_pred: np.ndarray, *, strict: bool = False) -> int:
    """Warn (strict=False) or raise (strict=True) on non-finite predictions.

    Returns the number of affected rows. Where: end of ``forward_emulator``
    / ``backward_emulator`` and ``evaluate_emulator_batched``.
    """
    y = np.asarray(Y_pred)
    bad_rows = ~np.isfinite(y).reshape(y.shape[0], -1).all(axis=1) if y.ndim else np.array([not np.isfinite(y)])
    n_bad = int(bad_rows.sum())
    if n_bad:
        msg = (
            f"{n_bad} of {bad_rows.size} prediction row(s) are non-finite (first row "
            f"{int(np.argmax(bad_rows))}); the input is probably far outside the "
            f"training box (polynomial overflow)."
        )
        if strict:
            raise FloatingPointError(msg)
        warnings.warn(msg, EmulatorWarning, stacklevel=2)
    return n_bad


# --------------------------------------------------------------------------
# Training domain / extrapolation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainBox:
    lo: np.ndarray
    hi: np.ndarray
    scale: np.ndarray  # per-parameter std used to express distance in sigma


def fit_domain_box(X_train: np.ndarray) -> DomainBox:
    """Record min / max / std per parameter (2n + n floats). Where: fit time."""
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2-D, got shape {X.shape}")
    scale = X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return DomainBox(lo=X.min(axis=0).copy(), hi=X.max(axis=0).copy(), scale=scale)


def extrapolation_distance(X: np.ndarray, box: DomainBox) -> np.ndarray:
    """Pure: (N, n) array of how far each entry lies outside [lo, hi], in sigma."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, box.lo.shape[0])
    below = np.maximum(box.lo - X, 0.0)
    above = np.maximum(X - box.hi, 0.0)
    return (below + above) / box.scale


def check_in_domain(
    X: np.ndarray,
    box: DomainBox,
    *,
    strict: bool = False,
    tol_sigma: float = 0.0,
) -> np.ndarray:
    """Per-parameter extrapolation guard. Returns the (N, n) distance array.

    Warns (or raises with strict=True) listing, per offending parameter, the
    number of rows outside the box and the farthest excursion in sigma.
    Where: ``forward_emulator`` (box on X) and ``backward_emulator`` (box on Y).
    """
    if tol_sigma < 0:
        raise ValueError(f"tol_sigma must be >= 0, got {tol_sigma}")
    dist = extrapolation_distance(X, box)
    out = dist > tol_sigma
    if out.any():
        parts = []
        for j in np.where(out.any(axis=0))[0]:
            parts.append(
                f"parameter {j}: {int(out[:, j].sum())} of {dist.shape[0]} row(s) outside "
                f"[{box.lo[j]:.6g}, {box.hi[j]:.6g}], farthest {dist[:, j].max():.2f} sigma beyond"
            )
        msg = "input lies outside the training box; " + "; ".join(parts)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, ExtrapolationWarning, stacklevel=2)
    return dist


# --------------------------------------------------------------------------
# Basis pruning
# --------------------------------------------------------------------------
def check_nonempty_basis(mask: np.ndarray, *, threshold: float, stage: str = "dimension reduction") -> None:
    """Raise when pruning removed every basis mode. Where: after ``filter_modes``."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1:
        raise ValueError(f"mask must be 1-D (homogeneous), got shape {m.shape}")
    if not m.any():
        raise ValueError(
            f"{stage} removed every basis mode (0 of {m.size} kept at threshold "
            f"{threshold:.1e}); lower per_mode_thres or set dim_reduction=False."
        )


def keep_constant_term(mask: np.ndarray, multi_indices: np.ndarray) -> np.ndarray:
    """Return a NEW mask with the constant monomial (alpha == 0) forced to True.

    Coerce-and-warn: a basis without the constant term cannot represent the
    mean of a non-centred output and is not downward closed. Where: after
    ``filter_modes`` in both ``generate_*_emulator`` paths.
    """
    m = np.asarray(mask, dtype=bool)
    mi = np.asarray(multi_indices)
    const = np.where(~mi.any(axis=1))[0]
    if const.size == 0:
        return m.copy()
    j = int(const[0])
    if m[j]:
        return m.copy()
    warnings.warn(
        f"dimension reduction dropped the constant term (basis index {j}); it has been "
        f"kept so the basis stays downward closed and can represent the output mean.",
        EmulatorWarning,
        stacklevel=2,
    )
    new = m.copy()
    new[j] = True
    return new


# --------------------------------------------------------------------------
# Degree sweep and model selection
# --------------------------------------------------------------------------
def finite_candidates(rmse_list: RmseList) -> np.ndarray:
    """Indices of finite RMSE entries; raise if none. Where: ``select_best_model``."""
    rmse = np.asarray(rmse_list, dtype=np.float64)
    if rmse.ndim != 1 or rmse.size == 0:
        raise ValueError(f"rmse_list must be a non-empty 1-D sequence, got shape {rmse.shape}")
    finite = np.where(np.isfinite(rmse))[0]
    if finite.size == 0:
        raise ValueError(
            f"all {rmse.size} candidate models produced non-finite RMSE: {rmse.tolist()}; "
            f"check X/Y for NaN/inf and the moment matrix for singularity."
        )
    return finite


def select_within_tolerance(rmse_list: RmseList, rmse_tol: float) -> np.ndarray:
    """Finite indices whose RMSE <= min_finite * (1 + rmse_tol). Never empty."""
    if rmse_tol < 0:
        raise ValueError(f"rmse_tol must be >= 0, got {rmse_tol}")
    rmse = np.asarray(rmse_list, dtype=np.float64)
    finite = finite_candidates(rmse)
    rmse_min = rmse[finite].min()
    keep = finite[rmse[finite] <= rmse_min * (1.0 + rmse_tol)]
    return keep


def check_sweep_rmse(
    rmse_list: RmseList,
    degrees: Sequence[int],
    *,
    factor: float = SWEEP_BLOWUP_FACTOR,
) -> bool:
    """Warn when the newest validation RMSE is > factor x the best earlier one.

    Returns True when the blow-up fired. Where: inside the sweep loop after
    ``predictive_mse_aic_bic`` (round-1: RMSE rose 1e4x from d=10 to d=17 at
    N=200, n=2 with no message).
    """
    rmse = np.asarray(rmse_list, dtype=np.float64)
    if rmse.size < 2:
        return False
    prev = rmse[:-1]
    if not np.isfinite(prev).any():
        return False
    best_i = int(np.nanargmin(np.where(np.isfinite(prev), prev, np.nan)))
    ratio = rmse[-1] / prev[best_i]
    if not np.isfinite(rmse[-1]) or ratio > factor:
        warnings.warn(
            f"validation RMSE at degree {degrees[-1]} ({rmse[-1]:.3e}) is "
            f"{ratio:.1e}x the best earlier degree ({degrees[best_i]}: {prev[best_i]:.3e}); "
            f"the moment matrix is probably rank-deficient at this degree. Stopping the "
            f"sweep here is recommended.",
            IllConditionedWarning,
            stacklevel=2,
        )
        return True
    return False


# --------------------------------------------------------------------------
# Backend / wrapper helpers
# --------------------------------------------------------------------------
def output_scale(scaler: object, n_outputs: int) -> np.ndarray:
    """``scaler.scale_`` or ones when ``with_std=False`` left it None.

    Where: ``create_jax_emulator``, ``TorchMomentEmu.__init__`` (round-1
    ``scaler-none-breaks-autodiff``).
    """
    scale = getattr(scaler, "scale_", None)
    if scale is None:
        return np.ones(n_outputs, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (n_outputs,):
        raise ValueError(f"scaler.scale_ has shape {scale.shape}, expected ({n_outputs},)")
    return scale


def resolve_batch_shape(X: np.ndarray, n_params: int) -> tuple[np.ndarray, bool]:
    """Return (X as (N, n_params), single_sample). Raise on a trailing-axis mismatch.

    Where: both autodiff wrappers before scaling (round-1
    ``wrapper-1d-batch-silently-wrong``): a 1-D array is a single sample only
    when its length equals n_params.
    """
    a = np.asarray(X)
    if a.ndim == 0:
        if n_params != 1:
            raise ValueError(f"scalar input given but the emulator has n_params = {n_params}")
        return a.reshape(1, 1), True
    if a.shape[-1] != n_params:
        raise ValueError(
            f"input has {a.shape[-1]} element(s) along its last axis; expected n_params = "
            f"{n_params} (input shape {a.shape})"
        )
    return a.reshape(-1, n_params), a.ndim == 1


def check_backend_supports(emulator: object, backend: str) -> None:
    """Raise NotImplementedError for features a backend silently drops.

    Where: ``create_jax_emulator``, ``TorchMomentEmu.__init__``,
    ``SymbolicMomentEmu.__init__`` (round-1
    ``logy-ignored-by-all-three-backends``), until each applies ``exp``.
    """
    if getattr(emulator, "log_Y", False):
        raise NotImplementedError(
            f"the {backend} backend does not apply the inverse log transform for an "
            f"emulator fit with log_Y=True; it would return log(Y). Use "
            f"PolyEmu.forward_emulator, or fit with log_Y=False."
        )
    if not hasattr(emulator, "forward_coeffs"):
        raise ValueError(
            f"the {backend} backend needs a forward emulator; this PolyEmu was built "
            f"with forward=False."
        )
