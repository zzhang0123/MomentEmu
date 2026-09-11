from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from MomentEmu.guards import (
    COND_RAISE,
    COND_WARN,
    IllConditionedError,
    IllConditionedWarning,
    check_coefficients_finite,
    check_conditioning,
    finite_candidates,
)


####### Moment vector and matrix #################

def generate_moment_products(Phi, Y):
    """Generate moment products from evaluated basis functions Phi.
    
    Args:
        Phi: evaluated basis functions (N x D), where N is the number of samples and D is the number of basis functions.
        Y: data matrix (N x m), where m is the number of output variables.
        
    Returns:
        M: moment matrix (D x D)
        nu: moment vector (D x m)
    """
    N, D = Phi.shape
    M = (Phi.T @ Phi) / N                       # D x D
    nu = (Phi.T @ Y) / N                       # D x m
    return M, nu

def solve_emulator_coefficients(
    M,
    nu,
    *,
    on_singular: str = "raise",
    warn_at: float = COND_WARN,
    raise_at: float = COND_RAISE,
    degree: int | None = None,
    return_cond: bool = False,
):
    """Solve M c = nu for every output with a checked Cholesky factorisation.

    Args:
        M: moment matrix (D x D).
        nu: moment vector (D x m).
        on_singular: "raise" (default, direct calls) turns a numerically
            singular M or a failed Cholesky into IllConditionedError; "warn"
            warns and returns NaN coefficients with an infinite condition,
            which is what the degree sweep uses before dropping the rung.
        warn_at, raise_at: condition-number levels (D12).
        degree: sweep degree, used only in messages.
        return_cond: also return the condition estimate.

    Returns:
        coefficients (D x m), or (coefficients, cond) when return_cond=True.

    The estimate is the LAPACK dpocon 1-norm estimate from the Cholesky
    factor for large D and the exact symmetric 2-norm condition for D <= 200;
    it overestimates the 2-norm value by 2.7-6.3x on real moment matrices, so
    the 1e12 warning fires conservatively.
    """
    M = np.asarray(M, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a square 2-D matrix, got shape {M.shape}")
    if nu.ndim != 2 or nu.shape[0] != M.shape[0]:
        raise ValueError(
            f"nu must be (D, m) with D = {M.shape[0]}, got shape {nu.shape}"
        )
    # Warn at the D12 levels and raise at cond >= 1e16 when on_singular="raise".
    rep = check_conditioning(
        M,
        warn_at=warn_at,
        raise_at=raise_at,
        on_singular=on_singular,
        degree=degree,
        n_samples=None,
        method="auto",
    )
    from scipy.linalg import cho_factor, cho_solve

    try:
        cf, lower = cho_factor(M, lower=False, check_finite=False)
    except np.linalg.LinAlgError as exc:
        msg = (
            f"Cholesky factorisation of the {M.shape[0]}x{M.shape[1]} moment "
            f"matrix failed: M is not positive definite. The polynomial is not "
            f"identifiable on this design; reduce the degree or add distinct samples."
        )
        if on_singular == "raise":
            raise IllConditionedError(msg) from exc
        warnings.warn(msg, IllConditionedWarning, stacklevel=2)
        coeffs = np.full_like(nu, np.nan)
        return (coeffs, float("inf")) if return_cond else coeffs
    coeffs = cho_solve((cf, lower), nu, check_finite=False)
    check_coefficients_finite(coeffs, degree=degree)
    if return_cond:
        return coeffs, rep.cond
    return coeffs  # D x m

def filter_modes(coeffs, moment_matrix, threshold=1e-3, homogeneous=True):
    """
    Filter out modes with tiny contributions.
    
    Args:
        moment_matrix: moment matrix (D x D), where D is the number of basis functions.
        coeffs: coefficients array of shape D x m, where m is the number of variables (observables) to emulate.
        threshold: threshold for filtering out modes.
        homogeneous: all the observables use the same basis if True, otherwise allow different masks of basis functions.
        
    Returns: mask array, where True means the mode is kept. Of shape D if homogeneous, otherwise of shape D x m.
    """
    warnings.warn(
        "filter_modes is deprecated and will be removed in 3.0.0; post-hoc mode "
        "pruning raised validation RMSE by up to 963x and produced bases that are "
        "not downward closed. Use a declared basis (P5.3) or the low-rank / float32 "
        "storage options (P5.7) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Input validation
    if coeffs.ndim != 2:
        raise ValueError("coeffs must be a 2D array of shape (D, m)")
    if moment_matrix.ndim != 2 or moment_matrix.shape[0] != moment_matrix.shape[1]:
        raise ValueError("moment_matrix must be a square 2D array")
    if coeffs.shape[0] != moment_matrix.shape[0]:
        raise ValueError("coeffs and moment_matrix dimensions are incompatible")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    
    D, m = coeffs.shape
    
    # Total squared "energy" of each emulation function:
    # energy_list[j] = coeffs[:, j]^T @ moment_matrix @ coeffs[:, j]
    energy_list = np.einsum('ij,ij->j', coeffs, np.dot(moment_matrix, coeffs))
    
    # Handle edge case where energy is zero or negative
    if np.any(energy_list <= 0):
        # For zero or negative energy, keep all modes for safety
        if homogeneous:
            return np.ones(D, dtype=bool)
        else:
            return np.ones((D, m), dtype=bool)
    
    # Relative contribution of each mode to the total energy:
    # For mode i and observable j: (coeffs[i,j]^2 * moment_matrix[i,i]) / energy_list[j]
    moment_diag = np.diag(moment_matrix)
    relative_contribution = np.outer(moment_diag, np.ones(m)) * (coeffs**2)
    relative_contribution /= energy_list[np.newaxis, :]  # Broadcasting: (D, m) / (1, m)
    
    # Handle numerical issues
    relative_contribution = np.nan_to_num(relative_contribution, nan=0.0, posinf=0.0, neginf=0.0)
    
    if homogeneous:
        # Filter out modes with tiny contributions for all observables:
        # Keep a mode if it has significant contribution to ANY observable
        mask = np.any(relative_contribution >= threshold, axis=1)
    else:
        # Filter out modes with tiny contributions for each observable:
        # Keep modes independently for each observable
        mask = relative_contribution >= threshold
    
    return mask
    

####### Metrics, cost and penalties ##############
def metrics_and_penalties(RMSE, n, k):
    """
    Calculate AIC, AICc, and BIC. 
    Interpretation: Those are metrics defined with penalty on high dimensional representations. The lower the better.
    AIC: Akaike Information Criterion
    AICc: Corrected AIC (when n is not large when compared with k)
    BIC: Bayesian Information Criterion
    
    Args:
        RMSE: root mean squared error
        n: number of samples
        k: number of parameters
        
    Returns: AIC, AICc, BIC
    """

    AIC = 2 * (n * np.log(RMSE) + k)
    AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    BIC = 2 * n * np.log(RMSE) + k * np.log(n)

    return AIC, AICc, BIC


def predictive_mse_aic_bic(y_test, y_pred, k, n_train=None):
    """
    Compute predictive MSE, AIC and BIC on a test set, assuming Gaussian errors.
    
    Parameters
    ----------
    y_test : array-like
        True values on the test set
    y_pred : array-like
        Predicted values on the test set
    k : int
        Number of free parameters in the model
    n_train : int, optional
        Number of training samples. If provided, used for BIC penalty.
        If None, BIC uses n_test as a fallback (heuristic).
    
    Returns
    -------
    mse : float
        Predictive MSE
    aic : float
        Predictive AIC
    bic : float
        Predictive BIC
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    n_test = len(y_test)
    
    # Mean squared error on test set
    mse = np.mean((y_test - y_pred)**2)
    
    # AIC formula (up to additive constants)
    aic = n_test * np.log(mse) + 2 * k
    
    # BIC formula
    if n_train is None:
        n_bic = n_test  # fallback heuristic
    else:
        n_bic = n_train
    bic = n_bic * np.log(mse) + k * np.log(n_bic)

    rmse = np.sqrt(mse)
    
    return rmse, aic, bic


def select_best_model(rmse_list, aic_list=None, bic_list=None, rmse_tol=0.05):
    """
    Select the best model based on RMSE and optionally AIC/BIC.

    Parameters
    ----------
    rmse_list : array-like
        List of RMSE values for each model
    aic_list : array-like, optional
        List of AIC values for each model
    bic_list : array-like, optional
        List of BIC values for each model
    rmse_tol : float
        Fractional tolerance above the minimum RMSE to consider models (default 0.05 = 5%)

    Returns
    -------
    best_idx : int
        Index of the selected model
    """
    rmse = np.asarray(rmse_list, dtype=np.float64)

    # Mask to finite RMSE first; finite_candidates raises ValueError naming the
    # values when every candidate is NaN/inf (round-1 select-best-model-nan-crash).
    # The minimum finite RMSE is always within tolerance of itself, so the
    # candidate set is never empty and the old "expand the tolerance" branch was
    # dead code.
    finite = finite_candidates(rmse)
    rmse_min = rmse[finite].min()
    candidate_idxs = finite[rmse[finite] <= rmse_min * (1.0 + rmse_tol)]

    # Among candidates pick the lowest complexity proxy (BIC > AIC > RMSE).
    if bic_list is not None:
        bic = np.asarray(bic_list, dtype=np.float64)
        best_idx = int(candidate_idxs[np.argmin(bic[candidate_idxs])])
    elif aic_list is not None:
        aic = np.asarray(aic_list, dtype=np.float64)
        best_idx = int(candidate_idxs[np.argmin(aic[candidate_idxs])])
    else:
        best_idx = int(candidate_idxs[np.argmin(rmse[candidate_idxs])])

    return best_idx


####### Signal-mask aware fractional error #######
def signal_aware_frac_err(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    signal_floor_frac: float = 1e-3,
    absolute_floor: float = 1e-15,
    dr_threshold_decades: float = 3.0,
    per_output: bool = True,
) -> dict[str, Any]:
    """Signal-mask aware fractional-error diagnostic for emulator validation.

    Robustly compares a prediction to a reference whose entries may span many
    orders of magnitude. Naive ``|diff| / |ref|`` is undefined when ``|ref|``
    is at the floating-point noise floor. This helper computes a per-output
    signal floor of ``max(signal_floor_frac * peak, absolute_floor)`` and
    excludes below-floor entries from every fractional-error reduction
    (``max_rel`` and ``rmse``). Per-output dynamic range is measured and
    reported via ``dr_decades`` and ``strategy`` for diagnostics, but does
    not change the masking formula — see the ``dr_threshold_decades``
    parameter below.

    Parameters
    ----------
    pred, ref : ndarray
        Predicted and reference arrays. Must have matching shapes. Typical
        emulator-validation use is ``ref`` of shape ``(n_samples, n_outputs)``.
    signal_floor_frac : float, default 1e-3
        Mask floor as a fraction of the per-output peak amplitude. Entries
        whose magnitude is below ``signal_floor_frac * |ref|.max()`` (along
        the per-output axis) are treated as floating-point noise and excluded
        from the fractional-error reduction. ``0`` disables the mask and
        emits a ``UserWarning``.
    absolute_floor : float, default 1e-15
        Hard lower bound on the floor; defends against fixtures where the
        whole reference is at FP noise level. ``1e-15`` ≈ float64 ULP.
    dr_threshold_decades : float, default 3.0
        Diagnostic threshold on per-output dynamic range. Outputs whose
        dynamic range exceeds this value are tagged with strategy
        ``"signal_mask"``; the rest are tagged ``"plain_rel"``. The floor
        formula is identical on both branches
        (``max(signal_floor_frac * peak, absolute_floor)``) — this
        parameter controls the diagnostic *label* only, not the masking
        behaviour.
    per_output : bool, default True
        If True and ``ref`` is 2D, the dynamic-range detection and mask are
        applied per output column. If False (or ``ref`` is 1D), the whole
        array is treated as a single tensor.

    Returns
    -------
    dict
        Diagnostic-rich dict with the following keys. The shape of
        ``floor``, ``dr_decades`` and ``strategy`` depends on
        ``per_output``: in per-output mode they are arrays of shape
        ``(n_outputs,)``; in scalar mode (``per_output=False`` or 1D
        ``ref``) they collapse to Python ``float`` / ``str``.

        - ``max_rel`` (float): worst in-mask relative error across all
          outputs. ``nan`` when the mask is empty.
        - ``rmse`` (float): root-mean-square fractional error across
          in-mask entries. ``nan`` when the mask is empty.
        - ``n_above`` (int): number of entries above the signal floor.
        - ``n_total`` (int): total number of entries in ``ref``.
        - ``floor`` (float or ndarray): per-output (or scalar) floor
          actually used.
        - ``dr_decades`` (float or ndarray): per-output (or scalar)
          dynamic range estimate, ``log10(peak / min_positive)``. ``0``
          for all-zero columns.
        - ``strategy`` (str or ndarray): per-output (or scalar) label,
          ``"signal_mask"`` if the column's dynamic range exceeds
          ``dr_threshold_decades``, else ``"plain_rel"``. Diagnostic
          only — does not affect the floor.
        - ``argmax`` (tuple of int or None): index of the worst in-mask
          entry. ``None`` in two cases: ``n_above == 0`` (mask is empty),
          or ``max_rel == 0.0`` (every in-mask entry matched exactly).

    Raises
    ------
    ValueError
        If ``pred`` and ``ref`` have mismatched shapes, or any of
        ``signal_floor_frac``, ``absolute_floor``,
        ``dr_threshold_decades`` is negative.

    Warns
    -----
    UserWarning
        If ``signal_floor_frac == 0``: the mask degrades to
        ``absolute_floor`` for every output, effectively disabling the
        signal-aware behaviour.

    See Also
    --------
    PolyEmu : sets ``forward_frac_err_diag`` / ``backward_frac_err_diag``
        on the emulator instance using this helper when
        ``return_max_frac_err=True``.

    Notes
    -----
    The default ``signal_floor_frac=1e-3`` preserves entries that
    contribute ≥ 0.1 % to any aggregated downstream quantity. For
    near-strict relative-error gates with narrow dynamic range, lower it
    to ``1e-6`` or ``1e-9``. For wide-dynamic-range outputs where
    deep-tail entries are floating-point noise (e.g. ``1e-7`` to
    ``1e-24`` spans), the default is appropriate.

    This is a shape diagnostic, not an accuracy verdict: it does not
    determine posterior accuracy (use :meth:`PolyEmu.validate` with the
    data covariance), and a small ``max_rel`` can coexist with a large
    posterior shift.

    A useful calibration heuristic: pick ``signal_floor_frac`` so the
    masked-out entries' aggregate contribution to any user-facing
    observable is ≤ the rtol gate. For an rtol of ``1e-7`` on an
    observable that is at most cubic in this tensor, set the floor at
    ``(1e-7)^(1/3) ≈ 5e-3``.

    Examples
    --------
    Typical emulator-validation gate:

    >>> import numpy as np
    >>> from MomentEmu import signal_aware_frac_err
    >>> rng = np.random.default_rng(0)
    >>> ref = rng.uniform(1.0, 10.0, (100, 4))
    >>> pred = ref * (1.0 + 1e-9 * rng.standard_normal(ref.shape))
    >>> diag = signal_aware_frac_err(pred, ref)
    >>> diag["n_above"] > 0  # mask is non-empty
    True
    >>> diag["max_rel"] < 1e-7  # gate the worst in-mask entry
    True

    Branch-safely on the empty-mask case before reading ``argmax`` /
    ``max_rel`` for a log message:

    >>> if diag["n_above"] == 0:
    ...     print("signal mask is empty; check fixture for FP-noise output")
    ... elif diag["max_rel"] > 1e-7:
    ...     print(f"max rel err {diag['max_rel']:.2e} at {diag['argmax']}")
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(
            f"pred and ref must have matching shapes; got {pred.shape} vs {ref.shape}"
        )
    if signal_floor_frac < 0 or absolute_floor < 0 or dr_threshold_decades < 0:
        raise ValueError("signal_floor_frac, absolute_floor, dr_threshold_decades must be non-negative")
    if signal_floor_frac == 0:
        warnings.warn(
            "signal_floor_frac=0 disables the signal mask; the floor degrades to "
            "absolute_floor for every output. Set a positive value (default 1e-3) "
            "to engage the mask.",
            UserWarning,
            stacklevel=2,
        )

    abs_ref = np.abs(ref)
    abs_diff = np.abs(pred - ref)

    use_per_output = bool(per_output) and abs_ref.ndim == 2

    if use_per_output:
        # Per-column statistics broadcast back to original shape.
        peak = abs_ref.max(axis=0, keepdims=True)              # shape (1, m)
        # Per-column smallest positive entry; columns with no positive entry
        # collapse to NaN here and are forced to dr = 0 below.
        finite_pos = np.where(abs_ref > 0, abs_ref, np.nan)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            min_pos = np.nanmin(finite_pos, axis=0, keepdims=True)
        # dr_decades per column; 0 when all entries are zero or peak <= 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            dr = np.where(
                (peak > 0) & np.isfinite(min_pos) & (min_pos > 0),
                np.log10(peak / min_pos),
                0.0,
            )
        wide_dr = dr >= dr_threshold_decades                    # shape (1, m)
        # Single-formula floor: max(signal_floor_frac * peak, absolute_floor)
        # regardless of dynamic range. The wide_dr flag is kept purely as a
        # diagnostic label so a caller can see *why* a column was masked
        # aggressively.
        floor_signal = signal_floor_frac * peak                 # shape (1, m)
        floor = np.maximum(floor_signal, absolute_floor)        # shape (1, m)

        mask = abs_ref >= floor                                  # broadcasts to ref shape
        # Strategy label per column
        strategy = np.where(wide_dr.squeeze(0), "signal_mask", "plain_rel")
        floor_out = floor.squeeze(0).copy()
        dr_out = dr.squeeze(0).copy()
    else:
        peak = float(abs_ref.max()) if abs_ref.size else 0.0
        positive = abs_ref[abs_ref > 0]
        if positive.size and peak > 0:
            dr_scalar = float(np.log10(peak / positive.min()))
        else:
            dr_scalar = 0.0
        wide_dr_scalar = dr_scalar >= dr_threshold_decades
        # Same single-formula floor as the per-output branch; strategy
        # stays purely diagnostic.
        floor_scalar = max(signal_floor_frac * peak, absolute_floor)
        mask = abs_ref >= floor_scalar
        strategy = "signal_mask" if wide_dr_scalar else "plain_rel"
        floor_out = float(floor_scalar)
        dr_out = float(dr_scalar)

    n_total = int(ref.size)
    n_above = int(mask.sum())

    # Per-output mask counts (P1.3). A column with no in-mask entry is
    # reported separately: the old code let it drop out silently and reported
    # a finite max_rel for the rest, so a dead output looked healthy.
    if use_per_output:
        n_above_per_output = np.asarray(mask.sum(axis=0)).astype(int)
        fully_masked_outputs = np.flatnonzero(n_above_per_output == 0)
    else:
        n_above_per_output = int(n_above)
        fully_masked_outputs = np.array([0], dtype=int) if n_above == 0 else np.array([], dtype=int)
    if fully_masked_outputs.size:
        warnings.warn(
            f"signal_aware_frac_err: output column(s) {fully_masked_outputs.tolist()} "
            f"lie entirely below the floor {floor_out!r}; their fractional error is "
            f"undefined and they are excluded from max_rel and rmse.",
            UserWarning,
            stacklevel=2,
        )

    if n_above == 0:
        return {
            "max_rel": float("nan"),
            "rmse": float("nan"),
            "n_above": 0,
            "n_above_per_output": n_above_per_output,
            "fully_masked_outputs": fully_masked_outputs,
            "n_total": n_total,
            "floor": floor_out,
            "dr_decades": dr_out,
            "strategy": strategy,
            "argmax": None,
        }

    # Compute relative error only on masked entries; off-mask entries are
    # set to 0 so they neither contribute to the max nor inflate the RMSE.
    safe_abs_ref = np.where(mask, abs_ref, 1.0)                 # avoid /0 off-mask
    rel = np.where(mask, abs_diff / safe_abs_ref, 0.0)

    rel_in_mask = rel[mask]
    rmse = float(np.sqrt(np.mean(rel_in_mask ** 2)))
    max_rel = float(rel_in_mask.max())

    # When max_rel == 0 every in-mask entry is exact; np.argmax would still
    # return 0 but that index is arbitrary and would mislead a user reading
    # the diagnostic. Report None instead.
    if max_rel == 0.0:
        argmax_out = None
    else:
        flat_argmax = int(np.argmax(rel))
        argmax_out = tuple(int(i) for i in np.unravel_index(flat_argmax, ref.shape))

    return {
        "max_rel": max_rel,
        "rmse": rmse,
        "n_above": n_above,
        "n_above_per_output": n_above_per_output,
        "fully_masked_outputs": fully_masked_outputs,
        "n_total": n_total,
        "floor": floor_out,
        "dr_decades": dr_out,
        "strategy": strategy,
        "argmax": argmax_out,
    }





