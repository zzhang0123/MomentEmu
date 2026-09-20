from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from itertools import combinations_with_replacement
from typing import Any, NamedTuple

import numpy as np

from MomentEmu.core import (
    generate_moment_products,
    predictive_rmse_aic_bic,
    press_loo,
    select_best_model,
    signal_aware_frac_err,
    solve_emulator_coefficients,
)
from MomentEmu.guards import (
    COND_QR,
    COND_RAISE,
    BoxScaler,
    DomainBox,
    ExtrapolationWarning,
    IllConditionedError,
    IllConditionedWarning,
    InsufficientSamplesError,
    as_float64,
    basis_size,
    check_axis_levels,
    check_degree_range,
    check_design_columns,
    check_finite,
    check_log_domain,
    check_sample_count,
    check_sweep_rmse,
    check_test_pair,
    check_xy_shapes,
    connected_components,
    count_distinct_rows,
    extrapolation_distance,
    fit_domain_box,
    max_supported_degree,
    resolve_batch_shape,
)
from MomentEmu.monomials import (
    ChebyshevPlan,
    LegendrePlan,
    MonomialPlan,
    fold_output_affine,
)

#: An output column whose variation sits below this fraction of its own
#: magnitude carries no recoverable structure: its Legendre coefficients are
#: rounding noise, and normalising them by their own sum yields O(0.1)
#: "shares" indistinguishable from a real coupling.
_DEGENERATE_REL = 1e-10


def _degenerate_columns(fit: _LegendreFit) -> np.ndarray:
    """Output columns whose variation is at the rounding level of their scale.

    Their Legendre coefficients are noise, so normalising them by their own sum
    gives O(0.1) shares that are indistinguishable from a real signal.
    """
    const_row = int(np.flatnonzero(~fit.mi.any(axis=1))[0])
    signal_scale = np.maximum(
        np.abs(fit.coef[const_row]), np.sqrt(np.maximum(fit.total_var, 0.0))
    )
    return fit.explained <= (_DEGENERATE_REL * signal_scale) ** 2


#: Basis families the forward model can be fitted in. Which one is best is a
#: property of the DESIGN, not of the basis: a family is well conditioned when
#: the design is distributed like the measure it is orthogonal under.
BASIS_PLANS = {
    "monomial": MonomialPlan,
    "legendre": LegendrePlan,
    "chebyshev": ChebyshevPlan,
}


class _LegendreFit(NamedTuple):
    """One orthonormal-Legendre projection of Y over the training box.

    Every coefficient is an ANOVA component: ``coef[r] ** 2`` is the variance
    the interaction ``supp(mi[r])`` carries, so both the Sobol report and the
    interaction graph read the same numbers.
    """

    degree: int
    mi: np.ndarray            # (n_terms, n) multi-indices
    coef: np.ndarray          # (n_terms, m) Legendre coefficients
    var_terms: np.ndarray     # (n_terms, m) coef ** 2
    explained: np.ndarray     # (m,) variance carried by the non-constant terms
    total_var: np.ndarray     # (m,) empirical Var(Y)
    denom: np.ndarray         # (m,) explained, floored at 1 where it vanishes
    safe_total: np.ndarray    # (m,) total_var, floored at 1 where it vanishes
    n_terms: int
    n_samples: int


# Memory budget for a backward-sweep moment matrix M (D x D float64). The
# backward basis is built from n_outputs, which can be thousands of CMB bins,
# so a single degree rung can otherwise allocate tens of GB.
MAX_BACKWARD_MOMENT_BYTES = 1 << 30  # 1 GiB

# P5.2: default number of training rows per moment-build chunk. A cache-sized
# constant keeps the batched build enabled by default instead of one all-N
# batch that disables batching for every realistic N.
DEFAULT_BATCH_SIZE = 10000

# B-memory: the LOO path keeps the full Phi resident only when it fits this
# budget; above it the batched bordered build bounds the peak by batch_size.
# Set MOMENTEMU_PHI_BUDGET_BYTES=0 to force the strict batched path everywhere.
PHI_BUDGET_BYTES = int(
    os.environ.get("MOMENTEMU_PHI_BUDGET_BYTES", 512 * 1024 ** 2)
)

logger = logging.getLogger("MomentEmu")
logger.addHandler(logging.NullHandler())


def configure_logging(verbose: int = 1) -> None:
    """Send the sweep progress to stderr at INFO; verbose=0 stays silent."""
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if verbose and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


####### Multi-index generation and operations ####

# ---------------------------------------------------------------------------
# Per-output transform (P3.6, D4)
# ---------------------------------------------------------------------------
TRANSFORM_CODES = {"linear": 0, "log": 1, "asinh": 2}


def _normalize_transform(transform, log_Y, m):
    """Return an (m,) tuple of transform specs (P3.6, D4).

    ``None`` maps to all "log" when log_Y else all "linear"; a single string is
    broadcast.  A spec is "linear", "log", "asinh" or a (forward, inverse) pair
    of callables.
    """
    if transform is None:
        return tuple("log" if log_Y else "linear" for _ in range(m))
    if isinstance(transform, str):
        spec = (transform,) * m
    else:
        spec = tuple(transform)
    if len(spec) != m:
        raise ValueError(f"transform must have {m} entries, got {len(spec)}")
    for t in spec:
        if isinstance(t, str):
            if t not in TRANSFORM_CODES:
                raise ValueError(
                    f"unknown transform {t!r}; use linear, log, asinh or a pair"
                )
        elif not (isinstance(t, (tuple, list)) and len(t) == 2 and callable(t[0]) and callable(t[1])):
            raise ValueError(
                "each transform must be linear/log/asinh or a (forward, inverse) pair"
            )
    return spec


def _transform_forward(Y, transform):
    """Apply each column transform forward (Y -> model space)."""
    out = np.array(Y, dtype=np.float64, copy=True)
    for j, t in enumerate(transform):
        if t == "linear":
            continue
        if t == "log":
            out[:, j] = np.log(out[:, j])
        elif t == "asinh":
            out[:, j] = np.arcsinh(out[:, j])
        else:
            out[:, j] = t[0](out[:, j])
    return out


def _transform_inverse(Z, transform):
    """Apply each column transform inverse (model space -> Y)."""
    out = np.array(Z, dtype=np.float64, copy=True)
    for j, t in enumerate(transform):
        if t == "linear":
            continue
        if t == "log":
            out[:, j] = np.exp(out[:, j])
        elif t == "asinh":
            out[:, j] = np.sinh(out[:, j])
        else:
            out[:, j] = t[1](out[:, j])
    return out


def _transform_codes(transform):
    """Integer codes for the backends; callables are not supported there."""
    codes = []
    for t in transform:
        if t not in TRANSFORM_CODES:
            raise NotImplementedError(
                "the autodiff backends support linear/log/asinh column transforms "
                "only; use forward_emulator for a custom (forward, inverse) pair."
            )
        codes.append(TRANSFORM_CODES[t])
    return tuple(codes)


def refit_zoom(
    emulator,
    simulator,
    mean,
    sd,
    *,
    k=6.0,
    n_samples=None,
    seed=0,
    **kwargs,
):
    """Two-stage zoom refit around a posterior (P3.4).

    Samples the simulator uniformly in ``mean +/- k sd`` and fits a new
    PolyEmu there.  ``simulator`` takes an (N, n) array and returns (N,) or
    (N, m).  After the second-stage fit, check the posterior shift with
    :meth:`PolyEmu.posterior_bias`.
    """
    import numpy as _np

    mean = _np.asarray(mean, dtype=_np.float64)
    sd = _np.asarray(sd, dtype=_np.float64)
    n = mean.shape[0]
    if n_samples is None:
        n_samples = max(2000, 30 * n)
    rng = _np.random.default_rng(seed)
    X = rng.uniform(mean - k * sd, mean + k * sd, (n_samples, n))
    Y = _np.asarray(simulator(X), dtype=_np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    return PolyEmu(X, Y, **kwargs)


def given_order_indices(n, d):
    """Generate all multi-indices α with total degree = d.

    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for c in combinations_with_replacement(range(n), d):
        counter: dict[int, int] = Counter(c)
        alpha = [counter[i] for i in range(n)]
        indices.append(tuple(alpha))
    return np.array(indices)

def generate_multi_indices(n, d):
    """Generate all multi-indices α with total degree ≤ d.

    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for deg in range(d + 1):
        for c in combinations_with_replacement(range(n), deg):
            counter = Counter(c)
            alpha = [counter[i] for i in range(n)]
            indices.append(tuple(alpha))
    return np.array(indices)

def indices_selection(multi_indices, d_vec):
    """Select multi-indices where each component is ≤ corresponding component in d_vec.

    Args:
        multi_indices: array of multi-indices (each row is a multi-index)
        d_vec: vector of maximum degrees for each variable

    Returns:
        filtered array of multi-indices
    """
    # Convert to numpy array if not already
    multi_indices = np.array(multi_indices)
    d_vec = np.array(d_vec)

    # Check each multi-index: all components must be ≤ corresponding d_vec components
    mask = np.all(multi_indices <= d_vec, axis=1)

    return multi_indices[mask]

def generate_multi_indices_with_degree_vec(d_vec):
    """Generate all multi-indices α where α[i] ≤ d_vec[i] for each variable i.

    Args:
        d_vec: vector of maximum degrees for each variable

    Returns:
        array of multi-indices
    """
    from itertools import product

    # Convert to numpy array if not already
    d_vec = np.array(d_vec)
    n = len(d_vec)

    # Generate all combinations using Cartesian product
    # For each variable i, generate range(0, d_vec[i] + 1)
    ranges = [range(d_vec[i] + 1) for i in range(n)]

    # Use itertools.product to get all combinations
    indices = list(product(*ranges))

    return np.array(indices)

####### Monomial/Polynomial functions ############
def evaluate_monomials(X, multi_indices):
    """Evaluate φ_α(X) for all samples and all α."""
    N, n = X.shape
    D = len(multi_indices)
    Phi = np.empty((N, D), dtype=X.dtype)
    for j, alpha in enumerate(multi_indices):
        Phi[:, j] = np.prod(X ** alpha, axis=1)
    return Phi  # shape: N x D

def evaluate_monomials_lazy(X, multi_indices):
    """
    Efficiently evaluate monomials using on-demand caching to reduce memory use.
    """
    N, n = X.shape
    D = len(multi_indices)

    # Cache only needed powers: (i, d) -> X[:, i] ** d
    power_cache = {}

    Phi = np.empty((N, D), dtype=X.dtype)
    for j, alpha in enumerate(multi_indices):
        phi_j = np.ones(N, dtype=X.dtype)
        for i, deg in enumerate(alpha):
            if deg == 0:
                continue
            key = (i, deg)
            if key not in power_cache:
                power_cache[key] = X[:, i] ** deg
            phi_j *= power_cache[key]
        Phi[:, j] = phi_j
    return Phi

######## New method for solving the memory issue #########
def evaluate_monomials_batched(X, multi_indices, batch_size=10000, function=evaluate_monomials_lazy):
    """Batched wrapper around evaluate_monomials_lazy to limit peak memory."""
    N, n = X.shape
    D = len(multi_indices)
    Phi = np.empty((N, D), dtype=X.dtype)

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        Phi[i:batch_end] = evaluate_monomials_lazy(X[i:batch_end], multi_indices)
    return Phi
###########################################################


def compute_moments_vector_output(X, Y, multi_indices):
    """
    Vector-valued version of moment method.
    X: N x n input parameter array
    Y: N x m observable array
    multi_indices: list of multi-indices
    Returns: moment matrix Mm (D x D), moment vectors ν (D x m)
    """

    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D

    Mm, nu = generate_moment_products(Phi, Y)

    return Mm, nu

def compute_moments_vector_output_batched(X, Y, multi_indices, batch_size=10000,
                                          weights=None, plan_cls=None):
    """Batched moment build on the P0.7 plan, with optional sample weights (P5.2, P5.6).

    Args:
        X: N x n input parameter array
        Y: N x m observable array
        multi_indices: basis multi-indices
        batch_size: number of samples per batch
        weights: optional per-sample weights (N,), normalised to mean 1; None
            reproduces the unweighted moments bit for bit.
        plan_cls: basis plan class, defaulting to MonomialPlan. It MUST match
            the basis the caller evaluates with. Building the moments in one
            basis and predicting in another fits a different model and says
            nothing about it: the residual of the stored coefficients was 1.30
            where a direct solve in the same basis gave 2.5e-04.

    Returns: (Mm (D, D), nu (D, m)).
    """
    N, n = X.shape
    m = Y.shape[1]
    D = len(multi_indices)
    Mm = np.zeros((D, D), dtype=X.dtype)
    nu = np.zeros((D, m), dtype=X.dtype)
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != N:
            raise ValueError(f"weights has {w.shape[0]} entries, expected {N}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        w = w / w.mean()
        if np.all(w == 1.0):
            w = None  # P5.6: uniform weights == unweighted, bit for bit
    # One plan for all batches (P0.7/P5.2).
    plan = (plan_cls or MonomialPlan).build(multi_indices)
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        Phi_batch = plan.evaluate(X_batch)  # (batch, D)
        if w is None:
            Mm += Phi_batch.T @ Phi_batch
            nu += Phi_batch.T @ Y_batch
        else:
            w_batch = w[start_idx:end_idx]
            Mm += (Phi_batch.T * w_batch) @ Phi_batch
            nu += (Phi_batch.T * w_batch) @ Y_batch
    Mm /= N
    nu /= N
    return Mm, nu

def symbolic_polynomial_expressions(coeffs, multi_indices, variable_names=None,
                                    input_means=None, input_vars=None,
                                    output_means=None, output_vars=None,
                                    *, log_input=False, log_output=False,
                                    transform=None, input_transform=None,
                                    raw_units=False, family="monomial"):
    """Convert emulator coefficients into sympy expressions (P2.4, D11).

    The default form keeps the standardized coordinates
    z_i = (x_i - m_i)/s_i as an unexpanded rational expression, evaluated with
    17 significant digits. Expanding in raw units loses digits with degree (up
    to 8.3e-9 at n=6, d=6 in the review), so raw_units=True warns.
    log_input substitutes log of the input variables (backward export of a
    log_Y emulator); log_output wraps the result in exp(..., evaluate=False).

    ``family`` names the basis the coefficients were fitted against, and the
    expression is written in THAT basis: sympy's ``legendre`` and
    ``chebyshevt`` agree with the package's recurrences to 4.4e-16. Printing
    the numbers against monomials instead would be silently wrong, which is
    why the export used to refuse anything else.

    A tensor family clips its argument to [-1, 1], exactly as ``_TensorPlan``
    does, and the clip is part of the expression rather than an assumption
    about where it will be evaluated. Monomials are not clipped: they diverge
    outside the box and that is the model a monomial fit describes.
    """
    import sympy as sp

    if raw_units:
        warnings.warn(
            "raw_units=True expands the polynomial in raw coordinates, which "
            "loses digits as the degree grows (up to 8.3e-9 at n=6, d=6); the "
            "default standardized z-form is numerically safe.",
            UserWarning,
            stacklevel=2,
        )
    coeffs = np.asarray(coeffs, dtype=float)
    mi = np.asarray(multi_indices)
    D, m = coeffs.shape
    n = mi.shape[1]
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(n)]
    elif len(variable_names) != n:
        raise ValueError(
            f"expected {n} variable names, one per input, got "
            f"{len(variable_names)}"
        )
    vars_sym = [sp.Symbol(name) for name in variable_names]
    input_mean = np.zeros(n) if input_means is None else np.asarray(input_means, float)
    input_std = np.ones(n) if input_vars is None else np.sqrt(np.asarray(input_vars, float))
    output_mean = np.zeros(m) if output_means is None else np.asarray(output_means, float)
    output_std = np.ones(m) if output_vars is None else np.sqrt(np.asarray(output_vars, float))
    if family not in ("monomial", "legendre", "chebyshev"):
        raise NotImplementedError(f"symbolic export has no basis {family!r}")
    clip = family != "monomial"
    z_syms = sp.symbols([f"__z{i}" for i in range(n)])
    substitutions = {}
    for i in range(n):
        xi = vars_sym[i]
        if input_transform is not None:
            ti = input_transform[i]
            if ti == "log":
                xi = sp.log(xi, evaluate=False)
            elif ti == "asinh":
                xi = sp.asinh(xi)
            elif ti != "linear":
                raise NotImplementedError(
                    "symbolic export supports linear/log/asinh input transforms"
                )
        elif log_input:
            xi = sp.log(xi, evaluate=False)
        zi = (
            xi - sp.Float(float(input_mean[i]), 17)
        ) / sp.Float(float(input_std[i]), 17)
        substitutions[z_syms[i]] = sp.Max(-1, sp.Min(1, zi)) if clip else zi

    # One table of the family's 1-D polynomials per variable, in the z symbol;
    # the substitution above carries them into the caller's coordinates.
    tables: list[list] = []
    if clip:
        for i in range(n):
            top = int(mi[:, i].max()) if mi.size else 0
            tables.append([
                sp.legendre(k, z_syms[i]) * sp.sqrt(2 * k + 1)
                if family == "legendre" else sp.chebyshevt(k, z_syms[i])
                for k in range(top + 1)
            ])
    expressions = []
    for j in range(m):
        terms: dict = {}
        for c, alpha in zip(coeffs[:, j], mi):
            key = tuple(int(a) for a in alpha)
            terms[key] = terms.get(key, sp.Integer(0)) + sp.Float(float(c), 17)
        if clip:
            total = sp.Integer(0)
            for key, c in terms.items():
                term = c
                for i, a in enumerate(key):
                    if a:
                        term = term * tables[i][a]
                total = total + term
            base = total
        else:
            base = sp.Poly.from_dict(terms, z_syms).as_expr()
        # xreplace is a direct tree substitution and is ~8x faster than subs
        # here (0.019 s vs 0.157 s at D=462), which keeps the export under 0.1 s.
        expr = base.xreplace(substitutions)
        if raw_units:
            expr = sp.expand(expr)
        if float(output_std[j]) != 1.0:
            expr = expr * sp.Float(float(output_std[j]), 17)
        if float(output_mean[j]) != 0.0:
            expr = expr + sp.Float(float(output_mean[j]), 17)
        if transform is not None:
            tj = transform[j]
            if tj == "log":
                expr = sp.exp(expr, evaluate=False)
            elif tj == "asinh":
                expr = sp.sinh(expr)
            elif tj != "linear":
                raise NotImplementedError(
                    "symbolic export supports linear/log/asinh output transforms"
                )
        elif log_output:
            expr = sp.exp(expr, evaluate=False)
        expressions.append(expr)
    return expressions

def evaluate_emulator(X, coeffs, multi_indices):
    """
    Evaluate the polynomial emulator at inputs X using known coefficients.
    X: N x n
    coeffs: D x m
    multi_indices: list of α
    Returns: Y_pred: N x m
    """
    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D
    return Phi @ coeffs  # N x m

def evaluate_emulator_batched(X, coeffs, multi_indices, batch_size=10000):
    """
    Memory-efficient emulator evaluation for large datasets.
    """
    N = X.shape[0]
    m = coeffs.shape[1]
    Y_pred = np.empty((N, m), dtype=X.dtype)

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        X_batch = X[start_idx:end_idx]
        Phi_batch = evaluate_monomials_lazy(X_batch, multi_indices)
        Y_pred[start_idx:end_idx] = Phi_batch @ coeffs
        del Phi_batch, X_batch

    return Y_pred

def max_order(n_params, N_samples):
    """Deprecated: the old off-by-one degree helper.

    It returned the SMALLEST k with C(n+k, k) >= N, a degree whose basis is
    rank-deficient. Use :func:`MomentEmu.guards.max_supported_degree`, which
    returns the LARGEST identifiable degree (D < N).
    """
    warnings.warn(
        "max_order is deprecated and returned a degree with D >= N; use "
        "guards.max_supported_degree(n_params, N_samples, fill=1) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return max_supported_degree(n_params, N_samples, fill=1.0)


def _unscale_val(scaler_X, scaler_Y, X_val_scaled, Y_val_scaled, transform):
    """Invert StandardScaler and the per-output transform on a validation tuple."""
    X = scaler_X.inverse_transform(X_val_scaled)
    Y = scaler_Y.inverse_transform(Y_val_scaled)
    Y = _transform_inverse(Y, transform)
    return X, Y


def _public_max_frac_err(diag):
    """Public forward/backward_max_frac_err from a signal_aware_frac_err dict.

    Returns inf when the mask is empty, any output column is fully masked,
    or max_rel is not finite, so a threshold check fails loudly.
    """
    if diag["n_above"] == 0:
        return float("inf")
    if len(diag.get("fully_masked_outputs", ())) > 0:
        return float("inf")
    if not np.isfinite(diag["max_rel"]):
        return float("inf")
    return float(diag["max_rel"])


def _report_frac_err(
    label,
    pred,
    ref,
    *,
    signal_floor_frac=1e-3,
    absolute_floor=1e-15,
    dr_threshold_decades=3.0,
):
    """Compute and pretty-print the signal-aware fractional-error diagnostic.

    Returns the diagnostic dict from :func:`signal_aware_frac_err` so that the
    caller can attach it to the emulator instance. See that function's
    docstring for parameter calibration.
    """
    diag = signal_aware_frac_err(
        pred,
        ref,
        signal_floor_frac=signal_floor_frac,
        absolute_floor=absolute_floor,
        dr_threshold_decades=dr_threshold_decades,
    )

    if diag["n_above"] == 0:
        print(
            f"\n{label} emulator fractional-error diagnostic: signal mask empty "
            f"(no entries above floor {diag['floor']!r}); reference appears to be "
            f"at floating-point noise level."
        )
        return diag

    ind = diag["argmax"]
    strategy = diag["strategy"]
    if isinstance(strategy, np.ndarray):
        strategy_label = "per-output: " + ", ".join(str(s) for s in strategy.tolist())
    else:
        strategy_label = str(strategy)

    # When max_rel == 0 every in-mask entry matched exactly; argmax is None.
    # Indexing ref[None] would silently mean ref[np.newaxis] in NumPy, which
    # would dump the whole array instead of a single entry — handle this
    # branch explicitly with no per-entry indexing.
    if ind is None:
        print(
            f"\n{label} emulator signal-aware fractional error:"
            f"\n  max_rel = {diag['max_rel']:.3e}  rmse = {diag['rmse']:.3e}"
            f"\n  all in-mask entries match exactly"
            f"\n  in-mask entries: {diag['n_above']}/{diag['n_total']}"
            f"  | strategy: {strategy_label}"
            f"\n  (entries below the per-output signal floor are excluded)"
        )
        return diag

    print(
        f"\n{label} emulator signal-aware fractional error:"
        f"\n  max_rel = {diag['max_rel']:.3e}  rmse = {diag['rmse']:.3e}"
        f"\n  worst at index {ind}: true = {ref[ind]!r}, predicted = {pred[ind]!r}"
        f"\n  in-mask entries: {diag['n_above']}/{diag['n_total']}"
        f"  | strategy: {strategy_label}"
        f"\n  (entries below the per-output signal floor are excluded)"
    )
    return diag


def _warn_export_conditioning(cond, direction):
    """D12: warn when a symbolic export starts from cond(M) >= 1e8."""
    if cond is None:
        return
    cond = float(cond)
    if np.isfinite(cond) and cond >= 1e8:
        warnings.warn(
            f"{direction} symbolic export at cond(M) = {cond:.2e} >= 1e8: the "
            "exported coefficients are not trustworthy to full float64 precision.",
            IllConditionedWarning,
            stacklevel=3,
        )


def _safe_cholesky(M):
    """Cholesky factor M = U^T U, with jitter; None when even that fails (B6).

    A completed degree sweep must not be discarded because the selected rung is
    numerically singular. On failure the jitter is grown relative to the mean
    diagonal; leverage() then falls back to the pseudoinverse.
    """
    from scipy.linalg import cho_factor

    M = np.asarray(M, dtype=np.float64)
    try:
        return cho_factor(M, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        diag = float(np.mean(np.abs(np.diag(M))))
        if not np.isfinite(diag) or diag <= 0.0:
            diag = 1.0
        for jitter in (1e-12, 1e-10, 1e-8):
            try:
                return cho_factor(
                    M + (diag * jitter) * np.eye(M.shape[0]),
                    lower=False, check_finite=False,
                )
            except np.linalg.LinAlgError:
                continue
        warnings.warn(
            "Cholesky factorisation failed even with jitter; leverage() will "
            "fall back to the pseudoinverse.",
            IllConditionedWarning,
            stacklevel=2,
        )
        return None


class PolyEmu:
    """Polynomial moment-projection emulator; see __init__ for the arguments."""
    # Validation diagnostics, populated only when return_max_frac_err=True.
    # Class-level defaults make these attributes safe to read unconditionally
    # (returns None instead of AttributeError when the diagnostic was not
    # requested at construction time).
    forward_max_frac_err: float | None = None
    backward_max_frac_err: float | None = None
    forward_frac_err_diag: dict | None = None
    backward_frac_err_diag: dict | None = None
    # True when the last forward sweep grew the moment system incrementally
    # (P5.1); overridden per instance after a fit.
    forward_sweep_incremental_: bool = False
    # Storage-option diagnostics (P5.7/B7), absent unless the option ran.
    forward_rank_: int | None = None
    forward_singular_values_: np.ndarray | None = None
    rank_difference_: np.ndarray | None = None
    float32_difference_: np.ndarray | None = None

    @property
    def foward_degree(self):
        """Deprecated typo alias for forward_degree (removed in 3.0.0)."""
        warnings.warn(
            "foward_degree is a typo; use forward_degree (the alias is removed "
            "in 3.0.0).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.forward_degree

    @foward_degree.setter
    def foward_degree(self, value):
        self.forward_degree = value

    def __init__(self,
                X,
                Y,
                X_test=None,
                Y_test=None,
                log_Y=False,
                cross_validation=None,
                test_size=0.15,
                # RMSE_upper=1.0,
                RMSE_tol=1e-2,
                fRMSE_tol=1e-1,
                forward=True,
                backward=False,
                init_deg_forward=None,
                init_deg_backward=None,
                max_degree_forward=None,
                max_degree_backward=None,
                dim_reduction=False,
                per_mode_thres=None,
                return_max_frac_err=False,
                standardize_Y_with_std=True,
                batch_size=None,
                random_state=None,
                verbose=0,
                transform=None,
                weights=None,
                basis=None,
                scaling="standard",
                basis_kind="monomial",
                ridge=0.0,
                parameter_names=None):
        """
        Polynomial emulator class for both forward and backward emulation.
        X: N x n array of input parameters. N is the number of samples, n is the number of parameters.
        Y: N x m array of observables. m is the number of observables.
        X_test, Y_test: optional test/validation sets. If not provided, a split from X, Y will be used.
        test_size: fraction of data to use for validation if X_test, Y_test not provided.
        RMSE_tol: target validation RMSE, expressed as a fraction of the
            RMS of the validation Y in the fitting space, so it is
            scale-free (standardize_Y_with_std does not change its meaning).
        fRMSE_tol: tolerance for selecting best model based on RMSE. We select the simplest model within fRMSE_tol (fractional range) of the lowest RMSE.

        forward: whether to generate forward emulator.
        backward: whether to generate backward emulator.
        init_deg_forward, init_deg_backward: initial polynomial degree for forward/backward emulators.
        max_degree_forward, max_degree_backward: maximum polynomial degree for forward/backward emulators.
        dim_reduction: retired in 2.0.0 (D15); True only emits a DeprecationWarning.

        per_mode_thres: retired in 2.0.0 (D15); a value only warns.
        return_max_frac_err: whether to compute and store the signal-aware
            fractional-error diagnostic on the validation set. See the
            "Validation diagnostics" section below for the attributes set
            when this is enabled.
        standardize_Y_with_std: whether to standardize Y with standard deviation (True) or only mean (False).
        batch_size: batch size for batched computations to manage memory usage.
        random_state: seed passed to the internal train/validation split when
            X_test/Y_test are omitted. Stored as self.random_state.
        transform: per-output transform (P3.6, D4): "linear", "log", "asinh", a
            single string broadcast to every column, or a list/array of those
            (or a (forward, inverse) callable pair). None means "log" for every
            column when log_Y else "linear". log_Y=True is equivalent to
            transform="log".

        Validation diagnostics
        ----------------------
        Set on the instance only when ``return_max_frac_err=True``; these
        attributes default to ``None`` otherwise and are always safe to
        read.

        forward_max_frac_err : float or None
            Worst in-mask relative error from the forward emulator on the
            validation split. ``float('inf')`` when the signal mask is
            empty (so threshold checks fail loudly), ``None`` when the
            forward branch was not built or ``return_max_frac_err=False``.
        forward_frac_err_diag : dict or None
            Full diagnostic dict from :func:`signal_aware_frac_err` for the
            forward emulator. Keys: ``max_rel``, ``rmse``, ``n_above``,
            ``n_total``, ``floor``, ``dr_decades``, ``strategy``,
            ``argmax``. ``None`` when the diagnostic was not requested.
        backward_max_frac_err : float or None
            Analogous to ``forward_max_frac_err`` for the backward
            emulator.
        backward_frac_err_diag : dict or None
            Analogous to ``forward_frac_err_diag`` for the backward
            emulator.

        See :func:`signal_aware_frac_err` for the diagnostic's
        parameter calibration and the meaning of each dict key.
        """

        # Hyper-parameters kept so fit() can refit with the same settings (P4.1).
        self._hyperparameters = dict(
            log_Y=log_Y,
            cross_validation=cross_validation,
            test_size=test_size,
            RMSE_tol=RMSE_tol,
            fRMSE_tol=fRMSE_tol,
            forward=forward,
            backward=backward,
            init_deg_forward=init_deg_forward,
            init_deg_backward=init_deg_backward,
            max_degree_forward=max_degree_forward,
            max_degree_backward=max_degree_backward,
            dim_reduction=dim_reduction,
            per_mode_thres=per_mode_thres,
            return_max_frac_err=return_max_frac_err,
            standardize_Y_with_std=standardize_Y_with_std,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
            transform=transform,
            weights=weights,
            basis=basis,
            parameter_names=parameter_names,
        )

        # Imported here, not at module scope, so `import MomentEmu` and the
        # numpy-only inference path do not pay for sklearn (P0.9).
        from sklearn.preprocessing import StandardScaler

        # D15: post-hoc mode pruning is retired in 2.0.0. Keep accepting the
        # arguments for one release but ignore them, with one warning.
        if dim_reduction or per_mode_thres is not None:
            warnings.warn(
                "dim_reduction / per_mode_thres were removed in 2.0.0 (D15) and "
                "are ignored: the default pruning raised validation RMSE up to "
                "963x and pruned sets are not downward closed. For a smaller "
                "basis use basis= (P5.3); for smaller storage use the low-rank "
                "or float32 options (P5.7).",
                DeprecationWarning,
                stacklevel=2,
            )

        # D1: validate and promote at the entry.  as_float64 rejects object
        # dtype and warns on float32; check_xy_shapes rejects 1-D Y (with a
        # reshape hint); check_design_columns raises on constant columns and
        # warns on collinear ones; check_log_domain rejects Y <= 0 for log_Y.
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        check_finite(X, "X")
        check_finite(Y, "Y")
        check_xy_shapes(X, Y)
        check_test_pair(X_test, Y_test)
        if X_test is not None:
            X_test = as_float64(np.asarray(X_test), "X_test")
            Y_test = as_float64(np.asarray(Y_test), "Y_test")
            check_finite(X_test, "X_test")
            check_finite(Y_test, "Y_test")
        check_design_columns(X)
        self._X_data = X
        self._Y_data = Y
        self.basis = basis
        if basis is not None and backward:
            raise ValueError(
                "basis= describes the forward basis over the n_params inputs and "
                "cannot be applied to the backward map over the n_outputs outputs. "
                "Fit the forward model with basis=, or fit the backward model only "
                "(forward=False, backward=True) without basis."
            )
        # A Basis degree is a constraint like every other Basis constraint, so
        # it must not be silently overridden by the sweep. degree and
        # max_degree_forward are the same knob -- the top of the forward sweep --
        # so a disagreement is an error rather than a silent win for either.
        # init_deg_forward is a different knob (where the sweep starts) and is
        # left alone unless it starts above the cap.
        # getattr: a duck-typed basis only has to provide build() and spec().
        _basis_degree = None if basis is None else getattr(basis, "degree", None)
        if _basis_degree is not None:
            if max_degree_forward is not None and int(max_degree_forward) != _basis_degree:
                raise ValueError(
                    f"basis={basis.spec()} and max_degree_forward = "
                    f"{int(max_degree_forward)} both set the top of the forward degree "
                    f"sweep and disagree. The Basis degree bounds the index set, so the "
                    f"sweep cannot run past it. Drop max_degree_forward to sweep up to "
                    f"the Basis degree ({_basis_degree}), or build the Basis without a "
                    f"degree to let max_degree_forward drive the sweep."
                )
            if init_deg_forward is not None and int(init_deg_forward) > _basis_degree:
                raise ValueError(
                    f"init_deg_forward = {int(init_deg_forward)} starts the forward "
                    f"degree sweep above the degree of basis={basis.spec()}, so no rung "
                    f"of the sweep is admissible. Lower init_deg_forward to at most "
                    f"{_basis_degree}, or raise the Basis degree."
                )
            max_degree_forward = _basis_degree
        self.parameter_names = list(parameter_names) if parameter_names is not None else [f"x{i}" for i in range(X.shape[1])]

        self.n_params = X.shape[1]
        self.n_outputs = Y.shape[1]
        self.transform = _normalize_transform(transform, log_Y, self.n_outputs)
        # log_Y is kept as the legacy "every column is log" flag.
        self.log_Y = all(t == "log" for t in self.transform)
        if self.log_Y:
            check_log_domain(Y, "Y")
            if X_test is not None:
                check_log_domain(Y_test, "Y_test")
        else:
            for j, t in enumerate(self.transform):
                if t == "log":
                    check_log_domain(Y[:, j:j + 1], f"Y column {j}")
                    if X_test is not None:
                        check_log_domain(Y_test[:, j:j + 1], f"Y_test column {j}")
        self.standardize_Y_with_std = standardize_Y_with_std
        self.random_state = random_state
        self.verbose = verbose
        configure_logging(verbose)
        self.loo_rmse_ = None
        self.leverage_max_train_ = None
        self.forward_sweep_incremental_ = False
        # Training box (P1.6): raw min/max/std per parameter and per output.
        self.X_box_ = fit_domain_box(X)
        self.Y_box_ = fit_domain_box(Y)

        # D14: cross_validation is deprecated. Without an explicit X_test the
        # degree is selected by exact leave-one-out PRESS on all N (P1.5); with
        # X_test the held-out RMSE is used.
        if cross_validation is not None:
            warnings.warn(
                "cross_validation is deprecated and ignored in 2.0.0: without "
                "X_test/Y_test the degree is selected by leave-one-out PRESS on "
                "all N. Pass X_test/Y_test for a held-out split. Removed in 3.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        self.batch_size_ = int(batch_size)

        use_loo = X_test is None
        if use_loo:
            # Fit on all N; the sweep scores each degree by LOO PRESS.
            X_train, Y_train = X, Y
            X_val, Y_val = None, None
            cross_val = False
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = X_test, Y_test
            cross_val = True

        Y_train = _transform_forward(Y_train, self.transform)
        if cross_val:
            Y_val = _transform_forward(Y_val, self.transform)

        # Scale the training data
        if float(ridge) < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")
        # Tikhonov on the moment matrix, scaled to its mean diagonal. Off by
        # default: it trades bias for variance, and that is the caller's call.
        self.ridge = float(ridge)
        if basis_kind not in BASIS_PLANS:
            raise ValueError(
                f"basis_kind must be one of {tuple(BASIS_PLANS)}, got {basis_kind!r}"
            )
        self.basis_kind = str(basis_kind)
        if basis_kind != "monomial":
            # Both families are defined on [-1, 1] and clipped outside it, so
            # the box map is not a preference here but a precondition.
            if scaling == "standard":
                scaling = "box"
            if backward:
                raise ValueError(
                    f"basis_kind={basis_kind!r} covers the forward model only; "
                    "the backward map is fitted over the n_outputs outputs and "
                    "its symbolic and storage paths assume monomials. Fit the "
                    "forward model with this basis, or the backward model "
                    "separately with the default."
                )
        if scaling not in ("standard", "box"):
            raise ValueError(
                f"scaling must be 'standard' or 'box', got {scaling!r}"
            )
        self.scaling = str(scaling)
        # A polynomial basis wants a bounded argument; dividing by the standard
        # deviation does not bound one. See guards.BoxScaler.
        # Deliberately untyped: io.load rebinds this to an ArrayScaler, and a
        # StandardScaler | BoxScaler union would make that assignment an error.
        self.scaler_X: Any = (
            StandardScaler() if scaling == "standard" else BoxScaler()
        )
        self.scaler_Y = StandardScaler(with_std=self.standardize_Y_with_std)

        # in-place scaling transformation
        X_train = self.scaler_X.fit_transform(X_train)
        Y_train = self.scaler_Y.fit_transform(Y_train)
        # Cached inverses for the folded inference path (P0.7). X is always
        # standardized with std; Y may have scale_ = None (with_std=False).
        self._inv_scale_X = 1.0 / self.scaler_X.scale_
        _y_scale = self.scaler_Y.scale_
        if _y_scale is None:
            _y_scale = np.ones(self.n_outputs)
        self._inv_scale_Y = 1.0 / _y_scale
        if cross_val:
            X_val = self.scaler_X.transform(X_val)
            Y_val = self.scaler_Y.transform(Y_val)
        else:
            # using the training set to define the fitting error
            X_val, Y_val = X_train, Y_train



        if forward:
            logger.info("Generating forward emulator ...")

            # Largest degree whose basis is identifiable with a 2x oversampling
            # margin (D3): basis_size(n, d) <= N_train / 2. The old max_order
            # returned the smallest degree with D >= N.
            max_deg_forward = max_supported_degree(self.n_params, X_train.shape[0])
            if max_degree_forward is None:
                max_degree_forward = max_deg_forward
                warnings.warn(
                    f"auto-capped max_degree_forward at {max_degree_forward} for n_params = "
                    f"{self.n_params}, N_train = {X_train.shape[0]} and fill factor 2 "
                    f"(basis_size <= N_train / 2); a higher degree needs more samples.",
                    UserWarning,
                    stacklevel=2,
                )
            elif max_degree_forward > max_deg_forward:
                D = (
                    basis_size(self.n_params, max_degree_forward)
                    if self.basis is None
                    else int(self.basis.build(self.parameter_names, max_degree_forward).shape[0])
                )
                if D >= X_train.shape[0]:
                    raise ValueError(
                        f"max_degree_forward = {max_degree_forward} needs a basis of D = {D} "
                        f"terms but N_train = {X_train.shape[0]}; the largest admissible degree "
                        f"is {max_deg_forward}. Add distinct samples or lower the degree."
                    )

            self.generate_forward_emulator(
                X_train,
                Y_train,
                X_val,
                Y_val,
                # RMSE_upper=RMSE_upper,
                RMSE_tol=RMSE_tol,
                fRMSE_tol=fRMSE_tol,
                init_deg=init_deg_forward,
                max_degree=max_degree_forward,
                dim_reduction=dim_reduction,
                per_mode_thres=per_mode_thres,
                batch_size=batch_size,
                loo=use_loo,
                weights=weights,
            )
            if return_max_frac_err:
                # Convert scaled validation data back to original scale for proper comparison
                X_val_unscaled, Y_val_unscaled = _unscale_val(
                    self.scaler_X, self.scaler_Y, X_val, Y_val, self.transform
                )

                Y_val_pred = self.forward_emulator(X_val_unscaled)
                diag = _report_frac_err("Forward", Y_val_pred, Y_val_unscaled)
                self.forward_frac_err_diag = diag
                # Coerce nan -> inf on the public attribute so threshold
                # comparisons fail loudly when the signal mask is empty, any
                # output column is fully masked, or max_rel is not finite.
                self.forward_max_frac_err = _public_max_frac_err(diag)

        if backward:
            logger.info("Generating backward emulator ...")
            # The backward basis is built from n_outputs, so the same D cap and a
            # memory budget apply here (n = m).
            max_deg_backward = max_supported_degree(self.n_outputs, X_train.shape[0])
            if max_degree_backward is None:
                max_degree_backward = max_deg_backward
                warnings.warn(
                    f"auto-capped max_degree_backward at {max_degree_backward} for n_outputs = "
                    f"{self.n_outputs}, N_train = {X_train.shape[0]} and fill factor 2 "
                    f"(basis_size <= N_train / 2); a higher degree needs more samples.",
                    UserWarning,
                    stacklevel=2,
                )
            elif max_degree_backward > max_deg_backward:
                D = basis_size(self.n_outputs, max_degree_backward)
                if D >= X_train.shape[0]:
                    raise ValueError(
                        f"max_degree_backward = {max_degree_backward} needs a basis of D = {D} "
                        f"terms but N_train = {X_train.shape[0]}; the largest admissible degree "
                        f"is {max_deg_backward}. Add distinct samples or lower the degree."
                    )

            if max_degree_backward > 0:
                D_back = basis_size(self.n_outputs, max_degree_backward)
                reduced_from = max_degree_backward
                while max_degree_backward > 0 and D_back * D_back * 8 > MAX_BACKWARD_MOMENT_BYTES:
                    max_degree_backward -= 1
                    D_back = basis_size(self.n_outputs, max_degree_backward)
                if max_degree_backward < reduced_from:
                    warnings.warn(
                        f"reduced max_degree_backward from {reduced_from} to "
                        f"{max_degree_backward} to keep the D x D moment matrix under "
                        f"{MAX_BACKWARD_MOMENT_BYTES / 2**30:.1f} GiB (n_outputs = {self.n_outputs}).",
                        UserWarning,
                        stacklevel=2,
                    )
                if D_back * D_back * 8 > MAX_BACKWARD_MOMENT_BYTES:
                    raise ValueError(
                        f"even degree 0 builds a D = {D_back} basis with n_outputs = "
                        f"{self.n_outputs}; the backward moment matrix would exceed "
                        f"{MAX_BACKWARD_MOMENT_BYTES / 2**30:.1f} GiB."
                    )

            self.generate_backward_emulator(X_train,
                                            Y_train,
                                            X_val,
                                            Y_val,
                                            # RMSE_upper=RMSE_upper,
                                            RMSE_tol=RMSE_tol,
                                            fRMSE_tol=fRMSE_tol,
                                            init_deg=init_deg_backward,
                                            max_degree=max_degree_backward,
                                            dim_reduction=dim_reduction,
                                            per_mode_thres=per_mode_thres,
                                            batch_size=batch_size,
                                            loo=use_loo,
                                            weights=weights)
            if return_max_frac_err:
                # Convert scaled validation data back to original scale for proper comparison
                X_val_unscaled, Y_val_unscaled = _unscale_val(
                    self.scaler_X, self.scaler_Y, X_val, Y_val, self.transform
                )

                X_val_pred = self.backward_emulator(Y_val_unscaled)
                diag = _report_frac_err("Backward", X_val_pred, X_val_unscaled)
                self.backward_frac_err_diag = diag
                self.backward_max_frac_err = _public_max_frac_err(diag)

    def generate_forward_emulator(self,
                                  X_train_scaled,
                                  Y_train_scaled,
                                  X_val_scaled,
                                  Y_val_scaled,
                                #   RMSE_upper=0.1,
                                  RMSE_tol=1e-3,
                                  fRMSE_tol=1e-1,
                                  init_deg=None,
                                  max_degree=None,
                                  dim_reduction=False,
                                  per_mode_thres=None,
                                  batch_size=10000,
                                  loo=False,
                                  weights=None):
        """Fit the forward degree sweep (internal; normally called by the constructor)."""

        if init_deg is None:
            if self.n_params > 6:
                init_deg = 1
            elif self.n_params < 3:
                init_deg = 3
            else:
                init_deg = 2
            # A Basis degree caps max_degree, and the heuristic above can start
            # above a low cap (Basis(degree=0) with n_params > 6). The cap is a
            # deliberate user constraint, so the default start yields to it; an
            # explicit init_deg_forward above the cap is rejected in __init__.
            _bd = None if self.basis is None else getattr(self.basis, "degree", None)
            if _bd is not None:
                init_deg = min(init_deg, _bd)

        check_degree_range(
            init_deg,
            max_degree,
            direction="forward",
            n_train=X_train_scaled.shape[0],
            n_params=self.n_params,
        )

        RMSE_val_list = []
        AIC_list = []
        BIC_list = []
        coeffs_list = []
        multi_indices_list = []
        running_time_list = []
        degree_list = []
        cond_list = []
        loo_per_output_list = []
        leverage_list = []
        moment_list = []
        import time

        # D16: a k-level axis identifies x_i^d only for d <= k - 1, so drop
        # multi-indices above the per-axis cap before building Phi. The capped
        # set stays downward closed; the total degree keeps rising on it.
        axis_caps, level_counts = check_axis_levels(X_train_scaled)
        axis_warned = False
        n_distinct = None
        D_prev = -1
        validation_is_training = (not loo) and (X_val_scaled is X_train_scaled)
        if validation_is_training:
            warnings.warn(
                "the validation set is the training set (cross_validation=False "
                "without an explicit X_test/Y_test); the in-sample RMSE understates "
                "the true error and the RMSE_tol early stop is disabled.",
                UserWarning,
                stacklevel=2,
            )
        stop_reason = "reached max_degree"
        multi_indices = None
        # Scale-free RMSE_tol (P1.4): compare against the RMS of the
        # validation Y in the same space, so rescaling the data with
        # with_std=False does not change the selected degree.
        _ref_Y = Y_train_scaled if loo else Y_val_scaled
        _y_rms = float(np.sqrt(np.mean(np.asarray(_ref_Y) ** 2)))
        rmse_ref = _y_rms if _y_rms > 0 else 1.0

        # P5.1: grow the moment system by bordering the previous degree block
        # instead of rebuilding Phi, M and nu from scratch at every rung. The
        # index sets are nested, so the previous rows are a prefix of the new
        # set and the Gram matrix gains a border.
        _N_train = X_train_scaled.shape[0]
        if weights is None:
            _w_norm = None
        else:
            _w_norm = np.asarray(weights, dtype=np.float64).reshape(-1)
            if _w_norm.shape[0] != _N_train:
                raise ValueError(f"weights has {_w_norm.shape[0]} entries, expected {_N_train}")
            if np.any(_w_norm < 0):
                raise ValueError("weights must be non-negative")
            _w_norm = _w_norm / _w_norm.mean()
            if np.all(_w_norm == 1.0):
                _w_norm = None  # P5.6: uniform weights == unweighted, bit for bit
        _inc: dict[str, Any] = {
            "indices": None,
            "phi": None,
            "phiw": None,
            "buf": None,
            "cap": 0,
            "M": None,
            "nu": None,
        }
        _used_incremental = False
        # Decide once whether the LOO path may keep the full Phi resident.
        _D_est = (
            int(self.basis.build(self.parameter_names, max_degree).shape[0])
            if self.basis is not None
            else basis_size(self.n_params, max_degree)
        )
        _use_batched_loo = bool(
            loo
            and batch_size < _N_train
            and _N_train * _D_est * 8 > PHI_BUDGET_BYTES
        )


        def _moments(idx):
            """Moment products for idx, bordering the previous rung when possible."""
            nonlocal _used_incremental
            if not loo:
                # P5.2/B1-memory: the held-out path must keep the batched build,
                # which never materialises the full Phi, so batch_size bounds
                # the peak. The LOO path needs full Phi for PRESS anyway.
                M_b, nu_b = compute_moments_vector_output_batched(
                    X_train_scaled, Y_train_scaled, idx,
                    batch_size=batch_size, weights=weights,
                    plan_cls=self._plan_cls,
                )
                return M_b, nu_b, None
            prev = _inc["indices"]
            if (
                prev is not None
                and prev.shape[0] <= idx.shape[0]
                and np.array_equal(idx[: prev.shape[0]], prev)
            ):
                new = idx[prev.shape[0]:]
                if new.shape[0]:
                    Phi_new = self._plan_cls.build(new).evaluate(X_train_scaled)
                else:
                    Phi_new = np.empty((_N_train, 0))
                # Grow one buffer in place instead of hstacking (which holds two
                # full N x D arrays at the largest rung).
                prev_D = prev.shape[0]
                new_D = prev_D + new.shape[0]
                if _inc["buf"] is None or new_D > int(_inc["cap"]):
                    cap = max(new_D, int(int(_inc["cap"]) * 1.5) + 1, 64)
                    buf = np.empty((_N_train, cap))
                    if prev_D:
                        buf[:, :prev_D] = _inc["phi"]
                    _inc["buf"] = buf
                    _inc["cap"] = cap
                else:
                    buf = _inc["buf"]
                if new.shape[0]:
                    buf[:, prev_D:new_D] = Phi_new
                Phi = buf[:, :new_D]
                if _w_norm is None:
                    cross = Phi[:, :prev_D].T @ Phi_new
                    M = np.block([
                        [_inc["M"], cross / _N_train],
                        [cross.T / _N_train, (Phi_new.T @ Phi_new) / _N_train],
                    ])
                    nu = np.vstack([_inc["nu"], (Phi_new.T @ Y_train_scaled) / _N_train])
                else:
                    PhiW_new = Phi_new * _w_norm[:, None]
                    cross = _inc["phiw"].T @ Phi_new
                    M = np.block([
                        [_inc["M"], cross / _N_train],
                        [cross.T / _N_train, (PhiW_new.T @ Phi_new) / _N_train],
                    ])
                    nu = np.vstack(
                        [_inc["nu"], (PhiW_new.T @ Y_train_scaled) / _N_train]
                    )
                _used_incremental = True
            else:
                M, nu = compute_moments_vector_output_batched(
                    X_train_scaled, Y_train_scaled, idx,
                    batch_size=batch_size, weights=weights,
                    plan_cls=self._plan_cls,
                )
                Phi = self._plan_cls.build(idx).evaluate(X_train_scaled)
            _inc["indices"] = idx
            _inc["phi"] = Phi
            _inc["phiw"] = None if _w_norm is None else Phi * _w_norm[:, None]
            _inc["M"] = M
            _inc["nu"] = nu
            return M, nu, Phi

        for d in range(init_deg, max_degree + 1):
            start_time = time.time()

            if self.basis is not None:
                raw_indices = self.basis.build(self.parameter_names, d)
            elif d == init_deg:
                raw_indices = generate_multi_indices(self.n_params, d)
            else:
                aux_indices = given_order_indices(self.n_params, d)
                assert multi_indices is not None
                raw_indices = np.concatenate((multi_indices, aux_indices), axis=0)

            candidate_indices = indices_selection(raw_indices, axis_caps)
            if not axis_warned and candidate_indices.shape[0] < raw_indices.shape[0]:
                axis_warned = True
                warnings.warn(
                    f"grid design detected: per-axis level counts {level_counts.tolist()} "
                    f"cap each parameter power at {axis_caps.tolist()} (D16); "
                    f"{raw_indices.shape[0] - candidate_indices.shape[0]} monomial(s) "
                    f"dropped at degree {d}.",
                    UserWarning,
                    stacklevel=2,
                )
            if candidate_indices.shape[0] == D_prev:
                stop_reason = "capped basis stopped growing"
                break
            D_prev = candidate_indices.shape[0]
            multi_indices = candidate_indices
            D = multi_indices.shape[0]

            check_sample_count(
                X_train_scaled.shape[0], D, degree=d, n_params=self.n_params
            )
            if n_distinct is None:
                n_distinct = count_distinct_rows(X_train_scaled)
            if n_distinct <= D:
                raise InsufficientSamplesError(
                    f"degree {d}: only {n_distinct} distinct input rows for a basis of "
                    f"D = {D} terms (N = {X_train_scaled.shape[0]} rows); the polynomial "
                    f"is unconstrained between the distinct points. Add distinct samples "
                    f"or lower the degree."
                )

            _qr_used = False
            if loo:
                # Fit on all N and score by exact leave-one-out PRESS from the
                # same Cholesky factor (P1.5). The metric in RMSE_val_list is
                # the LOO-RMSE, so selection is uniform.
                if not _use_batched_loo:
                    # The full Phi fits the budget (or the caller asked for a
                    # single batch): use the fast incremental path.
                    M, nu, Phi = _moments(multi_indices)
                    _plan_loo = None
                else:
                    # Default: batch_size bounds the peak; no full Phi is held.
                    M, nu = compute_moments_vector_output_batched(
                        X_train_scaled, Y_train_scaled, multi_indices,
                        batch_size=batch_size, weights=weights,
                        plan_cls=self._plan_cls,
                    )
                    Phi = None
                    _plan_loo = self._plan_cls.build(multi_indices)
                coeffs, cond, loo_rmse, loo_per_out, lev_max = press_loo(
                    M, nu, Phi, Y_train_scaled, on_singular="warn", degree=d,
                    weights=weights, plan=_plan_loo, ridge=self.ridge,
                    X_scaled=None if Phi is not None else X_train_scaled,
                    batch_size=batch_size,
                )
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                RMSE_val_list.append(loo_rmse)
                loo_per_output_list.append(loo_per_out)
                leverage_list.append(lev_max)
                metric = loo_rmse
            else:
                M, nu, Phi = _moments(multi_indices)
                coeffs, cond = solve_emulator_coefficients(
                    M, nu, on_singular="warn", degree=d, return_cond=True,
                    ridge=self.ridge,
                )
                if self.ridge == 0.0 and cond >= COND_QR:
                    # P5.4: refit this rung with CholeskyQR2 / Householder QR.
                    _Phi_qr = self._plan_cls.build(multi_indices).evaluate(X_train_scaled)
                    _c_qr, _cond_qr = solve_emulator_coefficients(
                        M, nu, on_singular="warn", degree=d, return_cond=True,
                        Phi=_Phi_qr, Y=Y_train_scaled,
                    )
                    if np.isfinite(_c_qr).all():
                        coeffs, _qr_used = _c_qr, True
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                Y_val_pred = evaluate_emulator_batched(
                    X_val_scaled, coeffs, multi_indices, batch_size=batch_size
                )
                RMSE_val, AIC, BIC = predictive_rmse_aic_bic(
                    Y_val_scaled, Y_val_pred, multi_indices.shape[0],
                    n_train=X_train_scaled.shape[0],
                )
                RMSE_val_list.append(RMSE_val)
                AIC_list.append(AIC)
                BIC_list.append(BIC)
                metric = RMSE_val

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            running_time_list.append(time.time() - start_time)
            moment_list.append(M)

            if cond >= COND_RAISE and not _qr_used and (loo or validation_is_training):
                stop_reason = "cond(M) >= 1e16"
                break
            if check_sweep_rmse(RMSE_val_list, degree_list):
                stop_reason = "RMSE blow-up"
                break
            if (not validation_is_training) and metric < RMSE_tol * rmse_ref:
                stop_reason = "RMSE_tol met"
                break

        if not RMSE_val_list:
            raise IllConditionedError(
                f"no forward degree in [{init_deg}, {max_degree}] could be fitted; "
                f"the last stop reason was {stop_reason!r}."
            )
        if stop_reason == "reached max_degree":
            # AIC/BIC are no longer used for selection (P1.5): choose the
            # simplest model within fRMSE_tol of the best LOO/held-out RMSE.
            ind = select_best_model(RMSE_val_list, rmse_tol=fRMSE_tol)
        else:
            # Early stop (RMSE_tol, blow-up, singular or a frozen capped basis):
            # keep the lowest finite-RMSE rung fitted so far.
            finite = np.isfinite(np.asarray(RMSE_val_list, dtype=np.float64))
            masked = np.where(finite, RMSE_val_list, np.inf)
            ind = int(np.argmin(masked))
        coeffs = coeffs_list[ind]
        multi_indices = multi_indices_list[ind]
        self.forward_degree = degree_list[ind]
        self.forward_cond_est_ = cond_list[ind]
        if loo:
            self.loo_rmse_ = float(RMSE_val_list[ind])
            self.leverage_max_train_ = float(leverage_list[ind])
            self.forward_RMSE_per_output_ = np.asarray(loo_per_output_list[ind])
        else:
            _sel_pred = evaluate_emulator_batched(
                X_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )
            self.forward_RMSE_per_output_ = np.sqrt(
                np.mean((_sel_pred - Y_val_scaled) ** 2, axis=0)
            )
        self._X_train_scaled_ = X_train_scaled
        self._Y_train_scaled_ = Y_train_scaled

        self.forward_coeffs = coeffs
        self.forward_multi_indices = multi_indices
        self.forward_RMSE_list = RMSE_val_list
        self.forward_AIC_list = AIC_list
        self.forward_BIC_list = BIC_list
        self.forward_running_time_list = running_time_list
        self.forward_degree_list = degree_list
        self.forward_sweep_incremental_ = bool(_used_incremental)
        # Metrics of the model actually stored (the old dim_reduction path
        # reported the pre-pruning model).
        self.forward_RMSE = float(RMSE_val_list[ind])
        self.forward_AIC = float(AIC_list[ind]) if AIC_list else float("nan")
        self.forward_BIC = float(BIC_list[ind]) if BIC_list else float("nan")
        # Cholesky factor of the selected moment matrix, kept for leverage().
        # moment_list holds the D x D matrix of every rung, so the default
        # (batched) path never materialises the full N x D Phi here either.
        _M = moment_list[ind]
        self.forward_moment_matrix_ = _M
        self.forward_chol_ = _safe_cholesky(_M)
        self.forward_N_train_ = int(X_train_scaled.shape[0])
        # Per-output residual standard deviation in the fitted (standardized Y)
        # space, used by the noise-only predictive band (P3.5). Accumulate the
        # residual sum of squares in batches when no full Phi is held.
        _dof = max(self.forward_N_train_ - multi_indices.shape[0], 1)
        if (
            loo
            and _inc.get("phi") is not None
            and _inc.get("indices") is not None
            and np.array_equal(np.asarray(multi_indices), _inc["indices"])
        ):
            _rss = np.sum((Y_train_scaled - _inc["phi"] @ coeffs) ** 2, axis=0)
        else:
            _rss = np.zeros(self.n_outputs)
            _final_plan = self._plan_cls.build(multi_indices)
            for _s in range(0, self.forward_N_train_, batch_size):
                _e = min(_s + batch_size, self.forward_N_train_)
                _rb = Y_train_scaled[_s:_e] - _final_plan.evaluate(
                    X_train_scaled[_s:_e]
                ) @ coeffs
                _rss += np.sum(_rb ** 2, axis=0)
        self.forward_resid_std_ = np.sqrt(_rss / _dof)
        self._build_forward_plan()

    @property
    def _plan_cls(self):
        """Basis plan class; a pickle from before basis_kind was added has none."""
        return BASIS_PLANS[getattr(self, "basis_kind", "monomial")]

    def _build_forward_plan(self):
        """Build the P0.7 monomial plan and folded output affine for inference."""
        mi = np.asarray(self.forward_multi_indices)
        self.forward_plan = self._plan_cls.build(mi)
        scale_Y = self.scaler_Y.scale_
        if scale_Y is None:
            scale_Y = np.ones(self.n_outputs)
        self.forward_coeffs_folded = fold_output_affine(
            self.forward_coeffs, mi, self.scaler_Y.mean_, scale_Y
        )

    def _build_backward_plan(self):
        """Build the P0.7 plan and folded output affine for the backward map."""
        mi = np.asarray(self.backward_multi_indices)
        self.backward_plan = MonomialPlan.build(mi)
        scale_X = self.scaler_X.scale_
        if scale_X is None:
            scale_X = np.ones(self.n_params)
        self.backward_coeffs_folded = fold_output_affine(
            self.backward_coeffs, mi, self.scaler_X.mean_, scale_X
        )

    def in_domain(self, X):
        """True when each row of X lies inside the stored training box (P1.6).

        A 1-D input of length n_params returns a single bool; an (N, n_params)
        input returns an (N,) bool array. No warning is raised here.
        """
        arr = np.asarray(X, dtype=np.float64)
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)
        inside = np.all((arr >= self.X_box_.lo) & (arr <= self.X_box_.hi), axis=1)
        return bool(inside[0]) if single else inside

    def leverage(self, X):
        """Leverage h(x) = ||L^{-1} phi(x)||^2 / N of the stored forward model.

        L is the Cholesky factor of the training moment matrix on the selected
        basis, so h(x) is the diagonal of the hat matrix at arbitrary x. The
        training rows have sum(h) = D.
        """
        from scipy.linalg import solve_triangular

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if not hasattr(self, "_inv_scale_X"):
            self._inv_scale_X = 1.0 / self.scaler_X.scale_
        Xs = (arr - self.scaler_X.mean_) * self._inv_scale_X
        Phi = self.forward_plan.evaluate(Xs)
        if getattr(self, "forward_chol_", None) is not None:
            cf, lower = self.forward_chol_
            A = solve_triangular(cf, Phi.T, lower=lower, trans="T", check_finite=False)
            h = np.einsum("ij,ij->j", A, A) / self.forward_N_train_
        else:
            Minv = np.linalg.pinv(self.forward_moment_matrix_)
            h = np.einsum("ij,jk,ik->i", Phi, Minv, Phi) / self.forward_N_train_
        return float(h[0]) if h.size == 1 else h

    def hat_diagonal(self, X):
        """Alias for :meth:`leverage`: the hat-matrix diagonal at X (P3.5)."""
        return self.leverage(X)

    def jacobian(self, X, batch_size=None):
        """Analytic dY/dX of the forward emulator (P2.2).

        Returns (N, m, n) for an (N, n) input, or (m, n) for a single point.
        d phi_alpha / d z_i = alpha_i * z^(alpha - e_i), so all n derivative
        blocks come from one closure build and one batched GEMM with the
        folded coefficients.
        """
        arr = np.asarray(X, dtype=np.float64)
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)
        if hasattr(self, "X_box_"):
            self._check_box(arr, self.X_box_, "ignore", "input")
        if not hasattr(self, "_inv_scale_X"):
            self._inv_scale_X = 1.0 / self.scaler_X.scale_
        if getattr(self, "forward_plan", None) is None:
            self._build_forward_plan()
        Xs = (arr - self.scaler_X.mean_) * self._inv_scale_X
        dPhi = self.forward_plan.evaluate_derivatives(Xs)  # (n, N, D)
        J = dPhi @ self.forward_coeffs_folded               # (n, N, m)
        J = J.transpose(1, 2, 0) / self.scaler_X.scale_[None, None, :]
        Y_affine = self.forward_plan.evaluate(Xs) @ self.forward_coeffs_folded
        for j, t in enumerate(self._transforms()):
            if t == "linear":
                continue
            if t == "log":
                factor = np.exp(Y_affine[:, j])
            elif t == "asinh":
                factor = np.cosh(Y_affine[:, j])
            else:
                # Callable (forward, inverse) pair: differentiate the inverse.
                lo = Y_affine[:, j] - 1e-6
                hi = Y_affine[:, j] + 1e-6
                factor = (t[1](hi) - t[1](lo)) / 2e-6
            J[:, j, :] = J[:, j, :] * factor[:, None]
        return J[0] if single else J

    def validate(self, X_val, Y_val, sigma=None, cov=None, *, extrapolation="warn"):
        """Error metrics in units of the user data covariance (P3.1, D10).

        Exactly one of ``sigma`` (per-entry standard deviations, shape (m,) or
        (N, m)) or ``cov`` (shape (m, m) shared, or (N, m, m) per point) is
        required: there is no neutral default covariance. Returns median and
        p99 Delta-chi2 = r^T C^-1 r, the per-output RMS error in sigma, and the
        maximum |r| / sigma.
        """
        if sigma is None and cov is None:
            raise ValueError(
                "validate requires sigma or cov (D10): there is no neutral "
                "default covariance for a polynomial residual."
            )
        if sigma is not None and cov is not None:
            raise ValueError("give either sigma or cov, not both")
        X_val = np.asarray(X_val, dtype=np.float64)
        Y_val = np.asarray(Y_val, dtype=np.float64)
        if Y_val.ndim != 2:
            raise ValueError(f"Y_val must be 2-D, got shape {Y_val.shape}")
        pred = self.forward_emulator(X_val, extrapolation=extrapolation)
        r = pred - Y_val
        N, m = r.shape
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            from scipy.linalg import solve_triangular

            if cov.ndim == 2:
                L = np.linalg.cholesky(cov)
                z = solve_triangular(L, r.T, lower=True, check_finite=False)
                dchi2 = np.einsum("ij,ij->j", z, z)
                sigma_eff = np.sqrt(np.diag(cov))[None, :] * np.ones((N, 1))
            elif cov.ndim == 3:
                L = np.linalg.cholesky(cov)
                z = np.linalg.solve(L, r[:, :, None])[:, :, 0]
                dchi2 = np.sum(z ** 2, axis=1)
                sigma_eff = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
            else:
                raise ValueError(f"cov must be (m, m) or (N, m, m), got {cov.shape}")
        else:
            sigma = np.asarray(sigma, dtype=np.float64)
            if sigma.ndim == 0:
                sigma_eff = np.full((N, m), float(sigma))
            elif sigma.ndim == 1:
                sigma_eff = np.broadcast_to(sigma[None, :], (N, m))
            elif sigma.ndim == 2:
                sigma_eff = sigma
            else:
                raise ValueError(f"sigma must be scalar, (m,) or (N, m), got {sigma.shape}")
            dchi2 = np.sum((r / sigma_eff) ** 2, axis=1)
        if np.any(sigma_eff <= 0):
            raise ValueError("sigma (or the diagonal of cov) must be positive")
        rms_over_sigma = np.sqrt(np.mean((r / sigma_eff) ** 2, axis=0))
        result = {
            "n": int(N),
            "m": int(m),
            "dchi2_median": float(np.median(dchi2)),
            "dchi2_p99": float(np.percentile(dchi2, 99)),
            "rms_over_sigma": rms_over_sigma,
            "max_abs_over_sigma": float(np.max(np.abs(r / sigma_eff))),
        }
        logger.info(
            "validate: n=%d median Delta-chi2=%.4g p99=%.4g max|r|/sigma=%.4g",
            result["n"], result["dchi2_median"], result["dchi2_p99"],
            result["max_abs_over_sigma"],
        )
        return result

    def _whiten(self, J, r, *, sigma=None, cov=None):
        """Whiten J (m, n) and r (m,) with C = diag(sigma^2) or cov (m, m)."""
        from scipy.linalg import solve_triangular

        if cov is not None:
            L = np.linalg.cholesky(np.asarray(cov, dtype=np.float64))
            # Jw = L^-1 J and rw = L^-1 r, so Jw^T Jw = J^T C^-1 J.
            Jw = solve_triangular(L, J, lower=True, check_finite=False)
            rw = solve_triangular(L, r, lower=True, check_finite=False)
        else:
            sig = np.asarray(sigma, dtype=np.float64)
            if sig.ndim == 0:
                sig = np.full(J.shape[0], float(sig))
            Jw = J / sig[:, None]
            rw = r / sig
        return Jw, rw

    def posterior_bias(self, theta, Y_true, *, sigma=None, cov=None):
        """Linearised posterior mean shift at a point (P3.2, mode b).

        Delta = -(J^T C^-1 J)^-1 J^T C^-1 (emu(theta) - Y_true), with J from
        :meth:`jacobian`.  Returns the parameter shift, the marginal shift in
        posterior-sigma units and the Mahalanobis norm sqrt(Delta^T F Delta),
        with F = J^T C^-1 J.  This is the exact GLS mean shift for a linear
        model and an offset residual.
        """
        if (sigma is None) == (cov is None):
            raise ValueError("posterior_bias requires exactly one of sigma or cov")
        theta = np.asarray(theta, dtype=np.float64)
        Y_true = np.asarray(Y_true, dtype=np.float64).ravel()
        J = np.atleast_2d(self.jacobian(theta))
        r = self.forward_emulator(theta, extrapolation="ignore").ravel() - Y_true
        Jw, rw = self._whiten(J, r, sigma=sigma, cov=cov)
        F = Jw.T @ Jw
        delta = -np.linalg.solve(F, Jw.T @ rw)
        cov_d = np.linalg.inv(F)
        sd = np.sqrt(np.diag(cov_d))
        mahalanobis = float(np.sqrt(delta @ F @ delta))
        return {
            "delta": delta,
            "marginal_sd": sd,
            "marginal_shift": delta / sd,
            "max_marginal_shift": float(np.max(np.abs(delta / sd))),
            "mahalanobis": mahalanobis,
        }

    def posterior_bias_map(self, X_val, Y_val, *, sigma=None, cov=None, threshold=0.1):
        """Linearised bias over validation points (P3.2, mode a).

        Y_val must be the true simulator values, so r is the emulator error.
        Returns the median, p99 and fraction above ``threshold`` of the
        per-point maximum marginal shift, plus the same for the Mahalanobis
        norm.
        """
        if (sigma is None) == (cov is None):
            raise ValueError("posterior_bias_map requires exactly one of sigma or cov")
        X_val = np.asarray(X_val, dtype=np.float64)
        Y_val = np.asarray(Y_val, dtype=np.float64)
        N = X_val.shape[0]
        Jv = np.stack([np.atleast_2d(self.jacobian(X_val[i])) for i in range(N)])
        rv = self.forward_emulator(X_val, extrapolation="ignore") - Y_val
        Jw_list, rw_list = [], []
        for i in range(N):
            Jw, rw = self._whiten(Jv[i], rv[i], sigma=sigma, cov=cov)
            Jw_list.append(Jw)
            rw_list.append(rw)
        Jw = np.stack(Jw_list)
        rw = np.stack(rw_list)
        F = np.einsum("nmi,nmj->nij", Jw, Jw)
        rhs = np.einsum("nmi,nm->ni", Jw, rw)
        delta = -np.linalg.solve(F, rhs[..., None])[..., 0]
        sd = np.sqrt(np.diagonal(np.linalg.inv(F), axis1=1, axis2=2))
        marginal = np.max(np.abs(delta / sd), axis=1)
        mahalanobis = np.sqrt(np.einsum("ni,nij,nj->n", delta, F, delta))
        return {
            "n": int(N),
            "marginal_median": float(np.median(marginal)),
            "marginal_p99": float(np.percentile(marginal, 99)),
            "marginal_frac_above": float(np.mean(marginal > threshold)),
            "mahalanobis_median": float(np.median(mahalanobis)),
            "mahalanobis_p99": float(np.percentile(mahalanobis, 99)),
        }

    def _check_box(self, X, box: DomainBox, extrapolation: str, label: str) -> None:
        if extrapolation not in ("warn", "raise", "ignore"):
            raise ValueError(
                f"extrapolation must be 'warn', 'raise' or 'ignore', got "
                f"{extrapolation!r}"
            )
        if extrapolation == "ignore":
            return
        X2 = np.atleast_2d(np.asarray(X, dtype=np.float64))
        dist = extrapolation_distance(X2, box)
        out = dist > 0
        if not out.any():
            return
        rng = box.hi - box.lo
        parts = []
        for j in np.where(out.any(axis=0))[0]:
            excess = float(
                (np.maximum(box.lo[j] - X2[:, j], 0.0) + np.maximum(X2[:, j] - box.hi[j], 0.0)).max()
            )
            pct = 100.0 * excess / rng[j] if rng[j] > 0 else float("inf")
            parts.append(
                f"parameter {j}: {int(out[:, j].sum())} of {X2.shape[0]} row(s) outside "
                f"[{box.lo[j]:.6g}, {box.hi[j]:.6g}], farthest {pct:.1f}% of the range beyond"
            )
        msg = f"{label} lies outside the training box; " + "; ".join(parts)
        if extrapolation == "raise":
            raise ValueError(msg)
        warnings.warn(msg, ExtrapolationWarning, stacklevel=3)

    def refit(self, X, Y, box=None, **kwargs):
        """Fit a new emulator on (X, Y), optionally restricted to a box (P3.4).

        ``box`` is a (lo, hi) pair of per-parameter arrays; only training rows
        inside it are used. This is the second stage of the zoom recipe: fit on
        the prior box, run a chain, then refit on the posterior mean +/- k sd.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if box is not None:
            lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
            mask = np.all((X >= lo) & (X <= hi), axis=1)
            if int(mask.sum()) < 2:
                raise ValueError(
                    f"only {int(mask.sum())} training row(s) fall inside the refit "
                    f"box; widen it or supply more samples."
                )
            X, Y = X[mask], Y[mask]
        return PolyEmu(X, Y, **kwargs)

    def save(self, path, *, float32=False, dataset_sha256=None):
        """Write this emulator to a versioned .npz (P4.2)."""
        from MomentEmu.io import save_emulator

        return save_emulator(
            self, path, float32=float32, dataset_sha256=dataset_sha256
        )

    @classmethod
    def load(cls, path):
        """Load an emulator from a versioned .npz without sklearn (P4.2)."""
        from MomentEmu.io import load_emulator

        return load_emulator(path)

    def fingerprint(self):
        """SHA-256 fingerprint of the stored coefficients (P4.2)."""
        from MomentEmu.io import fingerprint

        return fingerprint(self)

    def fit(self, X, Y, X_val=None, Y_val=None, **kwargs):
        """(Re)fit this emulator from data (P4.1).

        The constructor keyword arguments are reused and any keyword here
        overrides them; X_val/Y_val map to X_test/Y_test. Returns self.
        """
        merged = dict(getattr(self, '_hyperparameters', {}))
        merged.update(kwargs)
        if X_val is not None and 'X_test' not in merged:
            merged['X_test'] = X_val
        if Y_val is not None and 'Y_test' not in merged:
            merged['Y_test'] = Y_val
        self.__dict__.clear()
        type(self).__init__(self, X, Y, **merged)
        return self

    def predict(self, X, **kwargs):
        """Alias for forward_emulator (P4.1)."""
        return self.forward_emulator(X, **kwargs)

    def fit_inverse(self, **kwargs):
        """Fit the backward map on the data already supplied (P4.1)."""
        merged = dict(getattr(self, '_hyperparameters', {}))
        merged.update(kwargs)
        merged['forward'] = True
        merged['backward'] = True
        X_data, Y_data = self._X_data, self._Y_data
        self.__dict__.clear()
        type(self).__init__(self, X_data, Y_data, **merged)
        return self

    def predict_inverse(self, Y, **kwargs):
        """Alias for backward_emulator (P4.1)."""
        return self.backward_emulator(Y, **kwargs)

    def compress(self, rank, *, X_val=None, Y_val=None, gate=0.01):
        """Store a closed-form rank-``rank`` forward coefficient set (P5.7).

        The reduced-rank fit is exact (C_r = L^-T [L^-1 nu]_r). With a
        validation set the per-output difference from the full-rank
        predictions is stored, and a difference above ``gate`` times that
        output validation RMSE raises ValueError (D1).
        """
        from MomentEmu.storage import reduced_rank_coefficients

        mi = self.forward_multi_indices
        Phi = self._plan_cls.build(mi).evaluate(self._X_train_scaled_)
        N = Phi.shape[0]
        M = Phi.T @ Phi / N
        nu = Phi.T @ self._Y_train_scaled_ / N
        C_r, s = reduced_rank_coefficients(M, nu, rank)
        old = self.forward_coeffs.copy()
        if X_val is None or Y_val is None:
            # D1/B7: never apply a storage reduction ungated. Fall back to a
            # deterministic held-out slice of the training data.
            X_all = np.asarray(self._X_data, dtype=np.float64)
            Y_all = np.asarray(self._Y_data, dtype=np.float64)
            n_val = max(20, X_all.shape[0] // 5)
            X_val, Y_val = X_all[-n_val:], Y_all[-n_val:]
        else:
            X_val = np.asarray(X_val, dtype=np.float64)
            Y_val = np.asarray(Y_val, dtype=np.float64)
        full = self.forward_emulator(X_val, extrapolation="ignore")
        self.forward_coeffs = C_r
        self._build_forward_plan()
        new = self.forward_emulator(X_val, extrapolation="ignore")
        diff = np.max(np.abs(new - full), axis=0)
        val_rmse = np.sqrt(np.mean((full - Y_val) ** 2, axis=0))
        self.rank_difference_ = diff
        if np.any(diff > gate * val_rmse):
            self.forward_coeffs = old
            self._build_forward_plan()
            bad = np.flatnonzero(diff > gate * val_rmse).tolist()
            raise ValueError(
                f"rank {rank} changes a prediction by more than {gate:.0%} of the "
                f"validation RMSE for output(s) {bad}; use a higher rank."
            )
        self.forward_rank_ = int(rank)
        self.forward_singular_values_ = np.asarray(s)
        return self

    def report(self, variable_names=None, include_sobol=False, sobol_degree=None):
        """Print a per-parameter report of the stored forward basis (P5.3).

        Returns a dict with the term count, the retained per-parameter degree
        and a copy-pasteable basis spec. With include_sobol=True the
        Legendre-based Sobol report (P5.5) is attached under "sobol".
        """
        names = list(variable_names) if variable_names is not None else self.parameter_names
        mi = np.asarray(self.forward_multi_indices)
        per_param = mi.max(axis=0) if mi.size else np.zeros(self.n_params, dtype=int)
        spec = self.basis.spec() if self.basis is not None else (
            f"default total-degree basis, selected degree {self.forward_degree}"
        )
        deg_list: list[int] = [int(v) for v in per_param]
        info = {
            "n_terms": int(mi.shape[0]),
            "parameter_names": list(names),
            "per_parameter_degree": deg_list,
            "basis": spec,
        }
        logger.info("forward basis: %d terms; per-parameter degree %s; %s",
                    info["n_terms"], dict(zip(names, deg_list)), spec)
        if include_sobol:
            info["sobol"] = self.sobol_report(degree=sobol_degree)
        return info

    def _warn_nonuniform(self, X: Any) -> None:
        """Warn when a training column is not uniform on the box (P5.5)."""
        from scipy.stats import kstest

        lo = np.asarray(self.X_box_.lo, float)
        hi = np.asarray(self.X_box_.hi, float)
        for i in range(self.n_params):
            span = hi[i] - lo[i]
            if not np.isfinite(span) or span <= 0:
                continue
            u = (np.asarray(X)[:, i] - lo[i]) / span
            u = u[(u >= 0.0) & (u <= 1.0)]
            if u.size < 10:
                continue
            p = float(kstest(u, "uniform").pvalue)
            if p < 1e-3:
                warnings.warn(
                    f"parameter {self.parameter_names[i]} is not uniform on the training "
                    f"box (KS p={p:.2e}); the Sobol indices are uniform-box quantities",
                    UserWarning,
                    stacklevel=2,
                )

    def _legendre_terms(
        self,
        degree: int | None = None,
        X: Any | None = None,
        Y: Any | None = None,
        warn_uniform: bool = True,
        max_terms: int = 4096,
    ) -> _LegendreFit:
        """Project Y onto the orthonormal Legendre product basis over the box.

        Shared by sobol_report (P5.5) and interaction_graph (T-001) so both
        read the same decomposition. The basis is orthonormal for a uniform
        design, which makes every coefficient an ANOVA component: coef[r]**2
        is the variance carried by the interaction supp(mi[r]).
        """
        d = int(self.forward_degree if degree is None else degree)
        n = int(self.n_params)
        m = int(self.n_outputs)
        if X is None:
            X = self._X_data
        if Y is None:
            Y = self._Y_data
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if warn_uniform:
            self._warn_nonuniform(X)
        mi = generate_multi_indices(n, d)
        n_terms = int(mi.shape[0])
        if n_terms > max_terms:
            raise ValueError(
                f"Legendre degree {d} needs {n_terms} terms for n={n}; "
                "pass a lower degree or raise max_terms"
            )
        if X.shape[0] < n_terms:
            raise ValueError(
                f"need at least {n_terms} samples to fit the degree-{d} "
                f"Legendre basis, got {X.shape[0]}"
            )
        lo = np.asarray(self.X_box_.lo, float)
        hi = np.asarray(self.X_box_.hi, float)
        span = np.where(hi > lo, hi - lo, 1.0)
        v = 2.0 * (X - lo) / span - 1.0

        from scipy.linalg import cho_factor, cho_solve
        from scipy.special import eval_legendre

        basis = [
            np.stack(
                [np.sqrt(2.0 * a + 1.0) * eval_legendre(a, v[:, i]) for a in range(d + 1)],
                axis=1,
            )
            for i in range(n)
        ]
        M = np.zeros((n_terms, n_terms))
        nu = np.zeros((n_terms, m))
        chunk = max(1, min(X.shape[0], 4_000_000 // max(n_terms, 1)))
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            P = np.empty((e - s, n_terms))
            for r in range(n_terms):
                col = np.ones(e - s)
                for i in range(n):
                    col = col * basis[i][s:e, mi[r][i]]
                P[:, r] = col
            M += P.T @ P
            nu += P.T @ Y[s:e]
        coef = cho_solve(cho_factor(M), nu)

        var_terms = coef ** 2
        # The constant Legendre coefficient is E[f]; it carries no variance.
        nonzero = np.any(mi != 0, axis=1)
        explained = var_terms[nonzero].sum(axis=0)
        total_var = np.var(Y, axis=0)
        # Sobol shares use the model variance over the uniform box (the exact
        # Legendre decomposition), not the empirical sample variance.
        denom = np.where(explained > 0, explained, 1.0)
        safe_total = np.where(total_var > 0, total_var, 1.0)
        return _LegendreFit(
            degree=d,
            mi=mi,
            coef=coef,
            var_terms=var_terms,
            explained=explained,
            total_var=total_var,
            denom=denom,
            safe_total=safe_total,
            n_terms=n_terms,
            n_samples=int(X.shape[0]),
        )

    def sobol_report(
        self,
        degree: int | None = None,
        X: Any | None = None,
        Y: Any | None = None,
        warn_uniform: bool = True,
        max_terms: int = 4096,
    ) -> dict:
        """Legendre-based Sobol report on the training box (P5.5).

        The training outputs are projected onto an orthonormal Legendre basis
        over the training box by a Cholesky solve (cho_solve), which is well
        conditioned for this basis. Inputs are treated as uniform on the box; a
        non-uniform design raises a UserWarning because the indices are then
        box-uniform quantities. Returns S1, ST, the top pairwise shares and the
        variance broken down by per-parameter degree. The shares are normalised
        by the variance captured by the fit (they sum to one), while
        explained_fraction reports that variance as a fraction of Var(Y).
        """
        fit = self._legendre_terms(degree, X, Y, warn_uniform, max_terms)
        d, mi, n_terms = fit.degree, fit.mi, fit.n_terms
        n, m = int(self.n_params), int(self.n_outputs)
        var_terms, explained, total_var = fit.var_terms, fit.explained, fit.total_var
        denom, safe_total = fit.denom, fit.safe_total
        S1 = np.zeros((n, m))
        ST = np.zeros((n, m))
        var_by_degree = np.zeros((n, d + 1, m))
        pair_var: dict = {}
        for r in range(mi.shape[0]):
            alpha = mi[r]
            support = np.nonzero(alpha)[0]
            if support.size == 0:
                continue
            contrib = var_terms[r]
            for i in support:
                ST[i] = ST[i] + contrib
                var_by_degree[i, alpha[i]] = var_by_degree[i, alpha[i]] + contrib
            if support.size == 1:
                S1[support[0]] = S1[support[0]] + contrib
            elif support.size == 2:
                key = (int(support[0]), int(support[1]))
                pair_var[key] = pair_var.get(key, np.zeros(m)) + contrib

        pair_keys = sorted(
            pair_var, key=lambda k: float(np.max(pair_var[k] / denom)), reverse=True
        )[:10]
        return {
            "parameter_names": list(self.parameter_names),
            "degree": d,
            "n_terms": n_terms,
            "n_samples": fit.n_samples,
            "S1": S1 / denom[None, :],
            "ST": ST / denom[None, :],
            "top_pairs": [(int(i), int(j), pair_var[(i, j)] / denom) for i, j in pair_keys],
            "variance_by_degree": var_by_degree / denom[None, None, :],
            "explained_variance": explained,
            "total_variance": total_var,
            "explained_fraction": explained / safe_total,
            # P5.5: the fraction of Var(Y) the orthonormal decomposition
            # accounts for; meaningful, not identically one.
            "shares_sum": explained / safe_total,
        }

    def interaction_graph(
        self,
        degree: int | None = None,
        X: Any | None = None,
        Y: Any | None = None,
        threshold: float = 1e-4,
        output: int | None = None,
        warn_uniform: bool = True,
        max_terms: int = 4096,
        min_explained: float = 0.9,
    ) -> dict:
        """Full pairwise interaction matrix and the blocks it implies (T-001).

        ``matrix[i, j]`` is the share of the explained variance carried by every
        Legendre term whose support contains both parameters, so a pure
        three-way coupling x_i x_j x_k registers on all three of its pairs.
        This is the quantity that decides whether two parameters may sit in
        different blocks of an additively separable model; ``sobol_report``
        instead reports only terms of support exactly two, and only the top ten
        of them, which hides a cross-block pair as soon as n grows.

        ``blocks`` are the connected components of ``matrix > threshold``. Feed
        them straight to :meth:`MomentEmu.basis.Basis.separable`.

        The projection can only see interactions it can represent: an even
        coupling such as x_i^2 x_j^2 needs ``degree >= 4``. A projection that
        explains less than ``min_explained`` of Var(Y) warns, because absence of
        interaction is not evidence when the fit itself is poor.

        Args:
            degree: Legendre degree; defaults to the fitted forward degree.
            X, Y: design and outputs; default to the training data.
            threshold: variance share above which a pair is called coupled.
                It is a VARIANCE share, so it is the square of the amplitude
                share: the 1e-4 default treats a coupling contributing less
                than 1% of the signal amplitude as absent. Measured on a
                9-parameter separable function at degree 4, the numerical
                noise floor was 3.4e-6 and a real 0.25 x0^2 x3^2 coupling read
                5.3e-4, so the default sits between them with margin.
            output: restrict to one output column; default aggregates by max.
            min_explained: warn below this explained fraction of Var(Y).

        Returns:
            dict with matrix (n, n), blocks, separable, threshold, degree,
            explained_variance, explained_fraction, n_samples,
            degenerate_outputs, parameter_names.
        """
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        n, m = int(self.n_params), int(self.n_outputs)
        if output is not None:
            output = int(output)
            if not 0 <= output < m:
                raise ValueError(f"output must be in [0, {m}), got {output}")
        fit = self._legendre_terms(degree, X, Y, warn_uniform, max_terms)

        degenerate = _degenerate_columns(fit)
        degenerate_outputs = tuple(int(i) for i in np.flatnonzero(degenerate))
        cols = self._usable_columns(degenerate, output, "blocks")
        shares = fit.var_terms[:, cols] / fit.denom[cols][None, :]
        matrix = np.zeros((n, n))
        for r in range(fit.n_terms):
            support = np.flatnonzero(fit.mi[r])
            if support.size < 2:
                continue
            contrib = float(np.max(shares[r]))
            for a in range(support.size):
                for b in range(a + 1, support.size):
                    i, j = int(support[a]), int(support[b])
                    matrix[i, j] += contrib
                    matrix[j, i] += contrib

        explained_fraction = fit.explained[cols] / fit.safe_total[cols]
        if float(np.min(explained_fraction)) < min_explained:
            warnings.warn(
                f"the degree-{fit.degree} Legendre projection explains only "
                f"{float(np.min(explained_fraction)):.3f} of Var(Y) (< {min_explained}); "
                "an interaction hidden in the unexplained variance would not appear "
                "in this graph, so raise the degree before trusting the blocks",
                UserWarning,
                stacklevel=2,
            )

        blocks = connected_components(matrix > threshold)

        return {
            "matrix": matrix,
            "blocks": blocks,
            "separable": len(blocks) > 1,
            "threshold": float(threshold),
            "degree": fit.degree,
            "explained_variance": fit.explained[cols],
            "explained_fraction": explained_fraction,
            "n_samples": fit.n_samples,
            "degenerate_outputs": degenerate_outputs,
            "parameter_names": list(self.parameter_names),
        }

    def _usable_columns(self, degenerate, output, what: str):
        """Output columns to read structure from; raise or warn on degenerate ones."""
        m = int(self.n_outputs)
        if output is not None:
            if degenerate[output]:
                raise ValueError(
                    f"output {output} is degenerate: its variation is at the "
                    "rounding level of its own magnitude, so no structure can "
                    "be read from it"
                )
            return np.array([int(output)])
        cols = np.flatnonzero(~degenerate)
        if cols.size == 0:
            raise ValueError(
                f"all {m} output(s) are degenerate: their variation is at the "
                "rounding level of their own magnitude, so no structure can be "
                "read from them"
            )
        if cols.size < m:
            names = tuple(int(i) for i in np.flatnonzero(degenerate))
            warnings.warn(
                f"ignoring {m - cols.size} degenerate output column(s) {names}: "
                "their variation is at the rounding level of their own "
                f"magnitude, and including them would corrupt the {what}",
                UserWarning,
                stacklevel=3,
            )
        return cols

    def degree_profile(
        self,
        degree: int | None = None,
        X: Any | None = None,
        Y: Any | None = None,
        tol: float = 1e-6,
        parity_tol: float = 1e-3,
        output: int | None = None,
        warn_uniform: bool = True,
        max_terms: int = 4096,
    ) -> dict:
        """Per-parameter maximum useful degree and parity (T-001).

        Reads the orthonormal Legendre decomposition to find, for each
        parameter, the highest power that carries variance and whether only
        even powers do. A target that depends on a parameter only through its
        square needs no odd powers of it at all, and a target that is exactly
        quadratic in a parameter needs no power above two; both are pure
        redundancy in the isotropic basis.

        Parity is reported as "even" or None only. Reporting "odd" would
        require the target to be globally odd in that parameter, so that no
        term omits it, which this decomposition cannot establish from the
        powers alone; ``Basis`` still accepts "odd" when the caller knows it.

        Redundant terms do not bias the fit -- the true coefficients are zero
        and least squares recovers that in the population limit -- so the gain
        is in sample count, memory and conditioning rather than in accuracy at
        a sample size that is already ample.

        Args:
            degree: Legendre degree; defaults to the fitted forward degree.
            tol: share of the explained variance below which a power counts as
                absent. It is a VARIANCE share, so the induced amplitude error
                is its square root: the 1e-6 default caps a parameter's degree
                only once the discarded powers are under 1e-3 in amplitude.
                Unlike a block boundary, where the cross terms are structurally
                zero, a degree cap truncates a convergent series, so raising
                this introduces real bias rather than just removing redundancy.
            parity_tol: a parameter is called "even" when its odd powers carry
                less than this fraction of the variance its even powers carry.
                This is deliberately relative, not absolute: the odd-power
                share of a genuinely even parameter is sampling noise, whose
                absolute size depends on N and would otherwise have to be
                guessed.
            output: restrict to one output; default aggregates by max.

        Returns:
            dict with max_degree, parity, basis (a ready-to-use Basis),
            variance_by_degree, degree, degenerate_outputs.
        """
        from MomentEmu.basis import Basis

        fit = self._legendre_terms(degree, X, Y, warn_uniform, max_terms)
        degenerate = _degenerate_columns(fit)
        cols = self._usable_columns(degenerate, output, "degree profile")
        shares = fit.var_terms[:, cols] / fit.denom[cols][None, :]

        n, d = int(self.n_params), fit.degree
        var_by_degree = np.zeros((n, d + 1))
        for r in range(fit.n_terms):
            alpha = fit.mi[r]
            contrib = float(np.max(shares[r]))
            for i in np.flatnonzero(alpha):
                var_by_degree[i, alpha[i]] += contrib

        max_degree, parity = [], []
        for i in range(n):
            active = [k for k in range(1, d + 1) if var_by_degree[i, k] > tol]
            max_degree.append(max(active) if active else 0)
            odd = float(sum(var_by_degree[i, k] for k in range(1, d + 1, 2)))
            even = float(sum(var_by_degree[i, k] for k in range(2, d + 1, 2)))
            parity.append("even" if even > 0.0 and odd < parity_tol * even else None)

        return {
            "max_degree": tuple(max_degree),
            "parity": tuple(parity),
            "basis": Basis(
                per_parameter=tuple(max_degree),
                parity=tuple(parity) if any(p is not None for p in parity) else None,
            ),
            "variance_by_degree": var_by_degree,
            "degree": d,
            "degenerate_outputs": tuple(int(i) for i in np.flatnonzero(degenerate)),
            "parameter_names": list(self.parameter_names),
        }

    def _transforms(self):
        """Per-output transform tuple; legacy pickles get the log_Y mapping."""
        t = getattr(self, "transform", None)
        if t is None:
            # B1: July-2025 pickles store neither transform nor log_Y; treat
            # missing log_Y as linear instead of raising AttributeError.
            log_y = bool(getattr(self, "log_Y", False))
            t = tuple("log" if log_y else "linear" for _ in range(self.n_outputs))
            self.transform = t
        return t

    def forward_emulator(self, X, batch_size=None, extrapolation="warn", return_std=False):
        """Evaluate the fitted forward emulator at X (see the class docstring)."""
        float_or_int = isinstance(X, (float, int))
        if isinstance(X, list):
            X = np.array(X)
        else:
            X = np.asarray(X)
        if X.dtype.kind not in "iuf":
            raise TypeError(f"X must be numeric, got dtype {X.dtype}")
        check_finite(X, "X")
        Xshape = X.shape
        # A 1-D input is one sample only when its length equals n_params; a
        # trailing-axis mismatch raises instead of silently misreading a batch.
        X, _single = resolve_batch_shape(X, self.n_params)
        # D1: predict in float64 even when the caller passes int/float32.
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self, "X_box_"):
            self._check_box(X, self.X_box_, extrapolation, "input")
        # Pickles from before 2.0 lack the cached inverses and the plan.
        if not hasattr(self, "_inv_scale_X"):
            self._inv_scale_X = 1.0 / self.scaler_X.scale_
        if getattr(self, "forward_plan", None) is None:
            self._build_forward_plan()
        X_scaled = (X - self.scaler_X.mean_) * self._inv_scale_X
        plan = self.forward_plan
        C_fold = self.forward_coeffs_folded
        if batch_size is None or batch_size >= X_scaled.shape[0]:
            Y_pred = plan.evaluate(X_scaled) @ C_fold
        else:
            Y_pred = np.concatenate(
                [
                    plan.evaluate(X_scaled[s:s + batch_size]) @ C_fold
                    for s in range(0, X_scaled.shape[0], batch_size)
                ],
                axis=0,
            )
        transform = self._transforms()
        std: Any = None
        if return_std:
            # Noise-only band: s_j sqrt(1 + h(x)) in physical units. Exact for
            # iid noise; it undercovers model error (P3.5).
            h = np.asarray(self.leverage(X), dtype=float).reshape(-1)
            s = np.asarray(self.forward_resid_std_, dtype=float)
            std = np.sqrt(1.0 + h)[:, None] * s[None, :]
        if return_std:
            # Chain rule from the model space (transformed, standardized Y) to
            # physical units: scale_Y * d(inverse)/dz per column transform.
            _sy = self.scaler_Y.scale_
            if _sy is None:
                _sy = np.ones(self.n_outputs)
            std = std * np.asarray(_sy)[None, :]
            Y_phys = _transform_inverse(Y_pred, transform)
            for j, t in enumerate(transform):
                if t == "log":
                    std[:, j] = std[:, j] * Y_phys[:, j]
                elif t == "asinh":
                    std[:, j] = std[:, j] * np.sqrt(1.0 + Y_phys[:, j] ** 2)
            Y_pred = Y_phys
        else:
            Y_pred = _transform_inverse(Y_pred, transform)
        if float_or_int:
            Y_pred = Y_pred[0]
            if return_std:
                std = std[0]
            if self.n_outputs == 1:
                Y_pred = Y_pred[0]
                if return_std:
                    std = std[0]
        else:
            Y_pred = Y_pred.reshape(Xshape[:-1] + (self.n_outputs,))
            if return_std:
                std = std.reshape(Xshape[:-1] + (self.n_outputs,))
        return (Y_pred, std) if return_std else Y_pred

    def generate_backward_emulator(self,
                                   X_train_scaled,
                                   Y_train_scaled,
                                   X_val_scaled,
                                   Y_val_scaled,
                                #    RMSE_upper=0.1,
                                   RMSE_tol=1e-2,
                                   fRMSE_tol=1e-1,
                                   init_deg=None,
                                   max_degree=None,
                                   dim_reduction=False,
                                   per_mode_thres=None,
                                   batch_size=10000,
                                   loo=False,
                                   weights=None):
        """Fit the backward degree sweep (internal; normally called by the constructor)."""
        if init_deg is None:
            if self.n_outputs > 6:
                init_deg = 1
            elif self.n_outputs < 3:
                init_deg = 3
            else:
                init_deg = 2

        check_degree_range(
            init_deg,
            max_degree,
            direction="backward",
            n_train=Y_train_scaled.shape[0],
            n_params=self.n_outputs,
        )

        coeffs_list = []
        RMSE_val_list = []
        AIC_list = []
        BIC_list = []
        multi_indices_list = []
        running_time_list = []
        degree_list = []
        cond_list = []
        loo_per_output_list = []
        leverage_list = []
        import time

        # Same D16 axis cap and sample guards as the forward sweep, with
        # n = n_outputs because the backward basis is built from Y.
        axis_caps, level_counts = check_axis_levels(Y_train_scaled)
        axis_warned = False
        n_distinct = None
        D_prev = -1
        validation_is_training = (not loo) and (Y_val_scaled is Y_train_scaled)
        if validation_is_training:
            warnings.warn(
                "the backward validation set is the training set "
                "(cross_validation=False without an explicit X_test/Y_test); the "
                "in-sample RMSE understates the true error and the RMSE_tol early "
                "stop is disabled.",
                UserWarning,
                stacklevel=2,
            )
        stop_reason = "reached max_degree"
        multi_indices = None
        _ref_X = X_train_scaled if loo else X_val_scaled
        _x_rms = float(np.sqrt(np.mean(np.asarray(_ref_X) ** 2)))
        rmse_ref = _x_rms if _x_rms > 0 else 1.0

        for d in range(init_deg, max_degree + 1):
            start_time = time.time()
            # No basis= branch here, unlike the forward sweep: a Basis is
            # specified over the n_params inputs, while the backward map is
            # over the n_outputs outputs, so its index set has the wrong
            # arity. The constructor rejects basis= with backward=True.
            if d == init_deg:
                raw_indices = generate_multi_indices(self.n_outputs, d)
            else:
                aux_indices = given_order_indices(self.n_outputs, d)
                assert multi_indices is not None
                raw_indices = np.concatenate((multi_indices, aux_indices), axis=0)

            candidate_indices = indices_selection(raw_indices, axis_caps)
            if not axis_warned and candidate_indices.shape[0] < raw_indices.shape[0]:
                axis_warned = True
                warnings.warn(
                    f"grid design detected: per-axis level counts {level_counts.tolist()} "
                    f"cap each parameter power at {axis_caps.tolist()} (D16); "
                    f"{raw_indices.shape[0] - candidate_indices.shape[0]} monomial(s) "
                    f"dropped at degree {d}.",
                    UserWarning,
                    stacklevel=2,
                )
            if candidate_indices.shape[0] == D_prev:
                stop_reason = "capped basis stopped growing"
                break
            D_prev = candidate_indices.shape[0]
            multi_indices = candidate_indices
            D = multi_indices.shape[0]

            check_sample_count(
                Y_train_scaled.shape[0], D, degree=d, n_params=self.n_outputs
            )
            if n_distinct is None:
                n_distinct = count_distinct_rows(Y_train_scaled)
            if n_distinct <= D:
                raise InsufficientSamplesError(
                    f"degree {d}: only {n_distinct} distinct output rows for a basis of "
                    f"D = {D} terms (N = {Y_train_scaled.shape[0]} rows); the polynomial "
                    f"is unconstrained between the distinct points. Add distinct samples "
                    f"or lower the degree."
                )

            _qr_used = False
            if loo:
                _plan = MonomialPlan.build(multi_indices)
                Phi = _plan.evaluate(Y_train_scaled)
                M, nu = generate_moment_products(
                    Phi, X_train_scaled, weights=weights
                )
                coeffs, cond, loo_rmse, loo_per_out, lev_max = press_loo(
                    M, nu, Phi, X_train_scaled, on_singular="warn", degree=d,
                    weights=weights,
                )
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                RMSE_val_list.append(loo_rmse)
                loo_per_output_list.append(loo_per_out)
                leverage_list.append(lev_max)
                metric = loo_rmse
            else:
                M, nu = compute_moments_vector_output_batched(
                    Y_train_scaled, X_train_scaled, multi_indices,
                    batch_size=batch_size, weights=weights,
                )
                coeffs, cond = solve_emulator_coefficients(
                    M, nu, on_singular="warn", degree=d, return_cond=True
                )
                if cond >= COND_QR:
                    _Phi_qr = MonomialPlan.build(multi_indices).evaluate(Y_train_scaled)
                    _c_qr, _cond_qr = solve_emulator_coefficients(
                        M, nu, on_singular="warn", degree=d, return_cond=True,
                        Phi=_Phi_qr, Y=X_train_scaled,
                    )
                    if np.isfinite(_c_qr).all():
                        coeffs, _qr_used = _c_qr, True
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                X_val_pred = evaluate_emulator_batched(
                    Y_val_scaled, coeffs, multi_indices, batch_size=batch_size
                )
                RMSE_val, AIC, BIC = predictive_rmse_aic_bic(
                    X_val_scaled, X_val_pred, multi_indices.shape[0],
                    n_train=Y_train_scaled.shape[0],
                )
                RMSE_val_list.append(RMSE_val)
                AIC_list.append(AIC)
                BIC_list.append(BIC)
                metric = RMSE_val

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            running_time_list.append(time.time() - start_time)

            if cond >= COND_RAISE and not _qr_used and (loo or validation_is_training):
                stop_reason = "cond(M) >= 1e16"
                break
            if check_sweep_rmse(RMSE_val_list, degree_list):
                stop_reason = "RMSE blow-up"
                break
            if (not validation_is_training) and metric < RMSE_tol * rmse_ref:
                stop_reason = "RMSE_tol met"
                break

        if not RMSE_val_list:
            raise IllConditionedError(
                f"no backward degree in [{init_deg}, {max_degree}] could be fitted; "
                f"the last stop reason was {stop_reason!r}."
            )
        if stop_reason == "reached max_degree":
            ind = select_best_model(RMSE_val_list, rmse_tol=fRMSE_tol)
        else:
            finite = np.isfinite(np.asarray(RMSE_val_list, dtype=np.float64))
            masked = np.where(finite, RMSE_val_list, np.inf)
            ind = int(np.argmin(masked))
        coeffs = coeffs_list[ind]
        multi_indices = multi_indices_list[ind]
        self.backward_degree = degree_list[ind]
        self.backward_cond_est_ = cond_list[ind]
        if loo:
            self.backward_loo_rmse_ = float(RMSE_val_list[ind])
            self.backward_leverage_max_train_ = float(leverage_list[ind])
            self.backward_RMSE_per_output_ = np.asarray(loo_per_output_list[ind])
        else:
            _sel_pred = evaluate_emulator_batched(
                Y_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )
            self.backward_RMSE_per_output_ = np.sqrt(
                np.mean((_sel_pred - X_val_scaled) ** 2, axis=0)
            )

        self.backward_coeffs = coeffs
        self.backward_multi_indices = multi_indices
        self.backward_RMSE_list = RMSE_val_list
        self.backward_AIC_list = AIC_list
        self.backward_BIC_list = BIC_list
        self.backward_running_time_list = running_time_list
        self.backward_degree_list = degree_list
        self.backward_RMSE = float(RMSE_val_list[ind])
        self.backward_AIC = float(AIC_list[ind]) if AIC_list else float("nan")
        self.backward_BIC = float(BIC_list[ind]) if BIC_list else float("nan")

        _final_plan = MonomialPlan.build(multi_indices)
        _Phi = _final_plan.evaluate(Y_train_scaled)
        _M = _Phi.T @ _Phi / Y_train_scaled.shape[0]
        self.backward_moment_matrix_ = _M
        self.backward_chol_ = _safe_cholesky(_M)
        self.backward_N_train_ = int(Y_train_scaled.shape[0])
        self._build_backward_plan()

    def backward_emulator(self, Y, batch_size=None, extrapolation="warn"):
        """Evaluate the fitted backward emulator at Y."""
        float_or_int = isinstance(Y, (float, int))
        if isinstance(Y, list):
            Y = np.array(Y)
        else:
            Y = np.asarray(Y)
        if Y.dtype.kind not in "iuf":
            raise TypeError(f"Y must be numeric, got dtype {Y.dtype}")
        check_finite(Y, "Y")
        Yshape = Y.shape
        Y, _single = resolve_batch_shape(Y, self.n_outputs)
        if hasattr(self, "Y_box_"):
            self._check_box(Y, self.Y_box_, extrapolation, "output")
        if not hasattr(self, "_inv_scale_Y"):
            _s = self.scaler_Y.scale_
            self._inv_scale_Y = 1.0 / (_s if _s is not None else np.ones(self.n_outputs))
        Y = np.asarray(Y, dtype=np.float64)
        transform = self._transforms()
        for j, t in enumerate(transform):
            if t == "log":
                check_log_domain(Y[:, j:j + 1], f"Y column {j}")
        Y = _transform_forward(Y, transform)
        if getattr(self, "backward_plan", None) is None:
            self._build_backward_plan()
        Y_scaled = (Y - self.scaler_Y.mean_) * self._inv_scale_Y
        plan = self.backward_plan
        C_fold = self.backward_coeffs_folded
        if batch_size is None or batch_size >= Y_scaled.shape[0]:
            X_pred = plan.evaluate(Y_scaled) @ C_fold
        else:
            X_pred = np.concatenate(
                [
                    plan.evaluate(Y_scaled[s:s + batch_size]) @ C_fold
                    for s in range(0, Y_scaled.shape[0], batch_size)
                ],
                axis=0,
            )
        if float_or_int:
            X_pred = X_pred[0]
            if self.n_params == 1:
                X_pred = X_pred[0]
        else:
            X_pred = X_pred.reshape(Yshape[:-1] + (self.n_params,))
        return X_pred

    def backward_pca(
        self,
        rank: int = 8,
        degree: int = 3,
        *,
        X: Any | None = None,
        Y: Any | None = None,
        X_val: Any | None = None,
        Y_val: Any | None = None,
        fill: float = 2.0,
    ) -> PolyEmu:
        """Fit X from the top-rank principal components of Y (P5.8).

        The transformed, standardized outputs are reduced to their top-rank
        principal components (whitened), and a total-degree polynomial is fit
        from those components to the standardized inputs. Returns E[X | Y] for
        non-injective maps and records per-parameter validation R^2 in
        backward_R2_per_parameter_, flagging parameters with R^2 <= 0 in
        backward_unconstrained_parameters_. Fails when the polynomial would
        exceed N_train / fill terms.
        """
        if int(rank) < 1:
            raise ValueError("rank must be >= 1")
        X = as_float64(np.asarray(self._X_data if X is None else X), "X")
        Y = as_float64(np.asarray(self._Y_data if Y is None else Y), "Y")
        transform = self._transforms()
        y_scale = self.scaler_Y.scale_
        if y_scale is None:
            y_scale = np.ones(self.n_outputs)
        Yt = _transform_forward(Y, transform)
        Ys = (Yt - self.scaler_Y.mean_) / y_scale
        Xs = self.scaler_X.transform(X)
        y_mean = Ys.mean(axis=0)
        Yc = Ys - y_mean
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        k = int(rank)
        if k > Vt.shape[0]:
            raise ValueError(f"rank {k} exceeds the number of Y components ({Vt.shape[0]})")
        if np.any(S[:k] <= 0.0):
            raise ValueError("cannot whiten a zero-variance principal component")
        root_n = float(np.sqrt(max(Yc.shape[0] - 1, 1)))
        scores = U[:, :k] * root_n
        mi = generate_multi_indices(k, int(degree))
        n_terms = int(mi.shape[0])
        check_sample_count(Ys.shape[0], n_terms, degree=int(degree), n_params=k, fill=fill)
        plan = MonomialPlan.build(mi)
        M, nu = generate_moment_products(plan.evaluate(scores), Xs)
        coeffs, cond = solve_emulator_coefficients(
            M, nu, on_singular="warn", degree=int(degree), return_cond=True
        )
        if not np.isfinite(coeffs).all():
            raise IllConditionedError("the backward PCA fit produced non-finite coefficients")
        if X_val is None or Y_val is None:
            Xv_s, Yv = Xs, Y
        else:
            Xv_s = self.scaler_X.transform(as_float64(np.asarray(X_val), "X_val"))
            Yv = as_float64(np.asarray(Y_val), "Y_val")
        Yv_s = (_transform_forward(Yv, transform) - self.scaler_Y.mean_) / y_scale
        scores_v = (Yv_s - y_mean) @ Vt[:k].T / S[:k] * root_n
        pred = plan.evaluate(scores_v) @ coeffs
        ss_res = np.sum((pred - Xv_s) ** 2, axis=0)
        ss_tot = np.sum((Xv_s - Xv_s.mean(axis=0)) ** 2, axis=0)
        r2 = 1.0 - ss_res / np.where(ss_tot > 0.0, ss_tot, 1.0)
        self.backward_pca_ = {
            "rank": k,
            "degree": int(degree),
            "n_terms": n_terms,
            "components": Vt[:k],
            "y_mean": y_mean,
            "score_scale": root_n / S[:k],
            "coeffs": coeffs,
            "multi_indices": mi,
            "cond": float(cond),
        }
        self.backward_pca_rank_ = k
        self.backward_pca_degree_ = int(degree)
        self.backward_pca_n_terms_ = n_terms
        self.backward_pca_cond_ = float(cond)
        self.backward_R2_per_parameter_ = np.asarray(r2)
        self.backward_unconstrained_parameters_ = [
            int(i) for i in np.nonzero(np.asarray(r2) <= 0.0)[0]
        ]
        return self

    def backward_pca_predict(self, Y: Any, batch_size: int | None = None) -> np.ndarray:
        """Evaluate the backward PCA fit at Y (E[X | Y]; P5.8)."""
        if getattr(self, "backward_pca_", None) is None:
            raise RuntimeError("call backward_pca() before backward_pca_predict()")
        info = self.backward_pca_
        Y = np.asarray(Y)
        single = Y.ndim == 1
        if single:
            Y = Y.reshape(1, -1)
        transform = self._transforms()
        y_scale = self.scaler_Y.scale_
        if y_scale is None:
            y_scale = np.ones(self.n_outputs)
        Ys = (_transform_forward(Y, transform) - self.scaler_Y.mean_) / y_scale
        scores = (Ys - info["y_mean"]) @ info["components"].T * info["score_scale"][None, :]
        plan = MonomialPlan.build(info["multi_indices"])
        Xs = plan.evaluate(scores) @ info["coeffs"]
        scale = self.scaler_X.scale_
        if scale is None:
            scale = np.ones(self.n_params)
        X_pred = Xs * scale[None, :] + self.scaler_X.mean_[None, :]
        return X_pred[0] if single else X_pred

    def generate_forward_symb_emu(self, variable_names=None, *, raw_units=False):
        """SymPy expressions for the forward emulator (P2.4, D11).

        Standardized z-form by default; log_Y wraps the result in exp.
        """
        _warn_export_conditioning(getattr(self, "forward_cond_est_", None), "forward")
        Y_var = np.ones(self.n_outputs) if not self.standardize_Y_with_std else self.scaler_Y.var_
        exprs = symbolic_polynomial_expressions(
            self.forward_coeffs,
            self.forward_multi_indices,
            variable_names=variable_names,
            input_means=self.scaler_X.mean_,
            input_vars=self.scaler_X.var_,
            output_means=self.scaler_Y.mean_,
            output_vars=Y_var,
            transform=self._transforms(),
            raw_units=raw_units,
            family=getattr(self, "basis_kind", "monomial"),
        )
        return exprs

    def generate_backward_symb_emu(self, variable_names=None, *, raw_units=False):
        """SymPy expressions for the backward emulator (P2.4, D11).

        z-form by default; log_Y substitutes log of the input variables.
        """
        _warn_export_conditioning(getattr(self, "backward_cond_est_", None), "backward")
        Y_var = np.ones(self.n_outputs) if not self.standardize_Y_with_std else self.scaler_Y.var_
        exprs = symbolic_polynomial_expressions(
            self.backward_coeffs,
            self.backward_multi_indices,
            variable_names=variable_names,
            input_means=self.scaler_Y.mean_,
            input_vars=Y_var,
            output_means=self.scaler_X.mean_,
            output_vars=self.scaler_X.var_,
            input_transform=self._transforms(),
            raw_units=raw_units,
        )
        return exprs
