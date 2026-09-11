from __future__ import annotations

import logging
import warnings
from itertools import combinations_with_replacement
from collections import Counter

import numpy as np
from MomentEmu.guards import (
    COND_RAISE,
    DomainBox,
    ExtrapolationWarning,
    IllConditionedError,
    InsufficientSamplesError,
    as_float64,
    basis_size,
    check_axis_levels,
    check_degree_range,
    check_design_columns,
    check_distinct_rows,
    check_finite,
    check_in_domain,
    check_log_domain,
    check_sample_count,
    check_sweep_rmse,
    check_test_pair,
    check_xy_shapes,
    count_distinct_rows,
    extrapolation_distance,
    fit_domain_box,
    max_supported_degree,
    resolve_batch_shape,
)
from MomentEmu.monomials import (
    MonomialPlan,
    fold_output_affine,
)
from MomentEmu.MomentEmu import (
    generate_moment_products,
    press_loo,
    solve_emulator_coefficients,
    predictive_mse_aic_bic,
    select_best_model,
    signal_aware_frac_err,
)

# Memory budget for a backward-sweep moment matrix M (D x D float64). The
# backward basis is built from n_outputs, which can be thousands of CMB bins,
# so a single degree rung can otherwise allocate tens of GB.
MAX_BACKWARD_MOMENT_BYTES = 1 << 30  # 1 GiB

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
        counter = Counter(c)
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
    n = X.shape[1]

    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D

    Mm, nu = generate_moment_products(Phi, Y)

    return Mm, nu

def compute_moments_vector_output_batched(X, Y, multi_indices, batch_size=10000):
    """
    Memory-efficient version of moment computation using batched processing.
    
    Args:
        X: N x n input parameter array
        Y: N x m observable array  
        multi_indices: list of multi-indices
        batch_size: number of samples to process at once
        
    Returns:
        Mm: moment matrix (D x D)
        nu: moment vectors (D x m)
    """     
    N, n = X.shape
    m = Y.shape[1]
    D = len(multi_indices)
    
    # Initialize accumulators
    Mm = np.zeros((D, D), dtype=X.dtype)
    nu = np.zeros((D, m), dtype=X.dtype)
    
    # Process data in batches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_size_actual = end_idx - start_idx
        
        # Evaluate monomials for this batch
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        Phi_batch = evaluate_monomials_lazy(X_batch, multi_indices)  # batch_size x D
        
        # Accumulate moment matrix: M += Phi_batch.T @ Phi_batch
        Mm += Phi_batch.T @ Phi_batch
        
        # Accumulate moment vector: nu += Phi_batch.T @ Y_batch
        nu += Phi_batch.T @ Y_batch
        
        # Clear batch from memory
        del Phi_batch, X_batch, Y_batch
    
    # Normalize by total number of samples
    Mm /= N
    nu /= N
    
    return Mm, nu

def symbolic_polynomial_expressions(coeffs, multi_indices, variable_names=None,
                                    input_means=None, input_vars=None,
                                    output_means=None, output_vars=None,
                                    *, log_input=False, log_output=False,
                                    transform=None, input_transform=None,
                                    raw_units=False):
    """Convert emulator coefficients into sympy expressions (P2.4, D11).

    The default form keeps the standardized coordinates
    z_i = (x_i - m_i)/s_i as an unexpanded rational expression, evaluated with
    17 significant digits. Expanding in raw units loses digits with degree (up
    to 8.3e-9 at n=6, d=6 in the review), so raw_units=True warns.
    log_input substitutes log of the input variables (backward export of a
    log_Y emulator); log_output wraps the result in exp(..., evaluate=False).
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
    vars_sym = [sp.Symbol(name) for name in variable_names]
    input_mean = np.zeros(n) if input_means is None else np.asarray(input_means, float)
    input_std = np.ones(n) if input_vars is None else np.sqrt(np.asarray(input_vars, float))
    output_mean = np.zeros(m) if output_means is None else np.asarray(output_means, float)
    output_std = np.ones(m) if output_vars is None else np.sqrt(np.asarray(output_vars, float))
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
        substitutions[z_syms[i]] = (
            xi - sp.Float(float(input_mean[i]), 17)
        ) / sp.Float(float(input_std[i]), 17)
    expressions = []
    for j in range(m):
        terms = {}
        for c, alpha in zip(coeffs[:, j], mi):
            key = tuple(int(a) for a in alpha)
            terms[key] = terms.get(key, sp.Integer(0)) + sp.Float(float(c), 17)
        poly = sp.Poly.from_dict(terms, z_syms)
        # xreplace is a direct tree substitution and is ~8x faster than subs
        # here (0.019 s vs 0.157 s at D=462), which keeps the export under 0.1 s.
        expr = poly.as_expr().xreplace(substitutions)
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


class PolyEmu():
    # Validation diagnostics, populated only when return_max_frac_err=True.
    # Class-level defaults make these attributes safe to read unconditionally
    # (returns None instead of AttributeError when the diagnostic was not
    # requested at construction time).
    forward_max_frac_err: float | None = None
    backward_max_frac_err: float | None = None
    forward_frac_err_diag: dict | None = None
    backward_frac_err_diag: dict | None = None

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
                transform=None):
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
            batch_size = X.shape[0]

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
        self.scaler_X = StandardScaler()
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
                D = basis_size(self.n_params, max_degree_forward)
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
                                            loo=use_loo)
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
                                  loo=False):

        if init_deg is None:
            if self.n_params > 6:
                init_deg = 1
            elif self.n_params < 3:
                init_deg = 3
            else:
                init_deg = 2

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
        loo_list = []
        loo_per_output_list = []
        leverage_list = []
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
        # Scale-free RMSE_tol (P1.4): compare against the RMS of the
        # validation Y in the same space, so rescaling the data with
        # with_std=False does not change the selected degree.
        _ref_Y = Y_train_scaled if loo else Y_val_scaled
        _y_rms = float(np.sqrt(np.mean(np.asarray(_ref_Y) ** 2)))
        rmse_ref = _y_rms if _y_rms > 0 else 1.0

        for d in range(init_deg, max_degree + 1):
            start_time = time.time()

            if d == init_deg:
                raw_indices = generate_multi_indices(self.n_params, d)
            else:
                aux_indices = given_order_indices(self.n_params, d)
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

            if loo:
                # Fit on all N and score by exact leave-one-out PRESS from the
                # same Cholesky factor (P1.5). The metric in RMSE_val_list is
                # the LOO-RMSE, so selection is uniform.
                _plan = MonomialPlan.build(multi_indices)
                Phi = _plan.evaluate(X_train_scaled)
                M = Phi.T @ Phi / X_train_scaled.shape[0]
                nu = Phi.T @ Y_train_scaled / X_train_scaled.shape[0]
                coeffs, cond, loo_rmse, loo_per_out, lev_max = press_loo(
                    M, nu, Phi, Y_train_scaled, on_singular="warn", degree=d
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
                    X_train_scaled, Y_train_scaled, multi_indices, batch_size=batch_size
                )
                coeffs, cond = solve_emulator_coefficients(
                    M, nu, on_singular="warn", degree=d, return_cond=True
                )
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                Y_val_pred = evaluate_emulator_batched(
                    X_val_scaled, coeffs, multi_indices, batch_size=batch_size
                )
                RMSE_val, AIC, BIC = predictive_mse_aic_bic(
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

            if cond >= COND_RAISE and (loo or validation_is_training):
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

        self.forward_coeffs = coeffs
        self.forward_multi_indices = multi_indices
        self.forward_RMSE_list = RMSE_val_list
        self.forward_AIC_list = AIC_list
        self.forward_BIC_list = BIC_list
        self.forward_running_time_list = running_time_list
        self.forward_degree_list = degree_list
        # Metrics of the model actually stored (the old dim_reduction path
        # reported the pre-pruning model).
        self.forward_RMSE = float(RMSE_val_list[ind])
        self.forward_AIC = float(AIC_list[ind]) if AIC_list else float("nan")
        self.forward_BIC = float(BIC_list[ind]) if BIC_list else float("nan")
        # Cholesky factor of the selected moment matrix, kept for leverage().
        from scipy.linalg import cho_factor

        _final_plan = MonomialPlan.build(multi_indices)
        _Phi = _final_plan.evaluate(X_train_scaled)
        _M = _Phi.T @ _Phi / X_train_scaled.shape[0]
        self.forward_chol_ = cho_factor(_M, lower=False, check_finite=False)
        self.forward_N_train_ = int(X_train_scaled.shape[0])
        # Per-output residual standard deviation in the fitted (standardized Y)
        # space, used by the noise-only predictive band (P3.5).
        _resid = Y_train_scaled - _Phi @ coeffs
        _dof = max(self.forward_N_train_ - multi_indices.shape[0], 1)
        self.forward_resid_std_ = np.sqrt(np.sum(_resid ** 2, axis=0) / _dof)
        self._build_forward_plan()

    def _build_forward_plan(self):
        """Build the P0.7 monomial plan and folded output affine for inference."""
        mi = np.asarray(self.forward_multi_indices)
        self.forward_plan = MonomialPlan.build(mi)
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
        cf, lower = self.forward_chol_
        A = solve_triangular(cf, Phi.T, lower=lower, trans="T", check_finite=False)
        h = np.einsum("ij,ij->j", A, A) / self.forward_N_train_
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
        if self.log_Y:
            Y_affine = self.forward_plan.evaluate(Xs) @ self.forward_coeffs_folded
            J = J * np.exp(Y_affine)[:, :, None]
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

    def _transforms(self):
        """Per-output transform tuple; legacy pickles get the log_Y mapping."""
        t = getattr(self, "transform", None)
        if t is None:
            t = tuple("log" if self.log_Y else "linear" for _ in range(self.n_outputs))
            self.transform = t
        return t

    def forward_emulator(self, X, batch_size=None, extrapolation="warn", return_std=False):
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
        std = None
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
                                   loo=False):
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
        loo_list = []
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
        _ref_X = X_train_scaled if loo else X_val_scaled
        _x_rms = float(np.sqrt(np.mean(np.asarray(_ref_X) ** 2)))
        rmse_ref = _x_rms if _x_rms > 0 else 1.0

        for d in range(init_deg, max_degree + 1):
            start_time = time.time()
            if d == init_deg:
                raw_indices = generate_multi_indices(self.n_outputs, d)
            else:
                aux_indices = given_order_indices(self.n_outputs, d)
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

            if loo:
                _plan = MonomialPlan.build(multi_indices)
                Phi = _plan.evaluate(Y_train_scaled)
                M = Phi.T @ Phi / Y_train_scaled.shape[0]
                nu = Phi.T @ X_train_scaled / Y_train_scaled.shape[0]
                coeffs, cond, loo_rmse, loo_per_out, lev_max = press_loo(
                    M, nu, Phi, X_train_scaled, on_singular="warn", degree=d
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
                    Y_train_scaled, X_train_scaled, multi_indices, batch_size=batch_size
                )
                coeffs, cond = solve_emulator_coefficients(
                    M, nu, on_singular="warn", degree=d, return_cond=True
                )
                if not np.isfinite(coeffs).all():
                    stop_reason = "Cholesky failed"
                    break
                degree_list.append(d)
                cond_list.append(float(cond))
                X_val_pred = evaluate_emulator_batched(
                    Y_val_scaled, coeffs, multi_indices, batch_size=batch_size
                )
                RMSE_val, AIC, BIC = predictive_mse_aic_bic(
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

            if cond >= COND_RAISE and (loo or validation_is_training):
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
        from scipy.linalg import cho_factor

        _final_plan = MonomialPlan.build(multi_indices)
        _Phi = _final_plan.evaluate(Y_train_scaled)
        _M = _Phi.T @ _Phi / Y_train_scaled.shape[0]
        self.backward_chol_ = cho_factor(_M, lower=False, check_finite=False)
        self.backward_N_train_ = int(Y_train_scaled.shape[0])
        self._build_backward_plan()

    def backward_emulator(self, Y, batch_size=None, extrapolation="warn"):
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

    def generate_forward_symb_emu(self, variable_names=None, *, raw_units=False):
        """SymPy expressions for the forward emulator (P2.4, D11).

        Standardized z-form by default; log_Y wraps the result in exp.
        """
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
        )
        return exprs

    def generate_backward_symb_emu(self, variable_names=None, *, raw_units=False):
        """SymPy expressions for the backward emulator (P2.4, D11).

        z-form by default; log_Y substitutes log of the input variables.
        """
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
