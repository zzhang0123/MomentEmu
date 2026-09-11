"""Tests for guards.py.

Layout follows the boundary-validation methodology: for every threshold the
guard is evaluated just below, at, and just above the threshold, plus extreme
parameter values. Round-1 failure modes are reproduced end-to-end against the
real package functions (``generate_multi_indices``, ``evaluate_monomials_lazy``,
``generate_moment_products``, ``max_order``) and pinned.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest


from MomentEmu import guards as g  # noqa: E402
from MomentEmu.MomentEmu import generate_moment_products  # noqa: E402
from MomentEmu.PolyEmu import evaluate_monomials_lazy, generate_multi_indices, max_order  # noqa: E402

RNG = np.random.default_rng(20260910)


def _moment_matrix(X, degree):
    mi = generate_multi_indices(X.shape[1], degree)
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    Phi = evaluate_monomials_lazy(Xs, mi)
    M, _ = generate_moment_products(Phi, np.zeros((X.shape[0], 1)))
    return M, mi


# ==========================================================================
# 1. basis_size / max_supported_degree vs the off-by-one max_order
# ==========================================================================
ROUND1_CELLS = [(1, 5), (1, 3000), (2, 170), (2, 200), (4, 1000), (4, 10000), (6, 20000), (7, 3432), (8, 100), (3, 20)]


@pytest.mark.parametrize("n,N", ROUND1_CELLS)
def test_max_order_is_deprecated_and_matches_fill1(n, N):
    """P0.5: the old off-by-one helper now warns and returns the safe degree."""
    with pytest.warns(DeprecationWarning):
        k = max_order(n, N)
    assert k == g.max_supported_degree(n, N, fill=1.0)


@pytest.mark.parametrize("n,N", ROUND1_CELLS)
def test_max_supported_degree_fill1_keeps_D_below_N(n, N):
    """Boundary: fill=1 must stay identifiable, i.e. D < N strictly."""
    k = g.max_supported_degree(n, N, fill=1.0)
    assert g.basis_size(n, k) < N
    assert g.basis_size(n, k + 1) >= N


@pytest.mark.parametrize("n,N,fill,expected", [(4, 1000, 2.0, 8), (4, 1000, 1.0, 9), (6, 20000, 2.0, 10), (6, 20000, 1.0, 12), (7, 3432, 1.0, 6), (1, 5, 1.0, 3), (1, 1, 1.0, 0)])
def test_max_supported_degree_pins(n, N, fill, expected):
    assert g.max_supported_degree(n, N, fill=fill) == expected


def test_max_supported_degree_rejects_bad_args():
    with pytest.raises(ValueError):
        g.max_supported_degree(0, 10)
    with pytest.raises(ValueError):
        g.max_supported_degree(2, 10, fill=0)


# ==========================================================================
# 2. check_sample_count: N <= D raise; N < fill*D warn; both sides + threshold
# ==========================================================================
@pytest.mark.parametrize("D", [1, 66, 462, 27132])
def test_sample_count_threshold_N_eq_D_raises_N_eq_D_plus_1_passes(D):
    with pytest.raises(g.InsufficientSamplesError, match=rf"D = {D} >= N_train = {D}"):
        g.check_sample_count(D, D)
    with pytest.raises(g.InsufficientSamplesError):
        g.check_sample_count(D - 1 if D > 1 else 0, D)
    if D + 1 < 2.0 * D:  # D=1: N=D+1=2 equals fill*D, which is the silent side
        with pytest.warns(g.EmulatorWarning, match="barely determined"):
            g.check_sample_count(D + 1, D, fill=2.0)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g.check_sample_count(D + 1, D, fill=2.0)


@pytest.mark.parametrize("D", [2, 66, 1001])
def test_sample_count_fill_threshold_both_sides(D):
    fill = 2.0
    N_at = int(fill * D)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.check_sample_count(N_at, D, fill=fill)       # N == fill*D: no warning
        g.check_sample_count(N_at + 1, D, fill=fill)
    with pytest.warns(g.EmulatorWarning):
        g.check_sample_count(N_at - 1, D, fill=fill)   # N == fill*D - 1: warn


def test_sample_count_message_names_supported_degree():
    with pytest.raises(g.InsufficientSamplesError, match="largest degree supported by N_train = 170 with fill factor 1 is 16"):
        g.check_sample_count(170, 171, fill=1.0, degree=17, n_params=2)


# ==========================================================================
# 3. check_distinct_rows: the round-1 repeated-row catastrophe
# ==========================================================================
@pytest.mark.parametrize("k_distinct,overshoot_floor", [(5, 1e3), (20, 1e2)])
def test_repeated_rows_are_caught_and_do_overshoot(k_distinct, overshoot_floor):
    """Round-1: 5/20 distinct rows x 400/100, D=66 -> 11986x / 1607x fresh-point overshoot."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-1, 1, (k_distinct, 2))
    X = np.repeat(base, 2000 // k_distinct, axis=0)
    M, mi = _moment_matrix(X, 10)
    assert mi.shape[0] == 66
    assert g.count_distinct_rows(X) == k_distinct
    with pytest.raises(g.InsufficientSamplesError, match=f"only {k_distinct} distinct input rows"):
        g.check_distinct_rows(X, 66)
    rep = g.assess_conditioning(M)
    assert rep.level == "singular" and rep.cond > 1e16 and rep.rank <= k_distinct + 1


def test_distinct_rows_passes_at_D_plus_1():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (67, 2))
    assert g.check_distinct_rows(X, 66) == 67
    with pytest.raises(g.InsufficientSamplesError):
        g.check_distinct_rows(X[:66], 66)


# ==========================================================================
# 4. check_finite / as_float64 / check_xy_shapes
# ==========================================================================
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", [(0, 0), (3, 1), (9, 2)])
def test_check_finite_reports_count_and_first_index(bad, where):
    X = np.zeros((10, 3))
    X[where] = bad
    X[9, 2] = bad
    n_bad = 1 if where == (9, 2) else 2
    with pytest.raises(ValueError, match=rf"X contains {n_bad} non-finite value\(s\); first at index {re_tuple(where)}"):
        g.check_finite(X, "X")


def re_tuple(t):
    return r"\(" + ", ".join(str(i) for i in t) + r"\)"


def test_check_finite_passes_int_and_clean_float():
    g.check_finite(np.arange(6).reshape(3, 2), "X")
    g.check_finite(np.ones((3, 2)), "X")


def test_as_float64_promotes_float32_with_warning_and_int_silently():
    with pytest.warns(g.PrecisionWarning, match="X is float32"):
        out = g.as_float64(np.ones(3, dtype=np.float32), "X")
    assert out.dtype == np.float64
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert g.as_float64(np.arange(3), "X").dtype == np.float64
        a = np.ones(2)
        assert g.as_float64(a, "X") is a
    with pytest.raises(TypeError):
        g.as_float64(np.array(["a"]), "X")


def test_integer_overflow_avoided_by_as_float64():
    """Round-1 `integer-dtype-overflow`: int64 x**alpha wraps silently."""
    X = np.array([[10_000_000]], dtype=np.int64)
    mi = np.array([[3]])
    wrapped = evaluate_monomials_lazy(X, mi)[0, 0]
    assert wrapped != 1e21  # wrapped modulo 2**64
    ok = evaluate_monomials_lazy(g.as_float64(X, "X"), mi)[0, 0]
    assert ok == pytest.approx(1e21)


@pytest.mark.parametrize("X,Y,msg", [
    (np.zeros(5), np.zeros((5, 1)), "X must be 2-D"),
    (np.zeros((5, 2)), np.zeros(5), "Y must be 2-D"),
    (np.zeros((5, 2)), np.zeros((4, 1)), "same number of rows"),
    (np.zeros((1, 2)), np.zeros((1, 1)), "at least 2 samples"),
    (np.zeros((5, 0)), np.zeros((5, 1)), "at least one column"),
])
def test_check_xy_shapes_messages(X, Y, msg):
    with pytest.raises(ValueError, match=msg):
        g.check_xy_shapes(X, Y)


def test_check_xy_shapes_returns_dims():
    assert g.check_xy_shapes(np.zeros((7, 3)), np.zeros((7, 4))) == (7, 3, 4)


# ==========================================================================
# 5. check_design_columns: constant raise, collinear warn
# ==========================================================================
def test_constant_column_raises_naming_index_and_value():
    X = RNG.uniform(-1, 1, (50, 3))
    X[:, 1] = 0.3
    with pytest.raises(ValueError, match=r"input column\(s\) \[1\] are constant \(value \[0.3\]\)"):
        g.check_design_columns(X)


def test_collinear_column_warns_with_rank():
    X = RNG.uniform(-1, 1, (50, 3))
    X[:, 2] = 2.0 * X[:, 0] - 1.0
    with pytest.warns(g.EmulatorWarning, match="numerical rank 2 of 3"):
        g.check_design_columns(X)


def test_design_columns_pass_on_generic_and_1d():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.check_design_columns(RNG.uniform(-1, 1, (50, 3)))
        g.check_design_columns(RNG.uniform(-1, 1, (50, 1)))


def test_collinear_rtol_boundary_both_sides():
    """Near-collinear column with controlled singular-value ratio."""
    n = 2000
    x = RNG.standard_normal(n)
    for ratio, expect_warn in [(1e-13, True), (1e-11, False)]:
        X = np.column_stack([x, x + ratio * RNG.standard_normal(n)])
        s = np.linalg.svd((X - X.mean(0)) / X.std(0), compute_uv=False)
        # the guard compares s[-1]/s[0] against COLLINEAR_RTOL = 1e-12
        expect_warn = (s[-1] / s[0]) <= g.COLLINEAR_RTOL
        if expect_warn:
            with pytest.warns(g.EmulatorWarning):
                g.check_design_columns(X)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                g.check_design_columns(X)


# ==========================================================================
# 6. check_log_domain
# ==========================================================================
@pytest.mark.parametrize("val,label", [(0.0, "0.0"), (-1e-300, "-1e-300"), (np.nan, "nan"), (-np.inf, "-inf"), (np.inf, "inf")])
def test_log_domain_rejects_nonpositive_and_nonfinite(val, label):
    Y = np.ones((4, 2))
    Y[2, 1] = val
    with pytest.raises(ValueError, match=rf"log_Y=True requires Y > 0 and finite but 1 entry is not; first at \(row 2, col 1\) = {label}"):
        g.check_log_domain(Y)


def test_log_domain_boundary_smallest_positive_passes():
    Y = np.full((3, 1), np.finfo(np.float64).tiny)
    g.check_log_domain(Y)
    Y[0, 0] = 5e-324  # smallest subnormal, still > 0
    g.check_log_domain(Y)


# ==========================================================================
# 7. check_test_pair / check_validation_split
# ==========================================================================
def test_test_pair_half_specified_raises_both_ways():
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match=r"got X_test=given, Y_test=None"):
        g.check_test_pair(x, None)
    with pytest.raises(ValueError, match=r"got X_test=None, Y_test=given"):
        g.check_test_pair(None, x)
    g.check_test_pair(None, None)
    g.check_test_pair(x, x)


@pytest.mark.parametrize("N,test_size,expect", [
    (10, 0.15, (8, 2)),      # ceil(1.5) = 2: exactly at threshold -> passes
    (100, 0.15, (85, 15)),
    (3, 0.5, (1, 2)),        # n_train = 1 -> raises (train side)
    (2, 0.5, (1, 1)),
])
def test_validation_split_threshold(N, test_size, expect):
    n_train, n_val = expect
    if n_val < 2:
        with pytest.raises(ValueError, match=rf"validation split has {n_val} row"):
            g.check_validation_split(N, test_size)
    elif n_train < 2:
        with pytest.raises(ValueError, match=rf"training split has {n_train} row"):
            g.check_validation_split(N, test_size)
    else:
        assert g.check_validation_split(N, test_size) == expect


def test_validation_split_one_row_below_threshold_raises():
    # N=6, test_size=0.15 -> ceil(0.9) = 1 < 2
    with pytest.raises(ValueError, match=r"validation split has 1 row\(s\) \(N = 6, test_size = 0.15\)"):
        g.check_validation_split(6, 0.15)


def test_validation_split_explicit_test_and_no_cv():
    assert g.check_validation_split(50, 0.15, n_test=2) == (50, 2)
    with pytest.raises(ValueError):
        g.check_validation_split(50, 0.15, n_test=1)
    assert g.check_validation_split(50, 0.15, cross_validation=False) == (50, 50)
    with pytest.raises(ValueError, match="test_size must be in"):
        g.check_validation_split(50, 1.0)


# ==========================================================================
# 8. check_degree_range: message states both numbers
# ==========================================================================
def test_degree_range_message_states_both_numbers_and_cap():
    with pytest.raises(ValueError, match=r"init_deg_forward = 5 exceeds max_degree_forward = 3 \(max_degree_forward was capped at 3 from N_train = 20, n = 2\)"):
        g.check_degree_range(5, 3, direction="forward", cap=3, n_train=20, n_params=2)
    with pytest.raises(ValueError, match=r"init_deg_backward = 4 exceeds max_degree_backward = 3;"):
        g.check_degree_range(4, 3, direction="backward")


@pytest.mark.parametrize("init,maxd", [(3, 3), (0, 0), (2, 3)])
def test_degree_range_boundary_passes(init, maxd):
    g.check_degree_range(init, maxd, direction="forward")


# ==========================================================================
# 9. assess_conditioning / check_conditioning: thresholds both sides, pins
# ==========================================================================
def _diag(cond):
    return np.diag([1.0, 1.0 / cond])


@pytest.mark.parametrize("cond", [1e12, 1e16])
def test_conditioning_threshold_below_at_above(cond):
    below = _diag(cond * (1 - 1e-6))
    at = _diag(cond)
    above = _diag(cond * (1 + 1e-6))
    lv_below = g.assess_conditioning(below).level
    lv_at = g.assess_conditioning(at).level
    lv_above = g.assess_conditioning(above).level
    expect_at = "warn" if cond == 1e12 else "singular"
    assert lv_at == expect_at and lv_above == expect_at
    assert lv_below != expect_at


def test_assess_conditioning_matches_numpy_cond():
    X = RNG.uniform(-1, 1, (2000, 2))
    M, _ = _moment_matrix(X, 8)
    rep = g.assess_conditioning(M)
    assert rep.cond == pytest.approx(np.linalg.cond(M), rel=1e-6)
    assert rep.rank == rep.size == 45 and rep.level == "ok"


def test_check_conditioning_warn_level_message():
    with pytest.warns(g.IllConditionedWarning, match=r"cond\(M\) = 1.00e\+13 exceeds 1e\+12: the solve keeps about 2.7 of 15.7 float64 digits"):
        rep = g.check_conditioning(_diag(1e13))
    assert rep.level == "warn"


def test_check_conditioning_raise_and_warn_policy():
    M = _diag(1e17)
    with pytest.raises(g.IllConditionedError, match=r"degree 12: cond\(M\) = 1.00e\+17 exceeds 1e\+16 \(1/eps = 4.5e\+15\): M is numerically singular \(numerical rank 1 of 2, lambda_min = 1.0e-17, eigh\); .* \(N_train = 200\)"):
        g.check_conditioning(M, degree=12, n_samples=200)
    with pytest.warns(g.IllConditionedWarning, match="numerically singular"):
        rep = g.check_conditioning(M, on_singular="warn")
    assert rep.level == "singular"
    # IllConditionedError is a LinAlgError so existing handlers still catch it
    assert issubclass(g.IllConditionedError, np.linalg.LinAlgError)
    with pytest.raises(ValueError):
        g.check_conditioning(M, on_singular="ignore")


def test_conditioning_ok_is_silent_and_rejects_bad_M():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert g.check_conditioning(np.eye(3)).level == "ok"
    with pytest.raises(ValueError, match="square"):
        g.assess_conditioning(np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        g.assess_conditioning(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_exactly_singular_M_reports_inf():
    rep = g.assess_conditioning(np.diag([1.0, 0.0]))
    assert rep.cond == math.inf and rep.rank == 1 and rep.level == "singular"
    # cholesky path: factorisation fails on a singular M -> eigh fallback, same verdict
    rep2 = g.assess_conditioning(np.diag([1.0, 0.0]), method="cholesky")
    assert rep2.method == "eigh-fallback" and rep2.level == "singular" and rep2.rank == 1


def _spd(D, cond, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    s = np.geomspace(1.0, 1.0 / cond, D)
    M = (Q * s) @ Q.T
    return 0.5 * (M + M.T)  # exactly symmetric so eigvalsh and SVD see the same matrix


@pytest.mark.parametrize("D", [g.COND_EIGH_MAX_D - 1, g.COND_EIGH_MAX_D, g.COND_EIGH_MAX_D + 1])
@pytest.mark.parametrize("cond", [1e4, 1e8, 1e10, 1e13])  # >= 1 decade away from COND_WARN=1e12 on both sides
def test_conditioning_method_dispatch_boundary_both_methods_agree(D, cond):
    """Boundary of the auto dispatch at D = COND_EIGH_MAX_D: evaluate BOTH methods directly.

    The 1-norm estimate (dpocon) and the 2-norm value can differ by at most a
    factor D in either direction; on a random dense SPD matrix at D = 200 the
    measured ratio is 10.7x (on real moment matrices 2.7-6.3x, pinned in
    test_cholesky_path_on_real_moment_matrices_matches_eigh_verdict). Both
    must give the same level whenever cond is a decade away from a threshold.
    """
    M = _spd(D, cond)
    e = g.assess_conditioning(M, method="eigh")
    c = g.assess_conditioning(M, method="cholesky")
    a = g.assess_conditioning(M)  # auto
    assert a.method == ("eigh" if D <= g.COND_EIGH_MAX_D else "cholesky")
    # lambda_min is resolved to ~D*eps*cond relative, so loosen the pin accordingly
    rel = max(1e-6, 10 * D * g.EPS64 * cond)
    assert e.cond == pytest.approx(np.linalg.cond(M), rel=rel)   # same quantity as numpy's SVD cond
    assert e.cond == pytest.approx(cond, rel=max(1e-3, rel))
    assert 1.0 / D <= c.cond / e.cond <= D, (c.cond, e.cond)
    assert c.method == "cholesky" and c.rank == D
    assert e.level == c.level == a.level == ("warn" if cond >= 1e12 else "ok")


@pytest.mark.parametrize("cond", [1e12, 1e16])
def test_cholesky_path_exact_on_diagonal_at_threshold(cond):
    for f, expect in [(1 - 1e-6, "below"), (1.0, "at"), (1 + 1e-6, "above")]:
        M = _diag(cond * f)
        e = g.assess_conditioning(M, method="eigh")
        c = g.assess_conditioning(M, method="cholesky")
        assert c.cond == pytest.approx(e.cond, rel=1e-9) and c.level == e.level, (expect, c, e)


def test_cholesky_path_on_real_moment_matrices_matches_eigh_verdict():
    rng = np.random.default_rng(5)
    for n, d, N in [(2, 10, 20000), (2, 10, 67), (4, 8, 20000)]:
        X = rng.uniform(-1, 1, (N, n))
        M, _ = _moment_matrix(X, d)
        e = g.assess_conditioning(M, method="eigh")
        c = g.assess_conditioning(M, method="cholesky")
        assert c.method == "cholesky" and c.rank == e.size
        assert 1.0 <= c.cond / e.cond < 10.0, (n, d, N, c.cond, e.cond)


def test_count_distinct_rows_edge_cases():
    assert g.count_distinct_rows(np.zeros((0, 3))) == 0
    assert g.count_distinct_rows(np.zeros((4, 3))) == 1
    assert g.count_distinct_rows(np.arange(12.0).reshape(4, 3)) == 4
    X = RNG.uniform(-1, 1, (5000, 2))
    assert g.count_distinct_rows(X) == np.unique(X, axis=0).shape[0] == 5000
    assert g.count_distinct_rows(np.repeat(X[:7], 3, axis=0)) == 7
    with pytest.raises(ValueError):
        g.count_distinct_rows(np.zeros(3))


@pytest.mark.parametrize("n,degree,N,expect_level,cond_lo,cond_hi", [
    # pins from robust_cond_calibration.py (uniform inputs, standardized); tolerant bounds
    (2, 10, 20000, "ok", 1e7, 1e9),
    (2, 16, 20000, "warn", 1e12, 1e15),
    (2, 20, 20000, "singular", 1e16, 1e19),
    (2, 10, 66, "warn", 1e13, 1e16),       # N == D exactly (round-1: cond 2e14)
    (2, 10, 67, "warn", 1e12, 1e14),       # N == D + 1
    (2, 10, 132, "ok", 1e8, 1e10),         # N == 2D
])
def test_conditioning_pins_on_real_moment_matrices(n, degree, N, expect_level, cond_lo, cond_hi):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (N, n))
    M, _ = _moment_matrix(X, degree)
    rep = g.assess_conditioning(M)
    assert cond_lo <= rep.cond <= cond_hi, rep
    assert rep.level == expect_level


def test_constant_column_moment_matrix_is_singular():
    """Round-1 `constant-or-collinear-x-column` crash branch: exact zero pivot."""
    X = RNG.uniform(-1, 1, (500, 2))
    X[:, 1] = 0.3
    M, _ = _moment_matrix(X, 4)
    rep = g.assess_conditioning(M)
    assert rep.level == "singular" and rep.rank == 5  # only x1^0..x1^4 survive


# ==========================================================================
# 10. coefficient / prediction finiteness
# ==========================================================================
def test_coefficients_finite_raises_with_location():
    c = np.ones((4, 3))
    c[2, 1] = np.nan
    with pytest.raises(g.IllConditionedError, match=r"degree 3: 1 of 12 fitted coefficients are non-finite \(first at \(basis 2, output 1\)\)"):
        g.check_coefficients_finite(c, degree=3)
    g.check_coefficients_finite(np.ones((4, 3)))


def test_predictions_finite_warn_then_strict():
    Y = np.ones((5, 2))
    Y[3, 0] = np.inf
    with pytest.warns(g.EmulatorWarning, match=r"1 of 5 prediction row\(s\) are non-finite \(first row 3\)"):
        assert g.check_predictions_finite(Y) == 1
    with pytest.raises(FloatingPointError):
        g.check_predictions_finite(Y, strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert g.check_predictions_finite(np.ones((5, 2))) == 0


# ==========================================================================
# 11. extrapolation: per-parameter, distance in sigma, both sides of the box
# ==========================================================================
def _box():
    X = np.column_stack([np.linspace(0.0, 1.0, 101), np.linspace(-2.0, 2.0, 101)])
    return g.fit_domain_box(X), X


def test_box_edges_are_inside_and_epsilon_outside_warns():
    box, X = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = g.check_in_domain(np.array([[0.0, -2.0], [1.0, 2.0]]), box)
    assert d.max() == 0.0
    with pytest.warns(g.ExtrapolationWarning, match=r"parameter 0: 1 of 1 row\(s\) outside \[0, 1\]"):
        g.check_in_domain(np.array([[1.0 + 1e-12, 0.0]]), box)


def test_extrapolation_distance_in_sigma_per_parameter():
    box, X = _box()
    sig = X.std(0)
    Xq = np.array([[-0.5 * sig[0], 0.0], [0.5, 2.0 + 3.0 * sig[1]], [0.5, 0.0]])
    with pytest.warns(g.ExtrapolationWarning) as rec:
        d = g.check_in_domain(Xq, box)
    msg = str(rec[0].message)
    assert "parameter 0: 1 of 3 row(s)" in msg and "farthest 0.50 sigma beyond" in msg
    assert "parameter 1: 1 of 3 row(s)" in msg and "farthest 3.00 sigma beyond" in msg
    np.testing.assert_allclose(d[:, 0], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d[:, 1], [0.0, 3.0, 0.0], atol=1e-12)


def test_extrapolation_tol_sigma_and_strict():
    box, X = _box()
    Xq = np.array([[1.0 + 0.5 * X.std(0)[0], 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.check_in_domain(Xq, box, tol_sigma=0.5)   # at tolerance: silent
    with pytest.warns(g.ExtrapolationWarning):
        g.check_in_domain(Xq, box, tol_sigma=0.5 - 1e-9)
    with pytest.raises(ValueError, match="outside the training box"):
        g.check_in_domain(Xq, box, strict=True)
    with pytest.raises(ValueError):
        g.check_in_domain(Xq, box, tol_sigma=-1)


def test_domain_box_constant_column_scale_is_one():
    box = g.fit_domain_box(np.column_stack([np.ones(5), np.arange(5.0)]))
    assert box.scale[0] == 1.0 and box.lo[0] == box.hi[0] == 1.0


# ==========================================================================
# 12. pruning: empty basis, constant term
# ==========================================================================
def test_empty_basis_raises_with_counts():
    with pytest.raises(ValueError, match=r"removed every basis mode \(0 of 6 kept at threshold 1.0e-06\)"):
        g.check_nonempty_basis(np.zeros(6, bool), threshold=1e-6)
    g.check_nonempty_basis(np.array([False, True]), threshold=1e-6)
    with pytest.raises(ValueError, match="1-D"):
        g.check_nonempty_basis(np.ones((2, 2), bool), threshold=1e-6)


def test_keep_constant_term_is_pure_and_warns():
    mi = generate_multi_indices(2, 2)  # row 0 is the constant
    mask = np.array([False, True, False, True, True, False])
    with pytest.warns(g.EmulatorWarning, match="dropped the constant term \\(basis index 0\\)"):
        new = g.keep_constant_term(mask, mi)
    assert new[0] and not mask[0] and new is not mask
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same = g.keep_constant_term(new, mi)
    np.testing.assert_array_equal(same, new)
    # no constant term in the basis at all -> unchanged copy
    np.testing.assert_array_equal(g.keep_constant_term(mask[1:], mi[1:]), mask[1:])


# ==========================================================================
# 13. select_best_model helpers: NaN handling + 5 % tolerance edge
# ==========================================================================
def test_finite_candidates_round1_nan_crash():
    """Round-1 `select-best-model-nan-crash`: one NaN -> argmin on empty."""
    rmse = [0.5, np.nan, 0.4, np.inf]
    np.testing.assert_array_equal(g.finite_candidates(rmse), [0, 2])
    with pytest.raises(ValueError, match="all 3 candidate models produced non-finite RMSE"):
        g.finite_candidates([np.nan, np.inf, np.nan])
    with pytest.raises(ValueError):
        g.finite_candidates([])


def test_select_within_tolerance_5pct_edge_both_sides():
    tol = 0.05
    rmse = [1.0, 1.0 * (1 + tol), 1.0 * (1 + tol) * (1 + 1e-12), np.nan]
    keep = g.select_within_tolerance(rmse, tol)
    np.testing.assert_array_equal(keep, [0, 1])  # exactly at 1.05 kept, 1.05+eps dropped, NaN ignored
    assert g.select_within_tolerance([np.nan, 2.0], 0.0).tolist() == [1]
    with pytest.raises(ValueError):
        g.select_within_tolerance([1.0], -0.1)


# ==========================================================================
# 14. sweep RMSE blow-up (round-1: 1.79e-1 at d=10 -> 1.80e3 at d=17, silent)
# ==========================================================================
def test_sweep_rmse_blowup_both_sides_of_factor():
    factor = 10.0
    base = [0.5, 0.2, 0.1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert g.check_sweep_rmse(base + [1.0], [2, 3, 4, 5], factor=factor) is False   # exactly 10x: no warn
    with pytest.warns(g.IllConditionedWarning, match=r"validation RMSE at degree 5 \(1.000e\+00\) is 1.0e\+01x the best earlier degree \(4: 1.000e-01\)"):
        assert g.check_sweep_rmse(base + [1.0 + 1e-9], [2, 3, 4, 5], factor=factor) is True
    with pytest.warns(g.IllConditionedWarning):
        assert g.check_sweep_rmse([0.18, 1.8e3], [10, 17]) is True
    assert g.check_sweep_rmse([0.5], [2]) is False
    assert g.check_sweep_rmse([np.nan, 0.5], [2, 3]) is False
    with pytest.warns(g.IllConditionedWarning):
        assert g.check_sweep_rmse([0.5, np.nan], [2, 3]) is True


def test_sweep_blowup_reproduced_end_to_end_N200_n2():
    """Round-1 end-to-end: N_train=170, n=2 -> D=171 > 170 at d=17; RMSE blows up.

    Reproduced with the package's own moment/solve path (no dispatcher), then
    checked that the guards fire: check_sample_count raises at d=17,
    check_sweep_rmse warns on the ratio.
    """
    from MomentEmu.MomentEmu import solve_emulator_coefficients

    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (200, 2))
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.1 * X[:, 0] ** 3
    Xtr, Xv, ytr, yv = X[:170], X[170:], y[:170, None], y[170:, None]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xv_s = (Xtr - mu) / sd, (Xv - mu) / sd
    rmse, degs = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in range(2, 18):
            mi = generate_multi_indices(2, d)
            Phi = evaluate_monomials_lazy(Xtr_s, mi)
            M, nu = generate_moment_products(Phi, ytr)
            c = solve_emulator_coefficients(M, nu, on_singular="warn")
            pred = evaluate_monomials_lazy(Xv_s, mi) @ c
            rmse.append(float(np.sqrt(np.mean((pred - yv) ** 2))))
            degs.append(d)
    with pytest.warns(DeprecationWarning):
        assert max_order(2, 170) == 16
    assert g.basis_size(2, 17) == 171
    assert not np.isfinite(rmse[-1]) or rmse[-1] / min(rmse[:-1]) > 1e2, rmse
    with pytest.raises(g.InsufficientSamplesError):
        g.check_sample_count(170, 171, degree=17, n_params=2)
    with pytest.warns(g.IllConditionedWarning):
        assert g.check_sweep_rmse(rmse, degs) is True


# ==========================================================================
# 15. wrapper helpers
# ==========================================================================
class _Scaler:
    def __init__(self, scale):
        self.scale_ = scale


def test_output_scale_none_gives_ones():
    np.testing.assert_array_equal(g.output_scale(_Scaler(None), 3), np.ones(3))
    np.testing.assert_array_equal(g.output_scale(_Scaler(np.array([2.0, 3.0])), 2), [2.0, 3.0])
    with pytest.raises(ValueError):
        g.output_scale(_Scaler(np.array([2.0])), 2)


@pytest.mark.parametrize("shape,n,expect_single,ok", [
    ((3,), 3, True, True),
    ((3,), 1, None, False),       # round-1 wrapper-1d-batch: 3 points for n=1 is ambiguous -> raise
    ((5, 3), 3, False, True),
    ((2, 5, 3), 3, False, True),
    ((), 1, True, True),
    ((), 2, None, False),
    ((4, 2), 3, None, False),
])
def test_resolve_batch_shape(shape, n, expect_single, ok):
    X = np.zeros(shape)
    if ok:
        X2, single = g.resolve_batch_shape(X, n)
        assert X2.shape == (int(np.prod(shape)) // n if shape else 1, n) and single is expect_single
    else:
        with pytest.raises(ValueError):
            g.resolve_batch_shape(X, n)


class _Emu:
    def __init__(self, log_Y, forward=True):
        self.log_Y = log_Y
        if forward:
            self.forward_coeffs = np.zeros((1, 1))


def test_backend_supports_logy_and_forward():
    with pytest.raises(NotImplementedError, match="jax backend does not apply the inverse log transform"):
        g.check_backend_supports(_Emu(True), "jax")
    with pytest.raises(ValueError, match="built with forward=False"):
        g.check_backend_supports(_Emu(False, forward=False), "torch")
    g.check_backend_supports(_Emu(False), "torch")


# ==========================================================================
# 16. purity: no guard mutates its input
# ==========================================================================
def test_guards_do_not_mutate_inputs():
    X = RNG.uniform(-1, 1, (30, 2))
    X0 = X.copy()
    M, mi = _moment_matrix(X, 3)
    M0 = M.copy()
    mask = np.ones(mi.shape[0], bool)
    mask[0] = False
    mask0 = mask.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.check_finite(X, "X"); g.check_design_columns(X); g.fit_domain_box(X)
        g.check_in_domain(X * 3, g.fit_domain_box(X)); g.assess_conditioning(M)
        g.keep_constant_term(mask, mi); g.as_float64(X, "X"); g.count_distinct_rows(X)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(M, M0)
    np.testing.assert_array_equal(mask, mask0)
