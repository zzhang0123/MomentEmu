"""T-003: tensor-product orthogonal bases (Legendre, Chebyshev).

It spans the same functions as the monomial basis over the same index set, so
the fitted model is identical wherever conditioning is not the limit. What it
changes is the conditioning, and at high degree that is the difference between
a usable fit and a broken one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.monomials import ChebyshevPlan, LegendrePlan, MonomialPlan


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (4000, 3))
    Y = (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]).reshape(-1, 1)
    return X, Y


def test_plan_spans_the_same_functions_as_the_monomial_plan():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (400, 3))
    y = np.sin(2 * Z[:, 0]) + Z[:, 1] * Z[:, 2]
    mi = generate_multi_indices(3, 5)
    fits = []
    for plan in (MonomialPlan.build(mi), ChebyshevPlan.build(mi)):
        A = plan.evaluate(Z)
        fits.append(A @ np.linalg.lstsq(A, y, rcond=None)[0])
    assert np.max(np.abs(fits[0] - fits[1])) < 1e-12


def test_plan_is_bounded_on_the_box_and_saturates_outside_it():
    mi = generate_multi_indices(2, 8)
    plan = ChebyshevPlan.build(mi)
    inside = plan.evaluate(np.random.default_rng(0).uniform(-1, 1, (200, 2)))
    outside = plan.evaluate(np.array([[-4.0, 4.0], [10.0, -10.0]]))
    assert np.abs(inside).max() <= 1.0 + 1e-12
    assert np.isfinite(outside).all() and np.abs(outside).max() <= 1.0 + 1e-12


def test_plan_derivatives_match_finite_differences():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-0.9, 0.9, (300, 3))
    plan = ChebyshevPlan.build(generate_multi_indices(3, 5))
    D = plan.evaluate_derivatives(Z)
    h = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fd = (plan.evaluate(Z + e) - plan.evaluate(Z - e)) / (2 * h)
        assert np.max(np.abs(D[i] - fd)) < 1e-7


@pytest.mark.parametrize("degree", [3, 6, 8])
def test_both_bases_fit_the_same_model_where_conditioning_is_not_the_limit(data, degree):
    """Regression for a real defect: the moment matrix was built with
    MonomialPlan while the design was evaluated with ChebyshevPlan, so the
    stored coefficients solved a different problem. The residual of those
    coefficients was 1.30 where a direct solve in the same basis gave 2.5e-04,
    and the emulator reported no error at all."""
    X, Y = data
    err = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("monomial", "chebyshev"):
            e = PolyEmu(X, Y, basis_kind=kind, init_deg_forward=degree,
                        max_degree_forward=degree, RMSE_tol=0.0, verbose=0)
            p = e.forward_emulator(X[:300], extrapolation="ignore")
            err[kind] = float(np.sqrt(np.mean((p - Y[:300]) ** 2)))
    assert err["chebyshev"] == pytest.approx(err["monomial"], rel=1e-6)


def test_chebyshev_implies_box_scaling(data):
    """T_k is bounded only on [-1, 1], so the box map is a precondition."""
    X, Y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = PolyEmu(X, Y, basis_kind="chebyshev", max_degree_forward=4, verbose=0)
    assert e.scaling == "box"


def test_chebyshev_is_far_better_conditioned_at_high_degree():
    rng = np.random.default_rng(3)
    raw = rng.uniform(-1, 1, (9000, 7))
    w = np.array([0.6, 0.5, -0.4, 0.3, 0.2, -0.15, 0.1])
    Z = np.column_stack([raw @ w, raw @ np.roll(w, 3), raw[:, 0]])
    lo, hi = Z.min(0), Z.max(0)
    B = 2 * (Z - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
    def cond_at(degree):
        mi = generate_multi_indices(3, degree)
        out = {}
        for name, plan in (("monomial", MonomialPlan.build(mi)),
                           ("chebyshev", ChebyshevPlan.build(mi))):
            P = plan.evaluate(B)
            out[name] = float(np.linalg.cond(P.T @ P))
        return out

    # Like the box map, the advantage needs degree and is not free below it:
    # measured 0.4x at degree 6 and 10, 48x at 14, 10770x at 18.
    low = cond_at(6)
    assert low["chebyshev"] > low["monomial"]
    high = cond_at(18)
    assert high["chebyshev"] < high["monomial"] / 1e3, high


@pytest.mark.parametrize("kwargs, match", [
    ({"basis_kind": "sideways"}, "basis_kind must be"),
    ({"basis_kind": "chebyshev", "backward": True, "forward": False},
     "forward model only"),
])
def test_validation(data, kwargs, match):
    X, Y = data
    with pytest.raises(ValueError, match=match):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PolyEmu(X, Y, max_degree_forward=3, verbose=0, **kwargs)


def test_symbolic_export_writes_chebyshev_not_monomials(data):
    """The coefficients multiply Chebyshev polynomials, so the export writes
    those. Reading them as monomial coefficients would be silently wrong, and
    this refused the export while that was the only alternative."""
    import sympy as sp

    X, Y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = PolyEmu(X, Y, basis_kind="chebyshev", max_degree_forward=4, verbose=0)
        exprs = e.generate_forward_symb_emu(["u", "v", "w"])
    syms = sp.symbols(["u", "v", "w"])
    got = np.column_stack([
        np.asarray(sp.lambdify(syms, x, "numpy")(*X.T), dtype=float)
        * np.ones(X.shape[0])
        for x in exprs
    ])
    np.testing.assert_allclose(got, e.forward_emulator(X), rtol=1e-9, atol=1e-10)


def test_jacobian_stays_analytic_under_the_new_basis(data):
    X, Y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = PolyEmu(X, Y, basis_kind="chebyshev", max_degree_forward=6, verbose=0)
    J = e.jacobian(X[:5])
    h = 1e-6
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        fd = (e.forward_emulator(X[:5] + d, extrapolation="ignore")
              - e.forward_emulator(X[:5] - d, extrapolation="ignore")) / (2 * h)
        np.testing.assert_allclose(J[:, :, i], fd, rtol=1e-5, atol=1e-8)


# --- Legendre --------------------------------------------------------------

def test_legendre_is_orthonormal_under_a_uniform_design():
    """sqrt(2k+1) P_k is orthonormal under the uniform probability measure on
    [-1, 1], which is what a Latin hypercube or a uniform box produces."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, (400000, 1))
    P = LegendrePlan.build(np.arange(8).reshape(-1, 1)).evaluate(z)
    G = P.T @ P / z.shape[0]
    assert np.abs(G - np.eye(8)).max() < 0.02


def test_legendre_spans_the_same_functions():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (400, 3))
    y = np.sin(2 * Z[:, 0]) + Z[:, 1] * Z[:, 2]
    mi = generate_multi_indices(3, 5)
    fits = []
    for plan in (MonomialPlan.build(mi), LegendrePlan.build(mi)):
        A = plan.evaluate(Z)
        fits.append(A @ np.linalg.lstsq(A, y, rcond=None)[0])
    assert np.max(np.abs(fits[0] - fits[1])) < 1e-12


def test_legendre_derivatives_match_finite_differences():
    """The recurrence P'_{k+1} = (2k+1) P_k + P'_{k-1} is used rather than the
    closed form with a 1/(1-z**2) factor, which is singular at the endpoints."""
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.uniform(-0.99, 0.99, (300, 2)),
                   np.array([[-0.999, 0.999], [0.999, -0.999]])])
    plan = LegendrePlan.build(generate_multi_indices(2, 6))
    D = plan.evaluate_derivatives(Z)
    h = 1e-7
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        fd = (plan.evaluate(Z + e) - plan.evaluate(Z - e)) / (2 * h)
        assert np.max(np.abs(D[i] - fd)) < 2e-4
    assert np.isfinite(D).all()


@pytest.mark.parametrize("kind", ["legendre", "chebyshev"])
def test_every_basis_fits_the_same_model_where_conditioning_allows(data, kind):
    X, Y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = PolyEmu(X, Y, init_deg_forward=8, max_degree_forward=8,
                      RMSE_tol=0.0, verbose=0)
        alt = PolyEmu(X, Y, basis_kind=kind, init_deg_forward=8,
                      max_degree_forward=8, RMSE_tol=0.0, verbose=0)
        pr = ref.forward_emulator(X[:300], extrapolation="ignore")
        pa = alt.forward_emulator(X[:300], extrapolation="ignore")
    def err(p):
        return float(np.sqrt(np.mean((p - Y[:300]) ** 2)))

    assert err(pa) == pytest.approx(err(pr), rel=1e-6)


def test_legendre_matches_the_design_a_training_set_usually_has():
    """A Latin hypercube or uniform box fills its box evenly, which is the
    measure sqrt(2k+1) P_k is orthonormal under. Measured at degree 14 on a
    uniform design: monomial 1.5e10, Legendre 9.7, Chebyshev 7.7e2."""
    rng = np.random.default_rng(5)
    D = rng.uniform(-1, 1, (8000, 2))
    assert np.mean(np.abs(D) > 0.8) > 0.15, "probe design is not uniform"
    mi = generate_multi_indices(2, 14)
    cond = {}
    for name, cls in (("monomial", MonomialPlan), ("legendre", LegendrePlan),
                      ("chebyshev", ChebyshevPlan)):
        A = cls.build(mi).evaluate(D)
        cond[name] = float(np.linalg.cond(A.T @ A))
    assert cond["legendre"] < cond["chebyshev"] < cond["monomial"], cond
    assert cond["legendre"] < cond["monomial"] / 1e8, cond


def test_the_ordering_survives_an_edge_starved_design():
    """Chebyshev is orthogonal under the arcsine weight, which concentrates at
    the edges, so a design that avoids them is its worst case. It does not
    overtake Legendre there either: at degree 14 on a bell-shaped design,
    monomial 7.8e13, Legendre 5.2e10, Chebyshev 1.8e12."""
    rng = np.random.default_rng(5)
    raw = rng.uniform(-1, 1, (8000, 6))
    W = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1.]]).T / 3.0
    bell = raw @ W
    lo, hi = bell.min(0), bell.max(0)
    D = 2 * (bell - lo) / (hi - lo) - 1
    assert np.mean(np.abs(D) > 0.8) < 0.05, "probe design is not edge-starved"
    mi = generate_multi_indices(2, 14)
    cond = {}
    for name, cls in (("monomial", MonomialPlan), ("legendre", LegendrePlan),
                      ("chebyshev", ChebyshevPlan)):
        A = cls.build(mi).evaluate(D)
        cond[name] = float(np.linalg.cond(A.T @ A))
    assert cond["legendre"] < cond["chebyshev"] < cond["monomial"], cond
