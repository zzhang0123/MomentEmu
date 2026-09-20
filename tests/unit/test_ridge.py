"""T-003: Tikhonov regularisation of the moment matrix.

It is not what the QR refit is. QR solves the SAME least-squares problem more
accurately; this solves a DIFFERENT, better-posed one, trading bias for
variance. That is why QR alone cannot rescue an over-complete basis.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.core import apply_ridge, solve_emulator_coefficients
from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.monomials import MonomialPlan


def test_apply_ridge_scales_per_column():
    """One common lambda is only safe when the diagonal is roughly uniform,
    and a polynomial moment matrix is the opposite of that."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 12)) * np.logspace(0, 6, 12)
    M = A.T @ A
    out, lam = apply_ridge(M, 1e-3)
    np.testing.assert_allclose(lam, 1e-3 * np.diag(M), rtol=1e-12)
    np.testing.assert_allclose(out, M + np.diag(lam))
    # dimensionless per column: rescaling a column rescales only its own lambda
    S = np.diag(np.concatenate([[1000.0], np.ones(11)]))
    _, lam2 = apply_ridge(S @ M @ S, 1e-3)
    assert lam2[0] == pytest.approx(1e6 * lam[0])
    np.testing.assert_allclose(lam2[1:], lam[1:], rtol=1e-12)


def test_a_column_with_no_variance_still_gets_a_penalty():
    M = np.diag([4.0, 1.0, 0.0])
    out, lam = apply_ridge(M, 1e-3)
    assert lam[2] > 0.0 and np.linalg.cond(out) < np.inf


def test_zero_ridge_is_bit_identical():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((60, 8))
    M, nu = A.T @ A, A.T @ rng.standard_normal((60, 2))
    out, lam = apply_ridge(M, 0.0)
    assert lam == 0.0 and out is M
    c0 = solve_emulator_coefficients(M, nu, on_singular="warn")
    c1 = solve_emulator_coefficients(M, nu, on_singular="warn", ridge=0.0)
    np.testing.assert_array_equal(c0, c1)


def test_ridge_is_validated():
    rng = np.random.default_rng(0)
    M = np.eye(4)
    with pytest.raises(ValueError, match="ridge must be"):
        apply_ridge(M, -1e-3)
    X = rng.uniform(-1, 1, (500, 2))
    with pytest.raises(ValueError, match="ridge must be"):
        PolyEmu(X, (X[:, 0] ** 2).reshape(-1, 1), ridge=-1.0,
                max_degree_forward=3, verbose=0)


def test_ridge_caps_the_condition_number():
    """Eigenvalues become sigma**2 + lam, so cond is capped near
    sigma_max**2 / lam however singular M was."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((200, 20))
    A[:, -1] = A[:, 0] + 1e-9 * A[:, 1]          # nearly dependent column
    M = A.T @ A
    bare = np.linalg.cond(M)
    ridged, lam = apply_ridge(M, 1e-8)
    assert bare > 1e14
    assert np.linalg.cond(ridged) < bare / 1e4
    assert np.all(lam > 0.0)


@pytest.fixture(scope="module")
def easy():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (3000, 3))
    return X, (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]).reshape(-1, 1)


def test_small_ridge_costs_nothing_and_large_ridge_biases(easy):
    """Where conditioning is not the limit, ridge buys nothing and eventually
    costs. Measured with the per-column form: 2.199e-04 at both 0 and 1e-8,
    2.212e-04 at 1e-6, 1.403e-03 at 1e-4 and 4.642e-02 at 1e-2."""
    X, Y = easy
    err = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in (0.0, 1e-8, 1e-2):
            e = PolyEmu(X, Y, ridge=r, init_deg_forward=6, max_degree_forward=6,
                        RMSE_tol=0.0, verbose=0)
            p = e.forward_emulator(X[:300], extrapolation="ignore")
            err[r] = float(np.sqrt(np.mean((p - Y[:300]) ** 2)))
    assert err[1e-8] == pytest.approx(err[0.0], rel=0.02)
    assert err[1e-2] > 10 * err[0.0]


def test_ridge_rescues_a_basis_the_data_cannot_support():
    """The case it exists for. With 969 terms on 3,200 samples of an
    effectively two-dimensional target, cond(M) is 2.3e21 and the
    unregularised Cholesky returns NaN outright. A ridge of 1e-12 returns a
    model with a held-out error of 1.1e-3.

    The held-out error is not monotone in the ridge -- measured 1.1e-3 at
    1e-12, 1.1e-2 at 1e-10 and 1e-8, 8.3e-3 at 1e-6 -- so this is a knob to
    scan, not one with a safe default. That is why it is off by default.
    """
    rng = np.random.default_rng(4)
    raw = rng.uniform(-1, 1, (4000, 6))
    W = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1.], [1, 0, 0, 0, 0, 1]]).T / 3.0
    Z = raw @ W
    y = np.tanh(2.0 * Z[:, 0] + 0.8 * Z[:, 1]).reshape(-1, 1)
    lo, hi = Z.min(0), Z.max(0)
    B = 2 * (Z - lo) / (hi - lo) - 1
    A = MonomialPlan.build(generate_multi_indices(3, 16)).evaluate(B)
    order = rng.permutation(B.shape[0])
    hold, keep = order[:800], order[800:]
    M, nu = A[keep].T @ A[keep], A[keep].T @ y[keep]
    assert np.linalg.cond(M) > 1e18

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = solve_emulator_coefficients(M, nu, on_singular="warn", ridge=0.0)
        ridged = solve_emulator_coefficients(M, nu, on_singular="warn", ridge=1e-12)
    assert not np.isfinite(bare).all(), "probe is not ill-conditioned enough"
    assert np.isfinite(ridged).all()
    held_out = float(np.sqrt(np.mean((A[hold] @ ridged - y[hold]) ** 2)))
    assert held_out < 1e-2, held_out


def test_qr_is_skipped_when_a_ridge_is_requested(easy):
    """QR answers the unregularised question, so it is not a refinement of a
    ridged solve. Passing Phi must not silently replace the ridged answer."""
    X, Y = easy
    plan = MonomialPlan.build(generate_multi_indices(3, 4))
    A = plan.evaluate(X)
    A[:, -1] = A[:, 1] + 1e-11 * A[:, 2]
    M, nu = A.T @ A, A.T @ Y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridged = solve_emulator_coefficients(M, nu, on_singular="warn",
                                             ridge=1e-6, Phi=A, Y=Y)
        no_phi = solve_emulator_coefficients(M, nu, on_singular="warn", ridge=1e-6)
    np.testing.assert_allclose(ridged, no_phi, rtol=1e-10)
