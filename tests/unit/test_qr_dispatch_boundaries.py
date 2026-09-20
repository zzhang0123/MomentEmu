"""T-001: boundary behaviour of the normal-equations / QR dispatcher.

Forming M = Phi^T Phi squares the conditioning, so the normal equations lose
accuracy long before M is singular. COND_RAISE is the hard-error level; the QR
refit is gated separately on COND_QR. Per boundary-validation.md these tests
evaluate BOTH methods directly rather than inferring agreement from the
dispatcher's own output, and they include the extreme degrees where the failure
modes actually live.
"""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.core import solve_emulator_coefficients
from MomentEmu.emulator import generate_multi_indices
from MomentEmu.guards import COND_QR, COND_RAISE
from MomentEmu.monomials import MonomialPlan


def _err(B, c, yt):
    return float(np.sqrt(np.mean((B @ c - yt[:, None]) ** 2)))


def _target(Z):
    return np.sum(np.sin(1.5 * Z), axis=1) + np.exp(0.5 * Z[:, 0])


def _case(P, d, N, seed=0):
    """Design, moments and a held-out set at a given (P, d); standardised as the
    emulator standardises, since that changes the conditioning materially."""
    rng = np.random.default_rng(seed)
    X, Xt = rng.uniform(-1, 1, (N, P)), rng.uniform(-1, 1, (2000, P))
    mu, sd = X.mean(0), X.std(0)
    plan = MonomialPlan.build(generate_multi_indices(P, d))
    A, B = plan.evaluate((X - mu) / sd), plan.evaluate((Xt - mu) / sd)
    y, yt = _target(X), _target(Xt)
    return A, B, y.reshape(-1, 1), yt


def test_qr_gate_sits_below_the_hard_error_level():
    assert COND_QR < COND_RAISE
    # M's conditioning is the square of Phi's, so the gap must be wide enough
    # to cover that squaring rather than being a token margin.
    assert COND_RAISE / COND_QR >= 100.0


def test_both_methods_agree_below_the_gate_and_diverge_above_it():
    """Bypass the dispatcher: solve each way directly and compare. Below the
    gate the two must agree (so routing to Cholesky is safe); above it the
    normal equations must be the worse of the two (so the gate earns its cost)."""
    below = _case(2, 12, 4000)
    above = _case(1, 18, 4000)
    for (A, B, y, yt), expect_gap in ((below, False), (above, True)):
        M, nu = A.T @ A, A.T @ y
        cond = np.linalg.cond(M)
        c_ne = np.linalg.solve(M, nu)
        c_qr = np.linalg.lstsq(A, y, rcond=None)[0]
        ratio = _err(B, c_ne, yt) / max(_err(B, c_qr, yt), 1e-300)
        if expect_gap:
            assert cond >= COND_QR
            assert ratio > 10.0
        else:
            assert cond < COND_QR
            assert ratio < 10.0


def test_solver_takes_the_qr_branch_inside_the_window():
    """cond in [COND_QR, COND_RAISE): the branch that used to run only above
    COND_RAISE must now run, and must beat the Cholesky answer."""
    A, B, y, yt = _case(1, 18, 4000)
    M, nu = A.T @ A, A.T @ y
    cond = np.linalg.cond(M)
    assert COND_QR <= cond < COND_RAISE, f"probe left the window: cond={cond:.2e}"
    c_dispatched, _ = solve_emulator_coefficients(
        M, nu, on_singular="warn", return_cond=True, Phi=A, Y=y
    )
    c_cholesky, _ = solve_emulator_coefficients(
        M, nu, on_singular="warn", return_cond=True
    )
    assert _err(B, c_dispatched, yt) < _err(B, c_cholesky, yt) / 10.0


def test_no_qr_branch_when_well_conditioned():
    """A well-conditioned rung must keep the Cholesky answer, so the extra
    factorisation is not paid on every fit."""
    A, B, y, yt = _case(2, 8, 4000)
    M, nu = A.T @ A, A.T @ y
    assert np.linalg.cond(M) < COND_QR
    with_phi, _ = solve_emulator_coefficients(
        M, nu, on_singular="warn", return_cond=True, Phi=A, Y=y
    )
    without, _ = solve_emulator_coefficients(M, nu, on_singular="warn", return_cond=True)
    np.testing.assert_allclose(with_phi, without, rtol=0, atol=0)


@pytest.mark.parametrize("P, d, N", [(1, 8, 2000), (1, 18, 4000), (1, 26, 4000),
                                     (2, 8, 4000), (2, 16, 6000), (3, 12, 8000)])
def test_dispatched_solve_is_finite_and_no_worse_than_cholesky(P, d, N):
    """Sweep the corners, including a degree far past where M is singular."""
    A, B, y, yt = _case(P, d, N)
    M, nu = A.T @ A, A.T @ y
    c, _ = solve_emulator_coefficients(
        M, nu, on_singular="warn", return_cond=True, Phi=A, Y=y
    )
    assert np.isfinite(c).all()
    c_chol, _ = solve_emulator_coefficients(M, nu, on_singular="warn", return_cond=True)
    if np.isfinite(c_chol).all():
        assert _err(B, c, yt) <= max(_err(B, c_chol, yt) * 1.5, 1e-14)
