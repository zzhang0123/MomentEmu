"""P5.4: QR / CholeskyQR2 path at the solver-policy levels."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.core import cholesky_qr2_solve, qr_solve, solve_emulator_coefficients


def _diagonal_problem(cond, D):
    s = np.geomspace(1.0, 1.0 / cond, D)
    M = np.diag(s)
    Phi = np.diag(np.sqrt(s))
    rng = np.random.default_rng(0)
    c_true = rng.normal(size=(D, 3))
    Y = Phi @ c_true
    return M, Phi, Y, c_true


@pytest.mark.parametrize("factor", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("D", [66, 3003])
def test_cholesky_and_qr_agree_at_1e16(factor, D):
    cond = 1e16 * factor
    M, Phi, Y, c_true = _diagonal_problem(cond, D)
    nu = M @ c_true
    c_chol = solve_emulator_coefficients(
        M, nu, on_singular="warn", warn_at=np.inf, raise_at=np.inf
    )
    c_cqr = cholesky_qr2_solve(Phi, Y)
    c_qr, rank = qr_solve(Phi, Y)
    ref = Phi @ c_true
    for c in (c_chol, c_cqr, c_qr):
        rel = np.max(np.abs(Phi @ c - ref)) / np.max(np.abs(ref))
        assert rel < 1e-8, (D, factor, rel)
    assert rank == D


def test_qr_refuses_rank_deficient_with_lstsq():
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(50, 10))
    Phi[:, 9] = Phi[:, 0]  # exact rank 9 < 10
    Y = rng.normal(size=(50, 2))
    with pytest.warns(UserWarning):
        c, rank = qr_solve(Phi, Y)
    assert rank < 10


def test_isotropic_low_degree_fit_unchanged():
    from MomentEmu.emulator import PolyEmu

    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (500, 3))
    Y = (X[:, 0] ** 2 + X[:, 1] ** 3 + X[:, 2]).reshape(-1, 1)
    emu = PolyEmu(
        X, Y, init_deg_forward=6, max_degree_forward=6, RMSE_tol=1e-300, verbose=0
    )
    # The isotropic d=6 moment matrix is far below 1e16, so the Cholesky path
    # is used and the fit is exact for a quadratic target.
    assert emu.forward_cond_est_ < 1e16
    assert emu.forward_RMSE_per_output_[0] < 1e-10
