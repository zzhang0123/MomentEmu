"""press_loo must reach a QR solve when the Cholesky of M fails.

M = Phi^T Phi = R^T R, so the R of a Householder QR of Phi IS the Cholesky
factor of M -- obtained without ever forming the normal equations, and so
without squaring cond(Phi). When cho_factor fails at cond(M) ~ 1e19 that
factor still delivers both the coefficients and the exact LOO leverage.

Before the fix press_loo returned NaN in that regime, and PolyEmu read the
NaN as "Cholesky failed", dropped the rung and raised IllConditionedError --
declining to use a solver the package already ships. On the 21cmGEM
benchmark that made every unregularised fit at degree >= 12 unreachable.

The degrees below straddle the boundary: at N=60 on [-1, 1] the Cholesky of
the monomial moment matrix succeeds up to degree 22 and fails from 26.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import cho_factor

from MomentEmu.core import press_loo
from MomentEmu.monomials import MonomialPlan

N_SAMPLES = 60
CHOLESKY_OK = (14, 18, 22)
CHOLESKY_FAILS = (26, 30)
ALL_DEGREES = CHOLESKY_OK + CHOLESKY_FAILS

# The LOO metric cannot be more accurate than the factorisation that produced
# it, so the tolerance tracks cond(Phi) rather than being one number. Measured
# agreement against brute-force refits, with headroom:
#   deg 14  cond(Phi) 9.3e4   2.6e-07      deg 26  cond(Phi) 5.0e9   2.8e-04
#   deg 18  cond(Phi) 3.2e6   9.6e-10      deg 30  cond(Phi) 2.3e11  5.9e-02
#   deg 22  cond(Phi) 1.2e8   4.0e-08
LOO_RTOL = {14: 1e-6, 18: 1e-6, 22: 1e-6, 26: 2e-3, 30: 2e-1}


def _design(degree: int):
    """Return (Phi, M, nu, Y, z, multi_indices) for a 1-D monomial fit."""
    z = np.linspace(-1.0, 1.0, N_SAMPLES).reshape(-1, 1)
    mi = np.arange(degree + 1).reshape(-1, 1)
    Phi = MonomialPlan.build(mi).evaluate(z)
    # A noiseless target is fitted to round-off by degree 22, and then both
    # PRESS and the brute-force refits difference quantities at 1e-10: the
    # comparison stops measuring anything. Noise keeps the LOO metric above
    # the floor and makes it do its actual job (it rises with overfitting).
    noise = np.random.default_rng(12345).standard_normal((N_SAMPLES, 1))
    Y = np.sin(3.0 * z) + 0.25 * np.cos(7.0 * z) + 1e-3 * noise
    # press_loo takes the NORMALISED moments: M = Phi^T Phi / N, nu = Phi^T Y / N.
    # The leverage h_i = colsum_i / N is only the hat diagonal under that convention.
    return Phi, Phi.T @ Phi / N_SAMPLES, Phi.T @ Y / N_SAMPLES, Y, z, mi


def _cholesky_succeeds(M) -> bool:
    try:
        cho_factor(M, lower=False, check_finite=False)
    except np.linalg.LinAlgError:
        return False
    return True


def _bruteforce_loo(Phi, Y) -> float:
    """Leave-one-out RMSE by explicit refits, solved with SVD least squares."""
    n = Phi.shape[0]
    err = np.empty((n, Y.shape[1]))
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        c, *_ = np.linalg.lstsq(Phi[keep], Y[keep], rcond=None)
        err[i] = Y[i] - Phi[i] @ c
        keep[i] = True
    return float(np.sqrt(np.mean(err**2)))


@pytest.mark.parametrize("degree", ALL_DEGREES)
def test_cholesky_boundary_is_where_the_sweep_assumes(degree: int) -> None:
    """Pin the precondition, so a LAPACK change reports itself here."""
    _Phi, M, _nu, _Y, _z, _mi = _design(degree)
    assert _cholesky_succeeds(M) is (degree in CHOLESKY_OK)


@pytest.mark.parametrize("degree", ALL_DEGREES)
def test_press_loo_finite_across_the_cholesky_boundary(degree: int) -> None:
    Phi, M, nu, Y, _z, _mi = _design(degree)
    coeffs, cond, loo, per_out, lev = press_loo(M, nu, Phi, Y, on_singular="warn")
    assert np.isfinite(coeffs).all(), "coefficients must survive a failed Cholesky"
    assert np.isfinite(loo) and loo > 0.0
    assert np.isfinite(per_out).all()
    assert 0.0 < lev <= 1.0
    assert np.isfinite(cond)


@pytest.mark.parametrize("degree", ALL_DEGREES)
def test_press_loo_leverage_stays_exact(degree: int) -> None:
    """The QR factor must give the SAME leave-one-out error as explicit refits."""
    Phi, M, nu, Y, _z, _mi = _design(degree)
    _c, _cond, loo, _per, _lev = press_loo(M, nu, Phi, Y, on_singular="warn")
    assert loo == pytest.approx(_bruteforce_loo(Phi, Y), rel=LOO_RTOL[degree])


@pytest.mark.parametrize("degree", CHOLESKY_FAILS)
def test_press_loo_batched_path_also_recovers(degree: int) -> None:
    """The no-Phi path must rebuild the design and take the same route."""
    Phi, M, nu, Y, z, mi = _design(degree)
    coeffs, _cond, loo, _per, _lev = press_loo(
        M, nu, None, Y, on_singular="warn",
        plan=MonomialPlan.build(mi), X_scaled=z, batch_size=16,
    )
    assert np.isfinite(coeffs).all()
    assert loo == pytest.approx(_bruteforce_loo(Phi, Y), rel=LOO_RTOL[degree])


@pytest.mark.parametrize("degree", CHOLESKY_FAILS)
def test_press_loo_residual_matches_least_squares(degree: int) -> None:
    """The recovered coefficients must be the least-squares ones, not a guess."""
    Phi, M, nu, Y, _z, _mi = _design(degree)
    coeffs, *_ = press_loo(M, nu, Phi, Y, on_singular="warn")
    ref, *_ = np.linalg.lstsq(Phi, Y, rcond=None)

    def rms(C) -> float:
        return float(np.sqrt(np.mean((Phi @ C - Y) ** 2)))

    assert rms(coeffs) == pytest.approx(rms(ref), rel=1e-6)


def test_leverage_uses_the_qr_factor_not_the_cholesky_one() -> None:
    """Regression: degree 22 has a SUCCESSFUL Cholesky whose factor is junk.

    cond(M) there is 1.5e16. Taking the leverage from cho_factor(M) -- which
    does not raise -- put the reported LOO 12% off the brute-force value while
    reporting no failure at all. Only the QR factor gets it right, so this
    pins the route rather than merely pinning finiteness.
    """
    degree = 22
    Phi, M, nu, Y, _z, _mi = _design(degree)
    _c, _cond, loo, _per, _lev = press_loo(M, nu, Phi, Y, on_singular="warn")
    ref = _bruteforce_loo(Phi, Y)
    assert abs(loo - ref) / ref < 1e-5, (
        "the leverage fell back to the Cholesky factor; it was 1.2e-1 off before"
    )


def test_qr_route_is_refused_when_M_is_not_the_gram_matrix() -> None:
    """ridge and weights both change M, so a factor of Phi factors the wrong thing."""
    from MomentEmu.core import normal_equation_factor

    Phi, M, _nu, _Y, _z, _mi = _design(CHOLESKY_OK[-1])
    common = dict(n_samples=N_SAMPLES, cond=1e18, qr_at=1e13)
    _u, _l, route = normal_equation_factor(M, lambda: Phi, ridge=0.0, **common)
    assert route == "qr"
    for kwargs in ({"ridge": 1e-8}, {"weighted": True}):
        _u, _l, route = normal_equation_factor(M, lambda: Phi, **kwargs, **common)
        assert route == "cholesky", f"{kwargs} must not take a factor of Phi"


def test_qr_route_is_refused_when_the_design_is_underdetermined() -> None:
    """N < D makes R non-square, so it does not factor M at all."""
    from MomentEmu.core import normal_equation_factor

    rng = np.random.default_rng(7)
    Phi = rng.standard_normal((8, 20))
    M = Phi.T @ Phi / 8 + np.eye(20)  # positive definite so the Cholesky works
    _u, _l, route = normal_equation_factor(
        M, lambda: Phi, n_samples=8, cond=1e18, qr_at=1e13,
    )
    assert route == "cholesky"
