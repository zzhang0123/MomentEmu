"""cond(Phi) is reported, never used to refuse a fit.

cond(M) = cond(Phi)^2, and the accuracy of a QR solve depends on cond(Phi).
But above cond(M) ~ 1e16 the smallest eigenvalue of the COMPUTED M is
round-off, so the cond(M) estimate saturates near 1/eps and stops carrying any
information about Phi. On a 1-D monomial design at N = 200:

    true cond(Phi)   sqrt(cond(M) estimate)
    1.08e+11         5.89e+08
    1.25e+14         5.79e+08      <- no longer tracks anything

So the diagnostic needs its own estimate, and LAPACK's triangular condition
estimator on R gives one for the price of an O(D^2) call.

A ceiling on that estimate was tried and removed. It refused fits that were
fine -- brute-force LOO RMSE on this design is 2.90e-03 at cond(Phi) = 3.7e12
and 2.26e-03 at 4.4e15, so the unusable region is not where cond alone puts it
-- and it could only refuse AFTER the factorisation, so it saved no work while
discarding a model the caller's own held-out or LOO error was about to judge
on the evidence. These tests pin the reporting AND the non-refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.core import press_loo, triangular_cond
from MomentEmu.guards import COND_PHI_WARN, IllConditionedWarning
from MomentEmu.monomials import MonomialPlan

N_SAMPLES = 200
# true cond(Phi) at N = 200: 26 -> 3.2e9, 30 -> 1.1e11, 34 -> 3.7e12,
# 38 -> 1.3e14, 42 -> 4.4e15.
QUIET = (26, 30)
LOUD = (34, 38, 42)
ALL_DEGREES = QUIET + LOUD


def _design(degree: int):
    z = np.linspace(-1.0, 1.0, N_SAMPLES).reshape(-1, 1)
    mi = np.arange(degree + 1).reshape(-1, 1)
    Phi = MonomialPlan.build(mi).evaluate(z)
    noise = np.random.default_rng(5).standard_normal((N_SAMPLES, 1))
    Y = np.sin(3.0 * z) + 1e-3 * noise
    return Phi, Phi.T @ Phi / N_SAMPLES, Phi.T @ Y / N_SAMPLES, Y


def _true_cond(Phi) -> float:
    sv = np.linalg.svd(Phi, compute_uv=False)
    return float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")


def _bruteforce_loo(Phi, Y) -> float:
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
def test_estimate_tracks_the_true_cond_phi(degree: int) -> None:
    """Measured ratio runs 1.92 to 2.21 over eight decades."""
    Phi, _M, _nu, _Y = _design(degree)
    ratio = triangular_cond(np.linalg.qr(Phi, mode="r")) / _true_cond(Phi)
    assert 1.0 <= ratio <= 5.0, f"degree {degree}: est/true = {ratio:.3g}"


@pytest.mark.parametrize("degree", LOUD)
def test_the_cond_m_estimate_cannot_do_this_job(degree: int) -> None:
    """Pins WHY a separate estimate exists, not merely that it does."""
    Phi, M, _nu, _Y = _design(degree)
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    from_m = np.sqrt(np.abs(ev).max() / max(np.abs(ev).min(), 1e-300))
    true = _true_cond(Phi)
    if true > 1e13:
        assert from_m < true / 100.0, "cond(M) stopped saturating; revisit this"


@pytest.mark.parametrize("degree", ALL_DEGREES)
def test_a_high_cond_phi_is_never_a_refusal(degree: int) -> None:
    """The regression guard: reporting must not become rejecting.

    Every degree here -- including cond(Phi) = 4.4e15 -- must come back with a
    finite model and a finite LOO score, because that score is what decides
    whether the fit is usable.
    """
    Phi, M, nu, Y = _design(degree)
    with pytest.warns() if degree in LOUD else _no_warning_required():
        coeffs, _cond, loo, _per, _lev = press_loo(M, nu, Phi, Y, on_singular="warn")
    assert np.isfinite(coeffs).all()
    assert np.isfinite(loo) and loo > 0.0


def _no_warning_required():
    import contextlib

    return contextlib.nullcontext()


@pytest.mark.parametrize("degree", LOUD)
def test_the_warning_names_cond_phi_and_the_digits_left(degree: int) -> None:
    Phi, M, nu, Y = _design(degree)
    assert triangular_cond(np.linalg.qr(Phi, mode="r")) >= COND_PHI_WARN
    with pytest.warns(IllConditionedWarning, match=r"cond\(Phi\) = .*digits"):
        press_loo(M, nu, Phi, Y, on_singular="warn")


@pytest.mark.parametrize("degree", QUIET)
def test_no_cond_phi_warning_when_the_solve_has_digits(degree: int) -> None:
    Phi, M, nu, Y = _design(degree)
    assert triangular_cond(np.linalg.qr(Phi, mode="r")) < COND_PHI_WARN
    with warnings_as_errors_for_cond_phi():
        press_loo(M, nu, Phi, Y, on_singular="warn")


def warnings_as_errors_for_cond_phi():
    import warnings as _w
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            yield
        assert not [w for w in caught if "cond(Phi)" in str(w.message)]

    return _ctx()


def test_cond_alone_does_not_locate_the_unusable_region() -> None:
    """The measurement that killed the ceiling, kept so it is not re-added.

    If a cond(Phi) ceiling were reinstated between these two degrees it would
    refuse the BETTER of the two fits.
    """
    Phi_lo, *_ = _design(34)
    Phi_hi, _M, _nu, Y = _design(42)
    assert _true_cond(Phi_hi) > 100.0 * _true_cond(Phi_lo)
    loo_lo = _bruteforce_loo(Phi_lo, _design(34)[3])
    loo_hi = _bruteforce_loo(Phi_hi, Y)
    assert loo_hi < loo_lo, (
        f"the worse-conditioned fit is no longer the better one "
        f"({loo_hi:.3e} vs {loo_lo:.3e}); re-read the ceiling argument"
    )


def test_triangular_cond_rejects_a_non_square_factor() -> None:
    with pytest.raises(ValueError, match="square"):
        triangular_cond(np.zeros((3, 5)))


def test_triangular_cond_of_a_singular_factor_is_infinite() -> None:
    R = np.triu(np.ones((4, 4)))
    R[2, 2] = 0.0
    assert triangular_cond(R) == float("inf")
