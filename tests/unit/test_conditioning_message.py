"""T-009: the cond(M) warning must not send a predictions-only user to a fix.

The remedy sentence used to read "Consider a lower degree, more samples, or
an orthonormal basis." with nothing saying who it was for. Measured on a
21cmVAE-jax target in rank-5 rotated coordinates, taking that advice made the
result worse: chebyshev at degree 14 scored 1.6251 percent against monomial
0.9647 percent, while carrying a condition number 95 times better.

The threshold's own calibration already says which consequence it is about.
Below cond(M) ~ 1e16 fresh-point prediction error stays at or below 1e-8,
while the coefficient relative error crosses 1e-6 somewhere between 8.1e12
and 4.4e16. The warning is a statement about the coefficients, and it now
says so before it suggests anything.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.guards import COND_WARN, IllConditionedWarning, check_conditioning


def _ill_conditioned(cond_target: float = 1e13) -> np.ndarray:
    """A symmetric positive-definite M with roughly the requested cond."""
    n = 6
    eigs = np.geomspace(1.0, cond_target, n)
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((n, n)))
    return q @ np.diag(eigs) @ q.T


def _message() -> str:
    with pytest.warns(IllConditionedWarning) as record:
        check_conditioning(_ill_conditioned(), warn_at=COND_WARN,
                           on_singular="warn")
    return " ".join(str(w.message) for w in record)


def test_the_warning_still_reports_cond_and_the_digits_left() -> None:
    text = _message()
    assert "cond(M)" in text
    assert "digits" in text


def test_it_names_the_consequence_before_the_remedy() -> None:
    """Coefficients are affected; predictions usually are not."""
    text = _message()
    lowered = text.lower()
    assert "coefficients" in lowered
    assert "symbolic" in lowered
    assert "predictions" in lowered
    # The point of the rewrite: the coefficient consequence is stated before
    # anything is suggested, so a reader who only wants predictions stops.
    assert lowered.index("predictions") < lowered.index("basis_kind")


def test_it_does_not_recommend_a_basis_switch_unconditionally() -> None:
    """The old text did, and on a real target that cost accuracy."""
    text = _message()
    assert "Consider a lower degree, more samples, or an orthonormal basis." not in text
    # Any basis suggestion must be tied to needing the coefficients, and must
    # not be phrased as free.
    assert "measure" in text.lower()


def test_a_healthy_matrix_still_warns_about_nothing() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_conditioning(np.eye(4), warn_at=COND_WARN, on_singular="warn")
