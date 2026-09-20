"""T-008: the degree ceilings in PreconditionedEmu must be visible and movable.

Reported from a 21cmVAE-jax port. ``_affordable_degree`` carried
``cap_terms=6000`` as a function default no caller could reach, and it bound
silently: at 5 preconditioned dimensions with 24,562 rows, asking for degree
12 or 14 both returned 11, while degree 12 passes the data-derived half of
the test (6,188 terms x 3 = 18,564 <= 24,562). Only the constant blocked it.

The warning that did fire named the sample count, which was not what bound,
so a user reading it would have gone looking for more data they did not need.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.precondition import (
    PreconditionedEmu,
    _affordable_degree,
    _degree_limit,
)

# The reported case, reproduced exactly.
REPORTED = dict(n_dims=5, n_rows=24562)


def _design(n_samples: int = 2000, n: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, n))
    Y = np.tanh(X @ np.linspace(1.0, 2.0, n)).reshape(-1, 1)
    return X, Y


def test_the_cap_is_what_binds_in_the_reported_case() -> None:
    degree, reason, blocked = _degree_limit(ceiling=12, **REPORTED)
    assert (degree, reason) == (11, "cap")
    assert blocked == 6188


def test_the_sample_count_binds_once_the_cap_is_lifted() -> None:
    """Degree 14 was unreachable anyway; the cap cost exactly one degree."""
    degree, reason, _blocked = _degree_limit(ceiling=14, cap_terms=10**9, **REPORTED)
    assert (degree, reason) == (12, "samples")


def test_affordable_degree_keeps_its_signature() -> None:
    """recommend.py imports it in four places; the int return must not move."""
    assert _affordable_degree(5, 24562, 12) == 11
    assert _affordable_degree(5, 24562, 14, cap_terms=10**9) == 12


def test_cap_terms_reaches_the_constructor() -> None:
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tight = PreconditionedEmu(X, Y, order="none", max_degree_forward=8,
                                  cap_terms=50, RMSE_tol=0.0, verbose=0)
        loose = PreconditionedEmu(X, Y, order="none", max_degree_forward=8,
                                  cap_terms=10**9, RMSE_tol=0.0, verbose=0)
    assert tight.emulator.forward_degree < loose.emulator.forward_degree


def test_the_warning_names_the_cap_when_the_cap_binds() -> None:
    X, Y = _design()
    with pytest.warns(UserWarning, match="cap_terms") as record:
        PreconditionedEmu(X, Y, order="none", max_degree_forward=8,
                          cap_terms=50, RMSE_tol=0.0, verbose=0)
    text = " ".join(str(w.message) for w in record)
    assert "cap_terms=50" in text
    assert "samples" not in text.split("cap_terms")[0].split("order")[-1]


def test_the_warning_still_names_the_sample_count_when_that_binds() -> None:
    """The existing message must not be replaced by the new one."""
    X, Y = _design(n_samples=60)
    with pytest.warns(UserWarning, match="samples") as record:
        PreconditionedEmu(X, Y, order="none", max_degree_forward=8,
                          RMSE_tol=0.0, verbose=0)
    assert "cap_terms" not in " ".join(str(w.message) for w in record)


def test_report_says_which_degree_was_fitted() -> None:
    """scan_degree chose the order; this says what the model actually carries.

    Without it a caller who sets scan_degree=14 and gets a small model has
    nothing in the report that contradicts their reading of the name.
    """
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PreconditionedEmu(X, Y, order="none", scan_degree=14,
                                max_degree_forward=4, RMSE_tol=0.0, verbose=0)
    report = emu.report()
    assert report["fitted_degree"] == emu.emulator.forward_degree
    assert report["fitted_degree"] <= 4
    assert report["cap_terms"] == 6000


def test_report_shows_the_gap_between_scan_degree_and_the_fitted_degree() -> None:
    """The gap the report exists to make visible.

    scan_degree=14 scores the orders at the highest degree each can afford;
    the model is then fitted at whatever the inner estimator's own knobs
    allow, here pinned to 3. One number is not the other.
    """
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PreconditionedEmu(X, Y, order="auto", scan_degree=14,
                                max_degree_forward=3, RMSE_tol=0.0, verbose=0,
                                random_state=0)
    report = emu.report()
    scanned = report["scan_degrees"][report["order"]]
    assert scanned > report["fitted_degree"], (scanned, report["fitted_degree"])
    assert report["fitted_degree"] == 3
