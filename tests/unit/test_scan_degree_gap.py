"""A fitted degree far below scan_degree says so, instead of only being visible.

scan_degree selects the ORDER; it does not raise the degree of the model that
is finally fitted. That separation is deliberate and measured: wiring the two
together lengthens the ladder PolyEmu selects from, and its selection takes
the simplest rung within tolerance of the best, so on a sharp tanh at 3
parameters and 4,000 samples scan_degree=14 alone fitted degree 12 at 17.93
percent while adding max_degree_forward=14 fitted degree 10 at 20.24 percent.

So the fix is not to wire them. But a caller who sets scan_degree=14, gets a
degree-6 model and reads no complaint has no reason to look for the second
keyword, and on the 21cmGEM backend that gap was a factor of 3.4 in test
error. The gap is now reported when the caller left every inner degree knob
at its default -- which is the only case where they cannot have meant it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.precondition import PreconditionedEmu


def _smooth(n=900, seed=0, n_params=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, n_params))
    Y = np.column_stack([np.tanh(2.5 * X[:, 0]) + 0.4 * X[:, 1] ** 2,
                         np.sin(1.7 * X[:, 2])])
    return X, Y


def _fit(**kwargs):
    X, Y = _smooth()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PreconditionedEmu(X, Y, order="none", random_state=0, **kwargs)
    return model, [str(w.message) for w in caught]


def test_the_gap_is_reported_when_the_inner_knobs_are_untouched() -> None:
    model, messages = _fit(scan_degree=14)
    fitted = model.report()["fitted_degree"]
    assert fitted is not None and fitted < 14
    said = [m for m in messages if "scan_degree" in m]
    assert said, f"fitted degree {fitted} against scan_degree 14, silently"
    text = said[0]
    assert "max_degree_forward" in text and "init_deg_forward" in text
    assert "RMSE_tol" in text


def test_the_warning_does_not_tell_the_caller_to_just_wire_them() -> None:
    """The obvious fix is measured to be worse, so the text must not suggest it."""
    _model, messages = _fit(scan_degree=14)
    text = " ".join(m for m in messages if "scan_degree" in m)
    assert "measure" in text.lower()


@pytest.mark.parametrize(
    "knob", ("max_degree_forward", "init_deg_forward", "RMSE_tol"),
)
def test_no_complaint_once_the_caller_has_set_a_degree_knob(knob: str) -> None:
    """Setting any of them means the degree was chosen, not defaulted into."""
    value = {"max_degree_forward": 6, "init_deg_forward": 2, "RMSE_tol": 0.0}[knob]
    _model, messages = _fit(scan_degree=14, **{knob: value})
    assert not [m for m in messages if "scan_degree" in m]


def test_no_complaint_when_the_fit_reaches_the_ceiling() -> None:
    _model, messages = _fit(scan_degree=2)
    assert not [m for m in messages if "scan_degree" in m]


def test_no_complaint_for_an_estimator_without_a_degree() -> None:
    X, Y = _smooth()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PreconditionedEmu(X, Y, order="none", estimator="sparse",
                          scan_degree=14, n_terms=20, random_state=0)
    assert not [w for w in caught if "scan_degree" in str(w.message)]
