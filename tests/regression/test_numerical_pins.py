"""P0.3: numerical gates that catch a real regression and tolerate timing noise.

These pins are exact for the selected degree and basis size, and relative for
the test RMSE (1e-6) and the leading coefficients (1e-10 of the largest
coefficient). The fit uses an explicit test set and explicit flags so later
selection / default changes do not move the pinned model.
"""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu

# Reference values generated on the CI Python 3.12 leg (see the P0.3 commit).
PIN = {
    "rmse": 0.4171145099884583,
    "degree": 2,
    "D": 10,
    "first_coeffs": [
        -0.5869641999028613,
        0.6509001097849724,
        -0.017413982023471417,
        0.6150228015299318,
        0.20974286493982733,
        -0.0012313619022376937,
        -0.0073163911566579915,
        0.33413342973672033,
        0.00658044191204465,
        0.04241854133990572,
    ],
    "max_abs_coeff": 0.6509001097849724,
}
RMSE_RTOL = 1e-6
COEFF_RTOL = 1e-10


def _fit():
    rng = np.random.default_rng(20260910)
    X = rng.uniform(-1.0, 1.0, (400, 3))
    Y = (np.exp(X[:, 0]) + X[:, 1] ** 2 + np.sin(3.0 * X[:, 2])).reshape(-1, 1)
    Xt = rng.uniform(-1.0, 1.0, (100, 3))
    Yt = (np.exp(Xt[:, 0]) + Xt[:, 1] ** 2 + np.sin(3.0 * Xt[:, 2])).reshape(-1, 1)
    emu = PolyEmu(
        X,
        Y,
        X_test=Xt,
        Y_test=Yt,
        max_degree_forward=5,
        RMSE_tol=1e3,
        dim_reduction=False,
        random_state=0,
    )
    return emu, Xt, Yt


def test_selected_degree_and_basis_exact():
    emu, _, _ = _fit()
    assert emu.forward_degree == PIN["degree"]
    assert emu.forward_multi_indices.shape[0] == PIN["D"]


def test_test_rmse_pin():
    emu, Xt, Yt = _fit()
    rmse = float(np.sqrt(np.mean((emu.forward_emulator(Xt) - Yt) ** 2)))
    assert rmse == pytest.approx(PIN["rmse"], rel=RMSE_RTOL)


def test_leading_coefficients_pin():
    emu, _, _ = _fit()
    got = emu.forward_coeffs[:10, 0]
    ref = np.asarray(PIN["first_coeffs"])
    assert np.max(np.abs(got - ref)) <= COEFF_RTOL * PIN["max_abs_coeff"]
