"""P4.1: fit/predict API with the legacy constructor kept working."""
from __future__ import annotations

import numpy as np

from MomentEmu.emulator import PolyEmu


def _data(seed=0, n=3, N=300):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2])).reshape(-1, 1)
    return X, Y



def test_fit_refits_and_predict_matches_forward_emulator():
    X, Y = _data()
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    X2, Y2 = _data(seed=1)
    same = PolyEmu(X2, Y2, max_degree_forward=3, verbose=0)
    emu.fit(X2, Y2)
    assert np.array_equal(emu.forward_coeffs, same.forward_coeffs)
    pts = np.linspace(-0.8, 0.8, 30).reshape(10, 3)
    np.testing.assert_array_equal(
        emu.predict(pts, extrapolation="ignore"),
        emu.forward_emulator(pts, extrapolation="ignore"),
    )


def test_fit_inverse_and_predict_inverse():
    X, Y = _data(seed=2)
    emu = PolyEmu(X, Y, forward=True, backward=False, max_degree_forward=3, verbose=0)
    emu.fit_inverse(init_deg_backward=2, max_degree_backward=2, verbose=0)
    Yp = emu.forward_emulator(X[:10], extrapolation="ignore")
    np.testing.assert_array_equal(
        emu.predict_inverse(Yp, extrapolation="ignore"),
        emu.backward_emulator(Yp, extrapolation="ignore"),
    )


def test_legacy_positional_constructor_bit_identical_to_fit():
    X, Y = _data(seed=3)
    legacy = PolyEmu(X, Y, max_degree_forward=4, verbose=0)
    modern = PolyEmu(X, Y, max_degree_forward=4, verbose=0).fit(X, Y)
    assert np.array_equal(legacy.forward_coeffs, modern.forward_coeffs)
    assert np.array_equal(legacy.forward_multi_indices, modern.forward_multi_indices)
