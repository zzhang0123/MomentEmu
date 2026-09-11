"""P0.2: seeded splits and strict X_test/Y_test pairing."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    Y = (X[:, 0] ** 2 + np.sin(X[:, 1]) + X[:, 2]).reshape(-1, 1)
    return X, Y


def test_same_random_state_gives_identical_fit(data):
    X, Y = data
    a = PolyEmu(X, Y, random_state=0, max_degree_forward=4)
    b = PolyEmu(X, Y, random_state=0, max_degree_forward=4)
    assert np.array_equal(a.forward_multi_indices, b.forward_multi_indices)
    assert np.array_equal(a.forward_coeffs, b.forward_coeffs)


def test_random_state_is_stored(data):
    X, Y = data
    emu = PolyEmu(X, Y, random_state=7, max_degree_forward=3)
    assert emu.random_state == 7


def test_lone_x_test_raises(data):
    X, Y = data
    with pytest.raises(ValueError, match="X_test"):
        PolyEmu(X, Y, X_test=X[:10])


def test_lone_y_test_raises(data):
    X, Y = data
    with pytest.raises(ValueError, match="Y_test"):
        PolyEmu(X, Y, Y_test=Y[:10])


def test_explicit_test_pair_accepted(data):
    X, Y = data
    emu = PolyEmu(X, Y, X_test=X[:20], Y_test=Y[:20], max_degree_forward=3)
    assert emu.forward_coeffs.shape[0] > 0
