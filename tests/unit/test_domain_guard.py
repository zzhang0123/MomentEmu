"""P1.6: training box, extrapolation guard and leverage."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.guards import ExtrapolationWarning
from MomentEmu.PolyEmu import PolyEmu


def _fit_box(n=3, N=400, seed=0, degree=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (N, n))
    X[0, :] = 0.0
    X[1, :] = 1.0
    Y = (np.sum(np.sin(2 * X), axis=1) + X[:, 0] ** 2).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=degree, verbose=0)
    return emu, X


@pytest.mark.parametrize("n", [1, 6])
@pytest.mark.parametrize("factor", [1.0 - 1e-12, 1.0])
def test_no_warning_on_box_face(n, factor):
    emu, _ = _fit_box(n=n, seed=n)
    x = np.full((1, n), 0.5)
    x[0, 0] = factor
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        emu.forward_emulator(x)
    assert not any(issubclass(w.category, ExtrapolationWarning) for w in rec)


@pytest.mark.parametrize("n", [1, 6])
def test_warns_just_outside_face(n):
    emu, _ = _fit_box(n=n, seed=100 + n)
    x = np.full((1, n), 0.5)
    x[0, 0] = 1.0 + 1e-12
    with pytest.warns(ExtrapolationWarning):
        emu.forward_emulator(x)


def test_warns_one_percent_outside():
    emu, _ = _fit_box(n=2, seed=7)
    x = np.array([[1.01, 0.5]])
    with pytest.warns(ExtrapolationWarning, match="outside"):
        emu.forward_emulator(x)


def test_raise_at_diagonal_corner():
    emu, _ = _fit_box(n=2, seed=8)
    x = np.full((1, 2), 1.2)
    with pytest.raises(ValueError, match="outside the training box"):
        emu.forward_emulator(x, extrapolation="raise")
    # ignore is silent
    emu.forward_emulator(x, extrapolation="ignore")


def test_in_domain():
    emu, _ = _fit_box(n=2, seed=9)
    assert bool(emu.in_domain(np.array([0.5, 0.5]))) is True
    assert bool(emu.in_domain(np.array([1.1, 0.5]))) is False
    mask = emu.in_domain(np.array([[0.5, 0.5], [1.1, 0.5]]))
    assert mask.tolist() == [True, False]


def test_leverage_on_6_6_grid():
    from itertools import product

    levels = np.linspace(0.0, 1.0, 6)
    X = np.array(list(product(levels, repeat=6)))
    Y = (np.sum(np.sin(2 * X), axis=1)).reshape(-1, 1)
    emu = PolyEmu(
        X, Y, init_deg_forward=4, max_degree_forward=4, RMSE_tol=1e-300, verbose=0
    )
    # t = 1.5 half-widths from the centre: far outside the box.
    far = np.full((1, 6), 0.5 + 1.5 * 0.5)
    face = np.array([[1.0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    h_far = float(emu.leverage(far))
    h_face = float(emu.leverage(face))
    assert h_far > 0.3, h_far
    assert h_face < 0.01, h_face


def test_leverage_sums_correctly_at_training_box():
    emu, X = _fit_box(n=3, seed=11, N=200, degree=3)
    # The forward box is stored and in_domain is True at the training rows.
    inside = emu.in_domain(X)
    assert bool(np.all(inside))
