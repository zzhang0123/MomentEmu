"""Analytic derivative of a :class:`~MomentEmu.warp.Warp` (T-005).

A rotation that follows a warp is computed in the warped coordinates, while a
caller's model differentiates in the raw ones. The chain rule between them is
``dY/du = (dY/dx) / (du/dx)``, so the rotation is only as right as this
derivative is. Getting it wrong is not loud: the rotation is still a set of
orthonormal vectors with a plausible spectrum, it just points elsewhere.

Every case here is checked against a central difference of ``Warp.__call__``
itself, so the two can only agree if the analytic form differentiates the map
the package actually applies -- clamps included.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.warp import Warp, _make

# (pre, shape, param, centre) covering every family and both pre-transforms,
# including the log + shape combination the scan never proposes but the
# dataclass allows.
CASES = [
    ("identity", "identity", 0.0, 0.0),
    ("log", "identity", 0.0, 0.0),
    *[("identity", "sinh", b, c) for b in (1.0, 3.0, 10.0, 30.0)
      for c in (-0.5, 0.0, 0.5)],
    *[("identity", "kte", a, 0.0) for a in (0.5, 0.9, 0.99)],
    ("log", "sinh", 10.0, 0.25),
    ("log", "kte", 0.9, 0.0),
]


def _column(pre: str) -> np.ndarray:
    """A training column for this pre-transform; log needs it positive."""
    if pre == "log":
        return np.exp(np.linspace(np.log(0.01), np.log(30.0), 64))
    return np.linspace(-4.0, 6.0, 64)


def _central_difference(warp: Warp, x: np.ndarray) -> np.ndarray:
    h = 1e-6 * np.maximum(np.abs(x), 1.0)
    return (warp(x + h) - warp(x - h)) / (2.0 * h)


@pytest.mark.parametrize("pre,shape,param,centre", CASES)
def test_derivative_matches_a_central_difference(pre, shape, param, centre) -> None:
    col = _column(pre)
    warp = _make(pre, shape, param, centre, col)
    # Interior points only: a central difference straddles the clamp at the
    # ends of the box, where the analytic derivative is one-sided by design.
    x = col[2:-2]
    got, ref = warp.derivative(x), _central_difference(warp, x)
    assert np.allclose(got, ref, rtol=2e-5, atol=1e-8), (
        f"{warp.spec()}: max rel "
        f"{np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)):.3e}"
    )


@pytest.mark.parametrize("pre,shape,param,centre", CASES)
def test_derivative_is_positive_and_finite_on_the_box(pre, shape, param, centre) -> None:
    """Monotone and non-degenerate, so dividing a Jacobian by it is safe."""
    col = _column(pre)
    d = _make(pre, shape, param, centre, col).derivative(col)
    assert np.all(np.isfinite(d))
    assert np.all(d > 0.0), f"{shape}: min derivative {d.min():.3e}"


def test_kte_derivative_is_zero_where_the_map_saturates() -> None:
    """``__call__`` clips, so beyond the clip the map is flat, not infinite."""
    col = np.linspace(-1.0, 1.0, 32)
    warp = _make("identity", "kte", 0.9, 0.0, col)
    outside = np.array([-4.0, -2.0, 2.0, 4.0])     # |a v| > 1 well past the box
    assert np.allclose(warp.derivative(outside), 0.0)
    assert np.all(np.isfinite(warp(outside)))


def test_log_derivative_is_zero_at_the_clamp() -> None:
    """Below ``tiny`` the log pre-transform is constant, so the slope is zero."""
    col = np.exp(np.linspace(np.log(0.1), np.log(10.0), 32))
    warp = _make("log", "identity", 0.0, 0.0, col)
    assert warp.derivative(np.array([0.0, -1.0]))[0] == 0.0
    assert np.allclose(warp.derivative(np.array([0.0, -1.0])), 0.0)


def test_derivative_carries_the_units_of_the_axis() -> None:
    """d/dx of an affine map onto [-1, 1] is 2 / span, so it scales with it."""
    wide = _make("identity", "identity", 0.0, 0.0, np.linspace(0.0, 100.0, 8))
    narrow = _make("identity", "identity", 0.0, 0.0, np.linspace(0.0, 1.0, 8))
    x = np.array([0.5])
    assert np.isclose(wide.derivative(x)[0], 2.0 / 100.0)
    assert np.isclose(narrow.derivative(x)[0], 2.0 / 1.0)
