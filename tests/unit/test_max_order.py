"""P0.5: max_supported_degree, the D >= N refusal and the backward cap."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.guards import basis_size, max_supported_degree
from MomentEmu.PolyEmu import PolyEmu

# The round-1 cells in which the old max_order returned D >= N.
ROUND1_CELLS = [
    (1, 5), (1, 3000), (2, 170), (2, 200), (4, 1000), (4, 10000),
    (6, 20000), (7, 3432), (8, 100), (3, 20),
]


@pytest.mark.parametrize("n,N", ROUND1_CELLS)
def test_fill1_degree_is_identifiable(n, N):
    k = max_supported_degree(n, N, fill=1.0)
    assert basis_size(n, k) < N
    assert basis_size(n, k + 1) >= N


@pytest.mark.parametrize(
    "n,N,fill,expected",
    [
        (4, 1000, 2.0, 8),
        (6, 20000, 2.0, 10),
        (4, 1000, 1.0, 9),
        (6, 20000, 1.0, 12),
        (1, 1, 1.0, 0),
    ],
)
def test_max_supported_degree_pins(n, N, fill, expected):
    assert max_supported_degree(n, N, fill=fill) == expected


@pytest.mark.parametrize("N,expected", [(20, 4), (21, 4), (22, 5)])
def test_boundary_around_D_equals_N(N, expected):
    # n = 2: basis_size(2, 5) = 21.
    assert basis_size(2, 5) == 21
    assert max_supported_degree(2, N, fill=1.0) == expected


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        max_supported_degree(0, 10)
    with pytest.raises(ValueError):
        max_supported_degree(2, 10, fill=0)


def test_explicit_degree_with_D_ge_N_raises():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (150, 4))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    with pytest.raises(ValueError) as excinfo:
        PolyEmu(X, Y, cross_validation=False, max_degree_forward=6)
    msg = str(excinfo.value)
    assert "210" in msg  # basis_size(4, 6)
    assert "150" in msg  # N_train


def test_auto_cap_warns_and_stays_identifiable():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    with pytest.warns(UserWarning, match="auto-capped max_degree_forward"):
        emu = PolyEmu(X, Y, cross_validation=False)
    D = emu.forward_multi_indices.shape[0]
    assert D * 2 <= X.shape[0]
    assert D < X.shape[0]


def test_backward_cap_below_init_raises():
    rng = np.random.default_rng(2)
    N, m = 2000, 2000
    X = rng.uniform(-1.0, 1.0, (N, 1))
    Y = rng.uniform(0.0, 1.0, (N, m))
    # max_supported_degree(2000, 2000, fill=2) == 0: degree 1 already needs
    # D = 2001 with a 2x margin, so the backward sweep cannot start.
    assert max_supported_degree(m, N, fill=2.0) == 0
    with pytest.raises(ValueError, match="init_deg_backward"):
        PolyEmu(X, Y, forward=False, backward=True, cross_validation=False)


def test_backward_fit_stays_under_half_N():
    rng = np.random.default_rng(3)
    N, m = 4000, 12
    X = rng.uniform(-1.0, 1.0, (N, 1))
    Y = rng.uniform(0.0, 1.0, (N, m))
    emu = PolyEmu(X, Y, forward=False, backward=True, cross_validation=False)
    D = emu.backward_multi_indices.shape[0]
    assert D * 2 <= N
