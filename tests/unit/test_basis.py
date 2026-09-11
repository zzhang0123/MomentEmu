"""P5.3: anisotropic / structured index sets (Basis)."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.basis import Basis
from MomentEmu.emulator import PolyEmu, generate_multi_indices


def _names(n):
    return [f"x{i}" for i in range(n)]


@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize("d", [0, 2, 4])
def test_total_degree_is_row_identical(n, d):
    got = Basis.total_degree().build(_names(n), d)
    ref = generate_multi_indices(n, d)
    assert np.array_equal(got, ref)


def test_constraints_compose_by_intersection():
    names = _names(4)
    b = Basis(max_interaction=2, per_parameter=(3, 1, 1, 1), degree=5)
    mi = b.build(names, 5)
    assert np.all(mi.sum(axis=1) <= 5)
    assert np.all(np.count_nonzero(mi, axis=1) <= 2)
    assert np.all(mi <= np.array([3, 1, 1, 1]))
    # A group limit restricts the sum inside the group.
    g = Basis(groups=(([0, 1], 2),), degree=5)
    mg = g.build(names, 5)
    assert np.all(mg[:, [0, 1]].sum(axis=1) <= 2)


def test_spec_is_copy_pasteable():
    b = Basis(q=0.5, weights=(1, 2, 8, 8, 4, 4), max_interaction=2, degree=16)
    spec = b.spec()
    assert spec.startswith("Basis(") and "q=0.5" in spec and "max_interaction=2" in spec


@pytest.mark.slow
def test_f1_q05_anisotropic_beats_isotropic():
    rng = np.random.default_rng(20260910)
    N = 20000
    X = rng.uniform(-1.0, 1.0, (N, 6))
    Xt = rng.uniform(-1.0, 1.0, (10000, 6))

    def f1(A):
        return (
            np.tanh(4.0 * A[:, 0]) + 0.5 * np.exp(A[:, 1]) + 0.1 * A[:, 2]
            + 0.1 * A[:, 3] + 0.05 * A[:, 4] * A[:, 5]
        )[:, None]

    Y, Yt = f1(X), f1(Xt)
    basis = Basis(q=0.5, weights=(1, 2, 8, 8, 4, 4), max_interaction=2, degree=16)
    emu = PolyEmu(
        X, Y, X_test=Xt, Y_test=Yt, basis=basis,
        init_deg_forward=16, max_degree_forward=16, RMSE_tol=1e-300, verbose=0,
    )
    D = emu.forward_multi_indices.shape[0]
    assert D == 67, D  # the q=0.5, max_interaction=2 set at d=16 has 67 terms
    pred = emu.forward_emulator(Xt, extrapolation="ignore")
    rel = float(np.sqrt(np.mean((pred - Yt) ** 2))) / float(np.abs(Yt).max())
    assert rel < 6e-4, rel


def test_report_returns_per_parameter_degree():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (500, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    info = emu.report()
    assert info["n_terms"] == emu.forward_multi_indices.shape[0]
    assert len(info["per_parameter_degree"]) == 3
