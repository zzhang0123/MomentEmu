"""P1.1: dim_reduction is retired, filter_modes is deprecated, metrics describe the stored model."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.MomentEmu import filter_modes
from MomentEmu.PolyEmu import PolyEmu, evaluate_monomials_lazy


def _data(seed=0, n=3, N=300):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (np.exp(0.5 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2])).reshape(-1, 1)
    Xt = rng.uniform(-1.0, 1.0, (50, n))
    Yt = (np.exp(0.5 * Xt[:, 0]) + Xt[:, 1] ** 2 + np.sin(2 * Xt[:, 2])).reshape(-1, 1)
    return X, Y, Xt, Yt


def test_dim_reduction_true_is_ignored_with_one_warning():
    X, Y, Xt, Yt = _data()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        emu_pruned = PolyEmu(
            X, Y, X_test=Xt, Y_test=Yt, dim_reduction=True, max_degree_forward=3
        )
    dep = [w for w in rec if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1, [str(w.message) for w in dep]
    assert "dim_reduction" in str(dep[0].message)
    emu_plain = PolyEmu(X, Y, X_test=Xt, Y_test=Yt, dim_reduction=False, max_degree_forward=3)
    assert np.array_equal(emu_pruned.forward_coeffs, emu_plain.forward_coeffs)
    assert np.array_equal(emu_pruned.forward_multi_indices, emu_plain.forward_multi_indices)


def test_per_mode_thres_alone_warns():
    X, Y, Xt, Yt = _data(seed=1)
    with pytest.warns(DeprecationWarning, match="dim_reduction"):
        PolyEmu(X, Y, X_test=Xt, Y_test=Yt, per_mode_thres=1e-6, max_degree_forward=3)


def test_forward_RMSE_describes_stored_model():
    X, Y, Xt, Yt = _data(seed=2)
    emu = PolyEmu(
        X, Y, X_test=Xt, Y_test=Yt, dim_reduction=False, max_degree_forward=4
    )
    Xs = emu.scaler_X.transform(Xt)
    Ys = emu.scaler_Y.transform(Yt)
    pred = evaluate_monomials_lazy(Xs, emu.forward_multi_indices) @ emu.forward_coeffs
    rmse = float(np.sqrt(np.mean((pred - Ys) ** 2)))
    assert emu.forward_RMSE == pytest.approx(rmse, rel=1e-12, abs=1e-15)
    sel = emu.forward_degree_list.index(emu.forward_degree)
    assert emu.forward_RMSE == pytest.approx(emu.forward_RMSE_list[sel])


def test_filter_modes_warns():
    coeffs = np.ones((3, 1))
    M = np.eye(3)
    with pytest.warns(DeprecationWarning, match="filter_modes"):
        mask = filter_modes(coeffs, M, threshold=1e-3)
    assert mask.shape == (3,)


def test_rosenbrock_default_is_exact():
    from benchmarks.targets import TARGETS
    from benchmarks.harness import accuracy, fixed_degree_kwargs

    t = TARGETS["rosenbrock"]
    X, Y, Xt, Yt = t.data()
    emu = PolyEmu(X, Y, X_test=Xt, Y_test=Yt, **fixed_degree_kwargs(4))
    assert accuracy(emu.forward_emulator(Xt, extrapolation="ignore"), Yt)["nrmse"] < 1e-12
