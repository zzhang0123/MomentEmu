"""P0.7: level-wise recursive monomial plan matches the lazy loop and the fast path."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import (
    PolyEmu,
    evaluate_monomials_lazy,
    generate_multi_indices,
)
from MomentEmu.monomials import evaluate_monomials_fast


def _index_set(n: int, D: int) -> np.ndarray:
    if n == 6:
        d = 5 if D == 462 else 8  # C(11,5)=462, C(14,8)=3003
        mi = generate_multi_indices(6, d)
        assert mi.shape[0] == D
        return mi
    # n = 1: exponents 0..D-1 (the plan still has to close and map them).
    return np.arange(D, dtype=np.int64).reshape(-1, 1)


def _column_scale_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("n,D", [(6, 462), (6, 3003), (1, 462), (1, 3003)])
@pytest.mark.parametrize("N", [1, 16, 256, 1000, 20000])
def test_recursive_matches_lazy(n, D, N):
    rng = np.random.default_rng(20260910)
    lo, hi = (0.9, 1.1) if n == 1 else (-1.0, 1.0)
    X = rng.uniform(lo, hi, (N, n))
    mi = _index_set(n, D)
    ref = evaluate_monomials_lazy(X, mi)
    got = evaluate_monomials_fast(X, mi)
    assert got.shape == ref.shape
    assert _column_scale_rel(got, ref) < 1e-14


def test_recursive_matches_lazy_pruned_subset():
    rng = np.random.default_rng(1)
    mi = generate_multi_indices(4, 6)
    keep = rng.random(mi.shape[0]) > 0.4
    keep[0] = True
    pruned = mi[keep]
    assert pruned.shape[0] < mi.shape[0]
    X = rng.uniform(-1.0, 1.0, (500, 4))
    ref = evaluate_monomials_lazy(X, pruned)
    got = evaluate_monomials_fast(X, pruned)
    assert _column_scale_rel(got, ref) < 1e-14


def _ref_forward(emu: PolyEmu, X: np.ndarray) -> np.ndarray:
    Xs = emu.scaler_X.transform(X)
    Ys = evaluate_monomials_lazy(Xs, emu.forward_multi_indices) @ emu.forward_coeffs
    Y = emu.scaler_Y.inverse_transform(Ys)
    if emu.log_Y:
        Y = np.exp(Y)
    return Y


def _fit(seed: int, **kw) -> PolyEmu:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (400, 3))
    Y = np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 5.0
    return PolyEmu(
        X,
        Y.reshape(-1, 1),

        max_degree_forward=4,
        dim_reduction=False,
        random_state=0,
        **kw,
    )


def _test_points(emu: PolyEmu, rng) -> np.ndarray:
    xmin = emu.scaler_X.mean_ - 2 * emu.scaler_X.scale_
    xmax = emu.scaler_X.mean_ + 2 * emu.scaler_X.scale_
    corners = np.array(np.meshgrid(*zip(xmin, xmax))).T.reshape(-1, emu.n_params)
    pts = [emu.scaler_X.mean_[None, :], corners]
    pts.append(rng.uniform(xmin, xmax, (1000 - 1 - len(corners), emu.n_params)))
    return np.vstack(pts)


@pytest.mark.parametrize("log_Y", [False, True])
@pytest.mark.parametrize("with_std", [True, False])
def test_forward_fast_matches_lazy(log_Y, with_std):
    emu = _fit(2, log_Y=log_Y, standardize_Y_with_std=with_std)
    rng = np.random.default_rng(3)
    X = _test_points(emu, rng)
    ref = _ref_forward(emu, X)
    got = emu.forward_emulator(X, extrapolation="ignore")
    assert _column_scale_rel(got, ref) < 1e-13


def test_integer_and_float32_inputs_give_float64():
    emu = _fit(4)
    X = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    out_i = emu.forward_emulator(X, extrapolation="ignore")
    assert out_i.dtype == np.float64
    out_f = emu.forward_emulator(X.astype(np.float32), extrapolation="ignore")
    assert out_f.dtype == np.float64
    np.testing.assert_allclose(out_i, out_f, rtol=1e-12)


def test_fitted_coefficients_unchanged_by_fast_path():
    emu = _fit(5)
    before = emu.forward_coeffs.copy()
    emu.forward_emulator(np.array([[0.1, -0.2, 0.3]]), extrapolation="ignore")
    assert np.array_equal(before, emu.forward_coeffs)
