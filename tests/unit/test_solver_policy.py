"""P0.6: checked solve, condition estimates, NaN-safe selection, axis cap."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu import guards as g
from MomentEmu.MomentEmu import select_best_model, solve_emulator_coefficients
from MomentEmu.PolyEmu import PolyEmu


def test_solve_matches_numpy_on_well_conditioned():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    M = A.T @ A + 8 * np.eye(8)
    nu = rng.standard_normal((8, 3))
    got = solve_emulator_coefficients(M, nu)
    np.testing.assert_allclose(got, np.linalg.solve(M, nu), rtol=1e-12, atol=1e-12)


def test_direct_solve_raises_on_non_spd():
    M = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues -1, 3
    with pytest.raises(g.IllConditionedError):
        solve_emulator_coefficients(M, np.ones((2, 1)))


def test_direct_solve_raises_on_singular_and_returns_cond():
    M = np.diag([1.0, 1e-30])
    with pytest.raises(g.IllConditionedError):
        solve_emulator_coefficients(M, np.ones((2, 1)))
    rep = solve_emulator_coefficients(M, np.ones((2, 1)), warn_at=1e-40, raise_at=1e40, return_cond=True)
    c, cond = rep
    assert cond > 1e20


def test_warn_mode_returns_nan_and_inf_cond():
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.warns(g.IllConditionedWarning):
        c, cond = solve_emulator_coefficients(M, np.ones((2, 1)), on_singular="warn", return_cond=True)
    assert not np.isfinite(c).any()
    assert np.isinf(cond)


def test_fit_stores_cond_estimates():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (300, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3, dim_reduction=False)
    assert np.isfinite(emu.forward_cond_est_)
    assert emu.forward_cond_est_ > 0


def test_select_best_model_all_nan_raises():
    with pytest.raises(ValueError, match="non-finite RMSE"):
        select_best_model([np.nan, np.inf, np.nan])


def test_select_best_model_masks_nan():
    # The finite minimum is index 1; the NaN/inf entries must not be selected
    # and must not enter the min (round-1 select-best-model-nan-crash).
    assert select_best_model([np.nan, 0.5, 0.6, np.inf]) == 1
    assert select_best_model([np.inf, 0.5]) == 1


def test_count_axis_levels_and_cap():
    grid = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    caps, levels = g.check_axis_levels(grid)
    np.testing.assert_array_equal(levels, [2, 2])
    np.testing.assert_array_equal(caps, [1, 1])
    # Three levels on one axis.
    caps, levels = g.check_axis_levels(np.column_stack([np.repeat([0.0, 1.0, 2.0], 2), np.zeros(6)]))
    np.testing.assert_array_equal(levels, [3, 1])
    np.testing.assert_array_equal(caps, [2, 0])


def test_grid_fit_uses_axis_cap_and_warns():
    # 3 axes x 2 levels = 8 distinct rows; the cap drops the three x_i^2 terms
    # at degree 2 (D goes from 10 to 7), and the fit stays identifiable.
    from itertools import product

    grid = np.array(list(product([0.0, 1.0], repeat=3)))
    rng = np.random.default_rng(2)
    rep = rng.integers(0, 8, size=400)
    X = grid[rep]
    Y = (X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2] + 0.3 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    with pytest.warns(UserWarning, match="grid design detected"):
        emu = PolyEmu(
            X, Y, max_degree_forward=2, dim_reduction=False
        )
    assert emu.forward_multi_indices.shape[0] == 7
