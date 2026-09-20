"""T-003: box scaling of the inputs.

A polynomial basis wants a bounded argument. Dividing by the standard
deviation does not bound one, and a coordinate that is a weighted sum of
several inputs reaches far more standard deviations than a single uniform
parameter does.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.guards import BoxScaler
from MomentEmu.monomials import MonomialPlan


def test_box_scaler_maps_the_training_range_onto_the_unit_box():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(3.0, 7.0, 500), rng.uniform(-2.0, 0.5, 500)])
    Z = BoxScaler().fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(Z.max(axis=0), [1.0, 1.0])


def test_box_scaler_round_trips_and_exposes_the_standard_interface():
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, (300, 3))
    s = BoxScaler().fit(X)
    np.testing.assert_allclose(s.inverse_transform(s.transform(X)), X, atol=1e-12)
    # same (x - mean_) / scale_ form as StandardScaler, which is what lets a
    # stored emulator reload through io.ArrayScaler unchanged
    np.testing.assert_allclose(s.transform(X), (X - s.mean_) / s.scale_)
    np.testing.assert_allclose(s.var_, s.scale_ ** 2)


def test_box_scaler_handles_a_constant_column():
    X = np.column_stack([np.full(100, 2.5), np.linspace(-1, 1, 100)])
    Z = BoxScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


def test_box_scaler_refuses_to_transform_before_fitting():
    with pytest.raises(AttributeError, match="not been fitted"):
        BoxScaler().transform(np.zeros((2, 2)))


def test_scaling_is_validated():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (500, 2))
    with pytest.raises(ValueError, match="scaling must be"):
        PolyEmu(X, (X[:, 0] ** 2).reshape(-1, 1), scaling="sideways",
                max_degree_forward=3, verbose=0)


def test_both_scalings_agree_where_conditioning_is_not_the_limit():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (3000, 3))
    Y = (np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]).reshape(-1, 1)
    err = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mode in ("standard", "box"):
            e = PolyEmu(X, Y, scaling=mode, max_degree_forward=6, verbose=0)
            p = e.forward_emulator(X[:500], extrapolation="ignore")
            err[mode] = float(np.sqrt(np.mean((p - Y[:500]) ** 2)))
    assert err["box"] == pytest.approx(err["standard"], rel=0.05)


def test_box_scaling_survives_save_and_load(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.uniform(-3, 4, (2000, 3))
    Y = (np.sin(X[:, 0]) + X[:, 1] * X[:, 2]).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, Y, scaling="box", max_degree_forward=5, verbose=0)
        before = emu.forward_emulator(X[:200], extrapolation="ignore")
        path = tmp_path / "boxed.npz"
        from MomentEmu.io import load_emulator, save_emulator

        save_emulator(emu, path)
        after = load_emulator(path).forward_emulator(X[:200], extrapolation="ignore")
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
def test_box_scaling_rescues_a_coordinate_that_reaches_many_sigma():
    """The case it exists for. A coordinate built as a weighted sum of seven
    inputs is bell-shaped and reaches several standard deviations, so a high
    power of it spans an enormous range once the scaling is by sigma. On the
    21cmGEM benchmark a rotated coordinate reached 11 sigma and degree 12
    under standard scaling scored WORSE than degree 10."""
    rng = np.random.default_rng(3)
    raw = rng.uniform(-1, 1, (9000, 7))
    w = np.array([0.6, 0.5, -0.4, 0.3, 0.2, -0.15, 0.1])
    Z = np.column_stack([raw @ w, raw @ np.roll(w, 3), raw[:, 0]])
    Y = np.tanh(1.8 * Z[:, 0] + 0.5 * Z[:, 1]).reshape(-1, 1)

    reach = float(np.max(np.abs(Z[:, 0] / Z[:, 0].std())))
    assert reach > 3.0, f"probe coordinate only reaches {reach:.1f} sigma"

    def cond_at(mode, degree):
        if mode == "standard":
            m, s = Z.mean(0), Z.std(0)
            A = (Z - m) / s
        else:
            lo, hi = Z.min(0), Z.max(0)
            A = 2 * (Z - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
        P = MonomialPlan.build(generate_multi_indices(3, degree)).evaluate(A)
        return float(np.linalg.cond(P.T @ P))

    # The gain needs both a long reach and a high degree. This coordinate
    # reaches only 3.5 sigma, a third of the benchmark's, so the crossover sits
    # near degree 12 and the box map is the WORSE choice below it.
    assert cond_at("box", 8) > cond_at("standard", 8)
    assert cond_at("box", 16) < cond_at("standard", 16) / 100.0
