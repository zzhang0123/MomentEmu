"""T-002 / P4: per-parameter monotone input warping.

P1 to P3 cut the term count at a fixed convergence rate. This moves the rate,
which is why it multiplies with them rather than overlapping.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu
from MomentEmu.warp import FAMILIES, Warp, WarpedEmu, fit_warps


def _decades(rng, n=9000):
    X = np.column_stack([
        10.0 ** rng.uniform(-3, 0, n),
        10.0 ** rng.uniform(-2, 1, n),
        rng.uniform(-1, 1, n),
        rng.uniform(-1, 1, n),
    ])
    Y = (np.log10(X[:, 0]) * np.sin(2.0 * np.log10(X[:, 1]))
         + 0.5 * X[:, 2] * X[:, 3]).reshape(-1, 1)
    return X, Y


def _sharp(rng, n=9000):
    X = rng.uniform(-1, 1, (n, 4))
    Y = (np.tanh(20.0 * (X[:, 0] - 0.3)) + 0.6 * np.tanh(8.0 * (X[:, 1] + 0.4))
         + 0.3 * X[:, 2] * X[:, 3]).reshape(-1, 1)
    return X, Y


def test_the_rate_is_set_by_the_complex_singularity():
    """The mechanism, pinned. tanh(20(x-0.3)) has poles at 0.3 + i pi/40, and
    the Bernstein ellipse through them predicts the measured per-degree rate."""
    z0 = 0.3 + 1j * np.pi / 40.0
    rho = abs(z0 + np.sqrt(z0 ** 2 - 1.0))
    predicted = 1.0 / rho

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 6000)
    xt = np.linspace(-0.999, 0.999, 3000)
    y, yt = np.tanh(20.0 * (x - 0.3)), np.tanh(20.0 * (xt - 0.3))
    degrees = (12, 16, 20, 24, 28, 32)
    errs = []
    for d in degrees:
        V = np.polynomial.legendre.legvander(x, d)
        c = np.linalg.lstsq(V, y, rcond=None)[0]
        errs.append(np.sqrt(np.mean(
            (np.polynomial.legendre.legvander(xt, d) @ c - yt) ** 2)))
    measured = float(np.exp(np.polyfit(degrees, np.log(errs), 1)[0]))
    assert abs(measured - predicted) < 0.01, (measured, predicted)


def test_recovers_log_on_decades_spanning_axes():
    X, Y = _decades(np.random.default_rng(0))
    warps = fit_warps(X, Y, random_state=0)
    assert [w.pre for w in warps] == ["log", "log", "identity", "identity"]


def test_a_warp_accepted_in_one_pass_is_not_reverted_in_the_next():
    """Regression. Scoring an axis against the CURRENT selection credits its
    own warp to the identity, so a warp accepted in pass one was undone in
    pass two and only one of the two log axes ever survived."""
    X, Y = _decades(np.random.default_rng(0))
    for passes in (1, 2, 3, 5):
        warps = fit_warps(X, Y, random_state=0, n_passes=passes)
        n_log = sum(w.pre == "log" for w in warps)
        assert n_log >= 1
        if passes >= 2:
            assert n_log == 2, (passes, [w.spec() for w in warps])


def test_axes_with_no_signal_are_left_alone():
    """Without the evidence margin an axis acting only through an interaction
    is handed whichever candidate best fitted the noise."""
    X, Y = _decades(np.random.default_rng(0))
    warps = fit_warps(X, Y, random_state=0)
    assert warps[2].spec() == "identity"
    assert warps[3].spec() == "identity"
    loose = fit_warps(X, Y, random_state=0, margin=1.0)
    assert sum(w.spec() != "identity" for w in loose) >= sum(
        w.spec() != "identity" for w in warps)


def test_the_joint_criterion_sees_what_the_marginal_one_misses():
    """On the sharp target the marginal effects are there, but the criterion
    has to score the axes together to pick the transitions up."""
    X, Y = _sharp(np.random.default_rng(0))
    joint = fit_warps(X, Y, random_state=0, criterion="joint")
    marginal = fit_warps(X, Y, random_state=0, criterion="marginal")
    assert sum(w.shape != "identity" for w in joint) > sum(
        w.shape != "identity" for w in marginal)


@pytest.mark.parametrize("shape, param", [("identity", 0.0), ("sinh", 10.0),
                                          ("kte", 0.9), ("sinh", 1.0)])
@pytest.mark.parametrize("centre", [-0.5, 0.0, 0.5])
def test_every_warp_is_monotone_and_lands_in_the_box(shape, param, centre):
    w = Warp("identity", shape, param, centre, -2.0, 3.0)
    x = np.linspace(-2.0, 3.0, 2001)
    u = w(x)
    assert np.all(np.diff(u) > 0), "not strictly increasing"
    assert np.isfinite(u).all()
    np.testing.assert_allclose([u[0], u[-1]], [-1.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("shape, param", [("sinh", 10.0), ("kte", 0.9)])
def test_warps_stay_finite_outside_the_training_box(shape, param):
    """kte uses arcsin, which is NaN past its domain; extrapolated points must
    saturate instead, leaving the emulator's own guard to do the complaining."""
    w = Warp("identity", shape, param, 0.0, -1.0, 1.0)
    u = w(np.array([-4.0, -1.5, 0.0, 1.5, 4.0]))
    assert np.isfinite(u).all()


def test_log_needs_positive_values():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (3000, 2))                 # straddles zero
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    assert all(w.pre == "identity" for w in fit_warps(X, Y, random_state=0))


def _log_target(ratio, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x0 = ratio ** rng.uniform(0, 1, n)
    X = np.column_stack([x0] + [rng.uniform(-1, 1, n) for _ in range(3)])
    t = np.log10(x0) / np.log10(ratio)
    Y = (np.sin(3.0 * t) + 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3]).reshape(-1, 1)
    return X, Y


@pytest.mark.parametrize("ratio", [3, 10, 100])
def test_log_is_offered_at_any_dynamic_range(ratio):
    """There is no threshold on the range ratio. An earlier version required
    above 10, which blocked the warp exactly where it paid: the gain rises
    smoothly from 23x at a ratio of 2 to 691x at 10, with no break."""
    X, Y = _log_target(ratio)
    assert fit_warps(X, Y, random_state=0)[0].pre == "log"


@pytest.mark.parametrize("ratio", [10, 1000])
def test_a_log_warp_that_does_not_help_is_still_rejected(ratio):
    """Nothing is needed on the other side of the removed threshold, because
    the evidence margin already rejects a useless log. Here the response is
    polynomial in x rather than in log x, so log must not be chosen however
    many decades the axis spans."""
    rng = np.random.default_rng(0)
    x0 = ratio ** rng.uniform(0, 1, 6000)
    X = np.column_stack([x0] + [rng.uniform(-1, 1, 6000) for _ in range(3)])
    u = (x0 - x0.min()) / (x0.max() - x0.min())
    Y = (np.sin(3.0 * u) + 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3]).reshape(-1, 1)
    assert fit_warps(X, Y, random_state=0)[0].pre == "identity"


def test_warped_emulator_beats_raw_coordinates():
    X, Y = _decades(np.random.default_rng(0))
    Xte, Yte, Xtr, Ytr = X[:2500], Y[:2500], X[2500:], Y[2500:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = PolyEmu(Xtr, Ytr, init_deg_forward=8, max_degree_forward=8,
                        RMSE_tol=0.0, verbose=0)
        warped = WarpedEmu(Xtr, Ytr, random_state=0, init_deg_forward=8,
                           max_degree_forward=8, RMSE_tol=0.0, verbose=0)

    def err(model):
        return float(np.sqrt(np.mean(
            (model.forward_emulator(Xte, extrapolation="ignore") - Yte) ** 2)))

    assert err(warped) < err(plain) / 100.0
    assert warped.gain > 1.0
    rep = warped.report()
    assert rep["warps"][:2] == ("log", "log")


def test_supplied_warps_are_used_and_validated():
    X, Y = _decades(np.random.default_rng(0))
    warps = fit_warps(X, Y, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = WarpedEmu(X[2500:], Y[2500:], warps=warps, check=False,
                        max_degree_forward=4, verbose=0)
    assert emu.warps == warps and emu.gain is None
    with pytest.raises(ValueError, match="warps for"):
        WarpedEmu(X, Y, warps=warps[:2], max_degree_forward=3, verbose=0)


def test_families_are_what_the_module_advertises():
    assert set(FAMILIES) == {"identity", "sinh", "kte"}
    X, Y = _sharp(np.random.default_rng(0))
    for w in fit_warps(X, Y, random_state=0):
        assert w.shape in FAMILIES and w.pre in ("identity", "log")
