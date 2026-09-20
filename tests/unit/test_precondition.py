"""T-002: composing rotation (P2) with warping (P4).

The two preconditioners do not commute, and neither order dominates. Which one
wins is a property of the target, so these tests pin both directions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.precondition import ORDERS, PreconditionedEmu

P = 7
_Q, _ = np.linalg.qr(np.random.default_rng(11).standard_normal((P, P)))


def rotated_sharp(X):
    """A sharp feature along a ROTATED direction: no raw axis carries it, so a
    per-axis warp alone can see nothing."""
    return np.tanh(12.0 * (X @ _Q[:, 0])) + 0.5 * (X @ _Q[:, 1]) ** 2


def log_ridge(X):
    """A ridge in LOG coordinates: it is not a ridge in the raw ones, so a rank
    reduction applied first discards real signal."""
    t = (0.6 * np.log10(X[:, 0]) + 0.5 * np.log10(X[:, 1])
         - 0.4 * np.log10(X[:, 2]))
    return np.tanh(1.8 * (t + 1.0)) + 0.3 * X[:, 3] * X[:, 4]


def uniform_design(rng, n):
    return rng.uniform(-1, 1, (n, P))


def decade_design(rng, n):
    return np.column_stack(
        [10.0 ** rng.uniform(-3, 0, n), 10.0 ** rng.uniform(-2, 1, n),
         10.0 ** rng.uniform(-1, 2, n)]
        + [rng.uniform(-1, 1, n) for _ in range(P - 3)]
    )


def _data(design, target, n=6000, seed=3):
    rng = np.random.default_rng(seed)
    X, Xte = design(rng, n), design(rng, 4000)
    return X, target(X).reshape(-1, 1), Xte, target(Xte).reshape(-1, 1)


def _fom(model, Xte, Yte):
    pred = model.forward_emulator(Xte, extrapolation="ignore")
    return float(np.sqrt(np.mean((pred - Yte) ** 2)) / np.sqrt(np.mean(Yte ** 2)))


def _fit(X, Y, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PreconditionedEmu(X, Y, rank=2, random_state=0, RMSE_tol=0.0,
                                 verbose=0, **kw)


@pytest.fixture(scope="module")
def rotated():
    return _data(uniform_design, rotated_sharp)


@pytest.fixture(scope="module")
def logged():
    return _data(decade_design, log_ridge)


@pytest.fixture(scope="module")
def rotated_auto(rotated):
    X, Y, _, _ = rotated
    return _fit(X, Y)


@pytest.fixture(scope="module")
def rotated_by_order(rotated):
    """One fit per explicit order, shared: each PreconditionedEmu with
    order="auto" fits all five candidates, so refitting per test is wasteful."""
    X, Y, Xte, Yte = rotated
    return {o: _fom(_fit(X, Y, order=o), Xte, Yte) for o in ORDERS}


@pytest.fixture(scope="module")
def logged_by_order(logged):
    X, Y, Xte, Yte = logged
    return {o: _fom(_fit(X, Y, order=o), Xte, Yte) for o in ORDERS}


@pytest.mark.slow
def test_rotation_must_come_first_for_a_rotated_feature(rotated_by_order):
    scores = rotated_by_order
    # warping alone is powerless here: no raw axis carries the feature
    assert scores["warp"] == pytest.approx(scores["none"], rel=1e-6)
    # and warping AFTER rotating is what pays
    assert scores["rotate_warp"] < scores["rotate"] / 3.0
    assert scores["rotate_warp"] < scores["warp_rotate"] / 3.0


@pytest.mark.slow
def test_warping_must_come_first_for_a_ridge_in_log_coordinates(logged_by_order):
    scores = logged_by_order
    # rotating first is not merely useless here, it is worse than doing nothing
    assert scores["rotate"] > scores["none"]
    assert scores["warp_rotate"] < scores["rotate_warp"] / 3.0


def test_the_two_orders_are_not_the_same_model(rotated_by_order):
    assert rotated_by_order["rotate_warp"] != pytest.approx(
        rotated_by_order["warp_rotate"], rel=1e-3)


def test_auto_picks_the_order_that_wins(rotated_auto):
    auto = rotated_auto
    assert auto.order == "rotate_warp"
    assert set(auto.scores) == set(ORDERS)
    assert auto.scores[auto.order] == min(auto.scores.values())


def test_each_order_is_scored_at_the_degree_it_can_afford(rotated_auto):
    """Scoring every order at one degree would compare a 2-D fit and a 7-D fit
    at a degree the 7-D one can reach, hiding what rotation is for."""
    auto = rotated_auto
    assert auto.scan_degrees["rotate"] > auto.scan_degrees["none"]
    assert auto.dimensions["rotate_warp"] == 2
    assert auto.dimensions["warp"] == P


@pytest.mark.slow
def test_parsimony_trades_accuracy_for_a_smaller_model(logged):
    """Which order is most ACCURATE depends on the sample count, so the
    contract pinned here is the trade, not the winner: parsimony never returns
    a larger model than accuracy, and pays at most its tolerance for it."""
    X, Y, Xte, Yte = logged
    acc = _fit(X, Y, select="accuracy")
    par = _fit(X, Y, select="parsimony", parsimony_tol=5.0)
    assert par.dimensions[par.order] <= acc.dimensions[acc.order]
    assert par.report()["n_terms"] <= acc.report()["n_terms"]
    assert par.scores[par.order] <= 5.0 * min(acc.scores.values())
    assert _fom(par, Xte, Yte) < 5.0 * _fom(acc, Xte, Yte)


@pytest.mark.slow
def test_parsimony_prefers_the_smallest_near_best_order(logged):
    X, Y, _, _ = logged
    par = _fit(X, Y, select="parsimony", parsimony_tol=5.0)
    floor = min(par.scores.values())
    near = {k: par.dimensions[k] for k, v in par.scores.items() if v <= 5.0 * floor}
    assert par.dimensions[par.order] == min(near.values())


def test_degree_is_clamped_with_a_warning_when_the_order_cannot_afford_it(logged):
    X, Y, _, _ = logged
    with pytest.warns(UserWarning, match="dimension"):
        PreconditionedEmu(X, Y, order="warp", rank=2, random_state=0,
                          init_deg_forward=12, max_degree_forward=12,
                          RMSE_tol=0.0, verbose=0)


def test_transform_chains_the_steps(rotated):
    X, Y, Xte, _ = rotated
    emu = _fit(X, Y, order="rotate_warp", max_degree_forward=6)
    Z = emu.transform(Xte)
    assert Z.shape == (Xte.shape[0], 2)
    manual = Xte
    for step in emu.steps:
        manual = step(manual)
    np.testing.assert_allclose(Z, manual)


@pytest.mark.parametrize("kwargs, match", [
    ({"order": "sideways"}, "order must be"),
    ({"select": "cheapest"}, "select must be"),
])
def test_validation(rotated, kwargs, match):
    X, Y, _, _ = rotated
    with pytest.raises(ValueError, match=match):
        PreconditionedEmu(X[:2000], Y[:2000], rank=2, max_degree_forward=3,
                          verbose=0, **kwargs)


# --- composing with the other estimators ----------------------------------

def test_sparse_estimator_composes_with_a_rotation(rotated):
    """The natural partner: a rotation cuts the dimension, which is what makes
    a large candidate set affordable in the first place."""
    from MomentEmu.basis import Basis

    X, Y, Xte, Yte = rotated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PreconditionedEmu(
            X, Y, order="rotate", rank=2, random_state=0,
            estimator="sparse", candidate=Basis(degree=20, max_interaction=2),
            degree=20, n_terms=40,
        )
    rep = emu.report()
    assert rep["estimator"] == "sparse" and rep["n_dims"] == 2
    assert rep["n_terms"] == 40
    # no extrapolation= here: keywords go to the estimator, and SparseEmu
    # has no such argument
    pred = emu.forward_emulator(Xte)
    err = float(np.sqrt(np.mean((pred - Yte) ** 2)) / np.sqrt(np.mean(Yte ** 2)))
    assert err < 0.2, err


def test_factored_estimator_refuses_to_follow_a_rotation(rotated):
    """A rotation replaces the parameters by linear combinations, so a block
    partition of the originals stops referring to anything."""
    X, Y, _, _ = rotated
    with pytest.raises(ValueError, match="cannot follow a rotation"):
        PreconditionedEmu(X, Y, order="rotate", rank=2, random_state=0,
                          estimator="factored", blocks=((0,), (1,)))


def test_factored_estimator_works_without_a_rotation():
    """Warping is per axis, so it leaves a block partition meaningful."""
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (6000, 6))
    blocks = ((0, 1, 2), (3, 4, 5))
    Y = ((1.2 + np.sin(0.9 * X[:, 0] + 0.7 * X[:, 1] + 0.5 * X[:, 2]))
         * (1.6 + 0.5 * np.exp(0.4 * (X[:, 3] + 0.8 * X[:, 4] + 0.6 * X[:, 5])))
         ).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PreconditionedEmu(X, Y, order="none", estimator="factored",
                                blocks=blocks, rank=1, degree=5,
                                random_state=0)
    rep = emu.report()
    assert rep["estimator"] == "factored" and rep["n_dims"] == 6
    pred = emu.forward_emulator(X[:500])
    err = float(np.sqrt(np.mean((pred - Y[:500]) ** 2)) / np.sqrt(np.mean(Y ** 2)))
    assert err < 1e-3, err


def test_unknown_estimator_is_rejected(rotated):
    X, Y, _, _ = rotated
    with pytest.raises(ValueError, match="estimator must be"):
        PreconditionedEmu(X[:1500], Y[:1500], order="none", estimator="magic",
                          max_degree_forward=3, verbose=0)
