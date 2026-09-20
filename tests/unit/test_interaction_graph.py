"""T-001 item 2: the full pairwise interaction matrix and its blocks.

sobol_report returns only the top 10 pairs, so a cross-block coupling is
crowded out by in-block pairs as soon as the parameter count grows.
interaction_graph returns the complete n x n matrix and the connected
components it implies.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, generate_multi_indices

P = 9
BLOCKS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def _g(u, v, w):
    return np.sin(u + 0.5 * v * w) * np.exp(0.3 * w) + 0.4 * np.cos(1.2 * u * v)


def _base(X):
    return sum(_g(X[:, a], X[:, b], X[:, c]) for a, b, c in BLOCKS)


def _fit(extra, n=8000, seed=7, max_degree=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, P))
    Y = (_base(X) + extra(X)).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PolyEmu(X, Y, forward=True, backward=False,
                       max_degree_forward=max_degree, verbose=0)


@pytest.fixture(scope="module")
def emu_separable():
    return _fit(lambda X: 0.0)


@pytest.fixture(scope="module")
def emu_cross():
    # a degree-4, even-parity cross term between block 0 and block 1
    return _fit(lambda X: 0.25 * X[:, 0] ** 2 * X[:, 3] ** 2)


def test_matrix_is_complete_symmetric_and_hollow(emu_separable):
    r = emu_separable.interaction_graph(degree=4, warn_uniform=False)
    M = r["matrix"]
    assert M.shape == (P, P)
    np.testing.assert_allclose(M, M.T, rtol=0, atol=0)
    np.testing.assert_array_equal(np.diag(M), np.zeros(P))
    # every one of the 36 pairs is represented, unlike sobol_report's top 10
    assert np.count_nonzero(M) == P * (P - 1)
    assert len(emu_separable.sobol_report(degree=4, warn_uniform=False)["top_pairs"]) == 10


def test_separable_function_recovers_the_blocks(emu_separable):
    r = emu_separable.interaction_graph(degree=4, warn_uniform=False)
    assert r["blocks"] == BLOCKS
    assert r["separable"] is True
    cross = [r["matrix"][i, j] for i in range(P) for j in range(i + 1, P) if i // 3 != j // 3]
    within = [r["matrix"][i, j] for i in range(P) for j in range(i + 1, P) if i // 3 == j // 3]
    # three orders of magnitude of separation between cross and within
    assert max(cross) < 1e-2 * min(within)


def test_cross_term_merges_two_blocks(emu_cross):
    r = emu_cross.interaction_graph(degree=4, warn_uniform=False)
    assert r["blocks"] == ((0, 1, 2, 3, 4, 5), (6, 7, 8))
    assert r["separable"] is True  # still separable, just coarser


def test_degree_must_cover_the_interaction_order(emu_cross):
    """A degree-4 coupling is invisible to a degree-3 projection (regression).

    x0^2 x3^2 needs total degree 4; at degree 3 the pair share collapses into
    the noise floor and the blocks are reported as un-merged.
    """
    lo = emu_cross.interaction_graph(degree=3, warn_uniform=False)
    hi = emu_cross.interaction_graph(degree=4, warn_uniform=False)
    assert lo["matrix"][0, 3] < lo["threshold"]   # invisible: below the noise floor
    assert hi["matrix"][0, 3] > hi["threshold"]   # detected
    assert lo["blocks"] == BLOCKS             # wrong, but that is the point
    assert hi["blocks"] == ((0, 1, 2, 3, 4, 5), (6, 7, 8))


def test_pair_share_matches_the_closed_form(emu_cross):
    """x^2 = 1/3 + (2/(3 sqrt 5)) Lhat2(x), so 0.25 x0^2 x3^2 puts
    0.25 (2/(3 sqrt5))^2 on Lhat2(x0) Lhat2(x3); its Sobol share is that squared
    over the explained variance."""
    r = emu_cross.interaction_graph(degree=4, warn_uniform=False)
    c = 0.25 * (2.0 / (3.0 * np.sqrt(5.0))) ** 2
    predicted = c ** 2 / float(r["explained_variance"][0])
    assert abs(r["matrix"][0, 3] / predicted - 1.0) < 0.05


def test_threshold_is_monotone(emu_cross):
    coarse = emu_cross.interaction_graph(degree=4, threshold=1e-6, warn_uniform=False)
    fine = emu_cross.interaction_graph(degree=4, threshold=1e-1, warn_uniform=False)
    assert len(coarse["blocks"]) <= len(fine["blocks"])
    assert fine["blocks"] == tuple((i,) for i in range(P))  # nothing survives


def test_output_selection(emu_separable):
    r_all = emu_separable.interaction_graph(degree=4, warn_uniform=False)
    r_0 = emu_separable.interaction_graph(degree=4, output=0, warn_uniform=False)
    np.testing.assert_allclose(r_all["matrix"], r_0["matrix"])
    with pytest.raises(ValueError, match="output"):
        emu_separable.interaction_graph(degree=4, output=5, warn_uniform=False)


def test_negative_threshold_is_rejected(emu_separable):
    with pytest.raises(ValueError, match="threshold"):
        emu_separable.interaction_graph(degree=4, threshold=-1e-3, warn_uniform=False)


def test_warns_when_the_projection_explains_little(emu_separable):
    """Absence of interaction is not evidence when the fit is poor."""
    with pytest.warns(UserWarning, match="explains"):
        emu_separable.interaction_graph(degree=1, min_explained=0.95, warn_uniform=False)
    # and stays quiet once the projection is adequate
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        emu_separable.interaction_graph(degree=4, min_explained=0.95, warn_uniform=False)


def test_blocks_feed_a_basis(emu_separable):
    from MomentEmu.basis import Basis
    r = emu_separable.interaction_graph(degree=4, warn_uniform=False)
    mi = Basis.separable(r["blocks"], degree=5).build([f"x{i}" for i in range(P)])
    labels = np.empty(P, dtype=int)
    for k, b in enumerate(r["blocks"]):
        labels[list(b)] = k
    for alpha in mi:
        support = np.flatnonzero(alpha)
        assert support.size == 0 or np.unique(labels[support]).size == 1


@pytest.mark.slow
def test_round_trip_detect_then_fit():
    """interaction_graph -> Basis.separable -> PolyEmu: fewer terms, no loss."""
    from MomentEmu.basis import Basis

    rng = np.random.default_rng(21)
    X = rng.uniform(-1.0, 1.0, (9000, P))
    Y = _base(X).reshape(-1, 1)
    Xte = rng.uniform(-1.0, 1.0, (3000, P))
    Yte = _base(Xte).reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = PolyEmu(X, Y, forward=True, backward=False, max_degree_forward=4, verbose=0)
        blocks = probe.interaction_graph(degree=4, warn_uniform=False)["blocks"]
        assert blocks == BLOCKS

        block_emu = PolyEmu(X, Y, forward=True, backward=False,
                            basis=Basis.separable(blocks), init_deg_forward=5,
                            max_degree_forward=5, RMSE_tol=0.0, verbose=0)
        full_emu = PolyEmu(X, Y, forward=True, backward=False,
                           init_deg_forward=5, max_degree_forward=5,
                           RMSE_tol=0.0, verbose=0)

    n_block = block_emu.forward_multi_indices.shape[0]
    n_full = full_emu.forward_multi_indices.shape[0]
    assert n_block == 166 and n_full == 2002

    def err(e):
        pred = e.forward_emulator(Xte, extrapolation="ignore")
        return float(np.sqrt(np.mean((pred - Yte) ** 2)))

    assert err(block_emu) <= err(full_emu)


# --- review follow-ups: degenerate output columns --------------------------

def _multi_output_emu(second):
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (4000, 6))
    Y = np.column_stack([np.sin(X[:, 0]) + X[:, 1] * X[:, 2], second(X)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PolyEmu(X, Y, forward=True, backward=False,
                       max_degree_forward=3, verbose=0)


def test_constant_output_column_does_not_contaminate_the_blocks():
    """A constant column has explained variance at the rounding level, so its
    "shares" are noise/noise ~ O(0.1). Aggregating by max over outputs would
    otherwise merge every parameter into one block."""
    emu = _multi_output_emu(lambda X: np.full(X.shape[0], 3.7))
    with pytest.warns(UserWarning, match="degenerate"):
        r = emu.interaction_graph(degree=3, warn_uniform=False)
    assert r["blocks"] == ((0,), (1, 2), (3,), (4,), (5,))
    assert r["degenerate_outputs"] == (1,)


def test_degenerate_output_requested_explicitly_raises():
    emu = _multi_output_emu(lambda X: np.full(X.shape[0], 3.7))
    with pytest.raises(ValueError, match="degenerate"):
        emu.interaction_graph(degree=3, output=1, warn_uniform=False)


def test_all_outputs_degenerate_raises():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (4000, 6))
    Y = np.column_stack([np.full(X.shape[0], 3.7), np.zeros(X.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, Y, forward=True, backward=False,
                      max_degree_forward=3, verbose=0)
        with pytest.raises(ValueError, match="degenerate"):
            emu.interaction_graph(degree=3, warn_uniform=False)


def test_small_but_real_output_is_not_called_degenerate():
    """Degeneracy must be relative to the column's own scale, not absolute:
    a genuine signal scaled to 1e-20 is still a signal."""
    emu = _multi_output_emu(lambda X: 1e-20 * (np.cos(X[:, 3]) + X[:, 4] * X[:, 5]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = emu.interaction_graph(degree=3, output=1, warn_uniform=False)
    assert r["degenerate_outputs"] == ()
    assert r["blocks"] == ((0,), (1,), (2,), (3,), (4, 5))


# --- T-001: per-parameter degree/parity profile ----------------------------

def test_degree_profile_recovers_caps_and_parity():
    """f is exactly quadratic in x0, even-only in x1 and x4."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (6000, 5))
    Y = ((1 + 0.5 * X[:, 0] + 0.3 * X[:, 0] ** 2) * np.sin(1.5 * X[:, 2])
         + (0.2 + X[:, 1] ** 2) * np.exp(0.4 * X[:, 3])
         + np.cos(2.0 * X[:, 4])).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, Y, forward=True, backward=False, max_degree_forward=5, verbose=0)
        prof = emu.degree_profile(degree=5, warn_uniform=False)
    assert prof["max_degree"][0] == 2          # exactly quadratic in x0
    assert prof["max_degree"][1] == 2
    assert prof["parity"][1] == "even"         # enters only through x1**2
    assert prof["parity"][4] == "even"         # enters only through cos(2 x4)
    assert prof["parity"][0] is None           # has both x0 and x0**2

    mi = prof["basis"].build([f"x{i}" for i in range(5)], 7)
    full = generate_multi_indices(5, 7).shape[0]
    assert mi.shape[0] < full
    assert np.all(mi[:, 1] % 2 == 0) and np.all(mi[:, 4] % 2 == 0)


def test_degree_profile_rejects_degenerate_outputs():
    emu = _multi_output_emu(lambda X: np.full(X.shape[0], 3.7))
    with pytest.warns(UserWarning, match="degenerate"):
        prof = emu.degree_profile(degree=3, warn_uniform=False)
    assert prof["degenerate_outputs"] == (1,)
