"""T-001: boundary behaviour of the block-separable index set.

The block basis is only valid where the target really is additively separable
in these coordinates. These tests pin both sides of that boundary: the cells
where the reduction is free, and the cells where it costs orders of magnitude.
Extreme degrees and block shapes are included because the failure modes are not
monotone in degree.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_legendre

from MomentEmu.basis import Basis
from MomentEmu.emulator import generate_multi_indices
from MomentEmu.monomials import MonomialPlan

BLOCKS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
P = 9


def _names(n):
    return [f"x{i}" for i in range(n)]


def _g(u, v, w):
    return np.sin(u + 0.5 * v * w) * np.exp(0.3 * w) + 0.4 * np.cos(1.2 * u * v)


def _separable(X):
    return sum(_g(X[:, a], X[:, b], X[:, c]) for a, b, c in BLOCKS)


_Q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((P, P)))


def _rotated(X):
    return _separable(X @ _Q.T / 1.8)


def _cross(X):
    return _separable(X) + 0.3 * X[:, 0] * X[:, 4]


def _fit_rmse(mi, Xtr, ytr, Xte, yte):
    plan = MonomialPlan.build(mi)
    A = plan.evaluate(Xtr)
    c = np.linalg.solve(A.T @ A, A.T @ ytr)
    return float(np.sqrt(np.mean((plan.evaluate(Xte) @ c - yte) ** 2)))


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(31)
    return (rng.uniform(-1, 1, (8000, P)), rng.uniform(-1, 1, (4000, P)))


@pytest.mark.parametrize("fn, lo, hi", [
    (_separable, 0.0, 1.5),      # free: block basis is at least as accurate
    (_cross, 5.0, 1e3),          # one cross-block term already costs ~31x
    (_rotated, 20.0, 1e5),       # separable only after a rotation: ~297x
])
def test_block_basis_degrades_exactly_when_the_assumption_breaks(data, fn, lo, hi):
    Xtr, Xte = data
    ytr, yte = fn(Xtr), fn(Xte)
    full = _fit_rmse(generate_multi_indices(P, 5), Xtr, ytr, Xte, yte)
    blk = _fit_rmse(Basis.separable(BLOCKS).build(_names(P), 5), Xtr, ytr, Xte, yte)
    assert np.isfinite(full) and np.isfinite(blk)
    assert lo <= blk / full <= hi


@pytest.mark.parametrize("d", [0, 1, 2, 5, 8])
@pytest.mark.parametrize("blocks", [
    ((0, 1, 2), (3, 4, 5), (6, 7, 8)),
    tuple((i,) for i in range(P)),          # extreme: every parameter alone
    (tuple(range(P)),),                     # extreme: one block = isotropic
    ((0,), tuple(range(1, P))),             # extreme: 1 + (P-1)
])
def test_index_set_is_well_formed_at_every_corner(d, blocks):
    mi = Basis.separable(blocks).build(_names(P), d)
    labels = np.empty(P, dtype=int)
    for k, b in enumerate(blocks):
        labels[list(b)] = k
    assert mi.shape[0] >= 1
    assert np.all(mi.sum(axis=1) <= d)
    assert len({tuple(r) for r in mi}) == mi.shape[0]
    for alpha in mi:
        support = np.flatnonzero(alpha)
        assert support.size == 0 or np.unique(labels[support]).size == 1
    # never larger than the isotropic set of the same degree
    assert mi.shape[0] <= generate_multi_indices(P, d).shape[0]


def test_singleton_blocks_give_the_additive_basis():
    """Every parameter in its own block is the purely additive model:
    1 + p*d terms, no interaction at all."""
    d = 4
    mi = Basis.separable(tuple((i,) for i in range(P))).build(_names(P), d)
    assert mi.shape[0] == 1 + P * d
    assert np.all(np.count_nonzero(mi, axis=1) <= 1)


@pytest.mark.parametrize("d", [2, 4, 6])
def test_monomial_and_legendre_span_the_same_functions(data, d):
    """The block index set is downward closed, so the two bases span the same
    space and the fitted function cannot depend on which one expresses it.
    This is why switching to an orthonormal basis does not change accuracy."""
    Xtr, Xte = data
    ytr = _cross(Xtr)
    mi = Basis.separable(BLOCKS).build(_names(P), d)

    def legendre(X):
        cols = [
            np.stack([np.sqrt(2 * a + 1) * eval_legendre(a, X[:, i]) for a in range(d + 1)], 1)
            for i in range(P)
        ]
        out = np.ones((X.shape[0], mi.shape[0]))
        for r, alpha in enumerate(mi):
            for i in range(P):
                if alpha[i]:
                    out[:, r] *= cols[i][:, alpha[i]]
        return out

    plan = MonomialPlan.build(mi)
    preds = []
    for build in (plan.evaluate, legendre):
        c = np.linalg.lstsq(build(Xtr), ytr, rcond=None)[0]
        preds.append(build(Xte) @ c)
    scale = float(np.std(preds[0]))
    assert np.max(np.abs(preds[0] - preds[1])) < 1e-9 * scale
