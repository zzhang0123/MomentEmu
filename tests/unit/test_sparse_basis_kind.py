"""SparseEmu builds its design from the shared basis plans, not its own copy.

sparse.py carried a second implementation of the orthonormal Legendre product
basis, written with scipy.special.eval_legendre, while monomials.py already
had one behind LegendrePlan. They agreed to 1.5e-15, so the duplicate was
removable -- and removing it is what makes basis_kind possible at all.

The default stays "legendre". It is the right default here: the selector maps
the training box onto [-1, 1] and the design is whatever the caller sampled,
which is usually closer to uniform than to the arcsine measure Chebyshev is
orthogonal under.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_legendre

from MomentEmu.emulator import generate_multi_indices
from MomentEmu.monomials import ChebyshevPlan, LegendrePlan, MonomialPlan
from MomentEmu.sparse import SparseEmu

KINDS = ("legendre", "chebyshev", "monomial")
PLANS = {"legendre": LegendrePlan, "chebyshev": ChebyshevPlan,
         "monomial": MonomialPlan}


def _design(n_samples: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, 3))
    Y = np.column_stack([
        np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2 - 0.4 * X[:, 0] * X[:, 2],
        np.cos(X[:, 2]) + 0.3 * X[:, 1] ** 3,
    ])
    return X, Y


def _legacy_legendre_design(V: np.ndarray, mi: np.ndarray) -> np.ndarray:
    """The construction sparse.py used to carry inline."""
    dmax = int(mi.max()) if mi.size else 0
    cols = [
        np.stack([np.sqrt(2.0 * a + 1.0) * eval_legendre(a, V[:, i])
                  for a in range(dmax + 1)], axis=1)
        for i in range(V.shape[1])
    ]
    out = np.ones((V.shape[0], mi.shape[0]))
    for r, alpha in enumerate(mi):
        for i in range(V.shape[1]):
            if alpha[i]:
                out[:, r] *= cols[i][:, alpha[i]]
    return out


def test_default_reproduces_the_construction_it_replaces() -> None:
    """The shared plan must be the same basis, not merely a similar one."""
    rng = np.random.default_rng(1)
    V = rng.uniform(-1.0, 1.0, (200, 3))
    mi = generate_multi_indices(3, 6)
    legacy = _legacy_legendre_design(V, mi)
    shared = LegendrePlan.build(mi).evaluate(V)
    np.testing.assert_allclose(shared, legacy, rtol=1e-12, atol=1e-12)


def test_default_basis_kind_is_legendre() -> None:
    X, Y = _design()
    a = SparseEmu(X, Y, degree=5, n_terms=30)
    b = SparseEmu(X, Y, degree=5, n_terms=30, basis_kind="legendre")
    np.testing.assert_array_equal(a.multi_indices, b.multi_indices)
    np.testing.assert_allclose(a.coefficients, b.coefficients, rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_prediction_uses_the_basis_the_selection_used(kind: str) -> None:
    """The failure this guards against is silent.

    PolyEmu once built its moment matrix with MonomialPlan while predicting
    with the requested basis; the stored coefficients gave residual 1.30 where
    a consistent fit gave 2.5e-04, and nothing raised. Rebuild the design here
    from the plan the caller asked for and check it reproduces the emulator.
    """
    X, Y = _design()
    emu = SparseEmu(X, Y, degree=5, n_terms=40, basis_kind=kind)
    V = 2.0 * (X - emu.lo_) / emu.span_ - 1.0
    manual = PLANS[kind].build(emu.multi_indices).evaluate(V) @ emu.coefficients
    manual = manual * emu.scale_Y_ + emu.mean_Y_
    np.testing.assert_allclose(emu.forward_emulator(X), manual,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_every_basis_fits_the_target(kind: str) -> None:
    X, Y = _design()
    emu = SparseEmu(X, Y, degree=6, n_terms=60, basis_kind=kind)
    rel = np.linalg.norm(emu.forward_emulator(X) - Y) / np.linalg.norm(Y)
    assert rel < 5e-3, f"{kind}: {rel:.3e}"


def test_the_bases_are_actually_different() -> None:
    """Otherwise the parameter would be decorative."""
    X, Y = _design()
    fits = {k: SparseEmu(X, Y, degree=5, n_terms=30, basis_kind=k) for k in KINDS}
    sets = {k: {tuple(r) for r in f.multi_indices.tolist()}
            for k, f in fits.items()}
    assert sets["monomial"] != sets["legendre"], (
        "a non-orthogonal basis selected the same terms; the greedy step is "
        "not seeing the basis"
    )


def test_basis_kind_is_validated() -> None:
    X, Y = _design()
    with pytest.raises(ValueError, match="basis_kind"):
        SparseEmu(X, Y, degree=4, n_terms=10, basis_kind="hermite")


def test_a_pickle_from_before_basis_kind_still_predicts() -> None:
    """Old pickles carry no basis_kind, so _plan_cls must not be stored state.

    Basis hit this exact failure earlier in the project: an attribute added to
    __init__ is absent from every object pickled before it existed, and the
    first use raises AttributeError far from the cause.
    """
    import pickle

    X, Y = _design()
    emu = SparseEmu(X, Y, degree=5, n_terms=30)
    expected = emu.forward_emulator(X)

    revived = pickle.loads(pickle.dumps(emu))
    del revived.__dict__["basis_kind"]          # what an old pickle looks like
    np.testing.assert_allclose(revived.forward_emulator(X), expected,
                               rtol=1e-12, atol=1e-14)
    assert revived._plan_cls is LegendrePlan
