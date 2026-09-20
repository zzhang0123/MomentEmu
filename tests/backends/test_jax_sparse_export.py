"""T-010: a sparse fit must reach the JAX backend, or say why it cannot.

Reported from the same 21cmVAE-jax port as T-007. ``from_polyemu`` is written
against ``PolyEmu``: it reads ``n_outputs``, ``forward_coeffs_folded`` and
``forward_plan``, none of which ``SparseEmu`` carries, so the sparse estimator
could not be exported at all. T-007 closed the same shape of gap one level
down, for the basis families.

It matters now rather than eventually, because sparse is better per term on
that target -- 1.3497 percent at 3,000 terms against 1.4334 percent at 3,003
for the dense fit -- and a batch of 1,024 through vmap is where the
polynomial backend loses worst to the network it replaces, 4.59x, with the
cost being the basis functions evaluated. Half the terms is the lever.

The monomial family is deferred, not forgotten. The JAX kernel evaluates
monomials from the downward-closed level tables a ``MonomialPlan`` carries,
and a selected index set is NOT downward closed -- that is what sparsity
means. Rebuilding the closure would restore exactly the terms the selection
removed. So a monomial sparse fit raises, and says that.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.sparse import SparseEmu

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from MomentEmu.jax_momentemu import JaxEmulator  # noqa: E402

TENSOR_BASES = ("legendre", "chebyshev")


def _design(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = np.column_stack([np.sin(1.4 * X[:, 0]) + X[:, 1] ** 2 - 0.3 * X[:, 2],
                         np.cos(X[:, 1] * X[:, 2]) + 0.5 * X[:, 0]])
    return X, Y


def _fit(basis_kind: str, **kwargs):
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SparseEmu(X, Y, degree=5, n_terms=45,
                         basis_kind=basis_kind, **kwargs), X, Y


@pytest.mark.parametrize("basis_kind", TENSOR_BASES)
def test_a_sparse_fit_exports_and_agrees_with_the_numpy_model(basis_kind: str):
    emu, X, _Y = _fit(basis_kind)
    jemu = JaxEmulator.from_sparse(emu)
    got = np.asarray(jemu(X))
    np.testing.assert_allclose(got, emu.forward_emulator(X),
                               rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("basis_kind", TENSOR_BASES)
def test_from_polyemu_accepts_a_sparse_fit_too(basis_kind: str):
    """Callers reach for the name they know; it must not raise AttributeError."""
    emu, X, _Y = _fit(basis_kind)
    np.testing.assert_allclose(
        np.asarray(JaxEmulator.from_polyemu(emu)(X)),
        np.asarray(JaxEmulator.from_sparse(emu)(X)),
        rtol=1e-12, atol=1e-13,
    )


@pytest.mark.parametrize("basis_kind", TENSOR_BASES)
def test_the_exported_jacobian_matches_finite_differences(basis_kind: str):
    emu, X, _Y = _fit(basis_kind)
    jemu = JaxEmulator.from_sparse(emu)
    point = X[:1]
    got = np.asarray(jax.jacfwd(lambda x: jemu(x))(point))[0, :, 0, :]
    step = 1e-6
    ref = np.empty_like(got)
    for i in range(X.shape[1]):
        lo, hi = point.copy(), point.copy()
        lo[0, i] -= step
        hi[0, i] += step
        ref[:, i] = (emu.forward_emulator(hi)[0] - emu.forward_emulator(lo)[0]) / (2 * step)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("basis_kind", TENSOR_BASES)
def test_a_batch_goes_through_vmap(basis_kind: str):
    """The reason the export is wanted: batched evaluation."""
    emu, X, _Y = _fit(basis_kind)
    jemu = JaxEmulator.from_sparse(emu)
    batched = np.asarray(jax.vmap(lambda x: jemu(x[None, :])[0])(X[:64]))
    np.testing.assert_allclose(batched, emu.forward_emulator(X[:64]),
                               rtol=1e-11, atol=1e-12)


def test_a_monomial_sparse_fit_says_why_it_cannot_be_exported():
    emu, _X, _Y = _fit("monomial")
    with pytest.raises(NotImplementedError, match="downward closed"):
        JaxEmulator.from_sparse(emu)


def test_the_selected_index_set_really_is_not_downward_closed():
    """Pin the reason, so the deferral is not taken on faith."""
    emu, _X, _Y = _fit("legendre")
    mi = emu.multi_indices
    present = {tuple(int(v) for v in row) for row in mi}
    missing = [
        (tuple(a), i) for a in present for i in range(mi.shape[1])
        if a[i] > 0 and tuple(
            v - 1 if j == i else v for j, v in enumerate(a)
        ) not in present
    ]
    assert missing, (
        "the selection happened to be downward closed; the monomial deferral "
        "needs a different justification on this design"
    )
