"""T-010: the Torch backend takes a sparse fit too, in every basis.

The same gap as the JAX one, and the same shape as T-007's: the constructor
was written against ``PolyEmu`` and read ``forward_coeffs`` first, so a
``SparseEmu`` was turned away with "the Torch backend needs a forward
emulator" -- a message that is not merely unhelpful but wrong, since a sparse
fit IS a forward emulator.

The evaluation changes mirror the JAX ones. A selected index set is not
downward closed, so the closure walk cannot evaluate it and the multi-index
path takes over; and that path must not clip a monomial fit, which describes a
diverging model outside its training box.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.sparse import SparseEmu

torch = pytest.importorskip("torch")

from MomentEmu.torch_momentemu import TorchMomentEmu  # noqa: E402

ALL_BASES = ("legendre", "chebyshev", "monomial")


def _design(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = np.column_stack([np.sin(1.4 * X[:, 0]) + X[:, 1] ** 2 - 0.3 * X[:, 2],
                         np.cos(X[:, 1] * X[:, 2]) + 0.5 * X[:, 0]])
    return X, Y


def _fit(basis_kind: str):
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SparseEmu(X, Y, degree=5, n_terms=45, basis_kind=basis_kind), X, Y


def _call(model, X):
    return model(torch.as_tensor(np.asarray(X), dtype=torch.float64)).detach().numpy()


@pytest.mark.parametrize("basis_kind", ALL_BASES)
def test_a_sparse_fit_agrees_with_the_numpy_model(basis_kind: str):
    emu, X, _Y = _fit(basis_kind)
    np.testing.assert_allclose(_call(TorchMomentEmu(emu), X),
                               emu.forward_emulator(X), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("basis_kind", ALL_BASES)
def test_the_gradient_matches_finite_differences(basis_kind: str):
    emu, X, _Y = _fit(basis_kind)
    model = TorchMomentEmu(emu)
    point = torch.tensor(X[:1], dtype=torch.float64, requires_grad=True)
    model(point).sum().backward()
    got = point.grad.detach().numpy()[0]

    step = 1e-6
    ref = np.empty_like(got)
    for i in range(X.shape[1]):
        lo, hi = X[:1].copy(), X[:1].copy()
        lo[0, i] -= step
        hi[0, i] += step
        ref[i] = (emu.forward_emulator(hi).sum() - emu.forward_emulator(lo).sum()) / (2 * step)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_a_monomial_export_diverges_outside_the_box_like_numpy():
    """The clip belongs to the tensor families, not to the shared path."""
    emu, _X, _Y = _fit("monomial")
    model = TorchMomentEmu(emu)
    far = np.array([[2.5, -3.0, 2.0]])
    got = _call(model, far)
    np.testing.assert_allclose(got, emu.forward_emulator(far), rtol=1e-9, atol=1e-10)
    edge = _call(model, np.array([[1.0, 1.0, 1.0]]))
    assert np.abs(got).max() > 5.0 * np.abs(edge).max(), (
        "the monomial export saturated; it is being clipped"
    )


@pytest.mark.parametrize("basis_kind", ("legendre", "chebyshev"))
def test_a_tensor_export_still_saturates_outside_the_box(basis_kind: str):
    emu, _X, _Y = _fit(basis_kind)
    far = np.array([[2.5, -3.0, 2.0]])
    np.testing.assert_allclose(_call(TorchMomentEmu(emu), far),
                               emu.forward_emulator(far), rtol=1e-9, atol=1e-10)


def test_a_dense_fit_still_uses_the_closure_walk():
    """Regression: the multi-index path must not take over the dense route."""
    from MomentEmu.emulator import PolyEmu

    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense = PolyEmu(X, Y, init_deg_forward=4, max_degree_forward=4,
                        RMSE_tol=0.0, verbose=0)
    model = TorchMomentEmu(dense)
    assert model._plan is not None and hasattr(model._plan, "n_closure")
    np.testing.assert_allclose(_call(model, X), dense.forward_emulator(X),
                               rtol=1e-10, atol=1e-11)


def test_the_old_message_no_longer_misdescribes_a_sparse_fit():
    """It said "needs a forward emulator" of something that is one."""
    emu, X, _Y = _fit("legendre")
    model = TorchMomentEmu(emu)
    assert model.n_params == 3
    assert model.n_outputs == 2
    assert _call(model, X).shape == (X.shape[0], 2)
