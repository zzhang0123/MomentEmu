"""T-011: the (k, m) output stage reaches every backend, or none of them.

The numpy layer saves nothing on export until the backends carry the two
arrays: they were re-expanding to m outputs, so the D x k coefficient matrix
became D x m again on the way out.

The derivation lives in CompressedEmu.export_payload and nowhere else. Three
backends deriving the same algebra three times is the T-007 failure, where the
Torch backend rebuilt the monomial design whatever the fit used and returned
4.8e-01 relative without raising. The last test here holds the three against
each other for that reason, not for coverage.

Compression is applied AFTER the model's output transform, so it folds into
the coefficients only when that transform is linear. A log_Y fit still
compresses in numpy and is refused for export, which is stated rather than
silently producing a different model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.compress import CompressedEmu
from MomentEmu.emulator import PolyEmu

RANK = 5


def _spectra(n=400, m=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    t = np.linspace(0.0, 1.0, m)
    Y = (np.tanh(2.0 * X[:, :1]) * np.sin(3.0 * t)[None, :]
         + (X[:, 1:2] ** 2) * np.exp(-((t - 0.4) ** 2) / 0.05)[None, :]
         + X[:, 2:3] * t[None, :])
    return X, Y


def _wrapped(basis_kind="monomial", rank=RANK):
    X, Y = _spectra()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PolyEmu(X, Y, init_deg_forward=4, max_degree_forward=4,
                        RMSE_tol=0.0, verbose=0, basis_kind=basis_kind)
        return CompressedEmu(model, X, Y, rank=rank), X, Y


def _reference(wrapped, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wrapped.forward_emulator(X)


def test_the_payload_reproduces_the_numpy_layer():
    """The algebra, checked once, where every backend reads it from."""
    wrapped, X, _Y = _wrapped()
    pay = wrapped.export_payload()
    inner = wrapped.model
    Phi = inner.forward_plan.evaluate(
        (X - inner.scaler_X.mean_) / inner.scaler_X.scale_
    )
    got = (Phi @ pay.coeffs) @ pay.modes + pay.offset
    np.testing.assert_allclose(got, _reference(wrapped, X), rtol=1e-10, atol=1e-11)


def test_the_payload_is_actually_smaller():
    wrapped, _X, Y = _wrapped()
    pay = wrapped.export_payload()
    assert pay.coeffs.shape[1] == RANK
    assert pay.modes.shape == (RANK, Y.shape[1])
    assert pay.coeffs.size + pay.modes.size < wrapped.model.forward_coeffs.size


def test_a_non_linear_output_transform_is_refused():
    X, Y = _spectra()
    Y = np.abs(Y) + 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PolyEmu(X, Y, init_deg_forward=3, max_degree_forward=3,
                        RMSE_tol=0.0, verbose=0, log_Y=True)
        wrapped = CompressedEmu(model, X, Y, rank=3)
        # numpy still works: it compresses after the transform
        assert wrapped.forward_emulator(X).shape == Y.shape
    with pytest.raises(NotImplementedError, match="linear"):
        wrapped.export_payload()


def test_jax_carries_the_stage():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from MomentEmu.jax_momentemu import JaxEmulator

    wrapped, X, Y = _wrapped()
    je = JaxEmulator.from_polyemu(wrapped)
    assert je.coeffs.shape[1] == RANK, "the export re-expanded the outputs"
    np.testing.assert_allclose(np.asarray(je(X)), _reference(wrapped, X),
                               rtol=1e-10, atol=1e-11)


def test_torch_carries_the_stage():
    torch = pytest.importorskip("torch")
    from MomentEmu.torch_momentemu import TorchMomentEmu

    wrapped, X, Y = _wrapped()
    tm = TorchMomentEmu(wrapped)
    assert tm.coeffs.shape[1] == RANK, "the export re-expanded the outputs"
    got = tm(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(got, _reference(wrapped, X), rtol=1e-10, atol=1e-11)


def test_symbolic_carries_the_stage():
    sp = pytest.importorskip("sympy")
    from MomentEmu.symbolic_momentemu import SymbolicMomentEmu

    wrapped, X, Y = _wrapped()
    sym = SymbolicMomentEmu(wrapped)
    assert len(sym.expressions) == Y.shape[1]
    got = np.column_stack([
        np.asarray(f(*X[:12].T), dtype=float) * np.ones(12)
        for f in sym.lambdified
    ])
    np.testing.assert_allclose(got, _reference(wrapped, X[:12]),
                               rtol=1e-9, atol=1e-10)


def test_the_three_backends_agree_with_each_other():
    """The anti-divergence check: one derivation, three consumers."""
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    sp = pytest.importorskip("sympy")
    jax.config.update("jax_enable_x64", True)
    from MomentEmu.jax_momentemu import JaxEmulator
    from MomentEmu.symbolic_momentemu import SymbolicMomentEmu
    from MomentEmu.torch_momentemu import TorchMomentEmu

    wrapped, X, _Y = _wrapped()
    Xs = X[:12]
    a = np.asarray(JaxEmulator.from_polyemu(wrapped)(Xs))
    b = TorchMomentEmu(wrapped)(
        torch.as_tensor(Xs, dtype=torch.float64)
    ).detach().numpy()
    sym = SymbolicMomentEmu(wrapped)
    c = np.column_stack([
        np.asarray(f(*Xs.T), dtype=float) * np.ones(Xs.shape[0])
        for f in sym.lambdified
    ])
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(a, c, rtol=1e-9, atol=1e-10)
