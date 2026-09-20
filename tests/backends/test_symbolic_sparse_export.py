"""T-010: a sparse fit exports symbolically, for the basis PolyEmu supports.

SymbolicMomentEmu delegates to ``trained_emulator.generate_forward_symb_emu``,
a PolyEmu method, so a SparseEmu failed with
``AttributeError: 'SparseEmu' object has no attribute 'n_params'`` -- the raw
shape of the same gap T-007 and the JAX and Torch halves of T-010 closed.

Both now go through the same builder, which writes the family's own
polynomials rather than reinterpreting the coefficients, so every basis
exports. Default variable names follow PolyEmu's, which are 1-based.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.sparse import SparseEmu

sp = pytest.importorskip("sympy")

from MomentEmu.symbolic_momentemu import SymbolicMomentEmu  # noqa: E402


def _design(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = np.column_stack([np.sin(1.2 * X[:, 0]) + X[:, 1] ** 2,
                         np.cos(X[:, 2]) - 0.4 * X[:, 0]])
    return X, Y


def _fit(basis_kind="monomial", n_terms=25):
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SparseEmu(X, Y, degree=4, n_terms=n_terms,
                         basis_kind=basis_kind), X, Y


def test_the_expressions_reproduce_the_numpy_model():
    emu, X, _Y = _fit()
    exprs = emu.generate_forward_symb_emu()
    assert len(exprs) == 2
    # The builder names variables x1..xn, as PolyEmu's export does.
    names = sp.symbols([f"x{i + 1}" for i in range(3)])
    funcs = [sp.lambdify(names, e, "numpy") for e in exprs]
    got = np.column_stack([f(*X.T) for f in funcs])
    np.testing.assert_allclose(got, emu.forward_emulator(X), rtol=1e-9, atol=1e-10)


def test_the_symbolic_wrapper_accepts_a_sparse_fit():
    emu, X, _Y = _fit()
    sym = SymbolicMomentEmu(emu)
    got = np.column_stack([f(*X[:20].T) for f in sym.lambdified])
    np.testing.assert_allclose(got, emu.forward_emulator(X[:20]),
                               rtol=1e-9, atol=1e-10)


def test_custom_variable_names_are_used():
    emu, _X, _Y = _fit()
    exprs = emu.generate_forward_symb_emu(["a", "b", "c"])
    free = set().union(*(e.free_symbols for e in exprs))
    assert {str(s) for s in free} <= {"a", "b", "c"}


def test_the_expression_is_sparse_not_the_closure():
    """The point of exporting a sparse fit is that it is small."""
    emu, _X, _Y = _fit(n_terms=25)
    expr = sp.expand(emu.generate_forward_symb_emu()[0])
    assert len(expr.as_ordered_terms()) <= 3 * emu.multi_indices.shape[0]


@pytest.mark.parametrize("basis_kind", ("legendre", "chebyshev"))
def test_a_tensor_basis_exports_in_its_own_family(basis_kind: str):
    """This used to raise. The refusal was right for a reinterpretation of the
    coefficients and wrong as a limit: the fix was to print the family."""
    emu, X, _Y = _fit(basis_kind)
    exprs = emu.generate_forward_symb_emu()
    names = sp.symbols([f"x{i + 1}" for i in range(3)])
    got = np.column_stack([
        np.asarray(sp.lambdify(names, e, "numpy")(*X.T), dtype=float)
        for e in exprs
    ])
    np.testing.assert_allclose(got, emu.forward_emulator(X), rtol=1e-9, atol=1e-10)


def test_n_params_and_n_outputs_are_available():
    """SymbolicMomentEmu reads n_params; the AttributeError started there."""
    emu, _X, _Y = _fit()
    assert emu.n_params == 3
    assert emu.n_outputs == 2
