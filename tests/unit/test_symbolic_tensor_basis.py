"""Symbolic export writes the family's polynomials, for every basis.

It used to refuse anything but monomial, and the refusal was right for what it
would otherwise have done: the stored coefficients multiply Legendre or
Chebyshev polynomials, so printing them against monomials is silently wrong.
The fix is to print the right polynomials, not to reinterpret the numbers.

sympy's families agree with the package's recurrences to 4.4e-16, so
``legendre(k, z) * sqrt(2k+1)`` and ``chebyshevt(k, z)`` are the same basis
the fit used.

The clip goes INTO the expression. ``_TensorPlan`` clips its argument to
[-1, 1] so the basis saturates outside the training box; an export that left
the clip out would agree inside the box and describe a different model outside
it, which is the failure T-007 found in the Torch backend. sympy's Min and Max
survive lambdify under numpy and vectorise, so there is no reason to accept
that.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu
from MomentEmu.sparse import SparseEmu

sp = pytest.importorskip("sympy")

BASES = ("monomial", "legendre", "chebyshev")
NAMES = ["a", "b"]


def _design(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 2))
    Y = np.column_stack([np.sin(1.3 * X[:, 0]) + X[:, 1] ** 2,
                         np.cos(X[:, 1]) - 0.4 * X[:, 0]])
    return X, Y


def _dense(basis_kind, degree=4):
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PolyEmu(X, Y, init_deg_forward=degree, max_degree_forward=degree,
                       RMSE_tol=0.0, verbose=0, basis_kind=basis_kind), X, Y


def _evaluate(exprs, X):
    syms = sp.symbols(NAMES)
    out = []
    for e in exprs:
        f = sp.lambdify(syms, e, "numpy")
        out.append(np.asarray(f(*X.T), dtype=float) * np.ones(X.shape[0]))
    return np.column_stack(out)


@pytest.mark.parametrize("basis_kind", BASES)
def test_the_export_reproduces_the_fit_inside_the_box(basis_kind: str):
    emu, X, _Y = _dense(basis_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = emu.generate_forward_symb_emu(NAMES)
    np.testing.assert_allclose(_evaluate(exprs, X), emu.forward_emulator(X),
                               rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("basis_kind", ("legendre", "chebyshev"))
def test_the_export_carries_the_clip_so_it_agrees_outside_too(basis_kind: str):
    emu, _X, _Y = _dense(basis_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = emu.generate_forward_symb_emu(NAMES)
        far = np.array([[3.0, -2.5], [-4.0, 5.0]])
        ref = emu.forward_emulator(far, extrapolation="ignore")
    np.testing.assert_allclose(_evaluate(exprs, far), ref, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("basis_kind", ("legendre", "chebyshev"))
def test_a_sparse_tensor_fit_exports_too(basis_kind: str):
    X, Y = _design()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(X, Y, degree=4, n_terms=20, basis_kind=basis_kind)
        exprs = emu.generate_forward_symb_emu(NAMES)
    np.testing.assert_allclose(_evaluate(exprs, X), emu.forward_emulator(X),
                               rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("basis_kind", BASES)
def test_only_a_tensor_export_carries_a_clip(basis_kind: str):
    """Assert the clip itself, not a magnitude that depends on the target.

    Whether a monomial fit blows up at a particular far point is a property of
    the coefficients; whether the export clips is a property of the code.
    """
    emu, _X, _Y = _dense(basis_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = emu.generate_forward_symb_emu(NAMES)
    clipped = set().union(*(e.atoms(sp.Min, sp.Max) for e in exprs))
    if basis_kind == "monomial":
        assert not clipped, "the monomial export picked up the tensor clip"
    else:
        assert clipped, "the tensor export lost its clip"


def test_the_monomial_export_still_matches_far_outside_the_box():
    emu, _X, _Y = _dense("monomial")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = emu.generate_forward_symb_emu(NAMES)
        far = np.array([[4.0, -4.0]])
        ref = emu.forward_emulator(far, extrapolation="ignore")
    np.testing.assert_allclose(_evaluate(exprs, far), ref, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("basis_kind", BASES)
def test_the_expression_uses_the_requested_names(basis_kind: str):
    emu, _X, _Y = _dense(basis_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exprs = emu.generate_forward_symb_emu(NAMES)
    free = set().union(*(e.free_symbols for e in exprs))
    assert {str(s) for s in free} == set(NAMES)
