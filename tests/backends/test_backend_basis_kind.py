"""T-007: every backend must honour basis_kind, or say it cannot.

Reported from a 21cmVAE-jax port. ``jax_momentemu`` read ``plan.levels``,
which only ``MonomialPlan`` has, so a Legendre or Chebyshev fit raised
``AttributeError: 'ChebyshevPlan' object has no attribute 'levels'`` and could
not reach JAX at all.

Probing the other two backends turned up a worse case. ``torch_momentemu``
rebuilt the design with ``MonomialPlan.build`` whatever the fit used, so it
did not raise: it returned the wrong numbers, 4.8e-01 and 3.7e-01 relative on
a degree-5 fit. ``symbolic_momentemu`` already raised ``NotImplementedError``
naming the basis, which is the behaviour the other two are held to here.

The gap matters because the package tells users to reach for these bases: the
cond(M) warning names ``basis_kind`` as a way to recover digits, and taking
that advice must not cost the backends. (T-009 later rewrote that warning to
say the advice is about coefficients, not predictions; the backends still
have to support the choice.)
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from MomentEmu.jax_momentemu import create_jax_emulator  # noqa: E402
from MomentEmu.symbolic_momentemu import create_symbolic_emulator  # noqa: E402
from MomentEmu.torch_momentemu import create_torch_emulator  # noqa: E402

KINDS = ("monomial", "legendre", "chebyshev")
A = np.array([1.0, -0.5, 2.0])


def _design(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = np.column_stack([np.tanh(X @ A), (X ** 2).sum(axis=1)])
    return X, Y


@pytest.fixture(scope="module", params=KINDS)
def fitted(request):
    X, Y = _design()
    emu = PolyEmu(X, Y, basis_kind=request.param, max_degree_forward=6,
                  RMSE_tol=0.0, verbose=0)
    return request.param, emu, X


def _rel(got, ref):
    return float(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-30))


def test_jax_export_reproduces_the_fit(fitted) -> None:
    kind, emu, X = fitted
    J = create_jax_emulator(emu)
    got = np.asarray(J(X[:32]))
    assert _rel(got, emu.forward_emulator(X[:32])) < 1e-11, kind


def test_torch_export_reproduces_the_fit(fitted) -> None:
    """The silent one: torch used to answer with a different basis entirely."""
    kind, emu, X = fitted
    T = create_torch_emulator(emu)
    got = T(torch.as_tensor(X[:32], dtype=torch.float64)).detach().numpy()
    assert _rel(got, emu.forward_emulator(X[:32])) < 1e-11, kind


def test_jax_jacobian_matches_the_analytic_one(fitted) -> None:
    """A wrong basis would still differentiate cleanly, so check the value."""
    kind, emu, X = fitted
    J = create_jax_emulator(emu)
    got = np.asarray(jax.jacfwd(J.evaluate)(X[:4]))
    got = np.stack([got[i, :, i, :] for i in range(4)])
    assert _rel(got, np.atleast_3d(emu.jacobian(X[:4]))) < 1e-9, kind


def test_jax_export_is_jittable(fitted) -> None:
    kind, emu, X = fitted
    J = create_jax_emulator(emu)
    assert np.allclose(np.asarray(jax.jit(J.evaluate)(X[:8])),
                       np.asarray(J.evaluate(X[:8]))), kind


@pytest.mark.parametrize("kind", ("legendre", "chebyshev"))
def test_symbolic_exports_a_non_monomial_fit_in_its_own_family(kind: str) -> None:
    """This asserted a refusal, and the refusal was half right.

    Reading the coefficients as monomial ones would indeed be silently wrong,
    which is what it protected against. But that is an argument for printing
    the family's own polynomials, not for declining to print anything, and
    while it stood this test held the other two backends to the weakest of the
    three. All three now export a non-monomial fit and agree with the numpy
    model, which is what the comparison below has teeth against.
    """
    import sympy as sp

    X, Y = _design()
    emu = PolyEmu(X, Y, basis_kind=kind, max_degree_forward=4, verbose=0)
    bundle = create_symbolic_emulator(emu)
    sym = bundle["emulator"]
    got = np.column_stack([
        np.asarray(f(*X[:16].T), dtype=float) * np.ones(16)
        for f in sym.lambdified
    ])
    assert np.allclose(got, emu.forward_emulator(X[:16]), rtol=1e-9, atol=1e-10), kind
    clipped = set().union(*(e.atoms(sp.Min, sp.Max) for e in sym.expressions))
    assert clipped, f"{kind}: the export lost the clip _TensorPlan applies"


def test_the_bases_disagree_so_the_checks_have_teeth() -> None:
    """A monomial evaluation of a Legendre fit must be visibly wrong.

    Without this, a backend that quietly rebuilt the monomial design could
    pass the tests above if the two happened to agree.
    """
    from MomentEmu.monomials import LegendrePlan, MonomialPlan

    X, Y = _design()
    emu = PolyEmu(X, Y, basis_kind="legendre", max_degree_forward=6,
                  RMSE_tol=0.0, verbose=0)
    mi = emu.forward_multi_indices
    Xs = emu.scaler_X.transform(X[:32])
    right = LegendrePlan.build(mi).evaluate(Xs)
    wrong = MonomialPlan.build(mi).evaluate(Xs)
    assert _rel(wrong, right) > 0.1
