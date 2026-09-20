"""T-002 / P3: response-driven sparse basis selection.

Selection runs in the orthonormal Legendre product basis. Sparsity is a
property of a basis, not of a function, and greedy selection needs a dictionary
of low mutual coherence, which monomials are not.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import eval_legendre

from MomentEmu.basis import Basis
from MomentEmu.emulator import generate_multi_indices
from MomentEmu.monomials import MonomialPlan
from MomentEmu.sparse import SparseEmu, simultaneous_omp

P = 5


def _L(a, v):
    return np.sqrt(2.0 * a + 1.0) * eval_legendre(a, v)


def _sparse_target(X):
    """Exactly four orthonormal terms, deliberately asymmetric: degree 12 in
    one parameter, degree 8 across a pair, degree 2 across another pair."""
    return (1.0 + 2.0 * _L(12, X[:, 0]) + 3.0 * _L(7, X[:, 1]) * _L(1, X[:, 2])
            + 0.5 * _L(1, X[:, 3]) * _L(1, X[:, 4]))


TRUTH = {(0,) * P, (12, 0, 0, 0, 0), (0, 7, 1, 0, 0), (0, 0, 0, 1, 1)}


@pytest.fixture(scope="module")
def grid():
    """Corner-anchored so the training box is exactly [-1, 1]^P and the
    Legendre map is the identity."""
    rng = np.random.default_rng(0)
    corners = np.array([[-1.0] * P, [1.0] * P])
    Xtr = np.vstack([corners, rng.uniform(-1, 1, (8000, P))])
    Xte = np.vstack([corners, rng.uniform(-1, 1, (3000, P))])
    return Xtr, Xte


def _fom(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def test_omp_recovers_an_exact_support_on_an_incoherent_dictionary():
    """Algorithm correctness, isolated from any basis question."""
    rng = np.random.default_rng(5)
    Phi = rng.standard_normal((400, 60))
    truth = np.zeros((60, 2))
    truth[[3, 17, 42], 0] = [2.0, -1.5, 0.75]
    truth[[3, 17, 42], 1] = [1.0, 0.5, -2.0]
    Y = Phi @ truth
    active, coef, info = simultaneous_omp(Phi, Y, n_terms=3)
    assert sorted(int(a) for a in active) == [3, 17, 42]
    rebuilt = np.zeros_like(truth)
    rebuilt[active] = coef
    np.testing.assert_allclose(rebuilt, truth, atol=1e-12)
    # The path is computed from ||Y||^2 - c.b_A, which cancels near an exact
    # fit and bottoms out around sqrt(eps); the coefficients above do not.
    assert info["residual_path"][-1] < 1e-6


def test_monomials_are_too_coherent_for_greedy_selection():
    """The design rationale, pinned. Greedy selection needs a dictionary whose
    columns are nearly orthogonal; monomials on [-1, 1] are nearly parallel."""
    rng = np.random.default_rng(0)
    v = rng.uniform(-1, 1, 200000)

    def coherence(cols):
        C = cols / np.linalg.norm(cols, axis=0, keepdims=True)
        G = np.abs(C.T @ C)
        np.fill_diagonal(G, 0.0)
        return float(G.max())

    mono = coherence(np.stack([v ** p for p in range(19)], axis=1))
    leg = coherence(np.stack([_L(a, v) for a in range(19)], axis=1))
    assert mono > 0.99, mono          # effectively one direction
    assert leg < 0.05, leg            # effectively orthogonal
    assert mono / leg > 20.0


def test_recovers_the_exact_legendre_support(grid):
    Xtr, Xte = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, ytr, candidate=Basis(degree=18, max_interaction=2),
                        degree=18, n_terms=4)
    assert {tuple(int(v) for v in r) for r in emu.multi_indices} == TRUTH
    assert _fom(emu.forward_emulator(Xte).ravel(), _sparse_target(Xte)) < 1e-8


def test_selection_is_shared_across_outputs(grid):
    """One index set for every output keeps inference a single GEMM."""
    Xtr, Xte = grid
    y = _sparse_target(Xtr)
    Ytr = np.column_stack([y, 2.0 * y + 1.0, -0.5 * y])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, Ytr, candidate=Basis(degree=18, max_interaction=2),
                        degree=18, n_terms=4)
    assert emu.coefficients.shape == (4, 3)
    assert {tuple(int(v) for v in r) for r in emu.multi_indices} == TRUTH
    ref = _sparse_target(Xte)
    pred = emu.forward_emulator(Xte)
    for col, expect in enumerate((ref, 2.0 * ref + 1.0, -0.5 * ref)):
        assert _fom(pred[:, col], expect) < 1e-7


def test_beats_a_fixed_rule_basis_at_a_fraction_of_the_size(grid):
    """The point of P3: an asymmetric reach no prior truncation can state."""
    Xtr, Xte = grid

    def target(X):
        return (1.0 / (1.0 + 20.0 * (X[:, 0] + 0.5 * X[:, 1]) ** 2)
                + 0.7 * X[:, 2] * X[:, 3] + 0.3 * X[:, 4])

    ytr, yte = target(Xtr).reshape(-1, 1), target(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sparse = SparseEmu(Xtr, ytr, candidate=Basis(degree=18, max_interaction=2),
                           degree=18, n_terms=60)
    e_sparse = _fom(sparse.forward_emulator(Xte).ravel(), yte)

    for d in (3, 6, 8):
        mi = generate_multi_indices(P, d)
        plan = MonomialPlan.build(mi)
        A = plan.evaluate(Xtr)
        c = np.linalg.solve(A.T @ A + 1e-10 * np.eye(mi.shape[0]), A.T @ ytr)
        e_iso = _fom((plan.evaluate(Xte) @ c).ravel(), yte)
        assert e_sparse < e_iso, f"degree {d} ({mi.shape[0]} terms): {e_sparse} vs {e_iso}"
    assert sparse.report()["compression"] > 20.0


def test_auto_selects_a_size_and_reports_the_path(grid):
    Xtr, Xte = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, ytr, candidate=Basis(degree=18, max_interaction=2),
                        degree=18, n_terms="auto", max_terms=25, random_state=0)
    assert len(emu.error_path) == 25
    assert emu.multi_indices.shape[0] == int(np.argmin(emu.error_path)) + 1
    assert emu.multi_indices.shape[0] <= 8       # it finds the short answer
    assert _fom(emu.forward_emulator(Xte).ravel(), _sparse_target(Xte)) < 1e-6


def test_auto_warns_when_the_path_was_truncated(grid):
    Xtr, _ = grid

    def hard(X):
        return 1.0 / (1.0 + 20.0 * (X[:, 0] + 0.5 * X[:, 1]) ** 2)

    with pytest.warns(UserWarning, match="still falling"):
        SparseEmu(Xtr, hard(Xtr).reshape(-1, 1),
                  candidate=Basis(degree=18, max_interaction=2), degree=18,
                  n_terms="auto", max_terms=4, random_state=0)


def test_residual_path_is_monotone(grid):
    """Each exact re-solve on a larger active set cannot increase the residual."""
    Xtr, _ = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, ytr, candidate=Basis(degree=12, max_interaction=2),
                        degree=12, n_terms=25)
    assert np.all(np.diff(np.asarray(emu.residual_path)) <= 1e-12)


def test_report_exposes_the_selected_structure(grid):
    Xtr, _ = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, ytr, candidate=Basis(degree=18, max_interaction=2),
                        degree=18, n_terms=4)
    r = emu.report()
    assert r["n_terms"] == 4 and r["n_candidates"] == 1621
    assert r["max_total_degree"] == 12
    assert r["max_interaction"] == 2
    assert r["max_degree_per_parameter"] == (12, 7, 1, 1, 1)


def test_omp_skips_candidates_already_in_the_span():
    rng = np.random.default_rng(3)
    Phi = rng.standard_normal((200, 6))
    Phi[:, 3] = Phi[:, 0] + Phi[:, 1]
    Y = (Phi[:, 0] - 2.0 * Phi[:, 2]).reshape(-1, 1)
    active, coef, info = simultaneous_omp(Phi, Y, n_terms=6)
    assert np.isfinite(coef).all()
    assert info["residual_path"][-1] < 1e-10


def test_memory_guard_refuses_an_oversized_candidate_design():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40000, 8))
    with pytest.raises(MemoryError, match="candidate design"):
        SparseEmu(X, X[:, :1], candidate=Basis(degree=12, max_interaction=4),
                  degree=12, n_terms=10)


def test_candidate_validation(grid):
    Xtr, _ = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with pytest.raises(ValueError, match="candidate|degree"):
        SparseEmu(Xtr, ytr)
    with pytest.raises(ValueError, match=r"\(D, 5\)"):
        SparseEmu(Xtr, ytr, candidate=np.zeros((10, 3), dtype=np.int64), n_terms=2)


# --- boundary probe follow-up: accumulated conditioning --------------------

def test_omp_warns_when_the_active_gram_is_ill_conditioned():
    """The per-column pivot guard is not a conditioning guard: conditioning
    accumulates. A column c0 + eps*u has pivot eps**2, so the 1e-12 guard
    admits it at eps > 1e-6, where the active Gram [[1,1],[1,1+eps**2]] has
    cond about 4/eps**2 -- four times the level at which the dense solver
    already calls its coefficients untrustworthy."""
    from MomentEmu.guards import COND_WARN, IllConditionedWarning

    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((400, 5)))
    c0, u = Q[:, 0], Q[:, 3]
    eps = 1.01e-6
    Phi = np.column_stack([c0, Q[:, 1], Q[:, 2], c0 + eps * u])
    Y = (c0 + 0.5 * u).reshape(-1, 1)

    with pytest.warns(IllConditionedWarning, match="active Gram"):
        active, coef, info = simultaneous_omp(Phi, Y, n_terms=4)
    assert 3 in {int(a) for a in active}
    G = Phi[:, active].T @ Phi[:, active]
    assert info["cond"] >= COND_WARN
    assert info["cond"] == pytest.approx(np.linalg.cond(G), rel=0.05)


def test_omp_is_quiet_on_a_well_conditioned_problem():
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((400, 5)))
    Y = (Q[:, 0] + 0.5 * Q[:, 3]).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, info = simultaneous_omp(Q, Y, n_terms=4)
    assert info["cond"] < 1e3


def test_conditioning_is_reported_even_when_it_stays_quiet(grid):
    """Every run reports the number, so a caller can gate on it without
    having to catch a warning."""
    Xtr, _ = grid
    ytr = _sparse_target(Xtr).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = SparseEmu(Xtr, ytr, candidate=Basis(degree=18, max_interaction=2),
                        degree=18, n_terms=8)
    assert np.isfinite(emu.cond) and emu.cond >= 1.0
