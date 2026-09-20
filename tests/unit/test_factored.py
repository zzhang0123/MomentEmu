"""T-002: multiplicative (low-rank) structure by CP alternating least squares."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.factored import FactoredEmu, separability_report
from MomentEmu.monomials import MonomialPlan

P = 9
BLOCKS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def _f1(X):
    return np.sin(1.5 * X[:, 0] + 0.4 * X[:, 1] * X[:, 2])


def _f2(X):
    return X[:, 3] + 0.5 * X[:, 4] * X[:, 5] - 0.2


def _f3(X):
    return np.cos(1.2 * X[:, 6] - 0.5 * X[:, 7]) - 0.3 * X[:, 8] - 0.55


def _product(X):
    return _f1(X) * _f2(X) * _f3(X)


def _rank2(X):
    b = ((X[:, 0] * X[:, 1] - 0.3) * np.sin(2.0 * X[:, 4] + 0.7 * X[:, 3])
         * (0.4 - X[:, 7] ** 2 + 0.5 * X[:, 6]))
    return _product(X) + 0.8 * b


@pytest.fixture(scope="module")
def grid():
    rng = np.random.default_rng(2)
    return rng.uniform(-1, 1, (12000, P)), rng.uniform(-1, 1, (3000, P))


def _fom(pred, ref):
    return float(np.sqrt(np.mean((pred.ravel() - ref.ravel()) ** 2))
                 / np.sqrt(np.mean(ref ** 2)))


def test_log_transform_is_unavailable_on_this_target(grid):
    """Motivation: the product changes sign and passes through zero."""
    Xtr, _ = grid
    y = _product(Xtr)
    assert (y <= 0).sum() > 0.4 * y.size
    assert np.min(np.abs(y)) < 1e-4


def test_rank_one_beats_the_isotropic_basis_by_orders_of_magnitude(grid):
    Xtr, Xte = grid
    ytr, yte = _product(Xtr), _product(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp = FactoredEmu(Xtr, ytr, BLOCKS, rank=1, degree=5, random_state=0)
    e_cp = _fom(cp.forward_emulator(Xte), yte)

    mi = generate_multi_indices(P, 6)
    plan = MonomialPlan.build(mi)
    A = plan.evaluate(Xtr)
    c = np.linalg.solve(A.T @ A, A.T @ ytr)
    e_poly = _fom(plan.evaluate(Xte) @ c, yte)

    assert cp.n_parameters < mi.shape[0] / 10
    assert e_cp < e_poly / 5.0, f"cp {e_cp:.3e} vs isotropic degree 6 {e_poly:.3e}"


def test_additive_basis_fails_on_a_product(grid):
    """The additive block basis explains almost nothing here, which is why the
    multiplicative case needs its own model rather than a different index set."""
    from MomentEmu.basis import Basis

    Xtr, Xte = grid
    ytr, yte = _product(Xtr), _product(Xte)
    mi = Basis.separable(BLOCKS).build([f"x{i}" for i in range(P)], 6)
    plan = MonomialPlan.build(mi)
    A = plan.evaluate(Xtr)
    c = np.linalg.solve(A.T @ A, A.T @ ytr)
    assert _fom(plan.evaluate(Xte) @ c, yte) > 0.9


def test_rank_must_reach_the_targets_true_rank(grid):
    """A rank-2 target is out of reach at rank 1 however many restarts are
    spent; at rank 2 it is reached."""
    Xtr, Xte = grid
    ytr, yte = _rank2(Xtr), _rank2(Xte)
    got = {}
    for rank in (1, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = FactoredEmu(Xtr, ytr, BLOCKS, rank=rank, degree=5, random_state=0)
        got[rank] = _fom(m.forward_emulator(Xte), yte)
    assert got[1] > 0.3
    assert got[2] < 0.1 * got[1]


def test_restart_spread_is_reported(grid):
    """ALS is non-convex at the correct rank too, so the selection over
    restarts is load-bearing and its spread is exposed, not hidden."""
    Xtr, _ = grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = FactoredEmu(Xtr, _rank2(Xtr), BLOCKS, rank=2, degree=5,
                        n_restarts=6, random_state=0)
    rep = m.report()
    assert len(rep["restart_residuals"]) == 6
    assert m.train_residual == min(rep["restart_residuals"])
    assert rep["restart_spread"] >= 1.0


def test_multi_output_is_supported(grid):
    Xtr, Xte = grid
    Ytr = np.column_stack([_product(Xtr), 2.0 * _product(Xtr) + 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp = FactoredEmu(Xtr, Ytr, BLOCKS, rank=1, degree=4, random_state=0)
    pred = cp.forward_emulator(Xte)
    assert pred.shape == (Xte.shape[0], 2)
    ref = np.column_stack([_product(Xte), 2.0 * _product(Xte) + 1.0])
    assert _fom(pred[:, 0], ref[:, 0]) < 0.05
    assert _fom(pred[:, 1], ref[:, 1]) < 0.05


def test_additive_target_is_the_rank_k_special_case(grid):
    """f = sum_k f_k is a rank-K product model with the other factors constant."""
    Xtr, Xte = grid
    def add(X):
        return _f1(X) + _f2(X) + _f3(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cp = FactoredEmu(Xtr, add(Xtr), BLOCKS, rank=3, degree=5,
                         n_restarts=4, random_state=0)
    assert _fom(cp.forward_emulator(Xte), add(Xte)) < 0.02


@pytest.mark.parametrize("kwargs, match", [
    ({"rank": 0}, "rank"),
    ({"degree": [3, 4]}, "one value per block"),
    ({"blocks": ((0, 1), (1, 2))}, "disjoint"),
])
def test_validation(grid, kwargs, match):
    Xtr, _ = grid
    blocks = kwargs.pop("blocks", BLOCKS)
    kwargs.setdefault("degree", 2)
    with pytest.raises(ValueError, match=match):
        FactoredEmu(Xtr[:500], _product(Xtr[:500]), blocks, **kwargs)


def _g1(X):
    return 1.2 + np.sin(0.9 * X[:, 0] + 0.7 * X[:, 1] + 0.5 * X[:, 2])


def _g2(X):
    return 1.6 + 0.5 * np.exp(0.4 * (X[:, 3] + 0.8 * X[:, 4] + 0.6 * X[:, 5]))


def _g3(X):
    return 1.4 + np.cos(1.2 * X[:, 6] - 0.5 * X[:, 7] + 0.8 * X[:, 8])


@pytest.mark.parametrize("build, expected", [
    (lambda X: _g1(X) * _g2(X) * _g3(X), "multiplicative"),
    (lambda X: _g1(X) + _g2(X) + _g3(X), "additive"),
])
def test_separability_report_names_the_structure(build, expected):
    """M_ij vanishes across blocks for a product, H_ij for a sum. Neither
    criterion takes a logarithm, so neither needs the data to be positive."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (4000, P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, build(X).reshape(-1, 1), max_degree_forward=5, verbose=0)
    rep = separability_report(emu, X[:400], blocks=BLOCKS)
    assert rep["structure"] == expected, rep
    if expected == "multiplicative":
        assert rep["multiplicative_ratio"] < rep["additive_ratio"] / 3.0
    else:
        assert rep["additive_ratio"] < rep["multiplicative_ratio"] / 3.0


def test_separability_report_handles_a_sign_changing_product():
    """The case the log transform cannot reach at all."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-0.8, 0.8, (4000, P))
    y = (np.sin(0.9*X[:,0] + 0.7*X[:,1] + 0.5*X[:,2])
         * (X[:,3] + 0.8*X[:,4] + 0.6*X[:,5])
         * np.cos(1.2*X[:,6] - 0.5*X[:,7] + 0.8*X[:,8]))
    assert (y <= 0).sum() > 0.3 * y.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, y.reshape(-1, 1), max_degree_forward=5, verbose=0)
    rep = separability_report(emu, X[:400], blocks=BLOCKS)
    assert rep["multiplicative_ratio"] < rep["additive_ratio"]


def test_separability_report_rejects_a_degenerate_partition():
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (800, P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, _g1(X).reshape(-1, 1), max_degree_forward=3, verbose=0)
    with pytest.raises(ValueError, match="within-block pair"):
        separability_report(emu, X[:100], blocks=tuple((i,) for i in range(P)))


# --- boundary probe follow-up: step validation ----------------------------

@pytest.mark.parametrize("step", [0.0, -1e-4, float("nan"), float("inf"), 1e-16, 1.0])
def test_separability_report_rejects_an_unusable_step(step):
    """A step of zero divides by zero and returns an infinite ratio, which
    reads as 'neither' rather than as the error it is; too small a step lets
    the central difference cancel and flips the verdict."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (1200, P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, _g1(X).reshape(-1, 1), max_degree_forward=3, verbose=0)
    with pytest.raises(ValueError, match="step must be finite"):
        separability_report(emu, X[:200], blocks=BLOCKS, step=step)


def test_separability_report_is_stable_across_the_usable_step_range():
    """Inside the accepted range the verdict must not depend on the step."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.8, 0.8, (3000, P))
    y = _g1(X) * _g2(X) * _g3(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = PolyEmu(X, y.reshape(-1, 1), max_degree_forward=5, verbose=0)
    verdicts = {
        s: separability_report(emu, X[:300], blocks=BLOCKS, step=s)["structure"]
        for s in (1e-9, 1e-7, 1e-5, 1e-4, 1e-3)
    }
    assert len(set(verdicts.values())) == 1, verdicts
