"""P5.3: anisotropic / structured index sets (Basis)."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.basis import Basis
from MomentEmu.emulator import PolyEmu, generate_multi_indices


def _names(n):
    return [f"x{i}" for i in range(n)]


@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize("d", [0, 2, 4])
def test_total_degree_is_row_identical(n, d):
    got = Basis.total_degree().build(_names(n), d)
    ref = generate_multi_indices(n, d)
    assert np.array_equal(got, ref)


def test_constraints_compose_by_intersection():
    names = _names(4)
    b = Basis(max_interaction=2, per_parameter=(3, 1, 1, 1), degree=5)
    mi = b.build(names, 5)
    assert np.all(mi.sum(axis=1) <= 5)
    assert np.all(np.count_nonzero(mi, axis=1) <= 2)
    assert np.all(mi <= np.array([3, 1, 1, 1]))
    # A group limit restricts the sum inside the group.
    g = Basis(groups=(([0, 1], 2),), degree=5)
    mg = g.build(names, 5)
    assert np.all(mg[:, [0, 1]].sum(axis=1) <= 2)


def test_spec_is_copy_pasteable():
    b = Basis(q=0.5, weights=(1, 2, 8, 8, 4, 4), max_interaction=2, degree=16)
    spec = b.spec()
    assert spec.startswith("Basis(") and "q=0.5" in spec and "max_interaction=2" in spec


@pytest.mark.slow
def test_f1_q05_anisotropic_beats_isotropic():
    rng = np.random.default_rng(20260910)
    N = 20000
    X = rng.uniform(-1.0, 1.0, (N, 6))
    Xt = rng.uniform(-1.0, 1.0, (10000, 6))

    def f1(A):
        return (
            np.tanh(4.0 * A[:, 0]) + 0.5 * np.exp(A[:, 1]) + 0.1 * A[:, 2]
            + 0.1 * A[:, 3] + 0.05 * A[:, 4] * A[:, 5]
        )[:, None]

    Y, Yt = f1(X), f1(Xt)
    basis = Basis(q=0.5, weights=(1, 2, 8, 8, 4, 4), max_interaction=2, degree=16)
    emu = PolyEmu(
        X, Y, X_test=Xt, Y_test=Yt, basis=basis,
        init_deg_forward=16, max_degree_forward=16, RMSE_tol=1e-300, verbose=0,
    )
    D = emu.forward_multi_indices.shape[0]
    assert D == 67, D  # the q=0.5, max_interaction=2 set at d=16 has 67 terms
    pred = emu.forward_emulator(Xt, extrapolation="ignore")
    rel = float(np.sqrt(np.mean((pred - Yt) ** 2))) / float(np.abs(Yt).max())
    assert rel < 6e-4, rel


def test_report_returns_per_parameter_degree():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (500, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    info = emu.report()
    assert info["n_terms"] == emu.forward_multi_indices.shape[0]
    assert len(info["per_parameter_degree"]) == 3


# --- T-001: block-separable index sets ------------------------------------

def _block_count(blocks, d):
    """1 + sum_k [C(p_k + d, d) - 1]: every support inside one block."""
    from math import comb
    return 1 + sum(comb(len(b) + d, d) - 1 for b in blocks)


@pytest.mark.parametrize("blocks", [((0, 1, 2), (3, 4, 5), (6, 7, 8)),
                                    ((0,), (1, 2, 3, 4)),
                                    ((0, 1), (2, 3), (4, 5))])
@pytest.mark.parametrize("d", [0, 1, 3, 5])
def test_blocks_count_matches_the_closed_form(blocks, d):
    n = sum(len(b) for b in blocks)
    mi = Basis.separable(blocks, degree=d).build(_names(n))
    assert mi.shape[0] == _block_count(blocks, d)


@pytest.mark.parametrize("d", [2, 4, 6])
def test_blocks_restrict_every_support_to_one_block(d):
    blocks = ((0, 1, 2), (3, 4), (5, 6, 7))
    mi = Basis.separable(blocks, degree=d).build(_names(8))
    labels = np.empty(8, dtype=int)
    for k, b in enumerate(blocks):
        labels[list(b)] = k
    assert np.all(mi.sum(axis=1) <= d)
    for alpha in mi:
        support = np.flatnonzero(alpha)
        assert support.size == 0 or np.unique(labels[support]).size == 1
    # and it is a strict subset of the isotropic basis of the same degree
    assert mi.shape[0] < generate_multi_indices(8, d).shape[0]


def test_blocks_index_set_is_downward_closed():
    """Required by MonomialPlan and by the Legendre/monomial span identity."""
    mi = Basis.separable(((0, 1, 2), (3, 4, 5)), degree=4).build(_names(6))
    rows = {tuple(int(v) for v in r) for r in mi}
    for r in rows:
        for i, e in enumerate(r):
            if e:
                lower = list(r)
                lower[i] -= 1
                assert tuple(lower) in rows


def test_blocks_rows_are_sorted_and_unique():
    mi = Basis.separable(((0, 1), (2, 3)), degree=3).build(_names(4))
    rows = [tuple(int(v) for v in r) for r in mi]
    assert rows == sorted(rows, key=lambda a: (sum(a), a))
    assert len(set(rows)) == len(rows)


@pytest.mark.parametrize("blocks, n, match", [
    (((0, 1), (1, 2)), 3, "disjoint"),
    (((0, 1), (2,)), 4, "cover"),
    (((0, 1), (2, 9)), 3, "range"),
    ((), 3, "at least one"),
])
def test_blocks_must_partition_the_parameters(blocks, n, match):
    with pytest.raises(ValueError, match=match):
        Basis.separable(blocks, degree=2).build(_names(n))


def test_blocks_compose_with_the_other_constraints():
    blocks = ((0, 1, 2), (3, 4, 5))
    plain = Basis.separable(blocks, degree=4).build(_names(6))
    capped = Basis(degree=4, blocks=blocks, max_interaction=1).build(_names(6))
    assert capped.shape[0] < plain.shape[0]
    assert np.all(np.count_nonzero(capped, axis=1) <= 1)
    grouped = Basis(degree=4, blocks=blocks, per_parameter=(2, 2, 2, 2, 2, 2)).build(_names(6))
    assert np.all(grouped <= 2)


def test_blocks_equal_one_block_is_the_isotropic_basis():
    """Same index SET; Basis.build sorts rows by (degree, index), which
    generate_multi_indices does not, so compare as sets."""
    got = Basis.separable((tuple(range(4)),), degree=3).build(_names(4))
    ref = generate_multi_indices(4, 3)
    assert {tuple(r) for r in got} == {tuple(r) for r in ref}
    assert got.shape == ref.shape


def test_blocks_are_prefix_nested_across_degrees():
    """The degree sweep borders the previous moment matrix only when the new
    index set extends the old one as a prefix (emulator.py:1190)."""
    blocks = ((0, 1, 2), (3, 4), (5, 6, 7))
    prev = None
    for d in range(0, 6):
        mi = Basis.separable(blocks, degree=d).build(_names(8))
        if prev is not None:
            assert np.array_equal(mi[: prev.shape[0]], prev)
        prev = mi


def test_blocks_spec_and_equality():
    b = Basis.separable(((0, 1), (2, 3)), degree=4)
    assert "blocks=" in b.spec()
    assert b == Basis(degree=4, blocks=((0, 1), (2, 3)))
    assert b != Basis(degree=4, blocks=((0,), (1,), (2, 3)))


def test_blocks_scale_to_many_parameters():
    """The point of the reduction: n=20, d=6 is 230,230 terms isotropic."""
    blocks = tuple(tuple(range(5 * k, 5 * k + 5)) for k in range(4))
    mi = Basis.separable(blocks, degree=6).build(_names(20))
    assert mi.shape[0] == _block_count(blocks, 6) == 1845


@pytest.mark.slow
def test_block_basis_matches_the_full_basis_on_a_separable_function():
    """Equal-or-better accuracy at 12x fewer terms, and usable where full is not."""
    from MomentEmu.monomials import MonomialPlan
    rng = np.random.default_rng(4)
    blocks = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    def g(u, v, w):
        return np.sin(u + 0.5 * v * w) * np.exp(0.3 * w) + 0.4 * np.cos(1.2 * u * v)

    def f(X):
        return sum(g(X[:, a], X[:, b], X[:, c]) for a, b, c in blocks)
    Xtr, Xte = rng.uniform(-1, 1, (8000, 9)), rng.uniform(-1, 1, (4000, 9))
    ytr, yte = f(Xtr), f(Xte)

    def rmse(mi):
        plan = MonomialPlan.build(mi)
        A = plan.evaluate(Xtr)
        c = np.linalg.solve(A.T @ A, A.T @ ytr)
        return float(np.sqrt(np.mean((plan.evaluate(Xte) @ c - yte) ** 2)))

    full = generate_multi_indices(9, 5)
    blk = Basis.separable(blocks, degree=5).build(_names(9))
    assert blk.shape[0] == 166 and full.shape[0] == 2002
    assert rmse(blk) <= rmse(full)


def test_basis_pickled_before_blocks_existed_still_works():
    """A Basis stored in a PolyEmu pickle from 2.0.0 has no `blocks` key in
    __dict__; restoring it must not break build/spec/eq (cf. the B1 guard in
    PolyEmu._transforms)."""
    import pickle

    b = Basis(degree=3, max_interaction=2)
    del b.__dict__["blocks"]
    restored = pickle.loads(pickle.dumps(b))
    assert restored.blocks is None
    mi = restored.build(_names(3), 3)
    assert np.all(np.count_nonzero(mi, axis=1) <= 2)
    assert "blocks=" not in restored.spec()
    assert restored == Basis(degree=3, max_interaction=2)


def test_block_validation_is_eager_not_deferred():
    """The partition is rejected before the index search starts, not part-way
    through it, so a bad partition never produces a partial basis."""
    from MomentEmu.basis import validate_blocks

    with pytest.raises(ValueError, match="disjoint"):
        validate_blocks(((0, 1), (1, 2)), 3)


def test_empty_inner_block_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        Basis.separable(((0, 1), (), (2,))).build(_names(3), 2)


# --- T-001: per-parameter parity ------------------------------------------

@pytest.mark.parametrize("d", [2, 5, 7])
def test_parity_restricts_the_powers_of_each_parameter(d):
    b = Basis(degree=d, parity=("even", None, "odd", None))
    mi = b.build(_names(4), d)
    assert np.all(mi[:, 0] % 2 == 0)
    assert np.all(mi[:, 2] % 2 == 1)          # odd means it never appears at power 0
    assert np.all(mi.sum(axis=1) <= d)
    assert mi.shape[0] < generate_multi_indices(4, d).shape[0]


def test_parity_even_keeps_the_constant_term():
    mi = Basis(degree=4, parity=("even", "even")).build(_names(2), 4)
    assert (mi.sum(axis=1) == 0).sum() == 1


def test_parity_index_set_is_downward_closed_only_for_even():
    """MonomialPlan takes the downward closure, so a non-closed set still
    evaluates; this pins which of the two is closed."""
    even = {tuple(r) for r in Basis(degree=4, parity=("even", None)).build(_names(2), 4)}
    assert all(
        tuple(np.subtract(r, 2 * np.eye(2, dtype=int)[0])) in even
        for r in even if r[0] >= 2
    )
    odd = Basis(degree=4, parity=("odd", None)).build(_names(2), 4)
    assert not np.any(odd[:, 0] == 0)         # deliberately not downward closed


def test_parity_composes_with_caps_and_blocks():
    b = Basis(degree=7, per_parameter=(2, 2, 7, 7, 7), parity=(None, "even", None, None, "even"))
    mi = b.build(_names(5), 7)
    assert np.all(mi[:, 1] % 2 == 0) and np.all(mi[:, 4] % 2 == 0)
    assert np.all(mi[:, :2] <= 2)
    assert mi.shape[0] == 223          # 792 isotropic -> 546 capped -> 223 with parity
    nested = Basis(degree=4, blocks=((0, 1), (2, 3)), parity=("even", None, None, None))
    mj = nested.build(_names(4), 4)
    assert np.all(mj[:, 0] % 2 == 0)


@pytest.mark.parametrize("parity, match", [
    (("even",), "length"),
    (("even", "sideways", None, None), "even.*odd"),
])
def test_parity_is_validated(parity, match):
    with pytest.raises(ValueError, match=match):
        Basis(degree=3, parity=parity).build(_names(4), 3)


def test_parity_in_spec_and_equality_and_legacy_pickle():
    import pickle
    b = Basis(degree=4, parity=("even", None))
    assert "parity=" in b.spec()
    assert b == Basis(degree=4, parity=("even", None))
    assert b != Basis(degree=4, parity=(None, None))
    legacy = Basis(degree=4)
    del legacy.__dict__["parity"]
    assert pickle.loads(pickle.dumps(legacy)).parity is None


# --- T-002: index search must cost O(|basis|), not O((d+1)^n) --------------

@pytest.mark.parametrize("d, expected", [(6, 43), (12, 85), (20, 141), (40, 281)])
def test_max_interaction_one_is_reachable_at_any_degree(d, expected):
    """1 + n*d terms. Enumerating the (d+1)**n box took 57 s at d=12 and did
    not finish at d=20; the search is now O(|basis| * n)."""
    import time

    t0 = time.perf_counter()
    mi = Basis(degree=d, max_interaction=1).build(_names(7), d)
    elapsed = time.perf_counter() - t0
    assert mi.shape[0] == expected == 1 + 7 * d
    assert np.all(np.count_nonzero(mi, axis=1) <= 1)
    assert elapsed < 5.0, f"{elapsed:.1f}s for {expected} terms at degree {d}"


def test_high_degree_sparse_basis_is_reachable():
    import time

    t0 = time.perf_counter()
    mi = Basis(degree=25, max_interaction=3).build(_names(7), 25)
    assert time.perf_counter() - t0 < 30.0
    assert np.all(np.count_nonzero(mi, axis=1) <= 3)
    assert np.all(mi.sum(axis=1) <= 25)


def test_q_norm_is_reachable_at_high_degree():
    import time

    t0 = time.perf_counter()
    mi = Basis(q=0.5, weights=(1, 2, 8, 8, 4, 4, 2), degree=16).build(_names(7), 16)
    assert time.perf_counter() - t0 < 10.0
    assert mi.shape[0] >= 1


@pytest.mark.parametrize("kwargs", [
    {"max_interaction": 2},
    {"per_parameter": (2, 3, 1, 4)},
    {"q": 1.5, "weights": (1.0, 2.0, 1.0, 3.0)},
    {"groups": (((0, 1), 2),)},
    {"max_interaction": 2, "per_parameter": (3, 3, 2, 2)},
    {"blocks": ((0, 1), (2, 3))},
    {"parity": ("even", None, "odd", None)},
    {"blocks": ((0, 1), (2, 3)), "parity": ("even", None, None, None)},
])
@pytest.mark.parametrize("d", [0, 1, 3, 5])
def test_search_matches_brute_force_filtering(kwargs, d):
    """The breadth-first search must return exactly the set the old
    box-enumeration-and-filter produced, on every constraint combination."""
    from itertools import product

    n = 4
    b = Basis(degree=d, **kwargs)
    got = {tuple(int(v) for v in r) for r in b.build(_names(n), d)}

    per = kwargs.get("per_parameter")
    parity = kwargs.get("parity")
    blocks = kwargs.get("blocks")
    labels = {}
    if blocks:
        for k, blk in enumerate(blocks):
            for i in blk:
                labels[i] = k
    w = np.asarray(kwargs.get("weights", [1.0] * n), float)
    want = set()
    for a in product(*[range(d + 1)] * n):
        arr = np.asarray(a)
        if arr.sum() > d:
            continue
        if per is not None and np.any(arr > np.asarray(per)):
            continue
        if "max_interaction" in kwargs and np.count_nonzero(arr) > kwargs["max_interaction"]:
            continue
        if "q" in kwargs and float(np.sum((arr * w) ** kwargs["q"]) ** (1 / kwargs["q"])) > d + 1e-12:
            continue
        if any(int(arr[list(idx)].sum()) > cap for idx, cap in kwargs.get("groups", ())):
            continue
        if blocks and len({labels[i] for i, e in enumerate(a) if e}) > 1:
            continue
        if parity and any(
            (a[i] % 2 == 1) if v == "even" else (a[i] % 2 == 0)
            for i, v in enumerate(parity) if v is not None
        ):
            continue
        want.add(tuple(a))
    assert got == want


# --- Basis degree vs the PolyEmu degree sweep -----------------------------

def _sweep_data(n=9, N=1500, seed=0):
    """A function needing degree > 2, so the cap is what stops the sweep."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (X[:, 0] ** 2 + np.sin(X[:, 3])).reshape(-1, 1)
    return X, Y


def test_basis_degree_caps_the_forward_sweep():
    """Basis(degree=2) must fit degree 2, not the sample-count cap of 4."""
    X, Y = _sweep_data()
    emu = PolyEmu(X, Y, basis=Basis(degree=2), RMSE_tol=0.0, verbose=0)
    # The selected degree is what the user sees; the sweep list is what proves
    # no rung above the cap was ever tried.
    assert emu.forward_degree == 2
    assert max(emu.forward_degree_list) == 2
    assert emu.forward_multi_indices.shape[0] == generate_multi_indices(9, 2).shape[0]


def test_basis_degree_disagreeing_with_max_degree_forward_raises():
    """Both set the top of the sweep, so neither may silently win."""
    X, Y = _sweep_data()
    with pytest.raises(ValueError, match="both set the top of the forward degree sweep"):
        PolyEmu(X, Y, basis=Basis(degree=2), max_degree_forward=5, verbose=0)


def test_basis_degree_agreeing_with_max_degree_forward_is_accepted():
    X, Y = _sweep_data()
    emu = PolyEmu(X, Y, basis=Basis(degree=2), max_degree_forward=2,
                  RMSE_tol=0.0, verbose=0)
    assert emu.forward_degree == 2


def test_init_deg_forward_below_the_basis_degree_is_a_separate_knob():
    """init_deg says where the sweep starts, not where it stops."""
    X, Y = _sweep_data()
    emu = PolyEmu(X, Y, basis=Basis(degree=3), init_deg_forward=1,
                  RMSE_tol=0.0, verbose=0)
    assert emu.forward_degree_list == [1, 2, 3]
    assert emu.forward_degree == 3


def test_init_deg_forward_above_the_basis_degree_raises():
    X, Y = _sweep_data()
    with pytest.raises(ValueError, match="starts the forward degree sweep above"):
        PolyEmu(X, Y, basis=Basis(degree=2), init_deg_forward=4, verbose=0)


def test_basis_degree_zero_is_a_degree_not_a_missing_value():
    """degree=0 is falsy; the heuristic init_deg (1 at n_params > 6) must yield."""
    X, Y = _sweep_data()
    emu = PolyEmu(X, Y, basis=Basis(degree=0), RMSE_tol=0.0, verbose=0)
    assert emu.forward_degree == 0
    assert emu.forward_multi_indices.shape[0] == 1


def test_basis_without_a_degree_leaves_the_sweep_alone():
    """No degree on the Basis, so max_degree_forward still drives the sweep."""
    X, Y = _sweep_data()
    emu = PolyEmu(X, Y, basis=Basis(max_interaction=2), init_deg_forward=1,
                  max_degree_forward=4, RMSE_tol=0.0, verbose=0)
    assert emu.forward_degree_list == [1, 2, 3, 4]


# --- the index search, however it is implemented --------------------------

def _reference_build(n, d, **kw):
    """Product-and-filter enumeration: the semantics build() had before the
    index search replaced it, written out independently. Small n, d only."""
    from itertools import product

    per = kw.get("per_parameter")
    bounds = [d] * n if per is None else [min(d, int(c)) for c in per]
    q, weights = kw.get("q"), kw.get("weights")
    w = None
    if q is not None:
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        for i in range(n):
            bounds[i] = min(bounds[i], int(np.floor(d / w[i])))
    parity = kw.get("parity")
    cap = kw.get("max_interaction")
    groups = kw.get("groups", ())
    rows = []
    for alpha in product(*[range(b + 1) for b in bounds]):
        if sum(alpha) > d:
            continue
        if parity is not None and any(
            (alpha[i] % 2 == 1) if v == "even" else (alpha[i] % 2 == 0)
            for i, v in enumerate(parity) if v is not None
        ):
            continue
        if cap is not None and sum(1 for a in alpha if a) > cap:
            continue
        if q is not None:
            arr = np.asarray(alpha, dtype=np.int64)
            if float(np.sum((arr * w) ** q) ** (1.0 / q)) > d + 1e-12:
                continue
        if any(sum(alpha[i] for i in idx) > limit for idx, limit in groups):
            continue
        rows.append(alpha)
    if not rows:
        return np.zeros((0, n), dtype=np.int64)
    rows.sort(key=lambda a: (sum(a), a))
    return np.array(rows, dtype=np.int64)


@pytest.mark.parametrize("kw", [
    dict(max_interaction=1),
    dict(max_interaction=2),
    dict(q=0.5),
    dict(q=2.0),
    dict(q=0.5, weights=(1.0, 2.0, 4.0, 1.0, 2.0)),
    dict(per_parameter=(0, 1, 2, 3, 4)),
    dict(parity=("even",) * 5),
    dict(parity=("odd",) * 5),
    dict(parity=("even", "odd", None, "even", None)),
    dict(groups=(((0, 1, 2), 2),)),
    dict(groups=(((0, 1), 1), ((1, 2, 3), 2))),
    dict(max_interaction=2, parity=("even",) * 5, groups=(((0, 1, 2), 2),)),
    dict(max_interaction=3, per_parameter=(2,) * 5, q=0.8),
])
@pytest.mark.parametrize("d", [0, 1, 2, 3, 4, 5])
def test_index_search_is_row_identical_to_the_product_enumeration(kw, d):
    """However build() searches, it is a performance change only: the same
    rows in the same order as enumerating the box and filtering it."""
    n = 5
    got = Basis(degree=d, **kw).build(_names(n), d)
    assert np.array_equal(got, _reference_build(n, d, **kw))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("kw,expected", [
    (dict(max_interaction=2), 2971),
    (dict(parity=("even",) * 20), 1771),
    (dict(q=0.5), 691),
])
def test_constrained_basis_is_feasible_at_twenty_parameters(kw, expected):
    """p=20, d=6 is the size the Basis guide quotes. Enumerating the box there
    is 7**20 candidates, so a build() that does will not finish: the timeout is
    the assertion. The counts pin the answer as well as the cost."""
    mi = Basis(degree=6, **kw).build(_names(20))
    assert mi.shape[0] == expected
    assert np.all(mi.sum(axis=1) <= 6)


def test_impossible_parity_budget_is_empty():
    """Five 'odd' positions owe at least 5 of the degree budget, so d=4 admits
    nothing and d=5 admits exactly the all-ones index."""
    assert Basis(degree=4, parity=("odd",) * 5).build(_names(5)).shape == (0, 5)
    mi = Basis(degree=5, parity=("odd",) * 5).build(_names(5))
    assert mi.shape == (1, 5) and np.array_equal(mi[0], np.ones(5, dtype=np.int64))
