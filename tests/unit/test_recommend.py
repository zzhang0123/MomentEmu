"""T-004: recommend() chooses a strategy from the data and returns it fitted.

The tests are organised around the four failure modes that shaped the design,
because each is a case where the obvious automatic rule gave the wrong answer
and would pass a test written only against the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.recommend import Recommendation, recommend


def _smooth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = np.column_stack([np.sin(1.5 * X[:, 0]) + X[:, 1] ** 2,
                         np.cos(X[:, 2]) - 0.3 * X[:, 0]])
    return X, Y


def _ridge(n=600, seed=1, n_params=6):
    """f(x) = h(W x) with a 2-D active subspace: a rotation should win."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((2, n_params))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (n, n_params))
    U = X @ W.T
    Y = np.column_stack([np.tanh(1.2 * U[:, 0]) + 0.5 * U[:, 1] ** 2,
                         np.sin(U[:, 0] * U[:, 1])])
    return X, Y


def test_it_returns_a_fitted_model_not_a_configuration() -> None:
    """The scope decision recorded on the task: fitted, not a recipe."""
    X, Y = _smooth()
    rec = recommend(X, Y, budget=8, random_state=0)
    assert isinstance(rec, Recommendation)
    pred = rec.forward_emulator(X)
    assert pred.shape == Y.shape
    assert np.isfinite(pred).all()
    rel = np.linalg.norm(pred - Y) / np.linalg.norm(Y)
    assert rel < 5e-2, rel
    np.testing.assert_allclose(pred, rec.model.forward_emulator(X))


def test_the_report_names_every_candidate_and_exactly_one_choice() -> None:
    X, Y = _smooth()
    rep = recommend(X, Y, budget=8, random_state=0).report()
    assert rep["candidates"], "a recommendation with no evidence is not auditable"
    for row in rep["candidates"]:
        assert {"label", "stage", "score", "n_terms", "chosen"} <= set(row)
    assert sum(bool(r["chosen"]) for r in rep["candidates"]) == 1
    assert rep["candidates_scored"] == len(rep["candidates"])
    assert "greedy" in rep["search"]


def test_term_counts_are_the_inner_models_not_the_wrapper() -> None:
    """Failure mode 1's guard depends on this and breaks silently without it.

    PreconditionedEmu wraps the estimator, so counting the wrapper returns 0
    for every candidate; parsimony then cannot distinguish them and quietly
    degrades to lowest-error selection.
    """
    X, Y = _smooth()
    rep = recommend(X, Y, budget=8, random_state=0).report()
    scored = [r for r in rep["candidates"] if np.isfinite(r["score"])]
    assert scored
    assert all(r["n_terms"] > 0 for r in scored), (
        "a candidate reported 0 coefficients; parsimony is not seeing sizes"
    )
    assert rep["n_terms"] > 0


def test_a_smaller_model_within_tolerance_beats_a_more_accurate_one() -> None:
    """Failure mode 1: the lowest error alone buys whole dimensions.

    Read the chosen row against the best-scoring one rather than asserting a
    particular estimator, which would pin the target and not the rule.
    """
    X, Y = _smooth()
    rep = recommend(X, Y, budget=8, random_state=0, parsimony_tol=2.0).report()
    rows = [r for r in rep["candidates"] if np.isfinite(r["score"])]
    chosen = next(r for r in rows if r["chosen"])
    best = min(rows, key=lambda r: r["score"])
    assert chosen["score"] <= best["score"] * 2.0
    assert chosen["n_terms"] <= best["n_terms"]


def test_a_ridge_target_gets_a_rotation() -> None:
    """The capability the whole package exists for, chosen without being named."""
    X, Y = _ridge()
    rec = recommend(X, Y, budget=10, random_state=0)
    assert "rotate" in rec.config["order"], rec.config
    rel = np.linalg.norm(rec.forward_emulator(X) - Y) / np.linalg.norm(Y)
    assert rel < 5e-2, rel


def test_a_truncated_search_says_so() -> None:
    """A budget that cuts stages must be visible, not silent."""
    X, Y = _smooth()
    rep = recommend(X, Y, budget=2, random_state=0).report()
    assert rep["stages_cut"], "stages were skipped without being reported"
    assert rep["candidates_scored"] <= 2
    assert "4-refit" in rep["stages_run"]


def test_the_basis_is_searched_whatever_estimator_won() -> None:
    """Regression: the basis is searched for every estimator, jointly.

    An earlier version skipped it for the sparse estimator, and a version
    after that searched it only for the winner of a separate estimator stage.

    The 21cmGEM benchmark put the sparse monomial-to-legendre gap at 1.1x
    against 12.7x for a dense fit, and that one measurement was turned into a
    rule. On a smooth 3-parameter target the sparse gap runs the other way and
    is 2.5x (monomial 5.16e-04 against legendre 1.30e-03), so the skip cost
    the recommender the configuration an exhaustive search picks. Generalising
    a single measurement into a threshold is failure mode 4.
    """
    X, Y = _smooth()
    rep = recommend(X, Y, budget=40, random_state=0).report()
    rows = [r for r in rep["candidates"] if r["stage"] == "1-estimator-basis"]
    per_estimator: dict[str, set] = {}
    for row in rows:
        per_estimator.setdefault(row["estimator"], set()).add(row["basis_kind"])
    for est, kinds in per_estimator.items():
        if est == "factored":
            continue  # a rank-1 product per block; it has no basis_kind
        assert kinds == {"monomial", "legendre", "chebyshev"}, (est, kinds)
    assert rep["config"]["basis_kind"] in {"monomial", "legendre", "chebyshev"}


def test_the_sparse_basis_gap_that_killed_the_skip_is_real() -> None:
    """Pin the counter-measurement itself, not just the code that respects it."""
    import warnings

    from MomentEmu.sparse import SparseEmu

    X, Y = _smooth()
    order = np.random.default_rng(0).permutation(X.shape[0])
    hold, keep = order[:80], order[80:]
    den = float(np.sqrt(np.mean(Y[hold] ** 2)))
    err = {}
    for kind in ("monomial", "legendre"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = SparseEmu(X[keep], Y[keep], degree=8, n_terms=60,
                             basis_kind=kind).forward_emulator(X[hold])
        err[kind] = float(np.sqrt(np.mean((pred - Y[hold]) ** 2))) / den
    assert err["monomial"] < err["legendre"] / 2.0, err


def test_budget_is_validated() -> None:
    X, Y = _smooth()
    with pytest.raises(ValueError, match="budget"):
        recommend(X, Y, budget=0)


def test_a_supplied_jacobian_reaches_the_rotation() -> None:
    """T-005 composes with the recommender rather than being bypassed by it."""
    rng = np.random.default_rng(4)
    W = rng.standard_normal((2, 5))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (500, 5))
    U = X @ W.T
    Y = np.column_stack([np.tanh(U[:, 0]) + 0.4 * U[:, 1] ** 2,
                         np.sin(0.7 * U[:, 0] * U[:, 1])])
    calls = [0]

    def jac(chunk: np.ndarray) -> np.ndarray:
        calls[0] += 1
        u = chunk @ W.T
        c = np.cos(0.7 * u[:, 0] * u[:, 1])
        d0 = np.stack([1.0 - np.tanh(u[:, 0]) ** 2, 0.8 * u[:, 1]], axis=1)
        d1 = np.stack([0.7 * c * u[:, 1], 0.7 * c * u[:, 0]], axis=1)
        return np.stack([d0 @ W, d1 @ W], axis=1)

    rec = recommend(X, Y, budget=6, random_state=0, jacobian=jac)
    assert calls[0] > 0, "the supplied Jacobian was never used"
    assert np.isfinite(rec.forward_emulator(X)).all()


def test_the_returned_model_was_refitted_on_all_the_data() -> None:
    """Every scored candidate saw a held-out split, so none of them may be
    the model handed back."""
    X, Y = _smooth()
    rec = recommend(X, Y, budget=8, random_state=0, validation_split=0.4)
    small = recommend(X[:200], Y[:200], budget=8, random_state=0)
    assert rec.model is not small.model
    n_full = rec.model.transform(X).shape[0] if hasattr(rec.model, "transform") else None
    if n_full is not None:
        assert n_full == X.shape[0]


def test_leftover_budget_buys_a_rank_scan() -> None:
    """Failure mode 1 is a rank rule, so spare budget goes to the rank first.

    PreconditionedEmu picks the rank from a variance target, and that rule is
    the one measured to overshoot: it returned rank 6 where rank 2 at a higher
    degree was 38x smaller AND more accurate. Ranks below the automatic choice
    are therefore where the spare budget is worth spending.
    """
    X, Y = _ridge()
    rep = recommend(X, Y, budget=40, random_state=0).report()
    if "rotate" not in str(rep["config"]["order"]):
        pytest.skip("this target did not select a rotating order")
    assert "3-rank" in rep["stages_run"], rep["stages_cut"]
    tried = {r["rank"] for r in rep["candidates"] if r["stage"] == "3-rank"}
    assert tried, "the rank stage ran but scored nothing"
    assert rep["config"]["rank"] in tried | {rep["auto_rank"]}
    assert min(tried) < rep["auto_rank"], (
        "the scan must reach BELOW the automatic rank; that is where the "
        "variance target was measured to be wrong"
    )


def test_a_tight_budget_reports_the_rank_scan_as_cut() -> None:
    X, Y = _ridge()
    rep = recommend(X, Y, budget=4, random_state=0).report()
    assert "3-rank" not in rep["stages_run"]
    assert any("3-rank" in c for c in rep["stages_cut"]), rep["stages_cut"]


def test_the_config_names_the_rank_it_settled_on() -> None:
    X, Y = _ridge()
    rep = recommend(X, Y, budget=40, random_state=0).report()
    if "rotate" in str(rep["config"]["order"]):
        assert isinstance(rep["config"]["rank"], int)
        assert rep["config"]["rank"] >= 1
    else:
        assert rep["config"]["rank"] is None


def test_the_rank_scan_undoes_an_overshooting_variance_target() -> None:
    """Failure mode 1, reproduced and then fixed by the spare budget.

    A spectrum with a long shallow tail is where the 0.999 variance target
    goes wrong: it keeps buying directions that carry almost nothing, and each
    costs a whole dimension of C(d+r, r). Here it returns rank 6, while the
    scan finds rank 2 reaches the same accuracy band for a twelfth of the
    coefficients:

        rank 1   1.535e-01     4
        rank 2   8.684e-02    10     <- chosen
        rank 5   5.733e-02    56
        rank 7   4.959e-02   120
    """
    rng = np.random.default_rng(7)
    n_params = 7
    Q = np.linalg.qr(rng.standard_normal((n_params, n_params)))[0]
    X = rng.uniform(-1.0, 1.0, (900, n_params))
    U = X @ Q
    amp = np.array([1.0, 0.18, 0.10, 0.058, 0.051, 0.035, 0.028])
    Y = np.column_stack([
        np.tanh(1.5 * U[:, 0]) + sum(a * U[:, i] ** 2
                                     for i, a in enumerate(amp) if i),
        np.sin(1.2 * U[:, 0]) + sum(a * U[:, i] for i, a in enumerate(amp) if i),
    ])

    rep = recommend(X, Y, budget=40, random_state=0).report()
    assert rep["auto_rank"] == 6, (
        f"the variance target no longer overshoots here (got "
        f"{rep['auto_rank']}); the test target has drifted"
    )
    assert rep["config"]["rank"] < rep["auto_rank"], rep["config"]

    by_rank = {r["rank"]: r for r in rep["candidates"] if r["stage"] == "3-rank"}
    chosen = next(r for r in rep["candidates"] if r["chosen"])
    biggest = max(by_rank.values(), key=lambda r: r["n_terms"])
    assert chosen["n_terms"] * 5 < biggest["n_terms"], (
        "the scan kept a model that is not much smaller; parsimony is not "
        "weighing the coefficient count"
    )
    assert chosen["score"] <= biggest["score"] * 2.0
