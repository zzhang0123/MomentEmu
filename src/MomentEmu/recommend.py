"""One entry point that reads the data and returns a configured, fitted model.

Every mechanism in this package already exists, but the user has to name it:
``estimator=``, ``order=``, ``rank=``, ``basis_kind=``, ``scaling=``,
``blocks=``. That asks them to know their target is a ridge, or additively
separable, or sparse in an orthogonal basis, before they can benefit from it.

The procedure
-------------

Four failure modes measured in T-001 to T-003 shape it, each a case where the
obvious automatic rule gave the wrong answer:

1. A variance target overshoots the rank. The 0.999 rule returned rank 6 where
   rank 2 at a higher degree was 38x smaller AND more accurate, because a
   direction costs a whole dimension of ``C(d+r, r)`` however little variance
   it carries. Every stage here therefore selects on PARSIMONY: the fewest
   coefficients among the candidates within ``parsimony_tol`` of the best
   error, not the lowest error.
2. Marginal criteria miss interaction-dominated axes.
3. Affine transforms applied for numerical tidiness destroy the structure
   being searched for, found three separate times.
4. Thresholds read off a single quantity overlap badly: cond(M) values where
   the normal equations were fine and where they lost five orders of accuracy
   overlapped by six orders of magnitude.

Failure mode 4 is why nothing here is decided from a statistic. Each candidate
is FITTED and scored on the same held-out split, and the choice is made on
those numbers. One fit each way is affordable when it replaces a wrong default.

Stages, in increasing cost:

* Stage 0 fits one cheap pilot and reads its interaction graph, which says
  whether an additively separable model is even applicable.
* Stage 1 chooses the estimator and the preconditioner order together, by
  running ``PreconditionedEmu(order="auto")`` once per applicable estimator.
  The two cannot be chosen separately: an order is scored with the estimator
  that will actually use it.
* Stage 2 chooses the input scaling, and only for a polynomial estimator,
  which is the only one that has the parameter. Neither scaling wins
  everywhere and the crossover moves with the degree, so it is measured.
* Stage 3 chooses the basis family.

Stages 2 and 3 tune the winner of stage 1 rather than re-running the whole
search, which is a greedy descent and therefore has failure mode 2's shape.
The report says so, and names every candidate with its score, so a user who
disagrees can see what was rejected and on what evidence. Every automatic rule
tried in this work was wrong somewhere; the output is meant to be auditable
rather than authoritative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from MomentEmu.emulator import BASIS_PLANS, PolyEmu
from MomentEmu.guards import as_float64, check_finite
from MomentEmu.precondition import (
    PreconditionedEmu,
    _affordable_degree,
    _term_count,
)

SCALINGS = ("standard", "box")


class Recommendation:
    """A fitted model, the configuration chosen for it, and the evidence.

    Delegates prediction to the fitted model, so it is usable in place of one.
    """

    def __init__(self, model: Any, config: dict, candidates: list[dict],
                 budget: int, spent: int, stages_run: list[str],
                 stages_cut: list[str], auto_rank: int | None = None) -> None:
        self.auto_rank = auto_rank
        self.model = model
        self.config = dict(config)
        self.candidates = tuple(candidates)
        self.budget = int(budget)
        self.spent = int(spent)
        self.stages_run = tuple(stages_run)
        self.stages_cut = tuple(stages_cut)

    @property
    def n_terms(self) -> int:
        """Coefficients in the fitted model, counted on the inner estimator."""
        return _term_count(getattr(self.model, "emulator", self.model))

    def forward_emulator(self, X: Any, **kwargs: Any) -> np.ndarray:
        return self.model.forward_emulator(X, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this object does not define itself.
        return getattr(self.__dict__["model"], name)

    def report(self) -> dict:
        """The choice, the runners-up, and what was never tried."""
        return {
            "config": dict(self.config),
            "auto_rank": self.auto_rank,
            "n_terms": self.n_terms,
            "candidates": tuple(dict(c) for c in self.candidates),
            "budget": self.budget,
            # Outer candidates. Each stage-1 entry ran its own scan over the
            # preconditioner orders inside PreconditionedEmu, so the fits
            # actually performed are several times this.
            "candidates_scored": self.spent,
            "stages_run": self.stages_run,
            "stages_cut": self.stages_cut,
            "search": (
                "estimator, basis and preconditioner order are searched "
                "jointly; the scaling then tunes that winner, which is greedy "
                "in that one axis"
            ),
        }

    def __repr__(self) -> str:
        c = self.config
        return (
            f"<Recommendation estimator={c.get('estimator')!r} "
            f"order={c.get('order')!r} scaling={c.get('scaling')!r} "
            f"basis_kind={c.get('basis_kind')!r} "
            f"n_terms={self.n_terms}>"
        )


def _split(n_rows: int, validation_split: float, random_state: Any):
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n_rows)
    cut = max(1, int(round(float(validation_split) * n_rows)))
    return order[cut:], order[:cut]


def _relative_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    den = float(np.sqrt(np.mean(truth**2))) or 1.0
    err = float(np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)))
    return err / den


def _blocks_from_pilot(Xf, Yf, degree: int) -> tuple[tuple[int, ...], ...]:
    """Connected components of the pairwise interaction matrix, or one block."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot = PolyEmu(
                Xf, Yf, init_deg_forward=degree, max_degree_forward=degree,
                RMSE_tol=0.0, verbose=0,
            )
            blocks = pilot.interaction_graph(warn_uniform=False)["blocks"]
    except Exception:                                      # noqa: BLE001
        return (tuple(range(Xf.shape[1])),)
    return tuple(tuple(int(i) for i in b) for b in blocks)


def _preconditioned_dim(model: Any, X: np.ndarray) -> int:
    """Dimensions the estimator actually sees, after any preconditioning."""
    try:
        return int(model.transform(X[:1]).shape[1])
    except Exception:                                      # noqa: BLE001
        return int(X.shape[1])


def _sparse_size(n_train: int) -> int:
    """Active-set size the training set supports, floored at the library default.

    SparseEmu defaults to 200 terms, which is a small model on a large design:
    on 21cmGEM the recommender stopped there while the same estimator at 1,500
    terms reached 1.4478 % against its 4.21 %. The size is the axis that
    dominated accuracy on that target, and it was the one not being varied.
    """
    return int(max(200, min(1500, n_train // 16)))


def _enable_the_degree_climb(extra: dict, scan_degree: int) -> None:
    """Let PolyEmu's own degree sweep run instead of stopping at its default.

    PolyEmu stops climbing at ``RMSE_tol=1e-2``, which on 21cmGEM left the
    polynomial candidates at degree 5 and 756 terms against the 3,102-term
    degree-7 baseline they were meant to beat. Its leave-one-out selection
    already picks the simplest rung within tolerance of the best, so removing
    the early stop costs time rather than accuracy. PreconditionedEmu clamps
    the ceiling to what the PRECONDITIONED dimension can afford, so a degree
    that is out of reach in seven parameters is still reachable in two.
    """
    extra.setdefault("max_degree_forward", int(scan_degree))
    extra.setdefault("RMSE_tol", 0.0)


def _fitted_degree(model: Any) -> int | None:
    """The forward degree a fitted polynomial candidate settled on."""
    inner = getattr(model, "emulator", model)
    d = getattr(inner, "forward_degree", None)
    return None if d is None else int(d)


def _chosen_rank(model: Any) -> int | None:
    """The rank a fitted PreconditionedEmu actually rotated onto, if it did."""
    for step in getattr(model, "steps", ()):
        rank = getattr(step, "rank", None)
        if rank is not None:
            return int(rank)
    return None


def _parsimonious(rows: list[dict], tol: float) -> dict:
    """Fewest coefficients among those within ``tol`` of the best error.

    Failure mode 1: taking the lowest error alone buys a direction, or a
    degree, whose cost is a whole dimension of the basis growth.
    """
    live = [r for r in rows if np.isfinite(r["score"])]
    if not live:
        return rows[0]
    best = min(r["score"] for r in live)
    near = [r for r in live if r["score"] <= best * float(tol)]
    return min(near, key=lambda r: (r["n_terms"], r["score"]))


def _fit_and_score(build, Xh, Yh, label: str, stage: str, config: dict) -> dict:
    """Fit one candidate and score it on the shared held-out split."""
    import warnings

    row = {"label": label, "stage": stage, "note": "", **config}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build()
            pred = model.forward_emulator(Xh)
    except Exception as exc:                               # noqa: BLE001
        row.update(score=float("inf"), n_terms=0, model=None,
                   note=f"{type(exc).__name__}: {exc}")
        return row
    # PreconditionedEmu wraps the estimator; the coefficients are the inner
    # model's, and counting the wrapper returns 0 for every candidate, which
    # silently turns parsimony selection into lowest-error selection.
    inner = getattr(model, "emulator", model)
    row.update(score=_relative_rmse(pred, Yh), n_terms=_term_count(inner),
               model=model)
    return row


def recommend(
    X: Any,
    Y: Any,
    *,
    budget: int = 12,
    X_test: Any = None,
    Y_test: Any = None,
    validation_split: float = 0.2,
    random_state: Any = None,
    parsimony_tol: float = 2.0,
    scan_degree: int = 12,
    pilot_degree: int = 3,
    jacobian: Any = None,
    gradient_covariance: Any = None,
    **kwargs: Any,
) -> Recommendation:
    """Choose a modelling strategy from the data and return it fitted.

    Args:
        X_test, Y_test: the data the choice is scored on. Without them a
            random ``validation_split`` of X is held out, which scores every
            candidate against the TRAINING design -- including its sparse
            corners, where a high-degree fit is least constrained. On 21cmGEM
            that made the two protocols disagree by 1.26x at degree 5 and
            4.03x at degree 7, and select different models. Pass the design
            the emulator will actually be queried on whenever there is one.
        budget: how many candidates may be fitted and scored. The stages run
            in increasing cost and stop when it is spent, and the report names
            the stages that were cut, so a truncated search is visible rather
            than silent. Below about 6 only the estimator is chosen.
        parsimony_tol: the error multiple a stage will pay for a smaller
            model, matching PreconditionedEmu's own default so the inner and
            outer searches value size the same way. Failure mode 1 is what it
            is for: the lowest error alone bought a direction whose cost was a
            whole dimension of the basis growth.
        jacobian, gradient_covariance: the caller's own derivatives for the
            rotation, in place of the pilot polynomial.

    Remaining keywords reach the estimators.

    Returns:
        A :class:`Recommendation`: the fitted model, the configuration, and
        every candidate that was scored.
    """
    X = as_float64(np.asarray(X), "X")
    Y = as_float64(np.asarray(Y), "Y")
    check_finite(X, "X")
    check_finite(Y, "Y")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if int(budget) < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if (X_test is None) != (Y_test is None):
        raise ValueError("X_test and Y_test must be given together")
    if X_test is None:
        keep, hold = _split(X.shape[0], validation_split, random_state)
        Xf, Yf, Xh, Yh = X[keep], Y[keep], X[hold], Y[hold]
    else:
        # Every candidate trains on ALL of X and is scored on the caller's
        # own data, so the selection is made on the criterion that defines
        # success rather than on a proxy for it.
        Xh = as_float64(np.asarray(X_test), "X_test")
        Yh = as_float64(np.asarray(Y_test), "Y_test")
        check_finite(Xh, "X_test")
        check_finite(Yh, "Y_test")
        if Yh.ndim == 1:
            Yh = Yh.reshape(-1, 1)
        if Xh.shape[1] != X.shape[1]:
            raise ValueError(
                f"X_test has {Xh.shape[1]} columns, expected {X.shape[1]}"
            )
        if Xh.shape[0] != Yh.shape[0]:
            raise ValueError(
                f"X_test has {Xh.shape[0]} rows and Y_test {Yh.shape[0]}"
            )
        keep = np.arange(X.shape[0])
        Xf, Yf = X, Y

    rows: list[dict] = []
    spent = 0
    stages_run: list[str] = []
    stages_cut: list[str] = []
    common = dict(
        scan_degree=scan_degree, pilot_degree=pilot_degree,
        validation_split=validation_split, random_state=random_state,
        select="parsimony", parsimony_tol=parsimony_tol,
        jacobian=jacobian, gradient_covariance=gradient_covariance,
    )

    # Stage 0: is an additively separable model even applicable? One pilot.
    blocks = _blocks_from_pilot(Xf, Yf, pilot_degree)
    stages_run.append("0-structure")

    # Stage 1: estimator, basis and preconditioner order, chosen together.
    #
    # These cannot be staged. Comparing estimators at a default basis and then
    # tuning the basis of the winner is a marginal criterion, and failure mode
    # 2 is that marginal criteria mislead when the axes interact. Measured: on
    # a smooth 3-parameter target the stage-1 comparison at the default basis
    # put polynomial (3.19e-03) ahead of sparse (1.89e-03) on parsimony, while
    # sparse in the MONOMIAL basis reached 3.65e-05 -- 87x better than the
    # configuration that staging returned, and unreachable from it.
    #
    # The joint product costs no more than the staged version did, because the
    # basis sweep that ran on one winner now runs on each estimator instead.
    estimators = ["polynomial", "sparse"]
    if len(blocks) > 1:
        estimators.append("factored")
    kinds = tuple(BASIS_PLANS)
    plan: list[tuple[str, str | None]] = []
    for est in estimators:
        # FactoredEmu has no basis_kind: it fits a rank-1 product per block.
        plan.extend((est, k) for k in kinds) if est != "factored" else \
            plan.append((est, None))

    for est, kind in plan:
        if spent >= int(budget):
            stages_cut.append(f"1-estimator-basis:{est}"
                              + (f"+{kind}" if kind else ""))
            continue
        extra = dict(kwargs)
        if kind is not None:
            extra["basis_kind"] = kind
        if est == "factored":
            extra["blocks"] = blocks
        elif est == "sparse":
            # SparseEmu builds its own candidate set and needs a degree for
            # it. Size it in the ORIGINAL dimension: a rotation only lowers
            # the dimension, so this stays affordable after preconditioning.
            # The degree is deliberately NOT set here. Sizing the candidate
            # set needs the dimension the chosen order leaves, which only
            # PreconditionedEmu knows; sizing it in the original dimension
            # wastes the headroom the rotation bought.
            extra.setdefault("n_terms", _sparse_size(keep.size))
        else:
            _enable_the_degree_climb(extra, scan_degree)
        label = f"{est}+{kind}" if kind else est
        rows.append(_fit_and_score(
            lambda e=est, x=extra: PreconditionedEmu(
                Xf, Yf, order="auto", estimator=e, **common, **x
            ),
            Xh, Yh, label, "1-estimator-basis",
            {"estimator": est, "order": "auto", "scaling": None,
             "basis_kind": kind},
        ))
        last = rows[-1]
        if last["model"] is not None:
            last["order"] = getattr(last["model"], "order", "auto")
        spent += 1
    stages_run.append("1-estimator-basis")

    best = _parsimonious(
        [r for r in rows if r["stage"] == "1-estimator-basis"], parsimony_tol
    )
    if best["model"] is None:
        raise RuntimeError(
            "no candidate could be fitted; the scored attempts and their "
            f"errors are {[(r['label'], r['note']) for r in rows]}"
        )

    # Stage 2: input scaling, only where the parameter exists.
    if best["estimator"] == "polynomial" and spent + len(SCALINGS) <= int(budget):
        for sc in SCALINGS:
            extra = dict(kwargs)
            if best["basis_kind"] is not None:
                extra["basis_kind"] = best["basis_kind"]
            _enable_the_degree_climb(extra, scan_degree)
            rows.append(_fit_and_score(
                lambda s=sc, x=extra: PreconditionedEmu(
                    Xf, Yf, order=best["order"], estimator="polynomial",
                    scaling=s, **common, **x
                ),
                Xh, Yh, f"polynomial+{sc}", "2-scaling",
                {"estimator": "polynomial", "order": best["order"],
                 "scaling": sc, "basis_kind": best["basis_kind"]},
            ))
            spent += 1
        stages_run.append("2-scaling")
        best = _parsimonious(
            [r for r in rows if r["stage"] == "2-scaling"] + [best],
            parsimony_tol,
        )
    else:
        stages_cut.append("2-scaling")

    # Stage 3: the rank, where the spare budget is worth most.
    #
    # PreconditionedEmu takes the rank from a variance target, and that is the
    # rule failure mode 1 records as overshooting: it returned rank 6 where
    # rank 2 at a higher degree was 38x smaller AND more accurate, because a
    # direction costs a whole dimension of C(d+r, r) however little variance
    # it carries. The scan therefore reaches BELOW the automatic rank, and is
    # ordered upwards so a tight budget spends itself on the small ones.
    auto_rank = _chosen_rank(best["model"])
    best["rank"] = auto_rank
    if auto_rank is None:
        stages_cut.append("3-rank (the chosen order does not rotate)")
    else:
        wanted = [r for r in range(1, min(auto_rank + 1, X.shape[1]) + 1)
                  if r != auto_rank]
        room = int(budget) - spent
        if room < 1 or not wanted:
            stages_cut.append("3-rank (no budget left)")
        else:
            extra = dict(kwargs)
            if best["basis_kind"] is not None:
                extra["basis_kind"] = best["basis_kind"]
            if best["scaling"] is not None:
                extra["scaling"] = best["scaling"]
            if best["estimator"] == "factored":
                extra["blocks"] = blocks
            elif best["estimator"] == "sparse":
                extra.setdefault("n_terms", _sparse_size(keep.size))
            elif best["estimator"] == "polynomial":
                _enable_the_degree_climb(extra, scan_degree)
            rank_common = {k: v for k, v in common.items() if k != "rank"}
            for r in wanted[:room]:
                rows.append(_fit_and_score(
                    lambda rr=r, x=extra, rc=rank_common: PreconditionedEmu(
                        Xf, Yf, order=best["order"], estimator=best["estimator"],
                        rank=rr, **rc, **x
                    ),
                    Xh, Yh, f"{best['estimator']}+rank{r}", "3-rank",
                    {"estimator": best["estimator"], "order": best["order"],
                     "scaling": best["scaling"],
                     "basis_kind": best["basis_kind"], "rank": r},
                ))
                spent += 1
            if len(wanted) > room:
                stages_cut.append(
                    f"3-rank (budget covered {room} of {len(wanted)} ranks)"
                )
            stages_run.append("3-rank")
            best = _parsimonious(
                [r for r in rows if r["stage"] == "3-rank"] + [best],
                parsimony_tol,
            )

    # Stage 3b: the degree, for a polynomial winner.
    #
    # Every other axis here is measured on the held-out split, but the degree
    # was left to PolyEmu's internal leave-one-out sweep -- a proxy, and one
    # that fails exactly where it matters. On 21cmGEM the sweep reached degree
    # 7 and then chose degree 5, because at cond(M) = 1.09e+18 and D/N = 0.126
    # the PRESS leverage is unreliable and a few points with h near 1 dominate
    # it. LOO ranked degree 5 best; the held-out error at degree 7 was 1.66x
    # BETTER (6.65e-02 against 1.10e-01, and 2.71 % against 4.69 % on the
    # benchmark's own figure of merit).
    #
    # So the degree is measured too. Each rung is fitted on its own, which
    # also avoids the sweep's early stop at cond(M) >= 1e16.
    settled = _fitted_degree(best["model"]) if best["estimator"] == "polynomial" \
        else None
    best["degree"] = settled
    if settled is None:
        stages_cut.append("3b-degree (only the polynomial estimator has one)")
    else:
        ceiling = _affordable_degree(
            _preconditioned_dim(best["model"], X), keep.size, scan_degree
        )
        wanted = [d for d in range(settled + 1, ceiling + 1)]
        room = int(budget) - spent
        if room < 1 or not wanted:
            stages_cut.append(
                f"3b-degree (settled at {settled}, ceiling {ceiling})"
            )
        else:
            extra = dict(kwargs)
            if best["basis_kind"] is not None:
                extra["basis_kind"] = best["basis_kind"]
            if best["scaling"] is not None:
                extra["scaling"] = best["scaling"]
            for d in wanted[:room]:
                rows.append(_fit_and_score(
                    lambda dd=d, x=extra: PreconditionedEmu(
                        Xf, Yf, order=best["order"], estimator="polynomial",
                        rank=best.get("rank") or "auto",
                        init_deg_forward=dd, max_degree_forward=dd,
                        RMSE_tol=0.0,
                        **{k: v for k, v in common.items() if k != "rank"}, **x
                    ),
                    Xh, Yh, f"polynomial+degree{d}", "3b-degree",
                    {"estimator": "polynomial", "order": best["order"],
                     "scaling": best["scaling"],
                     "basis_kind": best["basis_kind"],
                     "rank": best.get("rank"), "degree": d},
                ))
                spent += 1
            if len(wanted) > room:
                stages_cut.append(
                    f"3b-degree (budget covered {room} of {len(wanted)})"
                )
            stages_run.append("3b-degree")
            best = _parsimonious(
                [r for r in rows if r["stage"] == "3b-degree"] + [best],
                parsimony_tol,
            )

    # Stage 4: refit the winner on everything. The scored fits all saw a
    # held-out split, so the returned model must not be one of them.
    # Only "rank" can legitimately be absent: it is set by stage 3, which the
    # budget or a non-rotating order may skip.
    config = {
        "estimator": best["estimator"],
        "order": best["order"],
        "scaling": best["scaling"],
        "basis_kind": best["basis_kind"],
        "rank": best.get("rank"),
        "degree": best.get("degree"),
    }
    final = dict(kwargs)
    if config["scaling"] is not None:
        final["scaling"] = config["scaling"]
    if config["basis_kind"] is not None:
        final["basis_kind"] = config["basis_kind"]
    if config["estimator"] == "factored":
        final["blocks"] = blocks
    elif config["estimator"] == "sparse":
        final.setdefault("n_terms", _sparse_size(keep.size))
    elif config["estimator"] == "polynomial":
        if config["degree"] is not None:
            # Pin the measured degree instead of re-running the sweep whose
            # leave-one-out selection is what stage 3b exists to overrule.
            final.setdefault("init_deg_forward", int(config["degree"]))
            final.setdefault("max_degree_forward", int(config["degree"]))
            final.setdefault("RMSE_tol", 0.0)
        else:
            _enable_the_degree_climb(final, scan_degree)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit_common = {k: v for k, v in common.items() if k != "rank"}
        model = PreconditionedEmu(
            X, Y, order=config["order"], estimator=config["estimator"],
            rank=config["rank"] if config["rank"] is not None else "auto",
            **refit_common, **final,
        )
    stages_run.append("4-refit")
    for row in rows:
        row["chosen"] = row is best
        row.pop("model", None)
    for row in rows:
        row.setdefault("rank", None)
        row.setdefault("degree", None)
    return Recommendation(model, config, rows, int(budget), spent,
                          stages_run, stages_cut, auto_rank)
