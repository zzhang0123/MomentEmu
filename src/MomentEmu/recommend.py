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
                 stages_cut: list[str]) -> None:
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
    keep, hold = _split(X.shape[0], validation_split, random_state)
    Xf, Yf, Xh, Yh = X[keep], Y[keep], X[hold], Y[hold]

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
            extra.setdefault(
                "degree", _affordable_degree(X.shape[1], keep.size, scan_degree)
            )
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

    # Stage 4: refit the winner on everything. The scored fits all saw a
    # held-out split, so the returned model must not be one of them.
    config = {k: best[k] for k in ("estimator", "order", "scaling", "basis_kind")}
    final = dict(kwargs)
    if config["scaling"] is not None:
        final["scaling"] = config["scaling"]
    if config["basis_kind"] is not None:
        final["basis_kind"] = config["basis_kind"]
    if config["estimator"] == "factored":
        final["blocks"] = blocks
    elif config["estimator"] == "sparse":
        final.setdefault(
            "degree", _affordable_degree(X.shape[1], keep.size, scan_degree)
        )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PreconditionedEmu(
            X, Y, order=config["order"], estimator=config["estimator"],
            **common, **final,
        )
    stages_run.append("4-refit")
    for row in rows:
        row["chosen"] = row is best
        row.pop("model", None)
    return Recommendation(model, config, rows, int(budget), spent,
                          stages_run, stages_cut)
