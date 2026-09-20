"""The sparse candidate set is sized in the PRECONDITIONED dimension.

A rotation exists to cut the dimension, and the degree a design affords grows
fast as it falls. Sizing the candidate set in the ORIGINAL dimension is safe --
a rotation only lowers it, so the degree stays reachable -- but it throws away
the headroom the rotation just bought. On 21cmGEM seven parameters afford
degree 7 while the five rotated ones afford 12, and the recommender's sparse
fit reached 2.2691 percent where a hand-tuned run at degree 14 with
interactions capped at 3 reached 1.4478 percent on the same 1,500 terms.

Only PreconditionedEmu knows the reduced dimension: the order is chosen inside
its constructor, so a caller cannot size the set beforehand. The polynomial
branch already clamps max_degree_forward there; this is the same thing for the
other estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.precondition import PreconditionedEmu, _affordable_degree


def _ridge(n=600, seed=2, n_params=7):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((2, n_params))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (n, n_params))
    U = X @ W.T
    Y = np.column_stack([np.tanh(1.3 * U[:, 0]) + 0.4 * U[:, 1] ** 2,
                         np.sin(0.9 * U[:, 0] * U[:, 1])])
    return X, Y


def _max_degree(model) -> int:
    mi = model.emulator.candidate_indices
    return int(mi.sum(axis=1).max())


def test_the_original_dimension_would_afford_far_less() -> None:
    """Pin the gap the fix exists to close, on this design."""
    X, _Y = _ridge()
    assert _affordable_degree(X.shape[1], X.shape[0], 12) < 4
    assert _affordable_degree(2, X.shape[0], 12) >= 10


def test_a_rotated_sparse_fit_uses_the_reduced_dimension() -> None:
    X, Y = _ridge()
    model = PreconditionedEmu(
        X, Y, order="rotate", rank=2, estimator="sparse", scan_degree=12,
        n_terms=40, random_state=0,
    )
    assert model.emulator.candidate_indices.shape[1] == 2
    flat = _affordable_degree(X.shape[1], X.shape[0], 12)
    assert _max_degree(model) > flat, (
        f"candidate set reaches degree {_max_degree(model)}, no better than "
        f"the {flat} the original {X.shape[1]} dimensions afford"
    )


def test_an_unrotated_sparse_fit_is_unchanged() -> None:
    """With no rotation the two dimensions agree, so nothing may shift."""
    X, Y = _ridge()
    model = PreconditionedEmu(
        X, Y, order="none", estimator="sparse", scan_degree=12,
        n_terms=40, random_state=0,
    )
    assert _max_degree(model) == _affordable_degree(X.shape[1], X.shape[0], 12)


def test_an_explicit_degree_is_respected() -> None:
    """Sizing is a default, not an override: a caller who asks, gets."""
    X, Y = _ridge()
    model = PreconditionedEmu(
        X, Y, order="rotate", rank=2, estimator="sparse", scan_degree=12,
        degree=5, n_terms=40, random_state=0,
    )
    assert _max_degree(model) == 5


def test_an_explicit_candidate_set_is_respected() -> None:
    from MomentEmu.basis import Basis

    X, Y = _ridge()
    model = PreconditionedEmu(
        X, Y, order="rotate", rank=2, estimator="sparse", scan_degree=12,
        candidate=Basis(degree=6), n_terms=40, random_state=0,
    )
    assert _max_degree(model) == 6


@pytest.mark.parametrize("order", ("rotate", "rotate_warp"))
def test_the_reduced_fit_is_at_least_as_good(order: str) -> None:
    """The extra degree must buy accuracy, not just a bigger candidate set."""
    X, Y = _ridge()
    common = dict(estimator="sparse", n_terms=40, random_state=0, rank=2)
    wide = PreconditionedEmu(X, Y, order=order, scan_degree=12, **common)
    narrow = PreconditionedEmu(
        X, Y, order=order, scan_degree=12,
        degree=_affordable_degree(X.shape[1], X.shape[0], 12), **common,
    )
    def err(m) -> float:
        return float(np.linalg.norm(m.forward_emulator(X) - Y))

    assert err(wide) <= err(narrow) * 1.05, (
        f"{order}: sizing in the reduced dimension scored {err(wide):.4e} "
        f"against {err(narrow):.4e}"
    )
