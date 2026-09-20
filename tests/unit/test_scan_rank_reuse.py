"""scan_rank computes the rotation once, not once per rank.

The rotation does not depend on the rank: active_subspace returns the full
eigendecomposition and the rank only decides how many columns are kept. But
scan_rank built a fresh ActiveSubspaceEmu per rank, so a five-rank scan fitted
the pilot polynomial five times -- and with T-005, called the caller's
Jacobian five times, which for a JAX or PyTorch model is the expensive part.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.rotation import active_subspace, scan_rank

RANKS = (1, 2, 3)
DEGREE = 4


def _ridge_design(n_samples: int = 700, seed: int = 0):
    """A 2-D active subspace inside 5 parameters."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((2, 5))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (n_samples, 5))
    U = X @ W.T
    Y = np.column_stack([np.tanh(U[:, 0]) + 0.4 * U[:, 1] ** 2,
                         np.sin(0.8 * U[:, 0] * U[:, 1])])
    return X, Y, W


def _exact_jacobian_factory(W):
    calls = [0]

    def jac(chunk: np.ndarray) -> np.ndarray:
        calls[0] += 1
        u = chunk @ W.T
        c = np.cos(0.8 * u[:, 0] * u[:, 1])
        d0 = np.stack([1.0 - np.tanh(u[:, 0]) ** 2, 0.8 * u[:, 1]], axis=1)
        d1 = np.stack([0.8 * c * u[:, 1], 0.8 * c * u[:, 0]], axis=1)
        return np.stack([d0 @ W, d1 @ W], axis=1)

    return jac, calls


def test_the_jacobian_is_evaluated_once_for_the_whole_scan() -> None:
    X, Y, W = _ridge_design()
    jac, calls = _exact_jacobian_factory(W)
    rows = scan_rank(X, Y, RANKS, DEGREE, jacobian=jac,
                     RMSE_tol=0.0, verbose=0)
    assert len(rows) == len(RANKS)
    assert calls[0] == 1, (
        f"the Jacobian was evaluated {calls[0]} times for {len(RANKS)} ranks"
    )


def test_the_pilot_is_fitted_once_for_the_whole_scan(monkeypatch) -> None:
    import MomentEmu.rotation as rotation

    X, Y, _W = _ridge_design()
    calls = [0]
    real = rotation.gradient_covariance_matrix

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(rotation, "gradient_covariance_matrix", counting)
    rotation.scan_rank(X, Y, RANKS, DEGREE, RMSE_tol=0.0, verbose=0)
    assert calls[0] == 1, f"the gradient covariance was built {calls[0]} times"


def test_reuse_does_not_change_the_scan() -> None:
    """The rotation reused across ranks must be the one each rank would
    have computed for itself."""
    X, Y, _W = _ridge_design()
    rows = scan_rank(X, Y, RANKS, DEGREE, RMSE_tol=0.0, verbose=0)
    evals, V = active_subspace(X, Y)
    total = float(evals.sum())
    for row, rank in zip(sorted(rows, key=lambda r: r["rank"]), RANKS):
        assert row["rank"] == rank
        assert row["error"] is None
        expected = float(np.cumsum(evals)[rank - 1] / total)
        assert row["variance_share"] == pytest.approx(expected, rel=1e-9)


def test_a_supplied_covariance_matches_supplying_the_jacobian() -> None:
    """Only over the ranks the spectrum actually supports.

    This design has a 2-D active subspace in 5 parameters, so from rank 3 on
    the extra column is an arbitrary vector of the null space and its
    orientation depends on round-off. Comparing a fom there compares noise;
    the test would fail for a reason that has nothing to do with the reuse.
    """
    from MomentEmu.rotation import gradient_covariance_matrix

    X, Y, W = _ridge_design()
    jac, _calls = _exact_jacobian_factory(W)
    C = gradient_covariance_matrix(X, Y, jacobian=jac)
    evals, _V = active_subspace(X, Y, jacobian=jac)
    supported = [r for r in RANKS if evals[r - 1] > 1e-12 * evals[0]]
    assert supported, "the spectrum no longer has a usable leading block"

    from_c = {row["rank"]: row for row in
              scan_rank(X, Y, supported, DEGREE, gradient_covariance=C,
                        RMSE_tol=0.0, verbose=0)}
    from_j = {row["rank"]: row for row in
              scan_rank(X, Y, supported, DEGREE, jacobian=jac,
                        RMSE_tol=0.0, verbose=0)}
    for r in supported:
        assert from_c[r]["fom"] == pytest.approx(from_j[r]["fom"], rel=1e-8)
