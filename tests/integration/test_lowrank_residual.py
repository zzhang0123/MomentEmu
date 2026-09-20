"""P3.7: low-rank residual covariance experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "lowrank_residual_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("lowrank_residual", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lowrank_cov_matches_hand_computation():
    mod = _load()
    rng = np.random.default_rng(0)
    r = rng.normal(size=(500, 8))
    sig = rng.uniform(0.1, 0.3, size=8)
    k = 3
    cov = mod.lowrank_cov(r, sig, k)
    rc = r - r.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(rc, full_matrices=False)
    V = Vt[:k].T
    hand = np.diag(sig ** 2) + (V * (s[:k] ** 2 / (len(r) - 1))) @ V.T
    np.testing.assert_allclose(cov, hand, rtol=1e-12, atol=1e-14)


def test_lowrank_cov_k_zero_is_diagonal():
    mod = _load()
    rng = np.random.default_rng(1)
    r = rng.normal(size=(100, 4))
    sig = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(mod.lowrank_cov(r, sig, 0), np.diag(sig ** 2))


@pytest.mark.slow
def test_experiment_runs_and_reduces_shifts(archive_bayes_common):
    mod = _load()
    rows = mod.run(keys=("d5_log0",), k=36)
    diag = next(r for r in rows if r["model"] == "diag")
    low = next(r for r in rows if r["model"].startswith("lowrank"))
    assert np.isfinite(diag["max_marginal"]) and np.isfinite(low["max_marginal"])
    # The low-rank term reduces the shift but does not reach the 0.1 criterion,
    # so no helper ships (the plan result).
    assert low["max_marginal"] < diag["max_marginal"]


def test_max_marginal_shift_closed_form():
    mod = _load()
    rng = np.random.default_rng(2)
    J = rng.normal(size=(10, 2))
    delta = rng.normal(size=10)
    cov = np.diag(rng.uniform(0.5, 2.0, size=10) ** 2)
    marg, mah = mod.max_marginal_shift(J, delta, cov)
    L = np.linalg.cholesky(cov)
    Jw = np.linalg.solve(L, J)
    rw = np.linalg.solve(L, delta)
    F = Jw.T @ Jw
    shift = -np.linalg.solve(F, Jw.T @ rw)
    sd = np.sqrt(np.diag(np.linalg.inv(F)))
    assert marg == pytest.approx(float(np.max(np.abs(shift / sd))), rel=1e-12)
    assert mah == pytest.approx(float(np.sqrt(shift @ F @ shift)), rel=1e-12)
