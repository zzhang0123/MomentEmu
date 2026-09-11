"""P3.2: linearised posterior bias with a Mahalanobis norm."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu

ARCHIVE = Path("/Users/zzhang/Workspace/MomentEmu-review-2026-09/r2")


def test_linear_gaussian_gls_shift_to_1e_10():
    rng = np.random.default_rng(0)
    n, m = 3, 5
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    X = rng.uniform(-1.0, 1.0, (300, n))
    Y = X @ A.T + b
    emu = PolyEmu(
        X, Y, init_deg_forward=1, max_degree_forward=1, RMSE_tol=1e-300, verbose=0
    )
    theta = rng.uniform(-0.5, 0.5, n)
    offset = rng.normal(0.0, 0.1, m)
    Y_true = (A @ theta + b) + offset
    # SPD covariance
    B = rng.normal(size=(m, m))
    cov = B @ B.T + m * np.eye(m)
    res = emu.posterior_bias(theta, Y_true, cov=cov)
    L = np.linalg.cholesky(cov)
    Aw = np.linalg.solve(L, A)                # (m, n) = L^-1 A
    F = Aw.T @ Aw
    rhs = Aw.T @ np.linalg.solve(L, offset)
    delta_hand = np.linalg.solve(F, rhs)
    np.testing.assert_allclose(res["delta"], delta_hand, rtol=1e-10, atol=1e-12)
    assert res["mahalanobis"] == pytest.approx(
        float(np.sqrt(delta_hand @ F @ delta_hand)), rel=1e-10
    )


@pytest.mark.skipif(
    not (ARCHIVE / "bayes_cache_emus.pkl").exists(), reason="review archive not present"
)
@pytest.mark.parametrize(
    "key,lo,hi", [("d5_log1", 0.20, 0.30), ("d6_log0", 0.0, 0.05), ("d3_log1", 5.0, np.inf)],
)
def test_cached_problem1_mahalanobis(key, lo, hi):
    if str(ARCHIVE) not in sys.path:
        sys.path.insert(0, str(ARCHIVE))
    import bayes_common as bc

    with open(ARCHIVE / "bayes_cache_emus.pkl", "rb") as f:
        cache = pickle.load(f)
    with open(ARCHIVE / "bayes_cache_stage2.pkl", "rb") as f:
        stage2 = pickle.load(f)
    truth = stage2["truth"]
    mt = np.asarray(truth["mean"])
    sig = np.load(ARCHIVE / "bayes_cache_problem.npz")["sigma"]
    Y_true = np.asarray(bc.f_sim_np(mt)).ravel()
    emu = cache["emus"][key]
    res = emu.posterior_bias(mt, Y_true, sigma=sig)
    assert lo <= res["mahalanobis"] <= hi, (key, res["mahalanobis"])

@pytest.mark.skipif(
    not (ARCHIVE / "bayes_cache_emus.pkl").exists(), reason="review archive not present"
)
def test_cached_stage9_fraction_above_0_1_sigma():
    with open(ARCHIVE / "bayes_cache_emus.pkl", "rb") as f:
        cache = pickle.load(f)
    sig = np.load(ARCHIVE / "bayes_cache_problem.npz")["sigma"]
    emu = cache["emus"]["d6_log0"]
    res = emu.posterior_bias_map(cache["Xv"], cache["Yv"], sigma=sig)
    # Stage-9 (r2/bayes_stage9.log): 0.215 +/- 0.01.
    assert res["marginal_frac_above"] == pytest.approx(0.215, abs=0.01)
