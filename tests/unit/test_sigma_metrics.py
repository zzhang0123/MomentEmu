"""P3.1: error metrics in units of the user sigma (D10)."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu


def _fit(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + 5.0).reshape(-1, 1)
    return PolyEmu(X, Y, max_degree_forward=4, verbose=0), X, Y


def test_requires_sigma_or_cov():
    emu, X, Y = _fit()
    with pytest.raises(ValueError, match="requires sigma or cov"):
        emu.validate(X[:20], Y[:20])


def test_self_prediction_dchi2_is_zero():
    emu, X, Y = _fit(seed=1)
    pred = emu.forward_emulator(X[:50], extrapolation="ignore")
    res = emu.validate(X[:50], pred, sigma=np.ones((50, 1)))
    assert res["dchi2_median"] == 0.0
    assert res["dchi2_p99"] == 0.0
    assert res["max_abs_over_sigma"] == 0.0


def test_known_residual_matches_hand_computation():
    emu, X, Y = _fit(seed=2)
    Xv = X[:100]
    pred = emu.forward_emulator(Xv, extrapolation="ignore")
    rng = np.random.default_rng(3)
    delta = rng.normal(0.0, 0.05, pred.shape)
    sigma = np.full(pred.shape, 0.1)
    Yv = pred - delta  # r = pred - Yv = delta
    res = emu.validate(Xv, Yv, sigma=sigma)
    dchi2 = np.sum((delta / sigma) ** 2, axis=1)
    assert res["dchi2_median"] == pytest.approx(float(np.median(dchi2)), rel=1e-12)
    assert res["dchi2_p99"] == pytest.approx(float(np.percentile(dchi2, 99)), rel=1e-12)
    rms = np.sqrt(np.mean((delta / sigma) ** 2, axis=0))
    np.testing.assert_allclose(res["rms_over_sigma"], rms, rtol=1e-12)


def test_cov_and_sigma_agree_for_diagonal_cov():
    emu, X, Y = _fit(seed=4)
    Xv = X[:60]
    pred = emu.forward_emulator(Xv, extrapolation="ignore")
    rng = np.random.default_rng(5)
    delta = rng.normal(0.0, 0.03, pred.shape)
    Yv = pred - delta
    sig = np.full(pred.shape[1], 0.07)
    a = emu.validate(Xv, Yv, sigma=sig)
    b = emu.validate(Xv, Yv, cov=np.diag(sig ** 2))
    assert a["dchi2_median"] == pytest.approx(b["dchi2_median"], rel=1e-10)

ARCHIVE = __import__("pathlib").Path("/Users/zzhang/Workspace/MomentEmu-review-2026-09/r2")


@pytest.mark.skipif(
    not (ARCHIVE / "bayes_cache_emus.pkl").exists(), reason="review archive not present"
)
@pytest.mark.parametrize(
    "key,expected",
    [("d5_log0", 5.76e-01), ("d5_log1", 7.23e-02), ("d6_log0", 2.23e-02), ("d3_log1", 5.82e01)],
)
def test_cached_problem1_dchi2_median(key, expected):
    import pickle

    with open(ARCHIVE / "bayes_cache_emus.pkl", "rb") as f:
        cache = pickle.load(f)
    sig = np.load(ARCHIVE / "bayes_cache_problem.npz")["sigma"]
    emu = cache["emus"][key]
    res = emu.validate(cache["Xv"], cache["Yv"], sigma=sig, extrapolation="ignore")
    # The dchi2med column of r2/finish/bayesfin_metrics.log.
    assert res["dchi2_median"] == pytest.approx(expected, rel=5e-3)
