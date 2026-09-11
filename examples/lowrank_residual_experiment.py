"""P3.7: low-rank residual covariance experiment (research; no helper shipped).

Compares the linearised posterior bias under the diagonal data covariance with
C = C_data + U diag(s) U^T built from the top-k eigenvectors of the validation
residual covariance. The plan ships a helper only if the low-rank term brings
max |shift| < 0.1 sigma with widths < 1.2x; otherwise it is reported as a
negative result.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ARCHIVE = Path("/Users/zzhang/Workspace/MomentEmu-review-2026-09/r2")


def lowrank_cov(residual, sigma, k):
    """Return C_data + U diag(s) U^T from the top-k residual modes."""
    r = np.asarray(residual, float)
    r = r - r.mean(axis=0, keepdims=True)
    m = r.shape[1]
    if k <= 0:
        return np.diag(sigma ** 2)
    # Economy SVD of the residual matrix; the output covariance uses the
    # right singular vectors (columns are the m outputs).
    _, s, Vt = np.linalg.svd(r, full_matrices=False)
    V = Vt[:k].T
    return np.diag(sigma ** 2) + (V * (s[:k] ** 2 / (len(r) - 1))) @ V.T


def max_marginal_shift(J, delta, cov):
    L = np.linalg.cholesky(cov)
    Jw = np.linalg.solve(L, J)
    rw = np.linalg.solve(L, delta)
    F = Jw.T @ Jw
    shift = -np.linalg.solve(F, Jw.T @ rw)
    sd = np.sqrt(np.diag(np.linalg.inv(F)))
    return float(np.max(np.abs(shift / sd))), float(np.sqrt((shift) @ F @ shift))


def run(keys=("d4_log0", "d5_log0", "d4_log1", "d5_log1"), k=36):
    sys.path.insert(0, str(ARCHIVE))
    import bayes_common as bc

    with open(ARCHIVE / "bayes_cache_emus.pkl", "rb") as f:
        cache = pickle.load(f)
    with open(ARCHIVE / "bayes_cache_stage2.pkl", "rb") as f:
        stage2 = pickle.load(f)
    sig = np.load(ARCHIVE / "bayes_cache_problem.npz")["sigma"]
    truth = stage2["truth"]
    mt = np.asarray(truth["mean"])
    Y_true = np.asarray(bc.f_sim_np(mt)).ravel()
    rows = []
    for key in keys:
        emu = cache["emus"][key]
        J = np.atleast_2d(emu.jacobian(mt))
        delta = emu.forward_emulator(mt, extrapolation="ignore").ravel() - Y_true
        residual = emu.forward_emulator(cache["Xv"], extrapolation="ignore") - cache["Yv"]
        for label, cov in (
            ("diag", np.diag(sig ** 2)),
            (f"lowrank{k}", lowrank_cov(residual, sig, k)),
        ):
            marg, mah = max_marginal_shift(J, delta, cov)
            rows.append({"key": key, "model": label, "max_marginal": marg, "mahalanobis": mah})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=36)
    a = ap.parse_args(argv)
    if not (ARCHIVE / "bayes_cache_emus.pkl").exists():
        print("review archive not present; nothing to run")
        return 0
    rows = run(k=a.k)
    print(f"{'key':10s} {'model':10s} {'max|shift|':>10s} {'mahalanobis':>12s}")
    for r in rows:
        print(f"{r['key']:10s} {r['model']:10s} {r['max_marginal']:10.3f} {r['mahalanobis']:12.3f}")
    best = max(
        (r for r in rows if r["model"].startswith("lowrank")),
        key=lambda r: r["max_marginal"],
        default=None,
    )
    if best is not None:
        print(
            "low-rank passes the 0.1-sigma criterion:",
            best["max_marginal"] < 0.1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
