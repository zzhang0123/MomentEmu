"""Posterior-shift acceptance protocol (P3.3).

Compare the posterior of a simulator (CAMB or a synthetic truth) with the
posterior of an emulator on the same data.  The acceptance is: every marginal
mean shift < 0.1 sigma, the Mahalanobis shift of the mean < 0.1 and every
width ratio within 10 %.  A Monte-Carlo floor can be supplied.

Usage:
    python examples/validate_posterior.py [--chains DIR]

With no arguments the companion Planck TTTEEE+lowE chains are used when they
are present; otherwise a small synthetic example is run.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

COMPANION_CHAINS = (
    "/Users/zzhang/Workspace/MomentEmu-PolyCAMB-examples/chains"
)
PLANCK_PARAMS = [
    "omega_b", "omega_c", "theta_star", "logA", "ns", "tau", "A_planck"
]
ABS_SHIFT_TOL = 0.1
WIDTH_TOL = 0.10
MAHALANOBIS_TOL = 0.1


def shift_report(mean_ref, sd_ref, cov_ref, mean_emu, sd_emu, cov_emu, names,
                 *, mc_err=None, abs_tol=ABS_SHIFT_TOL, width_tol=WIDTH_TOL,
                 mahalanobis_tol=MAHALANOBIS_TOL, mahalanobis_indices=None):
    """Marginal shifts, width ratios, Mahalanobis shift and a PASS verdict.

    ``mc_err`` is an optional per-parameter Monte-Carlo error on the mean in
    sigma units; it is reported but does not enter the verdict (the plan uses
    3x the combined MC error only when comparing two NUTS runs directly).
    """
    mean_ref = np.asarray(mean_ref, float)
    mean_emu = np.asarray(mean_emu, float)
    sd_ref = np.asarray(sd_ref, float)
    sd_emu = np.asarray(sd_emu, float)
    cov_ref = np.asarray(cov_ref, float)
    shifts = (mean_emu - mean_ref) / sd_ref
    ratios = sd_emu / sd_ref
    if mahalanobis_indices is None:
        mahalanobis_indices = np.arange(len(names))
    mahalanobis_indices = np.asarray(mahalanobis_indices, dtype=int)
    d = (mean_emu - mean_ref)[mahalanobis_indices]
    cov_sub = cov_ref[np.ix_(mahalanobis_indices, mahalanobis_indices)]
    mahalanobis = float(np.sqrt(d @ np.linalg.solve(cov_sub, d)))
    passes = bool(
        np.all(np.abs(shifts) < abs_tol)
        and np.all(np.abs(ratios - 1.0) < width_tol)
        and mahalanobis < mahalanobis_tol
    )
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "name": name,
            "mean_ref": float(mean_ref[i]),
            "sd_ref": float(sd_ref[i]),
            "mean_emu": float(mean_emu[i]),
            "sd_emu": float(sd_emu[i]),
            "shift_sigma": float(shifts[i]),
            "width_ratio": float(ratios[i]),
            "mc_err": None if mc_err is None else float(mc_err[i]),
        })
    worst = max(rows, key=lambda r: abs(r["shift_sigma"]))
    worst_width = max(rows, key=lambda r: abs(r["width_ratio"] - 1.0))
    return {
        "rows": rows,
        "worst_shift": worst,
        "worst_width": worst_width,
        "mahalanobis": mahalanobis,
        "pass": passes,
    }


def print_report(report):
    print(f"{'param':12s} {'mean_ref':>12s} {'sd_ref':>10s} {'mean_emu':>12s} "
          f"{'sd_emu':>10s} {'shift/sig':>10s} {'width':>8s} {'MCerr':>7s}")
    for r in report["rows"]:
        mc = "-" if r["mc_err"] is None else f"{r['mc_err']:.3f}"
        print(f"{r['name']:12s} {r['mean_ref']:12.6g} {r['sd_ref']:10.4g} "
              f"{r['mean_emu']:12.6g} {r['sd_emu']:10.4g} "
              f"{r['shift_sigma']:+10.3f} {r['width_ratio']:8.4f} {mc:>7s}")
    w = report["worst_shift"]
    ww = report["worst_width"]
    print(f"worst |shift|: {w['name']} at {w['shift_sigma']:+.3f} sigma; "
          f"worst width: {ww['name']} at {100 * (ww['width_ratio'] - 1):+.1f}%")
    print(f"Mahalanobis shift: {report['mahalanobis']:.3f}")
    print("PASS 0.1-sigma / 10% / 0.1-Mahalanobis:", report["pass"])


def _load_chains(chains_dir):
    from getdist import loadMCSamples

    camb = loadMCSamples(os.path.join(chains_dir, "planck_camb_pol"), settings={"ignore_rows": 0.3})
    poly = loadMCSamples(os.path.join(chains_dir, "planck_polycamb_pol"), settings={"ignore_rows": 0.3})
    names = [p for p in PLANCK_PARAMS if camb.paramNames.numberOfName(p) >= 0 and poly.paramNames.numberOfName(p) >= 0]
    mean_c = np.array([camb.mean(p) for p in names])
    sd_c = np.array([camb.std(p) for p in names])
    mean_p = np.array([poly.mean(p) for p in names])
    sd_p = np.array([poly.std(p) for p in names])
    cov_c = camb.cov(names)
    cov_p = poly.cov(names)
    mc_err = np.full(len(names), np.nan)
    try:
        for i, p in enumerate(names):
            neff_c = camb.getEffectiveSamples(camb.paramNames.numberOfName(p))
            neff_p = poly.getEffectiveSamples(poly.paramNames.numberOfName(p))
            mc_err[i] = np.sqrt(1.0 / neff_c + 1.0 / neff_p)
    except Exception:
        pass
    # The Mahalanobis shift in the plan is over the six cosmology params
    # (A_planck is a nuisance parameter).
    mah_indices = [i for i, p in enumerate(names) if p != "A_planck"]
    return shift_report(
        mean_c, sd_c, cov_c, mean_p, sd_p, cov_p, names,
        mc_err=mc_err, mahalanobis_indices=mah_indices,
    )


def _synthetic():
    rng = np.random.default_rng(0)
    names = ["p0", "p1", "p2"]
    cov = np.diag([0.01, 0.02, 0.03])
    mean = np.zeros(3)
    sd = np.sqrt(np.diag(cov))
    shift = np.array([0.02, -0.01, 0.03])
    return shift_report(mean, sd, cov, mean + shift, sd, cov, names)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", default=COMPANION_CHAINS)
    a = ap.parse_args(argv)
    if (
        os.path.isdir(a.chains)
        and os.path.exists(os.path.join(a.chains, "planck_camb_pol.1.txt"))
        and os.path.exists(os.path.join(a.chains, "planck_polycamb_pol.1.txt"))
    ):
        report = _load_chains(a.chains)
    else:
        print("companion chains not found; running the synthetic example")
        report = _synthetic()
    print_report(report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
