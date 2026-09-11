"""Suite 5a: timing noise on this machine.

Repeats two representative fits and two inference timings R times in the same
process and reports the coefficient of variation, which is the floor any
CI tolerance must sit above.
"""
from __future__ import annotations

import statistics

import numpy as np

from benchmarks.harness import fit_polyemu, fixed_degree_kwargs, time_inference
from benchmarks.targets import TARGETS


def _cv(xs):
    return statistics.pstdev(xs) / statistics.mean(xs)


def run(quick: bool = False, repeats: int = 7) -> dict:
    R = 3 if quick else repeats
    rows = []
    for name in ("ishigami", "cmb_like"):
        t = TARGETS[name]
        X, Y, Xt, Yt = t.data()
        fit_times, fit_min3, single, batch = [], [], [], []
        emu = None
        for _ in range(R):
            emu, dt = fit_polyemu(X, Y, Xt, Yt, **fixed_degree_kwargs(t.fixed_degree))
            fit_times.append(dt)
            # what the gated suites record: the minimum of three consecutive fits
            fit_min3.append(min(fit_polyemu(X, Y, Xt, Yt, **fixed_degree_kwargs(t.fixed_degree))[1] for _ in range(3)))
            ti = time_inference(emu.forward_emulator, Xt, min_time=0.1, repeats=3)
            single.append(ti["single_us"])
            batch.append(ti["batch_us"])
        rows.append({
            "target": name, "repeats": R,
            "fit_s_mean": statistics.mean(fit_times), "fit_s_min": min(fit_times), "fit_s_max": max(fit_times),
            "fit_s_cv": _cv(fit_times), "fit_max_over_min": max(fit_times) / min(fit_times),
            "fit_min3_s_mean": statistics.mean(fit_min3), "fit_min3_cv": _cv(fit_min3),
            "fit_min3_max_over_min": max(fit_min3) / min(fit_min3),
            "single_us_mean": statistics.mean(single), "single_us_cv": _cv(single),
            "single_max_over_min": max(single) / min(single),
            "batch_us_mean": statistics.mean(batch), "batch_us_cv": _cv(batch),
            "batch_max_over_min": max(batch) / min(batch),
            # raw samples, so a CI-tolerance false-positive rate can be computed afterwards
            "fit_s_all": fit_times, "fit_min3_s_all": fit_min3,
            "single_us_all": single, "batch_us_all": batch,
        })
        r = rows[-1]
        print(f"[noise] {name:9s} fit cv={r['fit_s_cv']*100:.1f}% max/min={r['fit_max_over_min']:.2f} "
              f"(min-of-3: cv={r['fit_min3_cv']*100:.1f}% max/min={r['fit_min3_max_over_min']:.2f}) | "
              f"single cv={r['single_us_cv']*100:.1f}% max/min={r['single_max_over_min']:.2f} | "
              f"batch cv={r['batch_us_cv']*100:.1f}% max/min={r['batch_max_over_min']:.2f}", flush=True)
    return {"suite": "noise", "rows": rows}
