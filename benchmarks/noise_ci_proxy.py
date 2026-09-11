"""Timing noise under CI-like conditions, measured on this machine.

GitHub-hosted runners cannot be measured from here, so this script measures
the two things a runner changes and that can be reproduced locally:

  threads2   : BLAS restricted to 2 threads (VECLIB_MAXIMUM_THREADS /
               OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 2), idle machine.
  contended  : same, plus (cpu_count - 2) busy-loop processes so that the
               benchmark competes for ~2 free cores, a proxy for a 2-4 vCPU
               shared VM.

Each condition runs the noise suite in a fresh subprocess. From the raw
samples it then computes the false-positive rate of a timing gate at a given
relative tolerance: the fraction of ordered pairs (baseline sample, current
sample) for which current > baseline * (1 + tol), for a single-shot baseline
and for a min-of-3 baseline.

    python -m bench.noise_ci_proxy [--out results/noise_ci_proxy.json] [--repeats 7]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile

from benchmarks.harness import read_json, write_json

BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _busy():
    while True:
        pass


def _run_noise(out_dir: str, threads: int | None, repeats: int) -> list[dict]:
    env = dict(os.environ, PYTHONPATH=BENCH_ROOT)
    if threads is not None:
        for k in THREAD_VARS:
            env[k] = str(threads)
    subprocess.run([sys.executable, "-c",
                    f"from benchmarks.suite_noise import run; from benchmarks.harness import write_json; "
                    f"write_json(run(repeats={repeats}), {os.path.join(out_dir, 'noise.json')!r})"],
                   cwd=BENCH_ROOT, env=env, check=True, capture_output=True, text=True, timeout=3600)
    return read_json(os.path.join(out_dir, "noise.json"))["rows"]


def false_positive_rate(samples: list[float], tol: float, baseline_min_of: int = 1) -> float:
    """Fraction of (baseline, current) draws with current > baseline*(1+tol).

    baseline_min_of = k: the baseline is the minimum of k distinct samples;
    the current run is one further sample not in that set.
    """
    n = len(samples)
    fails = total = 0
    for base_idx in itertools.combinations(range(n), baseline_min_of):
        base = min(samples[i] for i in base_idx)
        for j in range(n):
            if j in base_idx:
                continue
            total += 1
            fails += samples[j] > base * (1 + tol)
    return fails / total if total else float("nan")


def summarise(rows: list[dict], condition: str) -> list[dict]:
    out = []
    for r in rows:
        rec = {"condition": condition, "target": r["target"], "repeats": r["repeats"],
               "fit_s_mean": r["fit_s_mean"], "fit_max_over_min": r["fit_max_over_min"],
               "fit_min3_max_over_min": r.get("fit_min3_max_over_min"),
               "single_us_mean": r["single_us_mean"], "single_max_over_min": r["single_max_over_min"],
               "batch_us_mean": r["batch_us_mean"], "batch_max_over_min": r["batch_max_over_min"]}
        for tol in (0.30, 0.50):
            key = f"{int(tol*100)}pct"
            rec[f"fp_fit_single_{key}"] = false_positive_rate(r["fit_s_all"], tol, 1)
            rec[f"fp_fit_min3base_{key}"] = false_positive_rate(r["fit_s_all"], tol, 3)
            rec[f"fp_fitmin3_vs_fitmin3_{key}"] = false_positive_rate(r["fit_min3_s_all"], tol, 1)
            rec[f"fp_single_{key}"] = false_positive_rate(r["single_us_all"], tol, 1)
            rec[f"fp_batch_{key}"] = false_positive_rate(r["batch_us_all"], tol, 1)
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(BENCH_ROOT, "results", "noise_ci_proxy.json"))
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--idle-noise", default=os.path.join(BENCH_ROOT, "results", "noise.json"),
                    help="noise.json of the idle, default-thread run (reused, not re-run)")
    a = ap.parse_args(argv)
    ncpu = os.cpu_count() or 4
    tmp = tempfile.mkdtemp(prefix="noise_ci_")
    rows = []
    if os.path.exists(a.idle_noise):
        rows += summarise(read_json(a.idle_noise)["rows"], "idle_default_threads")
    print("[noise-ci] threads=2, idle", flush=True)
    rows += summarise(_run_noise(os.path.join(tmp, "t2"), 2, a.repeats), "idle_2_threads")
    n_busy = max(ncpu - 2, 1)
    print(f"[noise-ci] threads=2, {n_busy} busy-loop processes", flush=True)
    procs = [multiprocessing.Process(target=_busy, daemon=True) for _ in range(n_busy)]
    for p in procs:
        p.start()
    try:
        rows += summarise(_run_noise(os.path.join(tmp, "t2busy"), 2, a.repeats), f"contended_2_threads_{n_busy}_busy")
    finally:
        for p in procs:
            p.terminate()
    for r in rows:
        print(f"{r['condition']:32s} {r['target']:9s} fit max/min={r['fit_max_over_min']:.2f} "
              f"min3 max/min={r['fit_min3_max_over_min']:.2f} 1pt max/min={r['single_max_over_min']:.2f} "
              f"1000pt max/min={r['batch_max_over_min']:.2f} | FP@30%: fit {r['fp_fit_single_30pct']:.2f} "
              f"min3-base {r['fp_fit_min3base_30pct']:.2f} 1pt {r['fp_single_30pct']:.2f} | FP@50%: fit {r['fp_fit_single_50pct']:.2f} "
              f"min3-base {r['fp_fit_min3base_50pct']:.2f}", flush=True)
    write_json({"suite": "noise_ci_proxy", "n_cpu": ncpu, "rows": rows}, a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
