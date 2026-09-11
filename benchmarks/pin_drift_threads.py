"""Measure pin drift when the BLAS thread count changes.

Runs the pins suite in two subprocesses (VECLIB_MAXIMUM_THREADS / OMP_NUM_THREADS
= 1 and = default) and reports the maximum relative coefficient difference and
RMSE difference per pin, i.e. the drift a 1e-10 gate has to tolerate across
thread configurations on one machine.

    python -m benchmarks.pin_drift_threads
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from benchmarks.harness import read_json

BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_pins(threads, out):
    env = dict(os.environ, PYTHONPATH=BENCH_ROOT)
    if threads is not None:
        for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[k] = str(threads)
    subprocess.run([sys.executable, "-m", "benchmarks", "--suites", "pins", "--out", out],
                   cwd=BENCH_ROOT, env=env, check=True, capture_output=True, text=True, timeout=1200)
    return read_json(os.path.join(out, "pins.json"))["rows"]


def main():
    tmp = tempfile.mkdtemp(prefix="pin_threads_")
    a = _run_pins(1, os.path.join(tmp, "t1"))
    b = _run_pins(None, os.path.join(tmp, "tdefault"))
    rows = []
    for ra, rb in zip(a, b):
        ca, cb = np.array(ra["forward_coeffs_first10"]), np.array(rb["forward_coeffs_first10"])
        rel = float(np.max(np.abs(ca - cb)) / ra["forward_coeff_abs_max"])
        rmse_rel = abs(ra["test_rmse"] - rb["test_rmse"]) / ra["test_rmse"]
        sum_rel = abs(ra["forward_coeff_sum"] - rb["forward_coeff_sum"]) / max(abs(ra["forward_coeff_sum"]), 1e-300)
        rows.append({"id": ra["id"], "cond_M": ra["cond_M"], "coef_first10_rel_diff": rel,
                     "coef_sum_rel_diff": sum_rel, "rmse_rel_diff": rmse_rel})
        print(f"{ra['id']:24s} cond={ra['cond_M']:.2e} first10 rel diff={rel:.2e} coef-sum rel diff={sum_rel:.2e} rmse rel diff={rmse_rel:.2e}")
    out = os.path.join(BENCH_ROOT, "results", "pin_drift_threads.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "threads_a": 1, "threads_b": "default"}, f, indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
