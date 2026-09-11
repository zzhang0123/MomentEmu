"""Suite 3d: peak memory of a fit vs N at batch_size in {None, 1000, 10000}.

Each cell runs in a fresh subprocess (bench.memcell) so ru_maxrss is per cell.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cell(N: int, bs, dim_reduction: bool) -> dict:
    args = [sys.executable, "-m", "bench.memcell", str(N), "none" if bs is None else str(bs), "1" if dim_reduction else "0"]
    env = dict(os.environ, PYTHONPATH=BENCH_ROOT)
    out = subprocess.run(args, capture_output=True, text=True, cwd=BENCH_ROOT, env=env, check=True, timeout=1800)
    return json.loads(out.stdout.strip().splitlines()[-1])


def run(quick: bool = False) -> dict:
    rows = []
    Ns = (2000, 10000, 50000) if not quick else (2000, 10000)
    for N in Ns:
        for bs in (None, 1000, 10000):
            rows.append(_cell(N, bs, False))
            r = rows[-1]
            print(f"[memory] N={N:6d} batch={str(bs):6s} dimred=0 rss_peak={r['rss_peak_MB']:8.1f}MB "
                  f"tracemalloc_peak={r['tracemalloc_peak_MB']:8.1f}MB phi_full={r['phi_full_MB']:.1f}MB fit={r['fit_s']:.2f}s", flush=True)
    # dim_reduction=True (the constructor default) at the largest N with the small batch:
    # round-1 finding dim-reduction-redundant-unbatched-moments predicts the batch bound is broken.
    rows.append(_cell(Ns[-1], 1000, True))
    r = rows[-1]
    print(f"[memory] N={Ns[-1]:6d} batch=1000   dimred=1 rss_peak={r['rss_peak_MB']:8.1f}MB "
          f"tracemalloc_peak={r['tracemalloc_peak_MB']:8.1f}MB", flush=True)
    return {"suite": "memory", "rows": rows}
