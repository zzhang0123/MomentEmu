"""One memory measurement in a fresh process: peak RSS and tracemalloc peak of a fit.

Usage: python -m bench.memcell N batch_size dim_reduction  ->  prints one JSON line.
batch_size 'none' maps to the constructor default (None).
"""
from __future__ import annotations

import json
import resource
import sys
import time
import tracemalloc

import numpy as np

from benchmarks.harness import fit_polyemu, fixed_degree_kwargs
from benchmarks.suite_scaling import smooth_target

N_PARAM, DEGREE, M_OUT = 6, 5, 100


def main(argv):
    N = int(argv[0])
    bs = None if argv[1] == "none" else int(argv[1])
    dim_reduction = argv[2] == "1"
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (N, N_PARAM))
    Xt = rng.uniform(-1, 1, (500, N_PARAM))
    Y, Yt = smooth_target(X, M_OUT, 3), smooth_target(Xt, M_OUT, 3)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    kw = fixed_degree_kwargs(DEGREE)
    kw["dim_reduction"] = dim_reduction
    t0 = time.perf_counter()
    emu, t_fit = fit_polyemu(X, Y, Xt, Yt, batch_size=bs, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    unit = 1 if sys.platform == "darwin" else 1024
    out = {
        "N": N, "batch_size": bs, "dim_reduction": dim_reduction,
        "D": int(len(emu.forward_multi_indices)), "m": M_OUT,
        "fit_s": t_fit,
        "tracemalloc_peak_MB": peak / 1e6,
        "rss_peak_MB": rss_after * unit / 1e6,
        "rss_before_fit_MB": rss_before * unit / 1e6,
        "data_MB": (X.nbytes + Y.nbytes) / 1e6,
        "phi_full_MB": N * len(emu.forward_multi_indices) * 8 / 1e6,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
