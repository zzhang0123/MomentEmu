"""Merge several result directories into one baseline by taking, per cell, the
minimum of every timing field and the first run's value for everything else.

    python -m bench.merge_min run1 run2 run3 --out baseline_dir

Timing fields are those whose name ends in `_s` or `_us` or `_MB`. Taking the
minimum removes most of the scheduling noise of shared CI runners; the gate
then compares a single new run against the best of the baseline runs, so the
tolerance must cover one run's upward noise, not two runs' noise.
"""
from __future__ import annotations

import argparse
import os
import shutil

from benchmarks.harness import read_json, write_json

TIMING_SUFFIXES = ("_s", "_us", "_MB")


def _is_timing(k: str) -> bool:
    return any(k.endswith(s) for s in TIMING_SUFFIXES)


def _merge_rows(rows_list):
    out = []
    for cells in zip(*rows_list):
        merged = dict(cells[0])
        for k, v in cells[0].items():
            if _is_timing(k) and isinstance(v, (int, float)):
                vals = [c.get(k) for c in cells if isinstance(c.get(k), (int, float))]
                merged[k] = min(vals)
        out.append(merged)
    return out


def merge(dirs, out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(dirs[0]):
        if not name.endswith(".json"):
            continue
        docs = [read_json(os.path.join(d, name)) for d in dirs if os.path.exists(os.path.join(d, name))]
        base = dict(docs[0])
        for key, val in docs[0].items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                base[key] = _merge_rows([doc[key] for doc in docs])
        base["merged_from"] = len(docs)
        write_json(base, os.path.join(out, name))
    for extra in ("environment.json",):
        src = os.path.join(dirs[0], extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, extra))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    merge(a.dirs, a.out)


if __name__ == "__main__":
    main()
