"""python -m benchmarks [--suites a,b,c] [--out DIR] [--quick] [--max-load L]

Suites: accuracy, baselines, scaling, memory, pins, noise, backends
(default: all). Writes <out>/<suite>.json, <out>/environment.json and
<out>/RESULTS.md.

Timing JSON is written only when the 1-minute load average is at or below
--max-load (default 0.5 x the CPU count). A suite that finishes on a loaded
machine is skipped (a .skipped marker is written instead), because timings
recorded under contention are not comparable to a quiet baseline.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

from benchmarks.env import environment
from benchmarks.harness import write_json
from benchmarks.report import build_report

SUITES = ["accuracy", "baselines", "scaling", "memory", "pins", "noise", "backends"]


def loadavg() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def wait_for_load(max_load: float | None, max_wait: float) -> float:
    """Block until the 1-minute load average is <= max_load; return seconds waited."""
    if max_load is None:
        return 0.0
    t0 = time.time()
    while True:
        la = loadavg()
        if la is None or la[0] <= max_load or time.time() - t0 > max_wait:
            return time.time() - t0
        print(f"    load avg {la[0]} > {max_load}; waiting ...", flush=True)
        time.sleep(10)


def main(argv=None) -> int:
    default_max_load = 0.5 * (os.cpu_count() or 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    ap.add_argument("--quick", action="store_true", help="small subset for smoke testing")
    ap.add_argument("--max-load", type=float, default=default_max_load,
                    help="wait for, and require, a 1-minute load average at or below this before "
                         "recording timing JSON (default: 0.5 x CPU count)")
    ap.add_argument("--max-wait", type=float, default=1800.0, help="seconds to wait for --max-load")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    write_json(environment(), os.path.join(a.out, "environment.json"))
    for name in a.suites.split(","):
        name = name.strip()
        if name not in SUITES:
            print(f"unknown suite {name!r}; choose from {SUITES}", file=sys.stderr)
            return 2
        mod = importlib.import_module(f"benchmarks.suite_{name}")
        waited = wait_for_load(a.max_load, a.max_wait)
        load_start = loadavg()
        t0 = time.perf_counter()
        print(f"=== suite {name} === (load avg {load_start})", flush=True)
        res = mod.run(quick=a.quick)
        res["wall_s"] = time.perf_counter() - t0
        # 1-, 5-, 15-minute load averages at suite start and end: timings recorded
        # while load_1min is well above the core count are not comparable.
        res["loadavg_start"] = load_start
        res["loadavg_end"] = loadavg()
        res["waited_for_load_s"] = waited
        res["max_load"] = a.max_load
        end_load = (res["loadavg_end"] or [float("inf")])[0]
        if a.max_load is not None and end_load > a.max_load:
            marker = os.path.join(a.out, f"{name}.skipped")
            with open(marker, "w") as f:
                f.write(
                    f"suite {name} finished at 1-min load {end_load} > max-load {a.max_load}; "
                    f"timing JSON not written.\n"
                )
            print(f"=== suite {name} done in {res['wall_s']:.1f}s; load {end_load} > {a.max_load}; "
                  f"NOT writing timing JSON -> {marker}", flush=True)
            continue
        write_json(res, os.path.join(a.out, f"{name}.json"))
        print(f"=== suite {name} done in {res['wall_s']:.1f}s -> {a.out}/{name}.json (load avg {res['loadavg_end']})", flush=True)
    md = build_report(a.out)
    with open(os.path.join(a.out, "RESULTS.md"), "w") as f:
        f.write(md)
    print(f"report -> {a.out}/RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
