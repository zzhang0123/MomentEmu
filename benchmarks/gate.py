"""Performance / accuracy regression gate.

    python -m bench.gate --baseline results_baseline --current results \
        [--time-tol 0.30] [--rmse-tol 1e-6] [--pin-tol 1e-10] [--time-floor-us 50]

Exit status 0 when every check passes, 1 otherwise. Prints one line per check.

Checks
- accuracy suite : per (target, mode): fit_s, infer_single_us, infer_batch_us
                   may not exceed baseline * (1 + time_tol); test_rmse may not
                   exceed baseline * (1 + rmse_tol); D_final and degree must match.
- scaling suite  : fit_grid fit_s and inference single/batch per cell, time_tol.
- pins suite     : test_rmse and the first-ten coefficients within pin_tol
                   (relative to max|coeff| of the pin); D must match.
- memory suite   : rss_peak_MB may not exceed baseline * (1 + time_tol).

Timings below `time_floor_us` are compared with an absolute slack of
`time_floor_us` instead of the relative tolerance (sub-50-us timings are
dominated by interpreter jitter).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from benchmarks.harness import read_json


class Gate:
    def __init__(self, time_tol, rmse_tol, pin_tol, time_floor_us):
        self.time_tol, self.rmse_tol, self.pin_tol = time_tol, rmse_tol, pin_tol
        self.time_floor_s = time_floor_us * 1e-6
        self.failures: list[str] = []
        self.checks = 0

    def _check(self, ok: bool, msg: str):
        self.checks += 1
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            self.failures.append(msg)

    def time(self, label, base, cur, unit_s=1.0):
        if base is None or cur is None:
            return
        b, c = base * unit_s, cur * unit_s
        if b < self.time_floor_s:
            ok = c <= b + self.time_floor_s
            self._check(ok, f"{label}: {c/unit_s:.4g} vs {b/unit_s:.4g} (+{self.time_floor_s/unit_s:.3g} abs slack)")
        else:
            ok = c <= b * (1 + self.time_tol)
            self._check(ok, f"{label}: {c/unit_s:.4g} vs {b/unit_s:.4g} ({(c/b-1)*100:+.1f}%, tol {self.time_tol*100:.0f}%)")

    def rmse(self, label, base, cur):
        if not (math.isfinite(base) and math.isfinite(cur)):
            self._check(math.isfinite(cur) == math.isfinite(base), f"{label}: finiteness {cur} vs {base}")
            return
        ok = cur <= base * (1 + self.rmse_tol)
        self._check(ok, f"{label}: {cur:.10e} vs {base:.10e} (rel {cur/base-1:+.2e}, tol {self.rmse_tol:.0e})")

    def equal(self, label, base, cur):
        self._check(base == cur, f"{label}: {cur} vs {base}")

    def pin(self, label, base, cur, scale):
        diff = max(abs(a - b) for a, b in zip(base, cur)) if base and cur else float("inf")
        ok = diff <= self.pin_tol * scale and len(base) == len(cur)
        self._check(ok, f"{label}: max|dc|/scale = {diff/scale:.2e} (tol {self.pin_tol:.0e})")


def _index(rows, keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def run_gate(base_dir, cur_dir, gate: Gate):
    def load(name):
        pb, pc = os.path.join(base_dir, f"{name}.json"), os.path.join(cur_dir, f"{name}.json")
        if not (os.path.exists(pb) and os.path.exists(pc)):
            print(f"SKIP {name}: missing {pb if not os.path.exists(pb) else pc}")
            return None, None
        return read_json(pb), read_json(pc)

    # Contention check: a 1-minute load average above the core count on either
    # side means the timing checks below compare unlike conditions.
    for name in ("accuracy", "scaling", "memory", "noise"):
        bb, cc = load(name)
        for side, doc in (("baseline", bb), ("current", cc)):
            la = (doc or {}).get("loadavg_start")
            if la and la[0] > (os.cpu_count() or 1):
                print(f"WARN {name} {side}: 1-min load average {la[0]} exceeds core count {os.cpu_count()}; "
                      f"timing checks are not meaningful")

    b, c = load("accuracy")
    if b:
        bi, ci = _index(b["rows"], ("target", "mode")), _index(c["rows"], ("target", "mode"))
        for k, br in bi.items():
            cr = ci.get(k)
            if cr is None:
                gate._check(False, f"accuracy {k}: missing in current")
                continue
            gate.equal(f"accuracy {k} degree", br["degree"], cr["degree"])
            gate.equal(f"accuracy {k} D_final", br["D_final"], cr["D_final"])
            gate.rmse(f"accuracy {k} test_rmse", br["test_rmse"], cr["test_rmse"])
            gate.time(f"accuracy {k} fit_s", br["fit_s"], cr["fit_s"])
            gate.time(f"accuracy {k} infer_single_us", br["infer_single_us"], cr["infer_single_us"], 1e-6)
            gate.time(f"accuracy {k} infer_batch_us", br["infer_batch_us"], cr["infer_batch_us"], 1e-6)

    b, c = load("scaling")
    if b:
        bi, ci = _index(b["fit_grid"], ("n", "d")), _index(c["fit_grid"], ("n", "d"))
        for k, br in bi.items():
            cr = ci.get(k)
            if cr and not br.get("skipped"):
                gate.time(f"scaling fit n={k[0]} d={k[1]}", br["fit_s"], cr["fit_s"])
        for key in ("inference_vs_D", "inference_vs_m"):
            bi, ci = _index(b[key], ("D", "m")), _index(c[key], ("D", "m"))
            for k, br in bi.items():
                cr = ci.get(k)
                if cr:
                    gate.time(f"{key} D={k[0]} m={k[1]} single_us", br["single_us"], cr["single_us"], 1e-6)
                    gate.time(f"{key} D={k[0]} m={k[1]} batch_us", br["batch_us"], cr["batch_us"], 1e-6)

    b, c = load("pins")
    if b:
        bi, ci = _index(b["rows"], ("id",)), _index(c["rows"], ("id",))
        for k, br in bi.items():
            cr = ci.get(k)
            if cr is None:
                gate._check(False, f"pin {k[0]}: missing in current")
                continue
            gate.equal(f"pin {k[0]} D", br["D"], cr["D"])
            rel = abs(cr["test_rmse"] - br["test_rmse"]) / max(br["test_rmse"], 1e-300)
            gate._check(rel <= gate.pin_tol, f"pin {k[0]} test_rmse rel diff {rel:.2e} (tol {gate.pin_tol:.0e})")
            gate.pin(f"pin {k[0]} forward_coeffs_first10", br["forward_coeffs_first10"], cr["forward_coeffs_first10"], br["forward_coeff_abs_max"])
            if "backward_coeffs_first10" in br:
                gate.pin(f"pin {k[0]} backward_coeffs_first10", br["backward_coeffs_first10"], cr.get("backward_coeffs_first10", []),
                         max(abs(v) for v in br["backward_coeffs_first10"]))

    b, c = load("memory")
    if b:
        bi, ci = _index(b["rows"], ("N", "batch_size", "dim_reduction")), _index(c["rows"], ("N", "batch_size", "dim_reduction"))
        for k, br in bi.items():
            cr = ci.get(k)
            if cr:
                gate.time(f"memory N={k[0]} batch={k[1]} dimred={k[2]} rss_peak_MB", br["rss_peak_MB"], cr["rss_peak_MB"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--time-tol", type=float, default=0.30)
    ap.add_argument("--rmse-tol", type=float, default=1e-6)
    ap.add_argument("--pin-tol", type=float, default=1e-10)
    ap.add_argument("--time-floor-us", type=float, default=50.0)
    a = ap.parse_args(argv)
    gate = Gate(a.time_tol, a.rmse_tol, a.pin_tol, a.time_floor_us)
    run_gate(a.baseline, a.current, gate)
    print(f"\n{gate.checks - len(gate.failures)}/{gate.checks} checks passed")
    return 1 if gate.failures else 0


if __name__ == "__main__":
    sys.exit(main())
