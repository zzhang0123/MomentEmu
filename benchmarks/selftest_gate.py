"""Gate self-tests: prove that the gate passes on identical input and fails on
(a) a 1e-9 relative perturbation of one pin coefficient, (b) a 1e-5 relative
RMSE increase, (c) a 35 % fit-time regression, (d) a D_final change.

    python -m benchmarks.selftest_gate results
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

from benchmarks.gate import Gate, run_gate
from benchmarks.harness import read_json, write_json


def _run(base, cur):
    g = Gate(0.30, 1e-6, 1e-10, 50.0)
    with redirect_stdout(io.StringIO()):
        run_gate(base, cur, g)
    return g


def main(results_dir):
    tmp = tempfile.mkdtemp(prefix="gate_selftest_")
    cases = []
    # identical
    cur = os.path.join(tmp, "same")
    shutil.copytree(results_dir, cur)
    g = _run(results_dir, cur)
    cases.append(("identical copy", len(g.failures) == 0, g.failures))

    def perturbed(name, mutate):
        d = os.path.join(tmp, name)
        shutil.copytree(results_dir, d)
        mutate(d)
        return d

    def pin_1e9(d):
        p = read_json(os.path.join(d, "pins.json"))
        r = p["rows"][0]
        r["forward_coeffs_first10"][3] += 1e-9 * r["forward_coeff_abs_max"]
        write_json(p, os.path.join(d, "pins.json"))

    def pin_1e11(d):
        p = read_json(os.path.join(d, "pins.json"))
        r = p["rows"][0]
        r["forward_coeffs_first10"][3] += 1e-11 * r["forward_coeff_abs_max"]
        write_json(p, os.path.join(d, "pins.json"))

    def rmse_1e5(d):
        p = read_json(os.path.join(d, "accuracy.json"))
        p["rows"][0]["test_rmse"] *= 1 + 1e-5
        write_json(p, os.path.join(d, "accuracy.json"))

    def time_35pct(d):
        p = read_json(os.path.join(d, "accuracy.json"))
        p["rows"][0]["fit_s"] *= 1.35
        write_json(p, os.path.join(d, "accuracy.json"))

    def time_20pct(d):
        p = read_json(os.path.join(d, "accuracy.json"))
        p["rows"][0]["fit_s"] *= 1.20
        write_json(p, os.path.join(d, "accuracy.json"))

    def d_change(d):
        p = read_json(os.path.join(d, "accuracy.json"))
        p["rows"][0]["D_final"] += 1
        write_json(p, os.path.join(d, "accuracy.json"))

    for name, mut, expect_fail in (
        ("pin coeff +1e-9 rel", pin_1e9, True),
        ("pin coeff +1e-11 rel", pin_1e11, False),
        ("rmse +1e-5 rel", rmse_1e5, True),
        ("fit time +35%", time_35pct, True),
        ("fit time +20%", time_20pct, False),
        ("D_final +1", d_change, True),
    ):
        g = _run(results_dir, perturbed(name.replace(" ", "_").replace("%", "pct").replace("+", "p"), mut))
        ok = (len(g.failures) > 0) == expect_fail
        cases.append((name, ok, g.failures))
    all_ok = True
    for name, ok, failures in cases:
        print(f"{'OK  ' if ok else 'BAD '} {name:28s} -> {len(failures)} failure(s)" + (f": {failures[0]}" if failures else ""))
        all_ok &= ok
    shutil.rmtree(tmp)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
