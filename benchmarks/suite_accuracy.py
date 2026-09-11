"""Suite 1: standard test functions, default settings and fixed degree."""
from __future__ import annotations

import time

from benchmarks.harness import (
    accuracy,
    basis_size,
    cond_M,
    emulator_summary,
    fit_polyemu_best,
    fixed_degree_kwargs,
    time_inference,
)

FIT_REPEATS = 3
from benchmarks.targets import STANDARD_ORDER, TARGETS


def _one_fit(t, X, Y, Xt, Yt, kw: dict, label: str) -> dict:
    emu, t_fit, t_all = fit_polyemu_best(X, Y, Xt, Yt, repeats=FIT_REPEATS, **kw)
    t0 = time.perf_counter()
    pred = emu.forward_emulator(Xt)
    t_pred_full = time.perf_counter() - t0
    row = {
        "target": t.name, "mode": label, "n": t.n, "m": t.m,
        "n_train": t.n_train, "n_test": t.n_test,
        "fit_s": t_fit,
        "fit_s_all": t_all,
        "fit_cpu_s": emu._bench_fit_cpu_s,
        "fit_cpu_s_all": emu._bench_fit_cpu_s_all,
        "predict_test_set_s": t_pred_full,
    }
    row.update(emulator_summary(emu))
    row["D_nominal"] = basis_size(t.n, row["degree"]) if row["degree"] >= 0 else None
    swept = row["degrees_swept"]
    row["last_rung_D"] = basis_size(t.n, swept[-1]) if swept else None
    row["sweep_hit_D_ge_N"] = bool(swept) and row["last_rung_D"] >= t.n_train
    row["worst_rung_rmse_scaled"] = max(row["rmse_val_scaled"]) if row["rmse_val_scaled"] else None
    row.update({f"test_{k}": v for k, v in accuracy(pred, Yt).items()})
    row.update(cond_M(emu, X))
    row.update({f"infer_{k}": v for k, v in time_inference(emu.forward_emulator, Xt).items()})
    return row


def run(quick: bool = False) -> dict:
    rows = []
    names = STANDARD_ORDER if not quick else ["ishigami", "rosenbrock", "cmb_like"]
    for name in names:
        t = TARGETS[name]
        X, Y, Xt, Yt = t.data()
        print(f"[accuracy] {name}: n={t.n} m={t.m} N={t.n_train} ...", flush=True)
        rows.append(_one_fit(t, X, Y, Xt, Yt, {}, "default"))
        print(f"    default : deg={rows[-1]['degree']} D={rows[-1]['D_final']} "
              f"fit={rows[-1]['fit_s']:.2f}s nrmse={rows[-1]['test_nrmse']:.3e} "
              f"sa_max_rel={rows[-1]['test_sa_max_rel']:.3e} cond={rows[-1]['cond_M']:.2e}", flush=True)
        rows.append(_one_fit(t, X, Y, Xt, Yt, fixed_degree_kwargs(t.fixed_degree), f"fixed_d{t.fixed_degree}"))
        print(f"    fixed d={t.fixed_degree}: D={rows[-1]['D_final']} fit={rows[-1]['fit_s']:.2f}s "
              f"nrmse={rows[-1]['test_nrmse']:.3e} sa_max_rel={rows[-1]['test_sa_max_rel']:.3e} "
              f"cond={rows[-1]['cond_M']:.2e} single={rows[-1]['infer_single_us']:.0f}us", flush=True)
    return {"suite": "accuracy", "rows": rows}
