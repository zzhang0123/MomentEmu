"""Turn the JSON result files of one run into a markdown report."""
from __future__ import annotations

import os

from benchmarks.harness import fmt, read_json


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _load(d, name):
    p = os.path.join(d, f"{name}.json")
    return read_json(p) if os.path.exists(p) else None


def accuracy_md(res) -> str:
    rows = []
    for r in res["rows"]:
        rows.append([
            r["target"], r["mode"], r["n"], r["m"], r["n_train"], r["degree"], r["D_final"],
            fmt(r["fit_s"]), fmt(r["test_rmse"]), fmt(r["test_nrmse"]), fmt(r["test_sa_max_rel"]),
            fmt(r["cond_M"]), fmt(r["infer_single_us"], 3), fmt(r["infer_batch_us"], 3),
            (f"{r['degrees_swept'][0]}-{r['degrees_swept'][-1]}" if len(r["degrees_swept"]) > 1 else str(r["degrees_swept"][0]))
            + (" (last rung D>=N)" if r.get("sweep_hit_D_ge_N") else ""),
        ])
    return _table(["target", "mode", "n", "m", "N", "deg", "D", "fit s", "test RMSE", "nRMSE",
                   "max rel err (SA)", "cond(M)", "1-pt us", "1000-pt us", "degrees swept"], rows)


def baselines_md(res) -> str:
    rows = []
    for r in res["rows"]:
        extra = ""
        if r["method"] == "poly_lr":
            extra = f"coef max rel diff {fmt(r['coef_max_rel_diff'], 2)}; pred diff {fmt(r['pred_max_rel_diff_vs_momentemu'], 2)}"
        else:
            extra = r.get("note", "")
        rows.append([r["target"], r["method"], fmt(r["fit_s"]), fmt(r["test_rmse"]), fmt(r["test_nrmse"]),
                     fmt(r["test_sa_max_rel"]), fmt(r["infer_single_us"], 3), fmt(r["infer_batch_us"], 3), extra])
    return _table(["target", "method", "fit s", "test RMSE", "nRMSE", "max rel err (SA)", "1-pt us", "1000-pt us", "note"], rows)


def scaling_md(res) -> str:
    grid = res["fit_grid"]
    ns = sorted({r["n"] for r in grid})
    ds = sorted({r["d"] for r in grid})
    by = {(r["n"], r["d"]): r for r in grid}
    rows_t = []
    rows_D = []
    for n in ns:
        rows_t.append([n] + [("-" if by[(n, d)]["skipped"] else fmt(by[(n, d)]["fit_s"])) for d in ds])
        rows_D.append([n] + [by[(n, d)]["D"] for d in ds])
    parts = [
        f"Fit time in seconds, min of 3 (N = {grid[0]['N']}, single output, fixed degree, dim_reduction off; cells with D > {res['D_cap']} skipped):",
        _table(["n \\ d"] + [str(d) for d in ds], rows_t),
        "",
        "Basis size D = C(n+d, d):",
        _table(["n \\ d"] + [str(d) for d in ds], rows_D),
        "",
        "Inference vs D (n = 6, m = 1000). Phi share = fraction of the single-point cost spent building the design row:",
        _table(["D", "1-pt us", "Phi build us", "matmul us", "Phi share", "1000-pt us", "per-sample us", "coeff MB"],
               [[r["D"], fmt(r["single_us"]), fmt(r["phi_build_single_us"]), fmt(r["matmul_single_us"]),
                 fmt(r["phi_share_single"], 2), fmt(r["batch_us"]), fmt(r["batch_per_sample_us"]), fmt(r["coeff_MB"], 2)]
                for r in res["inference_vs_D"]]),
        "",
        "Inference vs m (n = 6, d = 5, D = 462):",
        _table(["m", "1-pt us", "Phi build us", "matmul us", "Phi share", "1000-pt us", "per-sample us", "coeff MB"],
               [[r["m"], fmt(r["single_us"]), fmt(r["phi_build_single_us"]), fmt(r["matmul_single_us"]),
                 fmt(r["phi_share_single"], 2), fmt(r["batch_us"]), fmt(r["batch_per_sample_us"]), fmt(r["coeff_MB"], 2)]
                for r in res["inference_vs_m"]]),
    ]
    return "\n".join(parts)


def memory_md(res) -> str:
    rows = [[r["N"], "None" if r["batch_size"] is None else r["batch_size"], "on" if r["dim_reduction"] else "off",
             fmt(r["phi_full_MB"]), fmt(r["tracemalloc_peak_MB"]), fmt(r["rss_peak_MB"]),
             fmt(r["rss_peak_MB"] - r["rss_before_fit_MB"]), fmt(r["fit_s"])]
            for r in res["rows"]]
    return _table(["N", "batch_size", "dim_reduction", "full Phi MB", "tracemalloc peak MB", "RSS peak MB",
                   "RSS peak - RSS before fit MB", "fit s"], rows)


def pins_md(res) -> str:
    rows = [[r["id"], r["D"], r["m"], f"{r['test_rmse']:.12e}", f"{r['forward_coeffs_first10'][0]:+.17e}",
             fmt(r["cond_M"]), fmt(r["drift_repeat_rel"], 2), fmt(r["drift_batch256_rel"], 2)]
            for r in res["rows"]]
    return _table(["pin", "D", "m", "test RMSE", "c[0,0]", "cond(M)", "drift repeat", "drift batch=256"], rows)


def noise_md(res) -> str:
    rows = [[r["target"], r["repeats"], fmt(r["fit_s_mean"]), f"{r['fit_s_cv']*100:.1f}%", fmt(r["fit_max_over_min"]),
             f"{r.get('fit_min3_cv', float('nan'))*100:.1f}%", fmt(r.get("fit_min3_max_over_min")),
             fmt(r["single_us_mean"]), f"{r['single_us_cv']*100:.1f}%", fmt(r["single_max_over_min"]),
             fmt(r["batch_us_mean"]), f"{r['batch_us_cv']*100:.1f}%", fmt(r["batch_max_over_min"])]
            for r in res["rows"]]
    return _table(["target", "repeats", "fit s", "fit CV", "fit max/min", "min-of-3 fit CV", "min-of-3 fit max/min",
                   "1-pt us", "1-pt CV", "1-pt max/min",
                   "1000-pt us", "1000-pt CV", "1000-pt max/min"], rows)


def backends_md(res) -> str:
    rows = [[r["backend"], fmt(r["single_us"], 3), fmt(r["batch_us"], 3), fmt(r["batch_per_sample_us"]),
             fmt(r.get("max_rel_diff_vs_numpy"), 2), r.get("note", "")] for r in res["rows"]]
    return _table(["backend", "1-pt us", "1000-pt us", "per-sample us", "max rel diff vs numpy", "note"], rows)


def build_report(results_dir: str) -> str:
    env = _load(results_dir, "environment") or {}
    parts = ["# MomentEmu benchmark results", ""]
    if env:
        parts += [f"Environment: {env.get('cpu')} ({env.get('n_cpu')} cores), {env.get('platform')}, "
                  f"Python {env.get('python')}, numpy {env.get('numpy')} ({env.get('blas')}), scipy {env.get('scipy')}, "
                  f"sklearn {env.get('sklearn')}; MomentEmu {env.get('momentemu_git_sha')}; {env.get('timestamp_utc')}; "
                  f"load average at start {env.get('loadavg_at_start')}", ""]
    for name, fn, title in (
        ("accuracy", accuracy_md, "1. Standard test functions"),
        ("baselines", baselines_md, "2. Baselines on the same data"),
        ("scaling", scaling_md, "3. Scaling curves"),
        ("memory", memory_md, "3d. Peak memory of a fit vs N and batch_size (n=6, d=5, D=462, m=100)"),
        ("pins", pins_md, "4. Regression pins"),
        ("noise", noise_md, "5. Timing noise on this machine"),
        ("backends", backends_md, "6. Autodiff backends, single-point and batch inference on the CMB-like emulator"),
    ):
        res = _load(results_dir, name)
        if res:
            load = ""
            if res.get("loadavg_start") is not None:
                load = (f"Load average (1/5/15 min) at suite start {res['loadavg_start']}, at end {res['loadavg_end']}; "
                        f"suite wall time {res.get('wall_s', float('nan')):.0f} s.")
            parts += [f"## {title}", ""] + ([load, ""] if load else []) + [fn(res), ""]
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    print(build_report(sys.argv[1]))
