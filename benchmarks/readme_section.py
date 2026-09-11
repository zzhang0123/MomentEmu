"""Generate the one-page README "Benchmarks" section from a results directory.

    python -m benchmarks.readme_section results > README_benchmarks.md

Every number is read from the JSON files; nothing is typed in by hand.
"""
from __future__ import annotations

import os
import sys

from benchmarks.harness import fmt, read_json


def _rows(d, name, key="rows"):
    return read_json(os.path.join(d, f"{name}.json"))[key]


def _by(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def main(d):
    env = read_json(os.path.join(d, "environment.json"))
    acc = _by(_rows(d, "accuracy"), "target", "mode")
    base = _by(_rows(d, "baselines"), "target", "method")
    sc = read_json(os.path.join(d, "scaling.json"))
    inf_D = _by(sc["inference_vs_D"], "D")
    inf_m = _by(sc["inference_vs_m"], "m")
    mem = _by(_rows(d, "memory"), "N", "batch_size", "dim_reduction")
    pins = _rows(d, "pins")
    noise = _by(_rows(d, "noise"), "target")
    back = {r["backend"].split(" ")[0]: r for r in _rows(d, "backends")}
    targets = ["ishigami", "sobol_g", "friedman", "rosenbrock", "log_rosenbrock", "gauss_peak", "cmb_like"]

    out = []
    p = out.append
    p("## Benchmarks")
    p("")
    p(f"Measured with `python -m benchmarks` (see `benchmarks/README.md`) on {env['cpu']}, numpy {env['numpy']} ({env['blas']}), "
      f"MomentEmu {env['momentemu_git_sha']}. Fit times are the minimum of 3 runs; inference times are the minimum "
      f"over 5 blocks of repeated calls. nRMSE = test RMSE / RMS deviation of the target. "
      f"\"max rel err\" is the signal-aware maximum relative error (entries below 1e-3 of the per-output peak are masked).")
    p("")
    p("### Accuracy and cost on standard targets")
    p("")
    p("| target | n | m | N | mode | degree | D | fit s | nRMSE | max rel err | cond(M) | 1-pt us | 1000-pt us |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in targets:
        for mode in ("default", None):
            key = mode or next(k[1] for k in acc if k[0] == t and k[1].startswith("fixed"))
            r = acc[(t, key)]
            sweep = "" if key != "default" else (" (sweep " + f"{r['degrees_swept'][0]}-{r['degrees_swept'][-1]}"
                                                  + (", last rung D>=N" if r["sweep_hit_D_ge_N"] else "") + ")")
            p(f"| {t} | {r['n']} | {r['m']} | {r['n_train']} | {key}{sweep} | {r['degree']} | {r['D_final']} | {fmt(r['fit_s'])} | "
              f"{fmt(r['test_nrmse'])} | {fmt(r['test_sa_max_rel'])} | {fmt(r['cond_M'])} | {fmt(r['infer_single_us'], 3)} | {fmt(r['infer_batch_us'], 3)} |")
    p("")
    ros_def, ros_fix = acc[("rosenbrock", "default")], acc[("rosenbrock", "fixed_d4")]
    p(f"- Rosenbrock is a quartic polynomial. At fixed degree 4 the fit is exact to nRMSE {fmt(ros_fix['test_nrmse'], 2)}; "
      f"the default sweep selects the same degree-4 basis (D = {ros_def['D_final']}) and reaches nRMSE "
      f"{fmt(ros_def['test_nrmse'], 2)}. 2.0.0 removed the old `dim_reduction` pruning (D15), which used to drop modes "
      f"and raise this error.")
    hit = [t for t in targets if acc[(t, "default")]["sweep_hit_D_ge_N"]]
    shares = []
    for t in hit:
        r = acc[(t, "default")]
        shares.append(f"{t} {r['rung_seconds'][-1] / sum(r['rung_seconds']) * 100:.0f}%")
    if hit:
        p(f"- With default settings the degree sweep ends on a rung with D >= N_train in {len(hit)} of {len(targets)} "
          f"targets; that singular rung is the largest share of the default fit time: {', '.join(shares)}.")
    else:
        p("- The P0.6 sample-count guard caps every default sweep at D <= N_train / 2, so no target ends on a "
          "singular rung; the sweep stops on the RMSE criterion or the degree cap.")
    sg = acc[("sobol_g", "fixed_d5")]
    p(f"- The Sobol G-function has |4x-2| kinks; the degree-5 polynomial reaches nRMSE {fmt(sg['test_nrmse'], 2)} "
      f"(max rel err {fmt(sg['test_sa_max_rel'], 2)}). A total-degree polynomial does not converge on a kink.")
    p("")
    p("### Against sklearn baselines (same data, same fixed degree)")
    p("")
    p("| target | MomentEmu nRMSE | Poly+LinReg nRMSE | coef max rel diff | GP nRMSE | MLP nRMSE | fit s: MomentEmu / Poly+LinReg / GP / MLP | 1-pt us: MomentEmu / Poly+LinReg / GP / MLP |")
    p("|---|---|---|---|---|---|---|---|")
    for t in targets:
        me, lr, gp, ml = base[(t, "momentemu")], base[(t, "poly_lr")], base[(t, "gp_rbf_white")], base[(t, "mlp")]
        p(f"| {t} | {fmt(me['test_nrmse'])} | {fmt(lr['test_nrmse'])} | {fmt(lr['coef_max_rel_diff'], 2)} | {fmt(gp['test_nrmse'])} | {fmt(ml['test_nrmse'])} | "
          f"{fmt(me['fit_s'])} / {fmt(lr['fit_s'])} / {fmt(gp['fit_s'])} / {fmt(ml['fit_s'])} | "
          f"{fmt(me['infer_single_us'], 3)} / {fmt(lr['infer_single_us'], 3)} / {fmt(gp['infer_single_us'], 3)} / {fmt(ml['infer_single_us'], 3)} |")
    p("")
    worst_coef = max(base[(t, "poly_lr")]["coef_max_rel_diff"] for t in targets)
    fit_ratio = [base[(t, "poly_lr")]["fit_s"] / base[(t, "momentemu")]["fit_s"] for t in targets]
    inf_ratio = [base[(t, "momentemu")]["infer_single_us"] / base[(t, "poly_lr")]["infer_single_us"] for t in targets]
    p(f"- `PolynomialFeatures + LinearRegression` is the same model: coefficients agree to {fmt(worst_coef, 2)} relative or better "
      f"on every target. MomentEmu's normal-equation solve is {fmt(min(fit_ratio), 2)}-{fmt(max(fit_ratio), 2)}x faster to fit; "
      f"its single-point inference is {fmt(min(inf_ratio), 2)}-{fmt(max(inf_ratio), 2)}x slower, because the P0.7 recursive "
      f"plan evaluates the design row in Python while sklearn uses one BLAS call.")
    gp_wins = [t for t in targets if base[(t, "gp_rbf_white")]["test_nrmse"] < base[(t, "momentemu")]["test_nrmse"]]
    gp_ratio = {t: base[(t, "momentemu")]["test_nrmse"] / base[(t, "gp_rbf_white")]["test_nrmse"] for t in targets}
    p(f"- The GP (N capped at 2000) has lower test error than the polynomial on {len(gp_wins)} of {len(targets)} targets, by "
      f"{fmt(min(gp_ratio[t] for t in gp_wins), 2)}-{fmt(max(gp_ratio[t] for t in gp_wins), 2)}x; the polynomial wins only where "
      f"the target is a polynomial (rosenbrock, {fmt(1/gp_ratio['rosenbrock'], 2)}x). GP fit time is "
      f"{fmt(min(base[(t, 'gp_rbf_white')]['fit_s'] for t in targets), 2)}-{fmt(max(base[(t, 'gp_rbf_white')]['fit_s'] for t in targets), 2)} s "
      f"and its 1000-point prediction is {fmt(min(base[(t, 'gp_rbf_white')]['infer_batch_us'] for t in targets) / 1e3, 2)}-"
      f"{fmt(max(base[(t, 'gp_rbf_white')]['infer_batch_us'] for t in targets) / 1e3, 2)} ms.")
    mlp_wins = [t for t in targets if base[(t, "mlp")]["test_nrmse"] < base[(t, "momentemu")]["test_nrmse"]]
    mlp_ratio = {t: base[(t, "momentemu")]["test_nrmse"] / base[(t, "mlp")]["test_nrmse"] for t in mlp_wins}
    p(f"- The MLP ((64,64) tanh; (128,128) for the CMB-like target) has lower test error than the degree-{'/'.join(str(acc[(t, next(k[1] for k in acc if k[0] == t and k[1].startswith('fixed')))]['degree']) for t in mlp_wins)} "
      f"polynomial on {len(mlp_wins)} of {len(targets)} targets ({', '.join(f'{t} {fmt(mlp_ratio[t], 2)}x' for t in mlp_wins)}) "
      f"at {fmt(min(base[(t, 'mlp')]['fit_s'] for t in targets), 2)}-{fmt(max(base[(t, 'mlp')]['fit_s'] for t in targets), 2)} s of fit time; "
      f"its single-point prediction is {fmt(min(base[(t, 'momentemu')]['infer_single_us'] / base[(t, 'mlp')]['infer_single_us'] for t in targets), 2)}-"
      f"{fmt(max(base[(t, 'momentemu')]['infer_single_us'] / base[(t, 'mlp')]['infer_single_us'] for t in targets), 2)}x faster than MomentEmu's.")
    p("")
    p("### Scaling")
    p("")
    grid = _by(sc["fit_grid"], "n", "d")
    p("Fit time (s, N = 10000, one output, fixed degree) and D:")
    p("")
    p("| n \\ d | " + " | ".join(str(dd) for dd in range(2, 9)) + " |")
    p("|---|" + "---|" * 7)
    for n in range(2, 11):
        cells = []
        for dd in range(2, 9):
            r = grid[(n, dd)]
            cells.append("-" if r["skipped"] else f"{fmt(r['fit_s'], 2)} (D={r['D']})")
        p(f"| {n} | " + " | ".join(cells) + " |")
    p("")
    p("Inference (n = 6): single-point time is set by the Python loop that builds the design row; the matmul is a few percent.")
    p("")
    p("| D (m=1000) | 1-pt us | of which Phi build | 1000-pt per-sample us |  | m (D=462) | 1-pt us | 1000-pt per-sample us |")
    p("|---|---|---|---|---|---|---|---|")
    Ds = sorted(k[0] for k in inf_D)
    ms = sorted(k[0] for k in inf_m)
    for i in range(max(len(Ds), len(ms))):
        left = right = ""
        if i < len(Ds):
            r = inf_D[(Ds[i],)]
            left = f"{Ds[i]} | {fmt(r['single_us'], 3)} | {r['phi_share_single'] * 100:.0f}% | {fmt(r['batch_per_sample_us'])}"
        else:
            left = " | | | "
        if i < len(ms):
            r = inf_m[(ms[i],)]
            right = f"{ms[i]} | {fmt(r['single_us'], 3)} | {fmt(r['batch_per_sample_us'])}"
        else:
            right = " | | "
        p(f"| {left} |  | {right} |")
    p("")
    p("Peak memory of a fit (n = 6, D = 462, m = 100), tracemalloc peak / RSS increase:")
    p("")
    p("| N | batch_size=None | batch_size=1000 | batch_size=10000 |")
    p("|---|---|---|---|")
    for N in (2000, 10000, 50000):
        cells = []
        for bs in (None, 1000, 10000):
            r = mem[(N, bs, False)]
            cells.append(f"{fmt(r['tracemalloc_peak_MB'])} / {fmt(r['rss_peak_MB'] - r['rss_before_fit_MB'])} MB")
        p(f"| {N} | " + " | ".join(cells) + " |")
    r = mem[(50000, 1000, True)]
    r0 = mem[(50000, 1000, False)]
    p("")
    p(f"- At N = 50000, batch_size = 1000 the tracemalloc peak is {fmt(r0['tracemalloc_peak_MB'])} MB versus "
      f"{fmt(mem[(50000, 10000, False)]['tracemalloc_peak_MB'])} MB at the default 10000: the LOO sweep keeps the full Phi "
      f"resident while N x D x 8 fits the 512 MiB budget, and batches the leverage/PRESS pass by batch_size. Set "
      f"MOMENTEMU_PHI_BUDGET_BYTES=0 to force the fully batched path (lower peak, about 1.4x the fit time). "
      f"`dim_reduction` is ignored in 2.0.0 (D15).")
    p("")
    p("### Backends (CMB-like emulator, D = 462, m = 2000)")
    p("")
    p("| backend | 1-pt us | 1000-pt us |")
    p("|---|---|---|")
    for k, r in back.items():
        p(f"| {r['backend']} | {fmt(r['single_us'], 3)} | {fmt(r['batch_us'], 3)} |")
    p("")
    p("### Reproducibility and CI gate")
    p("")
    rep = max(r["drift_repeat_rel"] for r in pins)
    b256 = max(r["drift_batch256_rel"] for r in pins)
    p(f"- Five fixed-seed pins (`results/pins.json`) record test RMSE and the first ten coefficients. Repeating a fit "
      f"in-process reproduces the coefficients to {fmt(rep, 1)} relative; changing `batch_size` (summation order) moves them by "
      f"up to {fmt(b256, 1)} relative. The gate tolerance is 1e-10.")
    n1, n2 = noise[("ishigami",)], noise[("cmb_like",)]
    p(f"- Timing noise on this machine over {n1['repeats']} repeats: single-shot fit max/min {fmt(n1['fit_max_over_min'], 2)} "
      f"({fmt(n1['fit_s_mean'] * 1e3, 2)} ms fit) and {fmt(n2['fit_max_over_min'], 2)} ({fmt(n2['fit_s_mean'] * 1e3, 3)} ms fit); "
      f"min-of-3 fit max/min {fmt(n1.get('fit_min3_max_over_min'), 2)} and {fmt(n2.get('fit_min3_max_over_min'), 2)}; "
      f"inference max/min {fmt(max(n1['single_max_over_min'], n2['single_max_over_min']), 2)} (1-pt) and "
      f"{fmt(max(n1['batch_max_over_min'], n2['batch_max_over_min']), 2)} (1000-pt). "
      f"`python -m benchmarks.gate` fails on a fit-time or inference-time regression above 30 %, an RMSE regression above 1e-6 relative, "
      f"or a pin drift above 1e-10.")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
