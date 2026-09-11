"""Suite 3: scaling curves.

(a) fit time (min of 3) and D vs (n, d), n = 2..10, d = 2..8, N_train = 10000,
    one output, cells with D > D_CAP skipped.
(b) inference time vs D at n = 6, m = 1000 (d = 1..8).
(c) inference time vs m at n = 6, d = 5 (D = 462).
Both (b) and (c) also time the two halves of the inference path separately
(Phi build via evaluate_monomials_lazy, and the Phi @ C matmul).
"""
from __future__ import annotations

import numpy as np

from MomentEmu.emulator import evaluate_monomials_lazy

from benchmarks.harness import basis_size, fit_polyemu, fit_polyemu_best, fixed_degree_kwargs, time_call, time_inference

D_CAP = 5000
N_FIT = 10000


def smooth_target(X: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], m)) / np.sqrt(X.shape[1])
    b = rng.uniform(0, 2 * np.pi, m)
    return np.sin(X @ W + b) + 0.3 * np.cos(2.0 * (X @ W))


def fit_grid(quick: bool) -> list[dict]:
    rows = []
    ns = range(2, 11) if not quick else (2, 4, 6)
    ds = range(2, 9) if not quick else (2, 4, 6)
    rng = np.random.default_rng(7)
    Xt = rng.uniform(-1, 1, (500, 10))
    X = rng.uniform(-1, 1, (N_FIT, 10))
    # warm-up so the first cell does not pay one-off import/BLAS costs
    fit_polyemu(X[:2000, :2], smooth_target(X[:2000, :2], 1, 0), Xt[:, :2], smooth_target(Xt[:, :2], 1, 0),
                **fixed_degree_kwargs(2))
    for n in ns:
        Xn, Xtn = X[:, :n], Xt[:, :n]
        Yn, Ytn = smooth_target(Xn, 1, n), smooth_target(Xtn, 1, n)
        for d in ds:
            D = basis_size(n, d)
            row = {"n": n, "d": d, "D": D, "N": N_FIT}
            if D > D_CAP:
                row.update({"skipped": True, "fit_s": None})
            else:
                emu, t_fit, t_all = fit_polyemu_best(Xn, Yn, Xtn, Ytn, repeats=3, **fixed_degree_kwargs(d))
                row.update({"skipped": False, "fit_s": t_fit, "fit_s_all": t_all,
                            "fit_cpu_s": emu._bench_fit_cpu_s,
                            "rung_s": float(emu.forward_running_time_list[0])})
            rows.append(row)
            print(f"[scaling/fit] n={n} d={d} D={D:6d} fit={row['fit_s'] if row['fit_s'] is None else round(row['fit_s'], 3)}", flush=True)
    return rows


def _inference_cell(n: int, d: int, m: int, seed: int) -> dict:
    D = basis_size(n, d)
    N = max(2 * D, 2000)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, n))
    Xt = rng.uniform(-1, 1, (1000, n))
    emu, t_fit = fit_polyemu(X, smooth_target(X, m, seed), Xt, smooth_target(Xt, m, seed), **fixed_degree_kwargs(d))
    row = {"n": n, "d": d, "D": D, "m": m, "N": N, "fit_s": t_fit}
    row.update(time_inference(emu.forward_emulator, Xt))
    # Decompose the single-sample path.
    x1s = emu.scaler_X.transform(Xt[:1])
    mi = emu.forward_multi_indices
    C = emu.forward_coeffs
    Phi1 = evaluate_monomials_lazy(x1s, mi)
    row["phi_build_single_us"] = time_call(lambda: evaluate_monomials_lazy(x1s, mi))["min_s"] * 1e6
    row["matmul_single_us"] = time_call(lambda: Phi1 @ C)["min_s"] * 1e6
    xbs = emu.scaler_X.transform(Xt)
    Phib = evaluate_monomials_lazy(xbs, mi)
    row["phi_build_batch_us"] = time_call(lambda: evaluate_monomials_lazy(xbs, mi))["min_s"] * 1e6
    row["matmul_batch_us"] = time_call(lambda: Phib @ C)["min_s"] * 1e6
    row["phi_share_single"] = row["phi_build_single_us"] / (row["phi_build_single_us"] + row["matmul_single_us"])
    row["coeff_MB"] = C.nbytes / 1e6
    return row


def inference_vs_D(quick: bool) -> list[dict]:
    rows = []
    for d in (range(1, 9) if not quick else (1, 3, 5)):
        rows.append(_inference_cell(6, d, 1000, 11))
        r = rows[-1]
        print(f"[scaling/infer-D] D={r['D']:5d} m=1000 single={r['single_us']:8.1f}us "
              f"(phi {r['phi_build_single_us']:.1f} + matmul {r['matmul_single_us']:.1f}) "
              f"batch1000/sample={r['batch_per_sample_us']:.2f}us", flush=True)
    return rows


def inference_vs_m(quick: bool) -> list[dict]:
    rows = []
    for m in ((1, 10, 100, 1000, 10000) if not quick else (1, 1000)):
        rows.append(_inference_cell(6, 5, m, 12))
        r = rows[-1]
        print(f"[scaling/infer-m] D=462 m={m:5d} single={r['single_us']:8.1f}us "
              f"(phi {r['phi_build_single_us']:.1f} + matmul {r['matmul_single_us']:.1f}) "
              f"batch1000/sample={r['batch_per_sample_us']:.2f}us", flush=True)
    return rows


def run(quick: bool = False) -> dict:
    return {
        "suite": "scaling",
        "D_cap": D_CAP,
        "fit_grid": fit_grid(quick),
        "inference_vs_D": inference_vs_D(quick),
        "inference_vs_m": inference_vs_m(quick),
    }
