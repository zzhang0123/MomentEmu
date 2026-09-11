"""Suite 4: regression pins.

Five fixed-seed configurations. Each records the test RMSE and the first ten
forward coefficients (and backward coefficients where a backward emulator is
built) at full float64 precision. gate.py compares them at 1e-10 relative.

The suite also measures how large a coefficient change is produced by
(a) repeating the fit in-process, (b) changing batch_size (which changes the
summation order in the moment accumulation), so that the 1e-10 tolerance can
be judged against the drift the code itself produces.
"""
from __future__ import annotations

import numpy as np

from benchmarks.harness import accuracy, cond_M, fit_polyemu, fixed_degree_kwargs
from benchmarks.targets import TARGETS, cmb_like, friedman, ishigami, sobol_g


def _map2(X):
    """Smooth invertible 2 -> 2 map for a forward+backward pin."""
    x, y = X[:, 0], X[:, 1]
    return np.stack([x + 0.3 * np.sin(y), y + 0.3 * np.cos(x)], axis=1)


PINS = [
    dict(id="pin1_ishigami_d6", f=ishigami, n=3, lo=-np.pi, hi=np.pi, d=6, N=1000, seed=1, backward=False),
    dict(id="pin2_sobolg_d3", f=sobol_g, n=8, lo=0.0, hi=1.0, d=3, N=1500, seed=2, backward=False),
    dict(id="pin3_friedman_d3", f=friedman, n=10, lo=0.0, hi=1.0, d=3, N=1500, seed=3, backward=False),
    dict(id="pin4_cmb_d4", f=None, n=6, lo=None, hi=None, d=4, N=2000, seed=4, backward=False),
    dict(id="pin5_map2_fwd_bwd_d6", f=_map2, n=2, lo=-1.0, hi=1.0, d=6, N=1500, seed=5, backward=True),
]


def _data(p):
    rng = np.random.default_rng(p["seed"])
    if p["id"].startswith("pin4"):
        t = TARGETS["cmb_like"]
        X = t.sample(p["N"], rng)
        Xt = t.sample(500, rng)
        return X, cmb_like(X), Xt, cmb_like(Xt)
    X = rng.uniform(p["lo"], p["hi"], (p["N"], p["n"]))
    Xt = rng.uniform(p["lo"], p["hi"], (500, p["n"]))
    return X, p["f"](X), Xt, p["f"](Xt)


def _kwargs(p, batch_size=None):
    kw = fixed_degree_kwargs(p["d"])
    if p["backward"]:
        kw.update(backward=True, init_deg_backward=p["d"], max_degree_backward=p["d"])
    kw["batch_size"] = batch_size
    return kw


def _pin(p, emu, Xt, Yt) -> dict:
    c = emu.forward_coeffs
    out = {
        "id": p["id"], "degree": p["d"], "N": p["N"], "seed": p["seed"],
        "D": int(len(emu.forward_multi_indices)), "m": int(c.shape[1]),
        "test_rmse": accuracy(emu.forward_emulator(Xt), Yt)["rmse"],
        "forward_coeffs_first10": [float(v) for v in c[:10, 0]],
        "forward_coeff_sum": float(c.sum()),
        "forward_coeff_abs_max": float(np.abs(c).max()),
    }
    if p["backward"]:
        cb = emu.backward_coeffs
        out["backward_coeffs_first10"] = [float(v) for v in cb[:10, 0]]
        out["backward_coeff_sum"] = float(cb.sum())
        out["backward_test_rmse"] = accuracy(emu.backward_emulator(Yt), Xt)["rmse"]
    return out


def _rel_drift(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


def run(quick: bool = False) -> dict:
    rows = []
    for p in PINS:
        X, Y, Xt, Yt = _data(p)
        emu, t_fit = fit_polyemu(X, Y, Xt, Yt, **_kwargs(p))
        row = _pin(p, emu, Xt, Yt)
        row["fit_s"] = t_fit
        row.update({k: v for k, v in cond_M(emu, X).items() if k != "D"})
        # (a) repeat in-process
        emu2, _ = fit_polyemu(X, Y, Xt, Yt, **_kwargs(p))
        row["drift_repeat_rel"] = _rel_drift(emu.forward_coeffs, emu2.forward_coeffs)
        # (b) different summation order via batch_size
        emu3, _ = fit_polyemu(X, Y, Xt, Yt, **_kwargs(p, batch_size=256))
        row["drift_batch256_rel"] = _rel_drift(emu.forward_coeffs, emu3.forward_coeffs)
        row["drift_batch256_rmse_rel"] = abs(accuracy(emu3.forward_emulator(Xt), Yt)["rmse"] - row["test_rmse"]) / row["test_rmse"]
        rows.append(row)
        print(f"[pins] {p['id']:24s} D={row['D']:4d} rmse={row['test_rmse']:.6e} cond={row['cond_M']:.2e} "
              f"c0={row['forward_coeffs_first10'][0]:+.15e} drift(repeat)={row['drift_repeat_rel']:.1e} "
              f"drift(batch256)={row['drift_batch256_rel']:.1e}", flush=True)
    return {"suite": "pins", "rows": rows}
