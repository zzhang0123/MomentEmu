"""Final verification table (load < 0.5x cores; interleaved min/median).

Measures the rows of IMPROVEMENT_PLAN.md "How to verify the whole plan"
section 1 that are not already covered by the benchmark suites.
"""
from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np

from MomentEmu.emulator import PolyEmu, evaluate_monomials_lazy, generate_multi_indices
from MomentEmu.monomials import MonomialPlan


def bench(fn, *, min_time=0.2, repeats=5):
    fn()
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    number = max(1, int(min_time / max(single, 1e-9)))
    blocks = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        blocks.append((time.perf_counter() - t0) / number)
    return min(blocks), statistics.median(blocks)


def main():
    out = {"loadavg": [round(x, 2) for x in os.getloadavg()]}
    for d in (5, 8):
        n = 6
        D = len(generate_multi_indices(n, d))
        rng = np.random.default_rng(0)
        X1 = rng.uniform(-1.0, 1.0, (1, n))
        X20k = rng.uniform(-1.0, 1.0, (20000, n))
        mi = generate_multi_indices(n, d)
        plan = MonomialPlan.build(mi)
        mn, md = bench(lambda: plan.evaluate(X1))
        ln, ld = bench(lambda: evaluate_monomials_lazy(X1, mi))
        out[f"phi_N1_D{D}_plan_us"] = 1e6 * mn
        out[f"phi_N1_D{D}_loop_us"] = 1e6 * ln
        if D == 3003:
            pn, pd = bench(lambda: plan.evaluate(X20k), min_time=1.0, repeats=3)
            qn, qd = bench(lambda: evaluate_monomials_lazy(X20k, mi), min_time=1.0, repeats=3)
            out["phi_N20000_D3003_plan_ms"] = 1e3 * pn
            out["phi_N20000_D3003_loop_ms"] = 1e3 * qn
    # forward_emulator single/batch at D=462, m=2000
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (5000, 6))
    Y = np.hstack([
        (np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3] + np.cos(X[:, 4]) + X[:, 5])[:, None]
        * np.linspace(0.5, 2, 2000)[None, :]
    ])
    emu = PolyEmu(
        X, Y, init_deg_forward=5, max_degree_forward=5, RMSE_tol=1e-300, verbose=0
    )
    x1, xb = X[0].copy(), X[:1000].copy()
    mn, md = bench(lambda: emu.forward_emulator(x1))
    out["forward_single_us"] = 1e6 * mn
    mn, md = bench(lambda: emu.forward_emulator(xb), min_time=0.5)
    out["forward_batch1000_ms"] = 1e3 * mn
    mn, md = bench(lambda: emu.jacobian(x1), min_time=0.2)
    out["jacobian_single_us"] = 1e6 * mn
    # JAX
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from MomentEmu.jax_momentemu import create_jax_emulator

        je = create_jax_emulator(emu)
        jx = jnp.asarray(x1)
        t0 = time.perf_counter()
        je(jx).block_until_ready()
        out["jax_compile_single_s"] = time.perf_counter() - t0
        mn, md = bench(lambda: np.asarray(je(jx)))
        out["jax_single_us"] = 1e6 * mn
        gout = jax.grad(lambda v: je.evaluate(v).sum())
        mn, md = bench(lambda: gout(jx), min_time=0.2)
        out["jax_grad_single_us"] = 1e6 * mn
        # per-call overhead
        ov = bench(lambda: je(jx), min_time=0.05, repeats=7)
        out["jax_overhead_us"] = 1e6 * ov[0]
    except Exception as exc:  # noqa: BLE001
        out["jax_error"] = repr(exc)
    # MomentEmu vs poly_normal_eq on a well-conditioned design
    rng = np.random.default_rng(2)
    Xg = rng.uniform(-1.0, 1.0, (400, 3))
    Yg = (Xg[:, 0] ** 2 + np.sin(Xg[:, 1]) + Xg[:, 2]).reshape(-1, 1)
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    e = PolyEmu(
        Xg, Yg, init_deg_forward=5, max_degree_forward=5, RMSE_tol=1e-300, verbose=0
    )
    sx, sy = StandardScaler().fit(Xg), StandardScaler().fit(Yg)
    P = PolynomialFeatures(degree=5).fit_transform(sx.transform(Xg))
    c_ne = np.linalg.solve(P.T @ P / len(P), P.T @ sy.transform(Yg) / len(P))
    key = {tuple(int(a) for a in row): i for i, row in enumerate(PolynomialFeatures(degree=5).fit(Xg).powers_)}
    # rebuild powers to get the mapping
    poly = PolynomialFeatures(degree=5).fit(sx.transform(Xg))
    key = {tuple(int(a) for a in row): i for i, row in enumerate(poly.powers_)}
    order = np.array([key[tuple(int(a) for a in mi2)] for mi2 in e.forward_multi_indices])
    out["momentemu_vs_normal_eq"] = float(
        np.max(np.abs(c_ne[order] - e.forward_coeffs)) / np.max(np.abs(e.forward_coeffs))
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
