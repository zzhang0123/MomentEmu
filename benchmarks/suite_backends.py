"""Suite 6: inference through the shipped autodiff backends (JAX, Torch) vs numpy.

Fits the CMB-like target at the fixed degree and times single-point and
batch-1000 inference through PolyEmu.forward_emulator, the JAX wrapper
(jit, float64) and the Torch wrapper. Also records the JAX compile time and
the value agreement with numpy.
"""
from __future__ import annotations

import time

import numpy as np

from benchmarks.harness import fit_polyemu, fixed_degree_kwargs, time_inference
from benchmarks.targets import TARGETS


def run(quick: bool = False) -> dict:
    t = TARGETS["cmb_like"]
    X, Y, Xt, Yt = t.data()
    emu, _ = fit_polyemu(X, Y, Xt, Yt, **fixed_degree_kwargs(t.fixed_degree))
    ref1, refb = emu.forward_emulator(Xt[0]), emu.forward_emulator(Xt[:1000])
    rows = []
    r = {"backend": "numpy (PolyEmu.forward_emulator)", "note": f"D={len(emu.forward_multi_indices)}, m={t.m}"}
    r.update(time_inference(emu.forward_emulator, Xt))
    r["max_rel_diff_vs_numpy"] = 0.0
    rows.append(r)

    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from MomentEmu.jax_momentemu import create_jax_emulator
        f = create_jax_emulator(emu)
        x1, xb = jnp.asarray(Xt[0]), jnp.asarray(Xt[:1000])
        t0 = time.perf_counter(); f(x1).block_until_ready(); t_c1 = time.perf_counter() - t0
        t0 = time.perf_counter(); f(xb).block_until_ready(); t_cb = time.perf_counter() - t0
        r = {"backend": "jax (create_jax_emulator, jit, x64)",
             "jax_compile_single_ms": t_c1 * 1e3,
             "jax_compile_batch_ms": t_cb * 1e3,
             "note": f"first-call (compile) single {t_c1*1e3:.0f} ms, batch {t_cb*1e3:.0f} ms; "
                     f"{jax.__version__}, {jax.default_backend()}"}
        r.update(time_inference(lambda x: f(x).block_until_ready(), Xt, batch=1000))
        # time_inference passes numpy inputs; also time with device-resident inputs
        r["single_us_device_input"] = min(_rep(lambda: f(x1).block_until_ready()))
        r["batch_us_device_input"] = min(_rep(lambda: f(xb).block_until_ready()))
        r["max_rel_diff_vs_numpy"] = float(max(np.max(np.abs(np.asarray(f(x1)) - ref1)) / np.max(np.abs(ref1)),
                                               np.max(np.abs(np.asarray(f(xb)) - refb)) / np.max(np.abs(refb))))
        rows.append(r)
    except Exception as e:  # noqa: BLE001
        rows.append({"backend": "jax", "note": f"failed: {e!r}", "single_us": None, "batch_us": None, "batch_per_sample_us": None})

    try:
        import torch

        from MomentEmu.torch_momentemu import TorchMomentEmu
        tm = TorchMomentEmu(emu)
        tm = tm.double() if hasattr(tm, "double") else tm
        x1 = torch.as_tensor(Xt[0], dtype=torch.float64)
        xb = torch.as_tensor(Xt[:1000], dtype=torch.float64)
        with torch.no_grad():
            y1 = tm(x1).detach().numpy(); yb = tm(xb).detach().numpy()
        r = {"backend": "torch (TorchMomentEmu, float64, no_grad)", "note": f"torch {torch.__version__}"}
        with torch.no_grad():
            r["single_us"] = min(_rep(lambda: tm(x1)))
            r["batch_us"] = min(_rep(lambda: tm(xb)))
        r["batch_per_sample_us"] = r["batch_us"] / 1000
        r["max_rel_diff_vs_numpy"] = float(max(np.max(np.abs(y1.reshape(ref1.shape) - ref1)) / np.max(np.abs(ref1)),
                                               np.max(np.abs(yb.reshape(refb.shape) - refb)) / np.max(np.abs(refb))))
        rows.append(r)
    except Exception as e:  # noqa: BLE001
        rows.append({"backend": "torch", "note": f"failed: {e!r}", "single_us": None, "batch_us": None, "batch_per_sample_us": None})

    for r in rows:
        print(f"[backends] {r['backend']:45s} single={r['single_us'] and round(r['single_us'],1)}us "
              f"batch1000={r['batch_us'] and round(r['batch_us'],1)}us  {r['note']}", flush=True)
    return {"suite": "backends", "rows": rows}


def _rep(fn, repeats=5, min_time=0.2):
    from benchmarks.harness import time_call
    tc = time_call(fn, min_time=min_time, repeats=repeats)
    return [tc["min_s"] * 1e6]
