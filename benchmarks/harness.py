"""Shared measurement helpers: quiet fitting, timing, conditioning, accuracy."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import statistics
import time
from typing import Any, Callable

import numpy as np

from MomentEmu.core import signal_aware_frac_err
from MomentEmu.emulator import PolyEmu, evaluate_monomials_lazy


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    """PolyEmu prints its sweep; swallow it so the benchmark output is readable."""
    buf = io.StringIO()
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        logger.setLevel(old_level)


def fit_polyemu(X, Y, X_test=None, Y_test=None, **kw) -> tuple[PolyEmu, float]:
    """Fit and return (emulator, wall_seconds). Prints suppressed."""
    with quiet():
        t0, c0 = time.perf_counter(), time.process_time()
        emu = PolyEmu(X, Y, X_test=X_test, Y_test=Y_test, **kw)
        dt, dc = time.perf_counter() - t0, time.process_time() - c0
    # CPU time of the fit (all threads), stored next to the wall time: it is
    # less sensitive to other processes on a shared machine.
    emu._bench_fit_cpu_s = dc
    return emu, dt


def fit_polyemu_best(X, Y, X_test=None, Y_test=None, repeats: int = 3, **kw):
    """Fit `repeats` times; return (last emulator, min wall seconds, all wall seconds).

    The fit is deterministic when X_test/Y_test are given, so every repeat
    yields the same coefficients (verified by the pins suite); the minimum is
    the timing with the least scheduler noise.
    """
    times, cpu = [], []
    emu = None
    for _ in range(max(1, repeats)):
        emu, dt = fit_polyemu(X, Y, X_test, Y_test, **kw)
        times.append(dt)
        cpu.append(emu._bench_fit_cpu_s)
    emu._bench_fit_cpu_s = min(cpu)
    emu._bench_fit_cpu_s_all = cpu
    return emu, min(times), times


def fixed_degree_kwargs(d: int) -> dict:
    """Constructor arguments that pin the forward sweep to a single degree."""
    return dict(
        forward=True, backward=False,
        init_deg_forward=d, max_degree_forward=d,
        dim_reduction=False, RMSE_tol=1e-300,
    )


def basis_size(n: int, d: int) -> int:
    return math.comb(n + d, d)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_call(fn: Callable[[], Any], *, min_time: float = 0.2, repeats: int = 5,
              max_number: int = 100000) -> dict:
    """Return per-call seconds: min and median over `repeats` blocks.

    Each block runs `number` calls, where `number` is chosen so that a block
    lasts at least `min_time` seconds. One untimed warm-up call precedes.
    """
    fn()
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    number = int(min(max_number, max(1, math.ceil(min_time / max(single, 1e-9)))))
    blocks, cpu_blocks = [], []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        for _ in range(number):
            fn()
        blocks.append((time.perf_counter() - t0) / number)
        cpu_blocks.append((time.process_time() - c0) / number)
    return {
        "min_s": min(blocks),
        "median_s": statistics.median(blocks),
        "max_s": max(blocks),
        "min_cpu_s": min(cpu_blocks),
        "number": number,
        "repeats": repeats,
    }


def time_inference(predict: Callable[[np.ndarray], np.ndarray], X_test: np.ndarray,
                   batch: int = 1000, **kw) -> dict:
    """Single-point (1-D input) and batch-`batch` inference times."""
    x1 = np.ascontiguousarray(X_test[0])
    xb = np.ascontiguousarray(X_test[:batch])
    single = time_call(lambda: predict(x1), **kw)
    batched = time_call(lambda: predict(xb), **kw)
    return {
        "single_us": single["min_s"] * 1e6,
        "single_median_us": single["median_s"] * 1e6,
        "batch_n": int(len(xb)),
        "batch_us": batched["min_s"] * 1e6,
        "batch_median_us": batched["median_s"] * 1e6,
        "batch_per_sample_us": batched["min_s"] * 1e6 / len(xb),
        "single_cpu_us": single["min_cpu_s"] * 1e6,
        "batch_cpu_us": batched["min_cpu_s"] * 1e6,
    }


# ----------------------------------------------------------------------------
# Conditioning and accuracy
# ----------------------------------------------------------------------------
def moment_matrix(emu: PolyEmu, X: np.ndarray) -> np.ndarray:
    Xs = emu.scaler_X.transform(X)
    Phi = evaluate_monomials_lazy(Xs, emu.forward_multi_indices)
    return Phi.T @ Phi / len(Xs)


def cond_M(emu: PolyEmu, X: np.ndarray) -> dict:
    """Condition number of the moment matrix on the emulator's final basis.

    Uses the symmetric eigen-decomposition; lambda_min <= 0 means M is
    numerically singular and cond is reported as inf.
    """
    M = moment_matrix(emu, X)
    w = np.linalg.eigvalsh(M)
    lmin, lmax = float(w[0]), float(w[-1])
    cond = lmax / lmin if lmin > 0 else float("inf")
    return {"cond_M": cond, "lambda_min": lmin, "lambda_max": lmax, "D": int(M.shape[0])}


def accuracy(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float).reshape(ref.shape)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # RMS deviation of the reference about its per-output mean.
    scale = float(np.sqrt(np.mean(np.var(ref, axis=0))))
    sa = signal_aware_frac_err(pred, ref)
    return {
        "rmse": rmse,
        "nrmse": rmse / scale if scale > 0 else float("nan"),
        "max_abs_err": float(np.max(np.abs(err))),
        "sa_max_rel": float(sa["max_rel"]),
        "sa_rmse_rel": float(sa["rmse"]),
        "sa_in_mask_frac": sa["n_above"] / max(sa["n_total"], 1),
        "finite": bool(np.all(np.isfinite(pred))),
    }


def emulator_summary(emu: PolyEmu) -> dict:
    mi = emu.forward_multi_indices
    return {
        "degree": int(getattr(emu, "forward_degree", -1)),
        "D_final": int(len(mi)),
        "max_total_degree": int(mi.sum(axis=1).max()),
        "degrees_swept": [int(d) for d in getattr(emu, "forward_degree_list", [])],
        "rmse_val_scaled": [float(r) for r in getattr(emu, "forward_RMSE_list", [])],
        "rung_seconds": [float(t) for t in getattr(emu, "forward_running_time_list", [])],
        "n_coeffs": int(emu.forward_coeffs.size),
        "coeff_bytes": int(emu.forward_coeffs.nbytes),
    }


# ----------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default, allow_nan=True)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=3) -> str:
    """Compact number formatting for markdown tables."""
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if not np.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        return f"{x:.{digits}g}"
    return f"{x:.{max(digits - 1, 1)}e}"
