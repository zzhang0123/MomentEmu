"""P0.4: benchmark harness load control and the normal-equations cross-check."""
from __future__ import annotations

import types

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from benchmarks.harness import fit_polyemu, fixed_degree_kwargs


def _run_main(tmp_path, monkeypatch, load, max_load):
    import benchmarks.__main__ as m

    fake = types.ModuleType("benchmarks.suite_pins")
    fake.run = lambda quick=False: {"suite": "pins", "rows": []}
    monkeypatch.setattr(m.importlib, "import_module", lambda name: fake)
    monkeypatch.setattr(m, "wait_for_load", lambda *a, **k: 0.0)
    monkeypatch.setattr(m, "loadavg", lambda: [load, load, load])
    monkeypatch.setattr(m, "environment", lambda: {"ok": True})
    monkeypatch.setattr(m, "build_report", lambda out: "report")
    rc = m.main(["--suites", "pins", "--out", str(tmp_path), "--max-load", str(max_load), "--max-wait", "0"])
    return rc


def test_high_load_writes_no_timing_json(tmp_path, monkeypatch):
    rc = _run_main(tmp_path, monkeypatch, load=999.0, max_load=1.0)
    assert rc == 0
    assert not (tmp_path / "pins.json").exists()
    assert (tmp_path / "pins.skipped").exists()
    assert (tmp_path / "environment.json").exists()


def test_quiet_run_writes_timing_json(tmp_path, monkeypatch):
    rc = _run_main(tmp_path, monkeypatch, load=0.0, max_load=1.0)
    assert rc == 0
    assert (tmp_path / "pins.json").exists()
    assert not (tmp_path / "pins.skipped").exists()


def test_poly_normal_eq_matches_momentemu():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (400, 3))
    Y = (X[:, 0] ** 2 + np.sin(X[:, 1]) + X[:, 2]).reshape(-1, 1)
    Xt = rng.uniform(-1.0, 1.0, (80, 3))
    Yt = (Xt[:, 0] ** 2 + np.sin(Xt[:, 1]) + Xt[:, 2]).reshape(-1, 1)
    emu, _ = fit_polyemu(X, Y, Xt, Yt, **fixed_degree_kwargs(3))

    sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    P = poly.fit_transform(sx.transform(X))
    Ys = sy.transform(Y)
    c_ne = np.linalg.solve(P.T @ P / len(P), P.T @ Ys / len(P))
    key = {tuple(int(a) for a in row): i for i, row in enumerate(poly.powers_)}
    order = np.array([key[tuple(int(a) for a in mi)] for mi in emu.forward_multi_indices])
    diff = float(np.max(np.abs(c_ne[order] - emu.forward_coeffs)))
    assert diff < 1e-11
