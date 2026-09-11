"""P0.9: lazy backend imports, module isolation and the stale-file warning."""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import warnings

import pytest

from MomentEmu import _stale_check


def _run(code: str, *, ignore_stale: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if ignore_stale:
        env["MOMENTEMU_IGNORE_STALE"] = "1"
    else:
        env.pop("MOMENTEMU_IGNORE_STALE", None)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )


def test_import_time_and_module_isolation():
    times = []
    for _ in range(9):
        proc = _run(
            "import time, sys; t=time.perf_counter(); import MomentEmu; "
            "print(time.perf_counter()-t); "
            "print('BACKENDS', [m for m in ('jax','torch','sympy','sklearn') "
            "if m in sys.modules])"
        )
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.strip().splitlines()
        times.append(float(lines[0]))
        assert lines[1] == "BACKENDS []", lines[1]
    assert statistics.median(times) < 1.0, times


def test_star_import_with_torch_blocked():
    proc = _run(
        "import sys; sys.modules['torch'] = None; ns = {}; "
        "exec('from MomentEmu import *', ns); "
        "assert 'PolyEmu' in ns and 'signal_aware_frac_err' in ns; "
        "assert 'jax_momentemu' not in ns; print('ok')"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_importing_monomials_pulls_no_sklearn_or_sympy():
    env = dict(os.environ)
    env["MOMENTEMU_IGNORE_STALE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import MomentEmu.monomials"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sklearn" not in proc.stderr
    assert "sympy" not in proc.stderr


def test_stale_flat_module_warns(tmp_path, monkeypatch):
    fake = tmp_path / "MomentEmu.py"
    fake.write_text("")
    monkeypatch.delenv("MOMENTEMU_IGNORE_STALE", raising=False)
    monkeypatch.setattr(_stale_check, "find_stale_flat_modules", lambda search_path=None: [str(fake)])
    with pytest.warns(RuntimeWarning, match="stale single-file"):
        _stale_check.warn_if_stale()
    assert _stale_check.find_stale_flat_modules.__name__


def test_stale_warning_silenced(tmp_path, monkeypatch):
    monkeypatch.setenv("MOMENTEMU_IGNORE_STALE", "1")
    monkeypatch.setattr(
        _stale_check, "find_stale_flat_modules", lambda search_path=None: [str(tmp_path / "X")]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _stale_check.warn_if_stale()
