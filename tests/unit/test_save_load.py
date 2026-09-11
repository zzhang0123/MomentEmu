"""P4.2: versioned .npz save/load with a fingerprint."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu


def _fit(seed=0, log_Y=False, with_std=True, backward=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + np.sin(2 * X[:, 2]) + 5.0).reshape(-1, 1)
    kw = dict(log_Y=log_Y, standardize_Y_with_std=with_std, max_degree_forward=4, verbose=0)
    if backward:
        kw.update(forward=True, backward=True, max_degree_backward=3)
    return PolyEmu(X, Y, **kw)


def _col_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("log_Y", [False, True])
@pytest.mark.parametrize("with_std", [True, False])
def test_round_trip_predictions_identical(tmp_path, log_Y, with_std):
    emu = _fit(seed=1, log_Y=log_Y, with_std=with_std, backward=True)
    path = tmp_path / "emu.npz"
    emu.save(path)
    loaded = PolyEmu.load(path)
    assert np.array_equal(loaded.forward_coeffs, emu.forward_coeffs)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    np.testing.assert_array_equal(
        loaded.forward_emulator(X, extrapolation="ignore"),
        emu.forward_emulator(X, extrapolation="ignore"),
    )
    Y = loaded.forward_emulator(X, extrapolation="ignore") + 1e-3
    np.testing.assert_array_equal(
        loaded.backward_emulator(Y, extrapolation="ignore"),
        emu.backward_emulator(Y, extrapolation="ignore"),
    )


def test_load_with_only_numpy(tmp_path):
    emu = _fit(seed=3)
    path = tmp_path / "emu.npz"
    emu.save(path)
    code = (
        "import sys, numpy as np\n"
        "sys.modules['sklearn'] = None\n"
        "from MomentEmu.emulator import PolyEmu\n"
        f"e = PolyEmu.load({str(path)!r})\n"
        "X = np.linspace(-1, 1, 30).reshape(10, 3)\n"
        "print(e.forward_emulator(X).shape)\n"
    )
    env = dict(os.environ)
    env["MOMENTEMU_IGNORE_STALE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "(10, 1)" in proc.stdout


def test_tampered_coefficient_changes_fingerprint(tmp_path):
    emu = _fit(seed=4)
    path = tmp_path / "emu.npz"
    emu.save(path)
    before = emu.fingerprint()
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["forward_coeffs"] = arrays["forward_coeffs"].copy()
    arrays["forward_coeffs"][0, 0] += 1.0
    np.savez(path, **arrays)
    with pytest.warns(UserWarning, match="stored hash_"):
        tampered = PolyEmu.load(path)
    assert tampered.fingerprint() != before


def test_unknown_format_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, meta=np.frombuffer(b'{"format": "Nope"}', dtype=np.uint8), x=np.zeros(3))
    with pytest.raises(ValueError, match="unknown format"):
        PolyEmu.load(path)


def test_float32_round_trip_within_1e_6(tmp_path):
    emu = _fit(seed=5, log_Y=True)
    path = tmp_path / "emu32.npz"
    emu.save(path, float32=True)
    loaded = PolyEmu.load(path)
    assert loaded.forward_coeffs.dtype == np.float64
    rng = np.random.default_rng(6)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    ref = emu.forward_emulator(X, extrapolation="ignore")
    got = loaded.forward_emulator(X, extrapolation="ignore")
    assert _col_rel(got, ref) < 1e-6
