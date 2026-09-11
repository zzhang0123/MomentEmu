"""P1.2: input validation at fit and predict entry."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from MomentEmu import guards as g
from MomentEmu.PolyEmu import PolyEmu


def _good(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    Y = (X[:, 0] ** 2 + X[:, 1] + np.sin(X[:, 2])).reshape(-1, 1)
    return X, Y


def test_constant_column_raises_naming_it():
    X, Y = _good()
    X[:, 1] = 3.0
    with pytest.raises(ValueError, match=r"column.*1|\[1\]"):
        PolyEmu(X, Y, max_degree_forward=2)


def test_collinear_columns_warn():
    X, Y = _good()
    X[:, 2] = 2.0 * X[:, 0]
    with pytest.warns(g.EmulatorWarning, match="collinear"):
        g.check_design_columns(X)


def test_non_finite_raises_with_index():
    X, Y = _good()
    X[7, 1] = np.nan
    with pytest.raises(ValueError, match=r"X contains 1 non-finite.*first at index \(8, 1\)|first at index"):
        PolyEmu(X, Y, max_degree_forward=2)
    X, Y = _good()
    Y[3, 0] = np.inf
    with pytest.raises(ValueError, match="Y contains 1 non-finite"):
        PolyEmu(X, Y, max_degree_forward=2)


def test_object_dtype_raises_typeerror():
    X, Y = _good()
    Xo = X.astype(object)
    with pytest.raises(TypeError, match="must be numeric"):
        PolyEmu(Xo, Y, max_degree_forward=2)
    with pytest.raises(TypeError, match="must be numeric"):
        PolyEmu([["a", "b", "c"]] * 20, Y, max_degree_forward=2)


def test_1d_y_raises_with_reshape_hint():
    X, Y = _good()
    with pytest.raises(ValueError, match="reshape to"):
        PolyEmu(X, Y.ravel(), max_degree_forward=2)


def test_log_Y_nonpositive_raises():
    X, Y = _good()
    Y = Y - Y.min()  # contains zeros
    with pytest.raises(ValueError, match="log_Y=True requires Y > 0"):
        PolyEmu(X, Y, log_Y=True, max_degree_forward=2)


def test_float32_warns_and_int_fits_as_float64():
    X, Y = _good()
    with pytest.warns(g.PrecisionWarning, match="promoting to float64"):
        emu32 = PolyEmu(X.astype(np.float32), Y, max_degree_forward=2)
    assert emu32.forward_coeffs.dtype == np.float64
    # Integer inputs fit and predict in float64 (no int truncation).
    rng = np.random.default_rng(1)
    Xi = rng.integers(-5, 6, size=(400, 3))
    Yi = (Xi[:, 0] ** 2 + Xi[:, 1]).astype(float).reshape(-1, 1)
    emu = PolyEmu(Xi, Yi, max_degree_forward=2)
    out = emu.forward_emulator(Xi[:5])
    assert out.dtype == np.float64
    assert emu.forward_coeffs.dtype == np.float64


def test_lone_test_array_raises():
    X, Y = _good()
    with pytest.raises(ValueError, match="X_test and Y_test"):
        PolyEmu(X, Y, X_test=X[:10], max_degree_forward=2)


def test_predict_rejects_non_finite_and_bad_shape():
    X, Y = _good()
    emu = PolyEmu(X, Y, max_degree_forward=2)
    with pytest.raises(ValueError, match="non-finite"):
        emu.forward_emulator(np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="expected n_params"):
        emu.forward_emulator(np.zeros((4, 2)))


def test_python_O_keeps_validation():
    code = (
        "import numpy as np, sys; sys.path.insert(0, 'src'); "
        "from MomentEmu.PolyEmu import PolyEmu; "
        "X = np.zeros((50, 2)); X[:, 1] = np.arange(50); Y = X[:, 1:2]; "
        "\ntry:\n    PolyEmu(X, Y, max_degree_forward=2)\n"
        "except ValueError as e:\n    print('ValueError'); \n"
        "else:\n    print('NO ERROR')"
    )
    env = {"PYTHONPATH": "src", "MOMENTEMU_IGNORE_STALE": "1"}
    import os
    full = dict(os.environ)
    full.update(env)
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, env=full, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "ValueError" in proc.stdout, proc.stdout
