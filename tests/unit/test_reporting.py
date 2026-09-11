"""P1.4: sweep reporting semantics (logger, forward_degree, scale-free RMSE_tol)."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu


def _data(seed=0, n=3, N=400, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (N, n))
    Y = (X[:, 0] ** 3 + 0.5 * X[:, 0] * X[:, 1] + np.sin(X[:, 2])).reshape(-1, 1)
    return X, Y * scale


def test_verbose_zero_writes_nothing(capsys):
    X, Y = _data()
    PolyEmu(X, Y, cross_validation=False, max_degree_forward=3, verbose=0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_verbose_one_logs_to_stderr(capsys):
    X, Y = _data()
    PolyEmu(X, Y, cross_validation=False, max_degree_forward=2, verbose=1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "forward emulator" in captured.err.lower()


def test_forward_degree_and_deprecated_alias():
    X, Y = _data()
    emu = PolyEmu(X, Y, cross_validation=False, max_degree_forward=4, verbose=0)
    assert emu.forward_degree in emu.forward_degree_list
    with pytest.warns(DeprecationWarning, match="foward_degree"):
        alias = emu.foward_degree
    assert alias == emu.forward_degree


def test_rmse_tol_is_scale_free_with_std_false():
    X, Y = _data(seed=1)
    common = dict(
        cross_validation=False,
        standardize_Y_with_std=False,
        RMSE_tol=1e-4,
        max_degree_forward=5,
        verbose=0,
    )
    a = PolyEmu(X, Y, **common)
    b = PolyEmu(X, Y * 1e6, **common)
    assert a.forward_degree == b.forward_degree


def test_per_output_validation_rmse_shape():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (400, 3))
    Y = np.column_stack([
        X[:, 0] ** 2,
        np.sin(X[:, 1]),
        X[:, 2] ** 3,
    ])
    emu = PolyEmu(X, Y, cross_validation=False, max_degree_forward=4, verbose=0)
    assert emu.forward_RMSE_per_output_.shape == (3,)
    assert np.all(np.isfinite(emu.forward_RMSE_per_output_))
