"""P3.6: per-output transforms and backend parity (D4)."""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from MomentEmu.emulator import PolyEmu  # noqa: E402


def _mixed_fit(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (400, 3))
    Y = np.column_stack([
        X[:, 0] ** 2 + X[:, 1],                      # linear
        np.exp(0.5 * X[:, 0] + 0.2 * X[:, 1]) + 2.0,  # log
        np.sinh(0.8 * X[:, 2]) + 3.0 * X[:, 0],       # asinh
    ])
    return PolyEmu(
        X, Y, transform=["linear", "log", "asinh"], max_degree_forward=3, verbose=0
    ), X


def _col_rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_transform_codes_and_log_Y_equivalence():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = np.exp(0.4 * X[:, 0] + 0.2 * X[:, 1]).reshape(-1, 1)
    emu_logy = PolyEmu(X, Y, log_Y=True, max_degree_forward=3, verbose=0)
    emu_tf = PolyEmu(X, Y, transform="log", max_degree_forward=3, verbose=0)
    assert emu_tf.transform == ("log",)
    assert emu_tf.log_Y is True
    assert np.array_equal(emu_logy.forward_coeffs, emu_tf.forward_coeffs)


def test_mixed_transform_forward_parity():
    emu, X = _mixed_fit()
    rng = np.random.default_rng(2)
    Xe = rng.uniform(X.min(axis=0), X.max(axis=0), (500, 3))
    ref = emu.forward_emulator(Xe, extrapolation="ignore")
    assert ref.shape == (500, 3)
    # JAX
    from MomentEmu.jax_momentemu import create_jax_emulator

    je = create_jax_emulator(emu)
    got = np.asarray(je(jnp.asarray(Xe)))
    assert _col_rel(got, ref) < 1e-12
    # Torch
    import torch

    from MomentEmu.torch_momentemu import TorchMomentEmu

    tm = TorchMomentEmu(emu)
    with torch.no_grad():
        got_t = tm(torch.as_tensor(Xe, dtype=torch.float64)).numpy()
    assert _col_rel(got_t, ref) < 1e-12


def test_mixed_transform_symbolic_parity():
    import sympy as sp

    emu, X = _mixed_fit(seed=3)
    rng = np.random.default_rng(4)
    Xe = rng.uniform(X.min(axis=0), X.max(axis=0), (200, 3))
    ref = emu.forward_emulator(Xe, extrapolation="ignore")
    exprs = emu.generate_forward_symb_emu()
    syms = sp.symbols(["x1", "x2", "x3"])
    for j in range(3):
        f = sp.lambdify(syms, exprs[j], "numpy")
        got = np.asarray(f(*[Xe[:, i] for i in range(3)]), float).ravel()
        assert _col_rel(got, ref[:, j]) < 1e-12


def test_asinh_handles_negative_outputs():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = (np.sinh(1.5 * X[:, 0]) - 2.0 * X[:, 1]).reshape(-1, 1)
    emu = PolyEmu(X, Y, transform="asinh", max_degree_forward=4, verbose=0)
    Xe = rng.uniform(-1.0, 1.0, (200, 2))
    ref = emu.forward_emulator(Xe, extrapolation="ignore")
    assert np.all(np.isfinite(ref))

EMU_DIR = __import__("pathlib").Path("/Users/zzhang/Workspace/MomentEmu-PolyCAMB-examples/emulators")
DATA_DIR = __import__("pathlib").Path("/Users/zzhang/Workspace/MomentEmu-PolyCAMB-examples/datasets/100to108")


@pytest.mark.skipif(not (EMU_DIR / "PolyCAMB_Dl_EE.pkl").exists(), reason="companion emulators absent")
def test_shipped_ee_is_log_and_reproduces_0_175():
    import pickle

    with open(EMU_DIR / "PolyCAMB_Dl_EE.pkl", "rb") as f:
        emu = pickle.load(f)
    # The shipped degree-5 log fit stores the low-l max fractional error 0.175.
    assert emu.log_Y is True
    assert set(emu._transforms()) == {"log"}
    assert emu.forward_max_frac_err == pytest.approx(0.175, rel=0.02)


@pytest.mark.skipif(not (EMU_DIR / "PolyCAMB_Dl_TE.pkl").exists(), reason="companion emulators absent")
def test_shipped_te_is_linear():
    import pickle

    with open(EMU_DIR / "PolyCAMB_Dl_TE.pkl", "rb") as f:
        emu = pickle.load(f)
    assert emu.log_Y is False
    assert set(emu._transforms()) == {"linear"}


@pytest.mark.slow
@pytest.mark.skipif(
    not (DATA_DIR / "perturbed_LCDM_Dell_EE.npy").exists(),
    reason="companion datasets absent",
)
def test_ee_log_beats_linear_at_low_ell():

    P = np.load(DATA_DIR / "perturbed_LCDM_params.npy")
    # The first two columns are ell/garbage; the 4049 outputs follow.
    EE = np.load(DATA_DIR / "perturbed_LCDM_Dell_EE.npy")[:, 2:]
    Xtr, Ytr = P[:3000], EE[:3000]
    Xv, Yv = P[3000:], EE[3000:]
    low = slice(0, 30)  # low multipoles

    def low_err(transform):
        emu = PolyEmu(
            Xtr, Ytr, X_test=Xv, Y_test=Yv, transform=transform,
            init_deg_forward=5, max_degree_forward=5, RMSE_tol=1e-300, verbose=0,
        )
        pred = emu.forward_emulator(Xv, extrapolation="ignore")
        ref = np.asarray(Yv)[:, low]
        return float(np.max(np.abs(pred[:, low] - ref) / np.maximum(np.abs(ref), 1e-30)))

    err_log = low_err("log")
    err_linear = low_err("linear")
    assert err_log < err_linear
    assert err_log < 0.5, err_log
