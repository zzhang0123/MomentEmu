"""P1.7a: one regression test per round-1 finding id (mapping table).

Every function name is ``test_<finding id with underscores>``. Tests whose fix
has landed in Phases 0-1 must pass; tests for later tasks are
``xfail(strict=True)`` naming the task, so finishing that task flips the xfail
to XPASS (which strict mode reports as a failure).
"""
from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from MomentEmu import guards as g
from MomentEmu.core import select_best_model, solve_emulator_coefficients
from MomentEmu.emulator import (
    PolyEmu,
    evaluate_emulator_batched,
    evaluate_monomials,
    evaluate_monomials_lazy,
    generate_multi_indices,
    generate_multi_indices_with_degree_vec,
    given_order_indices,
    max_order,
)
from MomentEmu.monomials import evaluate_monomials_fast


@pytest.fixture(scope="module")
def emu():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (np.exp(0.4 * X[:, 0]) + X[:, 1] ** 2 + 5.0).reshape(-1, 1)
    return PolyEmu(X, Y, max_degree_forward=3, verbose=0)


@pytest.fixture(scope="module")
def logy_emu():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = np.exp(X[:, 0] + 0.3 * X[:, 1]).reshape(-1, 1)
    return PolyEmu(X, Y, log_Y=True, max_degree_forward=3, verbose=0)


# ---------------------------------------------------------------------------
# Critical / high findings
# ---------------------------------------------------------------------------
def test_logy_ignored_by_all_three_backends(logy_emu):
    # P2.1 added log_Y support to JAX; torch/symbolic still refuse.
    import jax.numpy as jnp

    from MomentEmu.jax_momentemu import create_jax_emulator
    from MomentEmu.torch_momentemu import TorchMomentEmu

    rng = np.random.default_rng(30)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    got = np.asarray(create_jax_emulator(logy_emu)(jnp.asarray(X)))
    assert np.max(np.abs(got - logy_emu.forward_emulator(X))) / np.max(
        np.abs(logy_emu.forward_emulator(X))
    ) < 1e-12
    # P2.3/P2.4: all three backends now support log_Y.
    import torch

    from MomentEmu.symbolic_momentemu import create_symbolic_emulator as make_sym

    assert make_sym(logy_emu, ["p0", "p1"])["expression"] is not None
    tm = TorchMomentEmu(logy_emu)
    with torch.no_grad():
        got_t = tm(torch.as_tensor(X, dtype=torch.float64)).numpy()
    assert np.max(np.abs(got_t - logy_emu.forward_emulator(X))) / np.max(
        np.abs(logy_emu.forward_emulator(X))
    ) < 1e-13


def test_autodiff_wrappers_ignore_log_y(logy_emu):
    import sympy as sp

    exprs = logy_emu.generate_forward_symb_emu()
    assert exprs[0].has(sp.exp)


def test_tests_do_not_import_repo_code():
    import MomentEmu

    assert MomentEmu.__file__.endswith("src/MomentEmu/__init__.py")


def test_autodiff_test_import_stale():
    from MomentEmu.emulator import PolyEmu as P  # noqa: F401
    assert callable(P)


def test_scaler_none_breaks_autodiff(emu):
    from MomentEmu.jax_momentemu import create_jax_emulator

    rng = np.random.default_rng(2)
    X = rng.uniform(-1.0, 1.0, (200, 3))
    Y = (X[:, 0] ** 2 + 4.0).reshape(-1, 1)
    e = PolyEmu(X, Y, standardize_Y_with_std=False, max_degree_forward=3, verbose=0)
    f = create_jax_emulator(e)
    import jax.numpy as jnp

    got = np.asarray(f(jnp.asarray(X[:20])))
    ref = e.forward_emulator(X[:20])
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


def test_scaler_scale_none_crashes_jax_and_torch():
    # Covered by test_scaler_none_breaks_autodiff; the guard is output_scale.
    assert np.array_equal(g.output_scale(type("S", (), {"scale_": None})(), 3), np.ones(3))


def test_jax_hessian_nan_at_training_mean(emu):
    import jax
    import jax.numpy as jnp

    from MomentEmu.jax_momentemu import create_jax_emulator

    jax.config.update("jax_enable_x64", True)
    f = create_jax_emulator(emu)
    h = jax.hessian(lambda v: f(v).sum())(jnp.asarray(emu.scaler_X.mean_))
    assert np.all(np.isfinite(np.asarray(h)))


def test_max_order_off_by_one():
    with pytest.warns(DeprecationWarning):
        k = max_order(2, 170)
    assert g.basis_size(2, k) < 170
    assert g.basis_size(2, k + 1) >= 170


def test_symb_emu_ignores_log_y(logy_emu):

    # A forward-only emulator has no backward coefficients; the export
    # must raise AttributeError rather than return log Y.
    if not hasattr(logy_emu, "backward_coeffs"):
        with pytest.raises(AttributeError):
            logy_emu.generate_backward_symb_emu()
    else:
        assert logy_emu.generate_backward_symb_emu()


def test_frac_err_masked_column_reports_zero():
    from MomentEmu.core import signal_aware_frac_err

    ref = np.array([[1.0, 1e-30], [2.0, 2e-30]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag = signal_aware_frac_err(ref.copy(), ref, absolute_floor=1e-15)
    assert diag["fully_masked_outputs"].tolist() == [1]


def test_select_best_model_nan_crash():
    assert select_best_model([np.nan, 1.0, 2.0]) == 1
    with pytest.raises(ValueError, match="non-finite RMSE"):
        select_best_model([np.nan, np.inf])


def test_no_conditioning_check_on_solve():
    with pytest.raises(g.IllConditionedError):
        solve_emulator_coefficients(np.diag([1.0, 1e-30]), np.ones((2, 1)))


def test_solve_no_conditioning_guard():
    with pytest.raises(g.IllConditionedError):
        solve_emulator_coefficients(np.array([[1.0, 2.0], [2.0, 1.0]]), np.ones((2, 1)))


def test_constant_or_collinear_x_column():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    Y = X[:, 0:1] ** 2
    X[:, 1] = 2.0
    with pytest.raises(ValueError, match="column"):
        PolyEmu(X, Y, max_degree_forward=2, verbose=0)


def test_backward_max_order_blowup():
    rng = np.random.default_rng(4)
    N, m = 4000, 12
    X = rng.uniform(-1.0, 1.0, (N, 1))
    Y = rng.uniform(0.0, 1.0, (N, m))
    e = PolyEmu(X, Y, forward=False, backward=True, verbose=0)
    assert e.backward_multi_indices.shape[0] * 2 <= N


def test_inference_python_loop_dominates():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, (500, 4))
    mi = generate_multi_indices(4, 4)
    a = evaluate_monomials_lazy(X, mi)
    b = evaluate_monomials_fast(X, mi)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-14


def test_forward_emulator_unbatched_inference(emu):
    rng = np.random.default_rng(6)
    X = rng.uniform(-0.5, 0.5, (200, 3))
    one = emu.forward_emulator(X, batch_size=1000, extrapolation="ignore")
    many = emu.forward_emulator(X, batch_size=13, extrapolation="ignore")
    np.testing.assert_allclose(one, many, rtol=1e-12)


def test_no_extrapolation_guard(emu):
    x = np.full((1, 3), 1e3)
    with pytest.warns(g.ExtrapolationWarning):
        emu.forward_emulator(x)


def test_unseeded_train_test_split():
    assert "random_state" in PolyEmu.__init__.__code__.co_varnames


def test_dim_reduction_rmse_not_stored():
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = (X[:, 0] ** 2 + X[:, 1]).reshape(-1, 1)
    Xt = rng.uniform(-1.0, 1.0, (50, 3))
    Yt = (Xt[:, 0] ** 2 + Xt[:, 1]).reshape(-1, 1)
    e = PolyEmu(X, Y, X_test=Xt, Y_test=Yt, max_degree_forward=3, verbose=0)
    Xs = e.scaler_X.transform(Xt)
    Ys = e.scaler_Y.transform(Yt)
    pred = evaluate_monomials_lazy(Xs, e.forward_multi_indices) @ e.forward_coeffs
    assert e.forward_RMSE == pytest.approx(float(np.sqrt(np.mean((pred - Ys) ** 2))))


def test_wrapper_1d_batch_silently_wrong(emu):
    import jax.numpy as jnp

    from MomentEmu.jax_momentemu import create_jax_emulator

    f = create_jax_emulator(emu)
    with pytest.raises(ValueError):
        f(jnp.zeros(5))  # n_params = 3, so a length-5 1-D input is ambiguous


# ---------------------------------------------------------------------------
# Medium / low findings
# ---------------------------------------------------------------------------

def test_constructor_does_all_work():
    assert hasattr(PolyEmu, "fit") and hasattr(PolyEmu, "predict")


def test_package_self_shadowing():
    import importlib
    import sys

    # P4.3: the old self-shadowing module names are deprecated aliases.
    for name in ("MomentEmu.PolyEmu", "MomentEmu.MomentEmu"):
        sys.modules.pop(name, None)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            mod = importlib.import_module(name)
        assert mod is not None


def test_requires_python_3_7_false():
    from pathlib import Path

    import tomllib

    cfg = tomllib.loads(Path("pyproject.toml").read_text())
    assert ">=3.7" not in cfg["project"]["requires-python"]


def test_torch_float32_hardcoded_no_escape(emu):
    from MomentEmu.torch_momentemu import TorchMomentEmu

    tm = TorchMomentEmu(emu)
    assert tm.coeffs.dtype == __import__("torch").float64


def test_torch_inplace_phi_breaks_vmap(emu):
    import torch

    from MomentEmu.torch_momentemu import TorchMomentEmu

    tm = TorchMomentEmu(emu)
    X = torch.rand((50, 3), dtype=torch.float64)
    grads = torch.func.vmap(torch.func.grad(lambda x: tm(x).sum()))(X)
    assert grads.shape == (50, 3) and bool(torch.isfinite(grads).all())


def test_symbolic_expr_unbound_monomial():
    from MomentEmu.emulator import symbolic_polynomial_expressions

    expr = symbolic_polynomial_expressions(np.ones((3, 1)), np.array([[0, 0], [1, 0], [0, 2]]))
    assert expr


def test_batched_evaluator_dtype_truncation(emu):
    out = emu.forward_emulator(np.array([[0, 0, 0]], dtype=np.int64))
    assert out.dtype == np.float64


def test_batched_eval_int_truncation(emu):
    rng = np.random.default_rng(8)
    X = np.array(rng.integers(-3, 4, (50, 3)), dtype=np.int64)
    out = emu.forward_emulator(X, extrapolation="ignore")
    assert out.dtype == np.float64


def test_x_test_without_y_test_ignored():
    rng = np.random.default_rng(9)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    Y = X[:, 0:1] ** 2
    with pytest.raises(ValueError, match="X_test and Y_test"):
        PolyEmu(X, Y, X_test=X[:10], verbose=0)


def test_nan_defeats_frac_err_gate():
    from MomentEmu.emulator import _public_max_frac_err

    ref = np.array([[1.0], [2.0]])
    diag = __import__("MomentEmu").signal_aware_frac_err(np.array([[1.0], [np.nan]]), ref)
    assert _public_max_frac_err(diag) == float("inf")


def test_float32_silent_precision_loss():
    rng = np.random.default_rng(10)
    X = rng.uniform(-1.0, 1.0, (200, 2)).astype(np.float32)
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    with pytest.warns(g.PrecisionWarning):
        e = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    assert e.forward_coeffs.dtype == np.float64


def test_integer_dtype_overflow():
    rng = np.random.default_rng(11)
    X = rng.integers(-5, 6, (300, 2))
    Y = (X[:, 0] ** 2 + X[:, 1]).astype(float).reshape(-1, 1)
    e = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    assert e.forward_emulator(X[:5]).dtype == np.float64


def test_constructor_no_input_validation():
    rng = np.random.default_rng(12)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    Y = X[:, 0] ** 2
    with pytest.raises(ValueError, match="reshape to"):
        PolyEmu(X, Y, max_degree_forward=2, verbose=0)


def test_assert_for_user_input_validation():
    # The message is a ValueError even under -O (no assert participates).
    rng = np.random.default_rng(13)
    X = rng.uniform(-1.0, 1.0, (100, 2))
    Y = X[:, 0:1] ** 2
    with pytest.raises(ValueError, match="expected n_params"):
        PolyEmu(X, Y, max_degree_forward=3, verbose=0).forward_emulator(np.zeros((3, 5)))


def test_metrics_and_penalties_dead_and_aicc_invalid():
    import MomentEmu.core as core

    assert not hasattr(core, "metrics_and_penalties")


def test_aic_bic_train_test_mixing():
    import inspect

    sig = inspect.signature(select_best_model)
    assert "rmse_tol" in sig.parameters


def test_rmse_tol_dimensionally_meaningless():
    # Scale-free by P1.4: the stored RMSE list is in the fitted space and the
    # tolerance is applied to the RMS of the target.
    rng = np.random.default_rng(14)
    X = rng.uniform(-1.0, 1.0, (300, 2))
    Y = (X[:, 0] ** 3 + np.sin(X[:, 1])).reshape(-1, 1)
    a = PolyEmu(X, Y, standardize_Y_with_std=False, RMSE_tol=1e-4, max_degree_forward=5, verbose=0)
    b = PolyEmu(X, Y * 1e6, standardize_Y_with_std=False, RMSE_tol=1e-4, max_degree_forward=5, verbose=0)
    assert a.forward_degree == b.forward_degree


def test_global_rmse_hides_worst_case():
    rng = np.random.default_rng(15)
    X = rng.uniform(-1.0, 1.0, (300, 3))
    Y = np.column_stack([X[:, 0] ** 2, np.sin(X[:, 1]), X[:, 2]])
    e = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    assert e.forward_RMSE_per_output_.shape == (3,)


def test_foward_degree_typo(emu):
    with pytest.warns(DeprecationWarning, match="foward_degree"):
        assert emu.foward_degree == emu.forward_degree


def test_unconditional_stdout_no_verbosity_flag(capsys):
    rng = np.random.default_rng(16)
    X = rng.uniform(-1.0, 1.0, (200, 2))
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    assert capsys.readouterr().out == ""


def test_no_sample_weighting():
    import inspect

    assert "weights" in inspect.signature(PolyEmu).parameters


def test_no_output_basis_reduction():
    # P4.2: persistence now ships as a versioned .npz.
    assert hasattr(PolyEmu, "save") and hasattr(PolyEmu, "load")
    assert hasattr(PolyEmu, "fingerprint")


def test_inverse_returns_conditional_mean():
    assert hasattr(PolyEmu, "backward_pca")


def test_jax_float32_default_silent():
    import inspect

    from MomentEmu.jax_momentemu import JaxEmulator

    rng = np.random.default_rng(40)
    X = rng.uniform(-1.0, 1.0, (200, 2))
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    e = PolyEmu(X, Y, max_degree_forward=3, verbose=0)
    je = JaxEmulator.from_polyemu(e)
    assert je.dtype == jax.numpy.float64
    assert "enable_x64" in inspect.getsource(JaxEmulator.from_polyemu)


def test_degree_sweep_rebuilds_moments():
    assert hasattr(PolyEmu, "forward_sweep_incremental_")


def test_solve_redone_each_degree_no_cholesky():
    assert hasattr(PolyEmu, "forward_sweep_incremental_")


def test_predictive_mse_returns_rmse():
    import MomentEmu.core as core

    assert hasattr(core, "predictive_rmse_aic_bic")


def test_license_missing_mit_attribution_clause():
    from pathlib import Path

    assert "shall be included in all copies" in Path("LICENSE").read_text()


def test_readme_jax_example_import_broken():
    from pathlib import Path

    text = Path("README.md").read_text()
    assert "from MomentEmu.jax_momentemu import create_jax_emulator" in text


def test_sympy_simplify_wasted_in_symbolic_export():
    from pathlib import Path

    assert "sp.simplify" not in Path("src/MomentEmu/emulator.py").read_text()


def test_notebook_test_uses_removed_parameter():
    from pathlib import Path

    assert "RMSE_lower" not in Path("tests/test_inhomo.ipynb").read_text()


def test_batch_size_default_disables_batching():
    rng = np.random.default_rng(17)
    X = rng.uniform(-1.0, 1.0, (1000, 3))
    Y = (X[:, 0] ** 2).reshape(-1, 1)
    e = PolyEmu(X, Y, max_degree_forward=2, verbose=0)
    assert e.batch_size_ < X.shape[0]


def test_filter_modes_joint_drop_no_control():
    from MomentEmu.core import filter_modes

    with pytest.warns(DeprecationWarning):
        filter_modes(np.ones((2, 1)), np.eye(2))


def test_lazy_monomials_vs_downward_closed_recursion():
    rng = np.random.default_rng(18)
    mi = generate_multi_indices(3, 5)
    keep = rng.random(mi.shape[0]) > 0.5
    keep[0] = True
    pruned = mi[keep]
    X = rng.uniform(-1.0, 1.0, (100, 3))
    a = evaluate_monomials(X, pruned)
    b = evaluate_monomials_fast(X, pruned)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-13


def test_generate_multi_indices_with_degree_vec():
    mi = generate_multi_indices_with_degree_vec([1, 2])
    assert tuple(mi.max(axis=0)) == (1, 2)
    assert mi.shape[0] == 6


def test_given_order_indices():
    assert given_order_indices(2, 2).shape[0] == 3


def test_evaluate_emulator_batched_matches_lazy():
    rng = np.random.default_rng(19)
    X = rng.uniform(-1.0, 1.0, (200, 2))
    mi = generate_multi_indices(2, 3)
    c = rng.standard_normal((mi.shape[0], 1))
    a = evaluate_monomials_lazy(X, mi) @ c
    b = evaluate_emulator_batched(X, c, mi, batch_size=17)
    np.testing.assert_allclose(a, b, rtol=1e-13)
