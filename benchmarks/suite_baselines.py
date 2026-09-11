"""Suite 2: sklearn baselines on the same data.

Baselines
- poly_lr    : PolynomialFeatures(d) + LinearRegression (lstsq on the design matrix).
               Same model class as MomentEmu at the same degree; coefficients compared.
- poly_normal_eq : PolynomialFeatures(d) + np.linalg.solve on the normal equations.
               The same estimator MomentEmu uses; coefficients must agree to < 1e-11.
- poly_ridge : PolynomialFeatures(d) + RidgeCV(alphas=logspace(-10, 2, 13), gcv_mode='eigen').
               (the default gcv_mode='svd' returned a mean predictor on the exact-quartic
               rosenbrock target at alpha=1e-10 while Ridge(alpha=1e-10) itself is exact)
- gp         : GaussianProcessRegressor, ConstantKernel*RBF(ARD)+WhiteKernel, N capped at 2000.
               For m > 20 outputs the kernel hyper-parameters are fitted on 20 evenly
               spaced output columns and reused (optimizer=None) for all m columns.
- mlp        : MLPRegressor (64,64) tanh (128,128 when m > 100), adam.

All baselines see StandardScaler-standardised X and Y, as MomentEmu does.
Fit times in this suite are single-shot (the GP alone is minutes at m=2000).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from benchmarks.harness import accuracy, fit_polyemu, fixed_degree_kwargs, time_inference
from benchmarks.targets import STANDARD_ORDER, TARGETS

GP_N_CAP = 2000
GP_M_HYPER = 20


class Scaled:
    """Wrap an estimator fitted on standardised X and Y so predict() takes raw X."""

    def __init__(self, est, sx: StandardScaler, sy: StandardScaler):
        self.est, self.sx, self.sy = est, sx, sy

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X2 = X.reshape(1, -1) if X.ndim == 1 else X
        Ys = self.est.predict(self.sx.transform(X2))
        if Ys.ndim == 1:
            Ys = Ys[:, None]
        return self.sy.inverse_transform(Ys)


def _timed_fit(make, Xs, Ys):
    t0 = time.perf_counter()
    est = make().fit(Xs, Ys)
    return est, time.perf_counter() - t0


def fit_gp(Xs, Ys, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), min(GP_N_CAP, len(Xs)), replace=False)
    X, Y = Xs[idx], Ys[idx]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * RBF(np.ones(n), (1e-2, 1e2))
              + WhiteKernel(1e-5, (1e-12, 1e-1)))
    t0 = time.perf_counter()
    if Y.shape[1] > GP_M_HYPER:
        cols = np.linspace(0, Y.shape[1] - 1, GP_M_HYPER).astype(int)
        gp0 = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0).fit(X, Y[:, cols])
        gp = GaussianProcessRegressor(gp0.kernel_, optimizer=None, normalize_y=True).fit(X, Y)
        note = f"hyper on {GP_M_HYPER} cols, N={len(X)}"
    else:
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0).fit(X, Y)
        note = f"N={len(X)}"
    return gp, time.perf_counter() - t0, note, str(gp.kernel_)


def compare_coefficients(emu, lr: LinearRegression, poly: PolynomialFeatures) -> dict:
    """Map PolynomialFeatures columns onto MomentEmu multi-indices and compare."""
    powers = poly.powers_
    key = {tuple(int(a) for a in row): i for i, row in enumerate(powers)}
    order = np.array([key[tuple(int(a) for a in mi)] for mi in emu.forward_multi_indices])
    c_lr = np.atleast_2d(lr.coef_)  # (m, D_poly)
    c_lr = c_lr[:, order].T          # (D, m) in MomentEmu order
    c_emu = emu.forward_coeffs
    diff = np.abs(c_lr - c_emu)
    scale = np.max(np.abs(c_emu))
    return {
        "coef_max_abs_diff": float(diff.max()),
        "coef_max_rel_diff": float(diff.max() / scale) if scale > 0 else float("nan"),
        "coef_scale": float(scale),
        "same_basis_size": bool(c_lr.shape == c_emu.shape),
    }


def run(quick: bool = False) -> dict:
    rows = []
    names = STANDARD_ORDER if not quick else ["ishigami", "cmb_like"]
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name in names:
        t = TARGETS[name]
        X, Y, Xt, Yt = t.data()
        d = t.fixed_degree
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(Y)
        Xs, Ys = sx.transform(X), sy.transform(Y)
        print(f"[baselines] {name}: n={t.n} m={t.m} d={d}", flush=True)

        # --- MomentEmu at the same fixed degree (reference row) -------------
        emu, t_fit = fit_polyemu(X, Y, Xt, Yt, **fixed_degree_kwargs(d))
        rows.append(_row(t, "momentemu", t_fit, emu.forward_emulator, Xt, Yt, note=f"d={d}, D={len(emu.forward_multi_indices)}"))

        # --- PolynomialFeatures + LinearRegression ---------------------------
        poly = PolynomialFeatures(degree=d, include_bias=True)
        t0 = time.perf_counter()
        P = poly.fit_transform(Xs)
        t_feat = time.perf_counter() - t0
        lr, t_lr = _timed_fit(lambda: LinearRegression(fit_intercept=False), P, Ys)

        class PolyPipe:
            def __init__(self, est):
                self.est = est

            def predict(self, Xs_):
                return self.est.predict(poly.transform(Xs_))

        pred_lr = Scaled(PolyPipe(lr), sx, sy)
        r = _row(t, "poly_lr", t_feat + t_lr, pred_lr.predict, Xt, Yt,
                 note=f"features {t_feat:.3f}s + lstsq {t_lr:.3f}s")
        r.update(compare_coefficients(emu, lr, poly))
        # prediction agreement between the two solvers on the test set
        r["pred_max_rel_diff_vs_momentemu"] = float(
            np.max(np.abs(pred_lr.predict(Xt) - emu.forward_emulator(Xt))) / np.max(np.abs(Yt)))
        rows.append(r)

        # --- PolynomialFeatures + normal equations ---------------------------
        # The same estimator MomentEmu uses (solve the normal equations of the
        # standardised design matrix). Coefficients should agree to ~1e-13, so
        # this row is the acceptance gate on the solver, not a baseline.
        M_ne = P.T @ P / len(P)
        nu_ne = P.T @ Ys / len(P)
        t0 = time.perf_counter()
        c_ne = np.linalg.solve(M_ne, nu_ne)
        t_ne = time.perf_counter() - t0

        class NormalEq:
            def __init__(self, c):
                self.c = c

            def predict(self, Xs_):
                return poly.transform(Xs_) @ self.c

        pred_ne = Scaled(NormalEq(c_ne), sx, sy)
        powers = poly.powers_
        key = {tuple(int(a) for a in row): i for i, row in enumerate(powers)}
        order = np.array([key[tuple(int(a) for a in mi)] for mi in emu.forward_multi_indices])
        scale = float(np.max(np.abs(emu.forward_coeffs)))
        r = _row(t, "poly_normal_eq", t_feat + t_ne, pred_ne.predict, Xt, Yt,
                 note=f"normal equations {t_ne:.3f}s")
        r["coef_max_abs_diff"] = float(np.max(np.abs(c_ne[order] - emu.forward_coeffs)))
        r["coef_max_rel_diff"] = r["coef_max_abs_diff"] / scale if scale > 0 else float("nan")
        r["pred_max_rel_diff_vs_momentemu"] = float(
            np.max(np.abs(pred_ne.predict(Xt) - emu.forward_emulator(Xt))) / np.max(np.abs(Yt)))
        rows.append(r)

        # --- PolynomialFeatures + RidgeCV -----------------------------------
        ridge, t_ridge = _timed_fit(lambda: RidgeCV(alphas=np.logspace(-10, 2, 13), fit_intercept=False, gcv_mode='eigen'), P, Ys)
        r = _row(t, "poly_ridgecv", t_feat + t_ridge, Scaled(PolyPipe(ridge), sx, sy).predict, Xt, Yt,
                 note=f"alpha={ridge.alpha_:.1e}")
        rows.append(r)

        # --- GP ---------------------------------------------------------------
        gp, t_gp, gp_note, kern = fit_gp(Xs, Ys, t.n)
        r = _row(t, "gp_rbf_white", t_gp, Scaled(gp, sx, sy).predict, Xt, Yt, note=gp_note)
        r["kernel"] = kern
        rows.append(r)

        # --- MLP --------------------------------------------------------------
        hidden = (128, 128) if t.m > 100 else (64, 64)
        max_iter = 500 if t.m > 100 else 1000
        mlp, t_mlp = _timed_fit(
            lambda: MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="adam",
                                 max_iter=max_iter, tol=1e-7, n_iter_no_change=50, random_state=0),
            Xs, Ys if Ys.shape[1] > 1 else Ys.ravel())
        r = _row(t, "mlp", t_mlp, Scaled(mlp, sx, sy).predict, Xt, Yt,
                 note=f"{hidden} tanh, {mlp.n_iter_} iters")
        rows.append(r)

        for rr in rows[-5:]:
            print(f"    {rr['method']:13s} fit={rr['fit_s']:8.3f}s nrmse={rr['test_nrmse']:.3e} "
                  f"sa_max_rel={rr['test_sa_max_rel']:.3e} single={rr['infer_single_us']:8.1f}us "
                  f"batch1000={rr['infer_batch_us']:9.1f}us  {rr.get('note','')}", flush=True)
    return {"suite": "baselines", "rows": rows, "gp_n_cap": GP_N_CAP, "gp_m_hyper": GP_M_HYPER}


def _row(t, method, fit_s, predict, Xt, Yt, note="") -> dict:
    pred = predict(Xt)
    row = {"target": t.name, "method": method, "n": t.n, "m": t.m, "n_train": t.n_train,
           "fit_s": fit_s, "note": note}
    row.update({f"test_{k}": v for k, v in accuracy(pred, Yt).items()})
    row.update({f"infer_{k}": v for k, v in time_inference(predict, Xt, min_time=0.1, repeats=3).items()})
    return row
