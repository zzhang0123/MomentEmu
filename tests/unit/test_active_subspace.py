"""T-002 / P2: active-subspace preconditioning.

A ridge target f(theta) = h(W theta) is low dimensional in the right
coordinates but has interactions at every ANOVA order in the original ones,
which is why interaction-order truncation fails on it while rotation works.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.emulator import PolyEmu, generate_multi_indices
from MomentEmu.rotation import ActiveSubspaceEmu, active_subspace

P = 7
_RNG = np.random.default_rng(20260920)
_W, _ = np.linalg.qr(_RNG.standard_normal((P, P)))
W1, W2 = _W[:, 0], _W[:, 1]          # the true 2-D active subspace


def _ridge(X):
    a, b = X @ W1, X @ W2
    return np.tanh(2.5 * a) * np.exp(-2.0 * b ** 2)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.uniform(-1, 1, (9000, P)), rng.uniform(-1, 1, (3000, P))
    return Xtr, _ridge(Xtr).reshape(-1, 1), Xte, _ridge(Xte).reshape(-1, 1)


def test_recovers_the_true_active_subspace(data):
    Xtr, Ytr, _, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evals, V = active_subspace(Xtr, Ytr)          # default pilot_degree=3
    share = evals / evals.sum()
    assert share[:2].sum() > 0.99, f"leading 2 carry only {share[:2].sum():.4f}"
    # the recovered 2-D span must contain the true directions
    Q = V[:, :2]
    for w in (W1, W2):
        w_std = w / np.linalg.norm(w)
        captured = float(np.linalg.norm(Q.T @ w_std) / np.linalg.norm(w_std))
        assert captured > 0.97, f"direction captured only {captured:.3f}"


def test_rank_auto_picks_the_variance_target(data):
    Xtr, Ytr, _, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = ActiveSubspaceEmu(Xtr, Ytr, variance_target=0.99,
                                max_degree_forward=6, verbose=0)
    assert emu.rank == 2
    assert emu.V.shape == (P, 2)


def test_rotation_beats_the_isotropic_basis_at_a_smaller_budget(data):
    """The measured claim: fewer coefficients, lower error."""
    Xtr, Ytr, Xte, Yte = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rotated = ActiveSubspaceEmu(Xtr, Ytr, rank=2,
                                    init_deg_forward=12, max_degree_forward=12,
                                    RMSE_tol=0.0, verbose=0)
        plain = PolyEmu(Xtr, Ytr, init_deg_forward=5, max_degree_forward=5,
                        RMSE_tol=0.0, verbose=0)
    d_rot = rotated.emulator.forward_multi_indices.shape[0]
    d_plain = plain.forward_multi_indices.shape[0]
    assert d_rot < d_plain, f"rotated {d_rot} vs plain {d_plain}"

    def nrmse(pred):
        return float(np.sqrt(np.mean((pred - Yte) ** 2)) / np.sqrt(np.mean(Yte ** 2)))

    e_rot = nrmse(rotated.forward_emulator(Xte, extrapolation="ignore"))
    e_plain = nrmse(plain.forward_emulator(Xte, extrapolation="ignore"))
    assert e_rot < e_plain, f"rotated {e_rot:.4e} vs plain {e_plain:.4e}"


def test_jacobian_matches_finite_differences(data):
    Xtr, Ytr, Xte, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emu = ActiveSubspaceEmu(Xtr, Ytr, rank=2,
                                max_degree_forward=8, verbose=0)
    pts = Xte[:5]
    J = emu.jacobian(pts)
    assert J.shape == (5, 1, P)
    h = 1e-5
    for i in range(P):
        e = np.zeros(P)
        e[i] = h
        fd = (emu.forward_emulator(pts + e, extrapolation="ignore")
              - emu.forward_emulator(pts - e, extrapolation="ignore")) / (2 * h)
        np.testing.assert_allclose(J[:, :, i], fd, rtol=2e-4, atol=1e-7)


def test_a_higher_degree_pilot_blurs_the_spectrum(data):
    """Regression for a counter-intuitive property: the pilot's own
    approximation wiggle enters the gradient covariance as signal, so a richer
    pilot makes the rank gap harder to read, not easier."""
    Xtr, Ytr, _, _ = data
    leak = {}
    for pilot in (3, 7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evals, _ = active_subspace(Xtr, Ytr, pilot_degree=pilot)
        leak[pilot] = float((evals[2:] / evals.sum()).sum())
    assert leak[7] > 2.0 * leak[3], leak


def test_rank_cannot_exceed_the_parameter_count(data):
    Xtr, Ytr, _, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="rank"):
            ActiveSubspaceEmu(Xtr, Ytr, rank=P + 1, verbose=0)


def test_full_rank_rotation_is_as_good_as_no_rotation(data):
    """r = n is a pure change of coordinates, so accuracy must be preserved."""
    Xtr, Ytr, Xte, Yte = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rot = ActiveSubspaceEmu(Xtr, Ytr, rank=P,
                                init_deg_forward=4, max_degree_forward=4,
                                RMSE_tol=0.0, verbose=0)
        plain = PolyEmu(Xtr, Ytr, init_deg_forward=4, max_degree_forward=4,
                        RMSE_tol=0.0, verbose=0)
    assert rot.emulator.forward_multi_indices.shape[0] == generate_multi_indices(P, 4).shape[0]
    def n(p):
        return float(np.sqrt(np.mean((p - Yte) ** 2)))

    a = n(rot.forward_emulator(Xte, extrapolation="ignore"))
    b = n(plain.forward_emulator(Xte, extrapolation="ignore"))
    assert a < 1.5 * b


def test_scan_rank_reports_the_budget_tradeoff(data):
    """The variance target overshoots on a shallow tail; scanning is what
    locates the knee."""
    from MomentEmu.rotation import scan_rank

    Xtr, Ytr, Xte, Yte = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = scan_rank(Xtr, Ytr, ranks=[1, 2, 3], degree=8,
                         X_test=Xte, Y_test=Yte, RMSE_tol=0.0, verbose=0)
    assert [r["rank"] for r in rows] == [1, 2, 3]       # sorted by n_terms
    assert all(r["error"] is None for r in rows)
    by_rank = {r["rank"]: r for r in rows}
    # rank 1 cannot represent a two-direction ridge; rank 2 can
    assert by_rank[2]["fom"] < 0.5 * by_rank[1]["fom"]
    # and the third direction buys little for its extra terms
    assert by_rank[3]["n_terms"] > by_rank[2]["n_terms"]


def test_scan_rank_rejects_mismatched_degree_list(data):
    from MomentEmu.rotation import scan_rank

    Xtr, Ytr, _, _ = data
    with pytest.raises(ValueError, match="one value per rank"):
        scan_rank(Xtr, Ytr, ranks=[1, 2], degree=[4, 5, 6])
