"""T-005: a supplied Jacobian carried through the preconditioner composition.

:func:`~MomentEmu.rotation.active_subspace` takes the caller's derivatives,
but :class:`~MomentEmu.precondition.PreconditionedEmu` scores five orders and
one of them, "warp_rotate", puts a warp BEFORE the rotation. There the
rotation is computed in the warped coordinates, so ``dY/dX_raw`` is not the
Jacobian of what is being rotated: it has to be divided by the warp's own
derivative first.

Leaving that out is silent. The eigenvectors stay orthonormal and the spectrum
stays plausible; only the direction is wrong. The target below is a ridge in
LOG x, so once the axes are logged it is exactly rank one -- which makes the
spectrum a sharp, falsifiable statement about whether the chain rule ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.precondition import PreconditionedEmu, _Rotate, _Warp
from MomentEmu.rotation import active_subspace, gradient_covariance_matrix
from MomentEmu.warp import _make

A_DIR = np.array([1.0, -0.5, 2.0])
# Decades apart and of different widths, so a missing chain rule cannot hide
# behind axes that happen to share a scale.
RANGES = ((0.01, 1.0), (1.0, 50.0), (0.1, 1000.0))


def _target(X: np.ndarray) -> np.ndarray:
    return np.tanh(0.5 * (np.log(X) @ A_DIR))


def _log_ridge(n_samples: int = 600, seed: int = 3, ranges=RANGES):
    """f(x) = tanh(a^T log x): exactly rank one once the axes are logged."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        np.exp(rng.uniform(np.log(lo), np.log(hi), n_samples))
        for lo, hi in ranges
    ])
    return X, _target(X)[:, None]


def _raw_jacobian(X: np.ndarray) -> np.ndarray:
    """Analytic dY/dX in RAW units, shape (N, 1, 3). Carries a 1/x per axis."""
    g = 0.5 * (1.0 - np.tanh(0.5 * (np.log(X) @ A_DIR)) ** 2)
    return (g[:, None] * (A_DIR[None, :] / X))[:, None, :]


class _PinnedWarp(_Warp):
    """The warp step with log on every axis.

    Pinned rather than scanned so these tests are about the chain rule and not
    about what ``fit_warps`` happens to choose on this design.
    """

    def __init__(self, X: np.ndarray) -> None:      # noqa: D107
        self.warps = tuple(
            _make("log", "identity", 0.0, 0.0, X[:, i])
            for i in range(X.shape[1])
        )


def _unwarp(warp: _PinnedWarp, U: np.ndarray) -> np.ndarray:
    """Inverse of a log-then-affine warp, for differentiating numerically."""
    return np.column_stack([
        np.exp(w.lo + (U[:, i] + 1.0) * (w.hi - w.lo) / 2.0)
        for i, w in enumerate(warp.warps)
    ])


def _numerical_jacobian_in_warped(warp: _PinnedWarp, U: np.ndarray,
                                  h: float = 1e-5) -> np.ndarray:
    """dY/du by central differences, taken THROUGH the inverse warp.

    Independent of the package's chain rule: it only ever calls the target.
    """
    J = np.empty((U.shape[0], 1, U.shape[1]))
    for i in range(U.shape[1]):
        up, um = U.copy(), U.copy()
        up[:, i] += h
        um[:, i] -= h
        J[:, 0, i] = (_target(_unwarp(warp, up))
                      - _target(_unwarp(warp, um))) / (2.0 * h)
    return J


def _true_warped_direction(warp: _PinnedWarp, U: np.ndarray) -> np.ndarray:
    """The ridge direction in standardised warped coordinates, analytically.

    ``log x_i = lo_i + (u_i + 1)(hi_i - lo_i) / 2``, so ``t = a^T log x`` is
    linear in ``u`` with coefficients ``a_i (hi_i - lo_i) / 2``, and
    standardising ``u`` multiplies each of those by its own deviation.
    """
    half_span = np.array([(w.hi - w.lo) / 2.0 for w in warp.warps])
    v = A_DIR * half_span * U.std(axis=0)
    return v / np.linalg.norm(v)


def _alignment(u: np.ndarray, v: np.ndarray) -> float:
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def _rotate_after(warp, X, Y, **kwargs) -> _Rotate:
    return _Rotate(warp(X), Y, "auto", 0.999, 3, X_raw=X, warp=warp, **kwargs)


# --------------------------------------------------------------------------
# The chain rule itself
# --------------------------------------------------------------------------
def test_chain_rule_recovers_the_exact_ridge_direction() -> None:
    """In log coordinates the target is rank one along a known direction."""
    X, Y = _log_ridge()
    warp = _PinnedWarp(X)
    rot = _rotate_after(warp, X, Y, jacobian=_raw_jacobian)
    truth = _true_warped_direction(warp, warp(X))
    assert _alignment(rot.V[:, 0], truth) > 1.0 - 1e-10
    share = rot.eigenvalues[0] / rot.eigenvalues.sum()
    assert share > 1.0 - 1e-12, f"leading share {share:.15f}"
    assert rot.rank == 1


@pytest.mark.parametrize("decades", [1, 3, 8])
def test_chain_rule_holds_across_the_dynamic_range(decades: int) -> None:
    """Both ends of the range a log warp exists for.

    ``d/dx`` carries a ``1 / x``, so the raw Jacobian spans the same decades
    the design does; the chain rule has to cancel that at one decade and at
    eight.
    """
    ranges = tuple((10.0 ** -decades, 10.0 ** decades) for _ in A_DIR)
    X, Y = _log_ridge(ranges=ranges)
    warp = _PinnedWarp(X)
    rot = _rotate_after(warp, X, Y, jacobian=_raw_jacobian)
    truth = _true_warped_direction(warp, warp(X))
    assert _alignment(rot.V[:, 0], truth) > 1.0 - 1e-10
    assert rot.rank == 1


def test_chain_rule_matches_differentiating_in_the_warped_coordinates() -> None:
    """The analytic factor against central differences through the inverse warp."""
    X, Y = _log_ridge()
    warp = _PinnedWarp(X)
    analytic = _rotate_after(warp, X, Y, jacobian=_raw_jacobian)
    _e, V_ref = active_subspace(
        warp(X), Y, jacobian=_numerical_jacobian_in_warped(warp, warp(X))
    )
    assert _alignment(analytic.V[:, 0], V_ref[:, 0]) > 1.0 - 1e-8


def test_a_precomputed_array_takes_the_same_route_as_a_callable() -> None:
    X, Y = _log_ridge()
    warp = _PinnedWarp(X)
    by_call = _rotate_after(warp, X, Y, jacobian=_raw_jacobian)
    by_array = _rotate_after(warp, X, Y, jacobian=_raw_jacobian(X))
    assert np.allclose(by_call.eigenvalues, by_array.eigenvalues)
    assert _alignment(by_call.V[:, 0], by_array.V[:, 0]) > 1.0 - 1e-12


def test_skipping_the_chain_rule_points_somewhere_else() -> None:
    """What the omission costs, so the tests above have teeth.

    Feeding raw-axis derivatives to a rotation of the warped axes leaves the
    ``1 / x`` the log warp was there to absorb, and it is largest on whichever
    axis reaches the smallest values -- so the leading direction collapses
    onto that axis. Measured here: ``[0.99998, -0.003, 0.006]`` against a
    truth of ``[0.227, -0.098, 0.969]``, an alignment of 0.233.

    The spectrum does not report any of this. The naive route still shows a
    leading variance share of 0.9992, which is what makes the omission silent
    and why this is checked against the analytic direction instead.
    """
    X, Y = _log_ridge()
    warp = _PinnedWarp(X)
    truth = _true_warped_direction(warp, warp(X))
    naive, naive_V = active_subspace(warp(X), Y, jacobian=_raw_jacobian(X))

    assert _alignment(naive_V[:, 0], truth) < 0.3
    assert naive[0] / naive.sum() > 0.99, "the naive spectrum no longer looks healthy"


def test_a_single_output_jacobian_may_omit_the_output_axis() -> None:
    X, Y = _log_ridge()
    warp = _PinnedWarp(X)
    three_d = _rotate_after(warp, X, Y, jacobian=_raw_jacobian)
    two_d = _rotate_after(warp, X, Y, jacobian=lambda c: _raw_jacobian(c)[:, 0, :])
    assert np.allclose(three_d.eigenvalues, two_d.eigenvalues)


# --------------------------------------------------------------------------
# Routing through PreconditionedEmu
# --------------------------------------------------------------------------
def test_rotate_first_passes_the_jacobian_through_unchanged() -> None:
    """No warp precedes it, so the rotation is the raw-coordinate one."""
    X, Y = _log_ridge()
    emu = PreconditionedEmu(X, Y, order="rotate", jacobian=_raw_jacobian,
                            max_degree_forward=3, RMSE_tol=0.0, verbose=0)
    rot = emu.steps[0]
    _e, V_ref = active_subspace(X, Y, jacobian=_raw_jacobian)
    assert _alignment(rot.V[:, 0], V_ref[:, 0]) > 1.0 - 1e-12
    assert "from=jacobian" in rot.spec()


def test_auto_builds_the_raw_rotation_once() -> None:
    """'rotate' and 'rotate_warp' rotate the same coordinates, so one call.

    Two calls in total: the shared raw rotation, and 'warp_rotate', which is
    a genuinely different coordinate system.
    """
    X, Y = _log_ridge()
    calls = [0]

    def counting(chunk):
        calls[0] += 1
        return _raw_jacobian(chunk)

    emu = PreconditionedEmu(X, Y, order="auto", jacobian=counting,
                            max_degree_forward=3, RMSE_tol=0.0, verbose=0,
                            random_state=0)
    assert set(emu.scores) == {"none", "warp", "rotate", "rotate_warp",
                               "warp_rotate"}
    assert calls[0] == 2, f"the Jacobian was evaluated {calls[0]} times"


def test_a_supplied_covariance_is_refused_across_a_warp() -> None:
    X, Y = _log_ridge()
    C = gradient_covariance_matrix(X, Y, jacobian=_raw_jacobian)
    with pytest.raises(ValueError, match="cannot be carried across a warp"):
        PreconditionedEmu(X, Y, order="warp_rotate", gradient_covariance=C,
                          max_degree_forward=3, RMSE_tol=0.0, verbose=0)


def test_a_supplied_covariance_drops_warp_rotate_from_auto() -> None:
    X, Y = _log_ridge()
    C = gradient_covariance_matrix(X, Y, jacobian=_raw_jacobian)
    with pytest.warns(UserWarning, match="warp_rotate"):
        emu = PreconditionedEmu(X, Y, order="auto", gradient_covariance=C,
                                max_degree_forward=3, RMSE_tol=0.0, verbose=0,
                                random_state=0)
    assert "warp_rotate" not in emu.scores
    assert "rotate" in emu.scores


def test_jacobian_and_covariance_are_mutually_exclusive() -> None:
    X, Y = _log_ridge()
    C = gradient_covariance_matrix(X, Y, jacobian=_raw_jacobian)
    with pytest.raises(ValueError, match="not both"):
        PreconditionedEmu(X, Y, order="rotate", jacobian=_raw_jacobian,
                          gradient_covariance=C, max_degree_forward=3,
                          RMSE_tol=0.0, verbose=0)


def test_report_says_where_the_rotation_came_from() -> None:
    X, Y = _log_ridge()
    pilot = PreconditionedEmu(X, Y, order="rotate", max_degree_forward=3,
                              RMSE_tol=0.0, verbose=0)
    exact = PreconditionedEmu(X, Y, order="rotate", jacobian=_raw_jacobian,
                              max_degree_forward=3, RMSE_tol=0.0, verbose=0)
    assert "from=pilot" in pilot.report()["steps"][0]
    assert "from=jacobian" in exact.report()["steps"][0]
