"""T-005: the active subspace from a supplied Jacobian instead of a pilot fit.

The rotation is currently read off a degree-3 pilot polynomial's analytic
Jacobian, so it inherits the pilot's error: on a synthetic ridge, leakage into
the dead directions grows from 0.0023 at pilot degree 3 to 0.0077 at degree 7.
A caller with a differentiable model (JAX, PyTorch, an autodiff simulator) has
the exact Jacobian and should be able to hand it over.

The trap these tests exist for is the coordinate system. active_subspace
standardises X and Y and works in those coordinates, while a caller's model
differentiates in physical units. For f(x) = g(a^T x),

    df/dxs = J_raw @ diag(scale_X) = g' * (a * scale_X)^T,

so the ridge direction in standardised coordinates is a * scale_X, NOT a. An
implementation that skips diag(scale_X) returns a different direction whenever
the parameters have different scales -- and it fails silently, because the
result is still a unit vector with a plausible spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.rotation import ActiveSubspaceEmu, active_subspace

# Deliberately lopsided, so a missing diag(scale_X) cannot hide.
SCALES = np.array([1.0, 10.0, 100.0])
DIRECTION = np.array([1.0, 1.0, 1.0])


def _ridge_design(n_samples: int = 400, seed: int = 0):
    """f(x) = g(a^T x) on columns whose scales differ by 100x."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, 3)) * SCALES
    t = X @ DIRECTION
    Y = np.column_stack([np.tanh(0.02 * t), 0.5 * np.tanh(0.02 * t) ** 2])
    return X, Y


def _exact_jacobian(X: np.ndarray) -> np.ndarray:
    """Analytic dY/dX in RAW units, shape (N, 2, 3)."""
    t = X @ DIRECTION
    s = np.tanh(0.02 * t)
    dg = 0.02 * (1.0 - s**2)                       # d/dt tanh(0.02 t)
    dh = 0.5 * 2.0 * s * dg                        # d/dt 0.5 tanh^2
    return np.stack([np.outer(dg, DIRECTION), np.outer(dh, DIRECTION)], axis=1)


def _true_standardised_direction(X: np.ndarray) -> np.ndarray:
    v = DIRECTION * X.std(axis=0)
    return v / np.linalg.norm(v)


def _alignment(v: np.ndarray, target: np.ndarray) -> float:
    return float(abs(v @ target) / (np.linalg.norm(v) * np.linalg.norm(target)))


def test_exact_jacobian_beats_the_pilot_estimate() -> None:
    """The whole point: an exact Jacobian is a better rotation than a pilot fit."""
    X, Y = _ridge_design()
    target = _true_standardised_direction(X)

    _e_pilot, V_pilot = active_subspace(X, Y, pilot_degree=3)
    _e_exact, V_exact = active_subspace(X, Y, jacobian=_exact_jacobian)

    exact = _alignment(V_exact[:, 0], target)
    pilot = _alignment(V_pilot[:, 0], target)
    assert exact > pilot, f"exact {exact:.6f} did not beat pilot {pilot:.6f}"
    assert exact > 1.0 - 1e-10


def test_supplied_jacobian_is_in_raw_coordinates() -> None:
    """The chain rule through diag(scale_X) must be applied by the package.

    Skipping it recovers DIRECTION itself rather than DIRECTION * scale_X;
    on these scales the two are 0.0057 apart in alignment terms, so this
    assertion separates them by a wide margin.
    """
    X, Y = _ridge_design()
    _e, V = active_subspace(X, Y, jacobian=_exact_jacobian)
    unscaled = DIRECTION / np.linalg.norm(DIRECTION)
    assert _alignment(V[:, 0], _true_standardised_direction(X)) > 1.0 - 1e-10
    assert _alignment(V[:, 0], unscaled) < 0.9


def test_callable_matches_precomputed_array() -> None:
    X, Y = _ridge_design()
    _e1, V1 = active_subspace(X, Y, jacobian=_exact_jacobian)
    _e2, V2 = active_subspace(X, Y, jacobian=_exact_jacobian(X))
    np.testing.assert_allclose(np.abs(V1), np.abs(V2), rtol=1e-12, atol=1e-14)


def test_callable_receives_raw_x() -> None:
    """The caller's model differentiates in its own units, so hand it those."""
    X, Y = _ridge_design()
    seen: list[np.ndarray] = []

    def spy(chunk: np.ndarray) -> np.ndarray:
        seen.append(chunk.copy())
        return _exact_jacobian(chunk)

    active_subspace(X, Y, jacobian=spy, batch_size=128)
    got = np.concatenate(seen, axis=0)
    assert got.shape == X.shape
    np.testing.assert_allclose(got, X, rtol=1e-12, atol=1e-14)


def test_batching_does_not_change_the_result() -> None:
    """Only the leading direction is comparable: the rest is the null space.

    The spectrum here is [1.925, 8.2e-20, 0] -- an exact rank-1 ridge. The
    trailing eigenvectors are an arbitrary basis of the null space and their
    ORDER flips with the summation order, so comparing them across batch
    sizes tests round-off, not the accumulation.
    """
    X, Y = _ridge_design()
    e1, V1 = active_subspace(X, Y, jacobian=_exact_jacobian, batch_size=64)
    e2, V2 = active_subspace(X, Y, jacobian=_exact_jacobian, batch_size=10**6)
    assert e1[1] / e1[0] < 1e-15, "design is no longer an exact rank-1 ridge"
    np.testing.assert_allclose(e1[0], e2[0], rtol=1e-12)
    np.testing.assert_allclose(np.abs(V1[:, 0]), np.abs(V2[:, 0]),
                               rtol=1e-10, atol=1e-12)


def test_single_output_jacobian_may_omit_the_output_axis() -> None:
    X, Y = _ridge_design()
    J = _exact_jacobian(X)[:, :1, :]
    _e1, V1 = active_subspace(X, Y[:, :1], jacobian=J)
    _e2, V2 = active_subspace(X, Y[:, :1], jacobian=J[:, 0, :])
    np.testing.assert_allclose(np.abs(V1), np.abs(V2), rtol=1e-12, atol=1e-14)


def test_gradient_covariance_can_be_supplied_directly() -> None:
    """A caller who already has C = E[J^T J] needs neither Y nor a pilot."""
    X, Y = _ridge_design()
    Js = _exact_jacobian(X) * X.std(axis=0)[None, None, :]
    Js = Js / float(np.sqrt(np.mean((Y - Y.mean(axis=0)) ** 2)))
    C = np.einsum("kmi,kmj->ij", Js, Js) / X.shape[0]
    _e1, V1 = active_subspace(X, Y, gradient_covariance=C)
    _e2, V2 = active_subspace(X, Y, jacobian=_exact_jacobian)
    np.testing.assert_allclose(np.abs(V1), np.abs(V2), rtol=1e-10, atol=1e-12)


def test_emulator_accepts_and_forwards_the_jacobian() -> None:
    X, Y = _ridge_design()
    # tanh(0.02 t) over |t| <= 111 needs the degree: measured relative error
    # 4.9e-02 at degree 4, 1.4e-02 at 6, 4.0e-03 at 8, 1.1e-03 at 10.
    emu = ActiveSubspaceEmu(
        X, Y, rank=1, jacobian=_exact_jacobian,
        init_deg_forward=10, max_degree_forward=10, RMSE_tol=0.0, verbose=0,
    )
    assert emu.rank == 1
    assert _alignment(emu.V[:, 0], _true_standardised_direction(X)) > 1.0 - 1e-10
    pred = emu.forward_emulator(X)
    rel = np.linalg.norm(pred - Y) / np.linalg.norm(Y)
    assert rel < 2e-3, rel


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((10, 3)), "rows"),
        (np.zeros((400, 2, 5)), "columns"),
        (np.zeros((400, 2, 3, 1)), "must be 2-D"),
    ],
)
def test_malformed_jacobian_is_rejected(bad, match) -> None:
    X, Y = _ridge_design()
    with pytest.raises(ValueError, match=match):
        active_subspace(X, Y, jacobian=bad)


def test_jacobian_and_covariance_are_mutually_exclusive() -> None:
    X, Y = _ridge_design()
    with pytest.raises(ValueError, match="not both"):
        active_subspace(
            X, Y, jacobian=_exact_jacobian, gradient_covariance=np.eye(3),
        )


def test_non_finite_jacobian_is_rejected() -> None:
    X, Y = _ridge_design()
    J = _exact_jacobian(X)
    J[3, 0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        active_subspace(X, Y, jacobian=J)


def test_exact_jacobian_kills_the_dead_directions() -> None:
    """A 2-D active subspace in 7 parameters: five directions are exactly dead.

    The pilot leaks into them and the leak does not go away by raising the
    pilot degree, it only gets more expensive. Measured on this design:

        rotation source   dead-direction share   subspace error
        pilot degree 3    2.6e-04                4.97e-03
        pilot degree 5    3.4e-04                1.66e-03
        pilot degree 7    1.1e-04                3.93e-04
        exact Jacobian    5.5e-16                0.0
    """
    rng = np.random.default_rng(3)
    n, n_samples = 7, 6000
    W = rng.standard_normal((2, n))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (n_samples, n))
    U = X @ W.T
    Y = np.column_stack(
        [np.tanh(U[:, 0]) + 0.5 * U[:, 1] ** 2, np.sin(U[:, 0] * U[:, 1])]
    )

    def exact(chunk: np.ndarray) -> np.ndarray:
        u = chunk @ W.T
        c = np.cos(u[:, 0] * u[:, 1])
        d0 = np.stack([1.0 - np.tanh(u[:, 0]) ** 2, u[:, 1]], axis=1)
        d1 = np.stack([c * u[:, 1], c * u[:, 0]], axis=1)
        return np.stack([d0 @ W, d1 @ W], axis=1)

    # f(x) = h(W x) with x = scale * xs + mean, so the STANDARDISED ridge
    # directions span the rows of W @ diag(scale_X), not of W.
    Q = np.linalg.qr((W * X.std(axis=0)[None, :]).T)[0]

    def subspace_error(V: np.ndarray) -> float:
        s = np.linalg.svd(Q.T @ V[:, :2], compute_uv=False)
        return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))

    e_ex, V_ex = active_subspace(X, Y, jacobian=exact)
    e_pi, V_pi = active_subspace(X, Y, pilot_degree=3)

    assert e_ex[2:].sum() / e_ex.sum() < 1e-12
    assert subspace_error(V_ex) < 1e-12
    assert e_pi[2:].sum() / e_pi.sum() > 1e-6
    assert subspace_error(V_pi) > subspace_error(V_ex)
