"""P5.2: the training moment build uses the P0.7 plan and matches the loop."""
from __future__ import annotations

import numpy as np
import pytest

from MomentEmu.core import generate_moment_products
from MomentEmu.emulator import (
    compute_moments_vector_output_batched,
    evaluate_monomials_lazy,
    generate_multi_indices,
)


@pytest.mark.parametrize("d", [2, 4, 6, 8])
def test_plan_build_matches_loop(d):
    rng = np.random.default_rng(d)
    N = 5000
    X = rng.uniform(-1.0, 1.0, (N, 6))
    Y = np.column_stack([X[:, 0] ** 2 + X[:, 1], np.sin(X[:, 2]) + X[:, 3]])
    mi = generate_multi_indices(6, d)
    M2, nu2 = compute_moments_vector_output_batched(X, Y, mi, batch_size=1000)
    Phi = evaluate_monomials_lazy(X, mi)
    M1, nu1 = generate_moment_products(Phi, Y)
    np.testing.assert_allclose(M2, M1, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(nu2, nu1, rtol=1e-10, atol=1e-14)


def test_weighted_build_matches_direct_weighted_moments():
    rng = np.random.default_rng(0)
    N = 2000
    X = rng.uniform(-1.0, 1.0, (N, 4))
    Y = (X[:, 0] ** 3 + X[:, 1]).reshape(-1, 1)
    w = rng.uniform(0.5, 2.0, N)
    mi = generate_multi_indices(4, 5)
    M, nu = compute_moments_vector_output_batched(X, Y, mi, batch_size=300, weights=w)
    Phi = evaluate_monomials_lazy(X, mi)
    ww = w / w.mean()
    M_hand = (Phi.T * ww) @ Phi / N
    nu_hand = (Phi.T * ww) @ Y / N
    np.testing.assert_allclose(M, M_hand, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(nu, nu_hand, rtol=1e-12, atol=1e-14)
