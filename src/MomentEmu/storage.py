"""Low-rank and float32 coefficient storage (P5.7), accuracy-gated under D1.

Reduced-rank regression has a closed form: with M = Phi^T Phi / N and
nu = Phi^T Y / N, the rank-r least-squares solution is
    C_r = L^-T [L^-1 nu]_r,
where M = L L^T and [.]_r truncates to the top r singular values.  The
training-MSE increase is exactly sum_{i>r} s_i^2 / m, and a rank above the
break-even r = D m / (D + m) is refused.
"""
from __future__ import annotations

import numpy as np


def break_even_rank(D, m):
    """Largest rank for which the reduced-rank fit does not increase the MSE."""
    return float(D) * float(m) / (float(D) + float(m))


def rank_mse_increase(singular_values, rank, m):
    """Exact training-MSE increase sum_{i>rank} s_i^2 / m."""
    s = np.asarray(singular_values, dtype=np.float64)
    return float(np.sum(s[int(rank):] ** 2) / m)


def reduced_rank_coefficients(M, nu, rank):
    """Closed-form rank-``rank`` least-squares coefficients.

    Returns (C_r (D, m), singular_values (min(D, m),)).
    """
    M = np.asarray(M, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    D, m = nu.shape
    if rank < 0 or rank > min(D, m):
        raise ValueError(f"rank must be in [0, {min(D, m)}], got {rank}")
    if rank > break_even_rank(D, m):
        raise ValueError(
            f"rank {rank} exceeds the break-even rank {break_even_rank(D, m):.1f} "
            f"for D = {D}, m = {m}; the reduced-rank fit would not save storage."
        )
    from scipy.linalg import solve_triangular

    L = np.linalg.cholesky(M)
    B = solve_triangular(L, nu, lower=True, check_finite=False)  # L^-1 nu
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    Br = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    C = solve_triangular(L.T, Br, lower=False, check_finite=False)  # L^-T Br
    return C, s
