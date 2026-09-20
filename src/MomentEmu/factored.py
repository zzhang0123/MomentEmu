"""Multiplicative (low-rank) structure over parameter blocks (T-002).

``Basis(blocks=...)`` exploits ADDITIVE separability, which is a statement
about which coefficients are zero. Multiplicative separability

    f(theta) = prod_k f_k(theta_{B_k})

is not sparsity at all. Expanding each factor,

    f = sum_{alpha_1 ... alpha_K} (prod_k c^(k)_{alpha_k}) z^(alpha_1 + ... + alpha_K)

so the coefficient tensor is a rank-one outer product over a FULL tensor-product
index set. The free parameters drop from prod_k |A_k| to sum_k |A_k| while the
effective total degree rises to sum_k d_k: three blocks at degree 5 reach total
degree 15 over nine parameters with 168 coefficients, where the isotropic
degree-15 basis has C(24, 15) = 1,307,504 terms.

The log transform turns a product into a sum, but only where the data is
strictly positive, and it degrades near any zero crossing whatever the sign. A
rank-R canonical (CP) model

    f(theta) ~ sum_{r=1}^{R} prod_k f_k^{(r)}(theta_{B_k})

needs no transform, handles sign changes, and contains the additive model as
the special case where the other factors are constant.

Fitting is multilinear rather than linear: holding every block but one fixed
leaves a weighted least-squares problem in that block, which is what
:class:`FactoredEmu` alternates over.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from MomentEmu.emulator import generate_multi_indices
from MomentEmu.guards import as_float64, check_finite, connected_components
from MomentEmu.monomials import MonomialPlan


def _standardise(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = A.mean(axis=0)
    scale = A.std(axis=0)
    return (A - mean) / np.where(scale > 0.0, scale, 1.0), mean, np.where(scale > 0.0, scale, 1.0)


class FactoredEmu:
    """Rank-R product-of-blocks emulator fitted by alternating least squares.

    Args:
        X, Y: training design (N, n) and outputs (N, m).
        blocks: partition of the parameters; each block gets its own factor.
        rank: number of product terms.
        degree: per-block polynomial degree (scalar or one per block).
        n_sweeps, tol: ALS iteration budget and relative-residual stopping rule.
        n_restarts: random restarts; the best training residual wins. ALS is
            non-convex and restarts are not optional: on a rank-2 target,
            single-start fits at the correct rank ranged over 64x depending on
            the seed (1.5e-2 to 9.6e-1). Selection is by training residual, so
            a bad start is discarded rather than returned. Inspect
            :attr:`restart_residuals` to see how close the run came to
            failing.
        ridge: Tikhonov term on each block solve.

    Attributes:
        factors: list of (D_k, rank) block coefficient arrays.
        weights: (m, rank) per-output mixing weights.
        n_parameters: total free coefficients.
        train_residual: relative rms residual of the retained fit.
    """

    def __init__(
        self,
        X: Any,
        Y: Any,
        blocks: Any,
        rank: int = 1,
        degree: Any = 4,
        n_sweeps: int = 80,
        tol: float = 1e-10,
        n_restarts: int = 5,
        ridge: float = 1e-8,
        random_state: Any = None,
    ) -> None:
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        check_finite(X, "X")
        check_finite(Y, "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n = X.shape[1]
        from MomentEmu.basis import validate_blocks

        blocks = tuple(tuple(int(i) for i in b) for b in blocks)
        validate_blocks(blocks, n)
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        degrees = (
            [int(degree)] * len(blocks) if np.ndim(degree) == 0
            else [int(v) for v in degree]
        )
        if len(degrees) != len(blocks):
            raise ValueError(
                f"degree must be a scalar or one value per block; got "
                f"{len(degrees)} for {len(blocks)} blocks"
            )

        self.blocks = blocks
        self.rank = rank
        self.degrees = tuple(degrees)
        Xs, self.mean_X_, self.scale_X_ = _standardise(X)
        # Outputs are SCALED but not centred. Subtracting the mean turns
        # prod_k f_k into prod_k f_k - c, which is not a product and needs one
        # extra rank to represent: on an exactly rank-one target, centring
        # left rank 1 at a test error of 7.0e-2 while rank 2 reached 6e-5.
        # Per-output scaling is harmless by contrast, because a per-column
        # factor is absorbed by the output weights.
        self.mean_Y_ = np.zeros(Y.shape[1])
        yscale = np.sqrt(np.mean(Y ** 2, axis=0))
        self.scale_Y_ = np.where(yscale > 0.0, yscale, 1.0)
        Ys = Y / self.scale_Y_
        self.plans = [
            MonomialPlan.build(generate_multi_indices(len(b), d))
            for b, d in zip(blocks, degrees)
        ]
        designs = [p.evaluate(Xs[:, list(b)]) for p, b in zip(self.plans, blocks)]
        self.n_parameters = int(
            rank * sum(d.shape[1] for d in designs) + rank * Ys.shape[1]
        )

        rng = np.random.default_rng(random_state)
        runs = [
            self._als(designs, Ys, rank, n_sweeps, tol, ridge, rng)
            for _ in range(max(1, int(n_restarts)))
        ]
        self.restart_residuals = tuple(run[2] for run in runs)
        self.factors, self.weights, self.train_residual = min(runs, key=lambda r: r[2])

    @staticmethod
    def _als(designs, Ys, rank, n_sweeps, tol, ridge, rng):
        """One ALS run. Returns (factors, weights, relative rms residual)."""
        K, N, m = len(designs), Ys.shape[0], Ys.shape[1]
        factors = [rng.standard_normal((d.shape[1], rank)) * 0.5 for d in designs]
        F = [d @ c for d, c in zip(designs, factors)]
        weights = rng.standard_normal((m, rank)) * 0.5
        scale = float(np.sqrt(np.mean(Ys ** 2))) or 1.0
        previous = np.inf

        for _ in range(int(n_sweeps)):
            for k in range(K):
                W = np.ones((N, rank))
                for j in range(K):
                    if j != k:
                        W *= F[j]
                B = [W[:, r: r + 1] * designs[k] for r in range(rank)]
                L = weights.T @ weights                      # (rank, rank)
                Dk = designs[k].shape[1]
                G = np.empty((rank * Dk, rank * Dk))
                rhs = np.empty(rank * Dk)
                YW = Ys @ weights                            # (N, rank)
                for r in range(rank):
                    rhs[r * Dk:(r + 1) * Dk] = B[r].T @ YW[:, r]
                    for s in range(rank):
                        G[r * Dk:(r + 1) * Dk, s * Dk:(s + 1) * Dk] = (
                            L[s, r] * (B[r].T @ B[s])
                        )
                G.flat[:: G.shape[0] + 1] += ridge
                try:
                    sol = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(G, rhs, rcond=None)[0]
                factors[k] = sol.reshape(rank, Dk).T
                F[k] = designs[k] @ factors[k]

            G_all = np.ones((N, rank))
            for f in F:
                G_all *= f
            GtG = G_all.T @ G_all
            GtG.flat[:: rank + 1] += ridge
            weights = np.linalg.solve(GtG, G_all.T @ Ys).T   # (m, rank)

            # Resolve the scale indeterminacy (lam f1)(f2 / lam) so the ridge
            # term penalises every factor comparably.
            for r in range(rank):
                norms = np.array([np.linalg.norm(c[:, r]) for c in factors])
                if np.all(norms > 0):
                    geo = float(np.exp(np.mean(np.log(norms))))
                    for k in range(len(factors)):
                        factors[k][:, r] *= geo / norms[k]
                    weights[:, r] *= float(np.prod(norms / geo))
            F = [d @ c for d, c in zip(designs, factors)]

            G_all = np.ones((N, rank))
            for f in F:
                G_all *= f
            resid = float(np.sqrt(np.mean((G_all @ weights.T - Ys) ** 2))) / scale
            if abs(previous - resid) <= tol * max(resid, 1e-300):
                previous = resid
                break
            previous = resid
        return factors, weights, previous

    def forward_emulator(self, X: Any) -> np.ndarray:
        """Predict at ``X`` in the original parameter coordinates."""
        arr = np.atleast_2d(as_float64(np.asarray(X), "X"))
        Xs = (arr - self.mean_X_) / self.scale_X_
        G = np.ones((arr.shape[0], self.rank))
        for plan, block, c in zip(self.plans, self.blocks, self.factors):
            G *= plan.evaluate(Xs[:, list(block)]) @ c
        return (G @ self.weights.T) * self.scale_Y_ + self.mean_Y_

    def report(self) -> dict:
        """Blocks, rank, degrees, free parameters and the restart spread."""
        return {
            "blocks": self.blocks,
            "rank": self.rank,
            "degrees": self.degrees,
            "n_parameters": self.n_parameters,
            "train_residual": self.train_residual,
            "restart_residuals": self.restart_residuals,
            # A wide spread means the objective has competing optima at this
            # rank; a narrow one at a high residual means the rank is too low.
            "restart_spread": (
                max(self.restart_residuals) / min(self.restart_residuals)
                if min(self.restart_residuals) > 0 else float("inf")
            ),
        }


def separability_report(
    model: Any,
    X: Any,
    blocks: Any = None,
    threshold: float = 0.1,
    step: float = 1e-4,
) -> dict:
    """Test additive and multiplicative block structure without a logarithm.

    Two criteria, both read from the fitted model's derivatives:

    * ``H_ij = d2f / dtheta_i dtheta_j`` vanishes across blocks when f is
      ADDITIVELY separable.
    * ``M_ij = f d2f/dtheta_i dtheta_j - (df/dtheta_i)(df/dtheta_j)`` vanishes
      across blocks when f is MULTIPLICATIVELY separable. It is the numerator
      of ``d2 log f / dtheta_i dtheta_j``, so it tests what the log transform
      tests while staying finite at ``f = 0`` and needing no positivity.

    ``interaction_graph`` cannot see multiplicative structure at all: a product
    has large cross-block ANOVA interaction and is reported as one block.

    Pass ``blocks`` to VERIFY a candidate partition, which is what this is good
    at. For each criterion it reports ``max cross-block / min within-block``;
    the structure is whichever criterion drives that ratio well below one and
    well below the other. Measured on nine parameters in three blocks at a fit
    nRMSE of 2.6e-4: a product gave M 0.31 against H 9.8, and a sum gave H
    0.067 against M 2.6.

    Leaving ``blocks`` as None asks the harder question of DISCOVERING the
    partition, and the answer is best-effort only. A single relative cut cannot
    serve blocks whose internal coupling differs by orders of magnitude, and
    the floor of both criteria is set by the fit's own error rather than by the
    criterion: on a product whose true cross-block M is zero, the observed
    cross-block value fell from 7.6e-2 to 2.7e-3 as the fit improved from
    1.9e-3 to 7.0e-6. Treat discovered blocks as a hypothesis to verify, and
    improve the fit before believing a merged answer.

    Args:
        model: anything with ``jacobian(X) -> (N, m, n)`` and
            ``forward_emulator(X)``; a fitted PolyEmu qualifies.
        X: points to evaluate the criteria at.
        blocks: candidate partition to verify, or None to attempt discovery.
        threshold: for discovery only, the fraction of the largest entry above
            which a pair counts as coupled.
        step: central-difference step for the second derivative. It must be
            finite and within [1e-10, 1e-1]. Below the floor the difference of
            two nearly equal Jacobians cancels and the verdict flips on noise
            (measured: the multiplicative ratio drifted from 2.5e-8 at 1e-10 to
            8.9e-1 at 1e-15 and crossed into "neither" at 1e-16); above the
            ceiling the difference stops being local. A step of zero, or a
            non-finite one, divides by zero and returns an infinite ratio,
            which reads as "neither" rather than as the error it is.

    Returns:
        dict with additive_matrix and multiplicative_matrix; with ``blocks``
        also additive_ratio, multiplicative_ratio and structure; without it
        additive_blocks and multiplicative_blocks.
    """
    step = float(step)
    if not np.isfinite(step) or not 1e-10 <= step <= 1e-1:
        raise ValueError(
            f"step must be finite and in [1e-10, 1e-1], got {step!r}; a step "
            "of zero or a non-finite one divides by zero and silently reports "
            "'neither', and below 1e-10 the central difference cancels"
        )
    Xa = np.atleast_2d(as_float64(np.asarray(X), "X"))
    n = Xa.shape[1]
    f = np.atleast_2d(model.forward_emulator(Xa))
    J = np.atleast_3d(model.jacobian(Xa))                  # (N, m, n)
    norm = float(np.sqrt(np.mean(f ** 2))) or 1.0

    # d2f/di dj from a central difference of the analytic Jacobian: exact in
    # the i index, second order in j, and only 2n Jacobian evaluations.
    hess = np.empty((n, n) + f.shape)
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        plus = np.atleast_3d(model.jacobian(Xa + e))
        minus = np.atleast_3d(model.jacobian(Xa - e))
        hess[:, j] = np.moveaxis((plus - minus) / (2.0 * step), 2, 0)

    add = np.zeros((n, n))
    mul = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = hess[i, j]
            add[i, j] = add[j, i] = float(np.sqrt(np.mean((h / norm) ** 2)))
            m_ij = (f * h - J[:, :, i] * J[:, :, j]) / norm ** 2
            mul[i, j] = mul[j, i] = float(np.sqrt(np.mean(m_ij ** 2)))

    out: dict = {"additive_matrix": add, "multiplicative_matrix": mul}
    if blocks is None:
        out["additive_blocks"] = connected_components(add > threshold * add.max())
        out["multiplicative_blocks"] = connected_components(mul > threshold * mul.max())
        out["threshold"] = float(threshold)
        return out

    from MomentEmu.basis import validate_blocks

    blocks = tuple(tuple(int(i) for i in b) for b in blocks)
    validate_blocks(blocks, n)
    label = [0] * n
    for k, block in enumerate(blocks):
        for i in block:
            label[i] = k
    cross = [(i, j) for i in range(n) for j in range(i + 1, n) if label[i] != label[j]]
    within = [(i, j) for i in range(n) for j in range(i + 1, n) if label[i] == label[j]]
    if not cross or not within:
        raise ValueError(
            "the ratio needs both a cross-block and a within-block pair; "
            f"got {len(cross)} and {len(within)} for blocks {blocks}"
        )

    def ratio(mat):
        lo = min(mat[i, j] for i, j in within)
        hi = max(mat[i, j] for i, j in cross)
        return float(hi / lo) if lo > 0 else float("inf")

    a_r, m_r = ratio(add), ratio(mul)
    structure = "neither"
    if m_r < 1.0 and m_r * 3.0 < a_r:
        structure = "multiplicative"
    elif a_r < 1.0 and a_r * 3.0 < m_r:
        structure = "additive"
    out.update({"blocks": blocks, "additive_ratio": a_r,
                "multiplicative_ratio": m_r, "structure": structure})
    return out

