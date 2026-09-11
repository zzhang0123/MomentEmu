"""Standard test functions with known structure.

Every target is a `Target` with a deterministic sampler (seeded), a box, the
function, the degree used for the fixed-degree fit and the sample sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Target:
    name: str
    n: int
    m: int
    lo: np.ndarray
    hi: np.ndarray
    f: Callable[[np.ndarray], np.ndarray]
    fixed_degree: int
    n_train: int
    n_test: int
    seed: int
    note: str = ""
    sampler: str = "uniform"  # "uniform" or "lhs"

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.sampler == "lhs":
            u = np.empty((n, self.n))
            for j in range(self.n):
                strata = (np.arange(n) + rng.random(n)) / n
                u[:, j] = rng.permutation(strata)
        else:
            u = rng.random((n, self.n))
        return self.lo + u * (self.hi - self.lo)

    def data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(X_train, Y_train, X_test, Y_test), train drawn before test from one stream."""
        rng = np.random.default_rng(self.seed)
        X = self.sample(self.n_train, rng)
        Xt = self.sample(self.n_test, rng)
        return X, self.f(X), Xt, self.f(Xt)


# ----------------------------------------------------------------------------
# Scalar functions
# ----------------------------------------------------------------------------
def ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1))[:, None]


SOBOL_A = np.array([0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0])


def sobol_g(X, a=SOBOL_A):
    return np.prod((np.abs(4.0 * X - 2.0) + a) / (1.0 + a), axis=1)[:, None]


def friedman(X):
    # 5 active inputs, inputs 6..10 inert.
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )[:, None]


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)[:, None]


def log_rosenbrock(X):
    return np.log1p(rosenbrock(X))


def gauss_peak(X, sigma=0.15):
    return np.exp(-np.sum((X - 0.5) ** 2, axis=1) / (2.0 * sigma ** 2))[:, None]


# ----------------------------------------------------------------------------
# Round-1 CMB-like synthetic spectrum (identical to proto_lowrank.py)
# ----------------------------------------------------------------------------
M_OUT = 2000
ELL = np.arange(2, 2 + M_OUT).astype(float)
CMB_LO = np.array([2.00e3, -0.10, 290.0, 1200.0, 0.50, -0.30])
CMB_HI = np.array([3.00e3, +0.10, 310.0, 1600.0, 0.90, +0.30])


def cmb_like(theta: np.ndarray) -> np.ndarray:
    A, tilt, lA, lD, r1, ph = (theta[:, i:i + 1] for i in range(6))
    ell = ELL[None, :]
    env = A * (ell / 220.0) ** tilt / (1.0 + (ell / 900.0) ** 2) ** 0.9
    damp = np.exp(-((ell / lD) ** 1.6))
    x = 2.0 * np.pi * ell / lA
    osc = (
        r1 * np.cos(x + ph)
        + 0.40 * r1 * np.cos(2.0 * x + 2.0 * ph + 0.30)
        + 0.15 * r1 * np.cos(3.0 * x + 3.0 * ph - 0.50)
    )
    plateau = 1.0 + 0.6 * np.exp(-((ell / 60.0) ** 1.2))
    return env * plateau * (1.0 + damp * osc)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def _box(n, lo, hi):
    return np.full(n, float(lo)), np.full(n, float(hi))


TARGETS: dict[str, Target] = {}


def _register(t: Target) -> Target:
    TARGETS[t.name] = t
    return t


_register(Target("ishigami", 3, 1, *_box(3, -np.pi, np.pi), ishigami, 9, 4000, 2000, 101,
                 note="strong x1*x3^4 interaction, sin terms; smooth"))
_register(Target("sobol_g", 8, 1, *_box(8, 0.0, 1.0), sobol_g, 5, 4000, 2000, 102,
                 note="anisotropic (a=[0,1,4.5,9,99x4]); |.| kinks: not smooth"))
_register(Target("friedman", 10, 1, *_box(10, 0.0, 1.0), friedman, 4, 4000, 2000, 103,
                 note="5 active + 5 inert inputs"))
_register(Target("rosenbrock", 4, 1, *_box(4, -2.0, 2.0), rosenbrock, 4, 4000, 2000, 104,
                 note="exact quartic polynomial: degree>=4 should be exact"))
_register(Target("log_rosenbrock", 4, 1, *_box(4, -2.0, 2.0), log_rosenbrock, 8, 4000, 2000, 105,
                 note="log(1+rosenbrock): curved valley, poorly polynomial"))
_register(Target("gauss_peak", 3, 1, *_box(3, 0.0, 1.0), gauss_peak, 10, 4000, 2000, 106,
                 note="sigma=0.15 peak in [0,1]^3: very poorly polynomial at low degree"))
_register(Target("cmb_like", 6, M_OUT, CMB_LO, CMB_HI, cmb_like, 5, 5000, 2000, 20260910,
                 note="round-1 synthetic D_ell, m=2000, LHS sampling (same RNG stream as proto_lowrank)",
                 sampler="lhs"))

STANDARD_ORDER = ["ishigami", "sobol_g", "friedman", "rosenbrock", "log_rosenbrock", "gauss_peak", "cmb_like"]
