"""Regenerate the figures in docs/assets.

Run from the repository root:

    python3 docs/figures/make_figures.py

Every number plotted is computed here from the package itself, so a figure
cannot drift from the code it describes. Each figure is written twice, as
``<name>-light.svg`` and ``<name>-dark.svg``, because mkdocs-material ships
both a light and a dark scheme and swaps them on the ``#only-light`` and
``#only-dark`` URL fragments. A single fixed-colour figure is unreadable in
one of the two.
"""
from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Colourblind-safe, and distinguishable against both backgrounds.
COLOURS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

THEMES = {
    "light": {"fg": "#20242c", "grid": "#d7dae0", "bg": "none"},
    "dark": {"fg": "#d6dae2", "grid": "#3b4049", "bg": "none"},
}


def _style(theme: dict) -> dict:
    return {
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["bg"],
        "savefig.facecolor": theme["bg"],
        "text.color": theme["fg"],
        "axes.labelcolor": theme["fg"],
        "axes.edgecolor": theme["fg"],
        "xtick.color": theme["fg"],
        "ytick.color": theme["fg"],
        "grid.color": theme["grid"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }


def render(name: str, draw) -> None:
    """Run ``draw(ax)`` once per theme and write the two SVGs."""
    for theme_name, theme in THEMES.items():
        with plt.rc_context(_style(theme)):
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            draw(ax)
            fig.tight_layout()
            out = ASSETS / f"{name}-{theme_name}.svg"
            fig.savefig(out, format="svg", transparent=True)
            plt.close(fig)
            print(f"wrote {out.relative_to(ASSETS.parent.parent)}")


def basis_growth(ax) -> None:
    """Term count against degree at 7 parameters, one line per lever."""
    from MomentEmu.basis import Basis

    n = 7
    names = [f"x{i}" for i in range(n)]
    degrees = list(range(2, 15))
    series = {
        "isotropic": lambda d: comb(n + d, d),
        "max_interaction=2": lambda d: Basis(max_interaction=2).build(names, d).shape[0],
        "blocks (4, 3)": lambda d: Basis(blocks=[(0, 1, 2, 3), (4, 5, 6)]).build(names, d).shape[0],
        "parity even": lambda d: Basis(parity=["even"] * n).build(names, d).shape[0],
        "q-norm q=0.5": lambda d: Basis(q=0.5).build(names, d).shape[0],
        "rank-2 rotation": lambda d: comb(2 + d, d),
    }
    for (label, fn), colour in zip(series.items(), COLOURS):
        style = "-" if label != "isotropic" else "--"
        ax.plot(degrees, [fn(d) for d in degrees], style, color=colour,
                label=label, linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("degree")
    ax.set_ylabel("basis terms  $D$")
    ax.grid(True, which="major", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)


def warp_convergence(ax) -> None:
    """Approximation error against degree on tanh(20 (x - 0.3)), 1-D.

    A warp is a conformal map: it moves the singularity that sets the
    convergence rate, so the curves have different SLOPES rather than
    different offsets. The three lines are the identity, the best candidate
    the shipped scan can actually propose, and a sinh placed by hand at the
    feature, which the scan cannot reach because it only offers the centres
    -0.5, 0 and 0.5.
    """
    import numpy as np
    from numpy.polynomial import legendre as L

    from MomentEmu.warp import _make

    x = np.linspace(-1.0, 1.0, 4001)
    y = np.tanh(20.0 * (x - 0.3))
    col = np.linspace(-1.0, 1.0, 2001)
    degrees = list(range(2, 101, 2))

    def curve(warp):
        u = warp(x)
        out = []
        for d in degrees:
            V = L.legvander(u, d)
            c, *_ = np.linalg.lstsq(V, y, rcond=None)
            out.append(float(np.max(np.abs(V @ c - y))))
        return out

    lines = [
        ("identity", _make("identity", "identity", 0.0, 0.0, col), COLOURS[0]),
        ("sinh(10, centre 0.5), best the scan offers",
         _make("identity", "sinh", 10.0, 0.5, col), COLOURS[1]),
        ("sinh(8, centre 0.3), placed by hand",
         _make("identity", "sinh", 8.0, 0.3, col), COLOURS[2]),
    ]
    for label, warp, colour in lines:
        ax.semilogy(degrees, curve(warp), color=colour, label=label, linewidth=1.6)
    ax.axhline(1e-3, color="#888888", linewidth=0.8, linestyle=":")
    ax.text(degrees[1], 1.3e-3, "1e-3", fontsize=8, color="#888888")
    ax.set_xlabel("degree")
    ax.set_ylabel("max error on $[-1, 1]$")
    ax.grid(True, which="major", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8)


def active_subspace_spectrum(ax) -> None:
    """Gradient-covariance spectrum of a target with a 2-D active subspace.

    Seven parameters, two of which the target actually uses, so five
    directions are exactly dead. The comparison is the rotation source: a
    degree-3 pilot polynomial against the caller's exact Jacobian. The pilot
    leaks about 1e-4 of the spectrum into each dead direction; the exact
    Jacobian leaves them at 1e-16 and below.

    This target is NOT an example of a variance target overshooting the rank:
    its first two directions already carry 0.99974, so a 0.999 target stops at
    rank 2, which is right. The overshoot case is a shallow tail, and it is
    described in the text rather than plotted here.
    """
    import numpy as np

    from MomentEmu.rotation import active_subspace

    rng = np.random.default_rng(3)
    n, n_samples = 7, 6000
    W = rng.standard_normal((2, n))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    X = rng.uniform(-1.0, 1.0, (n_samples, n))
    U = X @ W.T
    Y = np.column_stack(
        [np.tanh(U[:, 0]) + 0.5 * U[:, 1] ** 2, np.sin(U[:, 0] * U[:, 1])]
    )

    def exact(chunk):
        u = chunk @ W.T
        c = np.cos(u[:, 0] * u[:, 1])
        d0 = np.stack([1.0 - np.tanh(u[:, 0]) ** 2, u[:, 1]], axis=1)
        d1 = np.stack([c * u[:, 1], c * u[:, 0]], axis=1)
        return np.stack([d0 @ W, d1 @ W], axis=1)

    e_pilot, _v = active_subspace(X, Y, pilot_degree=3)
    e_exact, _v = active_subspace(X, Y, jacobian=exact)
    idx = np.arange(1, n + 1)
    floor = 1e-18
    ax.bar(idx - 0.2, np.maximum(e_pilot / e_pilot.sum(), floor), width=0.4,
           color=COLOURS[1], label="degree-3 pilot")
    ax.bar(idx + 0.2, np.maximum(e_exact / e_exact.sum(), floor), width=0.4,
           color=COLOURS[0], label="exact Jacobian")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 3.0)
    ax.axvline(2.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.text(2.6, 1e-3, "true rank 2", fontsize=8, color="#888888")
    ax.set_xlabel("direction")
    ax.set_ylabel("eigenvalue share")
    ax.grid(True, axis="y", which="major", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8)
    print("   pilot shares:", np.array2string(e_pilot / e_pilot.sum(), precision=2))
    print("   exact shares:", np.array2string(e_exact / e_exact.sum(), precision=2))


def sparse_pareto(ax) -> None:
    """Held-out error against retained terms, selected set versus isotropic.

    The target is anisotropic on purpose: it needs a high degree in one
    parameter and almost none in the others, which is the asymmetry no prior
    truncation can state. Both curves turn back up, which is the part worth
    plotting: past the target's own sparsity, more terms is overfitting.
    """
    import warnings

    import numpy as np

    from MomentEmu.emulator import PolyEmu
    from MomentEmu.sparse import SparseEmu

    def target(X):
        return (np.tanh(6.0 * X[:, 0]) + 0.4 * X[:, 1] ** 2
                + 0.2 * X[:, 2] * X[:, 3]).reshape(-1, 1)

    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, (3000, 4))
    Xt = rng.uniform(-1.0, 1.0, (1000, 4))
    Y, Yt = target(X), target(Xt)
    den = float(np.sqrt(np.mean(Yt ** 2)))

    def err(pred):
        return 100.0 * float(np.sqrt(np.mean((pred - Yt) ** 2)) / den)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = []
        for d in (3, 5, 7, 9, 11):
            e = PolyEmu(X, Y, max_degree_forward=d, init_deg_forward=d,
                        RMSE_tol=0.0, verbose=0)
            iso.append((e.forward_multi_indices.shape[0],
                        err(e.forward_emulator(Xt, extrapolation="ignore"))))
        spa = []
        for k in (10, 20, 40, 80, 160, 320):
            e = SparseEmu(X, Y, degree=14, n_terms=k, max_terms=k + 50,
                          random_state=0)
            spa.append((e.multi_indices.shape[0], err(e.forward_emulator(Xt))))

    ax.plot(*zip(*iso), "o--", color=COLOURS[0], linewidth=1.6, markersize=4,
            label="isotropic basis, degree 3 to 11")
    ax.plot(*zip(*spa), "o-", color=COLOURS[2], linewidth=1.6, markersize=4,
            label="SparseEmu, degree-14 candidates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("retained terms  $D$")
    ax.set_ylabel("held-out error (%)")
    ax.grid(True, which="major", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    print("   isotropic:", iso)
    print("   sparse   :", spa)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    render("basis-growth", basis_growth)
    render("sparse-pareto", sparse_pareto)
    render("warp-convergence", warp_convergence)
    render("active-subspace-spectrum", active_subspace_spectrum)
