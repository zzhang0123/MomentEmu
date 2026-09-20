"""Output compression as a layer over a fitted model, not inside one exporter.

A spectrum sampled at many points is nearly low rank across those points, and
the emulator pays for every one of them: the coefficient matrix is ``D x m``.
Keeping ``k`` output modes makes it ``D x k`` plus a ``k x m`` map. On the
21cmGEM backend the reporter's own SVD took 2,173,369 stored array elements to
131,946.

Where it belongs was decided by measurement. Least squares commutes with a
projection of the outputs: fitting the ``k`` leading modes of ``Y`` and
projecting the predictions of a full fit give the same model, agreeing to
3.7e-15 on a scale of 11.3. So compression does not need to touch the fit,
and this layer does not. That restraint is not decorative -- the third failure
mode recorded on T-004 is an affine transform applied for numerical tidiness
destroying the structure the model exists to exploit, found three times in
this work, once as centring the outputs costing a FactoredEmu exactly one
rank.

Measured on the 21cmGEM benchmark, a degree-7 fit of 451-point spectra whose
uncompressed figure of merit is 2.7145 percent on 1,399,002 stored elements:

    rank   reconstruction   FoM %    stored   smaller
       4        1.454e-01  4.2643    14,212     98.4x
       8        3.601e-02  2.8406    28,424     49.2x
      16        3.445e-03  2.7152    56,848     24.6x
      32        3.390e-04  2.7145   113,696     12.3x

Rank 16 costs 0.03 percent of the figure of merit for 24.6 times less
storage. The ``safety = 0.5`` default chose rank 9 there, 2.4 percent worse
and 43.8 times smaller; lower it for a tighter trade.

The quadrature estimate is systematically pessimistic, by about a factor 3 on
that target: at rank 9 it predicts 7.4 percent and the measured cost is 2.4.
The two errors are not independent -- the modes the SVD discards are ones the
polynomial fits poorly anyway -- so the bound is safe to rely on and loose.

It also does not belong in ``JaxEmulator.from_sparse``, which is where it
would land if added on demand. Compression is orthogonal to both axes that
exporter varies: it applies to a dense fit as much as a sparse one, and to
every backend. One constructor holding it would give the backends three
chances to implement it differently, which is the T-007 failure.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from MomentEmu.guards import as_float64, check_finite

#: Reconstruction error allowed, as a fraction of the error the emulator
#: already makes. Errors add in quadrature, so 0.5 costs about 12 percent.
DEFAULT_SAFETY = 0.5


class OutputBasis(NamedTuple):
    """``k`` leading output modes, the mean they are taken about, and the cost."""

    mean: np.ndarray            # (m,)
    modes: np.ndarray           # (k, m), orthonormal rows
    reconstruction_error: float
    singular_values: np.ndarray

    def project(self, Y: Any) -> np.ndarray:
        """(N, m) outputs to (N, k) mode amplitudes."""
        return (np.asarray(Y, dtype=np.float64) - self.mean) @ self.modes.T

    def expand(self, Z: Any) -> np.ndarray:
        """(N, k) mode amplitudes back to (N, m) outputs."""
        return np.asarray(Z, dtype=np.float64) @ self.modes + self.mean


def output_modes(Y: Any, rank: Any = "auto", *, target: float | None = None,
                 safety: float = DEFAULT_SAFETY) -> OutputBasis:
    """Leading output modes of ``Y``, by SVD about the mean.

    ``rank="auto"`` needs ``target``: the smallest rank whose reconstruction
    error falls at or below ``safety * target`` is taken. Pass the error the
    emulator itself makes, so the choice is made against what the model is
    already losing rather than against a variance share -- a variance target
    was measured on T-004 to overshoot badly, returning rank 6 where rank 2
    was smaller AND more accurate.
    """
    Y = as_float64(np.asarray(Y), "Y")
    check_finite(Y, "Y")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    m = int(Y.shape[1])
    mean = Y.mean(axis=0)
    centred = Y - mean
    _U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    total = float(np.linalg.norm(centred))
    tail = np.sqrt(np.maximum(np.cumsum(s[::-1] ** 2)[::-1], 0.0))
    # errors[k] is the relative error left after keeping k modes.
    errors = np.concatenate([tail, [0.0]]) / (total if total > 0.0 else 1.0)

    if isinstance(rank, str):
        if rank != "auto":
            raise ValueError(f"rank must be an int or 'auto', got {rank!r}")
        if target is None:
            raise ValueError("rank='auto' needs target=, the error to stay under")
        ceiling = float(safety) * float(target)
        k = int(np.argmax(errors <= ceiling))
        k = max(1, min(k, m))
    else:
        k = int(rank)
        if not 1 <= k <= m:
            raise ValueError(f"rank must be between 1 and {m}, got {k}")
    return OutputBasis(mean, np.ascontiguousarray(Vt[:k]), float(errors[k]), s)


class ExportPayload(NamedTuple):
    """What a backend needs for the ``(k, m)`` stage after its GEMM."""

    coeffs: np.ndarray          # (D, k), the model's folded coefficients projected
    modes: np.ndarray           # (k, m)
    offset: np.ndarray          # (m,)


class CompressedEmu:
    """A fitted forward model with its outputs carried in ``k`` modes.

    Wraps rather than refits, because the two are the same model. Prediction
    is ``(Phi C) V + mean`` with ``C`` the wrapped model's coefficients
    projected onto the modes.
    """

    def __init__(self, model: Any, X: Any, Y: Any, rank: Any = "auto", *,
                 safety: float = DEFAULT_SAFETY) -> None:
        if isinstance(model, CompressedEmu):
            raise ValueError(
                "this model is already compressed; wrapping it again would "
                "compose two output maps and report the storage of neither"
            )
        X = as_float64(np.asarray(X), "X")
        Y = as_float64(np.asarray(Y), "Y")
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.model = model
        pred = _predict(model, X)
        denom = float(np.linalg.norm(Y)) or 1.0
        self.model_error = float(np.linalg.norm(pred - Y)) / denom
        self.safety = float(safety)
        self.basis = output_modes(Y, rank, target=self.model_error,
                                  safety=self.safety)
        self.rank = int(self.basis.modes.shape[0])
        self.n_outputs = int(Y.shape[1])

    def forward_emulator(self, X: Any, **kwargs: Any) -> np.ndarray:
        """Predict in the original output space, through the modes."""
        full = _predict(self.model, X, **kwargs)
        return self.basis.expand(self.basis.project(full))

    @property
    def n_params(self) -> int:
        """Input dimensions, the name the backends read."""
        return int(getattr(self.model, "n_params", 0)) or _term_width(self.model)

    def export_payload(self) -> ExportPayload:
        """``(coeffs, modes, offset)`` for a backend, derived in ONE place.

        With ``Cf`` the wrapped model's folded coefficients, prediction is
        ``Phi Cf``; compression then gives ``((Phi Cf - mean) V^T) V + mean``.
        Fold the constant into the projected coefficients and it is
        ``(Phi C_k) V + mean`` with ``C_k = Cf V^T``, so the backend stores
        ``D x k`` and ``k x m`` instead of ``D x m``.

        Compression is applied AFTER the model's output transform, so it folds
        into the coefficients only when that transform is linear. A log or
        asinh output is refused here rather than exported as a different
        model; the numpy layer still compresses it, after the transform.

        Three backends deriving this separately is the T-007 failure, so they
        all read it from here.
        """
        transforms = _output_transforms(self.model)
        bad = sorted({t for t in transforms if t != "linear"})
        if bad:
            raise NotImplementedError(
                f"the output transform {bad} is not linear, so compression "
                f"cannot fold into the coefficients: it is applied after the "
                f"transform, and the backends apply the transform after the "
                f"GEMM. The numpy layer still compresses this model."
            )
        Cf = np.asarray(_folded_coefficients(self.model), dtype=np.float64)
        V = self.basis.modes
        mi = np.asarray(getattr(self.model, "forward_multi_indices",
                                getattr(self.model, "multi_indices", None)))
        C_k = Cf @ V.T
        shift = self.basis.mean @ V.T                     # (k,)
        const = np.flatnonzero(~mi.any(axis=1))
        if not const.size:
            raise NotImplementedError(
                "the basis has no constant term, so the compression offset "
                "cannot be folded into the coefficients"
            )
        C_k = C_k.copy()
        C_k[const[0]] -= shift
        return ExportPayload(C_k, np.ascontiguousarray(V),
                             np.asarray(self.basis.mean, dtype=np.float64))

    def generate_forward_symb_emu(self, variable_names: Any = None) -> list:
        """Expressions in the original output space, through the modes."""
        pay = self.export_payload()
        inner = self.model
        from MomentEmu.emulator import symbolic_polynomial_expressions

        mi = np.asarray(getattr(inner, "forward_multi_indices",
                                getattr(inner, "multi_indices", None)))
        modes = symbolic_polynomial_expressions(
            pay.coeffs, mi,
            variable_names=variable_names,
            input_means=inner.scaler_X.mean_,
            input_vars=inner.scaler_X.var_,
            family=getattr(inner, "basis_kind", "monomial"),
        )
        import sympy as sp

        out = []
        for j in range(pay.modes.shape[1]):
            expr = sp.Float(float(pay.offset[j]), 17)
            for i, mode in enumerate(modes):
                w = float(pay.modes[i, j])
                if w != 0.0:
                    expr = expr + sp.Float(w, 17) * mode
            out.append(expr)
        return out

    def report(self) -> dict:
        """The rank, both errors it was chosen from, and what it saves."""
        terms = _term_count(self.model)
        full = terms * self.n_outputs
        compressed = terms * self.rank + self.rank * self.n_outputs
        return {
            "rank": self.rank,
            "n_outputs": self.n_outputs,
            "n_terms": terms,
            "model_error": self.model_error,
            "reconstruction_error": self.basis.reconstruction_error,
            "safety": self.safety,
            "stored_full": full,
            "stored_compressed": compressed,
            "compression": full / compressed if compressed else float("nan"),
        }

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"<CompressedEmu rank={r['rank']} of {r['n_outputs']} outputs, "
            f"{r['compression']:.1f}x smaller, reconstruction "
            f"{r['reconstruction_error']:.2e} against model "
            f"{r['model_error']:.2e}>"
        )


def _predict(model: Any, X: Any, **kwargs: Any) -> np.ndarray:
    """Call a fitted model, tolerating estimators without extrapolation=."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.atleast_2d(model.forward_emulator(X, **kwargs))
        except TypeError:
            return np.atleast_2d(model.forward_emulator(X))


def _term_width(model: Any) -> int:
    """Input dimensions read off whatever index set the model carries."""
    for name in ("forward_multi_indices", "multi_indices"):
        mi = getattr(model, name, None)
        if mi is not None:
            return int(np.asarray(mi).shape[1])
    return 0


def _output_transforms(model: Any) -> tuple:
    """Per-output transform names, defaulting to linear."""
    fn = getattr(model, "_transforms", None)
    if callable(fn):
        return tuple(fn())
    t = getattr(model, "transform", None)
    if t is not None:
        return tuple(t)
    return ()


def _folded_coefficients(model: Any) -> np.ndarray:
    """Coefficients with the model's own output affine already folded in."""
    folded = getattr(model, "forward_coeffs_folded", None)
    if folded is not None:
        return np.asarray(folded, dtype=np.float64)
    from MomentEmu.monomials import fold_output_affine

    mi = np.asarray(getattr(model, "forward_multi_indices",
                            getattr(model, "multi_indices", None)))
    return fold_output_affine(
        np.asarray(model.coefficients, dtype=np.float64), mi,
        np.asarray(model.mean_Y_, dtype=np.float64),
        np.asarray(model.scale_Y_, dtype=np.float64),
    )


def _term_count(model: Any) -> int:
    """Retained basis terms, whichever estimator produced them."""
    for name in ("forward_multi_indices", "multi_indices"):
        mi = getattr(model, name, None)
        if mi is not None:
            return int(np.asarray(mi).shape[0])
    inner = getattr(model, "emulator", None)
    return _term_count(inner) if inner is not None else 0
