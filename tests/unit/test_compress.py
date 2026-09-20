"""T-011: output compression as a layer over any fitted model.

Where it belongs was decided by measurement, not by taste. Least squares
commutes with a projection of the outputs: fitting the k leading modes of Y
and projecting the predictions of a full fit give the same model, to 3.7e-15
on a scale of 11.3. So the layer does not need to touch the fit, and does not.
That matters here because the third failure mode recorded on T-004 is an
affine transform applied for tidiness destroying the structure the model
exists to exploit -- centring the outputs cost a FactoredEmu exactly one rank.

The rank is chosen against the model's OWN error rather than a variance
target. A compression whose reconstruction error sits well below the error the
emulator already makes is close to free; one above it is not. Errors add in
quadrature, so a reconstruction at half the fit's error costs about 12
percent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from MomentEmu.compress import CompressedEmu, output_modes
from MomentEmu.emulator import PolyEmu


def _spectra(n=500, m=120, seed=0):
    """A smooth family of spectra: low rank in the outputs, by construction."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 3))
    t = np.linspace(0.0, 1.0, m)
    Y = (
        np.tanh(2.0 * X[:, :1]) * np.sin(3.0 * t)[None, :]
        + (X[:, 1:2] ** 2) * np.exp(-((t - 0.4) ** 2) / 0.05)[None, :]
        + X[:, 2:3] * t[None, :]
    )
    return X, Y


def _fit(X, Y, degree=4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PolyEmu(X, Y, init_deg_forward=degree, max_degree_forward=degree,
                       RMSE_tol=0.0, verbose=0)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_modes_reconstruct_within_the_reported_error():
    X, Y = _spectra()
    basis = output_modes(Y, rank=4)
    assert basis.modes.shape == (4, Y.shape[1])
    back = basis.expand(basis.project(Y))
    assert _rel(back, Y) == pytest.approx(basis.reconstruction_error, rel=1e-6)


def test_a_full_rank_compression_is_exact():
    X, Y = _spectra(n=200, m=20)
    basis = output_modes(Y, rank=20)
    np.testing.assert_allclose(basis.expand(basis.project(Y)), Y,
                               rtol=1e-10, atol=1e-11)


def test_the_wrapper_matches_fitting_on_the_compressed_outputs():
    """The equivalence the design rests on, through the public API."""
    X, Y = _spectra()
    model = _fit(X, Y)
    wrapped = CompressedEmu(model, X, Y, rank=5)

    basis = output_modes(Y, rank=5)
    refit = _fit(X, basis.project(Y))
    direct = basis.expand(refit.forward_emulator(X, extrapolation="ignore"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = wrapped.forward_emulator(X)
    assert _rel(got, direct) < 1e-8, _rel(got, direct)


def test_auto_rank_stays_under_the_models_own_error():
    X, Y = _spectra()
    model = _fit(X, Y)
    wrapped = CompressedEmu(model, X, Y, rank="auto")
    rep = wrapped.report()
    assert rep["reconstruction_error"] <= rep["model_error"] * rep["safety"]
    assert 1 <= rep["rank"] < Y.shape[1]
    # Quadrature: the compressed error must not exceed the two combined.
    bound = np.hypot(rep["model_error"], rep["reconstruction_error"]) * 1.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _rel(wrapped.forward_emulator(X), Y) <= bound


def test_the_report_counts_what_is_stored():
    X, Y = _spectra()
    model = _fit(X, Y)
    rep = CompressedEmu(model, X, Y, rank=6).report()
    D = model.forward_multi_indices.shape[0]
    assert rep["stored_full"] == D * Y.shape[1]
    assert rep["stored_compressed"] == D * 6 + 6 * Y.shape[1]
    assert rep["compression"] == pytest.approx(
        rep["stored_full"] / rep["stored_compressed"]
    )


def test_the_underlying_fit_is_untouched():
    """The layer wraps; it must not rewrite what it wraps."""
    X, Y = _spectra()
    model = _fit(X, Y)
    before = np.array(model.forward_coeffs, copy=True)
    CompressedEmu(model, X, Y, rank=5)
    np.testing.assert_array_equal(model.forward_coeffs, before)


def test_it_composes_with_a_sparse_fit():
    from MomentEmu.sparse import SparseEmu

    X, Y = _spectra()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SparseEmu(X, Y, degree=4, n_terms=30)
        wrapped = CompressedEmu(model, X, Y, rank=5)
        got = wrapped.forward_emulator(X)
    assert got.shape == Y.shape
    assert _rel(got, Y) < 0.2


@pytest.mark.parametrize("bad", (0, -1, 999))
def test_the_rank_is_validated(bad):
    _X, Y = _spectra(n=100, m=20)
    with pytest.raises(ValueError, match="rank"):
        output_modes(Y, rank=bad)


def test_compressing_twice_is_not_silently_allowed():
    X, Y = _spectra()
    model = _fit(X, Y)
    once = CompressedEmu(model, X, Y, rank=5)
    with pytest.raises(ValueError, match="already"):
        CompressedEmu(once, X, Y, rank=3)
