"""Versioned .npz persistence (P4.2): sklearn-free load and a fingerprint.

One ``.npz`` holds a JSON ``meta`` entry (stored as bytes) plus the
coefficient, index, scaler and box arrays.  Loading reconstructs a PolyEmu
without importing sklearn, so a deployment only needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np

FORMAT = "MomentEmu.npz"
FORMAT_VERSION = 2


class ArrayScaler:
    """Minimal StandardScaler replacement with array state (sklearn-free)."""

    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = None if scale is None else np.asarray(scale, dtype=np.float64)

    def transform(self, X):
        """Standardize X with the stored mean and scale."""
        X = np.asarray(X, dtype=np.float64)
        if self.scale_ is None:
            return X - self.mean_
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        """Undo transform()."""
        X = np.asarray(X, dtype=np.float64)
        if self.scale_ is None:
            return X + self.mean_
        return X * self.scale_ + self.mean_


def _json_bytes(obj):
    return json.dumps(obj, sort_keys=True).encode("utf-8")


def coefficient_fingerprint(coeffs, multi_indices, transform):
    """SHA-256 over the coefficients, indices and transform (P4.2)."""
    h = hashlib.sha256()
    for arr in (coeffs, multi_indices):
        a = np.asarray(arr)
        if a.dtype.kind == "f":
            # Hash the float64 value so a float32-stored file and its
            # float64-loaded form share a fingerprint.
            a = a.astype(np.float64)
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(tuple(transform)).encode("utf-8"))
    return h.hexdigest()


def _check_transform_saveable(transform):
    for t in transform:
        if not isinstance(t, str):
            raise ValueError(
                "a custom (forward, inverse) transform pair cannot be saved; "
                "use linear/log/asinh to persist an emulator."
            )


def _float32_storage_gate(emulator, gate=0.01):
    """Per-output prediction change from casting coefficients to float32 (B7).

    Compares the float32 round-trip predictions with the float64 ones on the
    stored training data and raises when any output moves by more than ``gate``
    times that output validation RMSE. Returns the per-output max difference
    (or None when the training data is unavailable).
    """
    X = getattr(emulator, "_X_data", None)
    Y = getattr(emulator, "_Y_data", None)
    if X is None or not hasattr(emulator, "forward_coeffs"):
        warnings.warn(
            "float32=True without the training data: the D1 accuracy gate "
            "cannot run and no per-output difference is stored.",
            UserWarning,
            stacklevel=3,
        )
        return None
    ref = emulator.forward_emulator(X, extrapolation="ignore")
    old = emulator.forward_coeffs
    try:
        emulator.forward_coeffs = np.asarray(old, dtype=np.float32).astype(np.float64)
        emulator._build_forward_plan()
        got = emulator.forward_emulator(X, extrapolation="ignore")
    finally:
        emulator.forward_coeffs = old
        emulator._build_forward_plan()
    diff = np.max(np.abs(got - ref), axis=0)
    if Y is not None:
        val_rmse = np.sqrt(np.mean((ref - np.asarray(Y, dtype=np.float64)) ** 2, axis=0))
    else:
        val_rmse = np.std(ref, axis=0)
    bad = np.flatnonzero(diff > gate * val_rmse)
    if bad.size:
        raise ValueError(
            f"float32 coefficients change a prediction by more than {gate:.0%} "
            f"of the validation RMSE for output(s) {bad.tolist()}; keep float64 "
            "storage."
        )
    return diff


def save_emulator(emulator, path, *, float32=False, dataset_sha256=None):
    """Write a fitted PolyEmu to a versioned .npz (P4.2)."""
    from MomentEmu import __version__

    transform = tuple(getattr(emulator, "transform", ())) or emulator._transforms()
    _check_transform_saveable(transform)
    coeff_dtype = np.float32 if float32 else np.float64
    arrays = {}
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "package_version": __version__,
        "n_params": int(emulator.n_params),
        "n_outputs": int(emulator.n_outputs),
        "transform": list(transform),
        "log_Y": bool(emulator.log_Y),
        "standardize_Y_with_std": bool(getattr(emulator, "standardize_Y_with_std", True)),
        "random_state": getattr(emulator, "random_state", None),
        "forward_degree": int(getattr(emulator, "forward_degree", -1)),
        "backward_degree": int(getattr(emulator, "backward_degree", -1)),
        "forward_RMSE": float(getattr(emulator, "forward_RMSE", float("nan"))),
        "backward_RMSE": float(getattr(emulator, "backward_RMSE", float("nan"))),
        "forward_cond_est": float(getattr(emulator, "forward_cond_est_", float("nan"))),
        "backward_cond_est": float(getattr(emulator, "backward_cond_est_", float("nan"))),
        "forward_N_train": int(getattr(emulator, "forward_N_train_", 0)),
        "backward_N_train": int(getattr(emulator, "backward_N_train_", 0)),
        "float32_coefficients": bool(float32),
        "dataset_sha256": dataset_sha256,
    }
    if hasattr(emulator, "forward_coeffs"):
        if float32:
            diff = _float32_storage_gate(emulator)
            if diff is not None:
                arrays["float32_difference"] = diff
                meta["float32_difference_checked"] = True
        arrays["forward_coeffs"] = np.asarray(emulator.forward_coeffs, dtype=coeff_dtype)
        arrays["forward_multi_indices"] = np.asarray(emulator.forward_multi_indices)
        # Hash the stored (possibly float32) coefficients so the fingerprint is
        # stable across the storage dtype.
        meta["forward_hash"] = coefficient_fingerprint(
            arrays["forward_coeffs"], emulator.forward_multi_indices, transform
        )
        if hasattr(emulator, "forward_resid_std_"):
            arrays["forward_resid_std"] = np.asarray(emulator.forward_resid_std_)
        _cf = getattr(emulator, "forward_chol_", None)
        if _cf is not None:
            arrays["forward_chol"] = np.asarray(_cf[0])
            meta["forward_chol_lower"] = bool(_cf[1])
        elif hasattr(emulator, "forward_moment_matrix_"):
            arrays["forward_moment_matrix"] = np.asarray(emulator.forward_moment_matrix_)
        _rank = getattr(emulator, "forward_rank_", None)
        if _rank is not None:
            meta["forward_rank"] = int(_rank)
        _rd = getattr(emulator, "rank_difference_", None)
        if _rd is not None:
            arrays["rank_difference"] = np.asarray(_rd)
        _sv = getattr(emulator, "forward_singular_values_", None)
        if _sv is not None:
            arrays["forward_singular_values"] = np.asarray(_sv)
    if hasattr(emulator, "backward_coeffs"):
        arrays["backward_coeffs"] = np.asarray(emulator.backward_coeffs, dtype=coeff_dtype)
        arrays["backward_multi_indices"] = np.asarray(emulator.backward_multi_indices)
        meta["backward_hash"] = coefficient_fingerprint(
            arrays["backward_coeffs"], emulator.backward_multi_indices, transform
        )
    for name, scaler in (("X", emulator.scaler_X), ("Y", emulator.scaler_Y)):
        arrays[f"scaler_{name}_mean"] = np.asarray(scaler.mean_, dtype=np.float64)
        if scaler.scale_ is not None:
            arrays[f"scaler_{name}_scale"] = np.asarray(scaler.scale_, dtype=np.float64)
    for name, box in (("X", emulator.X_box_), ("Y", emulator.Y_box_)):
        arrays[f"box_{name}_lo"] = np.asarray(box.lo, dtype=np.float64)
        arrays[f"box_{name}_hi"] = np.asarray(box.hi, dtype=np.float64)
        arrays[f"box_{name}_scale"] = np.asarray(box.scale, dtype=np.float64)
    meta["hash_"] = meta.get("forward_hash", "")
    np.savez(path, meta=_json_bytes(meta), **arrays)
    return str(path)


def load_emulator(path):
    """Load a PolyEmu from a versioned .npz without importing sklearn (P4.2)."""
    from MomentEmu.emulator import PolyEmu
    from MomentEmu.guards import DomainBox

    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data:
            raise ValueError(f"{path} is not a MomentEmu .npz (no meta entry)")
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(
                f"unknown format tag {meta.get('format')!r}; expected {FORMAT!r}"
            )
        version = int(meta.get("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {version}; this build reads {FORMAT_VERSION}"
            )
        arrays = {k: data[k] for k in data.files if k != "meta"}

    emu = object.__new__(PolyEmu)
    emu.n_params = int(meta["n_params"])
    emu.n_outputs = int(meta["n_outputs"])
    emu.transform = tuple(meta["transform"])
    emu.log_Y = bool(meta["log_Y"])
    emu.standardize_Y_with_std = bool(meta["standardize_Y_with_std"])
    emu.random_state = meta.get("random_state")
    emu.verbose = 0
    emu.loo_rmse_ = None
    emu.leverage_max_train_ = None
    emu.scaler_X = ArrayScaler(
        arrays["scaler_X_mean"], arrays.get("scaler_X_scale")
    )
    emu.scaler_Y = ArrayScaler(
        arrays["scaler_Y_mean"], arrays.get("scaler_Y_scale")
    )
    emu.X_box_ = DomainBox(
        arrays["box_X_lo"], arrays["box_X_hi"], arrays["box_X_scale"]
    )
    emu.Y_box_ = DomainBox(
        arrays["box_Y_lo"], arrays["box_Y_hi"], arrays["box_Y_scale"]
    )
    emu._inv_scale_X = 1.0 / emu.scaler_X.scale_
    _ys = emu.scaler_Y.scale_
    emu._inv_scale_Y = 1.0 / (_ys if _ys is not None else np.ones(emu.n_outputs))
    if "forward_coeffs" in arrays:
        emu.forward_coeffs = np.asarray(arrays["forward_coeffs"], dtype=np.float64)
        emu.forward_multi_indices = np.asarray(arrays["forward_multi_indices"])
        emu.forward_degree = int(meta.get("forward_degree", -1))
        emu.forward_RMSE = float(meta.get("forward_RMSE", float("nan")))
        emu.forward_cond_est_ = float(meta.get("forward_cond_est", float("nan")))
        emu.forward_N_train_ = int(meta.get("forward_N_train", 0))
        if "forward_resid_std" in arrays:
            emu.forward_resid_std_ = np.asarray(arrays["forward_resid_std"], dtype=np.float64)
        emu._build_forward_plan()
        if "forward_chol" in arrays:
            emu.forward_chol_ = (
                np.asarray(arrays["forward_chol"], dtype=np.float64),
                bool(meta.get("forward_chol_lower", False)),
            )
        elif "forward_moment_matrix" in arrays:
            emu.forward_moment_matrix_ = np.asarray(
                arrays["forward_moment_matrix"], dtype=np.float64
            )
            emu.forward_chol_ = None
        if "forward_rank" in meta:
            emu.forward_rank_ = int(meta["forward_rank"])
        if "forward_singular_values" in arrays:
            emu.forward_singular_values_ = np.asarray(arrays["forward_singular_values"])
        if "rank_difference" in arrays:
            emu.rank_difference_ = np.asarray(arrays["rank_difference"])
        if "float32_difference" in arrays:
            emu.float32_difference_ = np.asarray(arrays["float32_difference"])
        if "forward_hash" in meta:
            got = coefficient_fingerprint(
                emu.forward_coeffs, emu.forward_multi_indices, emu.transform
            )
            if got != meta["forward_hash"]:
                warnings.warn(
                    "the loaded forward coefficients do not match the stored hash_ "
                    "(the file was modified); the fingerprint changed.",
                    UserWarning,
                    stacklevel=2,
                )
    if "backward_coeffs" in arrays:
        emu.backward_coeffs = np.asarray(arrays["backward_coeffs"], dtype=np.float64)
        emu.backward_multi_indices = np.asarray(arrays["backward_multi_indices"])
        emu.backward_degree = int(meta.get("backward_degree", -1))
        emu.backward_RMSE = float(meta.get("backward_RMSE", float("nan")))
        emu.backward_cond_est_ = float(meta.get("backward_cond_est", float("nan")))
        emu.backward_N_train_ = int(meta.get("backward_N_train", 0))
        emu._build_backward_plan()
    emu.hash_ = meta.get("hash_", "")  # type: ignore[attr-defined]
    return emu


def fingerprint(emulator):
    """Recompute the coefficient fingerprint of an in-memory emulator (P4.2)."""
    parts = []
    transform = tuple(getattr(emulator, "transform", ())) or emulator._transforms()
    if hasattr(emulator, "forward_coeffs"):
        parts.append(coefficient_fingerprint(
            emulator.forward_coeffs, emulator.forward_multi_indices, transform
        ))
    if hasattr(emulator, "backward_coeffs"):
        parts.append(coefficient_fingerprint(
            emulator.backward_coeffs, emulator.backward_multi_indices, transform
        ))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
