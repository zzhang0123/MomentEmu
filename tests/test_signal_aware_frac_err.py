"""Tests for the signal-mask aware fractional-error diagnostic.

The pattern under test is implemented by
:func:`MomentEmu.signal_aware_frac_err`. These tests pin the four
behaviours that motivated the refactor:

1. Narrow-dynamic-range outputs use the plain relative-error path (the
   historical behaviour, modulo the new ULP-aware floor).
2. Wide-dynamic-range outputs apply the signal mask, ignoring deep-tail
   FP-noise entries that would otherwise blow up the gate.
3. Deep-tail FP-noise entries (ref ~ 1e-24, pred ~ 1e-4): naive rel-err
   is huge, signal-mask rel-err is 0.
4. Per-output mode picks each strategy independently when the columns sit
   in different DR regimes.
"""

import numpy as np
import pytest

from MomentEmu.MomentEmu import signal_aware_frac_err
from MomentEmu.PolyEmu import _report_frac_err


def test_narrow_dr_uses_plain_relative_error() -> None:
    rng = np.random.default_rng(0)
    ref = rng.uniform(1.0, 10.0, size=(50, 3))      # DR ~ 1 decade
    pred = ref * (1.0 + 1e-8 * rng.standard_normal(ref.shape))

    diag = signal_aware_frac_err(pred, ref)

    assert np.all(diag["strategy"] == "plain_rel")
    assert diag["n_above"] == diag["n_total"]
    assert diag["max_rel"] < 1e-6
    # Floor follows the spec's single formula even on the plain_rel branch:
    # max(signal_floor_frac * peak, absolute_floor). For ref ~ U(1, 10) and
    # signal_floor_frac=1e-3 the floor sits well above absolute_floor.
    peak_per_col = np.abs(ref).max(axis=0)
    np.testing.assert_allclose(diag["floor"], 1e-3 * peak_per_col)


def test_wide_dr_triggers_signal_mask() -> None:
    # Reference spans ~7 decades; predicted matches at the top, FP-noise at the tail.
    n = 200
    ref = np.geomspace(1e-1, 1e-8, n).reshape(-1, 1)
    pred = ref.copy()
    # Inject FP-noise disagreement only in the deep tail (last 20%).
    tail = slice(int(0.8 * n), None)
    pred[tail, 0] = ref[tail, 0] + 1e-12             # absolute, but huge relative

    diag = signal_aware_frac_err(pred, ref, signal_floor_frac=1e-3)

    assert diag["strategy"].tolist() == ["signal_mask"]
    assert diag["dr_decades"][0] >= 6.0
    assert diag["n_above"] < diag["n_total"]          # tail is masked out
    assert diag["max_rel"] < 1e-6                     # in-mask agreement is tight


def test_deep_tail_fp_noise_example_from_signal_mask_md() -> None:
    # The motivating example: ref at ~1e-24, pred at floating-point noise ~1e-4.
    # Naive rel-err blows to ~1e+19; signal-mask treats the entry as below floor.
    # Shape is (4, 1): a single output column observed at four samples; the
    # last sample sits in the deep tail.
    ref = np.array([[1.0], [2.0], [3.0], [6.17e-24]])     # peak = 3.0
    pred = np.array([[1.0], [2.0], [3.0], [-1.40e-4]])

    naive_rel = float(np.max(np.abs(pred - ref) / (np.abs(ref) + 1e-30)))
    assert naive_rel > 1e10                                # confirms the trap

    diag = signal_aware_frac_err(pred, ref, signal_floor_frac=1e-3)
    # DR across the column is enormous, signal mask must engage.
    assert diag["strategy"][0] == "signal_mask"
    # The deep-tail entry is below floor; in-mask entries are exact.
    assert diag["n_above"] == 3
    assert diag["max_rel"] == 0.0
    # max_rel == 0 means there is no meaningful "worst" entry; argmax is None.
    assert diag["argmax"] is None

    # Perturb an in-mask row by 5 % to confirm the masking does not zero out
    # genuine in-mask errors. A bug that suppresses ALL relative errors (not
    # just the off-mask ones) would make this assertion fail.
    pred_perturbed = pred.copy()
    pred_perturbed[0, 0] = 1.05                            # 5 % relative error on row 0
    diag2 = signal_aware_frac_err(pred_perturbed, ref, signal_floor_frac=1e-3)
    assert diag2["n_above"] == 3                            # mask is unchanged
    np.testing.assert_allclose(diag2["max_rel"], 0.05, atol=1e-12)
    assert diag2["argmax"] == (0, 0)                        # worst entry is the perturbed one


def test_per_output_mixed_dr_picks_strategy_per_column() -> None:
    # Column 0: narrow DR (1 decade). Column 1: wide DR (8 decades).
    n = 100
    col_narrow = np.linspace(1.0, 5.0, n)
    col_wide = np.geomspace(1.0, 1e-8, n)
    ref = np.column_stack([col_narrow, col_wide])
    pred = ref.copy()
    # Disturb only the wide-DR tail at FP-noise level.
    pred[-10:, 1] = ref[-10:, 1] + 1e-12

    diag = signal_aware_frac_err(pred, ref, signal_floor_frac=1e-3)

    assert diag["strategy"].tolist() == ["plain_rel", "signal_mask"]
    assert diag["max_rel"] < 1e-6                     # tail is masked, columns clean
    # Narrow column sees full count; wide column has tail masked out.
    assert diag["n_above"] < diag["n_total"]


def test_empty_signal_returns_nan_with_zero_count() -> None:
    ref = np.zeros((10, 2))
    pred = np.full_like(ref, 1e-20)

    diag = signal_aware_frac_err(pred, ref)

    assert diag["n_above"] == 0
    assert np.isnan(diag["max_rel"])
    assert np.isnan(diag["rmse"])
    assert diag["argmax"] is None


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        signal_aware_frac_err(np.zeros((3, 2)), np.zeros((3, 3)))


def test_negative_parameter_raises() -> None:
    ref = np.array([[1.0, 2.0]])
    pred = ref.copy()
    with pytest.raises(ValueError, match="non-negative"):
        signal_aware_frac_err(pred, ref, signal_floor_frac=-1.0)


def test_report_frac_err_handles_none_argmax(capsys) -> None:
    """Regression guard: _report_frac_err must not index into ref/pred when
    every in-mask entry matches exactly (argmax is None). Indexing ``ref[None]``
    in NumPy is silently ``ref[np.newaxis]``, which would dump the whole array.
    """
    # Wide-DR fixture, perfect prediction -> n_above > 0 but max_rel == 0.
    ref = np.array([[1.0], [2.0], [3.0], [6.17e-24]])
    pred = ref.copy()

    diag = _report_frac_err("Forward", pred, ref)

    # Sanity: we are exercising the n_above > 0, max_rel == 0 branch.
    assert diag["n_above"] == 3
    assert diag["max_rel"] == 0.0
    assert diag["argmax"] is None

    captured = capsys.readouterr().out
    # The regression would dump the entire ref array; the fix prints a
    # short message that names the in-mask count without indexing.
    assert "all in-mask entries match exactly" in captured
    assert "in-mask entries: 3/4" in captured
    # Belt-and-suspenders: the array repr must not appear in the output.
    assert "6.17e-24" not in captured


def test_per_output_false_reduces_to_global_strategy() -> None:
    # Two columns: narrow + wide. With per_output=False, global DR is wide,
    # so a single "signal_mask" decision applies to the whole tensor.
    col_narrow = np.linspace(1.0, 5.0, 50)
    col_wide = np.geomspace(1.0, 1e-8, 50)
    ref = np.column_stack([col_narrow, col_wide])
    pred = ref.copy()

    diag = signal_aware_frac_err(pred, ref, per_output=False)

    assert isinstance(diag["strategy"], str)
    assert diag["strategy"] == "signal_mask"
    assert isinstance(diag["floor"], float)
    assert isinstance(diag["dr_decades"], float)
