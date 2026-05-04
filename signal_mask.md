# Signal-mask strategy for relative-error gates

> Sourced from canoes (cosmological angular statistics package; `~/projects/AngStats`)
> as a battle-tested pattern for comparing numerical implementations whose output
> spans many orders of magnitude. Drop-in candidate to replace / refine MomentEmu's
> fractional RMSE strategy where target functions have similar dynamic range.

## The problem this solves

A naive relative-error gate like

```python
rel = np.abs(predicted - reference) / np.abs(reference)
assert rel.max() < 1e-7
```

**fails catastrophically** on outputs with wide dynamic range. Concrete example
from canoes' chi-derivative C_ℓ tensor (output spans ~1e-7 to 1e-24, ~17 decades):

```
At a deep-tail entry:
  reference (truth) = +6.17e-24   ← essentially zero, rounded by reference impl
  predicted         = -1.40e-04   ← floating-point noise, also "essentially zero"
  abs diff          =  1.40e-04   ← physically negligible
  rel diff          =  2.27e+19   ← but kills the gate
```

The reference and predicted values agree on **"this entry is zero"** but disagree
by 19 orders of magnitude on what flavour of zero. **No relative-error metric can
make sense at floating-point-noise levels**: rel diff is undefined when the
reference is at the ULP floor.

The naive fix — adding an `atol` like `np.allclose(a, b, rtol=1e-7, atol=1e-30)` —
also fails because picking the right global `atol` requires you to know the
output's dynamic range a priori, and **that range varies per slice / per sample
/ per parameter setting** in a way that's hard to encode statically.

## The signal-mask gate

The pattern (in canoes:
[`tests/integration/canoes_vs_ccl/test_step2_jax_vs_ccl.py`](https://github.com/zzhang0123/canoes/blob/main/tests/integration/canoes_vs_ccl/test_step2_jax_vs_ccl.py)):

```python
def signal_mask_max_rel(predicted, reference,
                        signal_floor_frac=1e-3,
                        absolute_floor=1e-15):
    """Max relative error filtered to entries above the signal floor.

    The 'signal floor' is set RELATIVE to the reference's own peak amplitude:
    only entries whose magnitude exceeds `signal_floor_frac × |ref|.max()`
    contribute to the max. Entries below the floor are zeroed out (treated
    as floating-point noise, not a real disagreement).

    Returns
    -------
    max_rel : float
        Max relative error among in-mask entries. 0.0 if mask is empty.
    n_above : int
        Number of entries above the floor (sanity check: should be > 0).
    """
    abs_ref = np.abs(reference)
    floor = max(signal_floor_frac * float(abs_ref.max()), absolute_floor)
    mask = abs_ref >= floor
    rel = np.where(
        mask,
        np.abs(predicted - reference) / np.maximum(abs_ref, 1e-30),
        0.0,
    )
    return float(rel.max()), int(mask.sum())
```

Then assert:

```python
max_rel, n_above = signal_mask_max_rel(predicted, reference)
assert n_above > 0, "signal mask is empty — fixture has no signal at all"
assert max_rel < 1e-7, (
    f"max rel above signal floor = {max_rel:.3e} ({n_above} entries above floor)"
)
```

## The two parameters explained

### `signal_floor_frac` (default `1e-3`)

Sets the floor as a **fraction of the reference's own peak amplitude**. With
`1e-3`, entries weaker than 0.1 % of the maximum are masked out. Three
calibration anchors:

* **`1e-3` (canoes default)** — preserves entries that contribute ≥ 0.1 % to
  any aggregated downstream quantity. ASZ 2017 (the original FFTlog cosmology
  paper) used a numerically-equivalent `ε = 1e-5` cutoff but applied it
  per-row, not per-tensor; the per-tensor 1e-3 captures the same intuition with
  less bookkeeping.
* **`1e-6`** — only mask the truly-noise-floor entries. Use when you want a
  near-strict relative-error gate AND your reference's dynamic range isn't
  too wide.
* **`1e-9`** — almost no masking, recovers near-strict rtol. Reasonable for
  outputs whose dynamic range is ≤ 3 decades.

**MomentEmu calibration tip**: pick `signal_floor_frac` so that the masked-out
entries' aggregate contribution to any user-facing observable is ≤ the rtol
gate itself. If your gate is `1e-7` and observables are at most cubic in the
tensor, set the floor at ~ `(1e-7)^(1/3) ≈ 5e-3`. Larger floors are physically
defensible if you can argue downstream observables don't depend on the masked
region.

### `absolute_floor` (default `1e-15`)

Hard lower bound to prevent the floor from collapsing to 0 when **both**
reference and predicted are essentially zero everywhere (e.g., a fixture where
all entries are at the FP precision limit). `1e-15` ≈ float64 ULP-level noise.
Should rarely fire in practice — its purpose is defensive, not statistical.

If it fires often (lots of entries below `1e-15`), your test fixture is
producing essentially zero output, which usually means the test setup is wrong
(missing fixture parameter, wrong scaling).

## Composability with fractional RMSE

The pattern composes naturally with RMSE-style metrics if you want a **mean**
behaviour rather than worst-case:

```python
def signal_mask_frac_rmse(predicted, reference,
                          signal_floor_frac=1e-3, absolute_floor=1e-15):
    abs_ref = np.abs(reference)
    floor = max(signal_floor_frac * float(abs_ref.max()), absolute_floor)
    mask = abs_ref >= floor
    if not mask.any():
        return float("nan"), 0
    rel_sq = (np.abs(predicted - reference) / abs_ref) ** 2
    rel_sq = rel_sq[mask]                           # restrict to in-mask
    return float(np.sqrt(rel_sq.mean())), int(mask.sum())
```

This is **fractional RMSE on the signal mask**. Two flavours of gate side by side:

* **`max_rel`** — pessimistic; catches ANY single entry that disagrees at
  rtol > gate. Used as a hard correctness check.
* **`frac_rmse`** — optimistic; permits some bad entries if they're rare.
  Used when occasional outliers (FP edge cases) are acceptable as long as the
  bulk agrees.

Pair both: `max_rel < 1e-5 AND frac_rmse < 1e-7` is a reasonable combined gate
for emulator validation against analytic / numerical reference.

## When NOT to use a signal mask

* **Low dynamic range outputs** (≤ 2–3 decades). Plain `np.allclose(a, b,
  rtol=...)` is sufficient and clearer.
* **Conservation-law tests** (e.g., trace preservation, sum rules). Signal mask
  hides imbalance that may matter physically. Use absolute deviation from the
  invariant value instead.
* **Tests where the worst entry IS what you care about** (e.g., max-norm
  approximation, L∞ guarantees). Signal mask is the wrong gate by definition.

## When the signal mask is misleading

The classic failure mode: reference and predicted **both** have a bug at the
same low-signal entries that produces consistent-looking values, masking
gets fooled into thinking they agree because the entries are excluded.

Example from canoes: at deep `|Im ν|` the original numba and JAX paths gave
different wrong values, but BOTH were below the signal floor. Cross-backend
parity passed even though both were wrong vs the mathematical truth. The fix
was a **third reference** (mpmath at 50 digits) used as the absolute ground
truth — see `tests/unit/test_il_vs_mpmath_ground_truth.py` in canoes for the
3-tier validation pattern (production vs reference vs mpmath at FP precision).

For MomentEmu emulator validation: if you only compare emulator vs training
data, you can be fooled. Anchor at least one validation tier against an
**independent ground truth** (analytic limits, mpmath, alternate
implementation) to catch cases where emulator + training share the same bug.

## Suggested integration into MomentEmu

If MomentEmu's current fractional RMSE strategy is something like

```python
rmse = np.sqrt(np.mean(((pred - ref) / ref) ** 2))
```

a drop-in replacement that handles wide-dynamic-range outputs is

```python
def signal_aware_rmse(pred, ref, *, signal_floor_frac=1e-3,
                      absolute_floor=1e-15):
    abs_ref = np.abs(ref)
    floor = max(signal_floor_frac * float(abs_ref.max()), absolute_floor)
    mask = abs_ref >= floor
    if not mask.any():
        return {"rmse": float("nan"), "max_rel": float("nan"),
                "n_above": 0, "n_total": int(ref.size), "floor": floor}
    abs_diff = np.abs(pred - ref)
    rel = abs_diff[mask] / abs_ref[mask]
    return {
        "rmse": float(np.sqrt(np.mean(rel ** 2))),
        "max_rel": float(rel.max()),
        "n_above": int(mask.sum()),
        "n_total": int(ref.size),
        "floor": float(floor),
    }
```

The returned dict is **diagnostic-rich on purpose**: when a gate fails the
user immediately sees how many entries were in scope, what the signal floor
was, and whether the worst entry or the bulk caused the failure. This is
critical for emulator-iteration workflows where you want to know "did my
training set get worse, or did one outlier blow up?".

The same dict can be threaded through to a per-sample / per-parameter
breakdown for emulator validation:

```python
for i, sample_id in enumerate(test_set):
    diag = signal_aware_rmse(pred[i], ref[i])
    if diag["rmse"] > 1e-7 or diag["max_rel"] > 1e-5:
        print(f"sample {sample_id}: {diag}")
```

## Cross-references

* canoes integration test:
  `tests/integration/canoes_vs_ccl/test_step2_driving_field_cl.py`
  (the original numba/CCL gate this pattern was calibrated against)
* canoes JAX-backend mirror:
  `tests/integration/canoes_vs_ccl/test_step2_jax_vs_ccl.py`
* canoes mpmath ground-truth tier:
  `tests/unit/test_il_vs_mpmath_ground_truth.py`
* canoes deep-tail diagnosis (why the signal mask is necessary):
  `docs/decisions/F0.10-asz-research-synthesis.md`
* ASZ 2017 (the formalism canoes implements; original `t_min` mask):
  arXiv:1705.05022 (App. B p. 27–28)
