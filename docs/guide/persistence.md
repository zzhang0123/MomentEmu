# Save, load and cite an emulator

```python
from MomentEmu.io import save_emulator, load_emulator, fingerprint

save_emulator(emu, "emu.npz")
again = load_emulator("emu.npz")
```

A float64 round trip reproduces predictions exactly: measured
`max|diff| = 0.00e+00` over 500 fresh points.

## What the file holds

```text
forward_coeffs  forward_multi_indices  forward_chol  forward_resid_std
scaler_X_mean   scaler_X_scale         scaler_Y_mean  scaler_Y_scale
box_X_lo  box_X_hi  box_X_scale  box_Y_lo  box_Y_hi  box_Y_scale
meta
```

The coefficients, the index set, the scalers, the training box and the
metadata. **Not the training data.** The stored model is $c$, which is the
point of the moment formulation: the design was summarised into $M$ and $\nu$
once and is not needed again.

`meta` carries the format version, the package version, `n_params`,
`n_outputs`, the transform, the fitted degrees, the training RMSE, the
condition estimate and `forward_N_train`, so a file can be read without the
code that wrote it.

## The fingerprint

```python
fingerprint(emu)          # sha256 over coefficients, indices and transform
```

It identifies the model rather than the file. A float32-stored emulator and
its float64 original share a fingerprint, because the hash is taken over the
float64 values.

`save_emulator(..., dataset_sha256=...)` records which dataset a model was
fitted on. It is the field to fill in when the emulator will outlive the
session that produced it; nothing else in the file says what it was trained
against.

## float32 storage, and the gate on it

```python
save_emulator(emu, "emu.npz", float32=True)
```

Casting the coefficients to float32 is checked, not assumed. The gate
re-predicts the stored training data from the cast coefficients and refuses
when any output moves by more than 1 percent of **that output's** validation
RMSE:

```text
ValueError: float32 coefficients change a prediction by more than 1% of the
validation RMSE for output(s) [1]; keep float64 storage.
```

!!! warning "An output the model fits exactly cannot pass the gate"

    The tolerance is relative to each output's own RMSE, so an output the
    polynomial represents exactly drives the threshold to zero. Measured: a
    two-output fit whose per-output RMSE was `[1.06e-01, 2.49e-13]` was
    refused on output 1, because 1 percent of $2.5\times10^{-13}$ is below
    float32 resolution by a wide margin. The same fit with both outputs
    carrying a real residual, `[0.29, 0.01]`, passed.

    This is the gate working as specified rather than a defect, but it means
    `float32=True` can be refused on a model that is perfectly healthy. Drop
    the exactly-fitted output, or keep float64.

How much the file shrinks depends on how much of it is coefficients. The
coefficient block is `n_terms x n_outputs`, so on a many-output target it
dominates and float32 roughly halves the file; on a two-output fit at 210
terms the measured saving was nothing at all, 367,663 against 366,302 bytes,
because the index set and the Cholesky factor carry the rest.

Without the training data in memory the gate cannot run, and
`save_emulator` warns rather than casting silently.

## Citing

`CITATION.cff` in the repository carries the citation metadata. The method is
Zhang (2025), [arXiv:2507.02179](https://arxiv.org/abs/2507.02179). When a
result depends on a particular fit, quote the `fingerprint` alongside the
citation: the paper identifies the method and the fingerprint identifies the
model.
