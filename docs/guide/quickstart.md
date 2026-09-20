# Quick start

Everything below runs as written. The numbers are the ones it produced.

## Fit

```python
import numpy as np
from MomentEmu import PolyEmu

rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, (2000, 3))

a = np.array([1.0, -0.5, 2.0])
def target(X):
    return np.column_stack([np.tanh(X @ a), (X ** 2).sum(axis=1)])

emu = PolyEmu(X, Y := target(X))

print(emu.forward_degree)                      # 9
print(emu.forward_multi_indices.shape[0])      # 220
```

`X` is `(N, n)` and `Y` is `(N, m)`; a 1-D `Y` is reshaped to one column. The
constructor sweeps the degree upward, refitting incrementally, and stops when
the held-out RMSE reaches `RMSE_tol` (default `0.01`). Degree 9 in 3 parameters
is 220 of the 220 isotropic terms, so nothing was truncated here.

It also emits

```text
UserWarning: auto-capped max_degree_forward at 16 for n_params = 3,
N_train = 2000 and fill factor 2 (basis_size <= N_train / 2)
```

which is the sweep refusing to go past the degree the sample count supports:
degree 16 needs `C(3+16, 16) = 969` columns and degree 17 needs 1,140, against
`N/2 = 1000`.

## Predict

```python
X_new = rng.uniform(-1.0, 1.0, (500, 3))
pred = emu.forward_emulator(X_new)

truth = target(X_new)
rel = np.sqrt(np.mean((pred - truth) ** 2)) / np.sqrt(np.mean(truth ** 2))
print(f"{rel:.3e}")                            # 5.324e-03
```

A point outside the training box warns rather than extrapolating quietly:

```text
ExtrapolationWarning: input lies outside the training box; parameter 0:
1 of 500 row(s) outside [-0.99962, 0.999834], farthest 0.0% of the range beyond
```

The percentage is how far past the edge the worst row sits. Pass
`extrapolation="ignore"` to silence it once you have decided the margin is
acceptable.

## Derivatives

The model is a polynomial, so its Jacobian is exact for the polynomial and
approximate for the target, by more than the values are:

```python
J = emu.jacobian(X_new[:1])                    # (1, m, n)
```

| | dY0/dx0 | dY0/dx1 | dY0/dx2 |
|---|---|---|---|
| target | 0.11141 | -0.05571 | 0.22282 |
| emulator | 0.11380 | -0.06004 | 0.24625 |

Values agreed to 5e-3 relative at the same degree. Differentiation amplifies
the truncation error, so a fit tuned on values is not automatically tight on
gradients; raise the degree, or lower `RMSE_tol`, if the gradients are what you
need.

## Save and load

```python
from MomentEmu.io import save_emulator, load_emulator

save_emulator(emu, "emu.npz")
again = load_emulator("emu.npz")
```

The file holds the coefficients, the index set and the scalers, not the training
data. [Save, load and cite](persistence.md) covers the fingerprint that ties a
saved emulator to the data it was fitted on.

## Next

The fit above chose only the degree. Two directions from here:

- [`recommend(X, Y)`](recommend.md) chooses the estimator, the coordinates and
  the basis family as well, by fitting candidates and scoring them.
- [Moment projection](method.md) is what the constructor actually solved.
