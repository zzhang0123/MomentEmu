# Migrating from the flat module

Before 2.0 the package was two flat modules. They still import and still
work; each emits a `DeprecationWarning` and is removed in 3.0.0.

| old | new |
|---|---|
| `MomentEmu.PolyEmu` (module) | `MomentEmu.emulator` |
| `MomentEmu.MomentEmu` (module) | `MomentEmu.core` |

```python
from MomentEmu import PolyEmu            # the class: unchanged, still works
from MomentEmu.emulator import PolyEmu   # the same class, explicit
```

`from MomentEmu import PolyEmu` gives the **class** and always has. The
package routes attribute access so that importing the deprecated
`MomentEmu.PolyEmu` module cannot shadow it, which is what keeps pre-2.0
pickles loading.

## Renames

| old | new | note |
|---|---|---|
| `foward_degree` | `forward_degree` | the original was a typo |
| `predictive_mse_aic_bic` | `predictive_rmse_aic_bic` | it always returned the RMSE |

## Removed in 3.0.0

`filter_modes`, and the constructor arguments `cross_validation`,
`dim_reduction` and `per_mode_thres`.

The positional `PolyEmu(X, Y, ...)` constructor is the supported path and is
not going anywhere.

## Old pickles

A `PolyEmu` or `Basis` pickled by an earlier version loads. Two guards make
that work: `Basis.__setstate__` backfills fields that did not exist when the
object was written, and the emulator rebuilds its plan and cached inverses on
first use if they are absent. A pickle written before `basis_kind` existed is
read as `monomial`.

Prefer [`save_emulator`](persistence.md) over pickling for anything you
intend to keep. It is versioned, it does not execute code on load, and it
carries the metadata a pickle does not.
