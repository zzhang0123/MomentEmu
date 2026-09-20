# Reading the model

A fitted emulator is a coefficient array in a named index set, so the
structure of the target can be read back out of it. Every diagnostic here is
a projection of the fit, not a fresh study of the data.

## What the fit already carries

| attribute | meaning |
|---|---|
| `loo_rmse_` | exact leave-one-out RMSE, the quantity the degree sweep stopped on |
| `leverage_max_train_` | the largest $h_i$; at 1 the leave-one-out error is undefined for that row |
| `forward_cond_est_` | $\operatorname{cond}(M)$ at the chosen degree |
| `forward_RMSE_per_output_` | per-output training RMSE |
| `forward_degree`, `forward_multi_indices` | the degree and the index set actually fitted |
| `forward_sweep_incremental_` | whether the sweep bordered the previous moment matrix rather than rebuilding |

A high `leverage_max_train_` is worth more attention than it usually gets: it
says the fit is being carried by single points.

## Which parameters interact

```python
emu.interaction_graph()
```

On a target $\tanh(x_0 + 0.8x_1 + 1.2x_0x_1) + 0.5x_2 + 0.7x_3^2$ at 4
parameters, degree 6:

```text
matrix                        blocks
[[0.    0.133 0.    0.   ]    ((0, 1), (2,), (3,))
 [0.133 0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]]
```

which is the structure the target was built with. `blocks` is what
[`Basis(blocks=...)`](basis.md) wants.

The entry for a pair counts every term whose support contains both
parameters, so a pure three-way coupling registers on all three of its pairs.
The threshold is a **variance** share, the square of the amplitude share. Two
limits worth knowing:

- The projection only sees interactions it can represent. An even coupling
  such as $x_i^2x_j^2$ needs degree 4 before it appears at all.
- A projection explaining less than `min_explained` of $\operatorname{Var}(Y)$
  warns, because a graph read off a fit that does not describe the data
  describes nothing.

Output columns whose variation is at the rounding level of their own
magnitude are excluded and reported in `degenerate_outputs`. Before that
guard, one constant column merged every parameter into a single block,
because its Legendre coefficients are noise and normalising them by their own
sum gave shares of order 0.1.

## What degree each parameter needs

```python
emu.degree_profile()
```

On the same fit:

```text
max_degree  (6, 6, 2, 2)
parity      (None, None, None, 'even')
basis       <a ready Basis to pass back in>
```

It found that $x_3$ carries only even powers, which is true by construction.
`max_degree` is the highest variance-carrying power at tolerance `tol`, so it
reads 2 rather than 1 for the linear $x_2$: a power whose share is at the
noise floor is not distinguishable from one that is absent.

Only `"even"` is inferred, never `"odd"`. Odd parity would require the target
to be globally odd in that parameter, which the powers alone cannot
establish. Parity uses a **relative** odd-versus-even test rather than an
absolute tolerance, because the odd share of a genuinely even parameter is
sampling noise whose size depends on $N$.

## How variance is apportioned

```python
emu.sobol_report()
```

```text
S1  [0.356 0.231 0.184 0.096]
ST  [0.489 0.364 0.184 0.096]
```

`S1` is the first-order share, `ST` the total. The gap between them is the
interaction signature: $x_0$ and $x_1$ have $S_T > S_1$ and are the pair the
graph found, while $x_2$ and $x_3$ have $S_T = S_1$ exactly and enter alone.
`top_pairs` names the largest pairs of support exactly two.

!!! warning "Sobol indices are box-uniform quantities in the fitted coordinates"

    They are defined against a uniform measure on the training box. A design
    that is not box-uniform, or an emulator fitted after a warp or a
    rotation, gives indices for the **fitted** coordinates and not for the
    user's parameters: a design uniform in $\theta$ is not uniform in
    $w(\theta)$. `warn_uniform` fires when the design does not look uniform.
    On a `WarpedEmu` or an `ActiveSubspaceEmu`, reading these off the inner
    emulator answers a question about the warped or rotated axes.

## Trading rank against budget

```python
from MomentEmu.rotation import scan_rank

scan_rank(X, Y, ranks=[1, 2, 3, 4], degree=10, X_test=Xt, Y_test=Yt)
```

One row per fit with `rank`, `degree`, `n_terms`, `variance_share` and `fom`,
sorted by `n_terms`. This is the honest way to pick a rank, because the
spectrum alone overshoots. See [Better coordinates](coordinates.md).

## The report() dicts

| class | `report()` carries |
|---|---|
| `WarpedEmu` | one spec per axis, the held-out `gain`, the basis size |
| `ActiveSubspaceEmu` | the spectrum, its shares, the retained `rank` and `V` |
| `PreconditionedEmu` | the chosen `order`, the per-candidate `scores`, `dimensions`, `fitted_degree`, `cap_terms` |
| `SparseEmu` | the retained index set |
| `FactoredEmu` | `restart_residuals`, showing how close the ALS run came to failing |
| `Recommendation` | the chosen config, every candidate with its score, and the stages that were cut |

`PreconditionedEmu.report()["scores"]` is the one to read when the chosen
order surprises you: it says what the alternatives scored rather than
asserting the winner.

## The prior box

`forward_emulator` checks its input against the training box and warns:

```text
ExtrapolationWarning: input lies outside the training box; parameter 1:
1 of 500 row(s) outside [-0.99962, 0.999834], farthest 0.0% of the range beyond
```

`extrapolation="ignore"` silences it, `"raise"` makes it an error. What the
guard cannot see is the training **density**. A design that is log-spaced in
some coordinates and gridded in others has points well inside the box and far
outside the region the fit was constrained by, and the polynomial is
unconstrained there while the guard stays quiet.

The basis family changes what happens past the edge, and not in the direction
most people expect: see
[the note in Accuracy and conditioning](numerics.md#choosing-a-basis-family).

## The error band

```python
pred, std = emu.forward_emulator(X_new, return_std=True)
```

`std` is $s_j\sqrt{1 + h(x)}$ in physical units, where $s_j$ is the fitted
residual scale and $h$ the leverage. It is a **noise** band: exact for iid
noise, and it treats whatever residual the fit leaves as though it were iid
noise.

Measured on a degree-6 fit of a sharper target, 4,000 held-out points: 64.9
percent fell within one reported standard deviation against a nominal 68, and
the band was nearly flat across the box, 0.124 in the interior against 0.127
near an edge. It tracked the residual scale on that target.

It does not follow that it bounds the model error. The band is derived from
the spread of the residuals, so it cannot know about error that is systematic
rather than scattered, and the implementation records that it undercovers
model error. Use `loo_rmse_` or a held-out score for how good the fit is, and
`std` for how much the noise moves a single prediction.
