# Basis specification

The default emulator uses the isotropic total-degree index set
`A = {alpha : |alpha|_1 <= d}`, whose size is `C(p+d, d)` for `p` parameters.
That count grows fast: `p=20, d=6` is 230,230 terms, and MomentEmu stores the
moment matrix `M = Phi^T Phi` densely, so the matrix alone is 404 GiB.

`Basis` replaces that index set with a smaller one. All constraints compose by
intersection.

| constraint | meaning |
|---|---|
| `degree` | total degree bound (the default isotropic set) |
| `blocks` | every support lies inside one block of a partition |
| `max_interaction` | at most this many parameters active per term |
| `q`, `weights` | weighted q-norm bound instead of the 1-norm |
| `per_parameter` | per-parameter degree cap |
| `parity` | per-parameter `"even"`/`"odd"` power constraint |
| `groups` | total degree cap inside a group of parameters |

```python
from MomentEmu.basis import Basis
from MomentEmu.emulator import PolyEmu

emu = PolyEmu(X, Y, basis=Basis(degree=6, max_interaction=2))
```

`basis=` describes the forward basis over the inputs, so it cannot be combined
with `backward=True`.

A `Basis` degree sets the top of the forward degree sweep. `Basis(degree=2)`
fits at most degree 2. The sweep still rebuilds the basis at each rung, so a
rung below the Basis degree intersects the other constraints with that rung's
degree, and the top rung is the Basis as written.

`max_degree_forward` sets the same quantity, so passing both with different
values raises `ValueError` instead of one silently winning.

```python
PolyEmu(X, Y, basis=Basis(degree=2))                                 # sweeps up to 2
PolyEmu(X, Y, basis=Basis(max_interaction=2), max_degree_forward=5)  # sweeps up to 5
PolyEmu(X, Y, basis=Basis(degree=2), max_degree_forward=5)           # ValueError
```

`init_deg_forward` sets where the sweep starts, which is a separate quantity:
`Basis(degree=5)` with `init_deg_forward=1` sweeps 1 to 5. An `init_deg_forward`
above the Basis degree raises `ValueError`; the default start is lowered to the
Basis degree when the heuristic would overshoot it.

The `degree` argument to `Basis.build()` still overrides `Basis.degree` for
direct calls.

## Interaction order

Sorting the isotropic set by the number of active parameters
`s = |supp(alpha)|` gives

$$\binom{p+d}{d}=\sum_{s=0}^{\min(p,d)}\binom{p}{s}\binom{d}{s}$$

`max_interaction=q` truncates that sum at `s = q`. For `p=12, d=6` the full set
is 18,564 terms and `q=2` keeps 1,063.

## Block-separable parameters

If the target is additively separable over disjoint blocks,

$$f(\theta)=\sum_{k=1}^{K} f_k(\theta_{B_k}),$$

then every mixed second derivative across blocks vanishes, so only monomials
whose support lies inside a single block survive. The count drops from
`C(p+d, d)` to

$$1+\sum_k\left[\binom{p_k+d}{d}-1\right].$$

```python
blocks = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
emu = PolyEmu(X, Y, basis=Basis.separable(blocks), max_degree_forward=5)
```

| p, d, blocks | isotropic | block | dense M, isotropic | dense M, block |
|---|---|---|---|---|
| 9, 5, 3x3 | 2,002 | 166 | 30.6 MiB | 0.21 MiB |
| 12, 6, 3x4 | 18,564 | 628 | 2.6 GiB | 3.0 MiB |
| 20, 6, 4x5 | 230,230 | 1,845 | 404 GiB | 26 MiB |

`blocks` must partition the parameters. A parameter left out has no defined
coupling, so list it in its own block to declare it independent.

The usual gain is not memory but sample count: a stable least-squares fit needs
more samples than basis terms, and each sample is normally one expensive model
evaluation. On a nine-parameter separable target the 166-term block basis
reaches its accuracy floor at 400 training points, where the 2,002-term
isotropic basis is still underdetermined.

### When the reduction is lossless

Under a product design measure, which a Latin hypercube or a uniform box
satisfies, the `L^2` projection of a block-separable `f` onto the total-degree
space already lies entirely in the block subspace. Restricting to that subspace
loses nothing. The block index set is also downward closed, so it spans the same
functions as the corresponding orthonormal Legendre set and the fitted function
does not depend on which of the two is used to express it.

### When it is not

A coupling across blocks cannot be recovered after the fact. On a nine-parameter
target the block basis matched the isotropic basis while the target was
separable, was 31x worse once a single cross-block term was added, and was 297x
worse on a target that is separable only in a rotated frame. Separability in
rotated coordinates gives no reduction here, because the monomial index set is
not closed under rotation; find the active subspace first.

Multiplicative separability `f = prod_k f_k` is a low-rank structure rather than
a sparse index set. Fit `log f` with `transform="log"` to turn it into the
additive case, or use a factored fit.

## Finding the blocks

`PolyEmu.interaction_graph()` projects the outputs onto the orthonormal Legendre
product basis over the training box and returns the full pairwise interaction
matrix plus its connected components.

```python
probe = PolyEmu(X, Y, max_degree_forward=4)
report = probe.interaction_graph(degree=4)
report["blocks"]          # ((0, 1, 2), (3, 4, 5), (6, 7, 8))
emu = PolyEmu(X, Y, basis=Basis.separable(report["blocks"]), max_degree_forward=6)
```

`matrix[i, j]` counts every term whose support contains both parameters, so a
three-way coupling `x_i x_j x_k` registers on all three of its pairs.
`sobol_report` instead returns only the ten largest pairs of support exactly
two, which hides a cross-block pair as soon as `p` grows.

Two things decide whether the answer is trustworthy.

The degree must be able to represent the interaction. An even coupling such as
`0.25 x_0^2 x_3^2` is total degree 4 and is invisible at `degree=3`. On a
nine-parameter test at `degree=4` that coupling read `5.3e-4` against a
numerical noise floor of `3.4e-6`; at `degree=3` it read below the floor and the
blocks came back wrongly un-merged.

The projection must explain the data. A fit that leaves variance unexplained can
hide an interaction in the residual, so `interaction_graph` warns when the
explained fraction falls below `min_explained`.

`threshold` is a **variance** share, so it is the square of the amplitude share:
the `1e-4` default treats a coupling contributing under 1% of the signal
amplitude as absent.


## Bounded order in a parameter

If the target reaches only a fixed power of some parameter, every higher power
is a basis term whose true coefficient is zero. Those terms do not bias the fit
-- least squares recovers the zero in the population limit, whatever the basis
-- so the cost is sample count, memory and conditioning, not accuracy once `N`
is ample. On a 5-parameter degree-7 target that is exactly quadratic in two
parameters:

| basis | terms | nRMSE, N=2000 | nRMSE, N=8000 | cond(M), N=8000 |
|---|---|---|---|---|
| isotropic `d=7` | 792 | 3.51e-05 | 2.29e-05 | 1.90e+05 |
| `per_parameter=(2,2,7,7,7)` | 546 | 2.99e-05 | 2.25e-05 | 1.45e+05 |
| `+ parity` on the two even parameters | 223 | 2.38e-05 | 2.13e-05 | 1.27e+05 |

The accuracy gain shrinks as `N` grows (17 percent at N=2000, 7 percent at
N=8000) because the redundant terms add variance, not bias. What does not
shrink is the 3.6x smaller basis and the sample count it needs.

`PolyEmu.degree_profile()` reads both structures off the Legendre
decomposition and hands back a `Basis`:

```python
prof = probe.degree_profile(degree=5)
prof["max_degree"]      # (2, 2, 5, 3, 4)
prof["parity"]          # (None, 'even', None, None, 'even')
emu = PolyEmu(X, Y, basis=prof["basis"], max_degree_forward=7)
```

Unlike a block boundary, where the cross terms are structurally absent, a
degree cap truncates a convergent series. `tol` is therefore a real
bias/redundancy trade-off, not free: it is a variance share, so the induced
amplitude error is its square root.

Parity is inferred with a relative odd-versus-even test rather than an absolute
one, because the odd-power share of a genuinely even parameter is sampling
noise whose size depends on `N`. Only `"even"` is inferred; `"odd"` requires the
target to be globally odd in that parameter, which the powers alone cannot
establish, so `Basis` accepts it but `degree_profile` never reports it.


## When the structure is not additive

`Basis(blocks=...)` encodes which coefficients are zero. Two common structures
are not of that kind.

**Multiplicative.** For `f = prod_k f_k(theta_Bk)` the coefficient tensor is a
rank-one outer product over a *full* tensor-product index set, so the free
parameters drop from `prod_k |A_k|` to `sum_k |A_k|` while the effective total
degree rises to `sum_k d_k`. Three blocks at degree 5 reach total degree 15
over nine parameters with 168 coefficients, where the isotropic degree-15 basis
has `C(24, 15) = 1,307,504` terms. Fitting it is multilinear, not linear:

```python
from MomentEmu.factored import FactoredEmu, separability_report

report = separability_report(pilot, X[:400], blocks=blocks)
report["structure"]        # 'multiplicative', 'additive' or 'neither'
emu = FactoredEmu(X, Y, blocks, rank=2, degree=5)
```

Taking `log f` turns the product into a sum, but only where the data is
strictly positive, and it degrades near any zero crossing whatever the sign.
The rank-R model needs no transform. It contains the additive model as the case
where the other factors are constant.

Alternating least squares is non-convex and restarts are not optional: on a
rank-2 target, single-start fits at the correct rank ranged over 64x with the
seed. `FactoredEmu` runs several and keeps the best training residual;
`restart_residuals` shows how close the run came to failing.

**Ridge.** For `f(theta) = h(W theta)` the target is low dimensional in the
right coordinates while carrying interactions at every ANOVA order in the
original ones. Neither `max_interaction` nor `blocks` helps, because the
monomial index set is not closed under rotation. Rotate first:

```python
from MomentEmu.rotation import ActiveSubspaceEmu, scan_rank

scan_rank(X, Y, ranks=[2, 3, 4], degree=10, X_test=Xt, Y_test=Yt)
emu = ActiveSubspaceEmu(X, Y, rank=2, max_degree_forward=12)
```

On a travelling-trough target the isotropic degree-7 basis needed 3,432 terms
for 18.4%, while rank 2 at degree 12 needed 91 for 11.5%.

`rank="auto"` reaches a variance target and stops there, which overshoots
whenever the spectrum has a shallow tail: a direction costs a whole dimension
of the `C(d+r, r)` growth however little variance it carries. Use `scan_rank`.
Keep the pilot cheap, too -- a higher-degree pilot wiggles in directions the
target does not use and blurs the very gap the rank selection reads.


## Letting the response choose the index set

Every constraint above is a *prior* truncation: it fixes the index set before
seeing the response. Sparse polynomial chaos selects it from the response
instead, which states asymmetries no fixed rule can -- one pair of parameters
needing degree 18 while another needs 2.

```python
from MomentEmu.sparse import SparseEmu

emu = SparseEmu(X, Y, candidate=Basis(degree=18, max_interaction=2),
                degree=18, n_terms=60)
emu.report()["compression"]     # candidates per retained term
```

`n_terms="auto"` walks the greedy path and takes the size that minimises a
held-out error; it warns when the minimum lands on the last point of the path,
which means the search was truncated rather than converged.

Selection runs in the orthonormal Legendre basis rather than in monomials.
Greedy selection needs a dictionary whose columns are nearly orthogonal, and
monomials are nearly parallel: on `[-1, 1]` the mutual coherence of degrees
0-18 is 0.9984 for monomials against 0.0086 for Legendre, so a greedy step
choosing between `z^10` and `z^18` is decided by sampling noise. This is why
sparse polynomial chaos is formulated in an orthonormal basis (Blatman &
Sudret 2011).

Sparsity is a property of a basis, not of a function: a target that is four
monomials is not four Legendre terms, and the reverse. What transfers is
accuracy per retained term.

The candidate set is materialised as a design matrix, so its size is bounded by
memory rather than by the `D x D` Gram a textbook formulation would build. The
fast index search above is what makes a degree-25, `max_interaction=3`
candidate set (86,976 terms in 0.14 s) buildable at all.


## Choosing the coordinate, not the basis

Everything above decides which basis functions to keep at a given convergence
rate. Warping the inputs changes the rate itself, so it multiplies with the
rest rather than overlapping.

Polynomial approximation converges as `rho ** -d`, where `rho` is set by how far
the target's nearest singularity in the **complex** plane sits from the
interval, not by how sharply it bends on the real line. A monotone warp is a
conformal map: it moves those singularities.

```python
from MomentEmu.warp import WarpedEmu, fit_warps

warps = fit_warps(X, Y)
[w.spec() for w in warps]        # ['log', 'log', 'identity', 'identity']
emu = WarpedEmu(X, Y, max_degree_forward=8)
emu.gain                          # held-out improvement over raw coordinates
```

Taking the log of a parameter that spans decades is the special case of this
that a user would otherwise have to know to apply by hand. On a target that is
polynomial in the logs of two such parameters, the held-out error fell from
138.5% to 0.041%.

Three things about the search are worth knowing.

The scan covers a small single-parameter family per axis, not a free monotone
spline. Free arc-length equalisation is degenerate: equalising by `|f'|` yields
a coordinate in which the response is linear by construction. The blended
version that avoids that moved the rate only from 0.919 to 0.905, while a
parametric family wrong about the feature's sharpness by a factor of 2.5 was
worth 44,000x in term count. The family matters more than its parameter.

An axis is warped only on evidence. A candidate must cut the held-out error to
0.95 of the identity's, because an axis acting purely through an interaction
carries no marginal signal and would otherwise take whichever candidate best
fitted the noise.

Axes are scored together, not one at a time. `criterion="marginal"` is cheaper
but reads only an axis's marginal effect, and missed two sharp transitions
entirely on one test target.

A warp does not commute with the uniform-design tools. A design uniform in the
raw parameters is not uniform in the warped ones, so `sobol_report`,
`interaction_graph` and `degree_profile` run on the inner emulator describe the
warped coordinates.


## Composing the two preconditioners

Rotation and warping both change the coordinates, and they do not commute.
Neither order is right in general.

| target | warp only | rotate only | rotate then warp | warp then rotate |
|---|---|---|---|---|
| sharp feature on a rotated direction | 31.2% | 17.2% | **3.7%** | 17.2% |
| ridge in log coordinates | 3.6% | 36.9% | 35.9% | **7.1%** |

Two readings matter.

A per-axis warp cannot see a feature that lies along a rotated direction: on
the first target every axis came back `identity` and the result matched the
unpreconditioned fit exactly. Rotation has to come first.

Rotation applied first to the second target is *worse than doing nothing*
(36.9% against 33.2%). The active subspace is a linear projection, and a target
that is a ridge in log coordinates is not a ridge in the raw ones, so the rank
reduction discards real signal. Warping first turns it into a genuine ridge --
the leading gradient-covariance share went from 0.832 to 0.961 -- and the
projection then costs almost nothing.

```python
from MomentEmu.precondition import PreconditionedEmu

emu = PreconditionedEmu(X, Y, rank=2)          # order chosen from the data
emu.order                                       # 'rotate_warp' or 'warp_rotate'
emu.scores                                      # held-out error per candidate
```

Each order is scored at the highest degree *it* can afford, not at one degree
shared by all: comparing a two-dimensional fit and a seven-dimensional one at a
degree the seven-dimensional one can reach would hide the benefit rotation
exists for.

`select="accuracy"` takes the lowest error. `select="parsimony"` takes the
smallest model within a tolerance of it, which is usually the useful choice:
on the log-ridge target that is 105 coefficients instead of 1,716 for 1.9x the
error.
