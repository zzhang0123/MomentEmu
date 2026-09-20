# Product structure

## This is not sparsity

`Basis(blocks=...)` exploits **additive** separability, which is a statement
about which coefficients are zero. Multiplicative separability

$$
f(\theta) = \prod_k f_k\bigl(\theta_{B_k}\bigr)
$$

is not sparsity at all. Expand each factor and the coefficient tensor is a
rank-one outer product over a **full** tensor-product index set: every term is
present, and their values are constrained rather than zero. No index-set
truncation can state that.

What the structure buys is the ratio between $\prod_k |A_k|$ free parameters
and $\sum_k |A_k|$. Three blocks of three parameters at degree 5 reach a total
degree of 15 over nine parameters with **168** coefficients, where the
isotropic degree-15 basis has $C(24, 15) = 1{,}307{,}504$ terms.

## Why not take the logarithm

$\log$ turns a product into a sum, so an additive model in $\log f$ would do.
It needs the data strictly positive, and it degrades near any zero crossing
whatever the sign. A rank-$R$ canonical (CP) model

$$
f(\theta) \approx \sum_{r=1}^{R} \prod_k f_k^{(r)}\bigl(\theta_{B_k}\bigr)
$$

needs no transform, handles sign changes, and contains the additive model as
the case where the other factors are constant.

```python
from MomentEmu.factored import FactoredEmu

emu = FactoredEmu(X, Y, blocks=((0, 1), (2, 3)), rank=1, degree=5)
```

On $f = \tanh(x_0 + \tfrac12 x_1)\,\bigl(1 + 0.4 x_2 x_3 - 0.3 x_3^2\bigr)$ at
4 parameters and 3,000 samples:

| model | held-out error |
|---|---|
| isotropic degree 5, 126 terms | 0.593 % |
| `FactoredEmu` rank 1 | **0.260 %** |
| `FactoredEmu` rank 2 | 0.274 % |

Rank 2 is not better, because the target is rank 1. Rank is a cost like any
other dimension.

## Restarts are not optional

Fitting is multilinear rather than linear: holding every block but one fixed
leaves a weighted least-squares problem in that block, which is what the
alternating least squares sweeps over. That makes it non-convex. On a rank-2
target, single-start fits at the correct rank ranged over **64x** with the
seed alone. `FactoredEmu` runs `n_restarts` of them and keeps the best
training residual; `restart_residuals` shows how close the run came to
failing, and it is worth reading rather than assuming.

## Outputs are scaled, not centred

Subtracting the mean turns $\prod_k f_k$ into $\prod_k f_k - c$, and that is
not a product. It costs exactly one rank: on an exactly rank-one target,
centring left rank 1 at $7.0\times10^{-2}$ where rank 2 reached
$6\times10^{-5}$; without centring, rank 1 reaches $6.0\times10^{-5}$.

Per-output scaling is harmless by contrast, because a per-column factor is
absorbed by the output weights. An additive offset in an output still needs a
rank slot, which the model supplies because every factor carries a constant
term. Centring did not avoid that cost, it moved it onto the product.

## Confirm the structure first

`separability_report` tests both structures without a logarithm, from the
pilot model's analytic derivatives:

$$
H_{ij} = \frac{\partial^2 f}{\partial\theta_i \partial\theta_j},
\qquad
M_{ij} = f\,\frac{\partial^2 f}{\partial\theta_i \partial\theta_j}
 - \frac{\partial f}{\partial\theta_i}\frac{\partial f}{\partial\theta_j} .
$$

$H_{ij}$ vanishes across blocks when $f$ is additively separable; $M_{ij}$
vanishes across blocks when it is multiplicatively separable. Each ratio is
the largest cross-block entry over the smallest within-block one, so small
means "separable in this sense":

```python
from MomentEmu.factored import separability_report

separability_report(pilot, X[:400], blocks=((0, 1), (2, 3)))["structure"]
```

| target | `structure` | additive ratio | multiplicative ratio |
|---|---|---|---|
| $A \cdot B$ | `multiplicative` | 1.69 | 0.123 |
| $A + B$ | `additive` | 0.00918 | 0.687 |
| $\tanh(A B + x_0 x_2)$ | `neither` | 4.70 | 2.19 |

`interaction_graph` cannot see multiplicative structure at all: a product
couples every pair, so it reports one block and says nothing.

!!! warning "A block whose members do not interact breaks the ratio"

    The denominator is the **smallest** within-block entry, so a block
    containing two parameters that do not couple to each other drives it to
    zero and the ratio to nonsense. Measured: the additive target
    $\tanh(x_0 + \tfrac12 x_1) + 0.4 x_2 - 0.3 x_3^2$ with blocks
    $(0,1), (2,3)$ reports `neither` at an additive ratio of 1.30, because
    $\partial^2 f / \partial x_2 \partial x_3 = 0$ exactly. The target is
    additively separable; the diagnostic cannot say so through that partition.

    Read `additive_matrix` and `multiplicative_matrix` directly, or call
    `separability_report` without `blocks` and let it propose the partition,
    when a block might contain uncoupled parameters.

## It cannot follow a rotation

A rotation replaces the parameters by linear combinations, so a block
partition of the originals no longer refers to anything.
`PreconditionedEmu(estimator="factored")` therefore leaves the rotation orders
out of its candidate set rather than raising part-way through the scan, and
naming one explicitly raises with that explanation.
