# Moment projection

## The model

Each output is a polynomial in the parameters,

$$
y_j(\theta) \;\approx\; \sum_{\alpha \in A} c_{\alpha j}\, \phi_\alpha(\theta),
$$

where $A$ is a set of multi-indices $\alpha = (\alpha_1, \dots, \alpha_n)$ and
$\phi_\alpha$ is the corresponding basis function. `basis_kind="monomial"` makes
it $\prod_k \theta_k^{\alpha_k}$; `"legendre"` and `"chebyshev"` make it a
product of the orthonormal one-dimensional polynomials of those degrees.

Choosing $A$ is the whole design problem, and
[Making the basis smaller](reduction.md) is about it. Everything on this page
holds for any $A$.

## Coordinates

`X` is mapped before the basis is evaluated. `scaling="standard"` subtracts the
mean and divides by the standard deviation of each column; `scaling="box"` maps
the training range of each column onto $[-1, 1]$. The orthogonal bases are
orthonormal on $[-1, 1]$, so they assume the box map; the standard map leaves a
design that reaches many standard deviations spread well past the interval the
orthogonality is stated on.

## The moment system

Write $\Phi_{i\alpha} = \phi_\alpha(\theta^{(i)})$ for the $N \times D$ basis
matrix over the design, $D = |A|$. Least squares on $\Phi c = Y$ has normal
equations $\Phi^{\mathsf T}\Phi\, c = \Phi^{\mathsf T} Y$, which the package
forms as

$$
M = \frac{1}{N}\,\Phi^{\mathsf T}\Phi \in \mathbb{R}^{D \times D},
\qquad
\nu = \frac{1}{N}\,\Phi^{\mathsf T} Y \in \mathbb{R}^{D \times m},
\qquad
M c = \nu .
$$

$M_{\alpha\beta}$ is the empirical second moment of $\phi_\alpha \phi_\beta$
under the design measure and $\nu_{\alpha j}$ the cross-moment of the basis with
output $j$: hence the name, and hence three properties.

- The $N$ rows are summarised once. After $M$ and $\nu$ are formed the design is
  not needed again, and the stored model is $c$, not the training data.
- One $M$ serves every output. A new output is one more column of $\nu$, which
  costs $O(ND)$ and reuses the existing factorisation.
- Cost splits as $O(ND^2)$ to accumulate and $O(D^3)$ to solve. At large $N$ the
  accumulation dominates, which is why $D$ and not $N$ is what the reduction
  machinery attacks.

With `weights`, the sums are weighted and the weights are normalised to mean 1;
uniform weights reproduce the unweighted moments bit for bit.

## Solving it

Forming $M$ squares the conditioning: $\operatorname{cond}(M) =
\operatorname{cond}(\Phi)^2$. The solver therefore has two routes, gated on the
estimated $\operatorname{cond}(M)$:

| $\operatorname{cond}(M)$ | route |
|---|---|
| below $10^{13}$ | Cholesky of $M$, then two triangular solves |
| $10^{13}$ and above | QR of $\Phi$, coefficients as $U^{-1}Q^{\mathsf T}Y$ |

The QR route never forms $\Phi^{\mathsf T}Y$, so it keeps digits the normal
equations cannot. [Accuracy and conditioning](numerics.md) covers the
thresholds, what they were calibrated against, and `ridge`.

## Choosing the degree

The constructor sweeps the degree upward. Index sets are nested, so the moment
system at degree $d+1$ contains the one at degree $d$ as its leading block: the
sweep borders the previous $M$ with the new rows and columns instead of
rebuilding $\Phi$, $M$ and $\nu$. `emu.forward_sweep_incremental_` records
whether that path ran.

Each rung is scored by leave-one-out error computed from the factorisation it
already has. With $M = U^{\mathsf T}U$ the leverage of training row $i$ is

$$
h_i = \frac{1}{N}\,\bigl\|U^{-\mathsf T}\Phi_i\bigr\|^2 ,
\qquad
\mathrm{PRESS}_j = \sum_i \Bigl(\frac{r_{ij}}{1 - h_i}\Bigr)^{2},
$$

which is the exact leave-one-out squared error, not an approximation to it: no
row is ever refitted. The package checks it against brute-force refits and
records agreement to 3e-14. The sweep stops when
$\sqrt{\operatorname{mean}_j \mathrm{PRESS}_j / N}$ reaches `RMSE_tol`, and it
will not go past the degree the sample count supports, which is
$|A| \le N / 2$.

## Inverse emulation

`backward=True` fits the same system with the roles of $X$ and $Y$ exchanged, so
$\theta(y)$ is a polynomial in the outputs. It is a separate fit with its own
degree sweep, and it is only meaningful where the forward map is invertible over
the training box.

## What is exact and what is not

| | |
|---|---|
| the solve | the least-squares solution of $\Phi c = Y$ on the stated $A$, to the conditioning |
| PRESS | the exact leave-one-out error of that fit |
| `emu.jacobian` | the exact derivative of the fitted polynomial |
| the emulator | an approximation of the target: truncation plus whatever noise the design carries |

The third and fourth lines are the pair to keep apart. The Jacobian is exact for
the polynomial and inherits the polynomial's error against the target, amplified
by the differentiation; the [quick start](quickstart.md) shows the size of the
gap on a fit whose values are good to 5e-3.
