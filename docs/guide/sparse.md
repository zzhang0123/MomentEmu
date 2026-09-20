# Selection from the response

Every truncation in [Prior truncation](basis.md) is decided before the
response is seen. Sparse polynomial chaos decides it from the response
instead (Blatman & Sudret 2011), which states asymmetries no fixed rule can:
one parameter needing degree 18 while another needs 2.

```python
from MomentEmu.sparse import SparseEmu

emu = SparseEmu(X, Y, degree=14, n_terms=200)
emu.multi_indices          # the terms it kept
```

`degree` sets the candidate set, not the model. The model is the `n_terms`
columns selected from it by simultaneous orthogonal matching pursuit.

## What it buys

![Held-out error against retained terms](../assets/sparse-pareto-light.svg#only-light)
![Held-out error against retained terms](../assets/sparse-pareto-dark.svg#only-dark)

On a target needing a high degree in one parameter and almost none in the
other three, 4 parameters and 3,000 samples:

| basis | terms | held-out error |
|---|---|---|
| isotropic, degree 7 | 330 | 7.15 % |
| isotropic, degree 9 | 715 | 5.23 % |
| isotropic, degree 11 | 1,365 | 8.17 % |
| selected from degree-14 candidates | 10 | **1.41 %** |
| selected | 80 | 1.47 % |
| selected | 320 | 1.78 % |

Ten selected terms beat 715 isotropic ones by a factor 3.7. The isotropic
column is spending its budget on terms the target does not use, and past
degree 9 it spends enough of them to start overfitting: 1,365 terms on 3,000
samples is worse than 715.

Both columns turn back up, which is the part worth reading twice. **More
selected terms is not better either.** Past the target's own sparsity the
extra columns fit noise, so `n_terms` is a real choice and not a budget to
max out. Score it on held-out data.

## Why it selects in an orthogonal basis

Selection runs in the orthonormal Legendre product basis whatever
`basis_kind` the final fit uses, because a greedy step needs a dictionary of
low mutual coherence and monomials are the opposite. On $[-1, 1]$ the
normalised correlation of the columns $z^{10}$ through $z^{18}$ with a pure
$z^{12}$ signal spans 0.1981 to 0.2000: a 0.5 percent spread that sampling
noise overturns, so the greedy step picks a neighbouring power at random.

This has a consequence worth stating plainly:

!!! note "Sparsity is a property of a basis, not of a function"

    A target that is four monomials is not four Legendre terms, and the
    reverse. There is no basis-free notion of "this function has 20 terms".
    What transfers between bases is accuracy per retained term, which is what
    the selection optimises.

`basis_kind` chooses the family the design is built from, and since a85fe5d
every design in the selection, the held-out path, the final fit and the
prediction comes from one shared plan, so selecting in one basis and
predicting in another is unreachable. The default stays `legendre`; on the
21cmGEM benchmark the three families score 1.4478, 1.4555 and 1.5976 percent,
so the parameter buys no accuracy there and exists for the cases where the
design is distributed differently. See
[Choosing a basis family](numerics.md#choosing-a-basis-family).

## Why the candidate set can be large

The point of the method is a large candidate set, so a Gram matrix over it is
not an option: $D \times D$ at $D = 86{,}976$ is 60 GB. The search therefore
works from $\Phi$ with an incrementally extended Cholesky factor of the
**active** block only, which is $k \times k$ for $k$ selected terms. That is
what makes `degree=14` in several parameters affordable as a candidate set
when it would be unusable as a basis.

The index set is shared across outputs rather than selected per output, which
keeps inference a single GEMM. On a many-output target that matters more than
it sounds: per-output selection would give each output its own design matrix.

## Composing it

`SparseEmu` takes preconditioned inputs like any other estimator, and it is
the natural partner for a rotation, because the reduced dimension makes a
large candidate set affordable:

```python
from MomentEmu.precondition import PreconditionedEmu

emu = PreconditionedEmu(X, Y, order="auto", estimator="sparse", n_terms=200)
```

Candidate orders are scored with the estimator that will be used, so a sparse
model is scored as a sparse model. A rotation that helps a dense fit by
cutting the dimension helps a sparse one less, because the selection was
already paying only for the terms it kept.
