# Autodiff and symbolic export

The fitted model is a polynomial with known coefficients, so it converts to
whatever framework the surrounding code uses. All three backends reproduce
the numpy fit to float64 round-off: measured `max|diff|` of
$1.3\times10^{-15}$ (JAX), $1.1\times10^{-15}$ (Torch) and
$1.6\times10^{-15}$ (SymPy) over 256 points on a degree-5 fit.

The backends load lazily, so `import MomentEmu` does not pay for jax, torch
or sympy.

| | JAX | Torch | SymPy |
|---|---|---|---|
| entry point | `create_jax_emulator` | `create_torch_emulator` | `create_symbolic_emulator` |
| `basis_kind` | all three | all three | `monomial` only |
| backward direction | yes | forward only | forward only |
| gradients | `jax.grad`, `jacfwd`, `hessian` | autograd | `sp.diff` |
| batching | `jax.vmap` | native | `lambdify` over columns |

## JAX

```python
import jax
jax.config.update("jax_enable_x64", True)

from MomentEmu.jax_momentemu import create_jax_emulator

J = create_jax_emulator(emu)
J(X_new)                      # jitted
J.evaluate(X_new)             # un-jitted, safe inside your own jax.jit
```

float64 needs `jax_enable_x64` set before the emulator is built, and the
constructor says so rather than silently downcasting. Pass
`dtype=jnp.float32` to opt out deliberately.

The result is a frozen dataclass registered with
`jax.tree_util.register_dataclass`: the array leaves are the folded
coefficients, the scaler, the box and the plan tables, and the static
metadata is the counts, the degrees, the basis family and the direction. It
passes through `jit`, `grad`, `vmap` and `hessian` as a pytree, including as
a traced argument.

`evaluate` is deliberately un-jitted so it can be composed inside a user
log-density; `__call__` is the jitted entry. The `value_and_grad`, `jacobian`
and `hessian` helpers differentiate the traced evaluator rather than the
Python one, which is the difference between about 0.1 ms and about 20 ms per
call.

Measured against the analytic Jacobian: $6.7\times10^{-16}$.

## Torch

```python
from MomentEmu.torch_momentemu import create_torch_emulator

T = create_torch_emulator(emu)
x = torch.tensor(X_new, dtype=torch.float64, requires_grad=True)
T(x).sum().backward()
```

An `nn.Module` with the coefficients and scalers as buffers, so it moves with
`.to(device)` and participates in a larger graph. The design is built without
in-place writes, so `torch.func.vmap` can trace it. Measured against the
analytic Jacobian: $2.9\times10^{-16}$.

## SymPy

```python
from MomentEmu.symbolic_momentemu import create_symbolic_emulator

S = create_symbolic_emulator(emu, variable_names=["a", "b", "c"])
S["expression"]     # the closed form
S["lambdified"]     # a numpy callable
```

A degree-5 fit in 3 parameters comes out as a 56-term expression beginning
`0.86785600697620296*x0 - 0.42281290712854561*x1 + ...`. This is the form
that goes into a paper, a spreadsheet or another language.

Two limits:

- **Monomial only.** A fit with `basis_kind="legendre"` or `"chebyshev"`
  raises `NotImplementedError` naming the basis. Expanding an orthogonal fit
  into monomials would produce large coefficients that cancel, which is the
  conditioning the basis was chosen to avoid.
- **The digits are the fit's digits.** Symbolic export warns from
  $\operatorname{cond}(M) \ge 10^{8}$, well below the general `COND_WARN`,
  because printed coefficients are exactly the thing conditioning damages
  first. See [Accuracy and conditioning](numerics.md).

## Which to reach for

Use JAX when the emulator sits inside a gradient-based sampler or an
optimisation and you want it jitted with the rest. Use Torch when it is one
module among others in a training graph. Use SymPy when a human or another
language has to read the model, and accept the monomial basis for it.

If the emulator is preconditioned, remember that the exported object is the
**inner** estimator: a `WarpedEmu` or `ActiveSubspaceEmu` applies its
transform in numpy first. Reproducing the full pipeline in JAX means
exporting the polynomial and applying the warp and rotation in JAX yourself;
they are a per-axis map and a matrix product.
