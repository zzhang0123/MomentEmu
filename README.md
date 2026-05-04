# MomentEmu

A lightweight, interpretable polynomial emulator for smooth mappings, implemented in pure Python.


## 📖 Overview

**MomentEmu** implements the **moment-projection polynomial emulator** introduced in Zhang (2025) ([arXiv:2507.02179](https://arxiv.org/abs/2507.02179)).
It builds interpretable, closed-form polynomial emulators via moment matrices, achieving millisecond-level inference and symbolic transparency.

**Note:** Ideally, the test set should consist of random samples drawn independently from the parameter space. The user should avoid constructing the training and test sets as disjoint subsets of the same regular grid, since in that case the test set cannot reveal potential overfitting of the emulator.

For a complete working example demonstrating MomentEmu applied to cosmological parameter estimation (PolyCAMB), see the companion repository: [MomentEmu-PolyCAMB-examples](https://github.com/MomentEmu/MomentEmu-PolyCAMB-examples).

## 🚀 Features

- Pure Python implementation; minimal dependencies (`numpy`, `scipy`, `sympy`, ...)
- Closed-form polynomial expressions (symbolic)
- Supports **forward** (θ → y) and **inverse** (y → θ) emulation
- Fast training via moment matrices; near-instant inference
- **Modular auto-differentiation support** via JAX, PyTorch, and SymPy
- Suitable for MCMC, Bayesian inference, sensitivity analyses, and gradient-based tasks
- Compact—no heavy model files
- **Flexible installation** with optional dependencies for different use cases

## 🛠️ Installation

### Basic Installation
```bash
# Core functionality only (lightweight)
pip install git+https://github.com/zzhang0123/MomentEmu.git
```

### With Auto-Differentiation Support
```bash/zsh
# Core + JAX (high-performance computing)
pip install "MomentEmu[jax] @ git+https://github.com/zzhang0123/MomentEmu.git"

# Core + PyTorch (machine learning)
pip install "MomentEmu[torch] @ git+https://github.com/zzhang0123/MomentEmu.git"

# Core + all auto-differentiation frameworks
pip install "MomentEmu[autodiff] @ git+https://github.com/zzhang0123/MomentEmu.git"

# Everything including visualization tools
pip install "MomentEmu[all] @ git+https://github.com/zzhang0123/MomentEmu.git"
```

### Development Installation
```bash
git clone https://github.com/MomentEmu/MomentEmu.git
cd MomentEmu
pip install -e '.[all]'  # Install in development mode with all features
```

## 📋 Dependencies

### Core Dependencies (always installed)
- `numpy`
- `scipy` 
- `sympy`
- `scikit-learn`

### Optional Dependencies (install as needed)
- **JAX**: `jax`, `jaxlib` (for high-performance auto-differentiation)
- **PyTorch**: `torch` (for machine learning integration)  
- **Visualization**: `matplotlib` (for plotting and analysis)

## 🧪 Quick Start

**Note**: Make sure to install MomentEmu first using one of the installation methods above.

```python
from MomentEmu.PolyEmu import PolyEmu
import numpy as np

# Generate example training data
# 2D input parameters (e.g., physical parameters)
X_train = np.random.uniform(1, 2, (500, 2))

# Multi-output observables with different functional forms
Y_train1 = (X_train[:, 0]**2 + X_train[:, 1]**2).reshape(-1, 1)  # Quadratic combination
Y_train2 = (X_train[:, 0]**3 + X_train[:, 1]).reshape(-1, 1)     # Cubic + linear
Y_train = np.hstack((Y_train1, Y_train2))

print(f"Training data shape: X {X_train.shape}, Y {Y_train.shape}")

# Create emulator with both forward and inverse capabilities
emulator = PolyEmu(X_train, Y_train, 
                   forward=True,                    # Enable forward emulation: parameters → observables
                   backward=True,                   # Enable inverse emulation: observables → parameters
                   standardize_Y_with_std=False)    # Use only mean centering for Y (optional)

# Forward prediction: parameters → observables
X_new = np.array([[1.5, 1.8], [1.2, 1.9]])  # New parameter samples
Y_pred = emulator.forward_emulator(X_new)
print(f"Forward prediction: {Y_pred}")

# Inverse estimation: observables → parameters  
Y_new = np.array([[5.0, 4.2], [6.1, 5.8]])  # New observable samples
X_est = emulator.backward_emulator(Y_new)
print(f"Inverse estimation: {X_est}")

# Get symbolic polynomial expressions (interpretable models)
forward_expressions = emulator.generate_forward_symb_emu()
print(f"Forward symbolic expressions: {forward_expressions}")
```

### Key Features Demonstrated:
- **Multi-dimensional**: 2 input parameters, 2 output observables
- **Bidirectional**: Both forward (θ → y) and inverse (y → θ) emulation
- **Automatic model selection**: Optimal polynomial degree chosen via validation
- **Symbolic output**: Get interpretable closed-form polynomial expressions

## 🎯 Signal-Aware Validation Diagnostic

`PolyEmu` ships with a **signal-mask aware** fractional-error diagnostic that
robustly handles outputs spanning many orders of magnitude. Naive
`|diff| / |ref|` is undefined at floating-point-noise levels and would
otherwise produce spurious validation failures on wide-dynamic-range outputs.

Enable it by passing `return_max_frac_err=True`:

```python
from MomentEmu import PolyEmu

emu = PolyEmu(X_train, Y_train,
              forward=True,
              return_max_frac_err=True)

# Worst in-mask relative error (signal-mask filtered).
# float('inf') if the signal mask is empty so threshold checks fail loudly.
print(emu.forward_max_frac_err)

# Full diagnostic dict: max_rel, rmse, n_above, n_total,
# floor, dr_decades, strategy, argmax.
print(emu.forward_frac_err_diag)
```

The helper is also available as a standalone function for ad-hoc validation:

```python
from MomentEmu import signal_aware_frac_err

diag = signal_aware_frac_err(pred, ref, signal_floor_frac=1e-3)
if diag["n_above"] == 0:
    raise RuntimeError("signal mask empty; check fixture")
assert diag["max_rel"] < 1e-5, (
    f"max rel err {diag['max_rel']:.2e} at index {diag['argmax']}"
)
```

> **Upgrade note:** Pre-existing code that read `forward_max_frac_err` /
> `backward_max_frac_err` will see signal-mask-aware values now; numbers
> will differ from the prior naive relative-RMSE for wide-dynamic-range
> outputs. See [`signal_mask.md`](signal_mask.md) for the full design
> rationale, calibration anchors, and failure modes.

## Auto-Differentiation Support

**MomentEmu supports automatic differentiation** through three different frameworks, enabling gradient-based optimization, neural network integration, and exact symbolic analysis:

### Available Frameworks:
- **🚀 JAX**: High-performance computing with JIT compilation and GPU acceleration
- **🔥 PyTorch**: Native neural network integration and ML pipeline compatibility  
- **🔢 SymPy**: Exact symbolic differentiation with zero numerical error

### Quick Example:
```python
# JAX implementation
from jax_momentemu import create_jax_emulator
import jax.numpy as jnp
from jax import grad

# Convert trained emulator to JAX
jax_emu = create_jax_emulator(emulator)

# Compute gradients automatically
x = jnp.array([0.5, 0.3])
y = jax_emu(x)
gradient = grad(lambda x: jax_emu(x).sum())(x)
```


### 📖 Complete Auto-Differentiation Guide
For comprehensive documentation, performance comparisons, usage examples, and integration guidelines, see the **[Auto-Differentiation Guide](autodiff-guide.md)** in this repository.

The guide covers:
- Detailed usage for each framework (JAX, PyTorch, SymPy)
- Performance benchmarks and framework comparison
- Integration guidelines for different use cases
- Complete testing suite and troubleshooting tips

## 📚 Examples & Applications

For detailed examples and real-world applications, including:
- **PolyCAMB‑Dℓ**: Cosmological parameter → CMB power spectrum emulation
- **PolyCAMB‑peak**: Bidirectional parameter ↔ acoustic peak mapping
- Complete Jupyter notebooks with step-by-step tutorials

Visit the examples repository: **[MomentEmu-PolyCAMB-examples](https://github.com/MomentEmu/MomentEmu-PolyCAMB-examples)**

---

### 🧠 How It Works

MomentEmu builds:

- A **moment matrix**  
  $M_{\alpha\beta} = \frac{1}{N} \sum_{i} \theta_i^\alpha \theta_i^\beta$

- A **moment vector**  
  $\nu_\alpha = \frac{1}{N} \sum_{i} \theta_i^\alpha y_i$

Solving $M c = \nu$ finds polynomial coefficients $c$. No iterative optimization is needed --- model selection uses validation RMSE.  
[Read more in the arXiv paper](https://arxiv.org/abs/2507.02179). 


---
---

### Appendix: Derivative Errors in Polynomial Approximations

When a smooth function $f$ is approximated by a polynomial $P_n$ of degree $n$, the error in the approximation of its derivatives generally amplifies with the derivative order. If the function is fitted to a uniform accuracy $\|f - P_n\|_\infty \leq \delta $, then the worst-case error in the $r$-th derivative satisfies the bound
$$
\|f^{(r)} - P_n^{(r)}\|_\infty \lesssim n^r \cdot \delta,
$$
reflecting the fact that differentiation acts as a numerically unstable operator in the space of polynomials. This growth arises from Bernstein-type inequalities and classical results in approximation theory, such as Jackson’s theorem. Thus, while polynomial emulation can be highly accurate for the function itself, care must be taken when using it to infer high-order derivatives, especially for large $n$ or high $r$, as derivative estimates can become significantly less accurate even when the original approximation error is small.

---

### 📚 References

1. **Timothy J. Rivlin**, *An Introduction to the Approximation of Functions*  
   – Classic and accessible introduction. See Chapter 4–5 on uniform polynomial approximation and error estimates.

2. **E. W. Cheney**, *Introduction to Approximation Theory*  
   – Comprehensive, rigorous treatment. Bernstein and Jackson inequalities are covered in detail.

3. **L. N. Trefethen**, *Spectral Methods in MATLAB*  
   – Discusses how polynomial interpolation and spectral approximations behave under differentiation; very readable with practical insights.


