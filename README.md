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
from MomentEmu import PolyEmu
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
> outputs. See `help(signal_aware_frac_err)` for parameter calibration
> guidance and the meaning of each diagnostic key.

## Auto-Differentiation Support

**MomentEmu supports automatic differentiation** through three different frameworks, enabling gradient-based optimization, neural network integration, and exact symbolic analysis:

### Available Frameworks:
- **🚀 JAX**: High-performance computing with JIT compilation and GPU acceleration
- **🔥 PyTorch**: Native neural network integration and ML pipeline compatibility  
- **SymPy**: exact symbolic derivatives of the exported polynomial (its agreement with the emulator depends on the form and degree)

### Quick Example:
```python
# JAX implementation
from MomentEmu.jax_momentemu import create_jax_emulator
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



## Numerical accuracy and backends

- At a fixed degree MomentEmu solves the same least-squares problem as
  scikit-learn PolynomialFeatures + LinearRegression; the coefficients agree to
  about 1e-13 on a well-conditioned design. Fit-speed differences come from the
  solver (a Cholesky solve of the normal equations versus an SVD/lstsq), not
  from a different estimator.
- Backend choice:

  | use case | backend |
  |---|---|
  | cobaya / emcee (many independent single-point calls) | NumPy forward_emulator |
  | numpyro / blackjax / Laplace / Fisher (derivatives, jit) | JAX create_jax_emulator |

  The JAX backend adds derivatives of any order and composes inside a jit; a
  batch-1 standalone JAX predict is slower than the vectorised NumPy path.
- JAX needs jax.config.update("jax_enable_x64", True) for float64; construction
  raises otherwise unless dtype=jnp.float32 is passed explicitly.
- Training design: a k-level grid identifies a parameter power only up to k-1.
  Use iid uniform or scrambled Sobol samples with N >= 10-20 D.
- Validate in the units of your data: PolyEmu.validate(X, Y, sigma=...) reports
  Delta-chi2, and posterior_bias reports the linearised shift in sigma.
- The posterior-shift acceptance protocol (P3.3) is in
  examples/validate_posterior.py: it compares a simulator posterior with the
  emulator posterior on the same data and passes when every marginal mean
  shift is < 0.1 sigma, the Mahalanobis shift is < 0.1 and every width ratio
  is within 10%.
- See CHANGELOG.md for the 2.0.0 default-behaviour changes and deprecations.


## Benchmarks

Measured with `python -m benchmarks` (see `benchmarks/README.md`) on Apple M3 Ultra, numpy 2.3.5 (accelerate), MomentEmu 9a98c63. Fit times are the minimum of 3 runs; inference times are the minimum over 5 blocks of repeated calls. nRMSE = test RMSE / RMS deviation of the target. "max rel err" is the signal-aware maximum relative error (entries below 1e-3 of the per-output peak are masked).

### Accuracy and cost on standard targets

| target | n | m | N | mode | degree | D | fit s | nRMSE | max rel err | cond(M) | 1-pt us | 1000-pt us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ishigami | 3 | 1 | 4000 | default (sweep 2-10) | 10 | 286 | 0.04 | 1.50e-03 | 0.69 | 1.49e+08 | 31.3 | 662 |
| ishigami | 3 | 1 | 4000 | fixed_d9 | 9 | 220 | 0.0158 | 0.013 | 4.07 | 1.61e+07 | 29.2 | 550 |
| sobol_g | 8 | 1 | 4000 | default (sweep 1-5) | 4 | 495 | 0.105 | 0.166 | 23.8 | 9.73e+03 | 21 | 1.46e+03 |
| sobol_g | 8 | 1 | 4000 | fixed_d5 | 5 | 1287 | 0.105 | 0.199 | 26.1 | 4.40e+04 | 34.1 | 3.07e+03 |
| friedman | 10 | 1 | 4000 | default (sweep 1-4) | 4 | 1001 | 0.0996 | 0.0188 | 0.101 | 2.16e+04 | 27.5 | 3.22e+03 |
| friedman | 10 | 1 | 4000 | fixed_d4 | 4 | 1001 | 0.0877 | 0.0188 | 0.101 | 2.16e+04 | 27.5 | 3.21e+03 |
| rosenbrock | 4 | 1 | 4000 | default (sweep 2-4) | 4 | 70 | 9.40e-03 | 7.66e-15 | 1.13e-12 | 1.63e+03 | 18.4 | 220 |
| rosenbrock | 4 | 1 | 4000 | fixed_d4 | 4 | 70 | 6.92e-03 | 7.66e-15 | 1.13e-12 | 1.63e+03 | 18.5 | 221 |
| log_rosenbrock | 4 | 1 | 4000 | default (sweep 2-12) | 8 | 495 | 0.288 | 0.154 | 0.579 | 5.58e+06 | 28.6 | 1.24e+03 |
| log_rosenbrock | 4 | 1 | 4000 | fixed_d8 | 8 | 495 | 0.0348 | 0.154 | 0.579 | 5.58e+06 | 28.5 | 1.3e+03 |
| gauss_peak | 3 | 1 | 4000 | default (sweep 2-20) | 14 | 680 | 0.832 | 0.075 | 3 | 8.76e+11 | 40.5 | 1.69e+03 |
| gauss_peak | 3 | 1 | 4000 | fixed_d10 | 10 | 286 | 0.0192 | 0.12 | 16.7 | 1.64e+08 | 30.9 | 800 |
| cmb_like | 6 | 2000 | 5000 | default (sweep 2-5) | 5 | 462 | 0.239 | 2.29e-03 | 0.0367 | 1.47e+04 | 89.2 | 4.58e+03 |
| cmb_like | 6 | 2000 | 5000 | fixed_d5 | 5 | 462 | 0.184 | 2.29e-03 | 0.0367 | 1.47e+04 | 89 | 4.59e+03 |

- Rosenbrock is a quartic polynomial. At fixed degree 4 the fit is exact to nRMSE 7.7e-15; the default sweep selects the same degree-4 basis (D = 70) and reaches nRMSE 7.7e-15. 2.0.0 removed the old `dim_reduction` pruning (D15), which used to drop modes and raise this error.
- The P0.6 sample-count guard caps every default sweep at D <= N_train / 2, so no target ends on a singular rung; the sweep stops on the RMSE criterion or the degree cap.
- The Sobol G-function has |4x-2| kinks; the degree-5 polynomial reaches nRMSE 0.2 (max rel err 26). A total-degree polynomial does not converge on a kink.

### Against sklearn baselines (same data, same fixed degree)

| target | MomentEmu nRMSE | Poly+LinReg nRMSE | coef max rel diff | GP nRMSE | MLP nRMSE | fit s: MomentEmu / Poly+LinReg / GP / MLP | 1-pt us: MomentEmu / Poly+LinReg / GP / MLP |
|---|---|---|---|---|---|---|---|
| ishigami | 0.013 | 0.013 | 5.0e-11 | 2.15e-05 | 0.0206 | 0.017 / 0.0266 / 9.52 / 6.57 | 28.6 / 103 / 101 / 71 |
| sobol_g | 0.199 | 0.199 | 1.0e-13 | 0.0596 | 0.03 | 0.114 / 0.276 / 15.1 / 4.53 | 34.2 / 121 / 101 / 73.8 |
| friedman | 0.0188 | 0.0188 | 1.5e-13 | 5.86e-05 | 9.54e-03 | 0.0892 / 0.182 / 22.3 / 3.96 | 28.3 / 120 / 105 / 72.5 |
| rosenbrock | 7.66e-15 | 1.39e-15 | 1.4e-14 | 1.34e-05 | 0.0182 | 7.52e-03 / 7.83e-03 / 13.4 / 3.73 | 18.7 / 93 / 114 / 72.4 |
| log_rosenbrock | 0.154 | 0.154 | 1.8e-11 | 0.0758 | 0.0546 | 0.0376 / 0.0605 / 18.8 / 6.85 | 28.8 / 111 / 106 / 72.9 |
| gauss_peak | 0.12 | 0.12 | 1.1e-09 | 4.41e-05 | 0.0157 | 0.0205 / 0.0352 / 20.4 / 4.2 | 31.7 / 108 / 93.4 / 69.3 |
| cmb_like | 2.29e-03 | 2.29e-03 | 6.3e-13 | 1.95e-05 | 0.0128 | 0.184 / 0.311 / 113 / 24.4 | 92.8 / 151 / 484 / 83.6 |

- `PolynomialFeatures + LinearRegression` is the same model: coefficients agree to 1.1e-09 relative or better on every target. MomentEmu's normal-equation solve is 1-2.4x faster to fit; its single-point inference is 0.2-0.61x slower, because the P0.7 recursive plan evaluates the design row in Python while sklearn uses one BLAS call.
- The GP (N capped at 2000) has lower test error than the polynomial on 6 of 7 targets, by 2-2.7e+03x; the polynomial wins only where the target is a polynomial (rosenbrock, 1.8e+09x). GP fit time is 9.5-1.1e+02 s and its 1000-point prediction is 12-24 ms.
- The MLP ((64,64) tanh; (128,128) for the CMB-like target) has lower test error than the degree-5/4/8/10 polynomial on 4 of 7 targets (sobol_g 6.6x, friedman 2x, log_rosenbrock 2.8x, gauss_peak 7.7x) at 3.7-24 s of fit time; its single-point prediction is 0.26-1.1x faster than MomentEmu's.

### Scaling

Fit time (s, N = 10000, one output, fixed degree) and D:

| n \ d | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| 2 | 4.8e-03 (D=6) | 5.6e-03 (D=10) | 6.3e-03 (D=15) | 6.5e-03 (D=21) | 7.3e-03 (D=28) | 7.0e-03 (D=36) | 8.5e-03 (D=45) |
| 3 | 6.2e-03 (D=10) | 7.2e-03 (D=20) | 8.2e-03 (D=35) | 9.9e-03 (D=56) | 0.012 (D=84) | 0.014 (D=120) | 0.021 (D=165) |
| 4 | 7.0e-03 (D=15) | 9.0e-03 (D=35) | 0.012 (D=70) | 0.015 (D=126) | 0.024 (D=210) | 0.035 (D=330) | 0.048 (D=495) |
| 5 | 8.4e-03 (D=21) | 0.011 (D=56) | 0.015 (D=126) | 0.025 (D=252) | 0.045 (D=462) | 0.082 (D=792) | 0.14 (D=1287) |
| 6 | 9.5e-03 (D=28) | 0.014 (D=84) | 0.023 (D=210) | 0.046 (D=462) | 0.1 (D=924) | 0.21 (D=1716) | 0.47 (D=3003) |
| 7 | 0.011 (D=36) | 0.017 (D=120) | 0.038 (D=330) | 0.088 (D=792) | 0.22 (D=1716) | 0.62 (D=3432) | - |
| 8 | 0.011 (D=45) | 0.02 (D=165) | 0.056 (D=495) | 0.16 (D=1287) | 0.5 (D=3003) | - | - |
| 9 | 0.015 (D=55) | 0.028 (D=220) | 0.085 (D=715) | 0.28 (D=2002) | - | - | - |
| 10 | 0.014 (D=66) | 0.036 (D=286) | 0.12 (D=1001) | 0.51 (D=3003) | - | - | - |

Inference (n = 6): single-point time is set by the Python loop that builds the design row; the matmul is a few percent.

| D (m=1000) | 1-pt us | of which Phi build | 1000-pt per-sample us |  | m (D=462) | 1-pt us | 1000-pt per-sample us |
|---|---|---|---|---|---|---|---|
| 7 | 29.2 | 93% | 1.41 |  | 1 | 23.3 | 1.76 |
| 28 | 39.2 | 92% | 1.54 |  | 10 | 24.2 | 1.85 |
| 84 | 48.7 | 96% | 1.87 |  | 100 | 34.2 | 2.12 |
| 210 | 53.3 | 98% | 2.47 |  | 1000 | 58.5 | 3.8 |
| 462 | 59.4 | 99% | 3.77 |  | 10000 | 671 | 17.7 |
| 924 | 72.8 | 99% | 6.24 |  |  | |  |
| 1716 | 121 | 99% | 10.1 |  |  | |  |
| 3003 | 315 | 97% | 16.6 |  |  | |  |

Peak memory of a fit (n = 6, D = 462, m = 100), tracemalloc peak / RSS increase:

| N | batch_size=None | batch_size=1000 | batch_size=10000 |
|---|---|---|---|
| 2000 | 90.2 / 238 MB | 83.2 / 227 MB | 90.2 / 245 MB |
| 10000 | 159 / 373 MB | 90 / 331 MB | 159 / 363 MB |
| 50000 | 227 / 463 MB | 155 / 401 MB | 227 / 471 MB |

- At N = 50000, batch_size = 1000 the tracemalloc peak is 155 MB versus 227 MB at the default 10000: the LOO sweep keeps the full Phi resident while N x D x 8 fits the 512 MiB budget, and batches the leverage/PRESS pass by batch_size. Set MOMENTEMU_PHI_BUDGET_BYTES=0 to force the fully batched path (lower peak, about 1.4x the fit time). `dim_reduction` is ignored in 2.0.0 (D15).

### Backends (CMB-like emulator, D = 462, m = 2000)

| backend | 1-pt us | 1000-pt us |
|---|---|---|
| numpy (PolyEmu.forward_emulator) | 90 | 5.4e+03 |
| jax (create_jax_emulator, jit, x64) | 88.2 | 3.71e+03 |
| torch (TorchMomentEmu, float64, no_grad) | 868 | 5.81e+03 |

### Reproducibility and CI gate

- Five fixed-seed pins (`results/pins.json`) record test RMSE and the first ten coefficients. Repeating a fit in-process reproduces the coefficients to 0 relative; changing `batch_size` (summation order) moves them by up to 6.2e-13 relative. The gate tolerance is 1e-10.
- Timing noise on this machine over 7 repeats: single-shot fit max/min 1.1 (17 ms fit) and 1 (174 ms fit); min-of-3 fit max/min 1.1 and 1; inference max/min 1 (1-pt) and 1.1 (1000-pt). `python -m benchmarks.gate` fails on a fit-time or inference-time regression above 30 %, an RMSE regression above 1e-6 relative, or a pin drift above 1e-10.

### Corrections to the earlier hand-maintained table

The 2026-02 table was withdrawn for the plan-target rows. Phi at N=20000, D=3003 was quoted as 0.95x against a lazy-loop baseline; the reproducible run and the independent review both measure parity with the v1 loop (1.01x), so the speed-up is not claimed. Batch-1000 was quoted as 0.70x against v2's own lazy loop rather than v1; against v1 it is 0.749x, so the <= 0.7x target is missed. The single-point forward (88.7 us, target < 80 us) and the analytic Jacobian (285-290 us, target <= 148 us) targets remain missed. The tables above are generated from the checked-in `results/` run (`results/RESULTS.md`, with load averages per suite) by
`python -m benchmarks --out results && python -m benchmarks.readme_section results`.
