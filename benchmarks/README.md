# MomentEmu benchmark suite

```
python -m benchmarks                       # all suites, results/ (JSON per suite + RESULTS.md)
python -m benchmarks --suites accuracy,pins --out my_results
python -m benchmarks --quick               # smoke test, ~4 min
python -m benchmarks --max-load 8          # wait until the 1-min load average is <= 8 before each suite
python -m benchmarks.noise_ci_proxy        # timing noise at 2 BLAS threads, idle and under CPU contention
python -m benchmarks.gate --baseline results_baseline --current results
python -m benchmarks.merge_min run1 run2 run3 --out baseline   # per-cell min of several runs
python -m benchmarks.report results        # print the markdown report
```

The package under test is imported from `MOMENTEMU_SRC` (default: the repo
`src/` directory); `bench/__init__.py` refuses to run if `MomentEmu.__file__`
resolves anywhere else, because a stale site-packages copy shadows the repo on
at least one developer machine.

Every suite JSON records the 1/5/15-minute load average at its start and end
(`loadavg_start`, `loadavg_end`); `bench.gate` prints a WARN line when either
side's 1-minute load average exceeds the core count, because timings measured
on a contended machine are not comparable (a run of this suite on a 28-core
machine with load average 300 gave single-point inference 14x slower than the
idle value).

## Suites

| suite | what it measures | wall time here |
|---|---|---|
| accuracy | 7 targets x {default settings, fixed degree}: test RMSE, nRMSE, signal-aware max relative error, D, degree, cond(M), fit time, 1-point and 1000-point inference time, whether the default sweep reached a rung with D >= N | see RESULTS.md |
| baselines | same data: PolynomialFeatures+LinearRegression (coefficients compared to MomentEmu), PolynomialFeatures+RidgeCV, GP (RBF-ARD+White, N<=2000), MLP | GP dominates; skip in CI |
| scaling | fit time vs (n, d) grid; inference vs D and vs m with the Phi-build / matmul split | ~3 min |
| memory | peak RSS and tracemalloc peak of a fit vs N and batch_size, one subprocess per cell | ~1 min |
| pins | 5 fixed-seed fits: test RMSE + first 10 coefficients at full precision; in-process repeat drift and batch_size (summation order) drift | seconds |
| noise | 7 repeats of two fits and their inference timings: coefficient of variation and max/min | seconds |
| backends | numpy vs JAX-jit vs Torch inference on the CMB-like emulator | seconds |

## Targets (bench/targets.py)

| name | n | m | box | fixed d | N_train | structure |
|---|---|---|---|---|---|---|
| ishigami | 3 | 1 | [-pi, pi]^3 | 9 | 4000 | sin terms, x1*x3^4 interaction |
| sobol_g | 8 | 1 | [0,1]^8 | 5 | 4000 | a=[0,1,4.5,9,99,99,99,99]; |4x-2| kinks |
| friedman | 10 | 1 | [0,1]^10 | 4 | 4000 | 5 active + 5 inert inputs |
| rosenbrock | 4 | 1 | [-2,2]^4 | 4 | 4000 | exact quartic; degree-4 fit must be exact |
| log_rosenbrock | 4 | 1 | [-2,2]^4 | 8 | 4000 | log(1+f): curved valley |
| gauss_peak | 3 | 1 | [0,1]^3 | 10 | 4000 | exp(-r^2/2 sigma^2), sigma=0.15 |
| cmb_like | 6 | 2000 | round-1 box | 5 | 5000 | round-1 synthetic D_ell, same RNG stream as proto_lowrank.py |

All fits pass an explicit test set (`X_test`, `Y_test`) so that `PolyEmu`
does not call its unseeded `train_test_split`; "default settings" means every
other constructor argument at its default. Timings: `time_call` runs one
warm-up call, sizes a block to >= 0.2 s, repeats 5 blocks and reports the
per-call minimum (median also stored in the JSON).

## Gate

`bench.gate` compares two result directories and exits 1 on any failure:

| check | tolerance flag | default |
|---|---|---|
| fit_s, infer_single_us, infer_batch_us, scaling cells, rss_peak_MB | `--time-tol` (relative) | 0.30 |
| timings below `--time-floor-us` | absolute slack of one floor | 50 us |
| accuracy test_rmse | `--rmse-tol` (relative increase) | 1e-6 |
| pins: test_rmse, first-10 coefficients (relative to max|c|) | `--pin-tol` | 1e-10 |
| degree, D_final, pin D | exact match | |

What the tolerances mean against measured drift (numbers from `results/`):

- In-process repeat of a fit reproduces every coefficient bit-for-bit
  (`drift_repeat_rel = 0` on all 5 pins), so 1e-10 is a valid gate on one
  machine with one BLAS.
- Changing `batch_size` changes the summation order of M and moves the
  coefficients by up to ~1e-12 relative (pin table, `drift batch=256`), which
  is below 1e-10. A different BLAS or CPU changes summation order similarly,
  so a baseline recorded on one runner image is valid only on that image; the
  pins are also expected to drift by ~cond(M) x 1e-16 across BLAS builds,
  which for cond(M) ~ 1e8 is ~1e-8 and would fail the 1e-10 gate. Record the
  baseline on the CI image, not on a laptop.
- Timing noise on this machine (M3 Ultra, idle): fit CV 0.5-3 %, max/min
  1.01-1.07 over 7 repeats; inference CV ~1 %. A 30 % tolerance is 4-10x the
  local noise.

Hardware noise on GitHub-hosted runners is not measured here. Public
observations from projects that run benchmarks on hosted runners (pytest-
benchmark, asv users, the CPython benchmarks) are consistent: 2-4 vCPU shared
VMs, run-to-run variation of 10-30 % on sub-second timings, occasional 2x
outliers, and a step change when the runner image or CPU generation changes.
The workflow in `bench/ci/benchmark.yml` therefore:

1. pins the runner image (`ubuntu-24.04`) and the BLAS thread count
   (`OMP_NUM_THREADS=2`), so timings compare like with like;
2. records the baseline as the per-cell minimum of three runs
   (`bench.merge_min`) and compares one new run against it with
   `--time-tol 0.50` and `--time-floor-us 100`, so a single upward-noise
   run does not fail the gate while a 2x regression does;
3. gates accuracy and pins at their exact tolerances (1e-6 relative RMSE,
   1e-10 pins), which are unaffected by runner noise;
4. leaves GP/MLP baselines out of CI (the GP alone is 2-3 minutes at m=2000)
   and runs them on demand.

If a change is meant to alter numerics (a different solver, a different
basis), refresh the baseline in the same PR via the manual
`refresh-baseline` job, and say so in the PR description.
