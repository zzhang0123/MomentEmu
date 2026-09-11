# MomentEmu benchmark results

Environment: Apple M3 Ultra (28 cores), macOS-26.5-arm64-arm-64bit, Python 3.12.9, numpy 2.3.5 (accelerate), scipy 1.16.3, sklearn 1.8.0; MomentEmu 9a98c63; 2026-09-11T23:31:24+00:00; load average at start [11.23, 11.48, 10.48]

## 1. Standard test functions

Load average (1/5/15 min) at suite start [8.94, 10.11, 9.76], at end [7.2, 9.55, 9.56]; suite wall time 33 s.

| target | mode | n | m | N | deg | D | fit s | test RMSE | nRMSE | max rel err (SA) | cond(M) | 1-pt us | 1000-pt us | degrees swept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ishigami | default | 3 | 1 | 4000 | 10 | 286 | 0.04 | 5.65e-03 | 1.50e-03 | 0.69 | 1.49e+08 | 31.3 | 662 | 2-10 |
| ishigami | fixed_d9 | 3 | 1 | 4000 | 9 | 220 | 0.0158 | 0.049 | 0.013 | 4.07 | 1.61e+07 | 29.2 | 550 | 9 |
| sobol_g | default | 8 | 1 | 4000 | 4 | 495 | 0.105 | 0.113 | 0.166 | 23.8 | 9.73e+03 | 21 | 1.46e+03 | 1-5 |
| sobol_g | fixed_d5 | 8 | 1 | 4000 | 5 | 1287 | 0.105 | 0.135 | 0.199 | 26.1 | 4.40e+04 | 34.1 | 3.07e+03 | 5 |
| friedman | default | 10 | 1 | 4000 | 4 | 1001 | 0.0996 | 0.0923 | 0.0188 | 0.101 | 2.16e+04 | 27.5 | 3.22e+03 | 1-4 |
| friedman | fixed_d4 | 10 | 1 | 4000 | 4 | 1001 | 0.0877 | 0.0923 | 0.0188 | 0.101 | 2.16e+04 | 27.5 | 3.21e+03 | 4 |
| rosenbrock | default | 4 | 1 | 4000 | 4 | 70 | 9.40e-03 | 8.48e-12 | 7.66e-15 | 1.13e-12 | 1.63e+03 | 18.4 | 220 | 2-4 |
| rosenbrock | fixed_d4 | 4 | 1 | 4000 | 4 | 70 | 6.92e-03 | 8.48e-12 | 7.66e-15 | 1.13e-12 | 1.63e+03 | 18.5 | 221 | 4 |
| log_rosenbrock | default | 4 | 1 | 4000 | 8 | 495 | 0.288 | 0.157 | 0.154 | 0.579 | 5.58e+06 | 28.6 | 1.24e+03 | 2-12 |
| log_rosenbrock | fixed_d8 | 4 | 1 | 4000 | 8 | 495 | 0.0348 | 0.157 | 0.154 | 0.579 | 5.58e+06 | 28.5 | 1.3e+03 | 8 |
| gauss_peak | default | 3 | 1 | 4000 | 14 | 680 | 0.832 | 9.00e-03 | 0.075 | 3 | 8.76e+11 | 40.5 | 1.69e+03 | 2-20 |
| gauss_peak | fixed_d10 | 3 | 1 | 4000 | 10 | 286 | 0.0192 | 0.0144 | 0.12 | 16.7 | 1.64e+08 | 30.9 | 800 | 10 |
| cmb_like | default | 6 | 2000 | 5000 | 5 | 462 | 0.239 | 0.848 | 2.29e-03 | 0.0367 | 1.47e+04 | 89.2 | 4.58e+03 | 2-5 |
| cmb_like | fixed_d5 | 6 | 2000 | 5000 | 5 | 462 | 0.184 | 0.848 | 2.29e-03 | 0.0367 | 1.47e+04 | 89 | 4.59e+03 | 5 |

## 2. Baselines on the same data

Load average (1/5/15 min) at suite start [7.2, 9.55, 9.56], at end [13.42, 11.5, 10.33]; suite wall time 326 s.

| target | method | fit s | test RMSE | nRMSE | max rel err (SA) | 1-pt us | 1000-pt us | note |
|---|---|---|---|---|---|---|---|---|
| ishigami | momentemu | 0.017 | 0.049 | 0.013 | 4.07 | 28.6 | 610 | d=9, D=220 |
| ishigami | poly_lr | 0.0266 | 0.049 | 0.013 | 4.07 | 103 | 585 | coef max rel diff 5.0e-11; pred diff 2.7e-11 |
| ishigami | poly_normal_eq | 2.32e-03 | 0.049 | 0.013 | 4.07 | 85.6 | 539 | normal equations 0.000s |
| ishigami | poly_ridgecv | 3.65 | 0.0488 | 0.0129 | 3.93 | 103 | 557 | alpha=1.0e-02 |
| ishigami | gp_rbf_white | 9.52 | 8.10e-05 | 2.15e-05 | 0.0142 | 101 | 1.16e+04 | N=2000 |
| ishigami | mlp | 6.57 | 0.0779 | 0.0206 | 5.47 | 71 | 568 | (64, 64) tanh, 1000 iters |
| sobol_g | momentemu | 0.114 | 0.135 | 0.199 | 26.1 | 34.2 | 3.59e+03 | d=5, D=1287 |
| sobol_g | poly_lr | 0.276 | 0.135 | 0.199 | 26.1 | 121 | 2.26e+03 | coef max rel diff 1.0e-13; pred diff 9.4e-14 |
| sobol_g | poly_normal_eq | 0.0223 | 0.135 | 0.199 | 26.1 | 102 | 2e+03 | normal equations 0.012s |
| sobol_g | poly_ridgecv | 3.52 | 0.136 | 0.2 | 26.4 | 122 | 2.31e+03 | alpha=1.0e-09 |
| sobol_g | gp_rbf_white | 15.1 | 0.0404 | 0.0596 | 13.7 | 101 | 1.37e+04 | N=2000 |
| sobol_g | mlp | 4.53 | 0.0203 | 0.03 | 4.11 | 73.8 | 637 | (64, 64) tanh, 641 iters |
| friedman | momentemu | 0.0892 | 0.0923 | 0.0188 | 0.101 | 28.3 | 3.65e+03 | d=4, D=1001 |
| friedman | poly_lr | 0.182 | 0.0923 | 0.0188 | 0.101 | 120 | 2.17e+03 | coef max rel diff 1.5e-13; pred diff 2.0e-14 |
| friedman | poly_normal_eq | 0.0184 | 0.0923 | 0.0188 | 0.101 | 103 | 2.52e+03 | normal equations 0.007s |
| friedman | poly_ridgecv | 3.73 | 0.0923 | 0.0188 | 0.102 | 118 | 2.42e+03 | alpha=1.0e-01 |
| friedman | gp_rbf_white | 22.3 | 2.88e-04 | 5.86e-05 | 7.72e-04 | 105 | 1.44e+04 | N=2000 |
| friedman | mlp | 3.96 | 0.0468 | 9.54e-03 | 0.0475 | 72.5 | 636 | (64, 64) tanh, 565 iters |
| rosenbrock | momentemu | 7.52e-03 | 8.48e-12 | 7.66e-15 | 1.13e-12 | 18.7 | 229 | d=4, D=70 |
| rosenbrock | poly_lr | 7.83e-03 | 1.54e-12 | 1.39e-15 | 1.16e-13 | 93 | 294 | coef max rel diff 1.4e-14; pred diff 5.4e-15 |
| rosenbrock | poly_normal_eq | 1.38e-03 | 2.83e-11 | 2.56e-14 | 8.55e-12 | 77 | 304 | normal equations 0.000s |
| rosenbrock | poly_ridgecv | 3.69 | 2.02e-10 | 1.83e-13 | 6.56e-11 | 92.4 | 347 | alpha=1.0e-10 |
| rosenbrock | gp_rbf_white | 13.4 | 0.0149 | 1.34e-05 | 1.09e-03 | 114 | 1.20e+04 | N=2000 |
| rosenbrock | mlp | 3.73 | 20.2 | 0.0182 | 1.66 | 72.4 | 584 | (64, 64) tanh, 551 iters |
| log_rosenbrock | momentemu | 0.0376 | 0.157 | 0.154 | 0.579 | 28.8 | 1.31e+03 | d=8, D=495 |
| log_rosenbrock | poly_lr | 0.0605 | 0.157 | 0.154 | 0.579 | 111 | 1.04e+03 | coef max rel diff 1.8e-11; pred diff 2.9e-12 |
| log_rosenbrock | poly_normal_eq | 7.36e-03 | 0.157 | 0.154 | 0.579 | 90.5 | 944 | normal equations 0.002s |
| log_rosenbrock | poly_ridgecv | 3.79 | 0.15 | 0.147 | 0.6 | 109 | 1.01e+03 | alpha=1.0e+00 |
| log_rosenbrock | gp_rbf_white | 18.8 | 0.0776 | 0.0758 | 0.249 | 106 | 1.22e+04 | N=2000 |
| log_rosenbrock | mlp | 6.85 | 0.0558 | 0.0546 | 0.219 | 72.9 | 574 | (64, 64) tanh, 1000 iters |
| gauss_peak | momentemu | 0.0205 | 0.0144 | 0.12 | 16.7 | 31.7 | 834 | d=10, D=286 |
| gauss_peak | poly_lr | 0.0352 | 0.0144 | 0.12 | 16.7 | 108 | 803 | coef max rel diff 1.1e-09; pred diff 3.8e-10 |
| gauss_peak | poly_normal_eq | 2.76e-03 | 0.0144 | 0.12 | 16.7 | 90.7 | 679 | normal equations 0.001s |
| gauss_peak | poly_ridgecv | 4.14 | 0.0137 | 0.115 | 16.3 | 108 | 667 | alpha=1.0e-02 |
| gauss_peak | gp_rbf_white | 20.4 | 5.29e-06 | 4.41e-05 | 2.97e-03 | 93.4 | 1.23e+04 | N=2000 |
| gauss_peak | mlp | 4.2 | 1.88e-03 | 0.0157 | 6.42 | 69.3 | 539 | (64, 64) tanh, 636 iters |
| cmb_like | momentemu | 0.184 | 0.848 | 2.29e-03 | 0.0367 | 92.8 | 5.67e+03 | d=5, D=462 |
| cmb_like | poly_lr | 0.311 | 0.848 | 2.29e-03 | 0.0367 | 151 | 6.93e+03 | coef max rel diff 6.3e-13; pred diff 7.6e-14 |
| cmb_like | poly_normal_eq | 0.013 | 0.848 | 2.29e-03 | 0.0367 | 132 | 6.38e+03 | normal equations 0.006s |
| cmb_like | poly_ridgecv | 9.21 | 0.849 | 2.30e-03 | 0.0373 | 149 | 6.98e+03 | alpha=1.0e-09 |
| cmb_like | gp_rbf_white | 113 | 7.23e-03 | 1.95e-05 | 3.16e-04 | 484 | 2.37e+04 | hyper on 20 cols, N=2000 |
| cmb_like | mlp | 24.4 | 4.72 | 0.0128 | 0.0555 | 83.6 | 5.3e+03 | (128, 128) tanh, 199 iters |

## 3. Scaling curves

Load average (1/5/15 min) at suite start [13.42, 11.5, 10.33], at end [13.77, 11.88, 10.58]; suite wall time 90 s.

Fit time in seconds, min of 3 (N = 10000, single output, fixed degree, dim_reduction off; cells with D > 5000 skipped):
| n \ d | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| 2 | 4.76e-03 | 5.55e-03 | 6.30e-03 | 6.53e-03 | 7.32e-03 | 6.96e-03 | 8.50e-03 |
| 3 | 6.16e-03 | 7.18e-03 | 8.23e-03 | 9.86e-03 | 0.0124 | 0.014 | 0.0209 |
| 4 | 7.03e-03 | 9.03e-03 | 0.0116 | 0.015 | 0.0235 | 0.0352 | 0.0477 |
| 5 | 8.42e-03 | 0.0107 | 0.0151 | 0.0253 | 0.0451 | 0.0819 | 0.143 |
| 6 | 9.48e-03 | 0.0144 | 0.0227 | 0.0459 | 0.101 | 0.211 | 0.468 |
| 7 | 0.0106 | 0.0174 | 0.0378 | 0.0879 | 0.218 | 0.615 | - |
| 8 | 0.0106 | 0.0198 | 0.0558 | 0.157 | 0.498 | - | - |
| 9 | 0.0153 | 0.0277 | 0.0851 | 0.281 | - | - | - |
| 10 | 0.0144 | 0.036 | 0.12 | 0.512 | - | - | - |

Basis size D = C(n+d, d):
| n \ d | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| 2 | 6 | 10 | 15 | 21 | 28 | 36 | 45 |
| 3 | 10 | 20 | 35 | 56 | 84 | 120 | 165 |
| 4 | 15 | 35 | 70 | 126 | 210 | 330 | 495 |
| 5 | 21 | 56 | 126 | 252 | 462 | 792 | 1287 |
| 6 | 28 | 84 | 210 | 462 | 924 | 1716 | 3003 |
| 7 | 36 | 120 | 330 | 792 | 1716 | 3432 | 6435 |
| 8 | 45 | 165 | 495 | 1287 | 3003 | 6435 | 12870 |
| 9 | 55 | 220 | 715 | 2002 | 5005 | 11440 | 24310 |
| 10 | 66 | 286 | 1001 | 3003 | 8008 | 19448 | 43758 |

Inference vs D (n = 6, m = 1000). Phi share = fraction of the single-point cost spent building the design row:
| D | 1-pt us | Phi build us | matmul us | Phi share | 1000-pt us | per-sample us | coeff MB |
|---|---|---|---|---|---|---|---|
| 7 | 29.2 | 17.7 | 1.26 | 0.93 | 1.41e+03 | 1.41 | 0.056 |
| 28 | 39.2 | 70.3 | 5.89 | 0.92 | 1.54e+03 | 1.54 | 0.22 |
| 84 | 48.7 | 220 | 10.3 | 0.96 | 1.87e+03 | 1.87 | 0.67 |
| 210 | 53.3 | 576 | 13 | 0.98 | 2.47e+03 | 2.47 | 1.7 |
| 462 | 59.4 | 1.32e+03 | 18.9 | 0.99 | 3.77e+03 | 3.77 | 3.7 |
| 924 | 72.8 | 2.76e+03 | 27.8 | 0.99 | 6.24e+03 | 6.24 | 7.4 |
| 1716 | 121 | 5.27e+03 | 52.5 | 0.99 | 1.01e+04 | 10.1 | 14 |
| 3003 | 315 | 9.57e+03 | 264 | 0.97 | 1.66e+04 | 16.6 | 24 |

Inference vs m (n = 6, d = 5, D = 462):
| m | 1-pt us | Phi build us | matmul us | Phi share | 1000-pt us | per-sample us | coeff MB |
|---|---|---|---|---|---|---|---|
| 1 | 23.3 | 1.35e+03 | 0.556 | 1 | 1.76e+03 | 1.76 | 3.7e-03 |
| 10 | 24.2 | 1.34e+03 | 1.18 | 1 | 1.85e+03 | 1.85 | 0.037 |
| 100 | 34.2 | 1.33e+03 | 4.19 | 1 | 2.12e+03 | 2.12 | 0.37 |
| 1000 | 58.5 | 1.34e+03 | 18.8 | 0.99 | 3.8e+03 | 3.8 | 3.7 |
| 10000 | 671 | 1.34e+03 | 426 | 0.76 | 1.77e+04 | 17.7 | 37 |

## 3d. Peak memory of a fit vs N and batch_size (n=6, d=5, D=462, m=100)

Load average (1/5/15 min) at suite start [11.23, 11.48, 10.48], at end [9.34, 11.02, 10.35]; suite wall time 27 s.

| N | batch_size | dim_reduction | full Phi MB | tracemalloc peak MB | RSS peak MB | RSS peak - RSS before fit MB | fit s |
|---|---|---|---|---|---|---|---|
| 2000 | None | off | 7.39 | 90.2 | 286 | 238 | 2.49 |
| 2000 | 1000 | off | 7.39 | 83.2 | 275 | 227 | 2.44 |
| 2000 | 10000 | off | 7.39 | 90.2 | 293 | 245 | 2.52 |
| 10000 | None | off | 37 | 159 | 458 | 373 | 2.54 |
| 10000 | 1000 | off | 37 | 90 | 400 | 331 | 2.5 |
| 10000 | 10000 | off | 37 | 159 | 439 | 363 | 2.41 |
| 50000 | None | off | 185 | 227 | 630 | 463 | 2.54 |
| 50000 | 1000 | off | 185 | 155 | 567 | 401 | 2.64 |
| 50000 | 10000 | off | 185 | 227 | 637 | 471 | 2.5 |
| 50000 | 1000 | on | 185 | 155 | 575 | 410 | 2.63 |

## 4. Regression pins

Load average (1/5/15 min) at suite start [9.34, 11.02, 10.35], at end [9.34, 11.02, 10.35]; suite wall time 1 s.

| pin | D | m | test RMSE | c[0,0] | cond(M) | drift repeat | drift batch=256 |
|---|---|---|---|---|---|---|---|
| pin1_ishigami_d6 | 84 | 1 | 5.460225659515e-01 | -7.79275992741590873e-01 | 5.05e+04 | 0 | 6.2e-13 |
| pin2_sobolg_d3 | 165 | 1 | 2.682269365601e-01 | -1.63279462321297064e+00 | 304 | 0 | 1.7e-14 |
| pin3_friedman_d3 | 286 | 1 | 3.459815984364e-01 | +9.51741788085650464e-02 | 479 | 0 | 2.2e-14 |
| pin4_cmb_d4 | 210 | 2000 | 2.429907869053e+00 | -5.53598340717630089e-02 | 4.61e+03 | 0 | 6.4e-14 |
| pin5_map2_fwd_bwd_d6 | 28 | 2 | 3.962646023755e-07 | -9.52615286742536046e-04 | 2.08e+04 | 0 | 2.1e-13 |

## 5. Timing noise on this machine

Load average (1/5/15 min) at suite start [9.34, 11.02, 10.35], at end [8.29, 10.71, 10.25]; suite wall time 13 s.

| target | repeats | fit s | fit CV | fit max/min | min-of-3 fit CV | min-of-3 fit max/min | 1-pt us | 1-pt CV | 1-pt max/min | 1000-pt us | 1000-pt CV | 1000-pt max/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ishigami | 7 | 0.0168 | 3.2% | 1.1 | 1.9% | 1.06 | 29.2 | 0.5% | 1.01 | 579 | 2.2% | 1.08 |
| cmb_like | 7 | 0.174 | 1.7% | 1.04 | 1.3% | 1.03 | 90.9 | 0.4% | 1.01 | 5.36e+03 | 1.9% | 1.06 |

## 6. Autodiff backends, single-point and batch inference on the CMB-like emulator

Load average (1/5/15 min) at suite start [8.29, 10.71, 10.25], at end [7.77, 10.52, 10.18]; suite wall time 9 s.

| backend | 1-pt us | 1000-pt us | per-sample us | max rel diff vs numpy | note |
|---|---|---|---|---|---|
| numpy (PolyEmu.forward_emulator) | 90 | 5.4e+03 | 5.4 | 0 | D=462, m=2000 |
| jax (create_jax_emulator, jit, x64) | 88.2 | 3.71e+03 | 3.71 | 2.4e-15 | first-call (compile) single 64 ms, batch 69 ms; 0.10.0, cpu |
| torch (TorchMomentEmu, float64, no_grad) | 868 | 5.81e+03 | 5.81 | 3.0e-15 | torch 2.11.0 |
