# Changelog

All notable changes to MomentEmu are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-09-11

This release makes the package importable from the repository, fixes several
wrong-fit and wrong-backend paths, and rebuilds the autodiff backends on a
shared monomial plan. It is a breaking release.

### Changed (default behaviour)

- Degree cap (P0.5): max_order is replaced by guards.max_supported_degree; an
  automatic sweep is capped at the largest degree with basis_size(n, d) * 2 <=
  N_train and D < N_train, with a warning. An explicit degree whose basis has
  D >= N_train now raises ValueError instead of fitting a rank-deficient model.
  The backward sweep gets the same cap plus a 1 GiB moment-matrix budget.
- Checked solve and guards (P0.6): solve_emulator_coefficients uses a Cholesky
  factor and a LAPACK dpocon condition estimate, warns at 1e8/1e12 and raises
  IllConditionedError above 1e16 (or when called directly on a non-SPD matrix).
  Fits check the sample count, distinct rows and the D16 per-axis level caps;
  select_best_model masks non-finite RMSE first.
- Mode pruning retired (P1.1, D15): dim_reduction defaults to False;
  dim_reduction=True or per_mode_thres warns and is ignored. On a grid design
  each parameter power is capped at its number of levels minus one.
- LOO selection (P1.5, D14): without X_test the sweep fits on all N and selects
  the degree by exact leave-one-out PRESS. AIC/BIC are no longer used for
  selection; metrics_and_penalties is removed.
- Per-output transforms (P3.6, D4): transform= accepts linear, log, asinh or a
  (forward, inverse) pair per column. log_Y=True maps to all columns log.
- validate() (P3.1, D10) requires sigma or cov.
- forward_degree replaces the misspelled foward_degree; the old name warns.
- cross_validation is deprecated and ignored; a lone X_test/Y_test raises.

### Deprecated (removed in 3.0.0)

- MomentEmu.PolyEmu -> MomentEmu.emulator; MomentEmu.MomentEmu -> MomentEmu.core.
- foward_degree -> forward_degree.
- filter_modes.
- cross_validation, dim_reduction, per_mode_thres.
- predictive_mse_aic_bic -> predictive_rmse_aic_bic (the function has always
  returned the RMSE).
- The positional PolyEmu(X, Y, ...) constructor -> fit/predict (P4.1).

### Added

- MomentEmu.monomials.MonomialPlan (P0.7): a level-wise recursive evaluator
  reused by inference and the backends.
- MomentEmu.guards (P0.5/P0.6): pure numeric guards.
- PolyEmu.jacobian, validate, posterior_bias, posterior_bias_map, refit,
  in_domain, leverage, hat_diagonal, forward_emulator(..., return_std=True),
  save/load/fingerprint.
- MomentEmu.io (P4.2): versioned, sklearn-free .npz persistence.
- JAX (MomentEmu.jax_momentemu, P2.1), Torch (P2.3) and SymPy (P2.4) backends on
  the shared plan, with make_guarded_logdensity (P2.5a) and
  MomentEmu.cobaya_theory (P2.5b).

### Fixed

- pytest now imports the repository package and fails loudly on a stale
  single-file install (P0.1, P0.9).
- Unseeded train/validation splits and lone test arrays (P0.2).
- The JAX Hessian NaN at the training mean and the missing exp/scale_/
  trailing-axis handling in all three backends (P0.8).
- Input validation at fit and predict entry (P1.2), the fully-masked output
  column in the fractional-error diagnostic (P1.3), sweep logging and the
  scale-free RMSE_tol (P1.4), and the training box / extrapolation warning
  (P1.6).

### Packaging

- require-python >= 3.10 (>= 3.11 for the JAX extra), PEP 639 license = MIT,
  dependency floors, and extras jax, torch, all, dev, docs (P4.4, D5).

### Not in this release (D18, TODO T1)

- The shipped PolyCAMB emulators are not refit or reshipped. 2.0.0 still loads
  the pre-2.0 pickles.
