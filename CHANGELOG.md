# Changelog

All notable changes to MomentEmu are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MomentEmu.recommend` (T-004): `recommend(X, Y, budget=12)` reads the data,
  chooses the estimator, the preconditioner order, the input scaling and the
  basis family, and returns the fitted model as a `Recommendation` that
  predicts in its place. Every mechanism it selects among already existed;
  reaching one required knowing in advance that the target was a ridge, or
  additively separable, or sparse in an orthogonal basis.

  Nothing is decided from a statistic. Each candidate is fitted and scored on
  one held-out split, because cond(M) values at which the normal equations were
  fine and at which they lost five orders of accuracy overlap by six orders of
  magnitude. Selection is on parsimony, the fewest coefficients among the
  candidates within `parsimony_tol` of the best error, because a variance
  target returned rank 6 where rank 2 at a higher degree was 38x smaller and
  more accurate. Stages run in increasing cost and stop when `budget` is spent.
  Stages 2 and 3 tune the stage-1 winner rather than re-running the search,
  which is greedy in one axis; `report()` says so, and names every candidate
  with its score and every stage that was cut, so a truncated search is visible
  rather than silent.

- `active_subspace(jacobian=...)` and `active_subspace(gradient_covariance=...)`
  (T-005), reaching `ActiveSubspaceEmu`, `scan_rank`, `PreconditionedEmu` and
  `recommend`: the rotation from the caller's own derivatives instead of a
  pilot polynomial. `jacobian` takes a callable evaluated batch by batch on raw
  X, a precomputed `(N, m, n)` array, or `(N, n)` for a single output;
  `gradient_covariance` takes a ready `(n, n)` `C = E[J^T J]`. The
  standardisation chain rule is applied inside, so the derivatives stay in the
  caller's own units. The pilot is an error source and not only a cost: on a
  target with a 2-D active subspace in 7 parameters, the exact Jacobian held
  the five dead directions below 1e-12 of the spectrum, where a degree-3 pilot
  leaks above 1e-6 and raising the pilot degree does not fix it.

  Under `PreconditionedEmu(order="warp_rotate")` the rotation is computed in
  the warped coordinates, so the warp's own derivative is applied to the
  supplied Jacobian there and a callable is always evaluated on raw X whichever
  order is being scored. Omitting that division is silent: on a
  `tanh(a^T log x)` ridge over three decades the leading direction came out
  `[0.99998, -0.003, 0.006]` against an analytic truth of
  `[0.227, -0.098, 0.969]`, while the reported leading variance share stayed at
  0.9992. A `gradient_covariance` cannot make that crossing, because moving an
  average of `J^T J` into warped coordinates needs the per-sample Jacobian the
  average has already summed away; that order raises when named and is dropped
  from the candidates under `order="auto"`.

- `rotation.gradient_covariance_matrix`: the standardised, symmetrised
  `C = E[J^T J]` that `active_subspace` diagonalises, as a value rather than an
  intermediate, so one covariance serves many rotations of the same data.

- `warp.Warp.derivative`: analytic `du/dx` for every shape family and for the
  log pre-transform. It differentiates what `__call__` computes, clamps
  included, so a saturated region returns 0 rather than `inf` or `NaN`.

- `core.triangular_cond` and a cond(Phi) report on the QR path. cond(M) =
  cond(Phi)^2 and the accuracy of a QR solve depends on cond(Phi), but above
  cond(M) ~ 1e16 the smallest eigenvalue of the COMPUTED M is round-off, so the
  cond(M) estimate saturates near 1/eps and stops tracking Phi: a 1-D monomial
  design at N = 200 reported 5.9e8 at both cond(Phi) = 1.1e11 and 1.3e14, and
  every existing guard is stated on that saturating number.
  `triangular_cond` runs LAPACK's triangular condition estimator on the `R` the
  QR route already computed, an `O(D^2)` call against the QR's `O(N D^2)`, and
  tracked the true 2-norm condition number to within a factor 1.92 to 2.21 over
  eight decades. Above `COND_PHI_WARN = 1e12` the solve warns with cond(Phi)
  and the digits it leaves. Reporting only: a ceiling on cond(Phi) was
  implemented and then removed, because it refused fits that were accurate.

- `SparseEmu(basis_kind=...)`. `sparse` carried its own orthonormal Legendre
  construction alongside the one in `monomials`; the two agreed to 1.5e-15, so
  the copy was removable. The greedy selection, the held-out path, the final
  fit and the prediction now all come from one plan, which makes selecting in
  one basis and predicting in another unreachable. The default stays
  `legendre`, reproducing the previous figure of merit on the 21cmGEM benchmark
  to four decimals (1.4478 percent, against 1.4555 for chebyshev and 1.5976 for
  monomial), so the parameter buys no accuracy there.

- `PolyEmu.interaction_graph()` (T-001): the full n x n pairwise interaction
  matrix from the orthonormal Legendre projection, plus the connected
  components ("blocks") it implies. Unlike `sobol_report`, which reports only
  the ten largest pairs of support exactly two, this counts every term whose
  support contains both parameters, so a pure three-way coupling registers on
  all three of its pairs and no cross-block pair is crowded out by in-block
  pairs. The threshold is a VARIANCE share, so it is the square of the
  amplitude share. The projection only sees interactions it can represent: an
  even coupling such as x_i^2 x_j^2 needs degree >= 4. A projection explaining
  less than `min_explained` of Var(Y) warns.
- `Basis(blocks=...)` and `Basis.separable(blocks, degree=...)` (T-001): an
  index set in which every support lies inside one block of a partition of the
  parameters, which is the index set an additively separable
  f(theta) = sum_k f_k(theta_{B_k}) needs. The count drops from C(p+d, d) to
  1 + sum_k [C(p_k+d, d) - 1]: 230,230 to 1,845 at p=20, d=6 with four blocks
  of five, and the dense moment matrix from 404 GiB to 26 MiB. `blocks` must
  partition the parameters; the per-block enumeration is used directly rather
  than filtering the (d+1)**n product, which is unusable at those sizes. Rows
  stay sorted by (degree, index), so the degree sweep keeps bordering the
  previous moment matrix incrementally.

- `Basis(parity=...)` (T-001): per-parameter `"even"`/`"odd"` power constraint.
  A target entering a parameter only through its square needs no odd powers of
  it. On a 5-parameter degree-7 example the isotropic 792 terms drop to 546
  with `per_parameter` caps alone and to 223 once parity is applied.
- `PolyEmu.degree_profile()` (T-001): reads the Legendre decomposition and
  reports each parameter's highest variance-carrying power and whether only
  even powers carry variance, returning a ready-to-use `Basis`. Parity uses a
  relative odd-versus-even criterion because the odd share of a genuinely even
  parameter is sampling noise whose absolute size depends on N. Only `"even"`
  is inferred: `"odd"` would require the target to be globally odd in that
  parameter, which the powers alone cannot establish.

- `PolyEmu(ridge=...)` and `core.apply_ridge`: Tikhonov regularisation of the
  moment matrix, scaled PER COLUMN as `lam_i = ridge * M_ii`. Off by default.

  This is the one class of remedy the package had no form of. `COND_WARN` and
  `COND_RAISE` report conditioning, `COND_QR` changes the solver, and `Basis`
  and sparse selection cut terms before the fit; none of them addresses a
  basis that is over-complete for the data. QR solves the SAME least-squares
  problem more accurately, while a ridge solves a DIFFERENT, better-posed one.
  With 969 terms on 3,200 samples of an effectively two-dimensional target,
  cond(M) is 2.3e21 and the unregularised Cholesky returns NaN; a ridge of
  1e-12 returns a model with a held-out error of 1.1e-3.

  Per column, not one common lambda. A polynomial moment matrix has a wildly
  uneven diagonal -- on a degree-10 rotated fit of the 21cmGEM benchmark it
  spanned 7.0e-01 to 4.99e+16 -- so a single `ridge * trace(M) / D` crushes the
  low-order terms while barely touching the high-order ones, and the figure of
  merit went from 1.50 to 34.26 percent. Per column it went to 1.49.

  PRESS-LOO ridges the same matrix before factorising, so the leverage it
  reads is the ridge leverage and degree selection scores the model actually
  being fitted. The QR refit is skipped when a ridge is requested, since it
  answers the unregularised question.

- `PolyEmu(basis_kind="legendre"|"chebyshev")` and `monomials.LegendrePlan`,
  `monomials.ChebyshevPlan`, sharing a `_TensorPlan` base. A tensor-product
  family spans exactly what the monomial plan of the same index set spans, so
  the fitted function is identical wherever conditioning is not the limit;
  what changes is the conditioning.

  Which family is right is a property of the DESIGN, not of the basis: a
  family is well conditioned when the design is distributed like the measure
  it is orthogonal under. Legendre suits a design that fills its box evenly,
  which is what a Latin hypercube or uniform box gives, and at degree 14 on
  such a design cond(M) was 9.7 against Chebyshev's 7.7e2 and the monomial
  basis's 1.5e10. Chebyshev is orthogonal under the arcsine weight, which
  concentrates at the edges, so an edge-starved design is its worst case: on
  the 21cmGEM rotated coordinates, which put 1.2 to 5 percent of their samples
  beyond |z| = 0.8 where a uniform design puts 20 percent, it was 12x worse
  conditioned than monomials up to degree 10. Neither is the default.

  Both are defined on [-1, 1] and clip outside it, so they force
  `scaling="box"`, cover the forward model only, and refuse symbolic export
  rather than letting their coefficients be read as monomial ones.

- `PolyEmu(scaling="box")` and `guards.BoxScaler`: map each input column's
  training range onto [-1, 1] instead of dividing by its standard deviation.
  A polynomial basis wants a bounded argument, and dividing by sigma does not
  bound one. It matters most after a rotation, where a coordinate is a
  weighted sum of several inputs: on the 21cmGEM benchmark one reached 11
  standard deviations, making z**14 span 3.7e14. A degree-12 rotated fit there
  scored 5.93 percent under standard scaling -- worse than its own degree 10 --
  against 1.55 percent under the box map, with cond(M) 2.4e29 against 6.8e21.

  It is not uniformly better and is therefore not the default. The gain needs
  both a long reach and a high degree: on a coordinate reaching only 3.5
  standard deviations the box map was 10x worse conditioned at degree 8, level
  at 12, 18x better at 14 and 2151x better at 16. A stored emulator is
  unaffected either way, because the box map has the same
  `(x - mean_) / scale_` form and reloads through `io.ArrayScaler` unchanged.

- `MomentEmu.rotation` (T-002 / P2): `ActiveSubspaceEmu` fits the forward model
  in the leading eigenvectors of the gradient covariance `C = E[J^T J]`, and
  `scan_rank` reports accuracy against coefficient count over a set of ranks.
  A ridge target is low dimensional in the right coordinates while carrying
  interactions at every ANOVA order in the original ones, which is why
  interaction-order truncation saturates on it and rotation does not. Measured
  on a travelling-trough target, rank 2 at degree 12 used 91 coefficients
  against the isotropic degree-7 basis's 3,432 and was 1.6x more accurate.
  `rank="auto"` reaches a variance target and no further, which overshoots on a
  shallow spectrum tail; `scan_rank` is what locates the knee.
- `MomentEmu.factored` (T-002): `FactoredEmu` fits a rank-R canonical product
  over parameter blocks by alternating least squares, for multiplicative
  structure that `Basis(blocks=...)` cannot express. Multiplicative
  separability is a low-rank coefficient tensor over a full tensor-product
  index set, not sparsity, so it needs a multilinear fit rather than a
  different index set. It handles sign-changing data, which the log transform
  cannot. `separability_report` verifies a candidate partition against two
  log-free criteria: `H_ij = d2f/di dj` for additive structure and
  `M_ij = f d2f/di dj - (df/di)(df/dj)`, the numerator of `d2 log f/di dj`, for
  multiplicative structure.
- `guards.connected_components`, shared by the additive and multiplicative
  structure detectors.
- `MomentEmu.precondition` (T-002): `PreconditionedEmu` composes the rotation
  (P2) and the warp (P4), which do not commute, and picks the order from the
  data. Neither order dominates, and which one wins is a property of the
  target:

  A sharp feature along a ROTATED direction is invisible to a per-axis warp,
  because no raw axis carries it. Measured at seven parameters, warping alone
  changed nothing (every axis came back identity, matching the unpreconditioned
  error to six digits), rotation alone reached 17.2 percent, and rotating then
  warping reached 3.71 percent on the same 91 coefficients.

  A ridge in WARPED coordinates is not a ridge in the raw ones. The active
  subspace is a linear projection, so applying it first discards real signal:
  on `tanh(sum a_i log x_i)`, rotation alone scored 36.9 percent against 33.2
  percent for no preconditioning at all -- worse than doing nothing -- while
  warping first concentrated the gradient-covariance spectrum from 0.832 to
  0.961 in its leading direction and the composition then reached 7.06 percent
  at 105 coefficients against 1,716.

  Candidates are scored at the degree each can afford rather than at one shared
  degree, since comparing a two-dimensional fit with a seven-dimensional one at
  a degree the latter can reach hides what rotation is for. `select="accuracy"`
  takes the lowest held-out error; `select="parsimony"` takes the smallest model
  within a tolerance of it, which on the log-ridge target chose 105 coefficients
  over 1,716 for 1.9 times the error.
- `rotation.select_rank` and `ActiveSubspaceEmu.transform`, so the two
  preconditioners share their rank rule and compose in either order.

### Changed

- `guards.COND_QR = 1e13` (T-001): the QR refit is now gated on its own
  threshold instead of reusing `COND_RAISE`. The two were conflated, but they
  answer different questions: `COND_RAISE = 1e16` is where M is numerically
  singular, while forming `M = Phi^T Phi` squares the conditioning, so the
  normal equations start losing digits far earlier. `COND_RAISE` is unchanged,
  so no fit that works today starts raising.

  Measured through the fit path on a 4-parameter target, the stored
  coefficients improved from 4.9e-11 to 2.6e-14 at degree 14 and from 6.3e-10
  to 1.4e-14 at degree 16, both rungs sitting below the old trigger. The
  honest magnitude: predictions were already far inside any practical
  tolerance before the change, so this buys coefficient accuracy rather than
  usable prediction accuracy, and it matters where coefficients are read
  directly -- symbolic export (which already warns from cond 1e8) and
  derivatives. A target limited by truncation rather than conditioning is
  unaffected. Cost is 1.76x on a rung that triggers and nothing on one that
  does not.

- `Basis(degree=...)` now sets the top of the forward degree sweep instead of
  being silently discarded. `PolyEmu(X, Y, basis=Basis(degree=2))` fits degree
  2; previously the sweep ignored the Basis degree and ran to
  `max_degree_forward` (or the sample-count cap), because `Basis.build` takes
  the degree the caller passes in preference to its own. `max_degree_forward`
  sets the same quantity, so passing both with different values now raises
  `ValueError` rather than letting one win silently. `init_deg_forward` sets
  where the sweep starts and is unaffected below the Basis degree; above it,
  it raises. The heuristic default start is lowered to the Basis degree when it
  would overshoot, so `Basis(degree=0)` fits the constant term.

### Fixed

- `press_loo` refit the coefficients with QR above cond(M) = 1e13 but took the
  leverage from `cho_factor(M)`, so the two came from different factorisations.
  Against brute-force leave-one-out refits on a 1-D monomial design at N = 60,
  the LOO relative error went from 1.2e-1 to 4.0e-8 at cond(Phi) = 1.2e8, and
  from NaN to 2.8e-4 at 5.0e9.

- `PreconditionedEmu` scored candidate orders with a dense polynomial fit
  whatever estimator was going to be used, which ranks the coordinates rather
  than the model. The two need not agree: a rotation that helps a dense fit by
  cutting the dimension helps a sparse one less, because sparse selection was
  already paying only for the terms it kept. Orders are now scored with the
  estimator that will be used, and `scan_terms` sizes the sparse one. With
  `estimator="factored"` the rotation orders are left out of the candidate set
  rather than raising part-way through the scan.

- `FactoredEmu` centred its outputs, which cost exactly one rank. Subtracting
  the mean turns `prod_k f_k` into `prod_k f_k - c`, and that is not a product:
  on an exactly rank-one target, rank 1 reached only 7.0e-2 while rank 2
  reached 6e-5. Outputs are now scaled but not centred, and the same target at
  rank 1 reaches 6.0e-5. Per-output scaling is harmless by contrast, because a
  per-column factor is absorbed by the output weights. An additive offset in
  an output still needs a rank slot, which the model supplies because every
  factor carries a constant term; centring did not avoid that, it moved the
  cost onto the product instead.
- `warp` offered a log transform only when an axis spanned a range ratio above
  10. The gain from a log warp rises smoothly from 23x at a ratio of 2 to 691x
  at 10, with no break anywhere, so the threshold blocked the warp exactly
  where it paid and the scan settled for a much weaker sinh or kte substitute.
  Removed: log is now offered whenever it is defined. Nothing is needed on the
  other side, because the evidence margin already rejects a log that does not
  help -- on a response polynomial in x rather than in log x, the scan returned
  identity at every ratio up to 1000, where a log warp would have taken the
  held-out error from 0.0000 to 0.0220.

- `active_subspace` scaled each output column by its own standard deviation
  before accumulating the gradient covariance, which is wrong for the case
  this package targets: one quantity sampled at many points. It makes a
  near-silent channel as important as the loudest. `output_scaling` now
  defaults to "global", centring each output and dividing everything by one
  scalar; "per_output" keeps the old behaviour for genuinely different
  physical quantities.

  The effect is not a dilution but a wrong answer. On a synthetic case with
  three loud channels carrying one direction and thirty-seven channels
  carrying another ten thousand times weaker, per-output scaling returned the
  weak direction as the leading one. On the 21cmGEM benchmark, whose per-bin
  standard deviations run from exactly zero to 88.8, the fix moved a rank-5
  degree-10 fit from 2.27 to 1.50 percent and brought the whole rank-5 column
  into agreement with an independent implementation (2.2495 against 2.2386 at
  degree 8, 1.5044 against 1.4959 at degree 10). The pilot degree, which had
  looked important under the old scaling (2.78 at degree 3 against 2.41 at
  degree 5), turns out not to matter once the scaling is right (2.25 against
  2.23), so its default is unchanged.
- `MomentEmu.warp` (T-002 / P4): `WarpedEmu` and `fit_warps` choose a monotone
  map per input axis and fit the polynomial in those coordinates. The other
  reductions cut the term count at a fixed convergence rate; this moves the
  rate, so it multiplies with them. Polynomial convergence goes as `rho ** -d`,
  with `rho` set by the distance from the interval to the nearest singularity
  in the COMPLEX plane: for `tanh(20 (x - 0.3))` the poles at `0.3 + i pi/40`
  predict a rate of 0.921 against a measured 0.919, and a warp is a conformal
  map that moves them. Measured end to end on a target that is polynomial in
  the logs of two decades-spanning parameters, the held-out error fell from
  138.5 percent to 0.041 percent.

  The scan is over a small single-parameter family (`log`, `sinh`, `kte`) per
  axis rather than a free-form monotone spline. Free arc-length equalisation,
  the obvious default, is degenerate -- equalising by `|f'|` makes the warped
  response linear by construction -- and the blended version that avoids that
  moved the rate only from 0.919 to 0.905. A family that was wrong about the
  sharpness by a factor of 2.5 was worth 44,000x in term count, so the choice
  of family matters far more than the precision of its parameter.

  A candidate must cut the held-out error to 0.95 of what the identity gives
  before it is accepted, because an axis that acts only through an interaction
  has no marginal signal and would otherwise be handed a warp fitted to noise.
  Selection scores the axes jointly by default; the cheaper marginal criterion
  missed the transitions entirely on one test target.
- `MomentEmu.sparse` (T-002 / P3): `SparseEmu` selects the index set *from* the
  response over a large candidate set by simultaneous orthogonal matching
  pursuit, expressing asymmetries no prior truncation can state. On a target
  needing degree 18 in one direction and 2 in another, 60 of 1,621 candidates
  beat the isotropic degree-8 basis of 1,287 terms.

  Two constraints shaped it. A Gram matrix over the candidate set is not an
  option -- `D x D` at 86,976 candidates is 60 GB -- so the search extends a
  Cholesky factor of the active block only, which is `k x k`. And selection
  runs in the orthonormal Legendre basis, not in monomials: greedy selection
  needs a dictionary of low mutual coherence, and on [-1, 1] the monomial
  dictionary of degrees 0-18 has coherence 0.9984 against the Legendre basis's
  0.0086, so the greedy step picks a neighbouring power at random. Sparsity is
  a property of a basis, not of a function; what transfers between bases is
  accuracy per retained term. The index set is shared across outputs, which
  keeps inference a single GEMM.

- `interaction_graph` and `degree_profile` now exclude output columns whose
  variation is at the rounding level of their own magnitude. Such a column's
  Legendre coefficients are noise, and normalising them by their own sum gave
  O(0.1) "shares"; because the default aggregation takes the max over outputs,
  one constant column merged every parameter into a single block. Degenerate
  columns are reported in `degenerate_outputs`, warned about when skipped, and
  raise when all outputs or an explicitly requested output is degenerate.
- `Basis` gained `__setstate__`, so a `Basis` pickled before a field existed
  (directly or nested in a `PolyEmu`) no longer raises `AttributeError` from
  `build`, `spec` or `==`. Same failure class as the B1 guard in
  `PolyEmu._transforms`.
- `Basis.build` validates `blocks` eagerly rather than inside the enumeration
  generator, where the checks would not run until the first item was drawn,
  and rejects an empty inner block.

### Fixed (performance)

- `press_loo` ran two Householder QRs of the same `Phi`, one in
  `normal_equation_factor` for the leverage factor and one through `qr_solve`
  for the coefficient refit. `Phi` is `(N, D)` and a QR is `O(N D^2)`, so the
  duplicate was the dominant cost of every unregularised fit above
  cond(M) = 1e13. `Phi = Q U` with `U` upper holds for both QR routes, so one
  factorisation now yields `U^T U = Phi^T Phi` for the leverage and
  `U^{-1} Q^T Y` for the coefficients.

- `scan_rank` built a fresh `ActiveSubspaceEmu` per rank, so an r-rank scan
  fitted the pilot r times, and with `jacobian=` it called back into the
  caller's model r times. The rotation does not depend on the rank, so the
  gradient covariance is now built once for the whole scan. `PreconditionedEmu`
  likewise shares one rotation between the `rotate` and `rotate_warp`
  candidates: two Jacobian evaluations over the five orders rather than three.

- `Basis.build` enumerated the `(d+1)**n` box and filtered it, so the cost did
  not depend on how small the constrained basis was. At n=7 a
  `max_interaction=1` basis of 85 terms took 56.8 s at degree 12 and did not
  finish at degree 20, which made `max_interaction` and `q_norm` unreachable
  past degree ~10 -- the regime they exist for. At n=9 a `max_interaction=1`
  basis of 64 terms took 125.8 s at degree 7, visiting 134,217,728 candidates
  to keep 11,440.

  It now searches the admissible set by a depth-first descent carrying the
  remaining degree, at a cost set by the answer rather than by the box. Every
  constraint prunes the descent: the degree budget, the per-parameter bounds,
  `max_interaction`, the group limits, and parity, which becomes the step of
  each position rather than a filter over the output. The q-norm prunes with a
  1e-9 relative slack and the exact `d + 1e-12` test still decides, so the
  tie-break stays where it was. The degree-7 basis above now takes under a
  millisecond; at p=20, d=6, `parity="even"` takes 7 ms, `max_interaction=2`
  13 ms and `q=0.5` 5 ms, sizes at which the box is 7**20 candidates. Rows and
  row order are unchanged, checked row for row against the box enumeration
  over 574 constraint combinations including the empty and degenerate corners.

### Notes

- Redundant basis terms do not bias the fit: the true coefficients are zero and
  least squares recovers that in the population limit, so the cost is sample
  count, memory and conditioning rather than accuracy at an ample N. Measured
  on a 5-parameter degree-7 target, caps plus parity cut the basis from 792 to
  223 terms and the test error by 17 percent at N=2000 and 7 percent at N=8000.
- The reduction is exactly lossless when f is block separable and the design is
  a product measure (a Latin hypercube or uniform box qualifies). A coupling
  across blocks is not recoverable afterwards: confirm the blocks with
  `interaction_graph()` first. Separability that holds only in a rotated frame
  gives no reduction in the original coordinates.

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

### Deferred

- The positional PolyEmu(X, Y, ...) constructor remains the supported 2.0.0
  path. The data-less constructor and the runtime deprecation warning are not
  implemented (the fit/predict split is available); planned for a later
  release, not 2.0.0. Candidates review B-item "unimplemented but advertised".

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
