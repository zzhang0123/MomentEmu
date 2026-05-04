import numpy as np
from itertools import combinations_with_replacement
from collections import Counter
import sympy as sp
from logging import warning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from MomentEmu.MomentEmu import (
    generate_moment_products,
    solve_emulator_coefficients,
    predictive_mse_aic_bic,
    select_best_model,
    filter_modes,
    signal_aware_frac_err,
)



####### Multi-index generation and operations ####

def given_order_indices(n, d):
    """Generate all multi-indices α with total degree = d.
    
    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for c in combinations_with_replacement(range(n), d):
        counter = Counter(c)
        alpha = [counter[i] for i in range(n)]
        indices.append(tuple(alpha))
    return np.array(indices)

def generate_multi_indices(n, d):
    """Generate all multi-indices α with total degree ≤ d.
    
    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for deg in range(d + 1):
        for c in combinations_with_replacement(range(n), deg):
            counter = Counter(c)
            alpha = [counter[i] for i in range(n)]
            indices.append(tuple(alpha))
    return np.array(indices)

def indices_selection(multi_indices, d_vec):
    """Select multi-indices where each component is ≤ corresponding component in d_vec.
    
    Args:
        multi_indices: array of multi-indices (each row is a multi-index)
        d_vec: vector of maximum degrees for each variable
        
    Returns:
        filtered array of multi-indices
    """
    # Convert to numpy array if not already
    multi_indices = np.array(multi_indices)
    d_vec = np.array(d_vec)
    
    # Check each multi-index: all components must be ≤ corresponding d_vec components
    mask = np.all(multi_indices <= d_vec, axis=1)
    
    return multi_indices[mask]

def generate_multi_indices_with_degree_vec(d_vec):
    """Generate all multi-indices α where α[i] ≤ d_vec[i] for each variable i.
    
    Args:
        d_vec: vector of maximum degrees for each variable
        
    Returns:
        array of multi-indices
    """
    from itertools import product
    
    # Convert to numpy array if not already
    d_vec = np.array(d_vec)
    n = len(d_vec)
    
    # Generate all combinations using Cartesian product
    # For each variable i, generate range(0, d_vec[i] + 1)
    ranges = [range(d_vec[i] + 1) for i in range(n)]
    
    # Use itertools.product to get all combinations
    indices = list(product(*ranges))
    
    return np.array(indices)

####### Monomial/Polynomial functions ############
def evaluate_monomials(X, multi_indices):
    """Evaluate φ_α(X) for all samples and all α."""
    N, n = X.shape
    D = len(multi_indices)
    Phi = np.empty((N, D), dtype=X.dtype)
    for j, alpha in enumerate(multi_indices):
        Phi[:, j] = np.prod(X ** alpha, axis=1)
    return Phi  # shape: N x D

def evaluate_monomials_lazy(X, multi_indices):
    """
    Efficiently evaluate monomials using on-demand caching to reduce memory use.
    """
    N, n = X.shape
    D = len(multi_indices)
    
    # Cache only needed powers: (i, d) -> X[:, i] ** d
    power_cache = {}
    
    Phi = np.empty((N, D), dtype=X.dtype)
    for j, alpha in enumerate(multi_indices):
        phi_j = np.ones(N, dtype=X.dtype)
        for i, deg in enumerate(alpha):
            if deg == 0:
                continue
            key = (i, deg)
            if key not in power_cache:
                power_cache[key] = X[:, i] ** deg
            phi_j *= power_cache[key]
        Phi[:, j] = phi_j
    return Phi

######## New method for solving the memory issue #########
def evaluate_monomials_batched(X, multi_indices, batch_size=10000, function=evaluate_monomials_lazy):
    N, n = X.shape
    D = len(multi_indices)
    Phi = np.empty((N, D), dtype=X.dtype)
    
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        Phi[i:batch_end] = evaluate_monomials_lazy(X[i:batch_end], multi_indices)
    return Phi
###########################################################


def compute_moments_vector_output(X, Y, multi_indices):
    """
    Vector-valued version of moment method.
    X: N x n input parameter array
    Y: N x m observable array
    multi_indices: list of multi-indices
    Returns: moment matrix Mm (D x D), moment vectors ν (D x m)
    """
    n = X.shape[1]

    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D

    Mm, nu = generate_moment_products(Phi, Y)

    return Mm, nu

def compute_moments_vector_output_batched(X, Y, multi_indices, batch_size=10000):
    """
    Memory-efficient version of moment computation using batched processing.
    
    Args:
        X: N x n input parameter array
        Y: N x m observable array  
        multi_indices: list of multi-indices
        batch_size: number of samples to process at once
        
    Returns:
        Mm: moment matrix (D x D)
        nu: moment vectors (D x m)
    """     
    N, n = X.shape
    m = Y.shape[1]
    D = len(multi_indices)
    
    # Initialize accumulators
    Mm = np.zeros((D, D), dtype=X.dtype)
    nu = np.zeros((D, m), dtype=X.dtype)
    
    # Process data in batches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_size_actual = end_idx - start_idx
        
        # Evaluate monomials for this batch
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        Phi_batch = evaluate_monomials_lazy(X_batch, multi_indices)  # batch_size x D
        
        # Accumulate moment matrix: M += Phi_batch.T @ Phi_batch
        Mm += Phi_batch.T @ Phi_batch
        
        # Accumulate moment vector: nu += Phi_batch.T @ Y_batch
        nu += Phi_batch.T @ Y_batch
        
        # Clear batch from memory
        del Phi_batch, X_batch, Y_batch
    
    # Normalize by total number of samples
    Mm /= N
    nu /= N
    
    return Mm, nu

def symbolic_polynomial_expressions(coeffs, multi_indices, variable_names=None, 
                                    input_means=None, input_vars=None, 
                                    output_means=None, output_vars=None):
    """
    Convert emulator coefficients into sympy expressions.
    coeffs: D x m (number of basis terms × number of outputs)
    multi_indices: list of α
    Returns: list of sympy expressions, one per output dimension
    """
    D, m = coeffs.shape
    n = len(multi_indices[0])
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(n)]
    vars_sym = sp.symbols(variable_names)

    if input_vars is not None:
        input_stds = np.sqrt(input_vars)
    else:
        input_stds = None
    if output_vars is not None:
        output_stds = np.sqrt(output_vars)
    else:
        output_stds = None

    expressions = []
    for j in range(m):  # For each output dimension
        expr = 0
        for c, alpha in zip(coeffs[:, j], multi_indices):
            if input_means is not None and input_stds is not None:
                monomial = np.prod([ ( (vars_sym[i] - input_means[i]) / input_stds[i] )**alpha[i] for i in range(n)])
            elif input_means is not None:
                monomial = np.prod([ (vars_sym[i] - input_means[i])**alpha[i] for i in range(n)])
            expr += c * monomial
        if output_means is not None and output_stds is not None:
            expr = expr * output_stds[j] + output_means[j]
        elif output_means is not None:
            expr = expr + output_means[j]
        expressions.append(sp.simplify(expr))
    return expressions

def evaluate_emulator(X, coeffs, multi_indices):
    """
    Evaluate the polynomial emulator at inputs X using known coefficients.
    X: N x n
    coeffs: D x m
    multi_indices: list of α
    Returns: Y_pred: N x m
    """
    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D
    return Phi @ coeffs  # N x m

def evaluate_emulator_batched(X, coeffs, multi_indices, batch_size=10000):
    """
    Memory-efficient emulator evaluation for large datasets.
    """
    N = X.shape[0]
    m = coeffs.shape[1]
    Y_pred = np.empty((N, m), dtype=X.dtype)
    
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        X_batch = X[start_idx:end_idx]
        Phi_batch = evaluate_monomials_lazy(X_batch, multi_indices)
        Y_pred[start_idx:end_idx] = Phi_batch @ coeffs
        del Phi_batch, X_batch
    
    return Y_pred

def max_order(n_params, N_samples):
    import math
    k = 0
    while math.comb(n_params+k,k) < N_samples:
        k += 1
    return k


def _unscale_val(scaler_X, scaler_Y, X_val_scaled, Y_val_scaled, log_Y):
    """Invert StandardScaler (and the optional log transform on Y) on a
    validation tuple. Centralises the three lines that were duplicated at
    the forward and backward fractional-error sites.
    """
    X = scaler_X.inverse_transform(X_val_scaled)
    Y = scaler_Y.inverse_transform(Y_val_scaled)
    if log_Y:
        Y = np.exp(Y)
    return X, Y


def _report_frac_err(
    label,
    pred,
    ref,
    *,
    signal_floor_frac=1e-3,
    absolute_floor=1e-15,
    dr_threshold_decades=3.0,
):
    """Compute and pretty-print the signal-aware fractional-error diagnostic.

    Returns the diagnostic dict from :func:`signal_aware_frac_err` so that the
    caller can attach it to the emulator instance.

    See ``signal_mask.md`` (repository root) for parameter calibration.
    """
    diag = signal_aware_frac_err(
        pred,
        ref,
        signal_floor_frac=signal_floor_frac,
        absolute_floor=absolute_floor,
        dr_threshold_decades=dr_threshold_decades,
    )

    if diag["n_above"] == 0:
        print(
            f"\n{label} emulator fractional-error diagnostic: signal mask empty "
            f"(no entries above floor {diag['floor']!r}); reference appears to be "
            f"at floating-point noise level."
        )
        return diag

    ind = diag["argmax"]
    strategy = diag["strategy"]
    if isinstance(strategy, np.ndarray):
        strategy_label = "per-output: " + ", ".join(str(s) for s in strategy.tolist())
    else:
        strategy_label = str(strategy)

    # When max_rel == 0 every in-mask entry matched exactly; argmax is None.
    # Indexing ref[None] would silently mean ref[np.newaxis] in NumPy, which
    # would dump the whole array instead of a single entry — handle this
    # branch explicitly with no per-entry indexing.
    if ind is None:
        print(
            f"\n{label} emulator signal-aware fractional error:"
            f"\n  max_rel = {diag['max_rel']:.3e}  rmse = {diag['rmse']:.3e}"
            f"\n  all in-mask entries match exactly"
            f"\n  in-mask entries: {diag['n_above']}/{diag['n_total']}"
            f"  | strategy: {strategy_label}"
            f"\n  (entries below the per-output signal floor are excluded; see signal_mask.md)"
        )
        return diag

    print(
        f"\n{label} emulator signal-aware fractional error:"
        f"\n  max_rel = {diag['max_rel']:.3e}  rmse = {diag['rmse']:.3e}"
        f"\n  worst at index {ind}: true = {ref[ind]!r}, predicted = {pred[ind]!r}"
        f"\n  in-mask entries: {diag['n_above']}/{diag['n_total']}"
        f"  | strategy: {strategy_label}"
        f"\n  (entries below the per-output signal floor are excluded; see signal_mask.md)"
    )
    return diag


class PolyEmu():
    def __init__(self, 
                X, 
                Y, 
                X_test=None, 
                Y_test=None, 
                log_Y=False,
                cross_validation=True,
                test_size=0.15, 
                # RMSE_upper=1.0,
                RMSE_tol=1e-2, 
                fRMSE_tol=1e-1,
                forward=True, 
                backward=False,
                init_deg_forward=None,  
                init_deg_backward=None, 
                max_degree_forward=None,
                max_degree_backward=None,
                dim_reduction=True,
                per_mode_thres=None,
                return_max_frac_err=False,
                standardize_Y_with_std=True,
                batch_size=None):
        """
        Polynomial emulator class for both forward and backward emulation.
        X: N x n array of input parameters. N is the number of samples, n is the number of parameters.
        Y: N x m array of observables. m is the number of observables.
        X_test, Y_test: optional test/validation sets. If not provided, a split from X, Y will be used.
        test_size: fraction of data to use for validation if X_test, Y_test not provided.
        RMSE_tol: target RMSE to stop increasing polynomial degree.
        fRMSE_tol: tolerance for selecting best model based on RMSE. We select the simplest model within fRMSE_tol (fractional range) of the lowest RMSE.

        forward: whether to generate forward emulator.
        backward: whether to generate backward emulator.
        init_deg_forward, init_deg_backward: initial polynomial degree for forward/backward emulators.
        max_degree_forward, max_degree_backward: maximum polynomial degree for forward/backward emulators.
        dim_reduction: whether to perform dimension reduction after fitting.

        per_mode_thres: threshold for dimension reduction per mode.
        return_max_frac_err: whether to compute and store maximum fractional error on validation set.
            When enabled, the worst case is stored on ``forward_max_frac_err`` /
            ``backward_max_frac_err`` (signal-mask aware: entries below the per-output
            signal floor are excluded), and the full diagnostic dict on
            ``forward_frac_err_diag`` / ``backward_frac_err_diag``. See ``signal_mask.md``.
        standardize_Y_with_std: whether to standardize Y with standard deviation (True) or only mean (False).
        batch_size: batch size for batched computations to manage memory usage.
        """
        
        self.n_params = X.shape[1]
        self.n_outputs = Y.shape[1]
        self.standardize_Y_with_std = standardize_Y_with_std
        self.log_Y = log_Y

        if batch_size is None:
            batch_size = X.shape[0]

        if X_test is None or Y_test is None:
            if cross_validation:
                # Split into training and validation
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size)
                cross_val = True
            else:
                X_train, Y_train = X, Y
                cross_val = False
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = X_test, Y_test
            cross_val = True

        if self.log_Y:
            Y_train = np.log(Y_train)
            if cross_val:
                Y_val = np.log(Y_val)

        # Scale the training data
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler(with_std=self.standardize_Y_with_std)

        # in-place scaling transformation
        X_train = self.scaler_X.fit_transform(X_train) 
        Y_train = self.scaler_Y.fit_transform(Y_train) 
        if cross_val:
            X_val = self.scaler_X.transform(X_val)
            Y_val = self.scaler_Y.transform(Y_val)
        else:
            # using the training set to define the fitting error
            X_val, Y_val = X_train, Y_train



        if forward:
            print("\n Generating forward emulator...")

            max_deg_forward = max_order(self.n_params, X_train.shape[0])

            if max_degree_forward is None or max_degree_forward > max_deg_forward:
                max_degree_forward = max_deg_forward
                print(f"Set max_degree_forward to {max_degree_forward}. Otherwise, a higher degree will require more samples.")

            self.generate_forward_emulator(
                X_train, 
                Y_train,
                X_val,
                Y_val,
                # RMSE_upper=RMSE_upper,
                RMSE_tol=RMSE_tol, 
                fRMSE_tol=fRMSE_tol,
                init_deg=init_deg_forward, 
                max_degree=max_degree_forward,
                dim_reduction=dim_reduction,
                per_mode_thres=per_mode_thres,
                batch_size=batch_size
            )
            if return_max_frac_err:
                # Convert scaled validation data back to original scale for proper comparison
                X_val_unscaled, Y_val_unscaled = _unscale_val(
                    self.scaler_X, self.scaler_Y, X_val, Y_val, self.log_Y
                )

                Y_val_pred = self.forward_emulator(X_val_unscaled)
                diag = _report_frac_err("Forward", Y_val_pred, Y_val_unscaled)
                self.forward_frac_err_diag = diag
                # Coerce nan -> inf on the public attribute so downstream
                # threshold comparisons (`emu.forward_max_frac_err > tol`) fail
                # loudly when the signal mask is empty. The diag dict still
                # carries the spec-compliant nan for n_above == 0.
                self.forward_max_frac_err = (
                    diag["max_rel"] if diag["n_above"] > 0 else float("inf")
                )

        if backward:
            print("Generating backward emulator...")
            max_deg_backward = max_order(self.n_outputs, X_train.shape[0])

            if max_degree_backward is None or max_degree_backward > max_deg_backward:
                max_degree_backward = max_deg_backward
                print(f"Set max_degree_backward to {max_degree_backward}. Otherwise, a higher degree will require more samples.")

            self.generate_backward_emulator(X_train, 
                                            Y_train,
                                            X_val,
                                            Y_val,
                                            # RMSE_upper=RMSE_upper,
                                            RMSE_tol=RMSE_tol, 
                                            fRMSE_tol=fRMSE_tol,
                                            init_deg=init_deg_backward, 
                                            max_degree=max_degree_backward,
                                            dim_reduction=dim_reduction,
                                            per_mode_thres=per_mode_thres,
                                            batch_size=batch_size)
            if return_max_frac_err:
                # Convert scaled validation data back to original scale for proper comparison
                X_val_unscaled, Y_val_unscaled = _unscale_val(
                    self.scaler_X, self.scaler_Y, X_val, Y_val, self.log_Y
                )

                X_val_pred = self.backward_emulator(Y_val_unscaled)
                diag = _report_frac_err("Backward", X_val_pred, X_val_unscaled)
                self.backward_frac_err_diag = diag
                # See forward branch: coerce nan -> inf so threshold checks
                # fail loudly when the signal mask is empty.
                self.backward_max_frac_err = (
                    diag["max_rel"] if diag["n_above"] > 0 else float("inf")
                )

    def generate_forward_emulator(self, 
                                  X_train_scaled, 
                                  Y_train_scaled,
                                  X_val_scaled,
                                  Y_val_scaled,
                                #   RMSE_upper=0.1,
                                  RMSE_tol=1e-3, 
                                  fRMSE_tol=1e-1, 
                                  init_deg=None, 
                                  max_degree=None,
                                  dim_reduction=False,
                                  per_mode_thres=None,
                                  batch_size=10000):

        if init_deg is None:
            if self.n_params > 6:
                init_deg = 1
            elif self.n_params < 3:
                init_deg = 3
            else:
                init_deg = 2

        assert init_deg <= max_degree, "Initial degree must be less than or equal to max_degree"

        RMSE_val_list = []
        AIC_list = []
        BIC_list = []
        coeffs_list = []
        multi_indices_list = []
        running_time_list = []
        degree_list = []
        import time

        for d in range(init_deg, max_degree + 1):
            start_time = time.time()

            degree_list.append(d)
            if d == init_deg:
                multi_indices = generate_multi_indices(self.n_params, d)
            else:
                aux_indices = given_order_indices(self.n_params, d)
                multi_indices = np.concatenate((multi_indices, aux_indices), axis=0)

            # M, nu = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            # Use batched computation
            M, nu = compute_moments_vector_output_batched(
                X_train_scaled, Y_train_scaled, multi_indices, batch_size=batch_size
            )
            coeffs = solve_emulator_coefficients(M, nu)

            # Y_val_pred = evaluate_emulator(X_val_scaled, coeffs, multi_indices)
            Y_val_pred = evaluate_emulator_batched(
                X_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )

            RMSE_val, AIC, BIC = predictive_mse_aic_bic(Y_val_scaled, Y_val_pred, multi_indices.shape[0], n_train=X_train_scaled.shape[0])
            RMSE_val_list.append(RMSE_val)
            AIC_list.append(AIC)
            BIC_list.append(BIC)

            # if RMSE_val < RMSE_upper: # if the RMSE is lower than the upper bound, we accept the model, and save it for later selection
            #     coeffs_list.append(coeffs)
            #     multi_indices_list.append(multi_indices)
            # else: # If the RMSE exceeds the upper bound, we reject the model straight away.
            #     coeffs_list.append(None) 
            #     multi_indices_list.append(None)

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)

            end_time = time.time()
            running_time = end_time - start_time
            running_time_list.append(running_time)

            if RMSE_val < RMSE_tol:
                self.foward_degree = d
                print(f"Forward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                print(f"Maximum degree {max_degree} reached. Now choose the best fit. ")
                ind = select_best_model(RMSE_val_list, aic_list=AIC_list, bic_list=BIC_list, rmse_tol=fRMSE_tol)
                # assert RMSE_val_list[ind] < RMSE_upper, "Failed: The best model has RMSE higher than the upper bound."
                # if RMSE_val_list[ind] > RMSE_upper:
                #     warning("Warning: The best model has RMSE higher than {}.".format(RMSE_upper))
                
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.foward_degree = init_deg + ind
                print(f"Forward emulator generated with degree {init_deg+ind}, RMSE_val of {RMSE_val_list[ind]}.")

        if dim_reduction:
            print("Performing dimension reduction...")
            Mm, _ = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            if per_mode_thres is None:
                threshold = RMSE_tol * 1e-4
            else:
                threshold = min(per_mode_thres, RMSE_tol)
            mask = filter_modes(coeffs, Mm, threshold=threshold)
            multi_indices = multi_indices[mask]
            print(f"Dimension reduced  from {coeffs.shape[0]} modes to {multi_indices.shape[0]} modes.")
            Mm, nu = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(Mm, nu)

            # Y_val_pred = evaluate_emulator(X_val_scaled, coeffs, multi_indices)
            Y_val_pred = evaluate_emulator_batched(
                X_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )
            
            RMSE_val, AIC, BIC = predictive_mse_aic_bic(Y_val_scaled, Y_val_pred, multi_indices.shape[0], n_train=X_train_scaled.shape[0])
            print(f"After the dimension reduction, the RMSE: {RMSE_val}, AIC: {AIC}, BIC: {BIC}")



        self.forward_coeffs = coeffs
        self.forward_multi_indices = multi_indices
        self.forward_RMSE_list = RMSE_val_list
        self.forward_AIC_list = AIC_list
        self.forward_BIC_list = BIC_list
        self.forward_running_time_list = running_time_list
        self.forward_degree_list = degree_list
        
        pass

    def forward_emulator(self, X):
        # Check if the input is float, 1D or 2D
        float_or_int = isinstance(X, (float, int))
        if float_or_int:
            X = np.array([[X]])
        elif isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(f"Input must be a float, 1D list, or ND numpy array with last axis equal to the number of parameters ({self.n_params})")

        Xshape = X.shape
        assert Xshape[-1] == self.n_params, f"Input dimension (last axis) must be equal to the number of parameters ({self.n_params})"

        # if dim is not 2, reshape it
        if X.ndim != 2:
            X = X.reshape(-1, self.n_params)

        X_scaled = self.scaler_X.transform(X)
        Y_pred_scaled = evaluate_emulator(X_scaled, self.forward_coeffs, self.forward_multi_indices)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        if float_or_int:
            Y_pred = Y_pred[0]
            if self.n_outputs == 1:
                Y_pred = Y_pred[0]
        else:
            Y_pred = Y_pred.reshape(Xshape[:-1] + (self.n_outputs,))
        if self.log_Y:
            return np.exp(Y_pred)
        return Y_pred

    def generate_backward_emulator(self, 
                                   X_train_scaled, 
                                   Y_train_scaled,
                                   X_val_scaled,
                                   Y_val_scaled,
                                #    RMSE_upper=0.1,
                                   RMSE_tol=1e-2, 
                                   fRMSE_tol=1e-1, 
                                   init_deg=None, 
                                   max_degree=None,
                                   dim_reduction=False,
                                   per_mode_thres=None, 
                                   batch_size=10000):
        if init_deg is None:
            if self.n_outputs > 6:
                init_deg = 1
            elif self.n_outputs < 3:
                init_deg = 3
            else:
                init_deg = 2

        assert init_deg <= max_degree, "Initial degree must be less than or equal to max_degree"

        coeffs_list = []
        RMSE_val_list = []
        AIC_list = []
        BIC_list = []
        multi_indices_list = []

        for d in range(init_deg, max_degree + 1):
            if d == init_deg:
                multi_indices = generate_multi_indices(self.n_outputs, d)
            else:
                aux_indices = given_order_indices(self.n_outputs, d)
                multi_indices = np.concatenate((multi_indices, aux_indices), axis=0)
            # M, nu = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            # Use batched computation
            M, nu = compute_moments_vector_output_batched(
                Y_train_scaled, X_train_scaled, multi_indices, batch_size=batch_size
            )
            coeffs = solve_emulator_coefficients(M, nu)

            # X_val_pred = evaluate_emulator(Y_val_scaled, coeffs, multi_indices)
            X_val_pred = evaluate_emulator_batched(
                Y_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )

            RMSE_val, AIC, BIC = predictive_mse_aic_bic(X_val_scaled, X_val_pred, multi_indices.shape[0], n_train=Y_train_scaled.shape[0])
            RMSE_val_list.append(RMSE_val)
            AIC_list.append(AIC)
            BIC_list.append(BIC)

            # if RMSE_val < RMSE_upper: # if the RMSE is lower than the upper bound, we accept the model, and save it for later selection
            #     coeffs_list.append(coeffs)
            #     multi_indices_list.append(multi_indices)
            # else: # If the RMSE exceeds the upper bound, we reject the model straight away.
            #     coeffs_list.append(None) 
            #     multi_indices_list.append(None)

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)

            if RMSE_val < RMSE_tol:
                self.backward_degree = d
                print(f"Backward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                warning(f"Maximum degree {max_degree} reached. Now choose the best fit. ")
                ind = select_best_model(RMSE_val_list, aic_list=AIC_list, bic_list=BIC_list, rmse_tol=fRMSE_tol)
                # assert RMSE_val_list[ind] < RMSE_upper, "Failed: The best model has RMSE higher than the upper bound."

                # if RMSE_val_list[ind] > RMSE_upper:
                #     warning("Warning: The best model has RMSE higher than {}.".format(RMSE_upper))

                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.backward_degree = init_deg + ind
                print(f"Backward emulator generated with degree {init_deg+ind}, RMSE_val of {RMSE_val_list[ind]}.")

        if dim_reduction:
            print("Performing dimension reduction...")
            Mm, _ = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            if per_mode_thres is None:
                threshold = RMSE_tol * 1e-2
            else:
                threshold = min(per_mode_thres, RMSE_tol)
            mask = filter_modes(coeffs, Mm, threshold=threshold)
            multi_indices = multi_indices[mask]
            print(f"Dimension reduced  from {coeffs.shape[0]} modes to {multi_indices.shape[0]}  modes.")
            Mm, nu = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(Mm, nu)
            # X_val_pred = evaluate_emulator(Y_val_scaled, coeffs, multi_indices)
            X_val_pred = evaluate_emulator_batched(
                Y_val_scaled, coeffs, multi_indices, batch_size=batch_size
            )
            RMSE_val, AIC, BIC = predictive_mse_aic_bic(X_val_scaled, X_val_pred, multi_indices.shape[0], n_train=Y_train_scaled.shape[0])
            print(f"After the dimension reduction, the RMSE: {RMSE_val}, AIC: {AIC}, BIC: {BIC}")
            
        self.backward_coeffs = coeffs
        self.backward_multi_indices = multi_indices
        self.backward_RMSE_list = RMSE_val_list
        self.backward_AIC_list = AIC_list
        self.backward_BIC_list = BIC_list

        pass

    def backward_emulator(self, Y):
        float_or_int = isinstance(Y, (float, int))
        if float_or_int:
            Y = np.array([[Y]])
        elif isinstance(Y, list):
            Y = np.array(Y)
        elif isinstance(Y, np.ndarray):
            pass
        else:
            raise ValueError("Input must be a float, 1D list, or ND numpy array")

        if self.log_Y:
            Y = np.log(Y)

        Yshape = Y.shape
        assert Yshape[-1] == self.n_outputs, "Input must have the same number of outputs (i.e., the dimension of the last axis) as the emulator"

        if Y.ndim != 2:
            Y = Y.reshape(-1, self.n_outputs)

        Y_scaled = self.scaler_Y.transform(Y)
        X_pred_scaled = evaluate_emulator(Y_scaled, self.backward_coeffs, self.backward_multi_indices)
        X_pred = self.scaler_X.inverse_transform(X_pred_scaled)
        if float_or_int:
            X_pred = X_pred[0]
            if self.n_params == 1:
                X_pred = X_pred[0]
        else:
            X_pred = X_pred.reshape(Yshape[:-1] + (self.n_params,))
        return X_pred

    def generate_forward_symb_emu(self, variable_names=None):
        Y_var = np.ones(self.n_outputs) if not self.standardize_Y_with_std else self.scaler_Y.var_
        exprs = symbolic_polynomial_expressions(self.forward_coeffs, 
                                                self.forward_multi_indices, 
                                                variable_names=variable_names, 
                                                input_means=self.scaler_X.mean_, 
                                                input_vars=self.scaler_X.var_,
                                                output_means=self.scaler_Y.mean_, 
                                                output_vars=Y_var)
        return exprs
    
    def generate_backward_symb_emu(self, variable_names=None):
        Y_var = np.ones(self.n_outputs) if not self.standardize_Y_with_std else self.scaler_Y.var_
        exprs = symbolic_polynomial_expressions(self.backward_coeffs, 
                                                self.backward_multi_indices, 
                                                variable_names=variable_names, 
                                                input_means=self.scaler_Y.mean_, 
                                                input_vars=Y_var,
                                                output_means=self.scaler_X.mean_,
                                                output_vars=self.scaler_X.var_)
        return exprs
