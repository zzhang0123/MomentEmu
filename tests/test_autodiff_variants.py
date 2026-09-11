import numpy as np
import pytest

from MomentEmu.PolyEmu import PolyEmu
from MomentEmu import torch_momentemu, jax_momentemu, symbolic_momentemu

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Set random seed for reproducible results
np.random.seed(42)

# Create training data for f(x,y) = x^2 + y^2
# This gives us known analytical derivatives for validation
X_train = np.random.uniform(-1, 1, (50, 2))
Y_train = (X_train[:, 0]**2 + X_train[:, 1]**2).reshape(-1, 1)
emulator = PolyEmu(X_train, Y_train, forward=True, backward=False)

# Test point
test_x, test_y = 0.5, 0.3

# Analytical results for f(x,y) = x^2 + y^2 at (0.5, 0.3)
def analytical_function(x, y):
    return x**2 + y**2

def analytical_gradient(x, y):
    return np.array([2*x, 2*y])

def analytical_hessian(x, y):
    return np.array([[2.0, 0.0], [0.0, 2.0]])

# Expected values at test point
expected_value = analytical_function(test_x, test_y)  # 0.5^2 + 0.3^2 = 0.34
expected_grad = analytical_gradient(test_x, test_y)   # [1.0, 0.6]
expected_hess = analytical_hessian(test_x, test_y)    # [[2, 0], [0, 2]]

# Tolerance for numerical comparisons
TOLERANCE = 1e-3  # Allow some error due to polynomial approximation

# Torch test
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_torch_autodiff():
    torch_emu = torch_momentemu.create_torch_emulator(emulator)
    import torch
    
    # Test basic functionality
    x = torch.tensor([test_x, test_y], requires_grad=True, dtype=torch.float32)
    y = torch_emu(x)
    y.backward()
    grad = x.grad
    
    # Shape checks
    assert grad is not None
    assert grad.shape == x.shape
    
    # Consistency checks against analytical results
    predicted_value = float(y.detach().numpy().item())
    predicted_grad = grad.detach().numpy()
    
    print(f"PyTorch - Expected value: {expected_value:.4f}, Got: {predicted_value:.4f}")
    print(f"PyTorch - Expected grad: {expected_grad}, Got: {predicted_grad}")
    
    # Value check
    assert abs(predicted_value - expected_value) < TOLERANCE, \
        f"Function value mismatch: expected {expected_value:.4f}, got {predicted_value:.4f}"
    
    # Gradient check
    grad_error = np.linalg.norm(predicted_grad - expected_grad)
    assert grad_error < TOLERANCE, \
        f"Gradient mismatch: expected {expected_grad}, got {predicted_grad}, error: {grad_error:.6f}"

# JAX test
@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_autodiff():
    jax_emu = jax_momentemu.create_jax_emulator(emulator)
    import jax.numpy as jnp
    from jax import grad, hessian
    
    # Test basic functionality
    x = jnp.array([test_x, test_y])
    y = jax_emu(x)
    grad_fn = grad(lambda x: jax_emu(x).sum())
    gradient = grad_fn(x)
    
    # Compute Hessian
    hess_fn = hessian(lambda x: jax_emu(x).sum())
    hess = hess_fn(x)
    
    # Shape checks
    assert gradient.shape == x.shape
    assert hess.shape == (2, 2)
    
    # Consistency checks against analytical results
    predicted_value = float(y.item()) if hasattr(y, 'item') else float(np.asarray(y).item())
    predicted_grad = np.array(gradient)
    predicted_hess = np.array(hess)
    
    print(f"JAX - Expected value: {expected_value:.4f}, Got: {predicted_value:.4f}")
    print(f"JAX - Expected grad: {expected_grad}, Got: {predicted_grad}")
    print(f"JAX - Expected hess:\n{expected_hess}\nGot:\n{predicted_hess}")
    
    # Value check
    assert abs(predicted_value - expected_value) < TOLERANCE, \
        f"Function value mismatch: expected {expected_value:.4f}, got {predicted_value:.4f}"
    
    # Gradient check
    grad_error = np.linalg.norm(predicted_grad - expected_grad)
    assert grad_error < TOLERANCE, \
        f"Gradient mismatch: expected {expected_grad}, got {predicted_grad}, error: {grad_error:.6f}"
    
    # Hessian check
    hess_error = np.linalg.norm(predicted_hess - expected_hess)
    assert hess_error < TOLERANCE, \
        f"Hessian mismatch: expected\n{expected_hess}\ngot\n{predicted_hess}\nerror: {hess_error:.6f}"

# Symbolic test
def test_symbolic_autodiff():
    sym_dict = symbolic_momentemu.create_symbolic_emulator(emulator, ['x', 'y'])
    x_vals = [test_x, test_y]
    
    # Get predictions
    y = sym_dict["lambdified"](*x_vals)
    gradient = sym_dict["gradient_func"](*x_vals)
    hessian = sym_dict["hessian_func"](*x_vals)
    
    # Basic shape checks
    assert np.isscalar(y) or (hasattr(y, 'shape') and y.shape == (1,))
    
    # Handle gradient shape - it could be (2,) or (1,2)
    gradient = np.asarray(gradient)
    if gradient.ndim == 1:
        assert len(gradient) == 2
        predicted_grad = gradient
    else:
        assert gradient.shape[-1] == 2  # Last dimension should be 2
        predicted_grad = gradient.flatten() if gradient.ndim > 1 else gradient
    
    # Handle hessian shape - it could be (2,2) or (1,2,2)
    hessian = np.asarray(hessian)
    if hessian.ndim == 2:
        assert hessian.shape == (2, 2)
        predicted_hess = hessian
    else:
        assert hessian.shape[-2:] == (2, 2)  # Last two dimensions should be (2,2)
        predicted_hess = hessian.reshape(-1, 2, 2)[-1]  # Take the last (2,2) slice
    
    # Consistency checks against analytical results
    predicted_value = float(y) if np.isscalar(y) else float(y[0])
    
    print(f"Symbolic - Expected value: {expected_value:.4f}, Got: {predicted_value:.4f}")
    print(f"Symbolic - Expected grad: {expected_grad}, Got: {predicted_grad}")
    print(f"Symbolic - Expected hess:\n{expected_hess}\nGot:\n{predicted_hess}")
    
    # Value check
    assert abs(predicted_value - expected_value) < TOLERANCE, \
        f"Function value mismatch: expected {expected_value:.4f}, got {predicted_value:.4f}"
    
    # Gradient check
    grad_error = np.linalg.norm(predicted_grad - expected_grad)
    assert grad_error < TOLERANCE, \
        f"Gradient mismatch: expected {expected_grad}, got {predicted_grad}, error: {grad_error:.6f}"
    
    # Hessian check
    hess_error = np.linalg.norm(predicted_hess - expected_hess)
    assert hess_error < TOLERANCE, \
        f"Hessian mismatch: expected\n{expected_hess}\ngot\n{predicted_hess}\nerror: {hess_error:.6f}"


# Cross-validation test: Compare all three autodiff methods
def test_autodiff_consistency():
    """Test that all three autodiff methods give consistent results."""
    # Only run if all frameworks are available
    if not (HAS_TORCH and HAS_JAX):
        pytest.skip("Requires both PyTorch and JAX for cross-validation")
    
    # Test point
    x_vals = [test_x, test_y]
    
    # Get results from all three methods
    # PyTorch
    torch_emu = torch_momentemu.create_torch_emulator(emulator)
    x_torch = torch.tensor(x_vals, requires_grad=True, dtype=torch.float32)
    y_torch = torch_emu(x_torch)
    y_torch.backward()
    torch_value = float(y_torch.detach().numpy().item())
    torch_grad = x_torch.grad.detach().numpy()
    
    # JAX  
    jax_emu = jax_momentemu.create_jax_emulator(emulator)
    x_jax = jnp.array(x_vals)
    y_jax = jax_emu(x_jax)
    jax_value = float(y_jax.item()) if hasattr(y_jax, 'item') else float(np.asarray(y_jax).item())
    jax_grad = np.array(jax.grad(lambda x: jax_emu(x).sum())(x_jax))
    
    # Symbolic
    sym_dict = symbolic_momentemu.create_symbolic_emulator(emulator, ['x', 'y'])
    sym_value = float(sym_dict["lambdified"](*x_vals))
    sym_grad = np.asarray(sym_dict["gradient_func"](*x_vals))
    if sym_grad.ndim > 1:
        sym_grad = sym_grad.flatten()
    
    # Cross-validation: all methods should agree
    value_tolerance = 1e-5
    grad_tolerance = 1e-5
    
    print(f"Cross-validation values: PyTorch={torch_value:.6f}, JAX={jax_value:.6f}, Symbolic={sym_value:.6f}")
    print(f"Cross-validation grads: PyTorch={torch_grad}, JAX={jax_grad}, Symbolic={sym_grad}")
    
    # Value consistency
    assert abs(torch_value - jax_value) < value_tolerance, \
        f"PyTorch-JAX value mismatch: {torch_value:.6f} vs {jax_value:.6f}"
    assert abs(torch_value - sym_value) < value_tolerance, \
        f"PyTorch-Symbolic value mismatch: {torch_value:.6f} vs {sym_value:.6f}"
    assert abs(jax_value - sym_value) < value_tolerance, \
        f"JAX-Symbolic value mismatch: {jax_value:.6f} vs {sym_value:.6f}"
    
    # Gradient consistency  
    torch_jax_grad_error = np.linalg.norm(torch_grad - jax_grad)
    torch_sym_grad_error = np.linalg.norm(torch_grad - sym_grad)
    jax_sym_grad_error = np.linalg.norm(jax_grad - sym_grad)
    
    assert torch_jax_grad_error < grad_tolerance, \
        f"PyTorch-JAX gradient mismatch: error={torch_jax_grad_error:.8f}"
    assert torch_sym_grad_error < grad_tolerance, \
        f"PyTorch-Symbolic gradient mismatch: error={torch_sym_grad_error:.8f}"  
    assert jax_sym_grad_error < grad_tolerance, \
        f"JAX-Symbolic gradient mismatch: error={jax_sym_grad_error:.8f}"


def test_emulator_accuracy():
    """Test that the emulator accurately approximates the original function."""
    # Test on multiple points
    test_points = np.array([
        [0.0, 0.0],   # Origin
        [0.5, 0.3],   # Test point
        [1.0, 0.0],   # Edge case
        [0.0, 1.0],   # Edge case  
        [-0.5, 0.8],  # Negative values
        [0.7, -0.4],  # Mixed signs
    ])
    
    # Get emulator predictions
    emulator_predictions = emulator.forward_emulator(test_points)
    
    # Calculate true values
    true_values = (test_points[:, 0]**2 + test_points[:, 1]**2).reshape(-1, 1)
    
    # Check accuracy
    max_error = np.max(np.abs(emulator_predictions - true_values))
    rmse = np.sqrt(np.mean((emulator_predictions - true_values)**2))
    
    print(f"Emulator accuracy test:")
    print(f"Max error: {max_error:.2e}")
    print(f"RMSE: {rmse:.2e}")
    print(f"Test points shape: {test_points.shape}")
    print(f"Predictions shape: {emulator_predictions.shape}")
    
    # The emulator should be very accurate for this simple quadratic function
    assert max_error < 1e-10, f"Emulator max error {max_error:.2e} exceeds threshold"
    assert rmse < 1e-10, f"Emulator RMSE {rmse:.2e} exceeds threshold"
