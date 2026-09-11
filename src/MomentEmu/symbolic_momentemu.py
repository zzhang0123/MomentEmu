"""
Symbolic auto-differentiation MomentEmu implementation using SymPy

This module provides SymPy integration for MomentEmu, enabling:
- Exact symbolic differentiation with zero numerical error
- Arbitrary-order derivatives (gradients, Hessians, etc.)
- Mathematical expression analysis and manipulation
- Educational insights into polynomial structure

Key components:
- SymbolicMomentEmu: SymPy wrapper for symbolic computation
- create_symbolic_emulator(): Convert trained MomentEmu to symbolic format
- demo_symbolic_autodiff(): Demonstration of symbolic auto-differentiation capabilities
"""

import sympy as sp
import numpy as np
from MomentEmu.PolyEmu import PolyEmu

class SymbolicMomentEmu:
    """Symbolic differentiable MomentEmu using SymPy."""
    
    def __init__(self, trained_emulator, variable_names=None):
        self.emulator = trained_emulator
        
        # Get symbolic expressions
        n_inputs = trained_emulator.n_params
        if variable_names is None:
            variable_names = [f'x{i}' for i in range(n_inputs)]
        
        self.variables = sp.symbols(variable_names)
        self.expressions = trained_emulator.generate_forward_symb_emu(variable_names)
        
        # Create lambdified functions for fast numerical evaluation
        self.lambdified = [sp.lambdify(self.variables, expr, 'numpy') 
                          for expr in self.expressions]
        
        # Create gradient functions
        self.gradient_exprs = []
        self.gradient_funcs = []
        
        for expr in self.expressions:
            grad_expr = [sp.diff(expr, var) for var in self.variables]
            grad_func = [sp.lambdify(self.variables, g_expr, 'numpy') 
                        for g_expr in grad_expr]
            self.gradient_exprs.append(grad_expr)
            self.gradient_funcs.append(grad_func)
        
        # Create Hessian functions
        self.hessian_exprs = []
        self.hessian_funcs = []
        
        for expr in self.expressions:
            hess_expr = [[sp.diff(expr, var1, var2) for var2 in self.variables] 
                        for var1 in self.variables]
            hess_func = [[sp.lambdify(self.variables, h_expr, 'numpy') 
                         for h_expr in h_row] for h_row in hess_expr]
            self.hessian_exprs.append(hess_expr)
            self.hessian_funcs.append(hess_func)
    
    def predict(self, X):
        """Evaluate the emulator at X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for i, func in enumerate(self.lambdified):
            pred = np.array([func(*x) for x in X])
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def gradient(self, X):
        """Compute gradient at X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        gradients = []
        for x in X:
            grad_at_x = []
            for output_idx, grad_funcs in enumerate(self.gradient_funcs):
                grad_output = np.array([func(*x) for func in grad_funcs])
                grad_at_x.append(grad_output)
            gradients.append(np.array(grad_at_x))
        
        return np.array(gradients)
    
    def hessian(self, X):
        """Compute Hessian at X."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        hessians = []
        for x in X:
            hess_at_x = []
            for output_idx, hess_funcs in enumerate(self.hessian_funcs):
                hess_matrix = np.array([[func(*x) for func in row] for row in hess_funcs])
                hess_at_x.append(hess_matrix)
            hessians.append(np.array(hess_at_x))
        
        return np.array(hessians)
    
    def get_symbolic_expressions(self):
        """Get the symbolic expressions."""
        return {
            'expressions': self.expressions,
            'gradients': self.gradient_exprs,
            'hessians': self.hessian_exprs
        }

def create_symbolic_emulator(trained_emulator, variable_names=None):
    """Create a symbolic emulator from a trained MomentEmu."""
    symbolic_emu = SymbolicMomentEmu(trained_emulator, variable_names)
    
    # Return a dict with all the important components for easy access
    return {
        "emulator": symbolic_emu,
        "expression": symbolic_emu.expressions[0] if symbolic_emu.expressions else None,
        "variables": symbolic_emu.variables,
        "lambdified": symbolic_emu.lambdified[0] if symbolic_emu.lambdified else None,
        "gradient_func": lambda *args: symbolic_emu.gradient(np.array([args]))[0],
        "hessian_func": lambda *args: symbolic_emu.hessian(np.array([args]))[0]
    }

def demo_symbolic_autodiff():
    """Demonstrate symbolic auto-differentiation."""
    
    # Train regular MomentEmu
    print("Training MomentEmu...")
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (50, 2))
    Y_train = (X_train[:, 0]**2 + 2*X_train[:, 0]*X_train[:, 1] + X_train[:, 1]**2).reshape(-1, 1)
    
    emulator = PolyEmu(X_train, Y_train, forward=True, backward=False)
    
    # Create symbolic version
    print("Creating symbolic version...")
    sym_emu = SymbolicMomentEmu(emulator, ['x', 'y'])
    
    # Test point
    x_test = np.array([[0.5, 0.3]])
    
    # Predictions
    y_pred = sym_emu.predict(x_test)
    print(f"Prediction: {y_pred[0]}")
    
    # Exact gradients
    grad = sym_emu.gradient(x_test)
    print(f"Gradient: {grad[0]}")
    
    # Exact Hessians
    hess = sym_emu.hessian(x_test)
    print(f"Hessian: {hess[0]}")
    
    # Show symbolic expressions
    expressions = sym_emu.get_symbolic_expressions()
    print(f"\\nSymbolic expression: {expressions['expressions'][0]}")
    print(f"Symbolic gradient: {expressions['gradients'][0]}")

if __name__ == "__main__":
    demo_symbolic_autodiff()
