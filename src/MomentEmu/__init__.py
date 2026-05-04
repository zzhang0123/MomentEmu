"""
MomentEmu: A lightweight, interpretable polynomial emulator for smooth mappings.

This package implements the moment-projection polynomial emulator introduced in 
Zhang (2025), building interpretable, closed-form polynomial emulators via moment 
matrices with millisecond-level inference and symbolic transparency.

Auto-differentiation support is available through:
- jax_momentemu: JAX integration for high-performance computing
- torch_momentemu: PyTorch integration for machine learning
- symbolic_momentemu: SymPy integration for exact symbolic computation
"""

from MomentEmu.PolyEmu import PolyEmu, evaluate_emulator, symbolic_polynomial_expressions
from MomentEmu.MomentEmu import signal_aware_frac_err

# Auto-differentiation modules (optional imports)
__all_autodiff__ = []

try:
    from . import jax_momentemu
    __all_autodiff__.append("jax_momentemu")
except ImportError:
    pass
    
try:
    from . import torch_momentemu
    __all_autodiff__.append("torch_momentemu")
except ImportError:
    pass
    
try:
    from . import symbolic_momentemu
    __all_autodiff__.append("symbolic_momentemu")
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "Zheng Zhang"
__all__ = [
    "PolyEmu",
    "evaluate_emulator",
    "symbolic_polynomial_expressions",
    "signal_aware_frac_err",
] + __all_autodiff__