"""
TenfloweRS - Pure Rust Machine Learning Framework

A high-performance machine learning framework built entirely in Rust,
with comprehensive Python bindings for ease of use.
"""

from .tenflowers import *  # Import all Rust-defined functions and classes

__version__ = "0.1.0a2"
__all__ = [
    # Version
    "__version__",

    # Core tensor operations
    "PyTensor",
    "zeros",
    "ones",
    "rand",
    "randn",
    "add",
    "mul",
    "sub",
    "div",
    "matmul",
    "transpose",
    "reshape",

    # Mathematical operations
    "exp",
    "log",
    "sqrt",
    "abs",
    "neg",
    "sin",
    "cos",
    "tan",

    # Reduction operations
    "sum",
    "mean",
    "max",
    "min",
    "var",
    "std",
    "standard_deviation",

    # Comparison operations
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",

    # Utility operations
    "clamp",
    "argmax",
    "argmin",

    # Tensor manipulation
    "cat",
    "stack",
    "split",
    "squeeze",
    "unsqueeze",
    "flatten",

    # Device management
    "get_default_device",
    "set_default_device",

    # Memory profiling
    "enable_memory_profiling",
    "disable_memory_profiling",
    "get_memory_info",

    # Gradient management
    "is_grad_enabled",
    "set_grad_enabled",

    # NumPy interop
    "tensor_from_numpy",
    "tensor_to_numpy",
]
