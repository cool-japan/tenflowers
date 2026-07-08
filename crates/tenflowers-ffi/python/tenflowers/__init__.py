"""
TenfloweRS - Pure Rust Machine Learning Framework

A high-performance machine learning framework built entirely in Rust,
with comprehensive Python bindings for ease of use.
"""

from .tenflowers import *  # Import all Rust-defined functions and classes
from .tenflowers import __version__  # explicit import bypasses import*'s leading-underscore exclusion

# Dtype constants (float32, int64, bool, ...) are set on the compiled module via
# plain `m.setattr(...)` rather than `add_function`/`add_class`, so PyO3's
# auto-generated `__all__` for the compiled submodule never lists them and
# `import *` above silently skips them. Re-import explicitly, as for __version__.
from .tenflowers import (
    float32,
    float64,
    float16,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bool,
)

__all__ = [
    # Version
    "__version__",

    # Dtype constants
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",

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
