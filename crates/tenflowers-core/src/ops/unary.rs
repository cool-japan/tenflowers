//! Unary operations module
//!
//! This module provides unary tensor operations, re-exporting functions from other modules
//! to provide a clean unary operations API.

use crate::{Result, Tensor};

/// Compute element-wise negation (negative) of tensor
pub fn neg<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + std::ops::Neg<Output = T>,
{
    // Re-export from numpy_compat::negative
    crate::ops::numpy_compat::negative(tensor)
}

/// Compute natural logarithm of tensor elements
pub fn log<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + num_traits::Float,
{
    // Re-export from numpy_compat::log
    crate::ops::numpy_compat::log(tensor)
}

/// Compute absolute value of tensor elements
pub fn abs<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + num_traits::Signed,
{
    // Re-export from numpy_compat::absolute
    crate::ops::numpy_compat::absolute(tensor)
}

// Re-export for convenience
pub use crate::ops::numpy_compat::{
    absolute, ceil, cos, cosh, exp, exp2, expm1, fix, floor, log10, log1p, log2, negative,
    reciprocal, rint, sign, signbit, sin, sinh, sqrt, square, tan, tanh, trunc,
};
