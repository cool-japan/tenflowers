/*!
 * LAPACK operations specifically for f64
 */

use crate::{Result, Tensor, TensorError};
#[cfg(feature = "blas")]
use ndarray_linalg::{Cholesky, Determinant, Eig, Inverse, Solve, QR, SVD, UPLO};
#[cfg(feature = "blas")]
use scirs2_core::ndarray::Array2;

/// Matrix inverse for f64
pub fn inverse_f64(input: &Tensor<f64>) -> Result<Tensor<f64>> {
    // TODO: Fix LAPACK trait implementation for f64
    Err(TensorError::not_implemented_simple(
        "LAPACK inverse for f64 not yet implemented".to_string(),
    ))
}

/// Matrix determinant for f64
pub fn determinant_f64(input: &Tensor<f64>) -> Result<f64> {
    // TODO: Fix LAPACK trait implementation for f64
    Err(TensorError::not_implemented_simple(
        "LAPACK determinant for f64 not yet implemented".to_string(),
    ))
}

/// SVD for f64
pub fn svd_f64(input: &Tensor<f64>) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>)> {
    // TODO: Fix LAPACK trait implementation for f64
    Err(TensorError::not_implemented_simple(
        "LAPACK SVD for f64 not yet implemented".to_string(),
    ))
}

/// Solve linear system for f64
pub fn solve_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>> {
    // TODO: Fix LAPACK trait implementation for f64
    Err(TensorError::not_implemented_simple(
        "LAPACK solve for f64 not yet implemented".to_string(),
    ))
}
