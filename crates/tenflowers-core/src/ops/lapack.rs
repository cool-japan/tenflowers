/*!
 * LAPACK integration for enhanced linear algebra performance
 *
 * This module provides optimized linear algebra operations using LAPACK
 * when the appropriate feature flags are enabled.
 */

#[cfg(feature = "blas")]
use ndarray_linalg::{Cholesky, Determinant, Eig, Inverse, LeastSquaresSvd, Solve, QR, SVD, UPLO};
#[cfg(feature = "blas")]
use scirs2_autograd::ndarray::Array2;

use crate::{Result, Tensor, TensorError};
use num_traits::{Float, One, Zero};

/// Enhanced matrix multiplication using BLAS when available
pub fn matmul_blas<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static + bytemuck::Pod,
{
    #[cfg(feature = "blas")]
    {
        // Dot trait not needed here

        // Convert tensors to ndarray format
        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Matrix multiplication requires contiguous tensor data".to_string(),
            )
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Matrix multiplication requires contiguous tensor data".to_string(),
            )
        })?;

        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let a_array =
            Array2::from_shape_vec((a_shape[0], a_shape[1]), a_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;
        let b_array =
            Array2::from_shape_vec((b_shape[0], b_shape[1]), b_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;

        // Perform BLAS-accelerated matrix multiplication
        let result = a_array.dot(&b_array);

        // Convert back to tensor
        Ok(Tensor::from_array(result.into_dyn()))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular matmul implementation
        crate::ops::matmul(a, b)
    }
}

/// Enhanced LU decomposition using LAPACK when available
pub fn lu_decompose_lapack<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    #[cfg(feature = "blas")]
    {
        // TODO: Fix LAPACK operations to use proper ndarray_linalg API
        // use ndarray_linalg::*;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "LU decomposition requires contiguous tensor data".to_string(),
            )
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "LU decomposition requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // TODO: Fix LAPACK LU decomposition API usage
        // The ndarray_linalg API has changed and needs proper trait imports
        return Err(TensorError::BlasError {
            operation: "lu".to_string(),
            details: "LU decomposition not yet implemented with current ndarray_linalg API"
                .to_string(),
            context: None,
        });
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular LU decomposition
        let (l, u, p) = crate::ops::lu(input)?;
        Ok((l, u, p))
    }
}

/// Enhanced matrix inverse using LAPACK when available
pub fn inverse_lapack<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    // Dispatch to specific implementations based on type
    #[cfg(feature = "blas")]
    {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // Safety: We've checked the type
            let input_f32 = unsafe { &*(input as *const Tensor<T> as *const Tensor<f32>) };
            let result = super::lapack_f32::inverse_f32(input_f32)?;
            Ok(unsafe { std::mem::transmute_copy(&result) })
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            // Safety: We've checked the type
            let input_f64 = unsafe { &*(input as *const Tensor<T> as *const Tensor<f64>) };
            let result = super::lapack_f64::inverse_f64(input_f64)?;
            Ok(unsafe { std::mem::transmute_copy(&result) })
        } else {
            return Err(TensorError::BlasError {
                operation: "inv".to_string(),
                details: "LAPACK operations only supported for f32 and f64".to_string(),
                context: None,
            });
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular inverse implementation
        crate::ops::inv(input)
    }
}

/// Enhanced determinant computation using LAPACK when available
pub fn determinant_lapack<T>(input: &Tensor<T>) -> Result<T>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::Determinant;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Determinant requires contiguous tensor data".to_string(),
            )
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Determinant requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // TODO: Fix LAPACK trait bounds - det requires Scalar + Lapack traits
        // let result = matrix.det().map_err(|e| TensorError::BlasError {
        //     operation: "det".to_string(),
        //     details: format!("Determinant computation failed: {}", e),
        //     context: None,
        // })?;
        // Ok(result)

        Err(TensorError::not_implemented_simple(
            "Determinant requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular determinant implementation
        crate::ops::det(input).and_then(|det_tensor| {
            det_tensor.as_slice().map(|s| s[0]).ok_or_else(|| {
                TensorError::invalid_shape_simple(
                    "Determinant tensor is not contiguous".to_string(),
                )
            })
        })
    }
}

/// Enhanced eigenvalue decomposition using LAPACK when available
pub fn eigenvalues_lapack<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::Eig;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Eigenvalue decomposition requires contiguous tensor data".to_string(),
            )
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Eigenvalue decomposition requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // Perform LAPACK-accelerated eigenvalue decomposition
        // TODO: Fix LAPACK trait bounds - eig requires Scalar + Lapack traits
        // let (eigenvals, eigenvecs) = matrix.eig().map_err(|e| TensorError::BlasError {
        //     operation: "eig".to_string(),
        //     details: format!("Eigenvalue decomposition failed: {}", e),
        //     context: None,
        // })?;
        Err(TensorError::not_implemented_simple(
            "Eigenvalues requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular eigenvalue decomposition
        crate::ops::linalg::eig(input)
    }
}

/// Enhanced SVD decomposition using LAPACK when available
pub fn svd_lapack<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::SVD;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple("SVD requires contiguous tensor data".to_string())
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "SVD requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // Perform LAPACK-accelerated SVD
        // TODO: Fix LAPACK trait bounds - svd requires Scalar + Lapack traits
        // let (u, s, vt) = matrix.svd(true, true).map_err(|e| TensorError::BlasError {
        //     operation: "svd".to_string(),
        //     details: format!("SVD failed: {}", e),
        //     context: None,
        // })?;
        Err(TensorError::not_implemented_simple(
            "SVD requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular SVD implementation
        crate::ops::linalg::svd(input)
    }
}

/// Enhanced Cholesky decomposition using LAPACK when available
pub fn cholesky_lapack<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::Cholesky;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Cholesky decomposition requires contiguous tensor data".to_string(),
            )
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Cholesky decomposition requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // Perform LAPACK-accelerated Cholesky decomposition
        // TODO: Fix LAPACK trait bounds - cholesky requires Scalar + Lapack traits
        // let result = matrix.cholesky(UPLO::Lower).map_err(|e| TensorError::BlasError {
        //     operation: "cholesky".to_string(),
        //     details: format!("Cholesky decomposition failed: {}", e),
        //     context: None,
        // })?;
        // Ok(Tensor::from_array(result.into_dyn()))

        Err(TensorError::not_implemented_simple(
            "Cholesky requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular Cholesky implementation
        crate::ops::linalg::cholesky(input)
    }
}

/// Enhanced QR decomposition using LAPACK when available
pub fn qr_lapack<T>(input: &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::QR;

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "QR decomposition requires contiguous tensor data".to_string(),
            )
        })?;

        let shape = input.shape().dims();
        if shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "QR decomposition requires 2D tensor".to_string(),
            ));
        }

        let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|_| {
            TensorError::invalid_shape_simple(
                "Failed to create Array2 from tensor data".to_string(),
            )
        })?;

        // Perform LAPACK-accelerated QR decomposition
        let mut matrix_copy = matrix.clone();
        // TODO: Fix LAPACK trait bounds - qr requires Scalar + Lapack traits
        // let (q, r) = matrix_copy.qr().map_err(|e| TensorError::BlasError {
        //     operation: "qr".to_string(),
        //     details: format!("QR decomposition failed: {}", e),
        //     context: None,
        // })?;
        Err(TensorError::not_implemented_simple(
            "QR decomposition requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular QR implementation
        // Use the QR decomposition helper function from linalg module
        let n = input.shape().dims()[0];
        let m = input.shape().dims()[1];
        if n != m {
            return Err(TensorError::invalid_shape_simple(
                "QR decomposition requires square matrix for fallback".to_string(),
            ));
        }

        let input_data = input.as_slice().unwrap();
        let (q, r) = crate::ops::linalg::qr_decomposition(input_data, n)?;

        Ok((Tensor::from_vec(q, &[n, n])?, Tensor::from_vec(r, &[n, n])?))
    }
}

/// Enhanced solve linear system using LAPACK when available
pub fn solve_lapack<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static + bytemuck::Pod,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::Solve;

        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Linear system solve requires contiguous tensor data".to_string(),
            )
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Linear system solve requires contiguous tensor data".to_string(),
            )
        })?;

        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Linear system solve requires 2D matrices".to_string(),
            ));
        }

        let a_matrix =
            Array2::from_shape_vec((a_shape[0], a_shape[1]), a_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;

        let b_matrix =
            Array2::from_shape_vec((b_shape[0], b_shape[1]), b_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;

        // Perform LAPACK-accelerated linear system solve
        let mut a_matrix_copy = a_matrix.clone();
        let mut b_matrix_copy = b_matrix.clone();
        // TODO: Fix LAPACK trait bounds - solve requires Scalar + Lapack traits
        // let result = a_matrix_copy.solve(&b_matrix_copy).map_err(|e| TensorError::BlasError {
        //     operation: "solve".to_string(),
        //     details: format!("Linear system solve failed: {}", e),
        //     context: None,
        // })?;
        // Ok(Tensor::from_array(result.into_dyn()))

        Err(TensorError::not_implemented_simple(
            "Solve requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular solve implementation
        // For now, use LU decomposition approach
        let inv_a = crate::ops::inv(a)?;
        crate::ops::matmul(&inv_a, b)
    }
}

/// Enhanced least squares solve using LAPACK when available
pub fn lstsq_lapack<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    #[cfg(feature = "blas")]
    {
        use ndarray_linalg::LeastSquaresSvd;

        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Least squares solve requires contiguous tensor data".to_string(),
            )
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Least squares solve requires contiguous tensor data".to_string(),
            )
        })?;

        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Least squares solve requires 2D matrices".to_string(),
            ));
        }

        let a_matrix =
            Array2::from_shape_vec((a_shape[0], a_shape[1]), a_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;

        let b_matrix =
            Array2::from_shape_vec((b_shape[0], b_shape[1]), b_data.to_vec()).map_err(|_| {
                TensorError::invalid_shape_simple(
                    "Failed to create Array2 from tensor data".to_string(),
                )
            })?;

        // Perform LAPACK-accelerated least squares solve
        let mut a_matrix_copy = a_matrix.clone();
        let mut b_matrix_copy = b_matrix.clone();
        // TODO: Fix LAPACK trait bounds - least_squares requires Scalar + Lapack traits
        // let result = a_matrix_copy.least_squares(&b_matrix_copy).map_err(|e| TensorError::BlasError {
        //     operation: "least_squares".to_string(),
        //     details: format!("Least squares solve failed: {}", e),
        //     context: None,
        // })?;
        // Ok(Tensor::from_array(result.solution.into_dyn()))

        Err(TensorError::not_implemented_simple(
            "Least squares requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback to regular least squares using SVD
        let (u, s, vt) = crate::ops::linalg::svd(a)?;

        // Compute pseudoinverse using SVD
        // x = V * S^(-1) * U^T * b
        let ut_b = crate::ops::matmul(&crate::ops::transpose(&u)?, b)?;

        // Apply reciprocal of singular values (with threshold for numerical stability)
        let s_data = s.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Singular values tensor is not contiguous".to_string(),
            )
        })?;
        let s_inv_data: Vec<T> = s_data
            .iter()
            .map(|val| {
                if val.abs() > T::from(1e-10).unwrap_or(T::zero()) {
                    T::one() / *val
                } else {
                    T::zero()
                }
            })
            .collect();
        let s_inv = Tensor::from_vec(s_inv_data, s.shape().dims())?;

        let s_inv_ut_b = crate::ops::mul(&s_inv, &ut_b)?;
        let result = crate::ops::matmul(&crate::ops::transpose(&vt)?, &s_inv_ut_b)?;

        Ok(result)
    }
}

/// Compute pseudoinverse (Moore-Penrose inverse) using SVD
///
/// The pseudoinverse A^+ is computed as A^+ = V * S^+ * U^T where:
/// - A = U * S * V^T is the SVD decomposition
/// - S^+ is the pseudoinverse of the diagonal matrix S (reciprocal of non-zero elements)
pub fn pinv<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + One + Float + Send + Sync + 'static + bytemuck::Pod,
{
    #[cfg(feature = "blas")]
    {
        // TODO: Fix LAPACK operations to use proper ndarray_linalg API
        // use ndarray_linalg::*;

        let input_data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Pseudoinverse requires contiguous tensor data".to_string(),
            )
        })?;

        let input_shape = input.shape().dims();

        if input_shape.len() != 2 {
            return Err(TensorError::invalid_shape_simple(
                "Pseudoinverse requires 2D matrix".to_string(),
            ));
        }

        let input_matrix =
            Array2::from_shape_vec((input_shape[0], input_shape[1]), input_data.to_vec()).map_err(
                |_| {
                    TensorError::invalid_shape_simple(
                        "Failed to create Array2 from tensor data".to_string(),
                    )
                },
            )?;

        // Compute pseudoinverse using LAPACK
        let mut input_matrix_copy = input_matrix.clone();
        // TODO: Fix LAPACK trait bounds - pinv requires specific implementation
        // let pinv_matrix = input_matrix_copy.pinv(Some(T::from(1e-10).unwrap_or(T::zero())))
        //     .map_err(|e| TensorError::BlasError {
        //         operation: "pinv".to_string(),
        //         details: format!("Pseudoinverse computation failed: {}", e),
        //         context: None,
        //     })?;
        // Ok(Tensor::from_array(pinv_matrix.into_dyn()))

        Err(TensorError::not_implemented_simple(
            "Pseudo-inverse requires specific type (f32/f64) implementation".to_string(),
        ))
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fallback implementation using SVD
        let (u, s_full, v) = crate::ops::linalg::svd(input)?;

        let input_shape = input.shape().dims();
        let m = input_shape[0];
        let n = input_shape[1];
        let min_dim = m.min(n);

        // Extract diagonal singular values from the full matrix
        let s_data = s_full.as_slice().ok_or_else(|| {
            TensorError::invalid_shape_simple(
                "Singular values tensor is not contiguous".to_string(),
            )
        })?;

        let mut singular_values = Vec::new();
        for i in 0..min_dim {
            singular_values.push(s_data[i * n + i]);
        }

        // Threshold for numerical stability (relative to largest singular value)
        let threshold = if let Some(max_s) = singular_values
            .iter()
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
        {
            max_s.abs() * T::from(1e-15).unwrap_or(T::from(1e-10).unwrap())
        } else {
            T::from(1e-15).unwrap_or(T::from(1e-10).unwrap())
        };

        // Create pseudoinverse of sigma
        let mut s_pinv_data = vec![T::zero(); n * m]; // Note: transpose dimensions for pseudoinverse
        for i in 0..min_dim {
            if singular_values[i].abs() > threshold {
                s_pinv_data[i * m + i] = T::one() / singular_values[i];
            }
        }
        let s_pinv = Tensor::from_vec(s_pinv_data, &[n, m])?;

        // Compute A^+ = V * S^+ * U^T
        let ut = crate::ops::transpose(&u)?;
        let vs_pinv = crate::ops::matmul(&v, &s_pinv)?;
        let result = crate::ops::matmul(&vs_pinv, &ut)?;

        Ok(result)
    }
}

/// Check if LAPACK is available
pub fn is_lapack_available() -> bool {
    #[cfg(feature = "blas")]
    {
        true
    }
    #[cfg(not(feature = "blas"))]
    {
        false
    }
}

/// Get LAPACK provider information
pub fn lapack_provider() -> &'static str {
    #[cfg(feature = "blas-openblas")]
    {
        return "OpenBLAS";
    }
    #[cfg(feature = "blas-mkl")]
    {
        return "Intel MKL";
    }
    #[cfg(feature = "blas-accelerate")]
    {
        return "Apple Accelerate";
    }
    #[cfg(all(
        feature = "blas",
        not(any(
            feature = "blas-openblas",
            feature = "blas-mkl",
            feature = "blas-accelerate"
        ))
    ))]
    {
        return "Generic BLAS";
    }
    #[cfg(not(feature = "blas"))]
    {
        "None (Pure Rust)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_lapack_availability() {
        // This test will pass regardless of LAPACK availability
        let available = is_lapack_available();
        let provider = lapack_provider();

        println!("LAPACK available: {}", available);
        println!("Provider: {}", provider);

        assert!(provider.len() > 0);
    }

    #[test]
    fn test_matmul_consistency() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result_regular = crate::ops::matmul(&a, &b).unwrap();
        let result_lapack = matmul_blas(&a, &b).unwrap();

        // Results should be identical regardless of LAPACK availability
        for (r, l) in result_regular
            .as_slice()
            .unwrap()
            .iter()
            .zip(result_lapack.as_slice().unwrap().iter())
        {
            assert_abs_diff_eq!(r, l, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_determinant_consistency() {
        let matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result_regular = crate::ops::det(&matrix).unwrap().as_slice().unwrap()[0];
        let result_lapack = determinant_lapack(&matrix).unwrap();

        // Results should be close regardless of LAPACK availability
        assert_abs_diff_eq!(result_regular, result_lapack, epsilon = 1e-6);
    }
}
