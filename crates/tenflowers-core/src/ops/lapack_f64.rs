/*!
 * LAPACK operations specifically for f64 using scirs2-linalg (Pure Rust via OxiBLAS)
 */

use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{Array1, Array2};

/// Matrix inverse for f64 using scirs2-linalg
pub fn inverse_f64(input: &Tensor<f64>) -> Result<Tensor<f64>> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::invalid_shape_simple(
            "Matrix inverse requires contiguous tensor data".to_string(),
        )
    })?;

    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "Matrix inverse requires 2D tensor".to_string(),
        ));
    }

    if shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(format!(
            "Matrix inverse requires square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }

    let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|e| {
        TensorError::invalid_shape_simple(format!(
            "Failed to create Array2 from tensor data: {}",
            e
        ))
    })?;

    // Use scirs2-linalg's pure Rust inverse (via OxiBLAS)
    let result = scirs2_linalg::inv(&matrix.view(), None).map_err(|e| TensorError::BlasError {
        operation: "inv".to_string(),
        details: format!("scirs2-linalg inverse failed: {}", e),
        context: None,
    })?;

    Ok(Tensor::from_array(result.into_dyn()))
}

/// Matrix determinant for f64 using scirs2-linalg
pub fn determinant_f64(input: &Tensor<f64>) -> Result<f64> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::invalid_shape_simple(
            "Matrix determinant requires contiguous tensor data".to_string(),
        )
    })?;

    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "Matrix determinant requires 2D tensor".to_string(),
        ));
    }

    if shape[0] != shape[1] {
        return Err(TensorError::invalid_shape_simple(format!(
            "Matrix determinant requires square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }

    let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|e| {
        TensorError::invalid_shape_simple(format!(
            "Failed to create Array2 from tensor data: {}",
            e
        ))
    })?;

    // Use scirs2-linalg's pure Rust determinant (via OxiBLAS)
    scirs2_linalg::det(&matrix.view(), None).map_err(|e| TensorError::BlasError {
        operation: "det".to_string(),
        details: format!("scirs2-linalg determinant failed: {}", e),
        context: None,
    })
}

/// SVD for f64 using scirs2-linalg
pub fn svd_f64(input: &Tensor<f64>) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>)> {
    let data = input.as_slice().ok_or_else(|| {
        TensorError::invalid_shape_simple("SVD requires contiguous tensor data".to_string())
    })?;

    let shape = input.shape().dims();
    if shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "SVD requires 2D tensor".to_string(),
        ));
    }

    let matrix = Array2::from_shape_vec((shape[0], shape[1]), data.to_vec()).map_err(|e| {
        TensorError::invalid_shape_simple(format!(
            "Failed to create Array2 from tensor data: {}",
            e
        ))
    })?;

    // Use scirs2-linalg's pure Rust SVD (via OxiBLAS)
    // full_matrices=true to get full U and V matrices
    let (u, s, vt) =
        scirs2_linalg::svd(&matrix.view(), true, None).map_err(|e| TensorError::BlasError {
            operation: "svd".to_string(),
            details: format!("scirs2-linalg SVD failed: {}", e),
            context: None,
        })?;

    Ok((
        Tensor::from_array(u.into_dyn()),
        Tensor::from_array(s.into_dyn()),
        Tensor::from_array(vt.into_dyn()),
    ))
}

/// Solve linear system Ax = b for f64 using scirs2-linalg
pub fn solve_f64(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>> {
    let a_data = a.as_slice().ok_or_else(|| {
        TensorError::invalid_shape_simple(
            "Linear system solver requires contiguous tensor data for A".to_string(),
        )
    })?;

    let b_data = b.as_slice().ok_or_else(|| {
        TensorError::invalid_shape_simple(
            "Linear system solver requires contiguous tensor data for b".to_string(),
        )
    })?;

    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    if a_shape.len() != 2 {
        return Err(TensorError::invalid_shape_simple(
            "Linear system solver requires 2D matrix A".to_string(),
        ));
    }

    if b_shape.len() != 1 {
        return Err(TensorError::invalid_shape_simple(
            "Linear system solver requires 1D vector b".to_string(),
        ));
    }

    if a_shape[0] != a_shape[1] {
        return Err(TensorError::invalid_shape_simple(format!(
            "Linear system solver requires square matrix A, got {}x{}",
            a_shape[0], a_shape[1]
        )));
    }

    if a_shape[0] != b_shape[0] {
        return Err(TensorError::invalid_shape_simple(format!(
            "Matrix A and vector b dimensions don't match: A is {}x{}, b is {}",
            a_shape[0], a_shape[1], b_shape[0]
        )));
    }

    let a_matrix =
        Array2::from_shape_vec((a_shape[0], a_shape[1]), a_data.to_vec()).map_err(|e| {
            TensorError::invalid_shape_simple(format!(
                "Failed to create Array2 from tensor A: {}",
                e
            ))
        })?;

    let b_vector = Array1::from_shape_vec(b_shape[0], b_data.to_vec()).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Failed to create Array1 from tensor b: {}", e))
    })?;

    // Use scirs2-linalg's pure Rust linear solver (via OxiBLAS)
    let result = scirs2_linalg::solve(&a_matrix.view(), &b_vector.view(), None).map_err(|e| {
        TensorError::BlasError {
            operation: "solve".to_string(),
            details: format!("scirs2-linalg solve failed: {}", e),
            context: None,
        }
    })?;

    Ok(Tensor::from_array(result.into_dyn()))
}
