//! Shape Computation and Broadcasting Utilities
//!
//! This module handles shape validation, computation, and broadcasting
//! for matrix multiplication operations.

use crate::shape_error_taxonomy::ShapeErrorUtils;
use crate::{Result, Shape, TensorError};

/// Compute the result shape for matrix multiplication with broadcasting
pub fn compute_matmul_shape(a_shape: &[usize], b_shape: &[usize]) -> Result<Vec<usize>> {
    if a_shape.is_empty() || b_shape.is_empty() {
        let empty_shape = Shape::from_slice(&[]);
        return Err(ShapeErrorUtils::rank_range_mismatch(
            "matmul",
            2,
            None,
            &empty_shape,
        ));
    }

    // Get the last two dimensions for matrix multiplication
    let a_rows = a_shape[a_shape.len() - 2];
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = b_shape[b_shape.len() - 2];
    let b_cols = b_shape[b_shape.len() - 1];

    if a_cols != b_rows {
        return Err(ShapeErrorUtils::matmul_incompatible(
            "matmul",
            &Shape::from_slice(a_shape),
            &Shape::from_slice(b_shape),
            false,
            false,
        ));
    }

    // Broadcast the batch dimensions
    let batch_shape =
        broadcast_shapes(&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2])?;

    // Result shape is batch dimensions + [a_rows, b_cols]
    let mut result_shape = batch_shape;
    result_shape.push(a_rows);
    result_shape.push(b_cols);

    Ok(result_shape)
}

/// Broadcast two shapes according to numpy broadcasting rules
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let mut result = Vec::new();
    let max_len = a.len().max(b.len());

    for i in 0..max_len {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        let result_dim = if a_dim == 1 {
            b_dim
        } else if b_dim == 1 || a_dim == b_dim {
            a_dim
        } else {
            return Err(ShapeErrorUtils::broadcast_incompatible(
                "matmul_broadcast",
                &Shape::from_slice(a),
                &Shape::from_slice(b),
            ));
        };

        result.push(result_dim);
    }

    result.reverse();
    Ok(result)
}

/// Compute broadcast indices for a specific position in the result tensor
pub fn compute_broadcast_indices(indices: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut broadcast_indices = Vec::new();
    let offset = indices.len() - shape.len();

    for (i, &dim) in shape.iter().enumerate() {
        if dim == 1 {
            broadcast_indices.push(0);
        } else {
            broadcast_indices.push(indices[offset + i]);
        }
    }

    broadcast_indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_matmul_shape() {
        // Basic 2D case
        let result = compute_matmul_shape(&[3, 4], &[4, 5]).unwrap();
        assert_eq!(result, &[3, 5]);

        // Batch case
        let result = compute_matmul_shape(&[2, 3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(result, &[2, 3, 5]);

        // Broadcasting case
        let result = compute_matmul_shape(&[1, 3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(result, &[2, 3, 5]);
    }

    #[test]
    fn test_broadcast_shapes() {
        // Compatible shapes
        let result = broadcast_shapes(&[1, 3], &[2, 1]).unwrap();
        assert_eq!(result, &[2, 3]);

        // Empty and non-empty
        let result = broadcast_shapes(&[], &[3, 4]).unwrap();
        assert_eq!(result, &[3, 4]);

        // Same shapes
        let result = broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, &[3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_error() {
        // Incompatible shapes
        let result = broadcast_shapes(&[3, 4], &[2, 5]);
        assert!(result.is_err());
    }
}
