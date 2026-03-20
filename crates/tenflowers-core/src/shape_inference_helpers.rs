//! Shape Inference Helpers and Enhanced Diagnostics
//!
//! This module provides utilities for shape inference validation and
//! enhanced error messages for shape-related operations.

use crate::{Result, Shape, TensorError};

/// Validate and infer output shape for binary operations with broadcasting
///
/// # Arguments
/// * `shape_a` - Shape of first operand
/// * `shape_b` - Shape of second operand
/// * `op_name` - Name of operation for error messages
///
/// # Returns
/// Result containing the broadcasted output shape or detailed error
pub fn infer_binary_broadcast_shape(
    shape_a: &Shape,
    shape_b: &Shape,
    op_name: &str,
) -> Result<Shape> {
    let dims_a = shape_a.dims();
    let dims_b = shape_b.dims();

    // Handle scalar cases
    if dims_a.is_empty() {
        return Ok(shape_b.clone());
    }
    if dims_b.is_empty() {
        return Ok(shape_a.clone());
    }

    // Broadcast rules: iterate from the trailing dimensions
    let max_rank = dims_a.len().max(dims_b.len());
    let mut output_dims = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let dim_a = if i < dims_a.len() {
            dims_a[dims_a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < dims_b.len() {
            dims_b[dims_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a == dim_b {
            output_dims.push(dim_a);
        } else if dim_a == 1 {
            output_dims.push(dim_b);
        } else if dim_b == 1 {
            output_dims.push(dim_a);
        } else {
            return Err(TensorError::shape_mismatch(
                op_name,
                &format_shape_error(dims_a),
                &format_shape_error(dims_b),
            ));
        }
    }

    output_dims.reverse();
    Ok(Shape::from_slice(&output_dims))
}

/// Validate reduction operation along axis
///
/// # Arguments
/// * `input_shape` - Input tensor shape
/// * `axis` - Axis to reduce (None for all)
/// * `keep_dims` - Whether to keep reduced dimensions
/// * `op_name` - Operation name for errors
///
/// # Returns
/// Result containing output shape or detailed error
pub fn infer_reduction_shape(
    input_shape: &Shape,
    axis: Option<usize>,
    keep_dims: bool,
    op_name: &str,
) -> Result<Shape> {
    let dims = input_shape.dims();

    match axis {
        None => {
            // Reduce all dimensions
            if keep_dims {
                Ok(Shape::from_slice(&vec![1; dims.len()]))
            } else {
                Ok(Shape::from_slice(&[]))
            }
        }
        Some(ax) => {
            if ax >= dims.len() {
                return Err(TensorError::invalid_argument(format!(
                    "{}: axis {} out of bounds for tensor with rank {}. Valid axes: 0..{}",
                    op_name,
                    ax,
                    dims.len(),
                    dims.len()
                )));
            }

            let mut output_dims = dims.to_vec();
            if keep_dims {
                output_dims[ax] = 1;
            } else {
                output_dims.remove(ax);
            }

            Ok(Shape::from_slice(&output_dims))
        }
    }
}

/// Validate matmul shapes and infer output shape
///
/// # Arguments
/// * `shape_a` - Shape of left operand [... M, K]
/// * `shape_b` - Shape of right operand [... K, N]
/// * `op_name` - Operation name for errors
///
/// # Returns
/// Result containing output shape [... M, N] or detailed error
pub fn infer_matmul_shape(shape_a: &Shape, shape_b: &Shape, op_name: &str) -> Result<Shape> {
    let dims_a = shape_a.dims();
    let dims_b = shape_b.dims();

    if dims_a.len() < 2 {
        return Err(TensorError::invalid_argument(format!(
            "{}: left operand must have at least 2 dimensions, got shape {}",
            op_name,
            format_shape_error(dims_a)
        )));
    }

    if dims_b.len() < 2 {
        return Err(TensorError::invalid_argument(format!(
            "{}: right operand must have at least 2 dimensions, got shape {}",
            op_name,
            format_shape_error(dims_b)
        )));
    }

    let k_a = dims_a[dims_a.len() - 1];
    let k_b = dims_b[dims_b.len() - 2];

    if k_a != k_b {
        return Err(TensorError::shape_mismatch(
            op_name,
            &format!(
                "matmul inner dimensions: left[..., {}] vs right[{}, ...]",
                k_a, k_b
            ),
            &format!(
                "inner dimensions must match. Left shape: {}, Right shape: {}",
                format_shape_error(dims_a),
                format_shape_error(dims_b)
            ),
        ));
    }

    let m = dims_a[dims_a.len() - 2];
    let n = dims_b[dims_b.len() - 1];

    // Handle batch dimensions
    let max_batch_rank = (dims_a.len() - 2).max(dims_b.len() - 2);
    let mut output_dims = Vec::with_capacity(max_batch_rank + 2);

    for i in 0..max_batch_rank {
        let batch_dim_a = if i < dims_a.len() - 2 {
            dims_a[dims_a.len() - 3 - i]
        } else {
            1
        };
        let batch_dim_b = if i < dims_b.len() - 2 {
            dims_b[dims_b.len() - 3 - i]
        } else {
            1
        };

        if batch_dim_a != batch_dim_b && batch_dim_a != 1 && batch_dim_b != 1 {
            return Err(TensorError::shape_mismatch(
                op_name,
                &format_shape_error(dims_a),
                &format_shape_error(dims_b),
            ));
        }

        output_dims.push(batch_dim_a.max(batch_dim_b));
    }

    output_dims.reverse();
    output_dims.push(m);
    output_dims.push(n);

    Ok(Shape::from_slice(&output_dims))
}

/// Format shape for error messages with clear, readable output
///
/// # Arguments
/// * `dims` - Shape dimensions
///
/// # Returns
/// Formatted string like "[2, 3, 4]" for easier debugging
pub fn format_shape_error(dims: &[usize]) -> String {
    if dims.is_empty() {
        "[]".to_string()
    } else {
        format!(
            "[{}]",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Validate that shapes are compatible for element-wise operations
///
/// # Arguments
/// * `shape_a` - First shape
/// * `shape_b` - Second shape
/// * `op_name` - Operation name
///
/// # Returns
/// Result with () if compatible, or detailed error
pub fn validate_elementwise_compatible(
    shape_a: &Shape,
    shape_b: &Shape,
    op_name: &str,
) -> Result<()> {
    if shape_a.dims() == shape_b.dims() {
        return Ok(());
    }

    // Check if broadcasting is possible
    let _ = infer_binary_broadcast_shape(shape_a, shape_b, op_name)?;
    Ok(())
}

/// Validate reshape operation
///
/// # Arguments
/// * `input_shape` - Original shape
/// * `target_shape` - Desired shape
/// * `op_name` - Operation name
///
/// # Returns
/// Result with () if valid, or detailed error explaining the issue
pub fn validate_reshape(input_shape: &Shape, target_shape: &[usize], op_name: &str) -> Result<()> {
    let input_size: usize = input_shape.dims().iter().product();
    let target_size: usize = target_shape.iter().product();

    if input_size != target_size {
        return Err(TensorError::invalid_argument(format!(
            "{}: cannot reshape tensor of size {} (shape {}) into shape {}. Total elements must match.",
            op_name,
            input_size,
            format_shape_error(input_shape.dims()),
            format_shape_error(target_shape)
        )));
    }

    Ok(())
}

/// Generate helpful suggestion for common shape errors
///
/// # Arguments
/// * `actual` - Actual shape
/// * `expected` - Expected shape
/// * `op_name` - Operation name
///
/// # Returns
/// String with helpful suggestions for fixing the issue
pub fn suggest_shape_fix(actual: &[usize], expected: &[usize], op_name: &str) -> String {
    let actual_rank = actual.len();
    let expected_rank = expected.len();

    if actual_rank < expected_rank {
        format!(
            "Hint: Consider adding dimensions with .reshape() or .unsqueeze(). \
             {} expects {} dimensions but got {}",
            op_name, expected_rank, actual_rank
        )
    } else if actual_rank > expected_rank {
        format!(
            "Hint: Consider reducing dimensions with .squeeze() or selecting specific indices. \
             {} expects {} dimensions but got {}",
            op_name, expected_rank, actual_rank
        )
    } else {
        // Same rank, different sizes
        let mut suggestions = Vec::new();
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a != e && e != 1 && a != 1 {
                suggestions.push(format!(
                    "Dimension {}: expected {}, got {} (no broadcasting possible)",
                    i, e, a
                ));
            }
        }

        if suggestions.is_empty() {
            "Hint: Shapes differ but may be compatible through broadcasting".to_string()
        } else {
            format!("Incompatible dimensions:\n  {}", suggestions.join("\n  "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_broadcast_same_shape() {
        let shape_a = Shape::from_slice(&[2, 3, 4]);
        let shape_b = Shape::from_slice(&[2, 3, 4]);
        let result = infer_binary_broadcast_shape(&shape_a, &shape_b, "test_op")
            .expect("same shapes should broadcast");
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_binary_broadcast_compatible() {
        let shape_a = Shape::from_slice(&[2, 1, 4]);
        let shape_b = Shape::from_slice(&[2, 3, 4]);
        let result = infer_binary_broadcast_shape(&shape_a, &shape_b, "test_op")
            .expect("compatible shapes should broadcast");
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_binary_broadcast_scalar() {
        let shape_a = Shape::from_slice(&[]);
        let shape_b = Shape::from_slice(&[2, 3]);
        let result = infer_binary_broadcast_shape(&shape_a, &shape_b, "test_op")
            .expect("scalar should broadcast");
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_binary_broadcast_incompatible() {
        let shape_a = Shape::from_slice(&[2, 3]);
        let shape_b = Shape::from_slice(&[2, 4]);
        assert!(infer_binary_broadcast_shape(&shape_a, &shape_b, "test_op").is_err());
    }

    #[test]
    fn test_reduction_all_axes() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let result =
            infer_reduction_shape(&shape, None, false, "sum").expect("reduction should succeed");
        assert_eq!(result.dims(), &[] as &[usize]);
    }

    #[test]
    fn test_reduction_single_axis_keep_dims() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let result =
            infer_reduction_shape(&shape, Some(1), true, "sum").expect("reduction should succeed");
        assert_eq!(result.dims(), &[2, 1, 4]);
    }

    #[test]
    fn test_reduction_single_axis_no_keep() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        let result =
            infer_reduction_shape(&shape, Some(1), false, "sum").expect("reduction should succeed");
        assert_eq!(result.dims(), &[2, 4]);
    }

    #[test]
    fn test_reduction_invalid_axis() {
        let shape = Shape::from_slice(&[2, 3]);
        assert!(infer_reduction_shape(&shape, Some(5), false, "sum").is_err());
    }

    #[test]
    fn test_matmul_shape_valid() {
        let shape_a = Shape::from_slice(&[2, 3, 4]);
        let shape_b = Shape::from_slice(&[2, 4, 5]);
        let result = infer_matmul_shape(&shape_a, &shape_b, "matmul").expect("valid matmul shapes");
        assert_eq!(result.dims(), &[2, 3, 5]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let shape_a = Shape::from_slice(&[2, 3, 4]);
        let shape_b = Shape::from_slice(&[2, 5, 6]);
        assert!(infer_matmul_shape(&shape_a, &shape_b, "matmul").is_err());
    }

    #[test]
    fn test_format_shape_error() {
        assert_eq!(format_shape_error(&[2, 3, 4]), "[2, 3, 4]");
        assert_eq!(format_shape_error(&[]), "[]");
        assert_eq!(format_shape_error(&[10]), "[10]");
    }

    #[test]
    fn test_validate_reshape_compatible() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        assert!(validate_reshape(&shape, &[6, 4], "reshape").is_ok());
        assert!(validate_reshape(&shape, &[24], "reshape").is_ok());
        assert!(validate_reshape(&shape, &[2, 12], "reshape").is_ok());
    }

    #[test]
    fn test_validate_reshape_incompatible() {
        let shape = Shape::from_slice(&[2, 3, 4]);
        assert!(validate_reshape(&shape, &[2, 3, 5], "reshape").is_err());
        assert!(validate_reshape(&shape, &[25], "reshape").is_err());
    }

    #[test]
    fn test_suggest_shape_fix_rank_mismatch() {
        let actual = &[2, 3];
        let expected = &[2, 3, 4];
        let suggestion = suggest_shape_fix(actual, expected, "conv2d");
        assert!(suggestion.contains("unsqueeze") || suggestion.contains("reshape"));
    }

    #[test]
    fn test_suggest_shape_fix_dimension_mismatch() {
        let actual = &[2, 5, 4];
        let expected = &[2, 3, 4];
        let suggestion = suggest_shape_fix(actual, expected, "test_op");
        assert!(suggestion.contains("Dimension 1"));
    }
}
