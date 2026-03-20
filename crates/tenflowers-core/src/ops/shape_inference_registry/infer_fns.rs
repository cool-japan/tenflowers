//! Shape inference functions for all built-in operations.

use super::types::OperationMetadata;
use crate::ops::shape_inference::{
    infer_binary_elementwise, infer_matmul, BroadcastableConstraint, MinRankConstraint,
    RankConstraint, ShapeConstraint, ShapeValidator,
};
use crate::shape_error_taxonomy::{ShapeErrorBuilder, ShapeErrorCategory, ShapeErrorUtils};
use crate::{Result, Shape, TensorError};

// ============================================================================
// Binary Elementwise Operations
// ============================================================================

/// Infer shape for binary elementwise operations (add, sub, mul, div, pow)
pub(super) fn infer_add(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(
            ShapeErrorBuilder::new("add", ShapeErrorCategory::ElementwiseMismatch)
                .expected("exactly 2 input tensors")
                .got(&format!("{} input tensors", inputs.len()))
                .build(),
        );
    }
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

pub(super) fn infer_sub(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(
            ShapeErrorBuilder::new("sub", ShapeErrorCategory::ElementwiseMismatch)
                .expected("exactly 2 input tensors")
                .got(&format!("{} input tensors", inputs.len()))
                .build(),
        );
    }
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

pub(super) fn infer_mul(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(
            ShapeErrorBuilder::new("mul", ShapeErrorCategory::ElementwiseMismatch)
                .expected("exactly 2 input tensors")
                .got(&format!("{} input tensors", inputs.len()))
                .build(),
        );
    }
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

pub(super) fn infer_div(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(
            ShapeErrorBuilder::new("div", ShapeErrorCategory::ElementwiseMismatch)
                .expected("exactly 2 input tensors")
                .got(&format!("{} input tensors", inputs.len()))
                .build(),
        );
    }
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

pub(super) fn infer_pow(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(
            ShapeErrorBuilder::new("pow", ShapeErrorCategory::ElementwiseMismatch)
                .expected("exactly 2 input tensors")
                .got(&format!("{} input tensors", inputs.len()))
                .build(),
        );
    }
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Infer shape for unary elementwise operations
pub(super) fn infer_unary(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 1 {
        return Err(TensorError::invalid_argument(format!(
            "Unary operation expects exactly 1 input, got {}",
            inputs.len()
        )));
    }
    Ok(inputs[0].clone())
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Infer shape for matrix multiplication
pub(super) fn infer_matmul_op(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "matmul expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    let transpose_a = metadata
        .get("transpose_a")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let transpose_b = metadata
        .get("transpose_b")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    infer_matmul(&inputs[0], &inputs[1], transpose_a, transpose_b)
}

/// Infer shape for dot product
pub(super) fn infer_dot(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "dot expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    }

    // Dot product typically results in a scalar for 1D vectors
    if inputs[0].rank() == 1 && inputs[1].rank() == 1 {
        if inputs[0].dims()[0] != inputs[1].dims()[0] {
            return Err(ShapeErrorUtils::matmul_incompatible(
                "dot", &inputs[0], &inputs[1], false, false,
            ));
        }
        Ok(Shape::from_slice(&[]))
    } else {
        // For higher dimensions, fall back to matmul rules
        infer_matmul(&inputs[0], &inputs[1], false, false)
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Infer shape for reduction operations
pub(super) fn infer_reduction(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "Reduction operation requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    let axis = metadata.get("axis").and_then(|v| v.as_int());
    let keepdims = metadata
        .get("keepdims")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Some(ax) = axis {
        // Validate axis
        let axis_usize = if ax < 0 {
            let positive_axis = (input_shape.rank() as i64 + ax) as usize;
            if positive_axis >= input_shape.rank() {
                return Err(ShapeErrorBuilder::new(
                    "reduction",
                    ShapeErrorCategory::ReductionAxisInvalid,
                )
                .expected(&format!(
                    "axis in range [-{}, {})",
                    input_shape.rank(),
                    input_shape.rank()
                ))
                .got(&format!("axis = {}", ax))
                .build());
            }
            positive_axis
        } else {
            let ax_usize = ax as usize;
            if ax_usize >= input_shape.rank() {
                return Err(ShapeErrorBuilder::new(
                    "reduction",
                    ShapeErrorCategory::ReductionAxisInvalid,
                )
                .expected(&format!("axis in range [0, {})", input_shape.rank()))
                .got(&format!("axis = {}", ax))
                .build());
            }
            ax_usize
        };

        // Compute output shape
        if keepdims {
            let mut out_dims = input_shape.dims().to_vec();
            out_dims[axis_usize] = 1;
            Ok(Shape::from_slice(&out_dims))
        } else {
            let mut out_dims = input_shape.dims().to_vec();
            out_dims.remove(axis_usize);
            if out_dims.is_empty() {
                Ok(Shape::from_slice(&[]))
            } else {
                Ok(Shape::from_slice(&out_dims))
            }
        }
    } else {
        // Reduce all dimensions
        if keepdims {
            Ok(Shape::from_slice(&vec![1; input_shape.rank()]))
        } else {
            Ok(Shape::from_slice(&[]))
        }
    }
}

// ============================================================================
// Manipulation Operations
// ============================================================================

/// Infer shape for reshape operation
pub(super) fn infer_reshape(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "reshape requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    let new_shape_vec = metadata
        .get("shape")
        .and_then(|v| v.as_int_vec())
        .ok_or_else(|| {
            TensorError::invalid_argument("reshape requires 'shape' metadata".to_string())
        })?;

    // Convert i64 to usize and handle -1 (infer dimension)
    let input_numel = input_shape.elements();
    let mut new_dims: Vec<usize> = Vec::new();
    let mut infer_index: Option<usize> = None;

    for (i, &dim) in new_shape_vec.iter().enumerate() {
        if dim == -1 {
            if infer_index.is_some() {
                return Err(
                    ShapeErrorBuilder::new("reshape", ShapeErrorCategory::ReshapeInvalid)
                        .detail("Can only specify one -1 dimension in reshape")
                        .build(),
                );
            }
            infer_index = Some(i);
            new_dims.push(0); // Placeholder
        } else if dim <= 0 {
            return Err(
                ShapeErrorBuilder::new("reshape", ShapeErrorCategory::ReshapeInvalid)
                    .detail(&format!("Invalid dimension size: {}", dim))
                    .build(),
            );
        } else {
            new_dims.push(dim as usize);
        }
    }

    // Infer -1 dimension if present
    if let Some(idx) = infer_index {
        let known_numel: usize = new_dims.iter().filter(|&&d| d != 0).product();
        if known_numel == 0 || input_numel % known_numel != 0 {
            return Err(
                ShapeErrorBuilder::new("reshape", ShapeErrorCategory::ReshapeInvalid)
                    .expected(&format!(
                        "new shape compatible with {} elements",
                        input_numel
                    ))
                    .got("new shape would require non-integer dimension")
                    .build(),
            );
        }
        new_dims[idx] = input_numel / known_numel;
    }

    // Validate total number of elements
    let new_numel: usize = new_dims.iter().product();
    if new_numel != input_numel {
        return Err(
            ShapeErrorBuilder::new("reshape", ShapeErrorCategory::ReshapeInvalid)
                .expected(&format!("new shape with {} elements", input_numel))
                .got(&format!("new shape with {} elements", new_numel))
                .build(),
        );
    }

    Ok(Shape::from_slice(&new_dims))
}

/// Infer shape for transpose operation
pub(super) fn infer_transpose(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "transpose requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    if input_shape.rank() < 2 {
        return Err(
            ShapeErrorBuilder::new("transpose", ShapeErrorCategory::TransposeInvalid)
                .expected("tensor with rank >= 2")
                .got(&format!("tensor with rank {}", input_shape.rank()))
                .build(),
        );
    }

    // Default transpose swaps last two dimensions
    let mut out_dims = input_shape.dims().to_vec();
    let rank = out_dims.len();
    out_dims.swap(rank - 2, rank - 1);

    Ok(Shape::from_slice(&out_dims))
}

/// Infer shape for permute operation
pub(super) fn infer_permute(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "permute requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    let axes = metadata
        .get("axes")
        .and_then(|v| v.as_uint_vec())
        .ok_or_else(|| {
            TensorError::invalid_argument("permute requires 'axes' metadata".to_string())
        })?;

    if axes.len() != input_shape.rank() {
        return Err(
            ShapeErrorBuilder::new("permute", ShapeErrorCategory::TransposeInvalid)
                .expected(&format!("permutation with {} axes", input_shape.rank()))
                .got(&format!("permutation with {} axes", axes.len()))
                .build(),
        );
    }

    // Validate permutation
    let mut seen = vec![false; input_shape.rank()];
    for &ax in axes {
        if ax >= input_shape.rank() {
            return Err(
                ShapeErrorBuilder::new("permute", ShapeErrorCategory::TransposeInvalid)
                    .detail(&format!(
                        "Invalid axis {} (must be < {})",
                        ax,
                        input_shape.rank()
                    ))
                    .build(),
            );
        }
        if seen[ax] {
            return Err(
                ShapeErrorBuilder::new("permute", ShapeErrorCategory::TransposeInvalid)
                    .detail(&format!("Duplicate axis {} in permutation", ax))
                    .build(),
            );
        }
        seen[ax] = true;
    }

    // Apply permutation
    let in_dims = input_shape.dims();
    let out_dims: Vec<usize> = axes.iter().map(|&i| in_dims[i]).collect();

    Ok(Shape::from_slice(&out_dims))
}

/// Infer shape for squeeze operation
pub(super) fn infer_squeeze(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "squeeze requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    let axis = metadata.get("axis").and_then(|v| v.as_int());

    if let Some(ax) = axis {
        // Squeeze specific axis
        let ax_usize = if ax < 0 {
            (input_shape.rank() as i64 + ax) as usize
        } else {
            ax as usize
        };

        if ax_usize >= input_shape.rank() {
            return Err(TensorError::invalid_argument(format!(
                "squeeze axis {} out of bounds for shape {:?}",
                ax,
                input_shape.dims()
            )));
        }

        if input_shape.dims()[ax_usize] != 1 {
            return Err(TensorError::invalid_argument(format!(
                "Cannot squeeze axis {} with size {}",
                ax,
                input_shape.dims()[ax_usize]
            )));
        }

        let mut out_dims = input_shape.dims().to_vec();
        out_dims.remove(ax_usize);

        if out_dims.is_empty() {
            Ok(Shape::from_slice(&[]))
        } else {
            Ok(Shape::from_slice(&out_dims))
        }
    } else {
        // Squeeze all dimensions of size 1
        let out_dims: Vec<usize> = input_shape
            .dims()
            .iter()
            .filter(|&&d| d != 1)
            .copied()
            .collect();

        if out_dims.is_empty() {
            Ok(Shape::from_slice(&[]))
        } else {
            Ok(Shape::from_slice(&out_dims))
        }
    }
}

/// Infer shape for unsqueeze operation
pub(super) fn infer_unsqueeze(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "unsqueeze requires at least 1 input".to_string(),
        ));
    }

    let input_shape = &inputs[0];
    let axis = metadata
        .get("axis")
        .and_then(|v| v.as_int())
        .ok_or_else(|| {
            TensorError::invalid_argument("unsqueeze requires 'axis' metadata".to_string())
        })?;

    let new_rank = input_shape.rank() + 1;
    let ax_usize = if axis < 0 {
        (new_rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    if ax_usize > input_shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "unsqueeze axis {} out of bounds for new rank {}",
            axis, new_rank
        )));
    }

    let mut out_dims = input_shape.dims().to_vec();
    out_dims.insert(ax_usize, 1);

    Ok(Shape::from_slice(&out_dims))
}

// ============================================================================
// Concatenation Operations
// ============================================================================

/// Infer shape for concatenation operation
pub(super) fn infer_concat(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "concat requires at least 1 input".to_string(),
        ));
    }

    let axis = metadata.get("axis").and_then(|v| v.as_int()).unwrap_or(0);

    let first_shape = &inputs[0];
    let ax_usize = if axis < 0 {
        (first_shape.rank() as i64 + axis) as usize
    } else {
        axis as usize
    };

    if ax_usize >= first_shape.rank() {
        return Err(
            ShapeErrorBuilder::new("concat", ShapeErrorCategory::ConcatenationInvalid)
                .expected(&format!("axis in range [0, {})", first_shape.rank()))
                .got(&format!("axis = {}", axis))
                .build(),
        );
    }

    // Validate all shapes match except on concat axis
    let mut concat_size = first_shape.dims()[ax_usize];
    for (i, shape) in inputs.iter().enumerate().skip(1) {
        if shape.rank() != first_shape.rank() {
            return Err(
                ShapeErrorBuilder::new("concat", ShapeErrorCategory::ConcatenationInvalid)
                    .expected(&format!("all tensors to have rank {}", first_shape.rank()))
                    .got(&format!("tensor {} has rank {}", i, shape.rank()))
                    .build(),
            );
        }

        for (dim_idx, (&dim1, &dim2)) in first_shape
            .dims()
            .iter()
            .zip(shape.dims().iter())
            .enumerate()
        {
            if dim_idx != ax_usize && dim1 != dim2 {
                return Err(ShapeErrorBuilder::new(
                    "concat",
                    ShapeErrorCategory::ConcatenationInvalid,
                )
                .expected(&format!(
                    "dimension {} to match: {} == {}",
                    dim_idx, dim1, dim2
                ))
                .got(&format!(
                    "dimension {} mismatch: {} != {}",
                    dim_idx, dim1, dim2
                ))
                .build());
            }
        }

        concat_size += shape.dims()[ax_usize];
    }

    // Build output shape
    let mut out_dims = first_shape.dims().to_vec();
    out_dims[ax_usize] = concat_size;

    Ok(Shape::from_slice(&out_dims))
}

/// Infer shape for stack operation
pub(super) fn infer_stack(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "stack requires at least 1 input".to_string(),
        ));
    }

    let axis = metadata.get("axis").and_then(|v| v.as_int()).unwrap_or(0);

    let first_shape = &inputs[0];

    // All shapes must be identical for stack
    for (i, shape) in inputs.iter().enumerate().skip(1) {
        if shape.dims() != first_shape.dims() {
            return Err(
                ShapeErrorBuilder::new("stack", ShapeErrorCategory::ConcatenationInvalid)
                    .expected(&format!(
                        "all tensors to have shape {:?}",
                        first_shape.dims()
                    ))
                    .got(&format!("tensor {} has shape {:?}", i, shape.dims()))
                    .build(),
            );
        }
    }

    let new_rank = first_shape.rank() + 1;
    let ax_usize = if axis < 0 {
        (new_rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    if ax_usize > first_shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "stack axis {} out of bounds for new rank {}",
            axis, new_rank
        )));
    }

    // Insert new dimension at stack axis
    let mut out_dims = first_shape.dims().to_vec();
    out_dims.insert(ax_usize, inputs.len());

    Ok(Shape::from_slice(&out_dims))
}

// ============================================================================
// Comparison and Logical Operations
// ============================================================================

/// Infer shape for comparison operations
pub(super) fn infer_comparison(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "Comparison operation expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    }
    // Comparison operations broadcast like binary elementwise
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

/// Infer shape for logical operations
pub(super) fn infer_logical(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "Logical operation expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    }
    // Logical operations broadcast like binary elementwise
    infer_binary_elementwise(&inputs[0], &inputs[1])
}
