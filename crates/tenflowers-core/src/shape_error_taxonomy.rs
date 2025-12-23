/// Standardized shape error taxonomy for TenfloweRS
///
/// This module provides utilities for creating consistent, helpful shape error messages
/// across all tensor operations. All operations should use these utilities to ensure
/// a uniform error reporting experience.
use crate::{Result, Shape, TensorError};

/// Category of shape error for better diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeErrorCategory {
    /// Shapes don't match for elementwise operations
    ElementwiseMismatch,
    /// Broadcasting rules violated
    BroadcastIncompatible,
    /// Matrix multiplication dimension mismatch
    MatMulIncompatible,
    /// Convolution parameter mismatch
    ConvolutionInvalid,
    /// Reduction axis invalid
    ReductionAxisInvalid,
    /// Reshape/view parameters invalid
    ReshapeInvalid,
    /// Concatenation/stacking dimension mismatch
    ConcatenationInvalid,
    /// Transpose/permutation invalid
    TransposeInvalid,
    /// Padding parameters invalid
    PaddingInvalid,
    /// General dimension constraint violation
    DimensionConstraintViolated,
}

impl ShapeErrorCategory {
    /// Get a user-friendly name for this error category
    pub fn name(&self) -> &'static str {
        match self {
            Self::ElementwiseMismatch => "Elementwise Shape Mismatch",
            Self::BroadcastIncompatible => "Broadcasting Incompatibility",
            Self::MatMulIncompatible => "Matrix Multiplication Incompatibility",
            Self::ConvolutionInvalid => "Convolution Parameter Invalid",
            Self::ReductionAxisInvalid => "Reduction Axis Invalid",
            Self::ReshapeInvalid => "Reshape Invalid",
            Self::ConcatenationInvalid => "Concatenation Invalid",
            Self::TransposeInvalid => "Transpose Invalid",
            Self::PaddingInvalid => "Padding Invalid",
            Self::DimensionConstraintViolated => "Dimension Constraint Violated",
        }
    }

    /// Get a description of how to fix this category of error
    pub fn fix_suggestion(&self) -> &'static str {
        match self {
            Self::ElementwiseMismatch => {
                "Ensure input tensors have identical shapes for elementwise operations"
            }
            Self::BroadcastIncompatible => {
                "Review NumPy broadcasting rules: dimensions must be equal or one of them must be 1"
            }
            Self::MatMulIncompatible => "For matmul(A, B), ensure A.shape[-1] == B.shape[-2]",
            Self::ConvolutionInvalid => {
                "Check kernel size, stride, padding, and dilation parameters"
            }
            Self::ReductionAxisInvalid => {
                "Ensure reduction axis is within [0, ndim) or use -1 for last axis"
            }
            Self::ReshapeInvalid => "New shape must have same total number of elements as original",
            Self::ConcatenationInvalid => {
                "All tensors must have same shape except in the concatenation dimension"
            }
            Self::TransposeInvalid => "Permutation must be a valid reordering of axes [0..ndim)",
            Self::PaddingInvalid => "Padding values must be non-negative",
            Self::DimensionConstraintViolated => {
                "Review operation documentation for dimension requirements"
            }
        }
    }
}

/// Detailed shape error builder for maximum clarity
pub struct ShapeErrorBuilder {
    operation: String,
    category: ShapeErrorCategory,
    expected: String,
    got: String,
    details: Vec<String>,
    suggestions: Vec<String>,
}

impl ShapeErrorBuilder {
    /// Create a new shape error builder
    pub fn new(operation: &str, category: ShapeErrorCategory) -> Self {
        Self {
            operation: operation.to_string(),
            category,
            expected: String::new(),
            got: String::new(),
            details: Vec::new(),
            suggestions: vec![category.fix_suggestion().to_string()],
        }
    }

    /// Set expected shape description
    pub fn expected(mut self, expected: &str) -> Self {
        self.expected = expected.to_string();
        self
    }

    /// Set actual shape description
    pub fn got(mut self, got: &str) -> Self {
        self.got = got.to_string();
        self
    }

    /// Add a detail line
    pub fn detail(mut self, detail: &str) -> Self {
        self.details.push(detail.to_string());
        self
    }

    /// Add a suggestion for fixing the error
    pub fn suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    /// Build the final error
    pub fn build(self) -> TensorError {
        let mut message = format!(
            "[{}] in operation '{}'",
            self.category.name(),
            self.operation
        );

        if !self.expected.is_empty() {
            message.push_str(&format!("\nExpected: {}", self.expected));
        }

        if !self.got.is_empty() {
            message.push_str(&format!("\nGot:      {}", self.got));
        }

        if !self.details.is_empty() {
            message.push_str("\n\nDetails:");
            for detail in &self.details {
                message.push_str(&format!("\n  • {}", detail));
            }
        }

        if !self.suggestions.is_empty() {
            message.push_str("\n\nSuggestions:");
            for suggestion in &self.suggestions {
                message.push_str(&format!("\n  • {}", suggestion));
            }
        }

        TensorError::invalid_shape_simple(message)
    }
}

/// Utilities for common shape error scenarios
pub struct ShapeErrorUtils;

impl ShapeErrorUtils {
    /// Create error for elementwise operation shape mismatch
    pub fn elementwise_mismatch(operation: &str, shape1: &Shape, shape2: &Shape) -> TensorError {
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::ElementwiseMismatch)
            .expected(&format!("identical shapes: {}", shape1))
            .got(&format!("shapes {} and {}", shape1, shape2))
            .detail("Elementwise operations require tensors with identical shapes")
            .build()
    }

    /// Create error for broadcasting incompatibility
    pub fn broadcast_incompatible(operation: &str, shape1: &Shape, shape2: &Shape) -> TensorError {
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::BroadcastIncompatible)
            .expected(&format!(
                "broadcastable shapes (matching dims or dim=1): {} and {}",
                shape1, shape2
            ))
            .got(&format!(
                "non-broadcastable shapes {} and {}",
                shape1, shape2
            ))
            .detail("Broadcasting rules: dimensions must match or one must be 1")
            .build()
    }

    /// Create error for matrix multiplication dimension mismatch
    pub fn matmul_incompatible(
        operation: &str,
        shape_a: &Shape,
        shape_b: &Shape,
        transpose_a: bool,
        transpose_b: bool,
    ) -> TensorError {
        let (m, k1) = if transpose_a {
            (shape_a.dims()[1], shape_a.dims()[0])
        } else {
            (shape_a.dims()[0], shape_a.dims()[1])
        };

        let (k2, n) = if transpose_b {
            (shape_b.dims()[1], shape_b.dims()[0])
        } else {
            (shape_b.dims()[0], shape_b.dims()[1])
        };

        ShapeErrorBuilder::new(operation, ShapeErrorCategory::MatMulIncompatible)
            .expected(&format!(
                "compatible matrix dimensions: inner dimensions must match (k1={} should equal k2={})",
                k1, k2
            ))
            .got(&format!(
                "A{}: {} ({}×{}), B{}: {} ({}×{})",
                if transpose_a { ".T" } else { "" },
                shape_a,
                m,
                k1,
                if transpose_b { ".T" } else { "" },
                shape_b,
                k2,
                n
            ))
            .detail(&format!("Result shape would be: ({}, {})", m, n))
            .detail(&format!(
                "Transpose flags: transpose_a={}, transpose_b={}",
                transpose_a, transpose_b
            ))
            .build()
    }

    /// Create error for invalid reduction axis
    pub fn invalid_reduction_axis(operation: &str, axis: isize, shape: &Shape) -> TensorError {
        let ndim = shape.rank();
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::ReductionAxisInvalid)
            .expected(&format!("axis in range [0, {}) or [-{}, -1]", ndim, ndim))
            .got(&format!("axis = {}", axis))
            .detail(&format!("Tensor shape: {}", shape))
            .detail(&format!("Number of dimensions: {}", ndim))
            .suggestion("Use axis=-1 to reduce over the last dimension")
            .build()
    }

    /// Create error for invalid reshape
    pub fn invalid_reshape(
        operation: &str,
        original_shape: &Shape,
        new_shape: &[usize],
    ) -> TensorError {
        let original_size: usize = original_shape.dims().iter().product();
        let new_size: usize = new_shape.iter().product();

        ShapeErrorBuilder::new(operation, ShapeErrorCategory::ReshapeInvalid)
            .expected(&format!(
                "new shape with total elements = {} (same as original)",
                original_size
            ))
            .got(&format!(
                "shape {:?} with total elements = {}",
                new_shape, new_size
            ))
            .detail(&format!("Original shape: {}", original_shape))
            .detail(&format!("Original size: {}", original_size))
            .detail(&format!("New shape: {:?}", new_shape))
            .detail(&format!("New size: {}", new_size))
            .suggestion("Use -1 in one dimension to infer its size automatically")
            .build()
    }

    /// Create error for concatenation shape mismatch
    pub fn concatenation_mismatch(operation: &str, shapes: &[Shape], axis: usize) -> TensorError {
        let mut builder =
            ShapeErrorBuilder::new(operation, ShapeErrorCategory::ConcatenationInvalid);

        if let Some(first_shape) = shapes.first() {
            builder = builder.expected(&format!(
                "all tensors to have same shape as first tensor {} (except in axis {})",
                first_shape, axis
            ));

            for (i, shape) in shapes.iter().enumerate().skip(1) {
                if shape != first_shape {
                    let mut diff_axes = Vec::new();
                    for (ax, (d1, d2)) in first_shape.dims().iter().zip(shape.dims()).enumerate() {
                        if d1 != d2 && ax != axis {
                            diff_axes.push(ax);
                        }
                    }
                    if !diff_axes.is_empty() {
                        builder = builder.detail(&format!(
                            "Tensor {} differs from first tensor in axes {:?} (non-concat axes must match)",
                            i, diff_axes
                        ));
                    }
                }
            }
        }

        builder = builder.detail(&format!("Concatenation axis: {}", axis));
        for (i, shape) in shapes.iter().enumerate() {
            builder = builder.detail(&format!("Tensor {}: {}", i, shape));
        }

        builder.build()
    }

    /// Create error for dimension constraint violation
    pub fn dimension_constraint(
        operation: &str,
        constraint_description: &str,
        shape: &Shape,
    ) -> TensorError {
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::DimensionConstraintViolated)
            .expected(constraint_description)
            .got(&format!("shape {}", shape))
            .detail(&format!("Actual rank: {}", shape.rank()))
            .build()
    }

    /// Create error for invalid transpose/permutation
    pub fn invalid_transpose(operation: &str, shape: &Shape, axes: &[usize]) -> TensorError {
        let ndim = shape.rank();
        let expected_axes: Vec<usize> = (0..ndim).collect();

        ShapeErrorBuilder::new(operation, ShapeErrorCategory::TransposeInvalid)
            .expected(&format!("permutation of {:?}", expected_axes))
            .got(&format!("axes {:?}", axes))
            .detail(&format!("Tensor shape: {}", shape))
            .detail(&format!("Number of dimensions: {}", ndim))
            .detail("Permutation must contain each axis index exactly once")
            .build()
    }

    /// Create error for convolution parameter mismatch
    pub fn convolution_invalid(
        operation: &str,
        input_shape: &Shape,
        kernel_shape: &Shape,
        details: &str,
    ) -> TensorError {
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::ConvolutionInvalid)
            .detail(&format!("Input shape: {}", input_shape))
            .detail(&format!("Kernel shape: {}", kernel_shape))
            .detail(details)
            .suggestion("Check that kernel size, stride, padding, and dilation are valid")
            .suggestion("Ensure input channels match kernel input channels")
            .build()
    }

    /// Create error for rank mismatch
    pub fn rank_mismatch(
        operation: &str,
        expected_rank: usize,
        actual_shape: &Shape,
    ) -> TensorError {
        ShapeErrorBuilder::new(operation, ShapeErrorCategory::DimensionConstraintViolated)
            .expected(&format!("{}-dimensional tensor", expected_rank))
            .got(&format!(
                "{}-dimensional tensor with shape {}",
                actual_shape.rank(),
                actual_shape
            ))
            .build()
    }

    /// Create error for rank range mismatch
    pub fn rank_range_mismatch(
        operation: &str,
        min_rank: usize,
        max_rank: Option<usize>,
        actual_shape: &Shape,
    ) -> TensorError {
        let expected = if let Some(max) = max_rank {
            format!("tensor with rank in range [{}, {}]", min_rank, max)
        } else {
            format!("tensor with rank >= {}", min_rank)
        };

        ShapeErrorBuilder::new(operation, ShapeErrorCategory::DimensionConstraintViolated)
            .expected(&expected)
            .got(&format!(
                "rank {} tensor with shape {}",
                actual_shape.rank(),
                actual_shape
            ))
            .build()
    }
}

/// Validate shape compatibility and return detailed error if invalid
pub fn validate_elementwise_shapes(operation: &str, shape1: &Shape, shape2: &Shape) -> Result<()> {
    if shape1 != shape2 {
        Err(ShapeErrorUtils::elementwise_mismatch(
            operation, shape1, shape2,
        ))
    } else {
        Ok(())
    }
}

/// Validate broadcast compatibility
pub fn validate_broadcast_shapes(operation: &str, shape1: &Shape, shape2: &Shape) -> Result<Shape> {
    shape1
        .broadcast_shape(shape2)
        .ok_or_else(|| ShapeErrorUtils::broadcast_incompatible(operation, shape1, shape2))
}

/// Validate matrix multiplication shapes
pub fn validate_matmul_shapes(
    operation: &str,
    shape_a: &Shape,
    shape_b: &Shape,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Shape> {
    if shape_a.rank() != 2 || shape_b.rank() != 2 {
        return Err(TensorError::invalid_shape_simple(format!(
            "Matrix multiplication requires 2D tensors, got shapes {} and {}",
            shape_a, shape_b
        )));
    }

    let dims_a = shape_a.dims();
    let dims_b = shape_b.dims();

    let (m, k1) = if transpose_a {
        (dims_a[1], dims_a[0])
    } else {
        (dims_a[0], dims_a[1])
    };

    let (k2, n) = if transpose_b {
        (dims_b[1], dims_b[0])
    } else {
        (dims_b[0], dims_b[1])
    };

    if k1 != k2 {
        Err(ShapeErrorUtils::matmul_incompatible(
            operation,
            shape_a,
            shape_b,
            transpose_a,
            transpose_b,
        ))
    } else {
        Ok(Shape::from_slice(&[m, n]))
    }
}

/// Validate reduction axis
pub fn validate_reduction_axis(operation: &str, axis: isize, shape: &Shape) -> Result<usize> {
    let ndim = shape.rank() as isize;
    let normalized_axis = if axis < 0 { ndim + axis } else { axis };

    if normalized_axis < 0 || normalized_axis >= ndim {
        Err(ShapeErrorUtils::invalid_reduction_axis(
            operation, axis, shape,
        ))
    } else {
        Ok(normalized_axis as usize)
    }
}

/// Validate reshape compatibility
pub fn validate_reshape(
    operation: &str,
    original_shape: &Shape,
    new_shape: &[usize],
) -> Result<()> {
    let original_size: usize = original_shape.dims().iter().product();
    let new_size: usize = new_shape.iter().product();

    if original_size != new_size {
        Err(ShapeErrorUtils::invalid_reshape(
            operation,
            original_shape,
            new_shape,
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_mismatch_error() {
        let shape1 = Shape::from_slice(&[3, 4]);
        let shape2 = Shape::from_slice(&[3, 5]);
        let err = ShapeErrorUtils::elementwise_mismatch("add", &shape1, &shape2);
        let msg = format!("{}", err);
        assert!(msg.contains("Elementwise Shape Mismatch"));
        assert!(msg.contains("add"));
    }

    #[test]
    fn test_matmul_incompatible_error() {
        let shape_a = Shape::from_slice(&[3, 4]);
        let shape_b = Shape::from_slice(&[5, 6]);
        let err = ShapeErrorUtils::matmul_incompatible("matmul", &shape_a, &shape_b, false, false);
        let msg = format!("{}", err);
        assert!(msg.contains("Matrix Multiplication Incompatibility"));
        assert!(msg.contains("matmul"));
    }

    #[test]
    fn test_validate_matmul_shapes() {
        let shape_a = Shape::from_slice(&[3, 4]);
        let shape_b = Shape::from_slice(&[4, 5]);
        let result = validate_matmul_shapes("matmul", &shape_a, &shape_b, false, false);
        assert!(result.is_ok());
        let output_shape = result.unwrap();
        assert_eq!(output_shape.dims(), &[3, 5]);
    }

    #[test]
    fn test_validate_matmul_shapes_incompatible() {
        let shape_a = Shape::from_slice(&[3, 4]);
        let shape_b = Shape::from_slice(&[5, 6]);
        let result = validate_matmul_shapes("matmul", &shape_a, &shape_b, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_reduction_axis() {
        let shape = Shape::from_slice(&[3, 4, 5]);
        assert!(validate_reduction_axis("sum", 0, &shape).is_ok());
        assert!(validate_reduction_axis("sum", 1, &shape).is_ok());
        assert!(validate_reduction_axis("sum", 2, &shape).is_ok());
        assert!(validate_reduction_axis("sum", -1, &shape).is_ok());
        assert!(validate_reduction_axis("sum", -2, &shape).is_ok());
        assert!(validate_reduction_axis("sum", 3, &shape).is_err());
        assert!(validate_reduction_axis("sum", -4, &shape).is_err());
    }

    #[test]
    fn test_validate_reshape() {
        let shape = Shape::from_slice(&[3, 4]);
        assert!(validate_reshape("reshape", &shape, &[12]).is_ok());
        assert!(validate_reshape("reshape", &shape, &[2, 6]).is_ok());
        assert!(validate_reshape("reshape", &shape, &[2, 7]).is_err());
    }
}
