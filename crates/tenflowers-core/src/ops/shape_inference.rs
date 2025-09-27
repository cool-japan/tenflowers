use crate::{Result, Shape, TensorError};
use std::collections::HashMap;

/// Shape inference context for tracking tensor shapes through operations
#[derive(Debug, Clone)]
pub struct ShapeContext {
    shapes: HashMap<String, Shape>,
}

impl ShapeContext {
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    /// Register a tensor shape
    pub fn set_shape(&mut self, name: &str, shape: Shape) {
        self.shapes.insert(name.to_string(), shape);
    }

    /// Get a tensor shape
    pub fn get_shape(&self, name: &str) -> Option<&Shape> {
        self.shapes.get(name)
    }
}

impl Default for ShapeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time shape validation traits and types
/// Trait for compile-time shape constraints
pub trait ShapeConstraint {
    /// Check if a shape satisfies this constraint
    fn validate(&self, shape: &Shape) -> Result<()>;

    /// Get a description of this constraint for error messages
    fn description(&self) -> String;
}

/// Fixed rank constraint
#[derive(Debug, Clone)]
pub struct RankConstraint {
    expected_rank: usize,
}

impl RankConstraint {
    pub fn new(rank: usize) -> Self {
        Self {
            expected_rank: rank,
        }
    }
}

impl ShapeConstraint for RankConstraint {
    fn validate(&self, shape: &Shape) -> Result<()> {
        if shape.rank() == self.expected_rank {
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Expected rank {}, got {} (shape: {:?})",
                self.expected_rank,
                shape.rank(),
                shape.dims()
            )))
        }
    }

    fn description(&self) -> String {
        format!("rank = {}", self.expected_rank)
    }
}

/// Exact shape constraint
#[derive(Debug, Clone)]
pub struct ExactShapeConstraint {
    expected_shape: Shape,
}

impl ExactShapeConstraint {
    pub fn new(shape: Shape) -> Self {
        Self {
            expected_shape: shape,
        }
    }
}

impl ShapeConstraint for ExactShapeConstraint {
    fn validate(&self, shape: &Shape) -> Result<()> {
        if shape.dims() == self.expected_shape.dims() {
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Expected exact shape {:?}, got {:?}",
                self.expected_shape.dims(),
                shape.dims()
            )))
        }
    }

    fn description(&self) -> String {
        format!("exact shape = {:?}", self.expected_shape.dims())
    }
}

/// Minimum rank constraint
#[derive(Debug, Clone)]
pub struct MinRankConstraint {
    min_rank: usize,
}

impl MinRankConstraint {
    pub fn new(min_rank: usize) -> Self {
        Self { min_rank }
    }
}

impl ShapeConstraint for MinRankConstraint {
    fn validate(&self, shape: &Shape) -> Result<()> {
        if shape.rank() >= self.min_rank {
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Expected minimum rank {}, got {} (shape: {:?})",
                self.min_rank,
                shape.rank(),
                shape.dims()
            )))
        }
    }

    fn description(&self) -> String {
        format!("rank >= {}", self.min_rank)
    }
}

/// Broadcastable constraint - checks if two shapes can be broadcast together
#[derive(Debug, Clone)]
pub struct BroadcastableConstraint {
    reference_shape: Shape,
}

impl BroadcastableConstraint {
    pub fn new(reference_shape: Shape) -> Self {
        Self { reference_shape }
    }
}

impl ShapeConstraint for BroadcastableConstraint {
    fn validate(&self, shape: &Shape) -> Result<()> {
        if self.reference_shape.broadcast_shape(shape).is_some() {
            Ok(())
        } else {
            Err(TensorError::invalid_argument(format!(
                "Shape {:?} is not broadcastable with reference shape {:?}",
                shape.dims(),
                self.reference_shape.dims()
            )))
        }
    }

    fn description(&self) -> String {
        format!("broadcastable with {:?}", self.reference_shape.dims())
    }
}

/// Matrix multiplication compatible constraint
#[derive(Debug, Clone)]
pub struct MatMulCompatibleConstraint {
    other_shape: Shape,
    transpose_self: bool,
    transpose_other: bool,
}

impl MatMulCompatibleConstraint {
    pub fn new(other_shape: Shape, transpose_self: bool, transpose_other: bool) -> Self {
        Self {
            other_shape,
            transpose_self,
            transpose_other,
        }
    }
}

impl ShapeConstraint for MatMulCompatibleConstraint {
    fn validate(&self, shape: &Shape) -> Result<()> {
        // Delegate to the existing infer_matmul function for validation
        infer_matmul(
            shape,
            &self.other_shape,
            self.transpose_self,
            self.transpose_other,
        )
        .map(|_| ())
    }

    fn description(&self) -> String {
        format!(
            "matrix multiplication compatible with {:?} (transpose_self={}, transpose_other={})",
            self.other_shape.dims(),
            self.transpose_self,
            self.transpose_other
        )
    }
}

/// Shape validator that combines multiple constraints
pub struct ShapeValidator {
    constraints: Vec<Box<dyn ShapeConstraint + Send + Sync>>,
    operation_name: String,
}

impl ShapeValidator {
    pub fn new(operation_name: &str) -> Self {
        Self {
            constraints: Vec::new(),
            operation_name: operation_name.to_string(),
        }
    }

    /// Add a constraint to this validator
    pub fn add_constraint<C: ShapeConstraint + Send + Sync + 'static>(
        mut self,
        constraint: C,
    ) -> Self {
        self.constraints.push(Box::new(constraint));
        self
    }

    /// Validate a shape against all constraints
    pub fn validate(&self, shape: &Shape) -> Result<()> {
        for constraint in &self.constraints {
            constraint.validate(shape).map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Shape validation failed for operation '{}': {} (constraint: {})",
                    self.operation_name,
                    e,
                    constraint.description()
                ))
            })?;
        }
        Ok(())
    }

    /// Get a description of all constraints
    pub fn description(&self) -> String {
        let constraint_descriptions: Vec<String> =
            self.constraints.iter().map(|c| c.description()).collect();
        format!(
            "Operation '{}' requires: [{}]",
            self.operation_name,
            constraint_descriptions.join(", ")
        )
    }
}

/// Compile-time shape validation macro
#[macro_export]
macro_rules! validate_shapes {
    ($op_name:expr, $($shape:expr => $constraint:expr),* $(,)?) => {{
        let mut validator = $crate::ops::shape_inference::ShapeValidator::new($op_name);
        $(
            validator = validator.add_constraint($constraint);
            validator.validate($shape)?;
        )*
        Ok(())
    }};
}

/// Macro for creating rank constraints
#[macro_export]
macro_rules! rank {
    ($r:expr) => {
        $crate::ops::shape_inference::RankConstraint::new($r)
    };
}

/// Macro for creating minimum rank constraints
#[macro_export]
macro_rules! min_rank {
    ($r:expr) => {
        $crate::ops::shape_inference::MinRankConstraint::new($r)
    };
}

/// Macro for creating exact shape constraints
#[macro_export]
macro_rules! exact_shape {
    ($($dim:expr),*) => {
        $crate::ops::shape_inference::ExactShapeConstraint::new(
            $crate::Shape::from_slice(&[$($dim),*])
        )
    };
}

/// Common shape inference functions
/// Infer shape for element-wise binary operations with compile-time validation
pub fn infer_binary_elementwise(a: &Shape, b: &Shape) -> Result<Shape> {
    // Use the new validation system for better error messages
    let validator = ShapeValidator::new("binary_elementwise")
        .add_constraint(BroadcastableConstraint::new(a.clone()));

    validator.validate(b)?;

    a.broadcast_shape(b).ok_or_else(|| {
        TensorError::invalid_argument(format!(
            "Cannot broadcast shapes {:?} and {:?} for element-wise operation",
            a.dims(),
            b.dims()
        ))
    })
}

/// Validate and infer shape for element-wise binary operations with explicit constraints
pub fn infer_binary_elementwise_validated(a: &Shape, b: &Shape) -> Result<Shape> {
    // More detailed validation with constraint descriptions
    let broadcastable = BroadcastableConstraint::new(a.clone());
    let validator = ShapeValidator::new("binary_elementwise").add_constraint(broadcastable);

    // Validate inputs
    validator.validate(b).map_err(|e| {
        TensorError::invalid_argument(format!(
            "Binary elementwise operation validation failed: {e}"
        ))
    })?;

    // Perform inference
    infer_binary_elementwise(a, b)
}

/// Infer shape for matrix multiplication with compile-time validation
pub fn infer_matmul(a: &Shape, b: &Shape, transpose_a: bool, transpose_b: bool) -> Result<Shape> {
    // Pre-validate shapes using constraint system
    let min_rank_constraint = MinRankConstraint::new(2);
    let validator_a =
        ShapeValidator::new("matmul_input_a").add_constraint(min_rank_constraint.clone());
    let validator_b = ShapeValidator::new("matmul_input_b").add_constraint(min_rank_constraint);

    validator_a.validate(a)?;
    validator_b.validate(b)?;

    let a_dims = a.dims();
    let b_dims = b.dims();

    // Handle batch dimensions
    let batch_dims = if a.rank() > 2 || b.rank() > 2 {
        let a_batch = &a_dims[..a_dims.len() - 2];
        let b_batch = &b_dims[..b_dims.len() - 2];

        // Broadcast batch dimensions
        let mut result_batch = vec![];
        let max_batch_len = a_batch.len().max(b_batch.len());

        for i in 0..max_batch_len {
            let a_idx = a_batch.len() as i32 - max_batch_len as i32 + i as i32;
            let b_idx = b_batch.len() as i32 - max_batch_len as i32 + i as i32;

            let a_dim = if a_idx >= 0 {
                a_batch[a_idx as usize]
            } else {
                1
            };
            let b_dim = if b_idx >= 0 {
                b_batch[b_idx as usize]
            } else {
                1
            };

            if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
                return Err(TensorError::invalid_argument(format!(
                    "Incompatible batch dimensions: {} vs {a_dim} and {b_dim}",
                    a.dims()[i]
                )));
            }

            result_batch.push(a_dim.max(b_dim));
        }

        result_batch
    } else {
        vec![]
    };

    // Get matrix dimensions
    let (m, k1) = if transpose_a {
        (a_dims[a_dims.len() - 1], a_dims[a_dims.len() - 2])
    } else {
        (a_dims[a_dims.len() - 2], a_dims[a_dims.len() - 1])
    };

    let (k2, n) = if transpose_b {
        (b_dims[b_dims.len() - 1], b_dims[b_dims.len() - 2])
    } else {
        (b_dims[b_dims.len() - 2], b_dims[b_dims.len() - 1])
    };

    if k1 != k2 {
        return Err(TensorError::invalid_argument(format!(
            "Incompatible matrix dimensions for multiplication: ({m}, {k1}) x ({k2}, {n})"
        )));
    }

    // Construct result shape
    let mut result_shape = batch_dims;
    result_shape.push(m);
    result_shape.push(n);

    Ok(Shape::from_slice(&result_shape))
}

/// Infer shape for reduction operations
pub fn infer_reduction(input: &Shape, axes: Option<&[i32]>, keep_dims: bool) -> Result<Shape> {
    let rank = input.rank() as i32;
    let dims = input.dims();

    // Normalize axes
    let axes = if let Some(axes) = axes {
        axes.iter()
            .map(|&axis| {
                let normalized = if axis < 0 { rank + axis } else { axis };
                if normalized < 0 || normalized >= rank {
                    Err(TensorError::invalid_argument(format!(
                        "Axis {axis} is out of range for tensor of rank {rank}"
                    )))
                } else {
                    Ok(normalized as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        // Reduce all axes
        (0..rank as usize).collect()
    };

    // Compute result shape
    let mut result_dims = vec![];
    for (i, &dim) in dims.iter().enumerate() {
        if axes.contains(&i) {
            if keep_dims {
                result_dims.push(1);
            }
        } else {
            result_dims.push(dim);
        }
    }

    // Handle scalar result
    if result_dims.is_empty() && !keep_dims {
        result_dims = vec![]; // Scalar
    }

    Ok(Shape::from_slice(&result_dims))
}

/// Infer shape for reshape operation
pub fn infer_reshape(input: &Shape, target_shape: &[i64]) -> Result<Shape> {
    let input_size = input.size();
    let mut inferred_shape = vec![];
    let mut negative_idx = None;
    let mut known_size = 1usize;

    // First pass: identify -1 and compute known size
    for (i, &dim) in target_shape.iter().enumerate() {
        if dim == -1 {
            if negative_idx.is_some() {
                return Err(TensorError::invalid_argument(
                    "Only one dimension can be -1 in reshape ".to_string(),
                ));
            }
            negative_idx = Some(i);
            inferred_shape.push(0); // Placeholder
        } else if dim < 0 {
            return Err(TensorError::invalid_argument(format!(
                "Invalid dimension {dim} in reshape "
            )));
        } else {
            let dim = dim as usize;
            known_size *= dim;
            inferred_shape.push(dim);
        }
    }

    // Infer the -1 dimension if present
    if let Some(idx) = negative_idx {
        if known_size == 0 || input_size % known_size != 0 {
            return Err(TensorError::invalid_argument(format!(
                "Cannot reshape tensor of size {input_size} to shape {target_shape:?}"
            )));
        }
        inferred_shape[idx] = input_size / known_size;
    } else {
        // Verify the total size matches
        if known_size != input_size {
            return Err(TensorError::invalid_argument(format!(
                "Cannot reshape tensor of size {input_size} to size {known_size}"
            )));
        }
    }

    Ok(Shape::from_slice(&inferred_shape))
}

/// Infer shape for concatenation
pub fn infer_concat(inputs: &[&Shape], axis: i32) -> Result<Shape> {
    if inputs.is_empty() {
        return Err(TensorError::invalid_argument(
            "Concat requires at least one input ".to_string(),
        ));
    }

    let first = inputs[0];
    let rank = first.rank() as i32;

    // Normalize axis
    let axis = if axis < 0 { rank + axis } else { axis };
    if axis < 0 || axis >= rank {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} is out of range for tensor of rank {rank}"
        )));
    }
    let axis = axis as usize;

    // Check compatibility and compute result shape
    let mut result_dims = first.dims().to_vec();
    result_dims[axis] = 0;

    for &shape in inputs {
        if shape.rank() != first.rank() {
            return Err(TensorError::invalid_argument(format!(
                "All inputs must have the same rank, got {} and {}",
                shape.rank(),
                first.rank()
            )));
        }

        let dims = shape.dims();

        // Check non-concat dimensions for compatibility
        for i in 0..dims.len() {
            if i == axis {
                result_dims[i] += dims[i];
            } else if result_dims[i] != dims[i] {
                return Err(TensorError::invalid_argument(
                    format!("Incompatible shapes for concatenation along axis {axis}: dimension {i} differs ({} vs {})", result_dims[i], dims[i])
                ));
            }
        }
    }

    Ok(Shape::from_slice(&result_dims))
}

/// Infer shape for convolution operations with compile-time validation
pub fn infer_conv2d(
    input: &Shape,
    filter: &Shape,
    strides: &[usize],
    padding: &str,
    dilations: Option<&[usize]>,
) -> Result<Shape> {
    // Pre-validate shapes using constraint system
    let rank4_constraint = RankConstraint::new(4);
    let input_validator =
        ShapeValidator::new("conv2d_input").add_constraint(rank4_constraint.clone());
    let filter_validator = ShapeValidator::new("conv2d_filter").add_constraint(rank4_constraint);

    input_validator.validate(input)?;
    filter_validator.validate(filter)?;

    let input_dims = input.dims();
    let filter_dims = filter.dims();
    let dilations = dilations.unwrap_or(&[1, 1]);

    let batch = input_dims[0];
    let in_height = input_dims[1];
    let in_width = input_dims[2];
    let in_channels = input_dims[3];

    let filter_height = filter_dims[0];
    let filter_width = filter_dims[1];
    let filter_in_channels = filter_dims[2];
    let out_channels = filter_dims[3];

    if in_channels != filter_in_channels {
        return Err(TensorError::invalid_argument(
            format!("Input channels {in_channels} does not match filter input channels {filter_in_channels}")
        ));
    }

    // Compute effective filter size with dilation
    let effective_filter_height = (filter_height - 1) * dilations[0] + 1;
    let effective_filter_width = (filter_width - 1) * dilations[1] + 1;

    // Compute output spatial dimensions
    let (out_height, out_width) = match padding {
        "VALID" => {
            let h = (in_height - effective_filter_height) / strides[0] + 1;
            let w = (in_width - effective_filter_width) / strides[1] + 1;
            (h, w)
        }
        "SAME" => {
            let h = (in_height + strides[0] - 1) / strides[0];
            let w = (in_width + strides[1] - 1) / strides[1];
            (h, w)
        }
        _ => {
            return Err(TensorError::invalid_argument(format!(
                "Unknown padding type: {padding:?}"
            )));
        }
    };

    Ok(Shape::from_slice(&[
        batch,
        out_height,
        out_width,
        out_channels,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_elementwise_shapes() {
        let a = Shape::from_slice(&[2, 3, 4]);
        let b = Shape::from_slice(&[1, 3, 4]);
        let result = infer_binary_elementwise(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);

        let a = Shape::from_slice(&[5, 1, 4]);
        let b = Shape::from_slice(&[1, 3, 1]);
        let result = infer_binary_elementwise(&a, &b).unwrap();
        assert_eq!(result.dims(), &[5, 3, 4]);
    }

    #[test]
    fn test_matmul_shapes() {
        // Simple 2D
        let a = Shape::from_slice(&[3, 4]);
        let b = Shape::from_slice(&[4, 5]);
        let result = infer_matmul(&a, &b, false, false).unwrap();
        assert_eq!(result.dims(), &[3, 5]);

        // With transpose
        let a = Shape::from_slice(&[4, 3]);
        let b = Shape::from_slice(&[4, 5]);
        let result = infer_matmul(&a, &b, true, false).unwrap();
        assert_eq!(result.dims(), &[3, 5]);

        // Batch matmul
        let a = Shape::from_slice(&[2, 3, 4]);
        let b = Shape::from_slice(&[2, 4, 5]);
        let result = infer_matmul(&a, &b, false, false).unwrap();
        assert_eq!(result.dims(), &[2, 3, 5]);

        // Broadcast batch
        let a = Shape::from_slice(&[1, 3, 4]);
        let b = Shape::from_slice(&[2, 4, 5]);
        let result = infer_matmul(&a, &b, false, false).unwrap();
        assert_eq!(result.dims(), &[2, 3, 5]);
    }

    #[test]
    fn test_reduction_shapes() {
        let input = Shape::from_slice(&[2, 3, 4]);

        // Reduce all
        let result = infer_reduction(&input, None, false).unwrap();
        assert_eq!(result.dims(), &[] as &[usize]);

        // Reduce specific axes
        let result = infer_reduction(&input, Some(&[1]), false).unwrap();
        assert_eq!(result.dims(), &[2, 4]);

        // Keep dims
        let result = infer_reduction(&input, Some(&[1, 2]), true).unwrap();
        assert_eq!(result.dims(), &[2, 1, 1]);

        // Negative axes
        let result = infer_reduction(&input, Some(&[-1]), false).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_reshape_inference() {
        let input = Shape::from_slice(&[2, 3, 4]);

        // Simple reshape
        let result = infer_reshape(&input, &[6, 4]).unwrap();
        assert_eq!(result.dims(), &[6, 4]);

        // With -1
        let result = infer_reshape(&input, &[-1, 4]).unwrap();
        assert_eq!(result.dims(), &[6, 4]);

        let result = infer_reshape(&input, &[2, -1]).unwrap();
        assert_eq!(result.dims(), &[2, 12]);
    }

    #[test]
    fn test_compile_time_shape_validation() {
        let shape_2d = Shape::from_slice(&[3, 4]);
        let shape_3d = Shape::from_slice(&[2, 3, 4]);
        let shape_4d = Shape::from_slice(&[1, 2, 3, 4]);

        // Test rank constraints
        let rank2_constraint = RankConstraint::new(2);
        assert!(rank2_constraint.validate(&shape_2d).is_ok());
        assert!(rank2_constraint.validate(&shape_3d).is_err());

        // Test minimum rank constraints
        let min_rank2_constraint = MinRankConstraint::new(2);
        assert!(min_rank2_constraint.validate(&shape_2d).is_ok());
        assert!(min_rank2_constraint.validate(&shape_3d).is_ok());
        assert!(min_rank2_constraint.validate(&shape_4d).is_ok());

        // Test exact shape constraints
        let exact_shape_constraint = ExactShapeConstraint::new(shape_2d.clone());
        assert!(exact_shape_constraint.validate(&shape_2d).is_ok());
        assert!(exact_shape_constraint.validate(&shape_3d).is_err());

        // Test broadcastable constraints
        let broadcast_constraint = BroadcastableConstraint::new(Shape::from_slice(&[1, 4]));
        assert!(broadcast_constraint.validate(&shape_2d).is_ok()); // [3, 4] is broadcastable with [1, 4]

        let incompatible_shape = Shape::from_slice(&[3, 5]);
        assert!(broadcast_constraint.validate(&incompatible_shape).is_err());
    }

    #[test]
    fn test_shape_validator_multiple_constraints() {
        let shape = Shape::from_slice(&[2, 3, 4]);

        let validator = ShapeValidator::new("test_operation")
            .add_constraint(RankConstraint::new(3))
            .add_constraint(MinRankConstraint::new(2));

        assert!(validator.validate(&shape).is_ok());

        // Test with incompatible shape
        let incompatible_shape = Shape::from_slice(&[3, 4]);
        let result = validator.validate(&incompatible_shape);
        assert!(result.is_err());

        // Check error message contains operation name
        if let Err(e) = result {
            assert!(e.to_string().contains("test_operation"));
        }
    }

    #[test]
    fn test_matmul_compatibility_constraint() {
        let a_shape = Shape::from_slice(&[3, 4]);
        let b_shape = Shape::from_slice(&[4, 5]);
        let incompatible_shape = Shape::from_slice(&[3, 5]);

        let matmul_constraint = MatMulCompatibleConstraint::new(b_shape.clone(), false, false);

        // Compatible shapes
        assert!(matmul_constraint.validate(&a_shape).is_ok());

        // Incompatible shapes
        assert!(matmul_constraint.validate(&incompatible_shape).is_err());

        // Test with transpose
        let transpose_constraint =
            MatMulCompatibleConstraint::new(Shape::from_slice(&[5, 4]), false, true);
        assert!(transpose_constraint.validate(&a_shape).is_ok());
    }

    #[test]
    fn test_enhanced_binary_elementwise_validation() {
        let a = Shape::from_slice(&[2, 3, 4]);
        let b_compatible = Shape::from_slice(&[1, 3, 4]);
        let b_incompatible = Shape::from_slice(&[2, 2, 4]);

        // Test new validated function
        assert!(infer_binary_elementwise_validated(&a, &b_compatible).is_ok());

        let result = infer_binary_elementwise_validated(&a, &b_incompatible);
        assert!(result.is_err());

        // Check that error message is descriptive
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("Binary elementwise operation validation failed"));
        }
    }

    #[test]
    fn test_constraint_descriptions() {
        let rank_constraint = RankConstraint::new(3);
        assert_eq!(rank_constraint.description(), "rank = 3");

        let min_rank_constraint = MinRankConstraint::new(2);
        assert_eq!(min_rank_constraint.description(), "rank >= 2");

        let exact_shape_constraint = ExactShapeConstraint::new(Shape::from_slice(&[2, 3]));
        assert_eq!(exact_shape_constraint.description(), "exact shape = [2, 3]");

        let broadcast_constraint = BroadcastableConstraint::new(Shape::from_slice(&[1, 3]));
        assert_eq!(
            broadcast_constraint.description(),
            "broadcastable with [1, 3]"
        );
    }

    #[test]
    fn test_validator_description() {
        let validator = ShapeValidator::new("complex_operation")
            .add_constraint(RankConstraint::new(4))
            .add_constraint(MinRankConstraint::new(2));

        let description = validator.description();
        assert!(description.contains("complex_operation"));
        assert!(description.contains("rank = 4"));
        assert!(description.contains("rank >= 2"));
    }
}
