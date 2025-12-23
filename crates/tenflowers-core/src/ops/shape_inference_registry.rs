use crate::ops::shape_inference::{
    infer_binary_elementwise, infer_matmul, BroadcastableConstraint, MinRankConstraint,
    RankConstraint, ShapeConstraint, ShapeValidator,
};
/// Shape Inference Registry for TenfloweRS
///
/// This module provides a centralized registry for shape inference rules across
/// all tensor operations, ensuring consistent shape validation and error reporting.
///
/// ## Design Goals
/// - **Centralization**: Single source of truth for shape inference logic
/// - **Standardization**: Consistent error messages using ShapeErrorTaxonomy
/// - **Discoverability**: Easy to find shape inference rules for any operation
/// - **Type Safety**: Compile-time guarantees where possible
///
/// ## Usage
///
/// ```rust,ignore
/// use tenflowers_core::ops::shape_inference_registry::{ShapeInferenceRegistry, get_registry};
///
/// // Get the global registry
/// let registry = get_registry();
///
/// // Infer output shape for an operation
/// let output_shape = registry.infer("add", &[input1.shape(), input2.shape()])?;
///
/// // Validate inputs for an operation
/// registry.validate("matmul", &[a.shape(), b.shape()], &metadata)?;
/// ```
use crate::shape_error_taxonomy::{ShapeErrorBuilder, ShapeErrorCategory, ShapeErrorUtils};
use crate::{Result, Shape, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Metadata for an operation (e.g., axis, keepdims, transpose flags)
pub type OperationMetadata = HashMap<String, MetadataValue>;

/// Value types for operation metadata
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Bool(bool),
    Int(i64),
    UInt(usize),
    IntVec(Vec<i64>),
    UIntVec(Vec<usize>),
    String(String),
}

impl MetadataValue {
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_uint(&self) -> Option<usize> {
        if let Self::UInt(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    pub fn as_int_vec(&self) -> Option<&Vec<i64>> {
        if let Self::IntVec(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_uint_vec(&self) -> Option<&Vec<usize>> {
        if let Self::UIntVec(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

/// Shape inference function signature
pub type ShapeInferenceFn = fn(&[Shape], &OperationMetadata) -> Result<Shape>;

/// Operation category for organizing inference rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Element-wise binary operations (add, sub, mul, div)
    BinaryElementwise,
    /// Element-wise unary operations (abs, exp, log, sin, cos)
    UnaryElementwise,
    /// Matrix operations (matmul, dot, outer)
    MatrixOps,
    /// Reduction operations (sum, mean, max, min)
    Reduction,
    /// Manipulation operations (reshape, transpose, permute)
    Manipulation,
    /// Convolution operations
    Convolution,
    /// Pooling operations
    Pooling,
    /// Concatenation and stacking
    Concatenation,
    /// Padding operations
    Padding,
    /// Indexing and slicing
    Indexing,
    /// Comparison operations
    Comparison,
    /// Logical operations
    Logical,
    /// Other operations
    Other,
}

/// Registered operation with shape inference rules
struct RegisteredOperation {
    name: String,
    category: OperationCategory,
    inference_fn: ShapeInferenceFn,
    description: String,
}

/// Global shape inference registry
pub struct ShapeInferenceRegistry {
    operations: Arc<RwLock<HashMap<String, RegisteredOperation>>>,
}

impl Default for ShapeInferenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeInferenceRegistry {
    /// Create a new shape inference registry
    pub fn new() -> Self {
        let registry = Self {
            operations: Arc::new(RwLock::new(HashMap::new())),
        };
        registry.register_builtin_operations();
        registry
    }

    /// Register an operation's shape inference rule
    pub fn register(
        &self,
        name: &str,
        category: OperationCategory,
        inference_fn: ShapeInferenceFn,
        description: &str,
    ) -> Result<()> {
        let mut ops = self.operations.write().unwrap();

        if ops.contains_key(name) {
            return Err(TensorError::invalid_argument(format!(
                "Operation '{}' already registered in shape inference registry",
                name
            )));
        }

        ops.insert(
            name.to_string(),
            RegisteredOperation {
                name: name.to_string(),
                category,
                inference_fn,
                description: description.to_string(),
            },
        );

        Ok(())
    }

    /// Infer output shape for an operation
    pub fn infer(
        &self,
        operation: &str,
        inputs: &[Shape],
        metadata: &OperationMetadata,
    ) -> Result<Shape> {
        let ops = self.operations.read().unwrap();

        let op = ops.get(operation).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Operation '{}' not found in shape inference registry. Available operations: {}",
                operation,
                self.list_operations().join(", ")
            ))
        })?;

        (op.inference_fn)(inputs, metadata)
    }

    /// Validate inputs for an operation (convenience method)
    pub fn validate(
        &self,
        operation: &str,
        inputs: &[Shape],
        metadata: &OperationMetadata,
    ) -> Result<()> {
        // Validation happens during inference
        self.infer(operation, inputs, metadata).map(|_| ())
    }

    /// List all registered operations
    pub fn list_operations(&self) -> Vec<String> {
        let ops = self.operations.read().unwrap();
        let mut names: Vec<String> = ops.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get operations by category
    pub fn operations_by_category(&self, category: OperationCategory) -> Vec<String> {
        let ops = self.operations.read().unwrap();
        let mut names: Vec<String> = ops
            .values()
            .filter(|op| op.category == category)
            .map(|op| op.name.clone())
            .collect();
        names.sort();
        names
    }

    /// Register all built-in operations
    fn register_builtin_operations(&self) {
        // Binary elementwise operations
        let _ = self.register(
            "add",
            OperationCategory::BinaryElementwise,
            infer_add,
            "Element-wise addition",
        );
        let _ = self.register(
            "sub",
            OperationCategory::BinaryElementwise,
            infer_sub,
            "Element-wise subtraction",
        );
        let _ = self.register(
            "mul",
            OperationCategory::BinaryElementwise,
            infer_mul,
            "Element-wise multiplication",
        );
        let _ = self.register(
            "div",
            OperationCategory::BinaryElementwise,
            infer_div,
            "Element-wise division",
        );
        let _ = self.register(
            "pow",
            OperationCategory::BinaryElementwise,
            infer_pow,
            "Element-wise power",
        );

        // Unary elementwise operations
        let _ = self.register(
            "neg",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise negation",
        );
        let _ = self.register(
            "abs",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise absolute value",
        );
        let _ = self.register(
            "exp",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise exponential",
        );
        let _ = self.register(
            "log",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise natural logarithm",
        );
        let _ = self.register(
            "sqrt",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise square root",
        );
        let _ = self.register(
            "sin",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise sine",
        );
        let _ = self.register(
            "cos",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise cosine",
        );
        let _ = self.register(
            "tan",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise tangent",
        );
        let _ = self.register(
            "tanh",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Element-wise hyperbolic tangent",
        );
        let _ = self.register(
            "relu",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Rectified Linear Unit",
        );
        let _ = self.register(
            "sigmoid",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "Sigmoid activation",
        );
        let _ = self.register(
            "gelu",
            OperationCategory::UnaryElementwise,
            infer_unary,
            "GELU activation",
        );

        // Matrix operations
        let _ = self.register(
            "matmul",
            OperationCategory::MatrixOps,
            infer_matmul_op,
            "Matrix multiplication",
        );
        let _ = self.register(
            "dot",
            OperationCategory::MatrixOps,
            infer_dot,
            "Dot product",
        );

        // Reduction operations
        let _ = self.register(
            "sum",
            OperationCategory::Reduction,
            infer_reduction,
            "Sum reduction",
        );
        let _ = self.register(
            "mean",
            OperationCategory::Reduction,
            infer_reduction,
            "Mean reduction",
        );
        let _ = self.register(
            "max",
            OperationCategory::Reduction,
            infer_reduction,
            "Max reduction",
        );
        let _ = self.register(
            "min",
            OperationCategory::Reduction,
            infer_reduction,
            "Min reduction",
        );
        let _ = self.register(
            "prod",
            OperationCategory::Reduction,
            infer_reduction,
            "Product reduction",
        );

        // Manipulation operations
        let _ = self.register(
            "reshape",
            OperationCategory::Manipulation,
            infer_reshape,
            "Reshape tensor",
        );
        let _ = self.register(
            "transpose",
            OperationCategory::Manipulation,
            infer_transpose,
            "Transpose tensor",
        );
        let _ = self.register(
            "permute",
            OperationCategory::Manipulation,
            infer_permute,
            "Permute dimensions",
        );
        let _ = self.register(
            "squeeze",
            OperationCategory::Manipulation,
            infer_squeeze,
            "Remove dimensions of size 1",
        );
        let _ = self.register(
            "unsqueeze",
            OperationCategory::Manipulation,
            infer_unsqueeze,
            "Add dimension of size 1",
        );

        // Concatenation
        let _ = self.register(
            "concat",
            OperationCategory::Concatenation,
            infer_concat,
            "Concatenate tensors",
        );
        let _ = self.register(
            "stack",
            OperationCategory::Concatenation,
            infer_stack,
            "Stack tensors",
        );

        // Comparison operations
        let _ = self.register(
            "eq",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise equality",
        );
        let _ = self.register(
            "ne",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise not-equal",
        );
        let _ = self.register(
            "gt",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise greater-than",
        );
        let _ = self.register(
            "ge",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise greater-or-equal",
        );
        let _ = self.register(
            "lt",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise less-than",
        );
        let _ = self.register(
            "le",
            OperationCategory::Comparison,
            infer_comparison,
            "Element-wise less-or-equal",
        );

        // Logical operations
        let _ = self.register(
            "and",
            OperationCategory::Logical,
            infer_logical,
            "Element-wise logical AND",
        );
        let _ = self.register(
            "or",
            OperationCategory::Logical,
            infer_logical,
            "Element-wise logical OR",
        );
        let _ = self.register(
            "not",
            OperationCategory::Logical,
            infer_unary,
            "Element-wise logical NOT",
        );
        let _ = self.register(
            "xor",
            OperationCategory::Logical,
            infer_logical,
            "Element-wise logical XOR",
        );
    }
}

// ============================================================================
// Shape Inference Functions
// ============================================================================

/// Infer shape for binary elementwise operations (add, sub, mul, div, pow)
fn infer_add(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

fn infer_sub(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

fn infer_mul(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

fn infer_div(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

fn infer_pow(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

/// Infer shape for unary elementwise operations
fn infer_unary(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 1 {
        return Err(TensorError::invalid_argument(format!(
            "Unary operation expects exactly 1 input, got {}",
            inputs.len()
        )));
    }
    Ok(inputs[0].clone())
}

/// Infer shape for matrix multiplication
fn infer_matmul_op(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_dot(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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

/// Infer shape for reduction operations
fn infer_reduction(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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

/// Infer shape for reshape operation
fn infer_reshape(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_transpose(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_permute(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_squeeze(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_unsqueeze(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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

/// Infer shape for concatenation operation
fn infer_concat(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_stack(inputs: &[Shape], metadata: &OperationMetadata) -> Result<Shape> {
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

/// Infer shape for comparison operations
fn infer_comparison(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
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
fn infer_logical(inputs: &[Shape], _metadata: &OperationMetadata) -> Result<Shape> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "Logical operation expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    }
    // Logical operations broadcast like binary elementwise
    infer_binary_elementwise(&inputs[0], &inputs[1])
}

// ============================================================================
// Global Registry Access
// ============================================================================

use std::sync::OnceLock;

static GLOBAL_REGISTRY: OnceLock<ShapeInferenceRegistry> = OnceLock::new();

/// Get the global shape inference registry
pub fn get_registry() -> &'static ShapeInferenceRegistry {
    GLOBAL_REGISTRY.get_or_init(ShapeInferenceRegistry::new)
}

/// Initialize the global registry (called automatically on first access)
pub fn initialize_registry() {
    let _ = get_registry();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ShapeInferenceRegistry::new();
        let ops = registry.list_operations();
        assert!(!ops.is_empty(), "Registry should have builtin operations");
        assert!(ops.contains(&"add".to_string()));
        assert!(ops.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_binary_elementwise_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("add", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcasting_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 1, 3]);
        let shape2 = Shape::from_slice(&[1, 4, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("mul", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_matmul_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("transpose_a".to_string(), MetadataValue::Bool(false));
        metadata.insert("transpose_b".to_string(), MetadataValue::Bool(false));

        let result = registry.infer("matmul", &[shape1, shape2], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 4]);
    }

    #[test]
    fn test_reduction_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);

        // Reduce on axis 1, no keepdims
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(1));
        metadata.insert("keepdims".to_string(), MetadataValue::Bool(false));

        let result = registry.infer("sum", &[shape.clone()], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 4]);

        // Reduce on axis 1, with keepdims
        metadata.insert("keepdims".to_string(), MetadataValue::Bool(true));
        let result = registry.infer("sum", &[shape.clone()], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 1, 4]);

        // Reduce all dimensions
        let metadata_all = HashMap::new();
        let result = registry.infer("mean", &[shape], &metadata_all);
        assert!(result.is_ok());
        let result_shape = result.unwrap();
        assert_eq!(result_shape.dims().len(), 0); // Scalar shape has no dimensions
    }

    #[test]
    fn test_reshape_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), MetadataValue::IntVec(vec![6, 4]));

        let result = registry.infer("reshape", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[6, 4]);
    }

    #[test]
    fn test_reshape_with_infer() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), MetadataValue::IntVec(vec![-1, 4]));

        let result = registry.infer("reshape", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[6, 4]);
    }

    #[test]
    fn test_transpose_inference() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3, 4]);
        let metadata = HashMap::new();

        let result = registry.infer("transpose", &[shape], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_concat_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 4]);
        let shape3 = Shape::from_slice(&[2, 5]);
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(1));

        let result = registry.infer("concat", &[shape1, shape2, shape3], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[2, 12]);
    }

    #[test]
    fn test_stack_inference() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[2, 3]);
        let shape3 = Shape::from_slice(&[2, 3]);
        let mut metadata = HashMap::new();
        metadata.insert("axis".to_string(), MetadataValue::Int(0));

        let result = registry.infer("stack", &[shape1, shape2, shape3], &metadata);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dims(), &[3, 2, 3]);
    }

    #[test]
    fn test_operations_by_category() {
        let registry = get_registry();
        let binary_ops = registry.operations_by_category(OperationCategory::BinaryElementwise);
        assert!(binary_ops.contains(&"add".to_string()));
        assert!(binary_ops.contains(&"mul".to_string()));

        let matrix_ops = registry.operations_by_category(OperationCategory::MatrixOps);
        assert!(matrix_ops.contains(&"matmul".to_string()));
    }

    #[test]
    fn test_error_for_unknown_operation() {
        let registry = get_registry();
        let shape = Shape::from_slice(&[2, 3]);
        let metadata = HashMap::new();

        let result = registry.infer("unknown_op", &[shape], &metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_standardization() {
        let registry = get_registry();
        let shape1 = Shape::from_slice(&[2, 3]);
        let shape2 = Shape::from_slice(&[4, 5]);
        let metadata = HashMap::new();

        // Should produce standardized error message
        let result = registry.infer("add", &[shape1, shape2], &metadata);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{}", err);
        // Error should mention broadcasting
        assert!(err_msg.contains("broadcast") || err_msg.contains("Broadcast"));
    }
}
