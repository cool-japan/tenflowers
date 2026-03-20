//! ShapeInferenceRegistry: the central registry for shape inference rules.

use super::infer_fns::{
    infer_add, infer_comparison, infer_concat, infer_div, infer_dot, infer_logical,
    infer_matmul_op, infer_mul, infer_permute, infer_pow, infer_reduction, infer_reshape,
    infer_squeeze, infer_stack, infer_sub, infer_transpose, infer_unary, infer_unsqueeze,
};
use super::types::{OperationCategory, OperationMetadata, RegisteredOperation, ShapeInferenceFn};
use crate::{Result, Shape, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global shape inference registry
pub struct ShapeInferenceRegistry {
    pub(super) operations: Arc<RwLock<HashMap<String, RegisteredOperation>>>,
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
        let mut ops = self
            .operations
            .write()
            .expect("write lock should not be poisoned");

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
        let ops = self
            .operations
            .read()
            .expect("read lock should not be poisoned");

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
        let ops = self
            .operations
            .read()
            .expect("read lock should not be poisoned");
        let mut names: Vec<String> = ops.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get operations by category
    pub fn operations_by_category(&self, category: OperationCategory) -> Vec<String> {
        let ops = self
            .operations
            .read()
            .expect("read lock should not be poisoned");
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
