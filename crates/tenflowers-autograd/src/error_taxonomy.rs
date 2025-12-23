//! # Error Taxonomy Alignment for Autograd
//!
//! This module provides utilities and guidelines for aligning autograd error handling
//! with the tenflowers-core error taxonomy, ensuring consistent error reporting and
//! recovery strategies across the framework.
//!
//! ## Error Handling Principles
//!
//! 1. **Contextual Information**: All errors should include operation context
//! 2. **Actionable Messages**: Error messages should suggest solutions
//! 3. **Consistent Patterns**: Use core's TensorError variants consistently
//! 4. **Gradient-Specific Context**: Include gradient computation context
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use tenflowers_autograd::error_taxonomy::{AutogradErrorBuilder, GradientContext};
//! use tenflowers_core::{Tensor, TensorError};
//!
//! # fn gradient_computation() -> Result<(), TensorError> {
//! // Build gradient-specific errors
//! let error = AutogradErrorBuilder::new("backward_pass")
//!     .with_gradient_context(GradientContext {
//!         tape_id: 123,
//!         operation_index: 5,
//!         num_inputs: 2,
//!         num_outputs: 1,
//!     })
//!     .shape_mismatch("expected [10, 5]", "got [10, 4]");
//!
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use tenflowers_core::{DType, Device, TensorError};

/// Gradient computation context for error reporting
#[derive(Debug, Clone, Default)]
pub struct GradientContext {
    /// Gradient tape identifier
    pub tape_id: u64,
    /// Operation index in the tape
    pub operation_index: usize,
    /// Number of input tensors
    pub num_inputs: usize,
    /// Number of output tensors
    pub num_outputs: usize,
    /// Whether this is a higher-order gradient
    pub is_higher_order: bool,
    /// Parent operation name (for nested gradients)
    pub parent_operation: Option<String>,
}

/// Error builder for autograd operations aligned with core taxonomy
pub struct AutogradErrorBuilder {
    operation: String,
    gradient_context: Option<GradientContext>,
    input_shapes: Vec<Vec<usize>>,
    input_devices: Vec<Device>,
    input_dtypes: Vec<DType>,
    metadata: HashMap<String, String>,
}

impl AutogradErrorBuilder {
    /// Create a new error builder for an operation
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            gradient_context: None,
            input_shapes: Vec::new(),
            input_devices: Vec::new(),
            input_dtypes: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add gradient computation context
    pub fn with_gradient_context(mut self, context: GradientContext) -> Self {
        self.gradient_context = Some(context);
        self
    }

    /// Add input tensor information
    pub fn with_inputs(
        mut self,
        shapes: Vec<Vec<usize>>,
        devices: Vec<Device>,
        dtypes: Vec<DType>,
    ) -> Self {
        self.input_shapes = shapes;
        self.input_devices = devices;
        self.input_dtypes = dtypes;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a shape mismatch error
    pub fn shape_mismatch(
        self,
        expected: impl Into<String>,
        got: impl Into<String>,
    ) -> TensorError {
        let mut context = self.build_error_context();

        // Add gradient-specific metadata
        if let Some(grad_ctx) = self.gradient_context {
            context
                .metadata
                .insert("tape_id".to_string(), grad_ctx.tape_id.to_string());
            context.metadata.insert(
                "operation_index".to_string(),
                grad_ctx.operation_index.to_string(),
            );
        }

        TensorError::ShapeMismatch {
            operation: self.operation,
            expected: expected.into(),
            got: got.into(),
            context: Some(context),
        }
    }

    /// Build a gradient not enabled error
    pub fn gradient_not_enabled(self, suggestion: impl Into<String>) -> TensorError {
        let context = self.build_error_context();

        TensorError::GradientNotEnabled {
            operation: self.operation,
            suggestion: suggestion.into(),
            context: Some(context),
        }
    }

    /// Build an invalid operation error
    pub fn invalid_operation(self, reason: impl Into<String>) -> TensorError {
        let context = self.build_error_context();

        TensorError::InvalidOperation {
            operation: self.operation,
            reason: reason.into(),
            context: Some(context),
        }
    }

    /// Build a numerical error
    pub fn numerical_error(
        self,
        details: impl Into<String>,
        suggestions: Vec<String>,
    ) -> TensorError {
        let context = self.build_error_context();

        TensorError::NumericalError {
            operation: self.operation,
            details: details.into(),
            suggestions,
            context: Some(context),
        }
    }

    /// Build a device mismatch error
    pub fn device_mismatch(
        self,
        device1: impl Into<String>,
        device2: impl Into<String>,
    ) -> TensorError {
        let context = self.build_error_context();

        TensorError::DeviceMismatch {
            operation: self.operation,
            device1: device1.into(),
            device2: device2.into(),
            context: Some(context),
        }
    }

    /// Build an unsupported operation error
    pub fn unsupported_operation(
        self,
        reason: impl Into<String>,
        alternatives: Vec<String>,
    ) -> TensorError {
        let context = self.build_error_context();

        TensorError::UnsupportedOperation {
            operation: self.operation,
            reason: reason.into(),
            alternatives,
            context: Some(context),
        }
    }

    /// Build a compute error
    pub fn compute_error(self, details: impl Into<String>, retry_possible: bool) -> TensorError {
        let context = self.build_error_context();

        TensorError::ComputeError {
            operation: self.operation,
            details: details.into(),
            retry_possible,
            context: Some(context),
        }
    }

    // Helper method to build ErrorContext
    fn build_error_context(&self) -> tenflowers_core::error::ErrorContext {
        let mut metadata = self.metadata.clone();

        // Add gradient context if available
        if let Some(ref grad_ctx) = self.gradient_context {
            metadata.insert("gradient_tape_id".to_string(), grad_ctx.tape_id.to_string());
            metadata.insert(
                "gradient_operation_index".to_string(),
                grad_ctx.operation_index.to_string(),
            );
            metadata.insert(
                "num_gradient_inputs".to_string(),
                grad_ctx.num_inputs.to_string(),
            );
            metadata.insert(
                "num_gradient_outputs".to_string(),
                grad_ctx.num_outputs.to_string(),
            );
            metadata.insert(
                "is_higher_order_gradient".to_string(),
                grad_ctx.is_higher_order.to_string(),
            );

            if let Some(ref parent) = grad_ctx.parent_operation {
                metadata.insert("parent_operation".to_string(), parent.clone());
            }
        }

        tenflowers_core::error::ErrorContext {
            input_shapes: self.input_shapes.clone(),
            input_devices: self.input_devices.clone(),
            input_dtypes: self.input_dtypes.clone(),
            output_shape: None,
            thread_id: format!("{:?}", std::thread::current().id()),
            stack_trace: None,
            metadata,
        }
    }
}

/// Utility functions for common autograd error patterns
pub mod utils {
    use super::*;

    /// Create a gradient shape mismatch error
    pub fn gradient_shape_mismatch(
        operation: &str,
        expected_shape: &[usize],
        actual_shape: &[usize],
        tape_id: u64,
        op_index: usize,
    ) -> TensorError {
        AutogradErrorBuilder::new(operation)
            .with_gradient_context(GradientContext {
                tape_id,
                operation_index: op_index,
                num_inputs: 0,
                num_outputs: 0,
                is_higher_order: false,
                parent_operation: None,
            })
            .shape_mismatch(
                format!("{:?}", expected_shape),
                format!("{:?}", actual_shape),
            )
    }

    /// Create a gradient computation failure error
    pub fn gradient_computation_failed(operation: &str, reason: &str, tape_id: u64) -> TensorError {
        AutogradErrorBuilder::new(operation)
            .with_gradient_context(GradientContext {
                tape_id,
                ..Default::default()
            })
            .compute_error(reason, true)
    }

    /// Create a tape operation error
    pub fn tape_operation_error(
        operation: &str,
        reason: &str,
        suggestions: Vec<String>,
    ) -> TensorError {
        AutogradErrorBuilder::new(operation)
            .with_metadata("component", "gradient_tape")
            .numerical_error(reason, suggestions)
    }

    /// Create a higher-order gradient error
    pub fn higher_order_gradient_error(operation: &str, order: usize, reason: &str) -> TensorError {
        AutogradErrorBuilder::new(operation)
            .with_gradient_context(GradientContext {
                is_higher_order: true,
                ..Default::default()
            })
            .with_metadata("gradient_order", order.to_string())
            .unsupported_operation(
                reason,
                vec!["Consider using numerical differentiation".to_string()],
            )
    }

    /// Create a checkpointing error
    pub fn checkpointing_error(operation: &str, reason: &str) -> TensorError {
        AutogradErrorBuilder::new(operation)
            .with_metadata("component", "checkpointing")
            .compute_error(reason, false)
    }
}

/// Error pattern validator for ensuring consistent error usage
pub struct ErrorPatternValidator;

impl ErrorPatternValidator {
    /// Validate that an error follows autograd conventions
    pub fn validate_error(error: &TensorError) -> ValidationResult {
        let mut issues = Vec::new();

        match error {
            TensorError::ShapeMismatch { context, .. }
            | TensorError::DeviceMismatch { context, .. }
            | TensorError::GradientNotEnabled { context, .. }
            | TensorError::InvalidOperation { context, .. }
            | TensorError::ComputeError { context, .. } => {
                if context.is_none() {
                    issues.push("Error missing context information".to_string());
                } else if let Some(ctx) = context {
                    // Check for gradient-specific metadata
                    if !ctx.metadata.contains_key("gradient_tape_id")
                        && !ctx.metadata.contains_key("component")
                    {
                        issues.push(
                            "Gradient errors should include tape_id or component metadata"
                                .to_string(),
                        );
                    }
                }
            }
            _ => {}
        }

        ValidationResult {
            is_valid: issues.is_empty(),
            issues,
        }
    }

    /// Check if error has gradient context
    pub fn has_gradient_context(error: &TensorError) -> bool {
        match error {
            TensorError::ShapeMismatch { context, .. }
            | TensorError::DeviceMismatch { context, .. }
            | TensorError::GradientNotEnabled { context, .. }
            | TensorError::InvalidOperation { context, .. }
            | TensorError::ComputeError { context, .. } => {
                if let Some(ctx) = context {
                    ctx.metadata.contains_key("gradient_tape_id")
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Result of error pattern validation
#[derive(Debug)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub issues: Vec<String>,
}

impl ValidationResult {
    /// Print validation issues
    pub fn print_issues(&self) {
        if !self.is_valid {
            println!("Error pattern validation issues:");
            for (i, issue) in self.issues.iter().enumerate() {
                println!("  {}. {}", i + 1, issue);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_builder_shape_mismatch() {
        let error = AutogradErrorBuilder::new("test_op")
            .with_gradient_context(GradientContext {
                tape_id: 42,
                operation_index: 5,
                num_inputs: 2,
                num_outputs: 1,
                is_higher_order: false,
                parent_operation: None,
            })
            .shape_mismatch("[10, 5]", "[10, 4]");

        match error {
            TensorError::ShapeMismatch {
                operation, context, ..
            } => {
                assert_eq!(operation, "test_op");
                assert!(context.is_some());

                if let Some(ctx) = context {
                    assert_eq!(ctx.metadata.get("tape_id"), Some(&"42".to_string()));
                }
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_gradient_context_metadata() {
        let error = AutogradErrorBuilder::new("backward")
            .with_gradient_context(GradientContext {
                tape_id: 100,
                operation_index: 10,
                num_inputs: 3,
                num_outputs: 2,
                is_higher_order: true,
                parent_operation: Some("forward".to_string()),
            })
            .invalid_operation("test reason");

        match error {
            TensorError::InvalidOperation { context, .. } => {
                assert!(context.is_some());

                if let Some(ctx) = context {
                    assert_eq!(
                        ctx.metadata.get("gradient_tape_id"),
                        Some(&"100".to_string())
                    );
                    assert_eq!(
                        ctx.metadata.get("is_higher_order_gradient"),
                        Some(&"true".to_string())
                    );
                    assert_eq!(
                        ctx.metadata.get("parent_operation"),
                        Some(&"forward".to_string())
                    );
                }
            }
            _ => panic!("Expected InvalidOperation error"),
        }
    }

    #[test]
    fn test_utils_gradient_shape_mismatch() {
        let error = utils::gradient_shape_mismatch("matmul_backward", &[10, 5], &[10, 4], 123, 7);

        match error {
            TensorError::ShapeMismatch { operation, .. } => {
                assert_eq!(operation, "matmul_backward");
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_error_validator() {
        let good_error = AutogradErrorBuilder::new("test")
            .with_gradient_context(GradientContext::default())
            .shape_mismatch("a", "b");

        let result = ErrorPatternValidator::validate_error(&good_error);
        assert!(result.is_valid);

        let bad_error = TensorError::shape_mismatch("test", "a", "b");
        let result = ErrorPatternValidator::validate_error(&bad_error);
        assert!(!result.is_valid);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_has_gradient_context() {
        let with_context = AutogradErrorBuilder::new("test")
            .with_gradient_context(GradientContext::default())
            .shape_mismatch("a", "b");

        assert!(ErrorPatternValidator::has_gradient_context(&with_context));

        let without_context = TensorError::shape_mismatch("test", "a", "b");
        assert!(!ErrorPatternValidator::has_gradient_context(
            &without_context
        ));
    }
}
