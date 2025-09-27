//! Gradient operations module
//!
//! This module provides gradient computation functions for automatic differentiation.
//! It is organized into specialized submodules for better maintainability:
//!
//! - `utils`: Common utility functions for gradient computations
//! - `basic_ops`: Basic arithmetic operations (add, mul, sub, div, pow)
//! - `activation_ops`: Activation function gradients (ReLU, Sigmoid, Tanh, GELU, etc.)
//! - `tensor_ops`: Tensor manipulation operations (slice, concat, stack, transpose, etc.)
//! - `fused_ops`: Fused forward-backward operations for performance
//! - `advanced_ops`: Advanced mathematical operations and validation utilities

// Import all submodules
pub mod activation_ops;
pub mod advanced_ops;
pub mod basic_ops;
pub mod fused_ops;
pub mod tensor_ops;
pub mod utils;

// Re-export everything for backward compatibility
// This ensures existing code using these functions continues to work unchanged

// Basic arithmetic operations
pub use basic_ops::{
    add_backward, compute_general_power_grad_b, compute_integer_power_grad_b,
    compute_log_approximation, div_backward, is_integer_power, mul_backward, pow_backward,
    sub_backward,
};

// Activation function gradients
pub use activation_ops::{
    elu_backward, gelu_backward, leaky_relu_backward, mish_backward, prelu_backward, relu_backward,
    relu_forward, sigmoid_backward, softmax_backward, swish_backward, tanh_backward,
};

// Tensor manipulation operations
pub use tensor_ops::{
    add_to_slice_4d, concat_backward, einsum_backward, gather_backward, get_tensor_element_4d,
    scatter_backward, slice_backward, slice_tensor_4d, split_backward, squeeze_backward,
    stack_backward, transpose_backward, unsqueeze_backward, SliceSpec,
};

// Fused operations
pub use fused_ops::{
    batch_fused_activations_forward_backward, fused_batch_norm_forward_backward,
    fused_dropout_forward_backward, fused_gelu_forward_backward, fused_layer_norm_forward_backward,
    fused_log_softmax_forward_backward, fused_relu_forward_backward,
    fused_sigmoid_forward_backward, fused_tanh_forward_backward,
};

// Advanced operations
pub use advanced_ops::{cholesky_backward, eig_backward, qr_backward, svd_backward};

// Utility functions
pub use utils::{
    check_broadcast_compatible, compute_slice_indices, element_count, ones_like, unbroadcast,
    zeros_like,
};

// Validation utilities
pub use advanced_ops::validation::{
    check_gradient_consistency, gradient_check, validate_gradients, GradientValidationStats,
};

// Re-export imports that were used in the original file for compatibility

// Re-export operations from the refactored modules that were in the original file
pub use crate::ops::convolution_ops::*;
#[allow(unused_imports)] // Functions are used via qualified paths in tape.rs
pub use crate::ops::fft_ops::*;
pub use crate::ops::linalg_ops::*;
pub use crate::ops::normalization_ops::*;
pub use crate::ops::reduction_ops::*;

// Note: Any additional functions that were in the original grad_ops.rs
// that are not yet categorized can be added here temporarily and then
// moved to the appropriate submodule later.

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_backward_compatibility_basic_ops() {
        // Test that re-exported functions work as before
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], &[2]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap();

        // Test add_backward (from basic_ops)
        let result = add_backward(&grad_output, &a, &b);
        assert!(result.is_ok());

        // Test mul_backward (from basic_ops)
        let result = mul_backward(&grad_output, &a, &b);
        assert!(result.is_ok());

        // Test sub_backward (from basic_ops)
        let result = sub_backward(&grad_output, &a, &b);
        assert!(result.is_ok());

        // Test div_backward (from basic_ops)
        let result = div_backward(&grad_output, &a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility_activation_ops() {
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();

        // Test relu_backward (from activation_ops)
        let result = relu_backward(&grad_output, &input);
        assert!(result.is_ok());

        // Test sigmoid_backward (from activation_ops)
        let sigmoid_output = tenflowers_core::ops::sigmoid(&input).unwrap();
        let result = sigmoid_backward(&grad_output, &sigmoid_output);
        assert!(result.is_ok());

        // Test tanh_backward (from activation_ops)
        let tanh_output = tenflowers_core::ops::tanh(&input).unwrap();
        let result = tanh_backward(&grad_output, &tanh_output);
        assert!(result.is_ok());

        // Test gelu_backward (from activation_ops)
        let result = gelu_backward(&grad_output, &input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility_tensor_ops() {
        // Test slice_backward (from tensor_ops)
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let input_shape = &[4];
        let slice_specs = vec![SliceSpec::range(1, 3)];
        let result = slice_backward(&grad_output, input_shape, &slice_specs);
        assert!(result.is_ok());

        // Test concat_backward (from tensor_ops)
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let input_shapes = vec![vec![2], vec![2]];
        let result = concat_backward(&grad_output, &input_shapes, 0);
        assert!(result.is_ok());

        // Test transpose_backward (from tensor_ops)
        let grad_output = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = transpose_backward(&grad_output, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility_fused_ops() {
        let input = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], &[3]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], &[3]).unwrap();

        // Test fused operations (from fused_ops)
        let result = fused_relu_forward_backward(&input, &grad_output);
        assert!(result.is_ok());

        let result = fused_sigmoid_forward_backward(&input, &grad_output);
        assert!(result.is_ok());

        let result = fused_tanh_forward_backward(&input, &grad_output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility_advanced_ops() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let grad_output = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();

        // Test SVD backward (from advanced_ops)
        let result = svd_backward(&grad_output, &input);
        assert!(result.is_ok());

        // Test eigendecomposition backward (from advanced_ops)
        let eigenvalues = Tensor::from_vec(vec![1.0f32, 3.0], &[2]).unwrap();
        let eigenvectors = Tensor::from_vec(vec![1.0f32, -1.0, 1.0, 1.0], &[2, 2]).unwrap();
        let result = eig_backward(&grad_output, &input, &eigenvalues, &eigenvectors);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility_utility_functions() {
        let grad = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let target_shape = tenflowers_core::Shape::new(vec![2, 2]);

        // Test unbroadcast (from utils)
        let result = unbroadcast(&grad, &target_shape);
        assert!(result.is_ok());

        // Test other utility functions
        let reference = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let result = zeros_like(&reference);
        assert!(result.is_ok());

        let result = ones_like(&reference);
        assert!(result.is_ok());

        // Test element_count
        let count = element_count(&[2, 3, 4]);
        assert_eq!(count, 24);
    }

    #[test]
    fn test_backward_compatibility_validation() {
        use super::advanced_ops::validation::*;

        // Test gradient validation stats creation
        let stats = GradientValidationStats {
            max_absolute_error: 1e-6,
            mean_absolute_error: 1e-8,
            max_relative_error: 1e-4,
            mean_relative_error: 1e-6,
            num_elements: 100,
            validation_passed: true,
        };

        assert!(stats.validation_passed);
        assert_eq!(stats.num_elements, 100);

        // Test gradient consistency check
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let gradient = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], &[3]).unwrap();
        let tolerance = 1e-6_f32;

        let result = check_gradient_consistency(&input, &gradient, tolerance);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_modules_accessible() {
        // Verify all modules are accessible through the public interface

        // Test utils module
        let spec = utils::SliceSpec::all();
        assert_eq!(spec.step, Some(1));

        // Test basic_ops module (access through re-export)
        let a = Tensor::from_scalar(1.0f32);
        let b = Tensor::from_scalar(2.0f32);
        let grad = Tensor::from_scalar(1.0f32);
        let result = add_backward(&grad, &a, &b);
        assert!(result.is_ok());

        // Test activation_ops module (access through re-export)
        let input = Tensor::from_scalar(1.0f32);
        let result = relu_backward(&grad, &input);
        assert!(result.is_ok());

        // Test tensor_ops module (access through re-export)
        let slice_spec = SliceSpec::single(0);
        assert_eq!(slice_spec.start, Some(0));
        assert_eq!(slice_spec.end, Some(1));

        // Test fused_ops module (access through re-export)
        let result = fused_relu_forward_backward(&input, &grad);
        assert!(result.is_ok());

        // Test advanced_ops module (access through re-export)
        let matrix = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let result = svd_backward(&grad, &matrix);
        assert!(result.is_ok());
    }
}
