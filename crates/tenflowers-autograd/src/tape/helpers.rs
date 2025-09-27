//! Helper utilities for GradientTape implementation
//!
//! This module provides utility functions used throughout the gradient tape
//! implementation, including tensor value retrieval, gradient accumulation,
//! and numerical gradient checking utilities.

use num_traits::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

use super::structures::GradientTapeInner;
use super::TensorId;

/// Helper to get tensor value from stored values
pub(crate) fn get_tensor_value<T>(inner: &GradientTapeInner, id: TensorId) -> Option<Tensor<T>>
where
    T: Clone + 'static,
{
    inner
        .tensor_values
        .get(&id)
        .and_then(|any_tensor| any_tensor.downcast_ref::<Tensor<T>>())
        .cloned()
}

/// Helper to accumulate gradients
pub(crate) fn accumulate_gradient<T>(
    gradients: &mut HashMap<TensorId, Tensor<T>>,
    id: TensorId,
    grad: Tensor<T>,
) where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + std::ops::Add<Output = T>
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if let Some(existing_grad) = gradients.get_mut(&id) {
        // Accumulate gradients by addition
        if let Ok(sum) = existing_grad.add(&grad) {
            *existing_grad = sum;
        }
    } else {
        gradients.insert(id, grad);
    }
}

/// Helper function to compute the product of a gradient tensor with a vector
/// This handles the tensor contraction needed for JVP computation
#[allow(dead_code)]
pub(crate) fn compute_gradient_vector_product<T>(
    gradient: &Tensor<T>,
    vector: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For JVP, we need to compute the inner product of gradient and vector
    // The exact computation depends on the shapes:
    //
    // - If gradient and vector have same shape: element-wise multiply then sum
    // - If they're matrices: appropriate matrix multiplication
    // - General case: tensor contraction along matching dimensions

    let grad_shape = gradient.shape().dims();
    let vec_shape = vector.shape().dims();

    if grad_shape == vec_shape {
        // Same shape: element-wise multiply then sum all elements
        let elementwise_product = gradient.mul(vector)?;

        // Sum all elements to get scalar contribution to JVP
        let axes: Vec<i32> = (0..grad_shape.len() as i32).collect();
        elementwise_product.sum(Some(&axes), false)
    } else if grad_shape.len() == 2 && vec_shape.len() == 1 {
        // Matrix-vector multiplication case
        // grad is [m, n], vector is [n] -> result is [m]
        if grad_shape[1] == vec_shape[0] {
            gradient.matmul(vector)
        } else {
            // Shapes don't align for matrix multiplication
            // Fall back to broadcasting multiplication and sum
            let elementwise_product = gradient.mul(vector)?;
            elementwise_product.sum(Some(&[-1]), false) // Sum along last axis
        }
    } else {
        // General case: broadcasting multiplication followed by appropriate summation
        let elementwise_product = gradient.mul(vector)?;

        // Sum over dimensions that exist in vector but result should be scalar contribution
        // For simplicity, sum all elements (this works correctly for most cases)
        let axes: Vec<i32> = (0..elementwise_product.shape().dims().len() as i32).collect();
        elementwise_product.sum(Some(&axes), false)
    }
}

/// Compute numerical gradient for a specific input using finite differences
pub(crate) fn compute_numerical_gradient<T, F>(
    function: &F,
    inputs: &[Tensor<T>],
    input_index: usize,
    epsilon: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + num_traits::Float,
    F: Fn(&[Tensor<T>]) -> Result<Tensor<T>>,
{
    let input = &inputs[input_index];
    let input_data = input.as_slice().ok_or_else(|| {
        TensorError::other("Cannot access input tensor data for numerical gradient".into())
    })?;

    let mut numerical_grad_data = Vec::with_capacity(input_data.len());

    // Compute numerical gradient using central differences: (f(x+h) - f(x-h)) / (2*h)
    for elem_idx in 0..input_data.len() {
        // Create input with +epsilon perturbation
        let mut inputs_plus = inputs.to_vec();
        let mut input_plus_data = input_data.to_vec();
        input_plus_data[elem_idx] = input_plus_data[elem_idx] + epsilon;
        inputs_plus[input_index] = Tensor::from_vec(input_plus_data, input.shape().dims())?;

        // Create input with -epsilon perturbation
        let mut inputs_minus = inputs.to_vec();
        let mut input_minus_data = input_data.to_vec();
        input_minus_data[elem_idx] = input_minus_data[elem_idx] - epsilon;
        inputs_minus[input_index] = Tensor::from_vec(input_minus_data, input.shape().dims())?;

        // Evaluate function at perturbed points
        let f_plus = function(&inputs_plus)?;
        let f_minus = function(&inputs_minus)?;

        // Extract scalar values (assuming function returns scalar)
        let f_plus_val = f_plus.as_slice().ok_or_else(|| {
            TensorError::other(
                "Function output must be accessible as slice for gradient checking".into(),
            )
        })?[0];

        let f_minus_val = f_minus.as_slice().ok_or_else(|| {
            TensorError::other(
                "Function output must be accessible as slice for gradient checking".into(),
            )
        })?[0];

        // Compute numerical gradient: (f(x+h) - f(x-h)) / (2*h)
        let two_epsilon = epsilon + epsilon;
        let numerical_grad_elem = (f_plus_val - f_minus_val) / two_epsilon;
        numerical_grad_data.push(numerical_grad_elem);
    }

    Tensor::from_vec(numerical_grad_data, input.shape().dims())
}

/// Compare analytical and numerical gradients within specified tolerances
pub(crate) fn compare_gradients<T>(
    analytical: &Tensor<T>,
    numerical: &Tensor<T>,
    relative_tolerance: T,
    absolute_tolerance: T,
    input_index: usize,
) -> Result<()>
where
    T: Clone
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float,
{
    let analytical_data = analytical
        .as_slice()
        .ok_or_else(|| TensorError::other("Cannot access analytical gradient data".into()))?;

    let numerical_data = numerical
        .as_slice()
        .ok_or_else(|| TensorError::other("Cannot access numerical gradient data".into()))?;

    if analytical_data.len() != numerical_data.len() {
        return Err(TensorError::shape_mismatch(
            "gradient_comparison",
            &format!("analytical gradient length: {}", analytical_data.len()),
            &format!("numerical gradient length: {}", numerical_data.len()),
        ));
    }

    let mut max_error = T::zero();
    let mut max_relative_error = T::zero();

    for (i, (analytical_val, numerical_val)) in analytical_data
        .iter()
        .zip(numerical_data.iter())
        .enumerate()
    {
        let absolute_error = (*analytical_val - *numerical_val).abs();
        let relative_error = if numerical_val.abs() > T::zero() {
            absolute_error / numerical_val.abs()
        } else {
            absolute_error
        };

        max_error = max_error.max(absolute_error);
        max_relative_error = max_relative_error.max(relative_error);

        // Check if this element exceeds tolerances
        if absolute_error > absolute_tolerance && relative_error > relative_tolerance {
            return Err(TensorError::other(format!(
                "Gradient check failed for input {} element {}: analytical={}, numerical={}, abs_err={}, rel_err={}, abs_tol={}, rel_tol={}",
                input_index, i, analytical_val.to_f64().unwrap_or(0.0),
                numerical_val.to_f64().unwrap_or(0.0),
                absolute_error.to_f64().unwrap_or(0.0),
                relative_error.to_f64().unwrap_or(0.0),
                absolute_tolerance.to_f64().unwrap_or(0.0),
                relative_tolerance.to_f64().unwrap_or(0.0)
            )));
        }
    }

    println!(
        "✅ Gradient check passed for input {}: max_abs_err={:.2e}, max_rel_err={:.2e}",
        input_index,
        max_error.to_f64().unwrap_or(0.0),
        max_relative_error.to_f64().unwrap_or(0.0)
    );

    Ok(())
}
