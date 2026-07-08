//! Neural network operation gradients
//!
//! This module contains gradient computation logic for neural network operations
//! like convolution, batch normalization, layer normalization, dropout, etc.

use scirs2_core::numeric::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

use super::super::helpers::get_tensor_value;
use super::super::structures::GradientTapeInner;
use super::super::{GradientTape, TensorId};

/// Process backward pass for 2D convolution operation.
///
/// Delegates to the real `conv2d_backward` kernel, which computes:
/// 1. Input gradient: the transpose of the forward cross-correlation.
/// 2. Weight gradient: correlation of the input with `grad_output`.
/// 3. Bias gradient: sum of `grad_output` over batch and spatial dimensions.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_conv2d_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    weight: TensorId,
    bias: Option<TensorId>,
    stride: (usize, usize),
    padding: &str,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let input_tensor = get_tensor_value::<T>(inner, input).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Conv2D backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let weight_tensor = get_tensor_value::<T>(inner, weight).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Conv2D backward: weight tensor value not recorded on the tape".to_string(),
        )
    })?;

    let bias_tensor = match bias {
        Some(bias_id) => Some(get_tensor_value::<T>(inner, bias_id).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "Conv2D backward: bias tensor value not recorded on the tape".to_string(),
            )
        })?),
        None => None,
    };

    let (grad_input, grad_weight, grad_bias) = crate::ops::convolution_ops::conv2d_backward(
        grad_output,
        &input_tensor,
        &weight_tensor,
        bias_tensor.as_ref(),
        stride,
        padding,
    )?;

    super::super::utils::accumulate_gradient(gradients, input, grad_input)?;
    super::super::utils::accumulate_gradient(gradients, weight, grad_weight)?;

    if let (Some(bias_id), Some(grad_bias)) = (bias, grad_bias) {
        super::super::utils::accumulate_gradient(gradients, bias_id, grad_bias)?;
    }

    Ok(())
}

/// Process backward pass for batch normalization operation
pub(super) fn process_batchnorm_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gamma: TensorId,
    beta: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // BatchNorm gradients are complex - simplified implementation
    // Full implementation requires computing:
    // 1. Input gradient considering normalization statistics
    // 2. Gamma gradient: element-wise multiplication with normalized input
    // 3. Beta gradient: sum of grad_output

    if let Some(_input_tensor) = get_tensor_value::<T>(inner, input) {
        if let Some(gamma_tensor) = get_tensor_value::<T>(inner, gamma) {
            if let Some(beta_tensor) = get_tensor_value::<T>(inner, beta) {
                // Simplified gradient computation
                // Input gradient: pass through for now
                let input_grad = grad_output.clone();
                super::super::utils::accumulate_gradient(gradients, input, input_grad)?;

                // Gamma gradient: simplified as zeros
                let gamma_grad = Tensor::zeros(gamma_tensor.shape().dims());
                super::super::utils::accumulate_gradient(gradients, gamma, gamma_grad)?;

                // Beta gradient: simplified as zeros
                let beta_grad = Tensor::zeros(beta_tensor.shape().dims());
                super::super::utils::accumulate_gradient(gradients, beta, beta_grad)?;
            }
        }
    }
    Ok(())
}

/// Process backward pass for layer normalization operation
pub(super) fn process_layernorm_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gamma: TensorId,
    beta: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // LayerNorm gradients are similar to BatchNorm but across different dimensions
    // Simplified implementation for now

    if let Some(_input_tensor) = get_tensor_value::<T>(inner, input) {
        if let Some(gamma_tensor) = get_tensor_value::<T>(inner, gamma) {
            if let Some(beta_tensor) = get_tensor_value::<T>(inner, beta) {
                // Simplified gradient computation
                // Input gradient: pass through for now
                let input_grad = grad_output.clone();
                super::super::utils::accumulate_gradient(gradients, input, input_grad)?;

                // Gamma gradient: simplified as zeros
                let gamma_grad = Tensor::zeros(gamma_tensor.shape().dims());
                super::super::utils::accumulate_gradient(gradients, gamma, gamma_grad)?;

                // Beta gradient: simplified as zeros
                let beta_grad = Tensor::zeros(beta_tensor.shape().dims());
                super::super::utils::accumulate_gradient(gradients, beta, beta_grad)?;
            }
        }
    }
    Ok(())
}

/// Process backward pass for dropout operation
pub(super) fn process_dropout_backward<T>(
    _tape: &GradientTape,
    _inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gradients: &mut HashMap<TensorId, Tensor<T>>,
) -> Result<()>
where
    T: Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + PartialOrd
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Dropout gradient: apply the same mask that was used in forward pass
    // For now, simplified as identity (assumes training mode)
    super::super::utils::accumulate_gradient(gradients, input, grad_output.clone())?;
    Ok(())
}
