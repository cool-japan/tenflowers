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

/// Process backward pass for 1D convolution operation.
///
/// Delegates to the real `conv1d_backward` kernel (the 1D specialization of
/// `conv2d_backward`), which computes:
/// 1. Input gradient: the transpose of the forward cross-correlation.
/// 2. Weight gradient: correlation of the input with `grad_output`.
/// 3. Bias gradient: sum of `grad_output` over batch and length dimensions.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_conv1d_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    weight: TensorId,
    bias: Option<TensorId>,
    stride: usize,
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
            "Conv1D backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let weight_tensor = get_tensor_value::<T>(inner, weight).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Conv1D backward: weight tensor value not recorded on the tape".to_string(),
        )
    })?;

    let bias_tensor = match bias {
        Some(bias_id) => Some(get_tensor_value::<T>(inner, bias_id).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "Conv1D backward: bias tensor value not recorded on the tape".to_string(),
            )
        })?),
        None => None,
    };

    let (grad_input, grad_weight, grad_bias) = crate::ops::convolution_ops::conv1d_backward(
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

/// Process backward pass for 3D convolution operation.
///
/// Delegates to the real `conv3d_backward` kernel, which computes:
/// 1. Input gradient: the transpose of the forward cross-correlation.
/// 2. Weight gradient: correlation of the input with `grad_output`.
/// 3. Bias gradient: sum of `grad_output` over batch, depth, height, and
///    width dimensions.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_conv3d_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    weight: TensorId,
    bias: Option<TensorId>,
    stride: (usize, usize, usize),
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
            "Conv3D backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let weight_tensor = get_tensor_value::<T>(inner, weight).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "Conv3D backward: weight tensor value not recorded on the tape".to_string(),
        )
    })?;

    let bias_tensor = match bias {
        Some(bias_id) => Some(get_tensor_value::<T>(inner, bias_id).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "Conv3D backward: bias tensor value not recorded on the tape".to_string(),
            )
        })?),
        None => None,
    };

    let (grad_input, grad_weight, grad_bias) = crate::ops::convolution_ops::conv3d_backward(
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

/// Process backward pass for batch normalization operation.
///
/// Delegates to the real `batch_norm_backward` kernel, which computes:
/// 1. Input gradient: the full BatchNorm backward formula (training mode uses
///    batch statistics; eval mode uses the running statistics and is affine
///    in gamma).
/// 2. Gamma gradient: sum of `grad_output * normalized_input` over the
///    reduction axes.
/// 3. Beta gradient: sum of `grad_output` over the reduction axes.
///
/// `running_mean`/`running_var` are forward-only statistics (read for their
/// values but never differentiated through), so no gradient is accumulated
/// for them.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_batchnorm_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gamma: TensorId,
    beta: TensorId,
    running_mean: TensorId,
    running_var: TensorId,
    epsilon: f32,
    training: bool,
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
            "BatchNorm backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let gamma_tensor = get_tensor_value::<T>(inner, gamma).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "BatchNorm backward: gamma tensor value not recorded on the tape".to_string(),
        )
    })?;
    let beta_tensor = get_tensor_value::<T>(inner, beta).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "BatchNorm backward: beta tensor value not recorded on the tape".to_string(),
        )
    })?;
    let running_mean_tensor = get_tensor_value::<T>(inner, running_mean).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "BatchNorm backward: running_mean tensor value not recorded on the tape".to_string(),
        )
    })?;
    let running_var_tensor = get_tensor_value::<T>(inner, running_var).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "BatchNorm backward: running_var tensor value not recorded on the tape".to_string(),
        )
    })?;

    let epsilon_t = T::from(epsilon).unwrap_or_else(|| T::default());

    let (grad_input, grad_gamma, grad_beta) = crate::ops::normalization_ops::batch_norm_backward(
        grad_output,
        &input_tensor,
        &gamma_tensor,
        &beta_tensor,
        &running_mean_tensor,
        &running_var_tensor,
        training,
        epsilon_t,
    )?;

    super::super::utils::accumulate_gradient(gradients, input, grad_input)?;
    super::super::utils::accumulate_gradient(gradients, gamma, grad_gamma)?;
    super::super::utils::accumulate_gradient(gradients, beta, grad_beta)?;

    Ok(())
}

/// Process backward pass for layer normalization operation.
///
/// Delegates to the real `layer_norm_backward` kernel, which computes:
/// 1. Input gradient: the full LayerNorm backward formula over the
///    normalized dimensions.
/// 2. Gamma gradient: sum of `grad_output * normalized_input` over the
///    non-normalized (batch) dimensions.
/// 3. Beta gradient: sum of `grad_output` over the non-normalized
///    dimensions.
pub(super) fn process_layernorm_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    input: TensorId,
    gamma: TensorId,
    beta: TensorId,
    normalized_shape: Vec<usize>,
    epsilon: f32,
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
            "LayerNorm backward: input tensor value not recorded on the tape".to_string(),
        )
    })?;
    let gamma_tensor = get_tensor_value::<T>(inner, gamma).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "LayerNorm backward: gamma tensor value not recorded on the tape".to_string(),
        )
    })?;
    let beta_tensor = get_tensor_value::<T>(inner, beta).ok_or_else(|| {
        tenflowers_core::TensorError::invalid_operation_simple(
            "LayerNorm backward: beta tensor value not recorded on the tape".to_string(),
        )
    })?;

    let epsilon_t = T::from(epsilon).unwrap_or_else(|| T::default());

    let (grad_input, grad_gamma, grad_beta) = crate::ops::normalization_ops::layer_norm_backward(
        grad_output,
        &input_tensor,
        &gamma_tensor,
        &beta_tensor,
        &normalized_shape,
        epsilon_t,
    )?;

    super::super::utils::accumulate_gradient(gradients, input, grad_input)?;
    super::super::utils::accumulate_gradient(gradients, gamma, grad_gamma)?;
    super::super::utils::accumulate_gradient(gradients, beta, grad_beta)?;

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
