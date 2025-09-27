//! Basic arithmetic operation gradients
//!
//! This module contains gradient computation logic for fundamental arithmetic operations
//! like addition, multiplication, subtraction, division, and matrix multiplication.

use num_traits::{One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

use super::super::helpers::get_tensor_value;
use super::super::structures::GradientTapeInner;
use super::super::{GradientTape, TensorId};

/// Process backward pass for addition operation
pub(super) fn process_add_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Addition gradient: d/dx(x + y) = 1, d/dy(x + y) = 1
    // But we need to handle broadcasting properly

    // Get tensor values for proper unbroadcasting
    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Use proper add_backward that handles broadcasting/unbroadcasting
            let (lhs_grad, rhs_grad) =
                crate::grad_ops::basic_ops::add_backward(grad_output, &lhs_tensor, &rhs_tensor)?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        } else {
            // Fallback if tensors not available
            super::super::utils::accumulate_gradient(gradients, lhs, grad_output.clone())?;
            super::super::utils::accumulate_gradient(gradients, rhs, grad_output.clone())?;
        }
    } else {
        // Fallback if tensors not available
        super::super::utils::accumulate_gradient(gradients, lhs, grad_output.clone())?;
        super::super::utils::accumulate_gradient(gradients, rhs, grad_output.clone())?;
    }
    Ok(())
}

/// Process backward pass for multiplication operation
pub(super) fn process_mul_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Multiplication gradient: d/dx(x * y) = y, d/dy(x * y) = x

    // Get tensor values
    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Use proper mul_backward that handles broadcasting/unbroadcasting
            let (lhs_grad, rhs_grad) =
                crate::grad_ops::basic_ops::mul_backward(grad_output, &lhs_tensor, &rhs_tensor)?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        }
    }
    Ok(())
}

/// Process backward pass for subtraction operation
pub(super) fn process_sub_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Subtraction gradient: d/dx(x - y) = 1, d/dy(x - y) = -1
    // But we need to handle broadcasting properly

    // Get tensor values for proper unbroadcasting
    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Use proper sub_backward that handles broadcasting/unbroadcasting
            let (lhs_grad, rhs_grad) =
                crate::grad_ops::basic_ops::sub_backward(grad_output, &lhs_tensor, &rhs_tensor)?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        } else {
            // Fallback if tensors not available
            super::super::utils::accumulate_gradient(gradients, lhs, grad_output.clone())?;
            let neg_grad = tenflowers_core::ops::neg(grad_output)?;
            super::super::utils::accumulate_gradient(gradients, rhs, neg_grad)?;
        }
    } else {
        // Fallback if tensors not available
        super::super::utils::accumulate_gradient(gradients, lhs, grad_output.clone())?;
        let neg_grad = tenflowers_core::ops::neg(grad_output)?;
        super::super::utils::accumulate_gradient(gradients, rhs, neg_grad)?;
    }
    Ok(())
}

/// Process backward pass for division operation
pub(super) fn process_div_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Division gradient: d/dx(x / y) = 1/y, d/dy(x / y) = -x/y²

    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Gradient for lhs: grad_output / rhs
            let lhs_grad = tenflowers_core::ops::div(grad_output, &rhs_tensor)?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;

            // Gradient for rhs: -grad_output * lhs / rhs²
            let rhs_squared = tenflowers_core::ops::mul(&rhs_tensor, &rhs_tensor)?;
            let temp = tenflowers_core::ops::mul(grad_output, &lhs_tensor)?;
            let rhs_grad_pos = tenflowers_core::ops::div(&temp, &rhs_squared)?;
            let rhs_grad = tenflowers_core::ops::neg(&rhs_grad_pos)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        }
    }
    Ok(())
}

/// Process backward pass for matrix multiplication operation
pub(super) fn process_matmul_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Matrix multiplication gradient:
    // If C = A @ B, then dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC

    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Gradient for lhs: grad_output @ rhs^T
            let rhs_transposed = tenflowers_core::ops::transpose(&rhs_tensor)?;
            let lhs_grad = tenflowers_core::ops::matmul(grad_output, &rhs_transposed)?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;

            // Gradient for rhs: lhs^T @ grad_output
            let lhs_transposed = tenflowers_core::ops::transpose(&lhs_tensor)?;
            let rhs_grad = tenflowers_core::ops::matmul(&lhs_transposed, grad_output)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        }
    }
    Ok(())
}

/// Process backward pass for power operation
pub(super) fn process_pow_backward<T>(
    _tape: &GradientTape,
    inner: &GradientTapeInner,
    grad_output: &Tensor<T>,
    lhs: TensorId,
    rhs: TensorId,
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
        + num_traits::Float
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Power operation gradient: d/dx(x^y) = y * x^(y-1), d/dy(x^y) = x^y * ln(x)

    // Get tensor values
    if let Some(lhs_tensor) = get_tensor_value::<T>(inner, lhs) {
        if let Some(rhs_tensor) = get_tensor_value::<T>(inner, rhs) {
            // Compute the output tensor (needed for power backward)
            let output = lhs_tensor.pow(&rhs_tensor)?;

            // Use proper pow_backward that handles broadcasting/unbroadcasting
            let (lhs_grad, rhs_grad) = crate::ops::binary_ops::pow_backward(
                grad_output,
                &lhs_tensor,
                &rhs_tensor,
                &output,
            )?;
            super::super::utils::accumulate_gradient(gradients, lhs, lhs_grad)?;
            super::super::utils::accumulate_gradient(gradients, rhs, rhs_grad)?;
        }
    }
    Ok(())
}
