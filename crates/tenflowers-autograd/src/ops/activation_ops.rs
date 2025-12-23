//! Activation function gradient implementations
//!
//! This module contains gradient implementations for various activation functions including
//! ReLU, Sigmoid, Tanh, GELU, Swish/SiLU, LeakyReLU, Softmax, and others.

use crate::tensor_ext::TensorAutograd;
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor};

/// Backward pass for ReLU activation
/// For y = relu(x) = max(0, x), grad_x = grad_y * (x > 0)
pub fn relu_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Mul<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create a mask where input > 0
    let mask = input.relu_mask()?;

    // Multiply gradient by mask
    grad_output.mul(&mask)
}

/// Backward pass for sigmoid activation
/// For y = sigmoid(x) = 1 / (1 + exp(-x)), grad_x = grad_y * y * (1 - y)
/// Includes numerical stability improvements for edge cases
pub fn sigmoid_backward<T>(grad_output: &Tensor<T>, output: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For numerical stability, clamp sigmoid output to avoid exact 0 or 1 values
    // which would result in zero gradients and potential gradient vanishing
    let eps = T::from(1e-7).unwrap_or_else(|| T::from(0.0000001).unwrap());
    let one_minus_eps = T::one() - eps;

    // Clamp output values: max(eps, min(1-eps, y))
    let eps_tensor = Tensor::from_scalar(eps);
    let one_minus_eps_tensor = Tensor::from_scalar(one_minus_eps);

    // output_clamped = max(eps, min(1-eps, output))
    let clamped_max = tenflowers_core::ops::where_op(
        &output.gt(&one_minus_eps_tensor)?,
        &one_minus_eps_tensor,
        output,
    )?;
    let output_clamped =
        tenflowers_core::ops::where_op(&clamped_max.lt(&eps_tensor)?, &eps_tensor, &clamped_max)?;

    // grad_x = grad_y * y * (1 - y) using clamped values
    let one_tensor = Tensor::<T>::ones(output.shape().dims());
    let one_minus_output = one_tensor.sub(&output_clamped)?;

    let grad_times_output = grad_output.mul(&output_clamped)?;
    let result = grad_times_output.mul(&one_minus_output)?;

    Ok(result)
}

/// Backward pass for tanh activation
/// For y = tanh(x), grad_x = grad_y * (1 - y^2)
pub fn tanh_backward<T>(grad_output: &Tensor<T>, output: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // grad_x = grad_y * (1 - y^2)
    let one_tensor = Tensor::<T>::ones(output.shape().dims());
    let output_squared = output.mul(output)?;
    let one_minus_y_squared = one_tensor.sub(&output_squared)?;

    let result = grad_output.mul(&one_minus_y_squared)?;

    Ok(result)
}

/// Backward pass for GELU activation (Gaussian Error Linear Unit)
/// For y = GELU(x) = x * Phi(x) where Phi is the CDF of the standard normal distribution
/// grad_x = grad_y * (Phi(x) + x * phi(x)) where phi is the PDF of the standard normal
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Using the approximation gradient:
    // d/dx GELU(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) +
    //                0.5 * x * sech^2(sqrt(2/π) * (x + 0.044715 * x^3)) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)

    // For numerical stability, we'll use a simpler approximation:
    // GELU'(x) ≈ 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3))) +
    //            0.5 * x * (1 - tanh^2(0.7978845608 * (x + 0.044715 * x^3))) * 0.7978845608 * (1 + 0.134145 * x^2)

    let x_squared = input.mul(input)?;
    let x_cubed = x_squared.mul(input)?;

    // Constants
    let sqrt_2_over_pi = T::from(0.7978845608_f64).unwrap(); // sqrt(2/π)
    let alpha = T::from(0.044715_f64).unwrap();
    let three_alpha = T::from(0.134145_f64).unwrap(); // 3 * 0.044715
    let half = T::from(0.5_f64).unwrap();
    let _one = T::one();

    // Compute inner term: sqrt(2/π) * (x + 0.044715 * x^3)
    let alpha_x_cubed = Tensor::from_scalar(alpha).mul(&x_cubed)?;
    let inner_arg = input.add(&alpha_x_cubed)?;
    let scaled_arg = Tensor::from_scalar(sqrt_2_over_pi).mul(&inner_arg)?;

    // Compute tanh and its derivative components
    let tanh_val = tenflowers_core::ops::tanh(&scaled_arg)?;
    let one_tensor = Tensor::ones(input.shape().dims());
    let tanh_squared = tanh_val.mul(&tanh_val)?;
    let sech_squared = one_tensor.sub(&tanh_squared)?; // 1 - tanh^2 = sech^2

    // First term: 0.5 * (1 + tanh(...))
    let first_term = one_tensor.add(&tanh_val)?.mul(&Tensor::from_scalar(half))?;

    // Second term: 0.5 * x * sech^2(...) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2)
    let derivative_inner = one_tensor.add(&Tensor::from_scalar(three_alpha).mul(&x_squared)?)?;
    let second_term = Tensor::from_scalar(half)
        .mul(input)?
        .mul(&sech_squared)?
        .mul(&Tensor::from_scalar(sqrt_2_over_pi))?
        .mul(&derivative_inner)?;

    // Combined derivative
    let gelu_grad = first_term.add(&second_term)?;

    // Apply chain rule
    grad_output.mul(&gelu_grad)
}

/// Backward pass for Swish/SiLU activation
/// For y = Swish(x) = x * sigmoid(x), grad_x = grad_y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
/// Equivalently: grad_x = grad_y * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
pub fn swish_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute sigmoid(x)
    let sigmoid_x = tenflowers_core::ops::sigmoid(input)?;

    // Compute 1 - sigmoid(x)
    let one_tensor = Tensor::ones(input.shape().dims());
    let one_minus_sigmoid = one_tensor.sub(&sigmoid_x)?;

    // Compute x * (1 - sigmoid(x))
    let x_times_complement = input.mul(&one_minus_sigmoid)?;

    // Compute 1 + x * (1 - sigmoid(x))
    let inner_term = one_tensor.add(&x_times_complement)?;

    // Compute sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let swish_grad = sigmoid_x.mul(&inner_term)?;

    // Apply chain rule
    grad_output.mul(&swish_grad)
}

/// Backward pass for LeakyReLU activation
/// For y = LeakyReLU(x) = x if x > 0 else alpha * x, grad_x = grad_y * (1 if x > 0 else alpha)
pub fn leaky_relu_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    alpha: T,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create mask for x > 0
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let positive_mask = input.gt(&zero_tensor)?;

    // Create alpha tensor
    let alpha_tensor = Tensor::from_scalar(alpha);
    let one_tensor = Tensor::ones(input.shape().dims());

    // Create gradient mask: 1 where x > 0, alpha where x <= 0
    // Use where operation to select between 1.0 and alpha based on the condition
    let grad_mask = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, &alpha_tensor)?;

    // Apply chain rule
    grad_output.mul(&grad_mask)
}

/// Backward pass for softmax activation
/// For y = softmax(x), grad_x = y * (grad_y - sum(y * grad_y, axis=axis, keepdims=True))
pub fn softmax_backward<T>(
    grad_output: &Tensor<T>,
    output: &Tensor<T>,
    axis: Option<i32>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // First compute sum(y * grad_y) along the specified axis
    let y_times_grad = output.mul(grad_output)?;

    let axis_slice = axis.map(|a| vec![a]).unwrap_or_else(|| vec![-1]);
    let sum_y_grad = y_times_grad.sum(Some(&axis_slice), true)?;

    // grad_x = y * (grad_y - sum_y_grad)
    let grad_minus_sum = grad_output.sub(&sum_y_grad)?;
    let result = output.mul(&grad_minus_sum)?;

    Ok(result)
}

/// Backward pass for Mish activation
/// For y = Mish(x) = x * tanh(softplus(x)), grad_x = grad_y * mish_grad(x)
/// Uses approximation to avoid dependency on ln operation
pub fn mish_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Mish derivative approximation using numerical stability
    // d/dx Mish(x) ≈ sigmoid(x) * (1 + x * tanh(sigmoid(x)))

    let sigmoid_x = tenflowers_core::ops::sigmoid(input)?;
    let tanh_sigmoid = tenflowers_core::ops::tanh(&sigmoid_x)?;
    let one_tensor = Tensor::ones(input.shape().dims());

    // Compute x * tanh(sigmoid(x))
    let x_tanh_sigmoid = input.mul(&tanh_sigmoid)?;

    // Compute 1 + x * tanh(sigmoid(x))
    let inner_term = one_tensor.add(&x_tanh_sigmoid)?;

    // Compute sigmoid(x) * (1 + x * tanh(sigmoid(x)))
    let mish_grad = sigmoid_x.mul(&inner_term)?;

    // Apply chain rule
    grad_output.mul(&mish_grad)
}

/// Backward pass for ELU activation
/// For y = ELU(x) = x if x > 0 else alpha * (e^x - 1), grad_x = grad_y * (1 if x > 0 else alpha * e^x)
pub fn elu_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>, alpha: T) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create mask for x > 0
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let positive_mask = input.gt(&zero_tensor)?;

    // For positive values: gradient = 1
    let one_tensor = Tensor::ones(input.shape().dims());

    // For negative values: gradient = alpha * e^x
    let exp_x = input.exp()?;
    let alpha_tensor = Tensor::from_scalar(alpha);
    let negative_grad = alpha_tensor.mul(&exp_x)?;

    // Create gradient mask: 1 where x > 0, alpha * e^x where x <= 0
    let grad_mask = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, &negative_grad)?;

    // Apply chain rule
    grad_output.mul(&grad_mask)
}

/// Backward pass for PReLU activation
/// For y = PReLU(x) = x if x > 0 else alpha * x, grad_x = grad_y * (1 if x > 0 else alpha)
/// grad_alpha = grad_y * (0 if x > 0 else x)
pub fn prelu_backward<T>(
    grad_output: &Tensor<T>,
    input: &Tensor<T>,
    alpha: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create mask for x > 0
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let positive_mask = input.gt(&zero_tensor)?;

    // Gradient w.r.t. input
    let one_tensor = Tensor::ones(input.shape().dims());
    let grad_input_mask = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, alpha)?;
    let grad_input = grad_output.mul(&grad_input_mask)?;

    // Gradient w.r.t. alpha: 0 where x > 0, x where x <= 0
    let grad_alpha_intermediate =
        tenflowers_core::ops::where_op(&positive_mask, &zero_tensor, input)?;
    let grad_alpha = grad_output.mul(&grad_alpha_intermediate)?;

    Ok((grad_input, grad_alpha))
}
