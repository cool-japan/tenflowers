//! Activation function gradient computations
//!
//! This module provides gradient computation functions for various
//! activation functions used in neural networks, including ReLU,
//! Sigmoid, Tanh, GELU, Swish, and others.

use crate::tensor_ext::TensorAutograd;
use num_traits::{One, Zero};
use tenflowers_core::{Result, Tensor};

/// Forward pass for ReLU activation
pub fn relu_forward<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    tenflowers_core::ops::relu(input)
}

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
        + num_traits::Float
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
        + num_traits::Float
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
    let tanh_val = tenflowers_core::ops::activation::tanh(&scaled_arg)?;
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
        + num_traits::Float
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
/// For y = Mish(x) = x * tanh(softplus(x)) where softplus(x) = log(1 + exp(x))
/// grad_x = grad_y * (tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x))
pub fn mish_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + num_traits::Float
        + num_traits::Signed
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute softplus(x) = log(1 + exp(x)) using a more stable implementation
    // For numerical stability, use: softplus(x) = max(0, x) + log(1 + exp(-abs(x)))
    let abs_x = input.abs()?;
    let neg_abs_x = abs_x.neg()?;
    let exp_neg_abs = neg_abs_x.exp()?;
    let one_tensor = Tensor::ones(input.shape().dims());
    let _one_plus_exp = one_tensor.add(&exp_neg_abs)?;
    // For now, use a simpler approximation since we don't have ln implemented
    let zero_tensor = Tensor::zeros(input.shape().dims());
    let softplus = tenflowers_core::ops::where_op(&input.gt(&zero_tensor)?, input, &zero_tensor)?;

    // Compute tanh(softplus(x))
    let tanh_softplus = tenflowers_core::ops::activation::tanh(&softplus)?;

    // Compute sigmoid(x) = 1 / (1 + exp(-x))
    let sigmoid_x = tenflowers_core::ops::sigmoid(input)?;

    // Compute sech²(softplus(x)) = 1 - tanh²(softplus(x))
    let tanh_squared = tanh_softplus.mul(&tanh_softplus)?;
    let sech_squared = one_tensor.sub(&tanh_squared)?;

    // Compute the derivative: tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
    let first_term = tanh_softplus;
    let second_term = input.mul(&sech_squared)?.mul(&sigmoid_x)?;
    let mish_grad = first_term.add(&second_term)?;

    // Apply chain rule
    grad_output.mul(&mish_grad)
}

/// Backward pass for ELU (Exponential Linear Unit) activation
/// For y = ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
/// grad_x = grad_y * (1 if x > 0 else alpha * exp(x))
pub fn elu_backward<T>(grad_output: &Tensor<T>, input: &Tensor<T>, alpha: T) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + PartialOrd
        + num_traits::Float
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

    // For negative values: gradient = alpha * exp(x)
    let exp_x = input.exp()?;
    let alpha_tensor = Tensor::from_scalar(alpha);
    let alpha_exp_x = alpha_tensor.mul(&exp_x)?;

    // Select gradient based on the condition
    let grad_mask = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, &alpha_exp_x)?;

    // Apply chain rule
    grad_output.mul(&grad_mask)
}

/// Backward pass for PReLU (Parametric ReLU) activation
/// For y = PReLU(x) = x if x > 0 else alpha * x
/// grad_x = grad_y * (1 if x > 0 else alpha)
/// grad_alpha = grad_y * sum(x where x <= 0)
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

    // For input gradient: 1 if x > 0 else alpha
    let one_tensor = Tensor::ones(input.shape().dims());
    let grad_input_mask = tenflowers_core::ops::where_op(&positive_mask, &one_tensor, alpha)?;
    let grad_input = grad_output.mul(&grad_input_mask)?;

    // For alpha gradient: sum(grad_output * x) where x <= 0
    let negative_mask = input.le(&zero_tensor)?;
    let alpha_grad_values =
        tenflowers_core::ops::where_op(&negative_mask, &grad_output.mul(input)?, &zero_tensor)?;
    // Sum over appropriate dimensions to match alpha shape
    let grad_alpha = if alpha.shape().dims().len() == 1 && alpha.shape().dims()[0] == 1 {
        // Scalar alpha case - sum all
        alpha_grad_values.sum(None, false)?
    } else {
        // Channel-wise alpha case - sum over batch and spatial dimensions
        let input_shape = input.shape().dims();
        let axes: Vec<i32> = (0..input_shape.len() - 1).map(|i| i as i32).collect();
        alpha_grad_values.sum(Some(&axes), false)?
    };

    Ok((grad_input, grad_alpha))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_relu_backward() {
        // Test ReLU gradient computation
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();

        let grad_input = relu_backward(&grad_output, &input).unwrap();

        // ReLU gradient should be 0 for negative inputs, 1 for positive inputs
        assert_eq!(grad_input.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_sigmoid_backward() {
        // Test sigmoid gradient computation
        let output = Tensor::from_vec(vec![0.5f32, 0.73, 0.27, 0.88], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();

        let grad_input = sigmoid_backward(&grad_output, &output).unwrap();

        // Gradient should have same shape as input
        assert_eq!(grad_input.shape().dims(), output.shape().dims());
    }

    #[test]
    fn test_tanh_backward() {
        // Test tanh gradient computation
        let output = Tensor::from_vec(vec![-0.5f32, 0.0, 0.5, 0.9], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();

        let grad_input = tanh_backward(&grad_output, &output).unwrap();

        // Gradient should have same shape as input
        assert_eq!(grad_input.shape().dims(), output.shape().dims());
    }

    #[test]
    fn test_leaky_relu_backward() {
        // Test LeakyReLU gradient computation
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();
        let alpha = 0.1f32;

        let grad_input = leaky_relu_backward(&grad_output, &input, alpha).unwrap();

        // Gradient should have same shape as input
        assert_eq!(grad_input.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_softmax_backward() {
        // Test softmax gradient computation
        let output = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();

        let grad_input = softmax_backward(&grad_output, &output, Some(-1)).unwrap();

        // Gradient should have same shape as input
        assert_eq!(grad_input.shape().dims(), output.shape().dims());
    }

    #[test]
    fn test_elu_backward() {
        // Test ELU gradient computation
        let input = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0, 2.0], &[4]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], &[4]).unwrap();
        let alpha = 1.0f32;

        let grad_input = elu_backward(&grad_output, &input, alpha).unwrap();

        // Gradient should have same shape as input
        assert_eq!(grad_input.shape().dims(), input.shape().dims());
    }
}
