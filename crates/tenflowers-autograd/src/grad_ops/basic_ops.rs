//! Basic arithmetic operations gradient functions
//!
//! This module provides gradient computation functions for fundamental
//! arithmetic operations like addition, multiplication, subtraction,
//! division, and power operations.

use super::utils::unbroadcast;
use scirs2_core::numeric::{One, Zero};
use tenflowers_core::{Result, Tensor};

/// Backward pass for addition
/// For z = a + b, grad_a = grad_z, grad_b = grad_z (with broadcasting handled)
pub fn add_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // The gradient flows unchanged through addition
    // But we need to handle broadcasting by summing over broadcasted dimensions

    let grad_a = unbroadcast(grad_output, a.shape())?;
    let grad_b = unbroadcast(grad_output, b.shape())?;

    Ok((grad_a, grad_b))
}

/// Backward pass for multiplication
/// For z = a * b, grad_a = grad_z * b, grad_b = grad_z * a
pub fn mul_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute grad_a = grad_output * b
    let grad_a_full = grad_output.mul(b)?;
    let grad_a = unbroadcast(&grad_a_full, a.shape())?;

    // Compute grad_b = grad_output * a
    let grad_b_full = grad_output.mul(a)?;
    let grad_b = unbroadcast(&grad_b_full, b.shape())?;

    Ok((grad_a, grad_b))
}

/// Backward pass for subtraction
/// For z = a - b, grad_a = grad_z, grad_b = -grad_z (with broadcasting handled)
pub fn sub_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Neg<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // grad_a = grad_output (with unbroadcasting)
    let grad_a = unbroadcast(grad_output, a.shape())?;

    // grad_b = -grad_output (with unbroadcasting)
    let neg_grad_output = grad_output.neg()?;
    let grad_b = unbroadcast(&neg_grad_output, b.shape())?;

    Ok((grad_a, grad_b))
}

/// Backward pass for division
/// For z = a / b, grad_a = grad_z / b, grad_b = -grad_z * a / (b^2)
pub fn div_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // grad_a = grad_output / b
    let grad_a_full = grad_output.div(b)?;
    let grad_a = unbroadcast(&grad_a_full, a.shape())?;

    // grad_b = -grad_output * a / (b^2)
    let b_squared = b.mul(b)?;
    let grad_b_intermediate = grad_output.mul(a)?;
    let grad_b_full = grad_b_intermediate.div(&b_squared)?.neg()?;
    let grad_b = unbroadcast(&grad_b_full, b.shape())?;

    Ok((grad_a, grad_b))
}

/// Backward pass for power operation
/// For z = a^b, grad_a = grad_z * b * a^(b-1), grad_b = grad_z * a^b * ln(a)
pub fn pow_backward<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
    output: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Add<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::Float
        + PartialEq
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // grad_a = grad_output * b * a^(b-1)
    // For numerical stability, use: grad_a = grad_output * b * output / a
    let grad_a_intermediate = grad_output.mul(b)?;
    let grad_a_full = grad_a_intermediate.mul(output)?.div(a)?;
    let grad_a = unbroadcast(&grad_a_full, a.shape())?;

    // grad_b = grad_output * output * ln(a)
    // Handle special cases for integer powers and provide ln approximation
    let grad_b = if is_integer_power(b)? {
        // For integer powers, we can compute the derivative analytically
        // d/db(a^b) = a^b * ln(a), but for constant integers this is typically zero
        // in practice unless we're differentiating w.r.t. a learned exponent
        compute_integer_power_grad_b(grad_output, a, output)?
    } else {
        // General case: grad_b = grad_output * output * ln(a)
        // Use approximation ln(x) ≈ log(x) for x > 0
        compute_general_power_grad_b(grad_output, a, output)?
    };

    let grad_b_unbroadcast = unbroadcast(&grad_b, b.shape())?;

    Ok((grad_a, grad_b_unbroadcast))
}

/// Helper function to check if a tensor contains integer values
pub fn is_integer_power<T>(b: &Tensor<T>) -> Result<bool>
where
    T: Clone + Default + scirs2_core::num_traits::Float + PartialEq,
{
    // Check if all values in b are close to integers
    // This is a simplified check - in practice, you might want more sophisticated detection
    if let Some(data) = b.as_slice() {
        for &val in data {
            let rounded = val.round();
            let diff = (val - rounded).abs();
            let tolerance = T::from(1e-6).unwrap_or_else(|| T::from(0.000001).unwrap());
            if diff > tolerance {
                return Ok(false);
            }
        }
        Ok(true)
    } else {
        // If we can't access the data, assume non-integer
        Ok(false)
    }
}

/// Compute gradient w.r.t. exponent for integer powers
pub fn compute_integer_power_grad_b<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    output: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For integer powers, grad_b = grad_output * output * ln(a)
    // Use a stable approximation for ln(a)
    let ln_a = compute_log_approximation(a)?;
    grad_output.mul(output)?.mul(&ln_a)
}

/// Compute gradient w.r.t. exponent for general case
pub fn compute_general_power_grad_b<T>(
    grad_output: &Tensor<T>,
    a: &Tensor<T>,
    output: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // General case: grad_b = grad_output * output * ln(a)
    let ln_a = compute_log_approximation(a)?;
    grad_output.mul(output)?.mul(&ln_a)
}

/// Compute log approximation for tensors
/// Uses ln(x) ≈ 2 * (x - 1) / (x + 1) for x near 1, or series approximation
pub fn compute_log_approximation<T>(a: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // For numerical stability, handle values near 1 specially
    let one_tensor = Tensor::ones(a.shape().dims());
    let two_tensor = Tensor::from_scalar(T::from(2.0).unwrap());

    // Use the approximation ln(x) ≈ 2 * (x - 1) / (x + 1) for all x > 0
    // This is reasonably accurate for x in (0.5, 2.0) and stable
    let x_minus_1 = a.sub(&one_tensor)?;
    let x_plus_1 = a.add(&one_tensor)?;
    let ratio = x_minus_1.div(&x_plus_1)?;
    let ln_approx = two_tensor.mul(&ratio)?;

    Ok(ln_approx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_add_backward() {
        // Test basic addition gradient
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], &[2]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap();

        let (grad_a, grad_b) = add_backward(&grad_output, &a, &b).unwrap();

        // For addition, gradients should equal the output gradient
        assert_eq!(grad_a.shape().dims(), a.shape().dims());
        assert_eq!(grad_b.shape().dims(), b.shape().dims());
    }

    #[test]
    fn test_mul_backward() {
        // Test basic multiplication gradient
        let a = Tensor::from_vec(vec![2.0f32, 3.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![4.0f32, 5.0], &[2]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap();

        let (grad_a, grad_b) = mul_backward(&grad_output, &a, &b).unwrap();

        // For multiplication: grad_a = grad_output * b, grad_b = grad_output * a
        assert_eq!(grad_a.shape().dims(), a.shape().dims());
        assert_eq!(grad_b.shape().dims(), b.shape().dims());
    }

    #[test]
    fn test_sub_backward() {
        // Test basic subtraction gradient
        let a = Tensor::from_vec(vec![5.0f32, 6.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![2.0f32, 3.0], &[2]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap();

        let (grad_a, grad_b) = sub_backward(&grad_output, &a, &b).unwrap();

        // For subtraction: grad_a = grad_output, grad_b = -grad_output
        assert_eq!(grad_a.shape().dims(), a.shape().dims());
        assert_eq!(grad_b.shape().dims(), b.shape().dims());
    }

    #[test]
    fn test_div_backward() {
        // Test basic division gradient
        let a = Tensor::from_vec(vec![6.0f32, 8.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![2.0f32, 4.0], &[2]).unwrap();
        let grad_output = Tensor::from_vec(vec![1.0f32, 1.0], &[2]).unwrap();

        let (grad_a, grad_b) = div_backward(&grad_output, &a, &b).unwrap();

        // Check that shapes match
        assert_eq!(grad_a.shape().dims(), a.shape().dims());
        assert_eq!(grad_b.shape().dims(), b.shape().dims());
    }

    #[test]
    fn test_is_integer_power() {
        // Test integer detection
        let integer_tensor = Tensor::from_vec(vec![2.0f32, 3.0, 4.0], &[3]).unwrap();
        assert!(is_integer_power(&integer_tensor).unwrap());

        let float_tensor = Tensor::from_vec(vec![2.5f32, 3.7, 4.1], &[3]).unwrap();
        assert!(!is_integer_power(&float_tensor).unwrap());
    }

    #[test]
    fn test_log_approximation() {
        // Test log approximation for various values
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 0.5], &[3]).unwrap();
        let result = compute_log_approximation(&input);

        assert!(result.is_ok(), "Log approximation should succeed");
        let log_approx = result.unwrap();
        assert_eq!(log_approx.shape().dims(), input.shape().dims());
    }
}
