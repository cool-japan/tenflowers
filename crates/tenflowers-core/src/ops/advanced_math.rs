//! Advanced mathematical operations for deep learning
//!
//! This module provides advanced mathematical operations commonly used in deep learning
//! including numerically stable implementations of various activation functions,
//! special functions, and utility operations.

use crate::{Result, Tensor};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::Float;
use std::ops::{Add, Div, Mul, Sub};

// Note: logsumexp requires axis handling that doesn't match current API
// TODO: Implement logsumexp when axis handling is unified

/// Softplus activation: log(1 + exp(x))
///
/// Numerically stable implementation using the identity:
/// softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing softplus(input)
pub fn softplus<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    // Compute |x|
    let abs_x = crate::ops::abs(input)?;

    // Compute max(x, 0)
    let max_x_0 = crate::ops::activation::relu(input)?;

    // Compute exp(-|x|)
    let neg_abs = crate::ops::neg(&abs_x)?;
    let exp_neg_abs = crate::ops::exp(&neg_abs)?;

    // Compute log(1 + exp(-|x|))
    let one = Tensor::ones(input.shape().dims());
    let one_plus_exp = crate::ops::binary::add(&one, &exp_neg_abs)?;
    let log_term = crate::ops::log(&one_plus_exp)?;

    // Combine: log(1 + exp(-|x|)) + max(x, 0)
    crate::ops::binary::add(&log_term, &max_x_0)
}

/// Softsign activation: x / (1 + |x|)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing x / (1 + |x|)
pub fn softsign<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Div<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let abs_x = crate::ops::abs(input)?;
    let one = Tensor::ones(input.shape().dims());
    let denominator = crate::ops::binary::add(&one, &abs_x)?;
    crate::ops::binary::div(input, &denominator)
}

/// Mish activation: x * tanh(softplus(x))
///
/// Mish is a smooth, non-monotonic activation function.
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing x * tanh(softplus(x))
pub fn mish<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let sp = softplus(input)?;
    let tanh_sp = crate::ops::tanh(&sp)?;
    crate::ops::binary::mul(input, &tanh_sp)
}

/// Hard sigmoid activation: clamp((x + 3) / 6, 0, 1)
///
/// Computationally efficient approximation of sigmoid.
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing hard_sigmoid(input)
pub fn hard_sigmoid<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Div<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    // Compute (x + 3) / 6
    let three = Tensor::full(input.shape().dims(), T::from(3.0).unwrap());
    let six = Tensor::full(input.shape().dims(), T::from(6.0).unwrap());

    let x_plus_3 = crate::ops::binary::add(input, &three)?;
    let scaled = crate::ops::binary::div(&x_plus_3, &six)?;

    // Clamp to [0, 1]
    scaled.clamp(T::from(0.0).unwrap(), T::from(1.0).unwrap())
}

/// Hard swish activation: x * hard_sigmoid(x)
///
/// Computationally efficient approximation of swish/SiLU.
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing x * hard_sigmoid(x)
pub fn hard_swish<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let hs = hard_sigmoid(input)?;
    crate::ops::binary::mul(input, &hs)
}

/// Log-sigmoid: log(sigmoid(x))
///
/// Numerically stable implementation: log_sigmoid(x) = -softplus(-x)
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing log(sigmoid(input))
pub fn log_sigmoid<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let neg_x = crate::ops::neg(input)?;
    let sp = softplus(&neg_x)?;
    crate::ops::neg(&sp)
}

/// GELU activation with tanh approximation
///
/// Approximates: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing GELU(input) approximation
pub fn gelu_tanh<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let half = Tensor::full(input.shape().dims(), T::from(0.5).unwrap());
    let one = Tensor::ones(input.shape().dims());

    // Compute x^3
    let x_squared = crate::ops::binary::mul(input, input)?;
    let x_cubed = crate::ops::binary::mul(&x_squared, input)?;

    // Compute 0.044715 * x^3
    let coef = Tensor::full(input.shape().dims(), T::from(0.044715).unwrap());
    let term = crate::ops::binary::mul(&coef, &x_cubed)?;

    // Compute x + 0.044715 * x^3
    let sum = crate::ops::binary::add(input, &term)?;

    // Compute sqrt(2/π) ≈ 0.7978845608
    let sqrt_2_pi = Tensor::full(input.shape().dims(), T::from(0.7978845608).unwrap());
    let scaled = crate::ops::binary::mul(&sqrt_2_pi, &sum)?;

    // Compute tanh(...)
    let tanh_val = crate::ops::tanh(&scaled)?;

    // Compute 1 + tanh(...)
    let one_plus_tanh = crate::ops::binary::add(&one, &tanh_val)?;

    // Compute 0.5 * x * (1 + tanh(...))
    let half_x = crate::ops::binary::mul(&half, input)?;
    crate::ops::binary::mul(&half_x, &one_plus_tanh)
}

/// Logit function (inverse of sigmoid)
///
/// logit(p) = log(p / (1 - p))
///
/// # Arguments
/// * `input` - Input tensor (should be in range (0, 1))
/// * `eps` - Small epsilon to clip values away from 0 and 1 for numerical stability
///
/// # Returns
/// Result containing logit(input)
pub fn logit<T>(input: &Tensor<T>, eps: T) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable,
{
    // Clamp input to (eps, 1-eps)
    let one_minus_eps = T::from(1.0).unwrap() - eps;
    let clipped = input.clamp(eps, one_minus_eps)?;

    // Compute 1 - p
    let one = Tensor::ones(clipped.shape().dims());
    let one_minus_p = crate::ops::binary::sub(&one, &clipped)?;

    // Compute p / (1 - p)
    let ratio = crate::ops::binary::div(&clipped, &one_minus_p)?;

    // Take log
    crate::ops::log(&ratio)
}

/// Expit function (same as sigmoid, included for clarity)
///
/// expit(x) = 1 / (1 + exp(-x))
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing sigmoid(input)
pub fn expit<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Div<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    crate::ops::activation::sigmoid(input)
}

/// Scaled exponential linear unit (SELU)
///
/// SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
/// where scale ≈ 1.0507 and alpha ≈ 1.67326
///
/// # Arguments
/// * `input` - Input tensor
///
/// # Returns
/// Result containing SELU(input)
pub fn selu<T>(input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + 'static
        + Pod
        + Zeroable
        + scirs2_core::Signed,
{
    let scale = T::from(1.050_700_987_355_480_5).unwrap();
    let alpha = T::from(1.673_263_242_354_377_2).unwrap();

    // ELU part
    let elu = crate::ops::activation::elu(input, alpha)?;

    // Scale
    let scale_tensor = Tensor::full(elu.shape().dims(), scale);
    crate::ops::binary::mul(&scale_tensor, &elu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    // TODO: Uncomment when logsumexp is implemented
    // #[test]
    // fn test_logsumexp() {
    //     let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    //     let result = logsumexp(&input, None, false).unwrap();
    //     let result_val = result.to_vec().unwrap()[0];
    //
    //     // Should be approximately log(e^1 + e^2 + e^3 + e^4) ≈ 4.44019
    //     assert!((result_val - 4.44019).abs() < 0.001, "logsumexp mismatch: {}", result_val);
    // }

    #[test]
    fn test_softplus() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0, 10.0], &[4]).unwrap();
        let result = softplus(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        // softplus(0) ≈ 0.693
        assert!((result_data[0] - 0.693).abs() < 0.01);
        // softplus(10) ≈ 10 (for large positive values)
        assert!((result_data[3] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_softsign() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0, 2.0], &[4]).unwrap();
        let result = softsign(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.5).abs() < 1e-6);
        assert!((result_data[2] - (-0.5)).abs() < 1e-6);
        assert!((result_data[3] - 0.666666).abs() < 0.001);
    }

    #[test]
    fn test_mish() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]).unwrap();
        let result = mish(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        // Mish(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // Mish(1) > 0.8
        assert!(result_data[1] > 0.8);
    }

    #[test]
    fn test_hard_sigmoid() {
        let input = Tensor::from_vec(vec![-3.0_f32, 0.0, 3.0, 6.0], &[4]).unwrap();
        let result = hard_sigmoid(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.5).abs() < 1e-6);
        assert!((result_data[2] - 1.0).abs() < 1e-6);
        assert!((result_data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hard_swish() {
        let input = Tensor::from_vec(vec![-3.0_f32, 0.0, 3.0], &[3]).unwrap();
        let result = hard_swish(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.0).abs() < 1e-6);
        assert!((result_data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_sigmoid() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]).unwrap();
        let result = log_sigmoid(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        // log(sigmoid(0)) = log(0.5) ≈ -0.693
        assert!((result_data[0] - (-0.693)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_tanh() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]).unwrap();
        let result = gelu_tanh(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        // GELU(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // GELU(1) ≈ 0.84
        assert!((result_data[1] - 0.84).abs() < 0.05);
    }

    #[test]
    fn test_logit() {
        let input = Tensor::from_vec(vec![0.5_f32, 0.75, 0.25], &[3]).unwrap();
        let result = logit(&input, 1e-7).unwrap();
        let result_data = result.to_vec().unwrap();

        // logit(0.5) = 0
        assert!(result_data[0].abs() < 1e-6);
        // logit(0.75) = log(3) ≈ 1.099
        assert!((result_data[1] - 1.099).abs() < 0.01);
        // logit(0.25) = -log(3) ≈ -1.099
        assert!((result_data[2] - (-1.099)).abs() < 0.01);
    }

    #[test]
    fn test_selu() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3]).unwrap();
        let result = selu(&input).unwrap();
        let result_data = result.to_vec().unwrap();

        // SELU(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // SELU(1) ≈ 1.0507
        assert!((result_data[1] - 1.0507).abs() < 0.01);
    }

    // TODO: Uncomment when logsumexp is implemented
    // #[test]
    // fn test_numerical_stability_logsumexp() {
    //     // Test with large values that would cause overflow without stability
    //     let input = Tensor::from_vec(vec![100.0_f32, 101.0, 102.0], &[3]).unwrap();
    //     let result = logsumexp(&input, None, false);
    //     assert!(result.is_ok());
    //
    //     let result_val = result.unwrap().to_vec().unwrap()[0];
    //     // Should be close to 102 + log(e^(-2) + e^(-1) + 1)
    //     assert!(result_val.is_finite());
    //     assert!(result_val > 102.0 && result_val < 103.0);
    // }
}
