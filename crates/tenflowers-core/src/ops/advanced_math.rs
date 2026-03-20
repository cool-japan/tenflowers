//! Advanced mathematical operations for deep learning
//!
//! This module provides advanced mathematical operations commonly used in deep learning
//! including numerically stable implementations of various activation functions,
//! special functions, and utility operations.

use crate::{Result, Tensor};
use bytemuck::{Pod, Zeroable};
use scirs2_core::numeric::Float;
use std::ops::{Add, Div, Mul, Sub};

/// Helper macro to convert numeric constants without unwrap (no unwrap policy)
macro_rules! float_const {
    ($val:expr, $t:ty) => {
        <$t as scirs2_core::num_traits::NumCast>::from($val)
            .expect("float constant conversion should never fail for standard float types")
    };
}

/// Log-sum-exp: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
///
/// Numerically stable implementation that prevents overflow by subtracting the max
/// before exponentiation.
///
/// # Arguments
/// * `input` - Input tensor
/// * `axes` - Optional axes along which to reduce. `None` reduces all axes.
/// * `keepdims` - Whether to keep reduced dimensions as size 1
///
/// # Returns
/// Result containing logsumexp(input) along the specified axes
pub fn logsumexp<T>(input: &Tensor<T>, axes: Option<&[i32]>, keepdims: bool) -> Result<Tensor<T>>
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
        + scirs2_core::num_traits::Zero,
{
    // Step 1: Compute max(x) along the specified axes, keeping dims for broadcasting
    let max_val = crate::ops::reduction::statistical::max(input, axes, true)?;

    // Step 2: Compute x - max(x) (broadcasting handles shape mismatch)
    let shifted = crate::ops::binary::sub(input, &max_val)?;

    // Step 3: Compute exp(x - max(x))
    let exp_shifted = crate::ops::exp(&shifted)?;

    // Step 4: Compute sum(exp(x - max(x))) along the same axes
    let sum_exp = crate::ops::reduction::statistical::sum(&exp_shifted, axes, keepdims)?;

    // Step 5: Compute log(sum(exp(x - max(x))))
    let log_sum = crate::ops::log(&sum_exp)?;

    // Step 6: Get max_val with the correct keepdims setting
    let max_final = if keepdims {
        max_val
    } else {
        crate::ops::reduction::statistical::max(input, axes, false)?
    };

    // Step 7: max(x) + log(sum(exp(x - max(x))))
    crate::ops::binary::add(&max_final, &log_sum)
}

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
    let three = Tensor::full(input.shape().dims(), float_const!(3.0, T));
    let six = Tensor::full(input.shape().dims(), float_const!(6.0, T));

    let x_plus_3 = crate::ops::binary::add(input, &three)?;
    let scaled = crate::ops::binary::div(&x_plus_3, &six)?;

    // Clamp to [0, 1]
    scaled.clamp(float_const!(0.0, T), float_const!(1.0, T))
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
    let half = Tensor::full(input.shape().dims(), float_const!(0.5, T));
    let one = Tensor::ones(input.shape().dims());

    // Compute x^3
    let x_squared = crate::ops::binary::mul(input, input)?;
    let x_cubed = crate::ops::binary::mul(&x_squared, input)?;

    // Compute 0.044715 * x^3
    let coef = Tensor::full(input.shape().dims(), float_const!(0.044715, T));
    let term = crate::ops::binary::mul(&coef, &x_cubed)?;

    // Compute x + 0.044715 * x^3
    let sum = crate::ops::binary::add(input, &term)?;

    // Compute sqrt(2/π) ≈ 0.7978845608
    let sqrt_2_pi = Tensor::full(input.shape().dims(), float_const!(0.7978845608, T));
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
    let one_minus_eps = float_const!(1.0, T) - eps;
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
    let scale = float_const!(1.050_700_987_355_480_5, T);
    let alpha = float_const!(1.673_263_242_354_377_2, T);

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

    #[test]
    fn test_logsumexp() {
        let input = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let result = logsumexp(&input, None, false).expect("test: logsumexp should succeed");
        let result_val = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec")[0];

        // Should be approximately log(e^1 + e^2 + e^3 + e^4) ≈ 4.44019
        assert!(
            (result_val - 4.44019).abs() < 0.001,
            "logsumexp mismatch: {}",
            result_val
        );
    }

    #[test]
    fn test_softplus() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0, 10.0], &[4])
            .expect("test: from_vec should succeed");
        let result = softplus(&input).expect("test: softplus should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // softplus(0) ≈ 0.693
        assert!((result_data[0] - 0.693).abs() < 0.01);
        // softplus(10) ≈ 10 (for large positive values)
        assert!((result_data[3] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_softsign() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0, 2.0], &[4])
            .expect("test: from_vec should succeed");
        let result = softsign(&input).expect("test: softsign should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.5).abs() < 1e-6);
        assert!((result_data[2] - (-0.5)).abs() < 1e-6);
        assert!((result_data[3] - 0.666666).abs() < 0.001);
    }

    #[test]
    fn test_mish() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3])
            .expect("test: from_vec should succeed");
        let result = mish(&input).expect("test: mish should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // Mish(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // Mish(1) > 0.8
        assert!(result_data[1] > 0.8);
    }

    #[test]
    fn test_hard_sigmoid() {
        let input = Tensor::from_vec(vec![-3.0_f32, 0.0, 3.0, 6.0], &[4])
            .expect("test: from_vec should succeed");
        let result = hard_sigmoid(&input).expect("test: hard_sigmoid should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.5).abs() < 1e-6);
        assert!((result_data[2] - 1.0).abs() < 1e-6);
        assert!((result_data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hard_swish() {
        let input = Tensor::from_vec(vec![-3.0_f32, 0.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let result = hard_swish(&input).expect("test: hard_swish should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        assert!((result_data[0] - 0.0).abs() < 1e-6);
        assert!((result_data[1] - 0.0).abs() < 1e-6);
        assert!((result_data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_sigmoid() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3])
            .expect("test: from_vec should succeed");
        let result = log_sigmoid(&input).expect("test: log_sigmoid should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // log(sigmoid(0)) = log(0.5) ≈ -0.693
        assert!((result_data[0] - (-0.693)).abs() < 0.01);
    }

    #[test]
    fn test_gelu_tanh() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3])
            .expect("test: from_vec should succeed");
        let result = gelu_tanh(&input).expect("test: gelu_tanh should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // GELU(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // GELU(1) ≈ 0.84
        assert!((result_data[1] - 0.84).abs() < 0.05);
    }

    #[test]
    fn test_logit() {
        let input = Tensor::from_vec(vec![0.5_f32, 0.75, 0.25], &[3])
            .expect("test: from_vec should succeed");
        let result = logit(&input, 1e-7).expect("test: logit should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // logit(0.5) = 0
        assert!(result_data[0].abs() < 1e-6);
        // logit(0.75) = log(3) ≈ 1.099
        assert!((result_data[1] - 1.099).abs() < 0.01);
        // logit(0.25) = -log(3) ≈ -1.099
        assert!((result_data[2] - (-1.099)).abs() < 0.01);
    }

    #[test]
    fn test_selu() {
        let input = Tensor::from_vec(vec![0.0_f32, 1.0, -1.0], &[3])
            .expect("test: from_vec should succeed");
        let result = selu(&input).expect("test: selu should succeed");
        let result_data = result
            .to_vec()
            .expect("test: tensor data should be convertible to vec");

        // SELU(0) ≈ 0
        assert!(result_data[0].abs() < 0.01);
        // SELU(1) ≈ 1.0507
        assert!((result_data[1] - 1.0507).abs() < 0.01);
    }

    #[test]
    fn test_numerical_stability_logsumexp() {
        // Test with large values that would cause overflow without stability
        let input = Tensor::from_vec(vec![100.0_f32, 101.0, 102.0], &[3])
            .expect("test: from_vec should succeed");
        let result = logsumexp(&input, None, false);
        assert!(result.is_ok());

        let result_val = result
            .expect("test: operation should succeed")
            .to_vec()
            .expect("test: tensor data should be convertible to vec")[0];
        // Should be close to 102 + log(e^(-2) + e^(-1) + 1)
        assert!(result_val.is_finite());
        assert!(result_val > 102.0 && result_val < 103.0);
    }
}
