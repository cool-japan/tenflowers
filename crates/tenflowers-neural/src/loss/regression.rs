use num_traits::{Float, FromPrimitive, One, Signed, Zero};
use tenflowers_core::{Result, Tensor};

use super::utils::convert_u8_to_bool_tensor;

/// Mean Squared Error (L2) Loss
/// MSE(y, pred) = mean((y - pred)^2)
pub fn mse<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let diff = predictions.sub(targets)?;
    let squared = diff.mul(&diff)?;
    tenflowers_core::ops::mean(&squared, None, false)
}

/// L1 (Mean Absolute Error) Loss
/// L1(y, pred) = mean(|y - pred|)
pub fn l1_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "l1_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let diff = targets.sub(predictions)?;
    let abs_diff = diff.abs()?;

    // Return mean of absolute differences
    tenflowers_core::ops::mean(&abs_diff, None, false)
}

/// SmoothL1 Loss (also known as Huber Loss with delta=1)
/// SmoothL1(x) = 0.5 * x^2 if |x| < 1, |x| - 0.5 otherwise
/// where x = pred - target
pub fn smooth_l1_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "smooth_l1_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let diff = predictions.sub(targets)?;
    let abs_diff = diff.abs()?;

    let one = Tensor::from_scalar(T::one());
    let half = Tensor::from_scalar(T::from(0.5).unwrap());

    // Create mask for |diff| < 1
    let mask = abs_diff.lt(&one)?;

    // Quadratic part: 0.5 * diff^2
    let squared_loss = half.mul(&diff)?.mul(&diff)?;

    // Linear part: |diff| - 0.5
    let linear_loss = abs_diff.sub(&half)?;

    // Use where_op to select between quadratic and linear
    let loss = tenflowers_core::ops::where_op(&mask, &squared_loss, &linear_loss)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}

/// Huber Loss (robust regression loss)
/// Huber(y, pred, delta) = 0.5 * (y - pred)^2 if |y - pred| <= delta,
///                         delta * (|y - pred| - 0.5 * delta) otherwise
pub fn huber_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>, delta: T) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + std::cmp::PartialOrd
        + Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Huber loss:
    // - 0.5 * (y - pred)^2 if |y - pred| <= delta
    // - delta * (|y - pred| - 0.5 * delta) otherwise

    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "huber_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let diff = targets.sub(predictions)?;
    let abs_diff = diff.abs()?;

    let delta_tensor = Tensor::from_scalar(delta);
    let half = Tensor::from_scalar(T::from(0.5).unwrap());

    // Create mask for |diff| <= delta
    let mask = abs_diff.le(&delta_tensor)?;

    // Quadratic part: 0.5 * diff^2
    let squared_loss = half.mul(&diff)?.mul(&diff)?;

    // Linear part: delta * (|diff| - 0.5 * delta)
    let linear_part = abs_diff.sub(&half.mul(&delta_tensor)?)?;
    let linear_loss = delta_tensor.mul(&linear_part)?;

    // Use where_op to select between quadratic and linear
    let loss = tenflowers_core::ops::where_op(&mask, &squared_loss, &linear_loss)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}

/// Quantile Loss (Pinball Loss) for quantile regression
/// QuantileLoss(y, pred, q) = max(q * (y - pred), (q - 1) * (y - pred))
pub fn quantile_loss<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
    quantile: T,
) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::cmp::PartialOrd
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Quantile loss (pinball loss):
    // max(quantile * (target - pred), (quantile - 1) * (target - pred))

    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "quantile_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let diff = targets.sub(predictions)?; // target - pred
    let quantile_tensor = Tensor::from_scalar(quantile);
    let one = Tensor::from_scalar(T::one());

    // Compute both terms
    let term1 = quantile_tensor.mul(&diff)?; // quantile * (target - pred)
    let quantile_minus_one = quantile_tensor.sub(&one)?;
    let term2 = quantile_minus_one.mul(&diff)?; // (quantile - 1) * (target - pred)

    // Take maximum of the two terms using comparison operations
    let mask_u8 = tenflowers_core::ops::gt(&term1, &term2)?;
    let mask = convert_u8_to_bool_tensor(&mask_u8)?;
    let loss = tenflowers_core::ops::where_op(&mask, &term1, &term2)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}
