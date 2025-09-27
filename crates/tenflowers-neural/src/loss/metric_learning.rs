use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Triplet Loss for metric learning
/// L(a, p, n) = max(0, ||a - p||² - ||a - n||² + margin)
pub fn triplet_loss<T>(
    anchor: &Tensor<T>,
    positive: &Tensor<T>,
    negative: &Tensor<T>,
    margin: T,
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
    // Check that all tensors have the same shape
    if anchor.shape() != positive.shape() || anchor.shape() != negative.shape() {
        return Err(tenflowers_core::error::TensorError::InvalidArgument {
            operation: "triplet_loss".to_string(),
            reason: "Anchor, positive, and negative tensors must have the same shape".to_string(),
            context: None,
        });
    }

    // Compute ||a - p||²
    let ap_diff = anchor.sub(positive)?;
    let ap_squared = ap_diff.mul(&ap_diff)?;
    let ap_dist = tenflowers_core::ops::sum(&ap_squared, Some(&[-1]), false)?;

    // Compute ||a - n||²
    let an_diff = anchor.sub(negative)?;
    let an_squared = an_diff.mul(&an_diff)?;
    let an_dist = tenflowers_core::ops::sum(&an_squared, Some(&[-1]), false)?;

    // Compute ap_dist - an_dist + margin
    let margin_tensor = Tensor::from_scalar(margin);
    let diff = ap_dist.sub(&an_dist)?;
    let loss_raw = diff.add(&margin_tensor)?;

    // Apply max(0, loss_raw) - equivalent to ReLU
    let loss = tenflowers_core::ops::relu(&loss_raw)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}

/// Contrastive Loss for siamese networks
/// L(y, d) = (1-y) * 0.5 * d² + y * 0.5 * max(0, margin - d)²
/// where y = 1 if dissimilar, y = 0 if similar, d = distance
pub fn contrastive_loss<T>(distance: &Tensor<T>, labels: &Tensor<T>, margin: T) -> Result<Tensor<T>>
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
    if distance.shape() != labels.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "contrastive_loss",
            &labels.shape().to_string(),
            &distance.shape().to_string(),
        ));
    }

    let one = Tensor::from_scalar(T::one());
    let half = Tensor::from_scalar(T::from(0.5).unwrap());
    let margin_tensor = Tensor::from_scalar(margin);

    // Compute distance squared
    let d_squared = distance.mul(distance)?;

    // Term 1: (1-y) * 0.5 * d²
    let one_minus_y = one.sub(labels)?;
    let term1 = one_minus_y.mul(&half)?.mul(&d_squared)?;

    // Term 2: y * 0.5 * max(0, margin - d)²
    let margin_minus_d = margin_tensor.sub(distance)?;
    let margin_minus_d_relu = tenflowers_core::ops::relu(&margin_minus_d)?;
    let margin_minus_d_squared = margin_minus_d_relu.mul(&margin_minus_d_relu)?;
    let term2 = labels.mul(&half)?.mul(&margin_minus_d_squared)?;

    // Combine terms
    let loss = term1.add(&term2)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}
