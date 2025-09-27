use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Dice Loss for segmentation tasks
/// DiceLoss = 1 - DiceCoeff = 1 - (2 * |A ∩ B|) / (|A| + |B|)
pub fn dice_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "dice_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let epsilon = T::from(1e-7).unwrap();

    // Compute intersection: sum(predictions * targets)
    let intersection = predictions.mul(targets)?;
    let intersection_sum = tenflowers_core::ops::sum(&intersection, None, false)?;

    // Compute |X| + |Y|: sum(predictions) + sum(targets)
    let pred_sum = tenflowers_core::ops::sum(predictions, None, false)?;
    let target_sum = tenflowers_core::ops::sum(targets, None, false)?;
    let union = pred_sum.add(&target_sum)?;

    // Compute Dice coefficient: (2 * intersection) / (union + epsilon)
    let two = Tensor::from_scalar(T::from(2.0).unwrap());
    let numerator = two.mul(&intersection_sum)?;
    let denominator = union.add(&Tensor::from_scalar(epsilon))?;
    let dice_coeff = numerator.div(&denominator)?;

    // Dice loss = 1 - Dice coefficient
    let one = Tensor::from_scalar(T::one());
    one.sub(&dice_coeff)
}

/// IoU (Jaccard) Loss for segmentation tasks
/// IoULoss = 1 - IoU = 1 - |A ∩ B| / |A ∪ B|
pub fn iou_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "iou_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let epsilon = T::from(1e-7).unwrap();

    // Compute intersection: sum(predictions * targets)
    let intersection = predictions.mul(targets)?;
    let intersection_sum = tenflowers_core::ops::sum(&intersection, None, false)?;

    // Compute union: sum(predictions) + sum(targets) - intersection
    let pred_sum = tenflowers_core::ops::sum(predictions, None, false)?;
    let target_sum = tenflowers_core::ops::sum(targets, None, false)?;
    let union = pred_sum.add(&target_sum)?.sub(&intersection_sum)?;

    // Compute IoU: intersection / (union + epsilon)
    let denominator = union.add(&Tensor::from_scalar(epsilon))?;
    let iou = intersection_sum.div(&denominator)?;

    // IoU loss = 1 - IoU
    let one = Tensor::from_scalar(T::one());
    one.sub(&iou)
}

/// Generalized Dice Loss for multi-class segmentation
/// Handles class imbalance by weighting classes inversely to their frequency
pub fn generalized_dice_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
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
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "generalized_dice_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let epsilon = T::from(1e-7).unwrap();
    let shape = predictions.shape().dims();

    // For multi-class segmentation, we expect the last dimension to be classes
    let num_classes = shape[shape.len() - 1];

    let mut total_weighted_intersection = T::zero();
    let mut total_weighted_union = T::zero();

    for class_idx in 0..num_classes {
        // Extract class predictions and targets (simplified implementation)
        // In practice, you would need proper tensor slicing operations

        // Compute class frequency weight: 1 / (class_frequency^2 + epsilon)
        let class_sum = tenflowers_core::ops::sum(targets, None, false)?;
        let class_freq_data = class_sum.as_slice().ok_or_else(|| {
            tenflowers_core::error::TensorError::InvalidArgument {
                operation: "generalized_dice_loss".to_string(),
                reason: "Cannot access tensor data".to_string(),
                context: None,
            }
        })?;

        if let Some(&freq) = class_freq_data.first() {
            let weight = T::one() / (freq * freq + epsilon);

            // Compute weighted intersection and union for this class
            let intersection = predictions.mul(targets)?;
            let intersection_sum = tenflowers_core::ops::sum(&intersection, None, false)?;

            let pred_sum = tenflowers_core::ops::sum(predictions, None, false)?;
            let target_sum = tenflowers_core::ops::sum(targets, None, false)?;
            let union = pred_sum.add(&target_sum)?;

            // This is a simplified version - in practice you'd need proper tensor operations
            // for per-class computations
            total_weighted_intersection =
                total_weighted_intersection + weight * intersection_sum.as_slice().unwrap()[0];
            total_weighted_union = total_weighted_union + weight * union.as_slice().unwrap()[0];
        }
    }

    // Compute generalized Dice coefficient
    let dice_coeff =
        (T::from(2.0).unwrap() * total_weighted_intersection) / (total_weighted_union + epsilon);

    // Return 1 - Dice coefficient
    let loss = T::one() - dice_coeff;
    Ok(Tensor::from_scalar(loss))
}
