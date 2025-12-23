use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor, TensorError};

use super::regression::mse;

/// Binary Cross Entropy Loss
/// BCE(y, p) = -[y*log(p) + (1-y)*log(1-p)]
pub fn binary_cross_entropy<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Binary cross entropy: -[y*log(p) + (1-y)*log(1-p)]
    // We add a small epsilon for numerical stability
    let epsilon = T::from(1e-7).unwrap();

    // Clip predictions to avoid log(0)
    let p_clipped = predictions.add(&Tensor::from_scalar(epsilon))?;
    let one = Tensor::from_scalar(T::one());
    let one_minus_p = one.sub(&p_clipped)?;
    let one_minus_p_clipped = one_minus_p.add(&Tensor::from_scalar(epsilon))?;

    // Compute log(p) and log(1-p)
    let log_p = p_clipped.log()?;
    let log_one_minus_p = one_minus_p_clipped.log()?;

    // Compute y*log(p)
    let term1 = targets.mul(&log_p)?;

    // Compute (1-y)*log(1-p)
    let one_minus_y = one.sub(targets)?;
    let term2 = one_minus_y.mul(&log_one_minus_p)?;

    // Add terms and negate
    let sum = term1.add(&term2)?;
    let neg_sum = sum.neg()?;

    // Return mean over all elements
    tenflowers_core::ops::mean(&neg_sum, None, false)
}

/// Sparse Categorical Cross Entropy Loss
/// For use with integer labels (not one-hot encoded)
pub fn sparse_categorical_cross_entropy<T>(
    logits: &Tensor<T>,
    labels: &Tensor<i64>,
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Sparse categorical cross entropy: -log(softmax(logits)[labels])

    // Apply softmax to logits
    let probs = tenflowers_core::ops::softmax(logits, Some(-1))?;

    // Get batch size and number of classes
    let shape = logits.shape().dims();
    if shape.len() < 2 {
        return Err(tenflowers_core::error::TensorError::InvalidShape {
            operation: "sparse_categorical_cross_entropy".to_string(),
            reason: "Logits must be at least 2D for sparse categorical cross entropy".to_string(),
            shape: Some(shape.to_vec()),
            context: None,
        });
    }

    let batch_size = shape[0];
    let num_classes = shape[shape.len() - 1];

    // Get label data
    let label_data =
        labels
            .as_slice()
            .ok_or_else(|| tenflowers_core::error::TensorError::InvalidArgument {
                operation: "sparse_categorical_cross_entropy".to_string(),
                reason: "Cannot access label data".to_string(),
                context: None,
            })?;

    // Get probability data
    let prob_data =
        probs
            .as_slice()
            .ok_or_else(|| tenflowers_core::error::TensorError::InvalidArgument {
                operation: "sparse_categorical_cross_entropy".to_string(),
                reason: "Cannot access probability data".to_string(),
                context: None,
            })?;

    // Compute negative log likelihood
    let mut losses = Vec::new();
    for (batch_idx, &label_idx) in label_data.iter().enumerate() {
        if label_idx < 0 || label_idx as usize >= num_classes {
            return Err(tenflowers_core::error::TensorError::InvalidArgument {
                operation: "sparse_categorical_cross_entropy".to_string(),
                reason: format!(
                    "Label index {} out of bounds for {} classes",
                    label_idx, num_classes
                ),
                context: None,
            });
        }

        let prob_idx = batch_idx * num_classes + label_idx as usize;
        let prob = prob_data[prob_idx];
        let loss = -prob.ln();
        losses.push(loss);
    }

    // Create result tensor
    Tensor::from_vec(losses, &[batch_size])
        .and_then(|tensor| tenflowers_core::ops::mean(&tensor, None, false))
}

/// Categorical Cross Entropy Loss
/// For use with one-hot encoded targets
pub fn categorical_cross_entropy<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Categorical cross entropy for one-hot encoded targets
    // -sum(targets * log(softmax(predictions)))

    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "categorical_cross_entropy",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    // Apply softmax and then log for log_softmax
    let probs = tenflowers_core::ops::softmax(predictions, Some(-1))?;
    let epsilon = T::from(1e-7).unwrap();
    let probs_safe = probs.add(&Tensor::from_scalar(epsilon))?;
    let log_probs = probs_safe.log()?;

    // Multiply targets with log probabilities and sum
    let cross_entropy = targets.mul(&log_probs)?;
    let neg_cross_entropy = cross_entropy.neg()?;

    // Sum over the last dimension (classes) and take mean over batch
    tenflowers_core::ops::mean(&neg_cross_entropy, Some(&[-1]), false)
}

/// Focal Loss for addressing class imbalance
/// FocalLoss(y, p) = -alpha * (1 - p_t)^gamma * log(p_t)
pub fn focal_loss<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
    alpha: T,
    gamma: T,
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)
    // where p_t = p if target=1, else 1-p

    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "focal_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let epsilon = T::from(1e-7).unwrap();
    let one = Tensor::from_scalar(T::one());

    // Clip predictions for numerical stability
    let p_clipped = predictions.add(&Tensor::from_scalar(epsilon))?;

    // Compute p_t: where target=1, use p; where target=0, use 1-p
    let one_minus_p = one.sub(&p_clipped)?;
    let p_t_pos = targets.mul(&p_clipped)?; // target * p
    let p_t_neg = one.sub(targets)?.mul(&one_minus_p)?; // (1-target) * (1-p)
    let p_t = p_t_pos.add(&p_t_neg)?;

    // Compute log(p_t)
    let log_p_t = p_t.log()?;

    // Compute (1 - p_t)^gamma
    let one_minus_p_t = one.sub(&p_t)?;
    let modulating_factor = one_minus_p_t.pow(&Tensor::from_scalar(gamma))?;

    // Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    let alpha_tensor = Tensor::from_scalar(alpha);
    let focal = alpha_tensor.mul(&modulating_factor)?.mul(&log_p_t)?.neg()?;

    // Return mean over all elements
    tenflowers_core::ops::mean(&focal, None, false)
}

/// Hinge Loss for SVM-style classification
/// HingeLoss(y, pred) = max(0, 1 - y * pred)
pub fn hinge_loss<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>
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
    // Hinge loss: max(0, 1 - target * prediction)
    // Assumes targets are -1 or +1

    if predictions.shape() != targets.shape() {
        return Err(tenflowers_core::error::TensorError::shape_mismatch(
            "hinge_loss",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let one = Tensor::from_scalar(T::one());

    // Compute 1 - target * prediction
    let target_pred = targets.mul(predictions)?;
    let margin = one.sub(&target_pred)?;

    // Apply max(0, margin) - this is like ReLU
    let loss = tenflowers_core::ops::relu(&margin)?;

    // Return mean loss
    tenflowers_core::ops::mean(&loss, None, false)
}

/// Negative Log Likelihood Loss
/// For use with log-softmax outputs
/// NLL(log_probs, targets) = -log_probs[targets]
pub fn nll_loss<T>(log_probs: &Tensor<T>, targets: &Tensor<i64>) -> Result<Tensor<T>>
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
        + std::ops::Neg<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // NLL loss expects log probabilities and sparse integer targets
    let shape = log_probs.shape().dims();
    if shape.len() < 2 {
        return Err(tenflowers_core::error::TensorError::InvalidShape {
            operation: "nll_loss".to_string(),
            reason: "Log probabilities must be at least 2D".to_string(),
            shape: Some(shape.to_vec()),
            context: None,
        });
    }

    let batch_size = shape[0];
    let num_classes = shape[shape.len() - 1];

    // Get target data
    let target_data =
        targets
            .as_slice()
            .ok_or_else(|| tenflowers_core::error::TensorError::InvalidArgument {
                operation: "nll_loss".to_string(),
                reason: "Cannot access target data".to_string(),
                context: None,
            })?;

    // Get log probability data
    let log_prob_data = log_probs.as_slice().ok_or_else(|| {
        tenflowers_core::error::TensorError::InvalidArgument {
            operation: "nll_loss".to_string(),
            reason: "Cannot access log probability data".to_string(),
            context: None,
        }
    })?;

    // Compute negative log likelihood
    let mut losses = Vec::new();
    for (batch_idx, &target_idx) in target_data.iter().enumerate() {
        if target_idx < 0 || target_idx as usize >= num_classes {
            return Err(tenflowers_core::error::TensorError::InvalidArgument {
                operation: "nll_loss".to_string(),
                reason: format!(
                    "Target index {} out of bounds for {} classes",
                    target_idx, num_classes
                ),
                context: None,
            });
        }

        let log_prob_idx = batch_idx * num_classes + target_idx as usize;
        let log_prob = log_prob_data[log_prob_idx];
        let loss = -log_prob;
        losses.push(loss);
    }

    // Create result tensor and compute mean
    Tensor::from_vec(losses, &[batch_size])
        .and_then(|tensor| tenflowers_core::ops::mean(&tensor, None, false))
}

/// Knowledge Distillation Loss
/// Combines soft targets from teacher model and hard targets for training a student model
/// Loss = alpha * KL_divergence(teacher_logits, student_logits) + (1 - alpha) * cross_entropy(student_logits, hard_targets)
pub fn knowledge_distillation_loss<T>(
    student_logits: &Tensor<T>,
    teacher_logits: &Tensor<T>,
    hard_targets: &Tensor<i64>,
    temperature: T,
    alpha: T,
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Validate input shapes
    if student_logits.shape() != teacher_logits.shape() {
        return Err(TensorError::shape_mismatch(
            "loss_computation",
            &format!("{:?}", student_logits.shape()),
            &format!("{:?}", teacher_logits.shape()),
        ));
    }

    let batch_size = student_logits.shape().dims()[0];
    if hard_targets.shape().dims()[0] != batch_size {
        return Err(TensorError::shape_mismatch(
            "loss_computation",
            &format!("batch size {batch_size}"),
            &format!("target batch size {}", hard_targets.shape().dims()[0]),
        ));
    }

    // Validate parameters
    if alpha < T::zero() || alpha > T::one() {
        return Err(TensorError::InvalidArgument {
            operation: "knowledge_distillation_loss".to_string(),
            reason: "Alpha must be between 0 and 1".to_string(),
            context: None,
        });
    }

    if temperature <= T::zero() {
        return Err(TensorError::InvalidArgument {
            operation: "knowledge_distillation_loss".to_string(),
            reason: "Temperature must be positive".to_string(),
            context: None,
        });
    }

    // Create temperature tensor
    let temp_tensor = Tensor::from_scalar(temperature);

    // Scale logits by temperature for softer distributions
    let student_scaled = student_logits.div(&temp_tensor)?;
    let teacher_scaled = teacher_logits.div(&temp_tensor)?;

    // Compute soft targets using temperature-scaled softmax
    let teacher_soft = tenflowers_core::ops::softmax(&teacher_scaled, Some(-1))?;
    let student_soft = tenflowers_core::ops::softmax(&student_scaled, Some(-1))?;

    // Add epsilon to student_soft and compute log for numerical stability
    let epsilon = Tensor::from_scalar(T::from_f32(1e-8).unwrap_or(T::default()));
    let student_soft_safe = student_soft.add(&epsilon)?;
    let student_log_soft = student_soft_safe.log()?;

    // Compute distillation loss using KL divergence
    // KL(teacher_soft || student_soft) = sum(teacher_soft * (log(teacher_soft) - log_student_soft))
    let teacher_safe = teacher_soft.add(&epsilon)?;
    let log_teacher = teacher_safe.log()?;
    let log_diff = log_teacher.sub(&student_log_soft)?;
    let kl_terms = teacher_soft.mul(&log_diff)?;
    let distillation_loss = tenflowers_core::ops::mean(&kl_terms, None, false)?;

    // Scale distillation loss by temperature^2 (as per Hinton et al.)
    let temp_squared = Tensor::from_scalar(temperature * temperature);
    let scaled_distillation_loss = distillation_loss.mul(&temp_squared)?;

    // Compute hard target loss (standard cross-entropy)
    let hard_loss = sparse_categorical_cross_entropy(student_logits, hard_targets)?;

    // Combine losses: alpha * distillation_loss + (1 - alpha) * hard_loss
    let alpha_tensor = Tensor::from_scalar(alpha);
    let one_minus_alpha_tensor = Tensor::from_scalar(T::one() - alpha);

    let weighted_distillation = scaled_distillation_loss.mul(&alpha_tensor)?;
    let weighted_hard = hard_loss.mul(&one_minus_alpha_tensor)?;

    weighted_distillation.add(&weighted_hard)
}

/// Advanced Knowledge Distillation Loss with feature matching
/// Includes both output-level and intermediate feature-level distillation
#[allow(clippy::too_many_arguments)]
pub fn advanced_knowledge_distillation_loss<T>(
    student_logits: &Tensor<T>,
    teacher_logits: &Tensor<T>,
    student_features: Option<&[&Tensor<T>]>,
    teacher_features: Option<&[&Tensor<T>]>,
    hard_targets: &Tensor<i64>,
    temperature: T,
    alpha: T,
    beta: T, // Weight for feature matching loss
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
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Start with standard knowledge distillation loss
    let base_loss = knowledge_distillation_loss(
        student_logits,
        teacher_logits,
        hard_targets,
        temperature,
        alpha,
    )?;

    // Add feature matching loss if features are provided
    if let (Some(student_feats), Some(teacher_feats)) = (student_features, teacher_features) {
        if student_feats.len() != teacher_feats.len() {
            return Err(TensorError::InvalidArgument {
                operation: "advanced_knowledge_distillation_loss".to_string(),
                reason: "Number of student and teacher feature maps must match".to_string(),
                context: None,
            });
        }

        let mut feature_loss = Tensor::from_scalar(T::zero());
        let num_features = T::from(student_feats.len()).unwrap();

        // Compute MSE loss between corresponding feature maps
        for (student_feat, teacher_feat) in student_feats.iter().zip(teacher_feats.iter()) {
            let feat_loss = mse(student_feat, teacher_feat)?;
            feature_loss = feature_loss.add(&feat_loss)?;
        }

        // Average feature loss over all feature pairs
        let avg_feature_loss = feature_loss.div(&Tensor::from_scalar(num_features))?;

        // Combine with base loss: base_loss + beta * feature_loss
        let beta_tensor = Tensor::from_scalar(beta);
        let weighted_feature_loss = avg_feature_loss.mul(&beta_tensor)?;

        base_loss.add(&weighted_feature_loss)
    } else {
        Ok(base_loss)
    }
}
