use num_traits::ToPrimitive;
use tenflowers_core::{Result, Tensor, TensorError};

/// Compute accuracy between predictions and targets
/// For classification: assumes predictions are class indices or probabilities
pub fn accuracy<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let correct = pred_slice
        .iter()
        .zip(target_slice.iter())
        .filter(|(p, t)| **p == **t)
        .count();

    let total = pred_slice.len();
    if total == 0 {
        return Ok(0.0);
    }

    Ok(correct as f32 / total as f32)
}

/// Compute precision for binary classification
/// Precision = TP / (TP + FP)
pub fn precision<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let (mut tp, mut fp) = (0, 0);

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_i32().unwrap_or(0);
        let target_val = t.to_i32().unwrap_or(0);

        if pred_val == 1 {
            if target_val == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }

    if tp + fp == 0 {
        return Ok(0.0); // No positive predictions
    }

    Ok(tp as f32 / (tp + fp) as f32)
}

/// Compute recall for binary classification
/// Recall = TP / (TP + FN)
pub fn recall<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let (mut tp, mut fn_) = (0, 0);

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_i32().unwrap_or(0);
        let target_val = t.to_i32().unwrap_or(0);

        if target_val == 1 {
            if pred_val == 1 {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }

    if tp + fn_ == 0 {
        return Ok(0.0); // No positive targets
    }

    Ok(tp as f32 / (tp + fn_) as f32)
}

/// Compute F1 score for binary classification
/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    let prec = precision(predictions, targets)?;
    let rec = recall(predictions, targets)?;

    if prec + rec == 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * (prec * rec) / (prec + rec))
}

/// Compute top-k accuracy
/// Checks if the true label is within the top k predictions
pub fn top_k_accuracy<T>(predictions: &Tensor<T>, targets: &Tensor<i64>, k: usize) -> Result<f32>
where
    T: Clone + Default + PartialOrd + ToPrimitive + Send + Sync + 'static,
{
    let pred_shape = predictions.shape().dims();
    let target_shape = targets.shape().dims();

    if pred_shape.len() != 2 || target_shape.len() != 1 {
        return Err(TensorError::invalid_argument(
            "Predictions must be 2D and targets must be 1D for top-k accuracy".to_string(),
        ));
    }

    let batch_size = pred_shape[0];
    let num_classes = pred_shape[1];

    if target_shape[0] != batch_size {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &format!("[{batch_size}]"),
            &format!("[{}]", target_shape[0]),
        ));
    }

    if k > num_classes {
        return Err(TensorError::invalid_argument(format!(
            "k ({k}) cannot be larger than number of classes ({num_classes})"
        )));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let mut correct = 0;

    for (b, &target_val) in target_slice.iter().enumerate().take(batch_size) {
        let target_class = target_val as usize;
        if target_class >= num_classes {
            continue; // Skip invalid labels
        }

        // Get predictions for this batch element
        let start_idx = b * num_classes;
        let end_idx = start_idx + num_classes;
        let batch_preds = &pred_slice[start_idx..end_idx];

        // Create indexed pairs and sort by prediction value (descending)
        let mut indexed_preds: Vec<(usize, &T)> = batch_preds.iter().enumerate().collect();
        indexed_preds.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check if target class is in top k
        for (class_idx, _) in indexed_preds.iter().take(k) {
            if *class_idx == target_class {
                correct += 1;
                break;
            }
        }
    }

    Ok(correct as f32 / batch_size as f32)
}

/// Compute confusion matrix for binary classification
/// Returns (TN, FP, FN, TP)
pub fn confusion_matrix<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
) -> Result<(i32, i32, i32, i32)>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let (mut tn, mut fp, mut fn_, mut tp) = (0, 0, 0, 0);

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_i32().unwrap_or(0);
        let target_val = t.to_i32().unwrap_or(0);

        match (pred_val, target_val) {
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            (1, 1) => tp += 1,
            _ => {} // Ignore other values
        }
    }

    Ok((tn, fp, fn_, tp))
}

/// Compute Mean Absolute Percentage Error (MAPE)
/// MAPE = mean(|actual - predicted| / |actual|) * 100
pub fn mean_absolute_percentage_error<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
) -> Result<f32>
where
    T: Clone + Default + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let mut total_error = 0.0f32;
    let mut valid_count = 0;

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_f32().unwrap_or(0.0);
        let target_val = t.to_f32().unwrap_or(0.0);

        if target_val.abs() > 1e-8 {
            // Avoid division by zero
            let percentage_error = ((target_val - pred_val).abs() / target_val.abs()) * 100.0;
            total_error += percentage_error;
            valid_count += 1;
        }
    }

    if valid_count == 0 {
        return Ok(0.0);
    }

    Ok(total_error / valid_count as f32)
}

/// Compute R-squared (coefficient of determination)
/// R² = 1 - (SS_res / SS_tot)
pub fn r_squared<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    if target_slice.is_empty() {
        return Ok(0.0);
    }

    // Compute mean of targets
    let target_mean = target_slice
        .iter()
        .map(|x| x.to_f32().unwrap_or(0.0))
        .sum::<f32>()
        / target_slice.len() as f32;

    // Compute SS_res (residual sum of squares) and SS_tot (total sum of squares)
    let mut ss_res = 0.0f32;
    let mut ss_tot = 0.0f32;

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_f32().unwrap_or(0.0);
        let target_val = t.to_f32().unwrap_or(0.0);

        ss_res += (target_val - pred_val).powi(2);
        ss_tot += (target_val - target_mean).powi(2);
    }

    if ss_tot.abs() < 1e-8 {
        return Ok(0.0); // All targets are the same
    }

    Ok(1.0 - (ss_res / ss_tot))
}

/// Compute AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)
/// Uses trapezoidal rule for integration
pub fn auc_roc<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialOrd + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    // Create paired vector of (prediction, target)
    let mut pairs: Vec<(f32, i32)> = pred_slice
        .iter()
        .zip(target_slice.iter())
        .map(|(p, t)| (p.to_f32().unwrap_or(0.0), t.to_i32().unwrap_or(0)))
        .collect();

    // Sort by prediction scores (descending)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Count positive and negative samples
    let total_pos = pairs.iter().filter(|(_, t)| *t == 1).count();
    let total_neg = pairs.iter().filter(|(_, t)| *t == 0).count();

    if total_pos == 0 || total_neg == 0 {
        return Ok(0.5); // AUC = 0.5 when all samples are of one class
    }

    // Calculate TPR and FPR at different thresholds
    let mut tpr_fpr_pairs = Vec::new();
    let mut tp = 0;
    let mut fp = 0;

    // Start with (0, 0) point
    tpr_fpr_pairs.push((0.0f32, 0.0f32));

    for (_, target) in pairs.iter() {
        if *target == 1 {
            tp += 1;
        } else {
            fp += 1;
        }

        let tpr = tp as f32 / total_pos as f32;
        let fpr = fp as f32 / total_neg as f32;
        tpr_fpr_pairs.push((fpr, tpr));
    }

    // Calculate AUC using trapezoidal rule
    let mut auc = 0.0f32;
    for i in 1..tpr_fpr_pairs.len() {
        let (fpr1, tpr1) = tpr_fpr_pairs[i - 1];
        let (fpr2, tpr2) = tpr_fpr_pairs[i];

        let width = fpr2 - fpr1;
        let height = (tpr1 + tpr2) / 2.0;
        auc += width * height;
    }

    Ok(auc)
}

/// Compute AUC-PR (Area Under the Curve - Precision-Recall)
/// Uses trapezoidal rule for integration
pub fn auc_pr<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialOrd + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    // Create paired vector of (prediction, target)
    let mut pairs: Vec<(f32, i32)> = pred_slice
        .iter()
        .zip(target_slice.iter())
        .map(|(p, t)| (p.to_f32().unwrap_or(0.0), t.to_i32().unwrap_or(0)))
        .collect();

    // Sort by prediction scores (descending)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = pairs.iter().filter(|(_, t)| *t == 1).count();

    if total_pos == 0 {
        return Ok(0.0); // No positive samples
    }

    // Calculate precision and recall at different thresholds
    let mut pr_pairs = Vec::new();
    let mut tp = 0;
    let mut fp = 0;

    for (_, target) in pairs.iter() {
        if *target == 1 {
            tp += 1;
        } else {
            fp += 1;
        }

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            1.0
        };
        let recall = tp as f32 / total_pos as f32;
        pr_pairs.push((recall, precision));
    }

    // Add (0, 1) point if not already present
    if pr_pairs.is_empty() || pr_pairs[0].0 > 0.0 {
        pr_pairs.insert(0, (0.0, 1.0));
    }

    // Calculate AUC using trapezoidal rule
    let mut auc = 0.0f32;
    for i in 1..pr_pairs.len() {
        let (recall1, precision1) = pr_pairs[i - 1];
        let (recall2, precision2) = pr_pairs[i];

        let width = recall2 - recall1;
        let height = (precision1 + precision2) / 2.0;
        auc += width * height;
    }

    Ok(auc)
}

/// Compute F1 Score with micro/macro averaging for multi-class classification
pub fn f1_score_multiclass<T>(
    predictions: &Tensor<T>,
    targets: &Tensor<T>,
    average: &str,
) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    // Find unique classes
    let mut classes = std::collections::HashSet::new();
    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        classes.insert(p.to_i32().unwrap_or(0));
        classes.insert(t.to_i32().unwrap_or(0));
    }
    let classes: Vec<i32> = classes.into_iter().collect();

    match average {
        "micro" => {
            // Micro averaging: calculate metrics globally
            let mut total_tp = 0;
            let mut total_fp = 0;
            let mut total_fn = 0;

            for &class in &classes {
                for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
                    let pred_val = p.to_i32().unwrap_or(0);
                    let target_val = t.to_i32().unwrap_or(0);

                    let pred_positive = pred_val == class;
                    let target_positive = target_val == class;

                    if pred_positive && target_positive {
                        total_tp += 1;
                    } else if pred_positive && !target_positive {
                        total_fp += 1;
                    } else if !pred_positive && target_positive {
                        total_fn += 1;
                    }
                }
            }

            let precision = if total_tp + total_fp > 0 {
                total_tp as f32 / (total_tp + total_fp) as f32
            } else {
                0.0
            };
            let recall = if total_tp + total_fn > 0 {
                total_tp as f32 / (total_tp + total_fn) as f32
            } else {
                0.0
            };

            if precision + recall > 0.0 {
                Ok(2.0 * (precision * recall) / (precision + recall))
            } else {
                Ok(0.0)
            }
        }
        "macro" => {
            // Macro averaging: calculate metrics for each class, then average
            let mut class_f1_scores = Vec::new();

            for &class in &classes {
                let mut tp = 0;
                let mut fp = 0;
                let mut fn_ = 0;

                for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
                    let pred_val = p.to_i32().unwrap_or(0);
                    let target_val = t.to_i32().unwrap_or(0);

                    let pred_positive = pred_val == class;
                    let target_positive = target_val == class;

                    if pred_positive && target_positive {
                        tp += 1;
                    } else if pred_positive && !target_positive {
                        fp += 1;
                    } else if !pred_positive && target_positive {
                        fn_ += 1;
                    }
                }

                let precision = if tp + fp > 0 {
                    tp as f32 / (tp + fp) as f32
                } else {
                    0.0
                };
                let recall = if tp + fn_ > 0 {
                    tp as f32 / (tp + fn_) as f32
                } else {
                    0.0
                };

                let f1 = if precision + recall > 0.0 {
                    2.0 * (precision * recall) / (precision + recall)
                } else {
                    0.0
                };

                class_f1_scores.push(f1);
            }

            let macro_f1 = class_f1_scores.iter().sum::<f32>() / class_f1_scores.len() as f32;
            Ok(macro_f1)
        }
        _ => Err(TensorError::invalid_argument(
            "average must be 'micro' or 'macro'".to_string(),
        )),
    }
}

/// Compute Explained Variance for regression
/// Explained Variance = 1 - Var(y - y_pred) / Var(y)
pub fn explained_variance<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    if target_slice.is_empty() {
        return Ok(0.0);
    }

    // Convert to f32 for calculations
    let pred_vals: Vec<f32> = pred_slice
        .iter()
        .map(|x| x.to_f32().unwrap_or(0.0))
        .collect();
    let target_vals: Vec<f32> = target_slice
        .iter()
        .map(|x| x.to_f32().unwrap_or(0.0))
        .collect();

    // Calculate means
    let target_mean = target_vals.iter().sum::<f32>() / target_vals.len() as f32;
    let residual_vals: Vec<f32> = target_vals
        .iter()
        .zip(pred_vals.iter())
        .map(|(t, p)| t - p)
        .collect();
    let residual_mean = residual_vals.iter().sum::<f32>() / residual_vals.len() as f32;

    // Calculate variances
    let target_var = target_vals
        .iter()
        .map(|x| (x - target_mean).powi(2))
        .sum::<f32>()
        / target_vals.len() as f32;

    let residual_var = residual_vals
        .iter()
        .map(|x| (x - residual_mean).powi(2))
        .sum::<f32>()
        / residual_vals.len() as f32;

    if target_var.abs() < 1e-8 {
        return Ok(0.0); // All targets are the same
    }

    Ok(1.0 - (residual_var / target_var))
}

/// Compute IoU (Intersection over Union) / Jaccard Index for segmentation
/// IoU = (True Positive) / (True Positive + False Positive + False Negative)
pub fn iou_jaccard<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let mut intersection = 0;
    let mut union = 0;

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_i32().unwrap_or(0);
        let target_val = t.to_i32().unwrap_or(0);

        if pred_val == 1 && target_val == 1 {
            intersection += 1;
        }
        if pred_val == 1 || target_val == 1 {
            union += 1;
        }
    }

    if union == 0 {
        return Ok(1.0); // Both prediction and target are all zeros
    }

    Ok(intersection as f32 / union as f32)
}

/// Compute Dice coefficient for segmentation
/// Dice = 2 * (True Positive) / (2 * True Positive + False Positive + False Negative)
pub fn dice_coefficient<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for (p, t) in pred_slice.iter().zip(target_slice.iter()) {
        let pred_val = p.to_i32().unwrap_or(0);
        let target_val = t.to_i32().unwrap_or(0);

        match (pred_val, target_val) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {} // TN cases don't affect Dice coefficient
        }
    }

    let denominator = 2 * tp + fp + fn_;
    if denominator == 0 {
        return Ok(1.0); // Both prediction and target are all zeros
    }

    Ok((2 * tp) as f32 / denominator as f32)
}

/// Compute pixel accuracy for segmentation tasks
/// Pixel Accuracy = (Total Correct Pixels) / (Total Pixels)
pub fn pixel_accuracy<T>(predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<f32>
where
    T: Clone + Default + PartialEq + ToPrimitive + Send + Sync + 'static,
{
    if predictions.shape() != targets.shape() {
        return Err(TensorError::shape_mismatch(
            "metrics_computation",
            &targets.shape().to_string(),
            &predictions.shape().to_string(),
        ));
    }

    let pred_slice = predictions.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Predictions tensor is not contiguous".to_string())
    })?;
    let target_slice = targets.as_slice().ok_or_else(|| {
        TensorError::invalid_argument("Targets tensor is not contiguous".to_string())
    })?;

    let correct_pixels = pred_slice
        .iter()
        .zip(target_slice.iter())
        .filter(|(p, t)| **p == **t)
        .count();

    let total_pixels = pred_slice.len();
    if total_pixels == 0 {
        return Ok(0.0);
    }

    Ok(correct_pixels as f32 / total_pixels as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_accuracy() {
        let predictions = Tensor::<i32>::from_vec(vec![1, 0, 1, 1, 0], &[5]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 0], &[5]).unwrap();

        let acc = accuracy(&predictions, &targets).unwrap();
        assert_eq!(acc, 0.8); // 4 out of 5 correct
    }

    #[test]
    fn test_precision() {
        let predictions = Tensor::<i32>::from_vec(vec![1, 1, 0, 1, 0], &[5]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 0], &[5]).unwrap();

        let prec = precision(&predictions, &targets).unwrap();
        // TP = 2 (indices 0, 3), FP = 1 (index 1)
        // Precision = 2 / (2 + 1) = 2/3 ≈ 0.667
        assert!((prec - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall() {
        let predictions = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 0], &[5]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 1, 0, 1, 0], &[5]).unwrap();

        let rec = recall(&predictions, &targets).unwrap();
        // TP = 2 (indices 0, 3), FN = 1 (index 1)
        // Recall = 2 / (2 + 1) = 2/3 ≈ 0.667
        assert!((rec - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        // Empty tensors
        let empty_pred = Tensor::<i32>::from_vec(vec![], &[0]).unwrap();
        let empty_target = Tensor::<i32>::from_vec(vec![], &[0]).unwrap();

        assert_eq!(accuracy(&empty_pred, &empty_target).unwrap(), 0.0);

        // All zeros (no positive predictions/targets)
        let zeros_pred = Tensor::<i32>::from_vec(vec![0, 0, 0], &[3]).unwrap();
        let zeros_target = Tensor::<i32>::from_vec(vec![0, 0, 0], &[3]).unwrap();

        assert_eq!(precision(&zeros_pred, &zeros_target).unwrap(), 0.0);
        assert_eq!(recall(&zeros_pred, &zeros_target).unwrap(), 0.0);
    }

    #[test]
    fn test_f1_score() {
        let predictions = Tensor::<i32>::from_vec(vec![1, 0, 1, 1, 0], &[5]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 0], &[5]).unwrap();

        let f1 = f1_score(&predictions, &targets).unwrap();
        let expected_precision = 2.0 / 3.0; // 2 TP, 1 FP
        let expected_recall = 2.0 / 2.0; // 2 TP, 0 FN (both 1s in targets are correctly predicted)
        let expected_f1 =
            2.0 * (expected_precision * expected_recall) / (expected_precision + expected_recall);

        assert!((f1 - expected_f1).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_accuracy() {
        // 2 samples, 3 classes
        let predictions = Tensor::<f32>::from_vec(
            vec![
                0.1, 0.8, 0.1, // Sample 1: class 1 has highest prob
                0.3, 0.2, 0.5,
            ], // Sample 2: class 2 has highest prob
            &[2, 3],
        )
        .unwrap();
        let targets = Tensor::<i64>::from_vec(vec![1, 2], &[2]).unwrap();

        // Top-1 accuracy should be 100% (both predictions are correct)
        let top1 = top_k_accuracy(&predictions, &targets, 1).unwrap();
        assert_eq!(top1, 1.0);

        // Top-2 accuracy should also be 100%
        let top2 = top_k_accuracy(&predictions, &targets, 2).unwrap();
        assert_eq!(top2, 1.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = Tensor::<i32>::from_vec(vec![1, 0, 1, 0, 1], &[5]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 0, 0, 0, 1], &[5]).unwrap();

        let (tn, fp, fn_, tp) = confusion_matrix(&predictions, &targets).unwrap();

        // TN: predicted 0, actual 0 (indices 1, 3) = 2
        // FP: predicted 1, actual 0 (index 2) = 1
        // FN: predicted 0, actual 1 (none) = 0
        // TP: predicted 1, actual 1 (indices 0, 4) = 2
        assert_eq!(tn, 2);
        assert_eq!(fp, 1);
        assert_eq!(fn_, 0);
        assert_eq!(tp, 2);
    }

    #[test]
    fn test_mape() {
        let predictions = Tensor::<f32>::from_vec(vec![2.0, 4.0, 6.0], &[3]).unwrap();
        let targets = Tensor::<f32>::from_vec(vec![2.5, 4.0, 5.0], &[3]).unwrap();

        let mape = mean_absolute_percentage_error(&predictions, &targets).unwrap();

        // MAPE = mean([|2.5-2.0|/2.5, |4.0-4.0|/4.0, |5.0-6.0|/5.0]) * 100
        // MAPE = mean([0.2, 0.0, 0.2]) * 100 = 0.1333... * 100 = 13.333...
        let expected = (20.0 + 0.0 + 20.0) / 3.0; // 13.333...
        assert!((mape - expected).abs() < 1e-4);
    }

    #[test]
    fn test_r_squared() {
        // Perfect predictions should give R² = 1
        let predictions = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let targets = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let r2 = r_squared(&predictions, &targets).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);

        // Mean prediction should give R² = 0
        let mean_pred = Tensor::<f32>::from_vec(vec![2.5, 2.5, 2.5, 2.5], &[4]).unwrap();
        let r2_mean = r_squared(&mean_pred, &targets).unwrap();
        assert!(r2_mean.abs() < 1e-6);
    }

    #[test]
    fn test_auc_roc() {
        // Perfect classifier (all positives have higher scores than negatives)
        let predictions =
            Tensor::<f32>::from_vec(vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1], &[6]).unwrap();
        let targets = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0], &[6]).unwrap();

        let auc = auc_roc(&predictions, &targets).unwrap();
        assert!((auc - 1.0).abs() < 1e-6);

        // Random classifier should give AUC ≈ 0.5
        let random_pred = Tensor::<f32>::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[4]).unwrap();
        let random_targets = Tensor::<f32>::from_vec(vec![1.0, 0.0, 1.0, 0.0], &[4]).unwrap();

        let auc_random = auc_roc(&random_pred, &random_targets).unwrap();
        assert!(auc_random >= 0.0 && auc_random <= 1.0);
    }

    #[test]
    fn test_auc_pr() {
        // Perfect classifier
        let predictions =
            Tensor::<f32>::from_vec(vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1], &[6]).unwrap();
        let targets = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0], &[6]).unwrap();

        let auc = auc_pr(&predictions, &targets).unwrap();
        assert!(auc >= 0.0 && auc <= 1.0);

        // All negative case
        let all_neg_targets = Tensor::<f32>::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let some_pred = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();

        let auc_neg = auc_pr(&some_pred, &all_neg_targets).unwrap();
        assert_eq!(auc_neg, 0.0);
    }

    #[test]
    fn test_f1_score_multiclass() {
        // Multi-class example
        let predictions = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2], &[6]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![0, 1, 2, 1, 2, 0], &[6]).unwrap();

        let f1_micro = f1_score_multiclass(&predictions, &targets, "micro").unwrap();
        let f1_macro = f1_score_multiclass(&predictions, &targets, "macro").unwrap();

        assert!(f1_micro >= 0.0 && f1_micro <= 1.0);
        assert!(f1_macro >= 0.0 && f1_macro <= 1.0);

        // Perfect classification
        let perfect_pred = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2], &[6]).unwrap();
        let perfect_targets = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2], &[6]).unwrap();

        let f1_perfect = f1_score_multiclass(&perfect_pred, &perfect_targets, "micro").unwrap();
        assert!((f1_perfect - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_explained_variance() {
        // Perfect predictions should give explained variance = 1
        let predictions = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let targets = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let ev = explained_variance(&predictions, &targets).unwrap();
        assert!((ev - 1.0).abs() < 1e-6);

        // Random predictions should give lower explained variance
        let random_pred = Tensor::<f32>::from_vec(vec![1.5, 2.5, 2.8, 3.2], &[4]).unwrap();
        let ev_random = explained_variance(&random_pred, &targets).unwrap();
        assert!(ev_random < 1.0);
    }

    #[test]
    fn test_iou_jaccard() {
        // Perfect segmentation
        let predictions = Tensor::<i32>::from_vec(vec![1, 1, 0, 0, 1, 0], &[6]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 1, 0, 0, 1, 0], &[6]).unwrap();

        let iou = iou_jaccard(&predictions, &targets).unwrap();
        assert!((iou - 1.0).abs() < 1e-6);

        // Partial overlap
        let pred_partial = Tensor::<i32>::from_vec(vec![1, 1, 1, 0, 0, 0], &[6]).unwrap();
        let target_partial = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 1, 0], &[6]).unwrap();

        let iou_partial = iou_jaccard(&pred_partial, &target_partial).unwrap();
        // Intersection = 1, Union = 5, IoU = 1/5 = 0.2
        assert!((iou_partial - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_dice_coefficient() {
        // Perfect segmentation
        let predictions = Tensor::<i32>::from_vec(vec![1, 1, 0, 0, 1, 0], &[6]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![1, 1, 0, 0, 1, 0], &[6]).unwrap();

        let dice = dice_coefficient(&predictions, &targets).unwrap();
        assert!((dice - 1.0).abs() < 1e-6);

        // Partial overlap
        let pred_partial = Tensor::<i32>::from_vec(vec![1, 1, 1, 0, 0, 0], &[6]).unwrap();
        let target_partial = Tensor::<i32>::from_vec(vec![1, 0, 0, 1, 1, 0], &[6]).unwrap();

        let dice_partial = dice_coefficient(&pred_partial, &target_partial).unwrap();
        // TP = 1, FP = 2, FN = 2, Dice = 2*1 / (2*1 + 2 + 2) = 2/6 = 1/3
        assert!((dice_partial - 1.0 / 3.0).abs() < 1e-6);

        // All zeros case
        let zero_pred = Tensor::<i32>::from_vec(vec![0, 0, 0, 0], &[4]).unwrap();
        let zero_target = Tensor::<i32>::from_vec(vec![0, 0, 0, 0], &[4]).unwrap();

        let dice_zero = dice_coefficient(&zero_pred, &zero_target).unwrap();
        assert!((dice_zero - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_accuracy() {
        // Perfect segmentation
        let predictions = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1], &[8]).unwrap();
        let targets = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1], &[8]).unwrap();

        let pixel_acc = pixel_accuracy(&predictions, &targets).unwrap();
        assert!((pixel_acc - 1.0).abs() < 1e-6);

        // Half correct segmentation
        let pred_half = Tensor::<i32>::from_vec(vec![0, 1, 2, 2, 1, 0, 0, 1], &[8]).unwrap();
        let target_half = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1], &[8]).unwrap();

        let pixel_acc_half = pixel_accuracy(&pred_half, &target_half).unwrap();
        // Correct: indices 0, 1, 2, 4, 6, 7 = 6 out of 8 = 0.75
        assert!((pixel_acc_half - 0.75).abs() < 1e-6);

        // All wrong segmentation
        let pred_wrong = Tensor::<i32>::from_vec(vec![1, 2, 0, 1, 2, 0, 1, 2], &[8]).unwrap();
        let target_wrong = Tensor::<i32>::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1], &[8]).unwrap();

        let pixel_acc_wrong = pixel_accuracy(&pred_wrong, &target_wrong).unwrap();
        assert!((pixel_acc_wrong - 0.0).abs() < 1e-6);

        // Empty case
        let empty_pred = Tensor::<i32>::from_vec(vec![], &[0]).unwrap();
        let empty_target = Tensor::<i32>::from_vec(vec![], &[0]).unwrap();

        let pixel_acc_empty = pixel_accuracy(&empty_pred, &empty_target).unwrap();
        assert!((pixel_acc_empty - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_new_metrics_edge_cases() {
        // Test empty tensors
        let empty_pred = Tensor::<f32>::from_vec(vec![], &[0]).unwrap();
        let empty_target = Tensor::<f32>::from_vec(vec![], &[0]).unwrap();

        // AUC functions should handle empty gracefully
        let auc_roc_empty = auc_roc(&empty_pred, &empty_target);
        assert!(auc_roc_empty.is_ok());

        // Test single class case
        let single_class_pred = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
        let single_class_target = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();

        let auc_single = auc_roc(&single_class_pred, &single_class_target).unwrap();
        assert_eq!(auc_single, 0.5); // Should return 0.5 for single class
    }
}
