//! Evaluation metrics module for TenfloweRS FFI
//!
//! This module provides Python bindings for common evaluation metrics used in
//! machine learning model evaluation.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Compute accuracy for classification tasks
///
/// # Arguments
///
/// * `predictions` - Predicted labels or logits (will take argmax if not 1D)
/// * `targets` - Ground truth labels
///
/// # Returns
///
/// Accuracy as a float (0.0 to 1.0)
#[pyfunction]
pub fn accuracy(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    // Convert predictions to class labels if needed (argmax for multi-class)
    let pred_labels: Vec<usize> = if pred_shape.len() == 1 {
        // Already class labels
        pred_data.iter().map(|&x| x as usize).collect()
    } else if pred_shape.len() == 2 {
        // Logits/probabilities - take argmax along last dimension
        let num_samples = pred_shape[0];
        let num_classes = pred_shape[1];

        (0..num_samples)
            .map(|i| {
                let start = i * num_classes;
                let end = start + num_classes;
                let sample = &pred_data[start..end];

                sample
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    } else {
        return Err(PyValueError::new_err(format!(
            "Expected 1D or 2D predictions, got {}D",
            pred_shape.len()
        )));
    };

    let target_labels: Vec<usize> = target_data.iter().map(|&x| x as usize).collect();

    if pred_labels.len() != target_labels.len() {
        return Err(PyValueError::new_err(format!(
            "Predictions and targets must have same length: {} vs {}",
            pred_labels.len(),
            target_labels.len()
        )));
    }

    let correct: usize = pred_labels
        .iter()
        .zip(target_labels.iter())
        .filter(|(p, t)| p == t)
        .count();

    Ok(correct as f32 / pred_labels.len() as f32)
}

/// Compute precision, recall, and F1 score for binary classification
///
/// # Arguments
///
/// * `predictions` - Predicted labels (0 or 1)
/// * `targets` - Ground truth labels (0 or 1)
/// * `average` - Averaging method: 'binary', 'macro', 'micro' (default: 'binary')
///
/// # Returns
///
/// Tuple of (precision, recall, f1_score)
#[pyfunction]
#[pyo3(signature = (predictions, targets, average=None))]
pub fn precision_recall_f1(
    predictions: &PyTensor,
    targets: &PyTensor,
    average: Option<&str>,
) -> PyResult<(f32, f32, f32)> {
    let average = average.unwrap_or("binary");

    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    let pred_labels: Vec<usize> = pred_data.iter().map(|&x| x as usize).collect();
    let target_labels: Vec<usize> = target_data.iter().map(|&x| x as usize).collect();

    match average {
        "binary" => {
            let (tp, fp, tn, fn_) = confusion_matrix_binary(&pred_labels, &target_labels);

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
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            Ok((precision, recall, f1))
        }
        "micro" => {
            // For micro averaging, compute global TP, FP, FN
            let (tp, fp, _tn, fn_) = confusion_matrix_binary(&pred_labels, &target_labels);

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
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            Ok((precision, recall, f1))
        }
        "macro" => {
            // Compute per-class metrics and average
            let (tp, fp, _tn, fn_) = confusion_matrix_binary(&pred_labels, &target_labels);

            let precision_pos = if tp + fp > 0 {
                tp as f32 / (tp + fp) as f32
            } else {
                0.0
            };

            let recall_pos = if tp + fn_ > 0 {
                tp as f32 / (tp + fn_) as f32
            } else {
                0.0
            };

            // For binary case, macro averaging is same as binary
            let precision = precision_pos;
            let recall = recall_pos;

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            Ok((precision, recall, f1))
        }
        _ => Err(PyValueError::new_err(
            "average must be 'binary', 'micro', or 'macro'",
        )),
    }
}

/// Compute confusion matrix for binary classification
///
/// Returns (true_positives, false_positives, true_negatives, false_negatives)
fn confusion_matrix_binary(
    predictions: &[usize],
    targets: &[usize],
) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;

    for (&pred, &target) in predictions.iter().zip(targets.iter()) {
        match (pred, target) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {} // Ignore other values
        }
    }

    (tp, fp, tn, fn_)
}

/// Compute mean squared error
///
/// # Arguments
///
/// * `predictions` - Predicted values
/// * `targets` - Ground truth values
///
/// # Returns
///
/// Mean squared error
#[pyfunction]
pub fn mean_squared_error(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    let mse: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / pred_data.len() as f32;

    Ok(mse)
}

/// Compute mean absolute error
///
/// # Arguments
///
/// * `predictions` - Predicted values
/// * `targets` - Ground truth values
///
/// # Returns
///
/// Mean absolute error
#[pyfunction]
pub fn mean_absolute_error(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    let mae: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>()
        / pred_data.len() as f32;

    Ok(mae)
}

/// Compute R-squared (coefficient of determination)
///
/// # Arguments
///
/// * `predictions` - Predicted values
/// * `targets` - Ground truth values
///
/// # Returns
///
/// R-squared score
#[pyfunction]
pub fn r2_score(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    // Compute mean of targets
    let target_mean = target_data.iter().sum::<f32>() / target_data.len() as f32;

    // Compute SS_res (residual sum of squares)
    let ss_res: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum();

    // Compute SS_tot (total sum of squares)
    let ss_tot: f32 = target_data.iter().map(|t| (t - target_mean).powi(2)).sum();

    if ss_tot == 0.0 {
        return Ok(0.0);
    }

    Ok(1.0 - ss_res / ss_tot)
}

/// Compute top-k accuracy
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities/logits (shape: [batch, num_classes])
/// * `targets` - Ground truth labels
/// * `k` - Number of top predictions to consider (default: 5)
///
/// # Returns
///
/// Top-k accuracy as a float (0.0 to 1.0)
#[pyfunction]
#[pyo3(signature = (predictions, targets, k=None))]
pub fn top_k_accuracy(
    predictions: &PyTensor,
    targets: &PyTensor,
    k: Option<usize>,
) -> PyResult<f32> {
    let k = k.unwrap_or(5);

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "Predictions must be 2D (batch, num_classes)",
        ));
    }

    let num_samples = pred_shape[0];
    let num_classes = pred_shape[1];

    if k > num_classes {
        return Err(PyValueError::new_err(format!(
            "k ({}) cannot be larger than number of classes ({})",
            k, num_classes
        )));
    }

    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    let target_labels: Vec<usize> = target_data.iter().map(|&x| x as usize).collect();

    let mut correct = 0;

    for (i, &target_label) in target_labels.iter().enumerate().take(num_samples) {
        let start = i * num_classes;
        let end = start + num_classes;
        let sample = &pred_data[start..end];

        // Get top k indices
        let mut indices: Vec<usize> = (0..num_classes).collect();
        indices.sort_by(|&a, &b| sample[b].partial_cmp(&sample[a]).unwrap());

        let top_k_indices: Vec<usize> = indices.iter().take(k).copied().collect();

        if top_k_indices.contains(&target_label) {
            correct += 1;
        }
    }

    Ok(correct as f32 / num_samples as f32)
}

/// Compute Area Under ROC Curve (AUC-ROC) for binary classification
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities for positive class
/// * `targets` - Ground truth labels (0 or 1)
///
/// # Returns
///
/// AUC-ROC score
#[pyfunction]
pub fn auc_roc(predictions: &PyTensor, targets: &PyTensor) -> PyResult<f32> {
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    // Create pairs of (prediction, target)
    let mut pairs: Vec<(f32, usize)> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&pred, &target)| (pred, target as usize))
        .collect();

    // Sort by prediction score (descending)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Count positives and negatives
    let num_positives = pairs.iter().filter(|(_, label)| *label == 1).count();
    let num_negatives = pairs.len() - num_positives;

    if num_positives == 0 || num_negatives == 0 {
        return Ok(0.5); // Undefined, return 0.5
    }

    // Compute AUC using trapezoidal rule
    let mut tp = 0;
    let mut fp = 0;
    let mut auc = 0.0;
    let mut prev_fp = 0;

    for (_, label) in pairs {
        if label == 1 {
            tp += 1;
            auc += (fp - prev_fp) as f32 * (tp as f32 - 0.5);
            prev_fp = fp;
        } else {
            fp += 1;
        }
    }

    // Add final trapezoid
    auc += (fp - prev_fp) as f32 * tp as f32;

    // Normalize
    Ok(auc / (num_positives * num_negatives) as f32)
}

/// Compute confusion matrix for multi-class classification
///
/// # Arguments
///
/// * `predictions` - Predicted labels
/// * `targets` - Ground truth labels
/// * `num_classes` - Number of classes
///
/// # Returns
///
/// Flattened confusion matrix (row-major order)
#[pyfunction]
pub fn confusion_matrix(
    predictions: &PyTensor,
    targets: &PyTensor,
    num_classes: usize,
) -> PyResult<Vec<usize>> {
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    if pred_data.len() != target_data.len() {
        return Err(PyValueError::new_err(
            "Predictions and targets must have same length",
        ));
    }

    let mut matrix = vec![0usize; num_classes * num_classes];

    for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
        let pred_class = pred as usize;
        let target_class = target as usize;

        if pred_class >= num_classes || target_class >= num_classes {
            return Err(PyValueError::new_err(format!(
                "Class index out of bounds: pred={}, target={}, num_classes={}",
                pred_class, target_class, num_classes
            )));
        }

        matrix[target_class * num_classes + pred_class] += 1;
    }

    Ok(matrix)
}
