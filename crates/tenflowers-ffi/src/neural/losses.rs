//! Loss functions module for TenfloweRS FFI
//!
//! This module provides comprehensive loss function implementations for neural network training,
//! including MSE, CrossEntropy, BCE, and other common loss functions.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Mean Squared Error (MSE) Loss
///
/// Computes the mean squared error between predictions and targets.
/// Commonly used for regression tasks.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn mse_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    // Check shapes match
    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    // Compute squared differences
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;
    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    let squared_diffs: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| (p - t).powi(2))
        .collect();

    let loss_value = match reduction {
        "mean" => squared_diffs.iter().sum::<f32>() / squared_diffs.len() as f32,
        "sum" => squared_diffs.iter().sum::<f32>(),
        "none" => {
            // Return unreduced tensor
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(squared_diffs, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Binary Cross Entropy (BCE) Loss
///
/// Computes the binary cross entropy loss between predictions and targets.
/// Commonly used for binary classification tasks.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn binary_cross_entropy(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    // Check shapes match
    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    // Compute BCE: -[t*log(p) + (1-t)*log(1-p)]
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;
    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    let eps = 1e-7_f32; // Small epsilon for numerical stability
    let losses: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| {
            let p_clamped = p.clamp(eps, 1.0 - eps);
            -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
        })
        .collect();

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "none" => {
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(losses, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Cross Entropy Loss
///
/// Computes the cross entropy loss between predictions and targets.
/// Commonly used for multi-class classification tasks.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn cross_entropy(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    // Predictions should be (batch_size, num_classes)
    // Targets should be (batch_size,) with class indices or (batch_size, num_classes) one-hot
    if pred_shape.len() < 2 {
        return Err(PyValueError::new_err(
            "Predictions must be at least 2D (batch_size, num_classes)",
        ));
    }

    let batch_size = pred_shape[0];
    let num_classes = pred_shape[1];

    // For now, implement a simplified version
    let pred_data = predictions
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get predictions: {}", e)))?;

    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    let eps = 1e-7_f32;
    let mut losses = Vec::new();

    // Check if targets are class indices or one-hot
    if target_shape.len() == 1 || (target_shape.len() == 2 && target_shape[1] == 1) {
        // Class indices
        for (i, &target_val) in target_data.iter().enumerate().take(batch_size) {
            let class_idx = target_val as usize;
            if class_idx >= num_classes {
                return Err(PyValueError::new_err(format!(
                    "Target class index {} out of bounds for {} classes",
                    class_idx, num_classes
                )));
            }

            // Get prediction for the target class
            let pred_idx = i * num_classes + class_idx;
            let pred = pred_data[pred_idx].clamp(eps, 1.0 - eps);
            losses.push(-pred.ln());
        }
    } else {
        // One-hot encoded targets
        for i in 0..batch_size {
            let mut loss = 0.0;
            for j in 0..num_classes {
                let pred_idx = i * num_classes + j;
                let target_idx = i * num_classes + j;
                let pred = pred_data[pred_idx].clamp(eps, 1.0 - eps);
                let target = target_data[target_idx];
                loss -= target * pred.ln();
            }
            losses.push(loss);
        }
    }

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "none" => {
            let loss_tensor = Tensor::from_vec(losses, &[batch_size]).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// L1 Loss (Mean Absolute Error)
///
/// Computes the mean absolute error between predictions and targets.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn l1_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
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

    let abs_diffs: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| (p - t).abs())
        .collect();

    let loss_value = match reduction {
        "mean" => abs_diffs.iter().sum::<f32>() / abs_diffs.len() as f32,
        "sum" => abs_diffs.iter().sum::<f32>(),
        "none" => {
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(abs_diffs, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Smooth L1 Loss (Huber Loss)
///
/// Combines advantages of L1 and L2 loss. Less sensitive to outliers than L2.
#[pyfunction]
#[pyo3(signature = (predictions, targets, beta=1.0, reduction="mean"))]
pub fn smooth_l1_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    beta: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let beta = beta.unwrap_or(1.0);
    let reduction = reduction.unwrap_or("mean");

    if beta <= 0.0 {
        return Err(PyValueError::new_err("beta must be positive"));
    }

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
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

    let losses: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| {
            let diff = (p - t).abs();
            if diff < beta {
                0.5 * diff.powi(2) / beta
            } else {
                diff - 0.5 * beta
            }
        })
        .collect();

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "none" => {
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(losses, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Kullback-Leibler Divergence Loss
///
/// Measures how one probability distribution diverges from a second expected distribution.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn kl_div_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
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

    let eps = 1e-7_f32;
    let losses: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| {
            let p_clamped = p.clamp(eps, 1.0);
            let t_clamped = t.clamp(eps, 1.0);
            t_clamped * (t_clamped / p_clamped).ln()
        })
        .collect();

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "batchmean" => losses.iter().sum::<f32>() / pred_shape[0] as f32,
        "none" => {
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(losses, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', 'batchmean', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Hinge Embedding Loss
///
/// Measures the loss for embedding learning. Used in ranking and similarity learning.
#[pyfunction]
#[pyo3(signature = (predictions, targets, margin=1.0, reduction="mean"))]
pub fn hinge_embedding_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    margin: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let margin = margin.unwrap_or(1.0);
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
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

    let losses: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(p, t)| if *t == 1.0 { *p } else { (margin - p).max(0.0) })
        .collect();

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "none" => {
            let shape_vec: Vec<usize> = pred_shape.iter().copied().collect();
            let loss_tensor = Tensor::from_vec(losses, &shape_vec).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: predictions.requires_grad || targets.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: predictions.requires_grad || targets.requires_grad,
        is_pinned: false,
    })
}

/// Cosine Embedding Loss
///
/// Measures the cosine similarity between two embeddings.
#[pyfunction]
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction="mean"))]
pub fn cosine_embedding_loss(
    input1: &PyTensor,
    input2: &PyTensor,
    target: &PyTensor,
    margin: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let margin = margin.unwrap_or(0.0);
    let reduction = reduction.unwrap_or("mean");

    let shape1 = input1.tensor.shape();
    let shape2 = input2.tensor.shape();

    if shape1 != shape2 {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: input1 {:?} vs input2 {:?}",
            shape1, shape2
        )));
    }

    let data1 = input1
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input1: {}", e)))?;
    let data2 = input2
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input2: {}", e)))?;
    let target_data = target
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get target: {}", e)))?;

    // Compute cosine similarity for each pair
    let batch_size = shape1[0];
    let feature_size = if shape1.len() > 1 { shape1[1] } else { 1 };

    let mut losses = Vec::new();
    for (i, &target_val) in target_data.iter().enumerate().take(batch_size) {
        let start = i * feature_size;
        let end = start + feature_size;

        let vec1 = &data1[start..end];
        let vec2 = &data2[start..end];

        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        let eps = 1e-8;
        let cos_sim = dot_product / (norm1 * norm2 + eps);

        let loss = if target_val == 1.0 {
            1.0 - cos_sim
        } else {
            (cos_sim - margin).max(0.0)
        };

        losses.push(loss);
    }

    let loss_value = match reduction {
        "mean" => losses.iter().sum::<f32>() / losses.len() as f32,
        "sum" => losses.iter().sum::<f32>(),
        "none" => {
            let loss_tensor = Tensor::from_vec(losses, &[batch_size]).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e))
            })?;
            return Ok(PyTensor {
                tensor: Arc::new(loss_tensor),
                requires_grad: input1.requires_grad || input2.requires_grad || target.requires_grad,
                is_pinned: false,
            });
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
                reduction
            )));
        }
    };

    let loss_tensor = Tensor::from_vec(vec![loss_value], &[1])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loss tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(loss_tensor),
        requires_grad: input1.requires_grad || input2.requires_grad || target.requires_grad,
        is_pinned: false,
    })
}
