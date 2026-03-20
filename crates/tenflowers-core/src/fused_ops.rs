//! Fused Operation Utilities
//!
//! This module provides fused kernel utilities that avoid materialising
//! intermediate tensors, reducing memory bandwidth and improving performance.
//!
//! ## Available Operations
//!
//! - [`fused_linear_relu`]: Linear projection immediately followed by ReLU,
//!   without an intermediate allocation for the pre-activation values.
//! - [`fused_layer_norm_linear`]: Layer normalisation fused with a subsequent
//!   linear projection, reducing two matrix-read passes to one.
//! - [`layer_norm`]: Stand-alone numerically stable layer normalisation.
//!
//! ## Tensor Layout
//!
//! All functions use row-major flat `&[f32]` slices together with explicit
//! shape descriptors.  A 2-D matrix with shape `[rows, cols]` is stored as
//! `data[row * cols + col]`.
//!
//! ## Dimension Conventions
//!
//! | Name               | Shape           | Notes                               |
//! |--------------------|-----------------|-------------------------------------|
//! | `input`            | `[*, in_feat]`  | Leading dims are batch dimensions   |
//! | `weight`           | `[out_feat, in_feat]` | PyTorch / NumPy convention   |
//! | `bias`             | `[out_feat]`    | Optional                            |
//! | `norm_weight` (γ)  | `[in_feat]`     | Layer-norm gain                     |
//! | `norm_bias` (β)    | `[in_feat]`     | Layer-norm offset                   |

use crate::error::{Result, TensorError};

// ---------------------------------------------------------------------------
// Public fused kernels
// ---------------------------------------------------------------------------

/// Fused linear projection + ReLU activation.
///
/// Computes `output = relu(input @ weight.T + bias)` without allocating an
/// intermediate pre-activation buffer.
///
/// # Arguments
/// - `input`, `input_shape`: Input tensor with shape `[batch, in_features]`.
/// - `weight`, `weight_shape`: Weight matrix with shape `[out_features, in_features]`.
/// - `bias`: Optional bias vector with length `out_features`.
///
/// # Returns
/// `(output, output_shape)` where `output_shape = [batch, out_features]`.
///
/// # Errors
/// Returns `TensorError::InvalidShape` / `InvalidArgument` if shapes are inconsistent.
pub fn fused_linear_relu(
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<usize>)> {
    // ---- Shape validation ----
    if input_shape.len() < 1 {
        return Err(TensorError::InvalidShape {
            operation: "fused_linear_relu".to_string(),
            reason: "input_shape must have at least 1 dimension".to_string(),
            shape: Some(input_shape.to_vec()),
            context: None,
        });
    }
    if weight_shape.len() != 2 {
        return Err(TensorError::InvalidShape {
            operation: "fused_linear_relu".to_string(),
            reason: format!("weight_shape must be 2D [out, in], got {weight_shape:?}"),
            shape: Some(weight_shape.to_vec()),
            context: None,
        });
    }

    let in_features = *input_shape.last().expect("checked above");
    let out_features = weight_shape[0];
    let weight_in = weight_shape[1];

    if in_features != weight_in {
        return Err(TensorError::ShapeMismatch {
            operation: "fused_linear_relu".to_string(),
            expected: format!("in_features={in_features}"),
            got: format!("weight in_features={weight_in}"),
            context: None,
        });
    }

    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(TensorError::ShapeMismatch {
                operation: "fused_linear_relu".to_string(),
                expected: format!("bias length={out_features}"),
                got: format!("bias length={}", b.len()),
                context: None,
            });
        }
    }

    // Validate buffer lengths
    let input_expected: usize = input_shape.iter().product();
    let weight_expected: usize = weight_shape.iter().product();
    if input.len() != input_expected {
        return Err(TensorError::InvalidArgument {
            operation: "fused_linear_relu".to_string(),
            reason: format!(
                "input length {} != product of input_shape {:?} ({})",
                input.len(),
                input_shape,
                input_expected
            ),
            context: None,
        });
    }
    if weight.len() != weight_expected {
        return Err(TensorError::InvalidArgument {
            operation: "fused_linear_relu".to_string(),
            reason: format!(
                "weight length {} != product of weight_shape {:?} ({})",
                weight.len(),
                weight_shape,
                weight_expected
            ),
            context: None,
        });
    }

    // ---- Compute batch size (product of leading dimensions) ----
    let batch: usize = input_shape[..input_shape.len() - 1].iter().product::<usize>().max(1);
    // For a 1-D input_shape (e.g. [in_features]), batch = 1.
    let effective_batch = if input_shape.len() == 1 { 1 } else { batch };

    // ---- Fused linear + ReLU ----
    let out_len = effective_batch * out_features;
    let mut output = vec![0.0f32; out_len];

    for n in 0..effective_batch {
        let input_offset = n * in_features;
        let output_offset = n * out_features;

        for o in 0..out_features {
            let weight_offset = o * in_features;
            let mut sum = 0.0f32;
            for f in 0..in_features {
                sum += input[input_offset + f] * weight[weight_offset + f];
            }
            if let Some(b) = bias {
                sum += b[o];
            }
            // Fused ReLU: max(0, sum)
            output[output_offset + o] = sum.max(0.0);
        }
    }

    // Build output shape: leading dims from input + [out_features]
    let mut out_shape = if input_shape.len() > 1 {
        input_shape[..input_shape.len() - 1].to_vec()
    } else {
        Vec::new()
    };
    out_shape.push(out_features);

    Ok((output, out_shape))
}

/// Fused layer normalisation + linear projection.
///
/// Computes `output = layer_norm(input, norm_weight, norm_bias, eps) @ weight.T`
/// in a fused pass: each row of `input` is normalised and immediately projected,
/// avoiding an intermediate normalised-activations tensor.
///
/// # Arguments
/// - `input`, `shape`: Input tensor with shape `[batch, in_features]`.
/// - `weight`, `weight_shape`: Linear weight matrix `[out_features, in_features]`.
/// - `norm_weight` (γ): Layer-norm gain vector of length `in_features`.
/// - `norm_bias` (β): Layer-norm bias vector of length `in_features`.
/// - `eps`: Epsilon for variance stabilisation (e.g. `1e-5`).
///
/// # Returns
/// `(output, output_shape)` where `output_shape = [batch, out_features]`.
pub fn fused_layer_norm_linear(
    input: &[f32],
    shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    norm_weight: &[f32],
    norm_bias: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<usize>)> {
    // ---- Shape validation ----
    if shape.len() < 1 {
        return Err(TensorError::InvalidShape {
            operation: "fused_layer_norm_linear".to_string(),
            reason: "shape must have at least 1 dimension".to_string(),
            shape: Some(shape.to_vec()),
            context: None,
        });
    }
    if weight_shape.len() != 2 {
        return Err(TensorError::InvalidShape {
            operation: "fused_layer_norm_linear".to_string(),
            reason: format!("weight_shape must be 2D [out, in], got {weight_shape:?}"),
            shape: Some(weight_shape.to_vec()),
            context: None,
        });
    }

    let in_features = *shape.last().expect("checked above");
    let out_features = weight_shape[0];
    let weight_in = weight_shape[1];

    if in_features != weight_in {
        return Err(TensorError::ShapeMismatch {
            operation: "fused_layer_norm_linear".to_string(),
            expected: format!("in_features={in_features}"),
            got: format!("weight in_features={weight_in}"),
            context: None,
        });
    }
    if norm_weight.len() != in_features {
        return Err(TensorError::ShapeMismatch {
            operation: "fused_layer_norm_linear".to_string(),
            expected: format!("norm_weight length={in_features}"),
            got: format!("{}", norm_weight.len()),
            context: None,
        });
    }
    if norm_bias.len() != in_features {
        return Err(TensorError::ShapeMismatch {
            operation: "fused_layer_norm_linear".to_string(),
            expected: format!("norm_bias length={in_features}"),
            got: format!("{}", norm_bias.len()),
            context: None,
        });
    }

    let input_expected: usize = shape.iter().product();
    let weight_expected: usize = weight_shape.iter().product();
    if input.len() != input_expected {
        return Err(TensorError::InvalidArgument {
            operation: "fused_layer_norm_linear".to_string(),
            reason: format!(
                "input length {} != product of shape {:?} ({})",
                input.len(),
                shape,
                input_expected
            ),
            context: None,
        });
    }
    if weight.len() != weight_expected {
        return Err(TensorError::InvalidArgument {
            operation: "fused_layer_norm_linear".to_string(),
            reason: format!(
                "weight length {} != product of weight_shape {:?} ({})",
                weight.len(),
                weight_shape,
                weight_expected
            ),
            context: None,
        });
    }

    let effective_batch = if shape.len() == 1 {
        1
    } else {
        shape[..shape.len() - 1].iter().product::<usize>().max(1)
    };

    let out_len = effective_batch * out_features;
    let mut output = vec![0.0f32; out_len];

    // Temporary buffer for normalised row (avoids heap allocation per row
    // by reusing a single Vec).
    let mut normed_row = vec![0.0f32; in_features];

    for n in 0..effective_batch {
        let input_offset = n * in_features;
        let output_offset = n * out_features;
        let row = &input[input_offset..input_offset + in_features];

        // --- Layer normalisation of this row ---
        let mean = row.iter().sum::<f32>() / (in_features as f32);
        let var = row
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / (in_features as f32);
        let inv_std = 1.0 / (var + eps).sqrt();

        for (f, (&x, nw)) in row.iter().zip(norm_weight.iter()).enumerate() {
            normed_row[f] = nw * (x - mean) * inv_std + norm_bias[f];
        }

        // --- Linear projection (no ReLU) ---
        for o in 0..out_features {
            let weight_offset = o * in_features;
            let mut sum = 0.0f32;
            for f in 0..in_features {
                sum += normed_row[f] * weight[weight_offset + f];
            }
            output[output_offset + o] = sum;
        }
    }

    let mut out_shape = if shape.len() > 1 {
        shape[..shape.len() - 1].to_vec()
    } else {
        Vec::new()
    };
    out_shape.push(out_features);

    Ok((output, out_shape))
}

/// Stand-alone numerically stable layer normalisation.
///
/// Normalises the last dimension of `input` (treated as `in_features`) so
/// that each sample has mean ≈ 0 and standard deviation ≈ 1, then applies the
/// affine transform `y = gamma * x_normalised + beta`.
///
/// # Arguments
/// - `input`, `shape`: Tensor with shape `[*, in_features]`.
/// - `weight` (γ): Gain vector of length `in_features`.
/// - `bias` (β): Offset vector of length `in_features`.
/// - `eps`: Small constant for numerical stability (e.g. `1e-5`).
///
/// # Returns
/// Normalised output as a flat `Vec<f32>` with the same length as `input`.
pub fn layer_norm(
    input: &[f32],
    shape: &[usize],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    if shape.is_empty() {
        return Err(TensorError::InvalidShape {
            operation: "layer_norm".to_string(),
            reason: "shape must have at least 1 dimension".to_string(),
            shape: Some(shape.to_vec()),
            context: None,
        });
    }

    let in_features = *shape.last().expect("checked above");

    if weight.len() != in_features {
        return Err(TensorError::ShapeMismatch {
            operation: "layer_norm".to_string(),
            expected: format!("weight length={in_features}"),
            got: format!("{}", weight.len()),
            context: None,
        });
    }
    if bias.len() != in_features {
        return Err(TensorError::ShapeMismatch {
            operation: "layer_norm".to_string(),
            expected: format!("bias length={in_features}"),
            got: format!("{}", bias.len()),
            context: None,
        });
    }

    let total: usize = shape.iter().product();
    if input.len() != total {
        return Err(TensorError::InvalidArgument {
            operation: "layer_norm".to_string(),
            reason: format!(
                "input length {} != product of shape {:?} ({})",
                input.len(),
                shape,
                total
            ),
            context: None,
        });
    }

    let batch = total / in_features;
    let mut output = vec![0.0f32; total];

    for n in 0..batch {
        let offset = n * in_features;
        let row = &input[offset..offset + in_features];

        let mean = row.iter().sum::<f32>() / (in_features as f32);
        let var = row
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / (in_features as f32);
        let inv_std = 1.0 / (var + eps).sqrt();

        for f in 0..in_features {
            output[offset + f] = weight[f] * (row[f] - mean) * inv_std + bias[f];
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- layer_norm ----

    #[test]
    fn test_layer_norm_mean_zero_std_one() {
        // Without affine: gamma=1, beta=0 → mean≈0, std≈1 after normalisation
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3]; // batch=2, features=3
        let weight = vec![1.0f32; 3];
        let bias = vec![0.0f32; 3];

        let output = layer_norm(&input, &shape, &weight, &bias, 1e-5)
            .expect("layer_norm should succeed");

        assert_eq!(output.len(), input.len());

        // Check each sample independently
        for n in 0..2 {
            let row = &output[n * 3..(n + 1) * 3];
            let mean: f32 = row.iter().sum::<f32>() / 3.0;
            let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 3.0;
            assert!(
                mean.abs() < 1e-5,
                "sample {n} mean should be ~0, got {mean}"
            );
            assert!(
                (var.sqrt() - 1.0).abs() < 1e-4,
                "sample {n} std should be ~1, got {}",
                var.sqrt()
            );
        }
    }

    #[test]
    fn test_layer_norm_affine_transform() {
        // With gamma=2, beta=1: output = 2 * normalised + 1
        let input = vec![10.0f32, 20.0, 30.0];
        let shape = vec![3];
        let weight = vec![2.0f32; 3];
        let bias_vec = vec![1.0f32; 3];

        let output = layer_norm(&input, &shape, &weight, &bias_vec, 1e-5)
            .expect("layer_norm should succeed");

        // mean of output (gamma=2, beta=1): mean = 2*0 + 1 = 1
        let mean: f32 = output.iter().sum::<f32>() / 3.0;
        assert!((mean - 1.0).abs() < 1e-4, "affine output mean should be beta=1, got {mean}");
    }

    #[test]
    fn test_layer_norm_constant_input() {
        // All-same input: variance = 0 → eps protects; output = beta
        let input = vec![5.0f32; 4];
        let shape = vec![4];
        let weight = vec![1.0f32; 4];
        let bias_vec = vec![3.0f32; 4];

        let output = layer_norm(&input, &shape, &weight, &bias_vec, 1e-5)
            .expect("layer_norm should succeed on zero-variance input");

        for (i, &o) in output.iter().enumerate() {
            assert!(
                o.is_finite(),
                "output[{i}] must be finite for zero-variance input, got {o}"
            );
        }
    }

    // ---- fused_linear_relu ----

    #[test]
    fn test_fused_linear_relu_known_weights() {
        // input: [[1, 2]] (1 sample, 2 features)
        // weight: [[1, 0], [0, 1], [-1, -1]]  (3 outputs)
        // bias: [0, 0, 10]
        // pre_activation: [1, 2, -1-2+10] = [1, 2, 7]
        // after relu: [1, 2, 7]
        let input = vec![1.0f32, 2.0];
        let input_shape = vec![1, 2];
        let weight = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0];
        let weight_shape = vec![3, 2];
        let bias = vec![0.0f32, 0.0, 10.0];

        let (out, out_shape) =
            fused_linear_relu(&input, &input_shape, &weight, &weight_shape, Some(&bias))
                .expect("fused_linear_relu should succeed");

        assert_eq!(out_shape, vec![1, 3]);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6, "out[0] expected 1.0, got {}", out[0]);
        assert!((out[1] - 2.0).abs() < 1e-6, "out[1] expected 2.0, got {}", out[1]);
        assert!((out[2] - 7.0).abs() < 1e-6, "out[2] expected 7.0, got {}", out[2]);
    }

    #[test]
    fn test_fused_linear_relu_negative_clipped() {
        // All negative pre-activations → all outputs must be 0 after ReLU
        let input = vec![-1.0f32, -2.0];
        let input_shape = vec![1, 2];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0]; // 2 outputs, 2 inputs
        let weight_shape = vec![2, 2];
        // pre_act = [-3, -3]; after relu = [0, 0]

        let (out, _) =
            fused_linear_relu(&input, &input_shape, &weight, &weight_shape, None)
                .expect("fused_linear_relu should succeed");

        for (i, &x) in out.iter().enumerate() {
            assert_eq!(x, 0.0, "ReLU of negative input must be 0, got out[{i}]={x}");
        }
    }

    #[test]
    fn test_fused_linear_relu_no_bias() {
        // Simple 1-feature → 1-feature linear with no bias; relu(2*3) = 6
        let input = vec![3.0f32];
        let input_shape = vec![1, 1];
        let weight = vec![2.0f32];
        let weight_shape = vec![1, 1];

        let (out, out_shape) =
            fused_linear_relu(&input, &input_shape, &weight, &weight_shape, None)
                .expect("fused_linear_relu should succeed");

        assert_eq!(out_shape, vec![1, 1]);
        assert!((out[0] - 6.0).abs() < 1e-6, "expected 6.0, got {}", out[0]);
    }

    #[test]
    fn test_fused_linear_relu_shape_error() {
        // Mismatched weight dimensions should return Err
        let input = vec![1.0f32, 2.0];
        let input_shape = vec![1, 2];
        let weight = vec![1.0f32, 0.0, 0.0]; // weight in=3, not 2
        let weight_shape = vec![1, 3];

        let result = fused_linear_relu(&input, &input_shape, &weight, &weight_shape, None);
        assert!(result.is_err(), "mismatched weight dimensions must return Err");
    }

    // ---- fused_layer_norm_linear ----

    #[test]
    fn test_fused_layer_norm_linear_output_shape() {
        let batch = 4;
        let in_feat = 8;
        let out_feat = 3;
        let input: Vec<f32> = (0..batch * in_feat).map(|i| i as f32).collect();
        let shape = vec![batch, in_feat];
        let weight: Vec<f32> = vec![0.1f32; out_feat * in_feat];
        let weight_shape = vec![out_feat, in_feat];
        let norm_weight = vec![1.0f32; in_feat];
        let norm_bias = vec![0.0f32; in_feat];

        let (out, out_shape) = fused_layer_norm_linear(
            &input,
            &shape,
            &weight,
            &weight_shape,
            &norm_weight,
            &norm_bias,
            1e-5,
        )
        .expect("fused_layer_norm_linear should succeed");

        assert_eq!(out_shape, vec![batch, out_feat]);
        assert_eq!(out.len(), batch * out_feat);
    }

    #[test]
    fn test_fused_layer_norm_linear_no_nan() {
        let batch = 2;
        let in_feat = 4;
        let out_feat = 4;
        let input: Vec<f32> = (0..batch * in_feat).map(|i| (i as f32 - 3.5) * 2.0).collect();
        let shape = vec![batch, in_feat];
        let weight: Vec<f32> = vec![0.5f32; out_feat * in_feat];
        let weight_shape = vec![out_feat, in_feat];
        let norm_weight = vec![1.0f32; in_feat];
        let norm_bias = vec![0.0f32; in_feat];

        let (out, _) = fused_layer_norm_linear(
            &input,
            &shape,
            &weight,
            &weight_shape,
            &norm_weight,
            &norm_bias,
            1e-5,
        )
        .expect("fused_layer_norm_linear should succeed");

        for (i, &x) in out.iter().enumerate() {
            assert!(x.is_finite(), "output[{i}] must be finite, got {x}");
        }
    }
}
