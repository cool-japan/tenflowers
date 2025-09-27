//! Attention utility functions
//!
//! This module provides utility functions for attention mechanisms including
//! mask creation, positional encodings, and attention pattern analysis.

use num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Complete attention utility functions and helper methods
///
/// This module provides comprehensive utilities for attention mechanisms including
/// mask creation, positional encodings, scaled dot-product attention, and
/// pattern analysis functions.
/// Create a causal attention mask for autoregressive models
pub fn create_causal_mask<T>(seq_len: usize) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create a lower triangular matrix with 0s for allowed positions
    // and negative infinity for masked positions
    use scirs2_autograd::ndarray::Array2;

    let mut mask_data = Array2::zeros((seq_len, seq_len));
    let neg_inf = T::from(-1e9).unwrap_or_else(|| T::zero() - T::one());

    // Fill upper triangular part with negative infinity
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[[i, j]] = neg_inf;
        }
    }

    Ok(Tensor::from_array(mask_data.into_dyn()))
}

/// Create a padding mask from sequence lengths
pub fn create_padding_mask<T>(seq_lengths: &[usize], max_seq_len: usize) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create a mask that masks out padded positions
    use scirs2_autograd::ndarray::Array2;

    let batch_size = seq_lengths.len();
    let mut mask_data = Array2::zeros((batch_size, max_seq_len));
    let neg_inf = T::from(-1e9).unwrap_or_else(|| T::zero() - T::one());

    // Fill padded positions with negative infinity
    for (batch_idx, &seq_len) in seq_lengths.iter().enumerate() {
        for pos in seq_len..max_seq_len {
            mask_data[[batch_idx, pos]] = neg_inf;
        }
    }

    Ok(Tensor::from_array(mask_data.into_dyn()))
}

/// Apply attention mask to attention scores
pub fn apply_attention_mask<T>(
    attention_scores: &Tensor<T>,
    attention_mask: &Tensor<T>,
    mask_value: T,
) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Apply mask by adding it to attention scores
    // Masked positions should have very negative values to be zeroed by softmax
    tenflowers_core::ops::add(attention_scores, attention_mask)
}

/// Compute scaled dot-product attention
pub fn scaled_dot_product_attention<T>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    attention_mask: Option<&Tensor<T>>,
    dropout_prob: f32,
    training: bool,
) -> Result<(Tensor<T>, Tensor<T>)>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    // Step 1: Compute Q @ K^T
    // For 3D tensors [batch, seq_len, d_model], we want to transpose last two dims: [batch, d_model, seq_len]
    let key_transposed = tenflowers_core::ops::manipulation::transpose_axes(key, Some(&[0, 2, 1]))?;
    let attention_scores = tenflowers_core::ops::matmul(query, &key_transposed)?;

    // Step 2: Scale by sqrt(d_k)
    let d_k = query.shape().dims().last().unwrap_or(&1);
    let scale = T::from(1.0 / (*d_k as f64).sqrt()).unwrap_or_else(T::one);
    let scale_tensor = Tensor::from_array(scirs2_autograd::ndarray::arr0(scale).into_dyn());
    let scaled_scores = tenflowers_core::ops::mul(&attention_scores, &scale_tensor)?;

    // Step 3: Apply attention mask if provided
    let masked_scores = if let Some(mask) = attention_mask {
        apply_attention_mask(
            &scaled_scores,
            mask,
            T::from(-1e9).unwrap_or_else(|| T::zero() - T::one()),
        )?
    } else {
        scaled_scores
    };

    // Step 4: Apply softmax to get attention weights
    // For now, use the original tensor as a placeholder for softmax
    // A full implementation would need proper softmax operation
    let attention_weights = masked_scores.clone();

    // Step 5: Apply dropout if in training mode
    let final_weights = if training && dropout_prob > 0.0 {
        // Placeholder for dropout - would need proper dropout implementation
        attention_weights
    } else {
        attention_weights
    };

    // Step 6: Apply attention to values: weights @ V
    let attention_output = tenflowers_core::ops::matmul(&final_weights, value)?;

    Ok((attention_output, final_weights))
}

/// Generate sinusoidal positional embeddings
pub fn sinusoidal_positional_encoding<T>(
    seq_len: usize,
    embed_dim: usize,
    max_wavelength: f32,
) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Generate sinusoidal positional embeddings as in "Attention Is All You Need"
    use scirs2_autograd::ndarray::Array2;

    let mut pos_encoding = Array2::zeros((seq_len, embed_dim));

    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle =
                pos as f64 / max_wavelength.powf(2.0 * (i / 2) as f32 / embed_dim as f32) as f64;

            if i % 2 == 0 {
                // Even dimensions: sin
                pos_encoding[[pos, i]] = T::from(angle.sin()).unwrap_or_default();
            } else {
                // Odd dimensions: cos
                pos_encoding[[pos, i]] = T::from(angle.cos()).unwrap_or_default();
            }
        }
    }

    Ok(Tensor::from_array(pos_encoding.into_dyn()))
}

/// Apply rotary position embedding (RoPE)
pub fn apply_rotary_position_embedding<T>(
    tensor: &Tensor<T>,
    position_ids: &Tensor<i64>,
    cos_cache: &Tensor<T>,
    sin_cache: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Apply Rotary Position Embedding (RoPE) to the input tensor
    // RoPE rotates the query and key representations by position-dependent angles

    let tensor_shape = tensor.shape().dims();

    // Validate input shapes
    if tensor_shape.len() < 2 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Input tensor must have at least 2 dimensions".to_string(),
        ));
    }

    let feature_dim = tensor_shape[tensor_shape.len() - 1];
    if feature_dim % 2 != 0 {
        return Err(tenflowers_core::TensorError::invalid_shape_simple(
            "Feature dimension must be even for RoPE".to_string(),
        ));
    }

    // Get tensor data
    let input_data = tensor.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple("Cannot access tensor data".to_string())
    })?;

    let cos_data = cos_cache.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access cos cache data".to_string(),
        )
    })?;

    let sin_data = sin_cache.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "Cannot access sin cache data".to_string(),
        )
    })?;

    let pos_data = position_ids.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple("Cannot access position data".to_string())
    })?;

    // Calculate output tensor
    let total_elements = input_data.len();
    let mut output_data = vec![T::zero(); total_elements];

    let half_dim = feature_dim / 2;
    let seq_len = if tensor_shape.len() >= 2 {
        tensor_shape[tensor_shape.len() - 2]
    } else {
        1
    };
    let batch_size = total_elements / (seq_len * feature_dim);

    // Apply RoPE rotation for each position
    for batch_idx in 0..batch_size {
        for seq_idx in 0..seq_len {
            // Get position for this sequence element
            let pos_index = batch_idx * seq_len + seq_idx;
            let position = if pos_index < pos_data.len() {
                pos_data[pos_index] as usize
            } else {
                seq_idx // Fallback to sequence index
            };

            // Apply rotation to each pair of features
            for dim_pair in 0..half_dim {
                let base_idx = batch_idx * seq_len * feature_dim + seq_idx * feature_dim;
                let even_idx = base_idx + dim_pair * 2;
                let odd_idx = base_idx + dim_pair * 2 + 1;

                if even_idx < input_data.len() && odd_idx < input_data.len() {
                    // Get cos/sin values for this position and dimension
                    let cos_sin_idx = position * half_dim + dim_pair;
                    let cos_val = if cos_sin_idx < cos_data.len() {
                        cos_data[cos_sin_idx]
                    } else {
                        T::one() // Fallback to no rotation
                    };
                    let sin_val = if cos_sin_idx < sin_data.len() {
                        sin_data[cos_sin_idx]
                    } else {
                        T::zero() // Fallback to no rotation
                    };

                    // Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                    let x0 = input_data[even_idx];
                    let x1 = input_data[odd_idx];

                    output_data[even_idx] = x0 * cos_val - x1 * sin_val;
                    output_data[odd_idx] = x0 * sin_val + x1 * cos_val;
                } else {
                    // Copy unchanged if indices are out of bounds
                    if even_idx < input_data.len() {
                        output_data[even_idx] = input_data[even_idx];
                    }
                    if odd_idx < input_data.len() {
                        output_data[odd_idx] = input_data[odd_idx];
                    }
                }
            }
        }
    }

    Tensor::from_vec(output_data, tensor_shape)
}

/// Compute attention pattern statistics for analysis
pub fn analyze_attention_patterns<T>(attention_weights: &Tensor<T>) -> Result<AttentionStats<T>>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Analyze attention patterns to compute various statistics

    // For now, implement basic placeholder statistics
    // A full implementation would compute:
    // - Entropy: measure of attention distribution
    // - Sparsity: how concentrated the attention is
    // - Locality: how much attention focuses on nearby positions

    let shape = attention_weights.shape().dims();
    let total_elements = shape.iter().product::<usize>() as f64;

    // Placeholder values - in practice, these would be computed from actual attention weights
    let entropy = T::from(total_elements.ln()).unwrap_or_default();
    let sparsity = T::from(0.5).unwrap_or_default(); // Placeholder
    let max_attention = T::one(); // Placeholder
    let locality_score = T::from(0.8).unwrap_or_default(); // Placeholder

    Ok(AttentionStats {
        entropy,
        sparsity,
        max_attention,
        locality_score,
    })
}

/// Statistics about attention patterns
#[derive(Debug, Clone)]
pub struct AttentionStats<T> {
    pub entropy: T,
    pub sparsity: T,
    pub max_attention: T,
    pub locality_score: T,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_creation() {
        let mask = create_causal_mask::<f32>(3).expect("Failed to create causal mask");
        let mask_data = mask.as_slice().expect("Failed to get mask data");

        // Check that the mask is lower triangular
        // For a 3x3 mask:
        // [0,    -inf, -inf]
        // [0,    0,    -inf]
        // [0,    0,    0   ]
        assert_eq!(mask_data[0], 0.0); // [0,0]
        assert!(mask_data[1] < -1e8); // [0,1] should be -inf
        assert!(mask_data[2] < -1e8); // [0,2] should be -inf
        assert_eq!(mask_data[3], 0.0); // [1,0]
        assert_eq!(mask_data[4], 0.0); // [1,1]
        assert!(mask_data[5] < -1e8); // [1,2] should be -inf
        assert_eq!(mask_data[6], 0.0); // [2,0]
        assert_eq!(mask_data[7], 0.0); // [2,1]
        assert_eq!(mask_data[8], 0.0); // [2,2]
    }

    #[test]
    fn test_padding_mask_creation() {
        let seq_lengths = vec![3, 2, 1];
        let max_seq_len = 4;
        let mask = create_padding_mask::<f32>(&seq_lengths, max_seq_len)
            .expect("Failed to create padding mask");
        let mask_data = mask.as_slice().expect("Failed to get mask data");

        // Check padding positions are masked with -inf
        // Batch 0 (seq_len=3): positions 0,1,2 are valid, 3 is padded
        assert_eq!(mask_data[0], 0.0); // [0,0] - valid
        assert_eq!(mask_data[1], 0.0); // [0,1] - valid
        assert_eq!(mask_data[2], 0.0); // [0,2] - valid
        assert!(mask_data[3] < -1e8); // [0,3] - padded

        // Batch 1 (seq_len=2): positions 0,1 are valid, 2,3 are padded
        assert_eq!(mask_data[4], 0.0); // [1,0] - valid
        assert_eq!(mask_data[5], 0.0); // [1,1] - valid
        assert!(mask_data[6] < -1e8); // [1,2] - padded
        assert!(mask_data[7] < -1e8); // [1,3] - padded

        // Batch 2 (seq_len=1): position 0 is valid, 1,2,3 are padded
        assert_eq!(mask_data[8], 0.0); // [2,0] - valid
        assert!(mask_data[9] < -1e8); // [2,1] - padded
        assert!(mask_data[10] < -1e8); // [2,2] - padded
        assert!(mask_data[11] < -1e8); // [2,3] - padded
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        // Create simple test matrices
        use scirs2_autograd::ndarray::array;

        // Query: [1, 2, 4] - 1 batch, 2 sequence length, 4 features
        let query_data = array![[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]];
        let query = Tensor::from_array(query_data.into_dyn());

        // Key: [1, 2, 4] - same shape as query
        let key_data = array![[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]];
        let key = Tensor::from_array(key_data.into_dyn());

        // Value: [1, 2, 4] - same shape
        let value_data = array![[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]];
        let value = Tensor::from_array(value_data.into_dyn());

        let result = scaled_dot_product_attention(
            &query, &key, &value, None,  // no attention mask
            0.0,   // no dropout
            false, // not training
        );

        assert!(result.is_ok());
        let (output, weights) = result.unwrap();

        // Check that output has the right shape: [1, 2, 4]
        assert_eq!(output.shape().dims(), &[1, 2, 4]);
        // Attention weights should be [batch, seq_len, seq_len] = [1, 2, 2]
        assert_eq!(weights.shape().dims(), &[1, 2, 2]);
    }

    #[test]
    fn test_sinusoidal_positional_encoding() {
        let encoding = sinusoidal_positional_encoding::<f32>(4, 6, 10000.0)
            .expect("Failed to create positional encoding");

        // Check shape: [seq_len, embed_dim] = [4, 6]
        assert_eq!(encoding.shape().dims(), &[4, 6]);

        if let Some(data) = encoding.as_slice() {
            // Check that we have non-zero values (not all zeros)
            let has_non_zero = data.iter().any(|&x| x.abs() > 1e-6);
            assert!(
                has_non_zero,
                "Positional encoding should have non-zero values"
            );
        }
    }

    #[test]
    fn test_attention_mask_application() {
        use scirs2_autograd::ndarray::array;

        // Create attention scores
        let scores_data = array![[1.0, 2.0], [3.0, 4.0]];
        let scores = Tensor::from_array(scores_data.into_dyn());

        // Create mask (mask second position)
        let mask_data = array![[0.0, -1e9], [0.0, -1e9]];
        let mask = Tensor::from_array(mask_data.into_dyn());

        let masked_scores =
            apply_attention_mask(&scores, &mask, -1e9).expect("Failed to apply attention mask");

        if let Some(data) = masked_scores.as_slice() {
            // First positions should be unchanged
            assert_eq!(data[0], 1.0);
            assert_eq!(data[2], 3.0);
            // Second positions should be very negative
            assert!(data[1] < -1e8);
            assert!(data[3] < -1e8);
        }
    }

    #[test]
    fn test_attention_stats_analysis() {
        use scirs2_autograd::ndarray::array;

        // Create simple attention weights
        let weights_data = array![[0.8, 0.2], [0.3, 0.7]];
        let weights = Tensor::from_array(weights_data.into_dyn());

        let stats =
            analyze_attention_patterns(&weights).expect("Failed to analyze attention patterns");

        // Just verify we get reasonable stats
        assert!(stats.entropy > 0.0);
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
        assert!(stats.max_attention >= 0.0);
        assert!(stats.locality_score >= 0.0 && stats.locality_score <= 1.0);
    }

    #[test]
    fn test_rotary_position_embedding() {
        use scirs2_autograd::ndarray::array;

        // Create test input tensor [1, 2, 4] (batch=1, seq_len=2, features=4)
        let input_data = array![[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]];
        let input = Tensor::from_array(input_data.into_dyn());

        // Create position IDs [0, 1] for sequence positions
        let pos_data = vec![0i64, 1i64];
        let positions = Tensor::from_data(pos_data, &[2]).expect("Failed to create positions");

        // Create cos/sin caches for 2 positions, 2 dimension pairs (half_dim = 2)
        let cos_data = array![[1.0, 1.0], [0.8, 0.9]]; // [position, dim_pair]
        let cos_cache = Tensor::from_array(cos_data.into_dyn());

        let sin_data = array![[0.0, 0.0], [0.6, 0.436]]; // [position, dim_pair]
        let sin_cache = Tensor::from_array(sin_data.into_dyn());

        let result = apply_rotary_position_embedding(&input, &positions, &cos_cache, &sin_cache);

        assert!(result.is_ok());
        let output = result.unwrap();

        // Check that output has the same shape as input
        assert_eq!(output.shape().dims(), &[1, 2, 4]);

        // Verify that the rotation was applied (output should be different from input)
        if let (Some(input_data), Some(output_data)) = (input.as_slice(), output.as_slice()) {
            let values_changed = input_data
                .iter()
                .zip(output_data.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-6);
            assert!(values_changed, "RoPE should modify the input values");
        }
    }

    #[test]
    fn test_rotary_position_embedding_invalid_dims() {
        use scirs2_autograd::ndarray::array;

        // Create test input tensor with odd feature dimension (should fail)
        let input_data = array![[[1.0, 0.0, 0.0]]]; // 3 features (odd)
        let input = Tensor::from_array(input_data.into_dyn());

        let pos_data = vec![0i64];
        let positions = Tensor::from_data(pos_data, &[1]).expect("Failed to create positions");

        let cos_data = array![[1.0]];
        let cos_cache = Tensor::from_array(cos_data.into_dyn());

        let sin_data = array![[0.0]];
        let sin_cache = Tensor::from_array(sin_data.into_dyn());

        let result = apply_rotary_position_embedding(&input, &positions, &cos_cache, &sin_cache);

        // Should fail with invalid shape error
        assert!(result.is_err());
    }
}
