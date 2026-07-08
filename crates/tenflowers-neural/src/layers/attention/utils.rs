//! Attention utility functions
//!
//! This module provides utility functions for attention mechanisms including
//! mask creation, positional encodings, and attention pattern analysis.

use scirs2_core::num_traits::{Float, One, Zero};
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
    use scirs2_core::ndarray::Array2;

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
    use scirs2_core::ndarray::Array2;

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

/// Combine an optional additive attention mask and an optional key-padding mask
/// into a single additive bias tensor of shape `[batch, tgt_len, src_len]`.
///
/// This is the shared masking primitive behind the transformer FFI bindings, so a
/// `key_padding_mask` is folded into the real attention-score math rather than
/// being silently ignored.
///
/// * `attn_mask` — optional `[tgt_len, src_len]` additive mask (already expressed
///   in the additive `-1e9`/`0` convention). It is broadcast across the batch.
/// * `key_padding_mask` — optional `[batch, src_len]` mask following the PyTorch
///   convention where a *nonzero* entry marks a key position that must be ignored.
///   Every masked key contributes an additive `-1e9` bias to all query rows.
///
/// Returns `Ok(None)` when both inputs are absent, exactly reproducing the
/// unmasked attention path. Otherwise returns a materialised additive bias to be
/// added to the raw attention scores before the softmax.
///
/// The `-1e9` sentinel (instead of `-inf`) matches `create_padding_mask`,
/// `create_causal_mask`, and `apply_attention_mask` in this module and avoids NaN
/// propagation should an entire mask row degenerate.
pub fn combine_attention_masks<T>(
    attn_mask: Option<&Tensor<T>>,
    key_padding_mask: Option<&Tensor<T>>,
    batch: usize,
    tgt_len: usize,
    src_len: usize,
) -> Result<Option<Tensor<T>>>
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
    // Both masks absent: no additive bias, reproducing the unmasked path exactly.
    if attn_mask.is_none() && key_padding_mask.is_none() {
        return Ok(None);
    }

    let neg_bias = T::from(-1e9).unwrap_or_else(|| T::zero() - T::one());

    // Validate and read the additive attention mask ([tgt_len, src_len]).
    let attn_vec = match attn_mask {
        Some(mask) => {
            let dims = mask.shape().dims();
            if dims.len() != 2 || dims[0] != tgt_len || dims[1] != src_len {
                return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                    "combine_attention_masks: attn_mask must have shape [{}, {}], got {:?}",
                    tgt_len, src_len, dims
                )));
            }
            Some(mask.to_vec()?)
        }
        None => None,
    };

    // Validate and read the key-padding mask ([batch, src_len]).
    let key_padding_vec = match key_padding_mask {
        Some(mask) => {
            let dims = mask.shape().dims();
            if dims.len() != 2 || dims[0] != batch || dims[1] != src_len {
                return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                    "combine_attention_masks: key_padding_mask must have shape [{}, {}], got {:?}",
                    batch, src_len, dims
                )));
            }
            Some(mask.to_vec()?)
        }
        None => None,
    };

    // Materialise the [batch, tgt_len, src_len] additive bias. A materialised
    // triple loop (rather than tensor broadcasting) keeps the combination
    // directly hand-verifiable and independent of broadcasting semantics.
    let mut out = Vec::with_capacity(batch * tgt_len * src_len);
    for b in 0..batch {
        for t in 0..tgt_len {
            for s in 0..src_len {
                let attn_bias = match attn_vec {
                    Some(ref values) => values[t * src_len + s],
                    None => T::zero(),
                };
                let pad_bias = match key_padding_vec {
                    Some(ref values) if values[b * src_len + s] != T::zero() => neg_bias,
                    _ => T::zero(),
                };
                out.push(attn_bias + pad_bias);
            }
        }
    }

    Ok(Some(Tensor::from_vec(out, &[batch, tgt_len, src_len])?))
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
        + bytemuck::Zeroable
        + std::iter::Sum,
{
    // Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V

    // Step 1: Compute Q @ K^T
    // For 3D tensors [batch, seq_len, d_model], we want to transpose last two dims: [batch, d_model, seq_len]
    let key_transposed = tenflowers_core::ops::manipulation::transpose_axes(key, Some(&[0, 2, 1]))?;
    let attention_scores = tenflowers_core::ops::matmul(query, &key_transposed)?;

    // Step 2: Scale by sqrt(d_k)
    let d_k = query.shape().dims().last().unwrap_or(&1);
    let scale = T::from(1.0 / (*d_k as f64).sqrt()).unwrap_or_else(T::one);
    let scale_tensor = Tensor::from_array(scirs2_core::ndarray::arr0(scale).into_dyn());
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
    let attention_weights = masked_scores.softmax(Some(-1))?;

    // Step 5: Apply dropout if in training mode
    let final_weights = if training && dropout_prob > 0.0 && dropout_prob < 1.0 {
        // Apply dropout to attention weights
        use scirs2_core::ndarray::{ArrayD, IxDyn};
        use scirs2_core::random::thread_rng;

        let shape = attention_weights.shape().dims();
        let total_elements: usize = shape.iter().product();
        let mut rng = thread_rng();

        // Generate dropout mask
        let mut mask_data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let random_val: f64 = rng.gen_range(0.0..1.0);
            if random_val < dropout_prob as f64 {
                mask_data.push(T::zero()); // Drop this element
            } else {
                mask_data.push(T::one()); // Keep this element
            }
        }

        let mask_array = ArrayD::from_shape_vec(IxDyn(shape), mask_data).map_err(|_| {
            tenflowers_core::TensorError::invalid_shape_simple(
                "Failed to create dropout mask".to_string(),
            )
        })?;
        let mask = Tensor::from_array(mask_array);

        // Apply mask and scale by 1/(1-dropout_prob) for inverted dropout
        let dropped = attention_weights.mul(&mask)?;
        let keep_prob = 1.0 - dropout_prob;
        let scale = T::from(1.0 / keep_prob as f64).unwrap_or(T::one());
        dropped.mul(&Tensor::from_scalar(scale))?
    } else if dropout_prob >= 1.0 && training {
        // If dropout is 1.0, return zeros
        Tensor::zeros(attention_weights.shape().dims())
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
    use scirs2_core::ndarray::Array2;

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

/// Compute attention pattern statistics from the actual attention weights.
///
/// All four statistics are computed from the real tensor data rather than fabricated
/// constants. The final tensor dimension is treated as the "key" axis over which a
/// query attends (post-softmax attention rows are probability distributions), and the
/// penultimate dimension as the "query" axis used for the locality measure.
///
/// - `entropy`: mean Shannon entropy (natural log) of the per-query attention rows,
///   measuring how spread out attention is. A one-hot row has entropy 0; a uniform
///   row over `n` keys has entropy `ln(n)`.
/// - `sparsity`: fraction of weight elements that are effectively zero (`|w| < 1e-6`).
/// - `max_attention`: the single largest attention weight observed.
/// - `locality_score`: mean attention mass that lands on the diagonal band
///   `|query_index - key_index| <= 1`, i.e. how much each query attends to nearby
///   positions. Requires at least a 2-D pattern; returns 0 when no query/key axes
///   exist.
///
/// Returns an honest error for non-CPU tensors whose data cannot be inspected.
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
    let data = attention_weights.as_slice().ok_or_else(|| {
        tenflowers_core::TensorError::device_error_simple(
            "analyze_attention_patterns requires CPU-resident attention weights; the \
             tensor data could not be accessed (non-CPU tensor). Fabricated statistics \
             are not produced."
                .to_string(),
        )
    })?;

    if data.is_empty() {
        return Err(tenflowers_core::TensorError::invalid_argument(
            "analyze_attention_patterns received an empty attention tensor".to_string(),
        ));
    }

    let dims = attention_weights.shape().dims();
    // The last axis is the key axis (attention distribution per query row).
    let key_dim = dims.last().copied().unwrap_or(data.len()).max(1);
    let num_rows = data.len() / key_dim;

    // --- max_attention: real maximum across all weights ---
    let mut running_max = data[0];
    for &value in &data[1..] {
        if value > running_max {
            running_max = value;
        }
    }
    let max_attention = running_max;

    // --- sparsity: real fraction of near-zero weights ---
    let zero_threshold = T::from(1e-6_f64).unwrap_or_else(T::zero);
    let near_zero_count = data
        .iter()
        .filter(|&&value| value.abs() < zero_threshold)
        .count();
    let sparsity = T::from(near_zero_count as f64 / data.len() as f64).unwrap_or_default();

    // --- entropy: mean Shannon entropy of each per-query attention row ---
    // H(row) = -sum_k p_k * ln(p_k), using the (possibly unnormalised) row mass so the
    // measure is well defined even if the weights do not sum exactly to one.
    let mut entropy_accum = 0.0_f64;
    for row in data.chunks(key_dim) {
        let row_sum: f64 = row.iter().map(|&value| value.to_f64().unwrap_or(0.0)).sum();
        if row_sum <= 0.0 {
            continue;
        }
        let mut row_entropy = 0.0_f64;
        for &value in row {
            let p = value.to_f64().unwrap_or(0.0) / row_sum;
            if p > 0.0 {
                row_entropy -= p * p.ln();
            }
        }
        entropy_accum += row_entropy;
    }
    let entropy = T::from(entropy_accum / num_rows as f64).unwrap_or_default();

    // --- locality_score: mean attention mass on the |query - key| <= 1 diagonal band ---
    // Only meaningful when there are explicit query and key axes (rank >= 2).
    let locality_score = if dims.len() >= 2 {
        let query_dim = dims[dims.len() - 2].max(1);
        let num_patterns = num_rows / query_dim.max(1);
        let mut locality_accum = 0.0_f64;
        let mut counted_rows = 0_usize;

        for pattern_idx in 0..num_patterns {
            for query_idx in 0..query_dim {
                let row_offset = (pattern_idx * query_dim + query_idx) * key_dim;
                let row = &data[row_offset..row_offset + key_dim];
                let row_sum: f64 = row.iter().map(|&value| value.to_f64().unwrap_or(0.0)).sum();
                if row_sum <= 0.0 {
                    continue;
                }
                let mut local_mass = 0.0_f64;
                let lower = query_idx.saturating_sub(1);
                let upper = (query_idx + 1).min(key_dim.saturating_sub(1));
                for (key_idx, &value) in row.iter().enumerate() {
                    if key_idx >= lower && key_idx <= upper {
                        local_mass += value.to_f64().unwrap_or(0.0);
                    }
                }
                locality_accum += local_mass / row_sum;
                counted_rows += 1;
            }
        }

        if counted_rows == 0 {
            T::zero()
        } else {
            T::from(locality_accum / counted_rows as f64).unwrap_or_default()
        }
    } else {
        T::zero()
    };

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
        use scirs2_core::ndarray::array;

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
        let (output, weights) = result.expect("test: result should be valid");

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
        use scirs2_core::ndarray::array;

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
        use scirs2_core::ndarray::array;

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
    fn test_attention_stats_real_max_and_sparsity() {
        use scirs2_core::ndarray::array;

        // Known weights: real max is 0.9; half of the 4 elements are exactly zero.
        let weights_data = array![[0.9_f32, 0.0], [0.0, 0.1]];
        let weights = Tensor::from_array(weights_data.into_dyn());

        let stats =
            analyze_attention_patterns(&weights).expect("Failed to analyze attention patterns");

        // max_attention must equal the actual maximum element, not a fabricated 1.0.
        assert!((stats.max_attention - 0.9).abs() < 1e-6);
        // sparsity must equal the actual fraction of near-zero elements (2 of 4 = 0.5).
        assert!((stats.sparsity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_attention_stats_real_entropy_one_hot_vs_uniform() {
        use scirs2_core::ndarray::array;

        // A one-hot row has entropy 0; a uniform row over 4 keys has entropy ln(4).
        let one_hot = Tensor::from_array(array![[1.0_f32, 0.0, 0.0, 0.0]].into_dyn());
        let uniform = Tensor::from_array(array![[0.25_f32, 0.25, 0.25, 0.25]].into_dyn());

        let one_hot_stats =
            analyze_attention_patterns(&one_hot).expect("Failed to analyze one-hot");
        let uniform_stats =
            analyze_attention_patterns(&uniform).expect("Failed to analyze uniform");

        // Entropy is a real computation, not a constant.
        assert!(
            one_hot_stats.entropy.abs() < 1e-6,
            "one-hot entropy should be ~0"
        );
        let expected_uniform = (4.0_f32).ln();
        assert!(
            (uniform_stats.entropy - expected_uniform).abs() < 1e-5,
            "uniform entropy should be ln(4)"
        );
    }

    #[test]
    fn test_attention_stats_real_locality() {
        use scirs2_core::ndarray::array;

        // Perfectly diagonal attention (each query attends only to its own position)
        // must yield locality 1.0; a fully anti-diagonal pattern attends far away.
        let diagonal = Tensor::from_array(
            array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_dyn(),
        );
        let far = Tensor::from_array(
            array![[0.0_f32, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]].into_dyn(),
        );

        let diag_stats = analyze_attention_patterns(&diagonal).expect("Failed to analyze diagonal");
        let far_stats = analyze_attention_patterns(&far).expect("Failed to analyze far");

        assert!(
            (diag_stats.locality_score - 1.0).abs() < 1e-6,
            "diagonal attention must be fully local"
        );
        // The far pattern's locality must be strictly lower than the diagonal's.
        assert!(far_stats.locality_score < diag_stats.locality_score);
    }

    #[test]
    fn test_attention_stats_empty_errors() {
        // An empty attention tensor must surface an honest error, not fabricated stats.
        let empty = Tensor::<f32>::zeros(&[0]);
        let result = analyze_attention_patterns(&empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotary_position_embedding() {
        use scirs2_core::ndarray::array;

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
        let output = result.expect("test: result should be valid");

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
        use scirs2_core::ndarray::array;

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

    #[test]
    fn test_combine_masks_both_none() {
        // No masks must reproduce the unmasked path (no additive bias at all).
        let result =
            combine_attention_masks::<f32>(None, None, 2, 3, 3).expect("combine should succeed");
        assert!(result.is_none(), "no masks must yield no bias");
    }

    #[test]
    fn test_combine_masks_attn_only() {
        // attn_mask [tgt=2, src=2]: block key 1 for query 0 only.
        let attn = Tensor::from_vec(vec![0.0_f32, -1.0e9, 0.0, 0.0], &[2, 2]).expect("attn tensor");
        let combined = combine_attention_masks(Some(&attn), None, 2, 2, 2)
            .expect("combine")
            .expect("bias present");
        assert_eq!(combined.shape().dims(), &[2, 2, 2]);
        let data = combined.to_vec().expect("vec");
        // Broadcast across batch: index [b,t,s] = (b*2 + t)*2 + s.
        let idx = |b: usize, t: usize, s: usize| (b * 2 + t) * 2 + s;
        for b in 0..2 {
            assert_eq!(data[idx(b, 0, 0)], 0.0);
            assert!(
                data[idx(b, 0, 1)] < -1e8,
                "attn-blocked key must be very negative"
            );
            assert_eq!(data[idx(b, 1, 0)], 0.0);
            assert_eq!(data[idx(b, 1, 1)], 0.0);
        }
    }

    #[test]
    fn test_combine_masks_key_padding_only() {
        // key_padding_mask [batch=2, src=3]; nonzero = ignore.
        // Batch 0 masks key 2; batch 1 masks key 0.
        let kpm =
            Tensor::from_vec(vec![0.0_f32, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]).expect("kpm tensor");
        let (tgt_len, src_len) = (2usize, 3usize);
        let combined = combine_attention_masks(None, Some(&kpm), 2, tgt_len, src_len)
            .expect("combine")
            .expect("bias present");
        assert_eq!(combined.shape().dims(), &[2, 2, 3]);
        let data = combined.to_vec().expect("vec");
        let idx = |b: usize, t: usize, s: usize| (b * tgt_len + t) * src_len + s;
        // Batch 0: only key 2 masked, for every query row.
        for t in 0..tgt_len {
            assert_eq!(data[idx(0, t, 0)], 0.0);
            assert_eq!(data[idx(0, t, 1)], 0.0);
            assert!(data[idx(0, t, 2)] < -1e8, "batch 0 key 2 must be masked");
        }
        // Batch 1: only key 0 masked, for every query row.
        for t in 0..tgt_len {
            assert!(data[idx(1, t, 0)] < -1e8, "batch 1 key 0 must be masked");
            assert_eq!(data[idx(1, t, 1)], 0.0);
            assert_eq!(data[idx(1, t, 2)], 0.0);
        }
    }

    #[test]
    fn test_combine_masks_both_additive_distinct_positions() {
        // attn_mask [tgt=1, src=3] blocks key 1; key_padding blocks key 2.
        let attn = Tensor::from_vec(vec![0.0_f32, -1.0e9, 0.0], &[1, 3]).expect("attn");
        let kpm = Tensor::from_vec(vec![0.0_f32, 0.0, 1.0], &[1, 3]).expect("kpm");
        let combined = combine_attention_masks(Some(&attn), Some(&kpm), 1, 1, 3)
            .expect("combine")
            .expect("bias present");
        let data = combined.to_vec().expect("vec");
        assert_eq!(data[0], 0.0); // neither mask
        assert!(data[1] < -1e8, "attn-blocked key 1");
        assert!(data[2] < -1e8, "key-padding-blocked key 2");
    }

    #[test]
    fn test_combine_masks_both_additive_same_position() {
        // Both masks target key 1: their contributions must add.
        let attn = Tensor::from_vec(vec![0.0_f32, -1.0e9], &[1, 2]).expect("attn");
        let kpm = Tensor::from_vec(vec![0.0_f32, 1.0], &[1, 2]).expect("kpm");
        let combined = combine_attention_masks(Some(&attn), Some(&kpm), 1, 1, 2)
            .expect("combine")
            .expect("bias present");
        let data = combined.to_vec().expect("vec");
        assert_eq!(data[0], 0.0);
        // -1e9 (attn) + -1e9 (padding) ≈ -2e9.
        assert!(
            data[1] < -1.5e9,
            "additive combination should stack, got {}",
            data[1]
        );
    }

    #[test]
    fn test_combine_masks_attn_shape_mismatch() {
        // Claimed tgt=3, src=2 but the tensor is [2, 2].
        let attn = Tensor::from_vec(vec![0.0_f32; 4], &[2, 2]).expect("attn");
        let result = combine_attention_masks(Some(&attn), None, 1, 3, 2);
        assert!(result.is_err(), "attn_mask shape mismatch must error");
    }

    #[test]
    fn test_combine_masks_key_padding_shape_mismatch() {
        // Claimed batch=1, src=3 but the tensor is [2, 3].
        let kpm = Tensor::from_vec(vec![0.0_f32; 6], &[2, 3]).expect("kpm");
        let result = combine_attention_masks(None, Some(&kpm), 1, 2, 3);
        assert!(
            result.is_err(),
            "key_padding_mask shape mismatch must error"
        );
    }
}
