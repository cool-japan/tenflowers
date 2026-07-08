//! Ultra-High-Performance Multi-Head Attention implementation
//!
//! This module provides the standard multi-head attention mechanism with
//! SIMD-optimized operations, KV caching, Flash attention, and relative position bias.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive};
use std::sync::RwLock;
// Axis-permutation transpose (e.g. (0, 2, 1, 3)); the bare `transpose` re-export
// only reverses all axes, which is wrong for multi-head reshaping.
use tenflowers_core::ops::manipulation::transpose_axes;
use tenflowers_core::{Result, Tensor, TensorError};
// Note: SIMD optimizations available when scirs2_core::simd API is complete
use std::sync::Arc;

#[cfg(feature = "gpu")]
use std::any::TypeId;
#[cfg(feature = "gpu")]
use tenflowers_core::gpu::attention_ops::GpuAttentionOps;
#[cfg(feature = "gpu")]
use tenflowers_core::Device;

use super::KVCache;

/// Ultra-High-Performance Multi-Head Attention layer with advanced optimizations
///
/// Implements the attention mechanism from "Attention Is All You Need" with:
/// - Flash Attention for memory efficiency
/// - SIMD-optimized matrix operations
/// - KV caching for inference speed
/// - Relative position bias support
/// - Parallel computation across heads
///
/// <https://arxiv.org/abs/1706.03762>
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T>
where
    T: From<f32>,
{
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale_factor: f64,

    // Weight matrices with optimized storage
    query_weight: Tensor<T>,
    key_weight: Tensor<T>,
    value_weight: Tensor<T>,
    output_weight: Tensor<T>,

    // Biases (optional)
    query_bias: Option<Tensor<T>>,
    key_bias: Option<Tensor<T>>,
    value_bias: Option<Tensor<T>>,
    output_bias: Option<Tensor<T>>,

    // KV cache for efficient inference
    kv_cache: Arc<RwLock<Option<KVCache<T>>>>,

    // Flash attention configuration
    flash_attention_config: FlashAttentionConfig,

    // Relative position bias matrix (optional)
    relative_position_bias: Option<Tensor<T>>,

    // Performance optimization flags
    use_flash_attention: bool,
    use_kv_cache: bool,
    enable_simd_optimization: bool,
}

/// Flash Attention configuration for memory-efficient attention computation
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for tiling (usually 64, 128, or 256)
    pub block_size: usize,
    /// Enable causal masking for autoregressive models
    pub causal_mask: bool,
    /// Softmax temperature scaling
    pub temperature: f64,
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            causal_mask: false,
            temperature: 1.0,
            gradient_checkpointing: false,
        }
    }
}

impl<T> MultiHeadAttention<T>
where
    T: Float
        + FromPrimitive
        + Send
        + Sync
        + Clone
        + Default
        + From<f32>
        + std::iter::Sum
        + bytemuck::Pod
        + 'static,
{
    /// Create a new ultra-high-performance multi-head attention layer
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        enable_bias: bool,
        use_flash_attention: bool,
    ) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                embed_dim, num_heads
            )));
        }

        let head_dim = embed_dim / num_heads;
        let scale_factor = 1.0 / (head_dim as f64).sqrt();

        // Initialize weight matrices with optimal initialization
        let query_weight = Tensor::randn(&[embed_dim, embed_dim])?;
        let key_weight = Tensor::randn(&[embed_dim, embed_dim])?;
        let value_weight = Tensor::randn(&[embed_dim, embed_dim])?;
        let output_weight = Tensor::randn(&[embed_dim, embed_dim])?;

        // Optional bias vectors
        let (query_bias, key_bias, value_bias, output_bias) = if enable_bias {
            (
                Some(Tensor::zeros(&[embed_dim])),
                Some(Tensor::zeros(&[embed_dim])),
                Some(Tensor::zeros(&[embed_dim])),
                Some(Tensor::zeros(&[embed_dim])),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            scale_factor,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
            kv_cache: Arc::new(RwLock::new(None)),
            flash_attention_config: FlashAttentionConfig::default(),
            relative_position_bias: None,
            use_flash_attention,
            use_kv_cache: false,
            enable_simd_optimization: true,
        })
    }

    /// Forward pass with ultra-high-performance optimizations
    pub fn forward(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        attention_mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        if self.use_flash_attention {
            self.flash_attention_forward(query, key, value, attention_mask)
        } else {
            self.standard_attention_forward(query, key, value, attention_mask)
        }
    }

    /// Forward pass with KV cache for efficient inference
    pub fn forward_with_cache(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        cache: Option<&mut super::KVCache<T>>,
        attention_mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        // For now, just use regular forward - cache optimization to be implemented
        self.forward(query, key, value, attention_mask)
    }

    /// Ultra-optimized Flash Attention implementation
    fn flash_attention_forward(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        attention_mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let batch_size = query.shape().dims()[0];
        let seq_len = query.shape().dims()[1];

        // Project QKV with SIMD optimization
        let q = self.linear_projection(query, &self.query_weight, &self.query_bias)?;
        let k = self.linear_projection(key, &self.key_weight, &self.key_bias)?;
        let v = self.linear_projection(value, &self.value_weight, &self.value_bias)?;

        // Reshape for multi-head attention with optimal memory layout
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_len)?;
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_len)?;
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_len)?;

        // Flash attention computation with memory-efficient tiling
        let attention_output =
            self.compute_flash_attention(&q_heads, &k_heads, &v_heads, attention_mask)?;

        // Combine heads and apply output projection
        let combined = self.combine_heads(&attention_output, batch_size, seq_len)?;
        self.linear_projection(&combined, &self.output_weight, &self.output_bias)
    }

    /// Standard attention implementation with SIMD optimizations
    fn standard_attention_forward(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        attention_mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let batch_size = query.shape().dims()[0];
        let seq_len = query.shape().dims()[1];

        // Project QKV with optimized computation
        let q = self.linear_projection(query, &self.query_weight, &self.query_bias)?;
        let k = self.linear_projection(key, &self.key_weight, &self.key_bias)?;
        let v = self.linear_projection(value, &self.value_weight, &self.value_bias)?;

        // Reshape and compute attention in parallel across heads
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_len)?;
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_len)?;
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_len)?;

        // Compute scaled dot-product attention with SIMD optimization
        let attention_output = self.compute_scaled_dot_product_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            attention_mask,
        )?;

        // Combine heads and project
        let combined = self.combine_heads(&attention_output, batch_size, seq_len)?;
        self.linear_projection(&combined, &self.output_weight, &self.output_bias)
    }

    // Helper methods for ultra-high-performance computation

    fn linear_projection(
        &self,
        input: &Tensor<T>,
        weight: &Tensor<T>,
        bias: &Option<Tensor<T>>,
    ) -> Result<Tensor<T>> {
        // Optimized matrix multiplication with optional bias
        let result = input.matmul(weight)?;
        if let Some(bias) = bias {
            result.add(bias)
        } else {
            Ok(result)
        }
    }

    /// Return a row-major (C-contiguous) copy of `tensor`.
    ///
    /// `transpose_axes` only re-strides the underlying buffer, producing a
    /// non-contiguous view; `reshape` requires standard layout, so we
    /// materialise the elements in logical order before reshaping.
    fn to_contiguous(tensor: &Tensor<T>) -> Result<Tensor<T>> {
        let dims = tensor.shape().dims().to_vec();
        let values = tensor.to_vec()?;
        Tensor::from_vec(values, &dims)
    }

    fn reshape_for_heads(
        &self,
        tensor: &Tensor<T>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<T>> {
        // Split the embedding dimension into (heads, head_dim) and move the head
        // axis in front of the sequence axis so each head can be attended to
        // independently as a batched matmul.
        //
        //   [batch, seq, embed]
        //     -> reshape -> [batch, seq, heads, head_dim]
        //     -> permute (0, 2, 1, 3) -> [batch, heads, seq, head_dim]
        //
        // NOTE: a plain `transpose()` reverses *all* axes (yielding
        // [head_dim, heads, seq, batch]), which is incorrect for multi-head
        // attention. We require the specific (0, 2, 1, 3) permutation instead.
        let reshaped = tensor.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let permuted = transpose_axes(&reshaped, Some(&[0, 2, 1, 3]))?;
        Self::to_contiguous(&permuted)
    }

    fn combine_heads(
        &self,
        tensor: &Tensor<T>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<T>> {
        // Inverse of `reshape_for_heads`: move the head axis back behind the
        // sequence axis and merge it with head_dim to recover the embedding.
        //
        //   [batch, heads, seq, head_dim]
        //     -> permute (0, 2, 1, 3) -> [batch, seq, heads, head_dim]
        //     -> reshape -> [batch, seq, embed]
        let permuted = transpose_axes(tensor, Some(&[0, 2, 1, 3]))?;
        let contiguous = Self::to_contiguous(&permuted)?;
        contiguous.reshape(&[batch_size, seq_len, self.embed_dim])
    }

    fn compute_flash_attention(
        &self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        _mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        // Placeholder for Flash Attention implementation
        // In production, this would implement the memory-efficient tiled attention
        self.compute_scaled_dot_product_attention(q, k, v, _mask)
    }

    fn compute_scaled_dot_product_attention(
        &self,
        q: &Tensor<T>,
        k: &Tensor<T>,
        v: &Tensor<T>,
        _mask: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        // Compute Q @ K^T. Q and K are [batch, heads, seq, head_dim]; we only
        // transpose the last two axes of K to get [batch, heads, head_dim, seq]
        // so the batched matmul contracts over head_dim. A plain `transpose()`
        // would reverse all four axes and produce a wrong (and incompatible)
        // shape.
        let k_transposed = Self::to_contiguous(&transpose_axes(k, Some(&[0, 1, 3, 2]))?)?;
        let scores = q.matmul(&k_transposed)?;

        // Scale by sqrt(head_dim)
        let scale = T::from_f64(self.scale_factor).ok_or_else(|| {
            TensorError::invalid_argument(
                "Failed to convert attention scale factor to tensor element type".to_string(),
            )
        })?;
        let scaled_scores = scores.multiply_scalar(scale)?;

        // Apply softmax
        let attention_weights = scaled_scores.softmax(Some(-1))?;

        // Apply attention to values
        attention_weights.matmul(v)
    }
}

// Implement Layer trait for MultiHeadAttention
impl<T> Layer<T> for MultiHeadAttention<T>
where
    T: Float
        + FromPrimitive
        + Send
        + Sync
        + Clone
        + Default
        + From<f32>
        + std::iter::Sum
        + bytemuck::Pod
        + 'static,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // For self-attention, use input for query, key, and value
        self.forward(input, input, input, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![
            &self.query_weight,
            &self.key_weight,
            &self.value_weight,
            &self.output_weight,
        ];

        if let Some(ref bias) = self.query_bias {
            params.push(bias);
        }
        if let Some(ref bias) = self.key_bias {
            params.push(bias);
        }
        if let Some(ref bias) = self.value_bias {
            params.push(bias);
        }
        if let Some(ref bias) = self.output_bias {
            params.push(bias);
        }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![
            &mut self.query_weight,
            &mut self.key_weight,
            &mut self.value_weight,
            &mut self.output_weight,
        ];

        if let Some(ref mut bias) = self.query_bias {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.key_bias {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.value_bias {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.output_bias {
            params.push(bias);
        }

        params
    }

    fn set_training(&mut self, _training: bool) {
        // Training mode setting logic can be implemented here
        // For now, this is a no-op as the attention layer doesn't have training-specific behavior
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic [B, S, E] input so the test is reproducible.
    fn known_input(batch: usize, seq: usize, embed: usize) -> Tensor<f32> {
        let total = batch * seq * embed;
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 0.5).collect();
        Tensor::from_vec(data, &[batch, seq, embed])
            .expect("test: input tensor creation should succeed")
    }

    #[test]
    fn test_reshape_for_heads_round_trips_to_input_shape() {
        let (batch, seq, embed, heads) = (2usize, 3usize, 8usize, 2usize);
        let mha = MultiHeadAttention::<f32>::new(embed, heads, false, false)
            .expect("test: MHA construction should succeed");

        let input = known_input(batch, seq, embed);

        // [B, S, E] -> [B, H, S, Dh]
        let heads_view = mha
            .reshape_for_heads(&input, batch, seq)
            .expect("test: reshape_for_heads should succeed");
        assert_eq!(
            heads_view.shape().dims(),
            &[batch, heads, seq, embed / heads],
            "reshape_for_heads must permute to [B, H, S, Dh]"
        );

        // [B, H, S, Dh] -> [B, S, E] and recover the original values exactly.
        let merged = mha
            .combine_heads(&heads_view, batch, seq)
            .expect("test: combine_heads should succeed");
        assert_eq!(merged.shape().dims(), &[batch, seq, embed]);

        let original = input.to_vec().expect("test: to_vec");
        let recovered = merged.to_vec().expect("test: to_vec");
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "combine_heads must be the exact inverse of reshape_for_heads"
            );
        }
    }

    #[test]
    fn test_forward_preserves_shape_and_is_nonzero() {
        let (batch, seq, embed, heads) = (2usize, 3usize, 8usize, 2usize);
        let mha = MultiHeadAttention::<f32>::new(embed, heads, false, false)
            .expect("test: MHA construction should succeed");

        let input = known_input(batch, seq, embed);
        let output = mha
            .forward(&input, &input, &input, None)
            .expect("test: self-attention forward should succeed");

        // The bug produced a [4,2] vs [2,3] matmul mismatch; a correct
        // implementation returns the same shape as the input.
        assert_eq!(output.shape().dims(), &[batch, seq, embed]);

        // Output must be non-trivial (not all zeros).
        let values = output.to_vec().expect("test: to_vec");
        let any_nonzero = values.iter().any(|v| v.abs() > 1e-8);
        assert!(any_nonzero, "attention output should be non-zero");
    }

    #[test]
    fn test_flash_path_matches_standard_shape() {
        // The flash-attention entry point currently delegates to scaled
        // dot-product attention; it must also return [B, S, E].
        let (batch, seq, embed, heads) = (1usize, 4usize, 8usize, 4usize);
        let mha = MultiHeadAttention::<f32>::new(embed, heads, true, true)
            .expect("test: MHA construction should succeed");

        let input = known_input(batch, seq, embed);
        let output = mha
            .forward(&input, &input, &input, None)
            .expect("test: flash forward should succeed");
        assert_eq!(output.shape().dims(), &[batch, seq, embed]);
    }

    #[test]
    fn test_attention_weights_sum_to_one_per_query() {
        let (batch, seq, embed, heads) = (2usize, 3usize, 8usize, 2usize);
        let head_dim = embed / heads;
        let mha = MultiHeadAttention::<f32>::new(embed, heads, false, false)
            .expect("test: MHA construction should succeed");

        let input = known_input(batch, seq, embed);

        // Reproduce the internal score path to inspect the softmax weights:
        // scores = Q @ K^T scaled, then softmax over the key axis.
        let q = mha
            .reshape_for_heads(&input, batch, seq)
            .expect("test: reshape q");
        let k = mha
            .reshape_for_heads(&input, batch, seq)
            .expect("test: reshape k");

        let k_t =
            transpose_axes(&k, Some(&[0, 1, 3, 2])).expect("test: transpose last two axes of K");
        let scores = q.matmul(&k_t).expect("test: Q @ K^T");
        // Resulting attention logits are [B, H, S(query), S(key)].
        assert_eq!(scores.shape().dims(), &[batch, heads, seq, seq]);

        let scaled = scores
            .multiply_scalar(mha.scale_factor as f32)
            .expect("test: scale");
        let weights = scaled
            .softmax(Some(-1))
            .expect("test: softmax over key axis");

        // Each query distribution (last axis) must sum to ~1.
        let dims = weights.shape().dims().to_vec();
        for b in 0..dims[0] {
            for h in 0..dims[1] {
                for s in 0..dims[2] {
                    let mut sum = 0.0f32;
                    for k_idx in 0..dims[3] {
                        sum += weights
                            .get(&[b, h, s, k_idx])
                            .expect("test: index into attention weights");
                    }
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "attention weights for query (b={b}, h={h}, s={s}) must sum to 1, got {sum}"
                    );
                }
            }
        }

        // Sanity: head_dim partitioning is consistent with the layer config.
        assert_eq!(q.shape().dims()[3], head_dim);
    }
}
