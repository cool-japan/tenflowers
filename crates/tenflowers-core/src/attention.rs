//! Flash-Attention Style Tiled Computation (CPU Reference Implementation)
//!
//! This module provides a pure-Rust, CPU-based reference implementation of multi-head
//! scaled dot-product attention, following Flash Attention's tiled approach for
//! memory efficiency. The implementation uses the online softmax (log-sum-exp) trick
//! to avoid materialising the full attention matrix.
//!
//! ## Architecture
//!
//! The attention mechanism computes:
//!   `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V`
//!
//! where `d_k` is the head dimension. Tiled computation processes blocks of the
//! sequence, accumulating results with numerically stable rescaling.
//!
//! ## Tensor Layout
//!
//! All tensors use row-major (C-contiguous) layout with shape
//! `[batch, seq_len, num_heads, head_dim]`.
//!
//! ## Numerical Stability
//!
//! - Online softmax with running max subtraction avoids overflow.
//! - Log-sum-exp accumulation keeps rescaling exact across tiles.
//! - Causal masking sets future positions to `-inf` before softmax.

use crate::error::{Result, TensorError};

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Dropout probability (applied during training only).
    pub dropout_prob: f32,
    /// Whether to apply causal masking (autoregressive models).
    pub causal: bool,
    /// Scale factor for dot products. If `None`, uses `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
}

impl AttentionConfig {
    /// Returns the effective scale factor.
    pub fn effective_scale(&self) -> f32 {
        self.scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

// ---------------------------------------------------------------------------
// ScaledDotProductAttention
// ---------------------------------------------------------------------------

/// Multi-head scaled dot-product attention.
///
/// Tensors are passed as flat `&[f32]` slices with explicit shape descriptors.
/// Shape convention: `[batch, seq_len, num_heads, head_dim]`.
pub struct ScaledDotProductAttention {
    config: AttentionConfig,
}

impl ScaledDotProductAttention {
    /// Constructs a new attention module with the given configuration.
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Compute full (non-tiled) attention.
    ///
    /// # Arguments
    /// - `q`, `q_shape`: Query tensor and its shape `[batch, seq_q, num_heads, head_dim]`.
    /// - `k`, `k_shape`: Key tensor and its shape `[batch, seq_k, num_heads, head_dim]`.
    /// - `v`, `v_shape`: Value tensor and its shape `[batch, seq_k, num_heads, head_dim]`.
    ///
    /// # Returns
    /// `(output, output_shape)` where `output_shape == q_shape`.
    pub fn forward(
        &self,
        q: &[f32],
        q_shape: &[usize],
        k: &[f32],
        k_shape: &[usize],
        v: &[f32],
        v_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        validate_attention_shapes(q, q_shape, k, k_shape, v, v_shape, "forward")?;

        let (batch, seq_q, num_heads, head_dim) = unpack_4d(q_shape);
        let seq_k = k_shape[1];
        let scale = self.config.effective_scale();
        let causal = self.config.causal;

        let out_len = batch * seq_q * num_heads * head_dim;
        let mut output = vec![0.0f32; out_len];

        for b in 0..batch {
            for h in 0..num_heads {
                // For each query position, compute attention output using online softmax
                for i in 0..seq_q {
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0f32;
                    let mut acc = vec![0.0f32; head_dim];

                    for j in 0..seq_k {
                        // Causal masking: skip future key positions
                        if causal && j > i {
                            continue;
                        }

                        let dot = dot_product(q, k, b, seq_q, i, h, seq_k, j, num_heads, head_dim);
                        let logit = dot * scale;

                        // Online softmax update with running max rescaling
                        let prev_max = running_max;
                        running_max = running_max.max(logit);
                        let correction = (prev_max - running_max).exp();
                        running_sum = running_sum * correction + (logit - running_max).exp();

                        // Rescale accumulator
                        for d in 0..head_dim {
                            acc[d] *= correction;
                        }

                        // Weighted V contribution
                        let weight = (logit - running_max).exp();
                        let v_base = v_base_index(b, j, h, num_heads, head_dim, seq_k);
                        for d in 0..head_dim {
                            acc[d] += weight * v[v_base + d];
                        }
                    }

                    // Normalise
                    let inv_sum = if running_sum > 0.0 {
                        1.0 / running_sum
                    } else {
                        0.0
                    };

                    let out_base = out_base_index(b, i, h, num_heads, head_dim, seq_q);
                    for d in 0..head_dim {
                        output[out_base + d] = acc[d] * inv_sum;
                    }
                }
            }
        }

        Ok((output, q_shape.to_vec()))
    }

    /// Tiled attention for memory efficiency.
    ///
    /// Processes the key/value sequence in blocks of `tile_size`, using the online
    /// softmax trick so the full `seq_len × seq_len` score matrix is never
    /// materialised.  Results are numerically identical to `forward`.
    ///
    /// # Arguments
    /// - `tile_size`: Number of key/value positions processed per tile. Must be ≥ 1.
    pub fn forward_tiled(
        &self,
        q: &[f32],
        q_shape: &[usize],
        k: &[f32],
        k_shape: &[usize],
        v: &[f32],
        v_shape: &[usize],
        tile_size: usize,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        if tile_size == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "forward_tiled".to_string(),
                reason: "tile_size must be >= 1".to_string(),
                context: None,
            });
        }
        validate_attention_shapes(q, q_shape, k, k_shape, v, v_shape, "forward_tiled")?;

        let (batch, seq_q, num_heads, head_dim) = unpack_4d(q_shape);
        let seq_k = k_shape[1];
        let scale = self.config.effective_scale();
        let causal = self.config.causal;

        let out_len = batch * seq_q * num_heads * head_dim;
        let mut output = vec![0.0f32; out_len];

        // Tile-local logit buffer; reallocated to `tile_size` once
        let mut logit_tile = vec![0.0f32; tile_size];
        let mut acc = vec![0.0f32; head_dim];

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_q {
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0f32;
                    for x in acc.iter_mut() {
                        *x = 0.0;
                    }

                    // Process keys in tiles
                    let mut j_start = 0;
                    while j_start < seq_k {
                        let j_end = (j_start + tile_size).min(seq_k);

                        // Compute logits for this tile and find tile max
                        let mut tile_max = f32::NEG_INFINITY;
                        for (t, j) in (j_start..j_end).enumerate() {
                            let logit = if causal && j > i {
                                f32::NEG_INFINITY
                            } else {
                                let dot = dot_product(
                                    q, k, b, seq_q, i, h, seq_k, j, num_heads, head_dim,
                                );
                                dot * scale
                            };
                            logit_tile[t] = logit;
                            if logit > tile_max {
                                tile_max = logit;
                            }
                        }

                        // Skip fully-masked tiles
                        if tile_max == f32::NEG_INFINITY {
                            j_start = j_end;
                            continue;
                        }

                        // Update running max and rescale existing state
                        let new_max = running_max.max(tile_max);
                        let correction = (running_max - new_max).exp();
                        running_sum *= correction;
                        for d in 0..head_dim {
                            acc[d] *= correction;
                        }
                        running_max = new_max;

                        // Accumulate tile contributions
                        for (t, j) in (j_start..j_end).enumerate() {
                            let logit = logit_tile[t];
                            if logit == f32::NEG_INFINITY {
                                continue;
                            }
                            let w = (logit - running_max).exp();
                            running_sum += w;
                            let v_base = v_base_index(b, j, h, num_heads, head_dim, seq_k);
                            for d in 0..head_dim {
                                acc[d] += w * v[v_base + d];
                            }
                        }

                        j_start = j_end;
                    }

                    // Normalise and write output
                    let inv_sum = if running_sum > 0.0 {
                        1.0 / running_sum
                    } else {
                        0.0
                    };
                    let out_base = out_base_index(b, i, h, num_heads, head_dim, seq_q);
                    for d in 0..head_dim {
                        output[out_base + d] = acc[d] * inv_sum;
                    }
                }
            }
        }

        Ok((output, q_shape.to_vec()))
    }
}

// ---------------------------------------------------------------------------
// Stand-alone utility functions
// ---------------------------------------------------------------------------

/// Compute the causal mask for a given sequence length.
///
/// Returns a `seq_len × seq_len` boolean matrix (row-major) where
/// `mask[i * seq_len + j] == true` means position `j` should be *masked out*
/// (i.e. `j > i`).
///
/// # Example
///
/// For `seq_len = 3`:
/// ```text
/// i\j  0      1      2
/// 0  [ false  true   true  ]
/// 1  [ false  false  true  ]
/// 2  [ false  false  false ]
/// ```
pub fn causal_mask(seq_len: usize) -> Vec<bool> {
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = true;
            }
        }
    }
    mask
}

/// Online softmax using the numerically stable log-sum-exp trick.
///
/// Computes `softmax(x)` in a single pass by tracking the running maximum
/// and the running sum.  This is equivalent to the standard two-pass
/// algorithm but avoids the second scan over the input.
///
/// If all values are `-inf`, returns a zero vector (no valid probability mass).
pub fn online_softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }

    // Single-pass: maintain running max and normalised exponentials.
    // When max increases, rescale previously stored values.
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    let mut exps = vec![0.0f32; x.len()];

    for (i, &xi) in x.iter().enumerate() {
        if xi > running_max {
            // Rescale all previously computed exp values
            let correction = (running_max - xi).exp();
            for prev in exps[..i].iter_mut() {
                *prev *= correction;
            }
            running_sum = running_sum * correction + 1.0;
            running_max = xi;
            exps[i] = 1.0;
        } else {
            let e = (xi - running_max).exp();
            running_sum += e;
            exps[i] = e;
        }
    }

    // Normalise
    if running_sum > 0.0 {
        let inv = 1.0 / running_sum;
        for e in exps.iter_mut() {
            *e *= inv;
        }
    }
    exps
}

/// Fused softmax + optional dropout in a single pass.
///
/// 1. Applies temperature scaling: each logit is divided by `temperature`.
/// 2. Computes numerically stable softmax in-place.
/// 3. When `training == true` and `dropout_prob > 0`, applies inverted dropout:
///    each element is zeroed with probability `dropout_prob` and the remaining
///    elements are scaled by `1 / (1 - dropout_prob)`.
///
/// `temperature` should be `> 0`; values ≤ 0 are clamped to `f32::EPSILON`.
///
/// # Notes
///
/// The dropout path uses a deterministic LCG to avoid a `rand` dependency.
/// This is a reference implementation; production code should use a seeded PRNG.
pub fn fused_softmax_dropout(
    logits: &mut [f32],
    temperature: f32,
    dropout_prob: f32,
    training: bool,
) {
    if logits.is_empty() {
        return;
    }

    // Temperature scaling
    if (temperature - 1.0).abs() > f32::EPSILON {
        let inv_temp = 1.0 / temperature.max(f32::EPSILON);
        for x in logits.iter_mut() {
            *x *= inv_temp;
        }
    }

    // Numerically stable softmax in-place (two-pass: find max, then exp+normalise)
    let max_val = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in logits.iter_mut() {
            *x *= inv;
        }
    }

    // Optional inverted dropout (only during training)
    if training && dropout_prob > 0.0 && dropout_prob < 1.0 {
        let keep_prob = 1.0 - dropout_prob;
        let scale = 1.0 / keep_prob;
        // Deterministic LCG seeded from fixed constant (reference implementation)
        let mut state: u64 = 0x123456789ABCDEF0;
        for x in logits.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Use upper 32 bits for better quality
            let r = ((state >> 32) as f32) / (u32::MAX as f32);
            if r < dropout_prob {
                *x = 0.0;
            } else {
                *x *= scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Private indexing helpers
// ---------------------------------------------------------------------------

/// Validate that all shapes are consistent 4-D attention tensors.
fn validate_attention_shapes(
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
    op: &str,
) -> Result<()> {
    for (name, shape) in [("q", q_shape), ("k", k_shape), ("v", v_shape)] {
        if shape.len() != 4 {
            return Err(TensorError::InvalidShape {
                operation: op.to_string(),
                reason: format!(
                    "{name} must be 4D [batch, seq, heads, head_dim], got {shape:?}"
                ),
                shape: Some(shape.to_vec()),
                context: None,
            });
        }
    }

    let (batch, _seq_q, num_heads, head_dim) = unpack_4d(q_shape);

    // K and V must share batch, num_heads, head_dim with Q
    for (name, shape) in [("k", k_shape), ("v", v_shape)] {
        if shape[0] != batch || shape[2] != num_heads || shape[3] != head_dim {
            return Err(TensorError::ShapeMismatch {
                operation: op.to_string(),
                expected: format!("[{batch}, *, {num_heads}, {head_dim}]"),
                got: format!("{shape:?} for {name}"),
                context: None,
            });
        }
    }

    // K and V must have the same seq_len
    if k_shape[1] != v_shape[1] {
        return Err(TensorError::ShapeMismatch {
            operation: op.to_string(),
            expected: format!(
                "k_seq_len == v_seq_len, got {} vs {}",
                k_shape[1], v_shape[1]
            ),
            got: "k and v seq_len differ".to_string(),
            context: None,
        });
    }

    // Validate buffer lengths
    let q_expected: usize = q_shape.iter().product();
    let k_expected: usize = k_shape.iter().product();
    let v_expected: usize = v_shape.iter().product();

    if q.len() != q_expected {
        return Err(TensorError::InvalidArgument {
            operation: op.to_string(),
            reason: format!(
                "q length {} != product of q_shape {:?} ({})",
                q.len(),
                q_shape,
                q_expected
            ),
            context: None,
        });
    }
    if k.len() != k_expected {
        return Err(TensorError::InvalidArgument {
            operation: op.to_string(),
            reason: format!(
                "k length {} != product of k_shape {:?} ({})",
                k.len(),
                k_shape,
                k_expected
            ),
            context: None,
        });
    }
    if v.len() != v_expected {
        return Err(TensorError::InvalidArgument {
            operation: op.to_string(),
            reason: format!(
                "v length {} != product of v_shape {:?} ({})",
                v.len(),
                v_shape,
                v_expected
            ),
            context: None,
        });
    }

    Ok(())
}

/// Unpack a 4-D shape slice.
#[inline]
fn unpack_4d(shape: &[usize]) -> (usize, usize, usize, usize) {
    (shape[0], shape[1], shape[2], shape[3])
}

/// Compute Q[b, i, h, :] · K[b, j, h, :].
///
/// Layout for Q: `[batch, seq_q, num_heads, head_dim]`
/// Layout for K: `[batch, seq_k, num_heads, head_dim]`
#[inline]
fn dot_product(
    q: &[f32],
    k: &[f32],
    b: usize,
    seq_q: usize,
    i: usize,
    h: usize,
    seq_k: usize,
    j: usize,
    num_heads: usize,
    head_dim: usize,
) -> f32 {
    let q_base = b * seq_q * num_heads * head_dim
        + i * num_heads * head_dim
        + h * head_dim;
    let k_base = b * seq_k * num_heads * head_dim
        + j * num_heads * head_dim
        + h * head_dim;
    let mut dot = 0.0f32;
    for d in 0..head_dim {
        dot += q[q_base + d] * k[k_base + d];
    }
    dot
}

/// Base index for V or output at `[b, j, h, 0]`.
#[inline]
fn v_base_index(
    b: usize,
    j: usize,
    h: usize,
    num_heads: usize,
    head_dim: usize,
    seq: usize,
) -> usize {
    b * seq * num_heads * head_dim + j * num_heads * head_dim + h * head_dim
}

/// Base index for the output tensor at `[b, i, h, 0]`.
#[inline]
fn out_base_index(
    b: usize,
    i: usize,
    h: usize,
    num_heads: usize,
    head_dim: usize,
    seq_q: usize,
) -> usize {
    b * seq_q * num_heads * head_dim + i * num_heads * head_dim + h * head_dim
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- online_softmax ---

    #[test]
    fn test_online_softmax_sums_to_one() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let p = online_softmax(&x);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn test_online_softmax_large_values_stable() {
        // Values that would overflow naive exp(x) computation
        let x = vec![1000.0f32, 1001.0, 1002.0];
        let p = online_softmax(&x);
        let sum: f32 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "large-value softmax must sum to 1, got {sum}"
        );
        // Largest value should have highest probability
        assert!(
            p[2] > p[1] && p[1] > p[0],
            "probabilities must be monotone increasing: {p:?}"
        );
        // No NaN or infinity
        for (i, &pi) in p.iter().enumerate() {
            assert!(pi.is_finite(), "p[{i}] must be finite, got {pi}");
        }
    }

    #[test]
    fn test_online_softmax_uniform_input() {
        let x = vec![0.0f32; 4];
        let p = online_softmax(&x);
        for &pi in &p {
            assert!((pi - 0.25).abs() < 1e-6, "uniform input must give 0.25 each, got {pi}");
        }
    }

    #[test]
    fn test_online_softmax_empty() {
        let p = online_softmax(&[]);
        assert!(p.is_empty());
    }

    #[test]
    fn test_online_softmax_single_element() {
        let p = online_softmax(&[42.0f32]);
        assert_eq!(p.len(), 1);
        assert!((p[0] - 1.0).abs() < 1e-7, "single element must be 1.0, got {}", p[0]);
    }

    // --- causal_mask ---

    #[test]
    fn test_causal_mask_shape() {
        let mask = causal_mask(5);
        assert_eq!(mask.len(), 25, "5x5 mask must have 25 elements");
    }

    #[test]
    fn test_causal_mask_pattern() {
        let n = 4;
        let mask = causal_mask(n);
        for i in 0..n {
            for j in 0..n {
                let expected = j > i;
                assert_eq!(
                    mask[i * n + j],
                    expected,
                    "mask[{i},{j}] should be {expected}"
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_diagonal_not_masked() {
        let n = 6;
        let mask = causal_mask(n);
        for i in 0..n {
            assert!(
                !mask[i * n + i],
                "diagonal position ({i},{i}) must not be masked"
            );
        }
    }

    // --- fused_softmax_dropout ---

    #[test]
    fn test_fused_softmax_dropout_inference_no_dropout() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        fused_softmax_dropout(&mut logits, 1.0, 0.5, false /* inference */);
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "inference softmax must sum to 1, got {sum}"
        );
        // No dropout during inference
        for &x in &logits {
            assert!(x > 0.0, "no element should be zeroed during inference");
        }
    }

    #[test]
    fn test_fused_softmax_dropout_temperature() {
        let mut logits_t1 = vec![1.0f32, 2.0, 3.0];
        let mut logits_t2 = vec![1.0f32, 2.0, 3.0];
        fused_softmax_dropout(&mut logits_t1, 1.0, 0.0, false);
        fused_softmax_dropout(&mut logits_t2, 2.0, 0.0, false);
        // Higher temperature → smoother distribution (peak probability should be smaller)
        assert!(
            logits_t2[2] < logits_t1[2],
            "higher temperature should reduce peak prob: t1={}, t2={}",
            logits_t1[2],
            logits_t2[2]
        );
    }

    #[test]
    fn test_fused_softmax_dropout_sums_to_one() {
        let mut logits = vec![0.5f32, -0.5, 1.0, -1.0];
        fused_softmax_dropout(&mut logits, 1.0, 0.0, false);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    // --- ScaledDotProductAttention ---

    fn make_config(num_heads: usize, head_dim: usize, causal: bool) -> AttentionConfig {
        AttentionConfig {
            num_heads,
            head_dim,
            dropout_prob: 0.0,
            causal,
            scale: None,
        }
    }

    fn make_qkv(
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
        let n = batch * seq * num_heads * head_dim;
        let shape = vec![batch, seq, num_heads, head_dim];
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        (q, k, v, shape)
    }

    #[test]
    fn test_forward_output_shape() {
        let (q, k, v, shape) = make_qkv(2, 4, 2, 4);
        let attn = ScaledDotProductAttention::new(make_config(2, 4, false));
        let (out, out_shape) = attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("forward should succeed");
        assert_eq!(out_shape, shape, "output shape must match q_shape");
        assert_eq!(out.len(), q.len(), "output length must match input length");
    }

    #[test]
    fn test_forward_no_nan() {
        let (q, k, v, shape) = make_qkv(2, 4, 2, 4);
        let attn = ScaledDotProductAttention::new(make_config(2, 4, false));
        let (out, _) = attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("forward should succeed");
        for (i, &x) in out.iter().enumerate() {
            assert!(x.is_finite(), "output[{i}] is not finite: {x}");
        }
    }

    #[test]
    fn test_tiled_matches_non_tiled() {
        let (q, k, v, shape) = make_qkv(2, 4, 2, 4);
        let attn = ScaledDotProductAttention::new(make_config(2, 4, false));
        let (out_full, _) = attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("forward should succeed");
        let (out_tiled, _) = attn
            .forward_tiled(&q, &shape, &k, &shape, &v, &shape, 2)
            .expect("forward_tiled should succeed");
        assert_eq!(out_full.len(), out_tiled.len());
        for (i, (&a, &b)) in out_full.iter().zip(out_tiled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "tiled vs non-tiled differ at index {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_tiled_tile_size_one() {
        // Extreme: tile_size = 1 (one key per tile)
        let (q, k, v, shape) = make_qkv(1, 3, 1, 4);
        let attn = ScaledDotProductAttention::new(make_config(1, 4, false));
        let (out_full, _) = attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("forward should succeed");
        let (out_tiled, _) = attn
            .forward_tiled(&q, &shape, &k, &shape, &v, &shape, 1)
            .expect("forward_tiled should succeed");
        for (i, (&a, &b)) in out_full.iter().zip(out_tiled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "tile_size=1 differs at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_causal_attention_masks_future_positions() {
        // Batch=1, seq=3, heads=1, head_dim=2
        let batch = 1;
        let seq = 3;
        let nh = 1;
        let hd = 2;
        let shape = vec![batch, seq, nh, hd];
        // Q and K are all zeros → uniform attention over allowed positions
        let q = vec![0.0f32; batch * seq * nh * hd];
        let k = vec![0.0f32; batch * seq * nh * hd];
        // V: pos0 → [1, 0], pos1 → [0, 1], pos2 → [2, 2]
        let v = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0];

        let causal_attn = ScaledDotProductAttention::new(make_config(nh, hd, true));
        let (out, _) = causal_attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("causal forward should succeed");

        // Position 0 can only attend to position 0 → output ≈ V[0] = [1, 0]
        // out layout: [b=0, i, h=0, d] → index = i * nh * hd + d = i*2 + d
        assert!((out[0] - 1.0).abs() < 1e-5, "pos0_d0 expected 1.0, got {}", out[0]);
        assert!((out[1] - 0.0).abs() < 1e-5, "pos0_d1 expected 0.0, got {}", out[1]);

        // Position 1 can attend to pos 0 and 1 uniformly: mean([1,0],[0,1]) = [0.5, 0.5]
        assert!((out[2] - 0.5).abs() < 1e-5, "pos1_d0 expected 0.5, got {}", out[2]);
        assert!((out[3] - 0.5).abs() < 1e-5, "pos1_d1 expected 0.5, got {}", out[3]);
    }

    #[test]
    fn test_forward_tiled_invalid_tile_size_zero() {
        let (q, k, v, shape) = make_qkv(1, 2, 1, 2);
        let attn = ScaledDotProductAttention::new(make_config(1, 2, false));
        let result = attn.forward_tiled(&q, &shape, &k, &shape, &v, &shape, 0);
        assert!(result.is_err(), "tile_size=0 must return an error");
    }

    #[test]
    fn test_tiled_matches_non_tiled_causal() {
        let (q, k, v, shape) = make_qkv(2, 5, 2, 4);
        let attn = ScaledDotProductAttention::new(make_config(2, 4, true));
        let (out_full, _) = attn
            .forward(&q, &shape, &k, &shape, &v, &shape)
            .expect("causal forward should succeed");
        let (out_tiled, _) = attn
            .forward_tiled(&q, &shape, &k, &shape, &v, &shape, 3)
            .expect("causal forward_tiled should succeed");
        for (i, (&a, &b)) in out_full.iter().zip(out_tiled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "causal tiled vs non-tiled differ at {i}: {a} vs {b}"
            );
        }
    }
}
