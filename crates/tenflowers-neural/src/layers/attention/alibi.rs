//! Attention with Linear Biases (ALiBi)
//!
//! Implements the ALiBi positional encoding scheme from:
//! Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input
//! Length Extrapolation" (2021).
//!
//! Instead of adding positional embeddings to token representations, ALiBi adds a
//! head-specific linear bias to each attention score based on the distance between
//! query and key positions:
//!
//! ```text
//! score'[h, i, j] = score[h, i, j] - slope_h * |i - j|
//! ```
//!
//! Slopes follow a geometric sequence designed so the closest heads are the most
//! "positionally sensitive".

/// Precomputed ALiBi slopes for each attention head.
///
/// The slopes form a geometric sequence: for `n` heads the slopes are
/// `2^(-8/n), 2^(-16/n), …, 2^(-8)`, then doubled if `n` is not a power of two.
#[derive(Debug, Clone)]
pub struct AlibiSlopes {
    /// One slope value per attention head.
    pub slopes: Vec<f32>,
}

impl AlibiSlopes {
    /// Create `AlibiSlopes` for `num_heads` attention heads.
    ///
    /// Uses the same slope schedule as the original ALiBi paper implementation.
    pub fn new(num_heads: usize) -> Self {
        let slopes = compute_slopes(num_heads);
        Self { slopes }
    }

    /// Return the slope for a specific head index.
    ///
    /// Returns `None` if `head_idx >= num_heads`.
    pub fn slope_for_head(&self, head_idx: usize) -> Option<f32> {
        self.slopes.get(head_idx).copied()
    }
}

/// Compute the ALiBi slopes for `num_heads` attention heads.
///
/// The schedule mirrors the reference implementation:
/// - Find the largest power-of-two `n_pow2 ≤ num_heads`.
/// - Generate slopes for `n_pow2` heads: `2^(-8/n_pow2)`, `2^(-16/n_pow2)`, …, `2^(-8)`.
/// - If `num_heads > n_pow2`, fill remaining heads by halving the ratio
///   (equivalent to generating slopes for `2*n_pow2` and taking odd-indexed ones).
pub fn compute_slopes(num_heads: usize) -> Vec<f32> {
    if num_heads == 0 {
        return Vec::new();
    }

    // Largest power of two ≤ num_heads
    let n_pow2 = {
        let mut p = 1usize;
        while p * 2 <= num_heads {
            p *= 2;
        }
        p
    };

    // Base slopes for n_pow2 heads
    let base_slopes = geometric_slopes(n_pow2);

    if num_heads == n_pow2 {
        return base_slopes;
    }

    // Extra slopes for the remaining heads (use half-step series)
    let extra_needed = num_heads - n_pow2;
    let extra_slopes = geometric_slopes(2 * n_pow2);
    // Take odd-indexed slopes from the doubled series (indices 1, 3, 5, …)
    let extra: Vec<f32> = extra_slopes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, v)| v)
        .take(extra_needed)
        .collect();

    let mut all = base_slopes;
    all.extend(extra);
    all
}

/// Generate `n` slopes: `2^(-8/n), 2^(-16/n), …, 2^(-8)`.
fn geometric_slopes(n: usize) -> Vec<f32> {
    (1..=n)
        .map(|k| 2.0f32.powf(-8.0 * k as f32 / n as f32))
        .collect()
}

// ---------------------------------------------------------------------------
// AlibiMask — building and applying ALiBi bias matrices
// ---------------------------------------------------------------------------

/// Utilities for building and applying ALiBi bias matrices.
pub struct AlibiMask;

impl AlibiMask {
    /// Build a `seq_len × seq_len` bias matrix for one head.
    ///
    /// `bias[i][j] = -slope * |i - j|`
    ///
    /// The matrix is **symmetric** and has zeros on the diagonal.
    pub fn build_bias_matrix(seq_len: usize, slope: f32) -> Vec<Vec<f32>> {
        (0..seq_len)
            .map(|i| {
                (0..seq_len)
                    .map(|j| {
                        let dist = i.abs_diff(j);
                        -slope * dist as f32
                    })
                    .collect()
            })
            .collect()
    }

    /// Build a causal `seq_len × seq_len` bias matrix for one head.
    ///
    /// Lower triangle (and diagonal): `bias[i][j] = -slope * (i - j)` for `j ≤ i`.
    /// Upper triangle: `-∞` (future tokens are masked out).
    pub fn build_causal_bias(seq_len: usize, slope: f32) -> Vec<Vec<f32>> {
        let neg_inf = f32::NEG_INFINITY;
        (0..seq_len)
            .map(|i| {
                (0..seq_len)
                    .map(|j| {
                        if j > i {
                            neg_inf
                        } else {
                            -slope * (i - j) as f32
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Add ALiBi biases **in-place** to a flat attention-score buffer.
    ///
    /// The `scores` buffer is assumed to be laid out as
    /// `[head, query_pos, key_pos]` in row-major order with shape
    /// `[num_heads, seq_len, seq_len]`.
    ///
    /// # Errors
    /// Returns an error string if sizes are inconsistent.
    pub fn apply_to_attention_scores(
        scores: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        slopes: &[f32],
    ) -> Result<(), String> {
        let expected = num_heads * seq_len * seq_len;
        if scores.len() != expected {
            return Err(format!(
                "scores length mismatch: expected {} got {}",
                expected,
                scores.len()
            ));
        }
        if slopes.len() < num_heads {
            return Err(format!(
                "not enough slopes: need {} got {}",
                num_heads,
                slopes.len()
            ));
        }

        for h in 0..num_heads {
            let slope = slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dist = i.abs_diff(j);
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    scores[idx] -= slope * dist as f32;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AlibiAttention — full scaled-dot-product with ALiBi
// ---------------------------------------------------------------------------

/// Multi-head scaled-dot-product attention with ALiBi positional biases.
///
/// This computes the standard attention formula but adds the head-specific
/// ALiBi bias to the raw scores before softmax.
#[derive(Debug, Clone)]
pub struct AlibiAttention {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Precomputed ALiBi slopes.
    pub slopes: AlibiSlopes,
}

impl AlibiAttention {
    /// Create a new `AlibiAttention` module.
    pub fn new(num_heads: usize) -> Self {
        Self {
            num_heads,
            slopes: AlibiSlopes::new(num_heads),
        }
    }

    /// Compute scaled-dot-product attention with ALiBi bias.
    ///
    /// # Arguments
    /// * `q`        – flat `[seq_len, num_heads * head_dim]` query tensor.
    /// * `k`        – flat `[seq_len, num_heads * head_dim]` key tensor.
    /// * `v`        – flat `[seq_len, num_heads * head_dim]` value tensor.
    /// * `seq_len`  – sequence length.
    /// * `causal`   – if `true`, applies a causal mask (future positions masked).
    ///
    /// Returns flat `[seq_len, num_heads * head_dim]` output tensor.
    ///
    /// # Errors
    /// Returns an error string on size mismatches.
    pub fn scaled_dot_product(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        causal: bool,
    ) -> Result<Vec<f32>, String> {
        if q.is_empty() {
            return Err("q cannot be empty".to_string());
        }
        let total_head_dim = q.len() / seq_len;
        if q.len() != seq_len * total_head_dim {
            return Err(format!(
                "q length {} not divisible by seq_len {}",
                q.len(),
                seq_len
            ));
        }
        if k.len() != q.len() || v.len() != q.len() {
            return Err("q, k, v must have the same length".to_string());
        }
        if total_head_dim % self.num_heads != 0 {
            return Err(format!(
                "total_head_dim {} not divisible by num_heads {}",
                total_head_dim, self.num_heads
            ));
        }
        let head_dim = total_head_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let neg_inf = f32::NEG_INFINITY;

        // Output: [seq_len, num_heads * head_dim]
        let mut output = vec![0.0f32; seq_len * total_head_dim];

        for h in 0..self.num_heads {
            let slope = self.slopes.slopes[h];
            let h_offset = h * head_dim;

            // Compute attention scores for this head: [seq_len, seq_len]
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // dot product of q[i] and k[j] for this head
                    let q_off = i * total_head_dim + h_offset;
                    let k_off = j * total_head_dim + h_offset;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    dot *= scale;

                    // ALiBi bias
                    let dist = i.abs_diff(j);
                    dot -= slope * dist as f32;

                    // Causal mask
                    if causal && j > i {
                        dot = neg_inf;
                    }
                    scores[i * seq_len + j] = dot;
                }
            }

            // Softmax over key dimension, then weighted sum of values
            for i in 0..seq_len {
                let row = &scores[i * seq_len..(i + 1) * seq_len];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row
                    .iter()
                    .map(|&s| {
                        if s == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (s - max_val).exp()
                        }
                    })
                    .collect();
                let sum_exp: f32 = exps.iter().sum();
                let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };
                let attn_weights: Vec<f32> = exps.iter().map(|&e| e / safe_sum).collect();

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        let v_off = j * total_head_dim + h_offset;
                        acc += attn_weights[j] * v[v_off + d];
                    }
                    output[i * total_head_dim + h_offset + d] = acc;
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ------------------------------------------------------------------
    // Slope tests
    // ------------------------------------------------------------------

    #[test]
    fn test_alibi_slopes_4_heads_known_values() {
        // For 4 heads (power of two), slope_k = 2^(-8k/4) = 2^(-2k)
        // k=1: 2^(-2) = 0.25
        // k=2: 2^(-4) = 0.0625
        // k=3: 2^(-6) = 0.015625
        // k=4: 2^(-8) = 0.00390625
        let s = AlibiSlopes::new(4);
        assert!(approx_eq(s.slopes[0], 0.25), "slope[0]={}", s.slopes[0]);
        assert!(approx_eq(s.slopes[1], 0.0625), "slope[1]={}", s.slopes[1]);
        assert!(approx_eq(s.slopes[2], 0.015625), "slope[2]={}", s.slopes[2]);
        assert!(
            approx_eq(s.slopes[3], 0.00390625),
            "slope[3]={}",
            s.slopes[3]
        );
    }

    #[test]
    fn test_alibi_slopes_8_heads_known_values() {
        // For 8 heads: slope_k = 2^(-k) for k = 1..8
        let s = AlibiSlopes::new(8);
        for (k, slope) in s.slopes.iter().enumerate() {
            let expected = 2.0f32.powf(-(k as f32 + 1.0));
            assert!(
                approx_eq(*slope, expected),
                "slope[{}]={} != {}",
                k,
                slope,
                expected
            );
        }
    }

    #[test]
    fn test_alibi_slope_monotone_decreasing() {
        // Each successive head should have a strictly smaller slope
        let s = AlibiSlopes::new(4);
        for i in 1..s.slopes.len() {
            assert!(
                s.slopes[i - 1] > s.slopes[i],
                "slopes not monotone at {}: {} <= {}",
                i,
                s.slopes[i - 1],
                s.slopes[i]
            );
        }
    }

    #[test]
    fn test_alibi_slope_for_head_returns_correct() {
        let s = AlibiSlopes::new(4);
        assert!(approx_eq(s.slope_for_head(0).expect("h0"), 0.25));
    }

    #[test]
    fn test_alibi_slope_for_head_out_of_range() {
        let s = AlibiSlopes::new(4);
        assert!(s.slope_for_head(10).is_none());
    }

    #[test]
    fn test_alibi_compute_slopes_zero_heads() {
        let slopes = compute_slopes(0);
        assert!(slopes.is_empty());
    }

    // ------------------------------------------------------------------
    // Bias matrix tests
    // ------------------------------------------------------------------

    #[test]
    fn test_alibi_bias_matrix_diagonal_zero() {
        let mat = AlibiMask::build_bias_matrix(4, 0.5);
        for i in 0..4 {
            assert!(approx_eq(mat[i][i], 0.0), "diagonal[{}]={}", i, mat[i][i]);
        }
    }

    #[test]
    fn test_alibi_bias_matrix_symmetric() {
        let mat = AlibiMask::build_bias_matrix(5, 0.25);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    approx_eq(mat[i][j], mat[j][i]),
                    "symmetry fail at ({},{}) and ({},{}): {} != {}",
                    i,
                    j,
                    j,
                    i,
                    mat[i][j],
                    mat[j][i]
                );
            }
        }
    }

    #[test]
    fn test_alibi_bias_matrix_off_diagonal_negative() {
        let mat = AlibiMask::build_bias_matrix(4, 0.5);
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        mat[i][j] < 0.0,
                        "off-diagonal[{}][{}] should be negative",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_bias_matrix_values() {
        let slope = 0.25f32;
        let mat = AlibiMask::build_bias_matrix(3, slope);
        // |0-1|=1, bias=-0.25; |0-2|=2, bias=-0.5
        assert!(approx_eq(mat[0][1], -0.25), "mat[0][1]={}", mat[0][1]);
        assert!(approx_eq(mat[0][2], -0.50), "mat[0][2]={}", mat[0][2]);
        assert!(approx_eq(mat[1][2], -0.25), "mat[1][2]={}", mat[1][2]);
    }

    #[test]
    fn test_alibi_causal_bias_upper_triangle_neg_inf() {
        let mat = AlibiMask::build_causal_bias(4, 0.25);
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(
                    mat[i][j].is_infinite() && mat[i][j] < 0.0,
                    "causal upper triangle [{},{}] should be -inf",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_alibi_causal_bias_lower_triangle_correct() {
        let slope = 0.5f32;
        let mat = AlibiMask::build_causal_bias(3, slope);
        // bias[i][j] = -slope * (i - j) for j <= i
        assert!(approx_eq(mat[0][0], 0.0));
        assert!(approx_eq(mat[1][0], -0.5));
        assert!(approx_eq(mat[2][0], -1.0));
        assert!(approx_eq(mat[2][1], -0.5));
    }

    #[test]
    fn test_alibi_apply_to_attention_scores_in_place() {
        let seq_len = 2;
        let num_heads = 1;
        // scores flat: [h=0, q=0, k=0], [h=0, q=0, k=1], [h=0, q=1, k=0], [h=0, q=1, k=1]
        let mut scores = vec![1.0f32; 4]; // seq=2, head=1, 4 elements
        let slopes = vec![1.0f32];
        AlibiMask::apply_to_attention_scores(&mut scores, seq_len, num_heads, &slopes)
            .expect("apply");
        // Diagonal unchanged (dist=0), off-diagonal gets -slope * 1 = -1
        // (0,0): 1 - 0 = 1
        // (0,1): 1 - 1 = 0
        // (1,0): 1 - 1 = 0
        // (1,1): 1 - 0 = 1
        assert!(approx_eq(scores[0], 1.0), "scores[0]={}", scores[0]);
        assert!(approx_eq(scores[1], 0.0), "scores[1]={}", scores[1]);
        assert!(approx_eq(scores[2], 0.0), "scores[2]={}", scores[2]);
        assert!(approx_eq(scores[3], 1.0), "scores[3]={}", scores[3]);
    }

    #[test]
    fn test_alibi_apply_to_attention_scores_size_mismatch() {
        let mut scores = vec![0.0f32; 3]; // wrong size for (2,1,2)
        let slopes = vec![1.0f32];
        let result = AlibiMask::apply_to_attention_scores(&mut scores, 2, 1, &slopes);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // AlibiAttention forward pass tests
    // ------------------------------------------------------------------

    #[test]
    fn test_alibi_attention_output_shape() {
        let attn = AlibiAttention::new(2);
        let seq_len = 4;
        let head_dim = 8;
        let total = seq_len * 2 * head_dim;
        let q = vec![0.1f32; total];
        let k = vec![0.1f32; total];
        let v = vec![0.1f32; total];
        let out = attn
            .scaled_dot_product(&q, &k, &v, seq_len, false)
            .expect("forward");
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_alibi_attention_causal_output_shape() {
        let attn = AlibiAttention::new(2);
        let seq_len = 6;
        let head_dim = 4;
        let total = seq_len * 2 * head_dim;
        let q = vec![0.05f32; total];
        let k = vec![0.05f32; total];
        let v = vec![0.1f32; total];
        let out = attn
            .scaled_dot_product(&q, &k, &v, seq_len, true)
            .expect("causal forward");
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_alibi_attention_empty_q_rejected() {
        let attn = AlibiAttention::new(2);
        let result = attn.scaled_dot_product(&[], &[], &[], 0, false);
        assert!(result.is_err());
    }
}
