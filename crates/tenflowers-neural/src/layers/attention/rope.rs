//! Rotary Position Embeddings (RoPE)
//!
//! Implements the RoPE positional encoding scheme from:
//! Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
//!
//! The core idea is to rotate pairs of query/key dimensions by an angle proportional
//! to the position, encoding relative position information directly into the attention
//! dot-product without adding positional vectors.
//!
//! # Mathematical formulation
//!
//! For a head vector `x` of dimension `d`, the rotated vector at position `pos` is:
//!
//! ```text
//! x_rotated[2i]   = x[2i]   * cos(θ_i * pos) - x[2i+1] * sin(θ_i * pos)
//! x_rotated[2i+1] = x[2i]   * sin(θ_i * pos) + x[2i+1] * cos(θ_i * pos)
//! ```
//!
//! where `θ_i = 1 / base^(2i/d)`, with `base = 10000` by default.

/// Configuration for Rotary Position Embeddings.
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Dimensionality of a single attention head (must be even).
    pub head_dim: usize,
    /// Maximum sequence length for which tables are precomputed.
    pub max_seq_len: usize,
    /// Base frequency. Typically 10000; larger values extend effective context.
    pub base: f32,
}

impl RopeConfig {
    /// Create a new `RopeConfig` with the given parameters.
    ///
    /// # Arguments
    /// * `head_dim`    – Attention head dimension (must be even).
    /// * `max_seq_len` – Maximum precomputed sequence length.
    /// * `base`        – RoPE base frequency (default 10000).
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self {
            head_dim,
            max_seq_len,
            base,
        }
    }

    /// Create a default config with `base = 10000`.
    pub fn default_base(head_dim: usize, max_seq_len: usize) -> Self {
        Self::new(head_dim, max_seq_len, 10000.0)
    }
}

/// Precomputed cosine/sine tables for Rotary Position Embeddings.
///
/// `cos_table[pos][i]` = cos(θ_i · pos)
/// `sin_table[pos][i]` = sin(θ_i · pos)
/// where `i ∈ [0, head_dim/2)` and `pos ∈ [0, max_seq_len)`.
#[derive(Debug, Clone)]
pub struct RopeEmbedding {
    /// Precomputed cosine values. Shape: `[max_seq_len, head_dim/2]`.
    pub cos_table: Vec<Vec<f32>>,
    /// Precomputed sine values. Shape: `[max_seq_len, head_dim/2]`.
    pub sin_table: Vec<Vec<f32>>,
    /// Configuration used to build the tables.
    pub config: RopeConfig,
}

impl RopeEmbedding {
    /// Create a new `RopeEmbedding`, precomputing all cos/sin tables.
    ///
    /// # Panics-safety
    /// Returns an error string if `head_dim` is odd.
    ///
    /// # Arguments
    /// * `config` – RoPE configuration.
    pub fn new(config: RopeConfig) -> Result<Self, String> {
        if config.head_dim % 2 != 0 {
            return Err(format!("head_dim must be even, got {}", config.head_dim));
        }
        let half = config.head_dim / 2;
        let mut cos_table = Vec::with_capacity(config.max_seq_len);
        let mut sin_table = Vec::with_capacity(config.max_seq_len);

        for pos in 0..config.max_seq_len {
            let mut cos_row = Vec::with_capacity(half);
            let mut sin_row = Vec::with_capacity(half);
            for i in 0..half {
                let theta = Self::theta_i(i, config.head_dim, config.base);
                let angle = theta * pos as f32;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }
            cos_table.push(cos_row);
            sin_table.push(sin_row);
        }

        Ok(Self {
            cos_table,
            sin_table,
            config,
        })
    }

    /// Compute the frequency θ_i = 1 / base^(2i/d).
    fn theta_i(i: usize, head_dim: usize, base: f32) -> f32 {
        let exponent = (2 * i) as f32 / head_dim as f32;
        1.0 / base.powf(exponent)
    }

    /// Apply RoPE rotation to a single head vector at the given position.
    ///
    /// `x` must have length equal to `config.head_dim`.
    /// Returns a new `Vec<f32>` of the same length with the rotation applied.
    ///
    /// # Errors
    /// Returns an error string if `x.len() != head_dim` or `position >= max_seq_len`.
    pub fn apply_to_head(&self, x: &[f32], position: usize) -> Result<Vec<f32>, String> {
        if x.len() != self.config.head_dim {
            return Err(format!(
                "Expected head vector of length {}, got {}",
                self.config.head_dim,
                x.len()
            ));
        }
        if position >= self.config.max_seq_len {
            return Err(format!(
                "Position {} exceeds max_seq_len {}",
                position, self.config.max_seq_len
            ));
        }
        let half = self.config.head_dim / 2;
        let cos_row = &self.cos_table[position];
        let sin_row = &self.sin_table[position];
        let mut out = vec![0.0f32; self.config.head_dim];

        for i in 0..half {
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            let c = cos_row[i];
            let s = sin_row[i];
            out[2 * i] = x0 * c - x1 * s;
            out[2 * i + 1] = x0 * s + x1 * c;
        }
        Ok(out)
    }

    /// Apply RoPE to an entire batch of query and key tensors.
    ///
    /// `q` and `k` are flat arrays of shape `[seq_len * num_heads * head_dim]`
    /// laid out as `[seq_pos][head][dim]`.
    /// `positions` has length `seq_len` and gives the absolute position of each token.
    ///
    /// Returns `(q_rotated, k_rotated)` with the same shape.
    ///
    /// # Errors
    /// Returns an error string on size mismatches or out-of-range positions.
    pub fn apply_to_batch(
        &self,
        q: &[f32],
        k: &[f32],
        positions: &[usize],
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let seq_len = positions.len();
        let expected = seq_len * num_heads * head_dim;
        if q.len() != expected {
            return Err(format!(
                "q length mismatch: expected {} got {}",
                expected,
                q.len()
            ));
        }
        if k.len() != expected {
            return Err(format!(
                "k length mismatch: expected {} got {}",
                expected,
                k.len()
            ));
        }
        if head_dim != self.config.head_dim {
            return Err(format!(
                "head_dim mismatch: config={} provided={}",
                self.config.head_dim, head_dim
            ));
        }

        let mut q_out = vec![0.0f32; expected];
        let mut k_out = vec![0.0f32; expected];

        for (t, &pos) in positions.iter().enumerate() {
            for h in 0..num_heads {
                let offset = (t * num_heads + h) * head_dim;
                let q_head = &q[offset..offset + head_dim];
                let k_head = &k[offset..offset + head_dim];

                let q_rotated = self.apply_to_head(q_head, pos)?;
                let k_rotated = self.apply_to_head(k_head, pos)?;

                q_out[offset..offset + head_dim].copy_from_slice(&q_rotated);
                k_out[offset..offset + head_dim].copy_from_slice(&k_rotated);
            }
        }
        Ok((q_out, k_out))
    }

    /// Compute the theoretical relative position bias between positions `i` and `j`.
    ///
    /// This is the dot product of the RoPE-rotated unit vectors at positions `i` and `j`
    /// summed over all frequency components, demonstrating the relative-position property.
    /// Concretely, returns `sum_k cos(θ_k * (i - j))`.
    ///
    /// # Errors
    /// Returns an error string if either position exceeds `max_seq_len`.
    pub fn relative_position_bias(&self, pos_i: usize, pos_j: usize) -> Result<f32, String> {
        if pos_i >= self.config.max_seq_len {
            return Err(format!(
                "pos_i {} >= max_seq_len {}",
                pos_i, self.config.max_seq_len
            ));
        }
        if pos_j >= self.config.max_seq_len {
            return Err(format!(
                "pos_j {} >= max_seq_len {}",
                pos_j, self.config.max_seq_len
            ));
        }
        // For unit vectors e_1 = (1,0,1,0,...) (only cos components)
        // rotated by position i and j respectively, their dot product equals
        // sum_k [ cos(θ_k i) cos(θ_k j) + sin(θ_k i) sin(θ_k j) ]
        //      = sum_k cos(θ_k (i - j))
        let half = self.config.head_dim / 2;
        let mut bias = 0.0f32;
        for k in 0..half {
            let c_i = self.cos_table[pos_i][k];
            let s_i = self.sin_table[pos_i][k];
            let c_j = self.cos_table[pos_j][k];
            let s_j = self.sin_table[pos_j][k];
            bias += c_i * c_j + s_i * s_j;
        }
        Ok(bias)
    }
}

/// Extension methods for context-length interpolation via RoPE linear scaling.
///
/// This technique (Chen et al., 2023 "Extending Context Window of Large Language Models
/// via Positional Interpolation") allows a model trained with `max_seq_len = N` to be
/// used at longer sequences by scaling the effective position by `N / new_len`.
pub struct RotaryInterpolation;

impl RotaryInterpolation {
    /// Compute the linear scaling factor for context extension.
    ///
    /// When generating at `new_seq_len > original_seq_len`, multiply every position
    /// by `1 / linear_scaling_factor(new_seq_len, original_seq_len)` before computing
    /// the RoPE angle to keep effective positions within the trained range.
    ///
    /// Returns `new_seq_len / original_seq_len`.
    pub fn linear_scaling_factor(new_seq_len: usize, original_seq_len: usize) -> f32 {
        new_seq_len as f32 / original_seq_len as f32
    }

    /// Apply RoPE with a position scaling factor (for context-length interpolation).
    ///
    /// The effective position used for angle computation is `position / scale_factor`,
    /// so larger `scale_factor` compresses the positional range.
    ///
    /// `rope`         – a prebuilt `RopeEmbedding` (its tables are not used directly
    ///                   because we recompute with the scaled position).
    /// `x`            – flat head vector of length `rope.config.head_dim`.
    /// `position`     – absolute token position.
    /// `scale_factor` – linear scale (≥ 1.0 for interpolation).
    ///
    /// # Errors
    /// Returns an error string if `scale_factor <= 0.0` or head size mismatches.
    pub fn apply_with_scaling(
        rope: &RopeEmbedding,
        x: &[f32],
        position: usize,
        scale_factor: f32,
    ) -> Result<Vec<f32>, String> {
        if scale_factor <= 0.0 {
            return Err(format!(
                "scale_factor must be positive, got {}",
                scale_factor
            ));
        }
        if x.len() != rope.config.head_dim {
            return Err(format!(
                "Expected head vector of length {}, got {}",
                rope.config.head_dim,
                x.len()
            ));
        }

        let scaled_pos = position as f32 / scale_factor;
        let half = rope.config.head_dim / 2;
        let mut out = vec![0.0f32; rope.config.head_dim];

        for i in 0..half {
            let theta = RopeEmbedding::theta_i(i, rope.config.head_dim, rope.config.base);
            let angle = theta * scaled_pos;
            let c = angle.cos();
            let s = angle.sin();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            out[2 * i] = x0 * c - x1 * s;
            out[2 * i + 1] = x0 * s + x1 * c;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn rope_2d() -> RopeEmbedding {
        // head_dim=2, max_seq_len=16, base=10000
        RopeEmbedding::new(RopeConfig::new(2, 16, 10000.0)).expect("valid config")
    }

    // ------------------------------------------------------------------
    // RoPE rotation tests
    // ------------------------------------------------------------------

    #[test]
    fn test_rope_position_zero_is_identity() {
        let rope = rope_2d();
        let x = vec![3.0f32, 7.0f32];
        let y = rope.apply_to_head(&x, 0).expect("apply pos=0");
        // cos(0)=1, sin(0)=0 → identity
        assert!(
            approx_eq(y[0], x[0]),
            "pos=0 should be identity: y[0]={}",
            y[0]
        );
        assert!(
            approx_eq(y[1], x[1]),
            "pos=0 should be identity: y[1]={}",
            y[1]
        );
    }

    #[test]
    fn test_rope_rotation_pi_half_on_2d_vector() {
        // For head_dim=2, position=1, the angle is θ_0 * pos = 1 * 1 = 1 (radian)
        // (θ_0 = 1/base^(2*0/2) = 1/base^0 = 1 for any base)
        // To get a rotation of exactly π/2 at position=1, we need angle = π/2
        // but since θ_0 = 1 always, we use position floor(π/2) isn't clean.
        // Instead, verify with base=10000 at pos=1 (angle=1 rad):
        // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
        // For x=(1,0): out = (cos(1), sin(1))
        let rope = RopeEmbedding::new(RopeConfig::new(2, 16, 10000.0)).expect("valid config");
        let x = vec![1.0f32, 0.0f32]; // unit vector along x-axis
        let y = rope.apply_to_head(&x, 1).expect("apply pos=1");
        let expected_cos = 1.0f32.cos(); // cos(θ_0 * 1) = cos(1)
        let expected_sin = 1.0f32.sin(); // sin(θ_0 * 1) = sin(1)
        assert!(
            approx_eq(y[0], expected_cos),
            "expected y[0]={}, got {}",
            expected_cos,
            y[0]
        );
        assert!(
            approx_eq(y[1], expected_sin),
            "expected y[1]={}, got {}",
            expected_sin,
            y[1]
        );
    }

    #[test]
    fn test_rope_rotation_is_not_idempotent() {
        // Applying to the same position twice gives a different result than once
        let rope = rope_2d();
        let x = vec![1.0f32, 0.5f32];
        let y1 = rope.apply_to_head(&x, 1).expect("first apply");
        let y2 = rope.apply_to_head(&y1, 1).expect("second apply");
        // y2 should differ from y1 (rotation by 2*angle ≠ rotation by angle)
        let same = approx_eq(y1[0], y2[0]) && approx_eq(y1[1], y2[1]);
        assert!(!same, "double rotation should differ from single rotation");
    }

    #[test]
    fn test_rope_batch_shape_preserved() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let seq_len = 5;
        let num_heads = 2;
        let head_dim = 4;
        let total = seq_len * num_heads * head_dim;
        let q: Vec<f32> = (0..total).map(|i| i as f32 * 0.01).collect();
        let k: Vec<f32> = (0..total).map(|i| i as f32 * 0.02).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let (q_out, k_out) = rope
            .apply_to_batch(&q, &k, &positions, num_heads, head_dim)
            .expect("batch apply");
        assert_eq!(q_out.len(), total, "q_out shape mismatch");
        assert_eq!(k_out.len(), total, "k_out shape mismatch");
    }

    #[test]
    fn test_rope_batch_position_zero_is_identity_per_head() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let q: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 1 token, 1 head, head_dim=4
        let k: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let (q_out, k_out) = rope.apply_to_batch(&q, &k, &[0], 1, 4).expect("batch");
        for (a, b) in q.iter().zip(q_out.iter()) {
            assert!(approx_eq(*a, *b), "q identity fail: {}!={}", a, b);
        }
        for (a, b) in k.iter().zip(k_out.iter()) {
            assert!(approx_eq(*a, *b), "k identity fail: {}!={}", a, b);
        }
    }

    #[test]
    fn test_rope_head_dim_odd_rejected() {
        let result = RopeEmbedding::new(RopeConfig::new(3, 16, 10000.0));
        assert!(result.is_err(), "odd head_dim should be rejected");
    }

    #[test]
    fn test_rope_out_of_range_position_rejected() {
        let rope = rope_2d(); // max_seq_len=16
        let x = vec![1.0f32, 0.0f32];
        let result = rope.apply_to_head(&x, 20);
        assert!(result.is_err(), "position >= max_seq_len should error");
    }

    #[test]
    fn test_rope_relative_position_bias_same_pos_is_half_dim() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        // bias(p, p) = sum_k cos(0) = head_dim/2
        let bias = rope.relative_position_bias(3, 3).expect("bias");
        assert!(approx_eq(bias, 2.0), "expected half_dim=2, got {}", bias);
    }

    #[test]
    fn test_rope_relative_position_bias_symmetry() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let b_ij = rope.relative_position_bias(2, 5).expect("bias ij");
        let b_ji = rope.relative_position_bias(5, 2).expect("bias ji");
        // cos(θ(i-j)) = cos(θ(j-i)), so they should be equal
        assert!(
            approx_eq(b_ij, b_ji),
            "bias should be symmetric: {} != {}",
            b_ij,
            b_ji
        );
    }

    #[test]
    fn test_rope_linear_scaling_factor() {
        let factor = RotaryInterpolation::linear_scaling_factor(4096, 2048);
        assert!(approx_eq(factor, 2.0), "expected 2.0, got {}", factor);
    }

    #[test]
    fn test_rope_apply_with_scaling_pos_zero_is_identity() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = RotaryInterpolation::apply_with_scaling(&rope, &x, 0, 2.0).expect("scale");
        for (a, b) in x.iter().zip(y.iter()) {
            assert!(approx_eq(*a, *b), "scaled pos=0 should be identity");
        }
    }

    #[test]
    fn test_rope_apply_with_scaling_differs_from_unscaled() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y_unscaled = rope.apply_to_head(&x, 4).expect("unscaled");
        let y_scaled = RotaryInterpolation::apply_with_scaling(&rope, &x, 4, 2.0).expect("scaled");
        // scale=2 → effective pos=2, should differ from pos=4
        let same = y_unscaled
            .iter()
            .zip(y_scaled.iter())
            .all(|(a, b)| approx_eq(*a, *b));
        assert!(!same, "scaled and unscaled should differ");
    }

    #[test]
    fn test_rope_apply_with_scaling_invalid_scale_rejected() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = RotaryInterpolation::apply_with_scaling(&rope, &x, 1, 0.0);
        assert!(result.is_err(), "scale_factor=0 should error");
    }

    #[test]
    fn test_rope_apply_to_batch_wrong_q_length_rejected() {
        let rope = RopeEmbedding::new(RopeConfig::new(4, 16, 10000.0)).expect("valid config");
        // q is too short
        let q = vec![0.0f32; 3];
        let k = vec![0.0f32; 4];
        let result = rope.apply_to_batch(&q, &k, &[0], 1, 4);
        assert!(result.is_err(), "wrong q length should error");
    }

    #[test]
    fn test_rope_config_default_base() {
        let cfg = RopeConfig::default_base(8, 128);
        assert_eq!(cfg.base, 10000.0);
        assert_eq!(cfg.head_dim, 8);
        assert_eq!(cfg.max_seq_len, 128);
    }
}
