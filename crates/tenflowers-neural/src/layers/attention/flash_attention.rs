//! Flash Attention — memory-efficient tiled attention
//!
//! Implements the Flash Attention algorithm from:
//! Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with
//! IO-Awareness" (NeurIPS 2022).
//!
//! The key insight is to tile the K/V sequence into blocks and maintain a running
//! online-softmax denominator, so we never need to materialise the full O(N²)
//! attention matrix. The output is numerically equivalent to standard attention.
//!
//! # Algorithm outline
//! 1. Divide Q into blocks of size `B_r` (rows), K/V into blocks of size `B_c` (cols).
//! 2. For each Q-block `Q_i`:
//!    a. Initialise running max `m_i = -∞`, running sum `l_i = 0`, output `O_i = 0`.
//!    b. For each K/V-block `K_j, V_j`:
//!       - Compute local scores `S = Q_i K_j^T / sqrt(d)`.
//!       - Apply causal mask if needed.
//!       - Compute local max `m_ij = max(S)`.
//!       - Update running max: `m_new = max(m_i, m_ij)`.
//!       - Correct previous accumulation: `O_i *= exp(m_i - m_new)`, `l_i *= exp(m_i - m_new)`.
//!       - Accumulate: `O_i += exp(S - m_new) * V_j`.
//!       - `l_i += sum(exp(S - m_new))`.
//!       - `m_i = m_new`.
//!
//!    c. Normalise: `O_i /= l_i`.

/// Configuration for the Flash Attention kernel.
#[derive(Debug, Clone)]
pub struct FlashConfig {
    /// Number of tokens per tile (both Q-block and KV-block sizes).
    pub block_size: usize,
    /// Whether to apply a causal (auto-regressive) mask.
    pub causal: bool,
    /// Scale applied to raw dot-products before softmax (typically `1/sqrt(head_dim)`).
    pub scale: f32,
}

impl FlashConfig {
    /// Create a `FlashConfig` with explicit scale.
    pub fn new(block_size: usize, causal: bool, scale: f32) -> Self {
        Self {
            block_size,
            causal,
            scale,
        }
    }

    /// Create a `FlashConfig` with the standard scale `1 / sqrt(head_dim)`.
    pub fn default_scale(block_size: usize, causal: bool, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            block_size,
            causal,
            scale,
        }
    }
}

// ---------------------------------------------------------------------------
// OnlineSoftmax helper
// ---------------------------------------------------------------------------

/// Incremental online softmax state for a single output row.
///
/// Maintains the running maximum and running log-sum-exp denominator
/// so that new blocks of logits can be incorporated without re-reading
/// previously processed blocks.
#[derive(Debug, Clone)]
pub struct OnlineSoftmax {
    /// Current running maximum over all processed logits.
    pub running_max: f32,
    /// Current running (unnormalised) sum of `exp(logit - running_max)`.
    pub running_sum: f32,
}

impl OnlineSoftmax {
    /// Create a new `OnlineSoftmax` with uninitialised state (`-∞` max, `0` sum).
    pub fn new() -> Self {
        Self {
            running_max: f32::NEG_INFINITY,
            running_sum: 0.0,
        }
    }

    /// Incorporate a block of new logits into the running state.
    ///
    /// Returns `(new_max, correction_factor)` where `correction_factor` is
    /// `exp(old_max - new_max)`, which the caller must use to rescale any
    /// accumulator built with the previous maximum.
    ///
    /// # Arguments
    /// * `new_logits` – raw (unscaled-by-exp) attention scores for the current KV block.
    pub fn update(&mut self, new_logits: &[f32]) -> (f32, f32) {
        let block_max = new_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let new_max = self.running_max.max(block_max);
        let correction = (self.running_max - new_max).exp();

        // Update running sum: rescale old part, add new part
        let new_block_sum: f32 = new_logits
            .iter()
            .map(|&l| {
                if l == f32::NEG_INFINITY {
                    0.0
                } else {
                    (l - new_max).exp()
                }
            })
            .sum();
        self.running_sum = self.running_sum * correction + new_block_sum;
        self.running_max = new_max;

        (new_max, correction)
    }

    /// Normalise a weighted accumulator by the current running sum.
    ///
    /// `running_output` is a flat slice of length `head_dim`.
    /// Returns a normalised copy.
    pub fn finalize(&self, running_output: &[f32]) -> Vec<f32> {
        let safe_sum = if self.running_sum == 0.0 {
            1.0
        } else {
            self.running_sum
        };
        running_output.iter().map(|&v| v / safe_sum).collect()
    }
}

impl Default for OnlineSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Flash Attention core
// ---------------------------------------------------------------------------

/// Memory-efficient attention using the tiled Flash Attention algorithm.
///
/// Produces numerically equivalent output to naive scaled-dot-product attention
/// while using O(N · block_size) memory instead of O(N²).
#[derive(Debug, Clone)]
pub struct FlashAttention {
    /// Tile / block size for both Q and KV streams.
    pub config: FlashConfig,
}

impl FlashAttention {
    /// Create a new `FlashAttention` module.
    pub fn new(config: FlashConfig) -> Self {
        Self { config }
    }

    /// Determine whether a (query, key) pair is valid under the causal mask.
    ///
    /// Returns `true` if the pair is allowed (no masking required), i.e. `q_idx >= kv_idx`
    /// (a query token may attend to the current or any earlier key token).
    ///
    /// Arguments carry absolute sequence indices (not block-local indices).
    pub fn causal_mask_check(q_idx: usize, kv_idx: usize) -> bool {
        q_idx >= kv_idx
    }

    /// Run Flash Attention for a single head.
    ///
    /// # Arguments
    /// * `q`        – flat `[seq_len, head_dim]` query matrix.
    /// * `k`        – flat `[seq_len, head_dim]` key matrix.
    /// * `v`        – flat `[seq_len, head_dim]` value matrix.
    /// * `seq_len`  – number of tokens.
    /// * `head_dim` – dimension of each head.
    ///
    /// Returns flat `[seq_len, head_dim]` output.
    ///
    /// # Errors
    /// Returns an error string on size mismatches or zero block_size.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if self.config.block_size == 0 {
            return Err("block_size must be non-zero".to_string());
        }
        let expected = seq_len * head_dim;
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
        if v.len() != expected {
            return Err(format!(
                "v length mismatch: expected {} got {}",
                expected,
                v.len()
            ));
        }

        let scale = self.config.scale;
        let block_size = self.config.block_size;
        let causal = self.config.causal;

        let mut output = vec![0.0f32; seq_len * head_dim];

        // Iterate over Q blocks
        let q_blocks = div_ceil(seq_len, block_size);
        for q_block in 0..q_blocks {
            let q_start = q_block * block_size;
            let q_end = (q_start + block_size).min(seq_len);
            let q_block_len = q_end - q_start;

            // Running state per query in this block
            let mut online: Vec<OnlineSoftmax> =
                (0..q_block_len).map(|_| OnlineSoftmax::new()).collect();
            // Accumulated weighted-value outputs: [q_block_len, head_dim]
            let mut acc = vec![0.0f32; q_block_len * head_dim];

            // Iterate over KV blocks
            let kv_blocks = div_ceil(seq_len, block_size);
            for kv_block in 0..kv_blocks {
                let kv_start = kv_block * block_size;
                let kv_end = (kv_start + block_size).min(seq_len);
                let kv_block_len = kv_end - kv_start;

                // For causal: skip this KV block entirely if its first key is
                // strictly after the last query in this Q block.
                if causal && kv_start > q_end - 1 {
                    break;
                }

                // Compute local score block: [q_block_len, kv_block_len]
                let mut local_scores = vec![f32::NEG_INFINITY; q_block_len * kv_block_len];
                for qi in 0..q_block_len {
                    let q_idx = q_start + qi;
                    let q_off = q_idx * head_dim;
                    for ki in 0..kv_block_len {
                        let kv_idx = kv_start + ki;
                        // Causal check: query must be >= key position
                        if causal && !Self::causal_mask_check(q_idx, kv_idx) {
                            // Leave as -inf (masked)
                            continue;
                        }
                        let k_off = kv_idx * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        local_scores[qi * kv_block_len + ki] = dot * scale;
                    }
                }

                // Update online softmax and accumulate weighted values
                for qi in 0..q_block_len {
                    let row_scores = &local_scores[qi * kv_block_len..(qi + 1) * kv_block_len];
                    let (_, correction) = online[qi].update(row_scores);

                    // Rescale previous accumulation
                    let acc_off = qi * head_dim;
                    for d in 0..head_dim {
                        acc[acc_off + d] *= correction;
                    }

                    // Add exp(score - new_max) * v for each kv in the block
                    let new_max = online[qi].running_max;
                    for ki in 0..kv_block_len {
                        let score = row_scores[ki];
                        let weight = if score == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (score - new_max).exp()
                        };
                        let kv_idx = kv_start + ki;
                        let v_off = kv_idx * head_dim;
                        for d in 0..head_dim {
                            acc[acc_off + d] += weight * v[v_off + d];
                        }
                    }
                }
            }

            // Normalise and write to output
            for qi in 0..q_block_len {
                let q_idx = q_start + qi;
                let acc_slice = &acc[qi * head_dim..(qi + 1) * head_dim];
                let normalised = online[qi].finalize(acc_slice);
                let out_off = q_idx * head_dim;
                output[out_off..out_off + head_dim].copy_from_slice(&normalised);
            }
        }

        Ok(output)
    }

    /// Run Flash Attention for all heads in a multi-head tensor.
    ///
    /// # Arguments
    /// * `q`         – flat `[seq_len, num_heads * head_dim]` query.
    /// * `k`         – flat `[seq_len, num_heads * head_dim]` key.
    /// * `v`         – flat `[seq_len, num_heads * head_dim]` value.
    /// * `seq_len`   – number of tokens.
    /// * `num_heads` – number of attention heads.
    /// * `head_dim`  – dimension per head.
    ///
    /// Returns flat `[seq_len, num_heads * head_dim]` output.
    ///
    /// # Errors
    /// Returns an error string on size mismatches.
    pub fn forward_multi_head(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let total_dim = num_heads * head_dim;
        let expected = seq_len * total_dim;
        if q.len() != expected {
            return Err(format!(
                "q multi-head length mismatch: expected {} got {}",
                expected,
                q.len()
            ));
        }
        if k.len() != expected {
            return Err(format!(
                "k multi-head length mismatch: expected {} got {}",
                expected,
                k.len()
            ));
        }
        if v.len() != expected {
            return Err(format!(
                "v multi-head length mismatch: expected {} got {}",
                expected,
                v.len()
            ));
        }

        let mut output = vec![0.0f32; expected];

        for h in 0..num_heads {
            // Extract per-head slices: [seq_len, head_dim]
            let mut qh = vec![0.0f32; seq_len * head_dim];
            let mut kh = vec![0.0f32; seq_len * head_dim];
            let mut vh = vec![0.0f32; seq_len * head_dim];

            for t in 0..seq_len {
                let src_off = t * total_dim + h * head_dim;
                let dst_off = t * head_dim;
                qh[dst_off..dst_off + head_dim].copy_from_slice(&q[src_off..src_off + head_dim]);
                kh[dst_off..dst_off + head_dim].copy_from_slice(&k[src_off..src_off + head_dim]);
                vh[dst_off..dst_off + head_dim].copy_from_slice(&v[src_off..src_off + head_dim]);
            }

            let head_out = self.forward(&qh, &kh, &vh, seq_len, head_dim)?;

            // Write head output back into interleaved layout
            for t in 0..seq_len {
                let src_off = t * head_dim;
                let dst_off = t * total_dim + h * head_dim;
                output[dst_off..dst_off + head_dim]
                    .copy_from_slice(&head_out[src_off..src_off + head_dim]);
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Naive reference implementation (for testing only)
// ---------------------------------------------------------------------------

/// Compute standard scaled-dot-product attention (naive O(N²) algorithm).
///
/// Used exclusively for correctness comparison in tests.
/// Input shapes: `[seq_len, head_dim]` for q, k, v.
/// Returns `[seq_len, head_dim]` output.
pub fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        // Compute raw scores for row i
        let mut scores: Vec<f32> = (0..seq_len)
            .map(|j| {
                if causal && j > i {
                    return f32::NEG_INFINITY;
                }
                let q_off = i * head_dim;
                let k_off = j * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
                dot * scale
            })
            .collect();

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores
            .iter()
            .map(|&s| {
                if s == f32::NEG_INFINITY {
                    0.0
                } else {
                    (s - max_s).exp()
                }
            })
            .collect();
        let sum_e: f32 = exps.iter().sum();
        let safe = if sum_e == 0.0 { 1.0 } else { sum_e };
        for s in scores.iter_mut() {
            let _ = s; // silence unused mut warning
        }
        let attn: Vec<f32> = exps.iter().map(|&e| e / safe).collect();

        // Weighted sum
        for d in 0..head_dim {
            let val: f32 = (0..seq_len).map(|j| attn[j] * v[j * head_dim + d]).sum();
            output[i * head_dim + d] = val;
        }
    }
    output
}

/// Ceiling division: `ceil(a / b)`.
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSE_EPS: f32 = 1e-4;

    fn all_close(a: &[f32], b: &[f32]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= CLOSE_EPS)
    }

    /// Simple deterministic "random" numbers for tests (no rand dependency).
    fn make_data(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * seed + 0.37).sin() * 0.5 + 0.5) * 0.2 - 0.1)
            .collect()
    }

    // ------------------------------------------------------------------
    // Flash vs Naive equivalence tests
    // ------------------------------------------------------------------

    #[test]
    fn test_flash_matches_naive_basic() {
        let seq_len = 8;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 1.1);
        let k = make_data(seq_len * head_dim, 2.2);
        let v = make_data(seq_len * head_dim, 3.3);

        let config = FlashConfig::new(4, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("flash forward");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(
            all_close(&flash_out, &naive_out),
            "flash vs naive mismatch (non-causal)"
        );
    }

    #[test]
    fn test_flash_matches_naive_causal() {
        let seq_len = 8;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 4.4);
        let k = make_data(seq_len * head_dim, 5.5);
        let v = make_data(seq_len * head_dim, 6.6);

        let config = FlashConfig::new(4, true, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("flash causal");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);

        assert!(
            all_close(&flash_out, &naive_out),
            "flash vs naive mismatch (causal)"
        );
    }

    #[test]
    fn test_flash_block_size_1_degenerate() {
        // block_size=1 is the most granular possible tiling
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 7.7);
        let k = make_data(seq_len * head_dim, 8.8);
        let v = make_data(seq_len * head_dim, 9.9);

        let config = FlashConfig::new(1, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("block=1");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(all_close(&flash_out, &naive_out), "block_size=1 mismatch");
    }

    #[test]
    fn test_flash_seq_len_not_divisible_by_block() {
        // seq_len=7, block_size=4 — last block is partial
        let seq_len = 7;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 1.5);
        let k = make_data(seq_len * head_dim, 2.5);
        let v = make_data(seq_len * head_dim, 3.5);

        let config = FlashConfig::new(4, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("partial block");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(all_close(&flash_out, &naive_out), "partial block mismatch");
    }

    #[test]
    fn test_flash_matches_naive_seq1() {
        // Edge case: seq_len=1 → trivial (single token attends to itself)
        let seq_len = 1;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = vec![1.0f32, 0.5, -0.2, 0.8];
        let k = vec![0.3f32, -0.1, 0.7, 0.2];
        let v = vec![0.1f32, 0.2, 0.3, 0.4];

        let config = FlashConfig::new(4, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash.forward(&q, &k, &v, seq_len, head_dim).expect("seq1");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(all_close(&flash_out, &naive_out), "seq_len=1 mismatch");
    }

    #[test]
    fn test_flash_large_block_exceeds_seq() {
        // block_size > seq_len → single block, should still match naive
        let seq_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 2.3);
        let k = make_data(seq_len * head_dim, 3.4);
        let v = make_data(seq_len * head_dim, 4.5);

        let config = FlashConfig::new(16, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("large block");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(all_close(&flash_out, &naive_out), "large block mismatch");
    }

    #[test]
    fn test_flash_causal_block_size_1() {
        let seq_len = 5;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 5.5);
        let k = make_data(seq_len * head_dim, 6.6);
        let v = make_data(seq_len * head_dim, 7.7);

        let config = FlashConfig::new(1, true, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward(&q, &k, &v, seq_len, head_dim)
            .expect("causal block=1");
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, scale, true);

        assert!(
            all_close(&flash_out, &naive_out),
            "causal block_size=1 mismatch"
        );
    }

    // ------------------------------------------------------------------
    // Multi-head tests
    // ------------------------------------------------------------------

    #[test]
    fn test_flash_multi_head_output_shape() {
        let seq_len = 6;
        let num_heads = 2;
        let head_dim = 4;
        let total_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * total_dim, 1.0);
        let k = make_data(seq_len * total_dim, 2.0);
        let v = make_data(seq_len * total_dim, 3.0);

        let config = FlashConfig::new(4, false, scale);
        let flash = FlashAttention::new(config);
        let out = flash
            .forward_multi_head(&q, &k, &v, seq_len, num_heads, head_dim)
            .expect("multi-head forward");
        assert_eq!(out.len(), seq_len * total_dim);
    }

    #[test]
    fn test_flash_multi_head_matches_naive() {
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 4;
        let total_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * total_dim, 1.1);
        let k = make_data(seq_len * total_dim, 2.2);
        let v = make_data(seq_len * total_dim, 3.3);

        let config = FlashConfig::new(4, false, scale);
        let flash = FlashAttention::new(config);
        let flash_out = flash
            .forward_multi_head(&q, &k, &v, seq_len, num_heads, head_dim)
            .expect("multi-head");

        // Reference: apply naive per head
        let mut naive_out = vec![0.0f32; seq_len * total_dim];
        for h in 0..num_heads {
            let mut qh = vec![0.0f32; seq_len * head_dim];
            let mut kh = vec![0.0f32; seq_len * head_dim];
            let mut vh = vec![0.0f32; seq_len * head_dim];
            for t in 0..seq_len {
                let src = t * total_dim + h * head_dim;
                let dst = t * head_dim;
                qh[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                kh[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                vh[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
            let ho = naive_attention(&qh, &kh, &vh, seq_len, head_dim, scale, false);
            for t in 0..seq_len {
                let dst = t * total_dim + h * head_dim;
                naive_out[dst..dst + head_dim]
                    .copy_from_slice(&ho[t * head_dim..t * head_dim + head_dim]);
            }
        }

        assert!(
            all_close(&flash_out, &naive_out),
            "multi-head flash vs naive mismatch"
        );
    }

    // ------------------------------------------------------------------
    // Causal mask check helper tests
    // ------------------------------------------------------------------

    #[test]
    fn test_causal_mask_check_diagonal() {
        assert!(
            FlashAttention::causal_mask_check(3, 3),
            "diagonal should be allowed"
        );
    }

    #[test]
    fn test_causal_mask_check_lower_triangle() {
        assert!(
            FlashAttention::causal_mask_check(5, 2),
            "lower triangle should be allowed"
        );
    }

    #[test]
    fn test_causal_mask_check_upper_triangle() {
        assert!(
            !FlashAttention::causal_mask_check(2, 5),
            "upper triangle should be masked"
        );
    }

    // ------------------------------------------------------------------
    // OnlineSoftmax tests
    // ------------------------------------------------------------------

    #[test]
    fn test_online_softmax_initial_state() {
        let os = OnlineSoftmax::new();
        assert!(os.running_max.is_infinite() && os.running_max < 0.0);
        assert_eq!(os.running_sum, 0.0);
    }

    #[test]
    fn test_online_softmax_finalize_all_same() {
        // If all logits are equal, softmax gives uniform distribution.
        // For a single update with n equal logits, finalize(ones) should give ones / n.
        let mut os = OnlineSoftmax::new();
        let logits = vec![2.0f32, 2.0, 2.0, 2.0]; // 4 equal logits
        os.update(&logits);
        // Each exp(2-2)=1.0, sum=4
        let acc = vec![1.0f32; 1]; // pretend we accumulated sum of 4 * (1/4) = 1 weight
                                   // finalize normalises by running_sum=4
        let out = os.finalize(&acc);
        let expected = 1.0 / 4.0;
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "finalize fail: {}",
            out[0]
        );
    }

    #[test]
    fn test_online_softmax_update_correction() {
        let mut os = OnlineSoftmax::new();
        // First update: logits = [1.0]
        let (_, corr1) = os.update(&[1.0]);
        // Initial max was -inf, so correction = exp(-inf - new_max) should be 0
        // But since -inf - 1.0 = -inf, exp(-inf) = 0
        assert!(
            (corr1 - 0.0).abs() < 1e-5 || corr1 == 0.0,
            "first corr: {}",
            corr1
        );

        // Second update: logits = [3.0]
        let prev_max = os.running_max;
        let (new_max, corr2) = os.update(&[3.0]);
        let expected_corr = (prev_max - new_max).exp();
        assert!(
            (corr2 - expected_corr).abs() < 1e-5,
            "second corr: {}",
            corr2
        );
    }

    // ------------------------------------------------------------------
    // Error handling tests
    // ------------------------------------------------------------------

    #[test]
    fn test_flash_block_size_zero_rejected() {
        let config = FlashConfig::new(0, false, 0.5);
        let flash = FlashAttention::new(config);
        let result = flash.forward(&[1.0], &[1.0], &[1.0], 1, 1);
        assert!(result.is_err(), "block_size=0 should error");
    }

    #[test]
    fn test_flash_q_size_mismatch_rejected() {
        let config = FlashConfig::new(4, false, 0.5);
        let flash = FlashAttention::new(config);
        // q has wrong length
        let result = flash.forward(&[1.0; 3], &[1.0; 8], &[1.0; 8], 2, 4);
        assert!(result.is_err(), "q size mismatch should error");
    }

    #[test]
    fn test_flash_default_scale_config() {
        let head_dim = 16;
        let config = FlashConfig::default_scale(8, false, head_dim);
        let expected_scale = 1.0 / (16.0f32).sqrt();
        assert!((config.scale - expected_scale).abs() < 1e-6);
        assert_eq!(config.block_size, 8);
        assert!(!config.causal);
    }
}
