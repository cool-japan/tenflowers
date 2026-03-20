//! Temporal Convolutional Network (TCN) implementation.
//!
//! Implements the architecture from Bai et al. (2018) "An Empirical Evaluation of
//! Generic Convolutional and Recurrent Networks for Sequence Modeling".
//!
//! Key properties:
//! - **Causality**: convolutions never leak future information via left-padding only.
//! - **Exponential dilation**: level `i` uses dilation `2^i`, giving exponentially
//!   large receptive fields.
//! - **Residual connections**: skip connections (with optional linear projection when
//!   channels differ) stabilise deep stacks.
//! - **Weight initialisation**: Kaiming-uniform style, scaled by `1/√(fan_in)`.
//!
//! # Example
//!
//! ```rust
//! use tenflowers_neural::layers::tcn::{Tcn, TcnError};
//!
//! let tcn = Tcn::new(1, 16, 8, 3, 4, 0.0);
//! let seq: Vec<Vec<f32>> = (0..32).map(|t| vec![t as f32]).collect();
//! let out = tcn.forward(&seq).expect("forward failed");
//! assert_eq!(out.len(), 32);
//! assert_eq!(out[0].len(), 8);
//! ```

use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise inside TCN operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TcnError {
    /// Input sequence is empty.
    EmptyInput,
    /// A dimension did not match the expected value.
    DimensionMismatch { expected: usize, found: usize },
    /// Kernel size ≤ 0 is not allowed (must be ≥ 1).
    InvalidKernelSize { size: usize },
    /// Dilation must be ≥ 1.
    InvalidDilation { dilation: usize },
    /// Number of levels must be ≥ 1.
    InvalidNumLevels { levels: usize },
}

impl std::fmt::Display for TcnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TcnError::EmptyInput => write!(f, "TCN: input sequence is empty"),
            TcnError::DimensionMismatch { expected, found } => {
                write!(f, "TCN: dimension mismatch — expected {expected}, found {found}")
            }
            TcnError::InvalidKernelSize { size } => {
                write!(f, "TCN: kernel_size must be ≥ 1, got {size}")
            }
            TcnError::InvalidDilation { dilation } => {
                write!(f, "TCN: dilation must be ≥ 1, got {dilation}")
            }
            TcnError::InvalidNumLevels { levels } => {
                write!(f, "TCN: num_levels must be ≥ 1, got {levels}")
            }
        }
    }
}

impl std::error::Error for TcnError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal weight initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic Kaiming-uniform weight initialiser.
///
/// Produces a pseudo-random weight tensor scaled by `sqrt(2 / fan_in)` using a
/// simple LCG so the crate stays `rand`-free.
fn kaiming_weights(out_channels: usize, in_channels: usize, kernel_size: usize, seed_offset: u64) -> Vec<Vec<Vec<f32>>> {
    let fan_in = in_channels * kernel_size;
    let scale = (2.0_f32 / fan_in as f32).sqrt();
    let mut state: u64 = 6_364_136_223_846_793_005_u64
        .wrapping_mul(seed_offset.wrapping_add(1))
        .wrapping_add(1_442_695_040_888_963_407);

    let mut weights = Vec::with_capacity(out_channels);
    for _ in 0..out_channels {
        let mut in_vecs = Vec::with_capacity(in_channels);
        for _ in 0..in_channels {
            let mut ks = Vec::with_capacity(kernel_size);
            for _ in 0..kernel_size {
                // LCG step — produces a float in (-1, 1)
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let bits = ((state >> 33) as f32) / (u32::MAX as f32);
                let val = (bits * 2.0 - 1.0) * scale;
                ks.push(val);
            }
            in_vecs.push(ks);
        }
        weights.push(in_vecs);
    }
    weights
}

/// Tiny square-weight initialiser for linear projections: [rows, cols].
fn linear_weights(rows: usize, cols: usize, seed_offset: u64) -> Vec<Vec<f32>> {
    let scale = (2.0_f32 / rows as f32).sqrt();
    let mut state: u64 = 2_862_933_555_777_941_757_u64
        .wrapping_mul(seed_offset.wrapping_add(7))
        .wrapping_add(3_037_000_499);
    let mut mat = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            state = state.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3_037_000_499);
            let bits = ((state >> 33) as f32) / (u32::MAX as f32);
            row.push((bits * 2.0 - 1.0) * scale);
        }
        mat.push(row);
    }
    mat
}

// ─────────────────────────────────────────────────────────────────────────────
// CausalConv1d
// ─────────────────────────────────────────────────────────────────────────────

/// 1-D causal convolution.
///
/// The convolution pads the **left** side only with `(kernel_size - 1) * dilation`
/// zeros so that position `t` in the output depends only on positions `≤ t` in
/// the input.  No future information ever leaks.
///
/// # Shape
/// - Input:  `[seq_len, in_channels]`
/// - Output: `[seq_len, out_channels]`
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    /// Weights layout: `[out_channels][in_channels][kernel_size]`
    pub weight: Vec<Vec<Vec<f32>>>,
    /// Bias vector of length `out_channels`.
    pub bias: Vec<f32>,
}

impl CausalConv1d {
    /// Construct a new layer.  Weights are Kaiming-uniform initialised;
    /// biases are zero-initialised.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, dilation: usize) -> Self {
        let seed = (in_channels as u64)
            .wrapping_mul(31)
            .wrapping_add(out_channels as u64)
            .wrapping_mul(37)
            .wrapping_add(kernel_size as u64)
            .wrapping_mul(41)
            .wrapping_add(dilation as u64);
        let weight = kaiming_weights(out_channels, in_channels, kernel_size, seed);
        let bias = vec![0.0_f32; out_channels];
        CausalConv1d { in_channels, out_channels, kernel_size, dilation, weight, bias }
    }

    /// The effective receptive field of this single layer (in time-steps).
    ///
    /// `receptive_field = kernel_size + (kernel_size - 1) * (dilation - 1)`
    pub fn receptive_field(&self) -> usize {
        self.kernel_size + (self.kernel_size.saturating_sub(1)) * (self.dilation.saturating_sub(1))
    }

    /// Causal forward pass.
    ///
    /// Left-pads with `(kernel_size - 1) * dilation` zeros then slides the
    /// dilated kernel over the padded sequence.
    ///
    /// # Errors
    /// - [`TcnError::EmptyInput`] if `input` is empty.
    /// - [`TcnError::DimensionMismatch`] if a token width ≠ `in_channels`.
    /// - [`TcnError::InvalidKernelSize`] if `kernel_size == 0`.
    /// - [`TcnError::InvalidDilation`] if `dilation == 0`.
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TcnError> {
        if input.is_empty() {
            return Err(TcnError::EmptyInput);
        }
        if self.kernel_size == 0 {
            return Err(TcnError::InvalidKernelSize { size: 0 });
        }
        if self.dilation == 0 {
            return Err(TcnError::InvalidDilation { dilation: 0 });
        }
        let seq_len = input.len();
        // Validate channel widths
        for (t, token) in input.iter().enumerate() {
            if token.len() != self.in_channels {
                return Err(TcnError::DimensionMismatch {
                    expected: self.in_channels,
                    found: token.len(),
                });
            }
            let _ = t; // suppress unused warning
        }

        let pad = (self.kernel_size - 1) * self.dilation;
        // Build padded input: [seq_len + pad, in_channels]
        let padded_len = seq_len + pad;
        let mut padded: Vec<Vec<f32>> = Vec::with_capacity(padded_len);
        for _ in 0..pad {
            padded.push(vec![0.0_f32; self.in_channels]);
        }
        for token in input {
            padded.push(token.clone());
        }

        // Convolve
        let mut out = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut token_out = self.bias.clone();
            for oc in 0..self.out_channels {
                for k in 0..self.kernel_size {
                    let src_t = t + pad - k * self.dilation;
                    // src_t is always valid because we padded enough
                    let src = &padded[src_t];
                    for ic in 0..self.in_channels {
                        token_out[oc] += src[ic] * self.weight[oc][ic][k];
                    }
                }
            }
            out.push(token_out);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TcnBlock
// ─────────────────────────────────────────────────────────────────────────────

/// Dilated residual block as in Bai et al. (2018).
///
/// Layout:
/// ```text
/// input ─► CausalConv1(dilation) ─► ReLU ─► Dropout
///       ─► CausalConv2(dilation) ─► ReLU ─► Dropout ─► + residual ─► output
/// ```
///
/// The residual path linearly projects input channels to output channels when
/// they differ (`downsample` weight matrix).
#[derive(Debug, Clone)]
pub struct TcnBlock {
    pub conv1: CausalConv1d,
    pub conv2: CausalConv1d,
    /// Linear projection `[in_channels, out_channels]` applied to the residual
    /// when `in_channels != out_channels`.  `None` when they match.
    pub downsample: Option<Vec<Vec<f32>>>,
    pub dropout_rate: f32,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl TcnBlock {
    /// Construct a residual TCN block.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        dropout: f32,
    ) -> Self {
        let conv1 = CausalConv1d::new(in_channels, out_channels, kernel_size, dilation);
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, dilation);

        let downsample = if in_channels != out_channels {
            let seed = (in_channels as u64).wrapping_mul(53).wrapping_add(out_channels as u64).wrapping_mul(59);
            Some(linear_weights(in_channels, out_channels, seed))
        } else {
            None
        };

        TcnBlock { conv1, conv2, downsample, dropout_rate: dropout, in_channels, out_channels }
    }

    /// ReLU activation: `max(0, x)`.
    #[inline]
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// Scale activations by `1 - dropout_rate` (inference-time deterministic dropout).
    fn apply_dropout(&self, x: &[f32]) -> Vec<f32> {
        let scale = 1.0 - self.dropout_rate.clamp(0.0, 1.0);
        x.iter().map(|&v| v * scale).collect()
    }

    /// Compute the residual path.
    ///
    /// If channels match, returns the input unchanged.  Otherwise applies the
    /// stored `downsample` linear projection (no bias) to project into the
    /// output channel space.
    fn residual(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TcnError> {
        match &self.downsample {
            None => Ok(x.to_vec()),
            Some(w) => {
                // w: [in_channels, out_channels]
                let mut out = Vec::with_capacity(x.len());
                for token in x {
                    if token.len() != self.in_channels {
                        return Err(TcnError::DimensionMismatch {
                            expected: self.in_channels,
                            found: token.len(),
                        });
                    }
                    let mut proj = vec![0.0_f32; self.out_channels];
                    for ic in 0..self.in_channels {
                        for oc in 0..self.out_channels {
                            proj[oc] += token[ic] * w[ic][oc];
                        }
                    }
                    out.push(proj);
                }
                Ok(out)
            }
        }
    }

    /// Forward pass through the residual block.
    ///
    /// `conv1 → relu → dropout → conv2 → relu → dropout → add residual`
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TcnError> {
        if input.is_empty() {
            return Err(TcnError::EmptyInput);
        }

        // First causal conv + ReLU + dropout
        let h1 = self.conv1.forward(input)?;
        let h1: Vec<Vec<f32>> = h1
            .into_iter()
            .map(|tok| self.apply_dropout(&tok.into_iter().map(Self::relu).collect::<Vec<_>>()))
            .collect();

        // Second causal conv + ReLU + dropout
        let h2 = self.conv2.forward(&h1)?;
        let h2: Vec<Vec<f32>> = h2
            .into_iter()
            .map(|tok| self.apply_dropout(&tok.into_iter().map(Self::relu).collect::<Vec<_>>()))
            .collect();

        // Residual and addition
        let res = self.residual(input)?;
        let out: Vec<Vec<f32>> = h2
            .into_iter()
            .zip(res.into_iter())
            .map(|(h, r)| h.iter().zip(r.iter()).map(|(a, b)| a + b).collect())
            .collect();

        Ok(out)
    }

    /// The effective receptive field of this block (uses conv1 dilation/kernel).
    pub fn receptive_field(&self) -> usize {
        // Two causal convolutions stacked — each contributes equally.
        self.conv1.receptive_field() + self.conv2.receptive_field() - 1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tcn
// ─────────────────────────────────────────────────────────────────────────────

/// Full Temporal Convolutional Network.
///
/// Stacks `num_levels` [`TcnBlock`] layers.  Level `i` uses dilation `2^i`
/// so the receptive field grows exponentially with depth.
///
/// ```text
/// Input → Block(d=1) → Block(d=2) → Block(d=4) → … → Block(d=2^(L-1)) → Output
/// ```
///
/// The first block projects from `in_channels` to `hidden_channels`;
/// intermediate blocks stay at `hidden_channels`;
/// the last block projects to `out_channels`.
#[derive(Debug, Clone)]
pub struct Tcn {
    pub blocks: Vec<TcnBlock>,
    pub input_channels: usize,
    pub output_channels: usize,
    pub num_levels: usize,
    pub kernel_size: usize,
}

impl Tcn {
    /// Build a TCN.
    ///
    /// # Arguments
    /// - `in_channels`     – input feature width.
    /// - `hidden_channels` – channel width of intermediate blocks.
    /// - `out_channels`    – output feature width.
    /// - `kernel_size`     – convolution kernel size (applied to all blocks).
    /// - `num_levels`      – number of residual blocks.
    /// - `dropout`         – dropout rate in `[0, 1)`.
    ///
    /// # Panics
    /// Does not panic; validation happens in [`forward`](Self::forward).
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        num_levels: usize,
        dropout: f32,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_levels);
        for i in 0..num_levels {
            let dilation = 1usize << i; // 2^i
            let block_in = if i == 0 { in_channels } else { hidden_channels };
            let block_out = if i == num_levels - 1 { out_channels } else { hidden_channels };
            blocks.push(TcnBlock::new(block_in, block_out, kernel_size, dilation, dropout));
        }
        Tcn { blocks, input_channels: in_channels, output_channels: out_channels, num_levels, kernel_size }
    }

    /// Total receptive field of the entire TCN stack.
    ///
    /// Each block at dilation `2^i` contributes `2 * (kernel_size - 1) * 2^i`
    /// additional time-steps.  The overall RF is:
    ///
    /// `1 + Σ_{i=0}^{L-1}  2 * (kernel_size - 1) * 2^i`
    pub fn receptive_field(&self) -> usize {
        let mut rf = 1usize;
        for i in 0..self.num_levels {
            rf += 2 * (self.kernel_size.saturating_sub(1)) * (1usize << i);
        }
        rf
    }

    /// Forward pass through the full TCN stack.
    ///
    /// # Errors
    /// - [`TcnError::EmptyInput`] if `input` is empty.
    /// - [`TcnError::InvalidNumLevels`] if the TCN has zero blocks.
    /// - Propagated errors from underlying [`TcnBlock::forward`].
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TcnError> {
        if input.is_empty() {
            return Err(TcnError::EmptyInput);
        }
        if self.blocks.is_empty() {
            return Err(TcnError::InvalidNumLevels { levels: 0 });
        }

        let mut x: Vec<Vec<f32>> = input.to_vec();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    /// Return the output vector at the **last time-step** (single-step prediction).
    ///
    /// # Errors
    /// Inherits errors from [`forward`](Self::forward).
    pub fn predict(&self, input: &[Vec<f32>]) -> Result<Vec<f32>, TcnError> {
        let out = self.forward(input)?;
        // `out` is non-empty because forward guarantees seq_len is preserved.
        match out.last() {
            Some(v) => Ok(v.clone()),
            None => Err(TcnError::EmptyInput),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalAttention
// ─────────────────────────────────────────────────────────────────────────────

/// Scaled dot-product self-attention over a temporal sequence.
///
/// A single attention head operating on `[seq_len, d_model]` input,
/// returning an output of the same shape.  Weights are query, key, and value
/// projections each of shape `[d_model, d_model]`.
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    pub d_model: usize,
    /// Query projection `[d_model, d_model]` row-major.
    pub query_weight: Vec<Vec<f32>>,
    /// Key projection `[d_model, d_model]` row-major.
    pub key_weight: Vec<Vec<f32>>,
    /// Value projection `[d_model, d_model]` row-major.
    pub value_weight: Vec<Vec<f32>>,
}

impl TemporalAttention {
    /// Construct a `TemporalAttention` module with `d_model` hidden dimensions.
    pub fn new(d_model: usize) -> Self {
        let qw = linear_weights(d_model, d_model, 101);
        let kw = linear_weights(d_model, d_model, 103);
        let vw = linear_weights(d_model, d_model, 107);
        TemporalAttention { d_model, query_weight: qw, key_weight: kw, value_weight: vw }
    }

    /// Project all tokens through a `[d_model, d_model]` weight matrix.
    fn project(tokens: &[Vec<f32>], weight: &[Vec<f32>], d_model: usize) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .map(|tok| {
                let mut out = vec![0.0_f32; d_model];
                for i in 0..d_model {
                    for j in 0..d_model {
                        out[j] += tok[i] * weight[i][j];
                    }
                }
                out
            })
            .collect()
    }

    /// Numerically-stable softmax over a slice.
    fn softmax(v: &mut [f32]) {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for x in v.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in v.iter_mut() {
                *x /= sum;
            }
        }
    }

    /// Self-attention forward pass.
    ///
    /// # Errors
    /// - [`TcnError::EmptyInput`] if `input` is empty.
    /// - [`TcnError::DimensionMismatch`] if any token has width ≠ `d_model`.
    pub fn forward(&self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TcnError> {
        if input.is_empty() {
            return Err(TcnError::EmptyInput);
        }
        for token in input {
            if token.len() != self.d_model {
                return Err(TcnError::DimensionMismatch {
                    expected: self.d_model,
                    found: token.len(),
                });
            }
        }

        let seq_len = input.len();
        let scale = 1.0 / (self.d_model as f32).sqrt();

        let q = Self::project(input, &self.query_weight, self.d_model);
        let k = Self::project(input, &self.key_weight, self.d_model);
        let v = Self::project(input, &self.value_weight, self.d_model);

        // Attention scores: [seq_len, seq_len]
        let mut scores: Vec<Vec<f32>> = vec![vec![0.0_f32; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f32 = q[i].iter().zip(k[j].iter()).map(|(a, b)| a * b).sum();
                scores[i][j] = dot * scale;
            }
        }

        // Softmax per query position
        for row in scores.iter_mut() {
            Self::softmax(row);
        }

        // Weighted sum of values
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let mut token_out = vec![0.0_f32; self.d_model];
            for j in 0..seq_len {
                let w = scores[i][j];
                for d in 0..self.d_model {
                    token_out[d] += w * v[j][d];
                }
            }
            out.push(token_out);
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a synthetic sequence [seq_len, channels] filled with
    // a constant value.
    fn const_seq(seq_len: usize, channels: usize, val: f32) -> Vec<Vec<f32>> {
        vec![vec![val; channels]; seq_len]
    }

    // Helper: create an identity-like sequence where token t has all values = t.
    fn ramp_seq(seq_len: usize, channels: usize) -> Vec<Vec<f32>> {
        (0..seq_len).map(|t| vec![t as f32; channels]).collect()
    }

    // ─── CausalConv1d ────────────────────────────────────────────────────────

    #[test]
    fn test_causal_conv1d_output_shape() {
        let conv = CausalConv1d::new(3, 8, 3, 1);
        let input = const_seq(20, 3, 1.0);
        let out = conv.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 20, "output seq_len should match input");
        assert_eq!(out[0].len(), 8, "output channels should be out_channels");
    }

    #[test]
    fn test_causal_conv1d_single_token() {
        // kernel_size=1, dilation=1 → no padding needed.
        let conv = CausalConv1d::new(4, 4, 1, 1);
        let input = const_seq(1, 4, 2.0);
        let out = conv.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_causal_conv1d_receptive_field_formula() {
        // RF = kernel_size + (kernel_size - 1) * (dilation - 1)
        // k=3, d=1 → 3 + 2*0 = 3
        let conv1 = CausalConv1d::new(1, 1, 3, 1);
        assert_eq!(conv1.receptive_field(), 3);

        // k=3, d=2 → 3 + 2*1 = 5
        let conv2 = CausalConv1d::new(1, 1, 3, 2);
        assert_eq!(conv2.receptive_field(), 5);

        // k=5, d=4 → 5 + 4*3 = 17
        let conv3 = CausalConv1d::new(1, 1, 5, 4);
        assert_eq!(conv3.receptive_field(), 17);
    }

    #[test]
    fn test_causal_conv1d_causality() {
        // Causality check: output at time t should change if we modify input at t,
        // but should NOT change if we modify input at t+1 (future).
        // We use a manual weight setup for a single input/output channel.
        let mut conv = CausalConv1d::new(1, 1, 3, 1);
        // Set weights to all 1.0 and bias to 0 for a transparent sum.
        conv.weight = vec![vec![vec![1.0_f32; 3]]];
        conv.bias = vec![0.0_f32];

        // Sequence: [0, 0, 0, 10, 0, 0, 0] — spike at t=3
        let mut input: Vec<Vec<f32>> = vec![vec![0.0_f32]; 7];
        input[3] = vec![10.0_f32];

        let out = conv.forward(&input).expect("forward failed");

        // Output at t<3 must be 0 (no contribution from t=3 which is future/same).
        // Actually at t=3 the kernel [k=0,1,2] reads positions [3,2,1] → includes spike.
        // Output at t=2: reads positions [2,1,0] → all zeros → 0.
        assert_eq!(out[2][0], 0.0, "output before spike must be zero (causality)");
        assert!(out[3][0] != 0.0, "output at spike position should be non-zero");
        // Output at t=4: reads positions [4,3,2] → includes spike.
        assert!(out[4][0] != 0.0, "output after spike incorporates history");
    }

    #[test]
    fn test_causal_conv1d_dilation_shape() {
        let conv = CausalConv1d::new(2, 4, 3, 4);
        let input = ramp_seq(15, 2);
        let out = conv.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 15);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_causal_conv1d_empty_input_error() {
        let conv = CausalConv1d::new(3, 3, 3, 1);
        let result = conv.forward(&[]);
        assert!(matches!(result, Err(TcnError::EmptyInput)));
    }

    #[test]
    fn test_causal_conv1d_dimension_mismatch_error() {
        let conv = CausalConv1d::new(3, 3, 3, 1);
        let bad_input = vec![vec![1.0_f32; 5]; 10]; // 5 ≠ 3
        let result = conv.forward(&bad_input);
        assert!(matches!(result, Err(TcnError::DimensionMismatch { .. })));
    }

    // ─── TcnBlock ────────────────────────────────────────────────────────────

    #[test]
    fn test_tcn_block_output_shape() {
        let block = TcnBlock::new(4, 8, 3, 1, 0.0);
        let input = const_seq(12, 4, 1.0);
        let out = block.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 12);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_tcn_block_same_channels_no_downsample() {
        let block = TcnBlock::new(6, 6, 3, 1, 0.0);
        assert!(block.downsample.is_none(), "no downsample when channels match");
    }

    #[test]
    fn test_tcn_block_different_channels_has_downsample() {
        let block = TcnBlock::new(4, 8, 3, 1, 0.0);
        assert!(block.downsample.is_some(), "downsample required when channels differ");
    }

    #[test]
    fn test_tcn_block_residual_same_channels() {
        let block = TcnBlock::new(3, 3, 1, 1, 0.0);
        let input = const_seq(5, 3, 2.0);
        // When kernel_size=1 and channels match, residual is added directly.
        let out = block.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 3);
    }

    #[test]
    fn test_tcn_block_empty_input_error() {
        let block = TcnBlock::new(4, 4, 3, 1, 0.0);
        let result = block.forward(&[]);
        assert!(matches!(result, Err(TcnError::EmptyInput)));
    }

    #[test]
    fn test_tcn_block_receptive_field() {
        let block = TcnBlock::new(4, 4, 3, 2, 0.0);
        // Each conv: RF = 3 + 2*(2-1) = 5
        // Block RF = conv1.RF + conv2.RF - 1 = 5 + 5 - 1 = 9
        assert_eq!(block.receptive_field(), 9);
    }

    // ─── Tcn ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_tcn_forward_output_shape() {
        let tcn = Tcn::new(2, 8, 4, 3, 4, 0.0);
        let input = const_seq(30, 2, 0.5);
        let out = tcn.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 30, "seq_len preserved");
        assert_eq!(out[0].len(), 4, "output channels correct");
    }

    #[test]
    fn test_tcn_receptive_field_grows_with_levels() {
        let tcn1 = Tcn::new(1, 8, 4, 3, 3, 0.0);
        let tcn2 = Tcn::new(1, 8, 4, 3, 5, 0.0);
        assert!(
            tcn2.receptive_field() > tcn1.receptive_field(),
            "more levels → larger receptive field"
        );
    }

    #[test]
    fn test_tcn_receptive_field_formula() {
        // L=3, k=2: RF = 1 + 2*(k-1)*( 1 + 2 + 4 ) = 1 + 2*1*7 = 15
        let tcn = Tcn::new(1, 4, 4, 2, 3, 0.0);
        let expected = 1 + 2 * (2 - 1) * (1 + 2 + 4);
        assert_eq!(tcn.receptive_field(), expected);
    }

    #[test]
    fn test_tcn_predict_returns_single_timestep() {
        let tcn = Tcn::new(3, 8, 4, 3, 4, 0.0);
        let input = const_seq(20, 3, 1.0);
        let pred = tcn.predict(&input).expect("predict failed");
        assert_eq!(pred.len(), 4, "predict returns out_channels vector");
    }

    #[test]
    fn test_tcn_single_level() {
        let tcn = Tcn::new(2, 8, 2, 3, 1, 0.0);
        let input = const_seq(10, 2, 1.0);
        let out = tcn.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_tcn_empty_input_error() {
        let tcn = Tcn::new(2, 8, 4, 3, 4, 0.0);
        let result = tcn.forward(&[]);
        assert!(matches!(result, Err(TcnError::EmptyInput)));
    }

    #[test]
    fn test_tcn_dilation_doubles_each_level() {
        let tcn = Tcn::new(1, 4, 4, 3, 4, 0.0);
        assert_eq!(tcn.blocks[0].conv1.dilation, 1); // 2^0
        assert_eq!(tcn.blocks[1].conv1.dilation, 2); // 2^1
        assert_eq!(tcn.blocks[2].conv1.dilation, 4); // 2^2
        assert_eq!(tcn.blocks[3].conv1.dilation, 8); // 2^3
    }

    // ─── TemporalAttention ───────────────────────────────────────────────────

    #[test]
    fn test_temporal_attention_output_shape() {
        let attn = TemporalAttention::new(8);
        let input = const_seq(10, 8, 0.5);
        let out = attn.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_temporal_attention_empty_error() {
        let attn = TemporalAttention::new(4);
        let result = attn.forward(&[]);
        assert!(matches!(result, Err(TcnError::EmptyInput)));
    }

    #[test]
    fn test_temporal_attention_dim_mismatch_error() {
        let attn = TemporalAttention::new(8);
        let bad_input = vec![vec![1.0_f32; 5]; 4]; // 5 ≠ 8
        let result = attn.forward(&bad_input);
        assert!(matches!(result, Err(TcnError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_temporal_attention_single_token() {
        let attn = TemporalAttention::new(4);
        let input = vec![vec![1.0_f32, 2.0, 3.0, 4.0]];
        let out = attn.forward(&input).expect("forward failed");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_tcn_block_dropout_rate_zero() {
        // With dropout=0.0, scale factor is 1.0 so values are unaffected by dropout path.
        let block = TcnBlock::new(2, 2, 1, 1, 0.0);
        let v = vec![1.0_f32, 2.0, 3.0];
        let result = block.apply_dropout(&v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_tcn_block_dropout_rate_half() {
        let block = TcnBlock::new(2, 2, 1, 1, 0.5);
        let v = vec![2.0_f32, 4.0];
        let result = block.apply_dropout(&v);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }
}
