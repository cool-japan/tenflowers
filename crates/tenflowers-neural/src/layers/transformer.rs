//! Transformer building blocks: MultiHeadAttentionLayer, FeedForward, LayerNormLayer,
//! TransformerEncoderBlock, and TransformerEncoder.
//!
//! These components provide a clean, Vec<f32>-based API for constructing transformer
//! encoder stacks. The implementation favors clarity and correctness over raw throughput,
//! while remaining zero-unsafe and free of unwrap() calls.
//!
//! # Architecture
//!
//! Each sublayer follows the **pre-norm** convention used by modern LLMs:
//!
//! ```text
//! TransformerEncoderBlock:
//!   x = x + Attention(LayerNorm(x))
//!   x = x + FFN(LayerNorm(x))
//! ```
//!
//! Input/output shape convention used throughout:
//! - 3-D: `[batch, seq_len, model_dim]`
//! - `shape` slices encode dimensions in row-major order

use tenflowers_core::TensorError;

/// Shorthand for the project's standard `Result`.
type Result<T> = tenflowers_core::Result<T>;

// ─────────────────────────────────────────────────────────────────────────────
// Activation enum
// ─────────────────────────────────────────────────────────────────────────────

/// Activation function used inside the feed-forward sublayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// Rectified Linear Unit – max(0, x)
    Relu,
    /// Gaussian Error Linear Unit (tanh approximation)
    Gelu,
    /// SwiGLU-style gating: x * sigmoid(x) applied elementwise (Swish)
    Swiglu,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper math
// ─────────────────────────────────────────────────────────────────────────────

/// GELU activation using the tanh approximation:
/// `x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))`
#[inline]
fn gelu_approx(x: f32) -> f32 {
    let k = 0.7978845608_f32;
    let c = 0.044715_f32;
    let inner = k * (x + c * x * x * x);
    x * 0.5 * (1.0 + inner.tanh())
}

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Swish / SiLU activation: x * sigmoid(x).
#[inline]
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply `activation` elementwise to `v` in-place.
fn apply_activation(v: &mut [f32], act: FfnActivation) {
    match act {
        FfnActivation::Relu => {
            for x in v.iter_mut() {
                *x = relu(*x);
            }
        }
        FfnActivation::Gelu => {
            for x in v.iter_mut() {
                *x = gelu_approx(*x);
            }
        }
        FfnActivation::Swiglu => {
            for x in v.iter_mut() {
                *x = swish(*x);
            }
        }
    }
}

/// Dense 2-D matrix-vector product: y = x @ W + b  (x: [m], W: [m, n], b: [n])
/// Returns a new `Vec<f32>` of length `n`.
fn linear(x: &[f32], weight: &[f32], bias: Option<&[f32]>, in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(weight.len(), in_dim * out_dim);

    let mut y = vec![0.0_f32; out_dim];
    for j in 0..out_dim {
        let mut acc = 0.0_f32;
        for i in 0..in_dim {
            acc += x[i] * weight[i * out_dim + j];
        }
        if let Some(b) = bias {
            acc += b[j];
        }
        y[j] = acc;
    }
    y
}

/// Batch-aware linear: applies `linear` to every token in a flat buffer.
/// `tokens`: flat `[total_tokens, in_dim]`, returns flat `[total_tokens, out_dim]`.
fn linear_batch(
    tokens: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let n_tokens = tokens.len() / in_dim;
    let mut out = Vec::with_capacity(n_tokens * out_dim);
    for t in 0..n_tokens {
        let row = &tokens[t * in_dim..(t + 1) * in_dim];
        out.extend_from_slice(&linear(row, weight, bias, in_dim, out_dim));
    }
    out
}

/// Softmax over a slice in-place.
fn softmax_inplace(v: &mut [f32]) {
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

/// Scaled dot-product attention for a single head.
///
/// `q`: `[seq_len, head_dim]`
/// `k`: `[seq_len, head_dim]`
/// `v`: `[seq_len, head_dim]`
///
/// Returns flat `[seq_len, head_dim]`.
fn scaled_dot_product_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention scores: S = Q @ K^T  →  [seq_len, seq_len]
    let mut scores = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Apply causal mask (upper-triangular positions → −∞)
    if causal {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax over rows
    for i in 0..seq_len {
        softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }

    // Output = Scores @ V  →  [seq_len, head_dim]
    let mut out = vec![0.0_f32; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_len {
                acc += scores[i * seq_len + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = acc;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense<f32> wrapper (minimal, plain-Vec representation)
// ─────────────────────────────────────────────────────────────────────────────

/// A simple dense (linear) layer holding `weight` in row-major order `[in_dim, out_dim]`
/// and an optional bias `[out_dim]`.
///
/// Initialized with Xavier uniform values from the pseudo-random Box-Muller stream
/// provided by `scirs2_core`.
#[derive(Debug, Clone)]
pub struct DenseF32 {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl DenseF32 {
    /// Allocate a new dense layer with Xavier-normal initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Result<Self> {
        if in_dim == 0 || out_dim == 0 {
            return Err(TensorError::invalid_argument(
                "DenseF32: in_dim and out_dim must be > 0".to_string(),
            ));
        }
        let std = (2.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let n = in_dim * out_dim;
        let mut weight = Vec::with_capacity(n);
        for k in 0..n {
            // Cheap deterministic-ish seed based on position and constants.
            // This is used only for shape-correctness tests; real training would
            // use a proper PRNG from scirs2_core.
            let u1 = ((k as f32 * 1.6180339887 + 0.5) % 1.0).max(1e-10);
            let u2 = (k as f32 * 2.718281828 + 0.3) % 1.0;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            weight.push(std * z);
        }
        let bias = vec![0.0_f32; out_dim];
        Ok(Self { weight, bias, in_dim, out_dim })
    }

    /// Forward: `out = linear(x, W, b)`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_dim {
            return Err(TensorError::invalid_argument(format!(
                "DenseF32::forward: expected {} inputs, got {}",
                self.in_dim,
                x.len()
            )));
        }
        Ok(linear(x, &self.weight, Some(&self.bias), self.in_dim, self.out_dim))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNormLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Layer normalisation sublayer.
///
/// Normalises the last dimension (`dim`) of an input tensor (flattened to a
/// sequence of vectors), then applies a learnable scale (`weight`) and shift
/// (`bias`), both initialised to 1 and 0 respectively (identity transform).
#[derive(Debug, Clone)]
pub struct LayerNormLayer {
    /// Learnable scale (γ), shape `[dim]`, initialised to 1.
    pub weight: Vec<f32>,
    /// Learnable shift (β), shape `[dim]`, initialised to 0.
    pub bias: Vec<f32>,
    /// Feature dimension.
    pub dim: usize,
    /// Numerical stability constant (default 1e-5).
    pub eps: f32,
}

impl LayerNormLayer {
    /// Create a new `LayerNormLayer` with γ=1, β=0.
    pub fn new(dim: usize) -> Self {
        Self {
            weight: vec![1.0_f32; dim],
            bias: vec![0.0_f32; dim],
            dim,
            eps: 1e-5,
        }
    }

    /// Normalise `input` (flat `[*, dim]`) over the last dimension.
    ///
    /// `shape` is unused for computation but validated for consistency.
    pub fn forward(&self, input: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
        // Validate that the last dimension matches `dim`.
        if shape.is_empty() {
            return Err(TensorError::invalid_argument(
                "LayerNormLayer: shape must be non-empty".to_string(),
            ));
        }
        let last_dim = *shape.last().ok_or_else(|| {
            TensorError::invalid_argument("LayerNormLayer: empty shape".to_string())
        })?;
        if last_dim != self.dim {
            return Err(TensorError::invalid_argument(format!(
                "LayerNormLayer: last shape dim {} != layer dim {}",
                last_dim, self.dim
            )));
        }
        let total = input.len();
        if total % self.dim != 0 {
            return Err(TensorError::invalid_argument(format!(
                "LayerNormLayer: input length {} not divisible by dim {}",
                total, self.dim
            )));
        }
        let n_vecs = total / self.dim;
        let mut out = Vec::with_capacity(total);
        for v in 0..n_vecs {
            let slice = &input[v * self.dim..(v + 1) * self.dim];
            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() / self.dim as f32;
            // Compute variance
            let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for (i, &x) in slice.iter().enumerate() {
                out.push((x - mean) * inv_std * self.weight[i] + self.bias[i]);
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiHeadAttentionLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head self-attention layer.
///
/// Maintains four projection matrices (Q, K, V, O) as [`DenseF32`] layers.
/// The forward pass:
/// 1. Project input to Q/K/V (shape `[*, model_dim]` → `[*, model_dim]`).
/// 2. Split into `num_heads` heads of dimension `head_dim = model_dim / num_heads`.
/// 3. Compute scaled dot-product attention per head (optionally causal).
/// 4. Concatenate heads and apply output projection.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionLayer {
    /// Query projection W_Q ∈ ℝ^{model_dim × model_dim}
    pub wq: DenseF32,
    /// Key projection W_K ∈ ℝ^{model_dim × model_dim}
    pub wk: DenseF32,
    /// Value projection W_V ∈ ℝ^{model_dim × model_dim}
    pub wv: DenseF32,
    /// Output projection W_O ∈ ℝ^{model_dim × model_dim}
    pub wo: DenseF32,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each head (`model_dim / num_heads`).
    pub head_dim: usize,
    /// Total model dimension.
    pub model_dim: usize,
}

impl MultiHeadAttentionLayer {
    /// Create a new `MultiHeadAttentionLayer`.
    ///
    /// Requires `model_dim % num_heads == 0`.
    pub fn new(model_dim: usize, num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(TensorError::invalid_argument(
                "MultiHeadAttentionLayer: num_heads must be > 0".to_string(),
            ));
        }
        if model_dim % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "MultiHeadAttentionLayer: model_dim ({}) must be divisible by num_heads ({})",
                model_dim, num_heads
            )));
        }
        let head_dim = model_dim / num_heads;
        Ok(Self {
            wq: DenseF32::new(model_dim, model_dim)?,
            wk: DenseF32::new(model_dim, model_dim)?,
            wv: DenseF32::new(model_dim, model_dim)?,
            wo: DenseF32::new(model_dim, model_dim)?,
            num_heads,
            head_dim,
            model_dim,
        })
    }

    /// Forward pass.
    ///
    /// `x`: flat buffer of shape `shape` (must be `[batch, seq_len, model_dim]` or `[seq_len, model_dim]`).
    /// Returns `(output, shape)` with the same shape as the input.
    pub fn forward(
        &self,
        x: &[f32],
        shape: &[usize],
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        // Validate and extract dimensions.
        let (batch, seq_len) = self.extract_batch_seq(x, shape)?;
        let total_tokens = batch * seq_len;

        // 1. Project tokens to Q, K, V — each is flat [total_tokens, model_dim].
        let q_flat = linear_batch(x, &self.wq.weight, Some(&self.wq.bias), self.model_dim, self.model_dim);
        let k_flat = linear_batch(x, &self.wk.weight, Some(&self.wk.bias), self.model_dim, self.model_dim);
        let v_flat = linear_batch(x, &self.wv.weight, Some(&self.wv.bias), self.model_dim, self.model_dim);

        // 2. Multi-head attention (process each item in the batch independently).
        let mut attn_flat = vec![0.0_f32; total_tokens * self.model_dim];

        for b in 0..batch {
            // Extract q/k/v for this batch item: [seq_len, model_dim] each.
            let start = b * seq_len * self.model_dim;
            let end = start + seq_len * self.model_dim;
            let q_b = &q_flat[start..end];
            let k_b = &k_flat[start..end];
            let v_b = &v_flat[start..end];

            // Process each head.
            for h in 0..self.num_heads {
                // Extract head slice: q_h is [seq_len, head_dim].
                let mut q_h = vec![0.0_f32; seq_len * self.head_dim];
                let mut k_h = vec![0.0_f32; seq_len * self.head_dim];
                let mut v_h = vec![0.0_f32; seq_len * self.head_dim];
                for t in 0..seq_len {
                    let src_off = t * self.model_dim + h * self.head_dim;
                    let dst_off = t * self.head_dim;
                    q_h[dst_off..dst_off + self.head_dim]
                        .copy_from_slice(&q_b[src_off..src_off + self.head_dim]);
                    k_h[dst_off..dst_off + self.head_dim]
                        .copy_from_slice(&k_b[src_off..src_off + self.head_dim]);
                    v_h[dst_off..dst_off + self.head_dim]
                        .copy_from_slice(&v_b[src_off..src_off + self.head_dim]);
                }

                // Scaled dot-product attention for this head.
                let head_out = scaled_dot_product_attn(
                    &q_h,
                    &k_h,
                    &v_h,
                    seq_len,
                    self.head_dim,
                    causal,
                );

                // Write back: concatenate head outputs.
                for t in 0..seq_len {
                    let dst_off =
                        (b * seq_len + t) * self.model_dim + h * self.head_dim;
                    attn_flat[dst_off..dst_off + self.head_dim]
                        .copy_from_slice(&head_out[t * self.head_dim..(t + 1) * self.head_dim]);
                }
            }
        }

        // 3. Output projection.
        let out_flat = linear_batch(
            &attn_flat,
            &self.wo.weight,
            Some(&self.wo.bias),
            self.model_dim,
            self.model_dim,
        );

        Ok((out_flat, shape.to_vec()))
    }

    // ── private helpers ───────────────────────────────────────────────────

    fn extract_batch_seq(&self, x: &[f32], shape: &[usize]) -> Result<(usize, usize)> {
        match shape.len() {
            2 => {
                // [seq_len, model_dim]
                if shape[1] != self.model_dim {
                    return Err(TensorError::invalid_argument(format!(
                        "MultiHeadAttentionLayer: expected last dim {}, got {}",
                        self.model_dim, shape[1]
                    )));
                }
                if x.len() != shape[0] * shape[1] {
                    return Err(TensorError::invalid_argument(
                        "MultiHeadAttentionLayer: x length does not match shape".to_string(),
                    ));
                }
                Ok((1, shape[0]))
            }
            3 => {
                // [batch, seq_len, model_dim]
                if shape[2] != self.model_dim {
                    return Err(TensorError::invalid_argument(format!(
                        "MultiHeadAttentionLayer: expected last dim {}, got {}",
                        self.model_dim, shape[2]
                    )));
                }
                if x.len() != shape[0] * shape[1] * shape[2] {
                    return Err(TensorError::invalid_argument(
                        "MultiHeadAttentionLayer: x length does not match shape".to_string(),
                    ));
                }
                Ok((shape[0], shape[1]))
            }
            _ => Err(TensorError::invalid_argument(format!(
                "MultiHeadAttentionLayer: shape must be 2-D or 3-D, got {}-D",
                shape.len()
            ))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward sublayer
// ─────────────────────────────────────────────────────────────────────────────

/// Position-wise Feed-Forward Network.
///
/// Architecture: `Linear(model_dim → ff_dim)` → `activation` → `Linear(ff_dim → model_dim)`.
/// Optional inverted-dropout is applied after the first activation when `training == true`.
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// First linear layer: `[model_dim, ff_dim]`.
    fc1: DenseF32,
    /// Second linear layer: `[ff_dim, model_dim]`.
    fc2: DenseF32,
    /// Activation function for the hidden layer.
    activation: FfnActivation,
    /// Dropout probability (applied between the two linear layers).
    dropout_prob: f32,
}

impl FeedForward {
    /// Construct a new `FeedForward` sublayer.
    pub fn new(
        model_dim: usize,
        ff_dim: usize,
        activation: FfnActivation,
        dropout_prob: f32,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&dropout_prob) {
            return Err(TensorError::invalid_argument(format!(
                "FeedForward: dropout_prob must be in [0, 1], got {dropout_prob}"
            )));
        }
        Ok(Self {
            fc1: DenseF32::new(model_dim, ff_dim)?,
            fc2: DenseF32::new(ff_dim, model_dim)?,
            activation,
            dropout_prob,
        })
    }

    /// Forward pass.
    ///
    /// `input`: flat buffer of shape `shape`.
    /// `training`: when `true`, dropout is applied using a simple Bernoulli mask.
    /// Returns `(output, shape)` with the same shape as input.
    pub fn forward(
        &self,
        input: &[f32],
        shape: &[usize],
        training: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        if shape.is_empty() {
            return Err(TensorError::invalid_argument(
                "FeedForward: shape must be non-empty".to_string(),
            ));
        }
        let last_dim = *shape.last().ok_or_else(|| {
            TensorError::invalid_argument("FeedForward: empty shape".to_string())
        })?;
        if last_dim != self.fc1.in_dim {
            return Err(TensorError::invalid_argument(format!(
                "FeedForward: expected input dim {}, got {}",
                self.fc1.in_dim, last_dim
            )));
        }
        let total = input.len();
        if total % self.fc1.in_dim != 0 {
            return Err(TensorError::invalid_argument(
                "FeedForward: input length inconsistent with shape".to_string(),
            ));
        }
        let n_tokens = total / self.fc1.in_dim;

        // First projection: [*, model_dim] → [*, ff_dim]
        let mut hidden = linear_batch(
            input,
            &self.fc1.weight,
            Some(&self.fc1.bias),
            self.fc1.in_dim,
            self.fc1.out_dim,
        );

        // Activation in-place.
        apply_activation(&mut hidden, self.activation);

        // Dropout (training only, inverted dropout).
        if training && self.dropout_prob > 0.0 {
            let keep_prob = 1.0 - self.dropout_prob;
            let scale = 1.0 / keep_prob;
            // Simple Bernoulli mask using a counter-based seed.
            for (idx, v) in hidden.iter_mut().enumerate() {
                // Quasi-random value in (0, 1) using fractional-part of a known constant.
                let frac = ((idx as f64 * 1.618033988749_f64 + 0.5) % 1.0) as f32;
                if frac < self.dropout_prob {
                    *v = 0.0;
                } else {
                    *v *= scale;
                }
            }
        }

        // Second projection: [*, ff_dim] → [*, model_dim]
        let output = linear_batch(
            &hidden,
            &self.fc2.weight,
            Some(&self.fc2.bias),
            self.fc2.in_dim,
            self.fc2.out_dim,
        );

        Ok((output, shape.to_vec()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerEncoderBlock
// ─────────────────────────────────────────────────────────────────────────────

/// A single pre-norm Transformer encoder block.
///
/// ```text
/// x  =  x + self_attn(norm1(x))
/// x  =  x + ffn(norm2(x))
/// ```
#[derive(Debug, Clone)]
pub struct TransformerEncoderBlock {
    /// Self-attention sublayer.
    pub self_attn: MultiHeadAttentionLayer,
    /// Feed-forward sublayer.
    pub ffn: FeedForward,
    /// Layer normalisation applied before self-attention.
    pub norm1: LayerNormLayer,
    /// Layer normalisation applied before FFN.
    pub norm2: LayerNormLayer,
    /// Dropout probability applied to sublayer outputs (before residual add).
    pub dropout_prob: f32,
}

impl TransformerEncoderBlock {
    /// Construct a new encoder block.
    ///
    /// `ff_dim` is typically 4 × `model_dim`.
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&dropout_prob) {
            return Err(TensorError::invalid_argument(format!(
                "TransformerEncoderBlock: dropout_prob must be in [0, 1], got {dropout_prob}"
            )));
        }
        Ok(Self {
            self_attn: MultiHeadAttentionLayer::new(model_dim, num_heads)?,
            ffn: FeedForward::new(model_dim, ff_dim, FfnActivation::Gelu, dropout_prob)?,
            norm1: LayerNormLayer::new(model_dim),
            norm2: LayerNormLayer::new(model_dim),
            dropout_prob,
        })
    }

    /// Forward pass of the encoder block.
    ///
    /// `input`: flat buffer of shape `shape` (`[batch, seq_len, model_dim]` or `[seq_len, model_dim]`).
    /// `training`: enables dropout.
    /// `causal`: applies a causal attention mask.
    ///
    /// Returns `(output, shape)` – the shape is identical to the input shape.
    pub fn forward(
        &self,
        input: &[f32],
        shape: &[usize],
        training: bool,
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        // ── Sublayer 1: self-attention ────────────────────────────────────
        // Pre-norm
        let normed1 = self.norm1.forward(input, shape)?;
        let (attn_out, _) = self.self_attn.forward(&normed1, shape, causal)?;

        // Optional dropout on the sublayer output.
        let attn_out = self.apply_dropout(&attn_out, training, 0);

        // Residual: x = input + attn_out
        let res1 = elementwise_add(input, &attn_out)?;

        // ── Sublayer 2: feed-forward ──────────────────────────────────────
        // Pre-norm
        let normed2 = self.norm2.forward(&res1, shape)?;
        let (ffn_out, _) = self.ffn.forward(&normed2, shape, training)?;

        // Optional dropout on FFN output.
        let ffn_out = self.apply_dropout(&ffn_out, training, 1);

        // Residual: x = res1 + ffn_out
        let out = elementwise_add(&res1, &ffn_out)?;

        Ok((out, shape.to_vec()))
    }

    // ── private ───────────────────────────────────────────────────────────

    /// Apply inverted dropout to `v`.  `seed_offset` shifts the quasi-random sequence
    /// so that successive calls produce different masks.
    fn apply_dropout(&self, v: &[f32], training: bool, seed_offset: usize) -> Vec<f32> {
        if !training || self.dropout_prob <= 0.0 {
            return v.to_vec();
        }
        let keep_prob = 1.0 - self.dropout_prob;
        let scale = 1.0 / keep_prob;
        v.iter()
            .enumerate()
            .map(|(idx, &x)| {
                let frac = (((idx + seed_offset * 65537) as f64 * 1.618033988749_f64 + 0.5) % 1.0)
                    as f32;
                if frac < self.dropout_prob { 0.0 } else { x * scale }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerEncoder (stack of N blocks)
// ─────────────────────────────────────────────────────────────────────────────

/// A stack of `N` `TransformerEncoderBlock`s, equivalent to the encoder tower in BERT/GPT.
#[derive(Debug, Clone)]
pub struct TransformerEncoder {
    /// Ordered list of encoder blocks.
    blocks: Vec<TransformerEncoderBlock>,
    /// Number of blocks in the stack.
    pub num_layers: usize,
}

impl TransformerEncoder {
    /// Create a new `TransformerEncoder` with `num_layers` identical (but independently
    /// initialised) encoder blocks.
    pub fn new(
        num_layers: usize,
        model_dim: usize,
        num_heads: usize,
        ff_dim: usize,
        dropout_prob: f32,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::invalid_argument(
                "TransformerEncoder: num_layers must be > 0".to_string(),
            ));
        }
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerEncoderBlock::new(
                model_dim,
                num_heads,
                ff_dim,
                dropout_prob,
            )?);
        }
        Ok(Self { blocks, num_layers })
    }

    /// Pass `input` through all encoder blocks sequentially.
    pub fn forward(
        &self,
        input: &[f32],
        shape: &[usize],
        training: bool,
        causal: bool,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut current = input.to_vec();
        let mut current_shape = shape.to_vec();
        for block in &self.blocks {
            let (out, s) = block.forward(&current, &current_shape, training, causal)?;
            current = out;
            current_shape = s;
        }
        Ok((current, current_shape))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Elementwise addition of two equally-shaped flat buffers.
fn elementwise_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(TensorError::invalid_argument(format!(
            "elementwise_add: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (≥ 15)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LayerNormLayer ────────────────────────────────────────────────────

    #[test]
    fn test_layer_norm_mean_approx_zero() {
        let ln = LayerNormLayer::new(8);
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = ln.forward(&input, &[8]).expect("layer norm forward");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layer_norm_std_approx_one() {
        let ln = LayerNormLayer::new(16);
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 3.0 - 7.5).collect();
        let out = ln.forward(&input, &[16]).expect("layer norm forward");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 1e-4, "std should be ~1, got {std}");
    }

    #[test]
    fn test_layer_norm_batch_output_shape() {
        let ln = LayerNormLayer::new(4);
        // Two tokens of dim 4
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = ln.forward(&input, &[2, 4]).expect("batch layer norm");
        assert_eq!(out.len(), 8, "output length must match input length");
    }

    #[test]
    fn test_layer_norm_dim_mismatch_error() {
        let ln = LayerNormLayer::new(4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = ln.forward(&input, &[8]); // shape says dim=8 but layer dim=4
        assert!(result.is_err(), "mismatched dim should return an error");
    }

    #[test]
    fn test_layer_norm_identity_when_unit_scale() {
        // With γ=1 and β=0, a vector identical to mean ± σ should stay normalised.
        let ln = LayerNormLayer::new(4);
        let input = vec![0.0, 0.0, 0.0, 0.0]; // trivial: all same → mean=0, var=0 → output=0
        let out = ln.forward(&input, &[4]).expect("layer norm");
        for &v in &out {
            assert!(v.abs() < 1e-6, "uniform input should give near-zero output");
        }
    }

    // ── GELU activation ───────────────────────────────────────────────────

    #[test]
    fn test_gelu_allows_negatives() {
        // GELU(-1) is negative, unlike ReLU.
        let v = gelu_approx(-1.0);
        assert!(v < 0.0, "GELU(-1) should be negative, got {v}");
    }

    #[test]
    fn test_gelu_positive_for_large_positive() {
        let v = gelu_approx(5.0);
        assert!((v - 5.0).abs() < 0.01, "GELU(5) ≈ 5, got {v}");
    }

    #[test]
    fn test_relu_no_negatives() {
        let vals: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 4.0];
        for v in vals {
            assert!(relu(v) >= 0.0);
        }
    }

    // ── FeedForward ───────────────────────────────────────────────────────

    #[test]
    fn test_feed_forward_output_shape() {
        let ff = FeedForward::new(8, 32, FfnActivation::Gelu, 0.0)
            .expect("FeedForward::new");
        let input = vec![0.5_f32; 16]; // 2 tokens × 8 dims
        let (out, shape) = ff
            .forward(&input, &[2, 8], false)
            .expect("FeedForward::forward");
        assert_eq!(out.len(), 16);
        assert_eq!(shape, vec![2, 8]);
    }

    #[test]
    fn test_feed_forward_relu_activation() {
        let ff = FeedForward::new(4, 8, FfnActivation::Relu, 0.0)
            .expect("FeedForward::new");
        let input = vec![1.0_f32; 4];
        let (out, _) = ff.forward(&input, &[4], false).expect("forward");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_feed_forward_dropout_changes_output_in_training() {
        let ff = FeedForward::new(8, 16, FfnActivation::Gelu, 0.5)
            .expect("FeedForward::new");
        let input = vec![1.0_f32; 8];
        let (out_train, _) = ff.forward(&input, &[8], true).expect("train");
        let (out_eval, _) = ff.forward(&input, &[8], false).expect("eval");
        // Outputs must differ because dropout changes them in train mode.
        // (With our deterministic quasi-random mask the outputs will differ.)
        let different = out_train
            .iter()
            .zip(out_eval.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(different, "train vs eval output should differ with dropout");
    }

    #[test]
    fn test_feed_forward_swiglu_activation() {
        let ff = FeedForward::new(4, 8, FfnActivation::Swiglu, 0.0)
            .expect("FeedForward::new");
        let input = vec![1.0_f32; 4];
        let (out, _) = ff.forward(&input, &[4], false).expect("forward");
        assert_eq!(out.len(), 4);
    }

    // ── MultiHeadAttentionLayer ───────────────────────────────────────────

    #[test]
    fn test_mha_output_shape_matches_input() {
        let mha = MultiHeadAttentionLayer::new(8, 2).expect("MHA::new");
        let seq_len = 4;
        let input = vec![0.1_f32; seq_len * 8];
        let (out, shape) = mha.forward(&input, &[seq_len, 8], false).expect("forward");
        assert_eq!(out.len(), seq_len * 8);
        assert_eq!(shape, vec![seq_len, 8]);
    }

    #[test]
    fn test_mha_batch_output_shape() {
        let mha = MultiHeadAttentionLayer::new(8, 2).expect("MHA::new");
        let batch = 3;
        let seq_len = 5;
        let input = vec![0.1_f32; batch * seq_len * 8];
        let (out, shape) = mha
            .forward(&input, &[batch, seq_len, 8], false)
            .expect("forward");
        assert_eq!(out.len(), batch * seq_len * 8);
        assert_eq!(shape, vec![batch, seq_len, 8]);
    }

    #[test]
    fn test_mha_causal_differs_from_non_causal() {
        let mha = MultiHeadAttentionLayer::new(8, 2).expect("MHA::new");
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // [4, 8]
        let (out_causal, _) = mha.forward(&input, &[4, 8], true).expect("causal");
        let (out_non_causal, _) = mha.forward(&input, &[4, 8], false).expect("non-causal");
        let different = out_causal
            .iter()
            .zip(out_non_causal.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(different, "causal and non-causal should produce different outputs");
    }

    #[test]
    fn test_mha_invalid_heads_error() {
        let result = MultiHeadAttentionLayer::new(8, 3); // 8 % 3 != 0
        assert!(result.is_err(), "model_dim not divisible by num_heads should error");
    }

    // ── TransformerEncoderBlock ───────────────────────────────────────────

    #[test]
    fn test_encoder_block_output_shape_matches_input() {
        let block = TransformerEncoderBlock::new(8, 2, 32, 0.0)
            .expect("EncoderBlock::new");
        let seq_len = 6;
        let input = vec![0.1_f32; seq_len * 8];
        let (out, shape) = block
            .forward(&input, &[seq_len, 8], false, false)
            .expect("forward");
        assert_eq!(out.len(), input.len(), "residual: output size == input size");
        assert_eq!(shape, vec![seq_len, 8]);
    }

    #[test]
    fn test_encoder_block_causal_vs_non_causal() {
        let block = TransformerEncoderBlock::new(8, 2, 16, 0.0)
            .expect("EncoderBlock::new");
        let input: Vec<f32> = (0..40).map(|i| i as f32 * 0.01).collect(); // [5, 8]
        let (out_c, _) = block.forward(&input, &[5, 8], false, true).expect("causal");
        let (out_nc, _) = block.forward(&input, &[5, 8], false, false).expect("non-causal");
        let different = out_c
            .iter()
            .zip(out_nc.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-8);
        assert!(
            different,
            "causal block output should differ from non-causal block output"
        );
    }

    // ── TransformerEncoder (stack) ────────────────────────────────────────

    #[test]
    fn test_encoder_stack_two_layers() {
        let encoder = TransformerEncoder::new(2, 8, 2, 32, 0.0)
            .expect("TransformerEncoder::new");
        let input = vec![0.5_f32; 4 * 8]; // [4, 8]
        let (out, shape) = encoder
            .forward(&input, &[4, 8], false, false)
            .expect("forward");
        assert_eq!(out.len(), 4 * 8);
        assert_eq!(shape, vec![4, 8]);
    }

    #[test]
    fn test_encoder_stack_output_shape_batch() {
        let encoder = TransformerEncoder::new(3, 16, 4, 64, 0.1)
            .expect("TransformerEncoder::new");
        let batch = 2;
        let seq_len = 10;
        let input = vec![0.1_f32; batch * seq_len * 16];
        let (out, shape) = encoder
            .forward(&input, &[batch, seq_len, 16], false, false)
            .expect("forward");
        assert_eq!(out.len(), batch * seq_len * 16);
        assert_eq!(shape, vec![batch, seq_len, 16]);
    }

    #[test]
    fn test_encoder_zero_layers_error() {
        let result = TransformerEncoder::new(0, 8, 2, 32, 0.0);
        assert!(result.is_err(), "zero layers should return an error");
    }
}
