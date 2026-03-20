//! Protein Language Models — Round 45 Track A.
//!
//! ESM-2 / MSA-Transformer-style protein language models in pure Rust.
//!
//! # Sections
//! - §1  [`PlmTokenizer`]        — IUPAC amino-acid vocabulary with special tokens
//! - §2  [`PlmEmbedding`]        — token + learned positional embeddings (Xavier init)
//! - §3  [`PlmLayerNorm`]        — Pre-LN layer normalisation (as in ESM-2)
//! - §4  [`PlmMultiHeadAttention`] — MHA with Rotary Position Embeddings (RoPE)
//! - §5  [`PlmFeedForward`]      — SwiGLU feed-forward network
//! - §6  [`PlmTransformerBlock`] — Pre-LN transformer block
//! - §7  [`PlmEncoder`]          — Full ESM-2-style protein encoder
//! - §8  [`PlmContactPredictor`] — Attention-based residue–residue contact prediction
//! - §9  [`PlmFitnessPredictor`] — Zero-shot fitness prediction (masked marginal)
//! - §10 [`PlmMaskedLMLoss`]     — Masked language modelling loss
//! - §11 [`PlmMsaEncoder`]       — MSA-Transformer row + column attention block
//! - §12 [`PlmMetrics`]          — Perplexity, contact precision, sequence recovery

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════════════════

/// Errors produced by the protein-LM module.
#[derive(Debug, Clone)]
pub enum PlmError {
    /// The sequence contains a character not in the vocabulary.
    InvalidSequence(String),
    /// Incompatible tensor dimensions.
    DimensionMismatch(String),
    /// A numerical computation failed.
    ComputationError(String),
}

impl fmt::Display for PlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlmError::InvalidSequence(s) => write!(f, "PlmError::InvalidSequence: {s}"),
            PlmError::DimensionMismatch(s) => write!(f, "PlmError::DimensionMismatch: {s}"),
            PlmError::ComputationError(s) => write!(f, "PlmError::ComputationError: {s}"),
        }
    }
}

impl std::error::Error for PlmError {}

// ═══════════════════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Sigmoid: 1 / (1 + e^{-x})
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// SiLU / Swish: x * sigmoid(x)
#[inline]
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Numerically stable softmax over a slice.
fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return vec![];
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let s = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|e| e / s).collect()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Matrix–vector multiply: `(m × n) · n → m`.
fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

/// Matrix multiply: `(m × k) · (k × n) → (m × n)`.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i][p] * b[p][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Transpose a matrix.
fn transpose(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if mat.is_empty() {
        return vec![];
    }
    let rows = mat.len();
    let cols = mat[0].len();
    let mut out = vec![vec![0.0_f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = mat[i][j];
        }
    }
    out
}

/// Xavier (Glorot) initialisation standard deviation: sqrt(2 / (fan_in + fan_out)).
fn xavier_std(fan_in: usize, fan_out: usize) -> f64 {
    (2.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Allocate a matrix with Xavier-uniform values using the provided RNG.
fn xavier_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let std = xavier_std(rows, cols);
    // Approximate normal via Box-Muller.
    let mut mat = vec![vec![0.0_f64; cols]; rows];
    for row in mat.iter_mut() {
        for v in row.iter_mut() {
            let u1: f64 = rng.random::<f64>().max(1e-300);
            let u2: f64 = rng.random::<f64>();
            let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *v = n * std;
        }
    }
    mat
}

/// Uniform-random matrix in [-limit, limit].
fn uniform_matrix(rows: usize, cols: usize, limit: f64, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0_f64; cols]; rows];
    for row in mat.iter_mut() {
        for v in row.iter_mut() {
            *v = (rng.random::<f64>() * 2.0 - 1.0) * limit;
        }
    }
    mat
}

/// Element-wise add of two `[T × D]` matrices.
fn add_2d(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x + y).collect())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// §1  PlmTokenizer
// ═══════════════════════════════════════════════════════════════════════════════

/// Special token identifiers.
pub const PLM_PAD: usize = 0;
pub const PLM_MASK: usize = 1;
pub const PLM_CLS: usize = 2;
pub const PLM_EOS: usize = 3;

/// Tokeniser for protein sequences using the standard 20 amino-acid alphabet
/// plus IUPAC ambiguity codes and special tokens (PAD / MASK / CLS / EOS).
///
/// Token layout:
/// - 0 : PAD
/// - 1 : MASK
/// - 2 : CLS
/// - 3 : EOS
/// - 4–23: 20 canonical amino acids (A C D E F G H I K L M N P Q R S T V W Y)
/// - 24–28: IUPAC ambiguity (B Z X J U)
#[derive(Debug, Clone)]
pub struct PlmTokenizer {
    /// Character → token-id mapping.
    pub vocab: HashMap<char, usize>,
    /// Token-id → character mapping.
    pub reverse_vocab: HashMap<usize, char>,
    /// Total vocabulary size.
    pub vocab_size: usize,
}

impl PlmTokenizer {
    /// Build the default protein tokeniser.
    pub fn new() -> Self {
        let mut vocab: HashMap<char, usize> = HashMap::new();
        let mut reverse_vocab: HashMap<usize, char> = HashMap::new();

        // Special tokens.
        for (sym, id) in [('<', 0usize), ('!', 1), ('[', 2), (']', 3)] {
            vocab.insert(sym, id);
            reverse_vocab.insert(id, sym);
        }

        // 20 canonical amino acids in alphabetical order.
        let canonical = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'Y',
        ];
        for (i, &aa) in canonical.iter().enumerate() {
            let id = 4 + i;
            vocab.insert(aa, id);
            reverse_vocab.insert(id, aa);
        }

        // IUPAC ambiguity codes.
        let ambig = ['B', 'Z', 'X', 'J', 'U'];
        for (i, &aa) in ambig.iter().enumerate() {
            let id = 24 + i;
            vocab.insert(aa, id);
            reverse_vocab.insert(id, aa);
        }

        let vocab_size = reverse_vocab.len();
        PlmTokenizer {
            vocab,
            reverse_vocab,
            vocab_size,
        }
    }

    /// Encode a raw protein sequence string into token ids.
    ///
    /// CLS is prepended and EOS is appended automatically.
    pub fn encode(&self, seq: &str) -> Result<Vec<usize>, PlmError> {
        let mut tokens = Vec::with_capacity(seq.len() + 2);
        tokens.push(PLM_CLS);
        for ch in seq.chars() {
            let upper = ch.to_ascii_uppercase();
            match self.vocab.get(&upper) {
                Some(&id) if id >= 4 => tokens.push(id),
                _ => {
                    return Err(PlmError::InvalidSequence(format!(
                        "Character '{ch}' is not in the protein vocabulary"
                    )));
                }
            }
        }
        tokens.push(PLM_EOS);
        Ok(tokens)
    }

    /// Decode token ids back to a sequence string (strips CLS/EOS/PAD).
    pub fn decode(&self, tokens: &[usize]) -> Result<String, PlmError> {
        let mut out = String::new();
        for &id in tokens {
            if id == PLM_CLS || id == PLM_EOS || id == PLM_PAD {
                continue;
            }
            if id == PLM_MASK {
                out.push('_');
                continue;
            }
            match self.reverse_vocab.get(&id) {
                Some(&ch) => out.push(ch),
                None => {
                    return Err(PlmError::InvalidSequence(format!(
                        "Token id {id} is not in the vocabulary"
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Encode a batch of sequences into token-id lists.
    pub fn tokenize_batch(&self, seqs: &[&str]) -> Result<Vec<Vec<usize>>, PlmError> {
        seqs.iter().map(|s| self.encode(s)).collect()
    }
}

impl Default for PlmTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §2  PlmEmbedding
// ═══════════════════════════════════════════════════════════════════════════════

/// Token + learned positional embedding layer (ESM-2 style).
///
/// Both token and position embeddings are Xavier-initialised.  The forward
/// pass returns `tokens.len() × embed_dim` representations.
#[derive(Debug, Clone)]
pub struct PlmEmbedding {
    /// Token embedding matrix: `[vocab_size × embed_dim]`.
    pub token_embed: Vec<Vec<f64>>,
    /// Positional embedding matrix: `[max_len × embed_dim]`.
    pub pos_embed: Vec<Vec<f64>>,
    /// Embedding dimensionality.
    pub embed_dim: usize,
    /// Maximum supported sequence length (including CLS/EOS).
    pub max_len: usize,
}

impl PlmEmbedding {
    /// Create a new embedding layer.
    pub fn new(vocab_size: usize, embed_dim: usize, max_len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(0x504C4D5F454D4244);
        let token_embed = xavier_matrix(vocab_size, embed_dim, &mut rng);
        let pos_embed = xavier_matrix(max_len, embed_dim, &mut rng);
        PlmEmbedding {
            token_embed,
            pos_embed,
            embed_dim,
            max_len,
        }
    }

    /// Forward pass: look up token and position embeddings and sum them.
    ///
    /// Returns `[seq_len × embed_dim]`.
    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<Vec<f64>>, PlmError> {
        if tokens.len() > self.max_len {
            return Err(PlmError::DimensionMismatch(format!(
                "Sequence length {} exceeds max_len {}",
                tokens.len(),
                self.max_len
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (pos, &tok) in tokens.iter().enumerate() {
            if tok >= self.token_embed.len() {
                return Err(PlmError::InvalidSequence(format!(
                    "Token id {tok} out of vocabulary range {}",
                    self.token_embed.len()
                )));
            }
            let t_emb = &self.token_embed[tok];
            let p_emb = &self.pos_embed[pos];
            let combined: Vec<f64> = t_emb
                .iter()
                .zip(p_emb.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            out.push(combined);
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §3  PlmLayerNorm
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-LN layer normalisation as used in ESM-2.
///
/// Normalises each token vector independently: `(x − μ) / σ * γ + β`.
#[derive(Debug, Clone)]
pub struct PlmLayerNorm {
    /// Per-dimension gain parameter (initialised to 1).
    pub gamma: Vec<f64>,
    /// Per-dimension bias parameter (initialised to 0).
    pub beta: Vec<f64>,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Feature dimension.
    pub dim: usize,
}

impl PlmLayerNorm {
    /// Create a new layer-norm with identity initialisation.
    pub fn new(dim: usize) -> Self {
        PlmLayerNorm {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
            dim,
        }
    }

    /// Normalise each row of `x` (a `[seq_len × dim]` matrix).
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|v| {
                let n = v.len() as f64;
                let mean = v.iter().sum::<f64>() / n;
                let var = v.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / n;
                let std = (var + self.eps).sqrt();
                v.iter()
                    .enumerate()
                    .map(|(i, &xi)| (xi - mean) / std * self.gamma[i] + self.beta[i])
                    .collect()
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §4  PlmMultiHeadAttention  (with RoPE)
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-head self-attention with Rotary Position Embeddings (RoPE).
///
/// Unlike the sinusoidal or learned absolute PE used in GPT-2, ESM-2 uses RoPE
/// which encodes relative positions via pair-wise rotations of the query/key
/// vectors.  This avoids modifying the embedding separately and is applied
/// inside the attention computation.
#[derive(Debug, Clone)]
pub struct PlmMultiHeadAttention {
    /// Query projection: `[embed_dim × embed_dim]`.
    pub wq: Vec<Vec<f64>>,
    /// Key projection: `[embed_dim × embed_dim]`.
    pub wk: Vec<Vec<f64>>,
    /// Value projection: `[embed_dim × embed_dim]`.
    pub wv: Vec<Vec<f64>>,
    /// Output projection: `[embed_dim × embed_dim]`.
    pub wo: Vec<Vec<f64>>,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimensionality of each head (`embed_dim / n_heads`).
    pub head_dim: usize,
    /// Total embedding dimensionality.
    pub embed_dim: usize,
}

impl PlmMultiHeadAttention {
    /// Construct with Xavier-initialised projections.
    pub fn new(embed_dim: usize, n_heads: usize) -> Result<Self, PlmError> {
        if embed_dim % n_heads != 0 {
            return Err(PlmError::DimensionMismatch(format!(
                "embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
            )));
        }
        let head_dim = embed_dim / n_heads;
        let mut rng = StdRng::seed_from_u64(0x504C4D5F4154544E);
        let wq = xavier_matrix(embed_dim, embed_dim, &mut rng);
        let wk = xavier_matrix(embed_dim, embed_dim, &mut rng);
        let wv = xavier_matrix(embed_dim, embed_dim, &mut rng);
        let wo = xavier_matrix(embed_dim, embed_dim, &mut rng);
        Ok(PlmMultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            head_dim,
            embed_dim,
        })
    }

    /// Apply RoPE to a `[seq_len × head_dim]` Q or K matrix.
    ///
    /// For each position `pos` and each dimension pair `(d, d+1)`:
    /// ```text
    /// θ_d = pos / 10000^(2d / head_dim)
    /// x'_d   = x_d * cos(θ) - x_{d+1} * sin(θ)
    /// x'_{d+1} = x_d * sin(θ) + x_{d+1} * cos(θ)
    /// ```
    fn apply_rope(&self, x: &[Vec<f64>], head_dim: usize) -> Vec<Vec<f64>> {
        x.iter()
            .enumerate()
            .map(|(pos, row)| {
                let mut out = row.clone();
                let mut d = 0;
                while d + 1 < head_dim {
                    let theta = (pos as f64) / 10000_f64.powf(2.0 * d as f64 / head_dim as f64);
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let x0 = row[d];
                    let x1 = row[d + 1];
                    out[d] = x0 * cos_t - x1 * sin_t;
                    out[d + 1] = x0 * sin_t + x1 * cos_t;
                    d += 2;
                }
                out
            })
            .collect()
    }

    /// Scaled dot-product attention for a single head.
    ///
    /// Returns the attended value vectors `[seq_len × head_dim]` and the
    /// raw attention weights `[seq_len × seq_len]` for contact prediction.
    fn head_attention(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seq = q.len();
        let scale = (self.head_dim as f64).sqrt();
        let k_t = transpose(k);
        // scores: [seq × seq]
        let mut scores = matmul(q, &k_t)
            .into_iter()
            .map(|row| row.into_iter().map(|x| x / scale).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        // Apply optional boolean mask (true = masked/ignored).
        if let Some(m) = mask {
            for i in 0..seq {
                for j in 0..seq {
                    if m.get(i).and_then(|r| r.get(j)).copied().unwrap_or(false) {
                        scores[i][j] = f64::NEG_INFINITY;
                    }
                }
            }
        }
        let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();
        let out = matmul(&attn, v);
        (out, attn)
    }

    /// Multi-head self-attention forward pass.
    ///
    /// `x`    : `[seq_len × embed_dim]`
    /// `mask` : optional `[seq_len × seq_len]` boolean mask (true = mask out)
    ///
    /// Returns `[seq_len × embed_dim]`.
    pub fn forward(
        &self,
        x: &[Vec<f64>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> Result<Vec<Vec<f64>>, PlmError> {
        let seq = x.len();
        if seq == 0 {
            return Ok(vec![]);
        }

        // Compute global Q, K, V: [seq × embed_dim]
        let q_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wq, v)).collect();
        let k_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wk, v)).collect();
        let v_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wv, v)).collect();

        // Split into heads and apply RoPE to Q and K.
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            // Slice head: [seq × head_dim]
            let q_h: Vec<Vec<f64>> = q_full.iter().map(|v| v[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k_full.iter().map(|v| v[start..end].to_vec()).collect();
            let v_h: Vec<Vec<f64>> = v_full.iter().map(|v| v[start..end].to_vec()).collect();

            // Apply RoPE.
            let q_rope = self.apply_rope(&q_h, self.head_dim);
            let k_rope = self.apply_rope(&k_h, self.head_dim);

            let (out_h, _attn) = self.head_attention(&q_rope, &k_rope, &v_h, mask);
            head_outputs.push(out_h);
        }

        // Concatenate heads: [seq × embed_dim]
        let concat: Vec<Vec<f64>> = (0..seq)
            .map(|i| {
                let mut row = Vec::with_capacity(self.embed_dim);
                for h in 0..self.n_heads {
                    row.extend_from_slice(&head_outputs[h][i]);
                }
                row
            })
            .collect();

        // Output projection.
        let out: Vec<Vec<f64>> = concat.iter().map(|v| matvec(&self.wo, v)).collect();
        Ok(out)
    }

    /// Like `forward` but also returns the per-head attention maps.
    ///
    /// Returns `(output [seq × embed], attentions [n_heads × seq × seq])`.
    pub fn forward_with_attn(
        &self,
        x: &[Vec<f64>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>), PlmError> {
        let seq = x.len();
        if seq == 0 {
            return Ok((vec![], vec![]));
        }

        let q_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wq, v)).collect();
        let k_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wk, v)).collect();
        let v_full: Vec<Vec<f64>> = x.iter().map(|v| matvec(&self.wv, v)).collect();

        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        let mut all_attn: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);

        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            let q_h: Vec<Vec<f64>> = q_full.iter().map(|v| v[start..end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k_full.iter().map(|v| v[start..end].to_vec()).collect();
            let v_h: Vec<Vec<f64>> = v_full.iter().map(|v| v[start..end].to_vec()).collect();
            let q_rope = self.apply_rope(&q_h, self.head_dim);
            let k_rope = self.apply_rope(&k_h, self.head_dim);
            let (out_h, attn_h) = self.head_attention(&q_rope, &k_rope, &v_h, mask);
            head_outputs.push(out_h);
            all_attn.push(attn_h);
        }

        let concat: Vec<Vec<f64>> = (0..seq)
            .map(|i| {
                let mut row = Vec::with_capacity(self.embed_dim);
                for h in 0..self.n_heads {
                    row.extend_from_slice(&head_outputs[h][i]);
                }
                row
            })
            .collect();
        let out: Vec<Vec<f64>> = concat.iter().map(|v| matvec(&self.wo, v)).collect();
        Ok((out, all_attn))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §5  PlmFeedForward  (SwiGLU)
// ═══════════════════════════════════════════════════════════════════════════════

/// SwiGLU feed-forward network as used in ESM-2 / LLaMA.
///
/// Architecture (Liu et al., 2023):
/// ```text
/// gate = W1 · x          (embed_dim → ff_dim)
/// up   = W2 · x          (embed_dim → ff_dim)
/// y    = W3 · (silu(gate) ⊙ up)   (ff_dim → embed_dim)
/// ```
#[derive(Debug, Clone)]
pub struct PlmFeedForward {
    /// Gate projection: `[ff_dim × embed_dim]`.
    pub w1: Vec<Vec<f64>>,
    /// Up projection: `[ff_dim × embed_dim]`.
    pub w2: Vec<Vec<f64>>,
    /// Down projection: `[embed_dim × ff_dim]`.
    pub w3: Vec<Vec<f64>>,
    /// Input / output dimensionality.
    pub embed_dim: usize,
    /// Hidden (intermediate) dimensionality.
    pub ff_dim: usize,
}

impl PlmFeedForward {
    /// Create a new SwiGLU FFN.  `expansion` controls `ff_dim = expansion * embed_dim`.
    pub fn new(embed_dim: usize, expansion: usize) -> Self {
        let ff_dim = expansion * embed_dim;
        let mut rng = StdRng::seed_from_u64(0x504C4D5F46460000);
        let w1 = xavier_matrix(ff_dim, embed_dim, &mut rng);
        let w2 = xavier_matrix(ff_dim, embed_dim, &mut rng);
        let w3 = xavier_matrix(embed_dim, ff_dim, &mut rng);
        PlmFeedForward {
            w1,
            w2,
            w3,
            embed_dim,
            ff_dim,
        }
    }

    /// SiLU activation (Swish-1): x · σ(x).
    #[inline]
    fn silu(&self, x: f64) -> f64 {
        silu(x)
    }

    /// Forward pass over a token sequence `[seq_len × embed_dim]`.
    pub fn forward(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|v| {
                // gate and up projections
                let gate: Vec<f64> = matvec(&self.w1, v);
                let up: Vec<f64> = matvec(&self.w2, v);
                // element-wise silu(gate) * up
                let gated: Vec<f64> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&g, &u)| self.silu(g) * u)
                    .collect();
                // down projection
                matvec(&self.w3, &gated)
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §6  PlmTransformerBlock  (Pre-LN)
// ═══════════════════════════════════════════════════════════════════════════════

/// One Pre-LN transformer block: LayerNorm → Attention → residual,
/// then LayerNorm → FFN → residual.
#[derive(Debug, Clone)]
pub struct PlmTransformerBlock {
    /// Multi-head self-attention sub-layer.
    pub attn: PlmMultiHeadAttention,
    /// SwiGLU feed-forward sub-layer.
    pub ff: PlmFeedForward,
    /// Layer norm before attention.
    pub ln1: PlmLayerNorm,
    /// Layer norm before FFN.
    pub ln2: PlmLayerNorm,
}

impl PlmTransformerBlock {
    /// Construct a transformer block.
    pub fn new(embed_dim: usize, n_heads: usize, ff_expansion: usize) -> Result<Self, PlmError> {
        Ok(PlmTransformerBlock {
            attn: PlmMultiHeadAttention::new(embed_dim, n_heads)?,
            ff: PlmFeedForward::new(embed_dim, ff_expansion),
            ln1: PlmLayerNorm::new(embed_dim),
            ln2: PlmLayerNorm::new(embed_dim),
        })
    }

    /// Forward pass.
    ///
    /// `x`    : `[seq_len × embed_dim]`
    /// `mask` : optional attention mask
    ///
    /// Returns `[seq_len × embed_dim]`.
    pub fn forward(
        &self,
        x: &[Vec<f64>],
        mask: Option<&Vec<Vec<bool>>>,
    ) -> Result<Vec<Vec<f64>>, PlmError> {
        // Pre-LN attention sub-layer.
        let normed1 = self.ln1.forward(x);
        let attn_out = self.attn.forward(&normed1, mask)?;
        let after_attn = add_2d(x, &attn_out);

        // Pre-LN FFN sub-layer.
        let normed2 = self.ln2.forward(&after_attn);
        let ff_out = self.ff.forward(&normed2);
        let after_ff = add_2d(&after_attn, &ff_out);

        Ok(after_ff)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §7  PlmEncoder
// ═══════════════════════════════════════════════════════════════════════════════

/// Full ESM-2-style protein language model encoder.
///
/// Architecture:
/// 1. Token + positional embedding
/// 2. Stack of `n_layers` Pre-LN transformer blocks with RoPE
/// 3. Final layer normalisation
#[derive(Debug, Clone)]
pub struct PlmEncoder {
    /// Token + positional embedding layer.
    pub embedding: PlmEmbedding,
    /// Stack of transformer blocks.
    pub layers: Vec<PlmTransformerBlock>,
    /// Final layer normalisation.
    pub final_ln: PlmLayerNorm,
    /// Embedding dimensionality.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
}

impl PlmEncoder {
    /// Construct the encoder.
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        n_layers: usize,
        n_heads: usize,
    ) -> Result<Self, PlmError> {
        let embedding = PlmEmbedding::new(vocab_size, embed_dim, 1024);
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(PlmTransformerBlock::new(embed_dim, n_heads, 4)?);
        }
        let final_ln = PlmLayerNorm::new(embed_dim);
        Ok(PlmEncoder {
            embedding,
            layers,
            final_ln,
            embed_dim,
            n_layers,
        })
    }

    /// Run the encoder on a token sequence.
    ///
    /// Returns per-token embeddings `[seq_len × embed_dim]`.
    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<Vec<f64>>, PlmError> {
        let mut h = self.embedding.forward(tokens)?;
        for layer in &self.layers {
            h = layer.forward(&h, None)?;
        }
        h = self.final_ln.forward(&h);
        Ok(h)
    }

    /// Convenience: encode a raw sequence string using the provided tokeniser.
    pub fn embed_sequence(
        &self,
        seq: &str,
        tokenizer: &PlmTokenizer,
    ) -> Result<Vec<Vec<f64>>, PlmError> {
        let tokens = tokenizer.encode(seq)?;
        self.forward(&tokens)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §8  PlmContactPredictor
// ═══════════════════════════════════════════════════════════════════════════════

/// Residue–residue contact predictor from attention maps (Rao et al., 2020).
///
/// Averages all attention heads across all layers, symmetrises the map,
/// applies Average Product Correction (APC) to remove covariation due to
/// alignment depth, then projects to contact probabilities via a linear head.
#[derive(Debug, Clone)]
pub struct PlmContactPredictor {
    /// The underlying encoder (used to extract attention weights).
    pub encoder: PlmEncoder,
    /// Linear projection on symmetrised APC-corrected attention: `[1 × n_layers*n_heads]`.
    pub contact_head: Vec<Vec<f64>>,
    /// Embedding dimensionality.
    pub embed_dim: usize,
}

impl PlmContactPredictor {
    /// Build the contact predictor.
    pub fn new(embed_dim: usize, n_layers: usize, n_heads: usize) -> Result<Self, PlmError> {
        let vocab_size = 29; // standard PlmTokenizer size
        let encoder = PlmEncoder::new(vocab_size, embed_dim, n_layers, n_heads)?;
        // contact_head: 1 × (n_layers * n_heads) — learns which heads are informative
        let total_heads = n_layers * n_heads;
        let mut rng = StdRng::seed_from_u64(0x504C4D5F434F4E54);
        let contact_head = uniform_matrix(1, total_heads, 0.1, &mut rng);
        Ok(PlmContactPredictor {
            encoder,
            contact_head,
            embed_dim,
        })
    }

    /// Apply Average Product Correction to a symmetric attention map.
    ///
    /// `APC_{ij} = raw_{ij} − (row_mean_i × col_mean_j) / global_mean`
    fn apply_apc(&self, attn: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let l = attn.len();
        if l == 0 {
            return vec![];
        }
        // Row means.
        let row_mean: Vec<f64> = attn
            .iter()
            .map(|r| r.iter().sum::<f64>() / l as f64)
            .collect();
        // Col means (same as row_mean for symmetric matrix, but compute properly).
        let col_mean: Vec<f64> = (0..l)
            .map(|j| attn.iter().map(|r| r[j]).sum::<f64>() / l as f64)
            .collect();
        let global_mean = row_mean.iter().sum::<f64>() / l as f64;
        let denom = global_mean.max(1e-10);
        let mut out = vec![vec![0.0; l]; l];
        for i in 0..l {
            for j in 0..l {
                out[i][j] = attn[i][j] - row_mean[i] * col_mean[j] / denom;
            }
        }
        out
    }

    /// Predict an L×L contact probability matrix from a protein sequence.
    ///
    /// Only residue positions (excluding CLS/EOS tokens) are returned.
    pub fn predict_contacts(
        &self,
        seq: &str,
        tokenizer: &PlmTokenizer,
    ) -> Result<Vec<Vec<f64>>, PlmError> {
        let tokens = tokenizer.encode(seq)?;
        let seq_len = tokens.len(); // includes CLS and EOS
        let res_len = seq_len - 2; // residue positions only

        // Collect per-layer per-head attention maps.
        let mut layer_head_attns: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut h = self.encoder.embedding.forward(&tokens)?;
        for layer in &self.encoder.layers {
            let normed = layer.ln1.forward(&h);
            let (_out, heads) = layer.attn.forward_with_attn(&normed, None)?;
            h = layer.forward(&h, None)?;
            for head_attn in heads {
                // Extract the residue submatrix (rows 1..seq_len-1, cols 1..seq_len-1).
                let sub: Vec<Vec<f64>> = (1..=res_len)
                    .map(|i| (1..=res_len).map(|j| head_attn[i][j]).collect())
                    .collect();
                layer_head_attns.push(sub);
            }
        }

        let n_heads_total = layer_head_attns.len();
        if n_heads_total == 0 || res_len == 0 {
            return Ok(vec![vec![0.0; res_len]; res_len]);
        }

        // Average attention across heads, then symmetrize.
        let mut avg_attn = vec![vec![0.0_f64; res_len]; res_len];
        for attn in &layer_head_attns {
            for i in 0..res_len {
                for j in 0..res_len {
                    avg_attn[i][j] += attn[i][j] / n_heads_total as f64;
                }
            }
        }
        // Symmetrize: A_sym = (A + A^T) / 2
        let mut sym = vec![vec![0.0_f64; res_len]; res_len];
        for i in 0..res_len {
            for j in 0..res_len {
                sym[i][j] = (avg_attn[i][j] + avg_attn[j][i]) / 2.0;
            }
        }

        // Apply APC.
        let apc = self.apply_apc(&sym);

        // Clamp and apply sigmoid to get contact probabilities.
        let probs: Vec<Vec<f64>> = apc
            .iter()
            .map(|row| row.iter().map(|&v| sigmoid(v)).collect())
            .collect();
        Ok(probs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §9  PlmFitnessPredictor
// ═══════════════════════════════════════════════════════════════════════════════

/// Zero-shot fitness predictor via masked marginal likelihood (Meier et al., 2021).
///
/// For each mutated position, masks that position in the encoder input, then
/// computes the log-probability of the mutant amino acid minus the log-probability
/// of the wildtype amino acid.  The final score is the sum over all mutant positions.
///
/// Positive score → mutation is predicted beneficial.
#[derive(Debug, Clone)]
pub struct PlmFitnessPredictor {
    /// The underlying protein language model encoder.
    pub encoder: PlmEncoder,
    /// Language-model head: `[vocab_size × embed_dim]`.
    pub lm_head: Vec<Vec<f64>>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl PlmFitnessPredictor {
    /// Construct the fitness predictor.
    pub fn new(
        embed_dim: usize,
        n_layers: usize,
        n_heads: usize,
        vocab_size: usize,
    ) -> Result<Self, PlmError> {
        let encoder = PlmEncoder::new(vocab_size, embed_dim, n_layers, n_heads)?;
        let mut rng = StdRng::seed_from_u64(0x504C4D5F46495400);
        let lm_head = xavier_matrix(vocab_size, embed_dim, &mut rng);
        Ok(PlmFitnessPredictor {
            encoder,
            lm_head,
            vocab_size,
        })
    }

    /// Log-softmax of a logit vector.
    fn log_softmax(&self, logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = logits.iter().map(|&x| x - max).collect();
        let log_sum_exp = shifted.iter().map(|&x| x.exp()).sum::<f64>().ln();
        shifted.iter().map(|&x| x - log_sum_exp).collect()
    }

    /// Score a single mutant sequence relative to a wildtype.
    ///
    /// Returns `Σ_mut [log P(mut_aa | mask_pos) - log P(wt_aa | mask_pos)]`.
    pub fn score_mutation(
        &self,
        wt_seq: &str,
        mutant_seq: &str,
        tokenizer: &PlmTokenizer,
    ) -> Result<f64, PlmError> {
        let wt_chars: Vec<char> = wt_seq.chars().collect();
        let mut_chars: Vec<char> = mutant_seq.chars().collect();
        if wt_chars.len() != mut_chars.len() {
            return Err(PlmError::DimensionMismatch(
                "Wildtype and mutant sequences must have the same length".into(),
            ));
        }

        // Find all mutant positions.
        let mut_positions: Vec<usize> = wt_chars
            .iter()
            .zip(mut_chars.iter())
            .enumerate()
            .filter(|(_, (wt, mt))| wt != mt)
            .map(|(i, _)| i)
            .collect();

        let wt_tokens = tokenizer.encode(wt_seq)?;
        let mt_tokens = tokenizer.encode(mutant_seq)?;

        let mut total_score = 0.0_f64;
        for &pos in &mut_positions {
            // Token position in encoded sequence (offset by 1 for CLS).
            let tok_pos = pos + 1;

            // Mask that position in the wildtype sequence.
            let mut masked_tokens = wt_tokens.clone();
            masked_tokens[tok_pos] = PLM_MASK;

            // Encode the masked sequence.
            let hidden = self.encoder.forward(&masked_tokens)?;
            let h_pos = &hidden[tok_pos];

            // Apply the LM head to get logits over vocabulary.
            let logits: Vec<f64> = self.lm_head.iter().map(|row| dot(row, h_pos)).collect();
            let log_probs = self.log_softmax(&logits);

            let wt_tok = wt_tokens[tok_pos];
            let mt_tok = mt_tokens[tok_pos];

            if wt_tok >= log_probs.len() || mt_tok >= log_probs.len() {
                return Err(PlmError::InvalidSequence(format!(
                    "Token id out of vocabulary range at position {pos}"
                )));
            }

            total_score += log_probs[mt_tok] - log_probs[wt_tok];
        }

        Ok(total_score)
    }

    /// Score multiple mutants relative to a wildtype sequence.
    pub fn score_batch(
        &self,
        wt_seq: &str,
        mutants: &[&str],
        tokenizer: &PlmTokenizer,
    ) -> Result<Vec<f64>, PlmError> {
        mutants
            .iter()
            .map(|m| self.score_mutation(wt_seq, m, tokenizer))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §10  PlmMaskedLMLoss
// ═══════════════════════════════════════════════════════════════════════════════

/// Masked Language Modelling (MLM) loss as used to train ESM-style models.
///
/// Randomly masks `mask_ratio` fraction of amino-acid tokens, then computes
/// cross-entropy loss only at those positions.
#[derive(Debug, Clone)]
pub struct PlmMaskedLMLoss {
    /// Fraction of tokens to mask (default 0.15 as in BERT).
    pub mask_ratio: f64,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl PlmMaskedLMLoss {
    /// Create a new MLM loss.
    pub fn new(mask_ratio: f64, vocab_size: usize) -> Self {
        PlmMaskedLMLoss {
            mask_ratio,
            vocab_size,
        }
    }

    /// Create a masked version of the token sequence.
    ///
    /// Returns `(masked_tokens, masked_positions)` where `masked_positions` is
    /// a `Vec<(position, original_token_id)>`.
    ///
    /// Only amino-acid tokens (id ≥ 4) are candidates for masking.
    pub fn create_mask(
        &self,
        tokens: &[usize],
        mask_token_id: usize,
    ) -> (Vec<usize>, Vec<(usize, usize)>) {
        let mut rng = StdRng::seed_from_u64(0x504C4D5F4D41534B);
        let mut masked = tokens.to_vec();
        let mut positions: Vec<(usize, usize)> = Vec::new();

        for (i, &tok) in tokens.iter().enumerate() {
            // Only mask actual amino-acid tokens.
            if tok < 4 {
                continue;
            }
            let r: f64 = rng.random();
            if r < self.mask_ratio {
                positions.push((i, tok));
                masked[i] = mask_token_id;
            }
        }
        (masked, positions)
    }

    /// Compute the MLM cross-entropy loss at the masked positions.
    ///
    /// `logits`          : `[seq_len × vocab_size]`
    /// `masked_positions`: `[(position, true_token_id)]`
    pub fn compute_loss(&self, logits: &[Vec<f64>], masked_positions: &[(usize, usize)]) -> f64 {
        if masked_positions.is_empty() {
            return 0.0;
        }
        let mut total = 0.0_f64;
        for &(pos, true_tok) in masked_positions {
            if pos >= logits.len() {
                continue;
            }
            let row = &logits[pos];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = row.iter().map(|&x| (x - max).exp()).sum::<f64>().ln() + max;
            let log_p = if true_tok < row.len() {
                row[true_tok] - log_sum_exp
            } else {
                f64::NEG_INFINITY
            };
            total -= log_p;
        }
        total / masked_positions.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §11  PlmMsaEncoder  (MSA-Transformer style)
// ═══════════════════════════════════════════════════════════════════════════════

/// One block of the MSA-Transformer encoder (Rao et al., 2021).
///
/// Alternates between:
/// - **Row attention**: standard multi-head self-attention across the alignment
///   positions *within each sequence*.
/// - **Column attention**: multi-head self-attention across *sequences* at each
///   alignment column (captures evolutionary co-variation).
#[derive(Debug, Clone)]
pub struct PlmMsaEncoder {
    /// Row-wise multi-head attention (operates on alignment columns per row).
    pub row_attn: PlmMultiHeadAttention,
    /// Column-wise multi-head attention (operates on sequences per column).
    pub col_attn: PlmMultiHeadAttention,
    /// Feed-forward sub-layer.
    pub ff: PlmFeedForward,
    /// Layer norm before row attention.
    pub ln1: PlmLayerNorm,
    /// Layer norm before column attention.
    pub ln2: PlmLayerNorm,
    /// Layer norm before FFN.
    pub ln3: PlmLayerNorm,
}

impl PlmMsaEncoder {
    /// Construct an MSA encoder block.
    pub fn new(embed_dim: usize, n_heads: usize) -> Result<Self, PlmError> {
        Ok(PlmMsaEncoder {
            row_attn: PlmMultiHeadAttention::new(embed_dim, n_heads)?,
            col_attn: PlmMultiHeadAttention::new(embed_dim, n_heads)?,
            ff: PlmFeedForward::new(embed_dim, 4),
            ln1: PlmLayerNorm::new(embed_dim),
            ln2: PlmLayerNorm::new(embed_dim),
            ln3: PlmLayerNorm::new(embed_dim),
        })
    }

    /// Forward pass over a multiple sequence alignment.
    ///
    /// `msa_embeddings` : `[n_seqs × seq_len × embed_dim]`
    /// Returns           : `[n_seqs × seq_len × embed_dim]`
    pub fn forward(
        &self,
        msa_embeddings: &[Vec<Vec<f64>>],
    ) -> Result<Vec<Vec<Vec<f64>>>, PlmError> {
        let n_seqs = msa_embeddings.len();
        if n_seqs == 0 {
            return Ok(vec![]);
        }
        let seq_len = msa_embeddings[0].len();

        // ── Row attention (per-sequence, across columns) ──────────────────────
        let mut after_row: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_seqs);
        for seq_emb in msa_embeddings {
            let normed = self.ln1.forward(seq_emb);
            let attn_out = self.row_attn.forward(&normed, None)?;
            let res = add_2d(seq_emb, &attn_out);
            after_row.push(res);
        }

        // ── Column attention (per-column, across sequences) ───────────────────
        let mut after_col: Vec<Vec<Vec<f64>>> = after_row.clone();
        for col in 0..seq_len {
            // Gather the col-th token from every sequence: [n_seqs × embed_dim].
            let col_tokens: Vec<Vec<f64>> = after_row.iter().map(|s| s[col].clone()).collect();
            let normed_col = self.ln2.forward(&col_tokens);
            let col_out = self.col_attn.forward(&normed_col, None)?;
            // Scatter back with residual.
            for (s, out_vec) in col_out.iter().enumerate() {
                for (d, &v) in out_vec.iter().enumerate() {
                    after_col[s][col][d] = after_row[s][col][d] + v;
                }
            }
        }

        // ── Feed-forward (applied per sequence) ──────────────────────────────
        let mut out: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_seqs);
        for seq_emb in &after_col {
            let normed = self.ln3.forward(seq_emb);
            let ff_out = self.ff.forward(&normed);
            let res = add_2d(seq_emb, &ff_out);
            out.push(res);
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §12  PlmMetrics
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluation metrics for protein language models.
pub struct PlmMetrics;

impl PlmMetrics {
    /// Perplexity: exp(-mean log P(token)).
    ///
    /// `logits`  : `[seq_len × vocab_size]` (raw, un-normalised)
    /// `targets` : `[seq_len]` true token ids
    pub fn perplexity(logits: &[Vec<f64>], targets: &[usize]) -> f64 {
        if logits.is_empty() || targets.is_empty() {
            return f64::INFINITY;
        }
        let n = logits.len().min(targets.len());
        let mut total_nll = 0.0_f64;
        for i in 0..n {
            let row = &logits[i];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = row.iter().map(|&x| (x - max).exp()).sum::<f64>().ln() + max;
            let tok = targets[i];
            let log_p = if tok < row.len() {
                row[tok] - log_sum_exp
            } else {
                -1e10
            };
            total_nll -= log_p;
        }
        (total_nll / n as f64).exp()
    }

    /// Contact precision at L (long-range).
    ///
    /// Evaluates the precision of the top `floor(L * l_factor)` predicted
    /// contacts, restricting to pairs with `|i - j| >= 6` (standard benchmark).
    ///
    /// `contacts`     : predicted `[L × L]` probability matrix
    /// `true_contacts`: set of true contacts as `(i, j)` pairs
    /// `l_factor`     : typically 1.0 (P@L), 2.0 (P@L/2), 5.0 (P@L/5)
    pub fn contact_precision_at_l(
        contacts: &[Vec<f64>],
        true_contacts: &[(usize, usize)],
        l_factor: f64,
    ) -> f64 {
        let l = contacts.len();
        if l == 0 || true_contacts.is_empty() {
            return 0.0;
        }
        let k = ((l as f64 * l_factor).floor() as usize).max(1);

        // Collect all (i, j) with |i-j| >= 6, upper-triangle only.
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..l {
            for j in (i + 6)..l {
                candidates.push((contacts[i][j], i, j));
            }
        }
        // Sort descending by predicted probability.
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let true_set: std::collections::HashSet<(usize, usize)> = true_contacts
            .iter()
            .flat_map(|&(i, j)| {
                let (a, b) = if i <= j { (i, j) } else { (j, i) };
                std::iter::once((a, b))
            })
            .collect();

        let top_k = candidates.iter().take(k);
        let hits = top_k
            .filter(|&&(_, i, j)| {
                let (a, b) = if i <= j { (i, j) } else { (j, i) };
                true_set.contains(&(a, b))
            })
            .count();

        hits as f64 / k as f64
    }

    /// Mean Reciprocal Rank (MRR) of true contacts in the ranked predicted list.
    pub fn mean_reciprocal_rank_contacts(
        contacts: &[Vec<f64>],
        true_contacts: &[(usize, usize)],
    ) -> f64 {
        let l = contacts.len();
        if l == 0 || true_contacts.is_empty() {
            return 0.0;
        }
        // All upper-triangle pairs, sorted descending.
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..l {
            for j in (i + 1)..l {
                candidates.push((contacts[i][j], i, j));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let true_set: std::collections::HashSet<(usize, usize)> = true_contacts
            .iter()
            .flat_map(|&(i, j)| {
                let (a, b) = if i <= j { (i, j) } else { (j, i) };
                std::iter::once((a, b))
            })
            .collect();

        let mut mrr = 0.0_f64;
        for (rank, &(_, i, j)) in candidates.iter().enumerate() {
            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            if true_set.contains(&(a, b)) {
                mrr += 1.0 / (rank + 1) as f64;
            }
        }
        mrr / true_contacts.len() as f64
    }

    /// Sequence recovery: fraction of predicted tokens matching ground truth.
    pub fn sequence_recovery(predicted_tokens: &[usize], true_tokens: &[usize]) -> f64 {
        if true_tokens.is_empty() {
            return 0.0;
        }
        let n = predicted_tokens.len().min(true_tokens.len());
        let matches = predicted_tokens[..n]
            .iter()
            .zip(true_tokens[..n].iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / n as f64
    }
}
