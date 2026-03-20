//! Biological Sequence & Structure Modeling — bio_ml module.
//!
//! This module provides machine-learning primitives for computational biology:
//!
//! - **Protein Language Models**: ESM-style transformer blocks with RoPE, contact
//!   prediction heads, and full ESM-inspired architecture.
//! - **Protein Structure Prediction**: Distance matrices, secondary structure,
//!   torsion angle prediction, structure encoders, and AlphaFold-style FAPE loss.
//! - **DNA/RNA Modeling**: Nucleotide tokenizers, convolutional motif scanners,
//!   RNA folding energy, and splice site prediction.
//! - **Single-Cell Genomics**: scRNA-seq normalization, PCA via power iteration,
//!   scVI-style VAE, cell type classification, and trajectory inference.
//! - **Drug Discovery**: Fingerprint similarity, drug-target interaction, virtual
//!   screening, ADMET prediction, and molecular docking scoring.
//! - **Genomics & Survival Analysis**: scRNA-seq VAE (ZINB), genomic CNNs,
//!   survival analysis (Cox/DeepSurv/KM), and multi-omics integration (MoFa).
//!
//! All fallible operations return `Result<_, TensorError>`.
//! No `unsafe` code; no `unwrap()` calls outside tests.

pub mod extensions;
pub use extensions::*;

pub mod genomics;
pub use genomics::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Shared math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-xc).exp())
}

#[inline]
pub(crate) fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

pub(crate) fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum.abs() < 1e-300 { 1e-300 } else { sum };
    exps.iter().map(|e| e / denom).collect()
}

/// Layer norm over a vector (mean-zero, unit-variance + learned scale/shift).
pub(crate) fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| gamma[i] * (v - mean) / std + beta[i])
        .collect()
}

/// Matrix-vector product.
pub(crate) fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

pub(crate) fn vecadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Random matrix initialisation (Xavier-style).
pub(crate) fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

pub(crate) fn random_vec(n: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..n)
        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.01)
        .collect()
}

/// Box-Muller normal samples.
pub(crate) fn normal_samples_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        i += 1;
        if i < n {
            out.push(r * theta.sin());
            i += 1;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Protein Language Models
// ─────────────────────────────────────────────────────────────────────────────

/// The 20 standard amino acids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AminoAcid {
    Ala,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

impl AminoAcid {
    /// Parse from single-letter code (case-insensitive). Returns `None` for
    /// non-standard residues.
    pub fn from_char(c: char) -> Option<AminoAcid> {
        match c.to_ascii_uppercase() {
            'A' => Some(AminoAcid::Ala),
            'R' => Some(AminoAcid::Arg),
            'N' => Some(AminoAcid::Asn),
            'D' => Some(AminoAcid::Asp),
            'C' => Some(AminoAcid::Cys),
            'Q' => Some(AminoAcid::Gln),
            'E' => Some(AminoAcid::Glu),
            'G' => Some(AminoAcid::Gly),
            'H' => Some(AminoAcid::His),
            'I' => Some(AminoAcid::Ile),
            'L' => Some(AminoAcid::Leu),
            'K' => Some(AminoAcid::Lys),
            'M' => Some(AminoAcid::Met),
            'F' => Some(AminoAcid::Phe),
            'P' => Some(AminoAcid::Pro),
            'S' => Some(AminoAcid::Ser),
            'T' => Some(AminoAcid::Thr),
            'W' => Some(AminoAcid::Trp),
            'Y' => Some(AminoAcid::Tyr),
            'V' => Some(AminoAcid::Val),
            _ => None,
        }
    }

    /// 0-indexed position in the alphabet (A=0 … V=19).
    pub fn to_idx(self) -> usize {
        match self {
            AminoAcid::Ala => 0,
            AminoAcid::Arg => 1,
            AminoAcid::Asn => 2,
            AminoAcid::Asp => 3,
            AminoAcid::Cys => 4,
            AminoAcid::Gln => 5,
            AminoAcid::Glu => 6,
            AminoAcid::Gly => 7,
            AminoAcid::His => 8,
            AminoAcid::Ile => 9,
            AminoAcid::Leu => 10,
            AminoAcid::Lys => 11,
            AminoAcid::Met => 12,
            AminoAcid::Phe => 13,
            AminoAcid::Pro => 14,
            AminoAcid::Ser => 15,
            AminoAcid::Thr => 16,
            AminoAcid::Trp => 17,
            AminoAcid::Tyr => 18,
            AminoAcid::Val => 19,
        }
    }
}

/// Tokenizes amino acid sequence strings into token index vectors.
///
/// Token vocabulary:
/// - 0 = PAD
/// - 1 = MASK
/// - 2..=21 = amino acid (offset from `AminoAcid::to_idx()` by 2)
/// - Unknown characters are silently dropped.
#[derive(Debug, Clone)]
pub struct ProteinTokenizer {
    /// Maximum sequence length (longer sequences are truncated).
    pub max_length: usize,
}

impl ProteinTokenizer {
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }

    /// Tokenize a sequence string; unknown residues are skipped.
    pub fn tokenize(&self, seq: &str) -> Vec<usize> {
        seq.chars()
            .filter_map(AminoAcid::from_char)
            .map(|aa| aa.to_idx() + 2)
            .take(self.max_length)
            .collect()
    }

    /// Pad a token vector to `max_length` with PAD (0).
    pub fn pad(&self, tokens: &[usize]) -> Vec<usize> {
        let mut v = tokens.to_vec();
        v.resize(self.max_length, 0);
        v
    }
}

/// ESM-style rotary position embedding (RoPE) applied to query/key vectors.
///
/// For a head dimension `d`, pairs of dimensions are rotated by angle
/// `θ_i = pos / 10000^(2i/d)`.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Head dimensionality (must be even).
    pub dim: usize,
    /// Pre-computed cos/sin tables: `[max_len][dim/2]`.
    cos_table: Vec<Vec<f64>>,
    sin_table: Vec<Vec<f64>>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_len: usize) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(TensorError::invalid_argument_op(
                "RotaryEmbedding::new",
                "dim must be even",
            ));
        }
        let half = dim / 2;
        let mut cos_table = vec![vec![0.0_f64; half]; max_len];
        let mut sin_table = vec![vec![0.0_f64; half]; max_len];
        for pos in 0..max_len {
            for i in 0..half {
                let theta = (pos as f64) / (10000_f64).powf(2.0 * i as f64 / dim as f64);
                cos_table[pos][i] = theta.cos();
                sin_table[pos][i] = theta.sin();
            }
        }
        Ok(Self {
            dim,
            cos_table,
            sin_table,
        })
    }

    /// Apply RoPE to a single token vector at position `pos`.
    pub fn apply(&self, v: &[f64], pos: usize) -> Result<Vec<f64>> {
        if pos >= self.cos_table.len() {
            return Err(TensorError::invalid_argument_op(
                "RotaryEmbedding::apply",
                "position exceeds max_len",
            ));
        }
        let half = self.dim / 2;
        let mut out = v.to_vec();
        for i in 0..half {
            let x0 = v[2 * i];
            let x1 = v[2 * i + 1];
            let c = self.cos_table[pos][i];
            let s = self.sin_table[pos][i];
            out[2 * i] = x0 * c - x1 * s;
            out[2 * i + 1] = x0 * s + x1 * c;
        }
        Ok(out)
    }
}

/// ESM-style attention block: pre-norm + RoPE multi-head attention + FFN.
#[derive(Debug, Clone)]
pub struct EsmAttentionBlock {
    /// Model dimensionality.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// FFN inner dimensionality (typically 4 × d_model).
    pub ffn_dim: usize,
    // Attention projections: Wq, Wk, Wv, Wo
    wq: Vec<Vec<f64>>,
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    // FFN
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    // Layer norm params
    gamma1: Vec<f64>,
    beta1: Vec<f64>,
    gamma2: Vec<f64>,
    beta2: Vec<f64>,
    rope: RotaryEmbedding,
}

impl EsmAttentionBlock {
    pub fn new(d_model: usize, n_heads: usize, ffn_dim: usize, seed: u64) -> Result<Self> {
        if d_model % n_heads != 0 {
            return Err(TensorError::invalid_argument_op(
                "EsmAttentionBlock::new",
                "d_model must be divisible by n_heads",
            ));
        }
        let head_dim = d_model / n_heads;
        if head_dim % 2 != 0 {
            return Err(TensorError::invalid_argument_op(
                "EsmAttentionBlock::new",
                "head_dim (d_model / n_heads) must be even for RoPE",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let wq = random_matrix(d_model, d_model, &mut rng);
        let wk = random_matrix(d_model, d_model, &mut rng);
        let wv = random_matrix(d_model, d_model, &mut rng);
        let wo = random_matrix(d_model, d_model, &mut rng);
        let w1 = random_matrix(ffn_dim, d_model, &mut rng);
        let b1 = random_vec(ffn_dim, &mut rng);
        let w2 = random_matrix(d_model, ffn_dim, &mut rng);
        let b2 = random_vec(d_model, &mut rng);
        let gamma1 = vec![1.0_f64; d_model];
        let beta1 = vec![0.0_f64; d_model];
        let gamma2 = vec![1.0_f64; d_model];
        let beta2 = vec![0.0_f64; d_model];
        let rope = RotaryEmbedding::new(head_dim, 512)?;
        Ok(Self {
            d_model,
            n_heads,
            ffn_dim,
            wq,
            wk,
            wv,
            wo,
            w1,
            b1,
            w2,
            b2,
            gamma1,
            beta1,
            gamma2,
            beta2,
            rope,
        })
    }

    /// Forward pass for a sequence of embeddings (L × d_model).
    /// Returns updated embeddings of the same shape.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let seq_len = x.len();
        let head_dim = self.d_model / self.n_heads;
        let scale = (head_dim as f64).sqrt();

        // Pre-norm
        let xn: Vec<Vec<f64>> = x
            .iter()
            .map(|v| layer_norm(v, &self.gamma1, &self.beta1))
            .collect();

        // Project Q, K, V
        let q: Vec<Vec<f64>> = xn.iter().map(|v| matvec(&self.wq, v)).collect();
        let k: Vec<Vec<f64>> = xn.iter().map(|v| matvec(&self.wk, v)).collect();
        let v: Vec<Vec<f64>> = xn.iter().map(|v| matvec(&self.wv, v)).collect();

        // Apply RoPE per head per token
        let apply_rope = |vecs: &[Vec<f64>]| -> Result<Vec<Vec<f64>>> {
            vecs.iter()
                .enumerate()
                .map(|(pos, token_vec)| {
                    let mut out = vec![0.0_f64; self.d_model];
                    for h in 0..self.n_heads {
                        let start = h * head_dim;
                        let head_slice = &token_vec[start..start + head_dim];
                        let rotated = self.rope.apply(head_slice, pos)?;
                        out[start..start + head_dim].copy_from_slice(&rotated);
                    }
                    Ok(out)
                })
                .collect()
        };

        let q_r = apply_rope(&q)?;
        let k_r = apply_rope(&k)?;

        // Compute attention per head
        let mut attn_out = vec![vec![0.0_f64; self.d_model]; seq_len];

        for h in 0..self.n_heads {
            let start = h * head_dim;
            // Attention scores [seq_len × seq_len]
            let mut scores = vec![vec![0.0_f64; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dot: f64 = q_r[i][start..start + head_dim]
                        .iter()
                        .zip(k_r[j][start..start + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[i][j] = dot / scale;
                }
            }
            // Softmax over keys
            let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

            // Weighted sum of V
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let a = attn[i][j];
                    for d in 0..head_dim {
                        attn_out[i][start + d] += a * v[j][start + d];
                    }
                }
            }
        }

        // Output projection + residual
        let mut out1: Vec<Vec<f64>> = attn_out
            .iter()
            .zip(x.iter())
            .map(|(ao, xi)| vecadd(&matvec(&self.wo, ao), xi))
            .collect();

        // FFN with pre-norm + residual
        for i in 0..seq_len {
            let xn2 = layer_norm(&out1[i], &self.gamma2, &self.beta2);
            // W1 + relu
            let h1: Vec<f64> = vecadd(&matvec(&self.w1, &xn2), &self.b1)
                .into_iter()
                .map(relu)
                .collect();
            // W2
            let h2 = vecadd(&matvec(&self.w2, &h1), &self.b2);
            out1[i] = vecadd(&out1[i], &h2);
        }

        Ok(out1)
    }
}

/// Predict pairwise residue contact map (L×L) from per-residue embeddings.
///
/// For each pair (i, j) the feature vector is `[e_i || e_j || |e_i - e_j|]`
/// fed through a two-layer MLP → sigmoid → contact probability.
#[derive(Debug, Clone)]
pub struct ContactPredictionHead {
    pub embed_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl ContactPredictionHead {
    pub fn new(embed_dim: usize, hidden: usize, seed: u64) -> Self {
        let in_dim = embed_dim * 3;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = random_matrix(hidden, in_dim, &mut rng);
        let b1 = random_vec(hidden, &mut rng);
        let w2 = random_matrix(1, hidden, &mut rng);
        let b2 = random_vec(1, &mut rng);
        Self {
            embed_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Returns an L×L contact probability matrix.
    pub fn forward(&self, embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let l = embeddings.len();
        let mut contact_map = vec![vec![0.0_f64; l]; l];
        for i in 0..l {
            for j in 0..l {
                let mut feat = Vec::with_capacity(self.embed_dim * 3);
                feat.extend_from_slice(&embeddings[i]);
                feat.extend_from_slice(&embeddings[j]);
                for k in 0..self.embed_dim {
                    feat.push((embeddings[i][k] - embeddings[j][k]).abs());
                }
                let h1: Vec<f64> = vecadd(&matvec(&self.w1, &feat), &self.b1)
                    .into_iter()
                    .map(relu)
                    .collect();
                let logit = matvec(&self.w2, &h1)[0] + self.b2[0];
                contact_map[i][j] = sigmoid(logit);
            }
        }
        contact_map
    }
}

/// Embedding lookup table (token → dense vector).
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    pub vocab_size: usize,
    pub embed_dim: usize,
    table: Vec<Vec<f64>>,
}

impl EmbeddingTable {
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let table = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random::<f64>() * 0.02 - 0.01)
                    .collect()
            })
            .collect();
        Self {
            vocab_size,
            embed_dim,
            table,
        }
    }

    pub fn embed(&self, tokens: &[usize]) -> Result<Vec<Vec<f64>>> {
        tokens
            .iter()
            .map(|&t| {
                if t >= self.vocab_size {
                    Err(TensorError::invalid_argument_op(
                        "EmbeddingTable::embed",
                        &format!("token {t} out of vocab {}", self.vocab_size),
                    ))
                } else {
                    Ok(self.table[t].clone())
                }
            })
            .collect()
    }
}

/// Simplified ESM protein language model:
/// `EmbeddingTable` + N × `EsmAttentionBlock` + linear LM head.
#[derive(Debug, Clone)]
pub struct EvolutionaryScaleModeling {
    pub n_layers: usize,
    pub d_model: usize,
    /// Vocabulary size (PAD + MASK + 20 AAs = 22).
    pub vocab_size: usize,
    embedding: EmbeddingTable,
    blocks: Vec<EsmAttentionBlock>,
    /// LM head: d_model → vocab_size
    lm_head_w: Vec<Vec<f64>>,
    lm_head_b: Vec<f64>,
}

impl EvolutionaryScaleModeling {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        ffn_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let vocab_size = 22; // PAD=0, MASK=1, AA=2..21
        let mut rng = StdRng::seed_from_u64(seed);
        let embedding = EmbeddingTable::new(vocab_size, d_model, seed);
        let blocks: Result<Vec<_>> = (0..n_layers)
            .map(|i| {
                EsmAttentionBlock::new(d_model, n_heads, ffn_dim, seed.wrapping_add(i as u64 + 1))
            })
            .collect();
        let blocks = blocks?;
        let lm_head_w = random_matrix(vocab_size, d_model, &mut rng);
        let lm_head_b = random_vec(vocab_size, &mut rng);
        Ok(Self {
            n_layers,
            d_model,
            vocab_size,
            embedding,
            blocks,
            lm_head_w,
            lm_head_b,
        })
    }

    /// Forward pass: tokens → per-token logits over vocabulary.
    pub fn forward(&self, tokens: &[usize]) -> Result<Vec<Vec<f64>>> {
        let mut x = self.embedding.embed(tokens)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let logits: Vec<Vec<f64>> = x
            .iter()
            .map(|emb| vecadd(&matvec(&self.lm_head_w, emb), &self.lm_head_b))
            .collect();
        Ok(logits)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Protein Structure Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute pairwise Cα distance matrix from 3D coordinates.
#[derive(Debug, Clone)]
pub struct DistanceMatrix;

impl DistanceMatrix {
    /// Returns an L×L symmetric matrix of Euclidean distances.
    pub fn compute(coords: &[[f64; 3]]) -> Vec<Vec<f64>> {
        let l = coords.len();
        let mut mat = vec![vec![0.0_f64; l]; l];
        for i in 0..l {
            for j in 0..l {
                if i != j {
                    let d: f64 = coords[i]
                        .iter()
                        .zip(coords[j].iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    mat[i][j] = d;
                }
            }
        }
        mat
    }
}

/// 3-class secondary structure predictor (H=0, E=1, C=2).
///
/// Uses a sliding window of residue embeddings fed into a two-layer MLP.
#[derive(Debug, Clone)]
pub struct SecondaryStructurePredictor {
    pub embed_dim: usize,
    pub window: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl SecondaryStructurePredictor {
    pub fn new(embed_dim: usize, window: usize, hidden: usize, seed: u64) -> Self {
        let in_dim = embed_dim * window;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = random_matrix(hidden, in_dim, &mut rng);
        let b1 = random_vec(hidden, &mut rng);
        let w2 = random_matrix(3, hidden, &mut rng);
        let b2 = random_vec(3, &mut rng);
        Self {
            embed_dim,
            window,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Predict class (H/E/C) for each residue. Pads boundaries with zero vectors.
    pub fn predict(&self, embeddings: &[Vec<f64>]) -> Vec<usize> {
        let l = embeddings.len();
        let pad = vec![0.0_f64; self.embed_dim];
        let half = self.window / 2;

        (0..l)
            .map(|i| {
                let mut feat = Vec::with_capacity(self.embed_dim * self.window);
                for w in 0..self.window {
                    let idx = i + w;
                    let offset_idx = idx.saturating_sub(half);
                    if offset_idx < l {
                        feat.extend_from_slice(&embeddings[offset_idx]);
                    } else {
                        feat.extend_from_slice(&pad);
                    }
                }
                let h1: Vec<f64> = vecadd(&matvec(&self.w1, &feat), &self.b1)
                    .into_iter()
                    .map(relu)
                    .collect();
                let logits = vecadd(&matvec(&self.w2, &h1), &self.b2);
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(2)
            })
            .collect()
    }
}

/// Predict φ and ψ dihedral angles per residue.
///
/// Output shape: `L × 2` (φ in column 0, ψ in column 1), values in [−π, π].
#[derive(Debug, Clone)]
pub struct TorsionAnglePredictor {
    pub embed_dim: usize,
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl TorsionAnglePredictor {
    pub fn new(embed_dim: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = random_matrix(hidden, embed_dim, &mut rng);
        let b1 = random_vec(hidden, &mut rng);
        let w2 = random_matrix(2, hidden, &mut rng);
        let b2 = random_vec(2, &mut rng);
        Self {
            embed_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Returns L×2 matrix with (φ, ψ) per residue in radians [−π, π].
    pub fn predict(&self, embeddings: &[Vec<f64>]) -> Vec<[f64; 2]> {
        embeddings
            .iter()
            .map(|emb| {
                let h1: Vec<f64> = vecadd(&matvec(&self.w1, emb), &self.b1)
                    .into_iter()
                    .map(relu)
                    .collect();
                let out = vecadd(&matvec(&self.w2, &h1), &self.b2);
                // Map to [-π, π] via tanh × π
                [
                    tanh_act(out[0]) * std::f64::consts::PI,
                    tanh_act(out[1]) * std::f64::consts::PI,
                ]
            })
            .collect()
    }
}

/// Encode 3D backbone (N, Cα, C, O positions) as invariant inter-atom distance features.
///
/// For each residue we compute pairwise distances among the 4 backbone atoms,
/// yielding 6 values (N-Cα, N-C, N-O, Cα-C, Cα-O, C-O).
#[derive(Debug, Clone)]
pub struct StructureEncoder;

/// Per-residue backbone atom coordinates.
#[derive(Debug, Clone)]
pub struct BackboneAtoms {
    pub n: [f64; 3],
    pub ca: [f64; 3],
    pub c: [f64; 3],
    pub o: [f64; 3],
}

pub(crate) fn atom_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

impl StructureEncoder {
    /// Returns L×6 invariant distance features per residue.
    pub fn encode(residues: &[BackboneAtoms]) -> Vec<[f64; 6]> {
        residues
            .iter()
            .map(|r| {
                [
                    atom_dist(&r.n, &r.ca),
                    atom_dist(&r.n, &r.c),
                    atom_dist(&r.n, &r.o),
                    atom_dist(&r.ca, &r.c),
                    atom_dist(&r.ca, &r.o),
                    atom_dist(&r.c, &r.o),
                ]
            })
            .collect()
    }
}

/// Frame Aligned Point Error (FAPE) loss used in AlphaFold2.
///
/// Simplified implementation: for each residue, construct a local rigid frame
/// from (N, Cα, C) coordinates, transform all other Cα atoms into that frame,
/// then average L2 errors clamped at `d_clamp`.
#[derive(Debug, Clone)]
pub struct AlphaFoldLoss {
    pub d_clamp: f64,
}

impl AlphaFoldLoss {
    pub fn new(d_clamp: f64) -> Self {
        Self { d_clamp }
    }

    /// Build an orthonormal local frame from three points (origin=Cα, x=N-Cα, z=N×C).
    fn build_frame(n: &[f64; 3], ca: &[f64; 3], c: &[f64; 3]) -> [[f64; 3]; 3] {
        // x axis: Cα → N direction
        let mut x = [n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]];
        let xlen = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2))
            .sqrt()
            .max(1e-8);
        x[0] /= xlen;
        x[1] /= xlen;
        x[2] /= xlen;

        // z axis: cross(x, Cα→C)
        let d = [c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]];
        let mut z = [
            x[1] * d[2] - x[2] * d[1],
            x[2] * d[0] - x[0] * d[2],
            x[0] * d[1] - x[1] * d[0],
        ];
        let zlen = (z[0].powi(2) + z[1].powi(2) + z[2].powi(2))
            .sqrt()
            .max(1e-8);
        z[0] /= zlen;
        z[1] /= zlen;
        z[2] /= zlen;

        // y axis: cross(z, x)
        let y = [
            z[1] * x[2] - z[2] * x[1],
            z[2] * x[0] - z[0] * x[2],
            z[0] * x[1] - z[1] * x[0],
        ];
        [x, y, z]
    }

    fn transform(frame_rot: &[[f64; 3]; 3], origin: &[f64; 3], pt: &[f64; 3]) -> [f64; 3] {
        let d = [pt[0] - origin[0], pt[1] - origin[1], pt[2] - origin[2]];
        [
            frame_rot[0][0] * d[0] + frame_rot[0][1] * d[1] + frame_rot[0][2] * d[2],
            frame_rot[1][0] * d[0] + frame_rot[1][1] * d[1] + frame_rot[1][2] * d[2],
            frame_rot[2][0] * d[0] + frame_rot[2][1] * d[1] + frame_rot[2][2] * d[2],
        ]
    }

    fn point_dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
    }

    /// Compute FAPE loss between predicted and true backbone coordinates.
    /// Both inputs are `L` residues, each with `BackboneAtoms`.
    pub fn compute(&self, pred: &[BackboneAtoms], true_coords: &[BackboneAtoms]) -> Result<f64> {
        let l = pred.len();
        if l != true_coords.len() {
            return Err(TensorError::invalid_argument_op(
                "AlphaFoldLoss::compute",
                "pred and true_coords must have same length",
            ));
        }
        if l == 0 {
            return Ok(0.0);
        }

        let mut total = 0.0_f64;
        let mut count = 0_usize;

        for i in 0..l {
            let pred_frame = Self::build_frame(&pred[i].n, &pred[i].ca, &pred[i].c);
            let true_frame =
                Self::build_frame(&true_coords[i].n, &true_coords[i].ca, &true_coords[i].c);

            for j in 0..l {
                let p_local = Self::transform(&pred_frame, &pred[i].ca, &pred[j].ca);
                let t_local = Self::transform(&true_frame, &true_coords[i].ca, &true_coords[j].ca);
                let err = Self::point_dist(&p_local, &t_local).min(self.d_clamp);
                total += err;
                count += 1;
            }
        }

        Ok(total / count.max(1) as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. DNA/RNA Modeling
// ─────────────────────────────────────────────────────────────────────────────

/// Tokenize DNA/RNA sequences.
/// Vocabulary: A=0, C=1, G=2, T/U=3, N=4, PAD=5.
#[derive(Debug, Clone)]
pub struct NucleotideTokenizer {
    pub include_unknown: bool,
}

impl NucleotideTokenizer {
    pub fn new() -> Self {
        Self {
            include_unknown: true,
        }
    }

    pub fn tokenize(&self, seq: &str) -> Vec<usize> {
        seq.chars()
            .map(|c| match c.to_ascii_uppercase() {
                'A' => 0,
                'C' => 1,
                'G' => 2,
                'T' | 'U' => 3,
                _ => 4, // N or unknown
            })
            .collect()
    }
}

impl Default for NucleotideTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// One-hot or learned embedding for nucleotides.
#[derive(Debug, Clone)]
pub struct DnaEmbedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    /// If `use_one_hot=true`, ignore `table` and return one-hot vectors.
    pub use_one_hot: bool,
    table: Vec<Vec<f64>>,
}

impl DnaEmbedding {
    pub fn new_one_hot(vocab_size: usize) -> Self {
        let table: Vec<Vec<f64>> = (0..vocab_size)
            .map(|i| {
                let mut v = vec![0.0_f64; vocab_size];
                v[i] = 1.0;
                v
            })
            .collect();
        Self {
            vocab_size,
            embed_dim: vocab_size,
            use_one_hot: true,
            table,
        }
    }

    pub fn new_learned(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let table = (0..vocab_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.random::<f64>() * 0.02 - 0.01)
                    .collect()
            })
            .collect();
        Self {
            vocab_size,
            embed_dim,
            use_one_hot: false,
            table,
        }
    }

    pub fn embed(&self, tokens: &[usize]) -> Result<Vec<Vec<f64>>> {
        tokens
            .iter()
            .map(|&t| {
                if t >= self.vocab_size {
                    Err(TensorError::invalid_argument_op(
                        "DnaEmbedding::embed",
                        &format!("token {t} out of vocab {}", self.vocab_size),
                    ))
                } else {
                    Ok(self.table[t].clone())
                }
            })
            .collect()
    }
}

/// 1D convolutional motif scanner.
///
/// Slides a learned kernel of shape `[kernel_size × embed_dim]` over the
/// embedded sequence and returns per-position activation scores.
#[derive(Debug, Clone)]
pub struct ConvolutionalMotifScanner {
    pub kernel_size: usize,
    pub embed_dim: usize,
    pub n_filters: usize,
    /// Kernels: `[n_filters][kernel_size × embed_dim]`
    kernels: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl ConvolutionalMotifScanner {
    pub fn new(kernel_size: usize, embed_dim: usize, n_filters: usize, seed: u64) -> Self {
        let k_len = kernel_size * embed_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let kernels = (0..n_filters)
            .map(|_| {
                (0..k_len)
                    .map(|_| rng.random::<f64>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();
        let biases = random_vec(n_filters, &mut rng);
        Self {
            kernel_size,
            embed_dim,
            n_filters,
            kernels,
            biases,
        }
    }

    /// Scan a sequence: returns `[L - kernel_size + 1][n_filters]` activations.
    pub fn scan(&self, embeddings: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let l = embeddings.len();
        if l < self.kernel_size {
            return Err(TensorError::invalid_argument_op(
                "ConvolutionalMotifScanner::scan",
                "sequence shorter than kernel_size",
            ));
        }
        let out_len = l - self.kernel_size + 1;
        let k_len = self.kernel_size * self.embed_dim;

        let mut out = vec![vec![0.0_f64; self.n_filters]; out_len];
        for pos in 0..out_len {
            let mut flat = Vec::with_capacity(k_len);
            for w in 0..self.kernel_size {
                flat.extend_from_slice(&embeddings[pos + w]);
            }
            for f in 0..self.n_filters {
                let val: f64 = flat
                    .iter()
                    .zip(self.kernels[f].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                out[pos][f] = relu(val + self.biases[f]);
            }
        }
        Ok(out)
    }
}

/// RNA folding energy estimator.
///
/// Counts valid base pairs under a simple greedy stack model, scoring:
/// - A-U and U-A: −2 kcal/mol
/// - G-C and C-G: −3 kcal/mol
/// - G-U and U-G (wobble): −1 kcal/mol
#[derive(Debug, Clone)]
pub struct RnaFoldingScore;

impl RnaFoldingScore {
    /// Approximate MFE by greedy outer-inner bracket matching.
    /// Returns a negative score (more negative = more stable).
    pub fn mfe_approx(sequence: &str) -> f64 {
        let seq: Vec<char> = sequence
            .chars()
            .map(|c| {
                if c == 'T' {
                    'U'
                } else {
                    c.to_ascii_uppercase()
                }
            })
            .collect();
        let l = seq.len();
        let pair_score = |a: char, b: char| -> f64 {
            match (a, b) {
                ('A', 'U') | ('U', 'A') => -2.0,
                ('G', 'C') | ('C', 'G') => -3.0,
                ('G', 'U') | ('U', 'G') => -1.0,
                _ => 0.0,
            }
        };

        // Simple greedy: scan from outside inwards
        let mut total = 0.0_f64;
        let mut lo = 0usize;
        let mut hi = if l > 0 { l - 1 } else { 0 };
        while lo + 3 < hi {
            let s = pair_score(seq[lo], seq[hi]);
            if s < 0.0 {
                total += s;
                lo += 1;
                hi -= 1;
            } else {
                lo += 1;
            }
        }
        total
    }
}

/// Predict donor and acceptor splice sites using 5-mer features.
///
/// For each candidate position (all positions except last 4),
/// extract the 5-mer centered at that position and score via a linear
/// model over a 4^5=1024 dimensional one-hot feature.
#[derive(Debug, Clone)]
pub struct SpliceSitePredictor {
    /// Weights for donor prediction: 1024-dim → 1.
    donor_w: Vec<f64>,
    donor_b: f64,
    /// Weights for acceptor prediction: 1024-dim → 1.
    acceptor_w: Vec<f64>,
    acceptor_b: f64,
}

impl SpliceSitePredictor {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let donor_w = (0..1024)
            .map(|_| rng.random::<f64>() * 0.02 - 0.01)
            .collect();
        let donor_b = rng.random::<f64>() * 0.01;
        let acceptor_w = (0..1024)
            .map(|_| rng.random::<f64>() * 0.02 - 0.01)
            .collect();
        let acceptor_b = rng.random::<f64>() * 0.01;
        Self {
            donor_w,
            donor_b,
            acceptor_w,
            acceptor_b,
        }
    }

    fn kmer_index(tokens: &[usize], pos: usize) -> usize {
        let mut idx = 0usize;
        for k in 0..5 {
            idx = idx * 4 + tokens[pos + k].min(3);
        }
        idx
    }

    /// Returns `(donor_scores, acceptor_scores)` for positions `0..L-4`.
    pub fn predict(&self, tokens: &[usize]) -> Result<(Vec<f64>, Vec<f64>)> {
        let l = tokens.len();
        if l < 5 {
            return Err(TensorError::invalid_argument_op(
                "SpliceSitePredictor::predict",
                "sequence too short (need >= 5 nucleotides)",
            ));
        }
        let out_len = l - 4;
        let mut donor_scores = Vec::with_capacity(out_len);
        let mut acceptor_scores = Vec::with_capacity(out_len);
        for pos in 0..out_len {
            let ki = Self::kmer_index(tokens, pos);
            donor_scores.push(sigmoid(self.donor_w[ki] + self.donor_b));
            acceptor_scores.push(sigmoid(self.acceptor_w[ki] + self.acceptor_b));
        }
        Ok((donor_scores, acceptor_scores))
    }
}
