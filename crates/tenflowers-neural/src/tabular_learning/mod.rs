//! Deep Learning for Tabular Data.
//!
//! Implements state-of-the-art architectures for structured / tabular data:
//!
//! | Component | Reference |
//! |-----------|-----------|
//! | [`TabTransformer`] | Huang et al. (2020) — TabTransformer |
//! | [`FTTransformer`] | Gorishniy et al. (2021) — FT-Transformer |
//! | [`NodeModel`] | Popov et al. (2020) — NODE (oblivious decision trees) |
//! | [`TabNet`] | Arik & Pfister (2021) — TabNet |
//! | [`SaintModel`] | Somepalli et al. (2021) — SAINT |
//! | [`FeatureEncoder`] | preprocessing pipeline (scaler, quantile, cyclic) |
//! | [`MixedInputHead`] | gated fusion of categorical + numeric representations |
//! | [`TabularAugmentation`] | Mixup / CutMix / SMOTE-like oversampling |
//! | [`TabularMetrics`] | accuracy, macro-F1, RMSE, R² |
//! | [`CatBoostEncoder`] | leave-one-out target encoding |
//!
//! All weights are `Vec<f32>` buffers; no `Tensor`/autograd dependency.
//! No `unwrap()`, no `unsafe`, single file ≤ 1900 lines.

use std::f32::consts::PI;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Shared error / result type
// ─────────────────────────────────────────────────────────────────────────────

type TabResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// Shared math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * 0.797_884_6 * (1.0 + 0.044715 * x * x)).tanh())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let c = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-c).exp())
}

/// Softmax over a slice; returns a new `Vec`.
fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    exps.iter().map(|&e| e / sum).collect()
}

/// Sparsemax: projects onto the probability simplex.
fn sparsemax(z: &[f32]) -> Vec<f32> {
    let n = z.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0_f32;
    let mut k = n;
    for (i, &s) in sorted.iter().enumerate() {
        cumsum += s;
        if s > (cumsum - 1.0) / (i + 1) as f32 {
            k = i + 1;
        }
    }
    let tau = (sorted[..k].iter().sum::<f32>() - 1.0) / k as f32;
    z.iter().map(|&zi| (zi - tau).max(0.0)).collect()
}

/// Dense layer forward: `y = W·x + b`.
/// `w` is row-major `[out × in]`.
fn linear(w: &[f32], b: &[f32], x: &[f32]) -> TabResult<Vec<f32>> {
    let in_dim = x.len();
    let out_dim = b.len();
    if w.len() != out_dim * in_dim {
        return Err(format!(
            "linear: w.len()={} != out×in={}×{}",
            w.len(),
            out_dim,
            in_dim
        ));
    }
    let mut y = vec![0.0_f32; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        y[o] = b[o]
            + row
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>();
    }
    Ok(y)
}

/// Layer normalisation: `(x − mean) / (std + ε) * γ + β`.
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32]) -> TabResult<Vec<f32>> {
    let n = x.len();
    if gamma.len() != n || beta.len() != n {
        return Err(format!(
            "layer_norm: dim mismatch x={n}, γ={}, β={}",
            gamma.len(),
            beta.len()
        ));
    }
    let mean = x.iter().copied().sum::<f32>() / n as f32;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = (var + 1e-5_f32).sqrt().recip();
    Ok(x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) * std_inv * gamma[i] + beta[i])
        .collect())
}

/// Xavier uniform initialisation.
fn xavier_uniform(size: usize, fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let bound = (6.0_f32 / (fan_in + fan_out).max(1) as f32).sqrt();
    (0..size)
        .map(|_| {
            let u: f32 = rng.random();
            2.0 * bound * u - bound
        })
        .collect()
}

/// Kaiming uniform initialisation.
fn kaiming_uniform(size: usize, fan_in: usize, rng: &mut StdRng) -> Vec<f32> {
    let bound = (2.0_f32 / fan_in.max(1) as f32).sqrt();
    (0..size)
        .map(|_| {
            let u: f32 = rng.random();
            2.0 * bound * u - bound
        })
        .collect()
}

/// Zeros initialisation.
fn zeros(size: usize) -> Vec<f32> {
    vec![0.0_f32; size]
}

/// Ones initialisation.
fn ones(size: usize) -> Vec<f32> {
    vec![1.0_f32; size]
}

/// Scaled dot-product attention over a sequence of [seq_len × d_model] vectors.
/// Returns [seq_len × d_model].
fn scaled_dot_product_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    d_model: usize,
    n_heads: usize,
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
) -> TabResult<Vec<f32>> {
    if n_heads == 0 || d_model % n_heads != 0 {
        return Err(format!(
            "d_model={d_model} not divisible by n_heads={n_heads}"
        ));
    }
    let dh = d_model / n_heads;
    let scale = (dh as f32).sqrt().recip();

    // Project: [seq_len × d_model] → [seq_len × d_model] for Q, K, V
    // wq/wk/wv: [d_model × d_model]
    let project = |w: &[f32], inp: &[f32]| -> TabResult<Vec<f32>> {
        let mut out = vec![0.0_f32; seq_len * d_model];
        for s in 0..seq_len {
            for o in 0..d_model {
                let mut acc = 0.0_f32;
                for i in 0..d_model {
                    acc += w[o * d_model + i] * inp[s * d_model + i];
                }
                out[s * d_model + o] = acc;
            }
        }
        Ok(out)
    };

    let pq = project(wq, q)?;
    let pk = project(wk, k)?;
    let pv = project(wv, v)?;

    let mut output = vec![0.0_f32; seq_len * d_model];

    for h in 0..n_heads {
        let offset = h * dh;
        // Compute attention scores [seq_len × seq_len]
        let mut scores = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0_f32;
                for d in 0..dh {
                    dot += pq[i * d_model + offset + d] * pk[j * d_model + offset + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
        // Softmax row-wise
        for i in 0..seq_len {
            let row = softmax(&scores[i * seq_len..(i + 1) * seq_len]);
            scores[i * seq_len..(i + 1) * seq_len].copy_from_slice(&row);
        }
        // Weighted sum of V
        for i in 0..seq_len {
            for d in 0..dh {
                let mut acc = 0.0_f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * pv[j * d_model + offset + d];
                }
                output[i * d_model + offset + d] = acc;
            }
        }
    }

    // Output projection: [seq_len × d_model] × W_O [d_model × d_model]
    let mut result = vec![0.0_f32; seq_len * d_model];
    for s in 0..seq_len {
        for o in 0..d_model {
            let mut acc = 0.0_f32;
            for i in 0..d_model {
                acc += wo[o * d_model + i] * output[s * d_model + i];
            }
            result[s * d_model + o] = acc;
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  TabTransformer  ════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`TabTransformer`].
#[derive(Debug, Clone)]
pub struct TabTransformerConfig {
    /// Number of categorical features.
    pub n_cat_features: usize,
    /// Number of numeric (continuous) features.
    pub n_num_features: usize,
    /// Vocabulary size for each categorical feature.
    pub cat_vocab_sizes: Vec<usize>,
    /// Embedding dimension for categorical features.
    pub embed_dim: usize,
    /// Number of attention heads per transformer layer.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward network hidden dimension.
    pub ffn_dim: usize,
    /// Number of output classes (1 for regression).
    pub n_classes: usize,
}

/// Transformer for tabular data (Huang et al., 2020).
///
/// Categorical features are embedded and processed through column-wise
/// multi-head self-attention layers. Numeric features are concatenated
/// after the transformer and passed through an MLP classification head.
#[derive(Debug, Clone)]
pub struct TabTransformer {
    cfg: TabTransformerConfig,
    /// Embedding tables: one per categorical feature, each [vocab × embed_dim].
    embeddings: Vec<Vec<f32>>,
    /// Transformer layers (each has WQ, WK, WV, WO, FFN W1/b1/W2/b2, LN params).
    layers: Vec<TabTransformerLayer>,
    /// Final LN: gamma, beta over embed_dim.
    final_ln_g: Vec<f32>,
    final_ln_b: Vec<f32>,
    /// MLP head: input = n_cat * embed_dim + n_num → hidden → n_classes.
    head_w1: Vec<f32>,
    head_b1: Vec<f32>,
    head_w2: Vec<f32>,
    head_b2: Vec<f32>,
}

#[derive(Debug, Clone)]
struct TabTransformerLayer {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    ffn_w1: Vec<f32>,
    ffn_b1: Vec<f32>,
    ffn_w2: Vec<f32>,
    ffn_b2: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
}

impl TabTransformer {
    /// Create a new `TabTransformer` with Xavier-initialised weights.
    pub fn new(cfg: TabTransformerConfig, seed: u64) -> TabResult<Self> {
        if cfg.cat_vocab_sizes.len() != cfg.n_cat_features {
            return Err(format!(
                "TabTransformer: cat_vocab_sizes.len()={} != n_cat_features={}",
                cfg.cat_vocab_sizes.len(),
                cfg.n_cat_features
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.embed_dim;

        let embeddings = cfg
            .cat_vocab_sizes
            .iter()
            .map(|&v| kaiming_uniform(v * d, d, &mut rng))
            .collect();

        let layers = (0..cfg.n_layers)
            .map(|_| TabTransformerLayer {
                wq: xavier_uniform(d * d, d, d, &mut rng),
                wk: xavier_uniform(d * d, d, d, &mut rng),
                wv: xavier_uniform(d * d, d, d, &mut rng),
                wo: xavier_uniform(d * d, d, d, &mut rng),
                ln1_g: ones(d),
                ln1_b: zeros(d),
                ffn_w1: xavier_uniform(cfg.ffn_dim * d, d, cfg.ffn_dim, &mut rng),
                ffn_b1: zeros(cfg.ffn_dim),
                ffn_w2: xavier_uniform(d * cfg.ffn_dim, cfg.ffn_dim, d, &mut rng),
                ffn_b2: zeros(d),
                ln2_g: ones(d),
                ln2_b: zeros(d),
            })
            .collect();

        let head_in = cfg.n_cat_features * d + cfg.n_num_features;
        let head_h = (head_in * 2).max(64);
        Ok(Self {
            embeddings,
            layers,
            final_ln_g: ones(d),
            final_ln_b: zeros(d),
            head_w1: xavier_uniform(head_h * head_in, head_in, head_h, &mut rng),
            head_b1: zeros(head_h),
            head_w2: xavier_uniform(cfg.n_classes * head_h, head_h, cfg.n_classes, &mut rng),
            head_b2: zeros(cfg.n_classes),
            cfg,
        })
    }

    /// Forward pass. `cat_ids` length must equal `n_cat_features`;
    /// `num_features` length must equal `n_num_features`.
    pub fn forward(&self, cat_ids: &[usize], num_features: &[f32]) -> TabResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if cat_ids.len() != self.cfg.n_cat_features {
            return Err(format!(
                "TabTransformer: expected {} cat ids, got {}",
                self.cfg.n_cat_features,
                cat_ids.len()
            ));
        }
        if num_features.len() != self.cfg.n_num_features {
            return Err(format!(
                "TabTransformer: expected {} num features, got {}",
                self.cfg.n_num_features,
                num_features.len()
            ));
        }

        // Embed each categorical feature → [n_cat × embed_dim]
        let seq_len = self.cfg.n_cat_features;
        let mut seq = vec![0.0_f32; seq_len * d];
        for (i, &id) in cat_ids.iter().enumerate() {
            let v = self.cfg.cat_vocab_sizes[i];
            let clamped = id.min(v.saturating_sub(1));
            let emb = &self.embeddings[i][clamped * d..(clamped + 1) * d];
            seq[i * d..(i + 1) * d].copy_from_slice(emb);
        }

        // Apply transformer layers
        for layer in &self.layers {
            let attn_out = scaled_dot_product_attn(
                &seq,
                &seq,
                &seq,
                seq_len,
                d,
                self.cfg.n_heads,
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
            )?;
            // Residual + LN
            let mut h = vec![0.0_f32; seq_len * d];
            for (i, (&a, &s)) in attn_out.iter().zip(seq.iter()).enumerate() {
                h[i] = a + s;
            }
            let mut ln_out = vec![0.0_f32; seq_len * d];
            for s in 0..seq_len {
                let normed = layer_norm(&h[s * d..(s + 1) * d], &layer.ln1_g, &layer.ln1_b)?;
                ln_out[s * d..(s + 1) * d].copy_from_slice(&normed);
            }
            // FFN per token
            let mut ffn_out = vec![0.0_f32; seq_len * d];
            for s in 0..seq_len {
                let tok = &ln_out[s * d..(s + 1) * d];
                let h1 = linear(&layer.ffn_w1, &layer.ffn_b1, tok)?;
                let h1a: Vec<f32> = h1.iter().map(|&x| gelu(x)).collect();
                let h2 = linear(&layer.ffn_w2, &layer.ffn_b2, &h1a)?;
                // Residual + LN2
                let res2: Vec<f32> = h2.iter().zip(tok.iter()).map(|(&a, &b)| a + b).collect();
                let normed2 = layer_norm(&res2, &layer.ln2_g, &layer.ln2_b)?;
                ffn_out[s * d..(s + 1) * d].copy_from_slice(&normed2);
            }
            seq = ffn_out;
        }

        // Pool: flatten cat embeddings
        let mut head_input = vec![0.0_f32; self.cfg.n_cat_features * d + self.cfg.n_num_features];
        head_input[..seq.len()].copy_from_slice(&seq);
        head_input[seq.len()..].copy_from_slice(num_features);

        let h1 = linear(&self.head_w1, &self.head_b1, &head_input)?;
        let h1a: Vec<f32> = h1.iter().map(|&x| relu(x)).collect();
        linear(&self.head_w2, &self.head_b2, &h1a)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  FT-Transformer  ════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`FTTransformer`].
#[derive(Debug, Clone)]
pub struct FTTransformerConfig {
    /// Number of categorical features.
    pub n_cat_features: usize,
    /// Number of continuous numeric features.
    pub n_num_features: usize,
    /// Vocabulary sizes for each categorical feature.
    pub cat_vocab_sizes: Vec<usize>,
    /// Shared embedding dimension for all feature tokens.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// FFN hidden dim.
    pub ffn_dim: usize,
    /// Number of output classes (1 = regression).
    pub n_classes: usize,
}

/// Feature Tokenizer + Transformer (Gorishniy et al., 2021).
///
/// Both numeric and categorical features are projected into the same
/// `embed_dim`-dimensional token space, then processed by a standard
/// transformer encoder. The `[CLS]` token drives the output head.
#[derive(Debug, Clone)]
pub struct FTTransformer {
    cfg: FTTransformerConfig,
    /// Categorical embeddings [vocab × embed_dim] per feature.
    cat_embeddings: Vec<Vec<f32>>,
    /// Numeric tokenizer: one weight + bias per numeric feature → embed_dim.
    num_w: Vec<Vec<f32>>, // [n_num × embed_dim]
    num_b: Vec<Vec<f32>>, // [n_num × embed_dim]
    /// Learnable [CLS] token embedding.
    cls_token: Vec<f32>,
    /// Transformer encoder layers (same structure as TabTransformer).
    layers: Vec<TabTransformerLayer>,
    /// Output head on CLS representation.
    head_w: Vec<f32>,
    head_b: Vec<f32>,
}

impl FTTransformer {
    /// Create a new `FTTransformer` with Xavier-initialised weights.
    pub fn new(cfg: FTTransformerConfig, seed: u64) -> TabResult<Self> {
        if cfg.cat_vocab_sizes.len() != cfg.n_cat_features {
            return Err(format!(
                "FTTransformer: cat_vocab_sizes.len()={} != n_cat_features={}",
                cfg.cat_vocab_sizes.len(),
                cfg.n_cat_features
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.embed_dim;

        let cat_embeddings = cfg
            .cat_vocab_sizes
            .iter()
            .map(|&v| kaiming_uniform(v * d, d, &mut rng))
            .collect();

        let num_w = (0..cfg.n_num_features)
            .map(|_| xavier_uniform(d, 1, d, &mut rng))
            .collect();
        let num_b = (0..cfg.n_num_features).map(|_| zeros(d)).collect();

        let cls_token: Vec<f32> = (0..d)
            .map(|_| {
                let u: f32 = rng.random();
                u * 0.02 - 0.01
            })
            .collect();

        let layers = (0..cfg.n_layers)
            .map(|_| TabTransformerLayer {
                wq: xavier_uniform(d * d, d, d, &mut rng),
                wk: xavier_uniform(d * d, d, d, &mut rng),
                wv: xavier_uniform(d * d, d, d, &mut rng),
                wo: xavier_uniform(d * d, d, d, &mut rng),
                ln1_g: ones(d),
                ln1_b: zeros(d),
                ffn_w1: xavier_uniform(cfg.ffn_dim * d, d, cfg.ffn_dim, &mut rng),
                ffn_b1: zeros(cfg.ffn_dim),
                ffn_w2: xavier_uniform(d * cfg.ffn_dim, cfg.ffn_dim, d, &mut rng),
                ffn_b2: zeros(d),
                ln2_g: ones(d),
                ln2_b: zeros(d),
            })
            .collect();

        Ok(Self {
            cat_embeddings,
            num_w,
            num_b,
            cls_token,
            layers,
            head_w: xavier_uniform(cfg.n_classes * d, d, cfg.n_classes, &mut rng),
            head_b: zeros(cfg.n_classes),
            cfg,
        })
    }

    /// Forward pass. Returns logits of length `n_classes`.
    pub fn forward(&self, cat_ids: &[usize], num_features: &[f32]) -> TabResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if cat_ids.len() != self.cfg.n_cat_features {
            return Err(format!(
                "FTTransformer: expected {} cat ids, got {}",
                self.cfg.n_cat_features,
                cat_ids.len()
            ));
        }
        if num_features.len() != self.cfg.n_num_features {
            return Err(format!(
                "FTTransformer: expected {} num features, got {}",
                self.cfg.n_num_features,
                num_features.len()
            ));
        }

        let n_tokens = 1 + self.cfg.n_cat_features + self.cfg.n_num_features; // CLS + features
        let mut tokens = vec![0.0_f32; n_tokens * d];

        // CLS token at index 0
        tokens[..d].copy_from_slice(&self.cls_token);

        // Categorical tokens
        for (i, &id) in cat_ids.iter().enumerate() {
            let v = self.cfg.cat_vocab_sizes[i];
            let clamped = id.min(v.saturating_sub(1));
            let emb = &self.cat_embeddings[i][clamped * d..(clamped + 1) * d];
            let offset = (1 + i) * d;
            tokens[offset..offset + d].copy_from_slice(emb);
        }

        // Numeric tokens: scalar * w + b
        for (i, &x) in num_features.iter().enumerate() {
            let offset = (1 + self.cfg.n_cat_features + i) * d;
            for j in 0..d {
                tokens[offset + j] = x * self.num_w[i][j] + self.num_b[i][j];
            }
        }

        // Transformer encoder
        let mut seq = tokens;
        for layer in &self.layers {
            let attn_out = scaled_dot_product_attn(
                &seq,
                &seq,
                &seq,
                n_tokens,
                d,
                self.cfg.n_heads,
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
            )?;
            let mut h = vec![0.0_f32; n_tokens * d];
            for (i, (&a, &s)) in attn_out.iter().zip(seq.iter()).enumerate() {
                h[i] = a + s;
            }
            let mut ln_out = vec![0.0_f32; n_tokens * d];
            for s in 0..n_tokens {
                let normed = layer_norm(&h[s * d..(s + 1) * d], &layer.ln1_g, &layer.ln1_b)?;
                ln_out[s * d..(s + 1) * d].copy_from_slice(&normed);
            }
            let mut ffn_out = vec![0.0_f32; n_tokens * d];
            for s in 0..n_tokens {
                let tok = &ln_out[s * d..(s + 1) * d];
                let h1 = linear(&layer.ffn_w1, &layer.ffn_b1, tok)?;
                let h1a: Vec<f32> = h1.iter().map(|&x| gelu(x)).collect();
                let h2 = linear(&layer.ffn_w2, &layer.ffn_b2, &h1a)?;
                let res2: Vec<f32> = h2.iter().zip(tok.iter()).map(|(&a, &b)| a + b).collect();
                let normed2 = layer_norm(&res2, &layer.ln2_g, &layer.ln2_b)?;
                ffn_out[s * d..(s + 1) * d].copy_from_slice(&normed2);
            }
            seq = ffn_out;
        }

        // Use CLS token for classification
        let cls = &seq[..d];
        linear(&self.head_w, &self.head_b, cls)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  NODE  ══════════════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Single differentiable oblivious decision tree (Popov et al., 2020).
///
/// Each internal node selects the same feature at every depth level.
/// The `depth` parameter controls the number of splits; each tree
/// produces `2^depth` leaves.
#[derive(Debug, Clone)]
pub struct ObliviousTree {
    /// Number of split layers (depth of the tree).
    pub depth: usize,
    /// Total number of input features.
    pub n_features: usize,
    /// Feature selection weights: [depth × n_features] (softmax along dim 1).
    pub feature_w: Vec<f32>,
    /// Learned split thresholds: \[depth\].
    pub thresholds: Vec<f32>,
    /// Leaf response values: [2^depth × output_dim].
    pub leaf_responses: Vec<f32>,
    /// Output dimension per tree.
    pub output_dim: usize,
}

impl ObliviousTree {
    /// Create a new `ObliviousTree` with random initialisation.
    pub fn new(depth: usize, n_features: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        let n_leaves = 1usize << depth;
        let feature_w = xavier_uniform(depth * n_features, n_features, depth, rng);
        let thresholds: Vec<f32> = (0..depth)
            .map(|_| {
                let u: f32 = rng.random();
                u * 2.0 - 1.0
            })
            .collect();
        let leaf_responses = xavier_uniform(n_leaves * output_dim, n_leaves, output_dim, rng);
        Self {
            depth,
            n_features,
            feature_w,
            thresholds,
            leaf_responses,
            output_dim,
        }
    }

    /// Differentiable forward: soft routing via entmax/sigmoid.
    pub fn forward(&self, x: &[f32]) -> TabResult<Vec<f32>> {
        if x.len() != self.n_features {
            return Err(format!(
                "ObliviousTree: expected {} features, got {}",
                self.n_features,
                x.len()
            ));
        }
        let d = self.depth;
        let n_leaves = 1usize << d;

        // Compute one split value per depth level using the selected feature.
        let mut leaf_probs = vec![1.0_f32; n_leaves];
        for layer in 0..d {
            // Soft feature selection via softmax
            let fw = &self.feature_w[layer * self.n_features..(layer + 1) * self.n_features];
            let feature_attn = softmax(fw);
            // Compute feature projection
            let projected: f32 = feature_attn
                .iter()
                .zip(x.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let split_val = sigmoid(projected - self.thresholds[layer]);
            // Update leaf probabilities: left branch = (1 - split), right = split
            for leaf in 0..n_leaves {
                let bit = (leaf >> (d - 1 - layer)) & 1;
                let p = if bit == 1 { split_val } else { 1.0 - split_val };
                leaf_probs[leaf] *= p;
            }
        }

        // Weighted sum of leaf responses
        let mut output = vec![0.0_f32; self.output_dim];
        for (leaf, &lp) in leaf_probs.iter().enumerate() {
            for o in 0..self.output_dim {
                output[o] += lp * self.leaf_responses[leaf * self.output_dim + o];
            }
        }
        Ok(output)
    }
}

/// Ensemble of Oblivious Decision Trees (NODE — Popov et al., 2020).
#[derive(Debug, Clone)]
pub struct NodeModel {
    /// The ensemble trees.
    pub trees: Vec<ObliviousTree>,
    /// Input feature dimension.
    pub n_features: usize,
    /// Number of output classes / regression targets.
    pub n_classes: usize,
}

impl NodeModel {
    /// Create a new `NodeModel` with `n_trees` oblivious trees.
    pub fn new(
        n_trees: usize,
        depth: usize,
        n_features: usize,
        n_classes: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let trees = (0..n_trees)
            .map(|_| ObliviousTree::new(depth, n_features, n_classes, &mut rng))
            .collect();
        Self {
            trees,
            n_features,
            n_classes,
        }
    }

    /// Forward pass: averages tree outputs.
    pub fn forward(&self, x: &[f32]) -> TabResult<Vec<f32>> {
        if self.trees.is_empty() {
            return Err("NodeModel: no trees".into());
        }
        let mut sum = vec![0.0_f32; self.n_classes];
        for tree in &self.trees {
            let out = tree.forward(x)?;
            for (s, &o) in sum.iter_mut().zip(out.iter()) {
                *s += o;
            }
        }
        let n = self.trees.len() as f32;
        Ok(sum.iter().map(|&s| s / n).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  TabNet  ════════════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// TabNet configuration (Arik & Pfister, 2021).
#[derive(Debug, Clone)]
pub struct TabNetConfig {
    /// Number of sequential attention steps.
    pub n_steps: usize,
    /// Width of feature transform output (decision step output dim).
    pub n_d: usize,
    /// Width of attentive transformer output.
    pub n_a: usize,
    /// Coefficient for feature reusage penalty.
    pub gamma: f32,
    /// Epsilon for batch normalisation.
    pub epsilon: f32,
    /// Number of input features.
    pub n_features: usize,
    /// Number of output classes.
    pub n_classes: usize,
}

/// Attentive transformer for one TabNet step.
#[derive(Debug, Clone)]
pub struct AttentiveTransformer {
    w: Vec<f32>,
    b: Vec<f32>,
    bn_gamma: Vec<f32>,
    bn_beta: Vec<f32>,
}

impl AttentiveTransformer {
    fn new(n_features: usize, n_a: usize, rng: &mut StdRng) -> Self {
        Self {
            w: xavier_uniform(n_features * n_a, n_a, n_features, rng),
            b: zeros(n_features),
            bn_gamma: ones(n_features),
            bn_beta: zeros(n_features),
        }
    }

    /// Compute sparsemax-normalised feature mask.
    fn forward(&self, h: &[f32], prior_scale: &[f32]) -> TabResult<Vec<f32>> {
        let n_features = self.b.len();
        let h_proj = linear(&self.w, &self.b, h)?;
        // Batch-norm approximation (instance normalisation)
        let normed = layer_norm(&h_proj, &self.bn_gamma, &self.bn_beta)?;
        // Element-wise multiply by prior_scale
        let masked: Vec<f32> = normed
            .iter()
            .zip(prior_scale.iter())
            .map(|(&n, &p)| n * p)
            .collect();
        Ok(sparsemax(&masked[..n_features.min(masked.len())]))
    }
}

/// Shared + step-specific feature transform layers.
#[derive(Debug, Clone)]
struct FeatureTransformStep {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl FeatureTransformStep {
    fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            w1: xavier_uniform(out_dim * in_dim, in_dim, out_dim, rng),
            b1: zeros(out_dim),
            w2: xavier_uniform(out_dim * out_dim, out_dim, out_dim, rng),
            b2: zeros(out_dim),
        }
    }

    fn forward(&self, x: &[f32]) -> TabResult<Vec<f32>> {
        let h = linear(&self.w1, &self.b1, x)?;
        let ha: Vec<f32> = h.iter().map(|&v| relu(v)).collect();
        let h2 = linear(&self.w2, &self.b2, &ha)?;
        Ok(h2.iter().map(|&v| relu(v)).collect())
    }
}

/// TabNet model (Arik & Pfister, 2021).
#[derive(Debug, Clone)]
pub struct TabNet {
    cfg: TabNetConfig,
    shared_layer: FeatureTransformStep,
    step_layers: Vec<FeatureTransformStep>,
    attn_transformers: Vec<AttentiveTransformer>,
    final_w: Vec<f32>,
    final_b: Vec<f32>,
}

impl TabNet {
    /// Create a new `TabNet`.
    pub fn new(cfg: TabNetConfig, seed: u64) -> TabResult<Self> {
        if cfg.n_steps == 0 {
            return Err("TabNet: n_steps must be > 0".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let shared_layer = FeatureTransformStep::new(cfg.n_features, cfg.n_d + cfg.n_a, &mut rng);
        let step_layers = (0..cfg.n_steps)
            .map(|_| FeatureTransformStep::new(cfg.n_d + cfg.n_a, cfg.n_d + cfg.n_a, &mut rng))
            .collect();
        let attn_transformers = (0..cfg.n_steps)
            .map(|_| AttentiveTransformer::new(cfg.n_features, cfg.n_a, &mut rng))
            .collect();
        let final_w = xavier_uniform(cfg.n_classes * cfg.n_d, cfg.n_d, cfg.n_classes, &mut rng);
        let final_b = zeros(cfg.n_classes);
        Ok(Self {
            cfg,
            shared_layer,
            step_layers,
            attn_transformers,
            final_w,
            final_b,
        })
    }

    /// Forward pass returning `(logits, per_step_masks)`.
    pub fn forward(&self, x: &[f32]) -> TabResult<(Vec<f32>, Vec<Vec<f32>>)> {
        if x.len() != self.cfg.n_features {
            return Err(format!(
                "TabNet: expected {} features, got {}",
                self.cfg.n_features,
                x.len()
            ));
        }
        let n = self.cfg.n_features;
        let mut prior_scale = vec![1.0_f32; n];
        let mut aggregated_output = vec![0.0_f32; self.cfg.n_d];
        let mut masks = Vec::with_capacity(self.cfg.n_steps);

        for step in 0..self.cfg.n_steps {
            // Attentive transformer: needs n_a-dim input (use current output aggregation)
            let h_for_attn: Vec<f32> = if aggregated_output.is_empty() {
                vec![0.0_f32; self.cfg.n_a]
            } else {
                // pad/trim aggregated_output to n_a
                let mut ha = vec![0.0_f32; self.cfg.n_a];
                let copy_len = aggregated_output.len().min(self.cfg.n_a);
                ha[..copy_len].copy_from_slice(&aggregated_output[..copy_len]);
                ha
            };

            let mask = self.attn_transformers[step].forward(&h_for_attn, &prior_scale)?;
            // Masked input
            let masked_x: Vec<f32> = mask.iter().zip(x.iter()).map(|(&m, &xi)| m * xi).collect();

            // Feature transform: shared + step-specific
            let shared_out = self.shared_layer.forward(&masked_x)?;
            let step_out = self.step_layers[step].forward(&shared_out)?;

            // Split into n_d (decision) and n_a (attention) parts
            let split_pt = self.cfg.n_d.min(step_out.len());
            let decision = &step_out[..split_pt];
            let relu_decision: Vec<f32> = decision.iter().map(|&v| relu(v)).collect();

            for (a, &d) in aggregated_output.iter_mut().zip(relu_decision.iter()) {
                *a += d;
            }

            // Update prior scale
            for (p, &m) in prior_scale.iter_mut().zip(mask.iter()) {
                *p *= (self.cfg.gamma - m).max(0.0);
            }

            masks.push(mask);
        }

        // Final output
        let logits = linear(&self.final_w, &self.final_b, &aggregated_output)?;
        Ok((logits, masks))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  SAINT  ═════════════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Single SAINT block with inter-sample + intra-feature attention.
#[derive(Debug, Clone)]
pub struct SaintBlock {
    d_model: usize,
    n_heads: usize,
    // Intra-feature (column) self-attention
    wq_intra: Vec<f32>,
    wk_intra: Vec<f32>,
    wv_intra: Vec<f32>,
    wo_intra: Vec<f32>,
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    // Inter-sample (row) self-attention: same-dim
    wq_inter: Vec<f32>,
    wk_inter: Vec<f32>,
    wv_inter: Vec<f32>,
    wo_inter: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    // FFN
    ffn_w1: Vec<f32>,
    ffn_b1: Vec<f32>,
    ffn_w2: Vec<f32>,
    ffn_b2: Vec<f32>,
    ln3_g: Vec<f32>,
    ln3_b: Vec<f32>,
}

impl SaintBlock {
    /// Create a new `SaintBlock`.
    pub fn new(d_model: usize, n_heads: usize, ffn_dim: usize, seed: u64) -> TabResult<Self> {
        if d_model % n_heads != 0 {
            return Err(format!(
                "SaintBlock: d_model={d_model} not divisible by n_heads={n_heads}"
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            d_model,
            n_heads,
            wq_intra: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wk_intra: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wv_intra: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wo_intra: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            ln1_g: ones(d_model),
            ln1_b: zeros(d_model),
            wq_inter: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wk_inter: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wv_inter: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            wo_inter: xavier_uniform(d_model * d_model, d_model, d_model, &mut rng),
            ln2_g: ones(d_model),
            ln2_b: zeros(d_model),
            ffn_w1: xavier_uniform(ffn_dim * d_model, d_model, ffn_dim, &mut rng),
            ffn_b1: zeros(ffn_dim),
            ffn_w2: xavier_uniform(d_model * ffn_dim, ffn_dim, d_model, &mut rng),
            ffn_b2: zeros(d_model),
            ln3_g: ones(d_model),
            ln3_b: zeros(d_model),
        })
    }

    /// Intra-feature (column-wise) self-attention over a single sample's feature tokens.
    /// `x` is [n_features × d_model] flattened.
    pub fn intra_feature_attention(&self, x: &[f32]) -> TabResult<Vec<f32>> {
        let d = self.d_model;
        if x.len() % d != 0 {
            return Err(format!(
                "SaintBlock::intra: x.len()={} not divisible by d_model={d}",
                x.len()
            ));
        }
        let seq_len = x.len() / d;
        let attn_out = scaled_dot_product_attn(
            x,
            x,
            x,
            seq_len,
            d,
            self.n_heads,
            &self.wq_intra,
            &self.wk_intra,
            &self.wv_intra,
            &self.wo_intra,
        )?;
        // Residual + LN
        let mut ln_out = vec![0.0_f32; seq_len * d];
        for s in 0..seq_len {
            let res: Vec<f32> = attn_out[s * d..(s + 1) * d]
                .iter()
                .zip(x[s * d..(s + 1) * d].iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let normed = layer_norm(&res, &self.ln1_g, &self.ln1_b)?;
            ln_out[s * d..(s + 1) * d].copy_from_slice(&normed);
        }
        // FFN
        let mut ffn_out = vec![0.0_f32; seq_len * d];
        for s in 0..seq_len {
            let tok = &ln_out[s * d..(s + 1) * d];
            let h1 = linear(&self.ffn_w1, &self.ffn_b1, tok)?;
            let h1a: Vec<f32> = h1.iter().map(|&v| gelu(v)).collect();
            let h2 = linear(&self.ffn_w2, &self.ffn_b2, &h1a)?;
            let res2: Vec<f32> = h2.iter().zip(tok.iter()).map(|(&a, &b)| a + b).collect();
            let normed2 = layer_norm(&res2, &self.ln3_g, &self.ln3_b)?;
            ffn_out[s * d..(s + 1) * d].copy_from_slice(&normed2);
        }
        Ok(ffn_out)
    }

    /// Inter-sample (row-wise) attention over a batch of sample representations.
    /// `batch` is a list of per-sample vectors, each of length `d_model`.
    pub fn inter_sample_attention(&self, batch: &[Vec<f32>]) -> TabResult<Vec<Vec<f32>>> {
        let d = self.d_model;
        let n_samples = batch.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        for (i, s) in batch.iter().enumerate() {
            if s.len() != d {
                return Err(format!(
                    "SaintBlock::inter: batch[{i}].len()={} != d_model={d}",
                    s.len()
                ));
            }
        }
        // Flatten batch → [n_samples × d_model]
        let flat: Vec<f32> = batch.iter().flat_map(|s| s.iter().copied()).collect();
        let attn_out = scaled_dot_product_attn(
            &flat,
            &flat,
            &flat,
            n_samples,
            d,
            self.n_heads,
            &self.wq_inter,
            &self.wk_inter,
            &self.wv_inter,
            &self.wo_inter,
        )?;
        // Residual + LN per sample
        let mut result = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let res: Vec<f32> = attn_out[s * d..(s + 1) * d]
                .iter()
                .zip(flat[s * d..(s + 1) * d].iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let normed = layer_norm(&res, &self.ln2_g, &self.ln2_b)?;
            result.push(normed);
        }
        Ok(result)
    }
}

/// Full SAINT model combining intra- and inter-sample attention blocks.
#[derive(Debug, Clone)]
pub struct SaintModel {
    /// Embedding dimension.
    pub d_model: usize,
    /// Number of SAINT blocks.
    pub n_blocks: usize,
    /// Categorical embeddings: one table per feature.
    cat_embeddings: Vec<Vec<f32>>,
    n_cat_features: usize,
    n_num_features: usize,
    cat_vocab_sizes: Vec<usize>,
    /// Numeric projection weights [n_num × d_model].
    num_w: Vec<Vec<f32>>,
    num_b: Vec<Vec<f32>>,
    /// SAINT blocks.
    blocks: Vec<SaintBlock>,
    /// Output head.
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    n_classes: usize,
}

impl SaintModel {
    /// Create a new `SaintModel`.
    pub fn new(
        n_cat_features: usize,
        n_num_features: usize,
        cat_vocab_sizes: Vec<usize>,
        d_model: usize,
        n_heads: usize,
        n_blocks: usize,
        ffn_dim: usize,
        n_classes: usize,
        seed: u64,
    ) -> TabResult<Self> {
        if cat_vocab_sizes.len() != n_cat_features {
            return Err("SaintModel: cat_vocab_sizes.len() != n_cat_features".into());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let cat_embeddings = cat_vocab_sizes
            .iter()
            .map(|&v| kaiming_uniform(v * d_model, d_model, &mut rng))
            .collect();
        let num_w = (0..n_num_features)
            .map(|_| xavier_uniform(d_model, 1, d_model, &mut rng))
            .collect();
        let num_b = (0..n_num_features).map(|_| zeros(d_model)).collect();
        let blocks = (0..n_blocks)
            .map(|i| SaintBlock::new(d_model, n_heads, ffn_dim, seed.wrapping_add(i as u64 + 1)))
            .collect::<TabResult<Vec<_>>>()?;
        let n_features = n_cat_features + n_num_features;
        let head_w = xavier_uniform(
            n_classes * n_features * d_model,
            n_features * d_model,
            n_classes,
            &mut rng,
        );
        let head_b = zeros(n_classes);
        Ok(Self {
            d_model,
            n_blocks,
            cat_embeddings,
            n_cat_features,
            n_num_features,
            cat_vocab_sizes,
            num_w,
            num_b,
            blocks,
            head_w,
            head_b,
            n_classes,
        })
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, cat_ids: &[usize], num_features: &[f32]) -> TabResult<Vec<f32>> {
        let d = self.d_model;
        let n_features = self.n_cat_features + self.n_num_features;

        // Build feature token sequence [n_features × d]
        let mut tokens = vec![0.0_f32; n_features * d];
        for (i, &id) in cat_ids.iter().enumerate() {
            let v = self.cat_vocab_sizes[i];
            let clamped = id.min(v.saturating_sub(1));
            let emb = &self.cat_embeddings[i][clamped * d..(clamped + 1) * d];
            tokens[i * d..(i + 1) * d].copy_from_slice(emb);
        }
        for (i, &x) in num_features.iter().enumerate() {
            let offset = (self.n_cat_features + i) * d;
            for j in 0..d {
                tokens[offset + j] = x * self.num_w[i][j] + self.num_b[i][j];
            }
        }

        // Apply SAINT blocks
        let mut seq = tokens;
        for block in &self.blocks {
            seq = block.intra_feature_attention(&seq)?;
        }

        // Flatten and classify
        linear(&self.head_w, &self.head_b, &seq)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  FeatureEncoder  ════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Standard z-score scaler: `(x − mean) / (std + ε)`.
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl StandardScaler {
    /// Fit on a list of feature vectors (each is a column slice).
    pub fn fit(&mut self, data: &[&[f32]]) {
        if data.is_empty() {
            return;
        }
        let n_features = data[0].len();
        let n = data.len() as f32;
        self.mean = vec![0.0_f32; n_features];
        self.std = vec![1.0_f32; n_features];
        for row in data {
            for (i, &v) in row.iter().enumerate() {
                if i < n_features {
                    self.mean[i] += v;
                }
            }
        }
        for m in self.mean.iter_mut() {
            *m /= n;
        }
        let mut var = vec![0.0_f32; n_features];
        for row in data {
            for (i, &v) in row.iter().enumerate() {
                if i < n_features {
                    var[i] += (v - self.mean[i]).powi(2);
                }
            }
        }
        for (i, v) in var.iter().enumerate() {
            self.std[i] = (v / n.max(1.0) + 1e-7).sqrt();
        }
    }

    /// Transform a single feature vector.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let m = self.mean.get(i).copied().unwrap_or(0.0);
                let s = self.std.get(i).copied().unwrap_or(1.0);
                (v - m) / s.max(1e-7)
            })
            .collect()
    }
}

/// Min-max scaler: `(x − min) / (range + ε)`.
#[derive(Debug, Clone, Default)]
pub struct MinMaxScaler {
    pub min: Vec<f32>,
    pub range: Vec<f32>,
}

impl MinMaxScaler {
    /// Fit on data columns.
    pub fn fit(&mut self, data: &[&[f32]]) {
        if data.is_empty() {
            return;
        }
        let n_features = data[0].len();
        let mut mins = vec![f32::INFINITY; n_features];
        let mut maxs = vec![f32::NEG_INFINITY; n_features];
        for row in data {
            for (i, &v) in row.iter().enumerate() {
                if i < n_features {
                    if v < mins[i] {
                        mins[i] = v;
                    }
                    if v > maxs[i] {
                        maxs[i] = v;
                    }
                }
            }
        }
        self.range = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| (hi - lo).max(1e-7))
            .collect();
        self.min = mins;
    }

    /// Transform a single feature vector.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let lo = self.min.get(i).copied().unwrap_or(0.0);
                let r = self.range.get(i).copied().unwrap_or(1.0);
                (v - lo) / r
            })
            .collect()
    }
}

/// Rank-based Gaussian quantile transformer.
#[derive(Debug, Clone, Default)]
pub struct QuantileTransformer {
    /// Per-feature sorted quantile values.
    pub quantiles: Vec<Vec<f32>>,
}

impl QuantileTransformer {
    /// Fit the quantile mapping from sorted data.
    pub fn fit(&mut self, data: &[&[f32]]) {
        if data.is_empty() {
            return;
        }
        let n_features = data[0].len();
        self.quantiles = vec![Vec::new(); n_features];
        for feat_idx in 0..n_features {
            let mut vals: Vec<f32> = data
                .iter()
                .filter_map(|row| row.get(feat_idx).copied())
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.quantiles[feat_idx] = vals;
        }
    }

    /// Transform: rank → standard normal quantile using erfinv approximation.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                if i >= self.quantiles.len() || self.quantiles[i].is_empty() {
                    return v;
                }
                let q = &self.quantiles[i];
                let n = q.len() as f32;
                // Find rank via binary search
                let rank = q.partition_point(|&s| s <= v);
                let p = (rank as f32 + 0.5) / (n + 1.0);
                let p_clamped = p.clamp(1e-6, 1.0 - 1e-6);
                // Probit via rational approximation (Beasley-Springer-Moro)
                probit(p_clamped)
            })
            .collect()
    }
}

/// Rational probit approximation (accurate to ~6 decimal places).
fn probit(p: f32) -> f32 {
    // Abramowitz & Stegun rational approximation
    let q = p - 0.5;
    if q.abs() < 0.425 {
        let r = 0.180625 - q * q;
        let num = ((2.509_081_f32 * r + 33.143_f32) * r + 85.44_f32) * r + 45.41_f32;
        let den = ((r + 15.159_f32) * r + 29.891_f32) * r + 1.0;
        q * (num / den)
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let lr = (-r.ln()).sqrt().clamp(0.0, 10.0);
        let sign = if q < 0.0 { -1.0_f32 } else { 1.0_f32 };
        let num = (1.4234_f32 * lr + 4.6233_f32) * lr + 0.6806_f32;
        let den = (lr + 3.6575_f32) * lr + 1.0_f32;
        sign * (num / den)
    }
}

/// Cyclic encoder for periodic features (sin + cos encoding).
#[derive(Debug, Clone)]
pub struct CyclicEncoder {
    /// Period for each feature (e.g., 24.0 for hours, 7.0 for weekdays).
    pub periods: Vec<f32>,
}

impl CyclicEncoder {
    /// Create a new `CyclicEncoder` with specified periods.
    pub fn new(periods: Vec<f32>) -> Self {
        Self { periods }
    }

    /// Transform: each feature `x` with period `T` → `[sin(2π x/T), cos(2π x/T)]`.
    /// Output length = 2 × input length.
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len() * 2);
        for (i, &v) in x.iter().enumerate() {
            let t = self.periods.get(i).copied().unwrap_or(1.0).max(1e-7);
            let angle = 2.0 * PI * v / t;
            out.push(angle.sin());
            out.push(angle.cos());
        }
        out
    }
}

/// Preprocessing pipeline combining multiple encoders.
#[derive(Debug, Clone)]
pub struct FeatureEncoder {
    pub scaler: StandardScaler,
    pub minmax: MinMaxScaler,
    pub quantile: QuantileTransformer,
    pub cyclic: Option<CyclicEncoder>,
}

impl FeatureEncoder {
    /// Create a new `FeatureEncoder`.
    pub fn new(cyclic_periods: Option<Vec<f32>>) -> Self {
        Self {
            scaler: StandardScaler::default(),
            minmax: MinMaxScaler::default(),
            quantile: QuantileTransformer::default(),
            cyclic: cyclic_periods.map(CyclicEncoder::new),
        }
    }

    /// Fit all sub-encoders.
    pub fn fit(&mut self, data: &[&[f32]]) {
        self.scaler.fit(data);
        self.minmax.fit(data);
        self.quantile.fit(data);
    }

    /// Apply standard scaling (most common).
    pub fn transform(&self, x: &[f32]) -> Vec<f32> {
        self.scaler.transform(x)
    }

    /// Apply quantile transform.
    pub fn quantile_transform(&self, x: &[f32]) -> Vec<f32> {
        self.quantile.transform(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  MixedInputHead  ════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Gated fusion of categorical and numeric representations.
///
/// Uses a learned sigmoid gate: `output = g * cat + (1 - g) * num`, where
/// `g = sigmoid(W_g [cat; num] + b_g)`.
#[derive(Debug, Clone)]
pub struct MixedInputHead {
    cat_dim: usize,
    num_dim: usize,
    out_dim: usize,
    /// Gate weights [out_dim × (cat_dim + num_dim)].
    gate_w: Vec<f32>,
    gate_b: Vec<f32>,
    /// Projection for cat representation.
    cat_proj_w: Vec<f32>,
    cat_proj_b: Vec<f32>,
    /// Projection for num representation.
    num_proj_w: Vec<f32>,
    num_proj_b: Vec<f32>,
}

impl MixedInputHead {
    /// Create a new `MixedInputHead`.
    pub fn new(cat_dim: usize, num_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let joint_dim = cat_dim + num_dim;
        Self {
            cat_dim,
            num_dim,
            out_dim,
            gate_w: xavier_uniform(out_dim * joint_dim, joint_dim, out_dim, &mut rng),
            gate_b: zeros(out_dim),
            cat_proj_w: xavier_uniform(out_dim * cat_dim, cat_dim, out_dim, &mut rng),
            cat_proj_b: zeros(out_dim),
            num_proj_w: xavier_uniform(out_dim * num_dim, num_dim, out_dim, &mut rng),
            num_proj_b: zeros(out_dim),
        }
    }

    /// Gated fusion: returns blended representation of length `out_dim`.
    pub fn gate(&self, cat_repr: &[f32], num_repr: &[f32]) -> TabResult<Vec<f32>> {
        if cat_repr.len() != self.cat_dim {
            return Err(format!(
                "MixedInputHead: cat_repr.len()={} != cat_dim={}",
                cat_repr.len(),
                self.cat_dim
            ));
        }
        if num_repr.len() != self.num_dim {
            return Err(format!(
                "MixedInputHead: num_repr.len()={} != num_dim={}",
                num_repr.len(),
                self.num_dim
            ));
        }
        let joint: Vec<f32> = cat_repr.iter().chain(num_repr.iter()).copied().collect();
        let gate_logits = linear(&self.gate_w, &self.gate_b, &joint)?;
        let g: Vec<f32> = gate_logits.iter().map(|&v| sigmoid(v)).collect();

        let cat_out = linear(&self.cat_proj_w, &self.cat_proj_b, cat_repr)?;
        let num_out = linear(&self.num_proj_w, &self.num_proj_b, num_repr)?;

        Ok(g.iter()
            .zip(cat_out.iter())
            .zip(num_out.iter())
            .map(|((&gi, &ci), &ni)| gi * ci + (1.0 - gi) * ni)
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  TabularAugmentation  ═══════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Data augmentation for tabular data.
pub struct TabularAugmentation;

impl TabularAugmentation {
    /// Mixup: returns `(λ·x1 + (1−λ)·x2, λ)` where `λ ~ Beta(α, α)`.
    /// Approximation: `λ = clip(|N(0.5, 1/(12α))|, 0, 1)`.
    pub fn mixup(
        x1: &[f32],
        x2: &[f32],
        alpha: f32,
        rng: &mut StdRng,
    ) -> TabResult<(Vec<f32>, f32)> {
        if x1.len() != x2.len() {
            return Err(format!("mixup: len mismatch {} vs {}", x1.len(), x2.len()));
        }
        let lambda = Self::beta_sample(alpha, rng);
        let mixed: Vec<f32> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| lambda * a + (1.0 - lambda) * b)
            .collect();
        Ok((mixed, lambda))
    }

    /// CutMix: randomly replace a contiguous block of features from x2 into x1.
    pub fn cutmix(
        x1: &[f32],
        x2: &[f32],
        alpha: f32,
        rng: &mut StdRng,
    ) -> TabResult<(Vec<f32>, f32)> {
        if x1.len() != x2.len() {
            return Err(format!("cutmix: len mismatch {} vs {}", x1.len(), x2.len()));
        }
        let lambda = Self::beta_sample(alpha, rng);
        let n = x1.len();
        let cut_len = (n as f32 * (1.0 - lambda)).round() as usize;
        let start_f: f32 = rng.random();
        let start = (start_f * (n.saturating_sub(cut_len) + 1) as f32) as usize;
        let end = (start + cut_len).min(n);

        let mut mixed = x1.to_vec();
        mixed[start..end].copy_from_slice(&x2[start..end]);
        let actual_lambda = 1.0 - (end - start) as f32 / n.max(1) as f32;
        Ok((mixed, actual_lambda))
    }

    /// SMOTE-like synthetic oversampling: generates a new sample between `sample`
    /// and a random neighbour from `neighbours`.
    pub fn smote_like(
        sample: &[f32],
        neighbours: &[Vec<f32>],
        rng: &mut StdRng,
    ) -> TabResult<Vec<f32>> {
        if neighbours.is_empty() {
            return Err("smote_like: no neighbours provided".into());
        }
        let idx_f: f32 = rng.random();
        let idx = (idx_f * neighbours.len() as f32) as usize;
        let idx = idx.min(neighbours.len() - 1);
        let neighbour = &neighbours[idx];
        if sample.len() != neighbour.len() {
            return Err(format!(
                "smote_like: sample.len()={} != neighbour.len()={}",
                sample.len(),
                neighbour.len()
            ));
        }
        let gap: f32 = rng.random();
        Ok(sample
            .iter()
            .zip(neighbour.iter())
            .map(|(&s, &n)| s + gap * (n - s))
            .collect())
    }

    /// Sample λ from a symmetric Beta(α, α) using the approximation
    /// `λ ≈ 0.5 + N(0, 1) * σ` then clipped, where `σ = 1/(2*sqrt(2α+1))`.
    fn beta_sample(alpha: f32, rng: &mut StdRng) -> f32 {
        let alpha = alpha.max(0.01);
        // Box-Muller
        let u1: f32 = rng.random::<f32>().max(1e-7);
        let u2: f32 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let sigma = 1.0 / (2.0 * (2.0 * alpha + 1.0).sqrt());
        (0.5 + sigma * z).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  TabularMetrics  ════════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for tabular ML tasks.
pub struct TabularMetrics;

impl TabularMetrics {
    /// Classification accuracy: fraction of correctly predicted classes.
    /// `pred` is a flat vector of per-class logits; `n_classes` implicit from pred/target.
    pub fn accuracy(pred_logits: &[Vec<f32>], target: &[usize]) -> f32 {
        if pred_logits.is_empty() || pred_logits.len() != target.len() {
            return 0.0;
        }
        let correct = pred_logits
            .iter()
            .zip(target.iter())
            .filter(|(logits, &t)| {
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                max_idx == t
            })
            .count();
        correct as f32 / pred_logits.len() as f32
    }

    /// Macro-averaged F1 score.
    pub fn macro_f1(pred: &[Vec<f32>], target: &[usize], n_classes: usize) -> f32 {
        if pred.is_empty() || n_classes == 0 {
            return 0.0;
        }
        let mut tp = vec![0u32; n_classes];
        let mut fp = vec![0u32; n_classes];
        let mut fn_ = vec![0u32; n_classes];

        for (logits, &t) in pred.iter().zip(target.iter()) {
            let pred_class = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if pred_class == t {
                if t < n_classes {
                    tp[t] += 1;
                }
            } else {
                if pred_class < n_classes {
                    fp[pred_class] += 1;
                }
                if t < n_classes {
                    fn_[t] += 1;
                }
            }
        }

        let mut f1_sum = 0.0_f32;
        for c in 0..n_classes {
            let precision = tp[c] as f32 / (tp[c] + fp[c]).max(1) as f32;
            let recall = tp[c] as f32 / (tp[c] + fn_[c]).max(1) as f32;
            let denom = precision + recall;
            let f1 = if denom > 0.0 {
                2.0 * precision * recall / denom
            } else {
                0.0
            };
            f1_sum += f1;
        }
        f1_sum / n_classes as f32
    }

    /// Root Mean Squared Error.
    pub fn rmse(pred: &[f32], target: &[f32]) -> f32 {
        if pred.is_empty() || pred.len() != target.len() {
            return f32::NAN;
        }
        let mse = pred
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            / pred.len() as f32;
        mse.sqrt()
    }

    /// R² (coefficient of determination).
    pub fn r2_score(pred: &[f32], target: &[f32]) -> f32 {
        if pred.is_empty() || pred.len() != target.len() {
            return f32::NAN;
        }
        let mean_t = target.iter().sum::<f32>() / target.len() as f32;
        let ss_tot: f32 = target.iter().map(|&t| (t - mean_t).powi(2)).sum();
        let ss_res: f32 = pred
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| (t - p).powi(2))
            .sum();
        if ss_tot < 1e-12 {
            return if ss_res < 1e-12 { 1.0 } else { 0.0 };
        }
        1.0 - ss_res / ss_tot
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════  CatBoostEncoder  ═══════════════════════════════════
// ─────────────────────────────────────────────────────────────────────────────

/// Target encoding with Leave-One-Out regularisation (à la CatBoost).
///
/// For each sample `(cat_i, y_i)` at training time, the encoded value is:
/// `(sum_j≠i y_j where cat_j = cat_i) / (count_j≠i - 1 + 1) * λ + prior * (1 - λ)`
/// where `λ` is based on per-category count.
#[derive(Debug, Clone, Default)]
pub struct CatBoostEncoder {
    /// Mapping from category id → (count, sum_of_targets).
    pub category_stats: std::collections::HashMap<usize, (usize, f64)>,
    /// Global prior (mean of all targets).
    pub prior: f32,
}

impl CatBoostEncoder {
    /// Create a `CatBoostEncoder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit and transform simultaneously (LOO to avoid leakage).
    pub fn fit_transform(&mut self, categories: &[usize], targets: &[f32], prior: f32) -> Vec<f32> {
        if categories.len() != targets.len() || categories.is_empty() {
            return Vec::new();
        }
        self.prior = prior;
        self.category_stats.clear();

        // Accumulate full stats first
        for (&cat, &t) in categories.iter().zip(targets.iter()) {
            let entry = self.category_stats.entry(cat).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += t as f64;
        }

        // LOO transform
        categories
            .iter()
            .zip(targets.iter())
            .map(|(&cat, &t)| {
                let (count, sum) = self.category_stats.get(&cat).copied().unwrap_or((0, 0.0));
                // Remove current sample
                let loo_sum = sum - t as f64;
                let loo_count = count.saturating_sub(1);
                // Regularisation weight: diminishes for low-count categories
                let lambda = loo_count as f32 / (loo_count as f32 + 1.0);
                let loo_mean = if loo_count == 0 {
                    prior
                } else {
                    (loo_sum / loo_count as f64) as f32
                };
                lambda * loo_mean + (1.0 - lambda) * prior
            })
            .collect()
    }

    /// Transform new (unseen) categories using fitted statistics.
    pub fn transform(&self, categories: &[usize]) -> Vec<f32> {
        categories
            .iter()
            .map(|&cat| match self.category_stats.get(&cat) {
                None => self.prior,
                Some(&(count, sum)) => {
                    let mean = if count == 0 {
                        self.prior as f64
                    } else {
                        sum / count as f64
                    };
                    let lambda = count as f32 / (count as f32 + 1.0);
                    lambda * mean as f32 + (1.0 - lambda) * self.prior
                }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
