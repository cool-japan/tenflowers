//! Sequential recommendation models: SASRec and BERT4Rec.

use scirs2_core::random::{rngs::StdRng, SeedableRng};

use super::{
    dot, layer_norm, matvec, relu_f32, softmax_inplace, xavier_fill, RecResult, RecSysError,
};

// ─────────────────────────────────────────────────────────────────────────────
// SASRec — Self-Attentive Sequential Recommendation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SasRec`].
#[derive(Debug, Clone)]
pub struct SasRecConfig {
    /// Vocabulary size (number of items, 0-indexed, 0 = padding).
    pub n_items: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model dimension (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Feedforward hidden dimension.
    pub ffn_dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Dropout probability (applied in training; inference uses 0).
    pub dropout: f32,
}

impl Default for SasRecConfig {
    fn default() -> Self {
        Self {
            n_items: 1000,
            max_seq_len: 50,
            d_model: 64,
            n_heads: 2,
            ffn_dim: 128,
            n_layers: 2,
            dropout: 0.1,
        }
    }
}

/// Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).
///
/// Uses a stack of causal (left-only) self-attention blocks to model a user's
/// sequential item interaction history and predict the next item.
#[derive(Debug, Clone)]
pub struct SasRec {
    pub(crate) cfg: SasRecConfig,
    /// Item embeddings `[n_items × d_model]`.
    item_emb: Vec<f32>,
    /// Positional embeddings `[max_seq_len × d_model]`.
    pos_emb: Vec<f32>,
    /// Per-layer parameters.
    pub(crate) layers: Vec<SasRecLayer>,
    /// Final layer-norm gamma and beta.
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct SasRecLayer {
    // Self-attention projections (each row-major [d_model × d_model])
    pub(crate) wq: Vec<f32>,
    pub(crate) wk: Vec<f32>,
    pub(crate) wv: Vec<f32>,
    pub(crate) wo: Vec<f32>,
    // Layer norm 1
    pub(crate) ln1_gamma: Vec<f32>,
    pub(crate) ln1_beta: Vec<f32>,
    // FFN
    pub(crate) w1: Vec<f32>, // [ffn_dim × d_model]
    pub(crate) b1: Vec<f32>,
    pub(crate) w2: Vec<f32>, // [d_model × ffn_dim]
    pub(crate) b2: Vec<f32>,
    // Layer norm 2
    pub(crate) ln2_gamma: Vec<f32>,
    pub(crate) ln2_beta: Vec<f32>,
}

impl SasRecLayer {
    fn new(d_model: usize, ffn_dim: usize, rng: &mut StdRng) -> Self {
        let mut make_w = |rows: usize, cols: usize| {
            let mut w = vec![0.0_f32; rows * cols];
            xavier_fill(&mut w, cols, rows, rng);
            w
        };
        Self {
            wq: make_w(d_model, d_model),
            wk: make_w(d_model, d_model),
            wv: make_w(d_model, d_model),
            wo: make_w(d_model, d_model),
            ln1_gamma: vec![1.0_f32; d_model],
            ln1_beta: vec![0.0_f32; d_model],
            w1: make_w(ffn_dim, d_model),
            b1: vec![0.0_f32; ffn_dim],
            w2: make_w(d_model, ffn_dim),
            b2: vec![0.0_f32; d_model],
            ln2_gamma: vec![1.0_f32; d_model],
            ln2_beta: vec![0.0_f32; d_model],
        }
    }
}

impl SasRec {
    /// Create a randomly-initialised SASRec model.
    pub fn new(cfg: SasRecConfig, seed: u64) -> RecResult<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.d_model;
        let ni = cfg.n_items;
        let max_len = cfg.max_seq_len;

        let mut item_emb = vec![0.0_f32; ni * d];
        xavier_fill(&mut item_emb, d, d, &mut rng);

        // Fixed sinusoidal positional embeddings (like original SASRec)
        let mut pos_emb = vec![0.0_f32; max_len * d];
        for pos in 0..max_len {
            for i in 0..d {
                let angle = pos as f32 / (10000_f32).powf(2.0 * (i / 2) as f32 / d as f32);
                pos_emb[pos * d + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let layers = (0..cfg.n_layers)
            .map(|_| SasRecLayer::new(d, cfg.ffn_dim, &mut rng))
            .collect();

        Ok(Self {
            cfg,
            item_emb,
            pos_emb,
            layers,
            ln_gamma: vec![1.0_f32; d],
            ln_beta: vec![0.0_f32; d],
        })
    }

    /// Forward pass over an item sequence.
    ///
    /// `item_seq` is a list of item indices (0 = padding, ignored).
    /// Returns a score vector of length `n_items` representing the logit for
    /// each item being the next item.
    pub fn forward(&self, item_seq: &[usize]) -> RecResult<Vec<f32>> {
        if item_seq.is_empty() {
            return Err(RecSysError::EmptySequence);
        }
        let d = self.cfg.d_model;
        let n_items = self.cfg.n_items;
        let max_len = self.cfg.max_seq_len;

        // Truncate to max_seq_len (keep last tokens)
        let seq: Vec<usize> = if item_seq.len() > max_len {
            item_seq[item_seq.len() - max_len..].to_vec()
        } else {
            item_seq.to_vec()
        };
        let seq_len = seq.len();

        // Build token embeddings + positional embeddings
        let mut hidden = vec![0.0_f32; seq_len * d];
        for (t, &item) in seq.iter().enumerate() {
            let item_clamped = item.min(n_items - 1);
            for k in 0..d {
                hidden[t * d + k] = self.item_emb[item_clamped * d + k] + self.pos_emb[t * d + k];
            }
        }

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = self.apply_layer(layer, &hidden, seq_len, d)?;
        }

        // Final layer norm on the last token
        let last_h = &hidden[(seq_len - 1) * d..seq_len * d];
        let normed = layer_norm(last_h, &self.ln_gamma, &self.ln_beta, 1e-6)?;

        // Score all items via inner product with item embeddings
        let scores: Vec<f32> = (0..n_items)
            .map(|i| dot(&normed, &self.item_emb[i * d..(i + 1) * d]))
            .collect();
        Ok(scores)
    }

    fn apply_layer(
        &self,
        layer: &SasRecLayer,
        h: &[f32],
        seq_len: usize,
        d: usize,
    ) -> RecResult<Vec<f32>> {
        let head_d = d / self.cfg.n_heads;
        let scale = 1.0 / (head_d as f32).sqrt();

        // Multi-head causal self-attention
        let mut attn_out = vec![0.0_f32; seq_len * d];
        let nh = self.cfg.n_heads;

        for head in 0..nh {
            let h_off = head * head_d;
            let q: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    let row_q: Vec<f32> = (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wq[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect();
                    row_q
                })
                .collect();
            let k: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wk[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            let v: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wv[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();

            // Causal attention
            for t in 0..seq_len {
                let mut attn_scores = vec![f32::NEG_INFINITY; seq_len];
                for s in 0..=t {
                    let q_t = &q[t * head_d..(t + 1) * head_d];
                    let k_s = &k[s * head_d..(s + 1) * head_d];
                    attn_scores[s] = dot(q_t, k_s) * scale;
                }
                let mut valid_scores: Vec<f32> = attn_scores[..=t].to_vec();
                softmax_inplace(&mut valid_scores);
                let mut out_t = vec![0.0_f32; head_d];
                for (s, &weight) in valid_scores.iter().enumerate() {
                    let v_s = &v[s * head_d..(s + 1) * head_d];
                    for i in 0..head_d {
                        out_t[i] += weight * v_s[i];
                    }
                }
                for i in 0..head_d {
                    attn_out[t * d + h_off + i] = out_t[i];
                }
            }
        }

        // Output projection W_O
        let mut after_attn = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let a_t = &attn_out[t * d..(t + 1) * d];
            let proj = matvec(&layer.wo, a_t, d, d);
            for k in 0..d {
                after_attn[t * d + k] = proj[k];
            }
        }

        // Residual + layer norm 1
        let mut h1 = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let res: Vec<f32> = (0..d)
                .map(|k| h[t * d + k] + after_attn[t * d + k])
                .collect();
            let normed = layer_norm(&res, &layer.ln1_gamma, &layer.ln1_beta, 1e-6)?;
            for k in 0..d {
                h1[t * d + k] = normed[k];
            }
        }

        // FFN (ReLU)
        let ffn_dim = self.cfg.ffn_dim;
        let mut h2 = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let x = &h1[t * d..(t + 1) * d];
            let mut mid = matvec(&layer.w1, x, ffn_dim, d);
            for (j, b) in layer.b1.iter().enumerate() {
                mid[j] = relu_f32(mid[j] + b);
            }
            let mut out_ffn = matvec(&layer.w2, &mid, d, ffn_dim);
            for (j, b) in layer.b2.iter().enumerate() {
                out_ffn[j] += b;
            }
            // Residual + layer norm 2
            let res: Vec<f32> = (0..d).map(|k| h1[t * d + k] + out_ffn[k]).collect();
            let normed = layer_norm(&res, &layer.ln2_gamma, &layer.ln2_beta, 1e-6)?;
            for k in 0..d {
                h2[t * d + k] = normed[k];
            }
        }
        Ok(h2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BERT4Rec — Bidirectional Transformer for Recommendation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`BERT4Rec`].
#[derive(Debug, Clone)]
pub struct Bert4RecConfig {
    /// Vocabulary size (item count + special tokens).
    pub n_items: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// FFN hidden dim.
    pub ffn_dim: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Mask token index (typically `n_items`).
    pub mask_token: usize,
}

impl Default for Bert4RecConfig {
    fn default() -> Self {
        Self {
            n_items: 1000,
            max_seq_len: 50,
            d_model: 64,
            n_heads: 2,
            ffn_dim: 128,
            n_layers: 2,
            mask_token: 1000,
        }
    }
}

/// BERT4Rec (Sun et al., 2019) — bidirectional transformer for masked item
/// modelling in sequential recommendation.
#[derive(Debug, Clone)]
pub struct BERT4Rec {
    cfg: Bert4RecConfig,
    item_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    pub(crate) layers: Vec<Bert4RecLayer>,
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
    // Output head: predicts logits over item vocabulary
    head_w: Vec<f32>, // [vocab × d_model]
    head_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct Bert4RecLayer {
    pub(crate) wq: Vec<f32>,
    pub(crate) wk: Vec<f32>,
    pub(crate) wv: Vec<f32>,
    pub(crate) wo: Vec<f32>,
    pub(crate) ln1_gamma: Vec<f32>,
    pub(crate) ln1_beta: Vec<f32>,
    pub(crate) w1: Vec<f32>,
    pub(crate) b1: Vec<f32>,
    pub(crate) w2: Vec<f32>,
    pub(crate) b2: Vec<f32>,
    pub(crate) ln2_gamma: Vec<f32>,
    pub(crate) ln2_beta: Vec<f32>,
}

impl Bert4RecLayer {
    fn new(d_model: usize, ffn_dim: usize, rng: &mut StdRng) -> Self {
        let mut make_w = |rows: usize, cols: usize| {
            let mut w = vec![0.0_f32; rows * cols];
            xavier_fill(&mut w, cols, rows, rng);
            w
        };
        Self {
            wq: make_w(d_model, d_model),
            wk: make_w(d_model, d_model),
            wv: make_w(d_model, d_model),
            wo: make_w(d_model, d_model),
            ln1_gamma: vec![1.0_f32; d_model],
            ln1_beta: vec![0.0_f32; d_model],
            w1: make_w(ffn_dim, d_model),
            b1: vec![0.0_f32; ffn_dim],
            w2: make_w(d_model, ffn_dim),
            b2: vec![0.0_f32; d_model],
            ln2_gamma: vec![1.0_f32; d_model],
            ln2_beta: vec![0.0_f32; d_model],
        }
    }
}

impl BERT4Rec {
    /// Create a randomly-initialised BERT4Rec model.
    pub fn new(cfg: Bert4RecConfig, seed: u64) -> RecResult<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = cfg.d_model;
        // +1 for mask token
        let vocab = cfg.n_items + 1;
        let max_len = cfg.max_seq_len;

        let mut item_emb = vec![0.0_f32; vocab * d];
        xavier_fill(&mut item_emb, d, d, &mut rng);

        let mut pos_emb = vec![0.0_f32; max_len * d];
        for pos in 0..max_len {
            for i in 0..d {
                let angle = pos as f32 / (10000_f32).powf(2.0 * (i / 2) as f32 / d as f32);
                pos_emb[pos * d + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let layers = (0..cfg.n_layers)
            .map(|_| Bert4RecLayer::new(d, cfg.ffn_dim, &mut rng))
            .collect();

        let mut head_w = vec![0.0_f32; vocab * d];
        xavier_fill(&mut head_w, d, vocab, &mut rng);

        Ok(Self {
            cfg,
            item_emb,
            pos_emb,
            layers,
            ln_gamma: vec![1.0_f32; d],
            ln_beta: vec![0.0_f32; d],
            head_w,
            head_b: vec![0.0_f32; vocab],
        })
    }

    /// Forward pass for masked item modelling.
    ///
    /// Returns `Vec<Vec<f32>>` where entry `k` is a vocab-length logit vector
    /// for position `mask_pos[k]`.
    pub fn forward(&self, item_seq: &[usize], mask_pos: &[usize]) -> RecResult<Vec<Vec<f32>>> {
        if item_seq.is_empty() {
            return Err(RecSysError::EmptySequence);
        }
        let d = self.cfg.d_model;
        let vocab = self.cfg.n_items + 1;
        let max_len = self.cfg.max_seq_len;

        let seq: Vec<usize> = if item_seq.len() > max_len {
            item_seq[item_seq.len() - max_len..].to_vec()
        } else {
            item_seq.to_vec()
        };
        let seq_len = seq.len();

        // Embedding lookup + positional
        let mut hidden = vec![0.0_f32; seq_len * d];
        for (t, &item) in seq.iter().enumerate() {
            let item_clamped = item.min(vocab - 1);
            for k in 0..d {
                hidden[t * d + k] = self.item_emb[item_clamped * d + k] + self.pos_emb[t * d + k];
            }
        }

        // Full (bidirectional) transformer layers
        for layer in &self.layers {
            hidden = self.apply_bert_layer(layer, &hidden, seq_len, d)?;
        }

        // Output projection at each masked position
        let mut results = Vec::with_capacity(mask_pos.len());
        for &pos in mask_pos {
            let pos_clamped = pos.min(seq_len - 1);
            let h_t = &hidden[pos_clamped * d..(pos_clamped + 1) * d];
            let normed = layer_norm(h_t, &self.ln_gamma, &self.ln_beta, 1e-6)?;
            let logits = matvec(&self.head_w, &normed, vocab, d);
            let logits_b: Vec<f32> = logits
                .iter()
                .zip(self.head_b.iter())
                .map(|(l, b)| l + b)
                .collect();
            results.push(logits_b);
        }
        Ok(results)
    }

    fn apply_bert_layer(
        &self,
        layer: &Bert4RecLayer,
        h: &[f32],
        seq_len: usize,
        d: usize,
    ) -> RecResult<Vec<f32>> {
        let nh = self.cfg.n_heads;
        let head_d = d / nh;
        let scale = 1.0 / (head_d as f32).sqrt();

        // Full (non-causal) self-attention
        let mut attn_out = vec![0.0_f32; seq_len * d];
        for head in 0..nh {
            let h_off = head * head_d;
            let q: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wq[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            let k: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wk[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            let v: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let x = &h[t * d..(t + 1) * d];
                    (0..head_d)
                        .map(|i| {
                            let row_idx = h_off + i;
                            dot(x, &layer.wv[row_idx * d..(row_idx + 1) * d])
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();

            for t in 0..seq_len {
                // Full attention over all positions
                let mut scores: Vec<f32> = (0..seq_len)
                    .map(|s| {
                        let q_t = &q[t * head_d..(t + 1) * head_d];
                        let k_s = &k[s * head_d..(s + 1) * head_d];
                        dot(q_t, k_s) * scale
                    })
                    .collect();
                softmax_inplace(&mut scores);
                let mut out_t = vec![0.0_f32; head_d];
                for (s, &weight) in scores.iter().enumerate() {
                    for i in 0..head_d {
                        out_t[i] += weight * v[s * head_d + i];
                    }
                }
                for i in 0..head_d {
                    attn_out[t * d + h_off + i] = out_t[i];
                }
            }
        }

        // Output projection
        let mut after_attn = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let proj = matvec(&layer.wo, &attn_out[t * d..(t + 1) * d], d, d);
            for k in 0..d {
                after_attn[t * d + k] = proj[k];
            }
        }

        // Residual + LN1
        let ffn_dim = self.cfg.ffn_dim;
        let mut h1 = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let res: Vec<f32> = (0..d)
                .map(|k| h[t * d + k] + after_attn[t * d + k])
                .collect();
            let normed = layer_norm(&res, &layer.ln1_gamma, &layer.ln1_beta, 1e-6)?;
            for k in 0..d {
                h1[t * d + k] = normed[k];
            }
        }

        // FFN + residual + LN2
        let mut h2 = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let x = &h1[t * d..(t + 1) * d];
            let mut mid = matvec(&layer.w1, x, ffn_dim, d);
            for (j, b) in layer.b1.iter().enumerate() {
                mid[j] = relu_f32(mid[j] + b);
            }
            let mut out_ffn = matvec(&layer.w2, &mid, d, ffn_dim);
            for (j, b) in layer.b2.iter().enumerate() {
                out_ffn[j] += b;
            }
            let res: Vec<f32> = (0..d).map(|k| h1[t * d + k] + out_ffn[k]).collect();
            let normed = layer_norm(&res, &layer.ln2_gamma, &layer.ln2_beta, 1e-6)?;
            for k in 0..d {
                h2[t * d + k] = normed[k];
            }
        }
        Ok(h2)
    }
}
