//! Modality embedding tables and the unified multi-modal transformer backbone.

use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{
    gelu, mat_vec, ones_vec, softmax, xavier_init, zeros_vec, ModalSequence, MomModalityType,
};

// ─────────────────────────────────────────────────────────────────────────────
// §4 Modality Embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding table that combines token, position, and modality-type embeddings.
pub struct ModalityEmbedding {
    /// Model dimension.
    pub d_model: usize,
    /// One embedding vector per modality type: `[n_modalities][d_model]`.
    pub modality_embeds: Vec<Vec<f64>>,
    /// Token embedding table: `[total_vocab][d_model]`.
    pub token_embeds: Vec<Vec<f64>>,
    /// Positional embedding table: `[max_seq_len][d_model]`.
    pub position_embeds: Vec<Vec<f64>>,
}

impl ModalityEmbedding {
    /// Create embedding tables with sinusoidal position encodings and
    /// Xavier-initialized token / modality embeddings.
    pub fn new(d_model: usize, total_vocab: usize, max_seq_len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        let scale = (1.0 / d_model as f64).sqrt();

        let modality_embeds: Vec<Vec<f64>> = (0..MomModalityType::count())
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        use scirs2_core::random::Rng;
                        (rng.random::<f64>() * 2.0 - 1.0) * scale
                    })
                    .collect()
            })
            .collect();

        let token_embeds: Vec<Vec<f64>> = (0..total_vocab)
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        use scirs2_core::random::Rng;
                        (rng.random::<f64>() * 2.0 - 1.0) * scale
                    })
                    .collect()
            })
            .collect();

        // Sinusoidal positional encodings.
        let position_embeds: Vec<Vec<f64>> = (0..max_seq_len)
            .map(|pos| {
                (0..d_model)
                    .map(|i| {
                        let denom = (10000.0f64).powf(2.0 * (i / 2) as f64 / d_model as f64);
                        if i % 2 == 0 {
                            (pos as f64 / denom).sin()
                        } else {
                            (pos as f64 / denom).cos()
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            d_model,
            modality_embeds,
            token_embeds,
            position_embeds,
        }
    }

    /// Map a [`MomModalityType`] to its integer index (0..6).
    pub fn modality_type_index(m: &MomModalityType) -> usize {
        match m {
            MomModalityType::Text => 0,
            MomModalityType::Image => 1,
            MomModalityType::Audio => 2,
            MomModalityType::Video => 3,
            MomModalityType::Tabular => 4,
            MomModalityType::Code => 5,
            MomModalityType::Math => 6,
        }
    }

    /// Embed a [`ModalSequence`] into dense vectors.
    ///
    /// Returns `[seq_len][d_model]` where each vector is:
    /// `token_embed[token_id] + position_embed[pos] + modality_embed[modality]`.
    pub fn embed(&self, seq: &ModalSequence) -> Vec<Vec<f64>> {
        let max_pos = self.position_embeds.len();
        let total_vocab = self.token_embeds.len();
        seq.tokens
            .iter()
            .map(|tok| {
                let tid = tok.token_id % total_vocab;
                let pos = tok.position.min(max_pos - 1);
                let mid = Self::modality_type_index(&tok.modality);
                let te = &self.token_embeds[tid];
                let pe = &self.position_embeds[pos];
                let me = &self.modality_embeds[mid];
                te.iter()
                    .zip(pe.iter())
                    .zip(me.iter())
                    .map(|((t, p), m)| t + p + m)
                    .collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5 Unified Transformer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the unified transformer backbone.
#[derive(Debug, Clone)]
pub struct UnifiedTransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub total_vocab: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

/// A single transformer layer with pre-LayerNorm multi-head attention and FFN.
pub struct UnifiedTransformerLayer {
    pub w_q: Vec<Vec<f64>>,
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
    pub w_o: Vec<Vec<f64>>,
    pub w1: Vec<Vec<f64>>,
    pub w2: Vec<Vec<f64>>,
    pub ln1_scale: Vec<f64>,
    pub ln1_bias: Vec<f64>,
    pub ln2_scale: Vec<f64>,
    pub ln2_bias: Vec<f64>,
}

impl UnifiedTransformerLayer {
    pub(super) fn new(config: &UnifiedTransformerConfig, rng: &mut StdRng) -> Self {
        let d = config.d_model;
        let ff = config.d_ff;
        Self {
            w_q: xavier_init(rng, d, d),
            w_k: xavier_init(rng, d, d),
            w_v: xavier_init(rng, d, d),
            w_o: xavier_init(rng, d, d),
            w1: xavier_init(rng, ff, d),
            w2: xavier_init(rng, d, ff),
            ln1_scale: ones_vec(d),
            ln1_bias: zeros_vec(d),
            ln2_scale: ones_vec(d),
            ln2_bias: zeros_vec(d),
        }
    }
}

/// Unified multi-modal transformer.
pub struct UnifiedTransformer {
    pub config: UnifiedTransformerConfig,
    pub embedding: ModalityEmbedding,
    pub layers: Vec<UnifiedTransformerLayer>,
    /// Un-embedding matrix: `[total_vocab][d_model]`, logit\[j\] = dot(unembed\[j\], h).
    pub unembed: Vec<Vec<f64>>,
}

impl UnifiedTransformer {
    /// Create a new transformer with random weights.
    pub fn new(config: UnifiedTransformerConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(999);
        let embedding =
            ModalityEmbedding::new(config.d_model, config.total_vocab, config.max_seq_len);
        let layers: Vec<UnifiedTransformerLayer> = (0..config.n_layers)
            .map(|_| UnifiedTransformerLayer::new(&config, &mut rng))
            .collect();
        // unembed: [total_vocab][d_model] so that logit[j] = dot(unembed[j], h)
        let unembed = xavier_init(&mut rng, config.total_vocab, config.d_model);
        Self {
            config,
            embedding,
            layers,
            unembed,
        }
    }

    /// Layer normalization: `scale * (x - mean) / std + bias`.
    pub fn layer_norm(x: &[f64], scale: &[f64], bias: &[f64]) -> Vec<f64> {
        let n = x.len() as f64;
        let mean: f64 = x.iter().sum::<f64>() / n;
        let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = (var + 1e-6).sqrt();
        x.iter()
            .zip(scale.iter())
            .zip(bias.iter())
            .map(|((xi, s), b)| s * (xi - mean) / std + b)
            .collect()
    }

    /// Scaled dot-product multi-head attention.
    ///
    /// Returns `[seq_len][d_model]`.
    pub fn multi_head_attention(
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        w_q: &[Vec<f64>],
        w_k: &[Vec<f64>],
        w_v: &[Vec<f64>],
        w_o: &[Vec<f64>],
        n_heads: usize,
        _mask: Option<&Vec<Vec<f64>>>,
    ) -> Vec<Vec<f64>> {
        let seq_len = q.len();
        if seq_len == 0 {
            return Vec::new();
        }
        let d_model = q[0].len();
        let d_head = d_model / n_heads;
        let scale = (d_head as f64).sqrt().recip();

        let proj_q: Vec<Vec<f64>> = q.iter().map(|qi| mat_vec(w_q, qi)).collect();
        let proj_k: Vec<Vec<f64>> = k.iter().map(|ki| mat_vec(w_k, ki)).collect();
        let proj_v: Vec<Vec<f64>> = v.iter().map(|vi| mat_vec(w_v, vi)).collect();

        let mut output = vec![vec![0.0f64; d_model]; seq_len];

        for h in 0..n_heads {
            let h_start = h * d_head;
            let h_end = h_start + d_head;

            let head_q: Vec<&[f64]> = proj_q.iter().map(|r| &r[h_start..h_end]).collect();
            let head_k: Vec<&[f64]> = proj_k.iter().map(|r| &r[h_start..h_end]).collect();
            let head_v: Vec<&[f64]> = proj_v.iter().map(|r| &r[h_start..h_end]).collect();

            let scores: Vec<Vec<f64>> = head_q
                .iter()
                .map(|qi| {
                    head_k
                        .iter()
                        .map(|kj| qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum::<f64>() * scale)
                        .collect()
                })
                .collect();

            let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

            for (i, attn_row) in attn.iter().enumerate() {
                for (j, &a) in attn_row.iter().enumerate() {
                    for (d, &vd) in head_v[j].iter().enumerate() {
                        output[i][h_start + d] += a * vd;
                    }
                }
            }
        }

        output.iter().map(|row| mat_vec(w_o, row)).collect()
    }

    /// Forward pass: returns `[seq_len][total_vocab]` logits.
    pub fn forward(&self, seq: &ModalSequence) -> Vec<Vec<f64>> {
        let mut hidden = self.embedding.embed(seq);
        if hidden.is_empty() {
            return Vec::new();
        }

        for layer in &self.layers {
            // Pre-LN attention.
            let normed1: Vec<Vec<f64>> = hidden
                .iter()
                .map(|h| Self::layer_norm(h, &layer.ln1_scale, &layer.ln1_bias))
                .collect();

            let attn_out = Self::multi_head_attention(
                &normed1,
                &normed1,
                &normed1,
                &layer.w_q,
                &layer.w_k,
                &layer.w_v,
                &layer.w_o,
                self.config.n_heads,
                None,
            );

            for (h, a) in hidden.iter_mut().zip(attn_out.iter()) {
                for (hi, ai) in h.iter_mut().zip(a.iter()) {
                    *hi += ai;
                }
            }

            // Pre-LN FFN.
            let normed2: Vec<Vec<f64>> = hidden
                .iter()
                .map(|h| Self::layer_norm(h, &layer.ln2_scale, &layer.ln2_bias))
                .collect();

            let ffn_out: Vec<Vec<f64>> = normed2
                .iter()
                .map(|x| {
                    let h1: Vec<f64> = mat_vec(&layer.w1, x).iter().map(|&v| gelu(v)).collect();
                    mat_vec(&layer.w2, &h1)
                })
                .collect();

            for (h, f) in hidden.iter_mut().zip(ffn_out.iter()) {
                for (hi, fi) in h.iter_mut().zip(f.iter()) {
                    *hi += fi;
                }
            }
        }

        // Un-embed: [seq_len][total_vocab].
        hidden
            .iter()
            .map(|h| {
                self.unembed
                    .iter()
                    .map(|row| h.iter().zip(row.iter()).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect()
    }
}
