//! Minimal transformer model with activation caching for mechanistic interpretability.

use super::cache::ActivationCache;
use super::helpers::{
    dot, layer_norm, matvec, ones_vec, random_matrix, relu, softmax, vec_add, vecmat, zeros_vec,
};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

/// Single transformer layer (attention + MLP) used by [`MiTransformer`].
#[derive(Debug, Clone)]
pub struct MiLayer {
    /// Query projection `[d_model][d_model]`
    pub w_q: Vec<Vec<f64>>,
    /// Key projection `[d_model][d_model]`
    pub w_k: Vec<Vec<f64>>,
    /// Value projection `[d_model][d_model]`
    pub w_v: Vec<Vec<f64>>,
    /// Output projection `[d_model][d_model]`
    pub w_o: Vec<Vec<f64>>,
    /// MLP input projection `[d_model][d_ff]`
    pub w_mlp_in: Vec<Vec<f64>>,
    /// MLP output projection `[d_ff][d_model]`
    pub w_mlp_out: Vec<Vec<f64>>,
    /// Layer-norm 1 scale `[d_model]`
    pub ln1_scale: Vec<f64>,
    /// Layer-norm 1 bias `[d_model]`
    pub ln1_bias: Vec<f64>,
    /// Layer-norm 2 scale `[d_model]`
    pub ln2_scale: Vec<f64>,
    /// Layer-norm 2 bias `[d_model]`
    pub ln2_bias: Vec<f64>,
}

impl MiLayer {
    pub(super) fn new(d_model: usize, d_ff: usize, rng: &mut StdRng) -> Self {
        let s = (1.0 / d_model as f64).sqrt();
        Self {
            w_q: random_matrix(d_model, d_model, rng, s),
            w_k: random_matrix(d_model, d_model, rng, s),
            w_v: random_matrix(d_model, d_model, rng, s),
            w_o: random_matrix(d_model, d_model, rng, s),
            w_mlp_in: random_matrix(d_model, d_ff, rng, s),
            w_mlp_out: random_matrix(d_ff, d_model, rng, s),
            ln1_scale: ones_vec(d_model),
            ln1_bias: zeros_vec(d_model),
            ln2_scale: ones_vec(d_model),
            ln2_bias: zeros_vec(d_model),
        }
    }
}

/// A minimal transformer with cache hooks for mechanistic interpretability.
///
/// All weights are initialised randomly with a fixed seed so tests are
/// reproducible. This is deliberately a *toy* model; performance matters less
/// than exposing clean intermediate activations.
#[derive(Debug, Clone)]
pub struct MiTransformer {
    /// Hidden model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of stacked transformer layers.
    pub n_layers: usize,
    /// Feed-forward inner dimension.
    pub d_ff: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Token embedding matrix `[vocab_size][d_model]`.
    pub embed: Vec<Vec<f64>>,
    /// Per-layer weights.
    pub layers: Vec<MiLayer>,
    /// Unembedding matrix `[d_model][vocab_size]`.
    pub unembed: Vec<Vec<f64>>,
}

impl MiTransformer {
    /// Construct a randomly initialised `MiTransformer`.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        vocab_size: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let s = (1.0 / d_model as f64).sqrt();
        let embed = random_matrix(vocab_size, d_model, &mut rng, s);
        let layers: Vec<MiLayer> = (0..n_layers)
            .map(|_| MiLayer::new(d_model, d_ff, &mut rng))
            .collect();
        let unembed = random_matrix(d_model, vocab_size, &mut rng, s);
        Self {
            d_model,
            n_heads,
            n_layers,
            d_ff,
            vocab_size,
            embed,
            layers,
            unembed,
        }
    }

    /// Embed a token sequence: returns `[seq_len][d_model]`.
    pub(super) fn embed_tokens(&self, tokens: &[usize]) -> Vec<Vec<f64>> {
        tokens
            .iter()
            .map(|&t| self.embed[t.min(self.vocab_size - 1)].clone())
            .collect()
    }

    /// Multi-head scaled dot-product attention.
    ///
    /// Returns `(output [seq_len][d_model], attn_weights [n_heads][seq][seq])`.
    pub(super) fn attention(
        &self,
        x: &[Vec<f64>],
        layer_idx: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let seq_len = x.len();
        let layer = &self.layers[layer_idx];
        let head_dim = self.d_model / self.n_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let mut all_head_weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        let mut out = vec![zeros_vec(self.d_model); seq_len];

        for h in 0..self.n_heads {
            let h_start = h * head_dim;
            let h_end = h_start + head_dim;

            let q_h: Vec<Vec<f64>> = x
                .iter()
                .map(|xi| {
                    let qi = matvec(&layer.w_q, xi);
                    qi[h_start..h_end].to_vec()
                })
                .collect();
            let k_h: Vec<Vec<f64>> = x
                .iter()
                .map(|xi| {
                    let ki = matvec(&layer.w_k, xi);
                    ki[h_start..h_end].to_vec()
                })
                .collect();
            let v_h: Vec<Vec<f64>> = x
                .iter()
                .map(|xi| {
                    let vi = matvec(&layer.w_v, xi);
                    vi[h_start..h_end].to_vec()
                })
                .collect();

            // Attention scores [seq][seq] with causal mask.
            let mut scores: Vec<Vec<f64>> = vec![vec![0.0; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..=i {
                    scores[i][j] = dot(&q_h[i], &k_h[j]) * scale;
                }
                for j in (i + 1)..seq_len {
                    scores[i][j] = f64::NEG_INFINITY;
                }
            }

            let attn: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();

            let head_out: Vec<Vec<f64>> = (0..seq_len)
                .map(|i| {
                    let mut acc = zeros_vec(head_dim);
                    for j in 0..seq_len {
                        let a = attn[i][j];
                        for (k, &v) in acc.iter_mut().zip(v_h[j].iter()) {
                            *k += a * v;
                        }
                    }
                    acc
                })
                .collect();

            all_head_weights.push(attn);

            for (pos, h_vec) in head_out.iter().enumerate() {
                let mut full = zeros_vec(self.d_model);
                for (i, &v) in h_vec.iter().enumerate() {
                    full[h_start + i] = v;
                }
                let projected = matvec(&layer.w_o, &full);
                for (o, &p) in out[pos].iter_mut().zip(projected.iter()) {
                    *o += p;
                }
            }
        }

        (out, all_head_weights)
    }

    /// Apply one MLP layer with ReLU.
    pub(super) fn mlp(&self, x: &[f64], layer_idx: usize) -> Vec<f64> {
        let layer = &self.layers[layer_idx];
        let hidden: Vec<f64> = matvec(&layer.w_mlp_in, x).into_iter().map(relu).collect();
        matvec(&layer.w_mlp_out, &hidden)
    }

    /// Full forward pass; returns `(logits per position [seq_len][vocab], cache)`.
    pub fn forward_with_cache(&self, tokens: &[usize]) -> (Vec<Vec<f64>>, ActivationCache) {
        let mut cache = ActivationCache::new(self.n_layers, self.d_model);
        let mut residual: Vec<Vec<f64>> = self.embed_tokens(tokens);
        let seq_len = residual.len();

        for (i, r) in residual.iter().enumerate() {
            cache.store(&format!("embed_{i}"), r.clone());
        }

        for l in 0..self.n_layers {
            let normed: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &self.layers[l].ln1_scale, &self.layers[l].ln1_bias))
                .collect();

            let (attn_out, attn_weights) = self.attention(&normed, l);

            residual = residual
                .iter()
                .zip(attn_out.iter())
                .map(|(r, a)| vec_add(r, a))
                .collect();

            cache.store(
                &format!("layer_{l}_attn_output"),
                attn_out.last().cloned().unwrap_or_default(),
            );

            for h in 0..self.n_heads {
                cache.store(
                    &format!("layer_{l}_head_{h}_attn"),
                    attn_weights[h]
                        .last()
                        .cloned()
                        .unwrap_or_else(|| zeros_vec(seq_len)),
                );
            }

            let normed2: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &self.layers[l].ln2_scale, &self.layers[l].ln2_bias))
                .collect();

            let mlp_outs: Vec<Vec<f64>> = normed2.iter().map(|r| self.mlp(r, l)).collect();

            residual = residual
                .iter()
                .zip(mlp_outs.iter())
                .map(|(r, m)| vec_add(r, m))
                .collect();

            cache.store(
                &format!("layer_{l}_mlp"),
                mlp_outs.last().cloned().unwrap_or_default(),
            );

            cache.store(
                &format!("residual_{l}"),
                residual.last().cloned().unwrap_or_default(),
            );
        }

        // Unembed: unembed is [d_model][vocab_size], logit[v] = residual · unembed[:,v]
        let logits: Vec<Vec<f64>> = residual.iter().map(|r| vecmat(r, &self.unembed)).collect();

        (logits, cache)
    }

    /// `logits[correct] - logits[incorrect]` at the **last** token position.
    pub fn logit_diff(&self, tokens: &[usize], correct_idx: usize, incorrect_idx: usize) -> f64 {
        let (logits, _) = self.forward_with_cache(tokens);
        match logits.last() {
            Some(last) => {
                let c = correct_idx.min(last.len() - 1);
                let ic = incorrect_idx.min(last.len() - 1);
                last[c] - last[ic]
            }
            None => 0.0,
        }
    }
}
