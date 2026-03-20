//! Logit lens — project residual stream at each layer to vocabulary space.
//!
//! Reference: nostalgebraist (2020), "interpreting GPT: the logit lens".
//! Tuned Lens: Belrose et al. (2023).

use super::helpers::{entropy, softmax, vecmat, zeros_vec};
use super::transformer::MiTransformer;

/// Result of the logit-lens projection at a single layer.
#[derive(Debug, Clone)]
pub struct LayerLogitLensResult {
    /// Layer index (0-based).
    pub layer: usize,
    /// Top-k `(token_index, logit_value)` pairs, sorted descending by logit.
    pub top_predictions: Vec<(usize, f64)>,
    /// Shannon entropy (nats) of the softmax distribution over the vocabulary.
    pub entropy: f64,
}

/// Project the residual stream at each layer to vocabulary space.
pub struct LogitLens {
    /// The model whose activations are analysed.
    pub model: MiTransformer,
}

impl LogitLens {
    /// Wrap `model` in a `LogitLens`.
    pub fn new(model: MiTransformer) -> Self {
        Self { model }
    }

    /// For each layer, apply the unembedding matrix to the cached residual
    /// stream at the **last** token position and return the top-`k` predictions.
    pub fn compute(&self, tokens: &[usize], top_k: usize) -> Vec<LayerLogitLensResult> {
        let (_logits, cache) = self.model.forward_with_cache(tokens);

        (0..self.model.n_layers)
            .map(|l| {
                let key = format!("residual_{l}");
                let residual = cache
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| zeros_vec(self.model.d_model));

                let logits = vecmat(&residual, &self.model.unembed);
                let probs = softmax(&logits);
                let ent = entropy(&probs);

                let mut indexed: Vec<(usize, f64)> = logits.iter().cloned().enumerate().collect();
                indexed.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(top_k);

                LayerLogitLensResult {
                    layer: l,
                    top_predictions: indexed,
                    entropy: ent,
                }
            })
            .collect()
    }

    /// Tuned Lens step (Belrose et al. 2023): apply a learned per-layer affine
    /// translation `translate` to the residual stream before unembedding.
    ///
    /// This is a simplified version (identity transform + offset).
    pub fn tuned_lens_step(&self, residual: &[f64], _layer: usize, translate: &[f64]) -> Vec<f64> {
        let shifted: Vec<f64> = residual
            .iter()
            .zip(translate.iter())
            .map(|(r, t)| r + t)
            .collect();
        vecmat(&shifted, &self.model.unembed)
    }
}
