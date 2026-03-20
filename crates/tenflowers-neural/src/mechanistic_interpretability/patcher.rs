//! Causal activation patching (Meng et al. 2022 / ROME style).

use super::cache::ActivationCache;
use super::helpers::{layer_norm, vec_add, vecmat, zeros_vec};
use super::transformer::MiTransformer;

/// Result of a single activation-patching intervention.
#[derive(Debug, Clone)]
pub struct PatchResult {
    /// Which hook was patched.
    pub hook_name: String,
    /// Metric on the clean run.
    pub clean_metric: f64,
    /// Metric on the corrupted run.
    pub corrupted_metric: f64,
    /// Metric after patching the clean activation into the corrupted run.
    pub patched_metric: f64,
    /// `(patched − corrupted) / (clean − corrupted)` clamped to avoid NaN.
    pub normalized_effect: f64,
}

/// Per-layer result from [`ActivationPatcher::activation_patching_sweep`].
#[derive(Debug, Clone)]
pub struct LayerPatchResult {
    /// Layer index (0-based).
    pub layer: usize,
    /// Effect of patching the attention output at this layer.
    pub attn_effect: f64,
    /// Effect of patching the MLP output at this layer.
    pub mlp_effect: f64,
    /// Effect of patching the residual stream at this layer.
    pub resid_effect: f64,
}

/// Performs causal intervention by patching activations from a "clean" run
/// into a "corrupted" run (Meng et al. 2022, ROME).
pub struct ActivationPatcher {
    /// The model used for all forward passes.
    pub model: MiTransformer,
}

impl ActivationPatcher {
    /// Wrap `model` in an `ActivationPatcher`.
    pub fn new(model: MiTransformer) -> Self {
        Self { model }
    }

    /// Run the clean tokens, capture the named activation, then re-run the
    /// corrupted tokens with that activation patched in.
    pub fn patch_activation(
        &self,
        clean_tokens: &[usize],
        corrupted_tokens: &[usize],
        hook_name: &str,
        metric: impl Fn(&[Vec<f64>]) -> f64,
    ) -> PatchResult {
        let (clean_logits, clean_cache) = self.model.forward_with_cache(clean_tokens);
        let (corrupted_logits, _corrupted_cache) = self.model.forward_with_cache(corrupted_tokens);

        let clean_metric = metric(&clean_logits);
        let corrupted_metric = metric(&corrupted_logits);

        let patched_metric =
            self.forward_with_patch(corrupted_tokens, hook_name, &clean_cache, &metric);

        let denom = clean_metric - corrupted_metric;
        let normalized_effect = if denom.abs() < 1e-10 {
            0.0
        } else {
            (patched_metric - corrupted_metric) / denom
        };

        PatchResult {
            hook_name: hook_name.to_string(),
            clean_metric,
            corrupted_metric,
            patched_metric,
            normalized_effect,
        }
    }

    /// Forward pass that injects the specified activation from `patch_cache`
    /// at the matching key, running the rest of the model normally.
    pub(super) fn forward_with_patch(
        &self,
        tokens: &[usize],
        hook_name: &str,
        patch_cache: &ActivationCache,
        metric: impl Fn(&[Vec<f64>]) -> f64,
    ) -> f64 {
        let model = &self.model;
        let mut residual: Vec<Vec<f64>> = model.embed_tokens(tokens);

        for l in 0..model.n_layers {
            let attn_hook = format!("layer_{l}_attn_output");
            let mlp_hook = format!("layer_{l}_mlp");
            let resid_hook = format!("residual_{l}");

            let normed: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln1_scale, &model.layers[l].ln1_bias))
                .collect();
            let (attn_out, _) = model.attention(&normed, l);

            let attn_add: Vec<Vec<f64>> = if hook_name == attn_hook {
                if let Some(patch_val) = patch_cache.get(hook_name) {
                    let mut a = attn_out.clone();
                    if let Some(last) = a.last_mut() {
                        *last = patch_val.clone();
                    }
                    a
                } else {
                    attn_out
                }
            } else {
                attn_out
            };

            residual = residual
                .iter()
                .zip(attn_add.iter())
                .map(|(r, a)| vec_add(r, a))
                .collect();

            let normed2: Vec<Vec<f64>> = residual
                .iter()
                .map(|r| layer_norm(r, &model.layers[l].ln2_scale, &model.layers[l].ln2_bias))
                .collect();
            let mlp_outs: Vec<Vec<f64>> = normed2.iter().map(|r| model.mlp(r, l)).collect();

            let mlp_add: Vec<Vec<f64>> = if hook_name == mlp_hook {
                if let Some(patch_val) = patch_cache.get(hook_name) {
                    let mut m = mlp_outs.clone();
                    if let Some(last) = m.last_mut() {
                        *last = patch_val.clone();
                    }
                    m
                } else {
                    mlp_outs
                }
            } else {
                mlp_outs
            };

            residual = residual
                .iter()
                .zip(mlp_add.iter())
                .map(|(r, m)| vec_add(r, m))
                .collect();

            if hook_name == resid_hook {
                if let Some(patch_val) = patch_cache.get(hook_name) {
                    if let Some(last) = residual.last_mut() {
                        *last = patch_val.clone();
                    }
                }
            }
        }

        let logits: Vec<Vec<f64>> = residual.iter().map(|r| vecmat(r, &model.unembed)).collect();
        metric(&logits)
    }

    /// Compute activation patching across **all** layers.
    pub fn activation_patching_sweep(
        &self,
        clean_tokens: &[usize],
        corrupted_tokens: &[usize],
        metric: impl Fn(&[Vec<f64>]) -> f64,
    ) -> Vec<LayerPatchResult> {
        let (clean_logits, clean_cache) = self.model.forward_with_cache(clean_tokens);
        let (corrupted_logits, _) = self.model.forward_with_cache(corrupted_tokens);

        let clean_metric = metric(&clean_logits);
        let corrupted_metric = metric(&corrupted_logits);
        let denom = clean_metric - corrupted_metric;

        (0..self.model.n_layers)
            .map(|l| {
                let attn_m = self.forward_with_patch(
                    corrupted_tokens,
                    &format!("layer_{l}_attn_output"),
                    &clean_cache,
                    &metric,
                );
                let mlp_m = self.forward_with_patch(
                    corrupted_tokens,
                    &format!("layer_{l}_mlp"),
                    &clean_cache,
                    &metric,
                );
                let resid_m = self.forward_with_patch(
                    corrupted_tokens,
                    &format!("residual_{l}"),
                    &clean_cache,
                    &metric,
                );

                let norm = |m: f64| {
                    if denom.abs() < 1e-10 {
                        0.0
                    } else {
                        (m - corrupted_metric) / denom
                    }
                };

                LayerPatchResult {
                    layer: l,
                    attn_effect: norm(attn_m),
                    mlp_effect: norm(mlp_m),
                    resid_effect: norm(resid_m),
                }
            })
            .collect()
    }
}
