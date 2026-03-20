//! MI metrics: faithfulness, completeness, minimality, and probe accuracy.

use super::attention::AttentionAnalyzer;
use super::helpers::zeros_vec;
use super::patcher::ActivationPatcher;
use super::probe::{ProbeClassifier, ProbeConfig};
use super::sae::MiSparseAutoencoder;

/// Quantitative measures of a circuit explanation.
#[derive(Debug, Clone)]
pub struct MiMetrics {
    /// How well the circuit reproduces full-model behaviour (0–1, higher = better).
    pub faithfulness: f64,
    /// Fraction of performance the circuit captures (0–1, higher = better).
    pub completeness: f64,
    /// How parsimonious the circuit is: `1 − (circuit_nodes / total_nodes)`.
    pub minimality: f64,
    /// Best linear probe accuracy achieved on a per-layer sweep.
    pub probe_accuracy: f64,
}

/// High-level summary combining metrics, top heads, SAE features, and causal
/// layer importance.
#[derive(Debug, Clone)]
pub struct MiReport {
    /// Quantitative MI metrics.
    pub metrics: MiMetrics,
    /// Top-k heads by absolute attribution score: `(layer, head, score)`.
    pub top_heads: Vec<(usize, usize, f64)>,
    /// Top SAE features by firing frequency: `(feature_idx, frequency)`.
    pub top_sae_features: Vec<(usize, f64)>,
    /// Layers with largest causal effect: `(layer, normalized_effect)`.
    pub causal_important_layers: Vec<(usize, f64)>,
}

fn default_metric(logits: &[Vec<f64>]) -> f64 {
    match logits.last() {
        Some(last) if last.len() >= 2 => last[0] - last[1],
        Some(last) => last[0],
        None => 0.0,
    }
}

/// Compute [`MiMetrics`] via activation patching on all layers and a simple
/// probe trained on the last-layer residual.
pub fn compute_mi_metrics(
    patcher: &ActivationPatcher,
    clean_tokens: &[usize],
    corrupted_tokens: &[usize],
) -> MiMetrics {
    let model = &patcher.model;

    let sweep = patcher.activation_patching_sweep(clean_tokens, corrupted_tokens, default_metric);

    let total_effect: f64 = sweep.iter().map(|r| r.resid_effect.abs()).sum();
    let n_layers = sweep.len() as f64;
    let avg_effect = if n_layers > 0.0 {
        total_effect / n_layers
    } else {
        0.0
    };

    let best_effect = sweep.iter().map(|r| r.resid_effect).fold(0.0_f64, f64::max);
    let faithfulness = best_effect.clamp(0.0, 1.0);
    let completeness = avg_effect.clamp(0.0, 1.0);

    let threshold = 0.1;
    let n_important = sweep
        .iter()
        .filter(|r| r.resid_effect.abs() > threshold)
        .count();
    let minimality = if sweep.is_empty() {
        0.0
    } else {
        1.0 - (n_important as f64 / sweep.len() as f64)
    };

    let d_model = model.d_model;
    let (_cl, clean_cache) = model.forward_with_cache(clean_tokens);
    let (_cr, corrupted_cache) = model.forward_with_cache(corrupted_tokens);

    let last_layer = if model.n_layers > 0 {
        model.n_layers - 1
    } else {
        0
    };
    let key = format!("residual_{last_layer}");
    let clean_res = clean_cache
        .get(&key)
        .cloned()
        .unwrap_or_else(|| zeros_vec(d_model));
    let corrupted_res = corrupted_cache
        .get(&key)
        .cloned()
        .unwrap_or_else(|| zeros_vec(d_model));

    let mut probe = ProbeClassifier::new(ProbeConfig {
        d_input: d_model,
        n_classes: 2,
        learning_rate: 0.1,
        n_epochs: 20,
    });
    let acts = vec![clean_res, corrupted_res];
    let labels = vec![0_usize, 1_usize];
    let _ = probe.fit(&acts, &labels);
    let probe_accuracy = probe.accuracy(&acts, &labels);

    MiMetrics {
        faithfulness,
        completeness,
        minimality,
        probe_accuracy,
    }
}

/// Build a full [`MiReport`] from a patcher, an attention analyser, and an SAE.
pub fn build_mi_report(
    patcher: &ActivationPatcher,
    analyzer: &AttentionAnalyzer,
    sae: &MiSparseAutoencoder,
    clean_tokens: &[usize],
    corrupted_tokens: &[usize],
    sae_data: &[Vec<f64>],
) -> MiReport {
    let metrics = compute_mi_metrics(patcher, clean_tokens, corrupted_tokens);

    let vocab_size = patcher.model.vocab_size;
    let c_idx = 0_usize;
    let ic_idx = 1_usize.min(vocab_size - 1);
    let attr = analyzer.head_attribution(clean_tokens, c_idx, ic_idx);
    let mut head_scores: Vec<(usize, usize, f64)> = attr
        .iter()
        .enumerate()
        .flat_map(|(l, layer)| layer.iter().enumerate().map(move |(h, &s)| (l, h, s)))
        .collect();
    head_scores.sort_by(|(_, _, a), (_, _, b)| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    head_scores.truncate(10);

    let freq = sae.feature_frequency(sae_data);
    let mut feat_scores: Vec<(usize, f64)> = freq.iter().cloned().enumerate().collect();
    feat_scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    feat_scores.truncate(10);

    let sweep = patcher.activation_patching_sweep(clean_tokens, corrupted_tokens, default_metric);
    let mut causal_layers: Vec<(usize, f64)> =
        sweep.iter().map(|r| (r.layer, r.resid_effect)).collect();
    causal_layers.sort_by(|(_, a), (_, b)| {
        b.abs()
            .partial_cmp(&a.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    MiReport {
        metrics,
        top_heads: head_scores,
        top_sae_features: feat_scores,
        causal_important_layers: causal_layers,
    }
}
