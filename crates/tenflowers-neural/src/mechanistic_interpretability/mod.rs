//! Mechanistic Interpretability (MI) — scientific study of neural network internals.
//!
//! This module implements the core tools used at Anthropic and other AI safety
//! labs to discover what features neurons represent, how computations flow
//! through circuits, and what role each attention head plays.
//!
//! # Key components
//! * [`ActivationCache`] — stores every intermediate activation during a forward pass.
//! * [`MiTransformer`] — minimal transformer with cache hooks.
//! * [`ActivationPatcher`] — causal intervention via activation patching (Meng et al. 2022).
//! * [`LogitLens`] — project residual stream to vocab space (nostalgebraist 2020).
//! * [`MiSparseAutoencoder`] — learn a sparse dictionary of features (Anthropic 2023).
//! * [`AttentionAnalyzer`] — per-head attribution, induction-head detection, rollout.
//! * [`ProbeClassifier`] — linear probe trained on cached activations.
//! * [`MiMetrics`] / [`MiReport`] — circuit-level faithfulness / completeness metrics.

pub mod attention;
pub mod cache;
pub(crate) mod helpers;
pub mod logit_lens;
pub mod metrics;
pub mod patcher;
pub mod probe;
pub mod sae;
pub mod transformer;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use attention::AttentionAnalyzer;
pub use cache::ActivationCache;
pub use logit_lens::{LayerLogitLensResult, LogitLens};
pub use metrics::{build_mi_report, compute_mi_metrics, MiMetrics, MiReport};
pub use patcher::{ActivationPatcher, LayerPatchResult, PatchResult};
pub use probe::{ProbeClassifier, ProbeConfig};
pub use sae::{MiSaeConfig, MiSparseAutoencoder, SparseAutoencoder, SparseAutoencoderConfig};
pub use transformer::{MiLayer, MiTransformer};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
    use scirs2_core::RngExt;

    // Re-export helpers needed only in tests.
    use helpers::{layer_norm, softmax};

    fn small_model() -> MiTransformer {
        MiTransformer::new(8, 2, 2, 16, 20)
    }

    fn tokens() -> Vec<usize> {
        vec![0, 1, 2, 3, 4]
    }

    fn random_data(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(999);
        (0..n)
            .map(|_| (0..d).map(|_| rng.random::<f64>()).collect())
            .collect()
    }

    fn small_sae() -> MiSparseAutoencoder {
        MiSparseAutoencoder::new(MiSaeConfig {
            d_input: 8,
            d_hidden: 32,
            l1_coefficient: 0.01,
            learning_rate: 0.01,
        })
    }

    // ── ActivationCache ───────────────────────────────────────────────────────

    #[test]
    fn test_cache_store_retrieve() {
        let mut cache = ActivationCache::new(4, 16);
        cache.store("layer_0_mlp", vec![1.0, 2.0, 3.0]);
        let v = cache.get("layer_0_mlp").expect("key should exist");
        assert_eq!(v, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cache_missing_key() {
        let cache = ActivationCache::new(4, 16);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_keys_sorted() {
        let mut cache = ActivationCache::new(2, 8);
        cache.store("b_key", vec![0.0]);
        cache.store("a_key", vec![1.0]);
        let keys = cache.keys();
        assert_eq!(keys[0], "a_key");
        assert_eq!(keys[1], "b_key");
    }

    #[test]
    fn test_cache_diff() {
        let mut c1 = ActivationCache::new(2, 4);
        let mut c2 = ActivationCache::new(2, 4);
        c1.store("x", vec![3.0, 5.0]);
        c2.store("x", vec![1.0, 2.0]);
        let d = c1.diff(&c2, "x").expect("diff should succeed");
        assert!((d[0] - 2.0).abs() < 1e-9);
        assert!((d[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_diff_missing() {
        let c1 = ActivationCache::new(2, 4);
        let c2 = ActivationCache::new(2, 4);
        assert!(c1.diff(&c2, "missing").is_none());
    }

    #[test]
    fn test_cache_diff_length_mismatch() {
        let mut c1 = ActivationCache::new(2, 4);
        let mut c2 = ActivationCache::new(2, 4);
        c1.store("x", vec![1.0, 2.0]);
        c2.store("x", vec![3.0]);
        assert!(c1.diff(&c2, "x").is_none());
    }

    // ── MiTransformer ─────────────────────────────────────────────────────────

    #[test]
    fn test_forward_logits_shape() {
        let model = small_model();
        let toks = tokens();
        let (logits, _cache) = model.forward_with_cache(&toks);
        assert_eq!(logits.len(), toks.len(), "one logit vector per token");
        for row in &logits {
            assert_eq!(row.len(), model.vocab_size);
        }
    }

    #[test]
    fn test_forward_cache_has_residuals() {
        let model = small_model();
        let (_, cache) = model.forward_with_cache(&tokens());
        for l in 0..model.n_layers {
            assert!(
                cache.get(&format!("residual_{l}")).is_some(),
                "residual_{l} should be cached"
            );
        }
    }

    #[test]
    fn test_forward_cache_has_attn_output() {
        let model = small_model();
        let (_, cache) = model.forward_with_cache(&tokens());
        for l in 0..model.n_layers {
            assert!(
                cache.get(&format!("layer_{l}_attn_output")).is_some(),
                "layer_{l}_attn_output should be cached"
            );
        }
    }

    #[test]
    fn test_forward_cache_has_mlp() {
        let model = small_model();
        let (_, cache) = model.forward_with_cache(&tokens());
        for l in 0..model.n_layers {
            assert!(
                cache.get(&format!("layer_{l}_mlp")).is_some(),
                "layer_{l}_mlp should be cached"
            );
        }
    }

    #[test]
    fn test_logit_diff_is_finite() {
        let model = small_model();
        let diff = model.logit_diff(&tokens(), 0, 1);
        assert!(diff.is_finite());
    }

    #[test]
    fn test_logit_diff_correct_ne_incorrect() {
        let model = MiTransformer::new(8, 2, 2, 16, 20);
        let toks = vec![5, 6, 7];
        let d = model.logit_diff(&toks, 0, 1);
        let d2 = model.logit_diff(&toks, 1, 0);
        assert!((d + d2).abs() < 1e-9);
    }

    #[test]
    fn test_single_token_forward() {
        let model = small_model();
        let (logits, _) = model.forward_with_cache(&[3]);
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), model.vocab_size);
    }

    // ── ActivationPatcher ─────────────────────────────────────────────────────

    #[test]
    fn test_patch_result_fields_finite() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let clean = vec![0, 1, 2];
        let corrupted = vec![5, 6, 7];
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let result = patcher.patch_activation(&clean, &corrupted, "layer_0_attn_output", metric);
        assert!(result.clean_metric.is_finite());
        assert!(result.corrupted_metric.is_finite());
        assert!(result.patched_metric.is_finite());
        assert!(result.normalized_effect.is_finite());
    }

    #[test]
    fn test_patch_result_hook_name() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let result = patcher.patch_activation(&tokens(), &[5, 6, 7, 8, 9], "layer_1_mlp", metric);
        assert_eq!(result.hook_name, "layer_1_mlp");
    }

    #[test]
    fn test_patch_normalized_effect_range() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let toks = tokens();
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let result = patcher.patch_activation(&toks, &toks, "layer_0_attn_output", metric);
        assert!(result.normalized_effect.abs() < 1e-6);
    }

    #[test]
    fn test_activation_patching_sweep_length() {
        let model = small_model();
        let n_layers = model.n_layers;
        let patcher = ActivationPatcher::new(model);
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let sweep = patcher.activation_patching_sweep(&tokens(), &[5, 6, 7, 8, 9], metric);
        assert_eq!(sweep.len(), n_layers);
    }

    #[test]
    fn test_activation_patching_sweep_layer_indices() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let sweep = patcher.activation_patching_sweep(&tokens(), &[5, 6, 7, 8, 9], metric);
        for (expected, result) in sweep.iter().enumerate() {
            assert_eq!(result.layer, expected);
        }
    }

    #[test]
    fn test_activation_patching_sweep_effects_finite() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let metric = |logits: &[Vec<f64>]| -> f64 { logits.last().map_or(0.0, |l| l[0]) };
        let sweep = patcher.activation_patching_sweep(&tokens(), &[5, 6, 7, 8, 9], metric);
        for r in &sweep {
            assert!(r.attn_effect.is_finite());
            assert!(r.mlp_effect.is_finite());
            assert!(r.resid_effect.is_finite());
        }
    }

    // ── LogitLens ─────────────────────────────────────────────────────────────

    #[test]
    fn test_logit_lens_result_count() {
        let model = small_model();
        let n_layers = model.n_layers;
        let lens = LogitLens::new(model);
        let results = lens.compute(&tokens(), 3);
        assert_eq!(results.len(), n_layers);
    }

    #[test]
    fn test_logit_lens_top_k_length() {
        let model = small_model();
        let lens = LogitLens::new(model);
        let results = lens.compute(&tokens(), 5);
        for r in &results {
            assert!(r.top_predictions.len() <= 5);
        }
    }

    #[test]
    fn test_logit_lens_layer_indices() {
        let model = small_model();
        let n_layers = model.n_layers;
        let lens = LogitLens::new(model);
        let results = lens.compute(&tokens(), 2);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.layer, i);
            assert!(i < n_layers);
        }
    }

    #[test]
    fn test_logit_lens_entropy_nonnegative() {
        let model = small_model();
        let lens = LogitLens::new(model);
        let results = lens.compute(&tokens(), 3);
        for r in &results {
            assert!(r.entropy >= 0.0, "entropy must be non-negative");
        }
    }

    #[test]
    fn test_logit_lens_tuned_step_shape() {
        let model = small_model();
        let d_model = model.d_model;
        let vocab = model.vocab_size;
        let lens = LogitLens::new(model);
        let res = vec![0.1_f64; d_model];
        let trans = vec![0.0_f64; d_model];
        let logits = lens.tuned_lens_step(&res, 0, &trans);
        assert_eq!(logits.len(), vocab);
    }

    // ── SparseAutoencoder (MI) ────────────────────────────────────────────────

    #[test]
    fn test_sae_encode_nonnegative() {
        let sae = small_sae();
        let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
        let feats = sae.encode(&x);
        assert_eq!(feats.len(), sae.config.d_hidden);
        for &f in &feats {
            assert!(f >= 0.0, "ReLU output must be non-negative");
        }
    }

    #[test]
    fn test_sae_decode_shape() {
        let sae = small_sae();
        let feats = vec![0.5_f64; sae.config.d_hidden];
        let recon = sae.decode(&feats);
        assert_eq!(recon.len(), sae.config.d_input);
    }

    #[test]
    fn test_sae_forward_shapes() {
        let sae = small_sae();
        let x: Vec<f64> = vec![0.1; 8];
        let (feats, recon) = sae.forward(&x);
        assert_eq!(feats.len(), sae.config.d_hidden);
        assert_eq!(recon.len(), sae.config.d_input);
    }

    #[test]
    fn test_sae_loss_positive() {
        let sae = small_sae();
        let x: Vec<f64> = vec![0.5; 8];
        let loss = sae.loss(&x);
        assert!(loss >= 0.0, "loss must be non-negative");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_sae_train_step_loss_decreases() {
        let mut sae = small_sae();
        let data = random_data(64, 8);
        let loss0 = sae.train_step(&data);
        let loss1 = sae.train_step(&data);
        let loss2 = sae.train_step(&data);
        assert!(
            loss0.is_finite() && loss1.is_finite() && loss2.is_finite(),
            "all losses must be finite"
        );
    }

    #[test]
    fn test_sae_fit_returns_history() {
        let mut sae = small_sae();
        let data = random_data(32, 8);
        let history = sae.fit(&data, 5);
        assert_eq!(history.len(), 5);
        for &l in &history {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_sae_feature_sparsity() {
        let sae = small_sae();
        let x: Vec<f64> = vec![0.001; 8];
        let feats = sae.encode(&x);
        let nnz = feats.iter().filter(|&&f| f > 1e-6).count();
        assert!(
            nnz < sae.config.d_hidden,
            "expected sparse activation, got {nnz} / {}",
            sae.config.d_hidden
        );
    }

    #[test]
    fn test_sae_normalize_decoder_unit_columns() {
        let mut sae = small_sae();
        sae.normalize_decoder();
        let d_hid = sae.config.d_hidden;
        let d_in = sae.config.d_input;
        for j in 0..d_hid {
            let col_norm: f64 = (0..d_in)
                .map(|i| sae.w_dec[i][j].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (col_norm - 1.0).abs() < 1e-6,
                "column {j} norm = {col_norm}"
            );
        }
    }

    #[test]
    fn test_sae_active_features_subset() {
        let sae = small_sae();
        let x: Vec<f64> = vec![0.5; 8];
        let active = sae.active_features(&x);
        for &idx in &active {
            assert!(idx < sae.config.d_hidden);
        }
    }

    #[test]
    fn test_sae_feature_frequency_shape() {
        let sae = small_sae();
        let data = random_data(20, 8);
        let freq = sae.feature_frequency(&data);
        assert_eq!(freq.len(), sae.config.d_hidden);
        for &f in &freq {
            assert!((0.0..=1.0).contains(&f), "frequency out of [0,1]: {f}");
        }
    }

    #[test]
    fn test_sae_feature_frequency_empty_data() {
        let sae = small_sae();
        let freq = sae.feature_frequency(&[]);
        assert_eq!(freq.len(), sae.config.d_hidden);
        for &f in &freq {
            assert_eq!(f, 0.0);
        }
    }

    // ── AttentionAnalyzer ─────────────────────────────────────────────────────

    #[test]
    fn test_attention_patterns_shape() {
        let model = small_model();
        let n_layers = model.n_layers;
        let n_heads = model.n_heads;
        let toks = tokens();
        let seq_len = toks.len();
        let analyzer = AttentionAnalyzer::new(model);
        let pats = analyzer.attention_patterns(&toks);
        assert_eq!(pats.len(), n_layers);
        for layer_pats in &pats {
            assert_eq!(layer_pats.len(), n_heads);
            for head_pats in layer_pats {
                assert_eq!(head_pats.len(), seq_len);
                for row in head_pats {
                    assert_eq!(row.len(), seq_len);
                }
            }
        }
    }

    #[test]
    fn test_attention_patterns_causal() {
        let model = small_model();
        let toks = tokens();
        let analyzer = AttentionAnalyzer::new(model);
        let pats = analyzer.attention_patterns(&toks);
        let seq_len = toks.len();
        for layer_pats in &pats {
            for head_pats in layer_pats {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        assert!(
                            head_pats[i][j] < 1e-6,
                            "position {i} should not attend to future {j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_head_attribution_shape() {
        let model = small_model();
        let n_layers = model.n_layers;
        let n_heads = model.n_heads;
        let analyzer = AttentionAnalyzer::new(model);
        let attr = analyzer.head_attribution(&tokens(), 0, 1);
        assert_eq!(attr.len(), n_layers);
        for layer_attr in &attr {
            assert_eq!(layer_attr.len(), n_heads);
        }
    }

    #[test]
    fn test_head_attribution_finite() {
        let model = small_model();
        let analyzer = AttentionAnalyzer::new(model);
        let attr = analyzer.head_attribution(&tokens(), 0, 1);
        for layer_attr in &attr {
            for &s in layer_attr {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn test_detect_induction_heads_returns_all() {
        let model = small_model();
        let n_layers = model.n_layers;
        let n_heads = model.n_heads;
        let analyzer = AttentionAnalyzer::new(model);
        let toks = vec![0, 1, 2, 0, 1, 2, 0];
        let results = analyzer.detect_induction_heads(&toks);
        assert_eq!(results.len(), n_layers * n_heads);
    }

    #[test]
    fn test_detect_induction_heads_scores_in_01() {
        let model = small_model();
        let analyzer = AttentionAnalyzer::new(model);
        let toks = vec![3, 5, 3, 5, 3];
        let results = analyzer.detect_induction_heads(&toks);
        for (_, _, s) in &results {
            assert!(*s >= 0.0 && *s <= 1.0 + 1e-9, "score {s} out of [0, 1]");
        }
    }

    #[test]
    fn test_attention_rollout_shape() {
        let model = small_model();
        let toks = tokens();
        let seq_len = toks.len();
        let analyzer = AttentionAnalyzer::new(model);
        let rollout = analyzer.attention_rollout(&toks);
        assert_eq!(rollout.len(), seq_len);
        for row in &rollout {
            assert_eq!(row.len(), seq_len);
        }
    }

    #[test]
    fn test_attention_rollout_row_sums() {
        let model = small_model();
        let toks = tokens();
        let analyzer = AttentionAnalyzer::new(model);
        let rollout = analyzer.attention_rollout(&toks);
        for row in &rollout {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "rollout row sum = {sum}");
        }
    }

    // ── ProbeClassifier ───────────────────────────────────────────────────────

    #[test]
    fn test_probe_predict_valid_class() {
        let config = ProbeConfig {
            d_input: 8,
            n_classes: 3,
            learning_rate: 0.1,
            n_epochs: 5,
        };
        let probe = ProbeClassifier::new(config);
        let x = vec![0.1_f64; 8];
        let pred = probe.predict(&x);
        assert!(pred < 3);
    }

    #[test]
    fn test_probe_predict_proba_sums_to_one() {
        let config = ProbeConfig {
            d_input: 8,
            n_classes: 4,
            learning_rate: 0.1,
            n_epochs: 5,
        };
        let probe = ProbeClassifier::new(config);
        let x = vec![0.5_f64; 8];
        let proba = probe.predict_proba(&x);
        assert_eq!(proba.len(), 4);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_probe_fit_returns_history() {
        let config = ProbeConfig {
            d_input: 4,
            n_classes: 2,
            learning_rate: 0.05,
            n_epochs: 10,
        };
        let mut probe = ProbeClassifier::new(config);
        let acts: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![if i % 2 == 0 { 1.0 } else { -1.0 }; 4])
            .collect();
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let hist = probe.fit(&acts, &labels);
        assert_eq!(hist.len(), 10);
        for &l in &hist {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_probe_accuracy_linearly_separable() {
        let config = ProbeConfig {
            d_input: 4,
            n_classes: 2,
            learning_rate: 0.1,
            n_epochs: 100,
        };
        let mut probe = ProbeClassifier::new(config);
        let mut acts: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();
        for i in 0..40 {
            let v = if i % 2 == 0 { 2.0 } else { -2.0 };
            acts.push(vec![v, 0.0, 0.0, 0.0]);
            labels.push(i % 2);
        }
        probe.fit(&acts, &labels);
        let acc = probe.accuracy(&acts, &labels);
        assert!(
            acc > 0.5,
            "probe accuracy {acc:.2} should beat chance on separable data"
        );
    }

    #[test]
    fn test_probe_accuracy_empty() {
        let config = ProbeConfig {
            d_input: 4,
            n_classes: 2,
            learning_rate: 0.1,
            n_epochs: 5,
        };
        let probe = ProbeClassifier::new(config);
        assert_eq!(probe.accuracy(&[], &[]), 0.0);
    }

    // ── MiMetrics / MiReport ──────────────────────────────────────────────────

    #[test]
    fn test_compute_mi_metrics_faithfulness_range() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let m = compute_mi_metrics(&patcher, &tokens(), &[5, 6, 7, 8, 9]);
        assert!(
            m.faithfulness >= 0.0 && m.faithfulness <= 1.0,
            "faithfulness = {}",
            m.faithfulness
        );
    }

    #[test]
    fn test_compute_mi_metrics_completeness_range() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let m = compute_mi_metrics(&patcher, &tokens(), &[5, 6, 7, 8, 9]);
        assert!(
            m.completeness >= 0.0 && m.completeness <= 1.0,
            "completeness = {}",
            m.completeness
        );
    }

    #[test]
    fn test_compute_mi_metrics_minimality_range() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let m = compute_mi_metrics(&patcher, &tokens(), &[5, 6, 7, 8, 9]);
        assert!(
            m.minimality >= 0.0 && m.minimality <= 1.0,
            "minimality = {}",
            m.minimality
        );
    }

    #[test]
    fn test_compute_mi_metrics_probe_accuracy_range() {
        let model = small_model();
        let patcher = ActivationPatcher::new(model);
        let m = compute_mi_metrics(&patcher, &tokens(), &[5, 6, 7, 8, 9]);
        assert!(
            m.probe_accuracy >= 0.0 && m.probe_accuracy <= 1.0,
            "probe_accuracy = {}",
            m.probe_accuracy
        );
    }

    #[test]
    fn test_build_mi_report_structure() {
        let model = small_model();
        let model2 = model.clone();
        let model3 = model.clone();
        let patcher = ActivationPatcher::new(model);
        let analyzer = AttentionAnalyzer::new(model2);
        let sae = MiSparseAutoencoder::new(MiSaeConfig {
            d_input: 8,
            d_hidden: 16,
            l1_coefficient: 0.01,
            learning_rate: 0.01,
        });
        let sae_data = random_data(10, 8);
        let report = build_mi_report(
            &patcher,
            &analyzer,
            &sae,
            &tokens(),
            &[5, 6, 7, 8, 9],
            &sae_data,
        );
        assert!(report.top_heads.len() <= 10);
        for &(l, h, _) in &report.top_heads {
            assert!(l < model3.n_layers);
            assert!(h < model3.n_heads);
        }
        assert!(report.top_sae_features.len() <= 10);
        for &(_, f) in &report.top_sae_features {
            assert!((0.0..=1.0 + 1e-9).contains(&f));
        }
        assert_eq!(report.causal_important_layers.len(), model3.n_layers);
    }

    #[test]
    fn test_mi_report_causal_layers_order() {
        let model = small_model();
        let model2 = model.clone();
        let patcher = ActivationPatcher::new(model);
        let analyzer = AttentionAnalyzer::new(model2);
        let sae = small_sae();
        let sae_data = random_data(5, 8);
        let report = build_mi_report(
            &patcher,
            &analyzer,
            &sae,
            &tokens(),
            &[5, 6, 7, 8, 9],
            &sae_data,
        );
        if report.causal_important_layers.len() >= 2 {
            let first = report.causal_important_layers[0].1.abs();
            let second = report.causal_important_layers[1].1.abs();
            assert!(
                first >= second - 1e-9,
                "layers should be sorted by |effect|: {first} vs {second}"
            );
        }
    }

    // ── Utility helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sum_to_one() {
        let logits = vec![1.0, 2.0, 0.5, -1.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let out = layer_norm(&x, &scale, &bias);
        let mean = out.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_embed_tokens_shape() {
        let model = small_model();
        let toks = vec![0, 5, 10, 15];
        let embs = model.embed_tokens(&toks);
        assert_eq!(embs.len(), 4);
        for e in &embs {
            assert_eq!(e.len(), model.d_model);
        }
    }

    #[test]
    fn test_mi_transformer_clone() {
        let model = small_model();
        let model2 = model.clone();
        assert_eq!(model.d_model, model2.d_model);
        assert_eq!(model.n_heads, model2.n_heads);
        assert_eq!(model.n_layers, model2.n_layers);
    }

    #[test]
    fn test_sae_type_aliases() {
        let _config = SparseAutoencoderConfig {
            d_input: 4,
            d_hidden: 8,
            l1_coefficient: 0.01,
            learning_rate: 0.001,
        };
        let _sae: SparseAutoencoder = MiSparseAutoencoder::new(SparseAutoencoderConfig {
            d_input: 4,
            d_hidden: 8,
            l1_coefficient: 0.01,
            learning_rate: 0.001,
        });
    }
}
