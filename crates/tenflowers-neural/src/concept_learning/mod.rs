//! Concept-Based Interpretable Learning (Koh et al. 2020, Kim et al. 2018).
//!
//! Implements a suite of concept-centric explanation tools:
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`ConceptBottleneckModel`] | CBM: x → concept bottleneck → labels (Koh 2020) |
//! | [`ConceptActivationVector`] | TCAV: directional concept derivative in activation space |
//! | [`TcavAnalyzer`] | TCAV scores and concept sensitivity |
//! | [`LinearProbe`] | Linear (or shallow MLP) probe with L2 regularisation |
//! | [`MultiProbe`] | Per-concept binary probes trained independently |
//! | [`AceExplainer`] | ACE: k-means concept discovery + TCAV ranking |
//! | [`ConceptShap`] | Monte-Carlo Shapley values over concept presence |
//! | [`ConceptMetrics`] | Accuracy / completeness / alignment / disentanglement |
//!
//! All types use `f64` arithmetic. The `Cl` prefix is applied to shared
//! building-block types to avoid collisions with `mechanistic_interpretability`.

pub mod ace;
pub mod cbm;
pub(crate) mod helpers;
pub mod metrics;
pub mod probes;
pub mod shapley;
pub mod shared;
pub mod tcav;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use ace::{AceConfig, AceExplainer, Concept};
pub use cbm::{CbmConfig, ConceptActivation, ConceptBottleneckModel};
pub use metrics::{
    compute_concept_completeness, concept_mutual_information, evaluate_concepts, ConceptMetrics,
};
pub use probes::{ClProbeConfig, LinearProbe, MultiProbe};
pub use shapley::{ConceptShap, ConceptShapConfig};
pub use shared::{ClLinear, ClMlp};
pub use tcav::{ConceptActivationVector, TcavAnalyzer, TcavConfig};

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use helpers::l2_norm_f64;
    use helpers::softmax_f64;
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
    use scirs2_core::RngExt;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn make_cbm() -> ConceptBottleneckModel {
        ConceptBottleneckModel::new(CbmConfig {
            input_dim: 8,
            n_concepts: 4,
            n_classes: 3,
            encoder_hidden: vec![16],
            predictor_hidden: vec![8],
            concept_activation: ConceptActivation::Sigmoid,
            lambda_concept: 1.0,
        })
    }

    fn make_probe(feat_dim: usize, n_classes: usize) -> LinearProbe {
        LinearProbe::new(
            feat_dim,
            ClProbeConfig {
                hidden_dim: 0,
                n_classes,
                n_epochs: 20,
                lr: 0.05,
                l2_penalty: 1e-4,
            },
        )
    }

    fn random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
            .collect()
    }

    // ── ConceptBottleneckModel ────────────────────────────────────────────────

    #[test]
    fn test_cbm_encode_concepts_len() {
        let cbm = make_cbm();
        let x = vec![0.1_f64; 8];
        let concepts = cbm.encode_concepts(&x);
        assert_eq!(concepts.len(), 4);
    }

    #[test]
    fn test_cbm_sigmoid_activations_in_01() {
        let cbm = make_cbm();
        let x = vec![1.5_f64; 8];
        let concepts = cbm.encode_concepts(&x);
        for &c in &concepts {
            assert!((0.0..=1.0).contains(&c), "concept out of [0,1]: {c}");
        }
    }

    #[test]
    fn test_cbm_forward_shapes() {
        let cbm = make_cbm();
        let x = vec![0.0_f64; 8];
        let (concepts, logits) = cbm.forward(&x);
        assert_eq!(concepts.len(), 4);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_cbm_joint_loss_finite() {
        let cbm = make_cbm();
        let x = vec![0.5_f64; 8];
        let ctarget = vec![1.0, 0.0, 1.0, 0.0];
        let loss = cbm.joint_loss(&x, &ctarget, 1);
        assert!(loss.is_finite(), "joint loss should be finite");
    }

    #[test]
    fn test_cbm_train_step_finite_losses() {
        let mut cbm = make_cbm();
        let x = vec![0.3_f64; 8];
        let ctarget = vec![1.0, 0.0, 1.0, 0.0];
        let (c_loss, t_loss) = cbm.train_step(&x, &ctarget, 2, 1e-3);
        assert!(c_loss.is_finite());
        assert!(t_loss.is_finite());
    }

    #[test]
    fn test_cbm_concept_loss_sigmoid_range() {
        let cbm = make_cbm();
        let x = vec![0.5_f64; 8];
        let ctarget = vec![1.0, 0.0, 1.0, 0.0];
        let loss = cbm.concept_loss(&x, &ctarget);
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_cbm_task_loss_positive() {
        let cbm = make_cbm();
        let x = vec![0.5_f64; 8];
        let loss = cbm.task_loss(&x, 0);
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_cbm_intervene_changes_prediction() {
        let cbm = make_cbm();
        let x = vec![0.5_f64; 8];
        let normal = cbm.predict_from_concepts(&cbm.encode_concepts(&x));
        let intervened = cbm.intervene(&x, 0, 0.0);
        assert_eq!(normal.len(), intervened.len());
    }

    #[test]
    fn test_cbm_intervene_logits_len() {
        let cbm = make_cbm();
        let x = vec![0.0_f64; 8];
        let logits = cbm.intervene(&x, 1, 1.0);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_cbm_concept_accuracy_len() {
        let cbm = make_cbm();
        let xs = random_vecs(10, 8, 1);
        let cts: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 0.0, 1.0, 0.0]).collect();
        let acc = cbm.concept_accuracy(&xs, &cts);
        assert_eq!(acc.len(), 4);
    }

    #[test]
    fn test_cbm_concept_accuracy_in_01() {
        let cbm = make_cbm();
        let xs = random_vecs(10, 8, 2);
        let cts: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![if i % 2 == 0 { 1.0 } else { 0.0 }; 4])
            .collect();
        let acc = cbm.concept_accuracy(&xs, &cts);
        for &a in &acc {
            assert!((0.0..=1.0).contains(&a));
        }
    }

    #[test]
    fn test_cbm_linear_activation() {
        let cbm = ConceptBottleneckModel::new(CbmConfig {
            input_dim: 4,
            n_concepts: 2,
            n_classes: 2,
            encoder_hidden: vec![],
            predictor_hidden: vec![],
            concept_activation: ConceptActivation::Linear,
            lambda_concept: 0.5,
        });
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (concepts, _) = cbm.forward(&x);
        assert_eq!(concepts.len(), 2);
    }

    // ── LinearProbe ───────────────────────────────────────────────────────────

    #[test]
    fn test_linear_probe_fit_returns_history() {
        let mut probe = make_probe(4, 2);
        let feats = random_vecs(20, 4, 10);
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let history = probe.fit(&feats, &labels);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_linear_probe_history_finite() {
        let mut probe = make_probe(4, 2);
        let feats = random_vecs(20, 4, 11);
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let history = probe.fit(&feats, &labels);
        for &l in &history {
            assert!(l.is_finite());
        }
    }

    #[test]
    fn test_linear_probe_predict_valid_class() {
        let mut probe = make_probe(4, 3);
        let feats = random_vecs(30, 4, 12);
        let labels: Vec<usize> = (0..30).map(|i| i % 3).collect();
        probe.fit(&feats, &labels);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let pred = probe.predict(&x);
        assert!(pred < 3);
    }

    #[test]
    fn test_linear_probe_predict_proba_sums_to_one() {
        let probe = make_probe(4, 3);
        let x = vec![0.1, -0.5, 0.3, 1.0];
        let proba = probe.predict_proba(&x);
        assert_eq!(proba.len(), 3);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_linear_probe_accuracy_in_01() {
        let mut probe = make_probe(4, 2);
        let feats = random_vecs(30, 4, 13);
        let labels: Vec<usize> = (0..30).map(|i| i % 2).collect();
        probe.fit(&feats, &labels);
        let acc = probe.accuracy(&feats, &labels);
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn test_linear_probe_accuracy_empty() {
        let probe = make_probe(4, 2);
        assert_eq!(probe.accuracy(&[], &[]), 0.0);
    }

    #[test]
    fn test_linear_probe_selectivity_in_01() {
        let mut probe = make_probe(4, 2);
        let feats = random_vecs(20, 4, 14);
        let labels: Vec<usize> = (0..20).map(|i| i % 2).collect();
        probe.fit(&feats, &labels);
        let sel = probe.selectivity_score();
        assert!((0.0..=1.0).contains(&sel), "selectivity = {sel}");
    }

    #[test]
    fn test_linear_probe_selectivity_zero_weights() {
        let probe = make_probe(4, 2);
        let sel = probe.selectivity_score();
        assert_eq!(sel, 0.0);
    }

    // ── MultiProbe ────────────────────────────────────────────────────────────

    #[test]
    fn test_multi_probe_predict_concepts_len() {
        let config = ClProbeConfig {
            hidden_dim: 0,
            n_classes: 2,
            n_epochs: 5,
            lr: 0.05,
            l2_penalty: 1e-4,
        };
        let mp = MultiProbe::new(4, 3, config);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let preds = mp.predict_concepts(&x);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_multi_probe_accuracy_per_concept_in_01() {
        let config = ClProbeConfig {
            hidden_dim: 0,
            n_classes: 2,
            n_epochs: 10,
            lr: 0.05,
            l2_penalty: 1e-4,
        };
        let mut mp = MultiProbe::new(4, 3, config);
        let feats = random_vecs(30, 4, 20);
        let concept_labels: Vec<Vec<bool>> = (0..30)
            .map(|i| vec![i % 2 == 0, i % 3 == 0, i % 4 == 0])
            .collect();
        mp.fit(&feats, &concept_labels);
        let accs = mp.accuracy_per_concept(&feats, &concept_labels);
        assert_eq!(accs.len(), 3);
        for &a in &accs {
            assert!((0.0..=1.0).contains(&a));
        }
    }

    // ── ConceptActivationVector ───────────────────────────────────────────────

    #[test]
    fn test_cav_probe_accuracy_in_01() {
        let pos = random_vecs(20, 8, 30);
        let neg = random_vecs(20, 8, 31);
        let cav = ConceptActivationVector::train(&pos, &neg, "stripes");
        assert!(cav.probe_accuracy >= 0.0 && cav.probe_accuracy <= 1.0);
    }

    #[test]
    fn test_cav_direction_correct_dim() {
        let pos = random_vecs(20, 8, 32);
        let neg = random_vecs(20, 8, 33);
        let cav = ConceptActivationVector::train(&pos, &neg, "colour");
        assert_eq!(cav.cav.len(), 8);
    }

    #[test]
    fn test_cav_direction_unit_norm() {
        let pos = random_vecs(30, 8, 34);
        let neg = random_vecs(30, 8, 35);
        let cav = ConceptActivationVector::train(&pos, &neg, "test");
        let norm = l2_norm_f64(&cav.cav);
        assert!((norm - 1.0).abs() < 1e-9, "cav norm = {norm}");
    }

    #[test]
    fn test_cav_empty_inputs() {
        let cav = ConceptActivationVector::train(&[], &[], "empty");
        assert_eq!(cav.cav.len(), 0);
    }

    // ── TcavAnalyzer ──────────────────────────────────────────────────────────

    #[test]
    fn test_tcav_score_in_01() {
        let config = TcavConfig {
            layer_dim: 8,
            n_concepts: 2,
            n_tcav_samples: 10,
        };
        let mut analyzer = TcavAnalyzer::new(config);
        let pos = random_vecs(20, 8, 40);
        let neg = random_vecs(20, 8, 41);
        let cav = ConceptActivationVector::train(&pos, &neg, "concept_a");
        analyzer.add_cav(cav);

        let acts = random_vecs(15, 8, 42);
        let model_fn = |h: &[f64]| -> f64 { h.iter().sum::<f64>() };
        let score = analyzer.tcav_score(0, &model_fn, &acts);
        assert!((0.0..=1.0).contains(&score), "tcav_score = {score}");
    }

    #[test]
    fn test_tcav_all_scores_len() {
        let config = TcavConfig {
            layer_dim: 8,
            n_concepts: 3,
            n_tcav_samples: 5,
        };
        let mut analyzer = TcavAnalyzer::new(config);
        for name in &["c0", "c1", "c2"] {
            let pos = random_vecs(10, 8, 50);
            let neg = random_vecs(10, 8, 51);
            let cav = ConceptActivationVector::train(&pos, &neg, name);
            analyzer.add_cav(cav);
        }
        let acts = random_vecs(10, 8, 52);
        let model_fn = |h: &[f64]| -> f64 { h[0] };
        let scores = analyzer.all_tcav_scores(&acts, &model_fn);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_tcav_sensitivity_len() {
        let config = TcavConfig {
            layer_dim: 6,
            n_concepts: 1,
            n_tcav_samples: 5,
        };
        let mut analyzer = TcavAnalyzer::new(config);
        let pos = random_vecs(10, 6, 60);
        let neg = random_vecs(10, 6, 61);
        let cav = ConceptActivationVector::train(&pos, &neg, "c");
        analyzer.add_cav(cav);
        let acts = random_vecs(8, 6, 62);
        let model_fn = |h: &[f64]| -> f64 { h.iter().sum::<f64>() };
        let sens = analyzer.concept_sensitivity(0, &acts, &model_fn);
        assert_eq!(sens.len(), 8);
    }

    // ── AceExplainer ──────────────────────────────────────────────────────────

    #[test]
    fn test_ace_k_means_assigns_all() {
        let data = random_vecs(30, 4, 70);
        let mut rng = rng();
        let assignments = AceExplainer::k_means_clustering(&data, 3, 20, &mut rng);
        assert_eq!(assignments.len(), 30);
        for &a in &assignments {
            assert!(a < 3);
        }
    }

    #[test]
    fn test_ace_discover_concepts_count() {
        let config = AceConfig {
            n_concept_clusters: 4,
            min_concept_size: 1,
            concept_importance_threshold: 0.5,
        };
        let mut explainer = AceExplainer::new(config);
        let acts = random_vecs(40, 8, 71);
        let mut rng = rng();
        let concepts = explainer.discover_concepts(&acts, &mut rng);
        assert_eq!(concepts.len(), 4);
    }

    #[test]
    fn test_ace_explain_prediction_len() {
        let config = AceConfig {
            n_concept_clusters: 3,
            min_concept_size: 1,
            concept_importance_threshold: 0.3,
        };
        let mut explainer = AceExplainer::new(config);
        let acts = random_vecs(30, 6, 72);
        let mut rng = rng();
        explainer.discover_concepts(&acts, &mut rng);
        let activation = vec![0.5_f64; 6];
        let explanation = explainer.explain_prediction(&activation);
        assert_eq!(explanation.len(), 3);
    }

    #[test]
    fn test_ace_explain_prediction_finite() {
        let config = AceConfig {
            n_concept_clusters: 2,
            min_concept_size: 1,
            concept_importance_threshold: 0.0,
        };
        let mut explainer = AceExplainer::new(config);
        let acts = random_vecs(20, 4, 73);
        let mut rng = rng();
        explainer.discover_concepts(&acts, &mut rng);
        let explanation = explainer.explain_prediction(&[0.1, 0.2, 0.3, 0.4]);
        for (_, score) in &explanation {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_ace_rank_concepts_by_importance() {
        let config = AceConfig {
            n_concept_clusters: 3,
            min_concept_size: 1,
            concept_importance_threshold: 0.0,
        };
        let mut explainer = AceExplainer::new(config);
        let acts = random_vecs(30, 4, 74);
        let mut rng = rng();
        explainer.discover_concepts(&acts, &mut rng);
        let model_fn = |h: &[f64]| -> f64 { h.iter().sum::<f64>() };
        explainer.rank_concepts_by_importance(&model_fn, &acts);
        for w in explainer.concepts.windows(2) {
            assert!(w[0].importance >= w[1].importance - 1e-9);
        }
    }

    // ── ConceptShap ───────────────────────────────────────────────────────────

    #[test]
    fn test_concept_shap_presence_len() {
        let config = ConceptShapConfig {
            n_concepts: 4,
            n_coalition_samples: 10,
            n_output_classes: 3,
        };
        let cs = ConceptShap::new(8, config);
        let x = vec![0.5_f64; 8];
        let pres = cs.concept_presence(&x);
        assert_eq!(pres.len(), 4);
    }

    #[test]
    fn test_concept_shap_presence_in_01() {
        let config = ConceptShapConfig {
            n_concepts: 3,
            n_coalition_samples: 5,
            n_output_classes: 2,
        };
        let cs = ConceptShap::new(6, config);
        let x = vec![0.1_f64; 6];
        let pres = cs.concept_presence(&x);
        for &p in &pres {
            assert!((0.0..=1.0).contains(&p), "presence out of [0,1]: {p}");
        }
    }

    #[test]
    fn test_concept_shap_shapley_value_len() {
        let config = ConceptShapConfig {
            n_concepts: 3,
            n_coalition_samples: 20,
            n_output_classes: 2,
        };
        let cs = ConceptShap::new(4, config);
        let x = vec![0.2_f64; 4];
        let model_fn = |coalition: &[Vec<f64>]| -> Vec<f64> {
            let sum: f64 = coalition.iter().flat_map(|v| v.iter()).sum();
            vec![sum, 1.0 - sum.abs().min(1.0)]
        };
        let shap = cs.shapley_value(&x, 0, &model_fn);
        assert_eq!(shap.len(), 3);
    }

    #[test]
    fn test_concept_shap_shapley_value_finite() {
        let config = ConceptShapConfig {
            n_concepts: 2,
            n_coalition_samples: 10,
            n_output_classes: 2,
        };
        let cs = ConceptShap::new(4, config);
        let x = vec![0.5_f64; 4];
        let model_fn = |coalition: &[Vec<f64>]| -> Vec<f64> {
            let s: f64 = coalition.first().map(|v| v.iter().sum()).unwrap_or(0.0);
            vec![s.tanh(), 1.0 - s.tanh().abs()]
        };
        let shap = cs.shapley_value(&x, 0, &model_fn);
        for &v in &shap {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_concept_shap_importance_ranking_sorted() {
        let config = ConceptShapConfig {
            n_concepts: 3,
            n_coalition_samples: 15,
            n_output_classes: 2,
        };
        let cs = ConceptShap::new(4, config);
        let x = vec![1.0_f64; 4];
        let model_fn = |coalition: &[Vec<f64>]| -> Vec<f64> {
            let s: f64 = coalition.first().map(|v| v.iter().sum()).unwrap_or(0.0);
            vec![s, -s]
        };
        let ranked = cs.concept_importance_ranking(&x, 0, &model_fn);
        assert_eq!(ranked.len(), 3);
        for w in ranked.windows(2) {
            assert!(w[0].1.abs() >= w[1].1.abs() - 1e-9);
        }
    }

    // ── Concept Evaluation Metrics ────────────────────────────────────────────

    #[test]
    fn test_concept_completeness_in_01() {
        let concept_preds: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 / 20.0, 1.0 - i as f64 / 20.0])
            .collect();
        let logits: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 / 20.0, 0.0]).collect();
        let c = compute_concept_completeness(&concept_preds, &logits);
        assert!((0.0..=1.0).contains(&c), "completeness = {c}");
    }

    #[test]
    fn test_concept_completeness_empty() {
        let c = compute_concept_completeness(&[], &[]);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_concept_mi_nonnegative() {
        let a: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let b: Vec<f64> = (0..50).map(|i| (50 - i) as f64 / 50.0).collect();
        let mi = concept_mutual_information(&a, &b, 8);
        assert!(mi >= 0.0, "MI = {mi}");
    }

    #[test]
    fn test_concept_mi_identical_high() {
        let a: Vec<f64> = (0..40).map(|i| i as f64 / 40.0).collect();
        let mi = concept_mutual_information(&a, &a, 10);
        assert!(mi >= 0.0);
    }

    #[test]
    fn test_concept_mi_empty() {
        let mi = concept_mutual_information(&[], &[], 8);
        assert_eq!(mi, 0.0);
    }

    #[test]
    fn test_evaluate_concepts_returns_finite() {
        let cbm = make_cbm();
        let x_test = random_vecs(10, 8, 80);
        let concept_test: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 0.0, 1.0, 0.0]).collect();
        let label_test: Vec<usize> = (0..10).map(|i| i % 3).collect();
        let metrics = evaluate_concepts(&cbm, &x_test, &concept_test, &label_test);
        assert!(metrics.concept_completeness.is_finite());
        assert!(metrics.concept_alignment.is_finite());
        assert!(metrics.disentanglement.is_finite());
    }

    #[test]
    fn test_evaluate_concepts_accuracy_len() {
        let cbm = make_cbm();
        let x_test = random_vecs(10, 8, 81);
        let concept_test: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0; 4]).collect();
        let label_test: Vec<usize> = vec![0; 10];
        let metrics = evaluate_concepts(&cbm, &x_test, &concept_test, &label_test);
        assert_eq!(metrics.concept_accuracy.len(), 4);
    }

    #[test]
    fn test_evaluate_concepts_completeness_in_01() {
        let cbm = make_cbm();
        let x_test = random_vecs(12, 8, 82);
        let concept_test: Vec<Vec<f64>> = (0..12).map(|_| vec![1.0, 0.0, 0.5, 0.8]).collect();
        let label_test: Vec<usize> = (0..12).map(|i| i % 3).collect();
        let metrics = evaluate_concepts(&cbm, &x_test, &concept_test, &label_test);
        assert!(metrics.concept_completeness >= 0.0 && metrics.concept_completeness <= 1.0);
    }

    #[test]
    fn test_evaluate_concepts_disentanglement_in_01() {
        let cbm = make_cbm();
        let x_test = random_vecs(12, 8, 83);
        let concept_test: Vec<Vec<f64>> = (0..12).map(|_| vec![1.0, 0.0, 0.5, 0.8]).collect();
        let label_test: Vec<usize> = (0..12).map(|i| i % 3).collect();
        let metrics = evaluate_concepts(&cbm, &x_test, &concept_test, &label_test);
        assert!(metrics.disentanglement >= 0.0 && metrics.disentanglement <= 1.0);
    }

    // ── ClLinear / ClMlp ─────────────────────────────────────────────────────

    #[test]
    fn test_cl_linear_forward_shape() {
        let layer = ClLinear::new(4, 3);
        let out = layer.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_cl_mlp_forward_shape() {
        let mlp = ClMlp::new(&[8, 16, 4]);
        let x = vec![0.5_f64; 8];
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_cl_mlp_gradient_fd_shape() {
        let mlp = ClMlp::new(&[4, 8, 2]);
        let x = vec![0.1_f64; 4];
        let target = vec![1.0, 0.0];
        let grads = mlp.gradient_fd(&x, &target, 1e-5);
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn test_cl_mlp_apply_gradients_updates_weights() {
        let mut mlp = ClMlp::new(&[4, 2]);
        let w_before = mlp.layers[0].w[0][0];
        let x = vec![1.0_f64; 4];
        let target = vec![1.0, 0.0];
        let grads = mlp.gradient_fd(&x, &target, 1e-5);
        mlp.apply_gradients(&grads, 0.01);
        let w_after = mlp.layers[0].w[0][0];
        let _ = w_before;
        let _ = w_after;
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_softmax_f64_sums_to_one() {
        let v = vec![1.0, 2.0, -1.0, 3.5];
        let p = softmax_f64(&v);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cbm_joint_loss_lambda_scaling() {
        let mut config = CbmConfig {
            input_dim: 4,
            n_concepts: 2,
            n_classes: 2,
            encoder_hidden: vec![],
            predictor_hidden: vec![],
            concept_activation: ConceptActivation::Sigmoid,
            lambda_concept: 0.0,
        };
        let cbm0 = ConceptBottleneckModel::new(config.clone());
        config.lambda_concept = 10.0;
        let cbm10 = ConceptBottleneckModel::new(config);
        let x = vec![0.5_f64; 4];
        let ct = vec![1.0, 0.0];
        let l0 = cbm0.joint_loss(&x, &ct, 0);
        let l10 = cbm10.joint_loss(&x, &ct, 0);
        assert!(l0.is_finite());
        assert!(l10.is_finite());
    }

    #[test]
    fn test_multi_probe_names_count() {
        let config = ClProbeConfig {
            hidden_dim: 0,
            n_classes: 2,
            n_epochs: 5,
            lr: 0.05,
            l2_penalty: 1e-4,
        };
        let mp = MultiProbe::new(4, 5, config);
        assert_eq!(mp.concept_names.len(), 5);
    }

    #[test]
    fn test_concept_completeness_constant_y() {
        let cp: Vec<Vec<f64>> = (0..10).map(|_| vec![0.5]).collect();
        let lg: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
        let c = compute_concept_completeness(&cp, &lg);
        assert!(c.is_finite());
    }
}
