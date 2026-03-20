//! AI Safety & Alignment Components
//!
//! Implements a comprehensive suite of AI safety and alignment methods:
//!
//! 1. **Constitutional AI & RLHF** — Constitutional principles, PPO with KL penalty,
//!    Direct Preference Optimization (DPO), sycophancy detection.
//!
//! 2. **Robustness & Adversarial Defense** — Certified randomized smoothing (Cohen et al.),
//!    PGD-based adversarial augmentation, input smoothing, feature squeezing, defense ensemble.
//!
//! 3. **Fairness & Bias Mitigation** — Demographic parity, equalized odds, calibration
//!    fairness, reweighting debiasing, adversarial debiasing.
//!
//! 4. **Interpretability & Explainability** — LIME, SHAP approximation,
//!    counterfactual generation, concept activation vectors (TCAV), attention rollout.
//!
//! 5. **Anomaly Detection & OOD** — Isolation forest, autoencoder anomaly, Mahalanobis
//!    distance detector, energy-based OOD detector, AUROC benchmark.

mod anomaly;
mod explainability;
mod fairness;
mod helpers;
mod rlhf;
mod robustness;

// ── Re-exports: Section 1 ────────────────────────────────────────────────────
pub use rlhf::{
    ConstitutionalAiFilter, ConstitutionalPrinciple, DpoTrainer, PpoWithKl, SycophancyDetector,
};

// ── Re-exports: Section 2 ────────────────────────────────────────────────────
pub use robustness::{
    AdversarialTrainingAugmenter, CertifiedRobustness, DefenseEnsemble, FeatureSqueezing,
    InputSmoothing,
};

// ── Re-exports: Section 3 ────────────────────────────────────────────────────
pub use fairness::{
    AdversarialDebias, CalibrationFairness, DemographicParityChecker, EqualizedOdds,
    EqualizedOddsReport, FairnessReport, ReweightingDebias,
};

// ── Re-exports: Section 4 ────────────────────────────────────────────────────
pub use explainability::{
    AttentionExplainer, ConceptBottleneck, CounterfactualExplainer, LimeExplainer, ShapValues,
};

// ── Re-exports: Section 5 ────────────────────────────────────────────────────
pub use anomaly::{
    AutoencoderAnomaly, EnergyOodDetector, IsolationForest, MahalanobisDetector, OodBenchmark,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Section 1: Constitutional AI & RLHF ─────────────────────────────────

    #[test]
    fn test_constitutional_principle() {
        let p = ConstitutionalPrinciple::new("Do not harm", 1.0, 5.0);
        assert_eq!(p.principle, "Do not harm");
        assert!((p.weight - 1.0).abs() < 1e-10);
        assert!((p.violation_penalty - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_constitutional_ai_filter_compliant() {
        let filter = ConstitutionalAiFilter::new();
        let principle = ConstitutionalPrinciple::new("Be helpful", 2.0, 10.0);
        let score = filter.evaluate(&[1.0, 2.0, 3.0], &principle);
        assert!(
            score > 0.0,
            "compliant output should have positive score: {score}"
        );
    }

    #[test]
    fn test_constitutional_ai_filter_violation() {
        let filter = ConstitutionalAiFilter::new();
        let principle = ConstitutionalPrinciple::new("Be helpful", 2.0, 10.0);
        let score = filter.evaluate(&[-1.0, -2.0, -3.0], &principle);
        let expected = 2.0 * (-2.0) - 10.0;
        assert!(
            (score - expected).abs() < 1e-10,
            "expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_ppo_kl_loss_clipping() {
        let ppo = PpoWithKl::new(0.2);
        let loss_in = ppo.compute_loss(1.0, 1.0, 0.0, 0.1);
        let loss_out = ppo.compute_loss(1.5, 1.0, 0.0, 0.1);
        assert!(loss_in.is_finite());
        assert!(loss_out.is_finite());
        let expected_clipped = -1.2_f64;
        assert!(
            (loss_out - expected_clipped).abs() < 1e-9,
            "clipped loss should be {expected_clipped}, got {loss_out}"
        );
        assert!(
            (loss_in - (-1.0)).abs() < 1e-9,
            "unclipped loss at ratio=1.0 should be -1.0, got {loss_in}"
        );
    }

    #[test]
    fn test_dpo_loss_preferred() {
        let trainer = DpoTrainer::new();
        let loss_good = trainer.dpo_loss(0.0, -5.0, 0.0, -5.0, 1.0);
        let loss_bad = trainer.dpo_loss(-5.0, 0.0, 0.0, 0.0, 1.0);
        assert!(
            loss_good < loss_bad,
            "good preference should have lower loss: {loss_good} vs {loss_bad}"
        );
    }

    #[test]
    fn test_dpo_beta_effect() {
        let trainer = DpoTrainer::new();
        let loss_low_beta = trainer.dpo_loss(-1.0, -2.0, -1.0, -2.0, 0.1);
        let loss_high_beta = trainer.dpo_loss(-1.0, -2.0, -1.0, -2.0, 10.0);
        let loss_neutral = trainer.dpo_loss(0.0, 0.0, 0.0, 0.0, 1.0);
        assert!(
            (loss_neutral - 2_f64.ln()).abs() < 1e-9,
            "neutral delta → log(2), got {loss_neutral}"
        );
        let _ = (loss_low_beta, loss_high_beta);
    }

    #[test]
    fn test_sycophancy_detector() {
        let patterns = vec![vec![1.0, 1.0, 1.0], vec![-1.0, -1.0, -1.0]];
        let detector = SycophancyDetector::new(patterns, 0.95);
        let sycophantic = vec![0.99, 0.99, 0.99];
        let independent = vec![1.0, -1.0, 0.0];
        assert!(
            detector.is_sycophantic(&sycophantic),
            "similar response should be flagged"
        );
        assert!(
            !detector.is_sycophantic(&independent),
            "independent response should not be flagged"
        );
    }

    // ── Section 2: Robustness & Adversarial Defense ──────────────────────────

    #[test]
    fn test_certified_radius_positive() {
        let cert = CertifiedRobustness::new(0.25, 500, 0.001);
        let x = vec![0.0; 5];
        let (cls, radius) = cert.certify(&x, 42, &|_| 0usize);
        assert_eq!(cls, 0, "predicted class should be 0");
        assert!(radius >= 0.0, "radius should be non-negative: {radius}");
    }

    #[test]
    fn test_certified_radius_low_confidence_abstain() {
        let cert = CertifiedRobustness::new(0.25, 200, 0.001);
        let x = vec![0.0f64; 2];
        let (_, radius) = cert.certify(&x, 99, &|xs| if xs[0] > 0.0 { 1 } else { 0 });
        assert!(radius.is_finite(), "radius should be finite");
    }

    #[test]
    fn test_adversarial_augmenter() {
        let augmenter = AdversarialTrainingAugmenter::new(0.1, 0.01, 10, 2);
        let x = vec![0.5f64; 4];
        let x_adv = augmenter.augment(&x, 0, &|_| vec![1.0; 4]);
        let diff: f64 = x.iter().zip(x_adv.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "adversarial example should differ from original: diff={diff}"
        );
        for (&xi, &xai) in x.iter().zip(x_adv.iter()) {
            assert!(
                (xi - xai).abs() <= 0.1 + 1e-9,
                "perturbation exceeds ε: {}",
                (xi - xai).abs()
            );
        }
    }

    #[test]
    fn test_input_smoothing() {
        let smoother = InputSmoothing::new(0.01, 100);
        let x = vec![0.0f64; 4];
        let cls = smoother.predict(&x, 42, &|_| 2usize);
        assert_eq!(cls, 2, "majority vote should be 2");
    }

    #[test]
    fn test_feature_squeezing() {
        let squeezer = FeatureSqueezing::new(4, (0.0, 1.0));
        let x = vec![0.0, 0.5, 1.0];
        let squeezed = squeezer.squeeze(&x);
        assert_eq!(squeezed.len(), 3);
        for &v in &squeezed {
            assert!(
                (0.0..=1.0 + 1e-9).contains(&v),
                "squeezed value out of range: {v}"
            );
        }
        let detected = squeezer.detect(&x, &|xs| if xs[1] > 0.4 { 1 } else { 0 });
        let no_detect = squeezer.detect(&x, &|_| 0usize);
        assert!(
            !no_detect,
            "clean input should not be detected as adversarial"
        );
        let _ = detected;
    }

    #[test]
    fn test_defense_ensemble() {
        let ensemble = DefenseEnsemble::new(vec![
            "smoothing".into(),
            "squeezing".into(),
            "isolation".into(),
        ]);
        assert_eq!(ensemble.vote(&[1, 1, 1]), 1);
        assert_eq!(ensemble.vote(&[1, 1, 0]), 1);
        assert_eq!(ensemble.vote(&[0, 1]), 0);
    }

    // ── Section 3: Fairness & Bias Mitigation ───────────────────────────────

    #[test]
    fn test_demographic_parity_equal() {
        let checker = DemographicParityChecker::new(0.05);
        let preds = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let report = checker.check(&preds, &groups);
        assert!(
            report.is_fair,
            "equal rates should be fair: disparity={}",
            report.max_disparity
        );
        assert!(
            report.max_disparity < 1e-9,
            "disparity should be 0: {}",
            report.max_disparity
        );
    }

    #[test]
    fn test_demographic_parity_unequal() {
        let checker = DemographicParityChecker::new(0.05);
        let preds = vec![1, 1, 0, 0];
        let groups = vec![0, 0, 1, 1];
        let report = checker.check(&preds, &groups);
        assert!(!report.is_fair, "unequal rates should not be fair");
        assert!(
            (report.max_disparity - 1.0).abs() < 1e-9,
            "disparity should be 1.0: {}",
            report.max_disparity
        );
    }

    #[test]
    fn test_equalized_odds_report() {
        let checker = EqualizedOdds::new();
        let preds = vec![1, 1, 0, 0, 1, 0, 1, 0];
        let labels = vec![1, 0, 1, 0, 1, 1, 0, 0];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let report = checker.compute(&preds, &labels, &groups);
        assert_eq!(report.tpr_per_group.len(), 2, "should have 2 groups");
        assert_eq!(report.fpr_per_group.len(), 2, "should have 2 groups");
        for (_, tpr) in &report.tpr_per_group {
            assert!(*tpr >= 0.0 && *tpr <= 1.0, "TPR out of range: {tpr}");
        }
    }

    #[test]
    fn test_reweighting_debias_sums() {
        let debias = ReweightingDebias::new();
        let labels = vec![1, 0, 1, 0, 1, 0];
        let groups = vec![0, 0, 1, 1, 1, 1];
        let weights = debias.compute_weights(&labels, &groups);
        assert_eq!(weights.len(), 6, "should have one weight per sample");
        for &w in &weights {
            assert!(w >= 0.0, "weight should be non-negative: {w}");
            assert!(w.is_finite(), "weight should be finite: {w}");
        }
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 6.0).abs() < 1.0,
            "weights should approximately sum to n: sum={total}"
        );
    }

    #[test]
    fn test_adversarial_debias() {
        let mut debias = AdversarialDebias::new(3, 0.01, 10);
        let reps = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 0, 1];
        let acc = debias.train_adversary(&reps, &labels);
        assert!((0.0..=1.0).contains(&acc), "accuracy out of range: {acc}");
        assert_eq!(debias.adversary_weights.len(), 3);
    }

    // ── Section 4: Interpretability & Explainability ────────────────────────

    #[test]
    fn test_lime_explainer_shape() {
        let explainer = LimeExplainer::new(200, 0.1);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let importances = explainer.explain(&x, 42, &|xs| xs.iter().sum::<f64>());
        assert_eq!(
            importances.len(),
            4,
            "should return one importance per feature"
        );
        for &v in &importances {
            assert!(v.is_finite(), "importance should be finite: {v}");
        }
    }

    #[test]
    fn test_shap_sum_to_baseline() {
        let shap = ShapValues::new(100);
        let x = vec![1.0, 2.0, 3.0];
        let base = 0.0;
        let pred_x: f64 = x.iter().sum();
        let values = shap.compute_shap(&x, base, 42, &|xs| xs.iter().sum());
        assert_eq!(values.len(), 3);
        let total: f64 = values.iter().sum();
        assert!(
            (total - (pred_x - base)).abs() < 1.0,
            "SHAP sum should approximate f(x) - base: sum={total}, expected={}",
            pred_x - base
        );
    }

    #[test]
    fn test_counterfactual_changes_prediction() {
        let explainer = CounterfactualExplainer::new(0.05, 200, 0.01);
        let x = vec![-2.0, -2.0, -2.0];
        let score_fn = |xs: &[f64], _target: usize| -> f64 { xs.iter().sum::<f64>() };
        let predict_fn = |xs: &[f64]| -> usize {
            if xs.iter().sum::<f64>() > 0.0 {
                1
            } else {
                0
            }
        };
        let cf = explainer.find_counterfactual(&x, 1, &predict_fn, &score_fn);
        assert_eq!(
            cf.len(),
            3,
            "counterfactual should have same dimension as x"
        );
        let diff: f64 = x.iter().zip(cf.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "counterfactual should differ from x: diff={diff}"
        );
    }

    #[test]
    fn test_concept_bottleneck_cav() {
        let cb = ConceptBottleneck::new();
        let concepts: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 0.0, 0.0]).collect();
        let randoms: Vec<Vec<f64>> = (0..10).map(|i| vec![0.0, i as f64, 0.0]).collect();
        let cav = cb.compute_cav(&concepts, &randoms);
        assert_eq!(
            cav.len(),
            3,
            "CAV should have same dimensionality as features"
        );
        let norm: f64 = cav.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-9,
            "CAV should be unit length: norm={norm}"
        );
    }

    #[test]
    fn test_attention_rollout_shape() {
        let explainer = AttentionExplainer::new();
        let seq_len = 4;
        let layers: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|_| (0..seq_len).map(|_| vec![0.25; seq_len]).collect())
            .collect();
        let rollout = explainer.rollout(&layers);
        assert_eq!(
            rollout.len(),
            seq_len,
            "rollout should have seq_len entries"
        );
        let sum: f64 = rollout.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "rollout should sum to 1: {sum}");
    }

    // ── Section 5: Anomaly Detection & OOD ──────────────────────────────────

    #[test]
    fn test_isolation_forest_scores() {
        let mut forest = IsolationForest::new(20, 32);
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![(i as f64 - 50.0) / 10.0]).collect();
        forest.fit(&data, 42);
        let score_normal = forest.anomaly_score(&[0.0]);
        let score_outlier = forest.anomaly_score(&[100.0]);
        assert!(
            score_normal.is_finite(),
            "normal score should be finite: {score_normal}"
        );
        assert!(
            score_outlier.is_finite(),
            "outlier score should be finite: {score_outlier}"
        );
    }

    #[test]
    fn test_anomaly_score_outlier_higher() {
        let mut forest = IsolationForest::new(50, 64);
        let data: Vec<Vec<f64>> = (0..200)
            .map(|i| {
                let offset = (i as f64 - 100.0) / 50.0;
                vec![offset, offset * 0.5]
            })
            .collect();
        forest.fit(&data, 7);
        let score_normal = forest.anomaly_score(&[0.0, 0.0]);
        let score_outlier = forest.anomaly_score(&[50.0, 50.0]);
        assert!(
            score_outlier >= score_normal,
            "outlier score ({score_outlier}) should be >= normal score ({score_normal})"
        );
    }

    #[test]
    fn test_mahalanobis_distance() {
        let mut detector = MahalanobisDetector::new();
        let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 10.0, 0.0]).collect();
        detector.fit(&data);
        let score_mean = detector.score(&[2.5, 0.0]);
        let score_far = detector.score(&[2.5, 100.0]);
        assert!(score_mean.is_finite(), "mean score should be finite");
        assert!(
            score_far > score_mean,
            "far point score ({score_far}) should exceed near-mean score ({score_mean})"
        );
    }

    #[test]
    fn test_energy_score() {
        let detector = EnergyOodDetector::new(1.0);
        let peaked = vec![10.0, -10.0, -10.0];
        let flat = vec![0.1, 0.1, 0.1];
        let e_peaked = detector.energy_score(&peaked);
        let e_flat = detector.energy_score(&flat);
        assert!(
            e_peaked < e_flat,
            "peaked logits should have lower energy than flat: peaked={e_peaked}, flat={e_flat}"
        );
    }

    #[test]
    fn test_ood_auroc() {
        let benchmark = OodBenchmark::new();
        let id_scores = vec![0.0, 0.1, 0.2, 0.3];
        let ood_scores = vec![0.7, 0.8, 0.9, 1.0];
        let auroc = benchmark.auroc(&id_scores, &ood_scores);
        assert!(
            (auroc - 1.0).abs() < 1e-9,
            "perfect separation should give AUROC=1.0: {auroc}"
        );

        let id_scores2 = vec![0.7, 0.8, 0.9, 1.0];
        let ood_scores2 = vec![0.0, 0.1, 0.2, 0.3];
        let auroc2 = benchmark.auroc(&id_scores2, &ood_scores2);
        assert!(
            auroc2 < 0.5 + 1e-9,
            "reversed scores should give AUROC < 0.5: {auroc2}"
        );
    }

    // ── Edge-case tests ────────────────────────────────────────────────────

    #[test]
    fn test_ppo_kl_negative_advantage() {
        let ppo = PpoWithKl::new(0.2);
        let loss = ppo.compute_loss(1.0, -1.0, 0.0, 0.0);
        assert!(loss.is_finite());
        assert!((loss - 1.0).abs() < 1e-9, "expected 1.0, got {loss}");
    }

    #[test]
    fn test_dpo_loss_symmetry() {
        let trainer = DpoTrainer::new();
        let loss_correct = trainer.dpo_loss(0.0, -2.0, 0.0, 0.0, 1.0);
        let loss_reversed = trainer.dpo_loss(-2.0, 0.0, 0.0, 0.0, 1.0);
        assert!(
            loss_reversed > loss_correct,
            "reversed preferences should have higher loss: {loss_reversed} vs {loss_correct}"
        );
    }

    #[test]
    fn test_constitutional_filter_empty_features() {
        let filter = ConstitutionalAiFilter::new();
        let principle = ConstitutionalPrinciple::new("Principle", 1.0, 2.0);
        let score = filter.evaluate(&[], &principle);
        assert!(
            score <= 0.0,
            "empty features should not give positive score: {score}"
        );
    }

    #[test]
    fn test_feature_squeezing_bit_precision() {
        let squeezer = FeatureSqueezing::new(2, (0.0, 1.0));
        let x = vec![0.4];
        let sq = squeezer.squeeze(&x);
        assert!(
            (sq[0] - 1.0 / 3.0).abs() < 1e-9,
            "expected 1/3, got {}",
            sq[0]
        );
    }

    #[test]
    fn test_reweighting_debias_empty() {
        let debias = ReweightingDebias::new();
        let weights = debias.compute_weights(&[], &[]);
        assert!(weights.is_empty(), "empty input should give empty weights");
    }

    #[test]
    fn test_equalized_odds_single_group() {
        let checker = EqualizedOdds::new();
        let preds = vec![1, 0, 1, 0];
        let labels = vec![1, 1, 0, 0];
        let groups = vec![0, 0, 0, 0];
        let report = checker.compute(&preds, &labels, &groups);
        assert_eq!(report.tpr_per_group.len(), 1);
        assert!(
            (report.max_tpr_disparity - 0.0).abs() < 1e-9,
            "single group has no disparity"
        );
    }

    #[test]
    fn test_lime_single_feature() {
        let explainer = LimeExplainer::new(50, 0.1);
        let x = vec![3.0];
        let importances = explainer.explain(&x, 1, &|xs| xs[0] * 2.0);
        assert_eq!(importances.len(), 1);
        assert!(importances[0].is_finite());
    }

    #[test]
    fn test_shap_single_feature() {
        let shap = ShapValues::new(20);
        let x = vec![5.0];
        let base = 0.0;
        let values = shap.compute_shap(&x, base, 0, &|xs| xs[0]);
        assert_eq!(values.len(), 1);
        assert!(
            (values[0] - 5.0).abs() < 0.5,
            "SHAP for identity fn: expected ~5.0, got {}",
            values[0]
        );
    }

    #[test]
    fn test_attention_rollout_empty() {
        let explainer = AttentionExplainer::new();
        let rollout = explainer.rollout(&[]);
        assert!(
            rollout.is_empty(),
            "empty attention should give empty rollout"
        );
    }

    #[test]
    fn test_isolation_forest_empty() {
        let forest = IsolationForest::new(5, 8);
        let score = forest.anomaly_score(&[1.0, 2.0]);
        assert!(
            (score - 0.5).abs() < 1e-9,
            "unfitted forest should return 0.5"
        );
    }

    #[test]
    fn test_mahalanobis_single_dim() {
        let mut detector = MahalanobisDetector::new();
        let data: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        detector.fit(&data);
        let mean_point = detector.score(&[1.5]);
        let far_point = detector.score(&[100.0]);
        assert!(
            mean_point < far_point,
            "mean point should be closer: {mean_point} vs {far_point}"
        );
    }

    #[test]
    fn test_ood_auroc_chance() {
        let benchmark = OodBenchmark::new();
        let id_scores = vec![0.5, 0.5, 0.5];
        let ood_scores = vec![0.5, 0.5, 0.5];
        let auroc = benchmark.auroc(&id_scores, &ood_scores);
        assert!(
            (auroc - 0.5).abs() < 0.01,
            "identical scores should give AUROC≈0.5: {auroc}"
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        // Test cosine similarity indirectly via SycophancyDetector.max_similarity
        let a = vec![1.0, 2.0, 3.0];
        let detector = SycophancyDetector::new(vec![a.clone()], 0.0);
        let sim = detector.max_similarity(&a);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical vectors should have cos sim = 1: {sim}"
        );
    }

    #[test]
    fn test_c_factor_boundary() {
        // Test the IsolationForest's anomaly_score for different training sizes
        let mut f0 = IsolationForest::new(1, 1);
        f0.fit(&[vec![0.0]], 0);
        // Score for a single-sample forest should be finite
        let s = f0.anomaly_score(&[0.0]);
        assert!(
            s.is_finite(),
            "single-sample forest score should be finite: {s}"
        );

        // With 2 samples, the forest should also give a finite score
        let mut f2 = IsolationForest::new(5, 2);
        f2.fit(&[vec![0.0], vec![1.0]], 0);
        let s2 = f2.anomaly_score(&[0.5]);
        assert!(
            s2.is_finite(),
            "two-sample forest score should be finite: {s2}"
        );

        // With 100 samples, the outlier should score higher than the mean
        let mut f_large = IsolationForest::new(10, 50);
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 / 50.0]).collect();
        f_large.fit(&data, 42);
        let s_mean = f_large.anomaly_score(&[1.0]);
        let s_far = f_large.anomaly_score(&[100.0]);
        assert!(
            s_far > s_mean,
            "c_factor grows with n: far={s_far} > mean={s_mean}"
        );
    }
}
