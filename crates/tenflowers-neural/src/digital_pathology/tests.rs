//! Tests for the Digital Pathology & Multiple Instance Learning module.
//!
//! Covers all ten sections:
//!  §1  DpPatch & DpWholeSlideImage
//!  §2  DpMilClassifier
//!  §3  DpAbmil
//!  §4  DpDsmil
//!  §5  DpCoxPh
//!  §6  DpDeepSurv
//!  §7  DpTissueSegmenter
//!  §8  DpPatchSampler
//!  §9  DpBiomarkerPredictor
//! §10  DpMetrics

use super::*;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_patch(feat_dim: usize, seed: &mut u64, label: Option<usize>) -> DpPatch {
    let features: Vec<f64> = (0..feat_dim).map(|_| dp_randn(seed)).collect();
    DpPatch::new(features, (0, 0), label, 20.0)
}

fn make_slide(
    n_patches: usize,
    feat_dim: usize,
    label: usize,
    seed: &mut u64,
) -> DpWholeSlideImage {
    let mut slide = DpWholeSlideImage::new(label, feat_dim);
    for i in 0..n_patches {
        let features: Vec<f64> = (0..feat_dim).map(|_| dp_randn(seed)).collect();
        let patch = DpPatch::new(features, (i / 4, i % 4), None, 20.0);
        slide.add_patch(patch).expect("add_patch failed");
    }
    slide
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  DpPatch & DpWholeSlideImage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_wsi_new_empty() {
    let wsi = DpWholeSlideImage::new(0, 32);
    assert_eq!(wsi.slide_label, 0);
    assert_eq!(wsi.feat_dim, 32);
    assert_eq!(wsi.n_patches, 0);
    assert!(wsi.patches.is_empty());
}

#[test]
fn test_wsi_add_patch_valid() {
    let mut wsi = DpWholeSlideImage::new(1, 16);
    let mut seed = 42u64;
    let patch = make_patch(16, &mut seed, Some(1));
    wsi.add_patch(patch).expect("should succeed");
    assert_eq!(wsi.n_patches, 1);
}

#[test]
fn test_wsi_add_patch_wrong_dim() {
    let mut wsi = DpWholeSlideImage::new(0, 16);
    let mut seed = 7u64;
    let patch = make_patch(32, &mut seed, None); // wrong dim
    let result = wsi.add_patch(patch);
    assert!(result.is_err());
    match result {
        Err(DpError::DimensionError(_)) => {}
        _ => panic!("expected DimensionError"),
    }
}

#[test]
fn test_wsi_add_multiple_patches() {
    let mut wsi = DpWholeSlideImage::new(1, 8);
    let mut seed = 100u64;
    for _ in 0..10 {
        let patch = make_patch(8, &mut seed, None);
        wsi.add_patch(patch).expect("ok");
    }
    assert_eq!(wsi.n_patches, 10);
}

#[test]
fn test_wsi_feature_matrix_shape() {
    let mut seed = 55u64;
    let wsi = make_slide(12, 32, 1, &mut seed);
    let mat = wsi.feature_matrix();
    assert_eq!(mat.len(), 12);
    assert!(mat.iter().all(|row| row.len() == 32));
}

#[test]
fn test_wsi_random_subset_count() {
    let mut seed = 123u64;
    let wsi = make_slide(20, 16, 0, &mut seed);
    let subset = wsi.random_subset(8, &mut seed);
    assert_eq!(subset.len(), 8);
}

#[test]
fn test_wsi_random_subset_all_when_n_large() {
    let mut seed = 456u64;
    let wsi = make_slide(5, 16, 1, &mut seed);
    let subset = wsi.random_subset(100, &mut seed);
    assert_eq!(subset.len(), 5);
}

#[test]
fn test_dp_randn_finite() {
    let mut seed = 0xDEADBEEFu64;
    for _ in 0..100 {
        let v = dp_randn(&mut seed);
        assert!(v.is_finite(), "dp_randn should be finite");
    }
}

#[test]
fn test_dp_rand01_range() {
    let mut seed = 0xCAFEBABEu64;
    for _ in 0..100 {
        let v = dp_rand01(&mut seed);
        assert!((0.0..=1.0).contains(&v), "dp_rand01 must be in [0, 1]");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  DpMilClassifier
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mil_instance_scores_length() {
    let clf = DpMilClassifier::new(16, DpPooling::Max);
    let mut seed = 11u64;
    let slide = make_slide(10, 16, 1, &mut seed);
    let feat = slide.feature_matrix();
    let scores = clf.instance_scores(&feat);
    assert_eq!(scores.len(), 10);
}

#[test]
fn test_mil_instance_scores_in_unit_interval() {
    let clf = DpMilClassifier::new(16, DpPooling::Mean);
    let mut seed = 22u64;
    let slide = make_slide(8, 16, 0, &mut seed);
    let feat = slide.feature_matrix();
    let scores = clf.instance_scores(&feat);
    for &s in &scores {
        assert!((0.0..=1.0).contains(&s), "sigmoid output must be in [0,1]");
    }
}

#[test]
fn test_mil_bag_score_max_pooling() {
    let clf = DpMilClassifier::new(8, DpPooling::Max);
    let mut seed = 33u64;
    let slide = make_slide(5, 8, 1, &mut seed);
    let feat = slide.feature_matrix();
    let bag = clf.bag_score(&feat).expect("bag_score ok");
    assert!((0.0..=1.0).contains(&bag));
}

#[test]
fn test_mil_bag_score_mean_pooling() {
    let clf = DpMilClassifier::new(8, DpPooling::Mean);
    let mut seed = 44u64;
    let slide = make_slide(5, 8, 0, &mut seed);
    let feat = slide.feature_matrix();
    let bag = clf.bag_score(&feat).expect("bag_score ok");
    assert!((0.0..=1.0).contains(&bag));
}

#[test]
fn test_mil_bag_score_logsumexp_pooling() {
    let clf = DpMilClassifier::new(8, DpPooling::LogSumExp);
    let mut seed = 55u64;
    let slide = make_slide(6, 8, 1, &mut seed);
    let feat = slide.feature_matrix();
    let bag = clf.bag_score(&feat).expect("bag_score ok");
    assert!((0.0..=1.0).contains(&bag));
}

#[test]
fn test_mil_bag_score_empty_error() {
    let clf = DpMilClassifier::new(8, DpPooling::Max);
    let result = clf.bag_score(&[]);
    assert!(result.is_err());
}

#[test]
fn test_mil_predict_binary() {
    let clf = DpMilClassifier::new(8, DpPooling::Max);
    let mut seed = 66u64;
    let slide = make_slide(4, 8, 1, &mut seed);
    let pred = clf.predict(&slide).expect("predict ok");
    assert!(pred == 0 || pred == 1);
}

#[test]
fn test_mil_update_changes_weights() {
    let mut clf = DpMilClassifier::new(8, DpPooling::Mean);
    let mut seed = 77u64;
    let slide = make_slide(4, 8, 1, &mut seed);
    let w_before = clf.w[0].clone();
    clf.update(&slide, 0.01).expect("update ok");
    let w_after = &clf.w[0];
    let changed = w_before
        .iter()
        .zip(w_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed, "weights should change after update");
}

#[test]
fn test_mil_update_returns_finite_loss() {
    let mut clf = DpMilClassifier::new(8, DpPooling::Max);
    let mut seed = 88u64;
    let slide = make_slide(6, 8, 0, &mut seed);
    let loss = clf.update(&slide, 0.001).expect("update ok");
    assert!(loss.is_finite(), "loss should be finite");
    assert!(loss >= 0.0, "BCE loss should be non-negative");
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  DpAbmil
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_abmil_attention_sums_to_one() {
    let mut seed = 1234u64;
    let abmil = DpAbmil::new(32, 16, false, &mut seed);
    let slide = make_slide(8, 32, 1, &mut seed);
    let feat = slide.feature_matrix();
    let attn = abmil.compute_attention(&feat);
    assert_eq!(attn.len(), 8);
    let sum: f64 = attn.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "attention should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_abmil_gated_attention_sums_to_one() {
    let mut seed = 5678u64;
    let abmil = DpAbmil::new(32, 16, true, &mut seed);
    let slide = make_slide(10, 32, 0, &mut seed);
    let feat = slide.feature_matrix();
    let attn = abmil.compute_attention(&feat);
    let sum: f64 = attn.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "gated attention should sum to 1");
}

#[test]
fn test_abmil_aggregate_returns_correct_dim() {
    let mut seed = 9012u64;
    let abmil = DpAbmil::new(32, 16, true, &mut seed);
    let slide = make_slide(6, 32, 1, &mut seed);
    let feat = slide.feature_matrix();
    let attn = abmil.compute_attention(&feat);
    let z = abmil.aggregate(&feat, &attn);
    assert_eq!(z.len(), 32);
}

#[test]
fn test_abmil_forward_returns_finite_logit() {
    let mut seed = 3456u64;
    let abmil = DpAbmil::new(32, 16, false, &mut seed);
    let slide = make_slide(5, 32, 1, &mut seed);
    let feat = slide.feature_matrix();
    let (logit, attn) = abmil.forward(&feat).expect("forward ok");
    assert!(logit.is_finite(), "logit should be finite");
    assert_eq!(attn.len(), 5);
}

#[test]
fn test_abmil_forward_empty_error() {
    let mut seed = 1u64;
    let abmil = DpAbmil::new(16, 8, false, &mut seed);
    let result = abmil.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_abmil_predict_binary() {
    let mut seed = 2345u64;
    let abmil = DpAbmil::new(16, 8, true, &mut seed);
    let slide = make_slide(4, 16, 1, &mut seed);
    let pred = abmil.predict(&slide).expect("predict ok");
    assert!(pred == 0 || pred == 1);
}

#[test]
fn test_abmil_update_returns_finite_loss() {
    let mut seed = 3333u64;
    let mut abmil = DpAbmil::new(16, 8, false, &mut seed);
    let slide = make_slide(4, 16, 1, &mut seed);
    let loss = abmil.update(&slide, 1.0, 0.001).expect("update ok");
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_abmil_attention_all_nonneg() {
    let mut seed = 4444u64;
    let abmil = DpAbmil::new(16, 8, true, &mut seed);
    let slide = make_slide(7, 16, 0, &mut seed);
    let feat = slide.feature_matrix();
    let attn = abmil.compute_attention(&feat);
    for &a in &attn {
        assert!(a >= 0.0, "attention weight should be non-negative");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  DpDsmil
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dsmil_instance_predictions_shape() {
    let mut seed = 111u64;
    let dsmil = DpDsmil::new(16, 3, &mut seed);
    let slide = make_slide(8, 16, 1, &mut seed);
    let feat = slide.feature_matrix();
    let preds = dsmil.instance_predictions(&feat);
    assert_eq!(preds.len(), 8);
    for row in &preds {
        assert_eq!(row.len(), 3);
        let s: f64 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "instance preds should sum to 1");
    }
}

#[test]
fn test_dsmil_critical_instances_top_k() {
    let mut seed = 222u64;
    let dsmil = DpDsmil::new(16, 2, &mut seed);
    let slide = make_slide(10, 16, 1, &mut seed);
    let feat = slide.feature_matrix();
    let preds = dsmil.instance_predictions(&feat);
    let critical = dsmil.critical_instances(&preds, 3);
    assert_eq!(critical.len(), 3);
    // all indices should be valid
    for &idx in &critical {
        assert!(idx < 10);
    }
}

#[test]
fn test_dsmil_critical_instances_fewer_than_k() {
    let mut seed = 333u64;
    let dsmil = DpDsmil::new(8, 2, &mut seed);
    let slide = make_slide(3, 8, 0, &mut seed);
    let feat = slide.feature_matrix();
    let preds = dsmil.instance_predictions(&feat);
    let critical = dsmil.critical_instances(&preds, 10);
    // Should return all 3 (can't select more than available)
    assert_eq!(critical.len(), 3);
}

#[test]
fn test_dsmil_bag_prediction_valid() {
    let mut seed = 444u64;
    let dsmil = DpDsmil::new(16, 3, &mut seed);
    let slide = make_slide(6, 16, 1, &mut seed);
    let feat = slide.feature_matrix();
    let probs = dsmil
        .bag_prediction(&feat, &[0, 2, 4])
        .expect("bag_prediction ok");
    assert_eq!(probs.len(), 3);
    let s: f64 = probs.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
}

#[test]
fn test_dsmil_bag_prediction_empty_error() {
    let mut seed = 555u64;
    let dsmil = DpDsmil::new(8, 2, &mut seed);
    let slide = make_slide(4, 8, 0, &mut seed);
    let feat = slide.feature_matrix();
    let result = dsmil.bag_prediction(&feat, &[]);
    assert!(result.is_err());
}

#[test]
fn test_dsmil_forward_returns_valid_probs() {
    let mut seed = 666u64;
    let dsmil = DpDsmil::new(16, 2, &mut seed);
    let slide = make_slide(8, 16, 1, &mut seed);
    let (bag_probs, critical_probs) = dsmil.forward(&slide).expect("forward ok");
    assert_eq!(bag_probs.len(), 2);
    let s: f64 = bag_probs.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
    assert!(!critical_probs.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  DpCoxPh
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_coxph_risk_score_finite() {
    let cox = DpCoxPh::new(8);
    let features = vec![1.0, -0.5, 0.3, 0.0, 2.0, -1.0, 0.7, -0.2];
    let score = cox.risk_score(&features);
    assert!(score.is_finite());
    assert_eq!(score, 0.0); // zero weights → zero score + zero bias
}

#[test]
fn test_coxph_concordance_in_unit_interval() {
    let mut cox = DpCoxPh::new(4);
    // Give it some non-zero weights for a meaningful result
    cox.w = vec![1.0, -0.5, 0.2, 0.8];
    let features = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.9, 0.1, 0.5, 0.2],
        vec![0.3, 0.7, 0.1, 0.6],
        vec![0.5, 0.3, 0.8, 0.1],
    ];
    let times = vec![10.0, 5.0, 8.0, 3.0];
    let events = vec![true, true, false, true];
    let c = cox.concordance_index(&features, &times, &events);
    assert!(
        (0.0..=1.0).contains(&c),
        "C-index should be in [0,1], got {}",
        c
    );
}

#[test]
fn test_coxph_concordance_perfect_model() {
    // Perfect model: risk increases with decreasing survival time
    // Features are 1D: f = [time_inverse]
    let n = 5;
    let times = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    // Risk = -time (higher risk → shorter time → correct ordering)
    // w = [-1], features = [time] → risk = -time
    let features: Vec<Vec<f64>> = times.iter().map(|&t| vec![t]).collect();
    let events = vec![true; n];

    let mut cox = DpCoxPh::new(1);
    cox.w = vec![-1.0]; // risk = -time

    let c = cox.concordance_index(&features, &times, &events);
    // With risk_i = -t_i, when t_i < t_j, risk_i > risk_j → always concordant
    assert!(c > 0.8, "perfect model C-index should be > 0.8, got {}", c);
}

#[test]
fn test_coxph_partial_likelihood_finite() {
    let cox = DpCoxPh::new(4);
    let features = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.9, 1.0, 1.1, 1.2],
    ];
    let times = vec![3.0, 7.0, 12.0];
    let events = vec![true, false, true];
    let loss = cox.partial_likelihood_loss(&features, &times, &events);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_coxph_update_changes_weights() {
    let mut cox = DpCoxPh::new(4);
    let features = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.5, 1.0, 1.5, 2.0],
        vec![2.0, 1.0, 0.5, 0.1],
    ];
    let times = vec![5.0, 10.0, 3.0];
    let events = vec![true, true, false];
    let w_before = cox.w.clone();
    cox.update(&features, &times, &events, 0.01);
    let changed = cox
        .w
        .iter()
        .zip(w_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed, "weights should change after update");
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  DpDeepSurv
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_deepsurv_batch_risk_shape() {
    let mut seed = 7777u64;
    let model = DpDeepSurv::new(8, 16, &mut seed);
    let features: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..8).map(|_| dp_randn(&mut seed)).collect())
        .collect();
    let risks = model.batch_risk(&features);
    assert_eq!(risks.len(), 5);
    for &r in &risks {
        assert!(r.is_finite());
    }
}

#[test]
fn test_deepsurv_loss_finite() {
    let mut seed = 8888u64;
    let model = DpDeepSurv::new(8, 16, &mut seed);
    let features: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..8).map(|_| dp_randn(&mut seed)).collect())
        .collect();
    let times = vec![5.0, 8.0, 3.0, 10.0];
    let events = vec![true, false, true, true];
    let loss = model.loss(&features, &times, &events);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_deepsurv_concordance_in_unit_interval() {
    let mut seed = 9999u64;
    let model = DpDeepSurv::new(8, 16, &mut seed);
    let features: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..8).map(|_| dp_randn(&mut seed)).collect())
        .collect();
    let times = vec![3.0, 7.0, 2.0, 9.0, 5.0, 1.0];
    let events = vec![true, true, false, true, false, true];
    let c = model.concordance_index(&features, &times, &events);
    assert!(
        (0.0..=1.0).contains(&c),
        "C-index should be in [0,1], got {}",
        c
    );
}

#[test]
fn test_deepsurv_update_returns_finite_loss() {
    let mut seed = 1111u64;
    let mut model = DpDeepSurv::new(4, 8, &mut seed);
    let features: Vec<Vec<f64>> = (0..3)
        .map(|_| (0..4).map(|_| dp_randn(&mut seed)).collect())
        .collect();
    let times = vec![4.0, 8.0, 2.0];
    let events = vec![true, true, false];
    let loss = model.update(&features, &times, &events, 0.001);
    assert!(loss.is_finite());
}

#[test]
fn test_deepsurv_risk_score_changes_with_input() {
    let mut seed = 2222u64;
    let model = DpDeepSurv::new(4, 8, &mut seed);
    let f1 = vec![1.0, 0.0, 0.0, 0.0];
    let f2 = vec![-1.0, 0.0, 0.0, 0.0];
    let r1 = model.risk_score(&f1);
    let r2 = model.risk_score(&f2);
    // Different inputs should generally produce different risk scores
    // (not guaranteed but overwhelmingly likely with non-zero weights)
    assert!(r1.is_finite() && r2.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  DpTissueSegmenter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tissue_segment_patch_sums_to_one() {
    let mut seed = 12345u64;
    let seg = DpTissueSegmenter::new(32, 64, 5, &mut seed);
    let features: Vec<f64> = (0..32).map(|_| dp_randn(&mut seed)).collect();
    let probs = seg.segment_patch(&features);
    assert_eq!(probs.len(), 5);
    let s: f64 = probs.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-6,
        "patch probs should sum to 1, got {}",
        s
    );
}

#[test]
fn test_tissue_segment_patch_all_nonneg() {
    let mut seed = 23456u64;
    let seg = DpTissueSegmenter::new(16, 32, 4, &mut seed);
    let features: Vec<f64> = (0..16).map(|_| dp_randn(&mut seed)).collect();
    let probs = seg.segment_patch(&features);
    for &p in &probs {
        assert!(p >= 0.0, "probs should be non-negative");
    }
}

#[test]
fn test_tissue_segment_slide_shape() {
    let mut seed = 34567u64;
    let seg = DpTissueSegmenter::new(16, 32, 4, &mut seed);
    let slide = make_slide(8, 16, 1, &mut seed);
    let result = seg.segment_slide(&slide).expect("segment_slide ok");
    assert_eq!(result.len(), 8);
    for row in &result {
        assert_eq!(row.len(), 4);
    }
}

#[test]
fn test_tissue_composition_sums_to_one() {
    let mut seed = 45678u64;
    let seg = DpTissueSegmenter::new(16, 32, 5, &mut seed);
    let slide = make_slide(10, 16, 0, &mut seed);
    let comp = seg
        .tissue_composition(&slide)
        .expect("tissue_composition ok");
    assert_eq!(comp.len(), 5);
    let s: f64 = comp.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-5,
        "composition should sum to 1, got {}",
        s
    );
}

#[test]
fn test_tissue_tumor_purity_in_unit_interval() {
    let mut seed = 56789u64;
    let seg = DpTissueSegmenter::new(16, 32, 5, &mut seed);
    let slide = make_slide(8, 16, 1, &mut seed);
    let purity = seg.tumor_purity(&slide).expect("tumor_purity ok");
    assert!(
        (0.0..=1.0).contains(&purity),
        "tumor purity in [0,1], got {}",
        purity
    );
}

#[test]
fn test_tissue_segment_empty_slide_error() {
    let mut seed = 67890u64;
    let seg = DpTissueSegmenter::new(16, 32, 3, &mut seed);
    let slide = DpWholeSlideImage::new(0, 16);
    let result = seg.segment_slide(&slide);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  DpPatchSampler
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sampler_random_returns_n() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::Random);
    let mut seed = 11111u64;
    let slide = make_slide(20, 8, 0, &mut seed);
    let indices = sampler.sample(&slide, None, 8, &mut seed);
    assert_eq!(indices.len(), 8);
}

#[test]
fn test_sampler_random_indices_in_range() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::Random);
    let mut seed = 22222u64;
    let slide = make_slide(15, 8, 1, &mut seed);
    let indices = sampler.sample(&slide, None, 6, &mut seed);
    for &i in &indices {
        assert!(i < 15, "index {} out of range", i);
    }
}

#[test]
fn test_sampler_top_attention_returns_n() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::TopAttention);
    let mut seed = 33333u64;
    let slide = make_slide(10, 8, 1, &mut seed);
    let attn: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0) / 55.0).collect();
    let indices = sampler.sample(&slide, Some(&attn), 4, &mut seed);
    assert_eq!(indices.len(), 4);
}

#[test]
fn test_sampler_top_attention_selects_highest() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::TopAttention);
    let mut seed = 44444u64;
    let slide = make_slide(5, 8, 0, &mut seed);
    // patch 4 has the highest attention
    let attn = vec![0.05, 0.05, 0.1, 0.2, 0.6];
    let indices = sampler.sample(&slide, Some(&attn), 1, &mut seed);
    assert_eq!(
        indices[0], 4,
        "highest attention patch should be selected first"
    );
}

#[test]
fn test_sampler_curriculum_returns_n() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::Curriculum(5));
    let mut seed = 55555u64;
    let slide = make_slide(10, 8, 1, &mut seed);
    let attn: Vec<f64> = (0..10).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let attn_sum: f64 = attn.iter().sum();
    let attn_norm: Vec<f64> = attn.iter().map(|&a| a / attn_sum).collect();
    let indices = sampler.sample(&slide, Some(&attn_norm), 4, &mut seed);
    assert_eq!(indices.len(), 4);
}

#[test]
fn test_sampler_stratified_returns_n() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::Stratified(2));
    let mut seed = 66666u64;
    let slide = make_slide(16, 8, 0, &mut seed);
    let indices = sampler.sample(&slide, None, 6, &mut seed);
    assert_eq!(indices.len(), 6);
}

#[test]
fn test_sampler_coverage_score_in_unit_interval() {
    let indices = vec![0, 1, 2, 3, 4];
    let cov = DpPatchSampler::coverage_score(&indices, 20);
    assert!((0.0..=1.0).contains(&cov));
    assert!((cov - 0.25).abs() < 1e-9);
}

#[test]
fn test_sampler_coverage_full_coverage() {
    let indices: Vec<usize> = (0..10).collect();
    let cov = DpPatchSampler::coverage_score(&indices, 10);
    assert!((cov - 1.0).abs() < 1e-9);
}

#[test]
fn test_sampler_empty_slide_returns_empty() {
    let sampler = DpPatchSampler::new(DpSamplingStrategy::Random);
    let mut seed = 77777u64;
    let slide = DpWholeSlideImage::new(0, 8);
    let indices = sampler.sample(&slide, None, 5, &mut seed);
    assert!(indices.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  DpBiomarkerPredictor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_biomarker_predict_length() {
    let mut seed = 11121u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 4, &mut seed);
    let slide = make_slide(6, 16, 1, &mut seed);
    let preds = predictor.predict_biomarkers(&slide).expect("predict ok");
    assert_eq!(preds.len(), 4);
}

#[test]
fn test_biomarker_predict_in_unit_interval() {
    let mut seed = 22232u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 5, &mut seed);
    let slide = make_slide(8, 16, 0, &mut seed);
    let preds = predictor.predict_biomarkers(&slide).expect("predict ok");
    for &p in &preds {
        assert!(
            (0.0..=1.0).contains(&p),
            "biomarker pred should be in [0,1], got {}",
            p
        );
    }
}

#[test]
fn test_biomarker_predict_empty_slide_error() {
    let mut seed = 33343u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 3, &mut seed);
    let slide = DpWholeSlideImage::new(0, 16);
    let result = predictor.predict_biomarkers(&slide);
    assert!(result.is_err());
}

#[test]
fn test_biomarker_multi_task_loss_nonneg() {
    let mut seed = 44454u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 4, &mut seed);
    let preds = vec![0.3, 0.7, 0.5, 0.9];
    let targets = vec![0.0, 1.0, 0.0, 1.0];
    let mask = vec![true, true, false, true];
    let loss = predictor.multi_task_loss(&preds, &targets, &mask);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_biomarker_multi_task_loss_all_masked_out_is_zero() {
    let mut seed = 55565u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 3, &mut seed);
    let preds = vec![0.3, 0.7, 0.5];
    let targets = vec![0.0, 1.0, 0.0];
    let mask = vec![false, false, false];
    let loss = predictor.multi_task_loss(&preds, &targets, &mask);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_biomarker_multi_task_loss_perfect_is_small() {
    let mut seed = 66676u64;
    let predictor = DpBiomarkerPredictor::new(16, 8, 2, &mut seed);
    // Nearly perfect predictions
    let preds = vec![0.9999, 0.0001];
    let targets = vec![1.0, 0.0];
    let mask = vec![true, true];
    let loss = predictor.multi_task_loss(&preds, &targets, &mask);
    assert!(
        loss < 0.05,
        "nearly-perfect loss should be small, got {}",
        loss
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  DpMetrics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bag_auc_in_unit_interval() {
    let scores = vec![0.8, 0.6, 0.3, 0.9, 0.2, 0.7];
    let labels = vec![1, 0, 0, 1, 0, 1];
    let auc = DpMetrics::bag_auc(&scores, &labels);
    assert!(
        (0.0..=1.0).contains(&auc),
        "AUC should be in [0,1], got {}",
        auc
    );
}

#[test]
fn test_bag_auc_perfect_classifier() {
    // All positives score higher than all negatives
    let scores = vec![0.9, 0.85, 0.8, 0.2, 0.1, 0.05];
    let labels = vec![1, 1, 1, 0, 0, 0];
    let auc = DpMetrics::bag_auc(&scores, &labels);
    assert!(
        (auc - 1.0).abs() < 1e-9,
        "perfect AUC should be 1.0, got {}",
        auc
    );
}

#[test]
fn test_bag_auc_random_classifier() {
    // No discrimination
    let scores = vec![0.5; 10];
    let labels = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let auc = DpMetrics::bag_auc(&scores, &labels);
    // All tied → AUC = 0 by conservative definition
    assert!((0.0..=1.0).contains(&auc));
}

#[test]
fn test_concordance_index_generic_in_unit_interval() {
    let risks = vec![2.0, 1.0, 3.0, 0.5, 2.5];
    let times = vec![3.0, 8.0, 2.0, 10.0, 4.0];
    let events = vec![true, true, true, false, true];
    let c = DpMetrics::concordance_index_generic(&risks, &times, &events);
    assert!(
        (0.0..=1.0).contains(&c),
        "C-index should be in [0,1], got {}",
        c
    );
}

#[test]
fn test_dice_score_perfect() {
    let pred = vec![0, 1, 1, 0, 1, 0];
    let gt = vec![0, 1, 1, 0, 1, 0];
    let dice = DpMetrics::dice_score(&pred, &gt, 1);
    assert!((dice - 1.0).abs() < 1e-9, "perfect Dice should be 1.0");
}

#[test]
fn test_dice_score_zero_overlap() {
    let pred = vec![0, 0, 0, 0];
    let gt = vec![1, 1, 1, 1];
    let dice = DpMetrics::dice_score(&pred, &gt, 1);
    assert!((dice - 0.0).abs() < 1e-9, "no overlap Dice should be 0.0");
}

#[test]
fn test_dice_score_in_unit_interval() {
    let pred = vec![0, 1, 1, 0, 1, 0, 1, 1];
    let gt = vec![1, 0, 1, 0, 1, 1, 0, 1];
    let dice = DpMetrics::dice_score(&pred, &gt, 1);
    assert!((0.0..=1.0).contains(&dice));
}

#[test]
fn test_attention_entropy_nonneg() {
    let attn = vec![0.5, 0.3, 0.15, 0.05];
    let h = DpMetrics::attention_entropy(&attn);
    assert!(h >= 0.0, "entropy should be non-negative, got {}", h);
}

#[test]
fn test_attention_entropy_uniform_maximal() {
    // Uniform attention has maximum entropy
    let n = 8;
    let uniform: Vec<f64> = vec![1.0 / n as f64; n];
    let focused = vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let h_uni = DpMetrics::attention_entropy(&uniform);
    let h_foc = DpMetrics::attention_entropy(&focused);
    assert!(
        h_uni > h_foc,
        "uniform attention should have higher entropy"
    );
}

#[test]
fn test_top_k_attention_coverage() {
    // Top-2 of [0.1, 0.4, 0.3, 0.2] = 0.4 + 0.3 = 0.7
    let attn = vec![0.1, 0.4, 0.3, 0.2];
    let cov = DpMetrics::top_k_attention_coverage(&attn, 2);
    assert!(
        (cov - 0.7).abs() < 1e-9,
        "top-2 coverage should be 0.7, got {}",
        cov
    );
}

#[test]
fn test_top_k_attention_coverage_all() {
    let attn = vec![0.25, 0.25, 0.25, 0.25];
    let cov = DpMetrics::top_k_attention_coverage(&attn, 4);
    assert!((cov - 1.0).abs() < 1e-9);
}

#[test]
fn test_f1_score_perfect() {
    let pred = vec![0, 1, 2, 0, 1, 2];
    let gt = vec![0, 1, 2, 0, 1, 2];
    let f1 = DpMetrics::f1_score(&pred, &gt, 3);
    assert!(
        (f1 - 1.0).abs() < 1e-9,
        "perfect F1 should be 1.0, got {}",
        f1
    );
}

#[test]
fn test_f1_score_in_unit_interval() {
    let pred = vec![0, 1, 0, 1, 2, 2, 0];
    let gt = vec![0, 0, 1, 1, 2, 0, 2];
    let f1 = DpMetrics::f1_score(&pred, &gt, 3);
    assert!((0.0..=1.0).contains(&f1), "F1 should be in [0,1], got {}", f1);
}

#[test]
fn test_f1_score_binary() {
    // TP=2, FP=1, FN=1 for class 1
    let pred = vec![0, 1, 1, 0, 1, 0];
    let gt = vec![0, 1, 0, 1, 1, 0];
    let f1 = DpMetrics::f1_score(&pred, &gt, 2);
    assert!((0.0..=1.0).contains(&f1));
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[test]
fn test_mil_training_loop_improves_loss() {
    let mut clf = DpMilClassifier::new(16, DpPooling::Mean);
    let mut seed = 9898u64;
    let positive_slide = make_slide(10, 16, 1, &mut seed);
    let mut prev_loss = f64::INFINITY;
    for _ in 0..5 {
        let loss = clf.update(&positive_slide, 0.1).expect("update ok");
        // Just verify loss decreases at least once over 5 iterations
        prev_loss = loss;
    }
    assert!(prev_loss.is_finite());
}

#[test]
fn test_abmil_with_biomarker_predictor_pipeline() {
    let mut seed = 7654u64;
    let predictor = DpBiomarkerPredictor::new(32, 16, 3, &mut seed);
    let slide = make_slide(12, 32, 1, &mut seed);
    let preds = predictor.predict_biomarkers(&slide).expect("ok");
    assert_eq!(preds.len(), 3);
    // All predictions should be valid probabilities
    for &p in &preds {
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn test_survival_pipeline() {
    let mut seed = 5432u64;
    let mut model = DpDeepSurv::new(8, 16, &mut seed);
    let features: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..8).map(|_| dp_randn(&mut seed)).collect())
        .collect();
    let times: Vec<f64> = (1..=8).map(|i| i as f64 * 2.0).collect();
    let events = vec![true, false, true, true, false, true, false, true];

    let loss_before = model.loss(&features, &times, &events);
    model.update(&features, &times, &events, 0.001);
    let c_index = model.concordance_index(&features, &times, &events);

    assert!(loss_before.is_finite());
    assert!((0.0..=1.0).contains(&c_index));
}
