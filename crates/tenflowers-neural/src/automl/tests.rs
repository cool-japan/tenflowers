//! Tests for the automl module (original + new meta-feature tests).

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── Original tests ─────────────────────────────────────────────────────────────

#[test]
fn test_supernet_layer_forward() {
    let layer = SupernetLayer::new(4, 3, 42).expect("layer");
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.forward(&x, 0).expect("identity");
    assert_eq!(out.len(), 3);
    assert!((out[0] - 1.0).abs() < 1e-12);
    assert_eq!(layer.forward(&x, 1).expect("conv3x3").len(), 3);
    assert_eq!(layer.forward(&x, 2).expect("conv5x5").len(), 3);
    assert!(layer
        .forward(&x, 3)
        .expect("skip")
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn test_supernet_layer_invalid_op() {
    let layer = SupernetLayer::new(4, 3, 1).expect("layer");
    assert!(layer.forward(&[1.0, 2.0, 3.0, 4.0], 99).is_err());
}

#[test]
fn test_single_path_sampling() {
    let layers: Vec<_> = (0..3)
        .map(|i| SupernetLayer::new(4, 4, i as u64).expect("l"))
        .collect();
    let nas = SinglePathOneShot::new(layers);
    let mut rng = StdRng::seed_from_u64(7);
    let path = nas.sample_architecture(&mut rng);
    assert_eq!(path.len(), 3);
    assert!(path.iter().all(|&p| p < 4));
}

#[test]
fn test_single_path_forward() {
    let layers: Vec<_> = (0..2)
        .map(|i| SupernetLayer::new(4, 4, i as u64 + 10).expect("l"))
        .collect();
    let out = SinglePathOneShot::new(layers)
        .forward_path(&[1.0, 0.0, 0.0, 0.0], &[0, 0])
        .expect("fwd");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_proxyless_binarize() {
    let layer = SupernetLayer::new(4, 4, 1).expect("l");
    let p = ProxylessNas::new(vec![layer], true, 42).expect("p");
    assert_eq!(p.binarize(&[vec![0.1, 0.9, 0.2, 0.3]]), vec![1]);
}

#[test]
fn test_proxyless_forward_hard() {
    let layers: Vec<_> = (0..2)
        .map(|i| SupernetLayer::new(3, 3, i as u64 + 100).expect("l"))
        .collect();
    assert_eq!(
        ProxylessNas::new(layers, true, 7)
            .expect("p")
            .forward(&[1.0, 0.0, 0.0])
            .expect("fwd")
            .len(),
        3
    );
}

#[test]
fn test_proxyless_forward_soft() {
    let layers: Vec<_> = (0..2)
        .map(|i| SupernetLayer::new(3, 3, i as u64 + 200).expect("l"))
        .collect();
    assert_eq!(
        ProxylessNas::new(layers, false, 99)
            .expect("p")
            .forward(&[1.0, 0.0, 0.0])
            .expect("fwd")
            .len(),
        3
    );
}

#[test]
fn test_gradient_based_nas() {
    let layers: Vec<_> = (0..3)
        .map(|i| SupernetLayer::new(4, 4, i as u64 + 300).expect("l"))
        .collect();
    let out = GradientBasedNas::new(layers, 55)
        .expect("g")
        .forward(&[1.0, -1.0, 0.5, 0.2])
        .expect("fwd");
    assert_eq!(out.len(), 4);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_architecture_decoder() {
    let d = ArchitectureDecoder::new();
    assert_eq!(
        d.argmax_decode(&[vec![0.1, 0.7, 0.2], vec![0.8, 0.1, 0.1]]),
        vec![1, 0]
    );
}

#[test]
fn test_architecture_decoder_gumbel() {
    let d = ArchitectureDecoder::new();
    let mut rng = StdRng::seed_from_u64(123);
    let weights: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0_f64; 4]).collect();
    let dec = d.gumbel_top1_decode(&weights, &mut rng);
    assert_eq!(dec.len(), 3);
    assert!(dec.iter().all(|&d| d < 4));
}

#[test]
fn test_bayesian_optimizer_observe() {
    let mut opt = BayesianOptimizer::new(1.0, 0.01, 0.01, 42);
    opt.observe(vec![0.5, 0.5], 1.0);
    opt.observe(vec![0.2, 0.8], 0.5);
    assert_eq!(opt.observations_y.len(), 2);
}

#[test]
fn test_bayesian_suggest_in_bounds() {
    let mut opt = BayesianOptimizer::new(1.0, 0.01, 0.01, 7);
    opt.observe(vec![0.1, 0.9], 2.0);
    let x = opt.suggest(&[(0.0, 1.0), (0.0, 1.0)]).expect("suggest");
    assert_eq!(x.len(), 2);
    assert!(x[0] >= 0.0 && x[0] <= 1.0 && x[1] >= 0.0 && x[1] <= 1.0);
}

#[test]
fn test_bayesian_suggest_no_obs() {
    let mut opt = BayesianOptimizer::new(1.0, 0.01, 0.01, 3);
    assert_eq!(opt.suggest(&[(0.0, 1.0)]).expect("suggest").len(), 1);
}

#[test]
fn test_tpe_sampler_bounds() {
    let tpe = TpeSampler::new(0.25, 42);
    let hist = vec![
        (vec![0.1, 0.2], 0.5),
        (vec![0.3, 0.4], 0.3),
        (vec![0.5, 0.6], 0.8),
        (vec![0.7, 0.8], 0.2),
    ];
    let x = tpe.sample(&hist, &[(0.0, 1.0), (0.0, 1.0)]).expect("tpe");
    assert_eq!(x.len(), 2);
    assert!(x[0] >= 0.0 && x[0] <= 1.0 && x[1] >= 0.0 && x[1] <= 1.0);
}

#[test]
fn test_tpe_sampler_empty_history() {
    assert_eq!(
        TpeSampler::new(0.25, 1)
            .sample(&[], &[(0.0, 2.0)])
            .expect("tpe")
            .len(),
        1
    );
}

#[test]
fn test_hyperband_bracket() {
    let b = HyperBandBracket::new(81, 3.0, 4);
    let cfgs: Vec<Config> = (0..9).map(|i| Config::new(vec![i as f64])).collect();
    let s = b.run_bracket(&cfgs, 81);
    assert!(!s.is_empty() && s.len() <= cfgs.len());
}

#[test]
fn test_hyperband_bracket_single() {
    let s = HyperBandBracket::new(9, 3.0, 2).run_bracket(&[Config::new(vec![5.0])], 9);
    assert_eq!(s.len(), 1);
}

#[test]
fn test_multi_objective_pareto() {
    let mut opt = MultiObjectiveOptimizer::new(4, 1);
    let pop = vec![
        (Config::new(vec![1.0]), vec![0.1, 0.9]),
        (Config::new(vec![2.0]), vec![0.5, 0.5]),
        (Config::new(vec![3.0]), vec![0.9, 0.1]),
        (Config::new(vec![4.0]), vec![0.8, 0.8]),
    ];
    assert!(opt.pareto_front(&pop).len() >= 2);
}

#[test]
fn test_multi_objective_evolve() {
    let mut opt = MultiObjectiveOptimizer::new(3, 7);
    let pop = vec![
        (Config::new(vec![0.1]), vec![0.1, 0.9]),
        (Config::new(vec![0.5]), vec![0.5, 0.5]),
        (Config::new(vec![0.9]), vec![0.9, 0.1]),
        (Config::new(vec![0.8]), vec![0.8, 0.8]),
    ];
    assert_eq!(opt.evolve(&pop).len(), 3);
}

#[test]
fn test_early_stopping_rule() {
    let mut rule = EarlyStoppingRule::new();
    rule.record(1, 10, 0.8);
    rule.record(2, 10, 0.9);
    rule.record(3, 10, 0.7);
    assert!(rule.should_stop(4, 10, 0.5));
    assert!(!rule.should_stop(5, 10, 0.95));
}

#[test]
fn test_early_stopping_no_history() {
    assert!(!EarlyStoppingRule::new().should_stop(1, 5, 0.0));
}

#[test]
fn test_synflow_positive() {
    let w = vec![vec![vec![1.0, -2.0], vec![0.5, 3.0]], vec![vec![-1.0, 1.5]]];
    let s = SynflowScore::new().compute(&w);
    assert!(s > 0.0);
    assert!((s - 9.0).abs() < 1e-10);
}

#[test]
fn test_synflow_empty() {
    assert_eq!(SynflowScore::new().compute(&[]), 0.0);
}

#[test]
fn test_grad_norm_score() {
    let s = GradNormScore::new().compute(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]);
    assert!((s - (0.01_f64 + 0.16 + 0.81).sqrt()).abs() < 1e-10);
}

#[test]
fn test_naswot_score_positive() {
    let pats = vec![
        vec![true, false, true, false],
        vec![false, true, false, true],
        vec![true, true, false, false],
    ];
    assert!(NaswotScore::new().compute(&pats).is_finite());
}

#[test]
fn test_naswot_empty() {
    assert_eq!(NaswotScore::new().compute(&[]), 0.0);
}

#[test]
fn test_zen_score_std() {
    let acts = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![-1.0, -2.0, -3.0],
    ];
    assert!(ZenScore::new().compute(&acts) > 0.0);
}

#[test]
fn test_zen_score_constant() {
    assert!(
        ZenScore::new()
            .compute(&[vec![5.0, 5.0], vec![5.0, 5.0]])
            .abs()
            < 1e-12
    );
}

#[test]
fn test_jacobian_score() {
    let j = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
    let s = JacobianScore::new().compute(&j);
    assert!((s - (5.0_f64).sqrt().ln()).abs() < 1e-9);
}

#[test]
fn test_feature_selector_reduces_dims() {
    let x = vec![
        vec![1.0, 2.0, 0.0, -5.0],
        vec![2.0, 4.0, 0.0, -10.0],
        vec![3.0, 6.0, 0.0, -15.0],
        vec![4.0, 8.0, 0.0, -20.0],
    ];
    let mut sel = FeatureSelector::new(2);
    sel.fit(&x, &[1.0, 2.0, 3.0, 4.0]).expect("fit");
    assert_eq!(sel.selected.len(), 2);
    assert_eq!(sel.transform(&x).expect("t")[0].len(), 2);
}

#[test]
fn test_feature_selector_k_gt_features() {
    let mut sel = FeatureSelector::new(10);
    sel.fit(&[vec![1.0, 2.0], vec![3.0, 4.0]], &[1.0, 2.0])
        .expect("fit");
    assert!(sel.selected.len() <= 2);
}

#[test]
fn test_polynomial_features_count() {
    let out = PolynomialFeatures::new(2, false)
        .transform(&[vec![1.0, 2.0, 3.0]])
        .expect("t");
    assert_eq!(out[0].len(), 9);
}

#[test]
fn test_polynomial_features_with_bias() {
    let out = PolynomialFeatures::new(2, true)
        .transform(&[vec![2.0, 3.0]])
        .expect("t");
    assert_eq!(out[0].len(), 6);
    assert!((out[0][0] - 1.0).abs() < 1e-12);
}

#[test]
fn test_auto_normalizer() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 0.5]).collect();
    let mut norm = AutoNormalizer::new();
    norm.fit(&x).expect("fit");
    let out = norm.transform(&x).expect("t");
    assert_eq!(out.len(), 20);
    assert!(norm.distribution.is_some());
}

#[test]
fn test_auto_normalizer_not_fitted() {
    assert!(AutoNormalizer::new().transform(&[vec![1.0]]).is_err());
}

#[test]
fn test_feature_interaction_search() {
    let x = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 1.0],
        vec![3.0, 6.0, 2.0],
        vec![4.0, 8.0, 4.0],
    ];
    let mut fis = FeatureInteractionSearch::new(2);
    fis.fit(&x, &[3.0, 6.0, 9.0, 12.0]).expect("fit");
    assert_eq!(fis.selected_pairs.len(), 2);
    assert_eq!(fis.transform(&x).expect("t")[0].len(), 5);
}

#[test]
fn test_feature_interaction_search_not_fitted() {
    assert!(FeatureInteractionSearch::new(3)
        .transform(&[vec![1.0, 2.0]])
        .is_err());
}

#[test]
fn test_auto_feature_pipeline() {
    let x: Vec<Vec<f64>> = (1..=5)
        .map(|i| vec![i as f64, i as f64 * 2.0, i as f64 * 0.5, -(i as f64)])
        .collect();
    let mut p = AutoFeaturePipeline::new(3);
    p.fit(&x, &[1.0, 2.0, 3.0, 4.0, 5.0]).expect("fit");
    let out = p.transform(&x).expect("t");
    assert_eq!(out.len(), x.len());
    assert!(!out[0].is_empty());
}

#[test]
fn test_graph_encoding_shape() {
    let enc = GraphEncoding::new(4, 5);
    let v = enc.encode(&[0, 1, 2, 3, 0], &[(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(v.len(), 45);
}

#[test]
fn test_graph_encoding_adjacency() {
    let enc = GraphEncoding::new(3, 3);
    let v = enc.encode(&[0, 1, 2], &[(0, 2)]);
    assert!((v[2] - 1.0).abs() < 1e-12);
}

#[test]
fn test_path_encoding() {
    let enc = PathEncoding::new(3, 4);
    let v = enc.encode(&[0, 1, 2, 0], &[(0, 1), (1, 2), (2, 3)]);
    assert_eq!(v.len(), 81);
    assert!(v.iter().any(|&b| b > 0.0));
}

#[test]
fn test_path_encoding_no_edges() {
    let v = PathEncoding::new(2, 3).encode(&[0, 1, 0], &[]);
    assert_eq!(v.len(), 8);
}

#[test]
fn test_architecture_predictor() {
    let pred = ArchitecturePredictor::new(10, 16, 42).expect("p");
    assert!(pred.predict(&[0.1_f64; 10]).expect("pred").is_finite());
}

#[test]
fn test_architecture_predictor_short_input() {
    assert!(ArchitecturePredictor::new(10, 8, 1)
        .expect("p")
        .predict(&[0.5, 0.5])
        .is_err());
}

#[test]
fn test_architecture_bank_insert() {
    let mut bank = ArchitectureBank::new();
    bank.insert(vec![1.0, 0.0, 0.0], vec![0, 1, 2], 0.92);
    bank.insert(vec![0.0, 1.0, 0.0], vec![1, 2, 0], 0.88);
    assert_eq!(bank.encodings.len(), 2);
}

#[test]
fn test_architecture_bank_nearest() {
    let mut bank = ArchitectureBank::new();
    bank.insert(vec![1.0, 0.0], vec![0, 1], 0.9);
    bank.insert(vec![0.0, 1.0], vec![1, 0], 0.8);
    let (idx, dist) = bank.nearest(&[1.0, 0.0]).expect("nearest");
    assert_eq!(idx, 0);
    assert!(dist < 1e-12);
}

#[test]
fn test_architecture_bank_empty() {
    assert!(ArchitectureBank::new().nearest(&[1.0, 0.0]).is_none());
}

#[test]
fn test_edit_distance_same() {
    assert_eq!(EditDistance::new().compute(&[0, 1, 2, 3], &[0, 1, 2, 3]), 0);
}

#[test]
fn test_edit_distance_different() {
    assert_eq!(EditDistance::new().compute(&[0, 1, 2], &[0, 3, 2]), 1);
}

#[test]
fn test_edit_distance_empty() {
    let ed = EditDistance::new();
    assert_eq!(ed.compute(&[], &[]), 0);
    assert_eq!(ed.compute(&[1, 2], &[]), 2);
    assert_eq!(ed.compute(&[], &[3]), 1);
}

#[test]
fn test_edit_distance_insert_delete() {
    assert_eq!(EditDistance::new().compute(&[1, 2, 3], &[1, 2, 3, 4, 5]), 2);
}

#[test]
fn test_softmax_sums_to_one() {
    let probs = softmax(&[1.0, 2.0, 3.0, 4.0]);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(probs.iter().all(|&p| p > 0.0));
}

#[test]
fn test_normal_cdf_midpoint() {
    assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
}

#[test]
fn test_ei_positive() {
    assert!(expected_improvement(0.5, 0.3, 1.0, 0.0) > 0.0);
}

#[test]
fn test_hyperband_multiple_brackets() {
    let hb = HyperBand::new(81, 3.0, 10);
    assert!(hb.n_brackets >= 2);
    let cfgs: Vec<Config> = (0..27).map(|i| Config::new(vec![i as f64])).collect();
    assert!(!hb.run_bracket_s(hb.n_brackets - 1, &cfgs).is_empty());
}

#[test]
fn test_crowding_distance_infinite_extremes() {
    let obj = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
    let cd = crowding_distance(&[0, 1, 2], &obj);
    assert_eq!(cd[0], f64::INFINITY);
    assert_eq!(cd[2], f64::INFINITY);
}

#[test]
fn test_non_dominated_sort_basic() {
    let obj = vec![
        vec![0.1, 0.9],
        vec![0.5, 0.5],
        vec![0.9, 0.1],
        vec![0.5, 0.8],
    ];
    let fronts = non_dominated_sort(&obj);
    assert!(!fronts.is_empty());
    let f0: std::collections::HashSet<usize> = fronts[0].iter().cloned().collect();
    assert!(f0.contains(&0) && f0.contains(&1) && f0.contains(&2));
    assert!(!f0.contains(&3));
}

// ── New tests: MetaFeature Extraction ─────────────────────────────────────────

#[test]
fn test_dataset_meta_features_basic() {
    let x: Vec<Vec<f64>> = (0..50)
        .map(|i| vec![i as f64, (i as f64).sin(), (i as f64).cos()])
        .collect();
    let labels: Vec<f64> = (0..50).map(|i| (i % 2) as f64).collect();
    let mf = DatasetMetaFeatures::extract(&x, Some(&labels)).expect("extract");
    assert_eq!(mf.n_samples, 50);
    assert_eq!(mf.n_features, 3);
    assert!(mf.class_imbalance > 0.0 && mf.class_imbalance <= 1.0);
    assert!(mf.pca_var_pc1 >= 0.0 && mf.pca_var_pc1 <= 1.0);
}

#[test]
fn test_dataset_meta_features_to_vec_length() {
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
    let mf = DatasetMetaFeatures::extract(&x, None).expect("extract");
    assert_eq!(mf.to_vec().len(), 10);
}

#[test]
fn test_dataset_meta_features_empty_error() {
    assert!(DatasetMetaFeatures::extract(&[], None).is_err());
}

#[test]
fn test_dataset_meta_features_no_labels() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let mf = DatasetMetaFeatures::extract(&x, None).expect("extract");
    // class_imbalance defaults to 1.0 when no labels
    assert!((mf.class_imbalance - 1.0).abs() < 1e-12);
}

#[test]
fn test_dataset_meta_features_imbalanced_labels() {
    let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
    // 90 class-0, 10 class-1 → ratio = 10/90 ≈ 0.11
    let labels: Vec<f64> = (0..100).map(|i| if i < 90 { 0.0 } else { 1.0 }).collect();
    let mf = DatasetMetaFeatures::extract(&x, Some(&labels)).expect("extract");
    assert!(mf.class_imbalance < 0.5);
}

#[test]
fn test_landmarking_features_basic() {
    let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i % 3) as f64]).collect();
    let labels: Vec<f64> = (0..30).map(|i| (i % 2) as f64).collect();
    let lf = LandmarkingFeatures::compute(&x, &labels, 42).expect("landmark");
    assert!(lf.decision_stump_acc >= 0.0 && lf.decision_stump_acc <= 1.0);
    assert!(lf.knn1_acc >= 0.0 && lf.knn1_acc <= 1.0);
    assert!(lf.random_tree_acc >= 0.0 && lf.random_tree_acc <= 1.0);
}

#[test]
fn test_landmarking_features_to_vec() {
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let labels: Vec<f64> = (0..20).map(|i| (i % 3) as f64).collect();
    let lf = LandmarkingFeatures::compute(&x, &labels, 7).expect("lf");
    assert_eq!(lf.to_vec().len(), 3);
}

#[test]
fn test_landmarking_features_empty_error() {
    assert!(LandmarkingFeatures::compute(&[], &[], 1).is_err());
}

#[test]
fn test_meta_feature_normalizer_fit_transform() {
    let meta_matrix: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![i as f64, i as f64 * 2.0, i as f64 * 0.5])
        .collect();
    let mut norm = MetaFeatureNormalizer::new();
    norm.fit(&meta_matrix).expect("fit");
    assert_eq!(norm.means.len(), 3);
    assert_eq!(norm.stds.len(), 3);
    let t = norm.transform(&[2.0, 4.0, 1.0]).expect("transform");
    assert_eq!(t.len(), 3);
    assert!(t.iter().all(|v| v.is_finite()));
}

#[test]
fn test_meta_feature_normalizer_not_fitted() {
    assert!(MetaFeatureNormalizer::new().transform(&[1.0, 2.0]).is_err());
}

#[test]
fn test_meta_feature_normalizer_empty_error() {
    let mut n = MetaFeatureNormalizer::new();
    assert!(n.fit(&[]).is_err());
}

#[test]
fn test_algorithm_selector_recommend() {
    let mut sel = AlgorithmSelector::new(3);
    sel.add_entry(vec![1.0, 0.0, 0.0], "random_forest".to_string(), 0.9);
    sel.add_entry(vec![0.0, 1.0, 0.0], "svm".to_string(), 0.7);
    sel.add_entry(vec![0.5, 0.5, 0.0], "gradient_boosting".to_string(), 0.85);
    let recs = sel.recommend(&[1.0, 0.0, 0.0]);
    assert!(!recs.is_empty());
    // random_forest should score highest for a similar meta-feature
    assert_eq!(recs[0].0, "random_forest");
}

#[test]
fn test_algorithm_selector_empty_portfolio() {
    let sel = AlgorithmSelector::new(5);
    assert!(sel.recommend(&[0.5, 0.5]).is_empty());
}

#[test]
fn test_algorithm_selector_cosine_similarity() {
    let mut sel = AlgorithmSelector::new(2);
    sel.add_entry(vec![1.0, 0.0], "A".to_string(), 0.8);
    sel.add_entry(vec![0.0, 1.0], "B".to_string(), 0.6);
    let recs = sel.recommend(&[0.9, 0.1]);
    assert!(!recs.is_empty());
}

// ── New tests: ConfigSpace ─────────────────────────────────────────────────────

#[test]
fn test_config_space_sampling() {
    let mut space = ConfigSpace::new();
    space.add_param("lr", MfHpType::LogContinuous { lo: 1e-4, hi: 1e-1 });
    space.add_param("n_layers", MfHpType::Integer { lo: 1, hi: 5 });
    space.add_param(
        "activation",
        MfHpType::Categorical {
            choices: vec![0.0, 1.0, 2.0],
        },
    );
    let mut rng = StdRng::seed_from_u64(42);
    let cfg = space.sample(&mut rng);
    assert_eq!(cfg.len(), 3);
    assert!(cfg[0] >= 1e-4 && cfg[0] <= 1e-1);
    assert!(cfg[1] >= 1.0 && cfg[1] <= 5.0);
    assert!([0.0, 1.0, 2.0].contains(&cfg[2]));
}

#[test]
fn test_config_space_forbidden_clause() {
    let mut space = ConfigSpace::new();
    space.add_param(
        "a",
        MfHpType::Categorical {
            choices: vec![0.0, 1.0],
        },
    );
    space.add_param(
        "b",
        MfHpType::Categorical {
            choices: vec![0.0, 1.0],
        },
    );
    // Forbid (a=1, b=1)
    space.add_forbidden(vec![(0, 1.0), (1, 1.0)]);
    assert!(space.is_forbidden(&[1.0, 1.0]));
    assert!(!space.is_forbidden(&[0.0, 1.0]));
    assert!(!space.is_forbidden(&[1.0, 0.0]));
}

#[test]
fn test_config_space_dim() {
    let mut space = ConfigSpace::new();
    space.add_param("x", MfHpType::Continuous { lo: 0.0, hi: 1.0 });
    space.add_param("y", MfHpType::Continuous { lo: -1.0, hi: 1.0 });
    assert_eq!(space.dim(), 2);
}

#[test]
fn test_smac_optimizer_warm_up() {
    let mut space = ConfigSpace::new();
    space.add_param("lr", MfHpType::Continuous { lo: 0.001, hi: 0.1 });
    space.add_param("wd", MfHpType::Continuous { lo: 0.0, hi: 0.01 });
    let mut smac = SmacOptimizer::new(space, 99);
    let cfg = smac.suggest().expect("suggest");
    assert_eq!(cfg.len(), 2);
}

#[test]
fn test_smac_optimizer_with_history() {
    let mut space = ConfigSpace::new();
    space.add_param("lr", MfHpType::Continuous { lo: 0.001, hi: 0.1 });
    let mut smac = SmacOptimizer::new(space, 7);
    for i in 0..6 {
        smac.observe(vec![0.001 * (i + 1) as f64], 0.7 + i as f64 * 0.02);
    }
    let cfg = smac.suggest().expect("suggest");
    assert_eq!(cfg.len(), 1);
    assert!(cfg[0] >= 0.001 && cfg[0] <= 0.1);
}

// ── New tests: AlgorithmPerformancePredictor ───────────────────────────────────

#[test]
fn test_algorithm_performance_predictor_forward() {
    let pred = AlgorithmPerformancePredictor::new(5, 8, 42).expect("pred");
    let out = pred.predict(&[0.5; 5]).expect("predict");
    assert!(out.is_finite());
}

#[test]
fn test_algorithm_performance_predictor_train_step() {
    let mut pred = AlgorithmPerformancePredictor::new(4, 8, 1).expect("pred");
    for i in 0..10 {
        pred.add_example(vec![0.1 * i as f64; 4], 0.5 + 0.01 * i as f64);
    }
    let loss = pred.train_step(0.01, 4, 77).expect("train");
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_algorithm_performance_predictor_short_input_error() {
    let pred = AlgorithmPerformancePredictor::new(8, 16, 5).expect("pred");
    assert!(pred.predict(&[0.5; 3]).is_err());
}

// ── New tests: PortfolioSelector ──────────────────────────────────────────────

#[test]
fn test_portfolio_selector_build() {
    // 4 datasets, 3 algorithms
    let perf_matrix = vec![
        vec![0.8, 0.6, 0.9],
        vec![0.7, 0.85, 0.6],
        vec![0.9, 0.75, 0.7],
        vec![0.6, 0.9, 0.8],
    ];
    let names: Vec<String> = ["rf", "svm", "gb"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut sel = PortfolioSelector::new(2);
    sel.build(&perf_matrix, &names).expect("build");
    assert_eq!(sel.portfolio.len(), 2);
    // Both selected should be in the names list
    for alg in &sel.portfolio {
        assert!(names.contains(alg));
    }
}

#[test]
fn test_portfolio_selector_max_size_cap() {
    let perf_matrix = vec![vec![0.9, 0.8], vec![0.7, 0.95]];
    let names: Vec<String> = ["a", "b"].iter().map(|s| (*s).to_string()).collect();
    let mut sel = PortfolioSelector::new(5);
    sel.build(&perf_matrix, &names).expect("build");
    // Can't exceed n_algorithms
    assert!(sel.portfolio.len() <= 2);
}

#[test]
fn test_portfolio_selector_empty_error() {
    let mut sel = PortfolioSelector::new(3);
    assert!(sel.build(&[], &[]).is_err());
}

// ── New tests: CellBasedNas ────────────────────────────────────────────────────

#[test]
fn test_cell_based_nas_sample_architecture() {
    let mut nas = CellBasedNas::new(4, 2, 42);
    let arch = nas.sample_architecture();
    // 2 cells × (n_nodes-2) intermediate nodes × 2 edges = 2×2×2 = 8 entries
    assert_eq!(arch.len(), 8);
}

#[test]
fn test_cell_based_nas_encode_architecture() {
    let mut nas = CellBasedNas::new(4, 1, 7);
    let arch = nas.sample_architecture();
    let enc = CellBasedNas::encode_architecture(&arch, 4);
    // Each edge: 7 (op one-hot) + 4 (src one-hot) = 11 dims
    assert_eq!(enc.len(), arch.len() * 11);
    assert!(enc.iter().all(|v| *v == 0.0 || *v == 1.0));
}

#[test]
fn test_cell_op_from_idx() {
    assert_eq!(CellOp::from_idx(0), CellOp::SepConv3x3);
    assert_eq!(CellOp::from_idx(5), CellOp::Skip);
    assert_eq!(CellOp::from_idx(6), CellOp::Zero);
    assert_eq!(CellOp::from_idx(7), CellOp::SepConv3x3); // wraps
}

// ── New tests: EfficientNasPredictor ──────────────────────────────────────────

#[test]
fn test_efficient_nas_predictor_score_finite() {
    let mut pred = EfficientNasPredictor::new(8, 42);
    let enc = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let score = pred.score(&enc, 16);
    assert!(score.is_finite());
}

#[test]
fn test_efficient_nas_predictor_different_archs() {
    let mut pred = EfficientNasPredictor::new(8, 77);
    let enc1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let enc2 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let s1 = pred.score(&enc1, 8);
    let s2 = pred.score(&enc2, 8);
    // Scores should be valid finite numbers (may or may not differ)
    assert!(s1.is_finite() && s2.is_finite());
}

// ── New tests: ArchitectureEnsemble ───────────────────────────────────────────

#[test]
fn test_architecture_ensemble_add_truncates() {
    let mut ens = ArchitectureEnsemble::new(2);
    ens.add(vec![1.0, 0.0], 0.7);
    ens.add(vec![0.0, 1.0], 0.9);
    ens.add(vec![0.5, 0.5], 0.8);
    assert_eq!(ens.members.len(), 2);
    // Top-2 by performance: 0.9 and 0.8
    assert!(ens.members[0].1 >= ens.members[1].1);
}

#[test]
fn test_architecture_ensemble_predict() {
    let mut ens = ArchitectureEnsemble::new(3);
    ens.add(vec![1.0], 0.9);
    ens.add(vec![0.5], 0.7);
    let out = ens.predict(&[2.0], |_enc, x| Ok(x[0])).expect("predict");
    assert!((out - 2.0).abs() < 1e-10);
}

#[test]
fn test_architecture_ensemble_empty_error() {
    let ens = ArchitectureEnsemble::new(3);
    assert!(ens.predict(&[1.0], |_, _| Ok(1.0)).is_err());
}

// ── New tests: TransferNasFeatures ────────────────────────────────────────────

#[test]
fn test_transfer_nas_features_predict() {
    let mut tf = TransferNasFeatures::new();
    tf.register(vec![1.0, 0.0], vec![1.0, 0.0, 0.0], 0.9);
    tf.register(vec![0.0, 1.0], vec![0.0, 1.0, 0.0], 0.7);
    let pred = tf
        .predict_transfer(&[1.0, 0.0], &[1.0, 0.0, 0.0])
        .expect("pred");
    assert!(pred.is_finite());
    assert!(pred > 0.0);
}

#[test]
fn test_transfer_nas_features_empty_error() {
    let tf = TransferNasFeatures::new();
    assert!(tf.predict_transfer(&[1.0], &[0.5]).is_err());
}

#[test]
fn test_transfer_nas_features_low_similarity_fallback() {
    let mut tf = TransferNasFeatures::new();
    tf.register(vec![1.0, 0.0], vec![1.0, 0.0], 0.8);
    tf.register(vec![0.0, 1.0], vec![0.0, 1.0], 0.6);
    // Orthogonal query: falls back to mean
    let pred = tf.predict_transfer(&[0.0, 0.0], &[0.0, 0.0]).expect("pred");
    assert!(pred.is_finite());
}

// ── New tests: AutoMlPipeline ─────────────────────────────────────────────────

#[test]
fn test_automl_pipeline_preprocess_normalization() {
    let mut pipeline = AutoMlPipeline::new(42);
    pipeline.add_step(PipelineStep::Normalization);
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let out = pipeline.preprocess(&x).expect("preprocess");
    assert_eq!(out.len(), 3);
    // After normalization, mean should be ~0
    let col0: Vec<f64> = out.iter().map(|r| r[0]).collect();
    let mean0: f64 = col0.iter().sum::<f64>() / col0.len() as f64;
    assert!(mean0.abs() < 1e-10);
}

#[test]
fn test_automl_pipeline_preprocess_log_transform() {
    let mut pipeline = AutoMlPipeline::new(1);
    pipeline.add_step(PipelineStep::LogTransform);
    let x = vec![vec![0.0, 1.0, 4.0]];
    let out = pipeline.preprocess(&x).expect("preprocess");
    assert!((out[0][0] - 0.0_f64.ln_1p()).abs() < 1e-12);
    assert!((out[0][1] - 1.0_f64.ln_1p()).abs() < 1e-12);
}

#[test]
fn test_automl_pipeline_run_random_search() {
    let mut pipeline = AutoMlPipeline::new(77);
    let mut space = ConfigSpace::new();
    space.add_param("lr", MfHpType::Continuous { lo: 0.001, hi: 0.1 });
    space.add_param("momentum", MfHpType::Continuous { lo: 0.8, hi: 0.99 });
    pipeline.run_random_search(&space, 20).expect("search");
    assert_eq!(pipeline.n_evaluations, 20);
    assert!(pipeline.best_config.is_some());
    assert!(pipeline.best_cv_score >= 0.0 && pipeline.best_cv_score <= 1.0);
}

#[test]
fn test_automl_pipeline_evaluate_config_in_range() {
    let mut pipeline = AutoMlPipeline::new(3);
    let score = pipeline.evaluate_config(&[0.1, 0.9, 0.5]);
    assert!((0.0..=1.0).contains(&score));
}

// ── New tests: PipelineOptimizer ──────────────────────────────────────────────

#[test]
fn test_pipeline_optimizer_optimize() {
    let mut space = ConfigSpace::new();
    space.add_param("x", MfHpType::Continuous { lo: -1.0, hi: 1.0 });
    space.add_param("y", MfHpType::Continuous { lo: -1.0, hi: 1.0 });
    let mut opt = PipelineOptimizer::new(space, 42);
    opt.optimize(20, |cfg| {
        let x = cfg[0];
        let y = cfg[1];
        1.0 - x.powi(2) - y.powi(2)
    })
    .expect("optimize");
    assert_eq!(opt.history.len(), 20);
    assert!(opt.best_config.is_some());
    assert!(opt.best_score > -2.0);
}

#[test]
fn test_pipeline_optimizer_empty_space_error() {
    let space = ConfigSpace::new();
    let mut opt = PipelineOptimizer::new(space, 1);
    assert!(opt.optimize(5, |_| 0.5).is_err());
}

// ── New tests: AutoMlReport ────────────────────────────────────────────────────

#[test]
fn test_automl_report_summary() {
    let report = AutoMlReport::new("random_forest", vec![0.01, 100.0], 0.923, 42.5, 1000);
    let summary = report.summary();
    assert!(summary.contains("random_forest"));
    assert!(summary.contains("0.9230"));
    assert!(summary.contains("1000"));
}

#[test]
fn test_automl_report_fields() {
    let report = AutoMlReport::new("svm", vec![0.1], 0.85, 10.0, 50);
    assert_eq!(report.best_pipeline, "svm");
    assert_eq!(report.n_evaluations, 50);
    assert!((report.cv_score - 0.85).abs() < 1e-12);
    assert!((report.total_time_secs - 10.0).abs() < 1e-12);
}
