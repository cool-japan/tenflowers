use super::*;

use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── TabTransformer ─────────────────────────────────────────────────────

#[test]
fn test_tab_transformer_construction() {
    let cfg = TabTransformerConfig {
        n_cat_features: 3,
        n_num_features: 2,
        cat_vocab_sizes: vec![5, 10, 3],
        embed_dim: 8,
        n_heads: 2,
        n_layers: 2,
        ffn_dim: 16,
        n_classes: 3,
    };
    let model = TabTransformer::new(cfg, 42);
    assert!(model.is_ok(), "TabTransformer construction failed");
}

#[test]
fn test_tab_transformer_forward_shape() {
    let cfg = TabTransformerConfig {
        n_cat_features: 2,
        n_num_features: 4,
        cat_vocab_sizes: vec![5, 8],
        embed_dim: 8,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 16,
        n_classes: 2,
    };
    let model = TabTransformer::new(cfg, 7).expect("TabTransformer creation should succeed");
    let out = model
        .forward(&[1, 3], &[0.5, -1.0, 2.0, 0.1])
        .expect("forward should succeed");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_tab_transformer_wrong_cat_ids() {
    let cfg = TabTransformerConfig {
        n_cat_features: 2,
        n_num_features: 2,
        cat_vocab_sizes: vec![5, 5],
        embed_dim: 4,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 8,
        n_classes: 2,
    };
    let model = TabTransformer::new(cfg, 0).expect("TabTransformer creation should succeed");
    let result = model.forward(&[0], &[0.0, 0.0]); // wrong n_cat
    assert!(result.is_err());
}

#[test]
fn test_tab_transformer_vocab_mismatch() {
    let cfg = TabTransformerConfig {
        n_cat_features: 2,
        n_num_features: 2,
        cat_vocab_sizes: vec![5], // mismatch: only 1 but n_cat=2
        embed_dim: 4,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 8,
        n_classes: 2,
    };
    assert!(TabTransformer::new(cfg, 0).is_err());
}

#[test]
fn test_tab_transformer_oob_cat_id_clamped() {
    let cfg = TabTransformerConfig {
        n_cat_features: 1,
        n_num_features: 1,
        cat_vocab_sizes: vec![3],
        embed_dim: 4,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 8,
        n_classes: 2,
    };
    let model = TabTransformer::new(cfg, 99).expect("TabTransformer creation should succeed");
    // ID 100 is out of bounds; should be clamped
    let out = model.forward(&[100], &[0.0]);
    assert!(out.is_ok());
}

// ── FTTransformer ──────────────────────────────────────────────────────

#[test]
fn test_ft_transformer_construction() {
    let cfg = FTTransformerConfig {
        n_cat_features: 2,
        n_num_features: 3,
        cat_vocab_sizes: vec![5, 8],
        embed_dim: 8,
        n_heads: 2,
        n_layers: 2,
        ffn_dim: 16,
        n_classes: 3,
    };
    let model = FTTransformer::new(cfg, 42);
    assert!(model.is_ok());
}

#[test]
fn test_ft_transformer_forward_shape() {
    let cfg = FTTransformerConfig {
        n_cat_features: 2,
        n_num_features: 2,
        cat_vocab_sizes: vec![4, 6],
        embed_dim: 8,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 16,
        n_classes: 4,
    };
    let model = FTTransformer::new(cfg, 1).expect("FTTransformer creation should succeed");
    let out = model
        .forward(&[0, 2], &[1.0, -0.5])
        .expect("forward should succeed");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_ft_transformer_no_numeric() {
    let cfg = FTTransformerConfig {
        n_cat_features: 3,
        n_num_features: 0,
        cat_vocab_sizes: vec![5, 5, 5],
        embed_dim: 4,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 8,
        n_classes: 2,
    };
    let model = FTTransformer::new(cfg, 2).expect("FTTransformer creation should succeed");
    let out = model.forward(&[1, 2, 0], &[]);
    assert!(out.is_ok());
    assert_eq!(out.expect("forward should succeed").len(), 2);
}

// ── NODE ───────────────────────────────────────────────────────────────

#[test]
fn test_oblivious_tree_forward() {
    let mut rng = StdRng::seed_from_u64(42);
    let tree = ObliviousTree::new(3, 5, 2, &mut rng);
    let out = tree
        .forward(&[0.1, -0.5, 1.0, 0.3, 0.8])
        .expect("tree forward should succeed");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_oblivious_tree_leaf_probs_sum() {
    let mut rng = StdRng::seed_from_u64(1);
    let tree = ObliviousTree::new(2, 3, 1, &mut rng);
    let out = tree.forward(&[0.5, 0.5, 0.5]);
    assert!(out.is_ok());
}

#[test]
fn test_node_model_forward() {
    let model = NodeModel::new(5, 3, 4, 2, 42);
    let out = model
        .forward(&[1.0, 0.5, -1.0, 2.0])
        .expect("forward should succeed");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_node_model_wrong_input() {
    let model = NodeModel::new(3, 2, 4, 2, 0);
    let result = model.forward(&[1.0]); // wrong: should be n_features=4
    assert!(result.is_err());
}

#[test]
fn test_node_model_many_trees() {
    let model = NodeModel::new(20, 4, 6, 3, 77);
    let x = vec![0.1_f32; 6];
    let out = model.forward(&x).expect("forward should succeed");
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── TabNet ─────────────────────────────────────────────────────────────

#[test]
fn test_tabnet_construction() {
    let cfg = TabNetConfig {
        n_steps: 3,
        n_d: 8,
        n_a: 8,
        gamma: 1.3,
        epsilon: 1e-5,
        n_features: 10,
        n_classes: 3,
    };
    let model = TabNet::new(cfg, 42);
    assert!(model.is_ok());
}

#[test]
fn test_tabnet_forward_shape() {
    let cfg = TabNetConfig {
        n_steps: 3,
        n_d: 8,
        n_a: 8,
        gamma: 1.5,
        epsilon: 1e-5,
        n_features: 6,
        n_classes: 2,
    };
    let model = TabNet::new(cfg, 7).expect("TabNet creation should succeed");
    let (logits, masks) = model
        .forward(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        .expect("forward should succeed");
    assert_eq!(logits.len(), 2);
    assert_eq!(masks.len(), 3);
}

#[test]
fn test_tabnet_mask_shape() {
    let cfg = TabNetConfig {
        n_steps: 2,
        n_d: 4,
        n_a: 4,
        gamma: 1.3,
        epsilon: 1e-5,
        n_features: 5,
        n_classes: 3,
    };
    let model = TabNet::new(cfg, 0).expect("TabNet creation should succeed");
    let (_, masks) = model
        .forward(&[1.0, -1.0, 0.5, 0.0, 2.0])
        .expect("forward should succeed");
    for mask in &masks {
        assert_eq!(mask.len(), 5);
        let sum: f32 = mask.iter().sum();
        assert!(sum >= 0.0);
    }
}

#[test]
fn test_tabnet_wrong_n_steps() {
    let cfg = TabNetConfig {
        n_steps: 0, // invalid
        n_d: 8,
        n_a: 8,
        gamma: 1.3,
        epsilon: 1e-5,
        n_features: 10,
        n_classes: 2,
    };
    assert!(TabNet::new(cfg, 0).is_err());
}

// ── SAINT ──────────────────────────────────────────────────────────────

#[test]
fn test_saint_block_intra_attention() {
    let block = SaintBlock::new(4, 2, 8, 42).expect("SaintBlock creation should succeed");
    // 3 features × 4 d_model
    let x = vec![0.5_f32; 3 * 4];
    let out = block
        .intra_feature_attention(&x)
        .expect("intra_feature_attention should succeed");
    assert_eq!(out.len(), 3 * 4);
}

#[test]
fn test_saint_block_inter_attention() {
    let block = SaintBlock::new(4, 2, 8, 1).expect("SaintBlock creation should succeed");
    let batch: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 4]).collect();
    let out = block
        .inter_sample_attention(&batch)
        .expect("inter_sample_attention should succeed");
    assert_eq!(out.len(), 5);
    assert!(out.iter().all(|v| v.len() == 4));
}

#[test]
fn test_saint_model_forward() {
    let model = SaintModel::new(2, 3, vec![5, 8], 8, 2, 2, 16, 4, 42)
        .expect("SaintModel creation should succeed");
    let out = model
        .forward(&[1, 2], &[0.5, -0.5, 1.0])
        .expect("forward should succeed");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_saint_block_invalid_heads() {
    let result = SaintBlock::new(5, 3, 10, 0); // 5 not divisible by 3
    assert!(result.is_err());
}

// ── FeatureEncoder / Scalers ───────────────────────────────────────────

#[test]
fn test_standard_scaler_fit_transform() {
    let data: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut scaler = StandardScaler::default();
    scaler.fit(&data_refs);
    let out = scaler.transform(&[3.0, 4.0]);
    assert_eq!(out.len(), 2);
    // Mean of [1,3,5] = 3, so transform(3.0) ≈ 0
    assert!(out[0].abs() < 0.1, "Expected ~0, got {}", out[0]);
}

#[test]
fn test_minmax_scaler() {
    let data: Vec<Vec<f32>> = vec![vec![0.0, 10.0], vec![5.0, 20.0], vec![10.0, 30.0]];
    let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut scaler = MinMaxScaler::default();
    scaler.fit(&data_refs);
    let out = scaler.transform(&[0.0, 10.0]);
    assert!((out[0] - 0.0).abs() < 1e-5);
    let out2 = scaler.transform(&[10.0, 30.0]);
    assert!((out2[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_quantile_transformer() {
    let data: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
    let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut qt = QuantileTransformer::default();
    qt.fit(&data_refs);
    let out = qt.transform(&[10.0]);
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

#[test]
fn test_cyclic_encoder() {
    let enc = CyclicEncoder::new(vec![24.0, 7.0]);
    let out = enc.transform(&[0.0, 0.0]);
    assert_eq!(out.len(), 4);
    // sin(0) = 0, cos(0) = 1
    assert!(out[0].abs() < 1e-5);
    assert!((out[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_cyclic_encoder_period() {
    let enc = CyclicEncoder::new(vec![4.0]);
    let out = enc.transform(&[1.0]);
    let expected_sin = (2.0 * PI / 4.0_f32).sin();
    let expected_cos = (2.0 * PI / 4.0_f32).cos();
    assert!(
        (out[0] - expected_sin).abs() < 1e-5,
        "sin mismatch: {} vs {}",
        out[0],
        expected_sin
    );
    assert!(
        (out[1] - expected_cos).abs() < 1e-5,
        "cos mismatch: {} vs {}",
        out[1],
        expected_cos
    );
}

#[test]
fn test_feature_encoder_pipeline() {
    let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, -i as f32]).collect();
    let data_refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut enc = FeatureEncoder::new(None);
    enc.fit(&data_refs);
    let out = enc.transform(&[5.0, -5.0]);
    assert_eq!(out.len(), 2);
}

// ── MixedInputHead ─────────────────────────────────────────────────────

#[test]
fn test_mixed_input_head_gate() {
    let head = MixedInputHead::new(4, 3, 8, 42);
    let cat = vec![0.5_f32; 4];
    let num = vec![-0.3_f32; 3];
    let out = head.gate(&cat, &num).expect("gate should succeed");
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mixed_input_head_wrong_dims() {
    let head = MixedInputHead::new(4, 3, 8, 0);
    let result = head.gate(&[1.0, 2.0], &[0.5_f32; 3]); // wrong cat dim
    assert!(result.is_err());
}

// ── TabularAugmentation ────────────────────────────────────────────────

#[test]
fn test_mixup_output_shape() {
    let mut rng = StdRng::seed_from_u64(42);
    let x1 = vec![1.0_f32, 2.0, 3.0];
    let x2 = vec![4.0_f32, 5.0, 6.0];
    let (mixed, lambda) =
        TabularAugmentation::mixup(&x1, &x2, 0.4, &mut rng).expect("mixup should succeed");
    assert_eq!(mixed.len(), 3);
    assert!((0.0..=1.0).contains(&lambda));
}

#[test]
fn test_mixup_len_mismatch() {
    let mut rng = StdRng::seed_from_u64(0);
    let result = TabularAugmentation::mixup(&[1.0, 2.0], &[1.0], 0.5, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_cutmix() {
    let mut rng = StdRng::seed_from_u64(7);
    let x1 = vec![0.0_f32; 10];
    let x2 = vec![1.0_f32; 10];
    let (mixed, lambda) =
        TabularAugmentation::cutmix(&x1, &x2, 1.0, &mut rng).expect("cutmix should succeed");
    assert_eq!(mixed.len(), 10);
    assert!((0.0..=1.0).contains(&lambda));
}

#[test]
fn test_smote_like() {
    let mut rng = StdRng::seed_from_u64(42);
    let sample = vec![1.0_f32, 2.0, 3.0];
    let neighbours = vec![vec![2.0_f32, 3.0, 4.0], vec![0.0_f32, 1.0, 2.0]];
    let synth = TabularAugmentation::smote_like(&sample, &neighbours, &mut rng)
        .expect("smote_like should succeed");
    assert_eq!(synth.len(), 3);
    // Each component should be between sample and neighbour
    for (&s, (&n1, &n2)) in synth.iter().zip(sample.iter().zip(neighbours[0].iter())) {
        let lo = n1.min(n2);
        let hi = n1.max(n2);
        assert!(
            s >= lo && s <= hi,
            "out of range: {} not in [{lo}, {hi}]",
            s
        );
    }
}

#[test]
fn test_smote_no_neighbours() {
    let mut rng = StdRng::seed_from_u64(0);
    let result = TabularAugmentation::smote_like(&[1.0], &[], &mut rng);
    assert!(result.is_err());
}

// ── TabularMetrics ─────────────────────────────────────────────────────

#[test]
fn test_accuracy_perfect() {
    let pred = vec![
        vec![2.0_f32, 0.0, 0.0],
        vec![0.0, 3.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let target = vec![0, 1, 2];
    let acc = TabularMetrics::accuracy(&pred, &target);
    assert!((acc - 1.0).abs() < 1e-6);
}

#[test]
fn test_accuracy_none_correct() {
    let pred = vec![
        vec![0.0_f32, 1.0], // pred class 1
        vec![0.0_f32, 1.0], // pred class 1
    ];
    let target = vec![0, 0];
    let acc = TabularMetrics::accuracy(&pred, &target);
    assert!((acc - 0.0).abs() < 1e-6);
}

#[test]
fn test_macro_f1_perfect() {
    let pred = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let target = vec![0, 1, 0];
    let f1 = TabularMetrics::macro_f1(&pred, &target, 2);
    assert!((f1 - 1.0).abs() < 1e-5, "Expected 1.0 F1, got {f1}");
}

#[test]
fn test_rmse() {
    let pred = vec![1.0_f32, 2.0, 3.0];
    let target = vec![1.0_f32, 2.0, 3.0];
    let r = TabularMetrics::rmse(&pred, &target);
    assert!(r.abs() < 1e-6);
}

#[test]
fn test_rmse_nonzero() {
    let pred = vec![0.0_f32; 4];
    let target = vec![1.0_f32, 1.0, 1.0, 1.0];
    let r = TabularMetrics::rmse(&pred, &target);
    assert!((r - 1.0).abs() < 1e-5);
}

#[test]
fn test_r2_score_perfect() {
    let pred = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target = vec![1.0_f32, 2.0, 3.0, 4.0];
    let r2 = TabularMetrics::r2_score(&pred, &target);
    assert!((r2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_r2_score_null_model() {
    let target = vec![1.0_f32, 2.0, 3.0];
    let mean_t = 2.0_f32;
    let pred = vec![mean_t; 3];
    let r2 = TabularMetrics::r2_score(&pred, &target);
    assert!(r2 >= -1e-5); // ~0 for mean predictor
}

// ── CatBoostEncoder ────────────────────────────────────────────────────

#[test]
fn test_catboost_encoder_fit_transform() {
    let mut enc = CatBoostEncoder::new();
    let cats = vec![0, 1, 0, 1, 0];
    let targets = vec![1.0_f32, 0.0, 1.0, 0.0, 1.0];
    let encoded = enc.fit_transform(&cats, &targets, 0.5);
    assert_eq!(encoded.len(), 5);
    assert!(encoded.iter().all(|v| v.is_finite()));
}

#[test]
fn test_catboost_encoder_loo_no_leakage() {
    let mut enc = CatBoostEncoder::new();
    // All targets for cat=0 are 1.0, LOO should exclude current sample
    let cats = vec![0, 0, 0];
    let targets = vec![1.0_f32, 1.0, 0.0];
    let encoded = enc.fit_transform(&cats, &targets, 0.5);
    // For the last sample (target=0.0), LOO mean of others is 1.0
    assert!(
        encoded[2] > 0.5,
        "LOO should give ~1.0 for last sample, got {}",
        encoded[2]
    );
}

#[test]
fn test_catboost_encoder_transform_unseen() {
    let mut enc = CatBoostEncoder::new();
    let _ = enc.fit_transform(&[0, 1], &[1.0, 0.0], 0.5);
    let out = enc.transform(&[99]); // unseen category
    assert_eq!(out.len(), 1);
    assert!(
        (out[0] - 0.5).abs() < 1e-5,
        "Unseen category should use prior, got {}",
        out[0]
    );
}

#[test]
fn test_catboost_encoder_length_mismatch() {
    let mut enc = CatBoostEncoder::new();
    let result = enc.fit_transform(&[0, 1, 2], &[1.0, 0.0], 0.5);
    assert!(result.is_empty(), "Should return empty for length mismatch");
}

// ── Sparsemax ──────────────────────────────────────────────────────────

#[test]
fn test_sparsemax_sums_to_one() {
    let z = vec![0.5_f32, -0.3, 1.2, 0.8];
    let out = sparsemax(&z);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sparsemax sum={sum}");
}

#[test]
fn test_sparsemax_sparse_output() {
    let z = vec![10.0_f32, 0.0, -10.0];
    let out = sparsemax(&z);
    // Large logit should dominate
    assert!(out[0] > 0.5);
    assert!(out[2].abs() < 1e-5);
}

// ── Probit ─────────────────────────────────────────────────────────────

#[test]
fn test_probit_median() {
    let v = probit(0.5);
    assert!(v.abs() < 0.1, "probit(0.5) should be ~0, got {v}");
}

#[test]
fn test_probit_monotone() {
    let v1 = probit(0.25);
    let v2 = probit(0.75);
    assert!(v1 < v2, "probit must be monotone increasing");
}
