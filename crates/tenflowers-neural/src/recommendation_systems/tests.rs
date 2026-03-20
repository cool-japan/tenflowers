//! Tests for all recommendation system components.

use super::*;

// ── Matrix Factorization ─────────────────────────────────────────────────

#[test]
fn test_mf_construction() {
    let cfg = MfConfig {
        n_users: 10,
        n_items: 8,
        emb_dim: 4,
        ..Default::default()
    };
    let mf = MatrixFactorization::new(cfg.clone(), 0).expect("MF construction should succeed");
    assert_eq!(mf.user_emb.len(), cfg.n_users * cfg.emb_dim);
    assert_eq!(mf.item_emb.len(), cfg.n_items * cfg.emb_dim);
}

#[test]
fn test_mf_predict_finite() {
    let cfg = MfConfig {
        n_users: 5,
        n_items: 5,
        emb_dim: 8,
        ..Default::default()
    };
    let mf = MatrixFactorization::new(cfg, 1).expect("MF construction should succeed");
    let p = mf.predict(2, 3);
    assert!(p.is_finite());
}

#[test]
fn test_mf_sgd_update_decreases_error() {
    let cfg = MfConfig {
        n_users: 5,
        n_items: 5,
        emb_dim: 4,
        reg_lambda: 0.0,
        use_bias: false,
    };
    let mut mf = MatrixFactorization::new(cfg, 42).expect("MF construction should succeed");
    let err0 = mf.train_step(0, 0, 5.0, 0.01);
    for _ in 0..200 {
        mf.train_step(0, 0, 5.0, 0.01);
    }
    let pred = mf.predict(0, 0);
    assert!((pred - 5.0).abs() < err0.sqrt());
}

#[test]
fn test_mf_als_user_update() {
    let cfg = MfConfig {
        n_users: 4,
        n_items: 4,
        emb_dim: 4,
        ..Default::default()
    };
    let mut mf = MatrixFactorization::new(cfg, 7).expect("MF construction should succeed");
    let obs = vec![(0usize, 4.0_f32), (2, 3.0), (3, 1.0)];
    let result = mf.als_update_user(1, &obs);
    assert!(result.is_ok());
}

#[test]
fn test_mf_als_item_update() {
    let cfg = MfConfig {
        n_users: 4,
        n_items: 4,
        emb_dim: 4,
        ..Default::default()
    };
    let mut mf = MatrixFactorization::new(cfg, 7).expect("MF construction should succeed");
    let obs = vec![(0usize, 4.0_f32), (2, 3.0)];
    let result = mf.als_update_item(1, &obs);
    assert!(result.is_ok());
}

#[test]
fn test_mf_rank_items_returns_sorted() {
    let cfg = MfConfig {
        n_users: 3,
        n_items: 5,
        emb_dim: 4,
        ..Default::default()
    };
    let mf = MatrixFactorization::new(cfg, 99).expect("MF construction should succeed");
    let ranked = mf.rank_items_for_user(0);
    assert_eq!(ranked.len(), 5);
    for w in ranked.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

// ── BPR Loss ─────────────────────────────────────────────────────────────

#[test]
fn test_bpr_loss_scalar_positive() {
    let bpr = BprLoss::new(0.0);
    // pos > neg → loss should be < log(2) ≈ 0.693
    let loss = bpr.bpr_loss_scalar(2.0, 0.0);
    assert!(loss < 0.693);
    assert!(loss > 0.0);
}

#[test]
fn test_bpr_loss_batch() {
    let bpr = BprLoss::new(0.0);
    let pos = vec![1.0, 2.0, 3.0];
    let neg = vec![-1.0, -2.0, -3.0];
    let loss = bpr.bpr_loss(&pos, &neg).expect("BPR loss should succeed");
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_bpr_loss_mismatch_error() {
    let bpr = BprLoss::new(0.0);
    let result = bpr.bpr_loss(&[1.0, 2.0], &[1.0]);
    assert!(matches!(result, Err(RecSysError::DimensionMismatch { .. })));
}

#[test]
fn test_bpr_grad_direction() {
    let bpr = BprLoss::new(0.0);
    let g = bpr.bpr_grad(3.0, 0.0);
    assert!(g < 0.0);
}

// ── NeuralCF ─────────────────────────────────────────────────────────────

#[test]
fn test_ncf_construction() {
    let cfg = NcfConfig {
        n_users: 10,
        n_items: 10,
        gmf_dim: 8,
        mlp_emb_dim: 8,
        mlp_layers: vec![16, 8],
        output_dim: 1,
    };
    let ncf = NeuralCF::new(cfg, 0).expect("NCF construction should succeed");
    assert_eq!(ncf.mlp_weights.len(), 2);
}

#[test]
fn test_ncf_forward_finite() {
    let cfg = NcfConfig {
        n_users: 5,
        n_items: 5,
        gmf_dim: 4,
        mlp_emb_dim: 4,
        mlp_layers: vec![8, 4],
        output_dim: 1,
    };
    let ncf = NeuralCF::new(cfg, 1).expect("NCF construction should succeed");
    let out = ncf.forward(2, 3).expect("NCF forward should succeed");
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

#[test]
fn test_ncf_out_of_bounds_user() {
    let cfg = NcfConfig {
        n_users: 3,
        n_items: 3,
        ..Default::default()
    };
    let ncf = NeuralCF::new(cfg, 0).expect("NCF construction should succeed");
    let result = ncf.forward(5, 0);
    assert!(matches!(
        result,
        Err(RecSysError::IndexOutOfBounds { what: "user", .. })
    ));
}

#[test]
fn test_ncf_out_of_bounds_item() {
    let cfg = NcfConfig {
        n_users: 3,
        n_items: 3,
        ..Default::default()
    };
    let ncf = NeuralCF::new(cfg, 0).expect("NCF construction should succeed");
    let result = ncf.forward(0, 5);
    assert!(matches!(
        result,
        Err(RecSysError::IndexOutOfBounds { what: "item", .. })
    ));
}

// ── SASRec ───────────────────────────────────────────────────────────────

#[test]
fn test_sasrec_construction() {
    let cfg = SasRecConfig {
        n_items: 20,
        max_seq_len: 10,
        d_model: 8,
        n_heads: 2,
        ffn_dim: 16,
        n_layers: 2,
        dropout: 0.0,
    };
    let model = SasRec::new(cfg, 0).expect("SASRec construction should succeed");
    assert_eq!(model.layers.len(), 2);
}

#[test]
fn test_sasrec_forward_shape() {
    let cfg = SasRecConfig {
        n_items: 20,
        max_seq_len: 10,
        d_model: 8,
        n_heads: 2,
        ffn_dim: 16,
        n_layers: 1,
        dropout: 0.0,
    };
    let model = SasRec::new(cfg.clone(), 42).expect("SASRec construction should succeed");
    let seq = vec![1usize, 5, 3, 7];
    let scores = model.forward(&seq).expect("SASRec forward should succeed");
    assert_eq!(scores.len(), cfg.n_items);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn test_sasrec_empty_sequence_error() {
    let cfg = SasRecConfig {
        n_items: 10,
        ..Default::default()
    };
    let model = SasRec::new(cfg, 0).expect("SASRec construction should succeed");
    let result = model.forward(&[]);
    assert!(matches!(result, Err(RecSysError::EmptySequence)));
}

#[test]
fn test_sasrec_long_sequence_truncated() {
    let cfg = SasRecConfig {
        n_items: 20,
        max_seq_len: 5,
        d_model: 8,
        n_heads: 2,
        ffn_dim: 16,
        n_layers: 1,
        dropout: 0.0,
    };
    let model = SasRec::new(cfg.clone(), 1).expect("SASRec construction should succeed");
    let seq: Vec<usize> = (0..15).map(|i| i % cfg.n_items).collect();
    let scores = model.forward(&seq).expect("SASRec forward should succeed");
    assert_eq!(scores.len(), cfg.n_items);
}

// ── BERT4Rec ─────────────────────────────────────────────────────────────

#[test]
fn test_bert4rec_construction() {
    let cfg = Bert4RecConfig {
        n_items: 20,
        max_seq_len: 10,
        d_model: 8,
        n_heads: 2,
        ffn_dim: 16,
        n_layers: 2,
        mask_token: 20,
    };
    let model = BERT4Rec::new(cfg, 0).expect("BERT4Rec construction should succeed");
    assert_eq!(model.layers.len(), 2);
}

#[test]
fn test_bert4rec_forward_masked() {
    let cfg = Bert4RecConfig {
        n_items: 20,
        max_seq_len: 10,
        d_model: 8,
        n_heads: 2,
        ffn_dim: 16,
        n_layers: 1,
        mask_token: 20,
    };
    let model = BERT4Rec::new(cfg.clone(), 5).expect("BERT4Rec construction should succeed");
    let seq = vec![1usize, 20, 3, 7, 20];
    let mask_pos = vec![1usize, 4];
    let out = model.forward(&seq, &mask_pos).expect("BERT4Rec forward should succeed");
    assert_eq!(out.len(), 2);
    let vocab = cfg.n_items + 1;
    assert_eq!(out[0].len(), vocab);
    assert!(out[0].iter().all(|v| v.is_finite()));
}

#[test]
fn test_bert4rec_empty_error() {
    let cfg = Bert4RecConfig::default();
    let model = BERT4Rec::new(cfg, 0).expect("BERT4Rec construction should succeed");
    let result = model.forward(&[], &[0]);
    assert!(matches!(result, Err(RecSysError::EmptySequence)));
}

// ── LightGCN ─────────────────────────────────────────────────────────────

#[test]
fn test_lightgcn_construction() {
    let cfg = LightGcnConfig {
        n_users: 5,
        n_items: 5,
        emb_dim: 8,
        reg_lambda: 1e-4,
    };
    let model = LightGCN::new(cfg, 0);
    assert_eq!(model.user_emb.len(), 5 * 8);
}

#[test]
fn test_lightgcn_propagate_shape() {
    let cfg = LightGcnConfig {
        n_users: 4,
        n_items: 4,
        emb_dim: 8,
        reg_lambda: 1e-4,
    };
    let model = LightGCN::new(cfg.clone(), 0);
    let edges = vec![(0, 0), (0, 1), (1, 2), (2, 3)];
    let (ue, ie) = model.propagate(&edges, 2).expect("LightGCN propagate should succeed");
    assert_eq!(ue.len(), cfg.n_users);
    assert_eq!(ie.len(), cfg.n_items);
    assert_eq!(ue[0].len(), cfg.emb_dim);
}

#[test]
fn test_lightgcn_propagate_finite() {
    let cfg = LightGcnConfig {
        n_users: 3,
        n_items: 3,
        emb_dim: 4,
        reg_lambda: 0.0,
    };
    let model = LightGCN::new(cfg, 42);
    let edges = vec![(0, 0), (1, 1), (2, 2)];
    let (ue, ie) = model.propagate(&edges, 3).expect("LightGCN propagate should succeed");
    assert!(ue.iter().all(|row| row.iter().all(|v| v.is_finite())));
    assert!(ie.iter().all(|row| row.iter().all(|v| v.is_finite())));
}

#[test]
fn test_lightgcn_bpr_update() {
    let cfg = LightGcnConfig {
        n_users: 5,
        n_items: 5,
        emb_dim: 4,
        reg_lambda: 1e-4,
    };
    let mut model = LightGCN::new(cfg, 0);
    let loss = model.bpr_update(0, 1, 3, 0.01).expect("LightGCN bpr_update should succeed");
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

// ── SessionEncoder ───────────────────────────────────────────────────────

#[test]
fn test_session_encoder_construction() {
    let cfg = SessionEncoderConfig {
        n_items: 20,
        emb_dim: 8,
        hidden_dim: 16,
        n_layers: 1,
    };
    let enc = SessionEncoder::new(cfg, 0);
    assert_eq!(enc.item_emb.len(), 20 * 8);
}

#[test]
fn test_session_encoder_encode_shape() {
    let cfg = SessionEncoderConfig {
        n_items: 20,
        emb_dim: 8,
        hidden_dim: 16,
        n_layers: 1,
    };
    let enc = SessionEncoder::new(cfg.clone(), 1);
    let session = vec![1usize, 5, 3, 7, 2];
    let h = enc.encode(&session).expect("session encode should succeed");
    assert_eq!(h.len(), cfg.hidden_dim);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn test_session_encoder_empty_error() {
    let cfg = SessionEncoderConfig::default();
    let enc = SessionEncoder::new(cfg, 0);
    let result = enc.encode(&[]);
    assert!(matches!(result, Err(RecSysError::EmptySequence)));
}

#[test]
fn test_session_encoder_single_item() {
    let cfg = SessionEncoderConfig {
        n_items: 10,
        emb_dim: 4,
        hidden_dim: 8,
        n_layers: 2,
    };
    let enc = SessionEncoder::new(cfg.clone(), 7);
    let h = enc.encode(&[3]).expect("single item session encode should succeed");
    assert_eq!(h.len(), cfg.hidden_dim);
}

// ── RecommendationEvaluator ──────────────────────────────────────────────

#[test]
fn test_evaluator_perfect_recall() {
    let evaluator = RecommendationEvaluator::new();
    let preds = vec![vec![1usize, 2, 3], vec![4, 5, 6]];
    let gt = vec![vec![1usize], vec![4]];
    let metrics = evaluator.evaluate(&preds, &gt, 3).expect("evaluation should succeed");
    assert!((metrics.hit_rate_at_k - 1.0).abs() < 1e-6);
    assert!((metrics.recall_at_k - 1.0).abs() < 1e-6);
    assert!((metrics.mrr - 1.0).abs() < 1e-6);
}

#[test]
fn test_evaluator_zero_hit() {
    let evaluator = RecommendationEvaluator::new();
    let preds = vec![vec![0usize, 1, 2]];
    let gt = vec![vec![9usize]];
    let metrics = evaluator.evaluate(&preds, &gt, 3).expect("evaluation should succeed");
    assert_eq!(metrics.hit_rate_at_k, 0.0);
    assert_eq!(metrics.mrr, 0.0);
}

#[test]
fn test_evaluator_ndcg_ordering() {
    let evaluator = RecommendationEvaluator::new();
    let preds_good = vec![vec![1usize, 2, 3]];
    let preds_bad = vec![vec![2usize, 1, 3]];
    let gt = vec![vec![1usize]];
    let m_good = evaluator.evaluate(&preds_good, &gt, 3).expect("evaluation should succeed");
    let m_bad = evaluator.evaluate(&preds_bad, &gt, 3).expect("evaluation should succeed");
    assert!(m_good.ndcg_at_k > m_bad.ndcg_at_k);
}

#[test]
fn test_evaluator_mismatch_error() {
    let evaluator = RecommendationEvaluator::new();
    let result = evaluator.evaluate(&[vec![1, 2]], &[], 3);
    assert!(matches!(result, Err(RecSysError::DimensionMismatch { .. })));
}

// ── DotProductSimilarity ─────────────────────────────────────────────────

#[test]
fn test_dot_product_same_vector() {
    let sim = DotProductSimilarity::new();
    let v = vec![1.0_f32, 2.0, 3.0];
    let s = sim.score(&v, &v).expect("dot product score should succeed");
    assert!((s - 14.0).abs() < 1e-5);
}

#[test]
fn test_dot_product_ranking() {
    let sim = DotProductSimilarity::new();
    let query = vec![1.0_f32, 0.0];
    let candidates = vec![vec![0.5_f32, 0.0], vec![2.0_f32, 0.0], vec![-1.0_f32, 0.0]];
    let ranked = sim.rank(&query, &candidates).expect("dot product rank should succeed");
    assert_eq!(ranked[0].0, 1);
    assert_eq!(ranked[2].0, 2);
}

// ── CosineSimilarity ─────────────────────────────────────────────────────

#[test]
fn test_cosine_identical_vectors() {
    let sim = CosineSimilarity::new();
    let v = vec![3.0_f32, 4.0];
    let s = sim.score(&v, &v).expect("cosine score should succeed");
    assert!((s - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_orthogonal_vectors() {
    let sim = CosineSimilarity::new();
    let a = vec![1.0_f32, 0.0];
    let b = vec![0.0_f32, 1.0];
    let s = sim.score(&a, &b).expect("cosine score should succeed");
    assert!(s.abs() < 1e-5);
}

#[test]
fn test_cosine_zero_vector() {
    let sim = CosineSimilarity::new();
    let a = vec![1.0_f32, 2.0];
    let z = vec![0.0_f32, 0.0];
    let s = sim.score(&a, &z).expect("cosine score should succeed");
    assert_eq!(s, 0.0);
}

// ── NegativeSampler ──────────────────────────────────────────────────────

#[test]
fn test_negative_sampler_uniform_avoids_positives() {
    let mut sampler = NegativeSampler::new_uniform(20, 0);
    let positives = vec![0usize, 1, 2, 3];
    let negs = sampler.sample(&positives, 5);
    for &n in &negs {
        assert!(!positives.contains(&n));
    }
}

#[test]
fn test_negative_sampler_popularity() {
    let counts = vec![1usize, 10, 50, 1, 1, 1, 1, 1, 1, 1];
    let mut sampler = NegativeSampler::new_popularity(&counts, 42);
    let samples: Vec<usize> = (0..1000).map(|_| sampler.sample_one()).collect();
    let count_2 = samples.iter().filter(|&&i| i == 2).count();
    assert!(count_2 > 100, "item 2 appeared only {count_2} times");
}

#[test]
fn test_negative_sampler_update_popularity() {
    let counts = vec![1usize; 5];
    let mut sampler = NegativeSampler::new_popularity(&counts, 0);
    let new_counts = vec![1usize, 1, 100, 1, 1];
    sampler.update_popularity(&new_counts);
    let samples: Vec<usize> = (0..500).map(|_| sampler.sample_one()).collect();
    let count_2 = samples.iter().filter(|&&i| i == 2).count();
    assert!(
        count_2 > 50,
        "expected item 2 to be sampled heavily, got {count_2}"
    );
}

#[test]
fn test_negative_sampler_returns_unique() {
    let mut sampler = NegativeSampler::new_uniform(100, 13);
    let positives = vec![0usize];
    let negs = sampler.sample(&positives, 10);
    let unique: std::collections::HashSet<usize> = negs.iter().cloned().collect();
    assert_eq!(unique.len(), negs.len());
}

// ── Utility ──────────────────────────────────────────────────────────────

#[test]
fn test_build_popularity_map() {
    let interactions = vec![(0, 1), (1, 1), (2, 2), (3, 3)];
    let map = build_popularity_map(&interactions);
    assert_eq!(*map.get(&1).expect("key 1 should exist in popularity map"), 2);
    assert_eq!(*map.get(&2).expect("key 2 should exist in popularity map"), 1);
}

#[test]
fn test_layer_norm_unit_variance() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0_f32; 4];
    let beta = vec![0.0_f32; 4];
    let y = layer_norm(&x, &gamma, &beta, 1e-6).expect("layer_norm should succeed");
    let mean_y: f32 = y.iter().sum::<f32>() / y.len() as f32;
    let var_y: f32 = y.iter().map(|v| (v - mean_y) * (v - mean_y)).sum::<f32>() / y.len() as f32;
    assert!(mean_y.abs() < 1e-5, "mean not near zero: {mean_y}");
    assert!((var_y - 1.0).abs() < 0.01, "variance not near 1: {var_y}");
}

#[test]
fn test_solve_linear_system_identity() {
    let mut a = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut b = vec![1.0_f32, 2.0, 3.0];
    let x = solve_linear_system(&mut a, &mut b, 3).expect("solve_linear_system should succeed");
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!((x[1] - 2.0).abs() < 1e-5);
    assert!((x[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_full_pipeline_mf_bpr() {
    let cfg = MfConfig {
        n_users: 5,
        n_items: 5,
        emb_dim: 4,
        reg_lambda: 1e-4,
        use_bias: false,
    };
    let mut mf = MatrixFactorization::new(cfg, 0).expect("MF construction should succeed");
    let bpr = BprLoss::new(0.0);
    let mut total_loss = 0.0_f32;
    for _ in 0..10 {
        let s_pos = mf.predict(0, 1);
        let s_neg = mf.predict(0, 4);
        total_loss += bpr.bpr_loss_scalar(s_pos, s_neg);
        mf.train_step(0, 1, 1.0, 0.01);
    }
    assert!(total_loss.is_finite());
}
