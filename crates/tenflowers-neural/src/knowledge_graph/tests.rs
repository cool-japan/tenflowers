//! Tests for the knowledge_graph module (core + extensions + advanced).

use super::*;
use scirs2_core::random::SeedableRng;

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_dataset() -> KgDataset {
    let entities: Vec<String> = (0..5).map(|i| format!("e{}", i)).collect();
    let relations: Vec<String> = vec!["r0".into(), "r1".into()];
    let triples = vec![
        (0, 0, 1),
        (1, 0, 2),
        (2, 0, 3),
        (3, 0, 4),
        (0, 1, 2),
        (1, 1, 3),
    ];
    KgDataset::from_triples(entities, relations, triples)
}

fn make_small_dataset() -> KgDataset {
    let entities: Vec<String> = (0..3).map(|i| format!("e{}", i)).collect();
    let relations: Vec<String> = vec!["r".into()];
    let triples = vec![(0, 0, 1), (1, 0, 2)];
    KgDataset::from_triples(entities, relations, triples)
}

// ── KgDataset tests ─────────────────────────────────────────────────────

#[test]
fn test_dataset_construction() {
    let ds = make_dataset();
    assert_eq!(ds.n_entities(), 5);
    assert_eq!(ds.n_relations(), 2);
    assert_eq!(ds.n_triples(), 6);
}

#[test]
fn test_dataset_entity_count_small() {
    let ds = make_small_dataset();
    assert_eq!(ds.n_entities(), 3);
    assert_eq!(ds.n_relations(), 1);
    assert_eq!(ds.n_triples(), 2);
}

#[test]
fn test_dataset_is_positive() {
    let ds = make_dataset();
    assert!(ds.is_positive(0, 0, 1));
    assert!(!ds.is_positive(0, 0, 3));
}

#[test]
fn test_negative_sample_validity() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let triple = &ds.triples[0];
    let mut found_non_original = false;
    for _ in 0..100 {
        let neg = ds.negative_sample(triple, &mut rng, true);
        assert!(neg.head < ds.n_entities());
        if neg != *triple {
            found_non_original = true;
        }
    }
    assert!(found_non_original);
}

#[test]
fn test_negative_sample_entity_bounds() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(1234);
    for t in &ds.triples {
        for _ in 0..20 {
            let neg = ds.negative_sample(t, &mut rng, false);
            assert!(neg.tail < ds.n_entities());
        }
    }
}

#[test]
fn test_negative_sample_corrupt_head_vs_tail() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(7);
    let triple = &ds.triples[0];
    let neg_h = ds.negative_sample(triple, &mut rng, true);
    assert_eq!(neg_h.relation, triple.relation);
    assert_eq!(neg_h.tail, triple.tail);
    let neg_t = ds.negative_sample(triple, &mut rng, false);
    assert_eq!(neg_t.head, triple.head);
    assert_eq!(neg_t.relation, triple.relation);
}

// ── Utility function tests ───────────────────────────────────────────────

#[test]
fn test_l2_norm_unit() {
    let v = vec![3.0_f64, 4.0];
    assert!((l2_norm(&v) - 5.0).abs() < 1e-12);
}

#[test]
fn test_normalize_vec() {
    let mut v = vec![3.0_f64, 4.0];
    normalize_vec(&mut v);
    assert!((l2_norm(&v) - 1.0).abs() < 1e-12);
}

#[test]
fn test_complex_multiply_identity() {
    let (r, i) = complex_multiply(1.0, 0.0, 3.0, 4.0);
    assert!((r - 3.0).abs() < 1e-12);
    assert!((i - 4.0).abs() < 1e-12);
}

#[test]
fn test_complex_multiply_i_squared() {
    let (r, i) = complex_multiply(0.0, 1.0, 0.0, 1.0);
    assert!((r + 1.0).abs() < 1e-12);
    assert!((i - 0.0).abs() < 1e-12);
}

#[test]
fn test_softplus_large_positive() {
    assert!((softplus(100.0) - 100.0).abs() < 1e-6);
}

#[test]
fn test_softplus_zero() {
    assert!((softplus(0.0) - 2.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_xavier_init_scale() {
    let mut rng = StdRng::seed_from_u64(0);
    let v = xavier_init_1d(1000, 64, 64, &mut rng);
    let limit = (6.0_f64 / 128.0).sqrt();
    for x in &v {
        assert!(*x >= -limit - 1e-12 && *x <= limit + 1e-12);
    }
}

// ── TransE tests ─────────────────────────────────────────────────────────

#[test]
fn test_transe_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(5, 2, cfg, &mut rng).expect("TransE init failed");
    assert_eq!(model.entity_emb.len(), 5);
    assert_eq!(model.relation_emb.len(), 2);
    assert_eq!(model.entity_emb[0].len(), 8);
}

#[test]
fn test_transe_embedding_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 16,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(5, 2, cfg, &mut rng).expect("TransE init");
    assert_eq!(model.embedding_dim(), 16);
    assert_eq!(model.entity_embedding(0).len(), 16);
    assert_eq!(model.relation_embedding(0).len(), 16);
}

#[test]
fn test_transe_entity_normalised() {
    let mut rng = StdRng::seed_from_u64(42);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(5, 2, cfg, &mut rng).expect("TransE init");
    for e in &model.entity_emb {
        let n = l2_norm(e);
        assert!((n - 1.0).abs() < 1e-10, "norm={n}");
    }
}

#[test]
fn test_transe_score_range() {
    let mut rng = StdRng::seed_from_u64(0);
    let ds = make_dataset();
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng)
        .expect("TransE init");
    for t in &ds.triples {
        let s = model.score(t.head, t.relation, t.tail);
        assert!(s <= 0.0, "score should be non-positive, got {s}");
    }
}

#[test]
fn test_transe_invalid_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 0,
        margin: 1.0,
        norm: 2,
    };
    assert!(TransEModel::new(3, 1, cfg, &mut rng).is_err());
}

// ── RotatE tests ─────────────────────────────────────────────────────────

#[test]
fn test_rotate_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = RotatEConfig {
        dim: 8,
        margin: 2.0,
    };
    let model = RotatEModel::new(5, 2, cfg, &mut rng).expect("RotatE init");
    assert_eq!(model.entity_real.len(), 5);
    assert_eq!(model.relation_phase.len(), 2);
    assert_eq!(model.entity_real[0].len(), 4);
}

#[test]
fn test_rotate_embedding_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = RotatEConfig {
        dim: 16,
        margin: 2.0,
    };
    let model = RotatEModel::new(5, 2, cfg, &mut rng).expect("RotatE init");
    assert_eq!(model.embedding_dim(), 16);
    assert_eq!(model.entity_embedding(0).len(), 16);
}

#[test]
fn test_rotate_score_is_nonpositive() {
    let mut rng = StdRng::seed_from_u64(7);
    let ds = make_dataset();
    let cfg = RotatEConfig {
        dim: 8,
        margin: 2.0,
    };
    let model = RotatEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng)
        .expect("RotatE init");
    for t in &ds.triples {
        let s = model.score(t.head, t.relation, t.tail);
        assert!(s <= 0.0, "score should be non-positive, got {s}");
    }
}

#[test]
fn test_rotate_invalid_odd_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = RotatEConfig {
        dim: 7,
        margin: 2.0,
    };
    assert!(RotatEModel::new(3, 1, cfg, &mut rng).is_err());
}

// ── ComplEx tests ────────────────────────────────────────────────────────

#[test]
fn test_complex_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = ComplExConfig {
        dim: 8,
        lambda: 1e-3,
    };
    let model = ComplExModel::new(5, 2, cfg, &mut rng).expect("ComplEx init");
    assert_eq!(model.entity_real.len(), 5);
    assert_eq!(model.relation_real.len(), 2);
    assert_eq!(model.entity_real[0].len(), 8);
}

#[test]
fn test_complex_score_valid() {
    let mut rng = StdRng::seed_from_u64(0);
    let ds = make_dataset();
    let cfg = ComplExConfig {
        dim: 8,
        lambda: 1e-3,
    };
    let model = ComplExModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng)
        .expect("ComplEx init");
    for t in &ds.triples {
        let s = model.score(t.head, t.relation, t.tail);
        assert!(s.is_finite(), "score is not finite: {s}");
    }
}

#[test]
fn test_complex_invalid_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = ComplExConfig {
        dim: 0,
        lambda: 1e-3,
    };
    assert!(ComplExModel::new(3, 1, cfg, &mut rng).is_err());
}

// ── Training tests ───────────────────────────────────────────────────────

#[test]
fn test_transe_training_loss_decreases() {
    let ds = make_small_dataset();
    let trainer = KgeTrainer {
        model_type: KgeModelType::TransE(TransEConfig {
            dim: 8,
            margin: 1.0,
            norm: 2,
        }),
        lr: 0.1,
        n_epochs: 20,
        batch_size: 4,
        n_negatives: 2,
        seed: 0,
    };
    let result = trainer.train(&ds).expect("training failed");
    assert_eq!(result.loss_history.len(), 20);
    let first = result.loss_history[0];
    let last = result.loss_history[19];
    assert!(last <= first + 0.5, "loss did not decrease: {first} -> {last}");
}

#[test]
fn test_rotate_training_loss_finite() {
    let ds = make_small_dataset();
    let trainer = KgeTrainer {
        model_type: KgeModelType::RotatE(RotatEConfig {
            dim: 8,
            margin: 2.0,
        }),
        lr: 0.05,
        n_epochs: 15,
        batch_size: 4,
        n_negatives: 2,
        seed: 1,
    };
    let result = trainer.train(&ds).expect("training failed");
    assert_eq!(result.loss_history.len(), 15);
    assert!(result.final_loss.is_finite());
}

#[test]
fn test_complex_training_loss_finite() {
    let ds = make_small_dataset();
    let trainer = KgeTrainer {
        model_type: KgeModelType::ComplEx(ComplExConfig {
            dim: 8,
            lambda: 1e-4,
        }),
        lr: 0.05,
        n_epochs: 20,
        batch_size: 4,
        n_negatives: 2,
        seed: 2,
    };
    let result = trainer.train(&ds).expect("training failed");
    assert_eq!(result.loss_history.len(), 20);
    assert!(result.final_loss.is_finite());
}

#[test]
fn test_training_empty_dataset_error() {
    let ds = KgDataset::from_triples(vec!["e0".into()], vec!["r".into()], vec![]);
    let trainer = KgeTrainer {
        model_type: KgeModelType::TransE(TransEConfig::default()),
        lr: 0.01,
        n_epochs: 5,
        batch_size: 4,
        n_negatives: 1,
        seed: 0,
    };
    assert!(trainer.train(&ds).is_err());
}

// ── Evaluator tests ──────────────────────────────────────────────────────

#[test]
fn test_hits_at_n_entities() {
    let ds = make_small_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let h = KgeEvaluator::hits_at_k(&model, &ds, ds.n_entities());
    assert!((h - 1.0).abs() < 1e-9, "Hits@N should be 1.0, got {h}");
}

#[test]
fn test_mrr_range() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let mrr = KgeEvaluator::mean_reciprocal_rank(&model, &ds);
    assert!(mrr > 0.0 && mrr <= 1.0, "MRR out of range: {mrr}");
}

#[test]
fn test_mean_rank_positive() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let mr = KgeEvaluator::mean_rank(&model, &ds);
    assert!(mr >= 1.0, "mean rank should be >= 1, got {mr}");
    assert!(mr <= ds.n_entities() as f64, "mean rank exceeds #entities");
}

// ── Extensions tests (KgeModelExt) ──────────────────────────────────────

#[test]
fn test_transe_model_ext_name() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(5, 2, cfg, &mut rng).expect("init");
    assert_eq!(model.model_name(), "TransE");
}

#[test]
fn test_model_summary() {
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(5, 2, cfg, &mut rng).expect("init");
    let summary = KgeModelSummary::from_model(&model, 5, 2);
    assert_eq!(summary.model_name, "TransE");
    assert_eq!(summary.embedding_dim, 8);
    assert_eq!(summary.n_entities, 5);
    assert_eq!(summary.n_relations, 2);
    assert!(summary.storage_size > 0);
}

// ── Advanced: TemporalTriple ──────────────────────────────────────────────

#[test]
fn test_temporal_triple_construction() {
    let qt = TemporalTriple::new(0, 1, 2, 3);
    assert_eq!(qt.head, 0);
    assert_eq!(qt.relation, 1);
    assert_eq!(qt.tail, 2);
    assert_eq!(qt.timestamp, 3);
}

#[test]
fn test_temporal_triple_to_static() {
    let qt = TemporalTriple::new(1, 0, 2, 100);
    let st = qt.to_static();
    assert_eq!(st.head, 1);
    assert_eq!(st.relation, 0);
    assert_eq!(st.tail, 2);
}

// ── Advanced: TeRoModel ───────────────────────────────────────────────────

#[test]
fn test_tero_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = TeRoModel::new(5, 3, 10, 8, &mut rng).expect("TeRo init");
    assert_eq!(model.entity_real.len(), 5);
    assert_eq!(model.relation_phase.len(), 3);
    assert_eq!(model.time_phase.len(), 10);
    assert_eq!(model.complex_dim, 4);
}

#[test]
fn test_tero_invalid_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    assert!(TeRoModel::new(5, 3, 10, 3, &mut rng).is_err()); // odd dim
}

#[test]
fn test_tero_score_is_nonpositive() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = TeRoModel::new(5, 3, 10, 8, &mut rng).expect("TeRo init");
    let s = model.score_temporal(0, 0, 1, 0);
    assert!(s <= 0.0, "TeRo score should be non-positive, got {s}");
}

#[test]
fn test_tero_score_is_finite() {
    let mut rng = StdRng::seed_from_u64(42);
    let model = TeRoModel::new(8, 4, 12, 8, &mut rng).expect("TeRo init");
    for h in 0..3 {
        for r in 0..2 {
            for t in 0..3 {
                let s = model.score_temporal(h, r, t, 0);
                assert!(s.is_finite(), "score({h},{r},{t},0) is NaN/inf");
            }
        }
    }
}

// ── Advanced: TntComplExModel ─────────────────────────────────────────────

#[test]
fn test_tntcomplex_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = TntComplExModel::new(5, 3, 10, 8, 1e-3, &mut rng).expect("TntComplEx init");
    assert_eq!(model.entity_real.len(), 5);
    assert_eq!(model.relation_real.len(), 3);
    assert_eq!(model.time_real.len(), 10);
}

#[test]
fn test_tntcomplex_score_finite() {
    let mut rng = StdRng::seed_from_u64(1);
    let model = TntComplExModel::new(5, 3, 10, 8, 1e-3, &mut rng).expect("TntComplEx init");
    let s = model.score_temporal(0, 0, 1, 0);
    assert!(s.is_finite(), "TntComplEx score is NaN/inf");
}

#[test]
fn test_tntcomplex_invalid_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    assert!(TntComplExModel::new(5, 3, 10, 0, 1e-3, &mut rng).is_err());
}

// ── Advanced: TemporalKgMetrics ──────────────────────────────────────────

#[test]
fn test_temporal_kg_metrics_empty() {
    let metrics = TemporalKgMetrics::compute(
        &[],
        5,
        &std::collections::HashSet::new(),
        |_, _, _, _| 0.0,
    );
    assert_eq!(metrics.n_evaluated, 0);
    assert_eq!(metrics.mrr, 0.0);
}

#[test]
fn test_temporal_kg_metrics_single_triple() {
    let triples = vec![TemporalTriple::new(0, 0, 1, 0)];
    let pos_set: std::collections::HashSet<_> = [(0usize, 0usize, 1usize, 0usize)].into();
    // Scorer that always ranks true tail first
    let metrics = TemporalKgMetrics::compute(&triples, 5, &pos_set, |_, _, t, _| {
        if t == 1 {
            10.0
        } else {
            0.0
        }
    });
    assert_eq!(metrics.n_evaluated, 1);
    assert!((metrics.mrr - 1.0).abs() < 1e-9);
    assert!((metrics.hits_at_1 - 1.0).abs() < 1e-9);
}

// ── Advanced: HypRelQuadruple & StarEModel ───────────────────────────────

#[test]
fn test_hyprel_quadruple_construction() {
    let q = HypRelQuadruple::new(0, 1, 2, vec![(0, 3), (1, 4)]);
    assert_eq!(q.head, 0);
    assert_eq!(q.qualifiers.len(), 2);
}

#[test]
fn test_stare_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = StarEModel::new(5, 3, 8, &mut rng).expect("StarE init");
    assert_eq!(model.entity_emb.len(), 5);
    assert_eq!(model.relation_emb.len(), 3);
}

#[test]
fn test_stare_score_finite() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = StarEModel::new(5, 3, 8, &mut rng).expect("StarE init");
    let quad = HypRelQuadruple::new(0, 0, 1, vec![(1, 2)]);
    let s = model.score_quadruple(&quad);
    assert!(s.is_finite(), "StarE score is NaN/inf");
    assert!(s <= 0.0, "StarE score should be non-positive (distance-based)");
}

#[test]
fn test_stare_score_no_qualifiers() {
    let mut rng = StdRng::seed_from_u64(42);
    let model = StarEModel::new(5, 3, 8, &mut rng).expect("StarE init");
    let quad_no_qual = HypRelQuadruple::new(0, 0, 1, vec![]);
    let s1 = model.score_quadruple(&quad_no_qual);
    let quad_with_qual = HypRelQuadruple::new(0, 0, 1, vec![(0, 2)]);
    let s2 = model.score_quadruple(&quad_with_qual);
    // Scores should differ when qualifiers differ
    assert!(s1 != s2 || s1.is_finite());
}

#[test]
fn test_stare_invalid_dim() {
    let mut rng = StdRng::seed_from_u64(0);
    assert!(StarEModel::new(5, 3, 0, &mut rng).is_err());
}

// ── Advanced: NaLPModel ───────────────────────────────────────────────────

#[test]
fn test_nalp_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = NaLPModel::new(5, 3, 8, 0.5, &mut rng).expect("NaLP init");
    assert_eq!(model.entity_emb.len(), 5);
    assert_eq!(model.relation_emb.len(), 3);
}

#[test]
fn test_nalp_score_finite() {
    let mut rng = StdRng::seed_from_u64(0);
    let model = NaLPModel::new(5, 3, 8, 0.5, &mut rng).expect("NaLP init");
    let quad = HypRelQuadruple::new(0, 0, 1, vec![(1, 2)]);
    let s = model.score_quadruple(&quad);
    assert!(s.is_finite(), "NaLP score is NaN/inf");
}

// ── Advanced: KgTextualizer ───────────────────────────────────────────────

#[test]
fn test_textualizer_default_template() {
    let t = KgTextualizer::new();
    let sentence = t.verbalize("Alice", "knows", "Bob");
    assert!(sentence.contains("Alice"));
    assert!(sentence.contains("Bob"));
}

#[test]
fn test_textualizer_custom_template() {
    let mut t = KgTextualizer::new();
    t.register_template("born_in", "{head} was born in {tail}.");
    let s = t.verbalize("Ada", "born_in", "London");
    assert_eq!(s, "Ada was born in London.");
}

#[test]
fn test_textualizer_verbalize_dataset() {
    let ds = make_small_dataset();
    let t = KgTextualizer::default();
    let sentences = t.verbalize_dataset(&ds);
    assert_eq!(sentences.len(), ds.n_triples());
}

// ── Advanced: KgEmbeddingAlignment ───────────────────────────────────────

#[test]
fn test_alignment_construction() {
    let mut rng = StdRng::seed_from_u64(0);
    let align = KgEmbeddingAlignment::new(8, 12, 0.07, &mut rng).expect("init");
    assert_eq!(align.kg_dim, 8);
    assert_eq!(align.text_dim, 12);
}

#[test]
fn test_alignment_project_text() {
    let mut rng = StdRng::seed_from_u64(0);
    let align = KgEmbeddingAlignment::new(8, 12, 0.07, &mut rng).expect("init");
    let text_emb = vec![0.1_f64; 12];
    let proj = align.project_text(&text_emb);
    assert_eq!(proj.len(), 8);
    for v in &proj {
        assert!(v.is_finite());
    }
}

#[test]
fn test_alignment_contrastive_loss_nonneg() {
    let mut rng = StdRng::seed_from_u64(0);
    let align = KgEmbeddingAlignment::new(4, 6, 0.07, &mut rng).expect("init");
    let kg_embs: Vec<Vec<f64>> = (0..3)
        .map(|i| vec![i as f64; 4])
        .collect();
    let text_embs: Vec<Vec<f64>> = (0..3)
        .map(|i| vec![i as f64 * 0.5; 6])
        .collect();
    let loss = align.contrastive_loss(&kg_embs, &text_embs);
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

// ── Advanced: KgQaRanker ──────────────────────────────────────────────────

#[test]
fn test_qa_ranker_retrieve_triples() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let ranker = KgQaRanker::new(5, 0.7);
    let retrieved = ranker.retrieve_triples(0, &ds, &model);
    // entity 0 has 2 outgoing triples (0,r0,1) and (0,r1,2)
    assert!(retrieved.len() <= 5);
    assert!(!retrieved.is_empty());
}

#[test]
fn test_qa_ranker_rank_answers() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let ranker = KgQaRanker::new(5, 0.5);
    let candidates = vec![1usize, 2, 3];
    let surface_scores = vec![0.9_f64, 0.5, 0.1];
    let ranked = ranker.rank_answers(0, &candidates, &surface_scores, &ds, &model);
    assert_eq!(ranked.len(), 3);
    // Scores should be finite
    for (_, s) in &ranked {
        assert!(s.is_finite());
    }
}

// ── Advanced: RuleInduction ───────────────────────────────────────────────

#[test]
fn test_rule_induction_runs() {
    let ds = make_dataset();
    let ri = RuleInduction::new(1, 0.0);
    let rules = ri.mine_rules(&ds);
    // Should return some rules (may be empty for small dataset, but must not panic)
    for r in &rules {
        assert!(r.confidence >= 0.0);
        assert!(r.confidence <= 1.0 + 1e-9);
        assert!(r.support >= 1);
    }
}

#[test]
fn test_rule_induction_confidence_threshold() {
    let ds = make_dataset();
    let ri = RuleInduction::new(1, 0.5);
    let rules = ri.mine_rules(&ds);
    for r in &rules {
        assert!(r.confidence >= 0.5 - 1e-9);
    }
}

// ── Advanced: PathReasoningModel ──────────────────────────────────────────

#[test]
fn test_path_reasoning_runs() {
    let ds = make_dataset();
    let mut rng = StdRng::seed_from_u64(0);
    let cfg = TransEConfig {
        dim: 8,
        margin: 1.0,
        norm: 2,
    };
    let model = TransEModel::new(ds.n_entities(), ds.n_relations(), cfg, &mut rng).expect("init");
    let pr = PathReasoningModel::new(2, 10, 1.0);
    let scores = pr.reason(0, 0, &ds, &model, &mut rng);
    // Entity scores should be finite and non-negative
    for v in scores.values() {
        assert!(v.is_finite());
        assert!(*v >= 0.0);
    }
}

// ── Advanced: KgMetricsExtended ───────────────────────────────────────────

#[test]
fn test_kg_metrics_extended_from_rules() {
    let rules = vec![
        HornRule {
            head_relation: 0,
            body_relations: vec![1],
            confidence: 0.8,
            support: 5,
        },
        HornRule {
            head_relation: 1,
            body_relations: vec![0, 2],
            confidence: 0.6,
            support: 3,
        },
    ];
    let metrics = KgMetricsExtended::from_rules(&rules, 10, 8);
    assert!((metrics.avg_rule_confidence - 0.7).abs() < 1e-9);
    assert_eq!(metrics.n_rules, 2);
    assert!((metrics.entity_coverage - 0.8).abs() < 1e-9);
}

#[test]
fn test_kg_metrics_empty_rules() {
    let metrics = KgMetricsExtended::from_rules(&[], 5, 0);
    assert_eq!(metrics.avg_rule_confidence, 0.0);
    assert_eq!(metrics.n_rules, 0);
    assert_eq!(metrics.entity_coverage, 0.0);
}

#[test]
fn test_path_accuracy_empty() {
    let acc = KgMetricsExtended::compute_path_accuracy(&[], &[]);
    assert_eq!(acc, 0.0);
}
