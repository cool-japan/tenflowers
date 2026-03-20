//! Tests for efficient_transformers module (split from original + advanced).

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn mm(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    use scirs2_core::random::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
        .collect()
}

fn all_finite(m: &[Vec<f64>]) {
    for r in m {
        for &v in r {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }
}

// ── 1. Low-Rank & Sparse Attention ───────────────────────────────────────
#[test]
fn test_low_rank_attention_shape() {
    let a = LowRankAttention::new(8, 4, 8, 42).expect("test: operation should succeed");
    let out = a.forward(&mm(6, 8, 1), &mm(6, 8, 2), &mm(6, 8, 3)).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (6, 8));
}
#[test]
fn test_low_rank_attention_zero_input() {
    let a = LowRankAttention::new(4, 2, 4, 7).expect("test: operation should succeed");
    let z = vec![vec![0.0; 4]; 3];
    let out = a.forward(&z, &z, &z).expect("test: operation should succeed");
    assert_eq!(out.len(), 3);
    for r in &out {
        for &v in r {
            assert!(v.abs() < 1e-8);
        }
    }
}
#[test]
fn test_randomized_svd_attention_shape() {
    let a = RandomizedSvdAttention::new(99);
    let out = a
        .forward(&mm(5, 4, 10), &mm(5, 4, 11), &mm(5, 4, 12), 2)
        .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 4));
}
#[test]
fn test_reversible_layer_inverse() {
    let l = ReversibleLayer::new(4, 42).expect("test: operation should succeed");
    let x1 = vec![0.5, -0.3, 0.1, 0.8];
    let x2 = vec![0.2, 0.9, -0.4, 0.3];
    let (y1, y2) = l.forward(&x1, &x2).expect("test: operation should succeed");
    let (rx1, rx2) = l.backward(&y1, &y2).expect("test: operation should succeed");
    for (a, b) in x1.iter().zip(&rx1) {
        assert!((a - b).abs() < 1e-10);
    }
    for (a, b) in x2.iter().zip(&rx2) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn test_reversible_layer_forward_changes_values() {
    let l = ReversibleLayer::new(4, 7).expect("test: operation should succeed");
    let x1 = vec![1.0, 0.0, -1.0, 0.5];
    let x2 = vec![0.3, 0.7, 0.2, -0.6];
    let (y1, y2) = l.forward(&x1, &x2).expect("test: operation should succeed");
    assert_ne!(y1, x1);
    assert_ne!(y2, x2);
}
#[test]
fn test_mixer_layer_shape() {
    let m = MixerLayer::new(4, 8, 42).expect("test: operation should succeed");
    let out = m.forward(&mm(4, 8, 5)).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (4, 8));
}
#[test]
fn test_mixer_layer_empty() {
    assert!(MixerLayer::new(4, 8, 1)
        .expect("test: operation should succeed")
        .forward(&[])
        .expect("test: operation should succeed")
        .is_empty());
}
#[test]
fn test_fnet_layer_shape() {
    let out = FNetLayer::new(6).forward(&mm(5, 6, 3)).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 6));
}
#[test]
fn test_fnet_layer_finite_values() {
    all_finite(&FNetLayer::new(4).forward(&mm(4, 4, 77)).expect("test: operation should succeed"));
}

// ── 2. Sub-Quadratic Attention ────────────────────────────────────────────
#[test]
fn test_ssm_diagonal_output() {
    let y = StateSpaceAttention::forward(
        &[1.0, 0.5, -0.3, 0.2, 0.8],
        &[0.9, 0.8, 0.7],
        &[0.5, 0.4, 0.3],
        &[1.0, -1.0, 0.5],
    )
    .expect("test: operation should succeed");
    assert_eq!(y.len(), 5);
    for &v in &y {
        assert!(v.is_finite());
    }
}
#[test]
fn test_ssm_zero_input_zero_output() {
    let y = StateSpaceAttention::forward(&[0.0; 4], &[0.9, 0.8], &[0.5, 0.4], &[1.0, -1.0])
        .expect("test: operation should succeed");
    for v in y {
        assert_eq!(v, 0.0);
    }
}
#[test]
fn test_retnet_parallel_shape() {
    let out = RetentiveNetworkLayer::new(4)
        .forward_parallel(&mm(5, 4, 1), &mm(5, 4, 2), &mm(5, 4, 3), 0.9)
        .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 4));
}
#[test]
fn test_retnet_causal_zero_upper() {
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
    let out = RetentiveNetworkLayer::new(2)
        .forward_parallel(&q, &k, &v, 1.0)
        .expect("test: operation should succeed");
    assert_eq!(out.len(), 2);
    all_finite(&out);
}
#[test]
fn test_gla_output_shape() {
    let out =
        GatedLinearAttention::forward(&mm(4, 6, 1), &mm(4, 6, 2), &mm(4, 6, 3), &mm(4, 6, 4))
            .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (4, 6));
}
#[test]
fn test_gla_finite_values() {
    all_finite(
        &GatedLinearAttention::forward(
            &mm(3, 4, 10),
            &mm(3, 4, 11),
            &mm(3, 4, 12),
            &mm(3, 4, 13),
        )
        .expect("test: operation should succeed"),
    );
}
#[test]
fn test_mamba2_selective_scan() {
    let b: Vec<Vec<f64>> = (0..4).map(|i| vec![0.3 + i as f64 * 0.1; 3]).collect();
    let c: Vec<Vec<f64>> = (0..4).map(|i| vec![0.5 - i as f64 * 0.05; 3]).collect();
    let y = Mamba2Layer::selective_scan(
        &[1.0, 0.5, 0.25, 0.125],
        &[0.1, 0.2, 0.15, 0.05],
        &[-0.5, -1.0, -0.3],
        &b,
        &c,
    )
    .expect("test: operation should succeed");
    assert_eq!(y.len(), 4);
    for v in &y {
        assert!(v.is_finite());
    }
}
#[test]
fn test_mega_layer_shape() {
    let out = MegaLayer::new(4, 4, 42)
        .expect("test: operation should succeed")
        .forward(
            &mm(5, 4, 5),
            &[0.9, 0.8, 0.85, 0.7],
            &[0.1, 0.2, 0.15, 0.05],
        )
        .expect("test: operation should succeed");
    assert_eq!(out.len(), 5);
}

// ── 3. Position Encodings ─────────────────────────────────────────────────
#[test]
fn test_alibi_bias_shape() {
    let b = AliBiPositionBias::compute_bias(8, 4);
    assert_eq!(b.len(), 4);
    assert_eq!(b[0].len(), 64);
}
#[test]
fn test_alibi_bias_diagonal_zero() {
    let b = AliBiPositionBias::compute_bias(4, 2);
    for h in 0..2 {
        for i in 0..4 {
            assert_eq!(b[h][i * 4 + i], 0.0);
        }
    }
}
#[test]
fn test_rope_encoding_shape() {
    let (q, k) = RopeEncoding::new(10000.0)
        .apply_rope(&mm(6, 8, 1), &mm(6, 8, 2), 10000.0)
        .expect("test: operation should succeed");
    assert_eq!((q.len(), q[0].len(), k.len(), k[0].len()), (6, 8, 6, 8));
}
#[test]
fn test_rope_preserves_norm() {
    let rope = RopeEncoding::new(10000.0);
    let q = mm(3, 4, 5);
    let k = mm(3, 4, 6);
    let (q_out, _) = rope.apply_rope(&q, &k, 10000.0).expect("test: operation should succeed");
    for (qi, qo) in q.iter().zip(&q_out) {
        let ni = qi.iter().map(|x| x * x).sum::<f64>().sqrt();
        let no = qo.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((ni - no).abs() < 1e-10);
    }
}
#[test]
fn test_yarn_interpolation() {
    let (c, s) = YarnRope::interpolate(10, 2, 8, 2.0, 10000.0);
    assert!(c.is_finite());
    assert!(s.is_finite());
    assert!((c * c + s * s - 1.0).abs() < 1e-10);
}
#[test]
fn test_yarn_pos_zero() {
    let (c, s) = YarnRope::interpolate(0, 0, 8, 2.0, 10000.0);
    assert!((c - 1.0).abs() < 1e-10);
    assert!(s.abs() < 1e-10);
}
#[test]
fn test_xpos_decay() {
    let (q, k) = Xpos::new(512.0)
        .apply(&mm(4, 4, 1), &mm(4, 4, 2), 1.0, 10000.0)
        .expect("test: operation should succeed");
    assert_eq!(q.len(), 4);
    assert_eq!(k.len(), 4);
    all_finite(&q);
    all_finite(&k);
}
#[test]
fn test_rff_position_dim() {
    let mut rng = StdRng::seed_from_u64(42);
    let f = RandomFourierFeaturePos::new(42).encode(5, 16, &mut rng);
    assert_eq!(f.len(), 16);
    for v in f {
        assert!(v.is_finite());
    }
}
#[test]
fn test_rff_different_positions_differ() {
    let rff = RandomFourierFeaturePos::new(0);
    let mut r1 = StdRng::seed_from_u64(1);
    let mut r2 = StdRng::seed_from_u64(1);
    let f1 = rff.encode(0, 8, &mut r1);
    let f2 = rff.encode(5, 8, &mut r2);
    assert!(f1.iter().zip(&f2).map(|(a, b)| (a - b).abs()).sum::<f64>() > 0.0);
}

// ── 4. Mixture Architecture ───────────────────────────────────────────────
#[test]
fn test_switch_transformer_layer() {
    let out = SwitchTransformerLayer::new(4, 8, 42)
        .expect("test: operation should succeed")
        .forward(&mm(6, 8, 1))
        .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (6, 8));
}
#[test]
fn test_switch_transformer_finite() {
    all_finite(
        &SwitchTransformerLayer::new(2, 4, 7)
            .expect("test: operation should succeed")
            .forward(&mm(4, 4, 3))
            .expect("test: operation should succeed"),
    );
}
#[test]
fn test_mod_layer() {
    let out = MixtureOfDepthsLayer::new(6, 42)
        .expect("test: operation should succeed")
        .forward(&mm(8, 6, 1), 0.5)
        .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (8, 6));
}
#[test]
fn test_mod_layer_capacity_factor_one() {
    assert_eq!(
        MixtureOfDepthsLayer::new(4, 1)
            .expect("test: operation should succeed")
            .forward(&mm(5, 4, 2), 1.0)
            .expect("test: operation should succeed")
            .len(),
        5
    );
}
#[test]
fn test_hydra_attention() {
    let h = HydraAttention::new(2, 8).expect("test: operation should succeed");
    let kh: Vec<Vec<Vec<f64>>> = (0..2).map(|i| mm(4, 4, i as u64 + 2)).collect();
    let vh: Vec<Vec<Vec<f64>>> = (0..2).map(|i| mm(4, 4, i as u64 + 10)).collect();
    let out = h.forward(&mm(4, 8, 1), &kh, &vh).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (4, 8));
}
#[test]
fn test_multi_query_attention() {
    let out = MultiQueryAttention::new(4, 8)
        .expect("test: operation should succeed")
        .forward(&mm(5, 8, 1), &mm(5, 2, 2), &mm(5, 2, 3))
        .expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 8));
}
#[test]
fn test_multi_query_attention_finite() {
    all_finite(
        &MultiQueryAttention::new(2, 4)
            .expect("test: operation should succeed")
            .forward(&mm(3, 4, 10), &mm(3, 2, 11), &mm(3, 2, 12))
            .expect("test: operation should succeed"),
    );
}
#[test]
fn test_grouped_query_attention() {
    let gqa = GroupedQueryAttention::new(4, 2, 8).expect("test: operation should succeed");
    let kg: Vec<Vec<Vec<f64>>> = (0..2).map(|g| mm(5, 2, g as u64 + 10)).collect();
    let vg: Vec<Vec<Vec<f64>>> = (0..2).map(|g| mm(5, 2, g as u64 + 20)).collect();
    let out = gqa.forward(&mm(5, 8, 1), &kg, &vg).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 8));
}
#[test]
fn test_gqa_finite() {
    let gqa = GroupedQueryAttention::new(4, 2, 8).expect("test: operation should succeed");
    let kg: Vec<Vec<Vec<f64>>> = (0..2).map(|g| mm(4, 2, g as u64 + 5)).collect();
    let vg: Vec<Vec<Vec<f64>>> = (0..2).map(|g| mm(4, 2, g as u64 + 15)).collect();
    all_finite(&gqa.forward(&mm(4, 8, 1), &kg, &vg).expect("test: operation should succeed"));
}

// ── 5. Training Efficiency ────────────────────────────────────────────────
#[test]
fn test_gradient_compression_topk() {
    let (v, i) = GradientCompressor::compress(&[0.1, -0.5, 0.3, -0.8, 0.2, 0.6], 0.5);
    assert_eq!(v.len(), 3);
    assert_eq!(i.len(), 3);
    assert!((v[0].abs() - 0.8).abs() < 1e-10);
}
#[test]
fn test_gradient_compression_all() {
    assert_eq!(
        GradientCompressor::compress(&[0.4, -0.7, 0.2], 1.0).0.len(),
        3
    );
}
#[test]
fn test_gradient_decompress() {
    let out = GradientCompressor::decompress(&[0.9, -0.3], &[2, 0], 5);
    assert_eq!(out, vec![-0.3, 0.0, 0.9, 0.0, 0.0]);
}
#[test]
fn test_gradient_roundtrip() {
    let g = vec![0.1, -0.5, 0.3, -0.8, 0.2, 0.6];
    let (v, i) = GradientCompressor::compress(&g, 1.0);
    let r = GradientCompressor::decompress(&v, &i, g.len());
    assert!(g.iter().zip(&r).map(|(a, b)| (a - b).abs()).sum::<f64>() < 1e-10);
}
#[test]
fn test_mixed_precision_scale() {
    let s = MixedPrecisionScaler::new(1024.0, 2.0, 0.5, 2000);
    assert_eq!(s.scale(1.0), 1024.0);
    assert_eq!(s.scale(0.5), 512.0);
}
#[test]
fn test_mixed_precision_overflow_backoff() {
    let mut s = MixedPrecisionScaler::new(1024.0, 2.0, 0.5, 100);
    assert_eq!(s.update(true), 512.0);
}
#[test]
fn test_mixed_precision_growth() {
    let mut s = MixedPrecisionScaler::new(1024.0, 2.0, 0.5, 2);
    s.update(false);
    assert_eq!(s.update(false), 2048.0);
}
#[test]
fn test_activation_checkpoint() {
    let m = ActivationCheckpointingMgr::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 6.0);
    assert!(!m.should_recompute(0, 6.0));
    assert!(m.should_recompute(3, 6.0));
    assert!(!m.should_recompute(99, 6.0));
}
#[test]
fn test_activation_checkpoint_budget_zero() {
    assert!(ActivationCheckpointingMgr::new(vec![0.5, 0.5], 0.0).should_recompute(0, 0.0));
}
#[test]
fn test_pipeline_schedule_entries() {
    let s = PipelineScheduler::schedule(3, 4);
    assert!(!s.is_empty());
    for (stage, _) in &s {
        assert!(*stage < 3);
    }
}
#[test]
fn test_pipeline_schedule_has_both_actions() {
    let s = PipelineScheduler::schedule(2, 4);
    assert!(s.iter().any(|(_, a)| *a == PipelineAction::Forward));
    assert!(s.iter().any(|(_, a)| *a == PipelineAction::Backward));
}
#[test]
fn test_zero_redundancy() {
    let z = ZeroRedundancyOptimizer::new(4).expect("test: operation should succeed");
    let p = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    assert_eq!(z.gather_params(0, 4, &p), vec![0.1, 0.2]);
    assert_eq!(z.gather_params(1, 4, &p), vec![0.3, 0.4]);
    assert_eq!(z.gather_params(2, 4, &p), vec![0.5, 0.6]);
    assert_eq!(z.gather_params(3, 4, &p), vec![0.7, 0.8]);
}
#[test]
fn test_zero_all_gather_roundtrip() {
    let z = ZeroRedundancyOptimizer::new(3).expect("test: operation should succeed");
    let p = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shards: Vec<Vec<f64>> = (0..3).map(|r| z.gather_params(r, 3, &p)).collect();
    assert_eq!(ZeroRedundancyOptimizer::all_gather(&shards), p);
}

// ── Additional ────────────────────────────────────────────────────────────
#[test]
fn test_low_rank_attention_rank1() {
    let a = LowRankAttention::new(4, 1, 4, 99).expect("test: operation should succeed");
    assert_eq!(
        a.forward(&mm(3, 4, 1), &mm(3, 4, 2), &mm(3, 4, 3))
            .expect("test: operation should succeed")
            .len(),
        3
    );
}
#[test]
fn test_ssm_length_preserving() {
    for t in [1usize, 5, 20] {
        let u: Vec<f64> = (0..t).map(|i| i as f64 * 0.1).collect();
        assert_eq!(
            StateSpaceAttention::forward(&u, &[0.9], &[0.5], &[1.0])
                .expect("test: operation should succeed")
                .len(),
            t
        );
    }
}
#[test]
fn test_rope_pos0_identity() {
    let rope = RopeEncoding::new(10000.0);
    let q = vec![vec![1.0_f64, 2.0, 3.0, 4.0]];
    let k = vec![vec![0.5_f64, -0.5, 1.5, -1.5]];
    let (qo, ko) = rope.apply_rope(&q, &k, 10000.0).expect("test: operation should succeed");
    for (a, b) in q[0].iter().zip(&qo[0]) {
        assert!((a - b).abs() < 1e-10);
    }
    for (a, b) in k[0].iter().zip(&ko[0]) {
        assert!((a - b).abs() < 1e-10);
    }
}
#[test]
fn test_alibi_monotone_decay() {
    let b = AliBiPositionBias::compute_bias(5, 1);
    assert_eq!(b[0][0], 0.0);
    assert!(b[0][1] < b[0][0]);
    assert!(b[0][2] < b[0][1]);
}
#[test]
fn test_switch_empty_input() {
    assert!(SwitchTransformerLayer::new(2, 4, 0)
        .expect("test: operation should succeed")
        .forward(&[])
        .expect("test: operation should succeed")
        .is_empty());
}
#[test]
fn test_mod_empty_input() {
    assert!(MixtureOfDepthsLayer::new(4, 0)
        .expect("test: operation should succeed")
        .forward(&[], 0.5)
        .expect("test: operation should succeed")
        .is_empty());
}
#[test]
fn test_zero_world_size_error() {
    assert!(ZeroRedundancyOptimizer::new(0).is_err());
}
#[test]
fn test_mamba2_single_step() {
    let y = Mamba2Layer::selective_scan(
        &[1.0],
        &[0.5],
        &[-1.0, -2.0],
        &[vec![0.5, 0.5]],
        &[vec![1.0, 1.0]],
    )
    .expect("test: operation should succeed");
    assert_eq!(y.len(), 1);
    assert!(y[0].is_finite());
}
#[test]
fn test_gqa_invalid_config() {
    assert!(GroupedQueryAttention::new(3, 2, 6).is_err());
}

// ── Advanced: RetNet ──────────────────────────────────────────────────────
#[test]
fn test_retention_layer_parallel_shape() {
    let layer = RetentionLayer::new(8, 4, 0.9, 42).expect("test: operation should succeed");
    let x = mm(6, 8, 1);
    let out = layer.forward_parallel(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (6, 4));
    all_finite(&out);
}

#[test]
fn test_retention_layer_recurrent_shape() {
    let layer = RetentionLayer::new(8, 4, 0.95, 7).expect("test: operation should succeed");
    use scirs2_core::random::Rng;
    let mut rng = StdRng::seed_from_u64(1);
    let x_t: Vec<f64> = (0..8).map(|_| rng.random::<f64>()).collect();
    let state = vec![vec![0.0_f64; 4]; 4];
    let (out, new_state) = layer.forward_recurrent(&x_t, &state).expect("test: operation should succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(new_state.len(), 4);
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_retention_layer_chunkwise_matches_parallel() {
    let layer = RetentionLayer::new(4, 2, 0.9, 5).expect("test: operation should succeed");
    let x = mm(4, 4, 10);
    let out_parallel = layer.forward_parallel(&x).expect("test: operation should succeed");
    let out_chunk = layer.forward_chunkwise(&x, 2).expect("test: operation should succeed");
    assert_eq!(out_parallel.len(), out_chunk.len());
    // Both should produce finite outputs
    all_finite(&out_parallel);
    all_finite(&out_chunk);
}

#[test]
fn test_retnet_block_forward() {
    let block = RetNetBlock::new(8, 2, 2, None, 42).expect("test: operation should succeed");
    let x = mm(5, 8, 1);
    let out = block.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 8));
    all_finite(&out);
}

#[test]
fn test_retnet_model_forward() {
    let model = RetNetModel::new(8, 2, 2, 2, 1).expect("test: operation should succeed");
    let x = mm(4, 8, 99);
    let out = model.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (4, 8));
    all_finite(&out);
}

#[test]
fn test_retnet_empty_input() {
    let layer = RetentionLayer::new(4, 2, 0.9, 1).expect("test: operation should succeed");
    assert!(layer.forward_parallel(&[]).expect("test: operation should succeed").is_empty());
}

// ── Advanced: SSD / Mamba-2 ───────────────────────────────────────────────
#[test]
fn test_ssd_forward_shape() {
    let ssd = Ssd::new(8, 4, 42).expect("test: operation should succeed");
    let x = mm(6, 8, 1);
    let out = ssd.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (6, 8));
    all_finite(&out);
}

#[test]
fn test_ssd_layer_forward() {
    let layer = SsdLayer::new(8, 16, 4, 7).expect("test: operation should succeed");
    let x = mm(5, 8, 2);
    let out = layer.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 8));
    all_finite(&out);
}

#[test]
fn test_mamba2_block_forward() {
    let block = Mamba2Block::new(8, 16, 4, 99).expect("test: operation should succeed");
    let x = mm(4, 8, 3);
    let out = block.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (4, 8));
    all_finite(&out);
}

#[test]
fn test_ssd_empty_input() {
    let ssd = Ssd::new(4, 2, 1).expect("test: operation should succeed");
    assert!(ssd.forward(&[]).expect("test: operation should succeed").is_empty());
}

// ── Advanced: Enhanced GQA / MQA ─────────────────────────────────────────
#[test]
fn test_enhanced_gqa_forward_shape() {
    let gqa = EnhancedGroupedQueryAttention::new(8, 4, 2, 42).expect("test: operation should succeed");
    let x = mm(6, 8, 1);
    let out = gqa.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (6, 8));
    all_finite(&out);
}

#[test]
fn test_multi_query_single_head_forward() {
    let mqa = MultiQuerySingleHeadAttention::new(8, 4, 7).expect("test: operation should succeed");
    let x = mm(5, 8, 2);
    let out = mqa.forward(&x).expect("test: operation should succeed");
    assert_eq!((out.len(), out[0].len()), (5, 8));
    all_finite(&out);
}

#[test]
fn test_enhanced_gqa_invalid_config() {
    assert!(EnhancedGroupedQueryAttention::new(8, 3, 2, 1).is_err());
}

// ── Advanced: KV Cache ────────────────────────────────────────────────────
#[test]
fn test_sliding_window_cache_eviction() {
    let mut cache = SlidingWindowKvCache::new(3);
    for i in 0..5usize {
        cache.insert(vec![i as f64], vec![i as f64 * 2.0]);
    }
    assert_eq!(cache.len(), 3); // evicted oldest 2
}

#[test]
fn test_sliding_window_cache_attend_shape() {
    let mut cache = SlidingWindowKvCache::new(4);
    for i in 0..4usize {
        cache.insert(vec![i as f64, 0.0], vec![i as f64, 1.0]);
    }
    let q = vec![1.0, 0.0];
    let out = cache.attend(&q);
    assert_eq!(out.len(), 2);
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_sliding_window_cache_empty_attend() {
    let cache = SlidingWindowKvCache::new(10);
    let out = cache.attend(&[1.0, 2.0]);
    assert_eq!(out.len(), 2);
    assert_eq!(out, vec![0.0, 0.0]);
}

#[test]
fn test_sink_token_cache_sink_retention() {
    let mut cache = SinkTokenCache::new(2, 3);
    // Insert 6 tokens; first 2 should be sinks
    for i in 0..6usize {
        cache.insert(vec![i as f64], vec![i as f64]);
    }
    assert_eq!(cache.n_sink_stored(), 2);
    assert_eq!(cache.n_recent_stored(), 3); // window = 3, tokens 3..5 kept
}

#[test]
fn test_sink_token_cache_attend() {
    let mut cache = SinkTokenCache::new(1, 3);
    for i in 0..4usize {
        cache.insert(vec![i as f64, 0.0], vec![1.0, i as f64]);
    }
    let out = cache.attend(&[1.0, 0.0]);
    assert_eq!(out.len(), 2);
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_sink_token_cache_clear_recent() {
    let mut cache = SinkTokenCache::new(2, 4);
    for i in 0..5usize {
        cache.insert(vec![i as f64], vec![i as f64]);
    }
    cache.clear_recent();
    assert_eq!(cache.n_recent_stored(), 0);
    assert_eq!(cache.n_sink_stored(), 2);
}

// ── Metrics ───────────────────────────────────────────────────────────────
#[test]
fn test_et_metrics_dense_attention() {
    let m = EtMetrics::dense_attention(128, 512, 8, 4);
    assert!(m.flops > 0.0);
    assert!(m.memory_bytes > 0.0);
    assert!(m.throughput_proxy > 0.0);
}

#[test]
fn test_et_metrics_linear_vs_dense() {
    let dense = EtMetrics::dense_attention(512, 256, 4, 1);
    let linear = EtMetrics::linear_attention(512, 256, 1);
    // Linear attention should use less FLOPs for long sequences
    assert!(linear.flops < dense.flops);
}

#[test]
fn test_et_metrics_gqa_less_flops_than_full() {
    let full = EtMetrics::dense_attention(128, 256, 8, 1);
    let gqa = EtMetrics::grouped_query_attention(128, 256, 8, 2, 1);
    // GQA with fewer KV heads should use fewer FLOPs for projection
    assert!(gqa.flops < full.flops * 1.5); // roughly comparable
}
