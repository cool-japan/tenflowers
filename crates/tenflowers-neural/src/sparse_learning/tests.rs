use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn identity_dict(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

// ── SparseCode ────────────────────────────────────────────────────────────────

#[test]
fn test_sparse_code_nnz() {
    let code = SparseCode {
        coefficients: vec![1.0, 0.0, -2.0],
        support: vec![0, 1, 3],
    };
    assert_eq!(code.nnz(), 2);
}

#[test]
fn test_sparse_code_reconstruct_shape() {
    let dict = identity_dict(4);
    let code = SparseCode {
        coefficients: vec![1.5, -0.5],
        support: vec![0, 2],
    };
    let rec = code.reconstruct(&dict);
    assert_eq!(rec.len(), 4);
}

// ── OMP ───────────────────────────────────────────────────────────────────────

#[test]
fn test_omp_encode_sparsity() {
    let dict = identity_dict(8);
    let signal: Vec<f32> = (0..8).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();
    let omp = OrthogonalMatchingPursuit { n_nonzero: 3 };
    let code = omp.encode(&signal, &dict);
    assert_eq!(code.support.len(), 3);
}

#[test]
fn test_omp_encode_residual_decreases() {
    let dict = identity_dict(6);
    let signal: Vec<f32> = vec![2.0, -1.0, 0.5, 0.0, 0.0, 0.0];
    let omp2 = OrthogonalMatchingPursuit { n_nonzero: 2 };
    let omp3 = OrthogonalMatchingPursuit { n_nonzero: 3 };
    let c2 = omp2.encode(&signal, &dict);
    let c3 = omp3.encode(&signal, &dict);
    let rec2 = c2.reconstruct(&dict);
    let rec3 = c3.reconstruct(&dict);
    let err2 = reconstruction_error(&signal, &rec2);
    let err3 = reconstruction_error(&signal, &rec3);
    assert!(err3 <= err2 + 1e-4, "err3={err3} should be <= err2={err2}");
}

#[test]
fn test_omp_encode_with_identity_dict() {
    let dict = identity_dict(5);
    let signal: Vec<f32> = vec![0.0, 3.0, 0.0, -2.0, 0.0];
    let omp = OrthogonalMatchingPursuit { n_nonzero: 2 };
    let code = omp.encode(&signal, &dict);
    let rec = code.reconstruct(&dict);
    for (a, b) in signal.iter().zip(rec.iter()) {
        assert!((a - b).abs() < 1e-4, "a={a} b={b}");
    }
}

// ── Soft threshold ────────────────────────────────────────────────────────────

#[test]
fn test_soft_threshold_positive() {
    let v = LassoEncoder::soft_threshold(0.8, 0.3);
    assert!((v - 0.5).abs() < 1e-6);
}

#[test]
fn test_soft_threshold_zero_below_lambda() {
    let v = LassoEncoder::soft_threshold(0.2, 0.3);
    assert_eq!(v, 0.0);
}

// ── ISTA LASSO ────────────────────────────────────────────────────────────────

#[test]
fn test_lasso_ista_encode_sparse() {
    let dict = identity_dict(6);
    let signal: Vec<f32> = vec![0.0, 2.0, 0.0, -1.5, 0.0, 0.0];
    let enc = LassoEncoder {
        lambda: 0.1,
        max_iter: 200,
        tol: 1e-6,
    };
    let code = enc.encode_ista(&signal, &dict);
    assert!(code.support.len() <= 6);
}

#[test]
fn test_lasso_ista_convergence() {
    let dict: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let signal = vec![1.0, -0.5, 0.0];
    let enc = LassoEncoder {
        lambda: 0.01,
        max_iter: 500,
        tol: 1e-7,
    };
    let code = enc.encode_ista(&signal, &dict);
    let rec = code.reconstruct(&dict);
    let err = reconstruction_error(&signal, &rec);
    assert!(err < 0.3, "err={err}");
}

// ── Dictionary ────────────────────────────────────────────────────────────────

#[test]
fn test_dictionary_normalize() {
    let atoms = vec![vec![3.0, 4.0], vec![0.0, 2.0]];
    let mut dict = Dictionary::new(atoms);
    dict.normalize_atoms();
    for atom in dict.atoms.iter() {
        let n: f32 = atom.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm={n}");
    }
}

#[test]
fn test_dictionary_coherence_range() {
    let atoms = vec![vec![1.0, 0.0], vec![0.707, 0.707], vec![0.0, 1.0]];
    let dict = Dictionary::new(atoms);
    let coh = dict.coherence();
    assert!((0.0..=1.0).contains(&coh), "coh={coh}");
}

// ── K-SVD ─────────────────────────────────────────────────────────────────────

#[test]
fn test_ksvd_fit_returns_dict_and_codes() {
    let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i as f32).sin()]).collect();
    let config = DlConfig {
        n_atoms: 4,
        n_nonzero: 2,
        max_iter: 5,
        tol: 1e-3,
    };
    let ksvd = KSvd { config };
    let (dict, codes) = ksvd.fit(&data);
    assert_eq!(codes.len(), 10);
    assert!(dict.n_atoms > 0);
}

#[test]
fn test_ksvd_dict_atom_count() {
    let data: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32, -(i as f32)]).collect();
    let config = DlConfig {
        n_atoms: 3,
        n_nonzero: 1,
        max_iter: 3,
        tol: 1e-2,
    };
    let ksvd = KSvd { config };
    let (dict, _) = ksvd.fit(&data);
    assert_eq!(dict.n_atoms, 3);
}

// ── Online DL ─────────────────────────────────────────────────────────────────

#[test]
fn test_online_dl_update_atom_count() {
    let config = DlConfig {
        n_atoms: 4,
        n_nonzero: 2,
        max_iter: 5,
        tol: 1e-3,
    };
    let signal_dim = 6;
    let mut odl = OnlineDictionaryLearning::new(config.clone(), signal_dim);
    let sample: Vec<f32> = vec![1.0, -1.0, 2.0, 0.5, -0.5, 0.0];
    let atoms: Vec<Vec<f32>> = (0..config.n_atoms)
        .map(|k| {
            (0..signal_dim)
                .map(|i| if i == k % signal_dim { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let dict = Dictionary::new(atoms);
    let new_dict = odl.update(&sample, &dict);
    assert_eq!(new_dict.n_atoms, 4);
}

#[test]
fn test_online_dl_fit_stream() {
    let config = DlConfig {
        n_atoms: 3,
        n_nonzero: 1,
        max_iter: 3,
        tol: 1e-3,
    };
    let signal_dim = 4;
    let data: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..signal_dim).map(|j| (i * j) as f32 * 0.1).collect())
        .collect();
    let mut odl = OnlineDictionaryLearning::new(config, signal_dim);
    let dict = odl.fit_stream(&data);
    assert_eq!(dict.n_atoms, 3);
    assert_eq!(dict.atom_dim, signal_dim);
}

// ── Compressed Sensing ────────────────────────────────────────────────────────

#[test]
fn test_cs_gaussian_matrix_shape() {
    let mut rng = make_rng();
    let mm = MeasurementMatrix::Gaussian(10, 20);
    let mat = generate_matrix(&mm, &mut rng);
    assert_eq!(mat.nrows(), 10);
    assert_eq!(mat.ncols(), 20);
}

#[test]
fn test_cs_bernoulli_matrix_shape() {
    let mut rng = make_rng();
    let mm = MeasurementMatrix::Bernoulli(8, 16);
    let mat = generate_matrix(&mm, &mut rng);
    assert_eq!(mat.nrows(), 8);
    assert_eq!(mat.ncols(), 16);
}

#[test]
fn test_cs_measure_shape() {
    let mut rng = make_rng();
    let signal: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let mm = MeasurementMatrix::Gaussian(10, 20);
    let mat = generate_matrix(&mm, &mut rng);
    let meas = measure(&signal, &mat);
    assert_eq!(meas.n_measurements, 10);
    assert_eq!(meas.signal_len, 20);
}

#[test]
fn test_admm_recover_shape() {
    let mut rng = make_rng();
    let signal: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
    let mm = MeasurementMatrix::Gaussian(12, 16);
    let mat = generate_matrix(&mm, &mut rng);
    let meas = measure(&signal, &mat);
    let bp = BasisPursuit {
        max_iter: 50,
        rho: 1.0,
        tol: 1e-4,
    };
    let recovered = bp.recover(&meas, &mat);
    assert_eq!(recovered.len(), 16);
}

#[test]
fn test_cosamp_recover_shape() {
    let mut rng = make_rng();
    let signal: Vec<f32> = (0..16).map(|i| if i < 2 { 1.0 } else { 0.0 }).collect();
    let mm = MeasurementMatrix::Gaussian(10, 16);
    let mat = generate_matrix(&mm, &mut rng);
    let meas = measure(&signal, &mat);
    let cosamp = CoSaMP {
        n_nonzero: 2,
        max_iter: 30,
    };
    let recovered = cosamp.recover(&meas, &mat);
    assert_eq!(recovered.len(), 16);
}

#[test]
fn test_rip_constant_estimate_range() {
    let mut rng = make_rng();
    let mm = MeasurementMatrix::Gaussian(20, 40);
    let mat = generate_matrix(&mm, &mut rng);
    let delta = rip_constant_estimate(&mat, 3, 50, &mut rng);
    assert!(delta >= 0.0, "delta={delta}");
    assert!(delta < 10.0, "delta={delta}");
}

// ── SparseAutoencoder ─────────────────────────────────────────────────────────

#[test]
fn test_sae_encode_shape() {
    let mut rng = make_rng();
    let config = SaeConfig {
        input_dim: 8,
        hidden_dim: 16,
        sparsity_target: 0.05,
        sparsity_weight: 0.1,
    };
    let sae = SparseAutoencoder::new(&config, &mut rng);
    let x = vec![0.5_f32; 8];
    let z = sae.encode(&x);
    assert_eq!(z.len(), 16);
}

#[test]
fn test_sae_decode_shape() {
    let mut rng = make_rng();
    let config = SaeConfig {
        input_dim: 10,
        hidden_dim: 20,
        sparsity_target: 0.1,
        sparsity_weight: 0.01,
    };
    let sae = SparseAutoencoder::new(&config, &mut rng);
    let z = vec![0.5_f32; 20];
    let x_hat = sae.decode(&z);
    assert_eq!(x_hat.len(), 10);
}

#[test]
fn test_sae_reconstruction_loss_positive() {
    let mut rng = make_rng();
    let config = SaeConfig {
        input_dim: 6,
        hidden_dim: 12,
        sparsity_target: 0.1,
        sparsity_weight: 0.01,
    };
    let sae = SparseAutoencoder::new(&config, &mut rng);
    let x = vec![1.0, -1.0, 2.0, 0.5, -0.5, 0.0];
    let loss = sae.reconstruction_loss(&x);
    assert!(loss >= 0.0, "loss={loss}");
}

#[test]
fn test_kl_divergence_bernoulli_zero_match() {
    let kl = SparseAutoencoder::kl_divergence_bernoulli(0.1, 0.1);
    assert!(kl.abs() < 1e-5, "kl={kl}");
}

// ── MatchingPursuit ───────────────────────────────────────────────────────────

#[test]
fn test_mp_decompose_atoms_count() {
    let dict = identity_dict(5);
    let signal = vec![1.0, 2.0, -1.0, 0.5, 0.0];
    let mp = MatchingPursuit { n_atoms: 3 };
    let result = mp.decompose(&signal, &dict);
    assert!(result.atoms.len() <= 3);
}

#[test]
fn test_mp_residual_decreases() {
    let dict = identity_dict(4);
    let signal = vec![1.0, -2.0, 0.5, 0.0];
    let mp1 = MatchingPursuit { n_atoms: 1 };
    let mp2 = MatchingPursuit { n_atoms: 2 };
    let r1 = mp1.decompose(&signal, &dict);
    let r2 = mp2.decompose(&signal, &dict);
    assert!(
        r2.residual_norm <= r1.residual_norm + 1e-4,
        "r2={} r1={}",
        r2.residual_norm,
        r1.residual_norm
    );
}

#[test]
fn test_weak_mp_decompose_runs() {
    let dict = identity_dict(6);
    let signal = vec![0.5, -1.0, 2.0, 0.0, 0.0, 0.3];
    let wmp = WeakMatchingPursuit {
        threshold_ratio: 0.5,
        n_atoms: 3,
    };
    let result = wmp.decompose(&signal, &dict);
    assert!(!result.atoms.is_empty());
    assert!(result.residual_norm >= 0.0);
}

#[test]
fn test_subspace_pursuit_runs() {
    let dict = identity_dict(5);
    let signal = vec![0.0, 1.5, 0.0, -0.8, 0.0];
    let sp = SubspacePursuit {
        n_nonzero: 2,
        max_iter: 10,
    };
    let result = sp.recover(&signal, &dict);
    assert_eq!(result.atoms.len(), 2);
    assert!(result.residual_norm >= 0.0);
}

// ── SparseRegression ──────────────────────────────────────────────────────────

#[test]
fn test_lasso_regression_fit_shape() {
    let x_data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, -(i as f32)]).collect();
    let y: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let lasso = LassoRegression {
        lambda: 0.1,
        max_iter: 100,
        tol: 1e-5,
    };
    let coefs = lasso.fit(&x_data, &y);
    assert_eq!(coefs.len(), 2);
}

#[test]
fn test_elastic_net_fit_shape() {
    let x_data: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32, (i as f32).powi(2)]).collect();
    let y: Vec<f32> = (0..8).map(|i| (i as f32) * 2.0).collect();
    let en = ElasticNetRegression {
        alpha: 0.1,
        l1_ratio: 0.5,
        max_iter: 200,
    };
    let coefs = en.fit(&x_data, &y);
    assert_eq!(coefs.len(), 2);
}

#[test]
fn test_sparse_group_lasso_fit_shape() {
    let x_data: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32, -(i as f32), 1.0, (i as f32 * 0.5)])
        .collect();
    let y: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
    let sgl = SparseGroupLasso {
        groups: vec![vec![0, 1], vec![2, 3]],
        lambda1: 0.1,
        lambda2: 0.2,
        max_iter: 100,
    };
    let coefs = sgl.fit(&x_data, &y);
    assert_eq!(coefs.len(), 4);
}

// ── Metrics ───────────────────────────────────────────────────────────────────

#[test]
fn test_reconstruction_error_zero() {
    let s = vec![1.0, -1.0, 2.0];
    let err = reconstruction_error(&s, &s);
    assert!(err.abs() < 1e-6, "err={err}");
}

#[test]
fn test_coherence_bound_positive() {
    let b = coherence_bound(10, 5);
    assert!(b > 0.0, "b={b}");
}

#[test]
fn test_recovery_quality_finite() {
    let orig = vec![1.0, -0.5, 2.0];
    let rec = vec![0.9, -0.6, 1.8];
    let psnr = recovery_quality(&orig, &rec);
    assert!(psnr.is_finite(), "psnr={psnr}");
}

// ── BigBirdAttention ──────────────────────────────────────────────────────────

#[test]
fn test_bigbird_attention_mask_shape() {
    let mut rng = make_rng();
    let attn = BigBirdAttention {
        seq_len: 16,
        n_heads: 4,
        head_dim: 8,
        block_size: 4,
        n_random: 2,
        n_global: 2,
    };
    let mask = attn.compute_attention_mask(&mut rng);
    assert_eq!(mask.len(), 16);
    assert_eq!(mask[0].len(), 16);
}

#[test]
fn test_bigbird_global_tokens_attend_all() {
    let mut rng = make_rng();
    let n_global = 2;
    let seq_len = 12;
    let attn = BigBirdAttention {
        seq_len,
        n_heads: 2,
        head_dim: 4,
        block_size: 3,
        n_random: 1,
        n_global,
    };
    let mask = attn.compute_attention_mask(&mut rng);
    // Global tokens must attend to all positions
    for g in 0..n_global {
        for j in 0..seq_len {
            assert!(mask[g][j], "global token {g} should attend to {j}");
        }
    }
}

#[test]
fn test_bigbird_sparse_attention_output_shape() {
    let mut rng = make_rng();
    let seq_len = 8;
    let head_dim = 4;
    let attn = BigBirdAttention {
        seq_len,
        n_heads: 2,
        head_dim,
        block_size: 2,
        n_random: 1,
        n_global: 1,
    };
    let mask = attn.compute_attention_mask(&mut rng);
    let (q, k, v) = init_attention_matrices(seq_len, head_dim, &mut rng);
    let out = attn.sparse_attention(&q, &k, &v, &mask);
    assert_eq!(out.len(), seq_len);
    assert_eq!(out[0].len(), head_dim);
}

#[test]
fn test_bigbird_mask_sparsity_range() {
    let mut rng = make_rng();
    let attn = BigBirdAttention {
        seq_len: 20,
        n_heads: 2,
        head_dim: 4,
        block_size: 4,
        n_random: 2,
        n_global: 1,
    };
    let mask = attn.compute_attention_mask(&mut rng);
    let sparsity = BigBirdAttention::mask_sparsity(&mask);
    assert!(sparsity > 0.0 && sparsity <= 1.0, "sparsity={sparsity}");
}

// ── SparseSlidingWindowAttention ───────────────────────────────────────────────────────

#[test]
fn test_longformer_mask_shape() {
    let attn = SparseSlidingWindowAttention {
        seq_len: 16,
        n_heads: 2,
        head_dim: 4,
        window_size: 2,
        global_tokens: vec![0],
    };
    let mask = attn.compute_attention_mask();
    assert_eq!(mask.len(), 16);
    assert_eq!(mask[0].len(), 16);
}

#[test]
fn test_longformer_global_token_attends_all() {
    let global_tok = 0;
    let seq_len = 10;
    let attn = SparseSlidingWindowAttention {
        seq_len,
        n_heads: 1,
        head_dim: 4,
        window_size: 1,
        global_tokens: vec![global_tok],
    };
    let mask = attn.compute_attention_mask();
    for j in 0..seq_len {
        assert!(mask[global_tok][j], "global token should attend to position {j}");
    }
}

#[test]
fn test_longformer_local_attention_output_shape() {
    let mut rng = make_rng();
    let seq_len = 8;
    let head_dim = 4;
    let attn = SparseSlidingWindowAttention {
        seq_len,
        n_heads: 1,
        head_dim,
        window_size: 2,
        global_tokens: vec![0],
    };
    let (q, k, v) = init_attention_matrices(seq_len, head_dim, &mut rng);
    let out = attn.local_attention(&q, &k, &v);
    assert_eq!(out.len(), seq_len);
    assert_eq!(out[0].len(), head_dim);
}

// ── SparseAttentionRouter ─────────────────────────────────────────────────────

#[test]
fn test_sparse_attention_router_output_shape() {
    let mut rng = make_rng();
    let seq_len = 8;
    let dim = 4;
    let top_k = 3;
    let router = SparseAttentionRouter::new(seq_len, dim, top_k, &mut rng);
    let queries: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![0.5_f32; dim]).collect();
    let keys: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![0.3_f32; dim]).collect();
    let values: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![1.0_f32; dim]).collect();
    let out = router.attend(&queries, &keys, &values);
    assert_eq!(out.len(), seq_len);
    assert_eq!(out[0].len(), dim);
}

#[test]
fn test_sparse_attention_router_top_k_selection() {
    let mut rng = make_rng();
    let seq_len = 10;
    let dim = 4;
    let top_k = 3;
    let router = SparseAttentionRouter::new(seq_len, dim, top_k, &mut rng);
    let queries: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![1.0_f32; dim]).collect();
    let keys: Vec<Vec<f32>> = (0..seq_len).map(|_| vec![1.0_f32; dim]).collect();
    let routing = router.route(&queries, &keys);
    for (indices, weights) in &routing {
        assert_eq!(indices.len(), top_k.min(seq_len));
        let sum_w: f32 = weights.iter().sum();
        assert!((sum_w - 1.0).abs() < 1e-4, "weights should sum to 1, got {sum_w}");
    }
}

// ── SparsePositionEncoding ────────────────────────────────────────────────────

#[test]
fn test_sparse_position_encoding_bucket_range() {
    let mut rng = make_rng();
    let pe = SparsePositionEncoding::new(32, 8, 16, &mut rng);
    for rel in [-10i32, -3, 0, 5, 15, 30] {
        let b = pe.relative_position_bucket(rel);
        assert!(b < pe.n_buckets, "bucket {b} out of range for rel={rel}");
    }
}

#[test]
fn test_sparse_position_encoding_bias_shape() {
    let mut rng = make_rng();
    let seq_len = 6;
    let dim = 8;
    let pe = SparsePositionEncoding::new(32, dim, 16, &mut rng);
    let biases = pe.get_position_biases(seq_len);
    assert_eq!(biases.len(), seq_len * seq_len);
    assert_eq!(biases[0].len(), dim);
}

// ── GroupLasso ────────────────────────────────────────────────────────────────

#[test]
fn test_group_lasso_fit_shape() {
    let x_data: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![i as f32 * 0.1, -(i as f32 * 0.1), (i as f32 * 0.05)])
        .collect();
    let y: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
    let gl = GroupLasso {
        groups: vec![vec![0, 1], vec![2]],
        lambda: 0.1,
        learning_rate: 0.01,
        max_iter: 100,
    };
    let w = gl.fit(&x_data, &y);
    assert_eq!(w.len(), 3);
}

#[test]
fn test_group_lasso_sparsity_computation() {
    let gl = GroupLasso {
        groups: vec![vec![0, 1], vec![2, 3]],
        lambda: 0.1,
        learning_rate: 0.01,
        max_iter: 50,
    };
    let w = vec![0.0, 0.0, 1.0, 0.5];
    let sparsity = gl.group_sparsity(&w);
    // First group is all zeros, second is not
    assert!((sparsity - 0.5).abs() < 1e-5, "sparsity={sparsity}");
}

// ── StructuredPruningMask ─────────────────────────────────────────────────────

#[test]
fn test_nm_24_sparsity() {
    let pruner = StructuredPruningMask::nm_24();
    let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (pruned, mask) = pruner.apply(&weights);
    // Exactly 2 out of every 4 should be zero
    assert_eq!(pruned.len(), 8);
    assert_eq!(mask.len(), 8);
    let sparsity = StructuredPruningMask::sparsity(&mask);
    assert!((sparsity - 0.5).abs() < 1e-5, "sparsity={sparsity}");
}

#[test]
fn test_structured_pruning_keeps_largest() {
    let pruner = StructuredPruningMask { n_keep: 1, group_size: 4 };
    let weights = vec![0.1_f32, 5.0, 0.3, 0.2];
    let (pruned, mask) = pruner.apply(&weights);
    // Should keep index 1 (5.0 is largest)
    assert!(mask[1], "largest element should be kept");
    assert_eq!(pruned[1], 5.0_f32);
}

// ── ChannelPruner ─────────────────────────────────────────────────────────────

#[test]
fn test_channel_pruner_l1_importance() {
    let filters: Vec<Vec<f32>> = vec![
        vec![0.1, 0.1, 0.1],
        vec![2.0, 2.0, 2.0],
        vec![0.5, 0.5, 0.5],
    ];
    let pruner = ChannelPruner {
        prune_ratio: 0.333,
        criterion: ChannelImportanceCriterion::L1Norm,
    };
    let importance = pruner.compute_importance(&filters, None);
    assert_eq!(importance.len(), 3);
    // Second filter should have highest importance
    assert!(importance[1] > importance[0], "filter 1 should be more important");
}

#[test]
fn test_channel_pruner_mask_shape() {
    let filters: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32; 4]).collect();
    let pruner = ChannelPruner {
        prune_ratio: 0.5,
        criterion: ChannelImportanceCriterion::L1Norm,
    };
    let importance = pruner.compute_importance(&filters, None);
    let mask = pruner.prune_mask(&importance);
    assert_eq!(mask.len(), 6);
    let kept = mask.iter().filter(|&&b| b).count();
    assert_eq!(kept, 3); // 50% pruned → 50% kept
}

// ── LayerPruner ───────────────────────────────────────────────────────────────

#[test]
fn test_layer_pruner_importance_nonneg() {
    let pruner = LayerPruner { importance_threshold: 0.01 };
    let w = vec![1.0_f32; 8];
    let g = vec![0.1_f32; 8];
    let imp = pruner.layer_importance(&w, &g);
    assert!(imp >= 0.0, "importance should be non-negative");
}

#[test]
fn test_layer_pruner_select_layers() {
    let pruner = LayerPruner { importance_threshold: 0.5 };
    let importances = vec![0.1, 0.8, 0.3, 1.2];
    let mask = pruner.select_layers_to_prune(&importances);
    assert_eq!(mask, vec![false, true, false, true]);
}

// ── SparseCodingLayer ─────────────────────────────────────────────────────────

#[test]
fn test_sparse_coding_layer_output_shape() {
    let mut rng = make_rng();
    let layer = SparseCodingLayer::new(8, 16, 3, 0.1, &mut rng);
    let x = vec![1.0_f32; 8];
    let z = layer.forward(&x);
    assert_eq!(z.len(), 16);
}

#[test]
fn test_sparse_coding_layer_sparsity() {
    let mut rng = make_rng();
    let layer = SparseCodingLayer::new(8, 32, 5, 0.5, &mut rng);
    let x = vec![0.5_f32; 8];
    let z = layer.forward(&x);
    let zeros = z.iter().filter(|&&v| v == 0.0).count();
    // With lambda=0.5, significant sparsity expected
    assert!(zeros > 0, "some entries should be zero with high lambda");
}

#[test]
fn test_sparse_coding_layer_reconstruct_shape() {
    let mut rng = make_rng();
    let layer = SparseCodingLayer::new(8, 12, 3, 0.1, &mut rng);
    let z = vec![0.5_f32; 12];
    let x_hat = layer.reconstruct(&z);
    assert_eq!(x_hat.len(), 8);
}

#[test]
fn test_sparse_coding_layer_reconstruction_loss_nonneg() {
    let mut rng = make_rng();
    let layer = SparseCodingLayer::new(6, 10, 3, 0.1, &mut rng);
    let x = vec![1.0, -1.0, 0.5, 0.0, 2.0, -0.5];
    let loss = layer.reconstruction_loss(&x);
    assert!(loss >= 0.0, "reconstruction loss should be non-negative");
}

// ── ListaNetwork ──────────────────────────────────────────────────────────────

#[test]
fn test_lista_network_forward_shape() {
    let mut rng = make_rng();
    let lista = ListaNetwork::new(8, 16, 4, 0.1, &mut rng);
    let x = vec![1.0_f32; 8];
    let z = lista.forward(&x);
    assert_eq!(z.len(), 16);
}

#[test]
fn test_lista_network_output_sparsity() {
    let mut rng = make_rng();
    let lista = ListaNetwork::new(8, 32, 5, 0.5, &mut rng);
    let x = vec![0.1_f32; 8];
    let sparsity = lista.output_sparsity(&x);
    assert!((0.0..=1.0).contains(&sparsity), "sparsity out of range: {sparsity}");
}

// ── PredictiveCodingLayer ─────────────────────────────────────────────────────

#[test]
fn test_predictive_coding_infer_shape() {
    let mut rng = make_rng();
    let layer = PredictiveCodingLayer::new(8, 4, &mut rng);
    let x = vec![1.0_f32; 8];
    let (r, e) = layer.infer(&x);
    assert_eq!(r.len(), 4);
    assert_eq!(e.len(), 8);
}

#[test]
fn test_predictive_coding_free_energy_nonneg() {
    let mut rng = make_rng();
    let layer = PredictiveCodingLayer::new(6, 3, &mut rng);
    let x = vec![0.5_f32; 6];
    let fe = layer.free_energy(&x);
    assert!(fe >= 0.0, "free energy should be non-negative");
}

// ── CompressedSensingMatrix ───────────────────────────────────────────────────

#[test]
fn test_cs_matrix_gaussian_shape() {
    let mut rng = make_rng();
    let mat = CompressedSensingMatrix::new(10, 20, CsMatrixType::Gaussian, &mut rng);
    assert_eq!(mat.m_rows, 10);
    assert_eq!(mat.n_cols, 20);
    assert_eq!(mat.matrix.len(), 10);
    assert_eq!(mat.matrix[0].len(), 20);
}

#[test]
fn test_cs_matrix_bernoulli_shape() {
    let mut rng = make_rng();
    let mat = CompressedSensingMatrix::new(8, 16, CsMatrixType::Bernoulli, &mut rng);
    assert_eq!(mat.matrix.len(), 8);
}

#[test]
fn test_cs_matrix_measure_output_length() {
    let mut rng = make_rng();
    let mat = CompressedSensingMatrix::new(6, 12, CsMatrixType::Gaussian, &mut rng);
    let x = vec![1.0_f32; 12];
    let y = mat.measure(&x);
    assert_eq!(y.len(), 6);
}

#[test]
fn test_cs_matrix_sufficient_measurements() {
    let m = CompressedSensingMatrix::sufficient_measurements(100, 5);
    assert!(m > 0, "sufficient measurements should be positive");
    assert!(m <= 100, "sufficient measurements should be less than n for reasonable sparsity");
}

#[test]
fn test_cs_matrix_hadamard_shape() {
    let mut rng = make_rng();
    let mat = CompressedSensingMatrix::new(4, 8, CsMatrixType::SubsampledHadamard, &mut rng);
    assert!(mat.matrix.len() <= 4);
}

// ── BasisPursuitDenoise ───────────────────────────────────────────────────────

#[test]
fn test_bpdn_solve_shape() {
    let mut rng = make_rng();
    let csmat = CompressedSensingMatrix::new(8, 16, CsMatrixType::Gaussian, &mut rng);
    let x_true = vec![1.0_f32, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = csmat.measure(&x_true);
    let bpdn = BasisPursuitDenoise {
        sigma: 0.01,
        rho: 1.0,
        max_iter: 50,
        tol: 1e-4,
    };
    let x_hat = bpdn.solve(&csmat.matrix, &b);
    assert_eq!(x_hat.len(), 16);
}

#[test]
fn test_bpdn_residual_norm_nonneg() {
    let a = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
    let b = vec![1.0_f32, 2.0];
    let x = vec![1.0_f32, 2.0];
    let res = BasisPursuitDenoise::residual_norm(&a, &b, &x);
    assert!(res.abs() < 1e-5, "residual should be near zero, got {res}");
}

// ── RecoveryGuarantees ────────────────────────────────────────────────────────

#[test]
fn test_recovery_guarantees_rip_constant_nonneg() {
    let mut rng = make_rng();
    let csmat = CompressedSensingMatrix::new(12, 20, CsMatrixType::Gaussian, &mut rng);
    let delta = RecoveryGuarantees::rip_constant(&csmat.matrix, 2, 20, &mut rng);
    assert!(delta >= 0.0, "RIP constant should be non-negative");
}

#[test]
fn test_recovery_guarantees_phase_transition_range() {
    for rho in [0.01, 0.05, 0.1, 0.2, 0.5] {
        let delta = RecoveryGuarantees::phase_transition(rho);
        assert!((0.0..=1.0).contains(&delta), "phase transition out of range for rho={rho}: delta={delta}");
    }
}

#[test]
fn test_recovery_guarantees_rip_sufficient() {
    assert!(RecoveryGuarantees::is_rip_sufficient(0.1), "delta=0.1 should be sufficient");
    assert!(!RecoveryGuarantees::is_rip_sufficient(0.5), "delta=0.5 should NOT be sufficient");
}

// ── CsMetrics ─────────────────────────────────────────────────────────────────

#[test]
fn test_cs_metrics_recovery_snr_perfect() {
    let x = vec![1.0_f32, -1.0, 2.0];
    let snr = CsMetrics::recovery_snr(&x, &x);
    assert!(snr > 90.0, "perfect recovery should have very high SNR: {snr}");
}

#[test]
fn test_cs_metrics_support_recovery_rate() {
    let true_support = vec![0, 3, 7];
    let mut recovered = vec![0.0_f32; 10];
    recovered[0] = 1.0;
    recovered[3] = -1.5;
    recovered[7] = 0.8;
    let rate = CsMetrics::support_recovery_rate(&true_support, &recovered, 0.1);
    assert!((rate - 1.0).abs() < 1e-5, "should recover full support: rate={rate}");
}

#[test]
fn test_cs_metrics_exact_support_recovery() {
    let true_support = vec![1, 4];
    let mut recovered = vec![0.0_f32; 6];
    recovered[1] = 2.0;
    recovered[4] = -1.0;
    let exact = CsMetrics::exact_support_recovery(&true_support, &recovered, 0.1);
    assert!(exact, "support should be exactly recovered");
}

#[test]
fn test_cs_metrics_normalized_error_zero() {
    let x = vec![1.0_f32, -1.0, 2.0];
    let err = CsMetrics::normalized_error(&x, &x);
    assert!(err.abs() < 1e-6, "same signal should have zero error: {err}");
}

#[test]
fn test_cs_metrics_sparsity() {
    let x = vec![0.0_f32, 1.0, 0.0, -2.0, 0.0];
    let sparsity = CsMetrics::recovered_sparsity(&x, 0.5);
    assert!((sparsity - 0.6).abs() < 1e-5, "3 of 5 below threshold 0.5: {sparsity}");
}
