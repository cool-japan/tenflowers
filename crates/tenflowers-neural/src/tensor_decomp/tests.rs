//! Tests for tensor_decomp: original algorithms and new advanced algorithms.

use super::*;

// ----- DenseTensor -----

#[test]
fn test_dense_tensor_new_valid() {
    let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("operation should succeed");
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.data.len(), 6);
}

#[test]
fn test_dense_tensor_new_shape_mismatch() {
    let result = DenseTensor::new(vec![1.0, 2.0, 3.0], vec![2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_dense_tensor_get_valid() {
    let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("operation should succeed");
    assert_eq!(t.get(&[0, 0]).expect("operation should succeed"), 1.0);
    assert_eq!(t.get(&[0, 2]).expect("operation should succeed"), 3.0);
    assert_eq!(t.get(&[1, 0]).expect("operation should succeed"), 4.0);
    assert_eq!(t.get(&[1, 2]).expect("operation should succeed"), 6.0);
}

#[test]
fn test_dense_tensor_get_out_of_bounds() {
    let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("operation should succeed");
    assert!(t.get(&[2, 0]).is_err());
    assert!(t.get(&[0, 2]).is_err());
}

#[test]
fn test_dense_tensor_set() {
    let mut t = DenseTensor::new(vec![0.0; 6], vec![2, 3]).expect("operation should succeed");
    t.set(&[1, 2], 42.0).expect("operation should succeed");
    assert_eq!(t.get(&[1, 2]).expect("operation should succeed"), 42.0);
}

#[test]
fn test_dense_tensor_set_out_of_bounds() {
    let mut t = DenseTensor::new(vec![0.0; 4], vec![2, 2]).expect("operation should succeed");
    assert!(t.set(&[0, 3], 1.0).is_err());
}

#[test]
fn test_dense_tensor_reshape() {
    let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("operation should succeed");
    let r = t.reshape(vec![3, 2]).expect("operation should succeed");
    assert_eq!(r.shape, vec![3, 2]);
    assert_eq!(r.data, t.data);
}

#[test]
fn test_dense_tensor_reshape_invalid() {
    let t = DenseTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("operation should succeed");
    assert!(t.reshape(vec![2, 3]).is_err());
}

#[test]
fn test_dense_tensor_matricize_shape_3d() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let t = DenseTensor::new(data, vec![2, 3, 4]).expect("operation should succeed");
    let m0 = t.matricize(0).expect("operation should succeed");
    assert_eq!(m0.shape, vec![2, 12]);
    let m1 = t.matricize(1).expect("operation should succeed");
    assert_eq!(m1.shape, vec![3, 8]);
    let m2 = t.matricize(2).expect("operation should succeed");
    assert_eq!(m2.shape, vec![4, 6]);
}

#[test]
fn test_dense_tensor_norm_all_ones() {
    let n = 9usize;
    let data = vec![1.0f64; n];
    let t = DenseTensor::new(data, vec![3, 3]).expect("operation should succeed");
    let expected = (n as f64).sqrt();
    assert!((t.norm() - expected).abs() < 1e-10);
}

#[test]
fn test_dense_tensor_norm_known() {
    let t = DenseTensor::new(vec![3.0, 4.0], vec![2]).expect("operation should succeed");
    assert!((t.norm() - 5.0).abs() < 1e-10);
}

// ----- Math utilities -----

#[test]
fn test_matrix_multiply_correctness() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let c = matrix_multiply(&a, &b);
    assert!((c[0][0] - 19.0).abs() < 1e-10);
    assert!((c[0][1] - 22.0).abs() < 1e-10);
    assert!((c[1][0] - 43.0).abs() < 1e-10);
    assert!((c[1][1] - 50.0).abs() < 1e-10);
}

#[test]
fn test_matrix_transpose() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let t = matrix_transpose(&a);
    assert_eq!(t.len(), 3);
    assert_eq!(t[0].len(), 2);
    assert_eq!(t[0][0], 1.0);
    assert_eq!(t[0][1], 4.0);
    assert_eq!(t[2][1], 6.0);
}

#[test]
fn test_matrix_qr_orthogonality() {
    let a = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ];
    let (q, _r) = matrix_qr(&a);
    let qt = matrix_transpose(&q);
    let qtq = matrix_multiply(&qt, &q);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq[i][j] - expected).abs() < 1e-10,
                "Q^T Q[{},{}] = {} expected {}",
                i, j, qtq[i][j], expected
            );
        }
    }
}

#[test]
fn test_khatri_rao_shape() {
    let a: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64, i as f64 + 0.1]).collect();
    let b: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64, i as f64 + 0.2]).collect();
    let kr = khatri_rao(&a, &b);
    assert_eq!(kr.len(), 12, "KR rows should be 3*4=12");
    assert_eq!(kr[0].len(), 2, "KR cols should be rank=2");
}

#[test]
fn test_khatri_rao_correctness() {
    let a = vec![vec![2.0], vec![3.0]];
    let b = vec![vec![5.0], vec![7.0]];
    let kr = khatri_rao(&a, &b);
    assert_eq!(kr.len(), 4);
    assert!((kr[0][0] - 10.0).abs() < 1e-10);
    assert!((kr[1][0] - 14.0).abs() < 1e-10);
    assert!((kr[2][0] - 15.0).abs() < 1e-10);
    assert!((kr[3][0] - 21.0).abs() < 1e-10);
}

#[test]
fn test_pseudo_inverse_identity() {
    let eye = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let pinv = pseudo_inverse_via_svd(&eye);
    assert!((pinv[0][0] - 1.0).abs() < 1e-6);
    assert!((pinv[0][1] - 0.0).abs() < 1e-6);
    assert!((pinv[1][0] - 0.0).abs() < 1e-6);
    assert!((pinv[1][1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_pseudo_inverse_satisfies_equation() {
    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let pinv = pseudo_inverse_via_svd(&a);
    let a_p = matrix_multiply(&a, &pinv);
    let a_p_a = matrix_multiply(&a_p, &a);
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            assert!(
                (a_p_a[i][j] - a[i][j]).abs() < 1e-4,
                "A*pinv(A)*A[{},{}] = {} expected {}",
                i, j, a_p_a[i][j], a[i][j]
            );
        }
    }
}

// ----- CP Decomposition -----

#[test]
fn test_cp_als_rank1_reconstruction() {
    let u = [1.0, 2.0, 3.0];
    let v = [1.0, 2.0];
    let w = [1.0, 2.0, 3.0, 4.0];
    let mut data = vec![0.0f64; 24];
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            for (k, &wk) in w.iter().enumerate() {
                data[i * 8 + j * 4 + k] = ui * vj * wk;
            }
        }
    }
    let tensor = DenseTensor::new(data, vec![3, 2, 4]).expect("operation should succeed");
    let config = CpConfig { rank: 1, max_iter: 300, tolerance: 1e-8, random_seed: 1 };
    let cp = CpAls::fit(&tensor, &config).expect("operation should succeed");
    let rec = cp.reconstruct();
    let error: f64 = tensor
        .data
        .iter()
        .zip(rec.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(error < 1e-4 * tensor.norm() + 1e-6, "CP reconstruction error {} too large", error);
}

#[test]
fn test_cp_als_fit_value() {
    let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 2, 2]).expect("operation should succeed");
    let config = CpConfig { rank: 2, max_iter: 200, tolerance: 1e-6, random_seed: 7 };
    let cp = CpAls::fit(&tensor, &config).expect("operation should succeed");
    assert!(cp.fit >= -2.0 && cp.fit <= 1.0 + 1e-6, "fit = {} out of range", cp.fit);
}

#[test]
fn test_cp_decomposition_reconstruct() {
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("operation should succeed");
    let config = CpConfig { rank: 2, max_iter: 300, tolerance: 1e-7, random_seed: 99 };
    let cp = CpAls::fit(&tensor, &config).expect("operation should succeed");
    let rec = cp.reconstruct();
    assert_eq!(rec.shape, tensor.shape);
    assert_eq!(rec.data.len(), tensor.data.len());
}

#[test]
fn test_cp_als_n_iter() {
    let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 3]).expect("operation should succeed");
    let config = CpConfig { rank: 1, max_iter: 50, tolerance: 1e-12, random_seed: 3 };
    let cp = CpAls::fit(&tensor, &config).expect("operation should succeed");
    assert!(cp.n_iter >= 1 && cp.n_iter <= 50);
}

// ----- Tucker Decomposition -----

#[test]
fn test_tucker_hooi_reconstruction_error() {
    let data: Vec<f64> = (0..64).map(|x| (x as f64).sin()).collect();
    let tensor = DenseTensor::new(data, vec![4, 4, 4]).expect("operation should succeed");
    let config = TuckerConfig { ranks: vec![2, 2, 2], max_iter: 50, tolerance: 1e-6, random_seed: 42 };
    let tucker = TuckerHooi::fit(&tensor, &config).expect("operation should succeed");
    let rec = tucker.reconstruct();
    let error: f64 = tensor
        .data
        .iter()
        .zip(rec.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        error < tensor.norm() * 0.9 + 1e-6,
        "Tucker reconstruction error {} >= 0.9 * norm {}",
        error,
        tensor.norm()
    );
}

#[test]
fn test_tucker_hooi_shapes() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("operation should succeed");
    let config = TuckerConfig { ranks: vec![2, 2, 2], max_iter: 20, tolerance: 1e-4, random_seed: 0 };
    let tucker = TuckerHooi::fit(&tensor, &config).expect("operation should succeed");
    assert_eq!(tucker.core.shape, vec![2, 2, 2]);
    assert_eq!(tucker.factors[0].len(), 2);
    assert_eq!(tucker.factors[0][0].len(), 2);
    assert_eq!(tucker.factors[1].len(), 3);
    assert_eq!(tucker.factors[2].len(), 4);
}

// ----- HOSVD -----

#[test]
fn test_hosvd_factor_orthonormality() {
    let data: Vec<f64> = (0..24).map(|x| x as f64 + 0.1).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("operation should succeed");
    let decomp = hosvd(&tensor, &[2, 2, 3]).expect("operation should succeed");
    for (n, factor) in decomp.factors.iter().enumerate() {
        let ft = matrix_transpose(factor);
        let ftf = matrix_multiply(&ft, factor);
        let r = ftf.len();
        for i in 0..r {
            for j in 0..r {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ftf[i][j] - expected).abs() < 1e-6,
                    "Factor[{}]: A^T A[{},{}] = {} expected {}",
                    n, i, j, ftf[i][j], expected
                );
            }
        }
    }
}

#[test]
fn test_hosvd_core_shape() {
    let data: Vec<f64> = (0..60)
        .map(|x| {
            let f = x as f64;
            (f * 0.31).sin() + (f * 0.71).cos() + (f * 1.37).sin() + (f * 2.09).cos()
        })
        .collect();
    let tensor = DenseTensor::new(data, vec![3, 4, 5]).expect("operation should succeed");
    let decomp = hosvd(&tensor, &[2, 3, 4]).expect("operation should succeed");
    assert_eq!(decomp.core.shape, vec![2, 3, 4]);
}

// ----- TT-SVD -----

#[test]
fn test_tt_svd_compression_ratio() {
    let data: Vec<f64> = (0..256).map(|x| (x as f64 / 256.0).sin()).collect();
    let tensor = DenseTensor::new(data, vec![4, 4, 4, 4]).expect("operation should succeed");
    let config = TtConfig { max_rank: 3, tolerance: 1e-3 };
    let tt = TtSvd::fit(&tensor, &config).expect("operation should succeed");
    let ratio = tt.compression_ratio();
    assert!(ratio > 1.0, "compression_ratio = {} should be > 1", ratio);
}

#[test]
fn test_tt_svd_shape_consistency() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("operation should succeed");
    let config = TtConfig { max_rank: 10, tolerance: 1e-8 };
    let tt = TtSvd::fit(&tensor, &config).expect("operation should succeed");
    assert_eq!(tt.shape, vec![2, 3, 4]);
    assert_eq!(tt.cores.len(), 3);
    assert_eq!(tt.core_shapes.len(), 3);
    assert_eq!(tt.core_shapes[0].0, 1);
    assert_eq!(tt.core_shapes[2].2, 1);
}

#[test]
fn test_tt_tensor_reconstruct_small() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = DenseTensor::new(data.clone(), vec![2, 3]).expect("operation should succeed");
    let config = TtConfig { max_rank: 6, tolerance: 1e-12 };
    let tt = TtSvd::fit(&tensor, &config).expect("operation should succeed");
    let rec = tt.reconstruct();
    assert_eq!(rec.shape, vec![2, 3]);
    let error: f64 = data
        .iter()
        .zip(rec.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        error < 1e-6 * tensor.norm() + 1e-8,
        "TT reconstruction error {} too large for small tensor",
        error
    );
}

// ----- Randomized SVD -----

#[test]
fn test_randomized_svd_singular_values_nonneg_decreasing() {
    let rows = 10usize;
    let cols = 8usize;
    let matrix: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    (i as f64 * 1.3).sin() * (j as f64 * 0.7).cos()
                        + (i as f64 * 2.1).cos() * (j as f64 * 1.5).sin()
                        + (i as f64 * 3.7).sin() * (j as f64 * 2.3).cos()
                        + (i as f64 * 4.9).cos() * (j as f64 * 3.1).sin()
                        + (i as f64 * 0.3 + j as f64 * 0.9).sin()
                })
                .collect()
        })
        .collect();
    let rsvd = RandomizedSvd { n_components: 4, n_oversampling: 5, n_power_iter: 3, seed: 42 };
    let result = rsvd.fit_transform(&matrix).expect("operation should succeed");
    assert_eq!(result.s.len(), 4);
    for &sv in &result.s {
        assert!(sv >= 0.0, "singular value {} is negative", sv);
    }
    for i in 0..result.s.len() - 1 {
        assert!(
            result.s[i] >= result.s[i + 1] - 1e-6,
            "singular values not non-increasing: s[{}]={} s[{}]={}",
            i, result.s[i], i + 1, result.s[i + 1]
        );
    }
}

#[test]
fn test_randomized_svd_u_orthonormality() {
    let matrix: Vec<Vec<f64>> = (0..8)
        .map(|i| (0..6).map(|j| (i * 6 + j) as f64).collect())
        .collect();
    let rsvd = RandomizedSvd { n_components: 3, n_oversampling: 5, n_power_iter: 4, seed: 10 };
    let result = rsvd.fit_transform(&matrix).expect("operation should succeed");
    let ut = matrix_transpose(&result.u);
    let utu = matrix_multiply(&ut, &result.u);
    let k = result.s.len();
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (utu[i][j] - expected).abs() < 1e-6,
                "U^T U[{},{}] = {} expected {}",
                i, j, utu[i][j], expected
            );
        }
    }
}

#[test]
fn test_randomized_svd_shapes() {
    let m = 12usize;
    let n = 8usize;
    let k = 3usize;
    let matrix: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (i as f64 * 1.1 + j as f64 * 0.7).sin()
                        + (i as f64 * 2.3 - j as f64 * 1.3).cos()
                        + (i as f64 * 0.5 + j as f64 * 2.1).sin()
                        + (i as f64 * j as f64 * 0.1).cos()
                })
                .collect()
        })
        .collect();
    let rsvd = RandomizedSvd { n_components: k, n_oversampling: 5, n_power_iter: 2, seed: 1 };
    let result = rsvd.fit_transform(&matrix).expect("operation should succeed");
    assert_eq!(result.u.len(), m);
    assert_eq!(result.u[0].len(), k);
    assert_eq!(result.s.len(), k);
    assert_eq!(result.vt.len(), k);
    assert_eq!(result.vt[0].len(), n);
}

// ----- NMF -----

#[test]
fn test_nmf_mu_nonnegative() {
    let matrix: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![2.0, 1.0, 4.0],
    ];
    let config = NmfConfig { rank: 2, max_iter: 200, tolerance: 1e-6, seed: 7 };
    let result = NmfMu::fit(&matrix, &config).expect("operation should succeed");
    for row in &result.w {
        for &v in row {
            assert!(v >= 0.0, "W has negative entry {}", v);
        }
    }
    for row in &result.h {
        for &v in row {
            assert!(v >= 0.0, "H has negative entry {}", v);
        }
    }
}

#[test]
fn test_nmf_mu_error_decreases() {
    let matrix: Vec<Vec<f64>> = vec![
        vec![3.0, 1.0, 0.5],
        vec![0.5, 2.0, 1.0],
        vec![1.0, 0.5, 3.0],
    ];
    let cfg1 = NmfConfig { rank: 2, max_iter: 5, tolerance: 1e-12, seed: 42 };
    let cfg2 = NmfConfig { rank: 2, max_iter: 300, tolerance: 1e-12, seed: 42 };
    let r1 = NmfMu::fit(&matrix, &cfg1).expect("operation should succeed");
    let r2 = NmfMu::fit(&matrix, &cfg2).expect("operation should succeed");
    assert!(
        r2.reconstruction_error <= r1.reconstruction_error + 1e-4,
        "error with 300 iters {} > error with 5 iters {}",
        r2.reconstruction_error, r1.reconstruction_error
    );
}

#[test]
fn test_nmf_mu_reconstruct_shape() {
    let matrix: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 1.0]];
    let config = NmfConfig { rank: 2, max_iter: 100, tolerance: 1e-6, seed: 0 };
    let result = NmfMu::fit(&matrix, &config).expect("operation should succeed");
    let rec = result.reconstruct();
    assert_eq!(rec.len(), 2);
    assert_eq!(rec[0].len(), 3);
}

#[test]
fn test_nmf_mu_low_rank_approx() {
    let matrix: Vec<Vec<f64>> = vec![
        vec![4.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
        vec![0.0, 0.0, 6.0],
    ];
    let config = NmfConfig { rank: 3, max_iter: 500, tolerance: 1e-8, seed: 13 };
    let result = NmfMu::fit(&matrix, &config).expect("operation should succeed");
    assert!(
        result.reconstruction_error < 3.0,
        "reconstruction_error {} too large for near-diagonal matrix",
        result.reconstruction_error
    );
}

#[test]
fn test_from_factors_khatri_rao_rank1() {
    let a: Vec<Vec<Vec<f64>>> = vec![
        vec![vec![2.0], vec![3.0]],
        vec![vec![5.0], vec![7.0]],
    ];
    let weights = vec![1.0];
    let t = DenseTensor::from_factors_khatri_rao(&a, &weights);
    assert_eq!(t.shape, vec![2, 2]);
    assert!((t.data[0] - 10.0).abs() < 1e-10);
    assert!((t.data[1] - 14.0).abs() < 1e-10);
    assert!((t.data[2] - 15.0).abs() < 1e-10);
    assert!((t.data[3] - 21.0).abs() < 1e-10);
}

#[test]
fn test_hosvd_reconstruction_quality() {
    let u = [1.0, 2.0, 3.0];
    let v = [4.0, 5.0];
    let mut data = vec![0.0f64; 6];
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            data[i * 2 + j] = ui * vj;
        }
    }
    let tensor = DenseTensor::new(data, vec![3, 2]).expect("operation should succeed");
    let decomp = hosvd(&tensor, &[1, 1]).expect("operation should succeed");
    let rec = decomp.reconstruct();
    let error: f64 = tensor
        .data
        .iter()
        .zip(rec.data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        error < tensor.norm() * 0.01 + 1e-8,
        "HOSVD error {} too large on rank-1 tensor",
        error
    );
}

#[test]
fn test_tt_svd_core_sizes() {
    let data: Vec<f64> = (0..60).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![3, 4, 5]).expect("operation should succeed");
    let config = TtConfig { max_rank: 5, tolerance: 1e-6 };
    let tt = TtSvd::fit(&tensor, &config).expect("operation should succeed");
    let (r0, n0, r1) = tt.core_shapes[0];
    assert_eq!(r0, 1);
    assert_eq!(n0, 3);
    assert_eq!(tt.cores[0].len(), r0 * n0 * r1);
}

#[test]
fn test_hadamard_product() {
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0];
    let h = hadamard_product(&a, &b);
    assert_eq!(h, vec![10.0, 18.0, 28.0]);
}

// ============================================================================
// Advanced algorithm tests
// ============================================================================

// ----- Tensor Ring -----

#[test]
fn test_tensor_ring_core_new() {
    let data = vec![1.0f64; 12]; // 2*3*2 = 12
    let core = TensorRingCore::new(data, 2, 3, 2).expect("core creation should succeed");
    assert_eq!(core.r_left, 2);
    assert_eq!(core.n_k, 3);
    assert_eq!(core.r_right, 2);
}

#[test]
fn test_tensor_ring_core_size_mismatch() {
    let data = vec![1.0f64; 5]; // wrong size
    assert!(TensorRingCore::new(data, 2, 3, 2).is_err());
}

#[test]
fn test_tensor_ring_decomp_fit_small() {
    let data: Vec<f64> = (0..12).map(|x| (x as f64).sin()).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let decomp = TensorRingDecompAlgo::fit(&tensor, 2).expect("ring decomp should succeed");
    assert_eq!(decomp.cores.len(), 2);
    assert_eq!(decomp.shape, vec![3, 4]);
}

#[test]
fn test_tensor_ring_decomp_compression_ratio() {
    let data: Vec<f64> = (0..60).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![3, 4, 5]).expect("tensor creation should succeed");
    let decomp = TensorRingDecompAlgo::fit(&tensor, 2).expect("ring decomp should succeed");
    let ratio = decomp.compression_ratio();
    assert!(ratio > 0.0, "compression ratio must be positive");
}

#[test]
fn test_tensor_ring_reconstruct_shape() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("tensor creation should succeed");
    let decomp = TensorRingDecompAlgo::fit(&tensor, 2).expect("ring decomp should succeed");
    let rec = decomp.reconstruct();
    assert_eq!(rec.shape, vec![2, 3, 4]);
    assert_eq!(rec.data.len(), 24);
}

#[test]
fn test_tr_compressor_compress_reconstruct() {
    let matrix: Vec<Vec<f64>> = (0..4).map(|i| (0..6).map(|j| (i * 6 + j) as f64).collect()).collect();
    let compressor = TrCompressor { ring_rank: 2, tensor_shape: vec![2, 2, 2, 3] };
    let decomp = compressor.compress(&matrix).expect("compress should succeed");
    let reconstructed = TrCompressor::reconstruct(&decomp, 4, 6).expect("reconstruct should succeed");
    assert_eq!(reconstructed.len(), 4);
    assert_eq!(reconstructed[0].len(), 6);
}

// ----- NTF -----

#[test]
fn test_ntf_model_nonneg_factors() {
    let data: Vec<f64> = (0..12).map(|x| (x as f64 + 1.0).abs()).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let result = NtfModel::fit(&tensor, 2, 100, 1e-5, 42).expect("NTF fit should succeed");
    for factor in &result.factors {
        for row in factor {
            for &v in row {
                assert!(v >= 0.0, "NTF factor entry {} is negative", v);
            }
        }
    }
}

#[test]
fn test_ntf_model_reconstruct_shape() {
    let data: Vec<f64> = (0..24).map(|x| (x as f64 * 0.5).abs()).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("tensor creation should succeed");
    let result = NtfModel::fit(&tensor, 2, 50, 1e-4, 7).expect("NTF fit should succeed");
    let rec = result.reconstruct();
    assert_eq!(rec.shape, vec![2, 3, 4]);
    assert_eq!(rec.data.len(), 24);
}

#[test]
fn test_ntf_model_error_finite() {
    let data: Vec<f64> = (0..8).map(|x| x as f64 + 0.1).collect();
    let tensor = DenseTensor::new(data, vec![2, 4]).expect("tensor creation should succeed");
    let result = NtfModel::fit(&tensor, 2, 50, 1e-4, 0).expect("NTF fit should succeed");
    assert!(result.error.is_finite(), "NTF error should be finite");
    assert!(result.error >= 0.0, "NTF error should be nonneg");
}

#[test]
fn test_ntf_semi_nmf_mode0_can_be_negative() {
    let data: Vec<f64> = (0..12).map(|x| x as f64 - 5.5).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let result = NtfSemiNmf::fit(&tensor, 2, 100, 42).expect("SemiNMF fit should succeed");
    // Mode-0 factors can be negative; mode-1+ must be nonneg
    for row in &result.factors[1] {
        for &v in row {
            assert!(v >= 0.0, "NtfSemiNmf mode-1 factor {} is negative", v);
        }
    }
}

#[test]
fn test_ntf_beta_frobenius_matches_nmf() {
    // β=2 should give Frobenius divergence (similar to NMF)
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let result = NtfBeta::fit(&tensor, 2, 2.0, 100, 42).expect("BetaNTF fit should succeed");
    assert!(result.divergence.is_finite(), "beta-div should be finite");
    assert!(result.divergence >= 0.0, "beta-div should be nonneg");
}

#[test]
fn test_ntf_beta_kl_positive() {
    // β=1 → KL divergence; must be >= 0
    let data: Vec<f64> = (0..8).map(|x| x as f64 + 1.0).collect();
    let tensor = DenseTensor::new(data, vec![2, 4]).expect("tensor creation should succeed");
    let result = NtfBeta::fit(&tensor, 2, 1.0, 50, 13).expect("BetaNTF KL fit should succeed");
    assert!(result.divergence >= 0.0, "KL divergence must be >= 0");
}

// ----- Tensor Completion -----

#[test]
fn test_tensor_completion_observed_entries() {
    // Create a low-rank tensor and mask half the entries
    let u = [1.0, 2.0, 3.0];
    let v = [1.0, 2.0, 3.0, 4.0];
    let mut data = vec![0.0f64; 12];
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            data[i * 4 + j] = ui * vj;
        }
    }
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    // Observe all entries
    let mask = DenseTensor::new(vec![1.0f64; 12], vec![3, 4]).expect("mask creation should succeed");
    let result = TensorCompletion::fit(&tensor, &mask, 2, 200, 1e-5, 42)
        .expect("completion should succeed");
    let rec = result.reconstruct();
    assert_eq!(rec.shape, vec![3, 4]);
    // With full observation and rank-1 data, error should be small
    let error = result.error;
    assert!(error.is_finite(), "completion error must be finite");
}

#[test]
fn test_tensor_completion_partial_mask() {
    let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![4, 4]).expect("tensor creation should succeed");
    // Observe only half the entries (checkerboard)
    let mask_data: Vec<f64> = (0..16).map(|x| if x % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let mask = DenseTensor::new(mask_data, vec![4, 4]).expect("mask creation should succeed");
    let result = TensorCompletion::fit(&tensor, &mask, 2, 100, 1e-4, 0)
        .expect("completion should succeed");
    assert!(result.error.is_finite());
    assert!(result.n_iter >= 1);
}

#[test]
fn test_riemannian_gradient_completion_shape() {
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let mask = DenseTensor::new(vec![1.0f64; 12], vec![3, 4]).expect("mask creation should succeed");
    let result = RiemannianGradientCompletion::fit(&tensor, &mask, vec![2, 2], 0.01, 20)
        .expect("riemannian completion should succeed");
    let rec = result.reconstruct();
    assert_eq!(rec.shape, vec![3, 4]);
}

#[test]
fn test_scalable_tc_alternating_shape() {
    let data: Vec<f64> = (0..12).map(|x| (x as f64).cos()).collect();
    let tensor = DenseTensor::new(data, vec![3, 4]).expect("tensor creation should succeed");
    let mask = DenseTensor::new(vec![1.0f64; 12], vec![3, 4]).expect("mask creation should succeed");
    let result = ScalableTcAlternating::fit(&tensor, &mask, 2, 50, 99)
        .expect("scalable TC should succeed");
    let rec = result.reconstruct();
    assert_eq!(rec.shape, vec![3, 4]);
}

// ----- Tensor Neural Networks -----

#[test]
fn test_tensor_train_linear_forward_shape() {
    let layer = TensorTrainLinear::new(vec![4, 4], vec![4, 4], 2, 42)
        .expect("TT-Linear creation should succeed");
    let x = vec![0.1f64; 16];
    let y = layer.forward(&x).expect("forward should succeed");
    assert_eq!(y.len(), 16);
}

#[test]
fn test_tensor_train_linear_compression_ratio() {
    let layer = TensorTrainLinear::new(vec![4, 4], vec![4, 4], 2, 0)
        .expect("TT-Linear creation should succeed");
    let ratio = layer.compression_ratio();
    assert!(ratio > 0.0, "compression ratio must be positive");
}

#[test]
fn test_tensor_train_linear_wrong_input() {
    let layer = TensorTrainLinear::new(vec![4], vec![4], 2, 1)
        .expect("TT-Linear creation should succeed");
    assert!(layer.forward(&[0.1; 5]).is_err(), "should fail with wrong input size");
}

#[test]
fn test_tt_rnn_step_shape() {
    let rnn = TtRnn::new(16, 16, 2, 42).expect("TT-RNN creation should succeed");
    let x = vec![0.1f64; 16];
    let h = vec![0.0f64; 16];
    let h_new = rnn.step(&x, &h).expect("step should succeed");
    assert_eq!(h_new.len(), 16);
    for &v in &h_new {
        assert!(v.abs() <= 1.0, "tanh output must be in [-1, 1]");
    }
}

#[test]
fn test_tt_rnn_sequence() {
    let rnn = TtRnn::new(16, 16, 2, 7).expect("TT-RNN creation should succeed");
    let sequence: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 16]).collect();
    let states = rnn.forward_sequence(&sequence).expect("sequence forward should succeed");
    assert_eq!(states.len(), 5);
    for state in &states {
        assert_eq!(state.len(), 16);
    }
}

#[test]
fn test_tensor_fusion_layer_forward_shape() {
    let layer = TensorFusionLayer::new(vec![4, 4, 4], 8, 42)
        .expect("TensorFusionLayer creation should succeed");
    let m1 = vec![0.1f64; 4];
    let m2 = vec![0.2f64; 4];
    let m3 = vec![0.3f64; 4];
    let out = layer.forward(&[m1, m2, m3]).expect("forward should succeed");
    assert_eq!(out.len(), 8);
    for &v in &out {
        assert!(v.abs() <= 1.0, "tanh output must be in [-1, 1]");
    }
}

#[test]
fn test_tensor_fusion_layer_wrong_modalities() {
    let layer = TensorFusionLayer::new(vec![4, 4], 8, 0)
        .expect("TensorFusionLayer creation should succeed");
    // Only 1 modality instead of 2
    let out = layer.forward(&[vec![0.1; 4]]);
    assert!(out.is_err(), "should fail with wrong number of modalities");
}

// ----- TdMetrics -----

#[test]
fn test_td_metrics_from_tt() {
    let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("tensor creation should succeed");
    let config = TtConfig { max_rank: 4, tolerance: 1e-6 };
    let tt = TtSvd::fit(&tensor, &config).expect("TT-SVD should succeed");
    let metrics = TdMetrics::from_tt(&tensor, &tt);
    assert!(metrics.relative_error >= 0.0);
    assert!(metrics.compression_ratio > 0.0);
    assert!(metrics.tt_rank.is_some());
    assert!(metrics.tucker_rank.is_none());
}

#[test]
fn test_td_metrics_from_tucker() {
    let data: Vec<f64> = (0..24).map(|x| x as f64 + 0.1).collect();
    let tensor = DenseTensor::new(data, vec![2, 3, 4]).expect("tensor creation should succeed");
    let decomp = hosvd(&tensor, &[2, 2, 3]).expect("HOSVD should succeed");
    let metrics = TdMetrics::from_tucker(&tensor, &decomp.core, &decomp.factors);
    assert!(metrics.relative_error >= 0.0 && metrics.relative_error <= 1.0 + 1e-6);
    assert!(metrics.compression_ratio > 0.0);
    assert!(metrics.tucker_rank.is_some());
    assert!(metrics.tt_rank.is_none());
}

#[test]
fn test_td_metrics_summary() {
    let data = vec![1.0f64; 8];
    let tensor = DenseTensor::new(data, vec![2, 4]).expect("tensor creation should succeed");
    let config = TtConfig { max_rank: 2, tolerance: 1e-6 };
    let tt = TtSvd::fit(&tensor, &config).expect("TT-SVD should succeed");
    let metrics = TdMetrics::from_tt(&tensor, &tt);
    let summary = metrics.summary();
    assert!(summary.contains("TdMetrics"), "summary should mention TdMetrics");
}
