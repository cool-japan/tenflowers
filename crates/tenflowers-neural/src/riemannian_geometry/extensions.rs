//! Legacy tests for the riemannian_geometry module (migrated from flat file).

use super::*;
use scirs2_core::RngExt;

#[cfg(test)]
mod legacy_tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Build a random SPD matrix (seeded for reproducibility).
    fn random_spd(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let a: Vec<f64> = (0..n * n)
            .map(|_| rng.random_range(-1.0_f64..1.0_f64))
            .collect();
        // A^T A + ε I
        let at = mat_transpose(&a, n);
        let ata = mat_mul_nn(&at, &a, n);
        let mut spd = ata;
        for i in 0..n {
            spd[i * n + i] += 0.5;
        }
        spd
    }

    /// Build a random Stiefel point via Gram-Schmidt.
    fn random_stiefel(n: usize, p: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<f64> = (0..n * p)
            .map(|_| rng.random_range(-1.0_f64..1.0_f64))
            .collect();
        RgStiefelManifold::gram_schmidt(&raw, n, p)
    }

    /// Check that columns of an n×p matrix are orthonormal.
    fn check_orthonormal(q: &[f64], n: usize, p: usize) -> bool {
        for i in 0..p {
            for j in 0..p {
                let dot: f64 = (0..n).map(|k| q[k * p + i] * q[k * p + j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    // ── §3 SPD ─────────────────────────────────────────────────────────────

    #[test]
    fn spd_matrix_exp_zero_is_identity() {
        let n = 3;
        let zero = vec![0.0_f64; n * n];
        let result = matrix_exp_sym(&zero, n);
        let id = eye(n);
        for (r, e) in result.iter().zip(id.iter()) {
            assert!((r - e).abs() < 1e-8, "exp(0) != I");
        }
    }

    #[test]
    fn spd_mat_inv_product_is_identity() {
        let n = 3;
        let a = random_spd(n, 42);
        let inv = mat_inv_nn(&a, n).expect("inverse should exist");
        let prod = mat_mul_nn(&a, &inv, n);
        let id = eye(n);
        for (p, e) in prod.iter().zip(id.iter()) {
            assert!((p - e).abs() < 1e-7, "A*A^-1 not identity: {} vs {}", p, e);
        }
    }

    #[test]
    fn spd_mat_inv_2x2() {
        // [[2,1],[1,2]] -> inv = [[2/3, -1/3],[-1/3, 2/3]]
        let a = vec![2.0_f64, 1.0, 1.0, 2.0];
        let inv = mat_inv_nn(&a, 2).expect("should be invertible");
        assert!((inv[0] - 2.0 / 3.0).abs() < 1e-9);
        assert!((inv[1] + 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn spd_geodesic_distance_self_zero() {
        let man = RgSpdManifold::new(3);
        let p = random_spd(3, 10);
        let d = man.geodesic_distance(&p, &p);
        assert!(d < 1e-7, "dist(p, p) should be 0, got {}", d);
    }

    #[test]
    fn spd_exp_map_result_is_symmetric() {
        let man = RgSpdManifold::new(3);
        let p = random_spd(3, 7);
        let v = man.project_tangent(&p, &random_spd(3, 8));
        let q = man.exp_map(&p, &v);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (q[i * 3 + j] - q[j * 3 + i]).abs() < 1e-8,
                    "exp_map result not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn spd_exp_log_roundtrip() {
        let man = RgSpdManifold::new(2);
        let p = random_spd(2, 1);
        let q = random_spd(2, 2);
        let v = man.log_map(&p, &q);
        let q_reconstructed = man.exp_map(&p, &v);
        for (r, qi) in q_reconstructed.iter().zip(q.iter()) {
            assert!(
                (r - qi).abs() < 1e-4,
                "SPD roundtrip failed: {} vs {}",
                r,
                qi
            );
        }
    }

    #[test]
    fn spd_triangle_inequality() {
        let man = RgSpdManifold::new(2);
        let p = random_spd(2, 11);
        let q = random_spd(2, 12);
        let r = random_spd(2, 13);
        let d_pq = man.geodesic_distance(&p, &q);
        let d_qr = man.geodesic_distance(&q, &r);
        let d_pr = man.geodesic_distance(&p, &r);
        assert!(
            d_pr <= d_pq + d_qr + 1e-9,
            "Triangle inequality violated: {} > {} + {}",
            d_pr,
            d_pq,
            d_qr
        );
    }

    #[test]
    fn spd_project_tangent_is_symmetric() {
        let man = RgSpdManifold::new(3);
        let p = random_spd(3, 99);
        let mut rng = StdRng::seed_from_u64(100);
        let v_raw: Vec<f64> = (0..9)
            .map(|_| rng.random_range(-1.0_f64..1.0_f64))
            .collect();
        let v = man.project_tangent(&p, &v_raw);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (v[i * 3 + j] - v[j * 3 + i]).abs() < 1e-10,
                    "project_tangent not symmetric"
                );
            }
        }
    }

    #[test]
    fn spd_cholesky_basic() {
        // 2×2: [[4,2],[2,3]] → L = [[2,0],[1,√2]]
        let s = vec![4.0_f64, 2.0, 2.0, 3.0];
        let l = RgSpdManifold::cholesky(&s, 2).expect("should succeed");
        let lt = mat_transpose(&l, 2);
        let reconstructed = mat_mul_nn(&l, &lt, 2);
        for (r, orig) in reconstructed.iter().zip(s.iter()) {
            assert!((r - orig).abs() < 1e-9, "L*L^T != S");
        }
    }

    #[test]
    fn spd_cholesky_negative_fails() {
        let s = vec![-1.0_f64, 0.0, 0.0, 1.0];
        let result = RgSpdManifold::cholesky(&s, 2);
        assert!(result.is_none(), "Expected None for non-PD matrix");
    }

    #[test]
    fn spd_dim_correct() {
        assert_eq!(RgSpdManifold::new(3).dim(), 6);
        assert_eq!(RgSpdManifold::new(4).dim(), 10);
    }

    // ── §4 Stiefel ─────────────────────────────────────────────────────────

    #[test]
    fn stiefel_gram_schmidt_orthonormal() {
        let n = 5;
        let p = 3;
        let q = random_stiefel(n, p, 200);
        assert!(
            check_orthonormal(&q, n, p),
            "gram_schmidt should give orthonormal columns"
        );
    }

    #[test]
    fn stiefel_qr_retraction_orthonormal() {
        let n = 5;
        let p = 3;
        let x = random_stiefel(n, p, 201);
        let mut rng = StdRng::seed_from_u64(202);
        let v_raw: Vec<f64> = (0..n * p)
            .map(|_| rng.random_range(-0.1_f64..0.1_f64))
            .collect();
        let v = RgStiefelManifold::project_stiefel(&x, &v_raw, n, p);
        let q = RgStiefelManifold::qr_retraction(&x, &v, n, p);
        assert!(
            check_orthonormal(&q, n, p),
            "QR retraction should give orthonormal columns"
        );
    }

    #[test]
    fn stiefel_project_tangent_is_horizontal() {
        let n = 5;
        let p = 3;
        let x = random_stiefel(n, p, 203);
        let mut rng = StdRng::seed_from_u64(204);
        let z: Vec<f64> = (0..n * p)
            .map(|_| rng.random_range(-1.0_f64..1.0_f64))
            .collect();
        let proj = RgStiefelManifold::project_stiefel(&x, &z, n, p);
        let mut xtproj = vec![0.0_f64; p * p];
        for i in 0..p {
            for j in 0..p {
                xtproj[i * p + j] = (0..n).map(|k| x[k * p + i] * proj[k * p + j]).sum();
            }
        }
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (xtproj[i * p + j] + xtproj[j * p + i]).abs() < 1e-8,
                    "X^T*proj not skew-symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn stiefel_exp_map_on_manifold() {
        let man = RgStiefelManifold::new(6, 3);
        let x = random_stiefel(6, 3, 205);
        let mut rng = StdRng::seed_from_u64(206);
        let v_raw: Vec<f64> = (0..18)
            .map(|_| rng.random_range(-0.1_f64..0.1_f64))
            .collect();
        let v = man.project_tangent(&x, &v_raw);
        let q = man.exp_map(&x, &v);
        assert!(
            check_orthonormal(&q, 6, 3),
            "exp_map should stay on Stiefel manifold"
        );
    }

    #[test]
    fn stiefel_dim_correct() {
        assert_eq!(RgStiefelManifold::new(5, 3).dim(), 9);
        assert_eq!(RgStiefelManifold::new(4, 2).dim(), 5);
    }

    // ── §5 Grassmann ───────────────────────────────────────────────────────

    #[test]
    fn grassmann_principal_angles_self_zero() {
        let n = 5;
        let p = 2;
        let x = random_stiefel(n, p, 300);
        let angles = RgGrassmannManifold::principal_angles(&x, &x, n, p);
        for a in &angles {
            assert!(
                a.abs() < 1e-6,
                "angle should be ~0 for same subspace: {}",
                a
            );
        }
    }

    #[test]
    fn grassmann_principal_angles_in_range() {
        let n = 5;
        let p = 2;
        let x = random_stiefel(n, p, 301);
        let y = random_stiefel(n, p, 302);
        let angles = RgGrassmannManifold::principal_angles(&x, &y, n, p);
        for a in &angles {
            assert!(
                *a >= -1e-9 && *a <= std::f64::consts::FRAC_PI_2 + 1e-9,
                "principal angle {} out of [0, π/2]",
                a
            );
        }
    }

    #[test]
    fn grassmann_distance_self_zero() {
        let man = RgGrassmannManifold::new(5, 2);
        let x = random_stiefel(5, 2, 303);
        let d = man.geodesic_distance(&x, &x);
        assert!(d < 1e-6, "distance to self should be ~0, got {}", d);
    }

    #[test]
    fn grassmann_projection_matrix_idempotent() {
        let n = 5;
        let p = 2;
        let x = random_stiefel(n, p, 304);
        let proj = RgGrassmannManifold::projection_matrix(&x, n, p);
        let proj2 = mat_mul_nn(&proj, &proj, n);
        for (a, b) in proj.iter().zip(proj2.iter()) {
            assert!((a - b).abs() < 1e-8, "P^2 != P");
        }
    }

    #[test]
    fn grassmann_exp_on_manifold() {
        let man = RgGrassmannManifold::new(5, 2);
        let x = random_stiefel(5, 2, 305);
        let mut rng = StdRng::seed_from_u64(306);
        let v_raw: Vec<f64> = (0..10)
            .map(|_| rng.random_range(-0.1_f64..0.1_f64))
            .collect();
        let v = man.project_tangent(&x, &v_raw);
        let q = man.exp_map(&x, &v);
        assert!(
            check_orthonormal(&q, 5, 2),
            "exp_map should stay on Grassmann (orth. rep.)"
        );
    }

    #[test]
    fn grassmann_dim_correct() {
        assert_eq!(RgGrassmannManifold::new(5, 2).dim(), 6);
        assert_eq!(RgGrassmannManifold::new(6, 3).dim(), 9);
    }

    // ── §6 SO(3) ───────────────────────────────────────────────────────────

    #[test]
    fn so3_hat_vee_roundtrip() {
        let omega = vec![0.1_f64, 0.2, 0.3];
        let hat = RgSo3Manifold::hat(&omega);
        let recovered = RgSo3Manifold::vee(&hat);
        for (a, b) in omega.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-14, "vee(hat(ω)) != ω");
        }
    }

    #[test]
    fn so3_hat_is_skew_symmetric() {
        let omega = vec![1.0_f64, 2.0, 3.0];
        let hat = RgSo3Manifold::hat(&omega);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (hat[i * 3 + j] + hat[j * 3 + i]).abs() < 1e-14,
                    "hat not skew-symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn so3_rodrigues_zero_is_identity() {
        let r = RgSo3Manifold::rodrigues(&[0.0, 0.0, 0.0]);
        let id = eye(3);
        for (ri, ei) in r.iter().zip(id.iter()) {
            assert!((ri - ei).abs() < 1e-8, "rodrigues(0) != I");
        }
    }

    #[test]
    fn so3_rodrigues_det_one() {
        let omega = vec![0.3_f64, 0.4, 0.5];
        let r = RgSo3Manifold::rodrigues(&omega);
        let d = RgSo3Manifold::det3(&r);
        assert!((d - 1.0).abs() < 1e-9, "det(R) = {} != 1", d);
    }

    #[test]
    fn so3_rodrigues_orthogonal() {
        let omega = vec![0.1_f64, 0.2, 0.7];
        let r = RgSo3Manifold::rodrigues(&omega);
        let rt = RgSo3Manifold::transpose_3x3(&r);
        let rtr = RgSo3Manifold::mat_mul_3x3(&rt, &r);
        let id = eye(3);
        for (a, b) in rtr.iter().zip(id.iter()) {
            assert!((a - b).abs() < 1e-9, "R^T R != I");
        }
    }

    #[test]
    fn so3_exp_log_roundtrip() {
        let man = RgSo3Manifold::new();
        let r0 = eye(3);
        let omega = vec![0.1_f64, 0.2, 0.3];
        let r1 = man.exp_map(&r0, &omega);
        let omega_back = man.log_map(&r0, &r1);
        for (a, b) in omega.iter().zip(omega_back.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "SO(3) exp-log roundtrip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn so3_geodesic_distance_symmetric() {
        let man = RgSo3Manifold::new();
        let id = eye(3);
        let r = RgSo3Manifold::rodrigues(&[0.2_f64, 0.1, 0.3]);
        let d1 = man.geodesic_distance(&id, &r);
        let d2 = man.geodesic_distance(&r, &id);
        assert!(
            (d1 - d2).abs() < 1e-9,
            "geodesic distance not symmetric: {} vs {}",
            d1,
            d2
        );
    }

    #[test]
    fn so3_geodesic_distance_self_zero() {
        let man = RgSo3Manifold::new();
        let r = RgSo3Manifold::rodrigues(&[0.3_f64, 0.0, 0.1]);
        let d = man.geodesic_distance(&r, &r);
        assert!(d < 1e-8, "dist(R, R) should be 0, got {}", d);
    }

    #[test]
    fn so3_exp_stays_on_so3() {
        let man = RgSo3Manifold::new();
        let r0 = RgSo3Manifold::rodrigues(&[0.4_f64, 0.2, 0.1]);
        let v = vec![0.05_f64, 0.03, 0.07];
        let r1 = man.exp_map(&r0, &v);
        let d = RgSo3Manifold::det3(&r1);
        assert!(
            (d - 1.0).abs() < 1e-7,
            "exp_map result not in SO(3): det = {}",
            d
        );
    }

    #[test]
    fn so3_project_tangent_returns_r3() {
        let man = RgSo3Manifold::new();
        let r = eye(3);
        let v = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let proj = man.project_tangent(&r, &v);
        assert_eq!(proj.len(), 3, "project_tangent should return R^3 for SO(3)");
    }

    #[test]
    fn so3_dim_correct() {
        assert_eq!(RgSo3Manifold::new().dim(), 3);
    }

    // ── §7 Fréchet Mean ────────────────────────────────────────────────────

    #[test]
    fn frechet_mean_identical_spd_points() {
        let man = RgSpdManifold::new(2);
        let p = random_spd(2, 400);
        let points = vec![p.clone(), p.clone(), p.clone()];
        let fm = FrechetMean::new();
        let mean = fm.compute(&man, &points);
        for (a, b) in mean.iter().zip(p.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Fréchet mean of identical points != that point"
            );
        }
    }

    #[test]
    fn frechet_mean_single_point() {
        let man = RgSpdManifold::new(2);
        let p = random_spd(2, 401);
        let points = vec![p.clone()];
        let fm = FrechetMean::new();
        let mean = fm.compute(&man, &points);
        for (a, b) in mean.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn frechet_mean_weighted_uniform_eq_unweighted() {
        let man = RgSpdManifold::new(2);
        let p1 = random_spd(2, 402);
        let p2 = random_spd(2, 403);
        let p3 = random_spd(2, 404);
        let points = vec![p1, p2, p3];
        let fm = FrechetMean::new();
        let mean_unw = fm.compute(&man, &points);
        let weights = vec![1.0 / 3.0; 3];
        let mean_w = fm.compute_weighted(&man, &points, &weights);
        for (a, b) in mean_unw.iter().zip(mean_w.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "Weighted (uniform) != unweighted mean"
            );
        }
    }

    #[test]
    fn frechet_mean_so3_identical_points() {
        let man = RgSo3Manifold::new();
        let r = RgSo3Manifold::rodrigues(&[0.1_f64, 0.2, 0.3]);
        let points = vec![r.clone(), r.clone(), r.clone()];
        let fm = FrechetMean::new();
        let mean = fm.compute(&man, &points);
        let d = man.geodesic_distance(&mean, &r);
        assert!(
            d < 1e-5,
            "SO(3) Fréchet mean of identical points failed: d={}",
            d
        );
    }

    // ── §8 Riemannian Adam ─────────────────────────────────────────────────

    #[test]
    fn riemannian_adam_stiefel_stays_on_manifold() {
        let man = RgStiefelManifold::new(5, 3);
        let mut opt = RiemannianAdam::new(RiemannianAdamConfig::default());
        let mut x = random_stiefel(5, 3, 500);
        let mut rng = StdRng::seed_from_u64(501);
        for _ in 0..5 {
            let grad_raw: Vec<f64> = (0..15)
                .map(|_| rng.random_range(-0.1_f64..0.1_f64))
                .collect();
            let grad = man.project_tangent(&x, &grad_raw);
            x = opt.step(&man, &x, &grad);
        }
        assert!(
            check_orthonormal(&x, 5, 3),
            "Adam should keep point on Stiefel manifold"
        );
    }

    #[test]
    fn riemannian_adam_spd_result_symmetric() {
        let man = RgSpdManifold::new(2);
        let mut opt = RiemannianAdam::new(RiemannianAdamConfig::default());
        let mut p = random_spd(2, 502);
        let mut rng = StdRng::seed_from_u64(503);
        for _ in 0..5 {
            let g_raw: Vec<f64> = (0..4)
                .map(|_| rng.random_range(-0.1_f64..0.1_f64))
                .collect();
            let g = man.project_tangent(&p, &g_raw);
            p = opt.step(&man, &p, &g);
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (p[i * 2 + j] - p[j * 2 + i]).abs() < 1e-8,
                    "Adam SPD result not symmetric"
                );
            }
        }
    }

    #[test]
    fn riemannian_adam_moment_initialization() {
        let man = RgSo3Manifold::new();
        let mut opt = RiemannianAdam::new(RiemannianAdamConfig {
            lr: 0.01,
            ..Default::default()
        });
        let r = eye(3);
        let grad = vec![0.01_f64, 0.02, 0.03];
        let r_new = opt.step(&man, &r, &grad);
        assert!(
            opt.m.is_some(),
            "First moment should be initialized after step"
        );
        assert!(
            opt.v.is_some(),
            "Second moment should be initialized after step"
        );
        assert_eq!(opt.t, 1, "Step count should be 1");
        let d = RgSo3Manifold::det3(&r_new);
        assert!((d - 1.0).abs() < 1e-7, "Adam SO(3) result det = {}", d);
    }

    // ── §9 Riemannian SGD ──────────────────────────────────────────────────

    #[test]
    fn riemannian_sgd_stiefel_stays_on_manifold() {
        let man = RgStiefelManifold::new(4, 2);
        let mut sgd = RiemannianSgd::new(0.01, 0.9);
        let mut x = random_stiefel(4, 2, 600);
        let mut rng = StdRng::seed_from_u64(601);
        for _ in 0..5 {
            let grad_raw: Vec<f64> = (0..8)
                .map(|_| rng.random_range(-0.1_f64..0.1_f64))
                .collect();
            let grad = man.project_tangent(&x, &grad_raw);
            x = sgd.step(&man, &x, &grad);
        }
        assert!(
            check_orthonormal(&x, 4, 2),
            "SGD should keep point on Stiefel manifold"
        );
    }

    #[test]
    fn riemannian_sgd_so3_valid_rotation() {
        let man = RgSo3Manifold::new();
        let mut sgd = RiemannianSgd::new(0.01, 0.0);
        let mut r = eye(3);
        let mut rng = StdRng::seed_from_u64(602);
        for _ in 0..5 {
            let grad: Vec<f64> = (0..3)
                .map(|_| rng.random_range(-0.1_f64..0.1_f64))
                .collect();
            r = sgd.step(&man, &r, &grad);
        }
        let d = RgSo3Manifold::det3(&r);
        assert!((d - 1.0).abs() < 1e-7, "SGD SO(3) result det = {}", d);
    }

    #[test]
    fn riemannian_sgd_no_momentum_vanilla() {
        let man = RgStiefelManifold::new(3, 2);
        let mut sgd = RiemannianSgd::new(0.001, 0.0);
        let x = random_stiefel(3, 2, 603);
        let grad: Vec<f64> = vec![0.01_f64; 6];
        let x_new = sgd.step(&man, &x, &grad);
        assert!(
            check_orthonormal(&x_new, 3, 2),
            "SGD no-momentum should stay on manifold"
        );
    }

    // ── §10 Riemannian Batch Norm ──────────────────────────────────────────

    #[test]
    fn rbn_output_shape_matches_input() {
        let mut rbn = RiemannianBatchNorm::new(2);
        let inputs: Vec<Vec<f64>> = (0..4).map(|i| random_spd(2, 700 + i as u64)).collect();
        let outputs = rbn.forward(&inputs, true);
        assert_eq!(outputs.len(), inputs.len(), "Output batch size mismatch");
        for out in &outputs {
            assert_eq!(out.len(), 4, "Output SPD matrix size mismatch");
        }
    }

    #[test]
    fn rbn_output_symmetric() {
        let mut rbn = RiemannianBatchNorm::new(3);
        let inputs: Vec<Vec<f64>> = (0..3).map(|i| random_spd(3, 710 + i as u64)).collect();
        let outputs = rbn.forward(&inputs, true);
        for out in &outputs {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (out[i * 3 + j] - out[j * 3 + i]).abs() < 1e-7,
                        "RBN output not symmetric"
                    );
                }
            }
        }
    }

    #[test]
    fn rbn_running_mean_updated_after_training() {
        let mut rbn = RiemannianBatchNorm::new(2);
        assert!(
            rbn.running_mean.is_none(),
            "Running mean should be None initially"
        );
        let inputs: Vec<Vec<f64>> = (0..4).map(|i| random_spd(2, 720 + i as u64)).collect();
        rbn.forward(&inputs, true);
        assert!(
            rbn.running_mean.is_some(),
            "Running mean should be Some after training forward"
        );
    }

    #[test]
    fn rbn_inference_uses_running_mean() {
        let mut rbn = RiemannianBatchNorm::new(2);
        let inputs: Vec<Vec<f64>> = (0..4).map(|i| random_spd(2, 730 + i as u64)).collect();
        rbn.forward(&inputs, true);
        let outputs = rbn.forward(&inputs, false);
        assert_eq!(outputs.len(), inputs.len());
    }

    #[test]
    fn rbn_empty_input() {
        let mut rbn = RiemannianBatchNorm::new(2);
        let outputs = rbn.forward(&[], true);
        assert!(outputs.is_empty(), "Empty input should give empty output");
    }

    // ── Additional coverage ────────────────────────────────────────────────

    #[test]
    fn mat_mul_identity() {
        let n = 3;
        let id = eye(n);
        let a = random_spd(n, 800);
        let result = mat_mul_nn(&a, &id, n);
        for (r, orig) in result.iter().zip(a.iter()) {
            assert!((r - orig).abs() < 1e-14, "A * I != A");
        }
    }

    #[test]
    fn symmetrize_already_symmetric() {
        let a = vec![1.0_f64, 2.0, 2.0, 3.0];
        let s = symmetrize_nn(&a, 2);
        for (orig, sym) in a.iter().zip(s.iter()) {
            assert!(
                (orig - sym).abs() < 1e-14,
                "symmetrize changed symmetric matrix"
            );
        }
    }

    #[test]
    fn symmetrize_asymmetric() {
        let a = vec![1.0_f64, 3.0, 1.0, 4.0];
        let s = symmetrize_nn(&a, 2);
        assert!((s[1] - 2.0).abs() < 1e-14);
        assert!((s[2] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn jacobi_eigen_2x2() {
        let a = vec![3.0_f64, 1.0, 1.0, 3.0];
        let (vals, vecs) = jacobi_eigen(&a, 2, 100);
        let mut sorted_vals = vals.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (sorted_vals[0] - 2.0).abs() < 1e-9,
            "eigenvalue[0] should be 2: {}",
            sorted_vals[0]
        );
        assert!(
            (sorted_vals[1] - 4.0).abs() < 1e-9,
            "eigenvalue[1] should be 4: {}",
            sorted_vals[1]
        );
        let vt = mat_transpose(&vecs, 2);
        let vtav = mat_mul_nn(&vt, &mat_mul_nn(&a, &vecs, 2), 2);
        assert!(
            (vtav[1]).abs() < 1e-9,
            "Off-diagonal should be 0: {}",
            vtav[1]
        );
    }

    #[test]
    fn matrix_log_exp_3x3_roundtrip() {
        let n = 3;
        let s = random_spd(n, 900);
        let log_s = matrix_log_sym(&s, n).expect("log should exist for PD matrix");
        let exp_log_s = matrix_exp_sym(&log_s, n);
        for (a, b) in exp_log_s.iter().zip(s.iter()) {
            assert!((a - b).abs() < 1e-6, "exp(log(S)) != S");
        }
    }

    #[test]
    fn stiefel_geodesic_distance_positive() {
        let man = RgStiefelManifold::new(5, 2);
        let x = random_stiefel(5, 2, 950);
        let y = random_stiefel(5, 2, 951);
        let d = man.geodesic_distance(&x, &y);
        assert!(d >= 0.0, "geodesic_distance must be non-negative");
    }

    #[test]
    fn grassmann_distance_symmetric() {
        let man = RgGrassmannManifold::new(6, 2);
        let x = random_stiefel(6, 2, 960);
        let y = random_stiefel(6, 2, 961);
        let d1 = man.geodesic_distance(&x, &y);
        let d2 = man.geodesic_distance(&y, &x);
        assert!(
            (d1 - d2).abs() < 1e-9,
            "Grassmann distance not symmetric: {} vs {}",
            d1,
            d2
        );
    }

    #[test]
    fn spd_manifold_log_map_antisymmetry() {
        let man = RgSpdManifold::new(2);
        let p = random_spd(2, 970);
        let v = man.log_map(&p, &p);
        assert!(frob_norm(&v) < 1e-10, "log_P(P) should be zero vector");
    }

    #[test]
    fn so3_double_geodesic_distance() {
        let man = RgSo3Manifold::new();
        let r1 = RgSo3Manifold::rodrigues(&[0.5_f64, 0.1, 0.2]);
        let r2 = RgSo3Manifold::rodrigues(&[0.1_f64, 0.6, 0.0]);
        let d1 = man.geodesic_distance(&r1, &r2);
        let d2 = man.geodesic_distance(&r2, &r1);
        assert!(
            (d1 - d2).abs() < 1e-9,
            "SO(3) geodesic distance not symmetric"
        );
    }

    #[test]
    fn riemannian_adam_config_default() {
        let cfg = RiemannianAdamConfig::default();
        assert!((cfg.lr - 0.01).abs() < 1e-14);
        assert!((cfg.beta1 - 0.9).abs() < 1e-14);
        assert!((cfg.beta2 - 0.999).abs() < 1e-14);
        assert!((cfg.epsilon - 1e-8).abs() < 1e-14);
    }

    #[test]
    fn frechet_mean_empty_returns_empty() {
        let man = RgSpdManifold::new(2);
        let fm = FrechetMean::new();
        let result = fm.compute(&man, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn singular_values_identity() {
        let p = 3;
        let id = eye(p);
        let svs = singular_values_pp(&id, p);
        for sv in &svs {
            assert!((sv - 1.0).abs() < 1e-8, "SVD of I should give σ=1: {}", sv);
        }
    }

    #[test]
    fn grassmann_projection_orthogonal_complement() {
        let n = 5;
        let p = 2;
        let x = random_stiefel(n, p, 999);
        let proj = RgGrassmannManifold::projection_matrix(&x, n, p);
        let id = eye(n);
        let comp: Vec<f64> = id.iter().zip(proj.iter()).map(|(a, b)| a - b).collect();
        let comp2 = mat_mul_nn(&comp, &comp, n);
        for (a, b) in comp.iter().zip(comp2.iter()) {
            assert!((a - b).abs() < 1e-8, "(I-P)^2 != (I-P)");
        }
    }
}
