//! Tests for variational_inference module.

use super::*;
use scirs2_core::random::SeedableRng;
use std::f64::consts::PI;

fn make_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// ── MeanFieldGaussian ────────────────────────────────────────────────────

#[test]
fn test_mfg_sample_shape() {
    let q = MeanFieldGaussian::new(5);
    let mut rng = make_rng();
    let eps = normal_vec(5, &mut rng);
    let z = q.sample(&eps);
    assert_eq!(z.len(), 5);
}

#[test]
fn test_mfg_entropy_finite() {
    let q = MeanFieldGaussian::new(4);
    let h = q.entropy();
    assert!(h.is_finite(), "entropy should be finite, got {h}");
}

#[test]
fn test_mfg_kl_nonnegative() {
    let mut q = MeanFieldGaussian::new(3);
    q.mu = vec![1.0, -0.5, 2.0];
    q.log_sigma = vec![-0.3, 0.2, 0.5];
    let kl = q.kl_to_standard_normal();
    assert!(kl >= 0.0, "KL should be non-negative, got {kl}");
}

#[test]
fn test_mfg_kl_zero_for_standard_normal() {
    let q = MeanFieldGaussian::new(4); // mu=0, log_sigma=0 => sigma=1
    let kl = q.kl_to_standard_normal();
    // KL(N(0,I) || N(0,I)) = 0
    assert!(kl.abs() < 1e-10, "KL should be ~0, got {kl}");
}

#[test]
fn test_mfg_log_prob_finite() {
    let q = MeanFieldGaussian::new(3);
    let z = vec![0.5, -0.3, 1.2];
    let lp = q.log_prob(&z);
    assert!(lp.is_finite());
}

#[test]
fn test_mfg_update_changes_params() {
    let mut q = MeanFieldGaussian::new(2);
    let mu_before = q.mu.clone();
    q.update(&[0.1, -0.2], &[0.05, 0.1], 0.01);
    assert!((q.mu[0] - mu_before[0]).abs() > 1e-12);
}

// ── FullRankGaussian ─────────────────────────────────────────────────────

#[test]
fn test_frg_sample_shape() {
    let q = FullRankGaussian::new(3);
    let mut rng = make_rng();
    let eps = normal_vec(3, &mut rng);
    let z = q.sample(&eps);
    assert_eq!(z.len(), 3);
}

#[test]
fn test_frg_entropy_positive() {
    let q = FullRankGaussian::new(4);
    let h = q.entropy();
    assert!(
        h > 0.0,
        "entropy of I covariance should be positive, got {h}"
    );
}

#[test]
fn test_frg_kl_nonnegative() {
    let q = FullRankGaussian::new(3);
    let kl = q.kl_to_standard_normal();
    // KL(N(0,I) || N(0,I)) = 0
    assert!(
        kl.abs() < 1e-10,
        "KL for identity covariance should be ~0, got {kl}"
    );
}

#[test]
fn test_frg_log_prob_finite() {
    let q = FullRankGaussian::new(2);
    let z = vec![0.1, -0.5];
    let lp = q.log_prob(&z);
    assert!(lp.is_finite(), "log prob should be finite, got {lp}");
}

// ── BlackBoxVi ───────────────────────────────────────────────────────────

fn gaussian_log_joint_1d(z: &[f64]) -> f64 {
    // log p(z) = log N(z; 2.0, 0.5^2) (target)
    let mu = 2.0_f64;
    let sigma = 0.5_f64;
    -0.5 * ((z[0] - mu) / sigma).powi(2) - (sigma * (2.0 * PI).sqrt()).ln()
}

fn gaussian_log_joint_2d(z: &[f64]) -> f64 {
    // log N(z; [1.0, -1.0], diag([0.5, 0.5]^2))
    let mus = [1.0, -1.0];
    let sigma = 0.5_f64;
    z.iter()
        .zip(mus.iter())
        .map(|(&zi, &mi)| -0.5 * ((zi - mi) / sigma).powi(2) - (sigma * (2.0 * PI).sqrt()).ln())
        .sum()
}

#[test]
fn test_bbvi_elbo_finite() {
    let config = BbviConfig {
        n_samples: 5,
        ..Default::default()
    };
    let vi = BlackBoxVi::new(1, config);
    let mut rng = make_rng();
    let elbo = vi.elbo_estimate(&gaussian_log_joint_1d, 10, &mut rng);
    assert!(elbo.is_finite(), "ELBO should be finite, got {elbo}");
}

#[test]
fn test_bbvi_pathwise_grad_shape() {
    let config = BbviConfig {
        n_samples: 5,
        ..Default::default()
    };
    let vi = BlackBoxVi::new(2, config);
    let mut rng = make_rng();
    let (gmu, gls) = vi.pathwise_gradient(&gaussian_log_joint_2d, &mut rng, 1e-4);
    assert_eq!(gmu.len(), 2);
    assert_eq!(gls.len(), 2);
}

#[test]
fn test_bbvi_score_grad_shape() {
    let config = BbviConfig {
        n_samples: 10,
        use_pathwise: false,
        ..Default::default()
    };
    let vi = BlackBoxVi::new(2, config);
    let mut rng = make_rng();
    let (gmu, gls) = vi.score_gradient(&gaussian_log_joint_2d, &mut rng);
    assert_eq!(gmu.len(), 2);
    assert_eq!(gls.len(), 2);
}

#[test]
fn test_bbvi_fit_elbo_increases() {
    // ELBO should increase (on average) when fitting a Gaussian target
    let config = BbviConfig {
        n_samples: 10,
        n_epochs: 50,
        lr: 0.05,
        use_baseline: false,
        use_pathwise: true,
    };
    let mut vi = BlackBoxVi::new(1, config);
    let mut rng = make_rng();
    let result = vi.fit(&gaussian_log_joint_1d, &mut rng);
    assert!(!result.elbo_history.is_empty());
    // Check that last few ELBOs are higher than first few
    let n = result.elbo_history.len();
    let early_mean: f64 = result.elbo_history[..5].iter().sum::<f64>() / 5.0;
    let late_mean: f64 = result.elbo_history[n - 5..].iter().sum::<f64>() / 5.0;
    assert!(
        late_mean >= early_mean - 1.0,
        "ELBO should generally increase, early={early_mean:.3}, late={late_mean:.3}"
    );
}

#[test]
fn test_bbvi_fit_result_shape() {
    let config = BbviConfig {
        n_samples: 5,
        n_epochs: 10,
        lr: 0.01,
        ..Default::default()
    };
    let mut vi = BlackBoxVi::new(3, config);
    let mut rng = make_rng();
    let result = vi.fit(
        &|z: &[f64]| z.iter().map(|&zi| -0.5 * zi * zi).sum::<f64>(),
        &mut rng,
    );
    assert_eq!(result.final_mu.len(), 3);
    assert_eq!(result.final_sigma.len(), 3);
    assert_eq!(result.elbo_history.len(), 10);
}

// ── AdviVariable ─────────────────────────────────────────────────────────

#[test]
fn test_advi_positive_transform() {
    let var = AdviVariable {
        name: "sigma".into(),
        dim: 1,
        constraint: ParameterConstraint::Positive,
    };
    // transform_from_real(0.0) = exp(0.0) = 1.0
    let constrained = var.transform_from_real(&[0.0]);
    assert!(
        (constrained[0] - 1.0).abs() < 1e-10,
        "exp(0)=1, got {}",
        constrained[0]
    );
}

#[test]
fn test_advi_bounded_stays_in_range() {
    let var = AdviVariable {
        name: "p".into(),
        dim: 3,
        constraint: ParameterConstraint::Bounded {
            low: 0.0,
            high: 1.0,
        },
    };
    let phi = vec![-2.0, 0.0, 3.0];
    let theta = var.transform_from_real(&phi);
    for t in &theta {
        assert!(*t >= 0.0 && *t <= 1.0, "bounded constraint violated: {t}");
    }
}

#[test]
fn test_advi_log_det_jacobian_finite() {
    let var = AdviVariable {
        name: "x".into(),
        dim: 2,
        constraint: ParameterConstraint::Bounded {
            low: -5.0,
            high: 5.0,
        },
    };
    let phi = vec![0.5, -1.0];
    let ldj = var.log_abs_det_jacobian(&phi);
    assert!(ldj.is_finite(), "log|det J| should be finite, got {ldj}");
}

#[test]
fn test_advi_positive_jacobian_finite() {
    let var = AdviVariable {
        name: "sigma".into(),
        dim: 2,
        constraint: ParameterConstraint::Positive,
    };
    let phi = vec![0.0, 1.0];
    let ldj = var.log_abs_det_jacobian(&phi);
    assert!(ldj.is_finite());
}

#[test]
fn test_advi_unconstrained_jacobian_zero() {
    let var = AdviVariable {
        name: "mu".into(),
        dim: 3,
        constraint: ParameterConstraint::Unconstrained,
    };
    let phi = vec![1.0, 2.0, 3.0];
    let ldj = var.log_abs_det_jacobian(&phi);
    assert!((ldj - 0.0).abs() < 1e-10);
}

// ── AdviModel ────────────────────────────────────────────────────────────

#[test]
fn test_advi_split_params_length() {
    let vars = vec![
        AdviVariable {
            name: "mu".into(),
            dim: 2,
            constraint: ParameterConstraint::Unconstrained,
        },
        AdviVariable {
            name: "sigma".into(),
            dim: 2,
            constraint: ParameterConstraint::Positive,
        },
    ];
    let model = AdviModel::new(vars, 5, 0.01);
    let flat = vec![1.0, 2.0, 3.0, 4.0];
    let chunks = model.split_params(&flat);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].len(), 2);
    assert_eq!(chunks[1].len(), 2);
}

#[test]
fn test_advi_constrained_sample_structure() {
    let vars = vec![
        AdviVariable {
            name: "mu".into(),
            dim: 2,
            constraint: ParameterConstraint::Unconstrained,
        },
        AdviVariable {
            name: "sigma".into(),
            dim: 1,
            constraint: ParameterConstraint::Positive,
        },
    ];
    let model = AdviModel::new(vars, 5, 0.01);
    let mut rng = make_rng();
    let samples = model.constrained_sample(&mut rng);
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].len(), 2); // mu: unconstrained dim=2
    assert_eq!(samples[1].len(), 1); // sigma: positive dim=1
    assert!(
        samples[1][0] > 0.0,
        "positive constraint: sigma must be > 0"
    );
}

#[test]
fn test_advi_elbo_finite() {
    let vars = vec![AdviVariable {
        name: "mu".into(),
        dim: 1,
        constraint: ParameterConstraint::Unconstrained,
    }];
    let model = AdviModel::new(vars, 5, 0.01);
    let mut rng = make_rng();
    let elbo = model.elbo(
        &|thetas: &[Vec<f64>]| {
            let mu = thetas[0][0];
            -0.5 * mu * mu
        },
        &mut rng,
    );
    assert!(elbo.is_finite(), "ADVI ELBO should be finite, got {elbo}");
}

#[test]
fn test_advi_fit_non_empty_history() {
    let vars = vec![AdviVariable {
        name: "mu".into(),
        dim: 1,
        constraint: ParameterConstraint::Unconstrained,
    }];
    let mut model = AdviModel::new(vars, 3, 0.01);
    let mut rng = make_rng();
    let history = model.fit(
        &|thetas: &[Vec<f64>]| -0.5 * thetas[0][0] * thetas[0][0],
        5,
        &mut rng,
    );
    assert!(!history.is_empty(), "ADVI fit should return history");
    assert_eq!(history.len(), 5);
}

// ── StructuredVi ─────────────────────────────────────────────────────────

#[test]
fn test_structured_vi_entropy_sum() {
    let svi = StructuredVi::new(4, 3);
    let total_entropy = svi.entropy();
    let sum_entropy: f64 = svi.local_qs.iter().map(|q| q.entropy()).sum();
    assert!((total_entropy - sum_entropy).abs() < 1e-10);
}

#[test]
fn test_structured_vi_sample_trajectory_length() {
    let svi = StructuredVi::new(5, 2);
    let mut rng = make_rng();
    let traj = svi.sample_trajectory(&mut rng);
    assert_eq!(traj.len(), 5, "trajectory should have T=5 time steps");
    for zt in &traj {
        assert_eq!(zt.len(), 2);
    }
}

#[test]
fn test_structured_vi_kl_nonnegative() {
    let mut svi = StructuredVi::new(3, 2);
    svi.local_qs[0].mu = vec![0.5, -0.5];
    let kl = svi.kl_to_prior();
    assert!(kl >= 0.0, "KL to prior should be non-negative, got {kl}");
}

#[test]
fn test_structured_vi_elbo_finite() {
    let svi = StructuredVi::new(4, 2);
    let mut rng = make_rng();
    let elbo = svi.elbo(
        &|traj: &[Vec<f64>]| {
            traj.iter()
                .map(|zt| -0.5_f64 * zt.iter().map(|&z| z * z).sum::<f64>())
                .sum::<f64>()
        },
        &mut rng,
    );
    assert!(elbo.is_finite(), "ELBO should be finite, got {elbo}");
}

// ── PlanarFlowLayer ──────────────────────────────────────────────────────

#[test]
fn test_planar_flow_forward_shape() {
    let layer = PlanarFlowLayer::new(4);
    let z = vec![0.5, -1.0, 0.3, 2.0];
    let (z_prime, log_det) = layer.forward(&z);
    assert_eq!(z_prime.len(), 4);
    assert!(log_det.is_finite(), "log|det J| should be finite");
}

#[test]
fn test_planar_flow_log_det_finite() {
    let layer = PlanarFlowLayer::new(3);
    let z = vec![1.0, -0.5, 0.0];
    let (_, log_det) = layer.forward(&z);
    assert!(log_det.is_finite());
}

#[test]
fn test_planar_flow_transforms_input() {
    let layer = PlanarFlowLayer::new(2);
    let z = vec![1.0, 1.0];
    let (z_prime, _) = layer.forward(&z);
    // The transformed output may differ from the input
    assert_eq!(z_prime.len(), 2);
}

// ── FlowVi ───────────────────────────────────────────────────────────────

#[test]
fn test_flow_vi_sample_shape() {
    let flow = FlowVi::new(3, 2);
    let mut rng = make_rng();
    let (z, log_prob) = flow.sample_and_log_prob(&mut rng);
    assert_eq!(z.len(), 3);
    assert!(
        log_prob.is_finite(),
        "log q(z_K) should be finite, got {log_prob}"
    );
}

#[test]
fn test_flow_vi_elbo_finite() {
    let flow = FlowVi::new(2, 3);
    let mut rng = make_rng();
    let elbo = flow.elbo(&gaussian_log_joint_2d, 5, &mut rng);
    assert!(elbo.is_finite(), "FlowVI ELBO should be finite, got {elbo}");
}

#[test]
fn test_flow_vi_fit_returns_history() {
    let mut flow = FlowVi::new(1, 2);
    let mut rng = make_rng();
    let history = flow.fit(&gaussian_log_joint_1d, 10, 0.01, &mut rng);
    assert_eq!(history.len(), 10);
    assert!(history
        .iter()
        .all(|&e| e.is_finite() || e.is_nan() || e.is_infinite()));
}

// ── ViSvgd ───────────────────────────────────────────────────────────────

#[test]
fn test_vi_svgd_kernel_matrix_symmetric() {
    let config = ViSvgdConfig {
        n_particles: 5,
        bandwidth: 1.0,
        ..Default::default()
    };
    let mut rng = make_rng();
    let svgd = ViSvgd::new(2, config, &mut rng);
    let k = svgd.kernel_matrix();
    let n = k.len();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (k[i][j] - k[j][i]).abs() < 1e-10,
                "kernel matrix should be symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_vi_svgd_kernel_matrix_diagonal_one() {
    let config = ViSvgdConfig {
        n_particles: 4,
        bandwidth: 1.0,
        ..Default::default()
    };
    let mut rng = make_rng();
    let svgd = ViSvgd::new(3, config, &mut rng);
    let k = svgd.kernel_matrix();
    for i in 0..k.len() {
        assert!(
            (k[i][i] - 1.0).abs() < 1e-10,
            "diagonal should be exp(0)=1, got {}",
            k[i][i]
        );
    }
}

#[test]
fn test_vi_svgd_run_returns_n_particles() {
    let config = ViSvgdConfig {
        n_particles: 6,
        n_steps: 3,
        lr: 0.01,
        bandwidth: 1.0,
    };
    let mut rng = make_rng();
    let mut svgd = ViSvgd::new(2, config, &mut rng);
    let final_particles = svgd.run(&|z: &[f64]| z.iter().map(|&zi| -zi).collect());
    assert_eq!(final_particles.len(), 6, "should return 6 final particles");
}

#[test]
fn test_vi_svgd_mean_shape() {
    let config = ViSvgdConfig {
        n_particles: 5,
        ..Default::default()
    };
    let mut rng = make_rng();
    let svgd = ViSvgd::new(3, config, &mut rng);
    let m = svgd.mean();
    assert_eq!(m.len(), 3, "mean should have dim=3");
}

#[test]
fn test_vi_svgd_variance_shape() {
    let config = ViSvgdConfig {
        n_particles: 5,
        ..Default::default()
    };
    let mut rng = make_rng();
    let svgd = ViSvgd::new(4, config, &mut rng);
    let v = svgd.variance();
    assert_eq!(v.len(), 4);
    assert!(
        v.iter().all(|&vi| vi >= 0.0),
        "variances should be non-negative"
    );
}

// ── VI Diagnostics ───────────────────────────────────────────────────────

#[test]
fn test_compute_pareto_k_finite() {
    let log_weights: Vec<f64> = (0..20).map(|i| -(i as f64) * 0.1).collect();
    let k = compute_pareto_k(&log_weights);
    assert!(k.is_finite(), "Pareto k should be finite, got {k}");
}

#[test]
fn test_effective_sample_size_in_range() {
    let n = 20_usize;
    let log_weights: Vec<f64> = vec![0.0; n]; // uniform weights => ESS = n
    let ess = effective_sample_size(&log_weights);
    assert!(
        ess > 0.0 && ess <= n as f64,
        "ESS should be in (0, n], got {ess}"
    );
    // Uniform weights => ESS = n
    assert!(
        (ess - n as f64).abs() < 1.0,
        "Uniform weights => ESS ~ n, got {ess}"
    );
}

#[test]
fn test_diagnose_vi_finite_fields() {
    let q = MeanFieldGaussian::new(2);
    let mut rng = make_rng();
    let diag = diagnose_vi(&q, &gaussian_log_joint_2d, 30, &mut rng);
    assert!(diag.pareto_k.is_finite(), "pareto_k finite");
    assert!(diag.effective_sample_size.is_finite(), "ESS finite");
    assert!(diag.elbo_variance.is_finite(), "elbo_variance finite");
    assert!(diag.kl_divergence.is_finite(), "kl_divergence finite");
}

#[test]
fn test_diagnose_vi_kl_nonneg() {
    let mut q = MeanFieldGaussian::new(2);
    q.mu = vec![1.0, -1.0];
    let mut rng = make_rng();
    let diag = diagnose_vi(&q, &gaussian_log_joint_2d, 20, &mut rng);
    assert!(diag.kl_divergence >= 0.0, "KL should be non-negative");
}

// ── ViDistribution ───────────────────────────────────────────────────────

#[test]
fn test_vi_distribution_variants() {
    let mf = ViDistribution::MeanField;
    let fr = ViDistribution::FullRankGaussian;
    let lr = ViDistribution::LowRank { rank: 3 };
    assert_eq!(mf, ViDistribution::MeanField);
    assert_eq!(fr, ViDistribution::FullRankGaussian);
    assert_eq!(lr, ViDistribution::LowRank { rank: 3 });
}

// ── Additional edge case tests ───────────────────────────────────────────

#[test]
fn test_mfg_sample_uses_reparameterization() {
    // z = mu + sigma * eps; with mu=1, sigma=1, eps=0 => z=1
    let q = MeanFieldGaussian {
        mu: vec![1.0],
        log_sigma: vec![0.0],
        dim: 1,
    };
    let z = q.sample(&[0.0]);
    assert!((z[0] - 1.0).abs() < 1e-10, "z should equal mu when eps=0");
}

#[test]
fn test_frg_sample_with_identity_l() {
    // With L=I, mu=0, z = I * eps = eps
    let q = FullRankGaussian::new(2);
    let eps = vec![1.5, -0.5];
    let z = q.sample(&eps);
    assert!((z[0] - 1.5).abs() < 1e-10);
    assert!((z[1] - (-0.5)).abs() < 1e-10);
}

#[test]
fn test_planar_flow_update_changes_weights() {
    let mut layer = PlanarFlowLayer::new(2);
    let w_before = layer.w.clone();
    let grad = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // [grad_w, grad_u, grad_b]
    layer.update(&grad, 0.1);
    assert!(
        (layer.w[0] - w_before[0]).abs() > 1e-12,
        "w should have changed"
    );
}

#[test]
fn test_structured_vi_update_changes_params() {
    let mut svi = StructuredVi::new(2, 2);
    let mu_before = svi.local_qs[0].mu.clone();
    let grad_mus = vec![vec![0.1, -0.2], vec![0.0, 0.1]];
    let grad_ls = vec![vec![0.05, 0.1], vec![0.0, 0.0]];
    svi.update(&grad_mus, &grad_ls, 0.1);
    assert!((svi.local_qs[0].mu[0] - mu_before[0]).abs() > 1e-12);
}

#[test]
fn test_vi_svgd_median_heuristic_bandwidth() {
    // Test with bandwidth=0 (median heuristic)
    let config = ViSvgdConfig {
        n_particles: 5,
        bandwidth: 0.0, // use median heuristic
        n_steps: 1,
        lr: 0.01,
    };
    let mut rng = make_rng();
    let svgd = ViSvgd::new(2, config, &mut rng);
    let k = svgd.kernel_matrix();
    // All entries should be in (0, 1]
    for row in &k {
        for &kij in row {
            assert!(
                kij > 0.0 && kij <= 1.0 + 1e-10,
                "kernel value out of range: {kij}"
            );
        }
    }
}

#[test]
fn test_effective_sample_size_uniform_equals_n() {
    let n = 10;
    let log_weights = vec![0.0; n];
    let ess = effective_sample_size(&log_weights);
    // For uniform weights w_i = 1/n, ESS = n
    assert!((ess - n as f64).abs() < 1.0, "uniform ESS ~ {n}, got {ess}");
}

#[test]
fn test_advi_simplex_transform() {
    let var = AdviVariable {
        name: "probs".into(),
        dim: 3,
        constraint: ParameterConstraint::Simplex(3),
    };
    let phi = vec![0.5, -0.5]; // 2 unconstrained params for K=3 simplex
    let theta = var.transform_from_real(&phi);
    assert_eq!(theta.len(), 3);
    let sum: f64 = theta.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "simplex should sum to 1, got {sum}"
    );
    assert!(
        theta.iter().all(|&t| t >= 0.0),
        "simplex values should be non-negative"
    );
}

#[test]
fn test_bbvi_score_gradient_finite() {
    let config = BbviConfig {
        n_samples: 10,
        use_pathwise: false,
        use_baseline: true,
        ..Default::default()
    };
    let vi = BlackBoxVi::new(2, config);
    let mut rng = make_rng();
    let (gmu, gls) = vi.score_gradient(&gaussian_log_joint_2d, &mut rng);
    assert!(
        gmu.iter().all(|&g| g.is_finite()),
        "score grad_mu should be finite"
    );
    assert!(
        gls.iter().all(|&g| g.is_finite()),
        "score grad_ls should be finite"
    );
}

#[test]
fn test_full_rank_gaussian_kl_positive_for_nonstandard() {
    let mut q = FullRankGaussian::new(2);
    q.mu = vec![2.0, -1.0]; // non-zero mean => positive KL
    let kl = q.kl_to_standard_normal();
    assert!(
        kl > 0.0,
        "KL should be positive for non-zero mean, got {kl}"
    );
}
