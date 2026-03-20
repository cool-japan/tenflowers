// Tests for the Bayesian Optimization + CMA-ES module.
// Split into a separate file to satisfy the < 2000 line per-file policy.

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── helpers ──────────────────────────────────────────────────────────────────

fn simple_1d_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let xs: Vec<Vec<f64>> = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    let ys: Vec<f64> = vec![4.0, 1.0, 0.0, 1.0, 4.0]; // y = x²
    (xs, ys)
}

fn default_gp() -> GaussianProcess {
    GaussianProcess::new(GpConfig::default())
}

// ── GP tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_gp_fit_and_predict_basic() {
    let (xs, ys) = simple_1d_data();
    let mut gp = default_gp();
    gp.fit(&xs, &ys).expect("fit should succeed");
    assert!(gp.fitted);

    let preds = gp.predict(&[vec![0.0]]).expect("predict should succeed");
    let (mean, std) = preds[0];
    assert!((mean - 0.0).abs() < 1.0, "mean near 0: got {mean}");
    assert!(std >= 0.0, "std should be non-negative");
}

#[test]
fn test_gp_predict_near_training_points_low_variance() {
    let (xs, ys) = simple_1d_data();
    let mut gp = GaussianProcess::new(GpConfig {
        noise_std: 1e-4,
        ..GpConfig::default()
    });
    gp.fit(&xs, &ys).expect("fit");
    let preds = gp.predict(&xs).expect("predict on training set");
    for (i, (_, std)) in preds.iter().enumerate() {
        assert!(
            *std < 0.2,
            "std at training point {} should be small, got {std}",
            i
        );
    }
}

#[test]
fn test_gp_kernel_matrix_is_symmetric() {
    let gp = default_gp();
    let xs = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![-1.0, -1.0]];
    let k = gp.kernel_matrix(&xs, &xs);
    let n = k.len();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (k[i][j] - k[j][i]).abs() < 1e-12,
                "K[{i}][{j}] != K[{j}][{i}]"
            );
        }
    }
}

#[test]
fn test_gp_kernel_matrix_positive_diagonal() {
    let gp = default_gp();
    let xs = vec![vec![0.0], vec![1.0], vec![2.0]];
    let k = gp.kernel_matrix(&xs, &xs);
    for (i, row) in k.iter().enumerate() {
        assert!(row[i] > 0.0, "diagonal K[{i}][{i}] must be positive");
    }
}

#[test]
fn test_gp_cholesky_reconstruction() {
    let gp = default_gp();
    let xs = vec![vec![0.0], vec![1.0], vec![2.0]];
    let mut k = gp.kernel_matrix(&xs, &xs);
    let n = k.len();
    let noise = 1e-2;
    for i in 0..n {
        k[i][i] += noise * noise;
    }
    let l = gp.cholesky(&k).expect("cholesky");
    for i in 0..n {
        for j in 0..n {
            let mut llt_ij = 0.0_f64;
            for kk in 0..n {
                llt_ij += l[i][kk] * l[j][kk];
            }
            assert!(
                (llt_ij - k[i][j]).abs() < 1e-9,
                "L@Lᵀ[{i}][{j}] = {llt_ij}, K[{i}][{j}] = {}",
                k[i][j]
            );
        }
    }
}

#[test]
fn test_gp_predict_variance_near_noise_on_training_points() {
    let xs = vec![vec![0.0], vec![1.0]];
    let ys = vec![1.0, 2.0];
    let noise_std = 0.1;
    let mut gp = GaussianProcess::new(GpConfig {
        noise_std,
        ..GpConfig::default()
    });
    gp.fit(&xs, &ys).expect("fit");
    let preds = gp.predict(&xs).expect("predict");
    for (_, std) in &preds {
        assert!(
            *std < 0.5,
            "variance at training point should be small, std={std}"
        );
    }
}

#[test]
fn test_gp_rbf_kernel_values() {
    let gp = default_gp();
    let same = gp.kernel(&[0.0, 0.0], &[0.0, 0.0]);
    assert!(
        (same - 1.0).abs() < 1e-10,
        "k(x,x) should be σ²=1.0, got {same}"
    );

    let far = gp.kernel(&[0.0], &[100.0]);
    assert!(
        far < 1e-6,
        "k(x,x') should be near 0 for far points, got {far}"
    );
}

#[test]
fn test_gp_matern52_kernel_values() {
    let gp = GaussianProcess::new(GpConfig {
        kernel: KernelType::Matern52 {
            length_scale: 1.0,
            signal_std: 1.0,
        },
        ..GpConfig::default()
    });
    let same = gp.kernel(&[0.0], &[0.0]);
    assert!(
        (same - 1.0).abs() < 1e-10,
        "Matérn k(x,x) should be 1, got {same}"
    );

    let near = gp.kernel(&[0.0], &[0.01]);
    assert!(near > 0.9, "near points should have k~1, got {near}");

    let far = gp.kernel(&[0.0], &[100.0]);
    assert!(far < 1e-3, "far points should have k~0, got {far}");
}

#[test]
fn test_gp_periodic_kernel_values() {
    let gp = GaussianProcess::new(GpConfig {
        kernel: KernelType::Periodic {
            length_scale: 1.0,
            period: 2.0 * std::f64::consts::PI,
            signal_std: 1.0,
        },
        ..GpConfig::default()
    });
    let k_same = gp.kernel(&[0.0], &[0.0]);
    assert!((k_same - 1.0).abs() < 1e-10);
    let k_period = gp.kernel(&[0.0], &[2.0 * std::f64::consts::PI]);
    assert!(
        k_period > 0.5,
        "periodic kernel should return high value at period distance"
    );
}

#[test]
fn test_gp_log_marginal_likelihood_is_finite() {
    let (xs, ys) = simple_1d_data();
    let mut gp = default_gp();
    gp.fit(&xs, &ys).expect("fit");
    let lml = gp.log_marginal_likelihood().expect("lml");
    assert!(
        lml.is_finite(),
        "log marginal likelihood should be finite, got {lml}"
    );
}

#[test]
fn test_gp_fit_rejects_empty_data() {
    let mut gp = default_gp();
    let result = gp.fit(&[], &[]);
    assert!(result.is_err(), "should reject empty data");
}

#[test]
fn test_gp_predict_unfitted_returns_error() {
    let gp = default_gp();
    let result = gp.predict(&[vec![0.0]]);
    assert!(result.is_err(), "predict on unfitted GP should fail");
}

#[test]
fn test_gp_2d_fit_predict() {
    let xs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let ys = vec![0.0, 1.0, 1.0, 2.0];
    let mut gp = default_gp();
    gp.fit(&xs, &ys).expect("fit");
    let preds = gp.predict(&[vec![0.5, 0.5]]).expect("predict");
    let (mean, std) = preds[0];
    assert!(mean.is_finite(), "mean should be finite");
    assert!(std >= 0.0, "std should be non-negative");
}

#[test]
fn test_gp_fit_shape_mismatch_error() {
    let mut gp = default_gp();
    let xs = vec![vec![0.0], vec![1.0]];
    let ys = vec![1.0, 2.0, 3.0];
    let result = gp.fit(&xs, &ys);
    assert!(result.is_err(), "should fail on shape mismatch");
}

#[test]
fn test_gp_predict_extrapolation_higher_variance() {
    let (xs, ys) = simple_1d_data();
    let mut gp = default_gp();
    gp.fit(&xs, &ys).expect("fit");
    let training_preds = gp.predict(&[vec![0.0]]).expect("predict at 0");
    let far_preds = gp.predict(&[vec![10.0]]).expect("predict far");
    let std_training = training_preds[0].1;
    let std_far = far_preds[0].1;
    assert!(
        std_far >= std_training - 0.5,
        "extrapolated point should have higher or equal std: training={std_training}, far={std_far}"
    );
}

// ── Acquisition function tests ────────────────────────────────────────────────

#[test]
fn test_ucb_value() {
    let mut rng = StdRng::seed_from_u64(0);
    let acq = AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 };
    let val = acq.evaluate(3.0, 1.5, 0.0, &mut rng);
    let expected = 3.0 + 2.0 * 1.5;
    assert!(
        (val - expected).abs() < 1e-10,
        "UCB = μ + κσ, expected {expected}, got {val}"
    );
}

#[test]
fn test_ucb_zero_std() {
    let mut rng = StdRng::seed_from_u64(0);
    let acq = AcquisitionFunction::UpperConfidenceBound { kappa: 3.0 };
    let val = acq.evaluate(2.5, 0.0, 0.0, &mut rng);
    assert!(
        (val - 2.5).abs() < 1e-10,
        "UCB with std=0 should return mean"
    );
}

#[test]
fn test_ei_positive_for_improvement() {
    let mut rng = StdRng::seed_from_u64(0);
    let acq = AcquisitionFunction::ExpectedImprovement { xi: 0.0 };
    let val = acq.evaluate(2.0, 1.0, 1.0, &mut rng);
    assert!(
        val > 0.0,
        "EI should be positive when mean > best_y, got {val}"
    );
}

#[test]
fn test_ei_zero_when_std_zero_and_no_improvement() {
    let mut rng = StdRng::seed_from_u64(0);
    let acq = AcquisitionFunction::ExpectedImprovement { xi: 0.0 };
    let val = acq.evaluate(0.5, 0.0, 1.0, &mut rng);
    assert_eq!(val, 0.0, "EI should be 0 when std=0 and mean < best_y");
}

#[test]
fn test_pi_between_zero_and_one() {
    let mut rng = StdRng::seed_from_u64(0);
    let acq = AcquisitionFunction::ProbabilityOfImprovement { xi: 0.01 };
    for (mean, std, best) in [(1.5, 0.5, 1.0), (0.0, 1.0, 2.0), (5.0, 2.0, 3.0)] {
        let val = acq.evaluate(mean, std, best, &mut rng);
        assert!(
            (0.0..=1.0).contains(&val),
            "PI={val} not in [0,1] for mean={mean}"
        );
    }
}

#[test]
fn test_normal_cdf_known_values() {
    let phi0 = AcquisitionFunction::normal_cdf(0.0);
    assert!((phi0 - 0.5).abs() < 1e-4, "Φ(0)≈0.5, got {phi0}");

    let phi196 = AcquisitionFunction::normal_cdf(1.96);
    assert!((phi196 - 0.975).abs() < 1e-2, "Φ(1.96)≈0.975, got {phi196}");

    let phi_neg = AcquisitionFunction::normal_cdf(-1.96);
    assert!(
        (phi_neg - 0.025).abs() < 1e-2,
        "Φ(-1.96)≈0.025, got {phi_neg}"
    );
}

#[test]
fn test_normal_cdf_monotone() {
    let zs = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let cdfs: Vec<f64> = zs
        .iter()
        .map(|&z| AcquisitionFunction::normal_cdf(z))
        .collect();
    for w in cdfs.windows(2) {
        assert!(w[1] >= w[0], "CDF should be non-decreasing");
    }
}

#[test]
fn test_thompson_sampling_std_zero() {
    let mut rng = StdRng::seed_from_u64(42);
    let acq = AcquisitionFunction::ThompsonSampling;
    let test_val = std::f64::consts::PI;
    let val = acq.evaluate(test_val, 0.0, 0.0, &mut rng);
    assert!(
        (val - test_val).abs() < 1e-6,
        "TS with std=0 returns mean, got {val}"
    );
}

// ── BayesSearchSpace tests ────────────────────────────────────────────────────

#[test]
fn test_search_space_random_sample_within_bounds() {
    let space = BayesSearchSpace::new(vec![(-1.0, 1.0), (0.0, 10.0)]).expect("space");
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..100 {
        let x = space.random_sample(&mut rng);
        assert!(x[0] >= -1.0 && x[0] <= 1.0);
        assert!(x[1] >= 0.0 && x[1] <= 10.0);
    }
}

#[test]
fn test_search_space_lhs_returns_n_points() {
    let space = BayesSearchSpace::new(vec![(-1.0, 1.0), (0.0, 5.0)]).expect("space");
    let mut rng = StdRng::seed_from_u64(7);
    let pts = space.latin_hypercube_sample(20, &mut rng);
    assert_eq!(pts.len(), 20);
    for p in &pts {
        assert!(p[0] >= -1.0 && p[0] <= 1.0);
        assert!(p[1] >= 0.0 && p[1] <= 5.0);
    }
}

#[test]
fn test_search_space_invalid_bounds_error() {
    let result = BayesSearchSpace::new(vec![(5.0, 1.0)]);
    assert!(result.is_err(), "should reject lo >= hi");
}

#[test]
fn test_search_space_clamp() {
    let space = BayesSearchSpace::new(vec![(0.0, 1.0)]).expect("space");
    let clamped = space.clamp(&[-0.5]);
    assert!((clamped[0] - 0.0).abs() < 1e-10);
    let clamped2 = space.clamp(&[1.5]);
    assert!((clamped2[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_search_space_lhs_stratification() {
    let space = BayesSearchSpace::new(vec![(0.0, 1.0)]).expect("space");
    let mut rng = StdRng::seed_from_u64(100);
    let n = 10;
    let pts = space.latin_hypercube_sample(n, &mut rng);
    let mut strata: Vec<usize> = pts
        .iter()
        .map(|p| (p[0] * n as f64).min(n as f64 - 1.0) as usize)
        .collect();
    strata.sort();
    let expected: Vec<usize> = (0..n).collect();
    assert_eq!(strata, expected, "LHS should cover all strata");
}

// ── BayesianOptimizer tests ───────────────────────────────────────────────────

#[test]
fn test_bayesian_optimizer_1d_quadratic_minimize() {
    let space = BayesSearchSpace::new(vec![(-5.0, 5.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 5,
        n_iterations: 10,
        acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
        gp_config: GpConfig::default(),
        n_candidates: 500,
        seed: 0,
    };
    let mut opt = BayesianOptimizer::new(config, space, true);
    let result = opt.optimize(|x| x[0] * x[0]).expect("optimize");
    assert!(
        result.best_y < 2.0,
        "should find near-minimum, got {}",
        result.best_y
    );
    assert_eq!(result.n_evaluations, 15);
}

#[test]
fn test_bayesian_optimizer_best_y_improves() {
    let space = BayesSearchSpace::new(vec![(-5.0, 5.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 3,
        n_iterations: 8,
        acquisition: AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
        gp_config: GpConfig::default(),
        n_candidates: 200,
        seed: 1,
    };
    let mut opt = BayesianOptimizer::new(config, space, true);
    let result = opt.optimize(|x| x[0].powi(2) + 0.1).expect("optimize");
    assert!(
        result.best_y <= 25.1,
        "best_y should be reasonable, got {}",
        result.best_y
    );
    assert!(!result.observations.is_empty());
}

#[test]
fn test_bayesian_optimizer_suggest_in_bounds() {
    let space = BayesSearchSpace::new(vec![(0.0, 1.0), (0.0, 1.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 3,
        n_iterations: 5,
        acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
        gp_config: GpConfig::default(),
        n_candidates: 100,
        seed: 2,
    };
    let mut opt = BayesianOptimizer::new(config, space.clone(), true);
    opt.register(vec![0.2, 0.3], 0.5).expect("register");
    opt.register(vec![0.8, 0.7], 1.2).expect("register");
    opt.register(vec![0.5, 0.5], 0.3).expect("register");
    let next = opt.suggest().expect("suggest");
    assert_eq!(next.len(), 2);
    assert!(space.contains(&next), "suggested point should be in bounds");
}

#[test]
fn test_bayesian_optimizer_register_updates_observations() {
    let space = BayesSearchSpace::new(vec![(0.0, 1.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 2,
        n_iterations: 0,
        acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.0 },
        gp_config: GpConfig::default(),
        n_candidates: 10,
        seed: 3,
    };
    let mut opt = BayesianOptimizer::new(config, space, true);
    opt.register(vec![0.1], 0.9).expect("register");
    opt.register(vec![0.5], 0.2).expect("register");
    assert!((opt.best_y - 0.2).abs() < 1e-10, "best_y={}", opt.best_y);
}

#[test]
fn test_bayesian_optimizer_minimize_false_maximizes() {
    let space = BayesSearchSpace::new(vec![(-5.0, 5.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 5,
        n_iterations: 5,
        acquisition: AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
        gp_config: GpConfig::default(),
        n_candidates: 200,
        seed: 4,
    };
    let mut opt = BayesianOptimizer::new(config, space, false);
    let result = opt.optimize(|x| x[0]).expect("optimize");
    assert!(
        result.best_y > 0.0,
        "maximizer should find positive x, got {}",
        result.best_y
    );
}

#[test]
fn test_ei_prefers_unexplored_regions() {
    let space = BayesSearchSpace::new(vec![(-5.0, 5.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 0,
        n_iterations: 1,
        acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.0 },
        gp_config: GpConfig::default(),
        n_candidates: 1000,
        seed: 99,
    };
    let mut opt = BayesianOptimizer::new(config, space, true);
    for i in 0..5 {
        opt.register(vec![-5.0 + i as f64 * 0.5], 5.0)
            .expect("register");
    }
    let suggested = opt.suggest().expect("suggest");
    assert_eq!(suggested.len(), 1);
}

#[test]
fn test_bayesian_optimizer_2d_quadratic() {
    let space = BayesSearchSpace::new(vec![(-5.0, 5.0), (-5.0, 5.0)]).expect("space");
    let config = BayesOptConfig {
        n_initial: 8,
        n_iterations: 10,
        acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
        gp_config: GpConfig::default(),
        n_candidates: 500,
        seed: 10,
    };
    let mut opt = BayesianOptimizer::new(config, space, true);
    let result = opt
        .optimize(|x: &[f64]| x[0].powi(2) + x[1].powi(2))
        .expect("optimize");
    assert!(result.best_y < 5.0, "2D quadratic best_y={}", result.best_y);
}

// ── CMA-ES tests ──────────────────────────────────────────────────────────────

#[test]
fn test_cmaes_constants_mu_eff() {
    let mean = vec![0.0_f64; 5];
    let cfg = CmaEsConfig::new(5, mean, 0);
    let cmaes = CmaEs::new(cfg);
    assert!(cmaes.mu_eff > 1.0, "mu_eff={}", cmaes.mu_eff);
    assert!(
        cmaes.mu_eff < cmaes.config.n_parents as f64,
        "mu_eff={}",
        cmaes.mu_eff
    );
}

#[test]
fn test_cmaes_constants_c_sigma_range() {
    let mean = vec![0.0_f64; 5];
    let cfg = CmaEsConfig::new(5, mean, 0);
    let cmaes = CmaEs::new(cfg);
    assert!(
        cmaes.c_sigma > 0.0 && cmaes.c_sigma < 1.0,
        "c_sigma={}",
        cmaes.c_sigma
    );
    assert!(cmaes.d_sigma > 1.0, "d_sigma={}", cmaes.d_sigma);
    assert!(cmaes.c_1 > 0.0 && cmaes.c_1 < 1.0, "c_1={}", cmaes.c_1);
    assert!(
        cmaes.c_mu >= 0.0 && cmaes.c_mu <= 1.0,
        "c_mu={}",
        cmaes.c_mu
    );
}

#[test]
fn test_cmaes_sample_population_returns_lambda_points() {
    let n = 3;
    let mean = vec![0.0_f64; n];
    let cfg = CmaEsConfig::new(n, mean, 42);
    let lambda = cfg.population_size;
    let mut cmaes = CmaEs::new(cfg);
    let pop = cmaes.sample_population().expect("sample");
    assert_eq!(pop.len(), lambda, "population size should be λ={lambda}");
    for x in &pop {
        assert_eq!(x.len(), n, "each individual should have n={n} dims");
    }
}

#[test]
fn test_cmaes_optimize_sphere_converges() {
    let n = 3;
    let mean = vec![2.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 7);
    cfg.max_iterations = 500;
    cfg.tol_x = 1e-6;
    let mut cmaes = CmaEs::new(cfg);
    let result = cmaes
        .optimize(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
        .expect("optimize");
    assert!(
        result.best_y < 0.1,
        "should minimize sphere, got {}",
        result.best_y
    );
}

#[test]
fn test_cmaes_optimize_rosenbrock_2d() {
    let n = 2;
    let mean = vec![0.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 13);
    cfg.max_iterations = 2000;
    cfg.initial_sigma = 1.0;
    cfg.tol_x = 1e-8;
    let mut cmaes = CmaEs::new(cfg);
    let result = cmaes
        .optimize(|x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        })
        .expect("optimize");
    assert!(result.best_y < 1.0, "Rosenbrock best_y={}", result.best_y);
}

#[test]
fn test_cmaes_sigma_decreases_on_unimodal() {
    let n = 2;
    let mean = vec![5.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 55);
    cfg.max_iterations = 100;
    let initial_sigma = cfg.initial_sigma;
    let mut cmaes = CmaEs::new(cfg);
    let _ = cmaes.optimize(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>());
    assert!(
        cmaes.state.sigma <= initial_sigma + 0.5,
        "sigma should decrease, got {}",
        cmaes.state.sigma
    );
}

#[test]
fn test_cmaes_converged_near_optimum() {
    let n = 2;
    let mean = vec![0.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 77);
    cfg.tol_x = 1.0;
    cfg.initial_sigma = 0.5;
    let mut cmaes = CmaEs::new(cfg);
    let _ = cmaes.step(&|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>());
    assert!(
        cmaes.converged(),
        "should be converged with tol_x=1.0 and sigma=0.5"
    );
}

#[test]
fn test_cmaes_n_dims_1_edge_case() {
    let n = 1;
    let mean = vec![3.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 0);
    cfg.max_iterations = 200;
    let mut cmaes = CmaEs::new(cfg);
    let result = cmaes
        .optimize(|x: &[f64]| (x[0] - 1.5).powi(2))
        .expect("optimize");
    assert!(
        result.best_y < 1.0,
        "1D optimization should work, got {}",
        result.best_y
    );
}

#[test]
fn test_cmaes_population_sorted_after_evaluation() {
    let n = 2;
    let mean = vec![0.0_f64; n];
    let cfg = CmaEsConfig::new(n, mean, 5);
    let mut cmaes = CmaEs::new(cfg);
    let gen_best = cmaes
        .step(&|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
        .expect("step");
    assert!(
        gen_best <= cmaes.state.best_y + 1e-10,
        "generation best {gen_best} should be <= overall best {}",
        cmaes.state.best_y
    );
}

#[test]
fn test_cmaes_state_accessor() {
    let n = 3;
    let mean = vec![1.0_f64; n];
    let cfg = CmaEsConfig::new(n, mean.clone(), 0);
    let cmaes = CmaEs::new(cfg);
    let state = cmaes.state();
    assert_eq!(state.mean, mean);
    assert_eq!(state.iteration, 0);
}

#[test]
fn test_cmaes_weights_sum_to_one() {
    let mean = vec![0.0_f64; 4];
    let cfg = CmaEsConfig::new(4, mean, 0);
    let cmaes = CmaEs::new(cfg);
    let w_sum: f64 = cmaes.weights.iter().sum();
    assert!(
        (w_sum - 1.0).abs() < 1e-10,
        "weights should sum to 1, got {w_sum}"
    );
}

#[test]
fn test_cmaes_chi_n_formula() {
    let n = 10usize;
    let mean = vec![0.0_f64; n];
    let cfg = CmaEsConfig::new(n, mean, 0);
    let cmaes = CmaEs::new(cfg);
    let expected =
        (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));
    assert!((cmaes.chi_n - expected).abs() < 1e-10, "chi_n mismatch");
}

#[test]
fn test_cmaes_multiple_steps() {
    let n = 3;
    let mean = vec![1.0_f64; n];
    let cfg = CmaEsConfig::new(n, mean, 22);
    let mut cmaes = CmaEs::new(cfg);
    for _ in 0..10 {
        let _ = cmaes
            .step(&|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
            .expect("step");
    }
    assert_eq!(cmaes.state.iteration, 10);
}

#[test]
fn test_cmaes_n_dims_10() {
    let n = 10;
    let mean = vec![1.0_f64; n];
    let mut cfg = CmaEsConfig::new(n, mean, 33);
    cfg.max_iterations = 300;
    let mut cmaes = CmaEs::new(cfg);
    let result = cmaes
        .optimize(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
        .expect("optimize");
    assert!(result.best_y < 5.0, "10D sphere best_y={}", result.best_y);
}

// ── MultiFidelityOptimizer tests ──────────────────────────────────────────────

#[test]
fn test_multi_fidelity_constructs_correctly() {
    let levels = vec![
        FidelityLevel {
            id: 0,
            cost: 0.1,
            bias: 0.5,
        },
        FidelityLevel {
            id: 1,
            cost: 1.0,
            bias: 0.0,
        },
    ];
    let space = BayesSearchSpace::new(vec![(0.0, 1.0)]).expect("space");
    let config = MultiFidelityConfig {
        fidelity_levels: levels,
        budget: 10.0,
        base_config: BayesOptConfig {
            n_initial: 3,
            n_iterations: 3,
            acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
            gp_config: GpConfig::default(),
            n_candidates: 50,
            seed: 0,
        },
    };
    let mfo = MultiFidelityOptimizer::new(config, space);
    assert!(
        mfo.is_ok(),
        "MultiFidelityOptimizer should construct successfully"
    );
}

#[test]
fn test_multi_fidelity_runs_optimization() {
    let levels = vec![
        FidelityLevel {
            id: 0,
            cost: 0.1,
            bias: 0.5,
        },
        FidelityLevel {
            id: 1,
            cost: 1.0,
            bias: 0.0,
        },
    ];
    let space = BayesSearchSpace::new(vec![(-2.0, 2.0)]).expect("space");
    let config = MultiFidelityConfig {
        fidelity_levels: levels,
        budget: 5.0,
        base_config: BayesOptConfig {
            n_initial: 3,
            n_iterations: 2,
            acquisition: AcquisitionFunction::UpperConfidenceBound { kappa: 2.0 },
            gp_config: GpConfig::default(),
            n_candidates: 50,
            seed: 1,
        },
    };
    let mut mfo = MultiFidelityOptimizer::new(config, space).expect("create");
    let result = mfo.optimize(|x, fid| {
        let bias = if fid == 0 { 0.5 } else { 0.0 };
        x[0].powi(2) + bias
    });
    assert!(result.is_ok(), "multi-fidelity optimize should succeed");
}

#[test]
fn test_multi_fidelity_uses_low_fidelity_more() {
    let levels = vec![
        FidelityLevel {
            id: 0,
            cost: 0.1,
            bias: 0.3,
        },
        FidelityLevel {
            id: 1,
            cost: 1.0,
            bias: 0.0,
        },
    ];
    let space = BayesSearchSpace::new(vec![(-3.0, 3.0)]).expect("space");
    let config = MultiFidelityConfig {
        fidelity_levels: levels,
        budget: 3.0,
        base_config: BayesOptConfig {
            n_initial: 3,
            n_iterations: 2,
            acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
            gp_config: GpConfig::default(),
            n_candidates: 50,
            seed: 2,
        },
    };
    let mut mfo = MultiFidelityOptimizer::new(config, space).expect("create");
    let result = mfo
        .optimize(|x, fid| {
            let bias = if fid == 0 { 0.3 } else { 0.0 };
            x[0].powi(2) + bias
        })
        .expect("optimize");
    assert!(result.n_evaluations > 0, "should have evaluations");
}

#[test]
fn test_multi_fidelity_invalid_no_levels() {
    let space = BayesSearchSpace::new(vec![(0.0, 1.0)]).expect("space");
    let config = MultiFidelityConfig {
        fidelity_levels: vec![],
        budget: 10.0,
        base_config: BayesOptConfig {
            n_initial: 3,
            n_iterations: 3,
            acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
            gp_config: GpConfig::default(),
            n_candidates: 50,
            seed: 0,
        },
    };
    let result = MultiFidelityOptimizer::new(config, space);
    assert!(result.is_err(), "empty fidelity levels should fail");
}
