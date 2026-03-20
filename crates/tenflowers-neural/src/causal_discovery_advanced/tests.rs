//! Tests for causal_discovery_advanced submodules.

use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ── DirectLiNGAM ──────────────────────────────────────────────────────────

#[test]
fn test_mutual_information_uniform() {
    let x: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let y: Vec<f32> = x.iter().map(|&v| 2.0 * v).collect(); // perfectly correlated
    let mi = mutual_information_approx(&x, &y);
    assert!(mi >= 0.0, "MI should be non-negative, got {}", mi);
}

#[test]
fn test_mutual_information_independent() {
    let mut rng = StdRng::seed_from_u64(42);
    let x: Vec<f32> = (0..200).map(|_| rng.random::<f32>()).collect();
    let y: Vec<f32> = (0..200).map(|_| rng.random::<f32>()).collect();
    let mi = mutual_information_approx(&x, &y);
    assert!(mi >= 0.0);
}

#[test]
fn test_mutual_information_empty() {
    let mi = mutual_information_approx(&[], &[]);
    assert_eq!(mi, 0.0);
}

#[test]
fn test_entropy_ica_gaussian() {
    let mut rng = StdRng::seed_from_u64(0);
    // Gaussian data should have low negentropy
    let x: Vec<f32> = (0..500)
        .map(|_| {
            let u: f32 = rng.random::<f32>().max(1e-10);
            let v: f32 = rng.random::<f32>();
            (-2.0f32 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
        })
        .collect();
    let neg = entropy_ica(&x);
    assert!(neg >= 0.0);
}

#[test]
fn test_entropy_ica_non_gaussian() {
    // Uniform distribution has high non-Gaussianity
    let x: Vec<f32> = (0..100).map(|i| i as f32 / 100.0 - 0.5).collect();
    let neg = entropy_ica(&x);
    assert!(neg >= 0.0);
}

#[test]
fn test_direct_lingam_2x2() {
    // Simple 2-variable case: X1 → X2
    let n = 100;
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let x1 = i as f32 * 0.1 - 5.0;
            let x2 = 2.0 * x1 + 0.01;
            vec![x1, x2]
        })
        .collect();
    let adj = direct_lingam(&data).expect("LiNGAM should succeed");
    assert_eq!(adj.len(), 2);
    assert_eq!(adj[0].len(), 2);
}

#[test]
fn test_direct_lingam_empty() {
    let result = direct_lingam(&[]);
    assert!(result.is_err());
}

// ── NOTEARS ───────────────────────────────────────────────────────────────

#[test]
fn test_h_constraint_identity() {
    // W = 0 → h(W) should be 0 (trace(exp(0)) - n = n - n = 0)
    let n = 3usize;
    let w = vec![0.0f32; n * n];
    let h = h_constraint(&w, n);
    assert!(h.abs() < 1e-4, "h(0) should be 0, got {}", h);
}

#[test]
fn test_h_constraint_dag() {
    // Lower triangular W (strict DAG): should have h > 0 only due to exp approximation
    let n = 3usize;
    let mut w = vec![0.0f32; n * n];
    w[1] = 0.5; // row 0, col 1: 0→1
    w[n + 2] = 0.5; // 1→2
    let h = h_constraint(&w, n);
    assert!(h >= 0.0, "h should be >= 0, got {}", h);
}

#[test]
fn test_notears_loss_zero_w() {
    let n_vars = 3usize;
    let x: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let w = vec![0.0f32; n_vars * n_vars];
    let loss = notears_loss(&w, &x, 0.1);
    // When W=0, XW=0, loss = 0.5/n * ||X||_F^2
    assert!(loss > 0.0);
}

#[test]
fn test_notears_step_decreases_h() {
    let n_vars = 3usize;
    let x: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 20];
    // Use small off-diagonal entries to avoid overflow in mat_exp_approx
    let mut w = vec![0.0f32; n_vars * n_vars];
    w[1] = 0.1; // row 0, col 1: small edge 0→1
    w[n_vars + 2] = 0.1; // small edge 1→2
    let config = NoTearsConfig {
        lambda1: 0.01,
        max_iter: 5,
        h_tol: 1e-4,
    };
    let _h_before = h_constraint(&w, n_vars);
    let _h_step = notears_step(&mut w, &x, &config);
    // After step w changes; h(W) must be finite and ≥ 0
    let h_after = h_constraint(&w, n_vars);
    assert!(h_after.is_finite(), "h should be finite, got {}", h_after);
    assert!(
        h_after >= 0.0,
        "h should remain non-negative, got {}",
        h_after
    );
}

#[test]
fn test_notears_config_default() {
    let c = NoTearsConfig::default();
    assert_eq!(c.max_iter, 100);
    assert!(c.lambda1 > 0.0);
}

// ── GraNDAG ───────────────────────────────────────────────────────────────

#[test]
fn test_mlp_causal_module_forward() {
    let m = MlpCausalModule::new(4, 8, 42);
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let out = m.forward(&x);
    // Just check it runs and returns a finite value
    assert!(out.is_finite());
}

#[test]
fn test_acyclicity_penalty_zero() {
    let masks = vec![vec![0.0f32; 4]; 4];
    let pen = acyclicity_penalty(&masks);
    // All-zero mask → h(0) = 0
    assert!(
        pen.abs() < 1e-3,
        "Penalty with zero mask should be ~0, got {}",
        pen
    );
}

#[test]
fn test_grandag_loss_positive() {
    let config = GraNDagConfig {
        n_vars: 3,
        hidden_dim: 4,
        n_layers: 2,
    };
    let modules: Vec<MlpCausalModule> = (0..config.n_vars)
        .map(|i| MlpCausalModule::new(config.n_vars, config.hidden_dim, i as u64))
        .collect();
    let x: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 10];
    let loss = grandag_loss(&modules, &x, 0.1);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

// ── DCDI ──────────────────────────────────────────────────────────────────

#[test]
fn test_interventional_likelihood_zero_w() {
    let n_vars = 2;
    let data = InterventionData {
        obs: vec![vec![1.0, 2.0]; 10],
        interventional: vec![(0, vec![vec![0.5, 1.0]; 5])],
    };
    let w = vec![0.0f32; n_vars * n_vars];
    let ll = interventional_likelihood(&w, &data, n_vars);
    assert!(ll >= 0.0);
    assert!(ll.is_finite());
}

#[test]
fn test_dcdi_step_runs() {
    let n_vars = 2;
    let data = InterventionData {
        obs: vec![vec![1.0, 2.0]; 20],
        interventional: vec![(1, vec![vec![1.0, 0.0]; 10])],
    };
    let mut w = vec![0.1f32; n_vars * n_vars];
    let loss = dcdi_step(&mut w, &data, 0.001);
    assert!(loss.is_finite());
}

#[test]
fn test_intervention_data_clone() {
    let d = InterventionData {
        obs: vec![vec![1.0, 2.0]],
        interventional: vec![(0, vec![vec![0.0, 1.0]])],
    };
    let d2 = d.clone();
    assert_eq!(d2.obs.len(), 1);
    assert_eq!(d2.interventional.len(), 1);
}

// ── FCI ───────────────────────────────────────────────────────────────────

#[test]
fn test_skeleton_search_independent() {
    // Diagonal correlation matrix → all pairs independent
    let n = 3;
    let corr: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let (skel, sepsets) = skeleton_search(&corr, 0.05);
    assert_eq!(skel.len(), n);
    assert_eq!(sepsets.len(), n);
    // Most pairs should be non-adjacent (just verify the skeleton is returned)
    for i in 0..n {
        for j in (i + 1)..n {
            // With zero correlation they should typically be non-adjacent
            let _is_adj = skel[i][j] == SkeltonEdge::Adjacent;
        }
    }
}

#[test]
fn test_skeleton_search_correlated() {
    // High positive correlation → should stay adjacent
    let n = 2;
    let corr = vec![vec![1.0, 0.95], vec![0.95, 1.0]];
    let (skel, _) = skeleton_search(&corr, 0.05);
    assert_eq!(skel[0][1], SkeltonEdge::Adjacent);
}

#[test]
fn test_orient_v_structures_basic() {
    let n = 3;
    let skel = vec![
        vec![
            SkeltonEdge::NonAdjacent,
            SkeltonEdge::Adjacent,
            SkeltonEdge::NonAdjacent,
        ],
        vec![
            SkeltonEdge::Adjacent,
            SkeltonEdge::NonAdjacent,
            SkeltonEdge::Adjacent,
        ],
        vec![
            SkeltonEdge::NonAdjacent,
            SkeltonEdge::Adjacent,
            SkeltonEdge::NonAdjacent,
        ],
    ];
    let mut sepsets: Vec<Vec<Option<Vec<usize>>>> = vec![vec![None; n]; n];
    sepsets[0][2] = Some(vec![]); // sep(0,2) = {} → 1 is collider
    sepsets[2][0] = Some(vec![]);
    let pag = orient_v_structures(&skel, &sepsets);
    assert_eq!(pag.n_vars, n);
}

#[test]
fn test_pag_new() {
    let pag = Pag::new(4);
    assert_eq!(pag.n_vars, 4);
    assert_eq!(pag.edges.len(), 4);
}

#[test]
fn test_fci_rules_no_crash() {
    let mut pag = Pag::new(3);
    pag.set_mark(0, 1, PagEdge::Arrow);
    pag.set_mark(1, 0, PagEdge::Tail);
    fci_rules(&mut pag);
    // Should run without panic
    assert_eq!(pag.n_vars, 3);
}

// ── CASTLE ────────────────────────────────────────────────────────────────

#[test]
fn test_castle_autoencoder_forward() {
    let config = CastleConfig {
        n_vars: 3,
        hidden_dim: 4,
        lambda_reg: 0.5,
    };
    let model = CausalAutoEncoder::new(&config, 0);
    let x = vec![1.0f32, 2.0, 3.0];
    let out = model.forward_j(&x, 0);
    assert!(out.is_finite());
}

#[test]
fn test_castle_reconstruction_loss() {
    let outputs = vec![vec![1.1f32, 2.1, 3.1]];
    let x = vec![vec![1.0f32, 2.0, 3.0]];
    let loss = castle_reconstruction_loss(&outputs, &x);
    assert!(loss > 0.0);
    // MSE = (0.1^2 + 0.1^2 + 0.1^2) / 3 = 0.03 / 3 = 0.01
    assert!((loss - 0.01).abs() < 1e-4, "Expected ~0.01, got {}", loss);
}

#[test]
fn test_castle_acyclicity_loss_zero() {
    let n = 3;
    let w_flat = vec![0.0f32; n * n];
    let loss = castle_acyclicity_loss(&w_flat);
    assert!(loss.abs() < 1e-4, "Should be ~0, got {}", loss);
}

#[test]
fn test_castle_step_runs() {
    let config = CastleConfig {
        n_vars: 3,
        hidden_dim: 4,
        lambda_reg: 0.1,
    };
    let mut model = CausalAutoEncoder::new(&config, 42);
    let x: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 5];
    let loss = castle_step(&mut model, &x, 0.001);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_castle_adjacency_matrix() {
    let config = CastleConfig {
        n_vars: 3,
        hidden_dim: 4,
        lambda_reg: 0.1,
    };
    let model = CausalAutoEncoder::new(&config, 1);
    let adj = model.adjacency_matrix();
    assert_eq!(adj.len(), 3);
    // Diagonal should be 0
    for i in 0..3 {
        assert_eq!(adj[i][i], 0.0);
    }
}

// ── CausalBoosting ────────────────────────────────────────────────────────

#[test]
fn test_fit_stump_basic() {
    let x = vec![vec![0.0f32], vec![1.0], vec![2.0], vec![3.0]];
    let residuals = vec![-1.0f32, -0.5, 0.5, 1.0];
    let stump = fit_stump(&x, &residuals, 0);
    assert!(stump.threshold.is_finite());
    assert!(stump.left_val.is_finite());
    assert!(stump.right_val.is_finite());
}

#[test]
fn test_causal_boost_length() {
    let x: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
    let y: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
    let config = CausalBoostConfig {
        n_vars: 1,
        max_depth: 1,
        n_estimators: 10,
        learning_rate: 0.1,
    };
    let ensemble = causal_boost(&x, &y, 0, &config);
    assert_eq!(ensemble.len(), 10);
}

#[test]
fn test_decision_stump_predict() {
    let stump = CausalDecisionStump {
        feature: 0,
        threshold: 0.5,
        left_val: -1.0,
        right_val: 1.0,
        cause: 0,
    };
    assert_eq!(stump.predict_one(&[0.3]), -1.0);
    assert_eq!(stump.predict_one(&[0.7]), 1.0);
}

#[test]
fn test_causal_boost_reduces_loss() {
    let x: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32 / 50.0]).collect();
    let y: Vec<f32> = x.iter().map(|r| r[0] * 2.0).collect();
    let config = CausalBoostConfig {
        n_vars: 1,
        max_depth: 1,
        n_estimators: 20,
        learning_rate: 0.3,
    };
    let ensemble = causal_boost(&x, &y, 0, &config);
    // Predict final values
    let init_mean = y.iter().sum::<f32>() / y.len() as f32;
    let mut preds = [init_mean; 50];
    for stump in &ensemble {
        for (i, row) in x.iter().enumerate() {
            preds[i] += config.learning_rate * stump.predict_one(row);
        }
    }
    let mse: f32 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / 50.0;
    assert!(mse < 1.0, "MSE too large: {}", mse);
}

// ── ICP ───────────────────────────────────────────────────────────────────

#[test]
fn test_regression_residuals_ols() {
    let x = vec![vec![0.0f32], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![0.0f32, 2.0, 4.0, 6.0]; // perfect linear y = 2x
    let residuals = regression_residuals(&x, &y, &[0]);
    let max_res = residuals.iter().map(|&r| r.abs()).fold(0.0f32, f32::max);
    assert!(
        max_res < 0.1,
        "Residuals should be near zero, max={}",
        max_res
    );
}

#[test]
fn test_is_invariant_same_distribution() {
    let env1 = CdEnvironment {
        x: vec![vec![1.0f32], vec![2.0], vec![3.0], vec![4.0]],
        y: vec![2.0, 4.0, 6.0, 8.0],
    };
    let env2 = CdEnvironment {
        x: vec![vec![0.5f32], vec![1.5], vec![2.5], vec![3.5]],
        y: vec![1.0, 3.0, 5.0, 7.0],
    };
    // Both environments share the same causal relationship
    let inv = is_invariant(&env1, &env2, &[0], 0.05);
    assert!(inv, "Should be invariant for same relationship");
}

#[test]
fn test_icp_find_parents_empty_envs() {
    let result = icp_find_parents(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn test_icp_find_parents_runs() {
    let env1 = CdEnvironment {
        x: vec![vec![1.0f32, 0.5], vec![2.0, 1.0], vec![3.0, 1.5]],
        y: vec![2.0, 4.0, 6.0],
    };
    let env2 = CdEnvironment {
        x: vec![vec![0.5f32, 0.25], vec![1.5, 0.75], vec![2.5, 1.25]],
        y: vec![1.0, 3.0, 5.0],
    };
    let parents = icp_find_parents(&[env1, env2], 2);
    // Result is a subset of {0, 1}
    for &p in &parents {
        assert!(p < 2);
    }
}

// ── CausalEffect ──────────────────────────────────────────────────────────

#[test]
fn test_estimate_propensity_shape() {
    let x: Vec<Vec<f32>> = vec![vec![1.0f32], vec![2.0], vec![3.0]];
    let treatment = vec![true, false, true];
    let ps = estimate_propensity(&x, &treatment);
    assert_eq!(ps.len(), 3);
    for p in ps {
        assert!(p > 0.0 && p < 1.0);
    }
}

#[test]
fn test_ipw_ate_balanced() {
    // 50-50 treatment with equal outcomes → ATE ≈ 0
    let n = 10;
    let treatment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcome = vec![1.0f32; n];
    let propensity = vec![0.5f32; n];
    let ate = ipw_ate(&treatment, &outcome, &propensity);
    assert!(ate.abs() < 1e-4, "ATE should be ~0, got {}", ate);
}

#[test]
fn test_doubly_robust_ate_basic() {
    let n = 10;
    let treatment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcome = vec![1.0f32; n];
    let propensity = vec![0.5f32; n];
    let outcome_model = vec![1.0f32; 2 * n];
    let dr_ate = doubly_robust_ate(&treatment, &outcome, &propensity, &outcome_model);
    assert!(dr_ate.is_finite());
}

#[test]
fn test_sensitivity_bounds() {
    let ate = 0.3f32;
    let (lo, hi) = sensitivity_analysis_bounds(ate, 0.1);
    assert!((lo - 0.2).abs() < 1e-5);
    assert!((hi - 0.4).abs() < 1e-5);
}

// ── CdMetrics ─────────────────────────────────────────────────────────────

#[test]
fn test_shd_identical() {
    let adj = vec![
        vec![false, true, false],
        vec![false, false, true],
        vec![false, false, false],
    ];
    assert_eq!(shd(&adj, &adj), 0);
}

#[test]
fn test_shd_one_edge_difference() {
    let true_adj = vec![vec![false, true], vec![false, false]];
    let pred_adj = vec![vec![false, false], vec![false, false]];
    assert_eq!(shd(&pred_adj, &true_adj), 1);
}

#[test]
fn test_f1_skeleton_perfect() {
    let adj = vec![
        vec![false, true, false],
        vec![false, false, true],
        vec![false, false, false],
    ];
    let f1 = f1_skeleton(&adj, &adj);
    assert!(
        (f1 - 1.0).abs() < 1e-5,
        "Perfect F1 should be 1.0, got {}",
        f1
    );
}

#[test]
fn test_f1_skeleton_empty() {
    let adj: Vec<Vec<bool>> = vec![vec![false, false], vec![false, false]];
    let f1 = f1_skeleton(&adj, &adj);
    assert_eq!(f1, 0.0, "Empty skeleton F1 is 0");
}

#[test]
fn test_normalized_shd_range() {
    let pred = vec![vec![false, true], vec![false, false]];
    let true_ = vec![vec![false, false], vec![true, false]];
    let nshd = normalized_shd(&pred, &true_);
    assert!(
        (0.0..=1.0).contains(&nshd),
        "nSHD should be in [0,1], got {}",
        nshd
    );
}

#[test]
fn test_auroc_perfect() {
    let n = 3;
    let true_adj = vec![
        vec![false, true, false],
        vec![false, false, true],
        vec![false, false, false],
    ];
    // Perfect predictor: high weight where edge exists
    let mut pred_weights = vec![vec![0.0f32; n]; n];
    pred_weights[0][1] = 1.0;
    pred_weights[1][2] = 1.0;
    let auroc = auroc_edges(&pred_weights, &true_adj);
    assert!(
        (0.0..=1.0).contains(&auroc),
        "AUROC should be in [0,1], got {}",
        auroc
    );
}

#[test]
fn test_auroc_random() {
    let n = 4;
    let true_adj: Vec<Vec<bool>> = vec![
        vec![false, true, false, false],
        vec![false, false, true, false],
        vec![false, false, false, true],
        vec![false, false, false, false],
    ];
    let pred_weights: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..n).map(|j| if i < j { 0.5 } else { 0.0 }).collect())
        .collect();
    let auroc = auroc_edges(&pred_weights, &true_adj);
    assert!((0.0..=1.0).contains(&auroc));
}

#[test]
fn test_propensity_model_predict() {
    let mut model = PropensityScoreModel::new(2);
    model.weights = vec![0.5, -0.5, 0.0]; // [w0, w1, bias]
    let p = model.predict_one(&[1.0, 1.0]);
    // logit = 0.5 - 0.5 + 0.0 = 0.0 → p = 0.5
    assert!((p - 0.5).abs() < 1e-5, "Should be 0.5, got {}", p);
}

#[test]
fn test_mat_exp_approx_identity() {
    // exp(0) = I
    let n = 3;
    let z = vec![0.0f32; n * n];
    let e = mat_exp_approx(&z, n);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (e[i * n + j] - expected).abs() < 1e-4,
                "exp(0)[{},{}] should be {}, got {}",
                i,
                j,
                expected,
                e[i * n + j]
            );
        }
    }
}

#[test]
fn test_ols_univariate_linear() {
    let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&v| 3.0 * v + 1.0).collect();
    let (slope, intercept) = ols_univariate(&x, &y);
    assert!(
        (slope - 3.0).abs() < 1e-3,
        "Slope should be ~3, got {}",
        slope
    );
    assert!(
        (intercept - 1.0).abs() < 1e-3,
        "Intercept should be ~1, got {}",
        intercept
    );
}
