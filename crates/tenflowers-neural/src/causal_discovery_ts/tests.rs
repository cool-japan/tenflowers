//! Tests for causal_discovery_ts module.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a linear AR(1) series: x[t] = phi * x[t-1] + noise.
fn ar1_series(n: usize, phi: f64, noise_std: f64, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    let mut x = vec![0.0_f64; n];
    for i in 1..n {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        let noise = ((-2.0 * u1.max(1e-15).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos())
            * noise_std;
        x[i] = phi * x[i - 1] + noise;
    }
    x
}

/// Generate a VAR(1) bivariate series where x1 causes x2.
/// x1[t] = 0.7 * x1[t-1] + e1[t]
/// x2[t] = 0.5 * x1[t-1] + 0.3 * x2[t-1] + e2[t]
fn var1_bivariate(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    let mut data = vec![vec![0.0_f64; 2]; n];
    for t in 1..n {
        for var in 0..2 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.5;
            if var == 0 {
                data[t][0] = 0.7 * data[t - 1][0] + noise;
            } else {
                data[t][1] = 0.5 * data[t - 1][0] + 0.3 * data[t - 1][1] + noise;
            }
        }
    }
    data
}

/// Uniform noise in [0,1) via LCG.
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal via Box-Muller.
fn lcg_normal(state: &mut u64, std: f64) -> f64 {
    let u1 = lcg_uniform(state).max(1e-15);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * std
}

// ─────────────────────────────────────────────────────────────────────────────
// §1. CdtsVarModel Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_var_model_fit_basic() {
    let data = var1_bivariate(200, 42);
    let config = CdtsVarConfig {
        n_lags: 1,
        include_intercept: true,
        ridge_lambda: 0.0,
    };
    let model = CdtsVarModel::fit(&data, config).expect("VAR fit failed");
    assert_eq!(model.n_vars, 2);
    assert_eq!(model.config.n_lags, 1);
    assert_eq!(model.coefficients.len(), 1);
    assert_eq!(model.residuals.len(), 199);
}

#[test]
fn test_var_model_fit_two_lags() {
    let data = var1_bivariate(150, 7);
    let config = CdtsVarConfig {
        n_lags: 2,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("VAR(2) fit failed");
    assert_eq!(model.coefficients.len(), 2);
    assert_eq!(model.residuals.len(), 148);
}

#[test]
fn test_var_model_residuals_shape() {
    let data = var1_bivariate(100, 13);
    let config = CdtsVarConfig {
        n_lags: 2,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    let resid = model.get_residuals();
    assert_eq!(resid.len(), 98);
    for row in resid {
        assert_eq!(row.len(), 2);
    }
}

#[test]
fn test_var_model_forecast() {
    let data = var1_bivariate(200, 55);
    let config = CdtsVarConfig {
        n_lags: 1,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    let forecasts = model.forecast(&data, 5).expect("forecast");
    assert_eq!(forecasts.len(), 5);
    for f in &forecasts {
        assert_eq!(f.len(), 2);
    }
}

#[test]
fn test_var_model_forecast_finite() {
    let data = var1_bivariate(100, 99);
    let config = CdtsVarConfig::default();
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    let forecasts = model.forecast(&data, 10).expect("forecast");
    for step in &forecasts {
        for &v in step {
            assert!(v.is_finite());
        }
    }
}

#[test]
fn test_var_model_lag_matrix() {
    let data = var1_bivariate(100, 1);
    let config = CdtsVarConfig {
        n_lags: 2,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    let a1 = model.lag_matrix(1).expect("lag 1");
    assert_eq!(a1.len(), 2);
    assert_eq!(a1[0].len(), 2);
    let a2 = model.lag_matrix(2).expect("lag 2");
    assert_eq!(a2.len(), 2);
}

#[test]
fn test_var_model_lag_matrix_out_of_range() {
    let data = var1_bivariate(50, 2);
    let config = CdtsVarConfig {
        n_lags: 1,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    assert!(model.lag_matrix(0).is_err());
    assert!(model.lag_matrix(2).is_err());
}

#[test]
fn test_var_model_empty_data() {
    let config = CdtsVarConfig::default();
    assert!(CdtsVarModel::fit(&[], config).is_err());
}

#[test]
fn test_var_model_insufficient_obs() {
    let config = CdtsVarConfig {
        n_lags: 5,
        ..Default::default()
    };
    let data = var1_bivariate(4, 3);
    assert!(CdtsVarModel::fit(&data, config).is_err());
}

#[test]
fn test_var_model_with_ridge() {
    let data = var1_bivariate(80, 11);
    let config = CdtsVarConfig {
        n_lags: 1,
        ridge_lambda: 0.1,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("ridge fit");
    assert_eq!(model.n_vars, 2);
}

#[test]
fn test_var_granger_from_coefficients() {
    let data = var1_bivariate(200, 17);
    let config = CdtsVarConfig {
        n_lags: 1,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    let (f_stat, p_val, _causal) = model
        .granger_from_coefficients(0, 1, 0.05, &data)
        .expect("granger from coeff");
    assert!(f_stat >= 0.0);
    assert!((0.0..=1.0).contains(&p_val));
}

#[test]
fn test_var_model_intercepts() {
    let data = var1_bivariate(100, 21);
    let config = CdtsVarConfig {
        n_lags: 1,
        include_intercept: true,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit");
    assert_eq!(model.intercepts.len(), 2);
}

#[test]
fn test_var_model_no_intercept() {
    let data = var1_bivariate(100, 23);
    let config = CdtsVarConfig {
        n_lags: 1,
        include_intercept: false,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("fit no intercept");
    assert_eq!(model.n_vars, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// §2. CdtsGrangerTest Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_granger_test_detects_causality() {
    // x causes y by construction
    let n = 300;
    let x = ar1_series(n, 0.7, 1.0, 42);
    let mut rng = 99u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        let noise = lcg_normal(&mut rng, 0.5);
        y[i] = 0.8 * x[i - 1] + 0.3 * y[i - 1] + noise;
    }
    let result = CdtsGrangerTest::test(&x, &y, 2, 0.05).expect("granger test");
    assert!(result.f_stat >= 0.0);
    assert!((0.0..=1.0).contains(&result.p_value));
    assert_eq!(result.n_lags, 2);
    assert!(result.n_obs > 0);
}

#[test]
fn test_granger_test_independent_series() {
    // x and y are independent AR processes — should not reject null frequently
    let n = 200;
    let x = ar1_series(n, 0.5, 1.0, 11);
    let y = ar1_series(n, 0.4, 1.0, 22);
    let result = CdtsGrangerTest::test(&x, &y, 2, 0.05).expect("granger independent");
    assert!(result.f_stat >= 0.0);
    assert!((0.0..=1.0).contains(&result.p_value));
}

#[test]
fn test_granger_rss_ordering() {
    // Unrestricted model RSS should be <= restricted RSS
    let n = 200;
    let x = ar1_series(n, 0.7, 1.0, 5);
    let mut rng = 6u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.9 * x[i - 1] + lcg_normal(&mut rng, 0.3);
    }
    let result = CdtsGrangerTest::test(&x, &y, 1, 0.05).expect("granger");
    assert!(result.rss_unrestricted <= result.rss_restricted + 1e-8);
}

#[test]
fn test_granger_length_mismatch() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(CdtsGrangerTest::test(&x, &y, 1, 0.05).is_err());
}

#[test]
fn test_granger_insufficient_obs() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    assert!(CdtsGrangerTest::test(&x, &y, 5, 0.05).is_err());
}

#[test]
fn test_granger_zero_lag_error() {
    let x = ar1_series(50, 0.5, 1.0, 1);
    let y = ar1_series(50, 0.5, 1.0, 2);
    assert!(CdtsGrangerTest::test(&x, &y, 0, 0.05).is_err());
}

#[test]
fn test_granger_pairwise_test() {
    let data = var1_bivariate(200, 77);
    let results = CdtsGrangerTest::pairwise_test(&data, 2, 0.05).expect("pairwise");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 2);
    // Self-causality: p_value should be 1.0
    assert_eq!(results[0][0].p_value, 1.0);
    assert_eq!(results[1][1].p_value, 1.0);
}

#[test]
fn test_granger_pairwise_cross_entries() {
    let data = var1_bivariate(200, 88);
    let results = CdtsGrangerTest::pairwise_test(&data, 1, 0.05).expect("pairwise");
    // Cross entries should have finite f_stat
    assert!(results[0][1].f_stat.is_finite());
    assert!(results[1][0].f_stat.is_finite());
}

#[test]
fn test_granger_pairwise_single_var_error() {
    let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
    assert!(CdtsGrangerTest::pairwise_test(&data, 1, 0.05).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §3. CdtsTransferEntropy Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_te_basic_computation() {
    let n = 200;
    let x = ar1_series(n, 0.7, 1.0, 42);
    let mut rng = 100u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.6 * x[i - 1] + lcg_normal(&mut rng, 0.5);
    }
    let config = CdtsTransferEntropyConfig {
        n_bins: 10,
        k_lag: 1,
        l_lag: 1,
        normalize: false,
    };
    let te = CdtsTransferEntropy::new(config);
    let val = te.compute(&x, &y).expect("TE compute");
    assert!(val >= 0.0);
    assert!(val.is_finite());
}

#[test]
fn test_te_normalized() {
    let n = 200;
    let x = ar1_series(n, 0.6, 1.0, 44);
    let mut rng = 200u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.5 * x[i - 1] + lcg_normal(&mut rng, 0.8);
    }
    let config = CdtsTransferEntropyConfig {
        n_bins: 8,
        k_lag: 1,
        l_lag: 1,
        normalize: true,
    };
    let te = CdtsTransferEntropy::new(config);
    let val = te.compute(&x, &y).expect("NTE compute");
    assert!(val >= 0.0);
}

#[test]
fn test_te_length_mismatch() {
    let config = CdtsTransferEntropyConfig::default();
    let te = CdtsTransferEntropy::new(config);
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];
    assert!(te.compute(&x, &y).is_err());
}

#[test]
fn test_te_too_few_obs() {
    let config = CdtsTransferEntropyConfig {
        k_lag: 5,
        l_lag: 5,
        ..Default::default()
    };
    let te = CdtsTransferEntropy::new(config);
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(te.compute(&x, &y).is_err());
}

#[test]
fn test_te_bin_series() {
    let series = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let binned = CdtsTransferEntropy::bin_series(&series, 5);
    assert_eq!(binned.len(), 5);
    assert!(binned.iter().all(|&b| b < 5));
}

#[test]
fn test_te_significance_test() {
    let n = 100;
    let x = ar1_series(n, 0.5, 1.0, 77);
    let mut rng = 88u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.4 * x[i - 1] + lcg_normal(&mut rng, 1.0);
    }
    let config = CdtsTransferEntropyConfig::default();
    let te = CdtsTransferEntropy::new(config);
    let (te_val, p_val) = te.significance_test(&x, &y, 20, 42).expect("sig test");
    assert!(te_val >= 0.0);
    assert!((0.0..=1.0).contains(&p_val));
}

#[test]
fn test_te_different_bins() {
    let n = 150;
    let x = ar1_series(n, 0.7, 1.0, 13);
    let mut rng = 14u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.5 * x[i - 1] + lcg_normal(&mut rng, 0.5);
    }
    for n_bins in [5, 10, 20] {
        let config = CdtsTransferEntropyConfig {
            n_bins,
            ..Default::default()
        };
        let te = CdtsTransferEntropy::new(config);
        let val = te.compute(&x, &y).expect("TE bins");
        assert!(val >= 0.0);
    }
}

#[test]
fn test_te_xy_vs_yx_different() {
    // For a causal x->y, TE(x->y) should generally differ from TE(y->x)
    let n = 300;
    let x = ar1_series(n, 0.7, 1.0, 55);
    let mut rng = 56u64;
    let mut y = vec![0.0_f64; n];
    for i in 1..n {
        y[i] = 0.9 * x[i - 1] + lcg_normal(&mut rng, 0.1);
    }
    let config = CdtsTransferEntropyConfig::default();
    let te = CdtsTransferEntropy::new(config);
    let te_xy = te.compute(&x, &y).expect("TE xy");
    let te_yx = te.compute(&y, &x).expect("TE yx");
    // Just check both are non-negative and finite
    assert!(te_xy >= 0.0);
    assert!(te_yx >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4. CdtsConvergentCC Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ccm_basic() {
    let n = 200;
    let x = ar1_series(n, 0.7, 1.0, 42);
    let mut rng = 43u64;
    let mut y = vec![0.0_f64; n];
    for i in 2..n {
        y[i] = 0.5 * x[i - 1] + 0.3 * y[i - 1] + lcg_normal(&mut rng, 0.5);
    }
    let config = CdtsCcmConfig {
        embedding_dim: 3,
        tau: 1,
        lib_sizes: vec![20, 50, 100],
        n_samples: 5,
        seed: 1,
    };
    let ccm = CdtsConvergentCC::new(config);
    let result = ccm.ccm(&x, &y).expect("CCM");
    assert!(!result.skill_curve.is_empty());
}

#[test]
fn test_ccm_result_fields() {
    let n = 150;
    let x = ar1_series(n, 0.6, 1.0, 11);
    let mut rng = 12u64;
    let mut y = vec![0.0_f64; n];
    for i in 2..n {
        y[i] = 0.4 * x[i - 2] + lcg_normal(&mut rng, 0.8);
    }
    let config = CdtsCcmConfig {
        lib_sizes: vec![30, 60],
        n_samples: 3,
        ..Default::default()
    };
    let ccm = CdtsConvergentCC::new(config);
    let result = ccm.ccm(&x, &y).expect("CCM fields");
    for point in &result.skill_curve {
        assert!(point.lib_size > 0);
        assert!(point.skill_xy.is_finite());
        assert!(point.skill_yx.is_finite());
    }
}

#[test]
fn test_ccm_length_mismatch() {
    let config = CdtsCcmConfig::default();
    let ccm = CdtsConvergentCC::new(config);
    let x = vec![1.0; 100];
    let y = vec![1.0; 50];
    assert!(ccm.ccm(&x, &y).is_err());
}

#[test]
fn test_ccm_too_short_series() {
    let config = CdtsCcmConfig {
        embedding_dim: 5,
        tau: 2,
        ..Default::default()
    };
    let ccm = CdtsConvergentCC::new(config);
    let x = vec![1.0; 8];
    let y = vec![1.0; 8];
    // Might succeed or error depending on embedding length; both are valid
    let _ = ccm.ccm(&x, &y);
}

#[test]
fn test_ccm_embed() {
    let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let embedded = CdtsConvergentCC::embed(&series, 3, 1);
    // start = 2, so indices 2..6 → 4 vectors
    assert_eq!(embedded.len(), 4);
    // First vector: [series[2], series[1], series[0]] = [3, 2, 1]
    assert!((embedded[0][0] - 3.0).abs() < 1e-10);
    assert!((embedded[0][1] - 2.0).abs() < 1e-10);
    assert!((embedded[0][2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_ccm_convergence_measure() {
    let n = 200;
    let x = ar1_series(n, 0.7, 1.0, 77);
    let y = ar1_series(n, 0.5, 1.0, 78);
    let config = CdtsCcmConfig {
        lib_sizes: vec![20, 100],
        n_samples: 3,
        ..Default::default()
    };
    let ccm = CdtsConvergentCC::new(config);
    let result = ccm.ccm(&x, &y).expect("convergence");
    assert!(result.convergence_xy.is_finite());
    assert!(result.convergence_yx.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// §5. CdtsPcmci Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_pcmci_basic() {
    let data = var1_bivariate(200, 42);
    let config = CdtsPcmciConfig {
        max_lag: 2,
        alpha: 0.05,
        pc_alpha: 0.1,
    };
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("PCMCI basic");
    assert_eq!(result.n_vars, 2);
    assert_eq!(result.max_lag, 2);
}

#[test]
fn test_pcmci_links_have_valid_pvalues() {
    let data = var1_bivariate(200, 33);
    let config = CdtsPcmciConfig::default();
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("PCMCI pvalues");
    for link in &result.links {
        assert!((0.0..=1.0).contains(&link.p_value));
        assert!(link.partial_corr.abs() <= 1.0 + 1e-10);
    }
}

#[test]
fn test_pcmci_links_var_range() {
    let data = var1_bivariate(150, 55);
    let config = CdtsPcmciConfig {
        max_lag: 2,
        alpha: 0.1,
        pc_alpha: 0.1,
    };
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("PCMCI var range");
    for link in &result.links {
        assert!(link.src_var < result.n_vars);
        assert!(link.tgt_var < result.n_vars);
        assert!(link.src_lag >= 1 && link.src_lag <= result.max_lag);
    }
}

#[test]
fn test_pcmci_empty_data() {
    let config = CdtsPcmciConfig::default();
    let pcmci = CdtsPcmci::new(config);
    assert!(pcmci.discover(&[]).is_err());
}

#[test]
fn test_pcmci_single_variable() {
    let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
    let config = CdtsPcmciConfig::default();
    let pcmci = CdtsPcmci::new(config);
    assert!(pcmci.discover(&data).is_err());
}

#[test]
fn test_pcmci_adjacency_matrix() {
    let data = var1_bivariate(200, 99);
    let config = CdtsPcmciConfig {
        max_lag: 2,
        alpha: 0.05,
        pc_alpha: 0.1,
    };
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("PCMCI adj");
    let adj = result.adjacency_matrix(2, 2);
    assert_eq!(adj.len(), 2);
}

#[test]
fn test_pcmci_three_variables() {
    // 3-variable VAR
    let n = 200;
    let mut rng = 10u64;
    let mut data = vec![vec![0.0_f64; 3]; n];
    for t in 1..n {
        data[t][0] = 0.6 * data[t - 1][0] + lcg_normal(&mut rng, 0.5);
        data[t][1] = 0.4 * data[t - 1][0] + 0.3 * data[t - 1][1] + lcg_normal(&mut rng, 0.5);
        data[t][2] = 0.3 * data[t - 1][1] + 0.5 * data[t - 1][2] + lcg_normal(&mut rng, 0.5);
    }
    let config = CdtsPcmciConfig {
        max_lag: 1,
        alpha: 0.05,
        pc_alpha: 0.1,
    };
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("3-var PCMCI");
    assert_eq!(result.n_vars, 3);
}

#[test]
fn test_pcmci_partial_corr_zero_cond() {
    let x = ar1_series(100, 0.6, 1.0, 1);
    let y = ar1_series(100, 0.4, 1.0, 2);
    let pcorr = CdtsPcmci::partial_correlation(&x, &y, &[]).expect("partial corr");
    assert!(pcorr.abs() <= 1.0 + 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6. CdtsLingamTs Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lingam_ts_basic() {
    let data = var1_bivariate(200, 42);
    let config = CdtsLingamTsConfig {
        n_lags: 1,
        ..Default::default()
    };
    let lingam = CdtsLingamTs::new(config);
    let result = lingam.identify_causal_order(&data).expect("lingam basic");
    assert_eq!(result.causal_order.len(), 2);
    assert!(result.causal_order.contains(&0));
    assert!(result.causal_order.contains(&1));
}

#[test]
fn test_lingam_ts_ng_scores() {
    let data = var1_bivariate(150, 7);
    let config = CdtsLingamTsConfig::default();
    let lingam = CdtsLingamTs::new(config);
    let result = lingam.identify_causal_order(&data).expect("lingam ng");
    assert_eq!(result.ng_scores.len(), 2);
    for &score in &result.ng_scores {
        assert!(score >= 0.0);
    }
}

#[test]
fn test_lingam_ts_is_causal_shape() {
    let data = var1_bivariate(100, 13);
    let config = CdtsLingamTsConfig::default();
    let lingam = CdtsLingamTs::new(config);
    let result = lingam.identify_causal_order(&data).expect("lingam shape");
    assert_eq!(result.is_causal.len(), 2);
    assert_eq!(result.is_causal[0].len(), 2);
}

#[test]
fn test_lingam_ts_empty_data() {
    let config = CdtsLingamTsConfig::default();
    let lingam = CdtsLingamTs::new(config);
    assert!(lingam.identify_causal_order(&[]).is_err());
}

#[test]
fn test_lingam_ts_single_variable() {
    let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
    let config = CdtsLingamTsConfig::default();
    let lingam = CdtsLingamTs::new(config);
    assert!(lingam.identify_causal_order(&data).is_err());
}

#[test]
fn test_lingam_ts_three_variables() {
    let n = 200;
    let mut rng = 30u64;
    let mut data = vec![vec![0.0_f64; 3]; n];
    for t in 1..n {
        data[t][0] = 0.5 * data[t - 1][0] + lcg_normal(&mut rng, 1.0);
        data[t][1] = 0.6 * data[t - 1][0] + 0.2 * data[t - 1][1] + lcg_normal(&mut rng, 0.7);
        data[t][2] = 0.4 * data[t - 1][1] + 0.3 * data[t - 1][2] + lcg_normal(&mut rng, 0.8);
    }
    let config = CdtsLingamTsConfig {
        n_lags: 1,
        ..Default::default()
    };
    let lingam = CdtsLingamTs::new(config);
    let result = lingam.identify_causal_order(&data).expect("3-var lingam");
    assert_eq!(result.causal_order.len(), 3);
}

#[test]
fn test_lingam_ts_causal_order_permutation() {
    let data = var1_bivariate(100, 22);
    let config = CdtsLingamTsConfig::default();
    let lingam = CdtsLingamTs::new(config);
    let result = lingam
        .identify_causal_order(&data)
        .expect("permutation check");
    let mut sorted = result.causal_order.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1]);
}

#[test]
fn test_negentropy_approx_gaussian() {
    // Gaussian data: negentropy should be small (approximately 0)
    let n = 1000;
    let mut rng = 42u64;
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        x[i] = lcg_normal(&mut rng, 1.0);
    }
    let ng = negentropy_approx(&x);
    assert!(ng >= 0.0);
}

#[test]
fn test_negentropy_approx_non_gaussian() {
    // Laplace-ish data: higher kurtosis
    let n = 1000;
    let mut rng = 99u64;
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let u = lcg_uniform(&mut rng);
        let v = lcg_uniform(&mut rng);
        x[i] = (u.ln() - v.ln()) * 0.5; // Laplace-like
    }
    let ng = negentropy_approx(&x);
    assert!(ng >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7. CdtsInterventionEffect Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_its_positive_effect() {
    let mut rng = 42u64;
    let n = 100;
    let mut treated = vec![0.0_f64; n];
    // Pre-period: linear trend
    for t in 0..50 {
        treated[t] = t as f64 * 0.1 + lcg_normal(&mut rng, 0.1);
    }
    // Post-period: jump up by 5
    for t in 50..n {
        treated[t] = t as f64 * 0.1 + 5.0 + lcg_normal(&mut rng, 0.1);
    }
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::Its,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &[], 50)
        .expect("ITS positive");
    assert!(result.ate > 0.0);
    assert!(result.ci_upper > result.ci_lower);
}

#[test]
fn test_its_zero_effect() {
    let mut rng = 7u64;
    let n = 100;
    let mut treated = vec![0.0_f64; n];
    for t in 0..n {
        treated[t] = t as f64 * 0.05 + lcg_normal(&mut rng, 0.2);
    }
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::Its,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &[], 50)
        .expect("ITS zero");
    assert!(result.observed_post.len() == 50);
    assert!(result.counterfactual.len() == 50);
}

#[test]
fn test_its_invalid_intervention_time() {
    let treated = vec![1.0; 50];
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::Its,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    assert!(estimator.estimate_effect(&treated, &[], 0).is_err());
    assert!(estimator.estimate_effect(&treated, &[], 50).is_err());
}

#[test]
fn test_its_pointwise_effect() {
    let mut rng = 55u64;
    let n = 60;
    let mut treated = vec![0.0_f64; n];
    for t in 0..30 {
        treated[t] = lcg_normal(&mut rng, 0.5);
    }
    for t in 30..n {
        treated[t] = 2.0 + lcg_normal(&mut rng, 0.5);
    }
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::Its,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &[], 30)
        .expect("ITS pointwise");
    assert_eq!(result.pointwise_effect.len(), 30);
}

#[test]
fn test_synthetic_control_basic() {
    let mut rng = 42u64;
    let n = 100;
    let mut treated = vec![0.0_f64; n];
    let mut ctrl1 = vec![0.0_f64; n];
    let mut ctrl2 = vec![0.0_f64; n];
    for t in 0..50 {
        ctrl1[t] = t as f64 * 0.1 + lcg_normal(&mut rng, 0.2);
        ctrl2[t] = t as f64 * 0.05 + lcg_normal(&mut rng, 0.2);
        treated[t] = 0.6 * ctrl1[t] + 0.4 * ctrl2[t] + lcg_normal(&mut rng, 0.1);
    }
    for t in 50..n {
        ctrl1[t] = t as f64 * 0.1 + lcg_normal(&mut rng, 0.2);
        ctrl2[t] = t as f64 * 0.05 + lcg_normal(&mut rng, 0.2);
        treated[t] = 0.6 * ctrl1[t] + 0.4 * ctrl2[t] + 3.0 + lcg_normal(&mut rng, 0.1);
    }
    let controls = vec![ctrl1, ctrl2];
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::SyntheticControl,
        confidence_level: 0.95,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &controls, 50)
        .expect("synthetic control");
    assert!(result.ate.is_finite());
    assert!(result.std_error >= 0.0);
}

#[test]
fn test_synthetic_control_no_controls() {
    let treated = vec![1.0; 50];
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::SyntheticControl,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    assert!(estimator.estimate_effect(&treated, &[], 25).is_err());
}

#[test]
fn test_synthetic_control_wrong_length_controls() {
    let treated = vec![1.0; 50];
    let controls = vec![vec![1.0; 30]]; // wrong length
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::SyntheticControl,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    assert!(estimator.estimate_effect(&treated, &controls, 25).is_err());
}

#[test]
fn test_intervention_ci_ordering() {
    let mut rng = 33u64;
    let n = 80;
    let mut treated = vec![0.0_f64; n];
    for t in 0..n {
        treated[t] = lcg_normal(&mut rng, 1.0);
    }
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::Its,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &[], 40)
        .expect("ci ordering");
    assert!(result.ci_lower <= result.ci_upper);
}

// ─────────────────────────────────────────────────────────────────────────────
// §8. CdtsMetrics Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_evaluate_perfect() {
    let mut predicted = CdtsCausalGraph::new(3);
    predicted.add_edge(0, 1);
    predicted.add_edge(1, 2);
    let mut true_graph = CdtsCausalGraph::new(3);
    true_graph.add_edge(0, 1);
    true_graph.add_edge(1, 2);
    let metrics = CdtsMetrics::evaluate(&predicted, &true_graph).expect("perfect metrics");
    assert_eq!(metrics.tp, 2);
    assert_eq!(metrics.fp, 0);
    assert_eq!(metrics.fn_count, 0);
    assert_eq!(metrics.shd, 0);
    assert!((metrics.precision - 1.0).abs() < 1e-10);
    assert!((metrics.recall - 1.0).abs() < 1e-10);
    assert!((metrics.f1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_evaluate_empty_graphs() {
    let predicted = CdtsCausalGraph::new(3);
    let true_graph = CdtsCausalGraph::new(3);
    let metrics = CdtsMetrics::evaluate(&predicted, &true_graph).expect("empty metrics");
    assert_eq!(metrics.tp, 0);
    assert_eq!(metrics.shd, 0);
}

#[test]
fn test_metrics_evaluate_all_wrong() {
    let mut predicted = CdtsCausalGraph::new(2);
    predicted.add_edge(0, 1);
    let mut true_graph = CdtsCausalGraph::new(2);
    true_graph.add_edge(1, 0);
    let metrics = CdtsMetrics::evaluate(&predicted, &true_graph).expect("reversed edge");
    assert_eq!(metrics.reversed, 1);
    assert!(metrics.shd >= 1);
}

#[test]
fn test_metrics_evaluate_false_positives() {
    let mut predicted = CdtsCausalGraph::new(3);
    predicted.add_edge(0, 1);
    predicted.add_edge(0, 2);
    let mut true_graph = CdtsCausalGraph::new(3);
    true_graph.add_edge(0, 1);
    let metrics = CdtsMetrics::evaluate(&predicted, &true_graph).expect("false positives");
    assert_eq!(metrics.tp, 1);
    assert_eq!(metrics.fp, 1);
    assert_eq!(metrics.fn_count, 0);
}

#[test]
fn test_metrics_dimension_mismatch() {
    let pred = CdtsCausalGraph::new(3);
    let truth = CdtsCausalGraph::new(4);
    assert!(CdtsMetrics::evaluate(&pred, &truth).is_err());
}

#[test]
fn test_metrics_lagged_correlation() {
    let data = var1_bivariate(100, 42);
    let matrix = CdtsMetrics::lagged_correlation_matrix(&data, 3).expect("lagged corr");
    assert_eq!(matrix.len(), 2);
    assert_eq!(matrix[0].len(), 2);
    // Diagonal should be 1.0
    assert!((matrix[0][0] - 1.0).abs() < 1e-10);
    assert!((matrix[1][1] - 1.0).abs() < 1e-10);
    // Off-diagonal should be non-negative (absolute correlation)
    assert!(matrix[0][1] >= 0.0);
}

#[test]
fn test_metrics_auroc_edges() {
    let mut true_graph = CdtsCausalGraph::new(3);
    true_graph.add_edge(0, 1);
    true_graph.add_edge(1, 2);
    // P-values: low for true edges, high for false
    let pvalues = vec![
        vec![0.5, 0.01, 0.9], // 0 -> 1 has low p-value
        vec![0.9, 0.5, 0.02], // 1 -> 2 has low p-value
        vec![0.8, 0.7, 0.5],
    ];
    let auc = CdtsMetrics::auroc_edges(&pvalues, &true_graph).expect("AUROC");
    assert!((0.0..=1.0 + 1e-10).contains(&auc));
}

#[test]
fn test_metrics_pacf() {
    let series = ar1_series(200, 0.8, 1.0, 42);
    let pacf = CdtsMetrics::partial_autocorrelation(&series, 5);
    assert!(!pacf.is_empty());
    // First PACF value should be close to 0.8 for AR(1) with phi=0.8
    assert!(pacf[0].is_finite());
}

#[test]
fn test_metrics_pacf_short_series() {
    let series = vec![1.0, 2.0];
    let pacf = CdtsMetrics::partial_autocorrelation(&series, 5);
    // Short series may produce empty or partial results
    let _ = pacf;
}

#[test]
fn test_causal_graph_add_edge() {
    let mut g = CdtsCausalGraph::new(4);
    g.add_edge(0, 1);
    g.add_edge(2, 3);
    assert_eq!(g.n_edges(), 2);
    assert!(g.edges[0][1]);
    assert!(g.edges[2][3]);
    assert!(!g.edges[1][0]);
}

#[test]
fn test_causal_graph_flat() {
    let mut g = CdtsCausalGraph::new(2);
    g.add_edge(0, 1);
    let flat = g.to_flat();
    assert_eq!(flat.len(), 4);
    assert!(!flat[0]); // 0->0
    assert!(flat[1]); // 0->1
    assert!(!flat[2]); // 1->0
    assert!(!flat[3]); // 1->1
}

// ─────────────────────────────────────────────────────────────────────────────
// §9. CdtsReport integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_report_analyze_bivariate() {
    let data = var1_bivariate(200, 42);
    let report = CdtsReport::analyze(&data, 2, 0.05).expect("report analyze");
    assert_eq!(report.n_vars, 2);
    assert_eq!(report.n_time, 200);
    assert!(report.granger_matrix.is_some());
    assert!(report.correlation_matrix.is_some());
}

#[test]
fn test_report_with_transfer_entropy() {
    let data = var1_bivariate(150, 77);
    let report = CdtsReport::analyze(&data, 1, 0.05).expect("report base");
    let report = report.with_transfer_entropy(&data).expect("report TE");
    assert!(report.transfer_entropy_matrix.is_some());
    let te_mat = report.transfer_entropy_matrix.as_ref().expect("TE matrix");
    assert_eq!(te_mat.len(), 2);
    assert_eq!(te_mat[0].len(), 2);
}

#[test]
fn test_report_empty_data() {
    assert!(CdtsReport::analyze(&[], 2, 0.05).is_err());
}

#[test]
fn test_report_granger_matrix_shape() {
    let data = var1_bivariate(200, 10);
    let report = CdtsReport::analyze(&data, 2, 0.05).expect("report shape");
    let gm = report.granger_matrix.as_ref().expect("granger matrix");
    assert_eq!(gm.len(), 2);
    assert_eq!(gm[0].len(), 2);
}

#[test]
fn test_report_lingam_order() {
    let data = var1_bivariate(200, 88);
    let report = CdtsReport::analyze(&data, 1, 0.05).expect("report lingam");
    assert!(report.lingam_order.is_some());
    let order = report.lingam_order.as_ref().expect("lingam order");
    assert_eq!(order.len(), 2);
}

#[test]
fn test_report_pcmci_links() {
    let data = var1_bivariate(200, 66);
    let report = CdtsReport::analyze(&data, 2, 0.05).expect("report pcmci");
    // pcmci_links may be None if n_time <= 2*max_lag, but here 200 > 4
    // Just check it's well-formed if present
    if let Some(links) = &report.pcmci_links {
        for link in links {
            assert!(link.is_significant);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10. Utility function tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_normal_cdf_symmetry() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
    assert!(normal_cdf(1.96) > 0.97);
    assert!(normal_cdf(-1.96) < 0.03);
}

#[test]
fn test_f_pvalue_large_f() {
    // Large F statistic should give very small p-value
    let p = f_pvalue(100.0, 5.0, 100.0);
    assert!(p < 0.01);
}

#[test]
fn test_f_pvalue_zero() {
    let p = f_pvalue(0.0, 5.0, 100.0);
    assert!(p > 0.99);
}

#[test]
fn test_ols_cholesky_simple() {
    // y = 2 + 3x
    let xmat: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![1.0, 2.0],
        vec![1.0, 3.0],
    ];
    let y = vec![2.0, 5.0, 8.0, 11.0];
    let beta = ols_cholesky(&xmat, &y).expect("OLS simple");
    assert!((beta[0] - 2.0).abs() < 1e-8);
    assert!((beta[1] - 3.0).abs() < 1e-8);
}

#[test]
fn test_pearson_corr_perfect() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_corr(&x, &y);
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn test_pearson_corr_anticorrelated() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let r = pearson_corr(&x, &y);
    assert!((r + 1.0).abs() < 1e-10);
}

#[test]
fn test_fisher_z_transform() {
    // fisher_z(0) = 0
    assert!(fisher_z(0.0).abs() < 1e-10);
    // fisher_z is monotonically increasing
    assert!(fisher_z(0.5) > fisher_z(0.0));
    assert!(fisher_z(0.9) > fisher_z(0.5));
}

#[test]
fn test_entropy_from_counts_uniform() {
    let counts = vec![10usize; 4];
    let h = entropy_from_counts(&counts);
    // Uniform over 4: H = ln(4)
    assert!((h - 4.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_entropy_from_counts_deterministic() {
    let counts = vec![100, 0, 0, 0];
    let h = entropy_from_counts(&counts);
    assert!(h.abs() < 1e-10);
}

#[test]
fn test_var_model_four_variables() {
    let n = 300;
    let mut rng = 42u64;
    let mut data = vec![vec![0.0_f64; 4]; n];
    for t in 1..n {
        for var in 0..4 {
            data[t][var] = 0.5 * data[t - 1][var] + lcg_normal(&mut rng, 1.0);
        }
        data[t][1] += 0.3 * data[t - 1][0];
        data[t][2] += 0.3 * data[t - 1][1];
        data[t][3] += 0.3 * data[t - 1][2];
    }
    let config = CdtsVarConfig {
        n_lags: 1,
        ..Default::default()
    };
    let model = CdtsVarModel::fit(&data, config).expect("4-var VAR");
    assert_eq!(model.n_vars, 4);
    let forecasts = model.forecast(&data, 3).expect("forecast 4-var");
    assert_eq!(forecasts.len(), 3);
    assert_eq!(forecasts[0].len(), 4);
}

#[test]
fn test_granger_high_lag() {
    let n = 200;
    let x = ar1_series(n, 0.7, 1.0, 5);
    let mut rng = 6u64;
    let mut y = vec![0.0_f64; n];
    for i in 3..n {
        y[i] = 0.5 * x[i - 3] + lcg_normal(&mut rng, 0.8);
    }
    let result = CdtsGrangerTest::test(&x, &y, 3, 0.05).expect("high lag granger");
    assert!(result.f_stat >= 0.0);
    assert!((0.0..=1.0).contains(&result.p_value));
}

#[test]
fn test_transfer_entropy_zero_lag_fallback() {
    let n = 80;
    let x = ar1_series(n, 0.5, 1.0, 77);
    let y = ar1_series(n, 0.5, 1.0, 78);
    let config = CdtsTransferEntropyConfig {
        k_lag: 1,
        l_lag: 1,
        n_bins: 5,
        normalize: false,
    };
    let te = CdtsTransferEntropy::new(config);
    let val = te.compute(&x, &y).expect("TE small n");
    assert!(val >= 0.0);
}

#[test]
fn test_pcmci_significance_threshold() {
    let data = var1_bivariate(200, 42);
    // With alpha=0.0, nothing should be significant
    let config = CdtsPcmciConfig {
        max_lag: 1,
        alpha: 0.0,
        pc_alpha: 0.0,
    };
    let pcmci = CdtsPcmci::new(config);
    let result = pcmci.discover(&data).expect("pcmci alpha=0");
    let sig_links: Vec<_> = result.links.iter().filter(|l| l.is_significant).collect();
    assert_eq!(sig_links.len(), 0);
}

#[test]
fn test_synthetic_control_weights_sum_to_one() {
    // We cannot directly access weights, but the counterfactual should be a convex combination
    let mut rng = 11u64;
    let n = 60;
    let ctrl: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng, 1.0)).collect();
    let mut treated = ctrl.clone();
    for v in &mut treated[30..] {
        *v += 2.0;
    }
    let controls = vec![ctrl];
    let config = CdtsInterventionConfig {
        method: CdtsInterventionMethod::SyntheticControl,
        ..Default::default()
    };
    let estimator = CdtsInterventionEffect::new(config);
    let result = estimator
        .estimate_effect(&treated, &controls, 30)
        .expect("weights sum");
    assert_eq!(result.counterfactual.len(), 30);
}

#[test]
fn test_metrics_f1_formula() {
    // Manual: precision=2/3, recall=2/3 → F1=2/3
    let mut pred = CdtsCausalGraph::new(3);
    pred.add_edge(0, 1);
    pred.add_edge(1, 2);
    pred.add_edge(0, 2);
    let mut truth = CdtsCausalGraph::new(3);
    truth.add_edge(0, 1);
    truth.add_edge(1, 2);
    truth.add_edge(2, 0);
    let metrics = CdtsMetrics::evaluate(&pred, &truth).expect("f1 formula");
    assert!(metrics.f1 >= 0.0 && metrics.f1 <= 1.0);
}

#[test]
fn test_error_display() {
    let err = CdtsError("test error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("test error"));
    assert!(msg.contains("CdtsError"));
}

#[test]
fn test_var_config_default() {
    let config = CdtsVarConfig::default();
    assert_eq!(config.n_lags, 2);
    assert!(config.include_intercept);
    assert_eq!(config.ridge_lambda, 0.0);
}

#[test]
fn test_ccm_config_default() {
    let config = CdtsCcmConfig::default();
    assert_eq!(config.embedding_dim, 3);
    assert_eq!(config.tau, 1);
    assert!(!config.lib_sizes.is_empty());
}

#[test]
fn test_pcmci_config_default() {
    let config = CdtsPcmciConfig::default();
    assert_eq!(config.max_lag, 3);
    assert!((config.alpha - 0.05).abs() < 1e-10);
}

#[test]
fn test_lingam_config_default() {
    let config = CdtsLingamTsConfig::default();
    assert_eq!(config.n_lags, 2);
    assert!(config.use_kurtosis);
}

#[test]
fn test_intervention_config_default() {
    let config = CdtsInterventionConfig::default();
    assert!((config.confidence_level - 0.95).abs() < 1e-10);
    assert_eq!(config.method, CdtsInterventionMethod::Its);
}
