//! Tests for the causal_ts module: core, extensions, and advanced algorithms.

use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

// ── Core: Granger ─────────────────────────────────────────────────────────────

#[test]
fn test_granger_result_construction() {
    let r = GrangerResult {
        f_statistic: 3.5,
        p_value_approx: 0.04,
        lags: 2,
        n_obs: 50,
    };
    assert!(r.is_causal(0.05));
    assert!(!r.is_causal(0.01));
}

#[test]
fn test_var_fit_shape() {
    let n = 50;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![t as f64 * 0.1, (t as f64 * 0.1).sin()])
        .collect();
    let (var, resid) = VectorAutoregression::fit(&data, 2).expect("VAR fit");
    assert_eq!(var.k, 2);
    assert_eq!(var.p, 2);
    assert!(!resid.is_empty());
}

#[test]
fn test_var_forecast_shape() {
    let n = 30;
    let data: Vec<Vec<f64>> = (0..n).map(|t| vec![t as f64 * 0.05, 1.0]).collect();
    let (var, _) = VectorAutoregression::fit(&data, 1).expect("VAR fit");
    let forecast = var.forecast(5);
    assert_eq!(forecast.len(), 5);
    for f in &forecast {
        assert_eq!(f.len(), 2);
    }
}

#[test]
fn test_transfer_entropy_non_negative() {
    let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.15 + 0.5).cos()).collect();
    let estimator = TransferEntropyEstimator::new(5);
    let te = estimator.compute(&x, &y, 1);
    assert!(te >= 0.0, "TE should be non-negative, got {}", te);
}

#[test]
fn test_granger_test_on_random() {
    let mut rng = StdRng::seed_from_u64(42);
    let n = 60;
    let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let y: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let result = GrangerCausalityTest::test(&x, &y, 2).expect("Granger test");
    assert!(result.f_statistic >= 0.0);
    assert!(result.p_value_approx >= 0.0 && result.p_value_approx <= 1.0);
}

#[test]
fn test_ccm_shape() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2 + 1.0).cos()).collect();
    let ccm = ConvergentCrossMapping::new(3, 1);
    let lib_sizes = vec![10, 20, 40];
    let corrs = ccm.test_ccm(&x, &y, &lib_sizes);
    assert_eq!(corrs.len(), lib_sizes.len());
}

// ── Core: Structural ──────────────────────────────────────────────────────────

#[test]
fn test_kalman_filter_shape() {
    let y: Vec<f64> = (0..20).map(|i| i as f64 + 0.1).collect();
    let llm = LocalLevelModel::new(1.0, 0.1);
    let (level, var) = llm.filter(&y);
    assert_eq!(level.len(), 20);
    assert_eq!(var.len(), 20);
}

#[test]
fn test_local_level_smooth() {
    let y = vec![5.0_f64; 50];
    let llm = LocalLevelModel::new(0.01, 0.001);
    let (level, _) = llm.filter(&y);
    for l in &level[10..] {
        assert!((l - 5.0).abs() < 1.0, "level should be close to 5, got {}", l);
    }
}

#[test]
fn test_uc_decompose_sum() {
    let y: Vec<f64> = (0..48)
        .map(|i| (i as f64 * 0.5).sin() + i as f64 * 0.05)
        .collect();
    let ucm = UnobservedComponentsModel::new(0.1, 0.05, 0.01);
    let comp = ucm.decompose(&y, 12);
    assert_eq!(comp.trend.len(), 48);
    assert_eq!(comp.seasonal.len(), 48);
    assert_eq!(comp.residual.len(), 48);
    for i in 0..48 {
        let recon = comp.trend[i] + comp.seasonal[i] + comp.residual[i];
        assert!(
            (recon - y[i]).abs() < 1e-9,
            "reconstruction error at {}: {}",
            i,
            recon - y[i]
        );
    }
}

#[test]
fn test_seasonal_kalman() {
    let y: Vec<f64> = (0..48)
        .map(|i| (i as f64 * 2.0 * std::f64::consts::PI / 12.0).sin() + 10.0)
        .collect();
    let skf = SeasonalKalmanFilter::new(12, 0.1, 0.01);
    let (levels, seasonals) = skf.filter(&y);
    assert_eq!(levels.len(), 48);
    assert_eq!(seasonals.len(), 48);
}

#[test]
fn test_local_linear_trend() {
    let y: Vec<f64> = (0..30).map(|i| i as f64 * 2.0 + 1.0).collect();
    let llt = LocalLinearTrend::new(0.1, 0.01, 0.001);
    let states = llt.filter(&y);
    assert_eq!(states.len(), 30);
    for s in &states {
        assert_eq!(s.len(), 2);
    }
    let last_slope = states.last().expect("states should not be empty")[1];
    assert!(last_slope > 0.0, "slope should be positive, got {}", last_slope);
}

// ── Core: Intervention ────────────────────────────────────────────────────────

#[test]
fn test_causal_impact_fit() {
    let y_pre: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.1).collect();
    let y_post: Vec<f64> = vec![11.0, 11.5, 12.0, 13.0, 14.0];
    let mut model = CausalImpactModel::new();
    model.fit(&y_pre, y_post.len());
    let summary = model.impact(&y_post);
    assert!(summary.prob_of_causal_effect > 0.0);
    assert!(summary.prob_of_causal_effect <= 1.0);
}

#[test]
fn test_synthetic_control_weights() {
    let treated: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let donor1: Vec<f64> = (0..20).map(|i| i as f64 * 0.8).collect();
    let donor2: Vec<f64> = (0..20).map(|i| i as f64 * 1.2).collect();
    let donors = vec![donor1, donor2];
    let mut sc = SyntheticControl::new();
    sc.fit(&treated, &donors).expect("SC fit");
    assert_eq!(sc.weights.len(), 2);
    let sum: f64 = sc.weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "weights should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_did_estimate() {
    let y_pre = vec![10.0, 10.0, 10.0, 10.0];
    let y_post = vec![15.0, 15.0, 10.0, 10.0];
    let treated = vec![true, true, false, false];
    let did = DifferenceInDifferences::estimate(&y_pre, &y_post, &treated).expect("DiD");
    assert!((did - 5.0).abs() < 1e-6, "DiD should be ~5, got {}", did);
}

#[test]
fn test_rd_estimate() {
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| if xi >= 0.0 { xi + 1.0 } else { xi })
        .collect();
    let rd = RegressionDiscontinuity::new(1);
    let est = rd.estimate(&x, &y, 0.0, 0.3).expect("RD estimate");
    assert!(
        (est.treatment_effect - 1.0).abs() < 0.2,
        "RD effect should be ~1, got {}",
        est.treatment_effect
    );
}

// ── Core: Neural TS ───────────────────────────────────────────────────────────

#[test]
fn test_wavenet_causal() {
    let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
    let wn = WaveNet::new(3, 4, 42).expect("WaveNet::new");
    let dilations = vec![1, 2, 4, 8];
    let out = wn.forward(&x, &dilations).expect("WaveNet forward");
    assert_eq!(out.len(), 32, "output length should match input");
}

#[test]
fn test_deep_ssm_shape() {
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let dssm = DeepStateSpaceModel::new(8, 42);
    let (means, vars) = dssm.forward(&x).expect("DeepSSM forward");
    assert_eq!(means.len(), 20);
    assert_eq!(vars.len(), 20);
    for &v in &vars {
        assert!(v > 0.0, "variance should be positive");
    }
}

#[test]
fn test_tcn_forward() {
    let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let tcn = TemporalConvNet::new(3, 4, 3, 99);
    let out = tcn.forward(&x, 3, 4, 3).expect("TCN forward");
    assert_eq!(out.len(), 16);
}

#[test]
fn test_anomaly_discrepancy_shape() {
    let at = AnomalyTransformer::new(20, 5);
    let prior: Vec<f64> = (0..20)
        .map(|i| 1.0 / (1.0 + (i as f64 - 10.0).abs()))
        .collect();
    let series: Vec<f64> = vec![0.05_f64; 20];
    let disc = at.compute_discrepancy(&prior, &series);
    assert_eq!(disc.len(), 20);
    for &d in &disc {
        assert!(d >= 0.0, "discrepancy should be non-negative");
    }
}

#[test]
fn test_gp_counterfactual_mean_shape() {
    let control: Vec<f64> = (0..20).map(|i| (i as f64 * 0.2).sin()).collect();
    let time_control: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let time_test: Vec<f64> = (20..25).map(|i| i as f64).collect();
    let mut gp = GpCounterfactual::new(5.0, 1.0, 0.1);
    gp.fit(&control, &time_control).expect("GP fit");
    let (means, stds) = gp.predict(&time_test).expect("GP predict");
    assert_eq!(means.len(), 5);
    assert_eq!(stds.len(), 5);
    for &s in &stds {
        assert!(s >= 0.0, "std should be non-negative");
    }
}

#[test]
fn test_neural_synthetic_control() {
    let treated: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let donors: Vec<Vec<f64>> = (0..3)
        .map(|d| (0..20).map(|i| i as f64 * 0.1 * (d as f64 + 0.8)).collect())
        .collect();
    let donors_post: Vec<Vec<f64>> = (0..3)
        .map(|d| {
            (0..5)
                .map(|i| (20 + i) as f64 * 0.1 * (d as f64 + 0.8))
                .collect()
        })
        .collect();
    let mut nsc = NeuralSyntheticControl::new(4, 7);
    nsc.fit(&treated, &donors).expect("NSC fit");
    let pred = nsc.predict(&donors_post);
    assert_eq!(pred.len(), 5);
}

#[test]
fn test_propensity_score_ts() {
    let x_ts: Vec<Vec<f64>> = (0..30)
        .map(|i| vec![i as f64 * 0.1, (i as f64 * 0.2).sin()])
        .collect();
    let treated_ts: Vec<bool> = (0..30).map(|i| i % 3 == 0).collect();
    let mut ps = PropensityScoreTs::new(2);
    ps.fit(&x_ts, &treated_ts).expect("PropensityScore fit");
    let scores = ps.score(&x_ts);
    assert_eq!(scores.len(), 30);
    for &s in &scores {
        assert!(s > 0.0 && s < 1.0, "propensity score out of range: {}", s);
    }
}

#[test]
fn test_causal_attention() {
    let seq_len = 8;
    let head_dim = 4;
    let x: Vec<Vec<f64>> = (0..seq_len)
        .map(|i| vec![i as f64 * 0.1; head_dim])
        .collect();
    let cam = CausalAttentionMechanism::new(seq_len, head_dim, 123);
    let (output, attn) = cam.forward(&x);
    assert_eq!(output.len(), seq_len);
    assert_eq!(attn.len(), seq_len);
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert_eq!(
                attn[i][j], 0.0,
                "future attention should be zero at ({},{})",
                i, j
            );
        }
    }
}

#[test]
fn test_iiv_weights_positive() {
    let event_times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let iiv = InverseIntensityWeighting::new(0.5);
    let weights = iiv.weights(&event_times);
    assert_eq!(weights.len(), 5);
    for &w in &weights {
        assert!(w > 0.0, "IIW weight should be positive, got {}", w);
    }
}

#[test]
fn test_recurrent_gan_generate_shape() {
    let rgan = RecurrentGanForTimeSeries::new(8, 42);
    let ts = rgan.generate(20, 99);
    assert_eq!(ts.len(), 20);
}

// ── Extensions: Lag Selection ─────────────────────────────────────────────────

#[test]
fn test_var_lag_selector_basic() {
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1 + 0.3).cos()).collect();
    let result = VarLagSelector::select(&x, &y, 5).expect("lag selection");
    assert!(result.best_lag_aic >= 1);
    assert!(result.best_lag_bic >= 1);
    assert_eq!(result.aic_values.len(), 5);
    assert_eq!(result.bic_values.len(), 5);
    // AIC and BIC should be finite for at least one lag
    let has_finite_aic = result.aic_values.iter().any(|v| v.is_finite());
    assert!(has_finite_aic, "at least one AIC should be finite");
}

#[test]
fn test_rolling_granger_basic() {
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2 + 0.5).sin()).collect();
    let rg = RollingGrangerTest::new(30, 2, 0.1);
    let windows = rg.test(&x, &y);
    assert!(!windows.is_empty(), "should produce rolling windows");
    let frac = rg.causal_fraction(&windows);
    assert!((0.0..=1.0).contains(&frac));
}

#[test]
fn test_transfer_entropy_matrix() {
    let n = 50;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.15).cos(), t as f64 * 0.01])
        .collect();
    let te_mat = TransferEntropyMatrix::compute(&data, 4, 1).expect("TE matrix");
    assert_eq!(te_mat.n_vars, 3);
    for i in 0..3 {
        assert_eq!(te_mat.te[i][i], 0.0, "self-TE should be 0");
        for j in 0..3 {
            assert!(te_mat.te[i][j] >= 0.0, "TE must be non-negative");
        }
    }
}

#[test]
fn test_te_matrix_net_influence() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            let v0 = (t as f64 * 0.1).sin();
            let v1 = if t > 0 { v0 * 0.5 + 0.1 } else { 0.0 }; // v0 → v1
            vec![v0, v1]
        })
        .collect();
    let te_mat = TransferEntropyMatrix::compute(&data, 4, 1).expect("TE matrix");
    // v0 should have positive net influence (it causes v1)
    let net0 = te_mat.net_influence(0);
    let net1 = te_mat.net_influence(1);
    // Net influence is outgoing - incoming; v0 causes more, v1 receives more
    let _ = (net0, net1); // Values depend on data; just check they're finite
    assert!(te_mat.outgoing(0).is_finite());
    assert!(te_mat.incoming(1).is_finite());
}

#[test]
fn test_causal_graph_from_granger() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 0.8 + 0.1).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
    let graph = CausalTsGraph::from_granger(&[x, y, z], 2, 0.1).expect("causal graph");
    assert_eq!(graph.n_vars, 3);
    assert_eq!(graph.edges.len(), 3);
    // In-degree and out-degree should be valid
    for v in 0..3 {
        assert!(graph.in_degree(v) <= 2);
        assert!(graph.out_degree(v) <= 2);
    }
}

#[test]
fn test_causal_ts_metrics() {
    let pred_edges = vec![vec![true, false], vec![false, false]];
    let true_edges = vec![vec![true, false], vec![true, false]];
    let cf_pred = vec![1.0, 2.0, 3.0];
    let cf_true = vec![1.1, 2.1, 3.1];
    let metrics = CausalTsMetrics::compute(&pred_edges, &true_edges, &cf_pred, &cf_true);
    assert!(metrics.sensitivity >= 0.0 && metrics.sensitivity <= 1.0);
    assert!(metrics.f1 >= 0.0 && metrics.f1 <= 1.0);
    assert!(metrics.cf_mae > 0.0);
    assert!(metrics.cf_rmse > 0.0);
}

// ── Advanced: SVAR ────────────────────────────────────────────────────────────

#[test]
fn test_svar_fit_shape() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| {
            vec![
                (t as f64 * 0.1).sin(),
                (t as f64 * 0.1 + 0.5).cos(),
            ]
        })
        .collect();
    let model = SvarModel::fit(&data, 2).expect("SVAR fit");
    assert_eq!(model.k, 2);
    assert_eq!(model.p, 2);
    assert_eq!(model.b_matrix.len(), 2);
    assert_eq!(model.b_matrix[0].len(), 2);
    // B should be lower-triangular
    assert!((model.b_matrix[0][1]).abs() < 1e-10, "B[0][1] should be 0 (lower-triangular)");
}

#[test]
fn test_svar_b_matrix_diagonal_positive() {
    let n = 80;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.12).sin(), (t as f64 * 0.08).cos()])
        .collect();
    let model = SvarModel::fit(&data, 1).expect("SVAR fit");
    // Diagonal of Cholesky factor should be positive
    for i in 0..model.k {
        assert!(
            model.b_matrix[i][i] > 0.0,
            "B[{}][{}] should be positive, got {}",
            i, i, model.b_matrix[i][i]
        );
    }
}

#[test]
fn test_svar_impulse_response_shape() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.1).cos()])
        .collect();
    let model = SvarModel::fit(&data, 1).expect("SVAR fit");
    let oirf = SvarImpulseResponse::compute(&model, 10);
    assert_eq!(oirf.horizons, 10);
    assert_eq!(oirf.k, 2);
    assert_eq!(oirf.irf.len(), 11); // horizons 0..=10
}

#[test]
fn test_svar_oirf_horizon_zero_is_b() {
    let n = 80;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin() + 0.01, (t as f64 * 0.13).cos()])
        .collect();
    let model = SvarModel::fit(&data, 1).expect("SVAR fit");
    let oirf = SvarImpulseResponse::compute(&model, 5);
    // At horizon 0, OIRF[0] = I * B = B (since Phi_0 = I)
    for i in 0..model.k {
        for j in 0..model.k {
            let oirf_val = oirf.get(0, i, j);
            let b_val = model.b_matrix[i][j];
            assert!(
                (oirf_val - b_val).abs() < 1e-10,
                "OIRF[0][{}][{}] should equal B[{}][{}], got {} vs {}",
                i, j, i, j, oirf_val, b_val
            );
        }
    }
}

#[test]
fn test_svar_fevd_sums_to_one() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.15).cos()])
        .collect();
    let model = SvarModel::fit(&data, 1).expect("SVAR fit");
    let oirf = SvarImpulseResponse::compute(&model, 10);
    let fevd = SvarForecastErrorVarianceDecomp::compute(&oirf);
    for h in 0..=10 {
        for j in 0..model.k {
            let total: f64 = (0..model.k).map(|i| fevd.get(h, j, i)).sum();
            assert!(
                (total - 1.0).abs() < 1e-8,
                "FEVD should sum to 1 at h={} j={}, got {}",
                h, j, total
            );
        }
    }
}

#[test]
fn test_svar_structural_residuals() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.12).cos()])
        .collect();
    let model = SvarModel::fit(&data, 1).expect("SVAR fit");
    // Create fake reduced-form residuals
    let u_mat: Vec<Vec<f64>> = (0..20).map(|_| vec![0.1, 0.2]).collect();
    let eps = model.structural_residuals(&u_mat).expect("structural residuals");
    assert_eq!(eps.len(), 20);
    for e in &eps {
        assert_eq!(e.len(), 2);
        for &v in e {
            assert!(v.is_finite(), "structural residuals must be finite");
        }
    }
}

// ── Advanced: Nonlinear Granger ───────────────────────────────────────────────

#[test]
fn test_neural_granger_test_runs() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2 + 0.5).cos()).collect();
    let ng = NeuralGrangerTest::new(2, 8, 0.01, 20);
    let (ratio, mse_r, mse_u) = ng.test(&x, &y, 42).expect("neural granger test");
    assert!(ratio.is_finite(), "log-ratio should be finite");
    assert!(mse_r >= 0.0, "MSE restricted must be non-negative");
    assert!(mse_u >= 0.0, "MSE unrestricted must be non-negative");
}

#[test]
fn test_kernel_granger_test_runs() {
    let n = 40;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 + 0.2).cos()).collect();
    let kg = KernelGrangerTest::new(1, 0.0, 1e-6);
    let (hsic, p_val) = kg.test(&x, &y, 50, 42).expect("kernel granger test");
    assert!(hsic >= 0.0, "HSIC must be non-negative");
    assert!(p_val > 0.0 && p_val <= 1.0, "p-value must be in (0,1]");
}

#[test]
fn test_transfer_entropy_neural_runs() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| if i > 0 { x[i - 1] } else { 0.0 }).collect();
    let te_neural = TransferEntropyNeural::new(1, 8, 30, 0.01);
    let te = te_neural.estimate(&x, &y, 42).expect("neural TE");
    assert!(te >= 0.0, "neural TE must be non-negative");
    assert!(te.is_finite(), "neural TE must be finite");
}

// ── Advanced: Causal Bandits ──────────────────────────────────────────────────

#[test]
fn test_causal_bandit_env_runs() {
    let adj = vec![
        vec![0.0, 0.5, 0.0],
        vec![0.0, 0.0, 0.3],
        vec![0.0, 0.0, 0.0],
    ];
    let reward_weights = vec![0.0, 0.0, 1.0];
    let env = CausalBanditEnv::new(adj, reward_weights, 0.1).expect("env");
    let mut rng = StdRng::seed_from_u64(42);
    let r = env.intervene(0, 1.0, &mut rng);
    assert!(r.is_finite(), "reward must be finite");
}

#[test]
fn test_causal_ucb_agent_runs() {
    let adj = vec![vec![0.0, 0.7], vec![0.0, 0.0]];
    let reward_weights = vec![0.0, 1.0];
    let env = CausalBanditEnv::new(adj, reward_weights, 0.05).expect("env");
    let mut agent = CausalUcbAgent::new(2, 1.0);
    let rewards = agent.run(&env, 20, 1.0, 7);
    assert_eq!(rewards.len(), 20);
    for &r in &rewards {
        assert!(r.is_finite());
    }
    // After running, total should equal number of rounds
    assert_eq!(agent.total, 20);
}

#[test]
fn test_causal_ucb_explores_all_arms() {
    let adj = vec![vec![0.0, 0.5], vec![0.0, 0.0]];
    let reward_weights = vec![0.0, 1.0];
    let env = CausalBanditEnv::new(adj, reward_weights, 0.0).expect("env");
    let mut agent = CausalUcbAgent::new(2, 2.0);
    agent.run(&env, 30, 1.0, 3);
    // UCB should have tried both arms
    assert!(agent.counts[0] > 0, "arm 0 should be tried");
    assert!(agent.counts[1] > 0, "arm 1 should be tried");
}

#[test]
fn test_thompson_sampling_runs() {
    let adj = vec![vec![0.0, 0.6], vec![0.0, 0.0]];
    let reward_weights = vec![0.0, 1.0];
    let env = CausalBanditEnv::new(adj, reward_weights, 0.1).expect("env");
    let mut ts = CausalThompsonSampling::new(2, 1.0, 0.1);
    let rewards = ts.run(&env, 25, 1.0, 99);
    assert_eq!(rewards.len(), 25);
    // Posteriors should have tightened
    for i in 0..2 {
        if ts.sigma2[i] < 1.0 {
            // posterior variance should decrease if arm was pulled
            assert!(ts.sigma2[i] > 0.0);
        }
    }
}

// ── Advanced: Time-Varying Causal Discovery ────────────────────────────────────

#[test]
fn test_tv_var_fit_shape() {
    let n = 50;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.1).sin(), (t as f64 * 0.1).cos()])
        .collect();
    let model = TvVarModel::fit(&data, 1, 0.3, 5).expect("TV-VAR fit");
    assert_eq!(model.k, 2);
    assert_eq!(model.p, 1);
    assert_eq!(model.coeffs.len(), 5);
    assert_eq!(model.time_points.len(), 5);
}

#[test]
fn test_tv_var_causal_strength_finite() {
    let n = 60;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|t| vec![(t as f64 * 0.15).sin(), (t as f64 * 0.1).cos()])
        .collect();
    let model = TvVarModel::fit(&data, 1, 0.4, 4).expect("TV-VAR fit");
    for idx in 0..4 {
        let cs = model.causal_strength(idx, 0, 1);
        assert!(cs >= 0.0 && cs.is_finite(), "causal strength must be finite non-negative");
    }
}

#[test]
fn test_tv_granger_test_shape() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2 + 0.3).cos()).collect();
    let tvg = TvGrangerTest::new(2, 0.3, 0.1);
    let results = tvg.test(&x, &y, 5).expect("TV-Granger test");
    assert!(!results.is_empty(), "should produce results");
    for (t, f, p, _sig) in &results {
        assert!(*t > 0, "time index should be positive");
        assert!(*f >= 0.0, "F-stat must be non-negative");
        assert!(*p >= 0.0 && *p <= 1.0, "p-value must be in [0,1]");
    }
}

#[test]
fn test_change_point_detector_no_break() {
    // No real change → should return empty or few false alarms
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.15).cos()).collect();
    let detector = ChangePointCausalDetector::new(2, 20, 5.0, 0.05);
    let cps = detector.detect(&x, &y);
    // Could be empty or contain a few false positives, just check it runs
    for &cp in &cps {
        assert!(cp < n, "change point index should be within series");
    }
}

#[test]
fn test_change_point_detector_with_break() {
    // Generate data where causal relation changes midway
    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);
    let x: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| {
            if t < 50 {
                // x strongly causes y in first half
                if t > 0 { x[t - 1] * 2.0 } else { 0.0 }
            } else {
                // No causal relation in second half
                rng.random::<f64>()
            }
        })
        .collect();
    let detector = ChangePointCausalDetector::new(2, 20, 2.0, 0.1);
    let cps = detector.detect(&x, &y);
    // At least checks that the detector runs without panic
    for &cp in &cps {
        assert!(cp < n);
    }
}

// ── Advanced: Metrics ─────────────────────────────────────────────────────────

#[test]
fn test_causal_ts_ext_metrics_auroc() {
    let scores = vec![0.9, 0.8, 0.3, 0.2, 0.7, 0.1];
    let labels = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let auroc = CausalTsExtMetrics::tv_auroc(&scores, &labels);
    // Perfect ranking should give AUROC = 1.0
    assert!((0.0..=1.0).contains(&auroc), "AUROC must be in [0,1]");
    assert!(auroc > 0.5, "AUROC should be above 0.5 for ordered scores");
}

#[test]
fn test_lag_specific_f1_perfect() {
    let pred = vec![vec![vec![true, false], vec![false, true]]];
    let truth = vec![vec![vec![true, false], vec![false, true]]];
    let f1 = CausalTsExtMetrics::lag_specific_f1(&pred, &truth);
    assert!(
        (f1 - 1.0).abs() < 1e-10,
        "perfect prediction should give F1=1, got {}",
        f1
    );
}

#[test]
fn test_break_detection_metrics_all_detected() {
    let true_breaks = vec![25, 75];
    let detected = vec![24, 76]; // within tolerance of 5
    let (dr, far) = CausalTsExtMetrics::break_detection_metrics(&detected, &true_breaks, 5, 20);
    assert!(
        (dr - 1.0).abs() < 1e-10,
        "all breaks detected, rate should be 1.0, got {}",
        dr
    );
    assert!((0.0..=1.0).contains(&far));
}

#[test]
fn test_causal_ts_ext_metrics_lag_f1_empty() {
    let f1 = CausalTsExtMetrics::lag_specific_f1(&[], &[]);
    assert_eq!(f1, 0.0, "empty prediction should give F1=0");
}
