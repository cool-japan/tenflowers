//! Tests for the climate_ml module.
use super::*;

// Helper: absolute difference ≤ epsilon
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// ── FourCastNet ──────────────────────────────────────────────────────────

#[test]
fn test_fourcast_config_creation() {
    let cfg = FourCastConfig {
        lat_size: 4,
        lon_size: 4,
        n_channels: 2,
        n_modes: 2,
        embed_dim: 4,
        n_layers: 1,
    };
    assert_eq!(cfg.lat_size, 4);
    assert_eq!(cfg.n_channels, 2);
}

#[test]
fn test_fourcast_forward_shape() {
    let cfg = FourCastConfig {
        lat_size: 4,
        lon_size: 4,
        n_channels: 2,
        n_modes: 2,
        embed_dim: 4,
        n_layers: 1,
    };
    let model = FourCastNet::new(cfg);
    let x = vec![0.1_f32; 4 * 4 * 2];
    let out = model.forward(&x).expect("forward should succeed");
    assert_eq!(out.len(), 4 * 4 * 2);
}

#[test]
fn test_fourcast_wrong_size_returns_error() {
    let cfg = FourCastConfig {
        lat_size: 4,
        lon_size: 4,
        n_channels: 2,
        n_modes: 2,
        embed_dim: 4,
        n_layers: 1,
    };
    let model = FourCastNet::new(cfg);
    let x = vec![0.1_f32; 10]; // wrong size
    assert!(model.forward(&x).is_err());
}

#[test]
fn test_spherical_fourier_layer_forward() {
    let layer = SphericalFourierLayer::new(2, 4, 11);
    let x = vec![0.5_f32; 8 * 4]; // lat=8, embed=4
    let out = layer.forward(&x, 8);
    assert_eq!(out.len(), 8 * 4);
}

#[test]
fn test_rfft_irfft_roundtrip() {
    let n = 8;
    let channels = 2;
    let x: Vec<f32> = (0..n * channels).map(|i| i as f32 * 0.1).collect();
    let (re, im) = rfft_1d(&x, n);
    let reconstructed = irfft_1d(&re, &im, n);
    assert_eq!(reconstructed.len(), x.len());
    for (a, b) in x.iter().zip(reconstructed.iter()) {
        assert!(
            approx_eq(*a, *b, 1e-3),
            "IFFT reconstruction failed: {} vs {}",
            a,
            b
        );
    }
}

// ── PanGu ────────────────────────────────────────────────────────────────

#[test]
fn test_earth_position_bias_creation() {
    let bias = EarthPositionBias::new(4, 8);
    assert_eq!(bias.bias_table.len(), (2 * 4 - 1) * (2 * 8 - 1));
}

#[test]
fn test_earth_position_bias_toroidal() {
    let bias = EarthPositionBias::new(4, 8);
    // Should not panic for edge cases
    let _ = bias.get_bias(0, 0, 3, 7);
    let _ = bias.get_bias(3, 7, 0, 0);
}

#[test]
fn test_pangu_forward_output_shape() {
    let model = PanGuWeather::new(4, 8, 8, 2);
    let upper = vec![0.1_f32; 6 * 8]; // 6 upper pressure tokens
    let surface = vec![0.2_f32; 4 * 8]; // 4 surface tokens
    let (up_out, sf_out) = model.forward(&upper, &surface).expect("forward should succeed");
    assert_eq!(up_out.len(), upper.len());
    assert_eq!(sf_out.len(), surface.len());
}

#[test]
fn test_pangu_forward_residual_property() {
    let model = PanGuWeather::new(4, 8, 8, 2);
    let zeros = vec![0.0_f32; 4 * 8];
    let (up_out, sf_out) = model.forward(&zeros, &zeros).expect("forward should succeed");
    assert!(up_out.iter().all(|v| v.is_finite()));
    assert!(sf_out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_pangu_wrong_size_returns_error() {
    let model = PanGuWeather::new(4, 8, 8, 2);
    let upper = vec![0.1_f32; 7]; // not divisible by hidden=8
    let surface = vec![0.2_f32; 8];
    assert!(model.forward(&upper, &surface).is_err());
}

// ── ClimateDownscaler ────────────────────────────────────────────────────

#[test]
fn test_bicubic_upsample_shape() {
    let up = BicubicUpsample::new(2);
    let input = vec![1.0_f32; 4 * 4 * 3];
    let out = up.forward(&input, 4, 4, 3);
    assert_eq!(out.len(), 8 * 8 * 3);
}

#[test]
fn test_bicubic_upsample_uniform_preserves_value() {
    let up = BicubicUpsample::new(2);
    let input = vec![1.0_f32; 3 * 3];
    let out = up.forward(&input, 3, 3, 1);
    for &v in &out {
        assert!(approx_eq(v, 1.0, 1e-5));
    }
}

#[test]
fn test_residual_dense_block_shape() {
    let rdb = ResidualDenseBlock::new(4, 3, 42);
    let x = vec![0.5_f32; 4];
    let out = rdb.forward(&x);
    assert_eq!(out.len(), 4);
}

#[test]
fn test_downscaler_model_creation_and_forward() {
    let model = DownscalerModel::new(4, 8, 2).expect("DownscalerModel creation should succeed");
    let input = vec![0.1_f32; 4 * 4 * 2];
    let out = model.forward(&input).expect("forward should succeed");
    assert_eq!(out.len(), 8 * 8 * 2);
}

#[test]
fn test_downscaler_invalid_scale_returns_error() {
    assert!(DownscalerModel::new(4, 7, 2).is_err()); // 7 not divisible by 4
}

// ── ExtremeEventDetector ─────────────────────────────────────────────────

#[test]
fn test_focal_loss_positive_class() {
    let fl = FocalLoss::new(0.25, 2.0);
    let loss = fl.focal_loss(0.9, 1.0);
    assert!(loss >= 0.0, "Focal loss should be non-negative");
}

#[test]
fn test_focal_loss_negative_class() {
    let fl = FocalLoss::new(0.25, 2.0);
    let loss = fl.focal_loss(0.1, 0.0);
    assert!(loss >= 0.0);
}

#[test]
fn test_focal_loss_perfect_prediction_near_zero() {
    let fl = FocalLoss::new(0.25, 2.0);
    let loss = fl.focal_loss(0.9999, 1.0);
    assert!(loss < 0.01, "Perfect prediction should give near-zero loss");
}

#[test]
fn test_climate_sampler_oversample_positives() {
    let sampler = ClimateSampler::new(0.5, 3);
    let labels = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let indices = sampler.sample_indices(&labels, 100);
    let pos_count = indices.iter().filter(|&&i| labels[i] >= 0.5).count();
    let neg_count = indices.iter().filter(|&&i| labels[i] < 0.5).count();
    assert!(pos_count > neg_count, "Positives should be oversampled");
}

#[test]
fn test_extreme_event_model_probability_range() {
    let model = ExtremeEventModel::new(5, 16, 0.5);
    let features = vec![0.1_f32, -0.2, 0.3, 1.0, -0.5];
    let prob = model.predict_probability(&features);
    assert!(
        (0.0..=1.0).contains(&prob),
        "Probability out of range: {}",
        prob
    );
}

#[test]
fn test_extreme_event_model_batch_focal_loss() {
    let model = ExtremeEventModel::new(5, 16, 0.5);
    let preds = vec![0.8, 0.2, 0.6, 0.1];
    let targets = vec![1.0, 0.0, 1.0, 0.0];
    let loss = model.focal_loss.batch_loss(&preds, &targets);
    assert!(loss >= 0.0 && loss.is_finite());
}

// ── AtmosphericEmbedding ─────────────────────────────────────────────────

#[test]
fn test_variable_embedding_output_dim() {
    let ve = VariableEmbedding::new("temperature".to_string(), 8, 123);
    let emb = ve.embed(1.0);
    assert_eq!(emb.len(), 8);
}

#[test]
fn test_atmospheric_encoder_encode_shape() {
    let encoder = AtmosphericEncoder::new(&["temperature", "pressure", "humidity"], 16);
    let values = vec![
        ("temperature".to_string(), 25.0_f32),
        ("pressure".to_string(), 1013.0_f32),
        ("humidity".to_string(), 0.7_f32),
    ];
    let emb = encoder.encode(&values);
    assert_eq!(emb.len(), 16);
}

#[test]
fn test_atmospheric_encoder_unknown_variable_ignored() {
    let encoder = AtmosphericEncoder::new(&["temperature"], 8);
    let values = vec![("wind_speed".to_string(), 5.0_f32)]; // not registered
    let emb = encoder.encode(&values);
    assert_eq!(emb.len(), 8); // should equal pos_embedding (zeros)
}

// ── OceanCurrentPredictor ────────────────────────────────────────────────

#[test]
fn test_lstm_layer_step_output_shape() {
    let layer = LstmLayer::new(4, 8, 77);
    let x = vec![0.1_f32; 4];
    let h = vec![0.0_f32; 8];
    let c = vec![0.0_f32; 8];
    let (new_h, new_c) = layer.step(&x, &h, &c);
    assert_eq!(new_h.len(), 8);
    assert_eq!(new_c.len(), 8);
}

#[test]
fn test_lstm_layer_values_bounded() {
    let layer = LstmLayer::new(4, 8, 88);
    let x = vec![1.0_f32; 4];
    let h = vec![0.5_f32; 8];
    let c = vec![0.5_f32; 8];
    let (new_h, _) = layer.step(&x, &h, &c);
    // tanh output is bounded in [-1, 1]
    for &v in &new_h {
        assert!(
            (-1.0..=1.0).contains(&v),
            "LSTM output out of tanh range: {}",
            v
        );
    }
}

#[test]
fn test_ocean_lstm_forward_shape() {
    let config = OceanLstmConfig {
        input_dim: 3,
        hidden_dim: 8,
        n_layers: 2,
        n_steps_ahead: 4,
    };
    let model = OceanLstm::new(config);
    let x = vec![0.1_f32; 10 * 3]; // seq_len=10, input_dim=3
    let out = model.forward(&x, 10).expect("forward should succeed");
    assert_eq!(out.len(), 4 * 3); // n_steps_ahead * input_dim
}

// ── CarbonFluxEstimator ──────────────────────────────────────────────────

#[test]
fn test_photosynthesis_gpp_positive() {
    let model = PhotosynthesisModel::new();
    let features = EcosystemFeatures {
        lai: 3.0,
        soil_moisture: 0.4,
        air_temp: 20.0,
        vpd: 1.0,
        par: 300.0,
    };
    let gpp = model.estimate_gpp(&features);
    assert!(gpp >= 0.0, "GPP must be non-negative: {}", gpp);
}

#[test]
fn test_respiration_q10_sensitivity() {
    let model = RespirationModel::new();
    let low_temp = EcosystemFeatures {
        lai: 2.0,
        soil_moisture: 0.4,
        air_temp: 5.0,
        vpd: 0.5,
        par: 200.0,
    };
    let high_temp = EcosystemFeatures {
        lai: 2.0,
        soil_moisture: 0.4,
        air_temp: 25.0,
        vpd: 0.5,
        par: 200.0,
    };
    let r_low = model.estimate_respiration(&low_temp);
    let r_high = model.estimate_respiration(&high_temp);
    assert!(
        r_high > r_low,
        "Respiration should increase with temperature"
    );
}

#[test]
fn test_carbon_flux_estimator_predict_returns_tuple() {
    let estimator = CarbonFluxEstimator::new();
    let features = EcosystemFeatures {
        lai: 2.5,
        soil_moisture: 0.35,
        air_temp: 18.0,
        vpd: 0.8,
        par: 250.0,
    };
    let (gpp, resp) = estimator.predict(&features);
    assert!(gpp.is_finite() && resp.is_finite());
    assert!(gpp >= 0.0 && resp >= 0.0);
}

// ── ClimateProjectionEnsemble ────────────────────────────────────────────

#[test]
fn test_climate_model_predict_shape() {
    let model = ClimateModel::new(5, 3, 111);
    let x = vec![0.1_f32; 5];
    let out = model.predict(&x);
    assert_eq!(out.len(), 3);
}

#[test]
fn test_projection_ensemble_mean_std_shapes() {
    let ensemble = ProjectionEnsemble::new(5, 4, 3);
    let x = vec![0.2_f32; 4];
    let (mean, std) = ensemble.predict(&x).expect("ensemble prediction should succeed");
    assert_eq!(mean.len(), 3);
    assert_eq!(std.len(), 3);
}

#[test]
fn test_projection_ensemble_std_nonnegative() {
    let ensemble = ProjectionEnsemble::new(4, 4, 2);
    let x = vec![0.5_f32; 4];
    let (_, std) = ensemble.predict(&x).expect("ensemble prediction should succeed");
    for &s in &std {
        assert!(s >= 0.0, "Std dev must be non-negative");
    }
}

#[test]
fn test_weighted_ensemble_shape() {
    let ensemble = ProjectionEnsemble::new(3, 4, 2);
    let x = vec![0.1_f32; 4];
    let weights = vec![0.5_f32, 0.3, 0.2];
    let out = ensemble.weighted_ensemble(&x, &weights);
    assert_eq!(out.len(), 2);
}

// ── TeleconnectionAnalyzer ───────────────────────────────────────────────

#[test]
fn test_teleconnection_index_compute() {
    let idx = TeleconnectionIndex::new("ENSO".to_string(), vec![1.0, -1.0, 0.5, -0.5]);
    let field = vec![1.0_f32, 1.0, 1.0, 1.0];
    let val = idx.compute_index(&field);
    assert!(approx_eq(val, 0.0, 1e-5)); // 1-1+0.5-0.5 = 0
}

#[test]
fn test_eof_compute_pcs_shape() {
    let eof = EmpiricalOrthogonalFunction {
        n_modes: 2,
        eigenvectors: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        eigenvalues: vec![0.8, 0.1],
    };
    let data = vec![
        vec![1.0_f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let pcs = eof.compute_pcs(&data);
    assert_eq!(pcs.len(), 2); // n_modes
    assert_eq!(pcs[0].len(), 3); // n_time
}

#[test]
fn test_eof_fit_from_data() {
    let data: Vec<Vec<f32>> = (0..20)
        .map(|t| vec![t as f32 * 0.1, (t as f32 * 0.1).sin(), (t as f32).cos()])
        .collect();
    let eof = EmpiricalOrthogonalFunction::fit_from_data(&data, 2).expect("EOF fitting should succeed");
    assert_eq!(eof.n_modes, 2);
    assert!(eof.eigenvalues[0] >= eof.eigenvalues[1] - 1e-3);
}

#[test]
fn test_eof_new_mismatch_returns_error() {
    let result = EmpiricalOrthogonalFunction::new(
        3,
        vec![vec![1.0, 0.0]], // only 1 eigenvector for 3 modes
        vec![0.5, 0.3, 0.2],
    );
    assert!(result.is_err());
}

// ── ClimateMetrics ───────────────────────────────────────────────────────

#[test]
fn test_rmse_skill_score_perfect() {
    let pred = vec![1.0_f32, 2.0, 3.0];
    let obs = vec![1.0_f32, 2.0, 3.0];
    let clim = vec![0.0_f32, 0.0, 0.0];
    let score = rmse_skill_score(&pred, &obs, &clim);
    assert!(approx_eq(score, 1.0, 1e-5));
}

#[test]
fn test_rmse_skill_score_climatology_equivalent() {
    let pred = vec![0.0_f32, 0.0, 0.0];
    let obs = vec![1.0_f32, 2.0, 3.0];
    let clim = vec![0.0_f32, 0.0, 0.0];
    let score = rmse_skill_score(&pred, &obs, &clim);
    assert!(approx_eq(score, 0.0, 1e-5));
}

#[test]
fn test_anomaly_correlation_perfect() {
    let pred = vec![1.0_f32, 2.0, 3.0];
    let obs = vec![1.0_f32, 2.0, 3.0];
    let acc = anomaly_correlation(&pred, &obs);
    assert!(approx_eq(acc, 1.0, 1e-5));
}

#[test]
fn test_anomaly_correlation_anticorrelated() {
    let pred = vec![1.0_f32, 2.0, 3.0];
    let obs = vec![-1.0_f32, -2.0, -3.0];
    let acc = anomaly_correlation(&pred, &obs);
    assert!(approx_eq(acc, -1.0, 1e-5));
}

#[test]
fn test_reliability_diagram_bins() {
    let probs = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95];
    let labels = vec![
        false, false, true, true, true, false, true, true, true, true,
    ];
    let diag = reliability_diagram(&probs, &labels, 5);
    assert!(!diag.is_empty());
    for &(mp, of) in &diag {
        assert!((0.0..=1.0).contains(&mp));
        assert!((0.0..=1.0).contains(&of));
    }
}

#[test]
fn test_brier_score_perfect() {
    let probs = vec![1.0_f32, 0.0, 1.0, 0.0];
    let labels = vec![true, false, true, false];
    let bs = brier_score(&probs, &labels);
    assert!(approx_eq(bs, 0.0, 1e-5));
}

#[test]
fn test_brier_score_worst() {
    let probs = vec![0.0_f32, 1.0]; // completely wrong
    let labels = vec![true, false];
    let bs = brier_score(&probs, &labels);
    assert!(approx_eq(bs, 1.0, 1e-5));
}

#[test]
fn test_crps_ensemble_perfect() {
    let obs = vec![1.0_f32, 2.0, 3.0];
    let ensemble: Vec<Vec<f32>> = obs.iter().map(|&o| vec![o, o, o]).collect();
    let score = crps_ensemble(&ensemble, &obs);
    assert!(approx_eq(score, 0.0, 1e-4));
}

#[test]
fn test_crps_ensemble_length_mismatch_returns_nan() {
    let obs = vec![1.0_f32];
    let ensemble = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]]; // length mismatch
    let score = crps_ensemble(&ensemble, &obs);
    assert!(score.is_nan());
}
