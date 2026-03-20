use super::*;

fn simple_tree() -> HtsTreeSpec {
    vec![
        ("A".into(), "Total".into()),
        ("B".into(), "Total".into()),
        ("AA".into(), "A".into()),
        ("AB".into(), "A".into()),
        ("BA".into(), "B".into()),
        ("BB".into(), "B".into()),
    ]
}

fn build_hierarchy() -> HtsHierarchy {
    HtsHierarchy::build_from_tree(&simple_tree()).expect("build failed")
}

#[test]
fn test_hierarchy_build() {
    let h = build_hierarchy();
    assert_eq!(h.total_nodes, 7);
    assert_eq!(h.num_bottom, 4);
    assert_eq!(h.num_levels, 3);
}

#[test]
fn test_hierarchy_root_is_index_0() {
    let h = build_hierarchy();
    assert_eq!(h.nodes[0].name, "Total");
    assert!(h.nodes[0].parent.is_none());
}

#[test]
fn test_hierarchy_leaves() {
    let h = build_hierarchy();
    let leaf_names: Vec<&str> = h
        .bottom_indices
        .iter()
        .map(|&i| h.nodes[i].name.as_str())
        .collect();
    assert!(leaf_names.contains(&"AA"));
    assert!(leaf_names.contains(&"AB"));
    assert!(leaf_names.contains(&"BA"));
    assert!(leaf_names.contains(&"BB"));
}

#[test]
fn test_summing_matrix_shape() {
    let h = build_hierarchy();
    assert_eq!(h.summing_matrix.len(), 7 * 4);
}

#[test]
fn test_aggregate_bottom() {
    let h = build_hierarchy();
    let bottom = vec![1.0, 2.0, 3.0, 4.0];
    let agg = h.aggregate_bottom(&bottom).expect("agg");
    // Total = 1+2+3+4 = 10
    assert!((agg[0] - 10.0).abs() < 1e-10);
}

#[test]
fn test_aggregate_bottom_wrong_len() {
    let h = build_hierarchy();
    assert!(h.aggregate_bottom(&[1.0, 2.0]).is_err());
}

#[test]
fn test_nodes_at_level() {
    let h = build_hierarchy();
    assert_eq!(h.nodes_at_level(0).len(), 1);
    assert_eq!(h.nodes_at_level(1).len(), 2);
    assert_eq!(h.nodes_at_level(2).len(), 4);
}

#[test]
fn test_empty_tree_spec() {
    assert!(HtsHierarchy::build_from_tree(&vec![]).is_err());
}

#[test]
fn test_grouped_hierarchy() {
    let gh = HtsGroupedHierarchy::build(
        vec!["region".into()],
        vec![vec!["north".into(), "south".into()]],
    )
    .expect("build grouped");
    assert_eq!(gh.hierarchy.num_bottom, 2);
    assert_eq!(gh.hierarchy.total_nodes, 3);
}

#[test]
fn test_bottom_up_reconcile() {
    let h = build_hierarchy();
    let mut forecasts = vec![0.0; 7];
    // Set bottom forecasts via indices
    for (i, &idx) in h.bottom_indices.iter().enumerate() {
        forecasts[idx] = (i + 1) as f64;
    }
    let rec = HtsBottomUp::reconcile(&forecasts, &h).expect("bu");
    assert!((rec[0] - 10.0).abs() < 1e-10); // Total
}

#[test]
fn test_bottom_up_wrong_len() {
    let h = build_hierarchy();
    assert!(HtsBottomUp::reconcile(&[1.0], &h).is_err());
}

#[test]
fn test_top_down_ahp() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    // 3 periods of historical data
    let mut hist = vec![0.0; 3 * n];
    for t in 0..3 {
        for (b, &idx) in h.bottom_indices.iter().enumerate() {
            hist[t * n + idx] = (b + 1) as f64;
        }
        hist[t * n] = 10.0; // Total
    }
    let props = HtsTopDown::compute_proportions(&hist, 3, &h, HtsTopDownMethod::Ahp).expect("ahp");
    assert_eq!(props.len(), 4);
    assert!((props.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_top_down_pha() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    let mut hist = vec![0.0; 2 * n];
    for t in 0..2 {
        for (b, &idx) in h.bottom_indices.iter().enumerate() {
            hist[t * n + idx] = (b + 1) as f64;
        }
        hist[t * n] = 10.0;
    }
    let props = HtsTopDown::compute_proportions(&hist, 2, &h, HtsTopDownMethod::Pha).expect("pha");
    assert!((props.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_top_down_reconcile() {
    let h = build_hierarchy();
    let mut forecasts = vec![0.0; 7];
    forecasts[0] = 100.0;
    let props = vec![0.25, 0.25, 0.25, 0.25];
    let rec = HtsTopDown::reconcile(&forecasts, &h, &props).expect("td");
    assert!((rec[0] - 100.0).abs() < 1e-10);
}

#[test]
fn test_mintrace_ols() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    let forecasts: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let residuals: Vec<f64> = vec![0.1; 3 * n];
    let rec = HtsMinTrace::reconcile(&forecasts, &h, &residuals, 3, HtsCovarianceMethod::Ols)
        .expect("ols");
    assert_eq!(rec.len(), n);
    // Reconciled should be coherent
    let ce = HtsMetrics::coherence_error(&rec, &h).expect("ce");
    assert!(ce < 1e-8);
}

#[test]
fn test_mintrace_wls() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    let forecasts: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let residuals: Vec<f64> = (0..(5 * n)).map(|i| (i as f64 * 0.01).sin()).collect();
    let rec = HtsMinTrace::reconcile(&forecasts, &h, &residuals, 5, HtsCovarianceMethod::Wls)
        .expect("wls");
    let ce = HtsMetrics::coherence_error(&rec, &h).expect("ce");
    assert!(ce < 1e-8);
}

#[test]
fn test_mintrace_shrinkage() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    let forecasts: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let residuals: Vec<f64> = (0..(5 * n)).map(|i| (i as f64 * 0.03).cos()).collect();
    let rec = HtsMinTrace::reconcile(
        &forecasts,
        &h,
        &residuals,
        5,
        HtsCovarianceMethod::Shrinkage,
    )
    .expect("shr");
    let ce = HtsMetrics::coherence_error(&rec, &h).expect("ce");
    assert!(ce < 1e-8);
}

#[test]
fn test_erm_train_and_reconcile() {
    let h = build_hierarchy();
    let n = h.total_nodes;
    let t = 10;
    let base_fc: Vec<f64> = (0..(t * n)).map(|i| (i as f64 * 0.1).sin() + 5.0).collect();
    let actuals: Vec<f64> = (0..(t * n)).map(|i| (i as f64 * 0.1).cos() + 5.0).collect();
    let erm = HtsErmReconciliation::train(&base_fc, &actuals, t, &h, 1.0).expect("erm");
    let fc: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let rec = erm.reconcile(&fc, &h).expect("erm_rec");
    let ce = HtsMetrics::coherence_error(&rec, &h).expect("ce");
    assert!(ce < 1e-8);
}

#[test]
fn test_erm_wrong_dims() {
    let h = build_hierarchy();
    assert!(HtsErmReconciliation::train(&[1.0], &[1.0], 1, &h, 1.0).is_err());
}

#[test]
fn test_temporal_hierarchy_build() {
    let th = TemporalHierarchy::new(12, &[1, 3, 6, 12]).expect("th");
    assert_eq!(th.num_base, 12);
    // freq=1: 12, freq=3: 4, freq=6: 2, freq=12: 1 => total=19
    assert_eq!(th.total_temporal, 19);
}

#[test]
fn test_temporal_aggregate() {
    let th = TemporalHierarchy::new(6, &[1, 3]).expect("th");
    let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let agg = th.aggregate(&series).expect("agg");
    // freq=1: [1,2,3,4,5,6], freq=3: [6, 15]
    assert_eq!(agg.len(), 8);
    assert!((agg[6] - 6.0).abs() < 1e-10); // sum of first 3
    assert!((agg[7] - 15.0).abs() < 1e-10); // sum of last 3
}

#[test]
fn test_temporal_empty_freq() {
    assert!(TemporalHierarchy::new(12, &[]).is_err());
}

#[test]
fn test_temporal_zero_base() {
    assert!(TemporalHierarchy::new(0, &[1]).is_err());
}

#[test]
fn test_probabilistic_forecaster_gaussian() {
    let pf = ProbabilisticForecaster::default_levels();
    let means = vec![10.0, 11.0, 12.0];
    let stds = vec![1.0, 1.5, 2.0];
    let result = pf.forecast_gaussian(&means, &stds).expect("gauss");
    assert_eq!(result.horizon, 3);
    assert_eq!(result.quantiles.len(), 3 * 7);
    // Median (quantile 0.5) should be close to mean
    assert!((result.quantiles[3] - 10.0).abs() < 0.1);
}

#[test]
fn test_probabilistic_bad_quantile() {
    assert!(ProbabilisticForecaster::new(vec![1.5]).is_err());
}

#[test]
fn test_pinball_loss() {
    let loss = ProbabilisticForecaster::pinball_loss(10.0, 8.0, 0.5);
    assert!((loss - 1.0).abs() < 1e-10); // 0.5 * 2.0 = 1.0
}

#[test]
fn test_crps_gaussian_zero_sigma() {
    let crps = ProbabilisticForecaster::crps_gaussian(5.0, 3.0, 0.0);
    assert!((crps - 2.0).abs() < 1e-10);
}

#[test]
fn test_crps_gaussian_positive_sigma() {
    let crps = ProbabilisticForecaster::crps_gaussian(5.0, 5.0, 1.0);
    assert!(crps >= 0.0);
    assert!(crps < 1.0);
}

#[test]
fn test_forecast_ses() {
    let series: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 0.1).sin()).collect();
    let result = HtsForecaster::forecast_ets(&series, 5, HtsEtsType::Ses, 0.3, None, None, None)
        .expect("ses");
    assert_eq!(result.point.len(), 5);
    assert_eq!(result.residuals.len(), 20);
    assert!(result.residual_std >= 0.0);
}

#[test]
fn test_forecast_holt() {
    let series: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.5).collect();
    let result = HtsForecaster::forecast_ets(
        &series,
        3,
        HtsEtsType::HoltLinear,
        0.3,
        Some(0.1),
        None,
        None,
    )
    .expect("holt");
    assert_eq!(result.point.len(), 3);
    // Trend series: forecasts should increase
    assert!(result.point[2] > result.point[0]);
}

#[test]
fn test_forecast_holt_damped() {
    let series: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.5).collect();
    let result = HtsForecaster::forecast_ets(
        &series,
        5,
        HtsEtsType::HoltDamped,
        0.3,
        Some(0.1),
        None,
        Some(0.9),
    )
    .expect("damped");
    assert_eq!(result.point.len(), 5);
}

#[test]
fn test_forecast_hw_additive() {
    let period = 4;
    let series: Vec<f64> = (0..24)
        .map(|i| {
            10.0 + i as f64 * 0.2 + 3.0 * ((i % period) as f64 * std::f64::consts::PI / 2.0).sin()
        })
        .collect();
    let result = HtsForecaster::forecast_ets(
        &series,
        4,
        HtsEtsType::HoltWintersAdd { period },
        0.3,
        Some(0.1),
        Some(0.1),
        None,
    )
    .expect("hw_add");
    assert_eq!(result.point.len(), 4);
}

#[test]
fn test_forecast_hw_multiplicative() {
    let period = 4;
    let series: Vec<f64> = (0..24)
        .map(|i| {
            (10.0 + i as f64 * 0.3)
                * (1.0 + 0.3 * ((i % period) as f64 * std::f64::consts::PI / 2.0).sin())
        })
        .collect();
    let result = HtsForecaster::forecast_ets(
        &series,
        4,
        HtsEtsType::HoltWintersMul { period },
        0.3,
        Some(0.1),
        Some(0.1),
        None,
    )
    .expect("hw_mul");
    assert_eq!(result.point.len(), 4);
}

#[test]
fn test_forecast_hw_short_series() {
    let series = vec![1.0, 2.0, 3.0];
    assert!(HtsForecaster::forecast_ets(
        &series,
        2,
        HtsEtsType::HoltWintersAdd { period: 4 },
        0.3,
        None,
        None,
        None,
    )
    .is_err());
}

#[test]
fn test_forecast_theta() {
    let series: Vec<f64> = (0..30)
        .map(|i| 10.0 + i as f64 * 0.3 + (i as f64 * 0.5).sin())
        .collect();
    let result = HtsForecaster::forecast_theta(&series, 5, 0.5).expect("theta");
    assert_eq!(result.point.len(), 5);
    assert_eq!(result.fitted.len(), 30);
}

#[test]
fn test_forecast_theta_short() {
    assert!(HtsForecaster::forecast_theta(&[1.0, 2.0], 3, 0.5).is_err());
}

#[test]
fn test_mase() {
    let in_sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let actuals = vec![6.0, 7.0, 8.0];
    let forecasts = vec![5.5, 7.5, 7.5];
    let m = HtsMetrics::mase(&actuals, &forecasts, &in_sample).expect("mase");
    assert!(m > 0.0);
}

#[test]
fn test_rmsse() {
    let in_sample = vec![1.0, 3.0, 5.0, 7.0];
    let actuals = vec![9.0, 11.0];
    let forecasts = vec![8.5, 10.5];
    let r = HtsMetrics::rmsse(&actuals, &forecasts, &in_sample).expect("rmsse");
    assert!(r > 0.0);
}

#[test]
fn test_mase_constant_series() {
    let in_sample = vec![5.0, 5.0, 5.0];
    assert!(HtsMetrics::mase(&[5.0], &[5.0], &in_sample).is_err());
}

#[test]
fn test_coherence_error_coherent() {
    let h = build_hierarchy();
    let bottom = vec![1.0, 2.0, 3.0, 4.0];
    let coherent = h.aggregate_bottom(&bottom).expect("agg");
    let ce = HtsMetrics::coherence_error(&coherent, &h).expect("ce");
    assert!(ce < 1e-10);
}

#[test]
fn test_coherence_error_incoherent() {
    let h = build_hierarchy();
    let incoherent = vec![99.0; 7]; // All same -- not coherent
    let ce = HtsMetrics::coherence_error(&incoherent, &h).expect("ce");
    assert!(ce > 0.0);
}

#[test]
fn test_energy_score() {
    let actuals = vec![1.0, 2.0, 3.0];
    let samples = vec![
        vec![1.1, 2.1, 3.1],
        vec![0.9, 1.9, 2.9],
        vec![1.0, 2.0, 3.0],
    ];
    let es = HtsMetrics::energy_score(&actuals, &samples).expect("es");
    assert!(es >= 0.0);
}

#[test]
fn test_energy_score_empty() {
    assert!(HtsMetrics::energy_score(&[1.0], &[]).is_err());
}

#[test]
fn test_hts_report() {
    let actuals = vec![vec![10.0, 11.0], vec![5.0, 6.0]];
    let forecasts = vec![vec![10.5, 10.5], vec![5.2, 5.8]];
    let in_sample = vec![vec![8.0, 9.0, 10.0], vec![3.0, 4.0, 5.0]];
    let report = HtsReport::generate(&actuals, &forecasts, &in_sample, 0.01).expect("report");
    assert_eq!(report.mase_by_level.len(), 2);
    assert_eq!(report.rmsse_by_level.len(), 2);
    assert!((report.coherence_error - 0.01).abs() < 1e-10);
}

#[test]
fn test_cross_temporal_reconcile() {
    // Small test: 2-node hierarchy (Total + 1 bottom) x 4-base temporal
    let tree = vec![("A".into(), "Total".into())];
    let h = HtsHierarchy::build_from_tree(&tree).expect("h");
    let th = TemporalHierarchy::new(4, &[1, 2, 4]).expect("th");
    // total_temporal = 4 + 2 + 1 = 7, cs_n=2, expected = 14
    let base = vec![1.0; 14];
    let rec = th.cross_temporal_reconcile(&base, &h).expect("ct");
    assert_eq!(rec.len(), 14);
}

#[test]
fn test_erf_approx_symmetry() {
    let e1 = erf_approx(1.0);
    let e2 = erf_approx(-1.0);
    assert!((e1 + e2).abs() < 1e-10);
}

#[test]
fn test_normal_quantile_symmetry() {
    let q25 = ProbabilisticForecaster::normal_quantile(0.25);
    let q75 = ProbabilisticForecaster::normal_quantile(0.75);
    assert!((q25 + q75).abs() < 0.05);
}

#[test]
fn test_solve_linear_identity() {
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![3.0, 7.0];
    let x = solve_linear(&a, 2, &b).expect("solve");
    assert!((x[0] - 3.0).abs() < 1e-10);
    assert!((x[1] - 7.0).abs() < 1e-10);
}

#[test]
fn test_invert_matrix_2x2() {
    let a = vec![2.0, 1.0, 1.0, 3.0];
    let inv = invert_matrix(&a, 2).expect("inv");
    let prod = mat_mul(&a, 2, 2, &inv, 2);
    assert!((prod[0] - 1.0).abs() < 1e-10);
    assert!((prod[3] - 1.0).abs() < 1e-10);
    assert!(prod[1].abs() < 1e-10);
}

#[test]
fn test_singular_matrix() {
    let a = vec![1.0, 2.0, 2.0, 4.0];
    assert!(invert_matrix(&a, 2).is_err());
}

#[test]
fn test_grouped_two_factors() {
    let gh = HtsGroupedHierarchy::build(
        vec!["region".into(), "product".into()],
        vec![vec!["N".into(), "S".into()], vec!["X".into(), "Y".into()]],
    )
    .expect("grouped");
    assert_eq!(gh.hierarchy.num_bottom, 4); // N_X, N_Y, S_X, S_Y
}

#[test]
fn test_forecast_residual_std_positive() {
    let series: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 0.5).sin()).collect();
    let r = HtsForecaster::forecast_ets(&series, 3, HtsEtsType::Ses, 0.5, None, None, None)
        .expect("ses");
    assert!(r.residual_std > 0.0);
}
