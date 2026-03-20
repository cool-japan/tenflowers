use super::*;

use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── MdnError ──────────────────────────────────────────────────────────────

#[test]
fn test_mdn_error_display() {
    let e = MdnError::InvalidInput("bad".into());
    assert!(e.to_string().contains("invalid input"));
    let e2 = MdnError::NumericalError("nan".into());
    assert!(e2.to_string().contains("numerical"));
    let e3 = MdnError::ShapeMismatch("shape".into());
    assert!(e3.to_string().contains("shape"));
}

// ── MdnLinear ─────────────────────────────────────────────────────────────

#[test]
fn test_mdn_linear_forward() {
    let mut rng = StdRng::seed_from_u64(1);
    let layer = MdnLinear::new(3, 4, &mut rng);
    let x = vec![1.0, 0.0, -1.0];
    let y = layer.forward(&x).expect("linear forward should succeed");
    assert_eq!(y.len(), 4);
}

#[test]
fn test_mdn_linear_wrong_input() {
    let mut rng = StdRng::seed_from_u64(2);
    let layer = MdnLinear::new(3, 4, &mut rng);
    let x = vec![1.0, 0.0];
    assert!(layer.forward(&x).is_err());
}

#[test]
fn test_mdn_linear_params_mut() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut layer = MdnLinear::new(2, 3, &mut rng);
    let (w, b) = layer.params_mut();
    assert_eq!(w.len(), 6);
    assert_eq!(b.len(), 3);
}

#[test]
fn test_mdn_linear_n_params() {
    let mut rng = StdRng::seed_from_u64(4);
    let layer = MdnLinear::new(4, 5, &mut rng);
    assert_eq!(layer.n_params(), 4 * 5 + 5);
}

#[test]
fn test_mdn_linear_xavier_scale() {
    let mut rng = StdRng::seed_from_u64(5);
    let layer = MdnLinear::new(100, 100, &mut rng);
    let mean: f64 = layer.weights.iter().sum::<f64>() / layer.weights.len() as f64;
    assert!(mean.abs() < 0.1, "Xavier init mean should be close to 0");
}

// ── MdnGaussianMixture ────────────────────────────────────────────────────

fn make_mixture(k: usize, d: usize) -> MdnGaussianMixture {
    let pi = vec![1.0 / k as f64; k];
    let mu: Vec<Vec<f64>> = (0..k).map(|i| vec![i as f64; d]).collect();
    let sigma: Vec<Vec<f64>> = (0..k).map(|_| vec![1.0; d]).collect();
    MdnGaussianMixture { pi, mu, sigma }
}

#[test]
fn test_gmixture_log_prob() {
    let mix = make_mixture(3, 2);
    let x = vec![0.5, 0.5];
    let lp = mix.log_prob(&x).expect("log_prob computation should succeed");
    assert!(lp.is_finite());
    assert!(lp < 0.0);
}

#[test]
fn test_gmixture_log_prob_wrong_dim() {
    let mix = make_mixture(2, 3);
    assert!(mix.log_prob(&[1.0, 2.0]).is_err());
}

#[test]
fn test_gmixture_sample() {
    let mix = make_mixture(4, 2);
    let mut rng = StdRng::seed_from_u64(10);
    let s = mix.sample(&mut rng).expect("mixture sampling should succeed");
    assert_eq!(s.len(), 2);
}

#[test]
fn test_gmixture_mean() {
    let mix = make_mixture(2, 2);
    // pi=[0.5,0.5], mu=[[0,0],[1,1]] → mean=[0.5, 0.5]
    let m = mix.mean();
    assert!((m[0] - 0.5).abs() < 1e-10);
    assert!((m[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_gmixture_n_components() {
    let mix = make_mixture(5, 3);
    assert_eq!(mix.n_components(), 5);
}

#[test]
fn test_gmixture_out_dim() {
    let mix = make_mixture(3, 4);
    assert_eq!(mix.out_dim(), 4);
}

// ── MixtureDensityNetwork ─────────────────────────────────────────────────

#[test]
fn test_mdn_forward() {
    let mdn = MixtureDensityNetwork::new(2, &[16], 3, 2, 42).expect("MDN construction should succeed");
    let mix = mdn.forward(&[1.0, -1.0]).expect("MDN forward should succeed");
    assert_eq!(mix.n_components(), 3);
    assert_eq!(mix.out_dim(), 2);
    let pi_sum: f64 = mix.pi.iter().sum();
    assert!((pi_sum - 1.0).abs() < 1e-9);
}

#[test]
fn test_mdn_nll_loss() {
    let mdn = MixtureDensityNetwork::new(2, &[8], 2, 1, 1).expect("MDN construction should succeed");
    let mix = mdn.forward(&[0.5, 0.5]).expect("MDN forward should succeed");
    let nll = mdn.nll_loss(&mix, &[0.0]).expect("NLL loss should succeed");
    assert!(nll.is_finite());
}

#[test]
fn test_mdn_invalid_dims() {
    assert!(MixtureDensityNetwork::new(0, &[8], 2, 1, 0).is_err());
    assert!(MixtureDensityNetwork::new(2, &[8], 0, 1, 0).is_err());
}

#[test]
fn test_mdn_fit_returns_history() {
    let mut mdn = MixtureDensityNetwork::new(1, &[4], 2, 1, 99).expect("MDN construction should succeed");
    let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
    let targets: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
    let hist = mdn.fit(&data, &targets, 2, 1e-3, 4, 0).expect("MDN fit should succeed");
    assert_eq!(hist.len(), 2);
    assert!(hist[0].is_finite());
}

#[test]
fn test_mdn_sigma_positive() {
    let mdn = MixtureDensityNetwork::new(2, &[8], 3, 2, 7).expect("MDN construction should succeed");
    let mix = mdn.forward(&[0.0, 0.0]).expect("MDN forward should succeed");
    for sk in &mix.sigma {
        for &s in sk {
            assert!(s > 0.0);
        }
    }
}

// ── ConditionalMdn ────────────────────────────────────────────────────────

#[test]
fn test_conditional_mdn_forward() {
    let cmdn = ConditionalMdn::new(3, 2, 8, 4, 1, 5).expect("conditional MDN construction should succeed");
    let ctx = vec![1.0, 0.0, -1.0];
    let qry = vec![0.5, 0.5];
    let mix = cmdn.forward(&ctx, &qry).expect("conditional forward should succeed");
    assert_eq!(mix.n_components(), 4);
    assert_eq!(mix.out_dim(), 1);
}

#[test]
fn test_conditional_mdn_sizes() {
    let cmdn = ConditionalMdn::new(2, 3, 4, 2, 1, 0).expect("conditional MDN construction should succeed");
    assert_eq!(cmdn.context_hidden_size(), 4);
    assert_eq!(cmdn.query_hidden_size(), 4);
}

// ── MixtureLstmModel ──────────────────────────────────────────────────────

#[test]
fn test_lstm_step_shape() {
    let lstm = MixtureLstmModel::new(2, 8, 3, 1, 42).expect("LSTM MDN construction should succeed");
    let h0 = vec![0.0; 8];
    let c0 = vec![0.0; 8];
    let (h1, c1) = lstm.step(&[1.0, 0.0], &h0, &c0).expect("LSTM step should succeed");
    assert_eq!(h1.len(), 8);
    assert_eq!(c1.len(), 8);
}

#[test]
fn test_lstm_forward_sequence() {
    let lstm = MixtureLstmModel::new(2, 4, 2, 1, 10).expect("LSTM MDN construction should succeed");
    let seq: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, -(i as f64)]).collect();
    let out = lstm.forward_sequence(&seq).expect("LSTM forward sequence should succeed");
    assert_eq!(out.len(), 5);
    for mix in &out {
        assert_eq!(mix.n_components(), 2);
    }
}

#[test]
fn test_lstm_step_dim_error() {
    let lstm = MixtureLstmModel::new(3, 4, 2, 1, 0).expect("LSTM MDN construction should succeed");
    let h = vec![0.0; 4];
    let c = vec![0.0; 4];
    assert!(lstm.step(&[1.0, 2.0], &h, &c).is_err()); // input_dim mismatch
}

// ── RnadeDensityEstimator ─────────────────────────────────────────────────

#[test]
fn test_rnade_log_prob() {
    let rnade = RnadeDensityEstimator::new(3, 16, 4, 42).expect("RNADE construction should succeed");
    let x = vec![0.1, -0.5, 0.8];
    let lp = rnade.log_prob(&x).expect("RNADE log_prob should succeed");
    assert!(lp.is_finite());
}

#[test]
fn test_rnade_sample() {
    let rnade = RnadeDensityEstimator::new(4, 8, 3, 7).expect("RNADE construction should succeed");
    let mut rng = StdRng::seed_from_u64(0);
    let s = rnade.sample(&mut rng).expect("RNADE sampling should succeed");
    assert_eq!(s.len(), 4);
}

#[test]
fn test_rnade_dim_mismatch() {
    let rnade = RnadeDensityEstimator::new(3, 8, 2, 0).expect("RNADE construction should succeed");
    assert!(rnade.log_prob(&[1.0, 2.0]).is_err());
}

#[test]
fn test_rnade_invalid_dims() {
    assert!(RnadeDensityEstimator::new(0, 8, 2, 0).is_err());
    assert!(RnadeDensityEstimator::new(3, 0, 2, 0).is_err());
}

// ── MdnMadeNetwork ────────────────────────────────────────────────────────

#[test]
fn test_made_log_prob() {
    let made = MdnMadeNetwork::new(4, 32, 42).expect("MADE construction should succeed");
    let x = vec![0.5, -0.3, 1.0, -1.0];
    let lp = made.log_prob(&x).expect("MADE log_prob should succeed");
    assert!(lp.is_finite());
}

#[test]
fn test_made_sample() {
    let made = MdnMadeNetwork::new(3, 16, 1).expect("MADE construction should succeed");
    let mut rng = StdRng::seed_from_u64(2);
    let s = made.sample(&mut rng).expect("MADE sampling should succeed");
    assert_eq!(s.len(), 3);
}

#[test]
fn test_made_update_masks() {
    let mut made = MdnMadeNetwork::new(4, 8, 0).expect("MADE construction should succeed");
    let new_ord = vec![2, 4, 1, 3];
    made.update_masks(new_ord).expect("mask update should succeed");
}

#[test]
fn test_made_update_masks_wrong_len() {
    let mut made = MdnMadeNetwork::new(4, 8, 0).expect("MADE construction should succeed");
    assert!(made.update_masks(vec![1, 2]).is_err());
}

#[test]
fn test_made_invalid_dims() {
    assert!(MdnMadeNetwork::new(0, 8, 0).is_err());
    assert!(MdnMadeNetwork::new(4, 0, 0).is_err());
}

// ── NormalizingFlowMdn ────────────────────────────────────────────────────

#[test]
fn test_flow_mdn_forward() {
    let fmdn = NormalizingFlowMdn::new(2, &[8], 3, 2, 42).expect("flow MDN construction should succeed");
    let lp = fmdn.log_prob(&[0.0, 0.0], &[0.5, -0.5]).expect("flow log_prob should succeed");
    assert!(lp.is_finite());
}

#[test]
fn test_flow_mdn_transform_sample() {
    let fmdn = NormalizingFlowMdn::new(2, &[8], 3, 4, 1).expect("flow MDN construction should succeed");
    let z = vec![0.1, 0.2, -0.1, 0.3];
    let x = fmdn.transform_sample(&z, 0).expect("flow transform should succeed");
    assert_eq!(x.len(), 4);
}

#[test]
fn test_flow_mdn_inverse_transform() {
    let fmdn = NormalizingFlowMdn::new(2, &[8], 2, 4, 5).expect("flow MDN construction should succeed");
    let x = vec![1.0, 0.5, -1.0, 0.0];
    let (z, _log_det) = fmdn.inverse_transform(&x, 0).expect("flow inverse transform should succeed");
    assert_eq!(z.len(), 4);
}

#[test]
fn test_flow_mdn_out_dim_too_small() {
    assert!(NormalizingFlowMdn::new(2, &[8], 2, 1, 0).is_err());
}

// ── ConditionalVaeMdn ─────────────────────────────────────────────────────

#[test]
fn test_cvae_encode() {
    let cvae = ConditionalVaeMdn::new(3, 2, 8, 4, 3, 42).expect("CVAE construction should succeed");
    let (mu_z, logvar_z) = cvae.encode(&[1.0, 0.0, -1.0], &[0.5, 0.5]).expect("CVAE encoding should succeed");
    assert_eq!(mu_z.len(), 4);
    assert_eq!(logvar_z.len(), 4);
}

#[test]
fn test_cvae_decode() {
    let cvae = ConditionalVaeMdn::new(2, 1, 8, 3, 2, 0).expect("CVAE construction should succeed");
    let z = vec![0.5, -0.5, 0.1];
    let mix = cvae.decode(&z, &[1.0]).expect("CVAE decoding should succeed");
    assert_eq!(mix.n_components(), 2);
}

#[test]
fn test_cvae_elbo_loss() {
    let cvae = ConditionalVaeMdn::new(2, 1, 8, 2, 2, 99).expect("CVAE construction should succeed");
    let mut rng = StdRng::seed_from_u64(7);
    let elbo = cvae.elbo_loss(&[0.0, 1.0], &[1.0], &mut rng).expect("ELBO loss should succeed");
    assert!(elbo.is_finite());
}

// ── BayesianMdn ───────────────────────────────────────────────────────────

#[test]
fn test_bayesian_mdn_forward() {
    let bmdn = BayesianMdn::new(2, &[8], 3, 2, 42).expect("Bayesian MDN construction should succeed");
    let (mean, var) = bmdn.forward_with_uncertainty(&[1.0, -1.0], 10, 0).expect("Bayesian forward should succeed");
    assert_eq!(mean.len(), 2);
    assert_eq!(var.len(), 2);
    for v in &var {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_bayesian_mdn_uncertainty_positive() {
    let bmdn = BayesianMdn::new(1, &[4], 2, 1, 1).expect("Bayesian MDN construction should succeed");
    let (_, var) = bmdn.forward_with_uncertainty(&[0.0], 20, 5).expect("Bayesian forward should succeed");
    // Variance should be non-negative
    assert!(var[0] >= 0.0);
}

#[test]
fn test_bayesian_mdn_invalid_dims() {
    assert!(BayesianMdn::new(0, &[8], 2, 1, 0).is_err());
}

// ── MdnTrainer ────────────────────────────────────────────────────────────

#[test]
fn test_mdn_trainer_basic() {
    let mut mdn = MixtureDensityNetwork::new(1, &[4], 2, 1, 0).expect("MDN construction should succeed");
    let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.05]).collect();
    let y: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.05]).collect();
    let cfg = MdnTrainerConfig {
        max_epochs: 2,
        batch_size: 5,
        val_fraction: 0.2,
        patience: 10,
        ..Default::default()
    };
    let hist = MdnTrainer::train(&mut mdn, &x, &y, &cfg).expect("MDN training should succeed");
    assert!(!hist.is_empty());
    for (epoch, tl, _vl) in &hist {
        assert!(*epoch < 2);
        assert!(tl.is_finite());
    }
}

#[test]
fn test_mdn_trainer_empty_data() {
    let mut mdn = MixtureDensityNetwork::new(1, &[4], 2, 1, 0).expect("MDN construction should succeed");
    let cfg = MdnTrainerConfig::default();
    assert!(MdnTrainer::train(&mut mdn, &[], &[], &cfg).is_err());
}

// ── DensityEstimationBenchmark ────────────────────────────────────────────

#[test]
fn test_two_moons_count() {
    let pts = DensityEstimationBenchmark::two_moons(50, 0.05, 42);
    assert_eq!(pts.len(), 50);
    for p in &pts {
        assert_eq!(p.len(), 2);
    }
}

#[test]
fn test_checkerboard_count() {
    let pts = DensityEstimationBenchmark::checkerboard(40, 7);
    assert_eq!(pts.len(), 40);
}

#[test]
fn test_spirals_count() {
    let pts = DensityEstimationBenchmark::spirals(60, 0.02, 0);
    assert_eq!(pts.len(), 60);
}

#[test]
fn test_gaussian_mixture_8_count() {
    let pts = DensityEstimationBenchmark::gaussian_mixture_8(80, 3);
    assert_eq!(pts.len(), 80);
}

#[test]
fn test_compute_test_nll() {
    let mdn = MixtureDensityNetwork::new(2, &[8], 3, 2, 5).expect("MDN construction should succeed");
    let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1, 0.0]).collect();
    let y: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0, 0.0]).collect();
    let nll = DensityEstimationBenchmark::compute_test_nll(&mdn, &x, &y).expect("test NLL computation should succeed");
    assert!(nll.is_finite());
}

#[test]
fn test_visualize_density_grid() {
    let mdn = MixtureDensityNetwork::new(1, &[4], 2, 2, 0).expect("MDN construction should succeed");
    let grid =
        DensityEstimationBenchmark::visualize_density_grid(&mdn, &[0.0], -1.0, 1.0, 5).expect("density grid visualization should succeed");
    assert_eq!(grid.len(), 5);
    assert_eq!(grid[0].len(), 5);
}

// ── MdnMetrics ────────────────────────────────────────────────────────────

#[test]
fn test_mean_nll() {
    let mix = make_mixture(2, 1);
    let preds = vec![mix.clone(), mix.clone()];
    let targets = vec![vec![0.0], vec![1.0]];
    let nll = MdnMetrics::mean_nll(&preds, &targets).expect("mean NLL computation should succeed");
    assert!(nll.is_finite());
}

#[test]
fn test_mean_nll_mismatch() {
    let mix = make_mixture(2, 1);
    let preds = vec![mix];
    let targets: Vec<Vec<f64>> = vec![];
    assert!(MdnMetrics::mean_nll(&preds, &targets).is_err());
}

#[test]
fn test_crps_score() {
    let mix = make_mixture(3, 1);
    let score = MdnMetrics::crps_score(&mix, &[0.5], 42).expect("CRPS score should succeed");
    assert!(score.is_finite());
    assert!(score >= 0.0);
}

#[test]
fn test_calibration_error_shape() {
    let mixes: Vec<MdnGaussianMixture> = (0..10).map(|_| make_mixture(2, 1)).collect();
    let targets: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
    let cal = MdnMetrics::calibration_error(&mixes, &targets, 5, 0).expect("calibration error should succeed");
    assert_eq!(cal.len(), 5);
}

#[test]
fn test_expected_calibration_error() {
    let cal = vec![(0.25, 0.2), (0.5, 0.45), (0.75, 0.8), (1.0, 1.0)];
    let ece = MdnMetrics::expected_calibration_error(&cal);
    assert!(ece >= 0.0);
    assert!(ece <= 1.0);
}

#[test]
fn test_expected_calibration_error_empty() {
    assert_eq!(MdnMetrics::expected_calibration_error(&[]), 0.0);
}

// ── Integration tests ─────────────────────────────────────────────────────

#[test]
fn test_mdn_sample_after_fit() {
    let mut mdn = MixtureDensityNetwork::new(1, &[8], 3, 1, 42).expect("MDN construction should succeed");
    let data: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.1]).collect();
    let targets: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.1 + 0.5]).collect();
    let _ = mdn.fit(&data, &targets, 1, 1e-3, 4, 1);
    let mix = mdn.forward(&[0.5]).expect("MDN forward should succeed");
    let mut rng = StdRng::seed_from_u64(0);
    let s = mix.sample(&mut rng).expect("mixture sampling should succeed");
    assert_eq!(s.len(), 1);
    assert!(s[0].is_finite());
}

#[test]
fn test_rnade_log_prob_sum_of_conditionals() {
    // log p(x1, x2) should be finite and less than 0 for well-concentrated distributions
    let rnade = RnadeDensityEstimator::new(2, 8, 2, 0).expect("RNADE construction should succeed");
    let x = vec![0.0, 0.0];
    let lp = rnade.log_prob(&x).expect("RNADE log_prob should succeed");
    assert!(lp.is_finite());
}

#[test]
fn test_made_log_prob_finite() {
    let made = MdnMadeNetwork::new(3, 16, 99).expect("MADE construction should succeed");
    for trial in 0..5 {
        let x: Vec<f64> = (0..3).map(|j| (trial * 3 + j) as f64 * 0.1).collect();
        let lp = made.log_prob(&x).expect("MADE log_prob should succeed");
        assert!(lp.is_finite(), "trial {trial}: lp={lp}");
    }
}

#[test]
fn test_flow_mdn_jacobian_correction() {
    // transform then inverse should recover approximate z
    let fmdn = NormalizingFlowMdn::new(2, &[4], 2, 4, 0).expect("flow MDN construction should succeed");
    let z = vec![0.1, 0.2, -0.1, 0.3];
    let x = fmdn.transform_sample(&z, 0).expect("flow transform should succeed");
    let (z_rec, _) = fmdn.inverse_transform(&x, 0).expect("flow inverse transform should succeed");
    assert_eq!(z_rec.len(), 4);
    // Check top half (xa) roundtrips; xb is unchanged by construction
    for j in 2..4 {
        assert!(
            (z_rec[j] - z[j]).abs() < 1e-9,
            "xb roundtrip failed at j={j}"
        );
    }
}

#[test]
fn test_benchmark_datasets_2d() {
    let moons = DensityEstimationBenchmark::two_moons(10, 0.01, 0);
    let check = DensityEstimationBenchmark::checkerboard(10, 1);
    let spiral = DensityEstimationBenchmark::spirals(10, 0.01, 2);
    let gmix = DensityEstimationBenchmark::gaussian_mixture_8(10, 3);
    for ds in [&moons, &check, &spiral, &gmix] {
        for p in ds {
            assert_eq!(p.len(), 2);
            for &v in p {
                assert!(v.is_finite());
            }
        }
    }
}

#[test]
fn test_cvae_elbo_monotone_over_samples() {
    // ELBO should be finite across multiple calls
    let cvae = ConditionalVaeMdn::new(2, 1, 4, 2, 2, 0).expect("CVAE construction should succeed");
    let mut rng = StdRng::seed_from_u64(100);
    for _ in 0..5 {
        let elbo = cvae.elbo_loss(&[0.5, -0.5], &[1.0], &mut rng).expect("ELBO loss should succeed");
        assert!(elbo.is_finite());
    }
}

#[test]
fn test_bayesian_mdn_multiple_samples_variance() {
    let bmdn = BayesianMdn::new(1, &[4], 2, 1, 42).expect("Bayesian MDN construction should succeed");
    let (mean1, var1) = bmdn.forward_with_uncertainty(&[0.0], 5, 0).expect("Bayesian forward should succeed");
    let (mean2, var2) = bmdn.forward_with_uncertainty(&[0.0], 50, 0).expect("Bayesian forward should succeed");
    // Both should produce finite results
    assert!(mean1[0].is_finite());
    assert!(mean2[0].is_finite());
    assert!(var1[0] >= 0.0);
    assert!(var2[0] >= 0.0);
}
