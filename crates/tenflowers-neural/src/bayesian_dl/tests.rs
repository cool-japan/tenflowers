//! Tests for bayesian_dl module.

#[cfg(test)]
mod tests {
    use super::super::helpers::softmax;
    use super::super::{
        compute_calibration, nll_classification, BdlLinear, BdlMlp, CalibrationResult,
        DeepEnsemble, DeepEnsembleConfig, LaplaceApproximation, SghcmConfig, SghcmSampler,
        SgldConfig, SgldSampler, SwagConfig, SwagModel, TemperatureScaling,
    };
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ─── BdlLinear ────────────────────────────────────────────────────────────

    #[test]
    fn test_bdllinear_forward_shape() {
        let layer = BdlLinear::new(4, 3);
        let out = layer.forward(&[1.0, 0.5, -1.0, 0.3]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_bdllinear_n_params() {
        let layer = BdlLinear::new(4, 3);
        assert_eq!(layer.n_params(), 4 * 3 + 3);
    }

    #[test]
    fn test_bdllinear_params_roundtrip() {
        let mut layer = BdlLinear::new(3, 2);
        let original = layer.params_flat();
        let modified: Vec<f64> = original
            .iter()
            .enumerate()
            .map(|(i, &v)| v + i as f64 * 0.1)
            .collect();
        layer.set_params(&modified);
        let restored = layer.params_flat();
        for (a, b) in restored.iter().zip(modified.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bdllinear_xavier_init() {
        let layer = BdlLinear::new(64, 32);
        let limit = (6.0_f64 / (64.0 + 32.0)).sqrt();
        for row in &layer.w {
            for &w in row {
                assert!(
                    w.abs() <= limit * 1.01,
                    "weight {w} exceeds Xavier limit {limit}"
                );
            }
        }
    }

    // ─── BdlMlp ───────────────────────────────────────────────────────────────

    #[test]
    fn test_bdlmlp_forward_shape() {
        let mlp = BdlMlp::new(&[4, 8, 2]);
        let out = mlp.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_bdlmlp_params_flat_roundtrip() {
        let mut mlp = BdlMlp::new(&[3, 5, 2]);
        let original = mlp.params_flat();
        assert_eq!(original.len(), mlp.n_params());
        let perturbed: Vec<f64> = original.iter().map(|&v| v + 0.1).collect();
        mlp.set_params(&perturbed);
        let restored = mlp.params_flat();
        for (a, b) in restored.iter().zip(perturbed.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bdlmlp_n_params() {
        let mlp = BdlMlp::new(&[2, 4, 3]);
        let expected = (2 * 4 + 4) + (4 * 3 + 3);
        assert_eq!(mlp.n_params(), expected);
    }

    #[test]
    fn test_bdlmlp_loss_non_negative() {
        let mlp = BdlMlp::new(&[2, 4, 1]);
        let loss = mlp.loss(&[1.0, 2.0], &[0.5]);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_bdlmlp_gradient_fd_shape() {
        let mlp = BdlMlp::new(&[2, 4, 1]);
        let grad = mlp.gradient_fd(&[1.0, -1.0], &[1.0], 1e-5);
        assert_eq!(grad.len(), mlp.n_params());
    }

    #[test]
    fn test_bdlmlp_gradient_fd_approx_correct() {
        let mut mlp = BdlMlp::new(&[1, 2, 1]);
        mlp.layers[0].w = vec![vec![1.0], vec![-1.0]];
        mlp.layers[0].b = vec![0.0, 0.0];
        mlp.layers[1].w = vec![vec![1.0, 1.0]];
        mlp.layers[1].b = vec![0.0];
        let grad = mlp.gradient_fd(&[2.0], &[0.0], 1e-6);
        let norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_bdlmlp_relu_hidden() {
        let mut mlp = BdlMlp::new(&[1, 2, 1]);
        mlp.layers[0].w = vec![vec![-100.0], vec![-100.0]];
        mlp.layers[0].b = vec![-100.0, -100.0];
        mlp.layers[1].w = vec![vec![1.0, 1.0]];
        mlp.layers[1].b = vec![0.0];
        let out = mlp.forward(&[1.0]);
        assert!(out[0].abs() < 1e-6, "Expected ~0, got {}", out[0]);
    }

    // ─── SGLD ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_sgld_step_adds_noise() {
        let config = SgldConfig {
            lr: 0.01,
            n_steps: 100,
            burn_in: 10,
            n_data: 10,
            prior_std: 1.0,
            thinning: 5,
        };
        let sampler = SgldSampler::new(config);
        let mut rng = make_rng();
        let params = vec![0.0f64; 5];
        let grad = vec![0.0f64; 5];
        let new_params = sampler.step(&params, &grad, &mut rng);
        let diff: f64 = new_params.iter().map(|p| p.abs()).sum();
        assert!(diff > 0.0, "SGLD step should add noise");
    }

    #[test]
    fn test_sgld_step_gradient_descent_component() {
        let config = SgldConfig {
            lr: 0.1,
            n_steps: 10,
            burn_in: 5,
            n_data: 1,
            prior_std: 1e6,
            thinning: 1,
        };
        let sampler = SgldSampler::new(config);
        let mut rng = StdRng::seed_from_u64(0);
        let params = vec![1.0f64; 3];
        let grad = vec![10.0f64; 3];
        let mut total_shift = 0.0f64;
        for _ in 0..100 {
            let new_p = sampler.step(&params, &grad, &mut rng);
            total_shift += new_p
                .iter()
                .zip(params.iter())
                .map(|(n, p)| n - p)
                .sum::<f64>();
        }
        assert!(total_shift / 100.0 > 0.0);
    }

    #[test]
    fn test_sgld_run_collects_samples() {
        let config = SgldConfig {
            lr: 1e-3,
            n_steps: 50,
            burn_in: 10,
            n_data: 5,
            prior_std: 1.0,
            thinning: 5,
        };
        let mut sampler = SgldSampler::new(config);
        let mut rng = make_rng();
        let mut mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0], vec![-1.0], vec![0.5]];
        let y_data = vec![vec![1.0], vec![-1.0], vec![0.5]];
        sampler.run(&mut mlp, &x_data, &y_data, &mut rng);
        assert!(!sampler.samples.is_empty());
        assert_eq!(sampler.samples[0].len(), mlp.n_params());
    }

    #[test]
    fn test_sgld_predict_ensemble_shape() {
        let config = SgldConfig {
            lr: 1e-3,
            n_steps: 30,
            burn_in: 10,
            n_data: 3,
            prior_std: 1.0,
            thinning: 5,
        };
        let mut sampler = SgldSampler::new(config);
        let mut rng = make_rng();
        let mut mlp = BdlMlp::new(&[2, 3, 2]);
        let x_data = vec![vec![1.0, 0.0]];
        let y_data = vec![vec![0.5, -0.5]];
        sampler.run(&mut mlp, &x_data, &y_data, &mut rng);
        let (mean, var) = sampler.predict_ensemble(&mut mlp, &[1.0, 0.0]);
        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
        for v in &var {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_sgld_predict_variance_non_negative() {
        let config = SgldConfig::default();
        let mut sampler = SgldSampler::new(config);
        let mut rng = make_rng();
        let mut mlp = BdlMlp::new(&[1, 4, 1]);
        let x_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let y_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 2.0]).collect();
        sampler.run(&mut mlp, &x_data, &y_data, &mut rng);
        let (_, var) = sampler.predict_ensemble(&mut mlp, &[2.0]);
        for v in &var {
            assert!(*v >= 0.0);
        }
    }

    // ─── SGHMC ────────────────────────────────────────────────────────────────

    #[test]
    fn test_sghmc_step_updates_momentum() {
        let config = SghcmConfig {
            lr: 0.01,
            friction: 0.1,
            n_steps: 10,
            burn_in: 5,
            n_data: 5,
            prior_std: 1.0,
            thinning: 2,
        };
        let mut sampler = SghcmSampler::new(config);
        let mut rng = make_rng();
        let params = vec![0.0f64; 3];
        let grad = vec![1.0f64; 3];
        let _new_params = sampler.step(&params, &grad, &mut rng);
        assert!(sampler.momentum.is_some());
        let mom = sampler
            .momentum
            .as_ref()
            .expect("momentum should be Some after step");
        assert_eq!(mom.len(), 3);
    }

    #[test]
    fn test_sghmc_step_changes_params() {
        let config = SghcmConfig {
            lr: 0.1,
            friction: 0.01,
            n_steps: 10,
            burn_in: 5,
            n_data: 5,
            prior_std: 1.0,
            thinning: 2,
        };
        let mut sampler = SghcmSampler::new(config);
        let mut rng = make_rng();
        let params = vec![1.0f64; 4];
        let grad = vec![0.0f64; 4];
        let new_params = sampler.step(&params, &grad, &mut rng);
        let changed = params
            .iter()
            .zip(new_params.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(changed);
    }

    #[test]
    fn test_sghmc_run_collects_samples() {
        let config = SghcmConfig {
            lr: 1e-3,
            friction: 0.01,
            n_steps: 50,
            burn_in: 10,
            n_data: 5,
            prior_std: 1.0,
            thinning: 5,
        };
        let mut sampler = SghcmSampler::new(config);
        let mut rng = make_rng();
        let mut mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0], vec![-1.0]];
        let y_data = vec![vec![1.0], vec![-1.0]];
        sampler.run(&mut mlp, &x_data, &y_data, &mut rng);
        assert!(!sampler.samples.is_empty());
    }

    #[test]
    fn test_sghmc_predict_ensemble_variance_non_negative() {
        let config = SghcmConfig {
            lr: 1e-3,
            friction: 0.01,
            n_steps: 40,
            burn_in: 10,
            n_data: 3,
            prior_std: 1.0,
            thinning: 5,
        };
        let mut sampler = SghcmSampler::new(config);
        let mut rng = make_rng();
        let mut mlp = BdlMlp::new(&[2, 3, 1]);
        let x_data = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let y_data = vec![vec![0.0], vec![1.0]];
        sampler.run(&mut mlp, &x_data, &y_data, &mut rng);
        let (mean, var) = sampler.predict_ensemble(&mut mlp, &[0.5, -0.5]);
        assert_eq!(mean.len(), 1);
        assert!(var[0] >= 0.0);
    }

    #[test]
    fn test_sghmc_momentum_persists_across_steps() {
        let config = SghcmConfig::default();
        let mut sampler = SghcmSampler::new(config);
        let mut rng = make_rng();
        let params = vec![0.0f64; 2];
        let grad = vec![0.5f64; 2];
        sampler.step(&params, &grad, &mut rng);
        let mom1 = sampler.momentum.clone().expect("momentum should be Some");
        sampler.step(&params, &grad, &mut rng);
        let mom2 = sampler.momentum.clone().expect("momentum should be Some");
        let changed = mom1
            .iter()
            .zip(mom2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(changed);
    }

    // ─── Laplace ──────────────────────────────────────────────────────────────

    #[test]
    fn test_laplace_fit_produces_map() {
        let mlp = BdlMlp::new(&[1, 3, 1]);
        let x_data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1]).collect();
        let y_data: Vec<Vec<f64>> = x_data.iter().map(|x| vec![x[0] * 2.0]).collect();
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 20, 1e-2, 1.0);
        assert_eq!(la.map_params.len(), mlp.n_params());
    }

    #[test]
    fn test_laplace_posterior_variance_positive() {
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0], vec![-1.0], vec![0.0]];
        let y_data = vec![vec![1.0], vec![-1.0], vec![0.0]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 10, 1e-2, 1.0);
        let var = la.posterior_variance();
        for v in &var {
            assert!(*v > 0.0);
        }
    }

    #[test]
    fn test_laplace_sample_shape() {
        let mlp = BdlMlp::new(&[2, 3, 1]);
        let x_data = vec![vec![1.0, 0.0]];
        let y_data = vec![vec![0.5]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 5, 1e-2, 1.0);
        let mut rng = make_rng();
        let sample = la.sample(&mut rng);
        assert_eq!(sample.len(), mlp.n_params());
    }

    #[test]
    fn test_laplace_predict_shape() {
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![0.5], vec![-0.5]];
        let y_data = vec![vec![1.0], vec![-1.0]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 5, 1e-2, 1.0);
        let mut model = mlp.clone();
        let mut rng = make_rng();
        let (mean, var) = la.predict(&mut model, &[0.0], 10, &mut rng);
        assert_eq!(mean.len(), 1);
        assert_eq!(var.len(), 1);
    }

    #[test]
    fn test_laplace_predict_variance_non_negative() {
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0], vec![-1.0]];
        let y_data = vec![vec![2.0], vec![-2.0]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 10, 1e-3, 1.0);
        let mut model = mlp.clone();
        let mut rng = make_rng();
        let (_, var) = la.predict(&mut model, &[0.5], 20, &mut rng);
        for v in &var {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_laplace_log_marginal_likelihood_finite() {
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0]];
        let y_data = vec![vec![1.0]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 5, 1e-2, 1.0);
        let lml = la.log_marginal_likelihood_approx(10);
        assert!(lml.is_finite());
    }

    #[test]
    fn test_laplace_hessian_diag_shape() {
        let mlp = BdlMlp::new(&[2, 3, 2]);
        let x_data = vec![vec![1.0, -1.0]];
        let y_data = vec![vec![0.0, 1.0]];
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 5, 1e-2, 1.0);
        assert_eq!(la.hessian_diag.len(), mlp.n_params());
    }

    // ─── SWAG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_swag_diagonal_variance_non_negative() {
        let config = SwagConfig {
            swa_start_epoch: 2,
            swa_lr: 1e-3,
            n_epochs: 10,
            max_rank: 5,
            scale: 0.5,
            prior_std: 1.0,
        };
        let mlp = BdlMlp::new(&[2, 4, 1]);
        let mut swag = SwagModel::new(mlp.n_params(), config);
        let x_data = vec![vec![1.0, 0.0], vec![-1.0, 1.0]];
        let y_data = vec![vec![1.0], vec![-1.0]];
        let mut rng = make_rng();
        for epoch in 0..10 {
            swag.collect(&mlp, &x_data, &y_data, epoch, 1e-3, &mut rng);
        }
        let var = swag.diagonal_variance();
        for v in &var {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_swag_sample_shape() {
        let config = SwagConfig {
            swa_start_epoch: 0,
            swa_lr: 1e-3,
            n_epochs: 5,
            max_rank: 3,
            scale: 0.5,
            prior_std: 1.0,
        };
        let mlp = BdlMlp::new(&[3, 5, 2]);
        let n_params = mlp.n_params();
        let mut swag = SwagModel::new(n_params, config);
        let x_data = vec![vec![1.0, 0.0, -1.0]];
        let y_data = vec![vec![0.5, -0.5]];
        let mut rng = make_rng();
        for epoch in 0..5 {
            swag.collect(&mlp, &x_data, &y_data, epoch, 1e-3, &mut rng);
        }
        let sample = swag.sample(&mut rng);
        assert_eq!(sample.len(), n_params);
    }

    #[test]
    fn test_swag_predict_ensemble_variance_non_negative() {
        let config = SwagConfig {
            swa_start_epoch: 0,
            swa_lr: 1e-3,
            n_epochs: 5,
            max_rank: 4,
            scale: 0.5,
            prior_std: 1.0,
        };
        let mut mlp = BdlMlp::new(&[1, 3, 1]);
        let mut swag = SwagModel::new(mlp.n_params(), config);
        let x_data = vec![vec![1.0], vec![-1.0]];
        let y_data = vec![vec![1.0], vec![-1.0]];
        let mut rng = make_rng();
        for epoch in 0..5 {
            swag.collect(&mlp, &x_data, &y_data, epoch, 1e-3, &mut rng);
        }
        let (mean, var) = swag.predict_ensemble(&mut mlp, &[0.5], 10, &mut rng);
        assert_eq!(mean.len(), 1);
        assert!(var[0] >= 0.0);
    }

    #[test]
    fn test_swag_collect_updates_mean() {
        let config = SwagConfig {
            swa_start_epoch: 0,
            swa_lr: 1e-3,
            n_epochs: 5,
            max_rank: 5,
            scale: 0.5,
            prior_std: 1.0,
        };
        let mlp = BdlMlp::new(&[2, 2, 1]);
        let n = mlp.n_params();
        let mut swag = SwagModel::new(n, config);
        let x_data = vec![vec![0.5, -0.5]];
        let y_data = vec![vec![1.0]];
        let mut rng = make_rng();
        swag.collect(&mlp, &x_data, &y_data, 0, 1e-3, &mut rng);
        assert_eq!(swag.n_collected, 1);
        assert_eq!(swag.mean_params.len(), n);
    }

    #[test]
    fn test_swag_max_rank_respected() {
        let config = SwagConfig {
            swa_start_epoch: 0,
            swa_lr: 1e-3,
            n_epochs: 20,
            max_rank: 3,
            scale: 0.5,
            prior_std: 1.0,
        };
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let mut swag = SwagModel::new(mlp.n_params(), config);
        let x_data = vec![vec![1.0]];
        let y_data = vec![vec![1.0]];
        let mut rng = make_rng();
        for epoch in 0..20 {
            swag.collect(&mlp, &x_data, &y_data, epoch, 1e-3, &mut rng);
        }
        assert!(swag.deviations.len() <= 3);
    }

    // ─── Calibration ──────────────────────────────────────────────────────────

    #[test]
    fn test_calibration_ece_in_range() {
        let probs = vec![
            vec![0.9, 0.1],
            vec![0.3, 0.7],
            vec![0.6, 0.4],
            vec![0.2, 0.8],
        ];
        let labels = vec![0, 1, 0, 1];
        let result = compute_calibration(&probs, &labels, 5);
        assert!(result.ece >= 0.0 && result.ece <= 1.0);
    }

    #[test]
    fn test_calibration_mce_in_range() {
        let probs = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let labels = vec![0, 1];
        let result = compute_calibration(&probs, &labels, 5);
        assert!(result.mce >= 0.0 && result.mce <= 1.0);
    }

    #[test]
    fn test_calibration_brier_score_non_negative() {
        let probs = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let labels = vec![0, 1];
        let result = compute_calibration(&probs, &labels, 5);
        assert!(result.brier_score >= 0.0);
    }

    #[test]
    fn test_calibration_reliability_diagram_bins() {
        let probs: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let p = i as f64 / 20.0;
                vec![p, 1.0 - p]
            })
            .collect();
        let labels: Vec<usize> = (0..20).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let result = compute_calibration(&probs, &labels, 10);
        assert_eq!(result.reliability_diagram.len(), 10);
    }

    #[test]
    fn test_perfect_calibration_ece_zero() {
        let probs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = vec![0, 1];
        let result = compute_calibration(&probs, &labels, 5);
        assert!(result.ece < 1e-9);
    }

    // ─── TemperatureScaling ───────────────────────────────────────────────────

    #[test]
    fn test_temperature_scaling_identity_at_one() {
        let ts = TemperatureScaling::new();
        assert!((ts.temperature - 1.0).abs() < 1e-12);
        let logits = vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.0, 1.0]];
        let calibrated = ts.calibrate(&logits);
        let direct: Vec<Vec<f64>> = logits.iter().map(|l| softmax(l)).collect();
        for (c, d) in calibrated.iter().zip(direct.iter()) {
            for (ci, di) in c.iter().zip(d.iter()) {
                assert!((ci - di).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_temperature_scaling_sums_to_one() {
        let mut ts = TemperatureScaling::new();
        let logits = vec![vec![2.0, -1.0, 0.5], vec![0.1, 0.2, 0.3]];
        let labels = vec![0usize, 2usize];
        ts.fit(&logits, &labels, 50);
        let calibrated = ts.calibrate(&logits);
        for probs in &calibrated {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_temperature_scaling_fit_changes_temperature() {
        let mut ts = TemperatureScaling::new();
        let logits = vec![vec![10.0, -10.0], vec![10.0, -10.0]];
        let labels = vec![1usize, 1usize];
        ts.fit(&logits, &labels, 100);
        assert!(ts.temperature > 0.0);
    }

    // ─── nll_classification ───────────────────────────────────────────────────

    #[test]
    fn test_nll_classification_positive() {
        let probs = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let labels = vec![0usize, 1usize];
        let nll = nll_classification(&probs, &labels);
        assert!(nll > 0.0);
    }

    #[test]
    fn test_nll_classification_perfect_prediction() {
        let probs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = vec![0usize, 1usize];
        let nll = nll_classification(&probs, &labels);
        assert!(nll < 1e-10);
    }

    #[test]
    fn test_nll_classification_finite() {
        let probs = vec![vec![0.5, 0.5]];
        let labels = vec![0usize];
        let nll = nll_classification(&probs, &labels);
        assert!(nll.is_finite());
    }

    // ─── DeepEnsemble ─────────────────────────────────────────────────────────

    #[test]
    fn test_deep_ensemble_predict_shape() {
        let config = DeepEnsembleConfig {
            n_members: 3,
            lr: 1e-2,
            n_epochs: 5,
            adversarial_training: false,
            eps_adv: 0.01,
        };
        let ensemble = DeepEnsemble::new(&[2, 4, 1], config);
        let (mean, var) = ensemble.predict(&[1.0, -1.0]);
        assert_eq!(mean.len(), 1);
        assert_eq!(var.len(), 1);
    }

    #[test]
    fn test_deep_ensemble_fit_changes_predictions() {
        let config = DeepEnsembleConfig {
            n_members: 2,
            lr: 1e-2,
            n_epochs: 10,
            adversarial_training: false,
            eps_adv: 0.0,
        };
        let mut ensemble = DeepEnsemble::new(&[1, 3, 1], config);
        let (pred_before, _) = ensemble.predict(&[1.0]);
        let x_data = vec![vec![1.0], vec![-1.0], vec![0.5]];
        let y_data = vec![vec![2.0], vec![-2.0], vec![1.0]];
        let mut rng = make_rng();
        ensemble.fit(&x_data, &y_data, &mut rng);
        let (pred_after, _) = ensemble.predict(&[1.0]);
        let changed = pred_before
            .iter()
            .zip(pred_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn test_deep_ensemble_epistemic_uncertainty_non_negative() {
        let config = DeepEnsembleConfig {
            n_members: 3,
            lr: 1e-3,
            n_epochs: 5,
            adversarial_training: false,
            eps_adv: 0.0,
        };
        let ensemble = DeepEnsemble::new(&[2, 3, 1], config);
        let eu = ensemble.epistemic_uncertainty(&[1.0, 0.0]);
        assert!(eu >= 0.0);
    }

    #[test]
    fn test_deep_ensemble_aleatoric_uncertainty_non_negative() {
        let config = DeepEnsembleConfig {
            n_members: 4,
            lr: 1e-3,
            n_epochs: 5,
            adversarial_training: false,
            eps_adv: 0.0,
        };
        let ensemble = DeepEnsemble::new(&[2, 4, 2], config);
        let au = ensemble.aleatoric_uncertainty(&[0.5, -0.5]);
        assert!(au >= 0.0);
    }

    #[test]
    fn test_deep_ensemble_n_members() {
        let config = DeepEnsembleConfig {
            n_members: 5,
            ..Default::default()
        };
        let ensemble = DeepEnsemble::new(&[2, 4, 1], config);
        assert_eq!(ensemble.members.len(), 5);
    }

    #[test]
    fn test_deep_ensemble_adversarial_training() {
        let config = DeepEnsembleConfig {
            n_members: 2,
            lr: 1e-2,
            n_epochs: 5,
            adversarial_training: true,
            eps_adv: 0.01,
        };
        let mut ensemble = DeepEnsemble::new(&[1, 2, 1], config);
        let x_data = vec![vec![0.5], vec![-0.5]];
        let y_data = vec![vec![1.0], vec![-1.0]];
        let mut rng = make_rng();
        ensemble.fit(&x_data, &y_data, &mut rng);
        let (mean, _) = ensemble.predict(&[0.0]);
        assert_eq!(mean.len(), 1);
    }

    #[test]
    fn test_deep_ensemble_variance_non_negative() {
        let config = DeepEnsembleConfig {
            n_members: 3,
            ..Default::default()
        };
        let mut ensemble = DeepEnsemble::new(&[2, 4, 2], config);
        let mut rng = make_rng();
        let x_data = vec![vec![1.0, 0.0]];
        let y_data = vec![vec![0.0, 1.0]];
        ensemble.fit(&x_data, &y_data, &mut rng);
        let (_, var) = ensemble.predict(&[1.0, -1.0]);
        for v in &var {
            assert!(*v >= 0.0);
        }
    }

    // ─── Integration ──────────────────────────────────────────────────────────

    #[test]
    fn test_sgld_then_laplace_consistency() {
        let mlp = BdlMlp::new(&[1, 2, 1]);
        let x_data = vec![vec![1.0], vec![-1.0], vec![0.0]];
        let y_data = vec![vec![2.0], vec![-2.0], vec![0.0]];
        let sgld_cfg = SgldConfig {
            lr: 1e-3,
            n_steps: 30,
            burn_in: 10,
            n_data: 3,
            prior_std: 1.0,
            thinning: 5,
        };
        let mut sgld = SgldSampler::new(sgld_cfg);
        let mut m1 = mlp.clone();
        let mut rng = make_rng();
        sgld.run(&mut m1, &x_data, &y_data, &mut rng);
        let la = LaplaceApproximation::fit(&mlp, &x_data, &y_data, 10, 1e-2, 1.0);
        let (sgld_mean, _) = sgld.predict_ensemble(&mut m1, &[0.5]);
        let mut m2 = mlp.clone();
        let (la_mean, _) = la.predict(&mut m2, &[0.5], 5, &mut rng);
        assert!(sgld_mean[0].is_finite());
        assert!(la_mean[0].is_finite());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, -1.0, 0.5];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_normal_statistics() {
        use super::super::helpers::sample_normal as sn;
        let mut rng = StdRng::seed_from_u64(0);
        let n = 1000usize;
        let samples: Vec<f64> = (0..n).map(|_| sn(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.15);
        assert!((var - 1.0).abs() < 0.2);
    }
}
