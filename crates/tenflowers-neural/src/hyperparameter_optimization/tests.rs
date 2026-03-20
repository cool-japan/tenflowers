use super::*;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// § HyperParameter sampling

#[test]
fn test_continuous_sample_bounds() {
    let hp = HyperParameter::continuous("lr", 0.001, 0.1);
    let mut rng = make_rng(0);
    for _ in 0..50 {
        let v = hp.sample(&mut rng);
        assert!((0.001..=0.1).contains(&v), "out of bounds: {v}");
    }
}

#[test]
fn test_integer_sample_bounds() {
    let hp = HyperParameter::integer("layers", 1, 10);
    let mut rng = make_rng(1);
    for _ in 0..50 {
        let v = hp.sample(&mut rng);
        assert!((1.0..=10.0).contains(&v) && v == v.round(), "bad int: {v}");
    }
}

#[test]
fn test_log_sample_positive() {
    let hp = HyperParameter::log_continuous("lr", 1e-5, 1e-1);
    let mut rng = make_rng(2);
    for _ in 0..50 {
        let v = hp.sample(&mut rng);
        assert!(v > 0.0 && v <= 0.1 + 1e-10, "log sample out: {v}");
    }
}

#[test]
fn test_categorical_sample() {
    let hp = HyperParameter::categorical("opt", vec![0.0, 1.0, 2.0]);
    let mut rng = make_rng(3);
    for _ in 0..50 {
        let v = hp.sample(&mut rng);
        assert!([0.0, 1.0, 2.0].contains(&v), "unexpected cat: {v}");
    }
}

#[test]
fn test_to_unit_continuous_roundtrip() {
    let hp = HyperParameter::continuous("x", 2.0, 8.0);
    for &v in &[2.0, 5.0, 8.0] {
        let u = hp.to_unit(v);
        let back = hp.from_unit(u);
        assert!(
            (v - back).abs() < 1e-9,
            "roundtrip fail: {v} -> {u} -> {back}"
        );
    }
}

#[test]
fn test_to_unit_log_roundtrip() {
    let hp = HyperParameter::log_continuous("lr", 1e-4, 1e-1);
    for &v in &[1e-4, 1e-3, 1e-2, 1e-1] {
        let u = hp.to_unit(v);
        let back = hp.from_unit(u);
        assert!((v - back).abs() / v < 1e-9, "log roundtrip: {v} -> {back}");
    }
}

// § HpSpace

#[test]
fn test_hpspace_sample_ndim() {
    let space = HpSpace::new()
        .add(HyperParameter::continuous("a", 0.0, 1.0))
        .add(HyperParameter::integer("b", 1, 5))
        .add(HyperParameter::log_continuous("c", 1e-3, 1.0));
    let mut rng = make_rng(10);
    let sample = space.sample_random(&mut rng);
    assert_eq!(sample.len(), 3);
}

#[test]
fn test_hpspace_transform_unit_inversion() {
    let space = HpSpace::new()
        .add(HyperParameter::continuous("a", 1.0, 10.0))
        .add(HyperParameter::log_continuous("b", 1e-3, 1e-1));
    let raw = vec![5.0, 1e-2];
    let unit = space.transform_to_unit(&raw);
    let back = space.transform_from_unit(&unit);
    for (orig, recovered) in raw.iter().zip(back.iter()) {
        assert!((orig - recovered).abs() / (orig.abs() + 1e-10) < 1e-7);
    }
}

// § GpHpo

#[test]
fn test_gphpo_predict_before_fit() {
    let gp = GpHpo::new(1.0, 1e-3);
    let (mean, var) = gp.predict(&[0.5]);
    assert_eq!(mean, 0.0);
    assert_eq!(var, 1.0);
}

#[test]
fn test_gphpo_fit_and_predict() {
    let mut gp = GpHpo::new(1.0, 1e-3);
    gp.x_train = vec![vec![0.0], vec![0.5], vec![1.0]];
    gp.y_train = vec![0.0, 0.5, 1.0];
    gp.fit().expect("GP fit should succeed");
    let (mean, _var) = gp.predict(&[0.5]);
    assert!(
        (mean - 0.5).abs() < 0.2,
        "GP mean at 0.5 should be near 0.5, got {mean}"
    );
}

#[test]
fn test_gphpo_variance_positive() {
    let mut gp = GpHpo::new(1.0, 1e-3);
    gp.x_train = vec![vec![0.0], vec![1.0]];
    gp.y_train = vec![0.0, 1.0];
    gp.fit().expect("fit ok");
    let (_, var) = gp.predict(&[0.5]);
    assert!(var > 0.0, "variance should be positive");
}

// § HpoAcqFunction

#[test]
fn test_acq_ei_positive() {
    let acq = HpoAcqFunction::ExpectedImprovement { xi: 0.01 };
    let val = acq.evaluate(1.1, 0.1, 1.0);
    assert!(val > 0.0, "EI should be positive when mean > best");
}

#[test]
fn test_acq_ucb() {
    let acq = HpoAcqFunction::UpperConfidenceBound { kappa: 2.0 };
    let val = acq.evaluate(0.5, 0.3, 0.0);
    assert!((val - (0.5 + 2.0 * 0.3)).abs() < 1e-9);
}

#[test]
fn test_acq_pi_in_01() {
    let acq = HpoAcqFunction::ProbabilityOfImprovement { xi: 0.0 };
    let val = acq.evaluate(1.0, 0.1, 1.0);
    assert!((0.0..=1.0).contains(&val), "PI must be in [0,1]");
}

#[test]
fn test_acq_log_ei_finite() {
    let acq = HpoAcqFunction::LogExpectedImprovement { xi: 0.01 };
    let val = acq.evaluate(1.1, 0.1, 1.0);
    assert!(val.is_finite(), "log-EI should be finite");
}

// § HpoBayesianOptimizer

#[test]
fn test_bayes_suggest_random_warmup() {
    let space = HpSpace::new().add(HyperParameter::continuous("x", 0.0, 1.0));
    let config = HpoBayesianConfig {
        n_initial: 5,
        ..Default::default()
    };
    let opt = HpoBayesianOptimizer::new(space, config);
    let mut rng = make_rng(42);
    let s = opt.suggest(&mut rng);
    assert_eq!(s.len(), 1);
}

#[test]
fn test_bayes_observe_and_best() {
    let space = HpSpace::new().add(HyperParameter::continuous("x", 0.0, 1.0));
    let config = HpoBayesianConfig {
        n_initial: 2,
        ..Default::default()
    };
    let mut opt = HpoBayesianOptimizer::new(space, config);
    opt.observe(&[0.3], 0.5);
    opt.observe(&[0.7], 0.9);
    let (_, best_v) = opt.best().expect("best should exist");
    assert!((best_v - 0.9).abs() < 1e-9);
}

#[test]
fn test_bayes_suggest_after_warmup() {
    let space = HpSpace::new().add(HyperParameter::continuous("x", 0.0, 1.0));
    let config = HpoBayesianConfig {
        n_initial: 2,
        n_candidates: 20,
        ..Default::default()
    };
    let mut opt = HpoBayesianOptimizer::new(space, config);
    opt.observe(&[0.1], 0.1);
    opt.observe(&[0.9], 0.9);
    let mut rng = make_rng(5);
    let s = opt.suggest(&mut rng);
    assert_eq!(s.len(), 1);
    assert!(s[0] >= 0.0 && s[0] <= 1.0);
}

// § HyperBandScheduler

#[test]
fn test_hyperband_plan_schedule_nonempty() {
    let config = HyperBandConfig {
        max_iter: 81.0,
        eta: 3.0,
        min_resource: 1.0,
    };
    let brackets = HyperBandScheduler::plan_schedule(&config);
    assert!(!brackets.is_empty(), "should produce at least one bracket");
}

#[test]
fn test_hyperband_bracket_n_positive() {
    let config = HyperBandConfig::default();
    let brackets = HyperBandScheduler::plan_schedule(&config);
    for b in &brackets {
        assert!(b.n > 0, "bracket n must be positive");
    }
}

#[test]
fn test_hyperband_get_configurations_len() {
    let config = HyperBandConfig::default();
    let scheduler = HyperBandScheduler::new(config);
    let space = HpSpace::new().add(HyperParameter::continuous("x", 0.0, 1.0));
    let mut rng = make_rng(99);
    let b = &scheduler.brackets[0].clone();
    let configs = scheduler.get_configurations(b, &space, &mut rng);
    assert_eq!(configs.len(), b.n);
}

#[test]
fn test_hyperband_promote_top_n() {
    let scores: Vec<(Vec<f64>, f64)> = (0..10).map(|i| (vec![i as f64], i as f64)).collect();
    let kept = HyperBandScheduler::promote(&scores, 3);
    assert_eq!(kept.len(), 3);
    // Best score is 9, so top-3 should include params with scores 9, 8, 7
    assert!(kept.iter().any(|p| p[0] == 9.0));
}

#[test]
fn test_hyperband_successive_halving_rounds() {
    let config = HyperBandConfig::default();
    let scheduler = HyperBandScheduler::new(config);
    let rounds = scheduler.successive_halving_rounds(0);
    assert!(!rounds.is_empty());
    for (n, r) in &rounds {
        assert!(*n > 0);
        assert!(*r > 0.0);
    }
}

// § BOHB

#[test]
fn test_bohb_suggest_random_warmup() {
    let space = HpSpace::new().add(HyperParameter::continuous("lr", 1e-4, 1e-1));
    let bohb_config = BohbConfig {
        n_initial: 5,
        ..Default::default()
    };
    let hb_config = HyperBandConfig::default();
    let opt = BohbOptimizer::new(space, hb_config, bohb_config);
    let mut rng = make_rng(7);
    let s = opt.suggest_bohb(&mut rng);
    assert_eq!(s.len(), 1);
}

#[test]
fn test_bohb_observe_and_suggest() {
    let space = HpSpace::new().add(HyperParameter::continuous("lr", 1e-4, 1e-1));
    let bohb_config = BohbConfig {
        n_initial: 3,
        ..Default::default()
    };
    let hb_config = HyperBandConfig::default();
    let mut opt = BohbOptimizer::new(space, hb_config, bohb_config);
    let mut rng = make_rng(8);
    for i in 0..5 {
        let lr = 1e-4 + i as f64 * 1e-5;
        opt.observe(&[lr], i as f64);
    }
    let s = opt.suggest_bohb(&mut rng);
    assert!(s[0] >= 1e-4 && s[0] <= 1e-1);
}

#[test]
fn test_kde_pdf_nonzero() {
    let samples = vec![0.1, 0.5, 0.9];
    let pdf = KdeSampler::kde_pdf(0.5, &samples, 0.2);
    assert!(pdf > 0.0);
}

// § PBT

#[test]
fn test_pbt_population_init() {
    let space = HpSpace::new()
        .add(HyperParameter::continuous("lr", 0.001, 0.1))
        .add(HyperParameter::integer("bs", 16, 128));
    let config = PbtConfig {
        population_size: 5,
        ..Default::default()
    };
    let mut rng = make_rng(20);
    let pbt = PopulationBasedTraining::new(space, config, &mut rng);
    assert_eq!(pbt.population.members.len(), 5);
}

#[test]
fn test_pbt_step_changes_params() {
    let space = HpSpace::new().add(HyperParameter::continuous("lr", 0.001, 0.1));
    let config = PbtConfig {
        population_size: 4,
        exploit_frac: 0.5,
        explore_noise: 0.1,
    };
    let mut rng = make_rng(21);
    let mut pbt = PopulationBasedTraining::new(space, config, &mut rng);
    let scores = vec![0.1, 0.9, 0.5, 0.3];
    pbt.step(&scores, &mut rng);
    // After step, should still have 4 members
    assert_eq!(pbt.population.members.len(), 4);
}

#[test]
fn test_pbt_best_returns_some() {
    let space = HpSpace::new().add(HyperParameter::continuous("lr", 0.001, 0.1));
    let config = PbtConfig::default();
    let mut rng = make_rng(22);
    let mut pbt = PopulationBasedTraining::new(space, config, &mut rng);
    let scores: Vec<f64> = (0..pbt.population.members.len())
        .map(|i| i as f64 * 0.1)
        .collect();
    pbt.step(&scores, &mut rng);
    assert!(pbt.best().is_some());
}

// § CMA-ES

#[test]
fn test_cma_config_default_lambda() {
    let cfg = CmaHpoConfig::new(5, 0.5);
    assert!(cfg.lambda >= 4, "lambda should be >= 4");
}

#[test]
fn test_cma_sample_dimension() {
    let cfg = CmaHpoConfig::new(3, 0.5);
    let es = EvolutionaryStrategy::new(cfg, vec![0.0, 0.0, 0.0]);
    let mut rng = make_rng(30);
    let samples = es.sample(&mut rng);
    assert!(!samples.is_empty());
    assert_eq!(samples[0].len(), 3);
}

#[test]
fn test_cma_update_mean_changes() {
    let cfg = CmaHpoConfig::new(2, 0.3);
    let lambda = cfg.lambda;
    let mut es = EvolutionaryStrategy::new(cfg, vec![0.0, 0.0]);
    let mut rng = make_rng(31);
    let samples = es.sample(&mut rng);
    let mu = lambda / 2;
    let selected: Vec<Vec<f64>> = samples[..mu.min(samples.len())].to_vec();
    let old_mean = es.state.mean.clone();
    es.update(&selected);
    // Mean should have moved unless selected were all exactly at origin
    let moved = old_mean
        .iter()
        .zip(es.state.mean.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(moved || selected.iter().all(|s| s == &vec![0.0, 0.0]));
}

// § Multi-Objective HPO

#[test]
fn test_pareto_front_basic() {
    let obs = vec![
        MoObservation {
            params: vec![0.0],
            objectives: vec![0.1, 0.9],
        },
        MoObservation {
            params: vec![0.5],
            objectives: vec![0.5, 0.5],
        },
        MoObservation {
            params: vec![1.0],
            objectives: vec![0.9, 0.1],
        },
        MoObservation {
            params: vec![0.3],
            objectives: vec![0.6, 0.7],
        }, // dominated
    ];
    let front = pareto_front(&obs);
    assert!(front.contains(&0), "obs 0 should be on front");
    assert!(front.contains(&1), "obs 1 should be on front");
    assert!(front.contains(&2), "obs 2 should be on front");
    assert!(!front.contains(&3), "obs 3 should be dominated");
}

#[test]
fn test_hypervolume_contribution_positive() {
    let obs = vec![MoObservation {
        params: vec![0.0],
        objectives: vec![1.0, 2.0],
    }];
    let hvc = hypervolume_contribution(&obs, &[3.0, 4.0]);
    assert!(hvc[0] > 0.0, "HVC should be positive");
    assert!(
        (hvc[0] - 4.0).abs() < 1e-9,
        "HVC = (3-1)*(4-2) = 4, got {}",
        hvc[0]
    );
}

#[test]
fn test_nsga2_select_count() {
    let obs: Vec<MoObservation> = (0..8)
        .map(|i| MoObservation {
            params: vec![i as f64],
            objectives: vec![i as f64, (8 - i) as f64],
        })
        .collect();
    let selected = nsga2_select(&obs, 4);
    assert_eq!(selected.len(), 4);
}

// § Early Termination

#[test]
fn test_median_stop_no_stop_few_trials() {
    let ms = MedianStopping::new(3, 5);
    let stopped = ms.should_stop(&[0.1, 0.2, 0.05], &[]);
    assert!(!stopped, "should not stop with fewer than min_trials");
}

#[test]
fn test_median_stop_stops_below_median() {
    let ms = MedianStopping::new(1, 3);
    let all_curves = vec![vec![0.8], vec![0.7], vec![0.9]];
    let stopped = ms.should_stop(&[0.1], &all_curves);
    assert!(stopped, "should stop when below median");
}

#[test]
fn test_successive_halving_budget() {
    let sh = SuccessiveHalving::new(1.0, 81.0, 3.0);
    let b0 = sh.budget_for_round(0);
    let b1 = sh.budget_for_round(1);
    assert!((b0 - 1.0).abs() < 1e-9);
    assert!((b1 - 3.0).abs() < 1e-9);
}

#[test]
fn test_percentile_stop_below() {
    let ps = PercentileStop::new(50.0, 2);
    let history = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let stopped = ps.should_stop_percentile(0.2, 3, &history);
    assert!(stopped, "0.2 is below 50th percentile of history");
}

#[test]
fn test_percentile_stop_not_below_min_steps() {
    let ps = PercentileStop::new(50.0, 5);
    let history = vec![0.5, 0.6, 0.7];
    let stopped = ps.should_stop_percentile(0.1, 2, &history);
    assert!(!stopped, "should not stop before min_steps");
}

// § HpoStudy / HpoLogger

#[test]
fn test_study_best_minimize() {
    let mut study = HpoStudy::new("test", OptDirection::Minimize);
    study.add_trial(HpoTrial::new(
        0,
        vec![0.5],
        vec!["lr".into()],
        0.9,
        TrialStatus::Complete,
    ));
    study.add_trial(HpoTrial::new(
        1,
        vec![0.1],
        vec!["lr".into()],
        0.2,
        TrialStatus::Complete,
    ));
    study.add_trial(HpoTrial::new(
        2,
        vec![0.3],
        vec!["lr".into()],
        0.5,
        TrialStatus::Complete,
    ));
    let best = study.best_trial().expect("best should exist");
    assert!((best.value - 0.2).abs() < 1e-9);
}

#[test]
fn test_study_best_maximize() {
    let mut study = HpoStudy::new("test", OptDirection::Maximize);
    study.add_trial(HpoTrial::new(
        0,
        vec![0.5],
        vec!["lr".into()],
        0.9,
        TrialStatus::Complete,
    ));
    study.add_trial(HpoTrial::new(
        1,
        vec![0.1],
        vec!["lr".into()],
        0.2,
        TrialStatus::Complete,
    ));
    let best = study.best_trial().expect("best should exist");
    assert!((best.value - 0.9).abs() < 1e-9);
}

#[test]
fn test_study_ignores_non_complete() {
    let mut study = HpoStudy::new("test", OptDirection::Minimize);
    study.add_trial(HpoTrial::new(
        0,
        vec![0.5],
        vec!["lr".into()],
        0.01,
        TrialStatus::Running,
    ));
    study.add_trial(HpoTrial::new(
        1,
        vec![0.1],
        vec!["lr".into()],
        0.5,
        TrialStatus::Complete,
    ));
    let best = study.best_trial().expect("best should exist");
    assert!(
        (best.value - 0.5).abs() < 1e-9,
        "running trial should not be selected"
    );
}

#[test]
fn test_fanova_importances_len() {
    let mut study = HpoStudy::new("test", OptDirection::Minimize);
    let names = vec!["lr".to_string(), "bs".to_string()];
    for i in 0..20 {
        let lr = 0.001 * (i + 1) as f64;
        let bs = 16.0 * (i + 1) as f64;
        let val = lr + 0.0 * bs;
        study.add_trial(HpoTrial::new(
            i,
            vec![lr, bs],
            names.clone(),
            val,
            TrialStatus::Complete,
        ));
    }
    let importances = importance_by_fanova(&study);
    assert_eq!(importances.len(), 2);
    assert!(importances.iter().all(|(_, v)| v.is_finite()));
}

#[test]
fn test_hpologger_best() {
    let mut logger = HpoLogger::new("exp", OptDirection::Maximize);
    logger.log_trial(HpoTrial::new(
        0,
        vec![0.5],
        vec!["x".into()],
        0.7,
        TrialStatus::Complete,
    ));
    logger.log_trial(HpoTrial::new(
        1,
        vec![0.9],
        vec!["x".into()],
        0.95,
        TrialStatus::Complete,
    ));
    let best = logger.best().expect("best should exist");
    assert!((best.value - 0.95).abs() < 1e-9);
}

// § WarmStartSampler

#[test]
fn test_warm_start_map_by_name() {
    let space = HpSpace::new()
        .add(HyperParameter::continuous("lr", 0.001, 0.1))
        .add(HyperParameter::integer("bs", 16, 128));
    let source_names = vec!["lr".to_string(), "bs".to_string()];
    let params = vec![0.01, 32.0];
    let mapped = WarmStartSampler::map_parameters(&params, &source_names, &space);
    assert_eq!(mapped.len(), 2);
    assert!((mapped[0] - 0.01).abs() < 1e-9);
    assert!((mapped[1] - 32.0).abs() < 1e-9);
}

#[test]
fn test_warm_start_select_top_n() {
    let space = HpSpace::new().add(HyperParameter::continuous("lr", 0.001, 0.1));
    let prev = PreviousStudy::new(
        vec!["lr".to_string()],
        vec![(vec![0.01], 0.9), (vec![0.05], 0.5), (vec![0.001], 0.3)],
    );
    let sampler = WarmStartSampler::new(vec![prev]);
    let starts = sampler.select_warm_starts(&space, 2);
    assert_eq!(starts.len(), 2);
    // Top-1 should be lr=0.01 (score 0.9)
    assert!((starts[0][0] - 0.01).abs() < 1e-9);
}

#[test]
fn test_warm_start_mismatched_names_defaults() {
    let space = HpSpace::new().add(HyperParameter::continuous("momentum", 0.8, 0.999));
    let source_names = vec!["lr".to_string()];
    let params = vec![0.01];
    let mapped = WarmStartSampler::map_parameters(&params, &source_names, &space);
    // "momentum" not in source → midpoint (0.8+0.999)/2
    let mid = (0.8 + 0.999) / 2.0;
    assert!((mapped[0] - mid).abs() < 1e-9);
}

// § Linear algebra helpers

#[test]
fn test_cholesky_2x2() {
    // A = [[4, 2],[2, 3]]
    let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
    let l = cholesky(&a).expect("Cholesky should succeed");
    // Reconstruct A = L L^T
    let n = 2;
    for i in 0..n {
        for j in 0..n {
            let llt: f64 = (0..n).map(|k| l[i][k] * l[j][k]).sum();
            assert!(
                (llt - a[i][j]).abs() < 1e-9,
                "Reconstruction error at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_chol_solve_identity() {
    let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
    let l = cholesky(&a).expect("ok");
    let b = vec![1.0, 0.0];
    let x = chol_solve(&l, &b).expect("solve ok");
    // Verify A x = b
    let ax0 = 4.0 * x[0] + 2.0 * x[1];
    let ax1 = 2.0 * x[0] + 3.0 * x[1];
    assert!((ax0 - 1.0).abs() < 1e-9);
    assert!((ax1 - 0.0).abs() < 1e-9);
}

#[test]
fn test_standard_normal_mean_approx_zero() {
    let mut rng = make_rng(123);
    let samples: Vec<f64> = (0..1000).map(|_| standard_normal(&mut rng)).collect();
    let m = mean(&samples);
    assert!(m.abs() < 0.15, "sample mean should be near 0, got {m}");
}

#[test]
fn test_eigen_decompose_sym_2x2() {
    // Symmetric matrix
    let a = vec![3.0, 1.0, 1.0, 3.0];
    let (evecs, evals) = eigen_decompose_sym(&a, 2);
    // Eigenvalues should be 2 and 4
    let mut sorted_evals = evals.clone();
    sorted_evals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    assert!(
        (sorted_evals[0] - 2.0).abs() < 1e-6,
        "eval 0 = {}",
        sorted_evals[0]
    );
    assert!(
        (sorted_evals[1] - 4.0).abs() < 1e-6,
        "eval 1 = {}",
        sorted_evals[1]
    );
    // Eigenvectors should be orthonormal
    let d = 2usize;
    for i in 0..d {
        let norm: f64 = (0..d)
            .map(|k| evecs[k][i] * evecs[k][i])
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "eigenvec {i} not unit: norm={norm}"
        );
    }
}
