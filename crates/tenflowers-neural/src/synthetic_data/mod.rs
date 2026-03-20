//! Simulation & Synthetic Data Generation
//!
//! Comprehensive toolkit for generating synthetic datasets, time-series simulations,
//! statistical distribution sampling, and quality evaluation metrics.
//!
//! # Modules
//!
//! - **Statistical** (`stats`): GMM, KDE, Copula, DensityRatio, Bootstrap sampling
//! - **Tabular** (`tabular`): CTGAN, TVAE, SMOTE, MICE imputation, ColumnTransformer
//! - **Time Series** (`timeseries`): ARIMA, GARCH, Ornstein-Uhlenbeck, fBm, Multivariate GBM
//! - **Image/Points** (`synthesis`): Perlin noise, Poisson-disk, shape generators, augmentation
//! - **Evaluation** (`evaluation`): Fidelity, privacy, utility, diversity metrics

pub mod evaluation;
mod math;
pub mod stats;
pub mod synthesis;
pub mod tabular;
pub mod timeseries;

// Re-export Section 1: Statistical
pub use stats::{
    BootstrapSampler, CopulaModel, DensityRatioEstimator, GaussianMixture, KernelDensityEstimator,
};

// Re-export Section 2: Tabular
pub use tabular::{
    ColumnTransformer, CtganGenerator, MissingValueImputer, NumericScaler, OneHotEncoder,
    SmoteOversampler, TvaeGenerator,
};

// Re-export Section 3: Time Series
pub use timeseries::{
    ArimaSimulator, FractionalBrownianMotion, GarchSimulator, MultivariateGbm, OrnsteinUhlenbeck,
};

// Re-export Section 4: Image & Point Cloud
pub use synthesis::{
    Augment, DataAugmentationPipeline, FeatureDropout, GaussianNoise, NoisyLabelGenerator,
    PerlinNoise, PoissonDiskSampling, SyntheticShapeGenerator,
};

// Re-export Section 5: Evaluation
pub use evaluation::{
    BenchmarkScore, DataGenerator, DiversityMetric, FidelityMetrics, PrivacyMetrics,
    SyntheticBenchmark, UtilityEvaluator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use math::sample_normal;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_gaussian_data(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| vec![sample_normal(&mut rng), sample_normal(&mut rng)])
            .collect()
    }

    // ── Section 1: Statistical ─────────────────────────────────────────────

    #[test]
    fn test_gmm_sample_count() {
        let data = make_gaussian_data(100, 1);
        let gmm = GaussianMixture::fit(&data, 3, 20, 42).expect("test value");
        let samples = gmm.sample(50, 99);
        assert_eq!(samples.len(), 50);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_gmm_log_likelihood() {
        let data = make_gaussian_data(80, 2);
        let gmm = GaussianMixture::fit(&data, 2, 15, 43).expect("test value");
        let ll = gmm.log_likelihood(&[0.0, 0.0]);
        assert!(ll.is_finite(), "log-likelihood must be finite");
    }

    #[test]
    fn test_gmm_weights_sum_to_one() {
        let data = make_gaussian_data(60, 3);
        let gmm = GaussianMixture::fit(&data, 4, 10, 44).expect("test value");
        let total: f64 = gmm.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_copula_sample_shape() {
        let data = make_gaussian_data(60, 5);
        let copula = CopulaModel::fit(&data).expect("test value");
        let samples = copula.sample(30, 77);
        assert_eq!(samples.len(), 30);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_kde_evaluate_positive() {
        let data = make_gaussian_data(50, 6);
        let kde = KernelDensityEstimator::fit(&data, 0.5).expect("test value");
        let val = kde.evaluate(&[0.0, 0.0]);
        assert!(val > 0.0);
    }

    #[test]
    fn test_kde_sample_count() {
        let data = make_gaussian_data(40, 7);
        let kde = KernelDensityEstimator::fit(&data, 0.3).expect("test value");
        let samples = kde.sample(20, 88);
        assert_eq!(samples.len(), 20);
    }

    #[test]
    fn test_density_ratio() {
        let mut rng = StdRng::seed_from_u64(10);
        let p: Vec<Vec<f64>> = (0..40)
            .map(|_| vec![sample_normal(&mut rng) + 2.0])
            .collect();
        let q: Vec<Vec<f64>> = (0..40).map(|_| vec![sample_normal(&mut rng)]).collect();
        let dre = DensityRatioEstimator::fit(&p, &q, 1.0, 42).expect("test value");
        let r_at_center = dre.ratio(&[2.0]);
        assert!(r_at_center >= 0.0);
    }

    #[test]
    fn test_bootstrap_ci_bounds() {
        let data: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64]).collect();
        let stat_fn = |d: &[Vec<f64>]| d.iter().map(|r| r[0]).sum::<f64>() / d.len() as f64;
        let (lo, hi) = BootstrapSampler::confidence_interval(&data, &stat_fn, 200, 0.1, 42);
        assert!(lo <= hi);
        assert!(lo > 0.0 && hi < 50.0);
    }

    #[test]
    fn test_bootstrap_sample_count() {
        let data = make_gaussian_data(30, 11);
        let boot = BootstrapSampler::sample(&data, 15, 55);
        assert_eq!(boot.len(), 15);
    }

    // ── Section 2: Tabular ─────────────────────────────────────────────────

    #[test]
    fn test_ctgan_sample_shape() {
        let data = make_gaussian_data(60, 20);
        let gen = CtganGenerator::fit(&data, 10, 42).expect("test value");
        let samples = gen.sample(25, 77);
        assert_eq!(samples.len(), 25);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_tvae_sample_shape() {
        let data = make_gaussian_data(50, 21);
        let gen = TvaeGenerator::fit(&data, 10, 42).expect("test value");
        let samples = gen.sample(20, 88);
        assert_eq!(samples.len(), 20);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_smote_oversample_count() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, (i * 2) as f64]).collect();
        let labels: Vec<usize> = (0..20).map(|i| if i < 5 { 1 } else { 0 }).collect();
        let (new_x, new_y) = SmoteOversampler::oversample(&x, &labels, 1, 3, 42).expect("test value");
        assert!(new_x.len() > x.len());
        assert_eq!(new_x.len(), new_y.len());
    }

    #[test]
    fn test_column_transformer() {
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 3.0],
            vec![2.0, 1.0, 4.0],
            vec![3.0, 0.0, 5.0],
        ];
        let mut ct = ColumnTransformer::new();
        ct.add_numeric(0, NumericScaler::StandardScaler);
        ct.add_categorical(1, OneHotEncoder::new(vec![0.0, 1.0]));
        ct.fit(&data).expect("test value");
        let result = ct.transform(&data);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 3); // 1 numeric + 2 one-hot
    }

    #[test]
    fn test_missing_imputer() {
        let data: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let imputer = MissingValueImputer::fit(&data).expect("test value");
        let with_missing: Vec<Vec<Option<f64>>> =
            vec![vec![Some(1.0), None], vec![None, Some(4.0)]];
        let filled = imputer.impute(&with_missing);
        assert_eq!(filled.len(), 2);
        assert_eq!(filled[0].len(), 2);
        assert!(filled[0][1].is_finite());
        assert!(filled[1][0].is_finite());
    }

    // ── Section 3: Time Series ─────────────────────────────────────────────

    #[test]
    fn test_arima_length() {
        let ts = ArimaSimulator::simulate(100, &[0.5, -0.2], 0, &[0.3], 1.0, 42);
        assert_eq!(ts.len(), 100);
    }

    #[test]
    fn test_arima_integrated() {
        let ts = ArimaSimulator::simulate(50, &[0.3], 1, &[], 1.0, 7);
        assert_eq!(ts.len(), 50);
    }

    #[test]
    fn test_garch_volatility_positive() {
        let (_returns, vols) = GarchSimulator::simulate(200, 0.0001, 0.05, 0.9, 42);
        assert_eq!(vols.len(), 200);
        assert!(vols.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_garch_returns_length() {
        let (returns, _) = GarchSimulator::simulate(100, 0.0001, 0.1, 0.8, 99);
        assert_eq!(returns.len(), 100);
    }

    #[test]
    fn test_ou_mean_reversion() {
        let n = 1000_usize;
        let path = OrnsteinUhlenbeck::simulate(n, 2.0, 5.0, 0.1, 0.01, 0.0, 42);
        assert_eq!(path.len(), n);
        let mean = path.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 5.0).abs() < 3.0,
            "OU process should revert toward mu=5"
        );
    }

    #[test]
    fn test_fbm_length() {
        let path = FractionalBrownianMotion::simulate(64, 0.7, 42).expect("test value");
        assert_eq!(path.len(), 64);
    }

    #[test]
    fn test_fbm_invalid_h() {
        assert!(FractionalBrownianMotion::simulate(10, 1.1, 42).is_err());
        assert!(FractionalBrownianMotion::simulate(10, 0.0, 42).is_err());
    }

    #[test]
    fn test_mvgbm_shape() {
        let mu = vec![0.05, 0.08];
        let sigma = vec![0.2, 0.3];
        let corr = vec![1.0, 0.3, 0.3, 1.0];
        let s0 = vec![100.0, 200.0];
        let paths =
            MultivariateGbm::simulate(50, &mu, &sigma, &corr, 1.0 / 252.0, &s0, 42).expect("test value");
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].len(), 51);
    }

    #[test]
    fn test_mvgbm_positive_prices() {
        let mu = vec![0.05];
        let sigma = vec![0.2];
        let corr = vec![1.0];
        let s0 = vec![100.0];
        let paths = MultivariateGbm::simulate(100, &mu, &sigma, &corr, 0.01, &s0, 9).expect("test value");
        assert!(paths[0].iter().all(|&p| p > 0.0));
    }

    // ── Section 4: Synthesis ───────────────────────────────────────────────

    #[test]
    fn test_perlin_noise_range() {
        let pn = PerlinNoise::new(42);
        for i in 0..20 {
            for j in 0..20 {
                let v = pn.noise2d(i as f64 * 0.3, j as f64 * 0.3);
                assert!(v.abs() <= 2.0, "Perlin noise out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_perlin_fbm() {
        let pn = PerlinNoise::new(99);
        let v = pn.fbm(0.5, 0.5, 4, 2.0, 0.5);
        assert!(v.is_finite());
    }

    #[test]
    fn test_poisson_disk_min_dist() {
        let pts = PoissonDiskSampling::sample(10.0, 10.0, 1.5, 42);
        assert!(!pts.is_empty());
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let dx = pts[i][0] - pts[j][0];
                let dy = pts[i][1] - pts[j][1];
                let d = (dx * dx + dy * dy).sqrt();
                assert!(d >= 1.5 - 1e-9, "Poisson disk violation: d={}", d);
            }
        }
    }

    #[test]
    fn test_sphere_point_count() {
        let pts = SyntheticShapeGenerator::sphere(100, 1.0, 42);
        assert_eq!(pts.len(), 100);
        for p in &pts {
            let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-9,
                "sphere point not on surface: r={}",
                r
            );
        }
    }

    #[test]
    fn test_cylinder_count() {
        let pts = SyntheticShapeGenerator::cylinder(50, 2.0, 5.0, 7);
        assert_eq!(pts.len(), 50);
    }

    #[test]
    fn test_torus_count() {
        let pts = SyntheticShapeGenerator::torus(80, 3.0, 1.0, 13);
        assert_eq!(pts.len(), 80);
    }

    #[test]
    fn test_noisy_labels_count() {
        let y: Vec<usize> = (0..100).map(|i| i % 5).collect();
        let noisy = NoisyLabelGenerator::flip_labels(&y, 5, 0.2, 42);
        assert_eq!(noisy.len(), 100);
        assert!(noisy.iter().all(|&l| l < 5));
    }

    #[test]
    fn test_asymmetric_noise() {
        let y = vec![0_usize, 1, 0, 1, 0];
        let tm = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let noisy = NoisyLabelGenerator::asymmetric_noise(&y, &tm, 42);
        assert_eq!(noisy.len(), 5);
        assert!(noisy.iter().all(|&l| l <= 1));
    }

    #[test]
    fn test_augmentation_pipeline() {
        let mut pipeline = DataAugmentationPipeline::new();
        pipeline.add_step(Box::new(GaussianNoise::new(0.1)));
        pipeline.add_step(Box::new(FeatureDropout::new(0.1)));
        let x = vec![1.0, 2.0, 3.0];
        let mut rng = StdRng::seed_from_u64(42);
        let out = pipeline.apply(&x, &mut rng);
        assert_eq!(out.len(), x.len());
    }

    // ── Section 5: Evaluation ──────────────────────────────────────────────

    #[test]
    fn test_fidelity_marginal_coverage() {
        let real = make_gaussian_data(100, 30);
        let synthetic = make_gaussian_data(100, 31);
        let cov = FidelityMetrics::marginal_coverage(&real, &synthetic, 20);
        assert!(
            (0.0..=1.0).contains(&cov),
            "coverage must be in [0,1]: {}",
            cov
        );
    }

    #[test]
    fn test_fidelity_perfect_coverage() {
        let data = make_gaussian_data(100, 32);
        let cov = FidelityMetrics::marginal_coverage(&data, &data, 20);
        assert!(
            cov > 0.9,
            "identical distributions should have high coverage"
        );
    }

    #[test]
    fn test_correlation_similarity() {
        let real = make_gaussian_data(80, 33);
        let synthetic = make_gaussian_data(80, 34);
        let sim = FidelityMetrics::correlation_similarity(&real, &synthetic);
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_privacy_metrics_dcr() {
        let real = make_gaussian_data(50, 40);
        let synthetic = make_gaussian_data(50, 41);
        let dcr = PrivacyMetrics::nearest_neighbor_distance_ratio(&real, &synthetic);
        assert!(dcr >= 0.0);
    }

    #[test]
    fn test_diversity_coverage() {
        let real = make_gaussian_data(50, 50);
        let synthetic = make_gaussian_data(50, 51);
        let cov = DiversityMetric::compute_coverage(&real, &synthetic, 3.0);
        assert!((0.0..=1.0).contains(&cov));
    }

    #[test]
    fn test_diversity_density() {
        let real = make_gaussian_data(30, 52);
        let synthetic = make_gaussian_data(30, 53);
        let dens = DiversityMetric::compute_density(&real, &synthetic, 3);
        assert!(dens >= 0.0);
    }

    #[test]
    fn test_utility_evaluator() {
        let syn_x: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.1, 0.1]];
        let syn_y: Vec<usize> = vec![0, 1, 0];
        let real_x: Vec<Vec<f64>> = vec![vec![0.05, 0.05], vec![0.9, 0.9]];
        let real_y: Vec<usize> = vec![0, 1];
        let acc = UtilityEvaluator::evaluate(&syn_x, &syn_y, &real_x, &real_y);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_benchmark_score_count() {
        struct BootstrapGen;
        impl DataGenerator for BootstrapGen {
            fn name(&self) -> &str {
                "bootstrap"
            }
            fn generate(&self, real: &[Vec<f64>], n: usize, seed: u64) -> Vec<Vec<f64>> {
                BootstrapSampler::sample(real, n, seed)
            }
        }
        let real = make_gaussian_data(50, 60);
        let generators: Vec<Box<dyn DataGenerator>> = vec![Box::new(BootstrapGen)];
        let scores = SyntheticBenchmark::benchmark(&generators, &real, 30, 10, 2.0, 42);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].name, "bootstrap");
    }
}
