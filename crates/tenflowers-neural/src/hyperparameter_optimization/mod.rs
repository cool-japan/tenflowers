//! Advanced Hyperparameter Optimization (HPO) Methods.
//!
//! This module provides production-grade HPO methods beyond the basic grid/random
//! search in `hparam/`. It includes:
//!
//! - [`HpSpace`] / [`HyperParameter`] — flexible parameter space definitions
//! - \[`BayesianOptimizer`\] (aliased as [`HpoBayesianOptimizer`]) — GP-based Bayesian optimization
//! - [`HyperBandScheduler`] — multi-fidelity successive halving
//! - [`BohbOptimizer`] — BOHB (Bayesian Optimization + HyperBand)
//! - [`PopulationBasedTraining`] — evolutionary HPO (PBT)
//! - [`EvolutionaryStrategy`] — CMA-ES for HPO
//! - [`MultiObjectiveHpo`] — Pareto-front HPO with NSGA-II
//! - [`MedianStopping`] / [`SuccessiveHalving`] / [`PercentileStop`] — early termination
//! - [`HpoStudy`] / [`HpoLogger`] — experiment tracking
//! - [`WarmStartSampler`] — transfer-learning-based warm starting
//!
//! All randomness uses `scirs2_core::random` (no `rand` crate). No `unsafe` code.
//! No `unwrap()`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §1. HyperParameter and HpSpace
// ─────────────────────────────────────────────────────────────────────────────

/// The type of a hyperparameter.
#[derive(Debug, Clone, PartialEq)]
pub enum HpType {
    /// Continuous real-valued parameter in `[bounds.0, bounds.1]`.
    Continuous,
    /// Categorical parameter; valid values are indices into `choices`.
    Categorical,
    /// Integer parameter (rounded continuous).
    Integer,
    /// Log-scale continuous parameter (sampled uniformly in log space).
    Log,
}

/// A single hyperparameter definition.
#[derive(Debug, Clone)]
pub struct HyperParameter {
    /// Name of the parameter.
    pub name: String,
    /// Parameter type.
    pub hp_type: HpType,
    /// `(low, high)` bounds for Continuous / Integer / Log types.
    pub bounds: (f64, f64),
    /// Valid discrete values for Categorical type.
    pub choices: Vec<f64>,
    /// If true, sample/transform in log space (applicable to Continuous and Integer).
    pub log_scale: bool,
}

impl HyperParameter {
    /// Construct a continuous parameter.
    pub fn continuous(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            hp_type: HpType::Continuous,
            bounds: (low, high),
            choices: vec![],
            log_scale: false,
        }
    }

    /// Construct a log-scale continuous parameter.
    pub fn log_continuous(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            hp_type: HpType::Log,
            bounds: (low, high),
            choices: vec![],
            log_scale: true,
        }
    }

    /// Construct an integer parameter.
    pub fn integer(name: impl Into<String>, low: i64, high: i64) -> Self {
        Self {
            name: name.into(),
            hp_type: HpType::Integer,
            bounds: (low as f64, high as f64),
            choices: vec![],
            log_scale: false,
        }
    }

    /// Construct a categorical parameter from float-encoded choices.
    pub fn categorical(name: impl Into<String>, choices: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            hp_type: HpType::Categorical,
            bounds: (0.0, choices.len().saturating_sub(1) as f64),
            choices,
            log_scale: false,
        }
    }

    /// Sample a single value from this parameter using the provided RNG.
    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        match self.hp_type {
            HpType::Continuous => {
                let u: f64 = rng.random();
                let (lo, hi) = self.bounds;
                if self.log_scale && lo > 0.0 {
                    let log_lo = lo.ln();
                    let log_hi = hi.ln();
                    (log_lo + u * (log_hi - log_lo)).exp()
                } else {
                    lo + u * (hi - lo)
                }
            }
            HpType::Log => {
                let u: f64 = rng.random();
                let (lo, hi) = self.bounds;
                let log_lo = lo.max(1e-300).ln();
                let log_hi = hi.max(1e-300).ln();
                (log_lo + u * (log_hi - log_lo)).exp()
            }
            HpType::Integer => {
                let (lo, hi) = self.bounds;
                let range = (hi - lo + 1.0).max(1.0) as usize;
                let idx: usize = rng.random_range(0..range);
                lo + idx as f64
            }
            HpType::Categorical => {
                if self.choices.is_empty() {
                    return 0.0;
                }
                let idx: usize = rng.random_range(0..self.choices.len());
                self.choices[idx]
            }
        }
    }

    /// Transform a raw value to [0, 1] (for GP input normalisation).
    pub fn to_unit(&self, v: f64) -> f64 {
        let (lo, hi) = self.bounds;
        let range = (hi - lo).max(1e-300);
        match self.hp_type {
            HpType::Log => {
                let log_lo = lo.max(1e-300).ln();
                let log_hi = hi.max(1e-300).ln();
                let log_range = (log_hi - log_lo).max(1e-300);
                (v.max(1e-300).ln() - log_lo) / log_range
            }
            HpType::Categorical => {
                if self.choices.len() <= 1 {
                    return 0.0;
                }
                let idx = self
                    .choices
                    .iter()
                    .position(|&c| (c - v).abs() < 1e-12)
                    .unwrap_or(0);
                idx as f64 / (self.choices.len() - 1) as f64
            }
            _ => (v - lo) / range,
        }
    }

    /// Reverse of `to_unit`: from \[0,1\] back to parameter space.
    pub fn from_unit(&self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        let (lo, hi) = self.bounds;
        match self.hp_type {
            HpType::Log => {
                let log_lo = lo.max(1e-300).ln();
                let log_hi = hi.max(1e-300).ln();
                (log_lo + u * (log_hi - log_lo)).exp()
            }
            HpType::Integer => {
                let raw = lo + u * (hi - lo);
                raw.round().clamp(lo, hi)
            }
            HpType::Categorical => {
                if self.choices.is_empty() {
                    return 0.0;
                }
                let idx = (u * (self.choices.len() - 1) as f64).round() as usize;
                self.choices[idx.min(self.choices.len() - 1)]
            }
            _ => lo + u * (hi - lo),
        }
    }
}

/// A collection of hyperparameters forming the search space.
#[derive(Debug, Clone, Default)]
pub struct HpSpace {
    /// Parameter definitions.
    pub params: Vec<HyperParameter>,
}

impl HpSpace {
    /// Create an empty space.
    pub fn new() -> Self {
        Self { params: vec![] }
    }

    /// Add a parameter definition.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, hp: HyperParameter) -> Self {
        self.params.push(hp);
        self
    }

    /// Number of parameters in the space.
    pub fn ndim(&self) -> usize {
        self.params.len()
    }

    /// Sample a random point from the space.
    pub fn sample_random(&self, rng: &mut StdRng) -> Vec<f64> {
        self.params.iter().map(|p| p.sample(rng)).collect()
    }

    /// Normalise a raw parameter vector to \[0,1\]^d.
    pub fn transform_to_unit(&self, values: &[f64]) -> Vec<f64> {
        self.params
            .iter()
            .zip(values.iter())
            .map(|(p, &v)| p.to_unit(v))
            .collect()
    }

    /// Denormalise a unit-cube vector back to parameter space.
    pub fn transform_from_unit(&self, unit: &[f64]) -> Vec<f64> {
        self.params
            .iter()
            .zip(unit.iter())
            .map(|(p, &u)| p.from_unit(u))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2. GpHpo — lightweight GP surrogate
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight Gaussian Process surrogate for HPO (RBF kernel, Cholesky inference).
#[derive(Debug, Clone)]
pub struct GpHpo {
    /// Training inputs (unit-normalised), shape [n x d].
    pub x_train: Vec<Vec<f64>>,
    /// Training targets.
    pub y_train: Vec<f64>,
    /// RBF length scale.
    pub length_scale: f64,
    /// Observation noise variance.
    pub noise: f64,
    /// Cached Cholesky factor L (lower triangular), stored row-major.
    chol: Vec<Vec<f64>>,
    /// Cached alpha = L^{-T} L^{-1} y.
    alpha: Vec<f64>,
}

impl GpHpo {
    /// Create a new GP with given hyper-parameters.
    pub fn new(length_scale: f64, noise: f64) -> Self {
        Self {
            x_train: vec![],
            y_train: vec![],
            length_scale,
            noise,
            chol: vec![],
            alpha: vec![],
        }
    }

    /// Compute RBF kernel between two vectors.
    fn rbf(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| {
                let d = ai - bi;
                d * d
            })
            .sum();
        (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }

    /// Fit the GP to the current `x_train` / `y_train` by computing Cholesky.
    pub fn fit(&mut self) -> Result<()> {
        let n = self.x_train.len();
        if n == 0 {
            return Ok(());
        }
        // Build K + noise*I
        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.rbf(&self.x_train[i], &self.x_train[j]);
            }
            k[i][i] += self.noise;
        }
        // Cholesky decomposition
        let l = cholesky(&k).map_err(|e| TensorError::ComputeError {
            operation: "GpHpo::fit".into(),
            details: e,
            retry_possible: false,
            context: None,
        })?;
        // alpha = K^{-1} y via forward/backward substitution
        let alpha = chol_solve(&l, &self.y_train)?;
        self.chol = l;
        self.alpha = alpha;
        Ok(())
    }

    /// Predict mean and variance at a test point `x_star`.
    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 {
            return (0.0, 1.0);
        }
        // k_star = [k(x*, x_i)]
        let k_star: Vec<f64> = self.x_train.iter().map(|xi| self.rbf(x_star, xi)).collect();
        // mean = k_star^T alpha
        let mean: f64 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(ks, a)| ks * a)
            .sum();
        // variance = k(x*,x*) - k_star^T K^{-1} k_star
        let k_ss = self.rbf(x_star, x_star);
        // v = L^{-1} k_star
        let v = forward_sub(&self.chol, &k_star);
        let var: f64 = k_ss - v.iter().map(|vi| vi * vi).sum::<f64>();
        (mean, var.max(1e-10))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2b. Acquisition Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Acquisition function variants for Bayesian optimisation.
#[derive(Debug, Clone, PartialEq)]
pub enum HpoAcqFunction {
    /// Expected Improvement with `xi` exploration bonus.
    ExpectedImprovement { xi: f64 },
    /// Probability of Improvement.
    ProbabilityOfImprovement { xi: f64 },
    /// Upper Confidence Bound with `kappa` exploration weight.
    UpperConfidenceBound { kappa: f64 },
    /// Log Expected Improvement (numerically stable for small improvements).
    LogExpectedImprovement { xi: f64 },
}

impl HpoAcqFunction {
    /// Evaluate acquisition at (mean, std) given `best_y` (the current best observed value).
    pub fn evaluate(&self, mean: f64, std: f64, best_y: f64) -> f64 {
        match self {
            HpoAcqFunction::ExpectedImprovement { xi } => {
                let z = (mean - best_y - xi) / std.max(1e-9);
                std * (z * normal_cdf(z) + normal_pdf(z))
            }
            HpoAcqFunction::ProbabilityOfImprovement { xi } => {
                let z = (mean - best_y - xi) / std.max(1e-9);
                normal_cdf(z)
            }
            HpoAcqFunction::UpperConfidenceBound { kappa } => mean + kappa * std,
            HpoAcqFunction::LogExpectedImprovement { xi } => {
                let z = (mean - best_y - xi) / std.max(1e-9);
                let ei = std * (z * normal_cdf(z) + normal_pdf(z));
                ei.max(1e-300).ln()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2c. BayesianOptimizer (HPO variant)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the HPO Bayesian optimiser.
#[derive(Debug, Clone)]
pub struct HpoBayesianConfig {
    /// Number of random initial evaluations before using the GP.
    pub n_initial: usize,
    /// Number of Bayesian optimisation iterations after warm-up.
    pub n_iter: usize,
    /// Exploration parameter `xi` for EI/PI.
    pub xi: f64,
    /// Exploration parameter `kappa` for UCB.
    pub kappa: f64,
    /// Acquisition function variant.
    pub acq: HpoAcqFunction,
    /// Number of random candidates evaluated per suggestion step.
    pub n_candidates: usize,
}

impl Default for HpoBayesianConfig {
    fn default() -> Self {
        Self {
            n_initial: 5,
            n_iter: 25,
            xi: 0.01,
            kappa: 2.576,
            acq: HpoAcqFunction::ExpectedImprovement { xi: 0.01 },
            n_candidates: 1000,
        }
    }
}

/// A full Bayesian optimisation loop backed by a GP surrogate.
///
/// Aliased as `HpoBayesianOptimizer` at the module level to avoid name
/// collision with `bayesian_opt::BayesianOptimizer`.
#[derive(Debug, Clone)]
pub struct HpoBayesianOptimizer {
    /// The hyperparameter space.
    pub space: HpSpace,
    /// GP surrogate.
    pub gp: GpHpo,
    /// Observed (params, value) pairs — params stored in *raw* (not unit) space.
    pub observations: Vec<(Vec<f64>, f64)>,
    /// Configuration.
    pub config: HpoBayesianConfig,
}

impl HpoBayesianOptimizer {
    /// Construct a new Bayesian optimiser.
    pub fn new(space: HpSpace, config: HpoBayesianConfig) -> Self {
        let gp = GpHpo::new(1.0, 1e-3);
        Self {
            space,
            gp,
            observations: vec![],
            config,
        }
    }

    /// Record an observation.
    pub fn observe(&mut self, params: &[f64], value: f64) {
        self.observations.push((params.to_vec(), value));
        // Update GP training data
        let unit = self.space.transform_to_unit(params);
        self.gp.x_train.push(unit);
        self.gp.y_train.push(value);
        let _ = self.gp.fit();
    }

    /// Suggest the next point to evaluate.
    ///
    /// Returns random samples during warm-up, then maximises the acquisition function.
    pub fn suggest(&self, rng: &mut StdRng) -> Vec<f64> {
        if self.observations.len() < self.config.n_initial {
            return self.space.sample_random(rng);
        }
        // Find best observed value (maximisation convention)
        let best_y = self
            .observations
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut best_acq = f64::NEG_INFINITY;
        let mut best_candidate = self.space.sample_random(rng);

        for _ in 0..self.config.n_candidates {
            let candidate = self.space.sample_random(rng);
            let unit = self.space.transform_to_unit(&candidate);
            let (mean, var) = self.gp.predict(&unit);
            let std = var.sqrt();
            let acq = self.config.acq.evaluate(mean, std, best_y);
            if acq > best_acq {
                best_acq = acq;
                best_candidate = candidate;
            }
        }
        best_candidate
    }

    /// Return the best observed (params, value) pair.
    pub fn best(&self) -> Option<(&[f64], f64)> {
        self.observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, v)| (p.as_slice(), *v))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3. HyperBandScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for HyperBand.
#[derive(Debug, Clone)]
pub struct HyperBandConfig {
    /// Maximum resource (iterations / epochs) per configuration.
    pub max_iter: f64,
    /// Halving factor η (typically 3 or 4).
    pub eta: f64,
    /// Minimum resource per configuration.
    pub min_resource: f64,
}

impl Default for HyperBandConfig {
    fn default() -> Self {
        Self {
            max_iter: 81.0,
            eta: 3.0,
            min_resource: 1.0,
        }
    }
}

/// A single HyperBand bracket (one successive-halving run).
#[derive(Debug, Clone)]
pub struct HbBracket {
    /// Bracket identifier.
    pub bracket_id: usize,
    /// Number of configurations in this bracket.
    pub n: usize,
    /// Initial resource allocation for this bracket.
    pub r: f64,
    /// Number of successive-halving rounds in this bracket.
    pub s: usize,
}

/// The full HyperBand scheduler managing multiple brackets.
#[derive(Debug, Clone)]
pub struct HyperBandScheduler {
    /// Computed bracket schedule.
    pub brackets: Vec<HbBracket>,
    /// Index of the currently active bracket.
    pub current: usize,
    /// Configuration.
    pub config: HyperBandConfig,
}

impl HyperBandScheduler {
    /// Create a scheduler and plan the bracket schedule.
    pub fn new(config: HyperBandConfig) -> Self {
        let brackets = Self::plan_schedule(&config);
        Self {
            brackets,
            current: 0,
            config,
        }
    }

    /// Plan the full schedule of brackets for the given HyperBand config.
    pub fn plan_schedule(config: &HyperBandConfig) -> Vec<HbBracket> {
        // s_max = floor(log_{eta}(max_iter / min_resource))
        let s_max = (config.max_iter / config.min_resource)
            .max(1.0)
            .log(config.eta)
            .floor() as usize;
        let mut brackets = Vec::with_capacity(s_max + 1);
        for s in (0..=s_max).rev() {
            let n =
                ((s_max + 1) as f64 / (s + 1) as f64 * config.eta.powi(s as i32)).ceil() as usize;
            let r = config.max_iter / config.eta.powi(s as i32);
            brackets.push(HbBracket {
                bracket_id: s_max - s,
                n,
                r,
                s,
            });
        }
        brackets
    }

    /// Generate random configurations for a given bracket.
    pub fn get_configurations(
        &self,
        bracket: &HbBracket,
        space: &HpSpace,
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
        (0..bracket.n).map(|_| space.sample_random(rng)).collect()
    }

    /// Keep the top `n_keep` configurations by score (descending).
    pub fn promote(scores: &[(Vec<f64>, f64)], n_keep: usize) -> Vec<Vec<f64>> {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n_keep).map(|(p, _)| p).collect()
    }

    /// Run a single successive-halving round within bracket `b_idx`.
    /// Returns configurations to evaluate at each resource level.
    pub fn successive_halving_rounds(&self, bracket_idx: usize) -> Vec<(usize, f64)> {
        if bracket_idx >= self.brackets.len() {
            return vec![];
        }
        let b = &self.brackets[bracket_idx];
        (0..=b.s)
            .map(|i| {
                let n_i = (b.n as f64 / self.config.eta.powi(i as i32)).floor() as usize;
                let r_i = b.r * self.config.eta.powi(i as i32);
                (n_i.max(1), r_i)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4. BOHB — Bayesian Optimization + HyperBand
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for BOHB.
#[derive(Debug, Clone)]
pub struct BohbConfig {
    /// Number of random observations before fitting KDEs.
    pub n_initial: usize,
    /// KDE bandwidth.
    pub bandwidth: f64,
    /// Fraction of top observations used for the "good" KDE.
    pub top_frac: f64,
}

impl Default for BohbConfig {
    fn default() -> Self {
        Self {
            n_initial: 10,
            bandwidth: 1.0,
            top_frac: 0.15,
        }
    }
}

/// Parzen-window KDE sampler used by BOHB.
#[derive(Debug, Clone)]
pub struct KdeSampler {
    /// Observations classified as "good" (top performers).
    pub good_obs: Vec<f64>,
    /// Observations classified as "bad" (remaining).
    pub bad_obs: Vec<f64>,
    /// KDE bandwidth.
    pub bandwidth: f64,
}

impl KdeSampler {
    /// Evaluate the KDE density at `x` using Gaussian kernel.
    pub fn kde_pdf(x: f64, samples: &[f64], bw: f64) -> f64 {
        if samples.is_empty() {
            return 1.0;
        }
        let n = samples.len() as f64;
        let sum: f64 = samples.iter().map(|&s| normal_pdf((x - s) / bw) / bw).sum();
        sum / n
    }

    /// Sample a candidate by maximising the ratio l(x)/g(x) over random draws.
    pub fn sample_from_kde(
        good: &[f64],
        bad: &[f64],
        bw: f64,
        rng: &mut StdRng,
        n_candidates: usize,
    ) -> f64 {
        if good.is_empty() {
            // fall back to uniform
            return rng.random();
        }
        let mut best_ratio = f64::NEG_INFINITY;
        let mut best_x = 0.5_f64;
        for _ in 0..n_candidates {
            // sample from good KDE
            let idx: usize = rng.random_range(0..good.len());
            let noise: f64 = rng.random::<f64>() * 2.0 - 1.0; // uniform [-1,1]
            let x = (good[idx] + noise * bw).clamp(0.0, 1.0);
            let lx = Self::kde_pdf(x, good, bw).max(1e-300);
            let gx = Self::kde_pdf(x, bad, bw).max(1e-300);
            let ratio = lx / gx;
            if ratio > best_ratio {
                best_ratio = ratio;
                best_x = x;
            }
        }
        best_x
    }

    /// Fit the KDE sampler from a list of (value, score) observations.
    pub fn fit_from_observations(obs: &[(f64, f64)], top_frac: f64, bw: f64) -> Self {
        if obs.is_empty() {
            return Self {
                good_obs: vec![],
                bad_obs: vec![],
                bandwidth: bw,
            };
        }
        let mut sorted = obs.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n_good = ((obs.len() as f64 * top_frac).ceil() as usize).max(1);
        let good_obs: Vec<f64> = sorted[..n_good].iter().map(|(v, _)| *v).collect();
        let bad_obs: Vec<f64> = sorted[n_good..].iter().map(|(v, _)| *v).collect();
        Self {
            good_obs,
            bad_obs,
            bandwidth: bw,
        }
    }
}

/// BOHB optimiser — combines Bayesian Optimization with HyperBand.
#[derive(Debug, Clone)]
pub struct BohbOptimizer {
    /// Hyperparameter space.
    pub space: HpSpace,
    /// Underlying HyperBand scheduler.
    pub hb: HyperBandScheduler,
    /// Per-dimension KDE samplers (one per parameter).
    pub samplers: Vec<KdeSampler>,
    /// All recorded observations: (params, score).
    observations: Vec<(Vec<f64>, f64)>,
    /// Config.
    pub config: BohbConfig,
}

impl BohbOptimizer {
    /// Create a new BOHB optimiser.
    pub fn new(space: HpSpace, hb_config: HyperBandConfig, bohb_config: BohbConfig) -> Self {
        let hb = HyperBandScheduler::new(hb_config);
        let samplers = vec![
            KdeSampler {
                good_obs: vec![],
                bad_obs: vec![],
                bandwidth: bohb_config.bandwidth
            };
            space.ndim()
        ];
        Self {
            space,
            hb,
            samplers,
            observations: vec![],
            config: bohb_config,
        }
    }

    /// Record an observation and refit KDE samplers.
    pub fn observe(&mut self, params: &[f64], score: f64) {
        self.observations.push((params.to_vec(), score));
        // Refit per-dimension KDE samplers
        for (dim, sampler) in self.samplers.iter_mut().enumerate() {
            let dim_obs: Vec<(f64, f64)> = self
                .observations
                .iter()
                .map(|(p, s)| (self.space.params[dim].to_unit(p[dim]), *s))
                .collect();
            *sampler = KdeSampler::fit_from_observations(
                &dim_obs,
                self.config.top_frac,
                self.config.bandwidth,
            );
        }
    }

    /// Suggest a new configuration.
    pub fn suggest_bohb(&self, rng: &mut StdRng) -> Vec<f64> {
        if self.observations.len() < self.config.n_initial {
            return self.space.sample_random(rng);
        }
        // Sample each dimension independently via its KDE sampler
        let unit: Vec<f64> = self
            .samplers
            .iter()
            .map(|s| KdeSampler::sample_from_kde(&s.good_obs, &s.bad_obs, s.bandwidth, rng, 64))
            .collect();
        self.space.transform_from_unit(&unit)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5. Population-Based Training (PBT)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Population-Based Training.
#[derive(Debug, Clone)]
pub struct PbtConfig {
    /// Number of workers in the population.
    pub population_size: usize,
    /// Fraction of the population to replace during exploitation.
    pub exploit_frac: f64,
    /// Additive noise magnitude during exploration.
    pub explore_noise: f64,
}

impl Default for PbtConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            exploit_frac: 0.2,
            explore_noise: 0.2,
        }
    }
}

/// A single worker in the PBT population.
#[derive(Debug, Clone)]
pub struct PbtMember {
    /// Current hyperparameter values.
    pub params: Vec<f64>,
    /// Last reported score.
    pub score: f64,
    /// Number of training steps completed.
    pub steps_trained: usize,
}

impl PbtMember {
    /// Construct a new member.
    pub fn new(params: Vec<f64>) -> Self {
        Self {
            params,
            score: f64::NEG_INFINITY,
            steps_trained: 0,
        }
    }
}

/// The PBT population.
#[derive(Debug, Clone)]
pub struct PbtPopulation {
    /// Workers.
    pub members: Vec<PbtMember>,
}

impl PbtPopulation {
    /// Construct a population by random sampling.
    pub fn random(space: &HpSpace, size: usize, rng: &mut StdRng) -> Self {
        let members = (0..size)
            .map(|_| PbtMember::new(space.sample_random(rng)))
            .collect();
        Self { members }
    }

    /// Exploit: replace the bottom-`exploit_frac` workers' params with top workers' params.
    pub fn exploit(&mut self, rng: &mut StdRng, exploit_frac: f64) {
        let n = self.members.len();
        if n < 2 {
            return;
        }
        let n_replace = ((n as f64 * exploit_frac).ceil() as usize).min(n / 2);
        // Sort indices by score descending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            self.members[b]
                .score
                .partial_cmp(&self.members[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Replace bottom-n_replace with a random top-n_replace
        let top_indices: Vec<usize> = order[..n_replace].to_vec();
        let bottom_indices: Vec<usize> = order[n - n_replace..].to_vec();
        for &bot in &bottom_indices {
            let src_idx: usize = rng.random_range(0..top_indices.len());
            let src = top_indices[src_idx];
            let new_params = self.members[src].params.clone();
            self.members[bot].params = new_params;
        }
    }

    /// Explore: perturb each parameter by ±noise or resample.
    pub fn explore_member(
        params: &[f64],
        space: &HpSpace,
        noise: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        params
            .iter()
            .zip(space.params.iter())
            .map(|(&v, hp)| {
                let perturb: f64 = rng.random();
                if perturb < 0.1 {
                    // resample
                    hp.sample(rng)
                } else {
                    let delta: f64 = (rng.random::<f64>() * 2.0 - 1.0) * noise;
                    let (lo, hi) = hp.bounds;
                    (v * (1.0 + delta)).clamp(lo, hi)
                }
            })
            .collect()
    }

    /// Run a single PBT step: update scores, exploit, then explore.
    pub fn pbt_step(
        &mut self,
        scores: &[f64],
        space: &HpSpace,
        config: &PbtConfig,
        rng: &mut StdRng,
    ) {
        // Update scores and step counts
        for (member, &score) in self.members.iter_mut().zip(scores.iter()) {
            member.score = score;
            member.steps_trained += 1;
        }
        // Exploit
        self.exploit(rng, config.exploit_frac);
        // Explore
        let new_params: Vec<Vec<f64>> = self
            .members
            .iter()
            .map(|m| Self::explore_member(&m.params, space, config.explore_noise, rng))
            .collect();
        for (member, params) in self.members.iter_mut().zip(new_params) {
            member.params = params;
        }
    }
}

/// Standalone population-based training runner.
#[derive(Debug, Clone)]
pub struct PopulationBasedTraining {
    /// The population.
    pub population: PbtPopulation,
    /// Configuration.
    pub config: PbtConfig,
    /// Hyperparameter space.
    pub space: HpSpace,
}

impl PopulationBasedTraining {
    /// Create a new PBT instance with a randomly initialised population.
    pub fn new(space: HpSpace, config: PbtConfig, rng: &mut StdRng) -> Self {
        let population = PbtPopulation::random(&space, config.population_size, rng);
        Self {
            population,
            config,
            space,
        }
    }

    /// Run one PBT step given the current scores for each worker.
    pub fn step(&mut self, scores: &[f64], rng: &mut StdRng) {
        let config = self.config.clone();
        let space = self.space.clone();
        self.population.pbt_step(scores, &space, &config, rng);
    }

    /// Return the best member in the current population.
    pub fn best(&self) -> Option<&PbtMember> {
        self.population.members.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6. CMA-ES for HPO
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CMA-ES HPO.
#[derive(Debug, Clone)]
pub struct CmaHpoConfig {
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Initial step size σ₀.
    pub sigma0: f64,
    /// Population size λ (default: 4 + floor(3 ln d)).
    pub lambda: usize,
}

impl CmaHpoConfig {
    /// Create a config with auto-computed λ.
    pub fn new(n_dims: usize, sigma0: f64) -> Self {
        let lambda = (4.0 + (3.0 * (n_dims as f64).ln()).floor()) as usize;
        Self {
            n_dims,
            sigma0,
            lambda: lambda.max(4),
        }
    }
}

/// State of a running CMA-ES optimiser.
#[derive(Debug, Clone)]
pub struct CmaHpoState {
    /// Distribution mean.
    pub mean: Vec<f64>,
    /// Step size σ.
    pub sigma: f64,
    /// Covariance matrix C (stored row-major, d×d).
    pub cov: Vec<f64>,
    /// Evolution path p_σ.
    pub p_sigma: Vec<f64>,
    /// Evolution path p_c.
    pub p_c: Vec<f64>,
    /// Eigenvalues of C.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors of C (row-major, d×d).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Generation counter.
    pub generation: usize,
    /// Problem dimension.
    pub n_dims: usize,
}

impl CmaHpoState {
    /// Initialise with a given mean and σ₀.
    pub fn new(mean: Vec<f64>, sigma0: f64) -> Self {
        let d = mean.len();
        let mut cov = vec![0.0_f64; d * d];
        for i in 0..d {
            cov[i * d + i] = 1.0;
        }
        let eigenvectors = (0..d)
            .map(|i| {
                let mut row = vec![0.0_f64; d];
                row[i] = 1.0;
                row
            })
            .collect();
        Self {
            mean,
            sigma: sigma0,
            cov,
            p_sigma: vec![0.0; d],
            p_c: vec![0.0; d],
            eigenvalues: vec![1.0; d],
            eigenvectors,
            generation: 0,
            n_dims: d,
        }
    }
}

/// CMA-ES optimiser for hyperparameter optimisation.
#[derive(Debug, Clone)]
pub struct EvolutionaryStrategy {
    /// CMA-ES state.
    pub state: CmaHpoState,
    /// Configuration.
    pub config: CmaHpoConfig,
    /// Recombination weights.
    weights: Vec<f64>,
    /// Effective number of parents μ_eff.
    mu_eff: f64,
}

impl EvolutionaryStrategy {
    /// Create a new CMA-ES optimiser.
    pub fn new(config: CmaHpoConfig, initial_mean: Vec<f64>) -> Self {
        assert_eq!(config.n_dims, initial_mean.len());
        let lambda = config.lambda;
        let mu = lambda / 2;
        // Recombination weights (log-linear)
        let weights: Vec<f64> = (0..mu).map(|i| (mu as f64 + 0.5 - i as f64).ln()).collect();
        let w_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let state = CmaHpoState::new(initial_mean, config.sigma0);
        Self {
            state,
            config,
            weights,
            mu_eff,
        }
    }

    /// Sample λ candidate solutions.
    pub fn sample(&self, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let d = self.state.n_dims;
        let lambda = self.config.lambda;
        (0..lambda).map(|_| self.sample_one(rng, d)).collect()
    }

    fn sample_one(&self, rng: &mut StdRng, d: usize) -> Vec<f64> {
        // z ~ N(0, I)
        let z: Vec<f64> = (0..d).map(|_| standard_normal(rng)).collect();
        // x = mean + sigma * B D z
        let mut bd_z = vec![0.0_f64; d];
        for i in 0..d {
            let mut sum = 0.0;
            for j in 0..d {
                sum += self.state.eigenvectors[j][i] * self.state.eigenvalues[j].sqrt() * z[j];
            }
            bd_z[i] = sum;
        }
        (0..d)
            .map(|i| self.state.mean[i] + self.state.sigma * bd_z[i])
            .collect()
    }

    /// Update the CMA-ES state given `selected` (mu best solutions) and `weights`.
    pub fn update(&mut self, selected: &[Vec<f64>]) {
        let d = self.state.n_dims;
        let mu = self.weights.len().min(selected.len());
        if mu == 0 {
            return;
        }

        // Hansen (2016) CMA-ES constants
        let cc = (4.0 + self.mu_eff / d as f64) / (d as f64 + 4.0 + 2.0 * self.mu_eff / d as f64);
        let c_sigma = (self.mu_eff + 2.0) / (d as f64 + self.mu_eff + 5.0);
        let c1 = 2.0 / ((d as f64 + 1.3).powi(2) + self.mu_eff);
        let c_mu = (2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff))
            / ((d as f64 + 2.0).powi(2) + self.mu_eff);
        let d_sigma =
            1.0 + 2.0 * (((self.mu_eff - 1.0) / (d as f64 + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let chi_n =
            (d as f64).sqrt() * (1.0 - 1.0 / (4.0 * d as f64) + 1.0 / (21.0 * d as f64 * d as f64));

        // New mean
        let mut new_mean = vec![0.0_f64; d];
        for (w, x) in self.weights.iter().zip(selected[..mu].iter()) {
            for k in 0..d {
                new_mean[k] += w * x[k];
            }
        }

        // Step δ = (new_mean - old_mean) / sigma
        let delta: Vec<f64> = (0..d)
            .map(|i| (new_mean[i] - self.state.mean[i]) / self.state.sigma)
            .collect();

        // B^T delta (in eigenvector basis)
        let bt_delta: Vec<f64> = (0..d)
            .map(|i| {
                (0..d)
                    .map(|j| self.state.eigenvectors[i][j] * delta[j])
                    .sum::<f64>()
            })
            .collect();

        // Update evolution path p_sigma
        let sq_mu_eff = self.mu_eff.sqrt();
        let h_sigma_val = {
            let norm_p: f64 = self.state.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();
            let threshold = (1.4 + 2.0 / (d as f64 + 1.0)) * chi_n;
            if norm_p / (1.0 - (1.0 - c_sigma).powi(2 * (self.state.generation + 1) as i32)).sqrt()
                < threshold
            {
                1.0
            } else {
                0.0
            }
        };
        for i in 0..d {
            let d_inv_bt = bt_delta[i] / self.state.eigenvalues[i].sqrt().max(1e-15);
            self.state.p_sigma[i] = (1.0 - c_sigma) * self.state.p_sigma[i]
                + (c_sigma * (2.0 - c_sigma) * self.mu_eff).sqrt() * d_inv_bt;
        }
        // Update p_c
        for i in 0..d {
            self.state.p_c[i] = (1.0 - cc) * self.state.p_c[i]
                + h_sigma_val * (cc * (2.0 - cc) * self.mu_eff).sqrt() * delta[i];
        }
        // Update covariance
        let c1_term: Vec<f64> = (0..d * d)
            .map(|idx| {
                let r = idx / d;
                let c = idx % d;
                self.state.p_c[r] * self.state.p_c[c]
            })
            .collect();
        for idx in 0..d * d {
            let r = idx / d;
            let c = idx % d;
            let c_mu_sum: f64 = (0..mu)
                .map(|k| {
                    let di = (selected[k][r] - self.state.mean[r]) / self.state.sigma;
                    let dj = (selected[k][c] - self.state.mean[c]) / self.state.sigma;
                    self.weights[k] * di * dj
                })
                .sum();
            self.state.cov[idx] =
                (1.0 - c1 - c_mu) * self.state.cov[idx] + c1 * c1_term[idx] + c_mu * c_mu_sum;
        }
        // Update sigma via CSA
        let norm_ps: f64 = self.state.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();
        self.state.sigma *= ((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0)).exp();

        // Eigen-decomposition (symmetric power iteration — simplified)
        let (evecs, evals) = eigen_decompose_sym(&self.state.cov, d);
        self.state.eigenvectors = evecs;
        self.state.eigenvalues = evals;
        self.state.mean = new_mean;
        self.state.generation += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7. Multi-Objective HPO
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-objective HPO.
#[derive(Debug, Clone)]
pub struct MoHpoConfig {
    /// Number of objectives.
    pub n_objectives: usize,
    /// Total number of evaluation iterations.
    pub n_iter: usize,
}

/// A multi-objective observation.
#[derive(Debug, Clone)]
pub struct MoObservation {
    /// Parameter values.
    pub params: Vec<f64>,
    /// Objective values (one per objective; all are to be minimised).
    pub objectives: Vec<f64>,
}

/// Return indices of Pareto-optimal observations (non-dominated set).
pub fn pareto_front(observations: &[MoObservation]) -> Vec<usize> {
    let n = observations.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Check if j dominates i (j ≤ i in all objectives, strictly in at least one)
            let all_le = observations[j]
                .objectives
                .iter()
                .zip(observations[i].objectives.iter())
                .all(|(&oj, &oi)| oj <= oi);
            let any_lt = observations[j]
                .objectives
                .iter()
                .zip(observations[i].objectives.iter())
                .any(|(&oj, &oi)| oj < oi);
            if all_le && any_lt {
                dominated[i] = true;
                break;
            }
        }
    }
    (0..n).filter(|&i| !dominated[i]).collect()
}

/// Compute approximate hypervolume contribution of each observation w.r.t. a reference point.
///
/// Uses a simple bounding-box approximation: contribution of point i is
/// the product of (ref - obj_i) over all objectives (bounded from below at 0).
pub fn hypervolume_contribution(obs: &[MoObservation], ref_point: &[f64]) -> Vec<f64> {
    obs.iter()
        .map(|o| {
            o.objectives
                .iter()
                .zip(ref_point.iter())
                .map(|(&obj, &r)| (r - obj).max(0.0))
                .product()
        })
        .collect()
}

/// NSGA-II crowding-distance selection: return `n` indices.
pub fn nsga2_select(obs: &[MoObservation], n: usize) -> Vec<usize> {
    if obs.is_empty() || n == 0 {
        return vec![];
    }
    // Non-dominated sorting
    let mut remaining: Vec<usize> = (0..obs.len()).collect();
    let mut result: Vec<usize> = Vec::with_capacity(n);
    while result.len() < n && !remaining.is_empty() {
        // Find Pareto front within remaining
        let front_indices = {
            let subset: Vec<MoObservation> = remaining.iter().map(|&i| obs[i].clone()).collect();
            pareto_front(&subset)
                .into_iter()
                .map(|local_i| remaining[local_i])
                .collect::<Vec<_>>()
        };
        let need = n - result.len();
        if front_indices.len() <= need {
            result.extend_from_slice(&front_indices);
        } else {
            // Sort front by crowding distance (descending) and take `need`
            let mut cd = crowding_distance(obs, &front_indices);
            cd.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            result.extend(cd.into_iter().take(need).map(|(i, _)| i));
        }
        let front_set: std::collections::HashSet<usize> = front_indices.iter().cloned().collect();
        remaining.retain(|i| !front_set.contains(i));
    }
    result
}

fn crowding_distance(obs: &[MoObservation], indices: &[usize]) -> Vec<(usize, f64)> {
    let m = obs.first().map(|o| o.objectives.len()).unwrap_or(0);
    let n = indices.len();
    let mut distances = vec![0.0_f64; n];
    for obj_idx in 0..m {
        let mut sorted: Vec<(usize, f64)> = indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (local, obs[global].objectives[obj_idx]))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances[sorted[0].0] = f64::INFINITY;
        distances[sorted[n - 1].0] = f64::INFINITY;
        let f_min = sorted[0].1;
        let f_max = sorted[n - 1].1;
        let range = (f_max - f_min).max(1e-15);
        for k in 1..n - 1 {
            distances[sorted[k].0] += (sorted[k + 1].1 - sorted[k - 1].1) / range;
        }
    }
    indices
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, distances[local]))
        .collect()
}

/// Multi-objective HPO optimiser.
#[derive(Debug, Clone)]
pub struct MultiObjectiveHpo {
    /// Configuration.
    pub config: MoHpoConfig,
    /// Hyperparameter space.
    pub space: HpSpace,
    /// All recorded observations.
    pub observations: Vec<MoObservation>,
}

impl MultiObjectiveHpo {
    /// Create a new multi-objective HPO instance.
    pub fn new(space: HpSpace, config: MoHpoConfig) -> Self {
        Self {
            config,
            space,
            observations: vec![],
        }
    }

    /// Add an observation.
    pub fn observe(&mut self, params: Vec<f64>, objectives: Vec<f64>) {
        self.observations.push(MoObservation { params, objectives });
    }

    /// Return the current Pareto front.
    pub fn pareto_front(&self) -> Vec<usize> {
        pareto_front(&self.observations)
    }

    /// Suggest the next candidate (random for now — can be extended with SMS-EGO etc.).
    pub fn suggest(&self, rng: &mut StdRng) -> Vec<f64> {
        self.space.sample_random(rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8. Early Termination
// ─────────────────────────────────────────────────────────────────────────────

/// Median-stopping rule: stop a trial if its value is below the median of
/// all other trials at the same step.
#[derive(Debug, Clone)]
pub struct MedianStopping {
    /// Minimum number of steps before considering stopping.
    pub patience: usize,
    /// Minimum number of completed trials required before applying the rule.
    pub min_trials: usize,
}

impl MedianStopping {
    /// Create a new median-stopping instance.
    pub fn new(patience: usize, min_trials: usize) -> Self {
        Self {
            patience,
            min_trials,
        }
    }

    /// Return `true` if the trial should be stopped.
    ///
    /// `trial_curve` — sequence of metric values for the current trial (step-by-step).
    /// `all_curves` — sequences for all other completed trials.
    pub fn should_stop(&self, trial_curve: &[f64], all_curves: &[Vec<f64>]) -> bool {
        let step = trial_curve.len();
        if step < self.patience || all_curves.len() < self.min_trials {
            return false;
        }
        let current_val = match trial_curve.last() {
            Some(&v) => v,
            None => return false,
        };
        // Collect the best value at this step from all other trials
        let mut other_bests: Vec<f64> = all_curves
            .iter()
            .filter_map(|c| c.get(step - 1).copied())
            .collect();
        if other_bests.is_empty() {
            return false;
        }
        other_bests.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = percentile_sorted(&other_bests, 50.0);
        // Stop if current value is below median (assuming higher is better)
        current_val < median
    }
}

/// Successive halving early-termination rule.
#[derive(Debug, Clone)]
pub struct SuccessiveHalving {
    /// Minimum budget allocated per trial in round 0.
    pub min_budget: f64,
    /// Maximum budget.
    pub max_budget: f64,
    /// Halving factor η.
    pub eta: f64,
}

impl SuccessiveHalving {
    /// Create a new successive halving instance.
    pub fn new(min_budget: f64, max_budget: f64, eta: f64) -> Self {
        Self {
            min_budget,
            max_budget,
            eta,
        }
    }

    /// Return the budget for a given round index.
    pub fn budget_for_round(&self, round: usize) -> f64 {
        (self.min_budget * self.eta.powi(round as i32)).min(self.max_budget)
    }

    /// Total number of rounds from min to max budget.
    pub fn n_rounds(&self) -> usize {
        (self.max_budget / self.min_budget)
            .max(1.0)
            .log(self.eta)
            .ceil() as usize
    }
}

/// Percentile-based early stopping: stop if the current value is below
/// the `percentile`-th percentile of historical values at the same step.
#[derive(Debug, Clone)]
pub struct PercentileStop {
    /// Percentile threshold (0-100).
    pub percentile: f64,
    /// Minimum number of steps before this rule applies.
    pub min_steps: usize,
}

impl PercentileStop {
    /// Create a new percentile-stop instance.
    pub fn new(percentile: f64, min_steps: usize) -> Self {
        Self {
            percentile,
            min_steps,
        }
    }

    /// Return `true` if `value` at `step` is below the `percentile` of `history`.
    pub fn should_stop_percentile(&self, value: f64, step: usize, history: &[f64]) -> bool {
        if step < self.min_steps || history.is_empty() {
            return false;
        }
        let mut sorted = history.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = percentile_sorted(&sorted, self.percentile);
        value < threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9. HpoLogger / HpoStudy
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a single trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialStatus {
    /// Trial is currently running.
    Running,
    /// Trial completed successfully.
    Complete,
    /// Trial was pruned by early-termination.
    Pruned,
    /// Trial failed due to an error.
    Failed,
}

/// A single HPO trial record.
#[derive(Debug, Clone)]
pub struct HpoTrial {
    /// Unique trial ID.
    pub id: usize,
    /// Hyperparameter values.
    pub params: Vec<f64>,
    /// Names corresponding to `params`.
    pub param_names: Vec<String>,
    /// Observed objective value.
    pub value: f64,
    /// Trial status.
    pub status: TrialStatus,
    /// Start time (milliseconds since epoch, or arbitrary counter).
    pub start_ms: u64,
    /// End time.
    pub end_ms: u64,
}

impl HpoTrial {
    /// Create a new trial record.
    pub fn new(
        id: usize,
        params: Vec<f64>,
        param_names: Vec<String>,
        value: f64,
        status: TrialStatus,
    ) -> Self {
        Self {
            id,
            params,
            param_names,
            value,
            status,
            start_ms: 0,
            end_ms: 0,
        }
    }
}

/// Direction of optimisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptDirection {
    /// Minimise the objective.
    Minimize,
    /// Maximise the objective.
    Maximize,
}

/// An HPO study, collecting trials and providing analysis.
#[derive(Debug, Clone)]
pub struct HpoStudy {
    /// Study name.
    pub name: String,
    /// All recorded trials.
    pub trials: Vec<HpoTrial>,
    /// Optimisation direction.
    pub direction: OptDirection,
}

impl HpoStudy {
    /// Create a new study.
    pub fn new(name: impl Into<String>, direction: OptDirection) -> Self {
        Self {
            name: name.into(),
            trials: vec![],
            direction,
        }
    }

    /// Add a completed trial.
    pub fn add_trial(&mut self, trial: HpoTrial) {
        self.trials.push(trial);
    }

    /// Return the best trial.
    pub fn best_trial(&self) -> Option<&HpoTrial> {
        let complete: Vec<&HpoTrial> = self
            .trials
            .iter()
            .filter(|t| t.status == TrialStatus::Complete)
            .collect();
        match self.direction {
            OptDirection::Minimize => complete.into_iter().min_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            OptDirection::Maximize => complete.into_iter().max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
        }
    }
}

/// Stand-alone helper: return the best trial from a study.
pub fn best_trial(study: &HpoStudy) -> Option<&HpoTrial> {
    study.best_trial()
}

/// Compute simplified fANOVA-style importances by measuring the variance of
/// the objective when each parameter is varied independently.
///
/// For each parameter dim `k`, the importance is:
///   Var[ E[y | x_k] ] / Var\[y\]
///
/// Approximated by binning parameter `k` into 5 equal-width buckets and
/// computing the variance of bucket means.
pub fn importance_by_fanova(study: &HpoStudy) -> Vec<(String, f64)> {
    let complete: Vec<&HpoTrial> = study
        .trials
        .iter()
        .filter(|t| t.status == TrialStatus::Complete)
        .collect();
    if complete.is_empty() {
        return vec![];
    }
    let n_params = complete[0].params.len();
    let values: Vec<f64> = complete.iter().map(|t| t.value).collect();
    let total_var = variance(&values);
    if total_var < 1e-15 {
        return complete[0]
            .param_names
            .iter()
            .map(|n| (n.clone(), 0.0))
            .collect();
    }
    let n_buckets = 5usize;
    (0..n_params)
        .map(|k| {
            let name = complete
                .first()
                .and_then(|t| t.param_names.get(k))
                .cloned()
                .unwrap_or_else(|| format!("param_{k}"));
            let param_vals: Vec<f64> = complete.iter().map(|t| t.params[k]).collect();
            let p_min = param_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let p_max = param_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (p_max - p_min).max(1e-15);
            let mut bucket_means = vec![];
            for b in 0..n_buckets {
                let lo = p_min + (b as f64 / n_buckets as f64) * range;
                let hi = p_min + ((b + 1) as f64 / n_buckets as f64) * range;
                let bucket_vals: Vec<f64> = complete
                    .iter()
                    .filter(|t| {
                        let pv = t.params[k];
                        pv >= lo && (pv < hi || b == n_buckets - 1)
                    })
                    .map(|t| t.value)
                    .collect();
                if !bucket_vals.is_empty() {
                    bucket_means.push(mean(&bucket_vals));
                }
            }
            let importance = if bucket_means.len() > 1 {
                variance(&bucket_means) / total_var
            } else {
                0.0
            };
            (name, importance)
        })
        .collect()
}

/// Convenience struct wrapping study-level logging functionality.
#[derive(Debug, Clone, Default)]
pub struct HpoLogger {
    /// Internal study.
    pub study: Option<HpoStudy>,
}

impl HpoLogger {
    /// Create a logger with a new study.
    pub fn new(name: impl Into<String>, direction: OptDirection) -> Self {
        Self {
            study: Some(HpoStudy::new(name, direction)),
        }
    }

    /// Log a new trial.
    pub fn log_trial(&mut self, trial: HpoTrial) {
        if let Some(s) = &mut self.study {
            s.add_trial(trial);
        }
    }

    /// Return the best trial across all logged runs.
    pub fn best(&self) -> Option<&HpoTrial> {
        self.study.as_ref().and_then(|s| s.best_trial())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10. Transfer-Learning HPO (Warm Starting)
// ─────────────────────────────────────────────────────────────────────────────

/// A previous HPO study used for warm starting.
#[derive(Debug, Clone)]
pub struct PreviousStudy {
    /// Names of the parameters in this study.
    pub param_names: Vec<String>,
    /// Observed (params, score) pairs.
    pub trials: Vec<(Vec<f64>, f64)>,
}

impl PreviousStudy {
    /// Construct a previous study.
    pub fn new(param_names: Vec<String>, trials: Vec<(Vec<f64>, f64)>) -> Self {
        Self {
            param_names,
            trials,
        }
    }
}

/// Warm-start sampler — uses top results from previous studies as initial
/// candidates for a new study.
#[derive(Debug, Clone)]
pub struct WarmStartSampler {
    /// Previously completed studies.
    pub previous: Vec<PreviousStudy>,
}

impl WarmStartSampler {
    /// Create a new sampler from a set of previous studies.
    pub fn new(previous: Vec<PreviousStudy>) -> Self {
        Self { previous }
    }

    /// Map a parameter vector from a source study into the target space by
    /// matching parameters by name and normalising.
    pub fn map_parameters(
        params: &[f64],
        source_names: &[String],
        target_space: &HpSpace,
    ) -> Vec<f64> {
        target_space
            .params
            .iter()
            .map(|hp| {
                // Find matching dimension in source by name
                source_names
                    .iter()
                    .position(|n| n == &hp.name)
                    .and_then(|idx| params.get(idx).copied())
                    .map(|v| {
                        // Re-normalise to target bounds: assume source already in target range
                        v.clamp(hp.bounds.0, hp.bounds.1)
                    })
                    .unwrap_or_else(|| {
                        // Default: midpoint of target bounds
                        (hp.bounds.0 + hp.bounds.1) / 2.0
                    })
            })
            .collect()
    }

    /// Return the top-`n` warm-start configurations from all previous studies.
    /// They are projected into the `target_space` by name-matching.
    pub fn select_warm_starts(&self, target_space: &HpSpace, n: usize) -> Vec<Vec<f64>> {
        // Collect all (projected_params, score) from previous studies
        let mut all: Vec<(Vec<f64>, f64)> = self
            .previous
            .iter()
            .flat_map(|study| {
                study.trials.iter().map(|(params, score)| {
                    let mapped = Self::map_parameters(params, &study.param_names, target_space);
                    (mapped, *score)
                })
            })
            .collect();
        // Sort by score descending
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.into_iter().take(n).map(|(p, _)| p).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky decomposition: A = L L^T.  Returns L (lower triangular).
fn cholesky(a: &[Vec<f64>]) -> std::result::Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let val = a[i][i] - sum;
                if val < 0.0 {
                    return Err(format!("Matrix not positive definite at ({i},{i}): {val}"));
                }
                l[i][j] = val.sqrt();
            } else {
                let ljj = l[j][j];
                if ljj.abs() < 1e-300 {
                    return Err("Zero diagonal in Cholesky".into());
                }
                l[i][j] = (a[i][j] - sum) / ljj;
            }
        }
    }
    Ok(l)
}

/// Solve L x = b (forward substitution, L lower triangular).
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|j| l[i][j] * x[j]).sum();
        let lii = l[i][i];
        x[i] = if lii.abs() < 1e-300 {
            0.0
        } else {
            (b[i] - sum) / lii
        };
    }
    x
}

/// Solve L^T x = b (backward substitution, L lower triangular).
fn backward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| l[j][i] * x[j]).sum();
        let lii = l[i][i];
        x[i] = if lii.abs() < 1e-300 {
            0.0
        } else {
            (b[i] - sum) / lii
        };
    }
    x
}

/// Solve (L L^T) x = b via Cholesky (forward then backward substitution).
fn chol_solve(l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let y = forward_sub(l, b);
    Ok(backward_sub(l, &y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun approximation to erf(x).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// Compute the `p`-th percentile from a pre-sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Generate a standard normal sample using the Box-Muller transform.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigen-decomposition helper (Jacobi iterations for symmetric matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Symmetric Jacobi eigen-decomposition for a d×d matrix stored row-major.
/// Returns (eigenvectors row-major d×d, eigenvalues d).
fn eigen_decompose_sym(a_flat: &[f64], d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut a: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| a_flat[i * d + j]).collect())
        .collect();
    // Identity eigenvector matrix
    let mut v: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            let mut row = vec![0.0_f64; d];
            row[i] = 1.0;
            row
        })
        .collect();
    let max_iter = 100 * d * d;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0_f64;
        for i in 0..d {
            for j in i + 1..d {
                let val = a[i][j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-10 {
            break;
        }
        // Compute rotation angle
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };
        let (s, c) = (theta.sin(), theta.cos());
        // Rotate rows/cols p, q
        let app = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        let aqq = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        a[p][p] = app;
        a[q][q] = aqq;
        for r in 0..d {
            if r != p && r != q {
                let apr = c * a[p][r] - s * a[q][r];
                let aqr = s * a[p][r] + c * a[q][r];
                a[p][r] = apr;
                a[r][p] = apr;
                a[q][r] = aqr;
                a[r][q] = aqr;
            }
        }
        // Update eigenvectors
        for r in 0..d {
            let vpr = c * v[r][p] - s * v[r][q];
            let vqr = s * v[r][p] + c * v[r][q];
            v[r][p] = vpr;
            v[r][q] = vqr;
        }
    }
    let eigenvalues: Vec<f64> = (0..d).map(|i| a[i][i]).collect();
    (v, eigenvalues)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
