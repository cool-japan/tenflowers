//! Bayesian / Variational Neural Networks — Track Y.
//!
//! Implements uncertainty quantification via weight distributions using the
//! Bayes-by-Backprop framework (Blundell et al., 2015).  Every weight is
//! treated as a Gaussian random variable parameterised by `(μ, σ)`.  The
//! local reparameterization trick is used so that the stochasticity acts on
//! pre-activations rather than individual weights, which reduces variance of
//! gradient estimators.
//!
//! # Mathematical background
//!
//! Weight posterior: `w ~ N(μ, σ²)` where `σ = softplus(ρ)` and `ρ` is the
//! learnable log-sigma parameter stored as `weight_log_sigma`.
//!
//! Prior: `p(w) = N(0, prior_σ²)`.
//!
//! The ELBO objective is:
//! ```text
//! L = E_q[log p(D|w)] - KL(q(w) ‖ p(w))
//!   = -NLL - KL
//! ```
//! In practice we minimise `NLL + kl_weight * KL`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::bayesian::{BayesianLinear, BayesianMlp};
//!
//! let layer = BayesianLinear::new(4, 2, 1.0)?;
//! let out   = layer.forward_sample(&[1.0, 0.5, -1.0, 0.3], 42)?;
//! assert_eq!(out.len(), 2);
//!
//! let mlp = BayesianMlp::new(&[(4, 8), (8, 2)], 1.0)?;
//! let (mean, var) = mlp.predict_with_uncertainty(&[1.0, 0.5, -1.0, 0.3], 64, 0)?;
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Mathematical primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Softplus activation: `log(1 + exp(x))`.
///
/// Numerically stable, always-positive implementation:
/// - For `x > 20`:  returns `x` (difference from true softplus < 2e-9 in f32).
/// - For `x < -20`: returns `exp(x)` which is tiny but strictly positive,
///   avoiding the f32 underflow that would make `log(1 + exp(x))` collapse
///   to `log(1) = 0`.
/// - Otherwise: uses the direct formula `log(1 + exp(x))`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        // log(1 + exp(x)) ≈ exp(x) for x << 0; always > 0.
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Convert a raw `log_sigma` parameter to a positive `sigma` using softplus.
///
/// Using softplus instead of `exp` avoids unbounded growth for large positive
/// values and keeps sigma strictly positive.
#[inline]
pub fn log_sigma_to_sigma(log_sigma: f32) -> f32 {
    softplus(log_sigma)
}

/// Analytic KL divergence between two univariate Gaussians:
/// `KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²))`.
///
/// The formula is:
/// ```text
/// KL = log(σ₂/σ₁) + (σ₁² + (μ₁ − μ₂)²) / (2σ₂²) − 1/2
/// ```
///
/// Returns `0.0` when both distributions are identical (within floating-point
/// precision).  Panics are impossible; instead the function clamps `sigma`
/// values to `f32::EPSILON` to avoid division by zero.
pub fn kl_gaussian(mu1: f32, sigma1: f32, mu2: f32, sigma2: f32) -> f32 {
    let s1 = sigma1.max(f32::EPSILON);
    let s2 = sigma2.max(f32::EPSILON);
    let log_ratio = (s2 / s1).ln();
    let mu_diff = mu1 - mu2;
    log_ratio + (s1 * s1 + mu_diff * mu_diff) / (2.0 * s2 * s2) - 0.5
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal sampling (Box-Muller, seeded)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` i.i.d. N(0,1) samples using the Box-Muller transform with a
/// deterministic seed.  Pairs are discarded when `n` is odd.
fn normal_samples(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = (r * theta.cos()) as f32;
        let z1 = (r * theta.sin()) as f32;
        out.push(z0);
        i += 1;
        if i < n {
            out.push(z1);
            i += 1;
        }
    }
    out
}

/// Xavier / Glorot uniform initialisation for a weight matrix of shape
/// `[fan_in, fan_out]`.
fn xavier_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let n = fan_in * fan_out;
    (0..n)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ReLU helper for MLP hidden layers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// BayesianLinear
// ─────────────────────────────────────────────────────────────────────────────

/// Variational linear layer: weights ~ N(μ, σ²).
///
/// Uses the **local reparameterization trick**: rather than sampling each
/// weight independently we sample the pre-activation directly.
///
/// For input `x` (shape `[in_features]`) the pre-activation for output unit
/// `j` is:
/// ```text
/// act_j ~ N(Σ_i x_i · μ_ij + b_μ_j,
///           Σ_i x_i² · σ_ij² + σ_b_j²)
/// ```
/// where `σ = softplus(log_sigma)`.
///
/// This is equivalent to sampling weights but much cheaper for large layers.
#[derive(Debug, Clone)]
pub struct BayesianLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,

    // Weight posterior — stored row-major: index = out * in_features + in
    /// Weight means, shape `[out_features × in_features]`.
    pub weight_mu: Vec<f32>,
    /// Weight log-sigma (raw), same shape as `weight_mu`.
    /// Actual sigma = softplus(weight_log_sigma).
    pub weight_log_sigma: Vec<f32>,

    // Bias posterior
    /// Bias means, shape `[out_features]`.
    pub bias_mu: Vec<f32>,
    /// Bias log-sigma (raw), shape `[out_features]`.
    pub bias_log_sigma: Vec<f32>,

    /// Prior standard deviation (N(0, prior_sigma²)).
    pub prior_sigma: f32,
}

impl BayesianLinear {
    /// Construct a new `BayesianLinear` layer.
    ///
    /// Weight means are Xavier-initialised; `log_sigma` parameters are
    /// initialised to `-3.0` so that initial sigmas are small (~0.05),
    /// matching the common practice of starting close to a MAP solution.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `in_features == 0`, `out_features == 0`,
    /// or `prior_sigma <= 0`.
    pub fn new(in_features: usize, out_features: usize, prior_sigma: f32) -> Result<Self> {
        if in_features == 0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianLinear::new",
                "in_features must be > 0",
            ));
        }
        if out_features == 0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianLinear::new",
                "out_features must be > 0",
            ));
        }
        if prior_sigma <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianLinear::new",
                "prior_sigma must be > 0",
            ));
        }

        let n_weights = out_features * in_features;

        // Xavier init for weight means; bias means = 0.
        let weight_mu = xavier_uniform(in_features, out_features, 12345);
        let weight_log_sigma = vec![-3.0_f32; n_weights];
        let bias_mu = vec![0.0_f32; out_features];
        let bias_log_sigma = vec![-3.0_f32; out_features];

        Ok(Self {
            in_features,
            out_features,
            weight_mu,
            weight_log_sigma,
            bias_mu,
            bias_log_sigma,
            prior_sigma,
        })
    }

    /// Local reparameterization forward pass.
    ///
    /// Pre-activation mean and variance are computed analytically:
    /// ```text
    /// act_mean_j = Σ_i x_i · μ_{j,i} + μ_{b,j}
    /// act_var_j  = Σ_i x_i² · σ_{j,i}² + σ_{b,j}²
    /// act_j      = act_mean_j + ε_j · sqrt(act_var_j),  ε ~ N(0,1)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `input.len() != in_features`.
    pub fn forward_sample(&self, input: &[f32], seed: u64) -> Result<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(TensorError::invalid_argument_op(
                "BayesianLinear::forward_sample",
                &format!(
                    "input length {} does not match in_features {}",
                    input.len(),
                    self.in_features
                ),
            ));
        }

        let out_dim = self.out_features;
        let in_dim = self.in_features;

        // Compute mean and variance of each pre-activation.
        let mut act_mean = self.bias_mu.clone();
        let mut act_var: Vec<f32> = self
            .bias_log_sigma
            .iter()
            .map(|&ls| {
                let s = log_sigma_to_sigma(ls);
                s * s
            })
            .collect();

        for o in 0..out_dim {
            for i in 0..in_dim {
                let idx = o * in_dim + i;
                let mu_w = self.weight_mu[idx];
                let s_w = log_sigma_to_sigma(self.weight_log_sigma[idx]);
                let xi = input[i];
                act_mean[o] += xi * mu_w;
                act_var[o] += xi * xi * s_w * s_w;
            }
        }

        // Sample epsilon and compute output.
        let epsilons = normal_samples(out_dim, seed);
        let output: Vec<f32> = act_mean
            .iter()
            .zip(act_var.iter())
            .zip(epsilons.iter())
            .map(|((&m, &v), &eps)| m + eps * v.max(0.0).sqrt())
            .collect();

        Ok(output)
    }

    /// Monte-Carlo forward: run `n_samples` independent stochastic passes.
    ///
    /// Each sample uses `seed + sample_idx` to ensure decorrelated randomness
    /// across samples while remaining fully deterministic.
    ///
    /// # Errors
    ///
    /// Propagates errors from `forward_sample`.
    pub fn forward_mc(&self, input: &[f32], n_samples: usize, seed: u64) -> Result<Vec<Vec<f32>>> {
        if n_samples == 0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianLinear::forward_mc",
                "n_samples must be > 0",
            ));
        }
        let mut results = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let sample_seed = seed.wrapping_add(s as u64);
            results.push(self.forward_sample(input, sample_seed)?);
        }
        Ok(results)
    }

    /// Predict with uncertainty via Monte-Carlo integration.
    ///
    /// Returns `(mean, variance)` where both have length `out_features`.
    /// Variance is the unbiased estimator across `n_samples` MC draws.
    ///
    /// # Errors
    ///
    /// Propagates errors from `forward_mc`.
    pub fn predict_with_uncertainty(
        &self,
        input: &[f32],
        n_samples: usize,
        seed: u64,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let samples = self.forward_mc(input, n_samples, seed)?;
        let out_dim = self.out_features;
        let n = n_samples as f32;

        // Welford-style single-pass mean and variance.
        let mut mean = vec![0.0_f32; out_dim];
        let mut m2 = vec![0.0_f32; out_dim];

        for (k, sample) in samples.iter().enumerate() {
            let kf = (k + 1) as f32;
            for j in 0..out_dim {
                let delta = sample[j] - mean[j];
                mean[j] += delta / kf;
                let delta2 = sample[j] - mean[j];
                m2[j] += delta * delta2;
            }
        }

        // Unbiased variance: M2 / (n - 1); fall back to 0 for n=1.
        let var: Vec<f32> = m2
            .iter()
            .map(|&s| if n > 1.0 { s / (n - 1.0) } else { 0.0 })
            .collect();

        Ok((mean, var))
    }

    /// Analytic KL divergence of the full weight + bias posterior from the
    /// N(0, prior_sigma²) prior.
    ///
    /// `KL = Σ_{all params} KL(N(μ_i, σ_i²) ‖ N(0, prior_σ²))`
    pub fn kl_divergence(&self) -> f32 {
        let prior_s = self.prior_sigma;

        let weight_kl: f32 = self
            .weight_mu
            .iter()
            .zip(self.weight_log_sigma.iter())
            .map(|(&mu, &ls)| kl_gaussian(mu, log_sigma_to_sigma(ls), 0.0, prior_s))
            .sum();

        let bias_kl: f32 = self
            .bias_mu
            .iter()
            .zip(self.bias_log_sigma.iter())
            .map(|(&mu, &ls)| kl_gaussian(mu, log_sigma_to_sigma(ls), 0.0, prior_s))
            .sum();

        weight_kl + bias_kl
    }

    /// Total number of variational parameters (2 per weight + 2 per bias).
    pub fn num_params(&self) -> usize {
        let n_weights = self.in_features * self.out_features;
        let n_biases = self.out_features;
        2 * (n_weights + n_biases)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BayesianMlp
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer Bayesian MLP: a stack of `BayesianLinear` layers with ReLU
/// activations on all but the final layer.
#[derive(Debug, Clone)]
pub struct BayesianMlp {
    layers: Vec<BayesianLinear>,
    /// Shared prior standard deviation for all layers.
    pub prior_sigma: f32,
}

impl BayesianMlp {
    /// Construct a `BayesianMlp` from a slice of `(in_dim, out_dim)` pairs.
    ///
    /// At least one layer is required.  Each consecutive pair must be
    /// compatible (i.e. `out_dim[k] == in_dim[k+1]` — callers are responsible
    /// for this).
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `layer_dims` is empty or `prior_sigma <= 0`.
    pub fn new(layer_dims: &[(usize, usize)], prior_sigma: f32) -> Result<Self> {
        if layer_dims.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "BayesianMlp::new",
                "layer_dims must contain at least one layer",
            ));
        }
        if prior_sigma <= 0.0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianMlp::new",
                "prior_sigma must be > 0",
            ));
        }

        let mut layers = Vec::with_capacity(layer_dims.len());
        for (i, &(in_dim, out_dim)) in layer_dims.iter().enumerate() {
            let layer = BayesianLinear::new(in_dim, out_dim, prior_sigma).map_err(|e| {
                TensorError::invalid_argument_op("BayesianMlp::new", &format!("layer {i}: {e}"))
            })?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            prior_sigma,
        })
    }

    /// Forward pass through all layers with ReLU on hidden layers.
    ///
    /// Layer seeds are derived as `seed + layer_index * SEED_STRIDE` to keep
    /// each layer's noise decorrelated.
    ///
    /// # Errors
    ///
    /// Propagates errors from any `BayesianLinear::forward_sample`.
    pub fn forward_sample(&self, input: &[f32], seed: u64) -> Result<Vec<f32>> {
        const SEED_STRIDE: u64 = 0x9e37_79b9_7f4a_7c15; // large prime-ish constant
        let n_layers = self.layers.len();
        let mut current = input.to_vec();

        for (idx, layer) in self.layers.iter().enumerate() {
            let layer_seed = seed.wrapping_add((idx as u64).wrapping_mul(SEED_STRIDE));
            let mut out = layer.forward_sample(&current, layer_seed)?;
            // Apply ReLU on all but the last layer.
            if idx < n_layers - 1 {
                for v in out.iter_mut() {
                    *v = relu(*v);
                }
            }
            current = out;
        }

        Ok(current)
    }

    /// Monte-Carlo forward across `n_samples` stochastic passes.
    ///
    /// # Errors
    ///
    /// Propagates errors from `forward_sample`.
    pub fn forward_mc(&self, input: &[f32], n_samples: usize, seed: u64) -> Result<Vec<Vec<f32>>> {
        if n_samples == 0 {
            return Err(TensorError::invalid_argument_op(
                "BayesianMlp::forward_mc",
                "n_samples must be > 0",
            ));
        }
        let mut results = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let sample_seed = seed.wrapping_add(s as u64);
            results.push(self.forward_sample(input, sample_seed)?);
        }
        Ok(results)
    }

    /// Predict with MC uncertainty — returns `(mean, variance)`.
    ///
    /// # Errors
    ///
    /// Propagates errors from `forward_mc`.
    pub fn predict_with_uncertainty(
        &self,
        input: &[f32],
        n_samples: usize,
        seed: u64,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let samples = self.forward_mc(input, n_samples, seed)?;
        let out_dim = match self.layers.last() {
            Some(l) => l.out_features,
            None => {
                return Err(TensorError::invalid_argument_op(
                    "BayesianMlp::predict_with_uncertainty",
                    "MLP has no layers",
                ))
            }
        };

        let n = n_samples as f32;
        let mut mean = vec![0.0_f32; out_dim];
        let mut m2 = vec![0.0_f32; out_dim];

        for (k, sample) in samples.iter().enumerate() {
            let kf = (k + 1) as f32;
            for j in 0..out_dim {
                let delta = sample[j] - mean[j];
                mean[j] += delta / kf;
                let delta2 = sample[j] - mean[j];
                m2[j] += delta * delta2;
            }
        }

        let var: Vec<f32> = m2
            .iter()
            .map(|&s| if n > 1.0 { s / (n - 1.0) } else { 0.0 })
            .collect();

        Ok((mean, var))
    }

    /// Sum of KL divergences across all layers.
    pub fn total_kl(&self) -> f32 {
        self.layers.iter().map(|l| l.kl_divergence()).sum()
    }

    /// ELBO loss: `nll + kl_weight * total_KL`.
    ///
    /// Minimising this is equivalent to maximising the evidence lower bound.
    pub fn elbo_loss(&self, nll: f32, kl_weight: f32) -> f32 {
        nll + kl_weight * self.total_kl()
    }

    /// Number of layers in the MLP.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Uncertainty decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// **Epistemic uncertainty**: variance of predictions across Monte-Carlo
/// samples, capturing model / parameter uncertainty.
///
/// Each element of the returned vector is the sample variance across all MC
/// draws for that output unit.
///
/// # Errors
///
/// Returns `InvalidArgument` if `mc_outputs` is empty or if the output vectors
/// have inconsistent lengths.
pub fn epistemic_uncertainty(mc_outputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    if mc_outputs.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "epistemic_uncertainty",
            "mc_outputs must be non-empty",
        ));
    }

    let out_dim = mc_outputs[0].len();
    for (i, sample) in mc_outputs.iter().enumerate() {
        if sample.len() != out_dim {
            return Err(TensorError::invalid_argument_op(
                "epistemic_uncertainty",
                &format!(
                    "sample {} has length {} but sample 0 has length {}",
                    i,
                    sample.len(),
                    out_dim
                ),
            ));
        }
    }

    let n = mc_outputs.len() as f32;
    let mut mean = vec![0.0_f32; out_dim];
    let mut m2 = vec![0.0_f32; out_dim];

    for (k, sample) in mc_outputs.iter().enumerate() {
        let kf = (k + 1) as f32;
        for j in 0..out_dim {
            let delta = sample[j] - mean[j];
            mean[j] += delta / kf;
            let delta2 = sample[j] - mean[j];
            m2[j] += delta * delta2;
        }
    }

    let var: Vec<f32> = m2
        .iter()
        .map(|&s| if n > 1.0 { s / (n - 1.0) } else { 0.0 })
        .collect();

    Ok(var)
}

/// **Aleatoric uncertainty**: expected data-noise variance encoded in the
/// model's output.
///
/// This function assumes that the network outputs `2k` values per sample,
/// where the first `k` are predictions and the last `k` are log-variances of
/// the heteroscedastic noise.  It averages the exponentiated log-variances
/// over all MC samples to obtain an estimate of data uncertainty.
///
/// Returns a vector of length `k` (half of each sample's output).
///
/// # Errors
///
/// Returns `InvalidArgument` if `mc_outputs` is empty, output length is zero,
/// or output length is odd (cannot split into prediction + log-variance halves).
pub fn aleatoric_uncertainty(mc_outputs: &[Vec<f32>]) -> Result<Vec<f32>> {
    if mc_outputs.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "aleatoric_uncertainty",
            "mc_outputs must be non-empty",
        ));
    }

    let full_dim = mc_outputs[0].len();
    if full_dim == 0 {
        return Err(TensorError::invalid_argument_op(
            "aleatoric_uncertainty",
            "output dimension must be > 0",
        ));
    }
    if full_dim % 2 != 0 {
        return Err(TensorError::invalid_argument_op(
            "aleatoric_uncertainty",
            &format!(
                "output dimension must be even (got {full_dim}) — last half must be log-variances"
            ),
        ));
    }

    let half = full_dim / 2;

    for (i, sample) in mc_outputs.iter().enumerate() {
        if sample.len() != full_dim {
            return Err(TensorError::invalid_argument_op(
                "aleatoric_uncertainty",
                &format!(
                    "sample {} has length {} but sample 0 has length {}",
                    i,
                    sample.len(),
                    full_dim
                ),
            ));
        }
    }

    // Mean of exp(log_var) over MC samples.
    let n = mc_outputs.len() as f32;
    let mut mean_aleatoric = vec![0.0_f32; half];

    for sample in mc_outputs.iter() {
        for j in 0..half {
            let log_var = sample[half + j];
            // Clamp to avoid overflow.
            let log_var_safe = log_var.clamp(-20.0, 20.0);
            mean_aleatoric[j] += log_var_safe.exp() / n;
        }
    }

    Ok(mean_aleatoric)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── softplus ────────────────────────────────────────────────────────────

    #[test]
    fn softplus_always_positive() {
        for &x in &[-100.0_f32, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            assert!(
                softplus(x) > 0.0,
                "softplus({x}) should be positive, got {}",
                softplus(x)
            );
        }
    }

    #[test]
    fn softplus_at_zero_is_ln2() {
        let result = softplus(0.0);
        let expected = 2.0_f32.ln(); // ln(2) ≈ 0.6931
        assert!(
            (result - expected).abs() < 1e-5,
            "softplus(0) = {result}, expected {expected}"
        );
    }

    #[test]
    fn softplus_large_x_close_to_x() {
        // For x >> 0, softplus(x) ≈ x.
        let x = 30.0_f32;
        let result = softplus(x);
        assert!(
            (result - x).abs() < 1e-4,
            "softplus({x}) = {result}, expected close to {x}"
        );
    }

    // ── kl_gaussian ─────────────────────────────────────────────────────────

    #[test]
    fn kl_gaussian_identical_distributions_is_zero() {
        let kl = kl_gaussian(1.5, 0.8, 1.5, 0.8);
        assert!(
            kl.abs() < 1e-5,
            "KL of identical distributions should be 0, got {kl}"
        );
    }

    #[test]
    fn kl_gaussian_is_non_negative() {
        // KL divergence is always >= 0.
        for &(mu1, s1, mu2, s2) in &[
            (0.0_f32, 1.0, 0.0, 1.0),
            (1.0, 0.5, 0.0, 1.0),
            (-2.0, 2.0, 1.0, 0.5),
            (0.0, 0.1, 0.0, 2.0),
        ] {
            let kl = kl_gaussian(mu1, s1, mu2, s2);
            assert!(kl >= -1e-6, "KL({mu1},{s1}||{mu2},{s2}) = {kl} is negative");
        }
    }

    #[test]
    fn kl_gaussian_known_value() {
        // KL(N(0,1) || N(0,2)) = log(2/1) + (1 + 0) / (2*4) - 0.5
        //                      = ln2 + 1/8 - 0.5 ≈ 0.6931 + 0.125 - 0.5 = 0.3181
        let kl = kl_gaussian(0.0, 1.0, 0.0, 2.0);
        let expected = 2.0_f32.ln() + 1.0 / 8.0 - 0.5;
        assert!(
            (kl - expected).abs() < 1e-4,
            "KL(N(0,1)||N(0,2)) = {kl}, expected {expected}"
        );
    }

    // ── BayesianLinear::new ─────────────────────────────────────────────────

    #[test]
    fn bayesian_linear_new_correct_shapes() {
        let layer = BayesianLinear::new(4, 3, 1.0).expect("construction failed");
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 3);
        assert_eq!(layer.weight_mu.len(), 12);
        assert_eq!(layer.weight_log_sigma.len(), 12);
        assert_eq!(layer.bias_mu.len(), 3);
        assert_eq!(layer.bias_log_sigma.len(), 3);
    }

    #[test]
    fn bayesian_linear_new_rejects_zero_in() {
        assert!(BayesianLinear::new(0, 3, 1.0).is_err());
    }

    #[test]
    fn bayesian_linear_new_rejects_zero_out() {
        assert!(BayesianLinear::new(4, 0, 1.0).is_err());
    }

    #[test]
    fn bayesian_linear_new_rejects_non_positive_prior() {
        assert!(BayesianLinear::new(4, 3, 0.0).is_err());
        assert!(BayesianLinear::new(4, 3, -1.0).is_err());
    }

    // ── forward_sample ───────────────────────────────────────────────────────

    #[test]
    fn forward_sample_output_shape() {
        let layer = BayesianLinear::new(5, 3, 1.0).expect("construction failed");
        let input = vec![1.0_f32; 5];
        let out = layer.forward_sample(&input, 42).expect("forward failed");
        assert_eq!(out.len(), 3, "output length should match out_features");
    }

    #[test]
    fn forward_sample_different_seeds_give_different_outputs() {
        let layer = BayesianLinear::new(4, 2, 1.0).expect("construction failed");
        let input = vec![1.0_f32, -1.0, 0.5, -0.5];
        let out1 = layer.forward_sample(&input, 1).expect("fwd 1 failed");
        let out2 = layer.forward_sample(&input, 2).expect("fwd 2 failed");
        // With high probability the two samples differ (stochastic layer).
        let same = out1
            .iter()
            .zip(out2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!same, "different seeds should produce different outputs");
    }

    #[test]
    fn forward_sample_same_seed_is_deterministic() {
        let layer = BayesianLinear::new(4, 2, 1.0).expect("construction failed");
        let input = vec![1.0_f32, -1.0, 0.5, -0.5];
        let out1 = layer.forward_sample(&input, 99).expect("fwd 1 failed");
        let out2 = layer.forward_sample(&input, 99).expect("fwd 2 failed");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "same seed should give identical output"
            );
        }
    }

    #[test]
    fn forward_sample_rejects_wrong_input_len() {
        let layer = BayesianLinear::new(4, 2, 1.0).expect("construction failed");
        let bad_input = vec![1.0_f32; 3];
        assert!(layer.forward_sample(&bad_input, 0).is_err());
    }

    // ── forward_mc ───────────────────────────────────────────────────────────

    #[test]
    fn forward_mc_returns_n_samples() {
        let layer = BayesianLinear::new(3, 2, 1.0).expect("construction failed");
        let input = vec![0.5_f32; 3];
        let samples = layer.forward_mc(&input, 10, 0).expect("mc failed");
        assert_eq!(samples.len(), 10);
        for s in &samples {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn forward_mc_rejects_zero_samples() {
        let layer = BayesianLinear::new(3, 2, 1.0).expect("construction failed");
        let input = vec![0.5_f32; 3];
        assert!(layer.forward_mc(&input, 0, 0).is_err());
    }

    // ── predict_with_uncertainty ─────────────────────────────────────────────

    #[test]
    fn predict_with_uncertainty_mean_shape() {
        let layer = BayesianLinear::new(4, 5, 1.0).expect("construction failed");
        let input = vec![1.0_f32; 4];
        let (mean, var) = layer
            .predict_with_uncertainty(&input, 50, 7)
            .expect("predict failed");
        assert_eq!(mean.len(), 5);
        assert_eq!(var.len(), 5);
    }

    #[test]
    fn predict_with_uncertainty_variance_non_negative() {
        let layer = BayesianLinear::new(4, 3, 1.0).expect("construction failed");
        let input = vec![0.3_f32; 4];
        let (_mean, var) = layer
            .predict_with_uncertainty(&input, 100, 13)
            .expect("predict failed");
        for (i, &v) in var.iter().enumerate() {
            assert!(v >= 0.0, "variance[{i}] = {v} should be >= 0");
        }
    }

    // ── kl_divergence ────────────────────────────────────────────────────────

    #[test]
    fn kl_divergence_zero_when_at_prior() {
        // Set mu = 0, sigma = prior_sigma => KL should be 0.
        let prior_sigma = 1.0_f32;
        let mut layer = BayesianLinear::new(2, 2, prior_sigma).expect("construction failed");

        // Force mu = 0 and log_sigma such that softplus(log_sigma) = prior_sigma = 1.
        // softplus(x) = prior_sigma => x = log(exp(prior_sigma) - 1).
        let target_ls = (prior_sigma.exp() - 1.0).ln();
        layer.weight_mu.iter_mut().for_each(|v| *v = 0.0);
        layer
            .weight_log_sigma
            .iter_mut()
            .for_each(|v| *v = target_ls);
        layer.bias_mu.iter_mut().for_each(|v| *v = 0.0);
        layer.bias_log_sigma.iter_mut().for_each(|v| *v = target_ls);

        let kl = layer.kl_divergence();
        assert!(kl.abs() < 1e-4, "KL should be ~0 at the prior, got {kl}");
    }

    #[test]
    fn kl_divergence_positive_when_off_prior() {
        let mut layer = BayesianLinear::new(3, 3, 1.0).expect("construction failed");
        // Shift mu far from 0.
        layer.weight_mu.iter_mut().for_each(|v| *v = 5.0);
        let kl = layer.kl_divergence();
        assert!(
            kl > 0.0,
            "KL should be > 0 when weights are off-prior, got {kl}"
        );
    }

    #[test]
    fn kl_divergence_non_negative() {
        let layer = BayesianLinear::new(5, 4, 1.0).expect("construction failed");
        let kl = layer.kl_divergence();
        assert!(kl >= 0.0, "KL divergence must be non-negative, got {kl}");
    }

    // ── num_params ───────────────────────────────────────────────────────────

    #[test]
    fn num_params_correct() {
        let layer = BayesianLinear::new(3, 4, 1.0).expect("construction failed");
        // 2 * (3*4 + 4) = 2 * 16 = 32
        assert_eq!(layer.num_params(), 32);
    }

    // ── BayesianMlp ──────────────────────────────────────────────────────────

    #[test]
    fn bayesian_mlp_forward_sample_shape() {
        let mlp = BayesianMlp::new(&[(4, 8), (8, 2)], 1.0).expect("construction failed");
        let input = vec![1.0_f32; 4];
        let out = mlp.forward_sample(&input, 0).expect("fwd failed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn bayesian_mlp_num_layers() {
        let mlp = BayesianMlp::new(&[(2, 4), (4, 4), (4, 1)], 1.0).expect("construction failed");
        assert_eq!(mlp.num_layers(), 3);
    }

    #[test]
    fn bayesian_mlp_total_kl_non_negative() {
        let mlp = BayesianMlp::new(&[(4, 8), (8, 2)], 1.0).expect("construction failed");
        assert!(
            mlp.total_kl() >= 0.0,
            "total KL must be >= 0, got {}",
            mlp.total_kl()
        );
    }

    #[test]
    fn bayesian_mlp_elbo_loss_correct() {
        let mlp = BayesianMlp::new(&[(2, 2)], 1.0).expect("construction failed");
        let nll = 3.5_f32;
        let kl_weight = 0.01_f32;
        let kl = mlp.total_kl();
        let elbo = mlp.elbo_loss(nll, kl_weight);
        let expected = nll + kl_weight * kl;
        assert!(
            (elbo - expected).abs() < 1e-6,
            "elbo = {elbo}, expected {expected}"
        );
    }

    #[test]
    fn bayesian_mlp_predict_with_uncertainty_shapes() {
        let mlp = BayesianMlp::new(&[(3, 6), (6, 2)], 1.0).expect("construction failed");
        let input = vec![0.5_f32; 3];
        let (mean, var) = mlp
            .predict_with_uncertainty(&input, 30, 0)
            .expect("predict failed");
        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
        for &v in &var {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn bayesian_mlp_rejects_empty_layers() {
        assert!(BayesianMlp::new(&[], 1.0).is_err());
    }

    #[test]
    fn bayesian_mlp_different_seeds_stochastic() {
        let mlp = BayesianMlp::new(&[(4, 4), (4, 2)], 1.0).expect("construction failed");
        let input = vec![1.0_f32; 4];
        let out1 = mlp.forward_sample(&input, 100).expect("fwd 1 failed");
        let out2 = mlp.forward_sample(&input, 200).expect("fwd 2 failed");
        let same = out1
            .iter()
            .zip(out2.iter())
            .all(|(a, b)| (a - b).abs() < 1e-8);
        assert!(!same, "different seeds should give different MLP outputs");
    }

    // ── epistemic_uncertainty ────────────────────────────────────────────────

    #[test]
    fn epistemic_uncertainty_identical_samples_zero_variance() {
        let sample = vec![1.0_f32, 2.0, 3.0];
        let mc = vec![sample.clone(), sample.clone(), sample.clone()];
        let var = epistemic_uncertainty(&mc).expect("epi failed");
        for (i, &v) in var.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "epistemic_uncertainty[{i}] should be 0 for identical samples, got {v}"
            );
        }
    }

    #[test]
    fn epistemic_uncertainty_varying_samples_positive_variance() {
        let mc = vec![vec![0.0_f32, 0.0], vec![1.0_f32, 2.0], vec![2.0_f32, 4.0]];
        let var = epistemic_uncertainty(&mc).expect("epi failed");
        for &v in &var {
            assert!(
                v > 0.0,
                "epistemic_uncertainty should be > 0 for varying samples"
            );
        }
    }

    #[test]
    fn epistemic_uncertainty_rejects_empty_input() {
        let empty: &[Vec<f32>] = &[];
        assert!(epistemic_uncertainty(empty).is_err());
    }

    #[test]
    fn epistemic_uncertainty_rejects_mismatched_lengths() {
        let mc = vec![vec![1.0_f32, 2.0], vec![1.0_f32]];
        assert!(epistemic_uncertainty(&mc).is_err());
    }

    // ── aleatoric_uncertainty ────────────────────────────────────────────────

    #[test]
    fn aleatoric_uncertainty_output_shape() {
        // 4 outputs: first 2 predictions, last 2 log-variances.
        let mc = vec![
            vec![0.5_f32, -0.5, 0.0, -1.0],
            vec![0.6_f32, -0.4, 0.1, -0.9],
        ];
        let aleat = aleatoric_uncertainty(&mc).expect("aleat failed");
        assert_eq!(
            aleat.len(),
            2,
            "aleatoric output should have half the output dim"
        );
    }

    #[test]
    fn aleatoric_uncertainty_values_positive() {
        // exp of log-variance should always be positive.
        let mc = vec![vec![1.0_f32, 2.0, -1.0, 0.5], vec![1.5_f32, 2.5, -2.0, 1.0]];
        let aleat = aleatoric_uncertainty(&mc).expect("aleat failed");
        for (i, &v) in aleat.iter().enumerate() {
            assert!(v > 0.0, "aleatoric_uncertainty[{i}] = {v} should be > 0");
        }
    }

    #[test]
    fn aleatoric_uncertainty_rejects_odd_output_dim() {
        let mc = vec![vec![1.0_f32, 2.0, 3.0]];
        assert!(aleatoric_uncertainty(&mc).is_err());
    }

    #[test]
    fn aleatoric_uncertainty_rejects_empty_input() {
        let empty: &[Vec<f32>] = &[];
        assert!(aleatoric_uncertainty(empty).is_err());
    }
}
