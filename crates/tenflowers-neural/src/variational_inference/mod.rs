//! Black-Box Variational Inference (BBVI), ADVI, and Structured VI.
//!
//! Implements:
//! - [`MeanFieldGaussian`] — mean-field Gaussian variational family
//! - [`FullRankGaussian`] — full-rank Gaussian with Cholesky parameterization
//! - [`BlackBoxVi`] — BBVI with pathwise (reparameterization) and score gradients
//! - [`AdviModel`] — Automatic Differentiation Variational Inference (Kucukelbir 2017)
//! - [`StructuredVi`] — structured mean-field VI over latent time series
//! - [`FlowVi`] — normalizing flow variational family (planar flows)
//! - [`ViSvgd`] — Stein Variational Gradient Descent (Liu & Wang 2016)
//! - [`ViDiagnostics`] — diagnostics: Pareto-k, ESS, ELBO variance
//!
//! All types use the `Vi` prefix where names conflict with `monte_carlo.rs`.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Box-Muller normal sampling helper
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a single N(0,1) sample via Box-Muller transform.
#[inline]
fn box_muller_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Draw `n` i.i.d. N(0,1) samples.
fn normal_vec(n: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| box_muller_normal(rng)).collect()
}

/// Numerically stable log-sum-exp.
pub fn vi_log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  MeanFieldGaussian
// ─────────────────────────────────────────────────────────────────────────────

/// Mean-field Gaussian variational distribution.
///
/// Parameterized by `mu` and `log_sigma` (both unconstrained).
/// `sigma = exp(log_sigma)` is always positive.
#[derive(Debug, Clone)]
pub struct MeanFieldGaussian {
    /// Variational mean.
    pub mu: Vec<f64>,
    /// Log standard deviation (unconstrained).
    pub log_sigma: Vec<f64>,
    /// Dimensionality.
    pub dim: usize,
}

impl MeanFieldGaussian {
    /// Create with `mu = 0`, `log_sigma = 0` (so `sigma = 1`).
    pub fn new(dim: usize) -> Self {
        Self {
            mu: vec![0.0; dim],
            log_sigma: vec![0.0; dim],
            dim,
        }
    }

    /// Reparameterized sample: `z_i = mu_i + exp(log_sigma_i) * eps_i`.
    pub fn sample(&self, eps: &[f64]) -> Vec<f64> {
        self.mu
            .iter()
            .zip(self.log_sigma.iter())
            .zip(eps.iter())
            .map(|((&m, &ls), &e)| m + ls.exp() * e)
            .collect()
    }

    /// Log probability: `log N(z; mu, diag(sigma^2))`.
    pub fn log_prob(&self, z: &[f64]) -> f64 {
        let d = self.dim as f64;
        let lp: f64 = self
            .mu
            .iter()
            .zip(self.log_sigma.iter())
            .zip(z.iter())
            .map(|((&m, &ls), &zi)| {
                let sigma = ls.exp().max(1e-30);
                -0.5 * ((zi - m) / sigma).powi(2) - ls - 0.5 * (2.0 * PI).ln()
            })
            .sum();
        lp + 0.0 * d // d already accounted for by summing over dimensions
    }

    /// Differential entropy: `H = 0.5 * sum(1 + log(2πe) + 2*log_sigma_i)`.
    pub fn entropy(&self) -> f64 {
        let log_2pie = (2.0 * PI * std::f64::consts::E).ln();
        self.log_sigma
            .iter()
            .map(|&ls| 0.5 * (1.0 + log_2pie + 2.0 * ls))
            .sum()
    }

    /// KL divergence from `q` to `N(0, I)`:
    /// `KL = -0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)`.
    pub fn kl_to_standard_normal(&self) -> f64 {
        self.mu
            .iter()
            .zip(self.log_sigma.iter())
            .map(|(&m, &ls)| {
                let sigma2 = (2.0 * ls).exp();
                -0.5 * (1.0 + 2.0 * ls - m * m - sigma2)
            })
            .sum()
    }

    /// Gradient update (SGD step): `mu -= lr * grad_mu`, `log_sigma -= lr * grad_ls`.
    pub fn update(&mut self, grad_mu: &[f64], grad_log_sigma: &[f64], lr: f64) {
        for i in 0..self.dim {
            self.mu[i] -= lr * grad_mu[i];
            self.log_sigma[i] -= lr * grad_log_sigma[i];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  FullRankGaussian
// ─────────────────────────────────────────────────────────────────────────────

/// Full-rank Gaussian variational distribution with Cholesky parameterization.
///
/// Covariance: `Σ = L * L^T` where `L` is lower-triangular.
#[derive(Debug, Clone)]
pub struct FullRankGaussian {
    /// Variational mean.
    pub mu: Vec<f64>,
    /// Lower-triangular Cholesky factor `L` of the covariance, stored row-by-row.
    pub l: Vec<Vec<f64>>,
    /// Dimensionality.
    pub dim: usize,
}

impl FullRankGaussian {
    /// Create with `mu = 0`, `L = I`.
    pub fn new(dim: usize) -> Self {
        let mut l = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            l[i][i] = 1.0;
        }
        Self {
            mu: vec![0.0; dim],
            l,
            dim,
        }
    }

    /// Reparameterized sample: `z = mu + L * eps`.
    pub fn sample(&self, eps: &[f64]) -> Vec<f64> {
        let mut z = self.mu.clone();
        for i in 0..self.dim {
            for j in 0..=i {
                z[i] += self.l[i][j] * eps[j];
            }
        }
        z
    }

    /// Log probability under the full-rank Gaussian.
    ///
    /// Uses the Cholesky factor for efficient computation.
    pub fn log_prob(&self, z: &[f64]) -> f64 {
        let d = self.dim;
        // log|det(Sigma)| = 2 * sum(log L_ii)
        let log_det: f64 = (0..d)
            .map(|i| self.l[i][i].abs().max(1e-30).ln())
            .sum::<f64>()
            * 2.0;

        // Solve L * v = z - mu  (forward substitution)
        let mut diff = vec![0.0_f64; d];
        for i in 0..d {
            diff[i] = z[i] - self.mu[i];
        }
        let mut v = vec![0.0_f64; d];
        for i in 0..d {
            let mut s = diff[i];
            for j in 0..i {
                s -= self.l[i][j] * v[j];
            }
            let lii = self.l[i][i];
            if lii.abs() < 1e-30 {
                return f64::NEG_INFINITY;
            }
            v[i] = s / lii;
        }
        let maha: f64 = v.iter().map(|&vi| vi * vi).sum();
        -0.5 * (d as f64) * (2.0 * PI).ln() - 0.5 * log_det - 0.5 * maha
    }

    /// Entropy: `H = 0.5 * d * (1 + log(2π)) + sum_i log|L_ii|`.
    pub fn entropy(&self) -> f64 {
        let d = self.dim as f64;
        let log_diag: f64 = (0..self.dim)
            .map(|i| self.l[i][i].abs().max(1e-30).ln())
            .sum();
        0.5 * d * (1.0 + (2.0 * PI).ln()) + log_diag
    }

    /// KL divergence from `q` to `N(0, I)`:
    /// `KL = 0.5 * [tr(LL^T) + mu^T mu - d - log det(LL^T)]`.
    pub fn kl_to_standard_normal(&self) -> f64 {
        let d = self.dim;
        // tr(LL^T) = sum_{i,j} L_{ij}^2
        let trace: f64 = self
            .l
            .iter()
            .map(|row| row.iter().map(|&x| x * x).sum::<f64>())
            .sum();
        let mu_sq: f64 = self.mu.iter().map(|&m| m * m).sum();
        let log_det: f64 = (0..d)
            .map(|i| self.l[i][i].abs().max(1e-30).ln())
            .sum::<f64>()
            * 2.0;
        0.5 * (trace + mu_sq - d as f64 - log_det)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  BBVI — Black-Box Variational Inference
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`BlackBoxVi`].
#[derive(Debug, Clone)]
pub struct BbviConfig {
    /// Number of MC samples for gradient estimation (default: 10).
    pub n_samples: usize,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Use REINFORCE control variate baseline (reduces variance).
    pub use_baseline: bool,
    /// Use pathwise (reparameterization) gradient (preferred over score function).
    pub use_pathwise: bool,
}

impl Default for BbviConfig {
    fn default() -> Self {
        Self {
            n_samples: 10,
            n_epochs: 200,
            lr: 0.01,
            use_baseline: true,
            use_pathwise: true,
        }
    }
}

/// Result of fitting [`BlackBoxVi`].
#[derive(Debug, Clone)]
pub struct BbviResult {
    /// ELBO estimates per epoch.
    pub elbo_history: Vec<f64>,
    /// Final variational mean.
    pub final_mu: Vec<f64>,
    /// Final variational standard deviation (`exp(log_sigma)`).
    pub final_sigma: Vec<f64>,
}

/// Black-Box Variational Inference using a [`MeanFieldGaussian`] variational family.
#[derive(Debug, Clone)]
pub struct BlackBoxVi {
    /// Variational distribution.
    pub q: MeanFieldGaussian,
    /// Hyperparameter config.
    pub config: BbviConfig,
}

impl BlackBoxVi {
    /// Construct with given `dim` and `config`.
    pub fn new(dim: usize, config: BbviConfig) -> Self {
        Self {
            q: MeanFieldGaussian::new(dim),
            config,
        }
    }

    /// Estimate ELBO = E_q[log p(x,z) − log q(z)] via Monte Carlo.
    pub fn elbo_estimate(
        &self,
        log_joint: &dyn Fn(&[f64]) -> f64,
        n_samples: usize,
        rng: &mut StdRng,
    ) -> f64 {
        let mut total = 0.0_f64;
        for _ in 0..n_samples {
            let eps = normal_vec(self.q.dim, rng);
            let z = self.q.sample(&eps);
            let lp = log_joint(&z);
            let lq = self.q.log_prob(&z);
            if lp.is_finite() && lq.is_finite() {
                total += lp - lq;
            }
        }
        total / (n_samples as f64)
    }

    /// Pathwise (reparameterization) gradient via finite differences.
    ///
    /// Returns `(grad_mu, grad_log_sigma)`.
    pub fn pathwise_gradient(
        &self,
        log_joint: &dyn Fn(&[f64]) -> f64,
        rng: &mut StdRng,
        eps_fd: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = self.q.dim;
        let mut grad_mu = vec![0.0_f64; dim];
        let mut grad_log_sigma = vec![0.0_f64; dim];

        let n_samples = self.config.n_samples;

        // Accumulate gradient estimates over MC samples
        for _ in 0..n_samples {
            let eps = normal_vec(dim, rng);

            // Gradient w.r.t. mu_i: dELBO/d(mu_i) ≈ (ELBO(mu+h*e_i) - ELBO(mu-h*e_i)) / (2h)
            // But analytically: z = mu + sigma*eps => dz/d(mu_i) = e_i
            // So grad_mu_i = E[d(log p)/d(z_i)] (pathwise)
            // We use FD on the ELBO with perturbed z
            for i in 0..dim {
                // Perturb mu[i]
                let mut q_plus = self.q.clone();
                let mut q_minus = self.q.clone();
                q_plus.mu[i] += eps_fd;
                q_minus.mu[i] -= eps_fd;

                let z_plus = q_plus.sample(&eps);
                let z_minus = q_minus.sample(&eps);

                let elbo_plus = log_joint(&z_plus) - q_plus.log_prob(&z_plus);
                let elbo_minus = log_joint(&z_minus) - q_minus.log_prob(&z_minus);

                if elbo_plus.is_finite() && elbo_minus.is_finite() {
                    grad_mu[i] += (elbo_plus - elbo_minus) / (2.0 * eps_fd);
                }

                // Perturb log_sigma[i]
                let mut q_plus_ls = self.q.clone();
                let mut q_minus_ls = self.q.clone();
                q_plus_ls.log_sigma[i] += eps_fd;
                q_minus_ls.log_sigma[i] -= eps_fd;

                let z_plus_ls = q_plus_ls.sample(&eps);
                let z_minus_ls = q_minus_ls.sample(&eps);

                let elbo_plus_ls = log_joint(&z_plus_ls) - q_plus_ls.log_prob(&z_plus_ls);
                let elbo_minus_ls = log_joint(&z_minus_ls) - q_minus_ls.log_prob(&z_minus_ls);

                if elbo_plus_ls.is_finite() && elbo_minus_ls.is_finite() {
                    grad_log_sigma[i] += (elbo_plus_ls - elbo_minus_ls) / (2.0 * eps_fd);
                }
            }
        }

        let n = n_samples as f64;
        for i in 0..dim {
            grad_mu[i] /= n;
            grad_log_sigma[i] /= n;
        }

        (grad_mu, grad_log_sigma)
    }

    /// REINFORCE (score function) gradient estimator.
    ///
    /// `g_mu = E[(log p(x,z) - log q(z)) * ∂ log q / ∂ mu]`
    /// Optionally subtracts a mean-ELBO baseline to reduce variance.
    pub fn score_gradient(
        &self,
        log_joint: &dyn Fn(&[f64]) -> f64,
        rng: &mut StdRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let dim = self.q.dim;
        let n = self.config.n_samples;
        let mut elbos = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);

        // First pass: collect samples and ELBO values
        for _ in 0..n {
            let eps = normal_vec(dim, rng);
            let z = self.q.sample(&eps);
            let lp = log_joint(&z);
            let lq = self.q.log_prob(&z);
            let elbo_s = if lp.is_finite() && lq.is_finite() {
                lp - lq
            } else {
                0.0
            };
            elbos.push(elbo_s);
            zs.push(z);
        }

        let baseline = if self.config.use_baseline {
            elbos.iter().sum::<f64>() / (n as f64)
        } else {
            0.0
        };

        let mut grad_mu = vec![0.0_f64; dim];
        let mut grad_log_sigma = vec![0.0_f64; dim];

        // ∂ log q / ∂ mu_i = (z_i - mu_i) / sigma_i^2
        // ∂ log q / ∂ log_sigma_i = (z_i - mu_i)^2 / sigma_i^2 - 1
        for (elbo_s, z) in elbos.iter().zip(zs.iter()) {
            let f = elbo_s - baseline;
            for i in 0..dim {
                let sigma = self.q.log_sigma[i].exp().max(1e-30);
                let sigma2 = sigma * sigma;
                let diff = z[i] - self.q.mu[i];
                // grad log q / grad mu_i
                let score_mu = diff / sigma2;
                // grad log q / grad log_sigma_i
                let score_ls = diff * diff / sigma2 - 1.0;

                grad_mu[i] += f * score_mu;
                grad_log_sigma[i] += f * score_ls;
            }
        }

        let nf = n as f64;
        for i in 0..dim {
            grad_mu[i] /= nf;
            grad_log_sigma[i] /= nf;
        }

        (grad_mu, grad_log_sigma)
    }

    /// Fit the variational distribution to the log-joint.
    pub fn fit(&mut self, log_joint: &dyn Fn(&[f64]) -> f64, rng: &mut StdRng) -> BbviResult {
        let mut elbo_history = Vec::with_capacity(self.config.n_epochs);
        let fd_eps = 1e-4;

        for _ in 0..self.config.n_epochs {
            let elbo = self.elbo_estimate(log_joint, self.config.n_samples, rng);
            elbo_history.push(elbo);

            let (gmu, gls) = if self.config.use_pathwise {
                self.pathwise_gradient(log_joint, rng, fd_eps)
            } else {
                self.score_gradient(log_joint, rng)
            };

            // Gradient ascent (negate to ascend ELBO)
            let lr = self.config.lr;
            for i in 0..self.q.dim {
                self.q.mu[i] += lr * gmu[i];
                self.q.log_sigma[i] += lr * gls[i];
            }
        }

        let final_sigma: Vec<f64> = self.q.log_sigma.iter().map(|&ls| ls.exp()).collect();
        BbviResult {
            elbo_history,
            final_mu: self.q.mu.clone(),
            final_sigma,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  ADVI — Automatic Differentiation Variational Inference
// ─────────────────────────────────────────────────────────────────────────────

/// Constraint type for an ADVI model parameter.
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Parameter lives in ℝ — no transformation needed.
    Unconstrained,
    /// Parameter must be positive — use log transform: `θ = exp(φ)`.
    Positive,
    /// Parameter lives on the K-simplex — use softmax: `θ = softmax(φ)`.
    Simplex(usize),
    /// Parameter bounded: `θ = sigmoid(φ) * (high - low) + low`.
    Bounded { low: f64, high: f64 },
}

/// A named ADVI parameter variable with constraint information.
#[derive(Debug, Clone)]
pub struct AdviVariable {
    /// Human-readable parameter name.
    pub name: String,
    /// Dimensionality of the parameter.
    pub dim: usize,
    /// Constraint type governing the transformation.
    pub constraint: ParameterConstraint,
}

impl AdviVariable {
    /// Transform from constrained space `θ` to unconstrained `φ`.
    pub fn transform_to_real(&self, constrained: &[f64]) -> Vec<f64> {
        match &self.constraint {
            ParameterConstraint::Unconstrained => constrained.to_vec(),
            ParameterConstraint::Positive => {
                constrained.iter().map(|&t| t.max(1e-30).ln()).collect()
            }
            ParameterConstraint::Simplex(_k) => {
                // Stick-breaking: phi_i = log(theta_i / theta_K)
                let sum: f64 = constrained.iter().sum::<f64>().max(1e-30);
                let last = constrained.last().cloned().unwrap_or(1e-30).max(1e-30);
                constrained[..constrained.len().saturating_sub(1)]
                    .iter()
                    .map(|&t| (t.max(1e-30) / last).ln())
                    .collect()
            }
            ParameterConstraint::Bounded { low, high } => {
                let range = (high - low).max(1e-30);
                constrained
                    .iter()
                    .map(|&t| {
                        let p = ((t - low) / range).clamp(1e-15, 1.0 - 1e-15);
                        (p / (1.0 - p)).ln()
                    })
                    .collect()
            }
        }
    }

    /// Transform from unconstrained space `φ` to constrained `θ`.
    pub fn transform_from_real(&self, unconstrained: &[f64]) -> Vec<f64> {
        match &self.constraint {
            ParameterConstraint::Unconstrained => unconstrained.to_vec(),
            ParameterConstraint::Positive => unconstrained.iter().map(|&p| p.exp()).collect(),
            ParameterConstraint::Simplex(k) => {
                // Inverse stick-breaking: softmax of [phi_1,...,phi_{K-1}, 0]
                let mut extended: Vec<f64> = unconstrained
                    .iter()
                    .cloned()
                    .chain(std::iter::once(0.0))
                    .collect();
                let max_v = extended.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = extended.iter().map(|&x| (x - max_v).exp()).sum();
                extended
                    .iter_mut()
                    .for_each(|x| *x = ((*x - max_v).exp()) / sum);
                let _ = k;
                extended
            }
            ParameterConstraint::Bounded { low, high } => unconstrained
                .iter()
                .map(|&p| {
                    let sigmoid = 1.0 / (1.0 + (-p).exp());
                    sigmoid * (high - low) + low
                })
                .collect(),
        }
    }

    /// Log absolute Jacobian determinant: `log |∂θ/∂φ|`.
    pub fn log_abs_det_jacobian(&self, unconstrained: &[f64]) -> f64 {
        match &self.constraint {
            ParameterConstraint::Unconstrained => 0.0,
            ParameterConstraint::Positive => {
                // dθ/dφ = exp(φ) = θ  => log|J| = sum(φ)
                unconstrained.iter().sum()
            }
            ParameterConstraint::Simplex(_) => {
                // Dirichlet stick-breaking Jacobian (approximate: sum of log-softmax contributions)
                let max_v = unconstrained
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = unconstrained.iter().map(|&x| (x - max_v).exp()).sum();
                let n = unconstrained.len() as f64;
                // log|J| = sum(phi) - n * log(sum(exp(phi)))
                unconstrained.iter().sum::<f64>() - n * (sum.ln() + max_v)
            }
            ParameterConstraint::Bounded { low, high } => {
                // dθ/dφ = sigmoid(φ) * (1 - sigmoid(φ)) * (high - low)
                unconstrained
                    .iter()
                    .map(|&p| {
                        let s = 1.0 / (1.0 + (-p).exp());
                        let deriv = s * (1.0 - s) * (high - low);
                        deriv.abs().max(1e-30).ln()
                    })
                    .sum()
            }
        }
    }

    /// Unconstrained dimensionality (may differ from `dim` for Simplex).
    pub fn unconstrained_dim(&self) -> usize {
        match &self.constraint {
            ParameterConstraint::Simplex(k) => k.saturating_sub(1),
            _ => self.dim,
        }
    }
}

/// ADVI model: fits a mean-field Gaussian in unconstrained space.
#[derive(Debug, Clone)]
pub struct AdviModel {
    /// Parameter variables with constraint information.
    pub variables: Vec<AdviVariable>,
    /// Mean-field Gaussian in the unconstrained space (total dim = sum of unconstrained dims).
    pub q: MeanFieldGaussian,
    /// Number of MC samples per ELBO estimate.
    pub n_samples: usize,
    /// Learning rate.
    pub lr: f64,
}

impl AdviModel {
    /// Construct from variables with given `n_samples` and `lr`.
    pub fn new(variables: Vec<AdviVariable>, n_samples: usize, lr: f64) -> Self {
        let total_dim: usize = variables.iter().map(|v| v.unconstrained_dim()).sum();
        Self {
            variables,
            q: MeanFieldGaussian::new(total_dim),
            n_samples,
            lr,
        }
    }

    /// Split a flat unconstrained vector into per-variable chunks.
    pub fn split_params(&self, flat: &[f64]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(self.variables.len());
        let mut offset = 0;
        for v in &self.variables {
            let d = v.unconstrained_dim();
            out.push(flat[offset..offset + d].to_vec());
            offset += d;
        }
        out
    }

    /// Sample from `q` in unconstrained space, transform to constrained space.
    ///
    /// Returns a list of constrained parameter vectors (one per variable).
    pub fn constrained_sample(&self, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let eps = normal_vec(self.q.dim, rng);
        let phi = self.q.sample(&eps);
        let chunks = self.split_params(&phi);
        chunks
            .iter()
            .zip(self.variables.iter())
            .map(|(chunk, var)| var.transform_from_real(chunk))
            .collect()
    }

    /// Estimate ADVI ELBO:
    /// `E_q[log p(x, θ(φ)) + log|J(φ)|] + H[q(φ)]`.
    pub fn elbo(&self, log_joint: &dyn Fn(&[Vec<f64>]) -> f64, rng: &mut StdRng) -> f64 {
        let mut total = 0.0_f64;
        for _ in 0..self.n_samples {
            let eps = normal_vec(self.q.dim, rng);
            let phi = self.q.sample(&eps);
            let chunks = self.split_params(&phi);

            // Jacobian correction: log|J| = sum over variables of log|dθ/dφ|
            let log_jac: f64 = chunks
                .iter()
                .zip(self.variables.iter())
                .map(|(chunk, var)| var.log_abs_det_jacobian(chunk))
                .sum();

            let constrained: Vec<Vec<f64>> = chunks
                .iter()
                .zip(self.variables.iter())
                .map(|(chunk, var)| var.transform_from_real(chunk))
                .collect();

            let lp = log_joint(&constrained);
            if lp.is_finite() && log_jac.is_finite() {
                total += lp + log_jac;
            }
        }

        total / (self.n_samples as f64) + self.q.entropy()
    }

    /// Fit by stochastic gradient ascent on the ADVI ELBO.
    ///
    /// Returns ELBO history.
    pub fn fit(
        &mut self,
        log_joint: &dyn Fn(&[Vec<f64>]) -> f64,
        n_epochs: usize,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        let fd_eps = 1e-4_f64;
        let dim = self.q.dim;

        for _epoch in 0..n_epochs {
            let elbo = self.elbo(log_joint, rng);
            history.push(elbo);

            // Finite-difference gradient of ADVI ELBO w.r.t. (mu, log_sigma)
            let mut gmu = vec![0.0_f64; dim];
            let mut gls = vec![0.0_f64; dim];

            for i in 0..dim {
                // grad w.r.t. mu[i]: perturb and restore
                let saved_mu = self.q.mu[i];

                self.q.mu[i] = saved_mu + fd_eps;
                let e_plus = {
                    let mut local_rng = StdRng::seed_from_u64(99 + i as u64);
                    self.elbo(log_joint, &mut local_rng)
                };

                self.q.mu[i] = saved_mu - fd_eps;
                let e_minus = {
                    let mut local_rng = StdRng::seed_from_u64(99 + i as u64);
                    self.elbo(log_joint, &mut local_rng)
                };

                self.q.mu[i] = saved_mu;

                if e_plus.is_finite() && e_minus.is_finite() {
                    gmu[i] = (e_plus - e_minus) / (2.0 * fd_eps);
                }

                // grad w.r.t. log_sigma[i]: perturb and restore
                let saved_ls = self.q.log_sigma[i];

                self.q.log_sigma[i] = saved_ls + fd_eps;
                let els_plus = {
                    let mut local_rng = StdRng::seed_from_u64(199 + i as u64);
                    self.elbo(log_joint, &mut local_rng)
                };

                self.q.log_sigma[i] = saved_ls - fd_eps;
                let els_minus = {
                    let mut local_rng = StdRng::seed_from_u64(199 + i as u64);
                    self.elbo(log_joint, &mut local_rng)
                };

                self.q.log_sigma[i] = saved_ls;

                if els_plus.is_finite() && els_minus.is_finite() {
                    gls[i] = (els_plus - els_minus) / (2.0 * fd_eps);
                }
            }

            // Gradient ascent
            for i in 0..dim {
                self.q.mu[i] += self.lr * gmu[i];
                self.q.log_sigma[i] += self.lr * gls[i];
            }
        }

        history
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  StructuredVi — structured mean-field VI over latent time series
// ─────────────────────────────────────────────────────────────────────────────

/// Structured mean-field variational inference for temporal latent models.
///
/// Maintains a separate [`MeanFieldGaussian`] for each time step.
#[derive(Debug, Clone)]
pub struct StructuredVi {
    /// Number of time steps.
    pub n_time: usize,
    /// Dimensionality of each latent vector `z_t`.
    pub z_dim: usize,
    /// Per-time-step variational distributions.
    pub local_qs: Vec<MeanFieldGaussian>,
}

impl StructuredVi {
    /// Construct with zero-mean, unit-variance initialization.
    pub fn new(n_time: usize, z_dim: usize) -> Self {
        let local_qs = (0..n_time).map(|_| MeanFieldGaussian::new(z_dim)).collect();
        Self {
            n_time,
            z_dim,
            local_qs,
        }
    }

    /// Sample one trajectory `[z_1, ..., z_T]` (one sample per time step).
    pub fn sample_trajectory(&self, rng: &mut StdRng) -> Vec<Vec<f64>> {
        self.local_qs
            .iter()
            .map(|q| {
                let eps = normal_vec(q.dim, rng);
                q.sample(&eps)
            })
            .collect()
    }

    /// Total entropy: sum of per-step entropies.
    pub fn entropy(&self) -> f64 {
        self.local_qs.iter().map(|q| q.entropy()).sum()
    }

    /// Total KL to prior `N(0, I)`: sum of per-step KLs.
    pub fn kl_to_prior(&self) -> f64 {
        self.local_qs
            .iter()
            .map(|q| q.kl_to_standard_normal())
            .sum()
    }

    /// Estimate ELBO = E_q[log p(x | z_1:T)] - KL(q || p).
    pub fn elbo(&self, log_likelihood: &dyn Fn(&[Vec<f64>]) -> f64, rng: &mut StdRng) -> f64 {
        let traj = self.sample_trajectory(rng);
        let ll = log_likelihood(&traj);
        let kl = self.kl_to_prior();
        if ll.is_finite() {
            ll - kl
        } else {
            -kl
        }
    }

    /// Update all local variational parameters simultaneously.
    pub fn update(&mut self, grad_mus: &[Vec<f64>], grad_log_sigmas: &[Vec<f64>], lr: f64) {
        for t in 0..self.n_time {
            if t < grad_mus.len() && t < grad_log_sigmas.len() {
                self.local_qs[t].update(&grad_mus[t], &grad_log_sigmas[t], lr);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  FlowVi — Normalizing Flow as Variational Family
// ─────────────────────────────────────────────────────────────────────────────

/// A single planar flow layer.
///
/// Transform: `z' = z + u * tanh(w^T z + b)`.
#[derive(Debug, Clone)]
pub struct PlanarFlowLayer {
    /// Weight vector `[dim]`.
    pub w: Vec<f64>,
    /// Perturbation vector `[dim]`.
    pub u: Vec<f64>,
    /// Scalar bias.
    pub b: f64,
}

impl PlanarFlowLayer {
    /// Initialize with small random weights (seeded via index for reproducibility).
    pub fn new(dim: usize) -> Self {
        let scale = 0.1;
        // Use deterministic small initialization
        let w: Vec<f64> = (0..dim)
            .map(|i| scale * ((i + 1) as f64 * 0.1).sin())
            .collect();
        let u: Vec<f64> = (0..dim)
            .map(|i| scale * ((i + 1) as f64 * 0.2).cos())
            .collect();
        Self { w, u, b: 0.0 }
    }

    /// Forward pass through the planar flow.
    ///
    /// Returns `(z', log|det J|)`.
    ///
    /// `z' = z + u_hat * tanh(w^T z + b)`
    /// `log|det J| = log|1 + u_hat^T psi|` where `psi = (1 - tanh^2) * w`.
    pub fn forward(&self, z: &[f64]) -> (Vec<f64>, f64) {
        let dim = z.len();
        // w^T z + b
        let lin: f64 = self
            .w
            .iter()
            .zip(z.iter())
            .map(|(&wi, &zi)| wi * zi)
            .sum::<f64>()
            + self.b;
        let tanh_lin = lin.tanh();
        let dtanh = 1.0 - tanh_lin * tanh_lin; // sech^2

        // Enforce invertibility: u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        // where m(x) = -1 + softplus(x)
        let w_dot_u: f64 = self
            .w
            .iter()
            .zip(self.u.iter())
            .map(|(&wi, &ui)| wi * ui)
            .sum();
        let w_norm_sq: f64 = self.w.iter().map(|&wi| wi * wi).sum::<f64>().max(1e-30);
        let m_val = -1.0 + (1.0 + w_dot_u.exp()).ln(); // m(w^T u) = softplus(w^T u) - 1
        let correction = (m_val - w_dot_u) / w_norm_sq;
        let u_hat: Vec<f64> = self
            .u
            .iter()
            .zip(self.w.iter())
            .map(|(&ui, &wi)| ui + correction * wi)
            .collect();

        // z' = z + u_hat * tanh(lin)
        let z_prime: Vec<f64> = z
            .iter()
            .zip(u_hat.iter())
            .map(|(&zi, &ui)| zi + ui * tanh_lin)
            .collect();

        // log|det J| = log|1 + u_hat^T * psi|
        // psi = sech^2(lin) * w
        let u_dot_psi: f64 = u_hat
            .iter()
            .zip(self.w.iter())
            .map(|(&ui, &wi)| ui * wi)
            .sum::<f64>()
            * dtanh;
        let log_det = (1.0 + u_dot_psi).abs().max(1e-30).ln();

        let _ = dim;
        (z_prime, log_det)
    }

    /// Gradient-descent parameter update.
    ///
    /// `grad` is a flattened vector: `[grad_w (dim), grad_u (dim), grad_b (1)]`.
    pub fn update(&mut self, grad: &[f64], lr: f64) {
        let dim = self.w.len();
        for i in 0..dim {
            if i < grad.len() {
                self.w[i] -= lr * grad[i];
            }
        }
        for i in 0..dim {
            let idx = dim + i;
            if idx < grad.len() {
                self.u[i] -= lr * grad[idx];
            }
        }
        let b_idx = 2 * dim;
        if b_idx < grad.len() {
            self.b -= lr * grad[b_idx];
        }
    }
}

/// Normalizing flow variational family.
///
/// `z_K = f_K ∘ ... ∘ f_1(z_0)` where `z_0 ~ base_q`.
#[derive(Debug, Clone)]
pub struct FlowVi {
    /// Number of flow layers.
    pub n_flows: usize,
    /// Base distribution (mean-field Gaussian).
    pub base_q: MeanFieldGaussian,
    /// Planar flow layers.
    pub flows: Vec<PlanarFlowLayer>,
}

impl FlowVi {
    /// Construct with `n_flows` planar layers applied to `dim`-dimensional base.
    pub fn new(dim: usize, n_flows: usize) -> Self {
        let flows = (0..n_flows).map(|_| PlanarFlowLayer::new(dim)).collect();
        Self {
            n_flows,
            base_q: MeanFieldGaussian::new(dim),
            flows,
        }
    }

    /// Sample `z_0` from base distribution, transform through all flow layers.
    ///
    /// Returns `(z_K, log q_0(z_0) - sum log|det J_k|)`.
    pub fn sample_and_log_prob(&self, rng: &mut StdRng) -> (Vec<f64>, f64) {
        let eps = normal_vec(self.base_q.dim, rng);
        let z0 = self.base_q.sample(&eps);
        let log_q0 = self.base_q.log_prob(&z0);

        let mut z = z0;
        let mut sum_log_det = 0.0_f64;

        for flow in &self.flows {
            let (z_next, log_det) = flow.forward(&z);
            z = z_next;
            sum_log_det += log_det;
        }

        // log q(z_K) = log q_0(z_0) - sum log|det J_k|  (change of variables)
        let log_qk = log_q0 - sum_log_det;
        (z, log_qk)
    }

    /// Estimate ELBO = E[log p(x, z_K) - log q(z_K)].
    pub fn elbo(
        &self,
        log_joint: &dyn Fn(&[f64]) -> f64,
        n_samples: usize,
        rng: &mut StdRng,
    ) -> f64 {
        let mut total = 0.0_f64;
        let mut count = 0;
        for _ in 0..n_samples {
            let (z, log_qk) = self.sample_and_log_prob(rng);
            let lp = log_joint(&z);
            if lp.is_finite() && log_qk.is_finite() {
                total += lp - log_qk;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// Fit the flow VI by stochastic gradient ascent on the ELBO.
    ///
    /// Uses finite differences to estimate gradients of the base distribution parameters.
    pub fn fit(
        &mut self,
        log_joint: &dyn Fn(&[f64]) -> f64,
        n_epochs: usize,
        lr: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(n_epochs);
        let fd_eps = 1e-4_f64;
        let dim = self.base_q.dim;
        let n_mc = 5;

        for _ in 0..n_epochs {
            let elbo = self.elbo(log_joint, n_mc, rng);
            history.push(elbo);

            // Gradient w.r.t. base_q mu and log_sigma via FD
            for i in 0..dim {
                let e_mu_plus = {
                    let saved = self.base_q.mu[i];
                    self.base_q.mu[i] = saved + fd_eps;
                    let mut r = StdRng::seed_from_u64(42 + i as u64);
                    let ev = self.elbo(log_joint, n_mc, &mut r);
                    self.base_q.mu[i] = saved;
                    ev
                };
                let e_mu_minus = {
                    let saved = self.base_q.mu[i];
                    self.base_q.mu[i] = saved - fd_eps;
                    let mut r = StdRng::seed_from_u64(42 + i as u64);
                    let ev = self.elbo(log_joint, n_mc, &mut r);
                    self.base_q.mu[i] = saved;
                    ev
                };
                if e_mu_plus.is_finite() && e_mu_minus.is_finite() {
                    self.base_q.mu[i] += lr * (e_mu_plus - e_mu_minus) / (2.0 * fd_eps);
                }

                let e_ls_plus = {
                    let saved = self.base_q.log_sigma[i];
                    self.base_q.log_sigma[i] = saved + fd_eps;
                    let mut r = StdRng::seed_from_u64(142 + i as u64);
                    let ev = self.elbo(log_joint, n_mc, &mut r);
                    self.base_q.log_sigma[i] = saved;
                    ev
                };
                let e_ls_minus = {
                    let saved = self.base_q.log_sigma[i];
                    self.base_q.log_sigma[i] = saved - fd_eps;
                    let mut r = StdRng::seed_from_u64(142 + i as u64);
                    let ev = self.elbo(log_joint, n_mc, &mut r);
                    self.base_q.log_sigma[i] = saved;
                    ev
                };
                if e_ls_plus.is_finite() && e_ls_minus.is_finite() {
                    self.base_q.log_sigma[i] += lr * (e_ls_plus - e_ls_minus) / (2.0 * fd_eps);
                }
            }
        }

        history
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  ViSvgd — Stein Variational Gradient Descent (Liu & Wang 2016)
//
//  Named ViSvgd to avoid conflict with monte_carlo::SvgdOptimizer.
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ViSvgd`].
#[derive(Debug, Clone)]
pub struct ViSvgdConfig {
    /// Number of particles.
    pub n_particles: usize,
    /// Learning rate.
    pub lr: f64,
    /// Number of gradient descent steps.
    pub n_steps: usize,
    /// RBF kernel bandwidth. Use `0.0` to trigger the median heuristic.
    pub bandwidth: f64,
}

impl Default for ViSvgdConfig {
    fn default() -> Self {
        Self {
            n_particles: 20,
            lr: 0.01,
            n_steps: 100,
            bandwidth: 0.0,
        }
    }
}

/// Stein Variational Gradient Descent.
///
/// Evolves a set of particles to approximate the target distribution.
pub struct ViSvgd {
    /// Current particle positions `[n_particles][dim]`.
    pub particles: Vec<Vec<f64>>,
    /// Configuration.
    pub config: ViSvgdConfig,
}

impl ViSvgd {
    /// Initialize particles ~ `N(0, I)`.
    pub fn new(dim: usize, config: ViSvgdConfig, rng: &mut StdRng) -> Self {
        let particles = (0..config.n_particles)
            .map(|_| normal_vec(dim, rng))
            .collect();
        Self { particles, config }
    }

    /// Compute median-heuristic bandwidth.
    fn median_bandwidth(particles: &[Vec<f64>]) -> f64 {
        let n = particles.len();
        if n <= 1 {
            return 1.0;
        }
        let mut sq_dists: Vec<f64> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let sq: f64 = particles[i]
                    .iter()
                    .zip(particles[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                sq_dists.push(sq);
            }
        }
        sq_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sq_dists[sq_dists.len() / 2];
        median / ((n as f64).ln().max(1.0))
    }

    /// Compute the RBF kernel matrix `k(xi, xj) = exp(-||xi-xj||^2 / h)`.
    pub fn kernel_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.particles.len();
        let h = if self.config.bandwidth > 0.0 {
            self.config.bandwidth
        } else {
            Self::median_bandwidth(&self.particles).max(1e-10)
        };

        let mut k = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let sq: f64 = self.particles[i]
                    .iter()
                    .zip(self.particles[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                k[i][j] = (-sq / h).exp();
            }
        }
        k
    }

    /// Perform one SVGD update step.
    ///
    /// For each particle `xi`:
    /// `phi(xi) = (1/n) Σ_j [k(xj, xi) * ∇_xj log p(xj) + ∇_xj k(xj, xi)]`
    /// `xi ← xi + lr * phi(xi)`
    pub fn update(&mut self, log_prob_grad: &dyn Fn(&[f64]) -> Vec<f64>) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let dim = self.particles[0].len();

        let h = if self.config.bandwidth > 0.0 {
            self.config.bandwidth
        } else {
            Self::median_bandwidth(&self.particles).max(1e-10)
        };

        // Precompute k(xi, xj) and grad_xj k(xi, xj) for all pairs
        let mut k_mat = vec![vec![0.0_f64; n]; n];
        let mut grad_k = vec![vec![vec![0.0_f64; dim]; n]; n]; // grad_k[i][j] = ∇_xj k(xi, xj)

        for i in 0..n {
            for j in 0..n {
                let sq: f64 = self.particles[i]
                    .iter()
                    .zip(self.particles[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let kij = (-sq / h).exp();
                k_mat[i][j] = kij;
                // ∇_xj k(xi, xj) = kij * 2/h * (xi - xj)
                for d in 0..dim {
                    grad_k[i][j][d] =
                        kij * (2.0 / h) * (self.particles[i][d] - self.particles[j][d]);
                }
            }
        }

        // Compute score gradients ∇ log p(xj)
        let scores: Vec<Vec<f64>> = self.particles.iter().map(|p| log_prob_grad(p)).collect();

        // SVGD update: phi(xi) = (1/n) sum_j [k(xj,xi)*score_j + grad_xj k(xj,xi)]
        let mut updates = vec![vec![0.0_f64; dim]; n];
        let nf = n as f64;
        for i in 0..n {
            for j in 0..n {
                let kji = k_mat[j][i]; // k(xj, xi)
                for d in 0..dim {
                    // score contribution from particle j
                    let score_j_d = if d < scores[j].len() {
                        scores[j][d]
                    } else {
                        0.0
                    };
                    updates[i][d] += kji * score_j_d;
                    // kernel gradient: ∇_xj k(xj, xi) = -grad_k[i][j] (reversed argument)
                    // grad_k[i][j] = ∇_xj k(xi, xj) = kij * 2/h * (xi - xj)
                    // we need ∇_xj k(xj, xi) = k(xj,xi) * 2/h * (xj - xi)
                    updates[i][d] +=
                        kji * (2.0 / h) * (self.particles[j][d] - self.particles[i][d]);
                }
            }
            for d in 0..dim {
                updates[i][d] /= nf;
            }
        }

        // Apply updates
        for i in 0..n {
            for d in 0..dim {
                self.particles[i][d] += self.config.lr * updates[i][d];
            }
        }
    }

    /// Run for `config.n_steps` steps and return final particle positions.
    pub fn run(&mut self, log_prob_grad: &dyn Fn(&[f64]) -> Vec<f64>) -> Vec<Vec<f64>> {
        for _ in 0..self.config.n_steps {
            self.update(log_prob_grad);
        }
        self.particles.clone()
    }

    /// Empirical mean over particles.
    pub fn mean(&self) -> Vec<f64> {
        if self.particles.is_empty() {
            return Vec::new();
        }
        let dim = self.particles[0].len();
        let n = self.particles.len() as f64;
        let mut m = vec![0.0_f64; dim];
        for p in &self.particles {
            for (d, &v) in p.iter().enumerate() {
                m[d] += v / n;
            }
        }
        m
    }

    /// Empirical variance over particles.
    pub fn variance(&self) -> Vec<f64> {
        if self.particles.is_empty() {
            return Vec::new();
        }
        let mu = self.mean();
        let dim = mu.len();
        let n = self.particles.len() as f64;
        let mut var = vec![0.0_f64; dim];
        for p in &self.particles {
            for (d, &v) in p.iter().enumerate() {
                let diff = v - mu[d];
                var[d] += diff * diff / n;
            }
        }
        var
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  VI Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic summary for a variational inference fit.
#[derive(Debug, Clone)]
pub struct ViDiagnostics {
    /// Pareto `k` estimate for importance-weighted stability.
    /// Values > 0.7 indicate unreliable IS estimates.
    pub pareto_k: f64,
    /// Effective sample size of the importance weights.
    pub effective_sample_size: f64,
    /// Monte Carlo variance of the ELBO estimator.
    pub elbo_variance: f64,
    /// KL divergence from `q` to `N(0, I)`.
    pub kl_divergence: f64,
}

/// Fit a Pareto tail to the top 20% of importance weights.
///
/// Returns the estimated shape parameter `k` via moment matching.
pub fn compute_pareto_k(log_weights: &[f64]) -> f64 {
    let n = log_weights.len();
    if n < 5 {
        return 0.0;
    }

    // Normalize weights
    let log_sum = vi_log_sum_exp(log_weights);
    let mut weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - log_sum).exp()).collect();
    weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Top 20% of weights (largest values)
    let tail_start = (n as f64 * 0.8) as usize;
    let tail: Vec<f64> = weights[tail_start..].to_vec();
    let m = tail.len();
    if m < 2 {
        return 0.0;
    }

    // Threshold = min of tail
    let threshold = tail[0];
    let exceedances: Vec<f64> = tail.iter().map(|&w| w - threshold).collect();

    // Moment matching for Generalized Pareto Distribution
    // k = 0.5 * (1 - mean^2/var)  (Hill estimator approximation)
    let mean_e: f64 = exceedances.iter().sum::<f64>() / m as f64;
    let var_e: f64 = exceedances
        .iter()
        .map(|&e| (e - mean_e).powi(2))
        .sum::<f64>()
        / m as f64;

    if var_e < 1e-30 || mean_e < 1e-30 {
        return 0.0;
    }

    0.5 * (1.0 - mean_e * mean_e / var_e)
}

/// Effective sample size from log importance weights.
///
/// `ESS = exp(2 * log_sum_exp(log_w) - log_sum_exp(2 * log_w))`
pub fn effective_sample_size(log_weights: &[f64]) -> f64 {
    if log_weights.is_empty() {
        return 0.0;
    }
    let lse1 = vi_log_sum_exp(log_weights);
    let log_w2: Vec<f64> = log_weights.iter().map(|&lw| 2.0 * lw).collect();
    let lse2 = vi_log_sum_exp(&log_w2);
    let ess = (2.0 * lse1 - lse2).exp();
    ess.min(log_weights.len() as f64)
}

/// Compute full [`ViDiagnostics`] for a fitted mean-field variational distribution.
pub fn diagnose_vi(
    q: &MeanFieldGaussian,
    log_joint: &dyn Fn(&[f64]) -> f64,
    n_samples: usize,
    rng: &mut StdRng,
) -> ViDiagnostics {
    let n = n_samples.max(2);
    let mut elbos = Vec::with_capacity(n);
    let mut log_weights = Vec::with_capacity(n);

    for _ in 0..n {
        let eps = normal_vec(q.dim, rng);
        let z = q.sample(&eps);
        let lp = log_joint(&z);
        let lq = q.log_prob(&z);
        if lp.is_finite() && lq.is_finite() {
            elbos.push(lp - lq);
            log_weights.push(lp - lq); // importance weight = log p/q
        }
    }

    let elbo_mean = if elbos.is_empty() {
        0.0
    } else {
        elbos.iter().sum::<f64>() / elbos.len() as f64
    };
    let elbo_variance = if elbos.len() < 2 {
        0.0
    } else {
        let n = elbos.len() as f64;
        elbos.iter().map(|&e| (e - elbo_mean).powi(2)).sum::<f64>() / n
    };

    let pareto_k = compute_pareto_k(&log_weights);
    let ess = effective_sample_size(&log_weights);
    let kl = q.kl_to_standard_normal();

    ViDiagnostics {
        pareto_k,
        effective_sample_size: ess,
        elbo_variance,
        kl_divergence: kl,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  ViDistribution enum (variational family selector)
// ─────────────────────────────────────────────────────────────────────────────

/// Variational distribution family selector.
#[derive(Debug, Clone, PartialEq)]
pub enum ViDistribution {
    /// Fully factored: `q(z) = Π_i q_i(z_i)`.
    MeanField,
    /// Full-rank Gaussian with Cholesky covariance.
    FullRankGaussian,
    /// Low-rank + diagonal Gaussian approximation.
    LowRank { rank: usize },
}

#[cfg(test)]
mod tests;
