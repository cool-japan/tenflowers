//! Regime detection: HMM (Baum-Welch/Viterbi), change-point detection,
//! volatility regime, market regime classifier, regime switching model.

use super::order_book::FinResult;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Hidden Markov Model
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian HMM with N states for 1-D or multi-D observations.
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub obs_dim: usize,
    /// Initial state probabilities.
    pub pi: Vec<f64>,
    /// Transition matrix A\[i\]\[j\] = P(s_t=j | s_{t-1}=i).
    pub trans: Vec<f64>, // [n_states × n_states]
    /// Gaussian means [n_states × obs_dim].
    pub means: Vec<f64>,
    /// Gaussian variances [n_states × obs_dim].
    pub vars: Vec<f64>,
}

impl HiddenMarkovModel {
    pub fn new(n_states: usize, obs_dim: usize, seed: u64) -> FinResult<Self> {
        if n_states == 0 || obs_dim == 0 {
            return Err("HMM: n_states and obs_dim must be > 0".to_string());
        }
        let mut rng = StdRng::seed_from_u64(seed);
        // Uniform initial distribution
        let pi = vec![1.0 / n_states as f64; n_states];
        // Random transition matrix (row-normalised)
        let mut trans = vec![0.0_f64; n_states * n_states];
        for i in 0..n_states {
            let mut row_sum = 0.0;
            for j in 0..n_states {
                let v: f64 = rng.random::<f64>() + 0.1;
                trans[i * n_states + j] = v;
                row_sum += v;
            }
            for j in 0..n_states {
                trans[i * n_states + j] /= row_sum;
            }
        }
        // Spread means across range [-1, 1] per state
        let mut means = vec![0.0_f64; n_states * obs_dim];
        for s in 0..n_states {
            for d in 0..obs_dim {
                means[s * obs_dim + d] =
                    (s as f64 / n_states as f64) * 2.0 - 1.0 + rng.random::<f64>() * 0.1;
            }
        }
        let vars = vec![1.0_f64; n_states * obs_dim];
        Ok(Self {
            n_states,
            obs_dim,
            pi,
            trans,
            means,
            vars,
        })
    }

    pub(super) fn log_obs_prob(&self, state: usize, obs: &[f64]) -> f64 {
        let mut log_p = 0.0_f64;
        for d in 0..self.obs_dim {
            let mu = self.means[state * self.obs_dim + d];
            let var = self.vars[state * self.obs_dim + d].max(1e-8);
            let diff = obs[d] - mu;
            log_p -= 0.5 * (diff * diff / var + (2.0 * std::f64::consts::PI * var).ln());
        }
        log_p
    }

    /// Baum-Welch EM training.
    pub fn fit_baum_welch(&mut self, observations: &[Vec<f64>], n_iter: usize) -> FinResult<()> {
        let t_len = observations.len();
        if t_len == 0 {
            return Err("HMM: empty observations".to_string());
        }
        let s = self.n_states;

        for _iter in 0..n_iter {
            // E-step: forward-backward
            let (log_alpha, log_scale) = self.forward_pass(observations);
            let log_beta = self.backward_pass(observations, &log_scale);

            // Compute gamma (posterior state probabilities)
            let mut gamma = vec![0.0_f64; t_len * s];
            for t in 0..t_len {
                let mut log_norm = f64::NEG_INFINITY;
                for i in 0..s {
                    let v = log_alpha[t * s + i] + log_beta[t * s + i];
                    log_norm = log_sum_exp(log_norm, v);
                }
                for i in 0..s {
                    let log_g = log_alpha[t * s + i] + log_beta[t * s + i] - log_norm;
                    gamma[t * s + i] = log_g.exp();
                }
            }

            // Compute xi (pairwise posteriors) for transition update
            let mut xi_sum = vec![0.0_f64; s * s];
            for t in 0..t_len.saturating_sub(1) {
                let mut log_norm = f64::NEG_INFINITY;
                let mut xi_t = vec![0.0_f64; s * s];
                for i in 0..s {
                    for j in 0..s {
                        let log_obs_j = self.log_obs_prob(j, &observations[t + 1]);
                        let v = log_alpha[t * s + i]
                            + self.trans[i * s + j].max(1e-300).ln()
                            + log_obs_j
                            + log_beta[(t + 1) * s + j];
                        xi_t[i * s + j] = v;
                        log_norm = log_sum_exp(log_norm, v);
                    }
                }
                for i in 0..s {
                    for j in 0..s {
                        xi_sum[i * s + j] += (xi_t[i * s + j] - log_norm).exp();
                    }
                }
            }

            // M-step: update parameters
            // Update pi
            for i in 0..s {
                self.pi[i] = gamma[i].max(1e-12);
            }
            let pi_sum: f64 = self.pi.iter().sum();
            for p in &mut self.pi {
                *p /= pi_sum;
            }

            // Update transition matrix
            for i in 0..s {
                let row_sum: f64 = xi_sum[i * s..i * s + s].iter().sum::<f64>().max(1e-12);
                for j in 0..s {
                    self.trans[i * s + j] = xi_sum[i * s + j] / row_sum;
                }
            }

            // Update means and variances
            for i in 0..s {
                let gamma_sum: f64 = (0..t_len).map(|t| gamma[t * s + i]).sum::<f64>().max(1e-12);
                for d in 0..self.obs_dim {
                    let new_mean = (0..t_len)
                        .map(|t| gamma[t * s + i] * observations[t][d])
                        .sum::<f64>()
                        / gamma_sum;
                    let new_var = (0..t_len)
                        .map(|t| {
                            let diff = observations[t][d] - new_mean;
                            gamma[t * s + i] * diff * diff
                        })
                        .sum::<f64>()
                        / gamma_sum;
                    self.means[i * self.obs_dim + d] = new_mean;
                    self.vars[i * self.obs_dim + d] = new_var.max(1e-6);
                }
            }
        }
        Ok(())
    }

    pub(super) fn forward_pass(&self, obs: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let t_len = obs.len();
        let s = self.n_states;
        let mut log_alpha = vec![f64::NEG_INFINITY; t_len * s];
        let mut log_scale = vec![0.0_f64; t_len];

        // Initialise
        for i in 0..s {
            log_alpha[i] = self.pi[i].max(1e-300).ln() + self.log_obs_prob(i, &obs[0]);
        }
        let log_norm0 = log_alpha[..s]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, log_sum_exp);
        log_scale[0] = log_norm0;
        for i in 0..s {
            log_alpha[i] -= log_norm0;
        }

        // Recurse
        for t in 1..t_len {
            for j in 0..s {
                let mut acc = f64::NEG_INFINITY;
                for i in 0..s {
                    acc = log_sum_exp(
                        acc,
                        log_alpha[(t - 1) * s + i] + self.trans[i * s + j].max(1e-300).ln(),
                    );
                }
                log_alpha[t * s + j] = acc + self.log_obs_prob(j, &obs[t]);
            }
            let log_norm = log_alpha[t * s..t * s + s]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, log_sum_exp);
            log_scale[t] = log_norm;
            for j in 0..s {
                log_alpha[t * s + j] -= log_norm;
            }
        }
        (log_alpha, log_scale)
    }

    fn backward_pass(&self, obs: &[Vec<f64>], log_scale: &[f64]) -> Vec<f64> {
        let t_len = obs.len();
        let s = self.n_states;
        let mut log_beta = vec![0.0_f64; t_len * s];

        // log_beta at t_len-1 = log 1 = 0
        for t in (0..t_len.saturating_sub(1)).rev() {
            for i in 0..s {
                let mut acc = f64::NEG_INFINITY;
                for j in 0..s {
                    let v = self.trans[i * s + j].max(1e-300).ln()
                        + self.log_obs_prob(j, &obs[t + 1])
                        + log_beta[(t + 1) * s + j];
                    acc = log_sum_exp(acc, v);
                }
                log_beta[t * s + i] = acc - log_scale[t + 1];
            }
        }
        log_beta
    }

    /// Viterbi decoding: most likely state sequence.
    pub fn viterbi(&self, observations: &[Vec<f64>]) -> FinResult<Vec<usize>> {
        let t_len = observations.len();
        if t_len == 0 {
            return Err("HMM: empty observations for viterbi".to_string());
        }
        let s = self.n_states;
        let mut delta = vec![f64::NEG_INFINITY; t_len * s];
        let mut psi = vec![0usize; t_len * s];

        for i in 0..s {
            delta[i] = self.pi[i].max(1e-300).ln() + self.log_obs_prob(i, &observations[0]);
        }
        for t in 1..t_len {
            for j in 0..s {
                let mut best_val = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for i in 0..s {
                    let v = delta[(t - 1) * s + i] + self.trans[i * s + j].max(1e-300).ln();
                    if v > best_val {
                        best_val = v;
                        best_prev = i;
                    }
                }
                delta[t * s + j] = best_val + self.log_obs_prob(j, &observations[t]);
                psi[t * s + j] = best_prev;
            }
        }
        // Backtrack
        let mut path = vec![0usize; t_len];
        let last_row = &delta[(t_len - 1) * s..t_len * s];
        path[t_len - 1] = last_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for t in (0..t_len - 1).rev() {
            path[t] = psi[(t + 1) * s + path[t + 1]];
        }
        Ok(path)
    }

    /// Log-likelihood of the observation sequence.
    pub fn log_likelihood(&self, observations: &[Vec<f64>]) -> FinResult<f64> {
        if observations.is_empty() {
            return Err("HMM: empty observations".to_string());
        }
        let (_, log_scale) = self.forward_pass(observations);
        Ok(log_scale.iter().sum())
    }
}

#[inline]
pub(super) fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// Change Point Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Bayesian Online Change Point Detection (BOCPD) with Gaussian likelihood.
///
/// Reference: Adams & MacKay (2007).
#[derive(Debug, Clone)]
pub struct ChangePointDetector {
    /// Geometric hazard rate (probability of change point at each step).
    pub hazard: f64,
    /// Posterior run-length distribution (unnormalised log weights).
    log_r: Vec<f64>,
    /// Sufficient statistics per run-length hypothesis (mean, variance, count).
    hyp_mu: Vec<f64>,
    hyp_kappa: Vec<f64>,
    hyp_alpha: Vec<f64>,
    hyp_beta: Vec<f64>,
    /// Prior hyperparameters (Normal-Gamma).
    prior_mu: f64,
    prior_kappa: f64,
    prior_alpha: f64,
    prior_beta: f64,
    t: usize,
}

impl ChangePointDetector {
    pub fn new(
        hazard: f64,
        prior_mu: f64,
        prior_kappa: f64,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Self {
        let hazard = hazard.clamp(1e-6, 1.0 - 1e-6);
        Self {
            hazard,
            log_r: vec![0.0], // log P(r_0 = 0) = log 1
            hyp_mu: vec![prior_mu],
            hyp_kappa: vec![prior_kappa],
            hyp_alpha: vec![prior_alpha],
            hyp_beta: vec![prior_beta],
            prior_mu,
            prior_kappa,
            prior_alpha,
            prior_beta,
            t: 0,
        }
    }

    /// Process one new observation; returns the posterior run-length distribution.
    pub fn update(&mut self, x: f64) -> Vec<f64> {
        let n = self.log_r.len();
        // Predictive probabilities (Student-t) for each hypothesis
        let log_pred: Vec<f64> = (0..n).map(|i| self.log_student_t(x, i)).collect();

        // New hypotheses after growth
        let mut new_log_r = Vec::with_capacity(n + 1);
        let mut new_mu = Vec::with_capacity(n + 1);
        let mut new_kappa = Vec::with_capacity(n + 1);
        let mut new_alpha = Vec::with_capacity(n + 1);
        let mut new_beta = Vec::with_capacity(n + 1);

        // Change-point hypothesis (run-length = 0)
        let log_h = self.hazard.ln();
        let mut log_cp = f64::NEG_INFINITY;
        for i in 0..n {
            log_cp = log_sum_exp(log_cp, self.log_r[i] + log_pred[i] + log_h);
        }
        new_log_r.push(log_cp);
        new_mu.push(self.prior_mu);
        new_kappa.push(self.prior_kappa);
        new_alpha.push(self.prior_alpha);
        new_beta.push(self.prior_beta);

        // Growth hypotheses (run-length increases by 1)
        let log_no_h = (1.0 - self.hazard).ln();
        for i in 0..n {
            new_log_r.push(self.log_r[i] + log_pred[i] + log_no_h);
            // Update Normal-Gamma parameters
            let kappa_n = self.hyp_kappa[i] + 1.0;
            let mu_n = (self.hyp_kappa[i] * self.hyp_mu[i] + x) / kappa_n;
            let alpha_n = self.hyp_alpha[i] + 0.5;
            let beta_n =
                self.hyp_beta[i] + 0.5 * self.hyp_kappa[i] * (x - self.hyp_mu[i]).powi(2) / kappa_n;
            new_mu.push(mu_n);
            new_kappa.push(kappa_n);
            new_alpha.push(alpha_n);
            new_beta.push(beta_n.max(1e-12));
        }

        // Normalise
        let log_norm = new_log_r
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, log_sum_exp);
        for v in &mut new_log_r {
            *v -= log_norm;
        }

        self.log_r = new_log_r;
        self.hyp_mu = new_mu;
        self.hyp_kappa = new_kappa;
        self.hyp_alpha = new_alpha;
        self.hyp_beta = new_beta;
        self.t += 1;

        self.log_r.iter().map(|&v| v.exp()).collect()
    }

    fn log_student_t(&self, x: f64, i: usize) -> f64 {
        // Student-t predictive with 2*alpha dof
        let alpha = self.hyp_alpha[i];
        let beta = self.hyp_beta[i];
        let kappa = self.hyp_kappa[i];
        let mu = self.hyp_mu[i];
        let dof = 2.0 * alpha;
        let scale2 = beta * (kappa + 1.0) / (alpha * kappa);
        let z = (x - mu) / scale2.sqrt().max(1e-12);
        let log_c = lgamma(alpha + 0.5)
            - lgamma(alpha)
            - 0.5 * (std::f64::consts::PI * dof).ln()
            - 0.5 * scale2.max(1e-12).ln();
        log_c - (alpha + 0.5) * (1.0 + z * z / dof).ln()
    }
}

/// Log-gamma approximation (Stirling / Lanczos simplified).
fn lgamma(x: f64) -> f64 {
    // Use Lanczos approximation g=5, n=6
    if x < 0.5 {
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().max(1e-300).ln()
            - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let coeffs = [
            76.18009172947146_f64,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.001208650973866179,
            -0.000005395239384953,
        ];
        let mut ser = 1.000000000190015_f64;
        let mut tmp = x;
        for c in &coeffs {
            tmp += 1.0;
            ser += c / tmp;
        }
        let t = x + 5.5;
        (2.0 * std::f64::consts::PI).sqrt().ln() + ser.ln() + (x + 0.5) * t.ln() - t
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Volatility Regime Detector
// ─────────────────────────────────────────────────────────────────────────────

/// GARCH(1,1) volatility model.
///
/// σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
#[derive(Debug, Clone)]
pub struct VolatilityRegimeDetector {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl VolatilityRegimeDetector {
    pub fn new() -> Self {
        Self {
            omega: 0.0001,
            alpha: 0.05,
            beta: 0.90,
        }
    }

    /// Fit GARCH(1,1) parameters via quasi-MLE (gradient-free grid search + refinement).
    pub fn fit(&mut self, returns: &[f64], n_iter: usize) -> FinResult<(f64, f64, f64)> {
        if returns.len() < 3 {
            return Err("GARCH: need at least 3 returns".to_string());
        }
        let var_ret = returns.iter().map(|&r| r * r).sum::<f64>() / returns.len() as f64;

        // Initial parameter values
        let mut omega = var_ret * 0.05;
        let mut alpha = 0.05_f64;
        let mut beta = 0.90_f64;

        let lr = 0.001_f64;
        for _ in 0..n_iter {
            // Compute log-likelihood
            let (ll, grad_o, grad_a, grad_b) = self.garch_grad(returns, omega, alpha, beta);
            let _ = ll;
            // Gradient ascent
            omega += lr * grad_o;
            alpha += lr * grad_a;
            beta += lr * grad_b;
            // Constraints: omega > 0, alpha ≥ 0, beta ≥ 0, alpha + beta < 1
            omega = omega.max(1e-10);
            alpha = alpha.clamp(0.001, 0.9);
            beta = beta.clamp(0.001, 0.999);
            if alpha + beta >= 1.0 {
                let s = alpha + beta + 1e-4;
                alpha /= s;
                beta /= s;
            }
        }
        self.omega = omega;
        self.alpha = alpha;
        self.beta = beta;
        Ok((omega, alpha, beta))
    }

    fn garch_grad(
        &self,
        returns: &[f64],
        omega: f64,
        alpha: f64,
        beta: f64,
    ) -> (f64, f64, f64, f64) {
        let n = returns.len();
        let var_init = returns.iter().map(|&r| r * r).sum::<f64>() / n as f64;
        let mut sigma2 = var_init;
        let mut ll = 0.0_f64;
        let mut d_omega = 0.0_f64;
        let mut d_alpha = 0.0_f64;
        let mut d_beta = 0.0_f64;
        let mut ds2_domega = 0.0_f64;
        let mut ds2_dalpha = 0.0_f64;
        let mut ds2_dbeta = 0.0_f64;

        for t in 1..n {
            let eps2 = returns[t - 1].powi(2);
            let new_s2 = omega + alpha * eps2 + beta * sigma2;
            let new_ds2_domega = 1.0 + beta * ds2_domega;
            let new_ds2_dalpha = eps2 + beta * ds2_dalpha;
            let new_ds2_dbeta = sigma2 + beta * ds2_dbeta;
            sigma2 = new_s2.max(1e-12);
            ds2_domega = new_ds2_domega;
            ds2_dalpha = new_ds2_dalpha;
            ds2_dbeta = new_ds2_dbeta;

            let r2 = returns[t].powi(2);
            ll += -0.5 * (sigma2.ln() + r2 / sigma2);
            let dl_ds2 = -0.5 * (1.0 / sigma2 - r2 / sigma2.powi(2));
            d_omega += dl_ds2 * ds2_domega;
            d_alpha += dl_ds2 * ds2_dalpha;
            d_beta += dl_ds2 * ds2_dbeta;
        }
        (ll, d_omega, d_alpha, d_beta)
    }

    /// Forecast variance t steps ahead: unconditional long-run + decay.
    pub fn forecast_variance(&self, t_ahead: usize) -> f64 {
        let long_run = self.omega / (1.0 - self.alpha - self.beta).max(1e-12);
        let decay = (self.alpha + self.beta).powi(t_ahead as i32);
        long_run * (1.0 - decay) + decay * long_run
    }
}

impl Default for VolatilityRegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Market Regime Classifier
// ─────────────────────────────────────────────────────────────────────────────

/// Classifies market regime from rolling statistics.
///
/// Regimes: 0 = trending, 1 = mean-reverting, 2 = volatile.
#[derive(Debug, Clone)]
pub struct MarketRegimeClassifier {
    pub window: usize,
    pub vol_threshold: f64,
    pub trend_threshold: f64,
}

impl MarketRegimeClassifier {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            vol_threshold: 0.02,
            trend_threshold: 0.5,
        }
    }

    /// Classify a sequence of returns.
    ///
    /// Returns a vector of regime labels (0/1/2) of length `returns.len()`.
    pub fn classify(&self, returns: &[f64]) -> FinResult<Vec<usize>> {
        let n = returns.len();
        if n < self.window {
            return Err("MarketRegimeClassifier: not enough data".to_string());
        }
        let mut labels = vec![1usize; n]; // default: mean-reverting
        for t in self.window..n {
            let window = &returns[t - self.window..t];
            let mean = window.iter().sum::<f64>() / self.window as f64;
            let var = window.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / self.window as f64;
            let vol = var.sqrt();
            // Autocorrelation lag-1 as trend indicator
            let ac = autocorr_lag1(window);
            labels[t] = if vol > self.vol_threshold {
                2 // volatile
            } else if ac > self.trend_threshold {
                0 // trending
            } else {
                1 // mean-reverting
            };
        }
        Ok(labels)
    }
}

fn autocorr_lag1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>();
    if var == 0.0 {
        return 0.0;
    }
    let cov: f64 = (0..n - 1).map(|i| (x[i] - mean) * (x[i + 1] - mean)).sum();
    cov / var
}

// ─────────────────────────────────────────────────────────────────────────────
// Regime Switching Model
// ─────────────────────────────────────────────────────────────────────────────

/// 2-state Markov switching model (Hamilton, 1989).
#[derive(Debug, Clone)]
pub struct RegimeSwitchingModel {
    /// Transition probabilities: p\[i\]\[j\] = P(state_t = j | state_{t-1} = i)
    pub trans: [[f64; 2]; 2],
    /// Gaussian means per state.
    pub means: [f64; 2],
    /// Gaussian variances per state.
    pub vars: [f64; 2],
    /// Last filtered probabilities.
    filtered: Vec<[f64; 2]>,
}

impl RegimeSwitchingModel {
    pub fn new() -> Self {
        Self {
            trans: [[0.9, 0.1], [0.1, 0.9]],
            means: [0.001, -0.001],
            vars: [0.01_f64.powi(2), 0.03_f64.powi(2)],
            filtered: Vec::new(),
        }
    }

    /// Fit via Baum-Welch (2-state specialisation).
    pub fn fit(&mut self, returns: &[f64]) -> FinResult<()> {
        if returns.len() < 5 {
            return Err("RegimeSwitching: need at least 5 observations".to_string());
        }
        let t_len = returns.len();
        let mut hmm = HiddenMarkovModel {
            n_states: 2,
            obs_dim: 1,
            pi: vec![0.5, 0.5],
            trans: vec![
                self.trans[0][0],
                self.trans[0][1],
                self.trans[1][0],
                self.trans[1][1],
            ],
            means: vec![self.means[0], self.means[1]],
            vars: vec![self.vars[0], self.vars[1]],
        };
        let obs: Vec<Vec<f64>> = returns.iter().map(|&r| vec![r]).collect();
        hmm.fit_baum_welch(&obs, 20)?;
        self.means = [hmm.means[0], hmm.means[1]];
        self.vars = [hmm.vars[0].max(1e-12), hmm.vars[1].max(1e-12)];
        self.trans = [[hmm.trans[0], hmm.trans[1]], [hmm.trans[2], hmm.trans[3]]];
        // Re-run forward to compute filtered probabilities
        let mut filtered = Vec::with_capacity(t_len);
        let mut prob = [0.5_f64, 0.5_f64];
        for r in returns {
            let p0 = gaussian_pdf(*r, self.means[0], self.vars[0]);
            let p1 = gaussian_pdf(*r, self.means[1], self.vars[1]);
            let pred = [
                prob[0] * self.trans[0][0] + prob[1] * self.trans[1][0],
                prob[0] * self.trans[0][1] + prob[1] * self.trans[1][1],
            ];
            let lik = [pred[0] * p0, pred[1] * p1];
            let s = lik[0] + lik[1];
            prob = if s > 0.0 {
                [lik[0] / s, lik[1] / s]
            } else {
                [0.5, 0.5]
            };
            filtered.push(prob);
        }
        self.filtered = filtered;
        Ok(())
    }

    /// Probability of being in regime 1 at time t.
    pub fn regime_probability(&self, t: usize) -> f64 {
        self.filtered.get(t).map(|p| p[1]).unwrap_or(0.5)
    }
}

impl Default for RegimeSwitchingModel {
    fn default() -> Self {
        Self::new()
    }
}

pub(super) fn gaussian_pdf(x: f64, mu: f64, var: f64) -> f64 {
    let v = var.max(1e-12);
    let diff = x - mu;
    (-0.5 * diff * diff / v).exp() / (2.0 * std::f64::consts::PI * v).sqrt()
}
