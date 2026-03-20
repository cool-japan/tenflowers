//! SGLD and SGHMC samplers for Bayesian deep learning.

use super::helpers::sample_normal;
use super::shared::BdlMlp;
use scirs2_core::random::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// SGLD
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SGLD sampler.
#[derive(Clone, Debug)]
pub struct SgldConfig {
    pub lr: f64,
    pub n_steps: usize,
    pub burn_in: usize,
    pub n_data: usize,
    pub prior_std: f64,
    pub thinning: usize,
}

impl Default for SgldConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            n_steps: 1000,
            burn_in: 500,
            n_data: 100,
            prior_std: 1.0,
            thinning: 10,
        }
    }
}

/// SGLD posterior sampler (Welling & Teh 2011).
pub struct SgldSampler {
    pub config: SgldConfig,
    pub samples: Vec<Vec<f64>>,
}

impl SgldSampler {
    pub fn new(config: SgldConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

    /// Single SGLD step.
    /// θ_{t+1} = θ_t + (lr/2)*(prior_grad + grad) + N(0, lr)
    pub fn step(&self, params: &[f64], grad: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let lr = self.config.lr;
        let prior_var = self.config.prior_std * self.config.prior_std;
        let noise_std = lr.sqrt();
        params
            .iter()
            .zip(grad.iter())
            .map(|(&p, &g)| {
                let prior_grad = -p / prior_var;
                let drift = (lr / 2.0) * (prior_grad + g);
                let noise = noise_std * sample_normal(rng);
                p + drift + noise
            })
            .collect()
    }

    /// Run full SGLD chain, populating `self.samples`.
    pub fn run(
        &mut self,
        model: &mut BdlMlp,
        x_data: &[Vec<f64>],
        y_data: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        self.samples.clear();
        let eps = 1e-5;
        let mut params = model.params_flat();
        for step in 0..self.config.n_steps {
            let loss_grad = {
                let mut tmp = model.clone();
                tmp.set_params(&params);
                tmp.batch_gradient_fd(x_data, y_data, eps)
            };
            let log_lik_grad: Vec<f64> = loss_grad.iter().map(|g| -g).collect();
            params = self.step(&params, &log_lik_grad, rng);
            if step >= self.config.burn_in {
                let post_burn = step - self.config.burn_in;
                if post_burn % self.config.thinning == 0 {
                    self.samples.push(params.clone());
                }
            }
        }
        model.set_params(&params);
        params
    }

    /// Ensemble prediction: (mean, variance) over collected samples.
    pub fn predict_ensemble(&self, model: &mut BdlMlp, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if self.samples.is_empty() {
            let pred = model.forward(x);
            let zeros = vec![0.0; pred.len()];
            return (pred, zeros);
        }
        let preds: Vec<Vec<f64>> = self
            .samples
            .iter()
            .map(|s| {
                model.set_params(s);
                model.forward(x)
            })
            .collect();
        ensemble_stats(&preds)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGHMC
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SGHMC sampler.
#[derive(Clone, Debug)]
pub struct SghcmConfig {
    pub lr: f64,
    pub friction: f64,
    pub n_steps: usize,
    pub burn_in: usize,
    pub n_data: usize,
    pub prior_std: f64,
    pub thinning: usize,
}

impl Default for SghcmConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            friction: 0.01,
            n_steps: 1000,
            burn_in: 500,
            n_data: 100,
            prior_std: 1.0,
            thinning: 10,
        }
    }
}

/// SGHMC posterior sampler with momentum (Chen et al. 2014).
pub struct SghcmSampler {
    pub config: SghcmConfig,
    pub samples: Vec<Vec<f64>>,
    pub momentum: Option<Vec<f64>>,
}

impl SghcmSampler {
    pub fn new(config: SghcmConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            momentum: None,
        }
    }

    /// Single SGHMC step.
    /// r_{t+1} = r_t - lr*grad - lr*C*r_t + N(0, 2*lr*C)
    /// θ_{t+1} = θ_t + lr * r_{t+1}
    pub fn step(&mut self, params: &[f64], grad: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let lr = self.config.lr;
        let c = self.config.friction;
        let noise_std = (2.0 * lr * c).sqrt();
        let prior_var = self.config.prior_std * self.config.prior_std;
        let n = params.len();
        let r = self.momentum.get_or_insert_with(|| vec![0.0; n]);
        for i in 0..n {
            let prior_grad = params[i] / prior_var;
            let total_grad = grad[i] + prior_grad;
            r[i] = r[i] - lr * total_grad - lr * c * r[i] + noise_std * sample_normal(rng);
        }
        params
            .iter()
            .zip(r.iter())
            .map(|(&p, &ri)| p + lr * ri)
            .collect()
    }

    /// Run full SGHMC chain.
    pub fn run(
        &mut self,
        model: &mut BdlMlp,
        x_data: &[Vec<f64>],
        y_data: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Vec<f64> {
        self.samples.clear();
        self.momentum = None;
        let eps = 1e-5;
        let mut params = model.params_flat();
        for step in 0..self.config.n_steps {
            let loss_grad = {
                let mut tmp = model.clone();
                tmp.set_params(&params);
                tmp.batch_gradient_fd(x_data, y_data, eps)
            };
            params = self.step(&params, &loss_grad, rng);
            if step >= self.config.burn_in {
                let post_burn = step - self.config.burn_in;
                if post_burn % self.config.thinning == 0 {
                    self.samples.push(params.clone());
                }
            }
        }
        model.set_params(&params);
        params
    }

    /// Ensemble prediction: (mean, variance).
    pub fn predict_ensemble(&self, model: &mut BdlMlp, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        if self.samples.is_empty() {
            let pred = model.forward(x);
            let zeros = vec![0.0; pred.len()];
            return (pred, zeros);
        }
        let preds: Vec<Vec<f64>> = self
            .samples
            .iter()
            .map(|s| {
                model.set_params(s);
                model.forward(x)
            })
            .collect();
        ensemble_stats(&preds)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared ensemble statistics helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute (mean, variance) across a set of prediction vectors.
pub(super) fn ensemble_stats(preds: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let out_dim = preds[0].len();
    let n = preds.len() as f64;
    let mean: Vec<f64> = (0..out_dim)
        .map(|d| preds.iter().map(|p| p[d]).sum::<f64>() / n)
        .collect();
    let variance: Vec<f64> = (0..out_dim)
        .map(|d| {
            let m = mean[d];
            preds.iter().map(|p| (p[d] - m).powi(2)).sum::<f64>() / n.max(1.0)
        })
        .collect();
    (mean, variance)
}
