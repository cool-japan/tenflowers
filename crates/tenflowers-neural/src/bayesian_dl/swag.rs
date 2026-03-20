//! SWAG (Stochastic Weight Averaging Gaussian) — Maddox et al. 2019.

use super::helpers::sample_normal;
use super::mcmc::ensemble_stats;
use super::shared::BdlMlp;
use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

/// Configuration for SWAG.
#[derive(Clone, Debug)]
pub struct SwagConfig {
    /// Start collecting SWA snapshots after this epoch.
    pub swa_start_epoch: usize,
    /// Learning rate for the SWA gradient step before each snapshot.
    pub swa_lr: f64,
    /// Total training epochs.
    pub n_epochs: usize,
    /// Low-rank approximation rank (default 20).
    pub max_rank: usize,
    /// Posterior scale (default 0.5).
    pub scale: f64,
    /// Prior std (tracked for reference).
    pub prior_std: f64,
}

impl Default for SwagConfig {
    fn default() -> Self {
        Self {
            swa_start_epoch: 5,
            swa_lr: 1e-3,
            n_epochs: 20,
            max_rank: 20,
            scale: 0.5,
            prior_std: 1.0,
        }
    }
}

/// SWAG model: running SWA mean + second moment + low-rank deviations.
pub struct SwagModel {
    pub mean_params: Vec<f64>,
    pub sq_mean_params: Vec<f64>,
    pub deviations: Vec<Vec<f64>>,
    pub n_collected: usize,
    pub config: SwagConfig,
}

impl SwagModel {
    pub fn new(n_params: usize, config: SwagConfig) -> Self {
        Self {
            mean_params: vec![0.0; n_params],
            sq_mean_params: vec![0.0; n_params],
            deviations: Vec::new(),
            n_collected: 0,
            config,
        }
    }

    /// Collect a weight snapshot for the given epoch.
    pub fn collect(
        &mut self,
        model: &BdlMlp,
        x_data: &[Vec<f64>],
        y_data: &[Vec<f64>],
        epoch: usize,
        lr: f64,
        rng: &mut StdRng,
    ) {
        if epoch < self.config.swa_start_epoch {
            return;
        }
        let eps_fd = 1e-5;
        let mut m = model.clone();
        let params_before = m.params_flat();
        let grad = m.batch_gradient_fd(x_data, y_data, eps_fd);
        let new_params: Vec<f64> = params_before
            .iter()
            .zip(grad.iter())
            .map(|(&p, &g)| p - self.config.swa_lr * g)
            .collect();
        m.set_params(&new_params);
        let snapshot = m.params_flat();
        let n = snapshot.len();
        let nc = self.n_collected as f64;

        for i in 0..n {
            let old_mean = self.mean_params[i];
            self.mean_params[i] = (old_mean * nc + snapshot[i]) / (nc + 1.0);
            self.sq_mean_params[i] =
                (self.sq_mean_params[i] * nc + snapshot[i].powi(2)) / (nc + 1.0);
        }

        let dev: Vec<f64> = snapshot
            .iter()
            .zip(self.mean_params.iter())
            .map(|(&s, &m_)| s - m_)
            .collect();

        if self.deviations.len() >= self.config.max_rank {
            self.deviations.remove(0);
        }
        self.deviations.push(dev);
        self.n_collected += 1;

        // Consume rng to maintain RNG state consistency
        let _: f64 = rng.random();
        let _ = lr;
    }

    /// Diagonal variance: `max(0, sq_mean_i - mean_i²)`.
    pub fn diagonal_variance(&self) -> Vec<f64> {
        self.mean_params
            .iter()
            .zip(self.sq_mean_params.iter())
            .map(|(&m, &sq)| (sq - m * m).max(0.0))
            .collect()
    }

    /// Sample from the SWAG posterior.
    /// θ = mean + scale*(z1⊙diag_std + (1/√(2*(rank-1)))*Σ_k dev_k * z2_k)
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        let diag_var = self.diagonal_variance();
        let n = self.mean_params.len();
        let rank = self.deviations.len();
        let z1: Vec<f64> = (0..n).map(|_| sample_normal(rng)).collect();
        let diag_term: Vec<f64> = z1
            .iter()
            .zip(diag_var.iter())
            .map(|(&z, &v)| z * v.sqrt())
            .collect();
        let mut lr_term = vec![0.0f64; n];
        if rank > 1 {
            let scale_lr = 1.0 / (2.0 * (rank - 1) as f64).sqrt();
            for dev in &self.deviations {
                let z2 = sample_normal(rng);
                for i in 0..n {
                    lr_term[i] += dev[i] * z2 * scale_lr;
                }
            }
        }
        self.mean_params
            .iter()
            .zip(diag_term.iter().zip(lr_term.iter()))
            .map(|(&m, (&dt, &lt))| m + self.config.scale * (dt + lt))
            .collect()
    }

    /// Predict via SWAG posterior samples.
    pub fn predict_ensemble(
        &self,
        model: &mut BdlMlp,
        x: &[f64],
        n_samples: usize,
        rng: &mut StdRng,
    ) -> (Vec<f64>, Vec<f64>) {
        let ns = n_samples.max(1);
        let preds: Vec<Vec<f64>> = (0..ns)
            .map(|_| {
                let params = self.sample(rng);
                model.set_params(&params);
                model.forward(x)
            })
            .collect();
        ensemble_stats(&preds)
    }
}
