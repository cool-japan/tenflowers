//! FFJORD — Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
//!
//! Extends CNF with Hutchinson's trace estimator for scalable O(d) training cost.

use super::mlp::{CnfDynamics, CnfMlp};
use super::utils::{sample_standard_normal, standard_normal_log_prob};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ────────────────────────────────────────────────────────────────────────────
// FFJORD — Free-Form Jacobian of Reversible Dynamics (Grathwohl et al. 2019)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the FFJORD model.
#[derive(Clone)]
pub struct FfjordConfig {
    /// Dimensionality of the latent / data space.
    pub z_dim: usize,
    /// Hidden layer widths (same for all blocks).
    pub hidden_dims: Vec<usize>,
    /// Number of stacked CNF blocks.
    pub n_blocks: usize,
    /// Number of Euler integration steps per block.
    pub n_steps: usize,
    /// Number of Hutchinson trace samples (typically 1 for unbiased estimation).
    pub n_trace_samples: usize,
    /// Learning rate for `train_step`.
    pub lr: f64,
}

impl Default for FfjordConfig {
    fn default() -> Self {
        FfjordConfig {
            z_dim: 2,
            hidden_dims: vec![64],
            n_blocks: 2,
            n_steps: 10,
            n_trace_samples: 1,
            lr: 1e-3,
        }
    }
}

/// A single FFJORD block: one continuous-time flow segment with Hutchinson trace.
#[derive(Clone)]
pub struct FfjordBlock {
    /// The dynamics network for this block.
    pub dynamics: CnfDynamics,
}

impl FfjordBlock {
    /// Create a new FFJORD block.
    pub fn new(z_dim: usize, hidden_dim: usize) -> Self {
        FfjordBlock {
            dynamics: CnfDynamics::new(z_dim, hidden_dim, 2, true),
        }
    }

    /// Forward integration from `t=0` to `t=1` using Hutchinson trace estimator.
    ///
    /// Returns `(z_T, log_det_jacobian)`.
    pub fn forward(
        &self,
        z: &[f64],
        n_steps: usize,
        n_trace_samples: usize,
        rng: &mut StdRng,
    ) -> (Vec<f64>, f64) {
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;
        let mut cur_z = z.to_vec();
        let mut log_det = 0.0_f64;

        for step in 0..n {
            let t = step as f64 * dt;
            let dz = self.dynamics.forward(&cur_z, t);
            let tr = self
                .dynamics
                .trace_jac_approx(&cur_z, t, n_trace_samples, rng);
            for (zi, dzi) in cur_z.iter_mut().zip(dz.iter()) {
                *zi += dt * dzi;
            }
            log_det += dt * tr;
        }
        (cur_z, log_det)
    }

    /// Inverse integration from `t=1` to `t=0`.
    ///
    /// Returns `(z_0, log_det_jacobian)`.
    pub fn inverse(
        &self,
        x: &[f64],
        n_steps: usize,
        n_trace_samples: usize,
        rng: &mut StdRng,
    ) -> (Vec<f64>, f64) {
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;
        let mut cur_z = x.to_vec();
        let mut log_det = 0.0_f64;

        for step in 0..n {
            let t = 1.0 - step as f64 * dt;
            let dz = self.dynamics.forward(&cur_z, t);
            let tr = self
                .dynamics
                .trace_jac_approx(&cur_z, t, n_trace_samples, rng);
            for (zi, dzi) in cur_z.iter_mut().zip(dz.iter()) {
                *zi -= dt * dzi;
            }
            log_det += dt * tr;
        }
        (cur_z, log_det)
    }
}

/// FFJORD model: a stack of `FfjordBlock`s with a standard-normal base distribution.
pub struct Ffjord {
    /// Stacked CNF blocks.
    pub blocks: Vec<FfjordBlock>,
    /// Configuration used to construct this model.
    pub config: FfjordConfig,
}

impl Ffjord {
    /// Construct a new FFJORD model from config.
    pub fn new(config: FfjordConfig) -> Self {
        let hidden_dim = config.hidden_dims.first().copied().unwrap_or(64);
        let blocks = (0..config.n_blocks)
            .map(|_| FfjordBlock::new(config.z_dim, hidden_dim))
            .collect();
        Ffjord { blocks, config }
    }

    /// Compute log p(x) = log p_0(z_0) + Σ_k log|det J_k|.
    ///
    /// Passes `x` through the inverse of each block (right to left) to obtain `z_0`,
    /// accumulating the log-determinants.
    pub fn log_prob(&self, x: &[f64], rng: &mut StdRng) -> f64 {
        let mut z = x.to_vec();
        let mut total_log_det = 0.0_f64;

        // Apply blocks in reverse order for inverse pass
        for block in self.blocks.iter().rev() {
            let (z_prev, log_det) =
                block.inverse(&z, self.config.n_steps, self.config.n_trace_samples, rng);
            z = z_prev;
            total_log_det += log_det;
        }

        // Base distribution: standard normal
        let log_p0 = standard_normal_log_prob(&z);
        log_p0 + total_log_det
    }

    /// Sample by drawing `z_0 ~ N(0, I)` and applying each block forward.
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        let d = self.config.z_dim;
        let mut z = sample_standard_normal(d, rng);

        for block in &self.blocks {
            let (z_next, _log_det) =
                block.forward(&z, self.config.n_steps, self.config.n_trace_samples, rng);
            z = z_next;
        }
        z
    }

    /// One training step: update all block parameters to minimise NLL on `x_batch`.
    ///
    /// Returns mean NLL loss.
    pub fn train_step(&mut self, x_batch: &[Vec<f64>], lr: f64, rng: &mut StdRng) -> f64 {
        if x_batch.is_empty() {
            return 0.0;
        }
        let batch_size = x_batch.len();

        // Compute baseline loss
        let base_loss: f64 =
            x_batch.iter().map(|x| -self.log_prob(x, rng)).sum::<f64>() / batch_size as f64;

        // FD gradient update for each block
        let fd_eps = 1e-4;
        let mut update_rng = StdRng::seed_from_u64(0x13579bdf_u64);

        for block_idx in 0..self.blocks.len() {
            let n_layers = self.blocks[block_idx].dynamics.mlp.n_layers();
            let mut grad_w: Vec<Vec<Vec<f64>>> = self.blocks[block_idx]
                .dynamics
                .mlp
                .weights
                .iter()
                .map(|lw| lw.iter().map(|row| vec![0.0; row.len()]).collect())
                .collect();
            let mut grad_b: Vec<Vec<f64>> = self.blocks[block_idx]
                .dynamics
                .mlp
                .biases
                .iter()
                .map(|lb| vec![0.0; lb.len()])
                .collect();

            for l in 0..n_layers {
                for j in 0..self.blocks[block_idx].dynamics.mlp.weights[l].len() {
                    for i in 0..self.blocks[block_idx].dynamics.mlp.weights[l][j].len() {
                        if update_rng.random::<f64>() < 0.03 {
                            self.blocks[block_idx].dynamics.mlp.weights[l][j][i] += fd_eps;
                            let perturbed: f64 =
                                x_batch.iter().map(|x| -self.log_prob(x, rng)).sum::<f64>()
                                    / batch_size as f64;
                            self.blocks[block_idx].dynamics.mlp.weights[l][j][i] -= fd_eps;
                            grad_w[l][j][i] = (perturbed - base_loss) / fd_eps;
                        }
                    }
                }
                for j in 0..self.blocks[block_idx].dynamics.mlp.biases[l].len() {
                    if update_rng.random::<f64>() < 0.03 {
                        self.blocks[block_idx].dynamics.mlp.biases[l][j] += fd_eps;
                        let perturbed: f64 =
                            x_batch.iter().map(|x| -self.log_prob(x, rng)).sum::<f64>()
                                / batch_size as f64;
                        self.blocks[block_idx].dynamics.mlp.biases[l][j] -= fd_eps;
                        grad_b[l][j] = (perturbed - base_loss) / fd_eps;
                    }
                }
            }
            self.blocks[block_idx]
                .dynamics
                .mlp
                .update(&grad_w, &grad_b, lr);
        }
        base_loss
    }
}
