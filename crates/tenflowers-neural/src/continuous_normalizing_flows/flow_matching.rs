//! Flow Matching, OT-CFM, and Rectified Flow
//!
//! Implements:
//! - **CFM** (Lipman et al. 2022): Conditional Flow Matching
//! - **OT-CFM** (Tong et al. 2023): Optimal Transport CFM
//! - **Rectified Flow** (Liu et al. 2022): straight-line interpolation paths

use super::mlp::CnfMlp;
use super::utils::sample_standard_normal;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ────────────────────────────────────────────────────────────────────────────
// Flow Matching Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Flow Matching / CFM models.
#[derive(Clone)]
pub struct FlowMatchingConfig {
    /// Dimensionality of the data/latent space.
    pub z_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of hidden layers in the velocity network.
    pub n_layers: usize,
    /// Small constant `σ_min` for numerical stability (default `1e-4`).
    pub sigma_min: f64,
    /// Number of Euler steps during inference.
    pub n_steps: usize,
    /// Learning rate.
    pub lr: f64,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        FlowMatchingConfig {
            z_dim: 2,
            hidden_dim: 64,
            n_layers: 2,
            sigma_min: 1e-4,
            n_steps: 100,
            lr: 1e-3,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conditional Flow Matching (Lipman et al. 2022)
// ────────────────────────────────────────────────────────────────────────────

/// Conditional Flow Matching model (Lipman et al. 2022).
///
/// Learns a velocity field `v_θ(x, t)` such that Euler integration from `t=0` to `t=1`
/// transforms noise `p_0 = N(0,I)` into data `p_1`.
///
/// Loss: `E[||v_θ(x_t, t) - u_t(x_t|x_0,x_1)||^2]`
/// where `x_t = (1-(1-σ)t)·x_0 + t·x_1` and `u_t = x_1 - (1-σ)·x_0`.
pub struct FlowMatchingModel {
    /// Velocity network: input `[z; t]` (dim `z_dim + 1`) → output `z_dim`.
    pub velocity_net: CnfMlp,
    /// Configuration.
    pub config: FlowMatchingConfig,
}

impl FlowMatchingModel {
    /// Create a new FlowMatchingModel.
    pub fn new(config: FlowMatchingConfig) -> Self {
        let in_dim = config.z_dim + 1; // [z; t]
        let mut sizes = vec![in_dim];
        for _ in 0..config.n_layers {
            sizes.push(config.hidden_dim);
        }
        sizes.push(config.z_dim);
        FlowMatchingModel {
            velocity_net: CnfMlp::new(&sizes),
            config,
        }
    }

    /// Evaluate the velocity field `v_θ(z, t)`.
    pub fn velocity(&self, z: &[f64], t: f64) -> Vec<f64> {
        let mut inp = z.to_vec();
        inp.push(t);
        self.velocity_net.forward(&inp)
    }

    /// Conditional Flow Matching loss.
    ///
    /// For each pair `(x0, x1)`:
    /// - Sample `t ~ U[0,1]`
    /// - Compute `x_t = (1 - (1 - σ_min) * t) * x0 + t * x1`
    /// - Conditional vector field: `u_t = x1 - (1 - σ_min) * x0`
    /// - Loss term: `||v_θ(x_t, t) - u_t||^2`
    pub fn cfm_loss(&self, x0_batch: &[Vec<f64>], x1_batch: &[Vec<f64>], rng: &mut StdRng) -> f64 {
        let n = x0_batch.len().min(x1_batch.len());
        if n == 0 {
            return 0.0;
        }
        let sigma_min = self.config.sigma_min;
        let mut total_loss = 0.0_f64;

        for i in 0..n {
            let t: f64 = rng.random();
            let x0 = &x0_batch[i];
            let x1 = &x1_batch[i];
            let d = x0.len().min(x1.len()).min(self.config.z_dim);

            // x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
            let xt: Vec<f64> = (0..d)
                .map(|j| (1.0 - (1.0 - sigma_min) * t) * x0[j] + t * x1[j])
                .collect();

            // Conditional vector field u_t = x1 - (1 - sigma_min) * x0
            let ut: Vec<f64> = (0..d).map(|j| x1[j] - (1.0 - sigma_min) * x0[j]).collect();

            // Velocity prediction
            let vt = self.velocity(&xt, t);

            // MSE loss
            let loss: f64 = vt
                .iter()
                .zip(ut.iter())
                .map(|(v, u)| (v - u) * (v - u))
                .sum::<f64>();
            total_loss += loss / d.max(1) as f64;
        }
        total_loss / n as f64
    }

    /// One training step: compute CFM loss and update via finite-differences.
    ///
    /// Returns the CFM loss.
    pub fn train_step(&mut self, x1_batch: &[Vec<f64>], lr: f64, rng: &mut StdRng) -> f64 {
        if x1_batch.is_empty() {
            return 0.0;
        }
        let d = self.config.z_dim;
        let n = x1_batch.len();

        // Sample noise batch x0 ~ N(0, I)
        let x0_batch: Vec<Vec<f64>> = (0..n).map(|_| sample_standard_normal(d, rng)).collect();

        let base_loss = self.cfm_loss(&x0_batch, x1_batch, rng);

        // FD gradient update
        let fd_eps = 1e-4;
        let mut update_rng = StdRng::seed_from_u64(0x246810ac_u64);
        let n_layers = self.velocity_net.n_layers();
        let mut grad_w: Vec<Vec<Vec<f64>>> = self
            .velocity_net
            .weights
            .iter()
            .map(|lw| lw.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut grad_b: Vec<Vec<f64>> = self
            .velocity_net
            .biases
            .iter()
            .map(|lb| vec![0.0; lb.len()])
            .collect();

        for l in 0..n_layers {
            for j in 0..self.velocity_net.weights[l].len() {
                for i in 0..self.velocity_net.weights[l][j].len() {
                    if update_rng.random::<f64>() < 0.05 {
                        self.velocity_net.weights[l][j][i] += fd_eps;
                        let perturbed = self.cfm_loss(&x0_batch, x1_batch, rng);
                        self.velocity_net.weights[l][j][i] -= fd_eps;
                        grad_w[l][j][i] = (perturbed - base_loss) / fd_eps;
                    }
                }
            }
            for j in 0..self.velocity_net.biases[l].len() {
                if update_rng.random::<f64>() < 0.05 {
                    self.velocity_net.biases[l][j] += fd_eps;
                    let perturbed = self.cfm_loss(&x0_batch, x1_batch, rng);
                    self.velocity_net.biases[l][j] -= fd_eps;
                    grad_b[l][j] = (perturbed - base_loss) / fd_eps;
                }
            }
        }
        self.velocity_net.update(&grad_w, &grad_b, lr);
        base_loss
    }

    /// Sample a single data point by Euler-integrating the velocity field from `t=0` to `t=1`.
    pub fn sample(&self, n_steps: usize, rng: &mut StdRng) -> Vec<f64> {
        let d = self.config.z_dim;
        let mut x = sample_standard_normal(d, rng);
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;

        for step in 0..n {
            let t = step as f64 * dt;
            let v = self.velocity(&x, t);
            for (xi, vi) in x.iter_mut().zip(v.iter()) {
                *xi += dt * vi;
            }
        }
        x
    }

    /// Sample a batch of data points.
    pub fn sample_batch(
        &self,
        n_samples: usize,
        n_steps: usize,
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
        (0..n_samples).map(|_| self.sample(n_steps, rng)).collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OT-CFM — Optimal Transport Conditional Flow Matching (Tong et al. 2023)
// ────────────────────────────────────────────────────────────────────────────

/// Optimal-transport conditional flow matching model.
///
/// Uses mini-batch greedy OT to straighten probability paths, reducing the
/// curvature of learned trajectories compared to standard CFM.
pub struct OtCfmModel {
    /// Velocity network: input `[z; t]` → output `z_dim`.
    pub velocity_net: CnfMlp,
    /// Configuration (shared with FlowMatchingModel).
    pub config: FlowMatchingConfig,
}

impl OtCfmModel {
    /// Create a new OT-CFM model.
    pub fn new(config: FlowMatchingConfig) -> Self {
        let in_dim = config.z_dim + 1;
        let mut sizes = vec![in_dim];
        for _ in 0..config.n_layers {
            sizes.push(config.hidden_dim);
        }
        sizes.push(config.z_dim);
        OtCfmModel {
            velocity_net: CnfMlp::new(&sizes),
            config,
        }
    }

    /// Greedy OT matching: for each data point `x1[j]`, find the nearest noise point `x0[i]`.
    ///
    /// Returns a permutation `perm` of `x0` indices such that `x0[perm[j]]` is paired
    /// with `x1[j]`.  Uses a greedy nearest-neighbour algorithm in O(n^2).
    pub fn ot_match(x0_batch: &[Vec<f64>], x1_batch: &[Vec<f64>]) -> Vec<usize> {
        let n0 = x0_batch.len();
        let n1 = x1_batch.len();
        let n = n0.min(n1);
        let mut perm = vec![0usize; n];
        let mut used = vec![false; n0];

        for j in 0..n {
            let x1 = &x1_batch[j];
            let mut best_idx = 0usize;
            let mut best_dist = f64::INFINITY;
            for i in 0..n0 {
                if used[i] {
                    continue;
                }
                let dist: f64 = x0_batch[i]
                    .iter()
                    .zip(x1.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            perm[j] = best_idx;
            used[best_idx] = true;
        }
        perm
    }

    /// Evaluate the velocity field.
    pub fn velocity(&self, z: &[f64], t: f64) -> Vec<f64> {
        let mut inp = z.to_vec();
        inp.push(t);
        self.velocity_net.forward(&inp)
    }

    /// One training step with OT matching.
    ///
    /// Returns the CFM loss after matching.
    pub fn train_step(&mut self, x1_batch: &[Vec<f64>], lr: f64, rng: &mut StdRng) -> f64 {
        if x1_batch.is_empty() {
            return 0.0;
        }
        let d = self.config.z_dim;
        let n = x1_batch.len();

        // Sample noise batch x0 ~ N(0, I)
        let x0_raw: Vec<Vec<f64>> = (0..n).map(|_| sample_standard_normal(d, rng)).collect();

        // OT matching: reorder x0 to be close to x1
        let perm = Self::ot_match(&x0_raw, x1_batch);
        let x0_matched: Vec<Vec<f64>> = perm.iter().map(|&idx| x0_raw[idx].clone()).collect();

        let sigma_min = self.config.sigma_min;

        // Compute CFM loss with OT-matched pairs
        let compute_loss = |vel_net: &CnfMlp, rng_inner: &mut StdRng| -> f64 {
            let mut total = 0.0_f64;
            for i in 0..n.min(x0_matched.len()) {
                let t: f64 = rng_inner.random();
                let x0 = &x0_matched[i];
                let x1 = &x1_batch[i];
                let dim = x0.len().min(x1.len()).min(d);
                let xt: Vec<f64> = (0..dim)
                    .map(|j| (1.0 - (1.0 - sigma_min) * t) * x0[j] + t * x1[j])
                    .collect();
                let ut: Vec<f64> = (0..dim)
                    .map(|j| x1[j] - (1.0 - sigma_min) * x0[j])
                    .collect();
                let mut inp = xt.clone();
                inp.push(t);
                let vt = vel_net.forward(&inp);
                let loss: f64 = vt
                    .iter()
                    .zip(ut.iter())
                    .map(|(v, u)| (v - u) * (v - u))
                    .sum::<f64>();
                total += loss / dim.max(1) as f64;
            }
            total / n as f64
        };

        let mut eval_rng = StdRng::seed_from_u64(0xf0e1d2c3_u64);
        let base_loss = compute_loss(&self.velocity_net, &mut eval_rng);

        // FD gradient update
        let fd_eps = 1e-4;
        let mut update_rng = StdRng::seed_from_u64(0xa1b2c3d4_u64);
        let n_layers = self.velocity_net.n_layers();
        let mut grad_w: Vec<Vec<Vec<f64>>> = self
            .velocity_net
            .weights
            .iter()
            .map(|lw| lw.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut grad_b: Vec<Vec<f64>> = self
            .velocity_net
            .biases
            .iter()
            .map(|lb| vec![0.0; lb.len()])
            .collect();

        for l in 0..n_layers {
            for j in 0..self.velocity_net.weights[l].len() {
                for i in 0..self.velocity_net.weights[l][j].len() {
                    if update_rng.random::<f64>() < 0.04 {
                        self.velocity_net.weights[l][j][i] += fd_eps;
                        let mut r = StdRng::seed_from_u64(0xf0e1d2c3_u64);
                        let perturbed = compute_loss(&self.velocity_net, &mut r);
                        self.velocity_net.weights[l][j][i] -= fd_eps;
                        grad_w[l][j][i] = (perturbed - base_loss) / fd_eps;
                    }
                }
            }
            for j in 0..self.velocity_net.biases[l].len() {
                if update_rng.random::<f64>() < 0.04 {
                    self.velocity_net.biases[l][j] += fd_eps;
                    let mut r = StdRng::seed_from_u64(0xf0e1d2c3_u64);
                    let perturbed = compute_loss(&self.velocity_net, &mut r);
                    self.velocity_net.biases[l][j] -= fd_eps;
                    grad_b[l][j] = (perturbed - base_loss) / fd_eps;
                }
            }
        }
        self.velocity_net.update(&grad_w, &grad_b, lr);
        base_loss
    }

    /// Sample a single point from the model via Euler integration.
    pub fn sample(&self, n_steps: usize, rng: &mut StdRng) -> Vec<f64> {
        let d = self.config.z_dim;
        let mut x = sample_standard_normal(d, rng);
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;

        for step in 0..n {
            let t = step as f64 * dt;
            let v = self.velocity(&x, t);
            for (xi, vi) in x.iter_mut().zip(v.iter()) {
                *xi += dt * vi;
            }
        }
        x
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Rectified Flow (Liu et al. 2022)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Rectified Flow.
#[derive(Clone)]
pub struct RectifiedFlowConfig {
    /// Dimensionality of the data/latent space.
    pub z_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Number of Euler integration steps during inference.
    pub n_steps: usize,
    /// Learning rate.
    pub lr: f64,
}

impl Default for RectifiedFlowConfig {
    fn default() -> Self {
        RectifiedFlowConfig {
            z_dim: 2,
            hidden_dim: 64,
            n_layers: 2,
            n_steps: 100,
            lr: 1e-3,
        }
    }
}

/// Rectified Flow: learns a vector field `v_θ(x_t, t) ≈ x1 - x0` along straight paths.
///
/// At training time: `x_t = x0 + t * (x1 - x0)`, target = `x1 - x0`.
/// At sampling time: Euler-integrate `dX/dt = v_θ(X, t)` from `t=0` to `t=1`.
pub struct RectifiedFlow {
    /// Velocity network: input `[z; t]` → output `z_dim`.
    pub velocity_net: CnfMlp,
    /// Configuration.
    pub config: RectifiedFlowConfig,
}

impl RectifiedFlow {
    /// Create a new Rectified Flow model.
    pub fn new(config: RectifiedFlowConfig) -> Self {
        let in_dim = config.z_dim + 1;
        let mut sizes = vec![in_dim];
        for _ in 0..config.n_layers {
            sizes.push(config.hidden_dim);
        }
        sizes.push(config.z_dim);
        RectifiedFlow {
            velocity_net: CnfMlp::new(&sizes),
            config,
        }
    }

    /// Reflow loss: `E[||v_θ(x_t, t) - (x1 - x0)||^2]`.
    ///
    /// `x_t = x0 + t * (x1 - x0)`, `t ~ U[0,1]`.
    pub fn reflow_loss(
        &self,
        x0_batch: &[Vec<f64>],
        x1_batch: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> f64 {
        let n = x0_batch.len().min(x1_batch.len());
        if n == 0 {
            return 0.0;
        }
        let d = self.config.z_dim;
        let mut total = 0.0_f64;

        for i in 0..n {
            let t: f64 = rng.random();
            let x0 = &x0_batch[i];
            let x1 = &x1_batch[i];
            let dim = x0.len().min(x1.len()).min(d);

            // x_t = x0 + t * (x1 - x0)
            let xt: Vec<f64> = (0..dim).map(|j| x0[j] + t * (x1[j] - x0[j])).collect();

            // Target: straight-line velocity = x1 - x0
            let target: Vec<f64> = (0..dim).map(|j| x1[j] - x0[j]).collect();

            let mut inp = xt;
            inp.push(t);
            let pred = self.velocity_net.forward(&inp);

            let loss: f64 = pred
                .iter()
                .zip(target.iter())
                .map(|(p, tg)| (p - tg) * (p - tg))
                .sum::<f64>();
            total += loss / dim.max(1) as f64;
        }
        total / n as f64
    }

    /// One training step: compute reflow loss and update via finite-differences.
    ///
    /// Returns the reflow loss.
    pub fn train_step(&mut self, x1_batch: &[Vec<f64>], lr: f64, rng: &mut StdRng) -> f64 {
        if x1_batch.is_empty() {
            return 0.0;
        }
        let d = self.config.z_dim;
        let n = x1_batch.len();

        // Sample noise x0 ~ N(0, I)
        let x0_batch: Vec<Vec<f64>> = (0..n).map(|_| sample_standard_normal(d, rng)).collect();

        let mut eval_rng = StdRng::seed_from_u64(0x55aa77bb_u64);
        let base_loss = self.reflow_loss(&x0_batch, x1_batch, &mut eval_rng);

        // FD gradient update
        let fd_eps = 1e-4;
        let mut update_rng = StdRng::seed_from_u64(0xcc11ee22_u64);
        let n_layers = self.velocity_net.n_layers();
        let mut grad_w: Vec<Vec<Vec<f64>>> = self
            .velocity_net
            .weights
            .iter()
            .map(|lw| lw.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        let mut grad_b: Vec<Vec<f64>> = self
            .velocity_net
            .biases
            .iter()
            .map(|lb| vec![0.0; lb.len()])
            .collect();

        for l in 0..n_layers {
            for j in 0..self.velocity_net.weights[l].len() {
                for i in 0..self.velocity_net.weights[l][j].len() {
                    if update_rng.random::<f64>() < 0.05 {
                        self.velocity_net.weights[l][j][i] += fd_eps;
                        let mut r = StdRng::seed_from_u64(0x55aa77bb_u64);
                        let perturbed = self.reflow_loss(&x0_batch, x1_batch, &mut r);
                        self.velocity_net.weights[l][j][i] -= fd_eps;
                        grad_w[l][j][i] = (perturbed - base_loss) / fd_eps;
                    }
                }
            }
            for j in 0..self.velocity_net.biases[l].len() {
                if update_rng.random::<f64>() < 0.05 {
                    self.velocity_net.biases[l][j] += fd_eps;
                    let mut r = StdRng::seed_from_u64(0x55aa77bb_u64);
                    let perturbed = self.reflow_loss(&x0_batch, x1_batch, &mut r);
                    self.velocity_net.biases[l][j] -= fd_eps;
                    grad_b[l][j] = (perturbed - base_loss) / fd_eps;
                }
            }
        }
        self.velocity_net.update(&grad_w, &grad_b, lr);
        base_loss
    }

    /// Sample a data point by Euler-integrating `v_θ` from `t=0` to `t=1`.
    pub fn sample(&self, n_steps: usize, rng: &mut StdRng) -> Vec<f64> {
        let d = self.config.z_dim;
        let mut x = sample_standard_normal(d, rng);
        let n = n_steps.max(1);
        let dt = 1.0 / n as f64;

        for step in 0..n {
            let t = step as f64 * dt;
            let mut inp = x.clone();
            inp.push(t);
            let v = self.velocity_net.forward(&inp);
            for (xi, vi) in x.iter_mut().zip(v.iter()) {
                *xi += dt * vi;
            }
        }
        x
    }
}
