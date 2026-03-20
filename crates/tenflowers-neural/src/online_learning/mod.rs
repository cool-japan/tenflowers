//! Streaming & Online Learning — online gradient methods, bandit algorithms,
//! concept drift detection, streaming data structures, and online evaluation.

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Online Gradient Methods
// ─────────────────────────────────────────────────────────────────────────────

/// Learning rate schedule for online SGD.
#[derive(Debug, Clone)]
pub enum LrSchedule {
    Constant(f64),
    InverseTime { lr: f64, decay: f64 },
    SqrtDecay { lr: f64 },
    Exponential { lr: f64, decay: f64 },
}

impl LrSchedule {
    /// Effective learning rate at 1-based `step`.
    pub fn rate(&self, step: usize) -> f64 {
        match self {
            LrSchedule::Constant(lr) => *lr,
            LrSchedule::InverseTime { lr, decay } => lr / (1.0 + decay * step as f64),
            LrSchedule::SqrtDecay { lr } => lr / ((step as f64 + 1.0).sqrt()),
            LrSchedule::Exponential { lr, decay } => lr * decay.powi(step as i32),
        }
    }
}

/// Online SGD with a configurable learning-rate schedule and optional momentum.
#[derive(Debug, Clone)]
pub struct OnlineSgd {
    pub schedule: LrSchedule,
    pub momentum: f64,
    velocities: Vec<f64>,
}

impl OnlineSgd {
    pub fn new(lr: f64) -> Self {
        Self {
            schedule: LrSchedule::Constant(lr),
            momentum: 0.0,
            velocities: Vec::new(),
        }
    }

    pub fn with_schedule(schedule: LrSchedule, momentum: f64) -> Self {
        Self {
            schedule,
            momentum,
            velocities: Vec::new(),
        }
    }

    /// One SGD step (1-based `step`). Returns updated parameters.
    pub fn update(&mut self, params: &[f64], grads: &[f64], step: usize) -> Result<Vec<f64>> {
        if params.len() != grads.len() {
            return Err(TensorError::invalid_argument_op(
                "OnlineSgd::update",
                "params and grads must have the same length",
            ));
        }
        let n = params.len();
        if self.velocities.len() != n {
            self.velocities = vec![0.0; n];
        }
        let lr = self.schedule.rate(step);
        let m = self.momentum;
        let mut updated = vec![0.0_f64; n];
        for i in 0..n {
            self.velocities[i] = m * self.velocities[i] + grads[i];
            updated[i] = params[i] - lr * self.velocities[i];
        }
        Ok(updated)
    }
}

/// Online AdaGrad — per-coordinate adaptive learning rate.
#[derive(Debug, Clone)]
pub struct OnlineAdaGrad {
    pub lr: f64,
    pub eps: f64,
}

impl OnlineAdaGrad {
    pub fn new(lr: f64) -> Self {
        Self { lr, eps: 1e-8 }
    }

    /// One step. Returns `(updated_params, updated_acc_sq_grad)`.
    pub fn update(
        &self,
        params: &[f64],
        grads: &[f64],
        acc_sq_grad: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if params.len() != grads.len() || params.len() != acc_sq_grad.len() {
            return Err(TensorError::invalid_argument_op(
                "OnlineAdaGrad::update",
                "params, grads, and acc_sq_grad must all have the same length",
            ));
        }
        let n = params.len();
        let mut new_params = vec![0.0_f64; n];
        let mut new_acc = vec![0.0_f64; n];
        for i in 0..n {
            new_acc[i] = acc_sq_grad[i] + grads[i] * grads[i];
            new_params[i] = params[i] - (self.lr / (new_acc[i] + self.eps).sqrt()) * grads[i];
        }
        Ok((new_params, new_acc))
    }
}

/// Online Adam optimiser on plain `Vec<f64>` buffers.
#[derive(Debug, Clone)]
pub struct OnlineAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl OnlineAdam {
    /// Defaults: lr, β₁=0.9, β₂=0.999, ε=1e-8.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// One step (1-based `step`). Returns `(updated_params, updated_m, updated_v)`.
    pub fn update(
        &self,
        params: &[f64],
        grads: &[f64],
        m: &[f64],
        v: &[f64],
        step: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = params.len();
        if grads.len() != n || m.len() != n || v.len() != n {
            return Err(TensorError::invalid_argument_op(
                "OnlineAdam::update",
                "all slice arguments must have the same length",
            ));
        }
        let t = step as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        let mut new_params = vec![0.0_f64; n];
        let mut new_m = vec![0.0_f64; n];
        let mut new_v = vec![0.0_f64; n];
        for i in 0..n {
            new_m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grads[i];
            new_v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let m_hat = new_m[i] / bc1;
            let v_hat = new_v[i] / bc2;
            new_params[i] = params[i] - self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        Ok((new_params, new_m, new_v))
    }
}

/// FTRL-Proximal — per-coordinate adaptive L1/L2 update.
#[derive(Debug, Clone, Default)]
pub struct FollowTheRegularizedLeader;

impl FollowTheRegularizedLeader {
    pub fn new() -> Self {
        Self
    }

    /// One FTRL-Proximal step. Returns `(new_params, new_z, new_n)`.
    pub fn update(
        &self,
        params: &[f64],
        grads: &[f64],
        z: &[f64],
        n: &[f64],
        alpha: f64,
        beta: f64,
        lambda1: f64,
        lambda2: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let len = params.len();
        if grads.len() != len || z.len() != len || n.len() != len {
            return Err(TensorError::invalid_argument_op(
                "FollowTheRegularizedLeader::update",
                "all slice arguments must have the same length",
            ));
        }
        let mut new_params = vec![0.0_f64; len];
        let mut new_z = vec![0.0_f64; len];
        let mut new_n = vec![0.0_f64; len];
        for i in 0..len {
            new_n[i] = n[i] + grads[i] * grads[i];
            let sigma = (new_n[i].sqrt() - n[i].sqrt()) / alpha;
            new_z[i] = z[i] + grads[i] - sigma * params[i];
            if new_z[i].abs() <= lambda1 {
                new_params[i] = 0.0;
            } else {
                let sign = if new_z[i] > 0.0 { 1.0 } else { -1.0 };
                let denom = (beta + new_n[i].sqrt()) / alpha + lambda2;
                new_params[i] = -(new_z[i] - sign * lambda1) / denom;
            }
        }
        Ok((new_params, new_z, new_n))
    }
}

/// Online L-BFGS — two-loop recursion with a bounded (s, y) history buffer.
#[derive(Debug, Clone)]
pub struct OnlineLbfgs {
    pub history_size: usize,
    pub lr: f64,
    s_history: Vec<Vec<f64>>,
    y_history: Vec<Vec<f64>>,
    prev_params: Option<Vec<f64>>,
    prev_grads: Option<Vec<f64>>,
}

impl OnlineLbfgs {
    pub fn new(history_size: usize, lr: f64) -> Self {
        Self {
            history_size,
            lr,
            s_history: Vec::new(),
            y_history: Vec::new(),
            prev_params: None,
            prev_grads: None,
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// One L-BFGS step. Returns updated parameters.
    pub fn update(&mut self, params: &[f64], grads: &[f64]) -> Result<Vec<f64>> {
        let n = params.len();
        if grads.len() != n {
            return Err(TensorError::invalid_argument_op(
                "OnlineLbfgs::update",
                "params and grads must have the same length",
            ));
        }

        if let (Some(pp), Some(pg)) = (self.prev_params.take(), self.prev_grads.take()) {
            if pp.len() == n {
                let s: Vec<f64> = params.iter().zip(pp.iter()).map(|(a, b)| a - b).collect();
                let y: Vec<f64> = grads.iter().zip(pg.iter()).map(|(a, b)| a - b).collect();
                if Self::dot(&s, &y) > 1e-10 {
                    if self.s_history.len() >= self.history_size {
                        self.s_history.remove(0);
                        self.y_history.remove(0);
                    }
                    self.s_history.push(s);
                    self.y_history.push(y);
                }
            }
        }

        let m = self.s_history.len();
        let mut q = grads.to_vec();
        let mut alphas = vec![0.0_f64; m];
        for i in (0..m).rev() {
            let rho = 1.0 / Self::dot(&self.y_history[i], &self.s_history[i]);
            let alpha = rho * Self::dot(&self.s_history[i], &q);
            alphas[i] = alpha;
            for j in 0..n {
                q[j] -= alpha * self.y_history[i][j];
            }
        }
        let gamma = if m > 0 {
            let sy = Self::dot(&self.s_history[m - 1], &self.y_history[m - 1]);
            let yy = Self::dot(&self.y_history[m - 1], &self.y_history[m - 1]);
            if yy > 1e-10 {
                sy / yy
            } else {
                1.0
            }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|x| gamma * x).collect();
        for i in 0..m {
            let rho = 1.0 / Self::dot(&self.y_history[i], &self.s_history[i]);
            let beta = rho * Self::dot(&self.y_history[i], &r);
            for j in 0..n {
                r[j] += (alphas[i] - beta) * self.s_history[i][j];
            }
        }
        let updated: Vec<f64> = params
            .iter()
            .zip(r.iter())
            .map(|(p, d)| p - self.lr * d)
            .collect();
        self.prev_params = Some(params.to_vec());
        self.prev_grads = Some(grads.to_vec());
        Ok(updated)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Bandit Algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// ε-greedy multi-armed bandit.
#[derive(Debug, Clone, Default)]
pub struct EpsilonGreedy;

impl EpsilonGreedy {
    pub fn new() -> Self {
        Self
    }

    /// With prob ε explore uniformly, otherwise exploit.
    pub fn select_arm(&self, q_values: &[f64], epsilon: f64, rng: &mut StdRng) -> usize {
        if q_values.is_empty() {
            return 0;
        }
        let r: f64 = rng.random();
        if r < epsilon {
            rng.random_range(0.0..q_values.len() as f64) as usize
        } else {
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Sample-average update. Returns `(updated_q_values, updated_counts)`.
    pub fn update(
        &self,
        arm: usize,
        reward: f64,
        q_values: &[f64],
        counts: &[usize],
    ) -> Result<(Vec<f64>, Vec<usize>)> {
        if arm >= q_values.len() || q_values.len() != counts.len() {
            return Err(TensorError::invalid_argument_op(
                "EpsilonGreedy::update",
                "invalid arm index or mismatched q_values/counts length",
            ));
        }
        let mut new_q = q_values.to_vec();
        let mut new_c = counts.to_vec();
        new_c[arm] += 1;
        new_q[arm] += (reward - new_q[arm]) / new_c[arm] as f64;
        Ok((new_q, new_c))
    }
}

/// UCB1 upper-confidence-bound bandit.
#[derive(Debug, Clone, Default)]
pub struct Ucb1;

impl Ucb1 {
    pub fn new() -> Self {
        Self
    }

    /// UCB1 score: `q + sqrt(2 ln(t) / n)`. Returns ∞ when `n == 0`.
    pub fn compute_ucb(&self, q: f64, n: usize, t: usize) -> f64 {
        if n == 0 {
            f64::INFINITY
        } else {
            q + (2.0 * (t as f64).ln() / n as f64).sqrt()
        }
    }

    /// Arm with highest UCB1 score.
    pub fn select_arm(&self, q_values: &[f64], counts: &[usize], total_t: usize) -> usize {
        if q_values.is_empty() {
            return 0;
        }
        q_values
            .iter()
            .zip(counts.iter())
            .enumerate()
            .map(|(i, (&q, &n))| (i, self.compute_ucb(q, n, total_t)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Thompson Sampling with Beta(α, β) priors (Bernoulli rewards).
#[derive(Debug, Clone, Default)]
pub struct ThompsonSampling;

impl ThompsonSampling {
    pub fn new() -> Self {
        Self
    }

    fn sample_beta(alpha: f64, beta: f64, rng: &mut StdRng) -> f64 {
        let x = Self::sample_gamma(alpha, rng);
        let y = Self::sample_gamma(beta, rng);
        let s = x + y;
        if s < 1e-15 {
            0.5
        } else {
            x / s
        }
    }

    /// Marsaglia-Tsang Gamma(shape, 1).
    fn sample_gamma(shape: f64, rng: &mut StdRng) -> f64 {
        if shape < 1.0 {
            let u: f64 = rng.random();
            return Self::sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u: f64 = rng.random();
            if u < 1.0 - 0.0331 * (z * z) * (z * z) {
                return d * v3;
            }
            if u.ln() < 0.5 * z * z + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }

    /// Sample from each arm's Beta posterior; return arm with highest draw.
    pub fn sample(&self, alphas: &[f64], betas: &[f64], rng: &mut StdRng) -> usize {
        if alphas.is_empty() {
            return 0;
        }
        alphas
            .iter()
            .zip(betas.iter())
            .enumerate()
            .map(|(i, (&a, &b))| (i, Self::sample_beta(a.max(0.001), b.max(0.001), rng)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update Beta posteriors. Returns `(new_alphas, new_betas)`.
    pub fn update(
        &self,
        arm: usize,
        reward: f64,
        alphas: &[f64],
        betas: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if arm >= alphas.len() || alphas.len() != betas.len() {
            return Err(TensorError::invalid_argument_op(
                "ThompsonSampling::update",
                "invalid arm or length",
            ));
        }
        let mut new_a = alphas.to_vec();
        let mut new_b = betas.to_vec();
        if reward >= 0.5 {
            new_a[arm] += 1.0;
        } else {
            new_b[arm] += 1.0;
        }
        Ok((new_a, new_b))
    }
}

/// LinUCB — linear contextual bandit (disjoint model, A_inv and b stored flattened).
#[derive(Debug, Clone)]
pub struct LinUcb {
    pub n_arms: usize,
    pub dim: usize,
    pub alpha: f64,
}

impl LinUcb {
    pub fn new(n_arms: usize, dim: usize, alpha: f64) -> Self {
        Self { n_arms, dim, alpha }
    }

    fn mat_vec(mat: &[f64], vec: &[f64], d: usize) -> Vec<f64> {
        (0..d)
            .map(|i| (0..d).map(|j| mat[i * d + j] * vec[j]).sum())
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Sherman-Morrison update of `A_inv` and `b` for `arm`.
    pub fn update(
        &self,
        context: &[f64],
        arm: usize,
        reward: f64,
        a_inv: &mut [f64],
        b: &mut [f64],
    ) -> Result<()> {
        let d = self.dim;
        let mo = arm * d * d;
        let vo = arm * d;
        if a_inv.len() < (arm + 1) * d * d || b.len() < (arm + 1) * d {
            return Err(TensorError::invalid_argument_op(
                "LinUcb::update",
                "buffer too small",
            ));
        }
        let ax = Self::mat_vec(&a_inv[mo..mo + d * d], context, d);
        let denom = 1.0 + Self::dot(context, &ax);
        for i in 0..d {
            for j in 0..d {
                a_inv[mo + i * d + j] -= ax[i] * ax[j] / denom;
            }
        }
        for i in 0..d {
            b[vo + i] += reward * context[i];
        }
        Ok(())
    }

    /// Arm with highest LinUCB score.
    pub fn select_arm(&self, contexts: &[Vec<f64>], a_inv: &[f64], b: &[f64], alpha: f64) -> usize {
        let d = self.dim;
        contexts
            .iter()
            .enumerate()
            .map(|(arm, ctx)| {
                let mo = arm * d * d;
                let vo = arm * d;
                let theta = Self::mat_vec(&a_inv[mo..mo + d * d], &b[vo..vo + d], d);
                let mu = Self::dot(&theta, ctx);
                let ax = Self::mat_vec(&a_inv[mo..mo + d * d], ctx, d);
                let sigma = Self::dot(ctx, &ax).sqrt();
                (arm, mu + alpha * sigma)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Neural Bandit — per-arm two-layer MLP with diagonal Fisher uncertainty.
#[derive(Debug, Clone)]
pub struct NeuralBandit {
    pub n_arms: usize,
    pub dim: usize,
    pub hidden: usize,
    w1: Vec<Vec<f64>>,
    w2: Vec<Vec<f64>>,
    fisher1: Vec<Vec<f64>>,
    fisher2: Vec<Vec<f64>>,
}

impl NeuralBandit {
    /// Random initialisation (He scale).
    pub fn new(n_arms: usize, dim: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0 / dim as f64).sqrt();
        let mut w1 = Vec::with_capacity(n_arms);
        let mut w2 = Vec::with_capacity(n_arms);
        let mut fisher1 = Vec::with_capacity(n_arms);
        let mut fisher2 = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            let weights: Vec<f64> = (0..hidden * dim)
                .map(|_| (rng.random::<f64>() - 0.5) * scale)
                .collect();
            w1.push(weights);
            w2.push(
                (0..hidden)
                    .map(|_| (rng.random::<f64>() - 0.5) * scale)
                    .collect(),
            );
            fisher1.push(vec![1.0; hidden * dim]);
            fisher2.push(vec![1.0; hidden]);
        }
        Self {
            n_arms,
            dim,
            hidden,
            w1,
            w2,
            fisher1,
            fisher2,
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn forward(&self, arm: usize, ctx: &[f64]) -> f64 {
        let h = self.hidden;
        let d = self.dim;
        let mut hidden_out = vec![0.0_f64; h];
        for i in 0..h {
            let sum: f64 = (0..d).map(|j| self.w1[arm][i * d + j] * ctx[j]).sum();
            hidden_out[i] = Self::relu(sum);
        }
        (0..h).map(|i| self.w2[arm][i] * hidden_out[i]).sum()
    }

    fn uncertainty(&self, arm: usize, ctx: &[f64]) -> f64 {
        let (h, d) = (self.hidden, self.dim);
        let u1: f64 = (0..h * d)
            .map(|k| self.w1[arm][k] * self.w1[arm][k] / (self.fisher1[arm][k] + 1e-6))
            .sum();
        let u2: f64 = (0..h)
            .map(|i| self.w2[arm][i] * self.w2[arm][i] / (self.fisher2[arm][i] + 1e-6))
            .sum();
        let ctx_norm: f64 = ctx.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
        (u1 + u2).sqrt() * ctx_norm
    }

    /// ε-greedy arm selection with Fisher-based exploration bonus.
    pub fn select(&self, context: &[f64], epsilon: f64, rng: &mut StdRng) -> usize {
        let r: f64 = rng.random();
        if r < epsilon {
            return rng.random_range(0.0..self.n_arms as f64) as usize;
        }
        (0..self.n_arms)
            .map(|arm| {
                let pred = self.forward(arm, context);
                let bonus = self.uncertainty(arm, context);
                (arm, pred + bonus)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// SGD update for one arm.
    pub fn update(&mut self, arm: usize, context: &[f64], reward: f64, lr: f64) -> Result<()> {
        if arm >= self.n_arms || context.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "NeuralBandit::update",
                "invalid arm or context dimension",
            ));
        }
        let (h, d) = (self.hidden, self.dim);
        let mut hidden_pre = vec![0.0_f64; h];
        let mut hidden_out = vec![0.0_f64; h];
        for i in 0..h {
            let sum: f64 = (0..d).map(|j| self.w1[arm][i * d + j] * context[j]).sum();
            hidden_pre[i] = sum;
            hidden_out[i] = Self::relu(sum);
        }
        let err = (0..h).map(|i| self.w2[arm][i] * hidden_out[i]).sum::<f64>() - reward;
        let dw2: Vec<f64> = (0..h).map(|i| err * hidden_out[i]).collect();
        let dh: Vec<f64> = (0..h)
            .map(|i| err * self.w2[arm][i] * if hidden_pre[i] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        for i in 0..h {
            self.fisher2[arm][i] += dw2[i] * dw2[i];
            self.w2[arm][i] -= lr * dw2[i];
            for j in 0..d {
                let g = dh[i] * context[j];
                self.fisher1[arm][i * d + j] += g * g;
                self.w1[arm][i * d + j] -= lr * g;
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Concept Drift Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a drift detection test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftDetected {
    None,
    Positive,
    Negative,
}

/// Page-Hinkley test (two-sided) — detects abrupt mean shifts.
#[derive(Debug, Clone)]
pub struct PageHinkleyTest {
    pub threshold: f64,
    pub delta: f64,
    ph_plus: f64,
    ph_minus: f64,
    mean: f64,
    n: usize,
}

impl PageHinkleyTest {
    pub fn new(threshold: f64, delta: f64) -> Self {
        Self {
            threshold,
            delta,
            ph_plus: 0.0,
            ph_minus: 0.0,
            mean: 0.0,
            n: 0,
        }
    }

    /// Update with new observation; return drift status.
    pub fn update(&mut self, x: f64) -> DriftDetected {
        self.n += 1;
        self.mean += (x - self.mean) / self.n as f64;
        self.ph_plus += x - self.mean - self.delta;
        self.ph_minus += self.mean - x - self.delta;
        self.ph_plus = self.ph_plus.max(0.0);
        self.ph_minus = self.ph_minus.max(0.0);
        if self.ph_plus > self.threshold {
            DriftDetected::Positive
        } else if self.ph_minus > self.threshold {
            DriftDetected::Negative
        } else {
            DriftDetected::None
        }
    }

    pub fn reset(&mut self) {
        self.ph_plus = 0.0;
        self.ph_minus = 0.0;
        self.mean = 0.0;
        self.n = 0;
    }
}

/// ADWIN — Adaptive Windowing: detects mean changes via sub-window comparison.
#[derive(Debug, Clone)]
pub struct Adwin {
    pub delta: f64,
    pub max_window: usize,
    window: Vec<f64>,
}

impl Adwin {
    pub fn new(delta: f64, max_window: usize) -> Self {
        Self {
            delta,
            max_window,
            window: Vec::new(),
        }
    }

    fn mean_of(slice: &[f64]) -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    /// Returns `(drift_detected, current_mean)`.
    pub fn update(&mut self, x: f64) -> (bool, f64) {
        self.window.push(x);
        if self.window.len() > self.max_window {
            self.window.remove(0);
        }
        let n = self.window.len();
        let mut drift = false;
        if n >= 4 {
            let mu_all = Self::mean_of(&self.window);
            // Test every split point.
            for split in 1..n {
                let n0 = split as f64;
                let n1 = (n - split) as f64;
                let mu0 = Self::mean_of(&self.window[..split]);
                let mu1 = Self::mean_of(&self.window[split..]);
                let eps_cut =
                    ((1.0 / (2.0 * n0) + 1.0 / (2.0 * n1)) * ((n as f64 / self.delta).ln())).sqrt();
                if (mu0 - mu1).abs() >= eps_cut {
                    // Trim oldest half of window.
                    self.window.drain(..split);
                    drift = true;
                    break;
                }
            }
            let _ = mu_all;
        }
        let current_mean = Self::mean_of(&self.window);
        (drift, current_mean)
    }
}

/// DDM detector status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    Normal,
    Warning,
    Drift,
}

/// DDM — Drift Detection Method (Gama et al., 2004).
#[derive(Debug, Clone)]
pub struct DdmDetector {
    n: usize,
    error_count: usize,
    p_min: f64,
    s_min: f64,
}

impl DdmDetector {
    pub fn new() -> Self {
        Self {
            n: 0,
            error_count: 0,
            p_min: f64::INFINITY,
            s_min: f64::INFINITY,
        }
    }

    /// Update (`error=true` if the model made a mistake).
    pub fn update(&mut self, error: bool) -> DriftStatus {
        self.n += 1;
        if error {
            self.error_count += 1;
        }
        if self.n < 30 {
            return DriftStatus::Normal;
        }
        let p = self.error_count as f64 / self.n as f64;
        let s = (p * (1.0 - p) / self.n as f64).sqrt();
        let ps = p + s;
        if ps < self.p_min + self.s_min {
            self.p_min = p;
            self.s_min = s;
        }
        if ps > self.p_min + 3.0 * self.s_min {
            self.n = 0;
            self.error_count = 0;
            self.p_min = f64::INFINITY;
            self.s_min = f64::INFINITY;
            DriftStatus::Drift
        } else if ps > self.p_min + 2.0 * self.s_min {
            DriftStatus::Warning
        } else {
            DriftStatus::Normal
        }
    }
}

impl Default for DdmDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Kolmogorov-Smirnov two-sample test.
#[derive(Debug, Clone, Default)]
pub struct KsTest;

impl KsTest {
    pub fn new() -> Self {
        Self
    }

    /// KS statistic D = max|F₁(x) − F₂(x)|.
    pub fn test(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        if sample1.is_empty() || sample2.is_empty() {
            return 0.0;
        }
        let n1 = sample1.len();
        let n2 = sample2.len();
        let mut s1 = sample1.to_vec();
        let mut s2 = sample2.to_vec();
        s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0usize;
        let mut j = 0usize;
        let mut max_diff = 0.0_f64;
        while i < n1 || j < n2 {
            let val = if i < n1 && (j >= n2 || s1[i] <= s2[j]) {
                s1[i]
            } else {
                s2[j]
            };
            while i < n1 && s1[i] <= val {
                i += 1;
            }
            while j < n2 && s2[j] <= val {
                j += 1;
            }
            let f1 = i as f64 / n1 as f64;
            let f2 = j as f64 / n2 as f64;
            max_diff = max_diff.max((f1 - f2).abs());
        }
        max_diff
    }
}

/// Sliding-window running statistics.
#[derive(Debug, Clone)]
pub struct WindowedStatistics {
    pub capacity: usize,
    buffer: std::collections::VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl WindowedStatistics {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: std::collections::VecDeque::with_capacity(capacity),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a value, evicting the oldest when full.
    pub fn push(&mut self, x: f64) {
        if self.buffer.len() == self.capacity {
            let old = self.buffer.pop_front().unwrap_or(0.0);
            self.sum -= old;
            self.sum_sq -= old * old;
        }
        self.buffer.push_back(x);
        self.sum += x;
        self.sum_sq += x * x;
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        self.sum / self.buffer.len() as f64
    }

    pub fn variance(&self) -> f64 {
        let n = self.buffer.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        (self.sum_sq / n as f64) - m * m
    }

    /// Approximate quantile (`q` in [0, 1]) by sorting the window.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.buffer.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}
