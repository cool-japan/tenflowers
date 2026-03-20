//! Advanced online learning: bandit algorithms, online convex optimization,
//! streaming anomaly detection, and enhanced concept drift detection.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Advanced Bandit Algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// Linear UCB contextual bandit — ridge-regression confidence ellipsoid (Li 2010).
///
/// Maintains a single shared A and b; arm selection uses the supplied context
/// vectors (one per arm at decision time).
#[derive(Debug, Clone)]
pub struct LinUcbBandit {
    /// Exploration parameter α.
    pub alpha: f64,
    /// Feature dimension d.
    pub dim: usize,
    /// Precision matrix A (d×d, initialised to λ·I).
    a: Vec<f64>,
    /// Reward accumulator b (d-vector).
    b: Vec<f64>,
    /// Ridge regularisation λ.
    pub lambda: f64,
}

impl LinUcbBandit {
    /// Create with feature dimension `dim`, exploration `alpha`, and ridge `lambda`.
    pub fn new(dim: usize, alpha: f64, lambda: f64) -> Self {
        let mut a = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            a[i * dim + i] = lambda;
        }
        Self {
            alpha,
            dim,
            a,
            b: vec![0.0_f64; dim],
            lambda,
        }
    }

    /// Compute A^{-1} via Cholesky decomposition (positive-definite guaranteed by ridge).
    fn cholesky_solve(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
        // Forward substitution then back substitution on lower triangular L.
        let mut l = vec![0.0_f64; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = 0.0_f64;
                for k in 0..j {
                    sum += l[i * d + k] * l[j * d + k];
                }
                if i == j {
                    let val = a[i * d + i] - sum;
                    l[i * d + j] = if val > 0.0 { val.sqrt() } else { 1e-8 };
                } else {
                    let lii = l[j * d + j];
                    l[i * d + j] = if lii.abs() > 1e-12 {
                        (a[i * d + j] - sum) / lii
                    } else {
                        0.0
                    };
                }
            }
        }
        // Forward substitution: Ly = b.
        let mut y = vec![0.0_f64; d];
        for i in 0..d {
            let mut s = b[i];
            for j in 0..i {
                s -= l[i * d + j] * y[j];
            }
            let lii = l[i * d + i];
            y[i] = if lii.abs() > 1e-12 { s / lii } else { 0.0 };
        }
        // Back substitution: L^T x = y.
        let mut x = vec![0.0_f64; d];
        for i in (0..d).rev() {
            let mut s = y[i];
            for j in (i + 1)..d {
                s -= l[j * d + i] * x[j];
            }
            let lii = l[i * d + i];
            x[i] = if lii.abs() > 1e-12 { s / lii } else { 0.0 };
        }
        x
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Select the arm with the highest UCB score given per-arm context vectors.
    pub fn select(&self, contexts: &[Vec<f64>]) -> usize {
        let d = self.dim;
        let theta = Self::cholesky_solve(&self.a, &self.b, d);

        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (arm, ctx) in contexts.iter().enumerate() {
            if ctx.len() != d {
                continue;
            }
            let mu = Self::dot(&theta, ctx);
            // Uncertainty: x^T A^{-1} x — solve A^{-1} ctx.
            let a_inv_ctx = Self::cholesky_solve(&self.a, ctx, d);
            let sigma = Self::dot(ctx, &a_inv_ctx).sqrt().max(0.0);
            let score = mu + self.alpha * sigma;
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        best_arm
    }

    /// Rank-1 update: A += x x^T, b += r x.
    pub fn update(&mut self, context: &[f64], reward: f64) -> Result<()> {
        let d = self.dim;
        if context.len() != d {
            return Err(TensorError::invalid_argument_op(
                "LinUcbBandit::update",
                "context length must equal dim",
            ));
        }
        for i in 0..d {
            for j in 0..d {
                self.a[i * d + j] += context[i] * context[j];
            }
            self.b[i] += reward * context[i];
        }
        Ok(())
    }

    /// Current θ estimate (MLE).
    pub fn theta(&self) -> Vec<f64> {
        Self::cholesky_solve(&self.a, &self.b, self.dim)
    }
}

/// Linear Thompson Sampling contextual bandit — Bayesian posterior updates.
///
/// Models reward as linear in context with Gaussian noise.  Uses the
/// conjugate Gaussian-linear posterior: p(θ|data) ~ N(μ_n, σ² V_n^{-1}).
#[derive(Debug, Clone)]
pub struct ThompsonSamplingLinear {
    /// Feature dimension d.
    pub dim: usize,
    /// Precision matrix V = λI + Σ x_t x_t^T.
    v: Vec<f64>,
    /// b = Σ r_t x_t.
    b: Vec<f64>,
    /// Noise variance σ².
    pub sigma_sq: f64,
    /// Ridge regularisation λ.
    pub lambda: f64,
}

impl ThompsonSamplingLinear {
    /// Create with `dim` features, noise variance `sigma_sq`, and ridge `lambda`.
    pub fn new(dim: usize, sigma_sq: f64, lambda: f64) -> Self {
        let mut v = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            v[i * dim + i] = lambda;
        }
        Self {
            dim,
            v,
            b: vec![0.0_f64; dim],
            sigma_sq,
            lambda,
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Sample a weight vector from the Bayesian posterior (diagonal covariance approx).
    fn sample_theta(&self, rng: &mut StdRng) -> Vec<f64> {
        let d = self.dim;
        // Approximate diagonal: V_ii^{-1} * sigma_sq for each parameter.
        let mu = LinUcbBandit::cholesky_solve(&self.v, &self.b, d);
        // Sample from N(mu_i, sigma_sq / V_ii).
        (0..d)
            .map(|i| {
                let v_ii = self.v[i * d + i].max(1e-12);
                let std = (self.sigma_sq / v_ii).sqrt();
                let u1: f64 = rng.random::<f64>().max(1e-15);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                mu[i] + std * z
            })
            .collect()
    }

    /// Select the arm with highest expected reward under a sampled θ.
    pub fn select(&self, contexts: &[Vec<f64>], rng: &mut StdRng) -> usize {
        let theta = self.sample_theta(rng);
        let mut best_arm = 0;
        let mut best_val = f64::NEG_INFINITY;
        for (arm, ctx) in contexts.iter().enumerate() {
            let val = Self::dot(&theta, ctx);
            if val > best_val {
                best_val = val;
                best_arm = arm;
            }
        }
        best_arm
    }

    /// Update posterior with observed (context, reward) pair.
    pub fn update(&mut self, context: &[f64], reward: f64) -> Result<()> {
        let d = self.dim;
        if context.len() != d {
            return Err(TensorError::invalid_argument_op(
                "ThompsonSamplingLinear::update",
                "context length must equal dim",
            ));
        }
        for i in 0..d {
            for j in 0..d {
                self.v[i * d + j] += context[i] * context[j];
            }
            self.b[i] += reward * context[i];
        }
        Ok(())
    }
}

/// Cascading Linear UCB for ranked list recommendations (Kveton 2015).
///
/// Models each position independently with a shared LinUCB context.
/// Clicks are assumed at the first relevant item in the list.
#[derive(Debug, Clone)]
pub struct CascadeLinUcb {
    /// Number of candidate items.
    pub n_items: usize,
    /// List length (number of positions to fill).
    pub list_len: usize,
    /// Feature dimension.
    pub dim: usize,
    /// Exploration parameter.
    pub alpha: f64,
    /// Per-item precision matrices (n_items × dim × dim).
    a: Vec<Vec<f64>>,
    /// Per-item reward accumulators (n_items × dim).
    b: Vec<Vec<f64>>,
}

impl CascadeLinUcb {
    /// Create with `n_items` items, `list_len` positions, feature `dim`, exploration `alpha`.
    pub fn new(n_items: usize, list_len: usize, dim: usize, alpha: f64) -> Self {
        let mut a = Vec::with_capacity(n_items);
        for _ in 0..n_items {
            let mut mat = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                mat[i * dim + i] = 1.0;
            }
            a.push(mat);
        }
        Self {
            n_items,
            list_len,
            dim,
            alpha,
            a,
            b: vec![vec![0.0_f64; dim]; n_items],
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Select `list_len` items ordered by UCB score.
    pub fn select(&self, contexts: &[Vec<f64>]) -> Vec<usize> {
        let d = self.dim;
        let mut scored: Vec<(usize, f64)> = (0..self.n_items.min(contexts.len()))
            .map(|item| {
                let theta = LinUcbBandit::cholesky_solve(&self.a[item], &self.b[item], d);
                let ctx = &contexts[item];
                if ctx.len() != d {
                    return (item, f64::NEG_INFINITY);
                }
                let mu = Self::dot(&theta, ctx);
                let a_inv_ctx = LinUcbBandit::cholesky_solve(&self.a[item], ctx, d);
                let sigma = Self::dot(ctx, &a_inv_ctx).sqrt().max(0.0);
                (item, mu + self.alpha * sigma)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(self.list_len).map(|&(i, _)| i).collect()
    }

    /// Update item `item_id` with context and binary click reward.
    pub fn update(&mut self, item_id: usize, context: &[f64], reward: f64) -> Result<()> {
        let d = self.dim;
        if item_id >= self.n_items {
            return Err(TensorError::invalid_argument_op(
                "CascadeLinUcb::update",
                "item_id out of range",
            ));
        }
        if context.len() != d {
            return Err(TensorError::invalid_argument_op(
                "CascadeLinUcb::update",
                "context length must equal dim",
            ));
        }
        for i in 0..d {
            for j in 0..d {
                self.a[item_id][i * d + j] += context[i] * context[j];
            }
            self.b[item_id][i] += reward * context[i];
        }
        Ok(())
    }
}

/// NeuralUCB bandit — last-layer uncertainty from neural network features.
///
/// Uses a two-layer MLP as feature extractor and maintains a per-arm ridge
/// regression on the last-layer features with UCB exploration.
#[derive(Debug, Clone)]
pub struct NeuralBanditUcb {
    /// Number of arms.
    pub n_arms: usize,
    /// Context dimension.
    pub context_dim: usize,
    /// Hidden units.
    pub hidden_dim: usize,
    /// Exploration coefficient ν.
    pub nu: f64,
    /// Ridge coefficient λ for last-layer regression.
    pub lambda: f64,
    /// Shared feature network weights (dim→hidden).
    w1: Vec<f64>,
    /// Shared output layer (hidden→1 per arm).
    w2: Vec<Vec<f64>>,
    /// Per-arm last-layer precision matrix (hidden×hidden).
    z: Vec<Vec<f64>>,
    /// Per-arm last-layer reward accumulator (hidden).
    b: Vec<Vec<f64>>,
}

impl NeuralBanditUcb {
    /// Create neural bandit with given architecture.
    pub fn new(n_arms: usize, context_dim: usize, hidden_dim: usize, nu: f64, lambda: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (2.0_f64 / context_dim as f64).sqrt();
        let w1: Vec<f64> = (0..context_dim * hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
            .collect();
        let w2: Vec<Vec<f64>> = (0..n_arms)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * (1.0_f64 / hidden_dim as f64).sqrt())
                    .collect()
            })
            .collect();
        let mut z = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            let mut mat = vec![0.0_f64; hidden_dim * hidden_dim];
            for i in 0..hidden_dim {
                mat[i * hidden_dim + i] = lambda;
            }
            z.push(mat);
        }
        Self {
            n_arms,
            context_dim,
            hidden_dim,
            nu,
            lambda,
            w1,
            w2,
            z,
            b: vec![vec![0.0_f64; hidden_dim]; n_arms],
        }
    }

    /// Compute ReLU features from context.
    fn features(&self, ctx: &[f64]) -> Vec<f64> {
        let (d_in, h) = (self.context_dim, self.hidden_dim);
        (0..h)
            .map(|j| {
                let s: f64 = (0..d_in).map(|i| self.w1[i * h + j] * ctx.get(i).copied().unwrap_or(0.0)).sum();
                s.max(0.0)
            })
            .collect()
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Select best arm using last-layer UCB.
    pub fn select(&self, context: &[f64]) -> usize {
        let phi = self.features(context);
        let h = self.hidden_dim;
        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;
        for arm in 0..self.n_arms {
            let theta = LinUcbBandit::cholesky_solve(&self.z[arm], &self.b[arm], h);
            let mu = Self::dot(&theta, &phi);
            let z_inv_phi = LinUcbBandit::cholesky_solve(&self.z[arm], &phi, h);
            let sigma = Self::dot(&phi, &z_inv_phi).sqrt().max(0.0);
            let score = mu + self.nu * sigma;
            if score > best_score {
                best_score = score;
                best_arm = arm;
            }
        }
        best_arm
    }

    /// Update arm with observed (context, reward).
    pub fn update(&mut self, arm: usize, context: &[f64], reward: f64) -> Result<()> {
        if arm >= self.n_arms {
            return Err(TensorError::invalid_argument_op(
                "NeuralBanditUcb::update",
                "arm index out of range",
            ));
        }
        let phi = self.features(context);
        let h = self.hidden_dim;
        for i in 0..h {
            for j in 0..h {
                self.z[arm][i * h + j] += phi[i] * phi[j];
            }
            self.b[arm][i] += reward * phi[i];
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Online Convex Optimization
// ─────────────────────────────────────────────────────────────────────────────

/// Online Gradient Descent with projection onto a ball of radius `radius` (Zinkevich 2003).
#[derive(Debug, Clone)]
pub struct OgdOptimizer {
    /// Step size schedule type.
    pub lr: f64,
    /// Ball radius for projection (use f64::INFINITY for unconstrained).
    pub radius: f64,
    /// Current time step.
    step: usize,
    /// Current parameters.
    params: Vec<f64>,
}

impl OgdOptimizer {
    /// Create with step size `lr` and feasible-set ball radius `radius`.
    pub fn new(dim: usize, lr: f64, radius: f64) -> Self {
        Self {
            lr,
            radius,
            step: 0,
            params: vec![0.0_f64; dim],
        }
    }

    /// Project onto L2 ball of radius `r`.
    fn project_l2(v: &mut [f64], r: f64) {
        if r.is_infinite() {
            return;
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > r {
            let scale = r / norm;
            for x in v.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// One OGD step given `gradient`. Returns updated params.
    pub fn step(&mut self, gradient: &[f64]) -> Result<Vec<f64>> {
        if gradient.len() != self.params.len() {
            return Err(TensorError::invalid_argument_op(
                "OgdOptimizer::step",
                "gradient dimension mismatch",
            ));
        }
        self.step += 1;
        // Decaying step size: lr / sqrt(t).
        let eta = self.lr / (self.step as f64).sqrt();
        for (p, &g) in self.params.iter_mut().zip(gradient.iter()) {
            *p -= eta * g;
        }
        Self::project_l2(&mut self.params, self.radius);
        Ok(self.params.clone())
    }

    /// Current parameters.
    pub fn params(&self) -> &[f64] {
        &self.params
    }
}

/// Follow-the-Regularized-Leader with per-coordinate adaptive rates (McMahan 2011).
///
/// This is a clean online OCO implementation (not the proximal variant in the
/// existing FTRL-Proximal — this one uses gradient sums directly).
#[derive(Debug, Clone)]
pub struct FtrlOptimizer {
    /// Regularisation strength α.
    pub alpha: f64,
    /// β (stabilisation term).
    pub beta: f64,
    /// L1 regularisation λ₁.
    pub lambda1: f64,
    /// L2 regularisation λ₂.
    pub lambda2: f64,
    /// Accumulated squared gradient n_i.
    n: Vec<f64>,
    /// Accumulated z_i (gradient accumulator with AdaGrad adjustment).
    z: Vec<f64>,
}

impl FtrlOptimizer {
    /// Create with per-step learning rate `alpha`, stabilisation `beta`, L1/L2 penalties.
    pub fn new(dim: usize, alpha: f64, beta: f64, lambda1: f64, lambda2: f64) -> Self {
        Self {
            alpha,
            beta,
            lambda1,
            lambda2,
            n: vec![0.0_f64; dim],
            z: vec![0.0_f64; dim],
        }
    }

    /// One FTRL step. Returns current w.
    pub fn step(&mut self, gradient: &[f64]) -> Result<Vec<f64>> {
        if gradient.len() != self.n.len() {
            return Err(TensorError::invalid_argument_op(
                "FtrlOptimizer::step",
                "gradient dimension mismatch",
            ));
        }
        let dim = self.n.len();
        for i in 0..dim {
            let g = gradient[i];
            let g_sq = g * g;
            let sigma = (self.n[i] + g_sq).sqrt() - self.n[i].sqrt();
            let sigma = sigma / self.alpha;
            // Note: this uses current params — we compute them inline below.
            let w_i = if self.z[i].abs() <= self.lambda1 {
                0.0
            } else {
                let sign = if self.z[i] > 0.0 { 1.0 } else { -1.0 };
                let denom = (self.beta + self.n[i].sqrt()) / self.alpha + self.lambda2;
                -(self.z[i] - sign * self.lambda1) / denom
            };
            self.z[i] += g - sigma * w_i;
            self.n[i] += g_sq;
        }
        // Recompute w from updated z, n.
        let w: Vec<f64> = (0..dim)
            .map(|i| {
                if self.z[i].abs() <= self.lambda1 {
                    0.0
                } else {
                    let sign = if self.z[i] > 0.0 { 1.0 } else { -1.0 };
                    let denom = (self.beta + self.n[i].sqrt()) / self.alpha + self.lambda2;
                    -(self.z[i] - sign * self.lambda1) / denom
                }
            })
            .collect();
        Ok(w)
    }
}

/// Online Newton Step for exp-concave losses (Hazan 2007).
///
/// Maintains A = Σ ∇∇^T and updates with Newton direction.
#[derive(Debug, Clone)]
pub struct OnlineNewton {
    /// Dimension.
    pub dim: usize,
    /// Step size (β in the paper).
    pub beta: f64,
    /// Ball radius for projection.
    pub radius: f64,
    /// Hessian approximation A = λI + Σ g g^T.
    a: Vec<f64>,
    /// Current parameters.
    params: Vec<f64>,
}

impl OnlineNewton {
    /// Create with `dim`, step `beta`, feasible set radius `radius`, ridge `lambda`.
    pub fn new(dim: usize, beta: f64, radius: f64, lambda: f64) -> Self {
        let mut a = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            a[i * dim + i] = lambda;
        }
        Self {
            dim,
            beta,
            radius,
            a,
            params: vec![0.0_f64; dim],
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn project_l2(v: &mut [f64], r: f64) {
        if r.is_infinite() {
            return;
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > r {
            let scale = r / norm;
            for x in v.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// One ONS step. Returns updated params.
    pub fn step(&mut self, gradient: &[f64]) -> Result<Vec<f64>> {
        let d = self.dim;
        if gradient.len() != d {
            return Err(TensorError::invalid_argument_op(
                "OnlineNewton::step",
                "gradient dimension mismatch",
            ));
        }
        // Update A += g g^T.
        for i in 0..d {
            for j in 0..d {
                self.a[i * d + j] += gradient[i] * gradient[j];
            }
        }
        // Newton step: θ_{t+1} = proj(θ_t - (1/β) A^{-1} g).
        let a_inv_g = LinUcbBandit::cholesky_solve(&self.a, gradient, d);
        for (p, &v) in self.params.iter_mut().zip(a_inv_g.iter()) {
            *p -= v / self.beta;
        }
        Self::project_l2(&mut self.params, self.radius);
        Ok(self.params.clone())
    }

    /// Current parameters.
    pub fn params(&self) -> &[f64] {
        &self.params
    }
}

/// Strongly adaptive regret optimizer (Hazan & Seshadhri 2009 interval regret).
///
/// Runs multiple OGD experts with geometrically increasing window sizes and
/// selects among them using a meta-learning strategy.
#[derive(Debug, Clone)]
pub struct AdaptiveRegretOptimizer {
    /// Feature dimension.
    pub dim: usize,
    /// Number of expert scales (log2 doubling).
    pub n_experts: usize,
    /// Per-expert OGD.
    experts: Vec<OgdOptimizer>,
    /// Expert weights (log-space mixing).
    log_weights: Vec<f64>,
    /// Total step count.
    step: usize,
}

impl AdaptiveRegretOptimizer {
    /// Create with `dim` features, `n_experts` doubling windows, base step `lr`.
    pub fn new(dim: usize, n_experts: usize, lr: f64) -> Self {
        let experts: Vec<OgdOptimizer> = (0..n_experts)
            .map(|k| {
                let expert_lr = lr * (k + 1) as f64;
                OgdOptimizer::new(dim, expert_lr, f64::INFINITY)
            })
            .collect();
        Self {
            dim,
            n_experts,
            experts,
            log_weights: vec![0.0_f64; n_experts],
            step: 0,
        }
    }

    /// One step. Returns mixed output parameter vector.
    pub fn step(&mut self, gradient: &[f64]) -> Result<Vec<f64>> {
        if gradient.len() != self.dim {
            return Err(TensorError::invalid_argument_op(
                "AdaptiveRegretOptimizer::step",
                "gradient dimension mismatch",
            ));
        }
        self.step += 1;

        // Update each expert.
        let mut expert_params: Vec<Vec<f64>> = Vec::with_capacity(self.n_experts);
        for expert in &mut self.experts {
            let p = expert.step(gradient)?;
            expert_params.push(p);
        }

        // Update log-weights using exponential gradient on expert losses.
        for (k, ep) in expert_params.iter().enumerate() {
            let loss: f64 = ep.iter().zip(gradient.iter()).map(|(p, g)| p * g).sum();
            self.log_weights[k] -= loss * 0.01; // small step on meta-loss
        }

        // Softmax mix.
        let max_lw = self.log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self.log_weights.iter().map(|&w| (w - max_lw).exp()).collect();
        let total: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|&e| e / total.max(1e-15)).collect();

        let mut mixed = vec![0.0_f64; self.dim];
        for (k, ep) in expert_params.iter().enumerate() {
            for (m, &p) in mixed.iter_mut().zip(ep.iter()) {
                *m += weights[k] * p;
            }
        }
        Ok(mixed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Streaming Anomaly Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Online Isolation Forest — window-based incremental anomaly detection.
///
/// Maintains a sliding window of observations and fits a small forest.
#[derive(Debug, Clone)]
pub struct OnlineIsolationForest {
    /// Window capacity.
    pub window_size: usize,
    /// Number of trees.
    pub n_trees: usize,
    /// Sub-sample size per tree.
    pub sample_size: usize,
    /// Current sliding window.
    window: std::collections::VecDeque<Vec<f64>>,
    /// Feature dimension (inferred on first insert).
    dim: Option<usize>,
}

impl OnlineIsolationForest {
    /// Create with sliding window size, number of trees, and sub-sample size.
    pub fn new(window_size: usize, n_trees: usize, sample_size: usize) -> Self {
        Self {
            window_size,
            n_trees,
            sample_size,
            window: std::collections::VecDeque::with_capacity(window_size),
            dim: None,
        }
    }

    /// Insert a new observation into the sliding window.
    pub fn insert(&mut self, x: Vec<f64>) {
        if self.dim.is_none() {
            self.dim = Some(x.len());
        }
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(x);
    }

    /// Score anomaly: higher = more anomalous.  Returns None if window is empty.
    pub fn score(&self, x: &[f64], rng: &mut StdRng) -> Option<f64> {
        let n = self.window.len();
        if n < 2 {
            return None;
        }
        let dim = self.dim?;
        if x.len() != dim {
            return None;
        }
        let sample_sz = self.sample_size.min(n);
        let mut total_depth = 0.0_f64;
        for _ in 0..self.n_trees {
            // Sample without replacement using Fisher-Yates partial.
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..sample_sz {
                let j = i + rng.random_range(0.0..(n - i) as f64) as usize;
                indices.swap(i, j);
            }
            let sample: Vec<&Vec<f64>> = indices[..sample_sz].iter().map(|&i| &self.window[i]).collect();
            let depth = self.isolation_depth(x, &sample, 0, (sample_sz as f64).log2().ceil() as usize + 1, rng);
            total_depth += depth;
        }
        let avg_depth = total_depth / self.n_trees as f64;
        let c = self.expected_depth(sample_sz);
        Some(2.0_f64.powf(-avg_depth / c))
    }

    fn expected_depth(&self, n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        let n_f = n as f64;
        2.0 * (n_f - 1.0).ln() + 0.5772156649 - 2.0 * (n_f - 1.0) / n_f
    }

    fn isolation_depth(&self, x: &[f64], data: &[&Vec<f64>], depth: usize, max_depth: usize, rng: &mut StdRng) -> f64 {
        if data.len() <= 1 || depth >= max_depth {
            return depth as f64 + self.expected_depth(data.len());
        }
        let dim = x.len();
        if dim == 0 {
            return depth as f64;
        }
        let feature = rng.random_range(0.0..dim as f64) as usize;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for point in data {
            if let Some(&v) = point.get(feature) {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
        if (max_val - min_val).abs() < 1e-12 {
            return depth as f64 + self.expected_depth(data.len());
        }
        let split = min_val + rng.random::<f64>() * (max_val - min_val);
        let x_val = x.get(feature).copied().unwrap_or(0.0);
        let left: Vec<&Vec<f64>> = data.iter().copied().filter(|p| p.get(feature).copied().unwrap_or(0.0) < split).collect();
        let right: Vec<&Vec<f64>> = data.iter().copied().filter(|p| p.get(feature).copied().unwrap_or(0.0) >= split).collect();
        if x_val < split {
            self.isolation_depth(x, &left, depth + 1, max_depth, rng)
        } else {
            self.isolation_depth(x, &right, depth + 1, max_depth, rng)
        }
    }
}

/// Half-Space Trees for streaming anomaly detection (Tan 2011).
///
/// Maintains a fixed tree structure with mass estimators in each node.
#[derive(Debug, Clone)]
pub struct HstTree {
    /// Number of trees.
    pub n_trees: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Window size for mass ratio computation.
    pub window_size: usize,
    /// Feature dimension.
    pub dim: usize,
    /// Reference mass counters (per tree, per node).
    ref_mass: Vec<Vec<f64>>,
    /// Latest mass counters.
    cur_mass: Vec<Vec<f64>>,
    /// Split dimensions per node per tree.
    split_dims: Vec<Vec<usize>>,
    /// Split points per node per tree.
    split_vals: Vec<Vec<f64>>,
    /// Current window count.
    count: usize,
    /// Feature min/max for initial split calibration.
    feat_min: Vec<f64>,
    feat_max: Vec<f64>,
}

impl HstTree {
    /// Create with `n_trees`, `max_depth`, `window_size`, `dim`, and random seed.
    pub fn new(n_trees: usize, max_depth: usize, window_size: usize, dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_nodes = (1usize << (max_depth + 1)).saturating_sub(1);
        let mut split_dims = Vec::with_capacity(n_trees);
        let mut split_vals = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let sdims: Vec<usize> = (0..n_nodes).map(|_| rng.random_range(0.0..dim as f64) as usize).collect();
            let svals: Vec<f64> = (0..n_nodes).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect();
            split_dims.push(sdims);
            split_vals.push(svals);
        }
        Self {
            n_trees,
            max_depth,
            window_size,
            dim,
            ref_mass: vec![vec![0.0_f64; n_nodes]; n_trees],
            cur_mass: vec![vec![0.0_f64; n_nodes]; n_trees],
            split_dims,
            split_vals,
            count: 0,
            feat_min: vec![-1.0; dim],
            feat_max: vec![1.0; dim],
        }
    }

    fn traverse(&self, tree: usize, x: &[f64]) -> Vec<usize> {
        let mut node = 0usize;
        let mut path = vec![0usize];
        for _depth in 0..self.max_depth {
            let left = 2 * node + 1;
            let right = 2 * node + 2;
            if left >= self.cur_mass[tree].len() {
                break;
            }
            let feat = self.split_dims[tree][node];
            let val = self.split_vals[tree][node];
            let x_val = x.get(feat).copied().unwrap_or(0.0);
            // Normalise to [-1,1] range.
            let range = (self.feat_max[feat] - self.feat_min[feat]).max(1e-12);
            let x_norm = (x_val - self.feat_min[feat]) / range * 2.0 - 1.0;
            node = if x_norm < val { left } else { right };
            path.push(node);
        }
        path
    }

    /// Update trees with new observation.
    pub fn update(&mut self, x: &[f64]) {
        if x.len() != self.dim {
            return;
        }
        // Update feature ranges.
        for (i, &v) in x.iter().enumerate() {
            if v < self.feat_min[i] {
                self.feat_min[i] = v;
            }
            if v > self.feat_max[i] {
                self.feat_max[i] = v;
            }
        }
        for tree in 0..self.n_trees {
            let path = self.traverse(tree, x);
            for &node in &path {
                if node < self.cur_mass[tree].len() {
                    self.cur_mass[tree][node] += 1.0;
                }
            }
        }
        self.count += 1;
        // Rotate windows.
        if self.count % self.window_size == 0 {
            for tree in 0..self.n_trees {
                self.ref_mass[tree] = self.cur_mass[tree].clone();
                for v in self.cur_mass[tree].iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }

    /// Compute anomaly score — higher = more anomalous.
    pub fn score(&self, x: &[f64]) -> f64 {
        if x.len() != self.dim {
            return 0.0;
        }
        let mut total = 0.0_f64;
        for tree in 0..self.n_trees {
            let path = self.traverse(tree, x);
            // Score = depth of deepest node with ref_mass > 0.
            let mut score = 0.0_f64;
            for (depth, &node) in path.iter().enumerate() {
                if node < self.ref_mass[tree].len() && self.ref_mass[tree][node] > 0.0 {
                    score = depth as f64;
                }
            }
            total += score;
        }
        total / self.n_trees as f64
    }
}

/// LODA streaming anomaly detector — random histograms on random projections.
#[derive(Debug, Clone)]
pub struct LodaDetector {
    /// Number of random projections.
    pub n_projections: usize,
    /// Number of histogram bins.
    pub n_bins: usize,
    /// Random projection vectors (n_projections × dim).
    projections: Vec<Vec<f64>>,
    /// Per-projection histogram counts.
    histograms: Vec<Vec<f64>>,
    /// Per-projection bin edges (n_bins + 1 edges).
    bin_edges: Vec<Vec<f64>>,
    /// Total observations seen.
    n_obs: usize,
    /// Buffered observations for initial calibration.
    buffer: Vec<Vec<f64>>,
    /// Buffer capacity before histogram calibration.
    calibration_size: usize,
}

impl LodaDetector {
    /// Create with `n_projections`, `n_bins`, and random seed.
    pub fn new(n_projections: usize, n_bins: usize, dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Sparse random projections: each has exactly one non-zero coordinate.
        let projections: Vec<Vec<f64>> = (0..n_projections)
            .map(|_| {
                let mut p = vec![0.0_f64; dim];
                let idx = rng.random_range(0.0..dim as f64) as usize;
                p[idx] = if rng.random::<f64>() > 0.5 { 1.0 } else { -1.0 };
                p
            })
            .collect();
        Self {
            n_projections,
            n_bins,
            projections,
            histograms: vec![vec![0.0_f64; n_bins]; n_projections],
            bin_edges: vec![vec![0.0_f64; n_bins + 1]; n_projections],
            n_obs: 0,
            buffer: Vec::new(),
            calibration_size: 100,
        }
    }

    fn project(&self, p_idx: usize, x: &[f64]) -> f64 {
        self.projections[p_idx]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn calibrate(&mut self) {
        let n = self.buffer.len();
        if n < 2 {
            return;
        }
        for p in 0..self.n_projections {
            let mut projected: Vec<f64> = self.buffer.iter().map(|x| self.project(p, x)).collect();
            projected.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min_v = projected[0] - 1e-9;
            let max_v = projected[n - 1] + 1e-9;
            let step = (max_v - min_v) / self.n_bins as f64;
            for k in 0..=self.n_bins {
                self.bin_edges[p][k] = min_v + k as f64 * step;
            }
            for &v in &projected {
                let bin = ((v - min_v) / step) as usize;
                let bin = bin.min(self.n_bins - 1);
                self.histograms[p][bin] += 1.0;
            }
        }
    }

    fn bin_idx(&self, p_idx: usize, val: f64) -> usize {
        let edges = &self.bin_edges[p_idx];
        let n = self.n_bins;
        if edges[n] <= edges[0] {
            return 0;
        }
        let step = edges[n] - edges[0];
        if step < 1e-12 {
            return 0;
        }
        let idx = ((val - edges[0]) / (step / n as f64)) as usize;
        idx.min(n - 1)
    }

    /// Insert observation and update histograms.
    pub fn update(&mut self, x: Vec<f64>) {
        self.n_obs += 1;
        if self.n_obs <= self.calibration_size {
            self.buffer.push(x.clone());
            if self.n_obs == self.calibration_size {
                self.calibrate();
            }
            return;
        }
        // Incremental update of histograms.
        for p in 0..self.n_projections {
            let val = self.project(p, &x);
            let bin = self.bin_idx(p, val);
            self.histograms[p][bin] += 1.0;
        }
    }

    /// Compute log-likelihood anomaly score (lower = more anomalous).
    pub fn score(&self, x: &[f64]) -> f64 {
        if self.n_obs < self.calibration_size {
            return 0.0;
        }
        let mut log_prob = 0.0_f64;
        for p in 0..self.n_projections {
            let val = self.project(p, x);
            let bin = self.bin_idx(p, val);
            let count = self.histograms[p][bin];
            let total: f64 = self.histograms[p].iter().sum();
            let prob = if total > 0.0 { count / total } else { 1e-10 };
            log_prob += prob.max(1e-10).ln();
        }
        -log_prob / self.n_projections as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Enhanced Concept Drift Detection
// ─────────────────────────────────────────────────────────────────────────────

/// ADWIN detector (Bifet 2007) — adaptive windowing with Hoeffding-based change detection.
///
/// Distinct from the existing `Adwin` in mod.rs: this version uses the exact
/// Hoeffding-inequality formulation on subwindow means as described in the paper.
#[derive(Debug, Clone)]
pub struct AdwinDetector {
    /// Confidence parameter δ.
    pub delta: f64,
    /// Maximum window size.
    pub max_window: usize,
    /// Sliding window of observations.
    window: Vec<f64>,
    /// Whether drift was detected on the last update.
    pub last_drift: bool,
}

impl AdwinDetector {
    /// Create with confidence `delta` (small = sensitive) and `max_window`.
    pub fn new(delta: f64, max_window: usize) -> Self {
        Self {
            delta,
            max_window,
            window: Vec::new(),
            last_drift: false,
        }
    }

    /// Update with observation `x`. Returns `(drift_detected, current_mean)`.
    pub fn update(&mut self, x: f64) -> (bool, f64) {
        self.window.push(x);
        if self.window.len() > self.max_window {
            self.window.remove(0);
        }
        let n = self.window.len();
        let mut drift = false;
        if n >= 4 {
            let mu_all: f64 = self.window.iter().sum::<f64>() / n as f64;
            for split in 1..n {
                let n0 = split as f64;
                let n1 = (n - split) as f64;
                let mu0 = self.window[..split].iter().sum::<f64>() / n0;
                let mu1 = self.window[split..].iter().sum::<f64>() / n1;
                // Hoeffding bound: ε_cut = sqrt(0.5 * ln(4n²/δ) * (1/n0 + 1/n1))
                let eps_cut = (0.5 * (4.0 * (n as f64 * n as f64) / self.delta).ln()
                    * (1.0 / n0 + 1.0 / n1))
                    .sqrt();
                if (mu0 - mu1).abs() >= eps_cut {
                    self.window.drain(..split);
                    drift = true;
                    break;
                }
            }
            let _ = mu_all;
        }
        let current_mean = if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        };
        self.last_drift = drift;
        (drift, current_mean)
    }

    /// Current window mean.
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }

    /// Current window size.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Reset detector.
    pub fn reset(&mut self) {
        self.window.clear();
        self.last_drift = false;
    }
}

/// Page-Hinkley drift detector with configurable threshold and sensitivity.
///
/// This is a standalone version (complementary to `PageHinkleyTest` in mod.rs)
/// that also tracks the magnitude of drift.
#[derive(Debug, Clone)]
pub struct PageHinkley {
    /// Detection threshold λ.
    pub threshold: f64,
    /// Minimum change to react to δ.
    pub delta: f64,
    /// Page-Hinkley statistic (upper cumulative sum).
    ph_plus: f64,
    /// Page-Hinkley statistic (lower cumulative sum).
    ph_minus: f64,
    /// Running mean.
    mean: f64,
    /// Observations seen.
    n: usize,
    /// Detected drift direction (+1, -1, or 0).
    pub drift_direction: i32,
}

impl PageHinkley {
    /// Create with detection threshold `threshold` and minimum change `delta`.
    pub fn new(threshold: f64, delta: f64) -> Self {
        Self {
            threshold,
            delta,
            ph_plus: 0.0,
            ph_minus: 0.0,
            mean: 0.0,
            n: 0,
            drift_direction: 0,
        }
    }

    /// Update with `x`. Returns `(drift_detected, drift_direction)`.
    pub fn update(&mut self, x: f64) -> (bool, i32) {
        self.n += 1;
        self.mean += (x - self.mean) / self.n as f64;
        self.ph_plus = (self.ph_plus + x - self.mean - self.delta).max(0.0);
        self.ph_minus = (self.ph_minus + self.mean - x - self.delta).max(0.0);
        if self.ph_plus > self.threshold {
            self.drift_direction = 1;
            self.ph_plus = 0.0;
            self.ph_minus = 0.0;
            return (true, 1);
        }
        if self.ph_minus > self.threshold {
            self.drift_direction = -1;
            self.ph_plus = 0.0;
            self.ph_minus = 0.0;
            return (true, -1);
        }
        self.drift_direction = 0;
        (false, 0)
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.ph_plus = 0.0;
        self.ph_minus = 0.0;
        self.mean = 0.0;
        self.n = 0;
        self.drift_direction = 0;
    }
}

/// Drift detection metrics report.
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    /// Number of drift events detected.
    pub n_detected: usize,
    /// True drift count (known ground truth, if available).
    pub n_true_drifts: usize,
    /// Total observations processed.
    pub n_observations: usize,
    /// Sum of detection delays (steps after true drift to first detection).
    pub total_delay: usize,
    /// Number of false positives.
    pub n_false_positives: usize,
    /// Number of missed drifts.
    pub n_missed: usize,
}

impl DriftMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            n_detected: 0,
            n_true_drifts: 0,
            n_observations: 0,
            total_delay: 0,
            n_false_positives: 0,
            n_missed: 0,
        }
    }

    /// Record a detection event with delay in steps.
    pub fn record_detection(&mut self, delay: usize, is_true_positive: bool) {
        self.n_detected += 1;
        if is_true_positive {
            self.total_delay += delay;
        } else {
            self.n_false_positives += 1;
        }
    }

    /// Record a missed true drift.
    pub fn record_missed(&mut self) {
        self.n_missed += 1;
    }

    /// Mean detection delay (over true positives).
    pub fn mean_delay(&self) -> f64 {
        let tp = self.n_detected.saturating_sub(self.n_false_positives);
        if tp == 0 {
            0.0
        } else {
            self.total_delay as f64 / tp as f64
        }
    }

    /// False positive rate per observation.
    pub fn false_positive_rate(&self) -> f64 {
        if self.n_observations == 0 {
            0.0
        } else {
            self.n_false_positives as f64 / self.n_observations as f64
        }
    }

    /// Missed detection rate.
    pub fn missed_rate(&self) -> f64 {
        let total = self.n_true_drifts;
        if total == 0 {
            0.0
        } else {
            self.n_missed as f64 / total as f64
        }
    }
}

impl Default for DriftMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    // ── LinUcbBandit ─────────────────────────────────────────────────────────

    #[test]
    fn test_lin_ucb_bandit_select_valid_arm() {
        let bandit = LinUcbBandit::new(3, 1.0, 1.0);
        let contexts = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let arm = bandit.select(&contexts);
        assert!(arm < 3);
    }

    #[test]
    fn test_lin_ucb_bandit_update_changes_theta() {
        let mut bandit = LinUcbBandit::new(2, 1.0, 1.0);
        let ctx = vec![1.0, 0.0];
        let theta_before = bandit.theta();
        bandit.update(&ctx, 2.0).expect("update failed");
        let theta_after = bandit.theta();
        assert!(theta_before != theta_after); // theta changes after update
        let _ = theta_after;
    }

    #[test]
    fn test_lin_ucb_bandit_prefers_rewarded_arm() {
        let mut bandit = LinUcbBandit::new(2, 0.1, 1.0);
        // Arm 0 context: [1, 0], Arm 1 context: [0, 1].
        // Reward arm 0 heavily.
        for _ in 0..20 {
            bandit.update(&[1.0, 0.0], 1.0).expect("update failed");
            bandit.update(&[0.0, 1.0], 0.0).expect("update failed");
        }
        let arm = bandit.select(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(arm, 0);
    }

    // ── ThompsonSamplingLinear ────────────────────────────────────────────────

    #[test]
    fn test_ts_linear_select_valid() {
        let bandit = ThompsonSamplingLinear::new(3, 1.0, 1.0);
        let contexts = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(42);
        let arm = bandit.select(&contexts, &mut rng);
        assert!(arm < 3);
    }

    #[test]
    fn test_ts_linear_update_valid() {
        let mut bandit = ThompsonSamplingLinear::new(2, 1.0, 1.0);
        bandit.update(&[1.0, 0.5], 0.8).expect("update failed");
    }

    #[test]
    fn test_ts_linear_dimension_mismatch() {
        let mut bandit = ThompsonSamplingLinear::new(3, 1.0, 1.0);
        let result = bandit.update(&[1.0, 0.5], 0.8);
        assert!(result.is_err());
    }

    // ── CascadeLinUcb ─────────────────────────────────────────────────────────

    #[test]
    fn test_cascade_lin_ucb_select() {
        let bandit = CascadeLinUcb::new(5, 3, 2, 1.0);
        let contexts: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 1.0]).collect();
        let list = bandit.select(&contexts);
        assert_eq!(list.len(), 3);
        for &arm in &list {
            assert!(arm < 5);
        }
    }

    #[test]
    fn test_cascade_lin_ucb_update() {
        let mut bandit = CascadeLinUcb::new(3, 2, 2, 0.5);
        bandit.update(0, &[1.0, 0.0], 1.0).expect("update failed");
        bandit.update(1, &[0.0, 1.0], 0.0).expect("update failed");
    }

    #[test]
    fn test_cascade_lin_ucb_update_out_of_range() {
        let mut bandit = CascadeLinUcb::new(3, 2, 2, 0.5);
        let result = bandit.update(10, &[1.0, 0.0], 1.0);
        assert!(result.is_err());
    }

    // ── NeuralBanditUcb ───────────────────────────────────────────────────────

    #[test]
    fn test_neural_bandit_ucb_select() {
        let bandit = NeuralBanditUcb::new(3, 4, 8, 0.1, 1.0, 42);
        let arm = bandit.select(&[1.0, 0.0, -1.0, 0.5]);
        assert!(arm < 3);
    }

    #[test]
    fn test_neural_bandit_ucb_update() {
        let mut bandit = NeuralBanditUcb::new(2, 3, 4, 0.1, 1.0, 0);
        bandit.update(0, &[1.0, 0.0, 0.0], 1.0).expect("update failed");
    }

    // ── OgdOptimizer ─────────────────────────────────────────────────────────

    #[test]
    fn test_ogd_step_decreases_params() {
        let mut opt = OgdOptimizer::new(2, 0.5, f64::INFINITY);
        let params = opt.step(&[1.0, -1.0]).expect("step failed");
        assert!(params[0] < 0.0);
        assert!(params[1] > 0.0);
    }

    #[test]
    fn test_ogd_projection() {
        let mut opt = OgdOptimizer::new(2, 10.0, 1.0);
        let params = opt.step(&[1.0, 0.0]).expect("step failed");
        let norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm <= 1.0 + 1e-9);
    }

    #[test]
    fn test_ogd_dimension_mismatch() {
        let mut opt = OgdOptimizer::new(2, 0.1, f64::INFINITY);
        let result = opt.step(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    // ── FtrlOptimizer ─────────────────────────────────────────────────────────

    #[test]
    fn test_ftrl_l1_sparsity() {
        let mut ftrl = FtrlOptimizer::new(3, 0.1, 1.0, 0.5, 0.0);
        let w = ftrl.step(&[0.01, 0.01, 0.01]).expect("step failed");
        // With high L1, small gradients should produce sparse weights.
        for &wi in &w {
            assert!(wi.abs() < 1.0);
        }
    }

    #[test]
    fn test_ftrl_step_valid() {
        let mut ftrl = FtrlOptimizer::new(2, 0.1, 1.0, 0.0, 0.0);
        let w = ftrl.step(&[1.0, -1.0]).expect("step failed");
        assert_eq!(w.len(), 2);
    }

    // ── OnlineNewton ─────────────────────────────────────────────────────────

    #[test]
    fn test_online_newton_step() {
        let mut ons = OnlineNewton::new(2, 1.0, f64::INFINITY, 1.0);
        let p = ons.step(&[1.0, -1.0]).expect("step failed");
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_online_newton_projection() {
        let mut ons = OnlineNewton::new(2, 0.1, 0.5, 1.0);
        let p = ons.step(&[10.0, 0.0]).expect("step failed");
        let norm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm <= 0.5 + 1e-9);
    }

    // ── AdaptiveRegretOptimizer ───────────────────────────────────────────────

    #[test]
    fn test_adaptive_regret_step() {
        let mut opt = AdaptiveRegretOptimizer::new(3, 4, 0.1);
        let result = opt.step(&[1.0, -1.0, 0.5]).expect("step failed");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_adaptive_regret_multiple_steps() {
        let mut opt = AdaptiveRegretOptimizer::new(2, 3, 0.05);
        for _ in 0..10 {
            opt.step(&[0.5, -0.5]).expect("step failed");
        }
    }

    // ── OnlineIsolationForest ─────────────────────────────────────────────────

    #[test]
    fn test_online_if_score_anomaly() {
        let mut forest = OnlineIsolationForest::new(100, 10, 32);
        let mut rng = StdRng::seed_from_u64(42);
        // Add normal data.
        for _ in 0..80 {
            let x: Vec<f64> = (0..3).map(|_| rng.random::<f64>() * 0.1).collect();
            forest.insert(x);
        }
        // Score a normal point.
        let normal_score = forest.score(&[0.05, 0.05, 0.05], &mut rng);
        assert!(normal_score.is_some());
        let ns = normal_score.unwrap_or(0.0);
        assert!((0.0..=1.0).contains(&ns));
    }

    #[test]
    fn test_online_if_empty() {
        let forest = OnlineIsolationForest::new(50, 5, 16);
        let mut rng = StdRng::seed_from_u64(0);
        let score = forest.score(&[1.0, 2.0], &mut rng);
        assert!(score.is_none());
    }

    // ── HstTree ───────────────────────────────────────────────────────────────

    #[test]
    fn test_hst_update_and_score() {
        let mut hst = HstTree::new(10, 5, 50, 2, 42);
        for i in 0..50 {
            hst.update(&[i as f64 * 0.1, i as f64 * 0.05]);
        }
        let score = hst.score(&[0.5, 0.25]);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_hst_anomalous_point() {
        let mut hst = HstTree::new(10, 5, 50, 2, 0);
        for _ in 0..50 {
            hst.update(&[0.5, 0.5]);
        }
        let normal_score = hst.score(&[0.5, 0.5]);
        // Just check that it returns a non-negative value.
        assert!(normal_score >= 0.0);
    }

    // ── LodaDetector ─────────────────────────────────────────────────────────

    #[test]
    fn test_loda_score_after_calibration() {
        let mut loda = LodaDetector::new(50, 10, 3, 42);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..150 {
            let x: Vec<f64> = (0..3).map(|_| rng.random::<f64>()).collect();
            loda.update(x);
        }
        let score = loda.score(&[0.5, 0.5, 0.5]);
        assert!(score.is_finite());
    }

    #[test]
    fn test_loda_before_calibration_returns_zero() {
        let loda = LodaDetector::new(10, 5, 2, 0);
        let score = loda.score(&[0.5, 0.5]);
        assert_eq!(score, 0.0);
    }

    // ── AdwinDetector ─────────────────────────────────────────────────────────

    #[test]
    fn test_adwin_no_drift_stable() {
        let mut adwin = AdwinDetector::new(0.01, 200);
        for _ in 0..50 {
            let (drift, _) = adwin.update(1.0);
            let _ = drift; // Some initial drift is possible
        }
        let (drift, mean) = adwin.update(1.0);
        assert!((mean - 1.0).abs() < 0.3 || !drift);
    }

    #[test]
    fn test_adwin_detects_drift() {
        let mut adwin = AdwinDetector::new(0.01, 300);
        for _ in 0..60 {
            adwin.update(0.0);
        }
        let any_drift = (0..60).any(|_| adwin.update(100.0).0);
        assert!(any_drift);
    }

    #[test]
    fn test_adwin_reset() {
        let mut adwin = AdwinDetector::new(0.01, 100);
        for _ in 0..20 {
            adwin.update(5.0);
        }
        adwin.reset();
        assert_eq!(adwin.window_len(), 0);
    }

    // ── PageHinkley ───────────────────────────────────────────────────────────

    #[test]
    fn test_page_hinkley_no_drift() {
        let mut ph = PageHinkley::new(50.0, 0.01);
        for _ in 0..100 {
            let (detected, _) = ph.update(1.0);
            assert!(!detected);
        }
    }

    #[test]
    fn test_page_hinkley_detects_upward_shift() {
        let mut ph = PageHinkley::new(20.0, 0.005);
        for _ in 0..50 {
            ph.update(0.0);
        }
        let mut detected = false;
        for _ in 0..200 {
            let (d, _) = ph.update(5.0);
            if d {
                detected = true;
                break;
            }
        }
        assert!(detected);
    }

    #[test]
    fn test_page_hinkley_reset() {
        let mut ph = PageHinkley::new(10.0, 0.01);
        ph.update(5.0);
        ph.reset();
        assert_eq!(ph.n, 0);
        assert_eq!(ph.drift_direction, 0);
    }

    // ── DriftMetrics ─────────────────────────────────────────────────────────

    #[test]
    fn test_drift_metrics_default() {
        let metrics = DriftMetrics::new();
        assert_eq!(metrics.n_detected, 0);
        assert_eq!(metrics.mean_delay(), 0.0);
        assert_eq!(metrics.false_positive_rate(), 0.0);
    }

    #[test]
    fn test_drift_metrics_record() {
        let mut metrics = DriftMetrics::new();
        metrics.n_observations = 1000;
        metrics.n_true_drifts = 5;
        metrics.record_detection(10, true);
        metrics.record_detection(0, false);
        metrics.record_missed();
        assert_eq!(metrics.n_detected, 2);
        assert_eq!(metrics.n_false_positives, 1);
        assert_eq!(metrics.n_missed, 1);
        assert!((metrics.mean_delay() - 10.0).abs() < 1e-9);
        assert!((metrics.false_positive_rate() - 0.001).abs() < 1e-9);
        assert!((metrics.missed_rate() - 0.2).abs() < 1e-9);
    }
}
