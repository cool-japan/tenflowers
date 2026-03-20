//! # Safe Reinforcement Learning & Constrained Optimization
//!
//! Production-grade pure-Rust implementations of safe RL algorithms for
//! Constrained Markov Decision Processes (CMDPs).
//!
//! ## Modules
//!
//! - \[`SrlCmdpModel`\] — CMDP with cost functions and thresholds.
//! - [`SrlLagrangianRl`] — Lagrangian relaxation for CMDPs (PID-based lambda).
//! - [`SrlCpoAgent`] — Constrained Policy Optimization (Achiam et al. 2017).
//! - [`SrlSafetyLayer`] — Action correction via learned constraint models (Dalal et al. 2018).
//! - [`SrlSafeExplorer`] — Ensemble-based pessimistic safe exploration.
//! - [`SrlRobustMdp`] — Robust MDP with minimax value iteration.
//! - [`SrlShieldedPolicy`] — Runtime safety shield (BFS reachability).
//! - [`SrlBarrierFunction`] — Control Barrier Functions (CBFs) with QP projection.
//! - [`SrlMetrics`] / [`SrlReport`] — Safety evaluation metrics.
//!
//! ## References
//!
//! - Achiam, J. et al. (2017). "Constrained Policy Optimization."
//! - Dalal, G. et al. (2018). "Safe Exploration in Continuous Action Spaces."
//! - Iyengar, G. (2005). "Robust Dynamic Programming."
//! - Ames, A. et al. (2017). "Control Barrier Functions."
//! - Altman, E. (1999). "Constrained Markov Decision Processes."

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// §0  Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Stable softmax over a slice of f64 values.
fn srl_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Dot product of two slices.
fn srl_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
fn srl_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Clip a value to [lo, hi].
fn srl_clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Xavier initialization for a weight matrix (flattened).
fn srl_xavier_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.random_range(-limit..limit))
        .collect()
}

/// Simple linear layer: y = Wx + b.
#[derive(Debug, Clone)]
pub struct SrlLinear {
    /// Weight matrix (out_dim x in_dim), stored row-major.
    pub weights: Vec<f64>,
    /// Bias vector (out_dim).
    pub bias: Vec<f64>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl SrlLinear {
    /// Create a new linear layer with Xavier initialization.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let weights = srl_xavier_init(in_dim, out_dim, seed);
        let bias = vec![0.0; out_dim];
        Self {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: y = Wx + b.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.in_dim {
            return Err(TensorError::compute_error_simple(format!(
                "SrlLinear: expected input dim {}, got {}",
                self.in_dim,
                input.len()
            )));
        }
        let mut output = self.bias.clone();
        for i in 0..self.out_dim {
            let row_start = i * self.in_dim;
            for j in 0..self.in_dim {
                output[i] += self.weights[row_start + j] * input[j];
            }
        }
        Ok(output)
    }

    /// Update weights by gradient descent: w -= lr * grad_w, b -= lr * grad_b.
    pub fn sgd_step(&mut self, grad_w: &[f64], grad_b: &[f64], lr: f64) {
        for (w, g) in self.weights.iter_mut().zip(grad_w.iter()) {
            *w -= lr * g;
        }
        for (b, g) in self.bias.iter_mut().zip(grad_b.iter()) {
            *b -= lr * g;
        }
    }
}

/// Simple 2-layer MLP with ReLU activation.
#[derive(Debug, Clone)]
pub struct SrlMlp {
    /// First linear layer.
    pub layer1: SrlLinear,
    /// Second linear layer.
    pub layer2: SrlLinear,
}

impl SrlMlp {
    /// Create an MLP: in_dim -> hidden_dim (ReLU) -> out_dim.
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            layer1: SrlLinear::new(in_dim, hidden_dim, seed),
            layer2: SrlLinear::new(hidden_dim, out_dim, seed.wrapping_add(1)),
        }
    }

    /// Forward pass with ReLU after first layer.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let h = self.layer1.forward(input)?;
        let h_relu: Vec<f64> = h.iter().map(|&x| x.max(0.0)).collect();
        self.layer2.forward(&h_relu)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  SrlCmdpModel — Constrained Markov Decision Process
// ─────────────────────────────────────────────────────────────────────────────

/// Transition result from a CMDP step.
#[derive(Debug, Clone)]
pub struct SrlStepResult {
    /// Next state vector.
    pub next_state: Vec<f64>,
    /// Scalar reward.
    pub reward: f64,
    /// Cost vector (one per constraint).
    pub costs: Vec<f64>,
    /// Whether the episode terminated.
    pub done: bool,
}

/// Configuration for a CMDP.
#[derive(Debug, Clone)]
pub struct SrlCmdpConfig {
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension (discrete count).
    pub action_dim: usize,
    /// Number of cost constraints.
    pub num_constraints: usize,
    /// Cost thresholds d_i: E[sum gamma^t c_i] <= d_i.
    pub cost_thresholds: Vec<f64>,
    /// Discount factor gamma.
    pub gamma: f64,
}

/// CMDP environment trait — step function returning rewards and costs.
pub trait SrlCmdpEnv {
    /// Reset the environment, returning the initial state.
    fn reset(&mut self, seed: u64) -> Result<Vec<f64>>;
    /// Take an action, returning (next_state, reward, costs, done).
    fn step(&mut self, action: usize) -> Result<SrlStepResult>;
    /// Get current state dimension.
    fn state_dim(&self) -> usize;
    /// Get action dimension (number of discrete actions).
    fn action_dim(&self) -> usize;
    /// Get number of cost constraints.
    fn num_constraints(&self) -> usize;
}

/// SafeGridWorld: a grid environment with obstacles.
/// Agent gets reward for reaching goal, cost for entering obstacle-adjacent cells.
#[derive(Debug, Clone)]
pub struct SrlSafeGridWorld {
    /// Grid size (grid_size x grid_size).
    pub grid_size: usize,
    /// Current agent position (row, col).
    pub position: (usize, usize),
    /// Goal position.
    pub goal: (usize, usize),
    /// Obstacle positions.
    pub obstacles: Vec<(usize, usize)>,
    /// Max steps per episode.
    pub max_steps: usize,
    /// Current step count.
    step_count: usize,
}

impl SrlSafeGridWorld {
    /// Create a new SafeGridWorld.
    pub fn new(grid_size: usize, obstacles: Vec<(usize, usize)>) -> Self {
        let goal = (grid_size.saturating_sub(1), grid_size.saturating_sub(1));
        Self {
            grid_size,
            position: (0, 0),
            goal,
            obstacles,
            max_steps: grid_size * grid_size * 2,
            step_count: 0,
        }
    }

    fn state_vec(&self) -> Vec<f64> {
        vec![
            self.position.0 as f64 / self.grid_size.max(1) as f64,
            self.position.1 as f64 / self.grid_size.max(1) as f64,
            self.goal.0 as f64 / self.grid_size.max(1) as f64,
            self.goal.1 as f64 / self.grid_size.max(1) as f64,
        ]
    }

    fn is_obstacle_adjacent(&self, r: usize, c: usize) -> bool {
        for &(or, oc) in &self.obstacles {
            let dr = r.abs_diff(or);
            let dc = c.abs_diff(oc);
            if dr + dc <= 1 {
                return true;
            }
        }
        false
    }
}

impl SrlCmdpEnv for SrlSafeGridWorld {
    fn reset(&mut self, _seed: u64) -> Result<Vec<f64>> {
        self.position = (0, 0);
        self.step_count = 0;
        Ok(self.state_vec())
    }

    fn step(&mut self, action: usize) -> Result<SrlStepResult> {
        // Actions: 0=up, 1=down, 2=left, 3=right
        let (mut r, mut c) = self.position;
        match action {
            0 => r = r.saturating_sub(1),
            1 => r = (r + 1).min(self.grid_size.saturating_sub(1)),
            2 => c = c.saturating_sub(1),
            3 => c = (c + 1).min(self.grid_size.saturating_sub(1)),
            _ => {} // no-op for invalid actions
        }

        // Check if on obstacle — stay in place
        if self.obstacles.contains(&(r, c)) {
            // stay
        } else {
            self.position = (r, c);
        }

        self.step_count += 1;
        let done = self.position == self.goal || self.step_count >= self.max_steps;
        let reward = if self.position == self.goal {
            1.0
        } else {
            -0.01
        };

        // Cost: 1.0 if adjacent to obstacle, 0.0 otherwise
        let cost = if self.is_obstacle_adjacent(self.position.0, self.position.1) {
            1.0
        } else {
            0.0
        };

        Ok(SrlStepResult {
            next_state: self.state_vec(),
            reward,
            costs: vec![cost],
            done,
        })
    }

    fn state_dim(&self) -> usize {
        4
    }
    fn action_dim(&self) -> usize {
        4
    }
    fn num_constraints(&self) -> usize {
        1
    }
}

/// SafeCartPole: CartPole with angle constraint.
/// Cost = 1 if |angle| > angle_limit.
#[derive(Debug, Clone)]
pub struct SrlSafeCartPole {
    /// Cart position.
    pub x: f64,
    /// Cart velocity.
    pub x_dot: f64,
    /// Pole angle (radians).
    pub theta: f64,
    /// Pole angular velocity.
    pub theta_dot: f64,
    /// Angle constraint limit (radians).
    pub angle_limit: f64,
    /// Max steps.
    pub max_steps: usize,
    /// Current step.
    step_count: usize,
    /// Timestep for dynamics.
    pub dt: f64,
}

impl SrlSafeCartPole {
    /// Create a new SafeCartPole with given angle limit.
    pub fn new(angle_limit: f64) -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            angle_limit,
            max_steps: 200,
            step_count: 0,
            dt: 0.02,
        }
    }

    fn state_vec(&self) -> Vec<f64> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

impl SrlCmdpEnv for SrlSafeCartPole {
    fn reset(&mut self, seed: u64) -> Result<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.x = rng.random_range(-0.05..0.05);
        self.x_dot = rng.random_range(-0.05..0.05);
        self.theta = rng.random_range(-0.05..0.05);
        self.theta_dot = rng.random_range(-0.05..0.05);
        self.step_count = 0;
        Ok(self.state_vec())
    }

    fn step(&mut self, action: usize) -> Result<SrlStepResult> {
        // Simplified cart-pole dynamics
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5;
        let force_mag = 10.0;

        let force = if action == 1 { force_mag } else { -force_mag };
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        let temp =
            (force + mass_pole * length * self.theta_dot * self.theta_dot * sin_theta) / total_mass;
        let theta_acc = (gravity * sin_theta - cos_theta * temp)
            / (length * (4.0 / 3.0 - mass_pole * cos_theta * cos_theta / total_mass));
        let x_acc = temp - mass_pole * length * theta_acc * cos_theta / total_mass;

        // Euler integration
        self.x += self.dt * self.x_dot;
        self.x_dot += self.dt * x_acc;
        self.theta += self.dt * self.theta_dot;
        self.theta_dot += self.dt * theta_acc;

        self.step_count += 1;

        let failed = self.theta.abs() > 0.4 || self.x.abs() > 2.4;
        let done = failed || self.step_count >= self.max_steps;
        let reward = if failed { 0.0 } else { 1.0 };

        // Cost: 1.0 if angle exceeds safety limit
        let cost = if self.theta.abs() > self.angle_limit {
            1.0
        } else {
            0.0
        };

        Ok(SrlStepResult {
            next_state: self.state_vec(),
            reward,
            costs: vec![cost],
            done,
        })
    }

    fn state_dim(&self) -> usize {
        4
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn num_constraints(&self) -> usize {
        1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  SrlLagrangianRl — Lagrangian relaxation for CMDP
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Lagrangian RL.
#[derive(Debug, Clone)]
pub struct SrlLagrangianConfig {
    /// Number of constraints.
    pub num_constraints: usize,
    /// Cost thresholds d_i.
    pub cost_thresholds: Vec<f64>,
    /// Learning rate for dual variables (lambda).
    pub dual_lr: f64,
    /// Learning rate for policy.
    pub policy_lr: f64,
    /// PID proportional gain.
    pub pid_kp: f64,
    /// PID integral gain.
    pub pid_ki: f64,
    /// PID derivative gain.
    pub pid_kd: f64,
    /// Max lambda value (for stability).
    pub lambda_max: f64,
}

impl Default for SrlLagrangianConfig {
    fn default() -> Self {
        Self {
            num_constraints: 1,
            cost_thresholds: vec![25.0],
            dual_lr: 0.01,
            policy_lr: 0.001,
            pid_kp: 1.0,
            pid_ki: 0.01,
            pid_kd: 0.0,
            lambda_max: 100.0,
        }
    }
}

/// Result of a Lagrangian training step.
#[derive(Debug, Clone)]
pub struct SrlLagrangianStepResult {
    /// Policy loss (Lagrangian objective).
    pub policy_loss: f64,
    /// Per-constraint violation (J_ci - d_i).
    pub constraint_violations: Vec<f64>,
    /// Current Lagrange multipliers.
    pub lambdas: Vec<f64>,
}

/// Lagrangian RL agent with PID-based dual variable updates.
#[derive(Debug, Clone)]
pub struct SrlLagrangianRl {
    /// Configuration.
    pub config: SrlLagrangianConfig,
    /// Lagrange multipliers (one per constraint).
    pub lambdas: Vec<f64>,
    /// PID integral term.
    integral: Vec<f64>,
    /// PID previous error.
    prev_error: Vec<f64>,
    /// Policy network.
    pub policy: SrlMlp,
}

impl SrlLagrangianRl {
    /// Create a new Lagrangian RL agent.
    pub fn new(
        config: SrlLagrangianConfig,
        state_dim: usize,
        action_dim: usize,
        seed: u64,
    ) -> Self {
        let n = config.num_constraints;
        Self {
            lambdas: vec![0.0; n],
            integral: vec![0.0; n],
            prev_error: vec![0.0; n],
            policy: SrlMlp::new(state_dim, 64, action_dim, seed),
            config,
        }
    }

    /// Select action using softmax policy.
    pub fn select_action(&self, state: &[f64], seed: u64) -> Result<usize> {
        let logits = self.policy.forward(state)?;
        let probs = srl_softmax(&logits);
        let mut rng = StdRng::seed_from_u64(seed);
        let r: f64 = rng.random_range(0.0..1.0);
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i);
            }
        }
        Ok(probs.len().saturating_sub(1))
    }

    /// Perform a training step on collected trajectories.
    ///
    /// `rewards`: per-step rewards for batch of trajectories.
    /// `costs`: per-step costs, shape \[num_constraints\]\[num_steps\].
    pub fn train_step(
        &mut self,
        rewards: &[f64],
        costs: &[Vec<f64>],
    ) -> Result<SrlLagrangianStepResult> {
        if costs.len() != self.config.num_constraints {
            return Err(TensorError::compute_error_simple(format!(
                "SrlLagrangianRl: expected {} constraint cost vectors, got {}",
                self.config.num_constraints,
                costs.len()
            )));
        }

        let total_reward: f64 = rewards.iter().sum();
        let num_steps = rewards.len().max(1) as f64;
        let avg_reward = total_reward / num_steps;

        // Compute average cost per constraint
        let mut avg_costs = Vec::with_capacity(self.config.num_constraints);
        let mut violations = Vec::with_capacity(self.config.num_constraints);
        for i in 0..self.config.num_constraints {
            let avg_c: f64 = costs[i].iter().sum::<f64>() / num_steps;
            avg_costs.push(avg_c);
            let threshold = self.config.cost_thresholds.get(i).copied().unwrap_or(0.0);
            violations.push(avg_c - threshold);
        }

        // PID-based dual variable update
        for i in 0..self.config.num_constraints {
            let error = violations[i];
            self.integral[i] += error;
            let derivative = error - self.prev_error[i];
            self.prev_error[i] = error;

            let pid_update = self.config.pid_kp * error
                + self.config.pid_ki * self.integral[i]
                + self.config.pid_kd * derivative;

            self.lambdas[i] = srl_clip(
                self.lambdas[i] + self.config.dual_lr * pid_update,
                0.0,
                self.config.lambda_max,
            );
        }

        // Lagrangian loss: L = -J_r + sum lambda_i * (J_ci - d_i)
        let mut policy_loss = -avg_reward;
        for i in 0..self.config.num_constraints {
            policy_loss += self.lambdas[i] * violations[i];
        }

        Ok(SrlLagrangianStepResult {
            policy_loss,
            constraint_violations: violations,
            lambdas: self.lambdas.clone(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  SrlCpoAgent — Constrained Policy Optimization
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for CPO.
#[derive(Debug, Clone)]
pub struct SrlCpoConfig {
    /// KL divergence constraint (trust region).
    pub max_kl: f64,
    /// Cost limit per episode.
    pub cost_limit: f64,
    /// Backtracking coefficient.
    pub backtrack_coeff: f64,
    /// Maximum backtracking iterations.
    pub max_backtracks: usize,
    /// Line search acceptance ratio.
    pub accept_ratio: f64,
    /// Discount factor.
    pub gamma: f64,
    /// GAE lambda.
    pub gae_lambda: f64,
}

impl Default for SrlCpoConfig {
    fn default() -> Self {
        Self {
            max_kl: 0.01,
            cost_limit: 25.0,
            backtrack_coeff: 0.8,
            max_backtracks: 10,
            accept_ratio: 0.1,
            gamma: 0.99,
            gae_lambda: 0.97,
        }
    }
}

/// Result of a CPO update step.
#[derive(Debug, Clone)]
pub struct SrlCpoUpdateResult {
    /// Surrogate objective improvement.
    pub surrogate_improvement: f64,
    /// KL divergence after update.
    pub kl_divergence: f64,
    /// Mean constraint cost.
    pub mean_cost: f64,
    /// Whether the constraint was satisfied.
    pub constraint_satisfied: bool,
    /// Number of backtracking steps used.
    pub backtrack_steps: usize,
}

/// CPO agent (Achiam et al. 2017).
#[derive(Debug, Clone)]
pub struct SrlCpoAgent {
    /// Configuration.
    pub config: SrlCpoConfig,
    /// Policy network (state -> action logits).
    pub policy: SrlMlp,
    /// Value network (state -> V(s)).
    pub value_net: SrlMlp,
    /// Cost value network (state -> V_c(s)).
    pub cost_value_net: SrlMlp,
}

impl SrlCpoAgent {
    /// Create a new CPO agent.
    pub fn new(config: SrlCpoConfig, state_dim: usize, action_dim: usize, seed: u64) -> Self {
        Self {
            config,
            policy: SrlMlp::new(state_dim, 64, action_dim, seed),
            value_net: SrlMlp::new(state_dim, 64, 1, seed.wrapping_add(10)),
            cost_value_net: SrlMlp::new(state_dim, 64, 1, seed.wrapping_add(20)),
        }
    }

    /// Select action using softmax policy.
    pub fn select_action(&self, state: &[f64], seed: u64) -> Result<usize> {
        let logits = self.policy.forward(state)?;
        let probs = srl_softmax(&logits);
        let mut rng = StdRng::seed_from_u64(seed);
        let r: f64 = rng.random_range(0.0..1.0);
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return Ok(i);
            }
        }
        Ok(probs.len().saturating_sub(1))
    }

    /// Compute GAE advantages given rewards, values, and dones.
    pub fn compute_gae(&self, rewards: &[f64], values: &[f64], dones: &[bool]) -> Vec<f64> {
        let n = rewards.len();
        let mut advantages = vec![0.0; n];
        let mut last_gae = 0.0;
        for t in (0..n).rev() {
            let next_val = if t + 1 < n && !dones[t] {
                values[t + 1]
            } else {
                0.0
            };
            let delta = rewards[t] + self.config.gamma * next_val - values[t];
            let mask = if dones[t] { 0.0 } else { 1.0 };
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae;
            advantages[t] = last_gae;
        }
        advantages
    }

    /// Update policy with CPO constraints.
    ///
    /// `states`: trajectory states.
    /// `actions`: trajectory actions.
    /// `rewards`: trajectory rewards.
    /// `costs`: trajectory costs.
    /// `dones`: trajectory dones.
    pub fn update(
        &mut self,
        states: &[Vec<f64>],
        actions: &[usize],
        rewards: &[f64],
        costs: &[f64],
        dones: &[bool],
    ) -> Result<SrlCpoUpdateResult> {
        let n = states.len();
        if n == 0 {
            return Err(TensorError::compute_error_simple(
                "SrlCpoAgent::update: empty trajectory".to_string(),
            ));
        }

        // Compute values
        let mut values = Vec::with_capacity(n);
        let mut cost_values = Vec::with_capacity(n);
        for s in states {
            let v = self.value_net.forward(s)?;
            values.push(v.first().copied().unwrap_or(0.0));
            let cv = self.cost_value_net.forward(s)?;
            cost_values.push(cv.first().copied().unwrap_or(0.0));
        }

        // Compute reward advantages
        let reward_advantages = self.compute_gae(rewards, &values, dones);

        // Compute cost advantages
        let cost_advantages = self.compute_gae(costs, &cost_values, dones);

        // Compute old log-probs
        let mut old_log_probs = Vec::with_capacity(n);
        for (i, s) in states.iter().enumerate() {
            let logits = self.policy.forward(s)?;
            let probs = srl_softmax(&logits);
            let action = actions[i];
            let p = if action < probs.len() {
                probs[action].max(1e-10)
            } else {
                1e-10
            };
            old_log_probs.push(p.ln());
        }

        // Compute surrogate objective and constraint cost
        let mean_advantage: f64 = reward_advantages.iter().sum::<f64>() / n as f64;
        let mean_cost_advantage: f64 = cost_advantages.iter().sum::<f64>() / n as f64;
        let mean_cost: f64 = costs.iter().sum::<f64>() / n as f64;

        // CPO dual: solve for step direction
        // Linearized constraint: J_c(pi_k) + g^T (pi - pi_k) <= d
        // where g is the cost advantage gradient direction
        let constraint_violation = mean_cost - self.config.cost_limit / (n as f64).max(1.0);

        // Simple rescaled gradient approach with backtracking
        let step_size = (2.0 * self.config.max_kl / (mean_advantage.abs() + 1e-8)).sqrt();

        let mut best_step = 0;
        let mut best_improvement = 0.0;
        let mut final_kl = 0.0;
        

        // Backtracking line search
        let original_weights_l1 = self.policy.layer1.weights.clone();
        let original_bias_l1 = self.policy.layer1.bias.clone();
        let original_weights_l2 = self.policy.layer2.weights.clone();
        let original_bias_l2 = self.policy.layer2.bias.clone();

        let mut found = false;
        for step_idx in 0..self.config.max_backtracks {
            let alpha = step_size * self.config.backtrack_coeff.powi(step_idx as i32);

            // Apply gradient step to policy weights (simplified: direct weight perturbation)
            for (w, ow) in self
                .policy
                .layer2
                .weights
                .iter_mut()
                .zip(original_weights_l2.iter())
            {
                *w = ow + alpha * mean_advantage.signum() * 0.01;
            }

            // Check constraint and KL
            let mut new_log_probs = Vec::with_capacity(n);
            for (i, s) in states.iter().enumerate() {
                let logits = self.policy.forward(s)?;
                let probs = srl_softmax(&logits);
                let action = actions[i];
                let p = if action < probs.len() {
                    probs[action].max(1e-10)
                } else {
                    1e-10
                };
                new_log_probs.push(p.ln());
            }

            // Approximate KL as mean of ratio differences
            let kl: f64 = old_log_probs
                .iter()
                .zip(new_log_probs.iter())
                .map(|(&old, &new)| {
                    let ratio = (old - new).exp();
                    ratio * (old - new)
                })
                .sum::<f64>()
                / n as f64;

            let improvement = mean_advantage * alpha;

            if kl.abs() <= self.config.max_kl
                && (constraint_violation <= 0.0 || mean_cost_advantage * alpha <= 0.0)
            {
                best_step = step_idx;
                best_improvement = improvement;
                final_kl = kl.abs();
                found = true;
                break;
            }
        }

        if !found {
            // Restore original weights
            self.policy.layer1.weights = original_weights_l1;
            self.policy.layer1.bias = original_bias_l1;
            self.policy.layer2.weights = original_weights_l2;
            self.policy.layer2.bias = original_bias_l2;
            best_improvement = 0.0;
            final_kl = 0.0;
        }

        let constraint_satisfied = mean_cost <= self.config.cost_limit / (n as f64).max(1.0);

        Ok(SrlCpoUpdateResult {
            surrogate_improvement: best_improvement,
            kl_divergence: final_kl,
            mean_cost,
            constraint_satisfied,
            backtrack_steps: best_step,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  SrlSafetyLayer — Action correction (Dalal et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the safety layer.
#[derive(Debug, Clone)]
pub struct SrlSafetyLayerConfig {
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension (continuous).
    pub action_dim: usize,
    /// Number of constraints.
    pub num_constraints: usize,
    /// Learning rate for constraint model.
    pub lr: f64,
    /// Safety margin (how far below threshold to target).
    pub safety_margin: f64,
}

/// Learned linear constraint model: g(s, a) = w_s^T s + w_a^T a + b.
#[derive(Debug, Clone)]
pub struct SrlConstraintModel {
    /// Weight for state features.
    pub w_state: Vec<f64>,
    /// Weight for action features.
    pub w_action: Vec<f64>,
    /// Bias.
    pub bias: f64,
}

impl SrlConstraintModel {
    /// Create a zero-initialized constraint model.
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            w_state: vec![0.0; state_dim],
            w_action: vec![0.0; action_dim],
            bias: 0.0,
        }
    }

    /// Evaluate g(s, a).
    pub fn evaluate(&self, state: &[f64], action: &[f64]) -> f64 {
        srl_dot(&self.w_state, state) + srl_dot(&self.w_action, action) + self.bias
    }

    /// Gradient of g w.r.t. action: ∇_a g = w_action.
    pub fn action_gradient(&self) -> &[f64] {
        &self.w_action
    }

    /// Update constraint model from (state, action, observed_cost) samples.
    pub fn update(&mut self, state: &[f64], action: &[f64], target: f64, lr: f64) {
        let pred = self.evaluate(state, action);
        let error = pred - target;
        // SGD on MSE: grad = 2 * error * feature
        for (w, &s) in self.w_state.iter_mut().zip(state.iter()) {
            *w -= lr * 2.0 * error * s;
        }
        for (w, &a) in self.w_action.iter_mut().zip(action.iter()) {
            *w -= lr * 2.0 * error * a;
        }
        self.bias -= lr * 2.0 * error;
    }
}

/// Safety layer that corrects proposed actions to satisfy constraints.
#[derive(Debug, Clone)]
pub struct SrlSafetyLayer {
    /// Configuration.
    pub config: SrlSafetyLayerConfig,
    /// Learned constraint models (one per constraint).
    pub constraint_models: Vec<SrlConstraintModel>,
}

impl SrlSafetyLayer {
    /// Create a new safety layer.
    pub fn new(config: SrlSafetyLayerConfig) -> Self {
        let models = (0..config.num_constraints)
            .map(|_| SrlConstraintModel::new(config.state_dim, config.action_dim))
            .collect();
        Self {
            config,
            constraint_models: models,
        }
    }

    /// Correct a proposed action to satisfy constraints.
    ///
    /// Projects unsafe actions: a_safe = a - lambda * grad_a(g)
    /// where lambda is chosen so g(s, a_safe) <= -margin.
    pub fn correct_action(&self, state: &[f64], proposed_action: &[f64]) -> Result<Vec<f64>> {
        if proposed_action.len() != self.config.action_dim {
            return Err(TensorError::compute_error_simple(format!(
                "SrlSafetyLayer: expected action dim {}, got {}",
                self.config.action_dim,
                proposed_action.len()
            )));
        }

        let mut action = proposed_action.to_vec();

        for model in &self.constraint_models {
            let g_val = model.evaluate(state, &action);
            if g_val > -self.config.safety_margin {
                // Need to correct: project along -grad_a(g)
                let grad = model.action_gradient();
                let grad_norm_sq = srl_dot(grad, grad);
                if grad_norm_sq > 1e-12 {
                    // lambda such that g(s, a - lambda * grad) = -margin
                    // Linear approximation: g - lambda * ||grad||^2 = -margin
                    let lambda = (g_val + self.config.safety_margin) / grad_norm_sq;
                    for (a, &g) in action.iter_mut().zip(grad.iter()) {
                        *a -= lambda * g;
                    }
                }
            }
        }

        Ok(action)
    }

    /// Update constraint models from observed transitions.
    pub fn update_models(
        &mut self,
        state: &[f64],
        action: &[f64],
        observed_costs: &[f64],
    ) -> Result<()> {
        if observed_costs.len() != self.config.num_constraints {
            return Err(TensorError::compute_error_simple(format!(
                "SrlSafetyLayer: expected {} costs, got {}",
                self.config.num_constraints,
                observed_costs.len()
            )));
        }
        for (i, model) in self.constraint_models.iter_mut().enumerate() {
            model.update(state, action, observed_costs[i], self.config.lr);
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  SrlSafeExplorer — Uncertainty-based safe exploration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for safe exploration.
#[derive(Debug, Clone)]
pub struct SrlSafeExplorerConfig {
    /// Number of ensemble models.
    pub ensemble_size: usize,
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension (discrete).
    pub action_dim: usize,
    /// Pessimism coefficient beta: c + beta * sigma(c) <= threshold.
    pub beta: f64,
    /// Cost threshold for safety.
    pub cost_threshold: f64,
    /// Minimum safety probability for exploration.
    pub min_safety_prob: f64,
}

/// Ensemble-based safe exploration with pessimistic constraints.
#[derive(Debug, Clone)]
pub struct SrlSafeExplorer {
    /// Configuration.
    pub config: SrlSafeExplorerConfig,
    /// Ensemble of dynamics/cost models (each: state+action -> predicted cost).
    pub ensemble: Vec<SrlMlp>,
}

impl SrlSafeExplorer {
    /// Create a new safe explorer with ensemble of cost predictors.
    pub fn new(config: SrlSafeExplorerConfig, seed: u64) -> Self {
        let ensemble = (0..config.ensemble_size)
            .map(|i| {
                // Input: state concatenated with one-hot action
                let in_dim = config.state_dim + config.action_dim;
                SrlMlp::new(in_dim, 32, 1, seed.wrapping_add(i as u64 * 100))
            })
            .collect();
        Self { config, ensemble }
    }

    /// Predict cost mean and standard deviation for (state, action) pair.
    pub fn predict_cost_stats(&self, state: &[f64], action: usize) -> Result<(f64, f64)> {
        // Build input: [state; one_hot(action)]
        let mut input = state.to_vec();
        for i in 0..self.config.action_dim {
            input.push(if i == action { 1.0 } else { 0.0 });
        }

        let mut predictions = Vec::with_capacity(self.config.ensemble_size);
        for model in &self.ensemble {
            let out = model.forward(&input)?;
            predictions.push(out.first().copied().unwrap_or(0.0));
        }

        let n = predictions.len() as f64;
        let mean = predictions.iter().sum::<f64>() / n.max(1.0);
        let variance = predictions
            .iter()
            .map(|&p| (p - mean) * (p - mean))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        let std_dev = variance.sqrt();

        Ok((mean, std_dev))
    }

    /// Check if (state, action) is safe with pessimistic bound.
    /// Safe if: mean_cost + beta * std_cost <= threshold.
    pub fn is_safe(&self, state: &[f64], action: usize) -> Result<bool> {
        let (mean, std) = self.predict_cost_stats(state, action)?;
        let pessimistic_cost = mean + self.config.beta * std;
        Ok(pessimistic_cost <= self.config.cost_threshold)
    }

    /// Get safety probability (fraction of ensemble members predicting safe).
    pub fn safety_probability(&self, state: &[f64], action: usize) -> Result<f64> {
        let mut input = state.to_vec();
        for i in 0..self.config.action_dim {
            input.push(if i == action { 1.0 } else { 0.0 });
        }

        let mut safe_count = 0;
        for model in &self.ensemble {
            let out = model.forward(&input)?;
            let pred = out.first().copied().unwrap_or(0.0);
            if pred <= self.config.cost_threshold {
                safe_count += 1;
            }
        }

        let prob = safe_count as f64 / self.config.ensemble_size.max(1) as f64;
        Ok(prob)
    }

    /// Select the safest action among those meeting minimum safety probability.
    pub fn safe_select_action(
        &self,
        state: &[f64],
        policy_prefs: &[f64],
        seed: u64,
    ) -> Result<usize> {
        let n_actions = self.config.action_dim;
        let mut safe_actions = Vec::new();

        for a in 0..n_actions {
            let prob = self.safety_probability(state, a)?;
            if prob >= self.config.min_safety_prob {
                safe_actions.push((a, prob));
            }
        }

        if safe_actions.is_empty() {
            // Fall back to action with highest safety probability
            let mut best_action = 0;
            let mut best_prob = f64::NEG_INFINITY;
            for a in 0..n_actions {
                let prob = self.safety_probability(state, a)?;
                if prob > best_prob {
                    best_prob = prob;
                    best_action = a;
                }
            }
            return Ok(best_action);
        }

        // Among safe actions, sample proportional to policy preferences
        let mut rng = StdRng::seed_from_u64(seed);
        let filtered_prefs: Vec<f64> = safe_actions
            .iter()
            .map(|&(a, _)| {
                if a < policy_prefs.len() {
                    policy_prefs[a].max(0.0)
                } else {
                    1.0
                }
            })
            .collect();

        let sum: f64 = filtered_prefs.iter().sum();
        if sum <= 0.0 {
            return Ok(safe_actions[0].0);
        }

        let r: f64 = rng.random_range(0.0..sum);
        let mut cumulative = 0.0;
        for (i, &pref) in filtered_prefs.iter().enumerate() {
            cumulative += pref;
            if r < cumulative {
                return Ok(safe_actions[i].0);
            }
        }

        Ok(safe_actions.last().map(|&(a, _)| a).unwrap_or(0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  SrlRobustMdp — Robust MDP (minimax)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for robust MDP.
#[derive(Debug, Clone)]
pub struct SrlRobustMdpConfig {
    /// Number of states.
    pub num_states: usize,
    /// Number of actions.
    pub num_actions: usize,
    /// Discount factor gamma.
    pub gamma: f64,
    /// Maximum value iteration steps.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

/// Robust MDP solver with rectangular uncertainty sets.
#[derive(Debug, Clone)]
pub struct SrlRobustMdp {
    /// Configuration.
    pub config: SrlRobustMdpConfig,
    /// Nominal transition probabilities: P\[s\]\[a\][s'] = probability.
    pub transitions: Vec<Vec<Vec<f64>>>,
    /// Reward function: R\[s\]\[a\] = scalar reward.
    pub rewards: Vec<Vec<f64>>,
    /// Uncertainty radius for (s,a)-rectangular model.
    pub uncertainty_radii: Vec<Vec<f64>>,
}

impl SrlRobustMdp {
    /// Create a new robust MDP.
    pub fn new(
        config: SrlRobustMdpConfig,
        transitions: Vec<Vec<Vec<f64>>>,
        rewards: Vec<Vec<f64>>,
        uncertainty_radii: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let ns = config.num_states;
        let na = config.num_actions;
        if transitions.len() != ns {
            return Err(TensorError::compute_error_simple(format!(
                "SrlRobustMdp: transitions outer dimension should be {}, got {}",
                ns,
                transitions.len()
            )));
        }
        Ok(Self {
            config,
            transitions,
            rewards,
            uncertainty_radii,
        })
    }

    /// Compute the worst-case transition distribution for (s, a) given value function V.
    /// Under (s,a)-rectangular uncertainty, the adversary minimizes E_p[V(s')].
    /// We use a simple L1-ball perturbation: shift mass from max V(s') to min V(s').
    fn worst_case_expected_value(&self, s: usize, a: usize, value: &[f64]) -> f64 {
        let ns = self.config.num_states;
        let p_nominal = &self.transitions[s][a];
        let radius = if s < self.uncertainty_radii.len() && a < self.uncertainty_radii[s].len() {
            self.uncertainty_radii[s][a]
        } else {
            0.0
        };

        if radius <= 0.0 || ns == 0 {
            // No uncertainty — just compute E[V]
            return p_nominal
                .iter()
                .zip(value.iter())
                .map(|(&p, &v)| p * v)
                .sum();
        }

        // Sort states by value to identify where to shift mass
        let mut sv: Vec<(usize, f64)> = (0..ns).map(|i| (i, value[i])).collect();
        sv.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Worst case: shift mass toward lowest-value states
        let mut p_worst = p_nominal.to_vec();
        let mut budget = radius;

        // Move mass from highest-value states to lowest-value states
        let mut lo = 0;
        let mut hi = ns.saturating_sub(1);

        while lo < hi && budget > 1e-12 {
            let lo_idx = sv[lo].0;
            let hi_idx = sv[hi].0;

            let available = p_worst[hi_idx].min(budget / 2.0);
            if available <= 0.0 {
                hi = hi.saturating_sub(1);
                continue;
            }

            p_worst[hi_idx] -= available;
            p_worst[lo_idx] += available;
            budget -= 2.0 * available;

            if p_worst[hi_idx] < 1e-15 {
                hi = hi.saturating_sub(1);
            }
            lo += 1;
        }

        // Ensure probabilities are valid
        let sum: f64 = p_worst.iter().sum();
        if sum > 0.0 {
            for p in &mut p_worst {
                *p /= sum;
            }
        }

        p_worst.iter().zip(value.iter()).map(|(&p, &v)| p * v).sum()
    }

    /// Solve the robust MDP via robust value iteration.
    /// V(s) = max_a min_p [r(s,a) + gamma * E_p[V(s')]]
    pub fn solve(&self) -> Result<(Vec<f64>, Vec<usize>)> {
        let ns = self.config.num_states;
        let na = self.config.num_actions;
        let mut value = vec![0.0; ns];
        let mut policy = vec![0usize; ns];

        for _iter in 0..self.config.max_iterations {
            let mut new_value = vec![f64::NEG_INFINITY; ns];

            for s in 0..ns {
                for a in 0..na {
                    let r = if s < self.rewards.len() && a < self.rewards[s].len() {
                        self.rewards[s][a]
                    } else {
                        0.0
                    };

                    let worst_ev = self.worst_case_expected_value(s, a, &value);
                    let q_sa = r + self.config.gamma * worst_ev;

                    if q_sa > new_value[s] {
                        new_value[s] = q_sa;
                        policy[s] = a;
                    }
                }
            }

            // Check convergence
            let max_diff = value
                .iter()
                .zip(new_value.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);

            value = new_value;

            if max_diff < self.config.tolerance {
                break;
            }
        }

        Ok((value, policy))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  SrlShieldedPolicy — Runtime safety shield
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the shielded policy.
#[derive(Debug, Clone)]
pub struct SrlShieldConfig {
    /// Number of discretized states per dimension.
    pub grid_resolution: usize,
    /// State space bounds: (low, high) per dimension.
    pub state_bounds: Vec<(f64, f64)>,
    /// Number of discrete actions.
    pub num_actions: usize,
}

/// Runtime safety shield using backward reachability.
#[derive(Debug, Clone)]
pub struct SrlShieldedPolicy {
    /// Configuration.
    pub config: SrlShieldConfig,
    /// Safe set: grid_index -> is_safe.
    pub safe_set: Vec<bool>,
    /// Allowed actions per state: grid_index -> Vec<action_idx>.
    pub allowed_actions: Vec<Vec<usize>>,
    /// Total grid size (product of resolution^dim).
    grid_total: usize,
    /// State dimensionality.
    state_dim: usize,
}

impl SrlShieldedPolicy {
    /// Create a new shielded policy.
    pub fn new(config: SrlShieldConfig) -> Self {
        let state_dim = config.state_bounds.len();
        let grid_total = config.grid_resolution.pow(state_dim as u32);
        let safe_set = vec![true; grid_total]; // initially all safe
        let allowed_actions = vec![(0..config.num_actions).collect(); grid_total];
        Self {
            config,
            safe_set,
            allowed_actions,
            grid_total,
            state_dim,
        }
    }

    /// Convert continuous state to grid index.
    pub fn state_to_grid_index(&self, state: &[f64]) -> usize {
        let mut index = 0;
        let mut multiplier = 1;
        let res = self.config.grid_resolution;
        for d in 0..self.state_dim.min(state.len()) {
            let (lo, hi) = self.config.state_bounds[d];
            let range = (hi - lo).max(1e-10);
            let normalized = ((state[d] - lo) / range).clamp(0.0, 1.0 - 1e-10);
            let bin = (normalized * res as f64) as usize;
            let bin = bin.min(res.saturating_sub(1));
            index += bin * multiplier;
            multiplier *= res;
        }
        index.min(self.grid_total.saturating_sub(1))
    }

    /// Convert grid index to center of cell in continuous state space.
    pub fn grid_index_to_state(&self, index: usize) -> Vec<f64> {
        let mut state = Vec::with_capacity(self.state_dim);
        let res = self.config.grid_resolution;
        let mut remainder = index;
        for d in 0..self.state_dim {
            let bin = remainder % res;
            remainder /= res;
            let (lo, hi) = self.config.state_bounds[d];
            let center = lo + (bin as f64 + 0.5) * (hi - lo) / res as f64;
            state.push(center);
        }
        state
    }

    /// Compute safe set via BFS backward reachability from known-safe terminal states.
    ///
    /// `unsafe_states`: grid indices of definitely-unsafe states.
    /// `transition_fn`: given (grid_index, action) -> next_grid_index.
    pub fn compute_safe_set<F>(&mut self, unsafe_states: &[usize], transition_fn: F)
    where
        F: Fn(usize, usize) -> usize,
    {
        // Mark unsafe states
        for &idx in unsafe_states {
            if idx < self.grid_total {
                self.safe_set[idx] = false;
            }
        }

        // BFS backward from unsafe states to propagate unsafety
        let mut queue = std::collections::VecDeque::new();
        let na = self.config.num_actions;

        // Build reverse transition map
        let mut predecessors: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.grid_total];
        for s in 0..self.grid_total {
            for a in 0..na {
                let next = transition_fn(s, a);
                if next < self.grid_total {
                    predecessors[next].push((s, a));
                }
            }
        }

        // Initialize queue with unsafe states
        for &idx in unsafe_states {
            if idx < self.grid_total {
                queue.push_back(idx);
            }
        }

        // BFS: if ALL actions from a state lead to unsafe, mark it unsafe
        let mut changed = true;
        while changed {
            changed = false;
            for s in 0..self.grid_total {
                if !self.safe_set[s] {
                    continue;
                }
                // Update allowed actions: remove those leading to unsafe states
                let old_allowed = self.allowed_actions[s].clone();
                self.allowed_actions[s] = old_allowed
                    .iter()
                    .filter(|&&a| {
                        let next = transition_fn(s, a);
                        next < self.grid_total && self.safe_set[next]
                    })
                    .copied()
                    .collect();

                // If no safe actions remain, mark state as unsafe
                if self.allowed_actions[s].is_empty() && !old_allowed.is_empty() {
                    self.safe_set[s] = false;
                    changed = true;
                }
            }
        }
    }

    /// Shield a proposed action: if safe, allow it; otherwise, pick closest safe action.
    pub fn shield(&self, state: &[f64], proposed_action: usize) -> usize {
        let idx = self.state_to_grid_index(state);
        let allowed = &self.allowed_actions[idx.min(self.grid_total.saturating_sub(1))];

        if allowed.contains(&proposed_action) {
            proposed_action
        } else if let Some(&first_safe) = allowed.first() {
            // Pick the allowed action closest to proposed
            let mut best = first_safe;
            let mut best_dist = proposed_action.abs_diff(first_safe);
            for &a in allowed.iter().skip(1) {
                let dist = proposed_action.abs_diff(a);
                if dist < best_dist {
                    best_dist = dist;
                    best = a;
                }
            }
            best
        } else {
            // No safe actions; return proposed as fallback
            proposed_action
        }
    }

    /// Check if a state is in the safe set.
    pub fn is_state_safe(&self, state: &[f64]) -> bool {
        let idx = self.state_to_grid_index(state);
        idx < self.safe_set.len() && self.safe_set[idx]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  SrlBarrierFunction — Control Barrier Functions (CBFs)
// ─────────────────────────────────────────────────────────────────────────────

/// Barrier function type.
#[derive(Debug, Clone, Copy)]
pub enum SrlBarrierType {
    /// Standard CBF: dh/dt + alpha * h >= 0.
    Standard,
    /// Exponential CBF: dh/dt + alpha * h^gamma >= 0 (for higher relative degree).
    Exponential {
        /// Exponent gamma.
        gamma: f64,
    },
}

/// Configuration for control barrier function.
#[derive(Debug, Clone)]
pub struct SrlBarrierConfig {
    /// State dimension.
    pub state_dim: usize,
    /// Control dimension.
    pub control_dim: usize,
    /// CBF class-K function parameter alpha.
    pub alpha: f64,
    /// Barrier type.
    pub barrier_type: SrlBarrierType,
    /// Finite difference step for gradient estimation.
    pub fd_step: f64,
    /// Control bounds (min, max) per dimension.
    pub control_bounds: Vec<(f64, f64)>,
}

/// Control barrier function h(x): safe set = {x : h(x) > 0}.
pub trait SrlBarrierFn: std::fmt::Debug {
    /// Evaluate h(x) — positive means safe.
    fn evaluate(&self, state: &[f64]) -> f64;
    /// Evaluate the barrier function name (for debugging).
    fn name(&self) -> &str;
}

/// Sphere barrier: h(x) = R^2 - ||x - center||^2.
#[derive(Debug, Clone)]
pub struct SrlSphereBarrier {
    /// Center of safe region.
    pub center: Vec<f64>,
    /// Radius of safe region.
    pub radius: f64,
}

impl SrlBarrierFn for SrlSphereBarrier {
    fn evaluate(&self, state: &[f64]) -> f64 {
        let dist_sq: f64 = state
            .iter()
            .zip(self.center.iter())
            .map(|(s, c)| (s - c) * (s - c))
            .sum();
        self.radius * self.radius - dist_sq
    }

    fn name(&self) -> &str {
        "SphereBarrier"
    }
}

/// Halfspace barrier: h(x) = a^T x + b (safe if positive).
#[derive(Debug, Clone)]
pub struct SrlHalfspaceBarrier {
    /// Normal vector a.
    pub normal: Vec<f64>,
    /// Offset b.
    pub offset: f64,
}

impl SrlBarrierFn for SrlHalfspaceBarrier {
    fn evaluate(&self, state: &[f64]) -> f64 {
        srl_dot(&self.normal, state) + self.offset
    }

    fn name(&self) -> &str {
        "HalfspaceBarrier"
    }
}

/// CBF-based safe control computation.
#[derive(Debug)]
pub struct SrlBarrierFunction {
    /// Configuration.
    pub config: SrlBarrierConfig,
    /// Barrier function.
    pub barrier: Box<dyn SrlBarrierFn>,
}

impl SrlBarrierFunction {
    /// Create a new barrier function controller.
    pub fn new(config: SrlBarrierConfig, barrier: Box<dyn SrlBarrierFn>) -> Self {
        Self { config, barrier }
    }

    /// Compute gradient of h(x) via finite differences.
    fn compute_gradient(&self, state: &[f64]) -> Vec<f64> {
        let n = state.len();
        let mut grad = Vec::with_capacity(n);
        let eps = self.config.fd_step;
        for i in 0..n {
            let mut s_plus = state.to_vec();
            let mut s_minus = state.to_vec();
            s_plus[i] += eps;
            s_minus[i] -= eps;
            let dh =
                (self.barrier.evaluate(&s_plus) - self.barrier.evaluate(&s_minus)) / (2.0 * eps);
            grad.push(dh);
        }
        grad
    }

    /// Compute safe control given state and nominal (desired) control.
    ///
    /// Uses gradient projection to enforce CBF constraint:
    /// dh/dt + alpha * h(x) >= 0  (standard)
    /// dh/dt + alpha * h(x)^gamma >= 0  (exponential)
    pub fn compute_safe_control(&self, state: &[f64], nominal_control: &[f64]) -> Result<Vec<f64>> {
        if nominal_control.len() != self.config.control_dim {
            return Err(TensorError::compute_error_simple(format!(
                "SrlBarrierFunction: expected control dim {}, got {}",
                self.config.control_dim,
                nominal_control.len()
            )));
        }

        let h = self.barrier.evaluate(state);
        let grad_h = self.compute_gradient(state);

        // CBF constraint value
        let cbf_bound = match self.config.barrier_type {
            SrlBarrierType::Standard => self.config.alpha * h,
            SrlBarrierType::Exponential { gamma } => {
                if h > 0.0 {
                    self.config.alpha * h.powf(gamma)
                } else {
                    // When h <= 0, already unsafe, use strong correction
                    self.config.alpha * h
                }
            }
        };

        // dh/dt ≈ grad_h^T * f(x, u) ≈ grad_h^T * u (simplified: control-affine)
        // Need: grad_h^T u + cbf_bound >= 0
        // i.e., grad_h^T u >= -cbf_bound

        // Project to ensure: grad_h_ctrl^T * u >= -cbf_bound
        // where grad_h_ctrl is the part of grad_h corresponding to control dims
        let grad_ctrl: Vec<f64> = if grad_h.len() >= self.config.control_dim {
            grad_h[..self.config.control_dim].to_vec()
        } else {
            let mut g = grad_h.clone();
            g.resize(self.config.control_dim, 0.0);
            g
        };

        let constraint_value = srl_dot(&grad_ctrl, nominal_control) + cbf_bound;

        if constraint_value >= 0.0 {
            // Constraint already satisfied — return nominal control (clipped to bounds)
            let mut u = nominal_control.to_vec();
            for (i, u_i) in u.iter_mut().enumerate() {
                if i < self.config.control_bounds.len() {
                    let (lo, hi) = self.config.control_bounds[i];
                    *u_i = srl_clip(*u_i, lo, hi);
                }
            }
            return Ok(u);
        }

        // Need correction: u_safe = u + lambda * grad_ctrl
        // where lambda chosen so grad_ctrl^T (u + lambda * grad_ctrl) + cbf_bound = 0
        let grad_norm_sq = srl_dot(&grad_ctrl, &grad_ctrl);
        let mut u_safe = nominal_control.to_vec();

        if grad_norm_sq > 1e-12 {
            let lambda = -constraint_value / grad_norm_sq;
            for (i, u_i) in u_safe.iter_mut().enumerate() {
                if i < grad_ctrl.len() {
                    *u_i += lambda * grad_ctrl[i];
                }
            }
        }

        // Clip to bounds
        for (i, u_i) in u_safe.iter_mut().enumerate() {
            if i < self.config.control_bounds.len() {
                let (lo, hi) = self.config.control_bounds[i];
                *u_i = srl_clip(*u_i, lo, hi);
            }
        }

        Ok(u_safe)
    }

    /// Check if a state is in the safe set (h(x) > 0).
    pub fn is_safe(&self, state: &[f64]) -> bool {
        self.barrier.evaluate(state) > 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  SrlMetrics — Safety evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Safety evaluation metrics for a set of episodes.
#[derive(Debug, Clone)]
pub struct SrlMetrics {
    /// Constraint violation rate (fraction of steps with any violation).
    pub constraint_violation_rate: f64,
    /// Average constraint cost per step.
    pub average_constraint_cost: f64,
    /// Average reward per episode.
    pub average_reward: f64,
    /// Safety-adjusted return: reward * (1 - violation_rate).
    pub safety_adjusted_return: f64,
    /// Cumulative constraint violations (regret).
    pub cumulative_violations: f64,
    /// Cost rate (violations per episode).
    pub cost_rate: f64,
    /// Number of episodes evaluated.
    pub num_episodes: usize,
    /// Total steps across all episodes.
    pub total_steps: usize,
}

/// Full report with metrics and Pareto front points.
#[derive(Debug, Clone)]
pub struct SrlReport {
    /// Core metrics.
    pub metrics: SrlMetrics,
    /// Pareto front points: (reward, safety_score).
    pub pareto_front: Vec<(f64, f64)>,
}

/// Episode data for safety evaluation.
#[derive(Debug, Clone)]
pub struct SrlEpisodeData {
    /// Per-step rewards.
    pub rewards: Vec<f64>,
    /// Per-step costs (sum across all constraints).
    pub costs: Vec<f64>,
    /// Per-step constraint violation flags.
    pub violations: Vec<bool>,
}

/// Compute safety metrics from episode data.
pub fn compute_srl_metrics(episodes: &[SrlEpisodeData]) -> SrlMetrics {
    if episodes.is_empty() {
        return SrlMetrics {
            constraint_violation_rate: 0.0,
            average_constraint_cost: 0.0,
            average_reward: 0.0,
            safety_adjusted_return: 0.0,
            cumulative_violations: 0.0,
            cost_rate: 0.0,
            num_episodes: 0,
            total_steps: 0,
        };
    }

    let num_episodes = episodes.len();
    let mut total_steps = 0usize;
    let mut total_violations = 0usize;
    let mut total_cost = 0.0;
    let mut total_reward = 0.0;
    let mut cumulative_violations = 0.0;

    for ep in episodes {
        let steps = ep.rewards.len();
        total_steps += steps;
        total_reward += ep.rewards.iter().sum::<f64>();
        total_cost += ep.costs.iter().sum::<f64>();
        let ep_violations = ep.violations.iter().filter(|&&v| v).count();
        total_violations += ep_violations;
        cumulative_violations += ep_violations as f64;
    }

    let cvr = if total_steps > 0 {
        total_violations as f64 / total_steps as f64
    } else {
        0.0
    };
    let avg_cost = if total_steps > 0 {
        total_cost / total_steps as f64
    } else {
        0.0
    };
    let avg_reward = if num_episodes > 0 {
        total_reward / num_episodes as f64
    } else {
        0.0
    };
    let safety_adjusted = avg_reward * (1.0 - cvr);
    let cost_rate = if num_episodes > 0 {
        total_violations as f64 / num_episodes as f64
    } else {
        0.0
    };

    SrlMetrics {
        constraint_violation_rate: cvr,
        average_constraint_cost: avg_cost,
        average_reward: avg_reward,
        safety_adjusted_return: safety_adjusted,
        cumulative_violations,
        cost_rate,
        num_episodes,
        total_steps,
    }
}

/// Compute a simple Pareto front from (reward, safety) tuples.
/// Safety score = 1.0 - violation_rate. Both higher is better.
pub fn compute_srl_pareto_front(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut pareto = Vec::new();
    let mut max_safety = f64::NEG_INFINITY;

    for &(reward, safety) in &sorted {
        if safety > max_safety {
            pareto.push((reward, safety));
            max_safety = safety;
        }
    }

    pareto
}

/// Build a full safety report from episode data.
pub fn build_srl_report(episodes: &[SrlEpisodeData]) -> SrlReport {
    let metrics = compute_srl_metrics(episodes);

    // Build per-episode (reward, safety) points for Pareto front
    let points: Vec<(f64, f64)> = episodes
        .iter()
        .map(|ep| {
            let reward: f64 = ep.rewards.iter().sum();
            let violations = ep.violations.iter().filter(|&&v| v).count();
            let steps = ep.rewards.len().max(1) as f64;
            let safety = 1.0 - violations as f64 / steps;
            (reward, safety)
        })
        .collect();

    let pareto_front = compute_srl_pareto_front(&points);

    SrlReport {
        metrics,
        pareto_front,
    }
}

#[cfg(test)]
mod tests;
