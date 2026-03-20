//! Robotics & Embodied AI Components.
//!
//! Sections: imitation learning, world models (RSSM/Dreamer), robot kinematics,
//! navigation/mapping, and sim-to-real transfer.
//!
//! Additional advanced robotics algorithms are in [`extensions`] and [`advanced`].

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use tenflowers_core::{Result, TensorError};

// ── Shared math primitives ───────────────────────────────────────────────────

#[inline]
pub(crate) fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp())
}

pub(crate) fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(f32::EPSILON);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

pub(crate) fn kaiming_uniform(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / fan_in as f64).sqrt() as f32;
    (0..fan_in * fan_out)
        .map(|_| {
            let u: f32 = rng.random();
            u * 2.0 * limit - limit
        })
        .collect()
}

pub(crate) fn dense_relu(weights: &[f32], bias: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|o| {
            let mut s = if o < bias.len() { bias[o] } else { 0.0 };
            for i in 0..in_dim {
                s += weights[o * in_dim + i] * input[i];
            }
            relu_f32(s)
        })
        .collect()
}

pub(crate) fn dense_linear(
    weights: &[f32],
    bias: &[f32],
    input: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|o| {
            let mut s = if o < bias.len() { bias[o] } else { 0.0 };
            for i in 0..in_dim {
                s += weights[o * in_dim + i] * input[i];
            }
            s
        })
        .collect()
}

pub(crate) fn normal_samples_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        i += 1;
        if i < n {
            out.push((r * theta.sin()) as f32);
            i += 1;
        }
    }
    out
}

// ═══════════════════ SECTION 1 — IMITATION LEARNING ════════════════════════

/// A single (obs, action, reward, done) demonstration tuple.
#[derive(Debug, Clone)]
pub struct DemoTransition {
    /// Observation vector at time t.
    pub obs: Vec<f32>,
    /// Action vector taken at time t.
    pub action: Vec<f32>,
    /// Scalar reward received.
    pub reward: f32,
    /// Whether this is a terminal transition.
    pub done: bool,
}
impl DemoTransition {
    /// Construct a new demonstration transition.
    pub fn new(obs: Vec<f32>, action: Vec<f32>, reward: f32, done: bool) -> Self {
        Self {
            obs,
            action,
            reward,
            done,
        }
    }
}

/// Fixed-capacity ring buffer for expert demonstration data.
#[derive(Debug, Clone)]
pub struct DemonstrationBuffer {
    capacity: usize,
    buffer: Vec<DemoTransition>,
    head: usize,
}
impl DemonstrationBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "DemonstrationBuffer::new",
                "capacity must be > 0",
            ));
        }
        Ok(Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            head: 0,
        })
    }
    /// Add a transition, overwriting the oldest entry if full.
    pub fn add(&mut self, demo: DemoTransition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(demo);
        } else {
            self.buffer[self.head] = demo;
        }
        self.head = (self.head + 1) % self.capacity;
    }
    /// Current number of stored transitions.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    /// Sample `n` random transitions from the buffer.
    pub fn sample_batch(&self, n: usize, rng: &mut StdRng) -> Result<Vec<DemoTransition>> {
        if n > self.buffer.len() {
            return Err(TensorError::invalid_argument_op(
                "DemonstrationBuffer::sample_batch",
                &format!(
                    "requested {} samples but buffer only has {}",
                    n,
                    self.buffer.len()
                ),
            ));
        }
        let indices: Vec<usize> = (0..n)
            .map(|_| (rng.random::<u64>() as usize) % self.buffer.len())
            .collect();
        Ok(indices.iter().map(|&i| self.buffer[i].clone()).collect())
    }
}

/// Behavioural Cloning: supervised MLP policy trained with CE or MSE loss.
#[derive(Debug, Clone)]
pub struct BehavioralCloning {
    obs_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    lr: f32,
    discrete: bool,
}
impl BehavioralCloning {
    /// Create a new behavioral cloning agent.
    ///
    /// If `discrete` is true cross-entropy loss is used, otherwise MSE.
    pub fn new(
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        lr: f32,
        discrete: bool,
        seed: u64,
    ) -> Result<Self> {
        if obs_dim == 0 || action_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::new",
                "dimensions must be > 0",
            ));
        }
        Ok(Self {
            obs_dim,
            action_dim,
            hidden_dim,
            w1: kaiming_uniform(obs_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w2: kaiming_uniform(hidden_dim, action_dim, seed.wrapping_add(1)),
            b2: vec![0.0_f32; action_dim],
            lr,
            discrete,
        })
    }
    /// Forward pass: obs → action logits.
    pub fn forward(&self, obs: &[f32]) -> Vec<f32> {
        let h = dense_relu(&self.w1, &self.b1, obs, self.hidden_dim);
        dense_linear(&self.w2, &self.b2, &h, self.action_dim)
    }
    /// One gradient step on a single (obs, action) pair; returns scalar loss.
    pub fn train_step(&mut self, obs: &[f32], action_target: &[f32]) -> Result<f64> {
        if obs.len() != self.obs_dim {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::train_step",
                "obs length mismatch",
            ));
        }
        if action_target.len() != self.action_dim {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::train_step",
                "action_target length mismatch",
            ));
        }
        let mut logits = self.forward(obs);
        let loss = if self.discrete {
            softmax_inplace(&mut logits);
            let ce = action_target
                .iter()
                .zip(logits.iter())
                .map(|(&t, &p)| {
                    if t > 0.0 {
                        -(p.max(f32::EPSILON) as f64).ln() * t as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            let lr = self.lr as f64;
            for i in 0..self.action_dim {
                self.b2[i] -= ((logits[i] - action_target[i]) as f64 * lr) as f32;
            }
            ce
        } else {
            let mse = logits
                .iter()
                .zip(action_target.iter())
                .map(|(&p, &t)| {
                    let e = (p - t) as f64;
                    e * e
                })
                .sum::<f64>()
                / self.action_dim as f64;
            let lr = self.lr as f64;
            for i in 0..self.action_dim {
                let g = 2.0 * (logits[i] - action_target[i]) as f64 / self.action_dim as f64;
                self.b2[i] -= (g * lr) as f32;
            }
            mse
        };
        Ok(loss)
    }
}

/// DAgger (Dataset Aggregation) policy — linear beta schedule controlling expert mixing.
#[derive(Debug, Clone)]
pub struct DaggerPolicy {
    dataset: Vec<(Vec<f32>, Vec<f32>)>,
    beta_initial: f64,
    beta_final: f64,
    total_iterations: usize,
}
impl DaggerPolicy {
    /// Create a new DAgger policy with a linear beta schedule from `beta_initial` to `beta_final`.
    pub fn new(beta_initial: f64, beta_final: f64, total_iterations: usize) -> Self {
        Self {
            dataset: Vec::new(),
            beta_initial,
            beta_final,
            total_iterations: total_iterations.max(1),
        }
    }
    /// Aggregate new (obs, expert_action) pairs into the training dataset.
    pub fn aggregate(&mut self, policy_obs: Vec<Vec<f32>>, expert_actions: Vec<Vec<f32>>) {
        for (obs, act) in policy_obs.into_iter().zip(expert_actions) {
            self.dataset.push((obs, act));
        }
    }
    /// Linear decay: β(t) = β₀ − t/T·(β₀ − β_final).
    pub fn beta_schedule(&self, t: usize) -> f64 {
        let frac = (t as f64) / (self.total_iterations as f64);
        let beta = self.beta_initial - frac * (self.beta_initial - self.beta_final);
        beta.clamp(
            self.beta_final.min(self.beta_initial),
            self.beta_initial.max(self.beta_final),
        )
    }
    /// Number of pairs in the aggregated dataset.
    pub fn dataset_size(&self) -> usize {
        self.dataset.len()
    }
}

/// GAIL discriminator D(s,a) ∈ \[0,1\] — 2-layer MLP.
#[derive(Debug, Clone)]
pub struct GailDiscriminator {
    input_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    lr: f32,
}
impl GailDiscriminator {
    /// Create a GAIL discriminator with given input/hidden dimensions.
    pub fn new(input_dim: usize, hidden_dim: usize, lr: f32, seed: u64) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "GailDiscriminator::new",
                "dimensions must be > 0",
            ));
        }
        Ok(Self {
            input_dim,
            hidden_dim,
            lr,
            w1: kaiming_uniform(input_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w2: kaiming_uniform(hidden_dim, 1, seed.wrapping_add(1)),
            b2: vec![0.0_f32; 1],
        })
    }
    fn score(&self, sa: &[f32]) -> f32 {
        let h = dense_relu(&self.w1, &self.b1, sa, self.hidden_dim);
        sigmoid_f32(dense_linear(&self.w2, &self.b2, &h, 1)[0])
    }
    /// Predict D(obs, action) ∈ \[0,1\].
    pub fn predict(&self, obs: &[f32], action: &[f32]) -> f32 {
        let sa: Vec<f32> = obs.iter().chain(action.iter()).copied().collect();
        self.score(&sa)
    }
    /// BCE: −E_expert[log D] − E_policy[log(1−D)].
    pub fn discriminator_loss(
        &mut self,
        expert_batch: &[DemoTransition],
        policy_batch: &[DemoTransition],
    ) -> f64 {
        if expert_batch.is_empty() && policy_batch.is_empty() {
            return 0.0;
        }
        let eps = f64::EPSILON;
        let mut loss = 0.0_f64;
        for t in expert_batch {
            loss += -(self.predict(&t.obs, &t.action) as f64).max(eps).ln();
        }
        for t in policy_batch {
            loss += -((1.0 - self.predict(&t.obs, &t.action) as f64).max(eps)).ln();
        }
        loss / (expert_batch.len() + policy_batch.len()).max(1) as f64
    }
}

/// Offline RL agent — CQL-style conservative Q-learning with OOD penalty.
#[derive(Debug, Clone)]
pub struct OfflineRlAgent {
    obs_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    alpha: f32,
}
impl OfflineRlAgent {
    /// Create a new CQL-style offline RL agent.
    pub fn new(
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        alpha: f32,
        seed: u64,
    ) -> Result<Self> {
        if obs_dim == 0 || action_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "OfflineRlAgent::new",
                "dimensions must be > 0",
            ));
        }
        Ok(Self {
            obs_dim,
            action_dim,
            hidden_dim,
            alpha,
            w1: kaiming_uniform(obs_dim + action_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w2: kaiming_uniform(hidden_dim, 1, seed.wrapping_add(1)),
            b2: vec![0.0_f32; 1],
        })
    }
    /// Q(obs, action) forward pass.
    pub fn q_value(&self, obs: &[f32], action: &[f32]) -> f32 {
        let sa: Vec<f32> = obs.iter().chain(action.iter()).copied().collect();
        let h = dense_relu(&self.w1, &self.b1, &sa, self.hidden_dim);
        dense_linear(&self.w2, &self.b2, &h, 1)[0]
    }
    /// CQL penalty: α·(logsumexp Q(s,a_rand) − mean Q(s,a_policy)).
    pub fn cql_loss(
        &self,
        obs: &[f32],
        policy_actions: &[Vec<f32>],
        random_actions: &[Vec<f32>],
    ) -> Result<f64> {
        if policy_actions.is_empty() || random_actions.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "OfflineRlAgent::cql_loss",
                "actions must be non-empty",
            ));
        }
        let rqs: Vec<f64> = random_actions
            .iter()
            .map(|a| self.q_value(obs, a) as f64)
            .collect();
        let max_q = rqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let logsumexp = max_q
            + rqs
                .iter()
                .map(|&q| (q - max_q).exp())
                .sum::<f64>()
                .max(f64::EPSILON)
                .ln();
        let policy_q = policy_actions
            .iter()
            .map(|a| self.q_value(obs, a) as f64)
            .sum::<f64>()
            / policy_actions.len() as f64;
        Ok(((logsumexp - policy_q) * self.alpha as f64).max(0.0))
    }
}

// ═══════════════════ SECTION 2 — WORLD MODELS ══════════════════════════════

/// RSSM (Dreamer): det. hidden h_t = GRU(h,z,a), stochastic z_t ~ q(z|h,o).
#[derive(Debug, Clone)]
pub struct RecurrentStateSpaceModel {
    /// Dimensionality of the deterministic recurrent hidden state.
    pub hidden_dim: usize,
    /// Dimensionality of the stochastic latent state.
    pub latent_dim: usize,
    /// Dimensionality of the observation.
    pub obs_dim: usize,
    /// Dimensionality of the action.
    pub action_dim: usize,
    gru_w: Vec<f32>,
    gru_b: Vec<f32>,
    post_w: Vec<f32>,
    post_b: Vec<f32>,
    prior_w: Vec<f32>,
    prior_b: Vec<f32>,
}
impl RecurrentStateSpaceModel {
    /// Create a new RSSM with the given dimensions.
    pub fn new(
        hidden_dim: usize,
        latent_dim: usize,
        obs_dim: usize,
        action_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if hidden_dim == 0 || latent_dim == 0 || obs_dim == 0 || action_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "RecurrentStateSpaceModel::new",
                "all dims must be > 0",
            ));
        }
        let gru_in = hidden_dim + latent_dim + action_dim;
        let post_in = hidden_dim + obs_dim;
        Ok(Self {
            hidden_dim,
            latent_dim,
            obs_dim,
            action_dim,
            gru_w: kaiming_uniform(gru_in, hidden_dim, seed),
            gru_b: vec![0.0_f32; hidden_dim],
            post_w: kaiming_uniform(post_in, 2 * latent_dim, seed.wrapping_add(1)),
            post_b: vec![0.0_f32; 2 * latent_dim],
            prior_w: kaiming_uniform(hidden_dim, 2 * latent_dim, seed.wrapping_add(2)),
            prior_b: vec![0.0_f32; 2 * latent_dim],
        })
    }
    /// Compute the deterministic state update: h_t = tanh(W·\[h,z,a\]).
    pub fn deterministic_update(&self, h: &[f32], z: &[f32], action: &[f32]) -> Vec<f32> {
        let combined: Vec<f32> = h.iter().chain(z).chain(action).copied().collect();
        dense_linear(&self.gru_w, &self.gru_b, &combined, self.hidden_dim)
            .iter()
            .map(|&x| x.tanh())
            .collect()
    }
    /// Posterior q(z|h,o): returns (mu, log_var).
    pub fn posterior(&self, h: &[f32], obs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let input: Vec<f32> = h.iter().chain(obs).copied().collect();
        let out = dense_linear(&self.post_w, &self.post_b, &input, 2 * self.latent_dim);
        (
            out[..self.latent_dim].to_vec(),
            out[self.latent_dim..].to_vec(),
        )
    }
    /// Prior p(z|h): returns (mu, log_var).
    pub fn prior(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let out = dense_linear(&self.prior_w, &self.prior_b, h, 2 * self.latent_dim);
        (
            out[..self.latent_dim].to_vec(),
            out[self.latent_dim..].to_vec(),
        )
    }
    /// Sample z ~ N(mu, exp(0.5·log_var)) via reparameterization.
    pub fn sample_latent(&self, mu: &[f32], log_var: &[f32], seed: u64) -> Vec<f32> {
        let eps = normal_samples_f32(mu.len(), seed);
        mu.iter()
            .zip(log_var)
            .zip(&eps)
            .map(|((&m, &lv), &e)| m + (0.5 * lv).exp() * e)
            .collect()
    }
}

/// Observation encoder: obs → (μ, log σ²) in latent space.
#[derive(Debug, Clone)]
pub struct WorldModelEncoder {
    obs_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w_mu: Vec<f32>,
    b_mu: Vec<f32>,
    w_lv: Vec<f32>,
    b_lv: Vec<f32>,
}
impl WorldModelEncoder {
    /// Create a new world model encoder.
    pub fn new(obs_dim: usize, hidden_dim: usize, latent_dim: usize, seed: u64) -> Result<Self> {
        if obs_dim == 0 || hidden_dim == 0 || latent_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "WorldModelEncoder::new",
                "dims must be > 0",
            ));
        }
        Ok(Self {
            obs_dim,
            hidden_dim,
            latent_dim,
            w1: kaiming_uniform(obs_dim, hidden_dim, seed),
            b1: vec![0.0_f32; hidden_dim],
            w_mu: kaiming_uniform(hidden_dim, latent_dim, seed.wrapping_add(1)),
            b_mu: vec![0.0_f32; latent_dim],
            w_lv: kaiming_uniform(hidden_dim, latent_dim, seed.wrapping_add(2)),
            b_lv: vec![0.0_f32; latent_dim],
        })
    }
    /// Encode obs → (mu, log_var).
    pub fn encode(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let h = dense_relu(&self.w1, &self.b1, obs, self.hidden_dim);
        (
            dense_linear(&self.w_mu, &self.b_mu, &h, self.latent_dim),
            dense_linear(&self.w_lv, &self.b_lv, &h, self.latent_dim),
        )
    }
}

/// Observation decoder: (h, z) → reconstructed observation.
#[derive(Debug, Clone)]
pub struct WorldModelDecoder {
    hidden_dim: usize,
    latent_dim: usize,
    obs_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
}
impl WorldModelDecoder {
    /// Create a new world model decoder.
    pub fn new(hidden_dim: usize, latent_dim: usize, obs_dim: usize, seed: u64) -> Result<Self> {
        if hidden_dim == 0 || latent_dim == 0 || obs_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "WorldModelDecoder::new",
                "dims must be > 0",
            ));
        }
        let inp = hidden_dim + latent_dim;
        let inter = (inp + obs_dim).max(8);
        Ok(Self {
            hidden_dim,
            latent_dim,
            obs_dim,
            w1: kaiming_uniform(inp, inter, seed),
            b1: vec![0.0_f32; inter],
            w_out: kaiming_uniform(inter, obs_dim, seed.wrapping_add(1)),
            b_out: vec![0.0_f32; obs_dim],
        })
    }
    /// Decode (h, z) → observation.
    pub fn decode(&self, h: &[f32], z: &[f32]) -> Vec<f32> {
        let input: Vec<f32> = h.iter().chain(z).copied().collect();
        let inter = (self.hidden_dim + self.latent_dim + self.obs_dim).max(8);
        let hidden = dense_relu(&self.w1, &self.b1, &input, inter);
        dense_linear(&self.w_out, &self.b_out, &hidden, self.obs_dim)
    }
}

/// Reward predictor: (h, z) → scalar reward.
#[derive(Debug, Clone)]
pub struct RewardPredictor {
    hidden_dim: usize,
    latent_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl RewardPredictor {
    /// Create a new reward predictor.
    pub fn new(hidden_dim: usize, latent_dim: usize, seed: u64) -> Result<Self> {
        if hidden_dim == 0 || latent_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "RewardPredictor::new",
                "dims must be > 0",
            ));
        }
        let inp = hidden_dim + latent_dim;
        let inter = inp.max(8);
        Ok(Self {
            hidden_dim,
            latent_dim,
            w1: kaiming_uniform(inp, inter, seed),
            b1: vec![0.0_f32; inter],
            w2: kaiming_uniform(inter, 1, seed.wrapping_add(1)),
            b2: vec![0.0_f32; 1],
        })
    }
    /// Predict scalar reward from (h, z).
    pub fn predict(&self, h: &[f32], z: &[f32]) -> f32 {
        let inp: Vec<f32> = h.iter().chain(z).copied().collect();
        let inter = (self.hidden_dim + self.latent_dim).max(8);
        dense_linear(
            &self.w2,
            &self.b2,
            &dense_relu(&self.w1, &self.b1, &inp, inter),
            1,
        )[0]
    }
}

/// Dreamer trainer — reconstruction + KL + reward prediction losses.
#[derive(Debug, Clone)]
pub struct DreamerTrainer {
    /// Observation encoder.
    pub encoder: WorldModelEncoder,
    /// Observation decoder.
    pub decoder: WorldModelDecoder,
    /// Recurrent state space model.
    pub rssm: RecurrentStateSpaceModel,
    /// Scalar reward predictor.
    pub reward_predictor: RewardPredictor,
    /// Weight for the KL divergence term.
    pub kl_weight: f32,
    /// Weight for the reward prediction term.
    pub reward_weight: f32,
}
impl DreamerTrainer {
    /// Create a full Dreamer world model trainer.
    pub fn new(
        obs_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        action_dim: usize,
        kl_weight: f32,
        reward_weight: f32,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            encoder: WorldModelEncoder::new(obs_dim, hidden_dim, latent_dim, seed)?,
            decoder: WorldModelDecoder::new(
                hidden_dim,
                latent_dim,
                obs_dim,
                seed.wrapping_add(10),
            )?,
            rssm: RecurrentStateSpaceModel::new(
                hidden_dim,
                latent_dim,
                obs_dim,
                action_dim,
                seed.wrapping_add(20),
            )?,
            reward_predictor: RewardPredictor::new(hidden_dim, latent_dim, seed.wrapping_add(30))?,
            kl_weight,
            reward_weight,
        })
    }
    /// Returns (total, recon, kl, reward) losses.
    pub fn world_model_loss(
        &self,
        obs: &[f32],
        h: &[f32],
        _action: &[f32],
        actual_reward: f32,
        seed: u64,
    ) -> Result<(f64, f64, f64, f64)> {
        let (mu_post, lv_post) = self.encoder.encode(obs);
        let z = self.rssm.sample_latent(&mu_post, &lv_post, seed);
        let recon = self.decoder.decode(h, &z);
        let recon_loss: f64 = obs
            .iter()
            .zip(&recon)
            .map(|(&o, &r)| {
                let e = (o - r) as f64;
                e * e
            })
            .sum::<f64>()
            / obs.len() as f64;
        let kl: f64 = mu_post
            .iter()
            .zip(&lv_post)
            .map(|(&mu, &lv)| {
                let var = (lv as f64).exp();
                0.5 * (var + (mu as f64).powi(2) - 1.0 - lv as f64)
            })
            .sum::<f64>();
        let kl_loss = kl.max(0.0);
        let reward_loss =
            (self.reward_predictor.predict(h, &z) as f64 - actual_reward as f64).powi(2);
        let total =
            recon_loss + self.kl_weight as f64 * kl_loss + self.reward_weight as f64 * reward_loss;
        Ok((total, recon_loss, kl_loss, reward_loss))
    }
}

// ═══════════════════ SECTION 3 — MANIPULATION & PLANNING ═══════════════════

/// Denavit-Hartenberg parameters for a single joint.
#[derive(Debug, Clone)]
pub struct DhParam {
    /// Link length along previous z-axis.
    pub a: f64,
    /// Link offset along current z-axis.
    pub d: f64,
    /// Twist angle between z-axes.
    pub alpha: f64,
    /// Joint angle (variable for revolute joints).
    pub theta: f64,
}
impl DhParam {
    /// Construct DH parameters for one joint.
    pub fn new(a: f64, d: f64, alpha: f64, theta: f64) -> Self {
        Self { a, d, alpha, theta }
    }
    fn transform_matrix(&self, q: f64) -> [f64; 16] {
        let theta = self.theta + q;
        let (ct, st) = (theta.cos(), theta.sin());
        let (ca, sa) = (self.alpha.cos(), self.alpha.sin());
        let (a, d) = (self.a, self.d);
        [
            ct,
            -st,
            0.0,
            a,
            st * ca,
            ct * ca,
            -sa,
            -sa * d,
            st * sa,
            ct * sa,
            ca,
            ca * d,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }
}

fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut c = [0.0_f64; 16];
    for row in 0..4 {
        for col in 0..4 {
            for k in 0..4 {
                c[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
            }
        }
    }
    c
}

/// Forward kinematics for an N-DOF serial manipulator (DH parameters).
#[derive(Debug, Clone)]
pub struct RobotKinematics;
impl RobotKinematics {
    /// Compute end-effector position [x, y, z] via DH forward kinematics.
    pub fn forward_kinematics(params: &[DhParam], joint_angles: &[f64]) -> Result<[f64; 3]> {
        if params.len() != joint_angles.len() {
            return Err(TensorError::invalid_argument_op(
                "RobotKinematics::forward_kinematics",
                "params and joint_angles must have same length",
            ));
        }
        let mut t = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0_f64,
        ];
        for (param, &q) in params.iter().zip(joint_angles) {
            t = mat4_mul(&t, &param.transform_matrix(q));
        }
        Ok([t[3], t[7], t[11]])
    }
}

/// Dynamic Movement Primitive — parameterised motion via canonical system + RBF forcing term.
#[derive(Debug, Clone)]
pub struct MotionPrimitive {
    /// Time constant of the canonical system.
    pub alpha_x: f64,
    /// Number of RBF basis functions.
    pub n_basis: usize,
    /// Learnable weights for the forcing term.
    pub weights: Vec<f64>,
    centres: Vec<f64>,
    widths: Vec<f64>,
}
impl MotionPrimitive {
    /// Create a DMP with `n_basis` equally-spaced Gaussian basis functions.
    pub fn new(alpha_x: f64, n_basis: usize) -> Result<Self> {
        if n_basis == 0 {
            return Err(TensorError::invalid_argument_op(
                "MotionPrimitive::new",
                "n_basis must be > 0",
            ));
        }
        let centres: Vec<f64> = (0..n_basis)
            .map(|i| 0.01 + 0.99 * i as f64 / (n_basis - 1).max(1) as f64)
            .collect();
        let width = if n_basis > 1 {
            0.5 / (centres[1] - centres[0]).powi(2).max(f64::EPSILON)
        } else {
            1.0
        };
        Ok(Self {
            alpha_x,
            n_basis,
            weights: vec![0.0_f64; n_basis],
            centres,
            widths: vec![width; n_basis],
        })
    }
    #[inline]
    fn phase(&self, tau: f64, t: f64) -> f64 {
        (-self.alpha_x * t / tau.max(f64::EPSILON)).exp()
    }
    /// Execute the DMP at time `t` for a given goal and start position.
    pub fn execute(&self, tau: f64, goal: f64, start: f64, t: f64) -> f64 {
        let x = self.phase(tau, t);
        let psi: Vec<f64> = self
            .centres
            .iter()
            .zip(&self.widths)
            .map(|(&c, &h)| (-(x - c).powi(2) * h).exp())
            .collect();
        let psi_sum: f64 = psi.iter().sum();
        let f = if psi_sum.abs() > 1e-15 {
            psi.iter()
                .zip(&self.weights)
                .map(|(&p, &w)| p * w)
                .sum::<f64>()
                / psi_sum
                * x
                * (goal - start)
        } else {
            0.0
        };
        start + (goal - start) * (1.0 - x) + f * (t / tau.max(f64::EPSILON))
    }
}

/// Grasp quality metric — ε-quality approximation via contact normals.
#[derive(Debug, Clone)]
pub struct GraspQuality;
impl GraspQuality {
    /// Compute the ε-quality of a grasp from contact normals.
    pub fn epsilon_quality(contact_normals: &[[f64; 3]]) -> f64 {
        if contact_normals.len() < 2 {
            return 0.0;
        }
        let n = contact_normals.len();
        let mut wtwij = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                wtwij[i * n + j] = contact_normals[i]
                    .iter()
                    .zip(&contact_normals[j])
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
        wtwij.iter().map(|&v| v * v).sum::<f64>().sqrt() / n as f64
    }
    /// Check whether the grasp satisfies force closure (balanced normal forces).
    pub fn is_force_closed(contact_normals: &[[f64; 3]]) -> bool {
        if contact_normals.len() < 3 {
            return false;
        }
        let sum = contact_normals.iter().fold([0.0_f64; 3], |mut acc, n| {
            acc[0] += n[0];
            acc[1] += n[1];
            acc[2] += n[2];
            acc
        });
        (sum[0].powi(2) + sum[1].powi(2) + sum[2].powi(2)).sqrt()
            < 0.5 * contact_normals.len() as f64
    }
}

/// Symbolic object in the task planning space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    /// Human-readable object name.
    pub name: String,
}

/// Symbolic predicate with arguments (e.g. ON(block_A, table)).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate {
    /// Predicate name (e.g. "ON", "HOLDING").
    pub name: String,
    /// Argument names.
    pub args: Vec<String>,
}

/// Task and Motion Planning: symbolic task layer + motion feasibility checks.
#[derive(Debug, Clone)]
pub struct TaskAndMotionPlanning {
    /// Objects in the scene.
    pub objects: Vec<Symbol>,
    /// Current symbolic state (set of true predicates).
    pub state: Vec<Predicate>,
    blocked_predicates: Vec<String>,
}
impl TaskAndMotionPlanning {
    /// Create a TAMP instance with initial objects and symbolic state.
    pub fn new(objects: Vec<Symbol>, initial_state: Vec<Predicate>) -> Self {
        Self {
            objects,
            state: initial_state,
            blocked_predicates: Vec::new(),
        }
    }
    /// Mark a predicate as infeasible (motion planning failure).
    pub fn block_predicate(&mut self, predicate_name: &str) {
        self.blocked_predicates.push(predicate_name.to_string());
    }
    /// Check if a predicate is feasible (motion planning permits it).
    pub fn is_feasible(&self, predicate: &Predicate) -> bool {
        !self.blocked_predicates.contains(&predicate.name)
    }
    /// Check whether a predicate currently holds in the symbolic state.
    pub fn holds(&self, predicate: &Predicate) -> bool {
        self.state.contains(predicate)
    }
    /// Apply a symbolic action: add effects are added, delete effects are removed.
    pub fn apply_action(
        &mut self,
        add_effects: Vec<Predicate>,
        del_effects: Vec<Predicate>,
    ) -> bool {
        for p in &add_effects {
            if !self.is_feasible(p) {
                return false;
            }
        }
        for p in &del_effects {
            self.state.retain(|s| s != p);
        }
        for p in add_effects {
            if !self.state.contains(&p) {
                self.state.push(p);
            }
        }
        true
    }
}

// ═══════════════════ SECTION 4 — NAVIGATION & MAPPING ══════════════════════

/// 2D occupancy grid map using log-odds representation.
#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    log_odds: Vec<f32>,
    l_occ: f32,
    l_free: f32,
    l_min: f32,
    l_max: f32,
}
impl OccupancyGrid {
    /// Create a new occupancy grid initialised to slightly free.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(TensorError::invalid_argument_op(
                "OccupancyGrid::new",
                "width and height must be > 0",
            ));
        }
        Ok(Self {
            width,
            height,
            log_odds: vec![-0.01_f32; width * height],
            l_occ: 0.85,
            l_free: -0.4,
            l_min: -5.0,
            l_max: 5.0,
        })
    }
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
    /// Update log-odds for cell (x,y) based on an occupancy measurement.
    pub fn update(&mut self, x: usize, y: usize, occupied: bool) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = self.idx(x, y);
        self.log_odds[i] = (self.log_odds[i] + if occupied { self.l_occ } else { self.l_free })
            .clamp(self.l_min, self.l_max);
    }
    /// Return true if cell (x,y) is currently considered free.
    pub fn is_free(&self, x: usize, y: usize) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.log_odds[self.idx(x, y)] < 0.0
    }
    /// Return the occupancy probability P(occupied) for cell (x,y).
    pub fn probability(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.5;
        }
        1.0 / (1.0 + (-self.log_odds[self.idx(x, y)]).exp())
    }
    /// Trace a Bresenham line from `origin` to `endpoint`, returning intermediate cells.
    pub fn bresenham_ray(
        &self,
        origin: (usize, usize),
        endpoint: (usize, usize),
    ) -> Vec<(usize, usize)> {
        let (mut x0, mut y0) = (origin.0 as i64, origin.1 as i64);
        let (x1, y1) = (endpoint.0 as i64, endpoint.1 as i64);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i64 = if x0 < x1 { 1 } else { -1 };
        let sy: i64 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut cells = Vec::new();
        loop {
            if x0 == x1 && y0 == y1 {
                break;
            }
            cells.push((x0 as usize, y0 as usize));
            let e2 = 2 * err;
            if e2 >= dy {
                if x0 == x1 {
                    break;
                }
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                if y0 == y1 {
                    break;
                }
                err += dx;
                y0 += sy;
            }
        }
        cells
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AStarNode {
    pos: (usize, usize),
    f: f32,
}
impl Eq for AStarNode {}
impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.partial_cmp(&self.f).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* path planner on a 2D occupancy grid.
#[derive(Debug, Clone)]
pub struct AStarPlanner {
    /// Whether diagonal moves are allowed.
    pub allow_diagonal: bool,
}
impl AStarPlanner {
    /// Create a new A* planner.
    pub fn new(allow_diagonal: bool) -> Self {
        Self { allow_diagonal }
    }
    fn heuristic(a: (usize, usize), b: (usize, usize)) -> f32 {
        (a.0 as f32 - b.0 as f32).abs() + (a.1 as f32 - b.1 as f32).abs()
    }
    fn neighbours(&self, pos: (usize, usize), w: usize, h: usize) -> Vec<((usize, usize), f32)> {
        let (x, y) = pos;
        let mut nb = Vec::with_capacity(8);
        for &(dx, dy) in &[(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
            let nx = x as i64 + dx;
            let ny = y as i64 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                nb.push(((nx as usize, ny as usize), 1.0_f32));
            }
        }
        if self.allow_diagonal {
            for &(dx, dy) in &[(1i64, 1i64), (-1, 1), (1, -1), (-1, -1i64)] {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    nb.push(((nx as usize, ny as usize), std::f32::consts::SQRT_2));
                }
            }
        }
        nb
    }
    /// Plan a path from `start` to `goal` on the occupancy grid.
    pub fn plan(
        &self,
        grid: &OccupancyGrid,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        if !grid.is_free(start.0, start.1) || !grid.is_free(goal.0, goal.1) {
            return None;
        }
        let mut open: BinaryHeap<AStarNode> = BinaryHeap::new();
        let mut g_score: HashMap<(usize, usize), f32> = HashMap::new();
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        g_score.insert(start, 0.0);
        open.push(AStarNode {
            pos: start,
            f: Self::heuristic(start, goal),
        });
        while let Some(AStarNode { pos: current, .. }) = open.pop() {
            if current == goal {
                let mut path = vec![current];
                let mut node = current;
                while let Some(&prev) = came_from.get(&node) {
                    path.push(prev);
                    node = prev;
                }
                path.reverse();
                return Some(path);
            }
            let g_cur = *g_score.get(&current).unwrap_or(&f32::INFINITY);
            for (nb, cost) in self.neighbours(current, grid.width, grid.height) {
                if !grid.is_free(nb.0, nb.1) {
                    continue;
                }
                let tg = g_cur + cost;
                if tg < *g_score.get(&nb).unwrap_or(&f32::INFINITY) {
                    g_score.insert(nb, tg);
                    came_from.insert(nb, current);
                    open.push(AStarNode {
                        pos: nb,
                        f: tg + Self::heuristic(nb, goal),
                    });
                }
            }
        }
        None
    }
}

/// Particle for Monte Carlo Localization.
#[derive(Debug, Clone)]
pub struct Particle {
    /// X position.
    pub x: f64,
    /// Y position.
    pub y: f64,
    /// Heading angle in radians.
    pub theta: f64,
    /// Normalized particle weight.
    pub weight: f64,
}

/// Monte Carlo Localization particle filter.
#[derive(Debug, Clone)]
pub struct ParticleFilter {
    /// All particles in the current belief distribution.
    pub particles: Vec<Particle>,
    /// Translational motion noise standard deviation.
    pub motion_noise_trans: f64,
    /// Rotational motion noise standard deviation.
    pub motion_noise_rot: f64,
}
impl ParticleFilter {
    /// Create a new particle filter with `n` particles uniformly distributed in the given ranges.
    pub fn new(n: usize, x_range: (f64, f64), y_range: (f64, f64), seed: u64) -> Result<Self> {
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "ParticleFilter::new",
                "n must be > 0",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let particles = (0..n)
            .map(|_| Particle {
                x: x_range.0 + rng.random::<f64>() * (x_range.1 - x_range.0),
                y: y_range.0 + rng.random::<f64>() * (y_range.1 - y_range.0),
                theta: rng.random::<f64>() * std::f64::consts::TAU,
                weight: 1.0 / n as f64,
            })
            .collect();
        Ok(Self {
            particles,
            motion_noise_trans: 0.1,
            motion_noise_rot: 0.05,
        })
    }
    /// Propagate particles through the motion model with noise.
    pub fn predict(&mut self, motion: (f64, f64, f64), rng: &mut StdRng) {
        let (dt, dr1, dr2) = motion;
        let eps = normal_samples_f32(self.particles.len() * 3, rng.random::<u64>());
        for (i, p) in self.particles.iter_mut().enumerate() {
            let dr1n = dr1 + self.motion_noise_rot * eps[i * 3 + 1] as f64;
            let dtn = dt + self.motion_noise_trans * eps[i * 3] as f64;
            let dr2n = dr2 + self.motion_noise_rot * eps[i * 3 + 2] as f64;
            p.x += dtn * (p.theta + dr1n).cos();
            p.y += dtn * (p.theta + dr1n).sin();
            p.theta += dr1n + dr2n;
        }
    }
    /// Update particle weights based on sensor measurements.
    pub fn update(&mut self, measurements: &[(f64, f64)]) {
        if measurements.is_empty() {
            return;
        }
        let sigma = 0.3_f64;
        for p in self.particles.iter_mut() {
            p.weight *= measurements
                .iter()
                .map(|&(exp, obs)| {
                    let e = (obs - exp) / sigma;
                    (-0.5 * e * e).exp()
                })
                .product::<f64>()
                .max(1e-300);
        }
        let sum: f64 = self.particles.iter().map(|p| p.weight).sum();
        if sum > 0.0 {
            for p in self.particles.iter_mut() {
                p.weight /= sum;
            }
        }
    }
    /// Systematic resampling of particles.
    pub fn resample(&mut self, rng: &mut StdRng) {
        let n = self.particles.len();
        let mut cumsum = Vec::with_capacity(n);
        let mut acc = 0.0_f64;
        for p in &self.particles {
            acc += p.weight;
            cumsum.push(acc);
        }
        let step = 1.0 / n as f64;
        let start: f64 = rng.random::<f64>() * step;
        let mut new_particles = Vec::with_capacity(n);
        let mut j = 0_usize;
        for i in 0..n {
            let u = start + i as f64 * step;
            while j < n - 1 && cumsum[j] < u {
                j += 1;
            }
            let mut p = self.particles[j].clone();
            p.weight = step;
            new_particles.push(p);
        }
        self.particles = new_particles;
    }
    /// Compute the weighted mean pose (x, y, theta).
    pub fn mean_pose(&self) -> (f64, f64, f64) {
        let (mut x, mut y, mut cs, mut ss) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for p in &self.particles {
            x += p.x * p.weight;
            y += p.y * p.weight;
            cs += p.theta.cos() * p.weight;
            ss += p.theta.sin() * p.weight;
        }
        (x, y, ss.atan2(cs))
    }
}

/// Neural map builder: per-cell MLP embedding of sensor observations.
#[derive(Debug, Clone)]
pub struct NeuralMapBuilder {
    /// Dimensionality of per-cell sensor observation.
    pub cell_obs_dim: usize,
    /// Dimensionality of the cell embedding.
    pub embed_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl NeuralMapBuilder {
    /// Create a new neural map builder.
    pub fn new(cell_obs_dim: usize, embed_dim: usize, seed: u64) -> Result<Self> {
        if cell_obs_dim == 0 || embed_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "NeuralMapBuilder::new",
                "dims must be > 0",
            ));
        }
        let hidden = (cell_obs_dim + embed_dim).max(4);
        Ok(Self {
            cell_obs_dim,
            embed_dim,
            w1: kaiming_uniform(cell_obs_dim, hidden, seed),
            b1: vec![0.0_f32; hidden],
            w2: kaiming_uniform(hidden, embed_dim, seed.wrapping_add(1)),
            b2: vec![0.0_f32; embed_dim],
        })
    }
    /// Embed a single cell observation.
    pub fn embed_cell(&self, obs: &[f32]) -> Vec<f32> {
        let hidden = (self.cell_obs_dim + self.embed_dim).max(4);
        dense_linear(
            &self.w2,
            &self.b2,
            &dense_relu(&self.w1, &self.b1, obs, hidden),
            self.embed_dim,
        )
    }
    /// Build a map by embedding each cell observation in the sequence.
    pub fn build_map(&self, cell_observations: &[Vec<f32>]) -> Vec<Vec<f32>> {
        cell_observations
            .iter()
            .map(|obs| self.embed_cell(obs))
            .collect()
    }
}

/// Artificial potential field navigator: attractive to goal + repulsive from obstacles.
#[derive(Debug, Clone)]
pub struct PotentialFieldNavigator {
    /// Attractive gain toward the goal.
    pub k_att: f64,
    /// Repulsive gain from obstacles.
    pub k_rep: f64,
    /// Influence radius of obstacles.
    pub influence_radius: f64,
}
impl PotentialFieldNavigator {
    /// Create a new potential field navigator.
    pub fn new(k_att: f64, k_rep: f64, influence_radius: f64) -> Self {
        Self {
            k_att,
            k_rep,
            influence_radius,
        }
    }
    /// Compute the combined attractive + repulsive force at `pos`.
    pub fn compute_force(&self, pos: [f64; 2], goal: [f64; 2], obstacles: &[[f64; 2]]) -> [f64; 2] {
        let mut f = [
            self.k_att * (goal[0] - pos[0]),
            self.k_att * (goal[1] - pos[1]),
        ];
        for obs in obstacles {
            let dx = pos[0] - obs[0];
            let dy = pos[1] - obs[1];
            let dist = (dx * dx + dy * dy).sqrt().max(f64::EPSILON);
            if dist < self.influence_radius {
                let mag = self.k_rep * (1.0 / dist - 1.0 / self.influence_radius) / (dist * dist);
                f[0] += mag * dx / dist;
                f[1] += mag * dy / dist;
            }
        }
        f
    }
}
