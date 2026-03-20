//! # Inverse Reinforcement Learning & Imitation Learning
//!
//! Production-grade implementations of core IRL and imitation learning algorithms:
//!
//! - **MaxEntIrl** — Maximum Entropy IRL (Ziebart et al. 2008)
//! - **GailDiscriminator** / **GailTrainer** — Generative Adversarial Imitation Learning (Ho & Ermon 2016)
//! - **AirlModel** — Adversarial IRL with disentangled rewards (Fu et al. 2018)
//! - **BehavioralCloning** — Supervised imitation learning
//! - **DaggerTrainer** — Dataset Aggregation (Ross et al. 2011)
//! - **RewardShaping** — Potential-based reward shaping (Ng et al. 1999)
//! - **IrlMetrics** / **IrlReport** — Evaluation metrics for IRL

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice, returning a new `Vec<f64>`.
fn irl_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Numerically stable sigmoid.
#[inline]
fn irl_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Binary cross-entropy with epsilon clamping.
#[inline]
#[allow(dead_code)]
fn irl_bce(p: f64, y: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let pc = p.clamp(EPS, 1.0 - EPS);
    -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
}

/// ReLU activation.
#[inline]
fn irl_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Dot product of two slices.
fn irl_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a slice.
fn irl_l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: MLP Layer (shared utility)
// ─────────────────────────────────────────────────────────────────────────────

/// A single fully-connected layer used across IRL/IL components.
#[derive(Debug, Clone)]
struct IrlFcLayer {
    weights: Vec<f64>,
    biases: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl IrlFcLayer {
    /// Xavier/Glorot uniform initialisation.
    fn xavier_init(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let weights: Vec<f64> = (0..in_dim * out_dim)
            .map(|_| {
                let u: f64 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();
        let biases = vec![0.0; out_dim];
        Self {
            weights,
            biases,
            in_dim,
            out_dim,
        }
    }

    /// Linear transform: y = Wx + b (no activation).
    fn linear(&self, input: &[f64]) -> Vec<f64> {
        let mut out = self.biases.clone();
        for (o, val) in out.iter_mut().enumerate() {
            for i in 0..self.in_dim {
                *val += self.weights[o * self.in_dim + i] * input[i];
            }
        }
        out
    }

    /// Forward with ReLU activation.
    fn forward_relu(&self, input: &[f64]) -> Vec<f64> {
        self.linear(input).iter().map(|&v| irl_relu(v)).collect()
    }
}

/// Multi-layer perceptron for IRL/IL usage.
#[derive(Debug, Clone)]
struct IrlMlp {
    layers: Vec<IrlFcLayer>,
}

impl IrlMlp {
    /// Build an MLP with the given layer dimensions. Output layer is linear (no ReLU).
    fn new(dims: &[usize], rng: &mut StdRng) -> Result<Self> {
        if dims.len() < 2 {
            return Err(TensorError::invalid_argument_op(
                "IrlMlp::new",
                "need at least 2 dimensions (input + output)",
            ));
        }
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            layers.push(IrlFcLayer::xavier_init(dims[i], dims[i + 1], rng));
        }
        Ok(Self { layers })
    }

    /// Forward pass: ReLU on hidden layers, linear output.
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let n_hidden = self.layers.len().saturating_sub(1);
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            if i < n_hidden {
                x = layer.forward_relu(&x);
            } else {
                x = layer.linear(&x);
            }
        }
        x
    }

    /// Finite-difference gradient step on a loss function.
    /// `loss_fn` takes the MLP and returns the scalar loss.
    fn fd_step<F>(&mut self, loss_fn: &F, lr: f64)
    where
        F: Fn(&Self) -> f64,
    {
        const H: f64 = 1e-5;
        for l_idx in 0..self.layers.len() {
            let n_w = self.layers[l_idx].weights.len();
            for w_idx in 0..n_w {
                let orig = self.layers[l_idx].weights[w_idx];
                self.layers[l_idx].weights[w_idx] = orig + H;
                let lp = loss_fn(self);
                self.layers[l_idx].weights[w_idx] = orig - H;
                let lm = loss_fn(self);
                self.layers[l_idx].weights[w_idx] = orig;
                let grad = (lp - lm) / (2.0 * H);
                self.layers[l_idx].weights[w_idx] -= lr * grad;
            }
            let n_b = self.layers[l_idx].biases.len();
            for b_idx in 0..n_b {
                let orig = self.layers[l_idx].biases[b_idx];
                self.layers[l_idx].biases[b_idx] = orig + H;
                let lp = loss_fn(self);
                self.layers[l_idx].biases[b_idx] = orig - H;
                let lm = loss_fn(self);
                self.layers[l_idx].biases[b_idx] = orig;
                let grad = (lp - lm) / (2.0 * H);
                self.layers[l_idx].biases[b_idx] -= lr * grad;
            }
        }
    }

    /// Total number of parameters.
    fn n_params(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: IrlMdp — MDP Representation for IRL
// ─────────────────────────────────────────────────────────────────────────────

/// An expert demonstration: sequence of (state, action) pairs.
#[derive(Debug, Clone)]
pub struct IrlDemonstration {
    /// Sequence of states visited.
    pub states: Vec<usize>,
    /// Sequence of actions taken (same length as `states`).
    pub actions: Vec<usize>,
}

impl IrlDemonstration {
    /// Create a new demonstration from state-action sequences.
    pub fn new(states: Vec<usize>, actions: Vec<usize>) -> Result<Self> {
        if states.len() != actions.len() {
            return Err(TensorError::invalid_argument_op(
                "IrlDemonstration::new",
                "states and actions must have the same length",
            ));
        }
        if states.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IrlDemonstration::new",
                "demonstration must not be empty",
            ));
        }
        Ok(Self { states, actions })
    }

    /// Length of the demonstration trajectory.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Transition probability: `T(s, a, s') = probability`.
/// Key is `(state, action, next_state)`.
type IrlTransitionMap = HashMap<(usize, usize, usize), f64>;

/// MDP representation for inverse reinforcement learning.
///
/// Supports discrete state and action spaces with tabular transition probabilities.
#[derive(Debug, Clone)]
pub struct IrlMdp {
    /// Number of states.
    pub n_states: usize,
    /// Number of actions.
    pub n_actions: usize,
    /// Transition probabilities T(s,a,s').
    transitions: IrlTransitionMap,
    /// Feature vectors for each state, shape: `n_states` x `feature_dim`.
    features: Vec<Vec<f64>>,
    /// Dimensionality of state features.
    pub feature_dim: usize,
}

impl IrlMdp {
    /// Create a new MDP with the given state/action counts and feature dimension.
    pub fn new(n_states: usize, n_actions: usize, feature_dim: usize) -> Result<Self> {
        if n_states == 0 || n_actions == 0 || feature_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::new",
                "n_states, n_actions, and feature_dim must all be > 0",
            ));
        }
        let features = vec![vec![0.0; feature_dim]; n_states];
        Ok(Self {
            n_states,
            n_actions,
            transitions: HashMap::new(),
            features,
            feature_dim,
        })
    }

    /// Set the feature vector for a given state.
    pub fn set_features(&mut self, state: usize, feats: Vec<f64>) -> Result<()> {
        if state >= self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::set_features",
                "state index out of range",
            ));
        }
        if feats.len() != self.feature_dim {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::set_features",
                "feature vector dimension mismatch",
            ));
        }
        self.features[state] = feats;
        Ok(())
    }

    /// Get the feature vector for a given state.
    pub fn get_features(&self, state: usize) -> Result<&[f64]> {
        if state >= self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::get_features",
                "state index out of range",
            ));
        }
        Ok(&self.features[state])
    }

    /// Set a transition probability T(s, a, s').
    pub fn set_transition(
        &mut self,
        state: usize,
        action: usize,
        next_state: usize,
        prob: f64,
    ) -> Result<()> {
        if state >= self.n_states || next_state >= self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::set_transition",
                "state index out of range",
            ));
        }
        if action >= self.n_actions {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::set_transition",
                "action index out of range",
            ));
        }
        if !(0.0..=1.0).contains(&prob) {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::set_transition",
                "probability must be in [0, 1]",
            ));
        }
        self.transitions.insert((state, action, next_state), prob);
        Ok(())
    }

    /// Get transition probability T(s, a, s'). Returns 0 if not set.
    pub fn get_transition(&self, state: usize, action: usize, next_state: usize) -> f64 {
        self.transitions
            .get(&(state, action, next_state))
            .copied()
            .unwrap_or(0.0)
    }

    /// Value iteration: compute state values for given reward vector.
    ///
    /// `rewards[s]` is the reward at state `s`.
    /// Returns the state value function V(s).
    pub fn value_iteration(
        &self,
        rewards: &[f64],
        gamma: f64,
        max_iters: usize,
    ) -> Result<Vec<f64>> {
        if rewards.len() != self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::value_iteration",
                "rewards length must equal n_states",
            ));
        }
        let mut values = vec![0.0_f64; self.n_states];
        for _iter in 0..max_iters {
            let mut new_values = vec![0.0_f64; self.n_states];
            let mut max_delta = 0.0_f64;
            for s in 0..self.n_states {
                let mut best_q = f64::NEG_INFINITY;
                for a in 0..self.n_actions {
                    let q = self.compute_q_value(s, a, rewards, &values, gamma);
                    if q > best_q {
                        best_q = q;
                    }
                }
                if best_q == f64::NEG_INFINITY {
                    best_q = rewards[s];
                }
                new_values[s] = best_q;
                let delta = (new_values[s] - values[s]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
            }
            values = new_values;
            if max_delta < 1e-8 {
                break;
            }
        }
        Ok(values)
    }

    /// Compute Q(s, a) = R(s) + gamma * sum_s' T(s,a,s') * V(s').
    fn compute_q_value(
        &self,
        state: usize,
        action: usize,
        rewards: &[f64],
        values: &[f64],
        gamma: f64,
    ) -> f64 {
        let mut expected_next = 0.0_f64;
        let mut has_transition = false;
        for sp in 0..self.n_states {
            let p = self.get_transition(state, action, sp);
            if p > 0.0 {
                expected_next += p * values[sp];
                has_transition = true;
            }
        }
        if !has_transition {
            return f64::NEG_INFINITY;
        }
        rewards[state] + gamma * expected_next
    }

    /// Compute a greedy policy from a value function.
    /// Returns `policy[s]` = best action at state `s`.
    pub fn compute_policy(&self, values: &[f64], gamma: f64) -> Result<Vec<usize>> {
        if values.len() != self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::compute_policy",
                "values length must equal n_states",
            ));
        }
        let mut policy = vec![0usize; self.n_states];
        for s in 0..self.n_states {
            let mut best_a = 0;
            let mut best_q = f64::NEG_INFINITY;
            for a in 0..self.n_actions {
                let mut q = 0.0_f64;
                let mut has_trans = false;
                for sp in 0..self.n_states {
                    let p = self.get_transition(s, a, sp);
                    if p > 0.0 {
                        q += p * values[sp];
                        has_trans = true;
                    }
                }
                if has_trans && q > best_q {
                    best_q = q;
                    best_a = a;
                }
            }
            policy[s] = best_a;
        }
        Ok(policy)
    }

    /// Compute state visitation frequencies under a stochastic policy.
    ///
    /// Uses soft Bellman iteration to compute expected visitation counts.
    /// `policy[s]` is a probability distribution over actions at state `s`.
    /// `initial_dist[s]` is the probability of starting in state `s`.
    pub fn state_visitation_frequency(
        &self,
        policy: &[Vec<f64>],
        initial_dist: &[f64],
        gamma: f64,
        max_iters: usize,
    ) -> Result<Vec<f64>> {
        if policy.len() != self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::state_visitation_frequency",
                "policy length must equal n_states",
            ));
        }
        if initial_dist.len() != self.n_states {
            return Err(TensorError::invalid_argument_op(
                "IrlMdp::state_visitation_frequency",
                "initial_dist length must equal n_states",
            ));
        }
        // D_t(s) = p_0(s) + gamma * sum_{s',a} D_{t-1}(s') * pi(a|s') * T(s',a,s)
        let mut freq = initial_dist.to_vec();
        for _t in 0..max_iters {
            let mut new_freq = initial_dist.to_vec();
            for sp in 0..self.n_states {
                for a in 0..self.n_actions {
                    let pi_a = if a < policy[sp].len() {
                        policy[sp][a]
                    } else {
                        0.0
                    };
                    if pi_a < 1e-15 || freq[sp] < 1e-15 {
                        continue;
                    }
                    for s_next in 0..self.n_states {
                        let t_prob = self.get_transition(sp, a, s_next);
                        if t_prob > 0.0 {
                            new_freq[s_next] += gamma * freq[sp] * pi_a * t_prob;
                        }
                    }
                }
            }
            let delta: f64 = freq
                .iter()
                .zip(new_freq.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            freq = new_freq;
            if delta < 1e-8 {
                break;
            }
        }
        Ok(freq)
    }

    /// Compute a softmax (Boltzmann) policy from Q-values.
    /// Returns `policy[s][a]` = probability of taking action `a` in state `s`.
    pub fn soft_policy(
        &self,
        rewards: &[f64],
        gamma: f64,
        temperature: f64,
        max_iters: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let values = self.value_iteration(rewards, gamma, max_iters)?;
        let mut policy = vec![vec![0.0; self.n_actions]; self.n_states];
        for s in 0..self.n_states {
            let mut q_values = vec![f64::NEG_INFINITY; self.n_actions];
            for a in 0..self.n_actions {
                let q = self.compute_q_value(s, a, rewards, &values, gamma);
                q_values[a] = q / temperature.max(1e-10);
            }
            // Replace NEG_INFINITY with a large negative value for softmax stability
            let min_finite = q_values
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            let fallback = if min_finite.is_finite() {
                min_finite - 100.0
            } else {
                -100.0
            };
            for q in q_values.iter_mut() {
                if !q.is_finite() {
                    *q = fallback;
                }
            }
            policy[s] = irl_softmax(&q_values);
        }
        Ok(policy)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: MaxEntIrl — Maximum Entropy IRL (Ziebart 2008)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum Entropy IRL (Ziebart et al. 2008).
///
/// Learns a reward function `R(s) = theta . phi(s)` (linear in features) such that
/// the induced policy maximizes entropy subject to matching expert feature
/// expectations.
#[derive(Debug, Clone)]
pub struct MaxEntIrl {
    /// Learned reward weights.
    pub weights: Vec<f64>,
    /// Feature dimension.
    pub feature_dim: usize,
    /// Reference to MDP (cloned).
    mdp: IrlMdp,
    /// Discount factor.
    gamma: f64,
}

impl MaxEntIrl {
    /// Create a new MaxEntIRL learner for the given MDP.
    pub fn new(mdp: &IrlMdp, gamma: f64) -> Self {
        let feature_dim = mdp.feature_dim;
        Self {
            weights: vec![0.0; feature_dim],
            feature_dim,
            mdp: mdp.clone(),
            gamma,
        }
    }

    /// Compute feature expectations from expert demonstrations.
    ///
    /// `f_expert = (1/|D|) * sum_{traj in D} sum_{t} gamma^t phi(s_t)`
    pub fn expert_feature_expectations(
        &self,
        demonstrations: &[IrlDemonstration],
    ) -> Result<Vec<f64>> {
        if demonstrations.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "MaxEntIrl::expert_feature_expectations",
                "demonstrations must not be empty",
            ));
        }
        let mut f_expert = vec![0.0; self.feature_dim];
        for demo in demonstrations {
            let mut discount = 1.0_f64;
            for &s in &demo.states {
                if s >= self.mdp.n_states {
                    return Err(TensorError::invalid_argument_op(
                        "MaxEntIrl::expert_feature_expectations",
                        "state index in demonstration out of range",
                    ));
                }
                let feats = &self.mdp.features[s];
                for (j, &f) in feats.iter().enumerate() {
                    f_expert[j] += discount * f;
                }
                discount *= self.gamma;
            }
        }
        let n = demonstrations.len() as f64;
        for v in f_expert.iter_mut() {
            *v /= n;
        }
        Ok(f_expert)
    }

    /// Compute the reward vector R(s) = theta . phi(s) for all states.
    pub fn compute_rewards(&self) -> Vec<f64> {
        (0..self.mdp.n_states)
            .map(|s| irl_dot(&self.weights, &self.mdp.features[s]))
            .collect()
    }

    /// Train the MaxEnt IRL model via gradient ascent on the log-likelihood.
    ///
    /// Gradient: `nabla L = f_expert - f_policy` (feature matching).
    pub fn train(
        &mut self,
        demonstrations: &[IrlDemonstration],
        n_iters: usize,
        lr: f64,
    ) -> Result<Vec<f64>> {
        let f_expert = self.expert_feature_expectations(demonstrations)?;

        // Uniform initial distribution
        let initial_dist = vec![1.0 / self.mdp.n_states as f64; self.mdp.n_states];
        let mut loss_history = Vec::with_capacity(n_iters);

        for _iter in 0..n_iters {
            let rewards = self.compute_rewards();
            // Compute soft policy under current rewards
            let policy = self.mdp.soft_policy(&rewards, self.gamma, 1.0, 100)?;
            // State visitation frequencies
            let svf =
                self.mdp
                    .state_visitation_frequency(&policy, &initial_dist, self.gamma, 100)?;

            // Feature expectations under current policy
            let mut f_policy = vec![0.0; self.feature_dim];
            for s in 0..self.mdp.n_states {
                for (j, &feat) in self.mdp.features[s].iter().enumerate() {
                    f_policy[j] += svf[s] * feat;
                }
            }

            // Gradient = f_expert - f_policy
            let mut grad_norm_sq = 0.0_f64;
            for j in 0..self.feature_dim {
                let grad = f_expert[j] - f_policy[j];
                self.weights[j] += lr * grad;
                grad_norm_sq += grad * grad;
            }
            loss_history.push(grad_norm_sq.sqrt());
        }
        Ok(loss_history)
    }

    /// Get the learned reward weights.
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: GailDiscriminator — GAIL Discriminator (Ho & Ermon 2016)
// ─────────────────────────────────────────────────────────────────────────────

/// State-action pair for GAIL.
#[derive(Debug, Clone)]
pub struct IrlStateAction {
    /// State vector.
    pub state: Vec<f64>,
    /// Action (one-hot encoded or index as single element).
    pub action: Vec<f64>,
}

impl IrlStateAction {
    /// Create from a state vector and action vector.
    pub fn new(state: Vec<f64>, action: Vec<f64>) -> Self {
        Self { state, action }
    }

    /// Concatenate state and action into a single feature vector.
    pub fn as_features(&self) -> Vec<f64> {
        let mut feats = self.state.clone();
        feats.extend_from_slice(&self.action);
        feats
    }
}

/// GAIL discriminator: binary classifier distinguishing expert from policy data.
///
/// D(s,a) -> [0, 1], probability of being an expert sample.
/// Reward signal: r = -log(1 - D(s,a)).
#[derive(Debug, Clone)]
pub struct GailDiscriminator {
    /// MLP backbone.
    network: IrlMlp,
    /// Input dimension (state_dim + action_dim).
    pub input_dim: usize,
    /// Gradient penalty coefficient (WGAN-GP style).
    pub gradient_penalty_coeff: f64,
}

impl GailDiscriminator {
    /// Create a new GAIL discriminator.
    ///
    /// # Arguments
    /// * `state_dim` — dimensionality of state space
    /// * `action_dim` — dimensionality of action space (1 for discrete index)
    /// * `hidden_dims` — hidden layer sizes
    /// * `gradient_penalty_coeff` — lambda for gradient penalty (0 to disable)
    /// * `seed` — random seed
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: &[usize],
        gradient_penalty_coeff: f64,
        seed: u64,
    ) -> Result<Self> {
        if state_dim == 0 || action_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "GailDiscriminator::new",
                "state_dim and action_dim must be > 0",
            ));
        }
        let input_dim = state_dim + action_dim;
        let mut dims = vec![input_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(1); // binary output
        let mut rng = StdRng::seed_from_u64(seed);
        let network = IrlMlp::new(&dims, &mut rng)?;
        Ok(Self {
            network,
            input_dim,
            gradient_penalty_coeff,
        })
    }

    /// Forward pass: compute D(s,a) in (0, 1).
    pub fn forward(&self, sa: &IrlStateAction) -> f64 {
        let feats = sa.as_features();
        let logit = self.network.forward(&feats);
        irl_sigmoid(logit[0])
    }

    /// Compute reward signal: r = -log(1 - D(s,a)).
    pub fn reward(&self, sa: &IrlStateAction) -> f64 {
        let d = self.forward(sa);
        const EPS: f64 = 1e-10;
        -(1.0 - d + EPS).ln()
    }

    /// Compute discriminator loss on expert and policy batches.
    ///
    /// L = -E[log D(s_e, a_e)] - E[log(1 - D(s_p, a_p))] + lambda * GP
    pub fn compute_loss(
        &self,
        expert_batch: &[IrlStateAction],
        policy_batch: &[IrlStateAction],
    ) -> Result<f64> {
        if expert_batch.is_empty() || policy_batch.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "GailDiscriminator::compute_loss",
                "batches must not be empty",
            ));
        }
        let expert_loss: f64 = expert_batch
            .iter()
            .map(|sa| {
                let d = self.forward(sa);
                -((d + 1e-10).ln())
            })
            .sum::<f64>()
            / expert_batch.len() as f64;

        let policy_loss: f64 = policy_batch
            .iter()
            .map(|sa| {
                let d = self.forward(sa);
                -((1.0 - d + 1e-10).ln())
            })
            .sum::<f64>()
            / policy_batch.len() as f64;

        let gp = if self.gradient_penalty_coeff > 0.0 {
            self.gradient_penalty(expert_batch, policy_batch)
        } else {
            0.0
        };

        Ok(expert_loss + policy_loss + self.gradient_penalty_coeff * gp)
    }

    /// WGAN-GP style gradient penalty via finite differences.
    fn gradient_penalty(
        &self,
        expert_batch: &[IrlStateAction],
        policy_batch: &[IrlStateAction],
    ) -> f64 {
        let n = expert_batch.len().min(policy_batch.len());
        if n == 0 {
            return 0.0;
        }
        const H: f64 = 1e-5;
        let mut total_penalty = 0.0;

        for i in 0..n {
            let e_feats = expert_batch[i].as_features();
            let p_feats = policy_batch[i].as_features();
            let dim = e_feats.len().min(p_feats.len());

            // Interpolate: x = 0.5 * e + 0.5 * p
            let interp: Vec<f64> = (0..dim)
                .map(|j| 0.5 * e_feats[j] + 0.5 * p_feats[j])
                .collect();

            // Estimate gradient norm via finite differences
            let mut grad_norm_sq = 0.0_f64;
            let base_out = self.network.forward(&interp)[0];
            for j in 0..dim {
                let mut perturbed = interp.clone();
                perturbed[j] += H;
                let perturbed_out = self.network.forward(&perturbed)[0];
                let partial = (perturbed_out - base_out) / H;
                grad_norm_sq += partial * partial;
            }
            let grad_norm = grad_norm_sq.sqrt();
            total_penalty += (grad_norm - 1.0).powi(2);
        }
        total_penalty / n as f64
    }

    /// Train the discriminator for one step using finite-difference gradients.
    pub fn train_step(
        &mut self,
        expert_batch: &[IrlStateAction],
        policy_batch: &[IrlStateAction],
        lr: f64,
    ) -> Result<f64> {
        let loss_before = self.compute_loss(expert_batch, policy_batch)?;
        let eb = expert_batch.to_vec();
        let pb = policy_batch.to_vec();
        let gp_coeff = self.gradient_penalty_coeff;
        self.network.fd_step(
            &|net: &IrlMlp| {
                let temp_disc = GailDiscriminator {
                    network: net.clone(),
                    input_dim: eb[0].as_features().len(),
                    gradient_penalty_coeff: gp_coeff,
                };
                temp_disc.compute_loss(&eb, &pb).unwrap_or(f64::MAX)
            },
            lr,
        );
        Ok(loss_before)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: GailTrainer — Full GAIL Training Loop
// ─────────────────────────────────────────────────────────────────────────────

/// Result of one GAIL training step.
#[derive(Debug, Clone)]
pub struct GailStepResult {
    /// Discriminator loss.
    pub disc_loss: f64,
    /// Policy loss (negative of mean GAIL reward).
    pub policy_loss: f64,
    /// Mean entropy bonus.
    pub entropy_bonus: f64,
}

/// Full GAIL training loop: alternating discriminator and policy updates.
#[derive(Debug, Clone)]
pub struct GailTrainer {
    /// Discriminator.
    pub discriminator: GailDiscriminator,
    /// Policy network (maps state -> action logits).
    policy: IrlMlp,
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension (number of discrete actions).
    pub n_actions: usize,
    /// Entropy bonus coefficient.
    pub entropy_coeff: f64,
    /// PPO clip epsilon.
    pub ppo_epsilon: f64,
}

impl GailTrainer {
    /// Create a new GAIL trainer.
    pub fn new(
        state_dim: usize,
        n_actions: usize,
        disc_hidden: &[usize],
        policy_hidden: &[usize],
        entropy_coeff: f64,
        ppo_epsilon: f64,
        seed: u64,
    ) -> Result<Self> {
        let discriminator = GailDiscriminator::new(state_dim, 1, disc_hidden, 10.0, seed)?;

        let mut policy_dims = vec![state_dim];
        policy_dims.extend_from_slice(policy_hidden);
        policy_dims.push(n_actions);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        let policy = IrlMlp::new(&policy_dims, &mut rng)?;

        Ok(Self {
            discriminator,
            policy,
            state_dim,
            n_actions,
            entropy_coeff,
            ppo_epsilon,
        })
    }

    /// Select an action from the policy (stochastic via softmax).
    pub fn select_action(&self, state: &[f64], rng: &mut StdRng) -> Result<usize> {
        if state.len() != self.state_dim {
            return Err(TensorError::invalid_argument_op(
                "GailTrainer::select_action",
                "state dimension mismatch",
            ));
        }
        let logits = self.policy.forward(state);
        let probs = irl_softmax(&logits);
        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return Ok(i);
            }
        }
        Ok(probs.len().saturating_sub(1))
    }

    /// One training step: update discriminator then policy.
    pub fn train_step(
        &mut self,
        expert_batch: &[IrlStateAction],
        policy_batch: &[IrlStateAction],
        disc_lr: f64,
        policy_lr: f64,
    ) -> Result<GailStepResult> {
        // Step 1: Train discriminator
        let disc_loss = self
            .discriminator
            .train_step(expert_batch, policy_batch, disc_lr)?;

        // Step 2: Compute GAIL rewards for policy batch
        let gail_rewards: Vec<f64> = policy_batch
            .iter()
            .map(|sa| self.discriminator.reward(sa))
            .collect();

        // Step 3: Compute policy entropy bonus
        let mut total_entropy = 0.0;
        for sa in policy_batch {
            let logits = self.policy.forward(&sa.state);
            let probs = irl_softmax(&logits);
            let ent: f64 = probs
                .iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();
            total_entropy += ent;
        }
        let mean_entropy = if policy_batch.is_empty() {
            0.0
        } else {
            total_entropy / policy_batch.len() as f64
        };

        // Step 4: Policy gradient update via FD on the GAIL objective
        let mean_reward = if gail_rewards.is_empty() {
            0.0
        } else {
            gail_rewards.iter().sum::<f64>() / gail_rewards.len() as f64
        };
        let ent_c = self.entropy_coeff;
        let pb_clone = policy_batch.to_vec();
        let disc_clone = self.discriminator.clone();
        self.policy.fd_step(
            &|net: &IrlMlp| {
                let mut total_loss = 0.0;
                for sa in &pb_clone {
                    let logits = net.forward(&sa.state);
                    let probs = irl_softmax(&logits);
                    let action_idx = sa.action[0] as usize;
                    let log_prob = if action_idx < probs.len() {
                        (probs[action_idx] + 1e-15).ln()
                    } else {
                        (-15.0_f64).ln()
                    };
                    let sa_pair = IrlStateAction::new(sa.state.clone(), sa.action.clone());
                    let gail_r = disc_clone.reward(&sa_pair);
                    total_loss -= log_prob * gail_r;
                    let ent: f64 = probs
                        .iter()
                        .filter(|&&p| p > 1e-15)
                        .map(|&p| -p * p.ln())
                        .sum();
                    total_loss -= ent_c * ent;
                }
                if pb_clone.is_empty() {
                    0.0
                } else {
                    total_loss / pb_clone.len() as f64
                }
            },
            policy_lr,
        );

        Ok(GailStepResult {
            disc_loss,
            policy_loss: -mean_reward,
            entropy_bonus: mean_entropy,
        })
    }

    /// Get the current policy parameters count.
    pub fn policy_n_params(&self) -> usize {
        self.policy.n_params()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: DaggerTrainer — Dataset Aggregation (Ross et al. 2011)
// ─────────────────────────────────────────────────────────────────────────────

/// A (state, action) pair for the DAgger aggregated dataset.
#[derive(Debug, Clone)]
pub struct IrlLabeledSample {
    /// State feature vector.
    pub state: Vec<f64>,
    /// Expert action label.
    pub action: usize,
}

/// DAgger trainer: iterative dataset aggregation for imitation learning.
///
/// Alternates between rolling out the current policy and querying the expert
/// to label visited states, building an aggregated dataset.
#[derive(Debug, Clone)]
pub struct DaggerTrainer {
    /// Policy network (state -> action logits).
    policy: IrlMlp,
    /// State dimension.
    pub state_dim: usize,
    /// Number of actions.
    pub n_actions: usize,
    /// Aggregated dataset.
    pub dataset: Vec<IrlLabeledSample>,
    /// Current mixing coefficient beta.
    pub beta: f64,
    /// Decay rate for beta: beta_{t+1} = p * beta_t.
    pub beta_decay: f64,
    /// Current DAgger iteration.
    pub iteration: usize,
}

impl DaggerTrainer {
    /// Create a new DAgger trainer.
    pub fn new(
        state_dim: usize,
        n_actions: usize,
        hidden_dims: &[usize],
        beta_decay: f64,
        seed: u64,
    ) -> Result<Self> {
        if state_dim == 0 || n_actions == 0 {
            return Err(TensorError::invalid_argument_op(
                "DaggerTrainer::new",
                "state_dim and n_actions must be > 0",
            ));
        }
        let mut dims = vec![state_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(n_actions);
        let mut rng = StdRng::seed_from_u64(seed);
        let policy = IrlMlp::new(&dims, &mut rng)?;
        Ok(Self {
            policy,
            state_dim,
            n_actions,
            dataset: Vec::new(),
            beta: 1.0,
            beta_decay,
            iteration: 0,
        })
    }

    /// Select an action: with probability beta use the expert, else use the policy.
    pub fn select_action(&self, state: &[f64], expert_action: usize, rng: &mut StdRng) -> usize {
        let u: f64 = rng.random();
        if u < self.beta {
            expert_action
        } else {
            let logits = self.policy.forward(state);
            let probs = irl_softmax(&logits);
            let mut cumulative = 0.0;
            let sample: f64 = rng.random();
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if sample < cumulative {
                    return i;
                }
            }
            probs.len().saturating_sub(1)
        }
    }

    /// Perform one DAgger iteration step.
    ///
    /// - `trajectories`: new (state, expert_action) pairs from the current rollout
    /// - `lr`: learning rate for policy update
    /// - `n_epochs`: number of training epochs on the aggregated dataset
    ///
    /// Returns the mean cross-entropy loss after training.
    pub fn dagger_step(
        &mut self,
        trajectories: &[(Vec<f64>, usize)],
        lr: f64,
        n_epochs: usize,
    ) -> Result<f64> {
        if trajectories.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DaggerTrainer::dagger_step",
                "trajectories must not be empty",
            ));
        }
        // Aggregate new data
        for (state, action) in trajectories {
            self.dataset.push(IrlLabeledSample {
                state: state.clone(),
                action: *action,
            });
        }
        // Train the policy on the aggregated dataset
        let loss = self.train_policy(lr, n_epochs)?;
        // Decay beta
        self.beta *= self.beta_decay;
        self.iteration += 1;
        Ok(loss)
    }

    /// Train the policy network on the full aggregated dataset.
    fn train_policy(&mut self, lr: f64, n_epochs: usize) -> Result<f64> {
        let dataset = self.dataset.clone();
        let n_actions = self.n_actions;
        let mut final_loss = 0.0;
        for _epoch in 0..n_epochs {
            let loss_fn = |net: &IrlMlp| -> f64 {
                let mut total = 0.0;
                for sample in &dataset {
                    let logits = net.forward(&sample.state);
                    let probs = irl_softmax(&logits);
                    let target = sample.action.min(n_actions - 1);
                    total -= (probs[target] + 1e-15).ln();
                }
                if dataset.is_empty() {
                    0.0
                } else {
                    total / dataset.len() as f64
                }
            };
            #[allow(clippy::redundant_closure_call)]
            {
                final_loss = loss_fn(&self.policy);
            }
            self.policy.fd_step(&loss_fn, lr);
        }
        Ok(final_loss)
    }

    /// Get the current policy's action probabilities for a state.
    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.policy.forward(state);
        irl_softmax(&logits)
    }

    /// Get the current dataset size.
    pub fn dataset_size(&self) -> usize {
        self.dataset.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: AirlModel — Adversarial IRL (Fu et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Adversarial IRL (AIRL) model: disentangled reward recovery.
///
/// The discriminator is structured as:
///   D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + pi(a|s))
/// where f(s,a,s') = r(s,a) + gamma * h(s') - h(s) and h is a shaping potential.
#[derive(Debug, Clone)]
pub struct AirlModel {
    /// Reward network: r(s,a).
    reward_net: IrlMlp,
    /// Shaping potential: h(s).
    shaping_net: IrlMlp,
    /// State dimension.
    pub state_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Discount factor.
    pub gamma: f64,
    /// Whether to use state-only reward (ignoring action).
    pub state_only: bool,
}

impl AirlModel {
    /// Create a new AIRL model.
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: &[usize],
        gamma: f64,
        state_only: bool,
        seed: u64,
    ) -> Result<Self> {
        if state_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "AirlModel::new",
                "state_dim must be > 0",
            ));
        }
        let reward_input = if state_only {
            state_dim
        } else {
            state_dim + action_dim
        };
        let mut r_dims = vec![reward_input];
        r_dims.extend_from_slice(hidden_dims);
        r_dims.push(1);

        let mut h_dims = vec![state_dim];
        h_dims.extend_from_slice(hidden_dims);
        h_dims.push(1);

        let mut rng = StdRng::seed_from_u64(seed);
        let reward_net = IrlMlp::new(&r_dims, &mut rng)?;
        let shaping_net = IrlMlp::new(&h_dims, &mut rng)?;

        Ok(Self {
            reward_net,
            shaping_net,
            state_dim,
            action_dim,
            gamma,
            state_only,
        })
    }

    /// Compute the reward r(s, a) from the reward network.
    pub fn reward(&self, state: &[f64], action: &[f64]) -> f64 {
        if self.state_only {
            self.reward_net.forward(state)[0]
        } else {
            let mut input = state.to_vec();
            input.extend_from_slice(action);
            self.reward_net.forward(&input)[0]
        }
    }

    /// Compute the shaping potential h(s).
    pub fn shaping_potential(&self, state: &[f64]) -> f64 {
        self.shaping_net.forward(state)[0]
    }

    /// Compute f(s, a, s') = r(s,a) + gamma * h(s') - h(s).
    pub fn f_value(&self, state: &[f64], action: &[f64], next_state: &[f64]) -> f64 {
        let r = self.reward(state, action);
        let h_s = self.shaping_potential(state);
        let h_sp = self.shaping_potential(next_state);
        r + self.gamma * h_sp - h_s
    }

    /// Discriminator output: D(s,a,s') = sigma(f(s,a,s') - log pi(a|s)).
    pub fn discriminator(
        &self,
        state: &[f64],
        action: &[f64],
        next_state: &[f64],
        log_policy: f64,
    ) -> f64 {
        let f = self.f_value(state, action, next_state);
        irl_sigmoid(f - log_policy)
    }

    /// Train AIRL for one step on expert and policy transitions.
    ///
    /// Each transition is `(state, action, next_state, log_pi)`.
    pub fn train_step(
        &mut self,
        expert_transitions: &[(Vec<f64>, Vec<f64>, Vec<f64>, f64)],
        policy_transitions: &[(Vec<f64>, Vec<f64>, Vec<f64>, f64)],
        lr: f64,
    ) -> Result<f64> {
        if expert_transitions.is_empty() || policy_transitions.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "AirlModel::train_step",
                "transition batches must not be empty",
            ));
        }
        // Compute loss
        let loss = self.compute_airl_loss(expert_transitions, policy_transitions);

        // FD update for reward network
        let et = expert_transitions.to_vec();
        let pt = policy_transitions.to_vec();
        let gamma = self.gamma;
        let state_only = self.state_only;
        let shaping = self.shaping_net.clone();

        self.reward_net.fd_step(
            &|net: &IrlMlp| {
                let mut total = 0.0;
                for (s, a, sp, lp) in &et {
                    let r = if state_only {
                        net.forward(s)[0]
                    } else {
                        let mut inp = s.clone();
                        inp.extend_from_slice(a);
                        net.forward(&inp)[0]
                    };
                    let f = r + gamma * shaping.forward(sp)[0] - shaping.forward(s)[0];
                    let d = irl_sigmoid(f - lp);
                    total -= (d + 1e-10).ln();
                }
                for (s, a, sp, lp) in &pt {
                    let r = if state_only {
                        net.forward(s)[0]
                    } else {
                        let mut inp = s.clone();
                        inp.extend_from_slice(a);
                        net.forward(&inp)[0]
                    };
                    let f = r + gamma * shaping.forward(sp)[0] - shaping.forward(s)[0];
                    let d = irl_sigmoid(f - lp);
                    total -= (1.0 - d + 1e-10).ln();
                }
                total / (et.len() + pt.len()) as f64
            },
            lr,
        );

        // FD update for shaping network
        let et2 = expert_transitions.to_vec();
        let pt2 = policy_transitions.to_vec();
        let reward = self.reward_net.clone();
        let so2 = self.state_only;

        self.shaping_net.fd_step(
            &|net: &IrlMlp| {
                let mut total = 0.0;
                for (s, a, sp, lp) in &et2 {
                    let r = if so2 {
                        reward.forward(s)[0]
                    } else {
                        let mut inp = s.clone();
                        inp.extend_from_slice(a);
                        reward.forward(&inp)[0]
                    };
                    let f = r + gamma * net.forward(sp)[0] - net.forward(s)[0];
                    let d = irl_sigmoid(f - lp);
                    total -= (d + 1e-10).ln();
                }
                for (s, a, sp, lp) in &pt2 {
                    let r = if so2 {
                        reward.forward(s)[0]
                    } else {
                        let mut inp = s.clone();
                        inp.extend_from_slice(a);
                        reward.forward(&inp)[0]
                    };
                    let f = r + gamma * net.forward(sp)[0] - net.forward(s)[0];
                    let d = irl_sigmoid(f - lp);
                    total -= (1.0 - d + 1e-10).ln();
                }
                total / (et2.len() + pt2.len()) as f64
            },
            lr,
        );

        Ok(loss)
    }

    /// Compute the AIRL binary cross-entropy loss.
    fn compute_airl_loss(
        &self,
        expert: &[(Vec<f64>, Vec<f64>, Vec<f64>, f64)],
        policy: &[(Vec<f64>, Vec<f64>, Vec<f64>, f64)],
    ) -> f64 {
        let mut total = 0.0;
        for (s, a, sp, lp) in expert {
            let d = self.discriminator(s, a, sp, *lp);
            total -= (d + 1e-10).ln();
        }
        for (s, a, sp, lp) in policy {
            let d = self.discriminator(s, a, sp, *lp);
            total -= (1.0 - d + 1e-10).ln();
        }
        total / (expert.len() + policy.len()) as f64
    }

    /// Recover the transferable reward function for a set of states.
    pub fn recover_reward(&self, states: &[Vec<f64>]) -> Vec<f64> {
        states
            .iter()
            .map(|s| self.reward_net.forward(s)[0])
            .collect()
    }

    /// Number of parameters in the model.
    pub fn n_params(&self) -> usize {
        self.reward_net.n_params() + self.shaping_net.n_params()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: BehavioralCloning — Supervised Imitation Learning
// ─────────────────────────────────────────────────────────────────────────────

/// Behavioral cloning: supervised imitation learning with an MLP policy.
///
/// Supports cross-entropy loss for discrete actions and MSE for continuous actions.
#[derive(Debug, Clone)]
pub struct BehavioralCloning {
    /// Policy network.
    policy: IrlMlp,
    /// State dimension.
    pub state_dim: usize,
    /// Output dimension (n_actions for discrete, action_dim for continuous).
    pub output_dim: usize,
    /// Whether to use cross-entropy (true) or MSE (false).
    pub discrete: bool,
    /// Gaussian noise standard deviation for data augmentation.
    pub noise_std: f64,
}

impl BehavioralCloning {
    /// Create a new behavioral cloning model.
    pub fn new(
        state_dim: usize,
        output_dim: usize,
        hidden_dims: &[usize],
        discrete: bool,
        noise_std: f64,
        seed: u64,
    ) -> Result<Self> {
        if state_dim == 0 || output_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::new",
                "state_dim and output_dim must be > 0",
            ));
        }
        let mut dims = vec![state_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(output_dim);
        let mut rng = StdRng::seed_from_u64(seed);
        let policy = IrlMlp::new(&dims, &mut rng)?;
        Ok(Self {
            policy,
            state_dim,
            output_dim,
            discrete,
            noise_std,
        })
    }

    /// Train on expert demonstrations.
    ///
    /// For discrete: `expert_actions[i]` is a one-hot or index.
    /// For continuous: `expert_actions[i]` is the target vector.
    pub fn train(
        &mut self,
        expert_states: &[Vec<f64>],
        expert_actions: &[Vec<f64>],
        epochs: usize,
        lr: f64,
        seed: u64,
    ) -> Result<Vec<f64>> {
        if expert_states.len() != expert_actions.len() {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::train",
                "states and actions must have the same length",
            ));
        }
        if expert_states.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "BehavioralCloning::train",
                "training data must not be empty",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut loss_history = Vec::with_capacity(epochs);
        let discrete = self.discrete;
        let noise_std = self.noise_std;
        let output_dim = self.output_dim;

        for _epoch in 0..epochs {
            // Data augmentation: add Gaussian noise to states
            let augmented_states: Vec<Vec<f64>> = expert_states
                .iter()
                .map(|s| {
                    if noise_std > 0.0 {
                        s.iter()
                            .map(|&v| {
                                // Box-Muller transform for Gaussian noise
                                let u1: f64 = rng.random::<f64>().max(1e-15);
                                let u2: f64 = rng.random();
                                let z = (-2.0 * u1.ln()).sqrt()
                                    * (2.0 * std::f64::consts::PI * u2).cos();
                                v + noise_std * z
                            })
                            .collect()
                    } else {
                        s.clone()
                    }
                })
                .collect();

            let states_ref = augmented_states.clone();
            let actions_ref = expert_actions.to_vec();
            let loss_fn = |net: &IrlMlp| -> f64 {
                let mut total = 0.0;
                for (s, a) in states_ref.iter().zip(actions_ref.iter()) {
                    let out = net.forward(s);
                    if discrete {
                        let probs = irl_softmax(&out);
                        // Find the target action index
                        let target = if a.len() == 1 {
                            a[0] as usize
                        } else {
                            // One-hot: find argmax
                            a.iter()
                                .enumerate()
                                .max_by(|(_, x), (_, y)| {
                                    x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, _)| i)
                                .unwrap_or(0)
                        };
                        let target = target.min(output_dim - 1);
                        total -= (probs[target] + 1e-15).ln();
                    } else {
                        // MSE loss
                        let mse: f64 = out
                            .iter()
                            .zip(a.iter())
                            .map(|(&o, &t)| (o - t).powi(2))
                            .sum::<f64>()
                            / out.len().max(1) as f64;
                        total += mse;
                    }
                }
                total / states_ref.len().max(1) as f64
            };
            let epoch_loss = loss_fn(&self.policy);
            loss_history.push(epoch_loss);
            self.policy.fd_step(&loss_fn, lr);
        }
        Ok(loss_history)
    }

    /// Predict action probabilities (discrete) or action values (continuous).
    pub fn predict(&self, state: &[f64]) -> Vec<f64> {
        let out = self.policy.forward(state);
        if self.discrete {
            irl_softmax(&out)
        } else {
            out
        }
    }

    /// Predict the argmax action (for discrete actions).
    pub fn predict_action(&self, state: &[f64]) -> usize {
        let probs = self.predict(state);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Number of parameters.
    pub fn n_params(&self) -> usize {
        self.policy.n_params()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: RewardShaping — Potential-Based Reward Shaping (Ng 1999)
// ─────────────────────────────────────────────────────────────────────────────

/// Type of potential function for reward shaping.
#[derive(Debug, Clone)]
pub enum IrlPotentialType {
    /// Tabular: potential Phi(s) stored per state.
    Tabular(Vec<f64>),
    /// Linear: Phi(s) = w . phi(s).
    Linear(Vec<f64>),
    /// Neural: Phi(s) via a small MLP.
    Neural,
}

/// Potential-based reward shaping (Ng et al. 1999).
///
/// Shaping function: `F(s, s') = gamma * Phi(s') - Phi(s)`.
/// Guarantees: optimal policy is invariant under potential-based shaping.
#[derive(Debug, Clone)]
pub struct RewardShaping {
    /// Discount factor.
    pub gamma: f64,
    /// Potential function type.
    potential_type: IrlPotentialType,
    /// Neural potential network (used when `potential_type` is `Neural`).
    neural_potential: Option<IrlMlp>,
}

impl RewardShaping {
    /// Create a new reward shaping instance with tabular potential.
    pub fn tabular(gamma: f64, potentials: Vec<f64>) -> Self {
        Self {
            gamma,
            potential_type: IrlPotentialType::Tabular(potentials),
            neural_potential: None,
        }
    }

    /// Create a new reward shaping instance with linear potential Phi(s) = w . s.
    pub fn linear(gamma: f64, weights: Vec<f64>) -> Self {
        Self {
            gamma,
            potential_type: IrlPotentialType::Linear(weights),
            neural_potential: None,
        }
    }

    /// Create a new reward shaping instance with neural potential.
    pub fn neural(gamma: f64, input_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let net = IrlMlp::new(&[input_dim, hidden_dim, 1], &mut rng)?;
        Ok(Self {
            gamma,
            potential_type: IrlPotentialType::Neural,
            neural_potential: Some(net),
        })
    }

    /// Compute the potential Phi(s) for a state.
    pub fn potential(&self, state: &[f64]) -> f64 {
        match &self.potential_type {
            IrlPotentialType::Tabular(potentials) => {
                if state.is_empty() {
                    return 0.0;
                }
                let idx = state[0] as usize;
                if idx < potentials.len() {
                    potentials[idx]
                } else {
                    0.0
                }
            }
            IrlPotentialType::Linear(weights) => irl_dot(weights, state),
            IrlPotentialType::Neural => {
                if let Some(net) = &self.neural_potential {
                    net.forward(state)[0]
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute the shaped reward: R'(s, a, s') = R(s, a, s') + F(s, s').
    ///
    /// `F(s, s') = gamma * Phi(s') - Phi(s)`.
    pub fn shape_reward(&self, original_reward: f64, state: &[f64], next_state: &[f64]) -> f64 {
        let phi_s = self.potential(state);
        let phi_sp = self.potential(next_state);
        let shaping = self.gamma * phi_sp - phi_s;
        original_reward + shaping
    }

    /// Shape a sequence of rewards in a trajectory.
    pub fn shape_trajectory(&self, rewards: &[f64], states: &[Vec<f64>]) -> Result<Vec<f64>> {
        if rewards.len() + 1 != states.len() && rewards.len() != states.len() {
            return Err(TensorError::invalid_argument_op(
                "RewardShaping::shape_trajectory",
                "rewards and states length mismatch (expected |rewards| == |states| or |states|-1)",
            ));
        }
        let mut shaped = Vec::with_capacity(rewards.len());
        for i in 0..rewards.len() {
            let next_idx = (i + 1).min(states.len() - 1);
            shaped.push(self.shape_reward(rewards[i], &states[i], &states[next_idx]));
        }
        Ok(shaped)
    }

    /// Get the potential type.
    pub fn potential_type(&self) -> &IrlPotentialType {
        &self.potential_type
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 11: IrlMetrics — Evaluation Metrics for IRL
// ─────────────────────────────────────────────────────────────────────────────

/// IRL evaluation report.
#[derive(Debug, Clone)]
pub struct IrlReport {
    /// Expected value difference between expert and learned policies.
    pub evd: f64,
    /// Feature expectation error (L2 norm of difference).
    pub feature_expectation_error: f64,
    /// Pearson correlation between learned and ground-truth rewards.
    pub reward_correlation: f64,
    /// Entropy of the learned policy.
    pub policy_entropy: f64,
    /// Success rate of the learned policy.
    pub success_rate: f64,
}

impl std::fmt::Display for IrlReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IrlReport {{ EVD: {:.4}, FEE: {:.4}, Corr: {:.4}, Entropy: {:.4}, Success: {:.2}% }}",
            self.evd,
            self.feature_expectation_error,
            self.reward_correlation,
            self.policy_entropy,
            self.success_rate * 100.0,
        )
    }
}

/// Collection of evaluation metrics for IRL algorithms.
pub struct IrlMetrics;

impl IrlMetrics {
    /// Expected Value Difference: |V_expert - V_learned| under the true reward.
    ///
    /// Lower is better. Measures how close the learned policy's expected return
    /// is to the expert's expected return.
    pub fn expected_value_difference(
        expert_returns: &[f64],
        learned_returns: &[f64],
    ) -> Result<f64> {
        if expert_returns.is_empty() || learned_returns.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IrlMetrics::expected_value_difference",
                "return vectors must not be empty",
            ));
        }
        let expert_mean = expert_returns.iter().sum::<f64>() / expert_returns.len() as f64;
        let learned_mean = learned_returns.iter().sum::<f64>() / learned_returns.len() as f64;
        Ok((expert_mean - learned_mean).abs())
    }

    /// Feature expectation error: ||f_expert - f_learned||_2.
    pub fn feature_expectation_error(f_expert: &[f64], f_learned: &[f64]) -> Result<f64> {
        if f_expert.len() != f_learned.len() {
            return Err(TensorError::invalid_argument_op(
                "IrlMetrics::feature_expectation_error",
                "feature vectors must have the same length",
            ));
        }
        let diff: Vec<f64> = f_expert
            .iter()
            .zip(f_learned.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Ok(irl_l2_norm(&diff))
    }

    /// Pearson correlation between learned and ground-truth reward vectors.
    pub fn reward_correlation(learned_rewards: &[f64], true_rewards: &[f64]) -> Result<f64> {
        if learned_rewards.len() != true_rewards.len() {
            return Err(TensorError::invalid_argument_op(
                "IrlMetrics::reward_correlation",
                "reward vectors must have the same length",
            ));
        }
        let n = learned_rewards.len();
        if n < 2 {
            return Ok(0.0);
        }
        let mean_l = learned_rewards.iter().sum::<f64>() / n as f64;
        let mean_t = true_rewards.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_l = 0.0;
        let mut var_t = 0.0;
        for i in 0..n {
            let dl = learned_rewards[i] - mean_l;
            let dt = true_rewards[i] - mean_t;
            cov += dl * dt;
            var_l += dl * dl;
            var_t += dt * dt;
        }
        let denom = (var_l * var_t).sqrt();
        if denom < 1e-15 {
            Ok(0.0)
        } else {
            Ok(cov / denom)
        }
    }

    /// Compute the entropy of a stochastic policy.
    ///
    /// `policy[s]` is a probability distribution over actions at state `s`.
    pub fn policy_entropy(policy: &[Vec<f64>]) -> f64 {
        if policy.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for dist in policy {
            for &p in dist {
                if p > 1e-15 {
                    total -= p * p.ln();
                }
            }
        }
        total / policy.len() as f64
    }

    /// Compute the success rate: fraction of episodes where total return exceeds
    /// a threshold.
    pub fn success_rate(episode_returns: &[f64], threshold: f64) -> f64 {
        if episode_returns.is_empty() {
            return 0.0;
        }
        let successes = episode_returns.iter().filter(|&&r| r >= threshold).count();
        successes as f64 / episode_returns.len() as f64
    }

    /// Generate a full IRL evaluation report.
    pub fn evaluate(
        expert_returns: &[f64],
        learned_returns: &[f64],
        f_expert: &[f64],
        f_learned: &[f64],
        learned_rewards: &[f64],
        true_rewards: &[f64],
        policy: &[Vec<f64>],
        success_threshold: f64,
    ) -> Result<IrlReport> {
        let evd = Self::expected_value_difference(expert_returns, learned_returns)?;
        let fee = Self::feature_expectation_error(f_expert, f_learned)?;
        let corr = Self::reward_correlation(learned_rewards, true_rewards)?;
        let entropy = Self::policy_entropy(policy);
        let success = Self::success_rate(learned_returns, success_threshold);

        Ok(IrlReport {
            evd,
            feature_expectation_error: fee,
            reward_correlation: corr,
            policy_entropy: entropy,
            success_rate: success,
        })
    }
}
