//! # Causal Reinforcement Learning (CRL)
//!
//! This module combines structural causal models (SCMs) with reinforcement learning
//! for counterfactual reasoning, causal policy gradients, and causal model-based RL.
//!
//! ## Key Concepts
//!
//! - **Structural Causal Model (SCM)**: Variables connected by functional mechanisms
//!   with exogenous noise, enabling interventional and counterfactual queries.
//! - **Causal Policy Gradient**: Reduces variance by masking irrelevant state features
//!   per action based on causal graph structure.
//! - **Counterfactual Data Augmentation**: Abduction-Action-Prediction for generating
//!   synthetic experience from causal world model.
//! - **Invariant Policy Learning**: IRM-style regularization across environments to
//!   identify causally relevant features.
//!
//! ## References
//!
//! - Buesing et al. (2019) "Woulda, Coulda, Shoulda"
//! - Schölkopf et al. (2021) "Towards Causal Representation Learning"
//! - Peters et al. (2016) "Causal Inference by Using Invariant Prediction"
//! - Lu et al. (2021) "Regret Minimization for Causal Inference on Large Observed Data"

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error Type
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for Causal RL operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CrlError {
    /// SCM is invalid (e.g., cyclic graph, mismatched dimensions).
    InvalidSCM(String),
    /// Numerical computation failed (e.g., singular matrix, NaN).
    NumericalError(String),
    /// Action index out of range or otherwise invalid.
    InvalidAction(String),
    /// Insufficient data for the operation.
    InsufficientData(String),
}

impl fmt::Display for CrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrlError::InvalidSCM(msg) => write!(f, "InvalidSCM: {}", msg),
            CrlError::NumericalError(msg) => write!(f, "NumericalError: {}", msg),
            CrlError::InvalidAction(msg) => write!(f, "InvalidAction: {}", msg),
            CrlError::InsufficientData(msg) => write!(f, "InsufficientData: {}", msg),
        }
    }
}

impl std::error::Error for CrlError {}

// ─────────────────────────────────────────────────────────────────────────────
// Seeded RNG helpers (no external rand crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG: returns uniform float in [0, 1).
#[inline]
pub fn crl_rand01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    // Map to [0, 1) by dividing by 2^64
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform: returns standard normal sample.
#[inline]
pub fn crl_randn(seed: &mut u64) -> f64 {
    let u1 = crl_rand01(seed).max(1e-15);
    let u2 = crl_rand01(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  CrlStructuralCausalModel
// ─────────────────────────────────────────────────────────────────────────────

/// A single variable in the SCM (state, action, reward, or next-state component).
#[derive(Debug, Clone)]
pub struct CrlVariable {
    pub name: String,
    pub dim: usize,
}

/// Linear causal mechanism: V_out = W @ V_parents + bias + noise_std * U.
#[derive(Debug, Clone)]
pub struct CrlCausalMechanism {
    /// Weight matrix [out_dim × in_dim] stored row-major.
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub noise_std: f64,
}

impl CrlCausalMechanism {
    /// Create a new mechanism with given dimensions (zero-initialised).
    pub fn new(out_dim: usize, in_dim: usize, noise_std: f64) -> Self {
        CrlCausalMechanism {
            weights: vec![vec![0.0; in_dim]; out_dim],
            bias: vec![0.0; out_dim],
            noise_std,
        }
    }

    /// Initialise weights with small random values using seeded RNG.
    pub fn init_random(&mut self, scale: f64, seed: &mut u64) {
        for row in &mut self.weights {
            for w in row.iter_mut() {
                *w = crl_randn(seed) * scale;
            }
        }
        for b in &mut self.bias {
            *b = crl_randn(seed) * scale * 0.1;
        }
    }

    /// Apply mechanism: returns W @ input + bias (without noise).
    pub fn apply_deterministic(&self, input: &[f64]) -> Vec<f64> {
        let out_dim = self.weights.len();
        let mut out = self.bias.clone();
        for (i, row) in self.weights.iter().enumerate() {
            if i < out_dim {
                for (j, &w) in row.iter().enumerate() {
                    if j < input.len() {
                        out[i] += w * input[j];
                    }
                }
            }
        }
        out
    }

    /// Apply mechanism with additive Gaussian noise.
    pub fn apply_stochastic(&self, input: &[f64], seed: &mut u64) -> Vec<f64> {
        let mut out = self.apply_deterministic(input);
        if self.noise_std > 0.0 {
            for v in &mut out {
                *v += self.noise_std * crl_randn(seed);
            }
        }
        out
    }
}

/// SCM for MDP environments.
///
/// Variable layout (indices):
///   0 → state      (state_dim)
///   1 → action     (action_dim)
///   2 → reward     (1)
///   3 → next_state (state_dim)
#[derive(Debug, Clone)]
pub struct CrlStructuralCausalModel {
    pub variables: Vec<CrlVariable>,
    pub mechanisms: Vec<CrlCausalMechanism>,
    /// adjacency\[i\]\[j\] = true means variable j causally precedes variable i.
    pub adjacency: Vec<Vec<bool>>,
    pub var_dims: Vec<usize>,
    pub n_vars: usize,
}

impl CrlStructuralCausalModel {
    /// Create an SCM for a standard MDP with linear mechanisms.
    ///
    /// Variable layout:
    ///   \[0\] state (state_dim), \[1\] action (action_dim),
    ///   \[2\] reward (1),        \[3\] next_state (state_dim)
    pub fn new_mdp_scm(state_dim: usize, action_dim: usize) -> Self {
        let var_dims = vec![state_dim, action_dim, 1, state_dim];
        let n_vars = 4;

        let variables = vec![
            CrlVariable {
                name: "state".into(),
                dim: state_dim,
            },
            CrlVariable {
                name: "action".into(),
                dim: action_dim,
            },
            CrlVariable {
                name: "reward".into(),
                dim: 1,
            },
            CrlVariable {
                name: "next_state".into(),
                dim: state_dim,
            },
        ];

        // adjacency[i][j] = j → i
        // reward depends on state (0) and action (1)
        // next_state depends on state (0) and action (1)
        let mut adjacency = vec![vec![false; n_vars]; n_vars];
        adjacency[2][0] = true; // reward ← state
        adjacency[2][1] = true; // reward ← action
        adjacency[3][0] = true; // next_state ← state
        adjacency[3][1] = true; // next_state ← action

        // Mechanism for reward: input = [state; action], output = scalar
        let sa_dim = state_dim + action_dim;
        let mut reward_mech = CrlCausalMechanism::new(1, sa_dim, 0.05);
        // default: equal weight on state dims for reward
        for j in 0..state_dim.min(sa_dim) {
            reward_mech.weights[0][j] = 1.0 / state_dim.max(1) as f64;
        }

        // Mechanism for next_state: input = [state; action], output = state_dim
        let mut next_mech = CrlCausalMechanism::new(state_dim, sa_dim, 0.1);
        // default: identity on state part + small action influence
        for i in 0..state_dim {
            if i < sa_dim {
                next_mech.weights[i][i] = 0.9; // near-identity on state
            }
            if state_dim + i < sa_dim {
                next_mech.weights[i][state_dim + i % action_dim] = 0.1;
            }
        }

        // mechanisms indexed by variable (unused for var 0,1; real for 2,3)
        let dummy_reward_mech = CrlCausalMechanism::new(1, 0, 0.0);
        let dummy_state_mech = CrlCausalMechanism::new(state_dim, 0, 0.0);
        let mechanisms = vec![
            dummy_state_mech,                            // 0: state — exogenous, no mechanism
            CrlCausalMechanism::new(action_dim, 0, 0.0), // 1: action — exogenous
            reward_mech,                                 // 2: reward
            next_mech,                                   // 3: next_state
        ];
        drop(dummy_reward_mech);

        CrlStructuralCausalModel {
            variables,
            mechanisms,
            adjacency,
            var_dims,
            n_vars,
        }
    }

    /// Return the concatenated parent inputs for a given variable.
    fn collect_parents(&self, var_idx: usize, obs: &[f64], action: &[f64]) -> Vec<f64> {
        let mut parents = Vec::new();
        for j in 0..self.n_vars {
            if self.adjacency[var_idx][j] {
                match j {
                    0 => parents.extend_from_slice(obs),
                    1 => parents.extend_from_slice(action),
                    _ => {}
                }
            }
        }
        parents
    }

    /// Sample (next_state, reward) from SCM given current obs and action.
    pub fn sample(&self, obs: &[f64], action: &[f64], seed: &mut u64) -> (Vec<f64>, f64) {
        // Reward: mechanism[2], parents = [state, action]
        let r_parents = self.collect_parents(2, obs, action);
        let r_out = self.mechanisms[2].apply_stochastic(&r_parents, seed);
        let reward = r_out.first().copied().unwrap_or(0.0);

        // Next state: mechanism[3], parents = [state, action]
        let ns_parents = self.collect_parents(3, obs, action);
        let next_state = self.mechanisms[3].apply_stochastic(&ns_parents, seed);

        (next_state, reward)
    }

    /// Interventional query: do(V_{variable_idx} = value), propagate forward.
    ///
    /// Currently supports intervening on reward (2) or next_state (3).
    pub fn intervene(
        &self,
        variable_idx: usize,
        value: &[f64],
        obs: &[f64],
        action: &[f64],
        seed: &mut u64,
    ) -> (Vec<f64>, f64) {
        match variable_idx {
            2 => {
                // Fix reward, sample next_state normally
                let ns_parents = self.collect_parents(3, obs, action);
                let next_state = self.mechanisms[3].apply_stochastic(&ns_parents, seed);
                let reward = value.first().copied().unwrap_or(0.0);
                (next_state, reward)
            }
            3 => {
                // Fix next_state, sample reward normally
                let r_parents = self.collect_parents(2, obs, action);
                let r_out = self.mechanisms[2].apply_stochastic(&r_parents, seed);
                let reward = r_out.first().copied().unwrap_or(0.0);
                (value.to_vec(), reward)
            }
            _ => {
                // For state/action interventions: treat as standard sample
                self.sample(obs, action, seed)
            }
        }
    }

    /// Counterfactual query using Pearl's three steps:
    ///   1. **Abduction**: Infer exogenous noise U from factual observation.
    ///   2. **Action**: Intervene do(A = counterfactual_action).
    ///   3. **Prediction**: Propagate forward with same U.
    pub fn counterfactual(
        &self,
        obs: &[f64],
        action: &[f64],
        factual_next: &[f64],
        factual_reward: f64,
        counterfactual_action: &[f64],
        _seed: &mut u64,
    ) -> (Vec<f64>, f64) {
        // Step 1: Abduction — infer noise
        // U_next = factual_next - W_ns @ [obs; action]
        let ns_parents_fact = self.collect_parents(3, obs, action);
        let ns_det_fact = self.mechanisms[3].apply_deterministic(&ns_parents_fact);
        let state_dim = self.var_dims[0];
        let mut u_next = vec![0.0; state_dim];
        for i in 0..state_dim.min(factual_next.len()).min(ns_det_fact.len()) {
            u_next[i] = factual_next[i] - ns_det_fact[i];
        }

        // Reward noise
        let r_parents_fact = self.collect_parents(2, obs, action);
        let r_det_fact = self.mechanisms[2].apply_deterministic(&r_parents_fact);
        let u_reward = factual_reward - r_det_fact.first().copied().unwrap_or(0.0);

        // Step 2: Action — intervene with counterfactual action
        // Step 3: Prediction — propagate with same U
        let ns_parents_cf = self.collect_parents(3, obs, counterfactual_action);
        let mut cf_next = self.mechanisms[3].apply_deterministic(&ns_parents_cf);
        for i in 0..cf_next.len().min(u_next.len()) {
            cf_next[i] += u_next[i];
        }

        let r_parents_cf = self.collect_parents(2, obs, counterfactual_action);
        let r_det_cf = self.mechanisms[2].apply_deterministic(&r_parents_cf);
        let cf_reward = r_det_cf.first().copied().unwrap_or(0.0) + u_reward;

        (cf_next, cf_reward)
    }

    /// Return reference to the causal adjacency matrix.
    pub fn causal_graph_matrix(&self) -> &Vec<Vec<bool>> {
        &self.adjacency
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  CrlCausalPolicyGradient
// ─────────────────────────────────────────────────────────────────────────────

/// Causal policy gradient that exploits the causal structure to reduce variance.
///
/// For each action dimension `a`, only the state features that causally influence
/// it (per `causal_mask[a]`) contribute to the gradient — zeroing irrelevant terms
/// reduces gradient variance (Lattimore et al. 2016, Buesing et al. 2019).
#[derive(Debug, Clone)]
pub struct CrlCausalPolicyGradient {
    /// Linear policy weights [action_dim × state_dim].
    pub policy_weights: Vec<Vec<f64>>,
    pub policy_bias: Vec<f64>,
    /// causal_mask\[a\]\[s\] = true when state_s causally affects action_a.
    pub causal_mask: Vec<Vec<bool>>,
    pub state_dim: usize,
    pub action_dim: usize,
    pub lr: f64,
    pub gamma: f64,
}

impl CrlCausalPolicyGradient {
    /// Create with uniform causal mask (all features relevant to all actions).
    pub fn new(state_dim: usize, action_dim: usize, lr: f64, gamma: f64) -> Self {
        let policy_weights = vec![vec![0.0; state_dim]; action_dim];
        let policy_bias = vec![0.0; action_dim];
        let causal_mask = vec![vec![true; state_dim]; action_dim];
        CrlCausalPolicyGradient {
            policy_weights,
            policy_bias,
            causal_mask,
            state_dim,
            action_dim,
            lr,
            gamma,
        }
    }

    /// Compute softmax policy probabilities from state.
    pub fn policy(&self, state: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = (0..self.action_dim)
            .map(|a| {
                let mut logit = self.policy_bias[a];
                for s in 0..self.state_dim.min(state.len()) {
                    logit += self.policy_weights[a][s] * state[s];
                }
                logit
            })
            .collect();
        softmax(&logits)
    }

    /// Compute causal REINFORCE gradient for the policy parameters.
    ///
    /// For each action `a` taken at time `t`:
    ///   grad\[a\]\[s\] = mask\[a\]\[s\] * G_t * (1{a_t = a} - π(a|s_t)) * s_t\[s\]
    ///
    /// Returns gradient tensor [action_dim × state_dim].
    pub fn causal_gradient(
        &self,
        trajectory: &[(Vec<f64>, usize, f64)],
        returns: &[f64],
    ) -> Vec<Vec<f64>> {
        let mut grad = vec![vec![0.0; self.state_dim]; self.action_dim];
        let t_len = trajectory.len().min(returns.len());

        for t in 0..t_len {
            let (ref state, action, _) = trajectory[t];
            let g_t = returns[t];
            let pi = self.policy(state);

            for a in 0..self.action_dim {
                let indicator = if a == action { 1.0 } else { 0.0 };
                let score = (indicator - pi[a]) * g_t;

                for s in 0..self.state_dim.min(state.len()) {
                    if self.causal_mask[a][s] {
                        grad[a][s] += score * state[s];
                    }
                }
            }
        }

        // Normalise by trajectory length
        if t_len > 0 {
            let scale = 1.0 / t_len as f64;
            for row in &mut grad {
                for g in row.iter_mut() {
                    *g *= scale;
                }
            }
        }
        grad
    }

    /// Compute discounted returns G_t = Σ_{k≥0} γ^k r_{t+k}.
    pub fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
        let n = rewards.len();
        let mut returns = vec![0.0; n];
        let mut running = 0.0;
        for t in (0..n).rev() {
            running = rewards[t] + gamma * running;
            returns[t] = running;
        }
        returns
    }

    /// Perform one REINFORCE update step; returns mean episode return.
    pub fn update(&mut self, trajectory: &[(Vec<f64>, usize, f64)]) -> f64 {
        if trajectory.is_empty() {
            return 0.0;
        }
        let rewards: Vec<f64> = trajectory.iter().map(|(_, _, r)| *r).collect();
        let returns = Self::compute_returns(&rewards, self.gamma);
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

        let grad = self.causal_gradient(trajectory, &returns);

        // Ascent step (maximise return)
        for a in 0..self.action_dim {
            self.policy_bias[a] +=
                self.lr * grad[a].iter().sum::<f64>() / self.state_dim.max(1) as f64;
            for s in 0..self.state_dim {
                self.policy_weights[a][s] += self.lr * grad[a][s];
            }
        }
        mean_return
    }

    /// Infer causal mask from feature importance scores.
    ///
    /// `state_importance[a][s]` is the importance of feature s for action a;
    /// features with magnitude above the mean are marked as causally relevant.
    pub fn infer_causal_mask(&mut self, state_importance: &[Vec<f64>]) {
        for a in 0..self.action_dim.min(state_importance.len()) {
            let row = &state_importance[a];
            let n = row.len();
            if n == 0 {
                continue;
            }
            let mean_imp = row.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
            let threshold = mean_imp * 0.5;
            for s in 0..self.state_dim.min(n) {
                self.causal_mask[a][s] = row[s].abs() >= threshold;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  CrlCausalModelBasedRL
// ─────────────────────────────────────────────────────────────────────────────

/// Modular causal world model: each state dimension has its own causal mechanism.
#[derive(Debug, Clone)]
pub struct CrlModularModel {
    /// One mechanism per state dimension (parents = [state; action]).
    pub state_mechanisms: Vec<CrlCausalMechanism>,
    /// Reward mechanism (input = [state; action], output = scalar).
    pub reward_mechanism: CrlCausalMechanism,
    pub state_dim: usize,
    pub action_dim: usize,
}

impl CrlModularModel {
    /// Create a new modular model with small random initialisation.
    pub fn new(state_dim: usize, action_dim: usize, seed: &mut u64) -> Self {
        let sa_dim = state_dim + action_dim;
        let mut state_mechanisms: Vec<CrlCausalMechanism> = (0..state_dim)
            .map(|_| CrlCausalMechanism::new(1, sa_dim, 0.05))
            .collect();

        for (i, mech) in state_mechanisms.iter_mut().enumerate() {
            // Near-identity initialisation for state part
            if i < sa_dim {
                mech.weights[0][i] = 0.9;
            }
            mech.init_random(0.05, seed);
        }

        let mut reward_mechanism = CrlCausalMechanism::new(1, sa_dim, 0.05);
        reward_mechanism.init_random(0.1, seed);

        CrlModularModel {
            state_mechanisms,
            reward_mechanism,
            state_dim,
            action_dim,
        }
    }

    /// Predict next_state and reward deterministically (for planning).
    pub fn predict(&self, state: &[f64], action: &[f64]) -> (Vec<f64>, f64) {
        let sa: Vec<f64> = state.iter().chain(action.iter()).copied().collect();
        let next_state: Vec<f64> = self
            .state_mechanisms
            .iter()
            .map(|mech| {
                mech.apply_deterministic(&sa)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        let reward_out = self.reward_mechanism.apply_deterministic(&sa);
        let reward = reward_out.first().copied().unwrap_or(0.0);
        (next_state, reward)
    }

    /// Predict with stochastic noise.
    pub fn predict_stochastic(
        &self,
        state: &[f64],
        action: &[f64],
        seed: &mut u64,
    ) -> (Vec<f64>, f64) {
        let sa: Vec<f64> = state.iter().chain(action.iter()).copied().collect();
        let next_state: Vec<f64> = self
            .state_mechanisms
            .iter()
            .map(|mech| {
                mech.apply_stochastic(&sa, seed)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();
        let reward_out = self.reward_mechanism.apply_stochastic(&sa, seed);
        let reward = reward_out.first().copied().unwrap_or(0.0);
        (next_state, reward)
    }

    /// MSE gradient update for a single state dimension mechanism.
    fn update_state_module(
        &mut self,
        dim_idx: usize,
        sa_input: &[f64],
        target: f64,
        lr: f64,
    ) -> f64 {
        let pred = self.state_mechanisms[dim_idx]
            .apply_deterministic(sa_input)
            .first()
            .copied()
            .unwrap_or(0.0);
        let err = pred - target;
        // Gradient: d_MSE/d_w = err * x
        let in_len = sa_input.len();
        for j in 0..in_len.min(self.state_mechanisms[dim_idx].weights[0].len()) {
            self.state_mechanisms[dim_idx].weights[0][j] -= lr * err * sa_input[j];
        }
        self.state_mechanisms[dim_idx].bias[0] -= lr * err;
        err * err
    }

    /// MSE gradient update for reward mechanism.
    fn update_reward_module(&mut self, sa_input: &[f64], target: f64, lr: f64) -> f64 {
        let pred = self
            .reward_mechanism
            .apply_deterministic(sa_input)
            .first()
            .copied()
            .unwrap_or(0.0);
        let err = pred - target;
        let in_len = sa_input.len();
        for j in 0..in_len.min(self.reward_mechanism.weights[0].len()) {
            self.reward_mechanism.weights[0][j] -= lr * err * sa_input[j];
        }
        self.reward_mechanism.bias[0] -= lr * err;
        err * err
    }
}

/// Causal model-based RL: learn a modular causal world model and use it for
/// Dyna-style planning.
#[derive(Debug, Clone)]
pub struct CrlCausalModelBasedRL {
    pub world_model: CrlModularModel,
    /// Linear softmax policy weights [action_dim × state_dim].
    pub policy_weights: Vec<Vec<f64>>,
    pub policy_bias: Vec<f64>,
    /// Linear value function weights \[state_dim\].
    pub value_weights: Vec<f64>,
    pub state_dim: usize,
    pub action_dim: usize,
    pub model_lr: f64,
    pub policy_lr: f64,
    pub gamma: f64,
    pub rollout_horizon: usize,
}

impl CrlCausalModelBasedRL {
    /// Create a new CrlCausalModelBasedRL instance.
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        let mut seed = 0xDEAD_BEEF_1337u64;
        let world_model = CrlModularModel::new(state_dim, action_dim, &mut seed);
        CrlCausalModelBasedRL {
            world_model,
            policy_weights: vec![vec![0.0; state_dim]; action_dim],
            policy_bias: vec![0.0; action_dim],
            value_weights: vec![0.0; state_dim],
            state_dim,
            action_dim,
            model_lr: 1e-3,
            policy_lr: 1e-3,
            gamma: 0.99,
            rollout_horizon: 5,
        }
    }

    /// Update the world model on observed transitions.
    ///
    /// Returns mean squared prediction error across all transitions.
    pub fn update_world_model(
        &mut self,
        transitions: &[(Vec<f64>, Vec<f64>, f64, Vec<f64>)],
    ) -> f64 {
        if transitions.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0;
        let n = transitions.len() as f64;

        for (state, action, reward, next_state) in transitions {
            let sa: Vec<f64> = state.iter().chain(action.iter()).copied().collect();
            // Update each state dimension independently (modularity principle)
            for d in 0..self.state_dim {
                let target = next_state.get(d).copied().unwrap_or(0.0);
                total_loss += self
                    .world_model
                    .update_state_module(d, &sa, target, self.model_lr);
            }
            // Update reward module
            total_loss += self
                .world_model
                .update_reward_module(&sa, *reward, self.model_lr);
        }

        total_loss / (n * (self.state_dim as f64 + 1.0))
    }

    /// Roll out the world model from a start state for `horizon` steps.
    ///
    /// Returns sequence of (state, reward) pairs.
    pub fn model_rollout(
        &self,
        start_state: &[f64],
        horizon: usize,
        seed: &mut u64,
    ) -> Vec<(Vec<f64>, f64)> {
        let mut trajectory = Vec::with_capacity(horizon);
        let mut state = start_state.to_vec();

        for _ in 0..horizon {
            // Sample action from policy
            let pi = self.policy(&state);
            let action_idx = sample_categorical(&pi, seed);
            let mut action_vec = vec![0.0; self.action_dim];
            if action_idx < self.action_dim {
                action_vec[action_idx] = 1.0;
            }

            let (next_state, reward) =
                self.world_model
                    .predict_stochastic(&state, &action_vec, seed);
            trajectory.push((state.clone(), reward));
            state = next_state;
        }
        trajectory
    }

    /// Softmax policy over action space.
    pub fn policy(&self, state: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = (0..self.action_dim)
            .map(|a| {
                let mut logit = self.policy_bias[a];
                for s in 0..self.state_dim.min(state.len()) {
                    logit += self.policy_weights[a][s] * state[s];
                }
                logit
            })
            .collect();
        softmax(&logits)
    }

    /// Linear value function: V(s) = w · s.
    pub fn value(&self, state: &[f64]) -> f64 {
        let mut v = 0.0;
        for s in 0..self.state_dim.min(state.len()) {
            v += self.value_weights[s] * state[s];
        }
        v
    }

    /// Dyna-style planning: model rollout + policy gradient update.
    ///
    /// Returns total reward collected in the rollout.
    pub fn plan_update(&mut self, start_state: &[f64], seed: &mut u64) -> f64 {
        let horizon = self.rollout_horizon;
        let rollout = self.model_rollout(start_state, horizon, seed);
        let total_reward: f64 = rollout.iter().map(|(_, r)| r).sum();

        // Compute returns and update policy via REINFORCE
        let rewards: Vec<f64> = rollout.iter().map(|(_, r)| *r).collect();
        let n = rewards.len();
        let mut returns = vec![0.0; n];
        let mut running = 0.0;
        let gamma = self.gamma;
        for t in (0..n).rev() {
            running = rewards[t] + gamma * running;
            returns[t] = running;
        }

        // Simple policy gradient step on each rollout state
        for (t, (state, _)) in rollout.iter().enumerate() {
            let g_t = returns[t];
            let pi = self.policy(state);
            // Pseudo gradient: push towards higher-value actions
            let v_s = self.value(state);
            let advantage = g_t - v_s;

            for a in 0..self.action_dim {
                let score = advantage
                    * (if a
                        == pi
                            .iter()
                            .enumerate()
                            .max_by(|(_, x), (_, y)| {
                                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    {
                        1.0
                    } else {
                        0.0
                    } - pi[a]);
                for s in 0..self.state_dim.min(state.len()) {
                    self.policy_weights[a][s] += self.policy_lr * score * state[s];
                }
                self.policy_bias[a] += self.policy_lr * score;
            }

            // Value update
            let v_err = g_t - v_s;
            for s in 0..self.state_dim.min(state.len()) {
                self.value_weights[s] += self.policy_lr * v_err * state[s];
            }
        }

        total_reward
    }
}


/// Numerically stable softmax.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / sum).collect()
}

/// Sample from categorical distribution.
pub(crate) fn sample_categorical(probs: &[f64], seed: &mut u64) -> usize {
    let u = crl_rand01(seed);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

pub mod extensions;
pub use extensions::*;

#[cfg(test)]
mod tests;
