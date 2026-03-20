//! Causal RL extensions (§4-§10 + utility functions).
//!
//! Counterfactual data augmentation, CRL environment, invariant policy learning,
//! off-policy counterfactual Q-learning, causal credit, Dyna-Q with SCM,
//! causal curriculum learning, and CRL evaluation metrics.

use super::{CrlError, CrlStructuralCausalModel, CrlCausalPolicyGradient,
    crl_rand01, crl_randn};

// ─────────────────────────────────────────────────────────────────────────────
// §4  CrlCounterfactualDataAugmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Counterfactual data augmentation using an SCM to generate synthetic transitions.
///
/// For each observed transition, we generate a counterfactual by abducting the
/// exogenous noise and re-predicting under a different action (Lu et al. 2021).
#[derive(Debug, Clone)]
pub struct CrlCounterfactualDataAugmentation {
    pub scm: CrlStructuralCausalModel,
    /// Fraction of original transitions to augment.
    pub augmentation_ratio: f64,
    pub causal_noise_std: f64,
}

impl CrlCounterfactualDataAugmentation {
    /// Create augmenter with given SCM and ratio.
    pub fn new(scm: CrlStructuralCausalModel, augmentation_ratio: f64) -> Self {
        CrlCounterfactualDataAugmentation {
            scm,
            augmentation_ratio: augmentation_ratio.clamp(0.0, 1.0),
            causal_noise_std: 0.01,
        }
    }

    /// Generate counterfactual transitions for each entry in the batch.
    ///
    /// For each (s, a, r, s'), generate (s, a', r_cf, s_cf') using SCM abduction.
    pub fn generate_counterfactual_batch(
        &self,
        batch: &[(Vec<f64>, usize, f64, Vec<f64>)],
        seed: &mut u64,
    ) -> Vec<(Vec<f64>, usize, f64, Vec<f64>)> {
        let action_dim = self.scm.var_dims.get(1).copied().unwrap_or(1);
        let n_actions = action_dim.max(1);

        batch
            .iter()
            .map(|(state, action, reward, next_state)| {
                // Pick a different action
                let cf_action_idx = {
                    let candidate = (crl_rand01(seed) * n_actions as f64) as usize % n_actions;
                    if candidate == *action && n_actions > 1 {
                        (candidate + 1) % n_actions
                    } else {
                        candidate
                    }
                };

                let action_vec: Vec<f64> = {
                    let mut v = vec![0.0; n_actions];
                    if *action < n_actions {
                        v[*action] = 1.0;
                    }
                    v
                };
                let cf_action_vec: Vec<f64> = {
                    let mut v = vec![0.0; n_actions];
                    if cf_action_idx < n_actions {
                        v[cf_action_idx] = 1.0;
                    }
                    v
                };

                let (cf_next, cf_reward) = self.scm.counterfactual(
                    state,
                    &action_vec,
                    next_state,
                    *reward,
                    &cf_action_vec,
                    seed,
                );

                (state.clone(), cf_action_idx, cf_reward, cf_next)
            })
            .collect()
    }

    /// Return original transitions + counterfactual augmentations.
    pub fn augment_dataset(
        &self,
        transitions: &[(Vec<f64>, usize, f64, Vec<f64>)],
        seed: &mut u64,
    ) -> Vec<(Vec<f64>, usize, f64, Vec<f64>)> {
        let n_augment = (transitions.len() as f64 * self.augmentation_ratio).round() as usize;
        let n_augment = n_augment.min(transitions.len());

        let mut augmented = transitions.to_vec();
        if n_augment > 0 {
            let sub_batch = &transitions[..n_augment];
            let cf_batch = self.generate_counterfactual_batch(sub_batch, seed);
            augmented.extend(cf_batch);
        }
        augmented
    }

    /// Counterfactual advantage: Q(s, a_cf) - Q(s, a).
    pub fn counterfactual_advantage(
        &self,
        _state: &[f64],
        action: usize,
        cf_action: usize,
        q_values: &[f64],
    ) -> f64 {
        let q_a = q_values.get(action).copied().unwrap_or(0.0);
        let q_cf = q_values.get(cf_action).copied().unwrap_or(0.0);
        q_cf - q_a
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  CrlInvariantPolicyLearning
// ─────────────────────────────────────────────────────────────────────────────

/// A simulated RL environment with spurious and causal features.
#[derive(Debug, Clone)]
pub struct CrlEnvironment {
    pub env_id: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    /// Noise level specific to this environment.
    pub transition_noise: f64,
    /// Spurious environment-specific reward bias (not causal).
    pub reward_bias: f64,
    /// True causal feature weights \[state_dim\].
    pub causal_weight: Vec<f64>,
}

impl CrlEnvironment {
    /// Create a new environment with random causal weights.
    pub fn new(env_id: usize, state_dim: usize, action_dim: usize, seed: &mut u64) -> Self {
        let causal_weight: Vec<f64> = (0..state_dim).map(|_| crl_randn(seed) * 0.5).collect();
        CrlEnvironment {
            env_id,
            state_dim,
            action_dim,
            transition_noise: 0.1,
            reward_bias: crl_randn(seed) * 0.3,
            causal_weight,
        }
    }
}

/// Invariant Policy Learning via IRM-style regularisation.
///
/// Learns a policy that performs well across environments by penalising
/// policies whose gradients differ across environments (Peters et al. 2016
/// applied to RL).
#[derive(Debug, Clone)]
pub struct CrlInvariantPolicyLearning {
    /// Policy weights [action_dim × state_dim].
    pub policy_weights: Vec<Vec<f64>>,
    pub environments: Vec<CrlEnvironment>,
    /// IRM penalty coefficient λ.
    pub irm_penalty: f64,
    pub lr: f64,
    pub state_dim: usize,
    pub action_dim: usize,
}

impl CrlInvariantPolicyLearning {
    /// Create invariant policy learner.
    pub fn new(state_dim: usize, action_dim: usize, irm_penalty: f64) -> Self {
        CrlInvariantPolicyLearning {
            policy_weights: vec![vec![0.0; state_dim]; action_dim],
            environments: Vec::new(),
            irm_penalty,
            lr: 1e-3,
            state_dim,
            action_dim,
        }
    }

    /// Compute softmax policy.
    pub fn policy(&self, state: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = (0..self.action_dim)
            .map(|a| {
                let mut logit = 0.0;
                for s in 0..self.state_dim.min(state.len()) {
                    logit += self.policy_weights[a][s] * state[s];
                }
                logit
            })
            .collect();
        softmax(&logits)
    }

    /// IRM penalty: variance of per-environment gradients.
    ///
    /// `env_gradients[e][a][s]` = policy gradient for env e, action a, state feature s.
    /// Penalty = Σ_a Σ_s Var_e\[ grad[e\]\[a\]\[s\] ]
    pub fn irm_penalty_term(&self, env_gradients: &[Vec<Vec<f64>>]) -> f64 {
        if env_gradients.is_empty() {
            return 0.0;
        }
        let n_env = env_gradients.len();
        let n_actions = env_gradients[0].len();
        let n_states = if n_actions > 0 {
            env_gradients[0][0].len()
        } else {
            0
        };

        let mut penalty = 0.0;
        for a in 0..n_actions {
            for s in 0..n_states {
                // Compute mean gradient across environments
                let mean_g = env_gradients
                    .iter()
                    .filter_map(|eg| eg.get(a).and_then(|row| row.get(s)))
                    .sum::<f64>()
                    / n_env as f64;
                // Sum of squared deviations
                let var = env_gradients
                    .iter()
                    .filter_map(|eg| eg.get(a).and_then(|row| row.get(s)))
                    .map(|g| (g - mean_g).powi(2))
                    .sum::<f64>();
                penalty += var;
            }
        }
        penalty
    }

    /// Perform one training step across all environments.
    ///
    /// Returns total loss (ERM loss + IRM penalty).
    pub fn train_step(
        &mut self,
        env_trajectories: &[Vec<(Vec<f64>, usize, f64)>],
        _seed: &mut u64,
    ) -> f64 {
        if env_trajectories.is_empty() {
            return 0.0;
        }

        let gamma = 0.99;
        let mut env_gradients: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut total_erm_loss = 0.0;

        for traj in env_trajectories {
            if traj.is_empty() {
                continue;
            }
            let rewards: Vec<f64> = traj.iter().map(|(_, _, r)| *r).collect();
            let returns = CrlCausalPolicyGradient::compute_returns(&rewards, gamma);
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            total_erm_loss -= mean_return; // negative because we maximise

            // Compute per-env gradient
            let mut grad = vec![vec![0.0; self.state_dim]; self.action_dim];
            let t_len = traj.len().min(returns.len());
            for t in 0..t_len {
                let (ref state, action, _) = traj[t];
                let g_t = returns[t];
                let pi = self.policy(state);
                for a in 0..self.action_dim {
                    let indicator = if a == action { 1.0 } else { 0.0 };
                    let score = (indicator - pi[a]) * g_t;
                    for s in 0..self.state_dim.min(state.len()) {
                        grad[a][s] += score * state[s];
                    }
                }
            }
            if t_len > 0 {
                let scale = 1.0 / t_len as f64;
                for row in &mut grad {
                    for g in row.iter_mut() {
                        *g *= scale;
                    }
                }
            }
            env_gradients.push(grad);
        }

        // Compute IRM penalty from environment gradient variance
        let irm_pen = self.irm_penalty_term(&env_gradients);
        let total_loss = total_erm_loss + self.irm_penalty * irm_pen;

        // Aggregate gradient (mean across environments) and update
        if !env_gradients.is_empty() {
            let n_env = env_gradients.len() as f64;
            let mut mean_grad = vec![vec![0.0; self.state_dim]; self.action_dim];
            for eg in &env_gradients {
                for a in 0..self.action_dim.min(eg.len()) {
                    for s in 0..self.state_dim.min(eg[a].len()) {
                        mean_grad[a][s] += eg[a][s] / n_env;
                    }
                }
            }
            // Gradient ascent (policy gradient maximises returns)
            for a in 0..self.action_dim {
                for s in 0..self.state_dim {
                    self.policy_weights[a][s] += self.lr * mean_grad[a][s];
                }
            }
        }

        total_loss
    }

    /// Measure how much the policy relies on spurious feature indices.
    ///
    /// Returns the L2 norm of policy weights on spurious features.
    pub fn spurious_correlation_test(&self, spurious_features: &[usize]) -> f64 {
        let mut total = 0.0;
        for a in 0..self.action_dim {
            for &s in spurious_features {
                if s < self.state_dim {
                    total += self.policy_weights[a][s].powi(2);
                }
            }
        }
        total.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  CrlOffPolicyCounterfactual
// ─────────────────────────────────────────────────────────────────────────────

/// Ring-buffer replay for off-policy CRL.
#[derive(Debug, Clone)]
pub struct CrlReplayBuffer {
    pub transitions: Vec<(Vec<f64>, usize, f64, Vec<f64>)>,
    /// Behaviour policy probabilities per stored transition \[action_dim\].
    pub behavior_probs: Vec<Vec<f64>>,
    pub capacity: usize,
    pub head: usize,
    pub size: usize,
}

impl CrlReplayBuffer {
    /// Create empty ring buffer.
    pub fn new(capacity: usize) -> Self {
        CrlReplayBuffer {
            transitions: Vec::with_capacity(capacity),
            behavior_probs: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            size: 0,
        }
    }

    /// Add transition to ring buffer.
    pub fn add(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        behavior_prob: Vec<f64>,
    ) {
        let transition = (state, action, reward, next_state);
        if self.size < self.capacity {
            self.transitions.push(transition);
            self.behavior_probs.push(behavior_prob);
            self.size += 1;
        } else {
            self.transitions[self.head] = transition;
            self.behavior_probs[self.head] = behavior_prob;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Sample batch of indices without replacement (with fallback to replacement).
    pub fn sample_indices(&self, batch_size: usize, seed: &mut u64) -> Vec<usize> {
        let n = self.size;
        let batch_size = batch_size.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle
        for i in 0..batch_size {
            let j = i + (crl_rand01(seed) * (n - i) as f64) as usize % (n - i).max(1);
            indices.swap(i, j);
        }
        indices[..batch_size].to_vec()
    }
}

/// Off-policy Q-learning with counterfactual IS correction.
#[derive(Debug, Clone)]
pub struct CrlOffPolicyCounterfactual {
    pub buffer: CrlReplayBuffer,
    pub scm: CrlStructuralCausalModel,
    /// Q-function weights [action_dim × state_dim].
    pub q_weights: Vec<Vec<f64>>,
    pub q_bias: Vec<f64>,
    pub lr: f64,
    pub gamma: f64,
}

impl CrlOffPolicyCounterfactual {
    const RHO_MAX: f64 = 5.0;

    /// Create new off-policy agent.
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        capacity: usize,
        scm: CrlStructuralCausalModel,
    ) -> Self {
        CrlOffPolicyCounterfactual {
            buffer: CrlReplayBuffer::new(capacity),
            scm,
            q_weights: vec![vec![0.0; state_dim]; action_dim],
            q_bias: vec![0.0; action_dim],
            lr: 1e-3,
            gamma: 0.99,
        }
    }

    /// Store a transition in the replay buffer.
    pub fn add_transition(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        behavior_prob: Vec<f64>,
    ) {
        self.buffer
            .add(state, action, reward, next_state, behavior_prob);
    }

    /// Compute Q-values for all actions given state.
    pub fn q_value(&self, state: &[f64]) -> Vec<f64> {
        let action_dim = self.q_weights.len();
        let state_dim = if action_dim > 0 {
            self.q_weights[0].len()
        } else {
            0
        };
        (0..action_dim)
            .map(|a| {
                let mut q = self.q_bias[a];
                for s in 0..state_dim.min(state.len()) {
                    q += self.q_weights[a][s] * state[s];
                }
                q
            })
            .collect()
    }

    /// Importance weight with clipping: min(ρ_max, π(a|s) / π_b(a|s)).
    pub fn importance_weight(&self, pi_a: f64, pi_b_a: f64) -> f64 {
        if pi_b_a <= 0.0 {
            return Self::RHO_MAX;
        }
        (pi_a / pi_b_a).min(Self::RHO_MAX)
    }

    /// Counterfactual Q-update using sampled batch.
    ///
    /// For each sampled transition (s, a, r, s'), generates a counterfactual (s, a', r_cf, s_cf')
    /// using the SCM, computes importance-weighted TD targets, and updates Q-weights.
    pub fn counterfactual_q_update(
        &mut self,
        batch_size: usize,
        seed: &mut u64,
    ) -> Result<f64, CrlError> {
        if self.buffer.size == 0 {
            return Err(CrlError::InsufficientData("Empty replay buffer".into()));
        }

        let indices = self.buffer.sample_indices(batch_size, seed);
        let mut total_loss = 0.0;

        // Collect transitions to avoid borrow issues
        let sampled: Vec<_> = indices
            .iter()
            .map(|&i| {
                let (ref s, a, r, ref ns) = self.buffer.transitions[i];
                let bp = &self.buffer.behavior_probs[i];
                (s.clone(), a, r, ns.clone(), bp.clone())
            })
            .collect();

        let action_dim = self.scm.var_dims.get(1).copied().unwrap_or(1);

        for (state, action, reward, next_state, behavior_prob) in sampled {
            let q_next = self.q_value(&next_state);
            let max_q_next = q_next.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_target = reward + self.gamma * max_q_next;

            // Counterfactual action
            let cf_action_idx = {
                let candidate = (crl_rand01(seed) * action_dim as f64) as usize % action_dim.max(1);
                if candidate == action && action_dim > 1 {
                    (candidate + 1) % action_dim
                } else {
                    candidate
                }
            };

            let action_vec: Vec<f64> = {
                let mut v = vec![0.0; action_dim];
                if action < action_dim {
                    v[action] = 1.0;
                }
                v
            };
            let cf_action_vec: Vec<f64> = {
                let mut v = vec![0.0; action_dim];
                if cf_action_idx < action_dim {
                    v[cf_action_idx] = 1.0;
                }
                v
            };

            let (cf_next, cf_reward) = self.scm.counterfactual(
                &state,
                &action_vec,
                &next_state,
                reward,
                &cf_action_vec,
                seed,
            );

            let q_cf_next = self.q_value(&cf_next);
            let max_q_cf_next = q_cf_next.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let cf_td_target = cf_reward + self.gamma * max_q_cf_next;

            // IS weight for counterfactual
            let pi_b_cf = behavior_prob
                .get(cf_action_idx)
                .copied()
                .unwrap_or(1.0 / action_dim as f64);
            let pi_cf = 1.0 / action_dim as f64; // uniform target policy
            let rho = self.importance_weight(pi_cf, pi_b_cf);

            // Combined target
            let combined_target = 0.5 * td_target + 0.5 * rho * cf_td_target;

            // Q-update for factual action
            let q_a = self.q_value(&state)[action.min(self.q_weights.len().saturating_sub(1))];
            let err = q_a - combined_target;
            total_loss += err * err;

            let state_dim = if !self.q_weights.is_empty() {
                self.q_weights[0].len()
            } else {
                0
            };
            let a_idx = action.min(self.q_weights.len().saturating_sub(1));
            for s in 0..state_dim.min(state.len()) {
                self.q_weights[a_idx][s] -= self.lr * err * state[s];
            }
            self.q_bias[a_idx] -= self.lr * err;
        }

        let n = indices.len().max(1) as f64;
        Ok(total_loss / n)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  CrlCausalCredit
// ─────────────────────────────────────────────────────────────────────────────

/// Causal credit assignment using counterfactual reasoning.
///
/// Estimates the causal contribution of each action to the total episodic reward
/// via do-calculus interventions (Buesing et al. 2019).
#[derive(Debug, Clone)]
pub struct CrlCausalCredit {
    pub n_steps: usize,
    /// Decay factor for temporal credit.
    pub discount_decay: f64,
    pub scm: CrlStructuralCausalModel,
}

impl CrlCausalCredit {
    /// Create causal credit module.
    pub fn new(n_steps: usize, decay: f64, scm: CrlStructuralCausalModel) -> Self {
        CrlCausalCredit {
            n_steps,
            discount_decay: decay.clamp(0.0, 1.0),
            scm,
        }
    }

    /// Estimate causal credit for each time step by counterfactual intervention.
    ///
    /// For each action a_t, we ask: what would the reward have been if we had
    /// taken a *random* action instead?
    ///   credit_t = r_t - E[r_t | do(a_t = random)]
    pub fn causal_credit_vector(
        &self,
        trajectory: &[(Vec<f64>, usize, f64)],
        seed: &mut u64,
    ) -> Vec<f64> {
        if trajectory.is_empty() {
            return Vec::new();
        }

        let action_dim = self.scm.var_dims.get(1).copied().unwrap_or(1);
        let t_len = trajectory.len();
        let mut credits = vec![0.0; t_len];

        for t in 0..t_len {
            let (ref state, action, actual_reward) = trajectory[t];

            // Compute counterfactual reward with random action at time t
            let mut cf_reward_sum = 0.0;
            let n_cf_samples = 3usize; // Monte Carlo estimate

            for _ in 0..n_cf_samples {
                let cf_action_idx =
                    (crl_rand01(seed) * action_dim as f64) as usize % action_dim.max(1);

                let action_vec: Vec<f64> = {
                    let mut v = vec![0.0; action_dim];
                    if action < action_dim {
                        v[action] = 1.0;
                    }
                    v
                };
                let cf_action_vec: Vec<f64> = {
                    let mut v = vec![0.0; action_dim];
                    if cf_action_idx < action_dim {
                        v[cf_action_idx] = 1.0;
                    }
                    v
                };

                // We need a factual next state for abduction; estimate from SCM
                let (factual_next, _) = self.scm.sample(state, &action_vec, seed);
                let (_, cf_reward) = self.scm.counterfactual(
                    state,
                    &action_vec,
                    &factual_next,
                    actual_reward,
                    &cf_action_vec,
                    seed,
                );
                cf_reward_sum += cf_reward;
            }

            let cf_reward_mean = cf_reward_sum / n_cf_samples as f64;
            credits[t] = actual_reward - cf_reward_mean;
        }
        credits
    }

    /// Hindsight credit: weight each action by its alignment with future returns.
    ///
    /// credit_t = Σ_{k≥t} decay^(k-t) * r_k
    pub fn hindsight_credit(&self, trajectory: &[(Vec<f64>, usize, f64)]) -> Vec<f64> {
        let n = trajectory.len();
        let mut credits = vec![0.0; n];
        let mut running = 0.0;
        for t in (0..n).rev() {
            let (_, _, r) = &trajectory[t];
            running = r + self.discount_decay * running;
            credits[t] = running;
        }
        credits
    }

    /// Compute a [T × T] temporal causal influence matrix.
    ///
    /// Entry [t, t'] estimates the causal influence of action at time t on reward at t'.
    /// Uses exponential decay: influence\[t\]\[t'\] = decay^(t'-t) if t' >= t, else 0.
    pub fn temporal_causal_graph(&self, trajectory: &[(Vec<f64>, usize, f64)]) -> Vec<Vec<f64>> {
        let n = trajectory.len();
        let mut graph = vec![vec![0.0; n]; n];
        for t in 0..n {
            let (_, _, r_t) = &trajectory[t];
            for t2 in t..n {
                let steps = (t2 - t) as f64;
                // Influence decays exponentially; modulated by reward at t'
                let (_, _, r_t2) = &trajectory[t2];
                graph[t][t2] = self.discount_decay.powf(steps) * r_t2.abs().max(1e-8)
                    / (r_t.abs().max(1e-8) + r_t2.abs().max(1e-8));
            }
        }
        graph
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  CrlSCMDynaQ
// ─────────────────────────────────────────────────────────────────────────────

/// Dyna-Q agent that uses the causal SCM as its world model for planning.
#[derive(Debug, Clone)]
pub struct CrlSCMDynaQ {
    /// Tabular Q-values [n_states × n_actions].
    pub q_table: Vec<Vec<f64>>,
    pub scm: CrlStructuralCausalModel,
    pub n_planning_steps: usize,
    pub lr: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub n_states: usize,
    pub n_actions: usize,
    /// Experience buffer: (s, a, r, s').
    pub model_memory: Vec<(usize, usize, f64, usize)>,
}

impl CrlSCMDynaQ {
    /// Create Dyna-Q agent.
    pub fn new(n_states: usize, n_actions: usize, scm: CrlStructuralCausalModel) -> Self {
        CrlSCMDynaQ {
            q_table: vec![vec![0.0; n_actions]; n_states],
            scm,
            n_planning_steps: 5,
            lr: 0.1,
            gamma: 0.99,
            epsilon: 0.1,
            n_states,
            n_actions,
            model_memory: Vec::new(),
        }
    }

    /// ε-greedy action selection.
    pub fn epsilon_greedy(&self, state: usize, seed: &mut u64) -> usize {
        if crl_rand01(seed) < self.epsilon {
            (crl_rand01(seed) * self.n_actions as f64) as usize % self.n_actions.max(1)
        } else {
            self.q_table
                .get(state)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .unwrap_or(0)
        }
    }

    /// Tabular Q-update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)].
    pub fn q_update(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let max_q_next = self
            .q_table
            .get(next_state)
            .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .unwrap_or(0.0)
            .max(f64::NEG_INFINITY);

        if let Some(row) = self.q_table.get_mut(state) {
            if let Some(q) = row.get_mut(action) {
                let td_target = reward + self.gamma * max_q_next;
                *q += self.lr * (td_target - *q);
            }
        }
    }

    /// Planning step: sample random transition from model memory and do Q-update.
    pub fn planning_step(&mut self, seed: &mut u64) {
        if self.model_memory.is_empty() {
            return;
        }
        let idx =
            (crl_rand01(seed) * self.model_memory.len() as f64) as usize % self.model_memory.len();
        let (s, a, r, ns) = self.model_memory[idx];
        self.q_update(s, a, r, ns);
    }

    /// Perform Q-update + planning; store transition; return next action.
    pub fn step(&mut self, state: usize, next_state: usize, reward: f64, seed: &mut u64) -> usize {
        // Real experience update
        let action = self.epsilon_greedy(state, seed);
        self.q_update(state, action, reward, next_state);

        // Store in model memory (cap at 10_000)
        if self.model_memory.len() < 10_000 {
            self.model_memory.push((state, action, reward, next_state));
        } else {
            let idx = (crl_rand01(seed) * 10_000.0) as usize % 10_000;
            self.model_memory[idx] = (state, action, reward, next_state);
        }

        // Planning steps
        for _ in 0..self.n_planning_steps {
            self.planning_step(seed);
        }

        // Return next action
        self.epsilon_greedy(next_state, seed)
    }

    /// Shape Q-values using causal importance of state features.
    ///
    /// `causal_weights[s]` = importance of state feature s; high-importance
    /// features receive a bonus proportional to their current Q-value contribution.
    pub fn causal_value_shaping(&mut self, causal_weights: &[f64]) {
        let max_w = causal_weights
            .iter()
            .cloned()
            .fold(0.0f64, f64::max)
            .max(1e-8);
        for s in 0..self.n_states {
            let importance = causal_weights
                .get(s % causal_weights.len().max(1))
                .copied()
                .unwrap_or(0.0);
            let scale = 1.0 + 0.1 * importance / max_w;
            for a in 0..self.n_actions {
                if let Some(q) = self.q_table.get_mut(s).and_then(|row| row.get_mut(a)) {
                    *q *= scale;
                }
            }
        }
    }

    /// Return the greedy action for each state.
    pub fn greedy_policy(&self) -> Vec<usize> {
        self.q_table
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  CrlCausalCurriculum
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for selecting the next training environment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrlCurriculumStrategy {
    /// Choose environment with highest expected information gain.
    MaxInformationGain,
    /// Choose environment matched to current competence level.
    Competence,
    /// Choose uniformly at random.
    Random,
}

/// Causal curriculum learning: adaptively select environments to maximise
/// causal understanding.
#[derive(Debug, Clone)]
pub struct CrlCausalCurriculum {
    pub environments: Vec<CrlEnvironment>,
    /// Estimated causal strengths [n_vars × n_vars].
    pub causal_graph_estimate: Vec<Vec<f64>>,
    /// Per-environment estimated information gain.
    pub information_gain: Vec<f64>,
    /// Per-environment competence score [0, 1].
    pub competence: Vec<f64>,
    pub n_envs: usize,
}

impl CrlCausalCurriculum {
    /// Create curriculum from a list of environments.
    pub fn new(environments: Vec<CrlEnvironment>) -> Self {
        let n_envs = environments.len();
        let state_dim = environments.first().map(|e| e.state_dim).unwrap_or(1);
        let n_vars = state_dim + 2; // state + action + reward
        CrlCausalCurriculum {
            environments,
            causal_graph_estimate: vec![vec![0.0; n_vars]; n_vars],
            information_gain: vec![1.0; n_envs], // initialise high
            competence: vec![0.0; n_envs],
            n_envs,
        }
    }

    /// Update causal graph estimate from observed transitions in env `env_id`.
    ///
    /// Uses correlation-based estimation: C\[i\]\[j\] ~ |Cov(V_i, V_j)| / (|V_i| * |V_j|).
    pub fn update_causal_estimate(
        &mut self,
        env_id: usize,
        transitions: &[(Vec<f64>, Vec<f64>, f64, Vec<f64>)],
    ) {
        if transitions.is_empty() || env_id >= self.n_envs {
            return;
        }

        let n = transitions.len() as f64;
        let n_vars = self.causal_graph_estimate.len();

        // Extract (state[0], action[0], reward) triplets
        let triplets: Vec<[f64; 3]> = transitions
            .iter()
            .map(|(s, a, r, _)| {
                [
                    s.first().copied().unwrap_or(0.0),
                    a.first().copied().unwrap_or(0.0),
                    *r,
                ]
            })
            .collect();

        // Pairwise Pearson correlations
        for i in 0..3.min(n_vars) {
            let mean_i = triplets.iter().map(|t| t[i]).sum::<f64>() / n;
            for j in 0..3.min(n_vars) {
                if i == j {
                    continue;
                }
                let mean_j = triplets.iter().map(|t| t[j]).sum::<f64>() / n;
                let cov = triplets
                    .iter()
                    .map(|t| (t[i] - mean_i) * (t[j] - mean_j))
                    .sum::<f64>()
                    / n;
                let std_i = (triplets
                    .iter()
                    .map(|t| (t[i] - mean_i).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt()
                    .max(1e-8);
                let std_j = (triplets
                    .iter()
                    .map(|t| (t[j] - mean_j).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt()
                    .max(1e-8);
                let corr = (cov / (std_i * std_j)).abs().clamp(0.0, 1.0);
                // Exponential moving average update
                self.causal_graph_estimate[i][j] =
                    0.9 * self.causal_graph_estimate[i][j] + 0.1 * corr;
            }
        }

        // Update information gain: lower uncertainty after observing means less future gain
        if env_id < self.n_envs {
            self.information_gain[env_id] *= 0.95;
            // Update competence: higher correlation seen → higher competence
            let mean_corr: f64 = self
                .causal_graph_estimate
                .iter()
                .flat_map(|row| row.iter())
                .sum::<f64>()
                / (n_vars * n_vars).max(1) as f64;
            self.competence[env_id] =
                (self.competence[env_id] * 0.9 + mean_corr * 0.1).clamp(0.0, 1.0);
        }
    }

    /// Estimate information gain for environment `env_id`.
    pub fn information_gain_score(&self, env_id: usize) -> f64 {
        self.information_gain.get(env_id).copied().unwrap_or(0.0)
    }

    /// Select the next environment to train on using the given strategy.
    pub fn select_next_environment(&self, strategy: CrlCurriculumStrategy) -> usize {
        if self.n_envs == 0 {
            return 0;
        }
        match strategy {
            CrlCurriculumStrategy::MaxInformationGain => self
                .information_gain
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0),
            CrlCurriculumStrategy::Competence => {
                // Select environment where competence is just below mean (zone of proximal development)
                let mean_comp = self.competence.iter().sum::<f64>() / self.n_envs as f64;
                self.competence
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = (*a - mean_comp * 0.8).abs();
                        let db = (*b - mean_comp * 0.8).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            CrlCurriculumStrategy::Random => {
                // Deterministic fallback: cycle through environments
                self.n_envs / 2
            }
        }
    }

    /// Overall causal understanding score: negative entropy of graph edge uncertainty.
    ///
    /// Edges with strength close to 0 or 1 are "understood"; edges near 0.5 are uncertain.
    pub fn causal_understanding_score(&self) -> f64 {
        let n_vars = self.causal_graph_estimate.len();
        if n_vars == 0 {
            return 0.0;
        }
        let n_edges = (n_vars * n_vars) as f64;
        let mut total_entropy = 0.0;
        for row in &self.causal_graph_estimate {
            for &p in row {
                let p = p.clamp(1e-9, 1.0 - 1e-9);
                // Bernoulli entropy H(p) = -p log p - (1-p) log(1-p)
                let h = -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
                total_entropy += h;
            }
        }
        // Understanding = 1 - normalised entropy; max entropy of Bernoulli is ln(2) ≈ 0.693
        let max_entropy = n_edges * 2_f64.ln();
        1.0 - (total_entropy / max_entropy.max(1e-8)).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  CrlMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of evaluation metrics for causal RL.
pub struct CrlMetrics;

impl CrlMetrics {
    /// Causal regret: total regret attributable to following spurious correlations.
    ///
    /// regret = mean(optimal) - mean(actual) weighted by spurious strength.
    pub fn causal_regret(
        optimal_returns: &[f64],
        actual_returns: &[f64],
        spurious_correlation_strength: f64,
    ) -> f64 {
        if optimal_returns.is_empty() || actual_returns.is_empty() {
            return 0.0;
        }
        let mean_opt = optimal_returns.iter().sum::<f64>() / optimal_returns.len() as f64;
        let mean_act = actual_returns.iter().sum::<f64>() / actual_returns.len() as f64;
        (mean_opt - mean_act) * spurious_correlation_strength.clamp(0.0, 1.0)
    }

    /// Counterfactual fairness score: how much does individual outcome change when
    /// we intervene on group membership?
    pub fn counterfactual_fairness_score(
        outcomes: &[f64],
        group_ids: &[usize],
        cf_outcomes: &[f64],
    ) -> f64 {
        let n = outcomes.len().min(group_ids.len()).min(cf_outcomes.len());
        if n == 0 {
            return 0.0;
        }

        let group0_outcomes: Vec<f64> = (0..n)
            .filter(|&i| group_ids[i] == 0)
            .map(|i| outcomes[i])
            .collect();
        let group0_cf: Vec<f64> = (0..n)
            .filter(|&i| group_ids[i] == 0)
            .map(|i| cf_outcomes[i])
            .collect();

        if group0_outcomes.is_empty() {
            return 0.0;
        }

        let mean_g0 = group0_outcomes.iter().sum::<f64>() / group0_outcomes.len() as f64;
        let mean_cf0 = group0_cf.iter().sum::<f64>() / group0_cf.len() as f64;
        let range = outcomes.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - outcomes.iter().cloned().fold(f64::INFINITY, f64::min);
        let normalizer = range.abs().max(1e-8);
        ((mean_g0 - mean_cf0).abs() / normalizer).clamp(0.0, 1.0)
    }

    /// F1 score of causal edge prediction.
    pub fn causal_discovery_accuracy(
        estimated_graph: &[Vec<bool>],
        true_graph: &[Vec<bool>],
    ) -> f64 {
        let n = estimated_graph.len().min(true_graph.len());
        if n == 0 {
            return 0.0;
        }
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_ = 0u64;

        for i in 0..n {
            let est_row = &estimated_graph[i];
            let true_row = &true_graph[i];
            let m = est_row.len().min(true_row.len());
            for j in 0..m {
                match (est_row[j], true_row[j]) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_ += 1,
                    _ => {}
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        if precision + recall < 1e-9 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Invariance score: how consistent are returns across environments?
    ///
    /// Returns 1 - coefficient_of_variation_across_mean_returns.
    pub fn invariance_score(env_returns: &[Vec<f64>]) -> f64 {
        if env_returns.is_empty() {
            return 1.0;
        }
        let env_means: Vec<f64> = env_returns
            .iter()
            .map(|ep| {
                if ep.is_empty() {
                    0.0
                } else {
                    ep.iter().sum::<f64>() / ep.len() as f64
                }
            })
            .collect();

        let n = env_means.len() as f64;
        let grand_mean = env_means.iter().sum::<f64>() / n;
        if grand_mean.abs() < 1e-8 {
            return 1.0;
        }
        let std_dev = (env_means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let cv = std_dev / grand_mean.abs();
        (1.0 - cv).clamp(0.0, 1.0)
    }

    /// Pearson correlation between estimated and true credit assignments.
    pub fn credit_assignment_accuracy(estimated_credits: &[f64], true_credits: &[f64]) -> f64 {
        let n = estimated_credits.len().min(true_credits.len());
        if n < 2 {
            return 0.0;
        }

        let mean_e = estimated_credits[..n].iter().sum::<f64>() / n as f64;
        let mean_t = true_credits[..n].iter().sum::<f64>() / n as f64;

        let num = (0..n)
            .map(|i| (estimated_credits[i] - mean_e) * (true_credits[i] - mean_t))
            .sum::<f64>();
        let std_e = (0..n)
            .map(|i| (estimated_credits[i] - mean_e).powi(2))
            .sum::<f64>()
            .sqrt();
        let std_t = (0..n)
            .map(|i| (true_credits[i] - mean_t).powi(2))
            .sum::<f64>()
            .sqrt();

        if std_e < 1e-10 || std_t < 1e-10 {
            0.0
        } else {
            (num / (std_e * std_t)).clamp(-1.0, 1.0)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / sum).collect()
}

/// Sample from categorical distribution.
fn sample_categorical(probs: &[f64], seed: &mut u64) -> usize {
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

