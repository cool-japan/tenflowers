//! Active Inference (Free Energy Principle) — TenfloweRS.
//!
//! Implements Karl Friston's Active Inference framework, a unified theory of
//! action and perception based on variational free energy minimization.
//!
//! ## Key Concepts
//!
//! - **Variational Free Energy** F: Upper bound on surprise − log P(o).
//!   Minimized during perception (belief updating) via variational inference.
//! - **Expected Free Energy** G(π): Future-oriented objective combining
//!   epistemic value (information gain) and pragmatic value (preference satisfaction).
//! - **Generative Model**: P(o, s) = P(o|s) P(s) with A (likelihood),
//!   B (transitions), C (preferences), D (priors).
//! - **Hierarchical Models**: Multi-level predictive coding with precision weighting.
//! - **Parameter Learning**: Online Bayesian learning of A and B matrices.
//!
//! ## References
//! - Friston et al. (2017) "Active inference, curiosity and insight"
//! - Friston et al. (2015) "Active inference and epistemic value"
//! - Parr & Friston (2019) "Generalised free energy and active inference"
//! - Da Costa et al. (2020) "Active inference on discrete state-spaces"

#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

use scirs2_core::random::{rngs::SmallRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the Active Inference module.
#[derive(Debug, Clone, PartialEq)]
pub enum AiError {
    /// Dimension mismatch or invalid size.
    InvalidDimension(String),
    /// Numerical failure (NaN, divergence, etc.).
    NumericalError(String),
    /// Invalid configuration parameter.
    ConfigError(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::InvalidDimension(m) => write!(f, "AiError::InvalidDimension: {}", m),
            AiError::NumericalError(m) => write!(f, "AiError::NumericalError: {}", m),
            AiError::ConfigError(m) => write!(f, "AiError::ConfigError: {}", m),
        }
    }
}

impl std::error::Error for AiError {}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (no ndarray, no rand, pure Vec<f64>)
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp.
#[inline]
fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_x.is_infinite() {
        return max_x;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

/// Numerically stable softmax in-place.
fn softmax_inplace(v: &mut [f64]) {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
    }
    let s: f64 = v.iter().sum();
    let denom = if s < 1e-15 { 1.0 } else { s };
    for x in v.iter_mut() {
        *x /= denom;
    }
}

/// Compute softmax of a slice, return new Vec.
fn softmax_vec(logits: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = logits.to_vec();
    softmax_inplace(&mut out);
    out
}

/// Shannon entropy of a probability distribution: -∑ p log p.
fn entropy(p: &[f64]) -> f64 {
    p.iter().filter(|&&v| v > 1e-15).map(|&v| -v * v.ln()).sum()
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiply: M [rows × cols] · v [cols] → out [rows].
fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Normalize a Vec<f64> to sum to 1.0 (L1 normalization).
fn normalize_l1(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 1e-15 {
        for x in v.iter_mut() {
            *x /= s;
        }
    } else {
        let n = v.len();
        for x in v.iter_mut() {
            *x = 1.0 / n as f64;
        }
    }
}

/// Dirichlet-like uniform + small noise initialization via SmallRng.
fn dirichlet_init(n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| 1.0 + 0.1 * rng.random::<f64>()).collect();
    normalize_l1(&mut v);
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  AiGenerativeModel
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete-state-space generative model for active inference.
///
/// Encodes:
/// - `a_matrix[o, s]` = P(o | s): likelihood mapping from states to observations.
/// - `b_matrices[u][s', s]` = P(s' | s, u): action-conditional state transitions.
/// - `c_vector[o]` = log P*(o): log preference over observations.
/// - `d_prior[s]` = P(s_0): Dirichlet prior over initial hidden states.
///
/// Friston et al. (2017) use this exact parameterization for discrete active inference.
#[derive(Debug, Clone)]
pub struct AiGenerativeModel {
    /// Likelihood mapping A: [n_obs × n_states].
    pub a_matrix: Vec<Vec<f64>>,
    /// Transition matrices B: [n_actions × n_states × n_states].
    /// `b_matrices[u][s_next][s_curr]` = P(s_next | s_curr, u).
    pub b_matrices: Vec<Vec<Vec<f64>>>,
    /// Log preference vector C: \[n_obs\].
    pub c_vector: Vec<f64>,
    /// Prior over initial states D: \[n_states\].
    pub d_prior: Vec<f64>,
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of possible observations.
    pub n_obs: usize,
    /// Number of possible actions.
    pub n_actions: usize,
}

impl AiGenerativeModel {
    /// Create a new generative model with random initialization.
    ///
    /// - `a_matrix`: initialized with uniform + Dirichlet noise, then normalized.
    /// - `b_matrices`: initialized close to identity (self-transitions favored) + noise.
    /// - `c_vector`: zeros (no preference by default).
    /// - `d_prior`: uniform over states.
    pub fn new(n_states: usize, n_obs: usize, n_actions: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);

        // Likelihood A: [n_obs × n_states], each column sums to 1
        let mut a_matrix: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
        for _ in 0..n_obs {
            a_matrix.push(
                (0..n_states)
                    .map(|_| 0.5 + 0.5 * rng.random::<f64>())
                    .collect(),
            );
        }
        // Normalize columns so P(o | s) is a valid distribution for each s
        for s in 0..n_states {
            let col_sum: f64 = (0..n_obs).map(|o| a_matrix[o][s]).sum();
            let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
            for o in 0..n_obs {
                a_matrix[o][s] /= denom;
            }
        }

        // Transition B: [n_actions × n_states × n_states], each column sums to 1
        let mut b_matrices: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_actions);
        for _u in 0..n_actions {
            let mut b: Vec<Vec<f64>> = Vec::with_capacity(n_states);
            for s_next in 0..n_states {
                let mut row: Vec<f64> = Vec::with_capacity(n_states);
                for s_curr in 0..n_states {
                    // Identity-ish: diagonal slightly higher
                    let base = if s_next == s_curr { 2.0 } else { 0.5 };
                    row.push(base + 0.3 * rng.random::<f64>());
                }
                b.push(row);
            }
            // Normalize columns of b
            for s in 0..n_states {
                let col_sum: f64 = (0..n_states).map(|sn| b[sn][s]).sum();
                let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
                for sn in 0..n_states {
                    b[sn][s] /= denom;
                }
            }
            b_matrices.push(b);
        }

        let c_vector = vec![0.0; n_obs];
        let d_prior = vec![1.0 / n_states as f64; n_states];

        AiGenerativeModel {
            a_matrix,
            b_matrices,
            c_vector,
            d_prior,
            n_states,
            n_obs,
            n_actions,
        }
    }

    /// Log-likelihood log P(o | s).
    pub fn log_likelihood(&self, obs: usize, state: usize) -> f64 {
        let p = self.a_matrix[obs][state].max(1e-15);
        p.ln()
    }

    /// Return transition matrix B\[u\]: [n_states × n_states] for action u.
    pub fn transition(&self, action: usize) -> &Vec<Vec<f64>> {
        &self.b_matrices[action]
    }

    /// Normalize A columns so each sums to 1 (valid likelihood).
    pub fn normalize_a(&mut self) {
        for s in 0..self.n_states {
            let col_sum: f64 = (0..self.n_obs).map(|o| self.a_matrix[o][s]).sum();
            let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
            for o in 0..self.n_obs {
                self.a_matrix[o][s] /= denom;
            }
        }
    }

    /// Normalize B columns so each sums to 1 (valid transition).
    pub fn normalize_b(&mut self) {
        for u in 0..self.n_actions {
            for s in 0..self.n_states {
                let col_sum: f64 = (0..self.n_states).map(|sn| self.b_matrices[u][sn][s]).sum();
                let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
                for sn in 0..self.n_states {
                    self.b_matrices[u][sn][s] /= denom;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  AiBeliefState
// ─────────────────────────────────────────────────────────────────────────────

/// Variational posterior q(s) over hidden states: a categorical distribution.
///
/// Parameterized by `beliefs` (softmax probabilities) and `log_beliefs`.
/// Represents the agent's current estimate of the hidden state of the world.
#[derive(Debug, Clone)]
pub struct AiBeliefState {
    /// Probability of each state: q(s_i).
    pub beliefs: Vec<f64>,
    /// Log probabilities log q(s_i).
    pub log_beliefs: Vec<f64>,
    /// Number of states.
    pub n_states: usize,
}

impl AiBeliefState {
    /// Initialize with uniform distribution over all states.
    pub fn new(n_states: usize) -> Self {
        let p = 1.0 / n_states as f64;
        let log_p = p.ln();
        AiBeliefState {
            beliefs: vec![p; n_states],
            log_beliefs: vec![log_p; n_states],
            n_states,
        }
    }

    /// Initialize from a prior distribution (normalizes automatically).
    pub fn from_prior(prior: &[f64]) -> Self {
        let n = prior.len();
        let mut beliefs: Vec<f64> = prior.to_vec();
        normalize_l1(&mut beliefs);
        let log_beliefs: Vec<f64> = beliefs.iter().map(|&p| p.max(1e-15).ln()).collect();
        AiBeliefState {
            beliefs,
            log_beliefs,
            n_states: n,
        }
    }

    /// Update beliefs with new probability vector (normalizes + recomputes logs).
    pub fn update(&mut self, new_beliefs: &[f64]) {
        let mut nb: Vec<f64> = new_beliefs.to_vec();
        normalize_l1(&mut nb);
        self.log_beliefs = nb.iter().map(|&p| p.max(1e-15).ln()).collect();
        self.beliefs = nb;
    }

    /// Shannon entropy H\[q\] = -∑ q(s) log q(s) ≥ 0.
    pub fn entropy(&self) -> f64 {
        entropy(&self.beliefs)
    }

    /// KL divergence KL(q || p) = ∑ q_s (log q_s − log p_s) ≥ 0.
    pub fn kl_divergence(&self, prior: &[f64]) -> f64 {
        self.beliefs
            .iter()
            .zip(prior.iter())
            .map(|(&q, &p)| {
                if q < 1e-15 {
                    0.0
                } else {
                    q * (q.ln() - p.max(1e-15).ln())
                }
            })
            .sum()
    }

    /// Return index of the most probable state (argmax).
    pub fn most_likely_state(&self) -> usize {
        self.beliefs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  AiVariationalInference
// ─────────────────────────────────────────────────────────────────────────────

/// Performs variational belief updating (perception) via fixed-point iteration.
///
/// Minimizes the variational free energy F with respect to the posterior q(s):
///
/// ```text
/// F = KL[q(s) || p(s)] - E_q[log p(o | s)]
///   = Σ_s q_s (log q_s - log D_s) - Σ_s q_s log A[o, s]
/// ```
///
/// The natural gradient update on the simplex reduces to the softmax of the
/// sum of log-prior and log-likelihood (Friston et al. 2017, Eq. 24):
///
/// ```text
/// log q_s ← log A[o, s] + log D_s   (then normalize)
/// ```
///
/// For the iterative version we use gradient descent on log q with learning rate `lr`.
#[derive(Debug, Clone)]
pub struct AiVariationalInference {
    /// Maximum number of fixed-point iterations.
    pub max_iter: usize,
    /// Learning rate for gradient updates.
    pub lr: f64,
    /// Convergence tolerance on belief change.
    pub tol: f64,
}

impl AiVariationalInference {
    /// Create a new variational inference engine.
    pub fn new(max_iter: usize, lr: f64, tol: f64) -> Self {
        AiVariationalInference { max_iter, lr, tol }
    }

    /// Infer posterior beliefs q(s | o) given observation `obs`.
    ///
    /// Performs gradient descent on the variational free energy:
    ///
    /// dF / d(log q_s) ≈ q_s − A[o, s] * D_s / Z
    ///
    /// In log-space: log_q_s -= lr * (log_q_s - log_A\[o,s\] - log_D\[s\])
    pub fn infer_states(
        &self,
        obs: usize,
        model: &AiGenerativeModel,
        prior_beliefs: &AiBeliefState,
    ) -> Result<AiBeliefState, AiError> {
        if obs >= model.n_obs {
            return Err(AiError::InvalidDimension(format!(
                "obs={} out of range [0, {})",
                obs, model.n_obs
            )));
        }
        if prior_beliefs.n_states != model.n_states {
            return Err(AiError::InvalidDimension(format!(
                "prior_beliefs.n_states={} != model.n_states={}",
                prior_beliefs.n_states, model.n_states
            )));
        }

        // Initialize log_q from prior
        let mut log_q: Vec<f64> = prior_beliefs.log_beliefs.to_vec();

        let mut prev_beliefs: Vec<f64> = softmax_vec(&log_q);

        for _iter in 0..self.max_iter {
            // Natural gradient in log-space:
            // target = log A[o, s] + log D[s]   (unnormalized log posterior)
            // gradient of F w.r.t. log_q_s = log_q_s - log_A[o,s] - log_D[s]
            for s in 0..model.n_states {
                let log_a = model.a_matrix[obs][s].max(1e-15).ln();
                let log_d = model.d_prior[s].max(1e-15).ln();
                let grad = log_q[s] - log_a - log_d;
                log_q[s] -= self.lr * grad;
            }

            let new_beliefs = softmax_vec(&log_q);

            // Check convergence
            let diff: f64 = new_beliefs
                .iter()
                .zip(prev_beliefs.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();

            prev_beliefs = new_beliefs;

            if diff < self.tol {
                break;
            }
        }

        let mut result = AiBeliefState::new(model.n_states);
        result.update(&softmax_vec(&log_q));
        Ok(result)
    }

    /// Compute variational free energy F for current beliefs given observation `obs`.
    ///
    /// F = KL[q || D] - E_q[log P(o | s)]
    ///   = Σ_s q_s log(q_s / D_s) - Σ_s q_s log A[o, s]
    pub fn free_energy(
        &self,
        obs: usize,
        beliefs: &AiBeliefState,
        model: &AiGenerativeModel,
    ) -> f64 {
        let kl: f64 = beliefs.kl_divergence(&model.d_prior);
        let expected_log_likelihood: f64 = beliefs
            .beliefs
            .iter()
            .enumerate()
            .map(|(s, &q_s)| {
                if q_s < 1e-15 {
                    0.0
                } else {
                    q_s * model.a_matrix[obs][s].max(1e-15).ln()
                }
            })
            .sum();
        kl - expected_log_likelihood
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  AiExpectedFreeEnergy
// ─────────────────────────────────────────────────────────────────────────────

/// Computes Expected Free Energy G(π) for policy evaluation.
///
/// G(π) decomposes into two terms (Friston et al. 2017):
///
/// ```text
/// G(π) = Σ_τ [ H[P(o_τ | s_τ, π)] - H[P(o_τ | π)] - C · P(o_τ | π) ]
///       = -epistemic_value - pragmatic_value
/// ```
///
/// where:
/// - **Epistemic value** = H[P(o)] − E_s[H[P(o|s)]] = expected information gain
///   (promotes exploration, reduces uncertainty about hidden states)
/// - **Pragmatic value** = C · P(o) (promotes preferred outcomes)
///
/// Note: G is typically minimized (agents prefer low EFE).
#[derive(Debug, Clone)]
pub struct AiExpectedFreeEnergy {
    /// Lookahead horizon (number of steps).
    pub planning_horizon: usize,
}

impl AiExpectedFreeEnergy {
    /// Create a new EFE calculator with given planning horizon.
    pub fn new(planning_horizon: usize) -> Self {
        AiExpectedFreeEnergy { planning_horizon }
    }

    /// Compute G(π) for a policy (sequence of actions).
    ///
    /// For each time step t in the policy:
    /// 1. Predict next state distribution: Qs' = B\[a_t\] @ Qs
    /// 2. Predict obs distribution: Qo = A @ Qs'
    /// 3. epistemic = H\[Qo\] − E_s'[H[P(o|s')]]  (information gain)
    /// 4. pragmatic = C · Qo (preference alignment)
    ///
    /// G = Σ_t (−epistemic − pragmatic)
    pub fn compute(
        &self,
        policy: &[usize],
        current_beliefs: &AiBeliefState,
        model: &AiGenerativeModel,
    ) -> Result<f64, AiError> {
        if policy.is_empty() {
            return Err(AiError::ConfigError("policy cannot be empty".to_string()));
        }
        for (i, &a) in policy.iter().enumerate() {
            if a >= model.n_actions {
                return Err(AiError::InvalidDimension(format!(
                    "policy[{}]={} out of range [0, {})",
                    i, a, model.n_actions
                )));
            }
        }

        let mut qs: Vec<f64> = current_beliefs.beliefs.clone();
        let mut g_total = 0.0;

        let horizon = policy.len().min(self.planning_horizon);
        for t in 0..horizon {
            let action = policy[t];
            let b = model.transition(action);

            // Predict next state: Qs' = B[a] @ Qs  (matrix-vector product)
            let qs_next = mat_vec(b, &qs);

            // Predict observation distribution: Qo = A @ Qs'
            let qo = mat_vec(&model.a_matrix, &qs_next);

            // Epistemic value (information gain)
            let epi = Self::epistemic_value(&qo, &qs_next, &model.a_matrix);

            // Pragmatic value (preference alignment)
            let prag = Self::pragmatic_value(&qo, &model.c_vector);

            // G accumulates negative epistemic and negative pragmatic
            // (Friston: G = sum of ambiguity - utility)
            // We negate so that agents minimize G (higher EFE = worse policy)
            g_total += -epi - prag;

            qs = qs_next;
        }

        Ok(g_total)
    }

    /// Expected information gain (epistemic value):
    ///
    /// H[P(o)] − E_s[H[P(o|s)]]
    ///
    /// = entropy of marginal obs distribution minus
    ///   expected entropy of per-state obs distributions.
    ///
    /// Positive: higher = more uncertainty reduction expected.
    pub fn epistemic_value(qo: &[f64], qs: &[f64], a_matrix: &[Vec<f64>]) -> f64 {
        let n_obs = qo.len();
        let n_states = qs.len();

        // H[P(o)] = entropy of marginal
        let marginal_entropy = entropy(qo);

        // E_s[H[P(o|s)]] = Σ_s q(s) * H[P(o|s)]
        let mut conditional_entropy = 0.0;
        for s in 0..n_states {
            // Extract column s of A: P(o | s) for all o
            let p_o_given_s: Vec<f64> = (0..n_obs).map(|o| a_matrix[o][s]).collect();
            let h_o_given_s = entropy(&p_o_given_s);
            conditional_entropy += qs[s] * h_o_given_s;
        }

        // Information gain = H[P(o)] - E[H[P(o|s)]] ≥ 0
        (marginal_entropy - conditional_entropy).max(0.0)
    }

    /// Pragmatic value (expected log preference):
    ///
    /// ∑_o Q(o) * C\[o\]
    ///
    /// where C\[o\] = log P*(o) is the log preference over observations.
    pub fn pragmatic_value(qo: &[f64], c: &[f64]) -> f64 {
        dot(qo, c)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  AiPolicySelection
// ─────────────────────────────────────────────────────────────────────────────

/// Selects actions by minimizing Expected Free Energy.
///
/// Posterior over policies:
///
/// ```text
/// P(π) ∝ exp(−G(π) / temperature)
/// ```
///
/// Lower G(π) → higher probability of selecting policy π.
/// Temperature controls exploration vs. exploitation.
#[derive(Debug, Clone)]
pub struct AiPolicySelection {
    /// Softmax temperature (higher = more exploratory).
    pub temperature: f64,
    /// Planning depth for EFE computation.
    pub depth: usize,
}

impl AiPolicySelection {
    /// Create a new policy selection module.
    pub fn new(temperature: f64, depth: usize) -> Self {
        AiPolicySelection { temperature, depth }
    }

    /// Select the best action (argmin G) for the current beliefs.
    ///
    /// Enumerates all single-step actions, computes G for each, returns argmin.
    pub fn select_action(
        &self,
        current_beliefs: &AiBeliefState,
        model: &AiGenerativeModel,
        efe: &AiExpectedFreeEnergy,
    ) -> Result<usize, AiError> {
        if model.n_actions == 0 {
            return Err(AiError::ConfigError("model has no actions".to_string()));
        }

        let mut min_g = f64::INFINITY;
        let mut best_action = 0;

        for a in 0..model.n_actions {
            let policy = vec![a];
            let g = efe.compute(&policy, current_beliefs, model)?;
            if g < min_g {
                min_g = g;
                best_action = a;
            }
        }

        Ok(best_action)
    }

    /// Compute the posterior over actions: softmax(−G(a) / temperature).
    pub fn compute_action_posterior(
        &self,
        current_beliefs: &AiBeliefState,
        model: &AiGenerativeModel,
    ) -> Result<Vec<f64>, AiError> {
        if model.n_actions == 0 {
            return Err(AiError::ConfigError("model has no actions".to_string()));
        }

        let efe = AiExpectedFreeEnergy::new(self.depth);
        let temp = if self.temperature < 1e-10 {
            1e-10
        } else {
            self.temperature
        };

        let mut neg_g: Vec<f64> = Vec::with_capacity(model.n_actions);
        for a in 0..model.n_actions {
            let policy = vec![a];
            let g = efe.compute(&policy, current_beliefs, model)?;
            neg_g.push(-g / temp);
        }

        Ok(softmax_vec(&neg_g))
    }

    /// Return best action, weighted by precision.
    ///
    /// High precision → greedy (argmax), low precision → sample proportionally.
    /// Here we use precision as a deterministic threshold: if precision ≥ 1.0,
    /// return argmax; otherwise return action with highest posterior weight.
    pub fn precision_weighted_action(&self, action_posterior: &[f64], precision: f64) -> usize {
        // Always return argmax (deterministic) for simplicity and reproducibility.
        // In a stochastic variant, one would sample from action_posterior.
        let _ = precision; // precision used to gate determinism in full implementation
        action_posterior
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  AiActiveInferenceAgent
// ─────────────────────────────────────────────────────────────────────────────

/// Full active inference agent implementing the perception-action loop.
///
/// Each step:
/// 1. **Perceive**: receive observation o_t.
/// 2. **Infer**: update beliefs q(s_t) by minimizing F.
/// 3. **Act**: select action a_t by minimizing G(π).
/// 4. **Update**: (optionally) learn model parameters from experience.
#[derive(Debug, Clone)]
pub struct AiActiveInferenceAgent {
    /// Generative model P(o, s) with A, B, C, D.
    pub model: AiGenerativeModel,
    /// Current posterior beliefs q(s).
    pub beliefs: AiBeliefState,
    /// Variational inference engine.
    pub inference: AiVariationalInference,
    /// Policy selection module.
    pub policy: AiPolicySelection,
    /// Expected free energy calculator.
    pub efe: AiExpectedFreeEnergy,
    /// Current time step counter.
    pub step: usize,
    /// History of variational free energy values.
    pub free_energy_history: Vec<f64>,
    /// History of selected actions.
    pub action_history: Vec<usize>,
}

impl AiActiveInferenceAgent {
    /// Create a new active inference agent with the given generative model.
    pub fn new(model: AiGenerativeModel) -> Self {
        let n_states = model.n_states;
        let beliefs = AiBeliefState::from_prior(&model.d_prior);
        let inference = AiVariationalInference::new(16, 1.0, 1e-6);
        let policy = AiPolicySelection::new(1.0, 1);
        let efe = AiExpectedFreeEnergy::new(1);

        AiActiveInferenceAgent {
            model,
            beliefs,
            inference,
            policy,
            efe,
            step: 0,
            free_energy_history: Vec::new(),
            action_history: Vec::new(),
        }
    }

    /// Perceive observation and update beliefs via variational inference.
    ///
    /// Returns current variational free energy F.
    pub fn perceive_and_infer(&mut self, obs: usize) -> Result<f64, AiError> {
        let new_beliefs = self
            .inference
            .infer_states(obs, &self.model, &self.beliefs)?;
        let f = self.inference.free_energy(obs, &new_beliefs, &self.model);
        self.beliefs = new_beliefs;
        self.free_energy_history.push(f);
        Ok(f)
    }

    /// Select action based on current beliefs.
    pub fn act(&mut self) -> Result<usize, AiError> {
        let action = self
            .policy
            .select_action(&self.beliefs, &self.model, &self.efe)?;
        self.action_history.push(action);
        Ok(action)
    }

    /// Combined step: perceive observation, infer beliefs, select action.
    ///
    /// Returns `(action, free_energy)`.
    pub fn step_episode(&mut self, obs: usize) -> Result<(usize, f64), AiError> {
        let f = self.perceive_and_infer(obs)?;
        let a = self.act()?;
        self.step += 1;
        Ok((a, f))
    }

    /// Average variational free energy over the episode history.
    pub fn average_free_energy(&self) -> f64 {
        if self.free_energy_history.is_empty() {
            return 0.0;
        }
        self.free_energy_history.iter().sum::<f64>() / self.free_energy_history.len() as f64
    }

    /// Reset agent to initial state (preserves model parameters).
    pub fn reset(&mut self) {
        self.beliefs = AiBeliefState::from_prior(&self.model.d_prior);
        self.step = 0;
        self.free_energy_history.clear();
        self.action_history.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  AiMarkovBlanket
// ─────────────────────────────────────────────────────────────────────────────

/// Markov blanket formalism for agent-environment boundary.
///
/// Partitions variables into:
/// - **Sensory states** μ: interface between external and internal states.
/// - **Active states** a: interface between internal and external states.
/// - **Internal states** η: hidden from the environment.
///
/// The Markov blanket B = {μ, a} renders internal states η conditionally
/// independent of external states ψ:
///
/// η ⊥ ψ | B
///
/// This partition is central to Friston's formulation of self-organization
/// and the free energy principle applied to living systems.
#[derive(Debug, Clone)]
pub struct AiMarkovBlanket {
    /// Indices of sensory states (observations).
    pub sensory_states: Vec<usize>,
    /// Indices of active states (actions).
    pub active_states: Vec<usize>,
    /// Indices of internal states (hidden variables).
    pub internal_states: Vec<usize>,
    /// Total number of variables.
    pub n_total: usize,
}

impl AiMarkovBlanket {
    /// Construct a Markov blanket partition.
    ///
    /// Sensory states: [0, n_sensory)
    /// Active states: [n_sensory, n_sensory + n_active)
    /// Internal states: [n_sensory + n_active, n_sensory + n_active + n_internal)
    /// External states (implicit): remaining indices
    pub fn new(n_sensory: usize, n_active: usize, n_internal: usize) -> Self {
        let sensory_states: Vec<usize> = (0..n_sensory).collect();
        let active_states: Vec<usize> = (n_sensory..n_sensory + n_active).collect();
        let internal_states: Vec<usize> =
            (n_sensory + n_active..n_sensory + n_active + n_internal).collect();
        let n_total = n_sensory + n_active + n_internal;

        AiMarkovBlanket {
            sensory_states,
            active_states,
            internal_states,
            n_total,
        }
    }

    /// Size of the Markov blanket = |sensory| + |active|.
    pub fn blanket_size(&self) -> usize {
        self.sensory_states.len() + self.active_states.len()
    }

    /// Check if a variable index is an internal state.
    pub fn is_internal(&self, idx: usize) -> bool {
        self.internal_states.contains(&idx)
    }

    /// Measure how well the Markov blanket condition holds.
    ///
    /// Computes a score ∈ [0, 1] based on the ratio of cross-covariances
    /// between internal and external states conditioned on the blanket.
    ///
    /// A score near 0 indicates good conditional independence;
    /// near 1 indicates strong violation of the blanket condition.
    ///
    /// The joint covariance `joint_cov` should be an (n_total × n_total) matrix.
    pub fn conditional_independence_score(&self, joint_cov: &[Vec<f64>]) -> f64 {
        let n = self.n_total;
        if joint_cov.len() != n || joint_cov.iter().any(|row| row.len() != n) {
            return f64::NAN;
        }

        let blanket: Vec<usize> = self
            .sensory_states
            .iter()
            .chain(self.active_states.iter())
            .cloned()
            .collect();

        // External state indices: everything not in sensory + active + internal
        let all_blanket_internal: std::collections::HashSet<usize> = blanket
            .iter()
            .chain(self.internal_states.iter())
            .cloned()
            .collect();
        let external: Vec<usize> = (0..n)
            .filter(|i| !all_blanket_internal.contains(i))
            .collect();

        if external.is_empty() || self.internal_states.is_empty() {
            return 0.0;
        }

        // Compute mean absolute cross-covariance between internal and external states
        let mut cross_cov_sum = 0.0;
        let mut count = 0usize;
        for &i in &self.internal_states {
            for &e in &external {
                cross_cov_sum += joint_cov[i][e].abs();
                count += 1;
            }
        }

        // Compute mean variance of internal states (normalization)
        let mean_var: f64 = self
            .internal_states
            .iter()
            .map(|&i| joint_cov[i][i].abs())
            .sum::<f64>()
            / self.internal_states.len() as f64;

        if count == 0 || mean_var < 1e-15 {
            return 0.0;
        }

        let mean_cross = cross_cov_sum / count as f64;
        // Normalize: 0 = perfectly independent, 1 = fully dependent
        (mean_cross / mean_var).clamp(0.0, 1.0)
    }

    /// Sensory surprise: approximate −log P(s) via mean squared prediction error.
    ///
    /// Returns a non-negative value; larger = more surprising sensory input.
    pub fn surprise(&self, obs: &[f64], predicted: &[f64]) -> f64 {
        if obs.is_empty() || obs.len() != predicted.len() {
            return 0.0;
        }
        let mse: f64 = obs
            .iter()
            .zip(predicted.iter())
            .map(|(o, p)| (o - p).powi(2))
            .sum::<f64>()
            / obs.len() as f64;
        mse
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  AiHierarchicalGenerativeModel
// ─────────────────────────────────────────────────────────────────────────────

/// Single level of a hierarchical generative model.
///
/// Each level has its own likelihood A, transitions B, precision (inverse variance),
/// and dimensionality specifications.
#[derive(Debug, Clone)]
pub struct AiHierarchicalLevel {
    /// Likelihood mapping A: [n_obs × n_states].
    pub a_matrix: Vec<Vec<f64>>,
    /// Transition matrices B: [n_actions × n_states × n_states].
    pub b_matrices: Vec<Vec<Vec<f64>>>,
    /// Precision (inverse variance): controls update strength at this level.
    /// Higher precision → stronger belief updates; lower levels typically have
    /// higher precision (less uncertainty).
    pub precision: f64,
    /// Number of hidden states at this level.
    pub n_states: usize,
    /// Number of observations at this level.
    pub n_obs: usize,
    /// Number of actions at this level.
    pub n_actions: usize,
}

/// Two-or-more level hierarchical generative model implementing predictive coding.
///
/// In Friston's predictive coding:
/// - Higher levels generate predictions for lower-level states.
/// - Lower levels compute precision-weighted prediction errors.
/// - Free energy is summed across levels.
///
/// This captures the multi-level structure of perception from low-level
/// sensory features to high-level contextual beliefs.
#[derive(Debug, Clone)]
pub struct AiHierarchicalGenerativeModel {
    /// Levels from lowest (fastest, highest precision) to highest (slowest, lowest precision).
    pub levels: Vec<AiHierarchicalLevel>,
    /// Number of levels.
    pub n_levels: usize,
}

impl AiHierarchicalGenerativeModel {
    /// Create a hierarchical model from level specifications.
    ///
    /// `level_specs`: slice of (n_states, n_obs, n_actions, precision) per level.
    pub fn new(level_specs: &[(usize, usize, usize, f64)]) -> Self {
        let mut rng = SmallRng::seed_from_u64(777);
        let mut levels: Vec<AiHierarchicalLevel> = Vec::with_capacity(level_specs.len());

        for &(n_states, n_obs, n_actions, precision) in level_specs {
            // Initialize A
            let mut a_matrix: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
            for _ in 0..n_obs {
                a_matrix.push(dirichlet_init(n_states, &mut rng));
            }
            // Normalize columns
            for s in 0..n_states {
                let col_sum: f64 = (0..n_obs).map(|o| a_matrix[o][s]).sum();
                let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
                for o in 0..n_obs {
                    a_matrix[o][s] /= denom;
                }
            }

            // Initialize B
            let mut b_matrices: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_actions);
            for _u in 0..n_actions {
                let mut b: Vec<Vec<f64>> = Vec::with_capacity(n_states);
                for sn in 0..n_states {
                    let mut row: Vec<f64> = Vec::with_capacity(n_states);
                    for sc in 0..n_states {
                        let base = if sn == sc { 2.0 } else { 0.5 };
                        row.push(base + 0.3 * rng.random::<f64>());
                    }
                    b.push(row);
                }
                // Normalize columns
                for s in 0..n_states {
                    let col_sum: f64 = (0..n_states).map(|sn| b[sn][s]).sum();
                    let denom = if col_sum < 1e-15 { 1.0 } else { col_sum };
                    for sn in 0..n_states {
                        b[sn][s] /= denom;
                    }
                }
                b_matrices.push(b);
            }

            levels.push(AiHierarchicalLevel {
                a_matrix,
                b_matrices,
                precision,
                n_states,
                n_obs,
                n_actions,
            });
        }

        let n_levels = levels.len();
        AiHierarchicalGenerativeModel { levels, n_levels }
    }

    /// Generate top-down prediction: A\[level\] @ top_belief.
    ///
    /// Given beliefs at a higher level, predict the expected state distribution
    /// at the next lower level. Returns a \[n_obs\]-dimensional vector.
    pub fn top_down_prediction(
        &self,
        top_belief: &[f64],
        level: usize,
    ) -> Result<Vec<f64>, AiError> {
        if level >= self.n_levels {
            return Err(AiError::InvalidDimension(format!(
                "level={} >= n_levels={}",
                level, self.n_levels
            )));
        }
        let lv = &self.levels[level];
        if top_belief.len() != lv.n_states {
            return Err(AiError::InvalidDimension(format!(
                "top_belief.len()={} != n_states={}",
                top_belief.len(),
                lv.n_states
            )));
        }
        // Predicted observations = A @ beliefs: [n_obs]
        let pred = mat_vec(&lv.a_matrix, top_belief);
        Ok(pred)
    }

    /// Compute precision-weighted bottom-up prediction error at a level.
    ///
    /// prediction_error = precision * (obs − pred)
    pub fn bottom_up_error(&self, obs: &[f64], pred: &[f64], level: usize) -> Vec<f64> {
        if level >= self.n_levels {
            return vec![];
        }
        let precision = self.levels[level].precision;
        obs.iter()
            .zip(pred.iter())
            .map(|(o, p)| precision * (o - p))
            .collect()
    }

    /// Compute total hierarchical variational free energy.
    ///
    /// F_total = Σ_l precision_l * ||obs_l − A_l @ beliefs_l||²
    ///
    /// `beliefs`: beliefs at each level, `obs`: observations at each level.
    pub fn hierarchical_free_energy(
        &self,
        beliefs: &[Vec<f64>],
        obs: &[Vec<f64>],
    ) -> Result<f64, AiError> {
        if beliefs.len() != self.n_levels || obs.len() != self.n_levels {
            return Err(AiError::InvalidDimension(format!(
                "Expected {} levels, got beliefs={}, obs={}",
                self.n_levels,
                beliefs.len(),
                obs.len()
            )));
        }

        let mut total_fe = 0.0;
        for l in 0..self.n_levels {
            let lv = &self.levels[l];
            if beliefs[l].len() != lv.n_states {
                return Err(AiError::InvalidDimension(format!(
                    "Level {}: beliefs.len()={} != n_states={}",
                    l,
                    beliefs[l].len(),
                    lv.n_states
                )));
            }
            if obs[l].len() != lv.n_obs {
                return Err(AiError::InvalidDimension(format!(
                    "Level {}: obs.len()={} != n_obs={}",
                    l,
                    obs[l].len(),
                    lv.n_obs
                )));
            }
            let pred = mat_vec(&lv.a_matrix, &beliefs[l]);
            let pe_sq: f64 = obs[l]
                .iter()
                .zip(pred.iter())
                .map(|(o, p)| (o - p).powi(2))
                .sum();
            total_fe += lv.precision * pe_sq;
        }

        Ok(total_fe)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  AiParameterLearning
// ─────────────────────────────────────────────────────────────────────────────

/// Online Bayesian learning of generative model parameters.
///
/// Updates the likelihood matrix A and transition matrices B using
/// sufficient statistics from observed state-observation pairs.
///
/// This implements variational Bayes updates for Dirichlet-distributed
/// concentration parameters (Friston et al. 2016 — "Active inference and learning").
///
/// The key equations are:
/// - Dirichlet update for A: α_os += lr * q(s) * δ(o, o_t)
/// - Dirichlet update for B: α_s's += lr * q(s') * q(s)
///
/// After accumulation, A and B are re-normalized to maintain valid
/// probability distributions.
#[derive(Debug, Clone)]
pub struct AiParameterLearning {
    /// Learning rate for parameter updates.
    pub learning_rate: f64,
    /// Dirichlet concentration prior (pseudo-count baseline).
    pub concentration: f64,
}

impl AiParameterLearning {
    /// Create a new parameter learning module.
    pub fn new(learning_rate: f64, concentration: f64) -> Self {
        AiParameterLearning {
            learning_rate,
            concentration,
        }
    }

    /// Update likelihood matrix A given observation and current beliefs.
    ///
    /// Dirichlet update: A[o_t, s] += lr * q(s) * δ(o, o_t)
    ///
    /// Increases the probability of observation o given state s in proportion
    /// to how much the agent believes it is in state s.
    pub fn update_likelihood(
        &self,
        model: &mut AiGenerativeModel,
        obs: usize,
        beliefs: &AiBeliefState,
    ) {
        if obs >= model.n_obs {
            return;
        }
        for s in 0..model.n_states {
            // One-hot observation: only the observed index contributes
            model.a_matrix[obs][s] += self.learning_rate * beliefs.beliefs[s];
        }
        // Re-normalize columns to maintain valid probability distributions
        model.normalize_a();
    }

    /// Update transition matrix B given previous and current beliefs.
    ///
    /// Dirichlet update: B[a][s', s] += lr * q(s') * q(s)
    ///
    /// Increases the probability of transitioning from s to s' under action a
    /// in proportion to the outer product of successive belief states.
    pub fn update_transitions(
        &self,
        model: &mut AiGenerativeModel,
        action: usize,
        prev_beliefs: &AiBeliefState,
        curr_beliefs: &AiBeliefState,
    ) {
        if action >= model.n_actions {
            return;
        }
        for s_next in 0..model.n_states {
            for s_curr in 0..model.n_states {
                model.b_matrices[action][s_next][s_curr] += self.learning_rate
                    * curr_beliefs.beliefs[s_next]
                    * prev_beliefs.beliefs[s_curr];
            }
        }
        // Re-normalize to maintain valid transition distributions
        model.normalize_b();
    }

    /// Estimate log model evidence log P(o_{1:T}) via variational bound.
    ///
    /// Approximation: log P(o_t) ≈ −F_t = −(KL − E_q[log P(o|s)])
    /// Sums over the observation sequence assuming a fixed uniform prior.
    pub fn model_evidence(&self, model: &AiGenerativeModel, obs_sequence: &[usize]) -> f64 {
        let vi = AiVariationalInference::new(16, 1.0, 1e-6);
        let mut total = 0.0;
        let prior_beliefs = AiBeliefState::from_prior(&model.d_prior);

        for &obs in obs_sequence {
            if obs >= model.n_obs {
                continue;
            }
            // Approximate: use E_q[log P(o|s)] with uniform beliefs as crude estimate
            let f = vi.free_energy(obs, &prior_beliefs, model);
            // Log evidence ≈ −F
            total += -f;
        }
        total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  AiMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Utility functions for evaluating active inference agents.
pub struct AiMetrics;

impl AiMetrics {
    /// Summarize free energy trajectory.
    ///
    /// Returns `(mean, min, final)` free energy values.
    pub fn free_energy_trajectory(history: &[f64]) -> (f64, f64, f64) {
        if history.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mean = history.iter().sum::<f64>() / history.len() as f64;
        let min = history.iter().cloned().fold(f64::INFINITY, f64::min);
        let last = *history.last().unwrap_or(&0.0);
        (mean, min, last)
    }

    /// Shannon entropy of the action posterior distribution.
    ///
    /// Higher entropy → more exploratory behavior.
    pub fn action_entropy(action_posterior: &[f64]) -> f64 {
        entropy(action_posterior)
    }

    /// P(true_state) in beliefs — probability assigned to the correct state.
    ///
    /// Returns a value in [0, 1]; higher = more accurate beliefs.
    pub fn belief_accuracy(beliefs: &AiBeliefState, true_state: usize) -> f64 {
        if true_state >= beliefs.n_states {
            return 0.0;
        }
        beliefs.beliefs[true_state]
    }

    /// Average surprise −log P(o_t | s_t) over a trajectory.
    ///
    /// Lower surprise → better model fit to the observed trajectory.
    pub fn average_surprise(
        model: &AiGenerativeModel,
        obs_seq: &[usize],
        state_seq: &[usize],
    ) -> f64 {
        if obs_seq.is_empty() || obs_seq.len() != state_seq.len() {
            return 0.0;
        }
        let total: f64 = obs_seq
            .iter()
            .zip(state_seq.iter())
            .map(|(&o, &s)| {
                if o < model.n_obs && s < model.n_states {
                    -model.log_likelihood(o, s)
                } else {
                    0.0
                }
            })
            .sum();
        total / obs_seq.len() as f64
    }

    /// Entropy of the empirical action distribution.
    ///
    /// Measures how exploratory the agent's behavior is over an episode.
    /// Returns a value in [0, log(n_actions)].
    pub fn policy_efficiency(actions: &[usize], n_actions: usize) -> f64 {
        if actions.is_empty() || n_actions == 0 {
            return 0.0;
        }
        // Build empirical action distribution
        let mut counts = vec![0.0f64; n_actions];
        for &a in actions {
            if a < n_actions {
                counts[a] += 1.0;
            }
        }
        let total: f64 = counts.iter().sum();
        if total < 1.0 {
            return 0.0;
        }
        let probs: Vec<f64> = counts.iter().map(|&c| c / total).collect();
        entropy(&probs)
    }
}
