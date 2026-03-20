//! Core POMDP types: PomdpModel, BeliefState, AlphaVector

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

// ─── Type Aliases ─────────────────────────────────────────────────────────────

/// State index (0..n_states)
pub type PomdpState = usize;
/// Action index (0..n_actions)
pub type PomdpAction = usize;
/// Observation index (0..n_obs)
pub type PomdpObs = usize;

// ─── Error Type ───────────────────────────────────────────────────────────────

/// Error type for POMDP planning operations.
#[derive(Debug, Clone)]
pub struct PomdpError(pub String);

impl std::fmt::Display for PomdpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PomdpError: {}", self.0)
    }
}

impl std::error::Error for PomdpError {}

pub type Result<T> = std::result::Result<T, PomdpError>;

// ─── 1. PomdpModel ────────────────────────────────────────────────────────────

/// Core POMDP model: (S, A, O, T, Z, R, γ)
///
/// - T\[s\]\[a\][s'] = P(s' | s, a)
/// - Z\[a\]\[s'\]\[o\] = P(o | a, s')
/// - R\[s\]\[a\] = expected immediate reward
#[derive(Debug, Clone)]
pub struct PomdpModel {
    /// Number of states
    pub n_states: usize,
    /// Number of actions
    pub n_actions: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Transition: T\[s\]\[a\][s'] = P(s'|s,a)
    pub transition: Vec<Vec<Vec<f64>>>,
    /// Observation: Z\[a\]\[s'\]\[o\] = P(o|a,s')
    pub observation: Vec<Vec<Vec<f64>>>,
    /// Reward: R\[s\]\[a\]
    pub reward: Vec<Vec<f64>>,
    /// Discount factor γ ∈ (0,1]
    pub gamma: f64,
}

impl PomdpModel {
    /// Create a new POMDP model with zero-initialized tables.
    pub fn new(n_states: usize, n_actions: usize, n_obs: usize, gamma: f64) -> Self {
        Self {
            n_states,
            n_actions,
            n_obs,
            transition: vec![vec![vec![0.0; n_states]; n_actions]; n_states],
            observation: vec![vec![vec![0.0; n_obs]; n_states]; n_actions],
            reward: vec![vec![0.0; n_actions]; n_states],
            gamma,
        }
    }

    /// Set transition probability P(s'|s,a).
    pub fn set_transition(&mut self, s: usize, a: usize, sp: usize, prob: f64) {
        if s < self.n_states && a < self.n_actions && sp < self.n_states {
            self.transition[s][a][sp] = prob;
        }
    }

    /// Set observation probability P(o|a,s').
    pub fn set_observation(&mut self, a: usize, sp: usize, o: usize, prob: f64) {
        if a < self.n_actions && sp < self.n_states && o < self.n_obs {
            self.observation[a][sp][o] = prob;
        }
    }

    /// Set reward R(s,a).
    pub fn set_reward(&mut self, s: usize, a: usize, r: f64) {
        if s < self.n_states && a < self.n_actions {
            self.reward[s][a] = r;
        }
    }

    /// Validate that all distributions sum to 1 (within tolerance).
    pub fn validate(&self) -> Result<()> {
        let tol = 1e-6;
        for s in 0..self.n_states {
            for a in 0..self.n_actions {
                let sum: f64 = self.transition[s][a].iter().sum();
                if (sum - 1.0).abs() > tol && sum > tol {
                    return Err(PomdpError(format!(
                        "T[{s}][{a}] sums to {sum:.6}, expected 1.0"
                    )));
                }
            }
        }
        for a in 0..self.n_actions {
            for sp in 0..self.n_states {
                let sum: f64 = self.observation[a][sp].iter().sum();
                if (sum - 1.0).abs() > tol && sum > tol {
                    return Err(PomdpError(format!(
                        "Z[{a}][{sp}] sums to {sum:.6}, expected 1.0"
                    )));
                }
            }
        }
        Ok(())
    }
}

// ─── 2. BeliefState ───────────────────────────────────────────────────────────

/// Probability distribution over states: b\[s\] = P(S = s).
#[derive(Debug, Clone)]
pub struct BeliefState {
    /// Probability vector (sums to 1).
    pub prob: Vec<f64>,
}

impl BeliefState {
    /// Create a uniform belief over `n` states.
    pub fn uniform(n: usize) -> Self {
        let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        Self { prob: vec![p; n] }
    }

    /// Create a point mass belief at state `s`.
    pub fn point_mass(n: usize, s: usize) -> Self {
        let mut prob = vec![0.0; n];
        if s < n {
            prob[s] = 1.0;
        }
        Self { prob }
    }

    /// Create from a raw probability vector (normalized).
    pub fn from_vec(mut v: Vec<f64>) -> Self {
        let sum: f64 = v.iter().sum();
        if sum > 1e-15 {
            v.iter_mut().for_each(|x| *x /= sum);
        } else {
            let n = v.len();
            let p = if n > 0 { 1.0 / n as f64 } else { 0.0 };
            v.iter_mut().for_each(|x| *x = p);
        }
        Self { prob: v }
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.prob.len()
    }

    /// Bayesian belief update: b'(s') ∝ Z(a,s',o) · Σ_s T(s,a,s') b(s)
    pub fn update(&self, a: PomdpAction, o: PomdpObs, model: &PomdpModel) -> Self {
        let n = model.n_states;
        let mut new_prob = vec![0.0f64; n];
        for sp in 0..n {
            let predicted: f64 = (0..n)
                .map(|s| self.prob[s] * model.transition[s][a][sp])
                .sum();
            let z = if a < model.n_actions && sp < n && o < model.n_obs {
                model.observation[a][sp][o]
            } else {
                0.0
            };
            new_prob[sp] = predicted * z;
        }
        Self::from_vec(new_prob)
    }

    /// Shannon entropy H(b) = -Σ b(s) log b(s).
    pub fn entropy(&self) -> f64 {
        self.prob
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Return the state with highest probability.
    pub fn most_likely_state(&self) -> PomdpState {
        self.prob
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Sample a state according to the belief distribution.
    pub fn sample(&self, rng: &mut StdRng) -> PomdpState {
        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        for (s, &p) in self.prob.iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                return s;
            }
        }
        self.prob.len().saturating_sub(1)
    }

    /// L1 distance between two beliefs.
    pub fn distance(&self, other: &BeliefState) -> f64 {
        self.prob
            .iter()
            .zip(other.prob.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }
}

// ─── 3. AlphaVector ───────────────────────────────────────────────────────────

/// Hyperplane in belief space representing a policy fragment.
/// V(b) = max_α (α · b)
#[derive(Debug, Clone)]
pub struct AlphaVector {
    /// Action this vector is associated with.
    pub action: PomdpAction,
    /// Coefficients: one per state.
    pub coeffs: Vec<f64>,
}

impl AlphaVector {
    /// Create a new alpha vector.
    pub fn new(action: PomdpAction, coeffs: Vec<f64>) -> Self {
        Self { action, coeffs }
    }

    /// Value V(b) = α · b (dot product).
    pub fn value(&self, belief: &BeliefState) -> f64 {
        self.coeffs
            .iter()
            .zip(belief.prob.iter())
            .map(|(c, p)| c * p)
            .sum()
    }

    /// Compute the best alpha vector value for a given belief.
    pub fn max_value(alphas: &[AlphaVector], belief: &BeliefState) -> (f64, usize) {
        if alphas.is_empty() {
            return (f64::NEG_INFINITY, 0);
        }
        alphas
            .iter()
            .enumerate()
            .map(|(i, a)| (a.value(belief), i))
            .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((f64::NEG_INFINITY, 0))
    }

    /// Check if this vector is dominated by the set at all sampled beliefs.
    pub fn is_dominated(&self, others: &[AlphaVector], beliefs: &[BeliefState]) -> bool {
        if beliefs.is_empty() {
            return false;
        }
        beliefs.iter().all(|b| {
            let my_val = self.value(b);
            others
                .iter()
                .any(|o| std::ptr::eq(o, self) || o.value(b) > my_val)
        })
    }
}
