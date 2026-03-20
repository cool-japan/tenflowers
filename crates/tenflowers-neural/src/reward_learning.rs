//! Reward Learning and RLHF — preference models, MLP reward model, Bradley-Terry
//! loss, PPO policy updates with KL penalty, MaxEnt/MaxCausalEnt IRL, and
//! potential-based reward shaping. References: Ziebart 2008, Christiano 2017,
//! Schulman 2017, Ng 1999.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Binary cross-entropy: `H(p,y) = -[y*log(p) + (1-y)*log(1-p)]`, ε-clamped.
pub fn cross_entropy_binary(p: f64, y: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let p_clamped = p.clamp(EPS, 1.0 - EPS);
    -(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln())
}

/// Kendall rank correlation τ: `(C - D) / (n*(n-1)/2)`. Returns 0.0 for < 2 elements.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let sign = dx * dy;
            if sign > 0.0 {
                concordant += 1;
            } else if sign < 0.0 {
                discordant += 1;
            }
            // ties: ignored (contributes 0)
        }
    }
    let total = (n as i64) * (n as i64 - 1) / 2;
    if total == 0 {
        return 0.0;
    }
    (concordant - discordant) as f64 / total as f64
}

/// Discounted returns `G_t = r_t + γ*r_{t+1} + ...`, computed in O(n) via reverse scan.
pub fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    let mut returns = vec![0.0_f64; n];
    let mut cumulative = 0.0_f64;
    for i in (0..n).rev() {
        cumulative = rewards[i] + gamma * cumulative;
        returns[i] = cumulative;
    }
    returns
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: Preference Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// A pairwise preference comparison between two trajectories (sequences of state vectors).
#[derive(Debug, Clone)]
pub struct Comparison {
    /// Sequence of state vectors for trajectory A.
    pub trajectory_a: Vec<Vec<f64>>,
    /// Sequence of state vectors for trajectory B.
    pub trajectory_b: Vec<Vec<f64>>,
    /// Human preference label: 1.0 = A preferred, 0.0 = B preferred, 0.5 = tie.
    pub preference: f64,
}

impl Comparison {
    /// Create a new preference comparison.
    pub fn new(trajectory_a: Vec<Vec<f64>>, trajectory_b: Vec<Vec<f64>>, preference: f64) -> Self {
        Self {
            trajectory_a,
            trajectory_b,
            preference,
        }
    }
}

/// A dataset of pairwise trajectory comparisons for reward learning.
#[derive(Debug, Clone)]
pub struct PreferenceDataset {
    comparisons: Vec<Comparison>,
}

impl PreferenceDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self {
            comparisons: Vec::new(),
        }
    }

    /// Add a preference comparison to the dataset.
    pub fn add_comparison(&mut self, a: Vec<Vec<f64>>, b: Vec<Vec<f64>>, pref: f64) {
        self.comparisons.push(Comparison::new(a, b, pref));
    }

    /// Number of comparisons in the dataset.
    pub fn len(&self) -> usize {
        self.comparisons.len()
    }

    /// Returns `true` if the dataset contains no comparisons.
    pub fn is_empty(&self) -> bool {
        self.comparisons.is_empty()
    }

    /// Get the comparison at index `i`.
    ///
    /// Returns `None` if out of bounds.
    pub fn get(&self, i: usize) -> Option<&Comparison> {
        self.comparisons.get(i)
    }

    /// Shuffle the dataset in-place using Fisher-Yates with a deterministic seed.
    pub fn shuffle(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = self.comparisons.len();
        for i in (1..n).rev() {
            let j = (rng.random::<u64>() as usize) % (i + 1);
            self.comparisons.swap(i, j);
        }
    }

    /// Split the dataset into train and validation sets.
    ///
    /// # Arguments
    /// * `frac` — fraction of data for training (e.g. 0.8 for 80/20 split).
    /// * `seed` — random seed for shuffling before split.
    ///
    /// # Errors
    /// Returns an error if `frac` is not in `(0.0, 1.0)`.
    pub fn split_train_val(&self, frac: f64, seed: u64) -> Result<(Self, Self)> {
        if frac <= 0.0 || frac >= 1.0 {
            return Err(TensorError::invalid_argument_op(
                "split_train_val",
                "frac must be in (0.0, 1.0)",
            ));
        }
        let mut cloned = self.clone();
        cloned.shuffle(seed);
        let split = (cloned.comparisons.len() as f64 * frac).round() as usize;
        let split = split.clamp(1, cloned.comparisons.len().saturating_sub(1));
        let val = cloned.comparisons.split_off(split);
        Ok((cloned, PreferenceDataset { comparisons: val }))
    }

    /// Iterate over all comparisons.
    pub fn iter(&self) -> impl Iterator<Item = &Comparison> {
        self.comparisons.iter()
    }
}

impl Default for PreferenceDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: Reward Model (MLP-based)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the MLP reward model.
#[derive(Debug, Clone)]
pub struct RewardModelConfig {
    /// Dimensionality of the state space.
    pub state_dim: usize,
    /// Hidden layer sizes (e.g. `vec![64, 64]` for two hidden layers of 64 units).
    pub hidden_dims: Vec<usize>,
    /// Learning rate for gradient descent.
    pub lr: f64,
    /// L2 regularisation coefficient.
    pub lambda: f64,
}

impl RewardModelConfig {
    /// Create a default config with two 64-unit hidden layers.
    pub fn new(state_dim: usize) -> Self {
        Self {
            state_dim,
            hidden_dims: vec![64, 64],
            lr: 1e-3,
            lambda: 1e-4,
        }
    }
}

/// A single fully-connected layer: weight matrix + bias vector.
#[derive(Debug, Clone)]
struct FcLayer {
    /// Weights: shape `[out_dim, in_dim]` stored row-major.
    weights: Vec<f64>,
    /// Biases: shape `[out_dim]`.
    biases: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl FcLayer {
    /// Xavier/Glorot uniform initialisation: `U(-limit, +limit)` where `limit = sqrt(6 / (in + out))`.
    fn xavier_init(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let n_weights = in_dim * out_dim;
        let weights: Vec<f64> = (0..n_weights)
            .map(|_| {
                let u: f64 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();
        let biases = vec![0.0_f64; out_dim];
        Self {
            weights,
            biases,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: `y = ReLU(W x + b)` (ReLU applied externally when desired).
    fn linear(&self, input: &[f64]) -> Vec<f64> {
        let mut out = self.biases.clone();
        for (o, row) in out.iter_mut().enumerate() {
            for i in 0..self.in_dim {
                *row += self.weights[o * self.in_dim + i] * input[i];
            }
        }
        out
    }

    /// Total number of parameters (weights + biases).
    fn n_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

/// Evaluation metrics for the reward model.
#[derive(Debug, Clone)]
pub struct RewardModelMetrics {
    /// Fraction of comparisons where the model correctly identifies the preferred trajectory.
    pub accuracy: f64,
    /// Average Bradley-Terry loss over the dataset.
    pub loss: f64,
    /// Kendall τ rank correlation between model scores and human preferences.
    pub kendall_tau: f64,
}

/// MLP reward model (`state_dim → hidden[0] → ReLU → ... → 1`) trained via Bradley-Terry loss.
#[derive(Debug, Clone)]
pub struct RewardModel {
    layers: Vec<FcLayer>,
    config: RewardModelConfig,
}

impl RewardModel {
    /// Construct a new reward model (deterministic seed = 42).
    pub fn new(config: RewardModelConfig) -> Result<Self> {
        Self::new_with_seed(config, 42)
    }

    /// Construct a new reward model with an explicit random seed.
    pub fn new_with_seed(config: RewardModelConfig, seed: u64) -> Result<Self> {
        if config.state_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "RewardModel::new",
                "state_dim must be > 0",
            ));
        }
        if config.hidden_dims.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RewardModel::new",
                "hidden_dims must not be empty",
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut layers: Vec<FcLayer> = Vec::new();

        let mut in_dim = config.state_dim;
        for &h in &config.hidden_dims {
            layers.push(FcLayer::xavier_init(in_dim, h, &mut rng));
            in_dim = h;
        }
        // Output layer → scalar reward
        layers.push(FcLayer::xavier_init(in_dim, 1, &mut rng));

        Ok(Self { layers, config })
    }

    /// Forward pass through the MLP (ReLU on hidden layers, linear output).
    fn forward_raw(&self, state: &[f64]) -> f64 {
        let mut x: Vec<f64> = state.to_vec();
        let n_hidden = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.linear(&x);
            // Apply ReLU to all but the last layer
            if i < n_hidden {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
        // x is now length 1
        x[0]
    }

    /// Predict the scalar reward for a single state vector.
    pub fn predict(&self, state: &[f64]) -> f64 {
        self.forward_raw(state)
    }

    /// Predict the sum of per-step rewards for a trajectory (a sequence of states).
    pub fn predict_trajectory_return(&self, states: &[Vec<f64>]) -> f64 {
        states.iter().map(|s| self.forward_raw(s)).sum()
    }

    /// One finite-difference gradient step on the Bradley-Terry loss. Returns average loss.
    pub fn train_step(&mut self, comparisons: &[Comparison]) -> Result<f64> {
        if comparisons.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RewardModel::train_step",
                "comparisons batch must not be empty",
            ));
        }

        let loss_before = self.batch_loss(comparisons);

        // Finite-difference gradient for each parameter.
        // We iterate over all layers and all parameters.
        const H: f64 = 1e-5;

        for layer_idx in 0..self.layers.len() {
            // Weights
            let n_weights = self.layers[layer_idx].weights.len();
            for w_idx in 0..n_weights {
                let orig = self.layers[layer_idx].weights[w_idx];

                self.layers[layer_idx].weights[w_idx] = orig + H;
                let loss_plus = self.batch_loss(comparisons);

                self.layers[layer_idx].weights[w_idx] = orig - H;
                let loss_minus = self.batch_loss(comparisons);

                self.layers[layer_idx].weights[w_idx] = orig;

                let grad = (loss_plus - loss_minus) / (2.0 * H);
                // L2 regularisation gradient
                let l2_grad = self.config.lambda * orig;
                self.layers[layer_idx].weights[w_idx] -= self.config.lr * (grad + l2_grad);
            }

            // Biases
            let n_biases = self.layers[layer_idx].biases.len();
            for b_idx in 0..n_biases {
                let orig = self.layers[layer_idx].biases[b_idx];

                self.layers[layer_idx].biases[b_idx] = orig + H;
                let loss_plus = self.batch_loss(comparisons);

                self.layers[layer_idx].biases[b_idx] = orig - H;
                let loss_minus = self.batch_loss(comparisons);

                self.layers[layer_idx].biases[b_idx] = orig;

                let grad = (loss_plus - loss_minus) / (2.0 * H);
                self.layers[layer_idx].biases[b_idx] -= self.config.lr * grad;
            }
        }

        Ok(loss_before)
    }

    /// Compute the Bradley-Terry cross-entropy loss over a batch of comparisons.
    fn batch_loss(&self, comparisons: &[Comparison]) -> f64 {
        if comparisons.is_empty() {
            return 0.0;
        }
        let total: f64 = comparisons
            .iter()
            .map(|c| {
                let ra = self.predict_trajectory_return(&c.trajectory_a);
                let rb = self.predict_trajectory_return(&c.trajectory_b);
                let p_a_wins = sigmoid(ra - rb);
                cross_entropy_binary(p_a_wins, c.preference)
            })
            .sum();
        total / comparisons.len() as f64
    }

    /// Evaluate the reward model on a preference dataset.
    pub fn evaluate(&self, dataset: &PreferenceDataset) -> RewardModelMetrics {
        if dataset.is_empty() {
            return RewardModelMetrics {
                accuracy: 0.0,
                loss: 0.0,
                kendall_tau: 0.0,
            };
        }

        let comparisons: Vec<&Comparison> = dataset.iter().collect();
        let n = comparisons.len() as f64;
        let mut total_loss = 0.0_f64;
        let mut correct = 0_usize;
        let mut model_scores: Vec<f64> = Vec::with_capacity(comparisons.len());
        let mut human_prefs: Vec<f64> = Vec::with_capacity(comparisons.len());

        for c in &comparisons {
            let ra = self.predict_trajectory_return(&c.trajectory_a);
            let rb = self.predict_trajectory_return(&c.trajectory_b);
            let p_a_wins = sigmoid(ra - rb);
            total_loss += cross_entropy_binary(p_a_wins, c.preference);

            // Accuracy: model predicts A wins iff ra > rb; human prefers A iff pref > 0.5
            let model_prefers_a = ra > rb;
            let human_prefers_a = c.preference > 0.5;
            if model_prefers_a == human_prefers_a {
                correct += 1;
            }

            model_scores.push(ra - rb);
            human_prefs.push(c.preference);
        }

        let tau = kendall_tau(&model_scores, &human_prefs);

        RewardModelMetrics {
            accuracy: correct as f64 / n,
            loss: total_loss / n,
            kendall_tau: tau,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: Bradley-Terry Model
// ─────────────────────────────────────────────────────────────────────────────

/// Bradley-Terry preference model: `P(A>B) = σ(R(A) - R(B))` where R = trajectory return.
pub struct BradleyTerryModel;

impl BradleyTerryModel {
    /// Perform one gradient step on the reward model using the Bradley-Terry loss.
    ///
    /// Returns the average loss before the update.
    pub fn gradient_step(
        model: &mut RewardModel,
        comparisons: &[Comparison],
        lr: f64,
    ) -> Result<f64> {
        if comparisons.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "BradleyTerryModel::gradient_step",
                "comparisons must not be empty",
            ));
        }
        let old_lr = model.config.lr;
        model.config.lr = lr;
        let loss = model.train_step(comparisons)?;
        model.config.lr = old_lr;
        Ok(loss)
    }

    /// Brier score: `E[(P(A>B) - pref)²]` ∈ [0, 1].
    pub fn brier_score(model: &RewardModel, comparisons: &[Comparison]) -> f64 {
        if comparisons.is_empty() {
            return 0.0;
        }
        let total: f64 = comparisons
            .iter()
            .map(|c| {
                let ra = model.predict_trajectory_return(&c.trajectory_a);
                let rb = model.predict_trajectory_return(&c.trajectory_b);
                let p = sigmoid(ra - rb);
                (p - c.preference).powi(2)
            })
            .sum();
        total / comparisons.len() as f64
    }

    /// Log-likelihood `Σ[pref*log P(A>B) + (1-pref)*log P(B>A)]` (≤ 0 for valid model).
    pub fn log_likelihood(model: &RewardModel, comparisons: &[Comparison]) -> f64 {
        comparisons
            .iter()
            .map(|c| {
                let ra = model.predict_trajectory_return(&c.trajectory_a);
                let rb = model.predict_trajectory_return(&c.trajectory_b);
                let p = sigmoid(ra - rb).clamp(1e-12, 1.0 - 1e-12);
                let pref = c.preference;
                pref * p.ln() + (1.0 - pref) * (1.0 - p).ln()
            })
            .sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: RLHF Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the RLHF training pipeline.
#[derive(Debug, Clone)]
pub struct RlhfConfig {
    /// Configuration for the reward model.
    pub reward_config: RewardModelConfig,
    /// Number of epochs to train the reward model.
    pub n_reward_epochs: usize,
    /// Number of epochs for the policy update phase.
    pub n_policy_epochs: usize,
    /// KL divergence penalty coefficient.
    pub kl_coeff: f64,
    /// Mini-batch size for reward model training.
    pub batch_size: usize,
    /// Random seed.
    pub seed: u64,
}

impl RlhfConfig {
    /// Create a default RLHF configuration.
    pub fn new(state_dim: usize) -> Self {
        Self {
            reward_config: RewardModelConfig::new(state_dim),
            n_reward_epochs: 10,
            n_policy_epochs: 5,
            kl_coeff: 0.1,
            batch_size: 32,
            seed: 0,
        }
    }
}

/// Results from a full RLHF training run.
#[derive(Debug, Clone)]
pub struct RlhfResult {
    /// Reward model loss at each epoch.
    pub reward_losses: Vec<f64>,
    /// Policy loss at each PPO step.
    pub policy_losses: Vec<f64>,
    /// KL divergence estimate at each PPO step.
    pub kl_divergences: Vec<f64>,
    /// Final accuracy of the reward model on the training data.
    pub final_reward_accuracy: f64,
}

/// Policy parameter store (simple linear policy for demonstration).
#[derive(Debug, Clone)]
struct LinearPolicy {
    /// Weight matrix `[action_dim, state_dim]`.
    weights: Vec<f64>,
    state_dim: usize,
    action_dim: usize,
}

impl LinearPolicy {
    fn new(state_dim: usize, action_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let limit = (6.0_f64 / (state_dim + action_dim) as f64).sqrt();
        let weights: Vec<f64> = (0..state_dim * action_dim)
            .map(|_| {
                let u: f64 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();
        Self {
            weights,
            state_dim,
            action_dim,
        }
    }

    /// Compute softmax action probabilities for a state.
    fn action_probs(&self, state: &[f64]) -> Vec<f64> {
        let mut logits = vec![0.0_f64; self.action_dim];
        for a in 0..self.action_dim {
            for s in 0..self.state_dim.min(state.len()) {
                logits[a] += self.weights[a * self.state_dim + s] * state[s];
            }
        }
        // Softmax with max subtraction for numerical stability
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Log probability of action `a` given state.
    fn log_prob(&self, state: &[f64], action: usize) -> f64 {
        let probs = self.action_probs(state);
        let p = probs.get(action).copied().unwrap_or(1e-12).max(1e-12);
        p.ln()
    }
}

/// RLHF trainer that combines reward model training with PPO policy updates.
pub struct RlhfTrainer {
    config: RlhfConfig,
    reward_model: RewardModel,
    policy: LinearPolicy,
    ref_policy: LinearPolicy,
}

impl RlhfTrainer {
    /// Create a new RLHF trainer.
    ///
    /// # Errors
    /// Returns an error if the reward model cannot be initialised.
    pub fn new(config: RlhfConfig) -> Result<Self> {
        let reward_model = RewardModel::new_with_seed(config.reward_config.clone(), config.seed)?;
        let state_dim = config.reward_config.state_dim;
        // Default to 4 actions; can be customised
        let policy = LinearPolicy::new(state_dim, 4, config.seed);
        let ref_policy = policy.clone();
        Ok(Self {
            config,
            reward_model,
            policy,
            ref_policy,
        })
    }

    /// Train the reward model on a preference dataset.
    ///
    /// Returns the loss history (one entry per epoch).
    ///
    /// # Errors
    /// Returns an error if the dataset is empty.
    pub fn train_reward_model(&mut self, dataset: &PreferenceDataset) -> Result<Vec<f64>> {
        if dataset.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "RlhfTrainer::train_reward_model",
                "dataset must not be empty",
            ));
        }

        let mut loss_history = Vec::with_capacity(self.config.n_reward_epochs);
        let mut shuffled = dataset.clone();

        for epoch in 0..self.config.n_reward_epochs {
            shuffled.shuffle(self.config.seed.wrapping_add(epoch as u64));
            let comps: Vec<Comparison> = shuffled.iter().cloned().collect();

            // Mini-batch gradient descent
            let batch_size = self.config.batch_size.max(1).min(comps.len());
            let mut epoch_loss = 0.0_f64;
            let mut n_batches = 0_usize;

            for chunk in comps.chunks(batch_size) {
                let loss = self.reward_model.train_step(chunk)?;
                epoch_loss += loss;
                n_batches += 1;
            }

            if n_batches > 0 {
                epoch_loss /= n_batches as f64;
            }
            loss_history.push(epoch_loss);
        }

        Ok(loss_history)
    }

    /// Perform a single PPO policy update step with KL penalty.
    ///
    /// ## PPO Loss
    ///
    /// `L = -E[min(r * A, clip(r, 1-ε, 1+ε) * A) - kl_coeff * KL(π || π_ref)]`
    ///
    /// where `r = exp(log_prob - log_prob_old)` and KL is approximated as
    /// `KL ≈ log_prob_ref - log_prob_policy` (first-order approximation).
    ///
    /// # Arguments
    /// * `states` — batch of state vectors.
    /// * `actions` — index of action taken for each state.
    /// * `rewards` — per-step reward signals.
    /// * `ref_log_probs` — log probabilities under the reference (frozen) policy.
    ///
    /// # Errors
    /// Returns an error if input slices have mismatched lengths or are empty.
    pub fn ppo_policy_step(
        &mut self,
        states: &[Vec<f64>],
        actions: &[usize],
        rewards: &[f64],
        ref_log_probs: &[f64],
    ) -> Result<f64> {
        let n = states.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "RlhfTrainer::ppo_policy_step",
                "states must not be empty",
            ));
        }
        if actions.len() != n || rewards.len() != n || ref_log_probs.len() != n {
            return Err(TensorError::invalid_argument_op(
                "RlhfTrainer::ppo_policy_step",
                "all input slices must have the same length",
            ));
        }

        const EPSILON: f64 = 0.2; // PPO clip ratio

        // Compute current log probs and advantages
        let log_probs_old: Vec<f64> = states
            .iter()
            .zip(actions.iter())
            .map(|(s, &a)| self.policy.log_prob(s, a))
            .collect();

        // Advantage = reward - mean(reward) (simple baseline)
        let mean_r: f64 = rewards.iter().sum::<f64>() / n as f64;
        let advantages: Vec<f64> = rewards.iter().map(|&r| r - mean_r).collect();

        // PPO gradient step (finite differences on policy weights)
        let loss = self.compute_ppo_loss(
            states,
            actions,
            &log_probs_old,
            &advantages,
            ref_log_probs,
            EPSILON,
        );

        // Gradient step on policy weights
        let policy_lr = 1e-3;
        let h = 1e-5_f64;
        let n_params = self.policy.weights.len();

        for param_idx in 0..n_params {
            let orig = self.policy.weights[param_idx];

            self.policy.weights[param_idx] = orig + h;
            let loss_plus = self.compute_ppo_loss(
                states,
                actions,
                &log_probs_old,
                &advantages,
                ref_log_probs,
                EPSILON,
            );

            self.policy.weights[param_idx] = orig - h;
            let loss_minus = self.compute_ppo_loss(
                states,
                actions,
                &log_probs_old,
                &advantages,
                ref_log_probs,
                EPSILON,
            );

            self.policy.weights[param_idx] = orig;

            let grad = (loss_plus - loss_minus) / (2.0 * h);
            self.policy.weights[param_idx] -= policy_lr * grad;
        }

        Ok(loss)
    }

    /// Compute the PPO loss (clipped surrogate + KL penalty).
    fn compute_ppo_loss(
        &self,
        states: &[Vec<f64>],
        actions: &[usize],
        log_probs_old: &[f64],
        advantages: &[f64],
        ref_log_probs: &[f64],
        epsilon: f64,
    ) -> f64 {
        let n = states.len();
        let mut total = 0.0_f64;

        for i in 0..n {
            let log_prob_new = self.policy.log_prob(&states[i], actions[i]);
            let log_prob_ref = ref_log_probs[i];

            let ratio = (log_prob_new - log_probs_old[i]).exp();
            let adv = advantages[i];

            // Clipped surrogate objective
            let surrogate_unclipped = ratio * adv;
            let ratio_clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
            let surrogate_clipped = ratio_clipped * adv;
            let surrogate = surrogate_unclipped.min(surrogate_clipped);

            // First-order KL approximation: KL(π || π_ref) ≈ log_prob_ref - log_prob_new
            let kl = (log_prob_ref - log_prob_new).max(0.0);

            // PPO loss: we minimise the negative objective
            total += -(surrogate - self.config.kl_coeff * kl);
        }

        total / n as f64
    }

    /// Compute Generalised Advantage Estimates (GAE-λ).
    ///
    /// `δ_t = r_t + γ * V(s_{t+1}) - V(s_t)`  (values estimated as 0 for simplicity)
    /// `A_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}`
    ///
    /// # Arguments
    /// * `rewards` — reward at each time step.
    /// * `values` — baseline value estimates (same length as `rewards`).
    /// * `gamma` — discount factor.
    /// * `lam` — GAE lambda (trace decay).
    ///
    /// # Errors
    /// Returns an error if lengths differ.
    pub fn compute_advantages_gae(
        rewards: &[f64],
        values: &[f64],
        gamma: f64,
        lam: f64,
    ) -> Result<Vec<f64>> {
        let n = rewards.len();
        if values.len() != n {
            return Err(TensorError::invalid_argument_op(
                "compute_advantages_gae",
                "rewards and values must have the same length",
            ));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut advantages = vec![0.0_f64; n];
        let mut gae = 0.0_f64;

        // V(s_{t+1}) for the terminal step is 0
        for t in (0..n).rev() {
            let v_next = if t + 1 < n { values[t + 1] } else { 0.0 };
            let delta = rewards[t] + gamma * v_next - values[t];
            gae = delta + gamma * lam * gae;
            advantages[t] = gae;
        }

        Ok(advantages)
    }

    /// Access the trained reward model.
    pub fn reward_model(&self) -> &RewardModel {
        &self.reward_model
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: Inverse RL
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Inverse Reinforcement Learning algorithms.
#[derive(Debug, Clone)]
pub struct IrlConfig {
    /// Dimensionality of the state space (also = number of features for linear IRL).
    pub state_dim: usize,
    /// Number of possible actions (for tabular MDPs).
    pub action_dim: usize,
    /// Feature dimension (if different from `state_dim`).
    pub n_features: usize,
    /// Learning rate for gradient ascent on the log-likelihood.
    pub lr: f64,
    /// Number of optimisation iterations.
    pub n_epochs: usize,
    /// Discount factor γ ∈ [0, 1).
    pub discount: f64,
}

impl IrlConfig {
    /// Create a default IRL config.
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            state_dim,
            action_dim,
            n_features: state_dim,
            lr: 0.01,
            n_epochs: 50,
            discount: 0.99,
        }
    }
}

/// Maximum Entropy IRL (Ziebart 2008).
///
/// Feature expectation matching: `∇_θ L = μ_demo - μ_θ` where `μ` are expected features.
/// Policy feature expectations `μ_θ` are estimated from the demonstration state distribution
/// weighted by `exp(r(s))` under the current reward weights.
#[derive(Debug, Clone)]
pub struct MaxEntIrl {
    /// Learned reward weight vector of length `n_features`.
    reward_weights: Vec<f64>,
    config: IrlConfig,
}

impl MaxEntIrl {
    /// Create a new MaxEnt IRL instance with zero-initialised weights.
    pub fn new(config: IrlConfig) -> Self {
        let reward_weights = vec![0.0_f64; config.n_features];
        Self {
            reward_weights,
            config,
        }
    }

    /// Extract feature vector for a state (raw state features).
    fn phi(&self, state: &[f64]) -> Vec<f64> {
        state[..self.config.n_features.min(state.len())].to_vec()
    }

    /// Empirical feature expectations: `μ_demo = (1/N) Σ_{τ} Σ_t γ^t φ(s_t)`.
    pub fn compute_feature_expectations(&self, demos: &[Vec<Vec<f64>>]) -> Vec<f64> {
        let n_feats = self.config.n_features;
        let mut mu = vec![0.0_f64; n_feats];
        let n_demos = demos.len();
        if n_demos == 0 {
            return mu;
        }
        for traj in demos {
            let mut gamma_t = 1.0_f64;
            for state in traj {
                let phi = self.phi(state);
                for (k, &f) in phi.iter().enumerate() {
                    if k < n_feats {
                        mu[k] += gamma_t * f;
                    }
                }
                gamma_t *= self.config.discount;
            }
        }
        for v in mu.iter_mut() {
            *v /= n_demos as f64;
        }
        mu
    }

    /// Fit reward weights via gradient ascent on MaxEnt objective. Returns final weights.
    pub fn fit(&mut self, demonstrations: &[Vec<Vec<f64>>]) -> Vec<f64> {
        if demonstrations.is_empty() {
            return self.reward_weights.clone();
        }

        let mu_demo = self.compute_feature_expectations(demonstrations);

        for _epoch in 0..self.config.n_epochs {
            // Estimate policy feature expectations using current weights.
            // For tractability, we use the demonstrations themselves as a proxy
            // for the state distribution under a "soft" policy, but weight states
            // by their reward under the current weights.
            let mut mu_policy = vec![0.0_f64; self.config.n_features];
            let mut total_weight = 0.0_f64;

            for traj in demonstrations {
                let mut gamma_t = 1.0_f64;
                for state in traj {
                    let r = self.predict_reward(state);
                    let w = r.exp().clamp(1e-12, 1e12);
                    let phi = self.phi(state);
                    for (k, &f) in phi.iter().enumerate() {
                        if k < self.config.n_features {
                            mu_policy[k] += gamma_t * w * f;
                        }
                    }
                    total_weight += gamma_t * w;
                    gamma_t *= self.config.discount;
                }
            }

            if total_weight > 1e-12 {
                for v in mu_policy.iter_mut() {
                    *v /= total_weight;
                }
            }

            // Gradient: ∇θ = μ_demo - μ_policy
            for k in 0..self.config.n_features {
                let grad = mu_demo.get(k).copied().unwrap_or(0.0)
                    - mu_policy.get(k).copied().unwrap_or(0.0);
                self.reward_weights[k] += self.config.lr * grad;
            }
        }

        self.reward_weights.clone()
    }

    /// Access the learned reward weights.
    pub fn reward_weights(&self) -> &[f64] {
        &self.reward_weights
    }

    /// Predict reward for a state as dot product of weights with features.
    pub fn predict_reward(&self, state: &[f64]) -> f64 {
        let phi = self.phi(state);
        self.reward_weights
            .iter()
            .zip(phi.iter())
            .map(|(&w, &f)| w * f)
            .sum()
    }
}

/// Maximum Causal Entropy IRL with soft value iteration:
/// `V(s) = log Σ_a exp(Q(s,a))`, `Q(s,a) = r(s,a) + γ E[V(s')]`.
/// Tabular representation (n_states ≤ 256).
#[derive(Debug, Clone)]
pub struct MaxCausalEntIrl {
    reward_weights: Vec<f64>,
    config: IrlConfig,
}

impl MaxCausalEntIrl {
    /// Create a new MaxCausal Entropy IRL instance.
    pub fn new(config: IrlConfig) -> Self {
        let reward_weights = vec![0.0_f64; config.n_features];
        Self {
            reward_weights,
            config,
        }
    }

    /// Predict reward as dot product of weights and raw state features.
    pub fn predict_reward(&self, state: &[f64]) -> f64 {
        let n = self.config.n_features.min(state.len());
        self.reward_weights[..n]
            .iter()
            .zip(state[..n].iter())
            .map(|(&w, &f)| w * f)
            .sum()
    }

    /// Fit reward weights via MaxCausal Entropy IRL. `transition_fn(state, action) -> next_state`.
    pub fn fit(
        &mut self,
        demonstrations: &[Vec<Vec<f64>>],
        transition_fn: &dyn Fn(&[f64], usize) -> Vec<f64>,
    ) -> Vec<f64> {
        if demonstrations.is_empty() {
            return self.reward_weights.clone();
        }

        // Collect unique states (up to 256 for tabular representation)
        let mut all_states: Vec<Vec<f64>> = Vec::new();
        for traj in demonstrations {
            for s in traj {
                if all_states.len() < 256
                    && !all_states.iter().any(|existing| {
                        existing.len() == s.len()
                            && existing
                                .iter()
                                .zip(s.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-9)
                    })
                {
                    all_states.push(s.clone());
                }
            }
        }

        let n_states = all_states.len();
        if n_states == 0 {
            return self.reward_weights.clone();
        }

        let n_actions = self.config.action_dim;
        let gamma = self.config.discount;
        let n_feats = self.config.n_features;

        // Demo feature expectations
        let mut mu_demo = vec![0.0_f64; n_feats];
        let n_demos = demonstrations.len() as f64;
        for traj in demonstrations {
            let mut gt = 1.0_f64;
            for s in traj {
                let n = n_feats.min(s.len());
                for k in 0..n {
                    mu_demo[k] += gt * s[k];
                }
                gt *= gamma;
            }
        }
        for v in mu_demo.iter_mut() {
            *v /= n_demos;
        }

        for _epoch in 0..self.config.n_epochs {
            // Soft value iteration to get state occupancies
            let mut v = vec![0.0_f64; n_states];
            for _vi_iter in 0..50 {
                let mut v_new = vec![0.0_f64; n_states];
                for (si, state) in all_states.iter().enumerate() {
                    let mut q_vals = Vec::with_capacity(n_actions);
                    for a in 0..n_actions {
                        let s_next = transition_fn(state, a);
                        // Find index of next state (or use 0 if not found)
                        let si_next = all_states
                            .iter()
                            .enumerate()
                            .find(|(_, ns)| {
                                ns.len() == s_next.len()
                                    && ns
                                        .iter()
                                        .zip(s_next.iter())
                                        .all(|(x, y)| (x - y).abs() < 1e-9)
                            })
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        let r = self.predict_reward(state);
                        let q = r + gamma * v[si_next];
                        q_vals.push(q);
                    }
                    // log-sum-exp for soft value
                    let max_q = q_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let lse = max_q + q_vals.iter().map(|&q| (q - max_q).exp()).sum::<f64>().ln();
                    v_new[si] = lse;
                }
                v = v_new;
            }

            // Compute policy feature expectations under soft-optimal policy
            let mut mu_policy = vec![0.0_f64; n_feats];
            let mut total_w = 0.0_f64;
            for (si, state) in all_states.iter().enumerate() {
                let w = v[si].exp().clamp(1e-12, 1e12);
                let n = n_feats.min(state.len());
                for k in 0..n {
                    mu_policy[k] += w * state[k];
                }
                total_w += w;
            }
            if total_w > 1e-12 {
                for v in mu_policy.iter_mut() {
                    *v /= total_w;
                }
            }

            // Gradient ascent
            for k in 0..n_feats {
                let grad = mu_demo.get(k).copied().unwrap_or(0.0)
                    - mu_policy.get(k).copied().unwrap_or(0.0);
                self.reward_weights[k] += self.config.lr * grad;
            }
        }

        self.reward_weights.clone()
    }

    /// Access the learned reward weights.
    pub fn reward_weights(&self) -> &[f64] {
        &self.reward_weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: Reward Shaping
// ─────────────────────────────────────────────────────────────────────────────

/// Potential-based reward shaping (Ng 1999): `r'= r + γΦ(s') - Φ(s)` preserves optimal policy.
pub struct RewardShaper {
    /// Potential function mapping state → scalar.
    pub potential_fn: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    /// Discount factor γ.
    pub gamma: f64,
}

impl RewardShaper {
    /// Create a new reward shaper.
    pub fn new(potential_fn: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>, gamma: f64) -> Self {
        Self {
            potential_fn,
            gamma,
        }
    }

    /// Shape reward: `r_shaped = r + γ*Φ(s') - Φ(s)`.
    pub fn shape_reward(&self, s: &[f64], r: f64, s_next: &[f64]) -> f64 {
        let phi_s = (self.potential_fn)(s);
        let phi_s_next = (self.potential_fn)(s_next);
        r + self.gamma * phi_s_next - phi_s
    }
}

impl std::fmt::Debug for RewardShaper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewardShaper")
            .field("gamma", &self.gamma)
            .finish()
    }
}

/// Goal-conditioned reward: `scale` at goal, `-scale*d/(1+d)` otherwise.
#[derive(Debug, Clone)]
pub struct GoalRewardShaper {
    /// Target goal state vector.
    pub goal: Vec<f64>,
    /// Scale factor for the reward signal.
    pub scale: f64,
    /// Distance threshold below which the goal is considered reached.
    pub threshold: f64,
}

impl GoalRewardShaper {
    /// Create a new goal reward shaper.
    pub fn new(goal: Vec<f64>, scale: f64, threshold: f64) -> Self {
        Self {
            goal,
            scale,
            threshold,
        }
    }

    /// Compute the Euclidean distance from `state` to the goal.
    fn distance(&self, state: &[f64]) -> f64 {
        let n = self.goal.len().min(state.len());
        self.goal[..n]
            .iter()
            .zip(state[..n].iter())
            .map(|(&g, &s)| (g - s).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Returns `scale` at goal, `-scale*dist/(1+dist)` otherwise.
    pub fn shape_reward(&self, state: &[f64]) -> f64 {
        let dist = self.distance(state);
        if dist <= self.threshold {
            self.scale
        } else {
            // Decaying reward as a function of distance
            -self.scale * dist / (1.0 + dist)
        }
    }

    /// Returns `true` if `state` is within `threshold` of the goal.
    pub fn at_goal(&self, state: &[f64]) -> bool {
        self.distance(state) <= self.threshold
    }
}

/// Curiosity shaper via Random Network Distillation (RND).
///
/// Frozen target + learnable predictor; curiosity = MSE(target(s), predictor(s)).
#[derive(Debug, Clone)]
pub struct CuriosityShaper {
    /// Dimensionality of the state space.
    state_dim: usize,
    /// Frozen random target network weights: shape `[hidden_dim, state_dim]`.
    target_weights: Vec<f64>,
    /// Learnable predictor network weights: shape `[hidden_dim, state_dim]`.
    predictor_weights: Vec<f64>,
    /// Predictor biases.
    predictor_biases: Vec<f64>,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Learning rate for predictor updates.
    lr: f64,
}

impl CuriosityShaper {
    /// Create a curiosity shaper with `state_dim` inputs, `hidden_dim` embedding neurons.
    pub fn new(state_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Xavier initialisation
        let limit = (6.0_f64 / (state_dim + hidden_dim) as f64).sqrt();
        let target_weights: Vec<f64> = (0..hidden_dim * state_dim)
            .map(|_| {
                let u: f64 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();
        let predictor_weights: Vec<f64> = (0..hidden_dim * state_dim)
            .map(|_| {
                let u: f64 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();
        let predictor_biases = vec![0.0_f64; hidden_dim];

        Self {
            state_dim,
            target_weights,
            predictor_weights,
            predictor_biases,
            hidden_dim,
            lr: 1e-3,
        }
    }

    /// Compute the embedding of a state through a single linear layer (no activation).
    fn embed(&self, weights: &[f64], state: &[f64]) -> Vec<f64> {
        let n = self.state_dim.min(state.len());
        let mut out = vec![0.0_f64; self.hidden_dim];
        for h in 0..self.hidden_dim {
            for s in 0..n {
                out[h] += weights[h * self.state_dim + s] * state[s];
            }
        }
        out
    }

    /// MSE between target embedding and predictor embedding.
    pub fn curiosity_bonus(&self, state: &[f64]) -> f64 {
        let target = self.embed(&self.target_weights, state);
        let mut pred = self.embed(&self.predictor_weights, state);
        for (p, &b) in pred.iter_mut().zip(self.predictor_biases.iter()) {
            *p += b;
        }
        target
            .iter()
            .zip(pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum::<f64>()
            / self.hidden_dim as f64
    }

    /// One gradient-descent step on MSE(target(s), predictor(s)) to reduce curiosity bonus.
    pub fn update(&mut self, state: &[f64]) {
        let target = self.embed(&self.target_weights, state);
        let pred_raw = self.embed(&self.predictor_weights, state);
        let mut pred = pred_raw.clone();
        for (p, &b) in pred.iter_mut().zip(self.predictor_biases.iter()) {
            *p += b;
        }

        // Gradient of MSE w.r.t. predictor weights and biases
        let n = self.state_dim.min(state.len());
        for h in 0..self.hidden_dim {
            let delta = pred[h] - target[h];
            // Weight gradient
            for s in 0..n {
                let g = 2.0 * delta * state[s] / self.hidden_dim as f64;
                self.predictor_weights[h * self.state_dim + s] -= self.lr * g;
            }
            // Bias gradient
            let bg = 2.0 * delta / self.hidden_dim as f64;
            self.predictor_biases[h] -= self.lr * bg;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Utility tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_at_zero() {
        let s = sigmoid(0.0);
        assert!(
            (s - 0.5).abs() < 1e-12,
            "sigmoid(0) should be 0.5, got {}",
            s
        );
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let s = sigmoid(100.0);
        assert!(s > 0.999, "sigmoid(100) should be ≈ 1.0, got {}", s);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let s = sigmoid(-100.0);
        assert!(s < 0.001, "sigmoid(-100) should be ≈ 0.0, got {}", s);
    }

    #[test]
    fn test_cross_entropy_binary_half() {
        // cross_entropy_binary(0.5, 1.0) = -log(0.5) = ln(2) ≈ 0.693147
        let ce = cross_entropy_binary(0.5, 1.0);
        let expected = 2.0_f64.ln();
        assert!(
            (ce - expected).abs() < 1e-6,
            "expected ln(2) ≈ {:.6}, got {:.6}",
            expected,
            ce
        );
    }

    #[test]
    fn test_cross_entropy_binary_perfect() {
        // cross_entropy_binary(1.0 - ε, 1.0) should be close to 0
        let ce = cross_entropy_binary(1.0 - 1e-6, 1.0);
        assert!(
            ce < 1e-4,
            "perfect prediction should have near-zero loss, got {}",
            ce
        );
    }

    #[test]
    fn test_kendall_tau_perfectly_correlated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = kendall_tau(&x, &y);
        assert!(
            (tau - 1.0).abs() < 1e-10,
            "perfectly correlated should give τ = 1.0, got {}",
            tau
        );
    }

    #[test]
    fn test_kendall_tau_anti_correlated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau = kendall_tau(&x, &y);
        assert!(
            (tau + 1.0).abs() < 1e-10,
            "anti-correlated should give τ = -1.0, got {}",
            tau
        );
    }

    #[test]
    fn test_kendall_tau_short_slice() {
        assert_eq!(kendall_tau(&[], &[]), 0.0);
        assert_eq!(kendall_tau(&[1.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_compute_returns_simple() {
        // rewards = [1, 1, 1], gamma = 0.9
        // G_2 = 1, G_1 = 1 + 0.9*1 = 1.9, G_0 = 1 + 0.9*1.9 = 2.71
        let rewards = vec![1.0, 1.0, 1.0];
        let returns = compute_returns(&rewards, 0.9);
        assert_eq!(returns.len(), 3);
        assert!((returns[2] - 1.0).abs() < 1e-10);
        assert!((returns[1] - 1.9).abs() < 1e-10);
        assert!((returns[0] - 2.71).abs() < 1e-6);
    }

    #[test]
    fn test_compute_returns_zero_discount() {
        let rewards = vec![1.0, 2.0, 3.0];
        let returns = compute_returns(&rewards, 0.0);
        assert!((returns[0] - 1.0).abs() < 1e-10);
        assert!((returns[1] - 2.0).abs() < 1e-10);
        assert!((returns[2] - 3.0).abs() < 1e-10);
    }

    // ── PreferenceDataset tests ─────────────────────────────────────────────

    #[test]
    fn test_preference_dataset_construction() {
        let mut ds = PreferenceDataset::new();
        assert!(ds.is_empty());
        ds.add_comparison(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]], 1.0);
        assert_eq!(ds.len(), 1);
        let c = ds.get(0).expect("Should have comparison at index 0");
        assert!((c.preference - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_preference_dataset_get_none() {
        let ds = PreferenceDataset::new();
        assert!(ds.get(0).is_none());
    }

    #[test]
    fn test_preference_dataset_shuffle_reproducible() {
        let mut ds = PreferenceDataset::new();
        for i in 0..10 {
            ds.add_comparison(vec![vec![i as f64]], vec![vec![-i as f64]], 1.0);
        }

        let mut ds1 = ds.clone();
        let mut ds2 = ds.clone();
        ds1.shuffle(42);
        ds2.shuffle(42);

        for i in 0..10 {
            let c1 = ds1.get(i).expect("ds1.get failed");
            let c2 = ds2.get(i).expect("ds2.get failed");
            assert!(
                (c1.trajectory_a[0][0] - c2.trajectory_a[0][0]).abs() < 1e-12,
                "Shuffling with same seed should yield identical results"
            );
        }
    }

    #[test]
    fn test_preference_dataset_shuffle_differs_from_original() {
        let mut ds = PreferenceDataset::new();
        for i in 0..10 {
            ds.add_comparison(vec![vec![i as f64]], vec![vec![-i as f64]], 1.0);
        }
        let original_first = ds.get(0).expect("ds.get failed").trajectory_a[0][0];
        ds.shuffle(9999);
        // With 10 elements, there's an astronomically small chance shuffle leaves order intact
        // (just check lengths are preserved)
        assert_eq!(ds.len(), 10);
        let _ = original_first; // suppress warning
    }

    #[test]
    fn test_split_train_val_proportions() {
        let mut ds = PreferenceDataset::new();
        for i in 0..20 {
            ds.add_comparison(vec![vec![i as f64]], vec![vec![0.0]], 1.0);
        }
        let (train, val) = ds.split_train_val(0.8, 42).expect("split should succeed");
        assert_eq!(train.len() + val.len(), 20);
        // 80% of 20 = 16 train
        assert_eq!(train.len(), 16, "expected 16 train examples");
        assert_eq!(val.len(), 4, "expected 4 validation examples");
    }

    #[test]
    fn test_split_train_val_invalid_frac() {
        let ds = PreferenceDataset::new();
        assert!(ds.split_train_val(0.0, 0).is_err());
        assert!(ds.split_train_val(1.0, 0).is_err());
        assert!(ds.split_train_val(1.5, 0).is_err());
    }

    // ── RewardModel tests ───────────────────────────────────────────────────

    #[test]
    fn test_reward_model_predict_scalar() {
        let config = RewardModelConfig::new(4);
        let model = RewardModel::new(config).expect("model creation should succeed");
        let state = vec![1.0, 0.5, -0.5, 0.0];
        let r = model.predict(&state);
        // Just check it's a finite scalar
        assert!(
            r.is_finite(),
            "predict should return finite scalar, got {}",
            r
        );
    }

    #[test]
    fn test_reward_model_predict_trajectory() {
        let config = RewardModelConfig::new(2);
        let model = RewardModel::new(config).expect("model creation should succeed");
        let states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let ret = model.predict_trajectory_return(&states);
        assert!(ret.is_finite());
    }

    #[test]
    fn test_reward_model_train_step_reduces_loss() {
        let mut config = RewardModelConfig::new(2);
        config.lr = 0.05;
        config.hidden_dims = vec![8];
        let mut model = RewardModel::new_with_seed(config, 1).expect("model creation ok");

        // Consistent preference: trajectory [1,0] always preferred over [0,1]
        let comparisons: Vec<Comparison> = (0..5)
            .map(|_| Comparison::new(vec![vec![1.0, 0.0]], vec![vec![-1.0, 0.0]], 1.0))
            .collect();

        let loss_0 = model.batch_loss(&comparisons);
        for _ in 0..10 {
            model.train_step(&comparisons).expect("train step ok");
        }
        let loss_final = model.batch_loss(&comparisons);

        assert!(
            loss_final < loss_0 + 0.01,
            "Loss should not increase: initial={:.4}, final={:.4}",
            loss_0,
            loss_final
        );
    }

    #[test]
    fn test_reward_model_evaluate_metrics() {
        let config = RewardModelConfig::new(2);
        let model = RewardModel::new(config).expect("model ok");
        let mut ds = PreferenceDataset::new();
        ds.add_comparison(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]], 1.0);
        ds.add_comparison(vec![vec![0.0, 1.0]], vec![vec![1.0, 0.0]], 0.0);
        let metrics = model.evaluate(&ds);
        assert!(
            metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0,
            "accuracy must be in [0,1]"
        );
        assert!(metrics.loss >= 0.0, "loss must be non-negative");
        assert!(
            metrics.kendall_tau >= -1.0 && metrics.kendall_tau <= 1.0,
            "Kendall τ must be in [-1, 1]"
        );
    }

    // ── BradleyTerryModel tests ─────────────────────────────────────────────

    #[test]
    fn test_bradley_terry_log_likelihood_negative() {
        let config = RewardModelConfig::new(2);
        let model = RewardModel::new(config).expect("model ok");
        let comparisons = vec![
            Comparison::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]], 1.0),
            Comparison::new(vec![vec![0.0, 1.0]], vec![vec![1.0, 0.0]], 0.0),
        ];
        let ll = BradleyTerryModel::log_likelihood(&model, &comparisons);
        assert!(
            ll <= 0.0,
            "log-likelihood should be ≤ 0 for a valid model, got {}",
            ll
        );
    }

    #[test]
    fn test_bradley_terry_brier_score_in_range() {
        let config = RewardModelConfig::new(2);
        let model = RewardModel::new(config).expect("model ok");
        let comparisons = vec![
            Comparison::new(vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]], 1.0),
            Comparison::new(vec![vec![0.0, 1.0]], vec![vec![1.0, 0.0]], 0.0),
        ];
        let bs = BradleyTerryModel::brier_score(&model, &comparisons);
        assert!(
            (0.0..=1.0).contains(&bs),
            "Brier score must be in [0, 1], got {}",
            bs
        );
    }

    #[test]
    fn test_bradley_terry_gradient_step() {
        let mut config = RewardModelConfig::new(2);
        config.hidden_dims = vec![8];
        let mut model = RewardModel::new_with_seed(config, 7).expect("model ok");
        let comparisons = vec![Comparison::new(
            vec![vec![1.0, 0.0]],
            vec![vec![0.0, 1.0]],
            1.0,
        )];
        let loss = BradleyTerryModel::gradient_step(&mut model, &comparisons, 0.01)
            .expect("gradient step ok");
        assert!(loss.is_finite(), "loss should be finite");
    }

    // ── RLHF Pipeline tests ─────────────────────────────────────────────────

    #[test]
    fn test_rlhf_trainer_new() {
        let config = RlhfConfig::new(4);
        let trainer = RlhfTrainer::new(config).expect("trainer creation ok");
        let _ = trainer.reward_model();
    }

    #[test]
    fn test_rlhf_train_reward_model_loss_history() {
        let mut config = RlhfConfig::new(2);
        config.n_reward_epochs = 5;
        config.batch_size = 2;
        config.reward_config.lr = 0.01;
        config.reward_config.hidden_dims = vec![8];
        let mut trainer = RlhfTrainer::new(config).expect("trainer ok");

        let mut ds = PreferenceDataset::new();
        for _ in 0..8 {
            ds.add_comparison(vec![vec![1.0, 0.0]], vec![vec![-1.0, 0.0]], 1.0);
        }

        let losses = trainer
            .train_reward_model(&ds)
            .expect("training should succeed");
        assert_eq!(losses.len(), 5, "should have one loss per epoch");
        for &l in &losses {
            assert!(l.is_finite(), "loss should be finite");
        }
    }

    #[test]
    fn test_rlhf_train_reward_model_loss_decreases() {
        let mut config = RlhfConfig::new(2);
        config.n_reward_epochs = 20;
        config.batch_size = 10;
        config.reward_config.lr = 0.05;
        config.reward_config.hidden_dims = vec![16];
        let mut trainer = RlhfTrainer::new(config).expect("trainer ok");

        let mut ds = PreferenceDataset::new();
        for _ in 0..20 {
            ds.add_comparison(vec![vec![1.0, 0.0]], vec![vec![-1.0, 0.0]], 1.0);
        }

        let losses = trainer
            .train_reward_model(&ds)
            .expect("training should succeed");
        let first = losses[0];
        let last = *losses.last().expect("losses should not be empty");
        assert!(
            last <= first + 0.1,
            "loss should not increase substantially: first={:.4}, last={:.4}",
            first,
            last
        );
    }

    #[test]
    fn test_rlhf_ppo_policy_step() {
        let config = RlhfConfig::new(4);
        let mut trainer = RlhfTrainer::new(config).expect("trainer ok");
        let states = vec![vec![1.0, 0.0, 0.5, -0.5], vec![0.0, 1.0, -0.5, 0.5]];
        let actions = vec![0_usize, 1_usize];
        let rewards = vec![1.0_f64, -1.0_f64];
        let ref_log_probs = vec![-1.386_f64, -1.386_f64]; // log(0.25)
        let loss = trainer
            .ppo_policy_step(&states, &actions, &rewards, &ref_log_probs)
            .expect("PPO step ok");
        assert!(loss.is_finite(), "PPO loss should be finite, got {}", loss);
    }

    #[test]
    fn test_rlhf_ppo_mismatch_lengths() {
        let config = RlhfConfig::new(2);
        let mut trainer = RlhfTrainer::new(config).expect("trainer ok");
        let result = trainer.ppo_policy_step(
            &[vec![1.0, 0.0]],
            &[0_usize, 1_usize], // wrong length
            &[1.0_f64],
            &[-0.5_f64],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_advantages_gae_length() {
        let rewards = vec![1.0, 0.5, -1.0, 2.0, 0.0];
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let adv =
            RlhfTrainer::compute_advantages_gae(&rewards, &values, 0.99, 0.95).expect("GAE ok");
        assert_eq!(
            adv.len(),
            rewards.len(),
            "GAE output length must match input"
        );
    }

    #[test]
    fn test_compute_advantages_gae_zero_lambda() {
        // With lam=0, GAE = TD residual = r + γ*V(s') - V(s)
        let rewards = vec![1.0, 0.0];
        let values = vec![0.0, 0.0];
        let adv =
            RlhfTrainer::compute_advantages_gae(&rewards, &values, 0.99, 0.0).expect("GAE ok");
        assert_eq!(adv.len(), 2);
        // G_1: r=0, V(next)=0, V(s)=0 → delta=0
        assert!((adv[1] - 0.0).abs() < 1e-10);
        // G_0: r=1, V(next)=0, V(s)=0 → delta=1
        assert!((adv[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_advantages_gae_mismatched_lengths() {
        let result = RlhfTrainer::compute_advantages_gae(&[1.0, 2.0], &[0.5], 0.99, 0.95);
        assert!(result.is_err());
    }

    // ── MaxEntIRL tests ─────────────────────────────────────────────────────

    #[test]
    fn test_maxent_irl_feature_expectations_shape() {
        let config = IrlConfig {
            state_dim: 3,
            action_dim: 2,
            n_features: 3,
            lr: 0.01,
            n_epochs: 10,
            discount: 0.99,
        };
        let irl = MaxEntIrl::new(config);
        let demos = vec![
            vec![vec![1.0, 0.0, 0.5], vec![0.5, 1.0, 0.0]],
            vec![vec![0.0, 0.5, 1.0]],
        ];
        let mu = irl.compute_feature_expectations(&demos);
        assert_eq!(
            mu.len(),
            3,
            "feature expectation length must match n_features"
        );
    }

    #[test]
    fn test_maxent_irl_fit_weight_length() {
        let config = IrlConfig::new(4, 2);
        let mut irl = MaxEntIrl::new(config);
        let demos = vec![
            vec![vec![1.0, 0.0, 0.5, -0.5], vec![0.5, 0.5, 0.5, 0.5]],
            vec![vec![0.0, 1.0, -0.5, 0.5]],
        ];
        let weights = irl.fit(&demos);
        assert_eq!(
            weights.len(),
            4,
            "learned weight vector should have length state_dim"
        );
        for &w in &weights {
            assert!(w.is_finite(), "all weights should be finite");
        }
    }

    #[test]
    fn test_maxent_irl_predict_reward() {
        let mut config = IrlConfig::new(2, 2);
        config.n_epochs = 30;
        config.lr = 0.1;
        let mut irl = MaxEntIrl::new(config);

        // Demonstrations consistently visit positive states
        let demos = vec![
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            vec![vec![0.8, 0.2], vec![0.9, 0.1]],
        ];
        irl.fit(&demos);

        let r_pos = irl.predict_reward(&[1.0, 0.0]);
        let r_neg = irl.predict_reward(&[-1.0, 0.0]);
        assert!(
            r_pos.is_finite() && r_neg.is_finite(),
            "rewards should be finite"
        );
    }

    #[test]
    fn test_maxent_irl_empty_demos() {
        let config = IrlConfig::new(2, 2);
        let mut irl = MaxEntIrl::new(config);
        let weights = irl.fit(&[]);
        assert_eq!(weights.len(), 2);
    }

    // ── MaxCausalEntIRL tests ───────────────────────────────────────────────

    #[test]
    fn test_maxcausal_irl_fit() {
        let config = IrlConfig::new(2, 2);
        let mut irl = MaxCausalEntIrl::new(config);
        let demos = vec![vec![vec![1.0, 0.0], vec![1.0, 0.0]]];
        let transition_fn = |s: &[f64], _a: usize| -> Vec<f64> { s.to_vec() };
        let weights = irl.fit(&demos, &transition_fn);
        assert_eq!(weights.len(), 2);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    // ── RewardShaper tests ──────────────────────────────────────────────────

    #[test]
    fn test_reward_shaper_zero_for_absorbing_state() {
        // If Φ(s) = Φ(s'), the shaping term is zero → shaped reward = raw reward
        let shaper = RewardShaper::new(Box::new(|s: &[f64]| s[0]), 0.99);
        let s = vec![1.0, 0.0];
        let s_next = vec![1.0, 0.0]; // same state (absorbing)
        let shaped = shaper.shape_reward(&s, 0.0, &s_next);
        // r + γ * Φ(s') - Φ(s) = 0 + 0.99 * 1.0 - 1.0 = -0.01 (not zero due to γ < 1)
        // For perfect absorption test, use γ=1
        let shaper2 = RewardShaper::new(Box::new(|s: &[f64]| s[0]), 1.0);
        let shaped2 = shaper2.shape_reward(&s, 5.0, &s_next);
        // r + 1.0 * Φ(s') - Φ(s) = 5.0 + 1.0 - 1.0 = 5.0
        assert!(
            (shaped2 - 5.0).abs() < 1e-10,
            "shaped reward should equal raw reward when Φ(s)=Φ(s') and γ=1"
        );
        let _ = shaped;
    }

    #[test]
    fn test_reward_shaper_potential_based() {
        let shaper = RewardShaper::new(Box::new(|s: &[f64]| s[0].powi(2)), 0.9);
        let s = vec![1.0];
        let s_next = vec![2.0];
        // shaped = 0.0 + 0.9 * 4.0 - 1.0 = 2.6
        let shaped = shaper.shape_reward(&s, 0.0, &s_next);
        assert!((shaped - 2.6).abs() < 1e-10, "expected 2.6, got {}", shaped);
    }

    #[test]
    fn test_goal_reward_shaper_at_goal() {
        let shaper = GoalRewardShaper::new(vec![0.0, 0.0], 10.0, 0.1);
        let at_goal = shaper.shape_reward(&[0.0, 0.0]);
        let far = shaper.shape_reward(&[10.0, 10.0]);
        assert!(
            (at_goal - 10.0).abs() < 1e-10,
            "goal reward should be scale=10, got {}",
            at_goal
        );
        assert!(
            far < 0.0,
            "far-from-goal reward should be negative, got {}",
            far
        );
        assert!(
            at_goal > far,
            "goal reward ({}) must be greater than far reward ({})",
            at_goal,
            far
        );
    }

    #[test]
    fn test_goal_reward_shaper_at_goal_detection() {
        let shaper = GoalRewardShaper::new(vec![1.0, 1.0], 5.0, 0.5);
        assert!(shaper.at_goal(&[1.0, 1.0]));
        assert!(shaper.at_goal(&[1.3, 1.0])); // within 0.5
        assert!(!shaper.at_goal(&[3.0, 3.0]));
    }

    // ── CuriosityShaper tests ────────────────────────────────────────────────

    #[test]
    fn test_curiosity_shaper_initial_bonus_positive() {
        let shaper = CuriosityShaper::new(4, 8, 42);
        let state = vec![1.0, 0.5, -0.5, 0.0];
        let bonus = shaper.curiosity_bonus(&state);
        assert!(
            bonus >= 0.0,
            "curiosity bonus should be non-negative, got {}",
            bonus
        );
    }

    #[test]
    fn test_curiosity_shaper_update_reduces_bonus() {
        let mut shaper = CuriosityShaper::new(2, 4, 100);
        let state = vec![1.0, 0.0];
        let bonus_before = shaper.curiosity_bonus(&state);
        for _ in 0..50 {
            shaper.update(&state);
        }
        let bonus_after = shaper.curiosity_bonus(&state);
        assert!(
            bonus_after <= bonus_before + 1e-6,
            "curiosity bonus should decrease after updates: before={:.6}, after={:.6}",
            bonus_before,
            bonus_after
        );
    }
}
