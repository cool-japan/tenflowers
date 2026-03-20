//! Advanced world model components: TD-MPC2, generative world models, evaluation metrics.

use super::*;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// 1. TD-MPC2 — Temporal Difference Model Predictive Control 2
// ---------------------------------------------------------------------------

/// Configuration for TD-MPC2 world model.
#[derive(Debug, Clone)]
pub struct TdMpc2Config {
    /// Observation/state dimension.
    pub obs_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Latent state dimension.
    pub latent_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Planning horizon.
    pub horizon: usize,
    /// Discount factor γ.
    pub gamma: f32,
    /// Target network momentum τ.
    pub tau: f32,
}

impl TdMpc2Config {
    /// Create a new TD-MPC2 configuration.
    pub fn new(
        obs_dim: usize,
        action_dim: usize,
        latent_dim: usize,
        hidden_dim: usize,
        horizon: usize,
    ) -> Self {
        Self {
            obs_dim,
            action_dim,
            latent_dim,
            hidden_dim,
            horizon,
            gamma: 0.99,
            tau: 0.005,
        }
    }
}

/// TD-MPC2 encoder: maps observations to latent states.
///
/// Reference: Hansen et al. (2024) "TD-MPC2: Scalable, Robust World Models for Continuous Control".
pub struct TdMpc2Encoder {
    /// Weight matrix (hidden × obs_dim).
    pub w1: Vec<f32>,
    /// Weight matrix (latent × hidden).
    pub w2: Vec<f32>,
    /// Input dimension.
    pub obs_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Latent dimension.
    pub latent_dim: usize,
}

impl TdMpc2Encoder {
    /// Create a new encoder with Xavier initialization.
    pub fn new(obs_dim: usize, hidden_dim: usize, latent_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(obs_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, latent_dim, &mut rng);
        Self {
            w1,
            w2,
            obs_dim,
            hidden_dim,
            latent_dim,
        }
    }

    /// Encode observation to latent state.
    pub fn encode(&self, obs: &[f32]) -> Vec<f32> {
        let h = relu(&matvec(&self.w1, obs, self.hidden_dim, self.obs_dim));
        matvec(&self.w2, &h, self.latent_dim, self.hidden_dim)
    }
}

/// TD-MPC2 dynamics model: predicts next latent state and reward.
pub struct TdMpc2Dynamics {
    /// Weight matrix (hidden × (latent + action)).
    pub w1: Vec<f32>,
    /// Weight matrix (latent × hidden).
    pub w2: Vec<f32>,
    /// Reward head (1 × hidden).
    pub w_reward: Vec<f32>,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
}

impl TdMpc2Dynamics {
    /// Create a new dynamics model.
    pub fn new(latent_dim: usize, action_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let input_dim = latent_dim + action_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, latent_dim, &mut rng);
        let w_reward = xavier_init_vec(hidden_dim, 1, &mut rng);
        Self {
            w1,
            w2,
            w_reward,
            latent_dim,
            action_dim,
            hidden_dim,
        }
    }

    /// Predict (next_latent, reward) from current latent and action.
    pub fn forward(&self, z: &[f32], a: &[f32]) -> (Vec<f32>, f32) {
        let mut inp = z.to_vec();
        inp.extend_from_slice(a);
        let h = relu(&matvec(&self.w1, &inp, self.hidden_dim, z.len() + a.len()));
        let z_next = matvec(&self.w2, &h, self.latent_dim, self.hidden_dim);
        let reward = matvec(&self.w_reward, &h, 1, self.hidden_dim)[0];
        (z_next, reward)
    }
}

/// TD-MPC2 policy network: latent state → action.
pub struct TdMpc2Policy {
    /// Weight matrix (hidden × latent).
    pub w1: Vec<f32>,
    /// Weight matrix (action × hidden).
    pub w2: Vec<f32>,
    /// Latent dimension.
    pub latent_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
}

impl TdMpc2Policy {
    /// Create a new policy network.
    pub fn new(latent_dim: usize, hidden_dim: usize, action_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(latent_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, action_dim, &mut rng);
        Self {
            w1,
            w2,
            latent_dim,
            hidden_dim,
            action_dim,
        }
    }

    /// Compute action from latent state (tanh-bounded output ∈ (−1, 1)).
    pub fn act(&self, z: &[f32]) -> Vec<f32> {
        let h = relu(&matvec(&self.w1, z, self.hidden_dim, self.latent_dim));
        let out = matvec(&self.w2, &h, self.action_dim, self.hidden_dim);
        tanh_vec(&out)
    }
}

/// Complete TD-MPC2 model combining encoder, dynamics, and policy.
pub struct TdMpc2Model {
    /// Encoder: obs → latent.
    pub encoder: TdMpc2Encoder,
    /// Dynamics: (latent, action) → (next_latent, reward).
    pub dynamics: TdMpc2Dynamics,
    /// Policy: latent → action.
    pub policy: TdMpc2Policy,
    /// Configuration.
    pub config: TdMpc2Config,
}

impl TdMpc2Model {
    /// Create a new TD-MPC2 model.
    pub fn new(config: TdMpc2Config, seed: u64) -> Self {
        let encoder = TdMpc2Encoder::new(
            config.obs_dim,
            config.hidden_dim,
            config.latent_dim,
            seed,
        );
        let dynamics = TdMpc2Dynamics::new(
            config.latent_dim,
            config.action_dim,
            config.hidden_dim,
            seed.wrapping_add(1),
        );
        let policy = TdMpc2Policy::new(
            config.latent_dim,
            config.hidden_dim,
            config.action_dim,
            seed.wrapping_add(2),
        );
        Self {
            encoder,
            dynamics,
            policy,
            config,
        }
    }

    /// Encode observation to latent state.
    pub fn encode(&self, obs: &[f32]) -> Vec<f32> {
        self.encoder.encode(obs)
    }

    /// Step dynamics forward: (latent, action) → (next_latent, reward).
    pub fn step(&self, z: &[f32], a: &[f32]) -> (Vec<f32>, f32) {
        self.dynamics.forward(z, a)
    }

    /// Get policy action from latent state.
    pub fn act(&self, z: &[f32]) -> Vec<f32> {
        self.policy.act(z)
    }
}

/// TD-MPC2 MPPI planner using learned world model rollouts.
///
/// Uses Model Predictive Path Integral control to plan optimal action sequences
/// by sampling perturbations and weighting by exponentiated cumulative rewards.
pub struct TdMpc2Planner {
    /// Number of MPPI samples.
    pub n_samples: usize,
    /// Noise standard deviation for action perturbations.
    pub noise_std: f32,
    /// Temperature parameter for softmax weighting.
    pub temperature: f32,
}

impl TdMpc2Planner {
    /// Create a new TD-MPC2 MPPI planner.
    pub fn new(n_samples: usize, noise_std: f32, temperature: f32) -> Self {
        Self {
            n_samples,
            noise_std,
            temperature,
        }
    }

    /// Plan action sequence via MPPI with the learned world model.
    ///
    /// Returns optimized sequence of `horizon` actions.
    pub fn plan(
        &self,
        model: &TdMpc2Model,
        obs: &[f32],
        horizon: usize,
        rng: &mut StdRng,
    ) -> Vec<Vec<f32>> {
        let z0 = model.encode(obs);
        let action_dim = model.config.action_dim;

        // Nominal trajectory from policy
        let mut nominal: Vec<Vec<f32>> = (0..horizon)
            .scan(z0.clone(), |z, _| {
                let a = model.act(z);
                let (z_next, _) = model.step(z, &a);
                *z = z_next;
                Some(a)
            })
            .collect();

        // Sample perturbations and compute trajectory returns
        let mut scores = Vec::with_capacity(self.n_samples);
        let mut perturbations: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let pert: Vec<Vec<f32>> = (0..horizon)
                .map(|_| {
                    (0..action_dim)
                        .map(|_| {
                            let u1: f32 = rng.random::<f32>().max(1e-10);
                            let u2: f32 = rng.random::<f32>();
                            let noise =
                                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                            noise * self.noise_std
                        })
                        .collect()
                })
                .collect();

            // Roll out perturbed trajectory
            let mut z = z0.clone();
            let mut total_return = 0.0_f32;
            let mut discount = 1.0_f32;
            for (t, p) in pert.iter().enumerate() {
                let a: Vec<f32> = nominal[t]
                    .iter()
                    .zip(p.iter())
                    .map(|(n, pi)| (n + pi).clamp(-1.0, 1.0))
                    .collect();
                let (z_next, reward) = model.step(&z, &a);
                total_return += discount * reward;
                discount *= model.config.gamma;
                z = z_next;
            }
            scores.push(total_return);
            perturbations.push(pert);
        }

        // Softmax weighting
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores
            .iter()
            .map(|&s| ((s - max_score) / self.temperature.max(1e-8)).exp())
            .collect();
        let sum_exp: f32 = exps.iter().sum::<f32>().max(1e-8);
        let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        // Update nominal trajectory
        for t in 0..horizon {
            for d in 0..action_dim {
                let update: f32 = weights
                    .iter()
                    .zip(perturbations.iter())
                    .map(|(&w, p)| w * p[t][d])
                    .sum();
                nominal[t][d] = (nominal[t][d] + update).clamp(-1.0, 1.0);
            }
        }

        nominal
    }
}

// ---------------------------------------------------------------------------
// 2. Generative World Models (token-based)
// ---------------------------------------------------------------------------

/// VQ-VAE tokenizer for observations: maps continuous observations to discrete tokens.
///
/// Used in IRIS-style generative world models where observations are first
/// discretized into tokens, then modeled autoregressively.
pub struct GwmTokenizer {
    /// Codebook: n_codes × code_dim.
    pub codebook: Vec<Vec<f32>>,
    /// Number of discrete tokens.
    pub n_codes: usize,
    /// Dimension of each code vector.
    pub code_dim: usize,
    /// Patch size (observations are split into patches of this size).
    pub patch_size: usize,
}

impl GwmTokenizer {
    /// Create a new GWM tokenizer with random codebook.
    pub fn new(n_codes: usize, code_dim: usize, patch_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let codebook: Vec<Vec<f32>> = (0..n_codes)
            .map(|_| (0..code_dim).map(|_| rng.random::<f32>() * 0.1 - 0.05).collect())
            .collect();
        Self {
            codebook,
            n_codes,
            code_dim,
            patch_size,
        }
    }

    /// Tokenize an observation: split into patches, find nearest codebook entry.
    pub fn tokenize(&self, obs: &[f32]) -> Vec<usize> {
        let n_patches = obs.len() / self.patch_size;
        (0..n_patches)
            .map(|i| {
                let patch = &obs[i * self.patch_size..(i + 1) * self.patch_size];
                // Project patch to code_dim (mean pooling if needed)
                let code_len = patch.len().min(self.code_dim);
                let proj: Vec<f32> = (0..self.code_dim)
                    .map(|j| patch[j % code_len])
                    .collect();
                // Find nearest codebook entry (L2 distance)
                self.codebook
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da: f32 = a.iter().zip(proj.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                        let db: f32 = b.iter().zip(proj.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Decode token sequence back to continuous observations.
    pub fn detokenize(&self, tokens: &[usize]) -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&t| {
                let code = &self.codebook[t.min(self.n_codes - 1)];
                code.iter().take(self.patch_size).cloned().chain(
                    std::iter::repeat(0.0_f32).take(self.patch_size.saturating_sub(code.len())),
                )
            })
            .collect()
    }
}

/// GPT-style transformer for next-token prediction in world model imagination.
///
/// Models the future state sequence as a language model over discrete tokens,
/// enabling multi-step imagination via autoregressive sampling.
pub struct GwmTransformer {
    /// Embedding table: vocab_size × d_model.
    pub embedding: Vec<f32>,
    /// Single-layer attention weights: (d_model × d_model) × 3 (Q, K, V).
    pub attn_weights: Vec<f32>,
    /// Output projection: vocab_size × d_model.
    pub output_proj: Vec<f32>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl GwmTransformer {
    /// Create a new GWM transformer.
    pub fn new(vocab_size: usize, d_model: usize, max_seq_len: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale = (1.0_f32 / d_model as f32).sqrt();
        let embedding: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.random::<f32>() * scale - scale / 2.0)
            .collect();
        let attn_weights: Vec<f32> = (0..3 * d_model * d_model)
            .map(|_| rng.random::<f32>() * scale - scale / 2.0)
            .collect();
        let output_proj: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.random::<f32>() * scale - scale / 2.0)
            .collect();
        Self {
            embedding,
            attn_weights,
            output_proj,
            vocab_size,
            d_model,
            max_seq_len,
        }
    }

    /// Get token embedding vector.
    fn token_embed(&self, token: usize) -> Vec<f32> {
        let t = token.min(self.vocab_size - 1);
        self.embedding[t * self.d_model..(t + 1) * self.d_model].to_vec()
    }

    /// Simple single-head self-attention (causal, scaled dot-product).
    fn attend(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let d = self.d_model;
        let wq = &self.attn_weights[0..d * d];
        let wk = &self.attn_weights[d * d..2 * d * d];
        let wv = &self.attn_weights[2 * d * d..3 * d * d];

        let queries: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| matvec(wq, e, d, d))
            .collect();
        let keys: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| matvec(wk, e, d, d))
            .collect();
        let values: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| matvec(wv, e, d, d))
            .collect();

        let scale = (d as f32).sqrt();
        (0..n)
            .map(|i| {
                // Causal: only attend to positions <= i
                let raw_scores: Vec<f32> = (0..=i)
                    .map(|j| dot(&queries[i], &keys[j]) / scale)
                    .collect();
                let attn = softmax(&raw_scores);
                let mut out = vec![0.0_f32; d];
                for (j, &w) in attn.iter().enumerate() {
                    for (o, v) in out.iter_mut().zip(values[j].iter()) {
                        *o += w * v;
                    }
                }
                add_vecs(&embeddings[i], &out) // residual connection
            })
            .collect()
    }

    /// Forward pass: token sequence → logits over vocabulary.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let truncated: Vec<usize> = tokens
            .iter()
            .take(self.max_seq_len)
            .copied()
            .collect();
        let embeddings: Vec<Vec<f32>> = truncated.iter().map(|&t| self.token_embed(t)).collect();
        let attended = self.attend(&embeddings);
        attended
            .iter()
            .map(|h| {
                
                matvec(&self.output_proj, h, self.vocab_size, self.d_model)
            })
            .collect()
    }

    /// Sample next token given a context sequence (greedy or temperature sampling).
    pub fn sample_next_token(&self, context: &[usize], temperature: f32, rng: &mut StdRng) -> usize {
        if context.is_empty() {
            return rng.random_range(0..self.vocab_size);
        }
        let all_logits = self.forward(context);
        let last_logits = &all_logits[all_logits.len() - 1];
        if temperature <= 0.0 {
            // Greedy
            last_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            let scaled: Vec<f32> = last_logits.iter().map(|&l| l / temperature).collect();
            let probs = softmax(&scaled);
            let r: f32 = rng.random();
            let mut cumsum = 0.0_f32;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    return i;
                }
            }
            probs.len() - 1
        }
    }

    /// Imagine a future token sequence of `horizon` steps.
    pub fn imagine(&self, context: &[usize], horizon: usize, temperature: f32, rng: &mut StdRng) -> Vec<usize> {
        let mut sequence = context.to_vec();
        for _ in 0..horizon {
            let next = self.sample_next_token(&sequence, temperature, rng);
            sequence.push(next);
        }
        sequence[context.len()..].to_vec()
    }
}

/// Evaluation metrics for world models.
#[derive(Debug, Clone, Default)]
pub struct WorldModelEvaluation {
    /// Open-loop prediction errors (MSE of predicted vs true latent states).
    pub open_loop_errors: Vec<f32>,
    /// Imagination horizon reached before divergence.
    pub imagination_horizons: Vec<usize>,
    /// Compounding prediction errors over time steps.
    pub compounding_errors: Vec<Vec<f32>>,
    /// Planning performance (cumulative reward of planned trajectory).
    pub planning_returns: Vec<f32>,
}

impl WorldModelEvaluation {
    /// Create empty evaluation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record open-loop prediction error (MSE between predicted and true latent).
    pub fn record_open_loop_error(&mut self, predicted: &[f32], true_latent: &[f32]) {
        let n = predicted.len().min(true_latent.len());
        if n == 0 {
            return;
        }
        let mse: f32 = predicted
            .iter()
            .zip(true_latent.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / n as f32;
        self.open_loop_errors.push(mse);
    }

    /// Record imagination horizon (how many steps before error exceeds threshold).
    pub fn record_imagination_horizon(
        &mut self,
        errors_per_step: &[f32],
        error_threshold: f32,
    ) -> usize {
        let horizon = errors_per_step
            .iter()
            .position(|&e| e > error_threshold)
            .unwrap_or(errors_per_step.len());
        self.imagination_horizons.push(horizon);
        horizon
    }

    /// Record compounding errors over a trajectory.
    pub fn record_compounding_errors(&mut self, errors: Vec<f32>) {
        self.compounding_errors.push(errors);
    }

    /// Record planning return (cumulative reward of a planned trajectory).
    pub fn record_planning_return(&mut self, cumulative_reward: f32) {
        self.planning_returns.push(cumulative_reward);
    }

    /// Mean open-loop prediction error.
    pub fn mean_open_loop_error(&self) -> f32 {
        if self.open_loop_errors.is_empty() {
            return 0.0;
        }
        self.open_loop_errors.iter().sum::<f32>() / self.open_loop_errors.len() as f32
    }

    /// Mean imagination horizon.
    pub fn mean_imagination_horizon(&self) -> f32 {
        if self.imagination_horizons.is_empty() {
            return 0.0;
        }
        self.imagination_horizons.iter().sum::<usize>() as f32
            / self.imagination_horizons.len() as f32
    }

    /// Mean planning return.
    pub fn mean_planning_return(&self) -> f32 {
        if self.planning_returns.is_empty() {
            return 0.0;
        }
        self.planning_returns.iter().sum::<f32>() / self.planning_returns.len() as f32
    }

    /// Compute compounding error growth: ratio of error at step T vs step 1.
    ///
    /// Returns (mean_error_at_step_1, mean_error_at_last_step, growth_ratio).
    pub fn compounding_error_growth(&self) -> (f32, f32, f32) {
        if self.compounding_errors.is_empty() {
            return (0.0, 0.0, 1.0);
        }
        let first_step: f32 = self
            .compounding_errors
            .iter()
            .filter_map(|e| e.first().copied())
            .sum::<f32>()
            / self.compounding_errors.len() as f32;
        let last_step: f32 = self
            .compounding_errors
            .iter()
            .filter_map(|e| e.last().copied())
            .sum::<f32>()
            / self.compounding_errors.len() as f32;
        let ratio = if first_step > 1e-8 {
            last_step / first_step
        } else {
            1.0
        };
        (first_step, last_step, ratio)
    }
}
