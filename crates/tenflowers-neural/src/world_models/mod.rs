//! World Model Architectures for Model-Based RL and Prediction
//!
//! Implements: DreamerV3 (RSSM), Transformer World Model (TWM), IRIS (discrete tokenization),
//! MuZero (planning with learned model), Predictive Coding, EfficientZero, TDM, ObsDecoder,
//! WmRewardPredictor, and LatentPlanner (CEM/MPPI).

pub mod extensions;
pub use extensions::*;
pub mod advanced;
pub use advanced::*;
#[cfg(test)]
pub mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Xavier-uniform initializer for a weight vector of shape [fan_out x fan_in].
fn xavier_init_vec(fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f32> {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.random::<f32>() * 2.0 * limit - limit)
        .collect()
}

/// Simple matrix-vector multiply: W (rows x cols) times x (cols) → y (rows).
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut y = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += w[r * cols + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        exps
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

fn normal_sample(rng: &mut StdRng) -> f32 {
    // Box-Muller
    let u1: f32 = rng.random::<f32>().max(1e-10);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// ---------------------------------------------------------------------------
// 1. DreamerV3 — Recurrent State-Space Model (RSSM)
// ---------------------------------------------------------------------------

/// Configuration for the DreamerV3 RSSM.
#[derive(Debug, Clone)]
pub struct RsssmConfig {
    /// Dimension of the deterministic GRU hidden state.
    pub deter_dim: usize,
    /// Total stochastic latent dimension = stoch_dim * n_classes.
    pub stoch_dim: usize,
    /// Number of discrete categories per stochastic variable.
    pub n_classes: usize,
    /// Embedding dimension used for observation encoding.
    pub embed_dim: usize,
}

impl RsssmConfig {
    pub fn new(deter_dim: usize, stoch_dim: usize, n_classes: usize, embed_dim: usize) -> Self {
        Self {
            deter_dim,
            stoch_dim,
            n_classes,
            embed_dim,
        }
    }
}

/// GRU-based recurrent model (deterministic part of RSSM).
/// Computes h_t = GRU(h_{t-1}, cat(z_{t-1}, a_{t-1})).
pub struct RecurrentModel {
    /// Packed GRU weight matrix: concatenation of [W_z | W_r | W_h] each of shape [hidden_dim x input_dim].
    pub gru_weights: Vec<f32>,
    pub hidden_dim: usize,
    input_dim: usize,
}

impl RecurrentModel {
    /// Create a new RecurrentModel.
    ///
    /// `input_dim` is the dimension of the input (stoch_dim + action_dim).
    pub fn new(hidden_dim: usize, input_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let total = hidden_dim * (input_dim + hidden_dim) * 3;
        let gru_weights = (0..total)
            .map(|_| rng.random::<f32>() * 0.02 - 0.01)
            .collect();
        Self {
            gru_weights,
            hidden_dim,
            input_dim,
        }
    }

    /// Single GRU step: h_t = GRU(h, x).
    pub fn step(&self, h: &[f32], x: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let id = self.input_dim;
        // Each gate uses W_x (hd x id) and W_h (hd x hd) packed consecutively
        let block = hd * (id + hd);
        let wz = &self.gru_weights[0..block];
        let wr = &self.gru_weights[block..2 * block];
        let wn = &self.gru_weights[2 * block..3 * block];

        let z_gate_x: Vec<f32> = matvec(&wz[..hd * id], x, hd, id);
        let z_gate_h: Vec<f32> = matvec(&wz[hd * id..], h, hd, hd);
        let z_gate: Vec<f32> = z_gate_x
            .iter()
            .zip(z_gate_h.iter())
            .map(|(a, b)| sigmoid(a + b))
            .collect();

        let r_gate_x: Vec<f32> = matvec(&wr[..hd * id], x, hd, id);
        let r_gate_h: Vec<f32> = matvec(&wr[hd * id..], h, hd, hd);
        let r_gate: Vec<f32> = r_gate_x
            .iter()
            .zip(r_gate_h.iter())
            .map(|(a, b)| sigmoid(a + b))
            .collect();

        let rh: Vec<f32> = r_gate.iter().zip(h.iter()).map(|(r, hi)| r * hi).collect();
        let n_gate_x: Vec<f32> = matvec(&wn[..hd * id], x, hd, id);
        let n_gate_h: Vec<f32> = matvec(&wn[hd * id..], &rh, hd, hd);
        let n_gate: Vec<f32> = n_gate_x
            .iter()
            .zip(n_gate_h.iter())
            .map(|(a, b)| (a + b).tanh())
            .collect();

        z_gate
            .iter()
            .zip(h.iter())
            .zip(n_gate.iter())
            .map(|((z, hi), n)| (1.0 - z) * n + z * hi)
            .collect()
    }
}

/// Posterior model: encodes observation embedding + deter hidden → stoch logits.
pub struct RepresentationModel {
    pub weights: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl RepresentationModel {
    pub fn new(deter_dim: usize, embed_dim: usize, stoch_total: usize, seed: u64) -> Self {
        let input_dim = deter_dim + embed_dim;
        let hidden_dim = 256;
        let output_dim = stoch_total;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, output_dim, &mut rng);
        let mut weights = w1;
        weights.extend_from_slice(&w2);
        Self {
            weights,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Compute stochastic logits from (deter, embed).
    pub fn forward(&self, deter: &[f32], embed: &[f32]) -> Vec<f32> {
        let mut inp = deter.to_vec();
        inp.extend_from_slice(embed);
        let sz1 = self.input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.weights[..sz1],
            &inp,
            self.hidden_dim,
            self.input_dim,
        ));
        matvec(&self.weights[sz1..], &h, self.output_dim, self.hidden_dim)
    }
}

/// Prior (transition) model: deter hidden → stoch logits (no observation).
pub struct TransitionModel {
    pub weights: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl TransitionModel {
    pub fn new(deter_dim: usize, stoch_total: usize, seed: u64) -> Self {
        let input_dim = deter_dim;
        let hidden_dim = 256;
        let output_dim = stoch_total;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, output_dim, &mut rng);
        let mut weights = w1;
        weights.extend_from_slice(&w2);
        Self {
            weights,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Compute prior stochastic logits from deter hidden.
    pub fn forward(&self, deter: &[f32]) -> Vec<f32> {
        let sz1 = self.input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.weights[..sz1],
            deter,
            self.hidden_dim,
            self.input_dim,
        ));
        matvec(&self.weights[sz1..], &h, self.output_dim, self.hidden_dim)
    }
}

/// Straight-through categorical sample.
///
/// Returns (one_hot, straight_through_sample) where the gradient flows through
/// the logits via the straight-through estimator.
pub fn stoch_straight_through(
    logits: &[f32],
    n_classes: usize,
    rng: &mut StdRng,
) -> (Vec<f32>, Vec<f32>) {
    let n_groups = logits.len() / n_classes;
    let mut one_hot = vec![0.0_f32; logits.len()];
    let mut st = logits.to_vec();

    for g in 0..n_groups {
        let base = g * n_classes;
        let group = &logits[base..base + n_classes];
        let probs = softmax(group);
        // Sample via Gumbel-max
        let selected = probs
            .iter()
            .enumerate()
            .max_by(|a, b| {
                let ga = -(-(rng.random::<f32>().max(1e-10)).ln()).ln() + a.1.ln();
                let gb = -(-(rng.random::<f32>().max(1e-10)).ln()).ln() + b.1.ln();
                ga.partial_cmp(&gb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        for c in 0..n_classes {
            one_hot[base + c] = if c == selected { 1.0 } else { 0.0 };
            // Straight-through: st = one_hot + (logits - logits.detach())
            // In a forward-only setting, st equals logits (gradient placeholder).
            st[base + c] = one_hot[base + c] + logits[base + c] - logits[base + c];
        }
    }
    (one_hot, st)
}

/// DreamerV3 world model: combines RecurrentModel + TransitionModel.
pub struct DreamerV3 {
    pub config: RsssmConfig,
    pub recurrent: RecurrentModel,
    pub repr_model: RepresentationModel,
    pub transition: TransitionModel,
}

impl DreamerV3 {
    pub fn new(config: RsssmConfig, action_dim: usize, seed: u64) -> Self {
        let stoch_total = config.stoch_dim * config.n_classes;
        let recurrent = RecurrentModel::new(config.deter_dim, stoch_total + action_dim, seed);
        let repr_model = RepresentationModel::new(
            config.deter_dim,
            config.embed_dim,
            stoch_total,
            seed.wrapping_add(100),
        );
        let transition =
            TransitionModel::new(config.deter_dim, stoch_total, seed.wrapping_add(200));
        Self {
            config,
            recurrent,
            repr_model,
            transition,
        }
    }

    /// One imagination step: (deter, stoch, action) → (next_deter, next_stoch_logits).
    pub fn imagine_step(
        &self,
        deter: &[f32],
        stoch: &[f32],
        action: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut inp = stoch.to_vec();
        inp.extend_from_slice(action);
        let next_deter = self.recurrent.step(deter, &inp);
        let next_stoch_logits = self.transition.forward(&next_deter);
        (next_deter, next_stoch_logits)
    }

    /// Observe step: given observation embedding, return next (deter, stoch_one_hot).
    pub fn observe_step(
        &self,
        deter: &[f32],
        stoch: &[f32],
        action: &[f32],
        embed: &[f32],
        rng: &mut StdRng,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut inp = stoch.to_vec();
        inp.extend_from_slice(action);
        let next_deter = self.recurrent.step(deter, &inp);
        let logits = self.repr_model.forward(&next_deter, embed);
        let (one_hot, _) = stoch_straight_through(&logits, self.config.n_classes, rng);
        (next_deter, one_hot)
    }
}

// ---------------------------------------------------------------------------
// 2. TWM — Transformer World Model
// ---------------------------------------------------------------------------

/// Configuration for Transformer World Model.
#[derive(Debug, Clone)]
pub struct TwmConfig {
    pub seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
}

impl TwmConfig {
    pub fn new(seq_len: usize, d_model: usize, n_heads: usize, n_layers: usize) -> Self {
        Self {
            seq_len,
            d_model,
            n_heads,
            n_layers,
        }
    }
}

/// Encodes each state-action token for the context window.
pub struct TwmStateEncoder {
    pub token_embed: Vec<f32>,
    pub pos_embed: Vec<f32>,
    obs_dim: usize,
    d_model: usize,
    seq_len: usize,
}

impl TwmStateEncoder {
    pub fn new(obs_dim: usize, d_model: usize, seq_len: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let token_embed = xavier_init_vec(obs_dim, d_model, &mut rng);
        let pos_embed: Vec<f32> = (0..seq_len * d_model)
            .map(|_| rng.random::<f32>() * 0.02)
            .collect();
        Self {
            token_embed,
            pos_embed,
            obs_dim,
            d_model,
            seq_len,
        }
    }

    /// Embed a single state vector to d_model dimensions + positional encoding.
    pub fn embed_token(&self, obs: &[f32], pos: usize) -> Vec<f32> {
        let tok = matvec(&self.token_embed, obs, self.d_model, self.obs_dim);
        let pos_start = pos.min(self.seq_len.saturating_sub(1)) * self.d_model;
        let pos_enc = &self.pos_embed[pos_start..pos_start + self.d_model];
        add_vecs(&tok, pos_enc)
    }
}

/// Transformer-based transition head predicts next state distribution.
pub struct TwmTransitionHead {
    pub cross_attn_weights: Vec<f32>,
    d_model: usize,
    output_dim: usize,
}

impl TwmTransitionHead {
    pub fn new(d_model: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let cross_attn_weights = xavier_init_vec(d_model, output_dim, &mut rng);
        Self {
            cross_attn_weights,
            d_model,
            output_dim,
        }
    }

    /// Simple linear projection from context mean to next-state logits.
    fn project(&self, context: &[f32]) -> Vec<f32> {
        matvec(
            &self.cross_attn_weights,
            context,
            self.output_dim,
            self.d_model,
        )
    }
}

/// Full Transformer World Model.
pub struct TransformerWorldModel {
    pub config: TwmConfig,
    pub encoder: TwmStateEncoder,
    pub head: TwmTransitionHead,
    action_dim: usize,
}

impl TransformerWorldModel {
    pub fn new(config: TwmConfig, obs_dim: usize, action_dim: usize, seed: u64) -> Self {
        let encoder =
            TwmStateEncoder::new(obs_dim + action_dim, config.d_model, config.seq_len, seed);
        let head = TwmTransitionHead::new(config.d_model, obs_dim, seed.wrapping_add(1));
        Self {
            action_dim,
            config,
            encoder,
            head,
        }
    }

    /// Predict next state distribution from history of observations + current action.
    pub fn forward(&self, history: &[Vec<f32>], action: &[f32]) -> Vec<f32> {
        let d = self.config.d_model;
        let seq = history.len().min(self.config.seq_len);
        if seq == 0 {
            return vec![0.0; self.head.output_dim];
        }
        let mut context_sum = vec![0.0_f32; d];
        for (pos, obs) in history.iter().rev().take(seq).enumerate() {
            let mut obs_action = obs.clone();
            obs_action.resize(obs.len() + self.action_dim, 0.0);
            let act_start = obs.len();
            for (i, &a) in action.iter().enumerate() {
                obs_action[act_start + i] = a;
            }
            let tok = self.encoder.embed_token(&obs_action, pos);
            for (s, t) in context_sum.iter_mut().zip(tok.iter()) {
                *s += t;
            }
        }
        let scale = 1.0 / seq as f32;
        let context_mean: Vec<f32> = context_sum.iter().map(|&v| v * scale).collect();
        self.head.project(&context_mean)
    }
}

// ---------------------------------------------------------------------------
// 3. IRIS — Discrete Tokenization World Model
// ---------------------------------------------------------------------------

/// VQ-VAE style discrete tokenizer.
pub struct DiscreteTokenizer {
    pub codebook: Vec<Vec<f32>>,
    pub n_codes: usize,
    pub d_code: usize,
}

impl DiscreteTokenizer {
    pub fn new(n_codes: usize, d_code: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let codebook: Vec<Vec<f32>> = (0..n_codes)
            .map(|_| {
                (0..d_code)
                    .map(|_| rng.random::<f32>() * 0.1 - 0.05)
                    .collect()
            })
            .collect();
        Self {
            codebook,
            n_codes,
            d_code,
        }
    }

    /// Encode observation patches: for each d_code-sized patch, find nearest codebook entry.
    pub fn encode(&self, obs: &[f32]) -> Vec<usize> {
        obs.chunks(self.d_code)
            .map(|patch| {
                let mut best_idx = 0usize;
                let mut best_dist = f32::INFINITY;
                for (i, code) in self.codebook.iter().enumerate() {
                    let dist: f32 = patch
                        .iter()
                        .zip(code.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                best_idx
            })
            .collect()
    }

    /// Decode token indices back to flat observation via codebook lookup.
    pub fn decode(&self, tokens: &[usize]) -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&idx| {
                let code = &self.codebook[idx.min(self.n_codes - 1)];
                code.iter().cloned()
            })
            .collect()
    }

    /// VQ-VAE commitment + codebook loss.
    pub fn commitment_loss(&self, obs: &[f32], token_ids: &[usize], commitment_weight: f32) -> f32 {
        let mut total = 0.0_f32;
        for (chunk_idx, patch) in obs.chunks(self.d_code).enumerate() {
            let idx = if chunk_idx < token_ids.len() {
                token_ids[chunk_idx]
            } else {
                0
            };
            let code = &self.codebook[idx.min(self.n_codes - 1)];
            let codebook_loss: f32 = patch
                .iter()
                .zip(code.iter())
                .map(|(z, e)| (z - e) * (z - e))
                .sum();
            let commit_loss: f32 = patch
                .iter()
                .zip(code.iter())
                .map(|(z, e)| (z - e) * (z - e))
                .sum();
            total += codebook_loss + commitment_weight * commit_loss;
        }
        total
    }
}

/// IRIS world model: discrete tokenizer + transformer backbone.
pub struct IrisWorldModel {
    pub tokenizer: DiscreteTokenizer,
    pub transformer: TransformerWorldModel,
}

impl IrisWorldModel {
    pub fn new(
        n_codes: usize,
        d_code: usize,
        obs_dim: usize,
        action_dim: usize,
        seq_len: usize,
        seed: u64,
    ) -> Self {
        let tokenizer = DiscreteTokenizer::new(n_codes, d_code, seed);
        let config = TwmConfig::new(seq_len, 256, 8, 4);
        let transformer =
            TransformerWorldModel::new(config, obs_dim, action_dim, seed.wrapping_add(10));
        Self {
            tokenizer,
            transformer,
        }
    }
}

// ---------------------------------------------------------------------------
// 4. MuZero — Planning with Learned Model
// ---------------------------------------------------------------------------

/// Configuration for MuZero.
#[derive(Debug, Clone)]
pub struct MuZeroConfig {
    pub hidden_dim: usize,
    pub action_dim: usize,
    /// Number of value bins for categorical value prediction.
    pub n_values: usize,
}

impl MuZeroConfig {
    pub fn new(hidden_dim: usize, action_dim: usize, n_values: usize) -> Self {
        Self {
            hidden_dim,
            action_dim,
            n_values,
        }
    }
}

/// Representation network: observation → hidden state.
pub struct MuZeroRepresentationNet {
    pub weights: Vec<f32>,
    obs_dim: usize,
    hidden_dim: usize,
}

impl MuZeroRepresentationNet {
    pub fn new(obs_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(obs_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, hidden_dim, &mut rng);
        let mut weights = w1;
        weights.extend_from_slice(&w2);
        Self {
            weights,
            obs_dim,
            hidden_dim,
        }
    }

    pub fn forward(&self, obs: &[f32]) -> Vec<f32> {
        let sz1 = self.obs_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.weights[..sz1],
            obs,
            self.hidden_dim,
            self.obs_dim,
        ));
        tanh_vec(&matvec(
            &self.weights[sz1..],
            &h,
            self.hidden_dim,
            self.hidden_dim,
        ))
    }
}

/// Dynamics network: hidden + action one-hot → next hidden + reward.
pub struct MuZeroDynamicsNet {
    pub weights: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
}

impl MuZeroDynamicsNet {
    pub fn new(hidden_dim: usize, action_dim: usize, seed: u64) -> Self {
        let input_dim = hidden_dim + action_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        let w2 = xavier_init_vec(hidden_dim, hidden_dim + 1, &mut rng);
        let mut weights = w1;
        weights.extend_from_slice(&w2);
        Self {
            weights,
            input_dim,
            hidden_dim,
        }
    }

    /// Returns (next_hidden, reward_scalar).
    pub fn forward(&self, hidden: &[f32], action_onehot: &[f32]) -> (Vec<f32>, f32) {
        let mut inp = hidden.to_vec();
        inp.extend_from_slice(action_onehot);
        let sz1 = self.input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.weights[..sz1],
            &inp,
            self.hidden_dim,
            self.input_dim,
        ));
        let sz2 = self.hidden_dim * (self.hidden_dim + 1);
        let out = matvec(
            &self.weights[sz1..sz1 + sz2],
            &h,
            self.hidden_dim + 1,
            self.hidden_dim,
        );
        let next_hidden = tanh_vec(&out[..self.hidden_dim]);
        let reward = out[self.hidden_dim];
        (next_hidden, reward)
    }
}

/// Prediction network: hidden → policy logits + value.
pub struct MuZeroPredictionNet {
    pub weights: Vec<f32>,
    hidden_dim: usize,
    action_dim: usize,
    n_values: usize,
}

impl MuZeroPredictionNet {
    pub fn new(hidden_dim: usize, action_dim: usize, n_values: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = xavier_init_vec(hidden_dim, action_dim + n_values, &mut rng);
        Self {
            weights: w,
            hidden_dim,
            action_dim,
            n_values,
        }
    }

    /// Returns (policy_logits, value_scalar).
    pub fn forward(&self, hidden: &[f32]) -> (Vec<f32>, f32) {
        let out = matvec(
            &self.weights,
            hidden,
            self.action_dim + self.n_values,
            self.hidden_dim,
        );
        let policy_logits = out[..self.action_dim].to_vec();
        let value_logits = &out[self.action_dim..];
        // Categorical value: expectation of support {-value_max .. value_max}
        let value_probs = softmax(value_logits);
        let value_max = (self.n_values as f32 - 1.0) / 2.0;
        let value: f32 = value_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| p * (i as f32 - value_max))
            .sum();
        (policy_logits, value)
    }
}

/// MuZero combined loss: policy cross-entropy + value MSE + reward MSE.
pub fn muzero_loss(
    policy_logits: &[f32],
    value: f32,
    target_policy: &[f32],
    target_value: f32,
    reward: f32,
    target_reward: f32,
) -> f32 {
    // Policy: cross-entropy with target_policy as soft target
    let log_probs: Vec<f32> = {
        let probs = softmax(policy_logits);
        probs.iter().map(|&p| (p + 1e-8).ln()).collect()
    };
    let policy_loss: f32 = -target_policy
        .iter()
        .zip(log_probs.iter())
        .map(|(t, lp)| t * lp)
        .sum::<f32>();

    let value_loss = (value - target_value).powi(2);
    let reward_loss = (reward - target_reward).powi(2);
    policy_loss + value_loss + reward_loss
}

/// Full MuZero model.
pub struct MuZeroModel {
    pub repr_net: MuZeroRepresentationNet,
    pub dynamics_net: MuZeroDynamicsNet,
    pub prediction_net: MuZeroPredictionNet,
    pub config: MuZeroConfig,
}

impl MuZeroModel {
    pub fn new(obs_dim: usize, config: MuZeroConfig, seed: u64) -> Self {
        let repr_net = MuZeroRepresentationNet::new(obs_dim, config.hidden_dim, seed);
        let dynamics_net =
            MuZeroDynamicsNet::new(config.hidden_dim, config.action_dim, seed.wrapping_add(1));
        let prediction_net = MuZeroPredictionNet::new(
            config.hidden_dim,
            config.action_dim,
            config.n_values,
            seed.wrapping_add(2),
        );
        Self {
            repr_net,
            dynamics_net,
            prediction_net,
            config,
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Predictive Coding — Hierarchical Predictive Coding
// ---------------------------------------------------------------------------

/// A single layer in the predictive coding hierarchy.
pub struct PcLayer {
    pub pred_weights: Vec<f32>,
    pub err_weights: Vec<f32>,
    pub hidden_dim: usize,
    input_dim: usize,
}

impl PcLayer {
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let pred_weights = xavier_init_vec(hidden_dim, input_dim, &mut rng);
        let err_weights = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        Self {
            pred_weights,
            err_weights,
            hidden_dim,
            input_dim,
        }
    }

    /// Prediction pass: hidden → predicted input.
    pub fn predict(&self, hidden: &[f32]) -> Vec<f32> {
        matvec(&self.pred_weights, hidden, self.input_dim, self.hidden_dim)
    }

    /// Error: actual - prediction.
    pub fn error(&self, actual: &[f32], predicted: &[f32]) -> Vec<f32> {
        actual
            .iter()
            .zip(predicted.iter())
            .map(|(a, p)| a - p)
            .collect()
    }

    /// Forward pass: returns (prediction, prediction_error).
    pub fn forward_pass(&self, x: &[f32], hidden: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let prediction = self.predict(hidden);
        let err = self.error(x, &prediction);
        (prediction, err)
    }

    /// Backward pass: propagate error back to update hidden state.
    pub fn backward_pass(&self, error: &[f32], lr: f32) -> Vec<f32> {
        let delta = matvec(&self.err_weights, error, self.hidden_dim, self.input_dim);
        delta.iter().map(|&d| d * lr).collect()
    }
}

/// Hierarchical predictive coding network.
pub struct PcNetwork {
    pub layers: Vec<PcLayer>,
}

impl PcNetwork {
    pub fn new(dims: &[usize], seed: u64) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len().saturating_sub(1) {
            layers.push(PcLayer::new(
                dims[i],
                dims[i + 1],
                seed.wrapping_add(i as u64),
            ));
        }
        Self { layers }
    }

    /// Iterative inference minimizing free energy (prediction error).
    ///
    /// Returns hidden state activations at each layer.
    pub fn infer(&self, obs: &[f32], n_iters: usize) -> Vec<Vec<f32>> {
        let n_layers = self.layers.len();
        if n_layers == 0 {
            return vec![];
        }
        // Initialize hidden states to zeros
        let mut hiddens: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| vec![0.1_f32; l.hidden_dim])
            .collect();

        for _ in 0..n_iters {
            for layer_idx in 0..n_layers {
                let input = if layer_idx == 0 {
                    obs.to_vec()
                } else {
                    hiddens[layer_idx - 1].clone()
                };
                let (_, err) = self.layers[layer_idx].forward_pass(&input, &hiddens[layer_idx]);
                let update = self.layers[layer_idx].backward_pass(&err, 0.1);
                for (h, u) in hiddens[layer_idx].iter_mut().zip(update.iter()) {
                    *h += u;
                }
            }
        }
        hiddens
    }
}

// ---------------------------------------------------------------------------
// 6. EfficientZero — MuZero + Self-Supervised Consistency
// ---------------------------------------------------------------------------

/// Cosine similarity-based consistency loss: 2 - 2 * cosine(a, b).
pub fn consistency_loss(predicted_hidden: &[f32], target_hidden: &[f32]) -> f32 {
    let norm_p = l2_norm(predicted_hidden).max(1e-8);
    let norm_t = l2_norm(target_hidden).max(1e-8);
    let cosine = dot(predicted_hidden, target_hidden) / (norm_p * norm_t);
    2.0 - 2.0 * cosine
}

/// Augment hidden state with Gaussian noise (self-supervised augmentation).
pub fn augment_state(hidden: &[f32], noise_std: f32, rng: &mut StdRng) -> Vec<f32> {
    hidden
        .iter()
        .map(|&h| h + normal_sample(rng) * noise_std)
        .collect()
}

/// Simple linear projection head for self-supervised contrastive learning.
struct ProjectionHead {
    weights: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl ProjectionHead {
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = xavier_init_vec(input_dim, output_dim, &mut rng);
        Self {
            weights,
            input_dim,
            output_dim,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        relu(&matvec(&self.weights, x, self.output_dim, self.input_dim))
    }
}

/// EfficientZero model: MuZero + projection heads for self-supervised consistency.
pub struct EfficientZeroModel {
    pub repr_net: MuZeroRepresentationNet,
    pub dynamics_net: MuZeroDynamicsNet,
    pub prediction_net: MuZeroPredictionNet,
    projection_net: ProjectionHead,
    projection_pred_net: ProjectionHead,
    action_dim: usize,
}

impl EfficientZeroModel {
    pub fn new(obs_dim: usize, config: MuZeroConfig, proj_dim: usize, seed: u64) -> Self {
        let action_dim = config.action_dim;
        let repr_net = MuZeroRepresentationNet::new(obs_dim, config.hidden_dim, seed);
        let dynamics_net =
            MuZeroDynamicsNet::new(config.hidden_dim, action_dim, seed.wrapping_add(1));
        let prediction_net = MuZeroPredictionNet::new(
            config.hidden_dim,
            action_dim,
            config.n_values,
            seed.wrapping_add(2),
        );
        let projection_net = ProjectionHead::new(config.hidden_dim, proj_dim, seed.wrapping_add(3));
        let projection_pred_net = ProjectionHead::new(proj_dim, proj_dim, seed.wrapping_add(4));
        Self {
            repr_net,
            dynamics_net,
            prediction_net,
            projection_net,
            projection_pred_net,
            action_dim,
        }
    }

    /// Self-supervised consistency step: returns consistency loss scalar.
    pub fn self_supervised_step(
        &self,
        obs: &[f32],
        action: usize,
        next_obs: &[f32],
        rng: &mut StdRng,
    ) -> f32 {
        let hidden = self.repr_net.forward(obs);
        let next_hidden_true = self.repr_net.forward(next_obs);
        let mut action_onehot = vec![0.0_f32; self.action_dim];
        if action < self.action_dim {
            action_onehot[action] = 1.0;
        }
        let (predicted_next, _) = self.dynamics_net.forward(&hidden, &action_onehot);
        let augmented_target = augment_state(&next_hidden_true, 0.01, rng);
        let proj_pred = self
            .projection_pred_net
            .forward(&self.projection_net.forward(&predicted_next));
        let proj_target = self.projection_net.forward(&augmented_target);
        consistency_loss(&proj_pred, &proj_target)
    }
}

// ---------------------------------------------------------------------------
// 7. TDM — Temporal Difference Models
// ---------------------------------------------------------------------------

/// Configuration for Temporal Difference Models.
#[derive(Debug, Clone)]
pub struct TdmConfig {
    pub horizon: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    pub goal_dim: usize,
}

impl TdmConfig {
    pub fn new(horizon: usize, state_dim: usize, action_dim: usize, goal_dim: usize) -> Self {
        Self {
            horizon,
            state_dim,
            action_dim,
            goal_dim,
        }
    }
}

/// TDM network: goal-conditioned value function + policy.
pub struct TdmNetwork {
    pub value_weights: Vec<f32>,
    pub policy_weights: Vec<f32>,
    config: TdmConfig,
    value_input_dim: usize,
    policy_input_dim: usize,
    hidden_dim: usize,
}

impl TdmNetwork {
    pub fn new(config: TdmConfig, seed: u64) -> Self {
        let hidden_dim = 256;
        // Value input: state + action + goal + tau_embedding(32)
        let tau_dim = 32;
        let value_input_dim = config.state_dim + config.action_dim + config.goal_dim + tau_dim;
        let policy_input_dim = config.state_dim + config.goal_dim + tau_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut value_weights = xavier_init_vec(value_input_dim, hidden_dim, &mut rng);
        value_weights.extend_from_slice(&xavier_init_vec(hidden_dim, 1, &mut rng));
        let mut policy_weights = xavier_init_vec(policy_input_dim, hidden_dim, &mut rng);
        policy_weights.extend_from_slice(&xavier_init_vec(hidden_dim, config.action_dim, &mut rng));
        Self {
            value_weights,
            policy_weights,
            config,
            value_input_dim,
            policy_input_dim,
            hidden_dim,
        }
    }

    /// Embed horizon tau as sinusoidal features.
    fn embed_tau(&self, tau: usize) -> Vec<f32> {
        let tau_dim = 32;
        (0..tau_dim)
            .map(|i| {
                let freq = (tau as f32) * (i as f32 + 1.0) * std::f32::consts::PI
                    / self.config.horizon as f32;
                if i % 2 == 0 {
                    freq.sin()
                } else {
                    freq.cos()
                }
            })
            .collect()
    }

    /// Q-value at horizon tau for reaching goal.
    pub fn q_value(&self, state: &[f32], action: &[f32], goal: &[f32], tau: usize) -> f32 {
        let tau_emb = self.embed_tau(tau);
        let mut inp = state.to_vec();
        inp.extend_from_slice(action);
        inp.extend_from_slice(goal);
        inp.extend_from_slice(&tau_emb);
        inp.truncate(self.value_input_dim);
        inp.resize(self.value_input_dim, 0.0);
        let sz1 = self.value_input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.value_weights[..sz1],
            &inp,
            self.hidden_dim,
            self.value_input_dim,
        ));
        let out = matvec(&self.value_weights[sz1..], &h, 1, self.hidden_dim);
        out[0]
    }

    /// Policy for reaching goal in tau steps.
    pub fn policy(&self, state: &[f32], goal: &[f32], tau: usize) -> Vec<f32> {
        let tau_emb = self.embed_tau(tau);
        let mut inp = state.to_vec();
        inp.extend_from_slice(goal);
        inp.extend_from_slice(&tau_emb);
        inp.truncate(self.policy_input_dim);
        inp.resize(self.policy_input_dim, 0.0);
        let sz1 = self.policy_input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.policy_weights[..sz1],
            &inp,
            self.hidden_dim,
            self.policy_input_dim,
        ));
        let out = matvec(
            &self.policy_weights[sz1..],
            &h,
            self.config.action_dim,
            self.hidden_dim,
        );
        tanh_vec(&out)
    }
}

// ---------------------------------------------------------------------------
// 8. ObsDecoder — Decoder from Latent to Observations
// ---------------------------------------------------------------------------

/// Configuration for an observation decoder.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub latent_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub obs_dim: usize,
}

impl DecoderConfig {
    pub fn new(latent_dim: usize, hidden_dims: Vec<usize>, obs_dim: usize) -> Self {
        Self {
            latent_dim,
            hidden_dims,
            obs_dim,
        }
    }
}

/// Multi-layer observation decoder with ReLU activations.
pub struct ObsDecoder {
    /// Each element is (weight_matrix, bias_vector).
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
    layer_sizes: Vec<usize>,
}

impl ObsDecoder {
    pub fn new(config: &DecoderConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims = vec![config.latent_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.obs_dim);
        let mut layers = Vec::new();
        let layer_sizes = dims.clone();
        for i in 0..dims.len() - 1 {
            let w = xavier_init_vec(dims[i], dims[i + 1], &mut rng);
            let b = vec![0.0_f32; dims[i + 1]];
            layers.push((w, b));
        }
        Self {
            layers,
            layer_sizes,
        }
    }

    /// Decode latent vector z to observation space.
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        let mut x = z.to_vec();
        for (layer_idx, (w, b)) in self.layers.iter().enumerate() {
            let in_dim = self.layer_sizes[layer_idx];
            let out_dim = self.layer_sizes[layer_idx + 1];
            x.resize(in_dim, 0.0);
            let mut out = matvec(w, &x, out_dim, in_dim);
            for (o, bi) in out.iter_mut().zip(b.iter()) {
                *o += bi;
            }
            if layer_idx < self.layers.len() - 1 {
                out = relu(&out);
            }
            x = out;
        }
        x
    }

    /// MSE reconstruction loss.
    pub fn reconstruction_loss(&self, pred_obs: &[f32], true_obs: &[f32]) -> f32 {
        let n = pred_obs.len().min(true_obs.len());
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = pred_obs[..n]
            .iter()
            .zip(true_obs[..n].iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        sum / n as f32
    }

    /// Feature-level perceptual loss (MSE in feature space).
    pub fn perceptual_loss(&self, pred_features: &[f32], true_features: &[f32]) -> f32 {
        self.reconstruction_loss(pred_features, true_features)
    }
}

// ---------------------------------------------------------------------------
// 9. WmRewardPredictor — Learned Reward Model for World Models
// ---------------------------------------------------------------------------

/// World-model reward predictor (distinct from robotics::RewardPredictor).
pub struct WmRewardPredictor {
    pub weights: Vec<f32>,
    pub hidden_dim: usize,
    input_dim: usize,
}

impl WmRewardPredictor {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let input_dim = state_dim + action_dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = xavier_init_vec(input_dim, hidden_dim, &mut rng);
        // reward head (1 output) + done head (1 output) packed
        weights.extend_from_slice(&xavier_init_vec(hidden_dim, 2, &mut rng));
        Self {
            weights,
            hidden_dim,
            input_dim,
        }
    }

    fn forward_raw(&self, state: &[f32], action: &[f32]) -> (f32, f32) {
        let mut inp = state.to_vec();
        inp.extend_from_slice(action);
        inp.resize(self.input_dim, 0.0);
        let sz1 = self.input_dim * self.hidden_dim;
        let h = relu(&matvec(
            &self.weights[..sz1],
            &inp,
            self.hidden_dim,
            self.input_dim,
        ));
        let out = matvec(&self.weights[sz1..], &h, 2, self.hidden_dim);
        (out[0], out[1])
    }

    /// Predict scalar reward.
    pub fn predict_reward(&self, state: &[f32], action: &[f32]) -> f32 {
        self.forward_raw(state, action).0
    }

    /// Predict termination probability.
    pub fn predict_done(&self, state: &[f32], action: &[f32]) -> f32 {
        sigmoid(self.forward_raw(state, action).1)
    }

    /// MSE reward loss.
    pub fn reward_loss(&self, pred: f32, target: f32) -> f32 {
        (pred - target).powi(2)
    }

    /// Binary cross-entropy done loss.
    pub fn done_loss(&self, pred_logit: f32, target: bool) -> f32 {
        let t = if target { 1.0_f32 } else { 0.0_f32 };
        let p = sigmoid(pred_logit);
        -(t * (p + 1e-8).ln() + (1.0 - t) * (1.0 - p + 1e-8).ln())
    }
}

// ---------------------------------------------------------------------------
// 10. LatentPlanner — CEM and MPPI planning in latent space
// ---------------------------------------------------------------------------

/// Cross-Entropy Method planner configuration.
#[derive(Debug, Clone)]
pub struct Cem {
    pub pop_size: usize,
    pub elite_frac: f32,
    pub n_iters: usize,
    pub horizon: usize,
}

impl Cem {
    pub fn new(pop_size: usize, elite_frac: f32, n_iters: usize, horizon: usize) -> Self {
        Self {
            pop_size,
            elite_frac,
            n_iters,
            horizon,
        }
    }
}

/// Rollout a flat action sequence in latent space using dynamics model.
/// Returns total accumulated reward.
fn rollout_reward(
    init_state: &[f32],
    actions: &[Vec<f32>],
    goal: &[f32],
    dynamics: &MuZeroDynamicsNet,
) -> f32 {
    let mut state = init_state.to_vec();
    let mut total = 0.0_f32;
    for action in actions {
        let (next_state, reward) = dynamics.forward(&state, action);
        // Goal-conditioned shaping: negative distance to goal
        let goal_dist: f32 = state
            .iter()
            .zip(goal.iter())
            .map(|(s, g)| (s - g).powi(2))
            .sum::<f32>()
            .sqrt();
        total += reward - 0.01 * goal_dist;
        state = next_state;
    }
    total
}

/// LatentPlanner: CEM and MPPI planning in latent space.
pub struct LatentPlanner {
    pub cem: Cem,
    pub action_dim: usize,
}

impl LatentPlanner {
    pub fn new(cem: Cem, action_dim: usize) -> Self {
        Self { cem, action_dim }
    }

    /// CEM planning: returns sequence of optimized actions.
    ///
    /// Uses a stub dynamics rollout (reward from goal distance).
    pub fn plan(
        &self,
        init_state: &[f32],
        goal: &[f32],
        horizon: usize,
        rng: &mut StdRng,
    ) -> Vec<Vec<f32>> {
        let ad = self.action_dim;
        let n_elite = ((self.cem.pop_size as f32 * self.cem.elite_frac) as usize).max(1);
        // Initialize distribution parameters: mean=0, std=1 per action per step
        let mut mean = vec![vec![0.0_f32; ad]; horizon];
        let mut std = vec![vec![1.0_f32; ad]; horizon];

        for _iter in 0..self.cem.n_iters {
            // Sample population
            let mut population: Vec<Vec<Vec<f32>>> = (0..self.cem.pop_size)
                .map(|_| {
                    mean.iter()
                        .zip(std.iter())
                        .map(|(m, s)| {
                            m.iter()
                                .zip(s.iter())
                                .map(|(&mi, &si)| mi + normal_sample(rng) * si)
                                .collect::<Vec<f32>>()
                        })
                        .collect::<Vec<Vec<f32>>>()
                })
                .collect();

            // Score each trajectory using simple goal-distance reward
            let mut scores: Vec<(f32, usize)> = population
                .iter()
                .enumerate()
                .map(|(i, traj)| {
                    let score = self.score_trajectory(init_state, traj, goal);
                    (score, i)
                })
                .collect();

            // Sort by descending score (higher is better)
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let elite_indices: Vec<usize> = scores[..n_elite].iter().map(|(_, i)| *i).collect();

            // Update mean and std from elite set
            for t in 0..horizon {
                for d in 0..ad {
                    let elite_vals: Vec<f32> =
                        elite_indices.iter().map(|&i| population[i][t][d]).collect();
                    let new_mean = elite_vals.iter().sum::<f32>() / elite_vals.len() as f32;
                    let variance = elite_vals
                        .iter()
                        .map(|&v| (v - new_mean).powi(2))
                        .sum::<f32>()
                        / elite_vals.len() as f32;
                    mean[t][d] = new_mean;
                    std[t][d] = (variance + 1e-6).sqrt();
                }
            }

            // Suppress unused variable warning
            let _ = population.drain(..);
        }

        mean
    }

    /// Score a trajectory: negative sum of squared distances from goal at each step.
    fn score_trajectory(&self, init_state: &[f32], actions: &[Vec<f32>], goal: &[f32]) -> f32 {
        let state_dim = init_state.len().min(goal.len());
        let mut state = init_state.to_vec();
        let mut total = 0.0_f32;
        for action in actions {
            // Simple linear dynamics placeholder for latent planning
            let next_state: Vec<f32> = state
                .iter()
                .zip(action.iter().cycle())
                .map(|(s, a)| s + 0.01 * a)
                .collect();
            let dist: f32 = state[..state_dim]
                .iter()
                .zip(goal[..state_dim].iter())
                .map(|(s, g)| (s - g).powi(2))
                .sum::<f32>()
                .sqrt();
            total -= dist;
            state = next_state;
        }
        total
    }

    /// MPPI planning: importance-weighted action trajectory.
    pub fn mppi_plan(
        &self,
        init_state: &[f32],
        goal: &[f32],
        horizon: usize,
        n_samples: usize,
        temp: f32,
        rng: &mut StdRng,
    ) -> Vec<Vec<f32>> {
        let ad = self.action_dim;
        // Nominal action sequence (zeros)
        let mut nominal: Vec<Vec<f32>> = vec![vec![0.0_f32; ad]; horizon];

        // Sample perturbations
        let perturbations: Vec<Vec<Vec<f32>>> = (0..n_samples)
            .map(|_| {
                (0..horizon)
                    .map(|_| (0..ad).map(|_| normal_sample(rng)).collect::<Vec<f32>>())
                    .collect()
            })
            .collect();

        // Score each perturbed trajectory
        let scores: Vec<f32> = perturbations
            .iter()
            .map(|pert| {
                let actions: Vec<Vec<f32>> = nominal
                    .iter()
                    .zip(pert.iter())
                    .map(|(n, p)| n.iter().zip(p.iter()).map(|(ni, pi)| ni + pi).collect())
                    .collect();
                self.score_trajectory(init_state, &actions, goal)
            })
            .collect();

        // Compute importance weights via softmax over scores / temp
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores
            .iter()
            .map(|&s| ((s - max_score) / temp.max(1e-8)).exp())
            .collect();
        let sum_exp: f32 = exps.iter().sum::<f32>().max(1e-8);
        let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        // Update nominal with weighted perturbations
        for t in 0..horizon {
            for d in 0..ad {
                let weighted_update: f32 = weights
                    .iter()
                    .zip(perturbations.iter())
                    .map(|(&w, pert)| w * pert[t][d])
                    .sum();
                nominal[t][d] += weighted_update;
            }
        }

        nominal
    }
}

