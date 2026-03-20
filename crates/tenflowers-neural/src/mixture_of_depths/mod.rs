//! Mixture of Depths (MoD) — TenfloweRS.
//!
//! Implements the Mixture of Depths framework (Raposo et al. 2024) and related
//! adaptive-depth transformer components: token routing, early exit (ACT-style),
//! PonderNet, Universal Transformers, conditional computation gates, and efficiency
//! metrics for measuring actual versus theoretical computation.
//!
//! All public types carry the `Mod` prefix to avoid collisions with
//! `adaptive_computation` types.

use scirs2_core::random::{rngs::SmallRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

#[cfg(test)]
mod tests;

// ── Error type ─────────────────────────────────────────────────────────────────

/// Error type for all MoD operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ModError {
    /// Input has an unexpected shape or length.
    InvalidInput(String),
    /// Two shapes are incompatible.
    ShapeMismatch(String),
    /// A numerical computation produced an invalid result.
    NumericalError(String),
}

impl fmt::Display for ModError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModError::InvalidInput(m) => write!(f, "ModError::InvalidInput: {}", m),
            ModError::ShapeMismatch(m) => write!(f, "ModError::ShapeMismatch: {}", m),
            ModError::NumericalError(m) => write!(f, "ModError::NumericalError: {}", m),
        }
    }
}

impl std::error::Error for ModError {}

// ── Math utilities ─────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum < 1e-15 { 1.0 } else { sum };
    exps.iter().map(|&e| e / denom).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| dot(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn layer_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = (var + eps).sqrt();
    x.iter().map(|&v| (v - mean) / std).collect()
}

fn xavier_init(rows: usize, cols: usize, rng: &mut SmallRng) -> Vec<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    (0..rows * cols)
        .map(|_| rng.random::<f64>() * 2.0 * limit - limit)
        .collect()
}

fn he_init(rows: usize, cols: usize, rng: &mut SmallRng) -> Vec<f64> {
    let std = (2.0 / cols as f64).sqrt();
    (0..rows * cols)
        .map(|_| {
            // Box-Muller
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

/// Partial-sort top-k indices (descending).
fn top_k_indices(values: &[f64], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

// ── Inner MLP used across all Mod components ───────────────────────────────────

/// Shared MLP block: Linear(dim→hidden) → GELU → Linear(hidden→dim) + residual.
#[derive(Debug, Clone)]
struct ModMlp {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    dim: usize,
    hidden: usize,
}

impl ModMlp {
    fn new(dim: usize, hidden: usize, rng: &mut SmallRng) -> Self {
        ModMlp {
            w1: xavier_init(hidden, dim, rng),
            b1: vec![0.0; hidden],
            w2: xavier_init(dim, hidden, rng),
            b2: vec![0.0; dim],
            dim,
            hidden,
        }
    }

    fn forward(&self, x: &[f64]) -> Result<Vec<f64>, ModError> {
        if x.len() != self.dim {
            return Err(ModError::InvalidInput(format!(
                "ModMlp: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let mut h = matvec(&self.w1, x, self.hidden, self.dim);
        for (hi, &bi) in h.iter_mut().zip(self.b1.iter()) {
            *hi = gelu(*hi + bi);
        }
        let mut out = matvec(&self.w2, &h, self.dim, self.hidden);
        for (oi, &bi) in out.iter_mut().zip(self.b2.iter()) {
            *oi += bi;
        }
        // Residual connection.
        Ok(add_vec(x, &out))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. ModRouter — Token importance router with top-K selection
// ══════════════════════════════════════════════════════════════════════════════

/// Token importance router for Mixture-of-Depths.
///
/// Computes a scalar importance score per token via a learned linear projection,
/// then selects the top-C fraction to route through the expensive computation path.
/// The remaining tokens bypass the layer via the residual stream.
///
/// A load-balancing auxiliary loss encourages uniform utilisation across batches.
pub struct ModRouter {
    /// Linear: dim → 1 (importance score).
    w_score: Vec<f64>,
    b_score: f64,
    dim: usize,
    /// Capacity fraction in (0, 1] — fraction of tokens routed per call.
    capacity_fraction: f64,
}

impl ModRouter {
    /// Create a new `ModRouter`.
    ///
    /// * `dim` — token embedding dimension.
    /// * `capacity_fraction` — fraction of tokens to route through the layer (0 < f ≤ 1).
    pub fn new(dim: usize, capacity_fraction: f64, rng: &mut SmallRng) -> Self {
        let cap = capacity_fraction.clamp(1e-6, 1.0);
        ModRouter {
            w_score: xavier_init(1, dim, rng),
            b_score: 0.0,
            dim,
            capacity_fraction: cap,
        }
    }

    /// Compute scalar routing score for each token (sigmoid-activated).
    ///
    /// `tokens` is a flat slice of shape `[n_tokens × dim]`.
    pub fn score_tokens(&self, tokens: &[f64], n_tokens: usize) -> Result<Vec<f64>, ModError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(ModError::ShapeMismatch(format!(
                "ModRouter::score_tokens: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }
        let scores: Vec<f64> = (0..n_tokens)
            .map(|t| {
                let tok = &tokens[t * self.dim..(t + 1) * self.dim];
                sigmoid(dot(&self.w_score, tok) + self.b_score)
            })
            .collect();
        Ok(scores)
    }

    /// Route tokens: split into (routed_indices, bypassed_indices).
    ///
    /// Returns `(routed_indices, bypassed_indices, scores)`.
    pub fn route(
        &self,
        tokens: &[f64],
        n_tokens: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>), ModError> {
        let scores = self.score_tokens(tokens, n_tokens)?;
        let k = ((n_tokens as f64 * self.capacity_fraction).ceil() as usize)
            .max(1)
            .min(n_tokens);
        let routed = top_k_indices(&scores, k);
        let routed_set: std::collections::HashSet<usize> = routed.iter().cloned().collect();
        let bypassed: Vec<usize> = (0..n_tokens).filter(|i| !routed_set.contains(i)).collect();
        Ok((routed, bypassed, scores))
    }

    /// Load-balancing auxiliary loss.
    ///
    /// Encourages all tokens to have approximately equal routing probability.
    /// Loss = n_tokens · Σ f_i · p_i  where f_i = fraction of tokens selected,
    /// p_i = mean routing probability (simplified scalar variant).
    pub fn auxiliary_loss(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let n = scores.len() as f64;
        let mean_score: f64 = scores.iter().sum::<f64>() / n;
        let target = self.capacity_fraction;
        // Penalise deviation from target capacity.
        (mean_score - target).powi(2)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. ModTransformerLayer — MoD transformer layer (Raposo 2024)
// ══════════════════════════════════════════════════════════════════════════════

/// Mixture-of-Depths transformer layer.
///
/// Routes a subset of tokens (top-C by importance score) through a full
/// attention + FFN block. The remaining tokens bypass via the residual stream.
/// After processing, tokens are recombined into their original sequence order.
pub struct ModTransformerLayer {
    router: ModRouter,
    /// Simplified self-attention (scaled dot-product with linear projections).
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_o: Vec<f64>,
    ffn: ModMlp,
    dim: usize,
    n_heads: usize,
    head_dim: usize,
}

impl ModTransformerLayer {
    /// Create a new MoD transformer layer.
    ///
    /// * `dim` — model dimension (must be divisible by `n_heads`).
    /// * `ffn_hidden` — FFN intermediate dimension.
    /// * `n_heads` — number of attention heads.
    /// * `capacity_fraction` — fraction of tokens to process per forward pass.
    pub fn new(
        dim: usize,
        ffn_hidden: usize,
        n_heads: usize,
        capacity_fraction: f64,
        rng: &mut SmallRng,
    ) -> Result<Self, ModError> {
        if dim % n_heads != 0 {
            return Err(ModError::InvalidInput(format!(
                "ModTransformerLayer: dim={} must be divisible by n_heads={}",
                dim, n_heads
            )));
        }
        let head_dim = dim / n_heads;
        Ok(ModTransformerLayer {
            router: ModRouter::new(dim, capacity_fraction, rng),
            w_q: xavier_init(dim, dim, rng),
            w_k: xavier_init(dim, dim, rng),
            w_v: xavier_init(dim, dim, rng),
            w_o: xavier_init(dim, dim, rng),
            ffn: ModMlp::new(dim, ffn_hidden, rng),
            dim,
            n_heads,
            head_dim,
        })
    }

    /// Scaled dot-product attention for a single query token against key-value pairs.
    fn attend(&self, q: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_logits: Vec<f64> = keys.iter().map(|k| dot(q, k) * scale).collect();
        let attn_weights = softmax(&attn_logits);
        let mut out = vec![0.0_f64; self.head_dim];
        for (w, v) in attn_weights.iter().zip(values.iter()) {
            for (o, &vi) in out.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
        out
    }

    /// Multi-head self-attention for a set of routed tokens.
    ///
    /// `token_vecs` — slice of token embeddings (each of length `dim`).
    fn multi_head_attn(&self, token_vecs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModError> {
        let n = token_vecs.len();
        // Project Q, K, V.
        let qs: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_q, t, self.dim, self.dim))
            .collect();
        let ks: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_k, t, self.dim, self.dim))
            .collect();
        let vs: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_v, t, self.dim, self.dim))
            .collect();

        let mut outputs = vec![vec![0.0_f64; self.dim]; n];
        for qi in 0..n {
            let mut concat = Vec::with_capacity(self.dim);
            for h in 0..self.n_heads {
                let q_h: Vec<f64> = qs[qi][h * self.head_dim..(h + 1) * self.head_dim].to_vec();
                let k_h: Vec<Vec<f64>> = ks
                    .iter()
                    .map(|k| k[h * self.head_dim..(h + 1) * self.head_dim].to_vec())
                    .collect();
                let v_h: Vec<Vec<f64>> = vs
                    .iter()
                    .map(|v| v[h * self.head_dim..(h + 1) * self.head_dim].to_vec())
                    .collect();
                let head_out = self.attend(&q_h, &k_h, &v_h);
                concat.extend(head_out);
            }
            // Output projection.
            let projected = matvec(&self.w_o, &concat, self.dim, self.dim);
            outputs[qi] = add_vec(&token_vecs[qi], &projected); // residual
        }
        Ok(outputs)
    }

    /// Forward pass.
    ///
    /// `tokens` is a flat slice of shape `[n_tokens × dim]`.
    ///
    /// Returns `(output_tokens, routed_indices, aux_loss)`.
    pub fn forward(
        &self,
        tokens: &[f64],
        n_tokens: usize,
    ) -> Result<(Vec<f64>, Vec<usize>, f64), ModError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(ModError::ShapeMismatch(format!(
                "ModTransformerLayer::forward: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }

        let (routed, _bypassed, scores) = self.router.route(tokens, n_tokens)?;
        let aux_loss = self.router.auxiliary_loss(&scores);

        // Extract token vectors.
        let routed_vecs: Vec<Vec<f64>> = routed
            .iter()
            .map(|&t| tokens[t * self.dim..(t + 1) * self.dim].to_vec())
            .collect();

        // Apply attention + FFN to routed tokens.
        let attn_out = self.multi_head_attn(&routed_vecs)?;
        let mut ffn_out: Vec<Vec<f64>> = Vec::with_capacity(attn_out.len());
        for tok in &attn_out {
            let ln = layer_norm(tok, 1e-5);
            ffn_out.push(self.ffn.forward(&ln)?);
        }

        // Recombine: start with original tokens, scatter processed tokens back.
        let mut output = tokens.to_vec();
        for (rank, &t) in routed.iter().enumerate() {
            let processed = layer_norm(&ffn_out[rank], 1e-5);
            output[t * self.dim..(t + 1) * self.dim].copy_from_slice(&processed);
        }

        Ok((output, routed, aux_loss))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. ModEarlyExit — Early-exit transformer (Graves 2016 ACT adaptation)
// ══════════════════════════════════════════════════════════════════════════════

/// Per-token exit decision based on a confidence threshold.
///
/// At each layer, a small MLP computes a per-token exit probability.
/// Tokens whose confidence exceeds `threshold` are "frozen" and no longer
/// updated in subsequent layers. A ponder cost loss penalises total steps.
pub struct ModEarlyExit {
    layers: Vec<ModMlp>,
    /// Exit classifier: dim → 1 (sigmoid probability).
    w_exit: Vec<f64>,
    b_exit: f64,
    dim: usize,
    /// Exit confidence threshold in (0, 1).
    threshold: f64,
    /// Ponder cost regularisation weight λ.
    ponder_lambda: f64,
}

impl ModEarlyExit {
    /// Create a new `ModEarlyExit`.
    pub fn new(
        dim: usize,
        hidden: usize,
        n_layers: usize,
        threshold: f64,
        ponder_lambda: f64,
        rng: &mut SmallRng,
    ) -> Self {
        let layers: Vec<ModMlp> = (0..n_layers)
            .map(|_| ModMlp::new(dim, hidden, rng))
            .collect();
        ModEarlyExit {
            layers,
            w_exit: xavier_init(1, dim, rng),
            b_exit: 0.0,
            dim,
            threshold,
            ponder_lambda,
        }
    }

    /// Forward with per-token early exit.
    ///
    /// Returns `(output, exit_depths, ponder_loss)`.
    ///
    /// * `output` — final representations for all tokens.
    /// * `exit_depths` — depth at which each token exited (0-indexed layer).
    /// * `ponder_loss` — λ · mean(exit_depths).
    pub fn forward_with_exit(
        &self,
        tokens: &[f64],
        n_tokens: usize,
    ) -> Result<(Vec<f64>, Vec<usize>, f64), ModError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(ModError::ShapeMismatch(format!(
                "ModEarlyExit::forward_with_exit: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }

        let max_depth = self.layers.len();
        let mut states: Vec<Vec<f64>> = (0..n_tokens)
            .map(|t| tokens[t * self.dim..(t + 1) * self.dim].to_vec())
            .collect();
        let mut exit_depths = vec![max_depth.saturating_sub(1); n_tokens];
        let mut exited = vec![false; n_tokens];

        for (depth, layer) in self.layers.iter().enumerate() {
            for t in 0..n_tokens {
                if exited[t] {
                    continue;
                }
                states[t] = layer.forward(&states[t])?;
                let ln = layer_norm(&states[t], 1e-5);
                let conf = sigmoid(dot(&self.w_exit, &ln) + self.b_exit);
                if conf >= self.threshold || depth == max_depth - 1 {
                    exit_depths[t] = depth;
                    exited[t] = true;
                }
            }
        }

        let mut output = vec![0.0_f64; n_tokens * self.dim];
        for t in 0..n_tokens {
            output[t * self.dim..(t + 1) * self.dim].copy_from_slice(&states[t]);
        }

        let mean_depth = exit_depths.iter().map(|&d| d as f64).sum::<f64>() / n_tokens as f64;
        let ponder_loss = self.ponder_lambda * mean_depth;

        Ok((output, exit_depths, ponder_loss))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. ModAdaptiveDepth — Per-input adaptive depth via halting probability
// ══════════════════════════════════════════════════════════════════════════════

/// Adaptive depth network using per-step halting probabilities.
///
/// At each layer, a halting unit emits p_t = sigmoid(w·h_t).
/// The process halts when the cumulative sum exceeds 1 − ε.
/// Returns the probability-weighted output (ACT-style).
pub struct ModAdaptiveDepth {
    layers: Vec<ModMlp>,
    w_halt: Vec<f64>,
    b_halt: f64,
    dim: usize,
    eps: f64,
}

impl ModAdaptiveDepth {
    /// Create a new `ModAdaptiveDepth`.
    ///
    /// * `eps` — halting threshold epsilon (typical: 0.01).
    pub fn new(dim: usize, hidden: usize, n_layers: usize, eps: f64, rng: &mut SmallRng) -> Self {
        let layers: Vec<ModMlp> = (0..n_layers)
            .map(|_| ModMlp::new(dim, hidden, rng))
            .collect();
        ModAdaptiveDepth {
            layers,
            w_halt: xavier_init(1, dim, rng),
            b_halt: 0.0,
            dim,
            eps,
        }
    }

    /// Forward pass with adaptive depth.
    ///
    /// Returns `(weighted_output, computation_steps)`.
    pub fn forward(&self, x: &[f64], max_layers: usize) -> Result<(Vec<f64>, usize), ModError> {
        if x.len() != self.dim {
            return Err(ModError::InvalidInput(format!(
                "ModAdaptiveDepth: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let limit = max_layers.min(self.layers.len());
        let mut state = x.to_vec();
        let mut accum = vec![0.0_f64; self.dim];
        let mut cumulative = 0.0_f64;
        let mut n_steps = 0_usize;

        for (step, layer) in self.layers.iter().take(limit).enumerate() {
            state = layer.forward(&state)?;
            let p_t = sigmoid(dot(&self.w_halt, &state) + self.b_halt);
            n_steps = step + 1;

            let threshold = 1.0 - self.eps;
            if cumulative + p_t >= threshold || step == limit - 1 {
                // Last step: use remainder weight.
                let remainder = 1.0 - cumulative;
                for (a, &s) in accum.iter_mut().zip(state.iter()) {
                    *a += remainder * s;
                }
                break;
            } else {
                cumulative += p_t;
                for (a, &s) in accum.iter_mut().zip(state.iter()) {
                    *a += p_t * s;
                }
            }
        }

        Ok((accum, n_steps))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. ModUniversalTransformer — Universal Transformer (Dehghani 2019)
// ══════════════════════════════════════════════════════════════════════════════

/// Universal Transformer with weight sharing across depth.
///
/// A single transformer block is applied repeatedly up to `max_steps` times.
/// When combined with `ModAdaptiveDepth`-style ACT, computation varies per input.
pub struct ModUniversalTransformer {
    /// Shared transformer block applied at every depth step.
    shared_mlp: ModMlp,
    /// Shared self-attention projections.
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_o: Vec<f64>,
    /// Halting unit for ACT integration.
    w_halt: Vec<f64>,
    b_halt: f64,
    dim: usize,
    eps: f64,
}

impl ModUniversalTransformer {
    /// Create a new `ModUniversalTransformer`.
    pub fn new(dim: usize, hidden: usize, eps: f64, rng: &mut SmallRng) -> Self {
        ModUniversalTransformer {
            shared_mlp: ModMlp::new(dim, hidden, rng),
            w_q: xavier_init(dim, dim, rng),
            w_k: xavier_init(dim, dim, rng),
            w_v: xavier_init(dim, dim, rng),
            w_o: xavier_init(dim, dim, rng),
            w_halt: xavier_init(1, dim, rng),
            b_halt: 0.0,
            dim,
            eps,
        }
    }

    /// Apply the shared attention block to a sequence of tokens.
    fn attend_all(&self, token_vecs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ModError> {
        let n = token_vecs.len();
        let scale = (self.dim as f64).sqrt().recip();
        let qs: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_q, t, self.dim, self.dim))
            .collect();
        let ks: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_k, t, self.dim, self.dim))
            .collect();
        let vs: Vec<Vec<f64>> = token_vecs
            .iter()
            .map(|t| matvec(&self.w_v, t, self.dim, self.dim))
            .collect();

        let mut out_vecs = vec![vec![0.0_f64; self.dim]; n];
        for i in 0..n {
            let attn_logits: Vec<f64> = ks.iter().map(|k| dot(&qs[i], k) * scale).collect();
            let weights = softmax(&attn_logits);
            let mut ctx = vec![0.0_f64; self.dim];
            for (w, v) in weights.iter().zip(vs.iter()) {
                for (c, &vi) in ctx.iter_mut().zip(v.iter()) {
                    *c += w * vi;
                }
            }
            let proj = matvec(&self.w_o, &ctx, self.dim, self.dim);
            out_vecs[i] = add_vec(&token_vecs[i], &proj);
        }
        Ok(out_vecs)
    }

    /// Forward pass with ACT-style adaptive depth.
    ///
    /// Applies the shared block repeatedly until halting or `max_steps`.
    ///
    /// `tokens` — flat slice `[n_tokens × dim]`.
    ///
    /// Returns `(output_tokens, steps_used)`.
    pub fn forward(
        &self,
        tokens: &[f64],
        n_tokens: usize,
        max_steps: usize,
    ) -> Result<(Vec<f64>, usize), ModError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(ModError::ShapeMismatch(format!(
                "ModUniversalTransformer::forward: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }

        let mut token_vecs: Vec<Vec<f64>> = (0..n_tokens)
            .map(|t| tokens[t * self.dim..(t + 1) * self.dim].to_vec())
            .collect();

        // Per-token ACT accumulators.
        let mut accum = vec![vec![0.0_f64; self.dim]; n_tokens];
        let mut cumulative = vec![0.0_f64; n_tokens];
        let mut halted = vec![false; n_tokens];
        let mut steps_used = 0_usize;

        for step in 0..max_steps {
            let attn_out = self.attend_all(&token_vecs)?;
            let mut new_vecs = vec![vec![0.0_f64; self.dim]; n_tokens];
            for t in 0..n_tokens {
                let ln = layer_norm(&attn_out[t], 1e-5);
                new_vecs[t] = self.shared_mlp.forward(&ln)?;
            }
            steps_used = step + 1;

            // Update per-token ACT state.
            let all_halted = (0..n_tokens).all(|t| {
                if halted[t] {
                    return true;
                }
                let p_t = sigmoid(dot(&self.w_halt, &new_vecs[t]) + self.b_halt);
                let threshold = 1.0 - self.eps;
                if cumulative[t] + p_t >= threshold || step == max_steps - 1 {
                    let remainder = 1.0 - cumulative[t];
                    for (a, &s) in accum[t].iter_mut().zip(new_vecs[t].iter()) {
                        *a += remainder * s;
                    }
                    halted[t] = true;
                } else {
                    cumulative[t] += p_t;
                    for (a, &s) in accum[t].iter_mut().zip(new_vecs[t].iter()) {
                        *a += p_t * s;
                    }
                }
                halted[t]
            });

            token_vecs = new_vecs;
            if all_halted {
                break;
            }
        }

        let mut output = vec![0.0_f64; n_tokens * self.dim];
        for t in 0..n_tokens {
            output[t * self.dim..(t + 1) * self.dim].copy_from_slice(&accum[t]);
        }

        Ok((output, steps_used))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 6. ModConditionalComputation — Layer-wise conditional computation gate
// ══════════════════════════════════════════════════════════════════════════════

/// Layer-wise conditional computation using a learnable binary gate (Bengio 2013).
///
/// At each layer, a sigmoid gate g = sigmoid(w·x + b) decides whether to apply
/// the layer or skip it. In training a soft gate is used; at inference a hard
/// threshold produces a binary decision.
pub struct ModConditionalComputation {
    layers: Vec<ModMlp>,
    /// Gate parameters per layer: w ∈ R^dim, b ∈ R.
    gate_w: Vec<Vec<f64>>,
    gate_b: Vec<f64>,
    dim: usize,
}

impl ModConditionalComputation {
    /// Create a new `ModConditionalComputation`.
    pub fn new(dim: usize, hidden: usize, n_layers: usize, rng: &mut SmallRng) -> Self {
        let layers: Vec<ModMlp> = (0..n_layers)
            .map(|_| ModMlp::new(dim, hidden, rng))
            .collect();
        let gate_w: Vec<Vec<f64>> = (0..n_layers).map(|_| xavier_init(1, dim, rng)).collect();
        let gate_b = vec![0.0_f64; n_layers];
        ModConditionalComputation {
            layers,
            gate_w,
            gate_b,
            dim,
        }
    }

    /// Forward pass with conditional computation.
    ///
    /// `gate_threshold` — hard gate threshold for inference (e.g. 0.5).
    ///
    /// Returns `(output, active_fraction)` where `active_fraction` is the
    /// mean fraction of layers applied across the forward pass.
    pub fn forward(&self, x: &[f64], gate_threshold: f64) -> Result<(Vec<f64>, f64), ModError> {
        if x.len() != self.dim {
            return Err(ModError::InvalidInput(format!(
                "ModConditionalComputation: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let mut state = x.to_vec();
        let mut active_count = 0_usize;

        for (i, layer) in self.layers.iter().enumerate() {
            let gate_score = sigmoid(dot(&self.gate_w[i], &state) + self.gate_b[i]);
            if gate_score >= gate_threshold {
                let layer_out = layer.forward(&state)?;
                // Soft gate: multiply layer output by gate score (straight-through).
                state = layer_out
                    .iter()
                    .zip(state.iter())
                    .map(|(&o, &s)| gate_score * o + (1.0 - gate_score) * s)
                    .collect();
                active_count += 1;
            }
            // Else: skip this layer entirely (residual passthrough).
        }

        let active_fraction = active_count as f64 / self.layers.len().max(1) as f64;
        Ok((state, active_fraction))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 7. ModPonderNet — PonderNet (Banino et al. 2021)
// ══════════════════════════════════════════════════════════════════════════════

/// PonderNet recurrent block with geometric halting prior.
///
/// At each step `n`:
/// * h_{n+1} = tanh(W_h · [h_n; x] + b_h)
/// * y_n = W_y · h_n + b_y
/// * λ_n = sigmoid(w_λ · h_n + b_λ)   (halting probability)
///
/// Halting distribution p(n) = λ_n · Π_{i<n}(1 − λ_i).
/// KL loss against Geometric(prior_p) prior.
pub struct ModPonderNet {
    w_update: Vec<f64>,
    b_update: Vec<f64>,
    w_output: Vec<f64>,
    b_output: Vec<f64>,
    w_halt: Vec<f64>,
    b_halt: f64,
    prior_p: f64,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    rng: SmallRng,
}

impl ModPonderNet {
    /// Create a new `ModPonderNet`.
    ///
    /// * `prior_p` — prior halting probability for the Geometric distribution.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        prior_p: f64,
        seed: u64,
        rng: &mut SmallRng,
    ) -> Self {
        let combined = input_dim + hidden_dim;
        ModPonderNet {
            w_update: xavier_init(hidden_dim, combined, rng),
            b_update: vec![0.0; hidden_dim],
            w_output: xavier_init(output_dim, hidden_dim, rng),
            b_output: vec![0.0; output_dim],
            w_halt: xavier_init(1, hidden_dim, rng),
            b_halt: 0.0,
            prior_p: prior_p.clamp(1e-6, 1.0 - 1e-6),
            input_dim,
            hidden_dim,
            output_dim,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Single step: `(h_new, y_n, λ_n)`.
    pub fn step(&self, h: &[f64], x: &[f64]) -> Result<(Vec<f64>, Vec<f64>, f64), ModError> {
        if h.len() != self.hidden_dim {
            return Err(ModError::InvalidInput(format!(
                "ModPonderNet::step: h expected {}, got {}",
                self.hidden_dim,
                h.len()
            )));
        }
        if x.len() != self.input_dim {
            return Err(ModError::InvalidInput(format!(
                "ModPonderNet::step: x expected {}, got {}",
                self.input_dim,
                x.len()
            )));
        }
        let mut combined = h.to_vec();
        combined.extend_from_slice(x);

        let mut h_new = matvec(&self.w_update, &combined, self.hidden_dim, combined.len());
        for (hi, &bi) in h_new.iter_mut().zip(self.b_update.iter()) {
            *hi = (*hi + bi).tanh();
        }
        let mut y_n = matvec(&self.w_output, &h_new, self.output_dim, self.hidden_dim);
        for (yi, &bi) in y_n.iter_mut().zip(self.b_output.iter()) {
            *yi += bi;
        }
        let lambda = sigmoid(dot(&self.w_halt, &h_new) + self.b_halt);
        Ok((h_new, y_n, lambda))
    }

    /// Full PonderNet forward with stochastic halting.
    ///
    /// Returns `(output, halt_probs, kl_loss)`.
    ///
    /// * `halt_probs` — p(halt at step n) for each step up to `max_steps`.
    /// * `kl_loss` — KL(halt_dist || Geometric(prior_p)).
    pub fn forward(
        &mut self,
        x: &[f64],
        max_steps: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), ModError> {
        if x.len() != self.input_dim {
            return Err(ModError::InvalidInput(format!(
                "ModPonderNet::forward: x expected {}, got {}",
                self.input_dim,
                x.len()
            )));
        }
        if max_steps == 0 {
            return Err(ModError::InvalidInput(
                "ModPonderNet::forward: max_steps must be > 0".to_string(),
            ));
        }

        let mut h = vec![0.0_f64; self.hidden_dim];
        let mut lambdas = Vec::with_capacity(max_steps);
        let mut outputs = Vec::with_capacity(max_steps);

        for _ in 0..max_steps {
            let (h_new, y_n, lam) = self.step(&h, x)?;
            lambdas.push(lam);
            outputs.push(y_n);
            h = h_new;
        }

        // Compute p(halt at step n).
        let mut p_halt = Vec::with_capacity(max_steps);
        let mut not_halted = 1.0_f64;
        for &lam in &lambdas {
            let p_n = not_halted * lam;
            p_halt.push(p_n);
            not_halted *= 1.0 - lam;
        }
        if let Some(last) = p_halt.last_mut() {
            *last += not_halted; // accumulate remainder
        }

        // Sample halt step.
        let u: f64 = self.rng.random();
        let mut cumulative = 0.0_f64;
        let mut halt_step = p_halt.len().saturating_sub(1);
        for (n, &ph) in p_halt.iter().enumerate() {
            cumulative += ph;
            if u <= cumulative {
                halt_step = n;
                break;
            }
        }
        let output = outputs[halt_step].clone();

        // KL divergence: KL(p_halt || Geometric(prior_p)).
        let lp = self.prior_p;
        let kl = p_halt
            .iter()
            .enumerate()
            .map(|(n, &p)| {
                if p < 1e-15 {
                    return 0.0;
                }
                let q = (1.0 - lp).powi(n as i32) * lp;
                let q = q.max(1e-15);
                p * (p / q).ln()
            })
            .sum::<f64>();

        Ok((output, p_halt, kl))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 8. ModEfficiencyMetrics — Computation savings metrics
// ══════════════════════════════════════════════════════════════════════════════

/// Report produced by `ModEfficiencyMetrics::compute`.
#[derive(Debug, Clone)]
pub struct ModReport {
    /// FLOPs used as a fraction of the full (maximum) model cost.
    pub flops_fraction: f64,
    /// Average computation depth across samples (layers executed).
    pub avg_depth: f64,
    /// Fraction of tokens actually processed (not bypassed) per layer on average.
    pub token_throughput_ratio: f64,
    /// Total number of samples analysed.
    pub n_samples: usize,
}

/// Metrics for evaluating computational efficiency in MoD systems.
pub struct ModEfficiencyMetrics;

impl ModEfficiencyMetrics {
    /// Compute efficiency report from per-sample routing data.
    ///
    /// * `depths` — depth (number of layers executed) per sample.
    /// * `max_depth` — total number of layers in the full model.
    /// * `token_fractions` — fraction of tokens routed per sample (optional;
    ///   if empty, token_throughput_ratio is reported as 1.0).
    pub fn compute(
        depths: &[usize],
        max_depth: usize,
        token_fractions: &[f64],
    ) -> Result<ModReport, ModError> {
        if depths.is_empty() {
            return Err(ModError::InvalidInput(
                "ModEfficiencyMetrics::compute: depths slice is empty".to_string(),
            ));
        }
        if max_depth == 0 {
            return Err(ModError::InvalidInput(
                "ModEfficiencyMetrics::compute: max_depth must be > 0".to_string(),
            ));
        }

        let n = depths.len();
        let avg_depth = depths.iter().map(|&d| d as f64).sum::<f64>() / n as f64;
        let flops_fraction = avg_depth / max_depth as f64;

        let token_throughput_ratio = if token_fractions.is_empty() {
            1.0
        } else {
            token_fractions.iter().sum::<f64>() / token_fractions.len() as f64
        };

        Ok(ModReport {
            flops_fraction,
            avg_depth,
            token_throughput_ratio,
            n_samples: n,
        })
    }

    /// Mean ponder cost over a batch.
    pub fn mean_ponder_cost(costs: &[f64]) -> f64 {
        if costs.is_empty() {
            return 0.0;
        }
        costs.iter().sum::<f64>() / costs.len() as f64
    }

    /// Compute FLOPs saved as `1 − flops_fraction`.
    pub fn flops_saved(report: &ModReport) -> f64 {
        (1.0 - report.flops_fraction).max(0.0)
    }

    /// Distribution of exit depths: count of samples exiting at each depth.
    pub fn depth_distribution(depths: &[usize], max_depth: usize) -> Vec<usize> {
        let mut hist = vec![0_usize; max_depth.max(1)];
        for &d in depths {
            if d < hist.len() {
                hist[d] += 1;
            }
        }
        hist
    }

    /// Routing entropy: measures how uniform token routing is.
    ///
    /// High entropy = balanced routing; low entropy = unbalanced.
    pub fn routing_entropy(router_probs: &[Vec<f64>]) -> f64 {
        if router_probs.is_empty() {
            return 0.0;
        }
        let entropies: Vec<f64> = router_probs
            .iter()
            .map(|p| {
                p.iter()
                    .map(|&v| if v > 1e-15 { -v * v.ln() } else { 0.0 })
                    .sum::<f64>()
            })
            .collect();
        entropies.iter().sum::<f64>() / entropies.len() as f64
    }
}
