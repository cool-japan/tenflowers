//! Adaptive Computation / Conditional Computation — TenfloweRS.
//!
//! Implements ACT (Graves 2016), Mixture-of-Depths (Raposo 2024), PonderNet (Banino 2021),
//! early-exit networks, dynamic slimming, layer-drop, token skimming, and related utilities
//! for sample-adaptive inference with adaptive compute budgets.

use scirs2_core::random::{rngs::SmallRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

#[cfg(test)]
mod tests;

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur in adaptive-computation modules.
#[derive(Debug, Clone, PartialEq)]
pub enum AcError {
    /// Input tensor has unexpected shape.
    InvalidInput(String),
    /// Two operand shapes do not agree.
    ShapeMismatch(String),
    /// Computation failed for a numerical or algorithmic reason.
    ComputationFailed(String),
}

impl fmt::Display for AcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcError::InvalidInput(msg) => write!(f, "AcError::InvalidInput: {}", msg),
            AcError::ShapeMismatch(msg) => write!(f, "AcError::ShapeMismatch: {}", msg),
            AcError::ComputationFailed(msg) => write!(f, "AcError::ComputationFailed: {}", msg),
        }
    }
}

impl std::error::Error for AcError {}

// ── Math helpers ─────────────────────────────────────────────────────────────

#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax_vec(logits: &[f64]) -> Vec<f64> {
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

fn add_bias(v: &mut [f64], b: &[f64]) {
    for (vi, &bi) in v.iter_mut().zip(b.iter()) {
        *vi += bi;
    }
}

/// Box-Muller normal sample using SmallRng.
fn normal_bm(rng: &mut SmallRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn xavier_init(rows: usize, cols: usize, rng: &mut SmallRng) -> Vec<f64> {
    let limit = (6.0 / (rows + cols) as f64).sqrt();
    (0..rows * cols)
        .map(|_| rng.random::<f64>() * 2.0 * limit - limit)
        .collect()
}

/// Partial-sort top-k: returns k indices of largest values (unsorted order).
pub fn top_k_indices_f64(values: &[f64], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. AcMlpBlock — Simple MLP residual block
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple MLP residual block: x + GELU(W2·GELU(W1·x + b1) + b2).
#[derive(Debug, Clone)]
pub struct AcMlpBlock {
    pub(crate) dim: usize,
    pub(crate) hidden: usize,
    pub(crate) w1: Vec<f64>,
    pub(crate) b1: Vec<f64>,
    pub(crate) w2: Vec<f64>,
    pub(crate) b2: Vec<f64>,
}

impl AcMlpBlock {
    /// Create a new `AcMlpBlock` with the given input/output `dim` and `hidden` size.
    pub fn new(dim: usize, hidden: usize, rng: &mut SmallRng) -> Self {
        AcMlpBlock {
            dim,
            hidden,
            w1: xavier_init(hidden, dim, rng),
            b1: vec![0.0; hidden],
            w2: xavier_init(dim, hidden, rng),
            b2: vec![0.0; dim],
        }
    }

    /// Forward: x + FFN(x), returns vector of length `dim`.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, AcError> {
        if x.len() != self.dim {
            return Err(AcError::InvalidInput(format!(
                "AcMlpBlock: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let mut h = matvec(&self.w1, x, self.hidden, self.dim);
        add_bias(&mut h, &self.b1);
        let h: Vec<f64> = h.iter().map(|&v| gelu(v)).collect();
        let mut out = matvec(&self.w2, &h, self.dim, self.hidden);
        add_bias(&mut out, &self.b2);
        // Residual
        let result: Vec<f64> = out.iter().zip(x.iter()).map(|(o, xi)| o + xi).collect();
        Ok(result)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. PonderingState — per-sample state for ACT
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-sample state for Adaptive Computation Time tracking.
#[derive(Debug, Clone, Default)]
pub struct PonderingState {
    /// Halting probability output at this step.
    pub halting_prob: f64,
    /// Sum of all halting probabilities so far (excluding remainder).
    pub cumulative_prob: f64,
    /// Number of computation steps executed so far.
    pub n_steps: usize,
    /// Remainder = 1 - cumulative_prob at the halting step.
    pub remainder: f64,
}

impl PonderingState {
    /// Create a fresh state for a new sample.
    pub fn new() -> Self {
        Self::default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. ActLayer — Adaptive Computation Time (Graves 2016)
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the ACT layer.
#[derive(Debug, Clone)]
pub struct ActConfig {
    pub dim: usize,
    pub hidden: usize,
    pub max_steps: usize,
    pub threshold: f64,
}

/// ACT layer: Universal Transformer-style Adaptive Computation Time.
///
/// Given an input vector x, repeatedly applies an MLP block until cumulative
/// halting probability ≥ threshold (or max_steps is reached).
/// Ponder cost = N + R (steps + remainder).
pub struct ActLayer {
    config: ActConfig,
    block: AcMlpBlock,
    /// Halting unit: Linear (dim → 1) + sigmoid.
    halt_w: Vec<f64>,
    halt_b: f64,
}

impl ActLayer {
    /// Create a new ACT layer.
    pub fn new(config: ActConfig, rng: &mut SmallRng) -> Self {
        let block = AcMlpBlock::new(config.dim, config.hidden, rng);
        let halt_w = xavier_init(1, config.dim, rng);
        ActLayer {
            config,
            block,
            halt_w,
            halt_b: 0.0,
        }
    }

    /// Run ACT on a single sample vector.
    ///
    /// Returns `(weighted_output, ponder_cost)` where `weighted_output` is the
    /// probability-weighted sum of intermediate states and `ponder_cost = N + R`.
    pub fn act_forward(&self, x: &[f64]) -> Result<(Vec<f64>, f64), AcError> {
        if x.len() != self.config.dim {
            return Err(AcError::InvalidInput(format!(
                "ActLayer: expected dim={}, got {}",
                self.config.dim,
                x.len()
            )));
        }
        let threshold = self.config.threshold;
        let max_steps = self.config.max_steps;

        let mut state: Vec<f64> = x.to_vec();
        let mut accum_output: Vec<f64> = vec![0.0; self.config.dim];
        let mut cumulative_prob = 0.0_f64;
        let mut n_steps = 0_usize;
        let mut remainder = 0.0_f64;

        for step in 0..max_steps {
            state = self.block.forward(&state)?;

            // Compute halting probability via linear unit + sigmoid.
            let raw_halt = dot(&self.halt_w, &state) + self.halt_b;
            let p_halt = sigmoid(raw_halt);

            n_steps = step + 1;

            if cumulative_prob + p_halt >= threshold || step == max_steps - 1 {
                // Use remainder as weight for last step.
                remainder = 1.0 - cumulative_prob;
                for (acc, &s) in accum_output.iter_mut().zip(state.iter()) {
                    *acc += remainder * s;
                }
                cumulative_prob += remainder;
                break;
            } else {
                cumulative_prob += p_halt;
                for (acc, &s) in accum_output.iter_mut().zip(state.iter()) {
                    *acc += p_halt * s;
                }
            }
        }

        let ponder_cost = n_steps as f64 + remainder;
        Ok((accum_output, ponder_cost))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. AdaptiveDepthNetwork
// ═══════════════════════════════════════════════════════════════════════════════

/// A network that dynamically selects computation depth per sample.
///
/// Layers are applied sequentially; a halting unit decides whether to stop.
/// Returns `(output, actual_depth_used)`.
pub struct AdaptiveDepthNetwork {
    layers: Vec<AcMlpBlock>,
    halt_w: Vec<f64>,
    halt_b: f64,
    dim: usize,
    threshold: f64,
}

impl AdaptiveDepthNetwork {
    /// Create a new network with `n_layers` MLP residual blocks of `dim/hidden`.
    pub fn new(
        dim: usize,
        hidden: usize,
        n_layers: usize,
        threshold: f64,
        rng: &mut SmallRng,
    ) -> Self {
        let layers: Vec<AcMlpBlock> = (0..n_layers)
            .map(|_| AcMlpBlock::new(dim, hidden, rng))
            .collect();
        let halt_w = xavier_init(1, dim, rng);
        AdaptiveDepthNetwork {
            layers,
            halt_w,
            halt_b: 0.0,
            dim,
            threshold,
        }
    }

    /// Forward pass with compute budget (max number of layers to run).
    ///
    /// Halts when cumulative halting probability exceeds `threshold`.
    /// Returns `(output, depth_used)`.
    pub fn forward(&self, x: &[f64], budget: usize) -> Result<(Vec<f64>, usize), AcError> {
        if x.len() != self.dim {
            return Err(AcError::InvalidInput(format!(
                "AdaptiveDepthNetwork: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let limit = budget.min(self.layers.len());
        let mut state = x.to_vec();
        let mut cumulative_prob = 0.0_f64;

        for (depth, layer) in self.layers.iter().take(limit).enumerate() {
            state = layer.forward(&state)?;
            let raw = dot(&self.halt_w, &state) + self.halt_b;
            cumulative_prob += sigmoid(raw);
            if cumulative_prob >= self.threshold {
                return Ok((state, depth + 1));
            }
        }
        Ok((state, limit))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. AcEarlyExitNetwork — multi-exit network with confidence-based early exit
// ═══════════════════════════════════════════════════════════════════════════════

/// Exit classifier: linear projection dim → n_classes.
#[derive(Debug, Clone)]
struct ExitClassifier {
    w: Vec<f64>,
    b: Vec<f64>,
    n_classes: usize,
    dim: usize,
}

impl ExitClassifier {
    fn new(dim: usize, n_classes: usize, rng: &mut SmallRng) -> Self {
        ExitClassifier {
            w: xavier_init(n_classes, dim, rng),
            b: vec![0.0; n_classes],
            n_classes,
            dim,
        }
    }

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        let mut out = matvec(&self.w, x, self.n_classes, self.dim);
        add_bias(&mut out, &self.b);
        softmax_vec(&out)
    }
}

/// Multi-exit network: exits at layers [n/4, n/2, 3n/4, n].
/// Exits early when `max(softmax)` confidence exceeds `confidence_threshold`.
pub struct AcEarlyExitNetwork {
    layers: Vec<AcMlpBlock>,
    exit_classifiers: Vec<ExitClassifier>,
    exit_indices: Vec<usize>,
    n_classes: usize,
    dim: usize,
}

impl AcEarlyExitNetwork {
    /// Create a new early-exit network.
    ///
    /// `n_layers` total layers; exits are placed at 1/4, 1/2, 3/4, and full depth.
    pub fn new(
        dim: usize,
        hidden: usize,
        n_layers: usize,
        n_classes: usize,
        rng: &mut SmallRng,
    ) -> Self {
        let layers: Vec<AcMlpBlock> = (0..n_layers)
            .map(|_| AcMlpBlock::new(dim, hidden, rng))
            .collect();
        // Exit positions: quarter, half, three-quarter, full (0-indexed last).
        let exit_indices: Vec<usize> = vec![
            n_layers.saturating_sub(1) / 4,
            n_layers / 2,
            3 * n_layers / 4,
            n_layers.saturating_sub(1),
        ];
        let exit_classifiers: Vec<ExitClassifier> = exit_indices
            .iter()
            .map(|_| ExitClassifier::new(dim, n_classes, rng))
            .collect();
        AcEarlyExitNetwork {
            layers,
            exit_classifiers,
            exit_indices,
            n_classes,
            dim,
        }
    }

    /// Forward with confidence-based early exit.
    ///
    /// Returns `(output_probs, exit_layer_index)`.
    /// `exit_layer_index` is the index of the exit that fired (0..=3).
    pub fn forward_with_early_exit(
        &self,
        x: &[f64],
        confidence_threshold: f64,
    ) -> Result<(Vec<f64>, usize), AcError> {
        if x.len() != self.dim {
            return Err(AcError::InvalidInput(format!(
                "AcEarlyExitNetwork: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let mut state = x.to_vec();
        let mut exit_ptr = 0_usize;
        let last_exit = self.exit_classifiers.len().saturating_sub(1);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            state = layer.forward(&state)?;

            // Check if this layer index corresponds to an exit point.
            if exit_ptr < self.exit_indices.len() && layer_idx == self.exit_indices[exit_ptr] {
                let probs = self.exit_classifiers[exit_ptr].predict(&state);
                let max_prob = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                if max_prob >= confidence_threshold || exit_ptr == last_exit {
                    return Ok((probs, exit_ptr));
                }
                exit_ptr += 1;
            }
        }

        // Fallback: run final exit classifier on current state.
        let final_probs = self.exit_classifiers[last_exit].predict(&state);
        Ok((final_probs, last_exit))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. ConditionalDepthRouter — learned router for token selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Learned linear router: projects tokens to scalar scores, selects top-k.
/// Supports straight-through estimator for training.
pub struct ConditionalDepthRouter {
    w: Vec<f64>,
    b: f64,
    dim: usize,
}

impl ConditionalDepthRouter {
    /// Create a new router projecting `dim`-dimensional tokens to scalar scores.
    pub fn new(dim: usize, rng: &mut SmallRng) -> Self {
        ConditionalDepthRouter {
            w: xavier_init(1, dim, rng),
            b: 0.0,
            dim,
        }
    }

    /// Compute sigmoid routing scores for a batch of tokens.
    /// `tokens` is a flat slice of shape `[n_tokens, dim]`.
    pub fn score_tokens(&self, tokens: &[f64], n_tokens: usize) -> Result<Vec<f64>, AcError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(AcError::ShapeMismatch(format!(
                "ConditionalDepthRouter: expected {}*{}={} floats, got {}",
                n_tokens,
                self.dim,
                n_tokens * self.dim,
                tokens.len()
            )));
        }
        let scores: Vec<f64> = (0..n_tokens)
            .map(|t| {
                let tok = &tokens[t * self.dim..(t + 1) * self.dim];
                sigmoid(dot(&self.w, tok) + self.b)
            })
            .collect();
        Ok(scores)
    }

    /// Select top-k token indices via partial sort.
    pub fn top_k_routing(&self, scores: &[f64], k: usize) -> Vec<usize> {
        top_k_indices_f64(scores, k)
    }

    /// Straight-through estimator: binarise scores, but pass gradients as if identity.
    /// At inference this is simply `score >= 0.5`.
    pub fn straight_through_gate(&self, scores: &[f64]) -> Vec<f64> {
        scores
            .iter()
            .map(|&s| if s >= 0.5 { 1.0 } else { 0.0 })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. AcMixtureOfDepthsLayer — MoD (Raposo et al. 2024)
// ═══════════════════════════════════════════════════════════════════════════════

/// Mixture-of-Depths layer: routes top-k tokens through a transformer block;
/// remaining tokens pass through unchanged (residual stream preserved).
pub struct AcMixtureOfDepthsLayer {
    router: ConditionalDepthRouter,
    block: AcMlpBlock,
    dim: usize,
}

impl AcMixtureOfDepthsLayer {
    /// Create a new MoD layer.
    pub fn new(dim: usize, hidden: usize, rng: &mut SmallRng) -> Self {
        AcMixtureOfDepthsLayer {
            router: ConditionalDepthRouter::new(dim, rng),
            block: AcMlpBlock::new(dim, hidden, rng),
            dim,
        }
    }

    /// Forward pass.
    ///
    /// `tokens` — flat slice `[n_tokens × dim]`.
    /// `capacity_fraction` — fraction of tokens to process (0 < f ≤ 1).
    ///
    /// Returns `(output_tokens, routed_indices)`.
    pub fn forward(
        &self,
        tokens: &[f64],
        n_tokens: usize,
        capacity_fraction: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), AcError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(AcError::ShapeMismatch(format!(
                "AcMixtureOfDepthsLayer: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }
        let k = ((n_tokens as f64 * capacity_fraction).ceil() as usize)
            .max(1)
            .min(n_tokens);
        let scores = self.router.score_tokens(tokens, n_tokens)?;
        let routed = self.router.top_k_routing(&scores, k);

        // Copy input to output.
        let mut output = tokens.to_vec();

        // Apply block only to routed tokens.
        for &t in &routed {
            let tok_in = &tokens[t * self.dim..(t + 1) * self.dim];
            let tok_out = self.block.forward(tok_in)?;
            output[t * self.dim..(t + 1) * self.dim].copy_from_slice(&tok_out);
        }

        Ok((output, routed))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. SkimmingModel — fast text processing via token skimming
// ═══════════════════════════════════════════════════════════════════════════════

/// Assigns tokens importance scores and processes only the top-p fraction.
pub struct SkimmingModel {
    scorer_w: Vec<f64>,
    scorer_b: f64,
    processor: AcMlpBlock,
    dim: usize,
}

impl SkimmingModel {
    /// Create a new SkimmingModel.
    pub fn new(dim: usize, hidden: usize, rng: &mut SmallRng) -> Self {
        SkimmingModel {
            scorer_w: xavier_init(1, dim, rng),
            scorer_b: 0.0,
            processor: AcMlpBlock::new(dim, hidden, rng),
            dim,
        }
    }

    /// Skim forward.
    ///
    /// Selects top `skim_ratio` fraction of tokens by importance score,
    /// applies full processing, scatters results back.
    ///
    /// Returns `(processed_tokens, skimmed_mask)` where mask\[i\] = true if token i was processed.
    pub fn skim_forward(
        &self,
        tokens: &[f64],
        n_tokens: usize,
        skim_ratio: f64,
    ) -> Result<(Vec<f64>, Vec<bool>), AcError> {
        if tokens.len() != n_tokens * self.dim {
            return Err(AcError::ShapeMismatch(format!(
                "SkimmingModel: expected {} floats, got {}",
                n_tokens * self.dim,
                tokens.len()
            )));
        }
        let k = ((n_tokens as f64 * skim_ratio).ceil() as usize)
            .max(1)
            .min(n_tokens);

        // Compute importance scores.
        let scores: Vec<f64> = (0..n_tokens)
            .map(|t| {
                let tok = &tokens[t * self.dim..(t + 1) * self.dim];
                sigmoid(dot(&self.scorer_w, tok) + self.scorer_b)
            })
            .collect();

        let selected = top_k_indices_f64(&scores, k);
        let mut mask = vec![false; n_tokens];
        for &i in &selected {
            mask[i] = true;
        }

        let mut output = tokens.to_vec();
        for &t in &selected {
            let tok_in = &tokens[t * self.dim..(t + 1) * self.dim];
            let tok_out = self.processor.forward(tok_in)?;
            output[t * self.dim..(t + 1) * self.dim].copy_from_slice(&tok_out);
        }

        Ok((output, mask))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. AcGatingMechanism — gating strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Supported gating strategies for conditional computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AcGatingStrategy {
    /// Standard sigmoid gate (binary-ish).
    Sigmoid,
    /// Exact sparse projection via bisection (Sparsemax).
    Sparsemax,
    /// α=1.5 sparse-max (Entmax-1.5).
    Entmax15,
    /// Relaxed discrete top-K (SoftTop-K).
    SoftTopK { k: usize },
}

/// Gating mechanism supporting multiple strategies.
pub struct AcGatingMechanism {
    pub strategy: AcGatingStrategy,
}

impl AcGatingMechanism {
    /// Create a new gating mechanism with the given strategy.
    pub fn new(strategy: AcGatingStrategy) -> Self {
        AcGatingMechanism { strategy }
    }

    /// Apply the gate to logits, returning gate weights.
    pub fn apply_gate(&self, logits: &[f64]) -> Result<Vec<f64>, AcError> {
        if logits.is_empty() {
            return Err(AcError::InvalidInput(
                "apply_gate: empty logits".to_string(),
            ));
        }
        match self.strategy {
            AcGatingStrategy::Sigmoid => Ok(logits.iter().map(|&v| sigmoid(v)).collect()),
            AcGatingStrategy::Sparsemax => sparsemax(logits),
            AcGatingStrategy::Entmax15 => entmax15(logits),
            AcGatingStrategy::SoftTopK { k } => soft_top_k(logits, k),
        }
    }
}

/// Sparsemax projection via bisection (Peters et al. 2016).
fn sparsemax(logits: &[f64]) -> Result<Vec<f64>, AcError> {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Find threshold via bisection logic (cumulative support).
    let mut rho = 0_usize;
    let mut cumsum = 0.0_f64;
    for (i, &z) in sorted.iter().enumerate() {
        cumsum += z;
        if z > (cumsum - 1.0) / (i as f64 + 1.0) {
            rho = i + 1;
        } else {
            break;
        }
    }
    let tau = (sorted[..rho].iter().sum::<f64>() - 1.0) / rho as f64;
    let result: Vec<f64> = logits.iter().map(|&z| (z - tau).max(0.0)).collect();
    Ok(result)
}

/// Entmax-1.5 via bisection (Peters et al. 2019).
fn entmax15(logits: &[f64]) -> Result<Vec<f64>, AcError> {
    // p_i = max(0, (z_i - tau)^+)^2  normalized, solved by bisection.
    // We use a fixed number of bisection steps.
    let max_z = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = logits.iter().map(|&z| z - max_z).collect();

    let mut lo = -1.0_f64;
    let mut hi = shifted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let sum: f64 = shifted.iter().map(|&z| (z - mid).max(0.0).powi(2)).sum();
        if sum > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tau = (lo + hi) / 2.0;
    let unnorm: Vec<f64> = shifted
        .iter()
        .map(|&z| (z - tau).max(0.0).powi(2))
        .collect();
    let sum: f64 = unnorm.iter().sum();
    let denom = if sum < 1e-15 { 1.0 } else { sum };
    Ok(unnorm.iter().map(|&v| v / denom).collect())
}

/// Soft top-K gating: top-k entries get their softmax values, rest get 0.
fn soft_top_k(logits: &[f64], k: usize) -> Result<Vec<f64>, AcError> {
    let k = k.min(logits.len()).max(1);
    let selected = top_k_indices_f64(logits, k);
    let top_logits: Vec<f64> = selected.iter().map(|&i| logits[i]).collect();
    let sm = softmax_vec(&top_logits);
    let mut out = vec![0.0_f64; logits.len()];
    for (rank, &idx) in selected.iter().enumerate() {
        out[idx] = sm[rank];
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. DynamicSlimmingLayer — width-adaptive inference
// ═══════════════════════════════════════════════════════════════════════════════

/// Width-adaptive layer using BatchNorm γ as channel importance.
/// At full width: x + W2·GELU(W1·x). At slimmed width: only top-p channels.
pub struct DynamicSlimmingLayer {
    w1: Vec<f64>,
    b1: Vec<f64>,
    /// BatchNorm scale (γ) used as channel importance.
    bn_gamma: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    dim: usize,
    hidden: usize,
}

impl DynamicSlimmingLayer {
    /// Create a new DynamicSlimmingLayer with learned BatchNorm γ.
    pub fn new(dim: usize, hidden: usize, rng: &mut SmallRng) -> Self {
        DynamicSlimmingLayer {
            w1: xavier_init(hidden, dim, rng),
            b1: vec![0.0; hidden],
            bn_gamma: vec![1.0; hidden],
            w2: xavier_init(dim, hidden, rng),
            b2: vec![0.0; dim],
            dim,
            hidden,
        }
    }

    /// Run only the top `width_ratio` fraction of hidden channels.
    pub fn forward_slimmed(&self, x: &[f64], width_ratio: f64) -> Result<Vec<f64>, AcError> {
        if x.len() != self.dim {
            return Err(AcError::InvalidInput(format!(
                "DynamicSlimmingLayer: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let k = ((self.hidden as f64 * width_ratio).ceil() as usize)
            .max(1)
            .min(self.hidden);

        // Select top-k channels by BN γ magnitude.
        let gamma_abs: Vec<f64> = self.bn_gamma.iter().map(|&g| g.abs()).collect();
        let selected_channels = top_k_indices_f64(&gamma_abs, k);

        // Compute hidden activations only for selected channels.
        let mut h_slimmed = vec![0.0_f64; k];
        for (rank, &c) in selected_channels.iter().enumerate() {
            let row = &self.w1[c * self.dim..(c + 1) * self.dim];
            h_slimmed[rank] = gelu(dot(row, x) + self.b1[c]) * self.bn_gamma[c];
        }

        // Back-project from slimmed hidden to output.
        let mut out = self.b2.clone();
        for (rank, &c) in selected_channels.iter().enumerate() {
            for j in 0..self.dim {
                out[j] += self.w2[j * self.hidden + c] * h_slimmed[rank];
            }
        }

        // Residual.
        let result: Vec<f64> = out.iter().zip(x.iter()).map(|(o, xi)| o + xi).collect();
        Ok(result)
    }

    /// Full-width forward (training mode).
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, AcError> {
        self.forward_slimmed(x, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. LayerDropNetwork — stochastic depth (Huang et al. 2016)
// ═══════════════════════════════════════════════════════════════════════════════

/// Stochastic depth network: drops entire residual blocks during training.
/// Survival probability p_l = 1 - (l/L)(1 - p_L).
pub struct LayerDropNetwork {
    layers: Vec<AcMlpBlock>,
    /// p_L = survival probability for the last layer (typical: 0.5).
    p_last: f64,
    rng: SmallRng,
}

impl LayerDropNetwork {
    /// Create a new LayerDropNetwork.
    pub fn new(dim: usize, hidden: usize, n_layers: usize, p_last: f64, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed + 1000);
        let layers: Vec<AcMlpBlock> = (0..n_layers)
            .map(|_| AcMlpBlock::new(dim, hidden, &mut rng))
            .collect();
        LayerDropNetwork {
            layers,
            p_last,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Forward pass.
    ///
    /// During training, each layer l is dropped with probability 1 - p_l.
    /// During inference, all layers are applied (scaled by p_l).
    pub fn forward(&mut self, x: &[f64], training: bool) -> Result<Vec<f64>, AcError> {
        let n = self.layers.len();
        let mut state = x.to_vec();

        for (l, layer) in self.layers.iter().enumerate() {
            let p_l = if n <= 1 {
                self.p_last
            } else {
                1.0 - (l as f64 / (n - 1) as f64) * (1.0 - self.p_last)
            };

            if training {
                let r: f64 = self.rng.random();
                if r < p_l {
                    state = layer.forward(&state)?;
                }
                // else: skip this layer entirely (drop it)
            } else {
                // At inference, scale residual by survival probability.
                let out = layer.forward(&state)?;
                state = state
                    .iter()
                    .zip(out.iter())
                    .map(|(&s, &o)| {
                        // scale: p_l * (o - s) + s  = s + p_l * residual
                        s + p_l * (o - s)
                    })
                    .collect();
            }
        }
        Ok(state)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. AdaptiveMixturLayer — per-sample expert mixture
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-sample expert mixture with learned top-2 routing.
pub struct AdaptiveMixturLayer {
    experts: Vec<AcMlpBlock>,
    router_w: Vec<f64>,
    router_b: Vec<f64>,
    dim: usize,
    n_experts: usize,
}

impl AdaptiveMixturLayer {
    /// Create a new AdaptiveMixturLayer with `n_experts` MlpBlock experts.
    pub fn new(dim: usize, hidden: usize, n_experts: usize, rng: &mut SmallRng) -> Self {
        let experts: Vec<AcMlpBlock> = (0..n_experts)
            .map(|_| AcMlpBlock::new(dim, hidden, rng))
            .collect();
        let router_w = xavier_init(n_experts, dim, rng);
        let router_b = vec![0.0; n_experts];
        AdaptiveMixturLayer {
            experts,
            router_w,
            router_b,
            dim,
            n_experts,
        }
    }

    /// Forward: weighted sum of top-2 expert outputs.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, AcError> {
        if x.len() != self.dim {
            return Err(AcError::InvalidInput(format!(
                "AdaptiveMixturLayer: expected dim={}, got {}",
                self.dim,
                x.len()
            )));
        }
        let mut logits = matvec(&self.router_w, x, self.n_experts, self.dim);
        for (li, &bi) in logits.iter_mut().zip(self.router_b.iter()) {
            *li += bi;
        }

        // Top-2 selection.
        let top2 = top_k_indices_f64(&logits, 2.min(self.n_experts));
        let top2_logits: Vec<f64> = top2.iter().map(|&i| logits[i]).collect();
        let weights = softmax_vec(&top2_logits);

        let mut output = vec![0.0_f64; self.dim];
        for (rank, &expert_idx) in top2.iter().enumerate() {
            let expert_out = self.experts[expert_idx].forward(x)?;
            for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
                *o += weights[rank] * e;
            }
        }
        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 13. PonderNetBlock — PonderNet (Banino et al. 2021)
// ═══════════════════════════════════════════════════════════════════════════════

/// PonderNet recurrent block.
///
/// At each step: `(h_new, y_n, lambda_n) = step(h, x)`.
/// Halting probability λ_n drawn geometrically; KL penalty against Geo(p_g) prior.
pub struct PonderNetBlock {
    /// GRU-style update parameters (simplified MLP).
    update_w: Vec<f64>,
    update_b: Vec<f64>,
    /// Output head.
    output_w: Vec<f64>,
    output_b: Vec<f64>,
    /// Halting head: linear → sigmoid.
    halt_w: Vec<f64>,
    halt_b: f64,
    /// Prior geometric parameter.
    prior_lambda: f64,
    hidden_dim: usize,
    input_dim: usize,
    output_dim: usize,
    rng: SmallRng,
}

impl PonderNetBlock {
    /// Create a new PonderNetBlock.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        prior_lambda: f64,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed + 9999);
        let combined = input_dim + hidden_dim;
        PonderNetBlock {
            update_w: xavier_init(hidden_dim, combined, &mut rng),
            update_b: vec![0.0; hidden_dim],
            output_w: xavier_init(output_dim, hidden_dim, &mut rng),
            output_b: vec![0.0; output_dim],
            halt_w: xavier_init(1, hidden_dim, &mut rng),
            halt_b: 0.0,
            prior_lambda,
            hidden_dim,
            input_dim,
            output_dim,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Single PonderNet step: returns `(h_new, y_n, lambda_n)`.
    pub fn step(&self, h: &[f64], x: &[f64]) -> Result<(Vec<f64>, Vec<f64>, f64), AcError> {
        if h.len() != self.hidden_dim {
            return Err(AcError::InvalidInput(format!(
                "PonderNetBlock::step: h expected {}, got {}",
                self.hidden_dim,
                h.len()
            )));
        }
        if x.len() != self.input_dim {
            return Err(AcError::InvalidInput(format!(
                "PonderNetBlock::step: x expected {}, got {}",
                self.input_dim,
                x.len()
            )));
        }

        // Concatenate [h, x].
        let mut combined = h.to_vec();
        combined.extend_from_slice(x);

        // Update hidden.
        let mut h_new = matvec(&self.update_w, &combined, self.hidden_dim, combined.len());
        add_bias(&mut h_new, &self.update_b);
        let h_new: Vec<f64> = h_new.iter().map(|&v| v.tanh()).collect();

        // Output.
        let mut y_n = matvec(&self.output_w, &h_new, self.output_dim, self.hidden_dim);
        add_bias(&mut y_n, &self.output_b);

        // Halting probability.
        let raw = dot(&self.halt_w, &h_new) + self.halt_b;
        let lambda_n = sigmoid(raw);

        Ok((h_new, y_n, lambda_n))
    }

    /// Full PonderNet forward: sample halt step stochastically.
    ///
    /// Returns `(output, kl_penalty, mean_steps)`.
    pub fn ponder_forward(
        &mut self,
        x: &[f64],
        max_steps: usize,
    ) -> Result<(Vec<f64>, f64, f64), AcError> {
        if x.len() != self.input_dim {
            return Err(AcError::InvalidInput(format!(
                "PonderNetBlock::ponder_forward: x expected {}, got {}",
                self.input_dim,
                x.len()
            )));
        }

        let mut h = vec![0.0_f64; self.hidden_dim];
        let mut lambdas: Vec<f64> = Vec::new();
        let mut outputs: Vec<Vec<f64>> = Vec::new();

        for _ in 0..max_steps {
            let (h_new, y_n, lambda_n) = self.step(&h, x)?;
            lambdas.push(lambda_n);
            outputs.push(y_n);
            h = h_new;
        }

        // Compute halting distribution p(halt at step n).
        let mut p_halt = Vec::new();
        let mut not_halted = 1.0_f64;
        for &lam in &lambdas {
            let p_n = not_halted * lam;
            p_halt.push(p_n);
            not_halted *= 1.0 - lam;
        }
        // Add remainder to last step.
        if let Some(last) = p_halt.last_mut() {
            *last += not_halted;
        }

        // Sample halt step from p_halt (geometric approximation).
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
        let mean_steps: f64 = p_halt
            .iter()
            .enumerate()
            .map(|(n, &p)| (n + 1) as f64 * p)
            .sum();

        // KL divergence: KL(p_halt || Geo(prior_lambda)).
        let lp = self.prior_lambda;
        let kl = p_halt
            .iter()
            .enumerate()
            .map(|(n, &p)| {
                if p < 1e-15 {
                    return 0.0;
                }
                // Geometric PMF: (1-lp)^n * lp
                let q = (1.0 - lp).powi(n as i32) * lp;
                let q = q.max(1e-15);
                p * (p / q).ln()
            })
            .sum::<f64>();

        Ok((output, kl, mean_steps))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 14. AcMetrics — evaluation metrics for adaptive computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Metrics for evaluating adaptive computation systems.
pub struct AcMetrics;

impl AcMetrics {
    /// Mean ponder cost over a batch of `(output, ponder_cost)` pairs.
    pub fn mean_ponder_cost(costs: &[f64]) -> f64 {
        if costs.is_empty() {
            return 0.0;
        }
        costs.iter().sum::<f64>() / costs.len() as f64
    }

    /// Histogram of exit layer indices (how many samples exited at each layer).
    pub fn early_exit_distribution(exit_layers: &[usize], n_exits: usize) -> Vec<usize> {
        let mut counts = vec![0_usize; n_exits];
        for &e in exit_layers {
            if e < n_exits {
                counts[e] += 1;
            }
        }
        counts
    }

    /// Routing entropy: mean entropy over per-token router probability vectors.
    ///
    /// `router_probs` — slice of probability vectors (each summing to ~1).
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

    /// FLOP reduction ratio: actual_flops / full_model_flops.
    ///
    /// `exits` — slice of exit indices per sample (0-indexed).
    /// `full_cost` — number of layers in the full model.
    pub fn flop_reduction_ratio(exits: &[usize], full_cost: usize) -> f64 {
        if exits.is_empty() || full_cost == 0 {
            return 1.0;
        }
        let actual: f64 = exits.iter().map(|&e| (e + 1) as f64).sum::<f64>();
        let full = full_cost as f64 * exits.len() as f64;
        actual / full
    }
}
