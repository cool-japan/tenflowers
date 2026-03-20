//! Sparse Mixture-of-Experts training algorithms — TenfloweRS.
//!
//! This module implements *training algorithms* for sparse MoE models:
//! load balancing, expert specialization analysis, routing optimisation,
//! capacity management, and MoE transformer blocks with auxiliary losses.
//!
//! All public types are prefixed with `Sme` to avoid collisions with the
//! existing `mixture_of_experts_advanced` module.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ── Internal math helpers ──────────────────────────────────────────────────────

fn softmax_sme(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let denom = if sum < f32::EPSILON { 1.0 } else { sum };
    exps.iter().map(|&e| e / denom).collect()
}

fn gelu_sme(x: f32) -> f32 {
    // Approximate GeLU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn softplus_sme(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn dot_sme(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec_sme(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| dot_sme(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

fn xavier_sme(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f32> {
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| rng.random::<f32>() * 2.0 * limit - limit)
        .collect()
}

/// Returns indices of top-K largest values (descending by value).
fn top_k_sme(values: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

fn mean_sme(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn var_sme(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean_sme(v);
    v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
}

fn std_sme(v: &[f32]) -> f32 {
    var_sme(v).sqrt()
}

/// Box-Muller transform for N(0,1) samples.
fn normal_sample_sme(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(f32::EPSILON);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn layer_norm_sme(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let m = mean_sme(x);
    let v = var_sme(x);
    let std = (v + 1e-5).sqrt();
    x.iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(&xi, (&g, &b))| g * (xi - m) / std + b)
        .collect()
}

fn cosine_sim_sme(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_sme(a, b);
    let na = dot_sme(a, a).sqrt();
    let nb = dot_sme(b, b).sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. SmeExpert — Expert FFN with GeLU activation and Xavier init
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for an expert FFN.
#[derive(Debug, Clone)]
pub struct SmeExpertConfig {
    /// Input / output dimension of the expert.
    pub input_dim: usize,
    /// Intermediate (hidden) dimension of the expert FFN.
    pub hidden_dim: usize,
}

/// Two-layer MLP expert with GeLU activation and Xavier-initialised weights.
///
/// Forward: x → W1·x + b1 → GeLU → W2·h + b2 → output.
#[derive(Debug, Clone)]
pub struct SmeExpert {
    /// W1: \[hidden_dim × input_dim\] row-major.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// W2: \[input_dim × hidden_dim\] row-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub input_dim: usize,
    pub hidden_dim: usize,
}

impl SmeExpert {
    /// Create a new expert with Xavier-initialised weights.
    pub fn new(cfg: &SmeExpertConfig, rng: &mut StdRng) -> Self {
        Self {
            w1: xavier_sme(cfg.hidden_dim, cfg.input_dim, rng),
            b1: vec![0.0; cfg.hidden_dim],
            w2: xavier_sme(cfg.input_dim, cfg.hidden_dim, rng),
            b2: vec![0.0; cfg.input_dim],
            input_dim: cfg.input_dim,
            hidden_dim: cfg.hidden_dim,
        }
    }

    /// Forward pass: x → GeLU(W1·x + b1) → W2·h + b2.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "SmeExpert::forward",
                &format!("input length {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        // hidden = GeLU(W1 x + b1)
        let mut h = matvec_sme(&self.w1, x, self.hidden_dim, self.input_dim);
        for (v, &bias) in h.iter_mut().zip(self.b1.iter()) {
            *v = gelu_sme(*v + bias);
        }
        // out = W2 h + b2
        let mut out = matvec_sme(&self.w2, &h, self.input_dim, self.hidden_dim);
        for (v, &bias) in out.iter_mut().zip(self.b2.iter()) {
            *v += bias;
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. SmeRouter — Top-K sparse routing with noise and straight-through estimator
// ═══════════════════════════════════════════════════════════════════════════════

/// Output produced by `SmeRouter::route`.
#[derive(Debug, Clone)]
pub struct SmeRoutingOutput {
    /// Indices of the chosen experts (length = top_k, may have fewer if n_experts < top_k).
    pub expert_indices: Vec<usize>,
    /// Normalised gate weights corresponding to `expert_indices`.
    pub expert_weights: Vec<f32>,
    /// Full softmax probability vector over all experts (before top-K masking).
    pub router_probs: Vec<f32>,
    /// Raw logits before softmax (useful for z-loss computation).
    pub logits: Vec<f32>,
}

/// Sparse top-K router with optional noise for training exploration.
///
/// Implements:
/// - Linear gating: g(x) = softmax(W_g · x)
/// - Noisy top-K: adds N(0, softplus(W_noise·x)²) noise before selection
/// - Straight-through gradient approximation for discrete routing
#[derive(Debug, Clone)]
pub struct SmeRouter {
    /// Gating weights W_g: \[n_experts × input_dim\].
    pub gate_weights: Vec<f32>,
    /// Noise projection W_noise: \[n_experts × input_dim\].
    pub noise_weights: Vec<f32>,
    pub n_experts: usize,
    pub input_dim: usize,
    /// Whether to add noise during routing (enable for training).
    pub noisy: bool,
}

impl SmeRouter {
    /// Create a new router.
    pub fn new(n_experts: usize, input_dim: usize, noisy: bool, rng: &mut StdRng) -> Self {
        Self {
            gate_weights: xavier_sme(n_experts, input_dim, rng),
            noise_weights: xavier_sme(n_experts, input_dim, rng),
            n_experts,
            input_dim,
            noisy,
        }
    }

    /// Route a single token `x` through the top-K experts.
    ///
    /// Returns `SmeRoutingOutput` containing expert indices, normalised weights,
    /// full router probabilities, and raw logits.
    pub fn route(
        &self,
        x: &[f32],
        top_k: usize,
        rng: &mut StdRng,
    ) -> Result<SmeRoutingOutput, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "SmeRouter::route",
                &format!("input length {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        let k = top_k.min(self.n_experts).max(1);

        // Compute raw logits: W_g · x
        let mut logits = matvec_sme(&self.gate_weights, x, self.n_experts, self.input_dim);

        // Add noise for training exploration
        if self.noisy {
            let noise_scale = matvec_sme(&self.noise_weights, x, self.n_experts, self.input_dim);
            for (l, ns) in logits.iter_mut().zip(noise_scale.iter()) {
                let sigma = softplus_sme(*ns);
                *l += sigma * normal_sample_sme(rng);
            }
        }

        // Full softmax probabilities
        let router_probs = softmax_sme(&logits);

        // Select top-K experts
        let expert_indices = top_k_sme(&router_probs, k);

        // Normalise selected weights (straight-through: gradient flows through probs)
        let selected: Vec<f32> = expert_indices.iter().map(|&i| router_probs[i]).collect();
        let weight_sum: f32 = selected.iter().sum();
        let expert_weights = if weight_sum < f32::EPSILON {
            vec![1.0 / k as f32; k]
        } else {
            selected.iter().map(|&w| w / weight_sum).collect()
        };

        Ok(SmeRoutingOutput {
            expert_indices,
            expert_weights,
            router_probs,
            logits,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. SmeAuxiliaryLoss — Load balancing losses (Switch, Z-loss, Importance CV)
// ═══════════════════════════════════════════════════════════════════════════════

/// Collection of MoE auxiliary losses for training.
#[derive(Debug, Clone, Default)]
pub struct SmeAuxLossValues {
    /// Switch Transformer auxiliary loss.
    pub switch_loss: f32,
    /// Z-loss (router logit regularisation).
    pub z_loss: f32,
    /// Expert importance coefficient-of-variation loss.
    pub importance_loss: f32,
    /// Weighted combined loss: λ_sw·switch + λ_z·z + λ_imp·importance.
    pub total: f32,
}

/// Configuration for auxiliary loss weights.
#[derive(Debug, Clone)]
pub struct SmeAuxLossConfig {
    /// Weight for Switch auxiliary loss.
    pub lambda_switch: f32,
    /// Weight for Z-loss.
    pub lambda_z: f32,
    /// Weight for importance CV² loss.
    pub lambda_importance: f32,
}

impl Default for SmeAuxLossConfig {
    fn default() -> Self {
        Self {
            lambda_switch: 1e-2,
            lambda_z: 1e-3,
            lambda_importance: 1e-3,
        }
    }
}

/// Computes MoE auxiliary losses from per-token routing outputs.
pub struct SmeAuxiliaryLoss {
    pub config: SmeAuxLossConfig,
    pub n_experts: usize,
}

impl SmeAuxiliaryLoss {
    pub fn new(n_experts: usize, config: SmeAuxLossConfig) -> Self {
        Self { config, n_experts }
    }

    /// Compute all auxiliary losses from a batch of routing outputs.
    ///
    /// `routing_outputs`: one `SmeRoutingOutput` per token in the batch.
    pub fn compute(
        &self,
        routing_outputs: &[SmeRoutingOutput],
    ) -> Result<SmeAuxLossValues, TensorError> {
        let n = routing_outputs.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SmeAuxiliaryLoss::compute",
                "routing_outputs must be non-empty",
            ));
        }
        let e = self.n_experts;

        // ── Switch Transformer loss ────────────────────────────────────────────
        // f_i = fraction of tokens dispatched to expert i
        // p_i = mean routing probability for expert i
        // L_switch = n_experts · Σ_i f_i · p_i
        let mut token_counts = vec![0usize; e];
        let mut prob_sums = vec![0.0f32; e];

        for ro in routing_outputs.iter() {
            if ro.router_probs.len() != e {
                return Err(TensorError::invalid_argument_op(
                    "SmeAuxiliaryLoss::compute",
                    &format!(
                        "router_probs length {} != n_experts {}",
                        ro.router_probs.len(),
                        e
                    ),
                ));
            }
            for &idx in ro.expert_indices.iter() {
                if idx < e {
                    token_counts[idx] += 1;
                }
            }
            for (i, &p) in ro.router_probs.iter().enumerate() {
                prob_sums[i] += p;
            }
        }

        let f: Vec<f32> = token_counts.iter().map(|&c| c as f32 / n as f32).collect();
        let p: Vec<f32> = prob_sums.iter().map(|&s| s / n as f32).collect();
        let switch_loss = e as f32
            * f.iter()
                .zip(p.iter())
                .map(|(&fi, &pi)| fi * pi)
                .sum::<f32>();

        // ── Z-loss ────────────────────────────────────────────────────────────
        // L_z = (1/B) · Σ_b (log Σ_e exp(logit_{b,e}))²
        let z_loss: f32 = routing_outputs
            .iter()
            .map(|ro| {
                let max = ro.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lse = max + ro.logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
                lse * lse
            })
            .sum::<f32>()
            / n as f32;

        // ── Expert importance CV² loss ─────────────────────────────────────────
        // importance_i = Σ_b g_i(x_b) (sum of router probs for expert i)
        // loss = CV(importance)² = (std / mean)²
        let importance = prob_sums.clone(); // already summed over tokens
        let imp_mean = mean_sme(&importance);
        let imp_std = std_sme(&importance);
        let importance_loss = if imp_mean < f32::EPSILON {
            0.0
        } else {
            (imp_std / imp_mean).powi(2)
        };

        let total = self.config.lambda_switch * switch_loss
            + self.config.lambda_z * z_loss
            + self.config.lambda_importance * importance_loss;

        Ok(SmeAuxLossValues {
            switch_loss,
            z_loss,
            importance_loss,
            total,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. SmeExpertCapacity — Capacity factor management and token dispatch/combine
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-expert dispatch result.
#[derive(Debug, Clone)]
pub struct SmeDispatchResult {
    /// Inputs grouped per expert. `expert_inputs[i]` = tokens sent to expert i.
    /// Each inner `Vec<f32>` is a flattened \[token_count × token_dim\] matrix.
    pub expert_inputs: Vec<Vec<f32>>,
    /// Token indices sent to each expert. `expert_token_ids[i][j]` = original token index.
    pub expert_token_ids: Vec<Vec<usize>>,
    /// Weight applied to each expert's output for each assigned token.
    pub expert_token_weights: Vec<Vec<f32>>,
    /// Boolean mask: `overflow[i]` = true if token i was dropped due to capacity overflow.
    pub overflow: Vec<bool>,
}

/// Manages per-expert token capacity limits during forward pass.
pub struct SmeExpertCapacity {
    pub n_experts: usize,
    pub token_dim: usize,
}

impl SmeExpertCapacity {
    pub fn new(n_experts: usize, token_dim: usize) -> Self {
        Self {
            n_experts,
            token_dim,
        }
    }

    /// Compute integer capacity per expert.
    ///
    /// `n_tokens`: total tokens in batch.
    /// `capacity_factor`: multiplier over ideal even load.
    pub fn capacity_per_expert(&self, n_tokens: usize, capacity_factor: f32) -> usize {
        let ideal = n_tokens as f32 / self.n_experts as f32;
        ((ideal * capacity_factor).ceil() as usize).max(1)
    }

    /// Dispatch tokens to experts respecting per-expert capacity limits.
    ///
    /// `inputs`: \[n_tokens × token_dim\] flattened.
    /// `routing_outputs`: one per token (must have same length as `inputs / token_dim`).
    /// `capacity_factor`: capacity multiplier.
    pub fn dispatch(
        &self,
        inputs: &[f32],
        routing_outputs: &[SmeRoutingOutput],
        capacity_factor: f32,
    ) -> Result<SmeDispatchResult, TensorError> {
        let td = self.token_dim;
        let n_tokens = routing_outputs.len();

        if inputs.len() != n_tokens * td {
            return Err(TensorError::invalid_argument_op(
                "SmeExpertCapacity::dispatch",
                &format!(
                    "inputs length {} != n_tokens {} * token_dim {}",
                    inputs.len(),
                    n_tokens,
                    td
                ),
            ));
        }

        let cap = self.capacity_per_expert(n_tokens, capacity_factor);
        let mut expert_inputs = vec![Vec::new(); self.n_experts];
        let mut expert_token_ids = vec![Vec::new(); self.n_experts];
        let mut expert_token_weights = vec![Vec::new(); self.n_experts];
        let mut overflow = vec![false; n_tokens];

        for (tok_idx, ro) in routing_outputs.iter().enumerate() {
            for (&exp_idx, &weight) in ro.expert_indices.iter().zip(ro.expert_weights.iter()) {
                if exp_idx >= self.n_experts {
                    continue;
                }
                if expert_token_ids[exp_idx].len() < cap {
                    let start = tok_idx * td;
                    expert_inputs[exp_idx].extend_from_slice(&inputs[start..start + td]);
                    expert_token_ids[exp_idx].push(tok_idx);
                    expert_token_weights[exp_idx].push(weight);
                } else {
                    overflow[tok_idx] = true;
                }
            }
        }

        Ok(SmeDispatchResult {
            expert_inputs,
            expert_token_ids,
            expert_token_weights,
            overflow,
        })
    }

    /// Combine expert outputs back into a single output tensor.
    ///
    /// `expert_outputs`: \[expert_i: (token_j outputs)\] corresponding to dispatch result.
    /// `dispatch`: the dispatch result from `dispatch()`.
    /// Returns flattened \[n_tokens × token_dim\].
    pub fn combine(
        &self,
        expert_outputs: &[Vec<f32>],
        dispatch: &SmeDispatchResult,
        n_tokens: usize,
    ) -> Result<Vec<f32>, TensorError> {
        let td = self.token_dim;
        if expert_outputs.len() != self.n_experts {
            return Err(TensorError::invalid_argument_op(
                "SmeExpertCapacity::combine",
                &format!(
                    "expert_outputs length {} != n_experts {}",
                    expert_outputs.len(),
                    self.n_experts
                ),
            ));
        }

        let mut output = vec![0.0f32; n_tokens * td];

        for (exp_idx, exp_out) in expert_outputs.iter().enumerate() {
            let n_dispatched = dispatch.expert_token_ids[exp_idx].len();
            for j in 0..n_dispatched {
                let tok_idx = dispatch.expert_token_ids[exp_idx][j];
                let weight = dispatch.expert_token_weights[exp_idx][j];
                let out_start = j * td;
                let dst_start = tok_idx * td;

                if out_start + td > exp_out.len() {
                    return Err(TensorError::invalid_argument_op(
                        "SmeExpertCapacity::combine",
                        &format!(
                            "expert {} output too short: got {}, need {} for token slot {}",
                            exp_idx,
                            exp_out.len(),
                            out_start + td,
                            j
                        ),
                    ));
                }

                for d in 0..td {
                    output[dst_start + d] += weight * exp_out[out_start + d];
                }
            }
        }

        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. SmeExpertParallelism — Batched expert processing
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates expert parallelism by batching all tokens for each expert together.
pub struct SmeExpertParallelism {
    pub token_dim: usize,
}

impl SmeExpertParallelism {
    pub fn new(token_dim: usize) -> Self {
        Self { token_dim }
    }

    /// Group token inputs by their primary expert assignment.
    ///
    /// `inputs`: \[n_tokens × token_dim\] flattened.
    /// `expert_assignments`: primary expert index per token (length = n_tokens).
    /// Returns a Vec of (expert_index, token_indices, batched_inputs) tuples.
    pub fn batch_by_expert(
        &self,
        inputs: &[f32],
        expert_assignments: &[usize],
        n_experts: usize,
    ) -> Result<Vec<(usize, Vec<usize>, Vec<f32>)>, TensorError> {
        let td = self.token_dim;
        let n_tokens = expert_assignments.len();

        if inputs.len() != n_tokens * td {
            return Err(TensorError::invalid_argument_op(
                "SmeExpertParallelism::batch_by_expert",
                &format!(
                    "inputs length {} != n_tokens {} * token_dim {}",
                    inputs.len(),
                    n_tokens,
                    td
                ),
            ));
        }

        // Group token indices by expert
        let mut expert_token_map: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
        for (tok_idx, &exp_idx) in expert_assignments.iter().enumerate() {
            if exp_idx < n_experts {
                expert_token_map[exp_idx].push(tok_idx);
            }
        }

        let mut result = Vec::with_capacity(n_experts);
        for (exp_idx, token_ids) in expert_token_map.into_iter().enumerate() {
            if token_ids.is_empty() {
                continue;
            }
            let mut batched = Vec::with_capacity(token_ids.len() * td);
            for &tok_idx in token_ids.iter() {
                let start = tok_idx * td;
                batched.extend_from_slice(&inputs[start..start + td]);
            }
            result.push((exp_idx, token_ids, batched));
        }

        Ok(result)
    }

    /// Process each expert batch through the provided experts and scatter back.
    ///
    /// `experts_batches`: output of `batch_by_expert`.
    /// `experts`: slice of `SmeExpert` (one per expert index 0..n_experts).
    /// `n_tokens`: total number of tokens.
    /// Returns flattened \[n_tokens × token_dim\] output (tokens not processed remain zero).
    pub fn process_and_scatter(
        &self,
        expert_batches: &[(usize, Vec<usize>, Vec<f32>)],
        experts: &[SmeExpert],
        n_tokens: usize,
    ) -> Result<Vec<f32>, TensorError> {
        let td = self.token_dim;
        let mut output = vec![0.0f32; n_tokens * td];

        for (exp_idx, token_ids, batched_input) in expert_batches.iter() {
            let expert = experts.get(*exp_idx).ok_or_else(|| {
                TensorError::invalid_argument_op(
                    "SmeExpertParallelism::process_and_scatter",
                    &format!("expert index {} out of range", exp_idx),
                )
            })?;

            for (j, &tok_idx) in token_ids.iter().enumerate() {
                let in_start = j * td;
                let token_input = &batched_input[in_start..in_start + td];
                let expert_out = expert.forward(token_input)?;

                let out_start = tok_idx * td;
                let slice = output.get_mut(out_start..out_start + td).ok_or_else(|| {
                    TensorError::invalid_argument_op(
                        "SmeExpertParallelism::process_and_scatter",
                        &format!("token index {} out of n_tokens {}", tok_idx, n_tokens),
                    )
                })?;

                for (o, e) in slice.iter_mut().zip(expert_out.iter()) {
                    *o = *e;
                }
            }
        }

        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. SmeTokenChoiceRouter / SmeExpertChoiceRouter
// ═══════════════════════════════════════════════════════════════════════════════

/// Standard token-choice routing: each token independently selects top-K experts.
#[derive(Debug, Clone)]
pub struct SmeTokenChoiceRouter {
    pub router: SmeRouter,
    pub top_k: usize,
}

impl SmeTokenChoiceRouter {
    pub fn new(n_experts: usize, input_dim: usize, top_k: usize, rng: &mut StdRng) -> Self {
        Self {
            router: SmeRouter::new(n_experts, input_dim, true, rng),
            top_k,
        }
    }

    /// Route a batch of tokens.
    ///
    /// `inputs`: \[n_tokens × input_dim\] flattened.
    /// Returns `Vec<SmeRoutingOutput>`, one per token.
    pub fn route_batch(
        &self,
        inputs: &[f32],
        rng: &mut StdRng,
    ) -> Result<Vec<SmeRoutingOutput>, TensorError> {
        let dim = self.router.input_dim;
        let n_tokens = inputs.len() / dim.max(1);

        if inputs.len() != n_tokens * dim {
            return Err(TensorError::invalid_argument_op(
                "SmeTokenChoiceRouter::route_batch",
                &format!(
                    "inputs length {} not divisible by input_dim {}",
                    inputs.len(),
                    dim
                ),
            ));
        }

        let mut outputs = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let token = &inputs[i * dim..(i + 1) * dim];
            outputs.push(self.router.route(token, self.top_k, rng)?);
        }
        Ok(outputs)
    }
}

/// Expert-choice routing (Zhou et al., 2022): each expert selects top-C tokens.
///
/// Guarantees full expert utilisation regardless of input distribution.
#[derive(Debug, Clone)]
pub struct SmeExpertChoiceRouter {
    /// Gating weights: \[n_experts × input_dim\].
    pub gate_weights: Vec<f32>,
    pub n_experts: usize,
    pub input_dim: usize,
}

impl SmeExpertChoiceRouter {
    pub fn new(n_experts: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            gate_weights: xavier_sme(n_experts, input_dim, rng),
            n_experts,
            input_dim,
        }
    }

    /// Expert-choice routing: each expert picks its top-C tokens.
    ///
    /// `inputs`: \[n_tokens × input_dim\] flattened.
    /// `capacity`: number of tokens each expert selects.
    ///
    /// Returns `assignments[expert_i]` = sorted list of chosen token indices.
    pub fn expert_choice_route(
        &self,
        inputs: &[f32],
        capacity: usize,
    ) -> Result<Vec<Vec<usize>>, TensorError> {
        let dim = self.input_dim;
        let n_tokens = inputs.len() / dim.max(1);

        if inputs.len() != n_tokens * dim {
            return Err(TensorError::invalid_argument_op(
                "SmeExpertChoiceRouter::expert_choice_route",
                &format!(
                    "inputs length {} not divisible by input_dim {}",
                    inputs.len(),
                    dim
                ),
            ));
        }

        let cap = capacity.min(n_tokens);
        let e = self.n_experts;

        // Score matrix S[expert][token] = W_g[expert] · x[token]
        let mut scores: Vec<Vec<f32>> = (0..e)
            .map(|ex| {
                (0..n_tokens)
                    .map(|tok| {
                        let w_row = &self.gate_weights[ex * dim..(ex + 1) * dim];
                        let x_tok = &inputs[tok * dim..(tok + 1) * dim];
                        dot_sme(w_row, x_tok)
                    })
                    .collect()
            })
            .collect();

        // Each expert selects top-C tokens by score
        let assignments: Vec<Vec<usize>> = scores
            .iter_mut()
            .map(|expert_scores| {
                let mut indexed: Vec<(usize, f32)> =
                    expert_scores.iter().cloned().enumerate().collect();
                indexed.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut chosen: Vec<usize> = indexed.iter().take(cap).map(|(i, _)| *i).collect();
                chosen.sort_unstable();
                chosen
            })
            .collect();

        Ok(assignments)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. SmeSpecializationMetrics — Expert specialisation analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Analysis results for expert specialisation.
#[derive(Debug, Clone)]
pub struct SmeSpecializationAnalysis {
    /// Fraction of tokens routed to each expert (sums to approximately 1 over top-K routing).
    pub expert_utilization: Vec<f32>,
    /// Shannon entropy of the utilisation distribution (in nats).
    /// Lower entropy → higher specialisation.
    pub routing_entropy: f32,
    /// Pairwise cosine similarity matrix between expert W1 weight rows (expert × expert).
    /// Lower values → higher expert diversity.
    pub expert_overlap: Vec<Vec<f32>>,
    /// Number of experts with utilisation < `collapse_threshold`.
    pub collapsed_experts: usize,
    /// Load imbalance ratio: max_utilisation / mean_utilisation.
    pub load_imbalance: f32,
}

/// Computes expert specialisation metrics from routing data and expert weights.
pub struct SmeSpecializationMetrics {
    pub n_experts: usize,
    /// Experts with utilisation below this fraction are considered "collapsed".
    pub collapse_threshold: f32,
}

impl SmeSpecializationMetrics {
    pub fn new(n_experts: usize, collapse_threshold: f32) -> Self {
        Self {
            n_experts,
            collapse_threshold: collapse_threshold.max(f32::EPSILON),
        }
    }

    /// Analyse specialisation from a batch of routing outputs and expert networks.
    pub fn analyse(
        &self,
        routing_outputs: &[SmeRoutingOutput],
        experts: &[SmeExpert],
    ) -> Result<SmeSpecializationAnalysis, TensorError> {
        let n = routing_outputs.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SmeSpecializationMetrics::analyse",
                "routing_outputs must be non-empty",
            ));
        }

        let e = self.n_experts;

        // Expert utilisation: token fractions
        let mut token_counts = vec![0usize; e];
        let mut total_assignments = 0usize;
        for ro in routing_outputs.iter() {
            for &idx in ro.expert_indices.iter() {
                if idx < e {
                    token_counts[idx] += 1;
                    total_assignments += 1;
                }
            }
        }

        let denom = total_assignments.max(1) as f32;
        let expert_utilization: Vec<f32> = token_counts.iter().map(|&c| c as f32 / denom).collect();

        // Routing entropy
        let routing_entropy: f32 = expert_utilization
            .iter()
            .map(|&p| if p > f32::EPSILON { -p * p.ln() } else { 0.0 })
            .sum();

        // Expert overlap: pairwise cosine similarity of W1 rows (first row only for efficiency)
        let expert_overlap: Vec<Vec<f32>> = if experts.len() >= e {
            (0..e)
                .map(|i| {
                    let w_i = experts[i]
                        .w1
                        .chunks(experts[i].input_dim)
                        .next()
                        .unwrap_or(&[]);
                    (0..e)
                        .map(|j| {
                            let w_j = experts[j]
                                .w1
                                .chunks(experts[j].input_dim)
                                .next()
                                .unwrap_or(&[]);
                            cosine_sim_sme(w_i, w_j)
                        })
                        .collect()
                })
                .collect()
        } else {
            vec![vec![0.0; e]; e]
        };

        // Collapsed experts
        let collapsed_experts = expert_utilization
            .iter()
            .filter(|&&u| u < self.collapse_threshold)
            .count();

        // Load imbalance
        let max_util = expert_utilization.iter().cloned().fold(0.0_f32, f32::max);
        let mean_util = mean_sme(&expert_utilization);
        let load_imbalance = if mean_util < f32::EPSILON {
            1.0
        } else {
            max_util / mean_util
        };

        Ok(SmeSpecializationAnalysis {
            expert_utilization,
            routing_entropy,
            expert_overlap,
            collapsed_experts,
            load_imbalance,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. SmeMoeTransformerBlock — Full MoE transformer block with training support
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a MoE transformer block.
#[derive(Debug, Clone)]
pub struct SmeMoeBlockConfig {
    pub model_dim: usize,
    pub n_heads: usize,
    pub n_experts: usize,
    pub expert_hidden_dim: usize,
    pub top_k: usize,
    pub capacity_factor: f32,
    pub dropout_rate: f32,
    pub aux_loss_config: SmeAuxLossConfig,
}

impl Default for SmeMoeBlockConfig {
    fn default() -> Self {
        Self {
            model_dim: 64,
            n_heads: 4,
            n_experts: 8,
            expert_hidden_dim: 128,
            top_k: 2,
            capacity_factor: 1.25,
            dropout_rate: 0.1,
            aux_loss_config: SmeAuxLossConfig::default(),
        }
    }
}

/// Output of a forward pass through `SmeMoeTransformerBlock`.
#[derive(Debug, Clone)]
pub struct SmeMoeBlockOutput {
    /// Transformed token representations: \[n_tokens × model_dim\] flattened.
    pub output: Vec<f32>,
    /// Auxiliary loss computed over this forward pass.
    pub aux_loss: SmeAuxLossValues,
}

/// MoE transformer block: self-attention → LayerNorm → sparse MoE FFN → LayerNorm.
///
/// Self-attention is simplified (single-head dot-product) to keep the implementation
/// pure-Rust without external tensor backends.
pub struct SmeMoeTransformerBlock {
    /// Self-attention Q/K/V projections: each \[model_dim × model_dim\].
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    /// Layer norm 1 parameters.
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    /// Layer norm 2 parameters.
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    /// MoE FFN experts.
    pub experts: Vec<SmeExpert>,
    pub router: SmeRouter,
    pub aux_loss: SmeAuxiliaryLoss,
    pub config: SmeMoeBlockConfig,
}

impl SmeMoeTransformerBlock {
    pub fn new(config: SmeMoeBlockConfig, rng: &mut StdRng) -> Self {
        let d = config.model_dim;
        let expert_cfg = SmeExpertConfig {
            input_dim: d,
            hidden_dim: config.expert_hidden_dim,
        };
        let experts: Vec<SmeExpert> = (0..config.n_experts)
            .map(|_| SmeExpert::new(&expert_cfg, rng))
            .collect();

        let router = SmeRouter::new(config.n_experts, d, true, rng);

        let aux_loss = SmeAuxiliaryLoss::new(config.n_experts, config.aux_loss_config.clone());

        Self {
            w_q: xavier_sme(d, d, rng),
            w_k: xavier_sme(d, d, rng),
            w_v: xavier_sme(d, d, rng),
            w_o: xavier_sme(d, d, rng),
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
            experts,
            router,
            aux_loss,
            config,
        }
    }

    /// Forward pass through the MoE transformer block.
    ///
    /// `x`: \[n_tokens × model_dim\] flattened.
    /// `training`: if true, apply dropout and use noisy routing.
    pub fn forward(
        &self,
        x: &[f32],
        training: bool,
        rng: &mut StdRng,
    ) -> Result<SmeMoeBlockOutput, TensorError> {
        let d = self.config.model_dim;
        let n_tokens = x.len() / d.max(1);

        if x.len() != n_tokens * d {
            return Err(TensorError::invalid_argument_op(
                "SmeMoeTransformerBlock::forward",
                &format!("x length {} not divisible by model_dim {}", x.len(), d),
            ));
        }

        // ── Simplified multi-head self-attention (shared Q/K/V per token) ─────
        let scale = (d as f32).sqrt();
        let mut attn_out = vec![0.0f32; n_tokens * d];

        for i in 0..n_tokens {
            let xi = &x[i * d..(i + 1) * d];
            let qi = matvec_sme(&self.w_q, xi, d, d);

            // Compute attention scores over all tokens
            let mut scores: Vec<f32> = (0..n_tokens)
                .map(|j| {
                    let xj = &x[j * d..(j + 1) * d];
                    let kj = matvec_sme(&self.w_k, xj, d, d);
                    dot_sme(&qi, &kj) / scale
                })
                .collect();

            let attn_weights = softmax_sme(&scores);

            // Weighted sum of values
            let mut vi_sum = vec![0.0f32; d];
            for (j, &aw) in attn_weights.iter().enumerate() {
                let xj = &x[j * d..(j + 1) * d];
                let vj = matvec_sme(&self.w_v, xj, d, d);
                for (s, v) in vi_sum.iter_mut().zip(vj.iter()) {
                    *s += aw * v;
                }
            }

            let proj = matvec_sme(&self.w_o, &vi_sum, d, d);
            for (k, &p) in proj.iter().enumerate() {
                attn_out[i * d + k] = p;
            }
            // Clear scores to avoid confusion (already consumed)
            scores.clear();
        }

        // Residual + LayerNorm 1
        let mut h1 = vec![0.0f32; n_tokens * d];
        for i in 0..n_tokens {
            let xi = &x[i * d..(i + 1) * d];
            let ai = &attn_out[i * d..(i + 1) * d];
            let res: Vec<f32> = xi.iter().zip(ai.iter()).map(|(&a, &b)| a + b).collect();
            let normed = layer_norm_sme(&res, &self.ln1_gamma, &self.ln1_beta);
            h1[i * d..(i + 1) * d].copy_from_slice(&normed);
        }

        // ── Sparse MoE FFN ─────────────────────────────────────────────────────
        let mut routing_outputs = Vec::with_capacity(n_tokens);
        let mut moe_out = vec![0.0f32; n_tokens * d];

        for i in 0..n_tokens {
            let token = &h1[i * d..(i + 1) * d];
            let ro = self.router.route(token, self.config.top_k, rng)?;

            // Accumulate weighted expert outputs
            for (&exp_idx, &weight) in ro.expert_indices.iter().zip(ro.expert_weights.iter()) {
                if exp_idx < self.experts.len() {
                    let expert_out = self.experts[exp_idx].forward(token)?;
                    for (k, &v) in expert_out.iter().enumerate() {
                        let mut val = weight * v;
                        // Inverted dropout during training
                        if training && self.config.dropout_rate > 0.0 {
                            let keep_prob = 1.0 - self.config.dropout_rate;
                            if rng.random::<f32>() > keep_prob {
                                val = 0.0;
                            } else {
                                val /= keep_prob;
                            }
                        }
                        moe_out[i * d + k] += val;
                    }
                }
            }

            routing_outputs.push(ro);
        }

        // Auxiliary loss
        let aux_loss = self.aux_loss.compute(&routing_outputs)?;

        // Residual + LayerNorm 2
        let mut output = vec![0.0f32; n_tokens * d];
        for i in 0..n_tokens {
            let h = &h1[i * d..(i + 1) * d];
            let m = &moe_out[i * d..(i + 1) * d];
            let res: Vec<f32> = h.iter().zip(m.iter()).map(|(&a, &b)| a + b).collect();
            let normed = layer_norm_sme(&res, &self.ln2_gamma, &self.ln2_beta);
            output[i * d..(i + 1) * d].copy_from_slice(&normed);
        }

        Ok(SmeMoeBlockOutput { output, aux_loss })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. SmeAdaptiveRouter — Online router adaptation with bias re-balancing
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive router that tracks per-expert usage and adjusts routing bias.
///
/// Implements:
/// - Exponential moving average of per-expert token counts.
/// - Adaptive temperature: higher temperature for underused experts.
/// - Periodic routing bias re-balancing to counteract load imbalance.
#[derive(Debug, Clone)]
pub struct SmeAdaptiveRouter {
    pub base_router: SmeRouter,
    /// Per-expert routing bias added to logits before softmax.
    pub routing_bias: Vec<f32>,
    /// EMA of per-expert usage fractions (updated during `adapt_routing_bias`).
    pub usage_ema: Vec<f32>,
    /// EMA decay factor.
    pub ema_decay: f32,
    /// Target uniform fraction (1/n_experts).
    pub target_fraction: f32,
    /// Bias update learning rate.
    pub bias_lr: f32,
    pub n_experts: usize,
    pub top_k: usize,
}

impl SmeAdaptiveRouter {
    pub fn new(
        n_experts: usize,
        input_dim: usize,
        top_k: usize,
        ema_decay: f32,
        bias_lr: f32,
        rng: &mut StdRng,
    ) -> Self {
        let target = 1.0 / n_experts as f32;
        Self {
            base_router: SmeRouter::new(n_experts, input_dim, true, rng),
            routing_bias: vec![0.0; n_experts],
            usage_ema: vec![target; n_experts],
            ema_decay: ema_decay.clamp(0.0, 1.0),
            target_fraction: target,
            bias_lr: bias_lr.max(f32::EPSILON),
            n_experts,
            top_k,
        }
    }

    /// Route a single token using base router logits + routing bias.
    pub fn route(&self, x: &[f32], rng: &mut StdRng) -> Result<SmeRoutingOutput, TensorError> {
        let base_out = self.base_router.route(x, self.top_k, rng)?;

        // Apply routing bias to logits and re-compute softmax
        let biased_logits: Vec<f32> = base_out
            .logits
            .iter()
            .zip(self.routing_bias.iter())
            .map(|(&l, &b)| l + b)
            .collect();

        let biased_probs = softmax_sme(&biased_logits);
        let expert_indices = top_k_sme(&biased_probs, self.top_k);
        let selected: Vec<f32> = expert_indices.iter().map(|&i| biased_probs[i]).collect();
        let weight_sum: f32 = selected.iter().sum();
        let expert_weights = if weight_sum < f32::EPSILON {
            vec![1.0 / self.top_k as f32; self.top_k]
        } else {
            selected.iter().map(|&w| w / weight_sum).collect()
        };

        Ok(SmeRoutingOutput {
            expert_indices,
            expert_weights,
            router_probs: biased_probs,
            logits: biased_logits,
        })
    }

    /// Update routing bias based on observed usage history.
    ///
    /// `usage_history`: observed per-expert usage fractions for the current step.
    /// Updates EMA and adjusts bias to push underused experts higher.
    pub fn adapt_routing_bias(&mut self, usage_history: &[f32]) -> Result<(), TensorError> {
        if usage_history.len() != self.n_experts {
            return Err(TensorError::invalid_argument_op(
                "SmeAdaptiveRouter::adapt_routing_bias",
                &format!(
                    "usage_history length {} != n_experts {}",
                    usage_history.len(),
                    self.n_experts
                ),
            ));
        }

        // Update EMA
        for (ema, &obs) in self.usage_ema.iter_mut().zip(usage_history.iter()) {
            *ema = self.ema_decay * (*ema) + (1.0 - self.ema_decay) * obs;
        }

        // Update bias: increase bias for underused, decrease for overused
        for (bias, &ema_val) in self.routing_bias.iter_mut().zip(self.usage_ema.iter()) {
            let deficit = self.target_fraction - ema_val;
            *bias += self.bias_lr * deficit;
        }

        Ok(())
    }

    /// Reset routing bias to zero (e.g., at the start of a new epoch).
    pub fn reset_bias(&mut self) {
        for b in self.routing_bias.iter_mut() {
            *b = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. SmeMetrics & SmeReport — MoE training monitoring
// ═══════════════════════════════════════════════════════════════════════════════

/// Snapshot of MoE training metrics at a given training step.
#[derive(Debug, Clone, Default)]
pub struct SmeMetricsSnapshot {
    pub step: usize,
    /// Load imbalance factor: max_load / mean_load (1.0 = perfectly balanced).
    pub load_imbalance: f32,
    /// Switch auxiliary loss value.
    pub switch_loss: f32,
    /// Z-loss value.
    pub z_loss: f32,
    /// Importance CV² loss.
    pub importance_loss: f32,
    /// Number of collapsed experts at this step.
    pub collapsed_experts: usize,
    /// Routing entropy (nats).
    pub routing_entropy: f32,
    /// Per-expert utilisation fractions.
    pub expert_utilization: Vec<f32>,
}

/// Accumulates MoE training metrics over training steps.
pub struct SmeMetrics {
    pub n_experts: usize,
    pub collapse_threshold: f32,
    history: Vec<SmeMetricsSnapshot>,
}

impl SmeMetrics {
    pub fn new(n_experts: usize, collapse_threshold: f32) -> Self {
        Self {
            n_experts,
            collapse_threshold,
            history: Vec::new(),
        }
    }

    /// Record metrics from a batch of routing outputs.
    pub fn record(
        &mut self,
        step: usize,
        routing_outputs: &[SmeRoutingOutput],
        aux_loss: &SmeAuxLossValues,
    ) -> Result<SmeMetricsSnapshot, TensorError> {
        let n = routing_outputs.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SmeMetrics::record",
                "routing_outputs must be non-empty",
            ));
        }

        let e = self.n_experts;
        let mut token_counts = vec![0usize; e];
        let mut total = 0usize;

        for ro in routing_outputs.iter() {
            for &idx in ro.expert_indices.iter() {
                if idx < e {
                    token_counts[idx] += 1;
                    total += 1;
                }
            }
        }

        let denom = total.max(1) as f32;
        let expert_utilization: Vec<f32> = token_counts.iter().map(|&c| c as f32 / denom).collect();

        let max_load = expert_utilization.iter().cloned().fold(0.0_f32, f32::max);
        let mean_load = mean_sme(&expert_utilization);
        let load_imbalance = if mean_load < f32::EPSILON {
            1.0
        } else {
            max_load / mean_load
        };

        let collapsed_experts = expert_utilization
            .iter()
            .filter(|&&u| u < self.collapse_threshold)
            .count();

        let routing_entropy: f32 = expert_utilization
            .iter()
            .map(|&p| if p > f32::EPSILON { -p * p.ln() } else { 0.0 })
            .sum();

        let snap = SmeMetricsSnapshot {
            step,
            load_imbalance,
            switch_loss: aux_loss.switch_loss,
            z_loss: aux_loss.z_loss,
            importance_loss: aux_loss.importance_loss,
            collapsed_experts,
            routing_entropy,
            expert_utilization,
        };

        self.history.push(snap.clone());
        Ok(snap)
    }

    /// Detect if any expert is currently collapsed (< `collapse_threshold` utilisation).
    pub fn detect_collapse(&self) -> bool {
        if let Some(last) = self.history.last() {
            last.collapsed_experts > 0
        } else {
            false
        }
    }

    /// Return the average load imbalance over recorded history.
    pub fn mean_load_imbalance(&self) -> f32 {
        if self.history.is_empty() {
            return 1.0;
        }
        let vals: Vec<f32> = self.history.iter().map(|s| s.load_imbalance).collect();
        mean_sme(&vals)
    }

    /// Return the average routing entropy over recorded history.
    pub fn mean_routing_entropy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let vals: Vec<f32> = self.history.iter().map(|s| s.routing_entropy).collect();
        mean_sme(&vals)
    }

    /// Full history of recorded snapshots.
    pub fn history(&self) -> &[SmeMetricsSnapshot] {
        &self.history
    }
}

/// Summary report for a completed MoE training run.
#[derive(Debug, Clone)]
pub struct SmeReport {
    pub n_experts: usize,
    pub n_steps_recorded: usize,
    pub mean_load_imbalance: f32,
    pub final_load_imbalance: f32,
    pub mean_routing_entropy: f32,
    pub final_routing_entropy: f32,
    pub total_collapses_detected: usize,
    pub mean_switch_loss: f32,
    pub mean_z_loss: f32,
    pub final_expert_utilization: Vec<f32>,
}

impl SmeMetrics {
    /// Generate a summary report from accumulated history.
    pub fn report(&self) -> SmeReport {
        let n = self.history.len();
        if n == 0 {
            return SmeReport {
                n_experts: self.n_experts,
                n_steps_recorded: 0,
                mean_load_imbalance: 1.0,
                final_load_imbalance: 1.0,
                mean_routing_entropy: 0.0,
                final_routing_entropy: 0.0,
                total_collapses_detected: 0,
                mean_switch_loss: 0.0,
                mean_z_loss: 0.0,
                final_expert_utilization: vec![1.0 / self.n_experts as f32; self.n_experts],
            };
        }

        let mean_load_imbalance = self.mean_load_imbalance();
        let final_load_imbalance = self.history.last().map(|s| s.load_imbalance).unwrap_or(1.0);
        let mean_routing_entropy = self.mean_routing_entropy();
        let final_routing_entropy = self
            .history
            .last()
            .map(|s| s.routing_entropy)
            .unwrap_or(0.0);
        let total_collapses_detected = self.history.iter().map(|s| s.collapsed_experts).sum();
        let mean_switch_loss = {
            let v: Vec<f32> = self.history.iter().map(|s| s.switch_loss).collect();
            mean_sme(&v)
        };
        let mean_z_loss = {
            let v: Vec<f32> = self.history.iter().map(|s| s.z_loss).collect();
            mean_sme(&v)
        };
        let final_expert_utilization = self
            .history
            .last()
            .map(|s| s.expert_utilization.clone())
            .unwrap_or_else(|| vec![1.0 / self.n_experts as f32; self.n_experts]);

        SmeReport {
            n_experts: self.n_experts,
            n_steps_recorded: n,
            mean_load_imbalance,
            final_load_imbalance,
            mean_routing_entropy,
            final_routing_entropy,
            total_collapses_detected,
            mean_switch_loss,
            mean_z_loss,
            final_expert_utilization,
        }
    }
}

#[cfg(test)]
mod tests;
