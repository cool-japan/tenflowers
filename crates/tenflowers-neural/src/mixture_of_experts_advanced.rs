//! Advanced Mixture-of-Experts architectures — TenfloweRS.
//!
//! Implements sparse MoE, Mixture-of-Depths, MegaBlock MoE, hierarchical MoE,
//! load balancing utilities, expert merging (Fisher/DARE/TIES), specialist routing,
//! per-expert LoRA adapters, profiling, and conditional-compute early exit.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ── Math helpers ───────────────────────────────────────────────────────────────

fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let d = if sum < f32::EPSILON { 1.0 } else { sum };
    exps.iter().map(|&e| e / d).collect()
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec_f32(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| dot_f32(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn xavier_f32(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f32> {
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| rng.random::<f32>() * 2.0 * limit - limit)
        .collect()
}

fn entropy_f32(probs: &[f32]) -> f32 {
    probs
        .iter()
        .map(|&p| if p > f32::EPSILON { -p * p.ln() } else { 0.0 })
        .sum()
}

fn mean_f32(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

fn std_f32(v: &[f32]) -> f32 {
    let m = mean_f32(v);
    let var: f32 = v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32;
    var.sqrt()
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. SparseMoeLayer — standard sparse MoE with top-K routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a sparse MoE layer.
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub n_experts: usize,
    pub expert_dim: usize,
    pub input_dim: usize,
    pub top_k: usize,
    /// Capacity factor multiplied with (batch / n_experts) to set per-expert token limit.
    pub capacity_factor: f32,
}

/// Two-layer FFN expert with SiLU activation (SwiGLU-style gating omitted for clarity).
#[derive(Debug, Clone)]
pub struct SparseExpert {
    /// W1: [expert_dim x input_dim] flattened row-major.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// W2: [input_dim x expert_dim] flattened row-major.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub input_dim: usize,
    pub expert_dim: usize,
}

impl SparseExpert {
    pub fn new(input_dim: usize, expert_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            w1: xavier_f32(expert_dim, input_dim, rng),
            b1: vec![0.0; expert_dim],
            w2: xavier_f32(input_dim, expert_dim, rng),
            b2: vec![0.0; input_dim],
            input_dim,
            expert_dim,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "SparseExpert::forward",
                &format!("input len {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        // hidden = SiLU(W1 x + b1)
        let mut h = matvec_f32(&self.w1, x, self.expert_dim, self.input_dim);
        for (i, v) in h.iter_mut().enumerate() {
            *v = silu(*v + self.b1[i]);
        }
        // out = W2 h + b2
        let mut out = matvec_f32(&self.w2, &h, self.input_dim, self.expert_dim);
        for (i, v) in out.iter_mut().enumerate() {
            *v += self.b2[i];
        }
        Ok(out)
    }
}

/// Linear router projecting input -> logits over experts.
#[derive(Debug, Clone)]
pub struct SparseRouter {
    /// Weights [n_experts x input_dim] row-major.
    pub weights: Vec<f32>,
    pub n_experts: usize,
    pub input_dim: usize,
}

impl SparseRouter {
    pub fn new(n_experts: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            weights: xavier_f32(n_experts, input_dim, rng),
            n_experts,
            input_dim,
        }
    }

    /// Returns (expert_indices, normalised gate weights).
    pub fn dispatch(&self, x: &[f32], top_k: usize) -> Result<(Vec<usize>, Vec<f32>), TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "SparseRouter::dispatch",
                &format!("x len {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        let logits = matvec_f32(&self.weights, x, self.n_experts, self.input_dim);
        let probs = softmax_f32(&logits);
        let k = top_k.min(self.n_experts).max(1);
        let indices = top_k_indices(&probs, k);
        let raw: Vec<f32> = indices.iter().map(|&i| probs[i]).collect();
        let s: f32 = raw.iter().sum();
        let d = if s < f32::EPSILON { 1.0 } else { s };
        let gates: Vec<f32> = raw.iter().map(|&p| p / d).collect();
        Ok((indices, gates))
    }
}

/// Sparse Mixture-of-Experts layer with load-balanced dispatch.
#[derive(Debug, Clone)]
pub struct SparseMoeLayer {
    pub config: SparseConfig,
    pub router: SparseRouter,
    pub experts: Vec<SparseExpert>,
}

impl SparseMoeLayer {
    pub fn new(config: SparseConfig, rng: &mut StdRng) -> Self {
        let router = SparseRouter::new(config.n_experts, config.input_dim, rng);
        let experts = (0..config.n_experts)
            .map(|_| SparseExpert::new(config.input_dim, config.expert_dim, rng))
            .collect();
        Self {
            config,
            router,
            experts,
        }
    }

    /// Forward pass for a single token vector.
    /// Dispatches to top-K experts, gates and sums their outputs.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        let (indices, gates) = self.router.dispatch(x, self.config.top_k)?;
        let mut out = vec![0.0f32; self.config.input_dim];
        for (rank, &expert_idx) in indices.iter().enumerate() {
            let expert_out = self.experts[expert_idx].forward(x)?;
            let g = gates[rank];
            for (o, &e) in out.iter_mut().zip(expert_out.iter()) {
                *o += g * e;
            }
        }
        Ok(out)
    }

    /// Forward pass over a batch of tokens, returning per-token outputs and
    /// per-expert token counts (for load-balancing monitoring).
    pub fn forward_batch(
        &self,
        xs: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), TensorError> {
        let mut outputs = Vec::with_capacity(xs.len());
        let mut counts = vec![0usize; self.config.n_experts];
        for x in xs {
            let (indices, _) = self.router.dispatch(x, self.config.top_k)?;
            for &i in &indices {
                counts[i] += 1;
            }
            let out = self.forward(x)?;
            outputs.push(out);
        }
        Ok((outputs, counts))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. MixtureOfDepths (MoD) — dynamic compute allocation
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for a Mixture-of-Depths layer.
#[derive(Debug, Clone)]
pub struct ModConfig {
    /// Total sequence length.
    pub seq_len: usize,
    /// Hidden dimension of each token.
    pub d_model: usize,
    /// Number of tokens to route through the heavy block (router dimension = d_model).
    pub top_r: usize,
    /// Alias: n_tokens is a synonym for seq_len in external APIs.
    pub n_tokens: usize,
}

/// A simple two-layer transformer block used by MoD for selected tokens.
#[derive(Debug, Clone)]
pub struct TransformerBlockMod {
    /// W_ff1: [4*d x d], W_ff2: [d x 4*d] row-major.
    pub w_ff1: Vec<f32>,
    pub w_ff2: Vec<f32>,
    pub d_model: usize,
}

impl TransformerBlockMod {
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let ff_dim = 4 * d_model;
        Self {
            w_ff1: xavier_f32(ff_dim, d_model, rng),
            w_ff2: xavier_f32(d_model, ff_dim, rng),
            d_model,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        let ff_dim = 4 * self.d_model;
        if x.len() != self.d_model {
            return Err(TensorError::invalid_argument_op(
                "TransformerBlockMod::forward",
                &format!("x.len() {} != d_model {}", x.len(), self.d_model),
            ));
        }
        let mut h = matvec_f32(&self.w_ff1, x, ff_dim, self.d_model);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        let out = matvec_f32(&self.w_ff2, &h, self.d_model, ff_dim);
        // residual
        Ok(out.iter().zip(x.iter()).map(|(&a, &b)| a + b).collect())
    }
}

/// Mixture-of-Depths layer: selects top-R tokens for heavy compute.
#[derive(Debug, Clone)]
pub struct MixtureOfDepths {
    pub config: ModConfig,
    /// Router weights: \[d_model\] — scalar score per token via dot product.
    pub router_weights: Vec<f32>,
    pub block: TransformerBlockMod,
}

impl MixtureOfDepths {
    pub fn new(config: ModConfig, rng: &mut StdRng) -> Self {
        let router_weights: Vec<f32> = (0..config.d_model)
            .map(|_| rng.random::<f32>() * 0.02 - 0.01)
            .collect();
        let block = TransformerBlockMod::new(config.d_model, rng);
        Self {
            config,
            router_weights,
            block,
        }
    }

    /// Returns (selected_indices, pass_through_indices).
    pub fn route_tokens(&self, hidden: &[Vec<f32>]) -> (Vec<usize>, Vec<usize>) {
        let scores: Vec<f32> = hidden
            .iter()
            .map(|h| dot_f32(&self.router_weights, h))
            .collect();
        let r = self.config.top_r.min(hidden.len());
        let selected_set: std::collections::HashSet<usize> =
            top_k_indices(&scores, r).into_iter().collect();
        let selected: Vec<usize> = (0..hidden.len())
            .filter(|i| selected_set.contains(i))
            .collect();
        let pass_through: Vec<usize> = (0..hidden.len())
            .filter(|i| !selected_set.contains(i))
            .collect();
        (selected, pass_through)
    }

    /// Forward pass: selected tokens go through the transformer block; others pass through.
    pub fn forward(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, TensorError> {
        let (selected, _pass) = self.route_tokens(x);
        let selected_set: std::collections::HashSet<usize> = selected.iter().cloned().collect();
        let mut out = x.to_vec();
        for i in selected_set {
            out[i] = self.block.forward(&x[i])?;
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. MegaBlockMoe — GQA attention fused with sparse MoE
// ═══════════════════════════════════════════════════════════════════════════════

/// MegaBlock config combining grouped-query attention with MoE.
#[derive(Debug, Clone)]
pub struct MegaBlockConfig {
    pub n_experts: usize,
    pub n_heads: usize,
    pub kv_heads: usize,
    pub d_model: usize,
    pub expert_dim: usize,
    pub top_k: usize,
}

/// Grouped-Query Attention component for MegaBlock.
#[derive(Debug, Clone)]
pub struct GqaAttention {
    pub n_heads: usize,
    pub kv_heads: usize,
    pub d_model: usize,
    pub head_dim: usize,
    /// Q, K, V projection weights (flattened).
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl GqaAttention {
    pub fn new(n_heads: usize, kv_heads: usize, d_model: usize, rng: &mut StdRng) -> Self {
        let head_dim = (d_model / n_heads).max(1);
        let q_dim = n_heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        Self {
            n_heads,
            kv_heads,
            d_model,
            head_dim,
            w_q: xavier_f32(q_dim, d_model, rng),
            w_k: xavier_f32(kv_dim, d_model, rng),
            w_v: xavier_f32(kv_dim, d_model, rng),
            w_o: xavier_f32(d_model, q_dim, rng),
        }
    }

    /// Single-token, causal-free GQA forward (no mask for simplicity).
    /// `x` shape: [seq_len * d_model] flattened.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>, TensorError> {
        if x.len() != seq_len * self.d_model {
            return Err(TensorError::invalid_argument_op(
                "GqaAttention::forward",
                "input size mismatch",
            ));
        }
        let head_dim = self.head_dim;
        let kv_heads = self.kv_heads;
        let n_heads = self.n_heads;
        let d_model = self.d_model;
        let q_dim = n_heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        // Project all tokens.
        let mut q_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut k_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut v_all: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let tok = &x[t * d_model..(t + 1) * d_model];
            q_all.push(matvec_f32(&self.w_q, tok, q_dim, d_model));
            k_all.push(matvec_f32(&self.w_k, tok, kv_dim, d_model));
            v_all.push(matvec_f32(&self.w_v, tok, kv_dim, d_model));
        }

        // Per-token output via scaled dot-product.
        let mut out_flat = vec![0.0f32; seq_len * d_model];
        for t in 0..seq_len {
            let mut head_out = vec![0.0f32; q_dim];
            for h in 0..n_heads {
                let kv_h = h % kv_heads;
                let q_slice = &q_all[t][h * head_dim..(h + 1) * head_dim];
                // Attention over all tokens (non-causal, simplified).
                let raw_scores: Vec<f32> = (0..seq_len)
                    .map(|s| {
                        let k_slice = &k_all[s][kv_h * head_dim..(kv_h + 1) * head_dim];
                        dot_f32(q_slice, k_slice) * scale
                    })
                    .collect();
                let attn = softmax_f32(&raw_scores);
                let mut ctx = vec![0.0f32; head_dim];
                for s in 0..seq_len {
                    let v_slice = &v_all[s][kv_h * head_dim..(kv_h + 1) * head_dim];
                    for (c, &v) in ctx.iter_mut().zip(v_slice.iter()) {
                        *c += attn[s] * v;
                    }
                }
                let base = h * head_dim;
                for (i, &c) in ctx.iter().enumerate() {
                    head_out[base + i] = c;
                }
            }
            // Output projection.
            let proj = matvec_f32(&self.w_o, &head_out, d_model, q_dim);
            let base = t * d_model;
            for (i, &p) in proj.iter().enumerate() {
                out_flat[base + i] = x[base + i] + p; // residual
            }
        }
        Ok(out_flat)
    }
}

/// MegaBlock: GQA attention over all tokens, then per-token sparse MoE.
#[derive(Debug, Clone)]
pub struct MegaBlockMoe {
    pub config: MegaBlockConfig,
    pub attention: GqaAttention,
    pub moe: SparseMoeLayer,
}

impl MegaBlockMoe {
    pub fn new(config: MegaBlockConfig, rng: &mut StdRng) -> Self {
        let attn = GqaAttention::new(config.n_heads, config.kv_heads, config.d_model, rng);
        let sparse_cfg = SparseConfig {
            n_experts: config.n_experts,
            expert_dim: config.expert_dim,
            input_dim: config.d_model,
            top_k: config.top_k,
            capacity_factor: 1.25,
        };
        let moe = SparseMoeLayer::new(sparse_cfg, rng);
        Self {
            config,
            attention: attn,
            moe,
        }
    }

    /// Forward: x is [seq_len * d_model] flattened. Returns [seq_len * d_model].
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>, TensorError> {
        let d = self.config.d_model;
        // Step 1: GQA attention over all tokens.
        let attn_out = self.attention.forward(x, seq_len)?;
        // Step 2: Per-token MoE.
        let mut final_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let tok = &attn_out[t * d..(t + 1) * d];
            let moe_out = self.moe.forward(tok)?;
            let base = t * d;
            for (i, &v) in moe_out.iter().enumerate() {
                final_out[base + i] = attn_out[base + i] + v; // residual
            }
        }
        Ok(final_out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. HierarchicalMoe — two-level coarse→fine MoE routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the hierarchical two-level MoE.
#[derive(Debug, Clone)]
pub struct CoarseConfig {
    pub n_groups: usize,
    pub n_experts_per_group: usize,
    pub top_k_coarse: usize,
    pub top_k_fine: usize,
    pub input_dim: usize,
    pub expert_dim: usize,
}

/// Single routing level: linear router + leaf expert pool.
#[derive(Debug, Clone)]
pub struct RoutingLevel {
    /// Gate weights: [n_choices x input_dim] row-major.
    pub gate_w: Vec<f32>,
    pub n_choices: usize,
    pub input_dim: usize,
}

impl RoutingLevel {
    pub fn new(n_choices: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            gate_w: xavier_f32(n_choices, input_dim, rng),
            n_choices,
            input_dim,
        }
    }

    /// Returns (indices, gates) for top-k choices.
    pub fn route(&self, x: &[f32], top_k: usize) -> Result<(Vec<usize>, Vec<f32>), TensorError> {
        let logits = matvec_f32(&self.gate_w, x, self.n_choices, self.input_dim);
        let probs = softmax_f32(&logits);
        let k = top_k.min(self.n_choices).max(1);
        let indices = top_k_indices(&probs, k);
        let raw: Vec<f32> = indices.iter().map(|&i| probs[i]).collect();
        let s: f32 = raw.iter().sum();
        let d = if s < f32::EPSILON { 1.0 } else { s };
        Ok((indices, raw.iter().map(|&p| p / d).collect()))
    }
}

/// Hierarchical two-level MoE.
#[derive(Debug, Clone)]
pub struct HierarchicalMoe {
    pub config: CoarseConfig,
    /// Coarse-level routers (one per nothing — selects among groups).
    pub coarse_router: RoutingLevel,
    /// Fine-level routers — one per group, selects among experts within group.
    pub fine_routers: Vec<RoutingLevel>,
    /// Leaf experts: \[n_groups\]\[n_experts_per_group\].
    pub leaf_experts: Vec<Vec<SparseExpert>>,
}

impl HierarchicalMoe {
    pub fn new(config: CoarseConfig, rng: &mut StdRng) -> Self {
        let coarse_router = RoutingLevel::new(config.n_groups, config.input_dim, rng);
        let fine_routers = (0..config.n_groups)
            .map(|_| RoutingLevel::new(config.n_experts_per_group, config.input_dim, rng))
            .collect();
        let leaf_experts = (0..config.n_groups)
            .map(|_| {
                (0..config.n_experts_per_group)
                    .map(|_| SparseExpert::new(config.input_dim, config.expert_dim, rng))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self {
            config,
            coarse_router,
            fine_routers,
            leaf_experts,
        }
    }

    /// Forward: coarse router selects groups, fine routers select experts within groups.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        let (coarse_idx, coarse_gates) = self.coarse_router.route(x, self.config.top_k_coarse)?;
        let mut out = vec![0.0f32; self.config.input_dim];
        for (ci, &group_idx) in coarse_idx.iter().enumerate() {
            let cg = coarse_gates[ci];
            let (fine_idx, fine_gates) =
                self.fine_routers[group_idx].route(x, self.config.top_k_fine)?;
            for (fi, &expert_idx) in fine_idx.iter().enumerate() {
                let fg = fine_gates[fi];
                let expert_out = self.leaf_experts[group_idx][expert_idx].forward(x)?;
                let combined = cg * fg;
                for (o, &e) in out.iter_mut().zip(expert_out.iter()) {
                    *o += combined * e;
                }
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. MoELoadBalancer — auxiliary losses for balanced routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Summary of load balance for a batch.
#[derive(Debug, Clone)]
pub struct LoadBalanceReport {
    /// Per-expert fraction of tokens assigned.
    pub expert_loads: Vec<f32>,
    /// Coefficient of variation of importance scores across experts.
    pub cv: f32,
    /// Fraction of tokens that exceeded capacity and were dropped.
    pub dropped_fraction: f32,
}

/// Auxiliary loss utilities for MoE load balancing.
#[derive(Debug, Clone)]
pub struct MoELoadBalancer {
    pub n_experts: usize,
    pub z_coeff: f32,
    pub importance_coeff: f32,
    pub load_coeff: f32,
}

impl MoELoadBalancer {
    pub fn new(n_experts: usize) -> Self {
        Self {
            n_experts,
            z_coeff: 1e-3,
            importance_coeff: 1e-2,
            load_coeff: 1e-2,
        }
    }

    /// Z-loss: `z_coeff * (log Σ_i exp(z_i))^2`.
    pub fn z_loss(&self, logits: &[f32]) -> f32 {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
        self.z_coeff * log_sum.powi(2)
    }

    /// Importance loss: CV^2 of per-expert importance sums (squared coefficient of variation).
    /// `gates` is shape \[n_tokens\]\[n_experts\].
    pub fn importance_loss(&self, gates: &[Vec<f32>]) -> f32 {
        if gates.is_empty() {
            return 0.0;
        }
        // Per-expert importance = sum of gate weights for that expert.
        let mut importance = vec![0.0f32; self.n_experts];
        for token_gates in gates {
            let len = token_gates.len().min(self.n_experts);
            for (e, &g) in token_gates[..len].iter().enumerate() {
                importance[e] += g;
            }
        }
        let m = mean_f32(&importance);
        if m < f32::EPSILON {
            return 0.0;
        }
        let std = std_f32(&importance);
        let cv = std / m;
        self.importance_coeff * cv.powi(2)
    }

    /// Load loss: fraction of dropped tokens (those beyond per-expert capacity).
    pub fn load_loss(&self, gates: &[Vec<f32>], capacity: f32) -> f32 {
        if gates.is_empty() {
            return 0.0;
        }
        let n_tokens = gates.len();
        let cap = ((n_tokens as f32 / self.n_experts as f32) * capacity).ceil() as usize;
        let mut counts = vec![0usize; self.n_experts];
        let mut dropped = 0usize;
        for token_gates in gates {
            if token_gates.is_empty() {
                continue;
            }
            // Use argmax as assigned expert.
            let best = token_gates
                .iter()
                .cloned()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if best < self.n_experts {
                if counts[best] >= cap {
                    dropped += 1;
                } else {
                    counts[best] += 1;
                }
            }
        }
        self.load_coeff * dropped as f32 / n_tokens as f32
    }

    /// Full balance report from a batch of gate distributions.
    /// `gates` shape: \[n_tokens\]\[n_experts\].
    pub fn balance_report(&self, gates: &[Vec<f32>]) -> LoadBalanceReport {
        let n_tokens = gates.len().max(1) as f32;
        let mut expert_loads = vec![0.0f32; self.n_experts];
        for token_gates in gates {
            if token_gates.is_empty() {
                continue;
            }
            // Argmax assignment.
            let best = token_gates
                .iter()
                .cloned()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if best < self.n_experts {
                expert_loads[best] += 1.0 / n_tokens;
            }
        }
        let m = mean_f32(&expert_loads);
        let s = std_f32(&expert_loads);
        let cv = if m > f32::EPSILON { s / m } else { 0.0 };

        // Dropped fraction using capacity_factor = 1.25.
        let cap = ((n_tokens / self.n_experts as f32) * 1.25).ceil() as usize;
        let mut counts = vec![0usize; self.n_experts];
        let mut dropped = 0usize;
        for token_gates in gates {
            if token_gates.is_empty() {
                continue;
            }
            let best = token_gates
                .iter()
                .cloned()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if best < self.n_experts {
                if counts[best] >= cap {
                    dropped += 1;
                } else {
                    counts[best] += 1;
                }
            }
        }

        LoadBalanceReport {
            expert_loads,
            cv,
            dropped_fraction: dropped as f32 / n_tokens,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. ExpertMerger — post-training expert merging
// ═══════════════════════════════════════════════════════════════════════════════

/// Merges expert weight matrices using various strategies.
pub struct ExpertMerger;

impl ExpertMerger {
    /// Fisher-weighted merge: weighted average of expert weights by Fisher importance diagonal.
    /// `experts`: each expert is a list of weight matrices (`Vec<f32>` per layer).
    /// `fisher_diagonals`: same shape as experts, per-parameter Fisher information.
    pub fn fisher_weighted_merge(
        experts: &[Vec<Vec<f32>>],
        fisher_diagonals: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<f32>>, TensorError> {
        if experts.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ExpertMerger::fisher_weighted_merge",
                "no experts provided",
            ));
        }
        if experts.len() != fisher_diagonals.len() {
            return Err(TensorError::invalid_argument_op(
                "ExpertMerger::fisher_weighted_merge",
                "experts and fisher_diagonals length mismatch",
            ));
        }
        let n_layers = experts[0].len();
        let mut merged: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let n_params = experts[0][l].len();
            let mut numerator = vec![0.0f32; n_params];
            let mut denominator = vec![0.0f32; n_params];
            for e in 0..experts.len() {
                if l >= experts[e].len() || l >= fisher_diagonals[e].len() {
                    continue;
                }
                let layer = &experts[e][l];
                let fisher = &fisher_diagonals[e][l];
                let fp = fisher.len().min(n_params);
                for p in 0..fp {
                    let f_val = fisher[p].max(0.0);
                    numerator[p] += f_val * layer[p];
                    denominator[p] += f_val;
                }
            }
            let layer_merged: Vec<f32> = numerator
                .iter()
                .zip(denominator.iter())
                .map(|(&n, &d)| if d > f32::EPSILON { n / d } else { 0.0 })
                .collect();
            merged.push(layer_merged);
        }
        Ok(merged)
    }

    /// DARE merge: randomly drop parameters then rescale before averaging.
    /// `density` in (0, 1]: fraction of parameters to keep.
    pub fn dare_merge(
        experts: &[Vec<Vec<f32>>],
        density: f32,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f32>>, TensorError> {
        if experts.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ExpertMerger::dare_merge",
                "no experts provided",
            ));
        }
        let density = density.clamp(0.0, 1.0);
        let n_layers = experts[0].len();
        let n_experts = experts.len() as f32;
        let mut merged: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let n_params = experts[0][l].len();
            let mut sum = vec![0.0f32; n_params];
            for e in 0..experts.len() {
                if l >= experts[e].len() {
                    continue;
                }
                for p in 0..n_params.min(experts[e][l].len()) {
                    let keep: f32 = rng.random();
                    if keep < density {
                        // Rescale by 1/density to preserve expected magnitude.
                        sum[p] += experts[e][l][p] / density.max(f32::EPSILON);
                    }
                }
            }
            merged.push(sum.iter().map(|&v| v / n_experts).collect());
        }
        Ok(merged)
    }

    /// TIES merge: Trim insignificant weights, Elect sign by majority, Sum survivors.
    /// `k` is the fraction of parameters to keep per expert (top-k by magnitude).
    pub fn ties_merge(experts: &[Vec<Vec<f32>>], k: f32) -> Result<Vec<Vec<f32>>, TensorError> {
        if experts.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ExpertMerger::ties_merge",
                "no experts provided",
            ));
        }
        let k = k.clamp(0.0, 1.0);
        let n_layers = experts[0].len();
        let mut merged: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let n_params = experts[0][l].len();
            // Step 1 — Trim: zero out all but top-k fraction per expert.
            let mut trimmed: Vec<Vec<f32>> = experts
                .iter()
                .filter_map(|e| e.get(l))
                .map(|layer| {
                    let mut v = layer.clone();
                    let keep = ((v.len() as f32) * k).ceil() as usize;
                    let keep = keep.min(v.len());
                    // Find threshold.
                    let mut mags: Vec<f32> = v.iter().map(|&x| x.abs()).collect();
                    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let threshold = if keep < mags.len() { mags[keep] } else { 0.0 };
                    for x in v.iter_mut() {
                        if x.abs() < threshold {
                            *x = 0.0;
                        }
                    }
                    v
                })
                .collect();

            // Step 2 — Elect sign: majority vote per parameter.
            let mut elected = vec![0.0f32; n_params];
            for p in 0..n_params {
                let mut pos_count = 0i32;
                let mut neg_count = 0i32;
                for t in &trimmed {
                    if p < t.len() {
                        if t[p] > 0.0 {
                            pos_count += 1;
                        } else if t[p] < 0.0 {
                            neg_count += 1;
                        }
                    }
                }
                elected[p] = if pos_count >= neg_count { 1.0 } else { -1.0 };
            }

            // Step 3 — Sum survivors: only count params matching elected sign.
            let mut sum = vec![0.0f32; n_params];
            let mut counts = vec![0u32; n_params];
            for t in &mut trimmed {
                for p in 0..n_params.min(t.len()) {
                    let matches_sign =
                        (elected[p] > 0.0 && t[p] > 0.0) || (elected[p] < 0.0 && t[p] < 0.0);
                    if matches_sign {
                        sum[p] += t[p];
                        counts[p] += 1;
                    }
                }
            }
            let layer_merged: Vec<f32> = sum
                .iter()
                .zip(counts.iter())
                .map(|(&s, &c)| if c > 0 { s / c as f32 } else { 0.0 })
                .collect();
            merged.push(layer_merged);
        }
        Ok(merged)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. SpecialistRouter — domain-classification-based routing
// ═══════════════════════════════════════════════════════════════════════════════

/// Classifies input tokens into domains via argmax of a linear projection.
#[derive(Debug, Clone)]
pub struct DomainClassifier {
    pub n_domains: usize,
    /// Weights [n_domains x input_dim] row-major.
    pub weights: Vec<f32>,
    pub input_dim: usize,
}

impl DomainClassifier {
    pub fn new(n_domains: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            n_domains,
            weights: xavier_f32(n_domains, input_dim, rng),
            input_dim,
        }
    }

    /// Returns the predicted domain index (argmax of logits).
    pub fn classify(&self, x: &[f32]) -> Result<usize, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "DomainClassifier::classify",
                "input dimension mismatch",
            ));
        }
        let logits = matvec_f32(&self.weights, x, self.n_domains, self.input_dim);
        let domain = logits
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(domain)
    }
}

/// Specialist MoE: routes tokens to domain-specific experts based on domain classification.
#[derive(Debug, Clone)]
pub struct SpecialistMoe {
    pub routers: Vec<DomainClassifier>,
    /// Expert weights: [n_domains][input_dim x input_dim] flattened row-major (linear transform).
    pub experts: Vec<Vec<f32>>,
    pub input_dim: usize,
    pub n_domains: usize,
}

impl SpecialistMoe {
    pub fn new(n_domains: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        let routers = (0..1)
            .map(|_| DomainClassifier::new(n_domains, input_dim, rng))
            .collect();
        let experts = (0..n_domains)
            .map(|_| xavier_f32(input_dim, input_dim, rng))
            .collect();
        Self {
            routers,
            experts,
            input_dim,
            n_domains,
        }
    }

    /// Forward: classify domain, apply domain-specific expert linear transform.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, TensorError> {
        if self.routers.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SpecialistMoe::forward",
                "no routers",
            ));
        }
        let domain = self.routers[0].classify(x)?;
        let d = domain.min(self.n_domains.saturating_sub(1));
        let out = matvec_f32(&self.experts[d], x, self.input_dim, self.input_dim);
        // residual add
        Ok(out.iter().zip(x.iter()).map(|(&a, &b)| a + b).collect())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. ExpertAdapter — per-expert LoRA adapters
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for per-expert LoRA adapters.
#[derive(Debug, Clone)]
pub struct ExpertLoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub input_dim: usize,
    pub output_dim: usize,
}

/// LoRA adapter for one expert: A ∈ R^[rank x input_dim], B ∈ R^[output_dim x rank].
#[derive(Debug, Clone)]
pub struct ExpertAdapter {
    /// (A, B) matrices per expert, A: [rank x input_dim], B: [output_dim x rank].
    pub adapters: Vec<(Vec<f32>, Vec<f32>)>,
    pub config: ExpertLoraConfig,
}

impl ExpertAdapter {
    pub fn new(n_experts: usize, config: ExpertLoraConfig, rng: &mut StdRng) -> Self {
        let adapters = (0..n_experts)
            .map(|_| {
                let a = xavier_f32(config.rank, config.input_dim, rng);
                let b = vec![0.0f32; config.output_dim * config.rank]; // B initialized to zero (LoRA convention)
                (a, b)
            })
            .collect();
        Self { adapters, config }
    }

    /// Compute LoRA correction: base_output + (alpha/rank) * B * A * x.
    pub fn adapt(
        &self,
        x: &[f32],
        expert_idx: usize,
        base_output: &[f32],
    ) -> Result<Vec<f32>, TensorError> {
        if expert_idx >= self.adapters.len() {
            return Err(TensorError::invalid_argument_op(
                "ExpertAdapter::adapt",
                &format!(
                    "expert_idx {} >= n_experts {}",
                    expert_idx,
                    self.adapters.len()
                ),
            ));
        }
        if x.len() != self.config.input_dim {
            return Err(TensorError::invalid_argument_op(
                "ExpertAdapter::adapt",
                "x dimension mismatch",
            ));
        }
        let (a, b) = &self.adapters[expert_idx];
        let scale = self.config.alpha / self.config.rank.max(1) as f32;
        // r = A * x  shape [rank]
        let r = matvec_f32(a, x, self.config.rank, self.config.input_dim);
        // lora_out = B * r  shape [output_dim]
        let lora_out = matvec_f32(b, &r, self.config.output_dim, self.config.rank);
        let out: Vec<f32> = base_output
            .iter()
            .zip(lora_out.iter())
            .map(|(&base, &lo)| base + scale * lo)
            .collect();
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. MoeProfiler — runtime profiling for MoE efficiency
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-call and cumulative profiling report.
#[derive(Debug, Clone)]
pub struct MoeProfileReport {
    /// Fraction of tokens assigned to each expert.
    pub tokens_per_expert: Vec<f32>,
    /// Fraction of calls where at least one token was dropped (overflow).
    pub overflow_rate: f32,
    /// Average utilization relative to nominal capacity.
    pub effective_capacity: f32,
}

/// Accumulates dispatch statistics across multiple forward calls.
#[derive(Debug, Clone)]
pub struct MoeProfiler {
    pub n_experts: usize,
    /// Total tokens dispatched to each expert.
    cumulative_counts: Vec<u64>,
    /// Total tokens seen.
    total_tokens: u64,
    /// Number of batches with overflow.
    overflow_calls: u64,
    /// Total calls recorded.
    total_calls: u64,
    /// Nominal capacity per expert = seq_len / n_experts.
    nominal_capacity: usize,
}

impl MoeProfiler {
    pub fn new(n_experts: usize, nominal_capacity: usize) -> Self {
        Self {
            n_experts,
            cumulative_counts: vec![0u64; n_experts],
            total_tokens: 0,
            overflow_calls: 0,
            total_calls: 0,
            nominal_capacity,
        }
    }

    /// Record one forward pass dispatch.
    pub fn record_dispatch(&mut self, expert_assignments: &[usize], seq_len: usize) {
        self.total_calls += 1;
        self.total_tokens += seq_len as u64;
        let mut call_counts = vec![0usize; self.n_experts];
        for &e in expert_assignments {
            if e < self.n_experts {
                self.cumulative_counts[e] += 1;
                call_counts[e] += 1;
            }
        }
        // Check for overflow.
        let cap = self.nominal_capacity.max(1);
        let overflowed = call_counts.iter().any(|&c| c > cap);
        if overflowed {
            self.overflow_calls += 1;
        }
    }

    /// Per-expert fraction of total tokens dispatched.
    pub fn token_utilization(&self) -> Vec<f32> {
        let total = self.total_tokens.max(1) as f32;
        self.cumulative_counts
            .iter()
            .map(|&c| c as f32 / total)
            .collect()
    }

    pub fn report(&self) -> MoeProfileReport {
        let total = self.total_tokens.max(1) as f32;
        let tokens_per_expert: Vec<f32> = self
            .cumulative_counts
            .iter()
            .map(|&c| c as f32 / total)
            .collect();
        let overflow_rate = if self.total_calls > 0 {
            self.overflow_calls as f32 / self.total_calls as f32
        } else {
            0.0
        };
        let avg_load = mean_f32(&tokens_per_expert);
        let ideal_load = 1.0 / self.n_experts as f32;
        let effective_capacity = if ideal_load > f32::EPSILON {
            avg_load / ideal_load
        } else {
            0.0
        };
        MoeProfileReport {
            tokens_per_expert,
            overflow_rate,
            effective_capacity,
        }
    }

    pub fn reset(&mut self) {
        self.cumulative_counts.fill(0);
        self.total_tokens = 0;
        self.overflow_calls = 0;
        self.total_calls = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. ConditionalCompute — early-exit MoE variant
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for conditional-compute early exit.
#[derive(Debug, Clone)]
pub struct ExitConfig {
    pub n_layers: usize,
    /// Entropy threshold per layer — exit if softmax entropy < threshold.
    pub thresholds: Vec<f32>,
    pub d_model: usize,
}

impl ExitConfig {
    pub fn new(n_layers: usize, d_model: usize, threshold: f32) -> Self {
        Self {
            n_layers,
            thresholds: vec![threshold; n_layers],
            d_model,
        }
    }
}

/// Per-layer MLP used as a lightweight confidence scorer.
#[derive(Debug, Clone)]
pub struct ConfidenceScorer {
    /// [n_classes x d_model] row-major.
    pub w: Vec<f32>,
    pub n_classes: usize,
    pub d_model: usize,
}

impl ConfidenceScorer {
    pub fn new(d_model: usize, n_classes: usize, rng: &mut StdRng) -> Self {
        Self {
            w: xavier_f32(n_classes, d_model, rng),
            n_classes,
            d_model,
        }
    }

    /// Returns softmax-entropy of the class logits.
    pub fn entropy(&self, hidden: &[f32]) -> Result<f32, TensorError> {
        if hidden.len() != self.d_model {
            return Err(TensorError::invalid_argument_op(
                "ConfidenceScorer::entropy",
                "hidden dim mismatch",
            ));
        }
        let logits = matvec_f32(&self.w, hidden, self.n_classes, self.d_model);
        let probs = softmax_f32(&logits);
        Ok(entropy_f32(&probs))
    }
}

/// Stacked MoE layers with early exit based on per-layer confidence.
#[derive(Debug, Clone)]
pub struct ConditionalCompute {
    pub config: ExitConfig,
    /// Per-layer MoE.
    pub layers: Vec<SparseMoeLayer>,
    /// Per-layer confidence scorers.
    pub scorers: Vec<ConfidenceScorer>,
}

impl ConditionalCompute {
    pub fn new(
        config: ExitConfig,
        sparse_cfg: SparseConfig,
        n_classes: usize,
        rng: &mut StdRng,
    ) -> Self {
        let layers: Vec<SparseMoeLayer> = (0..config.n_layers)
            .map(|_| SparseMoeLayer::new(sparse_cfg.clone(), rng))
            .collect();
        let scorers: Vec<ConfidenceScorer> = (0..config.n_layers)
            .map(|_| ConfidenceScorer::new(config.d_model, n_classes, rng))
            .collect();
        Self {
            config,
            layers,
            scorers,
        }
    }

    /// Returns true if the model should exit at `layer` given the current hidden state.
    pub fn should_exit(&self, hidden: &[f32], layer: usize) -> bool {
        if layer >= self.config.n_layers || layer >= self.scorers.len() {
            return true;
        }
        let threshold = self.config.thresholds[layer];
        match self.scorers[layer].entropy(hidden) {
            Ok(ent) => ent < threshold,
            Err(_) => false,
        }
    }

    /// Forward with early exit. Returns (output, exit_layer).
    pub fn forward_with_early_exit(&self, x: &[f32]) -> Result<(Vec<f32>, usize), TensorError> {
        let mut h = x.to_vec();
        for layer_idx in 0..self.config.n_layers {
            h = self.layers[layer_idx].forward(&h)?;
            if self.should_exit(&h, layer_idx) {
                return Ok((h, layer_idx));
            }
        }
        Ok((h, self.config.n_layers.saturating_sub(1)))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── SparseExpert ───────────────────────────────────────────────────────────

    #[test]
    fn test_sparse_expert_forward_shape() {
        let mut rng = make_rng(1);
        let exp = SparseExpert::new(8, 16, &mut rng);
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let out = exp.forward(&x).expect("forward ok");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_sparse_expert_wrong_dim() {
        let mut rng = make_rng(2);
        let exp = SparseExpert::new(8, 16, &mut rng);
        let x = vec![0.0f32; 5]; // wrong dim
        assert!(exp.forward(&x).is_err());
    }

    #[test]
    fn test_sparse_expert_finite_outputs() {
        let mut rng = make_rng(3);
        let exp = SparseExpert::new(4, 8, &mut rng);
        let x = vec![1.0f32; 4];
        let out = exp.forward(&x).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── SparseRouter ──────────────────────────────────────────────────────────

    #[test]
    fn test_sparse_router_top_k_count() {
        let mut rng = make_rng(4);
        let router = SparseRouter::new(8, 16, &mut rng);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
        let (indices, gates) = router.dispatch(&x, 3).expect("test value");
        assert_eq!(indices.len(), 3);
        assert_eq!(gates.len(), 3);
    }

    #[test]
    fn test_sparse_router_gates_sum_to_one() {
        let mut rng = make_rng(5);
        let router = SparseRouter::new(4, 8, &mut rng);
        let x = vec![0.5f32; 8];
        let (_, gates) = router.dispatch(&x, 2).expect("test value");
        let s: f32 = gates.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "gates sum = {}", s);
    }

    #[test]
    fn test_sparse_router_top_k_clamped() {
        let mut rng = make_rng(6);
        let router = SparseRouter::new(3, 4, &mut rng);
        let x = vec![1.0f32; 4];
        let (indices, _) = router.dispatch(&x, 10).expect("test value"); // top_k > n_experts
        assert!(indices.len() <= 3);
    }

    // ── SparseMoeLayer ────────────────────────────────────────────────────────

    #[test]
    fn test_sparse_moe_forward_shape() {
        let mut rng = make_rng(7);
        let cfg = SparseConfig {
            n_experts: 4,
            expert_dim: 16,
            input_dim: 8,
            top_k: 2,
            capacity_factor: 1.25,
        };
        let moe = SparseMoeLayer::new(cfg, &mut rng);
        let x = vec![0.1f32; 8];
        let out = moe.forward(&x).expect("test value");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_sparse_moe_batch_forward() {
        let mut rng = make_rng(8);
        let cfg = SparseConfig {
            n_experts: 4,
            expert_dim: 8,
            input_dim: 4,
            top_k: 2,
            capacity_factor: 1.25,
        };
        let moe = SparseMoeLayer::new(cfg, &mut rng);
        let xs: Vec<Vec<f32>> = (0..6).map(|_| vec![0.1f32; 4]).collect();
        let (outs, counts) = moe.forward_batch(&xs).expect("test value");
        assert_eq!(outs.len(), 6);
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_sparse_moe_output_finite() {
        let mut rng = make_rng(9);
        let cfg = SparseConfig {
            n_experts: 2,
            expert_dim: 4,
            input_dim: 4,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let moe = SparseMoeLayer::new(cfg, &mut rng);
        let x = vec![1.0f32, -1.0, 0.5, -0.5];
        let out = moe.forward(&x).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── MixtureOfDepths ───────────────────────────────────────────────────────

    #[test]
    fn test_mod_route_tokens_counts() {
        let mut rng = make_rng(10);
        let cfg = ModConfig {
            seq_len: 8,
            d_model: 4,
            top_r: 3,
            n_tokens: 8,
        };
        let mod_layer = MixtureOfDepths::new(cfg, &mut rng);
        let hidden: Vec<Vec<f32>> = (0..8).map(|_| vec![0.1f32; 4]).collect();
        let (sel, pass) = mod_layer.route_tokens(&hidden);
        assert_eq!(sel.len() + pass.len(), 8);
        assert_eq!(sel.len(), 3);
    }

    #[test]
    fn test_mod_forward_shape_preserved() {
        let mut rng = make_rng(11);
        let cfg = ModConfig {
            seq_len: 5,
            d_model: 4,
            top_r: 2,
            n_tokens: 5,
        };
        let mod_layer = MixtureOfDepths::new(cfg, &mut rng);
        let hidden: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.1; 4]).collect();
        let out = mod_layer.forward(&hidden).expect("test value");
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|v| v.len() == 4));
    }

    #[test]
    fn test_mod_passthrough_unchanged() {
        let mut rng = make_rng(12);
        let cfg = ModConfig {
            seq_len: 4,
            d_model: 4,
            top_r: 0,
            n_tokens: 4,
        };
        let mod_layer = MixtureOfDepths::new(cfg, &mut rng);
        let hidden: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0f32; 4]).collect();
        let out = mod_layer.forward(&hidden).expect("test value");
        // All tokens should pass through unchanged (top_r = 0).
        assert_eq!(out, hidden);
    }

    // ── MegaBlockMoe ─────────────────────────────────────────────────────────

    #[test]
    fn test_mega_block_forward_shape() {
        let mut rng = make_rng(13);
        let cfg = MegaBlockConfig {
            n_experts: 2,
            n_heads: 2,
            kv_heads: 1,
            d_model: 4,
            expert_dim: 8,
            top_k: 1,
        };
        let block = MegaBlockMoe::new(cfg, &mut rng);
        let x = vec![0.1f32; 4 * 3]; // seq_len=3
        let out = block.forward(&x, 3).expect("test value");
        assert_eq!(out.len(), 4 * 3);
    }

    #[test]
    fn test_mega_block_output_finite() {
        let mut rng = make_rng(14);
        let cfg = MegaBlockConfig {
            n_experts: 2,
            n_heads: 2,
            kv_heads: 2,
            d_model: 4,
            expert_dim: 4,
            top_k: 1,
        };
        let block = MegaBlockMoe::new(cfg, &mut rng);
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.05).collect();
        let out = block.forward(&x, 2).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── HierarchicalMoe ───────────────────────────────────────────────────────

    #[test]
    fn test_hierarchical_moe_forward_shape() {
        let mut rng = make_rng(15);
        let cfg = CoarseConfig {
            n_groups: 3,
            n_experts_per_group: 2,
            top_k_coarse: 2,
            top_k_fine: 1,
            input_dim: 8,
            expert_dim: 16,
        };
        let hmoe = HierarchicalMoe::new(cfg, &mut rng);
        let x = vec![0.1f32; 8];
        let out = hmoe.forward(&x).expect("test value");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_hierarchical_moe_output_finite() {
        let mut rng = make_rng(16);
        let cfg = CoarseConfig {
            n_groups: 2,
            n_experts_per_group: 3,
            top_k_coarse: 1,
            top_k_fine: 2,
            input_dim: 4,
            expert_dim: 8,
        };
        let hmoe = HierarchicalMoe::new(cfg, &mut rng);
        let x = vec![0.5f32, -0.5, 0.1, -0.1];
        let out = hmoe.forward(&x).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── MoELoadBalancer ───────────────────────────────────────────────────────

    #[test]
    fn test_z_loss_positive() {
        let balancer = MoELoadBalancer::new(4);
        let logits = vec![0.1f32, 0.2, 0.3, 0.4];
        let loss = balancer.z_loss(&logits);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_importance_loss_uniform() {
        let balancer = MoELoadBalancer::new(4);
        // Uniform gate distribution -> CV = 0 -> importance_loss ≈ 0
        let gates: Vec<Vec<f32>> = (0..8).map(|_| vec![0.25; 4]).collect();
        let loss = balancer.importance_loss(&gates);
        assert!(
            loss < 0.01,
            "uniform gates should give near-zero importance loss"
        );
    }

    #[test]
    fn test_load_loss_no_overflow() {
        let balancer = MoELoadBalancer::new(4);
        // 1 token always goes to expert 0.
        let gates: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let loss = balancer.load_loss(&gates, 2.0);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_balance_report_expert_loads_sum_to_one() {
        let balancer = MoELoadBalancer::new(4);
        let gates: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let report = balancer.balance_report(&gates);
        let s: f32 = report.expert_loads.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_balance_report_cv_uniform() {
        let balancer = MoELoadBalancer::new(4);
        let gates: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let report = balancer.balance_report(&gates);
        // Perfectly balanced -> CV ≈ 0
        assert!(report.cv < 1e-5);
    }

    // ── ExpertMerger ──────────────────────────────────────────────────────────

    #[test]
    fn test_fisher_merge_shape() {
        let experts: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|_| vec![vec![1.0f32; 8], vec![2.0f32; 4]])
            .collect();
        let fishers: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|_| vec![vec![1.0f32; 8], vec![1.0f32; 4]])
            .collect();
        let merged = ExpertMerger::fisher_weighted_merge(&experts, &fishers).expect("test value");
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].len(), 8);
    }

    #[test]
    fn test_fisher_merge_identical_experts() {
        let experts: Vec<Vec<Vec<f32>>> = (0..3).map(|_| vec![vec![2.0f32; 4]]).collect();
        let fishers: Vec<Vec<Vec<f32>>> = (0..3).map(|_| vec![vec![1.0f32; 4]]).collect();
        let merged = ExpertMerger::fisher_weighted_merge(&experts, &fishers).expect("test value");
        for &v in &merged[0] {
            assert!((v - 2.0).abs() < 1e-5, "merged value {}", v);
        }
    }

    #[test]
    fn test_dare_merge_shape() {
        let mut rng = make_rng(20);
        let experts: Vec<Vec<Vec<f32>>> = (0..3).map(|_| vec![vec![1.0f32; 8]]).collect();
        let merged = ExpertMerger::dare_merge(&experts, 0.5, &mut rng).expect("test value");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].len(), 8);
    }

    #[test]
    fn test_ties_merge_shape() {
        let experts: Vec<Vec<Vec<f32>>> = (0..3).map(|e| vec![vec![e as f32 * 0.1; 6]]).collect();
        let merged = ExpertMerger::ties_merge(&experts, 0.5).expect("test value");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].len(), 6);
    }

    #[test]
    fn test_fisher_merge_empty_error() {
        let result = ExpertMerger::fisher_weighted_merge(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dare_merge_density_one() {
        let mut rng = make_rng(21);
        let experts: Vec<Vec<Vec<f32>>> = vec![vec![vec![4.0f32; 4]], vec![vec![4.0f32; 4]]];
        let merged = ExpertMerger::dare_merge(&experts, 1.0, &mut rng).expect("test value");
        // With density=1 and identical experts, merged ≈ 4.0
        assert!(merged[0].iter().all(|&v| (v - 4.0).abs() < 1e-4));
    }

    // ── DomainClassifier ──────────────────────────────────────────────────────

    #[test]
    fn test_domain_classifier_output_in_range() {
        let mut rng = make_rng(22);
        let clf = DomainClassifier::new(5, 8, &mut rng);
        let x = vec![0.1f32; 8];
        let domain = clf.classify(&x).expect("test value");
        assert!(domain < 5);
    }

    #[test]
    fn test_domain_classifier_wrong_dim() {
        let mut rng = make_rng(23);
        let clf = DomainClassifier::new(3, 8, &mut rng);
        let x = vec![0.1f32; 5];
        assert!(clf.classify(&x).is_err());
    }

    // ── SpecialistMoe ─────────────────────────────────────────────────────────

    #[test]
    fn test_specialist_moe_forward_shape() {
        let mut rng = make_rng(24);
        let moe = SpecialistMoe::new(4, 8, &mut rng);
        let x = vec![0.2f32; 8];
        let out = moe.forward(&x).expect("test value");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_specialist_moe_output_finite() {
        let mut rng = make_rng(25);
        let moe = SpecialistMoe::new(3, 4, &mut rng);
        let x = vec![1.0f32, -1.0, 0.5, -0.5];
        let out = moe.forward(&x).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── ExpertAdapter ─────────────────────────────────────────────────────────

    #[test]
    fn test_expert_adapter_output_shape() {
        let mut rng = make_rng(26);
        let cfg = ExpertLoraConfig {
            rank: 4,
            alpha: 8.0,
            input_dim: 8,
            output_dim: 8,
        };
        let adapter = ExpertAdapter::new(4, cfg, &mut rng);
        let x = vec![0.1f32; 8];
        let base = vec![0.5f32; 8];
        let out = adapter.adapt(&x, 0, &base).expect("test value");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_expert_adapter_invalid_expert_idx() {
        let mut rng = make_rng(27);
        let cfg = ExpertLoraConfig {
            rank: 2,
            alpha: 4.0,
            input_dim: 4,
            output_dim: 4,
        };
        let adapter = ExpertAdapter::new(2, cfg, &mut rng);
        let x = vec![0.1f32; 4];
        let base = vec![0.0f32; 4];
        assert!(adapter.adapt(&x, 5, &base).is_err());
    }

    #[test]
    fn test_expert_adapter_zero_b_init() {
        // B initialized to zero => lora_out = 0 => adapt returns base_output unchanged.
        let mut rng = make_rng(28);
        let cfg = ExpertLoraConfig {
            rank: 2,
            alpha: 1.0,
            input_dim: 4,
            output_dim: 4,
        };
        let adapter = ExpertAdapter::new(1, cfg, &mut rng);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let base = vec![0.5f32, 0.5, 0.5, 0.5];
        let out = adapter.adapt(&x, 0, &base).expect("test value");
        // B is zeros so out == base
        for (&a, &b) in out.iter().zip(base.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── MoeProfiler ───────────────────────────────────────────────────────────

    #[test]
    fn test_profiler_record_and_utilization() {
        let mut profiler = MoeProfiler::new(4, 2);
        profiler.record_dispatch(&[0, 1, 2, 3], 4);
        let util = profiler.token_utilization();
        assert_eq!(util.len(), 4);
        let s: f32 = util.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_profiler_overflow_detection() {
        let mut profiler = MoeProfiler::new(2, 1); // cap=1 per expert
                                                   // Send 3 tokens to expert 0 -> overflow.
        profiler.record_dispatch(&[0, 0, 0], 3);
        let report = profiler.report();
        assert!(report.overflow_rate > 0.0);
    }

    #[test]
    fn test_profiler_effective_capacity_balanced() {
        let mut profiler = MoeProfiler::new(4, 1);
        profiler.record_dispatch(&[0, 1, 2, 3], 4);
        let report = profiler.report();
        // Balanced assignment -> effective_capacity ≈ 1.0
        assert!((report.effective_capacity - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = MoeProfiler::new(4, 2);
        profiler.record_dispatch(&[0, 1], 2);
        profiler.reset();
        assert_eq!(profiler.total_tokens, 0);
        assert_eq!(profiler.total_calls, 0);
    }

    // ── ConditionalCompute ────────────────────────────────────────────────────

    #[test]
    fn test_conditional_compute_forward_shape() {
        let mut rng = make_rng(30);
        let exit_cfg = ExitConfig::new(3, 8, 1.0);
        let sparse_cfg = SparseConfig {
            n_experts: 2,
            expert_dim: 16,
            input_dim: 8,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let cc = ConditionalCompute::new(exit_cfg, sparse_cfg, 4, &mut rng);
        let x = vec![0.1f32; 8];
        let (out, _exit) = cc.forward_with_early_exit(&x).expect("test value");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_conditional_compute_exit_layer_valid() {
        let mut rng = make_rng(31);
        let exit_cfg = ExitConfig::new(4, 8, 1000.0); // very low threshold -> always exit at layer 0
        let sparse_cfg = SparseConfig {
            n_experts: 2,
            expert_dim: 8,
            input_dim: 8,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let cc = ConditionalCompute::new(exit_cfg, sparse_cfg, 4, &mut rng);
        let x = vec![0.1f32; 8];
        let (_, exit_layer) = cc.forward_with_early_exit(&x).expect("test value");
        assert!(exit_layer < 4);
    }

    #[test]
    fn test_conditional_compute_no_exit() {
        let mut rng = make_rng(32);
        let exit_cfg = ExitConfig::new(3, 8, -1.0); // threshold below any entropy -> never exits early
        let sparse_cfg = SparseConfig {
            n_experts: 2,
            expert_dim: 8,
            input_dim: 8,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let cc = ConditionalCompute::new(exit_cfg, sparse_cfg, 4, &mut rng);
        let x = vec![0.5f32; 8];
        let (out, exit_layer) = cc.forward_with_early_exit(&x).expect("test value");
        assert_eq!(out.len(), 8);
        assert_eq!(exit_layer, 2); // last layer index
    }

    #[test]
    fn test_should_exit_out_of_bounds() {
        let mut rng = make_rng(33);
        let exit_cfg = ExitConfig::new(2, 4, 0.5);
        let sparse_cfg = SparseConfig {
            n_experts: 2,
            expert_dim: 4,
            input_dim: 4,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let cc = ConditionalCompute::new(exit_cfg, sparse_cfg, 2, &mut rng);
        // Layer out of bounds -> should exit.
        assert!(cc.should_exit(&[0.0f32; 4], 99));
    }

    // ── GqaAttention ──────────────────────────────────────────────────────────

    #[test]
    fn test_gqa_attention_output_shape() {
        let mut rng = make_rng(34);
        let attn = GqaAttention::new(2, 1, 4, &mut rng);
        let x = vec![0.1f32; 4 * 2]; // seq_len=2, d_model=4
        let out = attn.forward(&x, 2).expect("test value");
        assert_eq!(out.len(), 4 * 2);
    }

    #[test]
    fn test_gqa_attention_finite_output() {
        let mut rng = make_rng(35);
        let attn = GqaAttention::new(2, 2, 4, &mut rng);
        let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect(); // seq_len=3
        let out = attn.forward(&x, 3).expect("test value");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Extra edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_hierarchical_moe_single_group() {
        let mut rng = make_rng(36);
        let cfg = CoarseConfig {
            n_groups: 1,
            n_experts_per_group: 2,
            top_k_coarse: 1,
            top_k_fine: 1,
            input_dim: 4,
            expert_dim: 8,
        };
        let hmoe = HierarchicalMoe::new(cfg, &mut rng);
        let x = vec![0.3f32; 4];
        let out = hmoe.forward(&x).expect("test value");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_sparse_moe_top1_deterministic_sign() {
        // Top-1 routing: output has non-trivial magnitude.
        let mut rng = make_rng(37);
        let cfg = SparseConfig {
            n_experts: 4,
            expert_dim: 8,
            input_dim: 4,
            top_k: 1,
            capacity_factor: 1.0,
        };
        let moe = SparseMoeLayer::new(cfg, &mut rng);
        let x = vec![1.0f32; 4];
        let out1 = moe.forward(&x).expect("test value");
        let out2 = moe.forward(&x).expect("test value");
        // Deterministic: same input -> same output.
        assert_eq!(out1, out2);
    }
}
