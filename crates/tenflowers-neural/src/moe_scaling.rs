//! Mixture of Experts & Large Model Scaling — TenfloweRS.
//!
//! Advanced MoE routing, expert networks, model parallelism utilities,
//! efficient attention variants, and quantization-aware inference.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ── Math helpers ──────────────────────────────────────────────────────────────

fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let d = if sum < f64::EPSILON { 1.0 } else { sum };
    exps.iter().map(|&e| e / d).collect()
}

#[inline]
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec_f64(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot_f64(row, x)).collect()
}

fn xavier_uniform(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| rng.random::<f64>() * 2.0 * limit - limit)
                .collect()
        })
        .collect()
}

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn elu_plus_one(x: f64) -> f64 {
    if x >= 0.0 {
        x + 1.0
    } else {
        x.exp()
    }
}

// ── Section 1: Advanced MoE Routing ──────────────────────────────────────────

/// Output of a top-K routing step.
#[derive(Debug, Clone)]
pub struct TopKRouterOutput {
    /// Selected expert indices per token `[batch x top_k]`.
    pub expert_indices: Vec<Vec<usize>>,
    /// Normalised gate weights for selected experts `[batch x top_k]`.
    pub gate_weights: Vec<Vec<f64>>,
    /// Raw router logits before softmax `[batch x n_experts]`.
    pub router_logits: Vec<Vec<f64>>,
}

/// Softmax gating router that selects the top-K experts per token.
#[derive(Debug, Clone)]
pub struct TopKRouter {
    pub n_experts: usize,
    pub top_k: usize,
    pub input_dim: usize,
    pub gate_weights: Vec<Vec<f64>>,
}

impl TopKRouter {
    pub fn new(n_experts: usize, top_k: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            n_experts,
            top_k,
            input_dim,
            gate_weights: xavier_uniform(n_experts, input_dim, rng),
        }
    }

    pub fn route(&self, tokens: &[Vec<f64>]) -> Result<TopKRouterOutput, TensorError> {
        let mut expert_indices = Vec::with_capacity(tokens.len());
        let mut gate_weights_out = Vec::with_capacity(tokens.len());
        let mut router_logits = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token.len() != self.input_dim {
                return Err(TensorError::invalid_argument_op(
                    "TopKRouter::route",
                    &format!("token dim {} != input_dim {}", token.len(), self.input_dim),
                ));
            }
            let logits = matvec_f64(&self.gate_weights, token);
            let probs = softmax_f64(&logits);
            let k = self.top_k.min(self.n_experts).max(1);
            let mut indexed: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top_idx: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
            let raw: Vec<f64> = top_idx.iter().map(|&i| probs[i]).collect();
            let s: f64 = raw.iter().sum();
            let d = if s < f64::EPSILON { 1.0 } else { s };
            expert_indices.push(top_idx);
            gate_weights_out.push(raw.iter().map(|&p| p / d).collect());
            router_logits.push(logits);
        }
        Ok(TopKRouterOutput {
            expert_indices,
            gate_weights: gate_weights_out,
            router_logits,
        })
    }
}

/// Expert-Choice Router — each expert picks its top-C tokens.
#[derive(Debug, Clone)]
pub struct ExpertChoiceRouter {
    pub n_experts: usize,
    pub capacity: usize,
    pub input_dim: usize,
    pub gate_weights: Vec<Vec<f64>>,
}

impl ExpertChoiceRouter {
    pub fn new(n_experts: usize, capacity: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            n_experts,
            capacity,
            input_dim,
            gate_weights: xavier_uniform(n_experts, input_dim, rng),
        }
    }

    /// Returns assignment matrix `[n_experts x batch_size]`: 1.0 if expert selects token.
    pub fn route(&self, tokens: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TensorError> {
        let batch = tokens.len();
        if batch == 0 {
            return Ok(vec![vec![0.0_f64; 0]; self.n_experts]);
        }
        let mut assignment = vec![vec![0.0_f64; batch]; self.n_experts];
        for (e, row_w) in self.gate_weights.iter().enumerate() {
            let scores: Vec<f64> = tokens.iter().map(|t| dot_f64(row_w, t)).collect();
            let probs = softmax_f64(&scores);
            let cap = self.capacity.min(batch).max(1);
            let mut idx_prob: Vec<(usize, f64)> = probs.iter().cloned().enumerate().collect();
            idx_prob
                .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for &(t, _) in idx_prob.iter().take(cap) {
                assignment[e][t] = 1.0;
            }
        }
        Ok(assignment)
    }
}

/// Deterministic hash-based router — no parameters, no training.
#[derive(Debug, Clone)]
pub struct HashRouter {
    pub n_experts: usize,
}

impl HashRouter {
    pub fn new(n_experts: usize) -> Self {
        Self {
            n_experts: n_experts.max(1),
        }
    }

    /// Maps `token_id` to an expert index via multiplicative hashing.
    pub fn route(&self, token_id: usize) -> usize {
        let mixed = token_id
            .wrapping_mul(2_654_435_761)
            .wrapping_add(token_id >> 16);
        mixed % self.n_experts
    }
}

/// Soft MoE Router — weighted sum over ALL experts (no discrete selection).
#[derive(Debug, Clone)]
pub struct SoftMoeRouter {
    pub n_experts: usize,
    pub input_dim: usize,
    pub gate_weights: Vec<Vec<f64>>,
}

impl SoftMoeRouter {
    pub fn new(n_experts: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            n_experts,
            input_dim,
            gate_weights: xavier_uniform(n_experts, input_dim, rng),
        }
    }

    /// Returns softmax convex combination weights `[n_experts]` for input `x`.
    pub fn route(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "SoftMoeRouter::route",
                &format!("x.len() {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        Ok(softmax_f64(&matvec_f64(&self.gate_weights, x)))
    }
}

/// Switch Transformer load-balancing auxiliary loss.
///
/// `loss = α · n_experts · Σ_e f_e · P_e`
#[derive(Debug, Clone)]
pub struct RouterAuxLoss {
    /// Scaling coefficient (typically 1e-2).
    pub alpha: f64,
}

impl RouterAuxLoss {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn compute(
        &self,
        router_logits: &[Vec<f64>],
        top_k_indices: &[Vec<usize>],
    ) -> Result<f64, TensorError> {
        if router_logits.is_empty() {
            return Ok(0.0);
        }
        let n_experts = router_logits[0].len();
        if n_experts == 0 {
            return Ok(0.0);
        }
        let batch = router_logits.len() as f64;
        let probs: Vec<Vec<f64>> = router_logits.iter().map(|l| softmax_f64(l)).collect();
        let mut p_e = vec![0.0_f64; n_experts];
        for tp in &probs {
            for (e, &p) in tp.iter().enumerate() {
                if e < n_experts {
                    p_e[e] += p;
                }
            }
        }
        for p in p_e.iter_mut() {
            *p /= batch;
        }
        let mut counts = vec![0.0_f64; n_experts];
        let mut total = 0.0_f64;
        for ti in top_k_indices {
            for &e in ti {
                if e < n_experts {
                    counts[e] += 1.0;
                    total += 1.0;
                }
            }
        }
        let d = if total < f64::EPSILON { 1.0 } else { total };
        let f_e: Vec<f64> = counts.iter().map(|&c| c / d).collect();
        let sum_fp: f64 = f_e.iter().zip(p_e.iter()).map(|(f, p)| f * p).sum();
        Ok(self.alpha * n_experts as f64 * sum_fp)
    }
}

// ── Section 2: Expert Network Types ──────────────────────────────────────────

/// Standard two-layer MLP expert: `input → hidden (ReLU) → output`.
#[derive(Debug, Clone)]
pub struct FeedForwardExpert {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub(crate) w1: Vec<Vec<f64>>,
    pub(crate) b1: Vec<f64>,
    pub(crate) w2: Vec<Vec<f64>>,
    pub(crate) b2: Vec<f64>,
}

impl FeedForwardExpert {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            w1: xavier_uniform(hidden_dim, input_dim, rng),
            b1: vec![0.0_f64; hidden_dim],
            w2: xavier_uniform(output_dim, hidden_dim, rng),
            b2: vec![0.0_f64; output_dim],
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        if x.len() != self.input_dim {
            return Err(TensorError::invalid_argument_op(
                "FeedForwardExpert::forward",
                &format!("x.len() {} != input_dim {}", x.len(), self.input_dim),
            ));
        }
        let mut h = matvec_f64(&self.w1, x);
        for (hi, &bi) in h.iter_mut().zip(self.b1.iter()) {
            *hi = relu(*hi + bi);
        }
        let mut out = matvec_f64(&self.w2, &h);
        for (oi, &bi) in out.iter_mut().zip(self.b2.iter()) {
            *oi += bi;
        }
        Ok(out)
    }
}

/// Always-active shared expert + sparse expert combo (DeepSeek-MoE style).
#[derive(Debug, Clone)]
pub struct SharedExpert {
    pub shared: FeedForwardExpert,
    pub experts: Vec<FeedForwardExpert>,
    pub router: TopKRouter,
}

impl SharedExpert {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        n_sparse: usize,
        top_k: usize,
        rng: &mut StdRng,
    ) -> Self {
        let ne = n_sparse.max(1);
        Self {
            shared: FeedForwardExpert::new(input_dim, hidden_dim, output_dim, rng),
            experts: (0..ne)
                .map(|_| FeedForwardExpert::new(input_dim, hidden_dim, output_dim, rng))
                .collect(),
            router: TopKRouter::new(ne, top_k, input_dim, rng),
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        let shared_out = self.shared.forward(x)?;
        let routing = self.router.route(&[x.to_vec()])?;
        let mut sparse = vec![0.0_f64; shared_out.len()];
        for (&idx, &w) in routing.expert_indices[0]
            .iter()
            .zip(routing.gate_weights[0].iter())
        {
            let eo = self.experts[idx].forward(x)?;
            for (s, &e) in sparse.iter_mut().zip(eo.iter()) {
                *s += w * e;
            }
        }
        Ok(shared_out
            .iter()
            .zip(sparse.iter())
            .map(|(a, b)| a + b)
            .collect())
    }
}

/// Block-sparse matmul simulation: group tokens by expert, process in batch.
#[derive(Debug, Clone)]
pub struct MegablocksExpert {
    pub experts: Vec<FeedForwardExpert>,
}

impl MegablocksExpert {
    pub fn new(
        n_experts: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            experts: (0..n_experts.max(1))
                .map(|_| FeedForwardExpert::new(input_dim, hidden_dim, output_dim, rng))
                .collect(),
        }
    }

    pub fn forward_batched(
        &self,
        tokens: &[Vec<f64>],
        assignments: &[usize],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        if tokens.len() != assignments.len() {
            return Err(TensorError::invalid_argument_op(
                "MegablocksExpert::forward_batched",
                "tokens and assignments must have the same length",
            ));
        }
        let ne = self.experts.len();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (t, &e) in assignments.iter().enumerate() {
            groups[e.min(ne - 1)].push(t);
        }
        let mut outputs: Vec<Vec<f64>> = tokens.iter().map(|t| vec![0.0; t.len()]).collect();
        for (e, tis) in groups.iter().enumerate() {
            for &t in tis {
                outputs[t] = self.experts[e].forward(&tokens[t])?;
            }
        }
        Ok(outputs)
    }
}

/// Full MoE layer: route → dispatch → experts → aggregate.
#[derive(Debug, Clone)]
pub struct MoeLayer {
    pub router: TopKRouter,
    pub experts: Vec<FeedForwardExpert>,
    pub output_dim: usize,
}

impl MoeLayer {
    pub fn new(
        n_experts: usize,
        top_k: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            router: TopKRouter::new(n_experts, top_k, input_dim, rng),
            experts: (0..n_experts.max(1))
                .map(|_| FeedForwardExpert::new(input_dim, hidden_dim, output_dim, rng))
                .collect(),
            output_dim,
        }
    }

    pub fn forward(&self, tokens: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TensorError> {
        let routing = self.router.route(tokens)?;
        let mut outputs = Vec::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let mut out = vec![0.0_f64; self.output_dim];
            for (&idx, &w) in routing.expert_indices[i]
                .iter()
                .zip(routing.gate_weights[i].iter())
            {
                let eo = self.experts[idx].forward(token)?;
                for (o, &e) in out.iter_mut().zip(eo.iter()) {
                    *o += w * e;
                }
            }
            outputs.push(out);
        }
        Ok(outputs)
    }
}

/// Transformer block where the FFN sublayer is replaced by a MoE layer (pre-norm).
#[derive(Debug, Clone)]
pub struct MoeTransformerBlock {
    pub d_model: usize,
    pub moe: MoeLayer,
    pub ln1_scale: Vec<f64>,
    pub ln2_scale: Vec<f64>,
    attn_w: Vec<Vec<f64>>,
}

impl MoeTransformerBlock {
    pub fn new(
        d_model: usize,
        n_experts: usize,
        top_k: usize,
        ffn_hidden: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            d_model,
            moe: MoeLayer::new(n_experts, top_k, d_model, ffn_hidden, d_model, rng),
            ln1_scale: vec![1.0_f64; d_model],
            ln2_scale: vec![1.0_f64; d_model],
            attn_w: xavier_uniform(d_model, d_model, rng),
        }
    }

    fn layer_norm(&self, x: &[f64], scale: &[f64]) -> Vec<f64> {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let var = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let std = (var + 1e-6).sqrt();
        x.iter()
            .zip(scale.iter())
            .map(|(&xi, &s)| s * (xi - mean) / std)
            .collect()
    }

    pub fn forward(&self, tokens: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TensorError> {
        let mut outputs = Vec::with_capacity(tokens.len());
        for token in tokens {
            if token.len() != self.d_model {
                return Err(TensorError::invalid_argument_op(
                    "MoeTransformerBlock::forward",
                    &format!("token dim {} != d_model {}", token.len(), self.d_model),
                ));
            }
            let n1 = self.layer_norm(token, &self.ln1_scale);
            let ao = matvec_f64(&self.attn_w, &n1);
            let after_attn: Vec<f64> = token.iter().zip(ao.iter()).map(|(a, b)| a + b).collect();
            let n2 = self.layer_norm(&after_attn, &self.ln2_scale);
            let mo = self.moe.forward(&[n2])?;
            let out: Vec<f64> = after_attn
                .iter()
                .zip(mo[0].iter())
                .map(|(a, b)| a + b)
                .collect();
            outputs.push(out);
        }
        Ok(outputs)
    }
}

// ── Section 3: Model Parallelism Utilities ────────────────────────────────────

/// Simulates column/row tensor parallelism split.
#[derive(Debug, Clone)]
pub struct TensorParallelLinear {
    pub weights: Vec<Vec<f64>>,
    pub output_dim: usize,
    pub input_dim: usize,
}

impl TensorParallelLinear {
    pub fn new(output_dim: usize, input_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            weights: xavier_uniform(output_dim, input_dim, rng),
            output_dim,
            input_dim,
        }
    }

    pub fn forward_column(
        &self,
        x: &[f64],
        world_size: usize,
        rank: usize,
    ) -> Result<Vec<f64>, TensorError> {
        let ws = world_size.max(1);
        let shard = (self.output_dim + ws - 1) / ws;
        let start = rank * shard;
        let end = (start + shard).min(self.output_dim);
        if start >= self.output_dim {
            return Ok(Vec::new());
        }
        Ok(matvec_f64(&self.weights[start..end], x))
    }

    pub fn forward_row(
        &self,
        x: &[f64],
        world_size: usize,
        rank: usize,
    ) -> Result<Vec<f64>, TensorError> {
        let ws = world_size.max(1);
        let shard = (self.input_dim + ws - 1) / ws;
        let start = rank * shard;
        let end = (start + shard).min(self.input_dim);
        if start >= self.input_dim {
            return Ok(vec![0.0_f64; self.output_dim]);
        }
        Ok(self
            .weights
            .iter()
            .map(|row| dot_f64(&row[start..end], &x[start..end.min(x.len())]))
            .collect())
    }
}

/// Single pipeline stage: holds layers, forwards micro-batches, tracks state.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub name: String,
    pub layers: Vec<FeedForwardExpert>,
    pub micro_batch_state: Vec<Vec<Vec<f64>>>,
}

impl PipelineStage {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            micro_batch_state: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: FeedForwardExpert) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TensorError> {
        if self.layers.is_empty() {
            let out = x.to_vec();
            self.micro_batch_state.push(out.clone());
            return Ok(out);
        }
        let mut current = x.to_vec();
        for layer in &self.layers {
            let mut next = Vec::with_capacity(current.len());
            for token in &current {
                next.push(layer.forward(token)?);
            }
            current = next;
        }
        self.micro_batch_state.push(current.clone());
        Ok(current)
    }
}

/// Wraps a layer with a simulated gradient checkpointing flag.
#[derive(Debug, Clone)]
pub struct GradientCheckpointLayer {
    pub layer: FeedForwardExpert,
    pub recompute: bool,
}

impl GradientCheckpointLayer {
    pub fn new(layer: FeedForwardExpert, recompute: bool) -> Self {
        Self { layer, recompute }
    }
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        self.layer.forward(x)
    }
}

/// Tracks which activation tensors are offloaded to CPU.
#[derive(Debug, Clone, Default)]
pub struct ActivationOffloader {
    offloaded: std::collections::HashSet<u64>,
    pub offload_count: usize,
    pub prefetch_count: usize,
}

impl ActivationOffloader {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn offload(&mut self, tensor_id: u64) {
        self.offloaded.insert(tensor_id);
        self.offload_count += 1;
    }
    pub fn prefetch(&mut self, tensor_id: u64) {
        self.offloaded.remove(&tensor_id);
        self.prefetch_count += 1;
    }
    pub fn is_offloaded(&self, tensor_id: u64) -> bool {
        self.offloaded.contains(&tensor_id)
    }
}

/// Memory breakdown for a model.
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    pub params_bytes: u64,
    pub gradients_bytes: u64,
    pub optimizer_bytes: u64,
    pub activations_bytes_per_sample: u64,
    pub total_bytes: u64,
}

/// Estimates memory consumption for Transformer + MoE configurations.
#[derive(Debug, Clone, Default)]
pub struct MemoryEstimator;

impl MemoryEstimator {
    pub fn new() -> Self {
        Self
    }

    pub fn estimate_params(
        &self,
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        n_experts: usize,
    ) -> MemoryBreakdown {
        let _ = n_heads;
        let bpe = 4_u64;
        let attn = 4_u64 * (d_model * d_model) as u64;
        let ffn = n_experts as u64 * 2 * d_model as u64 * (4 * d_model as u64);
        let embed = 32_768_u64 * d_model as u64;
        let total_params = embed + n_layers as u64 * (attn + ffn);
        let params_bytes = total_params * bpe;
        let gradients_bytes = params_bytes;
        let optimizer_bytes = 2 * params_bytes;
        let activations_bytes_per_sample = n_layers as u64 * 2048 * d_model as u64 * bpe;
        MemoryBreakdown {
            params_bytes,
            gradients_bytes,
            optimizer_bytes,
            activations_bytes_per_sample,
            total_bytes: params_bytes + gradients_bytes + optimizer_bytes,
        }
    }
}

// ── Section 4: Efficient Attention Variants ───────────────────────────────────

/// O(N) linear attention via ELU+1 kernel approximation.
#[derive(Debug, Clone)]
pub struct LinearAttention {
    pub head_dim: usize,
}

impl LinearAttention {
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    fn feature_map(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        m.iter()
            .map(|row| row.iter().map(|&x| elu_plus_one(x)).collect())
            .collect()
    }

    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let seq = q.len();
        if k.len() != seq || v.len() != seq {
            return Err(TensorError::invalid_argument_op(
                "LinearAttention::forward",
                "q,k,v must match in seq len",
            ));
        }
        let phi_q = self.feature_map(q);
        let phi_k = self.feature_map(k);
        let d = self.head_dim;
        let vd = v.first().map(|r| r.len()).unwrap_or(d);
        let mut ktv = vec![vec![0.0_f64; vd]; d];
        for t in 0..seq {
            for i in 0..d {
                let ki = phi_k[t].get(i).cloned().unwrap_or(0.0);
                for j in 0..vd {
                    ktv[i][j] += ki * v[t][j];
                }
            }
        }
        let k_sum: Vec<f64> = (0..d)
            .map(|i| {
                phi_k
                    .iter()
                    .map(|row| row.get(i).cloned().unwrap_or(0.0))
                    .sum::<f64>()
            })
            .collect();
        let mut output = Vec::with_capacity(seq);
        for t in 0..seq {
            let denom = {
                let d_ = dot_f64(&phi_q[t], &k_sum);
                if d_.abs() < f64::EPSILON {
                    1.0
                } else {
                    d_
                }
            };
            let mut out_t = vec![0.0_f64; vd];
            for i in 0..d {
                let qi = phi_q[t].get(i).cloned().unwrap_or(0.0);
                for j in 0..vd {
                    out_t[j] += qi * ktv[i][j];
                }
            }
            for oj in out_t.iter_mut() {
                *oj /= denom;
            }
            output.push(out_t);
        }
        Ok(output)
    }
}

/// FAVOR+ random feature maps for softmax approximation (Performer attention).
#[derive(Debug, Clone)]
pub struct PerformerAttention {
    pub head_dim: usize,
    pub n_features: usize,
    pub projection: Vec<Vec<f64>>,
}

impl PerformerAttention {
    pub fn new(head_dim: usize, n_features: usize, rng: &mut StdRng) -> Self {
        let projection = (0..n_features)
            .map(|_| {
                (0..head_dim)
                    .map(|_| {
                        let u1: f64 = rng.random::<f64>().max(f64::EPSILON);
                        let u2: f64 = rng.random::<f64>();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    })
                    .collect()
            })
            .collect();
        Self {
            head_dim,
            n_features,
            projection,
        }
    }

    fn random_features(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let scale = 1.0 / (self.n_features as f64).sqrt();
        m.iter()
            .map(|x| {
                let ns: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>();
                self.projection
                    .iter()
                    .map(|w| scale * (dot_f64(w, x) - ns / 2.0).exp())
                    .collect()
            })
            .collect()
    }

    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let seq = q.len();
        if k.len() != seq || v.len() != seq {
            return Err(TensorError::invalid_argument_op(
                "PerformerAttention::forward",
                "q,k,v must match in seq len",
            ));
        }
        let phi_q = self.random_features(q);
        let phi_k = self.random_features(k);
        let r = self.n_features;
        let vd = v.first().map(|r| r.len()).unwrap_or(self.head_dim);
        let mut ktv = vec![vec![0.0_f64; vd]; r];
        for t in 0..seq {
            for i in 0..r {
                let ki = phi_k[t].get(i).cloned().unwrap_or(0.0);
                for j in 0..vd {
                    ktv[i][j] += ki * v[t][j];
                }
            }
        }
        let k_sum: Vec<f64> = (0..r)
            .map(|i| {
                phi_k
                    .iter()
                    .map(|row| row.get(i).cloned().unwrap_or(0.0))
                    .sum::<f64>()
            })
            .collect();
        let mut output = Vec::with_capacity(seq);
        for t in 0..seq {
            let denom = {
                let d_ = dot_f64(&phi_q[t], &k_sum);
                if d_.abs() < f64::EPSILON {
                    1.0
                } else {
                    d_
                }
            };
            let mut out_t = vec![0.0_f64; vd];
            for i in 0..r {
                let qi = phi_q[t].get(i).cloned().unwrap_or(0.0);
                for j in 0..vd {
                    out_t[j] += qi * ktv[i][j];
                }
            }
            for oj in out_t.iter_mut() {
                *oj /= denom;
            }
            output.push(out_t);
        }
        Ok(output)
    }
}

/// Sliding window attention: each token attends to `window_size` nearest neighbours.
#[derive(Debug, Clone)]
pub struct LocalWindowAttention {
    pub head_dim: usize,
    pub window_size: usize,
}

impl LocalWindowAttention {
    pub fn new(head_dim: usize, window_size: usize) -> Self {
        Self {
            head_dim,
            window_size: window_size.max(1),
        }
    }

    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let seq = q.len();
        if k.len() != seq || v.len() != seq {
            return Err(TensorError::invalid_argument_op(
                "LocalWindowAttention::forward",
                "q,k,v must match in seq len",
            ));
        }
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let w = self.window_size;
        let vd = v.first().map(|r| r.len()).unwrap_or(self.head_dim);
        let mut output = Vec::with_capacity(seq);
        for t in 0..seq {
            let start = t.saturating_sub(w);
            let end = (t + w + 1).min(seq);
            let scores: Vec<f64> = (start..end)
                .map(|s| dot_f64(&q[t], &k[s]) * scale)
                .collect();
            let probs = softmax_f64(&scores);
            let mut out_t = vec![0.0_f64; vd];
            for (idx, s) in (start..end).enumerate() {
                for j in 0..vd {
                    out_t[j] += probs[idx] * v[s][j];
                }
            }
            output.push(out_t);
        }
        Ok(output)
    }
}

/// Longformer-style: local window + global tokens attend to all positions.
#[derive(Debug, Clone)]
pub struct LongformerAttention {
    pub head_dim: usize,
    pub window_size: usize,
    pub n_global_tokens: usize,
}

impl LongformerAttention {
    pub fn new(head_dim: usize, window_size: usize, n_global_tokens: usize) -> Self {
        Self {
            head_dim,
            window_size: window_size.max(1),
            n_global_tokens,
        }
    }

    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let seq = q.len();
        if k.len() != seq || v.len() != seq {
            return Err(TensorError::invalid_argument_op(
                "LongformerAttention::forward",
                "q,k,v must match in seq len",
            ));
        }
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let w = self.window_size;
        let ng = self.n_global_tokens.min(seq);
        let vd = v.first().map(|r| r.len()).unwrap_or(self.head_dim);
        let mut output = Vec::with_capacity(seq);
        for t in 0..seq {
            let mut attend: Vec<usize> = (0..ng).collect();
            if t >= ng {
                let start = t.saturating_sub(w);
                let end = (t + w + 1).min(seq);
                for s in start..end {
                    if s >= ng && !attend.contains(&s) {
                        attend.push(s);
                    }
                }
            } else {
                for s in ng..seq {
                    if !attend.contains(&s) {
                        attend.push(s);
                    }
                }
            }
            attend.sort_unstable();
            let scores: Vec<f64> = attend
                .iter()
                .map(|&s| dot_f64(&q[t], &k[s]) * scale)
                .collect();
            let probs = softmax_f64(&scores);
            let mut out_t = vec![0.0_f64; vd];
            for (idx, &s) in attend.iter().enumerate() {
                for j in 0..vd {
                    out_t[j] += probs[idx] * v[s][j];
                }
            }
            output.push(out_t);
        }
        Ok(output)
    }
}

/// BigBird-inspired sparse attention: local ∪ global ∪ random positions.
#[derive(Debug, Clone)]
pub struct SparseAttention {
    pub head_dim: usize,
    pub window_size: usize,
    pub n_global: usize,
    pub n_random: usize,
}

impl SparseAttention {
    pub fn new(head_dim: usize, window_size: usize, n_global: usize, n_random: usize) -> Self {
        Self {
            head_dim,
            window_size: window_size.max(1),
            n_global,
            n_random,
        }
    }

    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f64>>, TensorError> {
        let seq = q.len();
        if k.len() != seq || v.len() != seq {
            return Err(TensorError::invalid_argument_op(
                "SparseAttention::forward",
                "q,k,v must match in seq len",
            ));
        }
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let w = self.window_size;
        let ng = self.n_global.min(seq);
        let vd = v.first().map(|r| r.len()).unwrap_or(self.head_dim);
        let mut output = Vec::with_capacity(seq);
        for t in 0..seq {
            let local_start = t.saturating_sub(w);
            let local_end = (t + w + 1).min(seq);
            let mut attend: Vec<usize> = (local_start..local_end).collect();
            for g in 0..ng {
                if !attend.contains(&g) {
                    attend.push(g);
                }
            }
            for _ in 0..self.n_random {
                let pos = rng.random::<u64>() as usize % seq;
                if !attend.contains(&pos) {
                    attend.push(pos);
                }
            }
            attend.sort_unstable();
            let scores: Vec<f64> = attend
                .iter()
                .map(|&s| dot_f64(&q[t], &k[s]) * scale)
                .collect();
            let probs = softmax_f64(&scores);
            let mut out_t = vec![0.0_f64; vd];
            for (idx, &s) in attend.iter().enumerate() {
                for j in 0..vd {
                    out_t[j] += probs[idx] * v[s][j];
                }
            }
            output.push(out_t);
        }
        Ok(output)
    }
}

// ── Section 5: Quantization-Aware Efficient Inference ────────────────────────

/// INT8 weight quantization, f64 activations.
#[derive(Debug, Clone)]
pub struct WeightOnlyQuant;

impl WeightOnlyQuant {
    pub fn new() -> Self {
        Self
    }

    /// Returns `(quantized, scale, zero_point)` with symmetric INT8 range.
    pub fn quantize_weights(&self, w: &[f64]) -> (Vec<i8>, f64, f64) {
        if w.is_empty() {
            return (Vec::new(), 1.0, 0.0);
        }
        let max_abs = w.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max);
        let scale = if max_abs < f64::EPSILON {
            1.0
        } else {
            max_abs / 127.0
        };
        let quantized: Vec<i8> = w
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        (quantized, scale, 0.0)
    }

    pub fn dequantize(&self, q: &[i8], scale: f64, zero_point: f64) -> Vec<f64> {
        q.iter()
            .map(|&qi| scale * (qi as f64 - zero_point))
            .collect()
    }
}

impl Default for WeightOnlyQuant {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic per-tensor activation quantization.
#[derive(Debug, Clone)]
pub struct ActivationQuant {
    pub bits: u8,
}

impl ActivationQuant {
    pub fn new() -> Self {
        Self { bits: 8 }
    }

    pub fn quantize(&self, activations: &[f64]) -> (Vec<i8>, f64, f64) {
        if activations.is_empty() {
            return (Vec::new(), 1.0, 0.0);
        }
        let min = activations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = activations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let scale = if range < f64::EPSILON {
            1.0
        } else {
            range / 254.0
        };
        let zero_point = (-128.0 - min / scale).round();
        let quantized: Vec<i8> = activations
            .iter()
            .map(|&x| (x / scale + zero_point).round().clamp(-128.0, 127.0) as i8)
            .collect();
        (quantized, scale, zero_point)
    }

    pub fn dequantize(&self, q: &[i8], scale: f64, zero_point: f64) -> Vec<f64> {
        q.iter()
            .map(|&qi| scale * (qi as f64 - zero_point))
            .collect()
    }
}

impl Default for ActivationQuant {
    fn default() -> Self {
        Self::new()
    }
}

/// Precision mode for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerPrecision {
    Fp64,
    Int8WeightOnly,
    Int8Full,
}

/// Layer wrapper that applies the specified quantization on forward.
#[derive(Debug, Clone)]
pub struct MixedPrecisionLayer {
    pub layer: FeedForwardExpert,
    pub precision: LayerPrecision,
    quantized_w1: Vec<Vec<i8>>,
    scale_w1: Vec<f64>,
    quantized_w2: Vec<Vec<i8>>,
    scale_w2: Vec<f64>,
}

impl MixedPrecisionLayer {
    pub fn new(layer: FeedForwardExpert, precision: LayerPrecision) -> Self {
        let q = WeightOnlyQuant::new();
        let mut qw1 = Vec::new();
        let mut sw1 = Vec::new();
        for row in &layer.w1 {
            let (qv, s, _) = q.quantize_weights(row);
            qw1.push(qv);
            sw1.push(s);
        }
        let mut qw2 = Vec::new();
        let mut sw2 = Vec::new();
        for row in &layer.w2 {
            let (qv, s, _) = q.quantize_weights(row);
            qw2.push(qv);
            sw2.push(s);
        }
        Self {
            layer,
            precision,
            quantized_w1: qw1,
            scale_w1: sw1,
            quantized_w2: qw2,
            scale_w2: sw2,
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, TensorError> {
        match self.precision {
            LayerPrecision::Fp64 => self.layer.forward(x),
            LayerPrecision::Int8WeightOnly | LayerPrecision::Int8Full => {
                let q = WeightOnlyQuant::new();
                let w1d: Vec<Vec<f64>> = self
                    .quantized_w1
                    .iter()
                    .zip(self.scale_w1.iter())
                    .map(|(qv, &s)| q.dequantize(qv, s, 0.0))
                    .collect();
                let w2d: Vec<Vec<f64>> = self
                    .quantized_w2
                    .iter()
                    .zip(self.scale_w2.iter())
                    .map(|(qv, &s)| q.dequantize(qv, s, 0.0))
                    .collect();
                let mut h = matvec_f64(&w1d, x);
                for (hi, &bi) in h.iter_mut().zip(self.layer.b1.iter()) {
                    *hi = relu(*hi + bi);
                }
                let mut out = matvec_f64(&w2d, &h);
                for (oi, &bi) in out.iter_mut().zip(self.layer.b2.iter()) {
                    *oi += bi;
                }
                Ok(out)
            }
        }
    }
}

/// INT8 KV cache quantizer to reduce memory.
#[derive(Debug, Clone)]
pub struct KvQuantizer {
    key_quant: WeightOnlyQuant,
    val_quant: WeightOnlyQuant,
    quantized_keys: Vec<(Vec<i8>, f64)>,
    quantized_values: Vec<(Vec<i8>, f64)>,
}

impl KvQuantizer {
    pub fn new() -> Self {
        Self {
            key_quant: WeightOnlyQuant::new(),
            val_quant: WeightOnlyQuant::new(),
            quantized_keys: Vec::new(),
            quantized_values: Vec::new(),
        }
    }

    pub fn store_key(&mut self, key: &[f64]) {
        let (q, s, _) = self.key_quant.quantize_weights(key);
        self.quantized_keys.push((q, s));
    }

    pub fn store_value(&mut self, val: &[f64]) {
        let (q, s, _) = self.val_quant.quantize_weights(val);
        self.quantized_values.push((q, s));
    }

    pub fn get_keys(&self) -> Vec<Vec<f64>> {
        self.quantized_keys
            .iter()
            .map(|(q, s)| self.key_quant.dequantize(q, *s, 0.0))
            .collect()
    }

    pub fn get_values(&self) -> Vec<Vec<f64>> {
        self.quantized_values
            .iter()
            .map(|(q, s)| self.val_quant.dequantize(q, *s, 0.0))
            .collect()
    }

    pub fn cache_len(&self) -> usize {
        self.quantized_keys.len()
    }
}

impl Default for KvQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer type for quantization profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    MoeExpert,
}

/// Quantization configuration assigned to a layer.
#[derive(Debug, Clone)]
pub struct LayerQuantConfig {
    pub layer_id: usize,
    pub kind: LayerKind,
    pub precision: LayerPrecision,
}

/// Profiles layer types and assigns optimal quantization configurations.
#[derive(Debug, Clone)]
pub struct InferenceOptimizer {
    pub configs: Vec<LayerQuantConfig>,
}

impl InferenceOptimizer {
    pub fn new(kinds: &[LayerKind]) -> Self {
        let configs = kinds
            .iter()
            .enumerate()
            .map(|(id, &kind)| {
                let precision = match kind {
                    LayerKind::Embedding | LayerKind::Normalization => LayerPrecision::Fp64,
                    LayerKind::Attention => LayerPrecision::Int8WeightOnly,
                    LayerKind::FeedForward | LayerKind::MoeExpert => LayerPrecision::Int8Full,
                };
                LayerQuantConfig {
                    layer_id: id,
                    kind,
                    precision,
                }
            })
            .collect();
        Self { configs }
    }

    pub fn precision_for(&self, layer_id: usize) -> Option<LayerPrecision> {
        self.configs
            .iter()
            .find(|c| c.layer_id == layer_id)
            .map(|c| c.precision)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn make_qkv(
        seq: usize,
        d: usize,
        rng: &mut StdRng,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mk = |r: &mut StdRng| {
            (0..seq)
                .map(|_| (0..d).map(|_| r.random::<f64>() - 0.5).collect())
                .collect::<Vec<_>>()
        };
        (mk(rng), mk(rng), mk(rng))
    }

    // ── Routing ──

    #[test]
    fn test_top_k_router_output_shape() {
        let out = TopKRouter::new(8, 2, 16, &mut rng(1))
            .route(&vec![vec![0.1_f64; 16]; 4])
            .expect("test value");
        assert_eq!(out.expert_indices.len(), 4);
        assert_eq!(out.expert_indices[0].len(), 2);
        assert_eq!(out.router_logits[0].len(), 8);
    }

    #[test]
    fn test_top_k_router_gate_sum() {
        let router = TopKRouter::new(6, 3, 8, &mut rng(2));
        let tokens: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.1; 8]).collect();
        let out = router.route(&tokens).expect("test value");
        for w in &out.gate_weights {
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_top_k_router_indices_in_range() {
        let n = 5;
        let out = TopKRouter::new(n, 2, 4, &mut rng(3))
            .route(&vec![vec![1.0; 4]; 10])
            .expect("test value");
        for ti in &out.expert_indices {
            for &i in ti {
                assert!(i < n);
            }
        }
    }

    #[test]
    fn test_expert_choice_router() {
        let cap = 2;
        let batch = 6;
        let router = ExpertChoiceRouter::new(4, cap, 8, &mut rng(4));
        let assign = router.route(&vec![vec![0.5_f64; 8]; batch]).expect("test value");
        assert_eq!(assign.len(), 4);
        for row in &assign {
            assert_eq!(row.len(), batch);
            assert!(row.iter().map(|&v| v as usize).sum::<usize>() <= cap);
        }
    }

    #[test]
    fn test_hash_router_deterministic() {
        let router = HashRouter::new(8);
        for id in [0, 1, 100, 999, 42, 7] {
            assert_eq!(router.route(id), router.route(id));
            assert!(router.route(id) < 8);
        }
    }

    #[test]
    fn test_hash_router_all_experts_reachable() {
        let n = 8;
        let router = HashRouter::new(n);
        let seen: std::collections::HashSet<_> = (0..1000).map(|id| router.route(id)).collect();
        assert!(seen.len() >= n / 2);
    }

    #[test]
    fn test_soft_moe_router_convex() {
        let w = SoftMoeRouter::new(6, 8, &mut rng(5))
            .route(&[0.3_f64; 8])
            .expect("test value");
        assert_eq!(w.len(), 6);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(w.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_router_aux_loss_positive() {
        let loss = RouterAuxLoss::new(0.01)
            .compute(
                &[vec![1.0, 0.2, 0.1, 0.5], vec![0.3, 1.0, 0.4, 0.2]],
                &[vec![0, 1], vec![1, 3]],
            )
            .expect("test value");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_router_aux_loss_zero_on_empty() {
        assert_eq!(RouterAuxLoss::new(0.01).compute(&[], &[]).expect("test value"), 0.0);
    }

    // ── Experts ──

    #[test]
    fn test_ff_expert_forward() {
        let out = FeedForwardExpert::new(8, 16, 4, &mut rng(10))
            .forward(&[0.5_f64; 8])
            .expect("test value");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_ff_expert_wrong_input_dim() {
        assert!(FeedForwardExpert::new(8, 16, 4, &mut rng(11))
            .forward(&[0.5_f64; 5])
            .is_err());
    }

    #[test]
    fn test_shared_expert_forward() {
        let out = SharedExpert::new(8, 16, 4, 4, 2, &mut rng(12))
            .forward(&[0.1_f64; 8])
            .expect("test value");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_megablocks_expert_forward() {
        let layer = MegablocksExpert::new(4, 8, 16, 4, &mut rng(13));
        let tokens: Vec<Vec<f64>> = (0..6).map(|_| vec![0.2_f64; 8]).collect();
        let out = layer.forward_batched(&tokens, &[0, 1, 2, 3, 0, 1]).expect("test value");
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_moe_layer_output_shape() {
        let out = MoeLayer::new(8, 2, 16, 32, 16, &mut rng(20))
            .forward(&vec![vec![0.1_f64; 16]; 5])
            .expect("test value");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_moe_layer_token_count() {
        let tokens: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.01; 8]).collect();
        let out = MoeLayer::new(4, 1, 8, 16, 8, &mut rng(21))
            .forward(&tokens)
            .expect("test value");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn test_moe_transformer_block() {
        let block = MoeTransformerBlock::new(16, 4, 2, 32, &mut rng(22));
        let out = block.forward(&vec![vec![0.0_f64; 16]; 4]).expect("test value");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 16);
    }

    // ── Parallelism ──

    #[test]
    fn test_tensor_parallel_column() {
        let layer = TensorParallelLinear::new(8, 4, &mut rng(30));
        let x = vec![1.0_f64; 4];
        let s0 = layer.forward_column(&x, 2, 0).expect("test value");
        let s1 = layer.forward_column(&x, 2, 1).expect("test value");
        assert_eq!(s0.len() + s1.len(), 8);
    }

    #[test]
    fn test_tensor_parallel_row() {
        let layer = TensorParallelLinear::new(4, 8, &mut rng(31));
        let x = vec![0.5_f64; 8];
        assert_eq!(layer.forward_row(&x, 2, 0).expect("test value").len(), 4);
        assert_eq!(layer.forward_row(&x, 2, 1).expect("test value").len(), 4);
    }

    #[test]
    fn test_pipeline_stage_forward() {
        let mut stage = PipelineStage::new("s0");
        stage.add_layer(FeedForwardExpert::new(8, 16, 8, &mut rng(32)));
        let out = stage.forward(&vec![vec![0.2_f64; 8]; 3]).expect("test value");
        assert_eq!(out.len(), 3);
        assert_eq!(stage.micro_batch_state.len(), 1);
    }

    #[test]
    fn test_gradient_checkpoint_layer() {
        let ckpt =
            GradientCheckpointLayer::new(FeedForwardExpert::new(4, 8, 4, &mut rng(33)), true);
        assert!(ckpt.recompute);
        assert_eq!(ckpt.forward(&[1.0, 2.0, 3.0, 4.0]).expect("test value").len(), 4);
    }

    #[test]
    fn test_activation_offloader() {
        let mut off = ActivationOffloader::new();
        off.offload(42);
        assert!(off.is_offloaded(42));
        assert_eq!(off.offload_count, 1);
        off.prefetch(42);
        assert!(!off.is_offloaded(42));
        assert_eq!(off.prefetch_count, 1);
    }

    #[test]
    fn test_memory_estimator_breakdown() {
        let mb = MemoryEstimator::new().estimate_params(12, 768, 12, 8);
        assert!(mb.params_bytes > 0);
        assert_eq!(mb.gradients_bytes, mb.params_bytes);
        assert_eq!(mb.optimizer_bytes, 2 * mb.params_bytes);
        assert!(mb.total_bytes > mb.params_bytes);
        assert!(mb.activations_bytes_per_sample > 0);
    }

    // ── Attention ──

    #[test]
    fn test_linear_attention_output_shape() {
        let (q, k, v) = make_qkv(6, 8, &mut rng(40));
        let out = LinearAttention::new(8).forward(&q, &k, &v).expect("test value");
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_linear_attention_mismatched_seqlen() {
        let (q, k, v) = make_qkv(4, 4, &mut rng(41));
        assert!(LinearAttention::new(4).forward(&q, &k[..3], &v).is_err());
    }

    #[test]
    fn test_performer_attention_shape() {
        let (q, k, v) = make_qkv(5, 8, &mut rng(42));
        let out = PerformerAttention::new(8, 16, &mut rng(42))
            .forward(&q, &k, &v)
            .expect("test value");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_local_window_attention_shape() {
        let (q, k, v) = make_qkv(8, 8, &mut rng(43));
        let out = LocalWindowAttention::new(8, 2).forward(&q, &k, &v).expect("test value");
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_longformer_global_tokens() {
        let (q, k, v) = make_qkv(10, 8, &mut rng(44));
        let out = LongformerAttention::new(8, 2, 2)
            .forward(&q, &k, &v)
            .expect("test value");
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_longformer_zero_global_tokens() {
        let (q, k, v) = make_qkv(6, 4, &mut rng(45));
        let out = LongformerAttention::new(4, 1, 0)
            .forward(&q, &k, &v)
            .expect("test value");
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_sparse_attention_shape() {
        let (q, k, v) = make_qkv(12, 8, &mut rng(46));
        let out = SparseAttention::new(8, 2, 2, 2)
            .forward(&q, &k, &v, &mut rng(46))
            .expect("test value");
        assert_eq!(out.len(), 12);
        assert_eq!(out[0].len(), 8);
    }

    // ── Quantization ──

    #[test]
    fn test_weight_quant_range() {
        let (q, _, _) = WeightOnlyQuant::new().quantize_weights(&[-1.0, 0.5, 0.0, 1.0, -0.5]);
        assert!(q
            .iter()
            .all(|&qi| (qi as i16) >= -127 && (qi as i16) <= 127));
    }

    #[test]
    fn test_weight_quant_dequant_approx() {
        let quant = WeightOnlyQuant::new();
        let w = vec![-0.5_f64, 0.25, 0.75, -0.1];
        let (q, s, zp) = quant.quantize_weights(&w);
        let dq = quant.dequantize(&q, s, zp);
        for (o, r) in w.iter().zip(dq.iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }

    #[test]
    fn test_weight_quant_empty() {
        let (q, s, _) = WeightOnlyQuant::new().quantize_weights(&[]);
        assert!(q.is_empty());
        assert_eq!(s, 1.0);
    }

    #[test]
    fn test_activation_quant_range() {
        let (q, _, _) = ActivationQuant::new().quantize(&[0.0, 1.0, -1.0, 0.5, -0.5]);
        assert!(q
            .iter()
            .all(|&qi| (qi as i16) >= -128 && (qi as i16) <= 127));
    }

    #[test]
    fn test_activation_quant_dequant_approx() {
        let quant = ActivationQuant::new();
        let acts = vec![0.0_f64, 0.5, -0.5, 1.0, -1.0];
        let (q, s, zp) = quant.quantize(&acts);
        let dq = quant.dequantize(&q, s, zp);
        for (o, r) in acts.iter().zip(dq.iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }

    #[test]
    fn test_mixed_precision_layer_forward() {
        let mp = MixedPrecisionLayer::new(
            FeedForwardExpert::new(8, 16, 4, &mut rng(50)),
            LayerPrecision::Int8WeightOnly,
        );
        assert_eq!(mp.forward(&[0.1_f64; 8]).expect("test value").len(), 4);
    }

    #[test]
    fn test_kv_quantizer() {
        let mut kv = KvQuantizer::new();
        let key = vec![0.3_f64, -0.5, 0.8, 1.0];
        let val = vec![0.1_f64, 0.2, -0.3, 0.4];
        kv.store_key(&key);
        kv.store_value(&val);
        assert_eq!(kv.cache_len(), 1);
        let keys = kv.get_keys();
        let vals = kv.get_values();
        for (o, r) in key.iter().zip(keys[0].iter()) {
            assert!((o - r).abs() < 0.02);
        }
        for (o, r) in val.iter().zip(vals[0].iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }

    #[test]
    fn test_kv_quantizer_multiple_tokens() {
        let mut kv = KvQuantizer::new();
        for i in 0..5 {
            kv.store_key(&[i as f64 * 0.1; 4]);
            kv.store_value(&[i as f64 * -0.1; 4]);
        }
        assert_eq!(kv.cache_len(), 5);
    }

    #[test]
    fn test_inference_optimizer_config() {
        let opt = InferenceOptimizer::new(&[
            LayerKind::Embedding,
            LayerKind::Attention,
            LayerKind::FeedForward,
            LayerKind::Normalization,
            LayerKind::MoeExpert,
        ]);
        assert_eq!(opt.precision_for(0), Some(LayerPrecision::Fp64));
        assert_eq!(opt.precision_for(1), Some(LayerPrecision::Int8WeightOnly));
        assert_eq!(opt.precision_for(2), Some(LayerPrecision::Int8Full));
        assert_eq!(opt.precision_for(3), Some(LayerPrecision::Fp64));
        assert_eq!(opt.precision_for(4), Some(LayerPrecision::Int8Full));
        assert_eq!(opt.precision_for(99), None);
    }

    #[test]
    fn test_inference_optimizer_empty() {
        assert!(InferenceOptimizer::new(&[]).configs.is_empty());
    }
}
