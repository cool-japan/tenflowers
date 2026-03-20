//! Extension types for efficient transformers: mixture architectures and training efficiency.

use super::{rand_matrix, relu, softmax_rows, transpose, matmul};
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::RngExt;

type EtResult<T> = Result<T, String>;

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Mixture Architecture
// ─────────────────────────────────────────────────────────────────────────────

/// Switch Transformer: top-1 expert routing with auxiliary load-balancing loss.
pub struct SwitchTransformerLayer {
    n_experts: usize,
    d_model: usize,
    /// Expert weight matrices: each `(d_model × d_model)`.
    experts: Vec<Vec<Vec<f64>>>,
    /// Router weight: `(n_experts × d_model)` — one row per expert.
    w_router: Vec<Vec<f64>>,
}

impl SwitchTransformerLayer {
    /// Create a new Switch Transformer layer.
    pub fn new(n_experts: usize, d_model: usize, seed: u64) -> EtResult<Self> {
        let scale = 0.02;
        let mut rng = StdRng::seed_from_u64(seed);
        let experts: Vec<Vec<Vec<f64>>> = (0..n_experts)
            .map(|_| rand_matrix(&mut rng, d_model, d_model, scale))
            .collect();
        // Shape (n_experts × d_model): each row is the routing vector for one expert.
        let w_router = rand_matrix(&mut rng, n_experts, d_model, scale);
        Ok(Self {
            n_experts,
            d_model,
            experts,
            w_router,
        })
    }

    /// Forward pass through the Switch Transformer layer.
    pub fn forward(&self, x: &[Vec<f64>]) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let mut out = vec![vec![0.0_f64; self.d_model]; n];
        for (i, xi) in x.iter().enumerate() {
            // Router logits: (n_experts,)
            let logits: Vec<f64> = self
                .w_router
                .iter()
                .map(|row| {
                    let len = row.len().min(xi.len());
                    row[..len]
                        .iter()
                        .zip(xi[..len].iter())
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect();
            // Top-1 expert index (clamped to valid range)
            let expert_id = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
                .min(self.n_experts.saturating_sub(1));
            // Apply selected expert
            let w_e = &self.experts[expert_id];
            out[i] = w_e
                .iter()
                .map(|row| {
                    let len = row.len().min(xi.len());
                    relu(
                        row[..len]
                            .iter()
                            .zip(xi[..len].iter())
                            .map(|(a, b)| a * b)
                            .sum(),
                    )
                })
                .collect();
        }
        Ok(out)
    }
}

/// Mixture-of-Depths: route top-`capacity` tokens to a computation layer.
pub struct MixtureOfDepthsLayer {
    d_model: usize,
    w_router: Vec<f64>,         // (d_model,) scalar router
    w_transform: Vec<Vec<f64>>, // (d_model × d_model)
}

impl MixtureOfDepthsLayer {
    /// Create a new Mixture-of-Depths layer.
    pub fn new(d_model: usize, seed: u64) -> EtResult<Self> {
        let scale = 0.02;
        let mut rng = StdRng::seed_from_u64(seed);
        let w_router: Vec<f64> = (0..d_model)
            .map(|_| {
                use scirs2_core::random::Rng;
                (rng.random::<f64>() * 2.0 - 1.0) * scale
            })
            .collect();
        let w_transform = rand_matrix(&mut rng, d_model, d_model, scale);
        Ok(Self {
            d_model,
            w_router,
            w_transform,
        })
    }

    /// Route top `floor(capacity_factor * N)` tokens through the transform.
    pub fn forward(&self, x: &[Vec<f64>], capacity_factor: f64) -> EtResult<Vec<Vec<f64>>> {
        let n = x.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let capacity = ((capacity_factor * n as f64).ceil() as usize).min(n).max(1);
        // Router score per token: scalar dot product
        let scores: Vec<f64> = x
            .iter()
            .map(|xi| {
                let len = self.w_router.len().min(xi.len());
                self.w_router[..len]
                    .iter()
                    .zip(xi[..len].iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        // Top-capacity indices by score
        let mut indexed: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: std::collections::HashSet<usize> =
            indexed[..capacity].iter().map(|(i, _)| *i).collect();
        let mut out = x.to_vec();
        for i in 0..n {
            if selected.contains(&i) {
                let xi = &x[i];
                out[i] = self
                    .w_transform
                    .iter()
                    .map(|row| {
                        let len = row.len().min(xi.len());
                        relu(
                            row[..len]
                                .iter()
                                .zip(xi[..len].iter())
                                .map(|(a, b)| a * b)
                                .sum(),
                        )
                    })
                    .collect();
            }
        }
        Ok(out)
    }
}

/// Hydra Attention: single shared Q over multiple K,V heads.
pub struct HydraAttention {
    n_heads: usize,
    d_model: usize,
    d_head: usize,
}

impl HydraAttention {
    /// Create a new Hydra Attention with `n_heads` heads and `d_model` dimensions.
    pub fn new(n_heads: usize, d_model: usize) -> EtResult<Self> {
        if d_model % n_heads != 0 {
            return Err("HydraAttention: d_model must be divisible by n_heads".into());
        }
        Ok(Self {
            n_heads,
            d_model,
            d_head: d_model / n_heads,
        })
    }

    /// Forward: Q_shared `(N × d_model)`, K_heads/V_heads `(n_heads × N × d_head)` → `(N × d_model)`.
    pub fn forward(
        &self,
        q_shared: &[Vec<f64>],
        k_heads: &[Vec<Vec<f64>>],
        v_heads: &[Vec<Vec<f64>>],
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q_shared.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let n_heads = k_heads.len().min(v_heads.len()).min(self.n_heads);
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_heads);
        for h in 0..n_heads {
            let k_h = &k_heads[h];
            let v_h = &v_heads[h];
            // Extract Q slice for head h
            let q_h: Vec<Vec<f64>> = q_shared
                .iter()
                .map(|row| {
                    let start = h * self.d_head;
                    let end = (start + self.d_head).min(row.len());
                    row[start..end].to_vec()
                })
                .collect();
            // Scores: Q_h · K_h^T / sqrt(d_head)
            let k_ht = transpose(k_h);
            let mut scores = matmul(&q_h, &k_ht)?;
            let scale = 1.0 / (self.d_head as f64).sqrt();
            for row in scores.iter_mut() {
                for s in row.iter_mut() {
                    *s *= scale;
                }
            }
            softmax_rows(&mut scores);
            let ctx = matmul(&scores, v_h)?;
            head_outputs.push(ctx);
        }
        // Concatenate head outputs
        let mut out = vec![vec![0.0_f64; self.d_model]; n];
        for h in 0..n_heads {
            let start = h * self.d_head;
            for i in 0..n {
                for j in 0..head_outputs[h][i].len().min(self.d_head) {
                    if start + j < self.d_model {
                        out[i][start + j] = head_outputs[h][i][j];
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Multi-Query Attention: all heads share a single K and V.
pub struct MultiQueryAttention {
    n_heads: usize,
    d_model: usize,
    d_head: usize,
}

impl MultiQueryAttention {
    /// Create a new Multi-Query Attention.
    pub fn new(n_heads: usize, d_model: usize) -> EtResult<Self> {
        if d_model % n_heads != 0 {
            return Err("MultiQueryAttention: d_model must be divisible by n_heads".into());
        }
        Ok(Self {
            n_heads,
            d_model,
            d_head: d_model / n_heads,
        })
    }

    /// Forward: Q `(N × d_model)`, K_single/V_single `(N × d_head)` → `(N × d_model)`.
    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k_single: &[Vec<f64>],
        v_single: &[Vec<f64>],
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let k_t = transpose(k_single);
        let mut out = vec![vec![0.0_f64; self.d_model]; n];
        for h in 0..self.n_heads {
            let start = h * self.d_head;
            let end = (start + self.d_head).min(self.d_model);
            let q_h: Vec<Vec<f64>> = q.iter().map(|r| r[start..end].to_vec()).collect();
            let mut scores = matmul(&q_h, &k_t)?;
            for row in scores.iter_mut() {
                for s in row.iter_mut() {
                    *s *= scale;
                }
            }
            softmax_rows(&mut scores);
            let ctx = matmul(&scores, v_single)?;
            for i in 0..n {
                for j in 0..ctx[i].len().min(self.d_head) {
                    if start + j < self.d_model {
                        out[i][start + j] = ctx[i][j];
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Grouped-Query Attention: G groups of K,V for H heads (H = n*G).
pub struct GroupedQueryAttention {
    n_heads: usize,
    n_groups: usize,
    d_model: usize,
    d_head: usize,
}

impl GroupedQueryAttention {
    /// Create a new Grouped-Query Attention.
    pub fn new(n_heads: usize, n_groups: usize, d_model: usize) -> EtResult<Self> {
        if n_heads == 0 || n_groups == 0 {
            return Err("GQA: n_heads and n_groups must be > 0".into());
        }
        if n_heads % n_groups != 0 {
            return Err("GQA: n_heads must be divisible by n_groups".into());
        }
        if d_model % n_heads != 0 {
            return Err("GQA: d_model must be divisible by n_heads".into());
        }
        Ok(Self {
            n_heads,
            n_groups,
            d_model,
            d_head: d_model / n_heads,
        })
    }

    /// Forward: Q `(N × d_model)`, K_groups/V_groups `(n_groups × N × d_head)` → `(N × d_model)`.
    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k_groups: &[Vec<Vec<f64>>],
        v_groups: &[Vec<Vec<f64>>],
    ) -> EtResult<Vec<Vec<f64>>> {
        let n = q.len();
        if n == 0 {
            return Ok(vec![]);
        }
        let heads_per_group = self.n_heads / self.n_groups;
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let mut out = vec![vec![0.0_f64; self.d_model]; n];
        for g in 0..self.n_groups.min(k_groups.len()).min(v_groups.len()) {
            let k_g = &k_groups[g];
            let v_g = &v_groups[g];
            let k_gt = transpose(k_g);
            for h_local in 0..heads_per_group {
                let h = g * heads_per_group + h_local;
                let start = h * self.d_head;
                let end = (start + self.d_head).min(self.d_model);
                let q_h: Vec<Vec<f64>> = q.iter().map(|r| r[start..end].to_vec()).collect();
                let mut scores = matmul(&q_h, &k_gt)?;
                for row in scores.iter_mut() {
                    for s in row.iter_mut() {
                        *s *= scale;
                    }
                }
                softmax_rows(&mut scores);
                let ctx = matmul(&scores, v_g)?;
                for i in 0..n {
                    for j in 0..ctx[i].len().min(self.d_head) {
                        if start + j < self.d_model {
                            out[i][start + j] = ctx[i][j];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Training Efficiency
// ─────────────────────────────────────────────────────────────────────────────

/// Top-k gradient sparsification for gradient compression.
pub struct GradientCompressor;

impl GradientCompressor {
    /// Create a new gradient compressor.
    pub fn new() -> Self {
        Self
    }

    /// Keep the top `ceil(k_ratio * len)` gradient values by magnitude.
    /// Returns `(values, indices)` sorted by descending magnitude.
    pub fn compress(grad: &[f64], k_ratio: f64) -> (Vec<f64>, Vec<usize>) {
        let k = ((k_ratio * grad.len() as f64).ceil() as usize)
            .max(1)
            .min(grad.len());
        let mut indexed: Vec<(usize, f64)> = grad.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k = &indexed[..k];
        let values: Vec<f64> = top_k.iter().map(|(_, v)| *v).collect();
        let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        (values, indices)
    }

    /// Reconstruct the full gradient from sparse `(values, indices)`.
    pub fn decompress(values: &[f64], indices: &[usize], size: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; size];
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < size {
                out[idx] = val;
            }
        }
        out
    }
}

impl Default for GradientCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Dynamic loss scaling with overflow detection (mixed-precision training).
pub struct MixedPrecisionScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    consecutive_no_overflow: usize,
}

impl MixedPrecisionScaler {
    /// Create a new mixed precision scaler.
    pub fn new(
        init_scale: f64,
        growth_factor: f64,
        backoff_factor: f64,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            consecutive_no_overflow: 0,
        }
    }

    /// Scale a loss value by the current loss scale.
    pub fn scale(&self, loss: f64) -> f64 {
        loss * self.scale
    }

    /// Update the loss scale based on overflow status.
    /// Returns the new scale.
    pub fn update(&mut self, is_overflow: bool) -> f64 {
        if is_overflow {
            self.scale *= self.backoff_factor;
            self.consecutive_no_overflow = 0;
        } else {
            self.consecutive_no_overflow += 1;
            if self.consecutive_no_overflow >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.consecutive_no_overflow = 0;
            }
        }
        self.scale
    }

    /// Current scale value.
    pub fn current_scale(&self) -> f64 {
        self.scale
    }
}

/// Activation checkpointing manager: decide which layers to recompute.
///
/// Strategy: recompute layers if their cumulative memory exceeds budget.
pub struct ActivationCheckpointingMgr {
    /// Estimated memory cost per layer (arbitrary units).
    layer_memory: Vec<f64>,
    memory_budget: f64,
}

impl ActivationCheckpointingMgr {
    /// Create a new activation checkpointing manager.
    pub fn new(layer_memory: Vec<f64>, memory_budget: f64) -> Self {
        Self {
            layer_memory,
            memory_budget,
        }
    }

    /// Returns true if layer `layer_id` should be recomputed given `memory_budget`.
    pub fn should_recompute(&self, layer_id: usize, memory_budget: f64) -> bool {
        if layer_id >= self.layer_memory.len() {
            return false;
        }
        // Recompute if cumulative memory up to this layer exceeds budget
        let cumulative: f64 = self.layer_memory[..=layer_id].iter().sum();
        cumulative > memory_budget
    }
}

/// 1F1B pipeline schedule actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineAction {
    /// Forward pass action.
    Forward,
    /// Backward pass action.
    Backward,
    /// Idle (no operation).
    Idle,
}

/// A schedule entry: (stage_index, action).
pub type PipelineScheduleEntry = (usize, PipelineAction);

/// Generates a 1F1B (one-forward-one-backward) pipeline schedule.
pub struct PipelineScheduler;

impl PipelineScheduler {
    /// Create a new pipeline scheduler.
    pub fn new() -> Self {
        Self
    }

    /// Generate a 1F1B schedule for `n_stages` stages and `n_microbatches` microbatches.
    ///
    /// Returns a list of (stage, action) pairs in temporal order per stage.
    pub fn schedule(n_stages: usize, n_microbatches: usize) -> Vec<PipelineScheduleEntry> {
        let mut schedule = Vec::new();
        // Warm-up phase: each stage runs forward passes
        for mb in 0..n_stages.min(n_microbatches) {
            for stage in 0..=mb {
                schedule.push((stage, PipelineAction::Forward));
            }
        }
        // Steady state: 1F1B
        for mb in n_stages..n_microbatches {
            for stage in 0..n_stages {
                schedule.push((stage, PipelineAction::Forward));
                if mb >= n_stages {
                    schedule.push((stage, PipelineAction::Backward));
                }
            }
        }
        // Drain phase: backward passes
        for mb in 0..n_stages.min(n_microbatches) {
            for stage in (0..=mb).rev() {
                schedule.push((stage, PipelineAction::Backward));
            }
        }
        schedule
    }
}

impl Default for PipelineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// ZeRO-3 simulator: partition optimizer states, gradients, and params across ranks.
pub struct ZeroRedundancyOptimizer {
    world_size: usize,
}

impl ZeroRedundancyOptimizer {
    /// Create a new ZeRO redundancy optimizer.
    pub fn new(world_size: usize) -> EtResult<Self> {
        if world_size == 0 {
            return Err("ZeRO: world_size must be > 0".into());
        }
        Ok(Self { world_size })
    }

    /// Gather full parameter vector from rank-partitioned shards.
    ///
    /// Each rank holds `ceil(len / world_size)` parameters.
    /// Returns the full parameter vector owned by `rank`.
    pub fn gather_params(&self, rank: usize, world_size: usize, params: &[f64]) -> Vec<f64> {
        let shard_size = (params.len() + world_size - 1) / world_size;
        let start = rank * shard_size;
        let end = (start + shard_size).min(params.len());
        params[start..end].to_vec()
    }

    /// Scatter (partition) params across `world_size` ranks, return shard for `rank`.
    pub fn scatter_params(&self, rank: usize, params: &[f64]) -> Vec<f64> {
        self.gather_params(rank, self.world_size, params)
    }

    /// All-gather: given per-rank shard, reconstruct full buffer.
    pub fn all_gather(shards: &[Vec<f64>]) -> Vec<f64> {
        shards.iter().flat_map(|s| s.iter().cloned()).collect()
    }
}
