//! Advanced Continual & Lifelong Learning — Track Z + Extensions.
//!
//! GEM/A-GEM, replay-based, architecture-based, regularization-based methods,
//! evaluation metrics (BWT, FWT, AIA, forgetting, plasticity/stability), and
//! complementary lifelong learning extensions:
//!
//! * [`LllGemModel`] — Gradient Episodic Memory (Lopez-Paz 2017)
//! * [`LllAGemModel`] — A-GEM (Chaudhry 2019): averaged episodic memory
//! * [`LllER`] — Experience Replay with reservoir sampling (Vitter 1985)
//! * [`LllDer`] — Dark Experience Replay (Buzzega 2020)
//! * [`LllCoPE`] — Continual Prototype Evolution (De Lange 2021)
//! * [`LllHAT`] — Hard Attention to the Task (Serra 2018)
//! * [`LllTaskOracle`] — Task boundary detection via changepoint detection
//! * [`LllMetrics`] — Lifelong learning evaluation (AA/BT/FT/forgetting)

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ── §1  Gradient Episodic Memory ─────────────────────────────────────────────

/// Circular ring-buffer of `(input, gradient)` pairs per task for GEM.
#[derive(Debug, Clone)]
pub struct EpisodeMemory {
    memories: HashMap<usize, Vec<(Vec<f64>, Vec<f64>)>>,
    capacity: usize,
    heads: HashMap<usize, usize>,
}

impl EpisodeMemory {
    /// Create a new memory store with the given per-task capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            memories: HashMap::new(),
            capacity,
            heads: HashMap::new(),
        }
    }
    /// Add a batch of (input, gradient) samples for `task_id` (ring-buffer semantics).
    pub fn add_task_memory(&mut self, task_id: usize, samples: Vec<(Vec<f64>, Vec<f64>)>) {
        let buf = self.memories.entry(task_id).or_default();
        let head = self.heads.entry(task_id).or_insert(0);
        for sample in samples {
            if buf.len() < self.capacity {
                buf.push(sample);
            } else {
                let idx = *head % self.capacity;
                buf[idx] = sample;
            }
            *head = head.wrapping_add(1);
        }
    }
    /// Return all stored pairs for `task_id` (empty slice if none).
    pub fn get_task_memories(&self, task_id: usize) -> &[(Vec<f64>, Vec<f64>)] {
        self.memories
            .get(&task_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    /// Number of tasks with stored memories.
    pub fn num_tasks(&self) -> usize {
        self.memories.len()
    }
    /// Sorted task ids.
    pub fn task_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.memories.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

/// GEM gradient projection via Dykstra's alternating-projection algorithm.
///
/// Projects `grad` onto C = { g : ⟨g, g_k⟩ ≥ 0 ∀k }.
#[derive(Debug, Clone)]
pub struct GemConstraint {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for GemConstraint {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
        }
    }
}

impl GemConstraint {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Project `grad` so each inner product with a memory gradient is ≥ −`eps`.
    pub fn project_gradient(
        &self,
        grad: &[f64],
        memories: &[Vec<f64>],
        eps: f64,
    ) -> Result<Vec<f64>> {
        if memories.is_empty() {
            return Ok(grad.to_vec());
        }
        let dim = grad.len();
        for (i, m) in memories.iter().enumerate() {
            if m.len() != dim {
                return Err(TensorError::invalid_argument_op(
                    "GemConstraint::project_gradient",
                    &format!("memory {} len {} != grad len {}", i, m.len(), dim),
                ));
            }
        }
        let mut g = grad.to_vec();
        let n_mem = memories.len();
        let mut p: Vec<Vec<f64>> = vec![vec![0.0; dim]; n_mem];
        for _iter in 0..self.max_iter {
            let g_old = g.clone();
            for k in 0..n_mem {
                let ref_g = &memories[k];
                let y: Vec<f64> = (0..dim).map(|d| g[d] + p[k][d]).collect();
                let dot: f64 = y.iter().zip(ref_g.iter()).map(|(a, b)| a * b).sum();
                if dot < -eps {
                    let norm_sq: f64 = ref_g.iter().map(|x| x * x).sum();
                    if norm_sq > 1e-30 {
                        let coeff = (dot + eps) / norm_sq;
                        let y_proj: Vec<f64> = (0..dim).map(|d| y[d] - coeff * ref_g[d]).collect();
                        for d in 0..dim {
                            p[k][d] = y[d] - y_proj[d];
                        }
                        g = y_proj;
                    } else {
                        g = y;
                    }
                } else {
                    for d in 0..dim {
                        p[k][d] = y[d] - g[d];
                    }
                }
            }
            let delta: f64 = g
                .iter()
                .zip(g_old.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if delta < self.tol {
                break;
            }
        }
        Ok(g)
    }
}

/// A-GEM: single-step projection using an average reference gradient.
#[derive(Debug, Clone)]
pub struct AgemOptimizer {
    pub reference_size: usize,
}

impl AgemOptimizer {
    pub fn new(reference_size: usize) -> Self {
        Self { reference_size }
    }

    /// Average gradient from a random subset of memories.
    pub fn compute_reference(
        &self,
        memories: &[(Vec<f64>, Vec<f64>)],
        rng: &mut StdRng,
    ) -> Option<Vec<f64>> {
        if memories.is_empty() {
            return None;
        }
        let dim = memories[0].1.len();
        let n = memories.len();
        let k = self.reference_size.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j: usize = i + (rng.random::<u64>() as usize % (n - i));
            indices.swap(i, j);
        }
        let mut ref_grad = vec![0.0f64; dim];
        for &idx in &indices[..k] {
            let grad = &memories[idx].1;
            for d in 0..grad.len().min(dim) {
                ref_grad[d] += grad[d];
            }
        }
        let scale = 1.0 / k as f64;
        for v in &mut ref_grad {
            *v *= scale;
        }
        Some(ref_grad)
    }

    /// Project `grad` to satisfy ⟨grad, ref_grad⟩ ≥ 0 (single A-GEM step).
    pub fn project(&self, grad: &[f64], ref_grad: &[f64]) -> Vec<f64> {
        let dot: f64 = grad.iter().zip(ref_grad.iter()).map(|(a, b)| a * b).sum();
        if dot >= 0.0 {
            return grad.to_vec();
        }
        let norm_sq: f64 = ref_grad.iter().map(|x| x * x).sum();
        if norm_sq < 1e-30 {
            return grad.to_vec();
        }
        let coeff = dot / norm_sq;
        grad.iter()
            .zip(ref_grad.iter())
            .map(|(g, r)| g - coeff * r)
            .collect()
    }
}

/// Per-task gradient store with ring-buffer semantics and uniform past sampling.
#[derive(Debug, Clone)]
pub struct TaskGradientStore {
    gradients: HashMap<usize, Vec<Vec<f64>>>,
    capacity: usize,
    heads: HashMap<usize, usize>,
}

impl TaskGradientStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            gradients: HashMap::new(),
            capacity,
            heads: HashMap::new(),
        }
    }
    pub fn add_gradient(&mut self, task_id: usize, grad: Vec<f64>) {
        let buf = self.gradients.entry(task_id).or_default();
        let head = self.heads.entry(task_id).or_insert(0);
        if buf.len() < self.capacity {
            buf.push(grad);
        } else {
            let idx = *head % self.capacity;
            buf[idx] = grad;
        }
        *head = head.wrapping_add(1);
    }
    /// Sample `n` gradients uniformly at random from all past tasks.
    pub fn sample_past(&self, n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let all: Vec<&Vec<f64>> = self.gradients.values().flat_map(|v| v.iter()).collect();
        if all.is_empty() {
            return Vec::new();
        }
        (0..n)
            .map(|_| all[rng.random::<u64>() as usize % all.len()].clone())
            .collect()
    }
    pub fn total_stored(&self) -> usize {
        self.gradients.values().map(|v| v.len()).sum()
    }
}

/// Full GEM training helper: project current gradient, then apply to params.
#[derive(Debug, Clone)]
pub struct GemTrainer {
    pub constraint: GemConstraint,
    pub lr: f64,
}

impl GemTrainer {
    pub fn new(lr: f64) -> Self {
        Self {
            constraint: GemConstraint::default(),
            lr,
        }
    }

    /// Project gradient via GEM and update params in-place; returns projected gradient.
    pub fn step(
        &self,
        current_grad: &[f64],
        params: &mut [f64],
        memory: &EpisodeMemory,
    ) -> Result<Vec<f64>> {
        let all_mem_grads: Vec<Vec<f64>> = memory
            .task_ids()
            .into_iter()
            .flat_map(|tid| memory.get_task_memories(tid).iter().map(|(_, g)| g.clone()))
            .collect();
        let projected = self
            .constraint
            .project_gradient(current_grad, &all_mem_grads, 1e-7)?;
        if params.len() == projected.len() {
            for (p, g) in params.iter_mut().zip(projected.iter()) {
                *p -= self.lr * g;
            }
        }
        Ok(projected)
    }
}

// ── §2  Replay-Based Methods ──────────────────────────────────────────────────

/// Experience replay: stores (x, y, task_id) with per-task balanced sampling.
#[derive(Debug, Clone)]
pub struct ExperienceReplay {
    store: HashMap<usize, Vec<(Vec<f64>, Vec<f64>)>>,
    capacity_per_task: usize,
    heads: HashMap<usize, usize>,
}

impl ExperienceReplay {
    /// Create. `capacity_per_task = 0` means unlimited.
    pub fn new(capacity_per_task: usize) -> Self {
        Self {
            store: HashMap::new(),
            capacity_per_task,
            heads: HashMap::new(),
        }
    }
    pub fn add(&mut self, x: Vec<f64>, y: Vec<f64>, task_id: usize) {
        let buf = self.store.entry(task_id).or_default();
        let head = self.heads.entry(task_id).or_insert(0);
        if self.capacity_per_task == 0 {
            buf.push((x, y));
        } else if buf.len() < self.capacity_per_task {
            buf.push((x, y));
        } else {
            let idx = *head % self.capacity_per_task;
            buf[idx] = (x, y);
        }
        *head = head.wrapping_add(1);
    }
    /// Sample `n_per_task` examples per task (with replacement if needed).
    pub fn sample_balanced(
        &self,
        n_per_task: usize,
        rng: &mut StdRng,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut out = Vec::new();
        let mut task_ids: Vec<usize> = self.store.keys().copied().collect();
        task_ids.sort_unstable();
        for tid in task_ids {
            if let Some(buf) = self.store.get(&tid) {
                if buf.is_empty() {
                    continue;
                }
                for _ in 0..n_per_task {
                    out.push(buf[rng.random::<u64>() as usize % buf.len()].clone());
                }
            }
        }
        out
    }
    pub fn num_tasks(&self) -> usize {
        self.store.len()
    }
    pub fn total_examples(&self) -> usize {
        self.store.values().map(|v| v.len()).sum()
    }
}

/// Dark Experience Replay: store logits alongside inputs for distillation.
#[derive(Debug, Clone)]
pub struct DarkExperienceReplay {
    store: HashMap<usize, Vec<(Vec<f64>, Vec<f64>)>>,
    capacity_per_task: usize,
    heads: HashMap<usize, usize>,
}

impl DarkExperienceReplay {
    pub fn new(capacity_per_task: usize) -> Self {
        Self {
            store: HashMap::new(),
            capacity_per_task,
            heads: HashMap::new(),
        }
    }
    pub fn add(&mut self, input: Vec<f64>, logits: Vec<f64>, task_id: usize) {
        let buf = self.store.entry(task_id).or_default();
        let head = self.heads.entry(task_id).or_insert(0);
        if self.capacity_per_task == 0 {
            buf.push((input, logits));
        } else if buf.len() < self.capacity_per_task {
            buf.push((input, logits));
        } else {
            let idx = *head % self.capacity_per_task;
            buf[idx] = (input, logits);
        }
        *head = head.wrapping_add(1);
    }
    /// DER loss: α · MSE(model_out, stored_logits).
    pub fn der_loss(model_out: &[f64], stored_logits: &[f64], alpha: f64) -> f64 {
        let n = model_out.len().min(stored_logits.len());
        if n == 0 {
            return 0.0;
        }
        let mse: f64 = (0..n)
            .map(|i| (model_out[i] - stored_logits[i]).powi(2))
            .sum::<f64>()
            / n as f64;
        alpha * mse
    }
    pub fn sample(&self, task_id: usize, rng: &mut StdRng) -> Option<&(Vec<f64>, Vec<f64>)> {
        let buf = self.store.get(&task_id)?;
        if buf.is_empty() {
            return None;
        }
        Some(&buf[rng.random::<u64>() as usize % buf.len()])
    }
    pub fn total_examples(&self) -> usize {
        self.store.values().map(|v| v.len()).sum()
    }
}

/// Generative Replay: Gaussian model per task, generates samples via Box-Muller.
#[derive(Debug, Clone)]
pub struct GenerativeReplay {
    models: HashMap<usize, (Vec<f64>, Vec<f64>)>,
}

impl Default for GenerativeReplay {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerativeReplay {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    pub fn register_model(&mut self, task_id: usize, mean: Vec<f64>, std: Vec<f64>) {
        self.models.insert(task_id, (mean, std));
    }
    pub fn generate_samples(
        &self,
        task_id: usize,
        n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f64>>> {
        let (mean, std) = self.models.get(&task_id).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "GenerativeReplay::generate_samples",
                &format!("no model for task {}", task_id),
            )
        })?;
        let dim = mean.len();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let mut sample = Vec::with_capacity(dim);
            for d in 0..dim {
                let u1: f64 = rng.random::<f64>().clamp(1e-15, 1.0);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                sample.push(mean[d] + std[d].abs() * z);
            }
            samples.push(sample);
        }
        Ok(samples)
    }
    pub fn num_models(&self) -> usize {
        self.models.len()
    }
}

/// MAG: scale gradient by importance × |θ − θ_anchor|.
#[derive(Debug, Clone)]
pub struct MemoryAwareGradients;

impl MemoryAwareGradients {
    pub fn compute_mag_grad(
        grad: &[f64],
        params: &[f64],
        anchor_params: &[f64],
        importance: &[f64],
    ) -> Vec<f64> {
        let dim = grad
            .len()
            .min(params.len())
            .min(anchor_params.len())
            .min(importance.len());
        (0..dim)
            .map(|k| grad[k] * importance[k] * (params[k] - anchor_params[k]).abs())
            .collect()
    }
}

/// Ring-buffer with reservoir sampling.
#[derive(Debug, Clone)]
pub struct ReplayBuffer<T: Clone> {
    buffer: Vec<T>,
    capacity: usize,
    total_seen: usize,
}

impl<T: Clone + std::fmt::Debug> ReplayBuffer<T> {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "ReplayBuffer::new",
                "capacity must be > 0",
            ));
        }
        Ok(Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
        })
    }
    /// Reservoir-sampling update: each item survives with probability `capacity / total_seen`.
    pub fn reservoir_update(&mut self, new_item: T, rng: &mut StdRng) {
        self.total_seen += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push(new_item);
        } else {
            let threshold = self.capacity as f64 / self.total_seen as f64;
            if rng.random::<f64>() < threshold {
                let idx = rng.random::<u64>() as usize % self.capacity;
                self.buffer[idx] = new_item;
            }
        }
    }
    pub fn items(&self) -> &[T] {
        &self.buffer
    }
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    pub fn total_seen(&self) -> usize {
        self.total_seen
    }
}

// ── §3  Architecture-Based Methods ───────────────────────────────────────────

/// Progressive Neural Networks: grow a new column per task, optional lateral connections.
#[derive(Debug, Clone)]
pub struct ProgressiveGrowth {
    columns: Vec<Vec<usize>>,
    pub lateral_connections: bool,
    pub input_dim: usize,
}

impl ProgressiveGrowth {
    pub fn new(input_dim: usize, lateral_connections: bool) -> Self {
        Self {
            columns: Vec::new(),
            lateral_connections,
            input_dim,
        }
    }
    pub fn expand(&mut self, _new_task_id: usize, layer_widths: Vec<usize>) {
        self.columns.push(layer_widths);
    }
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }
    /// Effective input width for a given layer accounting for lateral connections.
    pub fn effective_input_width(&self, layer_idx: usize) -> usize {
        if !self.lateral_connections || self.columns.len() < 2 {
            return self.input_dim;
        }
        let n_prev = self.columns.len() - 1;
        let lateral: usize = self.columns[..n_prev]
            .iter()
            .map(|col| col.get(layer_idx).copied().unwrap_or(0))
            .sum();
        self.input_dim + lateral
    }
}

/// PackNet binary weight masking: magnitude-prune per task and freeze.
#[derive(Debug, Clone)]
pub struct PackNetMasker {
    masks: HashMap<usize, Vec<f64>>,
    pub num_params: usize,
}

impl PackNetMasker {
    pub fn new(num_params: usize) -> Self {
        Self {
            masks: HashMap::new(),
            num_params,
        }
    }

    /// Build a binary mask with `sparsity` fraction zeroed (smallest magnitudes).
    pub fn prune_for_task(&mut self, task_id: usize, weights: &[f64], sparsity: f64) {
        let sparsity = sparsity.clamp(0.0, 1.0);
        let n = weights.len().min(self.num_params);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            weights[a]
                .abs()
                .partial_cmp(&weights[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n_prune = (n as f64 * sparsity).round() as usize;
        let mut mask = vec![1.0f64; self.num_params];
        for &idx in &order[..n_prune.min(n)] {
            mask[idx] = 0.0;
        }
        self.masks.insert(task_id, mask);
    }

    pub fn apply_mask(&self, weights: &[f64], task_id: usize) -> Result<Vec<f64>> {
        let mask = self.masks.get(&task_id).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "PackNetMasker::apply_mask",
                &format!("no mask for task {}", task_id),
            )
        })?;
        let n = weights.len().min(mask.len());
        let mut out = weights.to_vec();
        for i in 0..n {
            out[i] *= mask[i];
        }
        Ok(out)
    }
    pub fn num_tasks(&self) -> usize {
        self.masks.len()
    }
    pub fn achieved_sparsity(&self, task_id: usize) -> Option<f64> {
        let mask = self.masks.get(&task_id)?;
        if mask.is_empty() {
            return Some(0.0);
        }
        Some(mask.iter().filter(|&&v| v == 0.0).count() as f64 / mask.len() as f64)
    }
}

/// HAT: cumulative hard-attention mask across tasks.
#[derive(Debug, Clone)]
pub struct HatMask {
    embeddings: HashMap<usize, Vec<f64>>,
    pub threshold: f64,
    pub temperature: f64,
}

impl HatMask {
    pub fn new(threshold: f64, temperature: f64) -> Self {
        Self {
            embeddings: HashMap::new(),
            threshold,
            temperature,
        }
    }
    pub fn register_embedding(&mut self, task_id: usize, embedding: Vec<f64>) {
        self.embeddings.insert(task_id, embedding);
    }
    /// Cumulative mask: max sigmoid across tasks, binarized at `threshold`.
    pub fn compute_cumulative_mask(&self, threshold: f64) -> Vec<f64> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }
        let max_dim = self.embeddings.values().map(|e| e.len()).max().unwrap_or(0);
        let mut cum = vec![0.0f64; max_dim];
        for emb in self.embeddings.values() {
            for (d, &e) in emb.iter().enumerate().take(max_dim) {
                let sig = 1.0 / (1.0 + (-self.temperature * e).exp());
                if sig > cum[d] {
                    cum[d] = sig;
                }
            }
        }
        cum.iter()
            .map(|&v| if v >= threshold { 1.0 } else { 0.0 })
            .collect()
    }
    pub fn num_tasks(&self) -> usize {
        self.embeddings.len()
    }
}

/// Dynamic expansion: grow hidden units when val_loss exceeds threshold.
#[derive(Debug, Clone)]
pub struct DynamicExpansionLayer {
    pub hidden_size: usize,
    pub expansion_factor: f64,
    pub expansion_history: Vec<(f64, usize)>,
    pub max_hidden_size: usize,
}

impl DynamicExpansionLayer {
    pub fn new(initial_size: usize, expansion_factor: f64, max_hidden_size: usize) -> Self {
        Self {
            hidden_size: initial_size,
            expansion_factor,
            expansion_history: Vec::new(),
            max_hidden_size,
        }
    }
    pub fn maybe_expand(&mut self, val_loss: f64, capacity_threshold: f64) -> bool {
        if val_loss > capacity_threshold && self.hidden_size < self.max_hidden_size {
            let new_size = ((self.hidden_size as f64 * self.expansion_factor).ceil() as usize)
                .min(self.max_hidden_size);
            self.expansion_history.push((val_loss, new_size));
            self.hidden_size = new_size;
            true
        } else {
            false
        }
    }
}

/// Modular network: shared module bank + task-specific routing.
#[derive(Debug, Clone)]
pub struct ModularNetwork {
    modules: Vec<Vec<f64>>,
    routing_table: HashMap<usize, Vec<usize>>,
    pub k: usize,
}

impl ModularNetwork {
    pub fn new(num_modules: usize, module_dim: usize, k: usize) -> Self {
        let modules = (0..num_modules)
            .map(|i| vec![(i as f64 + 1.0) * 0.1; module_dim])
            .collect();
        Self {
            modules,
            routing_table: HashMap::new(),
            k,
        }
    }
    pub fn register_route(&mut self, task_id: usize, module_indices: Vec<usize>) {
        self.routing_table.insert(task_id, module_indices);
    }
    fn auto_route(&self, task_id: usize) -> Vec<usize> {
        let n = self.modules.len();
        if n == 0 {
            return Vec::new();
        }
        (0..self.k)
            .map(|offset| (task_id.wrapping_add(offset * 31)) % n)
            .collect()
    }
    /// Sum module outputs for the routed modules applied to `x`.
    pub fn route(&self, x: &[f64], task_id: usize) -> Vec<f64> {
        let indices = self
            .routing_table
            .get(&task_id)
            .cloned()
            .unwrap_or_else(|| self.auto_route(task_id));
        if self.modules.is_empty() || indices.is_empty() {
            return x.to_vec();
        }
        let out_dim = self.modules[0].len();
        let mut out = vec![0.0f64; out_dim];
        for &idx in &indices {
            if idx < self.modules.len() {
                let module = &self.modules[idx];
                let in_dim = x.len().min(module.len());
                for d in 0..in_dim.min(out_dim) {
                    out[d] += x[d] * module[d];
                }
            }
        }
        out
    }
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }
}

// ── §4  Regularization-Based Methods ─────────────────────────────────────────

/// Synaptic Intelligence: online importance Ω_k = Σ_t −g_t · Δθ_k.
#[derive(Debug, Clone)]
pub struct SynapticIntelligence {
    pub omega: Vec<f64>,
    pub anchor: Vec<f64>,
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    pub xi: f64,
}

impl SynapticIntelligence {
    pub fn new(num_params: usize, xi: f64) -> Self {
        Self {
            omega: vec![0.0; num_params],
            anchor: vec![0.0; num_params],
            numerator: vec![0.0; num_params],
            denominator: vec![0.0; num_params],
            xi,
        }
    }
    /// Accumulate importance estimate from one step.
    pub fn update_omega(&mut self, grad: &[f64], param_delta: &[f64]) {
        let dim = grad.len().min(param_delta.len()).min(self.omega.len());
        for k in 0..dim {
            self.numerator[k] += grad[k] * param_delta[k];
            self.denominator[k] += param_delta[k] * param_delta[k];
        }
    }
    /// Consolidate importance and set new anchor at end of task.
    pub fn consolidate(&mut self, current_params: &[f64]) {
        let dim = self.omega.len().min(current_params.len());
        for k in 0..dim {
            let denom = self.denominator[k] + self.xi;
            self.omega[k] += (-self.numerator[k] / denom).max(0.0);
            self.anchor[k] = current_params[k];
            self.numerator[k] = 0.0;
            self.denominator[k] = 0.0;
        }
    }
    /// SI surrogate loss: c · Σ_k Ω_k · (θ_k − θ*_k)².
    pub fn si_loss(&self, params: &[f64], c: f64) -> f64 {
        let dim = params.len().min(self.omega.len()).min(self.anchor.len());
        c * (0..dim)
            .map(|k| self.omega[k] * (params[k] - self.anchor[k]).powi(2))
            .sum::<f64>()
    }
}

/// MemoRep: representation distillation loss.
#[derive(Debug, Clone)]
pub struct MemoRep {
    stored: HashMap<usize, Vec<(Vec<f64>, Vec<f64>)>>,
    pub regularization_strength: f64,
}

impl MemoRep {
    pub fn new(regularization_strength: f64) -> Self {
        Self {
            stored: HashMap::new(),
            regularization_strength,
        }
    }
    pub fn store_representation(&mut self, task_id: usize, input: Vec<f64>, repr: Vec<f64>) {
        self.stored.entry(task_id).or_default().push((input, repr));
    }
    pub fn repr_distillation_loss(
        &self,
        task_id: usize,
        current_representations: &[Vec<f64>],
    ) -> f64 {
        let pairs = match self.stored.get(&task_id) {
            Some(p) => p,
            None => return 0.0,
        };
        let n = pairs.len().min(current_representations.len());
        if n == 0 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .map(|i| {
                let old_repr = &pairs[i].1;
                let new_repr = &current_representations[i];
                let dim = old_repr.len().min(new_repr.len());
                (0..dim)
                    .map(|d| (old_repr[d] - new_repr[d]).powi(2))
                    .sum::<f64>()
            })
            .sum();
        self.regularization_strength * total / n as f64
    }
}

/// Learning without Forgetting (LwF): KL-divergence-based distillation.
#[derive(Debug, Clone)]
pub struct LearningWithoutForgetting {
    pub temperature: f64,
    pub lambda: f64,
}

impl LearningWithoutForgetting {
    pub fn new(temperature: f64, lambda: f64) -> Self {
        Self {
            temperature,
            lambda,
        }
    }

    /// LwF loss: λ · T² · KL(σ(old/T) ‖ σ(new/T)). Returns 0 when distributions match.
    pub fn lwf_loss(new_logits: &[f64], old_logits: &[f64], temperature: f64, lambda: f64) -> f64 {
        let n = new_logits.len().min(old_logits.len());
        if n == 0 {
            return 0.0;
        }
        let t = temperature.max(1e-8);

        let softmax_log_softmax = |logits: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = logits.iter().map(|&x| (x / t - max / t).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let soft: Vec<f64> = exp.iter().map(|&e| e / sum.max(1e-30)).collect();
            let log_sum = sum.max(1e-30).ln();
            let log_soft: Vec<f64> = logits.iter().map(|&x| x / t - max / t - log_sum).collect();
            (soft, log_soft)
        };

        let (old_soft, old_log_soft) = softmax_log_softmax(&old_logits[..n]);
        let (_, new_log_soft) = softmax_log_softmax(&new_logits[..n]);

        let kl: f64 = (0..n)
            .map(|j| {
                if old_soft[j] > 1e-30 {
                    old_soft[j] * (old_log_soft[j] - new_log_soft[j])
                } else {
                    0.0
                }
            })
            .sum();
        lambda * t * t * kl.max(0.0)
    }
}

/// OWM: project gradients orthogonal to past task activation subspace.
///
/// P = I − A(AᵀA + αI)⁻¹Aᵀ
#[derive(Debug, Clone)]
pub struct OWM {
    pub alpha: f64,
    past_activations: Vec<Vec<f64>>,
}

impl OWM {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            past_activations: Vec::new(),
        }
    }
    pub fn add_activation(&mut self, activation: Vec<f64>) {
        self.past_activations.push(activation);
    }

    pub fn compute_projection_matrix(&self, past_activations: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if past_activations.is_empty() {
            return Vec::new();
        }
        let dim = past_activations[0].len();
        let n = past_activations.len();
        let mut gram = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..dim
                    .min(past_activations[i].len())
                    .min(past_activations[j].len()))
                    .map(|d| past_activations[i][d] * past_activations[j][d])
                    .sum();
                gram[i][j] = dot;
            }
            gram[i][i] += self.alpha;
        }
        let gram_inv = match invert_matrix(&gram) {
            Some(m) => m,
            None => return Vec::new(),
        };
        let mut p = vec![vec![0.0f64; dim]; dim];
        for d in 0..dim {
            p[d][d] = 1.0;
        }
        for r in 0..dim {
            for c in 0..dim {
                let mut val = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        let a_ri = past_activations[i].get(r).copied().unwrap_or(0.0);
                        let a_cj = past_activations[j].get(c).copied().unwrap_or(0.0);
                        val += a_ri * gram_inv[i][j] * a_cj;
                    }
                }
                p[r][c] -= val;
            }
        }
        p
    }

    pub fn project_gradient(&self, g: &[f64]) -> Vec<f64> {
        let p = self.compute_projection_matrix(&self.past_activations);
        if p.is_empty() {
            return g.to_vec();
        }
        let dim = g.len().min(p.len());
        (0..dim)
            .map(|r| {
                (0..dim)
                    .map(|c| p[r][c] * g.get(c).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect()
    }
}

fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.resize(2 * n, 0.0);
            r[n + i] = 1.0;
            r
        })
        .collect();
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return None;
        }
        for v in &mut aug[col] {
            *v /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for c in 0..2 * n {
                    let sub = factor * aug[col][c];
                    aug[row][c] -= sub;
                }
            }
        }
    }
    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Functional regularization: MSE penalty on anchor (input, output) pairs.
#[derive(Debug, Clone)]
pub struct FunctionalRegularization {
    anchors: Vec<(Vec<f64>, Vec<f64>)>,
    pub regularization_strength: f64,
}

impl FunctionalRegularization {
    pub fn new(regularization_strength: f64) -> Self {
        Self {
            anchors: Vec::new(),
            regularization_strength,
        }
    }
    pub fn register_anchor(&mut self, input: Vec<f64>, output: Vec<f64>) {
        self.anchors.push((input, output));
    }
    pub fn functional_loss(&self, current_outputs: &[Vec<f64>]) -> f64 {
        let n = self.anchors.len().min(current_outputs.len());
        if n == 0 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .map(|i| {
                let stored = &self.anchors[i].1;
                let curr = &current_outputs[i];
                let dim = stored.len().min(curr.len());
                (0..dim).map(|d| (stored[d] - curr[d]).powi(2)).sum::<f64>()
            })
            .sum();
        self.regularization_strength * total / n as f64
    }
    pub fn num_anchors(&self) -> usize {
        self.anchors.len()
    }
}

// ── §5  Evaluation & Metrics ──────────────────────────────────────────────────

/// Scalar continual learning metrics: BWT, FWT, AIA, final_avg.
#[derive(Debug, Clone)]
pub struct ClMetrics {
    /// Backward Transfer (negative = forgetting).
    pub bwt: f64,
    /// Forward Transfer.
    pub fwt: f64,
    /// Average Incremental Accuracy.
    pub aia: f64,
    /// Final average accuracy across all tasks.
    pub final_avg: f64,
}

/// Compute BWT, FWT, AIA, final_avg from accuracy matrix `acc[i][j]` = accuracy on task j after step i.
#[derive(Debug, Clone, Default)]
pub struct ContinualLearningMetrics;

impl ContinualLearningMetrics {
    pub fn compute(accuracy_matrix: &[Vec<f64>]) -> ClMetrics {
        let t = accuracy_matrix.len();
        if t == 0 {
            return ClMetrics {
                bwt: 0.0,
                fwt: 0.0,
                aia: 0.0,
                final_avg: 0.0,
            };
        }
        let num_tasks = accuracy_matrix[0].len();

        let bwt = if t > 1 && num_tasks >= t {
            (0..t - 1)
                .map(|j| {
                    accuracy_matrix[t - 1].get(j).copied().unwrap_or(0.0)
                        - accuracy_matrix[j].get(j).copied().unwrap_or(0.0)
                })
                .sum::<f64>()
                / (t - 1) as f64
        } else {
            0.0
        };

        let fwt = if t > 1 && num_tasks >= t {
            (1..t)
                .map(|j| accuracy_matrix[j - 1].get(j).copied().unwrap_or(0.0))
                .sum::<f64>()
                / (t - 1) as f64
        } else {
            0.0
        };

        let aia: f64 = (0..t)
            .map(|i| {
                let row = &accuracy_matrix[i];
                let n_seen = (i + 1).min(row.len());
                if n_seen == 0 {
                    0.0
                } else {
                    row[..n_seen].iter().sum::<f64>() / n_seen as f64
                }
            })
            .sum::<f64>()
            / t as f64;

        let final_row = &accuracy_matrix[t - 1];
        let n_final = final_row.len().min(num_tasks);
        let final_avg = if n_final > 0 {
            final_row[..n_final].iter().sum::<f64>() / n_final as f64
        } else {
            0.0
        };

        ClMetrics {
            bwt,
            fwt,
            aia,
            final_avg,
        }
    }
}

/// Forgetting measure: mean of (max past accuracy − current accuracy) per task.
#[derive(Debug, Clone, Default)]
pub struct ForgettingMeasure;

impl ForgettingMeasure {
    pub fn compute(accuracy_matrix: &[Vec<f64>]) -> f64 {
        let t = accuracy_matrix.len();
        if t < 2 {
            return 0.0;
        }
        let num_tasks = accuracy_matrix.iter().map(|r| r.len()).max().unwrap_or(0);
        if num_tasks == 0 {
            return 0.0;
        }
        let final_row = &accuracy_matrix[t - 1];
        let n_tasks = num_tasks.min(t - 1);
        if n_tasks == 0 {
            return 0.0;
        }
        (0..n_tasks)
            .map(|j| {
                let max_past: f64 = (0..t)
                    .map(|i| accuracy_matrix[i].get(j).copied().unwrap_or(0.0))
                    .fold(f64::NEG_INFINITY, f64::max);
                (max_past - final_row.get(j).copied().unwrap_or(0.0)).max(0.0)
            })
            .sum::<f64>()
            / n_tasks as f64
    }
}

/// Plasticity–stability trade-off: (1 + BWT, FWT) ∈ (\[0,1\], [-1,1]).
#[derive(Debug, Clone, Default)]
pub struct PlasticityStabilityTradeoff;

impl PlasticityStabilityTradeoff {
    pub fn evaluate(metrics: &ClMetrics) -> (f64, f64) {
        (
            (1.0 + metrics.bwt).clamp(0.0, 1.0),
            metrics.fwt.clamp(-1.0, 1.0),
        )
    }
    pub fn evaluate_from_matrix(accuracy_matrix: &[Vec<f64>]) -> (f64, f64) {
        let forgetting = ForgettingMeasure::compute(accuracy_matrix);
        let metrics = ContinualLearningMetrics::compute(accuracy_matrix);
        (
            (1.0 - forgetting).clamp(0.0, 1.0),
            metrics.fwt.clamp(-1.0, 1.0),
        )
    }
}

/// Benchmark suite: simulated permuted-MNIST and split-CIFAR accuracy matrices.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkSuite;

impl BenchmarkSuite {
    /// Simulate 10-task permuted-MNIST accuracy matrix.
    pub fn run_permuted_mnist(seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 10;
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                m[i][j] = if j == i {
                    0.85 + rng.random::<f64>() * 0.13
                } else {
                    let decay = 0.02 * (i - j) as f64;
                    (m[j][j] - decay - rng.random::<f64>() * 0.05).clamp(0.5, 1.0)
                };
            }
        }
        m
    }
    /// Simulate 5-task split-CIFAR accuracy matrix.
    pub fn run_split_cifar(seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 5;
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                m[i][j] = if j == i {
                    0.88 + rng.random::<f64>() * 0.10
                } else {
                    let decay = 0.03 * (i - j) as f64;
                    (m[j][j] - decay - rng.random::<f64>() * 0.04).clamp(0.55, 1.0)
                };
            }
        }
        m
    }
}

pub mod extensions;
pub use extensions::*;

#[cfg(test)]
mod tests;
