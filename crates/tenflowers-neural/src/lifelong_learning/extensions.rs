//! Lll-prefixed lifelong learning extensions (§6).
//!
//! GEM model, A-GEM model, Experience Replay, Dark Experience Replay,
//! Continual Prototype Evolution, Hard Attention to Task, Task Oracle,
//! and Lifelong Learning evaluation metrics.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

use super::{
    AgemOptimizer, ContinualLearningMetrics, EpisodeMemory, ForgettingMeasure, GemConstraint,
};

// ── §6  Lll-Prefixed Extensions ───────────────────────────────────────────────

// ── §6.1  LllGemModel — Gradient Episodic Memory (Lopez-Paz 2017) ─────────────

/// A simple linear model layer used by LllGemModel.
#[derive(Debug, Clone)]
pub struct LllLinearLayer {
    pub weights: Vec<Vec<f64>>, // [out_dim x in_dim]
    pub bias: Vec<f64>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl LllLinearLayer {
    /// Xavier-uniform initialization.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let weights = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let bias = vec![0.0; out_dim];
        Self {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: y = Wx + b with ReLU.
    pub fn forward_relu(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_dim {
            return Err(TensorError::invalid_argument_op(
                "LllLinearLayer::forward_relu",
                &format!("expected in_dim={} but got {}", self.in_dim, x.len()),
            ));
        }
        Ok((0..self.out_dim)
            .map(|o| {
                let z: f64 = (0..self.in_dim)
                    .map(|i| self.weights[o][i] * x[i])
                    .sum::<f64>()
                    + self.bias[o];
                z.max(0.0)
            })
            .collect())
    }

    /// Forward pass without activation (linear).
    pub fn forward_linear(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_dim {
            return Err(TensorError::invalid_argument_op(
                "LllLinearLayer::forward_linear",
                &format!("expected in_dim={} but got {}", self.in_dim, x.len()),
            ));
        }
        Ok((0..self.out_dim)
            .map(|o| {
                (0..self.in_dim)
                    .map(|i| self.weights[o][i] * x[i])
                    .sum::<f64>()
                    + self.bias[o]
            })
            .collect())
    }

    /// Flatten all parameters into one vector.
    pub fn params_flat(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self
            .weights
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        v.extend_from_slice(&self.bias);
        v
    }

    /// Update params from a flat gradient vector (SGD step).
    pub fn apply_grad_flat(&mut self, grad_flat: &[f64], lr: f64) {
        let mut idx = 0;
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                if idx < grad_flat.len() {
                    self.weights[o][i] -= lr * grad_flat[idx];
                    idx += 1;
                }
            }
        }
        for b in &mut self.bias {
            if idx < grad_flat.len() {
                *b -= lr * grad_flat[idx];
                idx += 1;
            }
        }
    }
}

/// GEM model: two-layer MLP with episodic memory and gradient projection.
///
/// Implements Lopez-Paz 2017: gradient updates must not increase loss on
/// stored memory samples from previous tasks.
#[derive(Debug, Clone)]
pub struct LllGemModel {
    pub layer1: LllLinearLayer,
    pub layer2: LllLinearLayer,
    pub memory: EpisodeMemory,
    pub gem: GemConstraint,
    pub lr: f64,
    pub num_classes: usize,
}

impl LllGemModel {
    /// Create with `in_dim → hidden_dim → num_classes` architecture.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        num_classes: usize,
        memory_capacity: usize,
        lr: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            layer1: LllLinearLayer::new(in_dim, hidden_dim, &mut rng),
            layer2: LllLinearLayer::new(hidden_dim, num_classes, &mut rng),
            memory: EpisodeMemory::new(memory_capacity),
            gem: GemConstraint::default(),
            lr,
            num_classes,
        }
    }

    /// Forward pass returning logits.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let h = self.layer1.forward_relu(x)?;
        self.layer2.forward_linear(&h)
    }

    /// Softmax cross-entropy loss for one sample.
    pub fn cross_entropy_loss(&self, logits: &[f64], label: usize) -> Result<f64> {
        if label >= logits.len() {
            return Err(TensorError::invalid_argument_op(
                "LllGemModel::cross_entropy_loss",
                &format!("label {} out of range [0, {})", label, logits.len()),
            ));
        }
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let log_prob = logits[label] - max_l - exp_sum.max(1e-30).ln();
        Ok(-log_prob)
    }

    /// Compute pseudo-gradient via finite differences for all parameters.
    pub fn compute_grad_fd(&self, x: &[f64], label: usize, eps: f64) -> Result<Vec<f64>> {
        let logits = self.forward(x)?;
        let loss0 = self.cross_entropy_loss(&logits, label)?;
        let n_params1 = self.layer1.in_dim * self.layer1.out_dim + self.layer1.out_dim;
        let n_params2 = self.layer2.in_dim * self.layer2.out_dim + self.layer2.out_dim;
        let n_total = n_params1 + n_params2;
        let mut grad = vec![0.0; n_total];
        let mut model_p = self.clone();
        let p1 = model_p.layer1.params_flat();
        let p2 = model_p.layer2.params_flat();
        let mut all_params: Vec<f64> = p1;
        all_params.extend(p2);
        for j in 0..n_total {
            let mut perturbed = all_params.clone();
            perturbed[j] += eps;
            if j < n_params1 {
                model_p.layer1.apply_grad_flat(
                    &(0..n_params1)
                        .map(|k| if k == j { -eps } else { 0.0 })
                        .collect::<Vec<_>>(),
                    1.0,
                );
            } else {
                model_p.layer2.apply_grad_flat(
                    &(0..n_params2)
                        .map(|k| if k == j - n_params1 { -eps } else { 0.0 })
                        .collect::<Vec<_>>(),
                    1.0,
                );
            }
            let logits_p = model_p.forward(x)?;
            let loss_p = model_p.cross_entropy_loss(&logits_p, label)?;
            grad[j] = (loss_p - loss0) / eps;
            // reset
            model_p = self.clone();
        }
        Ok(grad)
    }

    /// Store episodic memory for task_id from (input, gradient) pairs.
    pub fn add_memory(&mut self, task_id: usize, samples: Vec<(Vec<f64>, Vec<f64>)>) {
        self.memory.add_task_memory(task_id, samples);
    }

    /// GEM-projected gradient update for a current sample.
    ///
    /// Returns the projected gradient vector (for introspection).
    pub fn gem_update(&mut self, x: &[f64], label: usize) -> Result<Vec<f64>> {
        let current_grad = self.compute_grad_fd(x, label, 1e-4)?;
        let all_mem_grads: Vec<Vec<f64>> = self
            .memory
            .task_ids()
            .into_iter()
            .flat_map(|tid| {
                self.memory
                    .get_task_memories(tid)
                    .iter()
                    .map(|(_, g)| g.clone())
            })
            .collect();
        let projected = self
            .gem
            .project_gradient(&current_grad, &all_mem_grads, 1e-7)?;
        let n1 = self.layer1.in_dim * self.layer1.out_dim + self.layer1.out_dim;
        self.layer1
            .apply_grad_flat(&projected[..n1.min(projected.len())], self.lr);
        let rem = if projected.len() > n1 {
            &projected[n1..]
        } else {
            &[]
        };
        self.layer2.apply_grad_flat(rem, self.lr);
        Ok(projected)
    }
}

// ── §6.2  LllAGemModel — A-GEM (Chaudhry 2019) ────────────────────────────────

/// A-GEM model: averaged episodic memory gradient projection.
///
/// Cheaper than full GEM: O(1) projection per step via single reference gradient.
#[derive(Debug, Clone)]
pub struct LllAGemModel {
    pub layer1: LllLinearLayer,
    pub layer2: LllLinearLayer,
    pub agem: AgemOptimizer,
    pub episodic_store: Vec<(Vec<f64>, Vec<f64>)>, // (input, gradient) pairs
    pub lr: f64,
}

impl LllAGemModel {
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        num_classes: usize,
        ref_size: usize,
        lr: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            layer1: LllLinearLayer::new(in_dim, hidden_dim, &mut rng),
            layer2: LllLinearLayer::new(hidden_dim, num_classes, &mut rng),
            agem: AgemOptimizer::new(ref_size),
            episodic_store: Vec::new(),
            lr,
        }
    }

    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>> {
        let h = self.layer1.forward_relu(x)?;
        self.layer2.forward_linear(&h)
    }

    /// Store a (input, gradient) pair in episodic memory.
    pub fn store_episode(&mut self, input: Vec<f64>, grad: Vec<f64>) {
        self.episodic_store.push((input, grad));
    }

    /// A-GEM update: compute reference from episodic store, project current grad.
    pub fn agem_update(&mut self, current_grad: &[f64], rng: &mut StdRng) -> Vec<f64> {
        match self.agem.compute_reference(&self.episodic_store, rng) {
            Some(ref_grad) => {
                let projected = self.agem.project(current_grad, &ref_grad);
                let n1 = self.layer1.in_dim * self.layer1.out_dim + self.layer1.out_dim;
                self.layer1
                    .apply_grad_flat(&projected[..n1.min(projected.len())], self.lr);
                let rem = if projected.len() > n1 {
                    &projected[n1..]
                } else {
                    &[]
                };
                self.layer2.apply_grad_flat(rem, self.lr);
                projected
            }
            None => {
                // No episodic memory yet: plain SGD
                let n1 = self.layer1.in_dim * self.layer1.out_dim + self.layer1.out_dim;
                self.layer1
                    .apply_grad_flat(&current_grad[..n1.min(current_grad.len())], self.lr);
                let rem = if current_grad.len() > n1 {
                    &current_grad[n1..]
                } else {
                    &[]
                };
                self.layer2.apply_grad_flat(rem, self.lr);
                current_grad.to_vec()
            }
        }
    }
}

// ── §6.3  LllER — Experience Replay (Vitter 1985 reservoir) ──────────────────

/// Sample type for LllER: (input features, one-hot/label index, task_id).
#[derive(Debug, Clone)]
pub struct LllErSample {
    pub x: Vec<f64>,
    pub label: usize,
    pub task_id: usize,
}

/// Experience Replay with reservoir sampling (global buffer, not per-task).
///
/// Guarantees a uniform random sample over all seen examples regardless of
/// order (Vitter Algorithm R).
#[derive(Debug, Clone)]
pub struct LllER {
    buffer: Vec<LllErSample>,
    capacity: usize,
    total_seen: usize,
}

impl LllER {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "LllER::new",
                "capacity must be > 0",
            ));
        }
        Ok(Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
        })
    }

    /// Reservoir-sample update (Vitter Algorithm R).
    pub fn update_buffer(&mut self, sample: LllErSample, rng: &mut StdRng) {
        self.total_seen += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push(sample);
        } else {
            // Each new item replaces a random existing one with prob capacity/total_seen
            let j = (rng.random::<u64>() as usize) % self.total_seen;
            if j < self.capacity {
                self.buffer[j] = sample;
            }
        }
    }

    /// Sample `batch_size` items uniformly at random (with replacement).
    pub fn sample_replay(&self, batch_size: usize, rng: &mut StdRng) -> Result<Vec<&LllErSample>> {
        if self.buffer.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "LllER::sample_replay",
                "replay buffer is empty",
            ));
        }
        let n = self.buffer.len();
        Ok((0..batch_size)
            .map(|_| &self.buffer[rng.random::<u64>() as usize % n])
            .collect())
    }

    /// Mixed training loss: alpha * current_loss + (1-alpha) * replay_loss.
    pub fn train_step(current_loss: f64, replay_loss: f64, alpha: f64) -> f64 {
        let a = alpha.clamp(0.0, 1.0);
        a * current_loss + (1.0 - a) * replay_loss
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
    pub fn total_seen(&self) -> usize {
        self.total_seen
    }
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}

// ── §6.4  LllDer — Dark Experience Replay (Buzzega 2020) ─────────────────────

/// Stored entry for DER: input + model logits at storage time.
#[derive(Debug, Clone)]
pub struct LllDerEntry {
    pub x: Vec<f64>,
    pub stored_logits: Vec<f64>,
    pub label: usize, // original label (for DER++)
    pub task_id: usize,
}

/// Dark Experience Replay: replays stored logits for knowledge distillation.
///
/// DER loss = MSE(current_logits, stored_logits)
/// DER++ loss = DER loss + CE(current_logits, stored_label)
#[derive(Debug, Clone)]
pub struct LllDer {
    buffer: Vec<LllDerEntry>,
    capacity: usize,
    total_seen: usize,
    pub alpha: f64, // DER distillation weight
    pub beta: f64,  // DER++ label CE weight
}

impl LllDer {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Result<Self> {
        if capacity == 0 {
            return Err(TensorError::invalid_argument_op(
                "LllDer::new",
                "capacity must be > 0",
            ));
        }
        Ok(Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
            alpha,
            beta,
        })
    }

    /// Add entry with reservoir sampling.
    pub fn add_entry(&mut self, entry: LllDerEntry, rng: &mut StdRng) {
        self.total_seen += 1;
        if self.buffer.len() < self.capacity {
            self.buffer.push(entry);
        } else {
            let j = (rng.random::<u64>() as usize) % self.total_seen;
            if j < self.capacity {
                self.buffer[j] = entry;
            }
        }
    }

    /// DER loss for a batch: α · MSE(current_logits\[i\], stored_logits\[i\]).
    pub fn der_loss(&self, current_logits_batch: &[Vec<f64>], entries: &[&LllDerEntry]) -> f64 {
        let n = current_logits_batch.len().min(entries.len());
        if n == 0 {
            return 0.0;
        }
        let mse_sum: f64 = (0..n)
            .map(|i| {
                let cur = &current_logits_batch[i];
                let stored = &entries[i].stored_logits;
                let dim = cur.len().min(stored.len());
                (0..dim).map(|d| (cur[d] - stored[d]).powi(2)).sum::<f64>() / dim.max(1) as f64
            })
            .sum();
        self.alpha * mse_sum / n as f64
    }

    /// DER++ loss: DER loss + β · CE(current_logits, stored_labels).
    pub fn der_plus_plus_loss(
        &self,
        current_logits_batch: &[Vec<f64>],
        entries: &[&LllDerEntry],
    ) -> Result<f64> {
        let n = current_logits_batch.len().min(entries.len());
        if n == 0 {
            return Ok(0.0);
        }
        let der_l = self.der_loss(current_logits_batch, entries);
        let mut ce_sum = 0.0;
        for i in 0..n {
            let logits = &current_logits_batch[i];
            let label = entries[i].label;
            if label >= logits.len() {
                return Err(TensorError::invalid_argument_op(
                    "LllDer::der_plus_plus_loss",
                    &format!("label {} out of logits len {}", label, logits.len()),
                ));
            }
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            ce_sum += -(logits[label] - max_l - exp_sum.max(1e-30).ln());
        }
        Ok(der_l + self.beta * ce_sum / n as f64)
    }

    /// Sample `batch_size` entries uniformly.
    pub fn sample_entries(&self, batch_size: usize, rng: &mut StdRng) -> Result<Vec<&LllDerEntry>> {
        if self.buffer.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "LllDer::sample_entries",
                "buffer is empty",
            ));
        }
        let n = self.buffer.len();
        Ok((0..batch_size)
            .map(|_| &self.buffer[rng.random::<u64>() as usize % n])
            .collect())
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

// ── §6.5  LllCoPE — Continual Prototype Evolution (De Lange 2021) ────────────

/// Per-class prototype with EMA update in feature space.
#[derive(Debug, Clone)]
pub struct LllCoPE {
    prototypes: HashMap<usize, Vec<f64>>, // class_id → prototype vector
    counts: HashMap<usize, usize>,
    pub ema_momentum: f64, // EMA decay (0 = full replace, 1 = no update)
    pub distillation_weight: f64,
    old_prototypes: HashMap<usize, Vec<f64>>, // snapshot for distillation
}

impl LllCoPE {
    pub fn new(ema_momentum: f64, distillation_weight: f64) -> Self {
        Self {
            prototypes: HashMap::new(),
            counts: HashMap::new(),
            ema_momentum,
            distillation_weight,
            old_prototypes: HashMap::new(),
        }
    }

    /// EMA update of prototype for class_id using new feature vector.
    pub fn update_prototype(&mut self, features: &[f64], class_id: usize) {
        let count = self.counts.entry(class_id).or_insert(0);
        *count += 1;
        let proto = self
            .prototypes
            .entry(class_id)
            .or_insert_with(|| vec![0.0; features.len()]);
        let m = self.ema_momentum;
        let n = features.len().min(proto.len());
        for d in 0..n {
            proto[d] = m * proto[d] + (1.0 - m) * features[d];
        }
        // Extend if features longer
        for d in proto.len()..features.len() {
            proto.push((1.0 - m) * features[d]);
        }
    }

    /// Batch update: update prototype for each (feature, label) pair.
    pub fn update_prototypes(&mut self, features: &[Vec<f64>], labels: &[usize]) -> Result<()> {
        if features.len() != labels.len() {
            return Err(TensorError::invalid_argument_op(
                "LllCoPE::update_prototypes",
                "features and labels must have the same length",
            ));
        }
        for (feat, &lbl) in features.iter().zip(labels.iter()) {
            self.update_prototype(feat, lbl);
        }
        Ok(())
    }

    /// Classify input by nearest prototype (L2 distance).
    pub fn classify(&self, features: &[f64]) -> Result<usize> {
        if self.prototypes.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "LllCoPE::classify",
                "no prototypes registered",
            ));
        }
        let mut best_class = 0;
        let mut best_dist = f64::INFINITY;
        for (&cls, proto) in &self.prototypes {
            let dim = features.len().min(proto.len());
            let dist: f64 = (0..dim).map(|d| (features[d] - proto[d]).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_class = cls;
            }
        }
        Ok(best_class)
    }

    /// Snapshot current prototypes for distillation (call at task boundary).
    pub fn snapshot_prototypes(&mut self) {
        self.old_prototypes = self.prototypes.clone();
    }

    /// Prototype distillation loss: penalise drift from snapshot.
    pub fn prototype_drift_loss(&self) -> f64 {
        if self.old_prototypes.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for (&cls, old_proto) in &self.old_prototypes {
            if let Some(new_proto) = self.prototypes.get(&cls) {
                let dim = old_proto.len().min(new_proto.len());
                let drift: f64 = (0..dim)
                    .map(|d| (new_proto[d] - old_proto[d]).powi(2))
                    .sum();
                total += drift;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            self.distillation_weight * total / count as f64
        }
    }

    pub fn num_classes(&self) -> usize {
        self.prototypes.len()
    }
    pub fn get_prototype(&self, class_id: usize) -> Option<&Vec<f64>> {
        self.prototypes.get(&class_id)
    }
}

// ── §6.6  LllHAT — Hard Attention to the Task (Serra 2018) ───────────────────

/// Binary task-specific masks via sigmoid with high temperature.
///
/// Implements Serra et al. 2018: soft masks during training (differentiable
/// via straight-through), hard binary masks at inference. Cumulative mask
/// prevents overwriting units used by previous tasks.
#[derive(Debug, Clone)]
pub struct LllHAT {
    /// Task embedding vectors (raw logits before sigmoid), one per task.
    task_embeddings: HashMap<usize, Vec<f64>>,
    /// Cumulative frozen mask: units already claimed by past tasks.
    pub cumulative_mask: Vec<f64>,
    pub layer_size: usize,
    /// Temperature for sigmoid during training (higher = harder mask).
    pub train_temperature: f64,
    /// Sparsity regularization weight.
    pub sparsity_lambda: f64,
}

impl LllHAT {
    pub fn new(layer_size: usize, train_temperature: f64, sparsity_lambda: f64) -> Self {
        Self {
            task_embeddings: HashMap::new(),
            cumulative_mask: vec![0.0; layer_size],
            layer_size,
            train_temperature,
            sparsity_lambda,
        }
    }

    /// Initialize embedding for a new task (small random values).
    pub fn init_task(&mut self, task_id: usize, rng: &mut StdRng) {
        let emb: Vec<f64> = (0..self.layer_size)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.01)
            .collect();
        self.task_embeddings.insert(task_id, emb);
    }

    /// Soft mask for task_id: sigmoid(temperature * embedding).
    pub fn soft_mask(&self, task_id: usize) -> Result<Vec<f64>> {
        let emb = self.task_embeddings.get(&task_id).ok_or_else(|| {
            TensorError::invalid_argument_op(
                "LllHAT::soft_mask",
                &format!("task_id {} not initialized", task_id),
            )
        })?;
        Ok(emb
            .iter()
            .map(|&e| 1.0 / (1.0 + (-self.train_temperature * e).exp()))
            .collect())
    }

    /// Hard binary mask: threshold at 0.5.
    pub fn hard_mask(&self, task_id: usize) -> Result<Vec<f64>> {
        let soft = self.soft_mask(task_id)?;
        Ok(soft
            .iter()
            .map(|&s| if s > 0.5 { 1.0 } else { 0.0 })
            .collect())
    }

    /// Forward pass: apply soft/hard mask to activations.
    pub fn forward(&self, x: &[f64], task_id: usize, hard: bool) -> Result<Vec<f64>> {
        let mask = if hard {
            self.hard_mask(task_id)?
        } else {
            self.soft_mask(task_id)?
        };
        let n = x.len().min(mask.len());
        let mut out: Vec<f64> = x.to_vec();
        for d in 0..n {
            out[d] *= mask[d];
        }
        Ok(out)
    }

    /// Sparsity regularization: penalise large mask values.
    pub fn sparsity_loss(&self, task_id: usize) -> Result<f64> {
        let soft = self.soft_mask(task_id)?;
        let mean_activation: f64 = soft.iter().sum::<f64>() / soft.len().max(1) as f64;
        Ok(self.sparsity_lambda * mean_activation)
    }

    /// Consolidate task: freeze units where hard_mask = 1 into cumulative_mask.
    /// New tasks cannot modify these units.
    pub fn consolidate_task(&mut self, task_id: usize) -> Result<()> {
        let hard = self.hard_mask(task_id)?;
        let n = hard.len().min(self.cumulative_mask.len());
        for d in 0..n {
            if hard[d] > 0.5 {
                self.cumulative_mask[d] = 1.0;
            }
        }
        Ok(())
    }

    /// Gradient mask for current task: zero out gradients for frozen units.
    /// Returns a mask where 1 = can train, 0 = frozen by previous tasks.
    pub fn gradient_mask(&self) -> Vec<f64> {
        self.cumulative_mask.iter().map(|&v| 1.0 - v).collect()
    }

    /// Apply gradient mask: protect frozen units from gradient updates.
    pub fn mask_gradient(&self, grad: &[f64]) -> Vec<f64> {
        let gmask = self.gradient_mask();
        let n = grad.len().min(gmask.len());
        let mut out = grad.to_vec();
        for d in 0..n {
            out[d] *= gmask[d];
        }
        out
    }

    /// Fraction of frozen units.
    pub fn frozen_fraction(&self) -> f64 {
        if self.cumulative_mask.is_empty() {
            return 0.0;
        }
        self.cumulative_mask.iter().filter(|&&v| v > 0.5).count() as f64
            / self.cumulative_mask.len() as f64
    }

    pub fn num_tasks(&self) -> usize {
        self.task_embeddings.len()
    }
}

// ── §6.7  LllTaskOracle — Online task boundary detection ─────────────────────

/// Task boundary detector: sliding window of losses, alert on significant rise.
///
/// Uses a CUSUM-like test: detects when mean loss of recent window exceeds
/// mean of baseline window by a threshold standard deviations.
#[derive(Debug, Clone)]
pub struct LllTaskOracle {
    /// Sliding window of recent loss values.
    window: std::collections::VecDeque<f64>,
    /// Window size for comparison.
    pub window_size: usize,
    /// Number of standard deviations above baseline to trigger alert.
    pub sensitivity: f64,
    /// Minimum window fill fraction before alerting.
    pub min_fill: f64,
    /// Baseline mean (first half of window).
    pub baseline_mean: f64,
    /// Exponential moving average of recent losses.
    pub ema_loss: f64,
    pub ema_alpha: f64,
    /// History of detected change points (step index).
    pub change_points: Vec<usize>,
    pub step: usize,
}

impl LllTaskOracle {
    pub fn new(window_size: usize, sensitivity: f64, ema_alpha: f64) -> Result<Self> {
        if window_size < 4 {
            return Err(TensorError::invalid_argument_op(
                "LllTaskOracle::new",
                "window_size must be >= 4",
            ));
        }
        Ok(Self {
            window: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            sensitivity,
            min_fill: 0.5,
            baseline_mean: 0.0,
            ema_loss: 0.0,
            ema_alpha,
            change_points: Vec::new(),
            step: 0,
        })
    }

    /// Record a loss value and check for task boundary.
    pub fn record_loss(&mut self, loss: f64) -> bool {
        self.step += 1;
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(loss);
        self.ema_loss = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.ema_loss;
        self.detect_task_change()
    }

    /// Detect task change from current window.
    pub fn detect_task_change(&mut self) -> bool {
        let n = self.window.len();
        let min_required = (self.window_size as f64 * self.min_fill).ceil() as usize;
        if n < min_required.max(4) {
            return false;
        }
        let half = n / 2;
        let first_half: Vec<f64> = self.window.iter().take(half).copied().collect();
        let second_half: Vec<f64> = self.window.iter().skip(half).copied().collect();
        let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;
        let var1: f64 = first_half.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>()
            / (first_half.len() as f64).max(1.0);
        let std1 = var1.sqrt().max(1e-8);
        self.baseline_mean = mean1;
        let z_score = (mean2 - mean1) / std1;
        if z_score > self.sensitivity {
            self.change_points.push(self.step);
            // Reset window to avoid repeated alerts
            self.window.clear();
            return true;
        }
        false
    }

    /// Detect task change from a pre-collected slice of losses.
    pub fn detect_from_slice(&mut self, recent_losses: &[f64]) -> bool {
        if recent_losses.len() < 4 {
            return false;
        }
        for &l in recent_losses {
            self.window.push_back(l);
        }
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }
        self.detect_task_change()
    }

    pub fn num_detected(&self) -> usize {
        self.change_points.len()
    }
}

// ── §6.8  LllMetrics — Lifelong Learning Evaluation ──────────────────────────

/// Complete lifelong learning evaluation report.
#[derive(Debug, Clone)]
pub struct LllReport {
    /// Average Accuracy: mean accuracy across all tasks after training all tasks.
    pub average_accuracy: f64,
    /// Backward Transfer: how learning new tasks affects old tasks (negative = forgetting).
    pub backward_transfer: f64,
    /// Forward Transfer: how past tasks help future tasks (positive = transfer).
    pub forward_transfer: f64,
    /// Forgetting: mean of (peak accuracy − final accuracy) per task.
    pub forgetting: f64,
    /// Intransigence: inability to learn new tasks (performance gap vs oracle).
    pub intransigence: f64,
    /// Number of tasks.
    pub num_tasks: usize,
    /// Plasticity score: capacity to learn new tasks = 1 − intransigence.
    pub plasticity: f64,
    /// Stability score: resistance to forgetting = 1 − forgetting (normalized).
    pub stability: f64,
}

/// Lifelong learning metrics calculator.
#[derive(Debug, Clone, Default)]
pub struct LllMetrics;

impl LllMetrics {
    /// Compute full LllReport from accuracy matrix.
    ///
    /// `acc[i][j]` = test accuracy on task j after learning up to task i (0-indexed).
    /// `oracle_accs[j]` = accuracy if model was trained on task j in isolation (joint oracle).
    pub fn compute(acc: &[Vec<f64>], oracle_accs: Option<&[f64]>) -> Result<LllReport> {
        let t = acc.len();
        if t == 0 {
            return Err(TensorError::invalid_argument_op(
                "LllMetrics::compute",
                "accuracy matrix must be non-empty",
            ));
        }
        let num_tasks = acc.iter().map(|r| r.len()).max().unwrap_or(0);
        if num_tasks == 0 {
            return Err(TensorError::invalid_argument_op(
                "LllMetrics::compute",
                "accuracy matrix rows must be non-empty",
            ));
        }

        // Average Accuracy: mean of last row
        let last = &acc[t - 1];
        let n_final = last.len().min(num_tasks);
        let average_accuracy = if n_final > 0 {
            last[..n_final].iter().sum::<f64>() / n_final as f64
        } else {
            0.0
        };

        // Backward Transfer: BWT = 1/(T-1) Σ_{j<T} [R_{T,j} − R_{j,j}]
        let backward_transfer = if t > 1 {
            let sum: f64 = (0..t.min(num_tasks) - 1)
                .map(|j| {
                    let r_tj = acc[t - 1].get(j).copied().unwrap_or(0.0);
                    let r_jj = acc[j].get(j).copied().unwrap_or(0.0);
                    r_tj - r_jj
                })
                .sum();
            sum / (t - 1).max(1) as f64
        } else {
            0.0
        };

        // Forward Transfer: FWT = 1/(T-1) Σ_{j>0} R_{j-1, j}
        let forward_transfer = if t > 1 {
            let sum: f64 = (1..t.min(num_tasks))
                .map(|j| acc[j - 1].get(j).copied().unwrap_or(0.0))
                .sum();
            sum / (t - 1).max(1) as f64
        } else {
            0.0
        };

        // Forgetting: mean max drop per task
        let forgetting = ForgettingMeasure::compute(acc);

        // Intransigence: gap to oracle (if provided)
        let (intransigence, plasticity) = match oracle_accs {
            Some(oracle) => {
                let n = num_tasks.min(oracle.len());
                let gap: f64 = (0..n)
                    .map(|j| {
                        let learned = acc[j.min(t - 1)].get(j).copied().unwrap_or(0.0);
                        (oracle[j] - learned).max(0.0)
                    })
                    .sum::<f64>()
                    / n.max(1) as f64;
                (gap, (1.0 - gap).clamp(0.0, 1.0))
            }
            None => (0.0, 1.0 - forgetting.clamp(0.0, 1.0)),
        };

        // Stability: 1 − normalized forgetting
        let max_forgetting = acc
            .iter()
            .flat_map(|r| r.iter().copied())
            .fold(0.0_f64, f64::max);
        let stability = if max_forgetting > 1e-8 {
            (1.0 - forgetting / max_forgetting).clamp(0.0, 1.0)
        } else {
            1.0
        };

        Ok(LllReport {
            average_accuracy,
            backward_transfer,
            forward_transfer,
            forgetting,
            intransigence,
            num_tasks,
            plasticity,
            stability,
        })
    }

    /// Compute plasticity-stability curve data from a series of accuracy matrices.
    ///
    /// Returns `(stability, plasticity)` per method for plotting.
    pub fn plasticity_stability_curve(acc: &[Vec<f64>]) -> (f64, f64) {
        let forgetting = ForgettingMeasure::compute(acc);
        let metrics = ContinualLearningMetrics::compute(acc);
        let stability = (1.0 - forgetting).clamp(0.0, 1.0);
        let plasticity = metrics.fwt.clamp(-1.0, 1.0);
        (stability, plasticity)
    }

    /// Per-task accuracy evolution: returns accuracy of task j over training steps.
    pub fn per_task_evolution(acc: &[Vec<f64>], task_id: usize) -> Vec<f64> {
        acc.iter()
            .map(|row| row.get(task_id).copied().unwrap_or(0.0))
            .collect()
    }

    /// Compute area under the learning curve (trapezoidal integration).
    pub fn area_under_curve(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return values.first().copied().unwrap_or(0.0);
        }
        let n = values.len();
        (0..n - 1)
            .map(|i| (values[i] + values[i + 1]) * 0.5)
            .sum::<f64>()
            / (n - 1) as f64
    }
}
