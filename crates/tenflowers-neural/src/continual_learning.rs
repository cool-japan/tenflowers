//! Continual / Lifelong Learning algorithms.
//!
//! This module implements state-of-the-art continual learning methods that
//! allow neural networks to learn sequential tasks without catastrophic
//! forgetting of previously acquired knowledge.
//!
//! # Algorithms
//!
//! * [`EwcRegularizer`] — Elastic Weight Consolidation (Kirkpatrick et al. 2017).
//!   Penalises changes to weights that were important for previous tasks by
//!   using the diagonal of the Fisher Information Matrix as importance weights.
//!
//! * [`SynapticIntelligence`] — Zenke et al. 2017. Accumulates an online path
//!   integral of parameter importance during training, avoiding the need for a
//!   separate Fisher estimation pass.
//!
//! * [`ProgressiveNeuralNetwork`] — Rusu et al. 2016. Grows a new column for
//!   every task while freezing all previous columns. Lateral connections allow
//!   knowledge transfer without forgetting.
//!
//! * [`PackNet`] — Mallya & Lazebnik 2018. Uses iterative magnitude pruning to
//!   carve out task-specific sub-networks inside a fixed-capacity network.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::continual_learning::{FisherInfo, EwcRegularizer};
//!
//! // Estimate Fisher from gradients collected over task 0
//! let grads: Vec<Vec<Vec<f32>>> = todo!(); // [n_samples, n_layers, param_size]
//! let fisher = FisherInfo::estimate(&grads);
//!
//! let mut ewc = EwcRegularizer::new(5000.0);
//! ewc.register_task(fisher);
//!
//! // During task 1 training, add penalty to the loss
//! let current_params: Vec<Vec<f32>> = todo!();
//! let penalty = ewc.penalty(&current_params);
//! ```

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise in continual learning operations.
#[derive(Debug)]
pub enum ContinualError {
    /// Requested task id is out of range.
    TaskNotFound { task_id: usize },
    /// No tasks have been registered yet.
    NoTasksRegistered,
    /// A layer's parameter vector has a different length than expected.
    DimensionMismatch {
        layer: usize,
        expected: usize,
        found: usize,
    },
    /// Not enough unused capacity to accommodate a new task.
    InsufficientCapacity { requested: f32, available: f32 },
}

impl fmt::Display for ContinualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContinualError::TaskNotFound { task_id } => {
                write!(f, "task {} not found", task_id)
            }
            ContinualError::NoTasksRegistered => {
                write!(f, "no tasks have been registered")
            }
            ContinualError::DimensionMismatch {
                layer,
                expected,
                found,
            } => {
                write!(
                    f,
                    "layer {}: expected {} parameters but found {}",
                    layer, expected, found
                )
            }
            ContinualError::InsufficientCapacity {
                requested,
                available,
            } => {
                write!(
                    f,
                    "requested capacity {:.4} but only {:.4} is available",
                    requested, available
                )
            }
        }
    }
}

impl std::error::Error for ContinualError {}

// ─────────────────────────────────────────────────────────────────────────────
// Fisher Information Matrix diagonal + EWC
// ─────────────────────────────────────────────────────────────────────────────

/// Diagonal approximation of the Fisher Information Matrix for one task.
///
/// The empirical Fisher is estimated by averaging the squared gradients of the
/// log-likelihood over samples drawn from the dataset under the current model:
///
/// ```text
/// F̂_i  ≈  (1/N) Σ_{n=1}^{N} (∂ log p(y_n|x_n,θ) / ∂θ_i)²
/// ```
///
/// Each element of `diag` corresponds to one layer; each inner `Vec<f32>` holds
/// the per-parameter importance values for that layer.  `means` stores the
/// optimal parameter values `θ*` at the end of learning the task.
#[derive(Debug, Clone)]
pub struct FisherInfo {
    /// Per-layer diagonal Fisher estimate (`F̂_i` for each parameter `i`).
    pub diag: Vec<Vec<f32>>,
    /// Optimal parameters `θ*` recorded at task completion (one Vec per layer).
    pub means: Vec<Vec<f32>>,
    /// Zero-based identifier of the task this estimate belongs to.
    pub task_id: usize,
}

impl FisherInfo {
    /// Estimate the diagonal Fisher from empirical squared gradients.
    ///
    /// `param_grads` has shape `[n_samples, n_layers, param_size]`.
    /// The function averages the squared gradient across the sample dimension
    /// and stores the result together with **zero-initialised means** (the
    /// caller must fill `means` with the actual optimal parameters after
    /// calling this function, or use [`EwcRegularizer::register_task`] which
    /// accepts a ready-made `FisherInfo`).
    ///
    /// # Panics
    ///
    /// Does not panic; returns an all-zero `FisherInfo` for an empty input.
    pub fn estimate(param_grads: &[Vec<Vec<f32>>]) -> Self {
        if param_grads.is_empty() {
            return FisherInfo {
                diag: Vec::new(),
                means: Vec::new(),
                task_id: 0,
            };
        }

        let n_samples = param_grads.len();
        let n_layers = param_grads[0].len();

        // Accumulate squared gradients layer-by-layer.
        let mut diag: Vec<Vec<f32>> = param_grads[0]
            .iter()
            .map(|layer| vec![0.0_f32; layer.len()])
            .collect();

        for sample in param_grads {
            for (l, layer_grads) in sample.iter().enumerate() {
                if l >= diag.len() {
                    break;
                }
                let diag_layer = &mut diag[l];
                for (i, &g) in layer_grads.iter().enumerate() {
                    if i < diag_layer.len() {
                        diag_layer[i] += g * g;
                    }
                }
            }
        }

        // Normalise by sample count.
        let n = n_samples as f32;
        for diag_layer in &mut diag {
            for v in diag_layer.iter_mut() {
                *v /= n;
            }
        }

        // means are zero-initialised; the caller should update them.
        let means: Vec<Vec<f32>> = diag.iter().map(|l| vec![0.0_f32; l.len()]).collect();

        FisherInfo {
            diag,
            means,
            task_id: 0,
        }
    }
}

/// Elastic Weight Consolidation (EWC) regulariser.
///
/// After finishing each task, register the task's `FisherInfo` (diagonal
/// Fisher + optimal parameters) via [`register_task`].  During training on
/// subsequent tasks, compute the regularisation penalty with \[`penalty`\] and
/// add it to the task loss.
///
/// The penalty for *K* registered tasks is:
///
/// ```text
/// L_EWC = (λ/2) Σ_{k=1}^{K} Σ_i  F̂_i^{(k)} · (θ_i − θ*_i^{(k)})²
/// ```
///
/// [`register_task`]: EwcRegularizer::register_task
#[derive(Debug, Clone)]
pub struct EwcRegularizer {
    /// Per-task Fisher information records.
    pub fisher_infos: Vec<FisherInfo>,
    /// Importance weight λ.  Values around 1 000 – 40 000 work well for most
    /// benchmarks.
    pub lambda: f32,
}

impl EwcRegularizer {
    /// Create a new EWC regulariser with the given importance weight `lambda`.
    pub fn new(lambda: f32) -> Self {
        EwcRegularizer {
            fisher_infos: Vec::new(),
            lambda,
        }
    }

    /// Register a completed task's Fisher information so it can be penalised
    /// in future training rounds.
    pub fn register_task(&mut self, fisher: FisherInfo) {
        self.fisher_infos.push(fisher);
    }

    /// Compute the EWC penalty for the given `current_params`.
    ///
    /// `current_params` must have the same number of layers and the same per-
    /// layer sizes as the registered Fisher infos.  Layers that are absent from
    /// a particular `FisherInfo` record are silently skipped.
    ///
    /// Returns `0.0` when no tasks have been registered.
    pub fn penalty(&self, current_params: &[Vec<f32>]) -> f32 {
        let mut total = 0.0_f32;
        for fi in &self.fisher_infos {
            for (l, (diag_layer, means_layer)) in fi.diag.iter().zip(fi.means.iter()).enumerate() {
                if l >= current_params.len() {
                    break;
                }
                let cur = &current_params[l];
                for (i, (&f_val, &m_val)) in diag_layer.iter().zip(means_layer.iter()).enumerate() {
                    if i < cur.len() {
                        let diff = cur[i] - m_val;
                        total += f_val * diff * diff;
                    }
                }
            }
        }
        (self.lambda / 2.0) * total
    }

    /// Compute the gradient of the EWC penalty w.r.t. `current_params`.
    ///
    /// The gradient for parameter `θ_i` is:
    ///
    /// ```text
    /// ∂L_EWC/∂θ_i = λ Σ_k F̂_i^{(k)} · (θ_i − θ*_i^{(k)})
    /// ```
    ///
    /// Returns a `Vec<Vec<f32>>` with the same shape as `current_params`.
    pub fn penalty_gradients(&self, current_params: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut grads: Vec<Vec<f32>> = current_params
            .iter()
            .map(|l| vec![0.0_f32; l.len()])
            .collect();

        for fi in &self.fisher_infos {
            for (l, (diag_layer, means_layer)) in fi.diag.iter().zip(fi.means.iter()).enumerate() {
                if l >= grads.len() {
                    break;
                }
                let grad_layer = &mut grads[l];
                let cur = &current_params[l];
                for (i, (&f_val, &m_val)) in diag_layer.iter().zip(means_layer.iter()).enumerate() {
                    if i < grad_layer.len() {
                        grad_layer[i] += self.lambda * f_val * (cur[i] - m_val);
                    }
                }
            }
        }

        grads
    }

    /// Online EWC: merge a newly computed `FisherInfo` into the existing
    /// running estimate using an exponential moving average.
    ///
    /// Following Schwarz et al. 2018:
    ///
    /// ```text
    /// F̂_new = γ · F̂_old + (1 − γ) · F̂_task
    /// ```
    ///
    /// If no tasks have been registered yet the new Fisher is simply pushed as
    /// the first entry.
    ///
    /// `gamma` is the decay factor in `[0, 1)`. Typical values: 0.9 – 0.99.
    pub fn online_update(&mut self, new_fisher: FisherInfo, gamma: f32) {
        if self.fisher_infos.is_empty() {
            self.fisher_infos.push(new_fisher);
            return;
        }

        // Update the last (running) estimate in place.
        let running = self
            .fisher_infos
            .last_mut()
            .expect("checked non-empty above");

        for (l, (diag_layer, new_diag_layer)) in running
            .diag
            .iter_mut()
            .zip(new_fisher.diag.iter())
            .enumerate()
        {
            let _ = l;
            for (v, &nv) in diag_layer.iter_mut().zip(new_diag_layer.iter()) {
                *v = gamma * (*v) + (1.0 - gamma) * nv;
            }
        }

        // Update means to the new optimal parameters.
        for (l, (mean_layer, new_mean_layer)) in running
            .means
            .iter_mut()
            .zip(new_fisher.means.iter())
            .enumerate()
        {
            let _ = l;
            for (m, &nm) in mean_layer.iter_mut().zip(new_mean_layer.iter()) {
                *m = nm;
            }
        }

        running.task_id = new_fisher.task_id;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synaptic Intelligence (SI)
// ─────────────────────────────────────────────────────────────────────────────

/// Synaptic Intelligence (Zenke et al. 2017).
///
/// Accumulates an online importance estimate `ω` during training by tracking
/// the *path integral* of each parameter's contribution to the loss decrease.
/// No extra gradient computation pass over the dataset is required.
///
/// Usage pattern:
///
/// 1. Call [`update_w`] after every gradient step to accumulate `w`.
/// 2. Call [`consolidate`] once after each task to update `ω` and reset `w`.
/// 3. Use [`penalty`] as an additional loss term during training.
///
/// [`update_w`]: SynapticIntelligence::update_w
/// [`consolidate`]: SynapticIntelligence::consolidate
/// [`penalty`]: SynapticIntelligence::penalty
#[derive(Debug, Clone)]
pub struct SynapticIntelligence {
    /// Regularisation strength `c` (analogous to λ in EWC).
    pub c: f32,
    /// Damping term `ξ` to avoid division by zero when computing `ω`.
    pub xi: f32,
    /// Accumulated importance per parameter (updated at task boundaries).
    pub omega: Vec<Vec<f32>>,
    /// Parameter snapshot at the start / end of the previous task.
    pub prev_params: Vec<Vec<f32>>,
    /// Running path integral `w` (accumulated within the current task).
    pub w: Vec<Vec<f32>>,
}

impl SynapticIntelligence {
    /// Create a new SI instance.
    ///
    /// `param_shapes` gives the number of parameters per layer.
    pub fn new(c: f32, xi: f32, param_shapes: &[usize]) -> Self {
        let omega: Vec<Vec<f32>> = param_shapes.iter().map(|&n| vec![0.0_f32; n]).collect();
        let prev_params: Vec<Vec<f32>> = param_shapes.iter().map(|&n| vec![0.0_f32; n]).collect();
        let w: Vec<Vec<f32>> = param_shapes.iter().map(|&n| vec![0.0_f32; n]).collect();
        SynapticIntelligence {
            c,
            xi,
            omega,
            prev_params,
            w,
        }
    }

    /// Update the running path integral `w` after one gradient step.
    ///
    /// The contribution of a single step with learning rate `lr` is:
    ///
    /// ```text
    /// w_i  ←  w_i  −  g_i · Δθ_i
    /// ```
    ///
    /// where `Δθ_i = lr · g_i` (gradient descent step), giving:
    ///
    /// ```text
    /// w_i  ←  w_i  +  g_i · (θ_i_new − θ_i_old)
    /// ```
    ///
    /// In practice we approximate this as `w_i ← w_i + g_i² · lr` because
    /// after a vanilla SGD step `Δθ_i = lr · g_i`.
    pub fn update_w(&mut self, params: &[Vec<f32>], grads: &[Vec<f32>], lr: f32) {
        for (l, (w_layer, (param_layer, grad_layer))) in self
            .w
            .iter_mut()
            .zip(params.iter().zip(grads.iter()))
            .enumerate()
        {
            let _ = l;
            for (i, (&p, &g)) in param_layer.iter().zip(grad_layer.iter()).enumerate() {
                if i < w_layer.len() {
                    // delta_theta = -lr * g  (gradient descent)
                    // contribution = -g * delta_theta = lr * g²
                    let delta_theta = -lr * g;
                    w_layer[i] -= g * delta_theta;
                    let _ = p;
                }
            }
        }
    }

    /// Update `ω` at the end of a task using the accumulated path integral `w`
    /// and the parameter displacement since the last task boundary.
    ///
    /// ```text
    /// ω_i  ←  ω_i  +  max(w_i, 0) / ((θ_i − θ*_i)² + ξ)
    /// ```
    ///
    /// Afterwards, `prev_params` is updated to `current_params` and `w` is reset.
    pub fn consolidate(&mut self, current_params: &[Vec<f32>]) {
        for (l, (omega_layer, (w_layer, (prev_layer, cur_layer)))) in self
            .omega
            .iter_mut()
            .zip(
                self.w
                    .iter_mut()
                    .zip(self.prev_params.iter_mut().zip(current_params.iter())),
            )
            .enumerate()
        {
            let _ = l;
            for i in 0..omega_layer.len() {
                if i < cur_layer.len() {
                    let delta = cur_layer[i] - prev_layer[i];
                    let denom = delta * delta + self.xi;
                    let w_val = w_layer[i].max(0.0);
                    omega_layer[i] += w_val / denom;

                    // Reset path integral and update snapshot.
                    w_layer[i] = 0.0;
                    prev_layer[i] = cur_layer[i];
                }
            }
        }
    }

    /// Compute the SI regularisation penalty.
    ///
    /// ```text
    /// L_SI = c · Σ_i  ω_i · (θ_i − θ*_i)²
    /// ```
    pub fn penalty(&self, current_params: &[Vec<f32>]) -> f32 {
        let mut total = 0.0_f32;
        for (l, (omega_layer, (prev_layer, cur_layer))) in self
            .omega
            .iter()
            .zip(self.prev_params.iter().zip(current_params.iter()))
            .enumerate()
        {
            let _ = l;
            for i in 0..omega_layer.len() {
                if i < cur_layer.len() {
                    let diff = cur_layer[i] - prev_layer[i];
                    total += omega_layer[i] * diff * diff;
                }
            }
        }
        self.c * total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Progressive Neural Networks
// ─────────────────────────────────────────────────────────────────────────────

/// A single column in a Progressive Neural Network.
///
/// Each column learns an independent task.  Lateral connections from all
/// earlier columns are added at each layer to allow forward knowledge transfer
/// without overwriting prior knowledge.
#[derive(Debug, Clone)]
pub struct PnnColumn {
    /// Layer-wise parameter tensors (flattened weight matrices).
    pub layers: Vec<Vec<f32>>,
    /// Lateral adapter parameters from previous columns at each layer.
    /// `lateral_connections[l]` has length `prev_columns * layer_sizes[l]`.
    pub lateral_connections: Vec<Vec<f32>>,
    /// Number of hidden layers.
    pub num_layers: usize,
    /// Width (number of units) at each layer.
    pub layer_sizes: Vec<usize>,
}

impl PnnColumn {
    /// Create a new column with the given layer widths and a specified number
    /// of previous columns (for lateral connection sizing).
    fn new(layer_sizes: Vec<usize>, num_prev_columns: usize) -> Self {
        let num_layers = layer_sizes.len();
        // Each layer has a weight matrix of shape [layer_sizes[l], layer_sizes[l-1]].
        // We store it flattened.  Input layer has no incoming weights from within
        // the column itself, so we use a placeholder of size 0.
        let layers: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                if l == 0 {
                    vec![]
                } else {
                    vec![0.0_f32; layer_sizes[l] * layer_sizes[l - 1]]
                }
            })
            .collect();

        // Lateral connections: for each hidden layer l > 0, shape is
        // [layer_sizes[l], num_prev_columns * layer_sizes[l]].
        let lateral_connections: Vec<Vec<f32>> = (0..num_layers)
            .map(|l| {
                if l == 0 || num_prev_columns == 0 {
                    vec![]
                } else {
                    vec![0.0_f32; layer_sizes[l] * num_prev_columns * layer_sizes[l]]
                }
            })
            .collect();

        PnnColumn {
            layers,
            lateral_connections,
            num_layers,
            layer_sizes,
        }
    }
}

/// Progressive Neural Network.
///
/// A PNN grows one column per task.  All previously learned columns are frozen
/// after their respective tasks.  The current column receives lateral input
/// from every earlier column at each layer.
///
/// This architecture completely prevents catastrophic forgetting at the cost of
/// linear model growth in the number of tasks.
#[derive(Debug, Clone)]
pub struct ProgressiveNeuralNetwork {
    /// Ordered list of columns; index equals task id.
    pub columns: Vec<PnnColumn>,
}

impl ProgressiveNeuralNetwork {
    /// Create an empty PNN (no tasks registered yet).
    pub fn new() -> Self {
        ProgressiveNeuralNetwork {
            columns: Vec::new(),
        }
    }

    /// Add a column for a new task.
    ///
    /// `layer_sizes` specifies the width at each layer (including the input
    /// layer at index 0).  Must be non-empty.
    pub fn add_column(&mut self, layer_sizes: Vec<usize>) -> Result<(), ContinualError> {
        if layer_sizes.is_empty() {
            return Err(ContinualError::DimensionMismatch {
                layer: 0,
                expected: 1,
                found: 0,
            });
        }
        let num_prev = self.columns.len();
        let column = PnnColumn::new(layer_sizes, num_prev);
        self.columns.push(column);
        Ok(())
    }

    /// Return the number of tasks (columns) in the network.
    pub fn num_tasks(&self) -> usize {
        self.columns.len()
    }

    /// Get a reference to a column by task id.  Returns `None` if the id is
    /// out of range.
    pub fn column(&self, task_id: usize) -> Option<&PnnColumn> {
        self.columns.get(task_id)
    }
}

impl Default for ProgressiveNeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PackNet
// ─────────────────────────────────────────────────────────────────────────────

/// PackNet (Mallya & Lazebnik 2018).
///
/// After training on each task, the least-important (smallest magnitude)
/// fraction `prune_fraction` of the *currently free* weights is pruned and
/// assigned permanently to the current task.  Weights already assigned to
/// earlier tasks are frozen; only free weights can be trained for new tasks.
///
/// This lets a single fixed-capacity network accommodate multiple tasks without
/// any forgetting, as long as sufficient capacity remains.
#[derive(Debug, Clone)]
pub struct PackNet {
    /// Binary masks per task per layer: `task_masks[t][l][i]` is `true` if
    /// parameter `i` of layer `l` is allocated to task `t`.
    pub task_masks: Vec<Vec<Vec<bool>>>,
    /// Index of the task currently being trained.
    pub current_task: usize,
    /// Fraction of free weights to prune (and thus assign) after each task.
    pub prune_fraction: f32,
}

impl PackNet {
    /// Create a new PackNet with the given pruning fraction.
    ///
    /// `prune_fraction` must be in `(0, 1]`.
    pub fn new(prune_fraction: f32) -> Self {
        PackNet {
            task_masks: Vec::new(),
            current_task: 0,
            prune_fraction: prune_fraction.clamp(f32::EPSILON, 1.0),
        }
    }

    /// Prune and assign weights to task `task_id` after training.
    ///
    /// Among the currently free parameters (those not yet assigned to any
    /// task), the `prune_fraction` with the *smallest absolute value* are
    /// assigned to `task_id`.
    ///
    /// Returns the binary mask for this task.  An entry is `true` iff the
    /// corresponding parameter belongs to this task.
    pub fn prune_for_task(&mut self, params: &[Vec<f32>], task_id: usize) -> Vec<Vec<bool>> {
        // Build a mask indicating which parameters are already used.
        let mut used: Vec<Vec<bool>> = params.iter().map(|l| vec![false; l.len()]).collect();

        for prev_mask in &self.task_masks {
            for (l, layer_mask) in prev_mask.iter().enumerate() {
                if l < used.len() {
                    for (i, &taken) in layer_mask.iter().enumerate() {
                        if i < used[l].len() && taken {
                            used[l][i] = true;
                        }
                    }
                }
            }
        }

        // Collect (layer_index, param_index, abs_value) for free parameters.
        let mut free_params: Vec<(usize, usize, f32)> = Vec::new();
        for (l, (layer_params, used_layer)) in params.iter().zip(used.iter()).enumerate() {
            for (i, (&v, &u)) in layer_params.iter().zip(used_layer.iter()).enumerate() {
                if !u {
                    free_params.push((l, i, v.abs()));
                }
            }
        }

        let n_to_prune = ((free_params.len() as f32) * self.prune_fraction).ceil() as usize;

        // Sort by absolute value ascending; smallest-magnitude = least important.
        free_params.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Assign the n_to_prune smallest free params to this task.
        let mut new_mask: Vec<Vec<bool>> = params.iter().map(|l| vec![false; l.len()]).collect();
        for (l, i, _) in free_params.iter().take(n_to_prune) {
            if *l < new_mask.len() && *i < new_mask[*l].len() {
                new_mask[*l][*i] = true;
            }
        }

        // Store the mask and advance current task counter.
        // Ensure the task_masks vector has enough entries.
        while self.task_masks.len() <= task_id {
            let empty: Vec<Vec<bool>> = params.iter().map(|l| vec![false; l.len()]).collect();
            self.task_masks.push(empty);
        }
        self.task_masks[task_id] = new_mask.clone();
        self.current_task = task_id + 1;

        new_mask
    }

    /// Apply a task-specific binary mask to `params`.
    ///
    /// Parameters not belonging to task `task_id` are zeroed out.  This
    /// simulates running inference with only the sub-network carved out for
    /// that task.
    pub fn apply_mask(&self, params: &[Vec<f32>], task_id: usize) -> Vec<Vec<f32>> {
        if task_id >= self.task_masks.len() {
            // Unknown task: return a copy of params unchanged.
            return params.to_vec();
        }
        let mask = &self.task_masks[task_id];
        params
            .iter()
            .enumerate()
            .map(|(l, layer)| {
                if l >= mask.len() {
                    layer.clone()
                } else {
                    layer
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            if i < mask[l].len() && mask[l][i] {
                                v
                            } else {
                                0.0
                            }
                        })
                        .collect()
                }
            })
            .collect()
    }

    /// Fraction of total parameters that are still free (not assigned to any task).
    ///
    /// Returns `1.0` when no tasks have been registered, `0.0` when all
    /// parameters have been assigned.
    pub fn free_capacity(&self, task_id: usize) -> f32 {
        if self.task_masks.is_empty() {
            return 1.0;
        }

        // Count tasks up to and including task_id.
        let up_to = (task_id + 1).min(self.task_masks.len());

        // We need a total parameter count to normalise.
        let total: usize = self
            .task_masks
            .first()
            .map(|m| m.iter().map(|l| l.len()).sum())
            .unwrap_or(0);

        if total == 0 {
            return 1.0;
        }

        let assigned: usize = self.task_masks[..up_to]
            .iter()
            .map(|mask| {
                mask.iter()
                    .map(|l| l.iter().filter(|&&b| b).count())
                    .sum::<usize>()
            })
            .sum();

        1.0 - (assigned as f32 / total as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    /// Build a simple two-layer parameter vector of shape [3, 2].
    fn two_layer_params() -> Vec<Vec<f32>> {
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]]
    }

    fn two_layer_means() -> Vec<Vec<f32>> {
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]]
    }

    fn fisher_unit_diag(means: Vec<Vec<f32>>) -> FisherInfo {
        let diag: Vec<Vec<f32>> = means.iter().map(|l| vec![1.0_f32; l.len()]).collect();
        FisherInfo {
            diag,
            means,
            task_id: 0,
        }
    }

    // ── EWC tests ────────────────────────────────────────────────────────────

    #[test]
    fn ewc_zero_penalty_same_params() {
        let params = two_layer_params();
        let fisher = fisher_unit_diag(two_layer_means());
        let mut ewc = EwcRegularizer::new(1000.0);
        ewc.register_task(fisher);
        let penalty = ewc.penalty(&params);
        assert!(
            penalty.abs() < 1e-5,
            "penalty should be ~0 when params equal means, got {}",
            penalty
        );
    }

    #[test]
    fn ewc_positive_penalty_different_params() {
        let means = two_layer_means();
        let fisher = fisher_unit_diag(means.clone());
        let mut ewc = EwcRegularizer::new(1000.0);
        ewc.register_task(fisher);

        // Perturb params away from means.
        let perturbed: Vec<Vec<f32>> = means
            .iter()
            .map(|l| l.iter().map(|&v| v + 1.0).collect())
            .collect();
        let penalty = ewc.penalty(&perturbed);
        assert!(
            penalty > 0.0,
            "penalty should be positive for perturbed params, got {}",
            penalty
        );
    }

    #[test]
    fn ewc_penalty_scales_with_lambda() {
        let params = two_layer_params();
        let means = two_layer_means();
        let perturbed: Vec<Vec<f32>> = params
            .iter()
            .map(|l| l.iter().map(|&v| v + 1.0).collect())
            .collect();

        let fisher1 = fisher_unit_diag(means.clone());
        let mut ewc1 = EwcRegularizer::new(1000.0);
        ewc1.register_task(fisher1);
        let p1 = ewc1.penalty(&perturbed);

        let fisher2 = fisher_unit_diag(means.clone());
        let mut ewc2 = EwcRegularizer::new(2000.0);
        ewc2.register_task(fisher2);
        let p2 = ewc2.penalty(&perturbed);

        let ratio = p2 / p1;
        assert!(
            (ratio - 2.0).abs() < 1e-4,
            "penalty should scale linearly with lambda, ratio={}",
            ratio
        );
    }

    #[test]
    fn ewc_penalty_gradient_direction() {
        let means = two_layer_means();
        // Shift params: params[0][0] is 1.0 above mean.
        let params: Vec<Vec<f32>> = vec![vec![2.0, 2.0, 3.0], vec![4.0, 5.0]];
        let fisher = fisher_unit_diag(means);
        let mut ewc = EwcRegularizer::new(1.0);
        ewc.register_task(fisher);

        let grads = ewc.penalty_gradients(&params);
        // grad[0][0] should be λ * F * (θ - θ*) = 1 * 1 * 1 = 1.0
        assert!(
            (grads[0][0] - 1.0).abs() < 1e-5,
            "gradient at shifted param should be 1.0, got {}",
            grads[0][0]
        );
        // Other params equal means → grad should be 0.
        assert!(
            grads[0][1].abs() < 1e-5,
            "gradient at unshifted param should be 0, got {}",
            grads[0][1]
        );
    }

    #[test]
    fn ewc_gradient_shape_matches_params() {
        let params = two_layer_params();
        let fisher = fisher_unit_diag(two_layer_means());
        let mut ewc = EwcRegularizer::new(100.0);
        ewc.register_task(fisher);
        let grads = ewc.penalty_gradients(&params);
        assert_eq!(grads.len(), params.len());
        for (g_layer, p_layer) in grads.iter().zip(params.iter()) {
            assert_eq!(g_layer.len(), p_layer.len());
        }
    }

    #[test]
    fn ewc_online_update_blends_fisher() {
        let means = two_layer_means();
        let fisher1 = FisherInfo {
            diag: vec![vec![2.0, 2.0, 2.0], vec![2.0, 2.0]],
            means: means.clone(),
            task_id: 0,
        };
        let mut ewc = EwcRegularizer::new(1.0);
        ewc.register_task(fisher1);

        let fisher2 = FisherInfo {
            diag: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0]],
            means: means.clone(),
            task_id: 1,
        };

        ewc.online_update(fisher2, 0.5);

        // After blending: F_new = 0.5 * 2.0 + 0.5 * 0.0 = 1.0
        let diag0 = &ewc.fisher_infos[0].diag[0][0];
        assert!(
            (*diag0 - 1.0).abs() < 1e-5,
            "blended Fisher value should be 1.0, got {}",
            diag0
        );
    }

    #[test]
    fn ewc_multiple_tasks_accumulate() {
        let means = two_layer_means();
        let mut ewc = EwcRegularizer::new(1.0);
        ewc.register_task(fisher_unit_diag(means.clone()));
        ewc.register_task(fisher_unit_diag(means.clone()));

        // Both tasks have same means; perturb params.
        let perturbed: Vec<Vec<f32>> = means
            .iter()
            .map(|l| l.iter().map(|&v| v + 1.0).collect())
            .collect();
        let penalty = ewc.penalty(&perturbed);

        // Each task contributes λ/2 * Σ F * Δ² = 0.5 * (3+2) * 1 = 2.5
        // Two tasks → 5.0
        assert!(
            (penalty - 5.0).abs() < 1e-4,
            "two-task penalty should sum contributions, got {}",
            penalty
        );
    }

    // ── Fisher estimation tests ───────────────────────────────────────────────

    #[test]
    fn fisher_estimate_shape() {
        // 4 samples, 2 layers of sizes [3, 2].
        let grads: Vec<Vec<Vec<f32>>> = (0..4)
            .map(|_| vec![vec![1.0_f32; 3], vec![1.0_f32; 2]])
            .collect();
        let fi = FisherInfo::estimate(&grads);
        assert_eq!(fi.diag.len(), 2);
        assert_eq!(fi.diag[0].len(), 3);
        assert_eq!(fi.diag[1].len(), 2);
    }

    #[test]
    fn fisher_estimate_value() {
        // Constant gradient of 2.0 over 2 samples → F = 4.0.
        let grads: Vec<Vec<Vec<f32>>> = (0..2).map(|_| vec![vec![2.0_f32; 3]]).collect();
        let fi = FisherInfo::estimate(&grads);
        assert!(
            (fi.diag[0][0] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            fi.diag[0][0]
        );
    }

    #[test]
    fn fisher_estimate_empty() {
        let fi = FisherInfo::estimate(&[]);
        assert!(fi.diag.is_empty());
    }

    // ── SI tests ─────────────────────────────────────────────────────────────

    #[test]
    fn si_omega_initially_zero() {
        let si = SynapticIntelligence::new(1.0, 1e-3, &[3, 2]);
        for layer in &si.omega {
            for &v in layer {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn si_penalty_zero_before_consolidate() {
        let si = SynapticIntelligence::new(1.0, 1e-3, &[3, 2]);
        let params = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let penalty = si.penalty(&params);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn si_consolidate_sets_omega() {
        let mut si = SynapticIntelligence::new(1.0, 0.0, &[2]);
        // Set up prev_params = [0, 0] and accumulate some w.
        si.w[0] = vec![2.0, 3.0];
        // Consolidate with current params = [1, 1].
        si.consolidate(&[vec![1.0, 1.0]]);
        // omega = max(w, 0) / ((delta)^2 + xi) = 2.0 / (1.0) = 2.0 for first param.
        assert!(
            (si.omega[0][0] - 2.0).abs() < 1e-5,
            "omega[0][0] should be 2.0, got {}",
            si.omega[0][0]
        );
    }

    #[test]
    fn si_consolidate_resets_w() {
        let mut si = SynapticIntelligence::new(1.0, 1e-3, &[2]);
        si.w[0] = vec![5.0, 5.0];
        si.consolidate(&[vec![1.0, 1.0]]);
        for &v in &si.w[0] {
            assert_eq!(v, 0.0, "w should be reset to 0 after consolidation");
        }
    }

    #[test]
    fn si_penalty_positive_after_param_change() {
        let mut si = SynapticIntelligence::new(1.0, 0.0, &[1]);
        si.w[0] = vec![1.0];
        // prev_params = [0.0], consolidate with current = [1.0].
        si.consolidate(&[vec![1.0]]);
        // Now move params away from prev_params.
        let penalty = si.penalty(&[vec![2.0]]);
        assert!(
            penalty > 0.0,
            "SI penalty should be positive after param move, got {}",
            penalty
        );
    }

    #[test]
    fn si_update_w_accumulates() {
        let mut si = SynapticIntelligence::new(1.0, 1e-3, &[2]);
        let params = vec![vec![1.0_f32, 1.0]];
        let grads = vec![vec![1.0_f32, 0.5]];
        si.update_w(&params, &grads, 0.1);
        // w[0][0] = 0.0 + 0.1 * 1.0^2 = 0.1
        assert!(
            si.w[0][0] > 0.0,
            "w should be positive after update, got {}",
            si.w[0][0]
        );
    }

    // ── PNN tests ─────────────────────────────────────────────────────────────

    #[test]
    fn pnn_starts_empty() {
        let pnn = ProgressiveNeuralNetwork::new();
        assert_eq!(pnn.num_tasks(), 0);
    }

    #[test]
    fn pnn_add_column_increments_task_count() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        pnn.add_column(vec![4, 8, 2]).expect("first column");
        assert_eq!(pnn.num_tasks(), 1);
        pnn.add_column(vec![4, 8, 2]).expect("second column");
        assert_eq!(pnn.num_tasks(), 2);
    }

    #[test]
    fn pnn_column_retrieval() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        pnn.add_column(vec![4, 8, 2]).expect("add column");
        assert!(pnn.column(0).is_some());
        assert!(pnn.column(1).is_none());
    }

    #[test]
    fn pnn_column_layer_sizes_stored() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        pnn.add_column(vec![4, 8, 2]).expect("add column");
        let col = pnn.column(0).expect("column 0");
        assert_eq!(col.layer_sizes, vec![4, 8, 2]);
        assert_eq!(col.num_layers, 3);
    }

    #[test]
    fn pnn_second_column_has_lateral_connections() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        pnn.add_column(vec![4, 8, 2]).expect("col 0");
        pnn.add_column(vec![4, 8, 2]).expect("col 1");
        let col1 = pnn.column(1).expect("column 1");
        // Layer 1 should have lateral connections from column 0.
        assert!(
            col1.lateral_connections[1].len() > 0,
            "column 1 should have lateral connections at layer 1"
        );
    }

    #[test]
    fn pnn_empty_layer_sizes_returns_error() {
        let mut pnn = ProgressiveNeuralNetwork::new();
        let result = pnn.add_column(vec![]);
        assert!(result.is_err(), "empty layer_sizes should return error");
    }

    // ── PackNet tests ─────────────────────────────────────────────────────────

    #[test]
    fn packnet_prune_fraction_correct() {
        let params: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]];
        let mut pn = PackNet::new(0.3);
        let mask = pn.prune_for_task(&params, 0);
        let assigned: usize = mask.iter().map(|l| l.iter().filter(|&&b| b).count()).sum();
        // 30% of 10 = 3 (ceiling)
        assert_eq!(
            assigned, 3,
            "should assign 3 params for 30% prune, got {}",
            assigned
        );
    }

    #[test]
    fn packnet_second_task_uses_remaining_free() {
        let params: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]];
        let mut pn = PackNet::new(0.5);
        let mask0 = pn.prune_for_task(&params, 0);
        let mask1 = pn.prune_for_task(&params, 1);

        // No parameter should be in both masks.
        for (l, (m0, m1)) in mask0.iter().zip(mask1.iter()).enumerate() {
            for (i, (&b0, &b1)) in m0.iter().zip(m1.iter()).enumerate() {
                assert!(!(b0 && b1), "layer {} param {} is double-assigned", l, i);
            }
        }
    }

    #[test]
    fn packnet_apply_mask_zeroes_other_params() {
        let params: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut pn = PackNet::new(0.5);
        pn.prune_for_task(&params, 0);

        let masked = pn.apply_mask(&params, 0);
        let assigned_sum: f32 = masked[0].iter().sum();
        // Only the 2 smallest-magnitude params (1.0, 2.0) are assigned to task 0.
        // sum = 1.0 + 2.0 = 3.0 (others zeroed out)
        assert!(
            (assigned_sum - 3.0).abs() < 1e-5,
            "masked sum should be 3.0, got {}",
            assigned_sum
        );
    }

    #[test]
    fn packnet_free_capacity_decreases() {
        let params: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut pn = PackNet::new(0.5);
        let cap_before = pn.free_capacity(0);
        pn.prune_for_task(&params, 0);
        let cap_after = pn.free_capacity(0);
        assert!(
            cap_after < cap_before,
            "free capacity should decrease after pruning, before={}, after={}",
            cap_before,
            cap_after
        );
    }

    #[test]
    fn packnet_unknown_task_returns_full_params() {
        let params: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        let pn = PackNet::new(0.5);
        let result = pn.apply_mask(&params, 99);
        assert_eq!(
            result[0], params[0],
            "unknown task should return unchanged params"
        );
    }

    #[test]
    fn continual_error_display() {
        let err = ContinualError::TaskNotFound { task_id: 3 };
        let msg = err.to_string();
        assert!(msg.contains("3"), "error message should contain task id");

        let err2 = ContinualError::DimensionMismatch {
            layer: 1,
            expected: 10,
            found: 5,
        };
        let msg2 = err2.to_string();
        assert!(msg2.contains("1") && msg2.contains("10") && msg2.contains("5"));
    }
}
