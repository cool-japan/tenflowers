//! Multi-Task Learning (MTL) utilities.
//!
//! This module provides building blocks for training a single model on
//! multiple related tasks simultaneously.  Multi-task learning regularises
//! representations through shared inductive biases and often outperforms
//! per-task single-task models in both accuracy and data efficiency.
//!
//! # Algorithms
//!
//! * [`MultiTaskLoss`] — Unified loss combination with four weighting strategies:
//!   [`TaskWeighting::Equal`], [`TaskWeighting::Uncertainty`] (Kendall et al.
//!   2018), [`TaskWeighting::GradNorm`] (Chen et al. 2018), and
//!   [`TaskWeighting::Dynamic`] (user-supplied weights).
//!
//! * [`pcgrad_project`] — PCGrad (Yu et al. 2020): projects each task's
//!   gradient onto the normal plane of conflicting gradients, reducing
//!   destructive interference between tasks.
//!
//! * [`MultiTaskModel`] — Combines a shared backbone (represented by its
//!   output dimension) with per-task [`TaskHead`] projections and a
//!   [`MultiTaskLoss`] combiner.
//!
//! * [`AuxiliaryTaskScheduler`] — Decays auxiliary task weights over epochs
//!   so that the main task gradually dominates training.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::multi_task::{MultiTaskModel, TaskHead, MultiTaskLoss, TaskWeighting};
//!
//! let mut model = MultiTaskModel::new(128);
//! model.add_task(TaskHead::new("classification", 128, 10));
//! model.add_task(TaskHead::new("regression", 128, 1));
//!
//! let repr = vec![0.0_f32; 128];
//! let outputs = model.forward_all(&repr);
//! ```

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise in multi-task learning operations.
#[derive(Debug)]
pub enum MultiTaskError {
    /// No task with the given name exists.
    TaskNotFound { name: String },
    /// A task's prediction/target dimension doesn't match expectations.
    DimensionMismatch {
        task: String,
        expected: usize,
        found: usize,
    },
    /// The model has no task heads registered.
    NoTasks,
    /// The number of supplied weights doesn't match the number of tasks.
    WeightMismatch {
        num_tasks: usize,
        num_weights: usize,
    },
}

impl fmt::Display for MultiTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiTaskError::TaskNotFound { name } => {
                write!(f, "task '{}' not found", name)
            }
            MultiTaskError::DimensionMismatch {
                task,
                expected,
                found,
            } => {
                write!(
                    f,
                    "task '{}': expected dimension {} but found {}",
                    task, expected, found
                )
            }
            MultiTaskError::NoTasks => write!(f, "no task heads have been added to the model"),
            MultiTaskError::WeightMismatch {
                num_tasks,
                num_weights,
            } => {
                write!(
                    f,
                    "weight count mismatch: {} tasks but {} weights provided",
                    num_tasks, num_weights
                )
            }
        }
    }
}

impl std::error::Error for MultiTaskError {}

// ─────────────────────────────────────────────────────────────────────────────
// Task weighting strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for combining per-task losses into a single scalar.
#[derive(Debug, Clone)]
pub enum TaskWeighting {
    /// All tasks receive weight `1 / num_tasks` (uniform).
    Equal,
    /// Uncertainty weighting (Kendall et al. 2018): each task has a learned
    /// log-variance `σ_t²`.  The loss for task `t` is
    /// `L_t / (2σ_t²) + log σ_t`.
    /// The inner `Vec<f32>` stores initial log-variance values.
    Uncertainty(Vec<f32>),
    /// GradNorm (Chen et al. 2018): dynamically re-weights tasks so that each
    /// task's gradient norm grows at the same rate relative to the shared
    /// backbone.  `alpha` controls the asymmetry strength (commonly 1.5).
    GradNorm { alpha: f32 },
    /// User-supplied static weights (one per task).
    Dynamic(Vec<f32>),
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiTaskLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Combined multi-task loss with pluggable weighting.
///
/// Maintains internal state for learnable weighting strategies (e.g.
/// [`TaskWeighting::Uncertainty`]) and exposes a simple [`combine`] interface
/// that maps a per-task loss vector to a scalar.
///
/// [`combine`]: MultiTaskLoss::combine
#[derive(Debug, Clone)]
pub struct MultiTaskLoss {
    /// Active weighting strategy.
    pub weighting: TaskWeighting,
    /// Number of tasks.
    pub num_tasks: usize,
    /// Learned log-variance parameters used by the `Uncertainty` strategy.
    /// For other strategies this vector is empty.
    pub log_vars: Vec<f32>,
}

impl MultiTaskLoss {
    /// Create a new multi-task loss combiner.
    ///
    /// For `TaskWeighting::Uncertainty(init_log_vars)`, the initial log-variance
    /// values must have exactly `num_tasks` elements; they are copied into
    /// `log_vars`.  For other strategies `log_vars` is initialised to zeros.
    pub fn new(num_tasks: usize, weighting: TaskWeighting) -> Self {
        let log_vars = match &weighting {
            TaskWeighting::Uncertainty(init) => init.clone(),
            _ => vec![0.0_f32; num_tasks],
        };
        MultiTaskLoss {
            weighting,
            num_tasks,
            log_vars,
        }
    }

    /// Combine per-task losses `task_losses` into a single scalar using the
    /// configured weighting strategy.
    ///
    /// The returned value is the weighted total loss ready for
    /// backpropagation.
    pub fn combine(&mut self, task_losses: &[f32]) -> f32 {
        let n = self.num_tasks;
        match &self.weighting.clone() {
            TaskWeighting::Equal => {
                let w = if n > 0 { 1.0 / n as f32 } else { 0.0 };
                task_losses.iter().map(|&l| w * l).sum()
            }
            TaskWeighting::Uncertainty(_) => {
                // L = Σ_t  L_t / (2 * exp(log_var_t))  +  (1/2) * log_var_t
                // Numerically stable: use exp(log_var) = precision.
                task_losses
                    .iter()
                    .zip(self.log_vars.iter())
                    .map(|(&lt, &lv)| {
                        let precision = (-lv).exp(); // 1 / σ_t²
                        lt * precision * 0.5 + lv * 0.5
                    })
                    .sum()
            }
            TaskWeighting::GradNorm { .. } => {
                // GradNorm weights are updated externally; here we fall back to
                // equal weighting since gradient norms require access to the
                // actual gradient tensors which are not available in this
                // scalar-only interface.
                let w = if n > 0 { 1.0 / n as f32 } else { 0.0 };
                task_losses.iter().map(|&l| w * l).sum()
            }
            TaskWeighting::Dynamic(weights) => {
                let weights = weights.clone();
                task_losses
                    .iter()
                    .zip(weights.iter().chain(std::iter::repeat(&1.0)))
                    .map(|(&l, &w)| l * w)
                    .sum()
            }
        }
    }

    /// Return the effective per-task weight vector given the current state.
    ///
    /// For `Uncertainty` weighting, weights are `1/(2σ_t²)` (precision halves).
    /// For `GradNorm` weighting, equal weights are returned (runtime update
    /// requires gradient access).
    pub fn weights(&self) -> Vec<f32> {
        let n = self.num_tasks;
        match &self.weighting {
            TaskWeighting::Equal | TaskWeighting::GradNorm { .. } => {
                let w = if n > 0 { 1.0 / n as f32 } else { 0.0 };
                vec![w; n]
            }
            TaskWeighting::Uncertainty(_) => {
                self.log_vars.iter().map(|&lv| 0.5 * (-lv).exp()).collect()
            }
            TaskWeighting::Dynamic(weights) => weights.clone(),
        }
    }

    /// Perform one gradient-based update step for the uncertainty `log_vars`.
    ///
    /// The gradient of the uncertainty loss w.r.t. `log_var_t` is:
    ///
    /// ```text
    /// ∂L / ∂log_var_t = 0.5 - 0.5 * L_t * exp(-log_var_t)
    ///                  = 0.5 * (1 - L_t / σ_t²)
    /// ```
    ///
    /// Only modifies `log_vars` when `TaskWeighting::Uncertainty` is active.
    pub fn update_uncertainty_weights(&mut self, task_losses: &[f32], lr: f32) {
        if !matches!(self.weighting, TaskWeighting::Uncertainty(_)) {
            return;
        }
        for (lv, &lt) in self.log_vars.iter_mut().zip(task_losses.iter()) {
            // Gradient ascent on log_var to minimise L (gradient descent on -L).
            let grad = 0.5 * (1.0 - lt * (-*lv).exp());
            *lv -= lr * grad;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PCGrad — Gradient Surgery
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the cosine similarity between two flat gradient vectors.
///
/// Returns a value in `[-1, 1]` for unit-norm inputs; returns `0.0` when
/// either vector has zero norm.
pub fn gradient_cosine_similarity(g1: &[f32], g2: &[f32]) -> f32 {
    let dot: f32 = g1.iter().zip(g2.iter()).map(|(&a, &b)| a * b).sum();
    let n1: f32 = g1.iter().map(|&a| a * a).sum::<f32>().sqrt();
    let n2: f32 = g2.iter().map(|&b| b * b).sum::<f32>().sqrt();
    if n1 < f32::EPSILON || n2 < f32::EPSILON {
        return 0.0;
    }
    (dot / (n1 * n2)).clamp(-1.0, 1.0)
}

/// PCGrad: Project Conflicting Gradients (Yu et al. 2020).
///
/// For each task `i`, the gradient `g_i` is projected onto the normal plane of
/// `g_j` for every other task `j` that conflicts with it (i.e.
/// `cos(g_i, g_j) < 0`):
///
/// ```text
/// g_i  ←  g_i  −  (g_i · ĝ_j) · ĝ_j     if  g_i · g_j < 0
/// ```
///
/// This removes the component of `g_i` that points in the opposite direction
/// to `g_j`, thus reducing destructive interference without affecting tasks
/// that are compatible.
///
/// All input gradients must have the same length.  Returns projected gradients
/// in the same order as the input.
pub fn pcgrad_project(task_grads: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = task_grads.len();
    if n == 0 {
        return Vec::new();
    }

    let dim = task_grads[0].len();
    let mut projected: Vec<Vec<f32>> = task_grads.to_vec();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Check for conflict.
            let dot: f32 = projected[i]
                .iter()
                .zip(task_grads[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            if dot >= 0.0 {
                // Gradients compatible — no projection needed.
                continue;
            }

            // Compute the unit vector of g_j.
            let norm_j_sq: f32 = task_grads[j].iter().map(|&b| b * b).sum();
            if norm_j_sq < f32::EPSILON {
                continue;
            }

            // Project g_i: g_i ← g_i − (g_i · g_j / |g_j|²) * g_j
            let scale = dot / norm_j_sq;
            for k in 0..dim {
                if k < task_grads[j].len() {
                    projected[i][k] -= scale * task_grads[j][k];
                }
            }
        }
    }

    projected
}

// ─────────────────────────────────────────────────────────────────────────────
// TaskHead
// ─────────────────────────────────────────────────────────────────────────────

/// Task-specific linear output head.
///
/// Projects the shared backbone representation into the task's output space:
///
/// ```text
/// y = W · h + b
/// ```
///
/// where `h ∈ ℝ^{shared_dim}` and `y ∈ ℝ^{out_features}`.
///
/// Weights are stored in row-major order: `weights[out * shared_dim + in]`.
#[derive(Debug, Clone)]
pub struct TaskHead {
    /// Unique task name used for lookup.
    pub task_name: String,
    /// Flattened weight matrix, shape `[out_features * shared_dim]`.
    pub weights: Vec<f32>,
    /// Bias vector, shape `[out_features]`.
    pub bias: Vec<f32>,
    /// Output dimension.
    pub out_features: usize,
    /// Input dimension (shared backbone output size).
    pub shared_dim: usize,
    /// Scalar loss weight for this task (default 1.0).
    pub loss_weight: f32,
}

impl TaskHead {
    /// Create a new task head with zero-initialised weights and biases.
    pub fn new(task_name: impl Into<String>, shared_dim: usize, out_features: usize) -> Self {
        TaskHead {
            task_name: task_name.into(),
            weights: vec![0.0_f32; out_features * shared_dim],
            bias: vec![0.0_f32; out_features],
            out_features,
            shared_dim,
            loss_weight: 1.0,
        }
    }

    /// Run a linear forward pass: `y = W · h + b`.
    ///
    /// `shared_repr` must have length `shared_dim`.  Missing elements are
    /// treated as zero.
    pub fn forward(&self, shared_repr: &[f32]) -> Vec<f32> {
        let mut out = self.bias.clone();
        for o in 0..self.out_features {
            let row_start = o * self.shared_dim;
            let dot: f32 = (0..self.shared_dim)
                .map(|i| {
                    let w = self.weights.get(row_start + i).copied().unwrap_or(0.0);
                    let x = shared_repr.get(i).copied().unwrap_or(0.0);
                    w * x
                })
                .sum();
            out[o] += dot;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiTaskModel
// ─────────────────────────────────────────────────────────────────────────────

/// A multi-task model composed of a shared backbone and per-task heads.
///
/// The backbone produces a shared representation vector of dimension
/// `shared_backbone_dim`.  Each [`TaskHead`] then projects this into its own
/// output space.  The [`MultiTaskLoss`] combiner aggregates per-task scalar
/// losses into a single training objective.
#[derive(Debug, Clone)]
pub struct MultiTaskModel {
    /// Dimension of the shared backbone output.
    pub shared_backbone_dim: usize,
    /// Ordered task-specific output heads.
    pub task_heads: Vec<TaskHead>,
    /// Loss combiner.
    pub loss: MultiTaskLoss,
}

impl MultiTaskModel {
    /// Create a new model with no task heads.  Uses equal task weighting.
    pub fn new(shared_backbone_dim: usize) -> Self {
        MultiTaskModel {
            shared_backbone_dim,
            task_heads: Vec::new(),
            loss: MultiTaskLoss::new(0, TaskWeighting::Equal),
        }
    }

    /// Add a task head to the model.
    ///
    /// The loss combiner is rebuilt after each addition to keep `num_tasks`
    /// in sync, preserving the `Equal` weighting strategy.
    pub fn add_task(&mut self, head: TaskHead) {
        self.task_heads.push(head);
        // Rebuild the loss combiner to reflect the new task count.
        let n = self.task_heads.len();
        self.loss = MultiTaskLoss::new(n, TaskWeighting::Equal);
    }

    /// Number of registered tasks.
    pub fn num_tasks(&self) -> usize {
        self.task_heads.len()
    }

    /// Names of all registered tasks in registration order.
    pub fn task_names(&self) -> Vec<&str> {
        self.task_heads
            .iter()
            .map(|h| h.task_name.as_str())
            .collect()
    }

    /// Run a forward pass through every task head.
    ///
    /// Returns a vector of `(task_name, output)` pairs in registration order.
    pub fn forward_all<'a>(&'a self, shared_repr: &[f32]) -> Vec<(&'a str, Vec<f32>)> {
        self.task_heads
            .iter()
            .map(|head| (head.task_name.as_str(), head.forward(shared_repr)))
            .collect()
    }

    /// Compute the total weighted multi-task loss.
    ///
    /// For each task, the per-task scalar loss is computed by applying
    /// `loss_fn(predictions, targets)`.  The results are then combined via
    /// the configured [`MultiTaskLoss`].
    ///
    /// `targets` is a slice of `(task_name, target_values)` pairs.  Tasks
    /// not present in `targets` are skipped (their loss is zero).
    ///
    /// # Errors
    ///
    /// Returns [`MultiTaskError::NoTasks`] when no heads are registered, or
    /// [`MultiTaskError::TaskNotFound`] when a target task name is not in the
    /// model.
    pub fn total_loss(
        &mut self,
        shared_repr: &[f32],
        targets: &[(&str, &[f32])],
        loss_fn: impl Fn(&[f32], &[f32]) -> f32,
    ) -> Result<f32, MultiTaskError> {
        if self.task_heads.is_empty() {
            return Err(MultiTaskError::NoTasks);
        }

        let mut per_task_losses: Vec<f32> = vec![0.0_f32; self.task_heads.len()];

        for &(name, tgt) in targets {
            // Find the task head index.
            let idx = self
                .task_heads
                .iter()
                .position(|h| h.task_name == name)
                .ok_or_else(|| MultiTaskError::TaskNotFound {
                    name: name.to_string(),
                })?;

            let pred = self.task_heads[idx].forward(shared_repr);
            let task_loss = loss_fn(&pred, tgt);
            per_task_losses[idx] = task_loss * self.task_heads[idx].loss_weight;
        }

        Ok(self.loss.combine(&per_task_losses))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AuxiliaryTaskScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Schedules loss weights for a main task and a set of auxiliary tasks.
///
/// Auxiliary tasks often help during the early phases of training (acting as a
/// regulariser), but should be de-emphasised as training progresses so that
/// the main task objective dominates.  Each call to [`step`] multiplies the
/// auxiliary weights by `(1 - decay_rate)`.
///
/// [`step`]: AuxiliaryTaskScheduler::step
#[derive(Debug, Clone)]
pub struct AuxiliaryTaskScheduler {
    /// Weight for the primary task (fixed).
    pub main_task_weight: f32,
    /// Current weights for each auxiliary task (decayed over time).
    pub aux_weights: Vec<f32>,
    /// Per-step multiplicative decay factor applied to auxiliary weights.
    pub decay_rate: f32,
}

impl AuxiliaryTaskScheduler {
    /// Create a new scheduler.
    ///
    /// `decay_rate` in `[0, 1)`: fraction of the auxiliary weight removed each
    /// epoch.  A value of `0.0` means no decay.
    pub fn new(main_weight: f32, aux_weights: Vec<f32>, decay_rate: f32) -> Self {
        AuxiliaryTaskScheduler {
            main_task_weight: main_weight,
            aux_weights,
            decay_rate: decay_rate.clamp(0.0, 1.0 - f32::EPSILON),
        }
    }

    /// Advance one step: decay all auxiliary weights by `(1 − decay_rate)`.
    pub fn step(&mut self) {
        for w in &mut self.aux_weights {
            *w *= 1.0 - self.decay_rate;
        }
    }

    /// Return the current weight vector: `[main_weight, aux_weight_0, aux_weight_1, …]`.
    pub fn weights(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(1 + self.aux_weights.len());
        out.push(self.main_task_weight);
        out.extend_from_slice(&self.aux_weights);
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Equal weighting ───────────────────────────────────────────────────────

    #[test]
    fn equal_weighting_uniform() {
        let mut loss = MultiTaskLoss::new(4, TaskWeighting::Equal);
        let task_losses = [1.0_f32, 2.0, 3.0, 4.0];
        let combined = loss.combine(&task_losses);
        let expected = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        assert!(
            (combined - expected).abs() < 1e-5,
            "equal weighting should average losses, expected {}, got {}",
            expected,
            combined
        );
    }

    #[test]
    fn equal_weighting_weights_vector() {
        let loss = MultiTaskLoss::new(3, TaskWeighting::Equal);
        let w = loss.weights();
        assert_eq!(w.len(), 3);
        for &wi in &w {
            assert!(
                (wi - 1.0 / 3.0).abs() < 1e-6,
                "each weight should be 1/3, got {}",
                wi
            );
        }
    }

    // ── Uncertainty weighting ─────────────────────────────────────────────────

    #[test]
    fn uncertainty_weighting_differs_from_equal() {
        let init_log_vars = vec![0.0_f32, 2.0]; // different initial uncertainties
        let mut uncertainty_loss = MultiTaskLoss::new(2, TaskWeighting::Uncertainty(init_log_vars));
        let mut equal_loss = MultiTaskLoss::new(2, TaskWeighting::Equal);

        let task_losses = [1.0_f32, 1.0];
        let u = uncertainty_loss.combine(&task_losses);
        let e = equal_loss.combine(&task_losses);
        assert!(
            (u - e).abs() > 1e-4,
            "uncertainty weighting should differ from equal when log_vars differ: u={}, e={}",
            u,
            e
        );
    }

    #[test]
    fn uncertainty_update_modifies_log_vars() {
        let init_log_vars = vec![0.0_f32, 0.0];
        let mut loss = MultiTaskLoss::new(2, TaskWeighting::Uncertainty(init_log_vars));
        let before = loss.log_vars.clone();
        loss.update_uncertainty_weights(&[2.0, 0.5], 0.01);
        let changed = loss
            .log_vars
            .iter()
            .zip(before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "update_uncertainty_weights should modify log_vars");
    }

    #[test]
    fn uncertainty_weights_vector_uses_precision() {
        let log_vars = vec![0.0_f32, 1.0]; // precision = exp(0)=1, exp(-1)≈0.368
        let loss = MultiTaskLoss::new(2, TaskWeighting::Uncertainty(log_vars));
        let w = loss.weights();
        assert_eq!(w.len(), 2);
        // w[0] = 0.5 * exp(0) = 0.5
        assert!(
            (w[0] - 0.5).abs() < 1e-5,
            "w[0] should be 0.5, got {}",
            w[0]
        );
    }

    // ── Dynamic weighting ─────────────────────────────────────────────────────

    #[test]
    fn dynamic_weighting_uses_supplied_weights() {
        let weights = vec![2.0_f32, 0.5];
        let mut loss = MultiTaskLoss::new(2, TaskWeighting::Dynamic(weights));
        let combined = loss.combine(&[1.0, 1.0]);
        // 2.0*1.0 + 0.5*1.0 = 2.5
        assert!(
            (combined - 2.5).abs() < 1e-5,
            "dynamic weighting should use supplied weights, got {}",
            combined
        );
    }

    // ── PCGrad ───────────────────────────────────────────────────────────────

    #[test]
    fn pcgrad_no_conflict_unchanged() {
        // Parallel gradients: no projection needed.
        let g1 = vec![1.0_f32, 0.0];
        let g2 = vec![1.0_f32, 0.0];
        let projected = pcgrad_project(&[g1.clone(), g2.clone()]);
        assert_eq!(projected.len(), 2);
        // g1 projected should be unchanged (dot > 0).
        let diff: f32 = projected[0]
            .iter()
            .zip(g1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-5, "parallel gradients should not be projected");
    }

    #[test]
    fn pcgrad_removes_conflicting_component() {
        // g1 = [1, 0] and g2 = [-1, 0] are fully opposed.
        let g1 = vec![1.0_f32, 0.0];
        let g2 = vec![-1.0_f32, 0.0];
        let projected = pcgrad_project(&[g1, g2]);

        // g1 projected onto plane normal to g2 should be zero (fully in conflict).
        let norm0: f32 = projected[0].iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            norm0 < 1e-5,
            "conflicting gradients should be projected to zero along conflict direction, norm={}",
            norm0
        );
    }

    #[test]
    fn pcgrad_empty_input() {
        let result = pcgrad_project(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn pcgrad_output_shape() {
        let grads: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let projected = pcgrad_project(&grads);
        assert_eq!(projected.len(), 3);
        for p in &projected {
            assert_eq!(p.len(), 2);
        }
    }

    // ── Cosine similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_similarity_parallel() {
        let g = vec![1.0_f32, 2.0, 3.0];
        let sim = gradient_cosine_similarity(&g, &g);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "parallel vectors should have cosine similarity 1, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_antiparallel() {
        let g1 = vec![1.0_f32, 0.0];
        let g2 = vec![-1.0_f32, 0.0];
        let sim = gradient_cosine_similarity(&g1, &g2);
        assert!(
            (sim + 1.0).abs() < 1e-5,
            "antiparallel vectors should have cosine similarity -1, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let g1 = vec![1.0_f32, 0.0];
        let g2 = vec![0.0_f32, 1.0];
        let sim = gradient_cosine_similarity(&g1, &g2);
        assert!(
            sim.abs() < 1e-5,
            "orthogonal vectors should have cosine similarity 0, got {}",
            sim
        );
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let g1 = vec![0.0_f32, 0.0];
        let g2 = vec![1.0_f32, 2.0];
        let sim = gradient_cosine_similarity(&g1, &g2);
        assert_eq!(sim, 0.0, "zero vector should yield similarity 0");
    }

    // ── TaskHead ──────────────────────────────────────────────────────────────

    #[test]
    fn task_head_forward_zero_weights_returns_bias() {
        let head = TaskHead::new("test", 4, 3);
        // All zero weights → output equals bias.
        let out = head.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert_eq!(v, 0.0, "zero-weight head should return all zeros");
        }
    }

    #[test]
    fn task_head_forward_identity_weight() {
        // 1x1 head with weight=2, bias=1 → output = 2*x + 1.
        let mut head = TaskHead::new("id", 1, 1);
        head.weights = vec![2.0];
        head.bias = vec![1.0];
        let out = head.forward(&[3.0]);
        assert!(
            (out[0] - 7.0).abs() < 1e-5,
            "expected 2*3+1=7, got {}",
            out[0]
        );
    }

    #[test]
    fn task_head_output_dimension_correct() {
        let head = TaskHead::new("task", 8, 5);
        let out = head.forward(&[1.0_f32; 8]);
        assert_eq!(out.len(), 5);
    }

    // ── MultiTaskModel ────────────────────────────────────────────────────────

    #[test]
    fn multi_task_model_forward_all_returns_all_tasks() {
        let mut model = MultiTaskModel::new(4);
        model.add_task(TaskHead::new("cls", 4, 2));
        model.add_task(TaskHead::new("reg", 4, 1));
        let repr = vec![1.0_f32, 0.0, 0.0, 0.0];
        let outputs = model.forward_all(&repr);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].0, "cls");
        assert_eq!(outputs[1].0, "reg");
        assert_eq!(outputs[0].1.len(), 2);
        assert_eq!(outputs[1].1.len(), 1);
    }

    #[test]
    fn multi_task_model_task_names() {
        let mut model = MultiTaskModel::new(4);
        model.add_task(TaskHead::new("a", 4, 1));
        model.add_task(TaskHead::new("b", 4, 1));
        let names = model.task_names();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn multi_task_model_total_loss_no_tasks_error() {
        let mut model = MultiTaskModel::new(4);
        let result = model.total_loss(&[0.0; 4], &[], |_, _| 0.0);
        assert!(matches!(result, Err(MultiTaskError::NoTasks)));
    }

    #[test]
    fn multi_task_model_total_loss_unknown_task_error() {
        let mut model = MultiTaskModel::new(4);
        model.add_task(TaskHead::new("cls", 4, 2));
        let result = model.total_loss(&[0.0; 4], &[("missing", &[1.0, 0.0])], |_, _| 0.0);
        assert!(matches!(result, Err(MultiTaskError::TaskNotFound { .. })));
    }

    #[test]
    fn multi_task_model_total_loss_computes() {
        let mut model = MultiTaskModel::new(2);
        model.add_task(TaskHead::new("t1", 2, 2));
        let repr = vec![0.0_f32; 2];
        let targets: &[(&str, &[f32])] = &[("t1", &[0.0, 0.0])];
        let loss_result = model.total_loss(&repr, targets, |pred, tgt| {
            pred.iter()
                .zip(tgt.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
        });
        assert!(
            loss_result.is_ok(),
            "total_loss should succeed for known task"
        );
    }

    // ── AuxiliaryTaskScheduler ────────────────────────────────────────────────

    #[test]
    fn aux_scheduler_initial_weights() {
        let sched = AuxiliaryTaskScheduler::new(1.0, vec![0.5, 0.3], 0.1);
        let w = sched.weights();
        assert_eq!(w.len(), 3);
        assert_eq!(w[0], 1.0);
        assert!((w[1] - 0.5).abs() < 1e-6);
        assert!((w[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn aux_scheduler_step_decays_aux() {
        let mut sched = AuxiliaryTaskScheduler::new(1.0, vec![1.0], 0.1);
        sched.step();
        let w = sched.weights();
        // After one step: 1.0 * (1 - 0.1) = 0.9
        assert!(
            (w[1] - 0.9).abs() < 1e-6,
            "aux weight should decay to 0.9, got {}",
            w[1]
        );
    }

    #[test]
    fn aux_scheduler_main_weight_unchanged() {
        let mut sched = AuxiliaryTaskScheduler::new(2.0, vec![1.0], 0.5);
        for _ in 0..10 {
            sched.step();
        }
        assert_eq!(
            sched.main_task_weight, 2.0,
            "main task weight should not decay"
        );
    }

    #[test]
    fn aux_scheduler_multiple_steps_geometric_decay() {
        let mut sched = AuxiliaryTaskScheduler::new(1.0, vec![1.0], 0.2);
        sched.step();
        sched.step();
        let w = sched.weights();
        // (1 - 0.2)^2 = 0.64
        assert!(
            (w[1] - 0.64).abs() < 1e-5,
            "two steps should give 0.64, got {}",
            w[1]
        );
    }

    #[test]
    fn aux_scheduler_no_aux_tasks() {
        let sched = AuxiliaryTaskScheduler::new(1.0, vec![], 0.1);
        let w = sched.weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0], 1.0);
    }

    #[test]
    fn multi_task_error_display() {
        let err = MultiTaskError::TaskNotFound {
            name: "seg".to_string(),
        };
        assert!(err.to_string().contains("seg"));

        let err2 = MultiTaskError::WeightMismatch {
            num_tasks: 3,
            num_weights: 2,
        };
        assert!(err2.to_string().contains("3") && err2.to_string().contains("2"));
    }
}
