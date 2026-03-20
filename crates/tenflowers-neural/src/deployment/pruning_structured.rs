//! Structured pruning extensions: StructuredPruner, PruningSchedule, PruningReport
//!
//! This module extends the core pruning infrastructure with higher-level structured
//! pruning primitives that operate on entire neurons/filters and support gradual
//! pruning schedules over training time.

use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// Report describing the outcome of a single structured pruning operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PruningReport {
    /// Name of the layer that was pruned.
    pub layer_name: String,
    /// Number of neurons/filters removed.
    pub pruned_count: usize,
    /// Total number of neurons/filters before pruning.
    pub total_count: usize,
    /// Fraction of units removed (pruned_count / total_count).
    pub sparsity_ratio: f32,
    /// Estimated FLOP reduction as a fraction in \[0, 1).
    ///
    /// Derived from the sparsity ratio: removing `k` out of `n` neurons eliminates
    /// `k/n` of the multiply-accumulate operations in both the current layer and
    /// the weight columns of the following layer, giving an effective reduction of
    /// roughly `sparsity_ratio` FLOPs.
    pub estimated_flop_reduction: f32,
    /// Layer-level importance scores for each unit (index → importance).
    pub unit_importances: Vec<f32>,
    /// Indices of the neurons/filters that were retained.
    pub retained_indices: Vec<usize>,
    /// Indices of the neurons/filters that were removed.
    pub pruned_indices: Vec<usize>,
}

impl PruningReport {
    /// Build a `PruningReport` given the full list of per-unit importance scores
    /// and the target sparsity.  Units are ranked ascending by importance; the
    /// least important ones are pruned first.
    pub fn from_importances(
        layer_name: &str,
        unit_importances: Vec<f32>,
        threshold: f32,
    ) -> Result<Self> {
        let total_count = unit_importances.len();
        if total_count == 0 {
            return Err(TensorError::invalid_argument(format!(
                "Layer '{}': unit importance list is empty",
                layer_name
            )));
        }
        if !(0.0..=1.0).contains(&threshold) {
            return Err(TensorError::invalid_argument(format!(
                "threshold must be in [0, 1]; got {}",
                threshold
            )));
        }

        // Rank units ascending by importance (least important first).
        let mut ranked: Vec<(usize, f32)> = unit_importances
            .iter()
            .copied()
            .enumerate()
            .collect();
        ranked.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let pruned_count = ((total_count as f32) * threshold).floor() as usize;

        let pruned_indices: Vec<usize> = ranked[..pruned_count]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        let pruned_set: std::collections::HashSet<usize> = pruned_indices.iter().copied().collect();
        let retained_indices: Vec<usize> = (0..total_count)
            .filter(|i| !pruned_set.contains(i))
            .collect();

        let sparsity_ratio = pruned_count as f32 / total_count as f32;
        // Removing sparsity_ratio of the neurons eliminates the same fraction of
        // multiply-accumulate operations in this layer.
        let estimated_flop_reduction = sparsity_ratio;

        Ok(Self {
            layer_name: layer_name.to_string(),
            pruned_count,
            total_count,
            sparsity_ratio,
            estimated_flop_reduction,
            unit_importances,
            retained_indices,
            pruned_indices,
        })
    }
}

/// Layer weight data supplied to `StructuredPruner` for analysis.
///
/// Contains the weight matrix dimensions and raw L1-importance scores per unit.
/// For a dense layer the "unit" dimension is `output_size`; for a conv layer it
/// is the filter (output channel) count.
#[derive(Debug, Clone)]
pub struct LayerWeightData {
    /// Flat list of per-unit L1-norm importance scores.
    pub unit_importances: Vec<f32>,
    /// Number of input features / in-channels.
    pub input_dim: usize,
    /// Number of output units / out-channels.
    pub output_dim: usize,
}

impl LayerWeightData {
    /// Construct from explicit importance scores.
    pub fn new(unit_importances: Vec<f32>, input_dim: usize, output_dim: usize) -> Self {
        Self {
            unit_importances,
            input_dim,
            output_dim,
        }
    }

    /// Construct synthetic data for testing: importances derived from a sine
    /// wave so values are predictable and deterministic.
    pub fn synthetic(input_dim: usize, output_dim: usize) -> Self {
        let unit_importances = (0..output_dim)
            .map(|i| {
                let mut norm = 0.0f32;
                for j in 0..input_dim {
                    let w = ((i * input_dim + j) as f32 * 0.001_f32).sin();
                    norm += w.abs();
                }
                norm
            })
            .collect();
        Self {
            unit_importances,
            input_dim,
            output_dim,
        }
    }
}

/// Structured pruner that removes entire neurons (dense) or filters (conv)
/// rather than individual weights.
///
/// Unlike magnitude-based unstructured pruning, structured pruning produces
/// architecturally smaller layers that benefit from dense linear-algebra
/// routines and do not require sparse-computation hardware support.
///
/// # Usage
///
/// ```rust,ignore
/// use tenflowers_neural::deployment::pruning_structured::{
///     LayerWeightData, StructuredPruner,
/// };
///
/// let data = LayerWeightData::synthetic(128, 64);
/// let pruner = StructuredPruner::new();
/// let report = pruner.prune_neurons("dense1", 0.25, &data).unwrap();
/// assert_eq!(report.pruned_count, 16); // 25% of 64
/// ```
pub struct StructuredPruner {
    /// Per-layer override thresholds. If absent the caller-supplied threshold is used.
    layer_thresholds: HashMap<String, f32>,
}

impl StructuredPruner {
    /// Create a structured pruner with no per-layer overrides.
    pub fn new() -> Self {
        Self {
            layer_thresholds: HashMap::new(),
        }
    }

    /// Register a per-layer threshold override.
    ///
    /// When `prune_neurons` is called for `layer_name`, this threshold is used
    /// instead of the one supplied by the caller.
    pub fn with_layer_threshold(mut self, layer_name: &str, threshold: f32) -> Self {
        self.layer_thresholds
            .insert(layer_name.to_string(), threshold);
        self
    }

    /// Prune entire neurons in a dense (or filters in a conv) layer.
    ///
    /// `threshold` is the fraction of units to remove, expressed as a value in
    /// `[0.0, 1.0)`.  Units are ranked by their L1-norm importance; the
    /// least-important `floor(total * threshold)` are removed.
    ///
    /// A per-layer threshold registered via `with_layer_threshold` takes precedence.
    pub fn prune_neurons(
        &self,
        layer_name: &str,
        threshold: f32,
        data: &LayerWeightData,
    ) -> Result<PruningReport> {
        let effective_threshold = self
            .layer_thresholds
            .get(layer_name)
            .copied()
            .unwrap_or(threshold);

        if effective_threshold < 0.0 || effective_threshold >= 1.0 {
            return Err(TensorError::invalid_argument(format!(
                "threshold must be in [0, 1); got {}",
                effective_threshold
            )));
        }

        PruningReport::from_importances(
            layer_name,
            data.unit_importances.clone(),
            effective_threshold,
        )
    }

    /// Prune all layers in a registry and collect per-layer reports.
    ///
    /// `layers` maps layer names to their weight data.  `default_threshold` is
    /// used for any layer without a per-layer override.
    pub fn prune_all(
        &self,
        layers: &HashMap<String, LayerWeightData>,
        default_threshold: f32,
    ) -> Result<Vec<PruningReport>> {
        let mut reports = Vec::with_capacity(layers.len());
        for (name, data) in layers {
            let report = self.prune_neurons(name, default_threshold, data)?;
            reports.push(report);
        }
        // Stable order for determinism in tests.
        reports.sort_by(|a, b| a.layer_name.cmp(&b.layer_name));
        Ok(reports)
    }
}

impl Default for StructuredPruner {
    fn default() -> Self {
        Self::new()
    }
}

/// Schedule that returns a target sparsity for each training step during gradual pruning.
///
/// The schedule implements a cubic warm-up adapted from the Zhu & Gupta 2017
/// "To Prune or Not to Prune" paper: sparsity increases smoothly from an initial
/// value to a final value over a given number of steps, held constant after the
/// final step.
///
/// Sparsity at step `t` (where `start_step ≤ t ≤ end_step`):
///
/// ```text
/// s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_f - t_0))^3
/// ```
///
/// where `s_i = initial_sparsity`, `s_f = final_sparsity`,
///       `t_0 = start_step`, `t_f = end_step`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PruningSchedule {
    /// Sparsity at the first pruning step (inclusive).
    pub initial_sparsity: f32,
    /// Sparsity at the last pruning step (inclusive).
    pub final_sparsity: f32,
    /// First step at which pruning begins.
    pub start_step: usize,
    /// Last step at which sparsity changes.
    pub end_step: usize,
    /// Pruning is applied every `frequency` steps.
    pub frequency: usize,
}

impl PruningSchedule {
    /// Create a new pruning schedule.
    ///
    /// # Errors
    ///
    /// Returns an error if `initial_sparsity > final_sparsity`, if either is
    /// outside `[0, 1]`, if `start_step >= end_step`, or if `frequency == 0`.
    pub fn new(
        initial_sparsity: f32,
        final_sparsity: f32,
        start_step: usize,
        end_step: usize,
        frequency: usize,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&initial_sparsity) {
            return Err(TensorError::invalid_argument(format!(
                "initial_sparsity must be in [0, 1]; got {}",
                initial_sparsity
            )));
        }
        if !(0.0..=1.0).contains(&final_sparsity) {
            return Err(TensorError::invalid_argument(format!(
                "final_sparsity must be in [0, 1]; got {}",
                final_sparsity
            )));
        }
        if initial_sparsity > final_sparsity {
            return Err(TensorError::invalid_argument(format!(
                "initial_sparsity ({}) must be ≤ final_sparsity ({})",
                initial_sparsity, final_sparsity
            )));
        }
        if start_step >= end_step {
            return Err(TensorError::invalid_argument(format!(
                "start_step ({}) must be < end_step ({})",
                start_step, end_step
            )));
        }
        if frequency == 0 {
            return Err(TensorError::invalid_argument(
                "frequency must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            initial_sparsity,
            final_sparsity,
            start_step,
            end_step,
            frequency,
        })
    }

    /// Return the target sparsity at `current_step`.
    ///
    /// * Before `start_step` → `initial_sparsity` (no pruning yet).
    /// * After `end_step` → `final_sparsity` (saturated).
    /// * Between `start_step` and `end_step` → cubic interpolation.
    /// * Only meaningful to apply pruning when `should_prune_at(current_step)` is true.
    pub fn step(&self, current_step: usize) -> f32 {
        if current_step < self.start_step {
            return self.initial_sparsity;
        }
        if current_step >= self.end_step {
            return self.final_sparsity;
        }
        let progress = (current_step - self.start_step) as f32
            / (self.end_step - self.start_step) as f32;
        // Cubic schedule: s_f + (s_i - s_f) * (1 - progress)^3
        let cubic_factor = (1.0 - progress).powi(3);
        self.final_sparsity
            + (self.initial_sparsity - self.final_sparsity) * cubic_factor
    }

    /// Return `true` when pruning masks should be updated at `current_step`.
    ///
    /// Pruning is applied at the `start_step` and then every `frequency` steps
    /// until the `end_step`.
    pub fn should_prune_at(&self, current_step: usize) -> bool {
        if current_step < self.start_step || current_step > self.end_step {
            return false;
        }
        (current_step - self.start_step) % self.frequency == 0
    }

    /// Return `true` once `current_step` has reached or passed `end_step`.
    pub fn is_final_step(&self, current_step: usize) -> bool {
        current_step >= self.end_step
    }

    /// Total number of pruning update events from `start_step` to `end_step`.
    pub fn total_pruning_steps(&self) -> usize {
        let span = self.end_step - self.start_step;
        span / self.frequency + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PruningReport ────────────────────────────────────────────────────────

    #[test]
    fn test_pruning_report_basic() {
        // Ten units with importances 0.0, 0.1, ..., 0.9
        let importances: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let report =
            PruningReport::from_importances("dense1", importances, 0.3).expect("should succeed");

        assert_eq!(report.layer_name, "dense1");
        assert_eq!(report.total_count, 10);
        assert_eq!(report.pruned_count, 3); // floor(10 * 0.3)
        assert_eq!(report.retained_indices.len(), 7);
        assert_eq!(report.pruned_indices.len(), 3);
        // Least-important are indices 0, 1, 2 (importances 0.0, 0.1, 0.2)
        let mut pruned = report.pruned_indices.clone();
        pruned.sort();
        assert_eq!(pruned, vec![0, 1, 2]);
    }

    #[test]
    fn test_pruning_report_zero_threshold() {
        let importances = vec![0.5f32, 0.3, 0.8, 0.1];
        let report =
            PruningReport::from_importances("layer", importances, 0.0).expect("should succeed");
        assert_eq!(report.pruned_count, 0);
        assert_eq!(report.retained_indices.len(), 4);
        assert_eq!(report.sparsity_ratio, 0.0);
        assert_eq!(report.estimated_flop_reduction, 0.0);
    }

    #[test]
    fn test_pruning_report_empty_importances_returns_error() {
        let result = PruningReport::from_importances("empty_layer", vec![], 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_pruning_report_invalid_threshold_returns_error() {
        let result = PruningReport::from_importances("layer", vec![0.1, 0.2], 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_pruning_report_flop_reduction_matches_sparsity() {
        let importances: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let report =
            PruningReport::from_importances("conv1", importances, 0.4).expect("should succeed");
        assert!(
            (report.estimated_flop_reduction - report.sparsity_ratio).abs() < 1e-6,
            "estimated_flop_reduction should equal sparsity_ratio"
        );
    }

    // ─── StructuredPruner ─────────────────────────────────────────────────────

    #[test]
    fn test_structured_pruner_prune_neurons() {
        let data = LayerWeightData::synthetic(32, 16);
        let pruner = StructuredPruner::new();
        let report = pruner.prune_neurons("fc1", 0.25, &data).expect("should succeed");

        assert_eq!(report.layer_name, "fc1");
        assert_eq!(report.total_count, 16);
        assert_eq!(report.pruned_count, 4); // floor(16 * 0.25)
    }

    #[test]
    fn test_structured_pruner_layer_override() {
        let data = LayerWeightData::synthetic(64, 32);
        let pruner = StructuredPruner::new().with_layer_threshold("special", 0.5);

        // "special" uses the registered threshold (0.5), not the caller's (0.1)
        let report = pruner
            .prune_neurons("special", 0.1, &data)
            .expect("should succeed");
        assert_eq!(report.pruned_count, 16); // floor(32 * 0.5)
    }

    #[test]
    fn test_structured_pruner_prune_all() {
        let mut layers: HashMap<String, LayerWeightData> = HashMap::new();
        layers.insert("fc1".to_string(), LayerWeightData::synthetic(128, 64));
        layers.insert("fc2".to_string(), LayerWeightData::synthetic(64, 32));

        let pruner = StructuredPruner::new();
        let reports = pruner.prune_all(&layers, 0.25).expect("should succeed");

        assert_eq!(reports.len(), 2);
        // Check both fc1 and fc2 are present (sorted alphabetically)
        let names: Vec<&str> = reports.iter().map(|r| r.layer_name.as_str()).collect();
        assert!(names.contains(&"fc1"));
        assert!(names.contains(&"fc2"));
    }

    #[test]
    fn test_structured_pruner_invalid_threshold() {
        let data = LayerWeightData::synthetic(10, 10);
        let pruner = StructuredPruner::new();
        let result = pruner.prune_neurons("layer", 1.0, &data); // threshold=1.0 is invalid
        assert!(result.is_err());
    }

    // ─── PruningSchedule ──────────────────────────────────────────────────────

    #[test]
    fn test_pruning_schedule_before_start() {
        let schedule = PruningSchedule::new(0.0, 0.8, 100, 1000, 10).expect("should succeed");
        // Step 0 is before start_step=100: should return initial_sparsity
        let sparsity = schedule.step(0);
        assert!((sparsity - 0.0).abs() < 1e-6);
        assert!(!schedule.should_prune_at(0));
    }

    #[test]
    fn test_pruning_schedule_after_end() {
        let schedule = PruningSchedule::new(0.0, 0.8, 100, 1000, 10).expect("should succeed");
        let sparsity = schedule.step(2000);
        assert!((sparsity - 0.8).abs() < 1e-6);
        assert!(schedule.is_final_step(2000));
    }

    #[test]
    fn test_pruning_schedule_at_midpoint() {
        let schedule = PruningSchedule::new(0.0, 0.8, 0, 100, 1).expect("should succeed");
        // At exactly step=50, progress=0.5, cubic_factor=(0.5)^3=0.125
        // s(50) = 0.8 + (0.0 - 0.8) * 0.125 = 0.8 - 0.1 = 0.7
        let sparsity = schedule.step(50);
        let expected = 0.8 + (0.0 - 0.8) * 0.5f32.powi(3);
        assert!(
            (sparsity - expected).abs() < 1e-5,
            "expected {expected}, got {sparsity}"
        );
    }

    #[test]
    fn test_pruning_schedule_should_prune_frequency() {
        let schedule = PruningSchedule::new(0.0, 0.5, 0, 100, 10).expect("should succeed");
        assert!(schedule.should_prune_at(0));
        assert!(schedule.should_prune_at(10));
        assert!(schedule.should_prune_at(20));
        assert!(!schedule.should_prune_at(5));
        assert!(!schedule.should_prune_at(15));
    }

    #[test]
    fn test_pruning_schedule_is_final_step() {
        let schedule = PruningSchedule::new(0.1, 0.9, 0, 50, 5).expect("should succeed");
        assert!(!schedule.is_final_step(49));
        assert!(schedule.is_final_step(50));
        assert!(schedule.is_final_step(100));
    }

    #[test]
    fn test_pruning_schedule_total_steps() {
        // span=100, frequency=10 → 11 events (0,10,20,...,100)
        let schedule = PruningSchedule::new(0.0, 0.8, 0, 100, 10).expect("should succeed");
        assert_eq!(schedule.total_pruning_steps(), 11);
    }

    #[test]
    fn test_pruning_schedule_invalid_parameters() {
        // initial > final
        assert!(PruningSchedule::new(0.9, 0.1, 0, 100, 1).is_err());
        // out of range
        assert!(PruningSchedule::new(-0.1, 0.5, 0, 100, 1).is_err());
        assert!(PruningSchedule::new(0.0, 1.1, 0, 100, 1).is_err());
        // start >= end
        assert!(PruningSchedule::new(0.0, 0.5, 100, 100, 1).is_err());
        // zero frequency
        assert!(PruningSchedule::new(0.0, 0.5, 0, 100, 0).is_err());
    }
}
