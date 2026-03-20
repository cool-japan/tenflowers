//! Trial management for hyperparameter studies.
//!
//! Tracks trial results for a named study and provides methods to retrieve the
//! best trial, best metric, and a human-readable summary.

use std::collections::HashMap;

use crate::hparam::space::HParamSet;

// ─────────────────────────────────────────────────────────────────────────────
// OptimizationDirection
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the objective metric should be minimized or maximized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationDirection {
    /// Lower metric values are better (e.g. loss).
    Minimize,
    /// Higher metric values are better (e.g. accuracy).
    Maximize,
}

// ─────────────────────────────────────────────────────────────────────────────
// TrialResult
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of a single hyperparameter trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Zero-based trial index (assigned externally by the caller).
    pub trial_id: usize,
    /// The hyperparameter values used in this trial.
    pub params: HParamSet,
    /// The objective metric value for this trial.
    pub metric: f64,
    /// Wall-clock duration of this trial in seconds.
    pub duration_secs: f64,
    /// Arbitrary key-value metadata (e.g. checkpoint path, epoch count).
    pub metadata: HashMap<String, String>,
}

impl TrialResult {
    /// Convenience constructor with empty metadata.
    pub fn new(trial_id: usize, params: HParamSet, metric: f64, duration_secs: f64) -> Self {
        TrialResult {
            trial_id,
            params,
            metric,
            duration_secs,
            metadata: HashMap::new(),
        }
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HParamStudy
// ─────────────────────────────────────────────────────────────────────────────

/// A study that accumulates trial results and tracks the best configuration.
pub struct HParamStudy {
    /// Human-readable name for this study.
    pub name: String,
    /// Whether to minimize or maximize the objective metric.
    pub direction: OptimizationDirection,
    trials: Vec<TrialResult>,
}

impl HParamStudy {
    /// Create a new, empty study.
    pub fn new(name: &str, direction: OptimizationDirection) -> Self {
        HParamStudy {
            name: name.to_owned(),
            direction,
            trials: Vec::new(),
        }
    }

    /// Add a completed trial result.
    pub fn add_trial(&mut self, result: TrialResult) {
        self.trials.push(result);
    }

    /// Return the trial with the best metric according to the study direction.
    ///
    /// Returns `None` if no trials have been added.
    pub fn best_trial(&self) -> Option<&TrialResult> {
        match self.direction {
            OptimizationDirection::Minimize => self
                .trials
                .iter()
                .filter(|t| t.metric.is_finite())
                .min_by(|a, b| {
                    a.metric
                        .partial_cmp(&b.metric)
                        .expect("finite metric comparison")
                }),
            OptimizationDirection::Maximize => self
                .trials
                .iter()
                .filter(|t| t.metric.is_finite())
                .max_by(|a, b| {
                    a.metric
                        .partial_cmp(&b.metric)
                        .expect("finite metric comparison")
                }),
        }
    }

    /// Return the hyperparameter set of the best trial.
    pub fn best_params(&self) -> Option<&HParamSet> {
        self.best_trial().map(|t| &t.params)
    }

    /// Return the metric value of the best trial.
    pub fn best_metric(&self) -> Option<f64> {
        self.best_trial().map(|t| t.metric)
    }

    /// Return the total number of trials added to this study.
    pub fn num_trials(&self) -> usize {
        self.trials.len()
    }

    /// Return all metric values in insertion order.
    pub fn all_metrics(&self) -> Vec<f64> {
        self.trials.iter().map(|t| t.metric).collect()
    }

    /// Return a human-readable summary of the study.
    pub fn summary(&self) -> String {
        let direction_str = match self.direction {
            OptimizationDirection::Minimize => "Minimize",
            OptimizationDirection::Maximize => "Maximize",
        };

        let best_str = match self.best_trial() {
            Some(t) => format!(
                "trial_id={}, metric={:.6}, duration={:.3}s",
                t.trial_id, t.metric, t.duration_secs
            ),
            None => "n/a".to_owned(),
        };

        let total_duration: f64 = self.trials.iter().map(|t| t.duration_secs).sum();

        format!(
            "Study '{}': direction={}, trials={}, best=[{}], total_duration={:.3}s",
            self.name,
            direction_str,
            self.trials.len(),
            best_str,
            total_duration,
        )
    }

    /// Return an immutable slice of all trials.
    pub fn trials(&self) -> &[TrialResult] {
        &self.trials
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hparam::space::{HParamSet, HParamValue};

    fn make_trial(id: usize, metric: f64) -> TrialResult {
        let mut params = HParamSet::new();
        params.set("lr", HParamValue::Float(0.001 * id as f64 + 0.0001));
        TrialResult::new(id, params, metric, 0.5 + id as f64 * 0.1)
    }

    #[test]
    fn test_study_best_trial_minimize() {
        let mut study = HParamStudy::new("loss-study", OptimizationDirection::Minimize);
        study.add_trial(make_trial(0, 0.5));
        study.add_trial(make_trial(1, 0.3));
        study.add_trial(make_trial(2, 0.4));

        let best = study.best_trial().expect("should have a best trial");
        assert_eq!(best.trial_id, 1);
        assert!((best.metric - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_study_best_trial_maximize() {
        let mut study = HParamStudy::new("accuracy-study", OptimizationDirection::Maximize);
        study.add_trial(make_trial(0, 0.80));
        study.add_trial(make_trial(1, 0.95));
        study.add_trial(make_trial(2, 0.88));

        let best = study.best_trial().expect("should have a best trial");
        assert_eq!(best.trial_id, 1);
        assert!((best.metric - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_study_best_trial_empty_returns_none() {
        let study = HParamStudy::new("empty", OptimizationDirection::Minimize);
        assert!(study.best_trial().is_none());
        assert!(study.best_params().is_none());
        assert!(study.best_metric().is_none());
    }

    #[test]
    fn test_study_num_trials() {
        let mut study = HParamStudy::new("count-study", OptimizationDirection::Minimize);
        assert_eq!(study.num_trials(), 0);
        study.add_trial(make_trial(0, 1.0));
        study.add_trial(make_trial(1, 0.5));
        assert_eq!(study.num_trials(), 2);
    }

    #[test]
    fn test_study_all_metrics_order() {
        let mut study = HParamStudy::new("metrics-study", OptimizationDirection::Minimize);
        study.add_trial(make_trial(0, 1.0));
        study.add_trial(make_trial(1, 0.5));
        study.add_trial(make_trial(2, 0.8));
        let metrics = study.all_metrics();
        assert_eq!(metrics, vec![1.0, 0.5, 0.8]);
    }

    #[test]
    fn test_study_summary_contains_expected_fields() {
        let mut study = HParamStudy::new("my-study", OptimizationDirection::Maximize);
        study.add_trial(make_trial(0, 0.90));
        study.add_trial(make_trial(1, 0.95));
        let summary = study.summary();
        assert!(
            summary.contains("my-study"),
            "summary must contain study name"
        );
        assert!(
            summary.contains("Maximize"),
            "summary must contain direction"
        );
        assert!(
            summary.contains("trials=2"),
            "summary must contain trial count"
        );
        assert!(summary.contains("0.95"), "summary must contain best metric");
    }

    #[test]
    fn test_trial_result_metadata() {
        let params = HParamSet::new();
        let trial = TrialResult::new(0, params, 0.1, 2.5)
            .with_metadata("checkpoint", "/tmp/ckpt.bin")
            .with_metadata("epoch", "10");
        assert_eq!(
            trial.metadata.get("checkpoint").map(|s| s.as_str()),
            Some("/tmp/ckpt.bin")
        );
        assert_eq!(trial.metadata.get("epoch").map(|s| s.as_str()), Some("10"));
    }

    #[test]
    fn test_study_best_metric_minimize() {
        let mut study = HParamStudy::new("s", OptimizationDirection::Minimize);
        study.add_trial(make_trial(0, 2.0));
        study.add_trial(make_trial(1, 0.1));
        assert!((study.best_metric().expect("best_metric should exist") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_study_best_metric_maximize() {
        let mut study = HParamStudy::new("s", OptimizationDirection::Maximize);
        study.add_trial(make_trial(0, 0.7));
        study.add_trial(make_trial(1, 0.99));
        assert!((study.best_metric().expect("best_metric should exist") - 0.99).abs() < 1e-9);
    }

    #[test]
    fn test_study_best_params_points_to_correct_trial() {
        let mut study = HParamStudy::new("s", OptimizationDirection::Minimize);
        let mut p0 = HParamSet::new();
        p0.set("lr", HParamValue::Float(0.01));
        let mut p1 = HParamSet::new();
        p1.set("lr", HParamValue::Float(0.001));
        study.add_trial(TrialResult::new(0, p0, 0.5, 1.0));
        study.add_trial(TrialResult::new(1, p1, 0.2, 1.0));
        let best_lr = study
            .best_params()
            .and_then(|p| p.get_float("lr"))
            .expect("must have lr");
        assert!((best_lr - 0.001).abs() < 1e-9);
    }
}
