//! Early stopping callback implementation
//!
//! This module provides early stopping functionality to prevent overfitting
//! by monitoring a metric and stopping training when it stops improving.

use crate::{
    optimizers::Optimizer,
    trainer::metrics::{CallbackAction, TrainingState},
    Model,
};
use tenflowers_core::Result;

use super::Callback;

/// Early stopping callback
///
/// Monitors a specified metric and stops training when it stops improving
/// for a specified number of epochs (patience).
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of epochs with no improvement after which training will be stopped
    pub patience: usize,
    /// Minimum change in the monitored quantity to qualify as an improvement
    pub min_delta: f32,
    /// Name of the metric to monitor
    pub monitor: String,
    /// Whether to maximize ("max") or minimize ("min") the monitored metric
    pub mode: String,
    /// Current number of epochs with no improvement
    wait: usize,
    /// Best score seen so far
    best_score: Option<f32>,
}

impl EarlyStopping {
    /// Create a new early stopping callback
    ///
    /// # Arguments
    /// * `patience` - Number of epochs with no improvement after which training will be stopped
    /// * `min_delta` - Minimum change in the monitored quantity to qualify as an improvement
    /// * `monitor` - Name of the metric to monitor (e.g., "val_loss", "val_accuracy")
    /// * `mode` - "min" for metrics that should decrease (like loss) or "max" for metrics that should increase (like accuracy)
    pub fn new(patience: usize, min_delta: f32, monitor: String, mode: String) -> Self {
        assert!(
            mode == "min" || mode == "max",
            "Mode must be either 'min' or 'max'"
        );
        assert!(patience > 0, "Patience must be greater than 0");

        Self {
            patience,
            min_delta,
            monitor,
            mode,
            wait: 0,
            best_score: None,
        }
    }

    /// Create early stopping for minimizing a metric (like loss)
    pub fn for_minimizing(patience: usize, min_delta: f32, monitor: String) -> Self {
        Self::new(patience, min_delta, monitor, "min".to_string())
    }

    /// Create early stopping for maximizing a metric (like accuracy)
    pub fn for_maximizing(patience: usize, min_delta: f32, monitor: String) -> Self {
        Self::new(patience, min_delta, monitor, "max".to_string())
    }

    /// Get the current wait count
    pub fn current_wait(&self) -> usize {
        self.wait
    }

    /// Get the best score seen so far
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Reset the early stopping state
    pub fn reset(&mut self) {
        self.wait = 0;
        self.best_score = None;
    }

    /// Check if the current score represents an improvement
    fn is_improvement(&self, current_score: f32, best_score: f32) -> bool {
        if self.mode == "min" {
            current_score < best_score - self.min_delta
        } else {
            current_score > best_score + self.min_delta
        }
    }
}

impl<T> Callback<T> for EarlyStopping
where
    T: Clone + Default,
{
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        state: &TrainingState,
        _model: &dyn Model<T>,
        _optimizer: &mut dyn Optimizer<T>,
    ) -> Result<CallbackAction> {
        // Get the latest validation metrics
        if let Some(latest_val) = state.val_history.last() {
            if let Some(current_score) = latest_val.metrics.get(&self.monitor) {
                let should_stop = match self.best_score {
                    None => {
                        // First epoch - set the initial best score
                        self.best_score = Some(*current_score);
                        false
                    }
                    Some(best) => {
                        if self.is_improvement(*current_score, best) {
                            // Improvement found - update best score and reset wait
                            self.best_score = Some(*current_score);
                            self.wait = 0;
                            println!(
                                "Early stopping: new best {} = {:.6}",
                                self.monitor, current_score
                            );
                            false
                        } else {
                            // No improvement - increment wait
                            self.wait += 1;
                            println!(
                                "Early stopping: no improvement for {}/{} epochs (current {} = {:.6}, best = {:.6})",
                                self.wait, self.patience, self.monitor, current_score, best
                            );
                            self.wait >= self.patience
                        }
                    }
                };

                if should_stop {
                    println!(
                        "Early stopping: no improvement for {} epochs. Training stopped.",
                        self.patience
                    );
                    Ok(CallbackAction::Stop)
                } else {
                    Ok(CallbackAction::Continue)
                }
            } else {
                println!(
                    "Early stopping: metric '{}' not found in validation metrics. Available metrics: {:?}",
                    self.monitor,
                    latest_val.metrics.keys().collect::<Vec<_>>()
                );
                Ok(CallbackAction::Continue) // Continue if metric not found
            }
        } else {
            // No validation history yet - continue training
            Ok(CallbackAction::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::metrics::TrainingMetrics;
    use std::collections::HashMap;

    #[test]
    fn test_early_stopping_creation() {
        let es = EarlyStopping::new(5, 0.001, "val_loss".to_string(), "min".to_string());
        assert_eq!(es.patience, 5);
        assert_eq!(es.min_delta, 0.001);
        assert_eq!(es.monitor, "val_loss");
        assert_eq!(es.mode, "min");
        assert_eq!(es.current_wait(), 0);
        assert_eq!(es.best_score(), None);
    }

    #[test]
    fn test_early_stopping_for_minimizing() {
        let es = EarlyStopping::for_minimizing(3, 0.01, "val_loss".to_string());
        assert_eq!(es.mode, "min");
    }

    #[test]
    fn test_early_stopping_for_maximizing() {
        let es = EarlyStopping::for_maximizing(3, 0.01, "val_accuracy".to_string());
        assert_eq!(es.mode, "max");
    }

    #[test]
    fn test_is_improvement_min_mode() {
        let es = EarlyStopping::for_minimizing(3, 0.01, "val_loss".to_string());

        // Lower score should be improvement for min mode
        assert!(es.is_improvement(0.5, 0.6));
        assert!(!es.is_improvement(0.6, 0.5));

        // Should respect min_delta
        assert!(!es.is_improvement(0.595, 0.6)); // improvement too small
        assert!(es.is_improvement(0.58, 0.6)); // improvement large enough
    }

    #[test]
    fn test_is_improvement_max_mode() {
        let es = EarlyStopping::for_maximizing(3, 0.01, "val_accuracy".to_string());

        // Higher score should be improvement for max mode
        assert!(es.is_improvement(0.9, 0.8));
        assert!(!es.is_improvement(0.8, 0.9));

        // Should respect min_delta
        assert!(!es.is_improvement(0.805, 0.8)); // improvement too small
        assert!(es.is_improvement(0.82, 0.8)); // improvement large enough
    }

    #[test]
    #[should_panic]
    fn test_invalid_mode() {
        EarlyStopping::new(5, 0.001, "val_loss".to_string(), "invalid".to_string());
    }

    #[test]
    #[should_panic]
    fn test_zero_patience() {
        EarlyStopping::new(0, 0.001, "val_loss".to_string(), "min".to_string());
    }
}
