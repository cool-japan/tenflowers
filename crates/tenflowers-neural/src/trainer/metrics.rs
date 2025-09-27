//! Training metrics and state management
//!
//! This module provides data structures and functionality for tracking
//! training progress, metrics, and state throughout the training process.

use std::collections::HashMap;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Training metrics and history
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub metrics: HashMap<String, f32>,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(epoch: usize, step: usize, loss: f32) -> Self {
        Self {
            epoch,
            step,
            loss,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the current metrics
    pub fn add_metric(&mut self, name: String, value: f32) {
        self.metrics.insert(name, value);
    }

    /// Get a metric value by name
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).copied()
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<&String> {
        self.metrics.keys().collect()
    }
}

/// Training state and configuration
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub best_metric: Option<f32>,
    pub history: Vec<TrainingMetrics>,
    pub val_history: Vec<TrainingMetrics>,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_metric: None,
            history: Vec::new(),
            val_history: Vec::new(),
        }
    }

    /// Add training metrics to history
    pub fn add_training_metrics(&mut self, metrics: TrainingMetrics) {
        self.history.push(metrics);
    }

    /// Add validation metrics to history
    pub fn add_validation_metrics(&mut self, metrics: TrainingMetrics) {
        self.val_history.push(metrics);
    }

    /// Update best metric if the current one is better
    pub fn update_best_metric(&mut self, metric: f32, higher_is_better: bool) -> bool {
        match self.best_metric {
            None => {
                self.best_metric = Some(metric);
                true
            }
            Some(best) => {
                let is_better = if higher_is_better {
                    metric > best
                } else {
                    metric < best
                };

                if is_better {
                    self.best_metric = Some(metric);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get the latest training loss
    pub fn latest_train_loss(&self) -> Option<f32> {
        self.history.last().map(|m| m.loss)
    }

    /// Get the latest validation loss
    pub fn latest_val_loss(&self) -> Option<f32> {
        self.val_history.last().map(|m| m.loss)
    }

    /// Get training loss history
    pub fn train_loss_history(&self) -> Vec<f32> {
        self.history.iter().map(|m| m.loss).collect()
    }

    /// Get validation loss history
    pub fn val_loss_history(&self) -> Vec<f32> {
        self.val_history.iter().map(|m| m.loss).collect()
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.epoch = 0;
        self.step = 0;
        self.best_metric = None;
        self.history.clear();
        self.val_history.clear();
    }

    /// Increment epoch
    pub fn next_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Increment step
    pub fn next_step(&mut self) {
        self.step += 1;
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Callback action returned by callbacks to control training behavior
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackAction {
    Continue,                // Continue training normally
    Stop,                    // Stop training
    ReduceLearningRate(f32), // Reduce learning rate by the given factor
    SaveModel(String),       // Save model to the given filepath
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_creation() {
        let mut metrics = TrainingMetrics::new(1, 100, 0.5);
        assert_eq!(metrics.epoch, 1);
        assert_eq!(metrics.step, 100);
        assert_eq!(metrics.loss, 0.5);
        assert!(metrics.metrics.is_empty());

        metrics.add_metric("accuracy".to_string(), 0.95);
        assert_eq!(metrics.get_metric("accuracy"), Some(0.95));
    }

    #[test]
    fn test_training_state_best_metric_update() {
        let mut state = TrainingState::new();

        // First metric is always best
        assert!(state.update_best_metric(0.8, true)); // higher is better
        assert_eq!(state.best_metric, Some(0.8));

        // Better metric
        assert!(state.update_best_metric(0.9, true));
        assert_eq!(state.best_metric, Some(0.9));

        // Worse metric
        assert!(!state.update_best_metric(0.7, true));
        assert_eq!(state.best_metric, Some(0.9));
    }

    #[test]
    fn test_training_state_history() {
        let mut state = TrainingState::new();

        let train_metrics = TrainingMetrics::new(1, 100, 0.5);
        state.add_training_metrics(train_metrics);

        let val_metrics = TrainingMetrics::new(1, 100, 0.4);
        state.add_validation_metrics(val_metrics);

        assert_eq!(state.latest_train_loss(), Some(0.5));
        assert_eq!(state.latest_val_loss(), Some(0.4));
    }
}
