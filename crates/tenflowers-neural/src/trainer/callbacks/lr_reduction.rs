//! Learning rate reduction callback implementation
//!
//! This module provides learning rate reduction functionality to adjust
//! the learning rate when training plateaus, helping to fine-tune training.

use crate::{
    optimizers::Optimizer,
    trainer::metrics::{CallbackAction, TrainingState},
    Model,
};
use tenflowers_core::Result;

use super::Callback;

/// Learning rate reduction callback
///
/// Reduces learning rate when a metric has stopped improving.
/// This callback monitors a quantity and if no improvement is seen
/// for a 'patience' number of epochs, the learning rate is reduced.
#[derive(Debug, Clone)]
pub struct LearningRateReduction<T> {
    /// Metric to monitor (e.g., "val_loss", "val_accuracy")
    pub monitor: String,
    /// Factor by which the learning rate will be reduced (new_lr = lr * factor)
    pub factor: f32,
    /// Number of epochs with no improvement to wait before reducing LR
    pub patience: usize,
    /// Threshold for measuring the new optimum, to focus on significant changes
    pub min_delta: f32,
    /// Number of epochs to wait before resuming operation after lr reduction
    pub cooldown: usize,
    /// Lower bound on the learning rate
    pub min_lr: f32,
    /// "min", "max". In "min" mode, lr will be reduced when monitored quantity stops decreasing
    pub mode: String,
    /// Whether to print messages when reducing learning rate
    pub verbose: bool,
    /// Number of epochs with no improvement
    wait: usize,
    /// Epochs since last reduction
    cooldown_counter: usize,
    /// Best score seen so far
    best_score: Option<f32>,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LearningRateReduction<T> {
    /// Create a new LearningRateReduction callback
    ///
    /// # Arguments
    /// * `monitor` - Quantity to be monitored
    /// * `factor` - Factor by which the learning rate will be reduced (new_lr = lr * factor)
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `min_delta` - Threshold for measuring the new optimum, to only focus on significant changes
    /// * `cooldown` - Number of epochs to wait before resuming normal operation after lr has been reduced
    /// * `min_lr` - Lower bound on the learning rate
    /// * `mode` - "min", "max". In "min" mode, lr will be reduced when the monitored quantity has stopped decreasing
    /// * `verbose` - Whether to print messages when reducing learning rate
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        monitor: String,
        factor: f32,
        patience: usize,
        min_delta: f32,
        cooldown: usize,
        min_lr: f32,
        mode: String,
        verbose: bool,
    ) -> Self {
        assert!(
            mode == "min" || mode == "max",
            "Mode must be either 'min' or 'max'"
        );
        assert!(
            factor > 0.0 && factor < 1.0,
            "Factor must be between 0 and 1"
        );
        assert!(patience > 0, "Patience must be greater than 0");
        assert!(min_lr >= 0.0, "Minimum learning rate must be non-negative");

        Self {
            monitor,
            factor,
            patience,
            min_delta,
            cooldown,
            min_lr,
            mode,
            verbose,
            wait: 0,
            cooldown_counter: 0,
            best_score: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with default parameters
    ///
    /// Default parameters:
    /// - factor: 0.1 (reduce learning rate by 10x)
    /// - patience: 10 epochs
    /// - min_delta: 1e-4
    /// - cooldown: 0 epochs
    /// - min_lr: 0.0
    /// - mode: inferred from monitor name ("max" if contains "acc", "min" otherwise)
    /// - verbose: false
    pub fn with_defaults(monitor: String) -> Self {
        let mode = if monitor.contains("acc") {
            "max".to_string()
        } else {
            "min".to_string()
        };
        Self::new(monitor, 0.1, 10, 1e-4, 0, 0.0, mode, false)
    }

    /// Create for minimizing a metric (like loss)
    pub fn for_minimizing(monitor: String, factor: f32, patience: usize) -> Self {
        Self::new(
            monitor,
            factor,
            patience,
            1e-4,
            0,
            0.0,
            "min".to_string(),
            true,
        )
    }

    /// Create for maximizing a metric (like accuracy)
    pub fn for_maximizing(monitor: String, factor: f32, patience: usize) -> Self {
        Self::new(
            monitor,
            factor,
            patience,
            1e-4,
            0,
            0.0,
            "max".to_string(),
            true,
        )
    }

    /// Builder method to set factor
    pub fn factor(mut self, factor: f32) -> Self {
        assert!(
            factor > 0.0 && factor < 1.0,
            "Factor must be between 0 and 1"
        );
        self.factor = factor;
        self
    }

    /// Builder method to set patience
    pub fn patience(mut self, patience: usize) -> Self {
        assert!(patience > 0, "Patience must be greater than 0");
        self.patience = patience;
        self
    }

    /// Builder method to set verbose
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Builder method to set minimum learning rate
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        assert!(min_lr >= 0.0, "Minimum learning rate must be non-negative");
        self.min_lr = min_lr;
        self
    }

    /// Builder method to set cooldown
    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Builder method to set min_delta
    pub fn min_delta(mut self, min_delta: f32) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Get the current wait count
    pub fn current_wait(&self) -> usize {
        self.wait
    }

    /// Get the current cooldown counter
    pub fn current_cooldown(&self) -> usize {
        self.cooldown_counter
    }

    /// Get the best score seen so far
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Reset the callback state
    pub fn reset(&mut self) {
        self.wait = 0;
        self.cooldown_counter = 0;
        self.best_score = None;
    }

    /// Check if the current score is better than the best score
    fn is_better(&self, current: f32, best: f32) -> bool {
        if self.mode == "min" {
            current < best - self.min_delta
        } else {
            current > best + self.min_delta
        }
    }

    /// Get current score from state, trying validation first then training
    fn get_current_score(&self, state: &TrainingState) -> Option<f32> {
        // Try validation history first
        if !state.val_history.is_empty() {
            if let Some(current_score) = state
                .val_history
                .last()
                .and_then(|metrics| metrics.metrics.get(&self.monitor).copied())
            {
                return Some(current_score);
            }

            // Fall back to val_loss if monitor not found and monitor is not val_loss
            if self.monitor != "val_loss" {
                if let Some(val_loss) = state.val_history.last().map(|m| m.loss) {
                    return Some(val_loss);
                }
            }
        }

        // Use training history if no validation history
        if let Some(current_score) = state
            .history
            .last()
            .and_then(|metrics| metrics.metrics.get(&self.monitor).copied())
        {
            return Some(current_score);
        }

        // Fall back to training loss if monitor not found and monitor is not loss
        if self.monitor != "loss" {
            state.history.last().map(|m| m.loss)
        } else {
            None
        }
    }
}

impl<T> Callback<T> for LearningRateReduction<T>
where
    T: Clone + Default + std::fmt::Debug,
{
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        state: &TrainingState,
        _model: &dyn Model<T>,
        _optimizer: &mut dyn Optimizer<T>,
    ) -> Result<CallbackAction> {
        // Skip if we're in cooldown period
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            if self.verbose {
                println!(
                    "LR Reduction: cooldown period, {} epochs remaining",
                    self.cooldown_counter
                );
            }
            return Ok(CallbackAction::Continue);
        }

        // Get current score from validation or training history
        if let Some(current) = self.get_current_score(state) {
            match self.best_score {
                None => {
                    // First epoch - set initial best score
                    self.best_score = Some(current);
                    self.wait = 0;
                    if self.verbose {
                        println!(
                            "LR Reduction: setting initial best {} = {:.6}",
                            self.monitor, current
                        );
                    }
                }
                Some(best) => {
                    if self.is_better(current, best) {
                        // Improvement found - update best score and reset wait
                        self.best_score = Some(current);
                        self.wait = 0;
                        if self.verbose {
                            println!(
                                "LR Reduction: new best {} = {:.6} (previous: {:.6})",
                                self.monitor, current, best
                            );
                        }
                    } else {
                        // No improvement - increment wait
                        self.wait += 1;
                        if self.verbose {
                            println!(
                                "LR Reduction: no improvement for {}/{} epochs ({} = {:.6}, best = {:.6})",
                                self.wait, self.patience, self.monitor, current, best
                            );
                        }

                        // Check if we should reduce learning rate
                        if self.wait >= self.patience {
                            if self.verbose {
                                println!(
                                    "Epoch {}: Reducing learning rate by factor of {:.3}",
                                    epoch + 1,
                                    self.factor
                                );
                            }

                            // Reset wait counter and set cooldown
                            self.wait = 0;
                            self.cooldown_counter = self.cooldown;

                            // Return action to reduce learning rate
                            return Ok(CallbackAction::ReduceLearningRate(self.factor));
                        }
                    }
                }
            }
        } else if self.verbose {
            println!(
                "LR Reduction: metric '{}' not found in training history",
                self.monitor
            );
        }

        Ok(CallbackAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::metrics::TrainingMetrics;
    use std::collections::HashMap;

    #[test]
    fn test_learning_rate_reduction_creation() {
        let lr_reducer = LearningRateReduction::<f32>::new(
            "val_loss".to_string(),
            0.5,
            3,
            0.001,
            2,
            1e-6,
            "min".to_string(),
            true,
        );

        assert_eq!(lr_reducer.monitor, "val_loss");
        assert_eq!(lr_reducer.factor, 0.5);
        assert_eq!(lr_reducer.patience, 3);
        assert_eq!(lr_reducer.min_delta, 0.001);
        assert_eq!(lr_reducer.cooldown, 2);
        assert_eq!(lr_reducer.min_lr, 1e-6);
        assert_eq!(lr_reducer.mode, "min");
        assert!(lr_reducer.verbose);
    }

    #[test]
    fn test_with_defaults() {
        let lr_reducer = LearningRateReduction::<f32>::with_defaults("val_loss".to_string());
        assert_eq!(lr_reducer.mode, "min");

        let lr_reducer = LearningRateReduction::<f32>::with_defaults("val_accuracy".to_string());
        assert_eq!(lr_reducer.mode, "max");
    }

    #[test]
    fn test_for_minimizing() {
        let lr_reducer =
            LearningRateReduction::<f32>::for_minimizing("val_loss".to_string(), 0.1, 5);
        assert_eq!(lr_reducer.mode, "min");
        assert_eq!(lr_reducer.factor, 0.1);
        assert_eq!(lr_reducer.patience, 5);
    }

    #[test]
    fn test_for_maximizing() {
        let lr_reducer =
            LearningRateReduction::<f32>::for_maximizing("val_accuracy".to_string(), 0.2, 3);
        assert_eq!(lr_reducer.mode, "max");
        assert_eq!(lr_reducer.factor, 0.2);
        assert_eq!(lr_reducer.patience, 3);
    }

    #[test]
    fn test_builder_methods() {
        let lr_reducer = LearningRateReduction::<f32>::with_defaults("val_loss".to_string())
            .factor(0.5)
            .patience(7)
            .verbose(true)
            .min_lr(1e-8)
            .cooldown(3)
            .min_delta(0.01);

        assert_eq!(lr_reducer.factor, 0.5);
        assert_eq!(lr_reducer.patience, 7);
        assert!(lr_reducer.verbose);
        assert_eq!(lr_reducer.min_lr, 1e-8);
        assert_eq!(lr_reducer.cooldown, 3);
        assert_eq!(lr_reducer.min_delta, 0.01);
    }

    #[test]
    fn test_is_better() {
        let lr_reducer_min =
            LearningRateReduction::<f32>::for_minimizing("val_loss".to_string(), 0.1, 3);
        let lr_reducer_max =
            LearningRateReduction::<f32>::for_maximizing("val_accuracy".to_string(), 0.1, 3);

        // For min mode, lower is better
        assert!(lr_reducer_min.is_better(0.5, 0.6));
        assert!(!lr_reducer_min.is_better(0.6, 0.5));

        // For max mode, higher is better
        assert!(lr_reducer_max.is_better(0.9, 0.8));
        assert!(!lr_reducer_max.is_better(0.8, 0.9));
    }

    #[test]
    #[should_panic]
    fn test_invalid_mode() {
        LearningRateReduction::<f32>::new(
            "val_loss".to_string(),
            0.1,
            3,
            0.001,
            0,
            0.0,
            "invalid".to_string(),
            false,
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_factor_too_high() {
        LearningRateReduction::<f32>::new(
            "val_loss".to_string(),
            1.1, // Invalid: > 1
            3,
            0.001,
            0,
            0.0,
            "min".to_string(),
            false,
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_factor_zero() {
        LearningRateReduction::<f32>::new(
            "val_loss".to_string(),
            0.0, // Invalid: <= 0
            3,
            0.001,
            0,
            0.0,
            "min".to_string(),
            false,
        );
    }

    #[test]
    #[should_panic]
    fn test_zero_patience() {
        LearningRateReduction::<f32>::new(
            "val_loss".to_string(),
            0.1,
            0, // Invalid: must be > 0
            0.001,
            0,
            0.0,
            "min".to_string(),
            false,
        );
    }
}
