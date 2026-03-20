//! Model checkpointing callback implementation
//!
//! This module provides functionality for saving model checkpoints during training,
//! either at every epoch or only when the monitored metric improves.

use crate::{
    optimizers::Optimizer,
    trainer::metrics::{CallbackAction, TrainingState},
    Model,
};
use tenflowers_core::Result;

use super::Callback;

/// Model checkpointing callback
///
/// Saves the model at regular intervals or when performance improves.
/// This is essential for preserving the best model state during training.
#[derive(Debug, Clone)]
pub struct ModelCheckpoint<T> {
    /// Filepath where the model should be saved
    pub filepath: String,
    /// Name of the metric to monitor (e.g., "val_loss", "val_accuracy")
    pub monitor: String,
    /// Whether to maximize ("max") or minimize ("min") the monitored metric
    pub mode: String,
    /// If True, save only when the monitored metric improves
    pub save_best_only: bool,
    /// Current best score
    best_score: Option<f32>,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ModelCheckpoint<T> {
    /// Create a new model checkpoint callback
    ///
    /// # Arguments
    /// * `filepath` - Path where the model should be saved. Can include format strings like "model_epoch_{epoch}.pth"
    /// * `monitor` - Name of the metric to monitor
    /// * `mode` - "min" for metrics that should decrease (like loss) or "max" for metrics that should increase (like accuracy)
    /// * `save_best_only` - If true, only save when the monitored metric improves
    pub fn new(filepath: String, monitor: String, mode: String, save_best_only: bool) -> Self {
        assert!(
            mode == "min" || mode == "max",
            "Mode must be either 'min' or 'max'"
        );

        Self {
            filepath,
            monitor,
            mode,
            save_best_only,
            best_score: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a checkpoint that saves the model with the lowest monitored metric
    pub fn save_best_min(filepath: String, monitor: String) -> Self {
        Self::new(filepath, monitor, "min".to_string(), true)
    }

    /// Create a checkpoint that saves the model with the highest monitored metric
    pub fn save_best_max(filepath: String, monitor: String) -> Self {
        Self::new(filepath, monitor, "max".to_string(), true)
    }

    /// Create a checkpoint that saves the model at every epoch
    pub fn save_all(filepath: String) -> Self {
        Self::new(
            filepath,
            "val_loss".to_string(), // Default monitor (not used for save_all)
            "min".to_string(),
            false,
        )
    }

    /// Get the current best score
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Reset the checkpoint state
    pub fn reset(&mut self) {
        self.best_score = None;
    }

    /// Check if the current score is better than the best score
    fn is_better_score(&self, current_score: f32, best_score: f32) -> bool {
        if self.mode == "min" {
            current_score < best_score
        } else {
            current_score > best_score
        }
    }

    /// Format the filepath with epoch information
    fn format_filepath(&self, epoch: usize) -> String {
        self.filepath.replace("{epoch}", &epoch.to_string())
    }
}

impl<T> Callback<T> for ModelCheckpoint<T>
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
        let should_save = if self.save_best_only {
            // Only save if the monitored metric improved
            if let Some(latest_val) = state.val_history.last() {
                if let Some(current_score) = latest_val.metrics.get(&self.monitor) {
                    match self.best_score {
                        None => {
                            // First epoch - save and record as best
                            self.best_score = Some(*current_score);
                            println!(
                                "ModelCheckpoint: first epoch, setting initial best {} = {:.6}",
                                self.monitor, current_score
                            );
                            true
                        }
                        Some(best) => {
                            if self.is_better_score(*current_score, best) {
                                // Improvement found - save and update best
                                self.best_score = Some(*current_score);
                                println!(
                                    "ModelCheckpoint: new best {} = {:.6} (previous: {:.6})",
                                    self.monitor, current_score, best
                                );
                                true
                            } else {
                                println!(
                                    "ModelCheckpoint: no improvement, {} = {:.6} (best: {:.6})",
                                    self.monitor, current_score, best
                                );
                                false
                            }
                        }
                    }
                } else {
                    println!(
                        "ModelCheckpoint: metric '{}' not found in validation metrics. Available: {:?}",
                        self.monitor,
                        latest_val.metrics.keys().collect::<Vec<_>>()
                    );
                    false
                }
            } else {
                println!("ModelCheckpoint: no validation history available");
                false
            }
        } else {
            // Save every epoch
            true
        };

        if should_save {
            let filepath = self.format_filepath(epoch);
            println!("Saving model checkpoint at epoch {} to {}", epoch, filepath);
            Ok(CallbackAction::SaveModel(filepath))
        } else {
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
    fn test_model_checkpoint_creation() {
        let checkpoint = ModelCheckpoint::<f32>::new(
            "model.pth".to_string(),
            "val_loss".to_string(),
            "min".to_string(),
            true,
        );

        assert_eq!(checkpoint.filepath, "model.pth");
        assert_eq!(checkpoint.monitor, "val_loss");
        assert_eq!(checkpoint.mode, "min");
        assert!(checkpoint.save_best_only);
        assert_eq!(checkpoint.best_score(), None);
    }

    #[test]
    fn test_save_best_min() {
        let checkpoint = ModelCheckpoint::<f32>::save_best_min(
            "best_model.pth".to_string(),
            "val_loss".to_string(),
        );
        assert_eq!(checkpoint.mode, "min");
        assert!(checkpoint.save_best_only);
    }

    #[test]
    fn test_save_best_max() {
        let checkpoint = ModelCheckpoint::<f32>::save_best_max(
            "best_model.pth".to_string(),
            "val_accuracy".to_string(),
        );
        assert_eq!(checkpoint.mode, "max");
        assert!(checkpoint.save_best_only);
    }

    #[test]
    fn test_save_all() {
        let checkpoint = ModelCheckpoint::<f32>::save_all("model_epoch_{epoch}.pth".to_string());
        assert!(!checkpoint.save_best_only);
    }

    #[test]
    fn test_is_better_score_min_mode() {
        let checkpoint =
            ModelCheckpoint::<f32>::save_best_min("model.pth".to_string(), "val_loss".to_string());

        // For min mode, lower is better
        assert!(checkpoint.is_better_score(0.5, 0.6));
        assert!(!checkpoint.is_better_score(0.6, 0.5));
        assert!(!checkpoint.is_better_score(0.5, 0.5)); // Equal is not better
    }

    #[test]
    fn test_is_better_score_max_mode() {
        let checkpoint = ModelCheckpoint::<f32>::save_best_max(
            "model.pth".to_string(),
            "val_accuracy".to_string(),
        );

        // For max mode, higher is better
        assert!(checkpoint.is_better_score(0.9, 0.8));
        assert!(!checkpoint.is_better_score(0.8, 0.9));
        assert!(!checkpoint.is_better_score(0.8, 0.8)); // Equal is not better
    }

    #[test]
    fn test_format_filepath() {
        let checkpoint = ModelCheckpoint::<f32>::save_all("model_epoch_{epoch}.pth".to_string());
        let formatted = checkpoint.format_filepath(42);
        assert_eq!(formatted, "model_epoch_42.pth");
    }

    #[test]
    #[should_panic]
    fn test_invalid_mode() {
        ModelCheckpoint::<f32>::new(
            "model.pth".to_string(),
            "val_loss".to_string(),
            "invalid".to_string(),
            true,
        );
    }
}
