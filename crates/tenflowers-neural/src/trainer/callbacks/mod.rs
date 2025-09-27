//! Training callbacks module
//!
//! This module provides the callback system for training, including
//! the base callback trait and various callback implementations.

pub mod checkpoint;
pub mod early_stopping;
pub mod lr_reduction;

#[cfg(feature = "tensorboard")]
pub mod tensorboard;

use crate::{optimizers::Optimizer, trainer::metrics::CallbackAction, Model};
use tenflowers_core::Result;

use super::metrics::{TrainingMetrics, TrainingState};

/// Callback trait for training events
///
/// Callbacks are called at various points during training to allow
/// custom behavior like early stopping, checkpointing, logging, etc.
pub trait Callback<T>: std::fmt::Debug
where
    T: Clone + Default,
{
    /// Called at the beginning of training
    fn on_train_begin(&mut self, _state: &TrainingState) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called at the end of training
    fn on_train_end(&mut self, _state: &TrainingState) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called at the beginning of each epoch
    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called at the end of each epoch
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        _state: &TrainingState,
        _model: &dyn Model<T>,
        _optimizer: &mut dyn Optimizer<T>,
    ) -> Result<CallbackAction>;

    /// Called at the beginning of each batch
    fn on_batch_begin(&mut self, _batch: usize, _state: &TrainingState) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called at the end of each batch
    fn on_batch_end(
        &mut self,
        _batch: usize,
        _metrics: &TrainingMetrics,
        _state: &TrainingState,
    ) -> CallbackAction {
        CallbackAction::Continue
    }
}

// Re-export commonly used callbacks
pub use checkpoint::ModelCheckpoint;
pub use early_stopping::EarlyStopping;
pub use lr_reduction::LearningRateReduction;

#[cfg(feature = "tensorboard")]
pub use tensorboard::TensorboardCallback;
