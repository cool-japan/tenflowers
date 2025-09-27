//! Tensorboard callback for training visualization
//!
//! This callback integrates with tensorboard for real-time training metrics visualization.

use super::{Callback, CallbackAction, TrainingMetrics, TrainingState};
use crate::{optimizers::Optimizer, Model};
use tenflowers_core::Result;

/// Tensorboard callback for training visualization
///
/// Records training metrics and losses to tensorboard for visualization
#[derive(Debug)]
pub struct TensorboardCallback {
    log_dir: String,
    write_graph: bool,
}

impl TensorboardCallback {
    /// Create a new tensorboard callback
    pub fn new(log_dir: impl Into<String>) -> Self {
        Self {
            log_dir: log_dir.into(),
            write_graph: false,
        }
    }

    /// Enable graph writing
    pub fn with_graph(mut self) -> Self {
        self.write_graph = true;
        self
    }
}

impl<T> Callback<T> for TensorboardCallback
where
    T: Clone + Default,
{
    fn on_epoch_end(
        &mut self,
        _epoch: usize,
        _state: &TrainingState,
        _model: &dyn Model<T>,
        _optimizer: &mut dyn Optimizer<T>,
    ) -> Result<CallbackAction> {
        // TODO: Implement tensorboard logging
        Ok(CallbackAction::Continue)
    }
}
