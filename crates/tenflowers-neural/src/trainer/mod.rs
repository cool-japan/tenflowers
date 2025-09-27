//! Training module for neural networks
//!
//! This module provides a comprehensive training system including:
//! - Training state and metrics tracking
//! - Callback system for training events
//! - Main trainer for model training

pub mod callbacks;
pub mod metrics;

// Re-export commonly used types
pub use callbacks::{Callback, EarlyStopping, LearningRateReduction, ModelCheckpoint};
pub use metrics::{CallbackAction, TrainingMetrics, TrainingState};

#[cfg(feature = "tensorboard")]
pub use callbacks::TensorboardCallback;

use crate::{optimizers::Optimizer, Model};
use tenflowers_core::{Result, Tensor};

/// Main trainer for neural network models
///
/// The trainer orchestrates the training process, handling:
/// - Training loop execution
/// - Callback management and execution
/// - Metrics collection and state tracking
/// - Model and optimizer coordination
#[derive(Debug)]
pub struct Trainer<T> {
    /// List of callbacks to execute during training
    callbacks: Vec<Box<dyn Callback<T>>>,
    /// Whether to print training progress
    verbose: bool,
}

impl<T> Trainer<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + num_traits::Zero
        + num_traits::One
        + num_traits::Signed
        + bytemuck::Pod,
{
    /// Create a new trainer
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            verbose: true,
        }
    }

    /// Create a new trainer with specified verbosity
    pub fn with_verbose(verbose: bool) -> Self {
        Self {
            callbacks: Vec::new(),
            verbose,
        }
    }

    /// Add a callback to the trainer
    pub fn add_callback(&mut self, callback: Box<dyn Callback<T>>) -> &mut Self {
        self.callbacks.push(callback);
        self
    }

    /// Set verbosity level
    pub fn set_verbose(&mut self, verbose: bool) -> &mut Self {
        self.verbose = verbose;
        self
    }

    /// Get number of registered callbacks
    pub fn num_callbacks(&self) -> usize {
        self.callbacks.len()
    }

    /// Clear all callbacks
    pub fn clear_callbacks(&mut self) {
        self.callbacks.clear();
    }

    /// Fit a model using training data
    ///
    /// # Arguments
    /// * `model` - The model to train
    /// * `optimizer` - The optimizer to use for parameter updates
    /// * `train_data` - Training dataset iterator
    /// * `val_data` - Optional validation dataset iterator
    /// * `epochs` - Number of training epochs
    /// * `loss_fn` - Loss function to use
    ///
    /// # Returns
    /// Final training state with metrics and history
    pub fn fit<M, O, D>(
        &mut self,
        model: &mut M,
        optimizer: &mut O,
        train_data: D,
        val_data: Option<D>,
        epochs: usize,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    ) -> Result<TrainingState>
    where
        M: Model<T>,
        O: Optimizer<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)> + Clone,
    {
        let mut state = TrainingState::new();

        // Call on_train_begin for all callbacks
        for callback in &mut self.callbacks {
            match callback.on_train_begin(&state) {
                CallbackAction::Continue => continue,
                CallbackAction::Stop => return Ok(state),
                _ => {} // Ignore other actions during train begin
            }
        }

        let mut should_continue = true;

        for epoch in 0..epochs {
            if !should_continue {
                break;
            }

            state.epoch = epoch;

            // Call on_epoch_begin for all callbacks
            for callback in &mut self.callbacks {
                match callback.on_epoch_begin(epoch, &state) {
                    CallbackAction::Continue => continue,
                    CallbackAction::Stop => {
                        should_continue = false;
                        break;
                    }
                    _ => {} // Ignore other actions during epoch begin
                }
            }

            if !should_continue {
                break;
            }

            // Training phase
            let train_metrics =
                self.train_epoch(model, optimizer, train_data.clone(), loss_fn, &mut state)?;

            state.add_training_metrics(train_metrics);

            // Validation phase (if validation data provided)
            if let Some(ref val_dataset) = val_data {
                let val_metrics =
                    self.validate_epoch(model, val_dataset.clone(), loss_fn, epoch, &state)?;
                state.add_validation_metrics(val_metrics);
            }

            // Call on_epoch_end for all callbacks
            for callback in &mut self.callbacks {
                match callback.on_epoch_end(epoch, &state, model, optimizer)? {
                    CallbackAction::Continue => continue,
                    CallbackAction::Stop => {
                        if self.verbose {
                            println!("Training stopped by callback at epoch {}", epoch + 1);
                        }
                        should_continue = false;
                        break;
                    }
                    CallbackAction::ReduceLearningRate(factor) => {
                        let current_lr = optimizer.get_learning_rate();
                        let new_lr = current_lr * factor;
                        optimizer.set_learning_rate(new_lr);
                        if self.verbose {
                            println!(
                                "Learning rate reduced from {:.6} to {:.6}",
                                current_lr, new_lr
                            );
                        }
                    }
                    CallbackAction::SaveModel(filepath) => {
                        if self.verbose {
                            println!("Saving model to: {}", filepath);
                        }
                        // Note: Actual model saving would be implemented here
                        // This would typically call model.save(&filepath) or similar
                    }
                }
            }

            // Print epoch summary
            if self.verbose {
                self.print_epoch_summary(epoch, &state);
            }
        }

        // Call on_train_end for all callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(&state);
        }

        Ok(state)
    }

    /// Train for one epoch
    fn train_epoch<M, O, D>(
        &mut self,
        model: &mut M,
        optimizer: &mut O,
        train_data: D,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
        state: &mut TrainingState,
    ) -> Result<TrainingMetrics>
    where
        M: Model<T>,
        O: Optimizer<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)>,
    {
        model.set_training(true);

        let mut total_loss = T::zero();
        let mut batch_count = 0;

        for (batch_idx, (inputs, targets)) in train_data.enumerate() {
            // Call on_batch_begin for all callbacks
            for callback in &mut self.callbacks {
                callback.on_batch_begin(batch_idx, state);
            }

            // Forward pass
            let predictions = model.forward(&inputs)?;
            let loss = loss_fn(&predictions, &targets)?;

            // Backward pass
            optimizer.zero_grad(model);
            // Note: Actual backward pass would be implemented here
            // This would typically involve computing gradients and calling optimizer.step()

            // Accumulate loss
            if let Some(loss_scalar) = loss.as_slice() {
                if let Some(&loss_val) = loss_scalar.first() {
                    total_loss = total_loss + T::from(loss_val).unwrap_or_else(|| T::zero());
                }
            }
            batch_count += 1;
            state.step += 1;

            // Create batch metrics
            let batch_metrics = TrainingMetrics::new(
                state.epoch,
                state.step,
                loss.as_slice()
                    .and_then(|s| s.first())
                    .and_then(|&val| num_traits::cast::<T, f32>(val))
                    .unwrap_or(0.0),
            );

            // Call on_batch_end for all callbacks
            for callback in &mut self.callbacks {
                callback.on_batch_end(batch_idx, &batch_metrics, state);
            }
        }

        let avg_loss = if batch_count > 0 {
            total_loss / T::from(batch_count).unwrap_or_else(|| T::one())
        } else {
            T::zero()
        };

        Ok(TrainingMetrics::new(
            state.epoch,
            state.step,
            avg_loss.to_f32().unwrap_or(0.0),
        ))
    }

    /// Validate for one epoch
    fn validate_epoch<M, D>(
        &self,
        model: &mut M,
        val_data: D,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
        epoch: usize,
        state: &TrainingState,
    ) -> Result<TrainingMetrics>
    where
        M: Model<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)>,
    {
        model.set_training(false);

        let mut total_loss = T::zero();
        let mut batch_count = 0;

        for (inputs, targets) in val_data {
            // Forward pass only (no gradient computation)
            let predictions = model.forward(&inputs)?;
            let loss = loss_fn(&predictions, &targets)?;

            // Accumulate loss
            if let Some(loss_scalar) = loss.as_slice() {
                if let Some(&loss_val) = loss_scalar.first() {
                    total_loss = total_loss + T::from(loss_val).unwrap_or_else(|| T::zero());
                }
            }
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 {
            total_loss / T::from(batch_count).unwrap_or_else(|| T::one())
        } else {
            T::zero()
        };

        Ok(TrainingMetrics::new(
            epoch,
            state.step,
            avg_loss.to_f32().unwrap_or(0.0),
        ))
    }

    /// Print epoch summary
    fn print_epoch_summary(&self, epoch: usize, state: &TrainingState) {
        let train_loss = state.latest_train_loss();
        let val_loss = state.latest_val_loss();

        print!("Epoch {}/{}", epoch + 1, state.epoch + 1);

        if let Some(tl) = train_loss {
            print!(" - loss: {:.4}", tl);
        }

        if let Some(vl) = val_loss {
            print!(" - val_loss: {:.4}", vl);
        }

        println!();
    }
}

impl<T> Default for Trainer<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + std::fmt::Debug
        + std::fmt::Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + num_traits::Zero
        + num_traits::One
        + num_traits::Signed
        + bytemuck::Pod,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::<f32>::new();
        assert_eq!(trainer.num_callbacks(), 0);
        assert!(trainer.verbose);
    }

    #[test]
    fn test_trainer_with_verbose() {
        let trainer = Trainer::<f32>::with_verbose(false);
        assert!(!trainer.verbose);
    }

    #[test]
    fn test_add_callback() {
        let mut trainer = Trainer::<f32>::new();
        let early_stopping = Box::new(EarlyStopping::for_minimizing(
            3,
            0.001,
            "val_loss".to_string(),
        ));

        trainer.add_callback(early_stopping);
        assert_eq!(trainer.num_callbacks(), 1);
    }

    #[test]
    fn test_clear_callbacks() {
        let mut trainer = Trainer::<f32>::new();
        let early_stopping = Box::new(EarlyStopping::for_minimizing(
            3,
            0.001,
            "val_loss".to_string(),
        ));

        trainer.add_callback(early_stopping);
        assert_eq!(trainer.num_callbacks(), 1);

        trainer.clear_callbacks();
        assert_eq!(trainer.num_callbacks(), 0);
    }

    #[test]
    fn test_set_verbose() {
        let mut trainer = Trainer::<f32>::new();
        assert!(trainer.verbose);

        trainer.set_verbose(false);
        assert!(!trainer.verbose);
    }
}
