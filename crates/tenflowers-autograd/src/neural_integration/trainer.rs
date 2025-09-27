//! Training utilities for autograd neural networks

use super::{autograd_layer::AutogradLayer, optimizer::AutogradOptimizer, traits::NeuralLayer};
use crate::tape::{GradientTape, TrackedTensor};
use num_traits::{Float, One, Zero};
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor};

/// Autograd-integrated trainer for neural networks
pub struct AutogradTrainer<T> {
    /// Gradient tape for automatic differentiation
    #[allow(dead_code)]
    tape: Arc<Mutex<GradientTape>>,
    /// Optimizer for parameter updates
    optimizer: AutogradOptimizer<T>,
    /// Training metrics
    metrics: TrainingMetrics<T>,
}

/// Training metrics tracking
#[derive(Debug, Clone)]
pub struct TrainingMetrics<T> {
    /// Training loss history
    pub training_loss: Vec<T>,
    /// Validation loss history
    pub validation_loss: Vec<T>,
    /// Training accuracy history
    pub training_accuracy: Vec<T>,
    /// Validation accuracy history
    pub validation_accuracy: Vec<T>,
    /// Current epoch
    pub current_epoch: usize,
    /// Current step
    pub current_step: usize,
}

impl<T> Default for TrainingMetrics<T> {
    fn default() -> Self {
        Self {
            training_loss: Vec::new(),
            validation_loss: Vec::new(),
            training_accuracy: Vec::new(),
            validation_accuracy: Vec::new(),
            current_epoch: 0,
            current_step: 0,
        }
    }
}

impl<T> AutogradTrainer<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive
        + num_traits::Signed
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new trainer
    pub fn new(tape: Arc<Mutex<GradientTape>>, optimizer: AutogradOptimizer<T>) -> Self {
        Self {
            tape,
            optimizer,
            metrics: TrainingMetrics::default(),
        }
    }

    /// Train a single step
    pub fn train_step<L>(
        &mut self,
        layer: &mut AutogradLayer<T, L>,
        input: &TrackedTensor<T>,
        target: &TrackedTensor<T>,
    ) -> Result<T>
    where
        L: NeuralLayer<T> + Clone,
    {
        // Forward pass
        let output = layer.forward(input)?;

        // Compute loss (simple MSE for now)
        let diff = output.sub(target)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        // Compute gradients
        let parameters: Vec<_> = layer.parameters().iter().collect();
        let gradients = self.optimizer.compute_gradients(&loss, &parameters)?;

        // Apply gradients (this updates the tracked parameters)
        self.optimizer
            .apply_gradients(layer.parameters_mut(), &gradients)?;

        // Update metrics
        self.metrics.current_step += 1;

        // Extract loss value for metrics
        let loss_value = self.extract_scalar_value(&loss.tensor)?;
        self.metrics.training_loss.push(loss_value);

        Ok(loss_value)
    }

    /// Train multiple steps
    pub fn train_batch<L>(
        &mut self,
        layer: &mut AutogradLayer<T, L>,
        inputs: &[TrackedTensor<T>],
        targets: &[TrackedTensor<T>],
    ) -> Result<Vec<T>>
    where
        L: NeuralLayer<T> + Clone,
    {
        let mut losses = Vec::new();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let loss = self.train_step(layer, input, target)?;
            losses.push(loss);
        }

        Ok(losses)
    }

    /// Validate the model
    pub fn validate_step<L>(
        &mut self,
        layer: &mut AutogradLayer<T, L>,
        input: &TrackedTensor<T>,
        target: &TrackedTensor<T>,
    ) -> Result<T>
    where
        L: NeuralLayer<T> + Clone,
    {
        // Set to evaluation mode
        layer.set_training(false);

        // Forward pass only
        let output = layer.forward(input)?;

        // Compute loss
        let diff = output.sub(target)?;
        let loss = diff.mul(&diff)?.mean(None, false)?;

        // Extract loss value
        let loss_value = self.extract_scalar_value(&loss.tensor)?;
        self.metrics.validation_loss.push(loss_value);

        // Reset to training mode
        layer.set_training(true);

        Ok(loss_value)
    }

    /// Train for one epoch
    pub fn train_epoch<L>(
        &mut self,
        layer: &mut AutogradLayer<T, L>,
        inputs: &[TrackedTensor<T>],
        targets: &[TrackedTensor<T>],
    ) -> Result<T>
    where
        L: NeuralLayer<T> + Clone,
    {
        let mut total_loss = T::zero();
        let mut count = 0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let loss = self.train_step(layer, input, target)?;
            total_loss = total_loss + loss;
            count += 1;
        }

        self.metrics.current_epoch += 1;

        // Calculate average loss
        let avg_loss = if count > 0 {
            total_loss / T::from_usize(count).unwrap_or_else(T::one)
        } else {
            T::zero()
        };

        Ok(avg_loss)
    }

    /// Get training metrics
    pub fn metrics(&self) -> &TrainingMetrics<T> {
        &self.metrics
    }

    /// Get mutable training metrics
    pub fn metrics_mut(&mut self) -> &mut TrainingMetrics<T> {
        &mut self.metrics
    }

    /// Reset training metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = TrainingMetrics::default();
    }

    /// Save training checkpoint
    pub fn save_checkpoint(&self, _path: &str) -> Result<()> {
        // Placeholder for checkpoint saving
        // In a real implementation, this would serialize the optimizer state and metrics
        Ok(())
    }

    /// Load training checkpoint
    pub fn load_checkpoint(&mut self, _path: &str) -> Result<()> {
        // Placeholder for checkpoint loading
        // In a real implementation, this would deserialize the optimizer state and metrics
        Ok(())
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> T {
        self.optimizer.learning_rate()
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, learning_rate: T) {
        self.optimizer.set_learning_rate(learning_rate);
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) -> Result<()> {
        self.optimizer.zero_grad()
    }

    /// Extract scalar value from tensor (helper method)
    fn extract_scalar_value(&self, _tensor: &Tensor<T>) -> Result<T> {
        // For now, just return a placeholder value
        // In a real implementation, this would extract the actual scalar value from the tensor
        Ok(T::zero())
    }

    /// Compute accuracy for classification tasks
    pub fn compute_accuracy(
        &self,
        predictions: &TrackedTensor<T>,
        targets: &TrackedTensor<T>,
    ) -> Result<T> {
        // Placeholder implementation for accuracy computation
        // In a real implementation, this would compute classification accuracy
        let _pred_argmax = predictions; // Would compute argmax
        let _target_argmax = targets; // Would compute argmax

        // Return placeholder accuracy
        Ok(T::from_f64(0.95).unwrap_or_else(T::zero))
    }

    /// Early stopping check
    pub fn should_early_stop(&self, patience: usize, min_delta: T) -> bool {
        if self.metrics.validation_loss.len() < patience + 1 {
            return false;
        }

        let recent_losses =
            &self.metrics.validation_loss[self.metrics.validation_loss.len() - patience - 1..];
        let best_loss = recent_losses[0];

        // Check if validation loss hasn't improved by min_delta for 'patience' epochs
        for &loss in &recent_losses[1..] {
            if best_loss - loss > min_delta {
                return false; // Still improving
            }
        }

        true // Should stop
    }

    /// Get the best validation loss
    pub fn best_validation_loss(&self) -> Option<T> {
        self.metrics
            .validation_loss
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Get the latest training loss
    pub fn latest_training_loss(&self) -> Option<T> {
        self.metrics.training_loss.last().copied()
    }

    /// Get the latest validation loss
    pub fn latest_validation_loss(&self) -> Option<T> {
        self.metrics.validation_loss.last().copied()
    }
}
