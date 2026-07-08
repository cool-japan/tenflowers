//! Training utilities for autograd neural networks

use super::{autograd_layer::AutogradLayer, optimizer::AutogradOptimizer, traits::NeuralLayer};
use crate::tape::{GradientTape, TrackedTensor};
use scirs2_core::numeric::{Float, One, Zero};
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
        + scirs2_core::num_traits::FromPrimitive
        + scirs2_core::num_traits::Signed
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
            total_loss / T::from_usize(count).unwrap_or_else(|| T::one())
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

    /// Save training checkpoint.
    ///
    /// Checkpointing the trainer requires serializing the optimizer's internal
    /// state (momentum/velocity buffers and the gradient tape) together with the
    /// generic training metrics. That serialization infrastructure does not yet
    /// exist for the autograd optimizer, so this returns an explicit error
    /// instead of silently pretending the checkpoint was written.
    pub fn save_checkpoint(&self, _path: &str) -> Result<()> {
        Err(tenflowers_core::TensorError::not_implemented_simple(
            "trainer checkpoint saving is not implemented: the autograd optimizer and gradient \
             tape state are not yet serializable"
                .to_string(),
        ))
    }

    /// Load training checkpoint.
    ///
    /// See [`AutogradTrainer::save_checkpoint`]: the matching deserialization
    /// path is not implemented, so this returns an explicit error rather than
    /// leaving the trainer silently unchanged.
    pub fn load_checkpoint(&mut self, _path: &str) -> Result<()> {
        Err(tenflowers_core::TensorError::not_implemented_simple(
            "trainer checkpoint loading is not implemented: the autograd optimizer and gradient \
             tape state are not yet deserializable"
                .to_string(),
        ))
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

    /// Extract the scalar value from a (reduced) tensor.
    ///
    /// This reads the first element of the tensor's flat data, which is the
    /// genuine value of a scalar loss/metric produced by a reduction such as
    /// `mean`. An empty tensor has no scalar value and is reported as an error.
    fn extract_scalar_value(&self, tensor: &Tensor<T>) -> Result<T> {
        let values = tensor.to_vec()?;
        values.into_iter().next().ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "cannot extract a scalar value from an empty tensor".to_string(),
            )
        })
    }

    /// Compute classification accuracy from real predictions and targets.
    ///
    /// The accuracy is the fraction of correctly classified samples:
    ///
    /// - For multi-class outputs shaped `[batch, num_classes]`, the predicted
    ///   class is the `argmax` over the class axis and the target class is the
    ///   `argmax` of the (one-hot) target row.
    /// - For 1-D outputs the comparison is per-element: the prediction and
    ///   target are each thresholded at `0.5` (binary classification) before
    ///   being compared.
    ///
    /// Returns a value in `[0, 1]`.
    pub fn compute_accuracy(
        &self,
        predictions: &TrackedTensor<T>,
        targets: &TrackedTensor<T>,
    ) -> Result<T> {
        let pred_dims = predictions.tensor.shape().dims().to_vec();
        let target_dims = targets.tensor.shape().dims().to_vec();

        let pred_values = predictions.tensor.to_vec()?;
        let target_values = targets.tensor.to_vec()?;

        if pred_values.is_empty() {
            return Err(tenflowers_core::TensorError::invalid_operation_simple(
                "cannot compute accuracy from empty predictions".to_string(),
            ));
        }

        // Determine whether we are in a multi-class (2-D) layout.
        let (num_samples, num_classes) = match pred_dims.len() {
            2 => (pred_dims[0], pred_dims[1]),
            _ => (pred_values.len(), 1usize),
        };

        let correct = if num_classes > 1 {
            // Multi-class: compare argmax of each prediction row with the argmax
            // of the corresponding (one-hot or probability) target row.
            if target_dims.len() != 2
                || target_dims[0] != num_samples
                || target_dims[1] != num_classes
            {
                return Err(tenflowers_core::TensorError::invalid_operation_simple(
                    format!(
                    "accuracy: prediction shape {pred_dims:?} and target shape {target_dims:?} \
                     are incompatible"
                ),
                ));
            }

            let mut hits = 0usize;
            for sample in 0..num_samples {
                let base = sample * num_classes;
                let pred_class = Self::argmax(&pred_values[base..base + num_classes]);
                let target_class = Self::argmax(&target_values[base..base + num_classes]);
                if pred_class == target_class {
                    hits += 1;
                }
            }
            hits
        } else {
            // Binary / regression-as-classification: threshold both sides at 0.5.
            if target_values.len() != pred_values.len() {
                return Err(tenflowers_core::TensorError::invalid_operation_simple(
                    format!(
                        "accuracy: {} predictions but {} targets",
                        pred_values.len(),
                        target_values.len()
                    ),
                ));
            }

            let half = T::from_f64(0.5).unwrap_or_else(T::zero);
            pred_values
                .iter()
                .zip(target_values.iter())
                .filter(|(pred, target)| {
                    let pred_label = **pred > half;
                    let target_label = **target > half;
                    pred_label == target_label
                })
                .count()
        };

        let total = num_samples.max(1);
        let accuracy = correct as f64 / total as f64;
        T::from_f64(accuracy).ok_or_else(|| {
            tenflowers_core::TensorError::invalid_operation_simple(
                "failed to convert computed accuracy into the tensor element type".to_string(),
            )
        })
    }

    /// Index of the maximum value in a slice (argmax). Returns 0 for an empty
    /// slice. Uses a total order tolerant of partially-ordered float types.
    fn argmax(values: &[T]) -> usize {
        let mut best_index = 0usize;
        if values.is_empty() {
            return best_index;
        }
        let mut best_value = values[0];
        for (index, value) in values.iter().enumerate().skip(1) {
            if *value > best_value {
                best_value = *value;
                best_index = index;
            }
        }
        best_index
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
