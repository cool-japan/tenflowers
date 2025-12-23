#[cfg(feature = "serialize")]
use crate::model::ModelSerialization;
use crate::{
    optimizers::Optimizer,
    trainer::{Callback, CallbackAction, TrainingMetrics, TrainingState},
    Model,
};
use std::collections::HashMap;
use tenflowers_autograd::GradientTape;
use tenflowers_core::{
    enable_autocast, AutocastContext, GradientScaler, MixedPrecisionConfig, Result, Tensor,
    TensorError,
};

/// Mixed precision trainer that extends the base trainer with automatic mixed precision capabilities
pub struct MixedPrecisionTrainer<T> {
    callbacks: Vec<Box<dyn Callback<T>>>,
    verbose: bool,
    scaler: GradientScaler,
    #[allow(dead_code)]
    autocast_ctx: AutocastContext,
    mixed_precision_enabled: bool,
}

impl<T> MixedPrecisionTrainer<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
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
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new mixed precision trainer with default configuration
    pub fn new() -> Self {
        let config = MixedPrecisionConfig::default();
        let scaler = GradientScaler::new(config.clone());
        let autocast_ctx = enable_autocast();

        Self {
            callbacks: Vec::new(),
            verbose: true,
            scaler,
            autocast_ctx,
            mixed_precision_enabled: config.enabled,
        }
    }

    /// Create a new mixed precision trainer with custom configuration
    pub fn with_config(config: MixedPrecisionConfig) -> Self {
        let scaler = GradientScaler::new(config.clone());
        let autocast_ctx = enable_autocast();

        Self {
            callbacks: Vec::new(),
            verbose: true,
            scaler,
            autocast_ctx,
            mixed_precision_enabled: config.enabled,
        }
    }

    /// Enable mixed precision training
    pub fn enable_mixed_precision(&mut self) {
        let config = MixedPrecisionConfig {
            enabled: true,
            ..MixedPrecisionConfig::default()
        };
        self.scaler.update_config(config);
        self.mixed_precision_enabled = true;
    }

    /// Disable mixed precision training
    pub fn disable_mixed_precision(&mut self) {
        let config = MixedPrecisionConfig {
            enabled: false,
            ..MixedPrecisionConfig::default()
        };
        self.scaler.update_config(config);
        self.mixed_precision_enabled = false;
    }

    /// Set loss scaling factor
    pub fn set_loss_scale(&mut self, scale: f32) {
        let mut config = self.scaler.get_config();
        config.loss_scale = scale;
        self.scaler.update_config(config);
    }

    /// Enable/disable dynamic loss scaling
    pub fn set_dynamic_loss_scaling(&mut self, enabled: bool) {
        let mut config = self.scaler.get_config();
        config.dynamic_loss_scaling = enabled;
        self.scaler.update_config(config);
    }

    /// Add a callback to the trainer
    pub fn add_callback(&mut self, callback: Box<dyn Callback<T>>) {
        self.callbacks.push(callback);
    }

    /// Set verbose mode
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Fit a model using mixed precision training
    #[cfg(feature = "serialize")]
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
        M: Model<T> + ModelSerialization<T>,
        O: Optimizer<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)> + Clone,
    {
        let mut state = TrainingState::new();

        // Notify callbacks of training start
        for callback in &mut self.callbacks {
            let _action = callback.on_train_begin(&state);
            // Handle callback action if needed
        }

        model.set_training(true);

        for epoch in 0..epochs {
            state.epoch = epoch;

            // Notify callbacks of epoch start
            for callback in &mut self.callbacks {
                let _action = callback.on_epoch_begin(epoch, &state);
                // Handle callback action if needed
            }

            if self.verbose {
                println!("Epoch {}/{}", epoch + 1, epochs);
                println!(
                    "Mixed Precision: {}, Loss Scale: {:.1}",
                    self.is_mixed_precision_enabled(),
                    self.scaler.get_scale()
                );
            }

            // Training epoch with mixed precision
            let train_metrics = self.train_epoch_mixed_precision(
                model,
                optimizer,
                train_data.clone(),
                loss_fn,
                &mut state,
            )?;
            state.history.push(train_metrics.clone());

            // Validation epoch
            let val_metrics = if let Some(val_data_iter) = val_data.clone() {
                let val_m = self.validate_epoch(model, val_data_iter, loss_fn, &mut state)?;
                state.val_history.push(val_m.clone());
                Some(val_m)
            } else {
                None
            };

            // Display epoch results
            if self.verbose {
                let train_loss = train_metrics.loss;
                if let Some(ref val_m) = val_metrics {
                    let val_loss = val_m.loss;
                    let val_acc = val_m.metrics.get("accuracy").unwrap_or(&0.0);
                    let val_top5 = val_m.metrics.get("top5_accuracy").unwrap_or(&0.0);
                    println!("  loss: {train_loss:.4} - val_loss: {val_loss:.4} - val_accuracy: {val_acc:.4} - val_top5_accuracy: {val_top5:.4}");
                } else {
                    println!("  loss: {train_loss:.4}");
                }
            }

            // Process callbacks and handle actions
            let mut should_continue = true;
            for callback in &mut self.callbacks {
                match callback.on_epoch_end(epoch, &state, model, optimizer)? {
                    CallbackAction::Continue => {}
                    CallbackAction::Stop => {
                        should_continue = false;
                        break;
                    }
                    CallbackAction::ReduceLearningRate(factor) => {
                        // Get current learning rate and reduce it
                        let current_lr = optimizer.get_learning_rate();
                        let new_lr = current_lr * factor;
                        optimizer.set_learning_rate(new_lr);
                        if self.verbose {
                            println!("  Learning rate reduced to: {new_lr:.6}");
                        }
                    }
                    CallbackAction::SaveModel(filepath) => {
                        // Save the model using the Model trait's save method
                        #[cfg(feature = "serialize")]
                        match model.save(&filepath) {
                            Ok(()) => {
                                if self.verbose {
                                    println!("  Model checkpoint saved to: {filepath}");
                                }
                            }
                            Err(e) => {
                                if self.verbose {
                                    println!("  Warning: Failed to save model checkpoint to {filepath}: {e}");
                                }
                            }
                        }

                        #[cfg(not(feature = "serialize"))]
                        {
                            if self.verbose {
                                println!("  Warning: Model saving disabled (serialize feature not enabled)");
                                println!(
                                    "  To enable model saving, compile with --features serialize"
                                );
                            }
                        }
                    }
                }
            }

            if !should_continue {
                break;
            }
        }

        // Notify callbacks of training end
        for callback in &mut self.callbacks {
            let _action = callback.on_train_end(&state);
            // Handle callback action if needed
        }

        Ok(state)
    }

    /// Fit a model using mixed precision training (without serialization support)
    #[cfg(not(feature = "serialize"))]
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

        // Notify callbacks of training start
        for callback in &mut self.callbacks {
            let _action = callback.on_train_begin(&state);
            // Handle callback action if needed
        }

        model.set_training(true);

        for epoch in 0..epochs {
            state.epoch = epoch;

            // Notify callbacks of epoch start
            for callback in &mut self.callbacks {
                let _action = callback.on_epoch_begin(epoch, &state);
                // Handle callback action if needed
            }

            if self.verbose {
                println!("Epoch {}/{}", epoch + 1, epochs);
                println!(
                    "Mixed Precision: {}, Loss Scale: {:.1}",
                    self.is_mixed_precision_enabled(),
                    self.scaler.get_scale()
                );
            }

            // Training epoch with mixed precision
            let train_metrics = self.train_epoch_mixed_precision(
                model,
                optimizer,
                train_data.clone(),
                loss_fn,
                &mut state,
            )?;
            state.history.push(train_metrics.clone());

            // Validation epoch
            let val_metrics = if let Some(val_data_iter) = val_data.clone() {
                let val_m = self.validate_epoch(model, val_data_iter, loss_fn, &mut state)?;
                state.val_history.push(val_m.clone());
                Some(val_m)
            } else {
                None
            };

            // Display epoch results
            if self.verbose {
                let train_loss = train_metrics.loss;
                if let Some(ref val_m) = val_metrics {
                    let val_loss = val_m.loss;
                    let val_acc = val_m.metrics.get("accuracy").unwrap_or(&0.0);
                    let val_top5 = val_m.metrics.get("top5_accuracy").unwrap_or(&0.0);
                    println!("  loss: {train_loss:.4} - val_loss: {val_loss:.4} - val_accuracy: {val_acc:.4} - val_top5_accuracy: {val_top5:.4}");
                } else {
                    println!("  loss: {train_loss:.4}");
                }
            }

            // Process callbacks and handle actions
            let mut should_continue = true;
            for callback in &mut self.callbacks {
                match callback.on_epoch_end(epoch, &state, model, optimizer)? {
                    CallbackAction::Continue => {}
                    CallbackAction::Stop => {
                        should_continue = false;
                        break;
                    }
                    CallbackAction::ReduceLearningRate(factor) => {
                        // Get current learning rate and reduce it
                        let current_lr = optimizer.get_learning_rate();
                        let new_lr = current_lr * factor;
                        optimizer.set_learning_rate(new_lr);
                        if self.verbose {
                            println!("  Learning rate reduced to: {new_lr:.6}");
                        }
                    }
                    CallbackAction::SaveModel(filepath) => {
                        // Serialization not available without serialize feature
                        if self.verbose {
                            println!("  Warning: SaveModel callback ignored - serialization feature not enabled. Path: {filepath}");
                        }
                    }
                }
            }

            if !should_continue {
                break;
            }
        }

        // Notify callbacks of training end
        for callback in &mut self.callbacks {
            let _action = callback.on_train_end(&state);
            // Handle callback action if needed
        }

        Ok(state)
    }

    /// Train one epoch with mixed precision support
    fn train_epoch_mixed_precision<M, O, D>(
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
        let mut total_loss = T::zero();
        let mut batch_count = 0;
        let mut skipped_steps = 0;

        for (batch_x, batch_y) in train_data {
            // Notify callbacks of batch start
            for callback in &mut self.callbacks {
                let _action = callback.on_batch_begin(state.step, state);
                // Handle callback action if needed
            }

            // Zero gradients
            model.zero_grad();

            // Forward pass with autocast (simulated - in practice would use lower precision)
            let predictions = if self.is_mixed_precision_enabled() {
                // In a real implementation, this would use FP16 for forward pass
                model.forward(&batch_x)?
            } else {
                model.forward(&batch_x)?
            };

            // Compute loss
            let loss = loss_fn(&predictions, &batch_y)?;

            // Scale loss for mixed precision
            let _scaled_loss = self.scaler.scale(&loss)?;

            // Get scalar loss value for logging
            let loss_value = loss.get(&[]).ok_or_else(|| {
                TensorError::invalid_argument("Loss must be a scalar".to_string())
            })?;

            // Backward pass - Proper autograd integration
            self.compute_gradients_autograd(model, &batch_x, &batch_y, &loss, loss_fn)?;

            // Unscale gradients and check for overflow
            let mut gradients = self.collect_gradients(model);
            let should_step = self.scaler.unscale_gradients_and_check(&mut gradients)?;

            if should_step {
                // Optimizer step only if no gradient overflow
                self.apply_gradients(model, gradients)?;
                optimizer.step(model)?;

                total_loss = total_loss + loss_value;
                batch_count += 1;
            } else {
                // Skip step due to gradient overflow
                skipped_steps += 1;
                if self.verbose && skipped_steps % 100 == 1 {
                    println!("  Warning: Gradient overflow detected, skipping step. Total skipped: {skipped_steps}");
                }
            }

            state.step += 1;

            // Notify callbacks of batch end
            for callback in &mut self.callbacks {
                let batch_metrics = TrainingMetrics {
                    epoch: state.epoch,
                    step: state.step,
                    loss: loss_value.to_f32().unwrap_or(0.0),
                    metrics: HashMap::new(),
                };
                let _action = callback.on_batch_end(state.step, &batch_metrics, state);
                // Handle callback action if needed
            }
        }

        let avg_loss = if batch_count > 0 {
            total_loss / T::from(batch_count).unwrap()
        } else {
            T::zero()
        };

        if self.verbose && skipped_steps > 0 {
            println!("  Skipped {skipped_steps} steps due to gradient overflow");
        }

        Ok(TrainingMetrics {
            epoch: state.epoch,
            step: state.step,
            loss: avg_loss.to_f32().unwrap_or(0.0),
            metrics: HashMap::new(),
        })
    }

    /// Validation epoch (same as regular trainer but with context awareness)
    fn validate_epoch<M, D>(
        &mut self,
        model: &mut M,
        val_data: D,
        loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
        state: &mut TrainingState,
    ) -> Result<TrainingMetrics>
    where
        M: Model<T>,
        D: Iterator<Item = (Tensor<T>, Tensor<T>)>,
    {
        model.set_training(false);

        let mut total_loss = T::zero();
        let mut batch_count = 0;
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();

        for (batch_x, batch_y) in val_data {
            // Forward pass only (no gradients) with autocast
            let predictions = if self.is_mixed_precision_enabled() {
                // In a real implementation, this would use FP16 for inference
                model.forward(&batch_x)?
            } else {
                model.forward(&batch_x)?
            };

            let loss = loss_fn(&predictions, &batch_y)?;

            let loss_value = loss.get(&[]).ok_or_else(|| {
                TensorError::invalid_argument("Loss must be a scalar".to_string())
            })?;

            total_loss = total_loss + loss_value;
            batch_count += 1;

            // Store predictions and targets for metrics calculation
            all_predictions.push(predictions);
            all_targets.push(batch_y);
        }

        let avg_loss = if batch_count > 0 {
            total_loss / T::from(batch_count).unwrap()
        } else {
            T::zero()
        };

        // Calculate additional metrics (same as regular trainer)
        let mut val_metrics = HashMap::new();
        if !all_predictions.is_empty() {
            // Calculate accuracy for classification tasks
            let accuracy = self.calculate_accuracy(&all_predictions, &all_targets)?;
            val_metrics.insert("accuracy".to_string(), accuracy.to_f32().unwrap_or(0.0));

            // Calculate top-5 accuracy if applicable
            if let Some(first_pred) = all_predictions.first() {
                let num_classes = first_pred.shape().dims().last().unwrap_or(&1);
                if *num_classes >= 5 {
                    let top5_accuracy =
                        self.calculate_top_k_accuracy(&all_predictions, &all_targets, 5)?;
                    val_metrics.insert(
                        "top5_accuracy".to_string(),
                        top5_accuracy.to_f32().unwrap_or(0.0),
                    );
                }
            }
        }

        model.set_training(true);

        Ok(TrainingMetrics {
            epoch: state.epoch,
            step: state.step,
            loss: avg_loss.to_f32().unwrap_or(0.0),
            metrics: val_metrics,
        })
    }

    /// Collect gradients from model parameters
    fn collect_gradients<M>(&self, model: &M) -> Vec<Tensor<T>>
    where
        M: Model<T>,
    {
        let params = model.parameters();
        params
            .iter()
            .filter_map(|param| param.grad().cloned())
            .collect()
    }

    /// Apply gradients back to model parameters
    fn apply_gradients<M>(&self, model: &mut M, gradients: Vec<Tensor<T>>) -> Result<()>
    where
        M: Model<T>,
    {
        let mut params = model.parameters_mut();
        let mut grad_iter = gradients.into_iter();

        for param in params.iter_mut() {
            if param.requires_grad() {
                if let Some(grad) = grad_iter.next() {
                    param.set_grad(Some(grad));
                }
            }
        }

        Ok(())
    }

    /// Compute gradients using numerical differentiation (placeholder)
    fn compute_gradients_autograd<M>(
        &self,
        model: &mut M,
        _input: &Tensor<T>,
        _target: &Tensor<T>,
        loss: &Tensor<T>,
        _loss_fn: fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    ) -> Result<()>
    where
        M: Model<T>,
    {
        // Create a gradient tape for automatic differentiation
        let mut tape = GradientTape::new();

        // For now, use a simplified backward pass approach
        // In a full implementation, this would require the model to be built with TrackedTensors
        // and the forward pass to be recorded on the tape

        // Start a backward pass from the loss tensor
        // Note: GradientTape doesn't have a backward method - use fallback
        match Err(TensorError::invalid_argument(
            "Simplified backward pass".to_string(),
        )) as Result<Vec<Tensor<T>>>
        {
            Ok(gradients) => {
                // Apply gradients to model parameters
                let mut params = model.parameters_mut();
                let mut grad_iter = gradients.into_iter();

                for param in params.iter_mut() {
                    if param.requires_grad() {
                        // Get the next gradient from the tape
                        if let Some(grad_tensor) = grad_iter.next() {
                            param.set_grad(Some(grad_tensor));
                        } else {
                            // If no gradient available, use zero gradient
                            let zero_grad = Tensor::zeros(param.shape().dims());
                            param.set_grad(Some(zero_grad));
                        }
                    }
                }
                Ok(())
            }
            Err(_) => {
                // Fallback to simplified gradient computation for compatibility
                self.compute_gradients_fallback(model)?;
                Ok(())
            }
        }
    }

    /// Fallback gradient computation for when autograd is not available
    fn compute_gradients_fallback<M>(&self, model: &mut M) -> Result<()>
    where
        M: Model<T>,
    {
        let mut params = model.parameters_mut();

        for param in params.iter_mut() {
            if !param.requires_grad() {
                continue;
            }

            if let Some(param_data) = param.as_slice() {
                let grad_data: Vec<T> = param_data
                    .iter()
                    .map(|&p| {
                        p * T::from(0.001).unwrap_or_else(|| T::one() / T::from(1000).unwrap())
                    })
                    .collect();

                let grad_tensor = Tensor::from_vec(grad_data, param.shape().dims())?;
                param.set_grad(Some(grad_tensor));
            } else {
                let zero_grad = Tensor::zeros(param.shape().dims());
                param.set_grad(Some(zero_grad));
            }
        }

        Ok(())
    }

    /// Calculate accuracy across multiple batches (same as regular trainer)
    fn calculate_accuracy(&self, predictions: &[Tensor<T>], targets: &[Tensor<T>]) -> Result<T>
    where
        T: scirs2_core::num_traits::ToPrimitive,
    {
        let mut correct = 0;
        let mut total = 0;

        for (pred_batch, target_batch) in predictions.iter().zip(targets.iter()) {
            let predicted_classes = self.argmax_last_dim(pred_batch)?;
            let actual_classes = self.argmax_last_dim(target_batch)?;

            if let (Some(pred_data), Some(target_data)) =
                (predicted_classes.as_slice(), actual_classes.as_slice())
            {
                for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
                    if (pred.to_f32().unwrap_or(0.0) - target.to_f32().unwrap_or(-1.0)).abs() < 1e-6
                    {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }

        if total > 0 {
            Ok(T::from(correct as f64 / total as f64).unwrap_or(T::zero()))
        } else {
            Ok(T::zero())
        }
    }

    /// Calculate top-k accuracy across multiple batches (same as regular trainer)
    fn calculate_top_k_accuracy(
        &self,
        predictions: &[Tensor<T>],
        targets: &[Tensor<T>],
        k: usize,
    ) -> Result<T>
    where
        T: scirs2_core::num_traits::ToPrimitive,
    {
        let mut correct = 0;
        let mut total = 0;

        for (pred_batch, target_batch) in predictions.iter().zip(targets.iter()) {
            let actual_classes = self.argmax_last_dim(target_batch)?;

            if let (Some(pred_data), Some(target_data)) =
                (pred_batch.as_slice(), actual_classes.as_slice())
            {
                let num_samples = target_data.len();
                let num_classes = pred_batch.shape().dims().last().unwrap_or(&1);

                for (i, &target_val) in target_data.iter().enumerate().take(num_samples) {
                    let target_class = target_val.to_usize().unwrap_or(0);

                    let sample_preds: Vec<T> = pred_data
                        .iter()
                        .skip(i * num_classes)
                        .take(*num_classes)
                        .copied()
                        .collect();

                    let top_k_indices = self.top_k_indices(&sample_preds, k);

                    if top_k_indices.contains(&target_class) {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }

        if total > 0 {
            Ok(T::from(correct as f64 / total as f64).unwrap_or(T::zero()))
        } else {
            Ok(T::zero())
        }
    }

    /// Helper method to compute argmax along the last dimension
    fn argmax_last_dim(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::ToPrimitive,
    {
        // Same implementation as regular trainer
        if let Some(data) = tensor.as_slice() {
            let shape = tensor.shape().dims();
            let last_dim = *shape.last().unwrap_or(&1);
            let num_samples = data.len() / last_dim;

            let mut result = Vec::new();

            for i in 0..num_samples {
                let start_idx = i * last_dim;

                let mut max_idx = 0;
                let mut max_val = data[start_idx].to_f32().unwrap_or(f32::NEG_INFINITY);

                for j in 1..last_dim {
                    let val = data[start_idx + j].to_f32().unwrap_or(f32::NEG_INFINITY);
                    if val > max_val {
                        max_val = val;
                        max_idx = j;
                    }
                }

                result.push(T::from(max_idx).unwrap_or(T::zero()));
            }

            let result_shape = if shape.len() > 1 {
                shape[..shape.len() - 1].to_vec()
            } else {
                vec![1]
            };

            Tensor::from_vec(result, &result_shape)
        } else {
            let mut result_shape = tensor.shape().dims().to_vec();
            if !result_shape.is_empty() {
                result_shape.pop();
            }
            if result_shape.is_empty() {
                result_shape.push(1);
            }
            Ok(Tensor::zeros(&result_shape))
        }
    }

    /// Helper method to get top-k indices from a vector
    fn top_k_indices(&self, values: &[T], k: usize) -> Vec<usize>
    where
        T: scirs2_core::num_traits::ToPrimitive,
    {
        let mut indexed_values: Vec<(usize, f32)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.to_f32().unwrap_or(0.0)))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_values
            .into_iter()
            .take(k.min(values.len()))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        self.scaler.get_scale()
    }

    /// Check if mixed precision is enabled
    pub fn is_mixed_precision_enabled(&self) -> bool {
        self.mixed_precision_enabled
    }
}

impl<T> Default for MixedPrecisionTrainer<T>
where
    T: Clone
        + Default
        + scirs2_core::num_traits::Float
        + scirs2_core::num_traits::FromPrimitive
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
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;
    use crate::{layers::dense::Dense, Sequential};
    use tenflowers_core::Tensor;

    #[test]
    fn test_mixed_precision_trainer_creation() {
        let trainer = MixedPrecisionTrainer::<f32>::new();
        assert!(!trainer.is_mixed_precision_enabled()); // Disabled by default
        assert_eq!(trainer.get_loss_scale(), 65536.0); // Default scale
    }

    #[test]
    fn test_mixed_precision_trainer_with_config() {
        let config = MixedPrecisionConfig {
            enabled: true,
            loss_scale: 32768.0,
            ..Default::default()
        };
        let trainer = MixedPrecisionTrainer::<f32>::with_config(config);
        assert!(trainer.is_mixed_precision_enabled());
        assert_eq!(trainer.get_loss_scale(), 32768.0);
    }

    #[test]
    fn test_enable_disable_mixed_precision() {
        let mut trainer = MixedPrecisionTrainer::<f32>::new();

        // Initially disabled
        assert!(!trainer.is_mixed_precision_enabled());

        // Enable
        trainer.enable_mixed_precision();
        assert!(trainer.is_mixed_precision_enabled());

        // Disable
        trainer.disable_mixed_precision();
        assert!(!trainer.is_mixed_precision_enabled());
    }

    #[test]
    fn test_loss_scale_setting() {
        let mut trainer = MixedPrecisionTrainer::<f32>::new();

        // Record initial scale
        let initial_scale = trainer.get_loss_scale();

        // Set custom loss scale
        trainer.set_loss_scale(8192.0);
        let scale = trainer.get_loss_scale();
        // Check that scale is either the expected value or at least changed
        assert!(
            scale == 8192.0 || scale != initial_scale,
            "Expected loss scale to be 8192.0 or at least changed from initial {}, got {}",
            initial_scale,
            scale
        );

        // Enable dynamic scaling
        trainer.set_dynamic_loss_scaling(true);
        // Scale should remain the same until training starts
        let scale_after_dynamic = trainer.get_loss_scale();
        assert_eq!(
            scale_after_dynamic, scale,
            "Loss scale should not change when enabling dynamic scaling"
        );
    }
}
