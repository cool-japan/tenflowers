use crate::model::Model;
/// Advanced training utilities with gradient accumulation support
///
/// This module provides high-level training utilities that make it easy
/// to train large models using gradient accumulation with various strategies.
use crate::optimizers::{
    AccumulationStrategy, EnhancedOptimizerWithAccumulation, MemoryConfig, Optimizer,
};
use std::time::Instant;
use tenflowers_core::{Result, Tensor};

/// Configuration for gradient accumulation training
#[derive(Debug, Clone)]
pub struct AccumulationTrainingConfig {
    /// Number of micro-batches to accumulate before optimizer step
    pub accumulation_steps: usize,
    /// Accumulation strategy to use
    pub strategy: AccumulationStrategy,
    /// Memory configuration for accumulation
    pub memory_config: MemoryConfig,
    /// Whether to enable progress tracking
    pub track_progress: bool,
    /// Whether to enable memory monitoring
    pub track_memory: bool,
    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_gradient_norm: Option<f32>,
    /// Whether to scale loss by accumulation steps (recommended for most cases)
    pub scale_loss: bool,
}

impl Default for AccumulationTrainingConfig {
    fn default() -> Self {
        AccumulationTrainingConfig {
            accumulation_steps: 4,
            strategy: AccumulationStrategy::Average,
            memory_config: MemoryConfig::default(),
            track_progress: true,
            track_memory: false,
            max_gradient_norm: None,
            scale_loss: true,
        }
    }
}

/// Statistics collected during gradient accumulation training
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub optimizer_steps: usize,
    pub average_loss: f32,
    pub gradient_norm: Option<f32>,
    pub memory_usage_mb: Option<f32>,
    pub step_time_ms: f32,
    pub accumulation_efficiency: f32, // ratio of optimizer steps to total steps
}

/// High-level trainer for models with gradient accumulation
pub struct GradientAccumulationTrainer<T, O> {
    optimizer: EnhancedOptimizerWithAccumulation<T, O>,
    config: AccumulationTrainingConfig,
    stats: TrainingStats,
    loss_accumulator: Vec<f32>,
    step_timer: Option<Instant>,
}

impl<T, O> GradientAccumulationTrainer<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod,
{
    /// Create a new gradient accumulation trainer
    pub fn new(optimizer: O, config: AccumulationTrainingConfig) -> Self {
        let enhanced_optimizer = EnhancedOptimizerWithAccumulation::new(
            optimizer,
            config.strategy,
            config.accumulation_steps,
            config.memory_config.clone(),
        );

        Self {
            optimizer: enhanced_optimizer,
            config,
            stats: TrainingStats {
                total_steps: 0,
                optimizer_steps: 0,
                average_loss: 0.0,
                gradient_norm: None,
                memory_usage_mb: None,
                step_time_ms: 0.0,
                accumulation_efficiency: 0.0,
            },
            loss_accumulator: Vec::new(),
            step_timer: None,
        }
    }

    /// Perform one training step with gradient accumulation
    /// Returns true if optimizer step was performed
    pub fn step<M: Model<T>>(&mut self, model: &mut M, loss: &Tensor<T>) -> Result<bool> {
        if self.step_timer.is_none() {
            self.step_timer = Some(Instant::now());
        }

        // Scale loss if configured
        let scaled_loss = if self.config.scale_loss {
            let scale = T::from_usize(self.config.accumulation_steps).unwrap();
            loss.div(&Tensor::from_scalar(scale))?
        } else {
            loss.clone()
        };

        // Backward pass to compute gradients
        // Note: In a real implementation, this would integrate with the autograd system
        // For now, we assume gradients are already computed

        // Apply gradient clipping if configured
        if let Some(max_norm) = self.config.max_gradient_norm {
            self.clip_gradients(model, max_norm)?;
        }

        // Accumulate gradients
        let optimizer_stepped = self.optimizer.step_with_accumulation(model)?;

        // Track statistics
        self.stats.total_steps += 1;
        if optimizer_stepped {
            self.stats.optimizer_steps += 1;
        }

        // Accumulate loss for averaging
        if let Ok(loss_scalar) = self.extract_scalar_loss(&scaled_loss) {
            self.loss_accumulator.push(loss_scalar);
        }

        // Update memory stats if tracking is enabled
        if self.config.track_memory {
            let memory_stats = self.optimizer.memory_stats();
            self.stats.memory_usage_mb = Some(memory_stats.estimated_memory_mb);
        }

        // Update step timing
        if let Some(start_time) = self.step_timer {
            self.stats.step_time_ms = start_time.elapsed().as_millis() as f32;
        }

        // Calculate efficiency
        self.stats.accumulation_efficiency = if self.stats.total_steps > 0 {
            self.stats.optimizer_steps as f32 / self.stats.total_steps as f32
        } else {
            0.0
        };

        // Update average loss
        if !self.loss_accumulator.is_empty() {
            self.stats.average_loss =
                self.loss_accumulator.iter().sum::<f32>() / self.loss_accumulator.len() as f32;
        }

        // Reset timer for next step
        self.step_timer = Some(Instant::now());

        Ok(optimizer_stepped)
    }

    /// Manually trigger optimizer step if enough gradients are accumulated
    pub fn apply_accumulated_gradients<M: Model<T>>(&mut self, model: &mut M) -> Result<bool> {
        if self.optimizer.progress().is_ready {
            self.optimizer.apply_gradients(model)?;
            self.stats.optimizer_steps += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Zero gradients
    pub fn zero_grad<M: Model<T>>(&self, model: &mut M) {
        self.optimizer.zero_grad(model);
    }

    /// Get current training statistics
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// Get accumulation progress
    pub fn accumulation_progress(&self) -> crate::optimizers::AccumulationProgress {
        self.optimizer.progress()
    }

    /// Reset statistics (useful for new epochs)
    pub fn reset_stats(&mut self) {
        self.stats = TrainingStats {
            total_steps: 0,
            optimizer_steps: 0,
            average_loss: 0.0,
            gradient_norm: None,
            memory_usage_mb: None,
            step_time_ms: 0.0,
            accumulation_efficiency: 0.0,
        };
        self.loss_accumulator.clear();
    }

    /// Change accumulation strategy dynamically
    pub fn set_accumulation_strategy(&mut self, strategy: AccumulationStrategy) {
        self.optimizer.set_strategy(strategy);
        self.config.strategy = strategy;
    }

    /// Set custom weights for weighted accumulation
    pub fn set_step_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        self.optimizer.set_step_weights(weights)
    }

    /// Get reference to underlying optimizer
    pub fn optimizer(&self) -> &EnhancedOptimizerWithAccumulation<T, O> {
        &self.optimizer
    }

    /// Get mutable reference to underlying optimizer
    pub fn optimizer_mut(&mut self) -> &mut EnhancedOptimizerWithAccumulation<T, O> {
        &mut self.optimizer
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Get learning rate
    pub fn get_learning_rate(&self) -> f32 {
        self.optimizer.get_learning_rate()
    }

    /// Simple gradient clipping by norm
    fn clip_gradients<M: Model<T>>(&mut self, model: &mut M, max_norm: f32) -> Result<()> {
        let mut total_norm = 0.0f32;

        // Calculate total gradient norm
        for param in model.parameters() {
            if let Some(grad) = param.grad() {
                // Simple norm calculation - in practice, you'd want a more efficient implementation
                if let Ok(grad_norm) = self.calculate_tensor_norm(grad) {
                    total_norm += grad_norm * grad_norm;
                }
            }
        }

        total_norm = total_norm.sqrt();
        self.stats.gradient_norm = Some(total_norm);

        // Clip gradients if necessary
        if total_norm > max_norm {
            let clip_coef = max_norm / total_norm;
            let clip_tensor = Tensor::from_scalar(T::from_f32(clip_coef).unwrap());

            for param in model.parameters_mut() {
                if let Some(grad) = param.grad() {
                    let clipped_grad = grad.mul(&clip_tensor)?;
                    param.set_grad(Some(clipped_grad));
                }
            }
        }

        Ok(())
    }

    /// Calculate tensor norm (proper implementation using tensor operations)
    fn calculate_tensor_norm(&self, tensor: &Tensor<T>) -> Result<f32> {
        // Calculate L2 norm using tensor operations
        let two = Tensor::from_scalar(T::from(2.0).unwrap_or(T::from(2.0).unwrap()));
        let squared = tenflowers_core::ops::pow(tensor, &two)?;
        let sum = tenflowers_core::ops::sum(&squared, None, false)?;
        let norm_tensor = tenflowers_core::ops::sqrt(&sum)?;

        // Extract scalar value - simplified for generic T
        let data = norm_tensor.to_vec()?;
        if let Some(first) = data.first() {
            // Use Float trait's to_f32() method for proper conversion
            Ok(first.to_f32().unwrap_or(0.0))
        } else {
            Ok(0.0)
        }
    }

    /// Extract scalar loss value for statistics
    fn extract_scalar_loss(&self, loss: &Tensor<T>) -> Result<f32> {
        // Extract scalar value from loss tensor
        let data = if loss.shape().dims() == [1] || loss.shape().dims().is_empty() {
            loss.to_vec()?
        } else {
            // For multi-dimensional tensors, compute mean
            let mean_loss = tenflowers_core::ops::mean(loss, None, false)?;
            mean_loss.to_vec()?
        };

        if let Some(first) = data.first() {
            // Use Float trait's to_f32() method for proper conversion
            Ok(first.to_f32().unwrap_or(0.0))
        } else {
            Ok(0.0)
        }
    }
}

/// Convenience function to create a trainer with commonly used configurations
pub fn create_trainer_for_large_model<T, O>(
    optimizer: O,
    effective_batch_size: usize,
    micro_batch_size: usize,
    use_cpu_offload: bool,
) -> GradientAccumulationTrainer<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod,
{
    let accumulation_steps = effective_batch_size / micro_batch_size;

    let config = AccumulationTrainingConfig {
        accumulation_steps,
        strategy: AccumulationStrategy::Average,
        memory_config: MemoryConfig {
            use_cpu_offload,
            use_compression: false,
            compression_ratio: 0.5,
            max_memory_mb: None,
        },
        track_progress: true,
        track_memory: true,
        max_gradient_norm: Some(1.0), // Common default
        scale_loss: true,
    };

    GradientAccumulationTrainer::new(optimizer, config)
}

/// Convenience function for memory-efficient training
pub fn create_memory_efficient_trainer<T, O>(
    optimizer: O,
    accumulation_steps: usize,
    max_memory_mb: usize,
) -> GradientAccumulationTrainer<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + scirs2_core::num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + PartialOrd
        + bytemuck::Pod,
{
    let config = AccumulationTrainingConfig {
        accumulation_steps,
        strategy: AccumulationStrategy::Average,
        memory_config: MemoryConfig {
            use_cpu_offload: true,
            use_compression: true,
            compression_ratio: 0.25,
            max_memory_mb: Some(max_memory_mb),
        },
        track_progress: true,
        track_memory: true,
        max_gradient_norm: Some(1.0),
        scale_loss: true,
    };

    GradientAccumulationTrainer::new(optimizer, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;

    #[test]
    fn test_trainer_creation() {
        let sgd = SGD::<f32>::new(0.01);
        let config = AccumulationTrainingConfig::default();
        let trainer = GradientAccumulationTrainer::new(sgd, config);

        assert_eq!(trainer.stats().total_steps, 0);
        assert_eq!(trainer.stats().optimizer_steps, 0);
        assert_eq!(trainer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_convenience_functions() {
        let sgd = SGD::<f32>::new(0.01);

        // Test large model trainer
        let trainer = create_trainer_for_large_model(sgd, 64, 16, true);
        assert_eq!(trainer.accumulation_progress().total_steps, 4); // 64/16

        // Test memory efficient trainer
        let sgd2 = SGD::<f32>::new(0.01);
        let trainer2 = create_memory_efficient_trainer(sgd2, 8, 512);
        assert_eq!(trainer2.accumulation_progress().total_steps, 8);
    }

    #[test]
    fn test_config_validation() {
        let config = AccumulationTrainingConfig {
            accumulation_steps: 0,
            ..Default::default()
        };

        // In a real implementation, we might want to validate configs
        assert_eq!(config.accumulation_steps, 0);
    }
}
