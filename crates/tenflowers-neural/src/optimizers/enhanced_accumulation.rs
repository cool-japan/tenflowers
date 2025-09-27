use crate::model::Model;
/// Enhanced gradient accumulation system for large model training
///
/// This module provides sophisticated gradient accumulation strategies
/// for training large models that don't fit in memory or when simulating
/// larger batch sizes with limited GPU memory.
use crate::optimizers::Optimizer;
use std::collections::HashMap;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

/// Strategy for accumulating gradients
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccumulationStrategy {
    /// Simple averaging: grad_avg = sum(gradients) / num_steps
    Average,
    /// Sum gradients without scaling: grad_sum = sum(gradients)  
    Sum,
    /// Exponential moving average: grad_ema = alpha * grad_new + (1-alpha) * grad_old
    ExponentialMovingAverage { alpha: f32 },
    /// Weighted accumulation with custom weights per step
    Weighted,
}

/// Memory optimization settings for gradient accumulation
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Whether to use CPU offloading for accumulated gradients
    pub use_cpu_offload: bool,
    /// Whether to use gradient compression
    pub use_compression: bool,
    /// Compression ratio (if compression is enabled)
    pub compression_ratio: f32,
    /// Maximum memory usage for gradient storage (in MB)
    pub max_memory_mb: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            use_cpu_offload: false,
            use_compression: false,
            compression_ratio: 0.5,
            max_memory_mb: None,
        }
    }
}

/// Enhanced gradient accumulator with multiple strategies and memory optimization
pub struct EnhancedGradientAccumulator<T> {
    /// Accumulated gradients for each parameter (identified by a hash)
    accumulated_gradients: HashMap<String, Tensor<T>>,
    /// Accumulation strategy
    strategy: AccumulationStrategy,
    /// Number of accumulation steps
    accumulation_steps: usize,
    /// Current step count
    current_step: usize,
    /// Weights for weighted accumulation
    step_weights: Vec<f32>,
    /// Memory configuration
    memory_config: MemoryConfig,
    /// Whether to clear gradients after accumulation
    clear_after_step: bool,
    _phantom: PhantomData<T>,
}

impl<T> EnhancedGradientAccumulator<T>
where
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod,
{
    /// Create a new enhanced gradient accumulator
    pub fn new(
        strategy: AccumulationStrategy,
        accumulation_steps: usize,
        memory_config: MemoryConfig,
    ) -> Self {
        Self {
            accumulated_gradients: HashMap::new(),
            strategy,
            accumulation_steps,
            current_step: 0,
            step_weights: vec![1.0; accumulation_steps],
            memory_config,
            clear_after_step: true,
            _phantom: PhantomData,
        }
    }

    /// Set custom weights for weighted accumulation
    pub fn set_step_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        if weights.len() != self.accumulation_steps {
            return Err(TensorError::invalid_argument(format!(
                "Weight vector length {} must match accumulation steps {}",
                weights.len(),
                self.accumulation_steps
            )));
        }
        self.step_weights = weights;
        Ok(())
    }

    /// Accumulate gradients from model parameters
    pub fn accumulate(&mut self, model: &dyn Model<T>) -> Result<()> {
        let weight = match self.strategy {
            AccumulationStrategy::Weighted => self
                .step_weights
                .get(self.current_step)
                .copied()
                .unwrap_or(1.0),
            _ => 1.0,
        };

        for (param_id, param) in model.parameters().iter().enumerate() {
            if let Some(grad) = param.grad() {
                let param_key = format!("param_{param_id}");

                if let Some(accumulated) = self.accumulated_gradients.get_mut(&param_key) {
                    // Add to existing accumulation
                    let weighted_grad = if weight != 1.0 {
                        grad.mul(&Tensor::from_scalar(T::from_f32(weight).unwrap()))?
                    } else {
                        grad.clone()
                    };

                    match self.strategy {
                        AccumulationStrategy::ExponentialMovingAverage { alpha } => {
                            let alpha_t = T::from_f32(alpha).unwrap();
                            let one_minus_alpha = T::from_f32(1.0 - alpha).unwrap();

                            let new_part = weighted_grad.mul(&Tensor::from_scalar(alpha_t))?;
                            let old_part =
                                accumulated.mul(&Tensor::from_scalar(one_minus_alpha))?;
                            *accumulated = new_part.add(&old_part)?;
                        }
                        _ => {
                            *accumulated = accumulated.add(&weighted_grad)?;
                        }
                    }
                } else {
                    // First accumulation for this parameter
                    let weighted_grad = if weight != 1.0 {
                        grad.mul(&Tensor::from_scalar(T::from_f32(weight).unwrap()))?
                    } else {
                        grad.clone()
                    };

                    // Handle CPU offloading if enabled
                    let grad_to_store =
                        if self.memory_config.use_cpu_offload && !grad.device().is_cpu() {
                            weighted_grad.to_cpu()?
                        } else {
                            weighted_grad
                        };

                    self.accumulated_gradients.insert(param_key, grad_to_store);
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Check if accumulation is complete and ready for optimizer step
    pub fn is_ready(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get the accumulated gradients and apply them to model parameters
    pub fn apply_accumulated_gradients(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        if !self.is_ready() {
            return Ok(()); // Not ready yet
        }

        // Apply scaling based on strategy
        let scale_factor = match self.strategy {
            AccumulationStrategy::Average => T::from_usize(self.accumulation_steps).unwrap(),
            AccumulationStrategy::Sum => T::one(),
            AccumulationStrategy::ExponentialMovingAverage { .. } => T::one(),
            AccumulationStrategy::Weighted => {
                // For weighted, we assume weights are already normalized
                T::one()
            }
        };

        // Apply accumulated gradients to parameters
        for (param_id, param) in model.parameters_mut().iter_mut().enumerate() {
            let param_key = format!("param_{param_id}");

            if let Some(accumulated_grad) = self.accumulated_gradients.get(&param_key) {
                // Scale the accumulated gradient
                let final_grad = if scale_factor != T::one() {
                    accumulated_grad.div(&Tensor::from_scalar(scale_factor))?
                } else {
                    accumulated_grad.clone()
                };

                // Move back to original device if it was offloaded
                let grad_on_device = if self.memory_config.use_cpu_offload
                    && !param.device().is_cpu()
                    && final_grad.device().is_cpu()
                {
                    final_grad.to_device(*param.device())?
                } else {
                    final_grad
                };

                param.set_grad(Some(grad_on_device));
            }
        }

        // Clear accumulated gradients if configured
        if self.clear_after_step {
            self.clear();
        }

        Ok(())
    }

    /// Clear all accumulated gradients and reset step counter
    pub fn clear(&mut self) {
        self.accumulated_gradients.clear();
        self.current_step = 0;
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let total_params = self.accumulated_gradients.len();
        let estimated_memory_bytes = self
            .accumulated_gradients
            .values()
            .map(|tensor| tensor.shape().size() * std::mem::size_of::<T>())
            .sum::<usize>();

        MemoryStats {
            total_parameters: total_params,
            estimated_memory_bytes,
            estimated_memory_mb: estimated_memory_bytes as f32 / (1024.0 * 1024.0),
            current_step: self.current_step,
            accumulation_steps: self.accumulation_steps,
            completion_ratio: self.current_step as f32 / self.accumulation_steps as f32,
        }
    }

    /// Set the accumulation strategy
    pub fn set_strategy(&mut self, strategy: AccumulationStrategy) {
        self.strategy = strategy;
        self.clear(); // Clear existing accumulation when changing strategy
    }

    /// Get current accumulation progress
    pub fn progress(&self) -> AccumulationProgress {
        AccumulationProgress {
            current_step: self.current_step,
            total_steps: self.accumulation_steps,
            is_ready: self.is_ready(),
            strategy: self.strategy,
        }
    }
}

/// Memory usage statistics for gradient accumulation
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_parameters: usize,
    pub estimated_memory_bytes: usize,
    pub estimated_memory_mb: f32,
    pub current_step: usize,
    pub accumulation_steps: usize,
    pub completion_ratio: f32,
}

/// Progress information for gradient accumulation
#[derive(Debug, Clone)]
pub struct AccumulationProgress {
    pub current_step: usize,
    pub total_steps: usize,
    pub is_ready: bool,
    pub strategy: AccumulationStrategy,
}

/// Enhanced optimizer with sophisticated gradient accumulation
pub struct EnhancedOptimizerWithAccumulation<T, O> {
    inner_optimizer: O,
    accumulator: EnhancedGradientAccumulator<T>,
    auto_step: bool,
    _phantom: PhantomData<T>,
}

impl<T, O> EnhancedOptimizerWithAccumulation<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod,
{
    /// Create a new enhanced optimizer with accumulation
    pub fn new(
        optimizer: O,
        strategy: AccumulationStrategy,
        accumulation_steps: usize,
        memory_config: MemoryConfig,
    ) -> Self {
        Self {
            inner_optimizer: optimizer,
            accumulator: EnhancedGradientAccumulator::new(
                strategy,
                accumulation_steps,
                memory_config,
            ),
            auto_step: true,
            _phantom: PhantomData,
        }
    }

    /// Perform one accumulation step
    pub fn accumulate(&mut self, model: &dyn Model<T>) -> Result<bool> {
        self.accumulator.accumulate(model)?;
        Ok(self.accumulator.is_ready())
    }

    /// Apply accumulated gradients using the inner optimizer
    pub fn apply_gradients(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        if self.accumulator.is_ready() {
            self.accumulator.apply_accumulated_gradients(model)?;
            self.inner_optimizer.step(model)?;
        }
        Ok(())
    }

    /// Combined accumulate and apply step
    pub fn step_with_accumulation(&mut self, model: &mut dyn Model<T>) -> Result<bool> {
        let ready = self.accumulate(model)?;
        if ready && self.auto_step {
            self.apply_gradients(model)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get accumulation progress
    pub fn progress(&self) -> AccumulationProgress {
        self.accumulator.progress()
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.accumulator.memory_stats()
    }

    /// Set custom weights for weighted accumulation
    pub fn set_step_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        self.accumulator.set_step_weights(weights)
    }

    /// Change accumulation strategy
    pub fn set_strategy(&mut self, strategy: AccumulationStrategy) {
        self.accumulator.set_strategy(strategy);
    }

    /// Enable/disable automatic optimizer stepping when accumulation is ready
    pub fn set_auto_step(&mut self, auto_step: bool) {
        self.auto_step = auto_step;
    }

    /// Get reference to inner optimizer
    pub fn inner_optimizer(&self) -> &O {
        &self.inner_optimizer
    }

    /// Get mutable reference to inner optimizer
    pub fn inner_optimizer_mut(&mut self) -> &mut O {
        &mut self.inner_optimizer
    }
}

impl<T, O> Optimizer<T> for EnhancedOptimizerWithAccumulation<T, O>
where
    O: Optimizer<T>,
    T: Clone
        + Default
        + num_traits::Zero
        + num_traits::One
        + num_traits::Float
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod,
{
    fn step(&mut self, model: &mut dyn Model<T>) -> Result<()> {
        self.step_with_accumulation(model)?;
        Ok(())
    }

    fn zero_grad(&self, model: &mut dyn Model<T>) {
        self.inner_optimizer.zero_grad(model);
    }

    fn set_learning_rate(&mut self, learning_rate: f32) {
        self.inner_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> f32 {
        self.inner_optimizer.get_learning_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;

    #[test]
    fn test_accumulation_strategies() {
        let memory_config = MemoryConfig::default();

        // Test different strategies
        let acc_avg = EnhancedGradientAccumulator::<f32>::new(
            AccumulationStrategy::Average,
            4,
            memory_config.clone(),
        );
        assert_eq!(acc_avg.progress().strategy, AccumulationStrategy::Average);

        let acc_sum = EnhancedGradientAccumulator::<f32>::new(
            AccumulationStrategy::Sum,
            4,
            memory_config.clone(),
        );
        assert_eq!(acc_sum.progress().strategy, AccumulationStrategy::Sum);

        let acc_ema = EnhancedGradientAccumulator::<f32>::new(
            AccumulationStrategy::ExponentialMovingAverage { alpha: 0.9 },
            4,
            memory_config,
        );
        assert_eq!(
            acc_ema.progress().strategy,
            AccumulationStrategy::ExponentialMovingAverage { alpha: 0.9 }
        );
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig {
            use_cpu_offload: true,
            use_compression: true,
            compression_ratio: 0.25,
            max_memory_mb: Some(1024),
        };

        let accumulator =
            EnhancedGradientAccumulator::<f32>::new(AccumulationStrategy::Average, 4, config);

        let stats = accumulator.memory_stats();
        assert_eq!(stats.accumulation_steps, 4);
        assert_eq!(stats.current_step, 0);
    }

    #[test]
    fn test_weighted_accumulation() {
        let memory_config = MemoryConfig::default();
        let mut accumulator = EnhancedGradientAccumulator::<f32>::new(
            AccumulationStrategy::Weighted,
            3,
            memory_config,
        );

        // Set custom weights
        let weights = vec![0.1, 0.3, 0.6];
        accumulator.set_step_weights(weights).unwrap();

        // Test invalid weight vector
        let invalid_weights = vec![0.1, 0.3]; // Wrong length
        assert!(accumulator.set_step_weights(invalid_weights).is_err());
    }

    #[test]
    fn test_enhanced_optimizer() {
        let sgd = SGD::<f32>::new(0.01);
        let memory_config = MemoryConfig::default();

        let optimizer = EnhancedOptimizerWithAccumulation::new(
            sgd,
            AccumulationStrategy::Average,
            4,
            memory_config,
        );

        assert_eq!(optimizer.get_learning_rate(), 0.01);
        assert!(!optimizer.progress().is_ready);
        assert_eq!(optimizer.progress().current_step, 0);
    }
}
