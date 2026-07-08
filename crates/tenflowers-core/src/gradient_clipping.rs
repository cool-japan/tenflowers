//! Advanced Gradient Clipping with Adaptive Scaling
//!
//! This module provides sophisticated gradient clipping techniques essential for training
//! stability in modern deep learning, particularly with large language models, RNNs, and
//! transformers. It implements both global norm clipping and adaptive scaling strategies
//! that automatically adjust based on training dynamics.
//!
//! # Features
//! - **Global Gradient Norm Clipping**: Scales all gradients by the same factor to maintain
//!   relative magnitudes while preventing exploding gradients
//! - **Adaptive Clipping**: Dynamically adjusts clipping thresholds based on gradient
//!   statistics and training progress
//! - **Per-Parameter Clipping**: Fine-grained control for different parameter groups
//! - **Gradient Statistics Tracking**: Monitors gradient norms for training diagnostics
//! - **Warmup and Decay**: Gradually adjusts clipping behavior during training phases

use crate::{Result, Tensor};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for gradient clipping behavior
#[derive(Debug, Clone)]
pub struct GradientClippingConfig {
    /// Maximum allowed gradient norm (global clipping threshold)
    pub max_norm: f64,
    /// Type of norm to use for clipping (1, 2, or infinity)
    pub norm_type: NormType,
    /// Enable adaptive threshold adjustment based on gradient statistics
    pub adaptive_scaling: bool,
    /// Momentum factor for adaptive threshold updates (0.0 - 1.0)
    pub adaptive_momentum: f64,
    /// Minimum allowed clipping threshold (prevents overly aggressive clipping)
    pub min_threshold: f64,
    /// Maximum allowed clipping threshold (prevents disabling clipping)
    pub max_threshold: f64,
    /// Warmup steps for gradually enabling clipping
    pub warmup_steps: usize,
    /// Enable per-parameter group clipping with different thresholds
    pub per_parameter_clipping: bool,
}

impl Default for GradientClippingConfig {
    fn default() -> Self {
        Self {
            max_norm: 1.0,
            norm_type: NormType::L2,
            adaptive_scaling: false,
            adaptive_momentum: 0.95,
            min_threshold: 0.1,
            max_threshold: 10.0,
            warmup_steps: 0,
            per_parameter_clipping: false,
        }
    }
}

/// Type of norm to use for gradient clipping
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// L1 norm (sum of absolute values)
    L1,
    /// L2 norm (Euclidean norm, most common)
    L2,
    /// Infinity norm (maximum absolute value)
    Infinity,
}

/// Statistics tracking for gradient clipping
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Current global gradient norm
    pub current_norm: f64,
    /// Exponential moving average of gradient norms
    pub avg_norm: f64,
    /// Standard deviation of recent gradient norms
    pub std_norm: f64,
    /// Number of times gradients were clipped
    pub clip_count: usize,
    /// Total number of gradient updates processed
    pub total_updates: usize,
    /// Current adaptive threshold (if adaptive scaling is enabled)
    pub adaptive_threshold: f64,
    /// History of recent gradient norms (for statistics calculation)
    pub norm_history: Vec<f64>,
}

impl Default for GradientStatistics {
    fn default() -> Self {
        Self {
            current_norm: 0.0,
            avg_norm: 0.0,
            std_norm: 0.0,
            clip_count: 0,
            total_updates: 0,
            adaptive_threshold: 1.0,
            norm_history: Vec::with_capacity(100), // Keep last 100 norms
        }
    }
}

/// Advanced gradient clipping system with adaptive scaling
pub struct GradientClipper<T> {
    config: GradientClippingConfig,
    statistics: GradientStatistics,
    parameter_groups: HashMap<String, f64>, // Group name -> threshold
    step_count: usize,
    _phantom: PhantomData<T>,
}

impl<T> GradientClipper<T>
where
    T: Float
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + Default
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new gradient clipper with the specified configuration
    pub fn new(config: GradientClippingConfig) -> Self {
        Self {
            config,
            statistics: GradientStatistics::default(),
            parameter_groups: HashMap::new(),
            step_count: 0,
            _phantom: PhantomData,
        }
    }

    /// Create a gradient clipper with default settings for stable training
    pub fn default_stable() -> Self {
        Self::new(GradientClippingConfig {
            max_norm: 1.0,
            norm_type: NormType::L2,
            adaptive_scaling: false,
            ..Default::default()
        })
    }

    /// Create a gradient clipper with adaptive scaling for dynamic adjustment
    pub fn default_adaptive() -> Self {
        Self::new(GradientClippingConfig {
            max_norm: 1.0,
            norm_type: NormType::L2,
            adaptive_scaling: true,
            adaptive_momentum: 0.95,
            min_threshold: 0.1,
            max_threshold: 5.0,
            ..Default::default()
        })
    }

    /// Add a parameter group with its own clipping threshold
    pub fn add_parameter_group(&mut self, group_name: String, threshold: f64) {
        self.parameter_groups.insert(group_name, threshold);
    }

    /// Clip gradients using global norm clipping
    ///
    /// This is the main method for applying gradient clipping. It computes the global
    /// gradient norm across all tensors and scales them proportionally if needed.
    pub fn clip_gradients(&mut self, gradients: &mut [Tensor<T>]) -> Result<f64> {
        if gradients.is_empty() {
            return Ok(0.0);
        }

        self.step_count += 1;

        // Compute global gradient norm
        let global_norm = self.compute_global_norm(gradients)?;

        // Update statistics
        self.update_statistics(global_norm);

        // Determine effective clipping threshold
        let effective_threshold = self.get_effective_threshold();

        // Apply warmup if configured
        let warmed_threshold = if self.step_count <= self.config.warmup_steps {
            let warmup_factor = self.step_count as f64 / self.config.warmup_steps as f64;
            effective_threshold * warmup_factor + self.config.max_norm * (1.0 - warmup_factor)
        } else {
            effective_threshold
        };

        // Apply clipping if necessary
        if global_norm > warmed_threshold {
            let scale_factor = warmed_threshold / global_norm;
            self.scale_gradients(
                gradients,
                T::from_f64(scale_factor).unwrap_or_else(|| T::one()),
            )?;
            self.statistics.clip_count += 1;
        }

        Ok(global_norm)
    }

    /// Clip gradients for a specific parameter group
    pub fn clip_parameter_group(
        &mut self,
        group_name: &str,
        gradients: &mut [Tensor<T>],
    ) -> Result<f64> {
        let threshold = self
            .parameter_groups
            .get(group_name)
            .copied()
            .unwrap_or(self.config.max_norm);

        let global_norm = self.compute_global_norm(gradients)?;

        if global_norm > threshold {
            let scale_factor = threshold / global_norm;
            self.scale_gradients(
                gradients,
                T::from_f64(scale_factor).unwrap_or_else(|| T::one()),
            )?;
        }

        Ok(global_norm)
    }

    /// Compute the global norm of all gradients
    fn compute_global_norm(&self, gradients: &[Tensor<T>]) -> Result<f64> {
        match self.config.norm_type {
            NormType::L1 => {
                let mut total_norm = 0.0;
                for grad in gradients {
                    total_norm += self.compute_tensor_l1_norm(grad)?;
                }
                Ok(total_norm)
            }
            NormType::L2 => {
                let mut total_squared_norm = 0.0;
                for grad in gradients {
                    let tensor_norm = self.compute_tensor_l2_norm(grad)?;
                    total_squared_norm += tensor_norm * tensor_norm;
                }
                Ok(total_squared_norm.sqrt())
            }
            NormType::Infinity => {
                let mut max_norm = 0.0;
                for grad in gradients {
                    let tensor_max = self.compute_tensor_inf_norm(grad)?;
                    max_norm = max_norm.max(tensor_max);
                }
                Ok(max_norm)
            }
        }
    }

    /// Compute L1 norm of a tensor
    fn compute_tensor_l1_norm(&self, tensor: &Tensor<T>) -> Result<f64> {
        // This is a simplified implementation - in practice, you'd use the tensor's
        // actual data to compute the norm
        match &tensor.storage {
            crate::tensor::TensorStorage::Cpu(array) => {
                let sum: f64 = array.iter().map(|&x| x.abs().to_f64().unwrap_or(0.0)).sum();
                Ok(sum)
            }
            #[cfg(feature = "gpu")]
            crate::tensor::TensorStorage::Gpu(_) => {
                let cpu_tensor = tensor.to_cpu()?;
                self.compute_tensor_l1_norm(&cpu_tensor)
            }
        }
    }

    /// Compute L2 norm of a tensor
    fn compute_tensor_l2_norm(&self, tensor: &Tensor<T>) -> Result<f64> {
        match &tensor.storage {
            crate::tensor::TensorStorage::Cpu(array) => {
                let sum_squares: f64 = array
                    .iter()
                    .map(|&x| {
                        let val = x.to_f64().unwrap_or(0.0);
                        val * val
                    })
                    .sum();
                Ok(sum_squares.sqrt())
            }
            #[cfg(feature = "gpu")]
            crate::tensor::TensorStorage::Gpu(_) => {
                let cpu_tensor = tensor.to_cpu()?;
                self.compute_tensor_l2_norm(&cpu_tensor)
            }
        }
    }

    /// Compute infinity norm of a tensor
    fn compute_tensor_inf_norm(&self, tensor: &Tensor<T>) -> Result<f64> {
        match &tensor.storage {
            crate::tensor::TensorStorage::Cpu(array) => {
                let max_val = array
                    .iter()
                    .map(|&x| x.abs().to_f64().unwrap_or(0.0))
                    .fold(0.0, f64::max);
                Ok(max_val)
            }
            #[cfg(feature = "gpu")]
            crate::tensor::TensorStorage::Gpu(_) => {
                let cpu_tensor = tensor.to_cpu()?;
                self.compute_tensor_inf_norm(&cpu_tensor)
            }
        }
    }

    /// Scale all gradients by a constant factor
    fn scale_gradients(&self, gradients: &mut [Tensor<T>], scale_factor: T) -> Result<()> {
        for grad in gradients.iter_mut() {
            *grad = grad.mul_scalar(scale_factor)?;
        }
        Ok(())
    }

    /// Update gradient statistics for adaptive scaling
    fn update_statistics(&mut self, global_norm: f64) {
        self.statistics.current_norm = global_norm;
        self.statistics.total_updates += 1;

        // Update exponential moving average
        if self.statistics.total_updates == 1 {
            self.statistics.avg_norm = global_norm;
        } else {
            let momentum = self.config.adaptive_momentum;
            self.statistics.avg_norm =
                momentum * self.statistics.avg_norm + (1.0 - momentum) * global_norm;
        }

        // Update norm history for standard deviation calculation
        self.statistics.norm_history.push(global_norm);
        if self.statistics.norm_history.len() > 100 {
            self.statistics.norm_history.remove(0);
        }

        // Calculate standard deviation
        if self.statistics.norm_history.len() > 1 {
            let mean = self.statistics.avg_norm;
            let variance: f64 = self
                .statistics
                .norm_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (self.statistics.norm_history.len() - 1) as f64;
            self.statistics.std_norm = variance.sqrt();
        }

        // Update adaptive threshold if enabled
        if self.config.adaptive_scaling {
            self.update_adaptive_threshold();
        }
    }

    /// Update the adaptive clipping threshold based on gradient statistics
    fn update_adaptive_threshold(&mut self) {
        let base_threshold = self.config.max_norm;

        // Adjust threshold based on gradient variance and average
        // Higher variance suggests need for more aggressive clipping
        let variance_factor = if self.statistics.std_norm > 0.0 {
            (self.statistics.std_norm / self.statistics.avg_norm).min(2.0)
        } else {
            1.0
        };

        // Adaptive adjustment based on recent clipping frequency
        let recent_clip_rate = if self.statistics.total_updates > 0 {
            self.statistics.clip_count as f64 / self.statistics.total_updates as f64
        } else {
            0.0
        };

        // If clipping too frequently, reduce threshold; if rarely clipping, increase threshold
        let frequency_adjustment = if recent_clip_rate > 0.5 {
            0.9 // Reduce threshold by 10%
        } else if recent_clip_rate < 0.1 {
            1.1 // Increase threshold by 10%
        } else {
            1.0 // Keep current threshold
        };

        let new_threshold = base_threshold * variance_factor * frequency_adjustment;

        // Clamp to configured bounds
        self.statistics.adaptive_threshold = new_threshold
            .max(self.config.min_threshold)
            .min(self.config.max_threshold);
    }

    /// Get the effective clipping threshold (adaptive or fixed)
    fn get_effective_threshold(&self) -> f64 {
        if self.config.adaptive_scaling {
            self.statistics.adaptive_threshold
        } else {
            self.config.max_norm
        }
    }

    /// Get current gradient statistics
    pub fn get_statistics(&self) -> &GradientStatistics {
        &self.statistics
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &GradientClippingConfig {
        &self.config
    }

    /// Reset statistics (useful for training phase transitions)
    pub fn reset_statistics(&mut self) {
        self.statistics = GradientStatistics::default();
        self.step_count = 0;
    }

    /// Get clipping rate (percentage of updates where clipping was applied)
    pub fn get_clipping_rate(&self) -> f64 {
        if self.statistics.total_updates > 0 {
            self.statistics.clip_count as f64 / self.statistics.total_updates as f64
        } else {
            0.0
        }
    }

    /// Check if gradients would be clipped with current threshold
    pub fn would_clip(&self, gradients: &[Tensor<T>]) -> Result<bool> {
        let global_norm = self.compute_global_norm(gradients)?;
        Ok(global_norm > self.get_effective_threshold())
    }
}

impl<T> Tensor<T>
where
    T: Float
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + Default
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Convenience method to multiply tensor by a scalar
    pub fn mul_scalar(&self, scalar: T) -> Result<Tensor<T>> {
        match &self.storage {
            crate::tensor::TensorStorage::Cpu(array) => {
                let scaled_array = array.mapv(|x| x * scalar);
                Ok(Tensor::from_array(scaled_array))
            }
            #[cfg(feature = "gpu")]
            crate::tensor::TensorStorage::Gpu(_) => {
                let gpu_id = match self.device() {
                    crate::Device::Gpu(id) => *id,
                    _ => {
                        return Err(crate::TensorError::unsupported_operation_simple(
                            "GPU tensor storage without a GPU device is an inconsistent state"
                                .to_string(),
                        ))
                    }
                };
                let cpu_tensor = self.to_cpu()?;
                let cpu_result = cpu_tensor.mul_scalar(scalar)?;
                cpu_result.to_gpu(gpu_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gradient_clipping_basic() {
        let mut clipper = GradientClipper::<f32>::default_stable();

        // Create test gradients with large norms
        let large_grad = Tensor::from_array(Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0]).into_dyn());
        let mut gradients = vec![large_grad];

        let norm = clipper
            .clip_gradients(&mut gradients)
            .expect("test: clip_gradients should succeed");

        // Should have clipped since norm > 1.0
        assert!(norm > 1.0);
        assert_eq!(clipper.get_statistics().clip_count, 1);
    }

    #[test]
    fn test_adaptive_clipping() {
        let mut clipper = GradientClipper::<f32>::default_adaptive();

        // Simulate multiple gradient updates to test adaptive behavior
        for i in 0..10 {
            let scale = 1.0 + i as f32 * 0.5;
            let grad = Tensor::from_array(Array1::from_vec(vec![scale, scale]).into_dyn());
            let mut gradients = vec![grad];

            let _norm = clipper
                .clip_gradients(&mut gradients)
                .expect("test: clip_gradients should succeed");
        }

        // Adaptive threshold should have adjusted
        let stats = clipper.get_statistics();
        assert!(stats.total_updates == 10);
        assert!(stats.adaptive_threshold > 0.0);
    }

    #[test]
    fn test_parameter_groups() {
        let mut clipper = GradientClipper::<f32>::new(GradientClippingConfig {
            per_parameter_clipping: true,
            ..Default::default()
        });

        clipper.add_parameter_group("embeddings".to_string(), 0.5);
        clipper.add_parameter_group("output".to_string(), 2.0);

        let grad = Tensor::from_array(Array1::from_vec(vec![1.5, 1.5]).into_dyn());
        let mut gradients = vec![grad];

        // Should clip with embedding threshold (0.5)
        let norm = clipper
            .clip_parameter_group("embeddings", &mut gradients)
            .expect("test: operation should succeed");
        assert!(norm > 0.5);
    }

    #[test]
    fn test_different_norm_types() {
        let l1_config = GradientClippingConfig {
            norm_type: NormType::L1,
            max_norm: 4.0,
            ..Default::default()
        };
        let mut l1_clipper = GradientClipper::<f32>::new(l1_config);

        let grad = Tensor::from_array(Array1::from_vec(vec![2.0, 2.0]).into_dyn());
        let mut gradients = vec![grad];

        let norm = l1_clipper
            .clip_gradients(&mut gradients)
            .expect("test: clip_gradients should succeed");
        assert_eq!(norm, 4.0); // L1 norm should be 4.0
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn gpu_l1_norm_matches_cpu_reference() {
        let clipper = GradientClipper::<f32>::new(GradientClippingConfig {
            norm_type: NormType::L1,
            ..Default::default()
        });
        let cpu_tensor = Tensor::from_array(Array1::from_vec(vec![3.0f32, -4.0, 1.0]).into_dyn());
        let gpu_tensor = match cpu_tensor.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };
        let expected = clipper
            .compute_tensor_l1_norm(&cpu_tensor)
            .expect("test: CPU L1 norm should succeed");
        let actual = clipper
            .compute_tensor_l1_norm(&gpu_tensor)
            .expect("test: GPU L1 norm should succeed with a real adapter");
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual {actual} vs expected {expected}"
        );
        assert!((expected - 8.0).abs() < 1e-5); // 3 + 4 + 1 = 8
    }

    #[test]
    fn gpu_l2_norm_matches_cpu_reference() {
        let clipper = GradientClipper::<f32>::new(GradientClippingConfig {
            norm_type: NormType::L2,
            ..Default::default()
        });
        let cpu_tensor = Tensor::from_array(Array1::from_vec(vec![3.0f32, 4.0]).into_dyn());
        let gpu_tensor = match cpu_tensor.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let expected = clipper
            .compute_tensor_l2_norm(&cpu_tensor)
            .expect("test: CPU L2 norm should succeed");
        let actual = clipper
            .compute_tensor_l2_norm(&gpu_tensor)
            .expect("test: GPU L2 norm should succeed with a real adapter");
        assert!((actual - expected).abs() < 1e-5);
        assert!((expected - 5.0).abs() < 1e-5); // sqrt(9+16) = 5
    }

    #[test]
    fn gpu_inf_norm_matches_cpu_reference() {
        let clipper = GradientClipper::<f32>::new(GradientClippingConfig {
            norm_type: NormType::Infinity,
            ..Default::default()
        });
        let cpu_tensor = Tensor::from_array(Array1::from_vec(vec![3.0f32, -7.0, 2.0]).into_dyn());
        let gpu_tensor = match cpu_tensor.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let expected = clipper
            .compute_tensor_inf_norm(&cpu_tensor)
            .expect("test: CPU inf norm should succeed");
        let actual = clipper
            .compute_tensor_inf_norm(&gpu_tensor)
            .expect("test: GPU inf norm should succeed with a real adapter");
        assert!((actual - expected).abs() < 1e-5);
        assert!((expected - 7.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_mul_scalar_matches_cpu_and_stays_on_gpu() {
        let cpu_tensor = Tensor::from_array(Array1::from_vec(vec![1.0f32, 2.0, 3.0]).into_dyn());
        let gpu_tensor = match cpu_tensor.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return,
        };
        let expected = cpu_tensor
            .mul_scalar(2.5)
            .expect("test: CPU mul_scalar should succeed");
        let actual = gpu_tensor
            .mul_scalar(2.5)
            .expect("test: GPU mul_scalar should succeed with a real adapter");

        // Critical regression guard: the result must still be GPU-resident, not
        // silently downgraded to CPU as a side effect of the readback+delegate fix.
        assert_eq!(actual.device(), &Device::Gpu(0));

        let actual_cpu = actual.to_cpu().expect("test: to_cpu should succeed");
        assert_eq!(
            actual_cpu.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }
}
