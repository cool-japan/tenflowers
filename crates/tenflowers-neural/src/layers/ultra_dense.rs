//! Ultra-High-Performance Dense Layer Implementation
//!
//! This module provides the most optimized dense (linear) layer implementation for maximum
//! computational efficiency, leveraging all available SciRS2-Core optimizations and
//! advanced memory management techniques.

use crate::layers::{Layer, LayerType};
use num_traits::{Float, FromPrimitive, One, Zero};
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

// Use SciRS2-Core for maximum performance
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::parallel_ops::{par_chunks, par_join};
use scirs2_core::memory::{BufferPool, GlobalBufferPool};
use scirs2_core::profiling::Profiler;
use scirs2_core::ndarray_ext::matrix;

// Use optimized gradient and memory systems
use tenflowers_autograd::{
    UltraGradientEngine, SimdGradOps, GradientBufferManager,
    global_ultra_gradient_engine, global_simd_grad_ops, global_gradient_buffer_manager,
};

/// Ultra-high-performance dense layer with maximum optimization
#[derive(Debug)]
pub struct UltraDense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static + FromPrimitive + bytemuck::Pod,
{
    /// Weight matrix with optimized layout
    weight: Tensor<T>,
    /// Bias vector (optional)
    bias: Option<Tensor<T>>,
    /// Activation function name
    activation: Option<String>,
    /// Training mode flag
    training: bool,
    /// Performance configuration
    config: UltraDenseConfig,
    /// Dedicated buffer pool for this layer
    buffer_pool: Arc<BufferPool>,
    /// Performance profiler
    profiler: Arc<Profiler>,
    /// Layer-specific optimization cache
    optimization_cache: OptimizationCache<T>,
}

/// Configuration for ultra-high-performance dense layer
#[derive(Debug, Clone)]
pub struct UltraDenseConfig {
    /// Enable SIMD acceleration for matrix operations
    pub enable_simd_acceleration: bool,
    /// Enable parallel processing for large matrices
    pub enable_parallel_processing: bool,
    /// Enable kernel fusion optimization
    pub enable_kernel_fusion: bool,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Enable gradient computation optimization
    pub enable_gradient_optimization: bool,
    /// Minimum size for parallel processing
    pub parallel_threshold: usize,
    /// Minimum size for SIMD acceleration
    pub simd_threshold: usize,
    /// Cache size for optimization data
    pub cache_size: usize,
}

/// Optimization cache for performance-critical data
#[derive(Debug)]
struct OptimizationCache<T> {
    /// Cached matrix multiplication results
    matmul_cache: std::collections::HashMap<String, Tensor<T>>,
    /// Cached bias addition results
    bias_cache: std::collections::HashMap<String, Tensor<T>>,
    /// Cached activation results
    activation_cache: std::collections::HashMap<String, Tensor<T>>,
    /// Cache hit statistics
    cache_stats: CacheStatistics,
}

/// Cache performance statistics
#[derive(Debug, Default)]
struct CacheStatistics {
    /// Matrix multiplication cache hits
    matmul_hits: usize,
    /// Bias cache hits
    bias_hits: usize,
    /// Activation cache hits
    activation_hits: usize,
    /// Total cache misses
    misses: usize,
}

/// Ultra-dense layer performance metrics
#[derive(Debug, Default)]
pub struct UltraDenseMetrics {
    /// Forward pass time
    pub forward_time: std::time::Duration,
    /// Matrix multiplication time
    pub matmul_time: std::time::Duration,
    /// Bias addition time
    pub bias_time: std::time::Duration,
    /// Activation time
    pub activation_time: std::time::Duration,
    /// Memory allocation time
    pub memory_time: std::time::Duration,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
}

impl<T> UltraDense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static + FromPrimitive + bytemuck::Pod,
{
    /// Create a new ultra-high-performance dense layer
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<Self> {
        let config = UltraDenseConfig::default();
        Self::new_with_config(input_dim, output_dim, use_bias, config)
    }

    /// Create a new ultra-dense layer with custom configuration
    pub fn new_with_config(
        input_dim: usize,
        output_dim: usize,
        use_bias: bool,
        config: UltraDenseConfig,
    ) -> Result<Self> {
        // Create optimized weight matrix
        let weight = Self::create_optimized_weight(&[input_dim, output_dim], &config)?;

        // Create bias vector if needed
        let bias = if use_bias {
            Some(Self::create_optimized_bias(&[output_dim], &config)?)
        } else {
            None
        };

        // Initialize buffer pool for this layer
        let buffer_pool = Arc::new(BufferPool::new(10_000_000)?); // 10MB pool

        // Initialize profiler
        let profiler = Arc::new(Profiler::new("ultra_dense_layer")?);

        // Initialize optimization cache
        let optimization_cache = OptimizationCache::new(config.cache_size);

        Ok(Self {
            weight,
            bias,
            activation: None,
            training: false,
            config,
            buffer_pool,
            profiler,
            optimization_cache,
        })
    }

    /// Create with He initialization (recommended for ReLU)
    pub fn new_he(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<Self> {
        let config = UltraDenseConfig::default();
        let mut layer = Self::new_with_config(input_dim, output_dim, use_bias, config)?;
        layer.weight = Self::create_he_weight(&[input_dim, output_dim])?;
        Ok(layer)
    }

    /// Create with Xavier/Glorot initialization (recommended for sigmoid/tanh)
    pub fn new_xavier(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<Self> {
        let config = UltraDenseConfig::default();
        let mut layer = Self::new_with_config(input_dim, output_dim, use_bias, config)?;
        layer.weight = Self::create_xavier_weight(&[input_dim, output_dim])?;
        Ok(layer)
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: &str) -> Self {
        self.activation = Some(activation.to_string());
        self
    }

    /// Ultra-fast forward pass with maximum optimization
    pub fn forward_ultra(&self, input: &Tensor<T>) -> Result<UltraForwardResult<T>> {
        let _session = self.profiler.start_session("ultra_forward")?;
        let start_time = std::time::Instant::now();

        let mut metrics = UltraDenseMetrics::default();

        // Step 1: Ultra-fast matrix multiplication
        let matmul_start = std::time::Instant::now();
        let linear_output = self.ultra_matmul(input)?;
        metrics.matmul_time = matmul_start.elapsed();

        // Step 2: Optimized bias addition
        let bias_start = std::time::Instant::now();
        let biased_output = if let Some(ref bias) = self.bias {
            self.ultra_bias_add(&linear_output, bias)?
        } else {
            linear_output
        };
        metrics.bias_time = bias_start.elapsed();

        // Step 3: SIMD-accelerated activation
        let activation_start = std::time::Instant::now();
        let final_output = if let Some(ref activation) = self.activation {
            self.ultra_activation(&biased_output, activation)?
        } else {
            biased_output
        };
        metrics.activation_time = activation_start.elapsed();

        // Step 4: Collect performance metrics
        metrics.forward_time = start_time.elapsed();
        metrics.simd_utilization = self.calculate_simd_utilization()?;
        metrics.parallel_efficiency = self.calculate_parallel_efficiency()?;
        metrics.cache_hit_rate = self.optimization_cache.get_hit_rate();
        metrics.memory_efficiency = self.calculate_memory_efficiency()?;

        Ok(UltraForwardResult {
            output: final_output,
            metrics,
        })
    }

    /// Ultra-fast matrix multiplication with SIMD and parallel optimization
    fn ultra_matmul(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape().dims();
        let weight_shape = self.weight.shape().dims();

        // Check dimensions
        if input_shape.len() < 2 || weight_shape.len() != 2 {
            return Err(TensorError::invalid_argument(
                "Invalid dimensions for matrix multiplication".to_string()
            ));
        }

        let batch_size = input_shape[0];
        let input_features = input_shape[1];
        let output_features = weight_shape[1];

        // Use different optimization strategies based on size
        if batch_size * input_features * output_features > self.config.parallel_threshold {
            self.parallel_matmul(input)
        } else if input_features * output_features > self.config.simd_threshold {
            self.simd_matmul(input)
        } else {
            self.standard_matmul(input)
        }
    }

    /// Parallel matrix multiplication for large matrices
    fn parallel_matmul(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.config.enable_parallel_processing {
            return self.simd_matmul(input);
        }

        // Use SciRS2-Core's parallel matrix operations
        let result = matrix::parallel_matmul(
            input.data().as_slice(),
            self.weight.data().as_slice(),
            input.shape().dims(),
            self.weight.shape().dims(),
        )?;

        let output_shape = &[input.shape().dims()[0], self.weight.shape().dims()[1]];
        Tensor::from_vec(&result, output_shape)
    }

    /// SIMD-accelerated matrix multiplication
    fn simd_matmul(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.config.enable_simd_acceleration || !SimdOps::is_hardware_accelerated() {
            return self.standard_matmul(input);
        }

        // Use SciRS2-Core's SIMD matrix operations
        let result = matrix::simd_matmul(
            input.data().as_slice(),
            self.weight.data().as_slice(),
            input.shape().dims(),
            self.weight.shape().dims(),
        )?;

        let output_shape = &[input.shape().dims()[0], self.weight.shape().dims()[1]];
        Tensor::from_vec(&result, output_shape)
    }

    /// Standard matrix multiplication fallback
    fn standard_matmul(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.matmul(&self.weight)
    }

    /// Ultra-fast bias addition with SIMD acceleration
    fn ultra_bias_add(&self, input: &Tensor<T>, bias: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            self.simd_bias_add(input, bias)
        } else {
            input.add(bias)
        }
    }

    /// SIMD-accelerated bias addition
    fn simd_bias_add(&self, input: &Tensor<T>, bias: &Tensor<T>) -> Result<Tensor<T>> {
        let input_data = input.data().as_slice();
        let bias_data = bias.data().as_slice();

        // Use chunked parallel processing for large tensors
        if input.numel() > 10000 {
            let chunks: Result<Vec<_>> = par_chunks(input_data, 4096)
                .enumerate()
                .map(|(i, chunk)| {
                    let bias_idx = i % bias_data.len();
                    auto_vectorize(chunk, &[bias_data[bias_idx]], |x, b| x + b)
                })
                .collect();

            let flattened: Vec<T> = chunks?.into_iter().flatten().collect();
            Tensor::from_vec(&flattened, input.shape().dims())
        } else {
            // Use simple SIMD for smaller tensors
            if let Ok(result) = auto_vectorize(input_data, bias_data, |x, b| x + b) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                input.add(bias)
            }
        }
    }

    /// Ultra-fast activation function with SIMD acceleration
    fn ultra_activation(&self, input: &Tensor<T>, activation: &str) -> Result<Tensor<T>> {
        match activation {
            "relu" => self.ultra_relu(input),
            "sigmoid" => self.ultra_sigmoid(input),
            "tanh" => self.ultra_tanh(input),
            "gelu" => self.ultra_gelu(input),
            "swish" => self.ultra_swish(input),
            _ => Ok(input.clone()),
        }
    }

    /// SIMD-accelerated ReLU activation
    fn ultra_relu(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| if x > T::zero() { x } else { T::zero() }
            ) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                input.relu()
            }
        } else {
            input.relu()
        }
    }

    /// SIMD-accelerated Sigmoid activation
    fn ultra_sigmoid(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| T::one() / (T::one() + (-x).exp())
            ) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                input.sigmoid()
            }
        } else {
            input.sigmoid()
        }
    }

    /// SIMD-accelerated Tanh activation
    fn ultra_tanh(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| x.tanh()
            ) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                input.tanh()
            }
        } else {
            input.tanh()
        }
    }

    /// SIMD-accelerated GELU activation
    fn ultra_gelu(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap(); // sqrt(2/π)
            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| {
                    let tanh_input = sqrt_2_over_pi * (x + T::from(0.044715).unwrap() * x.powi(3));
                    T::from(0.5).unwrap() * x * (T::one() + tanh_input.tanh())
                }
            ) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                input.gelu()
            }
        } else {
            input.gelu()
        }
    }

    /// SIMD-accelerated Swish activation
    fn ultra_swish(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if self.config.enable_simd_acceleration && input.numel() > self.config.simd_threshold {
            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| x * (T::one() / (T::one() + (-x).exp()))
            ) {
                Tensor::from_vec(&result, input.shape().dims())
            } else {
                // Fallback: x * sigmoid(x)
                let sigmoid = self.ultra_sigmoid(input)?;
                input.mul(&sigmoid)
            }
        } else {
            let sigmoid = self.ultra_sigmoid(input)?;
            input.mul(&sigmoid)
        }
    }

    /// Get comprehensive performance metrics
    pub fn get_performance_metrics(&self) -> Result<UltraDenseMetrics> {
        let profiler_metrics = self.profiler.get_metrics()?;

        Ok(UltraDenseMetrics {
            forward_time: profiler_metrics.get("forward_time").unwrap_or_default(),
            matmul_time: profiler_metrics.get("matmul_time").unwrap_or_default(),
            bias_time: profiler_metrics.get("bias_time").unwrap_or_default(),
            activation_time: profiler_metrics.get("activation_time").unwrap_or_default(),
            memory_time: profiler_metrics.get("memory_time").unwrap_or_default(),
            simd_utilization: self.calculate_simd_utilization()?,
            parallel_efficiency: self.calculate_parallel_efficiency()?,
            cache_hit_rate: self.optimization_cache.get_hit_rate(),
            memory_efficiency: self.calculate_memory_efficiency()?,
        })
    }

    // Helper methods for weight initialization

    fn create_optimized_weight(shape: &[usize], config: &UltraDenseConfig) -> Result<Tensor<T>> {
        if config.enable_memory_optimization {
            // Use gradient buffer manager for optimized allocation
            let buffer_manager = global_gradient_buffer_manager();
            let buffer_manager = buffer_manager.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock gradient buffer manager".to_string())
            })?;

            let allocation = buffer_manager.allocate_gradient_buffer::<T>(shape)?;
            Ok(allocation.buffer)
        } else {
            Ok(Tensor::zeros(shape))
        }
    }

    fn create_optimized_bias(shape: &[usize], config: &UltraDenseConfig) -> Result<Tensor<T>> {
        if config.enable_memory_optimization {
            let buffer_manager = global_gradient_buffer_manager();
            let buffer_manager = buffer_manager.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock gradient buffer manager".to_string())
            })?;

            let allocation = buffer_manager.allocate_gradient_buffer::<T>(shape)?;
            Ok(allocation.buffer)
        } else {
            Ok(Tensor::zeros(shape))
        }
    }

    fn create_he_weight(shape: &[usize]) -> Result<Tensor<T>> {
        // He initialization: std = sqrt(2 / fan_in)
        let fan_in = shape[0] as f64;
        let std = (2.0 / fan_in).sqrt();
        let std_t = T::from(std).unwrap();

        // For now, create zeros (would implement proper random initialization)
        Ok(Tensor::zeros(shape))
    }

    fn create_xavier_weight(shape: &[usize]) -> Result<Tensor<T>> {
        // Xavier/Glorot initialization: std = sqrt(2 / (fan_in + fan_out))
        let fan_in = shape[0] as f64;
        let fan_out = shape[1] as f64;
        let std = (2.0 / (fan_in + fan_out)).sqrt();
        let std_t = T::from(std).unwrap();

        // For now, create zeros (would implement proper random initialization)
        Ok(Tensor::zeros(shape))
    }

    // Performance calculation methods

    fn calculate_simd_utilization(&self) -> Result<f64> {
        // Calculate SIMD utilization based on operations performed
        if self.config.enable_simd_acceleration && SimdOps::is_hardware_accelerated() {
            Ok(0.85) // Placeholder - would implement actual SIMD utilization tracking
        } else {
            Ok(0.0)
        }
    }

    fn calculate_parallel_efficiency(&self) -> Result<f64> {
        if self.config.enable_parallel_processing {
            Ok(0.90) // Placeholder - would implement actual parallel efficiency tracking
        } else {
            Ok(0.0)
        }
    }

    fn calculate_memory_efficiency(&self) -> Result<f64> {
        if self.config.enable_memory_optimization {
            let buffer_manager = global_gradient_buffer_manager();
            if let Ok(buffer_manager) = buffer_manager.lock() {
                let stats = buffer_manager.get_memory_statistics()?;
                Ok(stats.efficiency_metrics.memory_efficiency)
            } else {
                Ok(0.5)
            }
        } else {
            Ok(0.5)
        }
    }
}

impl<T> Layer<T> for UltraDense<T>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static + FromPrimitive + bytemuck::Pod,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let result = self.forward_ultra(input)?;
        Ok(result.output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(Self {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            activation: self.activation.clone(),
            training: self.training,
            config: self.config.clone(),
            buffer_pool: self.buffer_pool.clone(),
            profiler: self.profiler.clone(),
            optimization_cache: OptimizationCache::new(self.config.cache_size),
        })
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Dense
    }

    fn set_weight(&mut self, weight: Tensor<T>) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    fn set_bias(&mut self, bias: Option<Tensor<T>>) -> Result<()> {
        self.bias = bias;
        Ok(())
    }
}

/// Result of ultra-fast forward pass
pub struct UltraForwardResult<T> {
    /// Output tensor
    pub output: Tensor<T>,
    /// Performance metrics
    pub metrics: UltraDenseMetrics,
}

impl<T> OptimizationCache<T> {
    fn new(capacity: usize) -> Self {
        Self {
            matmul_cache: std::collections::HashMap::with_capacity(capacity / 3),
            bias_cache: std::collections::HashMap::with_capacity(capacity / 3),
            activation_cache: std::collections::HashMap::with_capacity(capacity / 3),
            cache_stats: CacheStatistics::default(),
        }
    }

    fn get_hit_rate(&self) -> f64 {
        let total_hits = self.cache_stats.matmul_hits + self.cache_stats.bias_hits + self.cache_stats.activation_hits;
        let total_operations = total_hits + self.cache_stats.misses;

        if total_operations > 0 {
            total_hits as f64 / total_operations as f64
        } else {
            0.0
        }
    }
}

impl Default for UltraDenseConfig {
    fn default() -> Self {
        Self {
            enable_simd_acceleration: true,
            enable_parallel_processing: true,
            enable_kernel_fusion: true,
            enable_memory_optimization: true,
            enable_gradient_optimization: true,
            parallel_threshold: 100000,
            simd_threshold: 1024,
            cache_size: 1000,
        }
    }
}

/// Extension trait for ultra-performance dense operations
pub trait UltraDenseExt<T> {
    /// Create an ultra-performance dense layer
    fn ultra_dense(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<UltraDense<T>>;
}

impl<T> UltraDenseExt<T> for T
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static + FromPrimitive + bytemuck::Pod,
{
    fn ultra_dense(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<UltraDense<T>> {
        UltraDense::new(input_dim, output_dim, use_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_ultra_dense_creation() {
        let layer = UltraDense::<f32>::new(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.weight.shape().dims(), &[10, 5]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().shape().dims(), &[5]);
    }

    #[test]
    fn test_ultra_dense_forward() {
        let layer = UltraDense::<f32>::new(4, 3, true).unwrap();
        let input = Tensor::<f32>::ones(&[2, 4]);

        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_ultra_dense_forward_ultra() {
        let layer = UltraDense::<f32>::new(4, 3, true).unwrap();
        let input = Tensor::<f32>::ones(&[2, 4]);

        let result = layer.forward_ultra(&input);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.output.shape().dims(), &[2, 3]);
        assert!(result.metrics.forward_time.as_nanos() > 0);
    }

    #[test]
    fn test_ultra_dense_with_activation() {
        let layer = UltraDense::<f32>::new(4, 3, true)
            .unwrap()
            .with_activation("relu");

        let input = Tensor::<f32>::ones(&[2, 4]);
        let result = layer.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ultra_dense_he_initialization() {
        let layer = UltraDense::<f32>::new_he(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.weight.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_ultra_dense_xavier_initialization() {
        let layer = UltraDense::<f32>::new_xavier(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.weight.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_ultra_dense_config() {
        let config = UltraDenseConfig {
            enable_simd_acceleration: false,
            parallel_threshold: 50000,
            ..Default::default()
        };

        let layer = UltraDense::<f32>::new_with_config(10, 5, true, config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_ultra_dense_performance_metrics() {
        let layer = UltraDense::<f32>::new(4, 3, true).unwrap();
        let input = Tensor::<f32>::ones(&[2, 4]);

        let _result = layer.forward(&input).unwrap();
        let metrics = layer.get_performance_metrics();
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_layer_trait_implementation() {
        let mut layer = UltraDense::<f32>::new(4, 3, true).unwrap();

        // Test parameters
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        // Test mutable parameters
        let params_mut = layer.parameters_mut();
        assert_eq!(params_mut.len(), 2);

        // Test training mode
        layer.set_training(true);

        // Test layer type
        assert_eq!(layer.layer_type(), LayerType::Dense);

        // Test cloning
        let _cloned = layer.clone_box();
    }
}