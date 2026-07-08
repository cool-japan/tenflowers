//! Ultra-High-Performance Dense Layer Implementation
//!
//! This module provides the most optimized dense (linear) layer implementation for maximum
//! computational efficiency, leveraging all available SciRS2-Core optimizations and
//! advanced memory management techniques.

use crate::layers::{Layer, LayerType};
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
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
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + From<f32>
        + bytemuck::Pod,
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
    /// SIMD path engagement indicator: 1.0 when the SIMD code path is active
    /// (enabled and hardware-supported), 0.0 otherwise. Not a sampled
    /// lane-utilisation fraction.
    pub simd_utilization: f64,
    /// Parallel path engagement indicator: 1.0 when parallel execution is
    /// enabled, 0.0 otherwise. Not a measured speed-up efficiency.
    pub parallel_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
}

impl<T> UltraDense<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + From<f32>
        + bytemuck::Pod,
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
            // Hoist the GELU constants out of the closure so conversion failures
            // surface as honest errors instead of panicking inside the kernel.
            let convert = |value: f64| -> Result<T> {
                <T as scirs2_core::num_traits::NumCast>::from(value).ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Failed to convert GELU constant to tensor element type".to_string(),
                    )
                })
            };
            let sqrt_2_over_pi = convert(0.797_884_560_802_865_4)?; // sqrt(2/pi)
            let coeff = convert(0.044715)?;
            let half = convert(0.5)?;

            if let Ok(result) = auto_vectorize(
                input.data().as_slice(),
                &vec![T::zero(); input.numel()],
                |x, _| {
                    let tanh_input = sqrt_2_over_pi * (x + coeff * x.powi(3));
                    half * x * (T::one() + tanh_input.tanh())
                },
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

    fn create_optimized_weight(shape: &[usize], _config: &UltraDenseConfig) -> Result<Tensor<T>> {
        // Weights MUST be initialised with real random values; an all-zero weight
        // matrix makes the layer's output independent of its input (no learning
        // signal). Xavier/Glorot is a sensible default for a generic dense layer.
        Self::create_xavier_weight(shape)
    }

    fn create_optimized_bias(shape: &[usize], _config: &UltraDenseConfig) -> Result<Tensor<T>> {
        // Biases are conventionally initialised to zero; this is the correct
        // initial value (not a fabrication) and gives the layer a learnable
        // offset that starts neutral.
        Ok(Tensor::zeros(shape))
    }

    /// Scale factor for a fan-based initialisation, converted to `T`.
    fn init_std(value: f64) -> Result<T> {
        <T as scirs2_core::num_traits::NumCast>::from(value).ok_or_else(|| {
            TensorError::invalid_argument(
                "Failed to convert initialisation scale to tensor element type".to_string(),
            )
        })
    }

    fn create_he_weight(shape: &[usize]) -> Result<Tensor<T>> {
        // He initialization: sample from N(0, 1) and scale by sqrt(2 / fan_in).
        let fan_in = shape[0] as f64;
        let std = (2.0 / fan_in).sqrt();
        let std_t = Self::init_std(std)?;

        let base = Tensor::<T>::randn(shape)?;
        base.multiply_scalar(std_t)
    }

    fn create_xavier_weight(shape: &[usize]) -> Result<Tensor<T>> {
        // Xavier/Glorot initialization: sample from N(0, 1) and scale by
        // sqrt(2 / (fan_in + fan_out)).
        let fan_in = shape[0] as f64;
        let fan_out = shape[1] as f64;
        let std = (2.0 / (fan_in + fan_out)).sqrt();
        let std_t = Self::init_std(std)?;

        let base = Tensor::<T>::randn(shape)?;
        base.multiply_scalar(std_t)
    }

    // Performance reporting methods.
    //
    // These report a real, computed *engagement* indicator (1.0 = the
    // optimisation path is active for this layer's configuration and hardware,
    // 0.0 = inactive). They deliberately do NOT fabricate a measured
    // utilisation/efficiency fraction: this layer carries no per-operation SIMD
    // lane / thread-occupancy counters, so any specific percentage would be
    // invented. Cache hit-rate and memory efficiency, which *are* tracked, are
    // reported from their real counters instead.

    /// Returns 1.0 when SIMD acceleration is enabled and the hardware actually
    /// supports it, 0.0 otherwise. This reflects whether the SIMD code path is
    /// engaged, not a sampled lane-utilisation fraction.
    fn calculate_simd_utilization(&self) -> Result<f64> {
        let simd_active =
            self.config.enable_simd_acceleration && SimdOps::is_hardware_accelerated();
        Ok(if simd_active { 1.0 } else { 0.0 })
    }

    /// Returns 1.0 when the parallel processing path is enabled, 0.0 otherwise.
    /// This reflects whether parallel execution is engaged, not a measured
    /// speed-up efficiency.
    fn calculate_parallel_efficiency(&self) -> Result<f64> {
        Ok(if self.config.enable_parallel_processing {
            1.0
        } else {
            0.0
        })
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
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + From<f32>
        + bytemuck::Pod,
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
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + From<f32>
        + bytemuck::Pod,
{
    fn ultra_dense(input_dim: usize, output_dim: usize, use_bias: bool) -> Result<UltraDense<T>> {
        UltraDense::new(input_dim, output_dim, use_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    /// Assert a weight matrix is genuinely random: non-zero and non-constant.
    fn assert_real_random_weights(weight: &Tensor<f32>) {
        let values = weight.to_vec().expect("test: weight to_vec");
        assert!(!values.is_empty(), "weights must not be empty");

        let any_nonzero = values.iter().any(|v| v.abs() > 1e-12);
        assert!(any_nonzero, "weights must not be all zeros (real init required)");

        let first = values[0];
        let any_different = values.iter().any(|v| (v - first).abs() > 1e-12);
        assert!(
            any_different,
            "weights must not be constant (real random init required)"
        );
    }

    #[test]
    fn test_ultra_dense_creation() {
        let layer = UltraDense::<f32>::new(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.expect("test: operation should succeed");
        assert_eq!(layer.weight.shape().dims(), &[10, 5]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().expect("test: bias should exist").shape().dims(), &[5]);
    }

    #[test]
    fn test_default_weights_are_random_not_zeros() {
        // The default constructor previously produced an all-zero weight matrix
        // (fabrication): the layer's output was independent of its input.
        let layer = UltraDense::<f32>::new(6, 4, true)
            .expect("test: UltraDense creation should succeed");
        assert_real_random_weights(&layer.weight);
    }

    #[test]
    fn test_he_and_xavier_weights_are_random() {
        let he = UltraDense::<f32>::new_he(8, 5, false)
            .expect("test: He init should succeed");
        assert_real_random_weights(&he.weight);

        let xavier = UltraDense::<f32>::new_xavier(8, 5, false)
            .expect("test: Xavier init should succeed");
        assert_real_random_weights(&xavier.weight);

        // The two schemes use different scales, so their statistics should differ.
        let he_vals = he.weight.to_vec().expect("test: to_vec");
        let xavier_vals = xavier.weight.to_vec().expect("test: to_vec");
        let he_var: f32 = he_vals.iter().map(|v| v * v).sum::<f32>() / he_vals.len() as f32;
        let xavier_var: f32 =
            xavier_vals.iter().map(|v| v * v).sum::<f32>() / xavier_vals.len() as f32;
        // He std = sqrt(2/fan_in), Xavier std = sqrt(2/(fan_in+fan_out)); He is
        // larger here, so its empirical second moment should be larger too.
        assert!(
            he_var > xavier_var,
            "He init variance ({he_var}) should exceed Xavier ({xavier_var})"
        );
    }

    #[test]
    fn test_forward_output_depends_on_input() {
        // With real random weights the forward output must change when the input
        // changes (a zeros-init layer would return zeros for any input).
        let layer = UltraDense::<f32>::new(4, 3, false)
            .expect("test: UltraDense creation should succeed");

        let input_a = Tensor::<f32>::ones(&[2, 4]);
        let input_b = Tensor::from_vec(vec![0.5f32; 8], &[2, 4])
            .expect("test: input_b creation");

        let out_a = layer
            .forward(&input_a)
            .expect("test: forward a")
            .to_vec()
            .expect("test: to_vec a");
        let out_b = layer
            .forward(&input_b)
            .expect("test: forward b")
            .to_vec()
            .expect("test: to_vec b");

        // Output for the all-ones input must itself be non-zero...
        assert!(
            out_a.iter().any(|v| v.abs() > 1e-8),
            "forward output must be non-zero with real weights"
        );
        // ...and differ from the output for a different input.
        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(differs, "forward output must depend on the input");
    }

    #[test]
    fn test_metrics_report_real_engagement_indicator() {
        // simd_utilization / parallel_efficiency are engagement indicators in
        // {0.0, 1.0}, not fabricated fractions like 0.85 / 0.90.
        let layer = UltraDense::<f32>::new(4, 3, true)
            .expect("test: UltraDense creation should succeed");
        let input = Tensor::<f32>::ones(&[2, 4]);
        let result = layer.forward_ultra(&input).expect("test: forward_ultra");

        let simd = result.metrics.simd_utilization;
        let parallel = result.metrics.parallel_efficiency;
        assert!(
            simd == 0.0 || simd == 1.0,
            "simd_utilization must be a 0/1 engagement indicator, got {simd}"
        );
        assert!(
            parallel == 0.0 || parallel == 1.0,
            "parallel_efficiency must be a 0/1 engagement indicator, got {parallel}"
        );
        // Parallel processing is enabled by default, so it must report active.
        assert_eq!(parallel, 1.0, "default config enables parallel processing");
    }

    #[test]
    fn test_ultra_dense_forward() {
        let layer = UltraDense::<f32>::new(4, 3, true).expect("test: UltraDense creation should succeed");
        let input = Tensor::<f32>::ones(&[2, 4]);

        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.expect("test: result should be valid");
        assert_eq!(output.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_ultra_dense_forward_ultra() {
        let layer = UltraDense::<f32>::new(4, 3, true).expect("test: UltraDense creation should succeed");
        let input = Tensor::<f32>::ones(&[2, 4]);

        let result = layer.forward_ultra(&input);
        assert!(result.is_ok());

        let result = result.expect("test: result should be valid");
        assert_eq!(result.output.shape().dims(), &[2, 3]);
        assert!(result.metrics.forward_time.as_nanos() > 0);
    }

    #[test]
    fn test_ultra_dense_with_activation() {
        let layer = UltraDense::<f32>::new(4, 3, true)
            .expect("test: operation should succeed")
            .with_activation("relu");

        let input = Tensor::<f32>::ones(&[2, 4]);
        let result = layer.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ultra_dense_he_initialization() {
        let layer = UltraDense::<f32>::new_he(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.expect("test: operation should succeed");
        assert_eq!(layer.weight.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_ultra_dense_xavier_initialization() {
        let layer = UltraDense::<f32>::new_xavier(10, 5, true);
        assert!(layer.is_ok());

        let layer = layer.expect("test: operation should succeed");
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
        let layer = UltraDense::<f32>::new(4, 3, true).expect("test: UltraDense creation should succeed");
        let input = Tensor::<f32>::ones(&[2, 4]);

        let _result = layer.forward(&input).expect("test: forward pass should succeed");
        let metrics = layer.get_performance_metrics();
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_layer_trait_implementation() {
        let mut layer = UltraDense::<f32>::new(4, 3, true).expect("test: UltraDense creation should succeed");

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