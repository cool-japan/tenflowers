//! SIMD-Accelerated Gradient Operations
//!
//! This module provides ultra-high-performance gradient operations using SciRS2-Core's
//! SIMD capabilities for maximum computational efficiency in automatic differentiation.

use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

// Use SciRS2-Core for maximum performance
use scirs2_core::memory::GlobalBufferPool;
use scirs2_core::profiling::Profiler;
use scirs2_core::simd::{
    simd_add_f32, simd_add_f64, simd_mul_f32, simd_mul_f64, SimdCapabilities, SimdOps,
};

/// SIMD-accelerated gradient operations engine
pub struct SimdGradOps {
    /// Global buffer pool for SIMD operations
    global_buffer_pool: Arc<GlobalBufferPool>,
    /// Performance profiler
    profiler: Arc<Profiler>,
    /// SIMD configuration
    config: SimdGradConfig,
}

/// Configuration for SIMD gradient operations
#[derive(Debug, Clone)]
pub struct SimdGradConfig {
    /// Enable vectorized operations
    pub enable_vectorization: bool,
    /// Enable parallel SIMD processing
    pub enable_parallel_simd: bool,
    /// Minimum array size for SIMD acceleration
    pub simd_threshold: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Enable hardware-specific optimizations
    pub enable_hardware_optimizations: bool,
}

/// SIMD operation performance metrics
#[derive(Debug, Default)]
pub struct SimdPerformanceMetrics {
    /// Time spent on SIMD operations
    pub simd_time: std::time::Duration,
    /// Time spent on fallback operations
    pub fallback_time: std::time::Duration,
    /// SIMD utilization percentage
    pub simd_utilization: f64,
    /// Number of vectorized operations
    pub vectorized_ops: usize,
    /// Number of fallback operations
    pub fallback_ops: usize,
    /// Performance improvement ratio
    pub speedup_ratio: f64,
}

impl SimdGradOps {
    /// Create a new SIMD gradient operations engine
    pub fn new(config: SimdGradConfig) -> Result<Self> {
        let global_buffer_pool = Arc::new(GlobalBufferPool::new());
        let profiler = Arc::new(Profiler::new());

        Ok(Self {
            global_buffer_pool,
            profiler,
            config,
        })
    }

    /// SIMD-accelerated addition backward pass
    pub fn add_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        // For addition, gradients are simply passed through
        let grad_lhs = grad_output.clone();
        let grad_rhs = grad_output.clone();

        // Apply SIMD optimization for broadcasting if needed
        let grad_lhs = self.apply_simd_broadcasting(&grad_lhs, lhs.shape().dims())?;
        let grad_rhs = self.apply_simd_broadcasting(&grad_rhs, rhs.shape().dims())?;

        Ok((grad_lhs, grad_rhs))
    }

    /// SIMD-accelerated multiplication backward pass
    pub fn mul_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        // grad_lhs = grad_output * rhs
        let grad_lhs = self.simd_multiply(grad_output, rhs)?;
        // grad_rhs = grad_output * lhs
        let grad_rhs = self.simd_multiply(grad_output, lhs)?;

        Ok((grad_lhs, grad_rhs))
    }

    /// SIMD-accelerated subtraction backward pass
    pub fn sub_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        // grad_lhs = grad_output
        let grad_lhs = self.apply_simd_broadcasting(grad_output, lhs.shape().dims())?;
        // grad_rhs = -grad_output
        let grad_rhs = self.simd_negate(grad_output)?;
        let grad_rhs = self.apply_simd_broadcasting(&grad_rhs, rhs.shape().dims())?;

        Ok((grad_lhs, grad_rhs))
    }

    /// SIMD-accelerated division backward pass
    pub fn div_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        // grad_lhs = grad_output / rhs
        let grad_lhs = self.simd_divide(grad_output, rhs)?;

        // grad_rhs = -grad_output * lhs / (rhs^2)
        let rhs_squared = self.simd_multiply(rhs, rhs)?;
        let neg_grad_output = self.simd_negate(grad_output)?;
        let temp = self.simd_multiply(&neg_grad_output, lhs)?;
        let grad_rhs = self.simd_divide(&temp, &rhs_squared)?;

        Ok((grad_lhs, grad_rhs))
    }

    /// SIMD-accelerated ReLU backward pass
    pub fn relu_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + PartialOrd
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        if self.config.enable_vectorization && grad_output.numel() >= self.config.simd_threshold {
            self.simd_relu_backward(grad_output, input)
        } else {
            self.fallback_relu_backward(grad_output, input)
        }
    }

    /// SIMD-accelerated sigmoid backward pass
    pub fn sigmoid_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        sigmoid_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _profiling_active = self.config.enable_profiling;

        if self.config.enable_vectorization && grad_output.numel() >= self.config.simd_threshold {
            self.simd_sigmoid_backward(grad_output, sigmoid_output)
        } else {
            self.fallback_sigmoid_backward(grad_output, sigmoid_output)
        }
    }

    /// SIMD-accelerated tanh backward pass
    pub fn tanh_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        tanh_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _session = self.profiler.start_session("tanh_backward_simd")?;

        if self.config.enable_vectorization && grad_output.numel() >= self.config.simd_threshold {
            self.simd_tanh_backward(grad_output, tanh_output)
        } else {
            self.fallback_tanh_backward(grad_output, tanh_output)
        }
    }

    /// Ultra-fast matrix multiplication gradient with SIMD
    pub fn matmul_backward_simd<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        let _session = self.profiler.start_session("matmul_backward_simd")?;

        // Use SIMD-accelerated matrix operations if available
        if self.config.enable_vectorization && SimdOps::is_hardware_accelerated() {
            self.simd_matmul_backward(grad_output, lhs, rhs)
        } else {
            self.fallback_matmul_backward(grad_output, lhs, rhs)
        }
    }

    // Private SIMD implementation methods

    /// SIMD-optimized multiplication
    fn simd_multiply<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        if self.config.enable_vectorization && a.numel() >= self.config.simd_threshold {
            // Use SciRS2-Core's auto-vectorization
            if let Ok(result) =
                auto_vectorize(a.data().as_slice(), b.data().as_slice(), |x, y| x * y)
            {
                return Tensor::from_vec(&result, a.shape().dims());
            }
        }

        // Fallback to regular multiplication
        a.mul(b)
    }

    /// SIMD-optimized division
    fn simd_divide<T>(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        if self.config.enable_vectorization && a.numel() >= self.config.simd_threshold {
            // Use SciRS2-Core's auto-vectorization
            if let Ok(result) =
                auto_vectorize(a.data().as_slice(), b.data().as_slice(), |x, y| x / y)
            {
                return Tensor::from_vec(&result, a.shape().dims());
            }
        }

        // Fallback to regular division
        a.div(b)
    }

    /// SIMD-optimized negation
    fn simd_negate<T>(&self, a: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        if self.config.enable_vectorization && a.numel() >= self.config.simd_threshold {
            // Use SciRS2-Core's SIMD operations
            if let Ok(result) =
                auto_vectorize(a.data().as_slice(), &vec![T::zero(); a.numel()], |x, _| -x)
            {
                return Tensor::from_vec(&result, a.shape().dims());
            }
        }

        // Fallback to regular negation
        a.neg()
    }

    /// Apply SIMD-optimized broadcasting
    fn apply_simd_broadcasting<T>(
        &self,
        tensor: &Tensor<T>,
        target_shape: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        if tensor.shape().dims() == target_shape {
            return Ok(tensor.clone());
        }

        // Use SIMD-accelerated broadcasting if beneficial
        if self.config.enable_vectorization && tensor.numel() >= self.config.simd_threshold {
            self.simd_broadcast(tensor, target_shape)
        } else {
            tensor.broadcast(target_shape)
        }
    }

    /// SIMD-accelerated broadcasting
    fn simd_broadcast<T>(&self, tensor: &Tensor<T>, target_shape: &[usize]) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        // Implement SIMD-optimized broadcasting
        // For now, fallback to regular broadcasting
        tensor.broadcast(target_shape)
    }

    /// SIMD-accelerated ReLU backward
    fn simd_relu_backward<T>(&self, grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + PartialOrd
            + bytemuck::Pod,
    {
        // Use parallel SIMD processing for large tensors
        if self.config.enable_parallel_simd && input.numel() > 10000 {
            self.parallel_simd_relu_backward(grad_output, input)
        } else {
            // Use SciRS2-Core's auto-vectorization
            if let Ok(result) = auto_vectorize(
                grad_output.data().as_slice(),
                input.data().as_slice(),
                |grad, inp| if inp > T::zero() { grad } else { T::zero() },
            ) {
                Tensor::from_vec(&result, grad_output.shape().dims())
            } else {
                self.fallback_relu_backward(grad_output, input)
            }
        }
    }

    /// Parallel SIMD ReLU backward
    fn parallel_simd_relu_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + PartialOrd
            + bytemuck::Pod,
    {
        let grad_data = grad_output.data().as_slice();
        let input_data = input.data().as_slice();

        // Simplified processing without parallel chunks for now
        let mut result_data = Vec::with_capacity(grad_data.len());
        for (grad, inp) in grad_data.iter().zip(input_data.iter()) {
            if *inp > T::zero() {
                result_data.push(*grad);
            } else {
                result_data.push(T::zero());
            }
        }

        Tensor::from_vec(&result_data, grad_output.shape().dims())
    }

    /// SIMD-accelerated sigmoid backward
    fn simd_sigmoid_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        sigmoid_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        // sigmoid_backward: grad_output * sigmoid_output * (1 - sigmoid_output)
        if let Ok(result) = auto_vectorize(
            grad_output.data().as_slice(),
            sigmoid_output.data().as_slice(),
            |grad, sig| grad * sig * (T::one() - sig),
        ) {
            Tensor::from_vec(&result, grad_output.shape().dims())
        } else {
            self.fallback_sigmoid_backward(grad_output, sigmoid_output)
        }
    }

    /// SIMD-accelerated tanh backward
    fn simd_tanh_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        tanh_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        // tanh_backward: grad_output * (1 - tanh_output^2)
        if let Ok(result) = auto_vectorize(
            grad_output.data().as_slice(),
            tanh_output.data().as_slice(),
            |grad, tanh| grad * (T::one() - tanh * tanh),
        ) {
            Tensor::from_vec(&result, grad_output.shape().dims())
        } else {
            self.fallback_tanh_backward(grad_output, tanh_output)
        }
    }

    /// SIMD-accelerated matrix multiplication backward
    fn simd_matmul_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + bytemuck::Pod,
    {
        // Use SIMD-accelerated matrix operations
        // For now, implement basic matrix multiplication gradients
        let rhs_transposed = rhs.transpose()?;
        let lhs_transposed = lhs.transpose()?;

        let grad_lhs = grad_output.matmul(&rhs_transposed)?;
        let grad_rhs = lhs_transposed.matmul(grad_output)?;

        Ok((grad_lhs, grad_rhs))
    }

    // Fallback implementations for non-SIMD cases

    fn fallback_relu_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        input: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + Float
            + FromPrimitive
            + PartialOrd,
    {
        crate::grad_ops::relu_backward(grad_output, input)
    }

    fn fallback_sigmoid_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        sigmoid_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Zero + One + Send + Sync + 'static + Float + FromPrimitive,
    {
        crate::grad_ops::sigmoid_backward(grad_output, sigmoid_output)
    }

    fn fallback_tanh_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        tanh_output: &Tensor<T>,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default + Zero + One + Send + Sync + 'static + Float + FromPrimitive,
    {
        crate::grad_ops::tanh_backward(grad_output, tanh_output)
    }

    fn fallback_matmul_backward<T>(
        &self,
        grad_output: &Tensor<T>,
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone + Default + Zero + One + Send + Sync + 'static + Float + FromPrimitive,
    {
        let rhs_transposed = rhs.transpose()?;
        let lhs_transposed = lhs.transpose()?;

        let grad_lhs = grad_output.matmul(&rhs_transposed)?;
        let grad_rhs = lhs_transposed.matmul(grad_output)?;

        Ok((grad_lhs, grad_rhs))
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> Result<SimdPerformanceMetrics> {
        let metrics = self.profiler.get_metrics()?;

        Ok(SimdPerformanceMetrics {
            simd_time: metrics.get("simd_time").unwrap_or_default(),
            fallback_time: metrics.get("fallback_time").unwrap_or_default(),
            simd_utilization: metrics.get("simd_utilization").unwrap_or(0.0),
            vectorized_ops: metrics.get("vectorized_ops").unwrap_or(0) as usize,
            fallback_ops: metrics.get("fallback_ops").unwrap_or(0) as usize,
            speedup_ratio: metrics.get("speedup_ratio").unwrap_or(1.0),
        })
    }
}

impl Default for SimdGradConfig {
    fn default() -> Self {
        Self {
            enable_vectorization: true,
            enable_parallel_simd: true,
            simd_threshold: 1024,
            chunk_size: 4096,
            enable_hardware_optimizations: true,
        }
    }
}

/// Global SIMD gradient operations instance
static GLOBAL_SIMD_GRAD_OPS: std::sync::OnceLock<Arc<std::sync::Mutex<SimdGradOps>>> =
    std::sync::OnceLock::new();

/// Get the global SIMD gradient operations engine
pub fn global_simd_grad_ops() -> Arc<std::sync::Mutex<SimdGradOps>> {
    GLOBAL_SIMD_GRAD_OPS
        .get_or_init(|| {
            let config = SimdGradConfig::default();
            let ops = SimdGradOps::new(config).expect("Failed to create SIMD grad ops");
            Arc::new(std::sync::Mutex::new(ops))
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_simd_grad_ops_creation() {
        let config = SimdGradConfig::default();
        let ops = SimdGradOps::new(config);
        assert!(ops.is_ok());
    }

    #[test]
    fn test_simd_add_backward() {
        let config = SimdGradConfig::default();
        let ops = SimdGradOps::new(config).unwrap();

        let grad_output = Tensor::<f32>::ones(&[2, 2]);
        let lhs = Tensor::<f32>::ones(&[2, 2]);
        let rhs = Tensor::<f32>::ones(&[2, 2]);

        let result = ops.add_backward_simd(&grad_output, &lhs, &rhs);
        assert!(result.is_ok());

        let (grad_lhs, grad_rhs) = result.unwrap();
        assert_eq!(grad_lhs.shape().dims(), &[2, 2]);
        assert_eq!(grad_rhs.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_simd_mul_backward() {
        let config = SimdGradConfig::default();
        let ops = SimdGradOps::new(config).unwrap();

        let grad_output = Tensor::<f32>::ones(&[2, 2]);
        let lhs = Tensor::<f32>::full(&[2, 2], 2.0);
        let rhs = Tensor::<f32>::full(&[2, 2], 3.0);

        let result = ops.mul_backward_simd(&grad_output, &lhs, &rhs);
        assert!(result.is_ok());

        let (grad_lhs, grad_rhs) = result.unwrap();
        assert_eq!(grad_lhs.shape().dims(), &[2, 2]);
        assert_eq!(grad_rhs.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_simd_relu_backward() {
        let config = SimdGradConfig::default();
        let ops = SimdGradOps::new(config).unwrap();

        let grad_output = Tensor::<f32>::ones(&[2, 2]);
        let input = Tensor::<f32>::from_vec(&[1.0, -1.0, 2.0, -2.0], &[2, 2]).unwrap();

        let result = ops.relu_backward_simd(&grad_output, &input);
        assert!(result.is_ok());

        let grad_input = result.unwrap();
        assert_eq!(grad_input.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_global_simd_grad_ops() {
        let ops1 = global_simd_grad_ops();
        let ops2 = global_simd_grad_ops();

        // Should be the same instance
        assert!(Arc::ptr_eq(&ops1, &ops2));
    }

    #[test]
    fn test_simd_config() {
        let config = SimdGradConfig {
            enable_vectorization: false,
            simd_threshold: 2048,
            ..Default::default()
        };

        assert!(!config.enable_vectorization);
        assert_eq!(config.simd_threshold, 2048);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = SimdPerformanceMetrics::default();
        assert_eq!(metrics.vectorized_ops, 0);
        assert_eq!(metrics.fallback_ops, 0);
        assert_eq!(metrics.speedup_ratio, 0.0);
    }
}
