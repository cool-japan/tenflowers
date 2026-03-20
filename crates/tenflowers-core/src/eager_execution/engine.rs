//! EagerExecutionEngine implementation

use crate::device::context::{DeviceContext, DEVICE_MANAGER};
use crate::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::config::{CachedOperation, EagerExecutionConfig, ExecutionMetrics, OpSignature};
use super::memory_pool::{MemoryGuard, MemoryPool};
use super::reporting::{CacheStatistics, EagerPerformanceReport};

/// Eager execution engine optimized for low latency
pub struct EagerExecutionEngine {
    pub(super) config: EagerExecutionConfig,
    pub(super) op_cache: RwLock<HashMap<OpSignature, CachedOperation>>,
    pub(super) memory_pool: MemoryPool,
    pub(super) metrics: Mutex<Vec<ExecutionMetrics>>,
    pub(super) active_contexts: RwLock<HashMap<Device, Arc<dyn DeviceContext>>>,
    pub(super) fusion_opportunities: RwLock<Vec<FusionOpportunity>>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct FusionOpportunity {
    pub(super) operations: Vec<String>,
    pub(super) potential_speedup: f64,
    pub(super) memory_savings: usize,
}

impl EagerExecutionEngine {
    /// Create a new eager execution engine
    pub fn new(config: EagerExecutionConfig) -> Self {
        Self {
            memory_pool: MemoryPool::new(config.clone()),
            config,
            op_cache: RwLock::new(HashMap::new()),
            metrics: Mutex::new(Vec::new()),
            active_contexts: RwLock::new(HashMap::new()),
            fusion_opportunities: RwLock::new(Vec::new()),
        }
    }

    /// Execute an operation with optimized eager execution
    pub fn execute_operation<T, F>(
        &self,
        operation: &str,
        inputs: &[&Tensor<T>],
        params: &HashMap<String, String>,
        executor: F,
    ) -> Result<(Tensor<T>, ExecutionMetrics)>
    where
        T: Clone + Send + Sync + 'static,
        F: FnOnce(&[&Tensor<T>]) -> Result<Tensor<T>>,
    {
        let overall_start = Instant::now();

        // Create operation signature
        let signature = self.create_signature(operation, inputs, params)?;

        // Check cache first
        let setup_start = Instant::now();
        let cache_hit = self.check_cache(&signature);
        let setup_time = setup_start.elapsed();

        // Execute operation
        let exec_start = Instant::now();
        let result = if cache_hit && self.config.enable_op_cache {
            // For demonstration - in real implementation, cached results would be stored
            executor(inputs)?
        } else {
            // Optimize memory allocation
            let _memory_guard = if self.config.enable_memory_pool {
                Some(self.prepare_memory_for_operation(&signature)?)
            } else {
                None
            };

            // Execute with context optimization
            let result = if self.config.enable_context_optimization {
                self.execute_with_context_optimization(inputs, executor)?
            } else {
                executor(inputs)?
            };

            // Cache the operation
            if self.config.enable_op_cache {
                self.cache_operation(&signature, &result, exec_start.elapsed())?;
            }

            result
        };
        let execution_time = exec_start.elapsed();

        // Teardown
        let teardown_start = Instant::now();
        if self.config.enable_memory_pool {
            self.cleanup_operation_memory(&signature)?;
        }
        let teardown_time = teardown_start.elapsed();

        let total_time = overall_start.elapsed();
        let total_overhead = total_time - execution_time;

        // Record metrics
        let metrics = ExecutionMetrics {
            operation: operation.to_string(),
            device: *inputs[0].device(),
            setup_time,
            execution_time,
            teardown_time,
            total_overhead,
            memory_allocation_time: Duration::ZERO, // Would be measured in real implementation
            cache_hit,
            meets_target: total_overhead.as_nanos() <= self.config.target_overhead_ns as u128,
        };

        self.metrics
            .lock()
            .expect("lock should not be poisoned")
            .push(metrics.clone());

        // Check for fusion opportunities
        if self.config.enable_kernel_fusion {
            self.analyze_fusion_opportunity(operation, &signature);
        }

        Ok((result, metrics))
    }

    /// Create operation signature for caching
    fn create_signature<T: 'static>(
        &self,
        operation: &str,
        inputs: &[&Tensor<T>],
        params: &HashMap<String, String>,
    ) -> Result<OpSignature> {
        let input_shapes: Vec<Vec<usize>> =
            inputs.iter().map(|t| t.shape().dims().to_vec()).collect();

        let device = *inputs[0].device();
        let dtype = inputs[0].dtype();

        let params: Vec<(String, String)> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        Ok(OpSignature {
            operation: operation.to_string(),
            input_shapes,
            dtype,
            device,
            params,
        })
    }

    /// Check if operation is cached
    fn check_cache(&self, signature: &OpSignature) -> bool {
        let cache = self
            .op_cache
            .read()
            .expect("read lock should not be poisoned");
        cache.contains_key(signature)
    }

    /// Cache operation result
    fn cache_operation<T>(
        &self,
        signature: &OpSignature,
        result: &Tensor<T>,
        execution_time: Duration,
    ) -> Result<()> {
        let mut cache = self
            .op_cache
            .write()
            .expect("write lock should not be poisoned");

        // Check cache size limit
        if cache.len() >= self.config.max_cache_size {
            // Remove oldest entry
            let oldest_key = cache
                .iter()
                .min_by_key(|(_, cached_op)| cached_op.last_used)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                cache.remove(&key);
            }
        }

        // Add new entry
        let cached_op = CachedOperation {
            signature: signature.clone(),
            result_shape: result.shape().dims().to_vec(),
            execution_time,
            memory_usage: result.shape().size() * std::mem::size_of::<T>(),
            created_at: Instant::now(),
            last_used: Instant::now(),
            use_count: 1,
        };

        cache.insert(signature.clone(), cached_op);
        Ok(())
    }

    /// Prepare memory for operation
    fn prepare_memory_for_operation(&self, signature: &OpSignature) -> Result<MemoryGuard> {
        // Calculate required memory based on operation signature
        let output_memory_required = self.estimate_output_memory_requirements(signature)?;
        let intermediate_memory_required =
            self.estimate_intermediate_memory_requirements(signature)?;

        // Pre-warm memory pools for known patterns
        if output_memory_required > 1024 * 1024 {
            // > 1MB
            self.pre_warm_memory_pool(&signature.device, output_memory_required)?;
        }

        // Optimize memory layout for the specific operation
        self.optimize_memory_layout_for_operation(signature)?;

        // Return a guard that will cleanup when dropped
        Ok(MemoryGuard {
            device: signature.device,
            estimated_memory: output_memory_required + intermediate_memory_required,
            operation: signature.operation.clone(),
        })
    }

    /// Estimate output memory requirements for an operation
    fn estimate_output_memory_requirements(&self, signature: &OpSignature) -> Result<usize> {
        let element_size = self.get_dtype_size(&signature.dtype);

        let output_elements = match signature.operation.as_str() {
            // Element-wise operations preserve input shape
            "add" | "sub" | "mul" | "div" | "relu" | "sigmoid" | "tanh" | "gelu" => signature
                .input_shapes
                .iter()
                .map(|shape| shape.iter().product::<usize>())
                .max()
                .unwrap_or(0),

            // Matrix multiplication output shape
            "matmul" => {
                if signature.input_shapes.len() >= 2
                    && signature.input_shapes[0].len() >= 2
                    && signature.input_shapes[1].len() >= 2
                {
                    let m = signature.input_shapes[0][signature.input_shapes[0].len() - 2];
                    let n = signature.input_shapes[1][signature.input_shapes[1].len() - 1];
                    let batch_size = signature.input_shapes[0]
                        .iter()
                        .take(signature.input_shapes[0].len() - 2)
                        .product::<usize>();
                    batch_size * m * n
                } else {
                    0
                }
            }

            // Reduction operations reduce dimensionality
            "sum" | "mean" | "max" | "min" => {
                // For simplification, assume reduction to scalar (worst case for memory estimation)
                signature
                    .input_shapes
                    .iter()
                    .map(|shape| shape.iter().product::<usize>() / shape.len().max(1))
                    .sum()
            }

            // Convolution operations (simplified estimation)
            "conv2d" => {
                if !signature.input_shapes.is_empty() && signature.input_shapes[0].len() >= 4 {
                    let batch = signature.input_shapes[0][0];
                    let height = signature.input_shapes[0][2];
                    let width = signature.input_shapes[0][3];
                    // Assume output channels from parameters or default to input channels
                    let output_channels = signature.input_shapes[0][1]; // Simplified
                    batch * output_channels * height * width
                } else {
                    0
                }
            }

            _ => {
                // Conservative estimate: same as largest input
                signature
                    .input_shapes
                    .iter()
                    .map(|shape| shape.iter().product::<usize>())
                    .max()
                    .unwrap_or(0)
            }
        };

        Ok(output_elements * element_size)
    }

    /// Estimate intermediate memory requirements
    fn estimate_intermediate_memory_requirements(&self, signature: &OpSignature) -> Result<usize> {
        let element_size = self.get_dtype_size(&signature.dtype);
        let total_input_elements: usize = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();

        let intermediate_factor = match signature.operation.as_str() {
            // Simple element-wise operations need minimal intermediate storage
            "add" | "sub" | "mul" | "div" => 0.1,

            // Activations may need temporary storage for gradients
            "relu" | "sigmoid" | "tanh" | "gelu" => 0.2,

            // Matrix operations may need substantial temporary storage
            "matmul" => 0.5,

            // Normalization operations need statistics storage
            "batch_norm" | "layer_norm" | "group_norm" => 0.8,

            // Convolutions need intermediate feature maps
            "conv2d" | "conv3d" => 1.2,

            // Reductions need temporary partial results
            "sum" | "mean" | "max" | "min" => 0.3,

            _ => 0.5, // Conservative default
        };

        Ok((total_input_elements as f64 * intermediate_factor * element_size as f64) as usize)
    }

    /// Pre-warm memory pool for large allocations
    fn pre_warm_memory_pool(&self, device: &Device, required_memory: usize) -> Result<()> {
        // Pre-allocate memory chunks to avoid allocation overhead during operation
        if self.config.enable_memory_pool {
            let warmup_size = required_memory.next_power_of_two();

            // Pre-allocate 2-3 blocks to handle peak usage during operation
            // This reduces allocation overhead during critical execution paths
            let num_blocks = if warmup_size > 1024 * 1024 { 2 } else { 3 }; // Fewer large blocks

            // Use the memory pool's pre-warming functionality
            self.memory_pool.pre_warm(device, warmup_size, num_blocks)?;
        }
        Ok(())
    }

    /// Optimize memory layout for specific operations
    fn optimize_memory_layout_for_operation(&self, signature: &OpSignature) -> Result<()> {
        match signature.operation.as_str() {
            // Matrix operations benefit from contiguous layout
            "matmul" | "conv2d" | "conv3d" => {
                // Could implement memory layout optimization hints here
                // This would be device-specific optimization
            }

            // Element-wise operations are less layout-sensitive
            "add" | "sub" | "mul" | "div" => {
                // Minimal layout requirements
            }

            _ => {
                // Default layout optimization
            }
        }
        Ok(())
    }

    /// Get the size in bytes for a data type
    fn get_dtype_size(&self, dtype: &DType) -> usize {
        match dtype {
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Int4 => 1, // 4-bit packed, but minimum allocation unit is 1 byte
            DType::UInt8 => 1,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Bool => 1,
            DType::Complex32 => 8,
            DType::Complex64 => 16,
            DType::String => 8, // Pointer size (strings are heap-allocated)
        }
    }

    /// Execute with context optimization
    fn execute_with_context_optimization<T, F>(
        &self,
        inputs: &[&Tensor<T>],
        executor: F,
    ) -> Result<Tensor<T>>
    where
        F: FnOnce(&[&Tensor<T>]) -> Result<Tensor<T>>,
    {
        let device = *inputs[0].device();

        // Cache active context to avoid repeated lookups
        {
            let mut contexts = self
                .active_contexts
                .write()
                .expect("write lock should not be poisoned");
            if let std::collections::hash_map::Entry::Vacant(e) = contexts.entry(device) {
                let context = DEVICE_MANAGER.get_context(&device)?;
                e.insert(context);
            }
        }

        // Execute operation
        executor(inputs)
    }

    /// Cleanup memory after operation
    fn cleanup_operation_memory(&self, _signature: &OpSignature) -> Result<()> {
        // In a real implementation, this would release memory back to pool
        Ok(())
    }

    /// Analyze potential fusion opportunities
    fn analyze_fusion_opportunity(&self, operation: &str, signature: &OpSignature) {
        let mut opportunities = self
            .fusion_opportunities
            .write()
            .expect("write lock should not be poisoned");

        // Advanced fusion analysis based on operation patterns
        let fusion_speedup = match operation {
            // Element-wise operations have high fusion potential
            "add" | "sub" | "mul" | "div" => self.calculate_elementwise_fusion_benefit(signature),

            // Activation functions can be fused with previous operations
            "relu" | "sigmoid" | "tanh" | "gelu" => {
                self.calculate_activation_fusion_benefit(signature)
            }

            // Normalization operations benefit from fusion with preceding computations
            "batch_norm" | "layer_norm" | "group_norm" => {
                self.calculate_normalization_fusion_benefit(signature)
            }

            // Matrix operations with compatible dimensions
            "matmul" | "conv2d" | "conv3d" => {
                self.calculate_compute_intensive_fusion_benefit(signature)
            }

            // Reduction operations can be fused with element-wise operations
            "sum" | "mean" | "max" | "min" => self.calculate_reduction_fusion_benefit(signature),

            _ => 1.0, // No fusion benefit
        };

        // Only track meaningful fusion opportunities
        if fusion_speedup > 1.1 && opportunities.len() < 50 {
            let memory_savings = self.estimate_memory_savings(operation, signature);

            // Look for existing fusion chains to extend
            if let Some(existing) = opportunities
                .iter_mut()
                .find(|opp| self.can_extend_fusion_chain(&opp.operations, operation))
            {
                existing.operations.push(operation.to_string());
                existing.potential_speedup *= fusion_speedup.min(1.5); // Cap compounding
                existing.memory_savings += memory_savings;
            } else {
                opportunities.push(FusionOpportunity {
                    operations: vec![operation.to_string()],
                    potential_speedup: fusion_speedup,
                    memory_savings,
                });
            }
        }
    }

    /// Calculate fusion benefit for element-wise operations
    fn calculate_elementwise_fusion_benefit(&self, signature: &OpSignature) -> f64 {
        let total_elements: usize = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();

        // Larger tensors benefit more from fusion (reduced memory bandwidth)
        if total_elements > 10_000 {
            1.8 // High benefit for large tensors
        } else if total_elements > 1_000 {
            1.4 // Medium benefit
        } else {
            1.1 // Low benefit for small tensors
        }
    }

    /// Calculate fusion benefit for activation functions
    #[allow(unused_variables)] // signature used in conditional compilation
    fn calculate_activation_fusion_benefit(&self, signature: &OpSignature) -> f64 {
        // Activation functions are very good fusion candidates
        // as they're typically applied element-wise after compute operations
        let is_gpu = {
            #[cfg(feature = "gpu")]
            {
                matches!(signature.device, Device::Gpu(_))
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        if is_gpu {
            1.6 // GPU benefits more from activation fusion
        } else {
            1.3 // CPU still benefits from reduced memory transfers
        }
    }

    /// Calculate fusion benefit for normalization operations
    fn calculate_normalization_fusion_benefit(&self, signature: &OpSignature) -> f64 {
        // Normalization operations involve multiple passes over data
        // Fusion can eliminate intermediate allocations
        let input_size: usize = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .max()
            .unwrap_or(0);

        if input_size > 50_000 {
            1.7 // High benefit for large feature maps
        } else {
            1.2 // Moderate benefit for smaller inputs
        }
    }

    /// Calculate fusion benefit for compute-intensive operations
    fn calculate_compute_intensive_fusion_benefit(&self, signature: &OpSignature) -> f64 {
        // Matrix operations can benefit from fusion with element-wise post-processing
        let is_large_computation = signature
            .input_shapes
            .iter()
            .any(|shape| shape.iter().product::<usize>() > 100_000);

        if is_large_computation {
            1.4 // Moderate benefit - these operations are already compute-bound
        } else {
            1.1 // Low benefit for small computations
        }
    }

    /// Calculate fusion benefit for reduction operations
    fn calculate_reduction_fusion_benefit(&self, signature: &OpSignature) -> f64 {
        // Reductions can be fused with preceding element-wise operations
        let input_size: usize = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .max()
            .unwrap_or(0);

        if input_size > 20_000 {
            1.5 // Good benefit for large reductions
        } else {
            1.2 // Moderate benefit
        }
    }

    /// Estimate memory savings from fusion
    fn estimate_memory_savings(&self, operation: &str, signature: &OpSignature) -> usize {
        let element_size = match signature.dtype {
            DType::Float16 => 2,
            DType::BFloat16 => 2,
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Int4 => 1, // 4-bit packed, but minimum allocation unit is 1 byte
            DType::UInt8 => 1,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Bool => 1,
            DType::Complex32 => 8,
            DType::Complex64 => 16,
            DType::String => 8, // Pointer size (strings are heap-allocated)
        };

        let total_elements: usize = signature
            .input_shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();

        // Estimate intermediate buffer savings
        match operation {
            "add" | "sub" | "mul" | "div" => total_elements * element_size, // One intermediate buffer saved
            "relu" | "sigmoid" | "tanh" => total_elements * element_size / 2, // Smaller savings for activations
            "batch_norm" | "layer_norm" => total_elements * element_size * 2, // Multiple intermediate buffers
            _ => total_elements * element_size / 4, // Conservative estimate
        }
    }

    /// Check if an operation can extend an existing fusion chain
    fn can_extend_fusion_chain(&self, existing_ops: &[String], new_op: &str) -> bool {
        if existing_ops.is_empty() {
            return false;
        }

        let last_op = &existing_ops[existing_ops.len() - 1];

        // Define compatible operation sequences
        match (last_op.as_str(), new_op) {
            // Element-wise operations can be chained
            ("add" | "sub" | "mul" | "div", "add" | "sub" | "mul" | "div") => true,

            // Compute operations followed by activations
            ("matmul" | "conv2d" | "conv3d", "relu" | "sigmoid" | "tanh" | "gelu") => true,
            ("add" | "sub", "relu" | "sigmoid" | "tanh" | "gelu") => true,

            // Activations followed by normalization
            ("relu" | "sigmoid" | "tanh" | "gelu", "batch_norm" | "layer_norm") => true,

            // Any operation followed by reduction
            (_, "sum" | "mean" | "max" | "min") => existing_ops.len() < 3, // Limit chain length

            _ => false,
        }
    }

    /// Get execution metrics
    pub fn get_metrics(&self) -> Vec<ExecutionMetrics> {
        self.metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let cache = self
            .op_cache
            .read()
            .expect("read lock should not be poisoned");

        let total_entries = cache.len();
        let total_hits = cache.values().map(|op| op.use_count).sum();
        let avg_execution_time = if total_entries > 0 {
            cache
                .values()
                .map(|op| op.execution_time.as_nanos())
                .sum::<u128>()
                / total_entries as u128
        } else {
            0
        };

        CacheStatistics {
            total_entries,
            total_hits,
            hit_rate: if total_hits > 0 {
                cache.len() as f64 / total_hits as f64
            } else {
                0.0
            },
            avg_execution_time: Duration::from_nanos(avg_execution_time as u64),
        }
    }

    /// Generate performance report
    pub fn generate_performance_report(&self) -> EagerPerformanceReport {
        let metrics = self.get_metrics();
        let cache_stats = self.get_cache_stats();

        if metrics.is_empty() {
            return EagerPerformanceReport::default();
        }

        let total_operations = metrics.len();
        let meets_target = metrics.iter().filter(|m| m.meets_target).count();
        let success_rate = meets_target as f64 / total_operations as f64;

        let avg_overhead = Duration::from_nanos(
            (metrics
                .iter()
                .map(|m| m.total_overhead.as_nanos())
                .sum::<u128>()
                / total_operations as u128) as u64,
        );

        let min_overhead = metrics
            .iter()
            .map(|m| m.total_overhead)
            .min()
            .unwrap_or(Duration::ZERO);

        let max_overhead = metrics
            .iter()
            .map(|m| m.total_overhead)
            .max()
            .unwrap_or(Duration::ZERO);

        let cache_hit_rate =
            metrics.iter().filter(|m| m.cache_hit).count() as f64 / total_operations as f64;

        EagerPerformanceReport {
            total_operations,
            operations_meeting_target: meets_target,
            success_rate,
            avg_overhead,
            min_overhead,
            max_overhead,
            cache_statistics: cache_stats,
            cache_hit_rate,
            target_overhead: Duration::from_nanos(self.config.target_overhead_ns),
            recommendations: self.generate_recommendations(&metrics),
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, metrics: &[ExecutionMetrics]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let avg_overhead = if !metrics.is_empty() {
            metrics
                .iter()
                .map(|m| m.total_overhead.as_nanos())
                .sum::<u128>()
                / metrics.len() as u128
        } else {
            0
        };

        if avg_overhead > self.config.target_overhead_ns as u128 {
            recommendations
                .push("Consider enabling operation caching to reduce setup overhead".to_string());
            recommendations.push("Enable memory pooling to reduce allocation overhead".to_string());
        }

        let cache_hit_rate = if !metrics.is_empty() {
            metrics.iter().filter(|m| m.cache_hit).count() as f64 / metrics.len() as f64
        } else {
            0.0
        };

        if cache_hit_rate < 0.3 {
            recommendations.push("Increase cache size to improve hit rates".to_string());
        }

        let high_setup_ops = metrics
            .iter()
            .filter(|m| m.setup_time > Duration::from_micros(100))
            .count();

        if high_setup_ops > metrics.len() / 4 {
            recommendations
                .push("Enable context optimization to reduce setup overhead".to_string());
        }

        recommendations
    }

    /// Clean up old cache entries and memory blocks
    pub fn cleanup(&self) {
        // Clean up memory pool
        self.memory_pool.cleanup_old_blocks();

        // Clean up old cache entries
        let threshold = Duration::from_secs(300); // 5 minutes
        let now = Instant::now();

        let mut cache = self
            .op_cache
            .write()
            .expect("write lock should not be poisoned");
        cache.retain(|_, cached_op| now.duration_since(cached_op.last_used) <= threshold);
    }
}
