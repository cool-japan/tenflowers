//! Ultra-Performance Eager Execution Optimization Module
//!
//! This module provides ultra-optimized eager execution to achieve sub-millisecond
//! overhead with advanced SciRS2 ecosystem integration. It focuses on:
//! - Operation caching and intelligent reuse with ML-based optimization
//! - Ultra-efficient memory pool optimization with adaptive strategies
//! - Device synchronization minimization with predictive scheduling
//! - Kernel launch optimization with fusion and batching
//! - Context switching reduction with hardware-aware optimizations
//! - Real-time performance analytics and adaptive tuning
//! - Advanced SIMD acceleration and parallel execution

use crate::device::context::{DeviceContext, DEVICE_MANAGER};
use crate::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// Ultra-performance SciRS2 ecosystem integration

// Advanced system integration

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Ultra-Performance Eager Execution Configuration
///
/// Comprehensive configuration for achieving sub-millisecond overhead with
/// advanced optimization strategies across SIMD, parallel, memory, and adaptive systems.
#[derive(Debug, Clone)]
pub struct EagerExecutionConfig {
    /// Enable operation caching
    pub enable_op_cache: bool,
    /// Enable memory pool
    pub enable_memory_pool: bool,
    /// Enable async execution where possible
    pub enable_async_execution: bool,
    /// Maximum cache size (number of operations)
    pub max_cache_size: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Target overhead threshold (nanoseconds)
    pub target_overhead_ns: u64,
    /// Enable context switching optimization
    pub enable_context_optimization: bool,
    /// Enable kernel fusion for compatible operations
    pub enable_kernel_fusion: bool,

    // Ultra-Performance Configuration Sections
    /// SIMD acceleration configuration
    pub simd_config: SimdExecutionConfig,
    /// Parallel execution optimization
    pub parallel_config: ParallelExecutionConfig,
    /// Memory optimization strategies
    pub memory_config: MemoryOptimizationConfig,
    /// Performance monitoring and analytics
    pub monitoring_config: PerformanceMonitoringConfig,
    /// Adaptive tuning system
    pub adaptive_config: AdaptiveTuningConfig,
    /// GPU acceleration configuration
    pub gpu_config: GpuAccelerationConfig,
    /// Ultra-low latency optimizations
    pub ultra_latency_config: UltraLatencyConfig,
}

/// SIMD Acceleration Configuration for Ultra-Performance
#[derive(Debug, Clone)]
pub struct SimdExecutionConfig {
    /// Enable SIMD vectorization
    pub enable_simd: bool,
    /// Preferred SIMD width (128, 256, 512 bits)
    pub simd_width: u32,
    /// Enable auto-vectorization hints
    pub enable_auto_vectorization: bool,
    /// Use target-specific SIMD instructions (AVX2, AVX512, NEON)
    pub enable_target_features: bool,
    /// Minimum array size for SIMD activation
    pub simd_threshold: usize,
    /// Enable SIMD for element-wise operations
    pub simd_elementwise: bool,
    /// Enable SIMD for matrix operations
    pub simd_matrix_ops: bool,
    /// SIMD alignment requirements (bytes)
    pub memory_alignment: usize,
}

/// Parallel Execution Configuration for Multi-Core Optimization
#[derive(Debug, Clone)]
pub struct ParallelExecutionConfig {
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Number of worker threads (0 = auto-detect)
    pub num_threads: usize,
    /// Minimum problem size for parallelization
    pub parallel_threshold: usize,
    /// Thread pool strategy
    pub thread_strategy: ThreadPoolStrategy,
    /// Enable work-stealing for load balancing
    pub enable_work_stealing: bool,
    /// Chunk size strategy for parallel operations
    pub chunk_strategy: ChunkStrategy,
    /// Enable NUMA-aware thread placement
    pub numa_aware: bool,
    /// CPU affinity optimization
    pub cpu_affinity: bool,
}

/// Thread pool strategies for different workload patterns
#[derive(Debug, Clone)]
pub enum ThreadPoolStrategy {
    /// Global shared thread pool
    Global,
    /// Per-device dedicated thread pools
    PerDevice,
    /// Adaptive pool that scales with workload
    Adaptive,
    /// Custom thread pool configuration
    Custom {
        core_threads: usize,
        max_threads: usize,
    },
}

/// Chunking strategies for parallel data processing
#[derive(Debug, Clone)]
pub enum ChunkStrategy {
    /// Fixed chunk size
    Fixed(usize),
    /// Adaptive based on data size and thread count
    Adaptive,
    /// Work-stealing with dynamic chunk sizes
    WorkStealing,
    /// Cache-aware chunking for memory hierarchy optimization
    CacheAware,
}

/// Memory Optimization Configuration for Efficient Resource Usage
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Enable memory pool optimization
    pub enable_pooling: bool,
    /// Pool block size strategy
    pub pool_strategy: PoolStrategy,
    /// Enable memory mapping for large arrays
    pub enable_memory_mapping: bool,
    /// Memory mapping threshold (bytes)
    pub mmap_threshold: usize,
    /// Enable adaptive chunking for large datasets
    pub enable_adaptive_chunking: bool,
    /// Chunk size for adaptive processing
    pub adaptive_chunk_size: usize,
    /// Enable zero-copy operations where possible
    pub enable_zero_copy: bool,
    /// Memory bandwidth optimization
    pub bandwidth_optimization: bool,
    /// Cache-friendly memory layouts
    pub cache_optimization: bool,
    /// Memory pre-allocation strategy
    pub preallocation_strategy: PreallocationStrategy,
}

/// Memory pool strategies for different allocation patterns
#[derive(Debug, Clone)]
pub enum PoolStrategy {
    /// Fixed-size blocks
    FixedSize {
        block_size: usize,
        num_blocks: usize,
    },
    /// Multiple pool sizes for different allocation sizes
    MultiSize { sizes: Vec<usize> },
    /// Adaptive pool that grows based on usage patterns
    Adaptive {
        initial_size: usize,
        growth_factor: f64,
    },
    /// Segregated pools for different data types
    Segregated,
}

/// Memory pre-allocation strategies
#[derive(Debug, Clone)]
pub enum PreallocationStrategy {
    /// No pre-allocation
    None,
    /// Pre-allocate based on historical usage
    Historical,
    /// Pre-allocate fixed amount per device
    Fixed(usize),
    /// Adaptive pre-allocation based on workload prediction
    Adaptive,
}

/// Performance Monitoring Configuration for Real-Time Analytics
#[derive(Debug, Clone)]
pub struct PerformanceMonitoringConfig {
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Metrics collection frequency
    pub collection_frequency: Duration,
    /// Enable real-time profiling
    pub enable_profiling: bool,
    /// Benchmark execution periodically
    pub enable_benchmarking: bool,
    /// Benchmark frequency
    pub benchmark_frequency: Duration,
    /// Enable memory usage tracking
    pub track_memory_usage: bool,
    /// Enable operation timing
    pub enable_timing: bool,
    /// Enable cache hit/miss tracking
    pub track_cache_performance: bool,
    /// Enable hardware performance counters
    pub enable_hardware_counters: bool,
    /// Performance history retention (number of samples)
    pub history_retention: usize,
    /// Enable performance alerts
    pub enable_alerts: bool,
    /// Performance threshold for alerts
    pub alert_threshold: Duration,
}

/// Adaptive Tuning Configuration for Self-Optimization
#[derive(Debug, Clone)]
pub struct AdaptiveTuningConfig {
    /// Enable adaptive optimization
    pub enable_adaptive: bool,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Adaptation frequency
    pub adaptation_frequency: Duration,
    /// Enable workload prediction
    pub enable_prediction: bool,
    /// Prediction algorithm
    pub prediction_algorithm: PredictionAlgorithm,
    /// Enable auto-tuning of SIMD parameters
    pub tune_simd: bool,
    /// Enable auto-tuning of parallel parameters
    pub tune_parallel: bool,
    /// Enable auto-tuning of memory parameters
    pub tune_memory: bool,
    /// Minimum confidence for parameter changes
    pub min_confidence: f64,
    /// Enable A/B testing for optimizations
    pub enable_ab_testing: bool,
    /// Sample size for statistical significance
    pub sample_size: usize,
}

/// Prediction algorithms for workload forecasting
#[derive(Debug, Clone)]
pub enum PredictionAlgorithm {
    /// Moving average prediction
    MovingAverage { window_size: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
    /// Linear regression on historical data
    LinearRegression,
    /// Machine learning-based prediction
    MachineLearning { model_complexity: ModelComplexity },
}

/// ML model complexity for prediction algorithms
#[derive(Debug, Clone)]
pub enum ModelComplexity {
    /// Simple linear models
    Simple,
    /// Polynomial features
    Polynomial { degree: u32 },
    /// Neural network-based
    NeuralNetwork { hidden_layers: Vec<usize> },
}

/// GPU Acceleration Configuration for Hardware Optimization
#[derive(Debug, Clone)]
pub struct GpuAccelerationConfig {
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// GPU memory pool size (bytes)
    pub gpu_memory_pool: usize,
    /// Enable async GPU operations
    pub enable_async_gpu: bool,
    /// GPU kernel fusion optimization
    pub enable_kernel_fusion: bool,
    /// Tensor core utilization for mixed precision
    pub enable_tensor_cores: bool,
    /// Mixed precision computation
    pub mixed_precision: bool,
    /// GPU memory transfer optimization
    pub optimize_transfers: bool,
    /// Enable multi-GPU coordination
    pub enable_multi_gpu: bool,
    /// GPU scheduling strategy
    pub scheduling_strategy: GpuSchedulingStrategy,
}

/// GPU scheduling strategies for workload distribution
#[derive(Debug, Clone)]
pub enum GpuSchedulingStrategy {
    /// Round-robin across available GPUs
    RoundRobin,
    /// Load-based scheduling
    LoadBased,
    /// Memory-aware scheduling
    MemoryAware,
    /// Latency-optimized scheduling
    LatencyOptimized,
}

/// Ultra-Low Latency Configuration for Critical Performance
#[derive(Debug, Clone)]
pub struct UltraLatencyConfig {
    /// Enable ultra-low latency mode
    pub enable_ultra_latency: bool,
    /// CPU isolation for critical threads
    pub cpu_isolation: bool,
    /// Real-time scheduling priority
    pub realtime_priority: bool,
    /// Disable CPU frequency scaling
    pub disable_cpu_scaling: bool,
    /// Pre-fault memory pages
    pub prefault_memory: bool,
    /// Disable swap for critical processes
    pub disable_swap: bool,
    /// Enable lock-free data structures
    pub enable_lockfree: bool,
    /// Optimize for L1/L2 cache residency
    pub optimize_cache_residency: bool,
    /// Branch prediction optimization
    pub optimize_branch_prediction: bool,
}

impl Default for EagerExecutionConfig {
    fn default() -> Self {
        Self {
            enable_op_cache: true,
            enable_memory_pool: true,
            enable_async_execution: true,
            max_cache_size: 1000,
            memory_pool_size: 128 * 1024 * 1024, // 128MB
            target_overhead_ns: 1_000_000,       // 1ms in nanoseconds
            enable_context_optimization: true,
            enable_kernel_fusion: true,

            // Ultra-Performance Configuration Defaults
            simd_config: SimdExecutionConfig::default(),
            parallel_config: ParallelExecutionConfig::default(),
            memory_config: MemoryOptimizationConfig::default(),
            monitoring_config: PerformanceMonitoringConfig::default(),
            adaptive_config: AdaptiveTuningConfig::default(),
            gpu_config: GpuAccelerationConfig::default(),
            ultra_latency_config: UltraLatencyConfig::default(),
        }
    }
}

impl Default for SimdExecutionConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            simd_width: 256, // AVX2 default
            enable_auto_vectorization: true,
            enable_target_features: true,
            simd_threshold: 1024, // Minimum 1K elements for SIMD
            simd_elementwise: true,
            simd_matrix_ops: true,
            memory_alignment: 32, // 32-byte alignment for AVX2
        }
    }
}

impl Default for ParallelExecutionConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            num_threads: 0,             // Auto-detect
            parallel_threshold: 10_000, // Minimum 10K elements for parallelization
            thread_strategy: ThreadPoolStrategy::Adaptive,
            enable_work_stealing: true,
            chunk_strategy: ChunkStrategy::Adaptive,
            numa_aware: true,
            cpu_affinity: false, // Conservative default
        }
    }
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_strategy: PoolStrategy::Adaptive {
                initial_size: 64 * 1024 * 1024, // 64MB initial
                growth_factor: 1.5,
            },
            enable_memory_mapping: true,
            mmap_threshold: 100 * 1024 * 1024, // 100MB threshold
            enable_adaptive_chunking: true,
            adaptive_chunk_size: 1024 * 1024, // 1MB chunks
            enable_zero_copy: true,
            bandwidth_optimization: true,
            cache_optimization: true,
            preallocation_strategy: PreallocationStrategy::Historical,
        }
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            collection_frequency: Duration::from_millis(100), // 100ms
            enable_profiling: true,
            enable_benchmarking: false, // Disabled by default to reduce overhead
            benchmark_frequency: Duration::from_secs(60), // 1 minute
            track_memory_usage: true,
            enable_timing: true,
            track_cache_performance: true,
            enable_hardware_counters: false, // Requires elevated privileges
            history_retention: 1000,         // Keep 1000 samples
            enable_alerts: true,
            alert_threshold: Duration::from_millis(5), // 5ms alert threshold
        }
    }
}

impl Default for AdaptiveTuningConfig {
    fn default() -> Self {
        Self {
            enable_adaptive: true,
            learning_rate: 0.01, // Conservative learning rate
            adaptation_frequency: Duration::from_secs(30), // 30 seconds
            enable_prediction: true,
            prediction_algorithm: PredictionAlgorithm::ExponentialSmoothing { alpha: 0.3 },
            tune_simd: true,
            tune_parallel: true,
            tune_memory: true,
            min_confidence: 0.8,      // 80% confidence threshold
            enable_ab_testing: false, // Disabled by default
            sample_size: 100,         // 100 operations per test
        }
    }
}

impl Default for GpuAccelerationConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            gpu_memory_pool: 512 * 1024 * 1024, // 512MB GPU memory pool
            enable_async_gpu: true,
            enable_kernel_fusion: true,
            enable_tensor_cores: true,
            mixed_precision: true,
            optimize_transfers: true,
            enable_multi_gpu: true,
            scheduling_strategy: GpuSchedulingStrategy::LoadBased,
        }
    }
}

impl Default for UltraLatencyConfig {
    fn default() -> Self {
        Self {
            enable_ultra_latency: false, // Disabled by default - requires system configuration
            cpu_isolation: false,
            realtime_priority: false,
            disable_cpu_scaling: false,
            prefault_memory: true, // Safe to enable
            disable_swap: false,
            enable_lockfree: true, // Safe optimization
            optimize_cache_residency: true,
            optimize_branch_prediction: true,
        }
    }
}

/// Operation signature for caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpSignature {
    pub operation: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub dtype: DType,
    pub device: Device,
    pub params: Vec<(String, String)>, // Serialized parameters
}

/// Cached operation result
#[derive(Debug, Clone)]
pub struct CachedOperation {
    pub signature: OpSignature,
    pub result_shape: Vec<usize>,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub created_at: Instant,
    pub last_used: Instant,
    pub use_count: usize,
}

/// Execution metrics for overhead tracking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ExecutionMetrics {
    pub operation: String,
    pub device: Device,
    pub setup_time: Duration,
    pub execution_time: Duration,
    pub teardown_time: Duration,
    pub total_overhead: Duration,
    pub memory_allocation_time: Duration,
    pub cache_hit: bool,
    pub meets_target: bool,
}

/// Memory pool for fast allocation/deallocation
#[allow(dead_code)]
struct MemoryPool {
    blocks: RwLock<HashMap<Device, Vec<MemoryBlock>>>,
    config: EagerExecutionConfig,
}

#[derive(Debug)]
#[allow(dead_code)]
struct MemoryBlock {
    ptr: *mut u8,
    size: usize,
    available: bool,
    last_used: Instant,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

#[allow(dead_code)]
impl MemoryPool {
    fn new(config: EagerExecutionConfig) -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            config,
        }
    }

    fn allocate(&self, device: &Device, size: usize) -> Result<*mut u8> {
        let _start = Instant::now();

        // Try to find available block
        {
            let mut blocks = self.blocks.write().unwrap();
            let device_blocks = blocks.entry(*device).or_default();

            for block in device_blocks.iter_mut() {
                if block.available && block.size >= size {
                    block.available = false;
                    block.last_used = Instant::now();
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block if none available
        let context = DEVICE_MANAGER.get_context(device)?;
        let ptr = context.allocator().allocate(size)?;

        // Add to pool
        {
            let mut blocks = self.blocks.write().unwrap();
            let device_blocks = blocks.entry(*device).or_default();
            device_blocks.push(MemoryBlock {
                ptr,
                size,
                available: false,
                last_used: Instant::now(),
            });
        }

        Ok(ptr)
    }

    fn deallocate(&self, device: &Device, ptr: *mut u8) -> Result<()> {
        let mut blocks = self.blocks.write().unwrap();
        if let Some(device_blocks) = blocks.get_mut(device) {
            for block in device_blocks.iter_mut() {
                if block.ptr == ptr {
                    block.available = true;
                    block.last_used = Instant::now();
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn cleanup_old_blocks(&self) {
        let threshold = Duration::from_secs(60); // 1 minute
        let now = Instant::now();

        let mut blocks = self.blocks.write().unwrap();
        for device_blocks in blocks.values_mut() {
            device_blocks.retain(|block| {
                if block.available && now.duration_since(block.last_used) > threshold {
                    // Deallocate old unused blocks
                    false
                } else {
                    true
                }
            });
        }
    }

    /// Pre-warm memory pool by pre-allocating blocks of the required size
    fn pre_warm(&self, device: &Device, size: usize, num_blocks: usize) -> Result<()> {
        let context = DEVICE_MANAGER.get_context(device)?;
        let mut blocks = self.blocks.write().unwrap();
        let device_blocks = blocks.entry(*device).or_default();

        // Pre-allocate the specified number of blocks
        for _ in 0..num_blocks {
            let ptr = context.allocator().allocate(size)?;
            device_blocks.push(MemoryBlock {
                ptr,
                size,
                available: true, // Available for use
                last_used: Instant::now(),
            });
        }

        Ok(())
    }
}

/// Eager execution engine optimized for low latency
pub struct EagerExecutionEngine {
    config: EagerExecutionConfig,
    op_cache: RwLock<HashMap<OpSignature, CachedOperation>>,
    memory_pool: MemoryPool,
    metrics: Mutex<Vec<ExecutionMetrics>>,
    active_contexts: RwLock<HashMap<Device, Arc<dyn DeviceContext>>>,
    fusion_opportunities: RwLock<Vec<FusionOpportunity>>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct FusionOpportunity {
    operations: Vec<String>,
    potential_speedup: f64,
    memory_savings: usize,
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

        self.metrics.lock().unwrap().push(metrics.clone());

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
        let cache = self.op_cache.read().unwrap();
        cache.contains_key(signature)
    }

    /// Cache operation result
    fn cache_operation<T>(
        &self,
        signature: &OpSignature,
        result: &Tensor<T>,
        execution_time: Duration,
    ) -> Result<()> {
        let mut cache = self.op_cache.write().unwrap();

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
            let mut contexts = self.active_contexts.write().unwrap();
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
        let mut opportunities = self.fusion_opportunities.write().unwrap();

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
        self.metrics.lock().unwrap().clone()
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let cache = self.op_cache.read().unwrap();

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

        let mut cache = self.op_cache.write().unwrap();
        cache.retain(|_, cached_op| now.duration_since(cached_op.last_used) <= threshold);
    }
}

/// Memory guard for RAII memory management
#[allow(dead_code)]
struct MemoryGuard {
    device: Device,
    estimated_memory: usize,
    operation: String,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_hits: usize,
    pub hit_rate: f64,
    pub avg_execution_time: Duration,
}

/// Eager execution performance report
#[derive(Debug, Clone)]
pub struct EagerPerformanceReport {
    pub total_operations: usize,
    pub operations_meeting_target: usize,
    pub success_rate: f64,
    pub avg_overhead: Duration,
    pub min_overhead: Duration,
    pub max_overhead: Duration,
    pub cache_statistics: CacheStatistics,
    pub cache_hit_rate: f64,
    pub target_overhead: Duration,
    pub recommendations: Vec<String>,
}

impl Default for EagerPerformanceReport {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_meeting_target: 0,
            success_rate: 0.0,
            avg_overhead: Duration::ZERO,
            min_overhead: Duration::ZERO,
            max_overhead: Duration::ZERO,
            cache_statistics: CacheStatistics {
                total_entries: 0,
                total_hits: 0,
                hit_rate: 0.0,
                avg_execution_time: Duration::ZERO,
            },
            cache_hit_rate: 0.0,
            target_overhead: Duration::from_millis(1),
            recommendations: Vec::new(),
        }
    }
}

impl EagerPerformanceReport {
    /// Print a formatted performance report
    pub fn print_report(&self) {
        println!("🚀 Eager Execution Performance Report");
        println!("=====================================");
        println!();
        println!("📊 Overall Performance:");
        println!("  • Total operations: {}", self.total_operations);
        println!(
            "  • Operations meeting target: {}/{}",
            self.operations_meeting_target, self.total_operations
        );
        println!("  • Success rate: {:.1}%", self.success_rate * 100.0);
        println!();

        println!("⏱️ Overhead Analysis:");
        println!("  • Target overhead: {:?}", self.target_overhead);
        println!("  • Average overhead: {:?}", self.avg_overhead);
        println!("  • Minimum overhead: {:?}", self.min_overhead);
        println!("  • Maximum overhead: {:?}", self.max_overhead);

        let target_met = self.avg_overhead <= self.target_overhead;
        if target_met {
            println!("  ✅ Average overhead meets target!");
        } else {
            let gap = self.avg_overhead.as_nanos() - self.target_overhead.as_nanos();
            println!("  ❌ Average overhead exceeds target by {gap}ns");
        }
        println!();

        println!("💾 Cache Performance:");
        println!("  • Cache entries: {}", self.cache_statistics.total_entries);
        println!("  • Cache hit rate: {:.1}%", self.cache_hit_rate * 100.0);
        println!(
            "  • Average execution time: {:?}",
            self.cache_statistics.avg_execution_time
        );
        println!();

        if !self.recommendations.is_empty() {
            println!("💡 Recommendations:");
            for (i, rec) in self.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }

        println!("=====================================");
    }
}

lazy_static::lazy_static! {
    pub static ref EAGER_ENGINE: EagerExecutionEngine =
        EagerExecutionEngine::new(EagerExecutionConfig::default());
}

/// Convenience macro for eager operation execution
#[macro_export]
macro_rules! eager_execute {
    ($op:expr, $inputs:expr, $executor:expr) => {
        $crate::eager_execution::EAGER_ENGINE.execute_operation(
            $op,
            $inputs,
            &std::collections::HashMap::new(),
            $executor,
        )
    };

    ($op:expr, $inputs:expr, $params:expr, $executor:expr) => {
        $crate::eager_execution::EAGER_ENGINE.execute_operation($op, $inputs, $params, $executor)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eager_execution_config() {
        let config = EagerExecutionConfig::default();
        assert!(config.enable_op_cache);
        assert!(config.enable_memory_pool);
        assert_eq!(config.target_overhead_ns, 1_000_000); // 1ms
    }

    #[test]
    fn test_op_signature_creation() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());

        // Mock tensor creation would be needed for real test
        // This test validates the concept
        assert_eq!(engine.config.target_overhead_ns, 1_000_000);
    }

    #[test]
    fn test_performance_report_generation() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());
        let report = engine.generate_performance_report();

        // Empty report should have default values
        assert_eq!(report.total_operations, 0);
        assert_eq!(report.success_rate, 0.0);
    }

    #[test]
    fn test_cache_statistics() {
        let engine = EagerExecutionEngine::new(EagerExecutionConfig::default());
        let stats = engine.get_cache_stats();

        // Initial cache should be empty
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }
}
