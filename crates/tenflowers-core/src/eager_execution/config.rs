//! Configuration structs and enums for Ultra-Performance Eager Execution

use std::time::Duration;

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
    pub dtype: crate::DType,
    pub device: crate::Device,
    pub params: Vec<(String, String)>, // Serialized parameters
}

/// Cached operation result
#[derive(Debug, Clone)]
pub struct CachedOperation {
    pub signature: OpSignature,
    pub result_shape: Vec<usize>,
    pub execution_time: std::time::Duration,
    pub memory_usage: usize,
    pub created_at: std::time::Instant,
    pub last_used: std::time::Instant,
    pub use_count: usize,
}

/// Execution metrics for overhead tracking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ExecutionMetrics {
    pub operation: String,
    pub device: crate::Device,
    pub setup_time: std::time::Duration,
    pub execution_time: std::time::Duration,
    pub teardown_time: std::time::Duration,
    pub total_overhead: std::time::Duration,
    pub memory_allocation_time: std::time::Duration,
    pub cache_hit: bool,
    pub meets_target: bool,
}
