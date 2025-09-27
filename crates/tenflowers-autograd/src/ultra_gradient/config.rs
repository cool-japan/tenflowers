//! Configuration structures for ultra-high-performance gradient computation

use std::time::SystemTime;

/// Chunk strategy for parallel operations
#[derive(Debug, Clone)]
pub enum ChunkStrategy {
    Fixed(usize),
    Adaptive,
    Dynamic,
}

/// Configuration for ultra-high-performance gradient computation
#[derive(Debug, Clone)]
pub struct UltraGradientConfig {
    /// Enable kernel fusion for maximum throughput
    pub enable_kernel_fusion: bool,
    /// Enable SIMD acceleration for gradient operations
    pub enable_simd_acceleration: bool,
    /// Enable gradient caching for repeated computations
    pub enable_gradient_caching: bool,
    /// Enable asynchronous gradient computation
    pub enable_async_gradients: bool,
    /// Enable memory optimization with buffer reuse
    pub enable_memory_optimization: bool,
    /// Enable advanced parallelization strategies
    pub enable_advanced_parallelization: bool,
    /// Maximum number of parallel workers
    pub max_parallel_workers: usize,
    /// Chunk strategy for parallel operations
    pub chunk_strategy: ChunkStrategy,
    /// Gradient cache capacity
    pub gradient_cache_capacity: usize,
}

impl Default for UltraGradientConfig {
    fn default() -> Self {
        Self {
            enable_kernel_fusion: true,
            enable_simd_acceleration: true,
            enable_gradient_caching: true,
            enable_async_gradients: true,
            enable_memory_optimization: true,
            enable_advanced_parallelization: true,
            max_parallel_workers: 4, // Default to 4 workers
            chunk_strategy: ChunkStrategy::Adaptive,
            gradient_cache_capacity: 1024,
        }
    }
}

/// Cached gradient computation for performance optimization
#[derive(Debug, Clone)]
pub struct CachedGradient {
    /// Gradient computation hash
    pub hash: String,
    /// Cached gradient tensors
    pub gradients: Vec<Vec<f64>>,
    /// Cache hit count for optimization
    pub hit_count: usize,
    /// Last access time for cache eviction
    pub last_access: SystemTime,
}
