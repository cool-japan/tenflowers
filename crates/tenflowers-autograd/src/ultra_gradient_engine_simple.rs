//! Simplified Ultra-High-Performance Gradient Computation Engine
//!
//! This module provides ultra-fast gradient computation with maximum performance
//! optimizations while maintaining compatibility with all tensor operations.

use crate::tape::{GradientTape, TrackedTensor};
use scirs2_core::numeric::{Float, FromPrimitive, One, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

// Use available SciRS2-Core features
use scirs2_core::profiling::Profiler;

/// Ultra-high-performance gradient computation engine
pub struct UltraGradientEngine {
    /// Performance profiler for optimization tracking
    #[allow(dead_code)]
    profiler: Arc<Profiler>,
    /// Gradient computation cache for repeated operations
    gradient_cache: Arc<Mutex<HashMap<String, CachedGradient>>>,
    /// Configuration for ultra-performance
    config: UltraGradientConfig,
}

/// Configuration for ultra-high-performance gradient computation
#[derive(Debug, Clone)]
pub struct UltraGradientConfig {
    /// Enable gradient caching for repeated computations
    pub enable_gradient_caching: bool,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Gradient cache capacity
    pub gradient_cache_capacity: usize,
    /// Performance optimization threshold
    pub optimization_threshold: usize,
}

/// A single gradient tensor serialized to raw element bytes plus its shape, so
/// it can be faithfully reconstructed on a cache hit. `None` records a gradient
/// slot that was absent (no gradient flowed to that source).
#[derive(Debug, Clone)]
struct SerializedGradient {
    /// Shape of the original gradient tensor.
    shape: Vec<usize>,
    /// Raw little-endian element bytes (`bytemuck::cast_slice` of the data).
    bytes: Vec<u8>,
    /// Size in bytes of a single element of the original element type. Used to
    /// reject deserialization against a mismatched element type.
    element_size: usize,
}

/// Cached gradient computation for performance optimization
#[derive(Debug, Clone)]
struct CachedGradient {
    /// Gradient computation hash
    hash: u64,
    /// Cached gradients, one slot per source. Each slot is `Some` with the real
    /// serialized gradient bytes, or `None` if no gradient existed for that source.
    gradients: Vec<Option<SerializedGradient>>,
    /// Cache hit count for optimization
    hit_count: usize,
    /// Last access time for cache eviction
    last_access: std::time::Instant,
}

/// Ultra-fast gradient computation result
#[derive(Debug)]
pub struct UltraGradientResult<T> {
    /// Computed gradients with maximum efficiency
    pub gradients: Vec<Option<Tensor<T>>>,
    /// Detailed performance metrics
    pub performance_metrics: GradientPerformanceMetrics,
    /// Memory usage statistics
    pub memory_stats: GradientMemoryStats,
    /// Optimization insights
    pub optimization_insights: OptimizationInsights,
}

/// Comprehensive performance metrics for gradient computation
#[derive(Debug, Default)]
pub struct GradientPerformanceMetrics {
    /// Total computation time
    pub total_time: std::time::Duration,
    /// Time spent on gradient computation
    pub gradient_compute_time: std::time::Duration,
    /// Time spent on memory operations
    pub memory_operation_time: std::time::Duration,
    /// Number of operations processed
    pub operations_count: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
}

/// Memory usage statistics for gradient computation
#[derive(Debug, Default)]
pub struct GradientMemoryStats {
    /// Peak memory usage during computation
    pub peak_memory_usage: usize,
    /// Total memory allocated
    pub total_memory_allocated: usize,
    /// Memory reused from optimization
    pub memory_reused: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Memory optimization efficiency
    pub optimization_efficiency: f64,
}

/// Optimization insights for continuous performance improvement
#[derive(Debug, Default)]
pub struct OptimizationInsights {
    /// Recommended optimizations
    pub recommendations: Vec<String>,
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Potential speedup estimates
    pub potential_speedup: f64,
    /// Memory optimization opportunities
    pub memory_optimizations: Vec<String>,
}

impl UltraGradientEngine {
    /// Create a new ultra-high-performance gradient engine
    pub fn new(config: UltraGradientConfig) -> Result<Self> {
        let profiler = Arc::new(Profiler::new());
        let gradient_cache = Arc::new(Mutex::new(HashMap::with_capacity(
            config.gradient_cache_capacity,
        )));

        Ok(Self {
            profiler,
            gradient_cache,
            config,
        })
    }

    /// Compute gradients with maximum performance optimization
    pub fn compute_gradients_ultra<T>(
        &self,
        tape: &GradientTape,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<UltraGradientResult<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Neg<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + PartialOrd
            + Float
            + FromPrimitive
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let _profiling_session = self.config.enable_performance_monitoring;

        let start_time = std::time::Instant::now();

        let mut performance_metrics = GradientPerformanceMetrics::default();
        let mut memory_stats = GradientMemoryStats::default();
        let mut optimization_insights = OptimizationInsights::default();

        // Step 1: Check cache for repeated computations
        let cache_key = self.generate_cache_key(targets, sources);
        if self.config.enable_gradient_caching {
            if let Some(cached_result) = self.try_get_from_cache::<T>(&cache_key)? {
                performance_metrics.cache_hits = 1;
                performance_metrics.total_time = start_time.elapsed();
                return Ok(UltraGradientResult {
                    gradients: cached_result,
                    performance_metrics,
                    memory_stats,
                    optimization_insights,
                });
            }
        }

        // Step 2: Ultra-fast gradient computation using optimized tape
        let gradient_compute_start = std::time::Instant::now();
        let result_gradients = tape.gradient(targets, sources)?;
        performance_metrics.gradient_compute_time = gradient_compute_start.elapsed();

        // Step 3: Cache results for future use
        if self.config.enable_gradient_caching {
            self.cache_gradients(&cache_key, &result_gradients)?;
        }

        // Step 4: Collect comprehensive performance metrics
        performance_metrics.total_time = start_time.elapsed();
        performance_metrics.operations_count = targets.len() + sources.len();
        performance_metrics.cache_misses = 1;

        self.collect_memory_stats(&mut memory_stats)?;
        self.generate_optimization_insights(
            &performance_metrics,
            &memory_stats,
            &mut optimization_insights,
        )?;

        Ok(UltraGradientResult {
            gradients: result_gradients,
            performance_metrics,
            memory_stats,
            optimization_insights,
        })
    }

    /// Get comprehensive performance statistics
    ///
    /// Note: the timing fields (`total_time`, `gradient_compute_time`,
    /// `memory_operation_time`) are NOT populated here — they are measured
    /// per-invocation inside [`Self::compute_gradients_ultra`] and reported on its
    /// returned [`UltraGradientResult`]. This accessor reports only the
    /// cache-derived figures it can actually measure; the timing fields are left
    /// at zero ("not measured") rather than filled with invented constants.
    pub fn get_performance_statistics(&self) -> Result<GradientPerformanceMetrics> {
        // Aggregate performance statistics from the gradient cache.
        let mut metrics = GradientPerformanceMetrics::default();

        // Get cache statistics
        if let Ok(cache) = self.gradient_cache.lock() {
            let total_entries = cache.len();
            let total_hits: usize = cache.values().map(|cached| cached.hit_count).sum();
            metrics.cache_hits = total_hits;

            if total_entries > 0 {
                metrics.efficiency_score = total_hits as f64 / total_entries as f64;
            }
        }

        Ok(metrics)
    }

    /// Optimize the gradient engine for better performance
    pub fn optimize(&self) -> Result<OptimizationInsights> {
        let _session = self.config.enable_performance_monitoring;

        let mut insights = OptimizationInsights::default();

        // Analyze cache performance
        if let Ok(cache) = self.gradient_cache.lock() {
            let cache_size = cache.len();
            let total_hits: usize = cache.values().map(|cached| cached.hit_count).sum();

            if cache_size > 0 {
                let hit_rate = total_hits as f64 / cache_size as f64;

                if hit_rate < 0.5 {
                    insights
                        .recommendations
                        .push("Consider increasing gradient cache size".to_string());
                    insights.potential_speedup += 0.2;
                }

                if hit_rate > 0.8 {
                    insights.recommendations.push(
                        "Excellent cache performance - consider enabling more aggressive caching"
                            .to_string(),
                    );
                }
            }
        }

        // Performance optimization recommendations
        insights
            .recommendations
            .push("Enable memory optimization for better performance".to_string());
        insights
            .recommendations
            .push("Consider enabling performance monitoring for detailed insights".to_string());

        insights.potential_speedup += 0.1; // Conservative estimate

        Ok(insights)
    }

    // Private implementation methods

    fn generate_cache_key<T>(
        &self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> String {
        // Generate a simple cache key based on tensor IDs
        let target_ids: Vec<usize> = targets.iter().map(|t| t.id).collect();
        let source_ids: Vec<usize> = sources.iter().map(|s| s.id).collect();

        format!("targets:{:?}-sources:{:?}", target_ids, source_ids)
    }

    fn try_get_from_cache<T>(&self, cache_key: &str) -> Result<Option<Vec<Option<Tensor<T>>>>>
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
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let mut cache = match self.gradient_cache.lock() {
            Ok(cache) => cache,
            // A poisoned lock is not a cache miss; surface it honestly.
            Err(_) => {
                return Err(TensorError::compute_error_simple(
                    "gradient cache lock poisoned".to_string(),
                ))
            }
        };

        let Some(cached) = cache.get_mut(cache_key) else {
            return Ok(None);
        };

        // Defensive: confirm the stored entry was keyed by this exact key.
        if cached.hash != self.hash_string(cache_key) {
            return Ok(None);
        }

        cached.hit_count += 1;
        cached.last_access = std::time::Instant::now();

        let element_size = std::mem::size_of::<T>();
        let mut gradients: Vec<Option<Tensor<T>>> = Vec::with_capacity(cached.gradients.len());

        for slot in &cached.gradients {
            match slot {
                None => gradients.push(None),
                Some(serialized) => {
                    // Reject deserialization against an element type whose size
                    // differs from the one the bytes were produced with.
                    if serialized.element_size != element_size {
                        return Err(TensorError::compute_error_simple(format!(
                            "cached gradient element size {} does not match requested type size {}",
                            serialized.element_size, element_size
                        )));
                    }
                    if element_size == 0 || serialized.bytes.len() % element_size != 0 {
                        return Err(TensorError::compute_error_simple(
                            "cached gradient byte length is not a multiple of the element size"
                                .to_string(),
                        ));
                    }
                    let values: &[T] = bytemuck::cast_slice(&serialized.bytes);
                    let tensor = Tensor::from_vec(values.to_vec(), &serialized.shape)?;
                    gradients.push(Some(tensor));
                }
            }
        }

        Ok(Some(gradients))
    }

    fn cache_gradients<T>(&self, cache_key: &str, gradients: &[Option<Tensor<T>>]) -> Result<()>
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
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let element_size = std::mem::size_of::<T>();

        // Serialize the real gradients into raw element bytes plus shape so they
        // can be reconstructed faithfully on a cache hit.
        let mut serialized_gradients: Vec<Option<SerializedGradient>> =
            Vec::with_capacity(gradients.len());
        for gradient in gradients {
            match gradient {
                None => serialized_gradients.push(None),
                Some(tensor) => {
                    // Gradient data must be host-accessible to be serialized.
                    let data = tensor.as_slice().ok_or_else(|| {
                        TensorError::unsupported_operation_simple(
                            "cannot cache a gradient whose data is not host-accessible \
                             (move it to CPU before caching)"
                                .to_string(),
                        )
                    })?;
                    let bytes = bytemuck::cast_slice::<T, u8>(data).to_vec();
                    serialized_gradients.push(Some(SerializedGradient {
                        shape: tensor.shape().dims().to_vec(),
                        bytes,
                        element_size,
                    }));
                }
            }
        }

        let mut cache = match self.gradient_cache.lock() {
            Ok(cache) => cache,
            Err(_) => {
                return Err(TensorError::compute_error_simple(
                    "gradient cache lock poisoned".to_string(),
                ))
            }
        };

        let cached_gradient = CachedGradient {
            hash: self.hash_string(cache_key),
            gradients: serialized_gradients,
            hit_count: 1,
            last_access: std::time::Instant::now(),
        };

        cache.insert(cache_key.to_string(), cached_gradient);

        // Cleanup old entries if cache is full
        if cache.len() > self.config.gradient_cache_capacity {
            self.cleanup_cache(&mut cache)?;
        }

        Ok(())
    }

    fn cleanup_cache(&self, cache: &mut HashMap<String, CachedGradient>) -> Result<()> {
        let now = std::time::Instant::now();
        let cutoff = std::time::Duration::from_secs(300); // 5 minutes

        cache.retain(|_, cached| {
            now.duration_since(cached.last_access) < cutoff || cached.hit_count > 5
        });

        Ok(())
    }

    fn collect_memory_stats(&self, memory_stats: &mut GradientMemoryStats) -> Result<()> {
        // Pull REAL figures from the crate's gradient memory profiler rather than
        // inventing constants. The profiler tracks live allocations, peak usage,
        // pool reuse, fragmentation and pool efficiency.
        let profiler = crate::memory_profiler::get_global_profiler();
        let stats = {
            let guard = profiler.lock().map_err(|_| {
                TensorError::compute_error_simple(
                    "gradient memory profiler lock poisoned".to_string(),
                )
            })?;
            guard.get_stats()?
        };

        // Map measured quantities onto the gradient memory-stats fields.
        // - peak_memory_usage: high-water mark of bytes in use.
        // - total_memory_allocated: bytes currently allocated (live working set).
        // - memory_reused: bytes served from buffer-pool reuse.
        // - fragmentation_ratio / optimization_efficiency: as measured by the pool.
        memory_stats.peak_memory_usage = stats.peak_memory as usize;
        memory_stats.total_memory_allocated = stats.current_memory as usize;
        memory_stats.memory_reused = stats.pool_statistics.pool_allocated as usize;
        memory_stats.fragmentation_ratio = stats.fragmentation_ratio;
        memory_stats.optimization_efficiency = stats.pool_statistics.efficiency_ratio;

        Ok(())
    }

    fn generate_optimization_insights(
        &self,
        performance_metrics: &GradientPerformanceMetrics,
        memory_stats: &GradientMemoryStats,
        insights: &mut OptimizationInsights,
    ) -> Result<()> {
        // Analyze performance bottlenecks
        if performance_metrics.memory_operation_time > performance_metrics.gradient_compute_time {
            insights
                .bottlenecks
                .push("Memory operations are the bottleneck".to_string());
            insights
                .recommendations
                .push("Enable memory optimization".to_string());
        }

        if memory_stats.fragmentation_ratio > 0.2 {
            insights
                .memory_optimizations
                .push("Reduce memory fragmentation with better allocation strategies".to_string());
        }

        if performance_metrics.cache_hits == 0 && performance_metrics.cache_misses > 0 {
            insights
                .recommendations
                .push("Enable gradient caching for better performance".to_string());
            insights.potential_speedup += 0.3;
        }

        // Estimate potential speedup
        insights.potential_speedup += (1.0 - memory_stats.optimization_efficiency) * 0.5;

        Ok(())
    }

    fn hash_string(&self, s: &str) -> u64 {
        // Simple hash function for cache keys
        s.chars()
            .fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64))
    }
}

impl Default for UltraGradientConfig {
    fn default() -> Self {
        Self {
            enable_gradient_caching: true,
            enable_performance_monitoring: true,
            enable_memory_optimization: true,
            gradient_cache_capacity: 1000,
            optimization_threshold: 100,
        }
    }
}

/// Global ultra-gradient engine instance for maximum performance
static GLOBAL_ULTRA_GRADIENT_ENGINE: std::sync::OnceLock<Arc<Mutex<UltraGradientEngine>>> =
    std::sync::OnceLock::new();

/// Get the global ultra-gradient engine
pub fn global_ultra_gradient_engine() -> Arc<Mutex<UltraGradientEngine>> {
    GLOBAL_ULTRA_GRADIENT_ENGINE
        .get_or_init(|| {
            let config = UltraGradientConfig::default();
            let engine =
                UltraGradientEngine::new(config).expect("Failed to create ultra gradient engine");
            Arc::new(Mutex::new(engine))
        })
        .clone()
}

/// Extension trait for ultra-fast gradient computation
pub trait UltraGradientTapeExt {
    /// Compute gradients with maximum performance
    fn gradient_ultra<T>(
        &self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<UltraGradientResult<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Neg<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + PartialOrd
            + Float
            + FromPrimitive
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable;
}

impl UltraGradientTapeExt for GradientTape {
    fn gradient_ultra<T>(
        &self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<UltraGradientResult<T>>
    where
        T: Clone
            + Default
            + Zero
            + One
            + Send
            + Sync
            + 'static
            + std::ops::Add<Output = T>
            + std::ops::Neg<Output = T>
            + std::ops::Div<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Sub<Output = T>
            + PartialOrd
            + Float
            + FromPrimitive
            + scirs2_core::num_traits::Signed
            + bytemuck::Pod
            + bytemuck::Zeroable,
    {
        let engine = global_ultra_gradient_engine();
        let engine = engine.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock ultra gradient engine".to_string())
        })?;

        engine.compute_gradients_ultra(self, targets, sources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_ultra_gradient_engine_creation() {
        let config = UltraGradientConfig::default();
        let engine = UltraGradientEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_ultra_gradient_computation() {
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::<f32>::ones(&[2, 2]));
        let y = tape.watch(Tensor::<f32>::ones(&[2, 2]));

        #[allow(clippy::cloned_ref_to_slice_refs)]
        let result = tape.gradient_ultra(&[x.clone()], &[x, y]);
        assert!(result.is_ok());

        let result = result.expect("test: operation result should be valid");
        assert_eq!(result.gradients.len(), 2);
        assert!(result.performance_metrics.total_time.as_nanos() > 0);
    }

    #[test]
    fn test_ultra_gradient_config() {
        let config = UltraGradientConfig {
            enable_gradient_caching: false,
            enable_performance_monitoring: false,
            gradient_cache_capacity: 500,
            ..Default::default()
        };

        assert!(!config.enable_gradient_caching);
        assert!(!config.enable_performance_monitoring);
        assert_eq!(config.gradient_cache_capacity, 500);
    }

    #[test]
    fn test_global_ultra_gradient_engine() {
        let engine1 = global_ultra_gradient_engine();
        let engine2 = global_ultra_gradient_engine();

        // Should be the same instance
        assert!(Arc::ptr_eq(&engine1, &engine2));
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = GradientPerformanceMetrics::default();
        assert_eq!(metrics.operations_count, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
    }

    #[test]
    fn test_optimization_insights() {
        let config = UltraGradientConfig::default();
        let engine =
            UltraGradientEngine::new(config).expect("test: gradient computation should succeed");

        let insights = engine.optimize();
        assert!(insights.is_ok());

        let insights = insights.expect("test: operation should succeed");
        assert!(!insights.recommendations.is_empty());
    }

    #[test]
    fn test_gradient_cache_round_trip_serializes_real_gradients() {
        let engine = UltraGradientEngine::new(UltraGradientConfig::default())
            .expect("test: engine creation should succeed");

        // Real, distinct gradient tensors plus an absent slot.
        let grad_a = Tensor::<f32>::from_vec(vec![1.0, -2.0, 3.5, 0.25], &[2, 2])
            .expect("test: tensor construction should succeed");
        let grad_b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3])
            .expect("test: tensor construction should succeed");
        let gradients: Vec<Option<Tensor<f32>>> =
            vec![Some(grad_a.clone()), None, Some(grad_b.clone())];

        let key = "test-key";
        engine
            .cache_gradients(key, &gradients)
            .expect("test: caching real gradients should succeed");

        let restored = engine
            .try_get_from_cache::<f32>(key)
            .expect("test: cache lookup should succeed")
            .expect("test: a cached entry must be returned, not a silent None");

        assert_eq!(restored.len(), 3);

        // Slot 0: exact byte-faithful round trip.
        let r0 = restored[0]
            .as_ref()
            .expect("test: slot 0 gradient should be present");
        assert_eq!(r0.shape().dims(), &[2, 2]);
        assert_eq!(
            r0.as_slice().expect("test: host-accessible"),
            grad_a.as_slice().expect("test: host-accessible")
        );

        // Slot 1: genuinely absent gradient stays absent.
        assert!(restored[1].is_none());

        // Slot 2: exact round trip with a different shape.
        let r2 = restored[2]
            .as_ref()
            .expect("test: slot 2 gradient should be present");
        assert_eq!(r2.shape().dims(), &[3]);
        assert_eq!(
            r2.as_slice().expect("test: host-accessible"),
            grad_b.as_slice().expect("test: host-accessible")
        );
    }

    #[test]
    fn test_collect_memory_stats_are_not_fabricated_constants() {
        let engine = UltraGradientEngine::new(UltraGradientConfig::default())
            .expect("test: engine creation should succeed");

        let mut stats = GradientMemoryStats::default();
        engine
            .collect_memory_stats(&mut stats)
            .expect("test: collecting real memory stats should succeed");

        // The previous implementation always reported these invented constants.
        // Real profiler data must NOT reproduce that exact fabricated triple.
        let is_old_fabrication = stats.total_memory_allocated == 1_000_000
            && stats.peak_memory_usage == 1_200_000
            && stats.memory_reused == 500_000
            && (stats.fragmentation_ratio - 0.1).abs() < f64::EPSILON
            && (stats.optimization_efficiency - 0.9).abs() < f64::EPSILON;
        assert!(
            !is_old_fabrication,
            "memory stats must come from the real profiler, not hardcoded constants"
        );

        // And they must be internally consistent with the profiler's own readout.
        let profiler = crate::memory_profiler::get_global_profiler();
        let measured = {
            let guard = profiler.lock().expect("test: profiler lock");
            guard.get_stats().expect("test: profiler stats")
        };
        assert_eq!(stats.peak_memory_usage, measured.peak_memory as usize);
        assert_eq!(stats.fragmentation_ratio, measured.fragmentation_ratio);
    }
}
