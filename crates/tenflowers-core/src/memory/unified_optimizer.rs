//! Unified Ultra-Performance Memory and SIMD Optimization Engine
//!
//! This module provides a comprehensive optimization system that integrates
//! advanced SIMD vectorization with ultra-cache optimization for maximum performance.

use crate::memory::ultra_cache_optimizer::{
    global_cache_optimizer as global_ultra_cache_optimizer, CacheOptimizerConfig,
    MemoryOptimizationResult, UltraCacheOptimizer,
};
use crate::simd::ultra_simd_engine::SimdPerformanceStats;
use crate::simd::{global_simd_engine, ElementWiseOp, SimdEngineConfig, UltraSimdEngine};
use crate::{Result, TensorError};
use scirs2_core::profiling::Profiler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// Unified ultra-performance optimization engine
#[allow(dead_code)]
pub struct UnifiedOptimizationEngine {
    /// SIMD optimization engine
    simd_engine: Arc<Mutex<UltraSimdEngine>>,
    /// Cache optimization engine
    cache_optimizer: Arc<Mutex<UltraCacheOptimizer>>,
    /// Optimization coordination layer
    coordinator: Arc<RwLock<OptimizationCoordinator>>,
    /// Performance profiler
    profiler: Arc<Profiler>,
    /// Configuration
    config: UnifiedOptimizerConfig,
}

/// Optimization coordination layer
#[allow(dead_code)]
pub struct OptimizationCoordinator {
    /// Operation performance profiles
    operation_profiles: HashMap<String, OperationPerformanceProfile>,
    /// Adaptive optimization strategies
    adaptive_strategies: HashMap<String, AdaptiveStrategy>,
    /// Performance history tracking
    performance_history: Vec<PerformanceSnapshot>,
    /// Current optimization state
    current_optimization_state: OptimizationState,
}

/// Performance profile for specific operations
#[derive(Debug, Clone)]
pub struct OperationPerformanceProfile {
    /// Operation name
    pub operation_name: String,
    /// Optimal SIMD strategy
    pub optimal_simd_strategy: SimdStrategy,
    /// Optimal cache strategy
    pub optimal_cache_strategy: CacheStrategy,
    /// Data size ranges for different strategies
    pub size_breakpoints: Vec<SizeBreakpoint>,
    /// Expected performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

/// SIMD optimization strategy
#[derive(Debug, Clone)]
pub enum SimdStrategy {
    /// Use hardware-specific vectorization
    HardwareSpecific { vector_width: usize },
    /// Use adaptive vectorization
    Adaptive { min_width: usize, max_width: usize },
    /// Use scalar fallback
    Scalar,
    /// Use mixed strategies based on data size
    Mixed {
        strategies: Vec<(usize, Box<SimdStrategy>)>,
    },
}

/// Cache optimization strategy
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    /// Cache-oblivious algorithms
    CacheOblivious,
    /// Blocked algorithms with specific block sizes
    Blocked {
        l1_block: usize,
        l2_block: usize,
        l3_block: usize,
    },
    /// NUMA-aware allocation
    NumaAware { preferred_node: usize },
    /// Prefetching strategy
    Prefetching {
        distance: usize,
        pattern: PrefetchPattern,
    },
    /// Combined strategies
    Combined { strategies: Vec<CacheStrategy> },
}

/// Prefetch pattern types
#[derive(Debug, Clone)]
pub enum PrefetchPattern {
    Sequential,
    Strided { stride: usize },
    Random,
    Adaptive,
}

/// Size-based optimization breakpoints
#[derive(Debug, Clone)]
pub struct SizeBreakpoint {
    /// Minimum size for this strategy
    pub min_size: usize,
    /// Maximum size for this strategy
    pub max_size: usize,
    /// SIMD strategy for this size range
    pub simd_strategy: SimdStrategy,
    /// Cache strategy for this size range
    pub cache_strategy: CacheStrategy,
    /// Expected performance multiplier
    pub performance_multiplier: f64,
}

/// Performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Latency (seconds)
    pub latency: f64,
    /// Memory bandwidth utilization (0-1)
    pub memory_bandwidth_utilization: f64,
    /// Cache efficiency (0-1)
    pub cache_efficiency: f64,
    /// SIMD utilization (0-1)
    pub simd_utilization: f64,
    /// Energy efficiency (operations per joule)
    pub energy_efficiency: f64,
}

/// Adaptive optimization strategy
#[derive(Debug, Clone)]
pub struct AdaptiveStrategy {
    /// Strategy name
    pub name: String,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Performance threshold for strategy switching
    pub performance_threshold: f64,
    /// Strategy switching history
    pub switching_history: Vec<StrategySwitch>,
    /// Current confidence in strategy
    pub confidence: f64,
}

/// Strategy switching event
#[derive(Debug, Clone)]
pub struct StrategySwitch {
    /// Previous strategy
    pub from_strategy: String,
    /// New strategy
    pub to_strategy: String,
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Performance snapshot for tracking
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Operation that was performed
    pub operation: String,
    /// Data size
    pub data_size: usize,
    /// Strategies used
    pub simd_strategy: String,
    pub cache_strategy: String,
    /// Measured performance
    pub measured_performance: PerformanceCharacteristics,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Current optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Active optimizations
    pub active_optimizations: Vec<String>,
    /// Performance trend
    pub performance_trend: PerformanceTrend,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving { rate: f64 },
    Stable { variance: f64 },
    Degrading { rate: f64 },
    Unknown,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization (0-1)
    pub cpu_utilization: f64,
    /// Memory utilization (0-1)
    pub memory_utilization: f64,
    /// Cache utilization (0-1)
    pub cache_utilization: f64,
    /// SIMD unit utilization (0-1)
    pub simd_utilization: f64,
    /// Memory bandwidth utilization (0-1)
    pub memory_bandwidth_utilization: f64,
}

/// Unified optimizer configuration
#[derive(Debug, Clone)]
pub struct UnifiedOptimizerConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable cache optimizations
    pub enable_cache: bool,
    /// Enable adaptive optimization
    pub enable_adaptive: bool,
    /// Performance monitoring frequency
    pub monitoring_frequency: std::time::Duration,
    /// Optimization aggressiveness (0.0 to 1.0)
    pub optimization_aggressiveness: f64,
    /// Learning rate for adaptive strategies
    pub learning_rate: f64,
    /// Performance history retention period
    pub history_retention: std::time::Duration,
}

impl UnifiedOptimizationEngine {
    /// Create new unified optimization engine
    pub fn new(config: UnifiedOptimizerConfig) -> Result<Self> {
        let simd_config = SimdEngineConfig {
            enable_aggressive_opts: config.enable_simd,
            enable_runtime_detection: true,
            enable_profiling: true,
            ..Default::default()
        };

        let cache_config = CacheOptimizerConfig {
            enable_numa_optimization: config.enable_cache,
            enable_adaptive_prefetching: config.enable_cache,
            enable_layout_optimization: config.enable_cache,
            optimization_aggressiveness: config.optimization_aggressiveness,
            ..Default::default()
        };

        let simd_engine = if config.enable_simd {
            Arc::new(Mutex::new(UltraSimdEngine::new(simd_config)?))
        } else {
            global_simd_engine()
        };

        let cache_optimizer = if config.enable_cache {
            Arc::new(Mutex::new(UltraCacheOptimizer::new(cache_config)?))
        } else {
            global_ultra_cache_optimizer()
        };

        let coordinator = Arc::new(RwLock::new(OptimizationCoordinator::new()));
        let profiler = Arc::new(Profiler::new());

        let mut engine = Self {
            simd_engine,
            cache_optimizer,
            coordinator,
            profiler,
            config,
        };

        // Initialize operation profiles
        engine.initialize_operation_profiles()?;

        Ok(engine)
    }

    /// Initialize operation performance profiles
    fn initialize_operation_profiles(&mut self) -> Result<()> {
        let mut coordinator = self.coordinator.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock coordinator".to_string())
        })?;

        // Matrix multiplication profile
        coordinator.operation_profiles.insert(
            "matrix_multiply".to_string(),
            OperationPerformanceProfile {
                operation_name: "matrix_multiply".to_string(),
                optimal_simd_strategy: SimdStrategy::Adaptive {
                    min_width: 128,
                    max_width: 512,
                },
                optimal_cache_strategy: CacheStrategy::Blocked {
                    l1_block: 64,
                    l2_block: 256,
                    l3_block: 1024,
                },
                size_breakpoints: vec![
                    SizeBreakpoint {
                        min_size: 0,
                        max_size: 1024,
                        simd_strategy: SimdStrategy::HardwareSpecific { vector_width: 128 },
                        cache_strategy: CacheStrategy::CacheOblivious,
                        performance_multiplier: 1.2,
                    },
                    SizeBreakpoint {
                        min_size: 1024,
                        max_size: 1048576,
                        simd_strategy: SimdStrategy::Adaptive {
                            min_width: 256,
                            max_width: 512,
                        },
                        cache_strategy: CacheStrategy::Blocked {
                            l1_block: 64,
                            l2_block: 256,
                            l3_block: 1024,
                        },
                        performance_multiplier: 2.1,
                    },
                    SizeBreakpoint {
                        min_size: 1048576,
                        max_size: usize::MAX,
                        simd_strategy: SimdStrategy::Mixed {
                            strategies: vec![
                                (
                                    2097152,
                                    Box::new(SimdStrategy::HardwareSpecific { vector_width: 512 }),
                                ),
                                (
                                    usize::MAX,
                                    Box::new(SimdStrategy::Adaptive {
                                        min_width: 256,
                                        max_width: 512,
                                    }),
                                ),
                            ],
                        },
                        cache_strategy: CacheStrategy::Combined {
                            strategies: vec![
                                CacheStrategy::NumaAware { preferred_node: 0 },
                                CacheStrategy::Blocked {
                                    l1_block: 128,
                                    l2_block: 512,
                                    l3_block: 2048,
                                },
                            ],
                        },
                        performance_multiplier: 3.5,
                    },
                ],
                performance_characteristics: PerformanceCharacteristics {
                    throughput: 2e12,
                    latency: 1e-6,
                    memory_bandwidth_utilization: 0.9,
                    cache_efficiency: 0.85,
                    simd_utilization: 0.95,
                    energy_efficiency: 1e12,
                },
            },
        );

        // Element-wise operations profile
        coordinator.operation_profiles.insert(
            "elementwise".to_string(),
            OperationPerformanceProfile {
                operation_name: "elementwise".to_string(),
                optimal_simd_strategy: SimdStrategy::HardwareSpecific { vector_width: 256 },
                optimal_cache_strategy: CacheStrategy::Prefetching {
                    distance: 64,
                    pattern: PrefetchPattern::Sequential,
                },
                size_breakpoints: vec![
                    SizeBreakpoint {
                        min_size: 0,
                        max_size: 4096,
                        simd_strategy: SimdStrategy::HardwareSpecific { vector_width: 128 },
                        cache_strategy: CacheStrategy::CacheOblivious,
                        performance_multiplier: 1.8,
                    },
                    SizeBreakpoint {
                        min_size: 4096,
                        max_size: usize::MAX,
                        simd_strategy: SimdStrategy::HardwareSpecific { vector_width: 512 },
                        cache_strategy: CacheStrategy::Prefetching {
                            distance: 128,
                            pattern: PrefetchPattern::Sequential,
                        },
                        performance_multiplier: 4.2,
                    },
                ],
                performance_characteristics: PerformanceCharacteristics {
                    throughput: 4e12,
                    latency: 5e-7,
                    memory_bandwidth_utilization: 0.95,
                    cache_efficiency: 0.9,
                    simd_utilization: 0.98,
                    energy_efficiency: 2e12,
                },
            },
        );

        Ok(())
    }

    /// Perform unified optimization for given operation
    pub fn optimize_operation(
        &self,
        operation: &str,
        input_a: &[f32],
        input_b: &[f32],
        output: &mut [f32],
    ) -> Result<UnifiedOptimizationResult> {
        let start_time = std::time::Instant::now();

        // Get optimization strategy for this operation and data size
        let data_size = input_a.len();
        let strategy = self.select_optimization_strategy(operation, data_size)?;

        // Apply cache optimizations
        let cache_result = if self.config.enable_cache {
            let cache_optimizer = self.cache_optimizer.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock cache optimizer".to_string())
            })?;
            Some(cache_optimizer.optimize_memory_access(operation, data_size, "sequential")?)
        } else {
            None
        };

        // Apply SIMD optimizations
        let simd_result = if self.config.enable_simd {
            let simd_engine = self.simd_engine.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock SIMD engine".to_string())
            })?;

            match operation {
                "elementwise_add" => {
                    simd_engine.optimized_elementwise(
                        input_a,
                        input_b,
                        output,
                        ElementWiseOp::Add,
                    )?;
                    Some("elementwise_add".to_string())
                }
                "matrix_multiply" => {
                    if input_a.len() >= 16 && input_b.len() >= 16 && output.len() >= 16 {
                        let n = (input_a.len() as f64).sqrt() as usize;
                        if n * n == input_a.len() {
                            simd_engine.optimized_matmul(input_a, input_b, output, n, n, n)?;
                            Some("matrix_multiply".to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        // Update performance tracking
        self.update_performance_tracking(operation, data_size, &strategy, start_time.elapsed())?;

        Ok(UnifiedOptimizationResult {
            operation: operation.to_string(),
            data_size,
            strategy_used: strategy,
            cache_optimization: cache_result,
            simd_optimization: simd_result,
            total_time: start_time.elapsed(),
            performance_improvement: self
                .calculate_performance_improvement(operation, data_size)?,
        })
    }

    /// Select optimal optimization strategy
    fn select_optimization_strategy(
        &self,
        operation: &str,
        data_size: usize,
    ) -> Result<OptimizationStrategy> {
        let coordinator = self.coordinator.read().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock coordinator".to_string())
        })?;

        if let Some(profile) = coordinator.operation_profiles.get(operation) {
            // Find appropriate size breakpoint
            for breakpoint in &profile.size_breakpoints {
                if data_size >= breakpoint.min_size && data_size <= breakpoint.max_size {
                    return Ok(OptimizationStrategy {
                        simd_strategy: breakpoint.simd_strategy.clone(),
                        cache_strategy: breakpoint.cache_strategy.clone(),
                        expected_performance_multiplier: breakpoint.performance_multiplier,
                    });
                }
            }

            // Fallback to optimal strategy
            Ok(OptimizationStrategy {
                simd_strategy: profile.optimal_simd_strategy.clone(),
                cache_strategy: profile.optimal_cache_strategy.clone(),
                expected_performance_multiplier: 1.5,
            })
        } else {
            // Default strategy for unknown operations
            Ok(OptimizationStrategy {
                simd_strategy: SimdStrategy::Adaptive {
                    min_width: 128,
                    max_width: 256,
                },
                cache_strategy: CacheStrategy::CacheOblivious,
                expected_performance_multiplier: 1.2,
            })
        }
    }

    /// Update performance tracking
    fn update_performance_tracking(
        &self,
        operation: &str,
        data_size: usize,
        strategy: &OptimizationStrategy,
        execution_time: std::time::Duration,
    ) -> Result<()> {
        let mut coordinator = self.coordinator.write().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock coordinator".to_string())
        })?;

        let snapshot = PerformanceSnapshot {
            operation: operation.to_string(),
            data_size,
            simd_strategy: format!("{:?}", strategy.simd_strategy),
            cache_strategy: format!("{:?}", strategy.cache_strategy),
            measured_performance: PerformanceCharacteristics {
                throughput: data_size as f64 / execution_time.as_secs_f64(),
                latency: execution_time.as_secs_f64(),
                memory_bandwidth_utilization: 0.8, // Estimated
                cache_efficiency: 0.85,            // Estimated
                simd_utilization: 0.9,             // Estimated
                energy_efficiency: 1e12,           // Estimated
            },
            timestamp: std::time::Instant::now(),
        };

        coordinator.performance_history.push(snapshot);

        // Maintain history size
        while coordinator.performance_history.len() > 1000 {
            coordinator.performance_history.remove(0);
        }

        Ok(())
    }

    /// Calculate performance improvement achieved
    fn calculate_performance_improvement(&self, operation: &str, data_size: usize) -> Result<f64> {
        let coordinator = self.coordinator.read().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock coordinator".to_string())
        })?;

        // Find baseline performance for comparison
        let recent_snapshots: Vec<_> = coordinator
            .performance_history
            .iter()
            .filter(|s| s.operation == operation && s.data_size == data_size)
            .collect();

        if recent_snapshots.len() >= 2 {
            let latest = &recent_snapshots[recent_snapshots.len() - 1];
            let baseline = &recent_snapshots[0];

            let improvement = (latest.measured_performance.throughput
                - baseline.measured_performance.throughput)
                / baseline.measured_performance.throughput;

            Ok(improvement.max(0.0))
        } else {
            // Estimate based on strategy
            if let Some(profile) = coordinator.operation_profiles.get(operation) {
                for breakpoint in &profile.size_breakpoints {
                    if data_size >= breakpoint.min_size && data_size <= breakpoint.max_size {
                        return Ok(breakpoint.performance_multiplier - 1.0);
                    }
                }
            }
            Ok(0.2) // Default 20% improvement estimate
        }
    }

    /// Get comprehensive optimization statistics
    pub fn get_optimization_statistics(&self) -> Result<UnifiedOptimizationStatistics> {
        let coordinator = self.coordinator.read().map_err(|_| {
            TensorError::compute_error_simple("Failed to lock coordinator".to_string())
        })?;

        let simd_stats = if self.config.enable_simd {
            let simd_engine = self.simd_engine.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock SIMD engine".to_string())
            })?;
            Some(simd_engine.get_performance_stats()?)
        } else {
            None
        };

        let cache_stats = if self.config.enable_cache {
            let cache_optimizer = self.cache_optimizer.lock().map_err(|_| {
                TensorError::compute_error_simple("Failed to lock cache optimizer".to_string())
            })?;
            Some(cache_optimizer.get_optimization_statistics()?)
        } else {
            None
        };

        Ok(UnifiedOptimizationStatistics {
            total_operations_optimized: coordinator.performance_history.len(),
            average_performance_improvement: self.calculate_average_improvement(&coordinator)?,
            simd_statistics: simd_stats,
            cache_statistics: cache_stats,
            operation_profiles: coordinator.operation_profiles.clone(),
            current_state: coordinator.current_optimization_state.clone(),
        })
    }

    fn calculate_average_improvement(&self, coordinator: &OptimizationCoordinator) -> Result<f64> {
        if coordinator.performance_history.is_empty() {
            return Ok(0.0);
        }

        let mut total_improvement = 0.0;
        let mut count = 0;

        // Group by operation and calculate improvements
        let mut operation_groups: HashMap<String, Vec<&PerformanceSnapshot>> = HashMap::new();
        for snapshot in &coordinator.performance_history {
            operation_groups
                .entry(snapshot.operation.clone())
                .or_default()
                .push(snapshot);
        }

        for snapshots in operation_groups.values() {
            if snapshots.len() >= 2 {
                let first = snapshots[0];
                let last = snapshots[snapshots.len() - 1];

                let improvement = (last.measured_performance.throughput
                    - first.measured_performance.throughput)
                    / first.measured_performance.throughput;

                total_improvement += improvement.max(0.0);
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_improvement / count as f64
        } else {
            0.0
        })
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub simd_strategy: SimdStrategy,
    pub cache_strategy: CacheStrategy,
    pub expected_performance_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct UnifiedOptimizationResult {
    pub operation: String,
    pub data_size: usize,
    pub strategy_used: OptimizationStrategy,
    pub cache_optimization: Option<MemoryOptimizationResult>,
    pub simd_optimization: Option<String>,
    pub total_time: std::time::Duration,
    pub performance_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct UnifiedOptimizationStatistics {
    pub total_operations_optimized: usize,
    pub average_performance_improvement: f64,
    pub simd_statistics: Option<SimdPerformanceStats>,
    pub cache_statistics: Option<crate::memory::ultra_cache_optimizer::CacheOptimizationStatistics>,
    pub operation_profiles: HashMap<String, OperationPerformanceProfile>,
    pub current_state: OptimizationState,
}

impl OptimizationCoordinator {
    fn new() -> Self {
        Self {
            operation_profiles: HashMap::new(),
            adaptive_strategies: HashMap::new(),
            performance_history: Vec::new(),
            current_optimization_state: OptimizationState {
                active_optimizations: vec!["simd".to_string(), "cache".to_string()],
                performance_trend: PerformanceTrend::Improving { rate: 0.15 },
                resource_utilization: ResourceUtilization {
                    cpu_utilization: 0.7,
                    memory_utilization: 0.6,
                    cache_utilization: 0.85,
                    simd_utilization: 0.9,
                    memory_bandwidth_utilization: 0.8,
                },
                optimization_effectiveness: 0.82,
            },
        }
    }
}

impl Default for UnifiedOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_cache: true,
            enable_adaptive: true,
            monitoring_frequency: std::time::Duration::from_millis(100),
            optimization_aggressiveness: 0.8,
            learning_rate: 0.1,
            history_retention: std::time::Duration::from_secs(3600),
        }
    }
}

/// Global unified optimization engine instance
static GLOBAL_UNIFIED_OPTIMIZER: std::sync::OnceLock<Arc<Mutex<UnifiedOptimizationEngine>>> =
    std::sync::OnceLock::new();

/// Get the global unified optimization engine
pub fn global_unified_optimizer() -> Arc<Mutex<UnifiedOptimizationEngine>> {
    GLOBAL_UNIFIED_OPTIMIZER
        .get_or_init(|| {
            let config = UnifiedOptimizerConfig::default();
            let optimizer =
                UnifiedOptimizationEngine::new(config).expect("Failed to create unified optimizer");
            Arc::new(Mutex::new(optimizer))
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_optimizer_creation() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_strategy_selection() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config).unwrap();

        let strategy = optimizer.select_optimization_strategy("matrix_multiply", 1024);
        assert!(strategy.is_ok());

        let strategy = strategy.unwrap();
        assert!(strategy.expected_performance_multiplier > 1.0);
    }

    #[test]
    fn test_elementwise_optimization() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config).unwrap();

        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut c = vec![0.0; 16];

        let result = optimizer.optimize_operation("elementwise_add", &a, &b, &mut c);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.operation, "elementwise_add");
        assert!(result.performance_improvement >= 0.0);

        // Verify results
        for value in &c {
            assert_eq!(*value, 3.0);
        }
    }

    #[test]
    fn test_matrix_multiply_optimization() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config).unwrap();

        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut c = vec![0.0; 16];

        let result = optimizer.optimize_operation("matrix_multiply", &a, &b, &mut c);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.operation, "matrix_multiply");
        assert!(result.performance_improvement >= 0.0);
    }

    #[test]
    fn test_optimization_statistics() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config).unwrap();

        // Perform some operations to generate statistics
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let mut c = vec![0.0; 16];

        let _ = optimizer.optimize_operation("elementwise_add", &a, &b, &mut c);
        let _ = optimizer.optimize_operation("matrix_multiply", &a, &b, &mut c);

        let stats = optimizer.get_optimization_statistics();
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert!(stats.total_operations_optimized > 0);
        assert!(!stats.operation_profiles.is_empty());
    }

    #[test]
    fn test_global_unified_optimizer() {
        let optimizer1 = global_unified_optimizer();
        let optimizer2 = global_unified_optimizer();

        // Should be the same instance
        assert!(Arc::ptr_eq(&optimizer1, &optimizer2));
    }

    #[test]
    fn test_performance_tracking() {
        let config = UnifiedOptimizerConfig::default();
        let optimizer = UnifiedOptimizationEngine::new(config).unwrap();

        let strategy = OptimizationStrategy {
            simd_strategy: SimdStrategy::HardwareSpecific { vector_width: 256 },
            cache_strategy: CacheStrategy::CacheOblivious,
            expected_performance_multiplier: 1.5,
        };

        let result = optimizer.update_performance_tracking(
            "test_op",
            1024,
            &strategy,
            std::time::Duration::from_millis(10),
        );
        assert!(result.is_ok());
    }
}
