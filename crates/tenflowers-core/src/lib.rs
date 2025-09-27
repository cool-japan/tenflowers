#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::result_large_err)]

pub mod adaptive_tuning;
#[cfg(feature = "gpu")]
pub mod async_gpu_optimizations;
pub mod buffer;
pub mod collective;
pub mod complex;
pub mod context;
pub mod cross_platform_optimization;
pub mod deployment;
pub mod device;
pub mod dtype;
pub mod eager_execution;
pub mod error;
pub mod fallback;
pub mod gradient_clipping;
pub mod graph;
pub mod half_precision;
pub mod integration;
pub mod large_model_optimization;
pub mod layout;
pub mod memory;
pub mod memory_tensorflow_comparison;
pub mod mixed_precision;
pub mod monitoring;
pub mod neural_optimization;
pub mod onnx_interop;
pub mod ops;
pub mod performance_benchmarks;
pub mod production_benchmarks;
pub mod production_performance_monitoring;
pub mod quantization;
pub mod session;
pub mod shape;
pub mod simd;
pub mod simplified_benchmarks;
pub mod strided;
pub mod structured_arrays;
pub mod system_health;
pub mod tensor;
pub mod tensor_view;
pub mod ultra_performance_profiler;
pub mod wasm;
pub mod wasm_optimization;
// pub mod benchmarks;  // Temporarily disabled due to compilation issues

pub use complex::{Complex32, Complex64};
pub use device::Device;
pub use dtype::{dtype_from_type, DType};
pub use error::{Result, TensorError};
pub use fallback::{
    cleanup_memory_and_retry, execute_binary_op_with_fallback, execute_unary_op_with_fallback,
    get_fallback_config, is_auto_fallback_enabled, set_auto_fallback_enabled, set_fallback_config,
    FallbackConfig, FallbackWrapper,
};
pub use half_precision::{
    bf16, f16, HalfPrecision, MixedPrecisionConfig as HalfMixedPrecisionConfig,
};
pub use integration::{
    BaselinePerformance, OptimizationBreakdown, PerformanceTargets, UltraPerformanceValidator,
    ValidationReport, ValidationResult, ValidationTestSuite,
};
pub use layout::{convert_layout, infer_layout, DataLayout, LayoutOptimizer, OperationType};
pub use quantization::{
    dequantize, dynamic_quantize, fake_quantize, per_channel_quantize, quantize, QuantizationParams,
};
pub use shape::Shape;
#[cfg(feature = "simd")]
pub use simd::{benchmarks::Benchmarks as simd_benchmarks, SimdCapabilities, SimdOptimizer};
pub use simd::{
    global_simd_engine, AdvancedKernelRegistry, CacheFriendlyMatMul, CacheOptimizedTensorOps,
    ConvolutionParams, CpuFeatures, ElementWiseOp, KernelOptimizationStrategy, MemoryAccessPattern,
    ReductionOp as SimdReductionOp, SimdEngineConfig, SpecializedKernel, UltraSimdEngine,
};
pub use tensor::Tensor;
// pub use deployment::{GraphFreezer, GraphFreezingConfig, GraphFreezingStats, freeze_graph_for_inference, freeze_graph_with_config};
pub use adaptive_tuning::{
    execute_with_adaptive_tuning, AdaptiveTuner, ExecutionStrategy, OperationMetrics,
    PerformancePredictor, GLOBAL_TUNER,
};
#[cfg(feature = "gpu")]
pub use async_gpu_optimizations::{
    utils as async_gpu_utils, AccessPattern, AsyncGpuOperation, AsyncGpuScheduler,
    AsyncMatMulOperation, ComputeIntensity, OperationPriority,
    PerformanceMetrics as AsyncPerformanceMetrics,
};
pub use collective::{
    all_gather, all_reduce, broadcast, create_process_group, init_collective, CollectiveManager,
    CollectiveOp, CommunicationGroup, ReductionOp,
};
pub use context::{get_context, set_context, Context};
pub use cross_platform_optimization::{
    get_global_optimizer, get_optimal_configuration, initialize_cross_platform_optimizer,
    CrossPlatformOptimizer, OptimalConfiguration, TargetArchitecture, TargetPlatform,
};
pub use eager_execution::{
    CacheStatistics, EagerExecutionConfig, EagerExecutionEngine, EagerPerformanceReport,
    ExecutionMetrics, EAGER_ENGINE,
};
pub use gradient_clipping::{
    GradientClipper, GradientClippingConfig, GradientStatistics, NormType,
};
pub use graph::{
    AttributeValue, AttributeValueDef, EdgeId, Graph, GraphDef, GraphEdge, GraphNode, NodeDef,
    NodeId, NodeType,
};
pub use large_model_optimization::{
    LargeModelConfig, LargeModelOptimizationReport, LargeModelOptimizer, MemoryOptimizationStats,
    ModelExecutionPlan, LARGE_MODEL_OPTIMIZER,
};
pub use memory::{
    global_monitor, global_monitor_arc, KernelOccupancyStats, MemoryAliasDetector, MemoryPool,
    MemoryPoolStats, MultiStreamMemoryManager, OperationTimer, PerformanceMonitor, StridedView,
};
pub use memory_tensorflow_comparison::{
    MemoryComparisonReport, MemoryOptimizationSuggestion, MemoryProfilingConfig, MemorySnapshot,
    TensorFlowMemoryProfiler, MEMORY_PROFILER,
};
pub use mixed_precision::{
    disable_autocast, enable_autocast, enable_autocast_bfloat16, from_bfloat16_f32,
    from_bfloat16_f64, from_half, from_half_f32, from_half_f64, to_bfloat16_f32, to_bfloat16_f64,
    to_half, to_half_f32, to_half_f64, AutocastContext, GradientScaler, MixedPrecisionConfig,
    MixedPrecisionState,
};
pub use monitoring::{
    AlertSeverity,
    // Analytics and trends
    BottleneckType,
    MonitoringConfig as UltraMonitoringConfig,
    MonitoringReport,
    // Metrics
    OperationMetrics as MonitoringOperationMetrics,
    OptimizationOpportunity,
    PerformanceAlert,
    PerformanceDashboard,

    PerformancePrediction,

    // Use different name to avoid conflict with adaptive_tuning::PerformancePredictor
    PerformancePredictor as MonitoringPerformancePredictor,

    PerformanceSnapshot,
    SystemBottleneck,
    SystemMetrics,
    TrendDirection,
    TrendType,
    // Core monitoring components
    UltraPerformanceMonitor,
};
pub use neural_optimization::{
    LayerPerformanceMetrics, NetworkPerformanceReport,
    OptimizationBreakdown as NeuralOptimizationBreakdown, UltraOptimizedActivations,
    UltraOptimizedDenseLayer, UltraOptimizedNeuralNetwork,
};
pub use onnx_interop::{
    OnnxConfig,
    OnnxExporter,
    OnnxImporter,
    OnnxModel,
    // TODO: Add back when implemented: utils as onnx_utils, BenchmarkStats, CompatibilityReport, TenfloweRSModel
};
pub use ops::{
    print_framework_comparison_results, run_framework_comparison_benchmark,
    FrameworkBenchmarkConfig, FrameworkComparisonResult,
};
pub use production_benchmarks::{
    run_comprehensive_production_benchmarks, BenchmarkConfig, BenchmarkResult,
    BenchmarkSummary as ProductionBenchmarkSummary,
    OptimizationBreakdown as ProductionOptimizationBreakdown, ProblemSize,
    ProductionBenchmarkReport, ProductionBenchmarkSuite, QualityMetrics,
};
pub use production_performance_monitoring::{
    get_global_monitor, initialize_performance_monitoring, record_performance_event,
    AlertThresholds, MonitoringConfig, PerformanceEvent, PerformanceMetrics,
    ProductionPerformanceMonitor,
};
pub use session::{create_session, DefaultSession, FeedDict, FetchSpec, Session, SessionConfig};
pub use simplified_benchmarks::{
    run_simple_benchmarks, validate_optimizations, BenchmarkReport, BenchmarkSummary,
    SimpleBenchmarkConfig, SimpleBenchmarkResult, SimpleBenchmarkSuite,
};
pub use strided::{SliceParams, StridedLayout};
pub use structured_arrays::{FieldDescriptor, FieldValue, StructuredArray};
pub use system_health::{
    run_quick_health_check, run_system_health_check, FeaturesInfo, GpuMemoryInfo,
    HealthCheckConfig, HealthStatus, MemoryInfo, PerformanceBenchmarks, SystemHealthChecker,
    SystemInfo,
};
pub use tensor_view::{MemoryStats, TensorView, TensorViewOps};
pub use wasm::{utils as wasm_utils, WasmContext};
#[cfg(target_arch = "wasm32")]
pub use wasm::{WasmContextWithGpu, WasmWebGpuContext, WebGpuBackend, WebGpuLimits};
#[cfg(feature = "wasm")]
pub use wasm_optimization::{
    WasmBundleOptimizer, WasmEdgeInference, WasmMemoryManager, WasmOptimizationConfig,
    WasmOptimizedTensor, WasmTensorOperations,
};

#[cfg(feature = "gpu")]
pub use gpu_profiler::{
    disable_gpu_profiling, enable_gpu_profiling, generate_gpu_profiling_report,
    get_gpu_profiling_stats, global_profiler, GpuProfiler, OperationProfile, ProfileStats,
};

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod gpu_profiler;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tensor_creation() {
        let tensor = Tensor::<f32>::zeros(&[2, 3]);
        assert_eq!(tensor.shape(), &Shape::from_slice(&[2, 3]));
    }
}
