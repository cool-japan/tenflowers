//! # TenfloweRS Core
//!
//! The core tensor operations and device management library for the TenfloweRS machine learning framework.
//! This crate provides the foundational building blocks for building, training, and deploying deep learning
//! models in pure Rust with safety, performance, and cross-platform GPU acceleration.
//!
//! ## Features
//!
//! - **Tensor Operations**: Comprehensive n-dimensional array operations with automatic broadcasting
//! - **Device Management**: Unified CPU/GPU abstraction with automatic memory management
//! - **Performance**: SIMD vectorization, parallel execution, and GPU compute kernels
//! - **Cross-Platform GPU**: WGPU-based GPU support (Metal, Vulkan, DirectX, WebGPU)
//! - **Advanced Optimizations**: Mixed precision, quantization, kernel fusion, memory pooling
//! - **Production Features**: Checkpointing, serialization, deterministic execution, profiling
//! - **SciRS2 Integration**: Built on the robust SciRS2 scientific computing ecosystem
//!
//! ## Quick Start
//!
//! ### Basic Tensor Creation and Operations
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create tensors
//! let a = Tensor::<f32>::zeros(&[2, 3]);
//! let b = Tensor::<f32>::ones(&[2, 3]);
//!
//! // Arithmetic operations
//! let c = tenflowers_core::ops::add(&a, &b)?;
//! let d = tenflowers_core::ops::mul(&a, &b)?;
//!
//! // Matrix multiplication
//! let x = Tensor::<f32>::ones(&[2, 3]);
//! let y = Tensor::<f32>::ones(&[3, 4]);
//! let z = tenflowers_core::ops::matmul(&x, &y)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### GPU Acceleration
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "gpu")]
//! # {
//! // Create tensor on GPU
//! let device = Device::gpu(0)?;
//! let gpu_tensor = Tensor::<f32>::zeros(&[1000, 1000]).to_device(&device)?;
//!
//! // Operations automatically run on GPU
//! let result = tenflowers_core::ops::matmul(&gpu_tensor, &gpu_tensor)?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Features
//!
//! #### Mixed Precision Training
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, f16, MixedPrecisionConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Use f16 for faster training with less memory
//! let fp16_tensor = Tensor::<f16>::ones(&[1024, 1024]);
//! let result = tenflowers_core::ops::matmul(&fp16_tensor, &fp16_tensor)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Quantization
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, quantize, QuantizationParams};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let tensor = Tensor::<f32>::ones(&[100, 100]);
//!
//! // Quantize to 8-bit for inference
//! let quantized = quantize(&tensor, 8)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Deterministic Execution
//!
//! ```rust,no_run
//! use tenflowers_core::{set_deterministic_mode, set_global_seed};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Enable deterministic mode for reproducible results
//! set_deterministic_mode(true);
//! set_global_seed(42);
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture Overview
//!
//! The crate is organized into the following modules:
//!
//! - [`tensor`]: Core tensor type with device placement and memory management
//! - [`ops`]: Tensor operations (arithmetic, linear algebra, neural network primitives)
//! - [`device`]: Device abstraction (CPU, GPU, custom accelerators)
//! - [`dtype`]: Data type system (f32, f64, f16, bf16, i32, etc.)
//! - [`shape`]: Shape inference and validation
//! - [`memory`]: Memory management, pooling, and optimization
//! - [`graph`]: Computation graph construction and optimization
//! - [`session`]: Graph execution engine
//! - [`quantization`]: Model quantization for deployment
//! - [`mixed_precision`]: Mixed precision training utilities
//! - [`checkpointing`]: Model checkpointing and restoration
//! - [`deterministic`]: Deterministic execution controls
//! - [`monitoring`]: Performance monitoring and profiling
//!
//! ## Performance Features
//!
//! ### SIMD Optimization
//!
//! The crate automatically uses SIMD instructions when available for maximum performance:
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, SimdCapabilities};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Check available SIMD features
//! let capabilities = SimdCapabilities::detect();
//! println!("SIMD support: {:?}", capabilities);
//!
//! // Operations automatically use SIMD when beneficial
//! let a = Tensor::<f32>::ones(&[10000]);
//! let b = Tensor::<f32>::ones(&[10000]);
//! let c = tenflowers_core::ops::add(&a, &b)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Memory Optimization
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Device};
//! use tenflowers_core::memory::{BufferPool, GlobalBufferPool};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Use buffer pooling for efficient memory reuse
//! let pool = GlobalBufferPool::get();
//! pool.set_max_pool_size(1024 * 1024 * 1024); // 1GB
//!
//! // Tensors automatically use the pool
//! let tensor = Tensor::<f32>::zeros(&[1000, 1000]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Integration with TenfloweRS Ecosystem
//!
//! This crate integrates seamlessly with:
//! - `tenflowers-autograd`: Automatic differentiation engine
//! - `tenflowers-neural`: High-level neural network layers
//! - `tenflowers-dataset`: Data loading and preprocessing
//! - `scirs2-core`: Scientific computing primitives
//! - `scirs2-autograd`: Static graph optimization
//!
//! ## GPU Support
//!
//! TenfloweRS Core uses WGPU for cross-platform GPU acceleration, supporting:
//! - **Metal** (macOS, iOS)
//! - **Vulkan** (Windows, Linux, Android)
//! - **DirectX 12** (Windows)
//! - **WebGPU** (browsers)
//!
//! Enable GPU support with the `gpu` feature flag:
//!
//! ```toml
//! [dependencies]
//! tenflowers-core = { version = "0.1.0-alpha.2", features = ["gpu"] }
//! ```
//!
//! ## Safety and Correctness
//!
//! TenfloweRS Core is designed with safety as a primary concern:
//! - Memory-safe by default (no unsafe code in core tensor operations)
//! - Extensive shape validation and error handling
//! - Gradient checking utilities for numerical correctness
//! - Deterministic execution modes for reproducibility
//!
//! ## Performance Benchmarking
//!
//! Use the built-in benchmarking utilities to measure performance:
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Device};
//! use tenflowers_core::profiling::Profiler;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let profiler = Profiler::new();
//! profiler.start("matmul");
//!
//! let a = Tensor::<f32>::ones(&[1000, 1000]);
//! let b = Tensor::<f32>::ones(&[1000, 1000]);
//! let c = tenflowers_core::ops::matmul(&a, &b)?;
//!
//! profiler.stop("matmul");
//! profiler.print_summary();
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::result_large_err)]
// Allow common patterns in GPU code that clippy flags
#![allow(clippy::needless_borrow)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::upper_case_acronyms)]

pub mod adaptive_tuning;
#[cfg(feature = "gpu")]
pub mod async_gpu_optimizations;
pub mod buffer;
pub mod checkpointing;
pub mod collective;
pub mod complex;
pub mod context;
pub mod cross_platform_optimization;
pub mod deployment;
pub mod deterministic;
pub mod device;
pub mod dispatch_init;
pub mod dispatch_registry;
pub mod dispatch_registry_examples;
pub mod dispatch_registry_extended;
pub mod dtype;
pub mod eager_execution;
pub mod error;
pub mod fallback;
pub mod gpu_memory_metrics;
pub mod gradient_clipping;
pub mod gradient_coverage_audit;
pub mod gradient_validation_framework;
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
pub mod numerical_gradient;
pub mod onnx_interop;
pub mod ops;
pub mod performance_benchmarks;
pub mod performance_gates;
pub mod production_benchmarks;
pub mod production_performance_monitoring;
pub mod quantization;
#[cfg(feature = "serialize")]
pub mod serialization;
#[cfg(feature = "serialize")]
pub mod serialization_onnx;
pub mod session;
pub mod shape;
pub mod shape_error_taxonomy;
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
pub use shape_error_taxonomy::{
    validate_broadcast_shapes, validate_elementwise_shapes, validate_matmul_shapes,
    validate_reduction_axis, validate_reshape, ShapeErrorBuilder, ShapeErrorCategory,
    ShapeErrorUtils,
};
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
pub use deterministic::{
    clear_operation_log, get_global_seed, get_operation_log, get_operation_seed,
    get_state_snapshot, is_deterministic_mode, is_strict_mode, mark_non_deterministic,
    reset_operation_counter, restore_state_snapshot, set_deterministic_mode, set_global_seed,
    set_strict_mode, should_use_deterministic_gpu_ops, DeterministicConfig, DeterministicScope,
    DeterministicSnapshot, DeterministicState,
};
pub use dispatch_init::ensure_initialized as ensure_dispatch_initialized;
pub use dispatch_registry::{
    get_registry, BackendType, BinaryKernelFn, DispatchRegistry, KernelImplementation,
    OperationDescriptor, UnaryKernelFn, F32_REGISTRY, F64_REGISTRY, I32_REGISTRY,
};
pub use eager_execution::{
    CacheStatistics, EagerExecutionConfig, EagerExecutionEngine, EagerPerformanceReport,
    ExecutionMetrics, EAGER_ENGINE,
};
pub use gpu_memory_metrics::{
    generate_memory_report, get_gpu_memory_snapshot, get_gpu_memory_usage, get_gpu_peak_memory,
    print_memory_report, reset_gpu_memory_metrics, GpuMemoryMetrics, GpuMemoryReport,
    GpuMemorySnapshot, GPU_MEMORY_METRICS,
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
#[cfg(feature = "gpu")]
pub use memory::DiagnosticMemoryPool;
pub use memory::{
    global_monitor, global_monitor_arc, IntegratedDiagnosticReport, KernelOccupancyStats,
    MemoryAliasDetector, MemoryPool, MemoryPoolStats, MultiStreamMemoryManager, OperationTimer,
    OptimizationResult, PerformanceMonitor, PoolHealthMetrics, PoolHealthStatus,
    PoolOptimizationConfig, StridedView,
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
    execute_fused_graph, get_fusion_stats, infer_binary_elementwise,
    infer_binary_elementwise_validated, infer_concat, infer_conv2d, infer_matmul, infer_reduction,
    infer_reshape, print_framework_comparison_results, print_fusion_report,
    record_fusion_opportunity, reset_fusion_stats, run_framework_comparison_benchmark,
    BroadcastableConstraint, ElementwiseOpType, ExactShapeConstraint, FrameworkBenchmarkConfig,
    FrameworkComparisonResult, FusionGraph, FusionNode, FusionPassBuilder, FusionStats,
    MatMulCompatibleConstraint, MinRankConstraint, RankConstraint, ShapeConstraint, ShapeContext,
    ShapeValidator,
};
pub use performance_gates::{
    get_baseline, list_baselines, register_baseline, OperationBaseline, PerformanceGate,
    PerformanceGateSuite, PerformanceMeasurement,
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
pub use gpu::memory_diagnostics::{
    check_gpu_memory_leaks, print_gpu_diagnostics, run_gpu_diagnostics, DiagnosticReport,
    DiagnosticsConfig, FragmentationAnalysis, GpuMemoryDiagnostics, LeakDetectionResult,
    OperationProfile as MemoryOperationProfile, GLOBAL_GPU_DIAGNOSTICS,
};

#[cfg(feature = "gpu")]
pub use gpu::memory_tracing::{
    current_gpu_memory_usage, generate_gpu_memory_report, peak_gpu_memory_usage,
    print_gpu_memory_report, record_gpu_allocation, record_gpu_deallocation, AllocationInfo,
    GpuMemoryTracker, MemoryReport, MemoryTracingConfig, GLOBAL_GPU_MEMORY_TRACKER,
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
