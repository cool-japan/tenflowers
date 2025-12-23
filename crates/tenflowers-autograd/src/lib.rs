//! # TenfloweRS Automatic Differentiation
//!
//! `tenflowers-autograd` provides a comprehensive automatic differentiation engine for the TenfloweRS
//! machine learning framework. This crate implements both forward-mode and reverse-mode automatic
//! differentiation with support for higher-order derivatives, custom gradients, and advanced
//! optimization techniques.
//!
//! ## Features
//!
//! - **Complete Gradient Operations**: All fundamental tensor operations with mathematically correct gradients
//! - **Higher-Order Derivatives**: Efficient computation of Hessians, third-order derivatives, and beyond
//! - **Performance Optimization**: Kernel fusion, memory optimization, and distributed gradient computation
//! - **Advanced Differentiation**: Mixed-mode AD, implicit differentiation, and custom gradient functions
//! - **Neural Network Integration**: Seamless integration with tenflowers-neural for deep learning
//! - **Distributed Training**: Parameter servers, gradient compression, and cross-datacenter replication
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tenflowers_autograd::{GradientTape, TrackedTensor};
//! use tenflowers_core::{Tensor, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::Cpu;
//! let mut tape = GradientTape::new();
//!
//! // Create tracked tensors
//! let x = tape.watch(Tensor::<f32>::ones(&[2, 2]));
//! let y = tape.watch(Tensor::<f32>::ones(&[2, 2]));
//!
//! // Compute gradients using GradientTape::gradient
//! let z = tape.watch(Tensor::<f32>::ones(&[2, 2])); // Placeholder for x+y result
//! let gradients = tape.gradient(&[z], &[x, y])?;
//! println!("Gradient of x: {:?}", gradients[0]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Gradients
//!
//! ```rust,no_run
//! use tenflowers_autograd::{CustomGradientFunction, GradientTape};
//! use tenflowers_core::{Tensor, Result};
//!
//! struct MyCustomOp;
//!
//! impl CustomGradientFunction<f32> for MyCustomOp {
//!     fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
//!         // Custom forward implementation: y = x^2 + sin(x)
//!         let x = inputs[0];
//!         let x_squared = tenflowers_core::ops::mul(x, x)?;
//!         let sin_x = tenflowers_core::ops::sin(x)?;
//!         tenflowers_core::ops::add(&x_squared, &sin_x)
//!     }
//!
//!     fn backward(&self, grad_output: &Tensor<f32>, inputs: &[&Tensor<f32>], output: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
//!         // Custom backward implementation: dy/dx = 2x + cos(x)
//!         let x = inputs[0];
//!         let two = tenflowers_core::Tensor::from_array(scirs2_core::ndarray::arr0(2.0f32).into_dyn());
//!         let two_x = tenflowers_core::ops::mul(&two, x)?;
//!         let cos_x = tenflowers_core::ops::cos(x)?;
//!         let grad_x = tenflowers_core::ops::add(&two_x, &cos_x)?;
//!         let final_grad = tenflowers_core::ops::mul(grad_output, &grad_x)?;
//!         Ok(vec![final_grad])
//!     }
//!
//!     fn name(&self) -> &str {
//!         "MyCustomOp"
//!     }
//! }
//! ```
//!
//! ### Higher-Order Derivatives
//!
//! ```rust,no_run
//! use tenflowers_autograd::{GradientTape, TrackedTensor};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut tape = GradientTape::new();
//! let x = TrackedTensor::new(Tensor::<f32>::ones(&[1]));
//! let target = TrackedTensor::new(Tensor::<f32>::ones(&[1]));
//!
//! // Compute third-order derivatives
//! let third_order = tape.third_derivative(&target, &x)?;
//!
//! // Compute nth-order derivatives
//! let nth_order = tape.nth_derivative(&target, &x, 3)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Features
//!
//! - **Kernel Fusion**: Automatically fuses operations to reduce memory bandwidth
//! - **Gradient Compression**: Quantization and sparsification for distributed training
//! - **Memory Optimization**: Checkpointing and in-place operations for large models
//! - **JIT Compilation**: Runtime kernel optimization for specific tensor shapes
//!
//! ## Integration with TenfloweRS Ecosystem
//!
//! This crate integrates seamlessly with:
//! - `tenflowers-core`: Core tensor operations and device management
//! - `tenflowers-neural`: Neural network layers and training loops
//! - `tenflowers-dataset`: Data loading and preprocessing
//! - `scirs2-autograd`: Static graph optimization and analysis

#![deny(unsafe_code)]
#![allow(clippy::result_large_err)]

pub mod advanced_grad_ops;
// pub mod advanced_linalg; // TODO: Fix compilation issues
pub mod amp_policy;
pub mod boolean_indexing;
pub mod checkpointing;
pub mod context;
pub mod coverage_matrix;
pub mod custom_gradients;
pub mod debug;
pub mod deterministic;
pub mod device_placement;
pub mod efficient_memory;
pub mod ellipsis_newaxis;
pub mod error_taxonomy;
pub mod forward_ad;
pub mod forward_reverse;
pub mod global_pooling;
pub mod gpu_gradient_expansion;
pub mod grad_ops;
pub mod gradient_accumulation;
pub mod gradient_analyzer;
pub mod gradient_buffer_manager_simple;
pub mod gradient_compression;
pub mod gradient_compression_advanced;
pub mod gradient_ops;
// pub mod gradient_utils; // TODO: Fix API compatibility with tenflowers-core
// pub mod gradient_validation; // TODO: Fix compilation issues
pub mod gradient_visualization;
pub mod graph_optimization;
pub mod higher_order;
pub mod hybrid_scheduler;
pub mod implicit_differentiation;
pub mod inplace_ops;
pub mod jit_compiler;
pub mod jit_integration;
pub mod kernel_fusion;
pub mod memory_diff_reporter;
pub mod memory_profiler;
pub mod neural_integration;
pub mod no_grad;
pub mod numerical_checker;
pub mod ops;
#[cfg(feature = "parallel")]
pub mod parallel_gradients;
pub mod parameter_server;
pub mod performance_benchmark;
pub mod second_order;
pub mod second_order_utils;
pub mod simd_grad_ops_simple;
pub mod special_functions;
pub mod subgraph_extraction;
pub mod tape;
pub mod tape_optimization;
pub mod tensor_ext;
pub mod tensor_networks;
pub mod ultra_gradient;
pub mod ultra_gradient_engine_simple;

pub use boolean_indexing::{
    boolean_mask_backward, integer_array_indexing_backward, where_backward,
};
pub use checkpointing::{
    checkpoint_sequence, ActivationCheckpointPolicy, ActivationCheckpointing,
    ActivationRecomputeManager, CheckpointManager, CheckpointStrategy, CheckpointedFunction,
    CheckpointedGradientTape, CheckpointingStats, LayerMetadata, RecomputationContext,
};
pub use context::{AutogradContext, ShapeInferenceRule, StaticShapeInference};
pub use coverage_matrix::{
    CategoryCoverage, CoverageMatrix, CoverageReport, OperationCategory, OperationMetadata,
};
pub use custom_gradients::{
    CustomGradientFunction, CustomGradientOp, GradientClipFunction, GradientScaleFunction,
    StopGradientFunction,
};
pub use debug::{GradientDebugInfo, GradientDebugger};
pub use deterministic::{
    clear_operation_seeds, get_global_seed, get_operation_seed, get_seeded_operation_count,
    hash_tensor_data, is_deterministic, reset_deterministic_state, set_deterministic,
    set_global_seed, set_operation_seed, DeterministicConfig, DeterministicContext,
    DeterministicOperation, ReproducibilityChecker, ReproducibilityStats, SeedManager,
};
pub use device_placement::{
    DevicePlacementConfig, DevicePlacementOptimizer, GraphOperation, PlacementDecision,
    PlacementResult, PlacementStrategy,
};
pub use ellipsis_newaxis::{ellipsis_newaxis_backward, AdvancedIndexer, IndexSpec};
pub use error_taxonomy::{
    utils as error_utils, AutogradErrorBuilder, ErrorPatternValidator, GradientContext,
    ValidationResult,
};
pub use forward_ad::{forward_ops, DualTensor, ForwardADContext, ForwardMode};
pub use forward_reverse::{
    ComplexityEstimate, DifferentiationMode, ForwardReverseConfig, ForwardReverseDifferentiator,
};
pub use global_pooling::{
    adaptive_avg_pool2d_backward, adaptive_max_pool2d_backward,
    fractional_adaptive_avg_pool2d_backward, global_avg_pool2d_backward,
    global_max_pool2d_backward,
};
pub use gpu_gradient_expansion::{
    GpuCategoryCoverage, GpuCoverageAnalysis, GpuGradInfo, GpuGradStatus, GpuGradientPlanner,
    ImplementationPlan, ImplementationTask, Priority,
};
pub use grad_ops::{
    batch_fused_activations_forward_backward, fused_gelu_forward_backward,
    fused_log_softmax_forward_backward, fused_tanh_forward_backward,
};
#[cfg(feature = "parallel")]
pub use gradient_accumulation::{
    accumulate_gradients_distributed, DistributedGradientAccumulator, DistributedStats,
};
pub use gradient_accumulation::{accumulate_gradients_over_batch, GradientAccumulator};
pub use gradient_buffer_manager_simple::{
    global_gradient_buffer_manager, AllocationMetrics, EfficiencyMetrics, GradientBufferAllocation,
    GradientBufferConfig, GradientBufferManager, GradientMemoryStatistics,
    MemoryPressureStatistics,
};
pub use gradient_compression::{
    CompressedGradient, CompressionConfig, CompressionMethod, CompressionStats, GradientCompressor,
};
pub use gradient_ops::{
    accumulate_gradients, add_gradient_noise, average_gradients, clip_by_global_norm,
    clip_by_value, compute_gradient_statistics, has_invalid_gradients, scale_gradients,
    zero_gradients, GradientPipeline, GradientStatistics, NamedGradientAccumulator,
};
pub use gradient_visualization::{
    ColorScheme, EdgeType, GradientFlowAnalysis, GradientFlowEdge, GradientFlowIssue,
    GradientFlowNode, GradientFlowVisualizer, GradientStats, IssueType, LayoutAlgorithm, NodeType,
    OutputFormat, Severity, ValueStats, VisualizationSettings,
};
pub use graph_optimization::{
    CommunicationPlan, EnhancedGraphOptimizer, GradientFusion, GraphOptimizationConfig,
    GraphOptimizationResult, MemoryOptimization,
};
pub use hybrid_scheduler::{
    ExecutionStats, ExecutionSummary, GraphAnalysis, HybridScheduler, SchedulerConfig, StrategyCost,
};
pub use implicit_differentiation::{
    FixedPointFunction, GradientInfo, ImplicitDiffConfig, ImplicitDifferentiator, ImplicitFunction,
    OptimizationLayer,
};
pub use inplace_ops::{InPlaceOptimizer, InPlaceSequenceOptimizer};
pub use jit_compiler::{
    CompiledKernel, DeviceFeatures, GradientKernelTemplate, JitCompiler, KernelPerformance,
    KernelSignature, OptimizationLevel,
};
pub use jit_integration::{utils as jit_utils, JitConfig, JitGradientContext, JitGradientTapeExt};
pub use kernel_fusion::{FusableOp, FusedKernel, FusionStats, KernelFusionOptimizer, OpSequence};
pub use memory_diff_reporter::{MemoryDiff, MemoryDiffReporter, MemorySnapshot};
pub use memory_profiler::{get_global_profiler, GradientMemoryProfiler, MemoryReport, MemoryStats};
pub use neural_integration::{
    AutogradLayer, AutogradOptimizer, AutogradTrainer, OptimizerType, TrainingMetrics,
};
pub use no_grad::{
    enable_grad, is_grad_enabled, no_grad, set_grad_enabled, EnableGradGuard, NoGradGuard,
};
pub use numerical_checker::{
    CheckerConfig, ErrorAnalysis, FiniteDifferenceMethod, GradientCheckResult, NumericalChecker,
};
#[cfg(feature = "parallel")]
pub use parallel_gradients::{
    AsyncGradientHandle, CommunicationBackend, GradientTask, ParallelGradientConfig,
    ParallelGradientConfigBuilder, ParallelGradientEngine, ParallelGradientResult, PipelineConfig,
};
pub use parameter_server::{
    FaultToleranceMode, LoadBalancingStrategy, ParameterServer, ParameterServerClient,
    ParameterServerConfig, ParameterServerStats,
};
pub use performance_benchmark::{
    BenchmarkConfig, BenchmarkReport, BenchmarkResult, BenchmarkStatistics, BenchmarkSummary,
    ComparisonResult, PerformanceBenchmark, RegressionReport, RegressionSeverity,
    ThroughputMetrics,
};
pub use simd_grad_ops_simple::{
    global_simd_grad_ops, SimdGradConfig, SimdGradOps, SimdPerformanceMetrics,
};
pub use special_functions::{
    bessel_j0_backward, bessel_j1_backward, beta_backward, digamma_backward, erf_backward,
    erfc_backward, gamma_backward, lgamma_backward,
};
pub use subgraph_extraction::{
    ExtractionStrategy, Subgraph, SubgraphConfig, SubgraphExtractionResult, SubgraphExtractor,
    SubgraphOperation,
};
pub use tape::{GradientTape, Operation, TapeNode, TrackedTensor};
pub use tape_optimization::{TapeOptimizationConfig, TapeOptimizationStats, TapeOptimizer};
pub use tensor_ext::TensorAutograd;
pub use tensor_networks::{
    ContractionEdge, ContractionPath, ContractionStep, ContractionStrategy, TensorNetwork,
    TensorNetworkGradient, TensorNetworkNode, TensorNetworkOptimizer,
};
pub use ultra_gradient_engine_simple::{
    global_ultra_gradient_engine, GradientMemoryStats, GradientPerformanceMetrics,
    OptimizationInsights, UltraGradientConfig, UltraGradientEngine, UltraGradientResult,
    UltraGradientTapeExt,
};

use tenflowers_core::{Result, Tensor};

pub use advanced_grad_ops::{
    gradient_clipping, higher_order as advanced_higher_order, jacobian, optimization,
    AdaptiveGradientAccumulator,
};
pub use amp_policy::{
    AMPConfig, AMPPolicy, AMPStabilityMetrics, ScaleAdjustment, ScaleAdjustmentReason,
};
pub use efficient_memory::{
    AggregationStats, CheckpointStats, GradientCheckpointer, GradientMemoryManager,
    GradientMemoryPool, LazyGradient, MemoryManagerStats, MemoryPoolStats,
    StreamingGradientAggregator,
};
pub use gradient_analyzer::{
    AnalysisConfig, GradientAnalysisReport, GradientAnalyzer,
    GradientFlowAnalysis as AdvancedGradientFlowAnalysis, GradientIssue,
    GradientStatistics as AnalyzerGradientStatistics, PerformanceMetrics,
};
pub use second_order_utils::{
    compute_hessian, compute_hessian_diagonal, compute_jacobian, compute_laplacian,
    directional_second_derivative, hessian_vector_product,
};

pub trait Differentiable<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Tensor<T>>>;
    fn grad(&self) -> Option<&Tensor<T>>;
}
