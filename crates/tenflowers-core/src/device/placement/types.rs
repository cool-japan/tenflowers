//! Placement types: enums and structs for device placement.

use crate::Device;
use std::sync::Arc;
use std::time::Duration;

/// Ultra-performance device placement strategy with ML-based optimization
#[derive(Clone)]
pub enum PlacementStrategy {
    /// Place operations on CPU only
    CpuOnly,
    /// Place operations on GPU if available, otherwise CPU
    GpuPreferred,
    /// Automatically choose best device based on operation and data size
    Auto,
    /// Round-robin across available devices
    RoundRobin,
    /// Load-balanced placement based on current device utilization
    LoadBalanced,
    /// Memory-aware placement considering device memory constraints
    MemoryAware,
    /// Performance-optimized placement using learned heuristics
    PerformanceOptimized,
    /// ML-based adaptive placement with real-time learning
    MachineLearning,
    /// Multi-objective optimization (performance, energy, cost)
    MultiObjective {
        performance_weight: f64,
        energy_weight: f64,
        cost_weight: f64,
    },
    /// Predictive placement based on operation sequences
    Predictive,
    /// Latency-sensitive placement for real-time applications
    LatencySensitive,
    /// Throughput-optimized placement for batch processing
    ThroughputOptimized,
    /// Energy-efficient placement for mobile/edge devices
    EnergyEfficient,
    /// Custom placement function
    Custom(Arc<dyn Fn(&OpInfo) -> Device + Send + Sync>),
}

impl std::fmt::Debug for PlacementStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlacementStrategy::CpuOnly => write!(f, "CpuOnly"),
            PlacementStrategy::GpuPreferred => write!(f, "GpuPreferred"),
            PlacementStrategy::Auto => write!(f, "Auto"),
            PlacementStrategy::RoundRobin => write!(f, "RoundRobin"),
            PlacementStrategy::LoadBalanced => write!(f, "LoadBalanced"),
            PlacementStrategy::MemoryAware => write!(f, "MemoryAware"),
            PlacementStrategy::PerformanceOptimized => write!(f, "PerformanceOptimized"),
            PlacementStrategy::MachineLearning => write!(f, "MachineLearning"),
            PlacementStrategy::MultiObjective {
                performance_weight,
                energy_weight,
                cost_weight,
            } => {
                write!(
                    f,
                    "MultiObjective(perf={:.2}, energy={:.2}, cost={:.2})",
                    performance_weight, energy_weight, cost_weight
                )
            }
            PlacementStrategy::Predictive => write!(f, "Predictive"),
            PlacementStrategy::LatencySensitive => write!(f, "LatencySensitive"),
            PlacementStrategy::ThroughputOptimized => write!(f, "ThroughputOptimized"),
            PlacementStrategy::EnergyEfficient => write!(f, "EnergyEfficient"),
            PlacementStrategy::Custom(_) => write!(f, "Custom(...)"),
        }
    }
}

/// Ultra-comprehensive operation information for intelligent placement decisions
#[derive(Debug, Clone)]
pub struct OpInfo {
    pub name: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub estimated_flops: u64,
    pub memory_usage: usize,
    pub is_data_parallel: bool,
    pub preferred_device: Option<Device>,
    /// Memory bandwidth requirements (bytes per second)
    pub memory_bandwidth: u64,
    /// Computational intensity (FLOPs per byte)
    pub computational_intensity: f64,
    /// Operation priority (0.0 to 1.0)
    pub priority: f64,
    /// Latency sensitivity (0.0 = batch, 1.0 = real-time)
    pub latency_sensitivity: f64,
    /// Energy budget constraint (Watts)
    pub energy_budget: Option<f64>,
    /// Required precision (f16, f32, f64)
    pub precision_requirement: PrecisionType,
    /// Operation category for specialized optimization
    pub category: OpCategory,
    /// Expected execution frequency
    pub execution_frequency: u64,
    /// Dependencies on other operations
    pub dependencies: Vec<String>,
    /// Output tensor lifetimes (for memory optimization)
    pub output_lifetimes: Vec<Duration>,
}

/// Precision requirements for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Mixed, // Operation supports multiple precisions
}

/// Operation categories for specialized optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCategory {
    LinearAlgebra, // Matrix operations, BLAS
    Convolution,   // Conv2D, Conv3D, etc.
    Activation,    // ReLU, Sigmoid, etc.
    Normalization, // BatchNorm, LayerNorm
    Pooling,       // MaxPool, AvgPool
    Reduction,     // Sum, Mean, etc.
    ElementWise,   // Add, Mul, etc.
    Memory,        // Reshape, Transpose
    Control,       // Conditional, Loop
    Custom,        // User-defined operations
}

/// Cost model for device placement decisions
#[derive(Debug, Clone)]
pub struct PlacementCost {
    /// Execution time cost (normalized)
    pub execution_cost: f64,
    /// Memory usage cost (normalized)
    pub memory_cost: f64,
    /// Data transfer cost (normalized)
    pub transfer_cost: f64,
    /// Energy consumption cost (normalized)
    pub energy_cost: f64,
    /// Total weighted cost
    pub total_cost: f64,
}

impl PlacementCost {
    /// Calculate total cost with weights
    pub fn calculate_total(&mut self, weights: &CostWeights) {
        self.total_cost = self.execution_cost * weights.execution_weight
            + self.memory_cost * weights.memory_weight
            + self.transfer_cost * weights.transfer_weight
            + self.energy_cost * weights.energy_weight;
    }
}

/// Weights for different cost components
#[derive(Debug, Clone)]
pub struct CostWeights {
    pub execution_weight: f64,
    pub memory_weight: f64,
    pub transfer_weight: f64,
    pub energy_weight: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            execution_weight: 0.4, // Prioritize execution time
            memory_weight: 0.3,    // Memory efficiency important
            transfer_weight: 0.2,  // Consider transfer costs
            energy_weight: 0.1,    // Energy is less critical for most cases
        }
    }
}

/// Advanced placement information including graph context
#[derive(Debug, Clone)]
pub struct GraphOpInfo {
    pub op_info: OpInfo,
    pub producer_devices: Vec<Device>,
    pub consumer_devices: Vec<Device>,
    pub input_sizes: Vec<usize>,
    pub output_sizes: Vec<usize>,
    pub is_critical_path: bool,
    pub parallelizable: bool,
    pub fusion_candidates: Vec<String>,
}

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub compute_units: usize,
    pub memory_bandwidth: f64,        // GB/s
    pub peak_flops: f64,              // GFLOPS
    pub energy_efficiency: f64,       // GFLOPS/Watt
    pub specializations: Vec<String>, // e.g., ["conv2d", "matmul", "fft"]
}

/// Optimization statistics for analysis
#[derive(Debug, Default)]
pub struct OptimizationStats {
    pub total_optimizations: usize,
    pub cache_hits: usize,
    pub average_optimization_time: f64,
    pub cost_improvements: Vec<f64>,
}
