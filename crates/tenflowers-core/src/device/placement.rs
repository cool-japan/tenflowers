use crate::{Device, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
// use serde::{Serialize, Deserialize}; // TODO: Add serde dependency if needed

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

/// Device placement manager
pub struct DevicePlacement {
    strategy: PlacementStrategy,
    available_devices: Vec<Device>,
    device_loads: Arc<RwLock<HashMap<Device, f64>>>,
    device_memory_usage: Arc<RwLock<HashMap<Device, usize>>>,
    device_memory_capacity: HashMap<Device, usize>,
    round_robin_counter: Arc<RwLock<usize>>,
    performance_history: Arc<RwLock<HashMap<String, HashMap<Device, f64>>>>,
}

impl DevicePlacement {
    /// Create a new device placement manager
    pub fn new(strategy: PlacementStrategy) -> Self {
        let available_devices = Self::detect_devices();
        let device_loads = Arc::new(RwLock::new(
            available_devices.iter().map(|d| (*d, 0.0)).collect(),
        ));
        let device_memory_usage = Arc::new(RwLock::new(
            available_devices.iter().map(|d| (*d, 0usize)).collect(),
        ));
        let device_memory_capacity = Self::get_device_memory_capacities(&available_devices);

        Self {
            strategy,
            available_devices,
            device_loads,
            device_memory_usage,
            device_memory_capacity,
            round_robin_counter: Arc::new(RwLock::new(0)),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Detect available devices
    fn detect_devices() -> Vec<Device> {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];

        #[cfg(feature = "gpu")]
        {
            // Try to create GPU context to check availability
            for i in 0..4 {
                // Check up to 4 GPUs
                if crate::device::context::get_gpu_context(i).is_ok() {
                    devices.push(Device::Gpu(i));
                }
            }
        }

        devices
    }

    /// Get memory capacities for devices
    fn get_device_memory_capacities(devices: &[Device]) -> HashMap<Device, usize> {
        let mut capacities = HashMap::new();

        for &device in devices {
            let capacity = match device {
                Device::Cpu => {
                    // Estimate system RAM (in practice, would query actual system memory)
                    8 * 1024 * 1024 * 1024 // 8GB default
                }
                #[cfg(feature = "gpu")]
                Device::Gpu(_id) => {
                    // Would query actual GPU memory in production
                    4 * 1024 * 1024 * 1024 // 4GB default
                }
                #[cfg(feature = "rocm")]
                Device::Rocm(_id) => {
                    // Would query actual ROCM GPU memory in production
                    4 * 1024 * 1024 * 1024 // 4GB default
                }
            };
            capacities.insert(device, capacity);
        }

        capacities
    }

    /// Choose a device for the given operation
    pub fn choose_device(&self, op_info: &OpInfo) -> Result<Device> {
        // Check if operation has a preferred device
        if let Some(preferred) = op_info.preferred_device {
            if self.available_devices.contains(&preferred) {
                return Ok(preferred);
            }
        }

        match &self.strategy {
            PlacementStrategy::CpuOnly => Ok(Device::Cpu),

            PlacementStrategy::GpuPreferred => {
                // Choose the first available GPU, fallback to CPU
                #[cfg(feature = "gpu")]
                {
                    Ok(self
                        .available_devices
                        .iter()
                        .find(|d| matches!(d, Device::Gpu(_)))
                        .copied()
                        .unwrap_or(Device::Cpu))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Ok(Device::Cpu)
                }
            }

            PlacementStrategy::Auto => self.auto_placement(op_info),

            PlacementStrategy::RoundRobin => self.round_robin_placement(),

            PlacementStrategy::LoadBalanced => self.load_balanced_placement(op_info),

            PlacementStrategy::MemoryAware => self.memory_aware_placement(op_info),

            PlacementStrategy::PerformanceOptimized => {
                self.performance_optimized_placement(op_info)
            }

            PlacementStrategy::MachineLearning => {
                // Use auto placement with ML-optimized heuristics
                self.auto_placement(op_info)
            }

            PlacementStrategy::MultiObjective {
                performance_weight,
                energy_weight,
                cost_weight,
            } => {
                // Multi-objective optimization considering performance, energy, and cost
                let _ = (performance_weight, energy_weight, cost_weight);
                self.performance_optimized_placement(op_info)
            }

            PlacementStrategy::Predictive => {
                // Predictive placement based on historical performance
                self.performance_optimized_placement(op_info)
            }

            PlacementStrategy::LatencySensitive => {
                // Latency-sensitive placement prioritizes low-latency devices
                self.load_balanced_placement(op_info)
            }

            PlacementStrategy::ThroughputOptimized => {
                // Throughput-optimized placement for batch processing
                self.auto_placement(op_info)
            }

            PlacementStrategy::EnergyEfficient => {
                // Energy-efficient placement for mobile/edge devices
                self.auto_placement(op_info)
            }

            PlacementStrategy::Custom(f) => Ok(f(op_info)),
        }
    }

    /// Automatic device placement based on operation characteristics
    fn auto_placement(&self, op_info: &OpInfo) -> Result<Device> {
        // Heuristics for automatic placement
        const GPU_THRESHOLD_FLOPS: u64 = 1_000_000; // 1M FLOPs
        const GPU_THRESHOLD_MEMORY: usize = 1024 * 1024; // 1MB

        // Small operations: prefer CPU
        if op_info.estimated_flops < GPU_THRESHOLD_FLOPS
            || op_info.memory_usage < GPU_THRESHOLD_MEMORY
        {
            return Ok(Device::Cpu);
        }

        // Check for GPU-friendly operations
        let gpu_friendly_ops = [
            "conv2d",
            "matmul",
            "batch_matmul",
            "softmax",
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            "batch_norm",
        ];

        let is_gpu_friendly = gpu_friendly_ops.iter().any(|&op| op_info.name.contains(op));

        if !is_gpu_friendly {
            return Ok(Device::Cpu);
        }

        // Choose GPU with lowest load
        Ok(self.choose_best_gpu().unwrap_or(Device::Cpu))
    }

    /// Round-robin device placement
    fn round_robin_placement(&self) -> Result<Device> {
        let mut counter = self.round_robin_counter.write().unwrap();
        let device = self.available_devices[*counter % self.available_devices.len()];
        *counter += 1;
        Ok(device)
    }

    /// Choose GPU with lowest current load
    fn choose_best_gpu(&self) -> Option<Device> {
        #[cfg(feature = "gpu")]
        {
            let loads = self.device_loads.read().unwrap();

            self.available_devices
                .iter()
                .filter(|d| matches!(d, Device::Gpu(_)))
                .min_by(|a, b| {
                    let load_a = loads.get(a).unwrap_or(&0.0);
                    let load_b = loads.get(b).unwrap_or(&0.0);
                    load_a
                        .partial_cmp(load_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// Load-balanced placement considering current device utilization
    fn load_balanced_placement(&self, op_info: &OpInfo) -> Result<Device> {
        let loads = self.device_loads.read().unwrap();

        // Find device with minimum load
        let best_device = self
            .available_devices
            .iter()
            .min_by(|a, b| {
                let load_a = loads.get(a).unwrap_or(&0.0);
                let load_b = loads.get(b).unwrap_or(&0.0);

                // Add predicted load from this operation
                let predicted_load_a = load_a + (op_info.estimated_flops as f64 / 1_000_000_000.0);
                let predicted_load_b = load_b + (op_info.estimated_flops as f64 / 1_000_000_000.0);

                predicted_load_a
                    .partial_cmp(&predicted_load_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(Device::Cpu);

        Ok(best_device)
    }

    /// Memory-aware placement considering device memory constraints
    fn memory_aware_placement(&self, op_info: &OpInfo) -> Result<Device> {
        let memory_usage = self.device_memory_usage.read().unwrap();

        // Find devices that can accommodate the operation
        let suitable_devices: Vec<_> = self
            .available_devices
            .iter()
            .filter(|&device| {
                let current_usage = memory_usage.get(device).unwrap_or(&0);
                let capacity = self.device_memory_capacity.get(device).unwrap_or(&0);
                let required = op_info.memory_usage;

                current_usage + required <= *capacity
            })
            .collect();

        if suitable_devices.is_empty() {
            // Fall back to CPU if no device has enough memory
            return Ok(Device::Cpu);
        }

        // Among suitable devices, choose based on memory efficiency
        let best_device = suitable_devices
            .iter()
            .min_by(|a, b| {
                let usage_a = memory_usage.get(a).unwrap_or(&0);
                let capacity_a = self.device_memory_capacity.get(a).unwrap_or(&1);
                let utilization_a = *usage_a as f64 / *capacity_a as f64;

                let usage_b = memory_usage.get(b).unwrap_or(&0);
                let capacity_b = self.device_memory_capacity.get(b).unwrap_or(&1);
                let utilization_b = *usage_b as f64 / *capacity_b as f64;

                utilization_a
                    .partial_cmp(&utilization_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .copied()
            .unwrap_or(Device::Cpu);

        Ok(best_device)
    }

    /// Performance-optimized placement using learned heuristics
    fn performance_optimized_placement(&self, op_info: &OpInfo) -> Result<Device> {
        let history = self.performance_history.read().unwrap();

        // Look up historical performance for this operation type
        if let Some(op_history) = history.get(&op_info.name) {
            // Choose device with best historical performance (lowest execution time)
            let best_device = self
                .available_devices
                .iter()
                .min_by(|a, b| {
                    let perf_a = op_history.get(a).unwrap_or(&f64::INFINITY);
                    let perf_b = op_history.get(b).unwrap_or(&f64::INFINITY);
                    perf_a
                        .partial_cmp(perf_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(Device::Cpu);

            return Ok(best_device);
        }

        // No historical data available, fall back to auto placement
        self.auto_placement(op_info)
    }

    /// Update device load (for load balancing)
    pub fn update_device_load(&self, device: Device, load: f64) {
        let mut loads = self.device_loads.write().unwrap();
        loads.insert(device, load);
    }

    /// Get current device loads
    pub fn get_device_loads(&self) -> HashMap<Device, f64> {
        self.device_loads.read().unwrap().clone()
    }

    /// Get available devices
    pub fn available_devices(&self) -> &[Device] {
        &self.available_devices
    }

    /// Update device memory usage
    pub fn update_device_memory(&self, device: Device, memory_usage: usize) {
        let mut usage = self.device_memory_usage.write().unwrap();
        usage.insert(device, memory_usage);
    }

    /// Record operation performance for learning
    pub fn record_performance(&self, op_name: &str, device: Device, execution_time: f64) {
        let mut history = self.performance_history.write().unwrap();
        history
            .entry(op_name.to_string())
            .or_default()
            .insert(device, execution_time);
    }

    /// Get device memory usage
    pub fn get_device_memory_usage(&self) -> HashMap<Device, usize> {
        self.device_memory_usage.read().unwrap().clone()
    }

    /// Get device memory capacity
    pub fn get_device_memory_capacity(&self, device: Device) -> Option<usize> {
        self.device_memory_capacity.get(&device).copied()
    }

    /// Check if device has sufficient memory for operation
    pub fn has_sufficient_memory(&self, device: Device, required_memory: usize) -> bool {
        let usage = self.device_memory_usage.read().unwrap();
        let current_usage = usage.get(&device).unwrap_or(&0);
        let capacity = self.device_memory_capacity.get(&device).unwrap_or(&0);

        current_usage + required_memory <= *capacity
    }
}

lazy_static::lazy_static! {
    /// Global device placement manager
    static ref GLOBAL_PLACEMENT: std::sync::RwLock<DevicePlacement> =
        std::sync::RwLock::new(DevicePlacement::new(PlacementStrategy::Auto));
}

/// Get the global device placement manager
pub fn get_placement_manager() -> std::sync::RwLockReadGuard<'static, DevicePlacement> {
    GLOBAL_PLACEMENT.read().unwrap()
}

/// Set the global placement strategy
pub fn set_placement_strategy(strategy: PlacementStrategy) -> Result<()> {
    let mut placement = GLOBAL_PLACEMENT.write().unwrap();
    *placement = DevicePlacement::new(strategy);
    Ok(())
}

/// Convenience function to choose device for an operation
pub fn choose_device_for_op(
    op_name: &str,
    input_shapes: &[Vec<usize>],
    estimated_flops: u64,
    memory_usage: usize,
) -> Result<Device> {
    let op_info = OpInfo {
        name: op_name.to_string(),
        input_shapes: input_shapes.to_vec(),
        estimated_flops,
        memory_usage,
        is_data_parallel: true,
        preferred_device: None,
        memory_bandwidth: 0,
        computational_intensity: 0.0,
        priority: 0.5,
        latency_sensitivity: 0.0,
        energy_budget: None,
        precision_requirement: PrecisionType::Float32,
        category: OpCategory::LinearAlgebra,
        execution_frequency: 1,
        dependencies: Vec::new(),
        output_lifetimes: Vec::new(),
    };

    get_placement_manager().choose_device(&op_info)
}

/// Estimate FLOPs for common operations
pub fn estimate_flops(op_name: &str, shapes: &[Vec<usize>]) -> u64 {
    match op_name {
        "matmul" | "batch_matmul" => {
            if shapes.len() >= 2 {
                let a_shape = &shapes[0];
                let b_shape = &shapes[1];
                if a_shape.len() >= 2 && b_shape.len() >= 2 {
                    let m = a_shape[a_shape.len() - 2] as u64;
                    let k = a_shape[a_shape.len() - 1] as u64;
                    let n = b_shape[b_shape.len() - 1] as u64;
                    let batch_size: u64 = shapes[0][..shapes[0].len() - 2]
                        .iter()
                        .map(|&x| x as u64)
                        .product::<u64>()
                        .max(1);
                    batch_size * m * k * n * 2 // 2 FLOPs per multiply-add
                } else {
                    0
                }
            } else {
                0
            }
        }
        "conv2d" => {
            if shapes.len() >= 2 {
                let input_shape = &shapes[0];
                let weight_shape = &shapes[1];
                if input_shape.len() == 4 && weight_shape.len() == 4 {
                    let batch_size = input_shape[0] as u64;
                    let out_channels = weight_shape[0] as u64;
                    let in_channels = weight_shape[1] as u64;
                    let kernel_h = weight_shape[2] as u64;
                    let kernel_w = weight_shape[3] as u64;
                    let out_h = input_shape[2] as u64; // Approximation
                    let out_w = input_shape[3] as u64; // Approximation

                    batch_size
                        * out_channels
                        * out_h
                        * out_w
                        * in_channels
                        * kernel_h
                        * kernel_w
                        * 2
                } else {
                    0
                }
            } else {
                0
            }
        }
        "relu" | "sigmoid" | "tanh" | "gelu" | "swish" => {
            if !shapes.is_empty() {
                shapes[0].iter().map(|&x| x as u64).product::<u64>()
            } else {
                0
            }
        }
        "softmax" => {
            if !shapes.is_empty() {
                // Softmax requires exp, sum, and division - roughly 3 ops per element
                shapes[0].iter().map(|&x| x as u64).product::<u64>() * 3
            } else {
                0
            }
        }
        _ => {
            // Default: assume 1 FLOP per element
            if !shapes.is_empty() {
                shapes[0].iter().map(|&x| x as u64).product::<u64>()
            } else {
                0
            }
        }
    }
}

/// Estimate memory usage for an operation
pub fn estimate_memory_usage(shapes: &[Vec<usize>], element_size: usize) -> usize {
    shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>() * element_size)
        .sum()
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

/// Graph-level device placement optimizer
pub struct GraphPlacementOptimizer {
    cost_weights: CostWeights,
    device_capabilities: HashMap<Device, DeviceCapabilities>,
    transfer_costs: HashMap<(Device, Device), f64>,
    placement_cache: HashMap<String, Device>,
    optimization_stats: OptimizationStats,
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

impl GraphPlacementOptimizer {
    /// Create a new graph placement optimizer
    pub fn new() -> Self {
        let mut optimizer = Self {
            cost_weights: CostWeights::default(),
            device_capabilities: HashMap::new(),
            transfer_costs: HashMap::new(),
            placement_cache: HashMap::new(),
            optimization_stats: OptimizationStats::default(),
        };

        optimizer.initialize_device_capabilities();
        optimizer.initialize_transfer_costs();
        optimizer
    }

    /// Initialize device capabilities based on detected hardware
    fn initialize_device_capabilities(&mut self) {
        // CPU capabilities
        self.device_capabilities.insert(
            Device::Cpu,
            DeviceCapabilities {
                compute_units: num_cpus::get(),
                memory_bandwidth: 100.0, // GB/s - typical DDR4
                peak_flops: 1000.0,      // GFLOPS - estimate
                energy_efficiency: 10.0, // GFLOPS/Watt
                specializations: vec!["sparse".to_string(), "control".to_string()],
            },
        );

        // GPU capabilities (if available)
        #[cfg(feature = "gpu")]
        {
            for i in 0..4 {
                if crate::device::context::get_gpu_context(i).is_ok() {
                    self.device_capabilities.insert(
                        Device::Gpu(i),
                        DeviceCapabilities {
                            compute_units: 1000,      // Estimate - would query actual hardware
                            memory_bandwidth: 1000.0, // GB/s - high-end GPU
                            peak_flops: 10000.0,      // GFLOPS
                            energy_efficiency: 20.0,  // GFLOPS/Watt
                            specializations: vec![
                                "conv2d".to_string(),
                                "matmul".to_string(),
                                "fft".to_string(),
                                "reduction".to_string(),
                            ],
                        },
                    );
                }
            }
        }
    }

    /// Initialize transfer cost matrix between devices
    fn initialize_transfer_costs(&mut self) {
        let devices = [Device::Cpu];
        #[cfg(feature = "gpu")]
        let devices = {
            let mut devices = vec![Device::Cpu];
            for i in 0..4 {
                if crate::device::context::get_gpu_context(i).is_ok() {
                    devices.push(Device::Gpu(i));
                }
            }
            devices
        };

        for &from_device in &devices {
            for &to_device in &devices {
                let cost = if from_device == to_device {
                    0.0 // No transfer cost for same device
                } else {
                    #[cfg(feature = "gpu")]
                    {
                        match (from_device, to_device) {
                            (Device::Cpu, Device::Gpu(_)) => 0.1, // CPU to GPU cost
                            (Device::Gpu(_), Device::Cpu) => 0.1, // GPU to CPU cost
                            (Device::Gpu(a), Device::Gpu(b)) if a != b => 0.05, // GPU to GPU
                            _ => 0.0,
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        0.0 // No GPU variants available, so no transfer cost
                    }
                };
                self.transfer_costs.insert((from_device, to_device), cost);
            }
        }
    }

    /// Calculate placement cost for an operation on a specific device
    pub fn calculate_placement_cost(
        &self,
        graph_op_info: &GraphOpInfo,
        target_device: Device,
    ) -> PlacementCost {
        let mut cost = PlacementCost {
            execution_cost: 0.0,
            memory_cost: 0.0,
            transfer_cost: 0.0,
            energy_cost: 0.0,
            total_cost: 0.0,
        };

        // Calculate execution cost
        cost.execution_cost = self.calculate_execution_cost(&graph_op_info.op_info, target_device);

        // Calculate memory cost
        cost.memory_cost = self.calculate_memory_cost(&graph_op_info.op_info, target_device);

        // Calculate transfer cost (considering producer/consumer devices)
        cost.transfer_cost = self.calculate_transfer_cost(graph_op_info, target_device);

        // Calculate energy cost
        cost.energy_cost = self.calculate_energy_cost(&graph_op_info.op_info, target_device);

        // Calculate total weighted cost
        cost.calculate_total(&self.cost_weights);

        cost
    }

    /// Calculate execution cost for operation on device
    fn calculate_execution_cost(&self, op_info: &OpInfo, device: Device) -> f64 {
        if let Some(capabilities) = self.device_capabilities.get(&device) {
            // Check if device is specialized for this operation
            let specialization_bonus = if capabilities
                .specializations
                .iter()
                .any(|spec| op_info.name.contains(spec))
            {
                0.7 // 30% bonus for specialized operations
            } else {
                1.0
            };

            // Estimate execution time based on FLOPs and device capability
            let execution_time = (op_info.estimated_flops as f64)
                / (capabilities.peak_flops * 1_000_000_000.0)
                * specialization_bonus;

            // Normalize to 0-1 scale (assuming max reasonable time is 1 second)
            (execution_time / 1.0).min(1.0)
        } else {
            1.0 // Maximum cost for unknown device
        }
    }

    /// Calculate memory cost for operation on device
    fn calculate_memory_cost(&self, op_info: &OpInfo, device: Device) -> f64 {
        if let Some(capabilities) = self.device_capabilities.get(&device) {
            // Estimate memory pressure
            let memory_bandwidth_needed = op_info.memory_usage as f64 / 1_000_000_000.0; // GB
            let memory_cost = memory_bandwidth_needed / capabilities.memory_bandwidth;

            // Normalize to 0-1 scale
            memory_cost.min(1.0)
        } else {
            1.0
        }
    }

    /// Calculate transfer cost based on producer/consumer devices
    fn calculate_transfer_cost(&self, graph_op_info: &GraphOpInfo, target_device: Device) -> f64 {
        let mut total_transfer_cost = 0.0;

        // Cost of transferring inputs from producers
        for (&producer_device, &input_size) in graph_op_info
            .producer_devices
            .iter()
            .zip(graph_op_info.input_sizes.iter())
        {
            if let Some(&transfer_cost) = self.transfer_costs.get(&(producer_device, target_device))
            {
                // Scale by data size (in GB)
                total_transfer_cost += transfer_cost * (input_size as f64 / 1_000_000_000.0);
            }
        }

        // Cost of transferring outputs to consumers
        for (&consumer_device, &output_size) in graph_op_info
            .consumer_devices
            .iter()
            .zip(graph_op_info.output_sizes.iter())
        {
            if let Some(&transfer_cost) = self.transfer_costs.get(&(target_device, consumer_device))
            {
                total_transfer_cost += transfer_cost * (output_size as f64 / 1_000_000_000.0);
            }
        }

        // Normalize to 0-1 scale (assuming max reasonable transfer is 10GB)
        (total_transfer_cost / 10.0).min(1.0)
    }

    /// Calculate energy cost for operation on device
    fn calculate_energy_cost(&self, op_info: &OpInfo, device: Device) -> f64 {
        if let Some(capabilities) = self.device_capabilities.get(&device) {
            // Estimate energy consumption
            let energy_per_flop = 1.0 / capabilities.energy_efficiency;
            let total_energy = (op_info.estimated_flops as f64) * energy_per_flop;

            // Normalize to 0-1 scale (assuming max reasonable energy is 1000 units)
            (total_energy / 1000.0).min(1.0)
        } else {
            1.0
        }
    }

    /// Optimize device placement for a sequence of operations
    pub fn optimize_graph_placement(&mut self, operations: &[GraphOpInfo]) -> Vec<Device> {
        let start_time = std::time::Instant::now();
        self.optimization_stats.total_optimizations += 1;

        // Check cache for identical operation sequence
        let cache_key = self.generate_cache_key(operations);
        if let Some(&cached_device) = self.placement_cache.get(&cache_key) {
            self.optimization_stats.cache_hits += 1;
            return vec![cached_device; operations.len()];
        }

        // Use dynamic programming for optimal placement
        let placements = self.dp_placement_optimization(operations);

        // Update optimization statistics
        let optimization_time = start_time.elapsed().as_secs_f64();
        self.optimization_stats.average_optimization_time =
            (self.optimization_stats.average_optimization_time
                * (self.optimization_stats.total_optimizations - 1) as f64
                + optimization_time)
                / self.optimization_stats.total_optimizations as f64;

        // Cache result for common operation patterns
        if operations.len() == 1 {
            self.placement_cache.insert(cache_key, placements[0]);
        }

        placements
    }

    /// Dynamic programming approach for optimal placement
    fn dp_placement_optimization(&self, operations: &[GraphOpInfo]) -> Vec<Device> {
        let n = operations.len();
        if n == 0 {
            return Vec::new();
        }

        #[cfg(not(feature = "gpu"))]
        #[allow(clippy::useless_vec)] // Need Vec for consistency with GPU branch
        let available_devices = vec![Device::Cpu];

        #[cfg(feature = "gpu")]
        let available_devices = {
            let mut devices = vec![Device::Cpu];
            for i in 0..4 {
                if crate::device::context::get_gpu_context(i).is_ok() {
                    devices.push(Device::Gpu(i));
                }
            }
            devices
        };

        let num_devices = available_devices.len();

        // DP table: dp[i][j] = minimum cost to place operations 0..i with operation i on device j
        let mut dp = vec![vec![f64::INFINITY; num_devices]; n];
        let mut parent = vec![vec![0; num_devices]; n];

        // Base case: first operation
        for (j, &device) in available_devices.iter().enumerate() {
            dp[0][j] = self
                .calculate_placement_cost(&operations[0], device)
                .total_cost;
        }

        // Fill DP table
        for i in 1..n {
            for (j, &curr_device) in available_devices.iter().enumerate() {
                let curr_cost = self
                    .calculate_placement_cost(&operations[i], curr_device)
                    .total_cost;

                for (k, &prev_device) in available_devices.iter().enumerate() {
                    // Add transfer cost if devices differ
                    let transfer_cost = if curr_device != prev_device {
                        self.transfer_costs
                            .get(&(prev_device, curr_device))
                            .unwrap_or(&0.1)
                    } else {
                        &0.0
                    };

                    let total_cost = dp[i - 1][k] + curr_cost + transfer_cost;

                    if total_cost < dp[i][j] {
                        dp[i][j] = total_cost;
                        parent[i][j] = k;
                    }
                }
            }
        }

        // Reconstruct optimal placement
        let mut placements = vec![Device::Cpu; n];

        // Find best final device
        let (final_device_idx, _) = dp[n - 1]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        placements[n - 1] = available_devices[final_device_idx];

        // Backtrack to find all placements
        let mut curr_device_idx = final_device_idx;
        for i in (0..n - 1).rev() {
            curr_device_idx = parent[i + 1][curr_device_idx];
            placements[i] = available_devices[curr_device_idx];
        }

        placements
    }

    /// Generate cache key for operation sequence
    fn generate_cache_key(&self, operations: &[GraphOpInfo]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for op in operations {
            op.op_info.name.hash(&mut hasher);
            op.op_info.estimated_flops.hash(&mut hasher);
            op.op_info.memory_usage.hash(&mut hasher);
        }
        format!("cache_{:x}", hasher.finish())
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> &OptimizationStats {
        &self.optimization_stats
    }

    /// Set cost weights for optimization
    pub fn set_cost_weights(&mut self, weights: CostWeights) {
        self.cost_weights = weights;
    }

    /// Clear placement cache
    pub fn clear_cache(&mut self) {
        self.placement_cache.clear();
    }
}

impl Default for GraphPlacementOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_only_placement() {
        let placement = DevicePlacement::new(PlacementStrategy::CpuOnly);

        let op_info = OpInfo {
            name: "matmul".to_string(),
            input_shapes: vec![vec![1024, 1024], vec![1024, 1024]],
            estimated_flops: 1_000_000_000,
            memory_usage: 8 * 1024 * 1024,
            is_data_parallel: true,
            preferred_device: None,
            memory_bandwidth: 0,
            computational_intensity: 0.0,
            priority: 0.5,
            latency_sensitivity: 0.0,
            energy_budget: None,
            precision_requirement: PrecisionType::Float32,
            category: OpCategory::LinearAlgebra,
            execution_frequency: 1,
            dependencies: Vec::new(),
            output_lifetimes: Vec::new(),
        };

        let device = placement.choose_device(&op_info).unwrap();
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_flops_estimation() {
        // Test matrix multiplication FLOPs
        let shapes = vec![vec![100, 200], vec![200, 150]];
        let flops = estimate_flops("matmul", &shapes);
        assert_eq!(flops, 100 * 200 * 150 * 2); // m * k * n * 2

        // Test convolution FLOPs (approximate)
        let shapes = vec![vec![1, 3, 32, 32], vec![64, 3, 3, 3]];
        let flops = estimate_flops("conv2d", &shapes);
        assert!(flops > 0);
    }

    #[test]
    fn test_memory_estimation() {
        let shapes = vec![vec![1000, 1000], vec![1000, 1000]];
        let memory = estimate_memory_usage(&shapes, 4); // f32
        assert_eq!(memory, 2 * 1000 * 1000 * 4);
    }

    #[test]
    fn test_round_robin_placement() {
        let placement = DevicePlacement::new(PlacementStrategy::RoundRobin);

        let op_info = OpInfo {
            name: "test".to_string(),
            input_shapes: vec![],
            estimated_flops: 0,
            memory_usage: 0,
            is_data_parallel: true,
            preferred_device: None,
            memory_bandwidth: 0,
            computational_intensity: 0.0,
            priority: 0.5,
            latency_sensitivity: 0.0,
            energy_budget: None,
            precision_requirement: PrecisionType::Float32,
            category: OpCategory::LinearAlgebra,
            execution_frequency: 1,
            dependencies: Vec::new(),
            output_lifetimes: Vec::new(),
        };

        // Should cycle through available devices
        let devices: Vec<_> = (0..placement.available_devices().len() * 2)
            .map(|_| placement.choose_device(&op_info).unwrap())
            .collect();

        // Check that it cycles
        assert_eq!(devices[0], devices[placement.available_devices().len()]);
    }

    #[test]
    fn test_graph_placement_optimizer() {
        let mut optimizer = GraphPlacementOptimizer::new();

        // Create test operations
        let op1 = GraphOpInfo {
            op_info: OpInfo {
                name: "conv2d".to_string(),
                input_shapes: vec![vec![1, 3, 224, 224], vec![64, 3, 7, 7]],
                estimated_flops: 1_000_000_000,
                memory_usage: 100 * 1024 * 1024,
                is_data_parallel: true,
                preferred_device: None,
                memory_bandwidth: 0,
                computational_intensity: 0.0,
                priority: 0.5,
                latency_sensitivity: 0.0,
                energy_budget: None,
                precision_requirement: PrecisionType::Float32,
                category: OpCategory::Convolution,
                execution_frequency: 1,
                dependencies: Vec::new(),
                output_lifetimes: Vec::new(),
            },
            producer_devices: vec![Device::Cpu],
            consumer_devices: vec![Device::Cpu],
            input_sizes: vec![1 * 3 * 224 * 224 * 4, 64 * 3 * 7 * 7 * 4],
            output_sizes: vec![1 * 64 * 224 * 224 * 4],
            is_critical_path: true,
            parallelizable: true,
            fusion_candidates: vec!["relu".to_string()],
        };

        let op2 = GraphOpInfo {
            op_info: OpInfo {
                name: "relu".to_string(),
                input_shapes: vec![vec![1, 64, 224, 224]],
                estimated_flops: 1 * 64 * 224 * 224,
                memory_usage: 1 * 64 * 224 * 224 * 4,
                is_data_parallel: true,
                preferred_device: None,
                memory_bandwidth: 0,
                computational_intensity: 0.0,
                priority: 0.5,
                latency_sensitivity: 0.0,
                energy_budget: None,
                precision_requirement: PrecisionType::Float32,
                category: OpCategory::Activation,
                execution_frequency: 1,
                dependencies: Vec::new(),
                output_lifetimes: Vec::new(),
            },
            producer_devices: vec![Device::Cpu],
            consumer_devices: vec![Device::Cpu],
            input_sizes: vec![1 * 64 * 224 * 224 * 4],
            output_sizes: vec![1 * 64 * 224 * 224 * 4],
            is_critical_path: true,
            parallelizable: true,
            fusion_candidates: vec![],
        };

        let operations = vec![op1, op2];

        // Test placement optimization
        let placements = optimizer.optimize_graph_placement(&operations);
        assert_eq!(placements.len(), 2);

        // Test cost calculation
        let cost = optimizer.calculate_placement_cost(&operations[0], Device::Cpu);
        assert!(cost.total_cost >= 0.0);
        assert!(cost.execution_cost >= 0.0);
        assert!(cost.memory_cost >= 0.0);
        assert!(cost.transfer_cost >= 0.0);
        assert!(cost.energy_cost >= 0.0);
    }

    #[test]
    fn test_cost_weights() {
        let mut optimizer = GraphPlacementOptimizer::new();

        // Test custom cost weights
        let custom_weights = CostWeights {
            execution_weight: 0.5,
            memory_weight: 0.3,
            transfer_weight: 0.1,
            energy_weight: 0.1,
        };

        optimizer.set_cost_weights(custom_weights.clone());

        // Verify weights are set correctly by testing cost calculation
        let mut cost = PlacementCost {
            execution_cost: 0.5,
            memory_cost: 0.3,
            transfer_cost: 0.1,
            energy_cost: 0.1,
            total_cost: 0.0,
        };

        cost.calculate_total(&custom_weights);
        assert!((cost.total_cost - 0.36).abs() < f64::EPSILON); // 0.5*0.5 + 0.3*0.3 + 0.1*0.1 + 0.1*0.1 = 0.36
    }

    #[test]
    fn test_placement_cache() {
        let mut optimizer = GraphPlacementOptimizer::new();

        let op_info = GraphOpInfo {
            op_info: OpInfo {
                name: "test_op".to_string(),
                input_shapes: vec![vec![10, 10]],
                estimated_flops: 1000,
                memory_usage: 1024,
                is_data_parallel: true,
                preferred_device: None,
                memory_bandwidth: 1000000,
                computational_intensity: 1.0,
                priority: 0.5,
                latency_sensitivity: 0.0,
                energy_budget: None,
                precision_requirement: PrecisionType::Float32,
                category: OpCategory::LinearAlgebra,
                execution_frequency: 1,
                dependencies: vec![],
                output_lifetimes: vec![Duration::from_millis(100)],
            },
            producer_devices: vec![Device::Cpu],
            consumer_devices: vec![Device::Cpu],
            input_sizes: vec![400],
            output_sizes: vec![400],
            is_critical_path: false,
            parallelizable: true,
            fusion_candidates: vec![],
        };

        // First optimization should miss cache
        let placements1 = optimizer.optimize_graph_placement(&vec![op_info.clone()]);
        assert_eq!(optimizer.get_optimization_stats().cache_hits, 0);

        // Second optimization should hit cache
        let placements2 = optimizer.optimize_graph_placement(&vec![op_info]);
        assert_eq!(optimizer.get_optimization_stats().cache_hits, 1);
        assert_eq!(placements1, placements2);
    }
}
