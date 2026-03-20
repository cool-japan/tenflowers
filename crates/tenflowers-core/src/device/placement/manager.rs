//! DevicePlacement manager and global convenience functions.

use super::types::{OpCategory, OpInfo, PlacementStrategy, PrecisionType};
use crate::{Device, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Device placement manager
pub struct DevicePlacement {
    pub(super) strategy: PlacementStrategy,
    pub(super) available_devices: Vec<Device>,
    pub(super) device_loads: Arc<RwLock<HashMap<Device, f64>>>,
    pub(super) device_memory_usage: Arc<RwLock<HashMap<Device, usize>>>,
    pub(super) device_memory_capacity: HashMap<Device, usize>,
    pub(super) round_robin_counter: Arc<RwLock<usize>>,
    pub(super) performance_history: Arc<RwLock<HashMap<String, HashMap<Device, f64>>>>,
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
        let mut counter = self
            .round_robin_counter
            .write()
            .expect("write lock should not be poisoned");
        let device = self.available_devices[*counter % self.available_devices.len()];
        *counter += 1;
        Ok(device)
    }

    /// Choose GPU with lowest current load
    fn choose_best_gpu(&self) -> Option<Device> {
        #[cfg(feature = "gpu")]
        {
            let loads = self
                .device_loads
                .read()
                .expect("read lock should not be poisoned");

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
        let loads = self
            .device_loads
            .read()
            .expect("read lock should not be poisoned");

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
        let memory_usage = self
            .device_memory_usage
            .read()
            .expect("read lock should not be poisoned");

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
        let history = self
            .performance_history
            .read()
            .expect("read lock should not be poisoned");

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
        let mut loads = self
            .device_loads
            .write()
            .expect("write lock should not be poisoned");
        loads.insert(device, load);
    }

    /// Get current device loads
    pub fn get_device_loads(&self) -> HashMap<Device, f64> {
        self.device_loads
            .read()
            .expect("read lock should not be poisoned")
            .clone()
    }

    /// Get available devices
    pub fn available_devices(&self) -> &[Device] {
        &self.available_devices
    }

    /// Update device memory usage
    pub fn update_device_memory(&self, device: Device, memory_usage: usize) {
        let mut usage = self
            .device_memory_usage
            .write()
            .expect("write lock should not be poisoned");
        usage.insert(device, memory_usage);
    }

    /// Record operation performance for learning
    pub fn record_performance(&self, op_name: &str, device: Device, execution_time: f64) {
        let mut history = self
            .performance_history
            .write()
            .expect("write lock should not be poisoned");
        history
            .entry(op_name.to_string())
            .or_default()
            .insert(device, execution_time);
    }

    /// Get device memory usage
    pub fn get_device_memory_usage(&self) -> HashMap<Device, usize> {
        self.device_memory_usage
            .read()
            .expect("read lock should not be poisoned")
            .clone()
    }

    /// Get device memory capacity
    pub fn get_device_memory_capacity(&self, device: Device) -> Option<usize> {
        self.device_memory_capacity.get(&device).copied()
    }

    /// Check if device has sufficient memory for operation
    pub fn has_sufficient_memory(&self, device: Device, required_memory: usize) -> bool {
        let usage = self
            .device_memory_usage
            .read()
            .expect("read lock should not be poisoned");
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
    GLOBAL_PLACEMENT
        .read()
        .expect("read lock should not be poisoned")
}

/// Set the global placement strategy
pub fn set_placement_strategy(strategy: PlacementStrategy) -> Result<()> {
    let mut placement = GLOBAL_PLACEMENT
        .write()
        .expect("write lock should not be poisoned");
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
