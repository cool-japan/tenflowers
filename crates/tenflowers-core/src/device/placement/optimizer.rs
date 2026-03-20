//! Graph-level device placement optimizer.

use super::types::{
    CostWeights, DeviceCapabilities, GraphOpInfo, OpInfo, OptimizationStats, PlacementCost,
};
use crate::Device;
use std::collections::HashMap;

/// Graph-level device placement optimizer
pub struct GraphPlacementOptimizer {
    pub(super) cost_weights: CostWeights,
    pub(super) device_capabilities: HashMap<Device, DeviceCapabilities>,
    pub(super) transfer_costs: HashMap<(Device, Device), f64>,
    pub(super) placement_cache: HashMap<String, Device>,
    pub(super) optimization_stats: OptimizationStats,
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
            .expect("dp table should have at least one device option");

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
