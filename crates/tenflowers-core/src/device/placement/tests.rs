//! Tests for device placement module.

#[cfg(test)]
mod tests {
    use super::super::manager::{estimate_flops, estimate_memory_usage, DevicePlacement};
    use super::super::optimizer::GraphPlacementOptimizer;
    use super::super::types::{
        CostWeights, GraphOpInfo, OpCategory, OpInfo, PlacementCost, PlacementStrategy,
        PrecisionType,
    };
    use crate::Device;
    use std::time::Duration;

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

        let device = placement
            .choose_device(&op_info)
            .expect("test: choose_device should succeed");
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
            .map(|_| {
                placement
                    .choose_device(&op_info)
                    .expect("test: map should succeed")
            })
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
            input_sizes: vec![3 * 224 * 224 * 4, 64 * 3 * 7 * 7 * 4],
            output_sizes: vec![64 * 224 * 224 * 4],
            is_critical_path: true,
            parallelizable: true,
            fusion_candidates: vec!["relu".to_string()],
        };

        let op2 = GraphOpInfo {
            op_info: OpInfo {
                name: "relu".to_string(),
                input_shapes: vec![vec![1, 64, 224, 224]],
                estimated_flops: 64 * 224 * 224,
                memory_usage: 64 * 224 * 224 * 4,
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
            input_sizes: vec![64 * 224 * 224 * 4],
            output_sizes: vec![64 * 224 * 224 * 4],
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
        let placements1 = optimizer.optimize_graph_placement(std::slice::from_ref(&op_info));
        assert_eq!(optimizer.get_optimization_stats().cache_hits, 0);

        // Second optimization should hit cache
        let placements2 = optimizer.optimize_graph_placement(std::slice::from_ref(&op_info));
        assert_eq!(optimizer.get_optimization_stats().cache_hits, 1);
        assert_eq!(placements1, placements2);
    }
}
