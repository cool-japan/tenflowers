//! Integration tests for NUMA-aware scheduling functionality

use tenflowers_core::Tensor;
use tenflowers_dataset::{
    Dataset, EnhancedDataLoaderBuilder, NumaAssignmentStrategy, NumaConfig, SequentialSampler,
    TensorDataset,
};

#[test]
fn test_numa_scheduler_creation() {
    use tenflowers_dataset::NumaScheduler;

    let config = NumaConfig::default();
    let scheduler = NumaScheduler::new(config);
    assert!(scheduler.is_ok());
}

#[test]
fn test_numa_dataloader_basic() {
    let features =
        Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
    let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap();

    let dataset = TensorDataset::new(features, labels);
    let sampler = SequentialSampler::new();

    let loader = EnhancedDataLoaderBuilder::new()
        .batch_size(2)
        .num_workers(2)
        .numa_config(NumaConfig::default())
        .build(dataset, sampler);

    assert!(loader.is_ok());

    let loader = loader.unwrap();

    // Test that loader can process at least one batch
    let mut batch_count = 0;

    for batch in loader {
        assert!(batch.is_ok(), "Batch processing should succeed");
        batch_count += 1;

        // Just ensure we can process at least one batch successfully
        if batch_count >= 1 {
            break;
        }
    }

    // We should have processed at least one batch
    assert!(
        batch_count >= 1,
        "Expected at least 1 batch, got {}",
        batch_count
    );
}

#[test]
fn test_numa_assignment_strategies() {
    let strategies = vec![
        NumaAssignmentStrategy::RoundRobin,
        NumaAssignmentStrategy::FillFirst,
        NumaAssignmentStrategy::Interleave,
        NumaAssignmentStrategy::LoadBalanced,
    ];

    for strategy in strategies {
        let config = NumaConfig {
            enabled: true,
            assignment_strategy: strategy,
            strict_affinity: false,
            preferred_nodes: Vec::new(),
            balance_nodes: true,
        };

        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let sampler = SequentialSampler::new();

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(1)
            .num_workers(2)
            .numa_config(config)
            .build(dataset, sampler);

        assert!(loader.is_ok());
    }
}

#[test]
fn test_numa_topology_detection() {
    use tenflowers_dataset::NumaTopology;

    let topology = NumaTopology::detect();

    // Should always detect at least one node (even if pseudo-NUMA)
    assert!(!topology.nodes.is_empty());
    assert!(topology.total_cores > 0);

    // Each node should have some CPU cores
    for node in &topology.nodes {
        assert!(!node.cpu_cores.is_empty());
    }
}

#[test]
fn test_numa_stats_collection() {
    let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

    let dataset = TensorDataset::new(features, labels);
    let sampler = SequentialSampler::new();

    let loader = EnhancedDataLoaderBuilder::new()
        .batch_size(1)
        .num_workers(3)
        .numa_config(NumaConfig::default())
        .build(dataset, sampler)
        .unwrap();

    // Test NUMA statistics
    if let Some(stats) = loader.get_numa_stats() {
        assert_eq!(stats.total_workers, 3);
        assert!(stats.numa_nodes_used > 0);
        assert!(stats.total_numa_nodes > 0);
        assert!(stats.affinity_success_rate >= 0.0 && stats.affinity_success_rate <= 1.0);

        // Sum of workers across all nodes should equal total workers
        let total_assigned: usize = stats.workers_per_node.values().sum();
        assert_eq!(total_assigned, stats.total_workers);
    }

    // Test NUMA topology access
    if let Some(topology) = loader.get_numa_topology() {
        assert!(!topology.nodes.is_empty());
        assert!(topology.total_cores > 0);
    }
}

#[test]
fn test_numa_preferred_nodes() {
    let config = NumaConfig {
        enabled: true,
        assignment_strategy: NumaAssignmentStrategy::RoundRobin,
        strict_affinity: false,
        preferred_nodes: vec![0], // Only use node 0
        balance_nodes: false,
    };

    let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

    let dataset = TensorDataset::new(features, labels);
    let sampler = SequentialSampler::new();

    let loader = EnhancedDataLoaderBuilder::new()
        .batch_size(1)
        .num_workers(2)
        .numa_config(config)
        .build(dataset, sampler);

    assert!(loader.is_ok());

    let loader = loader.unwrap();

    // If NUMA stats are available, check that workers are assigned to preferred nodes
    if let Some(stats) = loader.get_numa_stats() {
        // All workers should be on node 0 if preferred nodes setting was respected
        if let Some(&workers_on_node_0) = stats.workers_per_node.get(&0) {
            assert!(workers_on_node_0 > 0);
        }
    }
}

#[test]
fn test_numa_disabled_config() {
    let config = NumaConfig {
        enabled: false, // Explicitly disabled
        assignment_strategy: NumaAssignmentStrategy::RoundRobin,
        strict_affinity: false,
        preferred_nodes: Vec::new(),
        balance_nodes: true,
    };

    let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

    let dataset = TensorDataset::new(features, labels);
    let sampler = SequentialSampler::new();

    let loader = EnhancedDataLoaderBuilder::new()
        .batch_size(1)
        .num_workers(2)
        .numa_config(config)
        .build(dataset, sampler)
        .unwrap();

    // When NUMA is disabled, stats should be None
    assert!(loader.get_numa_stats().is_none());
    assert!(loader.get_numa_topology().is_none());
}

#[test]
fn test_dataloader_config_numa_helpers() {
    use tenflowers_dataset::DataLoaderConfig;

    // Test default config has no NUMA
    let default_config = DataLoaderConfig::default();
    assert!(!default_config.is_numa_enabled());

    // Test enabling NUMA
    let config_with_numa = DataLoaderConfig::default().with_numa_scheduling();
    assert!(config_with_numa.is_numa_enabled());

    // Test custom NUMA config
    let custom_numa_config = NumaConfig {
        enabled: true,
        assignment_strategy: NumaAssignmentStrategy::FillFirst,
        strict_affinity: true,
        preferred_nodes: vec![0, 1],
        balance_nodes: false,
    };

    let config_with_custom_numa =
        DataLoaderConfig::default().with_numa_config(custom_numa_config.clone());

    assert!(config_with_custom_numa.is_numa_enabled());

    if let Some(numa_config) = &config_with_custom_numa.numa_config {
        assert_eq!(
            numa_config.assignment_strategy,
            NumaAssignmentStrategy::FillFirst
        );
        assert!(numa_config.strict_affinity);
        assert_eq!(numa_config.preferred_nodes, vec![0, 1]);
        assert!(!numa_config.balance_nodes);
    }
}
