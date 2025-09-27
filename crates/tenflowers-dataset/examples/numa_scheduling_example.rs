//! Example demonstrating NUMA-aware scheduling for multi-threaded data loading
//!
//! This example shows how to configure and use NUMA-aware scheduling to optimize
//! data loading performance on multi-socket systems by ensuring worker threads
//! are properly bound to specific NUMA nodes and CPU cores.
//!
//! Run with: `cargo run --example numa_scheduling_example --features numa`

use tenflowers_core::Tensor;
use tenflowers_dataset::{
    Dataset, EnhancedDataLoaderBuilder, NumaAssignmentStrategy, NumaConfig, SequentialSampler,
    TensorDataset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NUMA-aware scheduling example");

    // Create a sample dataset
    let features = Tensor::<f32>::from_vec(
        (0..1000).map(|i| i as f32).collect(),
        &[100, 10], // 100 samples, 10 features each
    )?;
    let labels = Tensor::<f32>::from_vec(
        (0..100).map(|i| (i % 2) as f32).collect(),
        &[100], // 100 binary labels
    )?;

    let dataset = TensorDataset::new(features, labels);
    let sampler = SequentialSampler::new();

    println!("Dataset created with {} samples", dataset.len());

    // Example 1: Basic NUMA-aware data loading
    println!("\n=== Example 1: Basic NUMA-aware data loading ===");

    let loader = EnhancedDataLoaderBuilder::new()
        .batch_size(8)
        .num_workers(4)
        .numa_config(NumaConfig {
            enabled: true,
            assignment_strategy: NumaAssignmentStrategy::RoundRobin,
            strict_affinity: false,
            preferred_nodes: Vec::new(),
            balance_nodes: true,
        })
        .build(dataset.clone(), sampler.clone())?;

    // Get NUMA topology information
    if let Some(topology) = loader.get_numa_topology() {
        println!("NUMA topology detected:");
        println!("  Total CPU cores: {}", topology.total_cores);
        println!("  NUMA nodes: {}", topology.nodes.len());
        println!("  NUMA available: {}", topology.numa_available);

        for node in &topology.nodes {
            println!(
                "  Node {}: {} cores ({:?})",
                node.id,
                node.cpu_cores.len(),
                &node.cpu_cores[..node.cpu_cores.len().min(4)] // Show first 4 cores
            );
        }
    } else {
        println!("NUMA scheduling not enabled or topology not available");
    }

    // Get NUMA assignment statistics
    if let Some(stats) = loader.get_numa_stats() {
        println!("\nNUMA assignment statistics:");
        println!("  Total workers: {}", stats.total_workers);
        println!(
            "  NUMA nodes used: {}/{}",
            stats.numa_nodes_used, stats.total_numa_nodes
        );
        println!(
            "  CPU affinity success rate: {:.1}%",
            stats.affinity_success_rate * 100.0
        );

        for (node_id, worker_count) in &stats.workers_per_node {
            println!("  Node {}: {} workers", node_id, worker_count);
        }
    }

    // Process a few batches to demonstrate performance
    println!("\nProcessing batches...");
    let start = std::time::Instant::now();
    let mut batch_count = 0;

    for batch in loader.take(5) {
        match batch {
            Ok(_) => {
                batch_count += 1;
                print!(".");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(e) => {
                eprintln!("Error processing batch: {}", e);
                break;
            }
        }
    }

    let duration = start.elapsed();
    println!("\nProcessed {} batches in {:?}", batch_count, duration);

    // Example 2: Custom NUMA configuration strategies
    println!("\n=== Example 2: Different NUMA assignment strategies ===");

    let strategies = vec![
        ("Round Robin", NumaAssignmentStrategy::RoundRobin),
        ("Fill First", NumaAssignmentStrategy::FillFirst),
        ("Interleave", NumaAssignmentStrategy::Interleave),
        ("Load Balanced", NumaAssignmentStrategy::LoadBalanced),
    ];

    for (name, strategy) in strategies {
        println!("\n--- {} Strategy ---", name);

        let numa_config = NumaConfig {
            enabled: true,
            assignment_strategy: strategy,
            strict_affinity: false,
            preferred_nodes: Vec::new(),
            balance_nodes: true,
        };

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(4)
            .num_workers(6)
            .numa_config(numa_config)
            .build(dataset.clone(), sampler.clone())?;

        if let Some(stats) = loader.get_numa_stats() {
            println!("Workers per NUMA node:");
            for (node_id, worker_count) in &stats.workers_per_node {
                println!("  Node {}: {} workers", node_id, worker_count);
            }
        }
    }

    // Example 3: Performance comparison with and without NUMA
    println!("\n=== Example 3: Performance comparison ===");

    let num_batches = 20;

    // Without NUMA
    println!("Testing without NUMA scheduling...");
    let loader_no_numa = EnhancedDataLoaderBuilder::new()
        .batch_size(8)
        .num_workers(4)
        .build(dataset.clone(), sampler.clone())?;

    let start = std::time::Instant::now();
    let mut count = 0;
    for batch in loader_no_numa.take(num_batches) {
        if batch.is_ok() {
            count += 1;
        }
    }
    let duration_no_numa = start.elapsed();
    println!("Without NUMA: {} batches in {:?}", count, duration_no_numa);

    // With NUMA
    println!("Testing with NUMA scheduling...");
    let loader_numa = EnhancedDataLoaderBuilder::new()
        .batch_size(8)
        .num_workers(4)
        .numa_config(NumaConfig::default())
        .build(dataset.clone(), sampler.clone())?;

    let start = std::time::Instant::now();
    let mut count = 0;
    for batch in loader_numa.take(num_batches) {
        if batch.is_ok() {
            count += 1;
        }
    }
    let duration_numa = start.elapsed();
    println!("With NUMA: {} batches in {:?}", count, duration_numa);

    // Calculate performance difference
    let speedup = duration_no_numa.as_secs_f64() / duration_numa.as_secs_f64();
    if speedup > 1.0 {
        println!("NUMA scheduling provided {:.2}x speedup", speedup);
    } else if speedup < 1.0 {
        println!(
            "NUMA scheduling was {:.2}x slower (overhead on this system)",
            1.0 / speedup
        );
    } else {
        println!("No significant performance difference detected");
    }

    // Example 4: Preferred NUMA nodes
    println!("\n=== Example 4: Preferred NUMA nodes ===");

    if let Some(topology) = {
        let temp_loader = EnhancedDataLoaderBuilder::new()
            .batch_size(4)
            .num_workers(2)
            .numa_config(NumaConfig::default())
            .build(dataset.clone(), sampler.clone())?;
        temp_loader.get_numa_topology().cloned()
    } {
        if topology.nodes.len() > 1 {
            // Only use the first NUMA node
            let preferred_nodes = vec![topology.nodes[0].id];
            println!("Restricting workers to NUMA node {}", topology.nodes[0].id);

            let numa_config = NumaConfig {
                enabled: true,
                assignment_strategy: NumaAssignmentStrategy::RoundRobin,
                strict_affinity: true,
                preferred_nodes,
                balance_nodes: false,
            };

            let loader = EnhancedDataLoaderBuilder::new()
                .batch_size(4)
                .num_workers(4)
                .numa_config(numa_config)
                .build(dataset.clone(), sampler.clone())?;

            if let Some(stats) = loader.get_numa_stats() {
                println!("Worker distribution with preferred nodes:");
                for (node_id, worker_count) in &stats.workers_per_node {
                    println!("  Node {}: {} workers", node_id, worker_count);
                }
            }
        } else {
            println!("Only one NUMA node available, skipping preferred nodes example");
        }
    }

    println!("\n=== Summary ===");
    println!("NUMA-aware scheduling can provide performance benefits on multi-socket systems by:");
    println!("1. Reducing memory access latency through NUMA-local allocations");
    println!("2. Improving cache locality by binding threads to specific cores");
    println!("3. Reducing inter-socket memory bandwidth contention");
    println!("4. Enabling better load balancing across NUMA domains");
    println!("\nPerformance gains depend on:");
    println!("- System NUMA topology (more nodes = more potential benefit)");
    println!("- Memory access patterns of the dataset");
    println!("- CPU cache behavior and memory bandwidth utilization");
    println!("- Whether the system is already under memory pressure");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_dataset::{NumaAssignmentStrategy, NumaConfig, NumaScheduler};

    #[test]
    fn test_numa_configuration() {
        let config = NumaConfig {
            enabled: true,
            assignment_strategy: NumaAssignmentStrategy::RoundRobin,
            strict_affinity: false,
            preferred_nodes: vec![0, 1],
            balance_nodes: true,
        };

        // Test that we can create a scheduler with the config
        let scheduler = NumaScheduler::new(config);
        assert!(scheduler.is_ok());
    }

    #[test]
    fn test_numa_dataloader_integration() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let sampler = SequentialSampler::new();

        let loader = EnhancedDataLoaderBuilder::new()
            .batch_size(1)
            .num_workers(2)
            .numa_config(NumaConfig::default())
            .build(dataset, sampler);

        assert!(loader.is_ok());

        let loader = loader.unwrap();

        // Test that we can get NUMA stats (even if NUMA is not available on test system)
        let _stats = loader.get_numa_stats();
        let _topology = loader.get_numa_topology();
    }
}
