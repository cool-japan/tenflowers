#![allow(clippy::result_large_err)]

//! Example demonstrating distributed streaming data loading
//!
//! This example shows how to use the distributed streaming loader for large-scale
//! distributed training with deterministic partitioning, checkpointing, and
//! multi-worker coordination.

use std::sync::Arc;
use tenflowers_core::{Result, Tensor};
use tenflowers_dataset::{
    Dataset, PartitionStrategy, StreamCoordinator, StreamingConfig, StreamingShardIterator,
    StreamingShardLoader, TensorDataset,
};

fn main() -> Result<()> {
    println!("=== Distributed Streaming Loader Example ===\n");

    // Create a sample dataset (in practice, this would be a large dataset)
    let num_samples = 1000;
    let features_data: Vec<f32> = (0..num_samples * 10).map(|i| (i as f32) * 0.1).collect();
    let labels_data: Vec<f32> = (0..num_samples).map(|i| (i % 10) as f32).collect();

    let features = Tensor::from_vec(features_data, &[num_samples, 10])?;
    let labels = Tensor::from_vec(labels_data, &[num_samples])?;
    let dataset = TensorDataset::new(features, labels);

    println!("Created dataset with {} samples", dataset.len());
    println!();

    // Example 1: Basic Round-Robin Partitioning
    example_round_robin(dataset.clone())?;

    // Example 2: Contiguous Partitioning
    example_contiguous(dataset.clone())?;

    // Example 3: Hash-Based Partitioning
    example_hash_based(dataset.clone())?;

    // Example 4: Deterministic Shuffling
    example_deterministic_shuffle(dataset.clone())?;

    // Example 5: Checkpointing and Resumption
    example_checkpointing(dataset.clone())?;

    // Example 6: Multi-Worker Coordination
    example_multi_worker_coordination(dataset.clone())?;

    // Example 7: Prefetching
    example_prefetching(dataset.clone())?;

    println!("All examples completed successfully!");
    Ok(())
}

fn example_round_robin(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 1: Round-Robin Partitioning ---");

    // Simulate 4 workers
    let world_size = 4;

    for rank in 0..world_size {
        let config = StreamingConfig::new(world_size, rank)?
            .with_partition_strategy(PartitionStrategy::RoundRobin);

        let loader = StreamingShardLoader::new(dataset.clone(), config)?;

        println!("Worker {}: Assigned {} samples", rank, loader.len());

        // Load a few samples to demonstrate
        let mut count = 0;
        for _ in 0..3 {
            if loader.next()?.is_some() {
                count += 1;
            }
        }
        println!("  Loaded {} samples", count);
    }

    println!();
    Ok(())
}

fn example_contiguous(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 2: Contiguous Partitioning ---");

    let world_size = 4;

    for rank in 0..world_size {
        let config = StreamingConfig::new(world_size, rank)?
            .with_partition_strategy(PartitionStrategy::Contiguous);

        let loader = StreamingShardLoader::new(dataset.clone(), config)?;

        println!(
            "Worker {}: Assigned {} samples (contiguous block)",
            rank,
            loader.len()
        );
    }

    println!();
    Ok(())
}

fn example_hash_based(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 3: Hash-Based Partitioning ---");

    let world_size = 4;
    let hash_seed = 42;

    for rank in 0..world_size {
        let config = StreamingConfig::new(world_size, rank)?.with_partition_strategy(
            PartitionStrategy::HashBased {
                num_partitions: world_size,
                hash_seed,
            },
        );

        let loader = StreamingShardLoader::new(dataset.clone(), config)?;

        println!(
            "Worker {}: Assigned {} samples (hash-based)",
            rank,
            loader.len()
        );
    }

    println!();
    Ok(())
}

fn example_deterministic_shuffle(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 4: Deterministic Shuffling ---");

    let world_size = 2;
    let shuffle_seed = 12345;

    // Create two loaders with same seed
    let config1 = StreamingConfig::new(world_size, 0)?.with_shuffle_seed(shuffle_seed);

    let config2 = StreamingConfig::new(world_size, 0)?.with_shuffle_seed(shuffle_seed);

    let loader1 = StreamingShardLoader::new(dataset.clone(), config1)?;
    let loader2 = StreamingShardLoader::new(dataset.clone(), config2)?;

    println!(
        "Loader 1 and Loader 2 with same seed: {} samples each",
        loader1.len()
    );
    println!("Deterministic: Both loaders will process identical samples in identical order");

    // Create loader with different seed
    let config3 = StreamingConfig::new(world_size, 0)?.with_shuffle_seed(54321);

    let loader3 = StreamingShardLoader::new(dataset.clone(), config3)?;

    println!("Loader 3 with different seed: {} samples", loader3.len());
    println!("Different ordering than loaders 1 and 2");

    println!();
    Ok(())
}

fn example_checkpointing(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 5: Checkpointing and Resumption ---");

    let config = StreamingConfig::new(1, 0)?.with_checkpointing(100); // Checkpoint every 100 samples

    let loader = StreamingShardLoader::new(dataset.clone(), config)?;

    println!("Loading samples with checkpointing every 100 samples");

    // Load samples
    for i in 0..150 {
        if loader.next()?.is_none() {
            break;
        }

        if (i + 1) % 50 == 0 {
            println!("  Processed {} samples", i + 1);
        }
    }

    // Get checkpoint
    let checkpoint = loader.get_checkpoint()?;
    println!("Checkpoint created at position: {}", checkpoint.position);

    // Simulate failure and restoration
    println!("Simulating restoration from checkpoint...");
    loader.restore_from_checkpoint(checkpoint.clone())?;

    let restored = loader.get_checkpoint()?;
    println!("Restored to position: {}", restored.position);

    // Get statistics
    let stats = loader.get_stats()?;
    println!("Statistics:");
    println!("  Total samples loaded: {}", stats.samples_loaded);
    println!("  Checkpoints created: {}", stats.num_checkpoints);
    println!("  Average load time: {} μs", stats.avg_load_time_us);

    println!();
    Ok(())
}

fn example_multi_worker_coordination(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 6: Multi-Worker Coordination ---");

    let world_size = 4;

    // Create coordinator
    let coordinator_config = StreamingConfig::new(world_size, 0)?;
    let coordinator = Arc::new(StreamCoordinator::new(coordinator_config)?);

    println!("Created coordinator for {} workers", world_size);

    // Simulate multiple workers
    for rank in 0..world_size {
        let config = StreamingConfig::new(world_size, rank)?
            .with_partition_strategy(PartitionStrategy::RoundRobin);

        let loader = StreamingShardLoader::new(dataset.clone(), config)?
            .with_coordinator(coordinator.clone());

        // Register worker with coordinator
        coordinator.register_worker(rank, vec![])?;

        println!("Worker {} registered with coordinator", rank);

        // Simulate some work and update health
        let samples_processed = (rank as u64 + 1) * 100;
        let throughput = 50.0 + (rank as f64 * 10.0);

        coordinator.update_worker_health(rank, samples_processed, throughput)?;

        // Get worker health
        if let Some(health) = coordinator.get_worker_health(rank)? {
            println!(
                "  Health: {:?}, Throughput: {:.2} samples/sec",
                health.status, health.average_throughput
            );
        }
    }

    // Check if rebalancing is needed
    let needs_rebalance = coordinator.rebalance_if_needed()?;
    println!("Load balancing needed: {}", needs_rebalance);

    println!();
    Ok(())
}

fn example_prefetching(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example 7: Prefetching ---");

    let config = StreamingConfig::new(1, 0)?.with_prefetch_buffer_size(32);

    let loader = StreamingShardLoader::new(dataset.clone(), config)?;

    println!("Prefetching 10 samples into buffer");
    loader.prefetch(10)?;

    // Load samples - should hit the prefetch buffer
    let mut loaded = 0;
    for _ in 0..5 {
        if loader.next()?.is_some() {
            loaded += 1;
        }
    }

    let stats = loader.get_stats()?;
    println!("Loaded {} samples", loaded);
    println!("Prefetch buffer stats:");
    println!("  Hits: {}", stats.prefetch_hits);
    println!("  Misses: {}", stats.prefetch_misses);
    println!(
        "  Hit rate: {:.2}%",
        if stats.prefetch_hits + stats.prefetch_misses > 0 {
            100.0 * stats.prefetch_hits as f64
                / (stats.prefetch_hits + stats.prefetch_misses) as f64
        } else {
            0.0
        }
    );

    println!();
    Ok(())
}

#[allow(dead_code)]
fn example_iterator_usage(dataset: TensorDataset<f32>) -> Result<()> {
    println!("--- Example: Iterator Usage ---");

    let config = StreamingConfig::new(1, 0)?;
    let loader = Arc::new(StreamingShardLoader::new(dataset, config)?);

    let iter = StreamingShardIterator::new(loader);

    let mut count = 0;
    for result in iter.take(10) {
        let (_features, _labels) = result?;
        count += 1;
    }

    println!("Processed {} samples using iterator", count);
    println!();

    Ok(())
}
