# Distributed Streaming Loaders

Advanced distributed streaming capabilities for large-scale data processing with deterministic partitioning, multi-worker coordination, and fault tolerance.

## Overview

The distributed streaming module provides sophisticated data loading for distributed training scenarios:

- **Deterministic Partitioning**: Reproducible data sharding across workers
- **Multi-Worker Coordination**: Centralized coordination for distributed systems
- **Advanced Partitioning Strategies**: Multiple algorithms for balanced load distribution
- **Checkpointing & Resumption**: Fault-tolerant stream state management
- **Performance Optimization**: Prefetching, caching, and adaptive load balancing

## Quick Start

### Basic Round-Robin Partitioning

```rust
use tenflowers_dataset::{
    StreamingConfig, StreamingShardLoader, PartitionStrategy, TensorDataset
};
use tenflowers_core::Tensor;

// Create dataset
let features = Tensor::<f32>::from_vec(vec![1.0; 1000], &[1000, 1])?;
let labels = Tensor::<f32>::from_vec(vec![1.0; 1000], &[1000])?;
let dataset = TensorDataset::new(features, labels);

// Configure for 4 workers, rank 0
let config = StreamingConfig::new(4, 0)?;

// Create streaming loader
let loader = StreamingShardLoader::new(dataset, config)?;

// Stream data
while let Some(sample) = loader.next()? {
    let (features, labels) = sample;
    // Process sample...
}
```

## Partition Strategies

### Round-Robin

Distributes samples evenly across workers in a round-robin fashion.

```rust
let config = StreamingConfig::new(4, 0)?
    .with_partition_strategy(PartitionStrategy::RoundRobin);
```

**Use cases:**
- Uniform data distribution
- Simple, balanced workloads
- No data dependencies

### Contiguous

Divides dataset into contiguous blocks, one per worker.

```rust
let config = StreamingConfig::new(4, 1)?
    .with_partition_strategy(PartitionStrategy::Contiguous);
```

**Use cases:**
- Sequential access patterns
- Sorted data
- Temporal locality

### Hash-Based

Uses deterministic hashing for consistent partitioning across runs.

```rust
let config = StreamingConfig::new(4, 0)?
    .with_partition_strategy(PartitionStrategy::HashBased {
        num_partitions: 16,
        hash_seed: 42,
    });
```

**Use cases:**
- Key-based data
- Reproducible partitioning
- Non-sequential access

### Range-Based

Assigns specific ranges to each worker.

```rust
let ranges = vec![
    (0, 250),      // Worker 0: samples 0-249
    (250, 500),    // Worker 1: samples 250-499
    (500, 750),    // Worker 2: samples 750-749
    (750, 1000),   // Worker 3: samples 750-999
];

let config = StreamingConfig::new(4, 0)?
    .with_partition_strategy(PartitionStrategy::RangeBased { ranges });
```

**Use cases:**
- Pre-computed partitions
- Custom data distribution
- Load-balanced workloads

### Stratified

Maintains class distribution across workers (requires label information).

```rust
let config = StreamingConfig::new(4, 0)?
    .with_partition_strategy(PartitionStrategy::Stratified {
        num_classes: 10,
    });
```

**Use cases:**
- Imbalanced datasets
- Classification tasks
- Fair class distribution

### Adaptive

Dynamically adjusts partitioning based on worker performance.

```rust
let base_strategy = Box::new(PartitionStrategy::RoundRobin);

let config = StreamingConfig::new(4, 0)?
    .with_partition_strategy(PartitionStrategy::Adaptive {
        base_strategy,
        rebalance_threshold: 0.2, // 20% variance triggers rebalancing
    })
    .with_dynamic_balancing(true);
```

**Use cases:**
- Heterogeneous worker performance
- Dynamic workload changes
- Automatic optimization

## Deterministic Shuffling

Ensure reproducibility across runs with seeded shuffling.

```rust
let config = StreamingConfig::new(4, 0)?
    .with_shuffle_seed(42); // Same seed = same order

let loader = StreamingShardLoader::new(dataset, config)?;
```

All workers with the same seed will process samples in the same deterministic order.

## Checkpointing

Save and restore stream state for fault tolerance.

```rust
// Enable automatic checkpointing
let config = StreamingConfig::new(4, 0)?
    .with_checkpointing(1000); // Checkpoint every 1000 samples

let loader = StreamingShardLoader::new(dataset, config)?;

// Load samples...
for _ in 0..5000 {
    let _ = loader.next()?;
}

// Get checkpoint state
let checkpoint = loader.get_checkpoint()?;
println!("Checkpoint at position: {}", checkpoint.position);

// Simulate failure and restoration
loader.restore_from_checkpoint(checkpoint)?;
```

### Checkpoint State

```rust
pub struct CheckpointState {
    pub epoch: usize,
    pub position: usize,
    pub shuffle_seed: Option<u64>,
    pub rank: usize,
    pub timestamp: u64,
    pub processed_indices: HashSet<usize>,
}
```

## Multi-Worker Coordination

Coordinate multiple workers for distributed training.

```rust
use std::sync::Arc;
use tenflowers_dataset::{StreamCoordinator, StreamingConfig};

// Create coordinator
let coordinator_config = StreamingConfig::new(4, 0)?;
let coordinator = Arc::new(StreamCoordinator::new(coordinator_config)?);

// Create workers
for rank in 0..4 {
    let worker_config = StreamingConfig::new(4, rank)?;
    let loader = StreamingShardLoader::new(dataset.clone(), worker_config)?
        .with_coordinator(coordinator.clone());

    // Register worker
    coordinator.register_worker(rank, vec![])?;

    // Update worker health
    coordinator.update_worker_health(rank, 1000, 50.0)?;
}

// Check if rebalancing is needed
let needs_rebalance = coordinator.rebalance_if_needed()?;
```

### Worker Health Monitoring

```rust
pub struct WorkerHealth {
    pub rank: usize,
    pub status: WorkerStatus,
    pub last_heartbeat: u64,
    pub samples_processed: u64,
    pub average_throughput: f64,
}

pub enum WorkerStatus {
    Active,    // Worker operating normally
    Idle,      // Worker waiting for work
    Slow,      // Worker below performance threshold
    Failed,    // Worker has failed
    Unknown,   // Worker status unknown
}
```

## Prefetching

Improve performance with sample prefetching.

```rust
let config = StreamingConfig::new(4, 0)?
    .with_prefetch_buffer_size(128); // Buffer up to 128 samples

let loader = StreamingShardLoader::new(dataset, config)?;

// Prefetch samples into buffer
loader.prefetch(32)?;

// Subsequent next() calls hit the buffer
while let Some(sample) = loader.next()? {
    // Process sample...
}

// Check prefetch statistics
let stats = loader.get_stats()?;
println!("Prefetch hit rate: {:.2}%",
    100.0 * stats.prefetch_hits as f64 /
    (stats.prefetch_hits + stats.prefetch_misses) as f64
);
```

## Performance Statistics

Monitor streaming performance with built-in metrics.

```rust
let stats = loader.get_stats()?;

println!("Samples loaded: {}", stats.samples_loaded);
println!("Local samples: {}", stats.local_samples);
println!("Remote samples: {}", stats.remote_samples);
println!("Prefetch hits: {}", stats.prefetch_hits);
println!("Prefetch misses: {}", stats.prefetch_misses);
println!("Avg load time: {} μs", stats.avg_load_time_us);
println!("Checkpoints: {}", stats.num_checkpoints);
println!("Worker utilization: {:.2}%", stats.worker_utilization * 100.0);
```

## Fault Tolerance

Enable fault tolerance with data replication.

```rust
let config = StreamingConfig::new(4, 0)?
    .with_fault_tolerance(2); // Replication factor of 2

let loader = StreamingShardLoader::new(dataset, config)?;
```

If a worker fails, another worker can take over its partition using the replication.

## Iterator Interface

Use the streaming loader as an iterator.

```rust
use std::sync::Arc;
use tenflowers_dataset::StreamingShardIterator;

let loader = Arc::new(StreamingShardLoader::new(dataset, config)?);
let iter = StreamingShardIterator::new(loader);

for result in iter {
    let (features, labels) = result?;
    // Process sample...
}
```

## Best Practices

### Choosing a Partition Strategy

1. **Round-Robin**: Default choice for most cases
2. **Contiguous**: Use for sequential/sorted data
3. **Hash-Based**: Use for reproducibility and key-based data
4. **Stratified**: Use for imbalanced classification datasets
5. **Adaptive**: Use for heterogeneous worker performance

### Shuffle Seeds

- Use deterministic seeds for reproducibility
- Use different seeds for training/validation splits
- Document seeds in experiment configs

### Checkpointing

- Set checkpoint interval based on dataset size
- Smaller intervals = more overhead, better recovery
- Larger intervals = less overhead, coarser recovery
- Typical range: 100-10000 samples

### Prefetch Buffer Size

- Larger buffers improve throughput but use more memory
- Typical range: 32-512 samples
- Adjust based on sample size and available memory

### Worker Coordination

- Use coordinator for >4 workers
- Monitor worker health for imbalanced workloads
- Enable dynamic balancing for heterogeneous systems

## Advanced Examples

### Multi-Epoch Training

```rust
let config = StreamingConfig::new(4, 0)?
    .with_shuffle_seed(42)
    .with_checkpointing(1000);

let loader = StreamingShardLoader::new(dataset, config)?;

for epoch in 0..10 {
    println!("Epoch {}", epoch);

    // Reset stream for new epoch
    loader.reset()?;

    while let Some(sample) = loader.next()? {
        // Training step...
    }

    // Save checkpoint at end of epoch
    let checkpoint = loader.get_checkpoint()?;
    // Save to disk...
}
```

### Distributed Training with Coordination

```rust
use std::sync::Arc;

// Setup coordinator on master node
let coordinator_config = StreamingConfig::new(num_workers, 0)?;
let coordinator = Arc::new(StreamCoordinator::new(coordinator_config)?);

// Each worker
let worker_config = StreamingConfig::new(num_workers, worker_rank)?
    .with_partition_strategy(PartitionStrategy::RoundRobin)
    .with_prefetch_buffer_size(64)
    .with_checkpointing(500);

let loader = StreamingShardLoader::new(dataset, worker_config)?
    .with_coordinator(coordinator.clone());

// Register worker
coordinator.register_worker(worker_rank, vec![])?;

// Training loop
loop {
    // Load batch
    let mut batch = Vec::new();
    for _ in 0..batch_size {
        if let Some(sample) = loader.next()? {
            batch.push(sample);
        } else {
            break;
        }
    }

    if batch.is_empty() {
        break;
    }

    // Training step...

    // Update worker health
    let stats = loader.get_stats()?;
    let throughput = stats.samples_loaded as f64 / elapsed_time;
    coordinator.update_worker_health(
        worker_rank,
        stats.samples_loaded,
        throughput
    )?;
}
```

## Performance Tuning

### Memory Usage

- Reduce `prefetch_buffer_size` to save memory
- Use `Contiguous` strategy for better cache locality
- Enable checkpointing only when needed

### Throughput

- Increase `prefetch_buffer_size` for I/O-bound workloads
- Use `parallel` feature for multi-threaded loading
- Optimize partition strategy for your data pattern

### Latency

- Use smaller prefetch buffers for lower latency
- Disable checkpointing for training-only scenarios
- Use `RoundRobin` for simplest, fastest partitioning

## Troubleshooting

### Imbalanced Worker Loads

**Problem**: Some workers process much more data than others.

**Solutions**:
- Use `Adaptive` partition strategy with dynamic balancing
- Check dataset distribution with stratified partitioning
- Monitor worker health and adjust manually

### Out of Memory

**Problem**: Worker runs out of memory during streaming.

**Solutions**:
- Reduce `prefetch_buffer_size`
- Disable caching if enabled
- Use more workers to distribute load

### Slow Streaming

**Problem**: Data loading is slower than expected.

**Solutions**:
- Increase prefetch buffer size
- Use faster storage (SSD over HDD)
- Enable parallel loading if available
- Profile with `get_stats()` to identify bottlenecks

### Non-Deterministic Results

**Problem**: Results differ across runs.

**Solutions**:
- Set explicit `shuffle_seed`
- Use deterministic partition strategies (Hash-Based)
- Verify checkpoint restoration logic

## API Reference

### StreamingConfig

Configuration for distributed streaming.

**Methods**:
- `new(world_size, rank)` - Create new configuration
- `with_partition_strategy(strategy)` - Set partition strategy
- `with_prefetch_buffer_size(size)` - Set prefetch buffer size
- `with_shuffle_seed(seed)` - Enable deterministic shuffling
- `with_checkpointing(interval)` - Enable automatic checkpointing
- `with_fault_tolerance(replication)` - Enable fault tolerance
- `with_dynamic_balancing(enabled)` - Enable dynamic load balancing
- `validate()` - Validate configuration

### StreamingShardLoader

Main loader for distributed streaming.

**Methods**:
- `new(dataset, config)` - Create new loader
- `with_coordinator(coordinator)` - Attach coordinator
- `next()` - Get next sample
- `prefetch(count)` - Prefetch samples into buffer
- `get_checkpoint()` - Get current checkpoint state
- `restore_from_checkpoint(checkpoint)` - Restore from checkpoint
- `get_stats()` - Get performance statistics
- `reset()` - Reset stream to beginning
- `len()` - Get total number of assigned samples
- `is_empty()` - Check if stream is empty

### StreamCoordinator

Multi-worker coordination layer.

**Methods**:
- `new(config)` - Create new coordinator
- `register_worker(rank, indices)` - Register worker
- `update_worker_health(rank, samples, throughput)` - Update health
- `get_worker_health(rank)` - Get worker health status
- `register_checkpoint(rank, checkpoint)` - Register checkpoint
- `get_checkpoint(rank)` - Get worker checkpoint
- `rebalance_if_needed()` - Check if rebalancing needed

## See Also

- [distributed_sharding.rs](../src/distributed_sharding.rs) - Static dataset sharding
- [distributed_loading.rs](../src/distributed_loading.rs) - Multi-node distributed loading
- [dataloader module](../src/dataloader/) - Standard data loading
