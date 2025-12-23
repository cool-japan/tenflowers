//! Deterministic shard loader for distributed training
//!
//! This module provides deterministic partitioning of datasets across multiple workers
//! for distributed training, ensuring reproducibility and balanced distribution.

use crate::{error_taxonomy::helpers as error_helpers, Dataset};
use std::collections::HashMap;
use std::sync::Arc;
use tenflowers_core::{Result, Tensor};

/// Configuration for distributed sharding
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Total number of workers (world size)
    pub world_size: usize,
    /// Current worker rank (0-indexed)
    pub rank: usize,
    /// Strategy for distributing samples
    pub strategy: ShardStrategy,
    /// Seed for deterministic shuffling
    pub seed: Option<u64>,
    /// Whether to drop last incomplete batch
    pub drop_last: bool,
    /// Number of replicas per shard (for fault tolerance)
    pub num_replicas: usize,
}

/// Strategy for distributing samples across workers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Round-robin distribution (sample i goes to worker i % world_size)
    RoundRobin,
    /// Contiguous blocks (each worker gets a contiguous range)
    Contiguous,
    /// Deterministic shuffle then round-robin
    ShuffledRoundRobin,
    /// Stratified sampling (requires label information)
    Stratified,
}

impl ShardConfig {
    /// Create a new shard configuration
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(error_helpers::invalid_configuration(
                "ShardConfig::new",
                "world_size",
                "world_size must be > 0",
            ));
        }

        if rank >= world_size {
            return Err(error_helpers::invalid_configuration(
                "ShardConfig::new",
                "rank",
                format!("rank {} must be < world_size {}", rank, world_size),
            ));
        }

        Ok(Self {
            world_size,
            rank,
            strategy: ShardStrategy::RoundRobin,
            seed: None,
            drop_last: false,
            num_replicas: 1,
        })
    }

    /// Set the sharding strategy
    pub fn with_strategy(mut self, strategy: ShardStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the seed for deterministic shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set whether to drop the last incomplete batch
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set the number of replicas for fault tolerance
    pub fn with_num_replicas(mut self, num_replicas: usize) -> Self {
        self.num_replicas = num_replicas;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            return Err(error_helpers::invalid_configuration(
                "ShardConfig::validate",
                "world_size",
                "world_size must be > 0",
            ));
        }

        if self.rank >= self.world_size {
            return Err(error_helpers::invalid_configuration(
                "ShardConfig::validate",
                "rank",
                format!(
                    "rank {} must be < world_size {}",
                    self.rank, self.world_size
                ),
            ));
        }

        if self.num_replicas == 0 {
            return Err(error_helpers::invalid_configuration(
                "ShardConfig::validate",
                "num_replicas",
                "num_replicas must be > 0",
            ));
        }

        Ok(())
    }
}

/// Trait for datasets that can be sharded for distributed training
pub trait ShardableDataset<T>: Dataset<T> {
    /// Get indices for this worker's shard
    fn get_shard_indices(&self, config: &ShardConfig) -> Result<Vec<usize>>;

    /// Get the total number of shards
    fn num_shards(&self, config: &ShardConfig) -> usize {
        config.world_size
    }

    /// Get shard size for this worker
    fn shard_size(&self, config: &ShardConfig) -> usize {
        let indices = self.get_shard_indices(config).unwrap_or_default();
        indices.len()
    }
}

/// Wrapper that makes any dataset shardable
pub struct ShardedDataset<T, D: Dataset<T>> {
    dataset: Arc<D>,
    config: ShardConfig,
    indices: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, D: Dataset<T>> ShardedDataset<T, D> {
    /// Create a new sharded dataset
    pub fn new(dataset: D, config: ShardConfig) -> Result<Self> {
        config.validate()?;

        let dataset = Arc::new(dataset);
        let indices = Self::compute_indices(&dataset, &config)?;

        Ok(Self {
            dataset,
            config,
            indices,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a sharded dataset with stratified sampling
    /// This requires a label extractor function to determine the class of each sample
    pub fn new_stratified<F>(dataset: D, config: ShardConfig, label_extractor: F) -> Result<Self>
    where
        F: Fn(&Tensor<T>) -> Result<usize>,
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        config.validate()?;

        let dataset = Arc::new(dataset);
        let indices = Self::compute_stratified_indices(&dataset, &config, label_extractor)?;

        Ok(Self {
            dataset,
            config,
            indices,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Compute the indices for this shard
    fn compute_indices(dataset: &D, config: &ShardConfig) -> Result<Vec<usize>> {
        let total_size = dataset.len();

        if total_size == 0 {
            return Ok(Vec::new());
        }

        let mut all_indices: Vec<usize> = (0..total_size).collect();

        // Apply strategy-specific ordering
        match &config.strategy {
            ShardStrategy::RoundRobin => {
                // No reordering needed, will filter by rank
            }
            ShardStrategy::Contiguous => {
                // Already in contiguous order
            }
            ShardStrategy::ShuffledRoundRobin => {
                // Deterministically shuffle
                if let Some(seed) = config.seed {
                    Self::deterministic_shuffle(&mut all_indices, seed);
                }
            }
            ShardStrategy::Stratified => {
                // Stratified sampling requires label information
                // Use `new_stratified` constructor instead of `new` for proper stratified sharding
                // Falling back to round-robin for backwards compatibility
            }
        }

        // Select indices for this rank
        let shard_indices = match &config.strategy {
            ShardStrategy::RoundRobin | ShardStrategy::ShuffledRoundRobin => {
                // Every world_size'th element starting from rank
                all_indices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % config.world_size == config.rank)
                    .map(|(_, &idx)| idx)
                    .collect()
            }
            ShardStrategy::Contiguous => {
                // Divide into contiguous blocks
                let samples_per_worker = total_size / config.world_size;
                let extra_samples = total_size % config.world_size;

                let start = if config.rank < extra_samples {
                    config.rank * (samples_per_worker + 1)
                } else {
                    config.rank * samples_per_worker + extra_samples
                };

                let count = if config.rank < extra_samples {
                    samples_per_worker + 1
                } else {
                    samples_per_worker
                };

                all_indices[start..start + count].to_vec()
            }
            ShardStrategy::Stratified => {
                // Fallback to round-robin for now
                all_indices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % config.world_size == config.rank)
                    .map(|(_, &idx)| idx)
                    .collect()
            }
        };

        Ok(shard_indices)
    }

    /// Deterministically shuffle indices using Fisher-Yates with seeded RNG
    fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
        let mut rng_state = seed;

        for i in (1..indices.len()).rev() {
            // Simple LCG for deterministic random numbers
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    /// Compute stratified indices ensuring balanced class distribution across workers
    fn compute_stratified_indices<F>(
        dataset: &D,
        config: &ShardConfig,
        label_extractor: F,
    ) -> Result<Vec<usize>>
    where
        F: Fn(&Tensor<T>) -> Result<usize>,
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        let total_size = dataset.len();

        if total_size == 0 {
            return Ok(Vec::new());
        }

        // Group indices by class
        let mut class_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..total_size {
            let (_, label_tensor) = dataset.get(i)?;
            let class = label_extractor(&label_tensor)?;
            class_to_indices.entry(class).or_default().push(i);
        }

        // For each class, shuffle deterministically and distribute across workers
        let mut worker_indices: Vec<Vec<usize>> = vec![Vec::new(); config.world_size];

        // Sort classes to ensure deterministic ordering
        let mut classes: Vec<_> = class_to_indices.keys().cloned().collect();
        classes.sort_unstable();

        for class in classes {
            let mut indices = class_to_indices.remove(&class).unwrap();

            // Deterministically shuffle within class
            if let Some(seed) = config.seed {
                // Use class as additional seed component for reproducibility
                Self::deterministic_shuffle(&mut indices, seed.wrapping_add(class as u64));
            }

            // Distribute class indices round-robin across workers
            for (idx_pos, &global_idx) in indices.iter().enumerate() {
                let worker_id = idx_pos % config.world_size;
                worker_indices[worker_id].push(global_idx);
            }
        }

        // Get indices for this worker's rank
        let mut shard_indices = worker_indices[config.rank].clone();

        // Optionally shuffle the final shard indices
        if let Some(seed) = config.seed {
            Self::deterministic_shuffle(&mut shard_indices, seed.wrapping_add(config.rank as u64));
        }

        Ok(shard_indices)
    }

    /// Get the underlying dataset
    pub fn inner(&self) -> &D {
        &self.dataset
    }

    /// Get the shard configuration
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }

    /// Get the shard indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get shard statistics
    pub fn shard_stats(&self) -> ShardStatistics {
        let total_size = self.dataset.len();
        let shard_size = self.indices.len();

        let min_shard_size = total_size / self.config.world_size;
        let max_shard_size = (total_size + self.config.world_size - 1) / self.config.world_size;

        ShardStatistics {
            total_samples: total_size,
            shard_size,
            min_shard_size,
            max_shard_size,
            world_size: self.config.world_size,
            rank: self.config.rank,
            imbalance_ratio: if min_shard_size > 0 {
                max_shard_size as f64 / min_shard_size as f64
            } else {
                0.0
            },
        }
    }
}

impl<T, D: Dataset<T>> Dataset<T> for ShardedDataset<T, D> {
    fn get(
        &self,
        index: usize,
    ) -> Result<(tenflowers_core::Tensor<T>, tenflowers_core::Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(error_helpers::index_out_of_bounds(
                "ShardedDataset::get",
                index,
                self.indices.len(),
            ));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Statistics about shard distribution
#[derive(Debug, Clone)]
pub struct ShardStatistics {
    /// Total number of samples across all shards
    pub total_samples: usize,
    /// Number of samples in this shard
    pub shard_size: usize,
    /// Minimum shard size across all workers
    pub min_shard_size: usize,
    /// Maximum shard size across all workers
    pub max_shard_size: usize,
    /// Total number of workers
    pub world_size: usize,
    /// Current worker rank
    pub rank: usize,
    /// Ratio of max to min shard size (measures imbalance)
    pub imbalance_ratio: f64,
}

impl ShardStatistics {
    /// Check if shards are balanced (imbalance ratio close to 1.0)
    pub fn is_balanced(&self) -> bool {
        self.imbalance_ratio <= 1.1 // Allow 10% imbalance
    }

    /// Generate a human-readable report
    pub fn report(&self) -> String {
        format!(
            "Shard Statistics:\n\
             - Total samples: {}\n\
             - World size: {} workers\n\
             - Rank: {}\n\
             - This shard size: {}\n\
             - Min shard size: {}\n\
             - Max shard size: {}\n\
             - Imbalance ratio: {:.2}\n\
             - Balanced: {}",
            self.total_samples,
            self.world_size,
            self.rank,
            self.shard_size,
            self.min_shard_size,
            self.max_shard_size,
            self.imbalance_ratio,
            if self.is_balanced() { "Yes" } else { "No" }
        )
    }
}

/// Extension trait to add sharding capabilities to any dataset
pub trait DatasetShardingExt<T>: Dataset<T> + Sized {
    /// Shard this dataset for distributed training
    fn shard(self, config: ShardConfig) -> Result<ShardedDataset<T, Self>> {
        ShardedDataset::new(self, config)
    }

    /// Create a round-robin sharded dataset
    fn shard_round_robin(self, world_size: usize, rank: usize) -> Result<ShardedDataset<T, Self>> {
        let config = ShardConfig::new(world_size, rank)?;
        ShardedDataset::new(self, config)
    }

    /// Create a contiguous sharded dataset
    fn shard_contiguous(self, world_size: usize, rank: usize) -> Result<ShardedDataset<T, Self>> {
        let config = ShardConfig::new(world_size, rank)?.with_strategy(ShardStrategy::Contiguous);
        ShardedDataset::new(self, config)
    }

    /// Create a shuffled sharded dataset with a seed
    fn shard_shuffled(
        self,
        world_size: usize,
        rank: usize,
        seed: u64,
    ) -> Result<ShardedDataset<T, Self>> {
        let config = ShardConfig::new(world_size, rank)?
            .with_strategy(ShardStrategy::ShuffledRoundRobin)
            .with_seed(seed);
        ShardedDataset::new(self, config)
    }
}

// Blanket implementation for all datasets
impl<T, D: Dataset<T>> DatasetShardingExt<T> for D {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_shard_config_creation() {
        let config = ShardConfig::new(4, 0).unwrap();
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
        assert_eq!(config.strategy, ShardStrategy::RoundRobin);
    }

    #[test]
    fn test_shard_config_validation() {
        assert!(ShardConfig::new(0, 0).is_err());
        assert!(ShardConfig::new(4, 4).is_err());
        assert!(ShardConfig::new(4, 5).is_err());
        assert!(ShardConfig::new(4, 3).is_ok());
    }

    #[test]
    fn test_round_robin_sharding() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10, 1],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Shard into 3 workers
        let config = ShardConfig::new(3, 0).unwrap();
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        // Rank 0 should get indices [0, 3, 6, 9]
        assert_eq!(sharded.len(), 4);
        assert_eq!(sharded.indices(), &[0, 3, 6, 9]);
    }

    #[test]
    fn test_contiguous_sharding() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10, 1],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Shard into 3 workers with contiguous strategy
        let config = ShardConfig::new(3, 1)
            .unwrap()
            .with_strategy(ShardStrategy::Contiguous);
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        // Rank 1 should get a contiguous block
        // 10 samples / 3 workers = 3 base + 1 extra for first worker
        // Rank 0: [0,1,2,3], Rank 1: [4,5,6], Rank 2: [7,8,9]
        assert_eq!(sharded.len(), 3);
        assert_eq!(sharded.indices(), &[4, 5, 6]);
    }

    #[test]
    fn test_shuffled_sharding_deterministic() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100]).unwrap();
        let dataset1 = TensorDataset::new(features.clone(), labels.clone());
        let dataset2 = TensorDataset::new(features, labels);

        let config1 = ShardConfig::new(4, 0)
            .unwrap()
            .with_strategy(ShardStrategy::ShuffledRoundRobin)
            .with_seed(42);
        let config2 = ShardConfig::new(4, 0)
            .unwrap()
            .with_strategy(ShardStrategy::ShuffledRoundRobin)
            .with_seed(42);

        let sharded1 = ShardedDataset::new(dataset1, config1).unwrap();
        let sharded2 = ShardedDataset::new(dataset2, config2).unwrap();

        // Same seed should produce same indices
        assert_eq!(sharded1.indices(), sharded2.indices());
    }

    #[test]
    fn test_shard_statistics() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ShardConfig::new(3, 0).unwrap();
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        let stats = sharded.shard_stats();
        assert_eq!(stats.total_samples, 100);
        assert_eq!(stats.world_size, 3);
        assert_eq!(stats.rank, 0);
        assert!(stats.imbalance_ratio >= 1.0);
    }

    #[test]
    fn test_extension_trait_round_robin() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let sharded = dataset.shard_round_robin(2, 0).unwrap();
        assert_eq!(sharded.len(), 5);
    }

    #[test]
    fn test_extension_trait_contiguous() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let sharded = dataset.shard_contiguous(2, 0).unwrap();
        assert_eq!(sharded.len(), 5);
        assert_eq!(sharded.indices(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_extension_trait_shuffled() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let sharded = dataset.shard_shuffled(2, 0, 42).unwrap();
        assert_eq!(sharded.len(), 5);
    }

    #[test]
    fn test_shard_access() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6, 1]).unwrap();
        let labels =
            Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ShardConfig::new(2, 0).unwrap();
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        // Rank 0 should get indices [0, 2, 4]
        let (f0, l0) = sharded.get(0).unwrap();
        let (f1, l1) = sharded.get(1).unwrap();
        let (f2, l2) = sharded.get(2).unwrap();

        // Verify we're accessing the correct original indices
        assert!((f0.to_vec().unwrap()[0] - 1.0).abs() < 1e-6);
        assert!((l0.to_vec().unwrap()[0] - 10.0).abs() < 1e-6);

        assert!((f1.to_vec().unwrap()[0] - 3.0).abs() < 1e-6);
        assert!((l1.to_vec().unwrap()[0] - 30.0).abs() < 1e-6);

        assert!((f2.to_vec().unwrap()[0] - 5.0).abs() < 1e-6);
        assert!((l2.to_vec().unwrap()[0] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_shard_out_of_bounds() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 6], &[6, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 6], &[6]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let sharded = dataset.shard_round_robin(2, 0).unwrap();
        assert_eq!(sharded.len(), 3);
        assert!(sharded.get(3).is_err());
    }

    #[test]
    fn test_empty_dataset_sharding() {
        let features = Tensor::<f32>::from_vec(vec![], &[0, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![], &[0]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let sharded = dataset.shard_round_robin(2, 0).unwrap();
        assert_eq!(sharded.len(), 0);
    }

    #[test]
    fn test_shard_statistics_balanced() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 12], &[12, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 12], &[12]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ShardConfig::new(3, 0).unwrap(); // 12/3 = 4 each, perfectly balanced
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        let stats = sharded.shard_stats();
        assert!(stats.is_balanced());
        assert_eq!(stats.imbalance_ratio, 1.0);
    }

    #[test]
    fn test_shard_statistics_report() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let config = ShardConfig::new(3, 0).unwrap();
        let sharded = ShardedDataset::new(dataset, config).unwrap();

        let report = sharded.shard_stats().report();
        assert!(report.contains("Total samples: 10"));
        assert!(report.contains("World size: 3"));
        assert!(report.contains("Rank: 0"));
    }

    #[test]
    fn test_stratified_sharding() {
        // Create dataset with 3 classes: [0,0,1,1,2,2] (repeated)
        let features = Tensor::<f32>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[12, 1],
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            &[12],
        )
        .unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Label extractor: extract the scalar value from label tensor
        let label_extractor = |label_tensor: &Tensor<f32>| -> Result<usize> {
            let data = label_tensor
                .to_vec()
                .map_err(|e| tenflowers_core::TensorError::invalid_argument(e.to_string()))?;
            Ok(data[0] as usize)
        };

        // Shard into 2 workers with stratified strategy
        let config = ShardConfig::new(2, 0)
            .unwrap()
            .with_strategy(ShardStrategy::Stratified)
            .with_seed(42);

        let sharded = ShardedDataset::new_stratified(dataset, config, label_extractor).unwrap();

        // Each worker should get balanced class distribution
        // With 4 samples of each class and 2 workers, each worker should get 2 of each class
        assert_eq!(sharded.len(), 6); // 12 samples / 2 workers = 6 per worker

        // Verify that we can access samples
        for i in 0..sharded.len() {
            let (feature, label) = sharded.get(i).unwrap();
            assert!(feature.to_vec().is_ok());
            assert!(label.to_vec().is_ok());
        }
    }

    #[test]
    fn test_stratified_sharding_balanced_classes() {
        // Create dataset with balanced classes
        let features = Tensor::<f32>::from_vec(vec![1.0; 60], &[60, 1]).unwrap();
        // 20 samples each of class 0, 1, 2
        let mut label_data = vec![0.0; 20];
        label_data.extend(vec![1.0; 20]);
        label_data.extend(vec![2.0; 20]);
        let labels = Tensor::<f32>::from_vec(label_data, &[60]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let label_extractor = |label_tensor: &Tensor<f32>| -> Result<usize> {
            let data = label_tensor
                .to_vec()
                .map_err(|e| tenflowers_core::TensorError::invalid_argument(e.to_string()))?;
            Ok(data[0] as usize)
        };

        // Shard into 3 workers
        let config = ShardConfig::new(3, 0)
            .unwrap()
            .with_strategy(ShardStrategy::Stratified)
            .with_seed(123);

        let sharded = ShardedDataset::new_stratified(dataset, config, label_extractor).unwrap();

        // Each worker should get approximately 20 samples (60 / 3)
        // Due to round-robin distribution within classes, the exact count may vary slightly
        // 20 samples of each class / 3 workers = 6-7 per class per worker
        // Total per worker = 6*3 + 1*3 = 18-21 samples (allowing for rounding)
        assert!(sharded.len() >= 18 && sharded.len() <= 21);
    }

    #[test]
    fn test_stratified_sharding_deterministic() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 30], &[30, 1]).unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
            ],
            &[30],
        )
        .unwrap();
        let dataset1 = TensorDataset::new(features.clone(), labels.clone());
        let dataset2 = TensorDataset::new(features, labels);

        let label_extractor1 = |label_tensor: &Tensor<f32>| -> Result<usize> {
            let data = label_tensor
                .to_vec()
                .map_err(|e| tenflowers_core::TensorError::invalid_argument(e.to_string()))?;
            Ok(data[0] as usize)
        };

        let label_extractor2 = |label_tensor: &Tensor<f32>| -> Result<usize> {
            let data = label_tensor
                .to_vec()
                .map_err(|e| tenflowers_core::TensorError::invalid_argument(e.to_string()))?;
            Ok(data[0] as usize)
        };

        // Same seed should produce same results
        let config1 = ShardConfig::new(2, 0)
            .unwrap()
            .with_strategy(ShardStrategy::Stratified)
            .with_seed(999);

        let config2 = ShardConfig::new(2, 0)
            .unwrap()
            .with_strategy(ShardStrategy::Stratified)
            .with_seed(999);

        let sharded1 = ShardedDataset::new_stratified(dataset1, config1, label_extractor1).unwrap();
        let sharded2 = ShardedDataset::new_stratified(dataset2, config2, label_extractor2).unwrap();

        // Same seed should produce same indices
        assert_eq!(sharded1.indices(), sharded2.indices());
    }
}
