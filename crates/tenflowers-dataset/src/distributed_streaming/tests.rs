//! Tests for distributed streaming module

#[cfg(test)]
mod tests {
    use crate::distributed_streaming::{
        CheckpointState, PartitionStrategy, StreamCoordinator, StreamingConfig,
        StreamingShardIterator, StreamingShardLoader, StreamingStats,
    };
    use crate::TensorDataset;
    use std::collections::HashSet;
    use std::sync::Arc;
    use tenflowers_core::Tensor;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
    }

    #[test]
    fn test_streaming_config_validation() {
        assert!(StreamingConfig::new(0, 0).is_err());
        assert!(StreamingConfig::new(4, 4).is_err());
        assert!(StreamingConfig::new(4, 3).is_ok());
    }

    #[test]
    fn test_round_robin_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 0)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::RoundRobin);

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        assert_eq!(loader.len(), 25);
    }

    #[test]
    fn test_contiguous_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 1)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::Contiguous);

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        assert_eq!(loader.len(), 25);
    }

    #[test]
    fn test_hash_based_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 0)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::HashBased {
                num_partitions: 4,
                hash_seed: 42,
            });

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        assert!(loader.len() > 0);
        assert!(loader.len() <= 100);
    }

    #[test]
    fn test_deterministic_shuffling() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 50], &[50, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 50], &[50]).expect("tensor creation should succeed");
        let dataset1 = TensorDataset::new(features.clone(), labels.clone());
        let dataset2 = TensorDataset::new(features, labels);

        let config1 = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_shuffle_seed(123);
        let config2 = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_shuffle_seed(123);

        let loader1 =
            StreamingShardLoader::new(dataset1, config1).expect("loader creation should succeed");
        let loader2 =
            StreamingShardLoader::new(dataset2, config2).expect("loader creation should succeed");

        assert_eq!(loader1.assigned_indices, loader2.assigned_indices);
    }

    #[test]
    fn test_streaming_next() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        let sample1 = loader.next().expect("next should succeed");
        assert!(sample1.is_some());

        let sample2 = loader.next().expect("next should succeed");
        assert!(sample2.is_some());
    }

    #[test]
    fn test_streaming_prefetch() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_prefetch_buffer_size(5);
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        loader.prefetch(5).expect("prefetch should succeed");

        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_some());

        let stats = loader.get_stats().expect("get_stats should succeed");
        assert!(stats.prefetch_hits > 0);
    }

    #[test]
    fn test_checkpoint_creation() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_checkpointing(5);
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        for _ in 0..6 {
            let _ = loader.next();
        }

        let stats = loader.get_stats().expect("get_stats should succeed");
        assert!(stats.num_checkpoints > 0);
    }

    #[test]
    fn test_checkpoint_restore() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        for _ in 0..3 {
            let _ = loader.next();
        }

        let checkpoint = loader
            .get_checkpoint()
            .expect("get_checkpoint should succeed");

        for _ in 0..3 {
            let _ = loader.next();
        }

        loader
            .restore_from_checkpoint(checkpoint)
            .expect("restore should succeed");

        let restored_checkpoint = loader
            .get_checkpoint()
            .expect("get_checkpoint should succeed");
        assert_eq!(restored_checkpoint.position, 3);
    }

    #[test]
    fn test_stream_reset() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        for _ in 0..5 {
            let _ = loader.next();
        }

        loader.reset().expect("reset should succeed");

        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_some());
    }

    #[test]
    fn test_stream_coordinator_creation() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator = StreamCoordinator::new(config);
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_worker_registration() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        let indices = vec![0, 1, 2, 3, 4];
        coordinator
            .register_worker(0, indices)
            .expect("worker registration should succeed");

        let health = coordinator
            .get_worker_health(0)
            .expect("get_worker_health should succeed");
        assert!(health.is_some());
        assert_eq!(health.expect("health should exist").rank, 0);
    }

    #[test]
    fn test_worker_health_update() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        coordinator
            .register_worker(0, vec![])
            .expect("worker registration should succeed");

        coordinator
            .update_worker_health(0, 100, 50.0)
            .expect("health update should succeed");

        let health = coordinator
            .get_worker_health(0)
            .expect("get_worker_health should succeed")
            .expect("health should exist");

        assert_eq!(health.samples_processed, 100);
        assert!((health.average_throughput - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_coordinator_checkpoint_management() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        let checkpoint = CheckpointState {
            epoch: 1,
            position: 100,
            shuffle_seed: Some(42),
            rank: 0,
            timestamp: 12345,
            processed_indices: HashSet::new(),
        };

        coordinator
            .register_checkpoint(0, checkpoint.clone())
            .expect("checkpoint registration should succeed");

        let retrieved = coordinator
            .get_checkpoint(0)
            .expect("get_checkpoint should succeed")
            .expect("checkpoint should exist");

        assert_eq!(retrieved.epoch, 1);
        assert_eq!(retrieved.position, 100);
    }

    #[test]
    fn test_iterator_adapter() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader = Arc::new(
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed"),
        );

        let iter = StreamingShardIterator::new(loader);

        let mut count = 0;
        for result in iter {
            assert!(result.is_ok());
            count += 1;
        }

        assert!(count > 0);
    }

    #[test]
    fn test_empty_dataset_streaming() {
        let features =
            Tensor::<f32>::from_vec(vec![], &[0, 1]).expect("empty tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![], &[0]).expect("empty tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        assert_eq!(loader.len(), 0);
        assert!(loader.is_empty());

        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_none());
    }

    #[test]
    fn test_partition_strategy_default() {
        let strategy = PartitionStrategy::default();
        assert!(matches!(strategy, PartitionStrategy::RoundRobin));
    }

    #[test]
    fn test_streaming_stats_default() {
        let stats = StreamingStats::default();
        assert_eq!(stats.samples_loaded, 0);
        assert_eq!(stats.prefetch_hits, 0);
        assert_eq!(stats.prefetch_misses, 0);
    }
}
