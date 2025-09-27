//! Caching utilities for datasets
//!
//! This module provides efficient caching mechanisms to reduce data loading overhead
//! and improve dataset iteration performance.

pub mod dataset;
pub mod extension;
pub mod lru;

#[cfg(feature = "serialize")]
pub mod persistent;

// Re-export public types for convenience
pub use dataset::{CacheStats, CachedDataset, WarmingStrategy};
pub use extension::CacheExt;
pub use lru::{LruCache, ThreadSafeLruCache};

#[cfg(feature = "serialize")]
pub use persistent::{PersistentCache, PersistentlyCachedDataset, TensorPersistentCache};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dataset, TensorDataset};
    use tenflowers_core::Tensor;

    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LruCache::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);

        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LruCache::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3); // Should evict "a"

        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.get(&"c"), Some(3));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_access_ordering() {
        let mut cache = LruCache::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);

        // Access "a" to make it more recently used
        let _ = cache.get(&"a");

        cache.insert("c", 3); // Should evict "b", not "a"

        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn test_cached_dataset() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let cached_dataset = dataset.cached(2);

        assert_eq!(cached_dataset.len(), 3);

        // First access - should be cache miss
        let (features1, _label1) = cached_dataset.get(0).unwrap();
        assert_eq!(features1.shape().dims(), &[2]);

        // Second access to same index - should be cache hit
        let (features2, _label2) = cached_dataset.get(0).unwrap();
        assert_eq!(features2.shape().dims(), &[2]);

        let stats = cached_dataset.cache_stats().unwrap();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cache_warming_sequential() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let cached_dataset = dataset
            .cached_with_warming(4, WarmingStrategy::Sequential { start: 0, count: 2 })
            .unwrap();

        // These should be cache hits since we pre-warmed indices 0 and 1
        let _ = cached_dataset.get(0).unwrap();
        let _ = cached_dataset.get(1).unwrap();

        let stats = cached_dataset.cache_stats().unwrap();
        assert!(stats.hits >= 2);
    }

    #[test]
    fn test_warming_strategy_indices() {
        let strategy = WarmingStrategy::Sequential { start: 1, count: 3 };
        let indices = strategy.generate_indices(10);
        assert_eq!(indices, vec![1, 2, 3]);

        let strategy = WarmingStrategy::Specific(vec![0, 2, 4]);
        let indices = strategy.generate_indices(10);
        assert_eq!(indices, vec![0, 2, 4]);

        let strategy = WarmingStrategy::Random {
            count: 3,
            seed: Some(42),
        };
        let indices = strategy.generate_indices(10);
        assert_eq!(indices.len(), 3);

        // Test with same seed should give same result
        let strategy2 = WarmingStrategy::Random {
            count: 3,
            seed: Some(42),
        };
        let indices2 = strategy2.generate_indices(10);
        assert_eq!(indices, indices2);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_persistent_cache_basic() {
        use std::env;

        let cache_dir = env::temp_dir().join("test_persistent_cache");
        let _ = std::fs::remove_dir_all(&cache_dir); // Clean up if exists

        let mut cache = PersistentCache::<usize, String>::new(&cache_dir, 2).unwrap();

        // Test insertion and retrieval
        cache.insert(0, "value0".to_string()).unwrap();
        cache.insert(1, "value1".to_string()).unwrap();

        assert_eq!(cache.get(&0).unwrap(), Some("value0".to_string()));
        assert_eq!(cache.get(&1).unwrap(), Some("value1".to_string()));
        assert_eq!(cache.len(), 2);

        // Test eviction
        cache.insert(2, "value2".to_string()).unwrap();
        assert_eq!(cache.len(), 2);

        // Clean up
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_persistent_cache_persistence() {
        use std::env;

        let cache_dir = env::temp_dir().join("test_persistent_cache_persistence");
        let _ = std::fs::remove_dir_all(&cache_dir); // Clean up if exists

        // Create cache and add some data
        {
            let mut cache = PersistentCache::<usize, String>::new(&cache_dir, 3).unwrap();
            cache.insert(0, "persistent_value0".to_string()).unwrap();
            cache.insert(1, "persistent_value1".to_string()).unwrap();
        }

        // Create new cache instance and verify data persists
        {
            let mut cache = PersistentCache::<usize, String>::new(&cache_dir, 3).unwrap();
            assert_eq!(
                cache.get(&0).unwrap(),
                Some("persistent_value0".to_string())
            );
            assert_eq!(
                cache.get(&1).unwrap(),
                Some("persistent_value1".to_string())
            );
        }

        // Clean up
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn test_persistently_cached_dataset() {
        use std::env;

        let cache_dir = env::temp_dir().join("test_persistently_cached_dataset");
        let _ = std::fs::remove_dir_all(&cache_dir); // Clean up if exists

        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        let cached_dataset = PersistentlyCachedDataset::new(dataset, &cache_dir, 2).unwrap();

        // First access - should be cache miss
        let (features1, _) = cached_dataset.get(0).unwrap();
        assert_eq!(features1.shape().dims(), &[2]);

        // Second access to same index - should be cache hit now that serialization works
        let (features2, _) = cached_dataset.get(0).unwrap();
        assert_eq!(features2.shape().dims(), &[2]);

        let stats = cached_dataset.cache_stats().unwrap();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.hits, 1); // Second access should be a hit
        assert_eq!(stats.misses, 1); // First access should be a miss

        // Clean up
        let _ = std::fs::remove_dir_all(&cache_dir);
    }
}
