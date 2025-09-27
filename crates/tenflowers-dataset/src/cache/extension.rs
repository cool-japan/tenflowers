//! Extension traits for adding caching to datasets
//!
//! This module provides convenient extension methods for any dataset type.

use crate::cache::dataset::{CachedDataset, WarmingStrategy};
use crate::Dataset;
use tenflowers_core::Result;

/// Extension trait for adding caching to any dataset
pub trait CacheExt<T>: Dataset<T> + Sized {
    /// Wrap this dataset with caching
    fn cached(self, capacity: usize) -> CachedDataset<T, Self>
    where
        T: Clone + Send + Sync + 'static,
    {
        CachedDataset::new(self, capacity)
    }

    /// Wrap this dataset with caching and pre-warm the cache
    fn cached_with_warming(
        self,
        capacity: usize,
        strategy: WarmingStrategy,
    ) -> Result<CachedDataset<T, Self>>
    where
        T: Clone + Send + Sync + 'static,
    {
        let cached = CachedDataset::new(self, capacity);
        let indices = strategy.generate_indices(cached.len());
        cached.warm_cache(&indices)?;
        Ok(cached)
    }
}

impl<T, D: Dataset<T>> CacheExt<T> for D {}
