//! Dataset caching wrappers
//!
//! This module provides dataset implementations that add caching functionality.

use crate::cache::lru::ThreadSafeLruCache;
use crate::Dataset;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, Tensor, TensorError};

/// Cached dataset wrapper that caches individual samples
pub struct CachedDataset<T, D: Dataset<T>> {
    dataset: D,
    cache: ThreadSafeLruCache<usize, (Tensor<T>, Tensor<T>)>,
    cache_stats: Arc<Mutex<CacheStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub total_requests: usize,
}

impl CacheStats {
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }
}

impl<T, D: Dataset<T>> CachedDataset<T, D>
where
    T: Clone + Send + Sync + 'static,
{
    /// Create a new cached dataset with the specified cache capacity
    pub fn new(dataset: D, cache_capacity: usize) -> Self {
        Self {
            dataset,
            cache: ThreadSafeLruCache::new(cache_capacity),
            cache_stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        match self.cache_stats.lock() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(TensorError::CacheError {
                operation: "cache_stats".to_string(),
                details: "Cache stats mutex poisoned".to_string(),
                recoverable: true,
                context: None,
            }),
        }
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        self.cache.clear()?;
        match self.cache_stats.lock() {
            Ok(mut stats) => {
                *stats = CacheStats::default();
                Ok(())
            }
            Err(_) => Err(TensorError::CacheError {
                operation: "clear_cache_stats".to_string(),
                details: "Cache stats mutex poisoned during clear".to_string(),
                recoverable: false,
                context: None,
            }),
        }
    }

    /// Pre-warm cache with specified indices
    pub fn warm_cache(&self, indices: &[usize]) -> Result<()> {
        for &index in indices {
            let _ = self.get(index)?;
        }
        Ok(())
    }

    /// Get underlying dataset
    pub fn into_inner(self) -> D {
        self.dataset
    }

    /// Get reference to underlying dataset
    pub fn inner(&self) -> &D {
        &self.dataset
    }
}

impl<T, D: Dataset<T>> Dataset<T> for CachedDataset<T, D>
where
    T: Clone + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        // Update stats
        match self.cache_stats.lock() {
            Ok(mut stats) => stats.total_requests += 1,
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "cache_stats_update".to_string(),
                    details: "Cache stats mutex poisoned during total requests update".to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        }

        // Try cache first
        if let Some(cached_sample) = self.cache.get(&index)? {
            // Cache hit - update hit stats
            match self.cache_stats.lock() {
                Ok(mut stats) => stats.hits += 1,
                Err(_) => {
                    return Err(TensorError::CacheError {
                        operation: "cache_hit_stats".to_string(),
                        details: "Cache stats mutex poisoned during hit update".to_string(),
                        recoverable: false,
                        context: None,
                    })
                }
            }
            return Ok(cached_sample);
        }

        // Cache miss - load from dataset
        let sample = self.dataset.get(index)?;

        // Cache the result
        self.cache.insert(index, sample.clone())?;

        // Update miss stats
        match self.cache_stats.lock() {
            Ok(mut stats) => stats.misses += 1,
            Err(_) => {
                return Err(TensorError::CacheError {
                    operation: "cache_miss_stats".to_string(),
                    details: "Cache stats mutex poisoned during miss update".to_string(),
                    recoverable: false,
                    context: None,
                })
            }
        }

        Ok(sample)
    }
}

/// Cache warming strategies
pub enum WarmingStrategy {
    /// Warm cache with sequential indices
    Sequential { start: usize, count: usize },
    /// Warm cache with random indices
    Random { count: usize, seed: Option<u64> },
    /// Warm cache with specific indices
    Specific(Vec<usize>),
}

impl WarmingStrategy {
    /// Generate indices based on the warming strategy
    pub fn generate_indices(&self, dataset_len: usize) -> Vec<usize> {
        match self {
            WarmingStrategy::Sequential { start, count } => {
                let end = (*start + *count).min(dataset_len);
                (*start..end).collect()
            }
            WarmingStrategy::Random { count, seed } => {
                use std::collections::HashSet;

                let mut indices = Vec::new();
                let mut seen = HashSet::new();
                let mut state = seed.unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_else(|_| std::time::Duration::from_secs(12345))
                        .as_secs()
                });

                while indices.len() < *count && indices.len() < dataset_len {
                    // Simple LCG random number generator
                    state = state.wrapping_mul(1103515245).wrapping_add(12345);
                    let idx = (state as usize) % dataset_len;

                    if seen.insert(idx) {
                        indices.push(idx);
                    }
                }

                indices
            }
            WarmingStrategy::Specific(indices) => indices
                .iter()
                .filter(|&&idx| idx < dataset_len)
                .copied()
                .collect(),
        }
    }
}
