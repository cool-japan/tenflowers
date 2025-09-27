//! LRU (Least Recently Used) cache implementations
//!
//! This module provides thread-safe and non-thread-safe LRU cache implementations.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Result, TensorError};

/// LRU (Least Recently Used) cache implementation
pub struct LruCache<K, V> {
    capacity: usize,
    data: HashMap<K, (V, usize)>, // (value, access_order)
    access_counter: usize,
}

impl<K: Clone + Eq + Hash, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: HashMap::new(),
            access_counter: 0,
        }
    }

    /// Get a value from the cache, updating its access time
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, access_time)) = self.data.get_mut(key) {
            self.access_counter += 1;
            *access_time = self.access_counter;
            Some(value.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: K, value: V) {
        self.access_counter += 1;

        if self.data.len() >= self.capacity && !self.data.contains_key(&key) {
            // Need to evict least recently used item
            if let Some(lru_key) = self.find_lru_key() {
                self.data.remove(&lru_key);
            }
        }

        self.data.insert(key, (value, self.access_counter));
    }

    /// Find the key of the least recently used item
    fn find_lru_key(&self) -> Option<K> {
        self.data
            .iter()
            .min_by_key(|(_, (_, access_time))| *access_time)
            .map(|(key, _)| key.clone())
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all cached items
    pub fn clear(&mut self) {
        self.data.clear();
        self.access_counter = 0;
    }

    /// Get cache hit ratio (for monitoring)
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Thread-safe wrapper around LruCache
pub struct ThreadSafeLruCache<K, V> {
    cache: Arc<Mutex<LruCache<K, V>>>,
}

impl<K: Clone + Eq + Hash, V: Clone> ThreadSafeLruCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
        }
    }

    pub fn get(&self, key: &K) -> Result<Option<V>> {
        let mut cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        Ok(cache.get(key))
    }

    pub fn insert(&self, key: K, value: V) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        cache.insert(key, value);
        Ok(())
    }

    pub fn len(&self) -> Result<usize> {
        let cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        Ok(cache.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        let cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        Ok(cache.is_empty())
    }

    pub fn clear(&self) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        cache.clear();
        Ok(())
    }

    pub fn capacity(&self) -> Result<usize> {
        let cache = self.cache.lock().map_err(|_| {
            TensorError::compute_error_simple("Failed to acquire cache lock".to_string())
        })?;
        Ok(cache.capacity())
    }
}

impl<K: Clone + Eq + Hash, V: Clone> Clone for ThreadSafeLruCache<K, V> {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
        }
    }
}
