//! Smart caching system with adaptive policies and multi-tier caching
//!
//! This module provides advanced caching strategies that adapt to access patterns
//! and provide multi-tier memory management for optimal performance.

use crate::Dataset;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor};

/// Access pattern tracking for adaptive caching decisions
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AccessPattern {
    /// Last access time
    last_access: Instant,
    /// Number of accesses
    access_count: u64,
    /// Average time between accesses
    avg_interval: Duration,
    /// Is this a sequential access pattern?
    is_sequential: bool,
    /// Frequency score (higher = more frequently accessed)
    frequency_score: f64,
}

impl AccessPattern {
    fn new() -> Self {
        Self {
            last_access: Instant::now(),
            access_count: 1,
            avg_interval: Duration::from_secs(0),
            is_sequential: false,
            frequency_score: 1.0,
        }
    }

    fn update(&mut self, now: Instant) {
        let interval = now.duration_since(self.last_access);

        // Update average interval with exponential moving average
        if self.access_count > 1 {
            let alpha = 0.1;
            self.avg_interval = Duration::from_secs_f64(
                alpha * interval.as_secs_f64() + (1.0 - alpha) * self.avg_interval.as_secs_f64(),
            );
        } else {
            self.avg_interval = interval;
        }

        self.last_access = now;
        self.access_count += 1;

        // Update frequency score (decay over time, boost with access)
        let time_decay = (-interval.as_secs_f64() / 300.0).exp(); // 5-minute half-life
        self.frequency_score = self.frequency_score * time_decay + 1.0;
    }

    fn priority_score(&self) -> f64 {
        let recency_score = 1.0 / (1.0 + self.last_access.elapsed().as_secs_f64() / 60.0);
        let frequency_weight = 0.7;
        let recency_weight = 0.3;

        frequency_weight * self.frequency_score + recency_weight * recency_score
    }
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Adaptive based on access patterns
    Adaptive,
    /// Time-based with TTL
    TimeBasedTTL(Duration),
    /// Hybrid: combines multiple strategies
    Hybrid,
}

/// Multi-tier cache levels
#[derive(Debug, Clone)]
pub enum CacheLevel {
    /// Fast memory cache (e.g., RAM)
    L1Memory,
    /// Slower but larger storage (e.g., SSD)
    L2Storage,
    /// Very slow but huge storage (e.g., HDD, remote)
    L3Remote,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry<T> {
    data: (Tensor<T>, Tensor<T>),
    pattern: AccessPattern,
    size: usize,
    level: CacheLevel,
    compressed: bool,
    ttl: Option<Instant>,
}

impl<T> CacheEntry<T> {
    fn new(data: (Tensor<T>, Tensor<T>), level: CacheLevel) -> Self {
        let size = data.0.shape().size() + data.1.shape().size();
        Self {
            data,
            pattern: AccessPattern::new(),
            size,
            level,
            compressed: false,
            ttl: None,
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            Instant::now() > ttl
        } else {
            false
        }
    }
}

/// Smart adaptive cache with multi-tier support
pub struct SmartCache<T, K>
where
    K: Eq + Hash + Clone,
{
    /// L1 cache: fast memory
    l1_cache: Arc<RwLock<HashMap<K, CacheEntry<T>>>>,
    /// L2 cache: larger but slower storage
    l2_cache: Arc<RwLock<HashMap<K, CacheEntry<T>>>>,
    /// L3 cache: very large remote/disk storage
    l3_cache: Arc<RwLock<HashMap<K, CacheEntry<T>>>>,

    /// Maximum size for each cache level
    l1_max_size: usize,
    l2_max_size: usize,
    l3_max_size: usize,

    /// Current size for each cache level
    l1_current_size: Arc<Mutex<usize>>,
    l2_current_size: Arc<Mutex<usize>>,
    l3_current_size: Arc<Mutex<usize>>,

    /// Eviction policy
    policy: EvictionPolicy,

    /// Access order tracking for LRU
    l1_access_order: Arc<Mutex<VecDeque<K>>>,
    l2_access_order: Arc<Mutex<VecDeque<K>>>,
    l3_access_order: Arc<Mutex<VecDeque<K>>>,

    /// Statistics
    stats: Arc<Mutex<CacheStats>>,

    /// Configuration
    config: CacheConfig,
}

/// Cache configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheConfig {
    /// Enable compression for larger cache levels
    enable_compression: bool,
    /// TTL for cache entries
    default_ttl: Option<Duration>,
    /// Threshold for promoting entries between cache levels
    promotion_threshold: f64,
    /// Threshold for demoting entries between cache levels
    demotion_threshold: f64,
    /// Maximum memory usage before triggering aggressive eviction
    memory_pressure_threshold: f64,
    /// Background cleanup interval
    cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            default_ttl: Some(Duration::from_secs(3600)), // 1 hour
            promotion_threshold: 3.0,
            demotion_threshold: 0.5,
            memory_pressure_threshold: 0.8,
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub total_requests: u64,
    pub avg_access_time: Duration,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            l1_hits: 0,
            l2_hits: 0,
            l3_hits: 0,
            misses: 0,
            evictions: 0,
            promotions: 0,
            demotions: 0,
            total_requests: 0,
            avg_access_time: Duration::from_secs(0),
        }
    }
}

impl<T, K> SmartCache<T, K>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
    K: Eq + Hash + Clone + Send + Sync,
{
    /// Create a new smart cache with specified capacities
    pub fn new(
        l1_max_size: usize,
        l2_max_size: usize,
        l3_max_size: usize,
        policy: EvictionPolicy,
        config: CacheConfig,
    ) -> Self {
        Self {
            l1_cache: Arc::new(RwLock::new(HashMap::new())),
            l2_cache: Arc::new(RwLock::new(HashMap::new())),
            l3_cache: Arc::new(RwLock::new(HashMap::new())),
            l1_max_size,
            l2_max_size,
            l3_max_size,
            l1_current_size: Arc::new(Mutex::new(0)),
            l2_current_size: Arc::new(Mutex::new(0)),
            l3_current_size: Arc::new(Mutex::new(0)),
            policy,
            l1_access_order: Arc::new(Mutex::new(VecDeque::new())),
            l2_access_order: Arc::new(Mutex::new(VecDeque::new())),
            l3_access_order: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(CacheStats::default())),
            config,
        }
    }

    /// Get an item from cache (checks all levels)
    pub fn get(&self, key: &K) -> Option<(Tensor<T>, Tensor<T>)> {
        let start_time = Instant::now();
        let mut stats = self.stats.lock().unwrap();
        stats.total_requests += 1;
        drop(stats);

        // Check L1 cache first
        if let Some(mut entry) = self.get_from_level(key, CacheLevel::L1Memory) {
            entry.pattern.update(Instant::now());
            self.update_stats_hit(CacheLevel::L1Memory, start_time);
            return Some(entry.data);
        }

        // Check L2 cache
        if let Some(mut entry) = self.get_from_level(key, CacheLevel::L2Storage) {
            entry.pattern.update(Instant::now());

            // Consider promotion to L1 based on access pattern
            if entry.pattern.priority_score() > self.config.promotion_threshold {
                self.promote_entry(key.clone(), entry.clone(), CacheLevel::L1Memory);
            }

            self.update_stats_hit(CacheLevel::L2Storage, start_time);
            return Some(entry.data);
        }

        // Check L3 cache
        if let Some(mut entry) = self.get_from_level(key, CacheLevel::L3Remote) {
            entry.pattern.update(Instant::now());

            // Consider promotion to L2 or L1 based on access pattern
            if entry.pattern.priority_score() > self.config.promotion_threshold {
                self.promote_entry(key.clone(), entry.clone(), CacheLevel::L2Storage);
            }

            self.update_stats_hit(CacheLevel::L3Remote, start_time);
            return Some(entry.data);
        }

        // Cache miss
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
        None
    }

    /// Put an item into cache (automatically selects appropriate level)
    pub fn put(&self, key: K, value: (Tensor<T>, Tensor<T>)) {
        let entry = CacheEntry::new(value, CacheLevel::L1Memory);

        // Try to insert into L1 first
        if self.try_insert_at_level(key.clone(), entry.clone(), CacheLevel::L1Memory) {
            return;
        }

        // If L1 is full, try L2
        if self.try_insert_at_level(key.clone(), entry.clone(), CacheLevel::L2Storage) {
            return;
        }

        // If L2 is full, use L3
        self.try_insert_at_level(key, entry, CacheLevel::L3Remote);
    }

    fn get_from_level(&self, key: &K, level: CacheLevel) -> Option<CacheEntry<T>> {
        let cache = match level {
            CacheLevel::L1Memory => &self.l1_cache,
            CacheLevel::L2Storage => &self.l2_cache,
            CacheLevel::L3Remote => &self.l3_cache,
        };

        let cache_read = cache.read().unwrap();
        cache_read.get(key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(entry.clone())
            }
        })
    }

    fn try_insert_at_level(&self, key: K, mut entry: CacheEntry<T>, level: CacheLevel) -> bool {
        let (cache, current_size, max_size, access_order) = match level {
            CacheLevel::L1Memory => (
                &self.l1_cache,
                &self.l1_current_size,
                self.l1_max_size,
                &self.l1_access_order,
            ),
            CacheLevel::L2Storage => (
                &self.l2_cache,
                &self.l2_current_size,
                self.l2_max_size,
                &self.l2_access_order,
            ),
            CacheLevel::L3Remote => (
                &self.l3_cache,
                &self.l3_current_size,
                self.l3_max_size,
                &self.l3_access_order,
            ),
        };

        entry.level = level.clone();
        if let Some(ttl) = self.config.default_ttl {
            entry.ttl = Some(Instant::now() + ttl);
        }

        let mut size_guard = current_size.lock().unwrap();

        // Check if we need to evict entries
        while *size_guard + entry.size > max_size {
            if !self.evict_from_level(level.clone()) {
                return false; // Cannot evict, cache full
            }
            *size_guard = current_size.lock().unwrap().saturating_sub(entry.size);
        }

        // Insert the entry
        let mut cache_write = cache.write().unwrap();
        cache_write.insert(key.clone(), entry.clone());
        *size_guard += entry.size;

        // Update access order for LRU
        let mut order = access_order.lock().unwrap();
        order.push_back(key);

        true
    }

    fn evict_from_level(&self, level: CacheLevel) -> bool {
        let (cache, current_size, access_order) = match level {
            CacheLevel::L1Memory => (&self.l1_cache, &self.l1_current_size, &self.l1_access_order),
            CacheLevel::L2Storage => (&self.l2_cache, &self.l2_current_size, &self.l2_access_order),
            CacheLevel::L3Remote => (&self.l3_cache, &self.l3_current_size, &self.l3_access_order),
        };

        let victim_key = match self.policy {
            EvictionPolicy::LRU => {
                let mut order = access_order.lock().unwrap();
                order.pop_front()
            }
            EvictionPolicy::LFU | EvictionPolicy::Adaptive | EvictionPolicy::Hybrid => {
                self.find_lfu_victim(cache)
            }
            EvictionPolicy::TimeBasedTTL(_) => self.find_expired_victim(cache),
        };

        if let Some(key) = victim_key {
            let mut cache_write = cache.write().unwrap();
            if let Some(entry) = cache_write.remove(&key) {
                let mut size_guard = current_size.lock().unwrap();
                *size_guard = size_guard.saturating_sub(entry.size);

                let mut stats = self.stats.lock().unwrap();
                stats.evictions += 1;

                return true;
            }
        }

        false
    }

    fn find_lfu_victim(&self, cache: &Arc<RwLock<HashMap<K, CacheEntry<T>>>>) -> Option<K> {
        let cache_read = cache.read().unwrap();
        cache_read
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.pattern
                    .priority_score()
                    .partial_cmp(&b.pattern.priority_score())
                    .unwrap()
            })
            .map(|(k, _)| k.clone())
    }

    fn find_expired_victim(&self, cache: &Arc<RwLock<HashMap<K, CacheEntry<T>>>>) -> Option<K> {
        let cache_read = cache.read().unwrap();
        cache_read
            .iter()
            .find(|(_, entry)| entry.is_expired())
            .map(|(k, _)| k.clone())
    }

    fn promote_entry(&self, key: K, entry: CacheEntry<T>, target_level: CacheLevel) {
        let original_level = entry.level.clone();
        if self.try_insert_at_level(key.clone(), entry, target_level) {
            // Remove from lower level
            match original_level {
                CacheLevel::L3Remote => {
                    let mut cache = self.l3_cache.write().unwrap();
                    cache.remove(&key);
                }
                CacheLevel::L2Storage => {
                    let mut cache = self.l2_cache.write().unwrap();
                    cache.remove(&key);
                }
                _ => {}
            }

            let mut stats = self.stats.lock().unwrap();
            stats.promotions += 1;
        }
    }

    fn update_stats_hit(&self, level: CacheLevel, start_time: Instant) {
        let mut stats = self.stats.lock().unwrap();
        match level {
            CacheLevel::L1Memory => stats.l1_hits += 1,
            CacheLevel::L2Storage => stats.l2_hits += 1,
            CacheLevel::L3Remote => stats.l3_hits += 1,
        }

        let access_time = start_time.elapsed();
        let alpha = 0.1;
        stats.avg_access_time = Duration::from_secs_f64(
            alpha * access_time.as_secs_f64() + (1.0 - alpha) * stats.avg_access_time.as_secs_f64(),
        );
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all cache levels
    pub fn clear(&self) {
        let mut l1 = self.l1_cache.write().unwrap();
        let mut l2 = self.l2_cache.write().unwrap();
        let mut l3 = self.l3_cache.write().unwrap();

        l1.clear();
        l2.clear();
        l3.clear();

        *self.l1_current_size.lock().unwrap() = 0;
        *self.l2_current_size.lock().unwrap() = 0;
        *self.l3_current_size.lock().unwrap() = 0;
    }

    /// Run background cleanup to remove expired entries
    pub fn cleanup_expired(&self) {
        for level in [
            CacheLevel::L1Memory,
            CacheLevel::L2Storage,
            CacheLevel::L3Remote,
        ] {
            while self
                .find_expired_victim(match level {
                    CacheLevel::L1Memory => &self.l1_cache,
                    CacheLevel::L2Storage => &self.l2_cache,
                    CacheLevel::L3Remote => &self.l3_cache,
                })
                .is_some()
            {
                self.evict_from_level(level.clone());
            }
        }
    }
}

/// Smart cached dataset wrapper
pub struct SmartCachedDataset<T, D: Dataset<T>> {
    dataset: D,
    cache: Arc<SmartCache<T, usize>>,
}

impl<T, D: Dataset<T>> SmartCachedDataset<T, D>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
{
    /// Create a new smart cached dataset
    pub fn new(
        dataset: D,
        l1_size: usize,
        l2_size: usize,
        l3_size: usize,
        policy: EvictionPolicy,
        config: CacheConfig,
    ) -> Self {
        let cache = Arc::new(SmartCache::new(l1_size, l2_size, l3_size, policy, config));

        Self { dataset, cache }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl<T, D: Dataset<T>> Dataset<T> for SmartCachedDataset<T, D>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        // Try cache first
        if let Some(cached) = self.cache.get(&index) {
            return Ok(cached);
        }

        // Cache miss - load from dataset
        let sample = self.dataset.get(index)?;

        // Store in cache
        self.cache.put(index, sample.clone());

        Ok(sample)
    }
}

/// Predictive access pattern analyzer for smart prefetching
#[derive(Debug, Clone)]
pub struct AccessPatternPredictor<K>
where
    K: Eq + Hash + Clone + Send + Sync,
{
    /// History of recent accesses (sliding window)
    access_history: VecDeque<(K, Instant)>,
    /// Patterns detected in access sequences
    sequence_patterns: HashMap<Vec<K>, f64>,
    /// Maximum history size to maintain
    max_history_size: usize,
    /// Minimum pattern length to consider
    min_pattern_length: usize,
    /// Maximum pattern length to consider
    max_pattern_length: usize,
}

impl<K> AccessPatternPredictor<K>
where
    K: Eq + Hash + Clone + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            access_history: VecDeque::with_capacity(1000),
            sequence_patterns: HashMap::new(),
            max_history_size: 1000,
            min_pattern_length: 2,
            max_pattern_length: 5,
        }
    }

    /// Record a new access and update patterns
    pub fn record_access(&mut self, key: K) {
        let now = Instant::now();

        // Add to history
        self.access_history.push_back((key.clone(), now));

        // Maintain sliding window
        if self.access_history.len() > self.max_history_size {
            self.access_history.pop_front();
        }

        // Update sequence patterns
        self.update_patterns();
    }

    /// Predict the next likely accesses based on recent patterns
    pub fn predict_next_accesses(&self, current_key: &K, max_predictions: usize) -> Vec<(K, f64)> {
        let mut predictions = Vec::new();

        // Look for patterns ending with the current key
        for pattern_len in self.min_pattern_length..=self.max_pattern_length {
            if let Some(recent_sequence) = self.get_recent_sequence(pattern_len) {
                if recent_sequence.last() == Some(current_key) {
                    // Find patterns that start with this sequence
                    for (pattern, confidence) in &self.sequence_patterns {
                        if pattern.len() > pattern_len
                            && pattern[..pattern_len] == recent_sequence[..]
                        {
                            let next_key = &pattern[pattern_len];
                            predictions.push((next_key.clone(), *confidence));
                        }
                    }
                }
            }
        }

        // Sort by confidence and return top predictions
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        predictions.truncate(max_predictions);
        predictions
    }

    /// Get recent access sequence of specified length
    fn get_recent_sequence(&self, length: usize) -> Option<Vec<K>> {
        if self.access_history.len() < length {
            return None;
        }

        let recent: Vec<K> = self
            .access_history
            .iter()
            .rev()
            .take(length)
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        Some(recent)
    }

    /// Update sequence patterns based on access history
    fn update_patterns(&mut self) {
        let history_keys: Vec<K> = self.access_history.iter().map(|(k, _)| k.clone()).collect();

        // Extract patterns of different lengths
        for pattern_len in self.min_pattern_length..=self.max_pattern_length {
            if history_keys.len() >= pattern_len {
                for i in 0..=(history_keys.len() - pattern_len) {
                    let pattern = history_keys[i..i + pattern_len].to_vec();

                    // Exponential decay for older patterns
                    let age_factor = 1.0 - (i as f64 / history_keys.len() as f64 * 0.1);

                    *self.sequence_patterns.entry(pattern).or_insert(0.0) += age_factor;
                }
            }
        }

        // Decay all patterns over time to prevent unbounded growth
        for confidence in self.sequence_patterns.values_mut() {
            *confidence *= 0.99; // Small decay factor
        }

        // Remove patterns with very low confidence
        self.sequence_patterns
            .retain(|_, confidence| *confidence > 0.1);
    }
}

impl<K> Default for AccessPatternPredictor<K>
where
    K: Eq + Hash + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced smart cache with predictive prefetching
pub struct PredictiveSmartCache<T, K>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
    K: Eq + Hash + Clone + Send + Sync,
{
    /// Base smart cache
    base_cache: SmartCache<T, K>,
    /// Pattern predictor for prefetching
    predictor: Arc<Mutex<AccessPatternPredictor<K>>>,
    /// Reference to the dataset for prefetching
    dataset: Option<Arc<dyn Dataset<T>>>,
    /// Prefetch queue
    prefetch_queue: Arc<Mutex<VecDeque<K>>>,
    /// Maximum prefetch queue size
    max_prefetch_size: usize,
}

impl<T, K> PredictiveSmartCache<T, K>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
    K: Eq + Hash + Clone + Send + Sync,
{
    pub fn new(
        l1_max_size: usize,
        l2_max_size: usize,
        l3_max_size: usize,
        policy: EvictionPolicy,
        config: CacheConfig,
        max_prefetch_size: usize,
    ) -> Self {
        Self {
            base_cache: SmartCache::new(l1_max_size, l2_max_size, l3_max_size, policy, config),
            predictor: Arc::new(Mutex::new(AccessPatternPredictor::new())),
            dataset: None,
            prefetch_queue: Arc::new(Mutex::new(VecDeque::with_capacity(max_prefetch_size))),
            max_prefetch_size,
        }
    }

    /// Set the dataset reference for prefetching
    pub fn set_dataset(&mut self, dataset: Arc<dyn Dataset<T>>) {
        self.dataset = Some(dataset);
    }

    /// Get item with predictive prefetching
    pub fn get(&self, key: &K) -> Option<(Tensor<T>, Tensor<T>)> {
        // Record access for pattern learning
        {
            let mut predictor = self.predictor.lock().unwrap();
            predictor.record_access(key.clone());
        }

        // Try to get from base cache first
        if let Some(result) = self.base_cache.get(key) {
            // Trigger predictive prefetching based on this access
            self.trigger_prefetch(key);
            return Some(result);
        }

        // Cache miss - load from dataset if available
        if let Some(ref dataset) = self.dataset {
            // For this example, assume K can be converted to usize for dataset access
            // In a real implementation, you'd need proper key-to-index mapping
            if let Some(data) = self.load_from_dataset(dataset, key) {
                self.base_cache.put(key.clone(), data.clone());
                self.trigger_prefetch(key);
                return Some(data);
            }
        }

        None
    }

    /// Put item in cache
    pub fn put(&self, key: K, data: (Tensor<T>, Tensor<T>)) {
        self.base_cache.put(key, data);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.base_cache.stats()
    }

    /// Trigger predictive prefetching based on current access
    fn trigger_prefetch(&self, current_key: &K) {
        let predictions = {
            let predictor = self.predictor.lock().unwrap();
            predictor.predict_next_accesses(current_key, 3) // Predict up to 3 next accesses
        };

        let mut prefetch_queue = self.prefetch_queue.lock().unwrap();

        for (predicted_key, confidence) in predictions {
            // Only prefetch if confidence is high enough and not already cached
            if confidence > 0.5 && self.base_cache.get(&predicted_key).is_none() {
                prefetch_queue.push_back(predicted_key);

                // Maintain queue size limit
                if prefetch_queue.len() > self.max_prefetch_size {
                    prefetch_queue.pop_front();
                }
            }
        }
    }

    /// Load data from dataset (placeholder implementation)
    fn load_from_dataset(
        &self,
        _dataset: &Arc<dyn Dataset<T>>,
        _key: &K,
    ) -> Option<(Tensor<T>, Tensor<T>)> {
        // This is a placeholder - in a real implementation, you would:
        // 1. Convert key to dataset index
        // 2. Load data from dataset
        // 3. Return the loaded data
        None
    }

    /// Process prefetch queue (should be called periodically)
    pub fn process_prefetch_queue(&self) {
        if let Some(ref dataset) = self.dataset {
            let mut prefetch_queue = self.prefetch_queue.lock().unwrap();

            // Process a few items from the prefetch queue
            for _ in 0..3 {
                if let Some(key) = prefetch_queue.pop_front() {
                    // Check if already cached
                    if self.base_cache.get(&key).is_none() {
                        if let Some(data) = self.load_from_dataset(dataset, &key) {
                            self.base_cache.put(key, data);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_smart_cache_creation() {
        let cache: SmartCache<f32, usize> = SmartCache::new(
            100,   // L1: 100 entries
            1000,  // L2: 1000 entries
            10000, // L3: 10000 entries
            EvictionPolicy::LRU,
            CacheConfig::default(),
        );

        let stats = cache.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.l1_hits, 0);
    }

    #[test]
    fn test_smart_cache_put_get() {
        let cache: SmartCache<f32, usize> = SmartCache::new(
            100,
            1000,
            10000,
            EvictionPolicy::LRU,
            CacheConfig::default(),
        );

        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0], &[]).unwrap();

        cache.put(0, (features.clone(), labels.clone()));

        let retrieved = cache.get(&0).unwrap();
        assert_eq!(retrieved.0.shape().dims(), features.shape().dims());
        assert_eq!(retrieved.1.shape().dims(), labels.shape().dims());

        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.total_requests, 1);
    }

    #[test]
    fn test_smart_cached_dataset() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();

        let base_dataset = TensorDataset::new(features, labels);
        let cached_dataset = SmartCachedDataset::new(
            base_dataset,
            10,   // L1 size
            100,  // L2 size
            1000, // L3 size
            EvictionPolicy::Adaptive,
            CacheConfig::default(),
        );

        assert_eq!(cached_dataset.len(), 2);

        // First access - cache miss
        let (feat0, _label0) = cached_dataset.get(0).unwrap();
        assert_eq!(feat0.shape().dims(), &[2]);

        // Second access - cache hit
        let (feat0_cached, _) = cached_dataset.get(0).unwrap();
        assert_eq!(feat0_cached.shape().dims(), &[2]);

        let stats = cached_dataset.cache_stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.l1_hits, 1);
    }
}
