//! Predictive prefetching with access pattern learning
//!
//! This module provides intelligent prefetching capabilities that learn from
//! access patterns to predict and preload data before it's requested.

use crate::Dataset;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor};

/// Access pattern types detected by the system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential access (i, i+1, i+2, ...)
    Sequential { stride: usize },
    /// Random access with no detectable pattern
    Random,
    /// Repeating pattern (cycles through a set of indices)
    Cyclic { pattern: Vec<usize> },
    /// Strided access (i, i+k, i+2k, ...)
    Strided { start: usize, stride: usize },
}

/// Statistics about access patterns
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    pub total_accesses: u64,
    pub sequential_accesses: u64,
    pub random_accesses: u64,
    pub pattern_hits: u64,
    pub pattern_misses: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub bandwidth_saved: u64, // Bytes saved by avoiding disk I/O
}

impl AccessStats {
    /// Calculate pattern prediction accuracy
    pub fn pattern_accuracy(&self) -> f64 {
        let total_predictions = self.pattern_hits + self.pattern_misses;
        if total_predictions == 0 {
            0.0
        } else {
            self.pattern_hits as f64 / total_predictions as f64
        }
    }

    /// Calculate prefetch efficiency
    pub fn prefetch_efficiency(&self) -> f64 {
        let total_prefetches = self.prefetch_hits + self.prefetch_misses;
        if total_prefetches == 0 {
            0.0
        } else {
            self.prefetch_hits as f64 / total_prefetches as f64
        }
    }

    /// Calculate sequential access ratio
    pub fn sequential_ratio(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.sequential_accesses as f64 / self.total_accesses as f64
        }
    }
}

/// Prefetch entry stored in cache
#[derive(Debug)]
struct PrefetchEntry<T> {
    data: (Tensor<T>, Tensor<T>),
    timestamp: Instant,
    access_count: u32,
}

/// Pattern detector that learns access patterns
#[derive(Debug)]
struct PatternDetector {
    /// Recent access history
    access_history: VecDeque<usize>,
    /// Detected patterns and their confidence scores
    detected_patterns: HashMap<AccessPattern, f64>,
    /// Maximum history size to keep
    max_history: usize,
    /// Minimum pattern length to detect
    min_pattern_length: usize,
}

impl PatternDetector {
    fn new(max_history: usize) -> Self {
        Self {
            access_history: VecDeque::new(),
            detected_patterns: HashMap::new(),
            max_history,
            min_pattern_length: 3,
        }
    }

    /// Record a new access
    fn record_access(&mut self, index: usize) {
        self.access_history.push_back(index);

        // Trim history if too long
        while self.access_history.len() > self.max_history {
            self.access_history.pop_front();
        }

        // Analyze patterns
        self.analyze_patterns();
    }

    /// Analyze recent accesses to detect patterns
    fn analyze_patterns(&mut self) {
        if self.access_history.len() < self.min_pattern_length {
            return;
        }

        // Clear old patterns with low confidence
        self.detected_patterns
            .retain(|_, confidence| *confidence > 0.1);

        // Check for sequential pattern
        self.detect_sequential_pattern();

        // Check for strided pattern
        self.detect_strided_pattern();

        // Check for cyclic pattern
        self.detect_cyclic_pattern();
    }

    /// Detect sequential access pattern
    fn detect_sequential_pattern(&mut self) {
        let history: Vec<_> = self.access_history.iter().cloned().collect();
        let mut sequential_count = 0;

        for window in history.windows(2) {
            if window[1] == window[0] + 1 {
                sequential_count += 1;
            }
        }

        let confidence = sequential_count as f64 / (history.len() - 1) as f64;
        if confidence > 0.7 {
            self.detected_patterns
                .insert(AccessPattern::Sequential { stride: 1 }, confidence);
        }
    }

    /// Detect strided access pattern
    fn detect_strided_pattern(&mut self) {
        let history: Vec<_> = self.access_history.iter().cloned().collect();
        if history.len() < 3 {
            return;
        }

        // Try different stride values
        for stride in 2..=10 {
            let mut matches = 0;
            let start = history[0];

            for (i, &index) in history.iter().enumerate() {
                if index == start + i * stride {
                    matches += 1;
                }
            }

            let confidence = matches as f64 / history.len() as f64;
            if confidence > 0.8 {
                self.detected_patterns
                    .insert(AccessPattern::Strided { start, stride }, confidence);
            }
        }
    }

    /// Detect cyclic access pattern
    fn detect_cyclic_pattern(&mut self) {
        let history: Vec<_> = self.access_history.iter().cloned().collect();

        // Look for repeating subsequences
        for pattern_length in self.min_pattern_length..=(history.len() / 2) {
            if history.len() < pattern_length * 2 {
                continue;
            }

            let pattern: Vec<_> = history[history.len() - pattern_length..].to_vec();
            let mut repeats = 0;
            let mut total_checks = 0;

            let mut pos = history.len() - pattern_length * 2;
            while pos < history.len() - pattern_length {
                total_checks += 1;
                let segment = &history[pos..pos + pattern_length];
                if segment == pattern {
                    repeats += 1;
                }
                pos += pattern_length;
            }

            if total_checks > 0 {
                let confidence = repeats as f64 / total_checks as f64;
                if confidence > 0.8 {
                    self.detected_patterns
                        .insert(AccessPattern::Cyclic { pattern }, confidence);
                }
            }
        }
    }

    /// Predict next indices based on detected patterns
    fn predict_next(&self, current_index: usize, count: usize) -> Vec<usize> {
        let mut predictions = Vec::new();

        // Use the pattern with highest confidence
        if let Some((pattern, _)) = self
            .detected_patterns
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            match pattern {
                AccessPattern::Sequential { stride } => {
                    for i in 1..=count {
                        predictions.push(current_index + i * stride);
                    }
                }
                AccessPattern::Strided { start: _, stride } => {
                    let next_in_sequence = current_index + stride;
                    predictions.push(next_in_sequence);
                    for i in 1..count {
                        predictions.push(next_in_sequence + i * stride);
                    }
                }
                AccessPattern::Cyclic { pattern } => {
                    if let Some(current_pos) = pattern.iter().position(|&x| x == current_index) {
                        for i in 1..=count {
                            let next_pos = (current_pos + i) % pattern.len();
                            predictions.push(pattern[next_pos]);
                        }
                    }
                }
                AccessPattern::Random => {
                    // No predictions for random access
                }
            }
        }

        predictions
    }

    /// Get the most confident pattern
    pub fn dominant_pattern(&self) -> Option<AccessPattern> {
        self.detected_patterns
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(pattern, _)| pattern.clone())
    }
}

/// Predictive prefetcher that learns access patterns and preloads data
pub struct PredictivePrefetcher<T, D: Dataset<T>>
where
    T: Clone + Send + Sync + 'static,
    D: Send + Sync + 'static,
{
    /// Reference to the dataset
    dataset: Arc<D>,
    /// Pattern detector for learning access patterns
    pattern_detector: Arc<RwLock<PatternDetector>>,
    /// Prefetch cache
    prefetch_cache: Arc<RwLock<HashMap<usize, PrefetchEntry<T>>>>,
    /// Configuration
    config: PrefetchConfig,
    /// Background prefetch worker
    worker_handle: Option<JoinHandle<()>>,
    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<RwLock<AccessStats>>,
    /// Pending prefetch requests
    prefetch_queue: Arc<Mutex<VecDeque<usize>>>,
}

/// Configuration for predictive prefetching
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Maximum number of items to prefetch ahead
    pub max_prefetch_count: usize,
    /// Maximum cache size (number of items)
    pub max_cache_size: usize,
    /// Pattern detection history size
    pub pattern_history_size: usize,
    /// Cache entry TTL (time to live)
    pub cache_ttl: Duration,
    /// Prefetch worker sleep duration when idle
    pub worker_sleep_duration: Duration,
    /// Enable bandwidth optimization
    pub bandwidth_optimization: bool,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_prefetch_count: 8,
            max_cache_size: 128,
            pattern_history_size: 50,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            worker_sleep_duration: Duration::from_millis(10),
            bandwidth_optimization: true,
        }
    }
}

impl<T, D> PredictivePrefetcher<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    /// Create a new predictive prefetcher
    pub fn new(dataset: Arc<D>) -> Self {
        Self::with_config(dataset, PrefetchConfig::default())
    }

    /// Create a new predictive prefetcher with custom configuration
    pub fn with_config(dataset: Arc<D>, config: PrefetchConfig) -> Self {
        let pattern_detector = Arc::new(RwLock::new(PatternDetector::new(
            config.pattern_history_size,
        )));
        let prefetch_cache = Arc::new(RwLock::new(HashMap::new()));
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(RwLock::new(AccessStats::default()));
        let prefetch_queue = Arc::new(Mutex::new(VecDeque::new()));

        // Start background prefetch worker
        let worker_handle = Self::start_prefetch_worker(
            dataset.clone(),
            prefetch_cache.clone(),
            prefetch_queue.clone(),
            shutdown_signal.clone(),
            config.clone(),
            stats.clone(),
        );

        Self {
            dataset,
            pattern_detector,
            prefetch_cache,
            config,
            worker_handle: Some(worker_handle),
            shutdown_signal,
            stats,
            prefetch_queue,
        }
    }

    /// Start the background prefetch worker
    fn start_prefetch_worker(
        dataset: Arc<D>,
        cache: Arc<RwLock<HashMap<usize, PrefetchEntry<T>>>>,
        queue: Arc<Mutex<VecDeque<usize>>>,
        shutdown: Arc<AtomicBool>,
        config: PrefetchConfig,
        stats: Arc<RwLock<AccessStats>>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                // Process prefetch requests
                let indices_to_prefetch: Vec<usize> = {
                    let mut queue_guard = queue.lock().unwrap();
                    let mut indices = Vec::new();

                    // Take up to max_prefetch_count items
                    for _ in 0..config.max_prefetch_count {
                        if let Some(index) = queue_guard.pop_front() {
                            indices.push(index);
                        } else {
                            break;
                        }
                    }
                    indices
                };

                // Prefetch the data
                for index in indices_to_prefetch {
                    if let Ok(data) = dataset.get(index) {
                        let mut cache_guard = cache.write().unwrap();

                        // Check cache size limit
                        if cache_guard.len() >= config.max_cache_size {
                            // Remove oldest entries
                            let oldest_key = cache_guard
                                .iter()
                                .min_by_key(|(_, entry)| entry.timestamp)
                                .map(|(k, _)| *k);

                            if let Some(key) = oldest_key {
                                cache_guard.remove(&key);
                            }
                        }

                        cache_guard.insert(
                            index,
                            PrefetchEntry {
                                data,
                                timestamp: Instant::now(),
                                access_count: 0,
                            },
                        );

                        // Update stats
                        let mut stats_guard = stats.write().unwrap();
                        stats_guard.bandwidth_saved +=
                            std::mem::size_of::<(Tensor<T>, Tensor<T>)>() as u64;
                    }
                }

                // Clean up expired entries
                {
                    let mut cache_guard = cache.write().unwrap();
                    let now = Instant::now();
                    cache_guard
                        .retain(|_, entry| now.duration_since(entry.timestamp) < config.cache_ttl);
                }

                thread::sleep(config.worker_sleep_duration);
            }
        })
    }

    /// Get data with predictive prefetching
    pub fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_accesses += 1;
        }

        // Record access pattern
        {
            let mut detector = self.pattern_detector.write().unwrap();
            detector.record_access(index);
        }

        // Check cache first
        {
            let mut cache = self.prefetch_cache.write().unwrap();
            if let Some(entry) = cache.get_mut(&index) {
                entry.access_count += 1;
                entry.timestamp = Instant::now(); // Update LRU

                let mut stats = self.stats.write().unwrap();
                stats.prefetch_hits += 1;

                return Ok(entry.data.clone());
            } else {
                let mut stats = self.stats.write().unwrap();
                stats.prefetch_misses += 1;
            }
        }

        // Predict and queue future accesses
        self.predict_and_queue_prefetch(index);

        // Get data from dataset
        self.dataset.get(index)
    }

    /// Predict future accesses and queue them for prefetching
    fn predict_and_queue_prefetch(&self, current_index: usize) {
        let predictions = {
            let detector = self.pattern_detector.read().unwrap();
            detector.predict_next(current_index, self.config.max_prefetch_count)
        };

        if !predictions.is_empty() {
            let mut queue = self.prefetch_queue.lock().unwrap();
            for predicted_index in predictions {
                // Only queue if not already cached
                let cache = self.prefetch_cache.read().unwrap();
                if !cache.contains_key(&predicted_index) {
                    queue.push_back(predicted_index);
                }
            }

            // Update pattern prediction stats
            let mut stats = self.stats.write().unwrap();
            stats.pattern_hits += 1;
        } else {
            let mut stats = self.stats.write().unwrap();
            stats.pattern_misses += 1;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> AccessStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the dominant access pattern
    pub fn dominant_pattern(&self) -> Option<AccessPattern> {
        self.pattern_detector.read().unwrap().dominant_pattern()
    }

    /// Clear the prefetch cache
    pub fn clear_cache(&self) {
        let mut cache = self.prefetch_cache.write().unwrap();
        cache.clear();
    }

    /// Get cache statistics
    pub fn cache_info(&self) -> (usize, usize) {
        let cache = self.prefetch_cache.read().unwrap();
        (cache.len(), self.config.max_cache_size)
    }
}

impl<T, D> Drop for PredictivePrefetcher<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    fn drop(&mut self) {
        // Signal shutdown and wait for worker to finish
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Dataset wrapper that provides predictive prefetching
pub struct PredictivePrefetchDataset<T, D: Dataset<T>>
where
    T: Clone + Send + Sync + 'static,
    D: Send + Sync + 'static,
{
    prefetcher: PredictivePrefetcher<T, D>,
}

impl<T, D> PredictivePrefetchDataset<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    /// Create a new predictive prefetch dataset
    pub fn new(dataset: D) -> Self {
        Self {
            prefetcher: PredictivePrefetcher::new(Arc::new(dataset)),
        }
    }

    /// Create with custom configuration
    pub fn with_config(dataset: D, config: PrefetchConfig) -> Self {
        Self {
            prefetcher: PredictivePrefetcher::with_config(Arc::new(dataset), config),
        }
    }

    /// Get access statistics
    pub fn stats(&self) -> AccessStats {
        self.prefetcher.stats()
    }

    /// Get the dominant access pattern
    pub fn dominant_pattern(&self) -> Option<AccessPattern> {
        self.prefetcher.dominant_pattern()
    }
}

impl<T, D> Dataset<T> for PredictivePrefetchDataset<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: Dataset<T> + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.prefetcher.dataset.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        self.prefetcher.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_pattern_detector_sequential() {
        let mut detector = PatternDetector::new(10);

        // Create sequential access pattern
        for i in 0..5 {
            detector.record_access(i);
        }

        let dominant = detector.dominant_pattern();
        assert!(matches!(
            dominant,
            Some(AccessPattern::Sequential { stride: 1 })
        ));
    }

    #[test]
    fn test_pattern_detector_strided() {
        let mut detector = PatternDetector::new(10);

        // Create strided access pattern (0, 2, 4, 6, 8)
        for i in 0..5 {
            detector.record_access(i * 2);
        }

        let dominant = detector.dominant_pattern();
        assert!(matches!(
            dominant,
            Some(AccessPattern::Strided {
                start: 0,
                stride: 2
            })
        ));
    }

    #[test]
    fn test_predictive_prefetcher() {
        // Create test dataset
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
        let dataset = Arc::new(TensorDataset::new(features, labels));

        let config = PrefetchConfig {
            max_prefetch_count: 2,
            max_cache_size: 10,
            pattern_history_size: 10,
            cache_ttl: Duration::from_secs(60),
            worker_sleep_duration: Duration::from_millis(1),
            bandwidth_optimization: true,
        };

        let prefetcher = PredictivePrefetcher::with_config(dataset, config);

        // Access in sequential pattern
        let _ = prefetcher.get(0).unwrap();
        let _ = prefetcher.get(1).unwrap();
        let _ = prefetcher.get(2).unwrap();

        // Give prefetcher time to work
        thread::sleep(Duration::from_millis(50));

        let stats = prefetcher.stats();
        assert!(stats.total_accesses >= 3);
    }

    #[test]
    fn test_predictive_prefetch_dataset() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let base_dataset = TensorDataset::new(features, labels);

        let dataset = PredictivePrefetchDataset::new(base_dataset);

        assert_eq!(dataset.len(), 2);

        let (feat, label) = dataset.get(0).unwrap();
        assert_eq!(feat.shape().dims(), &[2]);
        assert_eq!(label.shape().dims(), &[] as &[usize]);

        let stats = dataset.stats();
        assert_eq!(stats.total_accesses, 1);
    }
}
