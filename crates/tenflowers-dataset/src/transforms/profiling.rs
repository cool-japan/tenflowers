//! Transform profiling and performance monitoring
//!
//! This module provides utilities for profiling and monitoring the performance
//! of dataset transformations, including timing, memory usage, and throughput metrics.

use crate::transforms::Transform;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tenflowers_core::{Result, Tensor, TensorError};

/// Statistics collected during transform execution
#[derive(Debug, Clone)]
pub struct TransformStats {
    /// Total number of samples processed
    pub sample_count: usize,
    /// Total time spent in transform
    pub total_duration: Duration,
    /// Average time per sample
    pub avg_duration_per_sample: Duration,
    /// Minimum time per sample
    pub min_duration: Duration,
    /// Maximum time per sample
    pub max_duration: Duration,
    /// Memory usage statistics (in bytes)
    pub memory_stats: MemoryStats,
    /// Error count
    pub error_count: usize,
    /// Throughput (samples per second)
    pub throughput: f64,
}

impl Default for TransformStats {
    fn default() -> Self {
        Self {
            sample_count: 0,
            total_duration: Duration::ZERO,
            avg_duration_per_sample: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            memory_stats: MemoryStats::default(),
            error_count: 0,
            throughput: 0.0,
        }
    }
}

impl TransformStats {
    /// Update statistics with a new sample timing
    pub fn update(&mut self, duration: Duration, memory_usage: Option<usize>, had_error: bool) {
        self.sample_count += 1;
        self.total_duration += duration;

        if had_error {
            self.error_count += 1;
        }

        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }

        self.avg_duration_per_sample = self.total_duration / self.sample_count as u32;

        if let Some(memory) = memory_usage {
            self.memory_stats.update(memory);
        }

        // Calculate throughput (samples per second)
        if !self.total_duration.is_zero() {
            self.throughput = self.sample_count as f64 / self.total_duration.as_secs_f64();
        }
    }

    /// Get efficiency score (0.0 to 1.0, higher is better)
    pub fn efficiency_score(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }

        let success_rate = (self.sample_count - self.error_count) as f64 / self.sample_count as f64;
        let throughput_score = (self.throughput / 1000.0).min(1.0); // Normalize to [0, 1]

        (success_rate + throughput_score) / 2.0
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Average memory usage (bytes)
    pub avg_usage: usize,
    /// Total measurements taken
    pub measurement_count: usize,
    /// Running sum for average calculation
    pub total_usage: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            peak_usage: 0,
            avg_usage: 0,
            measurement_count: 0,
            total_usage: 0,
        }
    }
}

impl MemoryStats {
    pub fn update(&mut self, usage: usize) {
        self.measurement_count += 1;
        self.total_usage += usage;
        self.avg_usage = self.total_usage / self.measurement_count;

        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Format memory usage in human-readable format
    pub fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Transform profiler for collecting detailed performance metrics
pub struct TransformProfiler {
    stats: Arc<Mutex<HashMap<String, TransformStats>>>,
    enabled: bool,
    detailed_logging: bool,
}

impl Default for TransformProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformProfiler {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(HashMap::new())),
            enabled: true,
            detailed_logging: false,
        }
    }

    pub fn with_detailed_logging(mut self) -> Self {
        self.detailed_logging = true;
        self
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Record a transform execution
    pub fn record_execution(
        &self,
        transform_name: &str,
        duration: Duration,
        memory_usage: Option<usize>,
        had_error: bool,
    ) {
        if !self.enabled {
            return;
        }

        if let Ok(mut stats_map) = self.stats.lock() {
            let entry = stats_map
                .entry(transform_name.to_string())
                .or_insert_with(TransformStats::default);

            entry.update(duration, memory_usage, had_error);

            if self.detailed_logging && had_error {
                eprintln!(
                    "[PROFILER] Error in transform '{}' after {} samples",
                    transform_name, entry.sample_count
                );
            }
        }
    }

    /// Get statistics for a specific transform
    pub fn get_stats(&self, transform_name: &str) -> Option<TransformStats> {
        if let Ok(stats_map) = self.stats.lock() {
            stats_map.get(transform_name).cloned()
        } else {
            None
        }
    }

    /// Get all collected statistics
    pub fn get_all_stats(&self) -> HashMap<String, TransformStats> {
        if let Ok(stats_map) = self.stats.lock() {
            stats_map.clone()
        } else {
            HashMap::new()
        }
    }

    /// Generate a performance report
    pub fn generate_report(&self) -> String {
        let stats = self.get_all_stats();
        let mut report = String::from("Transform Performance Report\n");
        report.push_str(&"=".repeat(50));
        report.push('\n');

        if stats.is_empty() {
            report.push_str("No profiling data collected.\n");
            return report;
        }

        // Sort by throughput (descending)
        let mut sorted_stats: Vec<_> = stats.iter().collect();
        sorted_stats.sort_by(|a, b| b.1.throughput.partial_cmp(&a.1.throughput).unwrap());

        for (name, stat) in sorted_stats {
            report.push_str(&format!("\n{}: \n", name));
            report.push_str(&format!(
                "  Samples: {} (Errors: {})\n",
                stat.sample_count, stat.error_count
            ));
            report.push_str(&format!(
                "  Throughput: {:.2} samples/sec\n",
                stat.throughput
            ));
            report.push_str(&format!(
                "  Avg Time: {:.2}ms\n",
                stat.avg_duration_per_sample.as_millis()
            ));
            report.push_str(&format!(
                "  Time Range: {:.2}ms - {:.2}ms\n",
                stat.min_duration.as_millis(),
                stat.max_duration.as_millis()
            ));
            report.push_str(&format!(
                "  Memory: Peak {}, Avg {}\n",
                MemoryStats::format_bytes(stat.memory_stats.peak_usage),
                MemoryStats::format_bytes(stat.memory_stats.avg_usage)
            ));
            report.push_str(&format!(
                "  Efficiency Score: {:.2}%\n",
                stat.efficiency_score() * 100.0
            ));
        }

        report
    }

    /// Clear all collected statistics
    pub fn clear(&self) {
        if let Ok(mut stats_map) = self.stats.lock() {
            stats_map.clear();
        }
    }
}

/// A wrapper that profiles any transform
pub struct ProfiledTransform<T, Tr: Transform<T>> {
    inner_transform: Tr,
    profiler: Arc<TransformProfiler>,
    transform_name: String,
    _phantom: PhantomData<T>,
}

impl<T, Tr: Transform<T>> ProfiledTransform<T, Tr> {
    pub fn new(transform: Tr, profiler: Arc<TransformProfiler>, name: String) -> Self {
        Self {
            inner_transform: transform,
            profiler,
            transform_name: name,
            _phantom: PhantomData,
        }
    }

    /// Create a profiled transform with automatic naming
    pub fn wrap(transform: Tr, profiler: Arc<TransformProfiler>) -> Self {
        let name = std::any::type_name::<Tr>()
            .split("::")
            .last()
            .unwrap_or("UnknownTransform")
            .to_string();

        Self::new(transform, profiler, name)
    }

    /// Get reference to the inner transform
    pub fn inner(&self) -> &Tr {
        &self.inner_transform
    }

    /// Get mutable reference to the inner transform
    pub fn inner_mut(&mut self) -> &mut Tr {
        &mut self.inner_transform
    }

    /// Get the profiler
    pub fn profiler(&self) -> &Arc<TransformProfiler> {
        &self.profiler
    }

    /// Estimate memory usage of tensors
    fn estimate_memory_usage(&self, features: &Tensor<T>, labels: &Tensor<T>) -> usize
    where
        T: 'static,
    {
        let feature_size = features.shape().size() * std::mem::size_of::<T>();
        let label_size = labels.shape().size() * std::mem::size_of::<T>();
        feature_size + label_size
    }
}

impl<T, Tr: Transform<T>> Transform<T> for ProfiledTransform<T, Tr>
where
    T: 'static,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let start_time = Instant::now();
        let memory_before = self.estimate_memory_usage(&sample.0, &sample.1);

        let result = self.inner_transform.apply(sample);
        let duration = start_time.elapsed();

        let had_error = result.is_err();
        let memory_after = if let Ok(ref output) = result {
            self.estimate_memory_usage(&output.0, &output.1)
        } else {
            memory_before
        };

        let peak_memory = memory_before.max(memory_after);

        self.profiler.record_execution(
            &self.transform_name,
            duration,
            Some(peak_memory),
            had_error,
        );

        result
    }
}

/// Performance benchmark for transforms
pub struct TransformBenchmark<T> {
    profiler: Arc<TransformProfiler>,
    sample_count: usize,
    warmup_count: usize,
    _phantom: PhantomData<T>,
}

impl<T> TransformBenchmark<T> {
    pub fn new(sample_count: usize) -> Self {
        Self {
            profiler: Arc::new(TransformProfiler::new()),
            sample_count,
            warmup_count: 10,
            _phantom: PhantomData,
        }
    }

    pub fn with_warmup(mut self, warmup_count: usize) -> Self {
        self.warmup_count = warmup_count;
        self
    }

    /// Benchmark a transform with synthetic data
    pub fn benchmark_transform<Tr: Transform<T>>(
        &self,
        mut transform: Tr,
        sample_generator: impl Fn() -> Result<(Tensor<T>, Tensor<T>)>,
        transform_name: &str,
    ) -> Result<TransformStats>
    where
        T: 'static,
        Tr: 'static,
    {
        let profiled_transform =
            ProfiledTransform::new(transform, self.profiler.clone(), transform_name.to_string());

        // Warmup phase
        for _ in 0..self.warmup_count {
            let sample = sample_generator()?;
            let _ = profiled_transform.apply(sample);
        }

        // Clear warmup stats
        self.profiler.clear();

        // Benchmark phase
        for _ in 0..self.sample_count {
            let sample = sample_generator()?;
            let _ = profiled_transform.apply(sample);
        }

        self.profiler
            .get_stats(transform_name)
            .ok_or_else(|| TensorError::invalid_argument("Failed to collect stats".to_string()))
    }

    /// Compare multiple transforms
    pub fn compare_transforms<Tr1, Tr2>(
        &self,
        transform1: Tr1,
        transform2: Tr2,
        sample_generator: impl Fn() -> Result<(Tensor<T>, Tensor<T>)>,
        name1: &str,
        name2: &str,
    ) -> Result<(TransformStats, TransformStats)>
    where
        T: Clone + 'static,
        Tr1: Transform<T> + 'static,
        Tr2: Transform<T> + 'static,
    {
        let stats1 = self.benchmark_transform(transform1, &sample_generator, name1)?;
        self.profiler.clear();
        let stats2 = self.benchmark_transform(transform2, &sample_generator, name2)?;

        Ok((stats1, stats2))
    }

    /// Get the profiler for detailed analysis
    pub fn profiler(&self) -> &Arc<TransformProfiler> {
        &self.profiler
    }
}

/// Utility for creating sample data generators
pub struct SampleGenerator;

impl SampleGenerator {
    /// Generate random tensor samples
    pub fn random_tensors<T>(
        feature_shape: &[usize],
        label_shape: &[usize],
    ) -> impl Fn() -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        let f_shape = feature_shape.to_vec();
        let l_shape = label_shape.to_vec();

        move || {
            let features = Tensor::zeros(&f_shape);
            let labels = Tensor::zeros(&l_shape);
            Ok((features, labels))
        }
    }

    /// Generate samples with specific size for memory testing
    pub fn sized_samples<T>(total_elements: usize) -> impl Fn() -> Result<(Tensor<T>, Tensor<T>)>
    where
        T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        move || {
            let feature_size = total_elements / 2;
            let label_size = total_elements - feature_size;

            let features = Tensor::zeros(&[feature_size]);
            let labels = Tensor::zeros(&[label_size]);
            Ok((features, labels))
        }
    }
}

/// Macro for easy transform profiling
#[macro_export]
macro_rules! profile_transform {
    ($transform:expr, $profiler:expr) => {
        ProfiledTransform::wrap($transform, $profiler)
    };
    ($transform:expr, $profiler:expr, $name:expr) => {
        ProfiledTransform::new($transform, $profiler, $name.to_string())
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::noise::AddNoise;

    #[test]
    fn test_transform_stats() {
        let mut stats = TransformStats::default();

        stats.update(Duration::from_millis(100), Some(1024), false);
        stats.update(Duration::from_millis(200), Some(2048), false);
        stats.update(Duration::from_millis(150), Some(1536), true);

        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.min_duration, Duration::from_millis(100));
        assert_eq!(stats.max_duration, Duration::from_millis(200));
        assert!(stats.efficiency_score() > 0.0 && stats.efficiency_score() <= 1.0);
    }

    #[test]
    fn test_profiler() {
        let profiler = TransformProfiler::new();

        profiler.record_execution(
            "test_transform",
            Duration::from_millis(50),
            Some(512),
            false,
        );

        let stats = profiler.get_stats("test_transform").unwrap();
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.error_count, 0);
    }

    #[test]
    fn test_profiled_transform() {
        let profiler = Arc::new(TransformProfiler::new());
        let transform = AddNoise::new(0.1f32);
        let profiled = ProfiledTransform::wrap(transform, profiler.clone());

        let features = Tensor::<f32>::zeros(&[10]);
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = profiled.apply((features, labels));
        assert!(result.is_ok());

        // Get all stats to see what names are actually recorded
        let all_stats = profiler.get_all_stats();
        assert!(!all_stats.is_empty(), "No stats were recorded");

        // Get the first (and should be only) stats entry
        let stats = all_stats.values().next().unwrap();
        assert_eq!(stats.sample_count, 1);
    }
}
