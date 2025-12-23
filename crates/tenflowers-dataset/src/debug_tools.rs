//! Debug and profiling tools for data pipeline analysis
//!
//! This module provides comprehensive debugging and profiling capabilities for
//! analyzing data pipeline performance, identifying bottlenecks, and optimizing
//! data loading workflows.

use crate::Dataset;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tenflowers_core::{Result, TensorError};

/// Pipeline profiler for analyzing data loading performance
pub struct PipelineProfiler {
    /// Name of the pipeline being profiled
    name: String,
    /// Start time of profiling session
    start_time: Option<Instant>,
    /// Recorded events
    events: Vec<ProfileEvent>,
    /// Stage timings
    stage_timings: HashMap<String, Vec<Duration>>,
    /// Configuration
    config: ProfilerConfig,
}

/// Configuration for the profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable memory tracking
    pub track_memory: bool,
    /// Enable cache statistics
    pub track_cache: bool,
    /// Enable I/O statistics
    pub track_io: bool,
    /// Maximum events to store
    pub max_events: usize,
    /// Sample rate (1.0 = all events, 0.1 = 10% of events)
    pub sample_rate: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            track_memory: true,
            track_cache: true,
            track_io: true,
            max_events: 10000,
            sample_rate: 1.0,
        }
    }
}

/// A profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Event timestamp
    pub timestamp: Instant,
    /// Event type
    pub event_type: EventType,
    /// Stage name
    pub stage: String,
    /// Duration (if applicable)
    pub duration: Option<Duration>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of profiling events
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    /// Stage started
    StageStart,
    /// Stage completed
    StageEnd,
    /// Data loaded
    DataLoad,
    /// Transform applied
    Transform,
    /// Cache hit
    CacheHit,
    /// Cache miss
    CacheMiss,
    /// Memory allocation
    MemoryAlloc,
    /// I/O operation
    IoOperation,
    /// Custom event
    Custom(String),
}

impl PipelineProfiler {
    /// Create a new profiler
    pub fn new(name: impl Into<String>, config: ProfilerConfig) -> Self {
        Self {
            name: name.into(),
            start_time: None,
            events: Vec::new(),
            stage_timings: HashMap::new(),
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config(name: impl Into<String>) -> Self {
        Self::new(name, ProfilerConfig::default())
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.record_event(
            EventType::Custom("profiling_started".to_string()),
            "root",
            None,
        );
    }

    /// Stop profiling
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time {
            let duration = start.elapsed();
            self.record_event(
                EventType::Custom("profiling_stopped".to_string()),
                "root",
                Some(duration),
            );
        }
    }

    /// Record a profiling event
    pub fn record_event(
        &mut self,
        event_type: EventType,
        stage: impl Into<String>,
        duration: Option<Duration>,
    ) {
        // Apply sampling
        if self.config.sample_rate < 1.0 {
            use scirs2_core::random::rand_prelude::*;
            let mut rng = scirs2_core::random::rng();
            let sample: f64 = rng.random();
            if sample > self.config.sample_rate {
                return;
            }
        }

        if self.events.len() >= self.config.max_events {
            // Remove oldest event
            self.events.remove(0);
        }

        let event = ProfileEvent {
            timestamp: Instant::now(),
            event_type,
            stage: stage.into(),
            duration,
            metadata: HashMap::new(),
        };

        self.events.push(event);
    }

    /// Start timing a stage
    pub fn start_stage(&mut self, stage: impl Into<String>) -> StageTimer {
        let stage_name = stage.into();
        self.record_event(EventType::StageStart, &stage_name, None);
        StageTimer::new(stage_name, self.start_time.unwrap_or_else(Instant::now))
    }

    /// End timing a stage
    pub fn end_stage(&mut self, timer: StageTimer) {
        let duration = timer.elapsed();
        self.record_event(EventType::StageEnd, &timer.stage, Some(duration));

        self.stage_timings
            .entry(timer.stage.clone())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Generate profiling report
    pub fn generate_report(&self) -> ProfileReport {
        let total_duration = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::from_secs(0));

        // Aggregate stage statistics
        let mut stage_stats = HashMap::new();
        for (stage, durations) in &self.stage_timings {
            let stats = StageStatistics::from_durations(durations);
            stage_stats.insert(stage.clone(), stats);
        }

        // Count event types
        let mut event_counts = HashMap::new();
        for event in &self.events {
            let event_name = format!("{:?}", event.event_type);
            *event_counts.entry(event_name).or_insert(0) += 1;
        }

        // Calculate cache statistics
        let cache_hits = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheHit)
            .count();
        let cache_misses = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheMiss)
            .count();
        let cache_hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64
        } else {
            0.0
        };

        ProfileReport {
            pipeline_name: self.name.clone(),
            total_duration,
            total_events: self.events.len(),
            stage_stats,
            event_counts,
            cache_hit_rate,
            bottlenecks: self.identify_bottlenecks(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Find slow stages
        for (stage, durations) in &self.stage_timings {
            if durations.is_empty() {
                continue;
            }

            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;

            // Flag stages taking >100ms on average
            if avg_duration.as_millis() > 100 {
                bottlenecks.push(Bottleneck {
                    category: BottleneckCategory::SlowStage,
                    description: format!("Stage '{}' is slow (avg: {:?})", stage, avg_duration),
                    severity: if avg_duration.as_millis() > 1000 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                    affected_component: stage.clone(),
                });
            }
        }

        // Check cache hit rate
        let cache_hits = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheHit)
            .count();
        let cache_misses = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheMiss)
            .count();

        if cache_hits + cache_misses > 0 {
            let hit_rate = cache_hits as f64 / (cache_hits + cache_misses) as f64;
            if hit_rate < 0.5 {
                bottlenecks.push(Bottleneck {
                    category: BottleneckCategory::LowCacheHitRate,
                    description: format!("Low cache hit rate: {:.1}%", hit_rate * 100.0),
                    severity: Severity::Medium,
                    affected_component: "cache".to_string(),
                });
            }
        }

        bottlenecks
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check cache configuration
        let cache_hits = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheHit)
            .count();
        let cache_misses = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::CacheMiss)
            .count();

        if cache_hits + cache_misses > 0 {
            let hit_rate = cache_hits as f64 / (cache_hits + cache_misses) as f64;
            if hit_rate < 0.7 {
                recommendations.push(
                    "Consider increasing cache size or using predictive prefetching".to_string(),
                );
            }
        }

        // Check for slow stages
        for (stage, durations) in &self.stage_timings {
            if durations.is_empty() {
                continue;
            }

            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            if avg_duration.as_millis() > 500 {
                recommendations.push(format!(
                    "Optimize '{}' stage - consider parallelization or GPU acceleration",
                    stage
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Pipeline is well optimized".to_string());
        }

        recommendations
    }

    /// Export events to JSON-compatible format
    pub fn export_events(&self) -> Vec<HashMap<String, String>> {
        self.events
            .iter()
            .map(|event| {
                let mut map = HashMap::new();
                map.insert("stage".to_string(), event.stage.clone());
                map.insert("type".to_string(), format!("{:?}", event.event_type));
                if let Some(duration) = event.duration {
                    map.insert("duration_ms".to_string(), duration.as_millis().to_string());
                }
                map
            })
            .collect()
    }
}

/// Timer for measuring stage duration
pub struct StageTimer {
    stage: String,
    start: Instant,
}

impl StageTimer {
    fn new(stage: String, start: Instant) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }

    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Statistics for a pipeline stage
#[derive(Debug, Clone)]
pub struct StageStatistics {
    /// Number of executions
    pub count: usize,
    /// Total time spent
    pub total_duration: Duration,
    /// Average duration
    pub avg_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Standard deviation
    pub std_dev: Duration,
}

impl StageStatistics {
    fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self {
                count: 0,
                total_duration: Duration::from_secs(0),
                avg_duration: Duration::from_secs(0),
                min_duration: Duration::from_secs(0),
                max_duration: Duration::from_secs(0),
                std_dev: Duration::from_secs(0),
            };
        }

        let total: Duration = durations.iter().sum();
        let avg = total / durations.len() as u32;
        let min = *durations.iter().min().unwrap();
        let max = *durations.iter().max().unwrap();

        // Calculate standard deviation
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Self {
            count: durations.len(),
            total_duration: total,
            avg_duration: avg,
            min_duration: min,
            max_duration: max,
            std_dev,
        }
    }
}

/// Profiling report
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Pipeline name
    pub pipeline_name: String,
    /// Total duration
    pub total_duration: Duration,
    /// Total events recorded
    pub total_events: usize,
    /// Stage statistics
    pub stage_stats: HashMap<String, StageStatistics>,
    /// Event counts by type
    pub event_counts: HashMap<String, usize>,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl ProfileReport {
    /// Generate human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Pipeline Profiling Report: {}\n",
            self.pipeline_name
        ));
        report.push_str("=".repeat(60).as_str());
        report.push('\n');

        report.push_str(&format!("Total Duration: {:?}\n", self.total_duration));
        report.push_str(&format!("Total Events: {}\n", self.total_events));
        report.push_str(&format!(
            "Cache Hit Rate: {:.1}%\n\n",
            self.cache_hit_rate * 100.0
        ));

        // Stage statistics
        if !self.stage_stats.is_empty() {
            report.push_str("Stage Statistics:\n");
            report.push_str("-".repeat(60).as_str());
            report.push('\n');

            let mut stages: Vec<_> = self.stage_stats.iter().collect();
            stages.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

            for (stage, stats) in stages {
                report.push_str(&format!(
                    "  {}: {} calls, avg {:?}, total {:?}\n",
                    stage, stats.count, stats.avg_duration, stats.total_duration
                ));
            }
            report.push('\n');
        }

        // Bottlenecks
        if !self.bottlenecks.is_empty() {
            report.push_str("Identified Bottlenecks:\n");
            report.push_str("-".repeat(60).as_str());
            report.push('\n');

            for bottleneck in &self.bottlenecks {
                report.push_str(&format!(
                    "  [{:?}] {}\n",
                    bottleneck.severity, bottleneck.description
                ));
            }
            report.push('\n');
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            report.push_str("Recommendations:\n");
            report.push_str("-".repeat(60).as_str());
            report.push('\n');

            for (i, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        report
    }
}

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Bottleneck category
    pub category: BottleneckCategory,
    /// Description
    pub description: String,
    /// Severity
    pub severity: Severity,
    /// Affected component
    pub affected_component: String,
}

/// Bottleneck categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckCategory {
    /// Slow stage execution
    SlowStage,
    /// High memory usage
    HighMemoryUsage,
    /// Low cache hit rate
    LowCacheHitRate,
    /// Slow I/O
    SlowIo,
    /// Inefficient transform
    InefficientTransform,
}

/// Severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Dataset debugger for inspecting dataset contents
pub struct DatasetDebugger;

impl DatasetDebugger {
    /// Inspect dataset samples
    pub fn inspect_samples<T>(
        dataset: &dyn Dataset<T>,
        num_samples: usize,
    ) -> Result<Vec<SampleInfo>>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        let mut samples = Vec::new();
        let count = num_samples.min(dataset.len());

        for i in 0..count {
            if let Ok((features, labels)) = dataset.get(i) {
                samples.push(SampleInfo {
                    index: i,
                    feature_shape: features.shape().dims().to_vec(),
                    label_shape: labels.shape().dims().to_vec(),
                    feature_size: features.size(),
                    label_size: labels.size(),
                });
            }
        }

        Ok(samples)
    }

    /// Verify dataset consistency
    pub fn verify_consistency<T>(dataset: &dyn Dataset<T>) -> Result<ConsistencyReport>
    where
        T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
    {
        let mut issues = Vec::new();
        let samples_to_check = dataset.len().min(100);

        if samples_to_check == 0 {
            return Ok(ConsistencyReport {
                total_samples: 0,
                checked_samples: 0,
                issues,
                is_consistent: true,
            });
        }

        // Check first sample to establish expected shapes
        let (first_features, first_labels) = dataset.get(0)?;
        let expected_feature_shape = first_features.shape().dims().to_vec();
        let expected_label_shape = first_labels.shape().dims().to_vec();

        for i in 1..samples_to_check {
            if let Ok((features, labels)) = dataset.get(i) {
                if features.shape().dims() != expected_feature_shape.as_slice() {
                    issues.push(format!(
                        "Sample {}: Inconsistent feature shape {:?}, expected {:?}",
                        i,
                        features.shape().dims(),
                        expected_feature_shape
                    ));
                }
                if labels.shape().dims() != expected_label_shape.as_slice() {
                    issues.push(format!(
                        "Sample {}: Inconsistent label shape {:?}, expected {:?}",
                        i,
                        labels.shape().dims(),
                        expected_label_shape
                    ));
                }
            } else {
                issues.push(format!("Sample {}: Failed to load", i));
            }
        }

        let is_consistent = issues.is_empty();
        Ok(ConsistencyReport {
            total_samples: dataset.len(),
            checked_samples: samples_to_check,
            issues,
            is_consistent,
        })
    }
}

/// Information about a dataset sample
#[derive(Debug, Clone)]
pub struct SampleInfo {
    /// Sample index
    pub index: usize,
    /// Feature tensor shape
    pub feature_shape: Vec<usize>,
    /// Label tensor shape
    pub label_shape: Vec<usize>,
    /// Feature size (total elements)
    pub feature_size: usize,
    /// Label size (total elements)
    pub label_size: usize,
}

/// Consistency check report
#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    /// Total samples in dataset
    pub total_samples: usize,
    /// Number of samples checked
    pub checked_samples: usize,
    /// Issues found
    pub issues: Vec<String>,
    /// Whether dataset is consistent
    pub is_consistent: bool,
}

impl ConsistencyReport {
    /// Generate report string
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("Dataset Consistency Report\n");
        report.push_str("=".repeat(60).as_str());
        report.push('\n');

        report.push_str(&format!("Total Samples: {}\n", self.total_samples));
        report.push_str(&format!("Checked Samples: {}\n", self.checked_samples));
        report.push_str(&format!("Is Consistent: {}\n\n", self.is_consistent));

        if !self.issues.is_empty() {
            report.push_str(&format!("Issues Found ({}):\n", self.issues.len()));
            for (i, issue) in self.issues.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, issue));
            }
        } else {
            report.push_str("No issues found.\n");
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_profiler_creation() {
        let profiler = PipelineProfiler::default_config("test_pipeline");
        assert_eq!(profiler.name, "test_pipeline");
    }

    #[test]
    fn test_profiler_events() {
        let mut profiler = PipelineProfiler::default_config("test");
        profiler.start();

        profiler.record_event(EventType::DataLoad, "load_stage", None);
        profiler.record_event(
            EventType::Transform,
            "transform_stage",
            Some(Duration::from_millis(10)),
        );

        profiler.stop();

        let report = profiler.generate_report();
        assert!(report.total_events > 0);
    }

    #[test]
    fn test_stage_timing() {
        let mut profiler = PipelineProfiler::default_config("test");
        profiler.start();

        let timer = profiler.start_stage("test_stage");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_stage(timer);

        let report = profiler.generate_report();
        assert!(report.stage_stats.contains_key("test_stage"));
    }

    #[test]
    fn test_dataset_debugger_inspect() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let samples = DatasetDebugger::inspect_samples(&dataset, 5).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].feature_shape, vec![2]);
    }

    #[test]
    fn test_consistency_check() {
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0], &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let report = DatasetDebugger::verify_consistency(&dataset).unwrap();
        assert!(report.is_consistent);
        assert_eq!(report.total_samples, 2);
    }

    #[test]
    fn test_profile_report_generation() {
        let mut profiler = PipelineProfiler::default_config("test");
        profiler.start();

        let timer = profiler.start_stage("data_loading");
        std::thread::sleep(Duration::from_millis(5));
        profiler.end_stage(timer);

        profiler.stop();

        let report = profiler.generate_report();
        let report_string = report.format_report();

        assert!(report_string.contains("Pipeline Profiling Report"));
        assert!(report_string.contains("data_loading"));
    }
}
