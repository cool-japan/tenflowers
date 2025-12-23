//! Cache telemetry and metrics collection system
//!
//! This module provides comprehensive metrics collection, monitoring,
//! and analysis capabilities for cache operations. It tracks performance
//! metrics, identifies patterns, and helps optimize cache effectiveness.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Comprehensive cache telemetry metrics
#[derive(Debug, Clone)]
pub struct CacheTelemetryMetrics {
    /// Basic hit/miss counters
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub insertions: u64,

    /// Latency metrics (in microseconds)
    pub avg_hit_latency_us: f64,
    pub avg_miss_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,

    /// Memory metrics
    pub current_size_bytes: usize,
    pub peak_size_bytes: usize,
    pub total_allocated_bytes: u64,
    pub total_freed_bytes: u64,

    /// Throughput metrics
    pub requests_per_second: f64,
    pub bytes_per_second: f64,

    /// Time window for metrics
    pub window_start: Instant,
    pub window_duration: Duration,

    /// Cache effectiveness
    pub hit_ratio: f64,
    pub byte_hit_ratio: f64,
    pub eviction_rate: f64,
}

impl CacheTelemetryMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            insertions: 0,
            avg_hit_latency_us: 0.0,
            avg_miss_latency_us: 0.0,
            p50_latency_us: 0.0,
            p95_latency_us: 0.0,
            p99_latency_us: 0.0,
            current_size_bytes: 0,
            peak_size_bytes: 0,
            total_allocated_bytes: 0,
            total_freed_bytes: 0,
            requests_per_second: 0.0,
            bytes_per_second: 0.0,
            window_start: Instant::now(),
            window_duration: Duration::from_secs(0),
            hit_ratio: 0.0,
            byte_hit_ratio: 0.0,
            eviction_rate: 0.0,
        }
    }

    /// Calculate derived metrics
    pub fn calculate_derived(&mut self) {
        let total_requests = self.hits + self.misses;
        self.hit_ratio = if total_requests > 0 {
            self.hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let total_bytes = self.total_allocated_bytes;
        let hit_bytes =
            (self.hits as f64 / total_requests.max(1) as f64 * total_bytes as f64) as u64;
        self.byte_hit_ratio = if total_bytes > 0 {
            hit_bytes as f64 / total_bytes as f64
        } else {
            0.0
        };

        let duration_secs = self.window_duration.as_secs_f64();
        if duration_secs > 0.0 {
            self.requests_per_second = total_requests as f64 / duration_secs;
            self.bytes_per_second = total_bytes as f64 / duration_secs;
            self.eviction_rate = self.evictions as f64 / duration_secs;
        }
    }
}

impl Default for CacheTelemetryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual cache operation event
#[derive(Debug, Clone)]
pub struct CacheEvent {
    /// Type of cache operation
    pub event_type: CacheEventType,
    /// Timestamp of the event
    pub timestamp: Instant,
    /// Latency of the operation
    pub latency: Duration,
    /// Size of data involved (bytes)
    pub size_bytes: Option<usize>,
    /// Cache key identifier
    pub key_hash: u64,
}

/// Types of cache events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEventType {
    /// Cache hit
    Hit,
    /// Cache miss
    Miss,
    /// Item eviction
    Eviction,
    /// Item insertion
    Insertion,
    /// Cache clear
    Clear,
}

/// Time-series data point for tracking metrics over time
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Instant,
    /// Metrics at this point in time
    pub metrics: CacheTelemetryMetrics,
}

/// Cache telemetry collector with historical tracking
pub struct CacheTelemetryCollector {
    /// Current active metrics
    current_metrics: Arc<Mutex<CacheTelemetryMetrics>>,
    /// Recent events buffer (limited size)
    recent_events: Arc<Mutex<VecDeque<CacheEvent>>>,
    /// Historical snapshots for time-series analysis
    snapshots: Arc<Mutex<VecDeque<MetricsSnapshot>>>,
    /// Latency histogram buckets (microseconds -> count)
    latency_histogram: Arc<Mutex<HashMap<u64, u64>>>,
    /// Configuration
    config: TelemetryConfig,
}

/// Configuration for telemetry collection
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Maximum number of events to keep in buffer
    pub max_events: usize,
    /// Maximum number of snapshots to keep
    pub max_snapshots: usize,
    /// Snapshot interval
    pub snapshot_interval: Duration,
    /// Enable detailed latency tracking
    pub track_latency_histogram: bool,
    /// Enable per-key statistics
    pub track_per_key_stats: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            max_events: 10000,
            max_snapshots: 1000,
            snapshot_interval: Duration::from_secs(60),
            track_latency_histogram: true,
            track_per_key_stats: false,
        }
    }
}

impl CacheTelemetryCollector {
    /// Create a new telemetry collector
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            current_metrics: Arc::new(Mutex::new(CacheTelemetryMetrics::new())),
            recent_events: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_events))),
            snapshots: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_snapshots))),
            latency_histogram: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Record a cache hit
    pub fn record_hit(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.hits += 1;

        // Update running average for hit latency
        let latency_us = latency.as_micros() as f64;
        let total_hits = metrics.hits as f64;
        metrics.avg_hit_latency_us =
            (metrics.avg_hit_latency_us * (total_hits - 1.0) + latency_us) / total_hits;

        if let Some(size) = size_bytes {
            metrics.current_size_bytes = metrics.current_size_bytes.saturating_add(size);
        }
        drop(metrics);

        if self.config.track_latency_histogram {
            self.record_latency(latency);
        }

        self.record_event(CacheEvent {
            event_type: CacheEventType::Hit,
            timestamp: Instant::now(),
            latency,
            size_bytes,
            key_hash,
        });
    }

    /// Record a cache miss
    pub fn record_miss(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.misses += 1;

        // Update running average for miss latency
        let latency_us = latency.as_micros() as f64;
        let total_misses = metrics.misses as f64;
        metrics.avg_miss_latency_us =
            (metrics.avg_miss_latency_us * (total_misses - 1.0) + latency_us) / total_misses;
        drop(metrics);

        if self.config.track_latency_histogram {
            self.record_latency(latency);
        }

        self.record_event(CacheEvent {
            event_type: CacheEventType::Miss,
            timestamp: Instant::now(),
            latency,
            size_bytes,
            key_hash,
        });
    }

    /// Record an eviction
    pub fn record_eviction(&self, size_bytes: Option<usize>, key_hash: u64) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.evictions += 1;

        if let Some(size) = size_bytes {
            metrics.current_size_bytes = metrics.current_size_bytes.saturating_sub(size);
            metrics.total_freed_bytes += size as u64;
        }
        drop(metrics);

        self.record_event(CacheEvent {
            event_type: CacheEventType::Eviction,
            timestamp: Instant::now(),
            latency: Duration::from_micros(0),
            size_bytes,
            key_hash,
        });
    }

    /// Record an insertion
    pub fn record_insertion(&self, size_bytes: Option<usize>, key_hash: u64) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.insertions += 1;

        if let Some(size) = size_bytes {
            metrics.current_size_bytes = metrics.current_size_bytes.saturating_add(size);
            metrics.peak_size_bytes = metrics.peak_size_bytes.max(metrics.current_size_bytes);
            metrics.total_allocated_bytes += size as u64;
        }
        drop(metrics);

        self.record_event(CacheEvent {
            event_type: CacheEventType::Insertion,
            timestamp: Instant::now(),
            latency: Duration::from_micros(0),
            size_bytes,
            key_hash,
        });
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self) -> CacheTelemetryMetrics {
        let mut metrics = self.current_metrics.lock().unwrap().clone();
        metrics.window_duration = metrics.window_start.elapsed();
        metrics.calculate_derived();

        // Calculate percentiles from histogram
        if self.config.track_latency_histogram {
            let histogram = self.latency_histogram.lock().unwrap();
            let percentiles = calculate_percentiles(&histogram);
            metrics.p50_latency_us = percentiles.0;
            metrics.p95_latency_us = percentiles.1;
            metrics.p99_latency_us = percentiles.2;
        }

        metrics
    }

    /// Take a snapshot of current metrics
    pub fn snapshot(&self) {
        let snapshot = MetricsSnapshot {
            timestamp: Instant::now(),
            metrics: self.get_metrics(),
        };

        let mut snapshots = self.snapshots.lock().unwrap();
        snapshots.push_back(snapshot);

        // Maintain max size
        while snapshots.len() > self.config.max_snapshots {
            snapshots.pop_front();
        }
    }

    /// Get recent events
    pub fn get_recent_events(&self, count: usize) -> Vec<CacheEvent> {
        let events = self.recent_events.lock().unwrap();
        events.iter().rev().take(count).cloned().collect()
    }

    /// Get historical snapshots
    pub fn get_snapshots(&self) -> Vec<MetricsSnapshot> {
        self.snapshots.lock().unwrap().iter().cloned().collect()
    }

    /// Reset all metrics
    pub fn reset(&self) {
        *self.current_metrics.lock().unwrap() = CacheTelemetryMetrics::new();
        self.recent_events.lock().unwrap().clear();
        self.snapshots.lock().unwrap().clear();
        self.latency_histogram.lock().unwrap().clear();
    }

    /// Generate a human-readable report
    pub fn generate_report(&self) -> String {
        let metrics = self.get_metrics();
        let mut report = String::new();

        report.push_str("=== Cache Telemetry Report ===\n\n");

        report.push_str("## Request Statistics\n");
        report.push_str(&format!(
            "  Total Requests: {}\n",
            metrics.hits + metrics.misses
        ));
        report.push_str(&format!(
            "  Hits: {} ({:.2}%)\n",
            metrics.hits,
            metrics.hit_ratio * 100.0
        ));
        report.push_str(&format!("  Misses: {}\n", metrics.misses));
        report.push_str(&format!("  Evictions: {}\n", metrics.evictions));
        report.push_str(&format!("  Insertions: {}\n\n", metrics.insertions));

        report.push_str("## Latency Statistics (microseconds)\n");
        report.push_str(&format!(
            "  Avg Hit Latency: {:.2}\n",
            metrics.avg_hit_latency_us
        ));
        report.push_str(&format!(
            "  Avg Miss Latency: {:.2}\n",
            metrics.avg_miss_latency_us
        ));
        report.push_str(&format!("  P50: {:.2}\n", metrics.p50_latency_us));
        report.push_str(&format!("  P95: {:.2}\n", metrics.p95_latency_us));
        report.push_str(&format!("  P99: {:.2}\n\n", metrics.p99_latency_us));

        report.push_str("## Memory Statistics\n");
        report.push_str(&format!(
            "  Current Size: {} bytes\n",
            metrics.current_size_bytes
        ));
        report.push_str(&format!("  Peak Size: {} bytes\n", metrics.peak_size_bytes));
        report.push_str(&format!(
            "  Total Allocated: {} bytes\n",
            metrics.total_allocated_bytes
        ));
        report.push_str(&format!(
            "  Total Freed: {} bytes\n\n",
            metrics.total_freed_bytes
        ));

        report.push_str("## Throughput Statistics\n");
        report.push_str(&format!(
            "  Requests/sec: {:.2}\n",
            metrics.requests_per_second
        ));
        report.push_str(&format!("  Bytes/sec: {:.2}\n", metrics.bytes_per_second));
        report.push_str(&format!(
            "  Eviction Rate: {:.2}/sec\n",
            metrics.eviction_rate
        ));

        report
    }

    // Private helper methods
    fn record_event(&self, event: CacheEvent) {
        let mut events = self.recent_events.lock().unwrap();
        events.push_back(event);

        // Maintain max size
        while events.len() > self.config.max_events {
            events.pop_front();
        }
    }

    fn record_latency(&self, latency: Duration) {
        let bucket = (latency.as_micros() as u64 / 100) * 100; // 100us buckets
        let mut histogram = self.latency_histogram.lock().unwrap();
        *histogram.entry(bucket).or_insert(0) += 1;
    }
}

/// Calculate percentiles from histogram
fn calculate_percentiles(histogram: &HashMap<u64, u64>) -> (f64, f64, f64) {
    if histogram.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let total_count: u64 = histogram.values().sum();
    let mut sorted_buckets: Vec<_> = histogram.iter().collect();
    sorted_buckets.sort_by_key(|(bucket, _)| *bucket);

    let find_percentile = |target_pct: f64| -> f64 {
        let target_count = (total_count as f64 * target_pct) as u64;
        let mut cumulative = 0u64;

        for (bucket, count) in &sorted_buckets {
            cumulative += *count;
            if cumulative >= target_count {
                return **bucket as f64;
            }
        }

        sorted_buckets
            .last()
            .map(|(b, _)| **b as f64)
            .unwrap_or(0.0)
    };

    (
        find_percentile(0.50),
        find_percentile(0.95),
        find_percentile(0.99),
    )
}

// ==== Enhanced Telemetry Features ====

use std::time::SystemTime;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Alert types for cache performance issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum AlertType {
    LowHitRate,
    HighLatency,
    HighEvictionRate,
    MemoryPressure,
    AnomalyDetected,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Performance alert
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub description: String,
    pub metric_value: f64,
    pub threshold_value: f64,
    pub timestamp: SystemTime,
}

/// Alert configuration thresholds
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AlertThresholds {
    pub min_hit_rate: f64,
    pub max_latency_us: f64,
    pub max_eviction_rate: f64,
    pub max_memory_overhead: f64,
    pub anomaly_stddev_multiplier: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            min_hit_rate: 0.7,
            max_latency_us: 10_000.0,
            max_eviction_rate: 100.0,
            max_memory_overhead: 2.0,
            anomaly_stddev_multiplier: 3.0,
        }
    }
}

/// Performance baselines for comparison
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformanceBaselines {
    pub baseline_hit_rate: f64,
    pub baseline_latency_us: f64,
    pub baseline_throughput: f64,
    pub established_at: SystemTime,
    pub sample_count: usize,
}

impl Default for PerformanceBaselines {
    fn default() -> Self {
        Self {
            baseline_hit_rate: 0.0,
            baseline_latency_us: 0.0,
            baseline_throughput: 0.0,
            established_at: SystemTime::now(),
            sample_count: 0,
        }
    }
}

/// Aggregated statistics with moving averages
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AggregatedStats {
    pub moving_avg_hit_rate: f64,
    pub hit_rate_stddev: f64,
    pub moving_avg_latency_us: f64,
    pub latency_stddev: f64,
    pub peak_hit_rate: f64,
    pub lowest_hit_rate: f64,
    pub total_requests: u64,
}

impl Default for AggregatedStats {
    fn default() -> Self {
        Self {
            moving_avg_hit_rate: 0.0,
            hit_rate_stddev: 0.0,
            moving_avg_latency_us: 0.0,
            latency_stddev: 0.0,
            peak_hit_rate: 0.0,
            lowest_hit_rate: 1.0,
            total_requests: 0,
        }
    }
}

/// Enhanced telemetry collector with advanced features
pub struct EnhancedTelemetryCollector {
    base_collector: CacheTelemetryCollector,
    alert_thresholds: Arc<Mutex<AlertThresholds>>,
    baselines: Arc<Mutex<PerformanceBaselines>>,
    aggregated_stats: Arc<Mutex<AggregatedStats>>,
    active_alerts: Arc<Mutex<HashMap<AlertType, PerformanceAlert>>>,
}

impl EnhancedTelemetryCollector {
    /// Create a new enhanced telemetry collector
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            base_collector: CacheTelemetryCollector::new(config),
            alert_thresholds: Arc::new(Mutex::new(AlertThresholds::default())),
            baselines: Arc::new(Mutex::new(PerformanceBaselines::default())),
            aggregated_stats: Arc::new(Mutex::new(AggregatedStats::default())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a cache hit (delegates to base collector)
    pub fn record_hit(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        self.base_collector
            .record_hit(latency, size_bytes, key_hash);
        self.update_aggregated_stats();
        self.check_alerts();
    }

    /// Record a cache miss (delegates to base collector)
    pub fn record_miss(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        self.base_collector
            .record_miss(latency, size_bytes, key_hash);
        self.update_aggregated_stats();
        self.check_alerts();
    }

    /// Get base metrics
    pub fn get_metrics(&self) -> CacheTelemetryMetrics {
        self.base_collector.get_metrics()
    }

    /// Get aggregated statistics
    pub fn get_aggregated_stats(&self) -> AggregatedStats {
        self.aggregated_stats.lock().unwrap().clone()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Establish performance baselines from recent history
    pub fn establish_baselines(&self, sample_count: usize) {
        let snapshots = self.base_collector.get_snapshots();
        if snapshots.is_empty() {
            return;
        }

        let recent: Vec<_> = snapshots.iter().rev().take(sample_count).collect();
        if recent.is_empty() {
            return;
        }

        let avg_hit_rate =
            recent.iter().map(|s| s.metrics.hit_ratio).sum::<f64>() / recent.len() as f64;
        let avg_latency = recent
            .iter()
            .map(|s| s.metrics.avg_hit_latency_us)
            .sum::<f64>()
            / recent.len() as f64;
        let avg_throughput = recent
            .iter()
            .map(|s| s.metrics.requests_per_second)
            .sum::<f64>()
            / recent.len() as f64;

        let mut baselines = self.baselines.lock().unwrap();
        baselines.baseline_hit_rate = avg_hit_rate;
        baselines.baseline_latency_us = avg_latency;
        baselines.baseline_throughput = avg_throughput;
        baselines.established_at = SystemTime::now();
        baselines.sample_count = recent.len();
    }

    /// Get current baselines
    pub fn get_baselines(&self) -> PerformanceBaselines {
        self.baselines.lock().unwrap().clone()
    }

    /// Set custom alert thresholds
    pub fn set_alert_thresholds(&self, thresholds: AlertThresholds) {
        *self.alert_thresholds.lock().unwrap() = thresholds;
    }

    /// Export telemetry as JSON
    #[cfg(feature = "serialize")]
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let metrics = self.get_metrics();
        let stats = self.get_aggregated_stats();
        let alerts = self.get_active_alerts();
        let baselines = self.get_baselines();

        let export_data = serde_json::json!({
            "metrics": {
                "hits": metrics.hits,
                "misses": metrics.misses,
                "hit_ratio": metrics.hit_ratio,
                "avg_latency_us": metrics.avg_hit_latency_us,
                "eviction_rate": metrics.eviction_rate,
            },
            "aggregated_stats": {
                "moving_avg_hit_rate": stats.moving_avg_hit_rate,
                "hit_rate_stddev": stats.hit_rate_stddev,
                "peak_hit_rate": stats.peak_hit_rate,
                "total_requests": stats.total_requests,
            },
            "baselines": {
                "baseline_hit_rate": baselines.baseline_hit_rate,
                "baseline_latency_us": baselines.baseline_latency_us,
                "sample_count": baselines.sample_count,
            },
            "alerts": alerts.iter().map(|a| {
                serde_json::json!({
                    "type": format!("{:?}", a.alert_type),
                    "severity": format!("{:?}", a.severity),
                    "description": a.description,
                })
            }).collect::<Vec<_>>(),
        });

        serde_json::to_string_pretty(&export_data)
    }

    /// Export telemetry as CSV
    pub fn export_csv(&self) -> String {
        let snapshots = self.base_collector.get_snapshots();
        let mut csv =
            String::from("timestamp,hits,misses,hit_ratio,avg_latency_us,eviction_rate\n");

        for snapshot in snapshots {
            let metrics = &snapshot.metrics;
            csv.push_str(&format!(
                "{:?},{},{},{:.4},{:.2},{:.2}\n",
                snapshot.timestamp,
                metrics.hits,
                metrics.misses,
                metrics.hit_ratio,
                metrics.avg_hit_latency_us,
                metrics.eviction_rate
            ));
        }

        csv
    }

    /// Export telemetry in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let metrics = self.get_metrics();
        let stats = self.get_aggregated_stats();

        let mut output = String::new();

        output.push_str(&format!(
            "# HELP cache_hit_ratio Cache hit ratio (0.0-1.0)\n\
             # TYPE cache_hit_ratio gauge\n\
             cache_hit_ratio {}\n\n",
            metrics.hit_ratio
        ));

        output.push_str(&format!(
            "# HELP cache_latency_microseconds Average cache latency in microseconds\n\
             # TYPE cache_latency_microseconds gauge\n\
             cache_latency_microseconds {}\n\n",
            metrics.avg_hit_latency_us
        ));

        output.push_str(&format!(
            "# HELP cache_eviction_rate Evictions per second\n\
             # TYPE cache_eviction_rate gauge\n\
             cache_eviction_rate {}\n\n",
            metrics.eviction_rate
        ));

        output.push_str(&format!(
            "# HELP cache_total_requests Total cache requests\n\
             # TYPE cache_total_requests counter\n\
             cache_total_requests {}\n\n",
            stats.total_requests
        ));

        output
    }

    /// Generate comprehensive report
    pub fn generate_enhanced_report(&self) -> String {
        let mut report = self.base_collector.generate_report();
        let stats = self.get_aggregated_stats();
        let baselines = self.get_baselines();
        let alerts = self.get_active_alerts();

        report.push_str("\n## Aggregated Statistics\n");
        report.push_str(&format!(
            "  Moving Avg Hit Rate: {:.2}%\n",
            stats.moving_avg_hit_rate * 100.0
        ));
        report.push_str(&format!(
            "  Hit Rate Std Dev: {:.4}\n",
            stats.hit_rate_stddev
        ));
        report.push_str(&format!(
            "  Peak Hit Rate: {:.2}%\n",
            stats.peak_hit_rate * 100.0
        ));
        report.push_str(&format!("  Total Requests: {}\n\n", stats.total_requests));

        if baselines.sample_count > 0 {
            report.push_str("## Performance Baselines\n");
            report.push_str(&format!(
                "  Baseline Hit Rate: {:.2}%\n",
                baselines.baseline_hit_rate * 100.0
            ));
            report.push_str(&format!(
                "  Baseline Latency: {:.2}μs\n",
                baselines.baseline_latency_us
            ));
            report.push_str(&format!("  Samples: {}\n\n", baselines.sample_count));
        }

        if !alerts.is_empty() {
            report.push_str("## Active Alerts\n");
            for alert in alerts {
                report.push_str(&format!(
                    "  [{:?}] {:?}: {}\n",
                    alert.severity, alert.alert_type, alert.description
                ));
            }
        }

        report
    }

    // Private helper methods

    fn update_aggregated_stats(&self) {
        let snapshots = self.base_collector.get_snapshots();
        if snapshots.is_empty() {
            return;
        }

        let recent: Vec<_> = snapshots.iter().rev().take(60).collect();
        if recent.is_empty() {
            return;
        }

        let mut stats = self.aggregated_stats.lock().unwrap();

        // Calculate moving averages
        stats.moving_avg_hit_rate =
            recent.iter().map(|s| s.metrics.hit_ratio).sum::<f64>() / recent.len() as f64;
        stats.moving_avg_latency_us = recent
            .iter()
            .map(|s| s.metrics.avg_hit_latency_us)
            .sum::<f64>()
            / recent.len() as f64;

        // Calculate standard deviations
        let hit_rate_variance = recent
            .iter()
            .map(|s| (s.metrics.hit_ratio - stats.moving_avg_hit_rate).powi(2))
            .sum::<f64>()
            / recent.len() as f64;
        stats.hit_rate_stddev = hit_rate_variance.sqrt();

        let latency_variance = recent
            .iter()
            .map(|s| (s.metrics.avg_hit_latency_us - stats.moving_avg_latency_us).powi(2))
            .sum::<f64>()
            / recent.len() as f64;
        stats.latency_stddev = latency_variance.sqrt();

        // Track peaks
        for snapshot in &recent {
            stats.peak_hit_rate = stats.peak_hit_rate.max(snapshot.metrics.hit_ratio);
            stats.lowest_hit_rate = stats.lowest_hit_rate.min(snapshot.metrics.hit_ratio);
        }

        stats.total_requests = recent
            .last()
            .map(|s| s.metrics.hits + s.metrics.misses)
            .unwrap_or(0);
    }

    fn check_alerts(&self) {
        let metrics = self.get_metrics();
        let stats = self.get_aggregated_stats();
        let thresholds = self.alert_thresholds.lock().unwrap();
        let mut alerts = self.active_alerts.lock().unwrap();

        // Check hit rate
        if metrics.hit_ratio < thresholds.min_hit_rate {
            let alert = PerformanceAlert {
                alert_type: AlertType::LowHitRate,
                severity: if metrics.hit_ratio < thresholds.min_hit_rate * 0.8 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                description: format!(
                    "Cache hit rate ({:.2}%) below threshold ({:.2}%)",
                    metrics.hit_ratio * 100.0,
                    thresholds.min_hit_rate * 100.0
                ),
                metric_value: metrics.hit_ratio,
                threshold_value: thresholds.min_hit_rate,
                timestamp: SystemTime::now(),
            };
            alerts.insert(AlertType::LowHitRate, alert);
        } else {
            alerts.remove(&AlertType::LowHitRate);
        }

        // Check latency
        if metrics.avg_hit_latency_us > thresholds.max_latency_us {
            let alert = PerformanceAlert {
                alert_type: AlertType::HighLatency,
                severity: if metrics.avg_hit_latency_us > thresholds.max_latency_us * 1.5 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                description: format!(
                    "Average latency ({:.2}μs) exceeds threshold ({:.2}μs)",
                    metrics.avg_hit_latency_us, thresholds.max_latency_us
                ),
                metric_value: metrics.avg_hit_latency_us,
                threshold_value: thresholds.max_latency_us,
                timestamp: SystemTime::now(),
            };
            alerts.insert(AlertType::HighLatency, alert);
        } else {
            alerts.remove(&AlertType::HighLatency);
        }

        // Check eviction rate
        if metrics.eviction_rate > thresholds.max_eviction_rate {
            let alert = PerformanceAlert {
                alert_type: AlertType::HighEvictionRate,
                severity: AlertSeverity::Warning,
                description: format!(
                    "Eviction rate ({:.2}/s) exceeds threshold ({:.2}/s)",
                    metrics.eviction_rate, thresholds.max_eviction_rate
                ),
                metric_value: metrics.eviction_rate,
                threshold_value: thresholds.max_eviction_rate,
                timestamp: SystemTime::now(),
            };
            alerts.insert(AlertType::HighEvictionRate, alert);
        } else {
            alerts.remove(&AlertType::HighEvictionRate);
        }

        // Anomaly detection
        if stats.hit_rate_stddev > 0.0 {
            let z_score =
                (metrics.hit_ratio - stats.moving_avg_hit_rate).abs() / stats.hit_rate_stddev;
            if z_score > thresholds.anomaly_stddev_multiplier {
                let alert = PerformanceAlert {
                    alert_type: AlertType::AnomalyDetected,
                    severity: AlertSeverity::Warning,
                    description: format!("Hit rate anomaly detected (z-score: {:.2})", z_score),
                    metric_value: z_score,
                    threshold_value: thresholds.anomaly_stddev_multiplier,
                    timestamp: SystemTime::now(),
                };
                alerts.insert(AlertType::AnomalyDetected, alert);
            } else {
                alerts.remove(&AlertType::AnomalyDetected);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_collector_creation() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        let metrics = collector.get_metrics();

        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_ratio, 0.0);
    }

    #[test]
    fn test_record_hit() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), Some(1024), 12345);
        collector.record_hit(Duration::from_micros(200), Some(2048), 67890);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.hits, 2);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.hit_ratio, 1.0);
        assert!(metrics.avg_hit_latency_us > 0.0);
    }

    #[test]
    fn test_record_miss() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_miss(Duration::from_micros(500), Some(4096), 11111);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hit_ratio, 0.0);
        assert!(metrics.avg_miss_latency_us > 0.0);
    }

    #[test]
    fn test_hit_ratio_calculation() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), None, 1);
        collector.record_hit(Duration::from_micros(100), None, 2);
        collector.record_hit(Duration::from_micros(100), None, 3);
        collector.record_miss(Duration::from_micros(500), None, 4);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.hits, 3);
        assert_eq!(metrics.misses, 1);
        assert!((metrics.hit_ratio - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_eviction_tracking() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_insertion(Some(1024), 1);
        collector.record_eviction(Some(512), 2);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.insertions, 1);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.current_size_bytes, 512); // 1024 - 512
    }

    #[test]
    fn test_snapshot() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), None, 1);
        collector.snapshot();

        collector.record_miss(Duration::from_micros(200), None, 2);
        collector.snapshot();

        let snapshots = collector.get_snapshots();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].metrics.hits, 1);
        assert_eq!(snapshots[1].metrics.misses, 1);
    }

    #[test]
    fn test_recent_events() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig {
            max_events: 5,
            ..Default::default()
        });

        for i in 0..10 {
            collector.record_hit(Duration::from_micros(100), None, i);
        }

        let events = collector.get_recent_events(3);
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0].event_type, CacheEventType::Hit));
    }

    #[test]
    fn test_reset() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), None, 1);
        collector.record_miss(Duration::from_micros(200), None, 2);
        collector.snapshot();

        collector.reset();

        let metrics = collector.get_metrics();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(collector.get_snapshots().len(), 0);
    }

    #[test]
    fn test_generate_report() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), Some(1024), 1);
        collector.record_miss(Duration::from_micros(500), Some(2048), 2);

        let report = collector.generate_report();
        assert!(report.contains("Cache Telemetry Report"));
        assert!(report.contains("Hits:"));
        assert!(report.contains("Misses:"));
        assert!(report.contains("Latency Statistics"));
    }

    #[test]
    fn test_memory_tracking() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_insertion(Some(1024), 1);
        collector.record_insertion(Some(2048), 2);
        collector.record_eviction(Some(1024), 3);

        let metrics = collector.get_metrics();
        assert_eq!(metrics.current_size_bytes, 2048);
        assert_eq!(metrics.peak_size_bytes, 3072); // Max of 1024 + 2048
        assert_eq!(metrics.total_allocated_bytes, 3072);
        assert_eq!(metrics.total_freed_bytes, 1024);
    }
}
