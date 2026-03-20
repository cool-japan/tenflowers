//! Cache telemetry types: structs and enums used throughout the telemetry subsystem.

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

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

/// Calculate percentiles from a latency histogram (bucket → count map).
///
/// Buckets are in microseconds. Returns (p50, p95, p99) as `f64` microseconds.
pub(crate) fn calculate_percentiles(histogram: &HashMap<u64, u64>) -> (f64, f64, f64) {
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
