//! Core `CacheTelemetryCollector` implementation.
//!
//! Records cache events (hits, misses, evictions, insertions) and maintains
//! a latency histogram, an event ring-buffer, and periodic metric snapshots.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::types::{
    calculate_percentiles, CacheEvent, CacheEventType, CacheTelemetryMetrics, MetricsSnapshot,
    TelemetryConfig,
};

/// Cache telemetry collector with historical tracking.
pub struct CacheTelemetryCollector {
    /// Current active metrics
    pub(super) current_metrics: Arc<Mutex<CacheTelemetryMetrics>>,
    /// Recent events buffer (limited size)
    pub(super) recent_events: Arc<Mutex<VecDeque<CacheEvent>>>,
    /// Historical snapshots for time-series analysis
    pub(super) snapshots: Arc<Mutex<VecDeque<MetricsSnapshot>>>,
    /// Latency histogram buckets (microseconds → count)
    pub(super) latency_histogram: Arc<Mutex<HashMap<u64, u64>>>,
    /// Configuration
    pub(super) config: TelemetryConfig,
}

impl CacheTelemetryCollector {
    /// Create a new telemetry collector.
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            current_metrics: Arc::new(Mutex::new(CacheTelemetryMetrics::new())),
            recent_events: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_events))),
            snapshots: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_snapshots))),
            latency_histogram: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Record a cache hit.
    pub fn record_hit(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        {
            let mut metrics = self
                .current_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.hits += 1;

            let latency_us = latency.as_micros() as f64;
            let total_hits = metrics.hits as f64;
            metrics.avg_hit_latency_us =
                (metrics.avg_hit_latency_us * (total_hits - 1.0) + latency_us) / total_hits;

            if let Some(size) = size_bytes {
                metrics.current_size_bytes = metrics.current_size_bytes.saturating_add(size);
            }
        }

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

    /// Record a cache miss.
    pub fn record_miss(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        {
            let mut metrics = self
                .current_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.misses += 1;

            let latency_us = latency.as_micros() as f64;
            let total_misses = metrics.misses as f64;
            metrics.avg_miss_latency_us =
                (metrics.avg_miss_latency_us * (total_misses - 1.0) + latency_us) / total_misses;
        }

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

    /// Record an eviction.
    pub fn record_eviction(&self, size_bytes: Option<usize>, key_hash: u64) {
        {
            let mut metrics = self
                .current_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.evictions += 1;

            if let Some(size) = size_bytes {
                metrics.current_size_bytes = metrics.current_size_bytes.saturating_sub(size);
                metrics.total_freed_bytes += size as u64;
            }
        }

        self.record_event(CacheEvent {
            event_type: CacheEventType::Eviction,
            timestamp: Instant::now(),
            latency: Duration::from_micros(0),
            size_bytes,
            key_hash,
        });
    }

    /// Record an insertion.
    pub fn record_insertion(&self, size_bytes: Option<usize>, key_hash: u64) {
        {
            let mut metrics = self
                .current_metrics
                .lock()
                .expect("lock should not be poisoned");
            metrics.insertions += 1;

            if let Some(size) = size_bytes {
                metrics.current_size_bytes = metrics.current_size_bytes.saturating_add(size);
                metrics.peak_size_bytes = metrics.peak_size_bytes.max(metrics.current_size_bytes);
                metrics.total_allocated_bytes += size as u64;
            }
        }

        self.record_event(CacheEvent {
            event_type: CacheEventType::Insertion,
            timestamp: Instant::now(),
            latency: Duration::from_micros(0),
            size_bytes,
            key_hash,
        });
    }

    /// Get current metrics snapshot (with derived values calculated).
    pub fn get_metrics(&self) -> CacheTelemetryMetrics {
        let mut metrics = self
            .current_metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone();
        metrics.window_duration = metrics.window_start.elapsed();
        metrics.calculate_derived();

        if self.config.track_latency_histogram {
            let histogram = self
                .latency_histogram
                .lock()
                .expect("lock should not be poisoned");
            let percentiles = calculate_percentiles(&histogram);
            metrics.p50_latency_us = percentiles.0;
            metrics.p95_latency_us = percentiles.1;
            metrics.p99_latency_us = percentiles.2;
        }

        metrics
    }

    /// Take a snapshot of current metrics and store it.
    pub fn snapshot(&self) {
        let snapshot = MetricsSnapshot {
            timestamp: Instant::now(),
            metrics: self.get_metrics(),
        };

        let mut snapshots = self.snapshots.lock().expect("lock should not be poisoned");
        snapshots.push_back(snapshot);

        while snapshots.len() > self.config.max_snapshots {
            snapshots.pop_front();
        }
    }

    /// Get the most recent `count` events (newest first).
    pub fn get_recent_events(&self, count: usize) -> Vec<CacheEvent> {
        self.recent_events
            .lock()
            .expect("lock should not be poisoned")
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get all stored historical snapshots (oldest first).
    pub fn get_snapshots(&self) -> Vec<MetricsSnapshot> {
        self.snapshots
            .lock()
            .expect("lock should not be poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// Reset all metrics, events, snapshots, and the latency histogram.
    pub fn reset(&self) {
        *self
            .current_metrics
            .lock()
            .expect("lock should not be poisoned") = CacheTelemetryMetrics::new();
        self.recent_events
            .lock()
            .expect("lock should not be poisoned")
            .clear();
        self.snapshots
            .lock()
            .expect("lock should not be poisoned")
            .clear();
        self.latency_histogram
            .lock()
            .expect("lock should not be poisoned")
            .clear();
    }

    // ---- private helpers ----

    pub(super) fn record_event(&self, event: CacheEvent) {
        let mut events = self
            .recent_events
            .lock()
            .expect("lock should not be poisoned");
        events.push_back(event);

        while events.len() > self.config.max_events {
            events.pop_front();
        }
    }

    pub(super) fn record_latency(&self, latency: Duration) {
        let bucket = (latency.as_micros() as u64 / 100) * 100; // 100 µs buckets
        let mut histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");
        *histogram.entry(bucket).or_insert(0) += 1;
    }
}
