//! Derived metric calculation methods for `CacheTelemetryCollector`.
//!
//! Provides byte-level hit ratio, nanosecond-precision latency percentiles,
//! and other higher-order metrics derived from the raw event stream and
//! latency histogram maintained by the collector.

use std::time::Duration;

use super::collector::CacheTelemetryCollector;
use super::types::{calculate_percentiles, CacheEventType};

impl CacheTelemetryCollector {
    /// Returns the ratio of bytes served from cache versus total bytes requested.
    ///
    /// This is computed from bytes associated with recorded hit and miss events.
    /// Returns `0.0` when no byte-size information has been recorded.
    pub fn byte_hit_ratio(&self) -> f64 {
        let events = self
            .recent_events
            .lock()
            .expect("lock should not be poisoned");

        let mut hit_bytes: u64 = 0;
        let mut total_bytes: u64 = 0;

        for event in events.iter() {
            if let Some(size) = event.size_bytes {
                total_bytes += size as u64;
                if event.event_type == CacheEventType::Hit {
                    hit_bytes += size as u64;
                }
            }
        }

        if total_bytes == 0 {
            0.0
        } else {
            hit_bytes as f64 / total_bytes as f64
        }
    }

    /// Returns the 95th-percentile latency in **nanoseconds** from recorded latency samples.
    ///
    /// Uses the same histogram used for `get_metrics()` percentile calculations but
    /// converts from microseconds (histogram buckets) to nanoseconds.
    /// Returns `0.0` when no latency data has been recorded.
    pub fn p95_latency_ns(&self) -> f64 {
        let histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");

        if histogram.is_empty() {
            return 0.0;
        }

        let percentiles = calculate_percentiles(&histogram);
        // histogram buckets are in microseconds; convert to nanoseconds
        percentiles.1 * 1_000.0
    }

    /// Returns the 99th-percentile latency in **nanoseconds** from recorded latency samples.
    ///
    /// Uses the same histogram used for `get_metrics()` percentile calculations but
    /// converts from microseconds (histogram buckets) to nanoseconds.
    /// Returns `0.0` when no latency data has been recorded.
    pub fn p99_latency_ns(&self) -> f64 {
        let histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");

        if histogram.is_empty() {
            return 0.0;
        }

        let percentiles = calculate_percentiles(&histogram);
        // histogram buckets are in microseconds; convert to nanoseconds
        percentiles.2 * 1_000.0
    }

    /// Returns the 50th-percentile (median) latency in **microseconds**.
    ///
    /// Returns `0.0` when no latency data has been recorded.
    pub fn p50_latency_us(&self) -> f64 {
        let histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");

        if histogram.is_empty() {
            return 0.0;
        }

        calculate_percentiles(&histogram).0
    }

    /// Returns the 95th-percentile latency in **microseconds**.
    ///
    /// Returns `0.0` when no latency data has been recorded.
    pub fn p95_latency_us(&self) -> f64 {
        let histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");

        if histogram.is_empty() {
            return 0.0;
        }

        calculate_percentiles(&histogram).1
    }

    /// Returns the 99th-percentile latency in **microseconds**.
    ///
    /// Returns `0.0` when no latency data has been recorded.
    pub fn p99_latency_us(&self) -> f64 {
        let histogram = self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned");

        if histogram.is_empty() {
            return 0.0;
        }

        calculate_percentiles(&histogram).2
    }

    /// Returns the total number of requests (hits + misses) recorded so far.
    pub fn total_requests(&self) -> u64 {
        let metrics = self
            .current_metrics
            .lock()
            .expect("lock should not be poisoned");
        metrics.hits + metrics.misses
    }

    /// Returns the current hit ratio (0.0 – 1.0) without acquiring a full metrics snapshot.
    pub fn current_hit_ratio(&self) -> f64 {
        let metrics = self
            .current_metrics
            .lock()
            .expect("lock should not be poisoned");
        let total = metrics.hits + metrics.misses;
        if total == 0 {
            0.0
        } else {
            metrics.hits as f64 / total as f64
        }
    }

    /// Returns average hit latency in microseconds.
    pub fn avg_hit_latency_us(&self) -> f64 {
        self.current_metrics
            .lock()
            .expect("lock should not be poisoned")
            .avg_hit_latency_us
    }

    /// Returns average miss latency in microseconds.
    pub fn avg_miss_latency_us(&self) -> f64 {
        self.current_metrics
            .lock()
            .expect("lock should not be poisoned")
            .avg_miss_latency_us
    }

    /// Computes the eviction rate (evictions per second) over the elapsed window.
    ///
    /// Returns `0.0` if the window duration is zero.
    pub fn eviction_rate_per_sec(&self) -> f64 {
        let metrics = self
            .current_metrics
            .lock()
            .expect("lock should not be poisoned");
        let elapsed = metrics.window_start.elapsed();
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            metrics.evictions as f64 / secs
        }
    }

    /// Returns the latency histogram snapshot (bucket_us → count).
    ///
    /// Each bucket key represents the lower bound of a 100-µs interval.
    pub fn latency_histogram_snapshot(&self) -> std::collections::HashMap<u64, u64> {
        self.latency_histogram
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Checks whether there is enough latency data to compute meaningful percentiles.
    pub fn has_latency_data(&self) -> bool {
        !self
            .latency_histogram
            .lock()
            .expect("lock should not be poisoned")
            .is_empty()
    }

    /// Records a synthetic latency sample directly (useful for testing / benchmarking).
    pub fn inject_latency_sample(&self, latency: Duration) {
        self.record_latency(latency);
    }
}
