//! Report generation for `CacheTelemetryCollector` and `EnhancedTelemetryCollector`.
//!
//! Provides human-readable text reports, CSV export, Prometheus-format export,
//! and (under the `serialize` feature) JSON export.

use super::collector::CacheTelemetryCollector;
use super::enhanced::EnhancedTelemetryCollector;

// ---- CacheTelemetryCollector reports ----

impl CacheTelemetryCollector {
    /// Generate a concise human-readable report of the current metrics.
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

    /// Returns a comprehensive formatted report containing all available metrics.
    ///
    /// Includes request statistics, latency percentiles (including nanosecond p95/p99),
    /// byte hit ratio, memory statistics, and throughput metrics.
    pub fn comprehensive_report(&self) -> String {
        let metrics = self.get_metrics();
        let byte_ratio = self.byte_hit_ratio();
        let p95_ns = self.p95_latency_ns();
        let p99_ns = self.p99_latency_ns();

        let mut report = String::new();

        report.push_str("=== Comprehensive Cache Telemetry Report ===\n\n");

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

        report.push_str("## Cache Effectiveness\n");
        report.push_str(&format!(
            "  Hit Ratio (by count): {:.4}\n",
            metrics.hit_ratio
        ));
        report.push_str(&format!("  Byte Hit Ratio:       {:.4}\n", byte_ratio));
        report.push_str(&format!(
            "  Eviction Rate:        {:.2}/sec\n\n",
            metrics.eviction_rate
        ));

        report.push_str("## Latency Statistics\n");
        report.push_str(&format!(
            "  Avg Hit Latency:  {:.2} us\n",
            metrics.avg_hit_latency_us
        ));
        report.push_str(&format!(
            "  Avg Miss Latency: {:.2} us\n",
            metrics.avg_miss_latency_us
        ));
        report.push_str(&format!("  P50: {:.2} us\n", metrics.p50_latency_us));
        report.push_str(&format!(
            "  P95: {:.2} us  ({:.2} ns)\n",
            metrics.p95_latency_us, p95_ns
        ));
        report.push_str(&format!(
            "  P99: {:.2} us  ({:.2} ns)\n\n",
            metrics.p99_latency_us, p99_ns
        ));

        report.push_str("## Memory Statistics\n");
        report.push_str(&format!(
            "  Current Size:     {} bytes\n",
            metrics.current_size_bytes
        ));
        report.push_str(&format!(
            "  Peak Size:        {} bytes\n",
            metrics.peak_size_bytes
        ));
        report.push_str(&format!(
            "  Total Allocated:  {} bytes\n",
            metrics.total_allocated_bytes
        ));
        report.push_str(&format!(
            "  Total Freed:      {} bytes\n\n",
            metrics.total_freed_bytes
        ));

        report.push_str("## Throughput Statistics\n");
        report.push_str(&format!(
            "  Requests/sec: {:.2}\n",
            metrics.requests_per_second
        ));
        report.push_str(&format!(
            "  Bytes/sec:    {:.2}\n",
            metrics.bytes_per_second
        ));

        report
    }
}

// ---- EnhancedTelemetryCollector reports ----

impl EnhancedTelemetryCollector {
    /// Generate an enhanced report that extends the base report with aggregated
    /// statistics, performance baselines, and active alerts.
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
                "  Baseline Latency: {:.2}\u{b5}s\n",
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

    /// Export a snapshot of all telemetry as a CSV string.
    ///
    /// Columns: `timestamp,hits,misses,hit_ratio,avg_latency_us,eviction_rate`
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

    /// Export current metrics in Prometheus text exposition format.
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

    /// Export telemetry as a JSON string (no external crates required).
    ///
    /// Available unconditionally; uses hand-written serialisation.
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
}
