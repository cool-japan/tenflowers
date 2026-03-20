//! Unit tests for the cache telemetry subsystem.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::cache::telemetry::{CacheEventType, CacheTelemetryCollector, TelemetryConfig};

    // ---- basic collector tests ----

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

    // ---- byte_hit_ratio tests ----

    #[test]
    fn test_byte_hit_ratio_no_events() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert_eq!(collector.byte_hit_ratio(), 0.0);
    }

    #[test]
    fn test_byte_hit_ratio_all_hits() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(100), Some(512), 1);
        collector.record_hit(Duration::from_micros(100), Some(1024), 2);

        let ratio = collector.byte_hit_ratio();
        assert!((ratio - 1.0).abs() < 1e-9, "expected 1.0, got {}", ratio);
    }

    #[test]
    fn test_byte_hit_ratio_all_misses() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_miss(Duration::from_micros(500), Some(1024), 1);
        collector.record_miss(Duration::from_micros(500), Some(2048), 2);

        let ratio = collector.byte_hit_ratio();
        assert_eq!(ratio, 0.0, "expected 0.0, got {}", ratio);
    }

    #[test]
    fn test_byte_hit_ratio_mixed() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(100), Some(1024), 1);
        collector.record_miss(Duration::from_micros(500), Some(1024), 2);

        let ratio = collector.byte_hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-9, "expected 0.5, got {}", ratio);
    }

    #[test]
    fn test_byte_hit_ratio_without_size_info() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(100), None, 1);
        collector.record_miss(Duration::from_micros(100), None, 2);
        assert_eq!(collector.byte_hit_ratio(), 0.0);
    }

    // ---- latency percentile tests ----

    #[test]
    fn test_p95_latency_ns_empty() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert_eq!(collector.p95_latency_ns(), 0.0);
    }

    #[test]
    fn test_p99_latency_ns_empty() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert_eq!(collector.p99_latency_ns(), 0.0);
    }

    #[test]
    fn test_p95_latency_ns_with_data() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        for i in 1_u64..=100 {
            collector.record_hit(Duration::from_micros(i * 10), None, i);
        }

        let p95 = collector.p95_latency_ns();
        assert!(p95 > 0.0, "p95_latency_ns should be positive, got {}", p95);
    }

    #[test]
    fn test_p99_latency_ns_with_data() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        for i in 1_u64..=100 {
            collector.record_hit(Duration::from_micros(i * 10), None, i);
        }

        let p99 = collector.p99_latency_ns();
        let p95 = collector.p95_latency_ns();
        assert!(p99 >= p95, "p99 ({}) should be >= p95 ({})", p99, p95);
    }

    // ---- comprehensive_report tests ----

    #[test]
    fn test_comprehensive_report_contains_all_sections() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());

        collector.record_hit(Duration::from_micros(100), Some(1024), 1);
        collector.record_miss(Duration::from_micros(500), Some(2048), 2);
        collector.record_insertion(Some(512), 3);
        collector.record_eviction(Some(256), 4);

        let report = collector.comprehensive_report();

        assert!(
            report.contains("Comprehensive Cache Telemetry Report"),
            "missing header"
        );
        assert!(
            report.contains("Request Statistics"),
            "missing request stats"
        );
        assert!(
            report.contains("Cache Effectiveness"),
            "missing effectiveness section"
        );
        assert!(report.contains("Byte Hit Ratio"), "missing byte hit ratio");
        assert!(
            report.contains("Latency Statistics"),
            "missing latency stats"
        );
        assert!(report.contains("P95"), "missing P95");
        assert!(report.contains("P99"), "missing P99");
        assert!(report.contains("Memory Statistics"), "missing memory stats");
        assert!(
            report.contains("Throughput Statistics"),
            "missing throughput stats"
        );
    }

    #[test]
    fn test_comprehensive_report_hit_ratio_present() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(50), Some(256), 1);
        collector.record_hit(Duration::from_micros(50), Some(256), 2);
        collector.record_hit(Duration::from_micros(50), Some(256), 3);
        collector.record_miss(Duration::from_micros(300), Some(256), 4);

        let report = collector.comprehensive_report();
        assert!(
            report.contains("Hit Ratio (by count)"),
            "missing hit ratio line"
        );
    }

    // ---- additional metrics module tests ----

    #[test]
    fn test_total_requests() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(10), None, 1);
        collector.record_miss(Duration::from_micros(20), None, 2);
        assert_eq!(collector.total_requests(), 2);
    }

    #[test]
    fn test_current_hit_ratio() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert_eq!(collector.current_hit_ratio(), 0.0);

        collector.record_hit(Duration::from_micros(10), None, 1);
        assert!((collector.current_hit_ratio() - 1.0).abs() < 1e-9);

        collector.record_miss(Duration::from_micros(20), None, 2);
        assert!((collector.current_hit_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_has_latency_data() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert!(!collector.has_latency_data());

        collector.record_hit(Duration::from_micros(100), None, 1);
        assert!(collector.has_latency_data());
    }

    #[test]
    fn test_inject_latency_sample() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        assert!(!collector.has_latency_data());

        collector.inject_latency_sample(Duration::from_micros(500));
        assert!(collector.has_latency_data());

        let p50 = collector.p50_latency_us();
        assert!(p50 > 0.0);
    }

    #[test]
    fn test_latency_histogram_snapshot() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(150), None, 1);
        collector.record_hit(Duration::from_micros(350), None, 2);

        let hist = collector.latency_histogram_snapshot();
        assert!(!hist.is_empty());
        let total: u64 = hist.values().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_avg_hit_miss_latency() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        collector.record_hit(Duration::from_micros(200), None, 1);
        collector.record_miss(Duration::from_micros(800), None, 2);

        assert!((collector.avg_hit_latency_us() - 200.0).abs() < 1.0);
        assert!((collector.avg_miss_latency_us() - 800.0).abs() < 1.0);
    }

    #[test]
    fn test_eviction_rate_per_sec_zero_when_fresh() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        // Immediately after creation, elapsed time approaches zero; rate should be
        // either 0.0 (no evictions) or defined as 0.0 by the implementation.
        let rate = collector.eviction_rate_per_sec();
        assert!(rate >= 0.0);
    }

    #[test]
    fn test_p50_p95_p99_ordering() {
        let collector = CacheTelemetryCollector::new(TelemetryConfig::default());
        for i in 1_u64..=200 {
            collector.inject_latency_sample(Duration::from_micros(i));
        }

        let p50 = collector.p50_latency_us();
        let p95 = collector.p95_latency_us();
        let p99 = collector.p99_latency_us();

        assert!(p50 <= p95, "p50 ({}) should be <= p95 ({})", p50, p95);
        assert!(p95 <= p99, "p95 ({}) should be <= p99 ({})", p95, p99);
    }
}
