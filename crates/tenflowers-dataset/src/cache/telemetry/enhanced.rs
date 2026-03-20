//! Enhanced telemetry collector with alert generation, baseline tracking, and
//! aggregated statistics built on top of [`CacheTelemetryCollector`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::collector::CacheTelemetryCollector;
use super::types::{
    AggregatedStats, AlertSeverity, AlertThresholds, AlertType, CacheTelemetryMetrics,
    PerformanceAlert, PerformanceBaselines, TelemetryConfig,
};

/// Enhanced telemetry collector with advanced features.
///
/// Wraps [`CacheTelemetryCollector`] and adds:
/// - moving-average aggregated statistics
/// - configurable alert thresholds
/// - baseline establishment from historical snapshots
/// - multi-format export (CSV, Prometheus, JSON)
pub struct EnhancedTelemetryCollector {
    pub(super) base_collector: CacheTelemetryCollector,
    alert_thresholds: Arc<Mutex<AlertThresholds>>,
    baselines: Arc<Mutex<PerformanceBaselines>>,
    aggregated_stats: Arc<Mutex<AggregatedStats>>,
    active_alerts: Arc<Mutex<HashMap<AlertType, PerformanceAlert>>>,
}

impl EnhancedTelemetryCollector {
    /// Create a new enhanced telemetry collector.
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            base_collector: CacheTelemetryCollector::new(config),
            alert_thresholds: Arc::new(Mutex::new(AlertThresholds::default())),
            baselines: Arc::new(Mutex::new(PerformanceBaselines::default())),
            aggregated_stats: Arc::new(Mutex::new(AggregatedStats::default())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a cache hit (delegates to base collector, then updates stats and alerts).
    pub fn record_hit(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        self.base_collector
            .record_hit(latency, size_bytes, key_hash);
        self.update_aggregated_stats();
        self.check_alerts();
    }

    /// Record a cache miss (delegates to base collector, then updates stats and alerts).
    pub fn record_miss(&self, latency: Duration, size_bytes: Option<usize>, key_hash: u64) {
        self.base_collector
            .record_miss(latency, size_bytes, key_hash);
        self.update_aggregated_stats();
        self.check_alerts();
    }

    /// Get base metrics.
    pub fn get_metrics(&self) -> CacheTelemetryMetrics {
        self.base_collector.get_metrics()
    }

    /// Get the current aggregated statistics.
    pub fn get_aggregated_stats(&self) -> AggregatedStats {
        self.aggregated_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get all currently active performance alerts.
    pub fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts
            .lock()
            .expect("active_alerts lock should not be poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Establish performance baselines from the most recent `sample_count` snapshots.
    pub fn establish_baselines(&self, sample_count: usize) {
        let snapshots = self.base_collector.get_snapshots();
        if snapshots.is_empty() {
            return;
        }

        let recent: Vec<_> = snapshots.iter().rev().take(sample_count).collect();
        if recent.is_empty() {
            return;
        }

        let n = recent.len() as f64;
        let avg_hit_rate = recent.iter().map(|s| s.metrics.hit_ratio).sum::<f64>() / n;
        let avg_latency = recent
            .iter()
            .map(|s| s.metrics.avg_hit_latency_us)
            .sum::<f64>()
            / n;
        let avg_throughput = recent
            .iter()
            .map(|s| s.metrics.requests_per_second)
            .sum::<f64>()
            / n;

        let mut baselines = self.baselines.lock().expect("lock should not be poisoned");
        baselines.baseline_hit_rate = avg_hit_rate;
        baselines.baseline_latency_us = avg_latency;
        baselines.baseline_throughput = avg_throughput;
        baselines.established_at = SystemTime::now();
        baselines.sample_count = recent.len();
    }

    /// Get the current performance baselines.
    pub fn get_baselines(&self) -> PerformanceBaselines {
        self.baselines
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Override the alert thresholds.
    pub fn set_alert_thresholds(&self, thresholds: AlertThresholds) {
        *self
            .alert_thresholds
            .lock()
            .expect("lock should not be poisoned") = thresholds;
    }

    // ---- private helpers ----

    fn update_aggregated_stats(&self) {
        let snapshots = self.base_collector.get_snapshots();
        if snapshots.is_empty() {
            return;
        }

        let recent: Vec<_> = snapshots.iter().rev().take(60).collect();
        if recent.is_empty() {
            return;
        }

        let mut stats = self
            .aggregated_stats
            .lock()
            .expect("lock should not be poisoned");

        let n = recent.len() as f64;

        stats.moving_avg_hit_rate = recent.iter().map(|s| s.metrics.hit_ratio).sum::<f64>() / n;
        stats.moving_avg_latency_us = recent
            .iter()
            .map(|s| s.metrics.avg_hit_latency_us)
            .sum::<f64>()
            / n;

        let hit_rate_variance = recent
            .iter()
            .map(|s| (s.metrics.hit_ratio - stats.moving_avg_hit_rate).powi(2))
            .sum::<f64>()
            / n;
        stats.hit_rate_stddev = hit_rate_variance.sqrt();

        let latency_variance = recent
            .iter()
            .map(|s| (s.metrics.avg_hit_latency_us - stats.moving_avg_latency_us).powi(2))
            .sum::<f64>()
            / n;
        stats.latency_stddev = latency_variance.sqrt();

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
        let thresholds = self
            .alert_thresholds
            .lock()
            .expect("lock should not be poisoned");
        let mut alerts = self
            .active_alerts
            .lock()
            .expect("lock should not be poisoned");

        // Hit rate check
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

        // Latency check
        if metrics.avg_hit_latency_us > thresholds.max_latency_us {
            let alert = PerformanceAlert {
                alert_type: AlertType::HighLatency,
                severity: if metrics.avg_hit_latency_us > thresholds.max_latency_us * 1.5 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                description: format!(
                    "Average latency ({:.2}\u{b5}s) exceeds threshold ({:.2}\u{b5}s)",
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

        // Eviction rate check
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
