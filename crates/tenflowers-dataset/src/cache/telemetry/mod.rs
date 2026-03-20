//! Cache telemetry and metrics collection system.
//!
//! This module provides comprehensive metrics collection, monitoring,
//! and analysis capabilities for cache operations. It tracks performance
//! metrics, identifies patterns, and helps optimise cache effectiveness.
//!
//! ## Module layout
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | `types`    | All public structs and enums (`CacheTelemetryMetrics`, `CacheEvent`, alerts, …) |
//! | `collector`| `CacheTelemetryCollector` – core event recording and snapshot management |
//! | `metrics`  | Derived metric helpers: `byte_hit_ratio`, `p95/p99_latency_ns`, … |
//! | `enhanced` | `EnhancedTelemetryCollector` – alert generation, baselines, aggregated stats |
//! | `reports`  | Report generation: text, CSV, Prometheus, JSON |
//! | `tests`    | Unit-test suite |

mod collector;
mod enhanced;
mod metrics;
mod reports;
mod types;

#[cfg(test)]
mod tests;

// ---- public re-exports ----

pub use collector::CacheTelemetryCollector;
pub use enhanced::EnhancedTelemetryCollector;
pub use types::{
    AggregatedStats, AlertSeverity, AlertThresholds, AlertType, CacheEvent, CacheEventType,
    CacheTelemetryMetrics, MetricsSnapshot, PerformanceAlert, PerformanceBaselines,
    TelemetryConfig,
};
