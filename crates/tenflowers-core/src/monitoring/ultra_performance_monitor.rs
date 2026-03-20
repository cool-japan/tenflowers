//! Ultra-Advanced Production Performance Monitoring System
//!
//! This module provides comprehensive performance monitoring, alerting, and optimization
//! insights for production TenfloweRS deployments with real-time analytics and predictive capabilities.
//!
//! ## Modular Architecture
//!
//! The monitoring system has been refactored into logical modules for better maintainability:
//!
//! - **`core`**: Main UltraPerformanceMonitor orchestrator
//! - **`metrics`**: Data structures and collection for performance metrics
//! - **`analytics`**: Performance analysis, trend detection, and bottleneck identification
//! - **`alerts`**: Alert management, rules, and notifications
//! - **`dashboard`**: Visualization components, charts, and KPIs
//! - **`prediction`**: Predictive analytics, forecasting, and anomaly detection
//!
//! ## Usage
//!
//! ```rust
//! use tenflowers_core::monitoring::UltraPerformanceMonitor;
//! use tenflowers_core::monitoring::MonitoringConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create monitor with default configuration
//! let monitor = UltraPerformanceMonitor::default();
//!
//! // Or with custom configuration
//! let config = MonitoringConfig::default();
//! let monitor = UltraPerformanceMonitor::new(config)?;
//!
//! // Collect performance metrics
//! let snapshot = monitor.collect_metrics()?;
//!
//! // Generate comprehensive report
//! let report = monitor.generate_report()?;
//! # Ok(())
//! # }
//! ```

// Import modular components
pub mod ultra_performance;

// Re-export all public APIs from the modular structure
pub use ultra_performance::*;

// Legacy compatibility - re-export main types at module root
pub use ultra_performance::{
    AlertSeverity, MonitoringConfig, MonitoringReport, PerformanceAlert, PerformanceDashboard,
    PerformancePredictor, PerformanceSnapshot, SystemMetrics, UltraPerformanceMonitor,
};
