//! Monitoring system configuration
//!
//! This module provides configuration types and default settings for the
//! ultra-performance monitoring system.

use std::time::Duration;

/// Monitoring system configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Enable advanced analytics
    pub enable_analytics: bool,
    /// Enable alerting
    pub enable_alerting: bool,
    /// Enable dashboard
    pub enable_dashboard: bool,
    /// Enable prediction
    pub enable_prediction: bool,
    /// Performance impact tolerance
    pub performance_impact_tolerance: f64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(30),
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            enable_analytics: true,
            enable_alerting: true,
            enable_dashboard: true,
            enable_prediction: true,
            performance_impact_tolerance: 0.05, // 5% overhead tolerance
        }
    }
}

impl MonitoringConfig {
    /// Create a new monitoring configuration with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set collection interval
    pub fn with_collection_interval(mut self, interval: Duration) -> Self {
        self.collection_interval = interval;
        self
    }

    /// Set metrics retention period
    pub fn with_retention_period(mut self, period: Duration) -> Self {
        self.retention_period = period;
        self
    }

    /// Enable or disable analytics
    pub fn with_analytics(mut self, enabled: bool) -> Self {
        self.enable_analytics = enabled;
        self
    }

    /// Enable or disable alerting
    pub fn with_alerting(mut self, enabled: bool) -> Self {
        self.enable_alerting = enabled;
        self
    }

    /// Enable or disable dashboard
    pub fn with_dashboard(mut self, enabled: bool) -> Self {
        self.enable_dashboard = enabled;
        self
    }

    /// Enable or disable prediction
    pub fn with_prediction(mut self, enabled: bool) -> Self {
        self.enable_prediction = enabled;
        self
    }

    /// Set performance impact tolerance
    pub fn with_performance_tolerance(mut self, tolerance: f64) -> Self {
        self.performance_impact_tolerance = tolerance;
        self
    }

    /// Create a minimal configuration for lightweight monitoring
    pub fn minimal() -> Self {
        Self {
            collection_interval: Duration::from_secs(300), // 5 minutes
            retention_period: Duration::from_secs(24 * 3600), // 1 day
            enable_analytics: false,
            enable_alerting: false,
            enable_dashboard: false,
            enable_prediction: false,
            performance_impact_tolerance: 0.01, // 1% overhead tolerance
        }
    }

    /// Create a comprehensive configuration for full monitoring
    pub fn comprehensive() -> Self {
        Self {
            collection_interval: Duration::from_secs(10), // 10 seconds
            retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
            enable_analytics: true,
            enable_alerting: true,
            enable_dashboard: true,
            enable_prediction: true,
            performance_impact_tolerance: 0.1, // 10% overhead tolerance
        }
    }

    /// Validate the configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.collection_interval.as_secs() == 0 {
            return Err("Collection interval must be greater than zero".to_string());
        }

        if self.retention_period < self.collection_interval {
            return Err("Retention period must be greater than collection interval".to_string());
        }

        if !(0.0..=1.0).contains(&self.performance_impact_tolerance) {
            return Err("Performance impact tolerance must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}