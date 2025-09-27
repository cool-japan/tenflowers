//! Ultra Performance Monitoring System - Modular Architecture
//!
//! This module contains the refactored ultra performance monitoring system,
//! split into logical components for better maintainability and organization.

// Module declarations
pub mod alerts;
pub mod analytics;
pub mod core;
pub mod dashboard;
pub mod metrics;
pub mod prediction;

// Re-export main components
pub use core::{MonitoringReport, UltraPerformanceMonitor};

// Re-export key types from each module
pub use alerts::{
    AlertEvent, AlertEventType, AlertManager, AlertRule, AlertSeverity, ComparisonOperator,
    NotificationChannel, NotificationChannelType, PerformanceAlert, SuppressionRule,
};

pub use analytics::{
    AlgorithmType, BottleneckAlgorithm, BottleneckDetector, BottleneckType, CorrelationAnalyzer,
    OpportunityScoring, OpportunityType, OptimizationIdentifier, OptimizationOpportunity,
    PerformanceAnalyticsEngine, RegressionDetector, SystemBottleneck, TrendAnalyzer, TrendData,
    TrendType,
};

pub use dashboard::{
    ChartConfig, ChartType, DashboardConfig, DashboardWidget, DataPoint, DataSeries,
    KeyPerformanceIndicator, KpiStatus, LineStyle, PerformanceChart, PerformanceDashboard,
    TrendDirection, WidgetConfig, WidgetLayout, WidgetType,
};

pub use metrics::{
    CacheLevelMetrics, CustomMetric, CustomMetricDefinition, GpuMetrics, MemorySegmentMetrics,
    MetricsCollector, MonitoringConfig, NumaNodeMetrics, OperationMetrics, PerformanceSnapshot,
    ResourceMetrics, StorageMetrics, SystemMetrics,
};

pub use prediction::{
    AnomalyDetector, AnomalyModel, AnomalyModelType, Forecast, ForecastPoint, ForecastingEngine,
    ModelType, PatternDatabase, PatternOutcome, PerformanceAnomaly, PerformancePattern,
    PerformancePrediction, PerformancePredictor, PredictionModel,
};
