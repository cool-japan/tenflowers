//! Ultra-Advanced Production Performance Monitoring
//!
//! This module provides comprehensive performance monitoring, analytics, and optimization
//! insights for production TenfloweRS deployments.

pub mod ultra_performance_monitor;

pub use ultra_performance_monitor::{
    AlertEvent,
    AlertEventType,
    // Alert system
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlgorithmType,
    AnomalyDetector,
    AnomalyModel,
    AnomalyModelType,
    BottleneckAlgorithm,
    BottleneckDetector,
    BottleneckType,
    CacheLevelMetrics,
    ChartConfig,
    ChartType,
    ComparisonOperator,
    CorrelationAnalyzer,

    CustomMetric,
    CustomMetricDefinition,

    DashboardConfig,

    // Dashboard components
    DashboardWidget,
    DataPoint,
    DataSeries,
    Forecast,
    ForecastPoint,
    ForecastingEngine,
    GpuMetrics,
    KeyPerformanceIndicator,
    KpiStatus,
    LineStyle,
    MemorySegmentMetrics,
    // Metrics collection
    MetricsCollector,
    ModelType,
    MonitoringConfig,
    MonitoringReport,
    NotificationChannel,
    NotificationChannelType,
    NumaNodeMetrics,
    OperationMetrics,
    OpportunityScoring,
    OpportunityType,
    OptimizationIdentifier,
    OptimizationOpportunity,
    PatternDatabase,
    PatternOutcome,
    PerformanceAlert,
    // Analytics and trends
    PerformanceAnalyticsEngine,
    PerformanceAnomaly,
    PerformanceChart,
    PerformanceDashboard,
    PerformancePattern,
    PerformancePrediction,
    PerformancePredictor,

    PerformanceSnapshot,
    // Prediction and forecasting
    PredictionModel,
    RegressionDetector,
    ResourceMetrics,
    StorageMetrics,
    SuppressionRule,

    SystemBottleneck,
    SystemMetrics,
    TrendAnalyzer,
    TrendData,
    TrendDirection,
    TrendType,
    // Core monitoring components
    UltraPerformanceMonitor,
    WidgetConfig,
    WidgetLayout,
    WidgetType,
};
