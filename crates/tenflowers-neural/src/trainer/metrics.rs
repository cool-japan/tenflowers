//! Training metrics and state management
//!
//! This module provides comprehensive data structures and functionality for tracking
//! training progress, metrics, and state throughout the training process.
//!
//! Features:
//! - Real-time metric tracking with aggregation
//! - Statistical analysis (mean, std, percentiles)
//! - Moving averages and smoothing
//! - Early stopping detection
//! - Metric comparison and validation
//! - Export utilities for visualization

use std::collections::HashMap;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Training metrics and history
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub metrics: HashMap<String, f32>,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(epoch: usize, step: usize, loss: f32) -> Self {
        Self {
            epoch,
            step,
            loss,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the current metrics
    pub fn add_metric(&mut self, name: String, value: f32) {
        self.metrics.insert(name, value);
    }

    /// Get a metric value by name
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).copied()
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<&String> {
        self.metrics.keys().collect()
    }
}

/// Training state and configuration
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub best_metric: Option<f32>,
    pub history: Vec<TrainingMetrics>,
    pub val_history: Vec<TrainingMetrics>,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_metric: None,
            history: Vec::new(),
            val_history: Vec::new(),
        }
    }

    /// Add training metrics to history
    pub fn add_training_metrics(&mut self, metrics: TrainingMetrics) {
        self.history.push(metrics);
    }

    /// Add validation metrics to history
    pub fn add_validation_metrics(&mut self, metrics: TrainingMetrics) {
        self.val_history.push(metrics);
    }

    /// Update best metric if the current one is better
    pub fn update_best_metric(&mut self, metric: f32, higher_is_better: bool) -> bool {
        match self.best_metric {
            None => {
                self.best_metric = Some(metric);
                true
            }
            Some(best) => {
                let is_better = if higher_is_better {
                    metric > best
                } else {
                    metric < best
                };

                if is_better {
                    self.best_metric = Some(metric);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get the latest training loss
    pub fn latest_train_loss(&self) -> Option<f32> {
        self.history.last().map(|m| m.loss)
    }

    /// Get the latest validation loss
    pub fn latest_val_loss(&self) -> Option<f32> {
        self.val_history.last().map(|m| m.loss)
    }

    /// Get training loss history
    pub fn train_loss_history(&self) -> Vec<f32> {
        self.history.iter().map(|m| m.loss).collect()
    }

    /// Get validation loss history
    pub fn val_loss_history(&self) -> Vec<f32> {
        self.val_history.iter().map(|m| m.loss).collect()
    }

    /// Reset training state
    pub fn reset(&mut self) {
        self.epoch = 0;
        self.step = 0;
        self.best_metric = None;
        self.history.clear();
        self.val_history.clear();
    }

    /// Increment epoch
    pub fn next_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Increment step
    pub fn next_step(&mut self) {
        self.step += 1;
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Callback action returned by callbacks to control training behavior
#[derive(Debug, Clone, PartialEq)]
pub enum CallbackAction {
    Continue,                // Continue training normally
    Stop,                    // Stop training
    ReduceLearningRate(f32), // Reduce learning rate by the given factor
    SaveModel(String),       // Save model to the given filepath
}

// ============================================================================
// Advanced Metric Utilities
// ============================================================================

/// Moving average tracker for metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MovingAverage {
    window_size: usize,
    values: Vec<f32>,
    sum: f32,
}

impl MovingAverage {
    /// Create new moving average with given window size
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: Vec::new(),
            sum: 0.0,
        }
    }

    /// Add a value and return the current moving average
    pub fn add(&mut self, value: f32) -> f32 {
        self.values.push(value);
        self.sum += value;

        if self.values.len() > self.window_size {
            let removed = self.values.remove(0);
            self.sum -= removed;
        }

        self.average()
    }

    /// Get current moving average
    pub fn average(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f32
        }
    }

    /// Reset the moving average
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }

    /// Get number of values currently tracked
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Exponential moving average tracker
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ExponentialMovingAverage {
    alpha: f32,
    value: Option<f32>,
}

impl ExponentialMovingAverage {
    /// Create new exponential moving average with smoothing factor alpha
    /// alpha should be in (0, 1], where lower values = more smoothing
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            value: None,
        }
    }

    /// Add a value and return the current EMA
    pub fn add(&mut self, new_value: f32) -> f32 {
        match self.value {
            None => {
                self.value = Some(new_value);
                new_value
            }
            Some(current) => {
                let updated = self.alpha * new_value + (1.0 - self.alpha) * current;
                self.value = Some(updated);
                updated
            }
        }
    }

    /// Get current EMA value
    pub fn value(&self) -> Option<f32> {
        self.value
    }

    /// Reset the EMA
    pub fn reset(&mut self) {
        self.value = None;
    }
}

/// Statistical summary of metric values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MetricStatistics {
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub p25: f32, // 25th percentile
    pub p75: f32, // 75th percentile
    pub p95: f32, // 95th percentile
    pub p99: f32, // 99th percentile
}

impl MetricStatistics {
    /// Compute statistics from a slice of values
    pub fn from_values(values: &[f32]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let mean = values.iter().sum::<f32>() / count as f32;

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[count - 1];
        let median = Self::percentile(&sorted, 0.5);
        let p25 = Self::percentile(&sorted, 0.25);
        let p75 = Self::percentile(&sorted, 0.75);
        let p95 = Self::percentile(&sorted, 0.95);
        let p99 = Self::percentile(&sorted, 0.99);

        Some(Self {
            count,
            mean,
            std,
            min,
            max,
            median,
            p25,
            p75,
            p95,
            p99,
        })
    }

    /// Calculate percentile from sorted values
    fn percentile(sorted_values: &[f32], p: f32) -> f32 {
        let n = sorted_values.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted_values[0];
        }

        let index = p * (n - 1) as f32;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        let fraction = index - lower_index as f32;

        sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction
    }

    /// Get interquartile range (IQR)
    pub fn iqr(&self) -> f32 {
        self.p75 - self.p25
    }

    /// Check if a value is an outlier using IQR method
    pub fn is_outlier(&self, value: f32) -> bool {
        let iqr = self.iqr();
        let lower_bound = self.p25 - 1.5 * iqr;
        let upper_bound = self.p75 + 1.5 * iqr;
        value < lower_bound || value > upper_bound
    }

    /// Get coefficient of variation (std / mean)
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean.abs() > f32::EPSILON {
            self.std / self.mean.abs()
        } else {
            0.0
        }
    }
}

/// Metric aggregator for collecting and analyzing metrics over time
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct MetricAggregator {
    name: String,
    values: Vec<f32>,
    moving_avg: Option<MovingAverage>,
    ema: Option<ExponentialMovingAverage>,
    max_history: Option<usize>,
}

impl MetricAggregator {
    /// Create new metric aggregator
    pub fn new(name: String) -> Self {
        Self {
            name,
            values: Vec::new(),
            moving_avg: None,
            ema: None,
            max_history: None,
        }
    }

    /// Enable moving average with window size
    pub fn with_moving_average(mut self, window_size: usize) -> Self {
        self.moving_avg = Some(MovingAverage::new(window_size));
        self
    }

    /// Enable exponential moving average with smoothing factor
    pub fn with_ema(mut self, alpha: f32) -> Self {
        self.ema = Some(ExponentialMovingAverage::new(alpha));
        self
    }

    /// Set maximum history size (older values are discarded)
    pub fn with_max_history(mut self, max_size: usize) -> Self {
        self.max_history = Some(max_size);
        self
    }

    /// Add a new value
    pub fn add(&mut self, value: f32) {
        self.values.push(value);

        if let Some(ma) = &mut self.moving_avg {
            ma.add(value);
        }

        if let Some(ema) = &mut self.ema {
            ema.add(value);
        }

        // Trim history if needed
        if let Some(max_size) = self.max_history {
            if self.values.len() > max_size {
                let remove_count = self.values.len() - max_size;
                self.values.drain(0..remove_count);
            }
        }
    }

    /// Get all values
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get latest value
    pub fn latest(&self) -> Option<f32> {
        self.values.last().copied()
    }

    /// Get moving average
    pub fn moving_average(&self) -> Option<f32> {
        self.moving_avg.as_ref().map(|ma| ma.average())
    }

    /// Get exponential moving average
    pub fn exponential_moving_average(&self) -> Option<f32> {
        self.ema.as_ref().and_then(|ema| ema.value())
    }

    /// Get statistics
    pub fn statistics(&self) -> Option<MetricStatistics> {
        MetricStatistics::from_values(&self.values)
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get count of values
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Check if improved over last N values (lower is better)
    pub fn improved_recently(&self, lookback: usize, lower_is_better: bool) -> bool {
        if self.values.len() < lookback + 1 {
            return false;
        }

        let recent = &self.values[self.values.len() - lookback..];
        let previous = self.values[self.values.len() - lookback - 1];
        let recent_avg = recent.iter().sum::<f32>() / recent.len() as f32;

        if lower_is_better {
            recent_avg < previous
        } else {
            recent_avg > previous
        }
    }

    /// Reset aggregator
    pub fn reset(&mut self) {
        self.values.clear();
        if let Some(ma) = &mut self.moving_avg {
            ma.reset();
        }
        if let Some(ema) = &mut self.ema {
            ema.reset();
        }
    }
}

/// Early stopping monitor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct EarlyStoppingMonitor {
    patience: usize,
    min_delta: f32,
    lower_is_better: bool,
    best_value: Option<f32>,
    wait_counter: usize,
    stopped: bool,
}

impl EarlyStoppingMonitor {
    /// Create new early stopping monitor
    pub fn new(patience: usize, min_delta: f32, lower_is_better: bool) -> Self {
        Self {
            patience,
            min_delta,
            lower_is_better,
            best_value: None,
            wait_counter: 0,
            stopped: false,
        }
    }

    /// Update with new value, returns true if should stop
    pub fn update(&mut self, value: f32) -> bool {
        if self.stopped {
            return true;
        }

        let is_improvement = match self.best_value {
            None => true,
            Some(best) => {
                if self.lower_is_better {
                    value < best - self.min_delta
                } else {
                    value > best + self.min_delta
                }
            }
        };

        if is_improvement {
            self.best_value = Some(value);
            self.wait_counter = 0;
            false
        } else {
            self.wait_counter += 1;
            if self.wait_counter >= self.patience {
                self.stopped = true;
                true
            } else {
                false
            }
        }
    }

    /// Get best value seen so far
    pub fn best_value(&self) -> Option<f32> {
        self.best_value
    }

    /// Get current wait counter
    pub fn wait_counter(&self) -> usize {
        self.wait_counter
    }

    /// Check if stopped
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Reset monitor
    pub fn reset(&mut self) {
        self.best_value = None;
        self.wait_counter = 0;
        self.stopped = false;
    }
}

/// Metric comparison result
#[derive(Debug, Clone, PartialEq)]
pub enum MetricComparison {
    Improved,
    Degraded,
    NoChange,
}

/// Utilities for metric analysis
pub mod utils {
    use super::*;

    /// Compare two metrics with tolerance
    pub fn compare_metrics(
        current: f32,
        baseline: f32,
        tolerance: f32,
        lower_is_better: bool,
    ) -> MetricComparison {
        let diff = if lower_is_better {
            baseline - current
        } else {
            current - baseline
        };

        if diff > tolerance {
            MetricComparison::Improved
        } else if diff < -tolerance {
            MetricComparison::Degraded
        } else {
            MetricComparison::NoChange
        }
    }

    /// Calculate percentage change
    pub fn percentage_change(current: f32, baseline: f32) -> f32 {
        if baseline.abs() < f32::EPSILON {
            0.0
        } else {
            ((current - baseline) / baseline.abs()) * 100.0
        }
    }

    /// Smooth values using simple moving average
    pub fn smooth_values(values: &[f32], window_size: usize) -> Vec<f32> {
        if values.is_empty() || window_size == 0 {
            return values.to_vec();
        }

        let mut smoothed = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            let start = i.saturating_sub(window_size - 1);
            let window = &values[start..=i];
            let avg = window.iter().sum::<f32>() / window.len() as f32;
            smoothed.push(avg);
        }

        smoothed
    }

    /// Detect anomalies using z-score method
    pub fn detect_anomalies(values: &[f32], threshold: f32) -> Vec<usize> {
        if values.len() < 2 {
            return Vec::new();
        }

        let stats = match MetricStatistics::from_values(values) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut anomalies = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            let z_score = (value - stats.mean).abs() / stats.std;
            if z_score > threshold {
                anomalies.push(i);
            }
        }

        anomalies
    }

    /// Calculate trend (positive = increasing, negative = decreasing)
    pub fn calculate_trend(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < f32::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_creation() {
        let mut metrics = TrainingMetrics::new(1, 100, 0.5);
        assert_eq!(metrics.epoch, 1);
        assert_eq!(metrics.step, 100);
        assert_eq!(metrics.loss, 0.5);
        assert!(metrics.metrics.is_empty());

        metrics.add_metric("accuracy".to_string(), 0.95);
        assert_eq!(metrics.get_metric("accuracy"), Some(0.95));
    }

    #[test]
    fn test_training_state_best_metric_update() {
        let mut state = TrainingState::new();

        // First metric is always best
        assert!(state.update_best_metric(0.8, true)); // higher is better
        assert_eq!(state.best_metric, Some(0.8));

        // Better metric
        assert!(state.update_best_metric(0.9, true));
        assert_eq!(state.best_metric, Some(0.9));

        // Worse metric
        assert!(!state.update_best_metric(0.7, true));
        assert_eq!(state.best_metric, Some(0.9));
    }

    #[test]
    fn test_training_state_history() {
        let mut state = TrainingState::new();

        let train_metrics = TrainingMetrics::new(1, 100, 0.5);
        state.add_training_metrics(train_metrics);

        let val_metrics = TrainingMetrics::new(1, 100, 0.4);
        state.add_validation_metrics(val_metrics);

        assert_eq!(state.latest_train_loss(), Some(0.5));
        assert_eq!(state.latest_val_loss(), Some(0.4));
    }

    // ========================================================================
    // Advanced Metrics Tests
    // ========================================================================

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);

        assert_eq!(ma.add(1.0), 1.0);
        assert_eq!(ma.add(2.0), 1.5);
        assert_eq!(ma.add(3.0), 2.0);
        assert_eq!(ma.add(4.0), 3.0); // Window slides: [2, 3, 4]
        assert_eq!(ma.add(5.0), 4.0); // Window slides: [3, 4, 5]

        assert_eq!(ma.len(), 3);
    }

    #[test]
    fn test_moving_average_reset() {
        let mut ma = MovingAverage::new(3);
        ma.add(1.0);
        ma.add(2.0);

        ma.reset();
        assert!(ma.is_empty());
        assert_eq!(ma.average(), 0.0);
    }

    #[test]
    fn test_exponential_moving_average() {
        let mut ema = ExponentialMovingAverage::new(0.5);

        assert_eq!(ema.add(1.0), 1.0);
        assert_eq!(ema.add(3.0), 2.0); // 0.5 * 3.0 + 0.5 * 1.0
        assert_eq!(ema.add(5.0), 3.5); // 0.5 * 5.0 + 0.5 * 2.0
    }

    #[test]
    fn test_ema_smoothing() {
        let mut ema_high = ExponentialMovingAverage::new(0.9); // Less smoothing
        let mut ema_low = ExponentialMovingAverage::new(0.1); // More smoothing

        let values = vec![1.0, 10.0];

        for &val in &values {
            ema_high.add(val);
            ema_low.add(val);
        }

        let high_val = ema_high.value().unwrap();
        let low_val = ema_low.value().unwrap();

        // High alpha responds more to recent values
        assert!(high_val > low_val);
    }

    #[test]
    fn test_metric_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = MetricStatistics::from_values(&values).unwrap();

        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.01);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.median, 5.5);
    }

    #[test]
    fn test_metric_statistics_empty() {
        let values: Vec<f32> = vec![];
        assert!(MetricStatistics::from_values(&values).is_none());
    }

    #[test]
    fn test_iqr_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = MetricStatistics::from_values(&values).unwrap();

        let iqr = stats.iqr();
        assert!(iqr > 0.0);
        assert_eq!(iqr, stats.p75 - stats.p25);
    }

    #[test]
    fn test_outlier_detection() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]; // 100 is outlier
        let stats = MetricStatistics::from_values(&values).unwrap();

        assert!(stats.is_outlier(100.0));
        assert!(!stats.is_outlier(5.0));
    }

    #[test]
    fn test_coefficient_of_variation() {
        let values = vec![10.0, 10.0, 10.0];
        let stats = MetricStatistics::from_values(&values).unwrap();
        assert!((stats.coefficient_of_variation()).abs() < 0.01); // Should be near 0

        let values2 = vec![1.0, 5.0, 10.0];
        let stats2 = MetricStatistics::from_values(&values2).unwrap();
        assert!(stats2.coefficient_of_variation() > 0.0);
    }

    #[test]
    fn test_metric_aggregator_basic() {
        let mut agg = MetricAggregator::new("loss".to_string());

        agg.add(1.0);
        agg.add(2.0);
        agg.add(3.0);

        assert_eq!(agg.count(), 3);
        assert_eq!(agg.latest(), Some(3.0));
        assert_eq!(agg.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_metric_aggregator_with_moving_average() {
        let mut agg = MetricAggregator::new("loss".to_string()).with_moving_average(2);

        agg.add(1.0);
        agg.add(3.0);
        agg.add(5.0);

        assert_eq!(agg.moving_average(), Some(4.0)); // (3 + 5) / 2
    }

    #[test]
    fn test_metric_aggregator_with_ema() {
        let mut agg = MetricAggregator::new("loss".to_string()).with_ema(0.5);

        agg.add(1.0);
        agg.add(3.0);

        assert_eq!(agg.exponential_moving_average(), Some(2.0));
    }

    #[test]
    fn test_metric_aggregator_max_history() {
        let mut agg = MetricAggregator::new("loss".to_string()).with_max_history(3);

        agg.add(1.0);
        agg.add(2.0);
        agg.add(3.0);
        agg.add(4.0); // Should remove 1.0
        agg.add(5.0); // Should remove 2.0

        assert_eq!(agg.count(), 3);
        assert_eq!(agg.values(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_metric_aggregator_improvement() {
        let mut agg = MetricAggregator::new("loss".to_string());

        agg.add(10.0);
        agg.add(9.0);
        agg.add(8.0);
        agg.add(7.0);

        assert!(agg.improved_recently(2, true)); // Lower is better, and trending down
    }

    #[test]
    fn test_metric_aggregator_statistics() {
        let mut agg = MetricAggregator::new("loss".to_string());

        agg.add(1.0);
        agg.add(2.0);
        agg.add(3.0);

        let stats = agg.statistics().unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 2.0);
    }

    #[test]
    fn test_early_stopping_monitor_patience() {
        let mut monitor = EarlyStoppingMonitor::new(3, 0.01, true);

        assert!(!monitor.update(1.0)); // Best so far
        assert!(!monitor.update(1.1)); // Worse, counter = 1
        assert!(!monitor.update(1.2)); // Worse, counter = 2
        assert!(monitor.update(1.3)); // Worse, counter = 3, should stop

        assert!(monitor.is_stopped());
    }

    #[test]
    fn test_early_stopping_monitor_improvement() {
        let mut monitor = EarlyStoppingMonitor::new(3, 0.01, true);

        assert!(!monitor.update(1.0)); // Best so far
        assert!(!monitor.update(1.1)); // Worse, counter = 1
        assert!(!monitor.update(0.9)); // Better! Reset counter

        assert_eq!(monitor.wait_counter(), 0);
        assert_eq!(monitor.best_value(), Some(0.9));
    }

    #[test]
    fn test_early_stopping_monitor_min_delta() {
        let mut monitor = EarlyStoppingMonitor::new(2, 0.1, true);

        assert!(!monitor.update(1.0)); // Best so far
        assert!(!monitor.update(0.95)); // Not enough improvement (< 0.1)

        assert_eq!(monitor.wait_counter(), 1);
    }

    #[test]
    fn test_early_stopping_higher_is_better() {
        let mut monitor = EarlyStoppingMonitor::new(2, 0.01, false); // Higher is better

        assert!(!monitor.update(0.5)); // Best so far
        assert!(!monitor.update(0.6)); // Better
        assert!(!monitor.update(0.5)); // Worse, counter = 1

        assert_eq!(monitor.best_value(), Some(0.6));
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut monitor = EarlyStoppingMonitor::new(2, 0.01, true);

        monitor.update(1.0);
        monitor.update(1.1);
        monitor.update(1.2);

        assert!(monitor.is_stopped());

        monitor.reset();
        assert!(!monitor.is_stopped());
        assert_eq!(monitor.best_value(), None);
        assert_eq!(monitor.wait_counter(), 0);
    }

    #[test]
    fn test_compare_metrics_improved() {
        let result = utils::compare_metrics(0.8, 0.9, 0.05, true); // Lower is better
        assert_eq!(result, MetricComparison::Improved);
    }

    #[test]
    fn test_compare_metrics_degraded() {
        let result = utils::compare_metrics(0.9, 0.8, 0.05, true); // Lower is better
        assert_eq!(result, MetricComparison::Degraded);
    }

    #[test]
    fn test_compare_metrics_no_change() {
        let result = utils::compare_metrics(0.8, 0.81, 0.05, true); // Within tolerance
        assert_eq!(result, MetricComparison::NoChange);
    }

    #[test]
    fn test_percentage_change() {
        let change = utils::percentage_change(110.0, 100.0);
        assert!((change - 10.0).abs() < 0.01);

        let change2 = utils::percentage_change(90.0, 100.0);
        assert!((change2 + 10.0).abs() < 0.01);
    }

    #[test]
    fn test_percentage_change_zero_baseline() {
        let change = utils::percentage_change(10.0, 0.0);
        assert_eq!(change, 0.0); // Should handle division by zero
    }

    #[test]
    fn test_smooth_values() {
        let values = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let smoothed = utils::smooth_values(&values, 3);

        assert_eq!(smoothed.len(), values.len());
        assert_eq!(smoothed[0], 1.0); // Window [1]
        assert_eq!(smoothed[1], 3.0); // Window [1, 5]
        assert_eq!(smoothed[2], 3.0); // Window [1, 5, 3]
        assert_eq!(smoothed[3], 5.0); // Window [5, 3, 7]
    }

    #[test]
    fn test_smooth_values_empty() {
        let values: Vec<f32> = vec![];
        let smoothed = utils::smooth_values(&values, 3);
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_detect_anomalies() {
        let values = vec![1.0, 2.0, 3.0, 2.5, 2.0, 3.0, 100.0, 2.0]; // 100.0 is anomaly
        let anomalies = utils::detect_anomalies(&values, 2.5); // 2.5 sigma threshold (more sensitive)

        assert!(!anomalies.is_empty());
        assert!(anomalies.contains(&6)); // Index of 100.0
    }

    #[test]
    fn test_detect_anomalies_none() {
        let values = vec![1.0, 2.0, 3.0, 2.5, 2.0, 3.0, 2.0];
        let anomalies = utils::detect_anomalies(&values, 2.5);

        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_calculate_trend_increasing() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = utils::calculate_trend(&values);

        assert!(trend > 0.0); // Positive slope = increasing
    }

    #[test]
    fn test_calculate_trend_decreasing() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = utils::calculate_trend(&values);

        assert!(trend < 0.0); // Negative slope = decreasing
    }

    #[test]
    fn test_calculate_trend_flat() {
        let values = vec![3.0, 3.0, 3.0, 3.0];
        let trend = utils::calculate_trend(&values);

        assert!((trend).abs() < 0.01); // Should be near 0
    }

    #[test]
    fn test_calculate_trend_insufficient_data() {
        let values = vec![1.0];
        let trend = utils::calculate_trend(&values);

        assert_eq!(trend, 0.0);
    }
}
