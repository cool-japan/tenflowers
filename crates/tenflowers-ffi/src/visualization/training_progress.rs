//! Training progress visualization and monitoring
//!
//! This module provides comprehensive training progress visualization capabilities
//! including training curves, metric tracking, and performance analysis.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Training progress visualization
#[pyclass]
pub struct PyTrainingVisualizer {
    history: TrainingHistory,
}

impl Default for PyTrainingVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTrainingVisualizer {
    #[new]
    pub fn new() -> Self {
        PyTrainingVisualizer {
            history: TrainingHistory::new(),
        }
    }

    /// Record training metrics
    pub fn record_epoch(&mut self, epoch: usize, metrics: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut epoch_metrics = HashMap::new();

        for (key, value) in metrics.iter() {
            let key_str: String = key.extract()?;
            let value_f64: f64 = value.extract()?;
            epoch_metrics.insert(key_str, value_f64);
        }

        self.history.record_epoch(epoch, epoch_metrics);
        Ok(())
    }

    /// Generate training curves data
    pub fn get_training_curves(&self, py: Python) -> PyResult<PyObject> {
        let curves = self.history.get_training_curves();

        let py_dict = PyDict::new(py);
        py_dict.set_item("epochs", PyList::new(py, curves.epochs)?)?;

        let metrics_dict = PyDict::new(py);
        for (metric_name, values) in curves.metrics {
            metrics_dict.set_item(metric_name, PyList::new(py, values)?)?;
        }
        py_dict.set_item("metrics", metrics_dict)?;

        Ok(py_dict.into())
    }

    /// Get training statistics
    pub fn get_statistics(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.history.get_statistics();

        let py_dict = PyDict::new(py);
        py_dict.set_item("total_epochs", stats.total_epochs)?;
        py_dict.set_item("best_epoch", stats.best_epoch)?;
        py_dict.set_item("best_metric_value", stats.best_metric_value)?;
        py_dict.set_item("best_metric_name", &stats.best_metric_name)?;

        let convergence_dict = PyDict::new(py);
        convergence_dict.set_item("is_converged", stats.convergence_info.is_converged)?;
        convergence_dict.set_item("plateau_epochs", stats.convergence_info.plateau_epochs)?;
        convergence_dict.set_item(
            "improvement_threshold",
            stats.convergence_info.improvement_threshold,
        )?;
        py_dict.set_item("convergence", convergence_dict)?;

        Ok(py_dict.into())
    }

    /// Generate training report
    pub fn generate_report(&self, py: Python) -> PyResult<PyObject> {
        let report = self.history.generate_comprehensive_report();

        let py_dict = PyDict::new(py);
        py_dict.set_item("summary", &report.summary)?;
        py_dict.set_item("recommendations", PyList::new(py, report.recommendations)?)?;
        py_dict.set_item("warnings", PyList::new(py, report.warnings)?)?;

        // Add metric trends
        let trends_dict = PyDict::new(py);
        for (metric_name, trend) in report.metric_trends {
            let trend_dict = PyDict::new(py);
            trend_dict.set_item("direction", &trend.direction)?;
            trend_dict.set_item("stability", trend.stability)?;
            trend_dict.set_item("recent_change", trend.recent_change)?;
            trends_dict.set_item(metric_name, trend_dict)?;
        }
        py_dict.set_item("metric_trends", trends_dict)?;

        Ok(py_dict.into())
    }

    /// Clear training history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Export training data to dictionary
    pub fn export_data(&self, py: Python) -> PyResult<PyObject> {
        let data = self.history.export_data();

        let py_dict = PyDict::new(py);
        py_dict.set_item("epochs", PyList::new(py, data.epochs)?)?;

        let metrics_dict = PyDict::new(py);
        for (metric_name, values) in data.all_metrics {
            metrics_dict.set_item(metric_name, PyList::new(py, values)?)?;
        }
        py_dict.set_item("all_metrics", metrics_dict)?;

        py_dict.set_item("timestamps", PyList::new(py, data.timestamps)?)?;

        Ok(py_dict.into())
    }

    /// Set early stopping configuration
    pub fn set_early_stopping(&mut self, patience: usize, min_delta: f64, monitor: String) {
        self.history.set_early_stopping_config(EarlyStoppingConfig {
            patience,
            min_delta,
            monitor,
            best_value: None,
            wait_epochs: 0,
        });
    }

    /// Check if early stopping criteria is met
    pub fn should_stop_early(&mut self) -> bool {
        self.history.check_early_stopping()
    }

    /// Get learning rate recommendations
    pub fn get_lr_recommendations(&self) -> Vec<String> {
        self.history.analyze_learning_rate_schedule()
    }

    /// Analyze training stability
    pub fn analyze_stability(&self, py: Python) -> PyResult<PyObject> {
        let analysis = self.history.analyze_training_stability();

        let py_dict = PyDict::new(py);
        py_dict.set_item("overall_stability", analysis.overall_stability)?;
        py_dict.set_item("loss_volatility", analysis.loss_volatility)?;
        py_dict.set_item("metric_consistency", analysis.metric_consistency)?;
        py_dict.set_item(
            "recommendations",
            PyList::new(py, analysis.recommendations)?,
        )?;

        Ok(py_dict.into())
    }
}

/// Internal training history management
pub struct TrainingHistory {
    epochs: Vec<usize>,
    metrics: HashMap<String, Vec<f64>>,
    timestamps: Vec<String>,
    early_stopping_config: Option<EarlyStoppingConfig>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            metrics: HashMap::new(),
            timestamps: Vec::new(),
            early_stopping_config: None,
        }
    }

    pub fn record_epoch(&mut self, epoch: usize, epoch_metrics: HashMap<String, f64>) {
        self.epochs.push(epoch);

        // Record timestamp
        let now = chrono::Utc::now();
        self.timestamps.push(now.to_rfc3339());

        // Record metrics
        for (metric_name, value) in epoch_metrics {
            self.metrics.entry(metric_name).or_default().push(value);
        }
    }

    pub fn get_training_curves(&self) -> TrainingCurves {
        TrainingCurves {
            epochs: self.epochs.clone(),
            metrics: self.metrics.clone(),
        }
    }

    pub fn get_statistics(&self) -> TrainingStatistics {
        let total_epochs = self.epochs.len();
        let mut best_epoch = 0;
        let mut best_metric_value = f64::NEG_INFINITY;
        let mut best_metric_name = String::new();

        // Find best epoch based on primary metric (assume first metric is primary)
        if let Some((metric_name, values)) = self.metrics.iter().next() {
            best_metric_name = metric_name.clone();
            for (i, &value) in values.iter().enumerate() {
                if value > best_metric_value
                    || (metric_name.contains("loss") && value < best_metric_value.abs())
                {
                    best_metric_value = value;
                    best_epoch = i;
                }
            }
        }

        // Analyze convergence
        let convergence_info = self.analyze_convergence();

        TrainingStatistics {
            total_epochs,
            best_epoch,
            best_metric_value,
            best_metric_name,
            convergence_info,
        }
    }

    pub fn generate_comprehensive_report(&self) -> TrainingReport {
        let mut recommendations = Vec::new();
        let mut warnings = Vec::new();
        let mut metric_trends = HashMap::new();

        let summary = if self.epochs.is_empty() {
            "No training data recorded".to_string()
        } else {
            format!("Training completed {} epochs", self.epochs.len())
        };

        // Analyze each metric trend
        for (metric_name, values) in &self.metrics {
            let trend = self.analyze_metric_trend(values);

            // Generate recommendations based on trends
            if trend.direction == "increasing" && metric_name.contains("loss") {
                warnings.push(format!(
                    "Loss metric '{}' is increasing - possible overfitting",
                    metric_name
                ));
                recommendations
                    .push("Consider reducing learning rate or adding regularization".to_string());
            }

            if trend.stability < 0.5 {
                warnings.push(format!("Metric '{}' shows high volatility", metric_name));
                recommendations.push(
                    "Training appears unstable - consider adjusting hyperparameters".to_string(),
                );
            }

            // Insert trend after all analysis is complete
            metric_trends.insert(metric_name.clone(), trend);
        }

        if recommendations.is_empty() {
            recommendations.push("Training progressing normally".to_string());
        }

        TrainingReport {
            summary,
            recommendations,
            warnings,
            metric_trends,
        }
    }

    pub fn clear(&mut self) {
        self.epochs.clear();
        self.metrics.clear();
        self.timestamps.clear();
    }

    pub fn export_data(&self) -> TrainingData {
        TrainingData {
            epochs: self.epochs.clone(),
            all_metrics: self.metrics.clone(),
            timestamps: self.timestamps.clone(),
        }
    }

    pub fn set_early_stopping_config(&mut self, config: EarlyStoppingConfig) {
        self.early_stopping_config = Some(config);
    }

    pub fn check_early_stopping(&mut self) -> bool {
        if let Some(ref mut config) = self.early_stopping_config {
            if let Some(values) = self.metrics.get(&config.monitor) {
                if let Some(&current_value) = values.last() {
                    let is_improvement = if let Some(best) = config.best_value {
                        if config.monitor.contains("loss") {
                            current_value < best - config.min_delta
                        } else {
                            current_value > best + config.min_delta
                        }
                    } else {
                        true
                    };

                    if is_improvement {
                        config.best_value = Some(current_value);
                        config.wait_epochs = 0;
                        false
                    } else {
                        config.wait_epochs += 1;
                        config.wait_epochs >= config.patience
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn analyze_learning_rate_schedule(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze loss progression
        if let Some(loss_values) = self.metrics.get("loss") {
            let recent_losses = if loss_values.len() >= 10 {
                &loss_values[loss_values.len() - 10..]
            } else {
                loss_values
            };

            let avg_recent = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            let first_recent = recent_losses[0];
            let improvement = (first_recent - avg_recent) / first_recent;

            if improvement < 0.01 {
                recommendations
                    .push("Loss has plateaued - consider reducing learning rate".to_string());
            } else if improvement > 0.1 {
                recommendations.push(
                    "Good loss improvement - current learning rate appears effective".to_string(),
                );
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Insufficient data for learning rate analysis".to_string());
        }

        recommendations
    }

    pub fn analyze_training_stability(&self) -> StabilityAnalysis {
        let mut overall_stability = 1.0;
        let mut loss_volatility = 0.0;
        let mut metric_consistency = 1.0;
        let mut recommendations = Vec::new();

        if let Some(loss_values) = self.metrics.get("loss") {
            loss_volatility = self.calculate_volatility(loss_values);
            overall_stability *= (1.0 - loss_volatility).max(0.0);
        }

        // Analyze consistency across all metrics
        let mut total_consistency = 0.0;
        let mut metric_count = 0;
        for values in self.metrics.values() {
            total_consistency += 1.0 - self.calculate_volatility(values);
            metric_count += 1;
        }

        if metric_count > 0 {
            metric_consistency = total_consistency / metric_count as f64;
            overall_stability = (overall_stability + metric_consistency) / 2.0;
        }

        // Generate recommendations
        if overall_stability < 0.5 {
            recommendations.push(
                "Training shows high instability - consider reducing learning rate".to_string(),
            );
        }
        if loss_volatility > 0.5 {
            recommendations
                .push("High loss volatility detected - check for gradient clipping".to_string());
        }
        if metric_consistency < 0.7 {
            recommendations
                .push("Inconsistent metric behavior - review training configuration".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Training stability appears good".to_string());
        }

        StabilityAnalysis {
            overall_stability,
            loss_volatility,
            metric_consistency,
            recommendations,
        }
    }

    fn analyze_convergence(&self) -> ConvergenceInfo {
        // Simple convergence analysis based on loss improvement
        if let Some(loss_values) = self.metrics.get("loss") {
            if loss_values.len() < 5 {
                return ConvergenceInfo {
                    is_converged: false,
                    plateau_epochs: 0,
                    improvement_threshold: 0.001,
                };
            }

            let recent_window = &loss_values[loss_values.len().saturating_sub(5)..];
            let improvement =
                (recent_window[0] - recent_window[recent_window.len() - 1]) / recent_window[0];

            ConvergenceInfo {
                is_converged: improvement.abs() < 0.001,
                plateau_epochs: if improvement.abs() < 0.001 {
                    recent_window.len()
                } else {
                    0
                },
                improvement_threshold: 0.001,
            }
        } else {
            ConvergenceInfo {
                is_converged: false,
                plateau_epochs: 0,
                improvement_threshold: 0.001,
            }
        }
    }

    fn analyze_metric_trend(&self, values: &[f64]) -> MetricTrend {
        if values.len() < 2 {
            return MetricTrend {
                direction: "unknown".to_string(),
                stability: 1.0,
                recent_change: 0.0,
            };
        }

        let first_half_avg =
            values[..values.len() / 2].iter().sum::<f64>() / (values.len() / 2) as f64;
        let second_half_avg = values[values.len() / 2..].iter().sum::<f64>()
            / (values.len() - values.len() / 2) as f64;

        let direction = if second_half_avg > first_half_avg {
            "increasing".to_string()
        } else if second_half_avg < first_half_avg {
            "decreasing".to_string()
        } else {
            "stable".to_string()
        };

        let stability = 1.0 - self.calculate_volatility(values);
        let recent_change = if values.len() >= 2 {
            (values[values.len() - 1] - values[values.len() - 2])
                / values[values.len() - 2].abs().max(1e-8)
        } else {
            0.0
        };

        MetricTrend {
            direction,
            stability,
            recent_change,
        }
    }

    fn calculate_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Normalize by mean to get coefficient of variation
        if mean.abs() > 1e-8 {
            (std_dev / mean.abs()).min(1.0)
        } else {
            std_dev.min(1.0)
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

// Data structures for training analysis
pub struct TrainingCurves {
    pub epochs: Vec<usize>,
    pub metrics: HashMap<String, Vec<f64>>,
}

pub struct TrainingStatistics {
    pub total_epochs: usize,
    pub best_epoch: usize,
    pub best_metric_value: f64,
    pub best_metric_name: String,
    pub convergence_info: ConvergenceInfo,
}

pub struct ConvergenceInfo {
    pub is_converged: bool,
    pub plateau_epochs: usize,
    pub improvement_threshold: f64,
}

pub struct TrainingReport {
    pub summary: String,
    pub recommendations: Vec<String>,
    pub warnings: Vec<String>,
    pub metric_trends: HashMap<String, MetricTrend>,
}

pub struct MetricTrend {
    pub direction: String,
    pub stability: f64,
    pub recent_change: f64,
}

pub struct TrainingData {
    pub epochs: Vec<usize>,
    pub all_metrics: HashMap<String, Vec<f64>>,
    pub timestamps: Vec<String>,
}

pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub monitor: String,
    pub best_value: Option<f64>,
    pub wait_epochs: usize,
}

pub struct StabilityAnalysis {
    pub overall_stability: f64,
    pub loss_volatility: f64,
    pub metric_consistency: f64,
    pub recommendations: Vec<String>,
}
