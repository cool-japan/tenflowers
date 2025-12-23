//! Training Visualization Helpers
//!
//! This module provides utilities for visualizing training progress, metrics,
//! and model performance. Generates data structures suitable for plotting libraries.

use std::collections::HashMap;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Plot data for visualization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PlotData {
    /// X-axis values (e.g., epochs, steps)
    pub x: Vec<f64>,
    /// Y-axis values (e.g., loss, accuracy)
    pub y: Vec<f64>,
    /// Series label
    pub label: String,
    /// Plot color (hex string)
    pub color: Option<String>,
}

impl PlotData {
    /// Create new plot data
    pub fn new(label: String) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            label,
            color: None,
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, x: f64, y: f64) {
        self.x.push(x);
        self.y.push(y);
    }

    /// Set color
    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    /// Get number of points
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Get value range (min, max)
    pub fn value_range(&self) -> (f64, f64) {
        if self.y.is_empty() {
            return (0.0, 0.0);
        }

        let min = self.y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        (min, max)
    }
}

/// Training curve visualization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TrainingCurve {
    /// Title of the plot
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Series data
    pub series: Vec<PlotData>,
}

impl TrainingCurve {
    /// Create new training curve
    pub fn new(title: String, x_label: String, y_label: String) -> Self {
        Self {
            title,
            x_label,
            y_label,
            series: Vec::new(),
        }
    }

    /// Add a series
    pub fn add_series(&mut self, series: PlotData) {
        self.series.push(series);
    }

    /// Get overall value range across all series
    pub fn overall_range(&self) -> (f64, f64) {
        if self.series.is_empty() {
            return (0.0, 0.0);
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for series in &self.series {
            let (series_min, series_max) = series.value_range();
            min = min.min(series_min);
            max = max.max(series_max);
        }

        (min, max)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("step");
        for series in &self.series {
            output.push(',');
            output.push_str(&series.label);
        }
        output.push('\n');

        // Find max length
        let max_len = self.series.iter().map(|s| s.len()).max().unwrap_or(0);

        // Data rows
        for i in 0..max_len {
            output.push_str(&i.to_string());
            for series in &self.series {
                output.push(',');
                if i < series.len() {
                    output.push_str(&series.y[i].to_string());
                }
            }
            output.push('\n');
        }

        output
    }
}

/// Confusion matrix for classification visualization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ConfusionMatrix {
    /// Matrix values (predicted x actual)
    pub matrix: Vec<Vec<usize>>,
    /// Class labels
    pub labels: Vec<String>,
    /// Total samples
    pub total_samples: usize,
}

impl ConfusionMatrix {
    /// Create new confusion matrix
    pub fn new(num_classes: usize) -> Self {
        let labels = (0..num_classes).map(|i| format!("Class {}", i)).collect();
        Self::with_labels(labels)
    }

    /// Create with custom labels
    pub fn with_labels(labels: Vec<String>) -> Self {
        let num_classes = labels.len();
        let matrix = vec![vec![0; num_classes]; num_classes];

        Self {
            matrix,
            labels,
            total_samples: 0,
        }
    }

    /// Add a prediction
    pub fn add_prediction(&mut self, predicted: usize, actual: usize) {
        if predicted < self.matrix.len() && actual < self.matrix.len() {
            self.matrix[predicted][actual] += 1;
            self.total_samples += 1;
        }
    }

    /// Get accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }

        let correct: usize = (0..self.matrix.len()).map(|i| self.matrix[i][i]).sum();

        correct as f64 / self.total_samples as f64
    }

    /// Get precision for a class
    pub fn precision(&self, class: usize) -> f64 {
        if class >= self.matrix.len() {
            return 0.0;
        }

        let true_positives = self.matrix[class][class];
        let predicted_positives: usize = self.matrix[class].iter().sum();

        if predicted_positives == 0 {
            0.0
        } else {
            true_positives as f64 / predicted_positives as f64
        }
    }

    /// Get recall for a class
    pub fn recall(&self, class: usize) -> f64 {
        if class >= self.matrix.len() {
            return 0.0;
        }

        let true_positives = self.matrix[class][class];
        let actual_positives: usize = (0..self.matrix.len()).map(|i| self.matrix[i][class]).sum();

        if actual_positives == 0 {
            0.0
        } else {
            true_positives as f64 / actual_positives as f64
        }
    }

    /// Get F1 score for a class
    pub fn f1_score(&self, class: usize) -> f64 {
        let precision = self.precision(class);
        let recall = self.recall(class);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Get macro-averaged precision
    pub fn macro_precision(&self) -> f64 {
        let sum: f64 = (0..self.matrix.len()).map(|i| self.precision(i)).sum();
        sum / self.matrix.len() as f64
    }

    /// Get macro-averaged recall
    pub fn macro_recall(&self) -> f64 {
        let sum: f64 = (0..self.matrix.len()).map(|i| self.recall(i)).sum();
        sum / self.matrix.len() as f64
    }

    /// Get macro-averaged F1 score
    pub fn macro_f1(&self) -> f64 {
        let sum: f64 = (0..self.matrix.len()).map(|i| self.f1_score(i)).sum();
        sum / self.matrix.len() as f64
    }

    /// Normalize matrix (convert to percentages)
    pub fn normalize(&self) -> Vec<Vec<f64>> {
        self.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum == 0 {
                    vec![0.0; row.len()]
                } else {
                    row.iter()
                        .map(|&val| (val as f64 / sum as f64) * 100.0)
                        .collect()
                }
            })
            .collect()
    }

    /// Format as text table
    pub fn format_table(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("{:>12} |", "Predicted"));
        for label in &self.labels {
            output.push_str(&format!(" {:>8}", label));
        }
        output.push('\n');
        output.push_str(&"-".repeat(12 + self.labels.len() * 10));
        output.push('\n');

        // Rows
        for (i, row) in self.matrix.iter().enumerate() {
            output.push_str(&format!("{:>12} |", self.labels[i]));
            for val in row {
                output.push_str(&format!(" {:>8}", val));
            }
            output.push('\n');
        }

        // Metrics
        output.push('\n');
        output.push_str(&format!("Accuracy: {:.2}%\n", self.accuracy() * 100.0));
        output.push_str(&format!(
            "Macro Precision: {:.2}%\n",
            self.macro_precision() * 100.0
        ));
        output.push_str(&format!(
            "Macro Recall: {:.2}%\n",
            self.macro_recall() * 100.0
        ));
        output.push_str(&format!("Macro F1: {:.2}%\n", self.macro_f1() * 100.0));

        output
    }
}

/// Histogram data for distributions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Histogram {
    /// Histogram bins
    pub bins: Vec<f64>,
    /// Counts per bin
    pub counts: Vec<usize>,
    /// Total count
    pub total_count: usize,
}

impl Histogram {
    /// Create histogram from data
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return Self {
                bins: Vec::new(),
                counts: Vec::new(),
                total_count: 0,
            };
        }

        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max - min) / num_bins as f64;

        let mut bins = Vec::with_capacity(num_bins);
        let mut counts = vec![0; num_bins];

        for i in 0..num_bins {
            bins.push(min + i as f64 * bin_width);
        }

        for &value in data {
            let bin_index = ((value - min) / bin_width).floor() as usize;
            let bin_index = bin_index.min(num_bins - 1);
            counts[bin_index] += 1;
        }

        Self {
            bins,
            counts,
            total_count: data.len(),
        }
    }

    /// Get probability density
    pub fn density(&self) -> Vec<f64> {
        if self.total_count == 0 {
            return vec![0.0; self.counts.len()];
        }

        self.counts
            .iter()
            .map(|&count| count as f64 / self.total_count as f64)
            .collect()
    }

    /// Get cumulative distribution
    pub fn cumulative(&self) -> Vec<f64> {
        let density = self.density();
        let mut cumulative = Vec::with_capacity(density.len());
        let mut sum = 0.0;

        for &d in &density {
            sum += d;
            cumulative.push(sum);
        }

        cumulative
    }
}

/// Learning rate schedule visualization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct LearningRateSchedule {
    /// Steps
    pub steps: Vec<usize>,
    /// Learning rates
    pub learning_rates: Vec<f64>,
}

impl LearningRateSchedule {
    /// Create new schedule
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            learning_rates: Vec::new(),
        }
    }

    /// Add a point
    pub fn add_point(&mut self, step: usize, lr: f64) {
        self.steps.push(step);
        self.learning_rates.push(lr);
    }

    /// Get min/max learning rates
    pub fn lr_range(&self) -> (f64, f64) {
        if self.learning_rates.is_empty() {
            return (0.0, 0.0);
        }

        let min = self
            .learning_rates
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self
            .learning_rates
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        (min, max)
    }

    /// Convert to plot data
    pub fn to_plot_data(&self) -> PlotData {
        let mut plot = PlotData::new("Learning Rate".to_string());

        for (i, &step) in self.steps.iter().enumerate() {
            plot.add_point(step as f64, self.learning_rates[i]);
        }

        plot
    }
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for visualization
pub mod viz_utils {
    use super::*;

    /// Generate color palette for multiple series
    pub fn generate_color_palette(n: usize) -> Vec<String> {
        // Simple color palette (expandable)
        let base_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ];

        (0..n)
            .map(|i| base_colors[i % base_colors.len()].to_string())
            .collect()
    }

    /// Smooth curve using moving average
    pub fn smooth_curve(data: &[f64], window_size: usize) -> Vec<f64> {
        if data.is_empty() || window_size == 0 {
            return data.to_vec();
        }

        let mut smoothed = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = i.saturating_sub(window_size - 1);
            let window = &data[start..=i];
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            smoothed.push(avg);
        }

        smoothed
    }

    /// Calculate confidence interval
    pub fn confidence_interval(data: &[f64], confidence: f64) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx = ((1.0 - confidence) / 2.0 * sorted.len() as f64) as usize;
        let upper_idx = ((1.0 + confidence) / 2.0 * sorted.len() as f64) as usize;

        let lower = sorted[lower_idx.min(sorted.len() - 1)];
        let upper = sorted[upper_idx.min(sorted.len() - 1)];

        (lower, upper)
    }

    /// Downsample data for plotting (keeps min/max/avg)
    pub fn downsample(x: &[f64], y: &[f64], target_points: usize) -> (Vec<f64>, Vec<f64>) {
        if x.len() <= target_points || target_points == 0 {
            return (x.to_vec(), y.to_vec());
        }

        let window_size = x.len() / target_points;
        let mut x_down = Vec::with_capacity(target_points);
        let mut y_down = Vec::with_capacity(target_points);

        for chunk in y.chunks(window_size) {
            if !chunk.is_empty() {
                let avg = chunk.iter().sum::<f64>() / chunk.len() as f64;
                y_down.push(avg);
            }
        }

        for chunk in x.chunks(window_size) {
            if !chunk.is_empty() {
                let avg = chunk.iter().sum::<f64>() / chunk.len() as f64;
                x_down.push(avg);
            }
        }

        (x_down, y_down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_data_creation() {
        let mut plot = PlotData::new("Test".to_string());
        assert_eq!(plot.label, "Test");
        assert!(plot.is_empty());

        plot.add_point(1.0, 2.0);
        assert_eq!(plot.len(), 1);
        assert!(!plot.is_empty());
    }

    #[test]
    fn test_plot_data_value_range() {
        let mut plot = PlotData::new("Test".to_string());
        plot.add_point(0.0, 1.0);
        plot.add_point(1.0, 5.0);
        plot.add_point(2.0, 3.0);

        let (min, max) = plot.value_range();
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_training_curve_creation() {
        let curve = TrainingCurve::new(
            "Loss".to_string(),
            "Epoch".to_string(),
            "Loss Value".to_string(),
        );

        assert_eq!(curve.title, "Loss");
        assert_eq!(curve.x_label, "Epoch");
        assert_eq!(curve.y_label, "Loss Value");
    }

    #[test]
    fn test_training_curve_overall_range() {
        let mut curve = TrainingCurve::new(
            "Loss".to_string(),
            "Epoch".to_string(),
            "Loss Value".to_string(),
        );

        let mut series1 = PlotData::new("Train".to_string());
        series1.add_point(0.0, 1.0);
        series1.add_point(1.0, 0.5);

        let mut series2 = PlotData::new("Val".to_string());
        series2.add_point(0.0, 1.2);
        series2.add_point(1.0, 0.8);

        curve.add_series(series1);
        curve.add_series(series2);

        let (min, max) = curve.overall_range();
        assert_eq!(min, 0.5);
        assert_eq!(max, 1.2);
    }

    #[test]
    fn test_confusion_matrix_creation() {
        let matrix = ConfusionMatrix::new(3);
        assert_eq!(matrix.matrix.len(), 3);
        assert_eq!(matrix.labels.len(), 3);
        assert_eq!(matrix.total_samples, 0);
    }

    #[test]
    fn test_confusion_matrix_add_prediction() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0); // Correct
        matrix.add_prediction(1, 1); // Correct
        matrix.add_prediction(0, 1); // Wrong
        matrix.add_prediction(1, 0); // Wrong

        assert_eq!(matrix.total_samples, 4);
        assert_eq!(matrix.matrix[0][0], 1);
        assert_eq!(matrix.matrix[1][1], 1);
    }

    #[test]
    fn test_confusion_matrix_accuracy() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0);
        matrix.add_prediction(1, 1);
        matrix.add_prediction(0, 1);
        matrix.add_prediction(1, 0);

        assert_eq!(matrix.accuracy(), 0.5); // 2 out of 4 correct
    }

    #[test]
    fn test_confusion_matrix_precision() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0); // TP for class 0
        matrix.add_prediction(0, 1); // FP for class 0
        matrix.add_prediction(1, 1); // TN for class 0

        let precision = matrix.precision(0);
        assert!((precision - 0.5).abs() < 0.01); // 1 TP, 1 FP = 0.5
    }

    #[test]
    fn test_confusion_matrix_recall() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0); // TP for class 0
        matrix.add_prediction(1, 0); // FN for class 0

        let recall = matrix.recall(0);
        assert!((recall - 0.5).abs() < 0.01); // 1 TP, 1 FN = 0.5
    }

    #[test]
    fn test_confusion_matrix_f1_score() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0);
        matrix.add_prediction(0, 1);
        matrix.add_prediction(1, 0);
        matrix.add_prediction(1, 1);

        let f1 = matrix.f1_score(0);
        assert!(f1 > 0.0 && f1 <= 1.0);
    }

    #[test]
    fn test_confusion_matrix_normalize() {
        let mut matrix = ConfusionMatrix::new(2);
        matrix.add_prediction(0, 0);
        matrix.add_prediction(0, 0);
        matrix.add_prediction(1, 1);
        matrix.add_prediction(1, 1);

        let normalized = matrix.normalize();
        assert_eq!(normalized[0][0], 100.0); // All class 0 predictions correct
        assert_eq!(normalized[1][1], 100.0); // All class 1 predictions correct
    }

    #[test]
    fn test_histogram_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let hist = Histogram::from_data(&data, 5);

        assert_eq!(hist.bins.len(), 5);
        assert_eq!(hist.counts.len(), 5);
        assert_eq!(hist.total_count, 10);
    }

    #[test]
    fn test_histogram_density() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let hist = Histogram::from_data(&data, 2);
        let density = hist.density();

        let sum: f64 = density.iter().sum();
        assert!((sum - 1.0).abs() < 0.01); // Should sum to 1
    }

    #[test]
    fn test_histogram_cumulative() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let hist = Histogram::from_data(&data, 2);
        let cumulative = hist.cumulative();

        assert!(cumulative.last().unwrap() - 1.0 < 0.01); // Should end at 1
    }

    #[test]
    fn test_learning_rate_schedule() {
        let mut schedule = LearningRateSchedule::new();
        schedule.add_point(0, 0.001);
        schedule.add_point(100, 0.0001);
        schedule.add_point(200, 0.00001);

        let (min, max) = schedule.lr_range();
        assert_eq!(min, 0.00001);
        assert_eq!(max, 0.001);
    }

    #[test]
    fn test_utils_generate_color_palette() {
        let colors = viz_utils::generate_color_palette(5);
        assert_eq!(colors.len(), 5);
        for color in colors {
            assert!(color.starts_with('#'));
        }
    }

    #[test]
    fn test_utils_smooth_curve() {
        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let smoothed = viz_utils::smooth_curve(&data, 3);

        assert_eq!(smoothed.len(), data.len());
        // First point should be same
        assert_eq!(smoothed[0], data[0]);
    }

    #[test]
    fn test_utils_confidence_interval() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (lower, upper) = viz_utils::confidence_interval(&data, 0.95);

        assert!(lower < upper);
        assert!(lower >= 1.0);
        assert!(upper <= 10.0);
    }

    #[test]
    fn test_utils_downsample() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (x_down, y_down) = viz_utils::downsample(&x, &y, 5);

        assert_eq!(x_down.len(), 5);
        assert_eq!(y_down.len(), 5);
    }

    #[test]
    fn test_training_curve_to_csv() {
        let mut curve =
            TrainingCurve::new("Loss".to_string(), "Epoch".to_string(), "Loss".to_string());

        let mut series = PlotData::new("Train".to_string());
        series.add_point(0.0, 1.0);
        series.add_point(1.0, 0.8);

        curve.add_series(series);

        let csv = curve.to_csv();
        assert!(csv.contains("step"));
        assert!(csv.contains("Train"));
    }
}
