//! Display and formatting implementations for visualization types
//!
//! This module contains all the display methods and formatting logic
//! for visualization data structures.

use super::types::*;

impl SamplePreview {
    /// Display sample preview as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Dataset Sample Preview\n");
        output.push_str(&format!("Total samples: {}\n", self.total_samples));
        output.push_str(&format!("Samples shown: {}\n", self.samples_shown));
        output.push_str(&"─".repeat(50));
        output.push('\n');

        for sample in &self.samples {
            output.push_str(&format!(
                "Sample {}: Features{:?}, Labels{:?}\n",
                sample.index, sample.feature_shape, sample.label_shape
            ));
        }

        output
    }
}

impl<T: std::fmt::Display> DistributionInfo<T> {
    /// Display distribution information as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Dataset Distribution Analysis\n");
        output.push_str(&format!("Samples analyzed: {}\n", self.samples_analyzed));
        output.push_str(&"─".repeat(50));
        output.push('\n');

        output.push_str("Feature Statistics:\n");
        for stat in &self.feature_stats {
            output.push_str(&format!(
                "  Dim {}: mean={:.4}, std={:.4}\n",
                stat.dimension, stat.mean, stat.std_dev
            ));
        }

        output.push_str("\nLabel Statistics:\n");
        for stat in &self.label_stats {
            output.push_str(&format!(
                "  Dim {}: mean={:.4}, std={:.4}\n",
                stat.dimension, stat.mean, stat.std_dev
            ));
        }

        output
    }
}

impl ClassDistribution {
    /// Display class distribution as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Class Distribution\n");
        output.push_str(&format!("Total samples: {}\n", self.total_samples));
        output.push_str(&"─".repeat(50));
        output.push('\n');

        for (class, count) in &self.class_counts {
            let percentage = (*count as f64 / self.total_samples as f64) * 100.0;
            output.push_str(&format!("  {class}: {count} ({percentage:.1}%)\n"));
        }

        output
    }
}

impl<T: std::fmt::Display> FeatureHistogram<T> {
    /// Display histogram as text-based bar chart
    pub fn display(&self, max_bar_width: usize) -> String {
        let mut output = String::new();
        output.push_str(&format!("Feature {} Histogram\n", self.feature_index));
        output.push_str(&format!(
            "Range: {} to {}\n",
            self.min_value, self.max_value
        ));
        output.push_str(&"─".repeat(50));
        output.push('\n');

        let max_count = *self.bin_counts.iter().max().unwrap_or(&1);

        for (i, &count) in self.bin_counts.iter().enumerate() {
            let bar_length = if max_count > 0 {
                (count * max_bar_width) / max_count
            } else {
                0
            };

            let bar = "█".repeat(bar_length);
            output.push_str(&format!("  Bin {i}: {bar:<max_bar_width$} {count}\n"));
        }

        output
    }
}

impl<T: std::fmt::Display> AugmentationEffects<T> {
    /// Display augmentation effects analysis as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Augmentation Effects Analysis\n");
        output.push_str(&format!("Samples analyzed: {}\n", self.samples_analyzed));
        output.push_str(&format!(
            "Transform success rate: {:.1}%\n",
            self.transform_success_rate * 100.0
        ));
        output.push_str(&"─".repeat(50));
        output.push('\n');

        output.push_str(&self.feature_changes.display());
        output.push_str(&self.distribution_changes.display());

        output
    }
}

impl<T: std::fmt::Display> FeatureChangeAnalysis<T> {
    /// Display feature change analysis as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Feature Change Analysis:\n");
        output.push_str(&format!("  Features analyzed: {}\n", self.feature_count));
        output.push_str(&format!(
            "  Samples with changes: {}\n",
            self.samples_with_changes
        ));
        output.push_str(&format!("  Average change: {:.6}\n", self.average_change));
        output.push_str(&format!("  Max change: {:.6}\n", self.max_change));
        output.push_str(&format!("  Min change: {:.6}\n", self.min_change));
        output.push('\n');
        output
    }
}

impl<T: std::fmt::Display> DistributionChangeAnalysis<T> {
    /// Display distribution change analysis as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str("Distribution Change Analysis:\n");
        output.push_str(&format!(
            "  Original mean: {:.6}, Transformed mean: {:.6}\n",
            self.original_mean, self.transformed_mean
        ));
        output.push_str(&format!(
            "  Original std: {:.6}, Transformed std: {:.6}\n",
            self.original_std, self.transformed_std
        ));
        output.push_str(&format!("  Mean change: {:.6}\n", self.mean_change));
        output.push_str(&format!("  Std change: {:.6}\n", self.std_change));
        output.push('\n');
        output
    }
}

impl<T: std::fmt::Display> SampleComparison<T> {
    /// Display sample comparison as formatted text
    pub fn display(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Sample {} Comparison:\n", self.sample_index));
        output.push_str(&format!(
            "  Original - Mean: {:.4}, Std: {:.4}, Min: {:.4}, Max: {:.4}\n",
            self.original_stats.mean,
            self.original_stats.std,
            self.original_stats.min,
            self.original_stats.max
        ));
        output.push_str(&format!(
            "  Transformed - Mean: {:.4}, Std: {:.4}, Min: {:.4}, Max: {:.4}\n",
            self.transformed_stats.mean,
            self.transformed_stats.std,
            self.transformed_stats.min,
            self.transformed_stats.max
        ));
        output.push_str(&format!(
            "  Change magnitude (RMS): {:.6}\n",
            self.change_magnitude
        ));
        output.push('\n');
        output
    }
}
