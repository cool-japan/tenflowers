//! Core statistics types and data structures
//!
//! This module defines the fundamental types used for dataset statistics
//! including DatasetStats, Histogram, and StatisticsConfig.

use std::collections::HashMap;
use std::fmt;

/// Dataset statistics container
#[derive(Debug, Clone)]
pub struct DatasetStats<T> {
    pub mean: Option<Vec<T>>,
    pub std: Option<Vec<T>>,
    pub min: Option<Vec<T>>,
    pub max: Option<Vec<T>>,
    pub histogram: Option<Histogram<T>>,
    pub class_distribution: Option<HashMap<String, usize>>,
    pub feature_count: usize,
    pub sample_count: usize,
}

/// Histogram representation
#[derive(Debug, Clone)]
pub struct Histogram<T> {
    pub bins: Vec<T>,
    pub counts: Vec<usize>,
    pub bin_edges: Vec<T>,
}

/// Configuration for statistics computation
#[derive(Debug, Clone)]
pub struct StatisticsConfig {
    pub compute_mean: bool,
    pub compute_std: bool,
    pub compute_min_max: bool,
    pub compute_histogram: bool,
    pub compute_class_distribution: bool,
    pub histogram_bins: usize,
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            compute_mean: true,
            compute_std: true,
            compute_min_max: true,
            compute_histogram: false,
            compute_class_distribution: false,
            histogram_bins: 50,
        }
    }
}

impl<T> DatasetStats<T> {
    /// Create empty dataset statistics
    pub fn new(feature_count: usize, sample_count: usize) -> Self {
        Self {
            mean: None,
            std: None,
            min: None,
            max: None,
            histogram: None,
            class_distribution: None,
            feature_count,
            sample_count,
        }
    }

    /// Get feature count
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Check if statistics are computed
    pub fn has_mean(&self) -> bool {
        self.mean.is_some()
    }

    pub fn has_std(&self) -> bool {
        self.std.is_some()
    }

    pub fn has_min_max(&self) -> bool {
        self.min.is_some() && self.max.is_some()
    }

    pub fn has_histogram(&self) -> bool {
        self.histogram.is_some()
    }

    pub fn has_class_distribution(&self) -> bool {
        self.class_distribution.is_some()
    }
}

impl<T> fmt::Display for DatasetStats<T>
where
    T: fmt::Display + Clone + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dataset Statistics:")?;
        writeln!(f, "  Samples: {}", self.sample_count)?;
        writeln!(f, "  Features: {}", self.feature_count)?;

        if let Some(ref mean) = self.mean {
            writeln!(f, "  Mean: {mean:?}")?;
        }

        if let Some(ref std) = self.std {
            writeln!(f, "  Std: {std:?}")?;
        }

        if let Some(ref min) = self.min {
            writeln!(f, "  Min: {min:?}")?;
        }

        if let Some(ref max) = self.max {
            writeln!(f, "  Max: {max:?}")?;
        }

        if let Some(ref class_dist) = self.class_distribution {
            writeln!(f, "  Class Distribution:")?;
            for (class, count) in class_dist {
                writeln!(f, "    {class}: {count}")?;
            }
        }

        Ok(())
    }
}
