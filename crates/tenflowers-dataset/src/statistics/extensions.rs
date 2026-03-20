//! Extension traits for dataset statistics
//!
//! This module provides convenient trait extensions that allow datasets
//! to compute statistics directly via method calls.

use crate::Dataset;
use tenflowers_core::Result;

use super::computation::DatasetStatisticsComputer;
use super::core::{DatasetStats, StatisticsConfig};

/// Extension trait for Dataset to add statistics computation
pub trait DatasetStatisticsExt<T> {
    /// Compute statistics with default configuration
    fn compute_statistics(&self) -> Result<DatasetStats<T>>;

    /// Compute statistics with custom configuration
    fn compute_statistics_with_config(&self, config: StatisticsConfig) -> Result<DatasetStats<T>>;
}

impl<T, D> DatasetStatisticsExt<T> for D
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::Float
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
    D: Dataset<T>,
{
    fn compute_statistics(&self) -> Result<DatasetStats<T>> {
        DatasetStatisticsComputer::compute(self, StatisticsConfig::default())
    }

    fn compute_statistics_with_config(&self, config: StatisticsConfig) -> Result<DatasetStats<T>> {
        DatasetStatisticsComputer::compute(self, config)
    }
}
