//! Ultra-Performance Statistical Operations Module
//!
//! Provides highly optimized statistical operations for tensors with advanced
//! SciRS2 ecosystem integration.
//!
//! # Modules
//! - `config`: Configuration, metrics structures and global state
//! - `histogram`: Ultra-performance histogram computation
//! - `distribution`: Distribution statistics (quantile, covariance, correlation, median)
//! - `moments`: Moment-based statistics (percentile, range, skewness, kurtosis, moment)

pub mod config;
pub mod distribution;
pub mod histogram;
pub mod moments;

pub use config::{
    clear_performance_metrics, generate_performance_report, get_performance_metrics,
    get_stats_config, set_stats_config, StatisticalMetrics, StatsConfig,
};
pub use distribution::{correlation, covariance, median, quantile};
pub use histogram::histogram;
pub use moments::{kurtosis, moment, percentile, range, skewness};
