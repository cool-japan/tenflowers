//! Integration Module for Ultra-Performance Validation
//!
//! This module provides comprehensive end-to-end validation of all performance
//! optimizations implemented across TenfloweRS, including SIMD, cache, memory,
//! and neural network optimizations.

pub mod ultra_performance_validation;

pub use ultra_performance_validation::{
    BaselinePerformance, OptimizationBreakdown, PerformanceTargets, UltraPerformanceValidator,
    ValidationReport, ValidationResult, ValidationTestSuite,
};
