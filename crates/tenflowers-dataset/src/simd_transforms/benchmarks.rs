//! SIMD performance benchmarking utilities
//!
//! This module provides benchmarking tools to compare SIMD vs scalar performance
//! for data transformation operations.

use crate::simd_transforms::{SimdNormalize, SimdNormalizeScalarOnly};
use crate::transforms::Transform;
use tenflowers_core::{Result, Tensor};

/// SIMD benchmark utilities
pub struct SimdBenchmark;

impl SimdBenchmark {
    /// Benchmark SIMD vs scalar normalization performance
    pub fn benchmark_normalization<T>() -> Result<BenchmarkResult>
    where
        T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
    {
        use std::time::{Duration, Instant};

        let size = 10000;
        let features = 100;
        let iterations = 10; // Run multiple iterations for accurate measurement

        let mean: Vec<T> = (0..features).map(|_| T::zero()).collect();
        let std: Vec<T> = (0..features).map(|_| T::one()).collect();

        // Create test data
        let test_data: Vec<T> = (0..size * features)
            .map(|i| T::from(i as f64 / 1000.0).unwrap())
            .collect();

        let shape = vec![size, features];

        // Benchmark SIMD version with multiple iterations
        let simd_transform = SimdNormalize::new(mean.clone(), std.clone());
        let mut simd_total_duration = Duration::ZERO;

        for _ in 0..iterations {
            let tensor = Tensor::from_vec(test_data.clone(), &shape)?;
            let label_tensor = Tensor::from_vec(vec![T::zero(); size], &[size])?;
            let sample = (tensor, label_tensor);

            let simd_start = Instant::now();
            let _simd_result = simd_transform.apply(sample)?;
            simd_total_duration += simd_start.elapsed();
        }
        let simd_duration = simd_total_duration / iterations as u32;

        // Benchmark scalar-only version with multiple iterations
        let scalar_transform = SimdNormalizeScalarOnly::new();
        let mut scalar_total_duration = Duration::ZERO;

        for _ in 0..iterations {
            let tensor = Tensor::from_vec(test_data.clone(), &shape)?;
            let label_tensor = Tensor::from_vec(vec![T::zero(); size], &[size])?;
            let sample = (tensor, label_tensor);

            let scalar_start = Instant::now();
            let _scalar_result = scalar_transform.apply(sample)?;
            scalar_total_duration += scalar_start.elapsed();
        }
        let scalar_duration = scalar_total_duration / iterations as u32;

        // Calculate speedup factor
        let speedup_factor = if simd_duration.as_nanos() > 0 {
            scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64
        } else {
            1.0
        };

        Ok(BenchmarkResult {
            simd_duration,
            scalar_duration,
            speedup_factor,
            data_size: size * features,
            simd_enabled: simd_transform.is_simd_enabled(),
        })
    }
}

/// Results from SIMD performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub simd_duration: std::time::Duration,
    pub scalar_duration: std::time::Duration,
    pub speedup_factor: f64,
    pub data_size: usize,
    pub simd_enabled: bool,
}

impl BenchmarkResult {
    /// Display benchmark results in a human-readable format
    pub fn display(&self) {
        println!("SIMD Transform Benchmark Results:");
        println!("  Data size: {} elements", self.data_size);
        println!("  SIMD enabled: {}", self.simd_enabled);
        if self.simd_enabled {
            println!("  SIMD duration: {:?}", self.simd_duration);
            println!("  Scalar duration: {:?}", self.scalar_duration);
            println!("  Speedup factor: {:.2}x", self.speedup_factor);
        } else {
            println!("  SIMD not available on this platform");
            println!("  Scalar duration: {:?}", self.scalar_duration);
        }
    }
}
