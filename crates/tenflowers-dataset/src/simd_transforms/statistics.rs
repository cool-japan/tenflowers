//! SIMD-accelerated statistical operations
//!
//! This module provides vectorized implementations of statistical computations
//! using SIMD instructions for enhanced performance.

#![allow(unsafe_code)]

use std::marker::PhantomData;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated statistical operations for tensor data
pub struct SimdStats<T> {
    use_simd: bool,
    _phantom: PhantomData<T>,
}

impl<T> SimdStats<T>
where
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
{
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && std::mem::size_of::<T>() == 4;

        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;

        Self {
            use_simd,
            _phantom: PhantomData,
        }
    }

    /// Compute mean and variance using SIMD acceleration
    pub fn mean_variance(&self, data: &[T]) -> (T, T) {
        if self.use_simd && std::mem::size_of::<T>() == 4 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                return self.mean_variance_f32_simd(std::mem::transmute::<&[T], &[f32]>(data));
            }
        }

        // Fallback to scalar implementation
        self.mean_variance_scalar(data)
    }

    /// SIMD-accelerated mean and variance for f32 data
    #[cfg(target_arch = "x86_64")]
    unsafe fn mean_variance_f32_simd(&self, data: &[f32]) -> (T, T) {
        if data.is_empty() {
            return (T::zero(), T::zero());
        }

        let len = data.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum_vec = _mm256_setzero_ps();
        let mut sum_sq_vec = _mm256_setzero_ps();

        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));

            // Accumulate sum
            sum_vec = _mm256_add_ps(sum_vec, values);

            // Accumulate sum of squares
            let squares = _mm256_mul_ps(values, values);
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, squares);
        }

        // Horizontal sum of SIMD registers
        let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum_vec);
        let sum_sq_array = std::mem::transmute::<__m256, [f32; 8]>(sum_sq_vec);

        let mut sum = sum_array.iter().sum::<f32>();
        let mut sum_sq = sum_sq_array.iter().sum::<f32>();

        // Handle remaining elements
        if remainder > 0 {
            let start = chunks * 8;
            for &val in &data[start..] {
                sum += val;
                sum_sq += val * val;
            }
        }

        let mean = sum / len as f32;
        let variance = (sum_sq / len as f32) - (mean * mean);

        (
            *std::mem::transmute::<&f32, &T>(&mean),
            *std::mem::transmute::<&f32, &T>(&variance),
        )
    }

    /// Scalar fallback for mean and variance computation
    fn mean_variance_scalar(&self, data: &[T]) -> (T, T) {
        if data.is_empty() {
            return (T::zero(), T::zero());
        }

        let len = T::from(data.len()).unwrap_or_else(|| T::from(1).unwrap_or(T::one()));
        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / len;

        let sum_sq_diff = data.iter().fold(T::zero(), |acc, &x| {
            let diff = x - mean;
            acc + diff * diff
        });
        let variance = sum_sq_diff / len;

        (mean, variance)
    }
}

impl<T> Default for SimdStats<T>
where
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `statistics.rs` had zero test coverage before this module. These tests
// drive `SimdStats::mean_variance` -- the only public entry point in this
// file. Its `unsafe` blocks (`mean_variance_f32_simd` and the `transmute`
// call sites that feed it) are gated `#[cfg(target_arch = "x86_64")]`, so
// on this crate's Miri host (aarch64-apple-darwin) that code is not
// compiled in at all and these tests exercise `mean_variance_scalar` -- the
// same code path used on any non-x86_64 host, and the only path Miri's
// interpreter can check regardless of host architecture.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_stats_new_reports_simd_status() {
        let stats = SimdStats::<f32>::new();
        let _ = stats.use_simd; // must not panic to read
    }

    #[test]
    fn test_simd_stats_default_matches_new() {
        let a = SimdStats::<f32>::default();
        let b = SimdStats::<f32>::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(a.mean_variance(&data), b.mean_variance(&data));
    }

    #[test]
    fn test_mean_variance_empty_is_zero() {
        let stats = SimdStats::<f32>::new();
        let (mean, variance) = stats.mean_variance(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(variance, 0.0);
    }

    #[test]
    fn test_mean_variance_known_values() {
        let stats = SimdStats::<f32>::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (mean, variance) = stats.mean_variance(&data);
        // mean = 3, population variance = 2
        assert!((mean - 3.0).abs() < 1e-5, "mean={mean}");
        assert!((variance - 2.0).abs() < 1e-4, "variance={variance}");
    }

    #[test]
    fn test_mean_variance_constant_data_has_zero_variance() {
        let stats = SimdStats::<f32>::new();
        let data = vec![7.0f32; 20];
        let (mean, variance) = stats.mean_variance(&data);
        assert!((mean - 7.0).abs() < 1e-5, "mean={mean}");
        assert!(variance.abs() < 1e-5, "variance={variance}");
    }

    #[test]
    fn test_mean_variance_large_batch_crosses_simd_chunk_boundary() {
        // 33 elements: crosses an 8-lane chunk boundary with a non-zero
        // remainder (`chunks = 33 / 8 = 4`, `remainder = 1`), matching the
        // exact chunking arithmetic the x86_64 SIMD path uses. On this host
        // this exercises only `mean_variance_scalar`, but the element count
        // still pins down the boundary condition this file's chunking
        // logic must handle correctly everywhere.
        let stats = SimdStats::<f32>::new();
        let data: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let (mean, variance) = stats.mean_variance(&data);

        let n = data.len() as f32;
        let expected_mean = data.iter().sum::<f32>() / n;
        let expected_variance =
            data.iter().map(|&x| (x - expected_mean).powi(2)).sum::<f32>() / n;

        assert!(
            (mean - expected_mean).abs() < 1e-3,
            "mean={mean}, expected={expected_mean}"
        );
        assert!(
            (variance - expected_variance).abs() < 1e-2,
            "variance={variance}, expected={expected_variance}"
        );
    }

    #[test]
    fn test_mean_variance_f64_uses_generic_scalar_path() {
        // f64 never qualifies for the (f32-only) SIMD fast path even on
        // x86_64 (`size_of::<T>() == 4` guard fails), so this always goes
        // through `mean_variance_scalar` on every host.
        let stats = SimdStats::<f64>::new();
        let data = vec![2.0f64, 4.0, 6.0, 8.0];
        let (mean, variance) = stats.mean_variance(&data);
        assert!((mean - 5.0).abs() < 1e-10, "mean={mean}");
        assert!((variance - 5.0).abs() < 1e-10, "variance={variance}");
    }

    #[test]
    fn test_mean_variance_single_element_has_zero_variance() {
        let stats = SimdStats::<f32>::new();
        let (mean, variance) = stats.mean_variance(&[42.0]);
        assert!((mean - 42.0).abs() < 1e-5, "mean={mean}");
        assert!(variance.abs() < 1e-5, "variance={variance}");
    }
}
