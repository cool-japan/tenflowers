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

        let len = T::from(data.len()).unwrap();
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
