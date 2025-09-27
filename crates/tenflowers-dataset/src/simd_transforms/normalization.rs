//! SIMD-accelerated normalization transforms
//!
//! This module provides vectorized implementations of data normalization
//! using SIMD instructions for significant performance improvements.

#![allow(unsafe_code)]

use crate::Transform;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated normalization transform
///
/// Uses AVX2 instructions when available for up to 8x performance improvement
/// over scalar normalization on compatible hardware.
pub struct SimdNormalize<T> {
    mean: Vec<T>,
    std: Vec<T>,
    use_simd: bool,
}

impl<T> SimdNormalize<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new SIMD-accelerated normalization transform
    pub fn new(mean: Vec<T>, std: Vec<T>) -> Self {
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && std::mem::size_of::<T>() == 4;

        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;

        Self {
            mean,
            std,
            use_simd,
        }
    }

    /// Get SIMD capability status
    pub fn is_simd_enabled(&self) -> bool {
        self.use_simd
    }

    /// SIMD-accelerated normalization for f32 data
    #[cfg(target_arch = "x86_64")]
    unsafe fn normalize_f32_simd(&self, data: &mut [f32], mean: f32, std: f32) {
        if !self.use_simd || data.len() < 8 {
            // Fall back to scalar for small arrays
            self.normalize_scalar_f32(data, mean, std);
            return;
        }

        let mean_vec = _mm256_set1_ps(mean);
        let inv_std_vec = _mm256_set1_ps(1.0 / std);

        let chunks = data.len() / 8;
        let remainder = data.len() % 8;

        // Process 8 elements at a time using AVX2
        for i in 0..chunks {
            let offset = i * 8;
            let values = _mm256_loadu_ps(data.as_ptr().add(offset));

            // Subtract mean
            let centered = _mm256_sub_ps(values, mean_vec);

            // Multiply by inverse std
            let normalized = _mm256_mul_ps(centered, inv_std_vec);

            _mm256_storeu_ps(data.as_mut_ptr().add(offset), normalized);
        }

        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start = chunks * 8;
            self.normalize_scalar_f32(&mut data[start..], mean, std);
        }
    }

    /// Scalar fallback for normalization
    fn normalize_scalar(&self, data: &mut [T], mean: T, std: T)
    where
        T: num_traits::Float,
    {
        for value in data.iter_mut() {
            *value = (*value - mean) / std;
        }
    }

    /// Scalar fallback for f32 normalization
    #[allow(dead_code)]
    fn normalize_scalar_f32(&self, data: &mut [f32], mean: f32, std: f32) {
        for value in data.iter_mut() {
            *value = (*value - mean) / std;
        }
    }
}

impl<T> Transform<T> for SimdNormalize<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        // Note: For now we'll work with immutable data and create a new tensor
        // In a real implementation, we'd need mutable tensor access
        if let Some(data) = features.as_slice() {
            let mut mutable_data = data.to_vec();
            let feature_count = self.mean.len();

            if mutable_data.len() % feature_count != 0 {
                return Err(TensorError::invalid_argument(
                    "Feature tensor size must be divisible by number of features".to_string(),
                ));
            }

            let samples = mutable_data.len() / feature_count;

            // Normalize each feature dimension
            for feature_idx in 0..feature_count {
                let mean = self.mean[feature_idx];
                let std = self.std[feature_idx];

                // Skip normalization if std is zero
                if std == T::zero() {
                    continue;
                }

                // Extract feature values across all samples
                let mut feature_values: Vec<T> = (0..samples)
                    .map(|sample_idx| mutable_data[sample_idx * feature_count + feature_idx])
                    .collect();

                // Apply SIMD normalization if available and appropriate
                #[cfg(target_arch = "x86_64")]
                {
                    if self.use_simd && std::mem::size_of::<T>() == 4 {
                        let mean_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&mean) };
                        let std_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&std) };
                        let feature_f32 = unsafe {
                            std::slice::from_raw_parts_mut(
                                feature_values.as_mut_ptr() as *mut f32,
                                feature_values.len(),
                            )
                        };

                        unsafe {
                            self.normalize_f32_simd(feature_f32, mean_f32, std_f32);
                        }
                    } else {
                        self.normalize_scalar(&mut feature_values, mean, std);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.normalize_scalar(&mut feature_values, mean, std);
                }

                // Write normalized values back
                for (sample_idx, &normalized_value) in feature_values.iter().enumerate() {
                    mutable_data[sample_idx * feature_count + feature_idx] = normalized_value;
                }
            }

            // Create new tensor with normalized data
            let new_features = Tensor::from_vec(mutable_data, features.shape().dims())?;
            Ok((new_features, labels))
        } else {
            Err(TensorError::invalid_argument(
                "Cannot access tensor data for normalization".to_string(),
            ))
        }
    }
}

/// SIMD-accelerated normalization for scalar-only operations
///
/// Simplified version that only supports scalar operations for compatibility.
pub struct SimdNormalizeScalarOnly<T> {
    _marker: PhantomData<T>,
}

impl<T> SimdNormalizeScalarOnly<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Create a new scalar-only normalization transform
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for SimdNormalizeScalarOnly<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for SimdNormalizeScalarOnly<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;

        if let Some(data) = features.as_slice() {
            // Simple z-score normalization
            let mut values = data.to_vec();
            let n = T::from(values.len()).unwrap_or(T::one());

            // Calculate mean
            let sum = values.iter().fold(T::zero(), |acc, &x| acc + x);
            let mean = sum / n;

            // Calculate standard deviation
            let variance = values
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / n;

            let std = variance.sqrt();

            // Apply normalization if std is not zero
            if std > T::zero() {
                for value in &mut values {
                    *value = (*value - mean) / std;
                }
            }

            let normalized_features = Tensor::from_vec(values, features.shape().dims())?;
            Ok((normalized_features, labels))
        } else {
            Err(TensorError::invalid_argument(
                "Cannot access tensor data for scalar normalization".to_string(),
            ))
        }
    }
}
