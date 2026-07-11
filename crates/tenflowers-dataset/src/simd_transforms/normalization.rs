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
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
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
        T: scirs2_core::numeric::Float,
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
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
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
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests exercise `SimdNormalize`/`SimdNormalizeScalarOnly` end-to-end
// through the public `Transform::apply` API. On non-x86_64 hosts (including
// this crate's Miri host, aarch64-apple-darwin) `use_simd` is always `false`
// and the `#[cfg(target_arch = "x86_64")]`-gated `unsafe` blocks are not
// even compiled in, so these tests specifically validate the scalar
// fallback path -- which is also exactly what runs on any x86_64 host
// lacking AVX2, and is the only path Miri can interpret regardless of host
// architecture. Prior to this test module, `normalization.rs` had zero
// test coverage of any kind.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_normalize_new_reports_simd_status() {
        let normalize = SimdNormalize::<f32>::new(vec![0.0], vec![1.0]);
        // On non-x86_64 hosts SIMD is never enabled; on x86_64 it depends on
        // runtime AVX2 detection. Either way this must not panic, and the
        // getter must be a plain read of the flag set at construction time.
        let _ = normalize.is_simd_enabled();
    }

    #[test]
    fn test_simd_normalize_apply_single_feature() {
        let normalize = SimdNormalize::<f32>::new(vec![2.0], vec![2.0]);
        let features = Tensor::<f32>::from_vec(vec![0.0, 2.0, 4.0, 6.0], &[4, 1])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[4, 1]);

        let (normalized, _labels) = normalize
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = normalized
            .as_slice()
            .expect("test: normalized tensor should expose a CPU slice");

        // (value - mean) / std for mean=2, std=2
        let expected = [-1.0f32, 0.0, 1.0, 2.0];
        for (actual, expected) in data.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_simd_normalize_apply_multi_feature_large_batch() {
        // Large enough (>= 8 samples per feature) to exercise the same
        // sample-count regime the x86_64 SIMD path would chunk over in 8s,
        // just via the scalar fallback on this host.
        let normalize = SimdNormalize::<f32>::new(vec![10.0, -5.0], vec![2.0, 0.5]);
        let samples = 32usize;
        let feature_count = 2usize;
        let mut raw = Vec::with_capacity(samples * feature_count);
        for i in 0..samples {
            raw.push(10.0 + i as f32); // feature 0
            raw.push(-5.0 + 0.5 * i as f32); // feature 1
        }
        let features = Tensor::<f32>::from_vec(raw, &[samples, feature_count])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[samples, 1]);

        let (normalized, _labels) = normalize
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = normalized
            .as_slice()
            .expect("test: normalized tensor should expose a CPU slice");
        assert_eq!(data.len(), samples * feature_count);

        for i in 0..samples {
            let f0 = data[i * feature_count];
            let f1 = data[i * feature_count + 1];
            assert!(
                (f0 - i as f32 / 2.0).abs() < 1e-4,
                "sample {i} feature 0: {f0}"
            );
            assert!(
                (f1 - i as f32).abs() < 1e-4,
                "sample {i} feature 1: {f1}"
            );
        }
    }

    #[test]
    fn test_simd_normalize_apply_zero_std_skips_feature() {
        // std == 0 must skip normalization for that feature entirely
        // (division by zero would otherwise produce NaN/inf).
        let normalize = SimdNormalize::<f32>::new(vec![0.0], vec![0.0]);
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[3, 1]);

        let (normalized, _labels) = normalize
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = normalized
            .as_slice()
            .expect("test: normalized tensor should expose a CPU slice");
        assert_eq!(data, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_simd_normalize_apply_rejects_indivisible_length() {
        let normalize = SimdNormalize::<f32>::new(vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]);
        // 4 elements is not divisible by 3 features.
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[1]);

        let result = normalize.apply((features, labels));
        assert!(result.is_err());
    }

    #[test]
    fn test_simd_normalize_f64_apply() {
        // f64 never takes the (f32-only) SIMD path even on x86_64, but must
        // still be correct via the generic scalar fallback.
        let normalize = SimdNormalize::<f64>::new(vec![1.0], vec![4.0]);
        let features = Tensor::<f64>::from_vec(vec![1.0, 5.0, 9.0], &[3, 1])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f64>::zeros(&[3, 1]);

        let (normalized, _labels) = normalize
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = normalized
            .as_slice()
            .expect("test: normalized tensor should expose a CPU slice");
        assert_eq!(data, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_simd_normalize_scalar_only_apply() {
        let normalize = SimdNormalizeScalarOnly::<f32>::new();
        let features = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[1]);

        let (normalized, _labels) = normalize
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = normalized
            .as_slice()
            .expect("test: normalized tensor should expose a CPU slice");

        // mean = 3, population variance = 2 -> std = sqrt(2)
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "post-normalize mean={mean}");
    }

    #[test]
    fn test_simd_normalize_scalar_only_default_matches_new() {
        let a = SimdNormalizeScalarOnly::<f32>::default();
        let b = SimdNormalizeScalarOnly::<f32>::new();
        let features_a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: tensor construction should succeed");
        let features_b = features_a.clone();
        let labels = Tensor::<f32>::zeros(&[1]);

        let out_a = a
            .apply((features_a, labels.clone()))
            .expect("test: apply should succeed");
        let out_b = b
            .apply((features_b, labels))
            .expect("test: apply should succeed");
        assert_eq!(
            out_a.0.as_slice().expect("test: slice access"),
            out_b.0.as_slice().expect("test: slice access")
        );
    }
}
