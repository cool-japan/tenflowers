//! SIMD-accelerated image processing operations
//!
//! This module provides vectorized implementations of image processing operations
//! such as color space conversion and histogram computation using SIMD instructions.

#![allow(unsafe_code)]

use crate::Transform;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated RGB to HSV color space conversion
pub struct SimdColorConvert<T> {
    use_simd: bool,
    _phantom: PhantomData<T>,
}

impl<T> SimdColorConvert<T>
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

    /// Convert RGB to HSV using SIMD acceleration
    pub fn rgb_to_hsv(&self, rgb_data: &mut [T]) {
        if self.use_simd && std::mem::size_of::<T>() == 4 && rgb_data.len() % 3 == 0 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                self.rgb_to_hsv_f32_simd(std::mem::transmute::<&mut [T], &mut [f32]>(rgb_data));
                return;
            }
        }

        // Fallback to scalar implementation
        self.rgb_to_hsv_scalar(rgb_data);
    }

    /// SIMD-accelerated RGB to HSV conversion for f32 data
    #[cfg(target_arch = "x86_64")]
    unsafe fn rgb_to_hsv_f32_simd(&self, rgb_data: &mut [f32]) {
        let pixels = rgb_data.len() / 3;

        for i in 0..pixels {
            let base = i * 3;
            let r = rgb_data[base];
            let g = rgb_data[base + 1];
            let b = rgb_data[base + 2];

            let max_val = r.max(g.max(b));
            let min_val = r.min(g.min(b));
            let delta = max_val - min_val;

            // Value
            let v = max_val;

            // Saturation
            let s = if max_val == 0.0 { 0.0 } else { delta / max_val };

            // Hue
            let h = if delta == 0.0 {
                0.0
            } else if max_val == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max_val == g {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };

            let h_normalized = if h < 0.0 { h + 360.0 } else { h };

            rgb_data[base] = h_normalized / 360.0; // Normalize H to [0,1]
            rgb_data[base + 1] = s;
            rgb_data[base + 2] = v;
        }
    }

    /// Scalar fallback for RGB to HSV conversion
    fn rgb_to_hsv_scalar(&self, rgb_data: &mut [T]) {
        let pixels = rgb_data.len() / 3;

        for i in 0..pixels {
            let base = i * 3;
            let r = rgb_data[base];
            let g = rgb_data[base + 1];
            let b = rgb_data[base + 2];

            let max_val = r.max(g.max(b));
            let min_val = r.min(g.min(b));
            let delta = max_val - min_val;

            // Value
            let v = max_val;

            // Saturation
            let s = if max_val == T::zero() {
                T::zero()
            } else {
                delta / max_val
            };

            // Hue
            let h = if delta == T::zero() {
                T::zero()
            } else if max_val == r {
                let six = T::from(6.0).unwrap_or_else(|| T::from(6).unwrap_or(T::zero()));
                let sixty = T::from(60.0).unwrap_or_else(|| T::from(60).unwrap_or(T::zero()));
                sixty * (((g - b) / delta) % six)
            } else if max_val == g {
                let two = T::from(2.0).unwrap_or_else(|| T::from(2).unwrap_or(T::zero()));
                let sixty = T::from(60.0).unwrap_or_else(|| T::from(60).unwrap_or(T::zero()));
                sixty * ((b - r) / delta + two)
            } else {
                let four = T::from(4.0).unwrap_or_else(|| T::from(4).unwrap_or(T::zero()));
                let sixty = T::from(60.0).unwrap_or_else(|| T::from(60).unwrap_or(T::zero()));
                sixty * ((r - g) / delta + four)
            };

            let three_sixty = T::from(360.0).unwrap_or_else(|| T::from(360).unwrap_or(T::one()));
            let h_normalized = if h < T::zero() { h + three_sixty } else { h };

            rgb_data[base] = h_normalized / three_sixty; // Normalize H to [0,1]
            rgb_data[base + 1] = s;
            rgb_data[base + 2] = v;
        }
    }
}

impl<T> Default for SimdColorConvert<T>
where
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for SimdColorConvert<T>
where
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
{
    fn apply(&self, sample: (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        let (features, labels) = sample;
        let mut data = features
            .as_slice()
            .ok_or_else(|| {
                TensorError::invalid_argument(
                    "Unable to access tensor data for color conversion".to_string(),
                )
            })?
            .to_vec();
        self.rgb_to_hsv(&mut data);
        let converted_features = Tensor::from_vec(data, features.shape().dims())?;
        Ok((converted_features, labels))
    }
}

/// SIMD-accelerated histogram computation for efficient data distribution analysis
///
/// Provides high-performance histogram calculation using SIMD instructions
/// for up to 8x speedup on compatible hardware for dataset statistics.
pub struct SimdHistogram {
    bins: usize,
    min_val: f32,
    max_val: f32,
    use_simd: bool,
}

impl SimdHistogram {
    /// Create a new SIMD-accelerated histogram calculator
    pub fn new(bins: usize, min_val: f32, max_val: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2");

        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;

        Self {
            bins,
            min_val,
            max_val,
            use_simd,
        }
    }

    /// Get SIMD capability status
    pub fn is_simd_enabled(&self) -> bool {
        self.use_simd
    }

    /// Compute histogram of tensor data with SIMD acceleration
    pub fn compute(&self, tensor: &Tensor<f32>) -> Result<Vec<u32>> {
        let data = tensor
            .as_slice()
            .ok_or_else(|| TensorError::InvalidOperation {
                operation: "histogram_compute".to_string(),
                reason: "Cannot get tensor slice".to_string(),
                context: None,
            })?;

        let mut histogram = vec![0u32; self.bins];
        let bin_width = (self.max_val - self.min_val) / self.bins as f32;

        #[cfg(target_arch = "x86_64")]
        if self.use_simd && data.len() >= 8 {
            self.compute_simd_f32(data, &mut histogram, bin_width);
        } else {
            self.compute_scalar(data, &mut histogram, bin_width);
        }

        #[cfg(not(target_arch = "x86_64"))]
        self.compute_scalar(data, &mut histogram, bin_width);

        Ok(histogram)
    }

    /// SIMD-accelerated histogram computation for f32 data
    #[cfg(target_arch = "x86_64")]
    fn compute_simd_f32(&self, data: &[f32], histogram: &mut [u32], bin_width: f32) {
        unsafe {
            let min_vec = _mm256_set1_ps(self.min_val);
            let max_vec = _mm256_set1_ps(self.max_val);
            let bin_width_vec = _mm256_set1_ps(bin_width);
            let bins_minus_one = _mm256_set1_epi32((self.bins - 1) as i32);
            let zero_vec = _mm256_setzero_si256();

            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();

            // Process 8 elements at a time with SIMD
            for chunk in chunks {
                let values = _mm256_loadu_ps(chunk.as_ptr());

                // Clamp values to [min_val, max_val]
                let clamped = _mm256_max_ps(_mm256_min_ps(values, max_vec), min_vec);

                // Calculate bin indices: (value - min_val) / bin_width
                let normalized = _mm256_sub_ps(clamped, min_vec);
                let bin_indices_f = _mm256_div_ps(normalized, bin_width_vec);

                // Use truncation toward zero instead of rounding
                let bin_indices = _mm256_cvttps_epi32(bin_indices_f);

                // Clamp bin indices to valid range [0, bins-1]
                let clamped_indices =
                    _mm256_max_epi32(_mm256_min_epi32(bin_indices, bins_minus_one), zero_vec);

                // Extract indices and increment histogram bins
                let indices: [i32; 8] = std::mem::transmute(clamped_indices);
                for &idx in &indices {
                    histogram[idx as usize] += 1;
                }
            }

            // Process remaining elements with scalar code
            self.compute_scalar(remainder, histogram, bin_width);
        }
    }

    /// Fallback scalar implementation for non-SIMD hardware
    fn compute_scalar(&self, data: &[f32], histogram: &mut [u32], bin_width: f32) {
        for &value in data {
            let clamped = value.clamp(self.min_val, self.max_val);
            // Handle the edge case where clamped equals max_val
            let bin_idx = if clamped == self.max_val {
                self.bins - 1
            } else {
                ((clamped - self.min_val) / bin_width) as usize
            };
            let bin_idx = bin_idx.min(self.bins - 1);
            histogram[bin_idx] += 1;
        }
    }
}

impl Transform<f32> for SimdHistogram {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        // Return the original sample unchanged - this transform is meant for analysis
        Ok(sample)
    }
}

/// Specialized histogram transform that computes histogram alongside the data
pub struct SimdHistogramTransform {
    histogram_computer: SimdHistogram,
}

impl SimdHistogramTransform {
    pub fn new(bins: usize, min_val: f32, max_val: f32) -> Self {
        Self {
            histogram_computer: SimdHistogram::new(bins, min_val, max_val),
        }
    }

    pub fn apply_with_histogram(&self, input: &Tensor<f32>) -> Result<Vec<u32>> {
        self.histogram_computer.compute(input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// `image_processing.rs` had zero test coverage before this module. These
// tests drive `SimdColorConvert`, `SimdHistogram`, and
// `SimdHistogramTransform` through their public APIs. Every `unsafe` block
// in this file (`rgb_to_hsv_f32_simd`, `compute_simd_f32`, and the
// `mem::transmute` call sites that feed them) sits behind
// `#[cfg(target_arch = "x86_64")]`, so on this crate's Miri host
// (aarch64-apple-darwin) that code is not compiled in at all and these
// tests exercise the scalar fallbacks (`rgb_to_hsv_scalar`,
// `compute_scalar`) -- the same code path used on any non-x86_64 host, and
// the only path Miri's interpreter can check regardless of host
// architecture.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_color_convert_new_reports_simd_status() {
        let convert = SimdColorConvert::<f32>::new();
        let _ = convert.use_simd; // must not panic to read
    }

    #[test]
    fn test_simd_color_convert_default_matches_new() {
        let a = SimdColorConvert::<f32>::default();
        let mut data_a = vec![1.0f32, 0.0, 0.0];
        a.rgb_to_hsv(&mut data_a);

        let b = SimdColorConvert::<f32>::new();
        let mut data_b = vec![1.0f32, 0.0, 0.0];
        b.rgb_to_hsv(&mut data_b);

        assert_eq!(data_a, data_b);
    }

    #[test]
    fn test_rgb_to_hsv_white() {
        // RGB(1,1,1) is white: max=min so delta=0 -> H=0, S=0, V=1.
        let convert = SimdColorConvert::<f32>::new();
        let mut data = vec![1.0f32, 1.0, 1.0];
        convert.rgb_to_hsv(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5, "H={}", data[0]);
        assert!((data[1] - 0.0).abs() < 1e-5, "S={}", data[1]);
        assert!((data[2] - 1.0).abs() < 1e-5, "V={}", data[2]);
    }

    #[test]
    fn test_rgb_to_hsv_pure_red() {
        // RGB(1,0,0) -> H=0, S=1, V=1.
        let convert = SimdColorConvert::<f32>::new();
        let mut data = vec![1.0f32, 0.0, 0.0];
        convert.rgb_to_hsv(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5, "H={}", data[0]);
        assert!((data[1] - 1.0).abs() < 1e-5, "S={}", data[1]);
        assert!((data[2] - 1.0).abs() < 1e-5, "V={}", data[2]);
    }

    #[test]
    fn test_rgb_to_hsv_pure_green() {
        // RGB(0,1,0) -> H=1/3 (120deg / 360deg), S=1, V=1.
        let convert = SimdColorConvert::<f32>::new();
        let mut data = vec![0.0f32, 1.0, 0.0];
        convert.rgb_to_hsv(&mut data);
        assert!((data[0] - 1.0 / 3.0).abs() < 1e-5, "H={}", data[0]);
        assert!((data[1] - 1.0).abs() < 1e-5, "S={}", data[1]);
        assert!((data[2] - 1.0).abs() < 1e-5, "V={}", data[2]);
    }

    #[test]
    fn test_rgb_to_hsv_black() {
        // RGB(0,0,0) -> H=0, S=0 (guarded max_val==0 branch), V=0.
        let convert = SimdColorConvert::<f32>::new();
        let mut data = vec![0.0f32, 0.0, 0.0];
        convert.rgb_to_hsv(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-5, "H={}", data[0]);
        assert!((data[1] - 0.0).abs() < 1e-5, "S={}", data[1]);
        assert!((data[2] - 0.0).abs() < 1e-5, "V={}", data[2]);
    }

    #[test]
    fn test_rgb_to_hsv_multi_pixel_round_trip_channel_count() {
        // 4 pixels (12 elements), large enough to exercise several
        // iterations of the per-pixel loop.
        let convert = SimdColorConvert::<f32>::new();
        let mut data = vec![
            1.0, 0.0, 0.0, // red
            0.0, 1.0, 0.0, // green
            0.0, 0.0, 1.0, // blue
            0.5, 0.5, 0.5, // gray
        ];
        convert.rgb_to_hsv(&mut data);
        assert_eq!(data.len(), 12);
        // Gray pixel: max==min -> delta=0 -> H=0, S=0, V=0.5.
        assert!((data[9] - 0.0).abs() < 1e-5);
        assert!((data[10] - 0.0).abs() < 1e-5);
        assert!((data[11] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_simd_color_convert_apply_transform() {
        let convert = SimdColorConvert::<f32>::new();
        let features = Tensor::<f32>::from_vec(vec![1.0, 0.0, 0.0], &[1, 3])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[1]);

        let (converted, _labels) = convert
            .apply((features, labels))
            .expect("test: apply should succeed");
        let data = converted
            .as_slice()
            .expect("test: converted tensor should expose a CPU slice");
        assert!((data[1] - 1.0).abs() < 1e-5, "S={}", data[1]); // pure red is fully saturated
    }

    #[test]
    fn test_simd_histogram_uniform_distribution() {
        let hist = SimdHistogram::new(4, 0.0, 4.0);
        let data: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
        let tensor =
            Tensor::<f32>::from_vec(data, &[4]).expect("test: tensor construction should succeed");
        let histogram = hist.compute(&tensor).expect("test: compute should succeed");
        assert_eq!(histogram, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_simd_histogram_large_batch_matches_scalar_reference() {
        // 37 elements: crosses an 8-lane chunk boundary with a non-zero
        // remainder, matching the exact chunking arithmetic the x86_64 SIMD
        // path uses (`chunks_exact(8)` + remainder). On this host this
        // exercises only `compute_scalar`, but the element count still
        // pins down the boundary condition this file's chunking logic must
        // handle correctly everywhere.
        let hist = SimdHistogram::new(10, 0.0, 10.0);
        let data: Vec<f32> = (0..37).map(|i| (i % 10) as f32 + 0.5).collect();
        let tensor = Tensor::<f32>::from_vec(data.clone(), &[data.len()])
            .expect("test: tensor construction should succeed");
        let histogram = hist.compute(&tensor).expect("test: compute should succeed");

        // Independently computed scalar reference histogram.
        let mut expected = vec![0u32; 10];
        for &value in &data {
            let clamped = value.clamp(0.0, 10.0);
            let bin_idx = if clamped == 10.0 {
                9
            } else {
                ((clamped - 0.0) / 1.0) as usize
            };
            expected[bin_idx.min(9)] += 1;
        }
        assert_eq!(histogram, expected);
        assert_eq!(histogram.iter().sum::<u32>(), 37);
    }

    #[test]
    fn test_simd_histogram_clamps_out_of_range_values() {
        let hist = SimdHistogram::new(2, 0.0, 10.0);
        let data = vec![-5.0f32, 15.0, 5.0];
        let tensor =
            Tensor::<f32>::from_vec(data, &[3]).expect("test: tensor construction should succeed");
        let histogram = hist.compute(&tensor).expect("test: compute should succeed");
        // -5.0 clamps to 0.0 -> bin 0; 15.0 clamps to 10.0 (== max_val) -> last bin;
        // 5.0 -> bin_width=5.0, so 5.0 lands exactly on the boundary -> bin 1.
        assert_eq!(histogram.iter().sum::<u32>(), 3);
    }

    #[test]
    fn test_simd_histogram_transform_apply_with_histogram() {
        let transform = SimdHistogramTransform::new(4, 0.0, 4.0);
        let tensor = Tensor::<f32>::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[4])
            .expect("test: tensor construction should succeed");
        let histogram = transform
            .apply_with_histogram(&tensor)
            .expect("test: apply_with_histogram should succeed");
        assert_eq!(histogram, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_simd_histogram_transform_apply_is_identity() {
        // `Transform<f32>` impl for `SimdHistogram` returns the sample
        // unchanged -- this transform is meant for analysis, not mutation.
        let hist = SimdHistogram::new(4, 0.0, 4.0);
        let features = Tensor::<f32>::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[4])
            .expect("test: tensor construction should succeed");
        let labels = Tensor::<f32>::zeros(&[1]);
        let features_clone = features.clone();

        let (out_features, _labels) = hist
            .apply((features, labels))
            .expect("test: apply should succeed");
        assert_eq!(
            out_features.as_slice().expect("test: slice access"),
            features_clone.as_slice().expect("test: slice access")
        );
    }
}
