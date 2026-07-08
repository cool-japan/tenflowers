//! Free-function SIMD-accelerated data preprocessing transforms.
//!
//! All compute-intensive paths use 8-element unrolled loops that LLVM/rustc
//! auto-vectorizes to AVX/SSE/NEON without requiring unsafe intrinsics.
//! The pattern is:
//!
//! ```text
//! const LANE: usize = 8;
//! let full = data.len() / LANE * LANE;
//! // process 8 at a time ...
//! // then scalar tail from full..data.len()
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by SIMD-accelerated free-function transforms.
#[derive(Debug, Error)]
pub enum SIMDTransformError {
    /// Length mismatch between two slices (or between slice and expected value).
    #[error("Length mismatch: expected {expected}, got {got} for {operation}")]
    LengthMismatch {
        expected: usize,
        got: usize,
        operation: &'static str,
    },
    /// Shape-level invariant violated (e.g. `h * w * c != src.len()`).
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    /// Standard deviation is zero, so normalization would divide by zero.
    #[error("Division by zero in normalization (zero std deviation)")]
    ZeroStd,
    /// The input slice contains no elements.
    #[error("Empty input")]
    EmptyInput,
}

// ---------------------------------------------------------------------------
// simd_normalize_chw
// ---------------------------------------------------------------------------

/// Normalize tensor data by subtracting per-channel mean and dividing by
/// per-channel standard deviation.
///
/// Supports two layouts:
/// - **CHW** (`is_chw = true`):  `data` is `[C, H, W]`; elements for channel
///   `c` occupy `data[c * spatial .. (c+1) * spatial]`.
/// - **HWC** (`is_chw = false`): `data` is `[H, W, C]`; element `[h, w, c]`
///   is at `data[(h * w_dim + w) * C + c]`.
///
/// `channel_means` and `channel_stds` must each have length `n_channels`.
///
/// # Errors
///
/// Returns [`SIMDTransformError::ZeroStd`] if any `channel_stds[c]` is zero.
/// Returns [`SIMDTransformError::LengthMismatch`] if `data.len()` is not
/// divisible by `n_channels` (HWC) or if means/stds lengths differ from
/// `n_channels`.
pub fn simd_normalize_chw(
    data: &mut [f32],
    channel_means: &[f32],
    channel_stds: &[f32],
    n_channels: usize,
    is_chw: bool,
) -> Result<(), SIMDTransformError> {
    if data.is_empty() {
        return Err(SIMDTransformError::EmptyInput);
    }
    if channel_means.len() != n_channels {
        return Err(SIMDTransformError::LengthMismatch {
            expected: n_channels,
            got: channel_means.len(),
            operation: "simd_normalize_chw/means",
        });
    }
    if channel_stds.len() != n_channels {
        return Err(SIMDTransformError::LengthMismatch {
            expected: n_channels,
            got: channel_stds.len(),
            operation: "simd_normalize_chw/stds",
        });
    }
    for (c, &s) in channel_stds.iter().enumerate() {
        if s == 0.0 {
            return Err(SIMDTransformError::ZeroStd);
        }
        let _ = c; // suppress potential unused warning
    }

    if is_chw {
        // CHW layout: data[c * spatial + i]
        let total = data.len();
        if total % n_channels != 0 {
            return Err(SIMDTransformError::LengthMismatch {
                expected: total - total % n_channels,
                got: total,
                operation: "simd_normalize_chw/chw_layout",
            });
        }
        let spatial = total / n_channels;
        for c in 0..n_channels {
            let mean = channel_means[c];
            let inv_std = 1.0 / channel_stds[c];
            let start = c * spatial;
            let end = start + spatial;
            let slice = &mut data[start..end];

            // 8-element unrolled loop
            const LANE: usize = 8;
            let full = slice.len() / LANE * LANE;
            let mut i = 0usize;
            while i < full {
                slice[i] = (slice[i] - mean) * inv_std;
                slice[i + 1] = (slice[i + 1] - mean) * inv_std;
                slice[i + 2] = (slice[i + 2] - mean) * inv_std;
                slice[i + 3] = (slice[i + 3] - mean) * inv_std;
                slice[i + 4] = (slice[i + 4] - mean) * inv_std;
                slice[i + 5] = (slice[i + 5] - mean) * inv_std;
                slice[i + 6] = (slice[i + 6] - mean) * inv_std;
                slice[i + 7] = (slice[i + 7] - mean) * inv_std;
                i += LANE;
            }
            for elem in slice.iter_mut().skip(full) {
                *elem = (*elem - mean) * inv_std;
            }
        }
    } else {
        // HWC layout: data[(h*W + w)*C + c]
        let total = data.len();
        if total % n_channels != 0 {
            return Err(SIMDTransformError::LengthMismatch {
                expected: total - total % n_channels,
                got: total,
                operation: "simd_normalize_chw/hwc_layout",
            });
        }
        // Precompute inv_stds
        let inv_stds: Vec<f32> = channel_stds.iter().map(|&s| 1.0 / s).collect();

        // Process each pixel (group of n_channels elements)
        let n_pixels = total / n_channels;
        let mut base = 0usize;
        for _ in 0..n_pixels {
            for c in 0..n_channels {
                data[base + c] = (data[base + c] - channel_means[c]) * inv_stds[c];
            }
            base += n_channels;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// simd_standardize
// ---------------------------------------------------------------------------

/// Z-score standardize a flat `f32` slice in place: `x ← (x − mean) / std`.
///
/// The mean and standard deviation are computed from `data` itself using
/// a numerically stable two-pass algorithm.
///
/// # Returns
///
/// `(mean, std)` computed from the input.
///
/// # Errors
///
/// Returns [`SIMDTransformError::EmptyInput`] for zero-length input.
/// Returns [`SIMDTransformError::ZeroStd`] when all elements are equal
/// (standard deviation is zero).
pub fn simd_standardize(data: &mut [f32]) -> Result<(f32, f32), SIMDTransformError> {
    if data.is_empty() {
        return Err(SIMDTransformError::EmptyInput);
    }
    if data.len() == 1 {
        // Single element: std = 0 → cannot standardize.
        return Err(SIMDTransformError::ZeroStd);
    }

    let (mean, variance) = simd_mean_variance(data);
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Err(SIMDTransformError::ZeroStd);
    }

    let inv_std = 1.0 / std_dev;
    simd_scale_offset_inplace(data, inv_std, -mean * inv_std);

    Ok((mean, std_dev))
}

// ---------------------------------------------------------------------------
// simd_scale_offset_inplace
// ---------------------------------------------------------------------------

/// Apply `out[i] = data[i] * scale + offset` in place using an 8-element
/// unrolled loop.
pub fn simd_scale_offset_inplace(data: &mut [f32], scale: f32, offset: f32) {
    const LANE: usize = 8;
    let full = data.len() / LANE * LANE;
    let mut i = 0usize;
    while i < full {
        data[i] = data[i] * scale + offset;
        data[i + 1] = data[i + 1] * scale + offset;
        data[i + 2] = data[i + 2] * scale + offset;
        data[i + 3] = data[i + 3] * scale + offset;
        data[i + 4] = data[i + 4] * scale + offset;
        data[i + 5] = data[i + 5] * scale + offset;
        data[i + 6] = data[i + 6] * scale + offset;
        data[i + 7] = data[i + 7] * scale + offset;
        i += LANE;
    }
    for elem in data.iter_mut().skip(full) {
        *elem = *elem * scale + offset;
    }
}

// ---------------------------------------------------------------------------
// simd_clamp_inplace
// ---------------------------------------------------------------------------

/// Clamp all elements to `[min_val, max_val]` in place using an 8-element
/// unrolled loop.
pub fn simd_clamp_inplace(data: &mut [f32], min_val: f32, max_val: f32) {
    const LANE: usize = 8;
    let full = data.len() / LANE * LANE;
    let mut i = 0usize;
    while i < full {
        data[i] = data[i].clamp(min_val, max_val);
        data[i + 1] = data[i + 1].clamp(min_val, max_val);
        data[i + 2] = data[i + 2].clamp(min_val, max_val);
        data[i + 3] = data[i + 3].clamp(min_val, max_val);
        data[i + 4] = data[i + 4].clamp(min_val, max_val);
        data[i + 5] = data[i + 5].clamp(min_val, max_val);
        data[i + 6] = data[i + 6].clamp(min_val, max_val);
        data[i + 7] = data[i + 7].clamp(min_val, max_val);
        i += LANE;
    }
    for elem in data.iter_mut().skip(full) {
        *elem = elem.clamp(min_val, max_val);
    }
}

// ---------------------------------------------------------------------------
// simd_hwc_to_chw
// ---------------------------------------------------------------------------

/// Convert a flat `[H, W, C]` layout to `[C, H, W]` layout.
///
/// Element at `src[row * w * c + col * c + ch]` maps to
/// `dst[ch * h * w + row * w + col]`.
///
/// # Errors
///
/// Returns [`SIMDTransformError::LengthMismatch`] if `src.len() != h * w * c`
/// or `dst.len() != h * w * c`.
pub fn simd_hwc_to_chw(
    src: &[f32],
    dst: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
) -> Result<(), SIMDTransformError> {
    let expected = h * w * c;
    if src.len() != expected {
        return Err(SIMDTransformError::LengthMismatch {
            expected,
            got: src.len(),
            operation: "simd_hwc_to_chw/src",
        });
    }
    if dst.len() != expected {
        return Err(SIMDTransformError::LengthMismatch {
            expected,
            got: dst.len(),
            operation: "simd_hwc_to_chw/dst",
        });
    }
    if expected == 0 {
        return Ok(());
    }

    let spatial = h * w;

    // For each channel, copy scattered HWC elements into contiguous CHW slice.
    // Inner loop over spatial positions uses 8-element unrolling.
    for ch in 0..c {
        let dst_start = ch * spatial;
        let dst_ch = &mut dst[dst_start..dst_start + spatial];

        // Unrolled gather: process 8 spatial positions at a time
        const LANE: usize = 8;
        let full = spatial / LANE * LANE;
        let mut s = 0usize; // spatial index
        while s < full {
            dst_ch[s] = src[(s) * c + ch];
            dst_ch[s + 1] = src[(s + 1) * c + ch];
            dst_ch[s + 2] = src[(s + 2) * c + ch];
            dst_ch[s + 3] = src[(s + 3) * c + ch];
            dst_ch[s + 4] = src[(s + 4) * c + ch];
            dst_ch[s + 5] = src[(s + 5) * c + ch];
            dst_ch[s + 6] = src[(s + 6) * c + ch];
            dst_ch[s + 7] = src[(s + 7) * c + ch];
            s += LANE;
        }
        for r in full..spatial {
            dst_ch[r] = src[r * c + ch];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// simd_chw_to_hwc
// ---------------------------------------------------------------------------

/// Convert a flat `[C, H, W]` layout to `[H, W, C]` layout.
///
/// Element at `src[ch * h * w + row * w + col]` maps to
/// `dst[row * w * c + col * c + ch]`.
///
/// # Errors
///
/// Returns [`SIMDTransformError::LengthMismatch`] if `src.len() != c * h * w`
/// or `dst.len() != c * h * w`.
pub fn simd_chw_to_hwc(
    src: &[f32],
    dst: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
) -> Result<(), SIMDTransformError> {
    let expected = c * h * w;
    if src.len() != expected {
        return Err(SIMDTransformError::LengthMismatch {
            expected,
            got: src.len(),
            operation: "simd_chw_to_hwc/src",
        });
    }
    if dst.len() != expected {
        return Err(SIMDTransformError::LengthMismatch {
            expected,
            got: dst.len(),
            operation: "simd_chw_to_hwc/dst",
        });
    }
    if expected == 0 {
        return Ok(());
    }

    let spatial = h * w;

    // For each channel, scatter its contiguous CHW slice into HWC positions.
    for ch in 0..c {
        let src_start = ch * spatial;
        let src_ch = &src[src_start..src_start + spatial];

        const LANE: usize = 8;
        let full = spatial / LANE * LANE;
        let mut s = 0usize;
        while s < full {
            dst[(s) * c + ch] = src_ch[s];
            dst[(s + 1) * c + ch] = src_ch[s + 1];
            dst[(s + 2) * c + ch] = src_ch[s + 2];
            dst[(s + 3) * c + ch] = src_ch[s + 3];
            dst[(s + 4) * c + ch] = src_ch[s + 4];
            dst[(s + 5) * c + ch] = src_ch[s + 5];
            dst[(s + 6) * c + ch] = src_ch[s + 6];
            dst[(s + 7) * c + ch] = src_ch[s + 7];
            s += LANE;
        }
        for r in full..spatial {
            dst[r * c + ch] = src_ch[r];
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// simd_u8_to_f32_normalized
// ---------------------------------------------------------------------------

/// Convert a `u8` slice `[0, 255]` to a `f32` slice `[0.0, 1.0]` in place.
///
/// Uses an 8-element unrolled loop with the constant factor `1.0 / 255.0`.
///
/// # Errors
///
/// Returns [`SIMDTransformError::LengthMismatch`] if `src.len() != dst.len()`.
pub fn simd_u8_to_f32_normalized(src: &[u8], dst: &mut [f32]) -> Result<(), SIMDTransformError> {
    if src.len() != dst.len() {
        return Err(SIMDTransformError::LengthMismatch {
            expected: src.len(),
            got: dst.len(),
            operation: "simd_u8_to_f32_normalized",
        });
    }

    const INV_255: f32 = 1.0 / 255.0;
    const LANE: usize = 8;
    let n = src.len();
    let full = n / LANE * LANE;
    let mut i = 0usize;
    while i < full {
        dst[i] = src[i] as f32 * INV_255;
        dst[i + 1] = src[i + 1] as f32 * INV_255;
        dst[i + 2] = src[i + 2] as f32 * INV_255;
        dst[i + 3] = src[i + 3] as f32 * INV_255;
        dst[i + 4] = src[i + 4] as f32 * INV_255;
        dst[i + 5] = src[i + 5] as f32 * INV_255;
        dst[i + 6] = src[i + 6] as f32 * INV_255;
        dst[i + 7] = src[i + 7] as f32 * INV_255;
        i += LANE;
    }
    for j in full..n {
        dst[j] = src[j] as f32 * INV_255;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// simd_rgb_to_grayscale
// ---------------------------------------------------------------------------

/// Convert RGB data to grayscale using BT.601 coefficients:
/// `Y = 0.299·R + 0.587·G + 0.114·B`.
///
/// Two layouts are supported:
///
/// - **HWC / interleaved** (`is_chw = false`): `src` is length `total_pixels * 3`,
///   with pixels laid out as `[R₀,G₀,B₀, R₁,G₁,B₁, …]`.
///   `dst` must have length `total_pixels`.
///
/// - **CHW / planar** (`is_chw = true`): `src` is `[R-plane | G-plane | B-plane]`,
///   each plane of length `total_pixels`.
///   `dst` must have length `total_pixels`.
///
/// # Errors
///
/// Returns [`SIMDTransformError::LengthMismatch`] when slice lengths do not
/// match the stated `total_pixels`.
pub fn simd_rgb_to_grayscale(
    src: &[f32],
    dst: &mut [f32],
    total_pixels: usize,
    is_chw: bool,
) -> Result<(), SIMDTransformError> {
    if dst.len() != total_pixels {
        return Err(SIMDTransformError::LengthMismatch {
            expected: total_pixels,
            got: dst.len(),
            operation: "simd_rgb_to_grayscale/dst",
        });
    }

    const R_COEFF: f32 = 0.299;
    const G_COEFF: f32 = 0.587;
    const B_COEFF: f32 = 0.114;

    if is_chw {
        // Planar: src = [R0..RN, G0..GN, B0..BN]
        let expected_src = total_pixels * 3;
        if src.len() != expected_src {
            return Err(SIMDTransformError::LengthMismatch {
                expected: expected_src,
                got: src.len(),
                operation: "simd_rgb_to_grayscale/src_chw",
            });
        }
        let r_plane = &src[0..total_pixels];
        let g_plane = &src[total_pixels..2 * total_pixels];
        let b_plane = &src[2 * total_pixels..3 * total_pixels];

        const LANE: usize = 8;
        let full = total_pixels / LANE * LANE;
        let mut i = 0usize;
        while i < full {
            dst[i] = r_plane[i] * R_COEFF + g_plane[i] * G_COEFF + b_plane[i] * B_COEFF;
            dst[i + 1] =
                r_plane[i + 1] * R_COEFF + g_plane[i + 1] * G_COEFF + b_plane[i + 1] * B_COEFF;
            dst[i + 2] =
                r_plane[i + 2] * R_COEFF + g_plane[i + 2] * G_COEFF + b_plane[i + 2] * B_COEFF;
            dst[i + 3] =
                r_plane[i + 3] * R_COEFF + g_plane[i + 3] * G_COEFF + b_plane[i + 3] * B_COEFF;
            dst[i + 4] =
                r_plane[i + 4] * R_COEFF + g_plane[i + 4] * G_COEFF + b_plane[i + 4] * B_COEFF;
            dst[i + 5] =
                r_plane[i + 5] * R_COEFF + g_plane[i + 5] * G_COEFF + b_plane[i + 5] * B_COEFF;
            dst[i + 6] =
                r_plane[i + 6] * R_COEFF + g_plane[i + 6] * G_COEFF + b_plane[i + 6] * B_COEFF;
            dst[i + 7] =
                r_plane[i + 7] * R_COEFF + g_plane[i + 7] * G_COEFF + b_plane[i + 7] * B_COEFF;
            i += LANE;
        }
        for j in full..total_pixels {
            dst[j] = r_plane[j] * R_COEFF + g_plane[j] * G_COEFF + b_plane[j] * B_COEFF;
        }
    } else {
        // Interleaved: src = [R0,G0,B0, R1,G1,B1, ...]
        let expected_src = total_pixels * 3;
        if src.len() != expected_src {
            return Err(SIMDTransformError::LengthMismatch {
                expected: expected_src,
                got: src.len(),
                operation: "simd_rgb_to_grayscale/src_hwc",
            });
        }

        const LANE: usize = 8; // 8 pixels at a time = 24 src elements
        let full = total_pixels / LANE * LANE;
        let mut p = 0usize; // pixel index
        while p < full {
            let s = p * 3;
            dst[p] = src[s] * R_COEFF + src[s + 1] * G_COEFF + src[s + 2] * B_COEFF;
            dst[p + 1] =
                src[s + 3] * R_COEFF + src[s + 4] * G_COEFF + src[s + 5] * B_COEFF;
            dst[p + 2] =
                src[s + 6] * R_COEFF + src[s + 7] * G_COEFF + src[s + 8] * B_COEFF;
            dst[p + 3] =
                src[s + 9] * R_COEFF + src[s + 10] * G_COEFF + src[s + 11] * B_COEFF;
            dst[p + 4] =
                src[s + 12] * R_COEFF + src[s + 13] * G_COEFF + src[s + 14] * B_COEFF;
            dst[p + 5] =
                src[s + 15] * R_COEFF + src[s + 16] * G_COEFF + src[s + 17] * B_COEFF;
            dst[p + 6] =
                src[s + 18] * R_COEFF + src[s + 19] * G_COEFF + src[s + 20] * B_COEFF;
            dst[p + 7] =
                src[s + 21] * R_COEFF + src[s + 22] * G_COEFF + src[s + 23] * B_COEFF;
            p += LANE;
        }
        for (offset_q, dst_elem) in dst.iter_mut().enumerate().skip(full).take(total_pixels - full) {
            let s = offset_q * 3;
            *dst_elem = src[s] * R_COEFF + src[s + 1] * G_COEFF + src[s + 2] * B_COEFF;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// simd_mean_variance
// ---------------------------------------------------------------------------

/// Compute the mean and **population** variance of a flat `f32` slice.
///
/// Uses a numerically stable single-pass Welford algorithm with 8 parallel
/// accumulators to minimize rounding error while allowing auto-vectorization.
///
/// # Returns
///
/// `(mean, variance)` where `variance = E[(x − mean)²]`.
///
/// Returns `(0.0, 0.0)` for empty input (no error variant, to stay infallible).
pub fn simd_mean_variance(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (data[0], 0.0);
    }

    // Use 8 parallel Welford accumulators, then combine.
    // Each accumulator maintains (count, mean, M2) per the standard formulation.
    const LANE: usize = 8;

    let mut counts = [0u64; LANE];
    let mut means = [0.0f32; LANE];
    let mut m2s = [0.0f32; LANE];

    let full = n / LANE * LANE;
    let mut i = 0usize;

    // Vectorizable inner loop: 8 independent Welford streams.
    while i < full {
        for lane in 0..LANE {
            let x = data[i + lane];
            counts[lane] += 1;
            let delta = x - means[lane];
            means[lane] += delta / counts[lane] as f32;
            let delta2 = x - means[lane];
            m2s[lane] += delta * delta2;
        }
        i += LANE;
    }

    // Process tail with lane 0.
    for &x in data.iter().skip(full).take(n - full) {
        counts[0] += 1;
        let delta = x - means[0];
        means[0] += delta / counts[0] as f32;
        let delta2 = x - means[0];
        m2s[0] += delta * delta2;
    }

    // Combine the 8 streams using Chan's parallel Welford combination formula.
    // Combine lane 1..7 into lane 0.
    for lane in 1..LANE {
        if counts[lane] == 0 {
            continue;
        }
        let na = counts[0] as f32;
        let nb = counts[lane] as f32;
        let combined = na + nb;
        let delta = means[lane] - means[0];
        means[0] = (na * means[0] + nb * means[lane]) / combined;
        m2s[0] += m2s[lane] + delta * delta * (na * nb / combined);
        counts[0] += counts[lane];
    }

    let variance = if counts[0] > 1 {
        m2s[0] / counts[0] as f32
    } else {
        0.0
    };

    (means[0], variance)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a 2-channel CHW tensor of shape [2, H, W]
    fn make_chw_data(h: usize, w: usize, c: usize) -> Vec<f32> {
        (0..c * h * w).map(|i| i as f32).collect()
    }

    // -----------------------------------------------------------------------
    // simd_standardize
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_standardize_zero_mean_unit_var() {
        let mut data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let (mean, std) = simd_standardize(&mut data).expect("standardize should succeed");
        // After standardization, mean should be ~0 and variance ~1
        let (post_mean, post_var) = simd_mean_variance(&data);
        assert!(
            post_mean.abs() < 1e-4,
            "post-standardize mean={post_mean}, expected ≈ 0"
        );
        assert!(
            (post_var - 1.0).abs() < 1e-3,
            "post-standardize var={post_var}, expected ≈ 1"
        );
        assert!(mean > 0.0, "original mean should be positive");
        assert!(std > 0.0, "std should be positive");
    }

    #[test]
    fn test_simd_standardize_single_element() {
        let mut data = vec![42.0f32];
        let result = simd_standardize(&mut data);
        assert!(
            result.is_err(),
            "single-element input should fail (ZeroStd)"
        );
    }

    #[test]
    fn test_simd_standardize_all_same() {
        let mut data = vec![5.0f32; 50];
        let result = simd_standardize(&mut data);
        // All-same → std = 0 → ZeroStd error expected
        assert!(
            result.is_err(),
            "all-same input should return ZeroStd error"
        );
        match result {
            Err(SIMDTransformError::ZeroStd) => {}
            other => panic!("expected ZeroStd, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // simd_scale_offset
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_scale_offset() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        simd_scale_offset_inplace(&mut data, 2.0, 1.0);
        assert_eq!(data, vec![3.0, 5.0, 7.0]);
    }

    // -----------------------------------------------------------------------
    // simd_clamp
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_clamp() {
        let mut data = vec![-5.0f32, 0.0, 5.0];
        simd_clamp_inplace(&mut data, 0.0, 3.0);
        assert_eq!(data, vec![0.0, 0.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // simd_hwc_to_chw / simd_chw_to_hwc
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_hwc_to_chw_identity_round_trip() {
        let h = 4;
        let w = 4;
        let c = 3;
        let src: Vec<f32> = (0..(h * w * c)).map(|i| i as f32).collect();
        let mut intermediate = vec![0.0f32; h * w * c];
        let mut restored = vec![0.0f32; h * w * c];

        simd_hwc_to_chw(&src, &mut intermediate, h, w, c).expect("hwc_to_chw failed");
        simd_chw_to_hwc(&intermediate, &mut restored, c, h, w).expect("chw_to_hwc failed");

        for (idx, (&a, &b)) in src.iter().zip(restored.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "round-trip mismatch at index {idx}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_simd_hwc_to_chw_correctness() {
        // 1x1 image, 3 channels: src = [R, G, B]
        // CHW output: [R_plane | G_plane | B_plane]
        let src = vec![1.0f32, 2.0, 3.0]; // H=1, W=1, C=3
        let mut dst = vec![0.0f32; 3];
        simd_hwc_to_chw(&src, &mut dst, 1, 1, 3).expect("hwc_to_chw failed");
        // dst[0] = R (channel 0), dst[1] = G (channel 1), dst[2] = B (channel 2)
        assert_eq!(dst[0], 1.0, "R channel");
        assert_eq!(dst[1], 2.0, "G channel");
        assert_eq!(dst[2], 3.0, "B channel");
    }

    #[test]
    fn test_simd_chw_to_hwc_single_channel() {
        // C=1: CHW and HWC are equivalent layouts
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        simd_chw_to_hwc(&src, &mut dst, 1, 4, 4).expect("chw_to_hwc failed");
        assert_eq!(src, dst, "single-channel should be identity");
    }

    // -----------------------------------------------------------------------
    // simd_u8_to_f32_normalized
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_u8_to_f32() {
        let src = vec![0u8, 128u8, 255u8];
        let mut dst = vec![0.0f32; 3];
        simd_u8_to_f32_normalized(&src, &mut dst).expect("u8_to_f32 failed");
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[2], 1.0);
        // 128 / 255 ≈ 0.5020
        assert!((dst[1] - 128.0 / 255.0).abs() < 1e-6, "dst[1]={}", dst[1]);
    }

    // -----------------------------------------------------------------------
    // simd_rgb_to_grayscale
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_rgb_to_grayscale_white() {
        // RGB(1,1,1) → Y = 0.299 + 0.587 + 0.114 = 1.0
        let src = vec![1.0f32, 1.0, 1.0]; // 1 pixel, HWC
        let mut dst = vec![0.0f32; 1];
        simd_rgb_to_grayscale(&src, &mut dst, 1, false).expect("grayscale failed");
        assert!(
            (dst[0] - 1.0).abs() < 1e-5,
            "white pixel gray={}, expected 1.0",
            dst[0]
        );
    }

    #[test]
    fn test_simd_rgb_to_grayscale_red() {
        // RGB(1,0,0) → Y ≈ 0.299
        let src = vec![1.0f32, 0.0, 0.0];
        let mut dst = vec![0.0f32; 1];
        simd_rgb_to_grayscale(&src, &mut dst, 1, false).expect("grayscale failed");
        assert!(
            (dst[0] - 0.299).abs() < 1e-5,
            "red pixel gray={}, expected 0.299",
            dst[0]
        );
    }

    // -----------------------------------------------------------------------
    // simd_mean_variance
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_mean_variance() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (mean, var) = simd_mean_variance(&data);
        // mean = 3.0, population variance = 2.0
        assert!((mean - 3.0).abs() < 1e-5, "mean={mean}");
        assert!((var - 2.0).abs() < 1e-4, "var={var}");
    }

    #[test]
    fn test_simd_mean_variance_large() {
        let data = vec![5.0f32; 10_000];
        let (mean, var) = simd_mean_variance(&data);
        assert!((mean - 5.0).abs() < 1e-4, "mean={mean}");
        assert!(var < 1e-6, "var={var} should be ~0");
    }

    // -----------------------------------------------------------------------
    // simd_normalize_chw
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_normalize_chw_correctness() {
        // 2-channel CHW image: C=2, H=4, W=4 → 32 elements
        // Channel 0: values 0..15 (mean=7.5), Channel 1: values 16..31 (mean=23.5)
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let means = vec![7.5f32, 23.5];
        // Compute per-channel std
        let c0: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (_, var0) = simd_mean_variance(&c0);
        let c1: Vec<f32> = (16..32).map(|i| i as f32).collect();
        let (_, var1) = simd_mean_variance(&c1);
        let stds = vec![var0.sqrt(), var1.sqrt()];

        simd_normalize_chw(&mut data, &means, &stds, 2, true).expect("normalize failed");

        // After normalization, per-channel mean should be ~0
        let (post_mean0, _) = simd_mean_variance(&data[0..16]);
        let (post_mean1, _) = simd_mean_variance(&data[16..32]);
        assert!(
            post_mean0.abs() < 1e-4,
            "channel0 mean after normalize = {post_mean0}"
        );
        assert!(
            post_mean1.abs() < 1e-4,
            "channel1 mean after normalize = {post_mean1}"
        );
    }

    #[test]
    fn test_simd_normalize_chw_zero_std_error() {
        let mut data = vec![1.0f32; 8];
        let means = vec![1.0f32];
        let stds = vec![0.0f32]; // zero std → error
        let result = simd_normalize_chw(&mut data, &means, &stds, 1, true);
        assert!(result.is_err(), "zero std should produce an error");
        match result {
            Err(SIMDTransformError::ZeroStd) => {}
            other => panic!("expected ZeroStd, got {other:?}"),
        }
    }
}
