//! Common utility functions for TenfloweRS applications.
//!
//! This module collects small, frequently-used helpers that don't belong in
//! a single subcrate.  All functions are `#[inline]` and pure Rust — no
//! external crates required.
//!
//! # Contents
//!
//! - [`crate::utils::clamp`] / [`crate::utils::clamp_slice_inplace`] — scalar and slice clamping
//! - [`crate::utils::sigmoid`] / [`crate::utils::sigmoid_slice`] — logistic function
//! - [`crate::utils::softmax`] / [`crate::utils::log_softmax`] — probability distributions over logits
//! - [`crate::utils::lerp`] — linear interpolation
//! - [`crate::utils::next_power_of_two`] — useful for buffer sizes
//! - [`crate::utils::bytes_to_human_readable`] — human-friendly byte counts
//! - [`crate::utils::num_elements`] — element count for a shape slice
//! - [`crate::utils::shape_to_strides`] — row-major strides from a shape

/// Clamp a value to the inclusive range `[lo, hi]`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::clamp;
/// assert_eq!(clamp(5.0f32, 0.0, 3.0), 3.0);
/// assert_eq!(clamp(-1.0f32, 0.0, 3.0), 0.0);
/// assert_eq!(clamp(2.0f32, 0.0, 3.0), 2.0);
/// ```
#[inline]
pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

/// Clamp every element of `slice` to `[lo, hi]` in place.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::clamp_slice_inplace;
/// let mut data = vec![-1.0f32, 0.5, 2.5];
/// clamp_slice_inplace(&mut data, 0.0, 1.0);
/// assert_eq!(data, vec![0.0, 0.5, 1.0]);
/// ```
#[inline]
pub fn clamp_slice_inplace(slice: &mut [f32], lo: f32, hi: f32) {
    for x in slice.iter_mut() {
        *x = x.max(lo).min(hi);
    }
}

/// Compute the sigmoid (logistic) function `1 / (1 + exp(-x))`.
///
/// Numerically stable for large positive and large negative inputs.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::sigmoid;
/// assert!((sigmoid(0.0f32) - 0.5).abs() < 1e-6);
/// assert!(sigmoid(1000.0f32) > 0.999);
/// assert!(sigmoid(-1000.0f32) < 0.001);
/// ```
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Apply sigmoid element-wise to a slice, returning a new `Vec<f32>`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::sigmoid_slice;
/// let out = sigmoid_slice(&[0.0f32]);
/// assert!((out[0] - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid_slice(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| sigmoid(x)).collect()
}

/// Compute softmax over a logit slice, returning probabilities that sum to 1.
///
/// Uses the numerically stable `exp(x - max(x))` formulation.
///
/// # Panics
///
/// Panics if `logits` is empty.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::softmax;
/// let probs = softmax(&[1.0f32, 2.0, 3.0]);
/// let sum: f32 = probs.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// assert!(probs[2] > probs[1] && probs[1] > probs[0]);
/// ```
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty(), "softmax requires at least one element");
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for x in out.iter_mut() {
        *x /= sum;
    }
    out
}

/// Compute log-softmax over a logit slice.
///
/// Equivalent to `log(softmax(x))` but numerically more stable.
///
/// # Panics
///
/// Panics if `logits` is empty.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::log_softmax;
/// let ls = log_softmax(&[1.0f32, 2.0, 3.0]);
/// // Every log-prob should be negative (probabilities < 1).
/// assert!(ls.iter().all(|&v| v <= 0.0));
/// // log-sum-exp of the output should be 0 (i.e. probs sum to 1).
/// let lse: f32 = ls.iter().map(|&v| v.exp()).sum::<f32>().ln();
/// assert!((lse).abs() < 1e-5);
/// ```
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    assert!(
        !logits.is_empty(),
        "log_softmax requires at least one element"
    );
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = max + sum_exp.ln();
    logits.iter().map(|&x| x - log_sum).collect()
}

/// Linearly interpolate between `a` and `b` at parameter `t ∈ [0, 1]`.
///
/// `t = 0` returns `a`, `t = 1` returns `b`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::lerp;
/// assert_eq!(lerp(0.0f32, 10.0, 0.5), 5.0);
/// assert_eq!(lerp(0.0f32, 10.0, 0.0), 0.0);
/// assert_eq!(lerp(0.0f32, 10.0, 1.0), 10.0);
/// ```
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Return the smallest power of two that is `>= n`.
///
/// Returns 1 for `n == 0`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::next_power_of_two;
/// assert_eq!(next_power_of_two(0), 1);
/// assert_eq!(next_power_of_two(1), 1);
/// assert_eq!(next_power_of_two(5), 8);
/// assert_eq!(next_power_of_two(8), 8);
/// assert_eq!(next_power_of_two(9), 16);
/// ```
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Format a byte count as a human-readable string (e.g. `"1.25 GiB"`).
///
/// Uses binary prefixes (KiB = 1024, MiB = 1024², …).
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::bytes_to_human_readable;
/// assert_eq!(bytes_to_human_readable(0),               "0 B");
/// assert_eq!(bytes_to_human_readable(1023),            "1023 B");
/// assert_eq!(bytes_to_human_readable(1024),            "1.00 KiB");
/// assert_eq!(bytes_to_human_readable(1024 * 1024),     "1.00 MiB");
/// ```
pub fn bytes_to_human_readable(bytes: usize) -> String {
    const KIB: usize = 1024;
    const MIB: usize = 1024 * KIB;
    const GIB: usize = 1024 * MIB;
    const TIB: usize = 1024 * GIB;

    if bytes == 0 {
        "0 B".to_owned()
    } else if bytes < KIB {
        format!("{bytes} B")
    } else if bytes < MIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else if bytes < GIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes < TIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    }
}

/// Return the total number of elements in a tensor described by `shape`.
///
/// Returns 0 for an empty shape slice (scalar would return 1 from
/// `product` of an empty iterator; this function treats the empty slice
/// as a degenerate case and returns 1).
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::num_elements;
/// assert_eq!(num_elements(&[2, 3, 4]), 24);
/// assert_eq!(num_elements(&[]),        1);
/// assert_eq!(num_elements(&[0, 5]),    0);
/// ```
#[inline]
pub fn num_elements(shape: &[usize]) -> usize {
    shape
        .iter()
        .product::<usize>()
        .max(if shape.is_empty() { 1 } else { 0 })
}

/// Compute row-major (C-contiguous) strides for a given shape.
///
/// The stride of axis `i` is the product of all dimensions after `i`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::shape_to_strides;
/// assert_eq!(shape_to_strides(&[2, 3, 4]), vec![12, 4, 1]);
/// assert_eq!(shape_to_strides(&[5]),        vec![1]);
/// assert_eq!(shape_to_strides(&[]),         Vec::<usize>::new());
/// ```
pub fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the flat index for a multi-dimensional index given row-major strides.
///
/// # Panics
///
/// Panics (debug) if `indices.len() != strides.len()`.
///
/// # Example
///
/// ```rust
/// use tenflowers::utils::{shape_to_strides, flat_index};
/// let shape = [2usize, 3, 4];
/// let strides = shape_to_strides(&shape);
/// assert_eq!(flat_index(&[1, 2, 3], &strides), 1*12 + 2*4 + 3*1);
/// ```
#[inline]
pub fn flat_index(indices: &[usize], strides: &[usize]) -> usize {
    debug_assert_eq!(
        indices.len(),
        strides.len(),
        "indices and strides must have same length"
    );
    indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_in_range() {
        assert_eq!(clamp(2.0, 1.0, 3.0), 2.0);
    }
    #[test]
    fn test_clamp_below() {
        assert_eq!(clamp(0.0, 1.0, 3.0), 1.0);
    }
    #[test]
    fn test_clamp_above() {
        assert_eq!(clamp(5.0, 1.0, 3.0), 3.0);
    }

    #[test]
    fn test_clamp_slice() {
        let mut v = vec![-2.0f32, 0.5, 2.5];
        clamp_slice_inplace(&mut v, 0.0, 1.0);
        assert_eq!(v, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_sigmoid_positive_saturation() {
        assert!(sigmoid(20.0) > 0.999);
    }
    #[test]
    fn test_sigmoid_negative_saturation() {
        assert!(sigmoid(-20.0) < 0.001);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_monotone() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn test_log_softmax_non_positive() {
        let ls = log_softmax(&[1.0, 2.0, 3.0]);
        assert!(ls.iter().all(|&v| v <= 0.0));
    }

    #[test]
    fn test_lerp_midpoint() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
    }
    #[test]
    fn test_lerp_endpoints() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }

    #[test]
    fn test_bytes_human_readable() {
        assert_eq!(bytes_to_human_readable(0), "0 B");
        assert_eq!(bytes_to_human_readable(512), "512 B");
        assert_eq!(bytes_to_human_readable(1024), "1.00 KiB");
        assert_eq!(bytes_to_human_readable(1024 * 1024), "1.00 MiB");
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(num_elements(&[2, 3, 4]), 24);
        assert_eq!(num_elements(&[]), 1);
        assert_eq!(num_elements(&[0, 5]), 0);
    }

    #[test]
    fn test_shape_to_strides() {
        assert_eq!(shape_to_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(shape_to_strides(&[5]), vec![1]);
        assert_eq!(shape_to_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_flat_index() {
        let shape = [2usize, 3, 4];
        let strides = shape_to_strides(&shape);
        assert_eq!(flat_index(&[1, 2, 3], &strides), 12 + 2 * 4 + 3);
        assert_eq!(flat_index(&[0, 0, 0], &strides), 0);
    }
}
