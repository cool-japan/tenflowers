//! AutoCast context manager and dynamic loss scaling for mixed precision training.
//!
//! This module provides:
//! - [`AutoCast`]: A context manager that controls the precision used for forward pass
//!   computations, analogous to `torch.autocast`.
//! - [`GradScaler`]: A dynamic loss scaler that prevents gradient underflow in FP16 training,
//!   analogous to `torch.cuda.amp.GradScaler`.
//! - Precision simulation utilities for testing and analysis.
//! - Gradient safety utilities (overflow detection, finite checks, norm clipping).
//!
//! ## Example
//!
//! ```rust,no_run
//! use tenflowers_core::autocast::{AutoCast, AutoCastDtype, GradScaler};
//!
//! // Set up FP16 autocast and dynamic loss scaling
//! let ctx = AutoCast::enabled();
//! let mut scaler = GradScaler::new(65536.0);
//!
//! // During training loop:
//! let loss: f32 = 1.23;
//! let scaled_loss = scaler.scale_loss(loss);
//!
//! // After backward pass, unscale and check gradients
//! let mut grads = vec![0.001_f32; 100];
//! let step_ok = scaler.step_update(&grads);
//! if step_ok {
//!     scaler.unscale_gradients(&mut grads);
//!     // optimizer.step()
//! }
//! ```

use crate::{Result, TensorError};

// ---------------------------------------------------------------------------
// AutoCast dtype
// ---------------------------------------------------------------------------

/// Precision type selected by an [`AutoCast`] context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoCastDtype {
    /// 32-bit IEEE 754 single precision (no casting, default).
    Float32,
    /// 16-bit IEEE 754 half precision (FP16).
    Float16,
    /// 16-bit brain float (BF16, keeps FP32 exponent range with reduced mantissa).
    BFloat16,
}

// ---------------------------------------------------------------------------
// AutoCast context
// ---------------------------------------------------------------------------

/// AutoCast context: controls the effective precision for forward computations.
///
/// When enabled, [`cast_input`](AutoCast::cast_input) and
/// [`cast_scalar`](AutoCast::cast_scalar) quantize f32 values to the configured
/// half-precision format and back.  This simulates the precision loss that
/// occurs when real hardware executes operations natively in FP16/BF16.
#[derive(Debug, Clone)]
pub struct AutoCast {
    /// The target reduced precision dtype.
    pub dtype: AutoCastDtype,
    /// When `false`, all cast operations are no-ops (identity).
    pub enabled: bool,
}

impl AutoCast {
    /// Create an `AutoCast` context with explicit dtype and enabled flag.
    pub fn new(dtype: AutoCastDtype) -> Self {
        Self {
            dtype,
            enabled: true,
        }
    }

    /// Create an enabled FP16 autocast context (the most common case).
    pub fn enabled() -> Self {
        Self {
            dtype: AutoCastDtype::Float16,
            enabled: true,
        }
    }

    /// Create a disabled autocast context (identity — stays in FP32).
    pub fn disabled() -> Self {
        Self {
            dtype: AutoCastDtype::Float32,
            enabled: false,
        }
    }

    /// Cast a slice of f32 values to the autocast dtype and back.
    ///
    /// For `Float32` or when disabled, returns a clone of the input.
    /// For `Float16`, each value is rounded to the nearest representable f16 value.
    /// For `BFloat16`, each value is rounded to the nearest representable bf16 value.
    pub fn cast_input(&self, data: &[f32]) -> Vec<f32> {
        if !self.enabled {
            return data.to_vec();
        }
        match self.dtype {
            AutoCastDtype::Float32 => data.to_vec(),
            AutoCastDtype::Float16 => data.iter().map(|&x| simulate_f16(x)).collect(),
            AutoCastDtype::BFloat16 => data.iter().map(|&x| simulate_bf16(x)).collect(),
        }
    }

    /// Cast a single f32 scalar to the autocast dtype and back.
    pub fn cast_scalar(&self, x: f32) -> f32 {
        if !self.enabled {
            return x;
        }
        match self.dtype {
            AutoCastDtype::Float32 => x,
            AutoCastDtype::Float16 => simulate_f16(x),
            AutoCastDtype::BFloat16 => simulate_bf16(x),
        }
    }

    /// Return `true` if `|x| > 65504`, the maximum finite FP16 value.
    ///
    /// NaN also counts as "would overflow" because it cannot be represented
    /// as a finite FP16 either.
    #[inline]
    pub fn would_overflow_fp16(x: f32) -> bool {
        // FP16 max finite is 65504.0
        !x.is_finite() || x.abs() > 65504.0_f32
    }
}

// ---------------------------------------------------------------------------
// Low-level precision simulation
// ---------------------------------------------------------------------------

/// Simulate IEEE 754 FP16 rounding: round `x` to the nearest value
/// representable in half precision, then return it as an f32.
///
/// Sub-normal FP16 values (absolute value < 2^{-14} ≈ 6.10e-5) are flushed
/// to zero when their mantissa would be all-zero after quantisation, matching
/// the default flush-to-zero behaviour of many accelerators.
/// Values whose absolute value exceeds 65504 are clamped to ±Inf.
pub fn simulate_f16(x: f32) -> f32 {
    use half::f16;
    // The `half` crate implements the full IEEE 754-2008 conversion including
    // round-to-nearest-even, denormal flushing options, and NaN/Inf propagation.
    f16::from_f32(x).to_f32()
}

/// Simulate BF16 rounding: truncate the mantissa to 7 bits (keeping the
/// top 16 bits of an f32 bit pattern), then return as f32.
///
/// BF16 has the same exponent width as FP32 (8 bits) so it never overflows
/// for finite f32 values, but it has significantly reduced mantissa precision.
pub fn simulate_bf16(x: f32) -> f32 {
    use half::bf16;
    bf16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------
// ScalerState (serialisable snapshot)
// ---------------------------------------------------------------------------

/// Snapshot of [`GradScaler`] state for checkpointing and restoration.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalerState {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u32,
    pub steps_since_overflow: u32,
}

// ---------------------------------------------------------------------------
// GradScaler
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for mixed precision training (analogous to PyTorch `GradScaler`).
///
/// # Algorithm
///
/// 1. Before the backward pass, multiply the loss by `scale` so that small
///    gradients are amplified into the FP16 representable range.
/// 2. After the backward pass, divide the gradients by `scale` before passing
///    them to the optimiser (`unscale_gradients`).
/// 3. If any gradient is NaN or ±Inf (`check_overflow` returns `true`), the
///    optimiser step is *skipped* and `scale` is halved (`backoff_factor`).
/// 4. After `growth_interval` consecutive steps without overflow, `scale` is
///    doubled (`growth_factor`).
#[derive(Debug, Clone)]
pub struct GradScaler {
    /// Current loss scale factor.
    pub scale: f32,
    /// Multiply `scale` by this value when growing (default 2.0).
    pub growth_factor: f32,
    /// Multiply `scale` by this value on overflow (default 0.5).
    pub backoff_factor: f32,
    /// Number of consecutive non-overflow steps required before growing.
    pub growth_interval: u32,
    /// When `false`, all scale/unscale operations are no-ops.
    pub enabled: bool,
    /// Internal counter: steps elapsed since the last overflow.
    steps_since_overflow: u32,
    /// Total number of overflow events recorded.
    pub overflow_count: u64,
    /// Total number of `step_update` calls.
    pub step_count: u64,
}

impl GradScaler {
    /// Create a new scaler with default hyperparameters.
    ///
    /// - `growth_factor` = 2.0
    /// - `backoff_factor` = 0.5
    /// - `growth_interval` = 2000
    pub fn new(init_scale: f32) -> Self {
        Self {
            scale: init_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            enabled: true,
            steps_since_overflow: 0,
            overflow_count: 0,
            step_count: 0,
        }
    }

    /// Create a scaler with explicit hyperparameters, returning an error if any
    /// argument is out of the valid range.
    ///
    /// Constraints:
    /// - `init_scale` must be finite and positive
    /// - `growth_factor` must be > 1.0
    /// - `backoff_factor` must be in (0.0, 1.0)
    /// - `interval` must be ≥ 1
    pub fn with_config(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        interval: u32,
    ) -> Result<Self> {
        if !init_scale.is_finite() || init_scale <= 0.0 {
            return Err(TensorError::InvalidArgument {
                operation: "GradScaler::with_config".to_string(),
                reason: format!(
                    "init_scale must be a positive finite value, got {}",
                    init_scale
                ),
                context: None,
            });
        }
        if growth_factor <= 1.0 {
            return Err(TensorError::InvalidArgument {
                operation: "GradScaler::with_config".to_string(),
                reason: format!(
                    "growth_factor must be > 1.0, got {}",
                    growth_factor
                ),
                context: None,
            });
        }
        if backoff_factor <= 0.0 || backoff_factor >= 1.0 {
            return Err(TensorError::InvalidArgument {
                operation: "GradScaler::with_config".to_string(),
                reason: format!(
                    "backoff_factor must be in (0.0, 1.0), got {}",
                    backoff_factor
                ),
                context: None,
            });
        }
        if interval == 0 {
            return Err(TensorError::InvalidArgument {
                operation: "GradScaler::with_config".to_string(),
                reason: "growth_interval must be >= 1".to_string(),
                context: None,
            });
        }
        Ok(Self {
            scale: init_scale,
            growth_factor,
            backoff_factor,
            growth_interval: interval,
            enabled: true,
            steps_since_overflow: 0,
            overflow_count: 0,
            step_count: 0,
        })
    }

    /// Create a disabled scaler (all operations are identity / no-ops).
    pub fn disabled() -> Self {
        Self {
            scale: 1.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            enabled: false,
            steps_since_overflow: 0,
            overflow_count: 0,
            step_count: 0,
        }
    }

    /// Multiply the loss by the current scale factor before the backward pass.
    ///
    /// When disabled, returns `loss` unchanged.
    #[inline]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if self.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    /// Scale gradients **in-place** (multiply each element by `scale`).
    ///
    /// Call this immediately after the backward pass before `unscale_gradients`.
    /// Note: in typical usage you would call `scale_loss` once before the
    /// backward pass (which automatically scales all gradients proportionally),
    /// so you usually do *not* need to call this separately.
    pub fn scale_gradients(&self, grads: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let s = self.scale;
        for g in grads.iter_mut() {
            *g *= s;
        }
    }

    /// Unscale gradients **in-place** (divide each element by `scale`).
    ///
    /// Must be called before the optimiser step.  If `scale` is zero or
    /// non-finite the gradients are set to NaN so that a subsequent
    /// `check_overflow` call will detect the problem.
    pub fn unscale_gradients(&self, grads: &mut [f32]) {
        if !self.enabled {
            return;
        }
        if self.scale == 0.0 || !self.scale.is_finite() {
            // Mark all gradients as invalid
            for g in grads.iter_mut() {
                *g = f32::NAN;
            }
            return;
        }
        let inv = 1.0 / self.scale;
        for g in grads.iter_mut() {
            *g *= inv;
        }
    }

    /// Return `true` if any gradient is NaN or ±Inf.
    #[inline]
    pub fn check_overflow(&self, grads: &[f32]) -> bool {
        grads.iter().any(|&g| !g.is_finite())
    }

    /// Inspect gradients, update the scale, and return whether the optimiser
    /// step should proceed.
    ///
    /// - Checks for NaN/Inf in `grads`.
    /// - On overflow: multiplies `scale` by `backoff_factor`, increments
    ///   `overflow_count`, resets `steps_since_overflow`, and returns `false`.
    /// - On no overflow: increments `steps_since_overflow`; when it reaches
    ///   `growth_interval` the scale is multiplied by `growth_factor` and the
    ///   counter is reset. Returns `true`.
    ///
    /// When the scaler is disabled, always returns `true` without modifying
    /// state.
    pub fn step_update(&mut self, grads: &[f32]) -> bool {
        self.step_count += 1;
        if !self.enabled {
            return true;
        }

        if self.check_overflow(grads) {
            self.overflow_count += 1;
            self.steps_since_overflow = 0;
            self.scale = (self.scale * self.backoff_factor).max(1.0_f32);
            return false;
        }

        self.steps_since_overflow += 1;
        if self.steps_since_overflow >= self.growth_interval {
            // Clamp to f32::MAX to avoid Inf
            self.scale = (self.scale * self.growth_factor).min(f32::MAX / 2.0);
            self.steps_since_overflow = 0;
        }

        true
    }

    /// Return the current scale value.
    #[inline]
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Capture a serialisable snapshot of the scaler state.
    pub fn state_dict(&self) -> ScalerState {
        ScalerState {
            scale: self.scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            steps_since_overflow: self.steps_since_overflow,
        }
    }

    /// Restore scaler state from a previously captured [`ScalerState`].
    pub fn load_state_dict(&mut self, state: ScalerState) {
        self.scale = state.scale;
        self.growth_factor = state.growth_factor;
        self.backoff_factor = state.backoff_factor;
        self.growth_interval = state.growth_interval;
        self.steps_since_overflow = state.steps_since_overflow;
    }
}

// ---------------------------------------------------------------------------
// Precision utilities
// ---------------------------------------------------------------------------

/// Convert every element in `data` to FP16 and back to f32 (roundtrip).
///
/// This quantises the values to the set of representable FP16 values so that
/// tests can measure precision loss without allocating actual `half::f16` storage.
pub fn f32_to_f16_roundtrip(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| simulate_f16(x)).collect()
}

/// Compute the mean relative precision error introduced by an FP16 roundtrip.
///
/// For each element `x`, the relative error is `|x - simulate_f16(x)| / (|x| + eps)`.
/// Returns 0.0 for an empty slice.
pub fn f16_precision_error(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let eps = f32::EPSILON;
    let sum: f32 = data
        .iter()
        .map(|&x| {
            let q = simulate_f16(x);
            (x - q).abs() / (x.abs() + eps)
        })
        .sum();
    sum / data.len() as f32
}

/// Return `true` if all gradients are finite (not NaN, not ±Inf).
#[inline]
pub fn grads_are_finite(grads: &[f32]) -> bool {
    grads.iter().all(|&g| g.is_finite())
}

/// Clip gradients **in-place** so that their global L2 norm does not exceed
/// `max_norm`.
///
/// Returns the *pre-clipping* global L2 norm.  If `max_norm` is zero or the
/// gradient slice is empty the gradients are left unchanged and 0.0 is returned.
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    if grads.is_empty() || max_norm <= 0.0 {
        return 0.0;
    }

    let norm_sq: f32 = grads.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }

    norm
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- simulate_f16 ---

    #[test]
    fn test_simulate_f16_exact_representable_values() {
        // 1.0 is exactly representable in FP16
        assert_eq!(simulate_f16(1.0_f32), 1.0_f32);
        // 0.0 stays 0.0
        assert_eq!(simulate_f16(0.0_f32), 0.0_f32);
        // -1.0 stays -1.0
        assert_eq!(simulate_f16(-1.0_f32), -1.0_f32);
        // 2.0 is exactly representable
        assert_eq!(simulate_f16(2.0_f32), 2.0_f32);
    }

    #[test]
    fn test_simulate_f16_small_denormals_flush() {
        // Values much smaller than the FP16 minimum normal (~6.1e-5) should
        // flush to zero (or a very small denormal, but ultimately the
        // roundtrip through half::f16 gives 0.0 for very tiny values).
        let tiny = 1.0e-8_f32;
        let result = simulate_f16(tiny);
        // Either exactly 0 (flush-to-zero) or very small denormal — not the
        // full f32 value.
        assert!(
            result.abs() < 1.0e-6_f32,
            "Expected tiny value to flush toward 0 in FP16, got {}",
            result
        );
    }

    #[test]
    fn test_simulate_f16_large_value_clamps_to_inf() {
        // Values larger than 65504 should become ±Inf after FP16 conversion
        let huge = 1.0e10_f32;
        let result = simulate_f16(huge);
        assert!(
            result.is_infinite(),
            "Expected Inf for huge value, got {}",
            result
        );
    }

    #[test]
    fn test_simulate_f16_nan_stays_nan() {
        assert!(simulate_f16(f32::NAN).is_nan());
    }

    #[test]
    fn test_simulate_f16_precision_loss() {
        // π rounded to FP16 should differ from the f32 value
        let pi = std::f32::consts::PI;
        let approx = simulate_f16(pi);
        // The error should be small but non-zero
        assert!((pi - approx).abs() < 0.01_f32);
        assert!((pi - approx).abs() > 0.0_f32);
    }

    // --- simulate_bf16 vs simulate_f16 ---

    #[test]
    fn test_simulate_bf16_larger_range_than_f16() {
        // BF16 has the same exponent as FP32 so it does not overflow for 65504
        let v = 65504.0_f32;
        let f16_result = simulate_f16(v);
        let bf16_result = simulate_bf16(v);
        // FP16 exactly represents 65504 (it is the max finite value)
        assert!((f16_result - v).abs() < 1.0_f32);
        // BF16 should also be close (large exponent range)
        assert!(bf16_result.is_finite());
    }

    #[test]
    fn test_simulate_bf16_less_mantissa_precision_than_f16() {
        // For values in mid-range (e.g. 1.1), BF16 has only 7 mantissa bits
        // while FP16 has 10, so BF16 error should be larger.
        let x = 1.1_f32;
        let f16_err = (x - simulate_f16(x)).abs();
        let bf16_err = (x - simulate_bf16(x)).abs();
        // BF16 mantissa is shorter → larger quantisation error for 1.1
        assert!(
            bf16_err >= f16_err,
            "Expected bf16_err ({}) >= f16_err ({}) for x={}",
            bf16_err,
            f16_err,
            x
        );
    }

    #[test]
    fn test_simulate_bf16_does_not_overflow_large_f32() {
        // BF16 shares FP32's exponent range, so huge f32 values stay finite
        let big = 1.0e30_f32;
        let result = simulate_bf16(big);
        assert!(result.is_finite(), "Expected finite result for {}; got {}", big, result);
    }

    // --- AutoCast::would_overflow_fp16 ---

    #[test]
    fn test_would_overflow_fp16_below_max() {
        assert!(!AutoCast::would_overflow_fp16(65504.0_f32));
        assert!(!AutoCast::would_overflow_fp16(-65504.0_f32));
        assert!(!AutoCast::would_overflow_fp16(1.0_f32));
        assert!(!AutoCast::would_overflow_fp16(0.0_f32));
    }

    #[test]
    fn test_would_overflow_fp16_above_max() {
        assert!(AutoCast::would_overflow_fp16(65505.0_f32));
        assert!(AutoCast::would_overflow_fp16(-65505.0_f32));
        assert!(AutoCast::would_overflow_fp16(1.0e10_f32));
    }

    #[test]
    fn test_would_overflow_fp16_special() {
        assert!(AutoCast::would_overflow_fp16(f32::INFINITY));
        assert!(AutoCast::would_overflow_fp16(f32::NEG_INFINITY));
        assert!(AutoCast::would_overflow_fp16(f32::NAN));
    }

    // --- GradScaler::scale_loss ---

    #[test]
    fn test_grad_scaler_scale_loss() {
        let scaler = GradScaler::new(1024.0);
        assert_eq!(scaler.scale_loss(2.0), 2048.0_f32);
        assert_eq!(scaler.scale_loss(0.0), 0.0_f32);
    }

    #[test]
    fn test_grad_scaler_disabled_scale_loss_is_identity() {
        let scaler = GradScaler::disabled();
        assert_eq!(scaler.scale_loss(3.14), 3.14_f32);
    }

    // --- GradScaler::step_update — growth ---

    #[test]
    fn test_grad_scaler_grows_after_growth_interval() {
        let mut scaler = GradScaler::with_config(1.0, 2.0, 0.5, 3).expect("valid config");
        let clean_grads = vec![0.1_f32, 0.2_f32];

        // 2 clean steps — scale should not grow yet
        scaler.step_update(&clean_grads);
        scaler.step_update(&clean_grads);
        assert_eq!(scaler.get_scale(), 1.0_f32);

        // 3rd clean step crosses growth_interval → scale doubles
        scaler.step_update(&clean_grads);
        assert_eq!(scaler.get_scale(), 2.0_f32);
        // Counter should have been reset
        assert_eq!(scaler.steps_since_overflow, 0);
    }

    // --- GradScaler::step_update — backoff ---

    #[test]
    fn test_grad_scaler_backs_off_on_overflow() {
        let mut scaler = GradScaler::new(1024.0);
        let bad_grads = vec![f32::NAN, 0.5_f32];

        let proceed = scaler.step_update(&bad_grads);
        assert!(!proceed, "Step should be skipped on overflow");
        assert_eq!(scaler.get_scale(), 512.0_f32); // 1024 * 0.5
        assert_eq!(scaler.overflow_count, 1);
        assert_eq!(scaler.steps_since_overflow, 0);
    }

    #[test]
    fn test_grad_scaler_backs_off_on_inf_gradient() {
        let mut scaler = GradScaler::new(4096.0);
        let bad_grads = vec![f32::INFINITY];
        let proceed = scaler.step_update(&bad_grads);
        assert!(!proceed);
        assert_eq!(scaler.get_scale(), 2048.0_f32);
    }

    // --- GradScaler::check_overflow ---

    #[test]
    fn test_check_overflow_detects_nan() {
        let scaler = GradScaler::new(1.0);
        assert!(scaler.check_overflow(&[f32::NAN]));
        assert!(scaler.check_overflow(&[1.0, f32::NAN, 2.0]));
        assert!(!scaler.check_overflow(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_check_overflow_detects_inf() {
        let scaler = GradScaler::new(1.0);
        assert!(scaler.check_overflow(&[f32::INFINITY]));
        assert!(scaler.check_overflow(&[f32::NEG_INFINITY]));
        assert!(!scaler.check_overflow(&[1.0, -1.0]));
    }

    // --- clip_grad_norm ---

    #[test]
    fn test_clip_grad_norm_no_clip_needed() {
        let mut grads = vec![0.6_f32, 0.8_f32]; // norm = 1.0
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 1.0_f32).abs() < 1.0e-5);
        // Values unchanged since norm == max_norm
        assert!((grads[0] - 0.6_f32).abs() < 1.0e-5);
        assert!((grads[1] - 0.8_f32).abs() < 1.0e-5);
    }

    #[test]
    fn test_clip_grad_norm_clips_large_gradients() {
        let mut grads = vec![3.0_f32, 4.0_f32]; // norm = 5.0
        let pre_clip_norm = clip_grad_norm(&mut grads, 1.0);
        assert!((pre_clip_norm - 5.0_f32).abs() < 1.0e-5);
        // After clipping the norm should be ~1.0
        let post_norm: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        assert!((post_norm - 1.0_f32).abs() < 1.0e-5);
    }

    #[test]
    fn test_clip_grad_norm_zero_max_is_noop() {
        let mut grads = vec![3.0_f32, 4.0_f32];
        let norm = clip_grad_norm(&mut grads, 0.0);
        assert_eq!(norm, 0.0_f32);
        // Gradients should be unchanged
        assert_eq!(grads[0], 3.0_f32);
        assert_eq!(grads[1], 4.0_f32);
    }

    // --- grads_are_finite ---

    #[test]
    fn test_grads_are_finite_with_nan() {
        assert!(!grads_are_finite(&[1.0, f32::NAN]));
    }

    #[test]
    fn test_grads_are_finite_with_inf() {
        assert!(!grads_are_finite(&[f32::INFINITY, 1.0]));
        assert!(!grads_are_finite(&[f32::NEG_INFINITY]));
    }

    #[test]
    fn test_grads_are_finite_all_finite() {
        assert!(grads_are_finite(&[1.0, -2.0, 0.001]));
        assert!(grads_are_finite(&[]));
    }

    // --- f16_precision_error ---

    #[test]
    fn test_f16_precision_error_small_for_moderate_values() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let err = f16_precision_error(&data);
        // For integer values the roundtrip should be near-exact
        assert!(err < 1.0e-3_f32, "Relative error {} unexpectedly large", err);
    }

    #[test]
    fn test_f16_precision_error_empty_slice() {
        assert_eq!(f16_precision_error(&[]), 0.0_f32);
    }

    // --- GradScaler::state_dict roundtrip ---

    #[test]
    fn test_grad_scaler_state_dict_roundtrip() {
        let mut scaler = GradScaler::with_config(512.0, 1.5, 0.25, 100).expect("valid config");
        // Advance state a bit
        scaler.step_update(&[0.1_f32]);
        scaler.step_update(&[f32::NAN]);

        let state = scaler.state_dict();

        // Create a fresh scaler and restore
        let mut scaler2 = GradScaler::new(1.0);
        scaler2.load_state_dict(state.clone());

        assert_eq!(scaler2.scale, state.scale);
        assert_eq!(scaler2.growth_factor, state.growth_factor);
        assert_eq!(scaler2.backoff_factor, state.backoff_factor);
        assert_eq!(scaler2.growth_interval, state.growth_interval);
        assert_eq!(scaler2.steps_since_overflow, state.steps_since_overflow);
    }

    // --- with_config validation ---

    #[test]
    fn test_with_config_rejects_bad_arguments() {
        assert!(GradScaler::with_config(0.0, 2.0, 0.5, 100).is_err());
        assert!(GradScaler::with_config(-1.0, 2.0, 0.5, 100).is_err());
        assert!(GradScaler::with_config(f32::NAN, 2.0, 0.5, 100).is_err());
        assert!(GradScaler::with_config(1.0, 1.0, 0.5, 100).is_err()); // growth_factor == 1
        assert!(GradScaler::with_config(1.0, 0.5, 0.5, 100).is_err()); // growth_factor < 1
        assert!(GradScaler::with_config(1.0, 2.0, 0.0, 100).is_err()); // backoff == 0
        assert!(GradScaler::with_config(1.0, 2.0, 1.0, 100).is_err()); // backoff == 1
        assert!(GradScaler::with_config(1.0, 2.0, 0.5, 0).is_err()); // interval == 0
    }

    // --- AutoCast context ---

    #[test]
    fn test_autocast_disabled_is_identity() {
        let ctx = AutoCast::disabled();
        let data = vec![1.0_f32, 2.5_f32, -3.14_f32];
        let out = ctx.cast_input(&data);
        assert_eq!(out, data);
        assert_eq!(ctx.cast_scalar(1.1_f32), 1.1_f32);
    }

    #[test]
    fn test_autocast_f16_quantises_values() {
        let ctx = AutoCast::new(AutoCastDtype::Float16);
        let x = 1.1_f32;
        let cast = ctx.cast_scalar(x);
        // 1.1 is not exactly representable in FP16
        assert!((cast - x).abs() > 0.0);
        assert!((cast - x).abs() < 0.01);
    }

    #[test]
    fn test_autocast_bf16_quantises_values() {
        let ctx = AutoCast::new(AutoCastDtype::BFloat16);
        let x = 1.1_f32;
        let cast = ctx.cast_scalar(x);
        assert!((cast - x).abs() > 0.0);
        assert!((cast - x).abs() < 0.01);
    }

    // --- f32_to_f16_roundtrip ---

    #[test]
    fn test_f32_to_f16_roundtrip_preserves_length() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = f32_to_f16_roundtrip(&data);
        assert_eq!(out.len(), data.len());
    }

    #[test]
    fn test_f32_to_f16_roundtrip_integer_values_exact() {
        // Small integers are exactly representable in FP16
        let data = vec![1.0_f32, 2.0, 4.0, 8.0, 16.0];
        let out = f32_to_f16_roundtrip(&data);
        for (orig, rounded) in data.iter().zip(out.iter()) {
            assert_eq!(orig, rounded, "Integer {} should round-trip exactly through FP16", orig);
        }
    }
}
