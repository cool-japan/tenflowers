//! # Neural Quantization Module
//!
//! Comprehensive, pure-Rust quantization framework for model compression.
//!
//! This module provides standalone PTQ / QAT helpers that operate directly on
//! raw `f32` weight slices, independent of the higher-level deployment
//! quantization infrastructure.  It is designed for research, experimentation,
//! and lightweight PTQ/QAT workflows.
//!
//! ## Overview
//!
//! - [`QuantBits`] – INT4 / INT8 / Float16 bit-width enum.
//! - [`QuantScheme`] – symmetric vs. asymmetric integer mapping.
//! - [`QuantRange`] – scale, zero-point, and clamping range for one tensor.
//! - [`HistogramCalibrator`] – activation histogram for percentile-based calibration.
//! - [`QuantizedLinear`] – weight-only quantized linear layer for inference.
//! - [`QuantizationScheme`] – alias kept for backward-compat with older code.
//! - [`QuantizationConfig`] – bit-width, scheme, and per-channel settings.
//! - [`QuantizedTensor`] – compact integer container with scale / zero-point.
//! - [`PostTrainingQuantizer`] – calibration-based PTQ engine.
//!
//! ## Example
//!
//! ```rust,ignore
//! use tenflowers_neural::quantization::{
//!     QuantBits, QuantScheme, quantize_tensor, dequantize_tensor,
//! };
//!
//! let weights = vec![0.1_f32, -0.2, 0.5, -0.4];
//! let (q, range) = quantize_tensor(&weights, QuantBits::Int8, QuantScheme::Symmetric);
//! let dq = dequantize_tensor(&q, &range);
//! ```

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// QuantBits
// ─────────────────────────────────────────────────────────────────────────────

/// Bit-width for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantBits {
    /// 8-bit signed integer quantization.
    Int8,
    /// 4-bit signed integer quantization.
    Int4,
    /// 16-bit float (bfloat16/half) quantization.
    Float16,
}

impl QuantBits {
    /// Return the number of bits.
    pub fn bits(&self) -> u8 {
        match self {
            QuantBits::Int8 => 8,
            QuantBits::Int4 => 4,
            QuantBits::Float16 => 16,
        }
    }

    /// Return the maximum representable signed integer for this bit-width.
    ///
    /// For `Float16` this returns 127 as a placeholder (not meaningful for
    /// integer arithmetic but avoids panics in mixed code paths).
    pub fn qmax(&self) -> i32 {
        match self {
            QuantBits::Int8 => 127,
            QuantBits::Int4 => 7,
            QuantBits::Float16 => 127,
        }
    }

    /// Return the minimum representable signed integer for this bit-width.
    pub fn qmin(&self) -> i32 {
        match self {
            QuantBits::Int8 => -128,
            QuantBits::Int4 => -8,
            QuantBits::Float16 => -128,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantScheme
// ─────────────────────────────────────────────────────────────────────────────

/// Selects whether the integer range is centred on zero (symmetric) or uses an
/// explicit offset (asymmetric).
///
/// * **Symmetric**  – zero-point is always 0.
/// * **Asymmetric** – zero-point is chosen to cover `[min_val, max_val]` fully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// Zero-point is always 0.  Preserves symmetry around zero.
    Symmetric,
    /// Zero-point is computed to align the minimum floating-point value with
    /// the minimum integer value.
    Asymmetric,
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantRange
// ─────────────────────────────────────────────────────────────────────────────

/// Scale and zero-point parameters for one quantized tensor (or one channel).
#[derive(Debug, Clone)]
pub struct QuantRange {
    /// Minimum floating-point value seen during calibration.
    pub min: f32,
    /// Maximum floating-point value seen during calibration.
    pub max: f32,
    /// Scale factor: `x_float ≈ scale * (q - zero_point)`.
    pub scale: f32,
    /// Integer zero-point offset.
    pub zero_point: i32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar quantization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the scale and zero-point for a tensor given its observed float range.
///
/// # Symmetric
/// `scale = max(|min_val|, |max_val|) / (2^(bits-1) - 1)`, `zero_point = 0`.
///
/// # Asymmetric
/// `scale = (max_val - min_val) / (2^bits - 1)`,
/// `zero_point = round(-min_val / scale)`, clamped to `[qmin, qmax]`.
pub fn compute_scale_zero_point(
    min_val: f32,
    max_val: f32,
    bits: QuantBits,
    scheme: QuantScheme,
) -> QuantRange {
    let qmin = bits.qmin();
    let qmax = bits.qmax();
    let epsilon = f32::EPSILON;

    match scheme {
        QuantScheme::Symmetric => {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = if abs_max < epsilon {
                1.0_f32
            } else {
                abs_max / qmax as f32
            };
            QuantRange {
                min: min_val,
                max: max_val,
                scale,
                zero_point: 0,
            }
        }
        QuantScheme::Asymmetric => {
            let range = max_val - min_val;
            let scale = if range < epsilon {
                1.0_f32
            } else {
                range / (qmax - qmin) as f32
            };
            let zp_float = (qmin as f32) - min_val / scale;
            let zero_point = zp_float.round().clamp(qmin as f32, qmax as f32) as i32;
            QuantRange {
                min: min_val,
                max: max_val,
                scale,
                zero_point,
            }
        }
    }
}

/// Quantize a single `f32` value to an integer using the given range.
///
/// The result is clamped to the representable integer range `[qmin, qmax]`.
pub fn quantize_value(x: f32, range: &QuantRange, bits: QuantBits) -> i32 {
    let qmin = bits.qmin() as f32;
    let qmax = bits.qmax() as f32;
    let q = (x / range.scale + range.zero_point as f32).round();
    q.clamp(qmin, qmax) as i32
}

/// Reconstruct a float from a quantized integer value.
///
/// `x_float ≈ scale * (q - zero_point)`
pub fn dequantize_value(q: i32, range: &QuantRange) -> f32 {
    range.scale * (q - range.zero_point) as f32
}

/// Quantize an entire slice of `f32` values.
///
/// Computes the per-tensor min/max, derives scale and zero-point, then maps
/// every element to an integer.
///
/// Returns `(quantized_integers, QuantRange)`.
pub fn quantize_tensor(
    data: &[f32],
    bits: QuantBits,
    scheme: QuantScheme,
) -> (Vec<i32>, QuantRange) {
    if data.is_empty() {
        let range = QuantRange {
            min: 0.0,
            max: 0.0,
            scale: 1.0,
            zero_point: 0,
        };
        return (Vec::new(), range);
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = compute_scale_zero_point(min_val, max_val, bits, scheme);
    let quantized = data
        .iter()
        .map(|&x| quantize_value(x, &range, bits))
        .collect();
    (quantized, range)
}

/// Dequantize a slice of integers back to `f32`.
pub fn dequantize_tensor(data: &[i32], range: &QuantRange) -> Vec<f32> {
    data.iter().map(|&q| dequantize_value(q, range)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-channel quantization
// ─────────────────────────────────────────────────────────────────────────────

/// Quantize a weight matrix with independent scale/zero-point per output channel.
///
/// The weight matrix is treated as having `num_channels` rows (output channels),
/// each of length `weights.len() / num_channels`.
///
/// # Errors
/// Returns an error if `num_channels == 0` or `weights.len() % num_channels != 0`.
pub fn per_channel_quantize(
    weights: &[f32],
    num_channels: usize,
    bits: QuantBits,
    scheme: QuantScheme,
) -> Result<(Vec<i32>, Vec<QuantRange>)> {
    if num_channels == 0 {
        return Err(TensorError::invalid_argument(
            "num_channels must be > 0 in per_channel_quantize".to_string(),
        ));
    }
    if weights.len() % num_channels != 0 {
        return Err(TensorError::invalid_argument(format!(
            "weights length {} is not divisible by num_channels {}",
            weights.len(),
            num_channels
        )));
    }

    let channel_size = weights.len() / num_channels;
    let mut all_q = Vec::with_capacity(weights.len());
    let mut all_ranges = Vec::with_capacity(num_channels);

    for c in 0..num_channels {
        let row = &weights[c * channel_size..(c + 1) * channel_size];
        let (q, range) = quantize_tensor(row, bits, scheme);
        all_q.extend_from_slice(&q);
        all_ranges.push(range);
    }

    Ok((all_q, all_ranges))
}

/// Dequantize a weight matrix that was quantized per-channel.
///
/// # Errors
/// Returns an error if `ranges.len() == 0` or `data.len() % ranges.len() != 0`.
pub fn per_channel_dequantize(
    data: &[i32],
    ranges: &[QuantRange],
    num_channels: usize,
) -> Result<Vec<f32>> {
    if num_channels == 0 || ranges.len() != num_channels {
        return Err(TensorError::invalid_argument(format!(
            "num_channels={num_channels} must match ranges.len()={}",
            ranges.len()
        )));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() % num_channels != 0 {
        return Err(TensorError::invalid_argument(format!(
            "data length {} is not divisible by num_channels {num_channels}",
            data.len()
        )));
    }

    let channel_size = data.len() / num_channels;
    let mut out = Vec::with_capacity(data.len());

    for c in 0..num_channels {
        let row = &data[c * channel_size..(c + 1) * channel_size];
        let range = &ranges[c];
        for &q in row {
            out.push(dequantize_value(q, range));
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantization-Aware Training helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Straight-Through Estimator (STE) for backward pass through quantization.
///
/// Passes gradients through for values that were **not** clipped during
/// forward quantization; zeros out gradients for clipped values.
///
/// `clamped_mask[i] == true` means element `i` was clipped (gradient → 0).
///
/// # Errors
/// Returns an error if `grad_output.len() != clamped_mask.len()`.
pub fn straight_through_estimator(grad_output: &[f32], clamped_mask: &[bool]) -> Result<Vec<f32>> {
    if grad_output.len() != clamped_mask.len() {
        return Err(TensorError::invalid_argument(format!(
            "grad_output length {} != clamped_mask length {}",
            grad_output.len(),
            clamped_mask.len()
        )));
    }
    let out = grad_output
        .iter()
        .zip(clamped_mask.iter())
        .map(|(&g, &c)| if c { 0.0_f32 } else { g })
        .collect();
    Ok(out)
}

/// Fake-quantize a single value: quantize then immediately dequantize.
///
/// Used during QAT to simulate quantization noise while keeping the data in
/// floating-point for gradient computation.
pub fn fake_quantize(x: f32, bits: QuantBits, scheme: QuantScheme) -> f32 {
    // Single-value fake-quantize: use [x, x] as the range.
    let range = compute_scale_zero_point(x, x, bits, scheme);
    let q = quantize_value(x, &range, bits);
    dequantize_value(q, &range)
}

/// Fake-quantize an entire slice: quantize then immediately dequantize.
///
/// The range is computed from the entire slice, matching typical QAT practice.
pub fn fake_quantize_tensor(data: &[f32], bits: QuantBits, scheme: QuantScheme) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    let (q, range) = quantize_tensor(data, bits, scheme);
    dequantize_tensor(&q, &range)
}

// ─────────────────────────────────────────────────────────────────────────────
// Activation quantization — HistogramCalibrator
// ─────────────────────────────────────────────────────────────────────────────

/// Builds a histogram of observed activation values to find the optimal
/// min/max range for quantization.
///
/// Call [`observe`](HistogramCalibrator::observe) with batches of activations,
/// then call [`calibrate`](HistogramCalibrator::calibrate) to obtain the
/// clipping range at the desired percentile.
#[derive(Debug, Clone)]
pub struct HistogramCalibrator {
    /// Number of histogram bins.
    pub num_bins: usize,
    /// Minimum value seen so far across all observations.
    observed_min: f32,
    /// Maximum value seen so far across all observations.
    observed_max: f32,
    /// Histogram bin counts.
    bins: Vec<u64>,
    /// Total number of observed values.
    total_count: u64,
    /// Whether any data has been observed yet.
    initialised: bool,
}

impl HistogramCalibrator {
    /// Create a new `HistogramCalibrator` with `num_bins` histogram bins.
    ///
    /// # Errors
    /// Returns an error if `num_bins == 0`.
    pub fn new(num_bins: usize) -> Result<Self> {
        if num_bins == 0 {
            return Err(TensorError::invalid_argument(
                "num_bins must be > 0 in HistogramCalibrator".to_string(),
            ));
        }
        Ok(Self {
            num_bins,
            observed_min: f32::INFINITY,
            observed_max: f32::NEG_INFINITY,
            bins: vec![0u64; num_bins],
            total_count: 0,
            initialised: false,
        })
    }

    /// Incorporate a batch of activation values into the histogram.
    ///
    /// On the first call the global min/max is established.  On subsequent
    /// calls the histogram is recomputed if the range expands.
    pub fn observe(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let batch_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range_expanded =
            !self.initialised || batch_min < self.observed_min || batch_max > self.observed_max;

        if range_expanded {
            self.observed_min = self.observed_min.min(batch_min);
            self.observed_max = self.observed_max.max(batch_max);
            self.initialised = true;
            // Recount all bins with the new range.
            self.bins.fill(0);
            self.total_count = 0;
        }

        let range = self.observed_max - self.observed_min;
        let scale = if range < f32::EPSILON {
            1.0_f32
        } else {
            self.num_bins as f32 / range
        };

        for &v in data {
            let idx = ((v - self.observed_min) * scale) as usize;
            let idx = idx.min(self.num_bins - 1);
            self.bins[idx] += 1;
            self.total_count += 1;
        }
    }

    /// Return the `(min, max)` clipping range at the given percentile.
    ///
    /// `percentile = 100.0` returns the full observed range.
    /// `percentile = 99.9` clips the top and bottom 0.05% of values.
    ///
    /// # Errors
    /// Returns an error if no data has been observed or `percentile > 100.0`.
    pub fn calibrate(&self, percentile: f32) -> Result<(f32, f32)> {
        if !self.initialised || self.total_count == 0 {
            return Err(TensorError::invalid_argument(
                "HistogramCalibrator has no observed data".to_string(),
            ));
        }
        if !(0.0..=100.0).contains(&percentile) {
            return Err(TensorError::invalid_argument(format!(
                "percentile must be in [0.0, 100.0], got {percentile}"
            )));
        }

        let tail_frac = (1.0 - percentile / 100.0) / 2.0;
        let tail_count = (tail_frac * self.total_count as f32).round() as u64;

        let range = self.observed_max - self.observed_min;
        let bin_width = if range < f32::EPSILON {
            1.0_f32
        } else {
            range / self.num_bins as f32
        };

        // Find lower clip bin.
        let mut lower_bin = 0usize;
        let mut acc = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            acc += count;
            if acc > tail_count {
                lower_bin = i;
                break;
            }
        }

        // Find upper clip bin.
        let mut upper_bin = self.num_bins - 1;
        let mut acc = 0u64;
        for (i, &count) in self.bins.iter().enumerate().rev() {
            acc += count;
            if acc > tail_count {
                upper_bin = i;
                break;
            }
        }

        let calibrated_min = self.observed_min + lower_bin as f32 * bin_width;
        let calibrated_max = self.observed_min + (upper_bin + 1) as f32 * bin_width;

        // Ensure min <= max.
        let final_min = calibrated_min.min(calibrated_max);
        let final_max = calibrated_min.max(calibrated_max);
        Ok((final_min, final_max))
    }

    /// Clear all accumulated observations and reset to the initial state.
    pub fn reset(&mut self) {
        self.bins.fill(0);
        self.total_count = 0;
        self.observed_min = f32::INFINITY;
        self.observed_max = f32::NEG_INFINITY;
        self.initialised = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuantizedLinear
// ─────────────────────────────────────────────────────────────────────────────

/// A weight-quantized linear (fully-connected) layer for inference.
///
/// During construction, the weight matrix is quantized to integers.  At
/// inference time the weights are dequantized back to `f32` for the matrix
/// multiply, enabling exact gradient-free inference with compressed storage.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Quantized weight integers, shape `[out_features × in_features]` row-major.
    pub weight_q: Vec<i32>,
    /// Quantization range parameters for the weight matrix.
    pub weight_range: QuantRange,
    /// Optional bias vector of length `out_features`.
    pub bias: Option<Vec<f32>>,
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Bit-width used for weight quantization.
    pub bits: QuantBits,
    /// Quantization scheme used for weight quantization.
    pub scheme: QuantScheme,
}

impl QuantizedLinear {
    /// Construct a `QuantizedLinear` by quantizing a floating-point weight matrix.
    ///
    /// # Parameters
    /// * `weights`      – flat `[out_features × in_features]` weight matrix.
    /// * `bias`         – optional bias slice of length `out_features`.
    /// * `in_features`  – number of input features.
    /// * `out_features` – number of output features.
    /// * `bits`         – quantization bit-width.
    /// * `scheme`       – symmetric or asymmetric quantization.
    ///
    /// # Errors
    /// Returns an error when the weight length does not equal
    /// `in_features * out_features`, or the bias length does not equal
    /// `out_features`.
    pub fn from_float_weights(
        weights: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        bits: QuantBits,
        scheme: QuantScheme,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(TensorError::invalid_argument(
                "in_features and out_features must both be > 0".to_string(),
            ));
        }
        let expected = in_features * out_features;
        if weights.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "weights length {} != in_features({}) * out_features({}) = {}",
                weights.len(),
                in_features,
                out_features,
                expected
            )));
        }
        if let Some(b) = bias {
            if b.len() != out_features {
                return Err(TensorError::invalid_argument(format!(
                    "bias length {} != out_features {}",
                    b.len(),
                    out_features
                )));
            }
        }

        let (weight_q, weight_range) = quantize_tensor(weights, bits, scheme);

        Ok(Self {
            weight_q,
            weight_range,
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
            bits,
            scheme,
        })
    }

    /// Forward pass: dequantize weights and compute `output = input @ W^T + bias`.
    ///
    /// # Parameters
    /// * `input` – flat `[in_features]` input vector.
    ///
    /// # Returns
    /// A `Vec<f32>` of length `out_features`.
    ///
    /// # Errors
    /// Returns an error if `input.len() != self.in_features`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(TensorError::invalid_argument(format!(
                "input length {} != in_features {}",
                input.len(),
                self.in_features
            )));
        }

        // Dequantize weight matrix.
        let w_dq = dequantize_tensor(&self.weight_q, &self.weight_range);

        // Matrix-vector multiply: output[o] = sum_i w[o*in_features + i] * input[i]
        let mut output = vec![0.0_f32; self.out_features];
        for o in 0..self.out_features {
            let mut acc = 0.0_f32;
            for i in 0..self.in_features {
                acc += w_dq[o * self.in_features + i] * input[i];
            }
            output[o] = acc;
        }

        // Add bias if present.
        if let Some(ref b) = self.bias {
            for (o, &bv) in b.iter().enumerate() {
                output[o] += bv;
            }
        }

        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward-compat: QuantizationScheme, QuantizationConfig, QuantizedTensor,
// PostTrainingQuantizer, compute_scale_and_zero_point
// ─────────────────────────────────────────────────────────────────────────────

/// Selects whether the integer range is centred on zero (symmetric) or uses an
/// explicit offset (asymmetric).
///
/// * **Symmetric** – zero-point is always 0; the range is `[-2^(bits-1), 2^(bits-1)-1]`.
/// * **Asymmetric** – zero-point is chosen so that the full floating-point range maps
///   onto the full integer range `[-2^(bits-1), 2^(bits-1)-1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationScheme {
    /// Zero-point is always 0.  Preserves symmetry around zero.
    Symmetric,
    /// Zero-point is computed to align the min floating-point value with the
    /// minimum integer value.  More expressive but slightly slower.
    Asymmetric,
}

/// Configuration for the quantization process.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Number of integer bits.  Supported values: 4 or 8.
    pub bits: u8,
    /// Symmetric or asymmetric mapping.
    pub scheme: QuantizationScheme,
    /// When `true`, scale and zero-point are computed per output channel rather
    /// than globally across the whole tensor.
    pub per_channel: bool,
}

impl QuantizationConfig {
    /// Validate the configuration and return an error on unsupported settings.
    pub fn validate(&self) -> Result<()> {
        if self.bits != 4 && self.bits != 8 {
            return Err(TensorError::invalid_argument(format!(
                "Unsupported bit-width: {}. Only 4 or 8 bits are supported.",
                self.bits
            )));
        }
        Ok(())
    }

    /// Returns the minimum representable integer for this configuration.
    pub fn qmin(&self) -> i32 {
        -(1_i32 << (self.bits - 1))
    }

    /// Returns the maximum representable integer for this configuration.
    pub fn qmax(&self) -> i32 {
        (1_i32 << (self.bits - 1)) - 1
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            scheme: QuantizationScheme::Symmetric,
            per_channel: false,
        }
    }
}

/// A quantized tensor: a compact integer representation together with the scale
/// and zero-point needed to recover (approximately) the original floating-point
/// values.
///
/// The dequantized value is recovered as:
/// ```text
/// x_float ≈ scale * (q - zero_point)
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer values (stored as `i8` for memory efficiency).
    pub data: Vec<i8>,
    /// Per-tensor (or per-channel) scale factor.
    pub scale: f32,
    /// Per-tensor (or per-channel) zero point.
    pub zero_point: i8,
    /// Shape of the original tensor.
    pub shape: Vec<usize>,
    /// The quantization configuration used.
    pub config: QuantizationConfig,
}

impl QuantizedTensor {
    /// Quantize a flat `f32` buffer using `config`.
    ///
    /// # Parameters
    /// * `data`   – flat, row-major floating-point data.
    /// * `shape`  – dimensions; `shape.iter().product()` must equal `data.len()`.
    /// * `config` – quantization settings.
    pub fn quantize(data: &[f32], shape: &[usize], config: QuantizationConfig) -> Result<Self> {
        config.validate()?;

        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(TensorError::invalid_argument(format!(
                "Data length {} does not match shape product {}",
                data.len(),
                expected_len,
            )));
        }

        let (scale, zero_point) = compute_scale_and_zero_point(data, &config);

        let qmin = config.qmin() as f32;
        let qmax = config.qmax() as f32;

        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| {
                let q = (x / scale + zero_point as f32).round();
                q.clamp(qmin, qmax) as i8
            })
            .collect();

        Ok(Self {
            data: quantized,
            scale,
            zero_point,
            shape: shape.to_vec(),
            config,
        })
    }

    /// Reconstruct approximately the original `f32` values.
    ///
    /// Applies the formula `x ≈ scale * (q - zero_point)` element-wise.
    pub fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| self.scale * (q as f32 - self.zero_point as f32))
            .collect()
    }

    /// Compute the mean-squared quantization error between the dequantized
    /// approximation and the supplied original values.
    ///
    /// Lower is better.  Returns 0.0 if `original` is empty.
    ///
    /// # Errors
    /// Returns an error if `original.len()` does not match the stored `data.len()`.
    pub fn quantization_error(&self, original: &[f32]) -> Result<f32> {
        if original.len() != self.data.len() {
            return Err(TensorError::invalid_argument(format!(
                "original length {} != quantized length {}",
                original.len(),
                self.data.len()
            )));
        }
        if original.is_empty() {
            return Ok(0.0);
        }

        let dequantized = self.dequantize();
        let mse = original
            .iter()
            .zip(dequantized.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / original.len() as f32;
        Ok(mse)
    }

    /// Total number of quantized elements.
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// Return the bit-width of this quantized tensor.
    pub fn bits(&self) -> u8 {
        self.config.bits
    }
}

/// Compute the scale factor and zero-point for a slice of `f32` values.
///
/// Supports both symmetric and asymmetric schemes and 4-bit / 8-bit precision.
///
/// Returns `(scale, zero_point)`.  `scale` is always positive; `zero_point` is
/// clamped to the representable integer range.
pub fn compute_scale_and_zero_point(data: &[f32], config: &QuantizationConfig) -> (f32, i8) {
    if data.is_empty() {
        return (1.0, 0);
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let qmin = config.qmin() as f32;
    let qmax = config.qmax() as f32;

    match config.scheme {
        QuantizationScheme::Symmetric => {
            let abs_max = max_val.abs().max(min_val.abs());
            let scale = if abs_max < f32::EPSILON {
                1.0
            } else {
                abs_max / qmax
            };
            (scale, 0)
        }
        QuantizationScheme::Asymmetric => {
            let range = max_val - min_val;
            let scale = if range < f32::EPSILON {
                1.0
            } else {
                range / (qmax - qmin)
            };
            let zp_float = qmin - min_val / scale;
            let zp = zp_float.round().clamp(qmin, qmax) as i8;
            (scale, zp)
        }
    }
}

/// A post-training quantizer that uses calibration samples to derive robust
/// scale and zero-point parameters before quantizing weight tensors.
///
/// ## Calibration
///
/// Call [`calibrate`](PostTrainingQuantizer::calibrate) one or more times with
/// representative activation / weight samples.  The quantizer tracks the global
/// min and max across all calibration calls and uses them (instead of the
/// per-tensor statistics) when quantizing.
///
/// ## Quantization
///
/// After calibration, call [`quantize_weights`](PostTrainingQuantizer::quantize_weights)
/// to produce a [`QuantizedTensor`].
#[derive(Debug, Clone)]
pub struct PostTrainingQuantizer {
    /// Quantization configuration.
    config: QuantizationConfig,
    /// All calibration samples collected so far (concatenated).
    calibration_data: Vec<f32>,
}

impl PostTrainingQuantizer {
    /// Create a new `PostTrainingQuantizer` with the given configuration.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            calibration_data: Vec::new(),
        }
    }

    /// Append `samples` to the internal calibration buffer.
    ///
    /// May be called multiple times; each call extends the buffer.
    pub fn calibrate(&mut self, samples: &[f32]) {
        self.calibration_data.extend_from_slice(samples);
    }

    /// Quantize `weights` using the calibration statistics collected so far.
    ///
    /// If no calibration data has been provided, the per-tensor statistics of
    /// `weights` are used instead (i.e. the calibration buffer falls back to
    /// the data itself).
    ///
    /// # Parameters
    /// * `weights` – flat, row-major weight tensor.
    /// * `shape`   – tensor dimensions.
    pub fn quantize_weights(&self, weights: &[f32], shape: &[usize]) -> Result<QuantizedTensor> {
        self.config.validate()?;

        let expected: usize = shape.iter().product();
        if weights.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "Weights length {} does not match shape product {}",
                weights.len(),
                expected
            )));
        }

        let reference: &[f32] = if self.calibration_data.is_empty() {
            weights
        } else {
            &self.calibration_data
        };

        let (scale, zero_point) = self.compute_scale_and_zero_point(reference);

        let qmin = self.config.qmin() as f32;
        let qmax = self.config.qmax() as f32;

        let quantized: Vec<i8> = weights
            .iter()
            .map(|&x| {
                let q = (x / scale + zero_point as f32).round();
                q.clamp(qmin, qmax) as i8
            })
            .collect();

        Ok(QuantizedTensor {
            data: quantized,
            scale,
            zero_point,
            shape: shape.to_vec(),
            config: self.config.clone(),
        })
    }

    /// Compute scale and zero-point from a data slice using the stored config.
    pub fn compute_scale_and_zero_point(&self, data: &[f32]) -> (f32, i8) {
        compute_scale_and_zero_point(data, &self.config)
    }

    /// Returns the number of calibration samples currently stored.
    pub fn calibration_count(&self) -> usize {
        self.calibration_data.len()
    }

    /// Clear all calibration data.
    pub fn reset_calibration(&mut self) {
        self.calibration_data.clear();
    }

    /// Expose the current quantization configuration.
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config_8bit_sym() -> QuantizationConfig {
        QuantizationConfig {
            bits: 8,
            scheme: QuantizationScheme::Symmetric,
            per_channel: false,
        }
    }

    fn default_config_8bit_asym() -> QuantizationConfig {
        QuantizationConfig {
            bits: 8,
            scheme: QuantizationScheme::Asymmetric,
            per_channel: false,
        }
    }

    fn default_config_4bit_sym() -> QuantizationConfig {
        QuantizationConfig {
            bits: 4,
            scheme: QuantizationScheme::Symmetric,
            per_channel: false,
        }
    }

    // ── QuantBits ─────────────────────────────────────────────────────────────

    #[test]
    fn test_quant_bits_int8_bits() {
        assert_eq!(QuantBits::Int8.bits(), 8);
    }

    #[test]
    fn test_quant_bits_int4_bits() {
        assert_eq!(QuantBits::Int4.bits(), 4);
    }

    #[test]
    fn test_quant_bits_float16_bits() {
        assert_eq!(QuantBits::Float16.bits(), 16);
    }

    /// INT4 range is smaller than INT8.
    #[test]
    fn test_int4_range_smaller_than_int8() {
        let int4_range = QuantBits::Int4.qmax() - QuantBits::Int4.qmin();
        let int8_range = QuantBits::Int8.qmax() - QuantBits::Int8.qmin();
        assert!(
            int4_range < int8_range,
            "INT4 representable range should be smaller than INT8"
        );
    }

    // ── compute_scale_zero_point ──────────────────────────────────────────────

    /// Symmetric quantization has zero_point=0.
    #[test]
    fn test_symmetric_zero_point_is_zero() {
        let range = compute_scale_zero_point(-2.0, 3.0, QuantBits::Int8, QuantScheme::Symmetric);
        assert_eq!(
            range.zero_point, 0,
            "Symmetric quantization must have zero_point=0"
        );
        assert!(range.scale > 0.0, "scale must be positive");
    }

    /// Asymmetric quantization handles non-zero offset.
    #[test]
    fn test_asymmetric_nonzero_zero_point() {
        // Positive-only data: asymmetric should produce a non-zero zero_point.
        let range = compute_scale_zero_point(1.0, 5.0, QuantBits::Int8, QuantScheme::Asymmetric);
        assert!(
            range.zero_point != 0,
            "Asymmetric zero_point should be non-zero for positive-only data; got {}",
            range.zero_point
        );
        assert!(range.scale > 0.0);
    }

    // ── quantize_tensor / dequantize_tensor roundtrip ─────────────────────────

    /// Quantize then dequantize round-trips within 2 * scale tolerance.
    #[test]
    fn test_quantize_dequantize_roundtrip_int8_symmetric() {
        let data: Vec<f32> = (-50..=50).map(|x| x as f32 * 0.04).collect();
        let (q, range) = quantize_tensor(&data, QuantBits::Int8, QuantScheme::Symmetric);
        let dq = dequantize_tensor(&q, &range);
        assert_eq!(dq.len(), data.len());
        let tolerance = 2.0 * range.scale;
        for (&orig, &deq) in data.iter().zip(dq.iter()) {
            assert!(
                (orig - deq).abs() <= tolerance,
                "Round-trip error {:.6} > tolerance {:.6} for value {orig}",
                (orig - deq).abs(),
                tolerance
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_int8_asymmetric() {
        let data: Vec<f32> = (0..50).map(|x| x as f32 * 0.05).collect();
        let (q, range) = quantize_tensor(&data, QuantBits::Int8, QuantScheme::Asymmetric);
        let dq = dequantize_tensor(&q, &range);
        let tolerance = 2.0 * range.scale;
        for (&orig, &deq) in data.iter().zip(dq.iter()) {
            assert!(
                (orig - deq).abs() <= tolerance,
                "Round-trip error {:.6} > tolerance {:.6}",
                (orig - deq).abs(),
                tolerance
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_int4() {
        // 4-bit coarser; use larger tolerance.
        let data: Vec<f32> = (-6..=6).map(|x| x as f32).collect();
        let (q, range) = quantize_tensor(&data, QuantBits::Int4, QuantScheme::Symmetric);
        let dq = dequantize_tensor(&q, &range);
        let tolerance = 2.0 * range.scale;
        for (&orig, &deq) in data.iter().zip(dq.iter()) {
            assert!(
                (orig - deq).abs() <= tolerance,
                "Round-trip error {:.6} > tolerance {:.6} for value {orig}",
                (orig - deq).abs(),
                tolerance
            );
        }
    }

    // ── per_channel_quantize ──────────────────────────────────────────────────

    /// per_channel_quantize preserves per-channel statistics.
    #[test]
    fn test_per_channel_quantize_preserves_channel_stats() {
        // 2 channels, 4 weights each.  Channel 0: small range; channel 1: large range.
        let weights = vec![
            0.1_f32, 0.2, 0.1, 0.2, // channel 0: max abs ≈ 0.2
            10.0, -10.0, 5.0, -5.0, // channel 1: max abs = 10.0
        ];
        let (_, ranges) =
            per_channel_quantize(&weights, 2, QuantBits::Int8, QuantScheme::Symmetric)
                .expect("per_channel_quantize");

        assert_eq!(ranges.len(), 2);
        assert!(
            ranges[0].scale < ranges[1].scale,
            "Channel 0 (small range) should have smaller scale than channel 1 (large range)"
        );
    }

    #[test]
    fn test_per_channel_quantize_dequantize_roundtrip() {
        let weights: Vec<f32> = (0..12).map(|x| x as f32 * 0.1).collect();
        let num_ch = 3;
        let (q, ranges) =
            per_channel_quantize(&weights, num_ch, QuantBits::Int8, QuantScheme::Symmetric)
                .expect("quantize");
        let dq = per_channel_dequantize(&q, &ranges, num_ch).expect("dequantize");

        assert_eq!(dq.len(), weights.len());
        for (&orig, &deq) in weights.iter().zip(dq.iter()) {
            assert!(
                (orig - deq).abs() < 0.5,
                "Per-channel round-trip error too large: {orig} vs {deq}"
            );
        }
    }

    #[test]
    fn test_per_channel_quantize_error_on_indivisible_length() {
        let weights = vec![1.0_f32; 7];
        let result = per_channel_quantize(&weights, 3, QuantBits::Int8, QuantScheme::Symmetric);
        assert!(result.is_err());
    }

    // ── fake_quantize ─────────────────────────────────────────────────────────

    /// fake_quantize is idempotent (applying twice = applying once).
    #[test]
    fn test_fake_quantize_tensor_is_idempotent() {
        let data = vec![0.5_f32, -0.3, 1.2, -1.0, 0.0];
        let once = fake_quantize_tensor(&data, QuantBits::Int8, QuantScheme::Symmetric);
        let twice = fake_quantize_tensor(&once, QuantBits::Int8, QuantScheme::Symmetric);
        // Applying fake-quantize twice should give the same result as once.
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "fake_quantize should be idempotent; got {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_fake_quantize_preserves_approximate_values() {
        let x = 1.5_f32;
        let fq = fake_quantize(x, QuantBits::Int8, QuantScheme::Symmetric);
        assert!(
            (x - fq).abs() < 0.1,
            "fake_quantize should preserve value approximately; {x} vs {fq}"
        );
    }

    // ── straight_through_estimator ────────────────────────────────────────────

    /// STE passes gradient for unclipped values and zeros for clipped.
    #[test]
    fn test_straight_through_estimator_passes_grad_unclipped() {
        let grad = vec![1.0_f32, -2.0, 3.0, -1.5];
        let mask = vec![false, true, false, true]; // true = clipped
        let out = straight_through_estimator(&grad, &mask).expect("ste");
        assert_eq!(out, vec![1.0_f32, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_straight_through_estimator_all_unclipped() {
        let grad = vec![1.0_f32, 2.0, -3.0];
        let mask = vec![false, false, false];
        let out = straight_through_estimator(&grad, &mask).expect("ste");
        assert_eq!(out, grad);
    }

    #[test]
    fn test_straight_through_estimator_length_mismatch_error() {
        let grad = vec![1.0_f32, 2.0];
        let mask = vec![false];
        assert!(straight_through_estimator(&grad, &mask).is_err());
    }

    // ── HistogramCalibrator ───────────────────────────────────────────────────

    /// percentile 100% gives full range.
    #[test]
    fn test_histogram_calibrator_full_percentile_gives_full_range() {
        let mut cal = HistogramCalibrator::new(100).expect("calibrator");
        let data: Vec<f32> = (-50..=50).map(|x| x as f32).collect();
        cal.observe(&data);
        let (min, max) = cal.calibrate(100.0).expect("calibrate");
        assert!(
            min <= -50.0,
            "100% percentile min should cover full range; got {min}"
        );
        assert!(
            max >= 50.0,
            "100% percentile max should cover full range; got {max}"
        );
    }

    #[test]
    fn test_histogram_calibrator_error_on_no_data() {
        let cal = HistogramCalibrator::new(100).expect("calibrator");
        assert!(cal.calibrate(99.0).is_err());
    }

    #[test]
    fn test_histogram_calibrator_percentile_ordering() {
        let mut cal = HistogramCalibrator::new(200).expect("calibrator");
        let data: Vec<f32> = (0..200).map(|x| x as f32).collect();
        cal.observe(&data);

        let (min_100, max_100) = cal.calibrate(100.0).expect("100%");
        let (min_90, max_90) = cal.calibrate(90.0).expect("90%");

        // 90th percentile range should be contained within 100th.
        assert!(
            min_90 >= min_100 || (min_90 - min_100).abs() < 10.0,
            "90% min={min_90} should not be far below 100% min={min_100}"
        );
        assert!(
            max_90 <= max_100 || (max_90 - max_100).abs() < 10.0,
            "90% max={max_90} should not be far above 100% max={max_100}"
        );
    }

    #[test]
    fn test_histogram_calibrator_reset() {
        let mut cal = HistogramCalibrator::new(50).expect("calibrator");
        cal.observe(&[1.0, 2.0, 3.0]);
        cal.reset();
        assert!(
            cal.calibrate(100.0).is_err(),
            "After reset, no data should be present"
        );
    }

    // ── QuantizedLinear ───────────────────────────────────────────────────────

    /// QuantizedLinear forward output matches dequantized matmul.
    #[test]
    fn test_quantized_linear_forward_matches_dequantized_matmul() {
        // 2×3 weight matrix (out=2, in=3)
        let weights = vec![1.0_f32, 0.0, -1.0, 0.5, 0.5, 0.5];
        let ql = QuantizedLinear::from_float_weights(
            &weights,
            None,
            3,
            2,
            QuantBits::Int8,
            QuantScheme::Symmetric,
        )
        .expect("QuantizedLinear");

        let input = vec![1.0_f32, 2.0, 3.0];
        let output = ql.forward(&input).expect("forward");
        assert_eq!(output.len(), 2);

        // Reference: dequantize weights then matmul manually.
        let w_dq = dequantize_tensor(&ql.weight_q, &ql.weight_range);
        let ref_out0 = w_dq[0] * input[0] + w_dq[1] * input[1] + w_dq[2] * input[2];
        let ref_out1 = w_dq[3] * input[0] + w_dq[4] * input[1] + w_dq[5] * input[2];

        assert!(
            (output[0] - ref_out0).abs() < 1e-4,
            "output[0]={} != ref={}",
            output[0],
            ref_out0
        );
        assert!(
            (output[1] - ref_out1).abs() < 1e-4,
            "output[1]={} != ref={}",
            output[1],
            ref_out1
        );
    }

    #[test]
    fn test_quantized_linear_with_bias() {
        let weights = vec![1.0_f32, 0.0, -1.0, 0.0, 1.0, 0.0];
        let bias = vec![0.5_f32, -0.5];
        let ql = QuantizedLinear::from_float_weights(
            &weights,
            Some(&bias),
            3,
            2,
            QuantBits::Int8,
            QuantScheme::Symmetric,
        )
        .expect("QuantizedLinear with bias");

        let input = vec![1.0_f32, 1.0, 1.0];
        let output = ql.forward(&input).expect("forward");
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_quantized_linear_wrong_input_length_error() {
        let weights = vec![1.0_f32, 0.0, 0.0, 1.0];
        let ql = QuantizedLinear::from_float_weights(
            &weights,
            None,
            2,
            2,
            QuantBits::Int8,
            QuantScheme::Symmetric,
        )
        .expect("ql");
        let wrong_input = vec![1.0_f32]; // expects 2 inputs
        assert!(ql.forward(&wrong_input).is_err());
    }

    #[test]
    fn test_quantized_linear_wrong_weight_length_error() {
        let weights = vec![1.0_f32, 2.0, 3.0]; // 3 != 2*2
        let result = QuantizedLinear::from_float_weights(
            &weights,
            None,
            2,
            2,
            QuantBits::Int8,
            QuantScheme::Symmetric,
        );
        assert!(result.is_err());
    }

    // ── QuantizationConfig tests (backward-compat) ────────────────────────────

    #[test]
    fn test_quantization_config_qmin_qmax_8bit() {
        let cfg = default_config_8bit_sym();
        assert_eq!(cfg.qmin(), -128);
        assert_eq!(cfg.qmax(), 127);
    }

    #[test]
    fn test_quantization_config_qmin_qmax_4bit() {
        let cfg = default_config_4bit_sym();
        assert_eq!(cfg.qmin(), -8);
        assert_eq!(cfg.qmax(), 7);
    }

    #[test]
    fn test_quantization_config_validate_invalid_bits() {
        let cfg = QuantizationConfig {
            bits: 16,
            scheme: QuantizationScheme::Symmetric,
            per_channel: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_quantization_config_validate_valid_4bit() {
        let cfg = default_config_4bit_sym();
        assert!(cfg.validate().is_ok());
    }

    // ── QuantizedTensor: quantize / dequantize roundtrip ─────────────────────

    #[test]
    fn test_quantized_tensor_roundtrip_symmetric_8bit() {
        let data: Vec<f32> = (-50..=50).map(|x| x as f32 * 0.02).collect();
        let shape = vec![data.len()];
        let cfg = default_config_8bit_sym();

        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize should succeed");
        let dq = qt.dequantize();

        assert_eq!(dq.len(), data.len());
        let mse: f32 = data
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / data.len() as f32;
        assert!(
            mse < 1e-4,
            "MSE {mse} is larger than expected for 8-bit symmetric"
        );
    }

    #[test]
    fn test_quantized_tensor_roundtrip_asymmetric_8bit() {
        let data: Vec<f32> = (0..64).map(|x| x as f32 * 0.05).collect();
        let shape = vec![data.len()];
        let cfg = default_config_8bit_asym();

        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize should succeed");
        let dq = qt.dequantize();

        let mse: f32 = data
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / data.len() as f32;
        assert!(
            mse < 1e-3,
            "MSE {mse} is larger than expected for 8-bit asymmetric"
        );
    }

    #[test]
    fn test_quantized_tensor_roundtrip_4bit() {
        let data: Vec<f32> = (-7..=7).map(|x| x as f32).collect();
        let shape = vec![data.len()];
        let cfg = default_config_4bit_sym();

        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize should succeed");
        let dq = qt.dequantize();

        assert_eq!(dq.len(), data.len());
        for (&orig, &deq) in data.iter().zip(dq.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Large error for value {orig}: deq={deq}"
            );
        }
    }

    #[test]
    fn test_quantized_tensor_shape_mismatch_error() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let wrong_shape = vec![4];
        let cfg = default_config_8bit_sym();
        let result = QuantizedTensor::quantize(&data, &wrong_shape, cfg);
        assert!(result.is_err(), "Expected error for shape mismatch");
    }

    // ── quantization_error ────────────────────────────────────────────────────

    #[test]
    fn test_quantization_error_well_scaled_data() {
        let data: Vec<f32> = (0..128).map(|x| x as f32 * 0.01).collect();
        let shape = vec![data.len()];
        let cfg = default_config_8bit_sym();

        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize");
        let error = qt.quantization_error(&data).expect("error computation");
        assert!(
            error < 1e-4,
            "Quantization error {error} is unexpectedly large"
        );
    }

    #[test]
    fn test_quantization_error_length_mismatch() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![data.len()];
        let cfg = default_config_8bit_sym();
        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize");
        let wrong_original = vec![1.0_f32, 2.0];
        let result = qt.quantization_error(&wrong_original);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_error_zero_for_exact_roundtrip() {
        let data = vec![0.0_f32];
        let shape = vec![1];
        let cfg = default_config_8bit_sym();
        let qt = QuantizedTensor::quantize(&data, &shape, cfg).expect("quantize");
        let error = qt.quantization_error(&data).expect("error");
        assert!(
            error < 1e-9,
            "Expected near-zero error for all-zero input, got {error}"
        );
    }

    // ── PostTrainingQuantizer tests ───────────────────────────────────────────

    #[test]
    fn test_post_training_quantizer_calibrate_and_count() {
        let mut pq = PostTrainingQuantizer::new(default_config_8bit_sym());
        assert_eq!(pq.calibration_count(), 0);
        pq.calibrate(&[1.0, -1.0, 2.0, -2.0]);
        assert_eq!(pq.calibration_count(), 4);
        pq.calibrate(&[0.5, -0.5]);
        assert_eq!(pq.calibration_count(), 6);
    }

    #[test]
    fn test_post_training_quantizer_reset_calibration() {
        let mut pq = PostTrainingQuantizer::new(default_config_8bit_sym());
        pq.calibrate(&[1.0, 2.0, 3.0]);
        assert_eq!(pq.calibration_count(), 3);
        pq.reset_calibration();
        assert_eq!(pq.calibration_count(), 0);
    }

    #[test]
    fn test_post_training_quantizer_quantize_weights_basic() {
        let mut pq = PostTrainingQuantizer::new(default_config_8bit_sym());
        pq.calibrate(&[-1.0_f32, 1.0]);

        let weights = vec![0.5_f32, -0.5, 0.0, 1.0, -1.0];
        let shape = vec![weights.len()];

        let qt = pq
            .quantize_weights(&weights, &shape)
            .expect("quantize_weights should succeed");

        assert_eq!(qt.data.len(), weights.len());
        assert_eq!(qt.shape, shape);
        assert!(qt.scale > 0.0, "Scale must be positive");
    }

    #[test]
    fn test_post_training_quantizer_calibration_influences_scale() {
        let narrow_config = default_config_8bit_sym();
        let mut pq_narrow = PostTrainingQuantizer::new(narrow_config.clone());
        pq_narrow.calibrate(&[-0.1_f32, 0.1]);

        let wide_config = default_config_8bit_sym();
        let mut pq_wide = PostTrainingQuantizer::new(wide_config);
        pq_wide.calibrate(&[-100.0_f32, 100.0]);

        let weights = vec![0.0_f32, 0.05];
        let shape = vec![2];

        let qt_narrow = pq_narrow
            .quantize_weights(&weights, &shape)
            .expect("narrow quantize");
        let qt_wide = pq_wide
            .quantize_weights(&weights, &shape)
            .expect("wide quantize");

        assert!(
            qt_narrow.scale < qt_wide.scale,
            "Narrower calibration should produce smaller scale; narrow={}, wide={}",
            qt_narrow.scale,
            qt_wide.scale
        );
    }

    #[test]
    fn test_post_training_quantizer_fallback_without_calibration() {
        let pq = PostTrainingQuantizer::new(default_config_8bit_sym());
        let weights = vec![1.0_f32, -1.0, 0.5, -0.5];
        let shape = vec![weights.len()];

        let qt = pq
            .quantize_weights(&weights, &shape)
            .expect("should succeed with no calibration");
        assert_eq!(qt.data.len(), weights.len());
    }

    // ── compute_scale_and_zero_point standalone tests ────────────────────────

    #[test]
    fn test_compute_scale_symmetric_zero_point_is_zero() {
        let data = vec![-2.0_f32, 0.0, 2.0];
        let cfg = default_config_8bit_sym();
        let (scale, zp) = compute_scale_and_zero_point(&data, &cfg);
        assert_eq!(zp, 0, "Symmetric zero_point must be 0");
        assert!(scale > 0.0);
    }

    #[test]
    fn test_compute_scale_asymmetric_positive_data() {
        let data = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let cfg = default_config_8bit_asym();
        let (scale, zp) = compute_scale_and_zero_point(&data, &cfg);

        assert!(scale > 0.0, "scale must be positive; got {scale}");

        let q_zero = (0.0_f32 / scale + zp as f32).round();
        let qmin = cfg.qmin() as f32;
        let qmax = cfg.qmax() as f32;
        let q_zero_clamped = q_zero.clamp(qmin, qmax);

        let deq_zero = scale * (q_zero_clamped - zp as f32);
        assert!(
            deq_zero.abs() < scale + 0.1,
            "Dequantized min should be near 0.0; got {deq_zero} (scale={scale}, zp={zp})"
        );

        let q_max_val = (4.0_f32 / scale + zp as f32).round().clamp(qmin, qmax);
        let deq_max = scale * (q_max_val - zp as f32);
        assert!(
            (deq_max - 4.0).abs() < scale + 0.1,
            "Dequantized max should be near 4.0; got {deq_max} (scale={scale}, zp={zp})"
        );
    }

    #[test]
    fn test_compute_scale_empty_data() {
        let data: Vec<f32> = vec![];
        let cfg = default_config_8bit_sym();
        let (scale, zp) = compute_scale_and_zero_point(&data, &cfg);
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0);
    }
}
