//! Advanced Gradient Compression for Distributed Training
//!
//! This module provides sophisticated gradient compression techniques to reduce
//! communication overhead in distributed machine learning, including:
//! - Top-K sparsification
//! - Random sparsification
//! - Quantization (uniform, non-uniform)
//! - Error feedback/residual accumulation
//! - Hybrid compression schemes

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Number of non-zero elements (for sparse methods)
    pub num_nonzero: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Sparsity ratio
    pub sparsity: f32,
}

impl CompressionStats {
    /// Calculate compression statistics
    pub fn calculate(
        original_size: usize,
        compressed_size: usize,
        num_nonzero: usize,
        total_elements: usize,
    ) -> Self {
        let compression_ratio = original_size as f32 / compressed_size.max(1) as f32;
        let sparsity = 1.0 - (num_nonzero as f32 / total_elements.max(1) as f32);

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            num_nonzero,
            total_elements,
            sparsity,
        }
    }
}

/// Top-K sparsification
///
/// Keeps only the K largest (by absolute value) gradient elements.
/// This is one of the most popular gradient compression methods.
///
/// # Arguments
///
/// * `gradient` - Input gradient tensor
/// * `k` - Number of elements to keep
/// * `absolute` - If true, selects by absolute value; if false, by actual value
///
/// # Returns
///
/// Tuple of (sparse gradient, indices of kept elements, statistics)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::topk_sparsification;
/// use tenflowers_core::Tensor;
///
/// let grad = Tensor::from_vec(vec![0.1f32, -0.5, 0.3, -0.2, 0.05], vec![5])?;
/// let (sparse_grad, indices, stats) = topk_sparsification(&grad, 3, true)?;
///
/// // Keeps: -0.5, 0.3, -0.2 (top 3 by absolute value)
/// println!("Compression ratio: {:.2}x", stats.compression_ratio);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn topk_sparsification(
    gradient: &Tensor<f32>,
    k: usize,
    absolute: bool,
) -> Result<(Tensor<f32>, Vec<usize>, CompressionStats)> {
    let grad_data = gradient.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
    })?;

    let n = grad_data.len();

    if k > n {
        return Err(TensorError::invalid_operation_simple(format!(
            "K ({}) cannot exceed gradient size ({})",
            k, n
        )));
    }

    // Create (value, index) pairs
    let mut value_indices: Vec<(f32, usize)> = grad_data
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let sort_val = if absolute { val.abs() } else { val };
            (sort_val, i)
        })
        .collect();

    // Sort by value (descending)
    value_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Keep top K
    let mut sparse_data = vec![0.0f32; n];
    let mut kept_indices = Vec::with_capacity(k);

    for (_, idx) in value_indices.iter().take(k) {
        sparse_data[*idx] = grad_data[*idx];
        kept_indices.push(*idx);
    }

    let shape = gradient.shape().dims();
    let sparse_grad = Tensor::from_data(sparse_data, shape)?;

    // Calculate statistics
    let original_size = std::mem::size_of_val(grad_data);
    // Compressed: K values + K indices
    let compressed_size = k * std::mem::size_of::<f32>() + k * std::mem::size_of::<usize>();
    let stats = CompressionStats::calculate(original_size, compressed_size, k, n);

    Ok((sparse_grad, kept_indices, stats))
}

/// Random K sparsification
///
/// Randomly samples K gradient elements to keep. This is unbiased but has higher variance
/// than Top-K. Often combined with scaling to maintain expectation.
///
/// # Arguments
///
/// * `gradient` - Input gradient
/// * `k` - Number of elements to sample
/// * `seed` - Random seed for reproducibility
/// * `scale_up` - If true, scales kept values by n/k to maintain expectation
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::random_k_sparsification;
/// use tenflowers_core::Tensor;
///
/// let grad = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], vec![4])?;
/// let (sparse_grad, indices, stats) = random_k_sparsification(&grad, 2, Some(42), true)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn random_k_sparsification(
    gradient: &Tensor<f32>,
    k: usize,
    seed: Option<u64>,
    scale_up: bool,
) -> Result<(Tensor<f32>, Vec<usize>, CompressionStats)> {
    use scirs2_core::random::Random;

    let grad_data = gradient.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
    })?;

    let n = grad_data.len();

    if k > n {
        return Err(TensorError::invalid_operation_simple(format!(
            "K ({}) cannot exceed gradient size ({})",
            k, n
        )));
    }

    let mut rng = if let Some(s) = seed {
        Random::seed(s)
    } else {
        Random::seed(0)
    };

    // Randomly sample K indices without replacement
    let mut indices: Vec<usize> = (0..n).collect();

    // Fisher-Yates shuffle for first K elements
    for i in 0..k {
        // Use uniform distribution sampling
        let range = n - i;
        let random_val = (rng.random_f64() * range as f64).floor() as usize;
        let j = i + random_val;
        indices.swap(i, j);
    }

    let sampled_indices: Vec<usize> = indices[0..k].to_vec();

    // Create sparse gradient
    let mut sparse_data = vec![0.0f32; n];
    let scale_factor = if scale_up { n as f32 / k as f32 } else { 1.0 };

    for &idx in &sampled_indices {
        sparse_data[idx] = grad_data[idx] * scale_factor;
    }

    let shape = gradient.shape().dims();
    let sparse_grad = Tensor::from_data(sparse_data, shape)?;

    let original_size = std::mem::size_of_val(grad_data);
    let compressed_size = k * std::mem::size_of::<f32>() + k * std::mem::size_of::<usize>();
    let stats = CompressionStats::calculate(original_size, compressed_size, k, n);

    Ok((sparse_grad, sampled_indices, stats))
}

/// Uniform quantization
///
/// Quantizes gradient values to a fixed number of levels uniformly distributed
/// between min and max values.
///
/// # Arguments
///
/// * `gradient` - Input gradient
/// * `num_levels` - Number of quantization levels (e.g., 256 for 8-bit)
///
/// # Returns
///
/// Tuple of (quantized gradient, min value, max value, statistics)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::uniform_quantization;
/// use tenflowers_core::Tensor;
///
/// let grad = Tensor::from_vec(vec![0.1f32, 0.5, -0.3, 0.8], vec![4])?;
/// let (quant_grad, min_val, max_val, stats) = uniform_quantization(&grad, 256)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn uniform_quantization(
    gradient: &Tensor<f32>,
    num_levels: usize,
) -> Result<(Tensor<f32>, f32, f32, CompressionStats)> {
    let grad_data = gradient.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
    })?;

    if num_levels < 2 {
        return Err(TensorError::invalid_operation_simple(
            "Number of quantization levels must be at least 2".to_string(),
        ));
    }

    let n = grad_data.len();

    // Find min and max values
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in grad_data {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    // Handle edge case where all values are the same
    if (max_val - min_val).abs() < 1e-10 {
        return Ok((
            gradient.clone(),
            min_val,
            max_val,
            CompressionStats::calculate(
                std::mem::size_of_val(grad_data),
                std::mem::size_of_val(grad_data),
                n,
                n,
            ),
        ));
    }

    // Quantize
    let range = max_val - min_val;
    let scale = (num_levels - 1) as f32 / range;

    let mut quantized_data = Vec::with_capacity(n);

    for &val in grad_data {
        // Map to [0, num_levels-1]
        let normalized = (val - min_val) * scale;
        let quantized_level = normalized.round().clamp(0.0, (num_levels - 1) as f32);

        // Dequantize back to original range
        let dequantized = quantized_level / scale + min_val;
        quantized_data.push(dequantized);
    }

    let shape = gradient.shape().dims();
    let quantized_grad = Tensor::from_data(quantized_data, shape)?;

    // Calculate compressed size
    // Store: quantized indices + min + max
    let bits_per_value = (num_levels as f32).log2().ceil() as usize;
    let compressed_size = (n * bits_per_value + 7) / 8 + 2 * std::mem::size_of::<f32>();
    let original_size = std::mem::size_of_val(grad_data);

    let stats = CompressionStats::calculate(original_size, compressed_size, n, n);

    Ok((quantized_grad, min_val, max_val, stats))
}

/// Error feedback / residual accumulation
///
/// Maintains accumulated compression error and adds it to the next gradient
/// before compression. This compensates for information loss from compression.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::ErrorFeedbackCompressor;
/// use tenflowers_core::Tensor;
///
/// let mut compressor = ErrorFeedbackCompressor::new();
///
/// for iteration in 0..10 {
///     let grad = Tensor::ones(&[100]);  // Your gradient
///
///     // Compress with error feedback
///     let (compressed, stats) = compressor.compress_with_feedback(
///         &grad,
///         |g| {
///             // Your compression function (e.g., Top-K)
///             Ok(g.clone())  // Placeholder
///         }
///     )?;
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ErrorFeedbackCompressor {
    /// Accumulated error from previous compressions
    error_accumulator: HashMap<String, Tensor<f32>>,
}

impl ErrorFeedbackCompressor {
    /// Create new error feedback compressor
    pub fn new() -> Self {
        Self {
            error_accumulator: HashMap::new(),
        }
    }

    /// Compress gradient with error feedback
    ///
    /// # Arguments
    ///
    /// * `gradient` - Input gradient
    /// * `name` - Parameter name (for tracking separate error accumulators)
    /// * `compress_fn` - Compression function to apply
    ///
    /// # Returns
    ///
    /// Compressed gradient with error feedback applied
    pub fn compress_with_feedback<F>(
        &mut self,
        gradient: &Tensor<f32>,
        name: &str,
        compress_fn: F,
    ) -> Result<Tensor<f32>>
    where
        F: FnOnce(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Get accumulated error
        let accumulated_error = self
            .error_accumulator
            .get(name)
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(gradient.shape().dims()));

        // Add error to gradient: g' = g + error
        let gradient_with_error = gradient.add(&accumulated_error)?;

        // Apply compression
        let compressed = compress_fn(&gradient_with_error)?;

        // Calculate new error: error_new = g' - compressed
        let new_error = gradient_with_error.sub(&compressed)?;

        // Update error accumulator
        self.error_accumulator.insert(name.to_string(), new_error);

        Ok(compressed)
    }

    /// Reset error accumulator for a parameter
    pub fn reset(&mut self, name: &str) {
        self.error_accumulator.remove(name);
    }

    /// Reset all error accumulators
    pub fn reset_all(&mut self) {
        self.error_accumulator.clear();
    }
}

impl Default for ErrorFeedbackCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Power-law quantization (non-uniform)
///
/// Uses a power law to allocate more quantization levels to smaller values,
/// which is beneficial for gradients that follow a heavy-tailed distribution.
///
/// # Arguments
///
/// * `gradient` - Input gradient
/// * `num_levels` - Number of quantization levels
/// * `power` - Power for the quantization function (typically 0.5 for square root)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::power_law_quantization;
/// use tenflowers_core::Tensor;
///
/// let grad = Tensor::from_vec(vec![0.001f32, 0.1, 0.5], vec![3])?;
/// let (quant_grad, stats) = power_law_quantization(&grad, 256, 0.5)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn power_law_quantization(
    gradient: &Tensor<f32>,
    num_levels: usize,
    power: f32,
) -> Result<(Tensor<f32>, CompressionStats)> {
    let grad_data = gradient.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
    })?;

    let n = grad_data.len();

    if num_levels < 2 {
        return Err(TensorError::invalid_operation_simple(
            "Number of quantization levels must be at least 2".to_string(),
        ));
    }

    let mut quantized_data = Vec::with_capacity(n);

    for &val in grad_data {
        let sign = val.signum();
        let abs_val = val.abs();

        // Apply power law: x' = sign(x) * |x|^power
        let transformed = abs_val.powf(power);

        // Quantize in transformed space
        // This allocates more levels to smaller values
        let max_transformed = 1.0; // Assuming normalized inputs
        let normalized = transformed / max_transformed;
        let quantized_level = (normalized * (num_levels - 1) as f32)
            .round()
            .clamp(0.0, (num_levels - 1) as f32);

        // Dequantize: x = sign(x) * (x'^(1/power))
        let dequantized_transformed = quantized_level / (num_levels - 1) as f32 * max_transformed;
        let dequantized = sign * dequantized_transformed.powf(1.0 / power);

        quantized_data.push(dequantized);
    }

    let shape = gradient.shape().dims();
    let quantized_grad = Tensor::from_data(quantized_data, shape)?;

    let bits_per_value = (num_levels as f32).log2().ceil() as usize;
    let compressed_size = (n * bits_per_value + 7) / 8;
    let original_size = std::mem::size_of_val(grad_data);

    let stats = CompressionStats::calculate(original_size, compressed_size, n, n);

    Ok((quantized_grad, stats))
}

/// Threshold-based sparsification
///
/// Keeps only gradient elements whose absolute value exceeds a threshold.
/// This naturally adapts to gradient magnitude and can achieve high compression.
///
/// # Arguments
///
/// * `gradient` - Input gradient
/// * `threshold` - Minimum absolute value to keep
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_compression_advanced::threshold_sparsification;
/// use tenflowers_core::Tensor;
///
/// let grad = Tensor::from_vec(vec![0.001f32, 0.1, -0.05, 0.2], vec![4])?;
/// let (sparse_grad, stats) = threshold_sparsification(&grad, 0.05)?;
/// // Keeps: 0.1, -0.05, 0.2
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn threshold_sparsification(
    gradient: &Tensor<f32>,
    threshold: f32,
) -> Result<(Tensor<f32>, CompressionStats)> {
    let grad_data = gradient.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
    })?;

    let n = grad_data.len();
    let mut sparse_data = Vec::with_capacity(n);
    let mut num_kept = 0;

    for &val in grad_data {
        if val.abs() >= threshold {
            sparse_data.push(val);
            num_kept += 1;
        } else {
            sparse_data.push(0.0);
        }
    }

    let shape = gradient.shape().dims();
    let sparse_grad = Tensor::from_data(sparse_data, shape)?;

    let original_size = std::mem::size_of_val(grad_data);
    let compressed_size =
        num_kept * std::mem::size_of::<f32>() + num_kept * std::mem::size_of::<usize>();
    let stats = CompressionStats::calculate(original_size, compressed_size, num_kept, n);

    Ok((sparse_grad, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_sparsification() {
        // Use a larger tensor to demonstrate effective compression
        let mut grad_data = vec![0.0f32; 100];
        // Set a few large values
        grad_data[10] = 5.0;
        grad_data[25] = -4.0;
        grad_data[50] = 3.0;
        grad_data[75] = -2.0;
        grad_data[90] = 1.5;

        let grad = Tensor::from_data(grad_data.clone(), &[100]).unwrap();

        let (sparse_grad, indices, stats) = topk_sparsification(&grad, 5, true).unwrap();

        // Should keep only the 5 largest absolute values
        assert_eq!(stats.num_nonzero, 5);
        assert_eq!(stats.total_elements, 100);
        assert!(stats.compression_ratio > 1.0); // Compression should be effective

        let sparse_data = sparse_grad.as_slice().unwrap();
        // Verify the kept values are correct
        assert_eq!(sparse_data[10], 5.0);
        assert_eq!(sparse_data[25], -4.0);
        assert_eq!(sparse_data[50], 3.0);
        assert_eq!(sparse_data[75], -2.0);
        assert_eq!(sparse_data[90], 1.5);
    }

    #[test]
    fn test_uniform_quantization() {
        let grad_data = vec![0.0f32, 0.5, 1.0];
        let grad = Tensor::from_data(grad_data, &[3]).unwrap();

        let (quant_grad, min_val, max_val, stats) = uniform_quantization(&grad, 256).unwrap();

        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 1.0);
        assert!(stats.compression_ratio > 1.0);

        let quant_data = quant_grad.as_slice().unwrap();
        // Values should be close to original after quantization/dequantization
        assert!((quant_data[0] - 0.0).abs() < 0.01);
        assert!((quant_data[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_error_feedback() {
        let mut compressor = ErrorFeedbackCompressor::new();

        let grad = Tensor::from_data(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();

        // Simple compression that zeros everything
        let compressed = compressor
            .compress_with_feedback(&grad, "param1", |_g| {
                Tensor::from_data(vec![0.0f32, 0.0, 0.0], &[3])
            })
            .unwrap();

        // First compression should return zeros
        let compressed_data = compressed.as_slice().unwrap();
        assert_eq!(compressed_data, &[0.0, 0.0, 0.0]);

        // Error accumulator should now contain [1.0, 2.0, 3.0]
        // Next gradient should have this added
        let grad2 = Tensor::from_data(vec![0.0f32, 0.0, 0.0], &[3]).unwrap();

        let compressed2 = compressor
            .compress_with_feedback(&grad2, "param1", |g| Ok(g.clone()))
            .unwrap();

        let compressed2_data = compressed2.as_slice().unwrap();
        // Should recover the lost information
        assert!((compressed2_data[0] - 1.0).abs() < 1e-5);
        assert!((compressed2_data[1] - 2.0).abs() < 1e-5);
        assert!((compressed2_data[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_threshold_sparsification() {
        let grad_data = vec![0.001f32, 0.1, -0.05, 0.2, -0.001];
        let grad = Tensor::from_data(grad_data, &[5]).unwrap();

        let (sparse_grad, stats) = threshold_sparsification(&grad, 0.05).unwrap();

        // Should keep 3 elements: 0.1, -0.05, 0.2
        assert_eq!(stats.num_nonzero, 3);

        let sparse_data = sparse_grad.as_slice().unwrap();
        assert_eq!(sparse_data[0], 0.0);
        assert_eq!(sparse_data[1], 0.1);
        assert_eq!(sparse_data[2], -0.05);
        assert_eq!(sparse_data[3], 0.2);
        assert_eq!(sparse_data[4], 0.0);
    }
}
