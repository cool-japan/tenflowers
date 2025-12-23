//! # Gradient Computation Utilities
//!
//! This module provides convenient utility functions for common gradient computation
//! patterns, making it easier to work with automatic differentiation in practice.
//!
//! ## Features
//!
//! - **Gradient Clipping**: Various strategies for preventing gradient explosion
//! - **Gradient Analysis**: Tools for inspecting and debugging gradients
//! - **Gradient Scaling**: Utilities for scaling gradients during training
//! - **Gradient Accumulation**: Helpers for gradient accumulation patterns
//! - **Gradient Validation**: Quick checks for gradient health
//!
//! ## Usage Examples
//!
//! ### Gradient Clipping by Norm
//!
//! ```rust,no_run
//! use tenflowers_autograd::gradient_utils;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut grads = vec![
//!     Tensor::<f32>::ones(&[100, 10]),
//!     Tensor::<f32>::ones(&[10, 5]),
//! ];
//!
//! // Clip gradients to max norm of 1.0
//! gradient_utils::clip_gradients_by_norm(&mut grads, 1.0)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Gradient Validation
//!
//! ```rust,no_run
//! use tenflowers_autograd::gradient_utils;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let grads = vec![tenflowers_core::Tensor::<f32>::ones(&[10])];
//! // Check if gradients contain NaN or Inf
//! if !gradient_utils::are_gradients_finite(&grads)? {
//!     println!("Warning: Non-finite gradients detected!");
//! }
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Clip gradients by global norm
///
/// Computes the global norm of all gradients and scales them down if it exceeds
/// the specified maximum norm. This is useful for preventing gradient explosion.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `max_norm` - Maximum allowed global norm
///
/// # Returns
///
/// The actual global norm before clipping
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::clip_gradients_by_norm;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// let actual_norm = clip_gradients_by_norm(&mut grads, 1.0)?;
/// println!("Clipped from norm {} to 1.0", actual_norm);
/// # Ok(())
/// # }
/// ```
pub fn clip_gradients_by_norm(gradients: &mut [Tensor<f32>], max_norm: f32) -> Result<f32> {
    // Compute global norm
    let global_norm = compute_global_norm(gradients)?;

    // Scale if necessary
    if global_norm > max_norm {
        let scale = max_norm / global_norm;
        for grad in gradients.iter_mut() {
            *grad = grad.mul_scalar(scale)?;
        }
    }

    Ok(global_norm)
}

/// Clip gradients by value
///
/// Clips each gradient element to be within [-clip_value, clip_value].
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `clip_value` - Maximum absolute value
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::clip_gradients_by_value;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// clip_gradients_by_value(&mut grads, 1.0)?;
/// # Ok(())
/// # }
/// ```
pub fn clip_gradients_by_value(gradients: &mut [Tensor<f32>], clip_value: f32) -> Result<()> {
    for grad in gradients.iter_mut() {
        *grad = grad.clamp(-clip_value, clip_value)?;
    }
    Ok(())
}

/// Compute the global norm of gradients
///
/// Computes sqrt(sum(grad^2)) across all gradients.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
///
/// # Returns
///
/// The global L2 norm
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::compute_global_norm;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// let norm = compute_global_norm(&grads)?;
/// println!("Global norm: {}", norm);
/// # Ok(())
/// # }
/// ```
pub fn compute_global_norm(gradients: &[Tensor<f32>]) -> Result<f32> {
    let mut sum_sq = 0.0f32;

    for grad in gradients {
        let grad_sq = grad.pow(2.0)?;
        let grad_sum: f32 = grad_sq.sum()?.to_scalar()?;
        sum_sq += grad_sum;
    }

    Ok(sum_sq.sqrt())
}

/// Check if all gradients are finite (no NaN or Inf)
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
///
/// # Returns
///
/// `true` if all gradients are finite, `false` otherwise
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::are_gradients_finite;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// assert!(are_gradients_finite(&grads)?);
/// # Ok(())
/// # }
/// ```
pub fn are_gradients_finite(gradients: &[Tensor<f32>]) -> Result<bool> {
    for grad in gradients {
        if !grad.is_finite()? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Get gradient statistics for debugging
///
/// Computes useful statistics about gradients including min, max, mean, and std.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
///
/// # Returns
///
/// A `GradientStats` struct containing statistics
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::get_gradient_stats;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// let stats = get_gradient_stats(&grads)?;
/// println!("Gradient stats: min={}, max={}, mean={}", stats.min, stats.max, stats.mean);
/// # Ok(())
/// # }
/// ```
pub fn get_gradient_stats(gradients: &[Tensor<f32>]) -> Result<GradientUtilsStats> {
    if gradients.is_empty() {
        return Err(TensorError::invalid_input_simple(
            "Cannot compute stats for empty gradient list".to_string(),
        ));
    }

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut count = 0usize;

    for grad in gradients {
        let grad_min: f32 = grad.min()?.to_scalar()?;
        let grad_max: f32 = grad.max()?.to_scalar()?;
        let grad_sum: f32 = grad.sum()?.to_scalar()?;
        let grad_count = grad.shape().size();

        min_val = min_val.min(grad_min);
        max_val = max_val.max(grad_max);
        sum += grad_sum;
        count += grad_count;
    }

    let mean = sum / count as f32;

    // Compute std dev
    let mut var_sum = 0.0f32;
    for grad in gradients {
        let centered = grad.sub_scalar(mean)?;
        let squared = centered.pow(2.0)?;
        let squared_sum: f32 = squared.sum()?.to_scalar()?;
        var_sum += squared_sum;
    }

    let variance = var_sum / count as f32;
    let std_dev = variance.sqrt();

    Ok(GradientUtilsStats {
        min: min_val,
        max: max_val,
        mean,
        std_dev,
        l2_norm: compute_global_norm(gradients)?,
        num_elements: count,
        num_tensors: gradients.len(),
    })
}

/// Statistics about a set of gradients
#[derive(Debug, Clone)]
pub struct GradientUtilsStats {
    /// Minimum value across all gradients
    pub min: f32,
    /// Maximum value across all gradients
    pub max: f32,
    /// Mean value across all gradients
    pub mean: f32,
    /// Standard deviation across all gradients
    pub std_dev: f32,
    /// Global L2 norm
    pub l2_norm: f32,
    /// Total number of elements
    pub num_elements: usize,
    /// Number of gradient tensors
    pub num_tensors: usize,
}

impl GradientUtilsStats {
    /// Check if gradients show signs of vanishing (very small)
    pub fn has_vanishing_gradients(&self, threshold: f32) -> bool {
        self.l2_norm < threshold
    }

    /// Check if gradients show signs of explosion (very large)
    pub fn has_exploding_gradients(&self, threshold: f32) -> bool {
        self.l2_norm > threshold || self.max > threshold || self.min < -threshold
    }

    /// Get a summary string for logging
    pub fn summary(&self) -> String {
        format!(
            "Gradients: min={:.6}, max={:.6}, mean={:.6}, std={:.6}, norm={:.6} ({} elements in {} tensors)",
            self.min, self.max, self.mean, self.std_dev, self.l2_norm,
            self.num_elements, self.num_tensors
        )
    }
}

/// Zero out gradients (useful for manual gradient reset)
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors to zero
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::zero_gradients;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// zero_gradients(&mut grads)?;
/// # Ok(())
/// # }
/// ```
pub fn zero_gradients(gradients: &mut [Tensor<f32>]) -> Result<()> {
    for grad in gradients.iter_mut() {
        *grad = Tensor::zeros_like(grad)?;
    }
    Ok(())
}

/// Apply gradient noise for regularization
///
/// Adds Gaussian noise to gradients, which can act as a regularizer.
/// Noise scale typically decays during training: noise_scale / (1 + t)^gamma
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `noise_scale` - Standard deviation of the noise
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::add_gradient_noise;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// add_gradient_noise(&mut grads, 0.01, Some(42))?;
/// # Ok(())
/// # }
/// ```
pub fn add_gradient_noise(
    gradients: &mut [Tensor<f32>],
    noise_scale: f32,
    seed: Option<u64>,
) -> Result<()> {
    use scirs2_core::random::Random;

    let mut rng = if let Some(s) = seed {
        Random::with_seed(s)
    } else {
        Random::new()
    };

    for grad in gradients.iter_mut() {
        let shape = grad.shape().dims();
        let noise = rng.normal(shape, 0.0, noise_scale as f64);
        let noise_tensor = Tensor::from_array(noise)?;
        *grad = grad.add(&noise_tensor)?;
    }

    Ok(())
}

/// Compute gradient histogram for analysis
///
/// Returns a histogram of gradient values, useful for visualizing gradient distributions.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
/// * `num_bins` - Number of histogram bins
///
/// # Returns
///
/// A vector of bin counts and bin edges
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::compute_gradient_histogram;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// let (counts, edges) = compute_gradient_histogram(&grads, 20)?;
/// println!("Histogram with {} bins", counts.len());
/// # Ok(())
/// # }
/// ```
pub fn compute_gradient_histogram(
    gradients: &[Tensor<f32>],
    num_bins: usize,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let stats = get_gradient_stats(gradients)?;

    // Create bin edges
    let bin_width = (stats.max - stats.min) / num_bins as f32;
    let mut bin_edges: Vec<f32> = (0..=num_bins)
        .map(|i| stats.min + i as f32 * bin_width)
        .collect();

    // Handle edge case where min == max
    if bin_width == 0.0 {
        bin_edges = vec![stats.min - 0.5, stats.min + 0.5];
    }

    let mut bin_counts = vec![0usize; num_bins];

    // Count values in each bin
    for grad in gradients {
        // Convert to array for efficient iteration
        let data = grad.to_vec()?;

        for &value in &data {
            // Find appropriate bin
            let value_f32 = value as f32;
            if value_f32 < stats.min || value_f32 > stats.max {
                continue;
            }

            let bin_idx = ((value_f32 - stats.min) / bin_width) as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Clamp to valid range

            bin_counts[bin_idx] += 1;
        }
    }

    Ok((bin_counts, bin_edges))
}

/// Apply adaptive gradient clipping based on percentile
///
/// Clips gradients based on a percentile of the gradient distribution,
/// which can be more robust than fixed thresholds.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `percentile` - Percentile to use for clipping (0.0-100.0)
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::clip_gradients_by_percentile;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// // Clip to 95th percentile
/// clip_gradients_by_percentile(&mut grads, 95.0)?;
/// # Ok(())
/// # }
/// ```
pub fn clip_gradients_by_percentile(
    gradients: &mut [Tensor<f32>],
    percentile: f32,
) -> Result<()> {
    if percentile <= 0.0 || percentile >= 100.0 {
        return Err(TensorError::invalid_input_simple(
            "Percentile must be between 0 and 100".to_string(),
        ));
    }

    // Collect all gradient values
    let mut all_values = Vec::new();
    for grad in gradients.iter() {
        let values = grad.to_vec()?;
        all_values.extend(values.iter().map(|&v| (v as f32).abs()));
    }

    // Sort to find percentile
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((percentile / 100.0) * all_values.len() as f32) as usize;
    let idx = idx.min(all_values.len() - 1);
    let threshold = all_values[idx];

    // Clip gradients
    clip_gradients_by_value(gradients, threshold)
}

/// Scale gradients by a factor
///
/// Multiplies all gradients by a scalar factor. Useful for learning rate warmup
/// or other scaling strategies.
///
/// # Arguments
///
/// * `gradients` - Mutable slice of gradient tensors
/// * `scale` - Scaling factor
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_utils::scale_gradients;
/// use tenflowers_core::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut grads = vec![Tensor::<f32>::ones(&[100, 10])];
/// // Scale down by half
/// scale_gradients(&mut grads, 0.5)?;
/// # Ok(())
/// # }
/// ```
pub fn scale_gradients(gradients: &mut [Tensor<f32>], scale: f32) -> Result<()> {
    for grad in gradients.iter_mut() {
        *grad = grad.mul_scalar(scale)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_gradients_by_norm() -> Result<()> {
        let mut grads = vec![
            Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2])?,
        ];

        // Initial norm should be 5.0 (sqrt(9 + 16))
        let norm = compute_global_norm(&grads)?;
        assert!((norm - 5.0).abs() < 1e-5);

        // Clip to max norm of 1.0
        let actual_norm = clip_gradients_by_norm(&mut grads, 1.0)?;
        assert!((actual_norm - 5.0).abs() < 1e-5);

        // New norm should be 1.0
        let new_norm = compute_global_norm(&grads)?;
        assert!((new_norm - 1.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_gradient_stats() -> Result<()> {
        let grads = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?,
        ];

        let stats = get_gradient_stats(&grads)?;

        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 3.0).abs() < 1e-5);
        assert!((stats.mean - 2.0).abs() < 1e-5);
        assert_eq!(stats.num_elements, 3);
        assert_eq!(stats.num_tensors, 1);

        Ok(())
    }

    #[test]
    fn test_are_gradients_finite() -> Result<()> {
        let grads = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?,
        ];

        assert!(are_gradients_finite(&grads)?);

        Ok(())
    }

    #[test]
    fn test_zero_gradients() -> Result<()> {
        let mut grads = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])?,
        ];

        zero_gradients(&mut grads)?;

        let sum: f32 = grads[0].sum()?.to_scalar()?;
        assert!((sum - 0.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_scale_gradients() -> Result<()> {
        let mut grads = vec![
            Tensor::<f32>::from_vec(vec![2.0, 4.0], &[2])?,
        ];

        scale_gradients(&mut grads, 0.5)?;

        let values = grads[0].to_vec()?;
        assert!((values[0] as f32 - 1.0).abs() < 1e-5);
        assert!((values[1] as f32 - 2.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_clip_by_value() -> Result<()> {
        let mut grads = vec![
            Tensor::<f32>::from_vec(vec![-5.0, 5.0, 0.5], &[3])?,
        ];

        clip_gradients_by_value(&mut grads, 1.0)?;

        let values = grads[0].to_vec()?;
        assert!((values[0] as f32 - (-1.0)).abs() < 1e-5);
        assert!((values[1] as f32 - 1.0).abs() < 1e-5);
        assert!((values[2] as f32 - 0.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_gradient_histogram() -> Result<()> {
        let grads = vec![
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?,
        ];

        let (counts, edges) = compute_gradient_histogram(&grads, 5)?;

        assert_eq!(counts.len(), 5);
        assert_eq!(edges.len(), 6);

        // Each bin should have approximately 1 element
        let total: usize = counts.iter().sum();
        assert_eq!(total, 5);

        Ok(())
    }
}
