//! Gradient Operation Utilities
//!
//! This module provides common gradient manipulation operations that are frequently
//! needed during training, including clipping, normalization, accumulation, and analysis.

use scirs2_core::numeric::Float;
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor, TensorError};

/// Statistics about gradient values
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Global norm across all gradients
    pub global_norm: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Minimum absolute value (non-zero)
    pub min_abs: f32,
    /// Mean absolute value
    pub mean_abs: f32,
    /// Number of NaN values
    pub num_nan: usize,
    /// Number of Inf values
    pub num_inf: usize,
    /// Number of zero values
    pub num_zero: usize,
    /// Total number of elements
    pub total_elements: usize,
}

impl GradientStatistics {
    /// Check if gradients are healthy (no NaN/Inf)
    pub fn is_healthy(&self) -> bool {
        self.num_nan == 0 && self.num_inf == 0
    }

    /// Get sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f32 {
        self.num_zero as f32 / self.total_elements as f32
    }
}

/// Clip gradients by global norm
///
/// This prevents exploding gradients by scaling all gradients by the same factor
/// when their global norm exceeds a threshold.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
/// * `max_norm` - Maximum allowed global norm
///
/// # Returns
///
/// Vector of clipped gradients and the actual global norm before clipping
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::clip_by_global_norm;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![
///     Tensor::from_scalar(1000.0f32),
///     Tensor::from_scalar(2000.0f32),
/// ];
///
/// let (clipped, original_norm) = clip_by_global_norm(&grads, 1.0)?;
/// assert!(original_norm > 1.0);  // Was clipped
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn clip_by_global_norm(
    gradients: &[Tensor<f32>],
    max_norm: f32,
) -> Result<(Vec<Tensor<f32>>, f32)> {
    // Compute global norm: sqrt(sum of squared norms)
    let mut total_norm_sq = 0.0f32;

    for gradient in gradients {
        let grad_data = gradient.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        for &val in grad_data {
            total_norm_sq += val * val;
        }
    }

    let global_norm = total_norm_sq.sqrt();

    if global_norm <= max_norm || global_norm == 0.0 {
        // No clipping needed
        return Ok((gradients.to_vec(), global_norm));
    }

    // Clip: multiply all gradients by (max_norm / global_norm)
    let clip_factor = max_norm / global_norm;
    let mut clipped_gradients = Vec::with_capacity(gradients.len());

    for gradient in gradients {
        let clipped = gradient.mul(&Tensor::from_scalar(clip_factor))?;
        clipped_gradients.push(clipped);
    }

    Ok((clipped_gradients, global_norm))
}

/// Clip gradients by value
///
/// Clamps each gradient element to [-clip_value, clip_value]
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
/// * `clip_value` - Maximum absolute value
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::clip_by_value;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::from_scalar(-5.0f32), Tensor::from_scalar(10.0f32)];
/// let clipped = clip_by_value(&grads, 1.0)?;
/// // Results: [-1.0, 1.0]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn clip_by_value(gradients: &[Tensor<f32>], clip_value: f32) -> Result<Vec<Tensor<f32>>> {
    let mut clipped_gradients = Vec::with_capacity(gradients.len());

    for gradient in gradients {
        let grad_data = gradient.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        let clipped_data: Vec<f32> = grad_data
            .iter()
            .map(|&val| val.clamp(-clip_value, clip_value))
            .collect();

        let clipped = Tensor::from_data(clipped_data, gradient.shape().dims())?;
        clipped_gradients.push(clipped);
    }

    Ok(clipped_gradients)
}

/// Compute statistics about gradients
///
/// Analyzes gradient values to detect potential issues like NaN, Inf, or extreme values.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::compute_gradient_statistics;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::from_data(vec![1.0f32, &2.0, 3.0], vec![3])?];
/// let stats = compute_gradient_statistics(&grads)?;
///
/// if !stats.is_healthy() {
///     println!("Warning: {} NaN, {} Inf detected", stats.num_nan, stats.num_inf);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compute_gradient_statistics(gradients: &[Tensor<f32>]) -> Result<GradientStatistics> {
    let mut total_norm_sq = 0.0f32;
    let mut max_abs = f32::MIN;
    let mut min_abs = f32::MAX;
    let mut sum_abs = 0.0f32;
    let mut num_nan = 0;
    let mut num_inf = 0;
    let mut num_zero = 0;
    let mut total_elements = 0;

    for gradient in gradients {
        let grad_data = gradient.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        for &val in grad_data {
            total_elements += 1;

            if val.is_nan() {
                num_nan += 1;
                continue;
            }

            if val.is_infinite() {
                num_inf += 1;
                continue;
            }

            let abs_val = val.abs();

            if abs_val == 0.0 {
                num_zero += 1;
            } else {
                min_abs = min_abs.min(abs_val);
            }

            max_abs = max_abs.max(abs_val);
            sum_abs += abs_val;
            total_norm_sq += val * val;
        }
    }

    let global_norm = total_norm_sq.sqrt();
    let mean_abs = if total_elements > 0 {
        sum_abs / total_elements as f32
    } else {
        0.0
    };

    // Handle edge cases
    if min_abs == f32::MAX {
        min_abs = 0.0;
    }
    if max_abs == f32::MIN {
        max_abs = 0.0;
    }

    Ok(GradientStatistics {
        global_norm,
        max_abs,
        min_abs,
        mean_abs,
        num_nan,
        num_inf,
        num_zero,
        total_elements,
    })
}

/// Zero out all gradients
///
/// Resets gradients to zero, useful for manual gradient accumulation.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::zero_gradients;
/// use tenflowers_core::Tensor;
///
/// let mut grads = vec![Tensor::ones(&[3, 3])];
/// zero_gradients(&mut grads)?;
/// // All gradients are now zero
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn zero_gradients(gradients: &mut [Tensor<f32>]) -> Result<()> {
    for gradient in gradients.iter_mut() {
        let shape = gradient.shape().dims();
        *gradient = Tensor::zeros(shape);
    }
    Ok(())
}

/// Scale gradients by a constant factor
///
/// Multiplies all gradients by the same scalar value.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::scale_gradients;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::from_scalar(2.0f32)];
/// let scaled = scale_gradients(&grads, 0.5)?;
/// // Result: [1.0]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn scale_gradients(gradients: &[Tensor<f32>], scale: f32) -> Result<Vec<Tensor<f32>>> {
    let scale_tensor = Tensor::from_scalar(scale);
    let mut scaled_gradients = Vec::with_capacity(gradients.len());

    for gradient in gradients {
        let scaled = gradient.mul(&scale_tensor)?;
        scaled_gradients.push(scaled);
    }

    Ok(scaled_gradients)
}

/// Accumulate gradients with optional averaging
///
/// Adds new gradients to accumulated gradients, with optional division by accumulation count.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::accumulate_gradients;
/// use tenflowers_core::Tensor;
///
/// let mut accumulated = vec![Tensor::ones(&[2])];
/// let new_grads = vec![Tensor::ones(&[2])];
///
/// accumulate_gradients(&mut accumulated, &new_grads)?;
/// // accumulated now contains [2.0, 2.0]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn accumulate_gradients(
    accumulated: &mut [Tensor<f32>],
    new_gradients: &[Tensor<f32>],
) -> Result<()> {
    if accumulated.len() != new_gradients.len() {
        return Err(TensorError::invalid_shape_simple(format!(
            "Gradient count mismatch: {} vs {}",
            accumulated.len(),
            new_gradients.len()
        )));
    }

    for (accum, new_grad) in accumulated.iter_mut().zip(new_gradients.iter()) {
        *accum = accum.add(new_grad)?;
    }

    Ok(())
}

/// Average accumulated gradients
///
/// Divides all gradients by a count, typically used after gradient accumulation.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::average_gradients;
/// use tenflowers_core::Tensor;
///
/// let mut grads = vec![Tensor::from_scalar(10.0f32)];
/// average_gradients(&mut grads, 5)?;
/// // Result: [2.0]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn average_gradients(gradients: &mut [Tensor<f32>], count: usize) -> Result<()> {
    if count == 0 {
        return Err(TensorError::invalid_operation_simple(
            "Cannot average by zero".to_string(),
        ));
    }

    let divisor = Tensor::from_scalar(count as f32);

    for gradient in gradients.iter_mut() {
        *gradient = gradient.div(&divisor)?;
    }

    Ok(())
}

/// Check if any gradient contains NaN or Inf
///
/// Returns true if any gradient has NaN or Inf values.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::has_invalid_gradients;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::from_scalar(f32::NAN)];
/// assert!(has_invalid_gradients(&grads)?);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn has_invalid_gradients(gradients: &[Tensor<f32>]) -> Result<bool> {
    for gradient in gradients {
        let grad_data = gradient.as_slice().ok_or_else(|| {
            TensorError::invalid_operation_simple("Cannot access gradient data".to_string())
        })?;

        for &val in grad_data {
            if val.is_nan() || val.is_infinite() {
                return Ok(true);
            }
        }
    }

    Ok(false)
}
/// Add noise to gradients for privacy or regularization
///
/// Adds Gaussian noise to gradients, useful for differential privacy or regularization.
///
/// # Arguments
///
/// * `gradients` - Slice of gradient tensors
/// * `noise_stddev` - Standard deviation of noise
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::add_gradient_noise;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::zeros(&[3])];
/// let noisy_grads = add_gradient_noise(&grads, 0.01, Some(42))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn add_gradient_noise(
    gradients: &[Tensor<f32>],
    noise_stddev: f32,
    seed: Option<u64>,
) -> Result<Vec<Tensor<f32>>> {
    use scirs2_core::random::Random;

    use scirs2_core::ndarray::{Array, IxDyn};

    let seed_value = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let mut rng = Random::seed(seed_value);

    let mut noisy_gradients = Vec::with_capacity(gradients.len());

    for gradient in gradients {
        let shape = gradient.shape().dims();
        let size: usize = shape.iter().product();

        // Generate noise using Normal distribution
        use scirs2_core::random::Normal;
        let normal_dist = Normal::new(0.0f32, noise_stddev).map_err(|e| {
            TensorError::invalid_operation_simple(format!("Invalid noise stddev: {}", e))
        })?;

        let noise_data: Vec<f32> = (0..size).map(|_| rng.sample(normal_dist)).collect();

        let noise_array = Array::from_shape_vec(IxDyn(shape), noise_data).map_err(|e| {
            TensorError::invalid_shape_simple(format!("Failed to create noise array: {}", e))
        })?;

        let noise_tensor = Tensor::from_array(noise_array);

        let noisy = gradient.add(&noise_tensor)?;
        noisy_gradients.push(noisy);
    }

    Ok(noisy_gradients)
}

/// Gradient operation pipeline for chaining transformations
///
/// Provides a fluent API for applying multiple gradient transformations in sequence.
///
/// # Example
///
/// ```rust,no_run
/// use tenflowers_autograd::gradient_ops::GradientPipeline;
/// use tenflowers_core::Tensor;
///
/// let grads = vec![Tensor::from_scalar(100.0f32)];
///
/// let pipeline = GradientPipeline::new()
///     .clip_by_norm(1.0)
///     .add_noise(0.01, Some(42))
///     .scale(0.1);
///
/// let processed_grads = pipeline.apply(&grads)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct GradientPipeline {
    operations: Vec<GradientOperation>,
}

#[derive(Debug, Clone)]
enum GradientOperation {
    ClipByNorm(f32),
    ClipByValue(f32),
    Scale(f32),
    AddNoise { stddev: f32, seed: Option<u64> },
}

impl GradientPipeline {
    /// Create a new empty gradient pipeline
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add gradient clipping by global norm
    pub fn clip_by_norm(mut self, max_norm: f32) -> Self {
        self.operations
            .push(GradientOperation::ClipByNorm(max_norm));
        self
    }

    /// Add gradient clipping by value
    pub fn clip_by_value(mut self, clip_value: f32) -> Self {
        self.operations
            .push(GradientOperation::ClipByValue(clip_value));
        self
    }

    /// Add gradient scaling
    pub fn scale(mut self, scale: f32) -> Self {
        self.operations.push(GradientOperation::Scale(scale));
        self
    }

    /// Add noise to gradients
    pub fn add_noise(mut self, stddev: f32, seed: Option<u64>) -> Self {
        self.operations
            .push(GradientOperation::AddNoise { stddev, seed });
        self
    }

    /// Apply all operations in the pipeline to gradients
    pub fn apply(&self, gradients: &[Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        let mut result = gradients.to_vec();

        for operation in &self.operations {
            result = match operation {
                GradientOperation::ClipByNorm(max_norm) => {
                    let (clipped, _) = clip_by_global_norm(&result, *max_norm)?;
                    clipped
                }
                GradientOperation::ClipByValue(clip_value) => clip_by_value(&result, *clip_value)?,
                GradientOperation::Scale(scale) => scale_gradients(&result, *scale)?,
                GradientOperation::AddNoise { stddev, seed } => {
                    add_gradient_noise(&result, *stddev, *seed)?
                }
            };
        }

        Ok(result)
    }

    /// Get the number of operations in the pipeline
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Clear all operations from the pipeline
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

impl Default for GradientPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient accumulator with named parameters
///
/// Useful for accumulating gradients across multiple batches with
/// proper parameter tracking by name.
pub struct NamedGradientAccumulator {
    accumulated: HashMap<String, Tensor<f32>>,
    count: usize,
}

impl NamedGradientAccumulator {
    /// Create a new accumulator
    pub fn new() -> Self {
        Self {
            accumulated: HashMap::new(),
            count: 0,
        }
    }

    /// Accumulate gradients with parameter names
    pub fn accumulate(&mut self, names: &[String], gradients: &[Tensor<f32>]) -> Result<()> {
        if names.len() != gradients.len() {
            return Err(TensorError::invalid_operation_simple(format!(
                "Names and gradients length mismatch: {} vs {}",
                names.len(),
                gradients.len()
            )));
        }

        for (name, gradient) in names.iter().zip(gradients.iter()) {
            let entry = self
                .accumulated
                .entry(name.clone())
                .or_insert_with(|| Tensor::zeros(gradient.shape().dims()));

            *entry = entry.add(gradient)?;
        }

        self.count += 1;
        Ok(())
    }

    /// Get averaged gradients
    pub fn average(&self) -> Result<HashMap<String, Tensor<f32>>> {
        if self.count == 0 {
            return Err(TensorError::invalid_operation_simple(
                "No gradients accumulated".to_string(),
            ));
        }

        let divisor = Tensor::from_scalar(self.count as f32);
        let mut averaged = HashMap::new();

        for (name, grad) in &self.accumulated {
            let avg_grad = grad.div(&divisor)?;
            averaged.insert(name.clone(), avg_grad);
        }

        Ok(averaged)
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.accumulated.clear();
        self.count = 0;
    }

    /// Get current accumulation count
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for NamedGradientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_by_global_norm() -> Result<()> {
        let grads = vec![Tensor::from_data(vec![3.0f32, 4.0], &[2])?];

        // Global norm = sqrt(3^2 + 4^2) = 5.0
        let (clipped, original_norm) = clip_by_global_norm(&grads, 1.0)?;

        assert!((original_norm - 5.0).abs() < 1e-5);
        assert_eq!(clipped.len(), 1);

        let clipped_data = clipped[0].as_slice().unwrap();
        // Should be scaled by 1.0/5.0 = 0.2
        assert!((clipped_data[0] - 0.6).abs() < 1e-5);
        assert!((clipped_data[1] - 0.8).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_clip_by_value() -> Result<()> {
        let grads = vec![Tensor::from_data(vec![-5.0f32, 10.0, 0.5], &[3])?];

        let clipped = clip_by_value(&grads, 1.0)?;
        let clipped_data = clipped[0].as_slice().unwrap();

        assert_eq!(clipped_data[0], -1.0);
        assert_eq!(clipped_data[1], 1.0);
        assert_eq!(clipped_data[2], 0.5);
        Ok(())
    }

    #[test]
    fn test_gradient_statistics() -> Result<()> {
        let grads = vec![Tensor::from_vec(
            vec![1.0f32, 0.0, -2.0, f32::NAN, f32::INFINITY],
            &[5],
        )?];

        let stats = compute_gradient_statistics(&grads)?;

        assert_eq!(stats.num_nan, 1);
        assert_eq!(stats.num_inf, 1);
        assert_eq!(stats.num_zero, 1);
        assert_eq!(stats.total_elements, 5);
        assert!(!stats.is_healthy());
        Ok(())
    }

    #[test]
    fn test_scale_gradients() -> Result<()> {
        let grads = vec![Tensor::from_data(vec![2.0f32, 4.0], &[2])?];

        let scaled = scale_gradients(&grads, 0.5)?;
        let scaled_data = scaled[0].as_slice().unwrap();

        assert_eq!(scaled_data[0], 1.0);
        assert_eq!(scaled_data[1], 2.0);
        Ok(())
    }

    #[test]
    fn test_named_accumulator() -> Result<()> {
        let mut accumulator = NamedGradientAccumulator::new();

        let names = vec!["w1".to_string(), "w2".to_string()];
        let grads1 = vec![
            Tensor::from_data(vec![1.0f32], &[1])?,
            Tensor::from_data(vec![2.0f32], &[1])?,
        ];
        let grads2 = vec![
            Tensor::from_data(vec![3.0f32], &[1])?,
            Tensor::from_data(vec![4.0f32], &[1])?,
        ];

        accumulator.accumulate(&names, &grads1)?;
        accumulator.accumulate(&names, &grads2)?;

        let averaged = accumulator.average()?;
        assert_eq!(averaged["w1"].as_slice().unwrap()[0], 2.0); // (1+3)/2
        assert_eq!(averaged["w2"].as_slice().unwrap()[0], 3.0); // (2+4)/2
        Ok(())
    }

    #[test]
    fn test_add_gradient_noise() -> Result<()> {
        let grads = vec![Tensor::zeros(&[10])];

        // Add noise with seed for reproducibility
        let noisy1 = add_gradient_noise(&grads, 0.1, Some(42))?;
        let noisy2 = add_gradient_noise(&grads, 0.1, Some(42))?;

        // Same seed should produce same noise
        let data1 = noisy1[0].as_slice().unwrap();
        let data2 = noisy2[0].as_slice().unwrap();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Noise should not be all zeros
        assert!(data1.iter().any(|&x| x.abs() > 1e-5));

        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_empty() -> Result<()> {
        let pipeline = GradientPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);

        let grads = vec![Tensor::from_scalar(5.0f32)];
        let result = pipeline.apply(&grads)?;

        // Empty pipeline should return unchanged gradients
        assert_eq!(result[0].as_slice().unwrap()[0], 5.0);
        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_single_op() -> Result<()> {
        let grads = vec![Tensor::from_data(vec![10.0f32, 20.0], &[2])?];

        // Test clip by norm
        let pipeline = GradientPipeline::new().clip_by_norm(1.0);
        let result = pipeline.apply(&grads)?;
        let result_data = result[0].as_slice().unwrap();

        // Original norm = sqrt(10^2 + 20^2) = sqrt(500) ≈ 22.36
        // Clipped norm should be 1.0
        let clipped_norm = (result_data[0].powi(2) + result_data[1].powi(2)).sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_chaining() -> Result<()> {
        let grads = vec![Tensor::from_data(vec![100.0f32, 200.0], &[2])?];

        let pipeline = GradientPipeline::new()
            .clip_by_norm(10.0) // Reduce to norm 10
            .scale(0.5) // Scale by 0.5
            .clip_by_value(2.0); // Clip to [-2, 2]

        let result = pipeline.apply(&grads)?;
        let result_data = result[0].as_slice().unwrap();

        // All values should be within [-2, 2]
        for &val in result_data {
            assert!((-2.0..=2.0).contains(&val));
        }

        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_with_noise() -> Result<()> {
        let grads = vec![Tensor::zeros(&[5])];

        let pipeline = GradientPipeline::new()
            .add_noise(0.1, Some(42))
            .clip_by_value(0.15);

        let result = pipeline.apply(&grads)?;
        let result_data = result[0].as_slice().unwrap();

        // All values should be clipped to [-0.15, 0.15]
        for &val in result_data {
            assert!((-0.15..=0.15).contains(&val));
        }

        // Should have some non-zero values from noise
        assert!(result_data.iter().any(|&x| x.abs() > 1e-5));

        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_clear() -> Result<()> {
        let mut pipeline = GradientPipeline::new().clip_by_norm(1.0).scale(0.5);

        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());

        pipeline.clear();

        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.is_empty());

        Ok(())
    }

    #[test]
    fn test_gradient_pipeline_default() {
        let pipeline = GradientPipeline::default();
        assert!(pipeline.is_empty());
    }
}
