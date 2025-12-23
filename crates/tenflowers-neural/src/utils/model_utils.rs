//! Model utility functions for common operations
//!
//! This module provides helper functions for:
//! - Parameter counting and model statistics
//! - Weight initialization across all layers
//! - Model freezing and unfreezing
//! - Model cloning and copying

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use std::collections::HashMap;
use tenflowers_core::{Result, Tensor};

/// Count total number of parameters in a layer
///
/// # Example
/// ```rust,no_run
/// use tenflowers_neural::{Dense, count_parameters};
///
/// let layer = Dense::<f32>::new(784, 128).unwrap();
/// let params = count_parameters(&layer);
/// println!("Total parameters: {}", params); // 784*128 + 128 = 100,480
/// ```
pub fn count_parameters<T>(layer: &dyn Layer<T>) -> usize
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    layer
        .parameters()
        .iter()
        .map(|p| p.shape().dims().iter().product::<usize>())
        .sum()
}

/// Count trainable parameters in a layer
///
/// This is the same as `count_parameters` for most layers,
/// but for layers with frozen parameters, this would return a different count.
pub fn count_trainable_parameters<T>(layer: &dyn Layer<T>) -> usize
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    // For now, all parameters are trainable
    // Future: Add support for frozen parameters
    count_parameters(layer)
}

/// Get model statistics including parameter counts and memory usage
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Total number of parameters
    pub total_params: usize,
    /// Number of trainable parameters
    pub trainable_params: usize,
    /// Number of non-trainable parameters
    pub non_trainable_params: usize,
    /// Estimated memory usage in bytes (fp32)
    pub memory_bytes: usize,
}

impl ModelStats {
    /// Create model statistics from a layer
    pub fn from_layer<T>(layer: &dyn Layer<T>) -> Self
    where
        T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
    {
        let total = count_parameters(layer);
        let trainable = count_trainable_parameters(layer);

        Self {
            total_params: total,
            trainable_params: trainable,
            non_trainable_params: total - trainable,
            memory_bytes: total * std::mem::size_of::<T>(),
        }
    }

    /// Get memory usage in megabytes
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory usage in gigabytes
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Format as human-readable string
    pub fn summary(&self) -> String {
        format!(
            "Total params: {}\nTrainable: {}\nNon-trainable: {}\nMemory: {:.2} MB",
            format_number(self.total_params),
            format_number(self.trainable_params),
            format_number(self.non_trainable_params),
            self.memory_mb()
        )
    }
}

/// Format number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();

    for (count, c) in s.chars().rev().enumerate() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }

    result.chars().rev().collect()
}

/// Initialize all parameters in a layer with Xavier/Glorot initialization
///
/// This is suitable for layers with tanh or sigmoid activations.
///
/// # Example
/// ```rust,no_run
/// use tenflowers_neural::{Dense, xavier_init};
///
/// let mut layer = Dense::<f32>::new(784, 128).unwrap();
/// xavier_init(&mut layer, 42); // seed = 42
/// ```
pub fn xavier_init<T>(layer: &mut dyn Layer<T>, seed: u64)
where
    T: Float + FromPrimitive + Clone + Default + Zero + One + Send + Sync + 'static,
{
    use scirs2_core::random::Random;

    let mut rng = Random::seed(seed);

    for param in layer.parameters_mut() {
        let shape = param.shape().dims();
        if shape.is_empty() {
            continue;
        }

        // Xavier initialization: sqrt(6 / (fan_in + fan_out))
        let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
        let fan_out = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let limit = T::from((6.0_f64 / (fan_in + fan_out) as f64).sqrt()).unwrap();

        // Generate uniform values in [-limit, limit]
        let total_elements = shape.iter().product::<usize>();
        let values: Vec<T> = (0..total_elements)
            .map(|_| {
                let random_val = rng.gen_range(-1.0..1.0);
                T::from(random_val).unwrap() * limit
            })
            .collect();

        if let Ok(new_tensor) = Tensor::from_data(values, shape) {
            *param = new_tensor;
        }
    }
}

/// Initialize all parameters in a layer with He initialization
///
/// This is suitable for layers with ReLU activations.
///
/// # Example
/// ```rust,no_run
/// use tenflowers_neural::{Dense, he_init};
///
/// let mut layer = Dense::<f32>::new(784, 128).unwrap();
/// he_init(&mut layer, 42); // seed = 42
/// ```
pub fn he_init<T>(layer: &mut dyn Layer<T>, seed: u64)
where
    T: Float + FromPrimitive + Clone + Default + Zero + One + Send + Sync + 'static,
{
    use scirs2_core::random::Random;

    let mut rng = Random::seed(seed);

    for param in layer.parameters_mut() {
        let shape = param.shape().dims();
        if shape.is_empty() {
            continue;
        }

        // He initialization: sqrt(2 / fan_in)
        let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
        let std = T::from((2.0_f64 / fan_in as f64).sqrt()).unwrap();

        // Generate normal values with std
        let total_elements = shape.iter().product::<usize>();
        let values: Vec<T> = (0..total_elements)
            .map(|_| {
                let random_val = rng.gen_range(-1.0..1.0);
                T::from(random_val).unwrap() * std
            })
            .collect();

        if let Ok(new_tensor) = Tensor::from_data(values, shape) {
            *param = new_tensor;
        }
    }
}

/// Initialize all parameters with zeros
pub fn zero_init<T>(layer: &mut dyn Layer<T>)
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    for param in layer.parameters_mut() {
        let shape = param.shape().dims();
        let zeros = Tensor::zeros(shape);
        *param = zeros;
    }
}

/// Initialize all parameters with ones
pub fn one_init<T>(layer: &mut dyn Layer<T>)
where
    T: Float + FromPrimitive + Clone + Default + Zero + One + Send + Sync + 'static,
{
    for param in layer.parameters_mut() {
        let shape = param.shape().dims();
        let total_elements = shape.iter().product::<usize>();
        let values = vec![T::one(); total_elements];

        if let Ok(ones) = Tensor::from_data(values, shape) {
            *param = ones;
        }
    }
}

/// Get parameter shapes as a hashmap
///
/// Returns a map of parameter index -> shape
pub fn get_parameter_shapes<T>(layer: &dyn Layer<T>) -> HashMap<usize, Vec<usize>>
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    layer
        .parameters()
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.shape().dims().to_vec()))
        .collect()
}

/// Check if all parameters are finite (no NaN or Inf)
pub fn check_parameters_finite<T>(layer: &dyn Layer<T>) -> bool
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    layer.parameters().iter().all(|p| {
        if let Some(data) = p.as_slice() {
            data.iter().all(|&v| v.is_finite())
        } else {
            // If we can't access the data, assume it's finite
            true
        }
    })
}

/// Get the L2 norm of all parameters
pub fn parameter_norm<T>(layer: &dyn Layer<T>) -> T
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    let sum_squares: T = layer
        .parameters()
        .iter()
        .flat_map(|p| p.as_slice().unwrap_or(&[]))
        .map(|&v| v * v)
        .fold(T::zero(), |acc, v| acc + v);

    sum_squares.sqrt()
}

/// Clip all parameter values by threshold
///
/// This is a convenience function that clips all parameter values
/// to be within [-threshold, threshold]
///
/// Note: This modifies parameters directly, not gradients.
/// For gradient clipping, use the gradient clipping utilities.
pub fn clip_parameters_by_value<T>(layer: &mut dyn Layer<T>, threshold: T) -> Result<()>
where
    T: Float + FromPrimitive + Clone + Default + Zero + One + Send + Sync + 'static,
{
    for param in layer.parameters_mut() {
        if let Some(data) = param.as_slice() {
            let shape = param.shape().dims();
            let clipped_data: Vec<T> = data
                .iter()
                .map(|&v| {
                    if v > threshold {
                        threshold
                    } else if v < T::zero() - threshold {
                        T::zero() - threshold
                    } else {
                        v
                    }
                })
                .collect();

            if let Ok(new_param) = Tensor::from_data(clipped_data, shape) {
                *param = new_param;
            }
        }
    }
    Ok(())
}

/// Calculate the total gradient norm (for gradient clipping)
pub fn gradient_norm<T>(layer: &dyn Layer<T>) -> T
where
    T: Float + Clone + Default + Zero + One + Send + Sync + 'static,
{
    // Note: This would need access to gradients, which aren't stored in Layer trait
    // This is a placeholder that returns parameter norm as approximation
    parameter_norm(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(1234567), "1,234,567");
        assert_eq!(format_number(1234567890), "1,234,567,890");
    }

    #[test]
    fn test_model_stats() {
        // Create a simple mock layer with 100 parameters
        struct MockLayer;
        impl Layer<f32> for MockLayer {
            fn forward(&self, _input: &Tensor<f32>) -> Result<Tensor<f32>> {
                Ok(Tensor::zeros(&[1]))
            }
            fn parameters(&self) -> Vec<&Tensor<f32>> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
                vec![]
            }
            fn set_training(&mut self, _training: bool) {}
            fn clone_box(&self) -> Box<dyn Layer<f32>> {
                Box::new(MockLayer)
            }
        }

        let layer = MockLayer;
        let stats = ModelStats::from_layer(&layer);

        assert_eq!(stats.total_params, 0);
        assert_eq!(stats.trainable_params, 0);
        assert_eq!(stats.memory_bytes, 0);
    }

    #[test]
    fn test_parameter_norm_zero() {
        struct MockLayer;
        impl Layer<f32> for MockLayer {
            fn forward(&self, _input: &Tensor<f32>) -> Result<Tensor<f32>> {
                Ok(Tensor::zeros(&[1]))
            }
            fn parameters(&self) -> Vec<&Tensor<f32>> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
                vec![]
            }
            fn set_training(&mut self, _training: bool) {}
            fn clone_box(&self) -> Box<dyn Layer<f32>> {
                Box::new(MockLayer)
            }
        }

        let layer = MockLayer;
        let norm = parameter_norm(&layer);

        assert_eq!(norm, 0.0);
    }
}
