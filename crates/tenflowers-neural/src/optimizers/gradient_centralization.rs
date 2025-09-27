//! Gradient Centralization implementation
//!
//! Gradient Centralization (GC) is a simple yet effective optimization technique that
//! normalizes gradients by centering them at zero. This technique improves convergence
//! and generalization performance across various neural network architectures.
//!
//! Key benefits:
//! - Improves training stability and convergence speed
//! - Works as a plug-in technique with any existing optimizer
//! - Reduces overfitting and improves generalization
//! - Negligible computational overhead
//!
//! Reference: "Gradient Centralization: A New Optimization Technique for Deep Neural Networks" (Yong et al., 2020)
//! https://arxiv.org/abs/2004.01461

use num_traits::{Float, FromPrimitive, Zero};
use tenflowers_core::{Result, Tensor};

/// Gradient centralization configuration
#[derive(Debug, Clone)]
pub struct GradientCentralizationConfig {
    /// Whether to apply GC to convolutional layers (default: true)
    pub apply_to_conv: bool,
    /// Whether to apply GC to dense/linear layers (default: true)  
    pub apply_to_dense: bool,
    /// Minimum number of elements required to apply GC (default: 4)
    /// Small tensors (e.g., bias) are typically excluded
    pub min_elements: usize,
    /// Whether to center over all dimensions or preserve certain structures
    pub center_all_dims: bool,
}

impl Default for GradientCentralizationConfig {
    fn default() -> Self {
        Self {
            apply_to_conv: true,
            apply_to_dense: true,
            min_elements: 4,
            center_all_dims: true,
        }
    }
}

/// Apply gradient centralization to a tensor
///
/// For a gradient tensor G, gradient centralization computes:
/// GC(G) = G - mean(G)
///
/// This centers the gradient at zero, which has been shown to improve
/// optimization dynamics and convergence properties.
///
/// # Arguments
/// * `gradient` - The gradient tensor to be centralized
/// * `config` - Configuration for gradient centralization
///
/// # Returns
/// Centralized gradient tensor
pub fn apply_gradient_centralization<T>(
    gradient: &Tensor<T>,
    config: &GradientCentralizationConfig,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + Zero
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let shape = gradient.shape();
    let total_elements = shape.dims().iter().product::<usize>();

    // Skip very small tensors (typically bias terms)
    if total_elements < config.min_elements {
        return Ok(gradient.clone());
    }

    // Apply gradient centralization based on tensor dimensions
    match shape.dims().len() {
        // 1D tensors (bias terms) - typically skip
        1 => {
            if total_elements >= config.min_elements {
                centralize_1d(gradient)
            } else {
                Ok(gradient.clone())
            }
        }
        // 2D tensors (Dense/Linear layer weights)
        2 => {
            if config.apply_to_dense {
                centralize_2d(gradient, config.center_all_dims)
            } else {
                Ok(gradient.clone())
            }
        }
        // 3D+ tensors (Convolutional layer weights)
        _ => {
            if config.apply_to_conv {
                centralize_nd(gradient)
            } else {
                Ok(gradient.clone())
            }
        }
    }
}

/// Centralize 1D gradient (typically for bias terms)
fn centralize_1d<T>(gradient: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + Zero
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Compute mean along the single dimension
    let mean = gradient.mean(Some(&[0]), false)?;
    gradient.sub(&mean)
}

/// Centralize 2D gradient (for Dense/Linear layers)
///
/// For dense layers, we can either:
/// - Center over all dimensions (center_all_dims = true)
/// - Center per output neuron (center_all_dims = false)
fn centralize_2d<T>(gradient: &Tensor<T>, center_all_dims: bool) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + Zero
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if center_all_dims {
        // Center over all dimensions
        let mean = gradient.mean(Some(&[0, 1]), false)?;
        gradient.sub(&mean)
    } else {
        // Center per output neuron (along input dimension)
        let mean = gradient.mean(Some(&[0]), false)?;
        gradient.sub(&mean)
    }
}

/// Centralize N-dimensional gradient (for Convolutional layers)
///
/// For convolutional layers with shape [out_channels, in_channels, H, W],
/// we typically center over spatial and input channel dimensions,
/// preserving the output channel structure.
fn centralize_nd<T>(gradient: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Float
        + Zero
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let dims = gradient.shape().dims();

    match dims.len() {
        3 => {
            // 1D Conv: [out_channels, in_channels, kernel_size]
            // Center over in_channels and kernel dimensions
            let mean = gradient.mean(Some(&[1, 2]), false)?;
            gradient.sub(&mean)
        }
        4 => {
            // 2D Conv: [out_channels, in_channels, kernel_h, kernel_w]
            // Center over in_channels and spatial dimensions
            let mean = gradient.mean(Some(&[1, 2, 3]), false)?;
            gradient.sub(&mean)
        }
        5 => {
            // 3D Conv: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
            // Center over in_channels and spatial dimensions
            let mean = gradient.mean(Some(&[1, 2, 3, 4]), false)?;
            gradient.sub(&mean)
        }
        _ => {
            // General case: center over all dimensions except the first (output channels)
            let axes: Vec<i32> = (1..dims.len()).map(|x| x as i32).collect();
            let mean = gradient.mean(Some(&axes), false)?;
            gradient.sub(&mean)
        }
    }
}

/// Utility trait for applying gradient centralization to optimizers
///
/// This trait can be implemented by optimizers to automatically apply
/// gradient centralization before their standard update rules.
pub trait WithGradientCentralization<T> {
    /// Apply gradient centralization to gradients before optimization step
    fn apply_gc(
        &self,
        gradients: &mut Vec<Tensor<T>>,
        config: &GradientCentralizationConfig,
    ) -> Result<()>;
}

/// A wrapper that adds gradient centralization to any optimizer
///
/// # Example
/// ```rust,ignore
/// use tenflowers_neural::optimizers::{Adam, GradientCentralizationWrapper};
///
/// // Wrap any optimizer with gradient centralization
/// let base_optimizer = Adam::new(0.001);
/// let mut optimizer = GradientCentralizationWrapper::new(base_optimizer);
///
/// // The optimizer will now apply gradient centralization before each step
/// ```
pub struct GradientCentralizationWrapper<O> {
    optimizer: O,
    config: GradientCentralizationConfig,
}

impl<O> GradientCentralizationWrapper<O> {
    /// Create a new gradient centralization wrapper
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            config: GradientCentralizationConfig::default(),
        }
    }

    /// Create wrapper with custom configuration
    pub fn with_config(optimizer: O, config: GradientCentralizationConfig) -> Self {
        Self { optimizer, config }
    }

    /// Get reference to the underlying optimizer
    pub fn inner(&self) -> &O {
        &self.optimizer
    }

    /// Get mutable reference to the underlying optimizer
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get reference to the GC configuration
    pub fn config(&self) -> &GradientCentralizationConfig {
        &self.config
    }

    /// Set new GC configuration
    pub fn set_config(&mut self, config: GradientCentralizationConfig) {
        self.config = config;
    }
}

// The optimizer trait implementation would be added when integrating with the main optimizer trait

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_gradient_centralization_config_default() {
        let config = GradientCentralizationConfig::default();
        assert!(config.apply_to_conv);
        assert!(config.apply_to_dense);
        assert_eq!(config.min_elements, 4);
        assert!(config.center_all_dims);
    }

    #[test]
    fn test_apply_gc_small_tensor() {
        let gradient = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let config = GradientCentralizationConfig::default();

        // Should skip small tensors (< min_elements)
        let result = apply_gradient_centralization(&gradient, &config).unwrap();
        assert_eq!(gradient.as_slice(), result.as_slice());
    }

    #[test]
    fn test_apply_gc_1d_tensor() {
        let gradient = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let config = GradientCentralizationConfig::default();

        let result = apply_gradient_centralization(&gradient, &config).unwrap();

        // Mean should be 3.0, so result should be [-2, -1, 0, 1, 2]
        if let Some(data) = result.as_slice() {
            let expected = [-2.0, -1.0, 0.0, 1.0, 2.0];
            for (actual, expected) in data.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_apply_gc_2d_tensor() {
        let gradient =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let config = GradientCentralizationConfig::default();

        let result = apply_gradient_centralization(&gradient, &config).unwrap();

        // Should be centered (mean = 3.5)
        if let Some(data) = result.as_slice() {
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            assert!(mean.abs() < 1e-6); // Mean should be approximately zero
        }
    }

    #[test]
    fn test_apply_gc_4d_conv_tensor() {
        // Simulate a 2D conv weight: [2 out_channels, 2 in_channels, 2x2 kernel]
        let gradient = Tensor::<f32>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // First output channel
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ], // Second output channel
            &[2, 2, 2, 2],
        )
        .unwrap();

        let config = GradientCentralizationConfig::default();
        let result = apply_gradient_centralization(&gradient, &config).unwrap();

        // Each output channel should be centered independently
        // This is a basic test to ensure the function doesn't crash
        assert_eq!(result.shape().dims(), &[2, 2, 2, 2]);
    }

    #[test]
    fn test_gc_config_selective_application() {
        let gradient_2d = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let gradient_4d =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 1, 2])
                .unwrap();

        // Config that only applies to dense layers
        let config = GradientCentralizationConfig {
            apply_to_conv: false,
            apply_to_dense: true,
            min_elements: 1,
            center_all_dims: true,
        };

        // Should apply to 2D (dense) tensor
        let result_2d = apply_gradient_centralization(&gradient_2d, &config).unwrap();
        assert_ne!(gradient_2d.as_slice(), result_2d.as_slice()); // Should be modified

        // Should not apply to 4D (conv) tensor
        let result_4d = apply_gradient_centralization(&gradient_4d, &config).unwrap();
        assert_eq!(gradient_4d.as_slice(), result_4d.as_slice()); // Should be unchanged
    }

    #[test]
    fn test_gc_wrapper_creation() {
        use crate::optimizers::Adam;

        let base_optimizer = Adam::<f32>::new(0.001);
        let wrapper = GradientCentralizationWrapper::new(base_optimizer);

        assert!(wrapper.config().apply_to_conv);
        assert!(wrapper.config().apply_to_dense);
    }

    #[test]
    fn test_gc_wrapper_custom_config() {
        use crate::optimizers::Adam;

        let config = GradientCentralizationConfig {
            apply_to_conv: false,
            apply_to_dense: true,
            min_elements: 10,
            center_all_dims: false,
        };

        let base_optimizer = Adam::<f32>::new(0.001);
        let wrapper = GradientCentralizationWrapper::with_config(base_optimizer, config);

        assert!(!wrapper.config().apply_to_conv);
        assert!(wrapper.config().apply_to_dense);
        assert_eq!(wrapper.config().min_elements, 10);
        assert!(!wrapper.config().center_all_dims);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small values
        let gradient =
            Tensor::<f32>::from_vec(vec![1e-10, 2e-10, 3e-10, 4e-10, 5e-10], &[5]).unwrap();
        let config = GradientCentralizationConfig::default();

        let result = apply_gradient_centralization(&gradient, &config);
        assert!(result.is_ok());

        // Test with very large values
        let gradient_large =
            Tensor::<f32>::from_vec(vec![1e10, 2e10, 3e10, 4e10, 5e10], &[5]).unwrap();

        let result_large = apply_gradient_centralization(&gradient_large, &config);
        assert!(result_large.is_ok());
    }
}
