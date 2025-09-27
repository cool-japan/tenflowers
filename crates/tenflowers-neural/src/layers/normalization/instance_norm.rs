//! Instance Normalization implementation
//!
//! Instance Normalization normalizes each sample and channel independently.
//! It computes statistics over the spatial dimensions (height and width) for each
//! channel in each sample, making it particularly useful for style transfer tasks.

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Instance Normalization - normalizes each sample and channel independently
#[derive(Clone)]
pub struct InstanceNorm<T> {
    num_channels: usize,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    epsilon: f32,
    affine: bool,
}

impl<T> InstanceNorm<T>
where
    T: Clone + Default + Zero + One,
{
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            gamma: Tensor::ones(&[num_channels]),
            beta: Tensor::zeros(&[num_channels]),
            epsilon: 1e-5,
            affine: true,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_affine(mut self, affine: bool) -> Self {
        self.affine = affine;
        self
    }
}

impl<T> Layer<T> for InstanceNorm<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + Float
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let shape = input.shape();
        let ndim = shape.rank();

        // Assume NCHW format for instance norm
        if ndim != 4 {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "InstanceNorm expects 4D input (NCHW), got {ndim}D"
            )));
        }

        let _batch_size = shape.dims()[0];
        let channels = shape.dims()[1];
        let _height = shape.dims()[2];
        let _width = shape.dims()[3];

        if channels != self.num_channels {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Expected {} channels, got {}",
                self.num_channels, channels
            )));
        }

        // Compute mean and variance over spatial dimensions (H, W) for each (N, C) pair
        let axes = vec![2i32, 3i32]; // Normalize over height and width

        let mean = input.mean(Some(&axes), true)?;

        // Compute variance
        let centered = input.sub(&mean)?;
        let squared = centered.mul(&centered)?;
        let variance = squared.mean(Some(&axes), true)?;

        // Normalize
        let eps = T::from(self.epsilon).unwrap();
        let eps_tensor = Tensor::from_scalar(eps);
        let std = variance.add(&eps_tensor)?.sqrt()?;
        let normalized = centered.div(&std)?;

        if self.affine {
            // Apply gamma and beta scaling
            let gamma_reshaped = self.gamma.reshape(&[1, channels, 1, 1])?;
            let beta_reshaped = self.beta.reshape(&[1, channels, 1, 1])?;

            let output = normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)?;
            Ok(output)
        } else {
            // No affine transformation
            Ok(normalized)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        if self.affine {
            vec![&self.gamma, &self.beta]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        if self.affine {
            vec![&mut self.gamma, &mut self.beta]
        } else {
            vec![]
        }
    }

    fn set_training(&mut self, _training: bool) {
        // InstanceNorm behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm_creation() {
        let instance_norm = InstanceNorm::<f32>::new(64);
        assert_eq!(instance_norm.num_channels, 64);
        assert!(instance_norm.affine);
        assert_eq!(instance_norm.epsilon, 1e-5);
    }

    #[test]
    fn test_instance_norm_with_affine() {
        let instance_norm = InstanceNorm::<f32>::new(32).with_affine(false);
        assert!(!instance_norm.affine);
    }

    #[test]
    fn test_instance_norm_with_epsilon() {
        let instance_norm = InstanceNorm::<f32>::new(16).with_epsilon(1e-6);
        assert_eq!(instance_norm.epsilon, 1e-6);
    }

    #[test]
    fn test_instance_norm_parameters_with_affine() {
        let instance_norm = InstanceNorm::<f32>::new(128);
        let params = instance_norm.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }

    #[test]
    fn test_instance_norm_parameters_without_affine() {
        let instance_norm = InstanceNorm::<f32>::new(128).with_affine(false);
        let params = instance_norm.parameters();
        assert_eq!(params.len(), 0); // No parameters when affine=false
    }

    #[test]
    fn test_instance_norm_training_mode() {
        let mut instance_norm = InstanceNorm::<f32>::new(64);

        // InstanceNorm doesn't change behavior based on training mode
        instance_norm.set_training(true);
        instance_norm.set_training(false);

        // Should succeed without error
        assert_eq!(instance_norm.num_channels, 64);
    }
}
