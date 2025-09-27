//! Group Normalization implementation
//!
//! Group Normalization divides channels into groups and normalizes within each group.
//! It's a middle ground between LayerNorm (G=1) and InstanceNorm (G=C), providing
//! stability that's less dependent on batch size than BatchNorm.

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Group Normalization - divides channels into groups and normalizes within each group
#[derive(Clone)]
pub struct GroupNorm<T> {
    num_groups: usize,
    num_channels: usize,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    epsilon: f32,
}

impl<T> GroupNorm<T>
where
    T: Clone + Default + Zero + One,
{
    pub fn new(num_groups: usize, num_channels: usize) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(tenflowers_core::TensorError::invalid_argument(format!(
                "num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )));
        }

        Ok(Self {
            num_groups,
            num_channels,
            gamma: Tensor::ones(&[num_channels]),
            beta: Tensor::zeros(&[num_channels]),
            epsilon: 1e-5,
        })
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl<T> Layer<T> for GroupNorm<T>
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

        // Assume NCHW format for group norm
        if ndim != 4 {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "GroupNorm expects 4D input (NCHW), got {ndim}D"
            )));
        }

        let batch_size = shape.dims()[0];
        let channels = shape.dims()[1];
        let height = shape.dims()[2];
        let width = shape.dims()[3];

        if channels != self.num_channels {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Expected {} channels, got {} and {}",
                self.num_channels, self.num_channels, channels
            )));
        }

        let channels_per_group = channels / self.num_groups;

        // Reshape to (N, G, C//G, H, W) for group-wise normalization
        let reshaped_input = input.reshape(&[
            batch_size,
            self.num_groups,
            channels_per_group,
            height,
            width,
        ])?;

        // Compute mean and variance along (C//G, H, W) dimensions (axes 2, 3, 4)
        let axes = vec![2i32, 3i32, 4i32];
        let mean = reshaped_input.mean(Some(&axes), true)?;

        // Compute variance
        let centered = reshaped_input.sub(&mean)?;
        let squared = centered.mul(&centered)?;
        let variance = squared.mean(Some(&axes), true)?;

        // Normalize
        let eps = T::from(self.epsilon).unwrap();
        let eps_tensor = Tensor::from_scalar(eps);
        let std = variance.add(&eps_tensor)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Reshape back to (N, C, H, W)
        let normalized = normalized.reshape(&[batch_size, channels, height, width])?;

        // Apply gamma and beta
        let gamma_reshaped = self.gamma.reshape(&[1, channels, 1, 1])?;
        let beta_reshaped = self.beta.reshape(&[1, channels, 1, 1])?;

        let output = normalized.mul(&gamma_reshaped)?.add(&beta_reshaped)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn set_training(&mut self, _training: bool) {
        // GroupNorm behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_norm_creation() {
        let group_norm = GroupNorm::<f32>::new(8, 64).unwrap();
        assert_eq!(group_norm.num_groups, 8);
        assert_eq!(group_norm.num_channels, 64);
        assert_eq!(group_norm.epsilon, 1e-5);
    }

    #[test]
    fn test_group_norm_invalid_groups() {
        // Channels not divisible by groups should fail
        let result = GroupNorm::<f32>::new(7, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_group_norm_with_epsilon() {
        let group_norm = GroupNorm::<f32>::new(4, 32).unwrap().with_epsilon(1e-6);
        assert_eq!(group_norm.epsilon, 1e-6);
    }

    #[test]
    fn test_group_norm_parameters() {
        let group_norm = GroupNorm::<f32>::new(16, 128).unwrap();
        let params = group_norm.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }

    #[test]
    fn test_group_norm_special_cases() {
        // LayerNorm case: G=1
        let layer_norm_like = GroupNorm::<f32>::new(1, 64).unwrap();
        assert_eq!(layer_norm_like.num_groups, 1);

        // InstanceNorm case: G=C (one group per channel)
        let instance_norm_like = GroupNorm::<f32>::new(32, 32).unwrap();
        assert_eq!(instance_norm_like.num_groups, 32);
        assert_eq!(instance_norm_like.num_channels, 32);
    }

    #[test]
    fn test_group_norm_training_mode() {
        let mut group_norm = GroupNorm::<f32>::new(8, 64).unwrap();

        // GroupNorm doesn't change behavior based on training mode
        group_norm.set_training(true);
        group_norm.set_training(false);

        // Should succeed without error
        assert_eq!(group_norm.num_groups, 8);
    }
}
