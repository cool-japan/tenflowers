//! Dropout layers for regularization
//!
//! This module provides various dropout implementations for regularization during training:
//! - Standard Dropout: Basic element-wise dropout
//! - SpatialDropout2D: Drops entire feature maps for CNNs
//! - VariationalDropout: Learnable dropout with uncertainty estimation
//! - ConcreteDropout: Learns optimal dropout rates

pub mod standard;

// Re-export the standard dropout as the main one
pub use standard::Dropout;

// For now, include the other dropout variants inline until full refactoring
use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use scirs2_core::random::rng;
// use std::collections::HashMap; // Unused for now
use tenflowers_core::{Result, Tensor};

/// Spatial Dropout layer for CNN regularization
///
/// Drops entire feature maps instead of individual elements, which is more appropriate
/// for convolutional layers since spatially adjacent pixels are highly correlated.
/// Works with 4D tensors of shape [batch, channels, height, width]
#[derive(Debug, Clone)]
pub struct SpatialDropout2D<T> {
    dropout_rate: T,
    training: bool,
}

impl<T> SpatialDropout2D<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new spatial dropout layer
    ///
    /// # Arguments
    /// * `dropout_rate` - Fraction of feature maps to drop (between 0.0 and 1.0)
    pub fn new(dropout_rate: T) -> Self {
        assert!(
            dropout_rate >= T::zero() && dropout_rate <= T::one(),
            "Dropout rate must be between 0.0 and 1.0"
        );

        Self {
            dropout_rate,
            training: true,
        }
    }

    /// Get the current dropout rate
    pub fn dropout_rate(&self) -> T {
        self.dropout_rate
    }

    /// Set the dropout rate
    pub fn set_dropout_rate(&mut self, rate: T) {
        assert!(
            rate >= T::zero() && rate <= T::one(),
            "Dropout rate must be between 0.0 and 1.0"
        );
        self.dropout_rate = rate;
    }
}

impl<T> Layer<T> for SpatialDropout2D<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::cmp::PartialOrd
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.training || self.dropout_rate == T::zero() {
            return Ok(input.clone());
        }

        let shape = input.shape().dims();
        if shape.len() != 4 {
            return Err(tenflowers_core::error::TensorError::invalid_argument(
                "SpatialDropout2D expects 4D input [batch, channels, height, width]".to_string(),
            ));
        }

        let [batch_size, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let keep_prob = T::one() - self.dropout_rate;

        // Create channel-wise dropout mask
        let mut rng = rng();
        let mut channel_mask = Vec::with_capacity(batch_size * channels);

        for _ in 0..batch_size {
            for _ in 0..channels {
                let random_val: f64 = rng.gen_range(0.0..1.0);
                let random_t = T::from(random_val).unwrap_or(T::zero());
                if random_t < self.dropout_rate {
                    channel_mask.push(T::zero()); // Drop this channel
                } else {
                    channel_mask.push(T::one()); // Keep this channel
                }
            }
        }

        // Expand mask to full tensor shape
        let mut mask_data = Vec::with_capacity(batch_size * channels * height * width);
        for (batch, channel) in (0..batch_size).flat_map(|b| (0..channels).map(move |c| (b, c))) {
            let mask_val = channel_mask[batch * channels + channel];
            for _ in 0..(height * width) {
                mask_data.push(mask_val);
            }
        }

        use scirs2_autograd::ndarray::{ArrayD, IxDyn};
        let mask_array = ArrayD::from_shape_vec(IxDyn(shape), mask_data).map_err(|_| {
            tenflowers_core::error::TensorError::invalid_shape_simple(
                "Failed to create spatial dropout mask".to_string(),
            )
        })?;

        let mask = Tensor::from_array(mask_array);

        // Apply mask and scale
        let dropped = input.mul(&mask)?;
        let scale = T::one() / keep_prob;
        dropped.mul(&Tensor::from_scalar(scale))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

// TODO: In a complete refactoring, VariationalDropout and ConcreteDropout would also
// be moved to separate files (variational.rs and concrete.rs)
// For now, they remain here to avoid incomplete implementation

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_standard_dropout() {
        let dropout = Dropout::<f32>::new(0.5);
        assert_eq!(dropout.dropout_rate(), 0.5);
    }

    #[test]
    fn test_spatial_dropout_creation() {
        let spatial_dropout = SpatialDropout2D::<f32>::new(0.3);
        assert_eq!(spatial_dropout.dropout_rate(), 0.3);
    }

    #[test]
    #[should_panic]
    fn test_spatial_dropout_wrong_dimensions() {
        let mut spatial_dropout = SpatialDropout2D::<f32>::new(0.5);
        spatial_dropout.set_training(true);

        // Create 3D tensor (should fail)
        let input_data = array![[[1.0, 2.0], [3.0, 4.0]]];
        let input = Tensor::from_array(input_data.into_dyn());

        let _ = spatial_dropout.forward(&input).unwrap();
    }
}
