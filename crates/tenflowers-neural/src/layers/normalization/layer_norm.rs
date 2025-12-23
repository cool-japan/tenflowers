//! Layer Normalization implementation
//!
//! Layer Normalization normalizes inputs across the feature dimension (last N dimensions).
//! Unlike Batch Normalization, it normalizes across the features rather than the batch,
//! making it suitable for variable-length sequences and recurrent networks.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Layer Normalization - normalizes inputs across the feature dimension
#[derive(Debug, Clone)]
pub struct LayerNorm<T>
where
    T: FromPrimitive + bytemuck::Pod + bytemuck::Zeroable,
{
    #[allow(dead_code)]
    normalized_shape: Vec<usize>,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    epsilon: f32,
}

impl<T> LayerNorm<T>
where
    T: Clone
        + Default
        + Zero
        + One
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(normalized_shape: &[usize]) -> Self {
        Self {
            normalized_shape: normalized_shape.to_vec(),
            gamma: Tensor::ones(normalized_shape),
            beta: Tensor::zeros(normalized_shape),
            epsilon: 1e-5,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl<T> Layer<T> for LayerNorm<T>
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

        // LayerNorm normalizes over the last N dimensions
        let norm_dims = self.normalized_shape.len();
        if ndim < norm_dims {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Input rank {ndim} is less than normalized dimensions {norm_dims}"
            )));
        }

        // Check that the last dimensions match normalized_shape
        let input_dims = &shape.dims()[ndim - norm_dims..];
        if input_dims != self.normalized_shape.as_slice() {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Expected last {} dimensions to be {:?}, got {:?}",
                norm_dims, self.normalized_shape, input_dims
            )));
        }

        // Compute mean and variance over the normalized dimensions
        let axes: Vec<i32> = (ndim - norm_dims..ndim).map(|i| i as i32).collect();

        // Compute mean
        let mean = input.mean(Some(&axes), true)?;

        // Compute variance: E[(X - mean)^2]
        let centered = input.sub(&mean)?;
        let squared = centered.mul(&centered)?;
        let variance = squared.mean(Some(&axes), true)?;

        // Normalize: (x - mean) / sqrt(var + epsilon)
        let eps = T::from(self.epsilon).unwrap();
        let eps_tensor = Tensor::from_scalar(eps);
        let std = variance.add(&eps_tensor)?.sqrt()?;
        let normalized = centered.div(&std)?;

        // Scale and shift
        let output = normalized.mul(&self.gamma)?.add(&self.beta)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn set_training(&mut self, _training: bool) {
        // LayerNorm behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::<f32>::new(&[128]);
        assert_eq!(layer_norm.normalized_shape, vec![128]);
        assert_eq!(layer_norm.epsilon, 1e-5);
    }

    #[test]
    fn test_layer_norm_with_epsilon() {
        let layer_norm = LayerNorm::<f32>::new(&[64]).with_epsilon(1e-6);
        assert_eq!(layer_norm.epsilon, 1e-6);
    }

    #[test]
    fn test_layer_norm_parameters() {
        let layer_norm = LayerNorm::<f32>::new(&[256]);
        let params = layer_norm.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }

    #[test]
    fn test_layer_norm_multidimensional() {
        let layer_norm = LayerNorm::<f32>::new(&[64, 128]);
        assert_eq!(layer_norm.normalized_shape, vec![64, 128]);
    }

    #[test]
    fn test_layer_norm_training_mode() {
        let mut layer_norm = LayerNorm::<f32>::new(&[32]);

        // LayerNorm doesn't change behavior based on training mode
        layer_norm.set_training(true);
        layer_norm.set_training(false);

        // Should succeed without error
        assert_eq!(layer_norm.normalized_shape, vec![32]);
    }
}
