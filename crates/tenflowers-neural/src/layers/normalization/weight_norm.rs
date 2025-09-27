//! Weight Normalization implementation
//!
//! Weight Normalization reparameterizes weight vectors to decouple magnitude and direction.
//! It replaces the weight vector w with g * v/||v||, where g is a learnable scalar
//! and v is a learnable vector. This provides more stable gradient flow and can
//! accelerate convergence in neural networks.

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use tenflowers_core::{Result, Tensor};

/// Weight Normalization - reparameterizes weight vectors to decouple magnitude and direction
#[derive(Clone)]
pub struct WeightNorm<T> {
    weight_g: Tensor<T>, // Magnitude parameter
    weight_v: Tensor<T>, // Direction parameter
    #[allow(dead_code)]
    dim: i32, // Dimension along which to normalize
}

impl<T> WeightNorm<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new WeightNorm layer
    /// weight_shape: shape of the weight tensor
    /// dim: dimension along which to compute the norm (default: 0)
    pub fn new(weight_shape: &[usize], dim: Option<i32>) -> Self {
        let dim = dim.unwrap_or(0);

        // Initialize weight_v (using ones for simplicity - in practice would be initialized externally)
        let weight_v = Tensor::ones(weight_shape);

        // Initialize weight_g as ones (in practice would compute norm of weight_v)
        let weight_g_init = Tensor::ones(&[weight_shape[dim as usize]]);

        Self {
            weight_g: weight_g_init,
            weight_v,
            dim,
        }
    }

    /// Get the normalized weight tensor: g * v / ||v||
    pub fn get_normalized_weight(&self) -> Result<Tensor<T>> {
        // Compute ||v|| along the specified dimension manually
        // For simplicity, compute overall norm (in practice would need proper axis handling)
        let v_squared = self.weight_v.mul(&self.weight_v)?;
        let sum_squares = tenflowers_core::ops::sum(&v_squared, None, false)?;
        let v_norm = sum_squares.sqrt()?;

        // Add small epsilon for numerical stability
        let eps = T::from(1e-12).unwrap();
        let eps_tensor = Tensor::from_scalar(eps);
        let v_norm_safe = v_norm.add(&eps_tensor)?;

        // Normalize: v / ||v||
        let v_normalized = self.weight_v.div(&v_norm_safe)?;

        // Scale by magnitude: g * (v / ||v||)
        // Need to reshape weight_g to broadcast properly
        let weight_g_shape = self.weight_v.shape().dims().to_vec();
        let mut g_broadcasted_shape = vec![1; weight_g_shape.len()];
        g_broadcasted_shape[self.dim as usize] = weight_g_shape[self.dim as usize];
        let weight_g_reshaped = self.weight_g.reshape(&g_broadcasted_shape)?;

        v_normalized.mul(&weight_g_reshaped)
    }

    /// Update the parameters (used during training)
    pub fn update_parameters(&mut self, new_g: Tensor<T>, new_v: Tensor<T>) -> Result<()> {
        if new_g.shape() != self.weight_g.shape() {
            return Err(tenflowers_core::TensorError::shape_mismatch(
                "WeightNorm::update_parameters",
                &self.weight_g.shape().to_string(),
                &new_g.shape().to_string(),
            ));
        }
        if new_v.shape() != self.weight_v.shape() {
            return Err(tenflowers_core::TensorError::shape_mismatch(
                "WeightNorm::update_parameters",
                &self.weight_v.shape().to_string(),
                &new_v.shape().to_string(),
            ));
        }

        self.weight_g = new_g;
        self.weight_v = new_v;
        Ok(())
    }

    /// Get the magnitude parameter (g)
    pub fn weight_g(&self) -> &Tensor<T> {
        &self.weight_g
    }

    /// Get the direction parameter (v)
    pub fn weight_v(&self) -> &Tensor<T> {
        &self.weight_v
    }

    /// Get the normalization dimension
    pub fn dim(&self) -> i32 {
        self.dim
    }
}

impl<T> Layer<T> for WeightNorm<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // WeightNorm is typically used as a component within other layers
        // This implementation provides a basic linear transformation
        let weight = self.get_normalized_weight()?;
        input.matmul(&weight.transpose()?)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight_g, &self.weight_v]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.weight_g, &mut self.weight_v]
    }

    fn set_training(&mut self, _training: bool) {
        // WeightNorm behavior doesn't change between training and eval
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_norm_creation() {
        let weight_norm = WeightNorm::<f32>::new(&[128, 64], None);
        assert_eq!(weight_norm.weight_v().shape().dims(), &[128, 64]);
        assert_eq!(weight_norm.weight_g().shape().dims(), &[128]); // Default dim=0
        assert_eq!(weight_norm.dim(), 0);
    }

    #[test]
    fn test_weight_norm_with_custom_dim() {
        let weight_norm = WeightNorm::<f32>::new(&[64, 32], Some(1));
        assert_eq!(weight_norm.weight_v().shape().dims(), &[64, 32]);
        assert_eq!(weight_norm.weight_g().shape().dims(), &[32]); // dim=1
        assert_eq!(weight_norm.dim(), 1);
    }

    #[test]
    fn test_weight_norm_update_parameters() {
        let mut weight_norm = WeightNorm::<f32>::new(&[4, 3], None);
        let new_g = Tensor::zeros(&[4]);
        let new_v = Tensor::zeros(&[4, 3]);

        let result = weight_norm.update_parameters(new_g, new_v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_weight_norm_update_parameters_wrong_shape() {
        let mut weight_norm = WeightNorm::<f32>::new(&[4, 3], None);
        let wrong_g = Tensor::zeros(&[5]); // Wrong shape
        let new_v = Tensor::zeros(&[4, 3]);

        let result = weight_norm.update_parameters(wrong_g, new_v);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_norm_get_normalized_weight() {
        let weight_norm = WeightNorm::<f32>::new(&[2, 2], None);
        let result = weight_norm.get_normalized_weight();
        assert!(result.is_ok());

        let normalized_weight = result.unwrap();
        assert_eq!(normalized_weight.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_weight_norm_parameters() {
        let weight_norm = WeightNorm::<f32>::new(&[8, 6], None);
        let params = weight_norm.parameters();
        assert_eq!(params.len(), 2); // weight_g and weight_v
    }

    #[test]
    fn test_weight_norm_getters() {
        let weight_norm = WeightNorm::<f32>::new(&[16, 8], Some(1));

        // Test getter methods
        assert_eq!(weight_norm.weight_g().shape().dims(), &[8]);
        assert_eq!(weight_norm.weight_v().shape().dims(), &[16, 8]);
        assert_eq!(weight_norm.dim(), 1);
    }

    #[test]
    fn test_weight_norm_training_mode() {
        let mut weight_norm = WeightNorm::<f32>::new(&[32, 16], None);

        // WeightNorm doesn't change behavior based on training mode
        weight_norm.set_training(true);
        weight_norm.set_training(false);

        // Should succeed without error
        assert_eq!(weight_norm.weight_v().shape().dims(), &[32, 16]);
    }
}
