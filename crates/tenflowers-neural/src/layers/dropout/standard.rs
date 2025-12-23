//! Standard dropout layer implementation
//!
//! This module provides the basic dropout layer for regularization during training.
//! During training, randomly sets a fraction of inputs to zero. During inference,
//! applies identity transformation (no dropout).

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use scirs2_core::random::thread_rng;
use tenflowers_core::{Result, Tensor};

/// Standard dropout layer for regularization
///
/// During training, randomly sets a fraction of inputs to zero.
/// Uses inverted dropout: scales remaining elements by 1/(1-dropout_rate) during training
/// to maintain expected sum, and applies identity during inference.
#[derive(Debug, Clone)]
pub struct Dropout<T>
where
    T: scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::fmt::Debug,
{
    dropout_rate: T,
    training: bool,
}

impl<T> Dropout<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::fmt::Debug,
{
    /// Create a new dropout layer
    ///
    /// # Arguments
    /// * `dropout_rate` - Fraction of inputs to drop (between 0.0 and 1.0)
    ///
    /// # Panics
    /// Panics if dropout_rate is not in [0.0, 1.0]
    pub fn new(dropout_rate: T) -> Self {
        assert!(
            dropout_rate >= T::zero() && dropout_rate <= T::one(),
            "Dropout rate must be between 0.0 and 1.0, got {:?}",
            dropout_rate
        );

        Self {
            dropout_rate,
            training: true,
        }
    }

    /// Create a dropout layer with specific rate as f32
    pub fn with_rate(rate: f32) -> Self {
        let dropout_rate = T::from(rate).expect("Failed to convert dropout rate");
        Self::new(dropout_rate)
    }

    /// Get the current dropout rate
    pub fn dropout_rate(&self) -> T {
        self.dropout_rate
    }

    /// Set the dropout rate
    ///
    /// # Arguments
    /// * `rate` - New dropout rate (between 0.0 and 1.0)
    ///
    /// # Panics
    /// Panics if rate is not in [0.0, 1.0]
    pub fn set_dropout_rate(&mut self, rate: T) {
        assert!(
            rate >= T::zero() && rate <= T::one(),
            "Dropout rate must be between 0.0 and 1.0, got {:?}",
            rate
        );
        self.dropout_rate = rate;
    }

    /// Check if the layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get the keep probability (1 - dropout_rate)
    pub fn keep_prob(&self) -> T {
        T::one() - self.dropout_rate
    }

    /// Generate dropout mask for given shape
    fn generate_mask(&self, shape: &[usize]) -> Result<Tensor<T>> {
        let total_elements: usize = shape.iter().product();
        let mut rng = thread_rng();

        let mut mask_data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            let random_val: f64 = rng.gen_range(0.0..1.0);
            let random_t = T::from(random_val).unwrap_or(T::zero());
            if random_t < self.dropout_rate {
                mask_data.push(T::zero()); // Drop this element
            } else {
                mask_data.push(T::one()); // Keep this element
            }
        }

        use scirs2_core::ndarray::{ArrayD, IxDyn};
        let mask_array = ArrayD::from_shape_vec(IxDyn(shape), mask_data).map_err(|_| {
            tenflowers_core::error::TensorError::invalid_shape_simple(
                "Failed to create dropout mask".to_string(),
            )
        })?;

        Ok(Tensor::from_array(mask_array))
    }
}

impl<T> Layer<T> for Dropout<T>
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
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::fmt::Debug,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.training || self.dropout_rate == T::zero() {
            // During inference or zero dropout, no dropout is applied (identity function)
            return Ok(input.clone());
        }

        if self.dropout_rate == T::one() {
            // If dropout rate is 1.0, return zeros
            return Ok(Tensor::zeros(input.shape().dims()));
        }

        // During training, apply dropout
        let shape = input.shape().dims();
        let keep_prob = self.keep_prob();

        // Generate dropout mask
        let mask = self.generate_mask(shape)?;

        // Apply mask and scale by 1/keep_prob for inverted dropout
        let dropped = input.mul(&mask)?;
        let scale = T::one() / keep_prob;
        dropped.mul(&Tensor::from_scalar(scale))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // Dropout has no learnable parameters
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // Dropout has no learnable parameters
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

impl<T> Default for Dropout<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable
        + std::fmt::Debug,
{
    /// Create a dropout layer with 50% dropout rate
    fn default() -> Self {
        Self::new(T::from(0.5).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::<f32>::new(0.5);
        assert_eq!(dropout.dropout_rate(), 0.5);
        assert_eq!(dropout.keep_prob(), 0.5);
        assert!(dropout.is_training());
    }

    #[test]
    fn test_dropout_with_rate() {
        let dropout = Dropout::<f32>::with_rate(0.3);
        assert_eq!(dropout.dropout_rate(), 0.3);
        assert_eq!(dropout.keep_prob(), 0.7);
    }

    #[test]
    fn test_dropout_default() {
        let dropout = Dropout::<f32>::default();
        assert_eq!(dropout.dropout_rate(), 0.5);
    }

    #[test]
    #[should_panic]
    fn test_dropout_invalid_rate_high() {
        Dropout::<f32>::new(1.5);
    }

    #[test]
    #[should_panic]
    fn test_dropout_invalid_rate_negative() {
        Dropout::<f32>::new(-0.1);
    }

    #[test]
    fn test_dropout_set_rate() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_dropout_rate(0.2);
        assert_eq!(dropout.dropout_rate(), 0.2);
    }

    #[test]
    fn test_dropout_inference_mode() {
        let mut dropout = Dropout::<f32>::new(0.5);
        dropout.set_training(false);
        assert!(!dropout.is_training());

        let input_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let input = Tensor::from_array(input_data.into_dyn());

        let output = dropout.forward(&input).expect("Forward pass failed");

        // During inference, output should be identical to input
        assert_eq!(output.shape().dims(), input.shape().dims());
        // Values should be the same (no dropout applied)
        if let (Some(input_data), Some(output_data)) = (input.as_slice(), output.as_slice()) {
            for (a, b) in input_data.iter().zip(output_data.iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_dropout_zero_rate() {
        let mut dropout = Dropout::<f32>::new(0.0);
        dropout.set_training(true);

        let input_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let input = Tensor::from_array(input_data.into_dyn());

        let output = dropout.forward(&input).expect("Forward pass failed");

        // With zero dropout rate, output should be identical to input
        if let (Some(input_data), Some(output_data)) = (input.as_slice(), output.as_slice()) {
            for (a, b) in input_data.iter().zip(output_data.iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_dropout_full_rate() {
        let mut dropout = Dropout::<f32>::new(1.0);
        dropout.set_training(true);

        let input_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let input = Tensor::from_array(input_data.into_dyn());

        let output = dropout.forward(&input).expect("Forward pass failed");

        // With full dropout rate, all outputs should be zero
        if let Some(output_data) = output.as_slice() {
            for &val in output_data {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_dropout_parameters() {
        let mut dropout = Dropout::<f32>::new(0.5);

        // Dropout should have no parameters
        assert!(dropout.parameters().is_empty());
        assert!(dropout.parameters_mut().is_empty());
    }

    #[test]
    fn test_dropout_training_mode_toggle() {
        let mut dropout = Dropout::<f32>::new(0.5);

        assert!(dropout.is_training());

        dropout.set_training(false);
        assert!(!dropout.is_training());

        dropout.set_training(true);
        assert!(dropout.is_training());
    }
}
