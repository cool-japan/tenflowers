//! Batch Normalization layer implementation
//!
//! Batch Normalization normalizes the input by subtracting the batch mean and dividing
//! by the batch standard deviation, then applying learned scale and shift parameters.
//! This helps stabilize and accelerate training of deep neural networks.

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::cell::RefCell;
use tenflowers_core::{Result, Tensor};

#[derive(Clone)]
pub struct BatchNorm<T> {
    #[allow(dead_code)]
    num_features: usize,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    running_mean: RefCell<Tensor<T>>,
    running_var: RefCell<Tensor<T>>,
    #[allow(dead_code)]
    momentum: f32,
    #[allow(dead_code)]
    epsilon: f32,
    training: bool,
}

impl<T> BatchNorm<T>
where
    T: Clone + Default + Zero + One,
{
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: RefCell::new(Tensor::zeros(&[num_features])),
            running_var: RefCell::new(Tensor::ones(&[num_features])),
            momentum: 0.1,
            epsilon: 1e-5,
            training: false,
        }
    }

    /// Set the momentum for exponential moving average of running statistics
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the epsilon value for numerical stability
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set both momentum and epsilon values
    pub fn with_momentum_and_epsilon(mut self, momentum: f32, epsilon: f32) -> Self {
        self.momentum = momentum;
        self.epsilon = epsilon;
        self
    }
}

impl<T> Layer<T> for BatchNorm<T>
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

        // Batch norm expects at least 2 dimensions (batch, features, ...)
        if ndim < 2 {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "BatchNorm expects at least 2D input, got {ndim}D"
            )));
        }

        // For now, assume channel-last format (batch, ..., features)
        let num_features = shape.dims()[ndim - 1];
        if num_features != self.num_features {
            return Err(tenflowers_core::TensorError::invalid_shape_simple(format!(
                "Expected {} features, got {} and {}",
                self.num_features, self.num_features, num_features
            )));
        }

        if self.training {
            // Training mode: compute batch statistics
            // Compute mean and variance along all axes except the last (features)
            let axes: Vec<i32> = (0..ndim - 1).map(|i| i as i32).collect();

            // Mean along batch dimensions
            let batch_mean = input.mean(Some(&axes), true)?;

            // Variance: E[(X - mean)^2]
            let centered = input.sub(&batch_mean)?;
            let squared = centered.mul(&centered)?;
            let batch_var = squared.mean(Some(&axes), true)?;

            // Normalize
            let eps = T::from(self.epsilon).unwrap();
            let eps_tensor = Tensor::from_scalar(eps);
            let std = batch_var.add(&eps_tensor)?.sqrt()?;
            let normalized = centered.div(&std)?;

            // Scale and shift
            let output = normalized.mul(&self.gamma)?.add(&self.beta)?;

            // Update running statistics using exponential moving average
            let momentum_t = T::from(self.momentum).unwrap();
            let one_minus_momentum = T::one() - momentum_t;
            let momentum_tensor = Tensor::from_scalar(momentum_t);
            let one_minus_momentum_tensor = Tensor::from_scalar(one_minus_momentum);

            // Update running mean: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            {
                let mut running_mean = self.running_mean.borrow_mut();
                let scaled_running_mean = running_mean.mul(&one_minus_momentum_tensor)?;
                let scaled_batch_mean = batch_mean.mul(&momentum_tensor)?;
                *running_mean = scaled_running_mean.add(&scaled_batch_mean)?;
            }

            // Update running variance: running_var = (1 - momentum) * running_var + momentum * batch_var
            {
                let mut running_var = self.running_var.borrow_mut();
                let scaled_running_var = running_var.mul(&one_minus_momentum_tensor)?;
                let scaled_batch_var = batch_var.mul(&momentum_tensor)?;
                *running_var = scaled_running_var.add(&scaled_batch_var)?;
            }

            Ok(output)
        } else {
            // Inference mode: use running statistics
            let eps = T::from(self.epsilon).unwrap();
            let eps_tensor = Tensor::from_scalar(eps);

            // Normalize using running stats
            let running_mean = self.running_mean.borrow();
            let running_var = self.running_var.borrow();
            let centered = input.sub(&*running_mean)?;
            let std = running_var.add(&eps_tensor)?.sqrt()?;
            let normalized = centered.div(&std)?;

            // Scale and shift
            let output = normalized.mul(&self.gamma)?.add(&self.beta)?;

            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_creation() {
        let batch_norm = BatchNorm::<f32>::new(64);
        assert_eq!(batch_norm.num_features, 64);
        assert!(!batch_norm.training);
    }

    #[test]
    fn test_batch_norm_builder_pattern() {
        let batch_norm = BatchNorm::<f32>::new(64)
            .with_momentum(0.05)
            .with_epsilon(1e-6);

        assert_eq!(batch_norm.momentum, 0.05);
        assert_eq!(batch_norm.epsilon, 1e-6);
    }

    #[test]
    fn test_batch_norm_parameters() {
        let batch_norm = BatchNorm::<f32>::new(32);
        let params = batch_norm.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }

    #[test]
    fn test_batch_norm_training_mode() {
        let mut batch_norm = BatchNorm::<f32>::new(16);
        assert!(!batch_norm.training);

        batch_norm.set_training(true);
        assert!(batch_norm.training);
    }
}
