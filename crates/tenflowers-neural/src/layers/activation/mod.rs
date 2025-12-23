//! Activation function implementations for neural networks
//!
//! This module provides a comprehensive set of activation functions including:
//! - Basic activations (ReLU, Sigmoid, Tanh, etc.)
//! - Parametric activations (PReLU, ParametricSoftplus)
//! - Adaptive activations (AdaptiveSwish, AdaptivePiecewiseLinear, AdaptivePolynomial)
//! - Advanced activations (SwiGLU)

pub mod adaptive;
pub mod advanced;
pub mod parametric;
pub mod utils;

// Re-export the main activation functions for easy access
pub use adaptive::{AdaptivePiecewiseLinear, AdaptivePolynomial, AdaptiveSwish};
pub use advanced::SwiGLU;
pub use parametric::{PReLU, ParametricSoftplus};

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, Zero};
use tenflowers_core::{ops::manipulation::where_op, Result, Tensor};

/// Basic activation functions enum for common non-parametric activations
#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Mish,
    HardSwish,
    HardSigmoid,
    CELU { alpha: f64 },
}

impl<T> Layer<T> for Activation
where
    T: Float
        + Clone
        + Default
        + Zero
        + std::cmp::PartialOrd
        + std::iter::Sum
        + std::ops::Sub<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        match self {
            Activation::ReLU => tenflowers_core::ops::relu(input),
            Activation::Sigmoid => tenflowers_core::ops::sigmoid(input),
            Activation::Tanh => tenflowers_core::ops::tanh(input),
            Activation::Softmax => tenflowers_core::ops::softmax(input, None),
            Activation::Mish => {
                // Mish(x) = x * tanh(softplus(x))
                // where softplus(x) = log(1 + exp(x))
                let exp_x = input.exp()?;
                let one_plus_exp_x = exp_x.add(&Tensor::from_scalar(T::one()))?;
                let softplus_x = one_plus_exp_x.log()?;
                let tanh_softplus = tenflowers_core::ops::tanh(&softplus_x)?;
                input.mul(&tanh_softplus)
            }
            Activation::HardSwish => {
                // HardSwish(x) = x * hard_sigmoid(x)
                // where hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1)
                let three = Tensor::from_scalar(T::from(3.0).unwrap());
                let six = Tensor::from_scalar(T::from(6.0).unwrap());
                let zero = Tensor::from_scalar(T::zero());
                let one = Tensor::from_scalar(T::one());

                let x_plus_3 = input.add(&three)?;
                let hard_sigmoid = x_plus_3.div(&six)?;
                // Element-wise maximum with zero (equivalent to max(hard_sigmoid, zero))
                let max_with_zero = where_op(&hard_sigmoid.ge(&zero)?, &hard_sigmoid, &zero)?;
                // Element-wise minimum with one (equivalent to min(max_with_zero, one))
                let clamped = where_op(&max_with_zero.le(&one)?, &max_with_zero, &one)?;

                input.mul(&clamped)
            }
            Activation::HardSigmoid => {
                // HardSigmoid(x) = clamp((x + 3) / 6, 0, 1)
                let three = Tensor::from_scalar(T::from(3.0).unwrap());
                let six = Tensor::from_scalar(T::from(6.0).unwrap());
                let zero = Tensor::from_scalar(T::zero());
                let one = Tensor::from_scalar(T::one());

                let x_plus_3 = input.add(&three)?;
                let hard_sigmoid = x_plus_3.div(&six)?;
                // Element-wise clamp(hard_sigmoid, 0, 1)
                let max_with_zero = where_op(&hard_sigmoid.ge(&zero)?, &hard_sigmoid, &zero)?;
                where_op(&max_with_zero.le(&one)?, &max_with_zero, &one)
            }
            Activation::CELU { alpha } => {
                // CELU(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
                let alpha_tensor = Tensor::from_scalar(T::from(*alpha).unwrap());
                let zero = Tensor::from_scalar(T::zero());
                let one = Tensor::from_scalar(T::one());

                // Positive part: max(0, x)
                let positive_part = where_op(&input.ge(&zero)?, input, &zero)?;

                // Negative part: min(0, alpha * (exp(x / alpha) - 1))
                let x_div_alpha = input.div(&alpha_tensor)?;
                let exp_part = x_div_alpha.exp()?;
                let exp_minus_one = exp_part.sub(&one)?;
                let scaled_exp = alpha_tensor.mul(&exp_minus_one)?;
                let negative_part = where_op(&scaled_exp.le(&zero)?, &scaled_exp, &zero)?;

                positive_part.add(&negative_part)
            }
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Activation layers don't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    #[test]
    fn test_basic_activations() {
        let input = Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();

        // Test ReLU
        let relu = Activation::ReLU;
        let output = relu.forward(&input).unwrap();
        if let Some(data) = output.as_slice() {
            assert_eq!(data[0], 0.0); // ReLU(-1) = 0
            assert_eq!(data[1], 0.0); // ReLU(0) = 0
            assert_eq!(data[2], 1.0); // ReLU(1) = 1
        }

        // Test Sigmoid
        let sigmoid = Activation::Sigmoid;
        let output = sigmoid.forward(&input).unwrap();
        if let Some(data) = output.as_slice() {
            assert!(data[0] > 0.0 && data[0] < 0.5); // sigmoid(-1) ≈ 0.27
            assert!((data[1] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
            assert!(data[2] > 0.5 && data[2] < 1.0); // sigmoid(1) ≈ 0.73
        }

        // Test CELU with alpha
        let celu = Activation::CELU { alpha: 1.0 };
        let output = celu.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), input.shape().dims());
    }

    #[test]
    fn test_activation_clone() {
        let relu = Activation::ReLU;
        let relu_clone = relu.clone();
        assert_eq!(relu, relu_clone);

        let celu = Activation::CELU { alpha: 2.0 };
        let celu_clone = celu.clone();
        assert_eq!(celu, celu_clone);
    }

    #[test]
    fn test_activation_parameters() {
        let relu = Activation::ReLU;
        assert_eq!(Layer::<f32>::parameters(&relu).len(), 0); // Basic activations have no parameters

        let mut sigmoid = Activation::Sigmoid;
        assert_eq!(Layer::<f32>::parameters_mut(&mut sigmoid).len(), 0);
    }
}
