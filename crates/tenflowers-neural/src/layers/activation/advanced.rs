//! Advanced activation functions used in modern architectures

use crate::layers::activation::utils::create_random_tensor;
use crate::layers::Layer;
use num_traits::{Float, One, Zero};
use tenflowers_core::{Result, Tensor};

/// SwiGLU (Swish-Gated Linear Unit) activation function
///
/// SwiGLU is a gated activation function that has become popular in modern transformer
/// architectures like LLaMA, PaLM, and other large language models. It combines the
/// benefits of gated mechanisms with the smooth properties of Swish activation.
///
/// Formula: SwiGLU(x) = Swish(xW_gate + b_gate) ⊙ (xW_up + b_up)
/// where:
/// - Swish(x) = x * sigmoid(x)
/// - ⊙ is element-wise multiplication
/// - W_gate, W_up are weight matrices
/// - b_gate, b_up are bias vectors
///
/// This implementation projects the input to a higher-dimensional space and then
/// applies the gating mechanism, which has been shown to improve model performance
pub struct SwiGLU<T> {
    /// Weight matrix for the gate path
    pub w_gate: Tensor<T>,
    /// Bias vector for the gate path
    pub b_gate: Tensor<T>,
    /// Weight matrix for the up projection path
    pub w_up: Tensor<T>,
    /// Bias vector for the up projection path
    pub b_up: Tensor<T>,
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension (typically 2/3 * 4 * input_dim for transformer FFN)
    pub hidden_dim: usize,
}

impl<T> SwiGLU<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + std::fmt::Debug
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    /// Create a new SwiGLU layer
    ///
    /// # Arguments
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dim` - Hidden dimension (output dimension)
    ///
    /// The weights are initialized using Xavier/Glorot uniform initialization
    /// which is appropriate for activations with approximately unit variance.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(tenflowers_core::TensorError::invalid_operation_simple(
                "SwiGLU dimensions must be greater than 0".to_string(),
            ));
        }

        // Xavier/Glorot initialization: limit = sqrt(6 / (fan_in + fan_out))
        let gate_limit = T::from((6.0_f64 / (input_dim + hidden_dim) as f64).sqrt()).unwrap();
        let up_limit = T::from((6.0_f64 / (input_dim + hidden_dim) as f64).sqrt()).unwrap();

        // Initialize weights with Xavier/Glorot normal distribution
        let w_gate = create_random_tensor(&[input_dim, hidden_dim], gate_limit)?;
        let w_up = create_random_tensor(&[input_dim, hidden_dim], up_limit)?;

        // Initialize biases to zero (common practice)
        let b_gate = Tensor::zeros(&[hidden_dim]);
        let b_up = Tensor::zeros(&[hidden_dim]);

        Ok(SwiGLU {
            w_gate,
            b_gate,
            w_up,
            b_up,
            input_dim,
            hidden_dim,
        })
    }

    /// Create a new SwiGLU layer with custom weight initialization
    ///
    /// # Arguments
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dim` - Hidden dimension
    /// * `init_std` - Standard deviation for weight initialization
    pub fn new_with_init(input_dim: usize, hidden_dim: usize, init_std: T) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(tenflowers_core::TensorError::invalid_operation_simple(
                "SwiGLU dimensions must be greater than 0".to_string(),
            ));
        }

        let w_gate = create_random_tensor(&[input_dim, hidden_dim], init_std)?;
        let w_up = create_random_tensor(&[input_dim, hidden_dim], init_std)?;
        let b_gate = Tensor::zeros(&[hidden_dim]);
        let b_up = Tensor::zeros(&[hidden_dim]);

        Ok(SwiGLU {
            w_gate,
            b_gate,
            w_up,
            b_up,
            input_dim,
            hidden_dim,
        })
    }

    /// Forward pass implementation
    ///
    /// Input shape: [batch_size, input_dim] or [input_dim]
    /// Output shape: [batch_size, hidden_dim] or [hidden_dim]
    fn forward_impl(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Gate path: x * W_gate + b_gate
        let gate_linear = input.matmul(&self.w_gate)?.add(&self.b_gate)?;

        // Apply Swish activation: Swish(x) = x * sigmoid(x)
        let gate_sigmoid = tenflowers_core::ops::activation::sigmoid(&gate_linear)?;
        let gate_swish = gate_linear.mul(&gate_sigmoid)?;

        // Up projection path: x * W_up + b_up
        let up_linear = input.matmul(&self.w_up)?.add(&self.b_up)?;

        // Final gated output: Swish(gate) ⊙ up
        gate_swish.mul(&up_linear)
    }
}

impl<T> Layer<T> for SwiGLU<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + std::fmt::Debug
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // Validate input dimensions
        let input_shape = input.shape();
        let input_dims = input_shape.dims();

        if input_dims.is_empty() || input_dims.len() > 2 {
            return Err(tenflowers_core::TensorError::invalid_operation_simple(
                "SwiGLU input must be 1D or 2D tensor".to_string(),
            ));
        }

        let last_dim = *input_dims.last().unwrap();
        if last_dim != self.input_dim {
            return Err(tenflowers_core::TensorError::invalid_operation_simple(
                format!(
                    "Input last dimension {} does not match expected {}",
                    last_dim, self.input_dim
                ),
            ));
        }

        self.forward_impl(input)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.w_gate, &self.b_gate, &self.w_up, &self.b_up]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![
            &mut self.w_gate,
            &mut self.b_gate,
            &mut self.w_up,
            &mut self.b_up,
        ]
    }

    fn set_training(&mut self, _training: bool) {
        // SwiGLU doesn't have different behavior in training/eval mode
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new((*self).clone())
    }
}

impl<T> Clone for SwiGLU<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + std::fmt::Debug
        + Send
        + Sync
        + 'static
        + num_traits::FromPrimitive
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn clone(&self) -> Self {
        SwiGLU {
            w_gate: self.w_gate.clone(),
            b_gate: self.b_gate.clone(),
            w_up: self.w_up.clone(),
            b_up: self.b_up.clone(),
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
        }
    }
}

#[cfg(test)]
mod advanced_activation_tests {
    use super::*;

    #[test]
    fn test_swiglu_basic() {
        let swiglu = SwiGLU::<f32>::new(4, 8).unwrap();

        // Check parameters
        assert_eq!(swiglu.parameters().len(), 4); // W_gate, b_gate, W_up, b_up

        let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let output = swiglu.forward(&input).unwrap();

        // Output dimension should be hidden_dim (8)
        assert_eq!(output.shape().dims(), &[1, 8]);

        // All outputs should be finite
        if let Some(data) = output.as_slice() {
            for &val in data {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_swiglu_dimension_validation() {
        // Test invalid dimensions
        let result = SwiGLU::<f32>::new(0, 8);
        assert!(result.is_err());

        let result = SwiGLU::<f32>::new(4, 0);
        assert!(result.is_err());

        // Test valid dimensions
        let result = SwiGLU::<f32>::new(4, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_swiglu_custom_init() {
        let swiglu = SwiGLU::<f32>::new_with_init(4, 8, 0.02).unwrap();
        assert_eq!(swiglu.input_dim, 4);
        assert_eq!(swiglu.hidden_dim, 8);

        let input = Tensor::<f32>::from_vec(vec![1.0; 4], &[4]).unwrap();
        let output = swiglu.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[8]);
    }
}
