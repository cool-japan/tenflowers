//! Vanilla RNN layer implementation

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Random;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use tenflowers_core::{device::context::get_gpu_context, gpu::rnn_ops::GpuRnnOps};

/// RNN (Vanilla Recurrent Neural Network) layer
#[derive(Debug)]
pub struct RNN<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    dropout: f32,
    bidirectional: bool,

    // RNN parameters for each layer
    weight_ih: Vec<Tensor<T>>, // Input-to-hidden weights [input_size, hidden_size]
    weight_hh: Vec<Tensor<T>>, // Hidden-to-hidden weights [hidden_size, hidden_size]
    bias_ih: Option<Vec<Tensor<T>>>, // Input-to-hidden bias [hidden_size]
    bias_hh: Option<Vec<Tensor<T>>>, // Hidden-to-hidden bias [hidden_size]

    // For bidirectional RNN
    weight_ih_reverse: Option<Vec<Tensor<T>>>,
    weight_hh_reverse: Option<Vec<Tensor<T>>>,
    bias_ih_reverse: Option<Vec<Tensor<T>>>,
    bias_hh_reverse: Option<Vec<Tensor<T>>>,

    training: bool,
}

impl<T> Clone for RNN<T>
where
    T: Float
        + Clone
        + Default
        + Zero
        + One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn clone(&self) -> Self {
        Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            bias: self.bias,
            batch_first: self.batch_first,
            dropout: self.dropout,
            bidirectional: self.bidirectional,
            weight_ih: self.weight_ih.clone(),
            weight_hh: self.weight_hh.clone(),
            bias_ih: self.bias_ih.clone(),
            bias_hh: self.bias_hh.clone(),
            weight_ih_reverse: self.weight_ih_reverse.clone(),
            weight_hh_reverse: self.weight_hh_reverse.clone(),
            bias_ih_reverse: self.bias_ih_reverse.clone(),
            bias_hh_reverse: self.bias_hh_reverse.clone(),
            training: self.training,
        }
    }
}

impl<T> RNN<T>
where
    T: Float
        + Zero
        + One
        + FromPrimitive
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(TensorError::invalid_argument(
                "num_layers must be >= 1".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&dropout) {
            return Err(TensorError::invalid_argument(
                "dropout must be between 0 and 1".to_string(),
            ));
        }

        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = if bias { Some(Vec::new()) } else { None };
        let mut bias_hh = if bias { Some(Vec::new()) } else { None };

        // Initialize parameters for each layer
        for i in 0..num_layers {
            let input_dim = if i == 0 {
                input_size
            } else {
                hidden_size * if bidirectional { 2 } else { 1 }
            };

            // Xavier/Glorot initialization
            let scale = T::from(1.0 / (input_dim as f64).sqrt()).unwrap();

            // Input-to-hidden weights: [input_dim, hidden_size]
            let w_ih = Self::init_weight(&[input_dim, hidden_size], scale)?;
            weight_ih.push(w_ih);

            // Hidden-to-hidden weights: [hidden_size, hidden_size]
            let w_hh = Self::init_weight(&[hidden_size, hidden_size], scale)?;
            weight_hh.push(w_hh);

            if bias {
                bias_ih
                    .as_mut()
                    .unwrap()
                    .push(Tensor::zeros(&[hidden_size]));
                bias_hh
                    .as_mut()
                    .unwrap()
                    .push(Tensor::zeros(&[hidden_size]));
            }
        }

        // Initialize bidirectional parameters if needed
        let (weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse) =
            if bidirectional {
                let mut w_ih_rev = Vec::new();
                let mut w_hh_rev = Vec::new();
                let mut b_ih_rev = if bias { Some(Vec::new()) } else { None };
                let mut b_hh_rev = if bias { Some(Vec::new()) } else { None };

                for i in 0..num_layers {
                    let input_dim = if i == 0 { input_size } else { hidden_size * 2 };
                    let scale = T::from(1.0 / (input_dim as f64).sqrt()).unwrap();

                    w_ih_rev.push(Self::init_weight(&[input_dim, hidden_size], scale)?);
                    w_hh_rev.push(Self::init_weight(&[hidden_size, hidden_size], scale)?);

                    if bias {
                        b_ih_rev
                            .as_mut()
                            .unwrap()
                            .push(Tensor::zeros(&[hidden_size]));
                        b_hh_rev
                            .as_mut()
                            .unwrap()
                            .push(Tensor::zeros(&[hidden_size]));
                    }
                }

                (Some(w_ih_rev), Some(w_hh_rev), b_ih_rev, b_hh_rev)
            } else {
                (None, None, None, None)
            };

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            weight_ih_reverse,
            weight_hh_reverse,
            bias_ih_reverse,
            bias_hh_reverse,
            training: true,
        })
    }

    fn init_weight(shape: &[usize], scale: T) -> Result<Tensor<T>> {
        let mut rng = Random::seed(0);
        let total_elements = shape.iter().product::<usize>();

        // Generate random values in [-scale, scale] range
        let values: Vec<T> = (0..total_elements)
            .map(|_| {
                let random_val = rng.gen_range(-1.0..1.0);
                T::from(random_val).unwrap() * scale
            })
            .collect();

        Tensor::from_data(values, shape)
    }

    /// Forward pass through the RNN
    pub fn forward_with_hidden(
        &self,
        input: &Tensor<T>,
        hidden: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let input_shape = input.shape().dims();
        let (batch_size, seq_len, input_dim) = if self.batch_first {
            (input_shape[0], input_shape[1], input_shape[2])
        } else {
            (input_shape[1], input_shape[0], input_shape[2])
        };

        if input_dim != self.input_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected input size {}, got {}",
                self.input_size, input_dim
            )));
        }

        let hidden_size_total =
            self.hidden_size * self.num_layers * if self.bidirectional { 2 } else { 1 };

        // Initialize hidden state if not provided
        let mut h = if let Some(h) = hidden {
            h.clone()
        } else {
            Tensor::zeros(&[
                self.num_layers * if self.bidirectional { 2 } else { 1 },
                batch_size,
                self.hidden_size,
            ])
        };

        let mut output = if self.batch_first {
            input.clone()
        } else {
            // Transpose from [seq_len, batch, input_size] to [batch, seq_len, input_size]
            input.transpose()?
        };

        // Process each layer
        for layer in 0..self.num_layers {
            let (layer_output, layer_hidden) = self.forward_single_layer(layer, &output, &h)?;
            output = layer_output;

            // Update hidden state for this layer
            let start_idx = layer * if self.bidirectional { 2 } else { 1 };
            let end_idx = (layer + 1) * if self.bidirectional { 2 } else { 1 };
            h = Self::update_hidden_slice(&h, &layer_hidden, start_idx, end_idx)?;
        }

        // Transpose back if needed
        let output = if self.batch_first {
            output
        } else {
            output.transpose()?
        };

        Ok((output, h))
    }

    fn forward_single_layer(
        &self,
        layer: usize,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let input_size = input_shape[2];

        // Extract weights for this layer
        let w_ih = &self.weight_ih[layer];
        let w_hh = &self.weight_hh[layer];
        let b_ih = self.bias_ih.as_ref().map(|b| &b[layer]);
        let b_hh = self.bias_hh.as_ref().map(|b| &b[layer]);

        // Extract hidden state for this layer
        let h_start = layer * if self.bidirectional { 2 } else { 1 };
        let h_end = h_start + 1;
        let mut h_prev = Self::extract_hidden_slice(hidden, h_start, h_end)?;

        let mut outputs = Vec::new();

        // Forward pass through sequence
        for t in 0..seq_len {
            // Extract input at time step t
            let x_t = Self::extract_timestep(input, t)?;

            // RNN cell computation: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
            let ih_output = tenflowers_core::ops::matmul(&x_t, w_ih)?;
            let hh_output = tenflowers_core::ops::matmul(&h_prev, w_hh)?;

            let mut combined = tenflowers_core::ops::add(&ih_output, &hh_output)?;

            // Add biases if present
            if let Some(b_ih) = b_ih {
                combined = tenflowers_core::ops::add(&combined, b_ih)?;
            }
            if let Some(b_hh) = b_hh {
                combined = tenflowers_core::ops::add(&combined, b_hh)?;
            }

            // Apply activation function (tanh)
            let h_t = tenflowers_core::ops::activation::tanh(&combined)?;

            outputs.push(h_t.clone());
            h_prev = h_t;
        }

        // Stack outputs along sequence dimension
        let output = Self::stack_sequence_outputs(&outputs)?;

        // Handle bidirectional case
        if self.bidirectional {
            let w_ih_rev = self.weight_ih_reverse.as_ref().unwrap();
            let w_hh_rev = self.weight_hh_reverse.as_ref().unwrap();
            let b_ih_rev = self.bias_ih_reverse.as_ref().map(|b| &b[layer]);
            let b_hh_rev = self.bias_hh_reverse.as_ref().map(|b| &b[layer]);

            let h_rev_start = h_start + 1;
            let h_rev_end = h_rev_start + 1;
            let mut h_prev_rev = Self::extract_hidden_slice(hidden, h_rev_start, h_rev_end)?;

            let mut outputs_rev = Vec::new();

            // Backward pass through sequence (reverse order)
            for t in (0..seq_len).rev() {
                let x_t = Self::extract_timestep(input, t)?;

                let ih_output = tenflowers_core::ops::matmul(&x_t, &w_ih_rev[layer])?;
                let hh_output = tenflowers_core::ops::matmul(&h_prev_rev, &w_hh_rev[layer])?;

                let mut combined = tenflowers_core::ops::add(&ih_output, &hh_output)?;

                if let Some(b_ih) = b_ih_rev {
                    combined = tenflowers_core::ops::add(&combined, b_ih)?;
                }
                if let Some(b_hh) = b_hh_rev {
                    combined = tenflowers_core::ops::add(&combined, b_hh)?;
                }

                let h_t = tenflowers_core::ops::activation::tanh(&combined)?;

                outputs_rev.push(h_t.clone());
                h_prev_rev = h_t;
            }

            // Reverse the reverse outputs to match sequence order
            outputs_rev.reverse();
            let output_rev = Self::stack_sequence_outputs(&outputs_rev)?;

            // Concatenate forward and backward outputs
            let combined_output = Self::concatenate_tensors(&[&output, &output_rev], 2)?;
            let combined_hidden = Self::concatenate_tensors(&[&h_prev, &h_prev_rev], 1)?;

            Ok((combined_output, combined_hidden))
        } else {
            Ok((output, h_prev.unsqueeze(&[0])?))
        }
    }

    // Helper functions
    fn extract_timestep(input: &Tensor<T>, timestep: usize) -> Result<Tensor<T>> {
        // Extract a specific timestep from [batch, seq, features] -> [batch, features]
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let features = input_shape[2];

        // Create slice indices
        let mut result = Tensor::zeros(&[batch_size, features]);

        // This is a simplified implementation - in practice would use proper slicing
        // For now, just return zeros as placeholder
        Ok(result)
    }

    fn stack_sequence_outputs(outputs: &[Tensor<T>]) -> Result<Tensor<T>> {
        // Stack a sequence of tensors along a new dimension
        if outputs.is_empty() {
            return Err(TensorError::invalid_argument(
                "Empty outputs sequence".to_string(),
            ));
        }

        // Get the shape of individual outputs [batch_size, hidden_size]
        let first_shape = outputs[0].shape().dims();
        let batch_size = first_shape[0];
        let hidden_size = first_shape[1];
        let seq_len = outputs.len();

        // Create output tensor with shape [batch_size, seq_len, hidden_size]
        let mut result_data = Vec::new();

        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                if let Some(output_data) = outputs[seq_idx].as_slice() {
                    let start_idx = batch_idx * hidden_size;
                    let end_idx = start_idx + hidden_size;
                    result_data.extend_from_slice(&output_data[start_idx..end_idx]);
                }
            }
        }

        Tensor::from_vec(result_data, &[batch_size, seq_len, hidden_size])
    }

    fn extract_hidden_slice(hidden: &Tensor<T>, start: usize, end: usize) -> Result<Tensor<T>> {
        // Extract a slice from the hidden state tensor
        // This is a simplified implementation
        let shape = hidden.shape().dims();
        Ok(Tensor::zeros(&[shape[1], shape[2]]))
    }

    fn update_hidden_slice(
        hidden: &Tensor<T>,
        new_slice: &Tensor<T>,
        start: usize,
        end: usize,
    ) -> Result<Tensor<T>> {
        // Update a slice in the hidden state tensor
        // This is a simplified implementation
        Ok(hidden.clone())
    }

    /// Helper function to concatenate tensors along a specified axis
    fn concatenate_tensors(tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(TensorError::invalid_argument(
                "Empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        // Get shapes of all tensors
        let shapes: Vec<_> = tensors.iter().map(|t| t.shape().dims()).collect();
        let first_shape = shapes[0];

        // Validate that all tensors have same number of dimensions
        for (i, shape) in shapes.iter().enumerate() {
            if shape.len() != first_shape.len() {
                return Err(TensorError::invalid_argument(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    shape.len(),
                    first_shape.len()
                )));
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        for shape in &shapes[1..] {
            for (i, (&dim1, &dim2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i == axis {
                    output_shape[i] += dim2;
                } else if dim1 != dim2 {
                    return Err(TensorError::invalid_argument(format!(
                        "Dimension {} mismatch: {} vs {}",
                        i, dim1, dim2
                    )));
                }
            }
        }

        // Simple concatenation implementation
        // For now, just return the first tensor as a placeholder
        // In production, would properly concatenate along the specified axis
        let total_elements = output_shape.iter().product::<usize>();
        let output_data = vec![T::zero(); total_elements];

        Tensor::from_data(output_data, &output_shape)
    }
}

impl<T> Layer<T> for RNN<T>
where
    T: Float
        + Zero
        + One
        + FromPrimitive
        + Clone
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let (output, _hidden) = self.forward_with_hidden(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        // Add forward direction parameters
        for w in &self.weight_ih {
            params.push(w);
        }
        for w in &self.weight_hh {
            params.push(w);
        }

        if let Some(ref biases) = self.bias_ih {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref biases) = self.bias_hh {
            for b in biases {
                params.push(b);
            }
        }

        // Add reverse direction parameters if bidirectional
        if let Some(ref weights) = self.weight_ih_reverse {
            for w in weights {
                params.push(w);
            }
        }
        if let Some(ref weights) = self.weight_hh_reverse {
            for w in weights {
                params.push(w);
            }
        }
        if let Some(ref biases) = self.bias_ih_reverse {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref biases) = self.bias_hh_reverse {
            for b in biases {
                params.push(b);
            }
        }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();

        // Add forward direction parameters
        for w in &mut self.weight_ih {
            params.push(w);
        }
        for w in &mut self.weight_hh {
            params.push(w);
        }

        if let Some(ref mut biases) = self.bias_ih {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref mut biases) = self.bias_hh {
            for b in biases {
                params.push(b);
            }
        }

        // Add reverse direction parameters if bidirectional
        if let Some(ref mut weights) = self.weight_ih_reverse {
            for w in weights {
                params.push(w);
            }
        }
        if let Some(ref mut weights) = self.weight_hh_reverse {
            for w in weights {
                params.push(w);
            }
        }
        if let Some(ref mut biases) = self.bias_ih_reverse {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref mut biases) = self.bias_hh_reverse {
            for b in biases {
                params.push(b);
            }
        }

        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
