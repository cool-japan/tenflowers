//! LSTM (Long Short-Term Memory) layer implementation

use crate::layers::Layer;
use num_traits::{Float, FromPrimitive, One, Zero};
// use scirs2_autograd::ndarray::{Array1, Array2, Axis}; // Unused for now
// use scirs2_core::random::{rng, Random}; // Unused for now
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use tenflowers_core::{device::context::get_gpu_context, gpu::rnn_ops::GpuRnnOps};

/// LSTM (Long Short-Term Memory) layer
#[derive(Debug)]
pub struct LSTM<T>
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

    // LSTM parameters for each layer
    weight_ih: Vec<Tensor<T>>, // Input-to-hidden weights [input_size, 4*hidden_size]
    weight_hh: Vec<Tensor<T>>, // Hidden-to-hidden weights [hidden_size, 4*hidden_size]
    bias_ih: Option<Vec<Tensor<T>>>, // Input-to-hidden bias [4*hidden_size]
    bias_hh: Option<Vec<Tensor<T>>>, // Hidden-to-hidden bias [4*hidden_size]

    // For bidirectional LSTM
    weight_ih_reverse: Option<Vec<Tensor<T>>>,
    weight_hh_reverse: Option<Vec<Tensor<T>>>,
    bias_ih_reverse: Option<Vec<Tensor<T>>>,
    bias_hh_reverse: Option<Vec<Tensor<T>>>,

    training: bool,
}

impl<T> Clone for LSTM<T>
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

impl<T> LSTM<T>
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

            // Xavier/Glorot initialization for weights
            let scale = T::from(1.0 / (input_dim as f64).sqrt()).unwrap();

            // Input-to-hidden weights: [input_dim, 4*hidden_size] (i, f, g, o gates)
            let w_ih = Self::init_weight(&[input_dim, 4 * hidden_size], scale)?;
            weight_ih.push(w_ih);

            // Hidden-to-hidden weights: [hidden_size, 4*hidden_size]
            let w_hh = Self::init_weight(&[hidden_size, 4 * hidden_size], scale)?;
            weight_hh.push(w_hh);

            if bias {
                // Initialize biases: zero everywhere except forget gate bias set to 1
                let mut b_ih_data = vec![T::zero(); 4 * hidden_size];
                let mut b_hh_data = vec![T::zero(); 4 * hidden_size];

                // Set forget gate bias to 1 (indices hidden_size to 2*hidden_size)
                for idx in hidden_size..(2 * hidden_size) {
                    if idx < b_ih_data.len() {
                        b_ih_data[idx] = T::one();
                    }
                    if idx < b_hh_data.len() {
                        b_hh_data[idx] = T::one();
                    }
                }

                let b_ih = Tensor::from_vec(b_ih_data, &[4 * hidden_size])?;
                let b_hh = Tensor::from_vec(b_hh_data, &[4 * hidden_size])?;

                bias_ih.as_mut().unwrap().push(b_ih);
                bias_hh.as_mut().unwrap().push(b_hh);
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

                    w_ih_rev.push(Self::init_weight(&[input_dim, 4 * hidden_size], scale)?);
                    w_hh_rev.push(Self::init_weight(&[hidden_size, 4 * hidden_size], scale)?);

                    if bias {
                        let mut b_ih_data = vec![T::zero(); 4 * hidden_size];
                        let mut b_hh_data = vec![T::zero(); 4 * hidden_size];

                        // Set forget gate bias to 1
                        for idx in hidden_size..(2 * hidden_size) {
                            if idx < b_ih_data.len() {
                                b_ih_data[idx] = T::one();
                            }
                            if idx < b_hh_data.len() {
                                b_hh_data[idx] = T::one();
                            }
                        }

                        b_ih_rev
                            .as_mut()
                            .unwrap()
                            .push(Tensor::from_vec(b_ih_data, &[4 * hidden_size])?);
                        b_hh_rev
                            .as_mut()
                            .unwrap()
                            .push(Tensor::from_vec(b_hh_data, &[4 * hidden_size])?);
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
            training: false,
        })
    }

    /// Initialize weight tensor with Xavier/Glorot initialization
    fn init_weight(shape: &[usize], scale: T) -> Result<Tensor<T>> {
        let total_size: usize = shape.iter().product();
        let bound = scale * T::from(3.0).unwrap().sqrt(); // sqrt(3) for uniform distribution
        let data: Vec<T> = (0..total_size)
            .map(|i| {
                // Simple pseudo-random initialization based on index
                let pseudo_random = (i as f64 * 1.23456789) % 2.0 - 1.0; // Range [-1,1]
                T::from(pseudo_random).unwrap() * bound
            })
            .collect();
        Tensor::from_vec(data, shape)
    }

    /// LSTM cell computation for single timestep
    #[allow(clippy::too_many_arguments)]
    fn lstm_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        cell: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        // Compute input-hidden and hidden-hidden transformations
        let gi = input.matmul(weight_ih)?;
        let gh = hidden.matmul(weight_hh)?;

        // Add biases if present
        let gi = if let Some(bias) = bias_ih {
            gi.add(bias)?
        } else {
            gi
        };

        let gh = if let Some(bias) = bias_hh {
            gh.add(bias)?
        } else {
            gh
        };

        // Combine transformations
        let gates = gi.add(&gh)?;

        // Split gates: [input, forget, cell, output]
        let gate_size = self.hidden_size;
        let batch_size = gates.shape().dims()[0];
        let i_gate = gates.slice(&[0..batch_size, 0..gate_size])?; // Input gate
        let f_gate = gates.slice(&[0..batch_size, gate_size..2 * gate_size])?; // Forget gate
        let g_gate = gates.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?; // Cell gate
        let o_gate = gates.slice(&[0..batch_size, 3 * gate_size..4 * gate_size])?; // Output gate

        // Apply activations
        let i_gate = i_gate.sigmoid()?; // Input gate: sigmoid
        let f_gate = f_gate.sigmoid()?; // Forget gate: sigmoid
        let g_gate = g_gate.tanh()?; // Cell gate: tanh
        let o_gate = o_gate.sigmoid()?; // Output gate: sigmoid

        // Update cell state: C_t = f_t * C_{t-1} + i_t * g_t
        let new_cell = f_gate.mul(cell)?.add(&i_gate.mul(&g_gate)?)?;

        // Update hidden state: h_t = o_t * tanh(C_t)
        let new_hidden = o_gate.mul(&new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }

    /// Process sequence in forward direction
    fn forward_direction(
        &self,
        input: &Tensor<T>,
        layer: usize,
        reverse: bool,
    ) -> Result<Tensor<T>> {
        let input_shape = input.shape();
        let (batch_size, seq_length, input_size) = if self.batch_first {
            (input_shape[0], input_shape[1], input_shape[2])
        } else {
            (input_shape[1], input_shape[0], input_shape[2])
        };

        let weights_ih = if reverse {
            self.weight_ih_reverse.as_ref().unwrap()
        } else {
            &self.weight_ih
        };
        let weights_hh = if reverse {
            self.weight_hh_reverse.as_ref().unwrap()
        } else {
            &self.weight_hh
        };
        let biases_ih = if reverse {
            self.bias_ih_reverse.as_ref()
        } else {
            self.bias_ih.as_ref()
        };
        let biases_hh = if reverse {
            self.bias_hh_reverse.as_ref()
        } else {
            self.bias_hh.as_ref()
        };

        // Initialize hidden and cell states
        let mut hidden = Tensor::zeros(&[batch_size, self.hidden_size]);
        let mut cell = Tensor::zeros(&[batch_size, self.hidden_size]);

        let mut outputs = Vec::new();

        let time_indices: Vec<usize> = if reverse {
            (0..seq_length).rev().collect()
        } else {
            (0..seq_length).collect()
        };

        for &t in &time_indices {
            // Extract input at time t
            let x_t = if self.batch_first {
                input.slice(&[0..batch_size, t..t + 1, 0..input_size])?
            } else {
                input.slice(&[t..t + 1, 0..batch_size, 0..input_size])?
            };
            let x_t = x_t.squeeze(Some(if self.batch_first { &[1] } else { &[0] }))?;

            // LSTM cell computation
            let (new_hidden, new_cell) = self.lstm_cell(
                &x_t,
                &hidden,
                &cell,
                &weights_ih[layer],
                &weights_hh[layer],
                biases_ih.map(|b| &b[layer]),
                biases_hh.map(|b| &b[layer]),
            )?;

            hidden = new_hidden;
            cell = new_cell;
            outputs.push(hidden.clone());
        }

        // If reverse, reverse the outputs back to original time order
        if reverse {
            outputs.reverse();
        }

        // Stack outputs along time dimension
        // TODO: Implement Tensor::stack
        Err(
            tenflowers_core::error::TensorError::unsupported_operation_simple(
                "Tensor::stack not yet implemented".to_string(),
            ),
        )
    }
}

impl<T> Layer<T> for LSTM<T>
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
        let mut output = input.clone();

        for layer in 0..self.num_layers {
            if self.bidirectional {
                // Forward pass
                let forward_out = self.forward_direction(&output, layer, false)?;

                // Backward pass
                let backward_out = self.forward_direction(&output, layer, true)?;

                // Concatenate forward and backward outputs
                // TODO: Implement Tensor::concatenate
                return Err(
                    tenflowers_core::error::TensorError::unsupported_operation_simple(
                        "Tensor::concatenate not yet implemented".to_string(),
                    ),
                );
            } else {
                // Unidirectional forward pass
                output = self.forward_direction(&output, layer, false)?;
            }

            // Apply dropout between layers (except last layer)
            if self.training && layer < self.num_layers - 1 && self.dropout > 0.0 {
                output = self.apply_dropout(&output, self.dropout)?;
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        // Forward direction weights
        for w in &self.weight_ih {
            params.push(w);
        }
        for w in &self.weight_hh {
            params.push(w);
        }

        // Forward direction biases
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

        // Reverse direction weights (bidirectional)
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

        // Reverse direction biases (bidirectional)
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

        // Forward direction weights
        for w in &mut self.weight_ih {
            params.push(w);
        }
        for w in &mut self.weight_hh {
            params.push(w);
        }

        // Forward direction biases
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

        // Reverse direction weights (bidirectional)
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

        // Reverse direction biases (bidirectional)
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

impl<T> LSTM<T>
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
    /// Apply dropout to tensor during training
    fn apply_dropout(&self, input: &Tensor<T>, dropout_prob: f32) -> Result<Tensor<T>> {
        if !self.training || dropout_prob <= 0.0 {
            return Ok(input.clone());
        }

        // Simple dropout implementation - scale by 1/(1-p) during training
        let keep_prob = 1.0 - dropout_prob;
        let scale = T::from(1.0 / keep_prob as f64).unwrap();

        // For now, just apply scaling (full dropout implementation would need random mask)
        input.mul_scalar(scale)
    }
}
