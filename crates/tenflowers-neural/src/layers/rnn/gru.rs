//! GRU (Gated Recurrent Unit) layer implementation

use super::ResetGateVariation;
use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
// use scirs2_core::random::{rng, Random}; // Unused for now
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use tenflowers_core::{device::context::get_gpu_context, gpu::rnn_ops::GpuRnnOps};

/// GRU (Gated Recurrent Unit) layer
#[derive(Debug)]
pub struct GRU<T>
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
    reset_variation: ResetGateVariation,

    // GRU parameters for each layer
    weight_ih: Vec<Tensor<T>>, // Input-to-hidden weights [input_size, 3*hidden_size]
    weight_hh: Vec<Tensor<T>>, // Hidden-to-hidden weights [hidden_size, 3*hidden_size]
    bias_ih: Option<Vec<Tensor<T>>>, // Input-to-hidden bias [3*hidden_size]
    bias_hh: Option<Vec<Tensor<T>>>, // Hidden-to-hidden bias [3*hidden_size]

    // Additional weights for Coupled variation (z_t -> r_t connection)
    weight_zr: Option<Vec<Tensor<T>>>, // Update-to-reset gate weights [hidden_size]

    // For bidirectional GRU
    weight_ih_reverse: Option<Vec<Tensor<T>>>,
    weight_hh_reverse: Option<Vec<Tensor<T>>>,
    bias_ih_reverse: Option<Vec<Tensor<T>>>,
    bias_hh_reverse: Option<Vec<Tensor<T>>>,
    weight_zr_reverse: Option<Vec<Tensor<T>>>,

    training: bool,
}

impl<T> Clone for GRU<T>
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
            reset_variation: self.reset_variation,
            weight_ih: self.weight_ih.clone(),
            weight_hh: self.weight_hh.clone(),
            bias_ih: self.bias_ih.clone(),
            bias_hh: self.bias_hh.clone(),
            weight_zr: self.weight_zr.clone(),
            weight_ih_reverse: self.weight_ih_reverse.clone(),
            weight_hh_reverse: self.weight_hh_reverse.clone(),
            bias_ih_reverse: self.bias_ih_reverse.clone(),
            bias_hh_reverse: self.bias_hh_reverse.clone(),
            weight_zr_reverse: self.weight_zr_reverse.clone(),
            training: self.training,
        }
    }
}

impl<T> GRU<T>
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
        Self::new_with_variation(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            ResetGateVariation::Standard,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_variation(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
        reset_variation: ResetGateVariation,
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
        let mut weight_zr = if reset_variation == ResetGateVariation::Coupled {
            Some(Vec::new())
        } else {
            None
        };

        // Initialize parameters for each layer
        for i in 0..num_layers {
            let input_dim = if i == 0 {
                input_size
            } else {
                hidden_size * if bidirectional { 2 } else { 1 }
            };

            // Xavier/Glorot initialization
            let scale = T::from(1.0 / (input_dim as f64).sqrt()).unwrap();

            // Input-to-hidden weights: [input_dim, 3*hidden_size] (reset, update, new gates)
            let w_ih = Self::init_weight(&[input_dim, 3 * hidden_size], scale)?;
            weight_ih.push(w_ih);

            // Hidden-to-hidden weights: [hidden_size, 3*hidden_size]
            let w_hh = Self::init_weight(&[hidden_size, 3 * hidden_size], scale)?;
            weight_hh.push(w_hh);

            if bias {
                bias_ih
                    .as_mut()
                    .unwrap()
                    .push(Tensor::zeros(&[3 * hidden_size]));
                bias_hh
                    .as_mut()
                    .unwrap()
                    .push(Tensor::zeros(&[3 * hidden_size]));
            }

            // Initialize weight_zr for Coupled variation
            if reset_variation == ResetGateVariation::Coupled {
                let zr_scale = T::from(1.0 / (hidden_size as f64).sqrt()).unwrap();
                weight_zr
                    .as_mut()
                    .unwrap()
                    .push(Self::init_weight(&[hidden_size], zr_scale)?);
            }
        }

        // Initialize bidirectional parameters if needed
        let (
            weight_ih_reverse,
            weight_hh_reverse,
            bias_ih_reverse,
            bias_hh_reverse,
            weight_zr_reverse,
        ) = if bidirectional {
            let mut w_ih_rev = Vec::new();
            let mut w_hh_rev = Vec::new();
            let mut b_ih_rev = if bias { Some(Vec::new()) } else { None };
            let mut b_hh_rev = if bias { Some(Vec::new()) } else { None };
            let mut w_zr_rev = if reset_variation == ResetGateVariation::Coupled {
                Some(Vec::new())
            } else {
                None
            };

            for i in 0..num_layers {
                let input_dim = if i == 0 { input_size } else { hidden_size * 2 };
                let scale = T::from(1.0 / (input_dim as f64).sqrt()).unwrap();

                w_ih_rev.push(Self::init_weight(&[input_dim, 3 * hidden_size], scale)?);
                w_hh_rev.push(Self::init_weight(&[hidden_size, 3 * hidden_size], scale)?);

                if bias {
                    b_ih_rev
                        .as_mut()
                        .unwrap()
                        .push(Tensor::zeros(&[3 * hidden_size]));
                    b_hh_rev
                        .as_mut()
                        .unwrap()
                        .push(Tensor::zeros(&[3 * hidden_size]));
                }

                if reset_variation == ResetGateVariation::Coupled {
                    let zr_scale = T::from(1.0 / (hidden_size as f64).sqrt()).unwrap();
                    w_zr_rev
                        .as_mut()
                        .unwrap()
                        .push(Self::init_weight(&[hidden_size], zr_scale)?);
                }
            }

            (Some(w_ih_rev), Some(w_hh_rev), b_ih_rev, b_hh_rev, w_zr_rev)
        } else {
            (None, None, None, None, None)
        };

        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            reset_variation,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            weight_zr,
            weight_ih_reverse,
            weight_hh_reverse,
            bias_ih_reverse,
            bias_hh_reverse,
            weight_zr_reverse,
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

    /// GRU cell computation for single timestep
    #[allow(clippy::too_many_arguments)]
    fn gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
        weight_zr: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        match self.reset_variation {
            ResetGateVariation::Standard => {
                self.standard_gru_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh)
            }
            ResetGateVariation::Coupled => self.coupled_gru_cell(
                input, hidden, weight_ih, weight_hh, bias_ih, bias_hh, weight_zr,
            ),
            ResetGateVariation::Minimal => {
                self.minimal_gru_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh)
            }
            ResetGateVariation::Light => {
                self.light_gru_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh)
            }
            ResetGateVariation::ResetAfter => {
                self.reset_after_gru_cell(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh)
            }
        }
    }

    /// Standard GRU cell implementation
    fn standard_gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
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

        let gate_size = self.hidden_size;
        let batch_size = gi.shape().dims()[0];

        // Split gates: [reset, update, new]
        let gi_r = gi.slice(&[0..batch_size, 0..gate_size])?; // Reset gate (input)
        let gi_z = gi.slice(&[0..batch_size, gate_size..2 * gate_size])?; // Update gate (input)
        let gi_n = gi.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?; // New gate (input)

        let gh_r = gh.slice(&[0..batch_size, 0..gate_size])?; // Reset gate (hidden)
        let gh_z = gh.slice(&[0..batch_size, gate_size..2 * gate_size])?; // Update gate (hidden)
        let gh_n = gh.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?; // New gate (hidden)

        // Compute gates
        let r = gi_r.add(&gh_r)?.sigmoid()?; // Reset gate: r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1})
        let z = gi_z.add(&gh_z)?.sigmoid()?; // Update gate: z_t = sigmoid(W_iz * x_t + W_hz * h_{t-1})

        // New gate with reset applied to hidden state
        let n = gi_n.add(&r.mul(&gh_n)?)?.tanh()?; // n_t = tanh(W_in * x_t + W_hn * (r_t * h_{t-1}))

        // Update hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        let one_minus_z = z.neg()?.add(&Tensor::from_scalar(T::one()))?;
        let new_hidden = one_minus_z.mul(&n)?.add(&z.mul(hidden)?)?;

        Ok(new_hidden)
    }

    /// Coupled GRU cell implementation
    #[allow(clippy::too_many_arguments)]
    fn coupled_gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
        weight_zr: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let gi = input.matmul(weight_ih)?;
        let gh = hidden.matmul(weight_hh)?;

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

        let gate_size = self.hidden_size;
        let batch_size = gi.shape().dims()[0];
        let gi_r = gi.slice(&[0..batch_size, 0..gate_size])?;
        let gi_z = gi.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gi_n = gi.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        let gh_r = gh.slice(&[0..batch_size, 0..gate_size])?;
        let gh_z = gh.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gh_n = gh.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        // Update gate
        let z = gi_z.add(&gh_z)?.sigmoid()?;

        // Coupled reset gate: r_t = sigmoid(W_ir * x_t + W_hr * h_{t-1} + W_zr * z_t)
        let mut r_input = gi_r.add(&gh_r)?;
        if let Some(w_zr) = weight_zr {
            let z_contribution = z.matmul(w_zr)?;
            r_input = r_input.add(&z_contribution)?;
        }
        let r = r_input.sigmoid()?;

        let n = gi_n.add(&r.mul(&gh_n)?)?.tanh()?;

        let one_minus_z = z.neg()?.add(&Tensor::from_scalar(T::one()))?;
        let new_hidden = one_minus_z.mul(&n)?.add(&z.mul(hidden)?)?;

        Ok(new_hidden)
    }

    /// Minimal GRU cell implementation
    fn minimal_gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let gi = input.matmul(weight_ih)?;
        let gh = hidden.matmul(weight_hh)?;

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

        let gate_size = self.hidden_size;
        let batch_size = gi.shape().dims()[0];
        // Use first gate as forget gate, second as new gate
        let gi_f = gi.slice(&[0..batch_size, 0..gate_size])?; // Forget gate
        let gi_n = gi.slice(&[0..batch_size, gate_size..2 * gate_size])?; // New gate

        let gh_f = gh.slice(&[0..batch_size, 0..gate_size])?;
        let gh_n = gh.slice(&[0..batch_size, gate_size..2 * gate_size])?;

        let f = gi_f.add(&gh_f)?.sigmoid()?; // Forget gate
        let n = gi_n.add(&gh_n)?.tanh()?; // New gate (no reset applied)

        // h_t = f_t * h_{t-1} + (1 - f_t) * n_t
        let one_minus_f = f.neg()?.add(&Tensor::from_scalar(T::one()))?;
        let new_hidden = f.mul(hidden)?.add(&one_minus_f.mul(&n)?)?;

        Ok(new_hidden)
    }

    /// Light GRU cell implementation
    fn light_gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let gi = input.matmul(weight_ih)?;
        let gh = hidden.matmul(weight_hh)?;

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

        let gate_size = self.hidden_size;
        let batch_size = gi.shape().dims()[0];
        let gi_r = gi.slice(&[0..batch_size, 0..gate_size])?;
        let gi_z = gi.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gi_n = gi.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        let gh_z = gh.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gh_n = gh.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        // Light reset gate: no hidden state dependency
        let r = gi_r.sigmoid()?;
        let z = gi_z.add(&gh_z)?.sigmoid()?;
        let n = gi_n.add(&r.mul(&gh_n)?)?.tanh()?;

        let one_minus_z = z.neg()?.add(&Tensor::from_scalar(T::one()))?;
        let new_hidden = one_minus_z.mul(&n)?.add(&z.mul(hidden)?)?;

        Ok(new_hidden)
    }

    /// Reset After GRU cell implementation
    fn reset_after_gru_cell(
        &self,
        input: &Tensor<T>,
        hidden: &Tensor<T>,
        weight_ih: &Tensor<T>,
        weight_hh: &Tensor<T>,
        bias_ih: Option<&Tensor<T>>,
        bias_hh: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let gi = input.matmul(weight_ih)?;
        let gh = hidden.matmul(weight_hh)?;

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

        let gate_size = self.hidden_size;
        let batch_size = gi.shape().dims()[0];
        let gi_r = gi.slice(&[0..batch_size, 0..gate_size])?;
        let gi_z = gi.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gi_n = gi.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        let gh_r = gh.slice(&[0..batch_size, 0..gate_size])?;
        let gh_z = gh.slice(&[0..batch_size, gate_size..2 * gate_size])?;
        let gh_n = gh.slice(&[0..batch_size, 2 * gate_size..3 * gate_size])?;

        let r = gi_r.add(&gh_r)?.sigmoid()?;
        let z = gi_z.add(&gh_z)?.sigmoid()?;

        // Reset after: apply reset to the output of linear transformation
        let n = gi_n.add(&r.mul(&gh_n)?)?.tanh()?;

        let one_minus_z = z.neg()?.add(&Tensor::from_scalar(T::one()))?;
        let new_hidden = one_minus_z.mul(&n)?.add(&z.mul(hidden)?)?;

        Ok(new_hidden)
    }

    /// Process sequence in forward or reverse direction
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
        let weights_zr = if reverse {
            self.weight_zr_reverse.as_ref()
        } else {
            self.weight_zr.as_ref()
        };

        // Initialize hidden state
        let mut hidden = Tensor::zeros(&[batch_size, self.hidden_size]);

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

            // GRU cell computation
            let new_hidden = self.gru_cell(
                &x_t,
                &hidden,
                &weights_ih[layer],
                &weights_hh[layer],
                biases_ih.map(|b| &b[layer]),
                biases_hh.map(|b| &b[layer]),
                weights_zr.map(|w| &w[layer]),
            )?;

            hidden = new_hidden;
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

impl<T> Layer<T> for GRU<T>
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

        // weight_zr parameters for Coupled variation
        if let Some(ref weight_zr) = self.weight_zr {
            for w in weight_zr {
                params.push(w);
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
        if let Some(ref weight_zr_reverse) = self.weight_zr_reverse {
            for w in weight_zr_reverse {
                params.push(w);
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

        // weight_zr parameters for Coupled variation
        if let Some(ref mut weight_zr) = self.weight_zr {
            for w in weight_zr {
                params.push(w);
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
        if let Some(ref mut weight_zr_reverse) = self.weight_zr_reverse {
            for w in weight_zr_reverse {
                params.push(w);
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

impl<T> GRU<T>
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
