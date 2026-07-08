//! Vanilla RNN layer implementation

use super::RnnNonlinearity;
use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
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
    nonlinearity: RnnNonlinearity,

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
            nonlinearity: self.nonlinearity,
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
        Self::new_with_nonlinearity(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            RnnNonlinearity::Tanh,
        )
    }

    /// Construct an RNN with an explicit recurrent nonlinearity.
    ///
    /// [`RNN::new`] delegates here with [`RnnNonlinearity::Tanh`], preserving the
    /// original 7-argument constructor for existing callers.
    pub fn new_with_nonlinearity(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
        nonlinearity: RnnNonlinearity,
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
            let scale = T::from(1.0 / (input_dim as f64).sqrt())
                .expect("Failed to convert scale to tensor type");

            // Input-to-hidden weights: [input_dim, hidden_size]
            let w_ih = Self::init_weight(&[input_dim, hidden_size], scale)?;
            weight_ih.push(w_ih);

            // Hidden-to-hidden weights: [hidden_size, hidden_size]
            let w_hh = Self::init_weight(&[hidden_size, hidden_size], scale)?;
            weight_hh.push(w_hh);

            if bias {
                bias_ih
                    .as_mut()
                    .expect("bias_ih should be Some when bias is true")
                    .push(Tensor::zeros(&[hidden_size]));
                bias_hh
                    .as_mut()
                    .expect("bias_hh should be Some when bias is true")
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
                    let scale = T::from(1.0 / (input_dim as f64).sqrt())
                        .expect("Failed to convert scale to tensor type");

                    w_ih_rev.push(Self::init_weight(&[input_dim, hidden_size], scale)?);
                    w_hh_rev.push(Self::init_weight(&[hidden_size, hidden_size], scale)?);

                    if bias {
                        b_ih_rev
                            .as_mut()
                            .expect("b_ih_rev should be Some when bias is true")
                            .push(Tensor::zeros(&[hidden_size]));
                        b_hh_rev
                            .as_mut()
                            .expect("b_hh_rev should be Some when bias is true")
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
            nonlinearity,
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
                T::from(random_val).expect("Failed to convert random value to tensor type") * scale
            })
            .collect();

        Tensor::from_data(values, shape)
    }

    /// Apply the configured recurrent nonlinearity to a pre-activation tensor.
    fn apply_nonlinearity(&self, pre_activation: &Tensor<T>) -> Result<Tensor<T>> {
        match self.nonlinearity {
            RnnNonlinearity::Tanh => tenflowers_core::ops::activation::tanh(pre_activation),
            RnnNonlinearity::Relu => tenflowers_core::ops::activation::relu(pre_activation),
        }
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
            // Reorder [seq_len, batch, input_size] -> [batch, seq_len, input_size].
            Self::swap_time_batch(input)?
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

        // Reorder back to [seq_len, batch, hidden] if the caller used time-major input.
        let output = if self.batch_first {
            output
        } else {
            Self::swap_time_batch(&output)?
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

            // Apply the configured nonlinearity (tanh or relu)
            let h_t = self.apply_nonlinearity(&combined)?;

            outputs.push(h_t.clone());
            h_prev = h_t;
        }

        // Stack outputs along sequence dimension
        let output = Self::stack_sequence_outputs(&outputs)?;

        // Handle bidirectional case
        if self.bidirectional {
            let w_ih_rev = self
                .weight_ih_reverse
                .as_ref()
                .expect("Reverse weight_ih not initialized for bidirectional RNN");
            let w_hh_rev = self
                .weight_hh_reverse
                .as_ref()
                .expect("Reverse weight_hh not initialized for bidirectional RNN");
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

                let h_t = self.apply_nonlinearity(&combined)?;

                outputs_rev.push(h_t.clone());
                h_prev_rev = h_t;
            }

            // Reverse the reverse outputs to match sequence order
            outputs_rev.reverse();
            let output_rev = Self::stack_sequence_outputs(&outputs_rev)?;

            // Concatenate forward and backward outputs along the feature axis
            // -> [batch, seq, 2 * hidden].
            let combined_output = Self::concatenate_tensors(&[&output, &output_rev], 2)?;
            // Stack the forward/backward final hidden states along a new direction axis
            // -> [2, batch, hidden] so they map onto consecutive direction slots.
            let combined_hidden = tenflowers_core::ops::stack(&[&h_prev, &h_prev_rev], 0)?;

            Ok((combined_output, combined_hidden))
        } else {
            Ok((output, h_prev.unsqueeze(&[0])?))
        }
    }

    // Helper functions
    fn extract_timestep(input: &Tensor<T>, timestep: usize) -> Result<Tensor<T>> {
        // Extract a specific timestep from [batch, seq, features] -> [batch, features]
        let input_shape = input.shape().dims();
        if input_shape.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "extract_timestep expects a 3D tensor [batch, seq, features], got {input_shape:?}"
            )));
        }
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        if timestep >= seq_len {
            return Err(TensorError::invalid_argument(format!(
                "extract_timestep index {timestep} is out of range for sequence length {seq_len}"
            )));
        }

        // Gather the [batch, features] slab for `timestep` along the time axis. Reading
        // through `to_vec` materialises the elements in logical row-major order so this
        // works regardless of the source tensor's memory layout, and `from_data` yields
        // a fresh contiguous tensor for the recurrent math that follows.
        let data = input.to_vec()?;
        let mut step = Vec::with_capacity(batch_size * features);
        for b in 0..batch_size {
            let base = (b * seq_len + timestep) * features;
            step.extend_from_slice(&data[base..base + features]);
        }
        Tensor::from_data(step, &[batch_size, features])
    }

    /// Reorder a 3D tensor by swapping the first two axes: `[a, b, c] -> [b, a, c]`.
    ///
    /// This produces a freshly-laid-out, contiguous tensor (unlike a lazy transpose
    /// view), which keeps downstream operations that rely on standard-layout access
    /// correct.
    fn swap_time_batch(input: &Tensor<T>) -> Result<Tensor<T>> {
        let dims = input.shape().dims();
        if dims.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "swap_time_batch expects a 3D tensor, got {dims:?}"
            )));
        }
        let (a, b, c) = (dims[0], dims[1], dims[2]);
        let data = input.to_vec()?;

        let mut reordered = vec![T::zero(); a * b * c];
        for i in 0..a {
            for j in 0..b {
                let src = (i * b + j) * c;
                let dst = (j * a + i) * c;
                reordered[dst..dst + c].copy_from_slice(&data[src..src + c]);
            }
        }

        Tensor::from_data(reordered, &[b, a, c])
    }

    fn stack_sequence_outputs(outputs: &[Tensor<T>]) -> Result<Tensor<T>> {
        // Stack a sequence of per-timestep [batch_size, hidden_size] tensors along a
        // new time axis, producing the real [batch_size, seq_len, hidden_size] output.
        if outputs.is_empty() {
            return Err(TensorError::invalid_argument(
                "Empty outputs sequence".to_string(),
            ));
        }

        let first = outputs[0].shape().dims();
        if first.len() != 2 {
            return Err(TensorError::invalid_argument(format!(
                "stack_sequence_outputs expects per-step [batch, hidden] tensors, got {first:?}"
            )));
        }
        let batch_size = first[0];
        let hidden_size = first[1];
        let seq_len = outputs.len();

        // Materialise every step in logical row-major order (layout independent).
        let mut steps: Vec<Vec<T>> = Vec::with_capacity(seq_len);
        for output in outputs {
            let dims = output.shape().dims();
            if dims.len() != 2 || dims[0] != batch_size || dims[1] != hidden_size {
                return Err(TensorError::invalid_shape_simple(format!(
                    "stack_sequence_outputs: inconsistent per-step shape {dims:?}, expected [{batch_size}, {hidden_size}]"
                )));
            }
            steps.push(output.to_vec()?);
        }

        // Interleave into [batch, seq, hidden] row-major order -> contiguous output.
        let mut result = Vec::with_capacity(batch_size * seq_len * hidden_size);
        for b in 0..batch_size {
            for step in steps.iter() {
                let base = b * hidden_size;
                result.extend_from_slice(&step[base..base + hidden_size]);
            }
        }

        Tensor::from_data(result, &[batch_size, seq_len, hidden_size])
    }

    fn extract_hidden_slice(hidden: &Tensor<T>, start: usize, end: usize) -> Result<Tensor<T>> {
        // Extract a single layer/direction slice from the hidden state tensor
        // [layers*dirs, batch, hidden] -> [batch, hidden].
        let shape = hidden.shape().dims();
        if shape.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "extract_hidden_slice expects a 3D hidden state [layers*dirs, batch, hidden], got {shape:?}"
            )));
        }
        if end != start + 1 {
            return Err(TensorError::invalid_argument(format!(
                "extract_hidden_slice only supports single-index slices (end == start + 1), got start={start}, end={end}"
            )));
        }
        if end > shape[0] {
            return Err(TensorError::invalid_argument(format!(
                "extract_hidden_slice range [{start}, {end}) is out of range for {} layer/direction states",
                shape[0]
            )));
        }
        let batch_size = shape[1];
        let hidden_size = shape[2];

        // Layer/direction `start` occupies the contiguous logical block
        // [start * batch * hidden, (start + 1) * batch * hidden) in [batch, hidden] order.
        let data = hidden.to_vec()?;
        let block = batch_size * hidden_size;
        let base = start * block;
        let slab = data[base..base + block].to_vec();
        Tensor::from_data(slab, &[batch_size, hidden_size])
    }

    fn update_hidden_slice(
        hidden: &Tensor<T>,
        new_slice: &Tensor<T>,
        start: usize,
        end: usize,
    ) -> Result<Tensor<T>> {
        // Write `new_slice` into the [start, end) layer/direction range of the hidden
        // state tensor [layers*dirs, batch, hidden], returning the updated tensor.
        // `new_slice` must have shape [end - start, batch, hidden].
        let shape = hidden.shape().dims();
        if shape.len() != 3 {
            return Err(TensorError::invalid_argument(format!(
                "update_hidden_slice expects a 3D hidden state, got {shape:?}"
            )));
        }
        let total = shape[0];
        let batch_size = shape[1];
        let hidden_size = shape[2];

        if start > end || end > total {
            return Err(TensorError::invalid_argument(format!(
                "update_hidden_slice range [{start}, {end}) is invalid for {total} layer/direction states"
            )));
        }

        let new_dims = new_slice.shape().dims();
        if new_dims.len() != 3
            || new_dims[0] != end - start
            || new_dims[1] != batch_size
            || new_dims[2] != hidden_size
        {
            return Err(TensorError::invalid_shape_simple(format!(
                "update_hidden_slice new slice shape {new_dims:?} does not match expected [{}, {batch_size}, {hidden_size}]",
                end - start
            )));
        }

        // The layer/direction axis is outermost, so range [start, end) maps to the
        // contiguous logical block [start * batch * hidden, end * batch * hidden).
        // Reassemble left | new_slice | right -> fresh contiguous tensor.
        let block = batch_size * hidden_size;
        let hidden_data = hidden.to_vec()?;
        let new_data = new_slice.to_vec()?;

        let mut result = Vec::with_capacity(total * block);
        result.extend_from_slice(&hidden_data[0..start * block]);
        result.extend_from_slice(&new_data);
        result.extend_from_slice(&hidden_data[end * block..total * block]);

        Tensor::from_data(result, &[total, batch_size, hidden_size])
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

        // Real concatenation along `axis` using the core manipulation op.
        tenflowers_core::ops::concat(tensors, axis)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_tensor(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.05).collect();
        Tensor::from_data(data, shape).expect("failed to build test tensor")
    }

    // Read tensor elements in logical row-major order, independent of memory layout.
    fn values(tensor: &Tensor<f32>) -> Vec<f32> {
        tensor.to_vec().expect("to_vec")
    }

    fn has_nonzero(tensor: &Tensor<f32>) -> bool {
        values(tensor).iter().any(|&v| v.abs() > 0.0)
    }

    #[test]
    fn extract_timestep_returns_real_slice() {
        // [batch = 2, seq = 3, feat = 2]
        let input = ramp_tensor(&[2, 3, 2]);
        let data = values(&input);

        let x1 = RNN::<f32>::extract_timestep(&input, 1).expect("extract_timestep");
        assert_eq!(x1.shape().dims(), &[2, 2]);

        let got = values(&x1);
        // batch 0, t = 1 -> flat indices [(0*3 + 1) * 2 ..] = data[2], data[3]
        assert!((got[0] - data[2]).abs() < 1e-6);
        assert!((got[1] - data[3]).abs() < 1e-6);
        // batch 1, t = 1 -> flat indices [(1*3 + 1) * 2 ..] = data[8], data[9]
        assert!((got[2] - data[8]).abs() < 1e-6);
        assert!((got[3] - data[9]).abs() < 1e-6);
        // A real slice of a ramp tensor cannot be all zeros.
        assert!(has_nonzero(&x1));
    }

    #[test]
    fn swap_time_batch_reorders_axes() {
        // [seq = 2, batch = 3, feat = 2] -> [3, 2, 2]
        let input = ramp_tensor(&[2, 3, 2]);
        let data = values(&input);

        let swapped = RNN::<f32>::swap_time_batch(&input).expect("swap_time_batch");
        assert_eq!(swapped.shape().dims(), &[3, 2, 2]);

        let got = values(&swapped);
        // source (i = 0, j = 1) -> dest (j = 1, i = 0):
        //   src flat (0*3 + 1) * 2 = data[2], data[3]
        //   dst flat (1*2 + 0) * 2 = got[4], got[5]
        assert!((got[4] - data[2]).abs() < 1e-6);
        assert!((got[5] - data[3]).abs() < 1e-6);
    }

    #[test]
    fn rnn_forward_produces_nonzero_output_with_correct_shape() {
        let rnn = RNN::<f32>::new(3, 4, 1, true, true, 0.0, false).expect("rnn");
        let input = ramp_tensor(&[2, 5, 3]);
        let output = rnn.forward(&input).expect("forward");
        assert_eq!(output.shape().dims(), &[2, 5, 4]);
        assert!(has_nonzero(&output), "RNN output must not be all zeros");
    }

    #[test]
    fn rnn_forward_with_hidden_propagates_state() {
        let rnn = RNN::<f32>::new(3, 4, 1, true, true, 0.0, false).expect("rnn");
        let input = ramp_tensor(&[2, 5, 3]);
        let (output, hidden) = rnn.forward_with_hidden(&input, None).expect("forward");
        assert_eq!(output.shape().dims(), &[2, 5, 4]);
        assert_eq!(hidden.shape().dims(), &[1, 2, 4]);
        assert!(has_nonzero(&output));
        assert!(has_nonzero(&hidden));

        // The returned final hidden state must equal the last timestep of the output,
        // proving the per-timestep states were really assembled into the output.
        let last = RNN::<f32>::extract_timestep(&output, 4).expect("last step"); // [2, 4]
        let hidden2d = hidden.squeeze(Some(&[0])).expect("squeeze"); // [2, 4]
        let a = values(&last);
        let b = values(&hidden2d);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "final hidden must match last output step"
            );
        }
    }

    #[test]
    fn rnn_multilayer_forward_nonzero() {
        let rnn = RNN::<f32>::new(3, 4, 2, true, true, 0.0, false).expect("rnn");
        let input = ramp_tensor(&[2, 5, 3]);
        let (output, hidden) = rnn.forward_with_hidden(&input, None).expect("forward");
        assert_eq!(output.shape().dims(), &[2, 5, 4]);
        assert_eq!(hidden.shape().dims(), &[2, 2, 4]);
        assert!(has_nonzero(&output));
        assert!(has_nonzero(&hidden));
    }

    #[test]
    fn rnn_bidirectional_forward_shape_and_nonzero() {
        let rnn = RNN::<f32>::new(3, 4, 1, true, true, 0.0, true).expect("rnn");
        let input = ramp_tensor(&[2, 5, 3]);
        let (output, hidden) = rnn.forward_with_hidden(&input, None).expect("forward");
        // Bidirectional doubles the hidden feature dimension in the output.
        assert_eq!(output.shape().dims(), &[2, 5, 8]);
        assert_eq!(hidden.shape().dims(), &[2, 2, 4]);
        assert!(has_nonzero(&output));
        assert!(has_nonzero(&hidden));
    }

    #[test]
    fn rnn_time_major_forward_shape_and_nonzero() {
        // batch_first = false path must also yield a real, correctly shaped result.
        let rnn = RNN::<f32>::new(3, 4, 1, true, false, 0.0, false).expect("rnn");
        // time-major input: [seq = 5, batch = 2, input = 3]
        let input = ramp_tensor(&[5, 2, 3]);
        let output = rnn.forward(&input).expect("forward");
        assert_eq!(output.shape().dims(), &[5, 2, 4]);
        assert!(has_nonzero(&output));
    }

    /// Hand-set weights so the single-step pre-activation is `[-1.0, 1.5]` and
    /// prove the nonlinearity branch is real: ReLU zeroes the negative entry
    /// exactly, whereas Tanh (same weights) yields a strictly negative entry.
    fn set_single_step_weights(rnn: &mut RNN<f32>) {
        let mut params = rnn.parameters_mut();
        // Order for 1 layer with bias: [w_ih, w_hh, b_ih, b_hh].
        *params[0] = Tensor::from_data(vec![1.0, 1.0], &[1, 2]).expect("w_ih");
        *params[1] = Tensor::from_data(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).expect("w_hh");
        *params[2] = Tensor::from_data(vec![-2.0, 0.5], &[2]).expect("b_ih");
        *params[3] = Tensor::from_data(vec![0.0, 0.0], &[2]).expect("b_hh");
    }

    #[test]
    fn rnn_relu_nonlinearity_zeroes_negatives() {
        // input_size = 1, hidden_size = 2, single layer, single timestep.
        let mut relu_rnn = RNN::<f32>::new_with_nonlinearity(
            1,
            2,
            1,
            true,
            true,
            0.0,
            false,
            RnnNonlinearity::Relu,
        )
        .expect("relu rnn");
        set_single_step_weights(&mut relu_rnn);

        // batch_first input: [batch = 1, seq = 1, feat = 1]; h_0 defaults to zeros.
        let input = Tensor::from_data(vec![1.0], &[1, 1, 1]).expect("input");
        let relu_out = relu_rnn.forward(&input).expect("relu forward");
        let relu_vals = values(&relu_out);
        assert_eq!(relu_vals.len(), 2);
        // relu([-1.0, 1.5]) = [0.0, 1.5].
        assert!(
            (relu_vals[0] - 0.0).abs() < 1e-6,
            "relu must zero the negative pre-activation, got {}",
            relu_vals[0]
        );
        assert!(
            (relu_vals[1] - 1.5).abs() < 1e-5,
            "relu must pass the positive pre-activation, got {}",
            relu_vals[1]
        );
        assert!(
            relu_vals.iter().all(|&v| v >= 0.0),
            "relu output must be non-negative"
        );

        // Same weights with Tanh must produce a strictly negative first entry,
        // proving the branch genuinely changes the computation.
        let mut tanh_rnn = RNN::<f32>::new_with_nonlinearity(
            1,
            2,
            1,
            true,
            true,
            0.0,
            false,
            RnnNonlinearity::Tanh,
        )
        .expect("tanh rnn");
        set_single_step_weights(&mut tanh_rnn);
        let tanh_out = tanh_rnn.forward(&input).expect("tanh forward");
        let tanh_vals = values(&tanh_out);
        // tanh(-1.0) ~= -0.7616 < 0.
        assert!(
            tanh_vals[0] < 0.0,
            "tanh must produce a negative entry where relu produced 0, got {}",
            tanh_vals[0]
        );
    }
}
