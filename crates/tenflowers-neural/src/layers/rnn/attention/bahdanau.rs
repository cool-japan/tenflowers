//! Bahdanau Attention mechanism
//!
//! Implements the additive attention mechanism from:
//! "Neural Machine Translation by Jointly Learning to Align and Translate"
//! https://arxiv.org/abs/1409.0473
//!
//! This is also known as additive attention or content-based attention.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Random;
use tenflowers_core::{Result, Tensor, TensorError};

/// Bahdanau Attention mechanism
///
/// Computes attention using an additive mechanism with a feedforward network.
/// The attention score is computed as: score = v^T * tanh(W_a * h + U_a * s)
/// where h is the encoder hidden state, s is the decoder hidden state.
#[derive(Debug)]
pub struct BahdanauAttention<T>
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
    /// Size of the encoder hidden states
    encoder_hidden_size: usize,
    /// Size of the decoder hidden states
    decoder_hidden_size: usize,
    /// Size of the attention hidden layer
    attention_size: usize,

    /// Weight matrix for encoder hidden states: [encoder_hidden_size, attention_size]
    w_encoder: Tensor<T>,
    /// Weight matrix for decoder hidden states: [decoder_hidden_size, attention_size]
    w_decoder: Tensor<T>,
    /// Attention vector: [attention_size]
    v_attention: Tensor<T>,

    /// Optional bias terms
    bias_encoder: Option<Tensor<T>>,
    bias_decoder: Option<Tensor<T>>,
    bias_attention: Option<Tensor<T>>,

    /// Whether to use bias terms
    use_bias: bool,

    training: bool,
}

impl<
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
    > Clone for BahdanauAttention<T>
{
    fn clone(&self) -> Self {
        Self {
            encoder_hidden_size: self.encoder_hidden_size,
            decoder_hidden_size: self.decoder_hidden_size,
            attention_size: self.attention_size,
            w_encoder: self.w_encoder.clone(),
            w_decoder: self.w_decoder.clone(),
            v_attention: self.v_attention.clone(),
            bias_encoder: self.bias_encoder.clone(),
            bias_decoder: self.bias_decoder.clone(),
            bias_attention: self.bias_attention.clone(),
            use_bias: self.use_bias,
            training: self.training,
        }
    }
}

impl<T> BahdanauAttention<T>
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
    /// Create a new Bahdanau attention mechanism
    ///
    /// # Arguments
    /// * `encoder_hidden_size` - Size of encoder hidden states
    /// * `decoder_hidden_size` - Size of decoder hidden states
    /// * `attention_size` - Size of the attention hidden layer
    /// * `use_bias` - Whether to use bias terms
    pub fn new(
        encoder_hidden_size: usize,
        decoder_hidden_size: usize,
        attention_size: usize,
        use_bias: bool,
    ) -> Result<Self> {
        if encoder_hidden_size == 0 {
            return Err(TensorError::invalid_argument(
                "encoder_hidden_size must be > 0".to_string(),
            ));
        }
        if decoder_hidden_size == 0 {
            return Err(TensorError::invalid_argument(
                "decoder_hidden_size must be > 0".to_string(),
            ));
        }
        if attention_size == 0 {
            return Err(TensorError::invalid_argument(
                "attention_size must be > 0".to_string(),
            ));
        }

        // Initialize weights using Xavier/Glorot initialization
        let encoder_scale = T::from(1.0 / (encoder_hidden_size as f64).sqrt()).unwrap();
        let decoder_scale = T::from(1.0 / (decoder_hidden_size as f64).sqrt()).unwrap();
        let attention_scale = T::from(1.0 / (attention_size as f64).sqrt()).unwrap();

        let w_encoder = Self::init_weight(&[encoder_hidden_size, attention_size], encoder_scale)?;
        let w_decoder = Self::init_weight(&[decoder_hidden_size, attention_size], decoder_scale)?;
        let v_attention = Self::init_weight(&[attention_size], attention_scale)?;

        // Initialize bias terms if requested
        let (bias_encoder, bias_decoder, bias_attention) = if use_bias {
            (
                Some(Tensor::zeros(&[attention_size])),
                Some(Tensor::zeros(&[attention_size])),
                Some(Tensor::zeros(&[1])),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            encoder_hidden_size,
            decoder_hidden_size,
            attention_size,
            w_encoder,
            w_decoder,
            v_attention,
            bias_encoder,
            bias_decoder,
            bias_attention,
            use_bias,
            training: true,
        })
    }

    /// Simple constructor with default attention size
    pub fn new_simple(encoder_hidden_size: usize, decoder_hidden_size: usize) -> Result<Self> {
        // Use attention size as the maximum of encoder and decoder sizes
        let attention_size = std::cmp::max(encoder_hidden_size, decoder_hidden_size);
        Self::new(
            encoder_hidden_size,
            decoder_hidden_size,
            attention_size,
            true,
        )
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

    /// Compute attention scores and context vector
    ///
    /// # Arguments
    /// * `encoder_outputs` - Encoder hidden states [seq_len, batch_size, encoder_hidden_size]
    /// * `decoder_hidden` - Current decoder hidden state [batch_size, decoder_hidden_size]
    ///
    /// # Returns
    /// * Tuple of (context_vector, attention_weights)
    /// * context_vector: [batch_size, encoder_hidden_size]
    /// * attention_weights: [batch_size, seq_len]
    pub fn forward_with_weights(
        &self,
        encoder_outputs: &Tensor<T>,
        decoder_hidden: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let encoder_shape = encoder_outputs.shape().dims();
        let decoder_shape = decoder_hidden.shape().dims();

        let seq_len = encoder_shape[0];
        let batch_size = encoder_shape[1];
        let encoder_dim = encoder_shape[2];
        let decoder_dim = decoder_shape[1];

        // Validate input dimensions
        if encoder_dim != self.encoder_hidden_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected encoder hidden size {}, got {}",
                self.encoder_hidden_size, encoder_dim
            )));
        }
        if decoder_dim != self.decoder_hidden_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected decoder hidden size {}, got {}",
                self.decoder_hidden_size, decoder_dim
            )));
        }

        // Reshape encoder outputs to [seq_len * batch_size, encoder_hidden_size]
        let encoder_flat = encoder_outputs.reshape(&[seq_len * batch_size, encoder_dim])?;

        // Project encoder outputs: [seq_len * batch_size, attention_size]
        let mut encoder_projected = tenflowers_core::ops::matmul(&encoder_flat, &self.w_encoder)?;
        if let Some(ref bias) = self.bias_encoder {
            encoder_projected = tenflowers_core::ops::add(&encoder_projected, bias)?;
        }

        // Reshape back: [seq_len, batch_size, attention_size]
        let encoder_projected =
            encoder_projected.reshape(&[seq_len, batch_size, self.attention_size])?;

        // Project decoder hidden state: [batch_size, attention_size]
        let mut decoder_projected = tenflowers_core::ops::matmul(decoder_hidden, &self.w_decoder)?;
        if let Some(ref bias) = self.bias_decoder {
            decoder_projected = tenflowers_core::ops::add(&decoder_projected, bias)?;
        }

        // Broadcast decoder projection to match encoder: [seq_len, batch_size, attention_size]
        let decoder_broadcast = Self::broadcast_decoder(&decoder_projected, seq_len)?;

        // Add encoder and decoder projections
        let combined = tenflowers_core::ops::add(&encoder_projected, &decoder_broadcast)?;

        // Apply tanh activation
        let activated = tenflowers_core::ops::activation::tanh(&combined)?;

        // Compute attention scores by multiplying with v_attention
        // Reshape to [seq_len * batch_size, attention_size] for matrix multiplication
        let activated_flat = activated.reshape(&[seq_len * batch_size, self.attention_size])?;

        // Expand v_attention to [attention_size, 1] for proper matmul
        let v_expanded = self.v_attention.unsqueeze(&[1])?;
        let mut scores_flat = tenflowers_core::ops::matmul(&activated_flat, &v_expanded)?;

        if let Some(ref bias) = self.bias_attention {
            scores_flat = tenflowers_core::ops::add(&scores_flat, bias)?;
        }

        // Reshape scores back to [seq_len, batch_size]
        let scores = scores_flat.reshape(&[seq_len, batch_size])?;

        // Apply softmax to get attention weights: [seq_len, batch_size]
        let attention_weights = Self::softmax_over_sequence(&scores)?;

        // Compute context vector by weighted sum of encoder outputs
        let context = Self::compute_context(encoder_outputs, &attention_weights)?;

        // Transpose attention weights to [batch_size, seq_len] for standard format
        let attention_weights_transposed = attention_weights.transpose()?;

        Ok((context, attention_weights_transposed))
    }

    /// Helper function to broadcast decoder projection across sequence length
    fn broadcast_decoder(decoder_projected: &Tensor<T>, seq_len: usize) -> Result<Tensor<T>> {
        // Expand decoder projection to match sequence dimension
        // [batch_size, attention_size] -> [seq_len, batch_size, attention_size]
        let shape = decoder_projected.shape().dims();
        let batch_size = shape[0];
        let attention_size = shape[1];

        // Get the data from decoder_projected
        let decoder_data = decoder_projected.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access decoder data".to_string())
        })?;

        // Repeat the data across the sequence dimension
        let total_elements = seq_len * batch_size * attention_size;
        let mut result_data = Vec::with_capacity(total_elements);

        for _ in 0..seq_len {
            result_data.extend_from_slice(decoder_data);
        }

        Tensor::from_data(result_data, &[seq_len, batch_size, attention_size])
    }

    /// Apply softmax over the sequence dimension
    fn softmax_over_sequence(scores: &Tensor<T>) -> Result<Tensor<T>> {
        // Apply softmax over the first dimension (sequence length)
        // scores: [seq_len, batch_size]
        let shape = scores.shape().dims();
        let seq_len = shape[0];
        let batch_size = shape[1];

        let scores_data = scores.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access scores data".to_string())
        })?;

        let mut result_data = vec![T::zero(); seq_len * batch_size];

        // Apply softmax for each batch
        for b in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = T::neg_infinity();
            for s in 0..seq_len {
                let idx = s * batch_size + b;
                if scores_data[idx] > max_val {
                    max_val = scores_data[idx];
                }
            }

            // Compute exp and sum
            let mut sum = T::zero();
            for s in 0..seq_len {
                let idx = s * batch_size + b;
                let exp_val = (scores_data[idx] - max_val).exp();
                result_data[idx] = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for s in 0..seq_len {
                let idx = s * batch_size + b;
                result_data[idx] = result_data[idx] / sum;
            }
        }

        Tensor::from_data(result_data, &[seq_len, batch_size])
    }

    /// Compute context vector as weighted sum of encoder outputs
    fn compute_context(
        encoder_outputs: &Tensor<T>,
        attention_weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // encoder_outputs: [seq_len, batch_size, encoder_hidden_size]
        // attention_weights: [seq_len, batch_size]
        // output: [batch_size, encoder_hidden_size]

        let encoder_shape = encoder_outputs.shape().dims();
        let seq_len = encoder_shape[0];
        let batch_size = encoder_shape[1];
        let encoder_dim = encoder_shape[2];

        let encoder_data = encoder_outputs.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access encoder data".to_string())
        })?;

        let weights_data = attention_weights.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access weights data".to_string())
        })?;

        let mut context_data = vec![T::zero(); batch_size * encoder_dim];

        // Compute weighted sum for each batch
        for b in 0..batch_size {
            for d in 0..encoder_dim {
                let mut sum = T::zero();
                for s in 0..seq_len {
                    let encoder_idx = s * batch_size * encoder_dim + b * encoder_dim + d;
                    let weight_idx = s * batch_size + b;
                    sum = sum + encoder_data[encoder_idx] * weights_data[weight_idx];
                }
                context_data[b * encoder_dim + d] = sum;
            }
        }

        Tensor::from_data(context_data, &[batch_size, encoder_dim])
    }
}

impl<T> Layer<T> for BahdanauAttention<T>
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
    /// Standard forward pass that takes encoder and decoder states
    /// Input tensor should be concatenated [encoder_outputs, decoder_hidden]
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        // This is a simplified interface for the Layer trait
        // In practice, attention layers usually need separate encoder/decoder inputs

        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];

        // Return zeros for now as placeholder
        // In real usage, would split input into encoder and decoder parts
        Ok(Tensor::zeros(&[batch_size, self.encoder_hidden_size]))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = vec![&self.w_encoder, &self.w_decoder, &self.v_attention];

        if let Some(ref bias) = self.bias_encoder {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_decoder {
            params.push(bias);
        }
        if let Some(ref bias) = self.bias_attention {
            params.push(bias);
        }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = vec![
            &mut self.w_encoder,
            &mut self.w_decoder,
            &mut self.v_attention,
        ];

        if let Some(ref mut bias) = self.bias_encoder {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_decoder {
            params.push(bias);
        }
        if let Some(ref mut bias) = self.bias_attention {
            params.push(bias);
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
