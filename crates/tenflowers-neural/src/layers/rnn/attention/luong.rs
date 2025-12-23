//! Luong Attention mechanism
//!
//! Implements the multiplicative attention mechanism from:
//! "Effective Approaches to Attention-based Neural Machine Translation"
//! https://arxiv.org/abs/1508.04025
//!
//! This mechanism computes attention scores using three variants:
//! - Dot: score = h_t^T * h_s
//! - General: score = h_t^T * W_a * h_s
//! - Concat: score = v_a^T * tanh(W_a * [h_t; h_s])

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, FromPrimitive, One, Zero};
use scirs2_core::random::Random;
use tenflowers_core::{Result, Tensor, TensorError};

/// Luong Attention Type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LuongAttentionType {
    /// Dot-product attention: score = h_t^T * h_s
    Dot,
    /// General/Multiplicative attention: score = h_t^T * W_a * h_s
    General,
    /// Concat/Additive attention: score = v_a^T * tanh(W_a * [h_t; h_s])
    Concat,
}

/// Luong Attention mechanism
///
/// Computes attention using multiplicative (Luong) attention.
/// Unlike Bahdanau attention which uses additive scoring, Luong attention
/// uses the current decoder hidden state directly without a feedforward layer.
#[derive(Debug)]
pub struct LuongAttention<T>
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
    /// Size of the hidden states
    hidden_size: usize,
    /// Type of attention scoring mechanism
    attention_type: LuongAttentionType,

    /// Weight matrix for general attention: [hidden_size, hidden_size]
    w_general: Option<Tensor<T>>,
    /// Weight matrix for concat attention: [hidden_size * 2, hidden_size]
    w_concat: Option<Tensor<T>>,
    /// Attention vector for concat attention: [hidden_size]
    v_concat: Option<Tensor<T>>,

    /// Whether to use bias terms
    use_bias: bool,
    /// Bias for general/concat attention
    bias: Option<Tensor<T>>,

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
    > Clone for LuongAttention<T>
{
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            attention_type: self.attention_type,
            w_general: self.w_general.clone(),
            w_concat: self.w_concat.clone(),
            v_concat: self.v_concat.clone(),
            use_bias: self.use_bias,
            bias: self.bias.clone(),
            training: self.training,
        }
    }
}

impl<T> LuongAttention<T>
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
    /// Create a new Luong attention mechanism
    ///
    /// # Arguments
    /// * `hidden_size` - Size of hidden states
    /// * `attention_type` - Type of attention (Dot, General, or Concat)
    pub fn new(hidden_size: usize, attention_type: LuongAttentionType) -> Result<Self> {
        Self::new_with_bias(hidden_size, attention_type, true)
    }

    /// Create Luong attention with optional bias
    pub fn new_with_bias(
        hidden_size: usize,
        attention_type: LuongAttentionType,
        use_bias: bool,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(TensorError::invalid_argument(
                "hidden_size must be > 0".to_string(),
            ));
        }

        let scale = T::from(1.0 / (hidden_size as f64).sqrt()).unwrap();

        let (w_general, w_concat, v_concat, bias) = match attention_type {
            LuongAttentionType::Dot => {
                // Dot attention needs no parameters
                (None, None, None, None)
            }
            LuongAttentionType::General => {
                // General attention needs W_a: [hidden_size, hidden_size]
                let w = Self::init_weight(&[hidden_size, hidden_size], scale)?;
                let b = if use_bias {
                    Some(Tensor::zeros(&[hidden_size]))
                } else {
                    None
                };
                (Some(w), None, None, b)
            }
            LuongAttentionType::Concat => {
                // Concat attention needs W_a and v_a
                let w = Self::init_weight(&[hidden_size * 2, hidden_size], scale)?;
                let v = Self::init_weight(&[hidden_size], scale)?;
                let b = if use_bias {
                    Some(Tensor::zeros(&[hidden_size]))
                } else {
                    None
                };
                (None, Some(w), Some(v), b)
            }
        };

        Ok(Self {
            hidden_size,
            attention_type,
            w_general,
            w_concat,
            v_concat,
            use_bias,
            bias,
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

    /// Compute attention scores and context vector
    ///
    /// # Arguments
    /// * `encoder_outputs` - Encoder hidden states [seq_len, batch_size, hidden_size]
    /// * `decoder_hidden` - Current decoder hidden state [batch_size, hidden_size]
    ///
    /// # Returns
    /// * Tuple of (context_vector, attention_weights)
    /// * context_vector: [batch_size, hidden_size]
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

        // Validate dimensions
        if encoder_dim != self.hidden_size || decoder_dim != self.hidden_size {
            return Err(TensorError::invalid_argument(format!(
                "Expected hidden size {}, got encoder={}, decoder={}",
                self.hidden_size, encoder_dim, decoder_dim
            )));
        }

        // Compute attention scores based on type
        let scores = match self.attention_type {
            LuongAttentionType::Dot => self.compute_dot_scores(encoder_outputs, decoder_hidden)?,
            LuongAttentionType::General => {
                self.compute_general_scores(encoder_outputs, decoder_hidden)?
            }
            LuongAttentionType::Concat => {
                self.compute_concat_scores(encoder_outputs, decoder_hidden)?
            }
        };

        // Apply softmax to get attention weights: [seq_len, batch_size]
        let attention_weights = Self::softmax_over_sequence(&scores)?;

        // Compute context vector as weighted sum of encoder outputs
        let context = Self::compute_context(encoder_outputs, &attention_weights)?;

        // Transpose attention weights to [batch_size, seq_len]
        let attention_weights_transposed = attention_weights.transpose()?;

        Ok((context, attention_weights_transposed))
    }

    /// Compute dot-product attention scores
    fn compute_dot_scores(
        &self,
        encoder_outputs: &Tensor<T>,
        decoder_hidden: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // encoder_outputs: [seq_len, batch_size, hidden_size]
        // decoder_hidden: [batch_size, hidden_size]
        // output: [seq_len, batch_size]

        let encoder_shape = encoder_outputs.shape().dims();
        let seq_len = encoder_shape[0];
        let batch_size = encoder_shape[1];
        let hidden_size = encoder_shape[2];

        let encoder_data = encoder_outputs.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access encoder data".to_string())
        })?;

        let decoder_data = decoder_hidden.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access decoder data".to_string())
        })?;

        let mut scores_data = vec![T::zero(); seq_len * batch_size];

        // Compute dot product for each position and batch
        for s in 0..seq_len {
            for b in 0..batch_size {
                let mut score = T::zero();
                for h in 0..hidden_size {
                    let encoder_idx = s * batch_size * hidden_size + b * hidden_size + h;
                    let decoder_idx = b * hidden_size + h;
                    score = score + encoder_data[encoder_idx] * decoder_data[decoder_idx];
                }
                scores_data[s * batch_size + b] = score;
            }
        }

        Tensor::from_data(scores_data, &[seq_len, batch_size])
    }

    /// Compute general (multiplicative) attention scores
    fn compute_general_scores(
        &self,
        encoder_outputs: &Tensor<T>,
        decoder_hidden: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // score = h_t^T * W_a * h_s
        let w = self.w_general.as_ref().ok_or_else(|| {
            TensorError::device_error_simple("General weight matrix not initialized".to_string())
        })?;

        // Project decoder hidden: [batch_size, hidden_size] @ [hidden_size, hidden_size]
        let mut projected = tenflowers_core::ops::matmul(decoder_hidden, w)?;
        if let Some(ref bias) = self.bias {
            projected = tenflowers_core::ops::add(&projected, bias)?;
        }

        // Compute dot product with projected decoder hidden
        self.compute_dot_scores(encoder_outputs, &projected)
    }

    /// Compute concat (additive) attention scores
    fn compute_concat_scores(
        &self,
        encoder_outputs: &Tensor<T>,
        decoder_hidden: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        // score = v_a^T * tanh(W_a * [h_t; h_s])
        let w = self.w_concat.as_ref().ok_or_else(|| {
            TensorError::device_error_simple("Concat weight matrix not initialized".to_string())
        })?;
        let v = self.v_concat.as_ref().ok_or_else(|| {
            TensorError::device_error_simple("Concat attention vector not initialized".to_string())
        })?;

        let encoder_shape = encoder_outputs.shape().dims();
        let seq_len = encoder_shape[0];
        let batch_size = encoder_shape[1];
        let hidden_size = encoder_shape[2];

        // Broadcast decoder to match sequence length
        let decoder_broadcast = Self::broadcast_decoder(decoder_hidden, seq_len)?;

        // Concatenate encoder and decoder along hidden dimension
        // [seq_len, batch_size, hidden_size * 2]
        let concatenated = Self::concatenate_tensors(encoder_outputs, &decoder_broadcast)?;

        // Project through W_a
        let flat_concat = concatenated.reshape(&[seq_len * batch_size, hidden_size * 2])?;
        let mut projected = tenflowers_core::ops::matmul(&flat_concat, w)?;

        if let Some(ref bias) = self.bias {
            projected = tenflowers_core::ops::add(&projected, bias)?;
        }

        // Apply tanh activation
        let activated = tenflowers_core::ops::activation::tanh(&projected)?;

        // Project through v_a
        let v_expanded = v.unsqueeze(&[1])?;
        let scores_flat = tenflowers_core::ops::matmul(&activated, &v_expanded)?;

        // Reshape to [seq_len, batch_size]
        scores_flat.reshape(&[seq_len, batch_size])
    }

    /// Helper: broadcast decoder across sequence length
    fn broadcast_decoder(decoder_hidden: &Tensor<T>, seq_len: usize) -> Result<Tensor<T>> {
        let shape = decoder_hidden.shape().dims();
        let batch_size = shape[0];
        let hidden_size = shape[1];

        let decoder_data = decoder_hidden.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access decoder data".to_string())
        })?;

        let mut result_data = Vec::with_capacity(seq_len * batch_size * hidden_size);
        for _ in 0..seq_len {
            result_data.extend_from_slice(decoder_data);
        }

        Tensor::from_data(result_data, &[seq_len, batch_size, hidden_size])
    }

    /// Helper: concatenate tensors along last dimension
    fn concatenate_tensors(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        let a_shape = a.shape().dims();
        let b_shape = b.shape().dims();

        if a_shape[..2] != b_shape[..2] {
            return Err(TensorError::invalid_shape_simple(
                "Tensor shapes must match except last dimension".to_string(),
            ));
        }

        let seq_len = a_shape[0];
        let batch_size = a_shape[1];
        let a_hidden = a_shape[2];
        let b_hidden = b_shape[2];

        let a_data = a.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access tensor A data".to_string())
        })?;
        let b_data = b.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access tensor B data".to_string())
        })?;

        let mut result_data = Vec::with_capacity(seq_len * batch_size * (a_hidden + b_hidden));

        for s in 0..seq_len {
            for b in 0..batch_size {
                // Copy from A
                let a_start = s * batch_size * a_hidden + b * a_hidden;
                result_data.extend_from_slice(&a_data[a_start..a_start + a_hidden]);

                // Copy from B
                let b_start = s * batch_size * b_hidden + b * b_hidden;
                result_data.extend_from_slice(&b_data[b_start..b_start + b_hidden]);
            }
        }

        Tensor::from_data(result_data, &[seq_len, batch_size, a_hidden + b_hidden])
    }

    /// Apply softmax over sequence dimension (same as Bahdanau)
    fn softmax_over_sequence(scores: &Tensor<T>) -> Result<Tensor<T>> {
        let shape = scores.shape().dims();
        let seq_len = shape[0];
        let batch_size = shape[1];

        let scores_data = scores.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access scores data".to_string())
        })?;

        let mut result_data = vec![T::zero(); seq_len * batch_size];

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

    /// Compute context vector (same as Bahdanau)
    fn compute_context(
        encoder_outputs: &Tensor<T>,
        attention_weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
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

impl<T> Layer<T> for LuongAttention<T>
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
    /// Standard forward pass (simplified interface)
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];

        // Return zeros as placeholder for Layer trait
        // In practice, use forward_with_weights directly
        Ok(Tensor::zeros(&[batch_size, self.hidden_size]))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        let mut params = Vec::new();

        if let Some(ref w) = self.w_general {
            params.push(w);
        }
        if let Some(ref w) = self.w_concat {
            params.push(w);
        }
        if let Some(ref v) = self.v_concat {
            params.push(v);
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        let mut params = Vec::new();

        if let Some(ref mut w) = self.w_general {
            params.push(w);
        }
        if let Some(ref mut w) = self.w_concat {
            params.push(w);
        }
        if let Some(ref mut v) = self.v_concat {
            params.push(v);
        }
        if let Some(ref mut b) = self.bias {
            params.push(b);
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
