//! Positional Encoding Layers
//!
//! This module contains various positional encoding implementations for sequence models:
//!
//! - **SinusoidalPositionalEncoding**: Fixed sinusoidal positional encodings from "Attention Is All You Need"
//! - **LearnedPositionalEncoding**: Learnable positional embeddings trained with the model
//! - **RotaryPositionalEmbedding**: Rotary Position Embeddings (RoPE) from "RoFormer"
//!
//! All positional encoding layers implement the `Layer` trait and can be used as building blocks
//! in transformer and other sequence-to-sequence architectures.

use crate::layers::Layer;
use scirs2_core::num_traits::{Float, One, Zero};
use scirs2_core::random::rand_prelude::*;
use tenflowers_core::{ops::concat, Result, Tensor, TensorError};

/// Sinusoidal Positional Encoding
///
/// Implements the fixed sinusoidal positional encoding from "Attention Is All You Need".
/// Uses sine and cosine functions of different frequencies to encode position information.
#[derive(Clone)]
pub struct SinusoidalPositionalEncoding<T> {
    /// Maximum sequence length supported
    max_len: usize,
    /// Model dimension (embedding dimension)
    d_model: usize,
    /// Pre-computed positional encodings [max_len, d_model]
    encodings: Tensor<T>,
    /// Dropout rate for positional encodings
    _dropout: f32,
    training: bool,
}

impl<T> SinusoidalPositionalEncoding<T>
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
        + bytemuck::Zeroable,
{
    /// Create a new sinusoidal positional encoding layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (should match embedding dimension)
    /// * `max_len` - Maximum sequence length to support
    /// * `dropout` - Dropout rate applied to positional encodings
    pub fn new(d_model: usize, max_len: Option<usize>, dropout: Option<f32>) -> Result<Self> {
        let max_len = max_len.unwrap_or(5000);
        let dropout = dropout.unwrap_or(0.1);

        if d_model == 0 {
            return Err(TensorError::invalid_argument(
                "d_model must be > 0".to_string(),
            ));
        }

        let encodings = Self::create_sinusoidal_encodings(max_len, d_model)?;

        Ok(Self {
            max_len,
            d_model,
            encodings,
            _dropout: dropout,
            training: true,
        })
    }

    /// Create sinusoidal positional encodings
    ///
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    fn create_sinusoidal_encodings(max_len: usize, d_model: usize) -> Result<Tensor<T>> {
        let mut pe_data = Vec::with_capacity(max_len * d_model);

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = T::from(10000.0_f64.powf(2.0 * i as f64 / d_model as f64)).unwrap();
                let pos_t = T::from(pos as f64).unwrap();

                // sin(pos / 10000^(2i/d_model))
                let sin_val = (pos_t / div_term).sin();
                pe_data.push(sin_val);

                // cos(pos / 10000^(2i/d_model))
                let cos_val = (pos_t / div_term).cos();
                pe_data.push(cos_val);
            }

            // Handle odd d_model dimensions
            if d_model % 2 == 1 {
                let div_term =
                    T::from(10000.0_f64.powf(2.0 * (d_model / 2) as f64 / d_model as f64)).unwrap();
                let pos_t = T::from(pos as f64).unwrap();
                let sin_val = (pos_t / div_term).sin();
                pe_data.push(sin_val);
            }
        }

        Tensor::from_vec(pe_data, &[max_len, d_model])
    }

    /// Apply dropout to positional encodings during training
    fn apply_dropout(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::Float + scirs2_core::num_traits::FromPrimitive,
    {
        let dropout_rate = self._dropout;
        let keep_prob = 1.0 - dropout_rate;

        // Simple dropout implementation
        let mut rng = scirs2_core::random::thread_rng();
        let shape = input.shape().dims();
        let total_elements = shape.iter().product::<usize>();

        let input_data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access input tensor data".to_string())
        })?;

        let mut output_data: Vec<T> = Vec::with_capacity(total_elements);
        let scale = T::from_f32(1.0 / keep_prob).unwrap_or_else(|| T::one());

        for &val in input_data {
            let random_val = rng.random::<f32>();
            if random_val < keep_prob {
                output_data.push(val * scale);
            } else {
                output_data.push(T::zero());
            }
        }

        Tensor::from_vec(output_data, shape)
    }

    /// Apply positional encoding to input embeddings
    ///
    /// # Arguments
    /// * `x` - Input embeddings [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
    /// * `batch_first` - Whether input is batch-first format
    pub fn encode(&self, x: &Tensor<T>, batch_first: Option<bool>) -> Result<Tensor<T>> {
        let batch_first = batch_first.unwrap_or(true);
        let input_shape = x.shape().dims();

        if input_shape.len() != 3 {
            return Err(TensorError::invalid_shape_simple(
                "Input must be 3D".to_string(),
            ));
        }

        let (seq_len, d_model) = if batch_first {
            (input_shape[1], input_shape[2])
        } else {
            (input_shape[0], input_shape[2])
        };

        if d_model != self.d_model {
            return Err(TensorError::invalid_shape_simple(format!(
                "Input d_model {} doesn't match positional encoding d_model {}",
                d_model, self.d_model
            )));
        }

        if seq_len > self.max_len {
            return Err(TensorError::invalid_shape_simple(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        // Extract positional encodings for the required sequence length
        let pe_slice = self.encodings.slice(&[0..seq_len, 0..d_model])?;

        // Add positional encodings to input embeddings
        // Need to broadcast the positional encodings to match input shape
        let result = if batch_first {
            // Input: [batch_size, seq_len, d_model], PE: [seq_len, d_model]
            // Need to broadcast PE to [1, seq_len, d_model] then add
            let pe_broadcasted = pe_slice.unsqueeze(&[0])?;
            x.add(&pe_broadcasted)?
        } else {
            // Input: [seq_len, batch_size, d_model], PE: [seq_len, d_model]
            // Need to broadcast PE to [seq_len, 1, d_model] then add
            let pe_broadcasted = pe_slice.unsqueeze(&[1])?;
            x.add(&pe_broadcasted)?
        };

        // Apply dropout during training
        if self.training && self._dropout > 0.0 {
            self.apply_dropout(&result)
        } else {
            Ok(result)
        }
    }
}

impl<T> Layer<T> for SinusoidalPositionalEncoding<T>
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
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.encode(input, Some(true))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // Sinusoidal encodings are fixed, no learnable parameters
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // Sinusoidal encodings are fixed, no learnable parameters
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Learned Positional Encoding
///
/// Implements learnable positional embeddings that are trained along with the model.
/// Each position has its own embedding vector that's learned during training.
#[derive(Clone)]
pub struct LearnedPositionalEncoding<T> {
    /// Maximum sequence length supported
    max_len: usize,
    /// Model dimension (embedding dimension)
    d_model: usize,
    /// Learnable positional embeddings [max_len, d_model]
    embeddings: Tensor<T>,
    /// Dropout rate for positional encodings
    _dropout: f32,
    training: bool,
}

impl<T> LearnedPositionalEncoding<T>
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
        + bytemuck::Zeroable,
{
    /// Create a new learned positional encoding layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (should match embedding dimension)
    /// * `max_len` - Maximum sequence length to support
    /// * `dropout` - Dropout rate applied to positional encodings
    pub fn new(d_model: usize, max_len: Option<usize>, dropout: Option<f32>) -> Result<Self> {
        let max_len = max_len.unwrap_or(5000);
        let dropout = dropout.unwrap_or(0.1);

        if d_model == 0 {
            return Err(TensorError::invalid_argument(
                "d_model must be > 0".to_string(),
            ));
        }

        // Initialize learnable embeddings (typically with small random values)
        // For now, using zeros - in practice you'd want proper initialization
        let embeddings = Tensor::zeros(&[max_len, d_model]);

        Ok(Self {
            max_len,
            d_model,
            embeddings,
            _dropout: dropout,
            training: true,
        })
    }

    /// Apply dropout to positional encodings during training
    fn apply_dropout(&self, input: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: scirs2_core::num_traits::Float + scirs2_core::num_traits::FromPrimitive,
    {
        let dropout_rate = self._dropout;
        let keep_prob = 1.0 - dropout_rate;

        // Simple dropout implementation
        let mut rng = scirs2_core::random::thread_rng();
        let shape = input.shape().dims();
        let total_elements = shape.iter().product::<usize>();

        let input_data = input.as_slice().ok_or_else(|| {
            TensorError::device_error_simple("Cannot access input tensor data".to_string())
        })?;

        let mut output_data: Vec<T> = Vec::with_capacity(total_elements);
        let scale = T::from_f32(1.0 / keep_prob).unwrap_or_else(|| T::one());

        for &val in input_data {
            let random_val = rng.random::<f32>();
            if random_val < keep_prob {
                output_data.push(val * scale);
            } else {
                output_data.push(T::zero());
            }
        }

        Tensor::from_vec(output_data, shape)
    }

    /// Apply positional encoding to input embeddings
    ///
    /// # Arguments
    /// * `x` - Input embeddings [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
    /// * `batch_first` - Whether input is batch-first format
    pub fn encode(&self, x: &Tensor<T>, batch_first: Option<bool>) -> Result<Tensor<T>> {
        let batch_first = batch_first.unwrap_or(true);
        let input_shape = x.shape().dims();

        if input_shape.len() != 3 {
            return Err(TensorError::invalid_shape_simple(
                "Input must be 3D".to_string(),
            ));
        }

        let (seq_len, d_model) = if batch_first {
            (input_shape[1], input_shape[2])
        } else {
            (input_shape[0], input_shape[2])
        };

        if d_model != self.d_model {
            return Err(TensorError::invalid_shape_simple(format!(
                "Input d_model {} doesn't match positional encoding d_model {}",
                d_model, self.d_model
            )));
        }

        if seq_len > self.max_len {
            return Err(TensorError::invalid_shape_simple(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        // Extract positional embeddings for the required sequence length
        let pe_slice = self.embeddings.slice(&[0..seq_len, 0..d_model])?;

        // Add positional embeddings to input embeddings
        // Need to broadcast the positional embeddings to match input shape
        let result = if batch_first {
            // Input: [batch_size, seq_len, d_model], PE: [seq_len, d_model]
            // Need to broadcast PE to [1, seq_len, d_model] then add
            let pe_broadcasted = pe_slice.unsqueeze(&[0])?;
            x.add(&pe_broadcasted)?
        } else {
            // Input: [seq_len, batch_size, d_model], PE: [seq_len, d_model]
            // Need to broadcast PE to [seq_len, 1, d_model] then add
            let pe_broadcasted = pe_slice.unsqueeze(&[1])?;
            x.add(&pe_broadcasted)?
        };

        // Apply dropout during training
        if self.training && self._dropout > 0.0 {
            self.apply_dropout(&result)
        } else {
            Ok(result)
        }
    }
}

impl<T> Layer<T> for LearnedPositionalEncoding<T>
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
        + bytemuck::Zeroable,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.encode(input, Some(true))
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![&self.embeddings]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![&mut self.embeddings]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}

/// Rotary Position Embeddings (RoPE)
///
/// Implements rotary position embeddings as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding".
/// RoPE encodes position information by rotating the embedding vectors using rotation matrices.
/// It has nice properties like relative position invariance and is used in many modern LLMs.
#[derive(Clone)]
pub struct RotaryPositionalEmbedding<T> {
    /// Model dimension (must be even for proper rotation pairs)
    d_model: usize,
    /// Maximum sequence length supported
    max_len: usize,
    /// Base frequency for the rotary embeddings
    base: f64,
    /// Pre-computed cosine values [max_len, d_model/2]
    cos_cached: Tensor<T>,
    /// Pre-computed sine values [max_len, d_model/2]
    sin_cached: Tensor<T>,
    /// Whether to use linear scaling for longer sequences
    scaling_factor: Option<f64>,
    training: bool,
}

impl<T> RotaryPositionalEmbedding<T>
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
    /// Create a new RoPE layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (must be even)
    /// * `max_len` - Maximum sequence length to support
    /// * `base` - Base frequency (typically 10000.0)
    /// * `scaling_factor` - Optional linear scaling factor for longer sequences
    pub fn new(
        d_model: usize,
        max_len: Option<usize>,
        base: Option<f64>,
        scaling_factor: Option<f64>,
    ) -> Result<Self> {
        let max_len = max_len.unwrap_or(8192);
        let base = base.unwrap_or(10000.0);

        if d_model == 0 {
            return Err(TensorError::invalid_argument(
                "d_model must be > 0".to_string(),
            ));
        }

        if d_model % 2 != 0 {
            return Err(TensorError::invalid_argument(
                "d_model must be even for RoPE".to_string(),
            ));
        }

        let (cos_cached, sin_cached) =
            Self::precompute_freqs_cis(d_model, max_len, base, scaling_factor)?;

        Ok(Self {
            d_model,
            max_len,
            base,
            cos_cached,
            sin_cached,
            scaling_factor,
            training: true,
        })
    }

    /// Precompute cosine and sine values for all positions and frequencies
    fn precompute_freqs_cis(
        d_model: usize,
        max_len: usize,
        base: f64,
        scaling_factor: Option<f64>,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        let half_d = d_model / 2;

        // Create frequency inverse: 1.0 / (base^(2*i/d_model)) for i in 0..d_model/2
        let mut freqs = Vec::with_capacity(half_d);
        for i in 0..half_d {
            let freq = 1.0 / base.powf(2.0 * i as f64 / d_model as f64);
            freqs.push(T::from(freq).unwrap());
        }

        let mut cos_data = Vec::with_capacity(max_len * half_d);
        let mut sin_data = Vec::with_capacity(max_len * half_d);

        for pos in 0..max_len {
            let position = if let Some(scale) = scaling_factor {
                pos as f64 / scale
            } else {
                pos as f64
            };

            for &freq in &freqs {
                let angle = T::from(position).unwrap() * freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }

        let cos_cached = Tensor::from_vec(cos_data, &[max_len, half_d])?;
        let sin_cached = Tensor::from_vec(sin_data, &[max_len, half_d])?;

        Ok((cos_cached, sin_cached))
    }

    /// Apply rotary position embeddings to input tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor [..., seq_len, d_model]
    /// * `position_offset` - Optional offset for positions (for incremental decoding)
    pub fn forward(&self, x: &Tensor<T>, position_offset: Option<usize>) -> Result<Tensor<T>> {
        let input_shape = x.shape().dims();
        let seq_len = input_shape[input_shape.len() - 2];
        let d_model = input_shape[input_shape.len() - 1];

        if d_model != self.d_model {
            return Err(TensorError::invalid_shape_simple(format!(
                "Input d_model {} doesn't match RoPE d_model {}",
                d_model, self.d_model
            )));
        }

        let offset = position_offset.unwrap_or(0);
        if offset + seq_len > self.max_len {
            return Err(TensorError::invalid_shape_simple(format!(
                "Sequence length {} with offset {} exceeds max_len {}",
                seq_len, offset, self.max_len
            )));
        }

        // Extract rotation matrices for the required positions
        let cos_slice = self
            .cos_cached
            .slice(&[offset..(offset + seq_len), 0..(self.d_model / 2)])?;
        let sin_slice = self
            .sin_cached
            .slice(&[offset..(offset + seq_len), 0..(self.d_model / 2)])?;

        // Apply rotary embeddings
        self.apply_rotary_embeddings(x, &cos_slice, &sin_slice)
    }

    /// Apply rotary embeddings using rotation matrices
    fn apply_rotary_embeddings(
        &self,
        x: &Tensor<T>,
        cos: &Tensor<T>,
        sin: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let input_shape = x.shape().dims();
        let half_d = self.d_model / 2;

        // Split the input into two halves for rotation
        // x = [x0, x1, x2, x3, ...] -> [x0, x2, ...] and [x1, x3, ...]
        // Create proper slice ranges for all dimensions
        let mut ranges_even = Vec::new();
        let mut ranges_odd = Vec::new();

        // Add ranges for all dimensions except the last one (which we're slicing)
        for &dim_size in &input_shape[..input_shape.len() - 1] {
            ranges_even.push(0..dim_size);
            ranges_odd.push(0..dim_size);
        }
        // Add the slicing range for the last dimension
        ranges_even.push(0..half_d);
        ranges_odd.push(half_d..self.d_model);

        let x_even = x.slice(&ranges_even)?; // First half
        let x_odd = x.slice(&ranges_odd)?; // Second half

        // Broadcast cos and sin to match input batch dimensions
        let mut cos_broadcasted = cos.clone();
        let mut sin_broadcasted = sin.clone();

        // Add batch dimensions if needed
        for _ in 0..(input_shape.len() - 2) {
            cos_broadcasted = cos_broadcasted.unsqueeze(&[0])?;
            sin_broadcasted = sin_broadcasted.unsqueeze(&[0])?;
        }

        // Apply rotation:
        // rotated_even = x_even * cos - x_odd * sin
        // rotated_odd = x_even * sin + x_odd * cos
        let rotated_even = x_even
            .mul(&cos_broadcasted)?
            .sub(&x_odd.mul(&sin_broadcasted)?)?;
        let rotated_odd = x_even
            .mul(&sin_broadcasted)?
            .add(&x_odd.mul(&cos_broadcasted)?)?;

        // Concatenate the rotated halves back together
        let output = concat(&[&rotated_even, &rotated_odd], input_shape.len() - 1)?;

        Ok(output)
    }

    /// Get maximum supported sequence length
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Update cache for longer sequences if needed
    pub fn extend_cache(&mut self, new_max_len: usize) -> Result<()> {
        if new_max_len <= self.max_len {
            return Ok(());
        }

        let (cos_cached, sin_cached) =
            Self::precompute_freqs_cis(self.d_model, new_max_len, self.base, self.scaling_factor)?;

        self.cos_cached = cos_cached;
        self.sin_cached = sin_cached;
        self.max_len = new_max_len;

        Ok(())
    }
}

impl<T> Layer<T> for RotaryPositionalEmbedding<T>
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
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        self.forward(input, None)
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        // RoPE has no learnable parameters
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        // RoPE has no learnable parameters
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn clone_box(&self) -> Box<dyn Layer<T>> {
        Box::new(self.clone())
    }
}
